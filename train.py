import argparse
import os
import csv
import yaml
import comet_ml
import flowiz as fz
import numpy as np
import torch
import torch.nn.functional as F
import sys

from functools import partial
from torchinfo import summary
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from ptflops import get_model_complexity_info

from dataloader import VimeoDataset, VimeoDatasetBPGIframe, VideoTestData
from CANF_VC.entropy_models import EntropyBottleneck, estimate_bpp
from CANF_VC.networks import __CODER_TYPES__, AugmentedNormalizedFlowHyperPriorCoder
from CANF_VC.flownets import PWCNet, SPyNet
from CANF_VC.SDCNet import MotionExtrapolationNet
from CANF_VC.GridNet import GridNet, ResidualBlock, DownsampleBlock
from CANF_VC.models import Refinement
from CANF_VC.util.psnr import mse2psnr
from CANF_VC.util.sampler import Resampler
from CANF_VC.util.ssim import MS_SSIM
from CANF_VC.util.vision import PlotFlow, PlotHeatMap, save_image
from CANF_VC.util.tools import Alignment

plot_flow = PlotFlow().cuda() 

phase = {'trainMV': 10, 
         'trainMC': 15, 
         'trainRes_2frames': 23, 
         'trainAll_2frames': 28, 
         'trainAll_fullgop': 33, 
         'trainAll_fullgop_1': 38, 
         #'trainAll_RNN_1': 35, 
         #'trainAll_RNN_2': 40,
         'train_aux': 40}


class CompressesModel(LightningModule):
    """Basic Compress Model"""

    def __init__(self):
        super(CompressesModel, self).__init__()

    def named_main_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' not in name:
                yield (name, param)

    def main_parameters(self):
        for _, param in self.named_main_parameters():
            yield param

    def named_aux_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' in name:
                yield (name, param)

    def aux_parameters(self):
        for _, param in self.named_aux_parameters():
            yield param

    def aux_loss(self):
        aux_loss = []
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                aux_loss.append(m.aux_loss())

        return torch.stack(aux_loss).mean()


class Pframe(CompressesModel):
    def __init__(self, args, mo_coder, cond_mo_coder, res_coder):
        super(Pframe, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss(reduction='none') if not self.args.ssim else MS_SSIM(data_range=1.).cuda()

        self.if_model = AugmentedNormalizedFlowHyperPriorCoder(128, 320, 192, num_layers=2, use_QE=True, use_affine=False,
                                                               use_context=True, condition='GaussianMixtureModel', quant_mode='noise')

        if self.args.MENet == 'PWC':
            self.MENet = PWCNet(trainable=False)
        elif self.args.MENet == 'SPy':
            self.MENet = SPyNet(trainable=False)

        self.MWNet = MotionExtrapolationNet(sequence_length=3) # Motion extrapolation network
        self.MWNet.__delattr__('flownet')

        self.Motion = mo_coder
        self.CondMotion = cond_mo_coder

        self.Resampler = Resampler()

        if self.args.MCNet == 'UNet':
            self.MCNet = Refinement(6, 64, out_channels=3)
        elif self.args.MCNet == 'GridNet':
            self.MCNet = nn.ModuleList([ResidualBlock(3, 32), DownsampleBlock(32, 64), DownsampleBlock(64, 96), GridNet([6, 64, 128, 192], [32, 64, 96], 6, 3)])

        self.Residual = res_coder
        self.frame_buffer = list()
        # flow buffer will be handled in SDCNet

    def load_args(self, args):
        self.args = args

    def motion_compensation(self, ref_frame, flow):
        warped_frame = self.Resampler(ref_frame, flow)
        if self.args.MCNet == 'UNet':
            mc_frame = self.MCNet(ref_frame, warped_frame)

        elif self.args.MCNet == 'GridNet':
            feats1 = [warped_frame]
            feats2 = [ref_frame]
            for i, feature_extractor in enumerate(self.MCNet[:-1]):
                feat = feature_extractor(feats2[i])
                feats1.append(self.Resampler(feat, nn.functional.interpolate(flow, scale_factor=2**(-i), mode='bilinear', align_corners=True) * 2**(-i)))
                feats2.append(feat)

            feats = [torch.cat([feat1, feat2], axis=1)  for feat1, feat2 in zip(feats1, feats2)]
            mc_frame, _ = self.MCNet[-1](feats)
            
        return mc_frame, warped_frame

    def motion_forward(self, ref_frame, coding_frame, predict=False):
        if predict:
            assert len(self.frame_buffer) == 3 or len(self.frame_buffer) == 2

            if len(self.frame_buffer) == 3:
                frame_buffer = [self.frame_buffer[0], self.frame_buffer[1], self.frame_buffer[2]]
            else:
                frame_buffer = [self.frame_buffer[0], self.frame_buffer[0], self.frame_buffer[1]]

            pred_frame, pred_flow = self.MWNet(frame_buffer, None, True)

            flow = self.MENet(ref_frame, coding_frame)
            flow_hat, likelihood_m, pred_flow_hat, _ = self.CondMotion(flow, xc=pred_flow, x2_back=pred_flow, temporal_cond=pred_frame)

            self.MWNet.append_flow(flow_hat)
            
            mc_frame, warped_frame = self.motion_compensation(ref_frame, flow_hat)

            likelihoods = likelihood_m
            data = {'likelihood_m': likelihood_m, 
                    'flow': flow, 'flow_hat': flow_hat, 'mc_frame': mc_frame, 'warped_frame': warped_frame, 
                    'pred_frame': pred_frame, 'pred_flow': pred_flow, 'pred_flow_hat': pred_flow_hat}

        else:
            flow = self.MENet(ref_frame, coding_frame)
            flow_hat, likelihood_m = self.Motion(flow)

            self.MWNet.append_flow(flow_hat)
            
            mc_frame, warped_frame = self.motion_compensation(ref_frame, flow_hat)

            likelihoods = likelihood_m
            data = {'likelihood_m': likelihood_m, 
                    'flow': flow, 'flow_hat': flow_hat, 'mc_frame': mc_frame, 'warped_frame': warped_frame}

        return mc_frame, likelihoods, data

    def forward(self, ref_frame, coding_frame, predict=False):
        if not predict: # For the first P-frame, put I-frame into frame_buffer
            self.frame_buffer = [ref_frame]

        mc, likelihood_m, info = self.motion_forward(ref_frame, coding_frame, predict=predict)

        reconstructed, likelihood_r, mc_hat, BDQ = self.Residual(coding_frame, xc=mc, x2_back=mc, temporal_cond=mc)

        likelihoods = likelihood_m + likelihood_r
        
        # Update buffer
        self.frame_buffer.append(reconstructed)
        if len(self.frame_buffer) == 4:
            self.frame_buffer.pop(0)
            assert len(self.frame_buffer) == 3, str(len(self.frame_buffer))
        
        info.update({'rec_frame': reconstructed, 'likelihoods': likelihoods, 'mc_hat': mc_hat, 'BDQ': BDQ})
        return info
    
    def disable_modules(self, modules):
        for module in modules:
            module.requires_grad_(False)
            for param in module.parameters(): 
               self.optimizers().state[param] = {} # remove all state (step, exp_avg, exp_avg_sg)

    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch
        batch = batch.cuda()

        # I-frame
        with torch.no_grad():
            reconstructed, _, _ = self.if_model(batch[:, 0])
        
        # Determine which modules in P-frame codec are disabled
        self.requires_grad_(True)
        if epoch < phase['trainMC']:
            self.disable_modules([self.MENet, self.MWNet])
        elif epoch < phase['trainRes_2frames']:
            self.disable_modules([self.MENet, self.MWNet, self.Motion, self.CondMotion, self.MCNet])
        elif epoch < phase['trainAll_fullgop']:
            self.disable_modules([self.MENet, self.MWNet])
        
        #### Start training with a batch of sequences ####
        loss = torch.tensor(0., dtype=torch.float, device=reconstructed.device)
        dist_list = rate_list = mc_error_list = pred_frame_error_list = []
        self.MWNet.clear_buffer()
        self.frame_buffer = []

        if epoch < phase['trainMC']:
            for frame_idx in range(1, 3):
                ref_frame, coding_frame = batch[:, frame_idx-1], batch[:, frame_idx]

                info = self(ref_frame, coding_frame, predict=(frame_idx != 1))

                if epoch < phase['trainMV']:
                    distortion = self.criterion(coding_frame, info['warped_frame'])
                    if self.args.ssim:
                        distortion = (1 - distortion)/64
                else:
                    distortion = self.criterion(coding_frame, info['mc_frame'])
                    if self.args.ssim:
                        distortion = (1 - distortion)/64

                rate = estimate_bpp(info['likelihood_m'], input=coding_frame)
                
                if frame_idx == 1:
                    loss += self.args.lmda * distortion.mean() + rate.mean()
                else:
                    pred_frame_hat = self.Resampler(ref_frame, info['pred_flow_hat'])
                    pred_frame_error = nn.MSELoss(reduction='none')(info['pred_frame'], pred_frame_hat)
                    if self.args.ssim:
                        pred_frame_error = (1 - pred_frame_error)/64
                    loss += self.args.lmda * distortion.mean() + rate.mean() + 0.01 * self.args.lmda * pred_frame_error.mean()
                    pred_frame_error_list.append(pred_frame_error.mean())

                # Manually update buffer
                self.frame_buffer[-1] = coding_frame

                dist_list.append(distortion.mean())
                rate_list.append(rate.mean())

            loss = loss / frame_idx
            distortion = torch.mean(torch.tensor(dist_list))
            rate = torch.mean(torch.tensor(rate_list))
            pred_frame_error = torch.mean(torch.tensor(pred_frame_error_list))

            logs = {
                    'train/loss': loss.item(),
                    'train/distortion': distortion.item(), 
                    'train/PSNR': mse2psnr(distortion.item()), 
                    'train/rate': rate.item(), 
                    'train/pred_frame_error': pred_frame_error.item(), 
                   }

        elif epoch < phase['train_aux']:
            for frame_idx in range(1, 7):
                ref_frame, coding_frame = reconstructed, batch[:, frame_idx]
                if epoch < phase['trainAll_fullgop']:
                    ref_frame = ref_frame.detach()

                info = self(ref_frame, coding_frame, predict=(frame_idx != 1))

                distortion = self.criterion(coding_frame, info['rec_frame'])
                rate = estimate_bpp(info['likelihoods'], input=coding_frame)

                if self.args.ssim:
                    distortion = (1 - distortion)/64
                    #mc_error = (1 - self.criterion(mc, mc_hat))/64
                    mc_error = nn.MSELoss(reduction='none')(info['mc_frame'], info['mc_hat'])
                else:
                    mc_error = nn.MSELoss(reduction='none')(info['mc_frame'], info['mc_hat'])
                
                loss += self.args.lmda * distortion.mean() + rate.mean() + 0.01 * self.args.lmda * mc_error.mean()

                dist_list.append(distortion.mean())
                rate_list.append(rate.mean())
                mc_error_list.append(mc_error.mean())

                reconstructed = info['rec_frame']

                if epoch < phase['trainAll_2frames'] and frame_idx == 3:
                    break

            loss = loss / frame_idx
            distortion = torch.mean(torch.tensor(dist_list))
            rate = torch.mean(torch.tensor(rate_list))
            mc_error = torch.mean(torch.tensor(mc_error_list))

            logs = {
                    'train/loss': loss.item(),
                    'train/distortion': distortion.item(), 
                    'train/PSNR': mse2psnr(distortion.item()), 
                    'train/rate': rate.item(), 
                    'train/mc_error': mc_error.item(),
                   }
        else:
            loss = self.aux_loss()
            
            logs = {
                    'train/loss': loss.item(),
                   }
            # if epoch <= phase['trainMC']:
            #     auxloss = self.Motion.aux_loss()
            # else:
            #     auxloss = self.aux_loss()
            #
            # logs['train/aux_loss'] = auxloss.item()
            #
            # loss = loss + auxloss

        self.log_dict(logs)

        return loss 

    def validation_step(self, batch, batch_idx):
        def get_psnr(mse):
            if mse > 0:
                psnr = 10 * (torch.log(1 * 1 / mse) / np.log(10))
            else:
                psnr = mse + 100
            return psnr

        def create_grid(img):
            return make_grid(torch.unsqueeze(img, 1)).cpu().detach().numpy()[0]

        def upload_img(tnsr, tnsr_name, ch="first", grid=True):
            if grid:
                tnsr = create_grid(tnsr)

            self.logger.experiment.log_image(tnsr, name=tnsr_name, step=self.current_epoch, image_channels=ch, overwrite=True)

        dataset_name, seq_name, batch, frame_id_start = batch
        frame_id = int(frame_id_start)

        ref_frame, batch, seq_name, dataset_name = batch[:, 0], batch[:, 1:], seq_name[0], dataset_name[0]

        gop_size = batch.size(1)

        height, width = ref_frame.size()[2:]

        psnr_list = []
        mc_psnr_list = []
        mse_list = []
        rate_list = []
        m_rate_list = []
        loss_list = []
        align = Alignment()

        epoch = int(self.current_epoch)

        self.MWNet.clear_buffer()
        self.frame_buffer = None

        for frame_idx in range(gop_size):
            if frame_idx != 0:
                coding_frame = batch[:, frame_idx]

                info = self(align.align(ref_frame), align.align(coding_frame), predict=(frame_idx != 1))
                likelihoods, likelihood_m = info['likelihoods'], info['likelihood_m']
                rec_frame = align.resume(info['rec_frame']).clamp(0, 1)
                mc_frame = align.resume(info['mc_frame']).clamp(0, 1)
                mc_hat = align.resume(info['mc_hat']).clamp(0, 1)
                BDQ = align.resume(info['BDQ']).clamp(0, 1)

                rate = estimate_bpp(likelihoods, input=ref_frame).mean().item()
                m_rate = estimate_bpp(likelihood_m, input=ref_frame).mean().item()

                if frame_idx <= 2:
                    mse = torch.mean((rec_frame - coding_frame).pow(2))
                    mc_mse = torch.mean((mc_frame - coding_frame).pow(2))
                    psnr = get_psnr(mse).cpu().item()
                    mc_psnr = get_psnr(mc_mse).cpu().item()

                    if frame_idx == 2:
                        flow_hat = align.resume(info['pred_flow'])
                        flow_rgb = torch.from_numpy(
                            fz.convert_from_flow(flow_hat[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                        upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_{epoch}_pred_flow.png', grid=False)
                    
                        flow_hat = align.resume(info['pred_flow_hat'])
                        flow_rgb = torch.from_numpy(
                            fz.convert_from_flow(flow_hat[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                        upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_{epoch}_pred_flow_hat.png', grid=False)

                    flow_hat = align.resume(info['flow_hat'])
                    flow_rgb = torch.from_numpy(
                        fz.convert_from_flow(flow_hat[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                    upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_{epoch}_dec_flow_{frame_idx}.png', grid=False)
                    
                    upload_img(ref_frame.cpu().numpy()[0], f'{seq_name}_{epoch}_ref_frame_{frame_idx}.png', grid=False)
                    upload_img(coding_frame.cpu().numpy()[0], f'{seq_name}_{epoch}_gt_frame_{frame_idx}.png', grid=False)
                    upload_img(mc_frame.cpu().numpy()[0], seq_name + '_{:d}_mc_frame_{:d}_{:.3f}.png'.format(epoch, frame_idx, mc_psnr), grid=False)
                    upload_img(rec_frame.cpu().numpy()[0], seq_name + '_{:d}_rec_frame_{:d}_{:.3f}.png'.format(epoch, frame_idx, psnr), grid=False)

                ref_frame = rec_frame

                mse = self.criterion(ref_frame, coding_frame).mean().item()
                psnr = mse2psnr(mse)
                mc_mse = self.criterion(mc_frame, coding_frame).mean().item()
                mc_psnr = mse2psnr(mc_mse)
                loss = self.args.lmda * mse + rate

                mc_psnr_list.append(mc_psnr)
                m_rate_list.append(m_rate)

            else:
                with torch.no_grad():
                    rec_frame, likelihoods, _ = self.if_model(align.align(batch[:, frame_idx]))

                rec_frame = align.resume(rec_frame).clamp(0, 1)
                rate = estimate_bpp(likelihoods, input=rec_frame).mean().item()

                mse = self.criterion(rec_frame, batch[:, frame_idx]).mean().item()
                psnr = mse2psnr(mse)
                loss = self.args.lmda * mse + rate

                ref_frame = rec_frame

            psnr_list.append(psnr)
            rate_list.append(rate)
            mse_list.append(mse)
            loss_list.append(loss)

        psnr = np.mean(psnr_list)
        mc_psnr = np.mean(mc_psnr_list)
        rate = np.mean(rate_list)
        m_rate = np.mean(m_rate_list)
        mse = np.mean(mse_list)
        loss = np.mean(loss_list)

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 
                'val_loss': loss, 'val_mse': mse, 
                'val_psnr': psnr, 'val_rate': rate, 
                'val_mc_psnr': mc_psnr, 'val_m_rate': m_rate}

        return {'val_log': logs}

    def validation_epoch_end(self, outputs):
        rd_dict = {}
        loss = []

        for logs in [log['val_log'] for log in outputs]:
            dataset_name = logs['dataset_name']
            seq_name = logs['seq_name']

            if not (dataset_name in rd_dict.keys()):
                rd_dict[dataset_name] = {}
                rd_dict[dataset_name]['psnr'] = []
                rd_dict[dataset_name]['rate'] = []
                rd_dict[dataset_name]['mc_psnr'] = []
                rd_dict[dataset_name]['m_rate'] = []

            rd_dict[dataset_name]['psnr'].append(logs['val_psnr'])
            rd_dict[dataset_name]['rate'].append(logs['val_rate'])
            rd_dict[dataset_name]['mc_psnr'].append(logs['val_mc_psnr'])
            rd_dict[dataset_name]['m_rate'].append(logs['val_m_rate'])
   
            loss.append(logs['val_loss'])

        avg_loss = np.mean(loss)
        
        logs = {'val/loss': avg_loss}

        for dataset_name, rd in rd_dict.items():
            logs['val/'+dataset_name+' psnr'] = np.mean(rd['psnr'])
            logs['val/'+dataset_name+' rate'] = np.mean(rd['rate'])
            logs['val/'+dataset_name+' mc_psnr'] = np.mean(rd['mc_psnr'])
            logs['val/'+dataset_name+' m_rate'] = np.mean(rd['m_rate'])

        self.log_dict(logs)

        return None

    def test_step(self, batch, batch_idx):
        if self.args.msssim:
            similarity_metrics = 'MS-SSIM'
        else:
            similarity_metrics = 'PSNR'
        
        metrics_name = [similarity_metrics, 'Rate', 'Mo_Rate', 'MC-PSNR', 'MCrec-PSNR', 'MCerr-PSNR', 'BDQ-PSNR', 'QE-PSNR', 'back-PSNR', 'p1-PSNR', 'p1-BDQ-PSNR']
        metrics = {}
        for m in metrics_name:
            metrics[m] = []

        os.makedirs(self.args.save_dir + f'/{seq_name}/flow', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/mc_frame', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/pred_frame', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/BDQ', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/mc_hat', exist_ok=True)


        dataset_name, seq_name, batch, frame_id_start = batch

        ref_frame = batch[:, 0] # BPG-compressed I-frame in position 0
        batch = batch[:, 1:] # coding frames
        seq_name = seq_name[0]
        dataset_name = dataset_name[0]

        gop_size = batch.size(1)

        height, width = ref_frame.size()[2:]

        log_list = []

        # To align frame into multiplications of 64 ; zero-padding is performed
        align = Alignment()
        
        # Clear motion buffer & frame buffer
        self.MWNet.clear_buffer()
        self.frame_buffer = list()

        for frame_idx in range(gop_size):
            ref_frame = ref_frame.clamp(0, 1)
            coding_frame = batch[:, frame_idx]

            # P-frame
            if frame_idx != 0:
                coding_frame = batch[:, frame_idx]

                info = self(align.align(ref_frame), align.align(coding_frame), predict=(frame_idx != 1))

                likelihoods, likelihood_m = info['likelihoods'], info['likelihood_m']
                rec_frame = align.resume(info['rec_frame']).clamp(0, 1)
                mc_frame = align.resume(info['mc_frame']).clamp(0, 1)
                mc_hat = align.resume(info['mc_hat']).clamp(0, 1)
                BDQ = align.resume(info['BDQ']).clamp(0, 1)

                rate = estimate_bpp(likelihoods, input=ref_frame).mean().item()
                m_rate = estimate_bpp(likelihood_m, input=ref_frame).mean().item()

                mse = self.criterion(rec_frame, coding_frame).mean().item()
                if self.args.msssim:
                    similarity = mse
                else:
                    similarity = mse2psnr(mse)

                metrics[similarity_metrics].append(similarity)
                metrics['Rate'].append(rate)

                # likelihoods[0] & [1] are motion latent & hyper likelihood
                m_rate = estimate_bpp(likelihoods[0], input=coding_frame).mean().item() + \
                         estimate_bpp(likelihoods[1], input=coding_frame).mean().item()
                metrics['Mo_Rate'].append(m_rate)

                
                if TO_VISUALIZE:
                    flow_map = plot_flow(info['flow_hat'])
                    save_image(flow_map,os.path.join(self.args.save_dir, f'{seq_name}/flow/', f'frame_{int(frame_id_start + frame_idx)}_flow.png'), nrow=1)

                    if frame_idx > 1:
                        flow_map = plot_flow(info['pred_flow'])
                        save_image(flow_map, os.path.join(self.args.save_dir, f'{seq_name}/flow/', f'frame_{int(frame_id_start + frame_idx)}_flow_pred.png'), nrow=1)

                        flow_map = plot_flow(info['pred_flow_hat'])
                        save_image(flow_map, os.path.join(self.args.save_dir, f'{seq_name}/flow/', f'frame_{int(frame_id_start + frame_idx)}_flow_pred_hat.png'), nrow=1)

                        pred_frame = align.resume(info['pred_frame'])
                        save_image(pred_frame, os.path.join(self.args.save_dir, f'{seq_name}/pred_frame/', f'frame_{int(frame_id_start + frame_idx)}.png'), nrow=1)

                    save_image(coding_frame[0], os.path.join(self.args.save_dir, f'{seq_name}/gt_frame/', f'frame_{int(frame_id_start + frame_idx)}.png'), nrow=1)
                    save_image(mc_frame[0], os.path.join(self.args.save_dir, f'{seq_name}/mc_frame/', f'frame_{int(frame_id_start + frame_idx)}.png'), nrow=1)
                    save_image(warped_frame[0], os.path.join(self.args.save_dir, f'{seq_name}/mc_frame/', f'frame_{int(frame_id_start + frame_idx)}_warped.png'), nrow=1)
                    save_image(rec_frame[0], os.path.join(self.args.save_dir, f'{seq_name}/rec_frame/', f'frame_{int(frame_id_start + frame_idx)}.png'), nrow=1)
                    save_image(BDQ[0], os.path.join(self.args.save_dir, f'{seq_name}/BDQ/', f'frame_{int(frame_id_start + frame_idx)}.png'), nrow=1)
                    save_image(mc_hat[0], os.path.join(self.args.save_dir, f'{seq_name}/mc_hat/', f'frame_{int(frame_id_start + frame_idx)}.png'), nrow=1)

                mc_psnr = mse2psnr(self.criterion(warped_frame, coding_frame).mean().item())
                metrics['MC-PSNR'].append(mc_psnr)

                mc_rec_psnr = mse2psnr(self.criterion(mc_hat, coding_frame).mean().item())
                metrics['MCrec-PSNR'].append(mc_rec_psnr)

                mc_err_psnr = mse2psnr(self.criterion(mc_frame, mc_hat).mean().item())
                metrics['MCerr-PSNR'].append(mc_err_psnr)

                BDQ_psnr = mse2psnr(self.criterion(BDQ, coding_frame).mean().item())
                metrics['BDQ-PSNR'].append(BDQ_psnr)

                QE_psnr = mse2psnr(self.criterion(BDQ, ref_frame).mean().item())
                metrics['QE-PSNR'].append(QE_psnr)

                back_psnr = mse2psnr(self.criterion(mc_frame, BDQ).mean().item())
                metrics['back-PSNR'].append(back_psnr)

                if frame_idx == 1:
                    metrics['p1-PSNR'].append(psnr)
                    metrics['p1-BDQ-PSNR'].append(BDQ_psnr)

                log_list.append({similarity_metrics: similarity, 'Rate': rate, 'MC-PSNR': mc_psnr, 'Mo_Rate': m_rate,
                                 'my': estimate_bpp(likelihoods[0]).item(), 'mz': estimate_bpp(likelihoods[1]).item(),
                                 'ry': estimate_bpp(likelihoods[2]).item(), 'rz': estimate_bpp(likelihoods[3]).item(),
                                 'MCerr-PSNR': mc_err_psnr, 'BDQ-PSNR': BDQ_psnr})
            
            # I-frame
            else:
                rec_frame, likelihoods, _ = self.if_model(align.align(coding_frame))
                rec_frame = align.resume(rec_frame).clamp(0, 1)
                rate = estimate_bpp(likelihoods, input=rec_frame).mean().item()

                if self.args.msssim:
                    similarity = self.criterion(rec_frame, coding_frame).mean().item()
                else:
                    mse = self.criterion(rec_frame, coding_frame).mean().item()
                    similarity = mse2psnr(mse)

                metrics[similarity_metrics].append(similarity)
                metrics['Rate'].append(rate)

                save_image(coding_frame[0], os.path.join(self.args.save_dir, f'{seq_name}/gt_frame/', f'frame_{int(frame_id_start + frame_idx)}.png'), nrow=1)
                save_image(rec_frame[0], os.path.join(self.args.save_dir, f'{seq_name}/rec_frame/', f'frame_{int(frame_id_start + frame_idx)}.png'), nrow=1)

                log_list.append({similarity_metrics: similarity, 'Rate': rate})

            # Make reconstruction as next reference frame
            ref_frame = rec_frame

        for m in metrics_name:
            metrics[m] = np.mean(metrics[m])

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 'metrics': metrics, 'log_list': log_list,}
        return {'test_log': logs}

    def test_epoch_end(self, outputs):

        metrics_name = list(outputs[0]['test_log']['metrics'].keys())  # Get all metrics' names

        rd_dict = {}

        single_seq_logs = {}
        for metrics in metrics_name:
            single_seq_logs[metrics] = {}

        single_seq_logs['LOG'] = {}
        single_seq_logs['GOP'] = {}  # Will not be printed currently
        single_seq_logs['Seq_Names'] = []

        for logs in [log['test_log'] for log in outputs]:
            dataset_name = logs['dataset_name']
            seq_name = logs['seq_name']

            if not (dataset_name in rd_dict.keys()):
                rd_dict[dataset_name] = {}
                
                for metrics in metrics_name:
                    rd_dict[dataset_name][metrics] = []

            for metrics in logs['metrics'].keys():
                rd_dict[dataset_name][metrics].append(logs['metrics'][metrics])

            # Initialize
            if seq_name not in single_seq_logs['Seq_Names']:
                single_seq_logs['Seq_Names'].append(seq_name)
                for metrics in metrics_name:
                    single_seq_logs[metrics][seq_name] = []
                single_seq_logs['LOG'][seq_name] = []
                single_seq_logs['GOP'][seq_name] = []

            # Collect metrics logs
            for metrics in metrics_name:
                single_seq_logs[metrics][seq_name].append(logs['metrics'][metrics])
            single_seq_logs['LOG'][seq_name].extend(logs['log_list'])
            single_seq_logs['GOP'][seq_name] = len(logs['log_list'])

        os.makedirs(self.args.save_dir + f'/report', exist_ok=True)

        for seq_name, log_list in single_seq_logs['LOG'].items():
            with open(self.args.save_dir + f'/report/{seq_name}.csv', 'w', newline='') as report:
                writer = csv.writer(report, delimiter=',')
                columns = ['frame'] + list(log_list[1].keys())
                writer.writerow(columns)

                for idx in range(len(log_list)):
                    writer.writerow([f'frame_{idx + 1}'] + list(log_list[idx].values()))

        # Summary
        logs = {}
        print_log = '{:>16} '.format('Sequence_Name')
        for metrics in metrics_name:
            print_log += '{:>12}'.format(metrics)
        print_log += '\n'

        for seq_name in single_seq_logs['Seq_Names']:
            print_log += '{:>16} '.format(seq_name)

            for metrics in metrics_name:
                print_log += '{:12.4f}'.format(np.mean(single_seq_logs[metrics][seq_name]))

            print_log += '\n'
        print_log += '================================================\n'
        for dataset_name, rd in rd_dict.items():
            print_log += '{:>16} '.format(dataset_name)

            for metrics in metrics_name:
                logs['test/' + dataset_name + ' ' + metrics] = np.mean(rd[metrics])
                print_log += '{:12.4f}'.format(np.mean(rd[metrics]))

            print_log += '\n'

        print(print_log)

        with open(self.args.save_dir + f'/brief_summary.txt', 'w', newline='') as report:
            report.write(print_log)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        current_epoch = self.trainer.current_epoch
        
        # Learning rate degrades when RNN-based training
        lr_step = []
        for k, v in phase.items():
            if ('RNN' in k or 'fullgop' in k) and v > current_epoch: 
                lr_step.append(v-current_epoch)
        lr_gamma = 0.5
        print('lr decay =', lr_gamma, 'lr milestones =', lr_step)

        optimizer = optim.Adam([dict(params=self.main_parameters(), lr=self.args.lr),
                                dict(params=self.aux_parameters(), lr=self.args.lr * 10)])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_step, lr_gamma)

        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=None,
                       using_native_amp=None, using_lbfgs=None):

        def clip_gradient(opt, grad_clip):
            for group in opt.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

        clip_gradient(optimizer, 5)

        optimizer.step()
        optimizer.zero_grad()

    def setup(self, stage):
        self.logger.experiment.log_parameters(self.args)

        qp = {256: 37, 512: 32, 1024: 27, 2048: 22, 4096: 22}[self.args.lmda]

        if stage == 'fit':
            transformer = transforms.Compose([
                transforms.RandomCrop(self.args.patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            self.train_dataset = VimeoDataset(os.path.join(self.args.dataset_path, "vimeo_septuplet/"), 7, transform=transformer)
            self.val_dataset = VideoTestData(os.path.join(self.args.dataset_path, "video_dataset/"), self.args.lmda, sequence=('B'), GOP=32)
        elif stage == 'test':
            self.test_dataset = VideoTestData(os.path.join(self.args.dataset_path, "video_dataset/"), self.args.lmda, sequence=('U', 'B', 'M', 'K'), GOP=self.args.test_GOP)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        # REQUIRED
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.args.batch_size,
                                  num_workers=self.args.num_workers,
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        # OPTIONAL
        val_loader = DataLoader(self.val_dataset,
                                batch_size=1,
                                num_workers=self.args.num_workers,
                                shuffle=False)
        return val_loader

    def test_dataloader(self):
        # OPTIONAL
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=1,
                                 num_workers=self.args.num_workers,
                                 shuffle=False)
        return test_loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the arguments for this LightningModule
        """
        # MODEL specific
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--MENet', type=str, choices=['PWC', 'SPy'], default='PWC')
        parser.add_argument('--MCNet', type=str, choices=['UNet', 'GridNet'], default='UNet')
        parser.add_argument('--learning_rate', '-lr', dest='lr', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--patch_size', default=256, type=int)
        parser.add_argument('--lmda', default=2048, choices=[256, 512, 1024, 2048, 4096], type=int)
        parser.add_argument('--ssim', action="store_true")
        parser.add_argument('--debug', action="store_true")
        parser.add_argument('--verbose', action="store_true")

        # training specific (for this model)
        parser.add_argument('--num_workers', default=16, type=int)
        parser.add_argument('--dataset_path', default='./video_dataset', type=str)
        parser.add_argument('--log_path', default='./logs', type=str)
        parser.add_argument('--save_dir')

        return parser

if __name__ == '__main__':
    # sets seeds for numpy, torch, etc...
    # must do for DDP to work well
    seed_everything(888888)

    parser = argparse.ArgumentParser(add_help=True)

    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = Pframe.add_model_specific_args(parser)

    # training specific
    parser.add_argument('--restore', type=str, choices=['none', 'resume', 'load', 'custom', 'finetune'], default='none')
    parser.add_argument('--restore_key', type=str, default=None)
    parser.add_argument('--restore_epoch', type=int, default=49)
    parser.add_argument('--test', "-T", action="store_true")
    parser.add_argument('--test_GOP', type=int, default=32)
    parser.add_argument('--experiment_name', type=str, default='basic')
    parser.add_argument('--project_name', type=str, default="CANFVC")

    parser.add_argument('--motion_coder_conf', type=str, default=None)
    parser.add_argument('--cond_motion_coder_conf', type=str, default=None)
    parser.add_argument('--residual_coder_conf', type=str, default=None)

    parser.set_defaults(gpus=1)

    # parse params
    args = parser.parse_args()

    experiment_name = args.experiment_name
    project_name = args.project_name

    # I-frame coder ckpt
    if args.ssim:
        ANFIC_code = {4096: '0619_2320', 2048: '0619_2321', 1024: '0619_2321', 512: '0620_1330', 256: '0620_1330'}[args.lmda]
    else:
        ANFIC_code = {2048: '0821_0300', 1024: '0530_1212', 512: '0530_1213', 256: '0530_1215'}[args.lmda]

    torch.backends.cudnn.deterministic = True
  
    # Config codecs
    assert not (args.motion_coder_conf is None)
    mo_coder_cfg = yaml.safe_load(open(args.motion_coder_conf, 'r'))
    mo_coder_arch = __CODER_TYPES__[mo_coder_cfg['model_architecture']]
    mo_coder = mo_coder_arch(**mo_coder_cfg['model_params'])
 
    assert not (args.cond_motion_coder_conf is None)
    cond_mo_coder_cfg = yaml.safe_load(open(args.cond_motion_coder_conf, 'r'))
    cond_mo_coder_arch = __CODER_TYPES__[cond_mo_coder_cfg['model_architecture']]
    cond_mo_coder = cond_mo_coder_arch(**cond_mo_coder_cfg['model_params'])

    assert not (args.residual_coder_conf is None)
    res_coder_cfg = yaml.safe_load(open(args.residual_coder_conf, 'r'))
    res_coder_arch = __CODER_TYPES__[res_coder_cfg['model_architecture']]
    res_coder = res_coder_arch(**res_coder_cfg['model_params'])

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        save_last=True,
        #every_n_epochs=10, # Save at least every 10 epochs
        period=1, # Save at least every 3 epochs
        verbose=True,
        monitor='val/loss',
        mode='min',
        prefix=''
    )


    db = None
    if args.gpus > 1:
        db = 'ddp'

    comet_logger = CometLogger(
        api_key="", # Fill with your own
        project_name=project_name,
        workspace="", # Fill with your own
        experiment_name=experiment_name + "-" + str(args.lmda),
        experiment_key = args.restore_key if args.restore == 'resume' else None,
        disabled=args.test or args.debug
    )
    
    # args.save_dir will be created only when testing
    args.save_dir = os.path.join(args.log_path, project_name, experiment_name + '-' + str(args.lmda))
    
    ######## Restore usage:
    #   *(default) 'none': Train P-frame codec from scratch. 
    #                      Note that I-frame codec should be (and is) pre-loaded.
    #   *'resume': Resume an existing experiment. 
    #              No new experiments will be created in comet.ml. 
    #   *'load': Load an existing experiment and creat a new experiment. 
    #            Usually used when some intermediate phases should be conducted again.
    #   *'custom': Customize a new experiment from one or multiple experiments. 
    #              Usually used when you want to change architecture or load specific modules from several experiments.
    #   *'finetune': Enable all modules in P-frame codec for fine-tuning.
    #                Usually used for the very last step of codec training or fine-tuning for lower bit-rate models.
    ########
    
    if args.restore == 'resume' or args.restore == 'finetune':
        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=args.log_path,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=0,
                                             terminate_on_nan=True)

        epoch_num = args.restore_epoch
        if args.restore_key is None:
            raise ValueError
        else:  # When prev_key is specified in args
            checkpoint = torch.load(os.path.join(args.log_path, project_name, args.restore_key, "checkpoints", f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))
        
        if args.restore == 'resume':
            trainer.current_epoch = epoch_num + 1
        else:
            trainer.current_epoch = phase['trainAll_2frames']
        
        coder_ckpt = torch.load(os.path.join(args.log_path, f"ANFIC/ANFHyperPriorCoder_{ANFIC_code}/model.ckpt"),
                                map_location=(lambda storage, loc: storage))['coder']

        for k, v in coder_ckpt.items():
            key = 'if_model.' + k
            checkpoint['state_dict'][key] = v
        
        model = Pframe(args, mo_coder, cond_mo_coder, res_coder).cuda()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    
    elif args.restore == 'load':
        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=args.log_path,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=-1,
                                             terminate_on_nan=True)

        epoch_num = args.restore_epoch
        if args.restore_key is None:
            raise ValueError
        else:  # When prev_key is specified in args
            checkpoint = torch.load(os.path.join(args.log_path, project_name, args.restore_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))

        trainer.current_epoch = epoch_num + 1

        coder_ckpt = torch.load(os.path.join(args.log_path, f"ANFIC/ANFHyperPriorCoder_{ANFIC_code}/model.ckpt"),
                                map_location=(lambda storage, loc: storage))['coder']

        for k, v in coder_ckpt.items():
            key = 'if_model.' + k
            checkpoint['state_dict'][key] = v

        model = Pframe(args, mo_coder, cond_mo_coder, res_coder).cuda()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        #summary(model.Residual.DQ)
    
    elif args.restore == 'custom':
        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=args.log_path,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=-1,
                                             terminate_on_nan=True)
        
        epoch_num = args.restore_epoch

        checkpoint = torch.load(os.path.join(args.log_path, "ANF-based-resCoder-for-DVC", "cf8be0b8102c4a6eb2015b58f184f757", "checkpoints", "epoch=83.ckpt"),
                                map_location=(lambda storage, loc: storage))
        trainer.current_epoch = phase['trainMV']
   
        gridnet_ckpt = torch.load(os.path.join(args.log_path, "CANFVC_Plus", "gridnet.pth"),
                                map_location=(lambda storage, loc: storage))
        from collections import OrderedDict
        new_ckpt = OrderedDict()

        for k, v in checkpoint['state_dict'].items():
            if k.split('.')[0] != 'MCNet' and k.split('.')[0] != 'MENet':
                new_ckpt[k] = v

        coder_ckpt = torch.load(os.path.join(args.log_path, f"ANFIC/ANFHyperPriorCoder_{ANFIC_code}/model.ckpt"),
                                map_location=(lambda storage, loc: storage))['coder']

        for k, v in coder_ckpt.items():
            key = 'if_model.' + k
            new_ckpt[key] = v

        model = Pframe(args, mo_coder, cond_mo_coder, res_coder).cuda()
        model.load_state_dict(new_ckpt, strict=False)
        
    else:
        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=args.log_path,
                                             check_val_every_n_epoch=3,
                                             num_sanity_val_steps=0,
                                             terminate_on_nan=True)
    
        coder_ckpt = torch.load(os.path.join(args.log_path, f"ANFIC/ANFHyperPriorCoder_{ANFIC_code}/model.ckpt"),
                                map_location=(lambda storage, loc: storage))['coder']

        from collections import OrderedDict
        new_ckpt = OrderedDict()

        for k, v in coder_ckpt.items():
            key = 'if_model.' + k
            new_ckpt[key] = v

        model = Pframe(args, mo_coder, cond_mo_coder, res_coder).cuda()
        model.load_state_dict(new_ckpt, strict=False)

    if args.verbose:
        summary(model)

    if args.test:
        trainer.test(model)
    else:
        trainer.fit(model)
