import argparse
import os
import csv
import yaml
import comet_ml
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

from dataloader import VimeoDataset, VideoTestData
from VCT.entropy_models import EntropyBottleneck, estimate_bpp
from VCT.networks import __CODER_TYPES__
from VCT.util.psnr import mse2psnr
from VCT.util.ssim import MS_SSIM
from VCT.util.vision import PlotFlow, PlotHeatMap, save_image
from VCT.util.tools import Alignment

#phase = {'trainAE': 100000, # 100k
#         'trainPrior': 150000, # 50k
#         'trainAll': 175000} # 25K
phase = {'trainAE': 30,
         'trainPrior': 45,
         'trainAll': 55}


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


class VCT(CompressesModel):
    def __init__(self, args, codec):
        super(VCT, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss(reduction='none') if not self.args.ssim else MS_SSIM(data_range=1.).cuda()
        self.codec = codec
        self.latent_buffer = list()

    def load_args(self, args):
        self.args = args

    def forward(self, coding_frame, frame_idx=0, enable_LRP=True):
        if frame_idx == 0:
            self.latent_buffer = []
            reconstructed, likelihoods, latent = self.codec(coding_frame, 'hyper', enable_LRP=enable_LRP)

            # For first P-frame, 2 latents are needed so it needs to be duplicated
            self.latent_buffer.append(latent)
            self.latent_buffer.append(latent)
        else:
            reconstructed, likelihoods, latent = self.codec(coding_frame, 'temp', self.latent_buffer, enable_LRP=enable_LRP)

            self.latent_buffer = [self.latent_buffer[1], latent]
            
        return {
                'rec_frame': reconstructed, 
                'likelihoods': likelihoods, 
                'latent': latent
               }
    def disable_modules(self, modules):
        for module in modules:
            module.requires_grad_(False)
            for param in module.parameters(): 
               self.optimizers().state[param] = {} # remove all state (step, exp_avg, exp_avg_sg)

    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch
        batch = batch.cuda()
        
        if epoch < phase['trainAE']:
            coding_frame = batch[:, 0]
            info = self(coding_frame, 0)

            distortion = self.criterion(coding_frame, info['rec_frame'], enable_LRP=False)
            if self.args.ssim:
                distortion = (1 - distortion)/64

            rate = estimate_bpp(info['likelihoods'], input=coding_frame)
            
            loss = self.args.lmda * distortion.mean() + rate.mean()
            logs = {
                    'train/loss': loss.item(),
                    'train/distortion': distortion.mean().item(), 
                    'train/PSNR': mse2psnr(distortion.mean().item()), 
                    'train/rate': rate.mean().item(), 
                   }
        elif epoch < phase['trainPrior']:
            # Disable AE
            self.disable_modules([self.codec.analysis, self.codec.synthesis])

            # Prepare latents of first 2 frames
            with torch.no_grad():
                _ = self(batch[:, 0], 0, enable_LRP=False)
                _ = self(batch[:, 1], 1, enable_LRP=False)
            
            loss = torch.tensor(0., dtype=torch.float, device=batch.device)
            for idx in range(2, 5):
                coding_frame = batch[:, idx]
                info = self(coding_frame, idx, enable_LRP=False)

                rate = estimate_bpp(info['likelihoods'], input=coding_frame)
                
                loss += rate.mean()

            loss /= 3

            logs = {
                    'train/rate': loss.item(),
                   }

        elif epoch < phase['trainAll']:
            self.requires_grad_(True)
            # Prepare latents of first 2 frames
            with torch.no_grad():
                _ = self(batch[:, 0], 0)
                _ = self(batch[:, 1], 1)
            
            loss = torch.tensor(0., dtype=torch.float, device=batch.device)
            total_distortion = torch.tensor(0., dtype=torch.float, device=batch.device)
            total_rate = torch.tensor(0., dtype=torch.float, device=batch.device)
            
            for idx in range(2, 5):
                coding_frame = batch[:, idx]
                info = self(coding_frame, idx)

                distortion = self.criterion(coding_frame, info['rec_frame'])
                if self.args.ssim:
                    distortion = (1 - distortion)/64

                rate = estimate_bpp(info['likelihoods'], input=coding_frame)

                loss += self.args.lmda * distortion.mean() + rate.mean()
                total_distortion += distortion.mean().detach()
                total_rate += rate.mean().detach()

            total_distortion /= 3
            total_rate /= 3
            logs = {
                    'train/loss': loss.item(),
                    'train/distortion': total_distortion.item(), 
                    'train/PSNR': mse2psnr(total_distortion.item()), 
                    'train/rate': total_rate.item(), 
                   }
        else:
            loss = self.aux_loss()
            
            logs = {
                    'train/loss': loss.item(),
                   }

        self.log_dict(logs)

        return loss 

    def validation_step(self, batch, batch_idx):
        def create_grid(img):
            return make_grid(torch.unsqueeze(img, 1)).cpu().detach().numpy()[0]

        def upload_img(tnsr, tnsr_name, ch="first", grid=True):
            if grid:
                tnsr = create_grid(tnsr)

            self.logger.experiment.log_image(tnsr, name=tnsr_name, step=self.current_epoch, image_channels=ch, overwrite=True)

        if self.args.ssim:
            similarity_metrics = 'MS-SSIM'
        else:
            similarity_metrics = 'PSNR'

        dataset_name, seq_name, batch, frame_id_start = batch
        frame_id = int(frame_id_start)

        seq_name, dataset_name = seq_name[0], dataset_name[0]

        gop_size = batch.size(1)

        height, width = batch.size()[3:]

        similarity_list = []
        mse_list = []
        rate_list = []
        loss_list = []
        align = Alignment()

        epoch = int(self.current_epoch)

        for frame_idx in range(gop_size):
            coding_frame = batch[:, frame_idx]
            info = self(align.align(coding_frame), frame_idx, enable_LRP=(self.current_epoch >= phase['trainPrior']))

            rec_frame = align.resume(info['rec_frame']).clamp(0, 1)
            rate = estimate_bpp(info['likelihoods'], input=rec_frame).mean().item()

            mse = self.criterion(rec_frame, coding_frame).mean().item()
            if self.args.ssim:
                similarity = mse
            else:
                similarity = mse2psnr(mse)
            
            if frame_idx < 3:
                upload_img(coding_frame.cpu().numpy()[0], f'{seq_name}_{epoch}_gt_frame_{frame_idx+frame_id}.png', grid=False)
                upload_img(rec_frame.cpu().numpy()[0], seq_name + '_{:d}_rec_frame_{:d}_{:.3f}.png'.format(epoch, frame_idx+frame_id, similarity), grid=False)

            loss = self.args.lmda * mse + rate

            similarity_list.append(similarity)
            rate_list.append(rate)
            mse_list.append(mse)
            loss_list.append(loss)

        similarity = np.mean(similarity_list)
        rate = np.mean(rate_list)
        mse = np.mean(mse_list)
        loss = np.mean(loss_list)

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 
                'val_loss': loss, 'val_mse': mse, 
                'val_similarity': similarity, 'val_rate': rate}

        return {'val_log': logs}

    def validation_epoch_end(self, outputs):
        rd_dict = {}
        loss = []

        for logs in [log['val_log'] for log in outputs]:
            dataset_name = logs['dataset_name']
            seq_name = logs['seq_name']

            if not (dataset_name in rd_dict.keys()):
                rd_dict[dataset_name] = {}
                rd_dict[dataset_name]['similarity'] = []
                rd_dict[dataset_name]['rate'] = []

            rd_dict[dataset_name]['similarity'].append(logs['val_similarity'])
            rd_dict[dataset_name]['rate'].append(logs['val_rate'])
   
            loss.append(logs['val_loss'])

        avg_loss = np.mean(loss)
        
        logs = {'val/loss': avg_loss}

        for dataset_name, rd in rd_dict.items():
            if self.args.ssim:
                logs['val/'+dataset_name+' msssim'] = np.mean(rd['similarity'])
            else:
                logs['val/'+dataset_name+' psnr'] = np.mean(rd['similarity'])
            logs['val/'+dataset_name+' rate'] = np.mean(rd['rate'])

        self.log_dict(logs)
        return None

    def test_step(self, batch, batch_idx):
        if self.args.ssim:
            similarity_metrics = 'MS-SSIM'
        else:
            similarity_metrics = 'PSNR'
        
        metrics_name = [similarity_metrics, 'Rate']
        metrics = {}
        for m in metrics_name:
            metrics[m] = []

        dataset_name, seq_name, batch, frame_id_start = batch

        os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)

        seq_name, dataset_name = seq_name[0], dataset_name[0]

        gop_size = batch.size(1)

        height, width = batch.size()[3:]

        log_list = []

        # To align frame into multiplications of 64 ; zero-padding is performed
        align = Alignment()
        
        for frame_idx in range(gop_size):
            TO_VISUALIZE = False and frame_idx < 8

            coding_frame = batch[:, frame_idx]

            if self.args.verbose and frame_idx > 0:
                def dummy_cstr(f):
                    return{
                        'coding_frame': torch.ones(f).cuda(),
                        'frame_idx': frame_idx
                    }
                macs, params = get_model_complexity_info(self, tuple(align.align(coding_frame).shape), input_constructor=dummy_cstr)
                print(macs)
            info = self(align.align(coding_frame), frame_idx)

            rec_frame = align.resume(info['rec_frame']).clamp(0, 1)
            rate = estimate_bpp(info['likelihoods'], input=rec_frame).mean().item()

            mse = self.criterion(rec_frame, coding_frame).mean().item()
            if self.args.ssim:
                similarity = mse
            else:
                similarity = mse2psnr(mse)

            loss = self.args.lmda * mse + rate

            metrics[similarity_metrics].append(similarity)
            metrics['Rate'].append(rate)

            if TO_VISUALIZE:
                save_image(coding_frame[0], os.path.join(self.args.save_dir, f'{seq_name}/gt_frame/', f'frame_{int(frame_id_start + frame_idx)}.png'), nrow=1)
                save_image(rec_frame[0], os.path.join(self.args.save_dir, f'{seq_name}/rec_frame/', f'frame_{int(frame_id_start + frame_idx)}.png'), nrow=1)

            log_list.append({similarity_metrics: similarity, 'Rate': rate})
            
            
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
        
        optimizer = optim.Adam([dict(params=self.main_parameters(), lr=self.args.lr),
                                dict(params=self.aux_parameters(), lr=self.args.lr * 10)])
        
        def linearLRwithWarmup(current_epoch):
            warmup = 1
            if current_epoch < phase['trainAE']:
                if current_epoch < warmup:
                    return 1.
                return 0.1*(current_epoch - warmup)/(phase['trainAE'] - warmup)
            elif current_epoch < phase['trainPrior']:
                if current_epoch < phase['trainAE'] + warmup:
                    return 1.
                return 0.1*(current_epoch - phase['trainAE'] - warmup)/(phase['trainPrior'] - phase['trainAE'] - warmup)
            elif current_epoch < phase['trainAll']:
                if current_epoch < phase['trainPrior'] + warmup:
                    return 1.
                return 0.4*(current_epoch - phase['trainPrior'] - warmup)/(phase['trainAll'] - phase['trainPrior'] - warmup)
            
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[linearLRwithWarmup, lambda a: 1.])
 
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=None,
                       using_native_amp=None, using_lbfgs=None):

        def clip_gradient(opt, grad_clip):
            for group in opt.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

        #clip_gradient(optimizer, 5)

        optimizer.step()
        optimizer.zero_grad()

    def setup(self, stage):
        self.logger.experiment.log_parameters(self.args)

        if stage == 'fit':
            transformer = transforms.Compose([
                transforms.RandomCrop(self.args.patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            self.train_dataset = VimeoDataset(os.path.join(self.args.dataset_path, "vimeo_septuplet/"), 7, transform=transformer)
            self.val_dataset = VideoTestData(os.path.join(self.args.dataset_path, "video_dataset/"), self.args.lmda, sequence=('B'), GOP=32)
        elif stage == 'test':
            self.test_dataset = VideoTestData(os.path.join(self.args.dataset_path, "video_dataset/"), self.args.lmda,
                                              sequence=('B'), GOP=self.args.test_GOP)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        # REQUIRED
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.args.batch_size//self.args.gpus,
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
    parser = VCT.add_model_specific_args(parser)

    # training specific
    parser.add_argument('--restore', type=str, choices=['none', 'resume', 'load', 'custom'], default='none')
    parser.add_argument('--restore_key', type=str, default=None)
    parser.add_argument('--restore_epoch', type=int, default=49)
    parser.add_argument('--test', "-T", action="store_true")
    parser.add_argument('--test_GOP', type=int, default=32)
    parser.add_argument('--experiment_name', type=str, default='basic')
    parser.add_argument('--project_name', type=str, default="CANFVC")
    parser.add_argument('--codec_conf', type=str, default=None)

    parser.set_defaults(gpus=1)

    # parse params
    args = parser.parse_args()

    experiment_name = args.experiment_name
    project_name = args.project_name

    torch.backends.cudnn.deterministic = True

    # Config codecs
    assert not (args.codec_conf is None)
    codec_cfg = yaml.safe_load(open(args.codec_conf, 'r'))
    codec_arch = __CODER_TYPES__[codec_cfg['model_architecture']]
    codec = codec_arch(**codec_cfg['model_params'])

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
        api_key="bFaTNhLcuqjt1mavz02XPVwN8", # Fill with your own
        project_name=project_name,
        workspace="tl32rodan", # Fill with your own
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
    ########

    if args.restore == 'resume' or args.restore == 'load':
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
        
        model = VCT(args, codec).cuda()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        
        if args.restore == 'resume':
            trainer.current_epoch = epoch_num + 1
        else:
            trainer.current_epoch = epoch_num + 1
   
    elif args.restore == 'custom':
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
        
        from collections import OrderedDict
        new_ckpt = OrderedDict()

        for k, v in checkpoint['state_dict'].items():
            if k.split('.')[1] != 'temporal_prior':
                new_ckpt[k] = v

        model = VCT(args, codec).cuda()
        model.load_state_dict(new_ckpt, strict=False)
        
        trainer.current_epoch = epoch_num + 1
        
    else:
        trainer = Trainer.from_argparse_args(args,
                                             checkpoint_callback=checkpoint_callback,
                                             gpus=args.gpus,
                                             distributed_backend=db,
                                             logger=comet_logger,
                                             default_root_dir=args.log_path,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=0,
                                             terminate_on_nan=True)
    
        model = VCT(args, codec).cuda()

    if args.verbose:
        #summary(model)
        #summary(model.codec.analysis)
        #summary(model.codec.synthesis)
        summary(model.codec.temporal_prior)
        summary(model.codec.temporal_prior.trans_sep)
        summary(model.codec.temporal_prior.trans_joint)
        summary(model.codec.temporal_prior.trans_cur)

    if args.test:
        trainer.test(model)
    else:
        trainer.fit(model)
