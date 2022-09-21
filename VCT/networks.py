import torch
from torch import nn

from context_model import ContextModel
from entropy_models import __CONDITIONS__, EntropyBottleneck, SymmetricConditional
from modules import Conv2d, ConvTranspose2d
from transformer.Models import get_subsequent_mask, Encoder, Decoder
from util.tokenizer import feat2token, token2feat
import time

class CompressesModel(nn.Module):
    """Basic Compress Model"""

    def __init__(self):
        super(CompressesModel, self).__init__()
        self.divisor = None

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

    def _cal_base_cdf(self):
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                m._cal_base_cdf()

    def aux_loss(self):
        aux_loss = []
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                aux_loss.append(m.aux_loss())

        return torch.stack(aux_loss).sum() if len(aux_loss) else torch.zeros(1, device=next(self.parameters()).device)


class FactorizedCoder(CompressesModel):
    """FactorizedCoder"""

    def __init__(self, num_priors, quant_mode='noise'):
        super(FactorizedCoder, self).__init__()
        self.analysis = nn.Sequential()
        self.synthesis = nn.Sequential()

        self.entropy_bottleneck = EntropyBottleneck(
            num_priors, quant_mode=quant_mode)

        self.divisor = 16


class HyperPriorCoder(FactorizedCoder):
    """HyperPrior Coder"""

    def __init__(self, num_condition, num_priors, use_mean=False, use_abs=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(HyperPriorCoder, self).__init__(
            num_priors, quant_mode=quant_mode)
        self.use_mean = use_mean
        self.use_abs = not self.use_mean or use_abs
        self.conditional_bottleneck = __CONDITIONS__[condition](
            use_mean=use_mean, quant_mode=quant_mode)
        if use_context:
            self.conditional_bottleneck = ContextModel(
                num_condition, num_condition*2, self.conditional_bottleneck)
        self.hyper_analysis = nn.Sequential()
        self.hyper_synthesis = nn.Sequential()

        self.divisor = 64

    def compress(self, input, return_hat=False):
        features = self.analysis(input)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        side_stream, z_hat = self.entropy_bottleneck.compress(
            hyperpriors, return_sym=True)

        condition = self.hyper_synthesis(z_hat)

        ret = self.conditional_bottleneck.compress(
            features, condition=condition, return_sym=return_hat)

        if return_hat:
            stream, y_hat = ret
            x_hat = self.synthesis(y_hat)
            return x_hat, [stream, side_stream], [features.size(), hyperpriors.size()]
        else:
            stream = ret
            return [stream, side_stream], [features.size(), hyperpriors.size()]

    def decompress(self, strings, shape):
        stream, side_stream = strings
        y_shape, z_shape = shape

        z_hat = self.entropy_bottleneck.decompress(side_stream, z_shape)

        condition = self.hyper_synthesis(z_hat)

        y_hat = self.conditional_bottleneck.decompress(
            stream, y_shape, condition=condition)

        reconstructed = self.synthesis(y_hat)

        return reconstructed

    def forward(self, input):
        features = self.analysis(input)

        hyperpriors = self.hyper_analysis(
            features.abs() if self.use_abs else features)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            features, condition=condition)

        reconstructed = self.synthesis(y_tilde)

        return reconstructed, (y_likelihood, z_likelihood)


class GoogleAnalysisTransform(nn.Sequential):
    def __init__(self, in_channels, num_features, num_filters, kernel_size):
        super(GoogleAnalysisTransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters),
            Conv2d(num_filters, num_features, kernel_size, stride=2)
        )


class GoogleSynthesisTransform(nn.Sequential):
    def __init__(self, out_channels, num_features, num_filters, kernel_size):
        super(GoogleSynthesisTransform, self).__init__(
            ConvTranspose2d(num_features, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True),
            ConvTranspose2d(num_filters, out_channels, kernel_size, stride=2)
        )

class GoogleHyperScaleSynthesisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperScaleSynthesisTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_features,
                            kernel_size=3, stride=1)
        )


class GoogleHyperAnalysisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperAnalysisTransform, self).__init__(
            Conv2d(num_features, num_filters, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_filters, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_hyperpriors, kernel_size=5, stride=2)
        )


class GoogleHyperSynthesisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperSynthesisTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters * 3 // 2,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters * 3 // 2, num_features,
                            kernel_size=3, stride=1)
        )


class GoogleHyperPriorCoder(HyperPriorCoder):
    """GoogleHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors,
                 in_channels=3, out_channels=3, kernel_size=5, 
                 use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(GoogleHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        
        self.analysis = GoogleAnalysisTransform(
            in_channels, num_features, num_filters, kernel_size)

        self.synthesis = GoogleSynthesisTransform(
            out_channels, num_features, num_filters, kernel_size)

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, num_filters, num_hyperpriors)

        if self.use_mean:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, num_filters, num_hyperpriors)
        else:
            self.hyper_synthesis = GoogleHyperScaleSynthesisTransform(
                num_features, num_filters, num_hyperpriors)


class ResidualBlock(nn.Sequential):
    """Builds the residual block"""

    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__(
            nn.LeakyReLU(0.2),
            Conv2d(num_filters, num_filters, 3, stride=1),
            nn.LeakyReLU(0.2),
            Conv2d(num_filters, num_filters, 3, stride=1),
        )

    def forward(self, input):
        return input + super().forward(input)


class VCTAnalysisTransform(nn.Sequential):
    def __init__(self, in_channels, num_features, num_filters, kernel_size):
        super(VCTAnalysisTransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            nn.LeakyReLU(0.2),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            nn.LeakyReLU(0.2),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            nn.LeakyReLU(0.2),
            Conv2d(num_filters, num_features, kernel_size, stride=2)
        )


class VCTSynthesisTransform(nn.Sequential):
    def __init__(self, out_channels, num_features, num_filters, kernel_size):
        super(VCTSynthesisTransform, self).__init__(
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            nn.LeakyReLU(0.2),
            ConvTranspose2d(num_features, num_filters, kernel_size, stride=2),
            nn.LeakyReLU(0.2),
            ResidualBlock(num_filters),
            ResidualBlock(num_filters),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            nn.LeakyReLU(0.2),
            ResidualBlock(num_filters),
            ResidualBlock(num_filters),
            ConvTranspose2d(num_filters, num_filters, kernel_size, stride=2),
            nn.LeakyReLU(0.2),
            ConvTranspose2d(num_filters, out_channels, kernel_size, stride=2)
        )


class TransformerEntropyModel(nn.Module):
    def __init__(self, entropy_model, w_c=4, w_p=8, d_C=192, d_T=192, d_inner=2048, d_src_vocab=None, d_trg_vocab=None):
        super(TransformerEntropyModel, self).__init__()
        assert isinstance(
            entropy_model, SymmetricConditional), type(entropy_model)
        self.entropy_model = entropy_model

        self.d_T = d_T
        self.w_c = w_c
        self.w_p = w_p

        self.trans_sep = Encoder(d_word_vec=d_T, n_layers=6, n_head=16, d_model=d_T, d_inner=d_inner,
                                 use_proj=True, d_src_vocab=d_C if d_src_vocab is None else d_src_vocab,
                                 dropout=0.01, scale_emb=False)

        self.trans_joint = Encoder(d_word_vec=d_T, n_layers=4, n_head=16, d_model=d_T, d_inner=d_inner,
                                   use_proj=False, d_src_vocab=None, # Projection is not needed; it only use (temporal) embedding
                                   dropout=0.01, scale_emb=False)

        self.trans_cur = Decoder(d_word_vec=d_T, n_layers=5, n_head=16, d_model=d_T, d_inner=d_inner,
                                 use_proj=True, d_trg_vocab=d_C if d_trg_vocab is None else d_trg_vocab,
                                 dropout=0.01, scale_emb=False)

        self.start_token = nn.Parameter(torch.Tensor(1, 1, d_C))

        # Mean & scale prediction
        self.trg_word_prj = nn.Linear(d_T, d_C * 2, bias=False)

    def encode(self, src_seqs):
        """Transformer encoder"""
        assert isinstance(src_seqs, list), "`src_seqs` should be a list"
        
        z_sep = []
        for src_seq in src_seqs:
            enc_output = self.trans_sep(src_seq, None)
            z_sep.append(enc_output)
        z_sep = torch.cat(z_sep, dim=1)

        z_joint = self.trans_joint(z_sep, None)

        return z_joint
       
    def decode(self, trg_seq, enc_output, sz_limit=0, return_z_cur=True):
        """Transformer decoder"""
        mask = get_subsequent_mask(trg_seq, sz_limit).to(trg_seq.device)

        z_cur = self.trans_cur(trg_seq, mask, enc_output, None)

        dec_output = self.trg_word_prj(z_cur)

        # Multiply (\sqrt{d_model} ^ -1) to linear projection output
        dec_output *= (self.d_T ** -0.5)

        if return_z_cur:
            return dec_output, z_cur
        else:
            return dec_output

    def forward(self, feature, prev_features=None, sz_limit=0):
        assert not (prev_features is None) and isinstance(prev_features, list) and len(prev_features) > 0, ValueError

        prev_tokens = [feat2token(feat,
                                  block_size=(self.w_p, self.w_p),
                                  stride=(self.w_c, self.w_c),
                                  padding=[(self.w_p-self.w_c)//2]*2) 
                       for feat in prev_features]

        z_joint = self.encode(prev_tokens)
        
        b, c, h, w = prev_features[0].size()
        seq_len = self.w_c * self.w_c
        num_blocks = (h//self.w_c)*(w//self.w_c)

        if self.training:
            # Get quantized feature first
            output = self.entropy_model.quantize(feature, self.entropy_model.quant_mode)
            
            # Tokenize quantized feature
            cur_token = feat2token(output, block_size=(self.w_c, self.w_c)) # cur_token.shape=(b*num_blocks, seq_len, self.d_C)

            # Add start token to input tokens
            start_token = torch.cat([self.start_token]*(b*num_blocks), dim=0)
            trg_seq = torch.cat([start_token, cur_token], dim=1)[:, :-1, :] # Ignore the last cur_token
            
            # Generate prediction (condition ; mean & scale) all at once
            condition, z_cur = self.decode(trg_seq, z_joint, sz_limit, True)
            
            # Turn condition into 4D feature map
            condition = token2feat(condition, block_size=(self.w_c, self.w_c), feat_size=(b, c*2, h, w))
            
            # Entropy estimation
            output, likelihood = self.entropy_model(feature, condition=condition)
            
            z_cur = token2feat(z_cur, block_size=(self.w_c, self.w_c), feat_size=((b, self.d_T, h, w)))

        else:
            # Prepare empty condition. The generation condition needs to be sequential
            condition = torch.zeros((b, c*2, h, w)).to(prev_features[0].device)
            
            # First input token = start token
            trg_seq = start_token = torch.cat([self.start_token]*(b*num_blocks), dim=0)
            
            # Sequentially generate condition
            for i in range(self.w_c):
                for j in range(self.w_c):
                    idx = i*self.w_c + j

                    if idx == seq_len - 1:
                        _condition, z_cur = self.decode(trg_seq, z_joint, sz_limit, True) # The last z_cur will be the right z_cur
                        _condition = _condition[:, idx: idx+1, :] # mean & scale of p(t_idx)
                    else:
                        _condition = self.decode(trg_seq, z_joint, sz_limit, False)[:, idx: idx+1, :] # mean & scale of p(t_idx)
                        
                    # Turn condition into 4D feature map
                    _condition = token2feat(_condition, block_size=(1, 1), feat_size=(b, c*2, h//self.w_c, w//self.w_c))
                    
                    # Entropy estimation ; _output is correctly decoded. Note: _output = (output - mean).round() + mean != feature.round() 
                    _output, _likelihood = self.entropy_model(feature[:, :, i::self.w_c, j::self.w_c], _condition) # Quantized t_idx

                    condition[:, :, i::self.w_c, j::self.w_c] = _condition
                    
                    _cur_token = feat2token(_output, block_size=(1, 1))
                    
                    # Decoder input for next stage
                    trg_seq = torch.cat([trg_seq, _cur_token], dim=1)

            output, likelihood = self.entropy_model(feature, condition=condition)
            z_cur = token2feat(z_cur, block_size=(self.w_c, self.w_c), feat_size=((b, self.d_T, h, w)))

        return output, likelihood, (condition, z_cur)


class TransformerPriorCoder(CompressesModel):
    """Transformer-based Entropy Coder that takes 2 previously decoded latents as temporal condition"""

    def __init__(self, num_filters, num_features, num_hyperpriors,
                 in_channels=3, out_channels=3, kernel_size=5, 
                 w_c=4, w_p=8, d_C=192, d_T=192, d_inner=2048,
                 d_src_vocab=None, d_trg_vocab=None,
                 condition='Gaussian', quant_mode='noise'):

        super(TransformerPriorCoder, self).__init__()
        
        self.d_C = d_C
        self.d_T = d_T
        self.quant_mode = quant_mode

        self.entropy_bottleneck = EntropyBottleneck(num_hyperpriors, quant_mode=quant_mode)
        self.conditional_bottleneck = __CONDITIONS__[condition](use_mean=True, quant_mode=quant_mode)

        self.analysis = VCTAnalysisTransform(in_channels, num_features, num_filters, kernel_size)
        self.synthesis = VCTSynthesisTransform(out_channels, num_features, num_filters, kernel_size)

        self.temporal_prior = TransformerEntropyModel(self.conditional_bottleneck, w_c, w_p, d_C, d_T, d_inner, d_src_vocab, d_trg_vocab)

        self.hyper_analysis = GoogleHyperAnalysisTransform(num_features, num_filters, num_hyperpriors)
        self.hyper_synthesis = GoogleHyperSynthesisTransform(num_features*self.conditional_bottleneck.condition_size, num_filters, num_hyperpriors)

        # Latnet Residual Pretictor
        self.LRP = nn.Sequential(
            Conv2d(d_T, num_features, 1, stride=1),
            nn.LeakyReLU(0.1),
            ResidualBlock(num_features),
        )

        self.divisor = 64

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    def compress(self, input, return_hat=False):
        pass

    def decompress(self, strings, shape):
        pass
    
    def forward(self, input, use_prior='temp', prev_features=None, enable_LRP=True, encode_only=False):
        assert use_prior in ['temp', 'hyper'] # Use temporal prior or hyperprior
        if use_prior == 'temp':
            assert not (prev_features is None) and isinstance(prev_features, list), ValueError
        
        features = self.analysis(input)
        
        if use_prior == 'temp':
            y_tilde, y_likelihood, condition = self.temporal_prior(features, prev_features)
            condition, z_cur = condition

        else:
            hyperpriors = self.hyper_analysis(features)

            h_tilde, h_likelihood = self.entropy_bottleneck(hyperpriors)
            
            condition = self.hyper_synthesis(h_tilde)

            y_tilde, y_likelihood = self.conditional_bottleneck(features, condition=condition)

        if enable_LRP and use_prior == 'temp':
            y_tilde += self.LRP(z_cur)
        
        if not encode_only:
            reconstructed = self.synthesis(y_tilde)
        else:
            reconstructed = None

        if use_prior == 'temp':
            return reconstructed, (y_likelihood, ), y_tilde, (condition, z_cur, features)
        else:
            return reconstructed, (y_likelihood, h_likelihood), y_tilde, (condition, features)


class TransformerEntropyModelwithTargetPrediction(TransformerEntropyModel):
    def __init__(self, **kwargs):
        super(TransformerEntropyModelwithTargetPrediction, self).__init__(**kwargs)


    def decode(self, trg_seq, enc_output, sz_limit=0, return_z_cur=True):
        """Transformer decoder"""
        z_cur = self.trans_cur(trg_seq, None, enc_output, None)

        dec_output = self.trg_word_prj(z_cur)

        # Multiply (\sqrt{d_model} ^ -1) to linear projection output
        dec_output *= (self.d_T ** -0.5)

        if return_z_cur:
            return dec_output, z_cur
        else:
            return dec_output

    def forward(self, feature, prev_features=None, prediction=None, sz_limit=0):
        assert not (prev_features is None) and isinstance(prev_features, list) and len(prev_features) > 0, ValueError

        prev_tokens = [feat2token(feat,
                                  block_size=(self.w_p, self.w_p),
                                  stride=(self.w_c, self.w_c),
                                  padding=[(self.w_p-self.w_c)//2]*4) 
                       for feat in prev_features]

        prediction_token = feat2token(prediction,
                                   block_size=(self.w_p, self.w_p),
                                   stride=(self.w_c, self.w_c),
                                   padding=[(self.w_p-self.w_c)//2]*4)
        # Encode
        z_joint = self.encode(prev_tokens)

        b, c, h, w = prev_features[0].size()
        seq_len = self.w_c * self.w_c
        num_blocks = (h//self.w_c)*(w//self.w_c)

        # Decode
        if self.training:
            # Get quantized feature first
            output = self.entropy_model.quantize(feature, self.entropy_model.quant_mode)
            
            # Tokenize quantized feature
            cur_token = feat2token(output, block_size=(self.w_c, self.w_c)) # cur_token.shape=(b*num_blocks, seq_len, self.d_C)

            # Add start token to input tokens
            trg_seq = torch.cat([prediction_token]*2, dim=1)
            for idx in range(seq_len):
                # Generate prediction (condition ; mean & scale) all at once
                condition, z_cur = self.decode(trg_seq, z_joint, sz_limit, True)
            
            # Turn condition into 4D feature map
            condition = token2feat(condition, block_size=(self.w_c, self.w_c), feat_size=(b, c*2, h, w))
            
            # Entropy estimation
            output, likelihood = self.entropy_model(feature, condition=condition)
            
            z_cur = token2feat(z_cur, block_size=(self.w_c, self.w_c), feat_size=((b, self.d_T, h, w)))

        else:
            # Prepare empty condition. The generation condition needs to be sequential
            condition = torch.zeros((b, c*2, h, w)).to(prev_features[0].device)
            z_cur = torch.zeros((b, self.d_T, h, w)).to(prev_features[0].device)
            
            # First input token = start token
            trg_seq = start_token = torch.cat([self.start_token]*(b*num_blocks), dim=0)
            
            # Sequentially generate condition
            for i in range(self.w_c):
                for j in range(self.w_c):
                    idx = i*self.w_c + j

                    if idx == seq_len - 1:
                        _condition, z_cur = self.decode(trg_seq, z_joint, sz_limit, True) # The last z_cur will be the right z_cur
                        _condition = _condition[:, idx: idx+1, :] # mean & scale of p(t_idx)
                    else:
                        _condition = self.decode(trg_seq, z_joint, sz_limit, False)[:, idx: idx+1, :] # mean & scale of p(t_idx)
                        
                    # Turn condition into 4D feature map
                    _condition = token2feat(_condition, block_size=(1, 1), feat_size=(b, c*2, h//self.w_c, w//self.w_c))
                    
                    # Entropy estimation ; _output is correctly decoded. Note: _output = (output - mean).round() + mean != feature.round() 
                    _output, _likelihood = self.entropy_model(feature[:, :, i::self.w_c, j::self.w_c], _condition) # Quantized t_idx

                    condition[:, :, i::self.w_c, j::self.w_c] = _condition
                    
                    _cur_token = feat2token(_output, block_size=(1, 1))
                    
                    # Decoder input for next stage
                    trg_seq = torch.cat([trg_seq, _cur_token], dim=1)

            output, likelihood = self.entropy_model(feature, condition=condition)
            z_cur = token2feat(z_cur, block_size=(self.w_c, self.w_c), feat_size=((b, self.d_T, h, w)))

        return output, likelihood, (condition, z_cur)


    def forward(self, src_seq, trg_seq, trg_pred, sz_limit=0):
        z_joint = self.encode(src_seq)
        
        inputs = []
        updated = trg_pred
        inputs.append(torch.cat([updated, trg_pred], dim=2))
        for i in range(1, trg_seq.size(1)):
            updated = updated.clone()
            updated[:, i, :] = trg_seq[:, i, :]
            inputs.append(torch.cat([updated, trg_pred], dim=2))

        inputs = torch.cat(inputs, dim=0)

        z_cur = self.decode(inputs, z_joint, sz_limit)
        
        return z_cur


class TransformerPriorCoderSideInfo(TransformerPriorCoder):
    """
        Transformer-based Entropy Coder that takes 2 previously decoded latents as temporal condition
        Side information (hyperprior/low-resolution) latent is used in Transformer's decoder
        They will be concatenated with target tokens
    """

    def __init__(self, **kwargs):
        super(TransformerPriorCoderSideInfo, self).__init__(**kwargs)
        
        self.temporal_prior = TransformerEntropyModelwithTargetPrediction(kwargs['w_c'], 
                                                                          kwargs['w_p'],
                                                                          kwargs['d_C'],
                                                                          kwargs['d_T'], 
                                                                          kwargs['d_inner'], 
                                                                          kwargs['d_src_vocab'], 
                                                                          kwargs['d_trg_vocab'])
        self.SR_h = GoogleHyperScaleSynthesisTransform(kwargs['num_features'], kwargs['num_filters'], kwargs['num_hyperpriors'])

    def forward(self, input, use_prior='temp', prev_features=None, enable_LRP=True):
        assert use_prior in ['temp', 'hyper'] # Use temporal prior or hyperprior
        if use_prior == 'temp':
            assert not (prev_features is None) and isinstance(prev_features, list), ValueError

        features = self.analysis(input)
        
        if use_prior == 'temp':
            # Train
            if self.training:
                y_tilde = quantize(features, mode=self.quant_mode, mean=None)
            # Test
            else:
                y_tilde = features.round()

            cur_token = feat2token(y_tilde, block_size=(4, 4))
            prev_tokens = [feat2token(feat, block_size=(8, 8), stride=(4, 4), padding=[2]*4) for feat in prev_features]

            # Perform hyperprior coding when use_prior=='temp' as well
            # Extracts blocks from super-resoluted h_tilde and treat it as part of current tokens
            hyperpriors = self.hyper_analysis(features)
            h_tilde, h_likelihood = self.entropy_bottleneck(hyperpriors)
            
            sr_h = self.SR_h(h_tilde)
            h_token = feat2token(sr_h, block_size=(4, 4))

            z_cur = self.temporal_prior(prev_tokens, cur_token, h_token)

            bs, num_tokens, _ = cur_token.size()
            output = torch.zeros_like(z_cur[:bs, :, :])
            for i in range(num_tokens):
                output[: ,i: i+1, :] = z_cur[i*bs: (i+1)*bs, i: i+1, :]

            condition = self.trg_word_prj(output)

            # Multiply (\sqrt{d_model} ^ -1) to linear projection output
            condition *= self.d_T ** -0.5

            mean, scale = condition.chunk(2, dim=2)

            condition = torch.cat([token2feat(mean, block_size=(4, 4), feat_size=prev_features[0].size()),
                                   token2feat(scale, block_size=(4, 4), feat_size=prev_features[0].size())], dim=1)

            # Train
            if self.training:
                _, y_likelihood = self.conditional_bottleneck(features, condition=condition)
            # Test
            else:
                y_tilde, y_likelihood = self.conditional_bottleneck(features, condition=condition)

        else:
            hyperpriors = self.hyper_analysis(features)

            h_tilde, h_likelihood = self.entropy_bottleneck(hyperpriors)

            condition = self.hyper_synthesis(h_tilde)

            y_tilde, y_likelihood = self.conditional_bottleneck(features, condition=condition)

        if enable_LRP and use_prior == 'temp':
            b, c, h, w = features.size()
            z_cur = token2feat(z_cur, block_size=(4, 4), feat_size=((b, self.d_T, h, w)))
            y_tilde += self.LRP(z_cur)

        reconstructed = self.synthesis(y_tilde)

        return reconstructed, (y_likelihood, h_likelihood), y_tilde


__CODER_TYPES__ = {
                   "GoogleHyperPriorCoder": GoogleHyperPriorCoder,
                   "TransformerPriorCoder": TransformerPriorCoder,
                   "TransformerPriorCoderSideInfo": TransformerPriorCoderSideInfo,
                  }
