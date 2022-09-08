import torch
from torch import nn

from context_model import ContextModel
from entropy_models import __CONDITIONS__, EntropyBottleneck
from modules import Conv2d, ConvTranspose2d
from transformer.Models import get_subsequent_mask, Encoder, Decoder
from util.tokenizer import feat2token, token2feat


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
                            kernel_size=5, stride=2, parameterizer=None),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size=5, stride=2, parameterizer=None),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_features,
                            kernel_size=3, stride=1, parameterizer=None)
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
    def __init__(self, w_c=4, w_p=8, d_C=192, d_T=192, d_inner=2048, d_src_vocab=None, d_trg_vocab=None):
        super(TransformerEntropyModel, self).__init__()

        self.d_T = d_T
        self.trans_sep = Encoder(d_word_vec=d_T, n_layers=6, n_head=16, d_model=d_T, d_inner=d_inner,
                                 use_proj=True, d_src_vocab=d_C if d_src_vocab is None else d_src_vocab,
                                 dropout=0.1, scale_emb=False)

        self.trans_joint = Encoder(d_word_vec=d_T, n_layers=4, n_head=16, d_model=d_T, d_inner=d_inner,
                                   use_proj=False, d_src_vocab=None, # Projection is not needed; it only use (temporal) embedding
                                   dropout=0.1, scale_emb=False)

        self.trans_cur = Decoder(d_word_vec=d_T, n_layers=5, n_head=16, d_model=d_T, d_inner=d_inner,
                                 use_proj=True, d_trg_vocab=d_C if d_trg_vocab is None else d_trg_vocab,
                                 dropout=0.1, scale_emb=False)

        self.start_token = nn.Parameter(torch.Tensor(1, 1, d_C))
        

    def forward(self, src_seqs, trg_seq, sz_limit=0, trg_pred=None):
        assert isinstance(src_seqs, list), "`src_seqs` should be a list"
        
        enc_outputs = []
        for src_seq in src_seqs:
            enc_output = self.trans_sep(src_seq, None)
            enc_outputs.append(enc_output)

        enc_outputs = torch.cat(enc_outputs, dim=1)

        z_joint = self.trans_joint(enc_outputs, None)
        
        # Add start token
        start_token = torch.cat([self.start_token]*trg_seq.size(0), dim=0)
        trg_seq = torch.cat([start_token, trg_seq], dim=1)
        if not (trg_pred is None):
            trg_pred = torch.cat([trg_pred, start_token], dim=1)
            trg_seq = torch.cat([trg_seq, trg_pred], dim=2)

        mask = get_subsequent_mask(trg_seq, sz_limit).to(trg_seq.device)

        z_cur = self.trans_cur(trg_seq, mask, z_joint, None)
        
        return z_cur


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

        self.entropy_bottleneck = EntropyBottleneck(num_hyperpriors, quant_mode=quant_mode)
        self.conditional_bottleneck = __CONDITIONS__[condition](use_mean=True, quant_mode=quant_mode)

        self.analysis = VCTAnalysisTransform(in_channels, num_features, num_filters, kernel_size)
        self.synthesis = VCTSynthesisTransform(out_channels, num_features, num_filters, kernel_size)

        self.temporal_prior = TransformerEntropyModel(w_c, w_p, d_C, d_T, d_inner, d_src_vocab, d_trg_vocab)

        self.hyper_analysis = GoogleHyperAnalysisTransform(num_features, num_filters, num_hyperpriors)
        self.hyper_synthesis = GoogleHyperSynthesisTransform(num_features*self.conditional_bottleneck.condition_size, num_filters, num_hyperpriors)

        # Mean & scale prediction
        self.trg_word_prj = nn.Linear(d_T, d_C * 2, bias=False)

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
    
    def forward(self, input, use_prior='temp', prev_features=None):
        assert use_prior in ['temp', 'hyper'] # Use temporal prior or hyperprior
        if use_prior == 'temp':
            assert not (prev_features is None) and isinstance(prev_features, list), ValueError

        features = self.analysis(input)
        
        if use_prior == 'temp':
            cur_token = feat2token(features, block_size=(4, 4))
            prev_tokens = [feat2token(feat, block_size=(8, 8), stride=(4, 4), padding=[2]*4) for feat in prev_features]

            z_cur = self.temporal_prior(prev_tokens, cur_token)[:, :-1, :]

            condition = self.trg_word_prj(z_cur)

            # Multiply (\sqrt{d_model} ^ -1) to linear projection output
            condition *= self.d_T ** -0.5

            mean, scale = condition.chunk(2, dim=2)

            condition = torch.cat([token2feat(mean, block_size=(4, 4), feat_size=prev_features[0].size()),
                                   token2feat(scale, block_size=(4, 4), feat_size=prev_features[0].size())], dim=1)
        else:
            hyperpriors = self.hyper_analysis(features)

            z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)

            condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(features, condition=condition)

        if use_prior == 'temp':
            b, c, h, w = prev_features[0].size()
            z_cur = token2feat(z_cur, block_size=(4, 4), feat_size=((b, self.d_T, h, w)))
            y_tilde += self.LRP(z_cur)

        reconstructed = self.synthesis(y_tilde)

        if use_prior == 'temp':
            return reconstructed, (y_likelihood, ), y_tilde
        else:
            return reconstructed, (y_likelihood, z_likelihood), y_tilde


class TransformerPriorCoderSideInfoAtEncode(TransformerPriorCoder):
    """
        Transformer-based Entropy Coder that takes 2 previously decoded latents as temporal condition
        Side information (hyperprior/low-resolution) latent is used in Transformer's encoder:
            * sr_z = self.trains_sep(self.hyper_synthesis)
    """

    def __init__(self, **kwargs):
        super(TransformerPriorCoderSideInfoAtEncode, self).__init__(**kwargs)

    def forward(self, input, use_prior='temp', prev_features=None):
        assert use_prior in ['temp', 'hyper'] # Use temporal prior or hyperprior
        if use_prior == 'temp':
            assert not (prev_features is None) and isinstance(prev_features, list), ValueError

        features = self.analysis(input)
        
        if use_prior == 'temp':
            cur_token = feat2token(features, block_size=(4, 4))
            prev_tokens = [feat2token(feat, block_size=(8, 8), stride=(4, 4), padding=[2]*4) for feat in prev_features]

            ### --- Difference --- ###
            # Perform hyperprior coding when use_prior=='temp' as well
            # Directly extracts blocks from z_tilde and treat it as prev_features
            hyperpriors = self.hyper_analysis(features)
            z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)

            z_token = [feat2token(z_tilde, block_size=(4, 4), stride=(1, 1), padding=[2]*4)]
            prev_features.append(z_token)
            ### --- End Difference --- ###

            z_cur = self.temporal_prior(prev_tokens, cur_token)[:, :-1, :]

            condition = self.trg_word_prj(z_cur)

            # Multiply (\sqrt{d_model} ^ -1) to linear projection output
            condition *= self.d_T ** -0.5

            #condition = self.temporal_prior(prev_tokens, cur_token)[:, :-1, :]

            mean, scale = condition.chunk(2, dim=2)

            condition = torch.cat([token2feat(mean, block_size=(4, 4), feat_size=prev_features[0].size()),
                                   token2feat(scale, block_size=(4, 4), feat_size=prev_features[0].size())], dim=1)
        else:
            hyperpriors = self.hyper_analysis(features)

            z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)

            condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(features, condition=condition)

        if use_prior == 'temp':
            b, c, h, w = prev_features[0].size()
            z_cur = token2feat(z_cur, block_size=(4, 4), feat_size=((b, self.d_T, h, w)))
            y_tilde += self.LRP(z_cur)

        reconstructed = self.synthesis(y_tilde)

        ### --- Difference --- ###
        return reconstructed, (y_likelihood, z_likelihood), y_tilde
        ### --- End Difference --- ###


class TransformerPriorCoderSideInfoAtDecode(TransformerPriorCoder):
    """
        Transformer-based Entropy Coder that takes 2 previously decoded latents as temporal condition
        Side information (hyperprior/low-resolution) latent is used in Transformer's decoder
    """

    def __init__(self, **kwargs):
        super(TransformerPriorCoderSideInfoAtDecode, self).__init__(**kwargs)

    def forward(self, input, use_prior='temp', prev_features=None):
        assert use_prior in ['temp', 'hyper'] # Use temporal prior or hyperprior
        if use_prior == 'temp':
            assert not (prev_features is None) and isinstance(prev_features, list), ValueError

        features = self.analysis(input)
        
        if use_prior == 'temp':
            cur_token = feat2token(features, block_size=(4, 4))
            prev_tokens = [feat2token(feat, block_size=(8, 8), stride=(4, 4), padding=[2]*4) for feat in prev_features]

            ### --- Difference --- ###
            # Perform hyperprior coding when use_prior=='temp' as well
            # Directly extracts blocks from z_tilde and treat it as part of current tokens
            hyperpriors = self.hyper_analysis(features)
            z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)

            z_token = feat2token(z_tilde, block_size=(4, 4), stride=(1, 1), padding=[2]*4)
            cur_token = torch.cat([cur_token, z_token], dim=1)
            ### --- End Difference --- ###

            z_cur = self.temporal_prior(prev_tokens, cur_token, sz_limit=z_token.size(1))[:, :-(1+z_token.size(1)), :]

            condition = self.trg_word_prj(z_cur)

            # Multiply (\sqrt{d_model} ^ -1) to linear projection output
            condition *= self.d_T ** -0.5

            mean, scale = condition.chunk(2, dim=2)

            condition = torch.cat([token2feat(mean, block_size=(4, 4), feat_size=prev_features[0].size()),
                                   token2feat(scale, block_size=(4, 4), feat_size=prev_features[0].size())], dim=1)
        else:
            hyperpriors = self.hyper_analysis(features)

            z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)

            condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(features, condition=condition)

        if use_prior == 'temp':
            b, c, h, w = prev_features[0].size()
            z_cur = token2feat(z_cur, block_size=(4, 4), feat_size=((b, self.d_T, h, w)))
            y_tilde += self.LRP(z_cur)

        reconstructed = self.synthesis(y_tilde)

        ### --- Difference --- ###
        return reconstructed, (y_likelihood, z_likelihood), y_tilde
        ### --- End Difference --- ###


class TransformerPriorCoderSideInfoAtDecodeConcat(TransformerPriorCoder):
    """
        Transformer-based Entropy Coder that takes 2 previously decoded latents as temporal condition
        Side information (hyperprior/low-resolution) latent is used in Transformer's decoder
    """

    def __init__(self, **kwargs):
        super(TransformerPriorCoderSideInfoAtDecodeConcat, self).__init__(**kwargs)

    def forward(self, input, use_prior='temp', prev_features=None):
        assert use_prior in ['temp', 'hyper'] # Use temporal prior or hyperprior
        if use_prior == 'temp':
            assert not (prev_features is None) and isinstance(prev_features, list), ValueError

        features = self.analysis(input)
        
        if use_prior == 'temp':
            cur_token = feat2token(features, block_size=(4, 4))
            prev_tokens = [feat2token(feat, block_size=(8, 8), stride=(4, 4), padding=[2]*4) for feat in prev_features]

            ### --- Difference --- ###
            # Perform hyperprior coding when use_prior=='temp' as well
            # Directly extracts blocks from z_tilde and treat it as part of current tokens
            hyperpriors = self.hyper_analysis(features)
            z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)

            z_token = feat2token(z_tilde, block_size=(4, 4), stride=(1, 1), padding=[2]*4)
            ### --- End Difference --- ###

            z_cur = self.temporal_prior(prev_tokens, cur_token, trg_pred=z_token)[:, :-1, :]

            condition = self.trg_word_prj(z_cur)

            # Multiply (\sqrt{d_model} ^ -1) to linear projection output
            condition *= self.d_T ** -0.5

            mean, scale = condition.chunk(2, dim=2)

            condition = torch.cat([token2feat(mean, block_size=(4, 4), feat_size=prev_features[0].size()),
                                   token2feat(scale, block_size=(4, 4), feat_size=prev_features[0].size())], dim=1)
        else:
            hyperpriors = self.hyper_analysis(features)

            z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)

            condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(features, condition=condition)

        if use_prior == 'temp':
            b, c, h, w = prev_features[0].size()
            z_cur = token2feat(z_cur, block_size=(4, 4), feat_size=((b, self.d_T, h, w)))
            y_tilde += self.LRP(z_cur)

        reconstructed = self.synthesis(y_tilde)

        ### --- Difference --- ###
        return reconstructed, (y_likelihood, z_likelihood), y_tilde
        ### --- End Difference --- ###


__CODER_TYPES__ = {
                   "GoogleHyperPriorCoder": GoogleHyperPriorCoder,
                   "TransformerPriorCoder": TransformerPriorCoder,
                   "TransformerPriorCoderSideInfoAtEncode": TransformerPriorCoderSideInfoAtEncode,
                   "TransformerPriorCoderSideInfoAtDecode": TransformerPriorCoderSideInfoAtDecode,
                   "TransformerPriorCoderSideInfoAtDecodeConcat": TransformerPriorCoderSideInfoAtDecodeConcat,
                  }
