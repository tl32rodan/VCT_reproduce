import torch
from torch import nn

from context_model import ContextModel
from entropy_models import __CONDITIONS__, EntropyBottleneck
from generalizedivisivenorm import GeneralizedDivisiveNorm
from modules import Conv2d, ConvTranspose2d


class CompressesModel(nn.Module):
    """Basic Compress Model"""

    def __init__(self):
        super(CompressesModel, self).__init__()
        self.divisor = None
        self.num_bitstreams = 1

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
        self.num_bitstreams = 2

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


__CODER_TYPES__ = {
                   "GoogleHyperPriorCoder": GoogleHyperPriorCoder,
                  }
