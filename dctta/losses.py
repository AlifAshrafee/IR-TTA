"""Loss terms of the DCTTA adaptation objective.

The original ``tta.py::SRTTA.compute_loss2`` computed, for a prediction ``p`` and a
target ``q`` (both detached targets):

    L = 1.0 * L1(c*p, c*q) + 0.01 * VGG19-perceptual(p, q) + 0.1 * L1(LL(p), LL(q))

* ``c`` was a per-pixel "confidence" map ``3/2 - sigmoid(var / 4e-4)`` where ``var`` is the
  variance across four *identical* repeats of the deterministic self-ensemble. The variance is
  therefore exactly zero and ``c == 1`` everywhere, so the term is plain L1. The four repeats
  (32 teacher passes instead of 8) are dropped as well.
* The perceptual term is the KAIR-style multi-layer VGG19 feature L1 (layers 2/7/16/25/34,
  weights 0.1/0.1/1/1/1), not the single-layer VGG16 MSE that also sits unused in the repo.
* ``LL`` was ``WaveletTransform(scale=1)`` reading ``pretrain/wavelet.mat`` and keeping band 0.
  ``rec2[:4]`` in that file is the 2x2 Haar basis, so band 0 with the ``/ks`` scaling is exactly a
  2x2 box filter with stride 2 (0.25 per tap). ``HaarLowPass`` below reproduces it analytically,
  which removes the .mat dependency and scipy.

The 0.01 VGG weight matches the paper (Eqs. 5-6, 8-9). The low-pass term (weight 0.1) is not
described in the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=(2, 7, 16, 25, 34), use_input_norm=True, use_range_norm=False):
        super().__init__()
        model = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        feature_layer = list(feature_layer)
        self.features = nn.Sequential()
        feature_layer = [-1] + feature_layer
        for i in range(len(feature_layer) - 1):
            self.features.add_module(
                'child' + str(i),
                nn.Sequential(*list(model.features.children())[(feature_layer[i] + 1):(feature_layer[i + 1] + 1)]))
        for v in self.features.parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = []
        for child_model in self.features.children():
            x = child_model(x)
            output.append(x.clone())
        return output


class PerceptualLoss(nn.Module):
    """Multi-layer VGG19 feature loss (weights per layer), L1 by default."""

    def __init__(self, feature_layer=(2, 7, 16, 25, 34), weights=(0.1, 0.1, 1.0, 1.0, 1.0), lossfn_type='l1',
                 use_input_norm=True, use_range_norm=False):
        super().__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer, use_input_norm=use_input_norm,
                                       use_range_norm=use_range_norm)
        self.weights = weights
        self.lossfn = nn.L1Loss() if lossfn_type == 'l1' else nn.MSELoss()

    def forward(self, x, gt):
        x_vgg, gt_vgg = self.vgg(x), self.vgg(gt.detach())
        loss = 0.0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.lossfn(x_vgg[i], gt_vgg[i])
        return loss


class HaarLowPass(nn.Module):
    """LL band of a one-level 2x2 Haar transform == 2x2 box filter, stride 2 (per channel).

    Numerically identical to band 0 of the original ``WaveletTransform(scale=1, dec=True)``.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.full((1, 1, 2, 2), 0.25, dtype=torch.float32))

    def forward(self, x):
        b, c, h, w = x.shape
        y = F.conv2d(x.reshape(b * c, 1, h, w), self.weight, stride=2, padding=0)
        return y.reshape(b, c, y.shape[-2], y.shape[-1])


class DCTTALoss(nn.Module):
    """L1 + w_vgg * perceptual + w_lp * L1 on the Haar low-pass band."""

    def __init__(self, pixel_weight=1.0, perceptual_weight=0.01, lowpass_weight=0.1):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.lowpass_weight = lowpass_weight
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss() if perceptual_weight > 0 else None
        self.lowpass = HaarLowPass()

    def forward(self, pred, target):
        loss = self.pixel_weight * self.l1(pred, target)
        if self.perceptual is not None:
            loss = loss + self.perceptual_weight * self.perceptual(pred, target)
        if self.lowpass_weight > 0:
            loss = loss + self.lowpass_weight * self.l1(self.lowpass(pred), self.lowpass(target))
        return loss


def charbonnier(pred, target, eps=1e-3):
    """L1-Charbonnier, used by the TIPS Fisher pass (``tta.py::compute_loss``)."""
    return torch.sqrt(((pred - target) ** 2) + eps).mean()
