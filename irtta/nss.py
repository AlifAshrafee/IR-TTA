"""Natural-scene-statistics (NSS) features and Gaussian models of them, differentiable.

Used by transfer TTA for two distribution-level terms (see ``irtta/methods.py``):

* **ABC pair selection** (``--abc_k``): the target images define a Gaussian N(mu_T, Sigma_T) over NSS features of
  their crops. For every clean crop, K candidate degradations are drawn from the operator prior and the candidate
  whose degraded crop is closest to the target Gaussian (Mahalanobis distance) is kept. This is approximate
  Bayesian computation: sampling the degradation prior conditioned on matching the target's summary statistics,
  i.e. an approximate posterior over degradations given the (unlabelled) target set.
* **Output prior** (``--nss_weight``): the restored target patch should have the NSS of clean images. The loss is
  the Mahalanobis distance of its features to N(mu_C, Sigma_C), fitted on the clean bank (NIQE's construction,
  Mittal, Soundararajan & Bovik, IEEE SPL 2013, as a differentiable cross-entropy term).

Features (BRISQUE's, Mittal, Moorthy & Bovik, IEEE TIP 2012, with moment estimates instead of GGD/AGGD fits so that
they are differentiable). On luminance, at full and half resolution:
  MSCN coefficients m = (I - mu) / (sigma + C), mu/sigma from a 7x7 Gaussian window (sigma 7/6), C = 1/255:
    E[m^2], (E|m|)^2 / E[m^2]                          (variance and the GGD shape statistic)
  paired products m(p) m(p + d), d in {horizontal, vertical, two diagonals}:
    mean, variance, (E|p|)^2 / E[p^2]                  (AGGD-style asymmetry, spread and shape)
  log mean local contrast, log E[sigma]                (MSCN normalises contrast away; haze and blur change it)
-> 15 per scale, 30 in total. MSCN statistics identify distortion type and severity (BRISQUE, DIIVINE) and are
invariant to a global brightness gain, so they complement the exposure matching of the pairs.
"""
import torch
import torch.nn.functional as F

N_FEATURES = 30
_SHIFTS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _window(device, dtype, size=7, sigma=7 / 6):
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def luminance(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).clamp(0, 1)


def _scale_features(y, w, C=1 / 255, border=3):
    mu = F.conv2d(F.pad(y, (3, 3, 3, 3), mode='reflect'), w)
    var = F.conv2d(F.pad(y * y, (3, 3, 3, 3), mode='reflect'), w) - mu * mu
    sig = torch.sqrt(var.clamp_min(0) + 1e-8)
    m = (y - mu) / (sig + C)
    m = m[..., border:-border, border:-border]
    sig = sig[..., border:-border, border:-border]

    def shape(t):  # (E|t|)^2 / E[t^2]: 2/pi for a Gaussian, smaller for heavier tails
        return t.abs().mean(dim=(1, 2, 3)) ** 2 / (t.pow(2).mean(dim=(1, 2, 3)) + 1e-8)

    feats = [m.pow(2).mean(dim=(1, 2, 3)), shape(m)]
    H, W = m.shape[-2:]
    for dy, dx in _SHIFTS:
        if dx >= 0:
            a, b = m[..., :H - dy, :W - dx], m[..., dy:, dx:]
        else:
            a, b = m[..., :H - dy, -dx:], m[..., dy:, :W + dx]
        p = a * b
        pm = p.mean(dim=(1, 2, 3))
        feats += [pm, p.pow(2).mean(dim=(1, 2, 3)) - pm ** 2, shape(p)]
    feats.append(torch.log(sig.mean(dim=(1, 2, 3)) + 1e-3))
    return feats


def nss_features(x, color=False):
    """x: (B, 3, H, W) in [0, 1], H, W >= 32 -> (B, 30) features (differentiable); ``color`` appends 2 (B, 32).

    Colour features (gain-invariant ratios), for what luminance MSCN cannot see -- haze vs blur:
      dark channel / brightness: mean of min over RGB and a 15x15 window, over mean luminance. Haze-free images
        have a near-zero dark channel and airlight raises it (He, Sun & Tang, CVPR 2009);
      saturation: mean (max - min) / max over RGB; scattering desaturates toward the airlight colour."""
    y = luminance(x)
    w = _window(x.device, x.dtype)
    feats = _scale_features(y, w)
    y2 = F.avg_pool2d(y, 2)
    feats += _scale_features(y2, w)
    if color:
        dc = -F.max_pool2d(-x.min(dim=1, keepdim=True).values, 15, stride=1, padding=7)
        lum = y.mean(dim=(1, 2, 3)) + 1e-3
        feats.append(dc.mean(dim=(1, 2, 3)) / lum)
        mx, mn = x.max(dim=1).values, x.min(dim=1).values
        feats.append(((mx - mn) / (mx + 1e-3)).mean(dim=(1, 2)))
    return torch.stack(feats, dim=1)


class GaussianModel:
    """N(mu, Sigma) of feature rows, Sigma shrunk toward its diagonal (``shrink``) for a stable inverse.
    ``distance(f)`` = Mahalanobis^2 / D, about 1 for samples of the fitted distribution."""

    def __init__(self, feats, shrink=0.1, eps=1e-6):
        f = feats.detach().double()
        if f.shape[0] < 2:
            raise ValueError("need at least 2 feature rows to fit a Gaussian")
        self.mu = f.mean(0)
        d = f - self.mu
        cov = d.T @ d / (f.shape[0] - 1)
        cov = (1 - shrink) * cov + shrink * torch.diag(torch.diag(cov))
        cov = cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device) * max(
            float(torch.diag(cov).mean()), 1e-12)
        self.prec = torch.linalg.inv(cov)
        self.n = f.shape[0]
        self.dim = f.shape[1]

    def to(self, device):
        self.mu, self.prec = self.mu.to(device), self.prec.to(device)
        return self

    def distance(self, feats):
        d = feats.double() - self.mu
        return ((d @ self.prec) * d).sum(dim=1).float() / self.dim

    def kl_from(self, other):
        """KL(other || self) for two GaussianModels (diagnostic: how far apart two sets are)."""
        cov_o = torch.linalg.inv(other.prec)
        d = (self.mu - other.mu)
        k = self.dim
        return 0.5 * float(torch.trace(self.prec @ cov_o) + d @ self.prec @ d - k
                           + torch.logdet(torch.linalg.inv(self.prec)) - torch.logdet(cov_o))


def random_crop_batch(x, size, n, generator):
    """``n`` random size x size crops of a (1, C, H, W) tensor (positions from a CPU generator)."""
    _, _, H, W = x.shape
    s = min(size, H, W)
    out = []
    for _ in range(n):
        i = int(torch.randint(0, H - s + 1, (1,), generator=generator))
        j = int(torch.randint(0, W - s + 1, (1,), generator=generator))
        out.append(x[..., i:i + s, j:j + s])
    return torch.cat(out, 0)


def downscale_crops(x, factor):
    """Anti-aliased bicubic downscale (clean-corpus crops at a coarser scale)."""
    if factor == 1:
        return x
    H, W = x.shape[-2:]
    return F.interpolate(x, size=(max(32, round(H / factor)), max(32, round(W / factor))), mode='bicubic',
                         antialias=True, align_corners=False).clamp(0, 1)


__all__ = ['N_FEATURES', 'nss_features', 'GaussianModel', 'random_crop_batch', 'downscale_crops', 'luminance']
