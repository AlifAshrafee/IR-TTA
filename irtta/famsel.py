"""Set-level family selection for transfer TTA (``--fam_select abc``): ABC model choice over degradation families.

Per-crop ABC (``--abc_k``) picks, for each clean crop, the closest of k candidates whose families are drawn
uniformly -- the family is never decided for the target as a whole, and with five families the per-crop winner is
often the wrong one (SOTS-Indoor and LOLv2-S pick blur most often). Model choice decides it at the set level:

    for each family f:   n_pilot clean crops, k candidates from f alone (at a target crop's exposure), keep the
                         closest to the target's feature Gaussian N_T  ->  distances d_f
    acceptance a_f     = fraction of d_f within eps (the q-quantile of the target crops' own distances under N_T)
    weights  w_f       = a_f / sum a   (ABC model posterior under a uniform prior; Toni et al., J. R. Soc. Interface
                                        2009; Grelaud et al., Bayesian Analysis 2009)
    no acceptance at all -> the family with the smallest median distance.

Training then draws each pair's family from w (per-crop ABC still picks the candidate within the drawn family).

Features (``fam_features``): the 32 NSS + colour features of irtta/nss.py plus five gain-invariant statistics for
what MSCN normalises away -- exposure / tone and blur:
    luminance quantiles q05, q50, q95 over the mean (histogram shape: a gamma / tone curve changes it, a gain not)
    log relative noise sigma / mean (Immerkaer's estimator on the flatter half; short exposures are noisier)
    slope of the radially averaged log power spectrum vs log frequency (steeper with blur)
Caveat (Robert et al., PNAS 2011): ABC model choice is only as good as the summaries' power to tell the models apart;
tools/family_check.py measures that on sets whose family is known.
"""
import math

import torch
import torch.nn.functional as F

from .degrade import degrade_batch
from .nss import GaussianModel, luminance, nss_features

_IMM = torch.tensor([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])


def extra_features(x):
    """(B, 3, H, W) in [0, 1] -> (B, 5) gain-invariant exposure / blur statistics."""
    y = luminance(x)
    B, _, H, W = y.shape
    m = y.mean(dim=(1, 2, 3)) + 1e-3
    q = torch.quantile(y.flatten(1), torch.tensor([0.05, 0.5, 0.95], device=y.device, dtype=y.dtype), dim=1)
    qr = (q / m).T                                                         # (B, 3)
    # relative noise: Immerkaer (1996) on the flatter half (flatness from a 7x7-smoothed image)
    r = F.conv2d(y, _IMM.to(y)[None, None]).abs()                          # (B, 1, H-2, W-2)
    sm = F.avg_pool2d(F.pad(y, (3, 3, 3, 3), mode='reflect'), 7, stride=1)
    g = (sm[:, :, 2:, 1:-1] - sm[:, :, :-2, 1:-1]).abs() + (sm[:, :, 1:-1, 2:] - sm[:, :, 1:-1, :-2]).abs()
    med = g.flatten(1).median(dim=1).values.view(B, 1, 1, 1)
    mask = (g <= med).float()
    sigma = math.sqrt(math.pi / 2) / 6 * (r * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1)
    rel = torch.log(sigma / m + 1e-4)
    # spectral slope (Hann window, 12 log-spaced radial bins between 2 cycles/patch and Nyquist)
    yc = (y - y.mean(dim=(2, 3), keepdim=True)) * (torch.hann_window(H, device=y.device, dtype=y.dtype)[:, None]
                                                   * torch.hann_window(W, device=y.device, dtype=y.dtype)[None])
    P = torch.fft.fft2(yc).abs().pow(2).flatten(1)                         # (B, H*W)
    fy = torch.fft.fftfreq(H, device=y.device, dtype=y.dtype)
    fx = torch.fft.fftfreq(W, device=y.device, dtype=y.dtype)
    rad = torch.sqrt(fy[:, None] ** 2 + fx[None] ** 2).flatten()
    edges = torch.logspace(math.log10(2. / min(H, W)), math.log10(0.5), 13, device=y.device, dtype=y.dtype)
    idx = torch.bucketize(rad, edges) - 1                                  # bin of each frequency, -1 / 12 = outside
    ok = (idx >= 0) & (idx < 12)
    onehot = F.one_hot(idx.clamp(0, 11), 12).to(y.dtype) * ok[:, None].to(y.dtype)      # (H*W, 12)
    pb = (P @ onehot) / onehot.sum(0).clamp_min(1)                         # (B, 12)
    lr = torch.log(torch.sqrt(edges[:-1] * edges[1:]))
    lp = torch.log(pb + 1e-12)
    lrc = lr - lr.mean()
    slope = ((lp - lp.mean(dim=1, keepdim=True)) * lrc).sum(dim=1) / (lrc ** 2).sum()
    return torch.cat([qr, rel[:, None], slope[:, None]], dim=1)


def fam_features(x, bs=64):
    out = []
    for i in range(0, x.shape[0], bs):
        t = x[i:i + bs]
        out.append(torch.cat([nss_features(t, color=True), extra_features(t)], dim=1))
    return torch.cat(out, 0)


@torch.no_grad()
def family_posterior(tcrops, bank, families, g, *, k=8, n_pilot=64, eps_q=0.9, mode='camera', size=192,
                     ll='v1', blur='v1'):
    """``tcrops``: (N, 3, s, s) target crops (on the working device). Returns (weights list aligned with
    ``families``, {family: {'acc', 'med'}}, eps)."""
    dev = tcrops.device
    ft = fam_features(tcrops)
    tm = GaussianModel(ft).to(dev)
    own = tm.distance(ft)
    eps = float(torch.quantile(own, eps_q))
    sel = lambda cand: tm.distance(fam_features(cand))  # noqa: E731
    stats = {}
    for f in families:
        d = []
        for i in range(0, n_pilot, 8):
            c = bank.sample(min(8, n_pilot - i), size, g).to(dev)
            for j in range(c.shape[0]):
                x = tcrops[int(torch.randint(len(tcrops), (1,), generator=g))][None]
                tmean = x.mean(dim=(0, 2, 3), keepdim=True) if mode != 'none' else None
                _, _, _, sc = degrade_batch(c[j:j + 1], [f], g, tmean, mode if mode != 'none' else 'fit',
                                            select=sel, k=k, ll=ll, blur=blur)
                d += sc
        d = torch.tensor(d)
        stats[f] = {'acc': float((d <= eps).float().mean()), 'med': float(d.median())}
    acc = [stats[f]['acc'] for f in families]
    if sum(acc) > 0:
        w = [a / sum(acc) for a in acc]
    else:
        best = min(families, key=lambda f: stats[f]['med'])
        w = [1.0 if f == best else 0.0 for f in families]
    return w, stats, eps


__all__ = ['extra_features', 'fam_features', 'family_posterior']
