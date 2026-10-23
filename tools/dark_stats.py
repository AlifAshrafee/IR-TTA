"""Exposure diagnostic: can a dark input's *noise* tell 'under-exposed' (low-light: brighten) from 'dark scene,
normal exposure' (RealBlur at night: keep dark)? No model, no labels used for the statistic; GT only for the
'needs brightening' column.

    python tools/dark_stats.py LOLv2_Syn LOLv2_Syn_val RealBlurJ RealBlurJ_val RealBlurR HIDE_val

Per image (LQ): mean luminance m, noise sigma (Immerkaer 1996 fast estimator on 8-bit luminance, flat-ish pixels),
relative noise sigma / (255 m), and GT mean / LQ mean (how much the reference is brighter). A short exposure
collects few photons, so its relative noise is high (photon transfer); a long, blurred night exposure is dark but
comparatively clean. If the relative-noise ranges of the low-light and RealBlur sets do not overlap, a noise-based
under-exposure rule is viable without per-dataset settings.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from irtta.data import available_names, evenly_spaced, resolve_spec  # noqa: E402

M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def _conv3(y):
    out = np.zeros((y.shape[0] - 2, y.shape[1] - 2))
    for i in range(3):
        for j in range(3):
            out += M[i, j] * y[i:i + y.shape[0] - 2, j:j + y.shape[1] - 2]
    return out


def _box(y, k=7):
    """k x k mean filter (valid region, same size via edge padding)."""
    p = k // 2
    z = np.pad(y, p, mode='edge')
    c = z.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def noise_sigma(y255):
    """Immerkaer's estimator over the 50% flattest pixels; flatness from the gradient of a 7x7-smoothed image, so
    the selection does not depend on the noise itself (checked unbiased on smooth synthetic images, sigma 1-10)."""
    r = np.abs(_conv3(y255))
    sm = _box(y255)
    g = (np.abs(np.gradient(sm, axis=0)) + np.abs(np.gradient(sm, axis=1)))[1:-1, 1:-1]
    mask = g <= np.median(g)
    return float(np.sqrt(np.pi / 2) / 6 * r[mask].mean())


def lum(img):
    a = np.asarray(img.convert('RGB'), dtype=np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sets', nargs='+')
    ap.add_argument('--max_images', type=int, default=100)
    args = ap.parse_args()
    head = ['Set', 'n', 'mean lum (median)', 'noise sigma, 8-bit (median)', 'relative noise q10-q50-q90',
            'GT/LQ brightness (median)']
    print('| ' + ' | '.join(head) + ' |\n|' + '---|' * len(head))
    for name in args.sets:
        spec = resolve_spec(name)
        rows = []
        for nm in evenly_spaced(available_names(spec), args.max_images):
            yl = lum(Image.open(os.path.join(spec.lq, nm)))
            yg = lum(Image.open(os.path.join(spec.gt, nm)))
            m = yl.mean() / 255.
            s = noise_sigma(yl)
            rows.append((m, s, s / max(255. * m, 1e-3), yg.mean() / max(yl.mean(), 1e-3)))
        a = np.array(rows)
        q = np.quantile(a[:, 2], [0.1, 0.5, 0.9])
        print(f"| {name} | {len(a)} | {np.median(a[:, 0]):.3f} | {np.median(a[:, 1]):.2f} | "
              f"{q[0]:.3f} - {q[1]:.3f} - {q[2]:.3f} | {np.median(a[:, 3]):.2f} |", flush=True)


if __name__ == '__main__':
    main()
