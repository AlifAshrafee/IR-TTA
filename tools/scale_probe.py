"""Q1 (spectral-gain oracle) and Q2a (scale-teacher advantage): evaluation only, no training.
See the project doc ``ir_tta_build_order.md`` §2-3 (T1, T3).

    python tools/scale_probe.py --backbone promptir --ckpt pretrain/model.ckpt --target Rain100H
    python tools/scale_probe.py --report results/probe          # -> results/probe/probe_<setting>.md

One pass per (backbone, source, target) computes, for every image x with ground truth y:

* the source f(x) and the source applied twice f(f(x)) (the harness PSNR: clipped to [0, 1], data range 1);
* **Q1**: the source residual r = f(x) - x split into three bands that sum to r exactly
  (difference of Gaussians, sigma1 < sigma2 in pixels):
      low  = G_sigma2 * r                 periods >~ 32 px at the defaults (illumination, haze, exposure)
      mid  = G_sigma1 * r - G_sigma2 * r  ~8-32 px (edges, blur scale, streaks)
      high = r - G_sigma1 * r             <~ 8 px (noise, fine texture, thin streaks)
  and the sufficient statistics of ||x + sum_b lam_b r_b - y||^2 (Gram G, cross terms h, ||x - y||^2), so the
  set-level oracle gains (one global lam, or one lam per band, chosen with GT to maximise the set's mean PSNR)
  are solved exactly afterwards. These PSNRs are unclipped, i.e. a lower bound on the clipped metric at the
  same lam (clipping to [0, 1] never increases the error); "Source (unclipped)" shows the size of that gap.
* **Q2a**: for s in --scales, with S_s an anti-aliased bicubic downscale by s:
      delta_s = PSNR(f(S_s x), S_s y) - PSNR(S_s f(x), S_s y)
  delta_s > 0: restoring the downscaled image beats downscaling the restored one (a coarse-scale teacher with
  information the model's own output lacks). T3 predicts > 0 on blur/noise/rain, ~0 on haze/low-light.

Writes ``<out_root>/<exp>/<run_id>/{probe.json, per_image.csv, stats.npz}``; finished runs are skipped.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dctta.models import BACKBONES, build_model, parse_backbone_opts  # noqa: E402
from irtta.data import load_registry, make_eval_dataset, resolve_spec, DEFAULT_REGISTRY  # noqa: E402
from irtta.evaluation import eval_loader  # noqa: E402

BANDS = ('low', 'mid', 'high')
LAM_MAX = 5.0


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #
def forward(model, x, multiple=16):
    """f(x) for any size: reflect-pad to a multiple of ``multiple``, run, crop back (no-op on aligned input)."""
    H, W = x.shape[-2:]
    ph, pw = (-H) % multiple, (-W) % multiple
    xp = F.pad(x, (0, pw, 0, ph), mode='reflect') if ph or pw else x
    return model(xp)[..., :H, :W]


def downscale(t, s):
    """S_s: anti-aliased bicubic downscale by ``s`` (input clipped to [0, 1] first, output clipped)."""
    H, W = t.shape[-2:]
    size = (max(1, round(H / s)), max(1, round(W / s)))
    return F.interpolate(t.clamp(0, 1), size=size, mode='bicubic', antialias=True, align_corners=False).clamp(0, 1)


def gaussian_blur(t, sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=t.dtype, device=t.device) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    C = t.shape[1]
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(t, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def split_bands(r, sigma1, sigma2):
    """(low, mid, high) with low + mid + high == r exactly."""
    g1, g2 = gaussian_blur(r, sigma1), gaussian_blur(r, sigma2)
    return g2, g1 - g2, r - g1


def psnr_t(a, b):
    """PSNR of the clipped images, float64: the PSNR of dctta.metrics.compute_psnr_ssim (skimage), without SSIM."""
    mse = float(((a.clamp(0, 1) - b.clamp(0, 1)).double() ** 2).mean())
    return 10 * math.log10(1.0 / max(mse, 1e-12))


# --------------------------------------------------------------------------- #
# Q1 oracle: mean over images of -10 log10 MSE_i(lam), MSE_i = (c_i - 2 h_i.lam + lam' G_i lam) / N_i
# --------------------------------------------------------------------------- #
def mean_psnr(lam, G, h, c, N):
    lam = np.asarray(lam, dtype=np.float64)
    mse = (c - 2 * h @ lam + np.einsum('ibc,b,c->i', G, lam, lam)) / N
    return float(np.mean(-10 * np.log10(np.maximum(mse, 1e-12))))


def _neg_and_grad(lam, G, h, c, N):
    mse = np.maximum((c - 2 * h @ lam + np.einsum('ibc,b,c->i', G, lam, lam)) / N, 1e-12)
    dmse = (-2 * h + 2 * np.einsum('ibc,c->ib', G, lam)) / N[:, None]
    val = np.mean(10 * np.log10(mse))
    grad = np.mean(10 / np.log(10) * dmse / mse[:, None], axis=0)
    return val, grad


def solve_oracle(G, h, c, N):
    """Set-level oracle lams: one global lam (all bands equal) and one lam per band, both in [0, LAM_MAX]."""
    from scipy.optimize import minimize, minimize_scalar
    nb = h.shape[1]
    g = minimize_scalar(lambda a: -mean_psnr(np.full(nb, a), G, h, c, N), bounds=(0.0, LAM_MAX), method='bounded',
                        options={'xatol': 1e-4})
    a = float(g.x)
    starts = [np.full(nb, a), np.ones(nb)]
    best = None
    for x0 in starts:
        res = minimize(_neg_and_grad, x0, args=(G, h, c, N), jac=True, method='L-BFGS-B',
                       bounds=[(0.0, LAM_MAX)] * nb)
        if best is None or res.fun < best.fun:
            best = res
    band = best.x
    if mean_psnr(band, G, h, c, N) < mean_psnr(np.full(nb, a), G, h, c, N):  # per-band can only be >= global
        band = np.full(nb, a)
    return {'global_lambda': a, 'global_psnr': mean_psnr(np.full(nb, a), G, h, c, N),
            'band_lambda': [float(v) for v in band], 'band_psnr': mean_psnr(band, G, h, c, N),
            'source_unclipped_psnr': mean_psnr(np.ones(nb), G, h, c, N)}


# --------------------------------------------------------------------------- #
def run_probe(args):
    reg = load_registry(args.registry)
    spec = resolve_spec(args.target, reg)
    source = args.source or _infer_source(args.ckpt)
    run_id = '-'.join([args.backbone, source, spec.name] + ([f"mi{args.max_images}"] if args.max_images else []))
    out = os.path.join(args.out_root, args.exp, run_id)
    if os.path.exists(os.path.join(out, 'probe.json')) and not args.overwrite:
        print(f"[skip] {out}")
        return json.load(open(os.path.join(out, 'probe.json')))
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.backbone, args.ckpt, **parse_backbone_opts(args.backbone_opts)).to(device).eval()
    ds = make_eval_dataset(spec, base=16, max_images=args.max_images or None)
    rows, Gs, hs, cs, Ns = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (names, x, y) in enumerate(eval_loader(ds, args.num_workers)):
            x, y = x.to(device), y.to(device)
            fx = forward(model, x)
            ffx = forward(model, fx)
            row = {'name': names[0], 'input': psnr_t(x, y), 'source': psnr_t(fx, y), 'source_x2': psnr_t(ffx, y)}
            # Q1 statistics (float64)
            bands = [b.double().flatten() for b in split_bands(fx - x, args.sigma1, args.sigma2)]
            e0 = (y - x).double().flatten()
            Gs.append(torch.stack([torch.stack([bi @ bj for bj in bands]) for bi in bands]).cpu().numpy())
            hs.append(torch.stack([b @ e0 for b in bands]).cpu().numpy())
            cs.append(float(e0 @ e0))
            Ns.append(e0.numel())
            # Q2a
            for s in args.scales:
                xs, ys = downscale(x, s), downscale(y, s)
                teacher = forward(model, xs)
                own = downscale(fx, s)
                row[f'in_s{s:g}'] = psnr_t(xs, ys)
                row[f'teacher_s{s:g}'] = psnr_t(teacher, ys)
                row[f'own_s{s:g}'] = psnr_t(own, ys)
                row[f'delta_s{s:g}'] = row[f'teacher_s{s:g}'] - row[f'own_s{s:g}']
            rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(ds)}  ({time.time() - t0:.0f}s)")
    G, h, c, N = np.stack(Gs), np.stack(hs), np.array(cs), np.array(Ns, dtype=np.float64)
    np.savez(os.path.join(out, 'stats.npz'), G=G, h=h, c=c, N=N, names=np.array([r['name'] for r in rows]))
    with open(os.path.join(out, 'per_image.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    mean = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    res = {'run_id': run_id, 'backbone': args.backbone, 'source': source, 'target': spec.name, 'split': spec.split,
           'task': spec.task, 'n': len(rows), 'ckpt': args.ckpt, 'sigma1': args.sigma1, 'sigma2': args.sigma2,
           'bands': list(BANDS), 'input_psnr': mean('input'), 'source_psnr': mean('source'),
           'source_x2_psnr': mean('source_x2'), 'time_s': time.time() - t0, **solve_oracle(G, h, c, N)}
    for s in args.scales:
        d = np.array([r[f'delta_s{s:g}'] for r in rows])
        res[f'delta_s{s:g}'] = float(d.mean())
        res[f'delta_s{s:g}_pos'] = float((d > 0).mean())
        res[f'teacher_s{s:g}'] = mean(f'teacher_s{s:g}')
        res[f'own_s{s:g}'] = mean(f'own_s{s:g}')
        res[f'in_s{s:g}'] = mean(f'in_s{s:g}')
    with open(os.path.join(out, 'probe.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(f"{run_id}: source {res['source_psnr']:.2f}  x2 {res['source_x2_psnr']:.2f}  "
          f"global lam {res['global_lambda']:.2f} -> {res['global_psnr']:.2f}  "
          f"bands {['%.2f' % v for v in res['band_lambda']]} -> {res['band_psnr']:.2f}  "
          + '  '.join(f"delta_s{s:g} {res[f'delta_s{s:g}']:+.2f} ({100 * res[f'delta_s{s:g}_pos']:.0f}% > 0)"
                      for s in args.scales))
    return res


def _infer_source(ckpt):
    from run_tta import infer_source
    return infer_source(ckpt)


# --------------------------------------------------------------------------- #
def report(root):
    res = []
    for d, _, files in os.walk(root):
        if 'probe.json' in files:
            res.append(json.load(open(os.path.join(d, 'probe.json'))))
    if not res:
        raise SystemExit(f"no probe.json under {root}")
    scales = sorted({k[len('delta_s'):] for r in res for k in r if k.startswith('delta_s') and not k.endswith('_pos')},
                    key=float)
    for setting in sorted({r['source'] for r in res}):
        lines = [f"## Q1 / Q2a, {setting} sources (PSNR dB)\n",
                 "Q1: x + sum_b lam_b B_b(f(x) - x), lam chosen per set with GT (oracle, unclipped = lower bound). "
                 "Bands low/mid/high. Q2a: delta_s = PSNR(f(S_s x), S_s y) - PSNR(S_s f(x), S_s y), "
                 "% = images with delta > 0.\n"]
        head = (['Backbone', 'Set', 'n', 'Input', 'Source', 'Source x2', 'Src unclip', 'lam*', 'Global',
                 'lam* low/mid/high', 'Per-band', 'Per-band - src']
                + [f"delta_s{s} (%>0)" for s in scales])
        lines += ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
        for r in sorted((r for r in res if r['source'] == setting), key=lambda r: (r['backbone'], r['split'] != 'test',
                                                                                    r['task'], r['target'])):
            tgt = r['target'] + ('' if r['split'] == 'test' else f" ({r['split']})")
            cells = [r['backbone'], tgt, str(r['n']), f"{r['input_psnr']:.2f}", f"{r['source_psnr']:.2f}",
                     f"{r['source_x2_psnr']:.2f}", f"{r['source_unclipped_psnr']:.2f}", f"{r['global_lambda']:.2f}",
                     f"{r['global_psnr']:.2f}", '/'.join(f"{v:.2f}" for v in r['band_lambda']),
                     f"{r['band_psnr']:.2f}", f"{r['band_psnr'] - r['source_unclipped_psnr']:+.2f}"]
            cells += [f"{r[f'delta_s{s}']:+.2f} ({100 * r[f'delta_s{s}_pos']:.0f}%)" if f'delta_s{s}' in r else ''
                      for s in scales]
            lines.append('| ' + ' | '.join(cells) + ' |')
        md = '\n'.join(lines)
        path = os.path.join(root, f'probe_{setting}.md')
        with open(path, 'w') as f:
            f.write(md + '\n')
        print(md + f"\n\n-> {path}\n")


def get_parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--report', type=str, default=None, help='folder of probe runs: write probe_<setting>.md')
    p.add_argument('--target', type=str)
    p.add_argument('--ckpt', type=str)
    p.add_argument('--backbone', type=str, default='promptir', choices=list(BACKBONES))
    p.add_argument('--backbone_opts', type=str, nargs='*', default=[])
    p.add_argument('--source', type=str, default=None, help='3task / 5task (default: from the checkpoint path)')
    p.add_argument('--registry', type=str, default=DEFAULT_REGISTRY)
    p.add_argument('--max_images', type=int, default=0, help='evenly spaced subset (validation sets: 300)')
    p.add_argument('--scales', type=float, nargs='+', default=[2, 3])
    p.add_argument('--sigma1', type=float, default=1.5, help='high/mid split (px); half-power at ~0.19/sigma cyc/px')
    p.add_argument('--sigma2', type=float, default=6.0, help='mid/low split (px)')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--out_root', type=str, default='results')
    p.add_argument('--exp', type=str, default='probe')
    p.add_argument('--overwrite', action='store_true')
    return p


def main(argv=None):
    args = get_parser().parse_args(argv)
    if args.report:
        return report(args.report)
    if not (args.target and args.ckpt):
        raise SystemExit("--target and --ckpt are required (or --report <dir>)")
    return run_probe(args)


if __name__ == '__main__':
    main()
