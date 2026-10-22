"""Which PSNR/SSIM convention reproduces the co-author's numbers? (Rain100H discrepancy, ~1.37 dB on every backbone)

    python tools/metric_check.py --target Rain100H                  # input + all 12 released checkpoints
    python tools/metric_check.py --target Rain100H --models promptir:3task r2r:5task

Evaluates the un-adapted source (and the input) once and scores every image under four conventions:

  rgb       ours: RGB, float outputs clipped to [0, 1], data range 1 (scikit-image, as the PromptIR/DCTTA code)
  rgb_u8    RGB after rounding to 8 bits (i.e. scored from saved PNGs), peak 255
  y_luma    luminance Y = 0.299 R + 0.587 G + 0.114 B in [0, 1], data range 1
  y_bt601   the deraining convention (MATLAB rgb2ycbcr / BasicSR ``test_y_channel``): 8-bit RGB ->
            Y = 16 + (65.481 R + 128.553 G + 24.966 B) / 255 in [16, 235], peak 255
  ssim_y    SSIM on y_bt601 as BasicSR computes it (11x11 Gaussian, sigma 1.5, valid region)

and prints them next to the co-author's value (configs/sota_<setting>.csv). The convention whose column matches
the co-author on every backbone is theirs.
"""
import argparse
import csv
import math
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dctta.models import build_model  # noqa: E402
from irtta.data import make_eval_dataset, resolve_spec  # noqa: E402
from irtta.evaluation import eval_loader  # noqa: E402

CK = {  # same layout as scripts/run.sh
    'promptir:3task': 'pretrain/model.ckpt', 'promptir:5task': 'pretrain/epoch=80.ckpt',
    'adair:3task': 'pretrain/adair/adair3d.ckpt', 'adair:5task': 'pretrain/adair/adair5d.ckpt',
    'bioir:3task': 'pretrain/bioir/BioIR3D.ckpt', 'bioir:5task': 'pretrain/bioir/BioIR5D.ckpt',
    'moceir:3task': 'pretrain/moceir/MoCE_IR_AIO3/last.ckpt', 'moceir:5task': 'pretrain/moceir/MoCE_IR_AIO5/last.ckpt',
    'mirage:3task': 'pretrain/mirage/3Deg_small.ckpt', 'mirage:5task': 'pretrain/mirage/5Deg_small.ckpt',
    'r2r:3task': 'pretrain/r2r/train_ckpt_3D_finetune/last.ckpt', 'r2r:5task': 'pretrain/r2r/train_ckpt_5D_finetune/last.ckpt',
}
NAMES = {'promptir': 'PromptIR', 'adair': 'AdaIR', 'bioir': 'Bio-IR', 'moceir': 'MoCE-IR', 'mirage': 'MIRAGE',
         'r2r': 'R2R'}
COLS = ['rgb', 'rgb_u8', 'y_luma', 'y_bt601', 'ssim_y']


def _psnr(a, b, peak):
    mse = float(((a.double() - b.double()) ** 2).mean())
    return 10 * math.log10(peak ** 2 / max(mse, 1e-12))


def _ybt601(u8):
    return 16. + (65.481 * u8[:, 0] + 128.553 * u8[:, 1] + 24.966 * u8[:, 2]) / 255.


def _ssim_basicsr(a, b):
    """a, b: (1, H, W) float64 on 0..255 (BasicSR ``_ssim``: Gaussian 11x11 / 1.5, valid region)."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = torch.exp(-(torch.arange(11, dtype=torch.float64, device=a.device) - 5) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    f = lambda t: F.conv2d(t[:, None], w)  # noqa: E731
    m1, m2 = f(a), f(b)
    s1, s2, s12 = f(a * a) - m1 ** 2, f(b * b) - m2 ** 2, f(a * b) - m1 * m2
    s = ((2 * m1 * m2 + C1) * (2 * s12 + C2)) / ((m1 ** 2 + m2 ** 2 + C1) * (s1 + s2 + C2))
    return float(s.mean())


def score(out, gt):
    o, g = out.clamp(0, 1), gt.clamp(0, 1)
    ou, gu = (o * 255).round(), (g * 255).round()
    lum = lambda t: 0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]  # noqa: E731
    yo, yg = _ybt601(ou).double(), _ybt601(gu).double()
    return {'rgb': _psnr(o, g, 1.0), 'rgb_u8': _psnr(ou, gu, 255.0), 'y_luma': _psnr(lum(o), lum(g), 1.0),
            'y_bt601': _psnr(yo, yg, 255.0), 'ssim_y': _ssim_basicsr(yo, yg)}


def reported(setting, target, method):
    path = os.path.join(REPO, 'configs', f'sota_{setting}.csv')
    if not os.path.exists(path):
        return '', ''
    for r in csv.DictReader(open(path)):
        if r['method'] == method:
            return r.get(target, ''), r.get(f'{target}_ssim', '')
    return '', ''


@torch.no_grad()
def evaluate(model, ds, device):
    tot, n = {k: 0.0 for k in COLS}, 0
    for _, x, y in eval_loader(ds, 2):
        x, y = x.to(device), y.to(device)
        s = score(model(x) if model is not None else x, y)
        for k in COLS:
            tot[k] += s[k]
        n += 1
    return {k: v / n for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='Rain100H')
    ap.add_argument('--models', nargs='*', default=list(CK), help='backbone:setting keys (default: all 12)')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = make_eval_dataset(resolve_spec(args.target))
    head = ['Model', 'rgb (ours)', 'rgb_u8', 'y_luma', 'y_bt601', 'SSIM-Y', 'co-author PSNR', 'co-author SSIM']
    print(f"## {args.target}: un-adapted source under four PSNR conventions\n")
    print('| ' + ' | '.join(head) + ' |\n|' + '---|' * len(head))
    res = evaluate(None, ds, device)
    p, s = reported('5task', args.target, 'No IR')
    print('| Input | ' + ' | '.join(f"{res[k]:.4f}" if k == 'ssim_y' else f"{res[k]:.2f}" for k in COLS) + f" | {p} | {s} |")
    for key in args.models:
        ck = os.path.join(REPO, CK[key])
        if not os.path.exists(ck):
            print(f"| {key} | checkpoint missing |")
            continue
        bb, setting = key.split(':')
        opts = {'routing_noise': False} if bb == 'moceir' else {}
        model = build_model(bb, ck, **opts).to(device).eval()
        res = evaluate(model, ds, device)
        p, s = reported(setting, args.target, NAMES[bb])
        print(f"| {NAMES[bb]} {setting} | " + ' | '.join(f"{res[k]:.4f}" if k == 'ssim_y' else f"{res[k]:.2f}"
                                                   for k in COLS) + f" | {p} | {s} |", flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
