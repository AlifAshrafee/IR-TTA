"""Full-image evaluation with per-image rows, optional LPIPS, brightness drift; the probe.

PSNR/SSIM are computed with ``dctta.metrics.compute_psnr_ssim`` (scikit-image, RGB, data_range 1),
i.e. exactly as the paper/original repo, and averaged in the same order as ``evaluate.py``.

Per-image ``drift`` = mean(f(x) - x): the brightness change the model applies (the collapse signal of
over-brightening, e.g. 5-task models on dark scenes).
"""
import csv
import logging
import math
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dctta.data import PairedImageDataset
from dctta.metrics import AverageMeter, compute_psnr_ssim
from dctta.utils import save_image_tensor

from .ymetrics import psnr_y


logger = logging.getLogger(__name__)


class LPIPSMetric:
    """LPIPS (AlexNet) via ``pyiqa`` if installed, else the ``lpips`` package. Built under fork_rng."""

    def __init__(self, device):
        self.device = device
        self.backend = None
        cuda_devs = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devs):
            try:
                import pyiqa
                self.fn = pyiqa.create_metric('lpips', device=device, as_loss=False)
                self.backend = 'pyiqa'
            except Exception:  # noqa: BLE001 -- fall back to the lpips package
                import lpips
                self.fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
                self.backend = 'lpips'
        logger.info(f"LPIPS backend: {self.backend}")

    @torch.no_grad()
    def __call__(self, pred, gt):
        pred, gt = pred.clamp(0, 1), gt.clamp(0, 1)
        if self.backend == 'pyiqa':
            return float(self.fn(pred, gt).mean())
        return float(self.fn(pred, gt, normalize=True).mean())


def image_row(name, degraded, restored, clean, lpips_metric=None):
    psnr, ssim = compute_psnr_ssim(restored, clean)
    row = {'name': name, 'psnr': float(psnr), 'ssim': float(ssim), 'psnr_y': psnr_y(restored, clean),
           'lpips': lpips_metric(restored, clean) if lpips_metric is not None else math.nan,
           'in_mean': float(degraded.mean()), 'out_mean': float(restored.clamp(0, 1).mean()),
           'gt_mean': float(clean.mean())}
    row['drift'] = row['out_mean'] - row['in_mean']
    return row


@torch.no_grad()
def evaluate_detailed(model, loader, device, results_dir=None, log=None, lpips_metric=None):
    """Returns ``(summary, rows)``; summary has psnr/ssim (same averaging as ``evaluate.py``), lpips, drift."""
    was_training = model.training
    model.eval()
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
    psnr_m, ssim_m = AverageMeter(), AverageMeter()
    rows = []
    for names, degraded, clean in tqdm(loader, total=len(loader), desc="Evaluating", leave=False):
        degraded, clean = degraded.to(device), clean.to(device)
        restored = model(degraded)
        row = image_row(names[0], degraded, restored, clean, lpips_metric)
        psnr_m.update(row['psnr'])
        ssim_m.update(row['ssim'])
        rows.append(row)
        if log is not None:
            log.info(f"{names[0]}: PSNR {row['psnr']:.3f} SSIM {row['ssim']:.4f} drift {row['drift']:+.4f}")
        if results_dir is not None:
            save_image_tensor(restored, os.path.join(results_dir, os.path.splitext(names[0])[0] + ".png"))
    model.train(was_training)
    return summarize(rows, psnr_m.avg, ssim_m.avg), rows


def summarize(rows, psnr=None, ssim=None):
    n = len(rows)

    def mean(k):
        v = [r[k] for r in rows if not (isinstance(r.get(k), float) and math.isnan(r[k]))]
        return sum(v) / len(v) if v else math.nan

    return {'n': n, 'psnr': float(psnr) if psnr is not None else mean('psnr'),
            'ssim': float(ssim) if ssim is not None else mean('ssim'),
            'psnr_y': mean('psnr_y'), 'lpips': mean('lpips'), 'drift': mean('drift')}


def write_rows(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def eval_loader(dataset, num_workers=0):
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------- #
# Probe
# ---------------------------------------------------------------------------- #
def read_probe_list(path):
    """Lines ``<file name>[,<tag>]``; ``#`` starts a comment. Tag defaults to 'probe'."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            entries.append((parts[0], parts[1] if len(parts) > 1 and parts[1] else 'probe'))
    if not entries:
        raise ValueError(f"empty probe list {path}")
    return entries


class Probe:
    """In-loop diagnostic on a fixed, stratified image list (uses GT; never feeds adaptation).

    Runs under ``fork_rng`` because creating a DataLoader iterator draws from the global RNG.
    Writes one row per call to ``csv_path``: step, psnr/drift over all images and per tag.
    """

    def __init__(self, list_path, lq_dir, gt_dir, device, base=16, csv_path=None, max_side=None):
        entries = read_probe_list(list_path)
        self.tags = {n: t for n, t in entries}
        self.loader = eval_loader(PairedImageDataset(lq_dir, gt_dir, base=base, names=[n for n, _ in entries],
                                                       max_side=max_side))
        self.device = device
        self.csv_path = csv_path
        self.history = []

    def __call__(self, model, step):
        with torch.random.fork_rng(devices=[]):
            summary, rows = evaluate_detailed(model, self.loader, self.device)
        out = {'step': step, 'psnr_all': summary['psnr'], 'drift_all': summary['drift']}
        for tag in sorted(set(self.tags.values())):
            sel = [r for r in rows if self.tags[r['name']] == tag]
            out[f'psnr_{tag}'] = sum(r['psnr'] for r in sel) / len(sel)
            out[f'drift_{tag}'] = sum(r['drift'] for r in sel) / len(sel)
        self.history.append(out)
        if self.csv_path:
            write_rows(self.csv_path, self.history)
        logger.info("[probe] " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                           for k, v in out.items()))
        return out
