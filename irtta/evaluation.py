"""Full-image evaluation: PSNR / SSIM (RGB) and luminance PSNR, per-image rows and set averages.

* PSNR / SSIM: scikit-image on RGB, outputs clipped to [0, 1], ``data_range=1`` -- the PromptIR / DCTTA evaluation.
* ``psnr_y``: the deraining convention (MATLAB ``rgb2ycbcr`` / BasicSR ``test_y_channel``): 8-bit RGB ->
  Y = 16 + (65.481 R + 128.553 G + 24.966 B) / 255 on [16, 235], peak 255. Rain sets are reported with it
  (``"metric": "y"`` in ``configs/datasets.json``); it reproduces the published Rain100H numbers of the backbones.
"""
import csv
import logging
import math
import os

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AverageMeter:
    def __init__(self):
        self.sum = self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_psnr_ssim(recovered, clean):
    """(B, 3, H, W) tensors -> batch-mean RGB PSNR and SSIM (scikit-image, data_range 1)."""
    assert recovered.shape == clean.shape
    recovered = np.clip(recovered.detach().cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)
    psnr = ssim = 0.0
    for i in range(recovered.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recovered[i], data_range=1)
        ssim += structural_similarity(clean[i], recovered[i], data_range=1, channel_axis=-1)
    return psnr / recovered.shape[0], ssim / recovered.shape[0]


def y_bt601(img01):
    """(B, 3, H, W) in [0, 1] -> (B, H, W) float64 Y on 16..235 from the 8-bit-rounded image."""
    u8 = (img01.clamp(0, 1) * 255).round().double()
    return 16. + (65.481 * u8[:, 0] + 128.553 * u8[:, 1] + 24.966 * u8[:, 2]) / 255.


def psnr_y(out, gt):
    mse = float(((y_bt601(out) - y_bt601(gt)) ** 2).mean())
    return 10 * math.log10(255. ** 2 / max(mse, 1e-12))


def save_image_tensor(image_tensor, path):
    """1x3xHxW in [0, 1] -> 8-bit PNG."""
    arr = np.clip(image_tensor.detach().cpu().numpy()[0] * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(arr).save(path)


def image_row(name, restored, clean):
    psnr, ssim = compute_psnr_ssim(restored, clean)
    return {'name': name, 'psnr': float(psnr), 'ssim': float(ssim), 'psnr_y': psnr_y(restored, clean)}


@torch.no_grad()
def evaluate_detailed(model, loader, device, results_dir=None, log=None, save_only=None):
    """Returns ``(summary, rows)``: set means of psnr / ssim / psnr_y and one row per image. With ``results_dir``
    the restored images are saved there (``save_only``: a set of name stems to save, default all)."""
    was_training = model.training
    model.eval()
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
    psnr_m, ssim_m = AverageMeter(), AverageMeter()
    rows = []
    for names, degraded, clean in tqdm(loader, total=len(loader), desc="Evaluating", leave=False):
        degraded, clean = degraded.to(device), clean.to(device)
        restored = model(degraded)
        row = image_row(names[0], restored, clean)
        psnr_m.update(row['psnr'])
        ssim_m.update(row['ssim'])
        rows.append(row)
        if log is not None:
            log.info(f"{names[0]}: PSNR {row['psnr']:.3f} SSIM {row['ssim']:.4f}")
        stem = os.path.splitext(names[0])[0]
        if results_dir is not None and (save_only is None or stem in save_only):
            save_image_tensor(restored, os.path.join(results_dir, stem + ".png"))
    model.train(was_training)
    return summarize(rows, psnr_m.avg, ssim_m.avg), rows


def summarize(rows, psnr=None, ssim=None):
    def mean(k):
        v = [r[k] for r in rows if not (isinstance(r.get(k), float) and math.isnan(r[k]))]
        return sum(v) / len(v) if v else math.nan

    return {'n': len(rows), 'psnr': float(psnr) if psnr is not None else mean('psnr'),
            'ssim': float(ssim) if ssim is not None else mean('ssim'), 'psnr_y': mean('psnr_y')}


def write_rows(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def eval_loader(dataset, num_workers=0):
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
