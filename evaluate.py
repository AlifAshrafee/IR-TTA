"""Full-image evaluation shared by adapt.py and test.py."""
import os

import torch
from tqdm import tqdm

from dctta.metrics import AverageMeter, compute_psnr_ssim
from dctta.utils import save_image_tensor


@torch.no_grad()
def evaluate(model, loader, device, results_dir=None, logger=None):
    model.eval()
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
    psnr_meter, ssim_meter = AverageMeter(), AverageMeter()
    for names, degraded, clean in tqdm(loader, total=len(loader), desc="Evaluating"):
        degraded, clean = degraded.to(device), clean.to(device)
        restored = model(degraded)
        psnr, ssim = compute_psnr_ssim(restored, clean)
        psnr_meter.update(psnr)
        ssim_meter.update(ssim)
        if logger is not None:
            logger.info(f"{names[0]}: PSNR {psnr:.3f} SSIM {ssim:.4f}")
        if results_dir is not None:
            save_image_tensor(restored, os.path.join(results_dir, os.path.splitext(names[0])[0] + ".png"))
    return psnr_meter.avg, ssim_meter.avg
