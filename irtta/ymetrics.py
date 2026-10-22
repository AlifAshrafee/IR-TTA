"""Luminance PSNR in the deraining convention (MATLAB rgb2ycbcr / BasicSR ``test_y_channel``).

8-bit RGB -> Y = 16 + (65.481 R + 128.553 G + 24.966 B) / 255 in [16, 235]; PSNR with peak 255.
tools/metric_check.py showed this reproduces the co-author's Rain100H numbers for all 12 released checkpoints
(their other sets are RGB). Rain sets carry ``"metric": "y"`` in configs/datasets.json; tables report ``psnr_y``
for them and RGB PSNR elsewhere.
"""
import math

import torch


def y_bt601(img01):
    """(B, 3, H, W) in [0, 1] -> (B, H, W) float64 Y on 16..235 from the 8-bit-rounded image."""
    u8 = (img01.clamp(0, 1) * 255).round().double()
    return 16. + (65.481 * u8[:, 0] + 128.553 * u8[:, 1] + 24.966 * u8[:, 2]) / 255.


def psnr_y(out, gt):
    mse = float(((y_bt601(out) - y_bt601(gt)) ** 2).mean())
    return 10 * math.log10(255. ** 2 / max(mse, 1e-12))


__all__ = ['y_bt601', 'psnr_y', 'torch']
