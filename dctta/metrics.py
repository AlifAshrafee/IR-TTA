"""PSNR / SSIM exactly as the original ``utils/val_utils.py`` computed them (scikit-image, RGB,
data_range=1, no Y-channel conversion). Keep this for comparability with the paper's numbers."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_psnr_ssim(recovered, clean):
    assert recovered.shape == clean.shape
    recovered = np.clip(recovered.detach().cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1).transpose(0, 2, 3, 1)
    psnr = ssim = 0.0
    for i in range(recovered.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recovered[i], data_range=1)
        ssim += structural_similarity(clean[i], recovered[i], data_range=1, channel_axis=-1)
    return psnr / recovered.shape[0], ssim / recovered.shape[0]
