"""Datasets for adaptation (random patches) and evaluation (full images).

Both were the ``*_by tn`` additions in the original ``utils/dataset_utils.py``
(``PromptTrainDataset_Simple`` and ``PairedImageDataset``); the PromptIR training
datasets, the degradation synthesiser (``Degradation`` / ``degradation_utils.py``)
and the four other dataset classes were never used by the TTA pipeline.
"""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

IMG_EXTS = ('.png', '.jpg', '.jpeg')


def list_images(d):
    return sorted([f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)])


def load_rgb(path, max_side=None):
    """Load an image as HxWx3 uint8. If ``max_side`` is set and the longer side exceeds it, downsample
    (bicubic, aspect ratio kept) so the longer side equals ``max_side``. LQ and GT of a pair have the
    same size, so they are resized identically. ``max_side=None`` is the original behaviour."""
    img = Image.open(path).convert('RGB')
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
    return np.array(img)


def crop_to_multiple(image, base=16):
    """Centre-crop an HxWxC array to a multiple of ``base`` (PromptIR's ``crop_img``)."""
    h, w = image.shape[0], image.shape[1]
    crop_h, crop_w = h % base, w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]


class AdaptationPatchDataset(Dataset):
    """Random ``patch_size`` crops of the degraded target images.

    Returns ``(filename, degraded_patch, clean_patch)`` for API compatibility with the original
    loop, but note: the clean patch is **never used** by the adaptation (it is only moved to the
    GPU and discarded), and in the original the degraded and clean crops were drawn with two
    independent ``randint`` calls, so they are not even aligned. Pass ``gt_dir=None`` to skip GT.

    ``seed``: the original constructor re-seeded ``random``/``numpy``/``torch`` **globally** with
    42 (after the script had already seeded everything with 23). This changes the RNG stream that
    initialises the RDDM generator, so it is kept (``seed=42``) for reproducibility of the
    reference runs; pass ``seed=None`` to disable the side effect.
    """

    def __init__(self, lq_dir, gt_dir=None, patch_size=320, seed=42, max_side=None, names=None):
        super().__init__()
        self.lq_dir, self.gt_dir, self.patch_size = lq_dir, gt_dir, patch_size
        self.max_side = max_side
        self.names = names  # optional fixed subset of LQ file names (GT matched by the same names)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.lq_list = list_images(lq_dir) if names is None else list(names)
        self.gt_list = (list_images(gt_dir) if names is None else list(names)) if gt_dir is not None else None
        if self.gt_list is not None:
            assert len(self.lq_list) == len(self.gt_list), \
                f"LQ ({len(self.lq_list)}) and GT ({len(self.gt_list)}) counts differ"
        self.to_tensor = ToTensor()

    def _crop_patch(self, img):
        H, W = img.shape[:2]
        if H < self.patch_size or W < self.patch_size:
            raise ValueError(f"image too small: {H}x{W}, patch={self.patch_size}")
        ind_H = random.randint(0, H - self.patch_size)
        ind_W = random.randint(0, W - self.patch_size)
        return img[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]

    def __len__(self):
        return len(self.lq_list)

    def __getitem__(self, idx):
        filename = self.lq_list[idx]
        degrad_img = load_rgb(os.path.join(self.lq_dir, filename), self.max_side)
        degrad_patch = self.to_tensor(self._crop_patch(degrad_img))
        if self.gt_list is None:
            return filename, degrad_patch, torch.zeros(0)
        clean_img = load_rgb(os.path.join(self.gt_dir, self.gt_list[idx]), self.max_side)
        clean_patch = self.to_tensor(self._crop_patch(clean_img))
        return filename, degrad_patch, clean_patch


class PairedImageDataset(Dataset):
    """Full-resolution (LQ, GT) pairs matched by file name, centre-cropped to a multiple of ``base``."""

    def __init__(self, lq_dir, gt_dir, base=16, names=None, max_side=None):
        super().__init__()
        self.lq_dir, self.gt_dir, self.base = lq_dir, gt_dir, base
        self.max_side = max_side
        lq_names, gt_names = list_images(lq_dir), set(list_images(gt_dir))
        self.names = [n for n in lq_names if n in gt_names]
        if names is not None:  # restrict to a given list (probe / subset), keeping the given order
            available = set(self.names)
            missing = [n for n in names if n not in available]
            if missing:
                raise ValueError(f"{len(missing)} requested images not found in {lq_dir}, e.g. {missing[:3]}")
            self.names = list(names)
        if not self.names:
            raise ValueError("no paired images: LQ and GT file names must match")
        self.to_tensor = ToTensor()

    def __getitem__(self, idx):
        name = self.names[idx]
        lq = crop_to_multiple(load_rgb(os.path.join(self.lq_dir, name), self.max_side), self.base)
        gt = crop_to_multiple(load_rgb(os.path.join(self.gt_dir, name), self.max_side), self.base)
        return name, self.to_tensor(lq), self.to_tensor(gt)

    def __len__(self):
        return len(self.names)
