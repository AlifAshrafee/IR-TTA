"""Dataset registry, evaluation datasets (incl. synthetic-noise source sets) and patch cropping.

Datasets are referred to by name through ``configs/datasets.json``:

    "Rain100H":     {"lq": "{root}/Test/derain/Rain100H/LQ", "gt": "{root}/Test/derain/Rain100H/GT",
                     "split": "test", "task": "derain"}
    "BSD68_s25":    {"gt": "{root}/Test/denoise/BSD68", "noise_sigma": 25, "split": "test", "task": "denoise"}

A set without ``lq`` but with ``noise_sigma`` gets its degraded images synthesised deterministically
(per-image seed = ``noise_seed`` + crc32(file name)), PromptIR-style:
``clip(gt + N(0, sigma^2), 0, 255).astype(uint8)`` on the 0-255 scale.
"""
import json
import os
import zlib
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from dctta.data import PairedImageDataset, crop_to_multiple, list_images

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REGISTRY = os.path.join(REPO_ROOT, 'configs', 'datasets.json')


@dataclass
class DataSpec:
    name: str
    lq: Optional[str]
    gt: str
    noise_sigma: Optional[float] = None
    noise_seed: int = 0
    split: str = 'test'
    task: str = ''
    max_side: Optional[int] = None
    extra: dict = field(default_factory=dict)

    @property
    def synthetic(self):
        return self.lq is None and self.noise_sigma is not None


def load_registry(path=DEFAULT_REGISTRY):
    with open(path) as f:
        reg = json.load(f)
    root = reg.get('_root', '')
    if not os.path.isabs(root):  # relative roots are relative to the repo, not the CWD
        root = os.path.normpath(os.path.join(REPO_ROOT, root))
    reg['_root_resolved'] = root
    return reg


def _expand(p, root):
    if p is None:
        return None
    p = p.replace('{root}', root)
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(REPO_ROOT, p))


def resolve_spec(name, registry=None):
    reg = registry if registry is not None else load_registry()
    if name not in reg or name.startswith('_'):
        known = sorted(k for k in reg if not k.startswith('_'))
        raise KeyError(f"dataset '{name}' not in registry; known: {known}")
    e = reg[name]
    root = reg['_root_resolved']
    known_keys = {'lq', 'gt', 'noise_sigma', 'noise_seed', 'split', 'task', 'max_side'}  # others -> extra (e.g. metric)
    return DataSpec(name=name, lq=_expand(e.get('lq'), root), gt=_expand(e['gt'], root),
                    noise_sigma=e.get('noise_sigma'), noise_seed=e.get('noise_seed', 0),
                    split=e.get('split', 'test'), task=e.get('task', ''), max_side=e.get('max_side'),
                    extra={k: v for k, v in e.items() if k not in known_keys})


def spec_from_dirs(lq_dir, gt_dir, name=None):
    name = name or os.path.basename(os.path.dirname(os.path.normpath(lq_dir))) or 'custom'
    return DataSpec(name=name, lq=lq_dir, gt=gt_dir)


def source_sets(source_tag, registry=None):
    """Names of the source-task in-distribution test sets for a source setting ('3task' / '5task')."""
    reg = registry if registry is not None else load_registry()
    table = reg.get('_source_sets', {})
    if source_tag not in table:
        raise KeyError(f"no '_source_sets' entry for source '{source_tag}' in the registry")
    return list(table[source_tag])


def evenly_spaced(names, n):
    """Deterministic, order-preserving subset of ``n`` names spread over the (sorted) list."""
    if n is None or n <= 0 or n >= len(names):
        return list(names)
    idx = np.unique(np.linspace(0, len(names) - 1, n).round().astype(int))
    return [names[i] for i in idx]


class SyntheticNoiseDataset(Dataset):
    """(name, noisy, clean) from a folder of clean images, Gaussian noise of std ``sigma`` (0-255 scale)."""

    def __init__(self, gt_dir, sigma, seed=0, base=16, names=None):
        super().__init__()
        self.gt_dir, self.sigma, self.seed, self.base = gt_dir, float(sigma), int(seed), base
        all_names = list_images(gt_dir)
        self.names = list(names) if names is not None else all_names
        missing = set(self.names) - set(all_names)
        if missing:
            raise ValueError(f"{len(missing)} requested images not in {gt_dir}")
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        clean = crop_to_multiple(np.array(Image.open(os.path.join(self.gt_dir, name)).convert('RGB')), self.base)
        rs = np.random.RandomState((self.seed + zlib.crc32(name.encode())) % (2 ** 32))
        noisy = np.clip(clean.astype(np.float64) + rs.randn(*clean.shape) * self.sigma, 0, 255).astype(np.uint8)
        return name, self.to_tensor(noisy), self.to_tensor(clean)


def available_names(spec: DataSpec) -> List[str]:
    if spec.synthetic:
        return list_images(spec.gt)
    gt = set(list_images(spec.gt))
    return [n for n in list_images(spec.lq) if n in gt]


def make_eval_dataset(spec: DataSpec, base=16, names=None, max_images=None):
    """Full-image (name, lq, gt) dataset; ``max_images`` takes an evenly spaced deterministic subset."""
    if names is None and max_images:
        names = evenly_spaced(available_names(spec), max_images)
    if spec.synthetic:
        return SyntheticNoiseDataset(spec.gt, spec.noise_sigma, spec.noise_seed, base, names)
    if spec.lq is None:
        raise ValueError(f"dataset {spec.name} has neither 'lq' nor 'noise_sigma'")
    return PairedImageDataset(spec.lq, spec.gt, base=base, names=names, max_side=spec.max_side)


def random_crops(img, size, n, generator, multiple=16):
    """``n`` random ``size``x``size`` crops of a 1xCxHxW tensor (positions from ``generator``, a CPU
    ``torch.Generator``, so the global RNG is untouched). If the image is smaller than ``size`` the
    crop is the largest ``multiple``-aligned square that fits."""
    _, _, H, W = img.shape
    s = min(size, (H // multiple) * multiple, (W // multiple) * multiple)
    if s <= 0:
        raise ValueError(f"image {H}x{W} too small for crops aligned to {multiple}")
    crops = []
    for _ in range(n):
        i = int(torch.randint(0, H - s + 1, (1,), generator=generator))
        j = int(torch.randint(0, W - s + 1, (1,), generator=generator))
        crops.append(img[..., i:i + s, j:j + s])
    return torch.cat(crops, 0)


# ---------------------------------------------------------------------------- #
# Clean content bank for transfer TTA
# ---------------------------------------------------------------------------- #
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


def resolve_bank(bank=None, registry=None):
    """``bank``: a folder path, a registry key under ``_banks``, or None for ``_banks['default']``."""
    reg = registry if registry is not None else load_registry()
    banks = reg.get('_banks', {})
    key = bank or 'default'
    if key in banks:
        return _expand(banks[key], reg['_root_resolved'])
    if bank is None:
        raise KeyError("no '_banks': {'default': ...} entry in the registry and no --bank given")
    return _expand(bank, reg['_root_resolved'])


class CleanBank:
    """Clean images (searched recursively) held in RAM as uint8; ``sample`` returns random augmented crops.

    ``max_images`` evenly spaced images are kept (sorted by path) so the bank is deterministic; images whose
    shorter side is below ``min_side`` are skipped. Crops use a caller-supplied CPU ``torch.Generator``.
    """

    def __init__(self, root, max_images=300, min_side=192):
        paths = []
        for dirpath, _, files in os.walk(root):
            paths += [os.path.join(dirpath, f) for f in files if f.lower().endswith(IMG_EXTS)]
        paths = sorted(paths)
        if not paths:
            raise ValueError(f"clean bank {root} contains no images (set _banks.default in configs/datasets.json "
                             f"to your DIV2K_train_HR folder, or pass --bank)")
        self.images = []
        for p in evenly_spaced(paths, max_images if max_images and max_images > 0 else len(paths)):
            img = np.array(Image.open(p).convert('RGB'))
            if min(img.shape[:2]) >= min_side:
                self.images.append(img)
        if not self.images:
            raise ValueError(f"no image in {root} has a shorter side >= {min_side}")
        self.root = root

    def __len__(self):
        return len(self.images)

    def sample(self, n, size, generator):
        """(n, 3, size, size) float tensor in [0, 1]: random image, crop, flip and 90-degree rotation."""
        out = []
        for _ in range(n):
            img = self.images[int(torch.randint(len(self.images), (1,), generator=generator))]
            H, W = img.shape[:2]
            s = min(size, H, W)
            i = int(torch.randint(0, H - s + 1, (1,), generator=generator))
            j = int(torch.randint(0, W - s + 1, (1,), generator=generator))
            t = torch.from_numpy(np.ascontiguousarray(img[i:i + s, j:j + s])).permute(2, 0, 1).float() / 255.
            if s != size:
                t = torch.nn.functional.interpolate(t[None], size=(size, size), mode='bicubic',
                                                    align_corners=False)[0].clamp(0, 1)
            if float(torch.rand(1, generator=generator)) < 0.5:
                t = torch.flip(t, dims=[2])
            t = torch.rot90(t, int(torch.randint(4, (1,), generator=generator)), dims=(1, 2))
            out.append(t)
        return torch.stack(out, 0)
