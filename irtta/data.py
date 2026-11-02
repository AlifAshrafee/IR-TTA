"""Datasets: the registry (``configs/datasets.json``), target patches for adaptation, full-image evaluation sets,
synthetic-noise sets, the clean content bank and the real-pair bank of the oracle analysis.

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
import random
import zlib
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REGISTRY = os.path.join(REPO_ROOT, 'configs', 'datasets.json')
IMG_EXTS = ('.png', '.jpg', '.jpeg')
BANK_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')


# ---------------------------------------------------------------------------- #
# image folders: adaptation patches and full-image pairs
# ---------------------------------------------------------------------------- #


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
        s = self.patch_size
        if H < s or W < s:   # smaller image (e.g. some SOTS-Outdoor scenes): largest 16-aligned square that fits
            s = min(H // 16 * 16, W // 16 * 16)
            if s < 16:
                raise ValueError(f"image too small: {H}x{W}, patch={self.patch_size}")
        ind_H = random.randint(0, H - s)
        ind_W = random.randint(0, W - s)
        return img[ind_H:ind_H + s, ind_W:ind_W + s]

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

    def arrays(self, idx):
        """(name, noisy uint8 HxWx3, clean uint8 HxWx3): deterministic per (seed, name)."""
        name = self.names[idx]
        clean = crop_to_multiple(np.array(Image.open(os.path.join(self.gt_dir, name)).convert('RGB')), self.base)
        rs = np.random.RandomState((self.seed + zlib.crc32(name.encode())) % (2 ** 32))
        noisy = np.clip(clean.astype(np.float64) + rs.randn(*clean.shape) * self.sigma, 0, 255).astype(np.uint8)
        return name, noisy, clean

    def __getitem__(self, idx):
        name, noisy, clean = self.arrays(idx)
        return name, self.to_tensor(noisy), self.to_tensor(clean)


def materialize_synthetic(spec, base=16, cache_root=None):
    """A synthetic-noise spec (no LQ folder) -> the same pairs written once as PNGs (``cache/synthetic/<name>``),
    returned as an ordinary paired spec. Bit-identical to the on-the-fly images (same noise, same uint8 clip and
    centre crop), so evaluation numbers do not change; transductive adaptation needs stored LQ images. Safe when
    several runs create the same set at once (written to a temporary folder, then renamed)."""
    import dataclasses
    import shutil
    out = os.path.join(cache_root or os.environ.get('IRTTA_CACHE') or os.path.join(REPO_ROOT, 'cache', 'synthetic'),
                       spec.name)
    if not os.path.isdir(out):
        tmp = f"{out}.tmp{os.getpid()}"
        ds = SyntheticNoiseDataset(spec.gt, spec.noise_sigma, spec.noise_seed, base)
        for sub in ('LQ', 'GT'):
            os.makedirs(os.path.join(tmp, sub), exist_ok=True)
        for i in range(len(ds)):
            name, noisy, clean = ds.arrays(i)
            fn = os.path.splitext(name)[0] + '.png'
            Image.fromarray(noisy).save(os.path.join(tmp, 'LQ', fn))
            Image.fromarray(clean).save(os.path.join(tmp, 'GT', fn))
        try:
            os.rename(tmp, out)
        except OSError:   # another run finished first
            shutil.rmtree(tmp, ignore_errors=True)
    return dataclasses.replace(spec, lq=os.path.join(out, 'LQ'), gt=os.path.join(out, 'GT'), noise_sigma=None)


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


# ---------------------------------------------------------------------------- #
# Clean content bank for transfer TTA
# ---------------------------------------------------------------------------- #
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
            paths += [os.path.join(dirpath, f) for f in files if f.lower().endswith(BANK_EXTS)]
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


class PairBank:
    """Real (LQ, GT) training pairs held in RAM as uint8 (``max_images`` evenly spaced, same names in both folders);
    ``sample`` returns aligned random augmented crops. Oracle diagnostic for transfer TTA (``--oracle_pairs``)."""

    def __init__(self, spec, max_images=300, min_side=192, exclude=()):
        """``exclude``: image names never used (the evaluated target images when the oracle split and the target
        share a folder, e.g. HIDE_val targets with HIDE_train pairs)."""
        exclude = set(exclude)
        pool = [n for n in available_names(spec) if n not in exclude]
        names = evenly_spaced(pool, max_images if max_images and max_images > 0 else 10 ** 9)
        self.pairs = []
        for nm in names:
            lq = Image.open(os.path.join(spec.lq, nm)).convert('RGB')
            gt = Image.open(os.path.join(spec.gt, nm)).convert('RGB')
            if spec.max_side and max(lq.size) > spec.max_side:
                f = spec.max_side / max(lq.size)
                size = (max(1, round(lq.size[0] * f)), max(1, round(lq.size[1] * f)))
                lq, gt = lq.resize(size, Image.BICUBIC), gt.resize(size, Image.BICUBIC)
            a, b = np.array(lq), np.array(gt)
            if a.shape == b.shape and min(a.shape[:2]) >= min_side:
                self.pairs.append((a, b))
        if not self.pairs:
            raise ValueError(f"oracle pairs {spec.name}: no aligned pair with a shorter side >= {min_side}"
                             f" (after excluding {len(exclude)} evaluated target images)")
        self.name = spec.name

    def __len__(self):
        return len(self.pairs)

    def sample(self, n, size, generator):
        ys, cs = [], []
        for _ in range(n):
            a, b = self.pairs[int(torch.randint(len(self.pairs), (1,), generator=generator))]
            H, W = a.shape[:2]
            s = min(size, H, W)
            i = int(torch.randint(0, H - s + 1, (1,), generator=generator))
            j = int(torch.randint(0, W - s + 1, (1,), generator=generator))
            flip = float(torch.rand(1, generator=generator)) < 0.5
            rot = int(torch.randint(4, (1,), generator=generator))
            pair = []
            for img in (a, b):
                t = torch.from_numpy(np.ascontiguousarray(img[i:i + s, j:j + s])).permute(2, 0, 1).float() / 255.
                if s != size:
                    t = torch.nn.functional.interpolate(t[None], size=(size, size), mode='bicubic',
                                                        align_corners=False)[0].clamp(0, 1)
                if flip:
                    t = torch.flip(t, dims=[2])
                pair.append(torch.rot90(t, rot, dims=(1, 2)))
            ys.append(pair[0])
            cs.append(pair[1])
        return torch.stack(ys, 0), torch.stack(cs, 0)
