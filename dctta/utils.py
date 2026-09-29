"""Small helpers: seeding, checkpoint loading, dihedral transforms, image saving."""
import logging
import random

import numpy as np
import torch
from PIL import Image


def set_seed(seed=23):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_promptir_checkpoint(model, ckpt_path, logger=None):
    """Load a PromptIR Lightning checkpoint (keys prefixed with ``net.``) into ``model``.

    The original used ``strict=False`` silently; we keep ``strict=False`` (the Lightning
    checkpoint may carry loss-module keys) but report anything missing/unexpected.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = {k.replace("net.", "", 1) if k.startswith("net.") else k: v for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)
    log = (logger or logging.getLogger(__name__))
    if result.missing_keys:
        log.warning(f"checkpoint {ckpt_path}: {len(result.missing_keys)} missing keys, e.g. {result.missing_keys[:5]}")
    if result.unexpected_keys:
        log.warning(f"checkpoint {ckpt_path}: {len(result.unexpected_keys)} unexpected keys, e.g. {result.unexpected_keys[:5]}")
    return model


# ---------------------------------------------------------------------------- #
# Dihedral (flip / transpose) group used by the TIPS Fisher pass. Verbatim from
# utils/utils_tta.py (SRTTA).
# ---------------------------------------------------------------------------- #
def transform_tensor(img, op, undo=False):
    """img: BxCxHxW tensor; op: subset of 'v' (flip W), 'h' (flip H), 't' (transpose)."""
    if undo:
        if op.find('t') >= 0:
            img = img.permute((0, 1, 3, 2)).contiguous()
    if op.find('v') >= 0:
        img = torch.flip(img, dims=[3]).contiguous()
    if op.find('h') >= 0:
        img = torch.flip(img, dims=[2]).contiguous()
    if not undo:
        if op.find('t') >= 0:
            img = img.permute((0, 1, 3, 2)).contiguous()
    return img


def augment_transform(img_in, undo=False):
    """The 8 dihedral transforms of ``img_in`` and their op strings ('' is the identity)."""
    tran_ops = ['', 'v', 'h', 't', 'vh', 'vt', 'ht', 'vht']
    img_outs = [transform_tensor(img_in, op, undo) for op in tran_ops]
    return img_outs, tran_ops


# ---------------------------------------------------------------------------- #
# Geometric self-ensemble used for the pseudo-label (paper Eq. 1). The original used
# torchvision RandomHorizontalFlip(p=1) / RandomRotation((k*90, k*90)); on square
# patches nearest-neighbour rotation by an exact multiple of 90 degrees is pixel-exact,
# so torch.rot90 / torch.flip give the same tensors without the grid_sample overhead.
# ---------------------------------------------------------------------------- #
def hflip(x):
    return torch.flip(x, dims=[-1])


def rot(x, k):
    return torch.rot90(x, k, dims=(-2, -1))


def self_ensemble(fn, x, fuse='mean'):
    """8-way flip/rotation self-ensemble of ``fn`` (a restorer) applied to ``x``.

    Same eight members and the same summation order as the original code:
    flip+rot{0,90,180,270} then rot{0,90,180,270}.
    """
    outs = [
        hflip(fn(hflip(x))),
        hflip(rot(fn(rot(hflip(x), 1)), -1)),
        hflip(rot(fn(rot(hflip(x), 2)), -2)),
        hflip(rot(fn(rot(hflip(x), 3)), -3)),
        fn(x),
        rot(fn(rot(x, 1)), -1),
        rot(fn(rot(x, 2)), -2),
        rot(fn(rot(x, 3)), -3),
    ]
    if fuse == 'mean':
        return (outs[0] + outs[1] + outs[2] + outs[3] + outs[4] + outs[5] + outs[6] + outs[7]) / 8.
    elif fuse == 'median':
        return torch.stack(outs, 0).median(dim=0).values
    raise ValueError(fuse)


class SelfEnsemble(torch.nn.Module):
    """Inference-time wrapper: ``model`` -> 8-way self-ensembled ``model`` (the paper's "*" rows)."""

    def __init__(self, model, fuse='mean'):
        super().__init__()
        self.model = model
        self.fuse = fuse

    def forward(self, x):
        return self_ensemble(self.model, x, self.fuse)


# ---------------------------------------------------------------------------- #
def tensor_to_uint8(image_tensor):
    """1xCxHxW in [0,1] -> HxWxC uint8 (clip + scale, as utils/image_io.py did)."""
    arr = image_tensor.detach().cpu().numpy()[0]
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)


def save_image_tensor(image_tensor, path):
    Image.fromarray(tensor_to_uint8(image_tensor)).save(path)
