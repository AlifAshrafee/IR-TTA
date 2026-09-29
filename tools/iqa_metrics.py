"""Optional: full-reference and no-reference IQA on a folder of restored images (was ``test_metrics.py``).

Not part of the adaptation pipeline; requires ``pip install pyiqa``. Hard-coded paths were replaced
by arguments and the ``basicsr.img2tensor`` dependency by a two-line conversion.

    python tools/iqa_metrics.py --sr results/Rain100H_DCTTA --gt testdata/Rain100H/GT
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

NO_REFERENCE = {'niqe', 'musiq', 'clipiqa', 'maniqa-pipal'}


def crop_to_multiple(image, base=16):
    h, w = image.shape[:2]
    crop_h, crop_w = h % base, w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]


def to_tensor(img, device):
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sr', required=True, help='folder of restored images')
    p.add_argument('--gt', required=True, help='folder of ground-truth images (same file names)')
    p.add_argument('--metrics', nargs='+', default=['psnr', 'ssim', 'lpips', 'dists', 'niqe', 'musiq', 'clipiqa'])
    p.add_argument('--crop_base', type=int, default=16)
    args = p.parse_args()

    import pyiqa  # noqa: E402  (optional dependency)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = {}
    for m in args.metrics:
        if m in ('psnr', 'ssim'):
            metrics[m] = pyiqa.create_metric(m, test_y_channel=False, color_space='rgb').to(device)
        else:
            metrics[m] = pyiqa.create_metric(m, device=device)

    sr_paths = sorted(glob.glob(os.path.join(args.sr, '*.[pj][np]g')) + glob.glob(os.path.join(args.sr, '*.jpeg')))
    acc = {m: 0.0 for m in metrics}
    n = 0
    for sr_path in sr_paths:
        name = os.path.splitext(os.path.basename(sr_path))[0]
        gt_candidates = glob.glob(os.path.join(args.gt, name + '.*'))
        if not gt_candidates:
            print(f"skip {name}: no GT")
            continue
        sr = to_tensor(crop_to_multiple(np.array(Image.open(sr_path).convert('RGB')), args.crop_base), device)
        gt = to_tensor(crop_to_multiple(np.array(Image.open(gt_candidates[0]).convert('RGB')), args.crop_base), device)
        with torch.no_grad():
            for m, fn in metrics.items():
                acc[m] += (fn(sr) if m in NO_REFERENCE else fn(sr, gt)).item()
        n += 1

    print(f"{n} images")
    for m in metrics:
        print(f"{m:>12s}: {acc[m] / max(n, 1):.4f}")


if __name__ == '__main__':
    main()
