"""DPDD: give each GT (all-in-focus) image its LQ (defocused) partner's file name, so the loader pairs them by name.

DPDD names are camera frame numbers: the two exposures of a scene are consecutive shots, so the names differ by 1-2.
Pairing = sorted order, verified per pair: frame numbers within 2 and the 32x32 grey thumbnails correlated.

    python tools/dpdd_pair.py <LQ dir> <GT dir>            # dry run: prints the pairs and any warning
    python tools/dpdd_pair.py <LQ dir> <GT dir> --apply    # renames the GT files (two-phase, no collisions)
"""
import argparse
import os
import re

import numpy as np
from PIL import Image


def thumb(path):
    t = np.asarray(Image.open(path).convert('L').resize((32, 32), Image.BILINEAR), np.float64)
    t = t - t.mean()
    return t.ravel() / (np.linalg.norm(t) + 1e-9)


def frame(name):
    m = re.search(r'(\d+)\D*$', os.path.splitext(name)[0])
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lq')
    ap.add_argument('gt')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--min_corr', type=float, default=0.9)
    args = ap.parse_args()
    ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    lq = sorted(f for f in os.listdir(args.lq) if f.lower().endswith(ext))
    gt = sorted(f for f in os.listdir(args.gt) if f.lower().endswith(ext))
    if len(lq) != len(gt):
        raise SystemExit(f"different counts: {len(lq)} LQ vs {len(gt)} GT")
    bad = 0
    for a, b in zip(lq, gt):
        corr = float(thumb(os.path.join(args.lq, a)) @ thumb(os.path.join(args.gt, b)))
        gap = abs(frame(a) - frame(b)) if frame(a) is not None and frame(b) is not None else None
        ok = corr >= args.min_corr and gap is not None and gap <= 2
        bad += not ok
        print(f"{'  ' if ok else '!!'} {b} -> {a}   frame gap {gap}, thumbnail corr {corr:.3f}")
    if bad:
        raise SystemExit(f"{bad} suspicious pair(s) (!!): nothing renamed; send me this output")
    print(f"all {len(lq)} pairs consistent")
    if not args.apply:
        print("dry run; add --apply to rename the GT files")
        return
    for b in gt:   # phase 1: temporary names, so no rename can overwrite a not-yet-renamed file
        os.rename(os.path.join(args.gt, b), os.path.join(args.gt, b + '.tmp_pair'))
    for a, b in zip(lq, gt):
        os.rename(os.path.join(args.gt, b + '.tmp_pair'), os.path.join(args.gt, a))
    print(f"renamed {len(gt)} GT files to their LQ names")


if __name__ == '__main__':
    main()
