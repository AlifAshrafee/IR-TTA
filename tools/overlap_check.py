"""Near-duplicate check between two image folders (e.g. a test set vs a source training set).

    python tools/overlap_check.py <test GT folder> <training GT folder> [--thr 0.98]

Each image -> 32x32 grey thumbnail, zero mean / unit norm; for every test image the most correlated training image
is reported. Correlation > thr (default 0.98) = same scene (re-encoding, small crops and exposure changes survive).
"""
import argparse
import os

import numpy as np
from PIL import Image

EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')


def thumbs(folder):
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))
    out = []
    for n in names:
        t = np.asarray(Image.open(os.path.join(folder, n)).convert('L').resize((32, 32), Image.BILINEAR), np.float64)
        t = t - t.mean()
        out.append(t.ravel() / (np.linalg.norm(t) + 1e-9))
    return names, np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('test')
    ap.add_argument('train')
    ap.add_argument('--thr', type=float, default=0.98)
    a = ap.parse_args()
    tn, tv = thumbs(a.test)
    rn, rv = thumbs(a.train)
    corr = tv @ rv.T
    best = corr.argmax(1)
    hits = [(tn[i], rn[best[i]], corr[i, best[i]]) for i in range(len(tn)) if corr[i, best[i]] > a.thr]
    print(f"{len(hits)} of {len(tn)} test images have a near-duplicate (corr > {a.thr}) among {len(rn)} training images")
    for t, r, c in hits[:20]:
        print(f"  {t}  ~  {r}  ({c:.3f})")


if __name__ == '__main__':
    main()
