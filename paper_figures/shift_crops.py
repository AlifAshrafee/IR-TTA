"""Fig. 1 (a): what a degradation shift looks like. Exports example images of each source (training) degradation
next to its shifted test degradation, for the illustration (no model, no GPU, no numbers).

    python paper_figures/shift_crops.py                       # all pairs below, 6 images per set
    python paper_figures/shift_crops.py --pairs Rain100L:Rain100H SOTS_Outdoor:OHAZY --n 10

Writes results/figures/shift/<source>_to_<target>/<set>_<k>.png (degraded images, longer side --max_side px,
evenly spaced through the set) and a contact sheet results/figures/shift/contact_<source>_to_<target>.png with
the source row above the target row, so you can pick one image per set. --gt also saves each image's clean
reference (same name, _gt suffix); --crop S exports a random S x S crop of each image instead of the whole image.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

sys.path.insert(0, S.REPO)

PAIRS = ['Rain100L:Rain100H', 'SOTS_Outdoor:OHAZY', 'SOTS_Outdoor:DenseHaze', 'BSD68_s25:Kodak24_Poisson',
         'GoPro:HIDE', 'GoPro:DPDD', 'LOL:LOLv2_Syn']
SHIFT = {'Rain100L:Rain100H': 'rain: light → heavy', 'SOTS_Outdoor:OHAZY': 'haze: synthetic → real',
         'SOTS_Outdoor:DenseHaze': 'haze: synthetic → dense real', 'BSD68_s25:Kodak24_Poisson': 'noise: Gaussian → Poisson',
         'GoPro:HIDE': 'blur: GoPro → HIDE', 'GoPro:DPDD': 'blur: motion → defocus', 'LOL:LOLv2_Syn': 'low light: real → synthetic'}


def images(name, n, rng, crop=None):
    from irtta.data import available_names, evenly_spaced, load_registry, make_eval_dataset, resolve_spec
    spec = resolve_spec(name, load_registry())
    names = evenly_spaced(available_names(spec), n)
    ds = make_eval_dataset(spec, base=16, names=names)
    out = []
    for i in range(len(ds)):
        stem, lq, gt = ds[i]
        lq = (lq.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        gt = (gt.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        if crop:
            H, W = lq.shape[:2]
            s = min(crop, H, W)
            y, x = rng.integers(0, H - s + 1), rng.integers(0, W - s + 1)
            lq, gt = lq[y:y + s, x:x + s], gt[y:y + s, x:x + s]
        out.append((os.path.splitext(stem)[0], Image.fromarray(lq), Image.fromarray(gt)))
    return out


def fit(im, max_side):
    im = im.copy()
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pairs', nargs='+', default=PAIRS, help='source_set:target_set')
    p.add_argument('--n', type=int, default=6, help='images per set')
    p.add_argument('--max_side', type=int, default=800)
    p.add_argument('--crop', type=int, default=0, help='export random square crops of this size instead')
    p.add_argument('--gt', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=os.path.join(S.REPO, 'results', 'figures', 'shift'), help=argparse.SUPPRESS)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    for pair in a.pairs:
        src, tgt = pair.split(':')
        od = os.path.join(a.out, f'{src}_to_{tgt}')
        os.makedirs(od, exist_ok=True)
        rows = []
        for name in (src, tgt):
            ims = images(name, a.n, rng, a.crop or None)
            h = 180
            thumbs = []
            for k, (stem, lq, gt) in enumerate(ims):
                fit(lq, a.max_side).save(os.path.join(od, f'{name}_{k}.png'))
                if a.gt:
                    fit(gt, a.max_side).save(os.path.join(od, f'{name}_{k}_gt.png'))
                thumbs.append((f'{name}_{k} ({stem})', lq.resize((int(lq.size[0] * h / lq.size[1]), h), Image.LANCZOS)))
            row = Image.new('RGB', (sum(t.size[0] + 4 for _, t in thumbs), h + 16), 'white')
            d, x = ImageDraw.Draw(row), 0
            for lab, t in thumbs:
                row.paste(t, (x, 0))
                d.text((x + 2, h + 2), lab, fill=(0, 0, 0))
                x += t.size[0] + 4
            rows.append(row)
        sheet = Image.new('RGB', (max(r.size[0] for r in rows), sum(r.size[1] + 8 for r in rows)), 'white')
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.size[1] + 8
        cs = os.path.join(a.out, f'contact_{src}_to_{tgt}.png')
        sheet.save(cs)
        print(f"[shift] {SHIFT.get(pair, pair)}: {a.n} + {a.n} images -> {od}/, contact sheet {cs}")


if __name__ == '__main__':
    main()
