"""Fig. 1 (left): image outputs of ONE test case for the problem-vs-remedy illustration (laid out in Lucidchart).

Three steps; `GPU=0 bash scripts/run.sh figures` runs all three for FIG_BB / FIG_S / FIG_SET (default: AdaIR,
5-task, SOTS-Indoor).

  1. select   rank the images of one test set by how clearly ours beats both the pretrained model and existing
              TTA, from the per-image PSNRs the tables' runs already saved (no GPU, nothing re-run):
                python paper_figures/qualitative.py select --backbone adair --setting 5task --target SOTS_Indoor
              -> prints the candidates; writes their names to results/figures/qualitative/<bb>_<setting>_<target>.txt
  2. re-run   pretrained, existing TTA and ours on that test set, saving only those images (run.sh figures does
              this; the numbers are the same runs as in the tables, written to results/figure_runs)
  3. export   one folder of PNGs per candidate image, plus a contact sheet to choose from:
                python paper_figures/qualitative.py export --backbone adair --setting 5task --target SOTS_Indoor
              -> results/figures/qualitative/<target>/<image>/
                   input.png  pretrained.png  existing_tta.png  ours.png  gt.png        full images
                   *_box.png                                                       the same with the zoom box
                   *_zoom.png                                                      the zoom region, 512 px
                   psnr.json                                                       PSNR of every image
                 results/figures/qualitative/contact_<target>.png                  all candidates, one row each

Arguments:
  --backbone   promptir | adair | bioir | moceir | mirage | r2r
  --setting    3task | 5task                (which pretrained model)
  --target     the test set, e.g. SOTS_Indoor, OHAZY, DenseHaze, DPDD, Kodak24_Poisson, LOLv2_Syn, HIDE
  --image      (export only) export just this image (its file name, with or without extension)
  --box X,Y,SIZE  (export only, with --image) set the zoom square by hand, in pixels; default: the square where
               ours removes the most error relative to the pretrained model
"""
import argparse
import csv
import json
import os
import statistics
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S  # noqa: E402

sys.path.insert(0, S.REPO)
sys.path.insert(0, os.path.join(S.REPO, 'tools'))
from table import Y_TARGETS, num, row_key  # noqa: E402

RUNS = os.path.join(S.REPO, 'results', 'suite')           # the tables' runs (per-image PSNRs)
FIG_RUNS = os.path.join(S.REPO, 'results', 'figure_runs')  # the same runs, re-run with --save_images
OUT = os.path.join(S.REPO, 'results', 'figures', 'qualitative')
COLS = ['input', 'pretrained', 'existing_tta', 'ours', 'gt']
ROWKEY = {'source': 'pretrained', 'dctta': 'existing_tta', 'ours': 'ours'}
BOX_COLOR = (255, 204, 0)


def find_runs(root, backbone, setting, target, seed=23):
    """{'pretrained' | 'existing_tta' | 'ours': run folder} of one table cell (tools/table.py's row rules)."""
    out = {}
    for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        rp = os.path.join(root, d, 'result.json')
        if not os.path.exists(rp):
            continue
        with open(rp) as f:
            r = {k: ('' if v is None else v) for k, v in json.load(f).items()}
        if ((r.get('backbone') or 'promptir'), r.get('source'), r.get('target')) != (backbone, setting, target):
            continue
        if int(r.get('seed') or 0) != seed:
            continue
        k = row_key(r, 'F1')
        if k in ROWKEY:
            out[ROWKEY[k]] = os.path.join(root, d)
    return out


def per_image(run_dir, target):
    """{image stem: PSNR} (Y-PSNR on rain sets, as in the tables)."""
    key = 'psnr_y' if target in Y_TARGETS else 'psnr'
    with open(os.path.join(run_dir, 'per_image.csv')) as f:
        return {os.path.splitext(r['name'])[0]: num(r[key]) for r in csv.DictReader(f)}


def names_file(a):
    return os.path.join(OUT, f"{a.backbone}_{a.setting}_{a.target}.txt")


# ---------------------------------------------------------------------------- #
def cmd_select(a):
    runs = find_runs(RUNS, a.backbone, a.setting, a.target)
    miss = [c for c in ROWKEY.values() if c not in runs]
    if miss:
        raise SystemExit(f"[select] {a.backbone} {a.setting} {a.target}: no finished {miss} run in {RUNS}")
    p = {c: per_image(d, a.target) for c, d in runs.items()}
    rows = []
    for s in p['ours']:
        v = {c: p[c].get(s) for c in p}
        if None not in v.values():
            rows.append((min(v['ours'] - v['pretrained'], v['ours'] - v['existing_tta']), s, v))
    rows.sort(key=lambda r: -r[0])
    med = statistics.median(v['ours'] - v['pretrained'] for _, _, v in rows)
    typical = min(rows, key=lambda r: abs(r[2]['ours'] - r[2]['pretrained'] - med))
    chosen = rows[:a.k] + ([typical] if typical not in rows[:a.k] else [])
    os.makedirs(OUT, exist_ok=True)
    with open(names_file(a), 'w') as f:
        f.write('\n'.join(s for _, s, _ in chosen) + '\n')
    print(f"[select] {a.target}: {len(rows)} images; median gain of ours over pretrained {med:+.2f} dB. "
          f"Candidates (largest margin over both first; last = the median-gain image):")
    print(f"  {'image':<26} {'pretrained':>10} {'existing':>9} {'ours':>7}")
    for _, s, v in chosen:
        print(f"  {s:<26} {v['pretrained']:10.2f} {v['existing_tta']:9.2f} {v['ours']:7.2f}")
    print(f"-> {names_file(a)}")


# ---------------------------------------------------------------------------- #
def load_pair(target, stem):
    """(input, GT) as uint8 HxWx3, exactly as evaluated (centre crop to x16, the set's max_side)."""
    from irtta.data import available_names, load_registry, make_eval_dataset, resolve_spec
    spec = resolve_spec(target, load_registry())
    name = next(n for n in available_names(spec) if os.path.splitext(n)[0] == stem)
    _, lq, gt = make_eval_dataset(spec, base=16, names=[name])[0]
    to8 = lambda t: (t.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)  # noqa: E731
    return to8(lq), to8(gt)


def psnr_u8(x, y, luma=False):
    x, y = x.astype(np.float64), y.astype(np.float64)
    if luma:   # BT.601 Y, as irtta/evaluation.py
        w = np.array([65.481, 128.553, 24.966]) / 255.
        x, y = 16 + x @ w, 16 + y @ w
    return 10 * np.log10(255. ** 2 / max(float(((x - y) ** 2).mean()), 1e-12))


def auto_box(pre, ours, gt, frac=0.2):
    """(x, y, size): the square where ours removes the most error relative to the pretrained model."""
    e = (np.abs(pre.astype(np.float32) - gt) - np.abs(ours.astype(np.float32) - gt)).mean(2)
    H, W = e.shape
    s = max(16, int(round(min(H, W) * frac)))
    ii = np.pad(e.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    tot = ii[s:, s:] - ii[:-s, s:] - ii[s:, :-s] + ii[:-s, :-s]
    m = max(2, s // 8)
    tot[:m], tot[-m:], tot[:, :m], tot[:, -m:] = -np.inf, -np.inf, -np.inf, -np.inf
    y, x = np.unravel_index(np.argmax(tot), tot.shape)
    return int(x), int(y), s


def export_one(a, runs, stem, box=None):
    lq, gt = load_pair(a.target, stem)
    imgs = {'input': lq, 'gt': gt}
    for c in ('pretrained', 'existing_tta', 'ours'):
        p = os.path.join(runs[c], 'images', stem + '.png')
        if not os.path.exists(p):
            print(f"[export] {stem}: {c} image not saved by the re-run (run.sh figures); skipped")
            return None
        imgs[c] = np.asarray(Image.open(p).convert('RGB'))
    bx, by, bs = box or auto_box(imgs['pretrained'], imgs['ours'], gt)
    od = os.path.join(OUT, a.target, stem)
    os.makedirs(od, exist_ok=True)
    psnr = {c: per_image(runs[c], a.target).get(stem) for c in ('pretrained', 'existing_tta', 'ours')}
    psnr['input'] = psnr_u8(lq, gt, a.target in Y_TARGETS)
    lw = max(2, int(round(gt.shape[1] / 250)))
    boxed = {}
    for c in COLS:
        im = Image.fromarray(imgs[c])
        im.save(os.path.join(od, f'{c}.png'))
        b = im.copy()
        ImageDraw.Draw(b).rectangle([bx, by, bx + bs - 1, by + bs - 1], outline=BOX_COLOR, width=lw)
        b.save(os.path.join(od, f'{c}_box.png'))
        im.crop((bx, by, bx + bs, by + bs)).resize((512, 512), Image.LANCZOS).save(os.path.join(od, f'{c}_zoom.png'))
        boxed[c] = b
    with open(os.path.join(od, 'psnr.json'), 'w') as f:
        json.dump({'metric': 'Y-PSNR' if a.target in Y_TARGETS else 'RGB PSNR', 'box_x_y_size': [bx, by, bs],
                   **{c: round(v, 2) for c, v in psnr.items() if v is not None}}, f, indent=1)
    return boxed, psnr


def cmd_export(a):
    runs = find_runs(FIG_RUNS, a.backbone, a.setting, a.target)
    miss = [c for c in ROWKEY.values() if c not in runs]
    if miss:
        raise SystemExit(f"[export] no {miss} re-run with saved images in {FIG_RUNS}: run `bash scripts/run.sh figures`")
    if a.image:
        stems = [os.path.splitext(a.image)[0]]
    else:
        with open(names_file(a)) as f:
            stems = [s.strip() for s in f if s.strip()]
    box = tuple(int(v) for v in a.box.split(',')) if a.box else None
    sheet_rows = []
    for stem in stems:
        r = export_one(a, runs, stem, box)
        if r is None:
            continue
        boxed, psnr = r
        h = 160
        ims = [boxed[c].resize((int(boxed[c].size[0] * h / boxed[c].size[1]), h), Image.LANCZOS) for c in COLS]
        row = Image.new('RGB', (sum(i.size[0] + 4 for i in ims), h + 16), 'white')
        d, x = ImageDraw.Draw(row), 0
        for c, im in zip(COLS, ims):
            row.paste(im, (x, 0))
            d.text((x + 2, h + 2), stem if c == 'gt' else f"{c} {psnr.get(c, 0):.2f}", fill=(0, 0, 0))
            x += im.size[0] + 4
        sheet_rows.append(row)
        print(f"[export] {stem}: " + ', '.join(f"{c} {psnr[c]:.2f}" for c in COLS if c in psnr)
              + f" dB -> {os.path.join(OUT, a.target, stem)}/")
    if sheet_rows and not a.image:
        sheet = Image.new('RGB', (max(r.size[0] for r in sheet_rows), sum(r.size[1] + 6 for r in sheet_rows)), 'white')
        y = 0
        for r in sheet_rows:
            sheet.paste(r, (0, y))
            y += r.size[1] + 6
        p = os.path.join(OUT, f'contact_{a.target}.png')
        sheet.save(p)
        print(f"[export] contact sheet -> {p}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('select', 'export'):
        q = sub.add_parser(name)
        q.add_argument('--backbone', default='adair')
        q.add_argument('--setting', default='5task')
        q.add_argument('--target', default='SOTS_Indoor')
        if name == 'select':
            q.add_argument('--k', type=int, default=8, help=argparse.SUPPRESS)
        else:
            q.add_argument('--image', default=None)
            q.add_argument('--box', default=None)
    a = p.parse_args()
    {'select': cmd_select, 'export': cmd_export}[a.cmd](a)


if __name__ == '__main__':
    main()
