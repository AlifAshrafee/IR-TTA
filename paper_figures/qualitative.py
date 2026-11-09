"""Qualitative strip (Fig. 2) and the collapse thumbnails of the teaser: choose images from the finished runs, then
export tiles you can lay out yourself (Lucidchart / Inkscape) plus a matplotlib preview.

    # 1. choose candidates from the suite's per-image PSNRs (no GPU, no re-run)
    python paper_figures/qualitative.py select --backbone adair --setting 5task --target Rain100H
    # 2. re-run those cases saving only the chosen images (scripts/run.sh figures does 1 + 2 for a list of sets)
    # 3. tiles + contact sheets + a preview strip
    python paper_figures/qualitative.py compose --backbone adair --setting 5task \\
        --targets Rain100H SOTS_Indoor OHAZY DPDD HIDE [--pick Rain100H=<stem> ...] [--box Rain100H=<stem>:x,y,size]

select   reads ``results/suite/<run>/per_image.csv`` of the pretrained model, + DCTTA and + ours (tag F1) for one
         backbone / setting / test set and ranks images by how clearly ours beats BOTH baselines,
         min(ours - pretrained, ours - DCTTA) (rain sets in Y-PSNR, as in the tables). Writes
         ``results/figures/select/<bb>_<setting>_<target>.{txt,csv}``: the top ``--k`` images, the image at the
         median gain of the set (an honest "typical" example for the caption) and the ``--k_collapse`` images where
         DCTTA falls furthest below the pretrained model.
compose  reads the re-run outputs (``results/figure_runs``) and, per selected image, writes into
         ``results/figures/tiles/<target>/<stem>/``:
           <col>.png            full image (optionally cropped to ``--aspect``), col in input/pretrained/dctta/ours/gt
           <col>_zoom.png       the zoom region, upsampled to ``--zoom_px`` (Lanczos)
           <col>_tile.png       full image + box + zoom inset in the bottom-right corner (ready to place)
           psnr.json            the PSNR of every column (per_image.csv of the re-run; Y-PSNR on rain sets)
         plus ``results/figures/contact_<bb>_<setting>_<target>.png`` (every candidate, one row each: pick from it)
         and ``results/figures/strip_preview.pdf`` (one row per target: the ``--pick`` image, or the top candidate).
         The zoom box (unless ``--box``) is the ``--zoom_frac`` square where ours removes the most error relative
         to the pretrained model (largest mean of |pretrained - GT| - |ours - GT|).
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

COLS = ['input', 'pretrained', 'dctta', 'ours', 'gt']
HEAD = {'input': 'Input', 'pretrained': 'Pretrained', 'dctta': '+ DCTTA', 'ours': '+ Ours', 'gt': 'Ground truth'}
ROWKEY = {'source': 'pretrained', 'dctta': 'dctta', 'ours': 'ours'}
BOX_COLOR = (255, 204, 0)
SHIFT = {'Rain100H': 'rain: light → heavy', 'SOTS_Indoor': 'haze: outdoor → indoor', 'DenseHaze': 'haze: synthetic → real',
         'OHAZY': 'haze: synthetic → real', 'Kodak24_Poisson': 'noise: Gaussian → Poisson', 'HIDE': 'blur: GoPro → HIDE',
         'LOLv2_Syn': 'low light: LOL → LOLv2-S', 'DPDD': 'blur: motion → defocus'}


# ---------------------------------------------------------------------------- #
# runs
# ---------------------------------------------------------------------------- #
def find_runs(root, backbone, setting, target, seed=23, tag='F1'):
    """{'pretrained' | 'dctta' | 'ours': run folder} of one cell, the row rules of tools/table.py."""
    out = {}
    if not os.path.isdir(root):
        return out
    for d in sorted(os.listdir(root)):
        rp = os.path.join(root, d, 'result.json')
        if not os.path.exists(rp):
            continue
        with open(rp) as f:
            r = json.load(f)
        r = {k: ('' if v is None else v) for k, v in r.items()}
        if (r.get('backbone') or 'promptir') != backbone or r.get('source') != setting or r.get('target') != target:
            continue
        if int(r.get('seed') or 0) != seed:
            continue
        k = row_key(r, tag)
        if k in ROWKEY:
            out[ROWKEY[k]] = os.path.join(root, d)
    return out


def per_image(run_dir, target):
    """{stem: psnr} (Y-PSNR on rain sets, as in the tables)."""
    key = 'psnr_y' if target in Y_TARGETS else 'psnr'
    with open(os.path.join(run_dir, 'per_image.csv')) as f:
        return {os.path.splitext(r['name'])[0]: num(r[key]) for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------- #
# select
# ---------------------------------------------------------------------------- #
def cmd_select(a):
    runs = find_runs(a.runs, a.backbone, a.setting, a.target, a.seed)
    miss = [c for c in ('pretrained', 'dctta', 'ours') if c not in runs]
    if miss:
        raise SystemExit(f"[select] {a.backbone} {a.setting} {a.target}: no finished {miss} run under {a.runs}")
    p = {c: per_image(d, a.target) for c, d in runs.items()}
    stems = [s for s in p['ours'] if all(s in p[c] and p[c][s] is not None for c in p)]
    rows = []
    for s in stems:
        g_src, g_dc = p['ours'][s] - p['pretrained'][s], p['ours'][s] - p['dctta'][s]
        rows.append(dict(stem=s, pretrained=p['pretrained'][s], dctta=p['dctta'][s], ours=p['ours'][s],
                         ours_minus_pretrained=g_src, ours_minus_dctta=g_dc, score=min(g_src, g_dc),
                         dctta_minus_pretrained=p['dctta'][s] - p['pretrained'][s]))
    rows.sort(key=lambda r: -r['score'])
    med = statistics.median(r['ours_minus_pretrained'] for r in rows)
    typical = min(rows, key=lambda r: abs(r['ours_minus_pretrained'] - med))
    collapse = sorted(rows, key=lambda r: r['dctta_minus_pretrained'])[:a.k_collapse]
    chosen, why = [], {}
    for r, w in [(r, 'top') for r in rows[:a.k]] + [(typical, 'median')] + [(r, 'dctta_collapse') for r in collapse]:
        if r['stem'] not in why:
            chosen.append(r)
        why[r['stem']] = '+'.join(filter(None, [why.get(r['stem']), w]))
    os.makedirs(a.sel_dir, exist_ok=True)
    base = os.path.join(a.sel_dir, f"{a.backbone}_{a.setting}_{a.target}")
    with open(base + '.txt', 'w') as f:
        f.write('\n'.join(r['stem'] for r in chosen) + '\n')
    with open(base + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['stem', 'why'] + [k for k in rows[0] if k != 'stem'])
        w.writeheader()
        for r in chosen:
            w.writerow({'why': why[r['stem']], **{k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()}})
    print(f"[select] {a.backbone} {a.setting} {a.target}: {len(rows)} images, median ours - pretrained {med:+.2f} dB;"
          f" {len(chosen)} chosen -> {base}.txt")
    for r in chosen:
        print(f"  {r['stem']:<28} {why[r['stem']]:<20} pre {r['pretrained']:6.2f}  dctta {r['dctta']:6.2f}  "
              f"ours {r['ours']:6.2f}")


# ---------------------------------------------------------------------------- #
# compose
# ---------------------------------------------------------------------------- #
def load_pair(target, stem):
    """(LQ, GT) uint8 HxWx3 exactly as evaluated (centre crop to a multiple of 16, O-HAZY's max_side)."""
    from irtta.data import available_names, load_registry, make_eval_dataset, resolve_spec
    spec = resolve_spec(target, load_registry())
    name = next(n for n in available_names(spec) if os.path.splitext(n)[0] == stem)
    _, lq, gt = make_eval_dataset(spec, base=16, names=[name])[0]
    to8 = lambda t: (t.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)  # noqa: E731
    return to8(lq), to8(gt)


def auto_box(pre, ours, gt, frac):
    """Top-left (x, y) and side of the square where ours removes the most error vs the pretrained model."""
    e = (np.abs(pre.astype(np.float32) - gt) - np.abs(ours.astype(np.float32) - gt)).mean(2)
    H, W = e.shape
    s = max(16, int(round(min(H, W) * frac)))
    ii = np.pad(e.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    tot = ii[s:, s:] - ii[:-s, s:] - ii[s:, :-s] + ii[:-s, :-s]
    m = max(2, s // 8)   # keep the box off the border
    tot[:m, :], tot[-m:, :], tot[:, :m], tot[:, -m:] = -np.inf, -np.inf, -np.inf, -np.inf
    y, x = np.unravel_index(np.argmax(tot), tot.shape)
    return int(x), int(y), s


def aspect_crop(img_shape, box, aspect):
    """Crop window (x0, y0, w, h) of the given aspect (w / h) that keeps the zoom box inside, else the full image."""
    H, W = img_shape[:2]
    if not aspect:
        return 0, 0, W, H
    if W / H > aspect:
        w, h = int(round(H * aspect)), H
    else:
        w, h = W, int(round(W / aspect))
    bx, by, bs = box
    x0 = int(np.clip(bx + bs / 2 - w / 2, 0, W - w))
    y0 = int(np.clip(by + bs / 2 - h / 2, 0, H - h))
    return x0, y0, w, h


def make_tiles(imgs, box, crop, zoom_px, inset_frac, line):
    """{col: (full, zoom, tile)} PIL images."""
    x0, y0, w, h = crop
    bx, by, bs = box
    out = {}
    for c, a in imgs.items():
        full = Image.fromarray(a[y0:y0 + h, x0:x0 + w])
        zoom = Image.fromarray(a[by:by + bs, bx:bx + bs]).resize((zoom_px, zoom_px), Image.LANCZOS)
        tile = full.copy()
        d = ImageDraw.Draw(tile)
        lw = max(line, int(round(w / 250)))
        d.rectangle([bx - x0, by - y0, bx - x0 + bs - 1, by - y0 + bs - 1], outline=BOX_COLOR, width=lw)
        iw = int(round(w * inset_frac))
        inset = zoom.resize((iw, iw), Image.LANCZOS)
        # inset in the corner farthest from the box, so it never hides the region it magnifies
        cx, cy = bx - x0 + bs / 2, by - y0 + bs / 2
        px = lw if cx > w / 2 else w - iw - lw
        py = lw if cy > h / 2 else h - iw - lw
        d.rectangle([px - lw, py - lw, px + iw + lw - 1, py + iw + lw - 1], fill=BOX_COLOR)
        tile.paste(inset, (px, py))
        out[c] = (full, zoom, tile)
    return out


def load_case(a, target, stem, runs, box=None):
    lq, gt = load_pair(target, stem)
    imgs = {'input': lq, 'gt': gt}
    for c in ('pretrained', 'dctta', 'ours'):
        p = os.path.join(runs[c], 'images', stem + '.png')
        if not os.path.exists(p):
            return None
        imgs[c] = np.asarray(Image.open(p).convert('RGB'))
        if imgs[c].shape != gt.shape:
            raise SystemExit(f"{p}: shape {imgs[c].shape} != GT {gt.shape}")
    box = box or auto_box(imgs['pretrained'], imgs['ours'], gt, a.zoom_frac)
    psnr = {c: per_image(runs[c], target).get(stem) for c in ('pretrained', 'dctta', 'ours')}
    psnr['input'] = psnr_u8(lq, gt, target in Y_TARGETS)
    return imgs, box, psnr


def psnr_u8(a, b, y=False):
    """PSNR of two uint8 images, RGB or BT.601 Y (irtta/evaluation.py conventions)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    if y:
        w = np.array([65.481, 128.553, 24.966]) / 255.
        a, b = 16 + a @ w, 16 + b @ w
    mse = max(float(((a - b) ** 2).mean()), 1e-12)
    return 10 * np.log10(255. ** 2 / mse)


def preview(rows, path, title_shift=True):
    """rows: [(target, stem, tiles, psnr)] -> one PDF, rows = targets, columns = COLS."""
    import matplotlib.pyplot as plt
    S.setup(7)
    aspects = [t['gt'][2].size[0] / t['gt'][2].size[1] for _, _, t, _ in rows]
    W = S.TEXT_W
    cw = W / len(COLS)
    fig_h = sum(cw / ar for ar in aspects) + 0.18 * len(rows) + 0.15
    fig, axes = plt.subplots(len(rows), len(COLS), figsize=(W, fig_h), squeeze=False,
                             gridspec_kw=dict(wspace=0.02, hspace=0.16, height_ratios=[1 / ar for ar in aspects]))
    for i, (t, stem, tiles, psnr) in enumerate(rows):
        for j, c in enumerate(COLS):
            ax = axes[i, j]
            ax.imshow(tiles[c][2])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(HEAD[c], fontsize=7.5, pad=2,
                             color=S.C_OURS if c == 'ours' else S.C_DCTTA if c == 'dctta' else S.INK)
            if c in psnr and psnr[c] is not None:
                ax.set_xlabel(f"{psnr[c]:.2f} dB", fontsize=6.8, labelpad=1.5,
                              color=S.C_OURS if c == 'ours' else S.INK)
            if j == 0:
                ax.set_ylabel(f"{S.SHORT.get(t, t)}\n{SHIFT.get(t, '')}" if title_shift else S.SHORT.get(t, t),
                              fontsize=6.5, labelpad=2)
    fig.savefig(path)
    fig.savefig(os.path.splitext(path)[0] + '.png', dpi=220)
    plt.close(fig)


def contact_sheet(cases, path, h=150):
    """Every candidate of one target: a row of the five tiles (scaled to height h) with the PSNRs written under."""
    rows = []
    for stem, tiles, psnr in cases:
        ims = [tiles[c][2] for c in COLS]
        ims = [im.resize((int(im.size[0] * h / im.size[1]), h), Image.LANCZOS) for im in ims]
        wsum = sum(im.size[0] for im in ims) + 4 * len(ims)
        row = Image.new('RGB', (wsum, h + 18), 'white')
        x = 0
        d = ImageDraw.Draw(row)
        for c, im in zip(COLS, ims):
            row.paste(im, (x, 0))
            lab = f"{c} {psnr[c]:.2f}" if psnr.get(c) is not None else c
            d.text((x + 2, h + 3), lab if c != 'input' else f"{stem}", fill=(0, 0, 0))
            x += im.size[0] + 4
        rows.append(row)
    if not rows:
        return
    sheet = Image.new('RGB', (max(r.size[0] for r in rows), sum(r.size[1] + 6 for r in rows)), 'white')
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.size[1] + 6
    sheet.save(path)


def parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split('=', 1)
        out[k] = v
    return out


def cmd_compose(a):
    picks = parse_kv(a.pick)
    boxes = {}
    for k, v in parse_kv(a.box).items():
        stem, xys = v.split(':')
        boxes[(k, stem)] = tuple(int(t) for t in xys.split(','))
    aspect = None
    if a.aspect:
        w, h = a.aspect.split(':')
        aspect = float(w) / float(h)
    strip = []
    for t in a.targets:
        runs = find_runs(a.fig_runs, a.backbone, a.setting, t, a.seed)
        miss = [c for c in ('pretrained', 'dctta', 'ours') if c not in runs]
        if miss:
            print(f"[compose] {t}: no {miss} run with saved images under {a.fig_runs} (run.sh figures); skipped")
            continue
        sel = os.path.join(a.sel_dir, f"{a.backbone}_{a.setting}_{t}.txt")
        stems = [picks[t]] if t in picks and a.only_picks else [s.strip() for s in open(sel) if s.strip()]
        if t in picks and picks[t] not in stems:
            stems.insert(0, picks[t])
        suite = find_runs(a.runs, a.backbone, a.setting, t, a.seed)
        cases = []
        for stem in stems:
            case = load_case(a, t, stem, runs, boxes.get((t, stem)))
            if case is None:
                print(f"[compose] {t}/{stem}: image not saved by the re-run; skipped")
                continue
            imgs, box, psnr = case
            for c in ('pretrained', 'dctta', 'ours'):   # the re-run must reproduce the tables' run
                if c in suite and psnr[c] is not None:
                    ref = per_image(suite[c], t).get(stem)
                    if ref is not None and abs(ref - psnr[c]) > 0.05:
                        print(f"  [warn] {t}/{stem} {c}: re-run {psnr[c]:.2f} dB vs suite {ref:.2f} dB")
            tiles = make_tiles(imgs, box, aspect_crop(imgs['gt'].shape, box, aspect), a.zoom_px, a.inset, a.line)
            od = os.path.join(a.out, 'tiles', t, stem)
            os.makedirs(od, exist_ok=True)
            for c, (full, zoom, tile) in tiles.items():
                full.save(os.path.join(od, f'{c}.png'))
                zoom.save(os.path.join(od, f'{c}_zoom.png'))
                tile.save(os.path.join(od, f'{c}_tile.png'))
            with open(os.path.join(od, 'psnr.json'), 'w') as f:
                json.dump({'target': t, 'stem': stem, 'metric': 'psnr_y' if t in Y_TARGETS else 'psnr',
                           'box_xy_size': box, **psnr}, f, indent=1)
            cases.append((stem, tiles, psnr))
        if not cases:
            continue
        cs = os.path.join(a.out, f"contact_{a.backbone}_{a.setting}_{t}.png")
        contact_sheet(cases, cs)
        print(f"[compose] {t}: {len(cases)} cases -> {a.out}/tiles/{t}/, contact sheet {cs}")
        stem = picks.get(t, cases[0][0])
        best = next(c for c in cases if c[0] == stem)
        strip.append((t, stem, best[1], best[2]))
    if strip:
        p = os.path.join(a.out, 'strip_preview.pdf')
        preview(strip, p)
        print(f"[compose] preview strip ({', '.join(f'{t}={s}' for t, s, _, _ in strip)}) -> {p}")
        for t, s, _, _ in strip:
            print(f"  teaser thumbnail: --thumb dctta:{a.backbone}:{a.setting}:{t}="
                  f"{os.path.relpath(os.path.join(a.out, 'tiles', t, s, 'dctta.png'), S.REPO)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('select', 'compose'):
        q = sub.add_parser(name)
        q.add_argument('--backbone', default='adair')
        q.add_argument('--setting', default='5task')
        q.add_argument('--seed', type=int, default=23)
        q.add_argument('--runs', default=os.path.join(S.REPO, 'results', 'suite'), help='the tables\' runs')
        q.add_argument('--sel_dir', default=os.path.join(S.REPO, 'results', 'figures', 'select'))
        if name == 'select':
            q.add_argument('--target', required=True)
            q.add_argument('--k', type=int, default=6, help='top images by min(ours - pretrained, ours - DCTTA)')
            q.add_argument('--k_collapse', type=int, default=2, help='images where DCTTA drops the most')
        else:
            q.add_argument('--targets', nargs='+', default=['Rain100H', 'SOTS_Indoor', 'OHAZY', 'DPDD', 'HIDE'])
            q.add_argument('--fig_runs', default=os.path.join(S.REPO, 'results', 'figure_runs'),
                           help='the re-runs with saved images (run.sh figures)')
            q.add_argument('--out', default=os.path.join(S.REPO, 'results', 'figures'))
            q.add_argument('--pick', action='append', default=[], help='target=stem: the image of the preview strip')
            q.add_argument('--only_picks', action='store_true', help='compose only the --pick images')
            q.add_argument('--box', action='append', default=[], help='target=stem:x,y,size (zoom box, pixels)')
            q.add_argument('--zoom_frac', type=float, default=0.2, help='zoom box side / short image side')
            q.add_argument('--zoom_px', type=int, default=512)
            q.add_argument('--inset', type=float, default=0.38, help='inset width / tile width')
            q.add_argument('--aspect', default='4:3', help="crop every tile to this aspect (w:h) around the box; '' = off")
            q.add_argument('--line', type=int, default=2, help='minimum box line width (px)')
    a = p.parse_args()
    {'select': cmd_select, 'compose': cmd_compose}[a.cmd](a)


if __name__ == '__main__':
    main()
