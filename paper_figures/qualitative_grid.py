"""Fig. `fig:qualitative` (main paper, text width) and the supplementary grids: one test case per row.

    GPU=0 bash scripts/run.sh qualitative          # picks the images, re-runs the 3 methods per row, composes
    python paper_figures/qualitative_grid.py       # compose again only (no GPU) -> results/figures/qualitative_grid.*

Row layout (six panels of one aspect ratio):
    full input with the zoom region marked | zoom of: Input | Pretrained | + DCTTA | + Ours | Ground truth
The methods are compared on the enlarged region, at about four times the size a corner inset would have, and the full
image keeps the context. Under each zoom is the PSNR of the full image (Y-PSNR on rain sets, as in the tables), read
from the same per-image results as the tables.

A row is  backbone:setting:target[:image[:x,y,w]]
    image   default: the first name in results/figures/qualitative/<backbone>_<setting>_<target>.txt, i.e. the image
            whose gain over the pretrained model is closest to the median of its set (`select --rule median`)
    x,y,w   zoom region by hand, in pixels of the evaluated image (the height follows the panel aspect); default: the
            region where ours and the pretrained output differ most in error
  python paper_figures/qualitative_grid.py --rows adair:5task:Rain100H mirage:5task:DPDD:0012:300,180,160

Options: --rows (default: the four rows of the paper), --name (output file name), --aspect (panel width / height,
1.7: four rows are 3.3 in tall), --zoom (width of the zoom region as a fraction of the image width, 0.25), --no_dctta (drop that column).
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

DEFAULT_ROWS = ['adair:5task:Rain100H', 'moceir:3task:OHAZY', 'bioir:3task:Kodak24_Poisson', 'promptir:5task:SOTS_Indoor']
COLS = [('input', 'Input', None), ('pretrained', 'Pretrained', S.C_PRE), ('existing_tta', '+ DCTTA', S.C_DCTTA),
        ('ours', '+ Ours', S.C_OURS), ('gt', 'Ground truth', None)]
BOX = '#ffd21f'
PX = 420   # pixels per panel in the PDF (1.1 in wide -> about 380 dpi)


# ---------------------------------------------------------------------------- #
# geometry (pure numpy, shared by the real rows and the demo)
# ---------------------------------------------------------------------------- #
def centre_window(h, w, aspect):
    """(x0, y0, ww, hh): the largest centred window of the given aspect ratio."""
    ww = min(w, int(round(h * aspect)))
    hh = min(h, int(round(ww / aspect)))
    return (w - ww) // 2, (h - hh) // 2, ww, hh


def auto_region(pre, ours, gt, win, bw, bh):
    """(x, y) of the bw x bh region inside ``win`` where ours removes the most error relative to the pretrained
    output (largest summed |pre - gt| - |ours - gt|). With ours == pre everywhere it falls back to the centre."""
    x0, y0, ww, hh = win
    e = (np.abs(pre.astype(np.float32) - gt) - np.abs(ours.astype(np.float32) - gt)).mean(2)[y0:y0 + hh, x0:x0 + ww]
    ii = np.pad(e.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    tot = ii[bh:, bw:] - ii[:-bh, bw:] - ii[bh:, :-bw] + ii[:-bh, :-bw]
    m = max(2, bh // 8)   # keep the box off the border of the panel
    if tot.shape[0] > 2 * m and tot.shape[1] > 2 * m:
        tot = tot[m:-m, m:-m]
    else:
        m = 0
    y, x = np.unravel_index(int(np.argmax(tot)), tot.shape)
    return x0 + x + m, y0 + y + m


def panel(img, box, aspect):
    return np.asarray(Image.fromarray(img).crop(box).resize((PX, int(round(PX / aspect))), Image.LANCZOS))


# ---------------------------------------------------------------------------- #
# rows
# ---------------------------------------------------------------------------- #
def parse_row(spec):
    p = spec.split(':')
    if len(p) < 3:
        raise SystemExit(f"row '{spec}': expected backbone:setting:target[:image[:x,y,w]]")
    return dict(backbone=p[0], setting=p[1], target=p[2], image=p[3] if len(p) > 3 and p[3] else None,
                box=tuple(int(v) for v in p[4].split(',')) if len(p) > 4 else None)


def load_row(r, need_dctta=True):
    """Images and full-image PSNRs of one row from results/figure_runs (written by `run.sh qualitative`)."""
    import qualitative as Q
    runs = Q.find_runs(Q.FIG_RUNS, r['backbone'], r['setting'], r['target'])
    need = ['pretrained', 'ours'] + (['existing_tta'] if need_dctta else [])
    miss = [c for c in need if c not in runs]
    hint = (f"run `QUAL_ROWS={r['backbone']}:{r['setting']}:{r['target']} GPU=0 bash scripts/run.sh qualitative`")
    if miss:
        raise SystemExit(f"[grid] {r['backbone']} {r['setting']} {r['target']}: no {miss} run with saved images; {hint}")
    stem = r['image']
    if stem is None:
        nf = os.path.join(Q.OUT, f"{r['backbone']}_{r['setting']}_{r['target']}.txt")
        if not os.path.exists(nf):
            raise SystemExit(f"[grid] no {nf}; {hint}")
        with open(nf) as f:
            stem = f.readline().strip()
    stem = os.path.splitext(stem)[0]
    lq, gt = Q.load_pair(r['target'], stem)
    imgs = {'input': lq, 'gt': gt}
    psnr = {'input': Q.psnr_u8(lq, gt, r['target'] in Q.Y_TARGETS)}
    for c in need:
        p = os.path.join(runs[c], 'images', stem + '.png')
        if not os.path.exists(p):
            raise SystemExit(f"[grid] {p} missing: the image was not among the names saved by that run. Delete "
                             f"{runs[c]} and {hint}")
        imgs[c] = np.asarray(Image.open(p).convert('RGB'))
        psnr[c] = Q.per_image(runs[c], r['target']).get(stem)
    return dict(imgs=imgs, psnr=psnr, stem=stem,
                label=f"{S.PRETTY.get(r['target'], r['target'])}, {S.NAMES.get(r['backbone'], r['backbone'])} "
                      f"({r['setting'][0]}-task)", box=r['box'])


def demo_rows(n):
    """Synthetic images to preview the layout without any results (not for the paper)."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        h, w = [(321, 481), (533, 800), (512, 768), (720, 1280)][i % 4]
        yy, xx = np.mgrid[0:h, 0:w]
        gt = np.stack([127 + 100 * np.sin(xx / (17 + 9 * i) + c) * np.cos(yy / (23 + 5 * i) - c) for c in range(3)], 2)
        mk = lambda s: np.clip(gt + rng.normal(0, s, gt.shape), 0, 255).astype(np.uint8)  # noqa: E731
        gt = np.clip(gt, 0, 255)
        imgs = {'gt': gt.astype(np.uint8), 'input': mk(60), 'pretrained': mk(40), 'existing_tta': mk(35),
                'ours': mk(12)}
        rows.append(dict(imgs=imgs, stem=f'demo{i}', box=None, label=['Rain100H, AdaIR (5-task)', 'SOTS-Indoor, MoCE-IR (3-task)',
                                'Kodak24, Bio-IR (3-task)', 'DPDD, MIRAGE (5-task)'][i % 4],
                         psnr={c: 20 + 3 * j + i for j, c in enumerate(['input', 'pretrained', 'existing_tta', 'ours'])}))
    return rows


# ---------------------------------------------------------------------------- #
def compose(rows, a):
    cols = [c for c in COLS if not (a.no_dctta and c[0] == 'existing_tta')]
    n = len(cols) + 1
    gap, big_gap, head, strip, vgap = 0.03, 0.07, 0.15, 0.115, 0.035   # inches
    pw = (S.TEXT_W - big_gap - (n - 2) * gap) / n
    ph = pw / a.aspect
    H = head + len(rows) * (ph + strip) + (len(rows) - 1) * vgap
    fig = plt.figure(figsize=(S.TEXT_W, H))
    X = [0.0] + [pw + big_gap + j * (pw + gap) for j in range(n - 1)]

    def axes_at(j, top):
        ax = fig.add_axes([X[j] / S.TEXT_W, 1 - (top + ph) / H, pw / S.TEXT_W, ph / H])
        ax.set_axis_off()
        return ax

    for j, (key, title, colour) in enumerate([('full', 'Input, full image', None)] + cols):
        xc = (X[j] + pw / 2) / S.TEXT_W
        fig.text(xc, 1 - 0.075 / H, title, ha='center', va='center', fontsize=8, color=S.INK)
        if colour:   # the identity colours of the other figures
            fig.add_artist(plt.Line2D([(X[j] + 0.22 * pw) / S.TEXT_W, (X[j] + 0.78 * pw) / S.TEXT_W],
                                      [1 - (head - 0.012) / H] * 2, color=colour, linewidth=1.6,
                                      solid_capstyle='butt'))
    for i, r in enumerate(rows):
        top = head + i * (ph + strip + vgap)
        im = r['imgs']
        h, w = im['gt'].shape[:2]
        win = centre_window(h, w, a.aspect)
        bw = max(16, int(round(win[2] * a.zoom)))
        if r['box']:
            bx, by, bw = r['box']
        bh = int(round(bw / a.aspect))
        if not r['box']:
            bx, by = auto_region(im['pretrained'], im['ours'], im['gt'], win, bw, bh)
        ax = axes_at(0, top)
        ax.imshow(panel(im['input'], (win[0], win[1], win[0] + win[2], win[1] + win[3]), a.aspect),
                  interpolation='lanczos')
        k = PX / win[2]
        for lw, ec in ((1.5, '#1f1f1d'), (0.8, BOX)):   # dark halo keeps the box visible on bright content
            ax.add_patch(Rectangle(((bx - win[0]) * k, (by - win[1]) * k), bw * k, bh * k, fill=False, edgecolor=ec,
                                   linewidth=lw))
        ty = 1 - (top + ph + strip * 0.52) / H
        fig.text(0.0, ty, r['label'], ha='left', va='center', fontsize=6.8, color=S.INK)
        for j, (key, _, _) in enumerate(cols, start=1):
            ax = axes_at(j, top)
            ax.imshow(panel(im[key], (bx, by, bx + bw, by + bh), a.aspect), interpolation='lanczos')
            v = r['psnr'].get(key)
            if v is not None:
                fig.text((X[j] + pw / 2) / S.TEXT_W, ty, f"{v:.2f} dB", ha='center', va='center', fontsize=7,
                         color=S.INK)
        print(f"  {r['label']:32s} {r['stem']:>14s}  zoom x,y,w = {bx},{by},{bw}  "
              + '  '.join(f"{c[:4]} {v:.2f}" for c, v in r['psnr'].items() if v is not None))
    plt.rcParams['savefig.bbox'] = None   # keep the exact text width
    S.save(fig, a.name, a.out)
    print(f"  size {S.TEXT_W:.3f} x {H:.2f} in; include with \\includegraphics[width=\\textwidth]")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rows', nargs='+', default=DEFAULT_ROWS)
    p.add_argument('--name', default='qualitative_grid')
    p.add_argument('--aspect', type=float, default=1.7)
    p.add_argument('--zoom', type=float, default=0.25)
    p.add_argument('--no_dctta', action='store_true')
    p.add_argument('--demo', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--out', default=S.FIGS, help=argparse.SUPPRESS)
    a = p.parse_args()
    S.setup()
    if a.demo:
        a.name += '_demo'
        rows = demo_rows(len(a.rows))
    else:
        sys.path.insert(0, S.REPO)
        rows = [load_row(parse_row(s), not a.no_dctta) for s in a.rows]
    compose(rows, a)


if __name__ == '__main__':
    main()
