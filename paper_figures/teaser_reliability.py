"""Teaser (Fig. 1): how reliably does test-time adaptation help? Change in PSNR over the frozen pretrained model for
every backbone x setting x test-set cell of the main tables (6 x {5 + 8} = 78), + DCTTA vs + Ours.

    python paper_figures/teaser_reliability.py --png                        # both styles, from results/suite
    python paper_figures/teaser_reliability.py --style matrix --width full
    python paper_figures/teaser_reliability.py --thumb dctta:adair:5task:HIDE=results/figures/tiles/HIDE/<img>/dctta.png

Reads the paper tables (``results/suite/table_main_{3task,5task}.md``, ``bash scripts/run.sh tables``), so every
number in the figure is the number printed in the tables. Writes into ``--out`` (default results/figures):

  teaser_matrix.pdf   "report card": rows = backbones, columns = test sets (3-task | 5-task), colour = change in
                      PSNR (diverging, clipped at +-``--clip`` dB; cells below -1 dB print their value)
  teaser_scatter.pdf  pretrained PSNR (x) vs adapted PSNR (y); the diagonal is "no change", the shaded half-plane
                      "adaptation hurts"; groups of collapsed cells (< -``--collapse`` dB) are labelled by test set
  figure_stats.md     the numbers for the caption / abstract / intro \\TODO s

``--thumb method:backbone:setting:target[@dx,dy]=image`` (repeatable) places a small image next to that cell / point,
e.g. the collapsed DCTTA output exported by ``paper_figures/qualitative.py compose``; ``@dx,dy`` moves it (points).
Default width: one column (scatter); the matrix reads best at ``--width full``.
"""
import argparse
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402

LABEL = {'dctta': '+ DCTTA', 'ours': '+ Ours'}
COLOR = {'dctta': S.C_DCTTA, 'ours': S.C_OURS}


def deltas(vals, keys, m):
    return np.array([vals[(m, bb, s, t)] - vals[('source', bb, s, t)] for bb, s, t in keys])


# ---------------------------------------------------------------------------- #
# statistics (caption, abstract and intro numbers)
# ---------------------------------------------------------------------------- #
def stats_md(vals, keys):
    L = [f"# Teaser statistics ({len(keys)} backbone x setting x test-set cells; PSNR change vs pretrained, dB)\n",
         "Computed from the 2-decimal paper tables, so they match what the tables print.\n",
         "| | below pretrained | below by > 1 dB | min | median | mean | max |", "|---|---|---|---|---|---|---|"]
    for m in ('dctta', 'ours'):
        d = deltas(vals, keys, m)
        L.append(f"| {LABEL[m]} | {(d < 0).sum()}/{len(d)} | {(d < -1).sum()}/{len(d)} | {d.min():+.2f} | "
                 f"{np.median(d):+.2f} | {d.mean():+.2f} | {d.max():+.2f} |")
    d_o, d_d = deltas(vals, keys, 'ours'), deltas(vals, keys, 'dctta')
    L += ["", f"Ours above pretrained: {(d_o > 0).sum()}/{len(d_o)} cells; ours above DCTTA: "
          f"{(d_o > d_d).sum()}/{len(d_o)} cells.", "", "## Per family (cells; below pretrained; min change)", "",
          "| family | n | DCTTA below | DCTTA min | Ours below | Ours min |", "|---|---|---|---|---|---|"]
    for fam in ('rain', 'haze', 'noise', 'blur', 'low-light'):
        idx = [i for i, k in enumerate(keys) if S.FAMILY[k[2]] == fam]
        if idx:
            dd, do = d_d[idx], d_o[idx]
            L.append(f"| {fam} | {len(idx)} | {(dd < 0).sum()} | {dd.min():+.2f} | {(do < 0).sum()} | {do.min():+.2f} |")
    L += ["", "## Per backbone x setting average (change of the table's Avg column vs pretrained)", "",
          "| backbone | setting | DCTTA | Ours |", "|---|---|---|---|"]
    gains = {'dctta': [], 'ours': []}
    for s in ('3task', '5task'):
        for bb in S.NAMES:
            ks = [k for k in keys if k[0] == bb and k[1] == s]
            if not ks:
                continue
            src = statistics.fmean(vals[('source', bb, s, t)] for _, _, t in ks)
            row = []
            for m in ('dctta', 'ours'):
                g = statistics.fmean(vals[(m, bb, s, t)] for _, _, t in ks) - src
                gains[m].append((s, g))
                row.append(f"{g:+.2f}")
            L.append(f"| {S.NAMES[bb]} | {s} | {row[0]} | {row[1]} |")
    for m in ('dctta', 'ours'):
        for s in ('3task', '5task'):
            g = [x for ss, x in gains[m] if ss == s]
            if g:
                L.append(f"{LABEL[m]}, {s}: average gain {min(g):+.2f} to {max(g):+.2f} dB (mean {statistics.fmean(g):+.2f})")
        g = [x for _, x in gains[m]]
        L.append(f"{LABEL[m]}, all: {min(g):+.2f} to {max(g):+.2f} dB")
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------------------- #
# thumbnails
# ---------------------------------------------------------------------------- #
def parse_thumbs(specs):
    """'method:backbone:setting:target[@dx,dy]=image' -> ({key: image}, {key: (dx, dy) offset in points})."""
    out, off = {}, {}
    for sp in specs or []:
        key, path = sp.split('=', 1)
        if '@' in key:
            key, o = key.split('@')
            off[tuple(key.split(':'))] = tuple(float(v) for v in o.split(','))
        out[tuple(key.split(':'))] = path
    return out, off


def place_thumb(ax, xy, path, offset, zoom_px=70, color=S.INK2):
    from PIL import Image
    im = Image.open(path).convert('RGB')
    im.thumbnail((zoom_px * 4, zoom_px * 4))
    oi = OffsetImage(np.asarray(im), zoom=zoom_px / max(im.size))
    ab = AnnotationBbox(oi, xy, xybox=offset, xycoords='data', boxcoords='offset points', pad=0.15,
                        bboxprops=dict(edgecolor=color, linewidth=0.6),
                        arrowprops=dict(arrowstyle='-', color=color, linewidth=0.6, shrinkA=0, shrinkB=1))
    ax.add_artist(ab)


# ---------------------------------------------------------------------------- #
# styles
# ---------------------------------------------------------------------------- #
def fig_matrix(vals, keys, args, thumbs):
    """Two 6 x 13 grids (DCTTA, ours) sharing a diverging colour scale."""
    cols = [('3task', t) for t in S.TESTS['3task']] + [None] + [('5task', t) for t in S.TESTS['5task']]
    bbs = [bb for bb in S.NAMES if any(k[0] == bb for k in keys)]
    cmap = LinearSegmentedColormap.from_list('harm_gain', [S.DIV_NEG, S.DIV_MID, S.DIV_POS])
    norm = TwoSlopeNorm(vmin=-args.clip, vcenter=0.0, vmax=args.clip)
    full = args.width == 'full'
    W = S.TEXT_W if full else S.COL_W
    if full:
        fig, axes = plt.subplots(1, 2, figsize=(W, 1.55), gridspec_kw=dict(wspace=0.06))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(W, 2.55), gridspec_kw=dict(hspace=0.12))
    for i, (ax, m) in enumerate(zip(axes, ('dctta', 'ours'))):
        grid = np.full((len(bbs), len(cols)), np.nan)
        for r, bb in enumerate(bbs):
            for c, col in enumerate(cols):
                if col and (m, bb, col[0], col[1]) in vals:
                    grid[r, c] = vals[(m, bb, *col)] - vals[('source', bb, *col)]
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
        for r in range(grid.shape[0]):   # white cell borders (spacer), values on harmful cells
            for c in range(grid.shape[1]):
                if np.isnan(grid[r, c]):
                    continue
                ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, edgecolor='white', linewidth=0.8))
                if grid[r, c] < -1:
                    ax.text(c, r, f"{grid[r, c]:.0f}".replace('-', '−'), ha='center', va='center', fontsize=5.2,
                            color='white' if grid[r, c] < -args.clip / 2 else S.INK)
        ax.set_xlim(-.5, len(cols) - .5)
        ax.set_ylim(len(bbs) - .5, -.5)
        ax.set_yticks(range(len(bbs)))
        ax.set_yticklabels([S.NAMES[b] for b in bbs] if (i == 0 or not full) else [])
        show_x = full or i == 1
        ax.set_xticks([c for c, col in enumerate(cols) if col])
        ax.set_xticklabels([S.SHORT[col[1]] for col in cols if col] if show_x else [], rotation=55, ha='right',
                           rotation_mode='anchor', fontsize=6.3)
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        d = deltas(vals, keys, m)
        ax.set_title(f"{LABEL[m]}: below pretrained in {(d < 0).sum()}/{len(d)} cells, worst {d.min():+.1f} dB"
                     .replace('-', '−'), fontsize=7.5, color=COLOR[m], loc='left', pad=13 if (i == 0 or full) else 3,
                     fontweight='bold')
        if i == 0 or full:
            for (a, b), lab in (((0, 4), '3-task'), ((6, 13), '5-task')):
                ax.annotate('', xy=(a - .4, -.75), xytext=(b + .4, -.75), annotation_clip=False,
                            arrowprops=dict(arrowstyle='-', color=S.INK2, linewidth=0.6))
                ax.text((a + b) / 2, -.85, lab, ha='center', va='bottom', fontsize=6.3, color=S.INK2)
        for k, p in thumbs.items():
            if k[0] == m and k[1] in bbs and (k[2], k[3]) in [c for c in cols if c]:
                c = [c for c in cols if c].index((k[2], k[3]))
                c += 1 if c >= 5 else 0
                place_thumb(ax, (c, bbs.index(k[1])), p, offset=(0, -40), zoom_px=args.thumb_px)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axes), fraction=0.025 if full else 0.04,
                      pad=0.015, aspect=25)
    cb.set_label('PSNR change vs pretrained (dB)', fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=1.5)
    cb.outline.set_visible(False)
    t = cb.get_ticks()
    cb.set_ticks(t)
    cb.set_ticklabels([(f"≤{v:.0f}" if v <= -args.clip else f"≥{v:.0f}" if v >= args.clip else f"{v:+.0f}" if v else '0')
                       .replace('-', '−') for v in t])
    return fig


def fig_scatter(vals, keys, args, thumbs):
    side = S.COL_W if args.width == 'col' else S.TEXT_W / 2
    fig, ax = plt.subplots(figsize=(side, side * 0.95))
    src = np.array([vals[('source', *k)] for k in keys])
    lo, hi = 0, 34
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=S.DIV_NEG, alpha=0.06, linewidth=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color=S.INK2, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax.text(32.6, 31.9, 'no change', rotation=45, ha='right', va='top', fontsize=6, color=S.INK2,
            rotation_mode='anchor')
    ax.text(8.6, lo + 0.6, 'adaptation hurts', ha='left', va='bottom', fontsize=6.5, color=S.DIV_NEG,
            style='italic')
    pos, summary = {}, []
    for m in ('dctta', 'ours'):
        y = np.array([vals[(m, *k)] for k in keys])
        d = y - src
        ax.scatter(src, y, s=12, color=COLOR[m], edgecolor='white', linewidth=0.4, zorder=3 if m == 'ours' else 2,
                   label=LABEL[m])
        summary.append((m, int((d < 0).sum()), len(d), float(d.min())))
        for k, xx, yy in zip(keys, src, y):
            pos[(m, *k)] = (xx, yy)
        if m == 'dctta':   # name the collapsed groups by test set (data-driven, no hand placement)
            groups = {}
            for k, xx, yy, dd in zip(keys, src, y, d):
                if dd < -args.collapse:
                    groups.setdefault(k[2], []).append((xx, yy))
            for t, pts in groups.items():
                pts = np.array(pts)
                ax.text(pts[:, 0].max() + 0.6, pts[:, 1].mean(), f"{S.SHORT[t]} ({S.FAMILY[t]})",
                        fontsize=6, color=S.C_DCTTA, va='center', ha='left')
    ax.set_xlim(8, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('PSNR of the pretrained model (dB)')
    ax.set_ylabel('PSNR after test-time adaptation (dB)')
    ax.legend(loc='upper left', frameon=False, handletextpad=0.1, borderaxespad=0.1, markerscale=1.3)
    # the two numbers of the figure, in the empty lower-right corner of the "hurts" half-plane
    x0, y0, dx = 21.6, 3.2, (7.7, 12.0)
    ax.text(x0 + dx[0], y0, 'harmed', fontsize=6.2, color=S.INK2, ha='right', va='bottom')
    ax.text(x0 + dx[1], y0, 'worst', fontsize=6.2, color=S.INK2, ha='right', va='bottom')
    for i, (m, nb, n, w) in enumerate(summary):
        yy = y0 - 1.45 * (i + 1)
        kw = dict(fontsize=6.6, color=COLOR[m], va='bottom', fontweight='bold')
        ax.text(x0, yy, LABEL[m][2:], ha='left', **kw)
        ax.text(x0 + dx[0], yy, f"{nb}/{n}", ha='right', **kw)
        ax.text(x0 + dx[1], yy, f"{w:+.1f}".replace('-', '\u2212'), ha='right', **kw)
    ax.grid(color=S.GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for k, p in thumbs.items():
        if k in pos:
            place_thumb(ax, pos[k], p, offset=args.thumb_offset.get(k, (-35, 35)), zoom_px=args.thumb_px,
                        color=COLOR[k[0]])
    return fig


STYLES = {'scatter': fig_scatter, 'matrix': fig_matrix}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tables', default=os.path.join(S.REPO, 'results', 'suite'),
                   help='folder with table_main_{3task,5task}.md')
    p.add_argument('--out', default=os.path.join(S.REPO, 'results', 'figures'))
    p.add_argument('--style', nargs='+', default=list(STYLES), choices=list(STYLES))
    p.add_argument('--width', default='col', choices=['col', 'full'], help='one column (3.28 in) or text width')
    p.add_argument('--clip', type=float, default=6.0, help='matrix colour scale saturates at +-clip dB')
    p.add_argument('--thumb', action='append', default=[],
                   help='method:backbone:setting:target[@dx,dy]=image (repeatable; dx,dy: offset in points)')
    p.add_argument('--collapse', type=float, default=5.0, help='scatter: label DCTTA groups below -collapse dB')
    p.add_argument('--thumb_px', type=int, default=40, help="thumbnail size in points")
    p.add_argument('--png', action='store_true', help='also write PNG previews')
    args = p.parse_args()
    S.setup()
    vals = S.load_main(args.tables)
    keys = S.pairs(vals)
    if not keys:
        raise SystemExit("no complete backbone x set cells in the tables")
    os.makedirs(args.out, exist_ok=True)
    md = stats_md(vals, keys)
    with open(os.path.join(args.out, 'figure_stats.md'), 'w') as f:
        f.write(md)
    print(md)
    thumbs, args.thumb_offset = parse_thumbs(args.thumb)
    for st in args.style:
        fig = STYLES[st](vals, keys, args, thumbs)
        base = os.path.join(args.out, f'teaser_{st}' + ('' if args.width == 'full' else '_col'))
        fig.savefig(base + '.pdf')
        if args.png:
            fig.savefig(base + '.png', dpi=300)
        plt.close(fig)
        print(f"-> {base}.pdf")


if __name__ == '__main__':
    main()
