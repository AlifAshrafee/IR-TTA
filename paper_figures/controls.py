"""Fig. `fig:controls` (main paper, one column): what the gain relies on.

    python paper_figures/controls.py            # -> results/figures/controls.{pdf,png}

Reads results/suite/table_controls.md (PromptIR, three-task, seed 23). One bar per variant: the mean change in PSNR
over the pretrained model on the five target sets, grouped by what is adapted (the pretrained weights or a random
initialisation). The column on the right counts the target sets a variant improves, which the mean hides (pairs
calibrated to another target set have a positive mean but lower four of the five sets). The 5000-step random
initialisation is left to the caption.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

# (group, table label, figure label, is ours)
ROWS = [('Pretrained weights', 'Ours', 'inferred pairs (ours)', True),
        ('Pretrained weights', 'Generic pairs (no inference)', 'generic pairs', False),
        ('Pretrained weights', 'Conditioned on another test set', "another set's pairs", False),
        ('Random initialization', 'Random init, 500 steps', 'inferred pairs', False),
        ('Random initialization', 'Random init, generic pairs', 'generic pairs', False)]
_DELTA = re.compile(r'\(([+\-−]\d+(?:\.\d+)?)\)')


def load(path):
    """{table label: (mean change, [per-set changes])}"""
    out = {}
    for cells in S.md_rows(path)[1:]:
        d = [float(m.group(1).replace('−', '-')) for c in cells[1:-1] for m in [_DELTA.search(c)] if m]
        if d:
            out[cells[0]] = (float(cells[-1].replace('−', '-')), d)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--height', type=float, default=1.55, help='inches (the placeholder allows up to 1.8)')
    p.add_argument('--tables', default=S.TABLES, help=argparse.SUPPRESS)
    p.add_argument('--out', default=S.FIGS, help=argparse.SUPPRESS)
    a = p.parse_args()
    S.setup()
    data = load(os.path.join(a.tables, 'table_controls.md'))
    miss = [r[1] for r in ROWS if r[1] not in data]
    if miss:
        raise SystemExit(f"table_controls.md has no row {miss}")

    # y positions: a header line per group, then its rows
    ys, heads, y, last = [], [], 0.0, None
    for g, *_ in ROWS:
        if g != last:
            y += 0.35 if last else 0.0
            heads.append((y, g))
            y += 0.95
            last = g
        ys.append(y)
        y += 1.0
    fig = plt.figure(figsize=(S.COL_W, a.height))
    ax = fig.add_axes([0.335, 0.215, 0.50, 0.775])
    means = [data[r[1]][0] for r in ROWS]
    lo, hi = min(means + [0]), max(means + [0])
    pad = 0.17 * (hi - lo)
    for yy, (g, key, lab, ours), m in zip(ys, ROWS, means):
        ax.barh(yy, m, height=0.56, color=S.C_OURS if ours else S.C_MUTED, linewidth=0, zorder=2)
        ax.text(m + (0.06 if m >= 0 else -0.06), yy, S.fmt(m), va='center', ha='left' if m >= 0 else 'right',
                fontsize=7, color=S.INK, zorder=3)
        ax.text(-0.035, yy, lab, transform=ax.get_yaxis_transform(), va='center', ha='right', fontsize=7.5,
                color=S.INK)
        n = sum(d > 0 for d in data[key][1])
        ax.text(1.17, yy, f"{n}/{len(data[key][1])}", transform=ax.get_yaxis_transform(), va='center', ha='center',
                fontsize=7, color=S.INK)
    for yy, g in heads:
        ax.text(-0.655, yy, g, transform=ax.get_yaxis_transform(), va='center', ha='left', fontsize=7.5,
                color=S.INK2)
    ax.text(1.17, heads[0][0], 'sets\nimproved', transform=ax.get_yaxis_transform(), va='center', ha='center',
            fontsize=6.5, color=S.INK2, linespacing=0.9)
    ax.axvline(0, color=S.INK2, linewidth=0.6, zorder=1)
    ax.set_xlim(lo - 1.9 * pad, hi + pad)
    ax.set_ylim(ys[-1] + 0.55, -0.45)
    ax.set_yticks([])
    ax.set_xlabel('mean change in PSNR over the pretrained model (dB)', fontsize=7.5, labelpad=2)
    S.clean(ax, 'x')
    ax.spines['left'].set_visible(False)
    S.save(fig, 'controls', a.out)
    for (g, key, lab, _), m in zip(ROWS, means):
        print(f"  {g:22s} {lab:24s} {m:+.2f} dB  improves {sum(d > 0 for d in data[key][1])}/5")


if __name__ == '__main__':
    main()
