"""Average PSNR over all test sets, pretrained -> + ours, for every backbone and setting (the abstract's claim as one
picture: twelve arrows, all up). Reads results/suite/table_main_{3task,5task}.md.

    python paper_figures/gain_per_backbone.py               # -> results/figures/gain_per_backbone.{pdf,png}
    python paper_figures/gain_per_backbone.py --with_tta    # also mark + existing TTA (hollow)

No per-set numbers, no thumbnails: one grey dot (pretrained), one blue dot (ours) and the connecting line per
backbone; the two settings side by side. Suited to the experiments section (summary of the main tables) or, if a
numeric panel is wanted at all, to the intro.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def averages(vals, keys):
    out = {}
    for s in ('3task', '5task'):
        for bb in S.NAMES:
            ks = [k for k in keys if k[0] == bb and k[1] == s]
            if ks:
                out[(bb, s)] = {m: statistics.fmean(vals[(m, *k)] for k in ks) for m in ('source', 'dctta', 'ours')}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--with_tta', action='store_true')
    p.add_argument('--tta_label', default='+ existing TTA')
    p.add_argument('--tables', default=os.path.join(S.REPO, 'results', 'suite'), help=argparse.SUPPRESS)
    p.add_argument('--out', default=os.path.join(S.REPO, 'results', 'figures'), help=argparse.SUPPRESS)
    a = p.parse_args()
    S.setup()
    vals = S.load_main(a.tables)
    av = averages(vals, S.pairs(vals))
    settings = [s for s in ('3task', '5task') if any(k[1] == s for k in av)]
    fig, axes = plt.subplots(1, len(settings), figsize=(S.COL_W, 1.6), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    axes = list(axes) if len(settings) > 1 else [axes]
    for ax, s in zip(axes, settings):
        bbs = [bb for bb in S.NAMES if (bb, s) in av]
        for i, bb in enumerate(bbs):
            v = av[(bb, s)]
            ax.plot([i, i], [v['source'], v['ours']], color=S.C_OURS, linewidth=1.3, zorder=1, solid_capstyle='round')
            ax.plot([i], [v['source']], marker='o', markersize=4, color=S.C_SOURCE, markeredgecolor='white',
                    markeredgewidth=0.5, zorder=2)
            ax.plot([i], [v['ours']], marker='o', markersize=4.5, color=S.C_OURS, markeredgecolor='white',
                    markeredgewidth=0.5, zorder=3)
            if a.with_tta:
                ax.plot([i], [v['dctta']], marker='o', markersize=4, markerfacecolor='white', color=S.C_DCTTA,
                        markeredgewidth=1.0, zorder=2)
        ax.set_xticks(range(len(bbs)))
        ax.set_xticklabels([S.NAMES[bb] for bb in bbs], rotation=40, ha='right', rotation_mode='anchor', fontsize=6.5)
        ax.set_xlim(-0.6, len(bbs) - 0.4)
        ax.set_title({'3task': 'three-task models', '5task': 'five-task models'}[s], fontsize=7.5, pad=3)
        ax.tick_params(axis='x', length=0)
        ax.grid(axis='y', color=S.GRID, linewidth=0.4)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel('mean PSNR over\nall test sets (dB)')
    h = [Line2D([], [], marker='o', linestyle='', markersize=4, color=S.C_SOURCE, label='pretrained'),
         Line2D([], [], marker='o', linestyle='', markersize=4.5, color=S.C_OURS, label='+ ours')]
    if a.with_tta:
        h.insert(1, Line2D([], [], marker='o', linestyle='', markersize=4, markerfacecolor='white', color=S.C_DCTTA,
                           markeredgewidth=1.0, label=a.tta_label))
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=len(h), frameon=False,
               handletextpad=0.3, columnspacing=1.2)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, 'gain_per_backbone' + ('_tta' if a.with_tta else ''))
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=300)
    for (bb, s), v in av.items():
        print(f"{S.NAMES[bb]:9s} {s}: pretrained {v['source']:.2f}  ours {v['ours']:.2f} ({v['ours'] - v['source']:+.2f})"
              + (f"  tta {v['dctta']:.2f}" if a.with_tta else ''))
    print(f"-> {base}.pdf")


if __name__ == '__main__':
    main()
