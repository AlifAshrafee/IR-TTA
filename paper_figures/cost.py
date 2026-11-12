"""Fig. `supp:cost` (supplementary): cost of adaptation against DCTTA, five-task setting.

    python paper_figures/cost.py                  # -> results/figures/cost.{pdf,png}  (text width)

Reads results/suite/table_cost.md (time and adapted parameters, mean over the five-task target sets) and
table_main_5task.md (mean change in PSNR over the pretrained model).

One row per backbone and one panel per quantity (time, adapted parameters, change in PSNR); DCTTA and ours joined by
a line, every value printed. Each quantity has its own axis, so nothing has to be read off a marker size. (A bubble
chart of the same numbers was tried and dropped: four of DCTTA's six points fall within 1200-1700 s and their labels
collide, and marker areas still needed the parameter counts printed next to them.)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

LABEL = {'dctta': '+ DCTTA', 'ours': '+ Ours'}
COLOR = {'dctta': S.C_DCTTA, 'ours': S.C_OURS}


def load(tables):
    """{backbone: {method: dict(time, params_M, gain)}} in table order."""
    inv = {v: k for k, v in S.NAMES.items()}
    out = {}
    for bbn, meth, t, _mem, par in S.md_rows(os.path.join(tables, 'table_cost.md'))[1:]:
        out.setdefault(inv[bbn], {})[meth.lower()] = dict(time=float(t), params=float(par.replace(',', '')) / 1e6)
    avg = {}   # the tables' Avg. column, so the changes equal the ones printed in the paper
    for cells in S.md_rows(os.path.join(tables, 'table_main_5task.md'))[1:]:
        name, _, meth = cells[0].partition(' + ')
        avg[(inv[name], (meth or 'source').lower())] = float(cells[-1])
    for bb, d in out.items():
        for m in d:
            d[m]['gain'] = avg[(bb, m)] - avg[(bb, 'source')]
    return out


def legend(fig, y=0.995):
    h = [Line2D([], [], marker=S.MARK[m], linestyle='', markersize=4.5, color=COLOR[m], label=LABEL[m])
         for m in ('dctta', 'ours')]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, y), ncol=2, frameon=False, handletextpad=0.2,
               columnspacing=1.6, borderpad=0)


def panels(d, a):
    bbs = list(d)
    fig, axes = plt.subplots(1, 3, figsize=(S.TEXT_W, 1.75), sharey=True,
                             gridspec_kw=dict(wspace=0.10, left=0.085, right=0.985, bottom=0.2, top=0.87))
    spec = [('time', 'adaptation time per target set (s)', '{:.0f}'),
            ('params', 'adapted parameters (M)', '{:.1f}'),
            ('gain', 'mean change in PSNR over pretrained (dB)', None)]
    for ax, (k, xl, f) in zip(axes, spec):
        lo = min(min(d[b][m][k] for m in LABEL) for b in bbs)
        hi = max(max(d[b][m][k] for m in LABEL) for b in bbs)
        lo = min(lo, 0.0)
        span = hi - lo
        for i, b in enumerate(bbs):
            v = {m: d[b][m][k] for m in LABEL}
            ax.plot([v['dctta'], v['ours']], [i, i], color=S.C_MUTED, linewidth=1.2, zorder=1,
                    solid_capstyle='round')
            for m in LABEL:
                ax.plot(v[m], i, marker=S.MARK[m], markersize=4.8, color=COLOR[m], markeredgecolor='white',
                        markeredgewidth=0.5, zorder=3)
                right = v[m] >= v['ours' if m == 'dctta' else 'dctta']
                ax.text(v[m] + (0.035 if right else -0.035) * span, i, f.format(v[m]) if f else S.fmt(v[m]),
                        va='center', ha='left' if right else 'right', fontsize=6.5, color=S.INK)
        if k == 'gain':
            ax.axvline(0, color=S.INK2, linewidth=0.6, zorder=0)
        ax.set_xlim(lo - (0.16 if k == 'gain' else 0.07) * span, hi + 0.17 * span)
        ax.set_xlabel(xl, labelpad=2)
        ax.set_ylim(len(bbs) - 0.5, -0.5)
        ax.set_yticks(range(len(bbs)))
        ax.set_yticklabels([S.NAMES[b] for b in bbs], fontsize=7.5, color=S.INK)
        ax.tick_params(axis='y', length=0)
        S.clean(ax, 'x')
        ax.spines['left'].set_visible(False)
    legend(fig)
    return fig, 'cost'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tables', default=S.TABLES, help=argparse.SUPPRESS)
    p.add_argument('--out', default=S.FIGS, help=argparse.SUPPRESS)
    a = p.parse_args()
    S.setup()
    d = load(a.tables)
    fig, name = panels(d, a)
    S.save(fig, name, a.out)
    for b, v in d.items():
        t, o = v['dctta'], v['ours']
        print(f"  {S.NAMES[b]:9s} time {t['time']:.0f} -> {o['time']:.0f} s ({t['time'] / o['time']:.1f}x)   "
              f"params {t['params']:.1f} -> {o['params']:.1f} M   gain {t['gain']:+.2f} -> {o['gain']:+.2f} dB")


if __name__ == '__main__':
    main()
