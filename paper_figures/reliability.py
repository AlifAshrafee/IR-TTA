"""Fig. `fig:reliability` (supplementary; main paper if space): every result of the main tables as one mark.

    python paper_figures/reliability.py           # -> results/figures/reliability.{pdf,png}

Reads results/suite/table_main_{3task,5task}.md. One row per target set; each mark is the change in PSNR over the
pretrained model of one backbone in one setting (12 marks per set present in both settings, 6 otherwise), DCTTA in the
upper lane and ours in the lower. The axis is broken once, with a visible break: the left part holds the large losses
on a compressed scale, so no result is clipped or left out.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

COLOR = {'dctta': S.C_DCTTA, 'ours': S.C_OURS}
LANE = {'dctta': -0.21, 'ours': 0.21}


def dodge(xs, tol, step=0.075, lim=2):
    """Vertical offsets so that marks closer than ``tol`` on x do not sit on top of each other."""
    placed, out = [], {}
    for i in sorted(range(len(xs)), key=lambda j: xs[j]):
        for k in sorted(range(-lim, lim + 1), key=abs):
            if all(abs(xs[i] - px) > tol or pk != k for px, pk in placed):
                break
        placed.append((xs[i], k))
        out[i] = k * step
    return [out[i] for i in range(len(xs))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--brk', type=float, default=-5.0, help='dB: losses beyond this go to the compressed left part')
    p.add_argument('--height', type=float, default=2.75)
    p.add_argument('--tables', default=S.TABLES, help=argparse.SUPPRESS)
    p.add_argument('--out', default=S.FIGS, help=argparse.SUPPRESS)
    a = p.parse_args()
    S.setup()
    vals = S.load_main(a.tables)
    keys = S.pairs(vals)
    delta = {m: {k: vals[(m, *k)] - vals[('source', *k)] for k in keys} for m in COLOR}
    sets = [t for t in S.ORDER if any(k[2] == t for k in keys)]
    lo = min(min(d.values()) for d in delta.values())
    hi = max(max(d.values()) for d in delta.values())
    broken = lo < a.brk

    fig = plt.figure(figsize=(S.COL_W, a.height))
    L, R, B, T = 0.225, 0.985, 0.13, 0.925
    wl = 0.20 * (R - L) if broken else 0.0
    gap = 0.025 if broken else 0.0
    axr = fig.add_axes([L + wl + gap, B, R - L - wl - gap, T - B])
    axl = fig.add_axes([L, B, wl, T - B], sharey=axr) if broken else None
    xr = (a.brk if broken else min(lo, 0) - 0.5, hi + 0.6)
    xl = (5 * (lo // 5), a.brk - 1.0) if broken else None
    for r, t in enumerate(sets):
        if r % 2 == 0:
            for ax in (axr, axl):
                if ax is not None:
                    ax.axhspan(r - 0.5, r + 0.5, color='#f5f4f1', linewidth=0, zorder=0)
        for m in COLOR:
            ks = [k for k in keys if k[2] == t]
            xs = [delta[m][k] for k in ks]
            for ax, rng in ((axr, xr), (axl, xl)):
                if ax is None:
                    continue
                sel = [x for x in xs if (x >= a.brk if ax is axr else x < a.brk)] if broken else xs
                if not sel:
                    continue
                off = dodge(sel, tol=0.021 * (rng[1] - rng[0]) * (0.76 / (0.76 if ax is axr else 0.20)))
                ax.scatter(sel, [r + LANE[m] + o for o in off], s=13, marker=S.MARK[m], color=COLOR[m],
                           edgecolor='white', linewidth=0.4, zorder=3, clip_on=False)
    axr.axvline(0, color=S.INK2, linewidth=0.6, zorder=1)
    axr.set_xlim(*xr)
    axr.set_ylim(len(sets) - 0.5, -0.5)
    S.clean(axr, 'x')
    axr.tick_params(axis='y', length=0)
    left = axl if broken else axr
    left.set_yticks(range(len(sets)))
    left.set_yticklabels([S.PRETTY[t] for t in sets], fontsize=7.5, color=S.INK)
    left.tick_params(axis='y', length=0)
    if broken:
        axl.set_xlim(*xl)
        axl.set_xticks([x for x in (-20, -10) if xl[0] <= x <= xl[1]])
        S.clean(axl, 'x')
        axl.spines['left'].set_visible(False)
        axr.spines['left'].set_visible(False)
        plt.setp(axr.get_yticklabels(), visible=False)
        kw = dict(marker=[(-1, -1.8), (1, 1.8)], markersize=5, linestyle='none', color=S.INK2, mew=0.7,
                  clip_on=False)
        axl.plot([1], [0], transform=axl.transAxes, **kw)   # break marks on the axis line
        axr.plot([0], [0], transform=axr.transAxes, **kw)
    else:
        axr.spines['left'].set_visible(False)
    fig.text((L + R) / 2, 0.012, 'change in PSNR over the pretrained model (dB)', ha='center', va='bottom',
             fontsize=8)
    n = len(keys)
    lab = {m: f"{'+ DCTTA' if m == 'dctta' else '+ Ours'} (improves {sum(v > 0 for v in delta[m].values())} of {n})"
           for m in COLOR}
    h = [Line2D([], [], marker=S.MARK[m], linestyle='', markersize=4.5, color=COLOR[m], label=lab[m]) for m in COLOR]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=((L + R) / 2 - 0.06, 1.0), ncol=2, frameon=False,
               handletextpad=0.1, columnspacing=1.0, borderpad=0)
    S.save(fig, 'reliability', a.out)
    for m in COLOR:
        v = list(delta[m].values())
        print(f"  {m:5s}: improves {sum(x > 0 for x in v)}/{n}, worst {min(v):+.2f} dB, best {max(v):+.2f} dB, "
              f"{sum(x < -1 for x in v)} below -1 dB, {sum(x < -6 for x in v)} below -6 dB")


if __name__ == '__main__':
    main()
