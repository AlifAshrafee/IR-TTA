"""Fig. 2 (right), option A: how much does test-time adaptation change the pretrained model, per degradation family?

    python paper_figures/summary_bars.py

One group of two bars per degradation family. The zero line is the pretrained model; a bar is the mean change in
PSNR over backbones (and both settings) for existing AiO TTA and for ours, and each dot is one backbone x setting
(mean over that family's test sets). Values below the y-range are drawn at the bottom edge with a down-pointing
marker, without a number. No thumbnails and almost no text inside the plot.

Reads the paper tables (results/suite/table_main_{3task,5task}.md, from `bash scripts/run.sh tables`), so the
figure uses exactly the numbers the tables print. Writes into results/figures/:
  summary_bars.pdf / .png   the plot
  figure_stats.md           numbers for the caption, abstract and intro \\TODO s (cells harmed, average gains, ...)

Options (all optional):
  --by family|set     one group per degradation family (default) or one per test set
  --setting both|3task|5task   which pretrained models to pool (default both)
  --ylim LO HI        y-range in dB (default -4 6)
  --tta_label TEXT    legend name of the existing method (default "Existing AiO TTA"; cite it in the caption)
"""
import argparse
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

FAMILY_SETS = [('Rain', ['Rain100H']), ('Haze', ['SOTS_Indoor', 'DenseHaze', 'OHAZY']), ('Noise', ['Kodak24_Poisson']),
               ('Blur', ['HIDE', 'DPDD']), ('Low light', ['LOLv2_Syn'])]
METHODS = ('dctta', 'ours')


def deltas(vals, keys, m):
    return np.array([vals[(m, bb, s, t)] - vals[('source', bb, s, t)] for bb, s, t in keys])


def groups(by):
    if by == 'family':
        return FAMILY_SETS
    return [(S.SHORT[t], [t]) for t in S.TESTS['5task']]


def group_dots(vals, keys, sets, m):
    """One value per backbone x setting: mean change over the group's test sets present in that setting."""
    out = []
    for s in ('3task', '5task'):
        for bb in S.NAMES:
            ks = [(bb, s, t) for t in sets if (bb, s, t) in keys]
            if ks:
                out.append(statistics.fmean(vals[(m, *k)] - vals[('source', *k)] for k in ks))
    return np.array(out)


def stats_md(vals, keys, label):
    L = [f"# Summary statistics ({len(keys)} backbone x setting x test-set cells; PSNR change vs pretrained, dB)\n",
         "From the 2-decimal paper tables, so they match what the tables print.\n",
         "| | below pretrained | below by > 1 dB | min | median | mean | max |", "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        d = deltas(vals, keys, m)
        L.append(f"| {label[m]} | {(d < 0).sum()}/{len(d)} | {(d < -1).sum()}/{len(d)} | {d.min():+.2f} | "
                 f"{np.median(d):+.2f} | {d.mean():+.2f} | {d.max():+.2f} |")
    d_o, d_d = deltas(vals, keys, 'ours'), deltas(vals, keys, 'dctta')
    L += ["", f"Ours above pretrained in {(d_o > 0).sum()}/{len(d_o)} cells; above {label['dctta']} in "
          f"{(d_o > d_d).sum()}/{len(d_o)}.", "", "## Per family (mean over backbone x setting; [min, max])", "",
          f"| family | n | {label['dctta']} | Ours |", "|---|---|---|---|"]
    for name, sets in FAMILY_SETS:
        r = [group_dots(vals, keys, sets, m) for m in METHODS]
        if len(r[0]):
            L.append(f"| {name} | {len(r[0])} | " + " | ".join(f"{x.mean():+.2f} [{x.min():+.2f}, {x.max():+.2f}]"
                                                             for x in r) + " |")
    L += ["", "## Average gain over the pretrained model (the tables' Avg column)", "",
          f"| backbone | setting | {label['dctta']} | Ours |", "|---|---|---|---|"]
    gains = {m: [] for m in METHODS}
    for s in ('3task', '5task'):
        for bb in S.NAMES:
            ks = [k for k in keys if k[0] == bb and k[1] == s]
            if not ks:
                continue
            src = statistics.fmean(vals[('source', *k)] for k in ks)
            g = {m: statistics.fmean(vals[(m, *k)] for k in ks) - src for m in METHODS}
            for m in METHODS:
                gains[m].append((s, g[m]))
            L.append(f"| {S.NAMES[bb]} | {s} | {g['dctta']:+.2f} | {g['ours']:+.2f} |")
    L.append("")
    for m in METHODS:
        for s in ('3task', '5task', None):
            g = [x for ss, x in gains[m] if s is None or ss == s]
            if g:
                L.append(f"- {label[m]}, {s or 'all'}: {min(g):+.2f} to {max(g):+.2f} dB (mean {statistics.fmean(g):+.2f})")
    return '\n'.join(L) + '\n'


def plot(vals, keys, args, label):
    gs = [(n, sets) for n, sets in groups(args.by) if any(k[2] in sets for k in keys)]
    lo, hi = args.ylim
    w = 0.34
    fig, ax = plt.subplots(figsize=(S.COL_W, 1.85))
    ax.axhline(0, color=S.INK, linewidth=0.8, zorder=2)
    ax.annotate('pretrained', xy=(len(gs) - 0.4, 0), xytext=(2, 0), textcoords='offset points', ha='left',
                va='center', fontsize=6.5, color=S.INK2, annotation_clip=False)
    rng = np.random.default_rng(0)
    for i, (_, sets) in enumerate(gs):
        for j, m in enumerate(METHODS):
            x = i + (j - 0.5) * (w + 0.04)
            dots = group_dots(vals, keys, sets, m)
            mean = float(dots.mean())
            ax.bar(x, np.clip(mean, lo, hi), width=w, color=S.C_DCTTA if m == 'dctta' else S.C_OURS,
                   alpha=0.85, linewidth=0, zorder=1, label=label[m] if i == 0 else None)
            if mean < lo:   # off-scale bar: arrow at the bottom edge instead of a number
                ax.plot([x], [lo], marker='v', markersize=4.5, color=S.INK, zorder=4, clip_on=False)
            jit = (rng.random(len(dots)) - 0.5) * w * 0.55
            inr = dots >= lo
            ax.scatter(x + jit[inr], dots[inr], s=5, color='white', edgecolor=S.INK, linewidth=0.45, zorder=3)
            if (~inr).any() and mean >= lo:   # off-scale dots of an on-scale bar: small marks on the bottom edge
                ax.scatter(x + jit[~inr], np.full((~inr).sum(), lo), s=6, marker='v', color=S.INK2, linewidth=0,
                           zorder=3, clip_on=False)
    ax.set_xticks(range(len(gs)))
    ax.set_xticklabels([n for n, _ in gs])
    ax.set_xlim(-0.6, len(gs) - 0.4)
    ax.set_ylim(lo, hi)
    ax.set_ylabel('PSNR change (dB)')
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='y', color=S.GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    for sp in ('top', 'right', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2, handlelength=1.0,
              handletextpad=0.4, columnspacing=1.2, borderaxespad=0.2)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tables', default=os.path.join(S.REPO, 'results', 'suite'), help=argparse.SUPPRESS)
    p.add_argument('--out', default=os.path.join(S.REPO, 'results', 'figures'), help=argparse.SUPPRESS)
    p.add_argument('--by', default='family', choices=['family', 'set'])
    p.add_argument('--setting', default='both', choices=['both', '3task', '5task'])
    p.add_argument('--ylim', type=float, nargs=2, default=(-4.0, 6.0))
    p.add_argument('--tta_label', default='Existing AiO TTA')
    args = p.parse_args()
    S.setup()
    vals = S.load_main(args.tables)
    keys = [k for k in S.pairs(vals) if args.setting == 'both' or k[1] == args.setting]
    if not keys:
        raise SystemExit("no complete backbone x set cells in the tables")
    label = {'dctta': args.tta_label, 'ours': 'Ours'}
    os.makedirs(args.out, exist_ok=True)
    md = stats_md(vals, keys, label)
    with open(os.path.join(args.out, 'figure_stats.md'), 'w') as f:
        f.write(md)
    fig = plot(vals, keys, args, label)
    base = os.path.join(args.out, 'summary_bars' + ('' if args.by == 'family' else '_sets'))
    fig.savefig(base + '.pdf')
    fig.savefig(base + '.png', dpi=300)
    print(md)
    print(f"-> {base}.pdf\n-> {os.path.join(args.out, 'figure_stats.md')}")


if __name__ == '__main__':
    main()
