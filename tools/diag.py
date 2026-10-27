"""Photometric vs structural share of each method's gain (diagnostic; uses the GT).

    python tools/aggregate.py results/suite && python tools/diag.py results/suite
    python tools/diag.py results/suite --setting 5task --targets HIDE LOLv2_Syn

Per method and target: change vs the source in Y-PSNR (dY) and in affine-corrected Y-PSNR (dAff, ``psnr_aff``:
Y-PSNR after the per-image least-squares map a * Y_out + b -> Y_gt). dAff is the gain that survives removing the
global brightness / contrast error, i.e. structural restoration (streaks, texture, edges); dY - dAff is the part a
global photometric correction would have given. Only runs made after ``psnr_aff`` was recorded have it
(``bash scripts/run.sh diag`` re-scores the PromptIR rows that matter).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table import NAMES, SHORT, _rows_for, label, num, sort_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--setting', default='3task')
    ap.add_argument('--backbone', default='promptir')
    ap.add_argument('--targets', nargs='+', default=['Rain100H', 'SOTS_Indoor', 'DenseHaze', 'Kodak24_Poisson'])
    args = ap.parse_args()
    with open(os.path.join(args.root, 'summary.csv')) as f:
        summary = list(csv.DictReader(f))
    cells, src = {}, {}
    for r in _rows_for(summary, args.setting, args.backbone, args.targets):
        y, a = num(r.get('psnr_y_mean')), num(r.get('psnr_aff_mean'))
        if y is None or a is None:
            continue
        if r['method'] == 'source':
            src[r['target']] = (y, a)
        cells.setdefault(label(r), {})[r['target']] = (y, a)
    head = ['Method'] + [f"{SHORT.get(t, t)} {k}" for t in args.targets for k in ('dY', 'dAff')]
    lines = [f"## {NAMES[args.backbone]} {args.setting}: change vs source, Y-PSNR (dY) and affine-corrected Y-PSNR "
             f"(dAff = structural part)\n", '| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    lines.insert(1, 'Source (Y / affine-corrected Y): ' + ', '.join(
        f"{SHORT.get(t, t)} {src[t][0]:.2f} / {src[t][1]:.2f}" for t in args.targets if t in src) + '\n')
    for lbl, cell in sorted(cells.items(), key=lambda kv: sort_key(kv[0])):
        if lbl == 'Source':
            continue
        out = [lbl]
        for t in args.targets:
            if t in cell and t in src:
                out += [f"{cell[t][0] - src[t][0]:+.2f}", f"{cell[t][1] - src[t][1]:+.2f}"]
            else:
                out += ['', '']
        lines.append('| ' + ' | '.join(out) + ' |')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
