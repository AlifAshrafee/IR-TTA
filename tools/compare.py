"""Our variants against ours (F1): PSNR change per backbone and test set (rain sets in Y-PSNR).

    python tools/aggregate.py results/suite
    python tools/compare.py results/suite                       # V1fill V2strat V3content V4all vs F1
    python tools/compare.py results/suite --tags V2strat --seed 23

writes results/suite/compare_<setting>.md. A cell is the variant's change vs F1 (dB); '=' marks a V1fill cell
taken from F1 (sets with >= 128 images: one pass already gives 256 crops, the run is bit-identical). The last
columns: mean change and wins / losses beyond --tol dB.
"""
import argparse
import os

from table import NAMES, SHORT, TESTS, metric, num, read_csv

SAME_AS_F1 = {'V1fill': {'SOTS_Indoor', 'HIDE'}}


def index(runs, seed):
    """{(tag, backbone, setting, target): psnr} of our runs at ``seed``."""
    out = {}
    for r in runs:
        if r['method'] != 'transfer' or str(r.get('seed')) != str(seed) or (num(r.get('eval_alpha')) or 1.0) != 1.0:
            continue
        p = metric(r, prefix=None)[0]
        if p is not None:
            out[(r.get('tag') or '', r.get('backbone') or 'promptir', r['source'], r['target'])] = p
    return out


def table(idx, setting, tags, ref, tol):
    cols = TESTS[setting]
    head = ['Backbone', 'Variant'] + [SHORT.get(c, c) for c in cols] + ['Mean Δ', 'W / L']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for bb, name in NAMES.items():
        base = [idx.get((ref, bb, setting, c)) for c in cols]
        if not any(v is not None for v in base):
            continue
        lines.append(f"| {name} | {ref} | " + ' | '.join('' if v is None else f"{v:.2f}" for v in base) + ' | | |')
        for t in tags:
            cells, ds = [], []
            for c, b in zip(cols, base):
                v = idx.get((t, bb, setting, c))
                same = v is None and c in SAME_AS_F1.get(t, ()) and b is not None
                if same:
                    cells.append('=')
                    ds.append(0.0)
                elif v is None or b is None:
                    cells.append('')
                else:
                    cells.append(f"{v - b:+.2f}")
                    ds.append(v - b)
            if not ds:
                continue
            full = len(ds) == len(cols)
            mean = f"{sum(ds) / len(ds):+.2f}" + ('' if full else f" ({len(ds)}/{len(cols)})")
            wl = f"{sum(d > tol for d in ds)} / {sum(d < -tol for d in ds)}"
            lines.append(f"| | {t} | " + ' | '.join(cells) + f" | {mean} | {wl} |")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', help='results folder with runs.csv (tools/aggregate.py)')
    p.add_argument('--tags', nargs='+', default=['V1fill', 'V2strat', 'V3content', 'V4all'])
    p.add_argument('--ref', default='F1')
    p.add_argument('--seed', type=int, default=23)
    p.add_argument('--tol', type=float, default=0.05, help='|change| above this counts as a win / loss (dB)')
    args = p.parse_args()
    idx = index(read_csv(os.path.join(args.root, 'runs.csv')), args.seed)
    for s in ('3task', '5task'):
        md = (f"## {s}: change vs {args.ref} (dB, seed {args.seed}); rain sets in Y-PSNR\n\n"
              + table(idx, s, args.tags, args.ref, args.tol))
        path = os.path.join(args.root, f'compare_{s}.md')
        with open(path, 'w') as f:
            f.write(md + '\n')
        print(md + f"\n\n-> {path}\n")


if __name__ == '__main__':
    main()
