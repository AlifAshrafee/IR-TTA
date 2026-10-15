"""Results tables: one per (backbone, source setting), plus a cross-backbone summary.

    python tools/aggregate.py results/suite        # -> results/suite/summary.csv
    python tools/table.py results/suite            # -> results/suite/table_<setting>_<backbone>.md, table_backbones.md

Per-backbone table: methods as rows, targets as columns (PSNR, dB): the test sets, then the validation sets
(``*_val``; O-HAZY is both, marked *). ``Avg test`` = mean over the test columns, shown only when the row has all
of them. Bold = best non-input value per column. The Input row is shared by all backbones (it is run once). The
"Reported" row is the co-author's number for that backbone (``configs/sota_<setting>.csv``), a reproduction check
for our own Source row.

Summary (``table_backbones.md``): per backbone and setting, Avg test of the source, the source applied twice and
our method (``--tag``/``--families``/``--exposure_match`` select which runs), and on how many test sets ours beats
the source.
"""
import argparse
import csv
import math
import os
from collections import OrderedDict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = {
    '3task': ['Rain100H', 'SOTS_Indoor', 'DenseHaze', 'OHAZY', 'Kodak24_Poisson'],
    '5task': ['Rain100H', 'SOTS_Indoor', 'DenseHaze', 'OHAZY', 'Kodak24_Poisson',
              'HIDE', 'LOLv2', 'LOLv2_Syn'],
}
VALS = {'3task': ['Rain100H_val'], '5task': ['Rain100H_val', 'HIDE_val', 'LOLv2_val']}
SHORT = {'Rain100H': 'Rain100H', 'SOTS_Indoor': 'SOTS-In', 'DenseHaze': 'DenseHaze', 'OHAZY': 'O-HAZY*',
         'Kodak24_Poisson': 'Kodak', 'HIDE': 'HIDE', 'LOLv2': 'LOLv2', 'LOLv2_Syn': 'LOLv2-S',
         'Rain100H_val': 'Rain val', 'HIDE_val': 'HIDE val', 'LOLv2_val': 'LOLv2 val'}
NAMES = {'promptir': 'PromptIR', 'adair': 'AdaIR', 'bioir': 'Bio-IR', 'moceir': 'MoCE-IR', 'mirage': 'MIRAGE',
         'r2r': 'R2R'}
ORDER = ['Input', 'Source', 'Reported', 'Source flow', 'DCTTA', 'Ours']


def num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def label(r):
    m = r['method']
    if m == 'identity':
        return 'Input'
    if m == 'source':
        return 'Source'
    if m == 'flow':
        return f"Source flow (a={num(r['ext_alpha']):g}, x{int(float(r['n_apply']))})"
    if m == 'dctta':
        return 'DCTTA'
    if m == 'transfer':
        a = num(r.get('eval_alpha')) or 1.0
        em = r.get('exposure_match') or 'fit'
        extra = '' if em == 'fit' else f", em {em}"
        return f"Ours {r.get('tag') or 'v0.1'} ({r['families']}{extra}, mix {a:g})"
    return m


def sort_key(lbl):
    for i, p in enumerate(ORDER):
        if lbl.startswith(p):
            return (i, lbl)
    return (len(ORDER), lbl)


def _rows_for(summary_rows, setting, backbone, cols):
    out = []
    for r in summary_rows:
        if r['source'] != setting or r['target'] not in cols or r.get('regime') != 'transductive':
            continue
        if r['method'] == 'identity' or (r.get('backbone') or 'promptir') == backbone:
            out.append(r)
    return out


def build(summary_rows, setting, backbone):
    cols = TESTS[setting] + [v for v in VALS[setting] if v not in TESTS[setting]]
    table = OrderedDict()   # label -> {target: (mean, std, n)}
    for r in _rows_for(summary_rows, setting, backbone, cols):
        lbl = label(r)
        cell = table.setdefault(lbl, {})
        if r['target'] in cell:  # two configs collapse to one label: keep them apart
            lbl = f"{lbl} [{r['cfg']}]"
            cell = table.setdefault(lbl, {})
        cell[r['target']] = (num(r['psnr_mean']), num(r['psnr_std']), int(float(r['n_seeds'])))
    sota_path = os.path.join(REPO, 'configs', f'sota_{setting}.csv')
    if os.path.exists(sota_path):
        with open(sota_path) as f:
            for s in csv.DictReader(f):
                if s['method'] == NAMES[backbone]:
                    table['Reported (co-author)'] = {t: (num(v), None, 1) for t, v in s.items() if t in cols}
    rows = sorted(table.items(), key=lambda kv: sort_key(kv[0]))
    return cols, rows


def _is_input(lbl):
    return lbl.startswith('Input')


def _avg(cell, tests):
    tv = [cell[c][0] for c in tests if c in cell and cell[c][0] is not None]
    return f"{sum(tv) / len(tv):.2f}" if len(tv) == len(tests) else ''


def render(cols, rows, setting):
    tests = TESTS[setting]
    best = {}
    for c in cols:
        vals = [cell[c][0] for lbl, cell in rows if c in cell and cell[c][0] is not None and not _is_input(lbl)]
        best[c] = max(vals) if vals else None
    head = ['Method'] + [SHORT.get(c, c) for c in cols] + ['Avg test']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for lbl, cell in rows:
        out = [lbl]
        for c in cols:
            if c not in cell or cell[c][0] is None:
                out.append('')
                continue
            m, sd, n = cell[c]
            txt = f"{m:.2f}" + (f"±{sd:.2f}" if n > 1 and sd else '')
            out.append(f"**{txt}**" if best[c] is not None and abs(m - best[c]) < 1e-9 and not _is_input(lbl) else txt)
        out.append(_avg(cell, tests))
        lines.append('| ' + ' | '.join(out) + ' |')
    return '\n'.join(lines)


def summary_table(summary, args):
    head = ['Setting', 'Backbone', 'Source', 'Source x2', 'Ours mix 0.5', 'Ours mix 1', 'Best ours - source',
            'Ours > source (mix 1)']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for setting in ('3task', '5task'):
        tests = TESTS[setting]
        for bb in NAMES:
            cols = {}
            for r in _rows_for(summary, setting, bb, tests):
                key = None
                if r['method'] == 'source':
                    key = 'src'
                elif r['method'] == 'flow' and num(r['ext_alpha']) == 1 and int(float(r['n_apply'])) == 2:
                    key = 'x2'
                elif (r['method'] == 'transfer' and (r.get('tag') or '') == args.tag and r['families'] == args.families
                      and (r.get('exposure_match') or 'fit') == args.exposure_match):
                    key = f"ours{num(r.get('eval_alpha')) or 1.0:g}"
                if key:
                    cols.setdefault(key, {})[r['target']] = (num(r['psnr_mean']), None, 1)
            if 'src' not in cols:
                continue
            avg = {k: _avg(v, tests) for k, v in cols.items()}
            src, o1 = cols['src'], cols.get('ours1', {})
            both = [t for t in tests if t in o1 and t in src and o1[t][0] is not None and src[t][0] is not None]
            n_win = sum(o1[t][0] > src[t][0] for t in both)
            best = [float(avg[k]) for k in ('ours0.5', 'ours1') if avg.get(k)]
            delta = f"{max(best) - float(avg['src']):+.2f}" if best and avg.get('src') else ''
            lines.append('| ' + ' | '.join([setting, NAMES[bb], avg.get('src', ''), avg.get('x2', ''),
                                            avg.get('ours0.5', ''), avg.get('ours1', ''), delta,
                                            f"{n_win}/{len(both)}" if both else '']) + ' |')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', help='folder holding summary.csv (tools/aggregate.py output)')
    p.add_argument('--tag', default='v02', help="summary table: our runs' --tag ('' = the untagged v0.1 runs)")
    p.add_argument('--families', default='compose', help="summary table: our runs' --families")
    p.add_argument('--exposure_match', default='fit', help="summary table: our runs' --exposure_match")
    args = p.parse_args()
    with open(os.path.join(args.root, 'summary.csv')) as f:
        summary = list(csv.DictReader(f))
    for setting in ('3task', '5task'):
        for bb in NAMES:
            cols, rows = build(summary, setting, bb)
            if not any(lbl == 'Source' for lbl, _ in rows):
                continue
            md = (f"## {NAMES[bb]} {setting} source -- PSNR (dB)\n\nTest sets, then validation sets "
                  f"(* O-HAZY is both). Bold = best non-input value per column.\n\n" + render(cols, rows, setting))
            path = os.path.join(args.root, f'table_{setting}_{bb}.md')
            with open(path, 'w') as f:
                f.write(md + '\n')
            print(md + '\n')
            print(f"-> {path}\n")
    md = (f"## Across backbones -- Avg test PSNR (dB)\n\nOurs = tag '{args.tag}', families {args.families}, "
          f"exposure match {args.exposure_match}.\n\n" + summary_table(summary, args))
    path = os.path.join(args.root, 'table_backbones.md')
    with open(path, 'w') as f:
        f.write(md + '\n')
    print(md + '\n')
    print(f"-> {path}")


if __name__ == '__main__':
    main()
