"""Paper tables from ``tools/aggregate.py`` output (PSNR in dB; rain sets in BT.601 Y-PSNR, the deraining convention).

    python tools/aggregate.py results/suite && python tools/aggregate.py results/oracle
    python tools/table.py results/suite --oracle results/oracle

writes into the results folder:
  table_main_3task.md, table_main_5task.md        test sets; per backbone: pretrained / + DCTTA / + ours
  table_noshift_3task.md, table_noshift_5task.md  the sources' own test distributions (no degradation shift)
  table_controls.md                               PromptIR 3-task: ours vs generic pairs (G1), mismatched
                                                  conditioning (S1), random initialisation (Z500, Z5000, ZG500)
  table_ablation_3task.md, table_ablation_5task.md  PromptIR: ours without each component
  table_ablation_family.md                        all backbones (run.sh ablation): mean change vs the pretrained
                                                  model per degradation family, both settings pooled, plus the
                                                  number of pairs where removing the component lowers PSNR
  table_ablation_backbone.md                      the same, mean over each backbone's pairs (consistency check)
  table_ablation_all_3task.md, ..._5task.md       the same per test set, mean over backbones (supplementary)
  table_oracle.md                                 PromptIR 5-task blur / low-light: ours, right family, real pairs
  table_cost.md                                   adaptation time, memory and adapted parameters (--count:
                                                  the run.sh count runs, parameters that receive a gradient)
Options: --input adds the unprocessed input row, --ssim PSNR/SSIM cells, --std seed std where n > 1.
"""
import argparse
import csv
import json
import math
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {'promptir': 'PromptIR', 'adair': 'AdaIR', 'bioir': 'Bio-IR', 'moceir': 'MoCE-IR', 'mirage': 'MIRAGE',
         'r2r': 'R2R'}
TESTS = {'3task': ['Rain100H', 'SOTS_Indoor', 'DenseHaze', 'OHAZY', 'Kodak24_Poisson'],
         '5task': ['Rain100H', 'SOTS_Indoor', 'DenseHaze', 'OHAZY', 'Kodak24_Poisson', 'HIDE', 'LOLv2_Syn', 'DPDD']}
NOSHIFT = {'3task': ['BSD68_s15', 'BSD68_s25', 'BSD68_s50', 'Rain100L', 'SOTS_Outdoor'],
           '5task': ['BSD68_s25', 'Rain100L', 'SOTS_Outdoor', 'GoPro', 'LOL']}
CONTROL_SETS = TESTS['3task']
CONTROLS = [('F1', 'Ours'), ('G1', 'Generic pairs (no inference)'), ('S1', 'Conditioned on another test set'),
            ('Z500', 'Random init, 500 steps'), ('Z5000', 'Random init, 5000 steps'),
            ('ZG500', 'Random init, generic pairs')]
ABLATION = [('F1', 'Ours'), ('noABC', 'w/o ABC pair selection'), ('noColor', 'w/o colour statistics'),
            ('noExposure', 'w/o exposure matching'), ('noDark', 'w/o under-exposure rule')]
ABLATION_ALL = [('F1', 'Ours'), ('noABC', 'w/o ABC pair selection'), ('noExposure', 'w/o exposure matching'),
                ('noDark', 'w/o under-exposure rule')]
FAMILIES = [('Rain', ['Rain100H']), ('Haze', ['SOTS_Indoor', 'DenseHaze', 'OHAZY']), ('Noise', ['Kodak24_Poisson']),
            ('Blur', ['HIDE', 'DPDD']), ('Low light', ['LOLv2_Syn'])]
ORACLE_SETS = ['HIDE_val', 'LOLv2_Syn_val', 'DPDD']
ORACLE_ROWS = [('F1', 'Ours'), ('F1fam', 'Right family (oracle)'), ('ORpair', 'Real pairs of the target operator')]
SHORT = {'Rain100H': 'Rain100H', 'SOTS_Indoor': 'SOTS-In', 'DenseHaze': 'DenseHaze', 'OHAZY': 'O-HAZY',
         'Kodak24_Poisson': 'Kodak24', 'HIDE': 'HIDE', 'LOLv2_Syn': 'LOLv2-S', 'DPDD': 'DPDD',
         'BSD68_s15': 'BSD68 σ15', 'BSD68_s25': 'BSD68 σ25', 'BSD68_s50': 'BSD68 σ50', 'Rain100L': 'Rain100L',
         'SOTS_Outdoor': 'SOTS-Out', 'GoPro': 'GoPro', 'LOL': 'LOL-v1', 'HIDE_val': 'HIDE (val)',
         'LOLv2_Syn_val': 'LOLv2-S (val)'}


def _y_targets():
    with open(os.path.join(REPO, 'configs', 'datasets.json')) as f:
        reg = json.load(f)
    return {k for k, v in reg.items() if isinstance(v, dict) and v.get('metric') == 'y'}


Y_TARGETS = _y_targets()


def num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def metric(r, prefix=''):
    """(PSNR, SSIM, std) of a summary row (``prefix`` ''), or of a runs row (``prefix`` None: no _mean suffix)."""
    sfx = '' if prefix is None else '_mean'
    k = 'psnr_y' if r['target'] in Y_TARGETS and num(r.get('psnr_y' + sfx)) is not None else 'psnr'
    std = num(r.get(f'{k}_std')) if prefix is not None else None
    return num(r.get(k + sfx)), num(r.get('ssim' + sfx)), std


def row_key(r, tag):
    """Which table row a result belongs to: 'input' / 'source' / 'dctta' / 'ours' / None."""
    m = r['method']
    if m == 'identity':
        return 'input'
    if m == 'source':
        return 'source'
    if m == 'dctta' and not re.sub(r'(^|_)bo=.*$', '', r.get('cfg') or ''):
        return 'dctta'   # the released configuration (backbone options such as MoCE-IR's routing flag allowed)
    if m == 'transfer' and (r.get('tag') or '') == tag and (num(r.get('eval_alpha')) or 1.0) == 1.0:
        return 'ours'
    return None


def collect(summary, tag):
    """{(row, backbone, setting, target): (psnr, ssim, std, n)}; identity rows are backbone '*'."""
    out = {}
    for r in summary:
        k = row_key(r, tag)
        if k is None:
            continue
        bb = '*' if k == 'input' else (r.get('backbone') or 'promptir')
        p, s, sd = metric(r)
        if p is not None:
            out[(k, bb, r['source'], r['target'])] = (p, s, sd, int(float(r.get('n_seeds') or 1)))
    return out


def cell(v, ssim=False, std=False, bold=False):
    if v is None:
        return ''
    p, s, sd, n = v
    txt = f"{p:.2f}" + (f"±{sd:.2f}" if std and n > 1 and sd else '') + (f"/{s:.4f}" if ssim and s is not None else '')
    return f"**{txt}**" if bold else txt


def main_table(vals, setting, cols, args):
    head = ['Method'] + [SHORT.get(c, c) for c in cols] + ['Avg']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    if args.input:
        inp = [vals.get(('input', '*', setting, c)) or vals.get(('input', '*', '3task', c))
               or vals.get(('input', '*', '5task', c)) for c in cols]
        avg = sum(v[0] for v in inp) / len(inp) if all(inp) else None
        lines.append('| Input | ' + ' | '.join(cell(v) for v in inp) + f" | {'' if avg is None else f'{avg:.2f}'} |")
    for bb, name in NAMES.items():
        if not any(('source', bb, setting, c) in vals for c in cols):
            continue
        rows = [('source', name), ('dctta', f'{name} + DCTTA'), ('ours', f'{name} + Ours')]
        best = {c: max((vals[(k, bb, setting, c)][0] for k, _ in rows if (k, bb, setting, c) in vals),
                       default=None) for c in cols}
        for k, label in rows:
            vs = [vals.get((k, bb, setting, c)) for c in cols]
            out = [label] + [cell(v, args.ssim, args.std, v is not None and abs(v[0] - best[c]) < 1e-9)
                             for v, c in zip(vs, cols)]
            out.append(f"{sum(v[0] for v in vs) / len(vs):.2f}" if all(vs) else '')
            lines.append('| ' + ' | '.join(out) + ' |')
    return '\n'.join(lines)


def runs_index(runs, tags):
    """{(tag or method, backbone, setting, target, seed): psnr} from runs.csv (single-seed controls / oracle)."""
    out = {}
    for r in runs:
        if r['method'] in ('source', 'identity'):
            key = r['method']
        elif r['method'] == 'transfer' and (r.get('tag') or '') in tags and (num(r.get('eval_alpha')) or 1.0) == 1.0:
            key = r['tag']
        else:
            continue
        p = metric(r, prefix=None)[0]
        if p is not None:
            out[(key, r.get('backbone') or 'promptir', r['source'], r['target'], str(r.get('seed')))] = p
    return out


def delta_table(idx, rows, setting, cols, seed, bb='promptir'):
    head = ['Variant'] + [SHORT.get(c, c) for c in cols] + ['Mean change']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    src = [next((v for k, v in idx.items() if k[:4] == ('source', bb, setting, c)), None) for c in cols]
    lines.append('| Pretrained | ' + ' | '.join('' if s is None else f"{s:.2f}" for s in src) + ' | |')
    for tag, label in rows:
        vs = [idx.get((tag, bb, setting, c, str(seed))) for c in cols]
        if not any(v is not None for v in vs):
            continue
        d = [None if v is None or s is None else v - s for v, s in zip(vs, src)]
        out = [label] + ['' if v is None else f"{v:.2f} ({dv:+.2f})" for v, dv in zip(vs, d)]
        out.append(f"{sum(d) / len(d):+.2f}" if all(x is not None for x in d) else '')
        lines.append('| ' + ' | '.join(out) + ' |')
    return '\n'.join(lines)


def ablation_pairs(idx, rows, seed):
    """{tag: {(backbone, setting, target): PSNR change vs the pretrained model}}. Pairs are those where ours and the
    pretrained model have runs; each variant covers the pairs it has finished, so the tables can be read while
    run.sh ablation is still going. Once every variant is complete, all rows share the same pairs."""
    src = {k[1:4]: v for k, v in idx.items() if k[0] == 'source'}
    pairs = [(bb, s, t) for bb in NAMES for s in ('3task', '5task') for t in TESTS[s]
             if (bb, s, t) in src and (rows[0][0], bb, s, t, str(seed)) in idx]
    return {tag: {p: idx[(tag,) + p + (str(seed),)] - src[p] for p in pairs if (tag,) + p + (str(seed),) in idx}
            for tag, _ in rows}, pairs


def ablation_tables(idx, rows, seed):
    """Pooled ablation over every backbone and setting with complete runs: (family table, backbone table,
    {setting: per-set table}). Cells are mean changes vs the pretrained model; 'Lower' counts the pairs where the
    variant is below ours (removing the component lowers PSNR)."""
    d, pairs = ablation_pairs(idx, rows, seed)
    if not pairs:
        return None
    mean = lambda xs: f"{sum(xs) / len(xs):+.2f}" if xs else ''  # noqa: E731
    ours = d[rows[0][0]]
    bbs = sorted({p[0] for p in pairs}, key=list(NAMES).index)
    head = ['Variant'] + [f for f, _ in FAMILIES] + ['All', 'Lower than ours']
    fam = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for tag, label in rows:
        cells = [mean([v for p, v in d[tag].items() if p[2] in sets]) for _, sets in FAMILIES]
        lower = '' if tag == rows[0][0] else f"{sum(d[tag][p] < ours[p] for p in d[tag])}/{len(d[tag])}"
        fam.append('| ' + ' | '.join([label] + cells + [mean(list(d[tag].values())), lower]) + ' |')  # noqa: E501
    partial = [label for tag, label in rows if len(d[tag]) < len(pairs)]
    note = (f"Mean over {len(pairs)} backbone-setting-set pairs ({', '.join(NAMES[b] for b in bbs)}; both settings "
            f"pooled; seed {seed})." + (f" INCOMPLETE, averaged over the finished pairs only: {', '.join(partial)} "
                                        f"(see the 'Lower than ours' counts)." if partial else ''))
    head = ['Backbone'] + [label for _, label in rows]
    bt = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for bb in bbs:
        bt.append('| ' + ' | '.join([NAMES[bb]] + [mean([v for p, v in d[tag].items() if p[0] == bb])
                                                    for tag, _ in rows]) + ' |')
    per_set = {}
    for st in ('3task', '5task'):
        cols = [t for t in TESTS[st] if any(p[1] == st and p[2] == t for p in pairs)]
        if not cols:
            continue
        head = ['Variant'] + [SHORT.get(c, c) for c in cols] + ['Mean']
        lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
        for tag, label in rows:
            sel = {p: v for p, v in d[tag].items() if p[1] == st}
            lines.append('| ' + ' | '.join([label] + [mean([v for p, v in sel.items() if p[2] == c]) for c in cols]
                                           + [mean(list(sel.values()))]) + ' |')
        per_set[st] = '\n'.join(lines)
    return note + '\n\n' + '\n'.join(fam), note + '\n\n' + '\n'.join(bt), per_set


def updated_counts(runs):
    """{(backbone, 'dctta' | 'ours'): adapted parameters that received a gradient} from runs.csv (any run that
    tracked it, e.g. ``run.sh count``). The count is architectural (which modules the forward pass uses), so one
    short run per backbone gives it; the maximum over runs covers routing-dependent experts (MoCE-IR)."""
    out = {}
    for r in runs:
        v = num(r.get('n_params_updated'))
        m = {'dctta': 'dctta', 'transfer': 'ours'}.get(r['method'])
        if v is None or m is None or r['source'] != '5task':
            continue
        k = (r.get('backbone') or 'promptir', m)
        out[k] = max(out.get(k, 0), v)
    return out


def cost_table(summary, tag, counts=None):
    """Time and memory from the main 5-task runs; adapted parameters = those that receive a gradient
    (``counts``: ``updated_counts``), blank when no run has tracked it."""
    counts = counts or {}
    lines = ['| Backbone | Method | Adapt time (s / test set) | Peak memory (GB) | Adapted parameters |',
             '|---|---|---|---|---|']
    for bb, name in NAMES.items():
        for m, label in (('dctta', 'DCTTA'), ('ours', 'Ours')):
            rs = [r for r in summary if row_key(r, tag) == m and (r.get('backbone') or 'promptir') == bb
                  and r['source'] == '5task' and r['target'] in TESTS['5task']]
            if not rs:
                continue
            mean = lambda k: sum(num(r.get(k + '_mean')) or 0 for r in rs) / len(rs)  # noqa: E731
            n = counts.get((bb, m))
            lines.append(f"| {name} | {label} | {mean('adapt_time_s'):.0f} | {mean('peak_mem_gb'):.1f} | "
                         f"{'' if n is None else f'{n:,.0f}'} |")
    return '\n'.join(lines)


def write(root, name, md):
    path = os.path.join(root, name)
    with open(path, 'w') as f:
        f.write(md + '\n')
    print(md + f"\n\n-> {path}\n")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', help='results folder with summary.csv / runs.csv (tools/aggregate.py)')
    p.add_argument('--oracle', default=None, help='results folder of the oracle analysis (run.sh oracle)')
    p.add_argument('--count', default=None, help='results folder of the parameter-count runs (run.sh count)')
    p.add_argument('--tag', default='F1', help="our runs' --tag")
    p.add_argument('--seed', type=int, default=23, help='seed of the single-seed control / oracle runs')
    p.add_argument('--input', action='store_true')
    p.add_argument('--ssim', action='store_true')
    p.add_argument('--std', action='store_true')
    args = p.parse_args()
    summary = read_csv(os.path.join(args.root, 'summary.csv'))
    vals = collect(summary, args.tag)
    note = "Rain sets: BT.601 Y-PSNR; others RGB PSNR. Bold: best of the three rows per backbone."
    for s in ('3task', '5task'):
        write(args.root, f'table_main_{s}.md', f"## {s} test sets -- PSNR (dB)\n\n{note}\n\n"
              + main_table(vals, s, TESTS[s], args))
        write(args.root, f'table_noshift_{s}.md', f"## {s} no-shift sets (the sources' own test distributions)"
              f" -- PSNR (dB)\n\n{note}\n\n" + main_table(vals, s, NOSHIFT[s], args))
    runs = read_csv(os.path.join(args.root, 'runs.csv'))
    idx = runs_index(runs, {t for t, _ in CONTROLS})
    write(args.root, 'table_controls.md', f"## Controls, PromptIR 3-task (seed {args.seed}) -- PSNR (dB), change vs the"
          f" pretrained model\n\n" + delta_table(idx, CONTROLS, '3task', CONTROL_SETS, args.seed))
    aidx = runs_index(runs, {t for t, _ in ABLATION})
    for s in ('3task', '5task'):
        write(args.root, f'table_ablation_{s}.md', f"## Ablation, PromptIR {s} (seed {args.seed}) -- PSNR (dB), change"
              f" vs the pretrained model\n\n" + delta_table(aidx, ABLATION, s, TESTS[s], args.seed))
    aidx_all = runs_index(runs, {t for t, _ in ABLATION_ALL})
    pooled = ablation_tables(aidx_all, ABLATION_ALL, args.seed)
    if pooled:
        fam, bt, per_set = pooled
        write(args.root, 'table_ablation_family.md', "## Ablation, all backbones -- change in PSNR (dB) vs the "
              "pretrained model, per degradation family\n\n" + fam)
        write(args.root, 'table_ablation_backbone.md', "## Ablation -- mean change in PSNR (dB) vs the pretrained "
              "model, per backbone\n\n" + bt)
        for st, md in per_set.items():
            write(args.root, f'table_ablation_all_{st}.md', f"## Ablation, {st}, mean over backbones -- change in PSNR"
                  f" (dB) vs the pretrained model\n\n" + md)
    if args.oracle:
        oidx = runs_index(read_csv(os.path.join(args.oracle, 'runs.csv')), {t for t, _ in ORACLE_ROWS})
        write(args.root, 'table_oracle.md', f"## Oracle analysis, PromptIR 5-task (seed {args.seed}) -- PSNR (dB), "
              f"change vs the pretrained model\n\n" + delta_table(oidx, ORACLE_ROWS, '5task', ORACLE_SETS, args.seed))
    counts = updated_counts(runs)
    if args.count:
        counts.update(updated_counts(read_csv(os.path.join(args.count, 'runs.csv'))))
    write(args.root, 'table_cost.md', "## Cost on the 5-task test sets (mean over sets)\n\nAdapted parameters: those "
          "that receive a gradient (modules outside the forward pass excluded).\n\n"
          + cost_table(summary, args.tag, counts))


if __name__ == '__main__':
    main()
