"""Supplementary tables E1 and E2 (``run.sh e1 e2``; E3 rows are in table.py's ablation tables).

    python tools/aggregate.py results/suite && python tools/aggregate.py results/supp
    python tools/table_supp.py results/suite results/supp      # -> results/supp/table_supp_E1.md, _E2.md

E1  supervision vs update constraint, PromptIR and R2R, seed 23: 2x2 {DCTTA objective, our pairs} x {DCTTA's update
    constraint (prefix rule + TIPS), our rank-16 update}. The released DCTTA and our runs come from results/suite;
    the two crossed cells (tag E1) from results/supp. Cells: mean PSNR change vs the pretrained model.
E2  GT-mean PSNR (each output scaled to the ground truth's mean grey level), 5-task LOLv2-S and DPDD, seed 23:
    pretrained and ours on every backbone, DCTTA on PromptIR (all from results/supp).
E5  (optional) ABC budget k in {2, 4, 8 (ours), 16}, PromptIR and R2R (results/suite, tags abcK<k>).
PSNR: rain sets in BT.601 Y-PSNR, others RGB (as in the paper tables); GT-mean PSNR is RGB.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import table as T  # noqa: E402

E1_BBS = ['promptir', 'r2r']
E1_CELLS = [  # (key, objective, update constraint)
    ('dctta', 'DCTTA (self-supervised)', "DCTTA's (prefix rule + TIPS)"),
    ('dctta_lora', 'DCTTA (self-supervised)', 'Ours (rank-16 LoRA)'),
    ('ours_dctta', 'Ours (simulated clean pairs)', "DCTTA's (prefix rule + TIPS)"),
    ('ours', 'Ours (simulated clean pairs)', 'Ours (rank-16 LoRA)'),
]
E2_SETS = ['LOLv2_Syn', 'DPDD']


def cell_key(r, suite):
    """Which E1/E2 cell a runs.csv row belongs to (None: not used)."""
    m, sub, tag = r['method'], r.get('subset') or '', r.get('tag') or ''
    if m == 'source':
        return 'source'
    if m == 'dctta' and sub == 'full' and T.row_key(r, 'F1') == 'dctta':
        return 'dctta'
    if m == 'transfer' and sub == 'lora16' and tag == 'F1':
        return 'ours'
    if not suite and tag == 'E1':
        if m == 'dctta' and sub == 'lora16':
            return 'dctta_lora'
        if m == 'transfer' and sub == 'dctta':
            return 'ours_dctta'
    return None


def index(runs, seed, suite):
    """{(cell, backbone, setting, target): (psnr, psnr_gtmean)}; the pretrained model at any seed."""
    out = {}
    for r in runs:
        k = cell_key(r, suite)
        if k is None or (k != 'source' and str(r.get('seed')) != str(seed)):
            continue
        p = T.metric(r, prefix=None)[0]
        if p is not None:
            out[(k, r.get('backbone') or 'promptir', r['source'], r['target'])] = (p, T.num(r.get('psnr_gtmean')))
    return out


def e1_table(suite, supp, seed):
    vals = {**index(suite, seed, True), **{k: v for k, v in index(supp, seed, False).items() if k[0] != 'source'}}
    groups = [(bb, s) for bb in E1_BBS for s in ('3task', '5task')]
    head = (['Objective', 'Update constraint'] + [f"{T.NAMES[bb]} {s}" for bb, s in groups]
            + ['All', 'Median', 'All w/o HIDE', 'Improved'])
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    missing = []
    for key, obj, upd in E1_CELLS:
        allv, nohide, cells = [], [], []
        for bb, s in groups:
            d = []
            for t in T.TESTS[s]:
                src, v = vals.get(('source', bb, s, t)), vals.get((key, bb, s, t))
                if src is None or v is None:
                    missing.append(f"{key}/{bb}/{s}/{t}")
                    continue
                d.append(v[0] - src[0])
                if t != 'HIDE':
                    nohide.append(v[0] - src[0])
            allv += d
            n = len(T.TESTS[s])
            cells.append('' if not d else f"{sum(d) / len(d):+.2f}" + ('' if len(d) == n else f" ({len(d)}/{n})"))
        tot = sum(len(T.TESTS[s]) for _, s in groups)
        allc = '' if not allv else f"{sum(allv) / len(allv):+.2f}"
        imp = '' if not allv else f"{sum(x > 0 for x in allv)}/{len(allv)}" + ('' if len(allv) == tot else '*')
        med = '' if not allv else f"{sorted(allv)[len(allv) // 2] if len(allv) % 2 else sum(sorted(allv)[len(allv) // 2 - 1:len(allv) // 2 + 1]) / 2:+.2f}"
        nh = '' if not nohide else f"{sum(nohide) / len(nohide):+.2f}"
        lines.append('| ' + ' | '.join([obj, upd] + cells + [allc, med, nh, imp]) + ' |')
    note = (f"Mean PSNR change (dB) vs the pretrained model over each setting's test sets (seed {seed}); "
            "'Improved': pairs with a gain. Lr follows the parameterisation (DCTTA's subset 2e-4, LoRA 1e-4); "
            "steps follow the objective (DCTTA: one pass over the target patches; ours: 500). DCTTA on LoRA: no "
            "TIPS (the rank constraint replaces it).")
    if missing:
        note += f" INCOMPLETE: {len(missing)} cells missing (e.g. {', '.join(missing[:4])}); '(k/n)' and '*' mark them."
    detail = []   # per-set values, for checking
    for bb, s in groups:
        dh = ['Cell'] + [T.SHORT.get(t, t) for t in T.TESTS[s]]
        dl = [f"\n{T.NAMES[bb]} {s} -- PSNR (dB)\n", '| ' + ' | '.join(dh) + ' |', '|' + '---|' * len(dh)]
        for key, lab in [('source', 'Pretrained')] + [(k, f"{o.split(' (')[0]} obj. / {u.split(' (')[0]} upd.")
                                                        for k, o, u in E1_CELLS]:
            vs = [vals.get((key, bb, s, t)) for t in T.TESTS[s]]
            dl.append('| ' + ' | '.join([lab] + ['' if v is None else f"{v[0]:.2f}" for v in vs]) + ' |')
        detail.append('\n'.join(dl))
    return note + '\n\n' + '\n'.join(lines) + '\n\n### Per set\n' + '\n'.join(detail)


def e2_table(supp, suite, seed):
    vals = index(supp, seed, False)
    head = ['Model'] + [f"{T.SHORT[t]} {m}" for t in E2_SETS for m in ('PSNR', 'GT-mean')]
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    gaps = {t: ([], []) for t in E2_SETS}
    for bb, name in T.NAMES.items():
        rows = [('source', name), ('ours', f'{name} + Ours')] + ([('dctta', f'{name} + DCTTA')] if bb == 'promptir'
                                                                 else [])
        if not any(('source', bb, '5task', t) in vals for t in E2_SETS):
            continue
        for k, lab in rows:
            out = [lab]
            for t in E2_SETS:
                v = vals.get((k, bb, '5task', t))
                out += ['', ''] if v is None else [f"{v[0]:.2f}", '' if v[1] is None else f"{v[1]:.2f}"]
            lines.append('| ' + ' | '.join(out) + ' |')
        for t in E2_SETS:
            s, o = vals.get(('source', bb, '5task', t)), vals.get(('ours', bb, '5task', t))
            if s and o and s[1] is not None and o[1] is not None:
                gaps[t][0].append(o[0] - s[0])
                gaps[t][1].append(o[1] - s[1])
    summ = []
    for t, (dp, dg) in gaps.items():
        if dp:
            summ.append(f"{T.SHORT[t]}: ours - pretrained = {sum(dp) / len(dp):+.2f} dB PSNR, "
                        f"{sum(dg) / len(dg):+.2f} dB GT-mean PSNR (mean over {len(dp)} backbones)")
    # determinism check: ours / pretrained here vs the same seed in results/suite (should be equal)
    ref = index(suite, seed, True)
    diffs = [abs(v[0] - ref[k][0]) for k, v in vals.items() if k in ref and k[0] in ('source', 'ours', 'dctta')]
    chk = f"Max |PSNR - results/suite| over {len(diffs)} re-run cells: {max(diffs):.3f} dB." if diffs else ''
    return (f"5-task, seed {seed}. GT-mean: RGB PSNR after one gain that matches the output's mean grey level to the "
            f"ground truth's.\n\n" + '\n'.join(lines) + '\n\n' + '\n'.join(f"- {x}" for x in summ)
            + (f"\n\n{chk}" if chk else ''))


def e5_table(suite, seed):
    """Mean change vs the pretrained model per k (None if no abcK run exists)."""
    tags = {f'abcK{k}': k for k in (2, 4, 16)}
    tags['F1'] = 8
    tags['noABC'] = 1   # run.sh ablation: --abc_k 1, i.e. no selection
    vals, src = {}, {}
    for r in suite:
        bb = r.get('backbone') or 'promptir'
        if bb not in E1_BBS:
            continue
        p = T.metric(r, prefix=None)[0]
        if p is None:
            continue
        if r['method'] == 'source':
            src[(bb, r['source'], r['target'])] = p
        elif (r['method'] == 'transfer' and (r.get('subset') or '') == 'lora16' and (r.get('tag') or '') in tags
              and str(r.get('seed')) == str(seed)):
            vals[(tags[r['tag']], bb, r['source'], r['target'])] = p
    if not any(k[0] != 8 for k in vals):
        return None
    groups = [(bb, s) for bb in E1_BBS for s in ('3task', '5task')]
    head = ['k'] + [f"{T.NAMES[bb]} {s}" for bb, s in groups] + ['All']
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for k in sorted({k[0] for k in vals}):
        allv, cells = [], []
        for bb, s in groups:
            d = [vals[(k, bb, s, t)] - src[(bb, s, t)] for t in T.TESTS[s] if (k, bb, s, t) in vals
                 and (bb, s, t) in src]
            allv += d
            n = len(T.TESTS[s])
            cells.append('' if not d else f"{sum(d) / len(d):+.2f}" + ('' if len(d) == n else f" ({len(d)}/{n})"))
        lab = {8: '8 (ours)', 1: '1 (w/o ABC)'}.get(k, str(k))
        lines.append('| ' + ' | '.join([lab] + cells + ['' if not allv else f"{sum(allv) / len(allv):+.2f}"]) + ' |')
    return (f"Mean PSNR change (dB) vs the pretrained model (seed {seed}); '(k/n)': sets finished.\n\n"
            + '\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('suite', help='results/suite (aggregated): pretrained, released DCTTA and ours')
    p.add_argument('supp', help='results/supp (aggregated): run.sh e1 / e2')
    p.add_argument('--seed', type=int, default=23)
    a = p.parse_args()
    suite = T.read_csv(os.path.join(a.suite, 'runs.csv'))
    supp = T.read_csv(os.path.join(a.supp, 'runs.csv'))
    T.write(a.supp, 'table_supp_E1.md', "## E1: supervision vs update constraint, PromptIR and R2R\n\n"
            + e1_table(suite, supp, a.seed))
    T.write(a.supp, 'table_supp_E2.md', "## E2: low light and defocus, PSNR vs GT-mean PSNR\n\n"
            + e2_table(supp, suite, a.seed))
    e5 = e5_table(suite, a.seed)
    if e5:
        T.write(a.supp, 'table_supp_E5.md', "## E5 (optional): ABC budget k, PromptIR and R2R\n\n" + e5)


if __name__ == '__main__':
    main()
