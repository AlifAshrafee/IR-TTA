"""Collect ``result.json`` files into CSV tables.

    python tools/aggregate.py results/step1_smoke            # -> runs.csv, summary.csv in that folder
    python tools/aggregate.py results --out results/all       # everything under results/

runs.csv     one row per run = (method, backbone, source, target, seed, regime, subset, anchor, ...)
             with PSNR/SSIM/LPIPS/drift, cost (adapt/predict time, peak memory, #params) and
             retention (ret_<set>_psnr), plus dpsnr_vs_source (same backbone/source/target, method=source).
             dpsnr_vs_bar = PSNR - max(identity, source) on the same target/subset.
summary.csv  the same grouped over seeds: mean and std of every metric, n_seeds, seeds.
"""
import argparse
import csv
import json
import math
import os
import statistics

GROUP_KEYS = ['method', 'backbone', 'source', 'target', 'split', 'regime', 'subset', 'anchor', 'anchor_lambda',
              'lr', 'families', 'exposure_match', 'ext_alpha', 'n_apply', 'eval_alpha', 'epochs', 'max_steps', 'iterations', 'steps_per_image', 'max_images',
              'cfg', 'tag']
METRICS = ['psnr', 'psnr_y', 'ssim', 'lpips', 'drift', 'dpsnr_vs_source', 'dpsnr_vs_bar', 'adapt_time_s', 'predict_time_s', 'total_time_s',
           'peak_mem_gb', 'n_params_adapted', 'n_params_effective', 'steps', 'skipped']
ID_KEYS = ['run_id', 'exp', 'seed', 'dataset_seed', 'n_images', 'rng_checksum', 'git', 'device']


def load_results(root):
    out = []
    for dirpath, _, files in os.walk(root):
        if 'result.json' in files:
            with open(os.path.join(dirpath, 'result.json')) as f:
                r = json.load(f)
            r['_dir'] = dirpath
            out.append(r)
    return out


def flatten(r):
    row = {k: r.get(k) for k in ID_KEYS + GROUP_KEYS + METRICS if not k.startswith('dpsnr')}
    for s, v in (r.get('retention') or {}).items():
        row[f'ret_{s}_psnr'] = v.get('psnr')
        row[f'ret_{s}_ssim'] = v.get('ssim')
    pf = r.get('probe_final') or {}
    for k, v in pf.items():
        if k != 'step':
            row[f'probe_{k}'] = v
    return row


def method_reference(rows, method):
    """PSNR of an evaluation-only method (source / identity) per (backbone, source, target, max_images);
    seed-independent, so averaged if several exist. Identity does not depend on the model: its backbone key is
    '*' (it is run once, with PromptIR)."""
    ref = {}
    for r in rows:
        if r['method'] == method and r.get('regime') == 'transductive':
            bb = '*' if method == 'identity' else r['backbone']
            ref.setdefault((bb, r['source'], r['target'], r.get('max_images')), []).append(r['psnr'])
    return {k: sum(v) / len(v) for k, v in ref.items()}


def fnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not (isinstance(v, float) and math.isnan(v))


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', help='folder searched recursively for result.json')
    p.add_argument('--out', default=None, help='output folder (default: root)')
    args = p.parse_args()
    results = load_results(args.root)
    if not results:
        raise SystemExit(f"no result.json under {args.root}")
    rows = [flatten(r) for r in results]
    ref_src, ref_id = method_reference(rows, 'source'), method_reference(rows, 'identity')
    for r in rows:
        key = (r['backbone'], r['source'], r['target'], r.get('max_images'))
        s, i = ref_src.get(key), ref_id.get(('*',) + key[1:])
        ok = fnum(r['psnr'])
        r['dpsnr_vs_source'] = (r['psnr'] - s) if s is not None and ok else math.nan
        # the bar a TTA method must clear: the better of the unprocessed input and the source model
        bar = max(v for v in (s, i) if v is not None) if (s is not None or i is not None) else None
        r['dpsnr_vs_bar'] = (r['psnr'] - bar) if bar is not None and ok else math.nan
    rows.sort(key=lambda r: tuple(str(r.get(k)) for k in GROUP_KEYS) + (r.get('seed') or 0,))

    groups = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in GROUP_KEYS), []).append(r)
    summary = []
    extra = sorted({k for r in rows for k in r if k.startswith(('ret_', 'probe_'))})
    for key, rs in groups.items():
        s = dict(zip(GROUP_KEYS, key))
        s['n_seeds'] = len(rs)
        s['seeds'] = ' '.join(str(r['seed']) for r in rs)
        for m in METRICS + extra:
            vals = [r.get(m) for r in rs if fnum(r.get(m))]
            s[f'{m}_mean'] = statistics.fmean(vals) if vals else math.nan
            s[f'{m}_std'] = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else math.nan)
        checks = {r.get('rng_checksum') for r in rs}
        s['rng_checksums'] = ' '.join(str(c) for c in sorted(checks, key=str))
        summary.append(s)

    out = args.out or args.root
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, 'runs.csv'), rows)
    write_csv(os.path.join(out, 'summary.csv'), summary)

    print(f"{len(rows)} runs, {len(summary)} groups -> {out}/runs.csv, {out}/summary.csv\n")
    hdr = f"{'method':<10}{'subset':<14}{'families':<10}{'regime':<13}{'target':<14}{'src':<6}{'n':>3}" \
          f"{'PSNR':>16}{'vs bar':>8}{'SSIM':>8}{'drift':>8}{'adapt s':>9}{'GB':>6}{'#params':>11}"
    print(hdr)
    print('-' * len(hdr))
    for s in summary:
        print(f"{str(s['method'])[:9]:<10}{str(s['subset'])[:13]:<14}{str(s['families'] or '-')[:9]:<10}"
              f"{str(s['regime'])[:12]:<13}{str(s['target'])[:13]:<14}{str(s['source'])[:5]:<6}{s['n_seeds']:>3}"
              f"{s['psnr_mean']:>9.3f}±{s['psnr_std']:<6.3f}{s['dpsnr_vs_bar_mean']:>8.3f}{s['ssim_mean']:>8.4f}"
              f"{s['drift_mean']:>+8.3f}{s['adapt_time_s_mean']:>9.1f}{s['peak_mem_gb_mean']:>6.1f}"
              f"{s['n_params_adapted_mean']:>11.0f}")


if __name__ == '__main__':
    main()
