"""Collect the ``result.json`` files of an experiment folder into CSV tables.

    python tools/aggregate.py results/suite        # -> results/suite/runs.csv, results/suite/summary.csv

runs.csv     one row per run (and per evaluated weight mix of older runs, ``alpha*/result.json``)
summary.csv  grouped over seeds: mean and std of every metric, n_seeds, seeds
"""
import argparse
import csv
import json
import math
import os
import statistics

GROUP_KEYS = ['method', 'backbone', 'source', 'target', 'subset', 'lr', 'families', 'exposure_match',
              'eval_alpha', 'max_steps', 'max_images', 'cfg', 'tag']
METRICS = ['psnr', 'psnr_y', 'ssim', 'adapt_time_s', 'predict_time_s', 'peak_mem_gb', 'n_params_adapted',
           'n_params_updated', 'steps']
ID_KEYS = ['run_id', 'exp', 'seed', 'n_images', 'git', 'device']


def load_results(root):
    out = []
    for dirpath, _, files in os.walk(root):
        if 'result.json' in files:
            with open(os.path.join(dirpath, 'result.json')) as f:
                out.append(json.load(f))
    return out


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
    args = p.parse_args()
    results = load_results(args.root)
    if not results:
        raise SystemExit(f"no result.json under {args.root}")
    rows = [{k: r.get(k) for k in ID_KEYS + GROUP_KEYS + METRICS} for r in results
            if r.get('regime', 'transductive') == 'transductive']
    rows.sort(key=lambda r: tuple(str(r.get(k)) for k in GROUP_KEYS) + (r.get('seed') or 0,))
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in GROUP_KEYS), []).append(r)
    summary = []
    for key, rs in groups.items():
        s = dict(zip(GROUP_KEYS, key))
        s['n_seeds'] = len(rs)
        s['seeds'] = ' '.join(str(r['seed']) for r in rs)
        for m in METRICS:
            vals = [r.get(m) for r in rs if fnum(r.get(m))]
            s[f'{m}_mean'] = statistics.fmean(vals) if vals else math.nan
            s[f'{m}_std'] = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else math.nan)
        summary.append(s)
    write_csv(os.path.join(args.root, 'runs.csv'), rows)
    write_csv(os.path.join(args.root, 'summary.csv'), summary)
    print(f"{len(rows)} runs, {len(summary)} groups -> {args.root}/runs.csv, {args.root}/summary.csv")


if __name__ == '__main__':
    main()
