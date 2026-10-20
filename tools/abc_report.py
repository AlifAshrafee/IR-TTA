"""What ABC selected, per transfer run: distance of the chosen pairs to the target, acceptance, families.

    python tools/abc_report.py results/tta            # all ABC runs
    python tools/abc_report.py results/tta --tag A7   # one configuration

Columns: chosen-pair distance (Mahalanobis^2 / D to the target NSS Gaussian; the target's own crops sit near 1),
median and 10/90 % over steps; acceptance rate and skipped steps (only with --abc_eps_q); the share of each chain
among the pairs trained on.
"""
import argparse
import json
import os
import statistics


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, round(p * (len(v) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()
    rows = []
    for d in sorted(os.listdir(args.root)):
        rp = os.path.join(args.root, d, 'result.json')
        sp = os.path.join(args.root, d, 'stats.jsonl')
        if not (os.path.exists(rp) and os.path.exists(sp)):
            continue
        r = json.load(open(rp))
        if r.get('method') != 'transfer' or (args.tag and r.get('tag') != args.tag):
            continue
        dist, acc = [], []
        for line in open(sp):
            s = json.loads(line)
            if 'abc_d' in s:
                dist.append(s['abc_d'])
            if 'abc_accept' in s:
                acc.append(s['abc_accept'])
        if not dist:
            continue
        fams = r.get('families_used') or {}
        tot = sum(fams.values()) or 1
        top = ', '.join(f"{k} {100 * v / tot:.0f}%" for k, v in sorted(fams.items(), key=lambda kv: -kv[1])[:4])
        rows.append([r.get('tag') or '', r['target'], f"{statistics.median(dist):.2f}",
                     f"{q(dist, 0.1):.2f}-{q(dist, 0.9):.2f}",
                     f"{100 * statistics.mean(acc):.0f}%" if acc else '', str(r.get('skipped', 0)), top])
    head = ['Tag', 'Target', 'd median', 'd 10-90%', 'Accepted', 'Skipped', 'Pairs trained on']
    print('| ' + ' | '.join(head) + ' |\n|' + '---|' * len(head))
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')


if __name__ == '__main__':
    main()
