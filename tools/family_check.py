"""Does set-level family selection (--fam_select abc, irtta/famsel.py) find the right degradation family?

    python tools/family_check.py --setting 5task            # CPU is fine (a few minutes per set); --device cuda
    python tools/family_check.py --setting 3task

Development criterion, fixed before any GPU run: the top family must be right on every NO-SHIFT set (the sources'
own test distributions, family known from the dataset: BSD68 noise, Rain100L rain, SOTS-Outdoor haze, GoPro blur,
LOL-v1 low-light). Validation and test sets are printed for information. Settings = F1's (k=8, camera exposure,
300 DIV2K crops, 192 px). No model, no GT.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from irtta.data import CleanBank, load_registry, make_eval_dataset, resolve_bank, resolve_spec  # noqa: E402
from irtta.degrade import TASK_TO_FAMILY, parse_families  # noqa: E402
from irtta.evaluation import eval_loader  # noqa: E402
from irtta.famsel import family_posterior  # noqa: E402
from irtta.nss import random_crop_batch  # noqa: E402

SETS = {
    '3task': {'no-shift': ['BSD68_s15', 'BSD68_s25', 'BSD68_s50', 'Rain100L', 'SOTS_Outdoor'],
              'val': ['Rain100H_val'],
              'test': ['Rain100H', 'SOTS_Indoor', 'DenseHaze', 'OHAZY', 'Kodak24_Poisson']},
    '5task': {'no-shift': ['BSD68_s25', 'Rain100L', 'SOTS_Outdoor', 'GoPro', 'LOL'],
              'val': ['Rain100H_val', 'HIDE_val', 'LOLv2_Syn_val', 'DPDD_val'],
              'test': ['Rain100H', 'SOTS_Indoor', 'DenseHaze', 'OHAZY', 'Kodak24_Poisson', 'HIDE', 'LOLv2_Syn',
                       'DPDD']},
}
FAMS = {'3task': 'noise+rain+haze', '5task': 'all'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', default='5task', choices=['3task', '5task'])
    ap.add_argument('--groups', nargs='+', default=['no-shift', 'val', 'test'])
    ap.add_argument('--n_img', type=int, default=128)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--n_pilot', type=int, default=64)
    ap.add_argument('--eps_q', type=float, default=0.9)
    ap.add_argument('--size', type=int, default=192)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    reg = load_registry()
    dev = torch.device(args.device)
    fams = parse_families(FAMS[args.setting])
    bank = CleanBank(resolve_bank(None, reg), max_images=300, min_side=args.size)
    head = ['Group', 'Set', 'True family'] + fams + ['Top', 'OK']
    print('| ' + ' | '.join(head) + ' |\n|' + '---|' * len(head), flush=True)
    n_ok = n_ns = 0
    for grp in args.groups:
        for name in SETS[args.setting][grp]:
            try:
                spec = resolve_spec(name, reg)
                ds = make_eval_dataset(spec, max_images=args.n_img)
            except (KeyError, ValueError, FileNotFoundError) as e:
                print(f"| {grp} | {name} | (skipped: {e}) |", flush=True)
                continue
            g = torch.Generator().manual_seed(args.seed)
            crops = [random_crop_batch(x, args.size, 2, g) for _, x, _ in eval_loader(ds, 0)
                     if min(x.shape[-2:]) >= args.size]
            tc = torch.cat(crops, 0).to(dev)
            w, st, eps = family_posterior(tc, bank, fams, g, k=args.k, n_pilot=args.n_pilot, eps_q=args.eps_q,
                                          mode='camera', size=args.size)
            true = TASK_TO_FAMILY.get(spec.task, '?')
            top = fams[max(range(len(fams)), key=lambda i: w[i])]
            ok = top == true
            if grp == 'no-shift':
                n_ns += 1
                n_ok += ok
            cells = [f"{wi:.2f}" for wi in w]
            print(f"| {grp} | {name} | {true} | " + ' | '.join(cells) + f" | {top} | {'yes' if ok else 'NO'} |",
                  flush=True)
    if n_ns:
        print(f"\nno-shift sets with the right top family: {n_ok}/{n_ns} "
              f"({'PASS' if n_ok == n_ns else 'FAIL'}: criterion = all)")


if __name__ == '__main__':
    main()
