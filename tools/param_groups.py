"""Print which PromptIR tensors DCTTA adapts, which the optimiser sees, and (optionally) the TIPS mask
histogram per top-level module. Useful for the TIPS diagnostics (layer-wise frozen fraction).

    python tools/param_groups.py                       # prefix rule + optimiser subset only
    python tools/param_groups.py --fisher fishers.pt   # + per-module frozen fraction from a saved mask
"""
import argparse
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dctta.models.promptir import PromptIR            # noqa: E402
from dctta.tta import DEFAULT_TRAIN_PREFIXES, collect_params, configure_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_prefixes', nargs='+', default=list(DEFAULT_TRAIN_PREFIXES))
    p.add_argument('--fisher', type=str, default=None, help="torch.save'd DCTTA.fishers dict")
    args = p.parse_args()

    model = configure_model(PromptIR(decoder=True), args.train_prefixes)
    opt_params, opt_names = collect_params(model)
    opt_names = set(opt_names)

    total = sum(p.numel() for p in model.parameters())
    per_module = defaultdict(lambda: [0, 0, 0])  # total, requires_grad, in optimiser
    for name, p in model.named_parameters():
        top = name.split('.')[0]
        per_module[top][0] += p.numel()
        per_module[top][1] += p.numel() if p.requires_grad else 0
        per_module[top][2] += p.numel() if name in opt_names else 0

    print(f"{'module':<24}{'params':>12}{'requires_grad':>16}{'in optimiser':>14}")
    for top, (n, g, o) in per_module.items():
        print(f"{top:<24}{n:>12,}{g:>16,}{o:>14,}")
    n_grad = sum(v[1] for v in per_module.values())
    n_opt = sum(v[2] for v in per_module.values())
    print(f"{'TOTAL':<24}{total:>12,}{n_grad:>16,}{n_opt:>14,}")
    print(f"requires_grad: {100 * n_grad / total:.1f}%   optimised: {100 * n_opt / total:.1f}%")
    skipped = [n for n, p in model.named_parameters() if p.requires_grad and n not in opt_names]
    print(f"\ntrainable tensors NOT in the optimiser ({len(skipped)}): "
          f"{sorted(set(n.split('.')[-1] for n in skipped))}")

    if args.fisher:
        fishers = torch.load(args.fisher, map_location='cpu')
        frozen = defaultdict(lambda: [0, 0])
        for name, (fisher, mask) in fishers.items():
            top = name.split('.')[0]
            frozen[top][0] += int(mask.sum())
            frozen[top][1] += mask.numel()
        print("\nTIPS frozen fraction per module (entries restored to theta_0 after every step):")
        for top, (f, n) in frozen.items():
            print(f"{top:<24}{f:>12,} / {n:<12,} = {100 * f / n:5.1f}%")
        f = sum(v[0] for v in frozen.values()); n = sum(v[1] for v in frozen.values())
        print(f"{'TOTAL':<24}{f:>12,} / {n:<12,} = {100 * f / n:5.1f}%  (of the optimised set: {100 * f / n_opt:.1f}%)")


if __name__ == '__main__':
    main()
