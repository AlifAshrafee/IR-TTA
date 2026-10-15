"""Check every dataset in configs/datasets.json before running anything.

    python tools/check_registry.py                     # all entries
    python tools/check_registry.py HIDE HIDE_val      # some

Per set: folders exist, #LQ, #GT, #pairs matched by *file name* (the harness pairs by exact name,
extension included), LQ/GT size mismatches, smallest image side (transductive adaptation needs
both sides >= --patch_size, default 320). Reads image headers only, so it is fast.
"""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dctta.data import list_images                        # noqa: E402
from irtta.data import load_registry, resolve_spec         # noqa: E402

PATCH = 320


def size(path):
    with Image.open(path) as im:
        return im.size  # (W, H)


def check(spec):
    issues = []
    if not os.path.isdir(spec.gt):
        return f"{spec.name:<18} MISSING gt dir {spec.gt}", ['missing']
    gt = list_images(spec.gt)
    if spec.synthetic:
        sizes = [size(os.path.join(spec.gt, n)) for n in gt]
        n_lq, pairs = '-', gt
    else:
        if not os.path.isdir(spec.lq):
            return f"{spec.name:<18} MISSING lq dir {spec.lq}", ['missing']
        lq = list_images(spec.lq)
        gts = set(gt)
        pairs = [n for n in lq if n in gts]
        n_lq = len(lq)
        if not pairs:
            ex = f"LQ e.g. {lq[:2]}, GT e.g. {gt[:2]}"
            return f"{spec.name:<18} NO PAIRS by file name ({ex})", ['unpaired']
        if len(pairs) < max(len(lq), len(gt)):
            issues.append(f"{max(len(lq), len(gt)) - len(pairs)} unpaired files")
        sizes = []
        mism = 0
        for n in pairs:
            a, b = size(os.path.join(spec.lq, n)), size(os.path.join(spec.gt, n))
            sizes.append(a)
            mism += int(a != b)
        if mism:
            issues.append(f"{mism} LQ/GT size mismatches")
    min_side = min(min(w, h) for w, h in sizes)
    small = sum(1 for w, h in sizes if min(w, h) < PATCH)
    if small and spec.split in ('test', 'val'):
        issues.append(f"{small} images smaller than {PATCH}px (transductive patches will fail; use --patch_size)")
    kind = f"synthetic sigma={spec.noise_sigma:g}" if spec.synthetic else "stored"
    line = (f"{spec.name:<18}{spec.split:<8}{kind:<22}LQ {str(n_lq):>6}  GT {len(gt):>6}  pairs {len(pairs):>6}  "
            f"min side {min_side:>5}  " + ("OK" if not issues else "; ".join(issues)))
    return line, issues


def main():
    reg = load_registry()
    names = sys.argv[1:] or [k for k in reg if not k.startswith('_')]
    bad = 0
    for n in names:
        line, issues = check(resolve_spec(n, reg))
        bad += bool(issues)
        print(line)
    for tag, sets in reg.get('_source_sets', {}).items():
        unknown = [s for s in sets if s not in reg]
        if unknown:
            bad += 1
            print(f"_source_sets[{tag}] refers to unknown entries: {unknown}")
    print(f"\n{bad} problem(s)" if bad else "\nall entries OK")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
