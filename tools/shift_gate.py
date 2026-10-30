"""Shift gate diagnostic (no model, no labels; the GT is never read). Two set-level questions per target:

  1. SHIFT  -- has the degradation changed?  The target inputs' NSS features (irtta/nss.py, the ABC features)
     against each of the source's own test distributions R (registry ``_source_sets``: same operator as training).
     score(T, R) = median Mahalanobis^2/D of T's crops under N_R, fitted on R's even-indexed images, divided by the
     in-sample median of those even-indexed crops. For T = R itself, R's odd-indexed (held-out) images are scored,
     which gives the content-only scale (same operator, new images: > 1 from finite-sample fit alone).
     SHIFT = min over R (the nearest training degradation). --refs can point at the sources' training splits.
  2. MISFIT -- can our simulator explain the target?  The F1 ABC step without a model: clean-bank crops, k
     candidates from the source's families at the target crop's exposure (camera matching), keep the closest to
     N_T. MISFIT = median distance of the kept candidates / median distance of T's own crops under N_T (~1 = the
     simulator reproduces the target's statistics). Also reports the family ABC picks most often.

    python tools/shift_gate.py --setting 5task
    python tools/shift_gate.py --setting 3task --targets Rain100H SOTS_Indoor BSD68_s25

Gate (to be fixed before reading any gated PSNR): adapt iff SHIFT > tau_s and MISFIT < tau_m; otherwise keep
the pretrained model. The thresholds are not set here: the printout is for deciding whether the two scores
separate the no-shift sets from the shifted ones at all.
"""
import argparse
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from irtta.data import CleanBank, load_registry, make_eval_dataset, resolve_bank, resolve_spec  # noqa: E402
from irtta.degrade import degrade_batch, parse_families  # noqa: E402
from irtta.evaluation import eval_loader  # noqa: E402
from irtta.nss import GaussianModel, nss_features, random_crop_batch  # noqa: E402

TESTS = {'3task': ['Rain100H', 'SOTS_Indoor', 'DenseHaze', 'OHAZY', 'Kodak24_Poisson'],
         '5task': ['Rain100H', 'SOTS_Indoor', 'DenseHaze', 'OHAZY', 'Kodak24_Poisson', 'HIDE', 'LOLv2_Syn', 'DPDD']}
FAMS = {'3task': 'noise+rain+haze', '5task': 'all'}


def crops_of(name, reg, n_img, per_img, size, g):
    """{'even': (N,3,s,s), 'odd': ...} LQ crops of up to n_img evenly spaced images of a registry set."""
    ds = make_eval_dataset(resolve_spec(name, reg), max_images=n_img)
    out = {'even': [], 'odd': []}
    for i, (_, x, _) in enumerate(eval_loader(ds, 0)):
        if min(x.shape[-2:]) < size:
            continue
        out['even' if i % 2 == 0 else 'odd'].append(random_crop_batch(x, size, per_img, g))
    return {k: torch.cat(v, 0) for k, v in out.items() if v}


@torch.no_grad()
def feats(x, bs=64):
    return torch.cat([nss_features(x[i:i + bs], color=True) for i in range(0, len(x), bs)], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', default='5task', choices=['3task', '5task'])
    ap.add_argument('--targets', nargs='*', default=None, help='default: the setting\'s test sets + no-shift sets')
    ap.add_argument('--n_img', type=int, default=100)
    ap.add_argument('--per_img', type=int, default=2)
    ap.add_argument('--size', type=int, default=192)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--n_sim', type=int, default=128, help='simulated pairs for MISFIT')
    ap.add_argument('--refs', nargs='*', default=None,
                    help="reference sets (default: the registry's _source_sets for the setting)")
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    reg = load_registry()
    refs = args.refs or reg['_source_sets'][args.setting]
    targets = args.targets or (TESTS[args.setting] + refs)
    g = torch.Generator().manual_seed(args.seed)

    ref = {}
    for r in refs:
        c = crops_of(r, reg, args.n_img, args.per_img, args.size, g)
        fe = feats(c['even'])
        model = GaussianModel(fe)
        fo = feats(c['odd'])
        ref[r] = (model, float(model.distance(fe).median()), fo)
        print(f"[ref] {r}: {len(c['even'])} / {len(c['odd'])} crops, in-sample median {ref[r][1]:.2f}, held-out "
              f"{float(model.distance(fo).median()):.2f}", file=sys.stderr)
    bank = CleanBank(resolve_bank(None, reg), max_images=300, min_side=args.size)
    fams = parse_families(FAMS[args.setting])

    print(f"| Target | Nearest training degradation | SHIFT | MISFIT | ABC's top family |\n|---|---|---|---|---|")
    for t in targets:
        if t in ref:
            ft = ref[t][2]
            tc = None
        else:
            c = crops_of(t, reg, args.n_img, args.per_img, args.size, g)
            tc = torch.cat(list(c.values()), 0)
            ft = feats(tc)
        shift = {r: float(m.distance(ft).median()) / max(s0, 1e-6) for r, (m, s0, _) in ref.items()}
        near = min(shift, key=shift.get)
        if tc is None:   # a reference set as target: its held-out half
            tc = crops_of(t, reg, args.n_img, args.per_img, args.size, g)['odd']
        tm = GaussianModel(ft)
        own = float(tm.distance(ft).median())
        sel = lambda cand: tm.distance(nss_features(cand, color=True))  # noqa: E731
        d, labels = [], Counter()
        for i in range(0, args.n_sim, 8):
            cl = bank.sample(8, args.size, g)
            for j in range(cl.shape[0]):
                x = tc[int(torch.randint(len(tc), (1,), generator=g))][None]
                tmean = x.mean(dim=(0, 2, 3), keepdim=True)
                _, _, lab, sc = degrade_batch(cl[j:j + 1], fams, g, tmean, 'camera', select=sel, k=args.k)
                d += sc
                labels[lab[0].split(':')[0].split('+')[0]] += 1
        misfit = float(torch.tensor(d).median()) / max(own, 1e-6)
        top, cnt = labels.most_common(1)[0]
        print(f"| {t} | {near} | {shift[near]:.2f} | {misfit:.2f} | {top} ({100 * cnt / sum(labels.values()):.0f}%) |",
              flush=True)


if __name__ == '__main__':
    main()
