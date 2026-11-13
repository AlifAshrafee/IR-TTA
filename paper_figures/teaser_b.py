"""Fig. 1 (b): the real images behind each box of the teaser (target stack, simulated pairs, outputs), for Lucidchart.

Nothing here is a ground-truth stand-in: the outputs are the restorations of existing TTA (DCTTA) and ours on the
chosen target images, and the pairs are clean DIV2K crops degraded by the method's own simulator, selected by the
method's own ABC rule against the target set. Everything is written to results/figures/teaser_b/.

1. outputs  (GPU once, then instant)
      CUDA_VISIBLE_DEVICES=0 python paper_figures/teaser_b.py outputs --targets <img> <img> <img>
   Runs DCTTA and ours on the whole target set with the tables' configuration (run_tta.py; results/teaser_runs,
   every image saved, so any other candidate can be exported later without re-running; finished runs are skipped).
   Prints every image of the set ranked by how clearly ours beats both the input and DCTTA, then exports per chosen
   target:  outputs/<image>/{input,existing_tta,ours}.png  + psnr.json,  and contact_outputs.png.
   input.png is the target image as the model sees it (the set's max_side, centre crop to x16): use it for the stack.
   Without --targets it only runs and prints the ranking.

2. pairs  (GPU optional, about a minute)
      python paper_figures/teaser_b.py pairs --targets <img> <img> --clean <DIV2K path>[:x,y] <DIV2K path> ...
   Per clean image: one crop (at x,y, or random from --seed), degraded exactly as during adaptation:
   the target set's NSS Gaussian is fitted as in the method (TransferMethod.prepare), a target patch of one of
   --targets sets the camera-consistent exposure (cycled over the targets), k = 8 candidate degradations are drawn
   from the source's families and the one closest to the target set is kept (TransferMethod._pairs, unchanged).
   --family haze (default) re-draws until the method's own selection is a haze pair (any: keep the first draw); the
   number of draws is recorded. Writes  pairs/<NN>_<image>/{clean,degraded}.png + info.json,  and contact_pairs.png.

Images: --targets take file names or paths of the target set (extension ignored); --clean takes any image path
(DIV2K train HR is the method's clean corpus). Optional ':x,y' after a clean path = top-left corner of the crop, in
pixels of the image after --downscale.

Options
  --backbone / --setting   the pretrained model (default bioir 3task; ours 19.40 vs DCTTA 16.57 dB on O-HAZY)
  --target                 the test set (default OHAZY)
  --ckpt                   override the checkpoint path (default: the one scripts/run.sh uses)
  pairs: --crop S          crop size; 192 = the method's. Larger crops show more of the scene; the target statistics
                           are then fitted on crops of the same size, so the selection stays like-for-like
         --downscale F     shrink the clean image by F before cropping (a wider view at the same crop size)
         --seed N          crop positions and degradation draws (try other values for other draws)
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S  # noqa: E402

sys.path.insert(0, S.REPO)
import qualitative as Q  # noqa: E402

OUT = os.path.join(S.REPO, 'results', 'figures', 'teaser_b')
EXP = 'teaser_runs'
CK = {   # = scripts/run.sh
    'promptir:3task': 'pretrain/model.ckpt', 'promptir:5task': 'pretrain/epoch=80.ckpt',
    'adair:3task': 'pretrain/adair/adair3d.ckpt', 'adair:5task': 'pretrain/adair/adair5d.ckpt',
    'bioir:3task': 'pretrain/bioir/BioIR3D.ckpt', 'bioir:5task': 'pretrain/bioir/BioIR5D.ckpt',
    'moceir:3task': 'pretrain/moceir/MoCE_IR_AIO3/last.ckpt', 'moceir:5task': 'pretrain/moceir/MoCE_IR_AIO5/last.ckpt',
    'mirage:3task': 'pretrain/mirage/3Deg_small.ckpt', 'mirage:5task': 'pretrain/mirage/5Deg_small.ckpt',
    'r2r:3task': 'pretrain/r2r/train_ckpt_3D_finetune/last.ckpt', 'r2r:5task': 'pretrain/r2r/train_ckpt_5D_finetune/last.ckpt',
}
BOPTS = {'moceir': ['routing_noise=False']}   # = scripts/run.sh
LABEL = {'lowlight': 'exposure+noise'}         # family -> its chain label (irtta/degrade.py)


def model_args(a):
    ck = a.ckpt or CK.get(f'{a.backbone}:{a.setting}')
    if not ck or not os.path.exists(os.path.join(S.REPO, ck)):
        raise SystemExit(f"checkpoint '{ck}' not found (pretrain/README.md), or pass --ckpt")
    o = BOPTS.get(a.backbone, [])
    return ['--backbone', a.backbone, '--ckpt', ck, '--source', a.setting] + (['--backbone_opts', *o] if o else [])


def target_names(a):
    """{stem: file name} of the target set, and the chosen stems (checked)."""
    from irtta.data import available_names, load_registry, resolve_spec
    spec = resolve_spec(a.target, load_registry())
    names = {os.path.splitext(n)[0]: n for n in available_names(spec)}
    chosen = [os.path.splitext(os.path.basename(t))[0] for t in (a.targets or [])]
    bad = [t for t in chosen if t not in names]
    if bad:
        raise SystemExit(f"not in {a.target}: {bad}; e.g. {sorted(names)[:5]}")
    return spec, names, chosen


def sheet(rows, path, h=180):
    """Contact sheet: rows of (label, [(caption, HxWx3 uint8), ...])."""
    out = []
    for label, cells in rows:
        ims = [Image.fromarray(im).resize((max(1, int(im.shape[1] * h / im.shape[0])), h), Image.LANCZOS)
               for _, im in cells]
        r = Image.new('RGB', (sum(i.size[0] + 4 for i in ims), h + 30), 'white')
        d, x = ImageDraw.Draw(r), 0
        d.text((2, h + 16), label, fill=(0, 0, 0))
        for (cap, _), im in zip(cells, ims):
            r.paste(im, (x, 0))
            d.text((x + 2, h + 2), cap, fill=(0, 0, 0))
            x += im.size[0] + 4
        out.append(r)
    full = Image.new('RGB', (max(r.size[0] for r in out), sum(r.size[1] + 6 for r in out)), 'white')
    y = 0
    for r in out:
        full.paste(r, (0, y))
        y += r.size[1] + 6
    full.save(path)
    print(f"contact sheet -> {path}")


# ---------------------------------------------------------------------------- #
def cmd_outputs(a):
    spec, names, chosen = target_names(a)
    base = [sys.executable, 'run_tta.py', '--num_workers', str(a.num_workers), '--exp', EXP, '--seed', '23',
            *model_args(a), '--target', a.target, '--save_images']
    for extra in (['--method', 'dctta'], ['--tag', 'F1']):   # existing TTA; ours (the tables' configuration)
        subprocess.run(base + extra, cwd=S.REPO, check=True)
    runs = Q.find_runs(os.path.join(S.REPO, 'results', EXP), a.backbone, a.setting, a.target)
    if not {'existing_tta', 'ours'} <= set(runs):
        raise SystemExit(f"no finished DCTTA / ours run in results/{EXP}")
    p = {c: Q.per_image(runs[c], a.target) for c in ('existing_tta', 'ours')}
    luma = a.target in Q.Y_TARGETS
    print(f"\n{a.target}, {a.backbone} {a.setting}: every image, ours' margin over max(input, DCTTA) first "
          f"({'Y-PSNR' if luma else 'RGB PSNR'}, dB; * = chosen)")
    rows = []
    for s in sorted(names):
        lq, gt = Q.load_pair(a.target, s)
        v = {'input': Q.psnr_u8(lq, gt, luma), **{c: p[c].get(s) for c in p}}
        if None not in v.values():
            rows.append((v['ours'] - max(v['input'], v['existing_tta']), s, v))
    print(f"  {'image':<28}{'input':>7}{'DCTTA':>8}{'ours':>8}{'margin':>8}")
    for m, s, v in sorted(rows, key=lambda r: -r[0]):
        print(f"{'*' if s in chosen else ' '} {s:<28}{v['input']:7.2f}{v['existing_tta']:8.2f}{v['ours']:8.2f}{m:+8.2f}")
    if not chosen:
        print("\npass --targets <names> to export images")
        return
    psnr = {s: v for _, s, v in rows}
    sheet_rows = []
    for s in chosen:
        lq, _ = Q.load_pair(a.target, s)
        imgs = {'input': lq}
        for c in ('existing_tta', 'ours'):
            imgs[c] = np.asarray(Image.open(os.path.join(runs[c], 'images', s + '.png')).convert('RGB'))
        od = os.path.join(OUT, 'outputs', s)
        os.makedirs(od, exist_ok=True)
        for c, im in imgs.items():
            Image.fromarray(im).save(os.path.join(od, f'{c}.png'))
        v = psnr[s]
        with open(os.path.join(od, 'psnr.json'), 'w') as f:
            json.dump({'metric': 'Y-PSNR' if luma else 'RGB PSNR', 'backbone': a.backbone, 'setting': a.setting,
                       **{c: round(x, 2) for c, x in v.items()}}, f, indent=1)
        if v['ours'] <= max(v['input'], v['existing_tta']):
            print(f"[warn] {s}: ours ({v['ours']:.2f}) does not beat both the input and DCTTA")
        print(f"[export] {s} -> {od}/")
        sheet_rows.append((s, [(f"input {v['input']:.2f}", lq), (f"DCTTA {v['existing_tta']:.2f}", imgs['existing_tta']),
                               (f"ours {v['ours']:.2f}", imgs['ours'])]))
    sheet(sheet_rows, os.path.join(OUT, 'contact_outputs.png'))


# ---------------------------------------------------------------------------- #
def parse_clean(s):
    m = re.match(r'^(.*):(\d+),(\d+)$', s)
    return (m.group(1), (int(m.group(2)), int(m.group(3)))) if m else (s, None)


class FixedBank:
    """Stands in for the CleanBank: ``TransferMethod._pairs`` samples its clean crops from here, so the chosen crop
    goes through the method's pair code unchanged."""
    crop = None

    def sample(self, n, size, generator):
        return self.crop


def cmd_pairs(a):
    import logging
    import random

    import torch
    from torch.utils.data import DataLoader

    from irtta.backbones import TRAIN_PREFIXES, build_model, parse_backbone_opts
    from irtta.config import get_args
    from irtta.data import DEFAULT_REGISTRY, AdaptationPatchDataset, load_registry, load_rgb, materialize_synthetic, \
        resolve_bank
    from irtta.methods import TransferMethod
    from irtta.subsets import apply_subset

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    log = logging.getLogger('teaser_b')
    if not a.targets or not a.clean:
        raise SystemExit("pairs needs --targets and --clean")
    spec, names, chosen = target_names(a)
    if spec.synthetic:
        spec = materialize_synthetic(spec)
    args = get_args(model_args(a) + ['--target', a.target, '--bank_max', '1', '--bank_patch', str(a.crop),
                                     '--patch_size', str(max(320, a.crop)), '--seed', '23'])
    reg = load_registry(DEFAULT_REGISTRY)
    args.registry, args.target_task, args.bank_path = DEFAULT_REGISTRY, spec.task, resolve_bank(args.bank, reg)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.backbone, os.path.join(S.REPO, args.ckpt), log,
                        **parse_backbone_opts(args.backbone_opts)).to(dev)
    subset = apply_subset(model, args.subset, train_prefixes=TRAIN_PREFIXES[args.backbone], seed=args.seed,
                          logger=log)
    method = TransferMethod(model, subset, args, dev)

    # the target set's NSS Gaussian, fitted as in the method (random patches of the whole unlabelled set)
    random.seed(a.seed)
    ds = AdaptationPatchDataset(spec.lq, None, patch_size=args.patch_size, seed=None, max_side=spec.max_side)
    method.prepare(DataLoader(ds, batch_size=1, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(a.seed)))
    method.bank = FixedBank()
    log.info(f"source families: {method.families}; ABC k = {args.abc_k}; crop {a.crop}px")

    g = torch.Generator().manual_seed(a.seed)
    rint = lambda hi: int(torch.randint(0, hi + 1, (1,), generator=g))  # noqa: E731
    tgt = [load_rgb(os.path.join(spec.lq, names[s]), spec.max_side) for s in chosen]
    os.makedirs(os.path.join(OUT, 'pairs'), exist_ok=True)
    sheet_rows = []
    for i, cs in enumerate(a.clean):
        path, xy = parse_clean(cs)
        im = Image.open(path).convert('RGB')
        if a.downscale > 1:
            im = im.resize((round(im.width / a.downscale), round(im.height / a.downscale)), Image.BICUBIC)
        c = np.asarray(im)
        H, W = c.shape[:2]
        if min(H, W) < a.crop:
            raise SystemExit(f"{path}: {W}x{H} after --downscale is smaller than --crop {a.crop}")
        x0, y0 = xy if xy else (rint(W - a.crop), rint(H - a.crop))
        x0, y0 = min(x0, W - a.crop), min(y0, H - a.crop)
        method.bank.crop = torch.from_numpy(np.ascontiguousarray(c[y0:y0 + a.crop, x0:x0 + a.crop])) \
            .permute(2, 0, 1).float()[None] / 255.
        # a target patch for the camera-consistent exposure, as one adaptation step would see it
        t = tgt[i % len(tgt)]
        P = min(args.patch_size, t.shape[0] // 16 * 16, t.shape[1] // 16 * 16)
        ty, tx = rint(t.shape[0] - P), rint(t.shape[1] - P)
        xpatch = (torch.from_numpy(np.ascontiguousarray(t[ty:ty + P, tx:tx + P])).permute(2, 0, 1).float()[None]
                  / 255.).to(dev)
        res = None
        for d in range(1, a.tries + 1):
            method.gen = torch.Generator().manual_seed(a.seed * 100003 + 1000 * i + d)
            st = {}
            with torch.no_grad():
                out = method._pairs(xpatch, st)
            if out is None:
                continue
            res = (out, st, d)
            if a.family == 'any' or out[2][0] == LABEL.get(a.family, a.family):
                break
        if res is None:
            raise SystemExit(f"{path}: no pair produced")
        (y, cl, fams), st, d = res
        if a.family != 'any' and fams[0] != LABEL.get(a.family, a.family):
            print(f"[warn] {path}: the selection never chose {a.family} in {a.tries} draws; kept {fams[0]}")
        to8 = lambda z: (z[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)  # noqa
        cl8, y8 = to8(cl), to8(y)
        stem = os.path.splitext(os.path.basename(path))[0]
        od = os.path.join(OUT, 'pairs', f'{i:02d}_{stem}')
        os.makedirs(od, exist_ok=True)
        Image.fromarray(cl8).save(os.path.join(od, 'clean.png'))
        Image.fromarray(y8).save(os.path.join(od, 'degraded.png'))
        info = {'clean_image': os.path.abspath(path), 'downscale': a.downscale, 'crop_x_y_size': [x0, y0, a.crop],
                'family': fams[0], 'abc_distance': st.get('abc_d'), 'draws': d, 'exposure_target': chosen[i % len(tgt)],
                'under_exposed_rule': st.get('dark'), 'backbone': a.backbone, 'setting': a.setting,
                'target_set': a.target, 'seed': a.seed}
        with open(os.path.join(od, 'info.json'), 'w') as f:
            json.dump(info, f, indent=1)
        dd = f"{st['abc_d']:.2f}" if st.get('abc_d') is not None else '-'
        print(f"[pair] {stem}: {fams[0]}, ABC distance {dd}, draw {d}, exposure from {info['exposure_target']} "
              f"-> {od}/")
        sheet_rows.append((f"{stem} @ {x0},{y0}", [('clean', cl8), (f"{fams[0]}  d={dd}", y8)]))
    sheet(sheet_rows, os.path.join(OUT, 'contact_pairs.png'))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('outputs', 'pairs'):
        q = sub.add_parser(name)
        q.add_argument('--backbone', default='bioir')
        q.add_argument('--setting', default='3task')
        q.add_argument('--target', default='OHAZY')
        q.add_argument('--ckpt', default=None)
        q.add_argument('--targets', nargs='*', default=[])
        if name == 'outputs':
            q.add_argument('--num_workers', type=int, default=8)
        else:
            q.add_argument('--clean', nargs='+', default=[])
            q.add_argument('--crop', type=int, default=192)
            q.add_argument('--downscale', type=float, default=1.0)
            q.add_argument('--family', default='haze', help="haze | rain | noise | blur | lowlight | any")
            q.add_argument('--tries', type=int, default=50)
            q.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    {'outputs': cmd_outputs, 'pairs': cmd_pairs}[a.cmd](a)


if __name__ == '__main__':
    main()
