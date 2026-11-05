"""Arguments of ``run_tta.py`` and the run id. Importable without torch.

Defaults are the final method: ``python run_tta.py --target X --backbone B --ckpt C`` adapts with it. The run id
encodes every argument that changes the numbers and differs from a fixed reference (``_ID_REFERENCE``), so a
finished run is recognised and skipped, and two configurations never share a folder.
"""
import argparse
import os
import re
from collections import OrderedDict

BACKBONES = ('promptir', 'adair', 'bioir', 'moceir', 'mirage', 'r2r')
SETTING_FAMILIES = {'3task': 'noise+rain+haze', '5task': 'all'}   # each source's own degradation families

# per-method defaults: loss weights (pixel, perceptual, low-pass), DCTTA teacher EMA, adaptation steps, lr, subset
METHOD_DEFAULTS = {
    'dctta':    dict(pixel_weight=1.0, perceptual_weight=0.01, lowpass_weight=0.1, ema_decay=0.45, max_steps=-1,
                     lr=2e-4, subset='full'),
    'transfer': dict(pixel_weight=1.0, perceptual_weight=0.0, lowpass_weight=0.0, ema_decay=0.999, max_steps=500,
                     lr=1e-4, subset='lora16'),
    'source':   dict(max_steps=-1, lr=5e-5, subset='none'),
    'identity': dict(max_steps=-1, lr=5e-5, subset='none'),
}
_METHOD_KEYS = ('pixel_weight', 'perceptual_weight', 'lowpass_weight', 'ema_decay', 'max_steps', 'lr', 'subset')


def get_parser():
    p = argparse.ArgumentParser(description="Test-time adaptation of all-in-one restoration models")
    g = p.add_argument_group('data')
    g.add_argument('--target', type=str, default=None, help='dataset name in the registry (configs/datasets.json)')
    g.add_argument('--lq_dir', type=str, default=None, help='... or explicit LQ / GT folders')
    g.add_argument('--gt_dir', type=str, default=None)
    g.add_argument('--registry', type=str, default=None, help='default: configs/datasets.json')
    g.add_argument('--crop_base', type=int, default=16, help='evaluation images are centre-cropped to a multiple')
    g.add_argument('--num_workers', type=int, default=8)
    g.add_argument('--max_images', type=int, default=0,
                   help='adapt and evaluate on an evenly spaced subset of this many images (0: all)')

    g = p.add_argument_group('pretrained model')
    g.add_argument('--backbone', type=str, default='promptir', choices=list(BACKBONES))
    g.add_argument('--ckpt', type=str, required=True, help='the released checkpoint (pretrain/README.md)')
    g.add_argument('--backbone_opts', type=str, nargs='*', default=[],
                   help='backbone key=value options, e.g. moceir: routing_noise=False; r2r: bank_dir=<dir>')
    g.add_argument('--source', type=str, default=None,
                   help='source setting, 3task | 5task (default: read off the checkpoint path)')

    g = p.add_argument_group('method')
    g.add_argument('--method', type=str, default='transfer', choices=sorted(METHOD_DEFAULTS),
                   help='transfer = ours | dctta (baseline) | source (pretrained, no adaptation) | identity (input)')
    g.add_argument('--subset', type=str, default=None,
                   help='adapted parameters: lora<r> | full (DCTTA rule) | all | none (default: lora16 / full)')
    g.add_argument('--lr', type=float, default=None, help='default: 1e-4 (ours), 2e-4 (dctta)')
    g.add_argument('--beta1', type=float, default=0.9)
    g.add_argument('--beta2', type=float, default=0.999)
    g.add_argument('--iterations', type=int, default=1, help='optimiser steps per target patch')
    g.add_argument('--patch_size', type=int, default=320, help='target patch size')
    g.add_argument('--batch_size', type=int, default=1, help='target patches per step')
    g.add_argument('--epochs', type=int, default=1, help='passes over the target set when --max_steps is -1')
    g.add_argument('--max_steps', type=int, default=None,
                   help='adaptation steps (default: 500 for ours, one pass over the target set for dctta)')

    g = p.add_argument_group('ours')
    g.add_argument('--families', type=str, default='auto',
                   help="degradation families: auto (the source's own: 3task noise+rain+haze, 5task all) | all | "
                        "'+'-joined subset of blur, noise, rain, haze, lowlight | oracle (the target's family; analysis)")
    g.add_argument('--bank', type=str, default=None,
                   help="clean images: a folder or a key of the registry's _banks (default: _banks['default'])")
    g.add_argument('--bank_max', type=int, default=300, help='clean images kept in RAM (evenly spaced)')
    g.add_argument('--bank_patch', type=int, default=192, help='clean crop size')
    g.add_argument('--bank_batch', type=int, default=4, help='synthetic pairs per step')
    g.add_argument('--exposure_match', type=str, default='camera', choices=['camera', 'none'],
                   help='camera: camera-consistent exposure to the target patch (irtta/degrade.py) | none')
    g.add_argument('--abc_k', type=int, default=8,
                   help='ABC pair selection: k candidate degradations per clean crop, keep the one closest to the '
                        'target set in NSS features; 1 = off')
    g.add_argument('--nss_color', action=argparse.BooleanOptionalAction, default=True,
                   help='add the dark-channel and saturation statistics to the NSS features')
    g.add_argument('--abc_fit_n', type=int, default=256, help='target crops for the target NSS Gaussian')
    g.add_argument('--abc_fit_passes', type=int, default=1,
                   help='max passes over the target set (fresh patches) to reach --abc_fit_n crops; 1 = one pass '
                        '(2 crops per image)')
    g.add_argument('--abc_strat', action=argparse.BooleanOptionalAction, default=False,
                   help='stratified ABC candidates: every family ceil(k/|F|) times instead of k uniform draws')
    g.add_argument('--abc_content', action=argparse.BooleanOptionalAction, default=False,
                   help="content-corrected ABC distance: remove the clean crop's NSS offset from the pristine mean")
    g.add_argument('--dark_rule', type=str, default='source', choices=['source', 'none'],
                   help='source: a target patch the pretrained model brightens by >= --dark_ratio is under-exposed')
    g.add_argument('--dark_ratio', type=float, default=1.414, help='brightening factor of --dark_rule (sqrt 2)')
    g.add_argument('--dark_action', type=str, default='anchor', choices=['anchor', 'skip'],
                   help='under-exposed patch: anchor (clean images take the pretrained output brightness) | skip')
    g.add_argument('--oracle_pairs', type=str, default=None,
                   help='ANALYSIS: train on real (LQ, GT) pairs of this registry set instead of synthetic ones')

    g = p.add_argument_group('controls')
    g.add_argument('--init', type=str, default='source', choices=['source', 'scratch'],
                   help='scratch: same architecture, random weights (use --subset all)')
    g.add_argument('--cond_target', type=str, default=None,
                   help="adapt on this registry set's images, evaluate on --target (mismatched conditioning)")

    g = p.add_argument_group('dctta (baseline)')
    g.add_argument('--pixel_weight', type=float, default=None)
    g.add_argument('--perceptual_weight', type=float, default=None)
    g.add_argument('--lowpass_weight', type=float, default=None)
    g.add_argument('--ema_decay', type=float, default=None, help='teacher EMA decay (0.45)')
    g.add_argument('--teacher_weight', type=float, default=5.0, help='pseudo-label loss weight')
    g.add_argument('--tips_ratio', type=float, default=0.6, help='per-tensor fraction restored by TIPS')
    g.add_argument('--gen_train_steps', type=int, default=5)
    g.add_argument('--gen_sampling_steps', type=int, default=5)
    g.add_argument('--gen_lr', type=float, default=2e-4)
    g.add_argument('--gen_sum_scale', type=float, default=0.01)

    g = p.add_argument_group('run')
    g.add_argument('--seed', type=int, default=23)
    g.add_argument('--dataset_seed', type=int, default=None,
                   help='global re-seed inside the patch dataset, as the released DCTTA code does (default: --seed)')
    g.add_argument('--save_images', action='store_true')
    g.add_argument('--out_root', type=str, default='results')
    g.add_argument('--exp', type=str, default='suite', help='experiment folder under --out_root')
    g.add_argument('--tag', type=str, default='', help='suffix of the run id; the tables select runs by it')
    g.add_argument('--overwrite', action='store_true', help='re-run even if result.json exists')
    return p


def infer_source(ckpt):
    """'3task' / '5task' from the released checkpoint path (PromptIR: model.ckpt / epoch=80.ckpt; the other
    backbones carry 3D/5D, AIO3/AIO5 or 3Deg/5Deg in the file or folder name)."""
    base = os.path.basename(ckpt)
    if base == 'model.ckpt':
        return '3task'
    if base.startswith('epoch=80'):
        return '5task'
    path = ckpt.replace('\\', '/').lower()
    for n in ('3', '5'):
        if re.search(rf'(^|[/_.-])(adair{n}d|bioir{n}d|aio{n}|{n}deg|{n}d|{n}task)([/_.-]|$)', path):
            return f'{n}task'
    return os.path.splitext(base)[0]


def finalize(args):
    d = METHOD_DEFAULTS[args.method]
    for k in _METHOD_KEYS:
        if getattr(args, k, None) is None:
            setattr(args, k, d.get(k, 0.0))
    if args.dataset_seed is None:
        args.dataset_seed = args.seed
    if args.source is None:
        args.source = infer_source(args.ckpt)
    if args.families == 'auto':
        args.families = SETTING_FAMILIES.get(args.source, 'all')
    args.anchor = 'tips' if args.method == 'dctta' else 'none'   # run-id field (DCTTA restores entries with TIPS)
    args.regime = 'transductive'
    if (args.target is None) == (args.lq_dir is None):
        raise SystemExit("give exactly one of --target or --lq_dir/--gt_dir")
    if args.lq_dir is not None and args.gt_dir is None:
        raise SystemExit("--lq_dir needs --gt_dir")
    return args


def get_args(argv=None):
    return finalize(get_parser().parse_args(argv))


# ---------------------------------------------------------------------------- #
# run id
# ---------------------------------------------------------------------------- #
# Arguments that change the numbers but are not already in the readable part of the id (method, subset, target,
# source, lr, seeds, tag), with their id codes, in id order. A value enters the id when it differs from
# ``_ID_REFERENCE`` (fixed, NOT the parser defaults: ids stay stable when defaults change).
SEMANTIC_ARGS = OrderedDict([
    ('families', 'fam'), ('bank', 'bk'), ('bank_max', 'bm'), ('bank_patch', 'bp'), ('bank_batch', 'bb'),
    ('exposure_match', 'em'), ('dark_rule', 'drl'), ('dark_ratio', 'drr'), ('dark_action', 'dac'),
    ('nss_color', 'nc'), ('abc_k', 'abc'), ('abc_fit_n', 'abn'), ('abc_fit_passes', 'afp'), ('abc_strat', 'ast'),
    ('abc_content', 'acn'), ('ema_decay', 'ema'), ('iterations', 'it'),
    ('pixel_weight', 'pix'), ('perceptual_weight', 'vgg'), ('lowpass_weight', 'lp'), ('teacher_weight', 'tw'),
    ('beta1', 'b1'), ('beta2', 'b2'), ('tips_ratio', 'tr'), ('gen_train_steps', 'gts'),
    ('gen_sampling_steps', 'gss'), ('gen_lr', 'glr'), ('gen_sum_scale', 'gsc'), ('patch_size', 'ps'),
    ('batch_size', 'bs'), ('epochs', 'ep'), ('max_steps', 'ms'), ('max_images', 'mi'), ('crop_base', 'cb'),
    ('backbone_opts', 'bo'), ('cond_target', 'ct'), ('init', 'ini'), ('oracle_pairs', 'orp'),
])
_ID_REFERENCE = dict(
    families='compose', bank=None, bank_max=300, bank_patch=192, bank_batch=4, exposure_match='fit',
    dark_rule='none', dark_ratio=1.414, dark_action='skip', nss_color=False, abc_k=1, abc_fit_n=256,
    abc_fit_passes=1, abc_strat=False, abc_content=False, iterations=1,
    teacher_weight=5.0, beta1=0.9, beta2=0.999, tips_ratio=0.6, gen_train_steps=5, gen_sampling_steps=5,
    gen_lr=2e-4, gen_sum_scale=0.01, patch_size=320, batch_size=1, epochs=1, max_steps=-1, max_images=0,
    crop_base=16, backbone_opts=[], cond_target=None, init='source', oracle_pairs=None)
_ID_METHOD_REFERENCE = {
    'dctta': dict(pixel_weight=1.0, perceptual_weight=0.01, lowpass_weight=0.1, ema_decay=0.45),
    'transfer': dict(pixel_weight=1.0, perceptual_weight=0.0, lowpass_weight=0.0, ema_decay=0.999),
}
_OURS = ('families', 'bank', 'bank_max', 'bank_patch', 'bank_batch', 'exposure_match', 'dark_rule', 'dark_ratio',
         'dark_action', 'nss_color', 'abc_k', 'abc_fit_n', 'abc_fit_passes', 'abc_strat', 'abc_content',
         'oracle_pairs')
_DCTTA = ('teacher_weight', 'gen_train_steps', 'gen_sampling_steps', 'gen_lr', 'gen_sum_scale', 'tips_ratio')
_OPTIM = ('ema_decay', 'iterations', 'pixel_weight', 'perceptual_weight', 'lowpass_weight', 'beta1', 'beta2',
          'patch_size', 'batch_size', 'epochs', 'max_steps')
_IRRELEVANT = {'source': _OURS + _DCTTA + _OPTIM, 'identity': _OURS + _DCTTA + _OPTIM,
               'transfer': _DCTTA + ('ema_decay',), 'dctta': _OURS}


def config_suffix(args):
    """``code=value`` parts of every semantic argument that differs from the reference; '' for the reference."""
    skip = set(_IRRELEVANT.get(args.method, ()))
    if args.method == 'transfer':
        if args.dark_rule == 'none':
            skip |= {'dark_ratio', 'dark_action'}
        abc_opts = {'nss_color', 'abc_fit_n', 'abc_fit_passes', 'abc_strat', 'abc_content'}
        if args.abc_k <= 1:
            skip |= abc_opts
        if args.oracle_pairs:   # real pairs: nothing is synthesised
            skip |= {'families', 'dark_rule', 'dark_ratio', 'dark_action', 'abc_k'} | abc_opts
    ref = {**_ID_REFERENCE, **_ID_METHOD_REFERENCE.get(args.method, {})}
    parts = []
    for k, code in SEMANTIC_ARGS.items():
        if k in skip or k not in ref:
            continue
        v, d = getattr(args, k), ref[k]
        if v != d:
            if k == 'bank':
                v = os.path.basename(os.path.normpath(v))
            elif k == 'backbone_opts':
                v = '+'.join(v)
            parts.append(f"{code}={v:g}" if isinstance(v, float) else f"{code}={v}")
    return '_'.join(parts)


def _safe(s):
    return re.sub(r'[^A-Za-z0-9._=-]+', '_', str(s))


def make_run_id(args, target):
    parts = [args.method, _safe(args.subset), args.anchor, args.regime, target, args.source]
    if args.backbone != 'promptir':
        parts.insert(0, args.backbone)
    if args.method not in ('source', 'identity'):
        parts.append(f"lr{args.lr:g}")
    parts.append(f"s{args.seed}" + (f"d{args.dataset_seed}" if args.dataset_seed != args.seed else ''))
    cfg = config_suffix(args)
    if cfg:
        parts.append(_safe(cfg))
    if args.tag:
        parts.append(_safe(args.tag))
    return '-'.join(parts)
