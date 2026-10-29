"""Arguments of ``run_tta.py``. Method-dependent defaults (``None`` below) come from ``METHOD_DEFAULTS``."""
import argparse
import os
from collections import OrderedDict

from dctta.models import BACKBONES

from .methods import METHOD_DEFAULTS
from .data import DEFAULT_REGISTRY


def get_parser():
    p = argparse.ArgumentParser(description="IR-TTA harness: method x subset x anchor x regime")
    g = p.add_argument_group('data')
    g.add_argument('--target', type=str, default=None, help='dataset name in the registry (configs/datasets.json)')
    g.add_argument('--cond_target', type=str, default=None,
                   help='control: adapt on this registry set\'s images, evaluate on --target (mismatched conditioning)')
    g.add_argument('--lq_dir', type=str, default=None, help='... or explicit LQ/GT folders')
    g.add_argument('--gt_dir', type=str, default=None)
    g.add_argument('--registry', type=str, default=DEFAULT_REGISTRY)
    g.add_argument('--crop_base', type=int, default=16)
    g.add_argument('--num_workers', type=int, default=8)
    g.add_argument('--max_images', type=int, default=0,
                   help='adapt and evaluate on an evenly spaced subset of this many images (validation sets: 300; 0: all)')

    g = p.add_argument_group('source model')
    g.add_argument('--ckpt', type=str, required=True)
    g.add_argument('--init', type=str, default='source', choices=['source', 'scratch'],
                   help='control: scratch = same architecture with random weights (use --subset all)')
    g.add_argument('--backbone', type=str, default='promptir', choices=list(BACKBONES),
                   help='source all-in-one model (dctta/models/__init__.py); --ckpt is its released checkpoint')
    g.add_argument('--backbone_opts', type=str, nargs='*', default=[],
                   help='backbone key=value options, e.g. moceir: routing_noise=False; r2r: bank_dir=<dir>')
    g.add_argument('--source', type=str, default=None,
                   help="source setting tag; default from the checkpoint path (3task / 5task, e.g. model.ckpt, "
                        "epoch=80.ckpt, adair3d, BioIR5D, MoCE_IR_AIO3, 5Deg_small, train_ckpt_3D_finetune)")

    g = p.add_argument_group('method')
    g.add_argument('--method', type=str, default='transfer', choices=sorted(METHOD_DEFAULTS))
    g.add_argument('--subset', type=str, default=None,
                   help="ln | lora<r> | full | all | re:<regex> | none, '+'-joined "
                        "(default: lora16 for transfer, full for dctta)")
    g.add_argument('--lora_alpha', type=float, default=None, help='LoRA alpha (default: = rank, scale 1)')
    g.add_argument('--lora_scope', type=str, default=None, help='regex on module names to restrict LoRA injection')
    g.add_argument('--lr', type=float, default=None, help='default: 1e-4 (transfer), 2e-4 (dctta)')
    g.add_argument('--beta1', type=float, default=0.9)
    g.add_argument('--beta2', type=float, default=0.999)
    g.add_argument('--iterations', type=int, default=1, help='optimiser steps per patch')
    g.add_argument('--pixel_weight', type=float, default=None)
    g.add_argument('--perceptual_weight', type=float, default=None)
    g.add_argument('--lowpass_weight', type=float, default=None)
    g.add_argument('--ema_decay', type=float, default=None, help='dctta teacher EMA decay (0.45)')
    # transfer
    g.add_argument('--families', type=str, default='compose',
                   help="synthetic degradations: compose (random physical-order chains) | all (one family per pair) | "
                        "oracle (the target's task) | '+'-joined subset of blur, noise, rain, haze, lowlight")
    g.add_argument('--bank', type=str, default=None,
                   help="clean content: a folder or a key of the registry's _banks (default: _banks['default'])")
    g.add_argument('--bank_max', type=int, default=300, help='clean images kept in RAM (evenly spaced)')
    g.add_argument('--bank_patch', type=int, default=192, help='clean crop size')
    g.add_argument('--bank_batch', type=int, default=4, help='synthetic pairs per optimiser step')
    g.add_argument('--exposure_match', type=str, default='fit', choices=['fit', 'clean', 'gain', 'camera', 'none'],
                   help='fit: the synthetic degraded image matches the target patch brightness, solved through the '
                        'chain (v0.2) | clean: the clean scene is gained to the target before degrading (v0.1) | '
                        'none | gain: v0, one gain on both images | camera: gain, except under-exposure chains '
                        'use fit (see irtta/degrade.py)')
    g.add_argument('--dark_q', type=float, default=0.0,
                   help='gain/camera matching: target patches darker than this quantile of clean-bank image '
                        'brightness are under-exposed; their pairs keep natural-exposure clean images (0 = off)')
    g.add_argument('--dark_anchor', action='store_true',
                   help='with --dark_q: clean images of under-exposed target patches take the frozen source\'s output '
                        'brightness for that patch instead of their natural exposure')
    g.add_argument('--dark_skip', action='store_true',
                   help='with --dark_q: skip the update on under-exposed target patches (abstain)')
    g.add_argument('--dark_rule', type=str, default='none', choices=['none', 'threshold', 'source'],
                   help='under-exposed target patches: none (legacy --dark_skip/--dark_anchor with --dark_q) | '
                        'threshold (mean < --dark_q quantile of the clean bank) | source (the frozen source '
                        'brightens the patch by >= --dark_ratio)')
    g.add_argument('--dark_ratio', type=float, default=1.414, help='--dark_rule source: brightening factor (sqrt 2)')
    g.add_argument('--dark_action', type=str, default='skip', choices=['skip', 'anchor', 'norm', 'natural'],
                   help='what --dark_rule does with an under-exposed patch (see TransferMethod); natural: clean '
                        'images keep their natural exposure, plain L1 (teaches brightening; for sources that '
                        'never learned low-light)')
    g.add_argument('--lowlight_op', type=str, default='v1', choices=['v1', 'v2'],
                   help='low-light operator: v1 global gain + gamma | v2 linear-light exposure field, noise before '
                        'a random tone curve (irtta/degrade.py)')
    g.add_argument('--blur_op', type=str, default='v1', choices=['v1', 'motion', 'mixed'],
                   help='blur kernels: v1 Gaussian/shake | motion long linear+curved motion (11-41 px, RwF) | mixed')
    g.add_argument('--bank_res', type=str, default='native', choices=['native', 'target'],
                   help='target: resize clean images to the target images\' typical long side before cropping')
    g.add_argument('--abc_topk', type=int, default=1, help='ABC: keep the k closest of --abc_k candidates per crop')
    g.add_argument('--nss_color', action='store_true',
                   help='add dark-channel and saturation statistics to the NSS features (haze vs blur)')
    g.add_argument('--abc_k', type=int, default=1,
                   help='ABC pair selection: draw k candidate degradations per clean crop, keep the one whose NSS '
                        'features are closest to the target set (Mahalanobis); 1 = off (irtta/nss.py)')
    g.add_argument('--abc_eps_q', type=float, default=0.0,
                   help='ABC tolerance: keep a selected pair only if its distance to the target Gaussian is within '
                        'this quantile of the target crops\' own distances; a step with no accepted pair is skipped '
                        '(0 = off, always accept the closest candidate)')
    g.add_argument('--abc_fit_n', type=int, default=256, help='crops used to fit the target / clean NSS Gaussians')
    g.add_argument('--oracle_pairs', type=str, default=None,
                   help='ORACLE diagnostic: train on real (LQ, GT) pairs of this registry set (e.g. the target\'s own '
                        'training split, --bank_max evenly spaced pairs) instead of synthesising pairs: the exact '
                        'target degradation operator, other images. Upper bound for any synthetic operator.')
    g.add_argument('--cons_weight', type=float, default=0.0,
                   help='target consistency L1(B f(x), B f_0(x)) on the target patch, f_0 = frozen source')
    g.add_argument('--cons_band', type=str, default='all', choices=['all', 'high'],
                   help='B for --cons_weight: all frequencies, or only the high-pass (x - Gaussian(2px))')
    g.add_argument('--redeg_weight', type=float, default=0.0,
                   help='re-degradation consistency L1(f(D(x)), f_0(x)), D a random chain at the target exposure')
    g.add_argument('--nss_weight', type=float, default=0.0,
                   help='output prior: Mahalanobis^2/D of NSS(f(x)) to the clean bank NSS Gaussian')
    g.add_argument('--eval_alphas', type=float, nargs='+', default=[1.0],
                   help='transductive transfer: also evaluate theta_0 + a*(theta - theta_0) for each a '
                        '(LoRA: delta scaled by a); each extra a writes <run>/alpha<a>/result.json')
    # flow (evaluation only)
    g.add_argument('--ext_alpha', type=float, default=1.0, help='flow: y <- y + ext_alpha * (f(y) - y)')
    g.add_argument('--n_apply', type=int, default=1, help='flow: number of applications')
    # dctta
    g.add_argument('--teacher_weight', type=float, default=5.0, help='dctta: pseudo-label loss weight (alpha)')
    g.add_argument('--gen_train_steps', type=int, default=5)
    g.add_argument('--gen_sampling_steps', type=int, default=5)
    g.add_argument('--gen_lr', type=float, default=2e-4)
    g.add_argument('--gen_sum_scale', type=float, default=0.01)
    g.add_argument('--gen_debug_dir', type=str, default=None)
    g.add_argument('--save_fisher', type=str, default=None, help='dctta: save the TIPS {name: [fisher, mask]} dict')

    g = p.add_argument_group('anchor')
    g.add_argument('--anchor', type=str, default=None, choices=['none', 'l2', 'tips'],
                   help='default: tips for dctta (released, dctta only), none otherwise')
    g.add_argument('--anchor_lambda', type=float, default=0.0, help='weight of the l2 anchor')
    g.add_argument('--tips_ratio', type=float, default=0.6, help='dctta: per-tensor fraction restored by TIPS')

    g = p.add_argument_group('regime')
    g.add_argument('--regime', type=str, default='transductive', choices=['transductive', 'online', 'episodic'])
    g.add_argument('--patch_size', type=int, default=320)
    g.add_argument('--batch_size', type=int, default=1, help='transductive patch batch size')
    g.add_argument('--epochs', type=int, default=1, help='transductive passes over the target set')
    g.add_argument('--max_steps', type=int, default=-1, help='transductive: stop after this many patches (-1: all)')
    g.add_argument('--steps_per_image', type=int, default=1, help='online/episodic: steps per streamed image')
    g.add_argument('--patches_per_step', type=int, default=1, help='online/episodic: random crops per step')
    g.add_argument('--stream_order', type=str, default='sorted', choices=['sorted', 'shuffled'])
    g.add_argument('--online_predict', type=str, default='after', choices=['after', 'before'])

    g = p.add_argument_group('seeds')
    g.add_argument('--seed', type=int, default=23)
    g.add_argument('--dataset_seed', type=int, default=None,
                   help='global re-seed inside the patch dataset (the original used 42); default: = --seed; -1 disables')

    g = p.add_argument_group('diagnostics / outputs')
    g.add_argument('--probe_list', type=str, default=None, help='probes/<target>.txt (name[,tag] per line)')
    g.add_argument('--probe_every', type=int, default=50, help='steps (transductive) or images (online)')
    g.add_argument('--log_stats', action='store_true', help='log per-step statistics (stats.jsonl + log)')
    g.add_argument('--lpips', action='store_true', help='also compute LPIPS (pyiqa or lpips package)')
    g.add_argument('--retention', action='store_true', help='evaluate the adapted model on the source sets')
    g.add_argument('--retention_sets', type=str, nargs='+', default=None,
                   help="registry names (default: the registry's _source_sets for --source)")
    g.add_argument('--retention_max', type=int, default=100, help='images per retention set (evenly spaced; 0: all)')
    g.add_argument('--save_images', action='store_true')
    g.add_argument('--save_ckpt', action='store_true', help='save the adapted state_dict to the run dir')
    g.add_argument('--out_root', type=str, default='results')
    g.add_argument('--exp', type=str, default='harness', help='experiment folder under --out_root')
    g.add_argument('--tag', type=str, default='', help='free-form suffix for the run id / aggregator group')
    g.add_argument('--overwrite', action='store_true', help='re-run even if result.json exists')
    g.add_argument('--gpu', type=str, default=None, help='CUDA_VISIBLE_DEVICES value (default: leave unchanged)')
    return p


def finalize(args):
    d = METHOD_DEFAULTS[args.method]
    for k in ('pixel_weight', 'perceptual_weight', 'lowpass_weight', 'ema_decay', 'anchor'):
        if getattr(args, k, None) is None:
            setattr(args, k, d.get(k, 0.0 if k != 'anchor' else 'none'))
    if args.subset is None:
        args.subset = {'dctta': 'full', 'transfer': 'lora16'}.get(args.method, 'none')
    if args.lr is None:
        args.lr = {'dctta': 2e-4, 'transfer': 1e-4}.get(args.method, 5e-5)
    if args.dataset_seed is None:
        args.dataset_seed = args.seed
    if args.method in ('source', 'identity', 'flow'):
        args.subset, args.anchor = 'none', 'none'
    if (args.target is None) == (args.lq_dir is None):
        raise SystemExit("give exactly one of --target or --lq_dir/--gt_dir")
    if args.lq_dir is not None and args.gt_dir is None:
        raise SystemExit("--lq_dir needs --gt_dir")
    return args


def get_args(argv=None):
    return finalize(get_parser().parse_args(argv))


# Arguments that change the numbers but are not already in the readable run id
# (method, subset, anchor+lambda, regime, target, source, lr, seeds, tag are).
SEMANTIC_ARGS = OrderedDict([
    ('families', 'fam'), ('bank', 'bk'), ('bank_max', 'bm'), ('bank_patch', 'bp'), ('bank_batch', 'bb'),
    ('exposure_match', 'em'), ('dark_q', 'dq'), ('dark_anchor', 'da'), ('dark_skip', 'dsk'), ('dark_rule', 'drl'), ('dark_ratio', 'drr'), ('dark_action', 'dac'), ('lowlight_op', 'llo'), ('blur_op', 'blo'), ('bank_res', 'bres'), ('abc_topk', 'abt'), ('nss_color', 'nc'), ('abc_k', 'abc'), ('abc_eps_q', 'abq'), ('abc_fit_n', 'abn'), ('cons_weight', 'cw'), ('cons_band', 'cbd'),
    ('redeg_weight', 'rw'), ('nss_weight', 'nw'), ('ext_alpha', 'ea'), ('n_apply', 'na'), ('ema_decay', 'ema'),
    ('iterations', 'it'), ('pixel_weight', 'pix'), ('perceptual_weight', 'vgg'), ('lowpass_weight', 'lp'),
    ('teacher_weight', 'tw'), ('beta1', 'b1'), ('beta2', 'b2'),
    ('lora_alpha', 'la'), ('lora_scope', 'lsc'), ('tips_ratio', 'tr'),
    ('gen_train_steps', 'gts'), ('gen_sampling_steps', 'gss'), ('gen_lr', 'glr'), ('gen_sum_scale', 'gsc'),
    ('patch_size', 'ps'), ('batch_size', 'bs'), ('epochs', 'ep'), ('max_steps', 'ms'),
    ('steps_per_image', 'spi'), ('patches_per_step', 'pps'), ('stream_order', 'so'), ('online_predict', 'op'),
    ('max_images', 'mi'), ('crop_base', 'cb'), ('backbone_opts', 'bo'), ('cond_target', 'ct'), ('init', 'ini'), ('oracle_pairs', 'orp'),
])
_TRANSFER_ARGS = ('families', 'bank', 'bank_max', 'bank_patch', 'bank_batch', 'exposure_match', 'dark_q', 'dark_anchor', 'dark_skip', 'dark_rule', 'dark_ratio', 'dark_action', 'lowlight_op', 'blur_op', 'bank_res', 'abc_topk', 'nss_color', 'abc_k', 'abc_eps_q',
                  'abc_fit_n', 'oracle_pairs',
                  'cons_weight', 'cons_band', 'redeg_weight', 'nss_weight')
_FLOW_ARGS = ('ext_alpha', 'n_apply')
_DCTTA_ARGS = ('teacher_weight', 'gen_train_steps', 'gen_sampling_steps', 'gen_lr', 'gen_sum_scale')
_IRRELEVANT = {
    'source': ('ema_decay', 'iterations', 'pixel_weight', 'perceptual_weight', 'lowpass_weight', 'beta1', 'beta2',
               'lora_alpha', 'lora_scope', 'tips_ratio', 'patch_size', 'batch_size', 'epochs', 'max_steps',
               'steps_per_image', 'patches_per_step', 'online_predict')
              + _TRANSFER_ARGS + _DCTTA_ARGS,
    'transfer': _DCTTA_ARGS + _FLOW_ARGS + ('ema_decay',),
    'dctta': _TRANSFER_ARGS + _FLOW_ARGS,
}
_IRRELEVANT['identity'] = _IRRELEVANT['source'] + _FLOW_ARGS
_IRRELEVANT['flow'] = _IRRELEVANT['source']
_IRRELEVANT['source'] = _IRRELEVANT['source'] + _FLOW_ARGS


def config_suffix(args):
    """Compact ``key=value`` string of every semantic argument that differs from its default for this
    method (e.g. ``t=ema_rdc=0.1_spi=5``); empty for a default configuration. Used in the run id and as
    a grouping key by the aggregator, so different configurations can never share a run folder."""
    ref = finalize(get_parser().parse_args(['--ckpt', args.ckpt, '--method', args.method, '--target', '_']))
    skip = set(_IRRELEVANT.get(args.method, ()))
    if args.regime == 'transductive':
        skip |= {'steps_per_image', 'patches_per_step', 'stream_order', 'online_predict'}
    else:
        skip |= {'epochs', 'max_steps', 'batch_size'}
    if args.anchor != 'tips':
        skip.add('tips_ratio')
    if 'lora' not in args.subset:
        skip |= {'lora_alpha', 'lora_scope'}
    parts = []
    for k, code in SEMANTIC_ARGS.items():
        if k in skip:
            continue
        v, d = getattr(args, k), getattr(ref, k)
        if v != d:
            if k == 'bank':
                v = os.path.basename(os.path.normpath(v))
            elif k == 'backbone_opts':
                v = '+'.join(v)
            parts.append(f"{code}={v:g}" if isinstance(v, float) else f"{code}={v}")
    return '_'.join(parts)
