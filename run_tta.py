"""One run: adapt a pretrained all-in-one restoration model to an unlabelled test set, then evaluate it.

    # ours (defaults = the final method; families follow the source setting)
    python run_tta.py --target DPDD --backbone promptir --ckpt pretrain/epoch=80.ckpt --tag F1
    # baselines: the pretrained model, the released DCTTA
    python run_tta.py --method source --target DPDD --ckpt pretrain/epoch=80.ckpt
    python run_tta.py --method dctta  --target DPDD --ckpt pretrain/epoch=80.ckpt

Protocol (transductive): the method sees the unlabelled degraded test images (random ``--patch_size`` patches), then
the adapted model restores every test image; ground truth is used only for the metrics. Every run writes
``<out_root>/<exp>/<run_id>/``: ``run.log``, ``config.json``, ``result.json`` and ``per_image.csv``; a finished run
is skipped. ``tools/aggregate.py`` and ``tools/table.py`` turn the result files into the paper tables.
"""
import json
import logging
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from irtta.backbones import TRAIN_PREFIXES, build_model, parse_backbone_opts
from irtta.config import config_suffix, get_args, make_run_id
from irtta.data import (DEFAULT_REGISTRY, AdaptationPatchDataset, available_names, evenly_spaced, load_registry,
                        make_eval_dataset, materialize_synthetic, resolve_bank, resolve_spec, spec_from_dirs)
from irtta.evaluation import eval_loader, evaluate_detailed, write_rows
from irtta.methods import build_method, rng_checksum
from irtta.subsets import apply_subset, summarize_subset

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger('run_tta')


def set_seed(seed=23):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_info():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO_ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                             cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip())
        return commit + ('-dirty' if dirty else '')
    except Exception:  # noqa: BLE001
        return 'unknown'


def setup_logging(run_dir):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True,
                        handlers=[logging.FileHandler(os.path.join(run_dir, 'run.log'), mode='w'),
                                  logging.StreamHandler()])


def _json_default(o):
    if hasattr(o, 'tolist'):
        return o.tolist()
    if hasattr(o, 'item'):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_gb():
    return torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0


def randomize_(model, seed):
    """The random-initialisation control: every layer re-initialised (no knowledge from the checkpoint)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 11)
        for m in model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            else:   # raw parameters (custom LayerNorms, attention temperatures, PromptIR's prompt bank)
                for n, p in m.named_parameters(recurse=False):
                    if p.dim() == 1:
                        p.data.fill_(1.0 if ('weight' in n or 'temperature' in n) else 0.0)
                    elif 'temperature' in n:
                        p.data.fill_(1.0)
                    else:
                        p.data.uniform_(0.0, 1.0)   # PromptIR initialises its prompts with torch.rand


def adapt(method, loader, args, device):
    """Steps on random target patches; ``--max_steps N >= 0`` cycles the loader until N steps (small target sets
    get the full budget), ``-1`` runs ``--epochs`` passes."""
    if len(loader) == 0:
        raise ValueError("empty target loader")
    step, t_adapt, epoch = 0, 0.0, 0
    while True:
        if args.max_steps < 0 and epoch >= args.epochs:
            break
        done = False
        for names, x, _ in tqdm(loader, total=len(loader), desc=f"Adapting (epoch {epoch})"):
            if 0 <= args.max_steps <= step:
                done = True
                break
            _sync()
            t0 = time.time()
            loss = method.step(x.to(device), name=names[0])
            _sync()
            t_adapt += time.time() - t0
            if loss is not None and (step % 50 == 0):
                logger.info(f"[{step}] {names[0]} loss {loss:.5f}")
            step += 1
        epoch += 1
        if done or (0 <= args.max_steps <= step):
            break
    return t_adapt


def main(argv=None):
    args = get_args(argv)
    registry = load_registry(args.registry or DEFAULT_REGISTRY)
    args.registry = args.registry or DEFAULT_REGISTRY
    spec = resolve_spec(args.target, registry) if args.target else spec_from_dirs(args.lq_dir, args.gt_dir)
    args.target_task = spec.task
    adapts = args.method not in ('source', 'identity')
    args.bank_path = resolve_bank(args.bank, registry) if args.method == 'transfer' else None
    # --cond_target (control): adapt on ANOTHER set's images, evaluate on the target
    cspec = resolve_spec(args.cond_target, registry) if args.cond_target else spec
    if adapts and cspec.synthetic:   # synthetic-noise sets (BSD68): write the identical pairs once, adapt on those
        same = cspec is spec
        cspec = materialize_synthetic(cspec, base=args.crop_base)
        spec = cspec if same else spec
    run_id = make_run_id(args, spec.name)
    run_dir = os.path.join(args.out_root, args.exp, run_id)
    if os.path.exists(os.path.join(run_dir, 'result.json')) and not args.overwrite:
        print(f"[skip] {run_dir} has result.json (pass --overwrite to re-run)")
        return json.load(open(os.path.join(run_dir, 'result.json')))
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(run_dir)
    commit = git_info()
    logger.info(f"run {run_id}  (commit {commit})")
    logger.info(vars(args))
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({**vars(args), 'run_id': run_id, 'git': commit, 'target_lq': spec.lq, 'target_gt': spec.gt},
                  f, indent=2, default=_json_default)
    t_start = time.time()

    # construction order of the released DCTTA code (seed -> patch dataset -> model -> method): same RNG stream
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset_names = evenly_spaced(available_names(spec), args.max_images) if args.max_images else None
    loader = None
    if adapts:
        trainset = AdaptationPatchDataset(cspec.lq, cspec.gt, patch_size=args.patch_size,
                                          seed=None if args.dataset_seed < 0 else args.dataset_seed,
                                          max_side=cspec.max_side, names=subset_names if cspec is spec else None)
        loader = DataLoader(trainset, batch_size=args.batch_size, pin_memory=True, shuffle=True, drop_last=True,
                            num_workers=args.num_workers)
        logger.info(f"target set: {len(loader)} patches per epoch")

    model = build_model(args.backbone, args.ckpt, logger, **parse_backbone_opts(args.backbone_opts))
    if args.init == 'scratch':
        randomize_(model, args.seed)
        logger.info("--init scratch: every layer re-initialised; the checkpoint is not used")
    model = model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    subset = apply_subset(model, args.subset, train_prefixes=TRAIN_PREFIXES[args.backbone], seed=args.seed,
                          logger=logger)
    if subset.params:
        logger.info("adapted per module: " + ", ".join(f"{k} {v:,}" for k, v in summarize_subset(subset).items()))
    method = build_method(args.method, model, subset, args, device)
    checksum = getattr(method, 'checksum', None)
    if checksum is None:
        checksum = rng_checksum()
        logger.info(f"CPU RNG checksum after model/method construction: {checksum}")

    _sync()
    t0 = time.time()
    if adapts:
        method.prepare(loader)
    _sync()
    prepare_time = time.time() - t0
    if adapts:   # after prepare: DCTTA's TIPS Fisher pass is not adaptation
        method.track_updates()
    t_steps = adapt(method, loader, args, device) if adapts else 0.0
    peak_adapt = _peak_gb()

    ds = make_eval_dataset(spec, base=args.crop_base, max_images=args.max_images or None)
    img_dir = os.path.join(run_dir, 'images') if args.save_images else None
    _sync()
    t0 = time.time()
    summary, rows = evaluate_detailed(method.eval_model, eval_loader(ds, args.num_workers), device,
                                      results_dir=img_dir, log=logger)
    _sync()
    predict_time = time.time() - t0
    write_rows(os.path.join(run_dir, 'per_image.csv'), rows)
    logger.info(f"{args.method}  PSNR {summary['psnr']:.3f}  SSIM {summary['ssim']:.4f}  "
                f"Y-PSNR {summary['psnr_y']:.3f}  ({spec.name}, {args.source})")

    adapt_time = t_steps + prepare_time
    result = {
        'run_id': run_id, 'exp': args.exp, 'tag': args.tag,
        'method': args.method, 'backbone': args.backbone, 'source': args.source, 'ckpt': args.ckpt,
        'target': spec.name, 'split': spec.split, 'regime': args.regime,
        'subset': args.subset, 'anchor': args.anchor, 'lr': args.lr if adapts else 0.0,
        'families': args.families if args.method == 'transfer' else '',
        'families_used': getattr(method, 'fam_counts', None),
        'exposure_match': args.exposure_match if args.method == 'transfer' else '',
        'eval_alpha': 1.0, 'max_steps': args.max_steps, 'max_images': args.max_images,
        'cfg': config_suffix(args), 'seed': args.seed, 'dataset_seed': args.dataset_seed,
        'n_images': summary['n'], 'psnr': summary['psnr'], 'ssim': summary['ssim'], 'psnr_y': summary['psnr_y'],
        'steps': method.steps, 'skipped': method.skipped,
        'prepare_time_s': prepare_time, 'adapt_time_s': adapt_time, 'predict_time_s': predict_time,
        'total_time_s': time.time() - t_start, 'peak_mem_adapt_gb': peak_adapt, 'peak_mem_gb': _peak_gb(),
        'n_params_adapted': method.n_params_adapted, 'n_params_effective': method.n_params_effective,
        'n_params_updated': method.n_params_updated,
        'n_params_total': n_total, 'rng_checksum': checksum, 'git': commit,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    }
    with open(os.path.join(run_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2, default=_json_default)
    logger.info(f"adapt {adapt_time:.1f}s (prepare {prepare_time:.1f}s), predict {predict_time:.1f}s, "
                f"peak {result['peak_mem_gb']:.2f} GB, adapted params {result['n_params_adapted']:,} "
                f"(updated {result['n_params_updated'] if result['n_params_updated'] is not None else '-'})"
                f"  -> {run_dir}")
    print(f"PSNR {summary['psnr']:.3f}  SSIM {summary['ssim']:.4f}  Y-PSNR {summary['psnr_y']:.3f}")
    return result


if __name__ == '__main__':
    sys.exit(0 if main() is not None else 1)
