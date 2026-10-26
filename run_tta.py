"""IR-TTA harness: one run = (method, backbone, source, target, subset, anchor, regime, seed).

    # bars: unprocessed input, un-adapted source, the source applied twice (any --backbone)
    python run_tta.py --method identity --target HIDE --ckpt pretrain/epoch=80.ckpt
    python run_tta.py --method source   --target HIDE --ckpt pretrain/epoch=80.ckpt
    python run_tta.py --method flow --n_apply 2 --target HIDE --backbone r2r \
        --ckpt pretrain/r2r/train_ckpt_5D_finetune/last.ckpt

    # the released DCTTA (== adapt.py for --seed 23 --dataset_seed 42)
    python run_tta.py --method dctta --target Rain100H --ckpt pretrain/model.ckpt --seed 23 --dataset_seed 42

    # transfer TTA (parked): LoRA16, physical-order degradation chains on clean bank crops
    python run_tta.py --method transfer --max_steps 500 --target HIDE_val --max_images 300 --ckpt pretrain/epoch=80.ckpt

Every run writes ``<out_root>/<exp>/<run_id>/``: ``run.log``, ``config.json``, ``result.json`` (one
aggregator row), ``per_image.csv``, ``stats.jsonl`` (per step), ``probe.csv`` (if ``--probe_list``).
``tools/aggregate.py`` collects the result.json files into CSV tables.
"""
import json
import logging
import os
import re
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader

from dctta.data import AdaptationPatchDataset
from dctta.models import TRAIN_PREFIXES, build_model, parse_backbone_opts
from dctta.utils import set_seed
from irtta.anchors import build_anchor
from irtta.config import config_suffix, get_args
from irtta.data import (available_names, evenly_spaced, load_registry, make_eval_dataset, resolve_bank, resolve_spec,
                        source_sets, spec_from_dirs, typical_long_side)
from irtta.evaluation import LPIPSMetric, Probe, eval_loader, evaluate_detailed, write_rows
from irtta.methods import build_method, rng_checksum
from irtta.regimes import StepLog, run_stream, run_transductive
from irtta.subsets import apply_subset, summarize_subset

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger('run_tta')


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


def _safe(s):
    return re.sub(r'[^A-Za-z0-9._=-]+', '_', str(s))


def make_run_id(args, target, source):
    anchor = args.anchor + (f"{args.anchor_lambda:g}" if args.anchor == 'l2' else '')
    parts = [args.method, _safe(args.subset), anchor, args.regime, target, source]
    if args.backbone != 'promptir':      # PromptIR ids are unchanged, so earlier PromptIR runs stay cached
        parts.insert(0, args.backbone)
    if args.method not in ('source', 'identity', 'flow'):
        parts.append(f"lr{args.lr:g}")
    parts.append(f"s{args.seed}" + (f"d{args.dataset_seed}" if args.dataset_seed != args.seed else ''))
    cfg = config_suffix(args)
    if cfg:
        parts.append(_safe(cfg))
    if args.tag:
        parts.append(_safe(args.tag))
    return '-'.join(parts)


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
    """numpy / torch scalars and arrays -> plain Python (numpy 2 keeps float32 results as float32)."""
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


def main(argv=None):
    args = get_args(argv)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    registry = load_registry(args.registry)
    spec = resolve_spec(args.target, registry) if args.target else spec_from_dirs(args.lq_dir, args.gt_dir)
    source = args.source or infer_source(args.ckpt)
    args.target_task = spec.task
    args.bank_path = resolve_bank(args.bank, registry) if args.method == 'transfer' else None
    # --cond_target: adapt on (= condition the synthesis on) ANOTHER set's images, evaluate on the target (a control:
    # if the gain comes from inferring the target's degradations, it must vanish when conditioned on the wrong set)
    cspec = resolve_spec(args.cond_target, registry) if args.cond_target else spec
    args.target_long = typical_long_side(cspec) if (args.method == 'transfer' and args.bank_res == 'target') else None
    run_id = make_run_id(args, spec.name, source)
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
        json.dump({**vars(args), 'run_id': run_id, 'source_tag': source, 'git': commit,
                   'target_lq': spec.lq, 'target_gt': spec.gt}, f, indent=2, default=_json_default)
    t_start = time.time()

    # --- construction order follows adapt.py (RNG parity for the DCTTA reference) ---
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = args.regime in ('online', 'episodic')
    adapts = args.method not in ('source', 'identity', 'flow')
    # --max_images: adapt and evaluate on the SAME evenly spaced subset (transductive on that subset)
    subset_names = evenly_spaced(available_names(spec), args.max_images) if args.max_images else None

    loader = None
    if args.regime == 'transductive' and adapts:
        if cspec.synthetic:
            raise SystemExit(f"{cspec.name} has no LQ folder; transductive adaptation needs stored LQ images")
        trainset = AdaptationPatchDataset(cspec.lq, cspec.gt, patch_size=args.patch_size,
                                          seed=None if args.dataset_seed < 0 else args.dataset_seed,
                                          max_side=cspec.max_side, names=subset_names if cspec is spec else None)
        loader = DataLoader(trainset, batch_size=args.batch_size, pin_memory=True, shuffle=True, drop_last=True,
                            num_workers=args.num_workers)
        logger.info(f"target set: {len(loader)} patches per epoch")

    model = build_model(args.backbone, args.ckpt, logger, **parse_backbone_opts(args.backbone_opts))
    if args.init == 'scratch':   # control: same architecture, random weights (no source knowledge)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 11)
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
        logger.info("--init scratch: every layer re-initialised; the source checkpoint is not used")
    model = model.to(device)
    n_total = sum(p.numel() for p in model.parameters())

    args.train_prefixes = TRAIN_PREFIXES[args.backbone]   # DCTTA's 'full' rule, per backbone
    subset = apply_subset(model, args.subset, lora_alpha=args.lora_alpha, lora_scope=args.lora_scope,
                          train_prefixes=args.train_prefixes,
                          seed=args.seed, logger=logger)
    if subset.params:
        logger.info("adapted per module: " + ", ".join(f"{k} {v:,}" for k, v in summarize_subset(subset).items()))
    anchor = build_anchor(args.anchor, model, subset, lam=args.anchor_lambda, method=args.method)
    method = build_method(args.method, model, subset, anchor, args, device)
    checksum = getattr(method, 'checksum', None)
    if checksum is None:
        checksum = rng_checksum()
        logger.info(f"CPU RNG checksum after model/method construction: {checksum}")
    lpips_metric = LPIPSMetric(device) if args.lpips else None

    # --- prepare (TIPS Fisher on the target loader, if any) ---
    _sync()
    t0 = time.time()
    if adapts:
        method.prepare(loader)
    _sync()
    prepare_time = time.time() - t0

    probe = None
    if args.probe_list and adapts and args.regime != 'episodic':
        if spec.synthetic:
            raise SystemExit("probes need stored LQ images")
        probe = Probe(args.probe_list, spec.lq, spec.gt, device, base=args.crop_base,
                      csv_path=os.path.join(run_dir, 'probe.csv'), max_side=spec.max_side)
        probe(method.eval_model, 0)

    step_log = StepLog(os.path.join(run_dir, 'stats.jsonl'), args.log_stats)
    img_dir = os.path.join(run_dir, 'images') if args.save_images else None
    if img_dir:
        os.makedirs(img_dir, exist_ok=True)

    if not stream:
        info = (run_transductive(method, loader, args, device, step_log, probe) if adapts
                else {'adapt_time_s': 0.0, 'patches_seen': 0})
        peak_adapt = _peak_gb()
        if args.save_ckpt and adapts:
            torch.save(model.state_dict(), os.path.join(run_dir, 'adapted.pt'))
        ds = make_eval_dataset(spec, base=args.crop_base, max_images=args.max_images or None)
        _sync()
        t0 = time.time()
        summary, rows = evaluate_detailed(method.eval_model, eval_loader(ds, args.num_workers), device,
                                          results_dir=img_dir, log=logger, lpips_metric=lpips_metric)
        _sync()
        info['predict_time_s'] = time.time() - t0
    else:
        if not adapts:
            args.steps_per_image = 0
        ds = make_eval_dataset(spec, base=args.crop_base, max_images=args.max_images or None)
        summary, rows, info = run_stream(method, ds, args, device, step_log, episodic=args.regime == 'episodic',
                                         probe=probe, lpips_metric=lpips_metric, save_dir=img_dir)
        peak_adapt = _peak_gb()
        if args.save_ckpt and adapts and args.regime == 'online':
            torch.save(model.state_dict(), os.path.join(run_dir, 'adapted.pt'))
    step_log.close()
    write_rows(os.path.join(run_dir, 'per_image.csv'), rows)
    logger.info(f"{args.method} [{args.regime}]  PSNR {summary['psnr']:.3f}  SSIM {summary['ssim']:.4f}  "
                f"LPIPS {summary['lpips']:.4f}  drift {summary['drift']:+.4f}  ({spec.name}, {source})")

    # --- source-domain retention ---
    def retention_eval():
        out = {}
        if not args.retention:
            return out
        if args.regime == 'episodic':
            logger.info("retention skipped: the episodic model is reset to theta_0 after every image")
            return out
        for set_name in (args.retention_sets or source_sets(source, registry)):
            rs = resolve_spec(set_name, registry)
            rds = make_eval_dataset(rs, base=args.crop_base, max_images=args.retention_max or None)
            rsum, _ = evaluate_detailed(method.eval_model, eval_loader(rds, args.num_workers), device)
            out[set_name] = {'psnr': rsum['psnr'], 'ssim': rsum['ssim'], 'n': rsum['n']}
            logger.info(f"[retention] {set_name}: PSNR {rsum['psnr']:.3f}  SSIM {rsum['ssim']:.4f}  (n={rsum['n']})")
        return out

    retention = retention_eval()
    adapt_time = info['adapt_time_s'] + prepare_time
    result = {
        'run_id': run_id, 'exp': args.exp, 'tag': args.tag,
        'method': args.method, 'backbone': args.backbone, 'source': source, 'ckpt': args.ckpt,
        'target': spec.name, 'split': spec.split, 'regime': args.regime,
        'subset': args.subset, 'anchor': args.anchor,
        'anchor_lambda': args.anchor_lambda if args.anchor == 'l2' else 0.0,
        'lr': args.lr if adapts else 0.0,
        'families': args.families if args.method == 'transfer' else '',
        'families_used': getattr(method, 'fam_counts', None),
        'bank': args.bank_path or '',
        'exposure_match': args.exposure_match if args.method == 'transfer' else '',
        'ext_alpha': args.ext_alpha if args.method == 'flow' else None,
        'n_apply': args.n_apply if args.method == 'flow' else None,
        'eval_alpha': 1.0,
        'epochs': args.epochs, 'max_steps': args.max_steps, 'iterations': args.iterations,
        'steps_per_image': args.steps_per_image, 'max_images': args.max_images,
        'cfg': config_suffix(args),
        'seed': args.seed, 'dataset_seed': args.dataset_seed,
        'n_images': summary['n'], 'psnr': summary['psnr'], 'ssim': summary['ssim'], 'psnr_y': summary.get('psnr_y'),
        'lpips': summary['lpips'],
        'drift': summary['drift'],
        'steps': method.steps, 'skipped': method.skipped, 'patches_seen': info.get('patches_seen', 0),
        'prepare_time_s': prepare_time, 'adapt_time_s': adapt_time, 'predict_time_s': info.get('predict_time_s', 0.0),
        'total_time_s': time.time() - t_start,
        'peak_mem_adapt_gb': peak_adapt, 'peak_mem_gb': _peak_gb(),
        'n_params_adapted': method.n_params_adapted, 'n_params_effective': method.n_params_effective,
        'n_params_total': n_total, 'rng_checksum': checksum, 'git': commit,
        'retention': retention,
        'probe_final': probe.history[-1] if probe is not None and probe.history else None,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    }
    with open(os.path.join(run_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2, default=_json_default)
    logger.info(f"adapt {adapt_time:.1f}s (prepare {prepare_time:.1f}s), predict {result['predict_time_s']:.1f}s, "
                f"peak {result['peak_mem_gb']:.2f} GB, adapted params {result['n_params_adapted']:,} "
                f"(effective {result['n_params_effective']:,})  -> {run_dir}")

    # --- the same adapted model at other weight mixes theta_0 + a * (theta - theta_0) ---
    alphas = sorted({float(a) for a in args.eval_alphas if float(a) != 1.0})
    if alphas and adapts and not stream and hasattr(method, 'model_state'):
        mixer = WeightMixer(method)
        for a in alphas:
            mixer.set(a)
            ds = make_eval_dataset(spec, base=args.crop_base, max_images=args.max_images or None)
            t0 = time.time()
            asum, arows = evaluate_detailed(method.eval_model, eval_loader(ds, args.num_workers), device,
                                            lpips_metric=lpips_metric)
            ares = dict(result, run_id=f"{run_id}-a{a:g}", eval_alpha=a, n_images=asum['n'], psnr=asum['psnr'],
                        psnr_y=asum.get('psnr_y'),
                        ssim=asum['ssim'], lpips=asum['lpips'], drift=asum['drift'], probe_final=None,
                        predict_time_s=time.time() - t0, retention=retention_eval())
            adir = os.path.join(run_dir, f"alpha{a:g}")
            os.makedirs(adir, exist_ok=True)
            with open(os.path.join(adir, 'result.json'), 'w') as f:
                json.dump(ares, f, indent=2, default=_json_default)
            write_rows(os.path.join(adir, 'per_image.csv'), arows)
            logger.info(f"[alpha {a:g}] PSNR {asum['psnr']:.3f}  SSIM {asum['ssim']:.4f}  drift {asum['drift']:+.4f}")
        mixer.set(1.0)
    elif alphas:
        logger.info("--eval_alphas ignored: only for transductive transfer runs")
    print(f"PSNR {summary['psnr']:.3f}  SSIM {summary['ssim']:.4f}")
    return result


class WeightMixer:
    """Sets the adapted parameters to theta_0 + a * (theta - theta_0). LoRA: the delta's scale is multiplied by a
    (exactly a * Delta W); other adapted tensors are interpolated against ``method.model_state`` (theta_0)."""

    def __init__(self, method):
        sub = method.subset
        lora_ids = sub.lora_param_ids()
        self.lora = [(m, m.scale) for m in sub.lora_modules.values()]
        theta0 = method.model_state
        self.plain = [(p, theta0[n].to(p.device).clone(), p.detach().clone())
                      for n, p in zip(sub.names, sub.params) if id(p) not in lora_ids]

    @torch.no_grad()
    def set(self, a):
        for m, scale in self.lora:
            m.scale = scale * a
        for p, p0, p1 in self.plain:
            p.copy_(p0 + a * (p1 - p0))


if __name__ == '__main__':
    sys.exit(0 if main() is not None else 1)
