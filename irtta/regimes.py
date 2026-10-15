"""The three regimes, reported separately (never mixed):

* **transductive** -- adapt on random patches of the whole target set (DCTTA's protocol: one pass,
  batch 1, ``iterations`` steps per patch, ``--epochs`` passes), then predict every image with the
  final weights.
* **online** -- stream the target images once; for each image take ``--steps_per_image`` steps on
  random crops of *that* image, predict it, keep the weights for the next image.
* **episodic** -- as online, but reset to theta_0 (weights, optimiser state, teacher) after every
  image (TAO/SRTTA-comparable per-image adaptation).

``--online_predict after`` (default) predicts an image after adapting on it; ``before`` predicts
with the weights from the previous images only (strictly causal).
Timing: ``adapt_time`` covers only optimisation steps (+ ``prepare``, added by the caller);
prediction time is reported separately; probe time is excluded from both.
"""
import json
import logging
import os
import time

import torch
from tqdm import tqdm

from dctta.utils import save_image_tensor

from .data import random_crops
from .evaluation import image_row, summarize

logger = logging.getLogger(__name__)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StepLog:
    """``stats.jsonl``: one line per adaptation step (loss + ``method.last_stats``)."""

    def __init__(self, path, log_stats=False):
        self.f = open(path, 'w') if path else None
        self.log_stats = log_stats

    def write(self, step, name, loss, stats, **extra):
        rec = {'step': step, 'name': name, 'loss': loss, **extra, **(stats or {})}
        if self.f:
            self.f.write(json.dumps(rec) + '\n')
        msg = f"[{step}] {name} loss {loss:.5f}" if loss is not None else f"[{step}] {name} (skipped)"
        if self.log_stats and stats:
            msg += "  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in stats.items())
        logger.info(msg)

    def close(self):
        if self.f:
            self.f.close()


def run_transductive(method, loader, args, device, step_log, probe=None):
    """One pass per epoch over the target patches. With ``--max_steps N >= 0`` the loader is cycled until N steps
    are taken (small target sets get the full budget); with ``--max_steps -1`` it runs ``--epochs`` passes."""
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
            step_log.write(step, names[0], loss, method.last_stats, epoch=epoch)
            step += 1
            if probe is not None and args.probe_every > 0 and step % args.probe_every == 0:
                probe(method.eval_model, step)
        epoch += 1
        if done or (0 <= args.max_steps <= step):
            break
    return {'adapt_time_s': t_adapt, 'patches_seen': step}


@torch.no_grad()
def _predict(method, lq):
    net = method.eval_model
    was = net.training
    net.eval()
    out = net(lq)
    net.train(was)
    return out


def run_stream(method, dataset, args, device, step_log, episodic, probe=None, lpips_metric=None, save_dir=None):
    """Online (``episodic=False``) or episodic stream over ``dataset`` (full images)."""
    crop_gen = torch.Generator().manual_seed(int(args.seed) + 1)
    order = list(range(len(dataset)))
    if args.stream_order == 'shuffled':
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(int(args.seed) + 2)).tolist()
    if args.max_images and args.max_images > 0:
        order = order[:args.max_images]

    rows, t_adapt, t_pred, step = [], 0.0, 0.0, 0
    for pos, idx in enumerate(tqdm(order, desc='episodic' if episodic else 'online')):
        name, lq, gt = dataset[idx]
        lq, gt = lq[None].to(device), gt[None].to(device)

        def predict():
            _sync()
            t0 = time.time()
            out, w = _predict(method, lq), {}
            _sync()
            return out, w, time.time() - t0

        if args.online_predict == 'before':
            out, wcols, dt = predict()
            t_pred += dt

        steps_here = 0
        for _ in range(args.steps_per_image):
            patch = random_crops(lq, args.patch_size, args.patches_per_step, crop_gen)
            _sync()
            t0 = time.time()
            loss = method.step(patch, name=name)
            _sync()
            t_adapt += time.time() - t0
            step_log.write(step, name, loss, method.last_stats, pos=pos)
            step += 1
            steps_here += 1

        if args.online_predict == 'after':
            out, wcols, dt = predict()
            t_pred += dt

        row = {'pos': pos, **image_row(name, lq, out, gt, lpips_metric), 'steps': steps_here, **wcols}
        rows.append(row)
        if save_dir is not None:
            save_image_tensor(out, os.path.join(save_dir, os.path.splitext(name)[0] + '.png'))
        if episodic:
            method.reset()
        elif probe is not None and args.probe_every > 0 and (pos + 1) % args.probe_every == 0:
            probe(method.eval_model, pos + 1)

    return summarize(rows), rows, {'adapt_time_s': t_adapt, 'predict_time_s': t_pred, 'patches_seen': step}
