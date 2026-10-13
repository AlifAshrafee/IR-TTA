"""Smoke check for the ported backbones and their released checkpoints (no data needed, ~1 min on GPU).

    python tools/check_backbones.py adair=pretrain/adair/adair3d.ckpt bioir=pretrain/bioir/BioIR3D.ckpt \
        moceir=pretrain/moceir/MoCE_IR_AIO3/last.ckpt mirage=pretrain/mirage/3Deg_small.ckpt \
        r2r=pretrain/r2r/train_ckpt_3D_finetune/last.ckpt

For each ``backbone=checkpoint`` pair: builds the model (the log says whether every checkpoint tensor
matched), runs a 320x320 forward in eval mode, and checks that ``model.train()`` does not change the
output (DCTTA and our TTA call ``model.train()``; only parameter updates may change the function).
MoCE-IR is checked with ``routing_noise=False`` because its official router adds noise even in eval.
The real reproduction check is ``test.py`` on an in-distribution set (e.g. Rain100L) against the
paper's number; for R2R that needs ``--backbone_opts oracle_label=<task>`` (see dctta/models/r2r.py).
"""
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dctta.models import build_model  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pairs = [a.split('=', 1) for a in sys.argv[1:]]
    if not pairs:
        print(__doc__)
        return
    torch.manual_seed(0)
    x = torch.rand(1, 3, 320, 320, device=device)
    for name, ckpt in pairs:
        opts = {'routing_noise': False} if name == 'moceir' else {}
        model = build_model(name, ckpt, **opts).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        with torch.no_grad():
            model.eval()
            y_eval = model(x)
            model.train()
            y_train = model(x)
        ok = y_eval.shape == x.shape and torch.isfinite(y_eval).all()
        diff = (y_eval - y_train).abs().max().item()
        print(f"{name:8s} {n_params:6.2f}M params  out {tuple(y_eval.shape)}  finite+shape {'OK' if ok else 'FAIL'}  "
              f"|train-eval| {diff:.2e} {'OK' if diff < 1e-4 else 'FAIL'}  mean|out-in| {(y_eval - x).abs().mean():.4f}")
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
