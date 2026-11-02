"""DCTTA's degradation generator: an RDDM trained *online* on the target set.

This replaces three layers of wrapping in the original code
(``train_test_promptir.py::Degradation`` -> ``RDDM/net.py::ResidualDiffusionModel`` ->
``RDDM/src/...::Trainer``) with one class, keeping the exact optimisation recipe:

* U-Net ``dim=64, dim_mults=(1,2,4,8)``, randomly initialised -- there is **no** RDDM
  checkpoint; the generator starts from scratch and is trained continuously across
  the target set (weights and optimiser state persist between images).
* Per test image: ``train_steps`` (=5) RAdam steps (lr 2e-4, betas (0.9, 0.999)) on the
  single pair (x_in, y_bar), loss = MSE(pred_res, y_bar - x_in) / 2 (the "/2" is the
  original ``gradient_accumulate_every=2`` divisor, applied without accumulation),
  gradient-norm clipping at 1.0.
* An ``ema_pytorch.EMA`` shadow (beta 0.995, ``update_every=10``, default
  ``update_after_step=100``) is what actually produces samples. Note that with
  ema-pytorch's warm-up schedule the effective decay is
  ``min(0.995, 1 - (1 + k)^(-2/3))`` with ``k = step - 101``: during the first 100
  optimiser steps (20 images) the EMA is a plain copy of the online model, and the
  decay only approaches 0.98 after ~400 steps. Keep the library to keep this behaviour.
* Sampling: 5 DDIM steps conditioned on x_in, output = redegraded image x_sd in [0, 1].

Removed side effects: the original saved a ``model.pt`` checkpoint and every
intermediate DDIM state as PNGs for *every* test image (``Trainer.save`` /
``Trainer.test``). The optional ``debug_dir`` argument restores only the final
x_sd dump, which is useful for mechanism analysis.
"""
import os

import torch
from ema_pytorch import EMA
from torch.optim import RAdam
from torchvision.utils import save_image

from .diffusion import ResidualDiffusion
from .unet import Unet


class ResidualDegradationGenerator:
    def __init__(self, device, *, dim=64, dim_mults=(1, 2, 4, 8), timesteps=1000, sampling_timesteps=5,
                 sum_scale=0.01, train_steps=5, lr=2e-4, loss_divisor=2., clip_grad_norm=1.0,
                 ema_decay=0.995, ema_update_every=10, ema_update_after_step=100, debug_dir=None):
        self.device = torch.device(device)
        self.train_steps = train_steps
        self.loss_divisor = loss_divisor
        self.clip_grad_norm = clip_grad_norm
        self.debug_dir = debug_dir
        self.total_steps = 0

        unet = Unet(dim=dim, dim_mults=dim_mults, condition=True)
        self.diffusion = ResidualDiffusion(unet, timesteps=timesteps, sampling_timesteps=sampling_timesteps,
                                           sum_scale=sum_scale, loss_type='l2')
        # Original order: optimiser and EMA copy were created on CPU *before* the model was
        # moved to the device by accelerate.prepare(); replicated for identical RNG consumption.
        self.optimizer = RAdam(self.diffusion.parameters(), lr=lr, weight_decay=0.0)
        self.ema = EMA(self.diffusion, beta=ema_decay, update_every=ema_update_every,
                       update_after_step=ema_update_after_step)
        self.diffusion.to(self.device)
        self.ema.to(self.device)

        if debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def fit(self, x_in, y_bar):
        """A few optimisation steps on one (degraded, pseudo-label) pair. Returns the mean loss."""
        self.diffusion.train()
        losses = []
        for _ in range(self.train_steps):
            loss = self.diffusion(x_in, y_bar) / self.loss_divisor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.total_steps += 1
            self.ema.update()
            losses.append(loss.item())
        return sum(losses) / len(losses)

    @torch.no_grad()
    def sample(self, x_in, name=None):
        """Redegrade ``x_in`` (in [0, 1]) with the EMA generator. Returns x_sd in [0, 1] (not clamped)."""
        self.ema.ema_model.eval()
        x_sd = self.ema.ema_model.sample(x_in)
        if self.debug_dir is not None and name is not None:
            save_image(x_sd, os.path.join(self.debug_dir, f'{os.path.splitext(name)[0]}_xsd.png'))
        return x_sd

    def fit_and_sample(self, x_in, y_bar, name=None):
        """The original ``Degradation.train(input, target, name)``: train, then sample."""
        self.fit(x_in, y_bar)
        return self.sample(x_in, name)

    def state_dict(self):
        return {'model': self.diffusion.state_dict(), 'ema': self.ema.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'total_steps': self.total_steps}
