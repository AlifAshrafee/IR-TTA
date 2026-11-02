"""Residual diffusion process (RDDM) restricted to the configuration DCTTA uses.

Original: ``RDDM/src/residual_denoising_diffusion_pytorch.py::ResidualDiffusion``.
Kept: conditional residual diffusion, ``objective='pred_res'``, one U-Net, the
DDIM-converted linear schedule (``convert_to_ddim=True`` branch), L2 training loss,
deterministic DDIM sampling (``eta=0``, ``pred_type='use_pred_noise'``).
Removed: ``pred_res_noise`` / ``pred_x0_noise`` / ``pred_noise`` objectives, the
two-U-Net variant, the ancestral ``p_sample_loop`` sampler (never reached because
``sampling_timesteps < timesteps`` always selected DDIM), self-conditioning,
``input_condition`` masks, the ``gen_coefficients`` hand-made schedules, the
unreachable ``u_loss`` branch, and ``ResidualDiffusion.init()`` (it re-registered the
same schedule with ``alphas[0]=alphas[1]`` before every sampling call; none of the
entries it changed are read by DDIM sampling under ``pred_res``).

Notation (paper Eq. 2 / RDDM):  x_t = x_0 + alphas_cumsum[t] * x_res + betas_cumsum[t] * eps,
with x_res = x_cond - x_0. For DCTTA, x_0 is the *degraded* test image and x_cond is the
pseudo-label, so the network learns the residual that maps the pseudo-clean image back to
the degraded one; at sampling time it is conditioned on the degraded image itself.
"""
import math
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

ModelResPrediction = namedtuple('ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    return (img + 1) * 0.5


class ResidualDiffusion(nn.Module):
    def __init__(self, model, *, timesteps=1000, sampling_timesteps=5, sum_scale=0.01,
                 loss_type='l2', ddim_sampling_eta=0.):
        super().__init__()
        assert model.condition, "DCTTA's generator is always conditioned"
        self.model = model
        self.channels = model.channels
        self.sum_scale = sum_scale
        self.loss_type = loss_type
        self.ddim_sampling_eta = ddim_sampling_eta

        # "convert_to_ddim" linear schedule, beta in [1e-4, 0.02]
        betas = torch.linspace(0.0001, 0.02, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumsum = 1 - alphas_cumprod ** 0.5
        betas2_cumsum = 1 - alphas_cumprod

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        alphas = alphas_cumsum - alphas_cumsum_prev
        alphas[0] = 0
        betas2 = betas2_cumsum - betas2_cumsum_prev
        betas2[0] = 0
        betas_cumsum = torch.sqrt(betas2_cumsum)

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = sampling_timesteps
        assert self.sampling_timesteps <= timesteps

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer('alphas', alphas)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)

    # ------------------------------------------------------------------ #
    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        raise ValueError(f'invalid loss type {self.loss_type}')

    def _time_input(self, t):
        # RDDM feeds the *continuous* cumulative coefficient scaled by T as "time".
        return self.alphas_cumsum[t] * self.num_timesteps

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return ((x_t - x_input - (extract(self.alphas_cumsum, t, x_t.shape) - 1) * pred_res)
                / extract(self.betas_cumsum, t, x_t.shape))

    def model_predictions(self, x_input, x, t, clip_denoised=True):
        x_in = torch.cat((x, x_input), dim=1)
        pred_res = self.model(x_in, self._time_input(t))
        if clip_denoised:
            pred_res = torch.clamp(pred_res, min=-1., max=1.)
        pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
        x_start = x_input - pred_res
        if clip_denoised:
            x_start = torch.clamp(x_start, min=-1., max=1.)
        return ModelResPrediction(pred_res, pred_noise, x_start)

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def ddim_sample(self, x_input, return_all=False):
        """Deterministic DDIM (eta=0 by default) starting from ``x_input + sqrt(sum_scale) * eps``.

        ``x_input`` is expected in [-1, 1]. Returns the final image in [0, 1]
        (or the list of every intermediate state when ``return_all``).
        """
        batch, device = x_input.shape[0], x_input.device
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = x_input + math.sqrt(self.sum_scale) * torch.randn(x_input.shape, device=device)
        img_list = [unnormalize_to_zero_to_one(img.clone())] if return_all else None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            preds = self.model_predictions(x_input, img, time_cond)
            x_start, pred_res, pred_noise = preds.pred_x_start, preds.pred_res, preds.pred_noise

            if time_next < 0:
                img = x_start
                if return_all:
                    img_list.append(unnormalize_to_zero_to_one(img))
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum - alpha_cumsum_next

            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum - betas2_cumsum_next
            betas_cumsum = self.betas_cumsum[time]
            sigma2 = eta * (betas2 * betas2_cumsum_next / betas2_cumsum)
            noise = 0 if eta == 0 else torch.randn_like(img)

            img = img - alpha * pred_res - (betas_cumsum - (betas2_cumsum_next - sigma2).sqrt()) * pred_noise \
                + sigma2.sqrt() * noise

            if return_all:
                img_list.append(unnormalize_to_zero_to_one(img))

        if return_all:
            return img_list
        return unnormalize_to_zero_to_one(img)

    @torch.no_grad()
    def sample(self, x_input, return_all=False):
        """``x_input`` in [0, 1]; output in [0, 1]."""
        return self.ddim_sample(normalize_to_neg_one_to_one(x_input), return_all=return_all)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def q_sample(self, x_start, x_res, t, noise):
        return (x_start + extract(self.alphas_cumsum, t, x_start.shape) * x_res
                + extract(self.betas_cumsum, t, x_start.shape) * noise)

    def p_losses(self, x_start, x_input, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        x_res = x_input - x_start
        x = self.q_sample(x_start, x_res, t, noise=noise)
        x_in = torch.cat((x, x_input), dim=1)
        pred_res = self.model(x_in, self._time_input(t))
        loss = self.loss_fn(pred_res, x_res, reduction='none')
        # einops.reduce(loss, 'b ... -> b (...)', 'mean').mean() in the original
        return loss.reshape(loss.shape[0], -1).mean(dim=1).mean()

    def forward(self, x_start, x_input):
        """``x_start`` (the image to be reconstructed) and ``x_input`` (the condition), both in [0, 1].

        In DCTTA ``x_start`` is the degraded test image and ``x_input`` the pseudo-label.
        """
        b = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        return self.p_losses(normalize_to_neg_one_to_one(x_start), normalize_to_neg_one_to_one(x_input), t)
