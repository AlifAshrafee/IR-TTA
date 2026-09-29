"""DCTTA: degradation-consistent test-time adaptation (Tang et al., CVPR 2026).

This is the original ``tta.py::SRTTA`` class (the name betrays its origin: the class was
copied from SRTTA, Deng et al., NeurIPS 2023) reduced to the code path that is actually
executed for PromptIR (``airnet == 0``). The AirNet (tuple-output) and text-conditioned
(DFPIR-style ``text_code``) branches were byte-for-byte duplicates of this path with a
different model call signature; adding a backbone means wrapping it so that
``model(x) -> tensor``.

One adaptation step on a degraded patch ``x_in`` (batch size 1 in all reported runs):

    y_bar   = mean of the 8-way flip/rotation self-ensemble of the EMA *teacher*      (Eq. 1)
    x_sd    = G(x_in)  after 5 online RDDM steps on the pair (x_in, y_bar)              (Eq. 2)
    f_gt    = f_theta(x_in)          [no grad]
    f_sd    = f_theta(x_sd)          [grad]
    L_s     = L(f_sd, f_gt)                       self-supervised consistency          (Eqs. 4-6)
    L_t     = teacher_weight * L(f_sd, y_bar)     pseudo-label distillation             (Eqs. 7-9)
    theta  <- Adam step on L_s + L_t
    theta  <- TIPS restoration: entries selected by the Fisher mask are reset to theta_0
    teacher <- ema_decay * teacher + (1 - ema_decay) * theta

Facts about the released implementation that the paper does not state (kept as-is;
all of them are exposed as arguments so they can be ablated):

* Gradients flow **only** through the redegraded branch f_theta(x_sd); f_theta(x_in) is
  detached. Both losses pull f_theta(x_sd) -- towards f_theta(x_in) and towards y_bar.
* ``teacher_weight`` (alpha in the paper, stated as 1) defaults to 5 in ``options.py``.
* ``ema_decay`` (eta in the paper, stated as 0.95) is hard-coded to 0.45.
* Only parameters whose top-level module name contains 'prompt', 'encoder' or 'decoder' get
  ``requires_grad=True`` (so: prompt{1,2,3}, encoder_level{1,2,3}, decoder_level{1,2,3}); latent,
  refinement, noise_level*, all 1x1 reduce convs, up/down-samplers, patch_embed and the output
  conv are frozen outright. Of the trainable ones, only tensors whose leaf name is 'weight' or
  'bias' are handed to the optimiser, so ``prompt_param`` (the prompt bank) and the attention
  ``temperature`` are never updated either.
* TIPS (``compute_fisher``): diagonal Fisher of the self-ensemble consistency loss, accumulated
  over the whole target set, then a **per-tensor** top-``fisher_ratio`` (=0.6) mask. Masked
  entries are restored to theta_0 after every optimiser step, i.e. 60% of every trainable tensor
  is frozen and 40% adapts (the paper describes it as a global 40% freeze).
* Evaluation uses the adapted *student*, not the EMA teacher.
"""
import logging
from copy import deepcopy

import torch
import torch.optim as optim
from tqdm import tqdm

from .losses import DCTTALoss, charbonnier
from .utils import augment_transform, self_ensemble, transform_tensor

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_PREFIXES = ('prompt', 'encoder', 'decoder')


def configure_model(model, train_prefixes=DEFAULT_TRAIN_PREFIXES):
    """Set ``requires_grad`` by top-level module name (substring match, as in the original)."""
    model.train()
    n_train = n_total = 0
    train_all = train_prefixes == 'all' or 'all' in train_prefixes
    for name, p in model.named_parameters():
        prefix = name.split('.')[0]
        p.requires_grad = train_all or any(tp in prefix for tp in train_prefixes)
        n_total += p.numel()
        n_train += p.numel() if p.requires_grad else 0
    logger.info(f"trainable parameters: {n_train / 1e6:.2f}M / {n_total / 1e6:.2f}M "
                f"(prefixes {train_prefixes})")
    return model


def collect_params(model, leaf_names=('weight', 'bias')):
    """Trainable tensors whose leaf name is in ``leaf_names`` (the original optimiser set)."""
    params, names = [], []
    for name, p in model.named_parameters():
        if p.requires_grad and name.split('.')[-1] in leaf_names:
            params.append(p)
            names.append(name)
    return params, names


class DCTTA:
    def __init__(self, model, optimizer, *, teacher_weight=5.0, ema_decay=0.45, fisher_ratio=0.6,
                 use_fisher=True, pixel_weight=1.0, perceptual_weight=0.01, lowpass_weight=0.1,
                 device='cuda', restore_leaf_names=('weight', 'bias')):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.teacher_weight = teacher_weight
        self.ema_decay = ema_decay
        self.fisher_ratio = fisher_ratio
        self.use_fisher = use_fisher
        self.restore_leaf_names = restore_leaf_names

        self.criterion = DCTTALoss(pixel_weight, perceptual_weight, lowpass_weight).to(self.device)

        # theta_0 (for TIPS restoration and reset) and the EMA teacher
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())
        self.fishers = {}
        self.model_teacher = deepcopy(model)
        for p in self.model_teacher.parameters():
            p.detach_()

    # ------------------------------------------------------------------ #
    # pseudo-label and one adaptation step
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def pseudo_label(self, x_in):
        return self_ensemble(self.model_teacher, x_in)

    def adapt_step(self, x_in, generator, name=None):
        self.model.train()
        self.optimizer.zero_grad()

        y_bar = self.pseudo_label(x_in)
        x_sd = generator.fit_and_sample(x_in, y_bar, name)

        with torch.no_grad():
            out_student_gt = self.model(x_in)
        out_student = self.model(x_sd)

        loss_s = self.criterion(out_student, out_student_gt)
        loss_t = self.teacher_weight * self.criterion(out_student, y_bar) if self.teacher_weight > 0 else 0.
        loss = loss_s + loss_t
        logger.info(f"loss_s: {float(loss_s):.5f}, loss_t: {float(loss_t):.5f}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.use_fisher:
            self.fisher_restoration()
        return float(loss)

    @torch.no_grad()
    def update_teacher(self):
        if self.ema_decay <= 0:
            return
        teacher_params = dict(self.model_teacher.named_parameters())
        student_params = dict(self.model.named_parameters())
        for k in teacher_params.keys():
            teacher_params[k].data.mul_(self.ema_decay).add_(student_params[k], alpha=1 - self.ema_decay)

    def adapt(self, x_in, generator, name=None, iterations=1):
        """The original ``SRTTA.__call__``: ``iterations`` (=1) steps on one patch, then teacher EMA."""
        x_in = x_in.to(self.device)
        loss = None
        for _ in range(iterations):
            loss = self.adapt_step(x_in, generator, name)
            self.update_teacher()
        return loss

    __call__ = adapt

    # ------------------------------------------------------------------ #
    # TIPS: Fisher mask + restoration
    # ------------------------------------------------------------------ #
    def fisher_restoration(self):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if not p.requires_grad or name.split('.')[-1] not in self.restore_leaf_names:
                    continue
                if name not in self.fishers:
                    continue
                mask = self.fishers[name][1]
                p.data.copy_(self.model_state[name] * mask + p * (1. - mask))

    def compute_fisher(self, loader):
        """Diagonal Fisher of the self-ensemble consistency loss on the *target* set (Eqs. 12-14).

        Returns ``{name: [fisher, mask]}``; ``mask==1`` marks the entries that are frozen (restored
        to theta_0 after every step). Uses the student, which at this point equals the teacher.
        """
        if len(self.fishers) > 0:
            return self.fishers

        fishers = {}
        self.model.zero_grad(set_to_none=True)
        for (_, img_lr, _) in tqdm(loader, total=len(loader), desc="TIPS Fisher"):
            img_lr = img_lr.to(self.device)
            self.model.zero_grad(set_to_none=True)

            tran_imgs, tran_ops = augment_transform(img_lr)
            tran_imgs.reverse()  # the identity transform comes last
            tran_ops.reverse()

            sr_imgs = []
            for idx, (tran_img, op) in enumerate(zip(tran_imgs, tran_ops), start=1):
                if idx < len(tran_imgs):
                    with torch.no_grad():
                        sr_img = transform_tensor(self.model(tran_img), op, undo=True)
                else:
                    sr_img = self.model(tran_img)  # identity, with grad
                sr_imgs.append(sr_img)
            sr_pseudo = torch.cat(sr_imgs, dim=0).mean(dim=0, keepdim=True).detach()
            loss = charbonnier(sr_imgs[-1], sr_pseudo)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher = param.grad.data.clone().detach() ** 2
                    fishers[name] = fishers[name] + fisher if name in fishers else fisher

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                fisher = fishers[name].flatten()
                _, mask_idx = torch.topk(fisher, k=int(len(fisher) * self.fisher_ratio))
                mask = param.new_zeros(param.shape).flatten()
                mask[mask_idx] = 1
                self.fishers[name] = [fisher, mask.view(param.shape)]
        self.model.zero_grad(set_to_none=True)

        for name, (fisher, mask) in self.fishers.items():
            logger.info(f"Fisher {name}: mean {fisher.mean().item():.3e}, frozen {int(mask.sum())}/{mask.numel()}")
        return self.fishers

    # ------------------------------------------------------------------ #
    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def save(self, path, extra=None):
        state = {'state_dict': self.model.state_dict(), 'teacher_state_dict': self.model_teacher.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 'ori_model': self.model_state}
        if extra:
            state.update(extra)
        torch.save(state, path)
        logger.info(f"model saved to {path}")


def setup_dctta(model, *, lr=2e-4, betas=(0.9, 0.999), train_prefixes=DEFAULT_TRAIN_PREFIXES, **kwargs):
    """``train_test_promptir.py::setup_tta``: configure trainable params, build Adam, wrap."""
    model = configure_model(model, train_prefixes)
    params, _ = collect_params(model)
    optimizer = optim.Adam(params, lr=lr, betas=betas)
    return DCTTA(model, optimizer, **kwargs)
