"""TTA methods behind one interface: ``step(x)`` adapts, ``eval_model`` predicts, ``reset()`` restores theta_0.

* ``identity`` -- the unprocessed input (the "do no harm" bar).
* ``source`` -- the un-adapted source model (B0).
* ``transfer`` -- transfer TTA: supervised L1 on synthetic pairs ``(op(c), c)`` from a clean content bank,
  ``op`` a chain of physical primitives (``irtta/degrade.py``), brightness matched to a target patch
  (parked: no gain over the source so far; kept as a building block).
* ``flow`` -- evaluation only: the source's own restoration direction, extrapolated / iterated
  (``y <- y + ext_alpha * (f(y) - y)``, ``n_apply`` times): does following the source further help?
* ``dctta`` -- the released DCTTA (``dctta/tta.py``) driven through the harness (baseline B4).
"""
import logging
from copy import deepcopy

import torch
import torch.optim as optim

from dctta.losses import DCTTALoss
from dctta.rddm.generator import ResidualDegradationGenerator
from dctta.tta import DCTTA

from .data import CleanBank
from .degrade import degrade_batch, parse_families

logger = logging.getLogger(__name__)

METHOD_DEFAULTS = {
    #               loss weights (pixel, perceptual, lowpass), teacher EMA decay, anchor
    'dctta':     dict(pixel_weight=1.0, perceptual_weight=0.01, lowpass_weight=0.1, ema_decay=0.45, anchor='tips'),
    'transfer':  dict(pixel_weight=1.0, perceptual_weight=0.0, lowpass_weight=0.0, ema_decay=0.999, anchor='none'),
    'source':    dict(anchor='none'),
    'identity':  dict(anchor='none'),
    'flow':      dict(anchor='none'),
}


def rng_checksum():
    """The regression guard of the DCTTA reference (319577 for --dataset_seed 42, any target)."""
    return int(torch.random.get_rng_state().double().sum())


class Method:
    name = 'base'
    adapts = True

    def __init__(self, model, subset, anchor, args, device):
        self.model, self.subset, self.anchor, self.args = model, subset, anchor, args
        self.device = torch.device(device)
        self.last_stats = {}
        self.steps = 0
        self.skipped = 0

    def prepare(self, loader=None):
        self.anchor.prepare(loader)

    def step(self, x, name=None):
        raise NotImplementedError

    @property
    def eval_model(self):
        return self.model

    def reset(self):
        pass

    @property
    def n_params_adapted(self):
        return self.subset.n_params if self.adapts else 0

    @property
    def n_params_effective(self):
        """Adapted entries that can actually move (DCTTA's TIPS freezes part of each tensor)."""
        return self.n_params_adapted


class SourceMethod(Method):
    name = 'source'
    adapts = False

    def __init__(self, model, subset, anchor, args, device):
        super().__init__(model, subset, anchor, args, device)
        self.model.eval()

    def prepare(self, loader=None):
        pass

    def step(self, x, name=None):
        return None


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


class IdentityMethod(SourceMethod):
    """Output = input: PSNR of the degraded image itself."""
    name = 'identity'

    def __init__(self, model, subset, anchor, args, device):
        super().__init__(model, subset, anchor, args, device)
        self._identity = _Identity()

    @property
    def eval_model(self):
        return self._identity


class _Flow(torch.nn.Module):
    def __init__(self, model, ext_alpha, n_apply):
        super().__init__()
        self.model, self.ext_alpha, self.n_apply = model, ext_alpha, n_apply

    def forward(self, x):
        y = x
        for _ in range(self.n_apply):
            y = y + self.ext_alpha * (self.model(y) - y)
        return y


class FlowMethod(SourceMethod):
    """Evaluation only: follow the source's restoration direction further (extrapolate and/or iterate)."""
    name = 'flow'

    def __init__(self, model, subset, anchor, args, device):
        super().__init__(model, subset, anchor, args, device)
        self._flow = _Flow(model, args.ext_alpha, args.n_apply)

    @property
    def eval_model(self):
        return self._flow


class _OptimMethod(Method):
    def _build_optimizer(self):
        if not self.subset.params:
            raise ValueError(f"method '{self.name}' needs a non-empty --subset")
        self.optimizer = optim.Adam(self.subset.params, lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))

    def _snapshot(self):
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def _restore(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)


class TransferMethod(_OptimMethod):
    """Transfer TTA v0.1: L1(f(op(c)), c') on clean bank crops c, op a physical-order chain whose brightness is
    matched to the current target patch x (``--exposure_match fit|clean``)."""
    name = 'transfer'

    def __init__(self, model, subset, anchor, args, device):
        super().__init__(model, subset, anchor, args, device)
        self._build_optimizer()
        with torch.random.fork_rng(devices=[]):
            self.criterion = DCTTALoss(args.pixel_weight, args.perceptual_weight, args.lowpass_weight).to(self.device)
        self.families = parse_families(args.families, getattr(args, 'target_task', None))
        self.bank = CleanBank(args.bank_path, max_images=args.bank_max, min_side=args.bank_patch)
        logger.info(f"transfer: families {self.families}, bank {args.bank_path} ({len(self.bank)} images), "
                    f"{args.bank_batch} x {args.bank_patch}px pairs per step, exposure match '{args.exposure_match}'")
        self.gen = torch.Generator().manual_seed(int(args.seed) + 3)
        self.fam_counts = {}
        self._snapshot()

    def step(self, x, name=None):
        a = self.args
        self.model.train()
        loss_val = None
        for _ in range(a.iterations):
            with torch.no_grad():
                c = self.bank.sample(a.bank_batch, a.bank_patch, self.gen).to(self.device)
                tmean = x.mean(dim=(0, 2, 3), keepdim=True) if a.exposure_match != 'none' else None
                y, c, fams = degrade_batch(c, self.families, self.gen, tmean, a.exposure_match)
            for f in fams:
                self.fam_counts[f] = self.fam_counts.get(f, 0) + 1
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(y)
            sup = self.criterion(out, c)
            loss = sup
            pen = self.anchor.penalty()
            if pen is not None:
                loss = loss + pen
            loss.backward()
            self.optimizer.step()
            self.anchor.post_step()
            self.steps += 1
            loss_val = float(loss)
            self.last_stats = {'sup': float(sup),
                               'penalty': float(pen) if pen is not None else 0.0,
                               'x_in': float(x.mean()), 'pair_in': float(y.mean()), 'pair_gt': float(c.mean()),
                               'f_pair': float(out.detach().mean()), 'fams': '+'.join(fams)}
        return loss_val

    def reset(self):
        self._restore()


class DCTTAMethod(_OptimMethod):
    """The released DCTTA, with the optimiser/subset/anchor supplied by the harness.

    Construction order (optimiser -> DCTTA [VGG under fork_rng] -> RNG checksum -> generator) is the
    order of ``adapt.py``; with ``--subset full --anchor tips`` and the default loss/teacher settings the
    run is numerically the released DCTTA (``tests/test_harness.py`` checks this against ``adapt.py``).
    """
    name = 'dctta'

    def __init__(self, model, subset, anchor, args, device):
        super().__init__(model, subset, anchor, args, device)
        self._build_optimizer()
        penalty_fn = anchor.penalty if anchor.kind == 'l2' else None
        self.tta = DCTTA(model, self.optimizer, teacher_weight=args.teacher_weight, ema_decay=args.ema_decay,
                         fisher_ratio=args.tips_ratio, use_fisher=(anchor.kind == 'tips'),
                         pixel_weight=args.pixel_weight, perceptual_weight=args.perceptual_weight,
                         lowpass_weight=args.lowpass_weight, device=device, log_stats=args.log_stats,
                         penalty_fn=penalty_fn)
        self.checksum = rng_checksum()
        logger.info(f"CPU RNG checksum before generator init: {self.checksum}")
        self.generator = ResidualDegradationGenerator(device, train_steps=args.gen_train_steps,
                                                      sampling_timesteps=args.gen_sampling_steps, lr=args.gen_lr,
                                                      sum_scale=args.gen_sum_scale, debug_dir=args.gen_debug_dir)

    def prepare(self, loader=None):
        if self.anchor.kind == 'tips':
            if loader is None:
                raise ValueError("DCTTA's TIPS needs the target patch loader; use --anchor none/l2 "
                                 "in the online/episodic regimes")
            self.tta.compute_fisher(loader)
            if getattr(self.args, 'save_fisher', None):
                torch.save({k: [f.cpu(), m.cpu()] for k, (f, m) in self.tta.fishers.items()}, self.args.save_fisher)

    def step(self, x, name=None):
        loss = self.tta(x, self.generator, name=name, iterations=self.args.iterations)
        self.steps += self.args.iterations
        self.last_stats = dict(self.tta.last_stats)
        return loss

    def reset(self):
        """theta_0, optimiser state and teacher; the online generator keeps learning across images."""
        self.tta.reset()
        self.tta.model_teacher.load_state_dict(self.tta.model_state, strict=True)

    @property
    def n_params_effective(self):
        if self.anchor.kind != 'tips' or not self.tta.fishers:
            return self.n_params_adapted
        tot = 0
        for n, p in zip(self.subset.names, self.subset.params):
            if n in self.tta.fishers and n.split('.')[-1] in self.tta.restore_leaf_names:
                tot += int(p.numel() - self.tta.fishers[n][1].sum())
            else:
                tot += p.numel()
        return tot


def build_method(name, model, subset, anchor, args, device):
    classes = {'identity': IdentityMethod, 'source': SourceMethod, 'flow': FlowMethod,
               'transfer': TransferMethod, 'dctta': DCTTAMethod}
    if name not in classes:
        raise ValueError(f"unknown method '{name}'")
    return classes[name](model, subset, anchor, args, device)
