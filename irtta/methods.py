"""Methods behind one interface: ``prepare(loader)``, ``step(x)`` adapts on a target patch, ``eval_model`` predicts.

* ``identity`` -- the unprocessed input.
* ``source``   -- the pretrained all-in-one model, not adapted.
* ``transfer`` -- ours: test-time adaptation on synthetic pairs whose degradations are inferred from the target set
  (``TransferMethod``).
* ``dctta``    -- the released DCTTA (``irtta/baselines/dctta``), the existing test-time adaptation method for
  all-in-one restoration (baseline).
"""
import logging
import os
from copy import deepcopy
from functools import partial

import torch
import torch.nn.functional as F
import torch.optim as optim

from .baselines.dctta.rddm.generator import ResidualDegradationGenerator
from .baselines.dctta.tta import DCTTA
from .data import CleanBank, PairBank, available_names, evenly_spaced, load_registry, resolve_spec
from .degrade import degrade_batch, parse_families
from .nss import GaussianModel, nss_features, random_crop_batch

logger = logging.getLogger(__name__)


def rng_checksum():
    """Sum of the global CPU RNG state: a cheap reproducibility fingerprint logged by every run."""
    return int(torch.random.get_rng_state().double().sum())


class Method:
    name = 'base'
    adapts = True

    def __init__(self, model, subset, args, device):
        self.model, self.subset, self.args = model, subset, args
        self.device = torch.device(device)
        self.last_stats = {}
        self.steps = 0
        self.skipped = 0
        self._updated = None   # indices into subset.params that received a gradient (track_updates)

    def prepare(self, loader=None):
        pass

    def track_updates(self):
        """Record which adapted tensors receive a gradient during adaptation. A tensor outside the forward pass
        (e.g. PromptIR's chnl_reduce*, R2R's mem_encoder) never does, so it never moves; ``n_params_updated``
        counts only the others. Gradient hooks do not change any computation or RNG draw."""
        self._updated = set()
        for i, p in enumerate(self.subset.params):
            p.register_hook(lambda g, i=i: self._updated.add(i))

    @property
    def n_params_updated(self):
        """Adapted parameters that received a gradient (None if not tracked)."""
        if self._updated is None:
            return None
        return int(sum(self.subset.params[i].numel() for i in self._updated))

    def step(self, x, name=None):
        raise NotImplementedError

    @property
    def eval_model(self):
        return self.model

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

    def __init__(self, model, subset, args, device):
        super().__init__(model, subset, args, device)
        self.model.eval()

    def step(self, x, name=None):
        return None


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


class IdentityMethod(SourceMethod):
    """Output = input: the PSNR of the degraded image itself."""
    name = 'identity'

    def __init__(self, model, subset, args, device):
        super().__init__(model, subset, args, device)
        self._identity = _Identity()

    @property
    def eval_model(self):
        return self._identity


class _OptimMethod(Method):
    def _build_optimizer(self):
        if not self.subset.params:
            raise ValueError(f"method '{self.name}' needs a non-empty --subset")
        self.optimizer = optim.Adam(self.subset.params, lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))


class TransferMethod(_OptimMethod):
    """Ours: adapt the pretrained model on synthetic pairs ``(op(c), c)`` whose degradation ``op`` is inferred from
    the unlabelled target set.

    prepare  fit N(mu_T, Sigma_T) to the NSS features of ``--abc_fit_n`` target crops (``irtta/nss.py``)
    step     for the current target patch x:
             * under-exposure rule (``--dark_rule source``): the frozen pretrained model f_0 brightens x by at least
               ``--dark_ratio`` -> x is under-exposed; the clean images then take the brightness of f_0(x)
               (``--dark_action anchor``), since exposure is not identifiable from x;
             * ``--bank_batch`` clean crops c; for each, ``--abc_k`` candidate degradations from the source's
               families with camera-consistent exposure to x (``irtta/degrade.py``); keep the candidate closest
               to N_T (ABC);
             * one Adam step on L1( f(op(c)), c' ).
    The adapted model is evaluated on every target image after ``--max_steps`` steps.

    ``--oracle_pairs <set>`` (analysis only): train on real pairs of that set (the exact target operator, other
    images) instead of synthesising them.
    """
    name = 'transfer'

    def __init__(self, model, subset, args, device):
        super().__init__(model, subset, args, device)
        self._build_optimizer()
        self.families = parse_families(args.families, getattr(args, 'target_task', None))
        self.bank = CleanBank(args.bank_path, max_images=args.bank_max, min_side=args.bank_patch)
        logger.info(f"transfer: families {self.families}, bank {args.bank_path} ({len(self.bank)} images), "
                    f"{args.bank_batch} x {args.bank_patch}px pairs per step, exposure match '{args.exposure_match}'")
        self.oracle = None
        if args.oracle_pairs:
            reg = load_registry(args.registry)
            ospec = resolve_spec(args.oracle_pairs, reg)
            exclude = ()
            if args.target:   # same folder as the evaluated target images: never train on those
                tspec = resolve_spec(args.target, reg)
                if tspec.lq is not None and os.path.normpath(tspec.lq) == os.path.normpath(ospec.lq or ''):
                    tn = available_names(tspec)
                    exclude = evenly_spaced(tn, args.max_images) if args.max_images else tn
            self.oracle = PairBank(ospec, max_images=args.bank_max, min_side=args.bank_patch, exclude=exclude)
            logger.info(f"ORACLE: training on {len(self.oracle)} real pairs of {args.oracle_pairs} (no synthesis)"
                        + (f", {len(exclude)} evaluated target images excluded" if exclude else ''))
        self.gen = torch.Generator().manual_seed(int(args.seed) + 3)     # pair draws
        self.gen_t = torch.Generator().manual_seed(int(args.seed) + 5)   # target-side draws (crops)
        self.fam_counts = {}
        self._feat = partial(nss_features, color=args.nss_color)
        self.teacher = None
        if args.dark_rule == 'source' and self.oracle is None:   # frozen theta_0 (LoRA zero-init)
            self.teacher = deepcopy(model).eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        self.target_model = None   # N(mu_T, Sigma_T) of the target NSS features (ABC)

    @torch.no_grad()
    def prepare(self, loader=None):
        a = self.args
        if a.abc_k <= 1 or self.oracle is not None:
            return
        if loader is None:
            raise ValueError("--abc_k needs the target loader")
        feats, n = [], 0
        with torch.random.fork_rng(devices=[]):   # iterating a DataLoader draws from the global RNG
            for _, x, _ in loader:
                crops = random_crop_batch(x, a.bank_patch, 2, self.gen_t).to(self.device)
                feats.append(self._feat(crops))
                n += crops.shape[0]
                if n >= a.abc_fit_n:
                    break
        tf = torch.cat(feats, 0)
        self.target_model = GaussianModel(tf).to(self.device)
        own = self.target_model.distance(tf)
        logger.info(f"ABC: target NSS Gaussian from {n} crops of {a.bank_patch}px; target crops' own distance "
                    f"median {float(own.median()):.2f}, q90 {float(torch.quantile(own, 0.9)):.2f}")

    def _select(self, cand):
        return self.target_model.distance(self._feat(cand))

    def _pairs(self, x, st):
        """Synthetic (degraded, clean) pairs for target patch ``x``, or None to skip the step."""
        a = self.args
        c = self.bank.sample(a.bank_batch, a.bank_patch, self.gen).to(self.device)
        tmean = x.mean(dim=(0, 2, 3), keepdim=True) if a.exposure_match != 'none' else None
        dark, cmean = False, None
        if a.dark_rule == 'source' and tmean is not None:
            f0x = self.teacher(x)
            ratio = float(f0x.clamp(0, 1).mean()) / max(float(x.mean()), 1e-3)
            dark = ratio >= a.dark_ratio
            st['bright_ratio'], st['dark'] = ratio, int(dark)
            if dark and a.dark_action == 'skip':
                return None
            if dark:   # anchor: the clean images take the exposure the pretrained model gives this patch
                cmean = f0x.clamp(0, 1).mean(dim=(0, 2, 3), keepdim=True)
        select = self._select if self.target_model is not None else None
        y, c, fams, scores = degrade_batch(c, self.families, self.gen, tmean, select=select, k=a.abc_k,
                                           clean_mean=cmean, dark=dark)
        if scores:
            st['abc_d'] = sum(scores) / len(scores)
        return y, c, fams

    def step(self, x, name=None):
        a = self.args
        self.model.train()
        loss_val = None
        for _ in range(a.iterations):
            st = {}
            with torch.no_grad():
                if self.oracle is not None:   # ORACLE: real target-operator pairs, nothing synthesised
                    y, c = (t.to(self.device) for t in self.oracle.sample(a.bank_batch, a.bank_patch, self.gen))
                    fams = ['oracle'] * y.shape[0]
                else:
                    pairs = self._pairs(x, st)
                    if pairs is None:
                        self.skipped += 1
                        self.last_stats = {**st, 'skipped': 1}
                        continue
                    y, c, fams = pairs
            for f in fams:
                self.fam_counts[f] = self.fam_counts.get(f, 0) + 1
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(y)
            loss = F.l1_loss(out, c)
            loss.backward()
            self.optimizer.step()
            self.steps += 1
            loss_val = float(loss)
            self.last_stats = {'loss': loss_val, 'fams': '+'.join(fams), **st}
        return loss_val


class DCTTAMethod(_OptimMethod):
    """The released DCTTA (EMA teacher, self-ensemble pseudo-labels, RDDM re-degradation, TIPS), on any backbone:
    ``--subset full`` applies its parameter rule per backbone (``irtta.backbones.TRAIN_PREFIXES``). Construction
    order (optimiser -> DCTTA [VGG under fork_rng] -> generator) is the released one, so for PromptIR the run is
    numerically the released code."""
    name = 'dctta'

    def __init__(self, model, subset, args, device):
        super().__init__(model, subset, args, device)
        self._build_optimizer()
        self.tta = DCTTA(model, self.optimizer, teacher_weight=args.teacher_weight, ema_decay=args.ema_decay,
                         fisher_ratio=args.tips_ratio, use_fisher=True,
                         pixel_weight=args.pixel_weight, perceptual_weight=args.perceptual_weight,
                         lowpass_weight=args.lowpass_weight, device=device)
        self.checksum = rng_checksum()
        logger.info(f"CPU RNG checksum before generator init: {self.checksum}")
        self.generator = ResidualDegradationGenerator(device, train_steps=args.gen_train_steps,
                                                      sampling_timesteps=args.gen_sampling_steps, lr=args.gen_lr,
                                                      sum_scale=args.gen_sum_scale)

    def prepare(self, loader=None):
        if loader is None:
            raise ValueError("DCTTA's TIPS needs the target patch loader")
        self.tta.compute_fisher(loader)

    def step(self, x, name=None):
        loss = self.tta(x, self.generator, name=name, iterations=self.args.iterations)
        self.steps += self.args.iterations
        return loss

    @property
    def n_params_effective(self):
        if not self.tta.fishers:
            return self.n_params_adapted
        tot = 0
        for n, p in zip(self.subset.names, self.subset.params):
            if n in self.tta.fishers and n.split('.')[-1] in self.tta.restore_leaf_names:
                tot += int(p.numel() - self.tta.fishers[n][1].sum())
            else:
                tot += p.numel()
        return tot


METHODS = {'identity': IdentityMethod, 'source': SourceMethod, 'transfer': TransferMethod, 'dctta': DCTTAMethod}


def build_method(name, model, subset, args, device):
    if name not in METHODS:
        raise ValueError(f"unknown method '{name}'")
    return METHODS[name](model, subset, args, device)
