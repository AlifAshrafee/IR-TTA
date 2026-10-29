"""TTA methods behind one interface: ``step(x)`` adapts, ``eval_model`` predicts, ``reset()`` restores theta_0.

* ``identity`` -- the unprocessed input (the "do no harm" bar).
* ``source`` -- the un-adapted source model (B0).
* ``transfer`` -- transfer TTA: supervised L1 on synthetic pairs ``(op(c), c)`` from a clean content bank,
  ``op`` a chain of physical primitives (``irtta/degrade.py``), brightness matched to a target patch; optional
  ABC pair selection, target consistency, re-degradation consistency and an NSS output prior (class docstring).
* ``flow`` -- evaluation only: the source's own restoration direction, extrapolated / iterated
  (``y <- y + ext_alpha * (f(y) - y)``, ``n_apply`` times): does following the source further help?
* ``dctta`` -- the released DCTTA (``dctta/tta.py``) driven through the harness (baseline B4).
"""
import logging
from copy import deepcopy
from functools import partial

import torch
import torch.nn.functional as F
import torch.optim as optim

from dctta.losses import DCTTALoss
from dctta.rddm.generator import ResidualDegradationGenerator
from dctta.tta import DCTTA

from .data import CleanBank, PairBank, load_registry, resolve_spec
from .degrade import apply_chain, chain_label, degrade_batch, draw_chain, parse_families
from .nss import GaussianModel, downscale_crops, nss_features, random_crop_batch

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
    """Transfer TTA: supervised pairs from clean content + optional target-side and distribution-level terms.

    Loss per step (target patch x, clean bank crops c, frozen source f_0, adapted f):

        L =   L1( f(op(c)), c' )                                   pairs (always); op from the prior, or
                                                                   ABC-selected to look like the target (--abc_k)
            + cons_weight  * L1( B f(x), B sg[f_0(x)] )             target consistency; B = identity or a
                                                                   high-pass (--cons_band high)
            + redeg_weight * L1( f(D(x)), sg[f_0(x')] )            re-degradation consistency (SRTTA-style): the
                                                                   target with an extra degradation D restores to
                                                                   the source's output of the target
            + nss_weight   * d_C( NSS(f(x)) )                      output prior: Mahalanobis^2/D of the restored
                                                                   target's NSS to the clean bank's Gaussian
            + anchor                                               --anchor l2: 0.5 lam ||theta - theta_0||^2

    ``sg`` = stop-gradient. Every extra term is off by default, which is transfer v0.1/v0.2.

    Under-exposed target patches (``--dark_rule threshold|source``) get ``--dark_action``: ``skip`` (no update),
    ``anchor`` (clean images take the source's output brightness) or ``norm`` (clean images keep their natural
    exposure and the pair loss ignores overall brightness: L1(f(y) * mean(c) / mean(f(y)), c), so the pairs teach
    noise / colour / detail and the brightening level stays where the source puts it) or ``natural`` (clean images
    keep their natural exposure, plain L1: the pairs teach brightening -- for a source that never saw low-light).
    """
    name = 'transfer'

    def __init__(self, model, subset, anchor, args, device):
        super().__init__(model, subset, anchor, args, device)
        self._build_optimizer()
        with torch.random.fork_rng(devices=[]):
            self.criterion = DCTTALoss(args.pixel_weight, args.perceptual_weight, args.lowpass_weight).to(self.device)
        self.families = parse_families(args.families, getattr(args, 'target_task', None))
        self.bank = CleanBank(args.bank_path, max_images=args.bank_max, min_side=args.bank_patch,
                              long_side=getattr(args, 'target_long', None))
        if self.bank.long_side:
            logger.info(f"clean bank resized to the target's typical long side ({self.bank.long_side}px)")
        logger.info(f"transfer: families {self.families}, bank {args.bank_path} ({len(self.bank)} images), "
                    f"{args.bank_batch} x {args.bank_patch}px pairs per step, exposure match '{args.exposure_match}'")
        self.oracle = None
        if getattr(args, 'oracle_pairs', None):
            if args.cons_weight > 0 or args.redeg_weight > 0:
                raise ValueError("--oracle_pairs replaces the pair term only; use it without --cons/--redeg_weight")
            self.oracle = PairBank(resolve_spec(args.oracle_pairs, load_registry(args.registry)), max_images=args.bank_max, min_side=args.bank_patch)
            logger.info(f"ORACLE: training on {len(self.oracle)} real pairs of {args.oracle_pairs} (no synthesis)")
        self.gen = torch.Generator().manual_seed(int(args.seed) + 3)
        self.gen_t = torch.Generator().manual_seed(int(args.seed) + 5)   # target-side draws (crops, D)
        self.fam_counts = {}
        self.need_teacher = (args.cons_weight > 0 or args.redeg_weight > 0 or args.dark_anchor
                             or args.dark_rule == 'source' or (args.dark_rule != 'none' and args.dark_action == 'anchor'))
        self._feat = partial(nss_features, color=args.nss_color)
        self.teacher = None
        if self.need_teacher:   # frozen theta_0 (LoRA zero-init: the source function)
            self.teacher = deepcopy(model).eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        self.floor = None
        if args.dark_rule == 'threshold' and args.dark_q <= 0:
            raise ValueError("--dark_rule threshold needs --dark_q > 0")
        if args.dark_q > 0:   # under-exposure threshold: a low quantile of clean-bank image brightness
            means = torch.tensor([float(img.mean()) / 255. for img in self.bank.images])
            self.floor = float(torch.quantile(means, args.dark_q))
            logger.info(f"under-exposure threshold: target patches darker than {self.floor:.3f} "
                        f"(q={args.dark_q} of {len(means)} clean-bank image means) keep natural-exposure clean images")
        self.target_model = None   # N(mu_T, Sigma_T) of target NSS (ABC)
        self.abc_eps = None        # ABC tolerance (--abc_eps_q)
        self.clean_model = None    # N(mu_C, Sigma_C) of clean NSS (output prior)
        self._snapshot()

    # ---- distribution models (prepare) ----
    @torch.no_grad()
    def prepare(self, loader=None):
        super().prepare(loader)
        a = self.args
        if a.abc_k > 1:
            if loader is None:
                raise ValueError("--abc_k needs the transductive target loader")
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
            if a.abc_eps_q > 0:   # ABC tolerance: as far from the target as its own q-quantile crop
                self.abc_eps = float(torch.quantile(own, a.abc_eps_q))
                logger.info(f"ABC tolerance eps = {self.abc_eps:.2f} (q={a.abc_eps_q} of the target crops' distances)")
        if a.nss_weight > 0:
            feats = []
            for i in range(0, a.abc_fit_n, 16):
                c = self.bank.sample(16, a.bank_patch * 2, self.gen_t)
                feats.append(self._feat(c[:8, :, :a.bank_patch, :a.bank_patch].to(self.device)))
                feats.append(self._feat(downscale_crops(c[8:], 2).to(self.device)))
            self.clean_model = GaussianModel(torch.cat(feats, 0)).to(self.device)
            if self.target_model is not None:
                logger.info(f"NSS: KL(target || clean) = {self.clean_model.kl_from(self.target_model):.2f}")

    def _select(self, cand):
        return self.target_model.distance(self._feat(cand))

    def _band(self, t):
        if self.args.cons_band == 'high':
            return t - _gauss_blur(t, 2.0)
        return t

    def step(self, x, name=None):
        a = self.args
        self.model.train()
        loss_val = None
        for _ in range(a.iterations):
            st = {}
            with torch.no_grad():
                if self.oracle is not None:   # ORACLE: real target-operator pairs, nothing synthesised
                    y, c = (t.to(self.device) for t in self.oracle.sample(a.bank_batch, a.bank_patch, self.gen))
                    fams, norm, scores = ['oracle'] * y.shape[0], False, []
                else:
                    c = self.bank.sample(a.bank_batch, a.bank_patch, self.gen).to(self.device)
                    tmean = x.mean(dim=(0, 2, 3), keepdim=True) if a.exposure_match != 'none' else None
                    dark, norm, f0x = None, False, None
                    if a.dark_rule != 'none' and tmean is not None:   # new path: decide, then act
                        if a.dark_rule == 'source':
                            f0x = self.teacher(x)
                            ratio = float(f0x.clamp(0, 1).mean()) / max(float(x.mean()), 1e-3)
                            st['bright_ratio'] = ratio
                            dark = ratio >= a.dark_ratio
                        else:
                            dark = float(tmean.mean()) < self.floor
                        st['dark'] = int(dark)
                        if dark and a.dark_action == 'skip':
                            self.skipped += 1
                            self.last_stats = {**st, 'skipped': 1, 'x_in': float(x.mean())}
                            continue
                        norm = dark and a.dark_action == 'norm'
                    if a.dark_rule == 'none' and a.dark_skip and self.floor is not None and tmean is not None \
                            and float(tmean.mean()) < self.floor:
                        # under-exposed patch: the clean exposure / tone curve is not identifiable from the input, so
                        # no synthetic pair can supervise it -- abstain (the source keeps its learned mapping)
                        self.skipped += 1
                        self.last_stats = {'skipped': 1, 'dark_skip': 1, 'x_in': float(x.mean())}
                        continue
                    select = self._select if self.target_model is not None else None
                    if f0x is None and (a.cons_weight > 0 or a.redeg_weight > 0):
                        f0x = self.teacher(x)
                    cmean = None
                    legacy_anchor = (a.dark_rule == 'none' and a.dark_anchor and self.floor is not None
                                     and tmean is not None and float(tmean.mean()) < self.floor)
                    if legacy_anchor or (dark and a.dark_action == 'anchor'):
                        if f0x is None:
                            f0x = self.teacher(x)
                        cmean = f0x.clamp(0, 1).mean(dim=(0, 2, 3), keepdim=True)   # the source's exposure for x
                        st['dark_anchor'] = float(cmean.mean())
                    y, c, fams, scores = degrade_batch(c, self.families, self.gen, tmean, a.exposure_match,
                                                       select=select, k=a.abc_k, floor=self.floor, clean_mean=cmean,
                                                       dark=dark, ll=a.lowlight_op, topk=a.abc_topk, blur=a.blur_op)
                    if scores:
                        st['abc_d'] = sum(scores) / len(scores)
                    if self.abc_eps is not None and scores:
                        keep = [i for i, d in enumerate(scores) if d <= self.abc_eps]
                        st['abc_accept'] = len(keep) / len(scores)
                        if not keep:   # no synthetic pair is a plausible target sample: no update
                            self.skipped += 1
                            self.last_stats = {**st, 'skipped': 1, 'x_in': float(x.mean())}
                            continue
                        if len(keep) < len(scores):
                            y, c = y[keep], c[keep]
                            fams = [fams[i] for i in keep]
                    if a.redeg_weight > 0:
                        chain = draw_chain(self.families, self.gen_t)
                        dx, xk = apply_chain(x, chain, self.gen_t, tmean, 'fit')
                        f0xk = f0x if torch.equal(xk, x) else self.teacher(xk)
                        st['redeg_fam'] = chain_label(chain)
            for f in fams:
                self.fam_counts[f] = self.fam_counts.get(f, 0) + 1
            # one backward per term (gradients add up; only one graph is alive at a time)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(y)
            if norm:   # under-exposed: compare after matching the output's brightness to the clean image's
                mo = out.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
                mc = c.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
                sup = (out * (mc / mo) - c).abs().mean()
            else:
                sup = self.criterion(out, c)
            sup.backward()
            loss = float(sup)
            out = out.detach()
            if a.cons_weight > 0 or a.nss_weight > 0:
                fx = self.model(x)
                tgt = 0.
                if a.cons_weight > 0:
                    cons = (self._band(fx) - self._band(f0x)).abs().mean()
                    tgt = tgt + a.cons_weight * cons
                    st['cons'] = float(cons)
                if a.nss_weight > 0:
                    nss = self.clean_model.distance(self._feat(fx.clamp(0, 1))).mean()
                    tgt = tgt + a.nss_weight * nss
                    st['nss'] = float(nss)
                tgt.backward()
                loss += float(tgt)
                del fx
            if a.redeg_weight > 0:
                redeg = (self.model(dx) - f0xk).abs().mean()
                (a.redeg_weight * redeg).backward()
                loss += a.redeg_weight * float(redeg)
                st['redeg'] = float(redeg)
            pen = self.anchor.penalty()
            if torch.is_tensor(pen):
                pen.backward()
                loss += float(pen)
            self.optimizer.step()
            self.anchor.post_step()
            self.steps += 1
            loss_val = loss
            self.last_stats = {'sup': float(sup),
                               'penalty': float(pen) if pen is not None else 0.0,
                               'x_in': float(x.mean()), 'pair_in': float(y.mean()), 'pair_gt': float(c.mean()),
                               'f_pair': float(out.mean()), 'fams': '+'.join(fams), **st}
        return loss_val

    def reset(self):
        self._restore()


def _gauss_blur(t, sigma):
    r = max(1, int(3 * sigma + 0.5))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=t.dtype, device=t.device) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    C = t.shape[1]
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(t, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


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
