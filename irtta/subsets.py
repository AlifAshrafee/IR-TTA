"""Parameter subsets for adaptation, selected by name.

``--subset`` is a ``+``-joined list of parts:

| part            | parameters (any backbone)                                                                  |
|-----------------|--------------------------------------------------------------------------------------------|
| ``ln``          | every LayerNorm affine (Restormer-style ``*_LayerNorm``, NAFNet ``LayerNorm2d``, nn.LayerNorm) |
| ``lora<r>``     | zero-init LoRA of rank r on every dense 1x1 convolution with min(in, out) > r (the channel   |
|                 | projections: qkv / project_in / project_out in Restormer-family blocks, conv1/3/4/5 in NAFNet) |
| ``full``        | DCTTA's rule: top-level prefix filter (``dctta.models.TRAIN_PREFIXES[backbone]``; PromptIR: |
|                 | prompt/encoder/decoder, bit-identical to the release) + ``weight``/``bias`` leaves          |
| ``all``         | every parameter                                                                            |
| ``re:<regex>``  | every parameter whose name matches the regex                                               |
| ``none``        | nothing (source model)                                                                     |

Only parts every backbone has are offered (the PromptIR-specific prompt-pathway subsets were removed after
Step 1: the routing softmax is saturated and adapting it changed nothing). LoRA A matrices are drawn from a
dedicated ``torch.Generator``, so building a subset never touches the global RNG: for a given ``--seed`` every subset sees the same
patch order and (for DCTTA) the same generator initialisation.
"""
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from dctta.tta import DEFAULT_TRAIN_PREFIXES, collect_params, configure_model


class LoRAConv1x1(nn.Module):
    """``base(x) + (alpha / r) * B(A(x))`` for a frozen 1x1 conv; B is zero-initialised.

    ``delta_weight()`` is the effective weight change (out, in, 1, 1), the quantity the anchors act on.
    ``orig_name`` is the name of the wrapped weight in the *un-injected* model (for Fisher lookup).
    """

    def __init__(self, base: nn.Conv2d, rank, alpha, generator, orig_name):
        super().__init__()
        if base.kernel_size != (1, 1) or base.groups != 1:
            raise ValueError(f"LoRA expects a dense 1x1 conv, got {base}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank, self.alpha, self.scale = rank, alpha, alpha / rank
        self.orig_name = orig_name
        w = base.weight
        bound = 1.0 / math.sqrt(base.in_channels)  # == nn.init.kaiming_uniform_(a=sqrt(5)) for (r, in)
        a = (torch.rand(rank, base.in_channels, generator=generator) * 2 - 1) * bound
        self.lora_A = nn.Parameter(a.to(device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_channels, rank, device=w.device, dtype=w.dtype))

    def delta_weight(self):
        return (self.lora_B @ self.lora_A).mul(self.scale)[:, :, None, None]

    def forward(self, x):
        z = F.conv2d(x, self.lora_A[:, :, None, None])
        return self.base(x) + self.scale * F.conv2d(z, self.lora_B[:, :, None, None])

    def extra_repr(self):
        return f"rank={self.rank}, alpha={self.alpha}"


def _is_lora_target(m, rank):
    return (type(m) is nn.Conv2d and m.kernel_size == (1, 1) and m.groups == 1
            and min(m.in_channels, m.out_channels) > rank)


def inject_lora(model, rank, alpha=None, scope=None, seed=0):
    """Wrap every dense 1x1 conv with min(in, out) > rank (optionally only names matching ``scope``) in
    ``LoRAConv1x1``. Returns ``{orig_weight_name: module}``."""
    alpha = float(rank if alpha is None else alpha)
    gen = torch.Generator().manual_seed(int(seed) + 7919 * rank)
    scope_re = re.compile(scope) if scope else None
    found = [(name, m) for name, m in model.named_modules()
             if _is_lora_target(m, rank) and (scope_re is None or scope_re.search(name))]
    if not found:
        raise ValueError(f"no LoRA targets matched (rank={rank}, scope={scope})")
    modules = OrderedDict()
    for name, conv in found:
        parent_name, _, child = name.rpartition('.')        # top-level convs have no parent prefix
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapped = LoRAConv1x1(conv, rank, alpha, gen, orig_name=name + '.weight')
        setattr(parent, child, wrapped)
        modules[name + '.weight'] = wrapped
    return modules


@dataclass
class Subset:
    spec: str
    params: list = field(default_factory=list)
    names: list = field(default_factory=list)
    lora_modules: dict = field(default_factory=dict)   # orig weight name -> LoRAConv1x1

    @property
    def n_params(self):
        return int(sum(p.numel() for p in self.params))

    def lora_param_ids(self):
        return {id(p) for m in self.lora_modules.values() for p in (m.lora_A, m.lora_B)}


def _ln_param_names(model):
    names = set()
    for mname, m in model.named_modules():
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)) or type(m).__name__.endswith(('_LayerNorm', 'LayerNorm2d')):
            for pname, _ in m.named_parameters(recurse=False):
                names.add(f"{mname}.{pname}")
    return names


def apply_subset(model, spec, *, lora_alpha=None, lora_scope=None, train_prefixes=DEFAULT_TRAIN_PREFIXES, seed=0,
                 logger=None):
    """Inject LoRA if requested, set ``requires_grad``, return the ``Subset`` to optimise."""
    parts = [p.strip() for p in spec.split('+') if p.strip()]
    if not parts:
        raise ValueError("empty --subset")
    exclusive = {'full', 'all', 'none'}
    if any(p in exclusive for p in parts) and len(parts) > 1:
        raise ValueError(f"'{'/'.join(sorted(exclusive))}' cannot be combined with other parts: {spec}")

    sub = Subset(spec=spec)
    if parts == ['full']:  # DCTTA's selection, through the very same functions (bit-identical)
        configure_model(model, train_prefixes)
        sub.params, sub.names = collect_params(model)
        return sub

    # injections first: they create parameters
    for part in parts:
        m = re.fullmatch(r'lora(\d*)', part)
        if m:
            rank = int(m.group(1) or 4)
            mods = inject_lora(model, rank, lora_alpha, lora_scope, seed)
            sub.lora_modules.update(mods)

    selected = set()
    named = OrderedDict(model.named_parameters())
    for part in parts:
        if part == 'none':
            continue
        if part == 'all':
            selected |= set(named)
        elif part == 'ln':
            selected |= _ln_param_names(model)
        elif re.fullmatch(r'lora(\d*)', part):
            selected |= {n for n in named if n.endswith('lora_A') or n.endswith('lora_B')}
        elif part.startswith('re:'):
            rx = re.compile(part[3:])
            selected |= {n for n in named if rx.search(n)}
        else:
            raise ValueError(f"unknown subset part '{part}' in '{spec}'")

    model.train()
    for n, p in named.items():
        p.requires_grad = n in selected
        if n in selected:
            sub.params.append(p)
            sub.names.append(n)
    if parts != ['none'] and not sub.params:
        raise ValueError(f"subset '{spec}' selected no parameters")
    if logger is not None:
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(f"subset '{spec}': {sub.n_params:,} adapted parameters "
                    f"({100 * sub.n_params / max(n_total, 1):.3f}% of {n_total:,}) in {len(sub.params)} tensors")
    return sub


def summarize_subset(sub: Subset):
    """``{top-level module: n adapted parameters}``."""
    out = OrderedDict()
    for n, p in zip(sub.names, sub.params):
        top = n.split('.')[0]
        out[top] = out.get(top, 0) + p.numel()
    return out
