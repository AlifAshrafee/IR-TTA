"""Backbone registry: ``build_model(name, ckpt, **opts) -> nn.Module`` with ``model(x) -> restored``.

Every backbone is the authors' architecture with their parameter names, so the officially released
checkpoints load directly (Lightning ``.ckpt`` with ``net.``-prefixed keys, or a plain state_dict such
as the one ``adapt.py --save_ckpt`` writes). The released variant (3 vs 5 tasks, tiny vs small, ...)
is read off the checkpoint, so callers only pass the backbone name and the checkpoint path.

=========  ==================================================  =====================================
name       released checkpoints                                notes
=========  ==================================================  =====================================
promptir   model.ckpt (3-task), epoch=80.ckpt (5-task)          unchanged DCTTA path
adair      adair3d.ckpt / adair5d.ckpt                          verbatim
bioir      BioIR3D.ckpt / BioIR5D.ckpt                          verbatim
moceir     <id>_AIO3/last.ckpt, <id>_AIO5/last.ckpt              MoCE_IR (dim 48) / MoCE_IR_S (dim 32)
                                                                inferred; opts: routing_noise
mirage     3Deg_{tiny,small}.ckpt / 5Deg_{tiny,small}.ckpt     tiny/small inferred; BN + DropPath stay
                                                                in eval mode under model.train()
r2r        train_ckpt_{3D,5D}_finetune/last.ckpt               + its degradation bank
           + save_prompts_{3D,5D}/last/{type}_{key,value}.pt   (opts: bank_dir; auto-found next to the
                                                                ckpt); blind retrieval by default
                                                                (opts: oracle_label, tlc_size)
=========  ==================================================  =====================================

``TRAIN_PREFIXES`` is the per-backbone analogue of DCTTA's released PromptIR rule (adapt the encoder
and decoder levels plus the degradation-conditioning modules; keep the bottleneck, refinement, up/down
samplers and in/out convolutions frozen). It is only the default of ``adapt.py --train_prefixes``.
"""
import ast
import glob
import logging
import os

import torch

__all__ = ['BACKBONES', 'TRAIN_PREFIXES', 'build_model', 'load_state_dict_file', 'parse_backbone_opts']

BACKBONES = ('promptir', 'adair', 'bioir', 'moceir', 'mirage', 'r2r')

TRAIN_PREFIXES = {
    'promptir': ['prompt', 'encoder', 'decoder'],        # DCTTA release: prompt{1,2,3}, encoder/decoder_level*
    'adair': ['fre', 'encoder', 'decoder'],              # fre{1,2,3}: frequency mining/modulation (the PGM analogue)
    'bioir': ['encoder', 'decoder'],                     # no separate conditioning module
    'moceir': ['enc', 'dec', 'freq_embed'],              # dec holds the complexity experts + routers
    'mirage': ['encoder', 'decoder'],                    # no separate conditioning module
    'r2r': ['encoder', 'decoder', 'key_', 'fuse_gate'],  # key_proj/key_comp: retrieval query; fuse_gate: bank fusion
}


def parse_backbone_opts(items):
    """``['bank_dir=pretrain/r2r', 'oracle_label=1', 'tlc_size=None']`` -> dict (values via literal_eval)."""
    opts = {}
    for item in items or []:
        key, _, value = item.partition('=')
        try:
            opts[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            opts[key] = value
    return opts


def load_state_dict_file(path):
    """State dict from a Lightning checkpoint (``net.`` prefix stripped) or a plain/our saved state_dict."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    return {k[4:] if k.startswith("net.") else k: v for k, v in state_dict.items()}


def _load(model, state_dict, ckpt_path, logger, allowed_missing=()):
    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if k not in allowed_missing]
    log = logger or logging.getLogger(__name__)
    if missing:
        log.warning(f"checkpoint {ckpt_path}: {len(missing)} missing keys, e.g. {missing[:5]}")
    if result.unexpected_keys:
        log.warning(f"checkpoint {ckpt_path}: {len(result.unexpected_keys)} unexpected keys, "
                    f"e.g. {result.unexpected_keys[:5]}")
    if not missing and not result.unexpected_keys:
        log.info(f"checkpoint {ckpt_path}: all {len(state_dict)} tensors matched")
    return model


def _find_r2r_bank(ckpt_path, num_classes):
    """Look for ``denoise_key.pt`` near the checkpoint: its dir, ``last/`` below it, or a sibling
    ``save_prompts_{3D,5D}/last`` (the official layout)."""
    d = os.path.dirname(os.path.abspath(ckpt_path))
    tag = f"{num_classes}D"
    candidates = [d, os.path.join(d, 'last'), os.path.join(d, 'prompts'), os.path.join(d, 'prompts', 'last'),
                  os.path.join(os.path.dirname(d), f'save_prompts_{tag}', 'last')]
    candidates += sorted(glob.glob(os.path.join(d, '**', 'denoise_key.pt'), recursive=True))
    for c in candidates:
        c = c[:-len('denoise_key.pt')] if c.endswith('denoise_key.pt') else c
        if os.path.isfile(os.path.join(c, 'denoise_key.pt')):
            return c
    return None


def build_model(name, ckpt=None, logger=None, **opts):
    """Instantiate backbone ``name`` and load ``ckpt`` (architecture only if ``ckpt`` is None).

    ``opts`` are backbone-specific (see the module docstring); for promptir this is exactly the
    original ``PromptIR(decoder=True)`` + ``load_promptir_checkpoint`` sequence (same RNG use).
    """
    name = name.lower()
    log = logger or logging.getLogger(__name__)
    if name == 'promptir':
        from .promptir import PromptIR
        from ..utils import load_promptir_checkpoint
        model = PromptIR(decoder=True)
        if ckpt is not None:
            load_promptir_checkpoint(model, ckpt, logger)
        return model

    state_dict = load_state_dict_file(ckpt) if ckpt is not None else None
    allowed_missing = ()

    if name == 'adair':
        from .adair import AdaIR
        model = AdaIR(decoder=True)
    elif name == 'bioir':
        from .bioir import BioIR
        model = BioIR()
    elif name == 'moceir':
        from .moceir import MoCEIR
        variant = opts.pop('variant', None)
        if variant is None:
            variant = 'MoCE_IR_S' if state_dict is not None and state_dict['patch_embed.proj.weight'].shape[0] == 32 \
                else 'MoCE_IR'
        model = MoCEIR(**MoCEIR.ckpt_config[variant], **opts)
        log.info(f"MoCE-IR variant {variant}, routing_noise={opts.get('routing_noise', True)}")
    elif name == 'mirage':
        from .mirage import MIRAGE
        variant = opts.pop('variant', None)
        if variant is None:
            variant = 'tiny' if state_dict is not None and state_dict['patch_embed.proj.weight'].shape[0] == 24 \
                else 'small'
        model = MIRAGE(**MIRAGE.ckpt_config[variant])
        log.info(f"MIRAGE variant {variant}")
    elif name == 'r2r':
        from .r2r import R2R
        bank_dir = opts.pop('bank_dir', None)
        num_classes = opts.pop('num_classes', None)
        if num_classes is None:
            num_classes = R2R.num_classes_from_state_dict(state_dict) if state_dict is not None else 3
        model = R2R(num_classes=num_classes, **opts)
        allowed_missing = tuple(f"dm.{n}" for n in model.dm.bank_names())
        has_bank = state_dict is not None and all(k in state_dict for k in allowed_missing)
        if not has_bank and ckpt is not None:
            bank_dir = bank_dir or _find_r2r_bank(ckpt, num_classes)
            if bank_dir is None:
                raise FileNotFoundError("R2R needs its degradation bank (save_prompts_*/last/{type}_{key,value}.pt): "
                                        "pass --backbone_opts bank_dir=<dir>")
        log.info(f"R2R {num_classes}-task, retrieval={'oracle %s' % model.oracle_label if model.oracle_label is not None else 'blind'}, "
                 f"tlc_size={model.tlc_size}, bank={'checkpoint' if has_bank else bank_dir}")
    else:
        raise ValueError(f"unknown backbone {name!r}; choose from {BACKBONES}")

    if state_dict is not None:
        _load(model, state_dict, ckpt, logger, allowed_missing)
    if name == 'r2r' and ckpt is not None and not has_bank:
        model.dm.load_bank(bank_dir)
    return model
