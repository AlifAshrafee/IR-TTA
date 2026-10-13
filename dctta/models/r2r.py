"""R2R backbone (Wang et al., "Retrieve-to-Restore", CVPR 2026), https://github.com/AlifAshrafee/R2R

Inference-path port of ``net/model_{3D,5D}.py`` + ``net/feature_bank_{3D,5D}.py`` (commit bf387d5).
All network blocks (NAFBlock, NAFKeyEncoder, NAFMemEncoder, NAFDecoder, the key/value projections,
``AvgPool2d`` / ``replace_layers`` for TLC) are verbatim. What changed, and why:

* **One class for 3 and 5 tasks.** ``model_3D.py`` and ``model_5D.py`` define the same network; they
  differ only in the number of degradation classes (bank slots, ``update_<type>_*_3Dconv`` modules,
  the memory-encoder classifier width). ``R2R(num_classes=3|5)`` covers both and keeps every
  parameter name, so the released Lightning checkpoints load after stripping ``net.``.
* **Degradation bank as buffers.** The original keeps the retrieval bank in Python lists loaded with
  ``load_prompts`` from ``<prompt_dir>/<name>/{type}_{key,value}.pt``; here it is stored in
  persistent buffers ``dm.bank_{key,value}_<type>`` (``[53, 64, 8, 8]`` / ``[53, 512, 8, 8]``), so it
  follows ``.to(device)``, ``deepcopy`` (EMA teachers) and ``state_dict`` (adapted checkpoints are
  self-contained). ``load_bank(dir)`` reads the released files with the same ``drop_last`` slicing
  (first ``T_max - T_max//5 + 1 = 53`` entries per class) as ``load_prompts``. Buffers are never
  parameters, so TTA cannot change the bank.
* **Inference path only.** The ``is_train`` branches (pretraining with clean/degraded pairs, bank
  updates through the 3D convs) are dropped; the forward pass is the original ``is_train=False``
  path. The ``update_*_3Dconv`` and ``mem_encoder`` modules are kept (unused at test time) so the
  checkpoints load with ``strict`` key matching. The 3D bank's ``simi_list.append(...)`` (it stored
  every similarity tensor, and with it the autograd graph, forever) is dropped.
* **Blind retrieval by default.** The official ``test_3D.py`` / ``test_5D.py`` call the network with
  ``interact_label=[task_id]``, which *overrides* the retrieved degradation class with the ground-truth
  task, i.e. the published numbers are evaluated with oracle task labels. Our setting is blind, so the
  default is the network's own prediction (argmax of the per-class top-k similarity, exactly the
  code path taken when ``interact_label is None``). ``R2R(oracle_label=k)`` reproduces the official
  protocol for a test set of task ``k`` (3D: 0 denoise, 1 derain, 2 dehaze; 5D: + 3 deblur, 4 lowlight).
* **TLC.** The official testers wrap the net in ``R2RLocal``, which replaces every
  ``nn.AdaptiveAvgPool2d`` (the NAFBlock channel attention) by TLC's local ``AvgPool2d``
  (Chu et al., ECCV 2022) calibrated on a ``train_size`` dummy forward: 1080x1080 for the 3-task and
  5-task all-in-one evaluations (the 5-task single-task deblur mode uses 224). ``tlc_size=1080``
  (default) does the same lazily on the first forward, on that input's device. With 1080, pooling is
  still global for any input up to ~1.5x1080 px per side; ``tlc_size=None`` keeps the plain global pool.
* The output is cropped back to the input size (the original returned the zero-padded size, which only
  differs for inputs that are not multiples of 16).
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

DEG_TYPES = {3: ["denoise", "derain", "dehaze"],
             5: ["denoise", "derain", "dehaze", "deblur", "lowlight"]}


# ---------------------------------------------------------------------------- #
# TLC (verbatim from net/model_5D.py)
# ---------------------------------------------------------------------------- #
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


# ---------------------------------------------------------------------------- #
# Network blocks (verbatim from net/model_5D.py)
# ---------------------------------------------------------------------------- #
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class NAFKeyEncoder(nn.Module):
    def __init__(self, width=32, middle_blk_num=1, enc_blk_nums=[1,1,1,28]):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.middle_blks = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

    def forward(self, x):
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)

        return x, encs

class Classifier(nn.Module):
    def __init__(self, in_channels=512, num_classes=5):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(x)       # [b, 512, 1, 1]
        x = torch.flatten(x, 1)   # [b, 512]
        x = self.fc(x)            # [b, num_classes]
        return x
class NAFMemEncoder(nn.Module):
    def __init__(self, image_channels=3, deg_channel=1, width=32, middle_blk_num=1, enc_blk_nums=[2,2,4,8], num_classes=5):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=image_channels+deg_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.middle_blks = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        self.classify = Classifier(in_channels=chan//2, num_classes=num_classes)

    def forward(self, x):
        encs = []
        x = self.intro(x)
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            x = down(x)
            encs.append(x)
        y = self.middle_blks(x)

        prob = self.classify(F.relu(encs[-2]))

        return y, prob

class NAFDecoder(nn.Module):
    def __init__(self, chan=512, dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skips = nn.ModuleList()
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

    def forward(self, x, encs, readout):
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip

            x = decoder(x)
        return x

class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        return self.key_proj(x)

class ValueProjection(nn.Module):
    def __init__(self, indim, valuedim):
        super().__init__()
        self.value_proj = nn.Conv2d(indim, valuedim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.value_proj.weight.data)
        nn.init.zeros_(self.value_proj.bias.data)

    def forward(self, x):
        return self.value_proj(x)

class ValueComp(nn.Module):
    def __init__(self, indim, valuedim):
        super().__init__()
        self.value_proj = nn.Conv2d(indim, valuedim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.value_proj.weight.data)
        nn.init.zeros_(self.value_proj.bias.data)

    def forward(self, x):
        return self.value_proj(x)


# ---------------------------------------------------------------------------- #
# Degradation memory (inference part of net/feature_bank_{3D,5D}.py)
# ---------------------------------------------------------------------------- #
def _update_conv(dim, update_step):
    """The (training-only) bank-update block of feature_bank_*.py; kept for checkpoint keys."""
    return nn.Sequential(
        nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(update_step, 3, 3),
                  padding=(0, 1, 1), groups=dim),
        nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
        nn.BatchNorm3d(dim),
        nn.ReLU(),
    )


class DegradationMemory(nn.Module):
    def __init__(self, num_classes=3, T_max=64, key_dim=64, value_dim=512, bank_hw=8):
        super().__init__()
        self.deg_types = DEG_TYPES[num_classes]
        self.T_max = T_max
        self.update_step = T_max // 5
        self.key_dim = key_dim
        self.value_dim = value_dim
        for t in self.deg_types:
            setattr(self, f"update_{t}_key_3Dconv", _update_conv(key_dim, self.update_step))
            setattr(self, f"update_{t}_value_3Dconv", _update_conv(value_dim, self.update_step))
        for t in self.deg_types:
            self.register_buffer(f"bank_key_{t}", torch.zeros(0, key_dim, bank_hw, bank_hw))
            self.register_buffer(f"bank_value_{t}", torch.zeros(0, value_dim, bank_hw, bank_hw))
        self.last_scores = None       # [B, num_classes] class scores of the last retrieval (detached)
        self.last_prediction = None   # [B] retrieved class index of the last call (detached)

    def bank_names(self):
        return [f"bank_{kind}_{t}" for t in self.deg_types for kind in ("key", "value")]

    def has_bank(self):
        return all(getattr(self, n).shape[0] > 0 for n in self.bank_names())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # resize the (initially empty) bank buffers to the stored ones before the default copy
        for n in self.bank_names():
            if prefix + n in state_dict:
                setattr(self, n, torch.empty_like(state_dict[prefix + n], device=getattr(self, n).device))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def load_bank(self, bank_dir, drop_last=True):
        """``load_prompts``: ``bank_dir`` holds ``{type}_key.pt`` / ``{type}_value.pt`` (lists of [1,C,h,w])."""
        kv_compress = self.T_max - self.update_step + 1 if drop_last else self.T_max
        for t in self.deg_types:
            for kind in ("key", "value"):
                entries = torch.load(os.path.join(bank_dir, f"{t}_{kind}.pt"), map_location="cpu")
                bank = torch.cat(list(entries)[:kv_compress], dim=0).float()
                setattr(self, f"bank_{kind}_{t}", bank.to(getattr(self, f"bank_{kind}_{t}").device))

    def get_deg_prompt(self, qk, interact_label=None):
        if not self.has_bank():
            raise RuntimeError("R2R degradation bank is empty: pass bank_dir=<dir with {type}_key.pt ...> "
                               "or a checkpoint that contains dm.bank_* buffers")
        keys = [getattr(self, f"bank_key_{t}") for t in self.deg_types]
        values = [getattr(self, f"bank_value_{t}") for t in self.deg_types]
        mk = torch.cat(keys, dim=0)
        mv = torch.cat(values, dim=0)
        sizes = [k.shape[0] for k in keys]
        return self.comprehensive_attention_processing(mk, qk, mv, sizes, interact_label=interact_label)

    def comprehensive_attention_processing(self, mk, qk, mv, sizes, interact_label=None):
        """Class-aware memory attention, ``pred_method='topk'`` (the only one the released code uses).

        Identical to the 3D/5D originals: cosine similarity / sqrt(C*h*w); class score = mean of the
        top-min(n_c) similarities of that class; predicted class = argmax (or ``interact_label``);
        soft attention restricted to the predicted class's entries.
        """
        device = mk.device
        batch_size = qk.shape[0]
        n_total = sum(sizes)
        h, w = qk.shape[2], qk.shape[3]
        ck = mk.shape[1]

        mk_flat = mk.view(n_total, -1)
        qk_flat = qk.view(batch_size, -1)
        mv_flat = mv.view(n_total, -1)
        mk_flat = F.normalize(mk_flat, dim=1)
        qk_flat = F.normalize(qk_flat, dim=1)
        similarity = torch.matmul(qk_flat, mk_flat.t()) / math.sqrt(ck * h * w)

        bounds = [0]
        for n in sizes:
            bounds.append(bounds[-1] + n)
        topk_k = min(sizes)
        scores = torch.stack([torch.topk(similarity[:, bounds[c]:bounds[c + 1]], topk_k, dim=1).values.mean(dim=1)
                              for c in range(len(sizes))], dim=1)  # [batch_size, num_classes]
        predictions = torch.argmax(scores, dim=1)
        if interact_label is not None:
            predictions = interact_label
        self.last_scores = scores.detach()
        self.last_prediction = torch.as_tensor(predictions).detach()

        class_masks = torch.zeros(len(sizes), n_total, dtype=torch.bool, device=device)
        for c in range(len(sizes)):
            class_masks[c, bounds[c]:bounds[c + 1]] = True

        sample_masks = class_masks[predictions]
        masked_similarity = similarity.clone()
        masked_similarity[~sample_masks] = -1e6

        weights = torch.softmax(masked_similarity, dim=1)
        weighted_output_flat = torch.matmul(weights, mv_flat)
        output = weighted_output_flat.view(batch_size, self.value_dim, h, w)
        return scores, output


# ---------------------------------------------------------------------------- #
# R2R (is_train=False path of net/model_{3D,5D}.py :: R2R.forward, + R2RLocal's TLC conversion)
# ---------------------------------------------------------------------------- #
class R2R(nn.Module):
    def __init__(self, num_classes=3, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28],
                 dec_blk_nums=[1, 1, 1, 1], oracle_label=None, tlc_size=1080):
        super().__init__()
        self.num_classes = num_classes
        key_dim = width * 2
        value_dim = width * 16
        self.encoder = NAFKeyEncoder(width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blk_nums)
        self.mem_encoder = NAFMemEncoder(img_channel, deg_channel=1, width=width, middle_blk_num=middle_blk_num,
                                         enc_blk_nums=[2, 2, 4, 8], num_classes=self.num_classes)
        self.decoder = NAFDecoder(chan=width * 16, dec_blk_nums=dec_blk_nums)

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)
        self.padder_size = 2 ** len(enc_blk_nums)

        self.key_proj = KeyProjection(width * 16, keydim=key_dim)
        self.key_comp = nn.Conv2d(width * 16, value_dim, kernel_size=3, padding=1)
        self.value_proj = ValueProjection(width * 16, valuedim=key_dim)
        self.value_comp = ValueComp(width * 16, valuedim=value_dim)

        self.dm = DegradationMemory(num_classes=num_classes, T_max=64, key_dim=key_dim, value_dim=value_dim)

        self.fuse_gate = nn.Conv2d(2 * value_dim, value_dim, kernel_size=1, groups=value_dim)

        self.oracle_label = oracle_label
        self.tlc_size = tlc_size
        self._tlc_pending = tlc_size is not None

    @torch.no_grad()
    def _convert_tlc(self, device):
        """``R2RLocal.__init__`` + ``Local_Base.convert``: swap the pools, then fix their kernels with one
        dummy forward at ``train_size`` (the original used ``torch.rand``; the values do not matter)."""
        self._tlc_pending = False
        train_size = (1, 3, self.tlc_size, self.tlc_size)
        base_size = (int(train_size[2] * 1.5), int(train_size[3] * 1.5))
        replace_layers(self, base_size, train_size, fast_imp=False)
        self.forward(torch.zeros(train_size, device=device))

    def forward(self, inp):
        if self._tlc_pending:
            self._convert_tlc(inp.device)
        H0, W0 = inp.shape[-2:]
        inp = self.check_image_size(inp)
        B, C, H, W = inp.shape
        x = self.intro(inp)
        x, encs = self.encoder(x)

        qk = self.key_proj(x)
        qv = self.key_comp(x)

        interact_label = None
        if self.oracle_label is not None:
            interact_label = torch.full((B,), int(self.oracle_label), dtype=torch.long, device=inp.device)
        qk_ = F.interpolate(qk, size=(8, 8), mode="area")
        scores, read_out = self.dm.get_deg_prompt(qk_, interact_label=interact_label)
        read_out = F.interpolate(read_out, size=(qv.shape[-2], qv.shape[-1]), mode='area')

        interleaved = torch.zeros(B, 2 * x.shape[1], x.shape[2], x.shape[3]).to(x.device)
        interleaved[:, 0::2, :, :] = qv
        interleaved[:, 1::2, :, :] = read_out
        gate = torch.sigmoid(self.fuse_gate(interleaved))
        x = qv * (1 - gate) + read_out * gate

        x = self.decoder(x, encs, read_out)
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H0, :W0]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    @staticmethod
    def num_classes_from_state_dict(state_dict):
        return 5 if any(k.startswith("dm.update_deblur_key_3Dconv") for k in state_dict) else 3
