"""Anchors that keep the adapted parameters close to the source theta_0.

| ``--anchor`` | what it does                                                                          |
|--------------|---------------------------------------------------------------------------------------|
| ``none``     | nothing                                                                               |
| ``l2``       | ``lam/2 * sum (theta - theta_0)^2`` over the adapted subset                           |
| ``tips``     | DCTTA's TIPS hard mask -- only for ``--method dctta`` (baseline), which runs it itself |

For LoRA parameters the L2 anchor acts on the effective weight change ``Delta W = (alpha/r) B A`` of the
wrapped conv (its theta_0 is 0), not on the factors.

(The source-Fisher anchor and TIPS for non-DCTTA methods were removed after Step 1: no measurable effect.)
"""


class Anchor:
    kind = 'none'

    def prepare(self, loader=None):
        """Called once before adaptation."""

    def penalty(self):
        return None

    def post_step(self):
        """Called after every optimiser step."""


class NoAnchor(Anchor):
    pass


class L2Anchor(Anchor):
    kind = 'l2'

    def __init__(self, subset, lam):
        self.lam = float(lam)
        lora_ids = subset.lora_param_ids()
        self.terms = [(p, p.detach().clone()) for p in subset.params if id(p) not in lora_ids]
        selected = {id(p) for p in subset.params}
        self.lora = [m for m in subset.lora_modules.values() if id(m.lora_A) in selected or id(m.lora_B) in selected]

    def penalty(self):
        total = 0.
        for p, p0 in self.terms:
            total = total + ((p - p0) ** 2).sum()
        for m in self.lora:
            total = total + (m.delta_weight() ** 2).sum()
        return 0.5 * self.lam * total


def build_anchor(kind, model, subset, *, lam=0.0, method=None):
    if kind in (None, 'none'):
        return NoAnchor()
    if kind == 'l2':
        return L2Anchor(subset, lam)
    if kind == 'tips':
        if method != 'dctta':
            raise ValueError("--anchor tips is DCTTA's own mask; use it only with --method dctta")
        marker = NoAnchor()
        marker.kind = 'tips'
        return marker
    raise ValueError(f"unknown anchor '{kind}'")
