"""Degradation operator library for transfer TTA (v0.1): physical primitives composed in physical order.

Transfer TTA trains the restorer on synthetic pairs ``(op(c), c)`` built from *clean* content ``c``, so the
supervision target is genuinely clean and never comes from the source model's own predictions.

Primitives, always applied in this order (each can be off):

| stage     | primitive | model                                                                                 |
|-----------|-----------|---------------------------------------------------------------------------------------|
| scene     | rain      | sparse seeds smeared by an oriented line kernel, screen-blended                      |
|           | haze      | scattering c*t + A*(1-t), t = exp(-beta*d), smooth random depth with a sky ramp       |
| optics    | blur      | non-negative kernel summing to 1 (mean-preserving): Gaussian or camera-shake, <=31 px |
| exposure  | exposure  | s * cast * y^gamma (under-exposure)                                                   |
| sensor    | noise     | Poisson-Gaussian, var = a*y + sigma^2                                                 |

A *chain* is a dict ``{'scene': None|'rain'|'haze', 'blur': bool, 'exposure': bool, 'noise': bool}``. The
single-family chains are blur={blur}, noise={noise}, rain={scene: rain}, haze={scene: haze},
lowlight={exposure, noise}; ``compose`` draws a random multi-primitive chain.

Exposure matching. Each synthetic degraded image is brought to the brightness (channel means) of a target patch,
and the clean image is whatever scene produced it:
* ``fit`` (v0.2, default): the free brightness variable is solved *through the chain*, so the degraded image
  matches the target --
  - chains that do not set the exposure themselves (blur, noise, rain): a per-channel scene gain k on c, refined
    by re-running the chain with the same random draws (rain screen-blends light into the image, so the clean
    scene is darker than the rainy observation -- v0 had this property, v0.1 lost it);
  - chains that do (low-light, haze): the operator's own parameter -- the exposure gain s, or the airlight A --
    and the clean image keeps its natural exposure (v0 gained the clean image too and over-exposed it).
* ``clean`` (v0.1): the clean scene is gained to the target means *before* degradation (rain then makes the
  degraded image brighter than the target); exposure-changing chains as in ``fit``.
* ``none``: no matching.

All random numbers come from a dedicated CPU ``torch.Generator`` (the global RNG is never touched).
"""
import math

import torch
import torch.nn.functional as F

FAMILIES = ('blur', 'noise', 'rain', 'haze', 'lowlight')
TASK_TO_FAMILY = {'deblur': 'blur', 'denoise': 'noise', 'derain': 'rain', 'dehaze': 'haze', 'enhance': 'lowlight'}
KSIZE = 31


def _u(g, lo, hi):
    return lo + (hi - lo) * float(torch.rand(1, generator=g))


# ---------------------------------------------------------------------------- #
# kernels
# ---------------------------------------------------------------------------- #
def gaussian_kernel(size, sigma_x, sigma_y, theta):
    """Anisotropic Gaussian, rotated by ``theta`` (radians), normalised to sum 1. (size, size) tensor."""
    r = (size - 1) / 2
    ax = torch.linspace(-r, r, size)
    yy, xx = torch.meshgrid(ax, ax, indexing='ij')
    c, s = math.cos(theta), math.sin(theta)
    u, v = c * xx + s * yy, -s * xx + c * yy
    k = torch.exp(-0.5 * ((u / sigma_x) ** 2 + (v / sigma_y) ** 2))
    return k / k.sum()


def shake_kernel(size, g, steps=64):
    """Camera-shake kernel: a random-walk trajectory with momentum, splatted bilinearly, normalised to sum 1."""
    length = _u(g, 0.15, 0.9) * (size // 2)
    vel = torch.randn(2, generator=g)
    pos = torch.zeros(2)
    pts = []
    for _ in range(steps):
        vel = 0.8 * vel + 0.6 * torch.randn(2, generator=g)
        pos = pos + vel / max(float(vel.norm()), 1e-6)
        pts.append(pos.clone())
    pts = torch.stack(pts)
    pts = pts - pts.mean(0)
    extent = float(pts.abs().max())
    pts = pts * (length / max(extent, 1e-6)) + (size - 1) / 2
    k = torch.zeros(size, size)
    for x, y in pts.tolist():
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - x0, y - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = x0 + dx, y0 + dy
            if 0 <= xi < size and 0 <= yi < size:
                k[yi, xi] += w
    k = F.conv2d(k[None, None], gaussian_kernel(5, 0.6, 0.6, 0.0)[None, None], padding=2)[0, 0]
    return k / k.sum()


def random_blur_kernel(g, size=KSIZE):
    if float(torch.rand(1, generator=g)) < 0.5:
        sx = _u(g, 0.5, 4.0)
        sy = _u(g, 0.5, 4.0)
        return gaussian_kernel(size, sx, sy, _u(g, 0, math.pi))
    return shake_kernel(size, g)


def conv_same(img, k):
    """img (1,C,H,W), k (kh,kw) -> per-channel convolution with reflect padding."""
    C = img.shape[1]
    kh, kw = k.shape
    w = k.to(img)[None, None].repeat(C, 1, 1, 1)
    x = F.pad(img, (kw // 2, kw - 1 - kw // 2, kh // 2, kh - 1 - kh // 2), mode='reflect')
    return F.conv2d(x, w, groups=C)


# ---------------------------------------------------------------------------- #
# operators: (1,3,H,W) in [0,1] -> (1,3,H,W) in [0,1]
# ---------------------------------------------------------------------------- #
def _pg_noise(y, g, a, sigma):
    n = torch.randn(y.shape, generator=g).to(y)
    return y + torch.sqrt(a * y.clamp_min(0) + sigma ** 2) * n


def op_blur(c, g):
    return conv_same(c, random_blur_kernel(g)).clamp(0, 1)


def op_noise(c, g):
    return _pg_noise(c, g, a=_u(g, 0.0, 0.02), sigma=_u(g, 0.01, 0.10)).clamp(0, 1)


def op_rain(c, g):
    _, _, H, W = c.shape
    density = _u(g, 0.002, 0.02)
    seeds = (torch.rand(1, 1, H, W, generator=g) < density).float() * (0.5 + 0.5 * torch.rand(1, 1, H, W, generator=g))
    length = _u(g, 8, 30)
    angle = math.pi / 2 + math.radians(_u(g, -30, 30))
    k = gaussian_kernel(KSIZE, length / 2.5, _u(g, 0.4, 1.2), angle)
    streak = conv_same(seeds.to(c), k)
    streak = streak / streak.amax().clamp_min(1e-6)
    r = (_u(g, 0.3, 0.9) * streak).clamp(0, 1)
    return (1 - (1 - c) * (1 - r)).clamp(0, 1)


def op_haze(c, g, target_mean=None):
    """``target_mean`` (1,3,1,1) or None: if given, the airlight A is solved per channel so the hazy image's
    channel means match it (clamped to [0.3, 1]); otherwise A is drawn from the prior."""
    _, _, H, W = c.shape
    field = F.interpolate(torch.rand(1, 1, 4, 4, generator=g), size=(H, W), mode='bicubic', align_corners=False)
    field = (field - field.amin()) / (field.amax() - field.amin()).clamp_min(1e-6)
    ramp = torch.linspace(1, 0, H).view(1, 1, H, 1).expand(1, 1, H, W)
    w = _u(g, 0.0, 0.7)
    depth = (0.2 + 0.8 * (w * ramp + (1 - w) * field)).to(c)
    t = torch.exp(-_u(g, 0.5, 2.5) * depth)
    if target_mean is not None:
        num = target_mean - (c * t).mean(dim=(2, 3), keepdim=True)
        A = (num / (1 - t).mean(dim=(2, 3), keepdim=True).clamp_min(1e-4)).clamp(0.3, 1.0)
    else:
        A = _u(g, 0.7, 1.0) + torch.tensor([_u(g, -0.05, 0.05) for _ in range(3)]).view(1, 3, 1, 1).to(c)
    return (c * t + A * (1 - t)).clamp(0, 1)


def op_exposure(c, g, target_mean=None):
    """Under-exposure s * cast * c^gamma. With ``target_mean`` the gain s is solved so the overall mean matches
    the target's (clamped to [0.01, 1]: this primitive only darkens); otherwise s ~ U(0.05, 0.35)."""
    gamma = _u(g, 1.0, 2.0)
    cast = torch.tensor([_u(g, 0.9, 1.1) for _ in range(3)]).view(1, 3, 1, 1).to(c)
    base = cast * c.clamp_min(0) ** gamma
    if target_mean is not None:
        s = (target_mean.mean() / base.mean().clamp_min(1e-4)).clamp(0.01, 1.0)
    else:
        s = _u(g, 0.05, 0.35)
    return (s * base).clamp(0, 1)


def op_sensor_noise(c, g):
    """Sensor noise after exposure: relative to a dark signal this is strong, as in real low-light captures."""
    return _pg_noise(c, g, a=_u(g, 0.0, 0.01), sigma=_u(g, 0.002, 0.03)).clamp(0, 1)


FAMILY_CHAINS = {
    'blur': {'scene': None, 'blur': True, 'exposure': False, 'noise': False},
    'noise': {'scene': None, 'blur': False, 'exposure': False, 'noise': True},
    'rain': {'scene': 'rain', 'blur': False, 'exposure': False, 'noise': False},
    'haze': {'scene': 'haze', 'blur': False, 'exposure': False, 'noise': False},
    'lowlight': {'scene': None, 'blur': False, 'exposure': True, 'noise': True},
}


def chain_label(chain):
    parts = ([chain['scene']] if chain['scene'] else []) + [k for k in ('blur', 'exposure', 'noise') if chain[k]]
    return '+'.join(parts) or 'none'


def random_chain(g, p_scene=(1 / 3, 1 / 3, 1 / 3), p_blur=0.5, p_exposure=0.3, p_noise=0.5):
    """A random multi-primitive chain (``compose``): scene none/rain/haze, then blur, exposure, noise, each on with
    its probability; redrawn until at least one primitive is active."""
    while True:
        r = float(torch.rand(1, generator=g))
        scene = None if r < p_scene[0] else ('rain' if r < p_scene[0] + p_scene[1] else 'haze')
        chain = {'scene': scene,
                 'blur': float(torch.rand(1, generator=g)) < p_blur,
                 'exposure': float(torch.rand(1, generator=g)) < p_exposure,
                 'noise': float(torch.rand(1, generator=g)) < p_noise}
        if chain['scene'] or chain['blur'] or chain['exposure'] or chain['noise']:
            return chain


def _run_chain(c, chain, g, target_mean):
    y = c
    if chain['scene'] == 'rain':
        y = op_rain(y, g)
    elif chain['scene'] == 'haze':
        y = op_haze(y, g, target_mean if not chain['exposure'] else None)
    if chain['blur']:
        y = op_blur(y, g)
    if chain['exposure']:
        y = op_exposure(y, g, target_mean)
    if chain['noise']:
        y = op_sensor_noise(y, g) if chain['exposure'] else op_noise(y, g)
    return y.clamp(0, 1)


def _chan_mean(t):
    return t.mean(dim=(2, 3), keepdim=True).clamp_min(1e-4)


def apply_chain(c, chain, g, target_mean=None, mode='fit'):
    """Degrade one clean image (1,3,H,W) with ``chain`` in physical order.

    ``target_mean`` (1,3,1,1): the target patch's channel means, or None for no exposure matching.
    ``mode``: 'fit' or 'clean' (module docstring). Returns (degraded, clean target)."""
    changes_exposure = chain['exposure'] or chain['scene'] == 'haze'
    if target_mean is None or changes_exposure:
        return _run_chain(c, chain, g, target_mean), c
    k = (target_mean / _chan_mean(c)).clamp(0.1, 10.0)
    if mode == 'fit':
        # same random draws on every pass, so only the scene gain changes. The output mean is (up to clipping)
        # affine in k (blur, noise: proportional; rain screen-blend: A + B*k), so one secant step after a
        # ratio step solves it; the gain stays in [0.1, 10] (a target darker than the rain light alone is
        # matched as closely as that allows)
        state = g.get_state()

        def out_mean(kk):
            y = _run_chain((kk * c).clamp(0, 1), chain, g, target_mean)
            g.set_state(state)
            return _chan_mean(y)

        k0, m0 = k, out_mean(k)
        k1 = (k0 * target_mean / m0).clamp(0.1, 10.0)
        m1 = out_mean(k1)
        d = m1 - m0
        safe = d.abs() > 1e-6
        k = torch.where(safe, k1 + (target_mean - m1) * (k1 - k0) / torch.where(safe, d, torch.ones_like(d)), k1)
        k = k.clamp(0.1, 10.0)
    elif mode != 'clean':
        raise ValueError(f"unknown exposure mode {mode!r}")
    c = (k * c).clamp(0, 1)
    return _run_chain(c, chain, g, target_mean), c


def parse_families(spec, task=None):
    """'oracle' (the target's task), 'all' (one family per pair, uniform), 'compose' (random multi-primitive
    chains) or a '+'-joined list of family names. Returns a list of family names or ['compose']."""
    if spec == 'oracle':
        if task not in TASK_TO_FAMILY:
            raise ValueError(f"--families oracle needs a registry target with a known task, got task={task!r}")
        return [TASK_TO_FAMILY[task]]
    if spec == 'all':
        return list(FAMILIES)
    if spec == 'compose':
        return ['compose']
    fams = [f for f in spec.split('+') if f]
    bad = [f for f in fams if f not in FAMILY_CHAINS]
    if bad or not fams:
        raise ValueError(f"unknown families {bad}; choose from {FAMILIES}, 'oracle', 'all' or 'compose'")
    return fams


def degrade_batch(c, families, g, target_mean=None, mode='fit'):
    """Degrade each image of ``c`` (B,3,H,W). ``families``: list from ``parse_families``; ``mode``: exposure
    matching ('fit' / 'clean', used when ``target_mean`` is given). Returns (degraded, clean targets, labels)."""
    ys, cs, used = [], [], []
    for i in range(c.shape[0]):
        if families == ['compose']:
            chain = random_chain(g)
        else:
            chain = FAMILY_CHAINS[families[int(torch.randint(len(families), (1,), generator=g))]]
        y, ci = apply_chain(c[i:i + 1], chain, g, target_mean, mode)
        ys.append(y)
        cs.append(ci)
        used.append(chain_label(chain))
    return torch.cat(ys, 0), torch.cat(cs, 0), used
