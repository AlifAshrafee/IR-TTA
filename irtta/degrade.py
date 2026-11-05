"""Physical degradation library for test-time adaptation: primitives composed in physical order.

The restorer is adapted on synthetic pairs ``(op(c), c)`` built from *clean* content ``c`` (a generic clean
corpus, DIV2K), so the supervision target is genuinely clean and never comes from the model's own predictions.

Primitives, always applied in this order (each can be off):

| stage     | primitive | model                                                                                 |
|-----------|-----------|---------------------------------------------------------------------------------------|
| scene     | rain      | sparse seeds smeared by an oriented line kernel, screen-blended                      |
|           | haze      | scattering c*t + A*(1-t), t = exp(-beta*d), smooth random depth with a sky ramp       |
| optics    | blur      | non-negative kernel summing to 1 (mean-preserving): Gaussian or camera-shake, <=31 px |
| exposure  | exposure  | s * cast * y^gamma (under-exposure)                                                   |
| sensor    | noise     | Poisson-Gaussian, var = a*y + sigma^2                                                 |

A *chain* is a dict ``{'scene': None|'rain'|'haze', 'blur': bool, 'exposure': bool, 'noise': bool}``. The
families are blur={blur}, noise={noise}, rain={scene: rain}, haze={scene: haze}, lowlight={exposure, noise}; a
source model is adapted with the families it was trained on (3-task: noise, rain, haze; 5-task: all five).

Camera-consistent exposure (``target_mean`` = the channel means of the current target patch). A camera's exposure
gain multiplies everything that reaches the sensor, so for a scene (rain, haze), optics (blur) or sensor (noise)
degradation the observed image and the clean reference of the same capture share ONE gain, chosen so the degraded
image matches the target patch's brightness. Under-exposure *is* a change of that gain, so an exposure chain solves
its own gain to the target and the clean image keeps its natural exposure. For a target patch that the frozen
source judges under-exposed (``dark``, see ``irtta/methods.py``), the clean image instead takes the brightness the
source gives that patch (``clean_mean``): how bright the clean version of a dark image should be cannot be read off
the dark image, so the pairs teach what they can identify (noise, blur, ...) and leave the exposure where the
pretrained model puts it.

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


def op_haze(c, g):
    _, _, H, W = c.shape
    field = F.interpolate(torch.rand(1, 1, 4, 4, generator=g), size=(H, W), mode='bicubic', align_corners=False)
    field = (field - field.amin()) / (field.amax() - field.amin()).clamp_min(1e-6)
    ramp = torch.linspace(1, 0, H).view(1, 1, H, 1).expand(1, 1, H, W)
    w = _u(g, 0.0, 0.7)
    depth = (0.2 + 0.8 * (w * ramp + (1 - w) * field)).to(c)
    t = torch.exp(-_u(g, 0.5, 2.5) * depth)
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


def _run_chain(c, chain, g, target_mean=None):
    """``target_mean`` is used only by the exposure primitive (its gain is solved to the target)."""
    y = c
    if chain['scene'] == 'rain':
        y = op_rain(y, g)
    elif chain['scene'] == 'haze':
        y = op_haze(y, g)
    if chain['blur']:
        y = op_blur(y, g)
    if chain['exposure']:
        y = op_exposure(y, g, target_mean)
    if chain['noise']:
        y = op_sensor_noise(y, g) if chain['exposure'] else op_noise(y, g)
    return y.clamp(0, 1)


def _chan_mean(t):
    return t.mean(dim=(2, 3), keepdim=True).clamp_min(1e-4)


def _to_mean(c, m):
    return (c * (m / _chan_mean(c))).clamp(0, 1)


def apply_chain(c, chain, g, target_mean=None, clean_mean=None, dark=False):
    """Degrade one clean image (1,3,H,W) with ``chain`` in physical order -> (degraded, clean target).

    ``target_mean`` (1,3,1,1): the target patch's channel means (camera-consistent exposure, module docstring), or
    None for no exposure matching. ``dark``: the target patch is under-exposed; ``clean_mean`` (1,3,1,1): the
    brightness the clean image then takes (the frozen source's output on that patch)."""
    if target_mean is None:
        return _run_chain(c, chain, g), c
    if chain['exposure']:   # under-exposure: the chain's own gain matches the target, the clean image is natural
        y = _run_chain(c, chain, g, target_mean)
        return y, (c if clean_mean is None else _to_mean(c, clean_mean))
    y = _run_chain(c, chain, g)
    a = (target_mean / _chan_mean(y)).clamp(0.1, 10.0)   # one camera gain for the capture and its clean reference
    if dark:
        return (a * y).clamp(0, 1), (c.clamp(0, 1) if clean_mean is None else _to_mean(c, clean_mean))
    return (a * y).clamp(0, 1), (a * c).clamp(0, 1)


def parse_families(spec, task=None):
    """'all' (the five families), 'oracle' (only the target's own family: the oracle analysis) or a '+'-joined list
    of family names -> list of family names."""
    if spec == 'oracle':
        if task not in TASK_TO_FAMILY:
            raise ValueError(f"--families oracle needs a registry target with a known task, got task={task!r}")
        return [TASK_TO_FAMILY[task]]
    if spec == 'all':
        return list(FAMILIES)
    fams = [f for f in spec.split('+') if f]
    bad = [f for f in fams if f not in FAMILY_CHAINS]
    if bad or not fams:
        raise ValueError(f"unknown families {bad}; choose from {FAMILIES}, 'all' or 'oracle'")
    return fams


def draw_chain(families, g):
    """A single-family chain, the family drawn uniformly."""
    return FAMILY_CHAINS[families[int(torch.randint(len(families), (1,), generator=g))]]


def candidate_chains(families, g, k, stratified=False):
    """The ``k`` candidate chains for one clean image: ``k`` independent uniform family draws, or (``stratified``)
    every family ceil(k / |families|) times, so each family is always among the candidates (3 families, k = 8 -> 9
    candidates; 5 families -> 10). Uniform draws miss a given family with probability (1 - 1/|F|)^k.

    A generator: each uniform family draw happens right before that candidate is degraded, the RNG order of the
    original loop (runs with ``stratified=False`` are bit-identical to before)."""
    if not stratified:
        for _ in range(k):
            yield draw_chain(families, g)
        return
    m = -(-k // len(families))
    for f in families:
        for _ in range(m):
            yield FAMILY_CHAINS[f]


def degrade_batch(c, families, g, target_mean=None, select=None, k=1, clean_mean=None, dark=False,
                  stratified=False):
    """Degrade each image of ``c`` (B,3,H,W) -> (degraded, clean targets, family labels, ABC scores or None).

    ``select`` + ``k`` > 1 (ABC pair selection): draw ``k`` candidate degradations per image (``candidate_chains``)
    and keep the one with the lowest ``select(candidates (k,3,H,W), clean image (1,3,H,W)) -> (k,)`` score, the
    distance of its NSS features to the target set's Gaussian (``irtta/nss.py``)."""
    ys, cs, used, scores = [], [], [], []
    for i in range(c.shape[0]):
        if select is None or k <= 1:
            chain = draw_chain(families, g)
            y, ci = apply_chain(c[i:i + 1], chain, g, target_mean, clean_mean, dark)
            ys.append(y)
            cs.append(ci)
            used.append(chain_label(chain))
            continue
        cand = []
        for ch in candidate_chains(families, g, k, stratified):
            yk, ck = apply_chain(c[i:i + 1], ch, g, target_mean, clean_mean, dark)
            cand.append((ch, yk, ck))
        sc = select(torch.cat([t[1] for t in cand], 0), c[i:i + 1])
        j = int(torch.argsort(sc)[0])
        chain, y, ci = cand[j]
        ys.append(y)
        cs.append(ci)
        used.append(chain_label(chain))
        scores.append(float(sc[j]))
    return torch.cat(ys, 0), torch.cat(cs, 0), used, (scores or None)
