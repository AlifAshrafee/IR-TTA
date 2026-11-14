"""Building blocks for the method figure (fig:method), to be arranged in Lucidchart.

    python paper_figures/method_assets.py --target Rain100H --setting 5task            # everything
    python paper_figures/method_assets.py --target Rain100H --setting 5task --image rain-042 --seed 3
    python paper_figures/method_assets.py --only space                                 # no data, no torch

Everything is written to results/figures/method_assets/<target>_<setting>/ (PNG with a transparent background and SVG):

  images/      real images, produced by the method's own code (irtta/nss.py, irtta/degrade.py)
     target_image.png, target_crop.png   one target image and the 192-px crop used below
     target_mscn.png                     its MSCN coefficients m = (I - mu) / (sigma + C)
     clean.png                           one clean DIV2K crop from the bank
     cand<j>_<family>.png                its J = 8 candidate degradations (random family and parameters, exposure
                                         matched to the target crop), `_accepted` marks the one the method keeps
     cand<j>_mscn.png                    MSCN coefficients of every candidate
     pair_input.png, pair_target.png     the accepted pair: degraded input and its clean target (same exposure gain)
     simulator_<family>.png              one draw of every operator on the same clean crop (for the simulator tiles)
  plots/       drawn from the same numbers
     mscn_hist.*            MSCN histograms: target crop, accepted candidate, the farthest rejected candidate
     strip_target.png, strip_cand<j>.png   the 32-d statistic of a crop as a colour strip, in units of the target
                                         set's standard deviation around its mean (white = equal to the target mean)
     candidates.*           all J candidates in one panel: family, statistic strip, distance, accepted mark
     distance_by_family.*   distance of every candidate of 64 clean crops to the target Gaussian, per family, and
                            the share of accepted pairs per family (how the family is inferred without a classifier)
  schematic/   stat_space.*   illustration (not data) of the selection in statistic space
  math/        (only with `--only math`) the formulas of the method as images; not part of the figure
  summary.md   the numbers behind the assets (families, distances, accepted shares)

Arguments
  --target    registry name of the target set (Rain100H, SOTS_Indoor, DenseHaze, OHAZY, Kodak24_Poisson, HIDE,
              LOLv2_Syn, DPDD)
  --setting   3task | 5task: which operators the simulator has (3task: noise, rain, haze; 5task: + blur, low light)
  --image     file name (with or without extension) of the target image to show; default: the image whose crop is
              closest to the target set's mean statistics
  --seed      changes the crops, the clean image and the candidate draws; try a few and keep the clearest
  --pick      use the n-th clean crop whose candidates cover at least --min_families families (default 0, the first)
  --min_families  default: 4 (5task) or 3 (3task), so the candidate strip shows several operators
  --only      any of: data plots space math (default: data plots space; formulas are not drawn unless asked).
              `plots` re-draws from the cached numbers without torch.
  --recompute redo the data step even if its cache exists
  --no_labels schematic/ without text, to label it in Lucidchart

The data step uses the same procedure as the method: a Gaussian fitted to the 32-d statistics of 256 random 192-px
target crops; for a clean crop, J = 8 candidates with a uniformly drawn family and prior-drawn parameters, exposure
matched to a target crop; the candidate with the smallest Mahalanobis distance is accepted. The under-exposure rule
(which needs the pretrained model) is not applied here; it only acts on dark target patches.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S  # noqa: E402

sys.path.insert(0, S.REPO)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

FAM_LABEL = {'noise': 'Noise', 'rain': 'Rain', 'haze': 'Haze', 'blur': 'Blur', 'lowlight': 'Low light'}
# validated subset of the categorical palette (blue and orange stay reserved for ours / the pretrained model);
# every family also has its own marker, so identity never rests on colour alone
FAM_COLOR = {'noise': '#6250d6', 'rain': '#1baf7a', 'haze': '#eda100', 'blur': '#e87ba4', 'lowlight': '#008300'}
FAM_MARK = {'noise': 's', 'rain': 'o', 'haze': '^', 'blur': 'D', 'lowlight': 'P'}
SETTING = {'3task': ['noise', 'rain', 'haze'], '5task': ['blur', 'noise', 'rain', 'haze', 'lowlight']}
C_ACC, C_REJ, INK, INK2 = '#2a78d6', '#b9b8b3', '#1f1f1d', '#5c5c58'
BINS = np.linspace(-3, 3, 61)


def setup():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Inter', 'Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
        'ytick.major.size': 2.5, 'svg.fonttype': 'path', 'pdf.fonttype': 42, 'savefig.transparent': True,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'mathtext.fontset': 'cm',
    })


def save_fig(fig, base, pdf=False):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    fig.savefig(base + '.svg')
    fig.savefig(base + '.png', dpi=600)
    if pdf:
        fig.savefig(base + '.pdf')
    plt.close(fig)


def save_rgb(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8)).save(path)


def save_mscn(m, path):
    """MSCN coefficients as a grey image: 0 is mid-grey, +-2.5 is white / black."""
    save_rgb(np.repeat((np.clip(m, -2.5, 2.5) / 5 + 0.5)[..., None], 3, 2), path)


# ---------------------------------------------------------------------------- #
# data (torch; the method's own code)
# ---------------------------------------------------------------------------- #
def compute(a, npz):
    import torch
    import torch.nn.functional as F
    from irtta.data import CleanBank, available_names, evenly_spaced, load_registry, make_eval_dataset, resolve_bank, \
        resolve_spec
    from irtta.degrade import FAMILY_CHAINS, apply_chain, draw_chain, parse_families
    from irtta.nss import GaussianModel, _window, luminance, nss_features, random_crop_batch

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reg = load_registry()
    g = torch.Generator().manual_seed(a.seed)
    feat = lambda x: nss_features(x.to(dev), color=True).cpu()  # noqa: E731  (the method's 32-d statistic)
    hwc = lambda t: t.clamp(0, 1).permute(0, 2, 3, 1).numpy().astype(np.float32)  # noqa: E731

    def mscn(x):
        """MSCN coefficients at full resolution, as in irtta/nss.py::_scale_features -> (B, h, w) numpy."""
        y = luminance(x)
        w = _window(x.device, x.dtype)
        mu = F.conv2d(F.pad(y, (3, 3, 3, 3), mode='reflect'), w)
        var = F.conv2d(F.pad(y * y, (3, 3, 3, 3), mode='reflect'), w) - mu * mu
        sig = torch.sqrt(var.clamp_min(0) + 1e-8)
        return ((y - mu) / (sig + 1 / 255))[:, 0, 3:-3, 3:-3].numpy().astype(np.float32)

    # target crops and their Gaussian (TransferMethod.prepare)
    spec = resolve_spec(a.target, reg)
    names = available_names(spec)
    sub = evenly_spaced(names, a.max_images) if a.max_images and len(names) > a.max_images else None
    ds = make_eval_dataset(spec, base=16, names=sub)
    imgs = [ds[i][1] for i in range(len(ds))]
    print(f"[method_assets] {a.target} ({a.setting}): {len(imgs)} target images, statistics on {dev}")
    idx = torch.randint(len(imgs), (a.n,), generator=g)
    t_crops = torch.cat([random_crop_batch(imgs[int(i)][None], a.patch, 1, g) for i in idx], 0)
    with torch.no_grad():
        f_t = torch.cat([feat(b) for b in t_crops.split(64)])
        gm = GaussianModel(f_t)
        d_t = gm.distance(f_t)
        if a.image:   # a chosen image: one random crop of it
            stem = os.path.splitext(a.image)[0]
            hit = [n for n in names if os.path.splitext(n)[0] == stem]
            if not hit:
                raise SystemExit(f"--image {a.image}: no such image in {a.target}")
            t_img = make_eval_dataset(spec, base=16, names=hit)[0][1]
            t_crop = random_crop_batch(t_img[None], a.patch, 1, g)
        else:         # the crop closest to the set's mean statistics
            i = int(torch.argmin(d_t))
            t_img, t_crop = imgs[int(idx[i])], t_crops[i:i + 1]
        f_crop = feat(t_crop)

        # candidates (irtta/degrade.py::degrade_batch, keeping every candidate)
        families = parse_families('+'.join(SETTING[a.setting]))
        bank = CleanBank(resolve_bank(None, reg), max_images=300, min_side=a.patch)
        tmean = t_crop.mean(dim=(0, 2, 3), keepdim=True)   # exposure reference: the shown target crop
        fam_of = lambda ch: next(f for f, v in FAMILY_CHAINS.items() if v is ch)  # noqa: E731

        def draw_patch():
            c = bank.sample(1, a.patch, g)
            ys, cs, fs = [], [], []
            for _ in range(a.k):
                ch = draw_chain(families, g)
                y, ci = apply_chain(c, ch, g, tmean)
                ys.append(y)
                cs.append(ci)
                fs.append(fam_of(ch))
            y = torch.cat(ys, 0)
            f = feat(y)
            return c, y, torch.cat(cs, 0), fs, f, gm.distance(f)

        pool = [draw_patch() for _ in range(a.n_clean)]
        good = [p for p in pool if len(set(p[3])) >= a.min_families]
        if not good:
            raise SystemExit(f"no clean crop among {a.n_clean} has candidates from {a.min_families} families; "
                             f"lower --min_families or change --seed")
        c, y, cs, fs, f, d = good[min(a.pick, len(good) - 1)]
        sim = torch.cat([apply_chain(c, FAMILY_CHAINS[fam], g, tmean)[0] for fam in SETTING[a.setting]], 0)
        pool_d = torch.cat([p[5] for p in pool]).numpy()
        pool_sel = np.zeros(len(pool_d), bool)
        for r, p in enumerate(pool):
            pool_sel[r * a.k + int(torch.argmin(p[5]))] = True
        np.savez(npz, target_image=hwc(t_img[None])[0], target_crop=hwc(t_crop)[0], target_mscn=mscn(t_crop)[0],
                 target_feat=f_crop[0].numpy(), mu=f_t.mean(0).numpy(), sd=f_t.std(0).numpy(), d_target=d_t.numpy(),
                 clean=hwc(c)[0], cand=hwc(y), cand_clean=hwc(cs), cand_fam=np.array(fs), cand_d=d.numpy(),
                 cand_feat=f.numpy(), cand_mscn=mscn(y), sim_fam=np.array(SETTING[a.setting]), sim=hwc(sim),
                 pool_d=pool_d, pool_fam=np.array([x for p in pool for x in p[3]]), pool_sel=pool_sel,
                 meta=json.dumps({k: v for k, v in vars(a).items() if k != 'only'}))
    print(f"[method_assets] numbers cached in {npz}")


def demo_data(a, npz):
    """Synthetic stand-in with the same fields, to preview the plots without data or torch (not for the paper)."""
    from scipy.ndimage import gaussian_filter
    r = np.random.default_rng(a.seed)
    p, J, fams = 96, a.k, SETTING[a.setting]

    def img(noise):
        base = gaussian_filter(r.random((p, p, 3)), (6, 6, 0))
        base = (base - base.min()) / (np.ptp(base) + 1e-6)
        return np.clip(base + noise * r.normal(size=base.shape), 0, 1).astype(np.float32)

    def mscn(x):
        y = x @ np.array([0.299, 0.587, 0.114])
        mu = gaussian_filter(y, 7 / 6)
        sig = np.sqrt(np.clip(gaussian_filter(y * y, 7 / 6) - mu * mu, 0, None) + 1e-8)
        return ((y - mu) / (sig + 1 / 255))[3:-3, 3:-3].astype(np.float32)

    cf = np.array([fams[i % len(fams)] for i in range(J)])
    lvl = {'noise': 0.10, 'rain': 0.04, 'haze': 0.01, 'blur': 0.0, 'lowlight': 0.06}
    cand = np.stack([img(lvl[f]) for f in cf])
    t = img(0.04)
    cd = np.array([{'rain': 1.6}.get(f, 9.0) for f in cf]) * r.uniform(0.7, 1.6, J)
    n_pool = 64 * J
    pf = np.array([fams[i] for i in r.integers(len(fams), size=n_pool)])
    pd = np.array([{'rain': 2.0}.get(f, 12.0) for f in pf]) * r.lognormal(0, 0.5, n_pool)
    ps = np.zeros(n_pool, bool)
    for k in range(64):
        ps[k * J + int(np.argmin(pd[k * J:(k + 1) * J]))] = True
    np.savez(npz, target_image=np.tile(t, (1, 2, 1))[:, :160], target_crop=t, target_mscn=mscn(t),
             target_feat=r.normal(0, 0.6, 32), mu=np.zeros(32), sd=np.ones(32), d_target=r.chisquare(32, 256) / 32,
             clean=img(0.0), cand=cand, cand_clean=np.stack([img(0.0)] * J), cand_fam=cf, cand_d=cd,
             cand_feat=r.normal(0, 1, (J, 32)) * np.sqrt(cd)[:, None], cand_mscn=np.stack([mscn(c) for c in cand]),
             sim_fam=np.array(fams), sim=np.stack([img(lvl[f]) for f in fams]), pool_d=pd, pool_fam=pf, pool_sel=ps,
             meta=json.dumps({'demo': True}))


# ---------------------------------------------------------------------------- #
# images and plots (numpy / matplotlib only)
# ---------------------------------------------------------------------------- #
def export_images(D, out):
    d = os.path.join(out, 'images')
    shutil.rmtree(d, ignore_errors=True)
    j = int(np.argmin(D['cand_d']))
    save_rgb(D['target_image'], os.path.join(d, 'target_image.png'))
    save_rgb(D['target_crop'], os.path.join(d, 'target_crop.png'))
    save_mscn(D['target_mscn'], os.path.join(d, 'target_mscn.png'))
    save_rgb(D['clean'], os.path.join(d, 'clean.png'))
    for k, fam in enumerate(D['cand_fam']):
        save_rgb(D['cand'][k], os.path.join(d, f"cand{k}_{fam}{'_accepted' if k == j else ''}.png"))
        save_mscn(D['cand_mscn'][k], os.path.join(d, f'cand{k}_mscn.png'))
    save_rgb(D['cand'][j], os.path.join(d, 'pair_input.png'))
    save_rgb(D['cand_clean'][j], os.path.join(d, 'pair_target.png'))
    for fam, im in zip(D['sim_fam'], D['sim']):
        save_rgb(im, os.path.join(d, f'simulator_{fam}.png'))


def _hist(m):
    h, _ = np.histogram(np.clip(m, BINS[0], BINS[-1]), bins=BINS, density=True)
    return h


def plot_hist(D, out):
    j, far = int(np.argmin(D['cand_d'])), int(np.argmax(D['cand_d']))
    x = (BINS[:-1] + BINS[1:]) / 2
    fig, ax = plt.subplots(figsize=(2.0, 1.05))
    ht = _hist(D['target_mscn'])
    ax.fill_between(x, ht, color='#d9d8d4', linewidth=0, zorder=1)
    ax.plot(x, ht, color=INK2, linewidth=1.0, zorder=2, label='target crop')
    ax.plot(x, _hist(D['cand_mscn'][far]), color=C_REJ, linewidth=1.0, linestyle=(0, (3, 1.5)), zorder=3,
            label='rejected')
    ax.plot(x, _hist(D['cand_mscn'][j]), color=C_ACC, linewidth=1.4, zorder=4, label='accepted')
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, None)
    ax.set_yticks([])
    ax.set_xticks([-2, 0, 2])
    ax.set_xlabel('MSCN coefficient', labelpad=1)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.1,
              borderpad=0, columnspacing=0.9, handletextpad=0.4)
    save_fig(fig, os.path.join(out, 'plots', 'mscn_hist'))


def _strip(ax, z):
    """32 cells: 15 (scale 1) | 15 (scale 2) | 2 (colour), with a small gap between the groups."""
    xs = np.concatenate([np.arange(15), 15.6 + np.arange(15), 31.2 + np.arange(2)])
    ax.bar(xs, np.ones(len(z)), width=0.94, bottom=0, align='edge', linewidth=0,
           color=plt.get_cmap('PRGn')(np.clip(z, -3, 3) / 6 + 0.5))
    ax.set_xlim(-0.2, 33.4)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def plot_strips(D, out):
    z_t = (D['target_feat'] - D['mu']) / (D['sd'] + 1e-8)
    z_c = (D['cand_feat'] - D['mu']) / (D['sd'] + 1e-8)
    for name, z in [('strip_target', z_t)] + [(f'strip_cand{k}', z_c[k]) for k in range(len(z_c))]:
        fig = plt.figure(figsize=(2.2, 0.14))
        _strip(fig.add_axes([0, 0, 1, 1]), z)
        plt.rcParams['savefig.bbox'] = None
        save_fig(fig, os.path.join(out, 'plots', name))
        plt.rcParams['savefig.bbox'] = 'tight'
    # all candidates in one panel, nearest first
    order = np.argsort(D['cand_d'])
    n = len(order)
    fig = plt.figure(figsize=(3.1, 0.2 * (n + 2.4)))
    H = n + 2.4
    for r, k in enumerate([None] + list(order)):
        y0 = 1 - (r + 2.0 + (0.35 if r else 0)) / H
        ax = fig.add_axes([0.26, y0, 0.56, 0.78 / H])
        _strip(ax, z_t if k is None else z_c[k])
        yc = y0 + 0.39 / H
        if k is None:
            fig.text(0.235, yc, 'target crop', ha='right', va='center', fontsize=7.5)
            continue
        fam = str(D['cand_fam'][k])
        acc = r == 1
        fig.text(0.235, yc, FAM_LABEL[fam], ha='right', va='center', fontsize=7.5)
        mk = fig.add_axes([0.005, y0, 0.05, 0.78 / H])
        mk.plot(0, 0, marker=FAM_MARK[fam], color=FAM_COLOR[fam], markersize=4.5, markeredgewidth=0)
        mk.set_axis_off()
        fig.text(0.835, yc, f"{D['cand_d'][k]:.1f}", ha='left', va='center', fontsize=7.5,
                 color=INK if acc else INK2)
        if acc:
            fig.text(0.925, yc, 'accepted', ha='left', va='center', fontsize=7, color=C_ACC)
            fig.add_artist(plt.Rectangle((0.255, y0 - 0.06 / H), 0.57, 0.90 / H, fill=False, edgecolor=C_ACC,
                                         linewidth=1.0, transform=fig.transFigure))
    fig.text(0.54, 1 - 0.55 / H, 'statistic, relative to the target set', ha='center', va='center', fontsize=7,
             color=INK2)
    fig.text(0.835, 1 - 0.55 / H, 'distance', ha='left', va='center', fontsize=7, color=INK2)
    save_fig(fig, os.path.join(out, 'plots', 'candidates'), pdf=True)


def plot_distance(D, out):
    fams = [f for f in D['sim_fam']]
    d, pf, sel = D['pool_d'], D['pool_fam'], D['pool_sel']
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(3.2, 0.26 * len(fams) + 0.5))
    ax = fig.add_axes([0.2, 0.3, 0.6, 0.66])
    share = {}
    for r, fam in enumerate(fams):
        m = pf == fam
        share[fam] = float((m & sel).sum()) / max(int(sel.sum()), 1)
        for mask, col, s, z in ((m & ~sel, C_REJ, 5, 2), (m & sel, C_ACC, 9, 3)):
            ax.scatter(d[mask], r + rng.uniform(-0.28, 0.28, int(mask.sum())), s=s, color=col, linewidth=0, zorder=z)
        ax.text(1.03, r, f"{100 * share[fam]:.0f}%", transform=ax.get_yaxis_transform(), va='center', ha='left',
                fontsize=7.5)
    ax.axvline(1.0, color=INK2, linewidth=0.6, zorder=1)
    ax.set_xscale('log')
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([FAM_LABEL[f] for f in fams], fontsize=7.5, color=INK)
    ax.set_ylim(len(fams) - 0.5, -0.5)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('distance to the target statistics (1 = a typical target crop)', fontsize=7, labelpad=2)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.text(1.03, 1.0, 'accepted', transform=ax.transAxes, va='bottom', ha='left', fontsize=7, color=C_ACC)
    save_fig(fig, os.path.join(out, 'plots', 'distance_by_family'), pdf=True)
    return share


def write_summary(D, share, out):
    order = np.argsort(D['cand_d'])
    L = ["# Method-figure assets: numbers", '', "## Candidates of the shown clean crop (nearest first)", '',
         "| Candidate | Family | Distance | |", "|---|---|---|---|"]
    L += [f"| cand{k} | {FAM_LABEL[str(D['cand_fam'][k])]} | {D['cand_d'][k]:.2f} | {'accepted' if r == 0 else ''} |"
          for r, k in enumerate(order)]
    L += ['', f"Target crops' own distance: median {np.median(D['d_target']):.2f} (about 1 by construction).", '',
          "## Accepted pairs per family (64 clean crops)", '', "| Family | Share of accepted pairs |", "|---|---|"]
    L += [f"| {FAM_LABEL[f]} | {100 * v:.0f}% |" for f, v in share.items()]
    with open(os.path.join(out, 'summary.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


# ---------------------------------------------------------------------------- #
# schematic of the selection in statistic space (an illustration, not data)
# ---------------------------------------------------------------------------- #
def plot_space(a, out):
    fams = SETTING[a.setting]
    r = np.random.default_rng(4)
    A = np.array([[1.0, 0.55], [0.0, 0.5]])                      # target covariance = A A^T
    cov = A @ A.T
    centre = {'rain': (0.9, 0.5), 'haze': (-3.5, 2.2), 'noise': (3.6, -1.9), 'blur': (-3.3, -2.3),
              'lowlight': (3.3, 2.5)}
    if a.near != 'rain':                                         # put the chosen family on the target
        centre[a.near], centre['rain'] = centre['rain'], centre[a.near]
    fig = plt.figure(figsize=(2.9, 2.25))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ev, evec = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))

    def ell(k, **kw):
        return Ellipse((0, 0), 2 * k * np.sqrt(ev[1]), 2 * k * np.sqrt(ev[0]), angle=ang, **kw)
    ax.add_patch(ell(2, facecolor='#ecebe7', edgecolor='none', zorder=1))
    ax.add_patch(ell(1, facecolor='#d9d8d4', edgecolor='none', zorder=1))
    for k in (1, 2):
        ax.add_patch(ell(k, facecolor='none', edgecolor=INK2, linewidth=0.7, zorder=2))
    tp = r.normal(size=(45, 2)) @ A.T
    ax.scatter(tp[:, 0], tp[:, 1], s=4, color=INK2, linewidth=0, zorder=3)
    ax.plot(0, 0, marker='+', color=INK, markersize=6, markeredgewidth=1.0, zorder=6)
    for f in fams:                                               # prior draws of every operator
        spread = np.array([[0.6, 0.2], [0.1, 0.45]]) * (0.8 if f == a.near else 1.0)
        p = np.array(centre[f]) + r.normal(size=(26, 2)) @ spread.T
        ax.scatter(p[:, 0], p[:, 1], s=9, marker=FAM_MARK[f], color=FAM_COLOR[f], alpha=0.35, linewidth=0, zorder=4)
    # the J candidates of one clean crop: fixed offsets from their family centres
    offs = [(a.near, (0.25, 0.1)), (a.near, (1.0, -0.75))] + [(f, o) for f, o in zip(
        [f for f in fams if f != a.near] * 2, [(0.5, -0.3), (-0.4, 0.45), (0.3, 0.55), (-0.5, -0.2), (-0.7, 0.5),
                                               (0.6, 0.6)])]
    cand = [(f, np.array(centre[f]) + np.array(o)) for f, o in offs[:a.k]]
    prec = np.linalg.inv(cov)
    dist = [float(p @ prec @ p) for _, p in cand]
    j = int(np.argmin(dist))
    for i, (f, p) in enumerate(cand):
        ax.plot([0, p[0]], [0, p[1]], color=C_ACC if i == j else C_REJ, linewidth=1.3 if i == j else 0.6,
                linestyle='-' if i == j else (0, (2, 2)), zorder=5)
        ax.scatter(*p, s=34, marker=FAM_MARK[f], color=FAM_COLOR[f], edgecolor='white', linewidth=0.7, zorder=7)
    pj = cand[j][1]
    ax.add_patch(ell(np.sqrt(dist[j]), facecolor='none', edgecolor=C_ACC, linewidth=0.9, linestyle=(0, (4, 2)),
                     zorder=5))
    ax.scatter(*pj, s=150, facecolor='none', edgecolor=C_ACC, linewidth=1.4, zorder=8)
    if not a.no_labels:
        for f in fams:
            cx, cy = centre[f]
            if f == a.near:
                ax.text(cx + 0.2, cy + 1.3, FAM_LABEL[f], ha='center', va='center', fontsize=7.5, color=INK)
            else:
                ax.text(cx, cy + (1.2 if cy > 0 else -1.2), FAM_LABEL[f], ha='center', va='center', fontsize=7.5,
                        color=INK)
        ax.text(-0.2, -1.75, 'Target set', ha='center', va='center', fontsize=7.5, color=INK2)
    ax.set_xlim(-5.4, 5.4)
    ax.set_ylim(-4.1, 4.1)
    ax.set_aspect('equal')
    ax.set_axis_off()
    plt.rcParams['savefig.bbox'] = None
    save_fig(fig, os.path.join(out, 'schematic', 'stat_space'), pdf=True)
    plt.rcParams['savefig.bbox'] = 'tight'


# ---------------------------------------------------------------------------- #
# formulas (the paper's notation)
# ---------------------------------------------------------------------------- #
B = r'\boldsymbol'
MATH = {
    'op_noise': r'\mathbf{y}=\mathbf{x}+\mathbf{n},\;\; \mathbf{n}\sim\mathcal{N}(\mathbf{0},\,\lambda\mathbf{x}+\sigma^{2})',
    'op_rain': r'\mathbf{y}=\mathbf{1}-(\mathbf{1}-\mathbf{x})\odot(\mathbf{1}-\mathbf{r})',
    'op_haze': r'\mathbf{y}=\mathbf{x}\odot\mathbf{t}+A\,(\mathbf{1}-\mathbf{t}),\;\; \mathbf{t}=e^{-\beta\mathbf{d}}',
    'op_blur': r'\mathbf{y}=\mathbf{k}*\mathbf{x}',
    'op_lowlight': r'\mathbf{y}=\kappa\,' + B + r'{\omega}\odot\mathbf{x}^{\gamma}+\mathbf{n}',
    'draw': r'\tau_j\sim\mathcal{U}(\mathcal{F}^{s}),\;\; ' + B + r'{\phi}_j\sim\pi_{\tau_j}',
    'candidate': r'\hat{\mathbf{y}}_j=\mathcal{D}_{\tau_j}(\mathbf{x};' + B + r'{\phi}_j)',
    'mscn': r'm=\frac{I-\mu}{\sigma+C}',
    'statistic': r'\mathbf{s}(\mathbf{y})\in\mathbb{R}^{32}',
    'gaussian': r'\mathcal{N}(' + B + r'{\mu}_{\mathcal{T}},\,' + B + r'{\Sigma}_{\mathcal{T}})',
    'abc': r'j^{\star}=ARGMIN\;(\mathbf{s}(\hat{\mathbf{y}}_j)-' + B + r'{\mu}_{\mathcal{T}})^{\top}' + B
           + r'{\Sigma}_{\mathcal{T}}^{-1}(\mathbf{s}(\hat{\mathbf{y}}_j)-' + B + r'{\mu}_{\mathcal{T}})',
    'gain': r'\tilde{\mathbf{y}}=' + B + r'{\kappa}\odot\hat{\mathbf{y}},\;\; \tilde{\mathbf{x}}=' + B
            + r'{\kappa}\odot\mathbf{x}',
    'lora': r'\mathbf{W}\leftarrow\mathbf{W}+\mathbf{U}\mathbf{V}',
    'lora_V': r'\mathbf{V}\in\mathbb{R}^{q\times C_{\mathrm{in}}}',
    'lora_U': r'\mathbf{U}\in\mathbb{R}^{C_{\mathrm{out}}\times q},\;\; \mathbf{U}_0=\mathbf{0}',
    'loss': r'\mathcal{L}=\frac{1}{M}\sum_{b=1}^{M}\|f_{' + B + r'{\theta}}(\tilde{\mathbf{y}}_b)-\tilde{\mathbf{x}}_b\|_1',
    'sym_x': r'\mathbf{x}', 'sym_y': r'\mathbf{y}', 'sym_T': r'\mathcal{T}', 'sym_B': r'\mathcal{B}',
    'sym_pair': r'(\tilde{\mathbf{y}},\,\tilde{\mathbf{x}})',
    'sym_f0': r'f_{' + B + r'{\theta}_0}', 'sym_fT': r'f_{' + B + r'{\theta}_{\mathcal{T}}}',
}


def plot_math(a, out):
    """matplotlib's mathtext; bold Greek (\\boldsymbol) needs matplotlib >= 3.8 and is dropped on older versions."""
    import re
    plain = 0
    for name, tex in MATH.items():
        variants = [tex.replace('ARGMIN', r'\underset{j}{\mathrm{arg\,min}}'),
                    re.sub(r'\\boldsymbol\{([^{}]*)\}', r'\1', tex).replace('ARGMIN', r'\mathrm{arg\,min}_{j}')]
        for i, s in enumerate(variants):
            fig = plt.figure(figsize=(0.1, 0.1))
            fig.text(0, 0, f'${s}$', fontsize=10, color=INK)
            try:
                save_fig(fig, os.path.join(out, 'math', name))
                plain += i
                break
            except Exception:
                plt.close(fig)
                if i:
                    raise
    if plain:
        print(f"[math] {plain} formulas drawn without bold Greek (matplotlib < 3.8); `pip install -U matplotlib` "
              f"gives the paper's notation")


# ---------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', default='Rain100H')
    p.add_argument('--setting', default='5task', choices=['3task', '5task'])
    p.add_argument('--image', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pick', type=int, default=0)
    p.add_argument('--min_families', type=int, default=None)
    p.add_argument('--only', nargs='+', default=['data', 'plots', 'space'],
                   choices=['data', 'plots', 'space', 'math'])
    p.add_argument('--recompute', action='store_true')
    p.add_argument('--no_labels', action='store_true')
    p.add_argument('--near', default=None, help=argparse.SUPPRESS)       # schematic: the family drawn on the target
    p.add_argument('--k', type=int, default=8, help=argparse.SUPPRESS)        # the method's J
    p.add_argument('--n', type=int, default=256, help=argparse.SUPPRESS)      # target crops (abc_fit_n)
    p.add_argument('--n_clean', type=int, default=64, help=argparse.SUPPRESS)
    p.add_argument('--patch', type=int, default=192, help=argparse.SUPPRESS)
    p.add_argument('--max_images', type=int, default=100, help=argparse.SUPPRESS)
    p.add_argument('--demo', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--out', default=os.path.join(S.FIGS, 'method_assets'), help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.min_families is None:
        a.min_families = 4 if a.setting == '5task' else 3
    if a.near is None:
        task = {'Rain100H': 'rain', 'SOTS_Indoor': 'haze', 'DenseHaze': 'haze', 'OHAZY': 'haze',
                'Kodak24_Poisson': 'noise', 'HIDE': 'blur', 'DPDD': 'blur', 'LOLv2_Syn': 'lowlight'}
        a.near = task.get(a.target, 'rain')
        if a.near not in SETTING[a.setting]:
            a.near = SETTING[a.setting][0]
    setup()
    out = os.path.join(a.out, f"{a.target}_{a.setting}" + ('_demo' if a.demo else ''))
    os.makedirs(out, exist_ok=True)
    npz = os.path.join(out, 'numbers.npz')
    if 'data' in a.only and (a.recompute or a.demo or not os.path.exists(npz)):
        (demo_data if a.demo else compute)(a, npz)
    if {'data', 'plots'} & set(a.only):
        if not os.path.exists(npz):
            raise SystemExit(f"no {npz}: run without --only first")
        D = np.load(npz)
        export_images(D, out)
        plot_hist(D, out)
        plot_strips(D, out)
        write_summary(D, plot_distance(D, out), out)
    if 'space' in a.only:
        plot_space(a, out)
    if 'math' in a.only:
        plot_math(a, out)
    print(f"-> {out}/")


if __name__ == '__main__':
    main()
