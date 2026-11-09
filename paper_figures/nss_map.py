"""Fig. 2 (right), option B: the degradation shift and our remedy, drawn in natural-scene-statistics (NSS) space.

    python paper_figures/nss_map.py --source_set SOTS_Outdoor --target OHAZY --setting 3task

What is drawn (no numbers on the plot; the axes are a 2-D projection of the 32 NSS features the method uses):
  Source images   crops of the source's own test set (the training degradation; e.g. SOTS-Outdoor for haze),
                  filled orange contours
  Target images   crops of the shifted test set (e.g. O-HAZY), dark contour lines
  Simulated pairs clean DIV2K crops degraded by the physical simulator, exactly as the method does it: for every
                  clean crop, k = 8 candidate degradations from the source's families with camera-consistent
                  exposure matched to a target crop; the candidate closest to the target Gaussian is kept (ABC).
                  Rejected candidates: small grey dots; kept ones: blue dots.
  Clean images    (--show_clean) the undegraded DIV2K crops, a dashed grey contour
Reading: source and target statistics are apart (the shift); the kept simulated pairs sit on the target, so the
model is adapted on pairs whose degradation matches the target and whose clean image is known.

Two steps, one command:
  compute  (once per pair, GPU if available, a few minutes) NSS features of every group, saved as
           results/figures/nss_map/<source>_to_<target>.npz, plus example crops (examples/) and the distances of
           every group to the target Gaussian (<source>_to_<target>.md)
  plot     reads the .npz -> <source>_to_<target>.pdf / .png. Re-runs only the plot when the .npz exists
           (--recompute to redo the features).

Arguments:
  --source_set   registry name of the source's own test set (the training degradation): Rain100L, SOTS_Outdoor,
                 BSD68_s25, GoPro, LOL
  --target       registry name of the shifted test set: Rain100H, SOTS_Indoor, DenseHaze, OHAZY, Kodak24_Poisson,
                 HIDE, LOLv2_Syn, DPDD
  --setting      3task | 5task: which families the simulator draws from (3task: noise, rain, haze; 5task: + blur,
                 low light), as for that pretrained model. Use 5task for blur / low-light targets.
  --proj         pca (default; unsupervised, fitted on source + target crops) | shift (x: the direction that best
                 separates source from target crops (Fisher), y: the main remaining variation). Neither uses the
                 simulated crops, so where those land is not forced by the projection.
  --show_clean   also draw the undegraded clean crops
Rarely needed: --n (crops per group, default 256, the method's abc_fit_n), --n_clean (clean crops, default 256),
--k (candidates per clean crop, default 8, the method's abc_k), --seed.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style as S  # noqa: E402

sys.path.insert(0, S.REPO)

GROUP_LABEL = {'source': 'Source degradation', 'target': 'Target images', 'rejected': 'Simulated, rejected',
               'selected': 'Simulated, selected (ours)', 'clean': 'Clean images'}
C_REJ = '#c9c8c3'
C_CLEAN = '#9a9994'


# ---------------------------------------------------------------------------- #
# compute (torch; the method's own code)
# ---------------------------------------------------------------------------- #
def compute(a, npz):
    import torch
    from irtta.config import SETTING_FAMILIES
    from irtta.data import CleanBank, available_names, evenly_spaced, load_registry, make_eval_dataset, resolve_bank, \
        resolve_spec
    from irtta.degrade import FAMILY_CHAINS, apply_chain, draw_chain, parse_families
    from irtta.nss import GaussianModel, nss_features, random_crop_batch

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reg = load_registry()
    g = torch.Generator().manual_seed(a.seed)
    feat = lambda x: nss_features(x.to(dev), color=True).cpu()  # noqa: E731  (the method's 32-d features)

    def set_crops(name, n):
        """n random 192-px crops of the set's degraded (LQ) images, as evaluated (centre crop to x16, max_side)."""
        spec = resolve_spec(name, reg)
        names = available_names(spec)
        names = evenly_spaced(names, a.max_images) if a.max_images and len(names) > a.max_images else None
        ds = make_eval_dataset(spec, base=16, names=names)
        imgs = [ds[i][1] for i in range(len(ds))]
        idx = torch.randint(len(imgs), (n,), generator=g)
        return torch.cat([random_crop_batch(imgs[int(i)][None], a.patch, 1, g) for i in idx], 0)

    print(f"[nss_map] {a.source_set} -> {a.target} ({a.setting}): loading crops on {dev}")
    t_crops = set_crops(a.target, a.n)
    s_crops = set_crops(a.source_set, a.n)
    with torch.no_grad():
        f_t = torch.cat([feat(b) for b in t_crops.split(64)])
        f_s = torch.cat([feat(b) for b in s_crops.split(64)])
        gm = GaussianModel(f_t)                              # N(mu_T, Sigma_T), as in TransferMethod.prepare
        families = parse_families(SETTING_FAMILIES[a.setting])
        bank = CleanBank(resolve_bank(None, reg), max_images=300, min_side=a.patch)
        clean, cand, cand_fam, cand_of = [], [], [], []
        ex = []                                              # (distance, degraded, clean) of kept candidates
        for i in range(a.n_clean):
            c = bank.sample(1, a.patch, g)
            x = t_crops[int(torch.randint(len(t_crops), (1,), generator=g))][None]
            tmean = x.mean(dim=(0, 2, 3), keepdim=True)      # camera-consistent exposure to a target patch
            ys, cs, fs = [], [], []
            for _ in range(a.k):
                ch = draw_chain(families, g)
                y, ci = apply_chain(c, ch, g, tmean)
                ys.append(y)
                cs.append(ci)
                fs.append(next(f for f, v in FAMILY_CHAINS.items() if v is ch))
            y = torch.cat(ys, 0)
            clean.append(c)
            cand.append(feat(y))
            cand_fam += fs
            cand_of += [i] * a.k
            d = gm.distance(cand[-1])
            j = int(torch.argmin(d))
            ex.append((float(d[j]), y[j], cs[j]))
        f_c = torch.cat([feat(b) for b in torch.cat(clean).split(64)])
        f_k = torch.cat(cand)
        d_k = gm.distance(f_k).numpy()
    cand_of = np.array(cand_of)
    sel = np.zeros(len(d_k), bool)
    for i in range(a.n_clean):
        idx = np.where(cand_of == i)[0]
        sel[idx[np.argmin(d_k[idx])]] = True
    np.savez(npz, source=f_s.numpy(), target=f_t.numpy(), clean=f_c.numpy(), cand=f_k.numpy(), selected=sel,
             family=np.array(cand_fam), d_cand=d_k, d_source=gm.distance(f_s).numpy(),
             d_target=gm.distance(f_t).numpy(), d_clean=gm.distance(f_c).numpy(),
             meta=json.dumps(vars(a)))
    # example crops for the illustration (outside the plot): kept pairs closest to the target, typical crops
    exd = os.path.join(os.path.dirname(npz), 'examples', os.path.basename(npz)[:-4])
    os.makedirs(exd, exist_ok=True)
    from PIL import Image

    def save(t, path):
        Image.fromarray((t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(path)
    for r, (_, y, c) in enumerate(sorted(ex, key=lambda e: e[0])[:4]):
        save(y, os.path.join(exd, f'selected{r}_degraded.png'))
        save(c[0], os.path.join(exd, f'selected{r}_clean.png'))
    for name, crops, f in (('target', t_crops, f_t), ('source', s_crops, f_s)):
        dmed = np.linalg.norm(f.numpy() - f.numpy().mean(0), axis=1)
        for r, i in enumerate(np.argsort(dmed)[:2]):     # crops nearest the group's mean statistics
            save(crops[int(i)], os.path.join(exd, f'{name}{r}.png'))
    print(f"[nss_map] features -> {npz}; example crops -> {exd}/")


# ---------------------------------------------------------------------------- #
# plot (numpy / matplotlib only)
# ---------------------------------------------------------------------------- #
def projection(Z, src, tgt, kind):
    """2-D linear map fitted on standardised source + target features only -> (W (D, 2), mean, scale)."""
    ref = np.concatenate([Z[src], Z[tgt]])
    mu, sd = ref.mean(0), ref.std(0) + 1e-8
    R = (ref - mu) / sd
    _, _, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
    if kind == 'pca':
        return Vt[:2].T, mu, sd
    A, B = (Z[src] - mu) / sd, (Z[tgt] - mu) / sd   # Fisher direction source vs target, ridge-regularised
    Sw = np.cov(A.T) + np.cov(B.T)
    w = np.linalg.solve(Sw + 1e-2 * np.trace(Sw) / len(Sw) * np.eye(len(Sw)), B.mean(0) - A.mean(0))
    w /= np.linalg.norm(w)
    P = R - R.mean(0)
    P = P - np.outer(P @ w, w)                          # the main variation orthogonal to it
    _, _, Vt2 = np.linalg.svd(P, full_matrices=False)
    return np.stack([w, Vt2[0]], 1), mu, sd


def hdr_levels(kde, pts, masses):
    """Density thresholds enclosing the given probability masses (highest-density regions)."""
    dens = kde(pts.T)
    return sorted(np.quantile(dens, 1 - m) for m in masses)


def plot(a, npz, out_base):
    import matplotlib.pyplot as plt
    S.setup()
    D = np.load(npz)
    groups = {k: D[k] for k in ('source', 'target', 'clean')}
    sel = D['selected']
    groups['rejected'], groups['selected'] = D['cand'][~sel], D['cand'][sel]
    Z = np.concatenate([groups[k] for k in groups])
    off, idx = 0, {}
    for k, v in groups.items():
        idx[k] = np.arange(off, off + len(v))
        off += len(v)
    W, mu, sd = projection(Z, idx['source'], idx['target'], a.proj)
    P = ((Z - mu) / sd) @ W
    if np.median(P[idx['target'], 0]) < np.median(P[idx['source'], 0]):   # source left, target right
        P[:, 0] *= -1
    pts = {k: P[i] for k, i in idx.items()}

    fig, ax = plt.subplots(figsize=(S.COL_W, S.COL_W * 0.78))
    keep = np.concatenate([pts['source'], pts['target'], pts['selected']] + ([pts['clean']] if a.show_clean else []))
    lo, hi = np.percentile(keep, 1, axis=0), np.percentile(keep, 99, axis=0)
    pad = (hi - lo) * 0.12
    lo, hi = lo - pad, hi + pad
    ax.scatter(*pts['rejected'].T, s=2.5, color=C_REJ, linewidth=0, zorder=1, rasterized=True)
    try:
        from scipy.stats import gaussian_kde
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
        grid = np.stack([gx.ravel(), gy.ravel()])

        def contour(k, **kw):
            kde = gaussian_kde(pts[k].T)
            lev = hdr_levels(kde, pts[k], (0.9, 0.5))
            return kde(grid).reshape(gx.shape), lev
        z, lev = contour('source')
        ax.contourf(gx, gy, z, levels=[lev[0], z.max() + 1], colors=[S.C_DCTTA], alpha=0.18, zorder=2)
        ax.contourf(gx, gy, z, levels=[lev[1], z.max() + 1], colors=[S.C_DCTTA], alpha=0.22, zorder=2)
        z, lev = contour('target')
        ax.contour(gx, gy, z, levels=lev, colors=[S.C_TARGET], linewidths=[0.7, 1.1], zorder=4)
        if a.show_clean:
            z, lev = contour('clean')
            ax.contour(gx, gy, z, levels=[lev[0]], colors=[C_CLEAN], linewidths=0.8, linestyles='--', zorder=2)
    except ImportError:   # no scipy: plain points
        ax.scatter(*pts['source'].T, s=4, color=S.C_DCTTA, alpha=0.5, linewidth=0, zorder=2)
        ax.scatter(*pts['target'].T, s=4, color=S.C_TARGET, alpha=0.5, linewidth=0, zorder=4)
    ax.scatter(*pts['selected'].T, s=6, color=S.C_OURS, edgecolor='white', linewidth=0.25, zorder=3)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(S.GRID)
    ax.set_xlabel('NSS projection 1' if a.proj == 'pca' else 'source → target statistics', color=S.INK2)
    ax.set_ylabel('NSS projection 2', color=S.INK2)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(facecolor=S.C_DCTTA, alpha=0.4, edgecolor='none', label=GROUP_LABEL['source']),
         Line2D([], [], color=S.C_TARGET, linewidth=1.1, label=GROUP_LABEL['target']),
         Line2D([], [], marker='o', linestyle='', markersize=3, color=C_REJ, label=GROUP_LABEL['rejected']),
         Line2D([], [], marker='o', linestyle='', markersize=3.5, color=S.C_OURS, label=GROUP_LABEL['selected'])]
    if a.show_clean:
        h.append(Line2D([], [], color=C_CLEAN, linewidth=0.8, linestyle='--', label=GROUP_LABEL['clean']))
    ax.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6.5,
              handlelength=1.2, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.2)
    fig.savefig(out_base + '.pdf', dpi=400)
    fig.savefig(out_base + '.png', dpi=300)
    plt.close(fig)

    # numbers for the caption (not in the plot): distances to the target Gaussian (Mahalanobis^2 / D, ~1 = typical
    # target crop) and the families of the kept candidates
    fam, F = D['family'], {}
    for f in fam[sel]:
        F[f] = F.get(f, 0) + 1
    md = [f"# NSS map {os.path.basename(out_base)}\n",
          f"Projection: {a.proj}. Median distance to the target Gaussian (Mahalanobis^2 / D; ~1 = a typical target "
          f"crop):\n", "| group | n | median | q90 |", "|---|---|---|---|"]
    for k, d in (('target', D['d_target']), ('source', D['d_source']), ('clean', D['d_clean']),
                 ('rejected', D['d_cand'][~sel]), ('selected', D['d_cand'][sel])):
        md.append(f"| {GROUP_LABEL[k]} | {len(d)} | {np.median(d):.2f} | {np.quantile(d, 0.9):.2f} |")
    md += ["", "Families of the kept candidates: " + ", ".join(f"{k} {v / sel.sum():.0%}"
                                                            for k, v in sorted(F.items(), key=lambda t: -t[1]))]
    with open(out_base + '.md', 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source_set', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--setting', default='3task', choices=['3task', '5task'])
    p.add_argument('--proj', default='pca', choices=['pca', 'shift'])
    p.add_argument('--show_clean', action='store_true')
    p.add_argument('--recompute', action='store_true')
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--n_clean', type=int, default=256)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--seed', type=int, default=23)
    p.add_argument('--patch', type=int, default=192, help=argparse.SUPPRESS)
    p.add_argument('--max_images', type=int, default=100, help=argparse.SUPPRESS)
    p.add_argument('--out', default=os.path.join(S.REPO, 'results', 'figures', 'nss_map'), help=argparse.SUPPRESS)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stem = f"{a.source_set}_to_{a.target}"
    npz = os.path.join(a.out, stem + '.npz')
    if a.recompute or not os.path.exists(npz):
        compute(a, npz)
    base = os.path.join(a.out, stem + ('' if a.proj == 'pca' else '_shift') + ('_clean' if a.show_clean else ''))
    plot(a, npz, base)
    print(f"-> {base}.pdf")


if __name__ == '__main__':
    main()
