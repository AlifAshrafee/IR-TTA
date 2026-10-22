"""CPU self-tests for the harness. Run before any GPU experiment:

    CUDA_VISIBLE_DEVICES= python tests/test_harness.py          # or: python -m pytest tests/test_harness.py -q

Uses a randomly initialised PromptIR saved as a fake Lightning checkpoint and three synthetic 96x96
images, so it needs no data, no checkpoints and no downloads (VGG is disabled with
--perceptual_weight 0). Takes a few minutes on CPU (the RDDM generator is the slow part).

What is checked:
  * zero-init LoRA reproduces the source output; subset construction leaves the global RNG alone
  * subset contents (LN / LoRA ranks / DCTTA's 'full' rule)
  * L2 anchor: zero at theta_0, closed-form value (incl. LoRA delta weights)
  * degradation operators: shapes, range, blur kernels mean-preserving, exposure matching (fit / clean)
  * every backbone builds and gets LN / LoRA subsets; zero-init LoRA keeps each backbone's output
  * self-ensemble member stacking does not change the fused pseudo-label
  * PARITY: run_tta.py --method dctta reproduces adapt.py exactly (same PSNR, same RNG checksum)
  * NSS features (gain invariance, separation, gradients); ABC selection; v0 'gain' matching; every new transfer
    loss term at once (ABC, target / re-degradation consistency, NSS prior, L2 anchor)
  * Q1/Q2a probe (tools/scale_probe.py): exact band split, padded forward, oracle bounds, same PSNR as the harness
  * end-to-end runs of identity / source / flow / transfer / dctta, all regimes, probe, aggregator, other backbones
"""
import json
import os
import sys
import tempfile
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
from PIL import Image                                # noqa: E402

torch.cuda.is_available()  # lock the device count before any script touches CUDA_VISIBLE_DEVICES
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

from dctta.models.promptir import PromptIR           # noqa: E402
from dctta.tta import DEFAULT_TRAIN_PREFIXES, collect_params, configure_model  # noqa: E402
from dctta.utils import self_ensemble                # noqa: E402
from irtta.anchors import L2Anchor                    # noqa: E402
from irtta import degrade as D                       # noqa: E402
from irtta.subsets import apply_subset               # noqa: E402

_TMP = {}


def _model(seed=0):
    torch.manual_seed(seed)
    return PromptIR(decoder=True).eval()


def _x(seed=1, s=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, s, s, generator=g)


# --------------------------------------------------------------------------- #
def test_lora_zero_init_and_rng():
    m = _model()
    x = _x()
    with torch.no_grad():
        y0 = m(x)
    state = torch.random.get_rng_state()
    sub = apply_subset(m, 'lora4', seed=3)
    assert torch.equal(state, torch.random.get_rng_state()), "subset construction consumed the global RNG"
    with torch.no_grad():
        assert torch.equal(m(x), y0), "zero-init LoRA changed the output"
    assert sub.lora_modules and all(n.endswith('lora_A') or n.endswith('lora_B') for n in sub.names)


def test_subset_contents():
    ln = apply_subset(_model(), 'ln').names
    assert ln and all(n.rsplit('.', 1)[1] in ('weight', 'bias') for n in ln)
    assert any('.norm1.body.' in n for n in ln) and any('.norm2.body.' in n for n in ln)
    s4, s16 = apply_subset(_model(), 'lora4'), apply_subset(_model(), 'lora16')
    assert s16.n_params > s4.n_params and all(m.lora_A.shape[0] == 16 for m in s16.lora_modules.values())
    assert any(k.endswith('attn.qkv.weight') for k in s16.lora_modules), "LoRA missed the attention projections"
    full = apply_subset(_model(), 'full')
    _, ref = collect_params(configure_model(_model(), DEFAULT_TRAIN_PREFIXES))
    assert full.names == ref, "'full' differs from DCTTA's prefix rule"
    for bad in ('prompt', 'prompt_offset'):
        try:
            apply_subset(_model(), bad)
            raise AssertionError(f"removed subset '{bad}' still accepted")
        except ValueError:
            pass


def test_l2_anchor():
    m = _model()
    sub = apply_subset(m, 'ln+lora4')
    a = L2Anchor(sub, lam=2.0)
    assert float(a.penalty()) == 0.0
    p = next(p for n, p in zip(sub.names, sub.params) if n.endswith('norm1.body.weight'))
    with torch.no_grad():
        p.add_(0.1)
    expect = 0.5 * 2.0 * (0.1 ** 2) * p.numel()
    assert abs(float(a.penalty()) - expect) < 1e-4 * expect
    mod = next(iter(sub.lora_modules.values()))
    with torch.no_grad():
        mod.lora_B.fill_(1.0)
    expect += 0.5 * 2.0 * float((mod.delta_weight() ** 2).sum())
    assert abs(float(a.penalty()) - expect) < 1e-3 * expect


def test_degradations():
    g = torch.Generator().manual_seed(0)
    c = torch.rand(3, 3, 64, 64, generator=g)
    for fams in [[f] for f in D.FAMILIES] + [['compose']]:
        y, cc, used, _ = D.degrade_batch(c, fams, g)
        assert y.shape == c.shape and cc.shape == c.shape and len(used) == 3, fams
        assert float(y.min()) >= 0 and float(y.max()) <= 1 and torch.isfinite(y).all(), fams
        assert not torch.allclose(y, cc), f"{fams} left the image unchanged"
        if fams != ['compose']:
            assert used == [D.chain_label(D.FAMILY_CHAINS[fams[0]])] * 3 and torch.equal(cc, c), fams
    for _ in range(20):
        k = D.random_blur_kernel(g)
        assert k.shape == (D.KSIZE, D.KSIZE) and float(k.min()) >= 0 and abs(float(k.sum()) - 1) < 1e-5
    flat = torch.full((1, 3, 64, 64), 0.4)
    assert torch.allclose(D.op_blur(flat, g), flat, atol=1e-5), "blur is not mean-preserving"
    # exposure matching. fit: the degraded image matches the target, solved through the chain; clean: the clean
    # scene is gained to the target before degrading; exposure-changing chains solve their own parameter
    c1 = c[:1] * 0.5 + 0.25
    rain = D.FAMILY_CHAINS['rain']
    for fam, tfit in (('blur', [0.10, 0.08, 0.06]), ('rain', [0.45, 0.40, 0.35])):  # reachable targets
        tfit = torch.tensor(tfit).view(1, 3, 1, 1)
        y, cc = D.apply_chain(c1, D.FAMILY_CHAINS[fam], g, tfit, 'fit')
        assert torch.allclose(y.mean(dim=(2, 3), keepdim=True), tfit, atol=4e-3), \
            f"fit: {fam} output not matched: {y.mean(dim=(2, 3)).tolist()} vs {tfit.flatten().tolist()}"
    tm = torch.tensor([0.40, 0.35, 0.30]).view(1, 3, 1, 1)
    y, cc = D.apply_chain(c1, rain, g, tm, 'clean')
    assert y.mean() > tm.mean() + 1e-3, "clean: rain must make the degraded image brighter than the target"
    y, cc = D.apply_chain(c1, rain, g, tm, 'fit')
    assert cc.mean() < y.mean(), "fit: the rain-free scene must be darker than the rainy observation"
    y, cc = D.apply_chain(c1, rain, g, tm, 'clean')
    assert torch.allclose(cc.mean(dim=(2, 3), keepdim=True), tm, atol=2e-3), "clean: scene not gained to target"
    st = g.get_state()
    D.apply_chain(c1, rain, g, tm, 'fit')
    st_fit = g.get_state()
    g.set_state(st)
    D.apply_chain(c1, rain, g, tm, 'clean')
    assert torch.equal(st_fit, g.get_state()), "fit must consume the same random draws as one chain pass"
    tm = torch.tensor([0.10, 0.08, 0.06]).view(1, 3, 1, 1)                   # a dark target patch
    y, cc = D.apply_chain(c1, {'scene': None, 'blur': False, 'exposure': True, 'noise': False}, g, tm)
    assert torch.equal(cc, c1), "low-light: the clean image must keep its natural exposure"
    assert abs(float(y.mean()) - float(tm.mean())) < 5e-3, "exposure gain not fitted to the target"
    hm = c1.mean(dim=(2, 3), keepdim=True) + 0.05                          # reachable with A in [0.3, 1]
    y, cc = D.apply_chain(c1, D.FAMILY_CHAINS['haze'], g, hm)
    assert torch.equal(cc, c1) and torch.allclose(y.mean(dim=(2, 3), keepdim=True), hm, atol=1e-3), \
        "haze airlight not fitted to the target"
    labels = {D.chain_label(D.random_chain(g)) for _ in range(200)}
    assert 'none' not in labels and len(labels) > 10 and any(l.count('+') >= 2 for l in labels), labels
    assert D.parse_families('oracle', 'deblur') == ['blur'] and len(D.parse_families('all')) == 5
    assert D.parse_families('rain+haze') == ['rain', 'haze'] and D.parse_families('compose') == ['compose']
    # gain (v0): one gain on both images, degraded matched to the target
    tm = torch.tensor([0.40, 0.35, 0.30]).view(1, 3, 1, 1)
    y, cc = D.apply_chain(c1, rain, g, tm, 'gain')
    assert torch.allclose(y.mean(dim=(2, 3), keepdim=True), tm, atol=2e-3) and cc.mean() < y.mean()
    # camera: gain for rain (same random draws -> identical pair), fit for an under-exposure chain
    st = g.get_state()
    yg, cg = D.apply_chain(c1, rain, g, tm, 'gain')
    g.set_state(st)
    yc, ccam = D.apply_chain(c1, rain, g, tm, 'camera')
    assert torch.equal(yg, yc) and torch.equal(cg, ccam)
    dark = torch.tensor([0.10, 0.08, 0.06]).view(1, 3, 1, 1)
    y, cc = D.apply_chain(c1, D.FAMILY_CHAINS['lowlight'], g, dark, 'camera')
    assert torch.equal(cc, c1), "camera: an under-exposure chain must keep the clean image's exposure"
    # under-exposure threshold: on a dark target the clean image of a rain pair keeps its natural exposure
    st = g.get_state()
    y0, c0 = D.apply_chain(c1, rain, g, dark, 'camera')
    g.set_state(st)
    yf, cf = D.apply_chain(c1, rain, g, dark, 'camera', floor=0.2)
    assert torch.equal(y0, yf) and float(c0.mean()) < 0.2 and torch.equal(cf, c1)
    g.set_state(st)
    yb, cb = D.apply_chain(c1, rain, g, tm, 'camera', floor=0.2)      # a bright target is unaffected
    g.set_state(st)
    assert torch.equal(cb, D.apply_chain(c1, rain, g, tm, 'camera')[1])
    # dark anchor: an under-exposed target's clean image takes the given exposure (here: from the source)
    anchor_m = torch.tensor([0.45, 0.40, 0.35]).view(1, 3, 1, 1)
    for ch in (rain, D.FAMILY_CHAINS['lowlight']):
        y, cc = D.apply_chain(c1, ch, g, dark, 'camera', floor=0.2, clean_mean=anchor_m)
        assert torch.allclose(cc.mean(dim=(2, 3), keepdim=True), anchor_m, atol=0.02), D.chain_label(ch)
    # v2 low-light: darkens, matches the target brightness, adds its own noise (no extra sensor-noise pass)
    ll = D.FAMILY_CHAINS['lowlight']
    y2, c2 = D.apply_chain(c1, ll, g, dark, 'camera', ll='v2')
    assert torch.equal(c2, c1) and abs(float(y2.mean()) - float(dark.mean())) < 0.03 and float(y2.std()) > 0
    # dark override: dark=True behaves like a floor hit, dark=False like a bright target
    st = g.get_state()
    yo, co = D.apply_chain(c1, rain, g, tm, 'camera', dark=True)
    assert torch.equal(co, c1)
    g.set_state(st)
    yf, cfl = D.apply_chain(c1, rain, g, dark, 'camera', dark=False)
    assert float(cfl.mean()) < 0.2
    # top-k ABC: k pairs per crop
    out = D.degrade_batch(c[:2], ['compose'], g, None, 'fit', select=lambda t: -t.mean(dim=(1, 2, 3)), k=5, topk=2)
    assert out[0].shape[0] == 4 and len(out[2]) == 4 and out[3][0] <= out[3][1]
    # ABC: with a selector the lowest-scoring candidate is kept
    picks = D.degrade_batch(c[:2], ['compose'], g, None, 'fit', select=lambda t: -t.mean(dim=(1, 2, 3)), k=6)
    assert len(picks[3]) == 2 and picks[0].shape == c[:2].shape


def test_nss():
    from irtta import nss as N
    g = torch.Generator().manual_seed(4)
    x = torch.rand(4, 3, 96, 96, generator=g) * 0.6 + 0.2
    f = N.nss_features(x)
    assert f.shape == (4, N.N_FEATURES) and torch.isfinite(f).all()
    fc = N.nss_features(x, color=True)
    assert fc.shape == (4, N.N_FEATURES + 2) and torch.equal(fc[:, :N.N_FEATURES], f)
    hazy = 0.5 * x + 0.5 * torch.tensor([0.9, 0.9, 0.95]).view(1, 3, 1, 1)
    hc = N.nss_features(hazy, color=True)
    assert (hc[:, -2] > fc[:, -2]).all() and (hc[:, -1] < fc[:, -1]).all(), "haze: dark channel up, saturation down"
    assert torch.allclose(N.nss_features(x * 0.7, color=True)[:, -2:], fc[:, -2:], rtol=0.05, atol=1e-3)
    # MSCN statistics nearly ignore a global gain (only the log-contrast features move). Not exactly: the
    # stabilising constant C = 1/255 in (I - mu) / (sigma + C) does not scale, so a few % change is expected
    f2 = N.nss_features((x * 0.7).clamp(0, 1))
    keep = [i for i in range(N.N_FEATURES) if i not in (14, 29)]
    assert torch.allclose(f[:, keep], f2[:, keep], rtol=0.1, atol=1e-2), (f[:, keep] - f2[:, keep]).abs().max()
    # blur and noise move the features; the Gaussian distance separates them from the clean fit
    clean = torch.cat([D.conv_same(x[i:i + 1], D.gaussian_kernel(9, 1.2, 1.2, 0.)) for i in range(4)])
    xs = torch.cat([clean, D.conv_same(clean, D.gaussian_kernel(9, 0.6, 0.6, 0.))], 0)
    model = N.GaussianModel(N.nss_features(xs.repeat(4, 1, 1, 1) + 0.002 * torch.randn(32, 3, 96, 96, generator=g)))
    noisy = (clean + 0.1 * torch.randn(clean.shape, generator=g)).clamp(0, 1)
    assert float(model.distance(N.nss_features(noisy)).mean()) > float(model.distance(N.nss_features(clean)).mean())
    xg = x.clone().requires_grad_(True)
    model.distance(N.nss_features(xg)).mean().backward()
    assert torch.isfinite(xg.grad).all() and float(xg.grad.abs().sum()) > 0


def test_self_ensemble_members():
    m, x = _model(), _x()
    with torch.no_grad():
        y = self_ensemble(m, x)
        y2, mem = self_ensemble(m, x, return_members=True)
    assert torch.equal(y, y2) and mem.shape[0] == 8 and torch.allclose(mem.mean(0), y, atol=1e-6)


def test_backbones():
    """Every backbone (random weights): builds, LN and LoRA subsets are non-empty, zero-init LoRA keeps the output,
    the DCTTA 'full' rule selects parameters. R2R is only built (its forward needs the released degradation bank)."""
    from dctta.models import BACKBONES, TRAIN_PREFIXES, build_model
    x = _x(s=64)
    for name in BACKBONES:
        torch.manual_seed(0)
        opts = {'routing_noise': False} if name == 'moceir' else {}
        m = build_model(name, None, **opts).eval()
        assert apply_subset(build_model(name, None, **opts), 'ln').names, f"{name}: no LayerNorm found"
        assert apply_subset(build_model(name, None, **opts), 'full', train_prefixes=TRAIN_PREFIXES[name]).params
        if name == 'r2r':
            assert apply_subset(m, 'lora16').lora_modules
            continue
        with torch.no_grad():
            y0 = m(x)
            sub = apply_subset(m, 'lora16', seed=1)
            m.eval()
            y1 = m(x)
        assert y0.shape == x.shape and sub.lora_modules, name
        assert torch.allclose(y0, y1, atol=1e-6), f"{name}: zero-init LoRA changed the output"


# --------------------------------------------------------------------------- #
def _setup_data():
    if _TMP:
        return _TMP
    d = tempfile.mkdtemp(prefix='irtta_test_')
    rs = np.random.RandomState(0)
    for sub in ('LQ', 'GT'):
        os.makedirs(os.path.join(d, 'tiny', sub))
    for i in range(3):
        gt = (rs.rand(96, 96, 3) * 255).astype(np.uint8)
        lq = np.clip(gt.astype(float) * 0.7 + rs.randn(96, 96, 3) * 10, 0, 255).astype(np.uint8)
        Image.fromarray(gt).save(os.path.join(d, 'tiny', 'GT', f'img{i}.png'))
        Image.fromarray(lq).save(os.path.join(d, 'tiny', 'LQ', f'img{i}.png'))
    m = _model(5)
    ckpt = os.path.join(d, 'fake.ckpt')
    torch.save({'state_dict': {'net.' + k: v for k, v in m.state_dict().items()}}, ckpt)
    from dctta.models import build_model
    torch.manual_seed(6)
    ad = build_model('adair', None)
    os.makedirs(os.path.join(d, 'adair'))
    ckpt_adair = os.path.join(d, 'adair', 'adair3d.ckpt')
    torch.save({'state_dict': {'net.' + k: v for k, v in ad.state_dict().items()}}, ckpt_adair)
    reg = {'_root': d, '_source_sets': {'tiny': ['TinyNoise']}, '_banks': {'default': '{root}/tiny/GT'},
           'Tiny': {'lq': '{root}/tiny/LQ', 'gt': '{root}/tiny/GT', 'split': 'test', 'task': 'deblur'},
           'TinyNoise': {'gt': '{root}/tiny/GT', 'noise_sigma': 25, 'split': 'source', 'task': 'denoise'}}
    regp = os.path.join(d, 'registry.json')
    json.dump(reg, open(regp, 'w'))
    with open(os.path.join(d, 'probe.txt'), 'w') as f:
        f.write("img0.png,normal\nimg2.png,hard\n")
    _TMP.update(dir=d, ckpt=ckpt, ckpt_adair=ckpt_adair, reg=regp, probe=os.path.join(d, 'probe.txt'),
                lq=os.path.join(d, 'tiny', 'LQ'), gt=os.path.join(d, 'tiny', 'GT'))
    return _TMP


def _run(extra):
    import run_tta
    t = _setup_data()
    base = ['--ckpt', t['ckpt'], '--registry', t['reg'], '--source', 'tiny', '--patch_size', '64',
            '--num_workers', '0', '--out_root', os.path.join(t['dir'], 'results'), '--exp', 'test',
            '--perceptual_weight', '0', '--overwrite']
    return run_tta.main(base + extra)


def test_parity_with_adapt_py():
    """The key regression guard: harness DCTTA == reference adapt.py (same numbers, same checksum)."""
    import adapt
    t = _setup_data()
    common = ['--lq_dir', t['lq'], '--gt_dir', t['gt'], '--ckpt', t['ckpt'], '--patch_size', '64',
              '--num_workers', '0', '--max_steps', '3', '--perceptual_weight', '0', '--gen_train_steps', '2']
    with mock.patch.object(sys, 'argv', ['adapt.py'] + common + [
            '--results_dir', os.path.join(t['dir'], 'adapt_ref'), '--no_save_images', '--gpu', '']):
        ref_psnr, ref_ssim = adapt.main()
    res = _run(['--method', 'dctta', '--lq_dir', t['lq'], '--gt_dir', t['gt'], '--max_steps', '3',
                '--gen_train_steps', '2', '--seed', '23', '--dataset_seed', '42', '--registry', t['reg']])
    assert res['psnr'] == ref_psnr and res['ssim'] == ref_ssim, \
        f"parity broken: adapt.py {ref_psnr:.6f}/{ref_ssim:.6f} vs run_tta {res['psnr']:.6f}/{res['ssim']:.6f}"
    print(f"    parity OK: PSNR {ref_psnr:.6f}, checksum {res['rng_checksum']}")


def test_end_to_end():
    import tools.aggregate as agg
    t = _setup_data()
    r_id = _run(['--method', 'identity', '--target', 'Tiny'])
    assert r_id['n_params_adapted'] == 0 and r_id['psnr'] > 0
    r_src = _run(['--method', 'source', '--target', 'Tiny', '--retention'])
    for fam in ('oracle', 'all'):
        r = _run(['--method', 'transfer', '--families', fam, '--target', 'Tiny', '--max_steps', '2',
                  '--bank_patch', '64', '--bank_batch', '2', '--log_stats', '--retention'])
        assert r['steps'] == 2 and sum(r['families_used'].values()) == 4, r['families_used']
    _run(['--method', 'transfer', '--subset', 'ln', '--anchor', 'l2', '--anchor_lambda', '1',
          '--probe_list', t['probe'], '--probe_every', '1', '--exposure_match', 'none', '--target', 'Tiny', '--max_steps', '2', '--bank_patch', '64', '--bank_batch', '2'])
    # v0.1: compose, step budget cycles the 3-image set, image subset, weight-mix evaluation
    r = _run(['--method', 'transfer', '--families', 'compose', '--target', 'Tiny', '--max_steps', '7',
              '--max_images', '2', '--bank_patch', '64', '--bank_batch', '2', '--eval_alphas', '0.5', '1.0'])
    assert r['steps'] == 7 and r['n_images'] == 2, (r['steps'], r['n_images'])
    ra = json.load(open(os.path.join(t['dir'], 'results', 'test', r['run_id'], 'alpha0.5', 'result.json')))
    assert ra['eval_alpha'] == 0.5 and ra['n_images'] == 2 and ra['psnr'] != r['psnr'], ra
    _run(['--method', 'transfer', '--exposure_match', 'clean', '--tag', 'v02', '--target', 'Tiny', '--max_steps', '2',
          '--bank_patch', '64', '--bank_batch', '2'])
    # every extra term at once (ABC selection, target and re-degradation consistency, NSS prior, L2 anchor)
    r_all = _run(['--method', 'transfer', '--families', 'all', '--exposure_match', 'camera', '--dark_q', '0.05', '--dark_anchor', '--nss_color',
                  '--abc_k', '3',
                  '--abc_fit_n', '8', '--cons_weight', '0.5', '--cons_band', 'high', '--redeg_weight', '0.5',
                  '--nss_weight', '0.02', '--anchor', 'l2', '--anchor_lambda', '0.01', '--target', 'Tiny',
                  '--max_steps', '3', '--bank_patch', '64', '--bank_batch', '2', '--log_stats',
                  '--eval_alphas', '0.5', '1.0'])
    assert r_all['steps'] == 3 and 'abc=3' in r_all['run_id'] and r_all['psnr'] > 0, r_all['run_id']
    stats = [json.loads(l) for l in open(os.path.join(t['dir'], 'results', 'test', r_all['run_id'], 'stats.jsonl'))]
    assert all(k in stats[-1] for k in ('abc_d', 'cons', 'redeg', 'nss')), stats[-1].keys()
    # ABC tolerance: an impossible tolerance skips every step; a loose one accepts every pair
    r_rej = _run(['--method', 'transfer', '--families', 'all', '--abc_k', '2', '--abc_fit_n', '8', '--abc_eps_q', '1e-9',
                  '--target', 'Tiny', '--max_steps', '3', '--bank_patch', '64', '--bank_batch', '2', '--log_stats',
                  '--tag', 'rej'])
    assert r_rej['steps'] + r_rej['skipped'] == 3, (r_rej['steps'], r_rej['skipped'])
    # new dark path: source rule (+ every action), v2 low-light, resolution-matched bank, top-2 ABC
    for act in ('skip', 'anchor', 'norm'):
        r_n = _run(['--method', 'transfer', '--families', 'all', '--exposure_match', 'camera', '--dark_rule', 'source',
                    '--dark_ratio', '0.5', '--dark_action', act, '--lowlight_op', 'v2', '--bank_res', 'target',
                    '--abc_k', '3', '--abc_topk', '2', '--abc_fit_n', '8', '--nss_color', '--target', 'Tiny',
                    '--max_steps', '2', '--bank_patch', '64', '--bank_batch', '2', '--log_stats', '--tag', f'n{act}'])
        st_n = [json.loads(l) for l in open(os.path.join(t['dir'], 'results', 'test', r_n['run_id'], 'stats.jsonl'))]
        assert all(s_.get('dark') == 1 for s_ in st_n), st_n          # ratio 0.5: every patch counts as dark
        assert (r_n['skipped'] == 2) if act == 'skip' else (r_n['steps'] == 2), (act, r_n['steps'], r_n['skipped'])
    # dark skip: the Tiny target patches are darker than the bank's 0.99 quantile -> every step abstains
    r_dk = _run(['--method', 'transfer', '--families', 'all', '--exposure_match', 'camera', '--dark_q', '0.99',
                 '--dark_skip', '--target', 'Tiny', '--max_steps', '3', '--bank_patch', '64', '--bank_batch', '2',
                 '--tag', 'dk'])
    assert r_dk['skipped'] == 3 and r_dk['steps'] == 0 and abs(r_dk['psnr'] - r_src['psnr']) < 1e-4, r_dk
    # another backbone through the same harness (run id carries the backbone; DCTTA uses its prefix rule)
    ckpt_ad = t['ckpt_adair']
    for meth in ('source', 'transfer', 'dctta'):
        r_ad = _run(['--method', meth, '--backbone', 'adair', '--ckpt', ckpt_ad, '--source', 'tiny', '--target', 'Tiny',
                     '--max_steps', '2', '--bank_patch', '64', '--bank_batch', '2', '--gen_train_steps', '1'])
        assert r_ad['run_id'].startswith('adair-') and r_ad['backbone'] == 'adair' and r_ad['psnr'] > 0, r_ad['run_id']
    # source flow: alpha 1 x1 is the source itself
    r_f = _run(['--method', 'flow', '--target', 'Tiny'])
    assert abs(r_f['psnr'] - r_src['psnr']) < 1e-3, (r_f['psnr'], r_src['psnr'])
    _run(['--method', 'flow', '--ext_alpha', '1.5', '--n_apply', '2', '--target', 'Tiny'])
    r_on = _run(['--method', 'transfer', '--regime', 'online', '--steps_per_image', '2', '--target', 'Tiny',
                 '--bank_patch', '64', '--bank_batch', '2', '--probe_list', t['probe'], '--probe_every', '1'])
    r_epi = _run(['--method', 'transfer', '--regime', 'episodic', '--steps_per_image', '2', '--target', 'Tiny',
                  '--bank_patch', '64', '--bank_batch', '2', '--online_predict', 'before'])
    assert r_on['n_images'] == 3 and r_epi['steps'] == 6
    _run(['--method', 'dctta', '--subset', 'lora16', '--anchor', 'none', '--regime', 'episodic', '--steps_per_image',
          '1', '--target', 'Tiny', '--gen_train_steps', '1'])
    with mock.patch.object(sys, 'argv', ['agg', os.path.join(t['dir'], 'results', 'test')]):
        agg.main()
    assert os.path.exists(os.path.join(t['dir'], 'results', 'test', 'summary.csv'))


def test_scale_probe():
    """Q1/Q2a probe: exact band split, padded forward, oracle bounds, report."""
    import tools.scale_probe as sp
    g = torch.Generator().manual_seed(3)
    r = torch.randn(1, 3, 50, 70, generator=g)
    lo, mid, hi = sp.split_bands(r, 1.5, 6.0)
    assert torch.allclose(lo + mid + hi, r, atol=1e-5)
    assert lo.std() < r.std() and abs(float(sp.gaussian_blur(torch.ones(1, 3, 40, 40), 2.0).mean()) - 1) < 1e-5
    m = _model(5)
    with torch.no_grad():
        x = torch.rand(1, 3, 50, 70, generator=g)
        assert sp.forward(m, x).shape == x.shape
        x = torch.rand(1, 3, 64, 64, generator=g)
        assert torch.equal(sp.forward(m, x), m(x))                          # aligned input: no padding
    assert sp.downscale(x, 2).shape[-2:] == (32, 32) and sp.downscale(x, 3).shape[-2:] == (21, 21)
    t = _setup_data()
    out = os.path.join(t['dir'], 'results')
    res = sp.main(['--target', 'Tiny', '--ckpt', t['ckpt'], '--registry', t['reg'], '--source', 'tiny',
                   '--num_workers', '0', '--out_root', out, '--exp', 'probe', '--overwrite'])
    assert res['n'] == 3 and res['source_psnr'] >= res['source_unclipped_psnr'] - 1e-6, res
    assert res['band_psnr'] >= res['global_psnr'] - 1e-6 >= res['source_unclipped_psnr'] - 2e-6, res
    r_src = _run(['--method', 'source', '--target', 'Tiny'])
    assert abs(res['source_psnr'] - r_src['psnr']) < 1e-4, (res['source_psnr'], r_src['psnr'])   # same metric
    sp.main(['--report', os.path.join(out, 'probe')])
    assert os.path.exists(os.path.join(out, 'probe', 'probe_tiny.md'))


def main():
    tests =[(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed" + (f"   (scratch dir: {_TMP.get('dir')})" if _TMP else ''))
    return failed


if __name__ == '__main__':
    sys.exit(main())
