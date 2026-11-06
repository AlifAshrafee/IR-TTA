"""CPU self-tests: ``python tests/test_harness.py`` (or ``bash scripts/run.sh test``).

Random-weight models and a 3-image synthetic dataset; no checkpoints or datasets needed.
"""
import json
import os
import sys
import tempfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
from PIL import Image                                # noqa: E402

torch.cuda.is_available()  # lock the device count before any script touches CUDA_VISIBLE_DEVICES
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

from irtta import degrade as D                       # noqa: E402
from irtta import nss as N                           # noqa: E402
from irtta.backbones.promptir import PromptIR        # noqa: E402
from irtta.baselines.dctta.tta import DEFAULT_TRAIN_PREFIXES, collect_params, configure_model  # noqa: E402
from irtta.baselines.dctta.utils import self_ensemble  # noqa: E402
from irtta.subsets import apply_subset               # noqa: E402

_TMP = {}


def _model(seed=0):
    torch.manual_seed(seed)
    return PromptIR(decoder=True).eval()


def _x(seed=1, s=64):
    return torch.rand(1, 3, s, s, generator=torch.Generator().manual_seed(seed))


# --------------------------------------------------------------------------- #
def test_run_ids():
    """Run ids are stable: these are ids of runs reported in the paper (finished runs are recognised)."""
    from irtta.config import get_args, make_run_id

    def rid(argv):
        a = get_args(argv)
        return make_run_id(a, a.target)
    p5, p3 = ['--ckpt', 'pretrain/epoch=80.ckpt'], ['--ckpt', 'pretrain/model.ckpt']
    assert rid(p5 + ['--target', 'SOTS_Outdoor', '--tag', 'F1']) == (
        'transfer-lora16-none-transductive-SOTS_Outdoor-5task-lr0.0001-s23-'
        'fam=all_em=camera_drl=source_dac=anchor_nc=True_abc=8_ms=500-F1')
    assert rid(p5 + ['--target', 'SOTS_Outdoor', '--method', 'source']) == \
        'source-none-none-transductive-SOTS_Outdoor-5task-s23'
    assert rid(p3 + ['--target', 'Rain100H', '--method', 'dctta']) == \
        'dctta-full-tips-transductive-Rain100H-3task-lr0.0002-s23'
    assert rid(p3 + ['--target', 'Rain100H', '--tag', 'G1', '--abc_k', '1', '--exposure_match', 'none',
                     '--dark_rule', 'none']) == \
        'transfer-lora16-none-transductive-Rain100H-3task-lr0.0001-s23-fam=noise_rain_haze_em=none_ms=500-G1'
    assert rid(['--ckpt', 'x/fake.ckpt', '--source', 'tiny', '--target', 'Tiny', '--oracle_pairs', 'Tiny',
                '--abc_k', '1', '--exposure_match', 'none', '--max_steps', '2', '--bank_patch', '32',
                '--bank_batch', '2', '--patch_size', '64', '--tag', 'o']) == \
        'transfer-lora16-none-transductive-Tiny-tiny-lr0.0001-s23-bp=32_bb=2_em=none_ps=64_ms=2_orp=Tiny-o'
    a = get_args(['--ckpt', 'pretrain/moceir/MoCE_IR_AIO3/last.ckpt', '--backbone', 'moceir', '--target', 'HIDE'])
    assert a.source == '3task' and a.families == 'noise+rain+haze' and a.max_steps == 500 and a.abc_k == 8


def test_lora_zero_init_and_rng():
    m = _model()
    x = _x()
    with torch.no_grad():
        y0 = m(x)
    state = torch.random.get_rng_state()
    sub = apply_subset(m, 'lora16', seed=3)
    assert torch.equal(state, torch.random.get_rng_state()), "subset construction consumed the global RNG"
    with torch.no_grad():
        assert torch.equal(m(x), y0), "zero-init LoRA changed the output"
    assert sub.lora_modules and all(n.endswith('lora_A') or n.endswith('lora_B') for n in sub.names)
    assert any(k.endswith('attn.qkv.weight') for k in sub.lora_modules), "LoRA missed the attention projections"
    full = apply_subset(_model(), 'full')
    _, ref = collect_params(configure_model(_model(), DEFAULT_TRAIN_PREFIXES))
    assert full.names == ref, "'full' differs from DCTTA's prefix rule"
    assert apply_subset(_model(), 'all').n_params == sum(p.numel() for p in _model().parameters())


def test_degradations():
    g = torch.Generator().manual_seed(0)
    c = torch.rand(3, 3, 64, 64, generator=g)
    for f in D.FAMILIES:
        y, cc, used, _ = D.degrade_batch(c, [f], g)
        assert y.shape == c.shape and torch.equal(cc, c) and len(used) == 3, f
        assert float(y.min()) >= 0 and float(y.max()) <= 1 and torch.isfinite(y).all(), f
        assert not torch.allclose(y, cc), f"{f} left the image unchanged"
    for _ in range(20):
        k = D.random_blur_kernel(g)
        assert k.shape == (D.KSIZE, D.KSIZE) and float(k.min()) >= 0 and abs(float(k.sum()) - 1) < 1e-5
    flat = torch.full((1, 3, 64, 64), 0.4)
    assert torch.allclose(D.op_blur(flat, g), flat, atol=1e-5), "blur is not mean-preserving"
    assert D.parse_families('oracle', 'deblur') == ['blur'] and len(D.parse_families('all')) == 5
    assert D.parse_families('noise+rain+haze') == ['noise', 'rain', 'haze']
    # camera-consistent exposure: one gain on both images, the degraded one matched to the target patch
    c1 = c[:1] * 0.5 + 0.25
    tm = torch.tensor([0.40, 0.35, 0.30]).view(1, 3, 1, 1)
    for f in ('rain', 'haze', 'blur', 'noise'):
        y, cc = D.apply_chain(c1, D.FAMILY_CHAINS[f], g, tm)
        assert torch.allclose(y.mean(dim=(2, 3), keepdim=True), tm, atol=0.02), f
        ratio = cc.mean(dim=(2, 3)) / c1.mean(dim=(2, 3))
        assert not torch.allclose(ratio, torch.ones_like(ratio)), f"{f}: clean image not gained with the capture"
    # an under-exposure chain solves its own gain; the clean image keeps its natural exposure
    dark = torch.tensor([0.10, 0.08, 0.06]).view(1, 3, 1, 1)
    y, cc = D.apply_chain(c1, D.FAMILY_CHAINS['lowlight'], g, dark)
    assert torch.equal(cc, c1) and abs(float(y.mean()) - float(dark.mean())) < 0.02
    # under-exposed target (dark): the clean image takes the given brightness (the pretrained model's output)
    anchor_m = torch.tensor([0.45, 0.40, 0.35]).view(1, 3, 1, 1)
    for f in ('rain', 'lowlight'):
        y, cc = D.apply_chain(c1, D.FAMILY_CHAINS[f], g, dark, clean_mean=anchor_m, dark=True)
        assert torch.allclose(cc.mean(dim=(2, 3), keepdim=True), anchor_m, atol=0.02), f
    st = g.get_state()
    y0, c0 = D.apply_chain(c1, D.FAMILY_CHAINS['rain'], g, dark, dark=True)
    assert torch.equal(c0, c1), "dark without an anchor: natural exposure"
    g.set_state(st)
    # ABC: with a selector the lowest-scoring candidate is kept
    y, cc, used, sc = D.degrade_batch(c[:2], D.parse_families('all'), g, tm, select=lambda t: -t.mean(dim=(1, 2, 3)),
                                      k=6)
    assert y.shape == c[:2].shape and len(sc) == 2


def test_nss():
    g = torch.Generator().manual_seed(4)
    x = torch.rand(4, 3, 96, 96, generator=g) * 0.6 + 0.2
    f = N.nss_features(x)
    assert f.shape == (4, N.N_FEATURES) and torch.isfinite(f).all()
    fc = N.nss_features(x, color=True)
    assert fc.shape == (4, N.N_FEATURES + 2) and torch.equal(fc[:, :N.N_FEATURES], f)
    hazy = 0.5 * x + 0.5 * torch.tensor([0.9, 0.9, 0.95]).view(1, 3, 1, 1)
    hc = N.nss_features(hazy, color=True)
    assert (hc[:, -2] > fc[:, -2]).all() and (hc[:, -1] < fc[:, -1]).all(), "haze: dark channel up, saturation down"
    clean = torch.cat([D.conv_same(x[i:i + 1], D.gaussian_kernel(9, 1.2, 1.2, 0.)) for i in range(4)])
    model = N.GaussianModel(N.nss_features(torch.cat([clean, clean.flip(-1)]), color=True))
    noisy = (clean + 0.1 * torch.randn(clean.shape, generator=g)).clamp(0, 1)
    assert float(model.distance(N.nss_features(noisy, color=True)).mean()) > \
        float(model.distance(N.nss_features(clean, color=True)).mean())


def test_self_ensemble():
    m, x = _model(), _x()
    with torch.no_grad():
        y, mem = self_ensemble(m, x, return_members=True)
    assert mem.shape[0] == 8 and torch.allclose(mem.mean(0), y, atol=1e-6)


def test_backbones():
    """Every backbone (random weights) builds; zero-init LoRA keeps its output; DCTTA's rule selects parameters.
    R2R is only built (its forward needs the released degradation bank)."""
    from irtta.backbones import BACKBONES, TRAIN_PREFIXES, build_model
    x = _x(s=64)
    for name in BACKBONES:
        torch.manual_seed(0)
        opts = {'routing_noise': False} if name == 'moceir' else {}
        m = build_model(name, None, **opts).eval()
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
    from irtta.backbones import build_model
    torch.manual_seed(6)
    ad = build_model('adair', None)
    os.makedirs(os.path.join(d, 'adair'))
    ckpt_adair = os.path.join(d, 'adair', 'adair3d.ckpt')
    torch.save({'state_dict': {'net.' + k: v for k, v in ad.state_dict().items()}}, ckpt_adair)
    reg = {'_root': d, '_banks': {'default': '{root}/tiny/GT'},
           'Tiny': {'lq': '{root}/tiny/LQ', 'gt': '{root}/tiny/GT', 'split': 'test', 'task': 'deblur'},
           'TinyNoise': {'gt': '{root}/tiny/GT', 'noise_sigma': 25, 'split': 'noshift', 'task': 'denoise'}}
    regp = os.path.join(d, 'registry.json')
    json.dump(reg, open(regp, 'w'))
    os.environ['IRTTA_CACHE'] = os.path.join(d, 'cache')
    _TMP.update(dir=d, ckpt=ckpt, ckpt_adair=ckpt_adair, reg=regp)
    return _TMP


def _run(extra, ckpt=None):
    import run_tta
    t = _setup_data()
    base = ['--ckpt', ckpt or t['ckpt'], '--registry', t['reg'], '--source', 'tiny', '--patch_size', '64',
            '--num_workers', '0', '--out_root', os.path.join(t['dir'], 'results'), '--exp', 'test', '--overwrite',
            '--bank_patch', '64', '--bank_batch', '2', '--abc_fit_n', '8']
    return run_tta.main(base + extra)


def test_end_to_end():
    t = _setup_data()
    r_id = _run(['--method', 'identity', '--target', 'Tiny'])
    r_src = _run(['--method', 'source', '--target', 'Tiny'])
    assert r_id['n_params_adapted'] == 0 and r_src['psnr'] != r_id['psnr']
    # ours with the default (final) configuration; the step budget cycles the 3-image set
    r = _run(['--target', 'Tiny', '--families', 'all', '--max_steps', '5', '--tag', 'F1'])
    assert r['steps'] + r['skipped'] == 5 and 'abc=8' in r['run_id'] and r['psnr'] != r_src['psnr'], r['run_id']
    # PromptIR's unused 1x1 convs (chnl_reduce*, reduce_noise_channel_*) get no gradient
    assert 0 < r['n_params_updated'] < r['n_params_adapted'], (r['n_params_updated'], r['n_params_adapted'])
    # every target patch counted as under-exposed (dark_ratio 0): anchored pairs; skip action abstains
    r_a = _run(['--target', 'Tiny', '--families', 'all', '--max_steps', '2', '--dark_ratio', '0', '--tag', 'a'])
    assert r_a['steps'] == 2
    r_s = _run(['--target', 'Tiny', '--families', 'all', '--max_steps', '2', '--dark_ratio', '0',
                '--dark_action', 'skip', '--tag', 's'])
    assert r_s['skipped'] == 2 and r_s['psnr'] == r_src['psnr']
    # component switches, image subset
    r_g = _run(['--target', 'Tiny', '--families', 'all', '--abc_k', '1', '--exposure_match', 'none',
                '--dark_rule', 'none', '--max_steps', '2', '--max_images', '2', '--tag', 'g'])
    assert r_g['steps'] == 2 and r_g['n_images'] == 2
    # controls: random initialisation; adaptation conditioned on another set
    r_z = _run(['--target', 'Tiny', '--families', 'all', '--init', 'scratch', '--subset', 'all', '--lr', '2e-4',
                '--dark_rule', 'none', '--max_steps', '2', '--tag', 'z'])
    assert 'ini=scratch' in r_z['run_id'] and r_z['psnr'] != r_src['psnr']
    r_c = _run(['--target', 'Tiny', '--families', 'all', '--cond_target', 'Tiny', '--max_steps', '2', '--tag', 'c'])
    assert 'ct=Tiny' in r_c['run_id'] and r_c['steps'] == 2
    # oracle analysis: the target's family only; real pairs (the evaluated image of the same folder excluded)
    r_f = _run(['--target', 'Tiny', '--families', 'oracle', '--max_steps', '2', '--tag', 'fo'])
    assert r_f['families_used'] and set(r_f['families_used']) == {'blur'}
    r_o = _run(['--target', 'Tiny', '--oracle_pairs', 'Tiny', '--abc_k', '1', '--exposure_match', 'none',
                '--max_images', '1', '--max_steps', '2', '--bank_patch', '32', '--tag', 'o'])
    assert 'orp=Tiny' in r_o['run_id'] and r_o['steps'] == 2
    from irtta.data import (PairBank, load_registry, make_eval_dataset, materialize_synthetic, resolve_spec)
    reg = load_registry(t['reg'])
    assert len(PairBank(resolve_spec('Tiny', reg), exclude=['img1.png'], min_side=32)) == 2
    # synthetic-noise target: materialised once, bit-identical to the on-the-fly pairs
    nspec = resolve_spec('TinyNoise', reg)
    a_, b_ = make_eval_dataset(nspec), make_eval_dataset(materialize_synthetic(nspec))
    for i in range(len(a_)):
        assert torch.equal(a_[i][1], b_[i][1]) and torch.equal(a_[i][2], b_[i][2])
    r_n = _run(['--target', 'TinyNoise', '--families', 'noise', '--max_steps', '2', '--tag', 'n'])
    assert r_n['steps'] + r_n['skipped'] == 2 and r_n['n_images'] == 3
    # target images smaller than --patch_size: the largest 16-aligned square
    r_sm = _run(['--target', 'Tiny', '--families', 'noise', '--patch_size', '128', '--max_steps', '1', '--tag', 'sm'])
    assert r_sm['steps'] + r_sm['skipped'] == 1
    # the DCTTA baseline (PromptIR and another backbone), and our method on another backbone
    r_d = _run(['--method', 'dctta', '--target', 'Tiny', '--perceptual_weight', '0'])
    assert r_d['steps'] == 3 and r_d['n_params_effective'] < r_d['n_params_adapted']
    assert 0 < r_d['n_params_updated'] <= r_d['n_params_adapted']
    r_dad = _run(['--method', 'dctta', '--backbone', 'adair', '--target', 'Tiny', '--perceptual_weight', '0'],
                 ckpt=t['ckpt_adair'])
    assert r_dad['run_id'].startswith('adair-dctta') and r_dad['steps'] == 3
    r_ad = _run(['--backbone', 'adair', '--target', 'Tiny', '--families', 'all', '--max_steps', '2', '--tag', 'F1'],
                ckpt=t['ckpt_adair'])
    assert r_ad['run_id'].startswith('adair-transfer')
    # aggregation and tables
    import subprocess
    res = os.path.join(t['dir'], 'results', 'test')
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'tools', 'aggregate.py'), res])
    subprocess.check_call([sys.executable, os.path.join(ROOT, 'tools', 'table.py'), res],
                          stdout=subprocess.DEVNULL)
    assert os.path.exists(os.path.join(res, 'table_main_3task.md'))


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
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
