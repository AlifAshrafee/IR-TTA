# DCTTA (cleaned) — Degradation-Consistent Test-Time Adaptation for All-in-One Image Restoration

A stripped-down, dependency-light rebuild of the official implementation of
Tang et al., *Degradation-Consistent Test-Time Adaptation for All-in-One Image Restoration*, CVPR 2026
(original: https://github.com/tonia86/DCTTA, built on PromptIR and RDDM).

Only the code that the released pipeline actually executes survives; every numerical choice of that
pipeline is preserved (see `CLEANUP_NOTES.md` for the full removal log, the component map, and the
places where the released code differs from the paper). `tests/test_equivalence.py` checks the rebuild
against the original repository tensor-for-tensor.

## Layout

```
adapt.py                      DCTTA adaptation + evaluation      (was train_test_promptir.py)
test.py                       un-adapted source baseline (+ "*" self-ensemble rows)  (was test_promptir.py)
evaluate.py                   full-image PSNR/SSIM loop shared by both scripts
dctta/
  config.py                   argparse options (only the ones that are read)
  data.py                     AdaptationPatchDataset (random crops), PairedImageDataset (evaluation)
  losses.py                   L1 + VGG19 perceptual + Haar low-pass loss; Charbonnier for the Fisher pass
  metrics.py                  PSNR / SSIM (scikit-image, RGB)
  tta.py                      DCTTA: self-ensemble pseudo-label, online generator, losses, TIPS, EMA teacher
  utils.py                    seeding, checkpoint loading, dihedral transforms, self-ensemble, image saving
  models/promptir.py          PromptIR backbone (verbatim)
  rddm/unet.py                RDDM U-Net (verbatim blocks)
  rddm/diffusion.py           residual diffusion, pred_res objective, DDIM sampler
  rddm/generator.py           online-trained degradation generator (replaces Trainer + 2 wrappers)
tools/iqa_metrics.py          optional pyiqa evaluation (LPIPS, DISTS, NIQE, MUSIQ, CLIP-IQA)
tests/test_equivalence.py     numerical equivalence against the original repo
pretrain/                     PromptIR checkpoints (see pretrain/README.md)
testdata/<set>/{LQ,GT}/       paired target images
```

## Setup

```bash
pip install -r requirements.txt
# put model.ckpt / epoch=80.ckpt into pretrain/, and e.g. Rain100H into testdata/Rain100H/{LQ,GT}
```

## Run

```bash
# source model, no adaptation (paper's "PromptIR" rows); add --self_ensemble for the "PromptIR*" rows
python test.py  --lq_dir testdata/Rain100H/LQ --gt_dir testdata/Rain100H/GT --ckpt pretrain/model.ckpt

# DCTTA (defaults = released configuration)
python adapt.py --lq_dir testdata/Rain100H/LQ --gt_dir testdata/Rain100H/GT --ckpt pretrain/model.ckpt \
                --results_dir results/Rain100H_DCTTA

# 5-task source
python adapt.py ... --ckpt pretrain/epoch=80.ckpt
```

Reference numbers (PromptIR 3-task source, Rain100H, from the log shipped with the original repo):
un-adapted 15.550 dB / 0.4868, after DCTTA 21.269 dB / 0.6179 (paper Table: 15.64 -> 20.21).

## Ablation switches (all default to the released values)

| flag | component | default |
|---|---|---|
| `--no_fisher`, `--fisher_ratio` | TIPS parameter freezing (per-tensor top-ratio restored to source) | on, 0.6 |
| `--teacher_weight` | pseudo-label distillation weight (alpha) | 5 |
| `--ema_decay` | teacher EMA decay (eta) | 0.45 |
| `--pixel_weight --perceptual_weight --lowpass_weight` | loss terms | 1 / 0.01 / 0.1 |
| `--train_prefixes` | which top-level PromptIR modules adapt (`all` for everything) | prompt encoder decoder |
| `--gen_train_steps --gen_sampling_steps --gen_lr --gen_sum_scale` | RDDM generator | 5 / 5 / 2e-4 / 0.01 |
| `--gen_debug_dir` | dump every redegraded image x_sd | off |
| `--iterations`, `--patch_size`, `--lr` | adaptation loop | 1 / 320 / 2e-4 |
| `--dataset_seed` | the original re-seeds all RNGs to 42 inside the dataset ctor; `-1` disables | 42 |

Swapping the generator: anything with `fit_and_sample(x_in, y_bar, name) -> x_sd` works in place of
`ResidualDegradationGenerator` (e.g. residual recombination, a CNN residual predictor). Swapping the
backbone: wrap it so that `model(x) -> tensor` and pass the right `--train_prefixes`.
