# DCTTA (cleaned) — Degradation-Consistent Test-Time Adaptation for All-in-One Image Restoration

A stripped-down, dependency-light rebuild of the official implementation of
Tang et al., *Degradation-Consistent Test-Time Adaptation for All-in-One Image Restoration*, CVPR 2026
(original: https://github.com/tonia86/DCTTA, built on PromptIR and RDDM).

Only the code that the released pipeline actually executes survives; every numerical choice of that
pipeline is preserved (see `CLEANUP_NOTES.md` for the full removal log, the component map, and the
places where the released code differs from the paper).

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
  models/__init__.py          backbone registry: build_model(name, ckpt), TRAIN_PREFIXES
  models/promptir.py          PromptIR backbone (verbatim)
  models/{adair,bioir,moceir,mirage,r2r}.py   other all-in-one backbones (see "Backbones")
  rddm/unet.py                RDDM U-Net (verbatim blocks)
  rddm/diffusion.py           residual diffusion, pred_res objective, DDIM sampler
  rddm/generator.py           online-trained degradation generator (replaces Trainer + 2 wrappers)
tools/iqa_metrics.py          optional pyiqa evaluation (LPIPS, DISTS, NIQE, MUSIQ, CLIP-IQA)
tools/check_backbones.py      load + forward smoke check of released backbone checkpoints
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
`ResidualDegradationGenerator` (e.g. residual recombination, a CNN residual predictor).

## Backbones

`--backbone {promptir,adair,bioir,moceir,mirage,r2r}` with the authors' released checkpoint in `--ckpt`
(`test.py` and `adapt.py`; default `promptir`, whose path is unchanged). The models in `dctta/models/`
keep the official parameter names, so the released Lightning checkpoints load as-is and the variant
(3/5-task, tiny/small, MoCE-IR vs MoCE-IR-S) is read off the checkpoint. Each file's docstring lists
exactly what differs from the official code. `--train_prefixes` defaults to the backbone's analogue
of DCTTA's PromptIR rule (`dctta.models.TRAIN_PREFIXES`).

| backbone | weights (official links) | caveat |
|---|---|---|
| AdaIR (ICLR'25) | [github.com/c-yn/AdaIR](https://github.com/c-yn/AdaIR) | verbatim |
| BioIR (NeurIPS'25) | [github.com/c-yn/BioIR](https://github.com/c-yn/BioIR) (All-in-one) | verbatim |
| MoCE-IR (CVPR'25) | [github.com/eduardzamfir/MoCE-IR](https://github.com/eduardzamfir/MoCE-IR) | router adds noise even at test time (official behaviour, kept); `--backbone_opts routing_noise=False` makes it deterministic |
| MIRAGE (ICLR'26) | [github.com/Amazingren/MIRAGE](https://github.com/Amazingren/MIRAGE) | BatchNorm + DropPath are kept in eval mode under `model.train()` |
| R2R (CVPR'26) | [github.com/AlifAshrafee/R2R](https://github.com/AlifAshrafee/R2R) | needs its degradation bank (`save_prompts_*/last/*.pt`, found next to the ckpt or `--backbone_opts bank_dir=...`); official test scripts pass the **ground-truth task id** to the retrieval, we retrieve blind by default (`oracle_label=<k>` reproduces the official protocol) |

```bash
python tools/check_backbones.py adair=pretrain/adair/adair3d.ckpt r2r=pretrain/r2r/train_ckpt_3D_finetune/last.ckpt
python test.py  --backbone adair --ckpt pretrain/adair/adair3d.ckpt --lq_dir ... --gt_dir ...
python adapt.py --backbone r2r   --ckpt pretrain/r2r/train_ckpt_3D_finetune/last.ckpt --lq_dir ... --gt_dir ...
```
