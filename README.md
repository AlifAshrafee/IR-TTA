# Test-time adaptation of all-in-one image restoration under degradation shift

All-in-one (AiO) restoration models are trained on a fixed set of degradation operators. At test time the operator
itself can change (heavier rain, real instead of synthetic haze, Poisson instead of Gaussian noise, defocus instead
of motion blur). This repository adapts a pretrained AiO model to an unlabelled test set by **inferring the test
set's degradation process with a physical simulator** and fine-tuning the model on synthetic pairs drawn from that
inferred process. It works as a plug-in for six AiO backbones and includes the DCTTA baseline.

## Method

Given a pretrained AiO model `f_θ0`, the unlabelled degraded test images and a generic clean image corpus (DIV2K):

1. **Degradation inference (set level).** Natural-scene statistics of the test images (BRISQUE-style MSCN moments
   at two scales, dark-channel ratio, saturation; `irtta/nss.py`) define a Gaussian `N_T` of the test set.
2. **Pairs per step.** Clean crops are degraded by a library of physical operators restricted to the families the
   model was trained on (rain, haze, blur, Poisson-Gaussian noise, under-exposure; `irtta/degrade.py`).
   * *ABC selection:* for each clean crop, `k = 8` candidate degradations are drawn from the operator prior and the
     one closest to `N_T` (Mahalanobis distance) is kept -- an approximate posterior sample of the test operator.
   * *Camera-consistent exposure:* a degraded crop and its clean target share one camera gain that matches the
     brightness of the current test patch; an under-exposure operator solves its own gain instead.
   * *Under-exposure rule:* if the frozen pretrained model brightens a test patch by at least √2, the patch is
     under-exposed and the clean targets take the brightness of the pretrained output (exposure cannot be
     identified from the images).
3. **Adaptation.** Zero-initialised LoRA (rank 16) on every 1x1 convolution, L1 loss, Adam (lr 1e-4), 500 steps;
   the adapted model then restores every test image.

## Setup

```bash
pip install -r requirements.txt
bash scripts/run.sh test            # CPU self-tests
```

* **Checkpoints:** the released AiO checkpoints, laid out as in `pretrain/README.md`.
* **Data:** `../AiO-Datasets/{Test,Train}/<task>/<set>/{LQ,GT}` with matching file names (registry:
  `configs/datasets.json`). Clean images: DIV2K train HR (`Train/deblur/DIV2K/DIV2K_train_HR`).
  DPDD: `tools/prepare_dpdd.py` pairs its file names. BSD68 noisy images are synthesised (fixed seed).

## Experiments

```bash
GPU=0 bash scripts/run.sh source ours       # pretrained models and ours, six backbones x {3-task, 5-task}
GPU=1 bash scripts/run.sh dctta             # the DCTTA baseline
GPU=0 SEED=24 BBS="promptir r2r" bash scripts/run.sh ours   # extra seeds
GPU=0 bash scripts/run.sh noshift controls ablation oracle
bash scripts/run.sh tables                  # -> results/suite/table_*.md
```

| stage | what |
|---|---|
| `source`, `ours`, `dctta` | the main tables: test sets as columns, per backbone pretrained / + DCTTA / + ours |
| `noshift` | the sources' own test distributions (BSD68, Rain100L, SOTS-Outdoor, GoPro, LOL-v1) |
| `controls` | PromptIR 3-task: generic synthetic pairs, conditioning on another test set, random initialisation |
| `ablation` | ours without ABC selection, colour statistics, exposure matching, or the under-exposure rule |
| `oracle` | PromptIR 5-task blur / low-light: ours vs the right family vs real pairs of the test operator |

A single run:

```bash
python run_tta.py --backbone adair --ckpt pretrain/adair/adair5d.ckpt --target DPDD --tag F1   # ours
python run_tta.py --method source --backbone adair --ckpt pretrain/adair/adair5d.ckpt --target DPDD
python run_tta.py --method dctta  --backbone adair --ckpt pretrain/adair/adair5d.ckpt --target DPDD
```

Each run writes `results/<exp>/<run_id>/{run.log, config.json, result.json, per_image.csv}`; finished runs are
skipped (`--overwrite` re-runs). Rain sets are scored in BT.601 Y-PSNR (the deraining convention), all others in
RGB PSNR / SSIM.

## Layout

```
run_tta.py                  one run: adapt, evaluate, write result.json
irtta/
  methods.py                identity (input), source (pretrained), transfer (ours), dctta (baseline)
  degrade.py                physical degradation operators, camera-consistent exposure, ABC candidate selection
  nss.py                    natural-scene statistics and Gaussian models
  subsets.py                LoRA, DCTTA's parameter rule, all parameters
  data.py                   dataset registry, target patches, evaluation sets, clean bank, real-pair bank
  evaluation.py             PSNR / SSIM / Y-PSNR
  config.py                 arguments (defaults = the final method) and run ids
  backbones/                PromptIR, AdaIR, Bio-IR, MoCE-IR, MIRAGE, R2R (official parameter names)
  baselines/dctta/          the released DCTTA algorithm (EMA teacher, RDDM re-degradation, TIPS)
configs/datasets.json       dataset registry
scripts/run.sh              all experiments
tools/aggregate.py          result.json files -> runs.csv / summary.csv
tools/table.py              summary.csv -> paper tables
tools/prepare_dpdd.py       DPDD file-name pairing
tests/test_harness.py       CPU self-tests
```

Notes: MoCE-IR runs with deterministic routing (`routing_noise=False`); R2R retrieves blind (no task label).
