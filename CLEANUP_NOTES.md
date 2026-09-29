# Cleanup notes: what was removed, what was kept, and where the code differs from the paper

Source: `AlifAshrafee/DCTTA@main` (mirror of `tonia86/DCTTA`), commit `0526bf7`. Python:
15.7k lines across 46 files → 1.7k lines across 13 files on the executed path (2.0k with the
optional tools and the equivalence test), roughly a third of which is documentation. Nothing on the executed path of
`train_test_promptir.py` / `test_promptir.py` was changed numerically; everything below either
was unreachable, was a no-op, or was pure I/O.

## 1. Execution trace of the original (what actually runs)

```
train_test_promptir.py::main
  set_seed(23)
  PromptTrainDataset_Simple(lq, gt, 320)        # !! re-seeds random/numpy/torch/cuda with 42
  DataLoader(shuffle, bs=1, drop_last)
  PromptIR(decoder=True)                          # second of two identical PromptIR defs in net/model.py
  load_state_dict(strip "net.", strict=False)
  setup_tta -> tta.configure_model (prefix rule) -> tta.collect_params (weight/bias only) -> Adam(2e-4)
            -> tta.SRTTA(...)                     # deepcopy theta_0, EMA teacher, WaveletTransform(wavelet.mat)
  Degradation(opt) -> RDDM/net.py::ResidualDiffusionModel -> UnetRes(num_unet=1) + ResidualDiffusion(pred_res)
                   -> Trainer(RAdam 2e-4, EMA 0.995/every 10, accelerate)   # random init, no checkpoint
  SRTTA.compute_fisher(train_loader)              # TIPS: one pass over the target set
  for each patch:  SRTTA.__call__ -> test_time_adaptation (1 iteration) -> EMA teacher update
  PairedImageDataset -> full-image inference with the *student* -> skimage PSNR/SSIM -> save PNGs
```

Everything else in the repository is off this path.

## 2. Removed

### Files / directories (never imported by the pipeline)
| path | what it was |
|---|---|
| `RDDM/train.py`, `RDDM/sample.py`, `RDDM/install.yaml`, `RDDM/README.md` | RDDM's own training/sampling CLI and env |
| `RDDM/src/denoising_diffusion_pytorch.py` (919 l.) | Gaussian DDPM; `GaussianDiffusion` imported in `RDDM/net.py` but never used |
| `RDDM/datasets/**`, `RDDM/eval/**`, `RDDM/experiments/**` | RDDM dataset loaders (CelebA/LSUN/FFHQ...), MATLAB eval scripts, DDIM→RDDM conversion |
| `RDDM/**/__pycache__/*.pyc` | committed bytecode |
| `net/network_soda.py` (1193 l.), `net/wat.py`, `net/arch_util.py`, `net/local_arch.py` | SODA / wavelet-attention / NAFNet utilities; `LayerNorm2d`, `Local_Base` imported into `net/model.py` and unused; `network_soda` imports `models.ops.modules.MSDeformAttn`, which does not exist in the repo |
| `utils/dataset_utils_tn.py` (598 l.) | earlier copy of `dataset_utils.py` |
| `utils/degradation_utils.py`, `utils/image_utils.py`, `utils/imresize.py`, `utils/schedulers.py`, `utils/pytorch_ssim/`, `utils/utils_image.py` (888 l.), `utils/image_io.py` (414 l.) | PromptIR training-time degradation synthesis, LR scheduler, KAIR/DIP utilities; only `crop_img`, `save_image_tensor` (2 functions) were used and are re-implemented in `dctta/data.py` / `dctta/utils.py` |
| `utils/utils_tta.py` | SRTTA helpers; `preprocess_img`/`get_paths` reference undefined `blindsr`, `args`; only `augment_transform`/`transform_tensor` used (kept verbatim in `dctta/utils.py`) |
| `pretrain/wavelet.mat` | replaced by the analytic 2×2 Haar low-pass (see §4) |
| `train.log`, `results/**`, `images/**` | run artefacts |
| `options.py` | replaced by `dctta/config.py` with only the options that are read (`--de_type`, `--epochs`, `--crop`, `--data_file_dir`, `--denoise_dir`, `--derain_dir`, `--dehaze_dir`, `--output_path`, `--ckpt_path`, `--wblogger`, `--ckpt_dir`, `--num_gpus`, `--save_results`, `--save_dir`, `--cuda` were dead) |

### Inside kept files
| location | removed | reason |
|---|---|---|
| `net/model.py` | first `PromptIR` definition (l.260) and its helper classes; `AirNet`, `MoCo`, `CBDE`, `DGRN`, `DCN_layer`, `AdaIR` + its blocks, `StudentModel`, `Restormer`, `ChannelShuffle_skip_textguaid`, `Topm_CrossAttention_Restormer`, `ch_shuffle_high_text` | Python binds `PromptIR` to the *second* definition (l.1716); the two are numerically identical (the first also had a `return_prompts` hook). `AirNet` needs `mmcv.ops.modulated_deform_conv2d` and no AirNet weights ship; `StudentModel` references an undefined `WAT`; nothing else is instantiated. Imports of `clip`, `huggingface_hub`, `mmcv`, `pdb`, `time`, `sys`, `os`, `torchvision.transforms` dropped |
| `tta.py` | `airnet==1` and `text_code` (DFPIR) branches of `test_time_adaptation`/`compute_fisher`; `resume`, `save(dfpir=…)`, `load_model_and_optimizer`; the second (commented) `configure_model`; ~60 commented lines; imports of `matplotlib`, `subprocess`, `itertools.cycle`, `utils_image`, `image_io`, `F`, `transforms` | duplicates of the PromptIR path with a different call signature |
| `tta.py::test_time_adaptation` | the `for _ in range(4)` loop that repeated the 8-way self-ensemble four times, the `torch.var` over the four copies and the `confidence = 3/2 - sigmoid(var/4e-4)` map | the four repeats are deterministic copies → `var == 0` → `confidence == 1.0` exactly; the map multiplied both sides of the L1. Saves 24 teacher forward passes per patch |
| `tta.py::test_time_adaptation` | `tmp = self.model_teacher(RPF(degrad_patch))` | unused extra forward pass |
| `tta.py::__call__` | `self.model.eval(); restored = self.model(input_img)` after each step | return value discarded by the caller |
| `tta.py::compute_loss2` | construction of `PerceptualLoss()` (VGG19 load from disk) on **every call** (2× per step) | hoisted to `DCTTA.__init__`; same fixed weights |
| `tta.py::compute_fisher` | `fisher_optimizer = Adam(model.parameters())` used only for `zero_grad()`; `return fisher` (returned the last tensor) | `model.zero_grad(set_to_none=True)` is equivalent |
| `train_test_promptir.py` | `MSELoss`, `TVLoss`, `VGGLoss` (VGG16), `GANLoss`, `Generator`, `ResBlock`; imports of `wandb`, `lightning`, `LinearWarmupCosineAnnealingLR`, `StudentModel`, `PromptTrainDataset`, `init`; `origin_model = deepcopy(model)`; hard-coded `/data2/tn/...` paths and `CUDA_VISIBLE_DEVICES="4"`; `subprocess mkdir` | all unused or hard-coded |
| `test_promptir.py` | same as above; the file was a copy of `train_test_promptir.py` with the adaptation loop deleted but still importing `tta`, `RDDM`, building the optimiser, etc. | |
| `RDDM/net.py::ResidualDiffusionModel` | `original_ddim_ddpm`, `input_condition(_mask)`, `debug`, `cuda_device`, `sampling_timesteps_original_ddim_ddpm`, `test_res_or_noise`, `num_unet`; `__main__` demo with `../../../DATA/HSTS` paths | fixed to the values used |
| `RDDM/.../UnetRes` | whole class | with `num_unet=1, objective='pred_res'` its forward was `[unet0(x, time[0])]`; folded into `ResidualDiffusion` (parameter names lose the `unet0.` prefix; no RDDM checkpoint exists so nothing breaks) |
| `RDDM/.../Unet` | `self_condition`, `learned_sinusoidal_cond`, `random_fourier_features`, `learned_variance`, `input_condition`, `RandomOrLearnedSinusoidalPosEmb` | never enabled; module creation order unchanged → identical init under the same seed |
| `RDDM/.../ResidualDiffusion` | objectives `pred_res_noise`, `pred_x0_noise`, `pred_noise`; `p_sample_loop` (ancestral sampler), `p_sample`, `p_mean_variance`, `q_posterior`, `predict_start_from_*`, `q_posterior_from_res_noise`; `gen_coefficients`, `betas_for_alpha_bar`, the `convert_to_ddim=False` and non-`linear` schedule branches; `u_loss`; `init()`; `ddim_sample` `pred_type` variants other than `use_pred_noise`; buffers `betas`, `one_minus_alphas_cumsum`, `posterior_*` | `sampling_timesteps(5) < timesteps(1000)` always selects DDIM; only `pred_res` is configured; `init()` re-registered an identical schedule except `alphas[0]`, `betas2[0]` and `one_minus_alphas_cumsum[-1]`, none of which DDIM/`pred_res` reads |
| `RDDM/.../Trainer` | `accelerate.Accelerator` (single process, `mixed_precision='no'`, `amp=False` → `backward`, `clip_grad_norm_`, `autocast` all reduce to plain torch), `save()`/`load()`, `sample()` (references non-existent `self.sample_loader`), `set_results_folder`, `Augmentor`/`cv2`/`glob`/`Path` imports | `Trainer.train` called `save(milestone)` (a `torch.save` of the whole generator) and `test()` wrote every intermediate DDIM state as a PNG **for every test image**; replaced by an optional `debug_dir` that dumps only x_sd |
| `RDDM/.../Trainer.__init__` | `adam_betas=(0.9, 0.99)` argument | it was never passed to `RAdam`, which therefore used its defaults (0.9, 0.999); kept the defaults |
| `utils/loss_utils.py` | `GANLoss`, `WaveletTransform` for `scale != 1` / `dec=False` | unused |
| `utils/dataset_utils.py` | `PromptTrainDataset`, `DenoiseTestDataset`, `DerainDehazeDataset`, `TestSpecificDataset`, `SimpleImagePairDataset`, two conflicting `crop_patch` definitions | unused |
| `utils/val_utils.py` | `accuracy`, `compute_psnr_ssim_single`, `compute_niqe` (`skvideo`), `timer` | unused |
| `test_metrics.py` | hard-coded paths, `basicsr.img2tensor`, logger boilerplate | kept as `tools/iqa_metrics.py` with CLI args (optional, needs `pyiqa`) |

Dependencies dropped: `accelerate`, `Augmentor`, `basicsr`, `clip`, `huggingface_hub`, `lightning`, `matplotlib`,
`mmcv`, `opencv`, `scipy`, `scikit-video`, `timm`, `wandb`, `pyiqa` (optional now). Kept: `torch`, `torchvision`,
`numpy`, `pillow`, `scikit-image`, `einops`, `ema-pytorch`, `tqdm`.

## 3. Kept exactly (and why it matters)

* **Construction order** `set_seed(23)` → dataset (re-seeds to 42) → loader → PromptIR → DCTTA → generator →
  Fisher pass → adaptation loop. The generator is randomly initialised, so the RNG stream up to its construction
  determines its weights; the dataset's global re-seed is therefore kept (`--dataset_seed -1` to drop it).
* `ema_pytorch.EMA(beta=0.995, update_every=10)` with the library's default `update_after_step=100` and warm-up
  schedule. Effective decay = `min(0.995, 1-(1+k)^(-2/3))`, `k = step-101`: the EMA is a plain copy for the first
  100 generator steps (20 images) and reaches only ≈0.98 by step 500. Re-implementing "EMA 0.995" by hand would
  change the generator's behaviour.
* `loss / 2` in the generator (the `gradient_accumulate_every=2` divisor applied without accumulation),
  `clip_grad_norm_(1.0)`, RAdam defaults, `torch.randint` for t then `randn_like` for noise (RNG order).
* Per-tensor TIPS mask, restoration after every step, Fisher computed with the un-adapted student on random
  320-patches from the same loader (so the Fisher pass consumes one epoch of the sampler's RNG before adaptation).
* Both loss terms back-propagate only through `f_theta(x_sd)`; `f_theta(x_in)` is detached.
* Evaluation with the student, centre crop to a multiple of 16, skimage PSNR/SSIM in RGB.

Two things that are equivalent but not bit-identical by construction:
the Haar low-pass is now computed separately for each loss term instead of once per tensor (same values, different
autograd graph shape → gradient differences at float round-off), and the self-ensemble uses `torch.rot90` /
`torch.flip` instead of torchvision `RandomRotation((k·90, k·90))` / `RandomHorizontalFlip(p=1)` (pixel-exact on
square patches; `tests/test_equivalence.py` asserts equality).

## 4. Facts about the released code that the paper does not state

These are all preserved as defaults and exposed as flags. They matter for the go/no-go experiments.

| topic | paper | released code |
|---|---|---|
| pseudo-label loss weight α | 1 | `--teacher_weight 5` |
| teacher EMA η | 0.95 | `E_decay = 0.45` (student lags the teacher by ~2 steps, not ~20) |
| loss | L1 + 0.01·VGG | L1 + 0.01·VGG19(5 layers, weights 0.1/0.1/1/1/1, L1 on features) + **0.1·L1 on the 2×2 Haar LL band** |
| what adapts | "TIPS selects degradation-sensitive parameters" | hard prefix filter first: only `prompt{1,2,3}`, `encoder_level{1,2,3}`, `decoder_level{1,2,3}` have `requires_grad`; `latent`, `refinement`, `noise_level*`, all reduce/up/down convs, `patch_embed`, `output` are frozen. Then only leaf tensors named `weight`/`bias` enter the optimiser, so `prompt_param` (the prompt bank) and attention `temperature` never move (`python tools/param_groups.py` prints the exact counts per module) |
| TIPS ratio | top-40 % frozen | `fisher_ratio = 0.6` restores the top-60 % **per tensor** (not a global threshold), so it cannot be a layer selection in disguise; 40 % of every trainable tensor adapts |
| gradient path | L_s between f(x_in) and f(x_sd) | f(x_in) detached; both terms move f(x_sd) only |
| generator | "RDDM ... 5 sampling steps" | trained **from scratch**, online, 5 RAdam steps per test image, weights/optimiser/EMA persisting across the target set; sampled from the EMA copy; 5 DDIM steps, η=0, `sum_scale=0.01` |
| cost | 32.25 s / 23.5 GB for Rain100H | the shipped `train.log` shows ≈12 s per image (100 images ≈ 20 min); the Fisher pass, VGG re-loading and the per-image checkpoint/PNG dumps are inside that |
| iterations | not stated | 1 step per patch, 1 pass over the set, batch 1, 320×320 random crops |
| confidence weighting | not mentioned | `3/2 − sigmoid(var/4e-4)` over 4 identical self-ensemble repeats — identically 1 (removed) |
| GT usage | — | the adaptation dataset loads GT patches (independently cropped) and moves them to GPU; never used |
| seeds | not stated | 23, then 42 (dataset ctor) |

Answers to open questions 1–2 of the project context: the RDDM is initialised from scratch and trained inside the
adaptation loop (5 steps per image, ≈500 steps for Rain100H); TIPS on PromptIR freezes 60 % of each tensor in
the prompt/encoder/decoder blocks — the layer set is fixed by the prefix rule *before* Fisher is involved.

## 5. Bugs fixed (behaviour-neutral)

* `loss_t` was undefined when `teacher_weight <= 0` (NameError); now 0.
* `WaveletTransform` fell back to `./wavelet.mat` when the author's absolute path was absent.
* `torchvision.models.vgg19(pretrained=True)` → `weights=VGG19_Weights.IMAGENET1K_V1` (same weights, non-deprecated API).
* `load_state_dict(strict=False)` now reports missing/unexpected keys instead of hiding them.
