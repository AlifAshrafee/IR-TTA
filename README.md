# IR-TTA — test-time adaptation for all-in-one image restoration

Research code for TTA on all-in-one (AiO) restoration models. It contains:

* **DCTTA** (Tang et al., CVPR 2026), a cleaned, numerically identical rebuild of the released code
  (`adapt.py`, `dctta/`; see `CLEANUP_NOTES.md`): the baseline to beat.
* **Six AiO backbones** with their official checkpoints: PromptIR, AdaIR, BioIR, MoCE-IR, MIRAGE, R2R.
* **The harness** (`run_tta.py`, `irtta/`): one run = (method, backbone, source, target, subset, anchor, regime,
  seed), with fixed RNG handling, retention on the source tasks, cost logging and result tables.

The plan and the current state of the research live in the project doc `ir_tta_build_order.md`.

## Layout

```
run_tta.py                    harness entry point
irtta/
  methods.py                  identity (input), source, flow (source applied repeatedly), transfer (parked), dctta
  subsets.py                  adapted parameters: ln (every LayerNorm), lora<r> (every dense 1x1 conv), full (DCTTA rule)
  anchors.py                  none / L2-to-theta_0 (TIPS only inside DCTTA)
  regimes.py                  transductive / online / episodic loops with cost timing
  evaluation.py               per-image PSNR/SSIM/(LPIPS)/brightness drift; probe
  degrade.py                  physical degradation primitives (rain/haze -> blur -> exposure -> noise)
  nss.py                      differentiable natural-scene statistics + Gaussian models (ABC selection, output prior)
  data.py, config.py          dataset registry (configs/datasets.json), clean content bank; arguments
adapt.py, test.py, evaluate.py   DCTTA and the un-adapted source, standalone (as released)
dctta/
  models/__init__.py          backbone registry: build_model(name, ckpt), TRAIN_PREFIXES
  models/*.py                 PromptIR, AdaIR, BioIR, MoCE-IR, MIRAGE, R2R (official parameter names)
  tta.py, losses.py, rddm/    DCTTA: self-ensemble pseudo-label, RDDM generator, TIPS, EMA teacher
  data.py, metrics.py, utils.py, config.py
configs/datasets.json         dataset registry ({root} = ../AiO-Datasets)
configs/sota_{3,5}task.csv    co-author's evaluation of the SOTA backbones (reference rows; no HIDE: `run.sh ref`)
tools/aggregate.py            result.json files -> runs.csv / summary.csv
tools/table.py                summary.csv -> one table per backbone x setting + cross-backbone summary
tools/check_registry.py       checks every dataset folder (pairs, sizes)
tools/check_backbones.py      load + forward check of the released checkpoints
tools/scale_probe.py          Q1 spectral-gain oracle + Q2a scale-teacher test (evaluation only)
scripts/run.sh                stages: test / base / dctta / probe / ref / agg / guard
tests/test_harness.py         CPU self-tests incl. adapt.py <-> run_tta.py parity
pretrain/README.md            where the checkpoints go
```

## Setup

```bash
pip install -r requirements.txt
# checkpoints: see pretrain/README.md;   data: ../AiO-Datasets/{Test,Train}/<task>/<set>/{LQ,GT}
python tools/check_registry.py              # every dataset pairs up
bash scripts/run.sh test                    # CPU self-tests
```

## Run

```bash
GPU=0 bash scripts/run.sh base agg          # input, source, source applied twice (BBS="promptir r2r" by default)
GPU=1 bash scripts/run.sh dctta agg         # DCTTA on PromptIR
GPU=0 bash scripts/run.sh probe             # Q1 + Q2a -> results/probe/probe_<setting>.md
GPU=0 bash scripts/run.sh ref agg           # HIDE source rows for all six 5-task backbones
GPU=0 bash scripts/run.sh tta_a             # loss study stage A (then tta_b; EXP=tta ... agg for the table)
# single runs
python run_tta.py --method source --target HIDE --backbone r2r --ckpt pretrain/r2r/train_ckpt_5D_finetune/last.ckpt
python run_tta.py --method dctta --target Rain100H --ckpt pretrain/model.ckpt --seed 23 --dataset_seed 42
```

Results: `results/<exp>/<run_id>/{run.log, config.json, result.json, per_image.csv, stats.jsonl}`, tables in
`results/<exp>/table_<setting>_<backbone>.md` and `table_backbones.md`. Finished runs are skipped
(`--overwrite` re-runs).

| flag | values |
|---|---|
| model | `--backbone promptir/adair/bioir/moceir/mirage/r2r`, `--ckpt`, `--backbone_opts k=v ...` (moceir: `routing_noise=False`; r2r: `bank_dir=`, `oracle_label=`), `--source 3task/5task` (inferred from the checkpoint path) |
| `--method` | `identity`, `source`, `flow` (`--ext_alpha`, `--n_apply`), `transfer`, `dctta` |
| `--subset` | `ln`, `lora<r>`, `full`, `all`, `re:<regex>`; `+`-joined |
| `--anchor` | `none`, `l2` (`--anchor_lambda`); `tips` only with `--method dctta` |
| transfer losses | `--exposure_match fit/clean/gain/none`, `--abc_k K` (ABC pair selection toward the target's NSS), `--cons_weight`/`--cons_band all/high` (target consistency to the frozen source), `--redeg_weight` (re-degradation consistency), `--nss_weight` (NSS output prior), `--dark_rule none/threshold/source` + `--dark_action skip/anchor/norm`, `--lowlight_op v1/v2`, `--bank_res native/target`, `--abc_topk`; see `irtta/methods.py`, `irtta/nss.py`, `irtta/degrade.py` |
| `--regime` | `transductive` (`--max_steps`, `--epochs`), `online` / `episodic` (`--steps_per_image`, `--online_predict`) |
| data | `--target <registry name>` or `--lq_dir/--gt_dir`; `--max_images N` adapts and evaluates on the same evenly spaced N images |
| seeds | `--seed S`, `--dataset_seed` (default = seed; the DCTTA reference used 23/42) |

Rules baked in:
- Anything the harness adds (LoRA init, LPIPS, probes) stays off the global RNG.
- With `--seed 23 --dataset_seed 42`, the transductive RNG checksum is 319577, and `--method dctta` equals `adapt.py`.
- Each source is evaluated only on its own degradation families.
- Hyperparameters are selected on the `*_val` entries (official train splits), never on test.
- MoCE-IR runs with deterministic routing.
- R2R retrieves blind (no task label at test time); its official test scripts pass the ground-truth task id instead.

## DCTTA standalone (as released)

```bash
python test.py  --lq_dir <LQ> --gt_dir <GT> --ckpt pretrain/model.ckpt            # source (+ --self_ensemble)
python adapt.py --lq_dir <LQ> --gt_dir <GT> --ckpt pretrain/model.ckpt --results_dir results/x
python adapt.py --backbone r2r --ckpt pretrain/r2r/train_ckpt_3D_finetune/last.ckpt --lq_dir <LQ> --gt_dir <GT>
```

Reference (PromptIR 3-task, Rain100H): source 15.550 dB, DCTTA 21.269 dB (log shipped with the original repo;
paper 15.64 -> 20.21). Its switches (`--no_fisher`, `--teacher_weight`, `--ema_decay`, loss weights,
`--train_prefixes`, generator steps, `--dataset_seed`) default to the released values; see `dctta/config.py`.
