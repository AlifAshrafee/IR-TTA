#!/usr/bin/env bash
# IR-TTA run script. From the repo root (inside screen). Finished runs are skipped, so stages can be re-run.
#
#   bash scripts/run.sh test                                 # CPU self-tests (after any code change)
#   GPU=0 bash scripts/run.sh base agg                       # input, source, source applied twice (dev backbones)
#   GPU=1 bash scripts/run.sh dctta agg                      # DCTTA, PromptIR (only sets not run before)
#   GPU=0 bash scripts/run.sh probe                          # Q1 spectral-gain oracle + Q2a scale-teacher test
#   GPU=0 bash scripts/run.sh ref agg                        # HIDE source rows for all six 5-task backbones
#   GPU=0 TSETS="Rain100H_val HIDE_val" bash scripts/run.sh tta_a     # loss study, stage A (split sets over GPUs)
#   GPU=0 EXP=tta bash scripts/run.sh agg                    # -> results/tta/table_5task_promptir.md
#
# Stages
#   test   CPU self-tests
#   base   input (once), source and source applied twice, for every backbone in $BBS, test + validation sets
#   dctta  DCTTA as released, PromptIR, test + validation sets (~1 GPU-day for a full suite)
#   probe  Q1 + Q2a (tools/scale_probe.py, no training), backbones in $BBS, test + validation sets;
#          report: results/probe/probe_<setting>.md
#   ref    source only, HIDE, all six 5-task backbones (the co-author's reference table has no HIDE column)
#   tta_a  loss study A, PromptIR 5-task validation sets ($TSETS): which pair signal (v0.2 'all', v0 'gain' on rain,
#          ABC-selected pairs) moves the model; + source/input rows in results/tta
#   tta_a2 A4/A5: 'camera' matching (v0's gain for rain/haze/blur/noise, fit for under-exposure), +/- ABC
#   tta_a4 A7 = $BASE: camera matching + ABC + under-exposure threshold (--dark_q 0.05)
#   tta_d  A8 = $BASE + --nss_color + --dark_anchor
#   tta_c  C1 = $BASE + ABC tolerance (--abc_eps_q 0.9); C2 = C1 + L2 anchor  [falsified by the A7 report; kept]
#   tta_b  loss study B: the regularisers on top of $BASE (default: ABC-selected pairs), one at a time, then all
#   agg    results/$EXP/summary.csv + tables (table_<setting>_<backbone>.md, table_backbones.md)
#   guard  DCTTA regression guard (adapt.py vs harness, Rain100H 3-task) -- only after touching dctta/
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
EXP=${EXP:-suite}
SEED=${SEED:-23}
VALN=${VALN:-300}          # validation sets: adapt and evaluate on the same evenly spaced 300 images
NW=${NW:-8}
BBS=${BBS:-"promptir r2r"} # development backbones; all: "promptir adair bioir moceir mirage r2r"
SETTINGS=${SETTINGS:-"5task 3task"}

# released checkpoints (see README "Backbones"); a missing file skips that backbone/setting with a note
declare -A CK=(
  [promptir:3task]=pretrain/model.ckpt                              [promptir:5task]=pretrain/epoch=80.ckpt
  [adair:3task]=pretrain/adair/adair3d.ckpt                         [adair:5task]=pretrain/adair/adair5d.ckpt
  [bioir:3task]=pretrain/bioir/BioIR3D.ckpt                         [bioir:5task]=pretrain/bioir/BioIR5D.ckpt
  [moceir:3task]=pretrain/moceir/MoCE_IR_AIO3/last.ckpt             [moceir:5task]=pretrain/moceir/MoCE_IR_AIO5/last.ckpt
  [mirage:3task]=pretrain/mirage/3Deg_small.ckpt                    [mirage:5task]=pretrain/mirage/5Deg_small.ckpt
  [r2r:3task]=pretrain/r2r/train_ckpt_3D_finetune/last.ckpt         [r2r:5task]=pretrain/r2r/train_ckpt_5D_finetune/last.ckpt
)
declare -A BOPTS=( [moceir]="routing_noise=False" )   # deterministic routing (the official router adds noise at test time)

# each source is evaluated on its own families only (O-HAZY is the haze validation set and a flagged test set)
declare -A TARGETS=(
  [3task]="Rain100H SOTS_Indoor DenseHaze OHAZY Kodak24_Poisson Rain100H_val"
  [5task]="Rain100H SOTS_Indoor DenseHaze OHAZY Kodak24_Poisson HIDE LOLv2 LOLv2_Syn Rain100H_val HIDE_val LOLv2_val"
)

run() { CUDA_VISIBLE_DEVICES=$GPU python run_tta.py --num_workers $NW --exp "$EXP" --seed $SEED "$@"; }
size_args() { [[ "$1" == *_val ]] && echo "--max_images $VALN" || echo ""; }
# model args for backbone $1, setting $2 (fails with a note if the checkpoint is missing)
model_args() {
  local ck=${CK[$1:$2]:-}
  if [[ -z "$ck" || ! -e "$ck" ]]; then echo "[skip] $1 $2: checkpoint '$ck' not found" >&2; return 1; fi
  local o=${BOPTS[$1]:-}
  echo "--backbone $1 --ckpt $ck --source $2${o:+ --backbone_opts $o}"
}

stage_test() {
  CUDA_VISIBLE_DEVICES= python tests/test_harness.py
}

stage_base() {
  local B S M T
  for B in $BBS; do for S in $SETTINGS; do
    M=$(model_args $B $S) || continue
    for T in ${TARGETS[$S]}; do
      [[ $B == promptir ]] && run --method identity $M --target $T $(size_args $T)   # model-independent
      run --method source           $M --target $T $(size_args $T)
      run --method flow --n_apply 2 $M --target $T $(size_args $T)
    done
  done; done
}

stage_dctta() {
  local S M T
  for S in $SETTINGS; do
    M=$(model_args promptir $S) || continue
    for T in ${TARGETS[$S]}; do run --method dctta $M --target $T $(size_args $T); done
  done
}

stage_probe() {
  local B S M T
  for B in $BBS; do for S in $SETTINGS; do
    M=$(model_args $B $S) || continue
    for T in ${TARGETS[$S]}; do
      CUDA_VISIBLE_DEVICES=$GPU python tools/scale_probe.py --num_workers $NW --exp probe $M --target $T $(size_args $T)
    done
  done; done
  python tools/scale_probe.py --report results/probe > /dev/null
  echo "tables: results/probe/probe_*.md"
}

stage_ref() {
  local B M
  for B in promptir adair bioir moceir mirage r2r; do
    M=$(model_args $B 5task) || continue
    run --method source $M --target HIDE
  done
}

# ---- loss study (results/tta, PromptIR 5-task, validation sets only; mixes 0.5 and 1 evaluated for free) ----
TSETS=${TSETS:-"Rain100H_val HIDE_val LOLv2_val OHAZY"}
BASE=${BASE:-"--families all --abc_k 8 --exposure_match camera --dark_q 0.05"}
tta() {   # tta <tag> <flags...>: one transfer run per set in $TSETS
  local tag=$1; shift
  local M T
  M=$(model_args promptir 5task) || return 0
  for T in $TSETS; do
    CUDA_VISIBLE_DEVICES=$GPU python run_tta.py --num_workers $NW --exp tta --seed $SEED $M --target $T $(size_args $T) \
      --method transfer --max_steps 500 --eval_alphas 0.5 1.0 --log_stats --tag $tag "$@"
  done
}

stage_tta_a() {
  local M T
  M=$(model_args promptir 5task) || return 0
  for T in $TSETS; do
    CUDA_VISIBLE_DEVICES=$GPU python run_tta.py --num_workers $NW --exp tta --seed $SEED $M --target $T $(size_args $T) --method source
    CUDA_VISIBLE_DEVICES=$GPU python run_tta.py --num_workers $NW --exp tta --seed $SEED $M --target $T $(size_args $T) --method identity
  done
  tta A0 --families all                                   # v0.2 pairs, nothing else (reference)
  tta A3 --families all --abc_k 8                         # + ABC selection toward the target's NSS
  if [[ " $TSETS " == *" Rain100H_val "* ]]; then          # where did v0's +2.5 dB on rain go?
    local TSETS=Rain100H_val
    tta A1 --families rain --exposure_match gain          # v0 matching, rain pairs only
    tta A2 --families rain                                # v0.2 matching, rain pairs only
  fi
}

stage_tta_a2() {   # the physical matching rule, with and without ABC
  tta A4 --families all --abc_k 8 --exposure_match camera
  tta A5 --families all --exposure_match camera
}

stage_tta_a4() {   # A4 + under-exposure threshold (LOLv2 fix; identical to A4 on bright targets)
  tta A7 $BASE
}

stage_tta_d() {   # A8: + colour NSS features (haze vs blur) + source-anchored exposure for under-exposed targets
  tta A8 $BASE --nss_color --dark_anchor
}

stage_tta_c() {   # ABC with a tolerance: adapt only on pairs that are plausible target samples
  tta C1 $BASE --abc_eps_q 0.9
  tta C2 $BASE --abc_eps_q 0.9 --anchor l2 --anchor_lambda 0.001
}

stage_tta_b() {
  tta B1 $BASE --cons_weight 0.5                          # target consistency, all bands
  tta B2 $BASE --cons_weight 1 --cons_band high           # target consistency, high band only
  tta B3 $BASE --anchor l2 --anchor_lambda 0.001          # L2-SP on the adapter (check 'penalty' vs 'sup' in run.log)
  tta B4 $BASE --nss_weight 0.02                          # NSS output prior toward clean images
  tta B5 $BASE --redeg_weight 0.5                         # re-degradation consistency (SRTTA-style)
  tta B6 $BASE --cons_weight 1 --cons_band high --anchor l2 --anchor_lambda 0.001 --nss_weight 0.02 --redeg_weight 0.5
}

stage_agg() {
  python tools/aggregate.py results/$EXP > /dev/null
  python tools/table.py results/$EXP > /dev/null
  echo "tables: results/$EXP/table_*.md"
}

stage_guard() {
  local LQ GT
  LQ=$(python -c "from irtta.data import resolve_spec; print(resolve_spec('Rain100H').lq)")
  GT=$(python -c "from irtta.data import resolve_spec; print(resolve_spec('Rain100H').gt)")
  CUDA_VISIBLE_DEVICES=$GPU python adapt.py --lq_dir "$LQ" --gt_dir "$GT" --ckpt "${CK[promptir:3task]}" --gpu "$GPU" \
      --results_dir results/guard_adapt_py --no_save_images --num_workers $NW | tail -1 | tee results/guard_adapt.txt
  CUDA_VISIBLE_DEVICES=$GPU python run_tta.py --num_workers $NW --exp guard --method dctta --target Rain100H \
      --ckpt "${CK[promptir:3task]}" --seed 23 --dataset_seed 42 --overwrite | tail -1 | tee results/guard_harness.txt
  if diff -q results/guard_adapt.txt results/guard_harness.txt >/dev/null; then echo "GUARD OK"; else echo "GUARD MISMATCH"; fi
}

mkdir -p results
for s in "$@"; do echo "=== $s ==="; "stage_$s"; done
