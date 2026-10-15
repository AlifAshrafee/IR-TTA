#!/usr/bin/env bash
# IR-TTA run script. From the repo root (inside screen). Finished runs are skipped, so stages can be re-run.
#
#   bash scripts/run.sh test                                 # CPU self-tests (after any code change)
#   GPU=0 bash scripts/run.sh base agg                       # input, source, source applied twice (dev backbones)
#   GPU=1 bash scripts/run.sh dctta agg                      # DCTTA, PromptIR (only sets not run before)
#
# Stages
#   test   CPU self-tests
#   base   input (once), source and source applied twice, for every backbone in $BBS, test + validation sets
#   dctta  DCTTA as released, PromptIR, test + validation sets (~1 GPU-day for a full suite)
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
