#!/usr/bin/env bash
# All experiments of the paper. Run from the repository root; finished runs are skipped, so stages can be re-run.
#
#   bash scripts/run.sh test                          # CPU self-tests
#   GPU=0 bash scripts/run.sh source ours tables      # pretrained models and ours on every test set
#   GPU=1 BBS="promptir adair" bash scripts/run.sh dctta
#   GPU=0 SEED=24 BBS="promptir r2r" bash scripts/run.sh ours    # extra seeds
#
# Stages
#   test      CPU self-tests (tests/test_harness.py)
#   source    the input (PromptIR row only) and every pretrained model on the test and no-shift sets
#   ours      ours (tag F1) on the test sets, $BBS x $SETTINGS
#   dctta     the DCTTA baseline on the test sets, $BBS x $SETTINGS
#   noshift   ours (+ DCTTA for PromptIR) on the sources' own test distributions (no degradation shift)
#   controls  PromptIR 3-task: generic pairs (G1), conditioning on another test set (S1), random initialisation
#             (Z500, Z5000) and random initialisation with generic pairs (ZG500)
#   ablation  PromptIR, both settings: ours without ABC selection / colour statistics / exposure matching /
#             the under-exposure rule (5-task)
#   oracle    results/oracle, PromptIR 5-task blur / low-light: ours, the right family only, real pairs
#   variants  candidate refinements of the ABC selection, $BBS x $SETTINGS on the test sets (compare with
#             tools/compare.py): V1fill (Gaussian from 256 crops on small sets), V2strat (every family among the
#             candidates), V3content (content-corrected distance), V4all (the three together)
#   count     results/count, 5-task: adapted parameters that receive a gradient, ours and DCTTA (table_cost)
#   tables    aggregate + paper tables (results/suite/table_*.md)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
EXP=${EXP:-suite}
SEED=${SEED:-23}
NW=${NW:-8}
BBS=${BBS:-"promptir adair bioir moceir mirage r2r"}
SETTINGS=${SETTINGS:-"3task 5task"}
TAG=F1

# released checkpoints (pretrain/README.md); a missing file skips that backbone / setting with a note
declare -A CK=(
  [promptir:3task]=pretrain/model.ckpt                              [promptir:5task]=pretrain/epoch=80.ckpt
  [adair:3task]=pretrain/adair/adair3d.ckpt                         [adair:5task]=pretrain/adair/adair5d.ckpt
  [bioir:3task]=pretrain/bioir/BioIR3D.ckpt                         [bioir:5task]=pretrain/bioir/BioIR5D.ckpt
  [moceir:3task]=pretrain/moceir/MoCE_IR_AIO3/last.ckpt             [moceir:5task]=pretrain/moceir/MoCE_IR_AIO5/last.ckpt
  [mirage:3task]=pretrain/mirage/3Deg_small.ckpt                    [mirage:5task]=pretrain/mirage/5Deg_small.ckpt
  [r2r:3task]=pretrain/r2r/train_ckpt_3D_finetune/last.ckpt         [r2r:5task]=pretrain/r2r/train_ckpt_5D_finetune/last.ckpt
)
declare -A BOPTS=( [moceir]="routing_noise=False" )   # deterministic routing (the official router adds noise)

declare -A TESTS=(
  [3task]="Rain100H SOTS_Indoor DenseHaze OHAZY Kodak24_Poisson"
  [5task]="Rain100H SOTS_Indoor DenseHaze OHAZY Kodak24_Poisson HIDE LOLv2_Syn DPDD"
)
declare -A NOSHIFT=(
  [3task]="BSD68_s15 BSD68_s25 BSD68_s50 Rain100L SOTS_Outdoor"
  [5task]="BSD68_s25 Rain100L SOTS_Outdoor GoPro LOL"
)

run() { CUDA_VISIBLE_DEVICES=$GPU python run_tta.py --num_workers $NW --exp "$EXP" --seed $SEED "$@"; }
size_args() { [[ "$1" == *_val ]] && echo "--max_images 300" || echo ""; }
model_args() {   # backbone $1, setting $2 (fails with a note if the checkpoint is missing)
  local ck=${CK[$1:$2]:-}
  if [[ -z "$ck" || ! -e "$ck" ]]; then echo "[skip] $1 $2: checkpoint '$ck' not found" >&2; return 1; fi
  local o=${BOPTS[$1]:-}
  echo "--backbone $1 --ckpt $ck --source $2${o:+ --backbone_opts $o}"
}
ours() { run "$@" --tag $TAG; }   # the defaults of run_tta.py are the final method

stage_test() {
  CUDA_VISIBLE_DEVICES= python tests/test_harness.py
}

stage_source() {
  local B S M T
  for S in $SETTINGS; do
    if M=$(model_args promptir $S); then
      for T in ${TESTS[$S]} ${NOSHIFT[$S]}; do run --method identity $M --target $T; done
    fi
    for B in $BBS; do
      M=$(model_args $B $S) || continue
      for T in ${TESTS[$S]} ${NOSHIFT[$S]}; do run --method source $M --target $T; done
    done
  done
}

stage_ours() {
  local B S M T
  for B in $BBS; do for S in $SETTINGS; do
    M=$(model_args $B $S) || continue
    for T in ${TESTS[$S]}; do ours $M --target $T; done
  done; done
}

stage_dctta() {
  local B S M T
  for B in $BBS; do for S in $SETTINGS; do
    M=$(model_args $B $S) || continue
    for T in ${TESTS[$S]}; do run --method dctta $M --target $T; done
  done; done
}

stage_noshift() {
  local B S M T
  for B in $BBS; do for S in $SETTINGS; do
    M=$(model_args $B $S) || continue
    for T in ${NOSHIFT[$S]}; do
      ours $M --target $T
      if [[ $B == promptir ]]; then run --method dctta $M --target $T; fi
    done
  done; done
}

CSETS="Rain100H SOTS_Indoor DenseHaze OHAZY Kodak24_Poisson"
declare -A SWAP=( [Rain100H]=SOTS_Indoor [SOTS_Indoor]=Kodak24_Poisson [DenseHaze]=Rain100H [OHAZY]=Kodak24_Poisson
                  [Kodak24_Poisson]=SOTS_Indoor )
stage_controls() {
  local M T N
  M=$(model_args promptir 3task) || return 0
  for T in $CSETS; do
    run $M --target $T --tag G1 --abc_k 1 --exposure_match none --dark_rule none
    run $M --target $T --tag S1 --cond_target ${SWAP[$T]}
    for N in 500 5000; do
      run $M --target $T --tag Z$N --max_steps $N --init scratch --subset all --lr 2e-4 --dark_rule none
    done
    run $M --target $T --tag ZG500 --init scratch --subset all --lr 2e-4 --abc_k 1 --exposure_match none \
        --dark_rule none
  done
}

stage_ablation() {
  local S M T
  for S in $SETTINGS; do
    M=$(model_args promptir $S) || continue
    for T in ${TESTS[$S]}; do
      run $M --target $T --tag noABC --abc_k 1
      run $M --target $T --tag noColor --no-nss_color
      run $M --target $T --tag noExposure --exposure_match none
      if [[ $S == 5task ]]; then run $M --target $T --tag noDark --dark_rule none; fi
    done
  done
}

declare -A OTRAIN=( [HIDE_val]=HIDE_train [LOLv2_Syn_val]=LOLv2_Syn_train [DPDD]=DPDD_train )
stage_oracle() {
  local EXP=oracle M T
  M=$(model_args promptir 5task) || return 0
  for T in HIDE_val LOLv2_Syn_val DPDD; do
    run --method source $M --target $T $(size_args $T)
    ours $M --target $T $(size_args $T)
    run $M --target $T $(size_args $T) --tag F1fam --families oracle
    run $M --target $T $(size_args $T) --tag ORpair --oracle_pairs ${OTRAIN[$T]} --abc_k 1 --exposure_match none
  done
}

# V1fill only changes sets with < 128 images (one pass already gives 256 crops otherwise: bit-identical to F1)
FILL_SKIP="SOTS_Indoor HIDE"
stage_variants() {
  local B S M T
  for B in $BBS; do for S in $SETTINGS; do
    M=$(model_args $B $S) || continue
    for T in ${TESTS[$S]}; do
      [[ " $FILL_SKIP " == *" $T "* ]] || run $M --target $T --tag V1fill --abc_fit_passes 16
    done
    for T in ${TESTS[$S]}; do run $M --target $T --tag V2strat --abc_strat; done
    for T in ${TESTS[$S]}; do run $M --target $T --tag V3content --abc_content; done
    for T in ${TESTS[$S]}; do
      run $M --target $T --tag V4all --abc_fit_passes 16 --abc_strat --abc_content
    done
  done; done
}

# The count is architectural (which modules the forward pass uses), so 50 steps on one small set suffice.
stage_count() {
  local EXP=count B M
  for B in $BBS; do
    M=$(model_args $B 5task) || continue
    run $M --target Kodak24_Poisson --max_steps 50 --tag CNT
    run --method dctta $M --target Kodak24_Poisson --max_steps 50 --tag CNT
  done
  python tools/aggregate.py results/count
}

stage_tables() {
  python tools/aggregate.py results/$EXP
  local o=""
  if [[ -d results/oracle ]]; then python tools/aggregate.py results/oracle; o="--oracle results/oracle"; fi
  if [[ -f results/count/runs.csv ]]; then o="$o --count results/count"; fi
  python tools/table.py results/$EXP $o > /dev/null
  echo "tables: results/$EXP/table_*.md"
}

mkdir -p results
for s in "$@"; do echo "=== $s ==="; "stage_$s"; done
