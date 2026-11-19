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
#   ablation  $BBS x $SETTINGS: ours without ABC selection (noABC), exposure matching (noExposure) or the
#             under-exposure rule (noDark); variants in $ABL_TAGS, run variant by variant so the first ones finish
#             first. PromptIR also has noColor (colour statistics), kept for the supplementary.
#             e.g. GPU=0 BBS=adair bash scripts/run.sh ablation   (one backbone per GPU)
#   oracle    results/oracle, PromptIR 5-task blur / low-light: ours, the right family only, real pairs
#   count     results/count, 5-task: adapted parameters that receive a gradient, ours and DCTTA (table_cost)
#   tables    aggregate + paper tables (results/suite/table_*.md)
#
# Supplementary (seed 23; finished runs are skipped, so every stage can be re-run)
#   e1        results/supp, $E1_BBS (promptir r2r) x $SETTINGS: supervision vs update constraint (tag E1).
#             E1_PART=dctta: DCTTA's objective on our rank-16 update (lr 1e-4, no TIPS); E1_PART=ours: our pairs
#             on DCTTA's update constraint (--subset dctta = prefix rule + TIPS, lr 2e-4); default both.
#             The other two cells of the 2x2 are the released DCTTA and ours in results/suite.
#   e2        results/supp, 5-task LOLv2-S and DPDD: pretrained and ours ($BBS), DCTTA (PromptIR), re-run for
#             GT-mean PSNR (outputs are not saved; same seed, so PSNR equals results/suite)
#   e3        results/suite, $BBS x $SETTINGS: ABC with diagonal Sigma_T (abcDiag) and Euclidean distance
#             (abcEuclid); ABC_METRICS="diag" or "euclid" runs one of them (one per GPU)
#   e5        (optional) results/suite, $E1_BBS x $SETTINGS: ABC budget k in $ABC_KS (tags abcK2, abcK4, abcK16)
#   supptables  aggregate suite + supp; table.py (E3 rows in the ablation tables) and tools/table_supp.py
#               (results/supp/table_supp_E1.md, _E2.md, _E5.md)
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

# Same tags and flags as the original PromptIR ablation, so its finished runs are reused (skipped), not repeated.
ABL_TAGS=${ABL_TAGS:-"noABC noExposure noDark"}
declare -A ABL_ARGS=( [noABC]="--abc_k 1" [noExposure]="--exposure_match none" [noDark]="--dark_rule none"
                      [noColor]="--no-nss_color" )
stage_ablation() {
  local A B S M T
  for A in $ABL_TAGS; do
    for B in $BBS; do for S in $SETTINGS; do
      M=$(model_args $B $S) || continue
      for T in ${TESTS[$S]}; do run $M --target $T --tag $A ${ABL_ARGS[$A]}; done
    done; done
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

# ---------------------------------------------------------------------------------------------- supplementary
E1_BBS=${E1_BBS:-"promptir r2r"}
E1_PART=${E1_PART:-"dctta ours"}
stage_e1() {
  local EXP=supp B S M T P
  for P in $E1_PART; do
    for B in $E1_BBS; do for S in $SETTINGS; do
      M=$(model_args $B $S) || continue
      for T in ${TESTS[$S]}; do
        case $P in
          dctta) run --method dctta $M --target $T --subset lora16 --lr 1e-4 --tag E1 ;;
          ours)  run $M --target $T --subset dctta --lr 2e-4 --tag E1 ;;
          *) echo "unknown E1_PART '$P' (dctta | ours)" >&2; return 1 ;;
        esac
      done
    done; done
  done
}

stage_e2() {
  local EXP=supp B M T
  for B in $BBS; do
    M=$(model_args $B 5task) || continue
    for T in LOLv2_Syn DPDD; do
      run --method source $M --target $T
      ours $M --target $T
      if [[ $B == promptir ]]; then run --method dctta $M --target $T; fi
    done
  done
}

ABC_METRICS=${ABC_METRICS:-"diag euclid"}
declare -A AM_TAG=( [diag]=abcDiag [euclid]=abcEuclid )
stage_e3() {
  local A B S M T
  for A in $ABC_METRICS; do
    for B in $BBS; do for S in $SETTINGS; do
      M=$(model_args $B $S) || continue
      for T in ${TESTS[$S]}; do run $M --target $T --tag ${AM_TAG[$A]} --abc_metric $A; done
    done; done
  done
}

ABC_KS=${ABC_KS:-"2 4 16"}
stage_e5() {
  local K B S M T
  for K in $ABC_KS; do
    for B in $E1_BBS; do for S in $SETTINGS; do
      M=$(model_args $B $S) || continue
      for T in ${TESTS[$S]}; do run $M --target $T --tag abcK$K --abc_k $K; done
    done; done
  done
}

stage_supptables() {
  EXP=suite stage_tables
  python tools/aggregate.py results/supp
  python tools/table_supp.py results/suite results/supp > /dev/null
  echo "tables: results/supp/table_supp_*.md; E3 rows in results/suite/table_ablation_*.md"
}

mkdir -p results
for s in "$@"; do echo "=== $s ==="; "stage_$s"; done
