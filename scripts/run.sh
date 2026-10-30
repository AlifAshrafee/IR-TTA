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
#   dctta  DCTTA as released, every backbone in $BBS (ONLYTEST=1: tests only; ~1 GPU-day per backbone)
#   dpdd   DPDD (defocus) test: source, F1, DCTTA for the 5-task $BBS
#   noshift  source / ours (+ DCTTA, PromptIR) on each source's own test distributions (no degradation shift)
#   oracle EXP=oracle: F1 / family oracle / pair oracle on HIDE, LOLv2-S, DPDD ($OBBS, 5-task)
#   probe  Q1 + Q2a (tools/scale_probe.py, no training), backbones in $BBS, test + validation sets;
#          report: results/probe/probe_<setting>.md
#   ref    source only, HIDE, all six 5-task backbones (the co-author's reference table has no HIDE column)
#   ours_dark  A9 = A8 + --dark_skip on the 5-task low-light sets (LOLv2_Syn, LOLv2_Syn_val)
#   ours_e E1-E6 (E6 = E4 + motion-blur kernels; source-decided dark rule, brightness-free loss, v2 low-light, bank scale, top-2 ABC),
#          5-task validation sets; ECFGS / ESETS select
#   open3  open set: 3-task sources on HIDE / LOLv2-Synthetic (O1 = all five families, threshold dark rule,
#          natural-exposure clean images, v2 low-light; O2 = O1 + bank scale + motion-blur kernels)
#          + input / source / x2 / DCTTA; table_open_<bb>.md
#   final  F1 (= E1) on the suite for $BBS x $SETTINGS (ONLYTEST=1: tests only)
#   rescore  identity / source / x2 (+ DCTTA) on the rain sets again, to record Y-PSNR
#   controls  G1 generic synthetic FT, S1 mismatched conditioning, Z500/Z5000 random-init (PromptIR 3-task, $CSETS)
#   diag   EXP=diag: source / F1 / G1 / Z500 / ZG500 (random init, generic pairs), PromptIR 3-task, $CSETS, + DCTTA
#          on Rain100H; every run records affine-corrected Y-PSNR (tools/diag.py: photometric vs structural gain)
#   controls5  EXP=diag: the same + S1, Z5000 on PromptIR 5-task, HIDE and LOLv2-Synthetic ($CSETS5)
#   ours   A8 on the suite for $BBS x $SETTINGS (ONLYTEST=1 skips the *_val sets)
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

# each source is evaluated on its own families only (O-HAZY is the haze validation set and a flagged test set).
# LOLv2-Real (test and train) is out: its scenes are LOL-v1 training scenes (tools/overlap_check.py: 91/100)
declare -A TARGETS=(
  [3task]="Rain100H SOTS_Indoor DenseHaze OHAZY Kodak24_Poisson Rain100H_val"
  [5task]="Rain100H SOTS_Indoor DenseHaze OHAZY Kodak24_Poisson HIDE LOLv2_Syn Rain100H_val HIDE_val LOLv2_Syn_val"
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

stage_dctta() {   # every backbone in $BBS (DCTTA's train-prefix rule per backbone: dctta/models TRAIN_PREFIXES)
  local B S M T
  for B in $BBS; do for S in $SETTINGS; do
    M=$(model_args $B $S) || continue
    for T in ${TARGETS[$S]}; do
      [[ $ONLYTEST == 1 && $T == *_val ]] && continue
      run --method dctta $M --target $T $(size_args $T)
    done
  done; done
}

# DPDD (defocus; 5-task sources were trained on GoPro motion blur): source, DCTTA, F1, for $BBS
stage_dpdd() {
  local B M
  for B in $BBS; do
    M=$(model_args $B 5task) || continue
    if [[ $B == promptir ]]; then run --method identity $M --target DPDD; fi
    run --method source $M --target DPDD
    run --method transfer $M --target DPDD --max_steps 500 --tag F1 $(f1_flags 5task)
    run --method dctta $M --target DPDD
  done
}

# ORACLE ladder on blur / low-light (EXP=oracle, $OBBS, 5-task): is the synthetic operator the bottleneck?
#   F1        ours (all families, ABC picks the family and the operator)
#   F1fam     family oracle: only the target's family (blur / low-light), ABC within it
#   ORpair    pair oracle: real pairs of the target's own training split (its exact operator, other images)
OBBS=${OBBS:-promptir}
# oracle pair splits; *_val targets share their folder with *_train -> the evaluated images are excluded
declare -A OTRAIN=( [HIDE]=HIDE_train [LOLv2_Syn]=LOLv2_Syn_train [DPDD]=DPDD_train
                    [HIDE_val]=HIDE_train [LOLv2_Syn_val]=LOLv2_Syn_train )
stage_oracle() {
  local EXP=oracle B M T
  for B in $OBBS; do
    M=$(model_args $B 5task) || continue
    for T in ${OSETS:-HIDE LOLv2_Syn DPDD}; do
      run --method source $M --target $T $(size_args $T)
      run --method transfer $M --target $T $(size_args $T) --max_steps 500 --tag F1 $(f1_flags 5task)
      run --method transfer $M --target $T $(size_args $T) --max_steps 500 --tag F1fam \
          $(f1_flags 5task | sed 's/--families all/--families oracle/')
      if [[ -n "${OTRAIN[$T]:-}" ]]; then
        run --method transfer $M --target $T $(size_args $T) --max_steps 500 --tag ORpair \
            --oracle_pairs ${OTRAIN[$T]} --abc_k 1 --exposure_match none
      else
        echo "[skip] ORpair $T: no oracle split in OTRAIN"
      fi
    done
  done
}

# NO-SHIFT sets: each source's own test distributions (registry _source_sets: BSD68, Rain100L, SOTS-Outdoor,
# + GoPro, LOL-v1 for 5-task) -- the degradation operator is the training one; only content changes.
# Source and ours for $BBS; DCTTA for PromptIR (NOSHIFT_DCTTA=0 skips it).
declare -A NOSHIFT=( [3task]="BSD68_s15 BSD68_s25 BSD68_s50 Rain100L SOTS_Outdoor"
                     [5task]="BSD68_s25 Rain100L SOTS_Outdoor GoPro LOL" )
stage_noshift() {
  local B S M T fam
  for B in $BBS; do for S in $SETTINGS; do
    M=$(model_args $B $S) || continue
    for T in ${NOSHIFT[$S]}; do
      run --method source $M --target $T
      run --method transfer $M --target $T --max_steps 500 --tag F1 $(f1_flags $S)
      if [[ $B == promptir && ${NOSHIFT_DCTTA:-1} == 1 ]]; then run --method dctta $M --target $T; fi
    done
  done; done
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
TSETS=${TSETS:-"Rain100H_val HIDE_val LOLv2_Syn_val OHAZY"}
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

# ---- our method on the suite (results/suite, next to input / source / source x2 / DCTTA / co-author rows) ----
# A8 = transfer + ABC selection (colour NSS) + camera matching + under-exposure rule + source-anchored dark exposure.
# Operators are restricted to the source's own families (3-task: noise, rain, haze; 5-task: all five).
# Pre-declared main number: mix 0.5 (chosen on the validation sets); mix 1 is reported alongside.
ONLYTEST=${ONLYTEST:-0}
stage_ours() {
  local B S M T FAM
  for B in $BBS; do for S in $SETTINGS; do
    M=$(model_args $B $S) || continue
    [[ $S == 3task ]] && FAM="noise+rain+haze" || FAM="all"
    for T in ${TARGETS[$S]}; do
      [[ $ONLYTEST == 1 && $T == *_val ]] && continue
      run --method transfer $M --target $T $(size_args $T) --max_steps 500 --eval_alphas 0.5 1.0 --tag A8 \
          --families $FAM --abc_k 8 --exposure_match camera --dark_q 0.05 --nss_color --dark_anchor
    done
  done; done
}

# A9 = A8 + abstention on under-exposed patches (--dark_skip), 5-task low-light sets only (elsewhere ~ A8)
stage_ours_dark() {
  local B M T
  for B in $BBS; do
    M=$(model_args $B 5task) || continue
    for T in LOLv2_Syn LOLv2_Syn_val; do
      run --method transfer $M --target $T $(size_args $T) --max_steps 500 --eval_alphas 0.5 1.0 --tag A9 \
          --families all --abc_k 8 --exposure_match camera --dark_q 0.05 --nss_color --dark_anchor --dark_skip
    done
  done
}

# E series: low-light and clean-bank ideas, one change at a time on top of A8's core, 5-task validation sets
# (results/suite, next to A8/A9). ECFGS selects configurations, EB the backbones' sets.
ECFGS=${ECFGS:-"E1 E2 E3 E4 E5"}
ESETS=${ESETS:-"Rain100H_val HIDE_val LOLv2_Syn_val OHAZY"}
ecfg() {
  local core="--families all --abc_k 8 --exposure_match camera --nss_color"
  case $1 in
    E1) echo "$core --dark_rule source --dark_action anchor" ;;                       # source decides dark (A8 action)
    E2) echo "$core --dark_rule source --dark_action norm" ;;                         # + brightness-free loss
    E3) echo "$core --dark_rule source --dark_action norm --lowlight_op v2" ;;        # + v2 low-light operator
    E4) echo "$core --dark_rule source --dark_action norm --lowlight_op v2 --bank_res target" ;;      # + bank scale
    E5) echo "$core --dark_rule source --dark_action norm --lowlight_op v2 --bank_res target --abc_topk 2" ;;
    E6) echo "$core --dark_rule source --dark_action norm --lowlight_op v2 --bank_res target --blur_op mixed" ;;  # E4 + motion kernels
  esac
}
stage_ours_e() {
  local B M T E
  for B in $BBS; do
    M=$(model_args $B 5task) || continue
    for E in $ECFGS; do for T in $ESETS; do
      run --method transfer $M --target $T $(size_args $T) --max_steps 500 --eval_alphas 0.5 1.0 --tag $E $(ecfg $E)
    done; done
  done
}

# Open set (TAO-style): 3-task sources (no blur / low-light training) on HIDE and LOLv2-Synthetic, operators from
# all five families. Validation sets first (OSETS); add the tests with OSETS="HIDE LOLv2_Syn HIDE_val LOLv2_Syn_val".
OSETS=${OSETS:-"HIDE_val LOLv2_Syn_val"}
stage_open3() {
  local B M T
  for B in $BBS; do
    M=$(model_args $B 3task) || continue
    for T in $OSETS; do
      if [[ $B == promptir ]]; then run --method identity $M --target $T $(size_args $T); fi
      run --method source $M --target $T $(size_args $T)
      run --method flow --n_apply 2 $M --target $T $(size_args $T)
      run --method transfer $M --target $T $(size_args $T) --max_steps 500 --eval_alphas 0.5 1.0 --tag O1 \
          --families all --abc_k 8 --exposure_match camera --nss_color \
          --dark_rule threshold --dark_q 0.05 --dark_action natural --lowlight_op v2
      run --method transfer $M --target $T $(size_args $T) --max_steps 500 --eval_alphas 0.5 1.0 --tag O2 \
          --families all --abc_k 8 --exposure_match camera --nss_color \
          --dark_rule threshold --dark_q 0.05 --dark_action natural --lowlight_op v2 --bank_res target --blur_op mixed
      if [[ $B == promptir ]]; then run --method dctta $M --target $T $(size_args $T); fi
    done
  done
}

# ---- final configuration F1 (= E1: ABC + camera exposure + colour NSS + source-decided under-exposure, anchor) ----
f1_flags() {   # $1 = setting
  local fam; [[ $1 == 3task ]] && fam="noise+rain+haze" || fam="all"
  echo "--families $fam --abc_k 8 --exposure_match camera --nss_color --dark_rule source --dark_action anchor"
}
stage_final() {   # every suite set (ONLYTEST=1: tests only), mixes 0.5 and 1
  local B S M T
  for B in $BBS; do for S in $SETTINGS; do
    M=$(model_args $B $S) || continue
    for T in ${TARGETS[$S]}; do
      [[ $ONLYTEST == 1 && $T == *_val ]] && continue
      run --method transfer $M --target $T $(size_args $T) --max_steps 500 --eval_alphas 0.5 1.0 --tag F1 $(f1_flags $S)
    done
  done; done
}

# re-run the cheap rows on the rain sets so they carry Y-PSNR (the deraining metric); DCTTA on PromptIR too
stage_rescore() {
  local B S M T
  for B in $BBS; do for S in $SETTINGS; do
    M=$(model_args $B $S) || continue
    for T in Rain100H Rain100H_val; do
      if [[ $B == promptir ]]; then run --method identity $M --target $T $(size_args $T) --overwrite; fi
      run --method source $M --target $T $(size_args $T) --overwrite
      run --method flow --n_apply 2 $M --target $T $(size_args $T) --overwrite
      if [[ $B == promptir && ${RESCORE_DCTTA:-1} == 1 ]]; then run --method dctta $M --target $T $(size_args $T) --overwrite; fi
    done
  done; done
}

# ---- controls for the TTA narrative (PromptIR 3-task, CSETS) ----
#   G1  generic synthetic fine-tuning: same pairs machinery, NO target conditioning (no ABC, no exposure matching)
#   S1  F1 conditioned on the WRONG target (--cond_target): adapt on another set's images, evaluate on this one
#   Z1/Z2  no source: same architecture, random init, all weights trained on F1's target-conditioned pairs,
#          500 steps (the TTA budget) and 5000 steps (10x)
CSETS=${CSETS:-"Rain100H SOTS_Indoor DenseHaze Kodak24_Poisson"}
declare -A SWAP=( [Rain100H]=SOTS_Indoor [SOTS_Indoor]=Kodak24_Poisson [DenseHaze]=Rain100H [Kodak24_Poisson]=SOTS_Indoor )
stage_controls() {
  local M T
  M=$(model_args promptir 3task) || return 0
  for T in $CSETS; do
    run --method transfer $M --target $T --max_steps 500 --eval_alphas 0.5 1.0 --tag G1 \
        --families noise+rain+haze --abc_k 1 --exposure_match none
    run --method transfer $M --target $T --max_steps 500 --eval_alphas 0.5 1.0 --tag S1 $(f1_flags 3task) \
        --cond_target ${SWAP[$T]}
    for N in 500 5000; do
      run --method transfer $M --target $T --max_steps $N --tag Z$N --init scratch --subset all --lr 2e-4 \
          --families noise+rain+haze --abc_k 8 --exposure_match camera --nss_color
    done
  done
}

# ---- diagnostics (EXP=diag, all runs record psnr_aff = affine-corrected Y-PSNR; tools/diag.py) ----
#   2x2 factorial {source init, random init} x {target-conditioned pairs (ABC + camera), generic pairs}:
#   F1, G1 (source) and Z500, ZG500 (random init), + the source row; DCTTA on Rain100H (does its gain survive
#   removing the global photometric error?)
stage_diag() {
  local EXP=diag M T
  M=$(model_args promptir 3task) || return 0
  for T in $CSETS; do
    run --method source $M --target $T
    run --method transfer $M --target $T --max_steps 500 --eval_alphas 0.5 1.0 --tag F1 $(f1_flags 3task)
    run --method transfer $M --target $T --max_steps 500 --eval_alphas 0.5 1.0 --tag G1 \
        --families noise+rain+haze --abc_k 1 --exposure_match none
    run --method transfer $M --target $T --max_steps 500 --tag Z500 --init scratch --subset all --lr 2e-4 \
        --families noise+rain+haze --abc_k 8 --exposure_match camera --nss_color
    run --method transfer $M --target $T --max_steps 500 --tag ZG500 --init scratch --subset all --lr 2e-4 \
        --families noise+rain+haze --abc_k 1 --exposure_match none
  done
  run --method dctta $M --target Rain100H
}

# the same on the 5-task-only families (PromptIR 5-task, HIDE / LOLv2-Synthetic), + S1 and Z5000. Blur and
# low-light should need learned priors. Random init gets the best source-free under-exposure rule (threshold on
# the clean bank's 5% brightness quantile, natural-exposure clean images): F1's rule asks the source.
CSETS5=${CSETS5:-"HIDE LOLv2_Syn"}
declare -A SWAP5=( [HIDE]=LOLv2_Syn [LOLv2_Syn]=HIDE )
ZDARK5="--dark_rule threshold --dark_q 0.05 --dark_action natural"
stage_controls5() {
  local EXP=diag M T N
  M=$(model_args promptir 5task) || return 0
  for T in $CSETS5; do
    run --method source $M --target $T
    run --method transfer $M --target $T --max_steps 500 --eval_alphas 0.5 1.0 --tag F1 $(f1_flags 5task)
    run --method transfer $M --target $T --max_steps 500 --eval_alphas 0.5 1.0 --tag G1 \
        --families all --abc_k 1 --exposure_match none
    run --method transfer $M --target $T --max_steps 500 --eval_alphas 0.5 1.0 --tag S1 $(f1_flags 5task) \
        --cond_target ${SWAP5[$T]}
    for N in 500 5000; do
      run --method transfer $M --target $T --max_steps $N --tag Z$N --init scratch --subset all --lr 2e-4 \
          --families all --abc_k 8 --exposure_match camera --nss_color $ZDARK5
    done
    run --method transfer $M --target $T --max_steps 500 --tag ZG500 --init scratch --subset all --lr 2e-4 \
        --families all --abc_k 1 --exposure_match none
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
