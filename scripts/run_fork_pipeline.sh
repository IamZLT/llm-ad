#!/usr/bin/env bash
# Fork pipeline for the Internal-Loop depth-curriculum experiment.
#
# Experiment matrix (RegionAdapter frozen at the converged Direct checkpoint for every branch):
#
#   Direct (region_sft_direct_e3)
#     ├── Loop2        (Direct -> K=2)
#     ├── Loop4_curr   (Direct -> Loop2 -> K=4)   [true depth curriculum]
#     └── Loop4_jump   (Direct -> K=4)            [ablation: no K=2 stage]
#
# Each stage is gated by `&&`-style fail-fast semantics (set -e), so a failure stops
# the chain instead of running downstream stages on a broken checkpoint.
#
# Usage:
#   GPU=0 bash scripts/run_fork_pipeline.sh
#
set -euo pipefail

ROOT=/root/data-fs/bljiao/anomaly_llm
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
N_ANOMALY="${N_ANOMALY:-16}"
N_NORMAL="${N_NORMAL:-16}"

DIRECT_DIR=outputs/train/region_sft_direct_e3
DIRECT_LOG=outputs/train/region_sft_direct_e3.log
LOOP2_DIR=outputs/train/region_sft_loop2_from_direct
LOOP4_CURR_DIR=outputs/train/region_sft_loop4_curr
LOOP4_JUMP_DIR=outputs/train/region_sft_loop4_jump
EVAL_ROOT=outputs/eval_fork_mvtec200

PROBE_CFG=configs/qwen35_2b_annos_probe_v2_loop4_rl.yaml
DIRECT_SFT_CFG=configs/qwen35_2b_annos_probe_v2_direct_sft.yaml
LOOP2_SFT_CFG=configs/qwen35_2b_annos_probe_v2_loop2_from_direct_sft.yaml
LOOP4_CURR_SFT_CFG=configs/qwen35_2b_annos_probe_v2_loop4_curr_sft.yaml
LOOP4_JUMP_SFT_CFG=configs/qwen35_2b_annos_probe_v2_loop4_jump_sft.yaml
DIRECT_RL_CFG=configs/qwen35_2b_annos_probe_v2_direct_rl.yaml
LOOP2_RL_CFG=configs/qwen35_2b_annos_probe_v2_loop2_rl.yaml
LOOP4_RL_CFG=configs/qwen35_2b_annos_probe_v2_loop4_rl.yaml
FSM_RL_CFG=configs/qwen35_2b_annos_probe_v2_fsm_final_only_rl.yaml

log() { echo "[pipeline] $(date '+%Y-%m-%d %H:%M:%S') $*"; }

# ---------------------------------------------------------------------------
# Stage 0: wait for the in-flight Direct SFT to finish, then gate on convergence.
# ---------------------------------------------------------------------------
wait_for_direct() {
  log "waiting for Direct SFT process to exit ..."
  while pgrep -f "train_region_sft.py.*direct_sft" >/dev/null 2>&1; do
    sleep 30
  done
  [ -f "$DIRECT_DIR/adapter_model.safetensors" ] || { log "ERROR: Direct checkpoint missing at $DIRECT_DIR"; return 1; }
  log "Direct SFT done -> $DIRECT_DIR"
}

# Returns 0 when Direct is considered converged; 1 when one more epoch is warranted.
direct_converged() {
  python3 - "$DIRECT_LOG" <<'PY'
import re, sys
losses = [float(m) for m in re.findall(r'dev_loss=([0-9.]+)', open(sys.argv[1]).read())]
if len(losses) < 2:
    print(f'NOT_CONVERGED (only {len(losses)} dev eval(s): {losses})')
    sys.exit(1)
prev, last = losses[-2], losses[-1]
rel = (prev - last) / max(prev, 1e-9)
if rel > 0.02:
    print(f'STILL_DROPPING prev={prev:.4f} last={last:.4f} rel={rel:.3%}')
    sys.exit(1)
print(f'CONVERGED last={last:.4f} prev={prev:.4f} rel={rel:.3%}')
PY
}

extra_direct_epoch() {
  log "dev loss still dropping -> one extra Direct epoch"
  python train_region_sft.py --config "$DIRECT_SFT_CFG" \
    --init-sft "$DIRECT_DIR" --output-dir "$DIRECT_DIR" --epochs 1 --num-gpu 1 \
    >> "$DIRECT_LOG" 2>&1
}

# ---------------------------------------------------------------------------
# SFT / probe / eval primitives
# ---------------------------------------------------------------------------
run_sft() {
  local cfg=$1 init=$2 out=$3 epochs=$4 tag=$5
  log "SFT $tag <- $init (epochs=$epochs)"
  python train_region_sft.py --config "$cfg" --init-sft "$init" --freeze-region-adapter \
    --output-dir "$out" --epochs "$epochs" --num-gpu 1 2>&1 | tee "outputs/train/$(basename "$out").log"
}

run_probe() {
  local sft=$1 steps=$2 tag=$3
  log "probe $tag: $sft @ K=$steps"
  python scripts/probe_loop_progress.py "$PROBE_CFG" \
    --sft-adapter "$sft" --loop-steps "$steps" \
    --n-anomaly "$N_ANOMALY" --n-normal "$N_NORMAL" 2>&1 | tee "outputs/probe_${tag}.log"
}

run_eval() {
  local rlcfg=$1 sft=$2 out=$3 tag=$4
  log "eval $tag -> $out"
  local extra=()
  if [ -n "$sft" ]; then extra=(--sft-adapter "$sft"); fi
  python train_outcome_multibox.py --config "$rlcfg" --mode eval --split test --eval-limit 200 \
    "${extra[@]}" --output-dir "$out"
}

# ---------------------------------------------------------------------------
# Main chain
# ---------------------------------------------------------------------------
main() {
  wait_for_direct

  if ! direct_converged; then
    extra_direct_epoch
  fi

  # Key diagnostic: bare recurrence on the Direct weights (theta_Direct @ K=1..4).
  run_probe "$DIRECT_DIR" 4 direct

  # Loop2 (Direct -> K=2)
  run_sft "$LOOP2_SFT_CFG" "$DIRECT_DIR" "$LOOP2_DIR" 1 loop2

  # Did Loop2 adaptation turn the 2nd recurrence from destructive -> refinement?
  run_probe "$LOOP2_DIR" 2 loop2

  # Loop4_curr (Loop2 -> K=4) and Loop4_jump (Direct -> K=4)
  run_sft "$LOOP4_CURR_SFT_CFG" "$LOOP2_DIR" "$LOOP4_CURR_DIR" 1 loop4_curr
  run_sft "$LOOP4_JUMP_SFT_CFG" "$DIRECT_DIR" "$LOOP4_JUMP_DIR" 1 loop4_jump

  run_probe "$LOOP4_CURR_DIR" 4 loop4_curr
  run_probe "$LOOP4_JUMP_DIR" 4 loop4_jump

  # MVTec-200 eval (new conditional metrics)
  run_eval "$DIRECT_RL_CFG" "$DIRECT_DIR" "$EVAL_ROOT/direct" direct
  run_eval "$LOOP2_RL_CFG" "$LOOP2_DIR" "$EVAL_ROOT/loop2" loop2
  run_eval "$LOOP4_RL_CFG" "$LOOP4_CURR_DIR" "$EVAL_ROOT/loop4_curr" loop4_curr
  run_eval "$LOOP4_RL_CFG" "$LOOP4_JUMP_DIR" "$EVAL_ROOT/loop4_jump" loop4_jump
  run_eval "$FSM_RL_CFG" "" "$EVAL_ROOT/fsm" fsm

  log "ALL DONE"
}

main "$@"
