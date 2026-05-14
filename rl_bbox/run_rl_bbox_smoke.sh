#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/VQA
RL_BBOX="$ROOT/speedup/rl_bbox"
OUT_DIR="$RL_BBOX/ckpt_bbox_grpo_smoke"

OUT_DIR="$OUT_DIR" \
NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
bash "$RL_BBOX/run_rl_bbox_4gpu.sh" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --max_steps "${MAX_STEPS:-1}" \
  --save_steps 1000
