#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/VQA
GEOSKILL="$ROOT/speedup/geoskill"

OUT_DIR="${OUT_DIR:-$GEOSKILL/ckpt_geoskill_smoke}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-1}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}" \
NUM_GENERATIONS="${NUM_GENERATIONS:-2}" \
bash "$GEOSKILL/run_geoskill_bbox_4gpu.sh" \
  --max_steps "${MAX_STEPS:-3}" \
  --save_steps 1000 \
  --skip_final_save
