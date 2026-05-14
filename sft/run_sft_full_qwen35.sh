#!/bin/bash
# Full-parameter SFT launcher for Qwen3.5-4B.
# Mirrors ZoomEarth/run_scripts/train_sft.sh hyperparams but uses the new
# Qwen3.5 model class.

set -e
cd /root/autodl-tmp/VQA

source .venv/bin/activate 2>/dev/null || true

# Default base = locally downloaded Qwen3.5-4B.
: "${PILOT_MODEL_PATH:=/root/autodl-tmp/VQA/models/Qwen3.5-4B}"
export PILOT_MODEL_PATH

OUT_DIR="/root/autodl-tmp/VQA/sft/ckpt_sft_full_qwen35_think1024"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python /root/autodl-tmp/VQA/sft/train_sft_full_qwen35.py \
    --model_name "${PILOT_MODEL_PATH}" \
    --train_jsonl "/root/autodl-tmp/VQA/json_data/zoom_seg_json/sft_level/sft-00000-of-00001.zoom_seg.think.jsonl" \
    --output_dir "${OUT_DIR}" \
    --num_train_epochs 3 \
    --batch_size_per_gpu 1 \
    --gradient_accumulation_steps 4 \
    --lr 3e-5 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --seed 42 \
    --dtype bfloat16 \
    --save_steps 200 \
    --log_steps 10 \
    --print_steps 20 \
    --max_pixels $((64*64*28*28)) \
    --max_length 4096 \
    2>&1 | tee "${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
