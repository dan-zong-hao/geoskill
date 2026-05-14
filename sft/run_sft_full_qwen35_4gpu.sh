#!/bin/bash
# 4-GPU full-parameter SFT launcher for Qwen3.5-4B.

set -e
cd /root/autodl-tmp/VQA

source .venv/bin/activate 2>/dev/null || true

: "${PILOT_MODEL_PATH:=/root/autodl-tmp/VQA/models/Qwen3.5-4B}"
: "${NPROC_PER_NODE:=4}"
export PILOT_MODEL_PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

OUT_DIR="/root/autodl-tmp/VQA/sft/ckpt_sft_full_qwen35_think1024_4gpu"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Single-GPU script used batch_size_per_gpu=1 and grad_accum=4, giving global
# batch 4. With 4 processes, grad_accum=1 preserves that global batch while
# reducing wall time.
accelerate launch \
    --multi_gpu \
    --num_processes "${NPROC_PER_NODE}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    /root/autodl-tmp/VQA/sft/train_sft_full_qwen35.py \
    --model_name "${PILOT_MODEL_PATH}" \
    --train_jsonl "/root/autodl-tmp/VQA/json_data/zoom_seg_json/sft_level/sft-00000-of-00001.zoom_seg.think.jsonl" \
    --output_dir "${OUT_DIR}" \
    --num_train_epochs 3 \
    --batch_size_per_gpu 1 \
    --gradient_accumulation_steps 1 \
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
