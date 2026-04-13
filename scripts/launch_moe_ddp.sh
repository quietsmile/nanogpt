#!/bin/bash
# Launch MoE training on 8 GPUs with DDP
# Usage: bash scripts/launch_moe_ddp.sh [config_file]
#
# Config: config/cybertron_moe_198.py
#   gradient_accumulation_steps=64 → 64/8=8 per GPU
#   batch_size=2, block_size=8192
#   global_batch = 8 GPUs × 8 accum × 2 batch = 128 sequences = 1,048,576 tokens ✓
#
# Per-iter time (estimated): ~8 accum × 1.5s × ~2 (fwd+bwd amortized) ≈ ~25s/iter
# Total: 10847 iters × 25s ≈ ~75 hours (~3 days)

set -e

CONFIG=${1:-config/cybertron_moe_198.py}
LOG_DIR=logs
mkdir -p $LOG_DIR

echo "Launching MoE DDP training with config: $CONFIG"
echo "Logging to $LOG_DIR/train_moe_ddp.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train.py "$CONFIG" \
    2>&1 | tee "$LOG_DIR/train_moe_ddp.log"
