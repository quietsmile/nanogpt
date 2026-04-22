#!/bin/bash
# Launch fresh 7485-step retrain with all alignment fixes (fused RoPE, SwiGLU fp32,
# MoE fp32 weighted sum, RMSNorm fp32 leak fix, strict-determinism NCCL/CUDA flags).
#
# Seed ckpt built separately via:
#   python3 scripts/seed_from_meg_ckpt.py \
#     --meg-dir /prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196_init/iter_0000000 \
#     --meg-optim /root/nanogpt/reports/short_window/meg_optim_iter0.pt \
#     --out /root/nanogpt/out-cybertron-moe-196-from0-fresh/ckpt.pt
set -eux
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Strict DDP determinism (matches the bitwise-resume fix set)
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_COLLNET_ENABLE=0
export NVIDIA_TF32_OVERRIDE=0

OUT_DIR=${OUT_DIR:-out-cybertron-moe-196-from0-fresh}
mkdir -p "$OUT_DIR" logs

torchrun --standalone --nproc_per_node=8 \
  train.py config/cybertron_moe_196_from0.py \
  --out_dir="$OUT_DIR" \
  2>&1 | tee "logs/train_from0_fresh_$(date +%Y%m%d_%H%M%S).log"
