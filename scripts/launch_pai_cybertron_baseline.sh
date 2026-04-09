#!/bin/bash
# PAI launch script for nanogpt cybertron baseline (deterministic, aligned with scaling_moe_00196)
#
# Usage (from this directory):
#   python /home/claudeuser/pai_manage.py create-job \
#       --name "nanogpt_cybertron_baseline" \
#       --command "$(cat scripts/launch_pai_cybertron_baseline.sh)"
#
# Or for dry-run: add --dry-run

set -x
set -e

export GLOO_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Bitwise deterministic: required for exact reproducibility
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

NANOGPT_PATH="/newcpfs/user/yuchen/nanogpt"
SAVE_DIR="/prodcpfs/user/yuchen/nanogpt_exp/cybertron_baseline"
EXP_NAME="nanogpt_cybertron_baseline_adam"

GPUS_PER_NODE=$(nvidia-smi | grep NVIDIA | grep On | wc -l)

mkdir -p "${SAVE_DIR}/${EXP_NAME}"

cd "${NANOGPT_PATH}"

# Step 1: Prepare data (skip if already done)
if [ ! -f "data/cybertron_baseline/train.bin" ]; then
    echo "=== Preparing cybertron data ==="
    PYTHONPATH="/newcpfs/user/yuchen/llm/megatron_dots3.0_swa:/newcpfs/user/yuchen/llm/cybertron_dots3.0_swa:${PYTHONPATH}" \
    python prepare_cybertron_data.py \
        --n_train_samples 479040 \
        --n_val_samples 2000 \
        --out_dir data/cybertron_baseline
fi

# Step 2: Run deterministic training
DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} --nnodes ${WORLD_SIZE:-1} --node_rank ${RANK:-0} --master_addr ${MASTER_ADDR:-localhost} --master_port ${MASTER_PORT:-29500}"

torchrun ${DISTRIBUTED_ARGS} train.py \
    config/cybertron_baseline.py \
    --out_dir="${SAVE_DIR}/${EXP_NAME}" \
    --wandb_log=True \
    --wandb_project="nanogpt-cybertron" \
    --wandb_run_name="${EXP_NAME}" \
    --compile=False \
    --deterministic=True \
    2>&1 | tee "${SAVE_DIR}/${EXP_NAME}/train.log"

echo "Training complete. Logs at: ${SAVE_DIR}/${EXP_NAME}/train.log"
