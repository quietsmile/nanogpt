#!/bin/bash
# Run on DSW. Waits for prep to finish on GPU box, then launches 8-GPU training.
set -eu
GPU_IP=${GPU_IP:-22.4.243.44}
SSH="ssh -o StrictHostKeyChecking=no root@$GPU_IP"

echo "[$(date +%H:%M:%S)] waiting for prep to complete on $GPU_IP..."
while true; do
  if $SSH "grep -q '^Done\\. Wrote' /root/nanogpt/logs/prepare_196_full.log 2>/dev/null"; then
    echo "[$(date +%H:%M:%S)] prep done"; break
  fi
  sleep 30
done

$SSH "ls -la /root/nanogpt/data/cybertron_baseline/"
echo "[$(date +%H:%M:%S)] launching training"
$SSH "cd /root/nanogpt && nohup bash scripts/launch_196_8gpu.sh > logs/train_bg.log 2>&1 &
echo train_pid=\$!"
sleep 5
$SSH "ps auxf | grep -E 'torchrun|train.py' | grep -v grep | head -5"
echo "[$(date +%H:%M:%S)] training started; tail log with: ssh root@$GPU_IP 'tail -f /root/nanogpt/logs/train_bg.log'"
