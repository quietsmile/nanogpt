#!/bin/bash
# Run on DSW. Every 10 min, pull train_log.jsonl from GPU box and rebuild dashboard.
set -eu
GPU_IP=${GPU_IP:-22.4.243.44}
REPO=/home/claudeuser/nanogpt

while true; do
  bash "$REPO/scripts/refresh_dashboard.sh" "$GPU_IP" >> "$REPO/logs/refresh.log" 2>&1 || true
  sleep 600
done
