#!/bin/bash
# Pull latest training log from GPU box, convert to dashboard format, rebuild single-file HTML.
# Usage: scripts/refresh_dashboard.sh [gpu_ip]
set -e

GPU_IP=${1:-22.4.243.44}
REPO=/home/claudeuser/nanogpt
REMOTE=/root/nanogpt/out-cybertron-moe-196/train_log.jsonl
LOCAL_JSONL=$REPO/reports/nanogpt_train_log.jsonl
LOCAL_JSON=$REPO/reports/nanogpt_train_log.json

mkdir -p "$REPO/reports"

# Pull jsonl (may not exist yet)
scp -o StrictHostKeyChecking=no root@$GPU_IP:$REMOTE "$LOCAL_JSONL" 2>/dev/null || {
  echo "no training log yet on $GPU_IP:$REMOTE"
  exit 0
}

# Also pull the plain-text training log for val-loss extraction
LOCAL_LOG="$REPO/reports/nanogpt_train_bg.log"
scp -o StrictHostKeyChecking=no root@$GPU_IP:/root/nanogpt/logs/train_bg.log "$LOCAL_LOG" 2>/dev/null || true

# Convert jsonl (+ val from raw log) → {train_loss, val_loss, lr, ...}
python3 "$REPO/scripts/extract_val_loss.py" "$LOCAL_LOG" "$LOCAL_JSONL" "$LOCAL_JSON"

# Regenerate alignment checklist
python3 "$REPO/tools/alignment_checklist.py" >/dev/null || true

cd "$REPO" && python3 dashboard/build_local.py --inline-plotly >/dev/null
echo "dashboard updated: $REPO/dashboard/alignment_report.html"
