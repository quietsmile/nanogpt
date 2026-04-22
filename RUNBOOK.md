# Runbook: reproducing scaling_moe_00196 on nanogpt

Target: PAI DLC job `dlc1q9arre48b0kx` (DisplayName `scaling_moe_00196`), 9L/512d MoE-144 top-8, ~447.3M params.

## 0. Where to run what

| Step | Machine |
|---|---|
| Pull reference, tokenizer / data / model alignment tests | DSW (this box, no GPU) |
| Data extraction (prepare_cybertron_data.py) | any box with CPFS mount; heavy IO |
| Training, bitwise resume, loss trajectory | 8× H100 (see `memory/gpu_8card_ips.md`) |
| Dashboard | any box with Python |

Memory files list the SSH targets — do not hardcode IPs here.

## 1. One-time setup (DSW)

```bash
export ALIBABA_CLOUD_CREDENTIALS_URI=http://localhost:7002/api/v1/credentials/0
cd /home/claudeuser/nanogpt
make reference-pull
```

This writes `reference/dlc1q9arre48b0kx.job.json` and is the source of truth for
all downstream paths (tokenizer, data cache, Megatron source, tensorboard).

## 2. DSW-only alignment tests

```bash
make align
```

Runs four pytest suites in order: tokenizer → data → model → loss. Each writes
`reports/*.json`. Fast (~80s total, dominated by data test loading Megatron IndexedDataset).

Current state (2026-04-20): all pass except `loss_trajectory` requires a nanogpt
training run to produce `reports/nanogpt_train_log.json` — until then only the
reference curve is shown.

## 3. Open the dashboard

```bash
make dashboard
# open http://<host>:8787/dashboard/
```

## 4. Extract training data (heavy)

```bash
# Run on a box with /prodcpfs mounted and enough disk (~8GB for train.bin)
python3 prepare_cybertron_data.py --exp 196
```

Produces `data/cybertron_baseline_cybertron/train.bin` and `val.bin` (name
controlled by the `dataset` field in the config).

## 5. Train on an 8-GPU box

```bash
ssh -o StrictHostKeyChecking=no root@<8gpu-ip>
cd /home/claudeuser/nanogpt   # or wherever the repo is mounted
torchrun --standalone --nproc_per_node=8 train.py config/cybertron_moe_196.py
```

Reference run used `global_batch_size=64, micro_batch=1, gradient_accumulation=64/8=8`.
train.py will divide gradient_accumulation_steps by world_size automatically; adjust
config if you change cluster size.

## 6. Bitwise resume validation (8-GPU box)

```bash
# Single-GPU (PASSES — bitwise model + optimizer match):
cd /root/nanogpt
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_bitwise_resume_test.py --n 10 --m 10

# DDP 4-rank (known limitation — matches iter 11 bitwise, diverges ~2e-6 at iter 12):
python3 scripts/run_bitwise_resume_ddp.py --n 10 --m 10 --nranks 4
```

**Status (2026-04-22)**:
- Single-GPU: PASSED. A-path (10 → save → resume → 10 more) produces bitwise-
  identical model state_dict + optimizer sha256 vs B-path (20 straight).
- DDP: Step 11 (first post-resume) matches bitwise. Iter 12+ diverges by ~2e-6
  growing to ~2e-3. Two fresh DDP runs from same seed DO match bitwise, so the
  resume path has unidentified hidden state issue. See `reports/bitwise_resume.json`.

The test uses `config/bitwise_resume_test.py` (tiny 3-layer MoE, bs=1024) so
each path finishes in <1min.

## 7. Loss trajectory comparison

Once a training run finishes, dump its per-step loss to JSON:

```bash
python3 -c "
import json, pickle
# adapt this based on how train.py writes losses (log file parse or pickled state)
"
# then rerun:
make loss-align
```

`reports/loss_curves.png` overlays Megatron reference (blue) and nanogpt (orange).

## 8. Checkpoint fingerprint

```bash
python -m tools.ckpt_fingerprint out-cybertron-moe-196/ckpt.pt \
  --json reports/ckpt_fingerprint.json
```

Two deterministic runs with identical config and seed must produce identical
`total_sha256`. Any mismatch is a determinism regression.

## Known alignment gaps (see `memory/nanogpt_align_gaps_00196.md`)

Require code changes (not just config) — NOT YET IMPLEMENTED:
- `eod_mask_loss` with EOD=151643 (yaml)
- `mask_loss_id=160000` masking in cross-entropy
- `sequence_wise_balance_loss` (α=0.0001) added to total loss
- `accurate_attn_mask_eod_token` — attention must not cross EOD within a packed sequence

These would affect exact loss numerics. The DSW-only alignment tests today
validate everything that can be checked without training; bit-exact loss needs
these code changes plus an 8-GPU run.
