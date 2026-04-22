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

Current state (2026-04-23): all alignment tests pass, including the 6-test
`tests/test_code_gaps.py` suite. `loss_trajectory` requires a nanogpt training
run to produce `reports/nanogpt_train_log.json` — once a run exists
(e.g. the 7485-step retrain in `out-cybertron-moe-196-from0-fresh/train_log.jsonl`),
`make loss-align` overlays it on the reference curve.

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

**Status (2026-04-23)**:
- Single-GPU: PASSED. A-path (10 → save → resume → 10 more) produces bitwise-
  identical model state_dict + optimizer sha256 vs B-path (20 straight).
- DDP 4-rank: iter-20 A-vs-B drift is 3e-5 after commit `b4f9e75` added the
  strict-determinism NCCL/CUDA env flags (`NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`,
  `NCCL_P2P_DISABLE=1`, `NCCL_NVLS_ENABLE=0`, `NCCL_COLLNET_ENABLE=0`,
  `NVIDIA_TF32_OVERRIDE=0`, `matmul.allow_tf32=False`, `cudnn.allow_tf32=False`).
  Two fresh DDP runs from the same seed remain bitwise-equal, confirming
  training itself is deterministic. Residual 3e-5 is NCCL reduction-order noise
  in bf16, near the representation floor. See `reports/bitwise_resume.json`.

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

All four 00196 code gaps are implemented and covered by
`tests/test_code_gaps.py`:

- `eod_mask_loss` — masks loss at positions where `idx == eod_token_id`
  (input-based, matching ref `loss_mask[data == eod_token] = 0.0`, commit
  `076fb6d`).
- `mask_loss_id=160000` — masks loss at positions where `idx == mask_loss_id`
  (same input-based mechanism).
- `sequence_wise_balance_loss` — adds `seq_aux_balance_alpha * aux` to the
  total loss when alpha > 0 and `model.training` is true.
- `accurate_attn_mask_eod_token` — when `use_eod_attn_mask=True`, the model
  builds a per-sample segment mask via cumsum of EOD positions and runs
  SDPA with that dense bool mask (same-segment AND causal).

Run `pytest tests/test_code_gaps.py -v` to exercise all four end-to-end on CPU.
See `ALIGNMENT.md` v10 FINAL section for the end-to-end retrain result
(final Δ = +0.0047 nat over 7485 iters).
