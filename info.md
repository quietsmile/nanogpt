# Muon Alignment Run — Briefing for the other agent

From the `/home/claudeuser/nanogpt` master-branch worker. This covers the
gotchas I hit while getting nano's scaling_moe_00196 retrain to align with
Megatron ref, plus the infra you can reuse for Muon-vs-Megatron-Muon alignment.

## 1. The data file problem you're hitting

**The in-repo `data/cybertron_baseline/{train,val}.bin` is the WRONG file.**

- ✅ Correct file (int32, Qwen vocab 152064): `22.4.243.44:/root/nanogpt/data/cybertron_baseline/train.bin` — 15.7 GB, 479,040 samples × 8192 tokens × 4 B.
- ❌ What's currently in the repo on DSW: a stale 16 MB uint16 stub (vocab-truncated — that's your `indexSelectLargeIndex` OOB).

Two options to fix on box 22.1.6.211:

**Option A — copy the real file from the other box (fastest):**
```bash
# Run ON 22.1.6.211
mkdir -p /root/nanogpt-muon-reimpl/data/cybertron_baseline
scp -o StrictHostKeyChecking=no root@22.4.243.44:/root/nanogpt/data/cybertron_baseline/train.bin \
    /root/nanogpt-muon-reimpl/data/cybertron_baseline/train.bin
scp -o StrictHostKeyChecking=no root@22.4.243.44:/root/nanogpt/data/cybertron_baseline/val.bin \
    /root/nanogpt-muon-reimpl/data/cybertron_baseline/val.bin
```
(The key for ssh between 22.4.243.44 and 22.1.6.211 is already authorized per `memory/gpu_8card_ips.md`.)

**Option B — regenerate via `prepare_cybertron_data.py --exp 196`:**
- Requires `/newcpfs/user/yuchen/llm/megatron_dots3.0_swa` and `/newcpfs/user/yuchen/llm/cybertron_dots3.0_swa` on PYTHONPATH.
- Hits `/prodcpfs/user/data/save/data/lossalign/data_cache/43adec39b46f5eb95d144361a0db6699-BlendedDataset-train-*` for the sampler index.
- ~20 min to generate.

Option A is simpler.

## 2. dtype + stride MUST be int32 with stride=block_size

Nano's `train.py` memmaps with `dtype=np.int32`. Stride is `block_size` (not `block_size+1`). Target is `X[s+1 : s+1+block_size]`. Don't let a "helpful" edit revert these — it was a real bug fixed in commit `076fb6d`-era work.

## 3. Seed from Megatron iter_0

For apples-to-apples alignment you seed nano from ref's iter_0 Megatron ckpt + the iter_0 Adam state. Script exists:

```bash
python3 scripts/seed_from_meg_ckpt.py \
  --meg-dir /prodcpfs/user/yuchen/share/scaling_moe_00196_muon/iter_0000000 \
  --meg-optim /root/nanogpt-muon-reimpl/reports/short_window/meg_optim_iter0.pt \
  --out /root/nanogpt-muon-reimpl/out-muon-align/ckpt.pt
```

You likely need `meg_optim_iter0.pt` — if the Muon ref has one it's in its `iter_0000000` dir or a companion. If it doesn't, you'll need to convert. The existing AdamW converter is `scripts/recover_megatron_optim.py` and `scripts/optim_megatron_to_nano.py` — those assume AdamW moment tensors. You'll need to adapt for Muon: momentum + spectral buffer if ref stores them.

`scripts/seed_from_meg_ckpt.py` expects `init_std=0.006`, `rotary_base=50000`, etc. — the muon ref config should share these; verify first.

## 4. Expected iter-0 loss

On the real data file, nano iter 0 loss should be **11.943** (matches ref iter 1 TB). If you get anything else, the data is wrong.

## 5. Launch scaffolding

Strict-determinism NCCL flags are load-bearing for reproducibility. Copy from `scripts/launch_196_from0_fresh.sh`:

```
NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0
NCCL_COLLNET_ENABLE=0 NVIDIA_TF32_OVERRIDE=0
```

Also `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or the MoE bucket dispatcher OOMs at peak memory.

DP=8 on 22.1.6.211. Config uses `gradient_accumulation_steps=64` pre-DDP-divide; train.py will `//= world_size` to 8. `batch_size=1` is fixed due to lm_head logits tensor memory — DO NOT try mb>=2 until a chunked-CE lands.

## 6. The `/root/nanogpt/logs/moediag` pattern

I just shipped per-iter MoE routing diagnostics (commit `cd7f27b`, refined in `865373b`) — the new train.py writes per-layer stats into `train_log.jsonl`:

```
moe_per_layer:  list of 8 dicts, one per MoE layer, each with
                {L, max, min, mean, dead, ent, sc_mean, sc_std, mg_p5, mg_med}
maxvio_mb4_apples: apples-to-apples synthetic-mb=4 maxvio that matches
                   ref's master-log `maxVio/micro_batch` formula exactly
                   (nano mb=1, so we aggregate 4 consecutive forwards)
```

**Use `maxvio_mb4_apples` for comparison with ref master log**, not the legacy `maxvio_micro_batch` field — the latter is a different aggregation (see below).

## 7. The maxvio measurement trap I fell into

Had to throw away a whole day's analysis: the previous `maxvio_micro_batch` was computed as max-across-(8-layers × 144-experts) of counts *summed across all 64 microbatches* of a step, divided by `gradient_accumulation_steps`. That LOOKED like it matched ref's `tokens_per_expert/{max,mean}`, but it doesn't — ref emits per-microbatch max/min/mean and then averages across (mbs × layers).

Summing 64 mbs smooths out per-mb peaks (because the hottest expert shifts between mbs). So the nano value came out *systematically lower* than ref's, making it look like nano routes more uniformly by 3.6×. It was purely measurement. With apples-to-apples: nano 6.93 vs ref 6.95 at iter 1.

**Lesson for Muon alignment:** when you add diagnostics, write a one-shot script that loads ref's ckpt into nano, runs a single forward on ref's matched batch, and verifies your metric output MATCHES ref's logged value to <1%. Do this on iter 1 first. If your metric drifts from ref at iter 1 with identical weights + same data, the metric formula is wrong.

## 8. Infrastructure you can reuse

- Dashboard: runs on this DSW (22.2.74.61) port 8882. URL: http://47.84.144.221:8882/dashboard/alignment_report.html. Served by `python -m http.server 8882 --directory /home/claudeuser/nanogpt` as pid 3485314. You can refresh its contents by running `python3 dashboard/refresh_runs.py --rebuild` after adding your run to `dashboard/refresh_runs.py::RUNS`.
- Per-run JSON at `reports/runs/<run_id>.json` — auto-built by refresh_runs.py from SSH-pulled `train_log.jsonl`.
- Ref TB (not Muon! AdamW) at `reference/tb/key_scalars.json` — 7485 iters of lm_loss, lr, grad_norm. For Muon you'd need the corresponding Muon ref TB.
- Ref master log routing stats extracted to `reference/ref_moe_routing_stats.json` (7485 rows). For Muon ref, run `/tmp/extract_ref_moe_stats.py <path>/logs/rank-0-*.log <out.json>`.
- v10-fresh checkpoint + log on 22.4.243.44 in `/root/nanogpt/out-cybertron-moe-196-from0-fresh/`. Use for apples-to-apples AdamW sanity checks.

## 9. Align targets (for Muon vs AdamW)

Muon ref at `/newcpfs/user/yuchen/share/scaling_moe_00196_muon` — if it exists. AdamW ref at `/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196`. Double-check the Muon ref actually trained with exactly Muon and not a hybrid, and that its data path / LR schedule / seed match. Train_args should be in `<ref>/train_args.json` or similar.

The expected iter-wise loss you cited (iter 1=11.94, iter 100=7.93, iter 500=4.71) is CONSISTENT with nano's AdamW run (iter 100 we see ~7.81). If the Muon ref really diverges on a curve very different from AdamW, expect harder alignment.

## 10. Open findings on the AdamW side (for context)

- `expert_bias` ranges match nano-vs-ref within 4% after 7000 steps (not the source of any gap)
- Final nano-vs-ref AdamW loss gap: +0.005 nat (last-100 mean at iter 7485). Not yet explained — initially thought it was bf16 floor, then thought it was MoE routing divergence, both refuted. The real source is still open. v6-vs-v10 (two nano runs, same seed, different code) mean diff ≈ 0 (overall), so pure bf16 noise between two nano runs is small. Nano-vs-ref +0.005 appears systematic.
- The 98.9% topk-boundary result (iter_0, 98.9% of tokens have top-8/top-9 score margin < 1 bf16 ULP) explains why two bf16 numerical paths CAN produce different routings at iter_1 even from identical weights. But the magnitude of that contribution to final loss isn't measured.

This is relevant to your Muon work because **Muon's update rule will change weight trajectories**, potentially by larger amounts than AdamW → could compound the bf16 topk sensitivity differently. Worth keeping an eye on.

## 11. Handy commands

```bash
# Pull latest train_log from a remote run
scp root@<box>:/root/nanogpt-muon-reimpl/out-.../train_log.jsonl /tmp/

# Quick comparison to ref (adapt paths)
python3 -c "
import json, numpy as np
nano = [json.loads(l) for l in open('/tmp/train_log.jsonl')]
ref_tb = json.load(open('/home/claudeuser/nanogpt/reference/tb/key_scalars.json'))
ref_by = {int(s): v for s, v in ref_tb['lm loss']}
print('iter  nano_loss  ref_loss  Δ  nano_mv_apples  ref_mv')
for d in nano[::50]:
    r = ref_by.get(d['iter'] + 1, None)  # nano iter N ↔ ref iter N+1
    if r is None: continue
    print(f\"{d['iter']:>4d}  {d['loss']:>8.4f}  {r:>8.4f}  {d['loss']-r:>+7.4f}  {d.get('maxvio_mb4_apples', 0):>6.3f}\")
"

# Verify your metric at iter 1 apples-to-apples:
# run /tmp/apples_maxvio.py on a box with CUDA. It loads ref iter_0 weights
# into nano and runs 8 mb=4 forwards. If your maxvio doesn't match ref within
# 2% at iter 1, your logging formula is wrong.
```

## 12. Git layout warning

Recently the working tree ended up on branch `muon-new` twice — something in my tooling auto-switches. If you see unexpected branch changes on DSW `/home/claudeuser/nanogpt`, it's probably me. Pull via `git fetch origin; git rebase origin/master` if anything looks stale. I'm at `865373b` on master right now.
