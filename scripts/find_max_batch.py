"""Binary-search the maximum per-rank micro-batch that fits in GPU memory.

Use case: for a given tier config (e.g. tier4_moe_1b.yaml), find the largest
`batch_size × gradient_accumulation_steps` combination that fits, so we
maximize DP/grad-accum while staying under 80GB per H100.

Strategy:
  - Fix batch_size=1 (to stay minimal), binary-search grad_accum_steps.
  - Try each candidate: build model → 1 forward → 1 backward → check peak memory.
  - Print peak mem + fit status to help user pick production value.

Example:
  python3 scripts/find_max_batch.py \
      --config configs/scaling/tier4_moe_1b.yaml \
      --micro-batch 1 \
      --grad-accum-bounds 4,64
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def _load_yaml(p):
    import yaml
    with open(p) as f:
        return yaml.safe_load(f)


def _build_model(cfg):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nanogpt.model import GPT, GPTConfig
    arch = cfg["arch"]
    gc = GPTConfig(
        block_size=arch["block_size"],
        vocab_size=arch.get("vocab_size_override", 50257),
        n_layer=arch["n_layer"],
        n_head=arch["n_head"],
        n_embd=arch["n_embd"],
        n_kv_head=arch.get("n_kv_head"),
        kv_channels=arch.get("kv_channels"),
        dropout=0.0,
        bias=False,
        init_std=arch.get("init_std", 0.02),
        use_rope=arch.get("use_rope", False),
        rotary_base=arch.get("rotary_base", 10000),
        use_rmsnorm=arch.get("use_rmsnorm", False),
        norm_eps=arch.get("norm_eps", 1e-5),
        use_swiglu=arch.get("use_swiglu", False),
        ffn_hidden_size=arch.get("ffn_hidden_size"),
        qk_layernorm=arch.get("qk_layernorm", False),
        tie_embeddings=arch.get("tie_embeddings", True),
        disable_scaled_init_method=arch.get("disable_scaled_init_method", False),
        use_moe=arch.get("use_moe", False),
        moe_layer_freq=arch.get("moe_layer_freq"),
        num_experts=arch.get("num_experts", 1),
        moe_ffn_hidden_size=arch.get("moe_ffn_hidden_size", 0),
        moe_router_topk=arch.get("moe_router_topk", 1),
        moe_n_group=arch.get("moe_n_group", 1),
        moe_topk_group=arch.get("moe_topk_group", 1),
        moe_norm_topk_prob=arch.get("moe_norm_topk_prob", True),
        moe_router_score_correction_coeff=arch.get("moe_router_score_correction_coeff", 0.0),
        moe_shared_expert_hidden_size=arch.get("moe_shared_expert_hidden_size"),
        moe_routing_type=arch.get("moe_routing_type", "greedy"),
        eod_token_id=arch.get("eod_token_id", 0),
        mask_loss_id=arch.get("mask_loss_id", -1),
        seq_aux_balance_alpha=arch.get("seq_aux_balance_alpha", 0.0),
    )
    m = GPT(gc).cuda()
    m.train()
    return m, gc


def _try_fit(model, cfg, micro_batch, grad_accum):
    """Return (fits, peak_gb, last_err)."""
    arch = cfg["arch"]
    block = arch["block_size"]
    vocab = arch.get("vocab_size_override", 50257)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        os.environ["NANO_TE_MOE"] = "0"
        for _ in range(grad_accum):
            x = torch.randint(0, vocab, (micro_batch, block), device="cuda")
            y = torch.randint(0, vocab, (micro_batch, block), device="cuda")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
            (loss / grad_accum).backward()
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        return True, peak_gb, None
    except torch.cuda.OutOfMemoryError as e:
        return False, 0.0, str(e)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="scaling yaml")
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum-bounds", default="1,64", help="lo,hi")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required — run on GPU box or PAI pod.")
        return 1
    lo, hi = map(int, args.grad_accum_bounds.split(","))

    cfg = _load_yaml(args.config)
    print(f"=== find_max_batch: {cfg['name']} mb={args.micro_batch} range [{lo}, {hi}] ===")
    model, _ = _build_model(cfg)

    results = []
    best_fit = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, peak, err = _try_fit(model, cfg, args.micro_batch, mid)
        results.append((mid, ok, peak, err))
        print(f"  grad_accum={mid:>3}  {'FIT' if ok else 'OOM'}  peak={peak:.2f} GB"
              + (f"  ({err[:80]})" if err else ""))
        if ok:
            best_fit = mid
            lo = mid + 1
        else:
            hi = mid - 1

    print(f"\nbest fitting grad_accum = {best_fit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
