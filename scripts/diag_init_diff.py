"""Compare nano fresh-init vs Megatron iter_0 — per-layer weight statistics.

Outputs reports/init_diff.json + console table showing for each matched parameter:
  - shape match
  - dtype
  - mean / std / abs_max
  - svd top1 / top10% / cond (only for 2D matrices, sampled)
  - delta in std (the most diagnostic stat — Muon's NS5 sensitivity scales with sigma_max/sigma_min)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def build_nano_fresh(seed: int = 1337):
    """Construct a nano GPT with cybertron_moe_196 config + given seed, return state_dict."""
    from nanogpt.model import GPT, GPTConfig
    torch.manual_seed(seed)
    cfg = GPTConfig(
        block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512,
        n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006,
        use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5,
        use_swiglu=True, ffn_hidden_size=1536, tie_embeddings=False,
        qk_layernorm=True, disable_scaled_init_method=True,
        use_moe=True, moe_layer_freq=[0] + [1] * 8, num_experts=144,
        moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1,
        moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001,
        moe_shared_expert_hidden_size=160,
    )
    m = GPT(cfg)
    return {k: v.detach().cpu().float() for k, v in m.state_dict().items()
            if v.dim() >= 1 and v.numel() > 1}


def load_meg_iter0(meg_dir: str):
    """Use scripts/megatron_to_nano.py to convert iter_0000000 to nano format."""
    from megatron_to_nano import load_all_megatron_shards, convert
    meg = load_all_megatron_shards(meg_dir)
    sd = convert(meg)
    return {k: v.detach().cpu().float() for k, v in sd.items()
            if v.dim() >= 1 and v.numel() > 1}


def stats(t: torch.Tensor) -> dict:
    out = {
        "shape": list(t.shape),
        "dtype": str(t.dtype).removeprefix("torch."),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "abs_max": float(t.abs().max()),
    }
    if t.dim() == 2 and min(t.shape) > 1:
        # Reshape weight matrix to [out, in] (already conventional in nn.Linear)
        try:
            # Subsample for speed if very large
            r = t
            if t.numel() > 4_000_000:
                idx = torch.randperm(min(t.shape))[:512]
                r = t.index_select(min(t.shape == t.shape[0]), idx) if False else t[:512, :512]
            s = torch.linalg.svdvals(r)
            out["sigma_max"] = float(s[0])
            out["sigma_min"] = float(s[-1])
            out["sigma_p10"] = float(s[len(s) * 9 // 10])
            out["cond"] = float(s[0] / max(s[-1], 1e-12))
        except Exception as e:
            out["svd_err"] = str(e)[:80]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meg-dir",
                    default="/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196_init/iter_0000000",
                    help="Megatron iter_0000000 directory")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(ROOT / "reports" / "init_diff.json"))
    args = ap.parse_args()

    print(f"=== building nano fresh-init seed={args.seed} ===", flush=True)
    nano_sd = build_nano_fresh(args.seed)
    print(f"  nano params with dim>=1: {len(nano_sd)}")

    print(f"\n=== loading Megatron iter_0 from {args.meg_dir} ===", flush=True)
    if not os.path.exists(args.meg_dir):
        print(f"  FATAL: {args.meg_dir} missing")
        return 1
    try:
        meg_sd = load_meg_iter0(args.meg_dir)
    except Exception as e:
        print(f"  FATAL converter failed: {e}")
        return 1
    print(f"  meg params with dim>=1: {len(meg_sd)}")

    # Match by name (converter already maps to nano names)
    common = sorted(set(nano_sd) & set(meg_sd))
    nano_only = sorted(set(nano_sd) - set(meg_sd))
    meg_only = sorted(set(meg_sd) - set(nano_sd))
    print(f"\n  matched: {len(common)}")
    print(f"  nano-only: {len(nano_only)}")
    print(f"  meg-only:  {len(meg_only)}")
    if nano_only[:5]:
        print(f"    nano-only sample: {nano_only[:5]}")
    if meg_only[:5]:
        print(f"    meg-only sample:  {meg_only[:5]}")

    rows = {}
    for k in common:
        a = nano_sd[k]; b = meg_sd[k]
        if a.shape != b.shape:
            rows[k] = {"shape_mismatch": [list(a.shape), list(b.shape)]}
            continue
        sa = stats(a); sb = stats(b)
        rows[k] = {
            "nano": sa, "meg": sb,
            "std_ratio": sa["std"] / max(sb["std"], 1e-12),
        }

    # Top deltas
    print("\n=== top 20 layers by std ratio (nano/meg) ===")
    fmt = "  {name:55s}  shape={shape:18s}  nano_std={ns:.4e}  meg_std={ms:.4e}  ratio={r:.3f}"
    sorted_rows = sorted(
        ((k, v) for k, v in rows.items() if "nano" in v),
        key=lambda kv: abs(math.log(kv[1]["std_ratio"]))
        if kv[1]["std_ratio"] > 0 else 0,
        reverse=True,
    )
    for k, v in sorted_rows[:25]:
        print(fmt.format(
            name=k[:55],
            shape=str(v["nano"]["shape"]),
            ns=v["nano"]["std"], ms=v["meg"]["std"],
            r=v["std_ratio"]))

    # SVD sigma_max ratio for matrices
    print("\n=== top 20 layers by sigma_max ratio (nano/meg) ===")
    svd_rows = [(k, v) for k, v in rows.items()
                if "nano" in v and "sigma_max" in v["nano"] and "sigma_max" in v["meg"]]
    for k, v in sorted(svd_rows,
                       key=lambda kv: abs(math.log(kv[1]["nano"]["sigma_max"] / max(kv[1]["meg"]["sigma_max"], 1e-12))),
                       reverse=True)[:20]:
        nm, mm = v["nano"]["sigma_max"], v["meg"]["sigma_max"]
        nc, mc = v["nano"].get("cond", 0), v["meg"].get("cond", 0)
        print(f"  {k[:50]:50s}  σmax: nano={nm:.3e} meg={mm:.3e} ratio={nm/mm:.3f} | cond: nano={nc:.1e} meg={mc:.1e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"matched": rows, "nano_only": nano_only, "meg_only": meg_only,
                   "n_matched": len(common), "n_nano_only": len(nano_only),
                   "n_meg_only": len(meg_only)}, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    import math
    sys.exit(main())
