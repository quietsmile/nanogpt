"""Compare init spectrum at seed=300 vs other seeds for Dense + MoE arch.

For each seed, build the model fresh (CPU is fine), then for every weight
matrix dump:
  - mean / std / abs_max
  - σ_max / σ_min / cond  (full SVD)
  - "stickiness": top-1 row alignment with global mean direction

Diff aggregate stats (e.g. mean cond) across all layers per seed →
see if seed=300 has any cross-arch outlier signature.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_moe(seed: int):
    from nanogpt.model import GPT, GPTConfig
    torch.manual_seed(seed)
    cfg = GPTConfig(
        block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512,
        n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006,
        use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5,
        use_swiglu=True, ffn_hidden_size=1536, tie_embeddings=False,
        qk_layernorm=True, disable_scaled_init_method=True,
        use_moe=True, moe_layer_freq=[0] + [1]*8, num_experts=144,
        moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1,
        moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001,
        moe_shared_expert_hidden_size=160,
    )
    return GPT(cfg)


def build_dense(seed: int):
    from nanogpt.model import GPT, GPTConfig
    torch.manual_seed(seed)
    cfg = GPTConfig(
        block_size=8192, vocab_size=152064, n_layer=8, n_head=4, n_embd=528,
        n_kv_head=2, kv_channels=128, dropout=0.0, bias=False, init_std=0.006,
        use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5,
        use_swiglu=True, ffn_hidden_size=1824, tie_embeddings=False,
        qk_layernorm=True, disable_scaled_init_method=True,
        use_moe=False,
    )
    return GPT(cfg)


def matrix_stats(t: torch.Tensor) -> dict:
    """For 2D weight matrix, return per-matrix scalar fingerprints."""
    if t.dim() == 3:
        # Expert weights [E, h, in] → average over batch dim
        return {
            "mean": float(t.mean()),
            "std":  float(t.std()),
            "abs_max": float(t.abs().max()),
            "n_3d": t.shape[0],
        }
    if t.dim() != 2 or min(t.shape) < 2:
        return {"mean": float(t.mean()), "std": float(t.std()),
                "abs_max": float(t.abs().max())}
    # Sub-sample if huge (e.g. wte / lm_head 152064 × 512)
    r = t
    if t.numel() > 1_500_000:
        r = t[:512, :512] if t.shape[0] > 512 else t[:, :512]
    s = torch.linalg.svdvals(r)
    return {
        "mean": float(t.mean()),
        "std": float(t.std()),
        "abs_max": float(t.abs().max()),
        "sigma_max": float(s[0]),
        "sigma_min": float(s[-1]),
        "cond": float(s[0] / max(s[-1], 1e-12)),
        "sigma_ratio_p10_p90": float(s[len(s)*9//10] / max(s[len(s)//10], 1e-12)),
    }


def aggregate(seed: int, build_fn, label: str) -> dict:
    m = build_fn(seed)
    sd = m.state_dict()
    rows = []
    for n, p in sd.items():
        if p.numel() < 4:
            continue
        rows.append({"name": n, **matrix_stats(p.detach().cpu().float())})
    # cross-layer aggregates
    matrix_rows = [r for r in rows if "cond" in r]
    out = {
        "n_matrices": len(matrix_rows),
        "mean_cond": sum(r["cond"] for r in matrix_rows) / max(1, len(matrix_rows)),
        "max_cond":  max((r["cond"] for r in matrix_rows), default=0.0),
        "median_cond": sorted(r["cond"] for r in matrix_rows)[len(matrix_rows)//2] if matrix_rows else 0.0,
        "mean_sigma_max": sum(r["sigma_max"] for r in matrix_rows) / max(1, len(matrix_rows)),
        "mean_std": sum(r["std"] for r in rows) / len(rows),
        "max_abs": max(r["abs_max"] for r in rows),
        # rank-deficiency proxy
        "n_cond_gt_30": sum(1 for r in matrix_rows if r["cond"] > 30),
    }
    # Identify the worst (highest cond) layer
    worst = max(matrix_rows, key=lambda r: r.get("cond", 0)) if matrix_rows else {}
    out["worst_layer"] = worst.get("name", "")
    out["worst_cond"] = worst.get("cond", 0)
    return out


def main():
    seeds = [1337, 42, 7, 50, 100, 123, 200, 300, 456, 789]
    print("=== building MoE @ each seed ===", file=sys.stderr)
    moe_stats = {s: aggregate(s, build_moe, "MoE") for s in seeds}
    print("=== building Dense @ each seed ===", file=sys.stderr)
    dense_stats = {s: aggregate(s, build_dense, "Dense") for s in seeds}

    # Print summary
    def show(label, stats):
        print(f"\n=== {label} init aggregates per seed ===")
        keys = ["mean_cond","max_cond","median_cond","mean_sigma_max",
                "mean_std","max_abs","n_cond_gt_30"]
        print(f"{'seed':>5}  " + "  ".join(f"{k:>14s}" for k in keys) + "  worst_layer")
        for s, d in stats.items():
            row = "  ".join(f"{d[k]:>14.4g}" for k in keys)
            mark = " <-- s300" if s == 300 else ""
            print(f"{s:>5}  {row}  {d['worst_layer'][:40]}{mark}")
    show("MoE", moe_stats)
    show("Dense", dense_stats)

    out_path = ROOT / "reports" / "seed_300_init_diff.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    json.dump({"moe": moe_stats, "dense": dense_stats}, open(out_path, "w"), indent=1, default=str)
    print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
