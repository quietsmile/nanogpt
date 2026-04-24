"""Compare bmm-path vs explicit-per-expert-loop of MoEFFN on same weights/input on CUDA.

Goal: if bmm path is numerically equivalent to the reference loop, the bug is elsewhere.
If bmm deviates from loop, the bmm/pad/unsort logic has a bug.
"""
import sys, os, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, '/root/nanogpt')
from model import GPTConfig, MoEFFN

torch.manual_seed(0)
cfg = GPTConfig(
    block_size=8192, vocab_size=152064,
    n_layer=1, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64,
    use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5,
    use_swiglu=True, ffn_hidden_size=1536, qk_layernorm=True,
    tie_embeddings=False, init_std=0.006, disable_scaled_init_method=True,
    use_moe=True, moe_layer_freq=[1], num_experts=144, moe_ffn_hidden_size=160,
    moe_router_topk=8, moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True,
    moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160,
)
m = MoEFFN(cfg).cuda().bfloat16()
m.eval()
x = torch.randn(1, 8192, 512, device='cuda', dtype=torch.bfloat16)

# Path A: bmm (current)
with torch.no_grad():
    out_bmm, _ = m(x)

# Path B: manual per-expert loop using same router decisions
with torch.no_grad():
    B, T, C = x.shape
    x_flat = x.view(B * T, C)
    topk_idx, weights, raw_scores = m.router(x_flat)
    S = x_flat.shape[0]; K = m.topk; E = m.num_experts
    # Manual per-token-per-expert (slow but canonical)
    out_routed = torch.zeros_like(x_flat)
    flat_tokens_exp = x_flat.unsqueeze(1).expand(S, K, C).reshape(S * K, C)
    flat_experts = topk_idx.reshape(-1)
    flat_weights = weights.reshape(-1).to(x_flat.dtype)
    for e in range(E):
        mask = (flat_experts == e)
        if not mask.any(): continue
        tok = flat_tokens_exp[mask]  # [n, C]
        ws = flat_weights[mask].unsqueeze(1)
        h = F.silu(tok @ m.gate_weight[e]) * (tok @ m.up_weight[e])
        o = h @ m.down_weight[e] * ws
        # Scatter to each source token's row
        src_token = torch.arange(S, device=x.device).unsqueeze(1).expand(S, K).reshape(-1)
        src_masked = src_token[mask]
        out_routed.index_add_(0, src_masked, o.to(out_routed.dtype))
    out_shared = m.shared_expert(x_flat) if m.shared_expert is not None else 0
    out_manual = (out_shared + out_routed).view(B, T, C)

diff = (out_bmm - out_manual).float().abs()
print(f'bmm vs manual:')
print(f'  max_abs_diff: {diff.max().item():.4e}')
print(f'  mean_abs_diff: {diff.mean().item():.4e}')
print(f'  max rel (vs |manual|): {(diff / (out_manual.float().abs() + 1e-10)).max().item():.4e}')
print(f'  out_bmm[0,0,:6]: {out_bmm[0,0,:6].tolist()}')
print(f'  out_manual[0,0,:6]: {out_manual[0,0,:6].tolist()}')

# Comparable sub-stats
print(f'bmm std: {out_bmm.float().std().item():.4f}, max: {out_bmm.float().abs().max().item():.4f}')
print(f'man std: {out_manual.float().std().item():.4f}, max: {out_manual.float().abs().max().item():.4f}')
