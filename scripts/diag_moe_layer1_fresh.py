"""Fresh comparison: feed nano block 1 MLP with ref's exact input, using ref weights.
Goal: is the residual 7.5e-4 truly bf16 ULP, or is there a localized bug?
"""
import torch, torch.nn.functional as F
import sys, os
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")

from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"
def load_out(p):
    t = torch.load(p, weights_only=False, map_location="cuda")
    return t[0] if isinstance(t, tuple) else t

meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")
sd = convert(meg)
cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512,
    n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True,
    rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536,
    qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True,
    moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8,
    moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True,
    moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160,
    moe_routing_type="greedy", eod_token_id=151643, mask_loss_id=160000,
    seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()
mlp = model.transformer.h[1].mlp

# Ref mlp IO: shape [T=8192, B=4, C=512]
ref_mlp_in = torch.load(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt",
                        weights_only=False, map_location="cuda")[0]
ref_mlp_out = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_shared = load_out(f"{DUMP}/decoder.layers.1.mlp.shared_experts-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")

print(f"ref_mlp_in  shape={ref_mlp_in.shape} dtype={ref_mlp_in.dtype}")
print(f"ref_mlp_out shape={ref_mlp_out.shape} dtype={ref_mlp_out.dtype}")
print(f"ref_mlp_out max abs = {ref_mlp_out.abs().max():.3f}")
print(f"ref_shared max abs  = {ref_shared.abs().max():.3f}")

# Nano MLP expects [B, T, C]. Ref is [T, B, C] (sbhd layout). Transpose.
x = ref_mlp_in.transpose(0, 1).contiguous()  # [B, T, C]

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out, aux = mlp(x)  # out shape [B, T, C]

out_tbc = out.transpose(0, 1).contiguous()  # [T, B, C] to match ref

print(f"\n--- Total MLP output comparison (ref ALREADY includes shared) ---")
d = (out_tbc.float() - ref_mlp_out.float()).abs()
print(f"  L1={d.mean():.3e}  L_inf={d.max():.3e}")
print(f"  max diff fraction (d > ref * 1e-3): {(d > ref_mlp_out.abs().float() * 1e-3).float().mean()*100:.2f}%")

# Per-position max diff (across channels) histogram
pos_max = d.max(dim=-1).values  # [T, B]
pos_max_flat = pos_max.flatten().cpu()
print(f"\n  per-position max diff percentiles (T×B={pos_max_flat.numel()} positions):")
for q in [50, 90, 99, 99.9, 99.99]:
    print(f"    p{q:5.2f}: {torch.quantile(pos_max_flat, q/100).item():.3e}")

# Extreme outlier positions
top_idx = pos_max_flat.argsort(descending=True)[:5]
print(f"\n  Top 5 outlier positions (t, b, max_diff, ref_mag):")
for i in top_idx:
    t, b = i.item() // 4, i.item() % 4
    print(f"    t={t} b={b}: diff={pos_max[t,b].item():.3e} ref|max|={ref_mlp_out[t,b].abs().max().item():.3f}")

# Break down: routed vs shared
ref_routed = ref_mlp_out - ref_shared  # our routed expectation
# nano returns combined out; let's compute routed-only separately by zeroing shared.
# Actually, nano's mlp internally adds shared_out + routed_out. To compare routed only,
# we'd need to re-run with shared off. Skip for now.
print("\n--- conclusion pending: is diff localized (bug) or uniform (bf16 ULP)? ---")
