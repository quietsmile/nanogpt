"""bf16 ULP boundary test: scale input, check if output diff scales proportionally.
If diff is bf16 ULP noise, it scales linearly with magnitude (relative diff ~ 2^-7 = 7.8e-3).
If diff is bug/systematic, it does NOT scale proportionally.
"""
import torch, torch.nn.functional as F
import sys, os
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"

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
    seq_aux_balance_alpha=0.0,  # disable aux to avoid affecting forward
    use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()
mlp = model.transformer.h[1].mlp

ref_mlp_in = torch.load(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt",
                        weights_only=False, map_location="cuda")[0]
ref_mlp_out = torch.load(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt",
                         weights_only=False, map_location="cuda")
if isinstance(ref_mlp_out, tuple): ref_mlp_out = ref_mlp_out[0]

# For scaled tests, we can't compare to ref directly (ref only has one magnitude).
# Instead: run nano TWICE with same input — once normal, once with scaling that should
# produce the SAME mathematical output. Since MLP is non-linear (SwiGLU + ReLU-like),
# we can't scale arbitrarily. But we CAN compare diff stability of nano's own compute
# under different bf16 rounding regimes.
#
# Better approach: compare nano's output to itself when run with tf32 / fp32.
# If the 7.5e-4 residual = bf16 rounding, then fp32 nano should be essentially = ref.
# If there's a bug, fp32 nano also differs from ref by similar amount.

x_bf = ref_mlp_in.transpose(0, 1).contiguous()  # [B, T, C] bf16

# Run 1: standard bf16 autocast
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out_bf, _ = mlp(x_bf)
out_bf_tbc = out_bf.transpose(0, 1).contiguous()

# Run 2: force everything to fp32 (no autocast)
x_fp = x_bf.float()
mlp_fp = mlp  # same weights — they are bf16 already but F.linear will upcast
with torch.no_grad():
    # Manually disable autocast by NOT using autocast context + pass fp32 input.
    # But model weights might be bf16. Let's cast a copy to fp32.
    import copy
    mlp32 = copy.deepcopy(mlp).float()
    out_fp, _ = mlp32(x_fp)
out_fp_tbc = out_fp.transpose(0, 1).contiguous()

def diff(a, b, name):
    d = (a.float() - b.float()).abs()
    print(f"  {name}: L1={d.mean():.3e} L_inf={d.max():.3e}")
    pos = d.max(dim=-1).values.flatten()
    ref_mag = b.abs().float().max(dim=-1).values.flatten() + 1e-20
    rel = (pos / ref_mag)
    print(f"    relative diff (diff / ref_max_per_pos): p50={torch.quantile(rel, 0.5):.3e} "
          f"p99={torch.quantile(rel, 0.99):.3e} p99.9={torch.quantile(rel, 0.999):.3e}")

print("=== bf16 vs ref ===")
diff(out_bf_tbc, ref_mlp_out, "nano bf16 vs ref")

print("\n=== fp32 vs ref ===")
diff(out_fp_tbc, ref_mlp_out, "nano fp32 vs ref")

print("\n=== bf16 vs fp32 (both nano) ===")
diff(out_bf_tbc, out_fp_tbc, "nano bf16 vs nano fp32")

print("\n--- Interpretation ---")
print("If bf16-vs-ref ~= bf16-vs-fp32, residual IS bf16 rounding (fp32 would match ref).")
print("If fp32-vs-ref is still large, residual is a REAL compute difference (bug).")
