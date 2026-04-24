"""Isolate: compare nano v_proj output to ref v, using ref's exact input."""
import torch, torch.nn.functional as F
import sys
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert, split_qkv
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"
def load_out(p):
    t = torch.load(p, weights_only=False, map_location="cuda")
    if isinstance(t, tuple): return t[0] if t else None
    return t

meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")
sd = convert(meg)

# Check weights
w_meg_qkv = meg['decoder.layers.1.self_attention.linear_qkv.weight'].cuda()
print(f"Megatron linear_qkv.weight shape={w_meg_qkv.shape} dtype={w_meg_qkv.dtype}")

# Split
q_conv, k_conv, v_conv = split_qkv(w_meg_qkv)
print(f"Split: q {q_conv.shape}, k {k_conv.shape}, v {v_conv.shape}")

# Ref qkv output (raw): [T,B,512]. For QKV interleaved layout.
ref_qkv_in  = load_out(f"{DUMP}/decoder.layers.1.self_attention.linear_qkv-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt")
ref_qkv_out = load_out(f"{DUMP}/decoder.layers.1.self_attention.linear_qkv-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
# Ref's linear_qkv includes layernorm PRE-linear! Let me check: linear_qkv forward-input is what goes IN, output is POST-norm-POST-linear.
# Actually Megatron fused linear_qkv does: y = linear(layer_norm(x)). So input = x, output = y.
# ref_qkv_out is [T, B, 512] = W @ norm(x).T

# Let me compute using nano's split weights + the nano layernorm
block1 = None
# Nano: ln_1 is pre-attention layernorm; q_proj/k_proj/v_proj after.
# The ref's qkv-input is BEFORE ln (the block's residual-stream input to the layernorm).
# So nano equivalent: ln_1(x) → q_proj/k_proj/v_proj.
# Load nano model to get ln_1
cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512,
    n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True,
    rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536,
    qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True,
    moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8,
    moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True,
    moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160,
    moe_routing_type="greedy", eod_token_id=151643, mask_loss_id=160000,
    seq_aux_balance_alpha=0.0, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()
block1 = model.transformer.h[1]

# Verify weights match
print(f"\nWeights: nano q_proj == split_qkv(meg)? {torch.equal(block1.attn.q_proj.weight, q_conv.cuda())}")
print(f"         nano k_proj == split_qkv(meg)? {torch.equal(block1.attn.k_proj.weight, k_conv.cuda())}")
print(f"         nano v_proj == split_qkv(meg)? {torch.equal(block1.attn.v_proj.weight, v_conv.cuda())}")
print(f"         nano ln_1.w == meg linear_qkv.layer_norm_weight? "
      f"{torch.equal(block1.ln_1.weight, meg['decoder.layers.1.self_attention.linear_qkv.layer_norm_weight'].cuda())}")

# Apply nano ln_1, then q/k/v proj to ref_qkv_in (which is pre-ln x)
x = ref_qkv_in.transpose(0, 1).contiguous()  # [B, T, C]
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    x_norm = block1.ln_1(x)
    q_nano = block1.attn.q_proj(x_norm)   # [B, T, 256]
    k_nano = block1.attn.k_proj(x_norm)   # [B, T, 128]
    v_nano = block1.attn.v_proj(x_norm)   # [B, T, 128]

# Ref core_in has q/k/v AFTER qk_layernorm AND RoPE. V is just linear (no modif).
core_in = torch.load(f"{DUMP}/decoder.layers.1.self_attention.core_attention-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt",
                     weights_only=False, map_location="cuda")
ref_v = core_in[2]  # [T, B, 2, 64]

# Reshape nano v to [T, B, 2, 64]
v_nano_tbhd = v_nano.view(x.shape[0], x.shape[1], 2, 64).transpose(0, 1).contiguous()

d = (v_nano_tbhd.float() - ref_v.float()).abs()
print(f"\n=== V comparison (simple linear, no LN/RoPE modifications) ===")
print(f"  nano V L1={v_nano_tbhd.abs().float().mean():.3e} max={v_nano_tbhd.abs().float().max():.3f}")
print(f"  ref  V L1={ref_v.abs().float().mean():.3e} max={ref_v.abs().float().max():.3f}")
print(f"  |nano - ref| L1={d.mean():.3e} max={d.max():.3e}")
print(f"  per-head diff mean: h0={d[:,:,0].mean():.3e} h1={d[:,:,1].mean():.3e}")
