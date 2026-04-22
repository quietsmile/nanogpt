"""Test block 1 self_attention forward + backward vs ref."""
import torch, torch.nn.functional as F
import sys
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"
def load_out(p):
    t = torch.load(p, weights_only=False, map_location="cuda")
    if isinstance(t, tuple):
        return t[0] if t else None
    return t

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
    seq_aux_balance_alpha=0.0, use_eod_attn_mask=False,
    attention_impl='fp32_manual')  # force fp32 softmax to match ref's attention_softmax_in_fp32=True
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()
attn = model.transformer.h[1].attn  # the CausalSelfAttention

ref_in     = load_out(f"{DUMP}/decoder.layers.1.self_attention-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt")
ref_out    = load_out(f"{DUMP}/decoder.layers.1.self_attention-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_grad_dx= load_out(f"{DUMP}/decoder.layers.1.self_attention-iter5988-mbs0-backward-input-tp0.1-pp0.1-ep3.4.pt")
ref_grad_dy= load_out(f"{DUMP}/decoder.layers.1.self_attention-iter5988-mbs0-backward-output-tp0.1-pp0.1-ep3.4.pt")
T, B, C = ref_in.shape
print(f"ref_in shape {ref_in.shape} dtype {ref_in.dtype}")

# Nano attn expects POST-norm input. Ref's self_attention-forward-input is PRE-norm
# (Megatron fuses layernorm into linear_qkv). Apply nano's ln_1 to ref_in first.
block1 = model.transformer.h[1]
# Pre-norm x is the actual input (matching ref's "self_attention-forward-input"
# which is pre-norm due to Megatron's fused layernorm-linear_qkv).
x = ref_in.transpose(0, 1).contiguous().requires_grad_(True)

with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    x_normed = block1.ln_1(x)      # fused in Megatron's linear_qkv
    y = attn(x_normed)  # [B, T, C]

y_tbc = y.transpose(0, 1).contiguous()  # back to [T, B, C] to match ref
d_fwd = (y_tbc.float() - ref_out.float()).abs()
print(f"\n=== Attention FORWARD ===")
print(f"  nano out L1={y_tbc.abs().float().mean():.3e} max={y_tbc.abs().float().max():.3f}")
print(f"  ref  out L1={ref_out.abs().float().mean():.3e} max={ref_out.abs().float().max():.3f}")
print(f"  |nano - ref| L1={d_fwd.mean():.3e} max={d_fwd.max():.3e}")

g_dy = ref_grad_dy.transpose(0, 1).contiguous().to(y.dtype)  # [B, T, C]
grad_x, = torch.autograd.grad(y, x, grad_outputs=g_dy)
grad_x_tbc = grad_x.transpose(0, 1).contiguous()  # [T, B, C] to match ref

d_bwd = (grad_x_tbc.float() - ref_grad_dx.float()).abs()

def cossim(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten().unsqueeze(0),
                                                  b.float().flatten().unsqueeze(0)).item()

print(f"\n=== Attention BACKWARD ===")
print(f"  nano grad_dx L1={grad_x_tbc.abs().float().mean():.3e} max={grad_x_tbc.abs().float().max():.3e}")
print(f"  ref  grad_dx L1={ref_grad_dx.abs().float().mean():.3e} max={ref_grad_dx.abs().float().max():.3e}")
print(f"  |nano - ref| L1={d_bwd.mean():.3e} max={d_bwd.max():.3e}")
print(f"  cos(nano, ref) = {cossim(grad_x_tbc, ref_grad_dx):.6f}")
