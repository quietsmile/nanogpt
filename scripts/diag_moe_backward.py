"""Test nano MoE backward: feed ref forward input + ref backward grad,
compare nano's backward output to ref's dumped backward output.
"""
import torch, torch.nn.functional as F
import sys
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
    seq_aux_balance_alpha=0.0,  # disable aux so we only compare main backward
    use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()  # avoid training-specific paths (but autograd still works)
mlp = model.transformer.h[1].mlp

ref_fwd_in   = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt")
# backward-input = gradient coming INTO the mlp (upstream gradient at mlp output)
ref_bwd_in   = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-backward-input-tp0.1-pp0.1-ep3.4.pt")
# backward-output = gradient at the mlp's INPUT (what the mlp back-propagates)
ref_bwd_out  = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-backward-output-tp0.1-pp0.1-ep3.4.pt")
T, B, C = ref_fwd_in.shape
print(f"ref fwd_in shape={ref_fwd_in.shape}  dtype={ref_fwd_in.dtype}")
print(f"ref bwd_in shape={ref_bwd_in.shape}  dtype={ref_bwd_in.dtype}")
print(f"ref bwd_out shape={ref_bwd_out.shape} dtype={ref_bwd_out.dtype}")
print(f"ref bwd_in max_abs={ref_bwd_in.abs().max():.3e}")

# Prepare nano input that flatten as T-major (to match ref's [T,B,C] semantics)
x = ref_fwd_in.reshape(1, T*B, C).contiguous().requires_grad_(True)

with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out, aux = mlp(x)  # out: [1, T*B, C]

# Use ref's backward-input as dL/dout
g_upstream = ref_bwd_in.reshape(1, T*B, C).contiguous()
# Convert bf16 grad to same dtype as out for autograd (matching bf16 path)
g_upstream = g_upstream.to(out.dtype)

# Compute dL/dx via autograd
x_grad, = torch.autograd.grad(out, x, grad_outputs=g_upstream, retain_graph=False)
x_grad_tbc = x_grad.view(T, B, C).contiguous()

d = (x_grad_tbc.float() - ref_bwd_out.float()).abs()
print(f"\n=== backward dL/dx_mlp vs ref backward output ===")
print(f"  nano x_grad max_abs = {x_grad_tbc.abs().max():.3e}")
print(f"  ref bwd_out max_abs = {ref_bwd_out.abs().max():.3e}")
print(f"  L1   = {d.mean():.3e}")
print(f"  L_inf= {d.max():.3e}")
print(f"  relative L1 (L1/ref_mean_abs): {d.mean() / (ref_bwd_out.abs().float().mean() + 1e-20):.3e}")

# percentile breakdown
pos = d.max(dim=-1).values.flatten().cpu()
for q in [50, 90, 99, 99.9, 99.99]:
    print(f"    p{q:5.2f}: {torch.quantile(pos, q/100).item():.3e}")

# Ratio analysis
print("\n=== Ratio nano/ref (element-wise) ===")
ratio = x_grad_tbc.float() / (ref_bwd_out.float() + 1e-20)
abs_mask = ref_bwd_out.abs() > 1e-8
r_valid = ratio[abs_mask].cpu()
print(f"  mean ratio: {r_valid.mean():.3f}")
print(f"  median ratio: {r_valid.median():.3f}")
for q in [10, 50, 90]:
    print(f"    p{q:5.2f}: {torch.quantile(r_valid, q/100).item():.3f}")

# Check if nano grad includes ref grad as a sub-component
print("\n=== Subtract ref bwd_out from nano x_grad ===")
diff_tensor = x_grad_tbc.float() - ref_bwd_out.float()
print(f"  (nano - ref) L1={diff_tensor.abs().mean():.3e} max={diff_tensor.abs().max():.3e}")
print(f"  nano L1={x_grad_tbc.abs().float().mean():.3e}")
print(f"  ref  L1={ref_bwd_out.abs().float().mean():.3e}")
print(f"  (nano - 2*ref) L1={(x_grad_tbc.float() - 2*ref_bwd_out.float()).abs().mean():.3e}")
print(f"  (nano - 3*ref) L1={(x_grad_tbc.float() - 3*ref_bwd_out.float()).abs().mean():.3e}")
