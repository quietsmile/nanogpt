"""Decompose nano backward into shared-contrib and routed-contrib. Compare each
to ref to localize the bug."""
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
import os as _os
NORM_TOPK = _os.environ.get('NORM_TOPK', '1') == '1'
print(f"\n>>> moe_norm_topk_prob = {NORM_TOPK} <<<\n")
cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512,
    n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True,
    rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536,
    qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True,
    moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8,
    moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=NORM_TOPK,
    moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160,
    moe_routing_type="greedy", eod_token_id=151643, mask_loss_id=160000,
    seq_aux_balance_alpha=0.0, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()
mlp = model.transformer.h[1].mlp
shared = mlp.shared_expert

ref_fwd_in  = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt")
# NOTE: register_full_backward_hook passes (grad_input, grad_output):
#   backward-INPUT  = grad_input  = dL/dx  (gradient at module INPUT)
#   backward-OUTPUT = grad_output = dL/dy  (gradient at module OUTPUT, upstream)
ref_grad_dx = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-backward-input-tp0.1-pp0.1-ep3.4.pt")
ref_grad_dy = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-backward-output-tp0.1-pp0.1-ep3.4.pt")
T, B, C = ref_fwd_in.shape

# Test 1: nano SHARED-only backward
x1 = ref_fwd_in.view(-1, C).contiguous().requires_grad_(True)  # [T*B, C]
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    y_shared = shared(x1)  # [T*B, C]
g_shared_in = ref_grad_dy.view(-1, C).contiguous().to(y_shared.dtype)
grad_shared, = torch.autograd.grad(y_shared, x1, grad_outputs=g_shared_in)
grad_shared_tbc = grad_shared.view(T, B, C)
print(f"\n=== nano SHARED-only backward ===")
print(f"  nano grad_shared L1={grad_shared_tbc.abs().float().mean():.3e} max={grad_shared_tbc.abs().float().max():.3e}")

# Test 2: nano FULL (shared + routed)
x2 = ref_fwd_in.view(1, T*B, C).contiguous().requires_grad_(True)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    y_full, _ = mlp(x2)
g_full = ref_grad_dy.view(1, T*B, C).contiguous().to(y_full.dtype)
grad_full, = torch.autograd.grad(y_full, x2, grad_outputs=g_full)
grad_full_tbc = grad_full.view(T, B, C)
print(f"\n=== nano FULL mlp backward ===")
print(f"  nano grad_full L1={grad_full_tbc.abs().float().mean():.3e} max={grad_full_tbc.abs().float().max():.3e}")

# Test 3: nano ROUTED = FULL - SHARED
grad_routed_tbc = grad_full_tbc - grad_shared_tbc
print(f"\n=== nano ROUTED-only (full - shared) backward ===")
print(f"  nano grad_routed L1={grad_routed_tbc.abs().float().mean():.3e} max={grad_routed_tbc.abs().float().max():.3e}")

print(f"\n=== Compare to ref bwd_out ===")
print(f"  ref bwd_out L1={ref_grad_dx.abs().float().mean():.3e} max={ref_grad_dx.abs().float().max():.3e}")
d_full = (grad_full_tbc.float() - ref_grad_dx.float()).abs()
d_shared = (grad_shared_tbc.float() - ref_grad_dx.float()).abs()
print(f"  |nano_full - ref| L1={d_full.mean():.3e} max={d_full.max():.3e}")
print(f"  |nano_shared_only - ref| L1={d_shared.mean():.3e} max={d_shared.max():.3e}  (suspicious if near 0!)")

# If nano_full looks like ref + nano_shared (double-counting), check:
test = grad_full_tbc - 2*grad_shared_tbc
d_test = (test.float() - ref_grad_dx.float()).abs()
print(f"  |nano_full - 2*nano_shared - ref| L1={d_test.mean():.3e} max={d_test.max():.3e}")

# Another angle: nano_full - (nano_shared / 2) = ref_routed?
ref_magnitude_ratio = ref_grad_dx.abs().float().mean() / grad_shared_tbc.abs().float().mean()
print(f"\n  mag ratio ref/nano_shared = {ref_magnitude_ratio:.3f}")

# Cosine similarity
def cossim(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()

print(f"\n=== Directional alignment ===")
print(f"  cos(nano_full, ref): {cossim(grad_full_tbc, ref_grad_dx):.4f}")
print(f"  cos(nano_shared, ref): {cossim(grad_shared_tbc, ref_grad_dx):.4f}")
print(f"  cos(nano_routed, ref): {cossim(grad_routed_tbc, ref_grad_dx):.4f}")

# Per-token cosine
def per_token_cos(a, b):
    a_f = a.view(-1, C).float()
    b_f = b.view(-1, C).float()
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=1)
ptc = per_token_cos(grad_full_tbc, ref_grad_dx).cpu()
print(f"  per-token cos(nano_full, ref):")
for q in [1, 10, 50, 90, 99]:
    print(f"    p{q:4.1f}: {torch.quantile(ptc, q/100).item():.4f}")
