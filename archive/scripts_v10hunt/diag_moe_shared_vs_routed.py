"""Decompose MoE layer 1 to isolate which component has the bug.
Compare: shared expert output, routed expert output.
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
    seq_aux_balance_alpha=0.0, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()
mlp = model.transformer.h[1].mlp

ref_mlp_in    = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt")
ref_mlp_out   = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_shared    = load_out(f"{DUMP}/decoder.layers.1.mlp.shared_experts-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_routed    = ref_mlp_out - ref_shared

def diff(a, b, name):
    d = (a.float() - b.float()).abs()
    relmag = b.abs().float().mean() + 1e-20
    print(f"  {name}:")
    print(f"    L1={d.mean():.3e} L_inf={d.max():.3e}  ref_max_abs={b.abs().max():.3f}")
    print(f"    L1/ref_mean = {d.mean()/relmag:.3e} (relative)")

# Shared expert is called on x_flat inside MoEFFN. Feed it flat [B*T, C] directly.
# Ref mlp input layout is [T, B, C]. The MLP internally does x_flat = x.view(-1, C).
# So we'll feed the full [B, T, C] for mlp() call but [B*T, C] for shared_expert().
x_btc = ref_mlp_in.transpose(0, 1).contiguous()  # [B, T, C]
B, T, C = x_btc.shape
x_flat = x_btc.reshape(-1, C)  # [B*T, C]
x = x_btc
print(f"input B={B} T={T} C={C}, input mean abs = {x.abs().mean().item():.4f}")

# --- SHARED EXPERT ONLY ---
print("\n=== SHARED EXPERT ONLY ===")
shared = mlp.shared_expert
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    nano_shared_bf = shared(x_flat)  # [B*T, C]
# Also fp32
import copy
shared32 = copy.deepcopy(shared).float()
with torch.no_grad():
    nano_shared_fp = shared32(x_flat.float())

# Reshape [B*T, C] → [T, B, C] to match ref
nano_shared_bf_tbc = nano_shared_bf.view(B, T, C).transpose(0, 1).contiguous()
nano_shared_fp_tbc = nano_shared_fp.view(B, T, C).transpose(0, 1).contiguous()
diff(nano_shared_bf_tbc, ref_shared, "nano bf16 shared vs ref_shared")
diff(nano_shared_fp_tbc, ref_shared, "nano fp32 shared vs ref_shared")

# --- ROUTED ONLY ---
# Can't easily isolate routed from nano's MoEFFN without modifying. Use full output minus shared.
print("\n=== ROUTED (total - shared) ===")
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    total_out, _ = mlp(x)
total_out_tbc = total_out.transpose(0, 1).contiguous()

nano_routed_bf = total_out_tbc.float() - nano_shared_bf_tbc.float()
diff(nano_routed_bf, ref_routed, "nano bf16 (total - shared) vs ref_routed")

# Summary
print("\n--- Conclusion ---")
print("If shared fp32 vs ref is LARGE, bug is in shared expert compute")
print("If routed is LARGE, bug is in routing + expert dispatch")
