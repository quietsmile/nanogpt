"""Trace cumulative forward drift layer-by-layer with ref ckpt iter 5988."""
import torch, torch.nn.functional as F
import sys, numpy as np
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"
def load_out(p):
    t = torch.load(p, weights_only=False, map_location="cuda")
    if isinstance(t, tuple): return t[0] if t else None
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
    seq_aux_balance_alpha=0.0, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()

# Ref dumps for each decoder layer output
layer_outs = {}
for L in range(9):
    layer_outs[L] = load_out(f"{DUMP}/decoder.layers.{L}-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
    print(f"ref layer {L} output shape={layer_outs[L].shape} dtype={layer_outs[L].dtype}  max_abs={layer_outs[L].abs().max():.3f}")

# Ref embedding output (= decoder layer 0 input)
ref_emb = load_out(f"{DUMP}/embedding-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
print(f"\nref embedding output shape={ref_emb.shape} max_abs={ref_emb.abs().max():.3f}")

# Starting from ref_emb, feed through nano blocks one by one, compare each layer's output
T, B, C = ref_emb.shape
x_btc = ref_emb.transpose(0, 1).contiguous()  # [B, T, C] nano layout

# Use nano's blocks
print("\n=== Per-layer cumulative diff (nano starts from ref embedding) ===")
print(f"{'L':>2s} {'nano→L+1 L1':>15s} {'ref_L+1 L1':>15s} {'|diff| L1':>12s} {'|diff| max':>12s} {'ratio_diff/ref_mean':>22s}")
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    for L in range(9):
        block = model.transformer.h[L]
        x_btc, aux = block(x_btc)
        # Convert to [T, B, C] to compare with ref
        x_tbc = x_btc.transpose(0, 1).contiguous()
        ref_out = layer_outs[L]
        d = (x_tbc.float() - ref_out.float()).abs()
        nano_L1 = x_tbc.abs().float().mean().item()
        ref_L1  = ref_out.abs().float().mean().item()
        print(f"{L:>2d} {nano_L1:>15.5e} {ref_L1:>15.5e} {d.mean().item():>12.4e} {d.max().item():>12.4e} {d.mean().item()/ref_L1:>22.4e}")

# Final layernorm + lm_head
print("\n=== final_layernorm + lm_head ===")
x_btc_normed = model.transformer.ln_f(x_btc)
logits = F.linear(x_btc_normed.float(), model.lm_head.weight.float())  # fp32
# Compare to ref output_layer output
ref_logits = load_out(f"{DUMP}/output_layer-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
print(f"  ref logits shape = {ref_logits.shape}")
# nano logits [B, T, V] → [T, B, V]
logits_tbv = logits.transpose(0, 1).contiguous()
d_logits = (logits_tbv.float() - ref_logits.float()).abs()
print(f"  |logits diff| L1={d_logits.mean():.4e} max={d_logits.max():.4e}")
print(f"  nano logits L1={logits_tbv.abs().float().mean():.4e} max={logits_tbv.abs().float().max():.4e}")
print(f"  ref  logits L1={ref_logits.abs().float().mean():.4e} max={ref_logits.abs().float().max():.4e}")
