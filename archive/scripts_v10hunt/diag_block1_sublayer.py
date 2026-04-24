"""Sublayer diff for block 1 with fused RoPE."""
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
cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536, qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True, moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160, moe_routing_type="greedy", eod_token_id=151643, mask_loss_id=160000, seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()
b1 = model.transformer.h[1]

# Ref dumps for block 1
ref_b0 = load_out(f"{DUMP}/decoder.layers.0-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_b1 = load_out(f"{DUMP}/decoder.layers.1-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_attn = load_out(f"{DUMP}/decoder.layers.1.self_attention-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_core = load_out(f"{DUMP}/decoder.layers.1.self_attention.core_attention-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_lproj = load_out(f"{DUMP}/decoder.layers.1.self_attention.linear_proj-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_pre_mlp_ln = load_out(f"{DUMP}/decoder.layers.1.pre_mlp_layernorm-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_mlp = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_shared = load_out(f"{DUMP}/decoder.layers.1.mlp.shared_experts-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")

x = ref_b0.transpose(0, 1).contiguous()  # [B, T, H] bf16
B, T = x.shape[:2]
a1 = b1.attn

def diff(a, b, name):
    d = (a.float() - b.float()).abs()
    print(f"  {name}: L_inf={d.max():.3e} L1={d.mean():.3e} nonzero={(d>0).sum().item()}/{d.numel()}")

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    # Forward through nano
    h_ln = b1.ln_1(x)
    q_raw = a1.q_proj(h_ln).view(B, T, 4, 64).transpose(1, 2)
    k_raw = a1.k_proj(h_ln).view(B, T, 2, 64).transpose(1, 2)
    v_raw = a1.v_proj(h_ln).view(B, T, 2, 64).transpose(1, 2)
    q = a1.q_layernorm(q_raw); k = a1.k_layernorm(k_raw)
    q, k = a1.rotary_emb(q, k, seq_len=T)  # now uses fused
    k_exp = k.unsqueeze(2).expand(B, 2, 2, T, 64).reshape(B, 4, T, 64)
    v_exp = v_raw.unsqueeze(2).expand(B, 2, 2, T, 64).reshape(B, 4, T, 64)
    attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
    attn_flat = attn_out.transpose(1, 2).contiguous().view(B, T, 256)
    diff(attn_flat.transpose(0, 1), ref_core, "block 1 core_attention")

    c_proj_out = a1.c_proj(attn_flat)
    diff(c_proj_out.transpose(0, 1), ref_lproj, "block 1 linear_proj")

    post_attn = x + c_proj_out
    mlp_in = b1.ln_2(post_attn)
    diff(mlp_in.transpose(0, 1), ref_pre_mlp_ln, "block 1 pre_mlp_layernorm")

    mlp = b1.mlp
    shared_out = mlp.shared_expert(mlp_in.view(-1, 512))
    diff(shared_out.view(B, T, 512).transpose(0, 1), ref_shared, "block 1 shared_expert")

    mlp_out, _ = mlp(mlp_in)
    diff(mlp_out.transpose(0, 1), ref_mlp, "block 1 MoE output")

    # Also feed ref's exact mlp input to MoE
    ref_mlp_in = torch.load(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")[0]
    ref_mlp_in_bth = ref_mlp_in.transpose(0, 1).contiguous()
    mlp_out_ref_in, _ = mlp(ref_mlp_in_bth)
    diff(mlp_out_ref_in.transpose(0, 1), ref_mlp, "block 1 MoE with REF mlp input")
