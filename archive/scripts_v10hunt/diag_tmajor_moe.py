"""Test if feeding MoE with T-major (matching ref's order) matches ref output."""
import torch, torch.nn.functional as F
import transformer_engine.pytorch as te
import sys, os
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"
def load_out(p):
    t = torch.load(p, weights_only=False, map_location="cuda")
    return t[0] if isinstance(t, tuple) else t

os.environ['NANO_TE_MOE'] = '1'

meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")
sd = convert(meg)
cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536, qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True, moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160, moe_routing_type="greedy", eod_token_id=151643, mask_loss_id=160000, seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()
mlp = model.transformer.h[1].mlp

# Ref's mlp input is in [T, B, C] format (sbhd). Flatten as T*B (T-major).
ref_mlp_in_tbc = torch.load(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")[0]
ref_mlp_out = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
# [T, B, C]
T, B, C = ref_mlp_in_tbc.shape

def diff(a, b, name):
    d = (a.float() - b.float()).abs()
    print(f"  {name}: L_inf={d.max():.3e} L1={d.mean():.3e} nonzero={(d>0).sum().item()}/{d.numel()}")

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    # Method A: current nano (B-major internal via view as [B, T, C] then flatten)
    x_bth = ref_mlp_in_tbc.transpose(0, 1).contiguous()  # [B, T, C]
    out_a, _ = mlp(x_bth)
    # out_a is [B, T, C], transpose to [T, B, C]
    out_a_tbc = out_a.transpose(0, 1).contiguous()
    diff(out_a_tbc, ref_mlp_out, "A: nano B-major flatten")

    # Method B: feed T-major layout — pretend shape is [T=8192, B=4, C], view [T*B, C]
    # Use shape trick: pass x as [B_fake=1, T_fake=T*B, C]. Then internal view(-1, C) gives T-major flat.
    x_tmajor_viewed = ref_mlp_in_tbc.view(T * B, C).unsqueeze(0)  # [1, T*B, C]
    out_b_single, _ = mlp(x_tmajor_viewed)
    # out_b_single is [1, T*B, C]. Reshape back to [T, B, C]
    out_b_tbc = out_b_single.view(T, B, C)
    diff(out_b_tbc, ref_mlp_out, "B: nano T-major flatten")
