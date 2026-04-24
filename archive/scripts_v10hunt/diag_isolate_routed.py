"""Isolate routed expert contribution vs ref - compute ref_routed = ref_mlp - ref_shared and compare."""
import torch, torch.nn.functional as F
import sys
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"
def load_out(p):
    t = torch.load(p, weights_only=False, map_location="cuda")
    return t[0] if isinstance(t, tuple) else t

import os
os.environ['NANO_TE_MOE'] = '1'

meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")
sd = convert(meg)
cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536, qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True, moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160, moe_routing_type="greedy", eod_token_id=151643, mask_loss_id=160000, seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()
b1 = model.transformer.h[1]
mlp = b1.mlp

ref_mlp_in = torch.load(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")[0]
ref_mlp = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")
ref_shared = load_out(f"{DUMP}/decoder.layers.1.mlp.shared_experts-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")

x_bth = ref_mlp_in.transpose(0, 1).contiguous()
B, T, H_ = x_bth.shape
x_flat = x_bth.view(-1, H_)

def diff(a, b, name):
    d = (a.float() - b.float()).abs()
    print(f"  {name}: L_inf={d.max():.3e} L1={d.mean():.3e} nonzero={(d>0).sum().item()}/{d.numel()}")

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    # Nano shared
    shared_out = mlp.shared_expert(x_flat)
    # Full MoE output
    mlp_out, _ = mlp(x_bth)
    # Routed part from nano = mlp_out - shared_out
    mlp_out_tb = mlp_out.transpose(0, 1).contiguous()  # [T, B, C]
    shared_tb = shared_out.view(B, T, H_).transpose(0, 1).contiguous()
    nano_routed = mlp_out_tb - shared_tb
    # Ref routed = ref_mlp - ref_shared
    ref_routed = ref_mlp - ref_shared
    diff(nano_routed, ref_routed, "routed_experts output (nano - shared vs ref - shared)")
    diff(shared_tb, ref_shared, "shared_expert output")
    diff(mlp_out_tb, ref_mlp, "full MoE output (for reference)")
