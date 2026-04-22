"""Use ref's core_attention output directly in nano path — see what final block 0 diff is."""
import torch, torch.nn.functional as F
import sys
sys.path.insert(0, "/root/nanogpt")
sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"
def load_out(p):
    t = torch.load(p, weights_only=False, map_location="cuda")
    return t[0] if isinstance(t, tuple) else t

meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")
sd = convert(meg)
cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536, qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True, moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160, moe_routing_type='greedy', eod_token_id=151643, mask_loss_id=160000, seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()
b0 = model.transformer.h[0]

# Load ref tensors
ref_embed = load_out(f"{DUMP}/embedding-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")  # [T, B, H]
ref_core = load_out(f"{DUMP}/decoder.layers.0.self_attention.core_attention-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")  # [T, B, 256]
ref_b0 = load_out(f"{DUMP}/decoder.layers.0-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")  # [T, B, H]

# nano format: [B, T, *]
x = ref_embed.transpose(0, 1).contiguous()
core_ref_bth = ref_core.transpose(0, 1).contiguous()  # [B, T, 256]
B, T, H = x.shape

def diff(a, b, name):
    d = (a.float() - b.float()).abs()
    print(f"  {name}: L_inf={d.max():.3e} L1={d.mean():.3e} nonzero={(d>0).sum().item()}/{d.numel()}")

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    # Pass ref's core_attention as attn output, run nano's linear_proj + residual + MLP
    c_proj_out = b0.attn.c_proj(core_ref_bth)
    post_attn = x + c_proj_out
    mlp_in = b0.ln_2(post_attn)
    mlp_out = b0.mlp(mlp_in)
    out = post_attn + mlp_out
    diff(out.transpose(0, 1), ref_b0, "block 0 with ref core_attention fed in")

    # For comparison: full nano path (same as isolated test baseline)
    out_full, _ = b0(x)
    diff(out_full.transpose(0, 1), ref_b0, "block 0 full nano path")
