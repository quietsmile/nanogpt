"""Compare aux loss value: nano's aux_sum/n_moe_layers vs ref's sequence_wise_balance_loss."""
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
    seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.train()  # Enable training to compute aux
mlp = model.transformer.h[1].mlp

ref_mlp_in = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt")
T, B, C = ref_mlp_in.shape
# T-major flatten (matching ref layout)
x = ref_mlp_in.reshape(1, T*B, C).contiguous()

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out, aux = mlp(x)

print(f"nano aux (layer 1) = {aux.item():.6f}")
print(f"ref seq_wise_bal_loss (iter 1 log)       = 1.087672")
print(f"ref seq_wise_bal_loss (iter 5988 log)    = ? — let me fetch")

import re
with open('/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/logs/rank-0-1-scaling_moe_00196-run.log') as f:
    for ln in f:
        m = re.search(r'iteration\s+5988/.* sequence_wise_balance_loss: ([\d.E+\-]+)', ln)
        if m:
            print(f'ref iter 5988 seq_wise_bal_loss = {float(m.group(1)):.6f}')
            break
