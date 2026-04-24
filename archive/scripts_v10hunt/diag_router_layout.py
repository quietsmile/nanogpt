"""Test: is the 10.8% probs mismatch just a T-major vs B-major flatten order bug?"""
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
    seq_aux_balance_alpha=0.0, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()
router = model.transformer.h[1].mlp.router

ref_mlp_in = load_out(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt")  # [T, B, C]
ref_router = torch.load(f"{DUMP}/decoder.layers.1.mlp.router-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt",
                        weights_only=False, map_location="cuda")
ref_probs, ref_rmap = ref_router

T, B, C = ref_mlp_in.shape
print(f"ref layout: [T={T}, B={B}, C={C}]")
print(f"ref_probs rows: {ref_probs.shape[0]} (= T*B = {T*B})")

# Method A: B-major (current nano behavior): transpose then flatten
x_A = ref_mlp_in.transpose(0, 1).contiguous().view(-1, C)  # [B*T, C] b-major

# Method B: T-major (what ref uses): direct flatten
x_B = ref_mlp_in.view(-1, C)  # [T*B, C] t-major

with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
    for name, x in [("B-major", x_A), ("T-major", x_B)]:
        logits = F.linear(x.float(), router.linear.weight.float())
        scores = torch.sigmoid(logits)
        scores_for_choice = scores + router.e_score_correction_bias.float()
        topk_weight, topk_idx = torch.topk(scores_for_choice, k=8, dim=-1, sorted=False)
        tokens_per_expert = torch.zeros_like(scores_for_choice, dtype=torch.float32).scatter_add_(
            1, topk_idx, torch.ones_like(topk_idx, dtype=torch.float32))
        map_ = tokens_per_expert > 0
        scores_selected = scores * map_.float()
        probs = scores_selected / (scores_selected.sum(dim=-1, keepdim=True) + 1e-20)

        d_probs = (probs - ref_probs.float()).abs()
        d_rmap = (map_.bool() != ref_rmap.bool()).float().mean()
        print(f"\n  {name}: probs L1={d_probs.mean():.3e} L_inf={d_probs.max():.3e} "
              f"mismatches={(d_probs>0).sum().item()}; rmap diff frac={d_rmap:.3e}")
