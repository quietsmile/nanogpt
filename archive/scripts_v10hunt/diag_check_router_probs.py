"""Check if nano router probs bitwise match ref router probs."""
import torch, sys, os
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
os.environ['NANO_TE_MOE'] = '1'
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

DUMP = "/newcpfs/user/yuchen/karpathy/cybertron_dump/dumps"
meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0005988")
sd = convert(meg)
cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536, qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True, moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160, moe_routing_type="greedy", eod_token_id=151643, mask_loss_id=160000, seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False)
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.eval()
b1 = model.transformer.h[1]

ref_mlp_in = torch.load(f"{DUMP}/decoder.layers.1.mlp-iter5988-mbs0-forward-input-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")[0]
ref_probs, ref_rmap = torch.load(f"{DUMP}/decoder.layers.1.mlp.router-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt", weights_only=False, map_location="cuda")

# Use T-major flat to match ref
x_flat = ref_mlp_in.view(-1, 512)
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    topk_idx, weights, _ = b1.mlp.router(x_flat)

S, K = topk_idx.shape
probs_nano = torch.zeros(S, 144, dtype=torch.float32, device="cuda")
probs_nano.scatter_(1, topk_idx, weights.float())

d = (probs_nano - ref_probs.float()).abs()
print(f"ref_probs dtype: {ref_probs.dtype}")
print(f"probs nano vs ref: L_inf={d.max():.3e} L1={d.mean():.3e} nonzero={(d>0).sum().item()}/{d.numel()}")

# Check: for each token, sum of probs. Ref sums to 1? Nano's?
print(f"ref probs sum(-1) mean: {ref_probs.float().sum(-1).mean():.6f}, std: {ref_probs.float().sum(-1).std():.3e}")
print(f"nano probs sum(-1) mean: {probs_nano.sum(-1).mean():.6f}, std: {probs_nano.sum(-1).std():.3e}")
