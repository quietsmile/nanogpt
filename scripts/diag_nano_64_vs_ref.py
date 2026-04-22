"""Run nano on all 64 samples of iter 5989 (= offsets 383232..383295 in nano data),
compute aggregate loss using ref's formula, compare to ref logged 3.057512."""
import torch, torch.nn.functional as F
import sys, numpy as np
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

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

data = np.memmap('/root/nanogpt/data/cybertron_baseline/train.bin', dtype=np.int32, mode='r')
block = 8192
# iter 5989 samples = offsets 383232..383295 (per our verified dumps)
start = 383232
GBS = 64
EOD = 151643
MASK_ID = 160000

total_sum = 0.0
total_n = 0
per_sample = []
for i in range(GBS):
    X = torch.from_numpy(data[(start+i)*block : (start+i)*block + block].astype(np.int64)[None]).cuda()
    Y = torch.from_numpy(data[(start+i)*block + 1 : (start+i)*block + 1 + block].astype(np.int64)[None]).cuda()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(X, targets=Y)
    # Compute token-weighted contrib
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        logits, _ = model(X, targets=Y)
        pass  # using the returned loss (F.cross_entropy mean)

    # Get per-token CE without mean
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        lgts, _ = model(X, targets=Y)
    # Actually re-forward cleanly
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        X_in = X
        B, T = X_in.shape
        # Just compute logits via full model; reuse "lgts" above
        pass
    # Compute per-token CE with input-side mask
    ce = F.cross_entropy(lgts.float().view(-1, lgts.size(-1)), Y.view(-1), reduction='none')
    mask = ((X.view(-1) != EOD) & (X.view(-1) != MASK_ID)).float()
    sample_sum = (ce * mask).sum().item()
    sample_n = int(mask.sum().item())
    total_sum += sample_sum
    total_n += sample_n
    per_sample.append(sample_sum / sample_n)
    del X, Y, lgts
    torch.cuda.empty_cache()

token_weighted = total_sum / total_n
sample_mean = sum(per_sample) / len(per_sample)
print(f'nano @ iter 5989 (iter_5988 ckpt, 64 samples):')
print(f'  token-weighted mean = {token_weighted:.6f}')
print(f'  per-sample mean     = {sample_mean:.6f}')
print(f'ref logged iter 5989 lm_loss = 3.057512')
print(f'  Δ(token-weighted - ref) = {token_weighted - 3.057512:+.6f}')
print(f'  Δ(per-sample - ref)     = {sample_mean - 3.057512:+.6f}')
