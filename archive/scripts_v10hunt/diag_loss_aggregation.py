"""Test loss aggregation: naive sample-mean vs token-weighted mean."""
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
start = 5988 * 64
GBS = 64
EOD = 151643
MASK_ID = 160000

naive_losses = []
total_sum_loss = 0.0
total_n_unmasked = 0

for i in range(GBS):
    X = data[(start+i)*block : (start+i)*block + block].astype(np.int64)[None]
    Y = data[(start+i)*block + 1 : (start+i)*block + 1 + block].astype(np.int64)[None]
    X_t = torch.from_numpy(X).cuda(); Y_t = torch.from_numpy(Y).cuda()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, loss_naive = model(X_t, targets=Y_t)
    naive_losses.append(loss_naive.item())
    # Recompute token-weighted: sum of per-token CE on unmasked positions
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        logits_fp = logits.float()
        ce_per_tok = F.cross_entropy(logits_fp.view(-1, logits_fp.size(-1)),
                                      Y_t.view(-1), ignore_index=-999, reduction='none')  # no builtin ignore
        # Apply mask: where INPUT (idx) is EOD or MASK_ID, mask loss
        idx_flat = X_t.view(-1)
        mask = (idx_flat != EOD) & (idx_flat != MASK_ID)
        sample_sum = (ce_per_tok * mask.float()).sum().item()
        sample_n = mask.sum().item()
        total_sum_loss += sample_sum
        total_n_unmasked += sample_n
    del X_t, Y_t, logits, logits_fp
    torch.cuda.empty_cache()

print(f"per-sample losses: min={min(naive_losses):.4f} max={max(naive_losses):.4f}")
print(f"  first 5: {[f'{x:.4f}' for x in naive_losses[:5]]}")
print(f"  last 5: {[f'{x:.4f}' for x in naive_losses[-5:]]}")
naive_avg = sum(naive_losses) / len(naive_losses)
weighted_avg = total_sum_loss / total_n_unmasked

print(f"Naive  per-sample mean (nano F.cross_entropy mean, then avg):  {naive_avg:.6f}")
print(f"Token-weighted mean  (sum_sum / sum_unmasked_tokens):         {weighted_avg:.6f}")
print(f"Difference: {naive_avg - weighted_avg:+.6f}")
print(f"Total unmasked tokens: {total_n_unmasked}/{GBS*block} = {total_n_unmasked/GBS/block*100:.2f}%")
print(f"\nref iter 5988 logged lm_loss = 3.025030")
print(f"  Δ (naive avg - ref)    = {naive_avg - 3.025030:+.6f}")
print(f"  Δ (weighted avg - ref) = {weighted_avg - 3.025030:+.6f}")
