"""Compute ref's CE on 8 dumped samples, compare to nano's CE on the same 8 samples."""
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

# Ref's output_layer dump for mbs0 and mbs1
ref_logits_mbs0 = load_out(f"{DUMP}/output_layer-iter5988-mbs0-forward-output-tp0.1-pp0.1-ep3.4.pt")  # [T, B=4, V]
ref_logits_mbs1 = load_out(f"{DUMP}/output_layer-iter5988-mbs1-forward-output-tp0.1-pp0.1-ep3.4.pt")  # [T, B=4, V]
print(f"ref mbs0 logits {ref_logits_mbs0.shape}, mbs1 logits {ref_logits_mbs1.shape}")

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

# Match indices from verify_samples.py output
# mbs0 b0..3 → nano idx 383260..383263
# mbs1 b0..3 → nano idx 383292..383295
idx_map = [
    ('mbs0 b0', 383260, ref_logits_mbs0[:, 0, :]),
    ('mbs0 b1', 383261, ref_logits_mbs0[:, 1, :]),
    ('mbs0 b2', 383262, ref_logits_mbs0[:, 2, :]),
    ('mbs0 b3', 383263, ref_logits_mbs0[:, 3, :]),
    ('mbs1 b0', 383292, ref_logits_mbs1[:, 0, :]),
    ('mbs1 b1', 383293, ref_logits_mbs1[:, 1, :]),
    ('mbs1 b2', 383294, ref_logits_mbs1[:, 2, :]),
    ('mbs1 b3', 383295, ref_logits_mbs1[:, 3, :]),
]

EOD = 151643
ref_sum_ce, ref_n = 0.0, 0
nano_sum_ce, nano_n = 0.0, 0
print(f"\n{'name':10s}  {'ref_ce':>10s}  {'nano_ce':>10s}  {'Δ':>10s}  unmasked")
for name, s_idx, ref_logits in idx_map:
    X = torch.from_numpy(data[s_idx*block : s_idx*block + block].astype(np.int64)[None]).cuda()
    Y = torch.from_numpy(data[s_idx*block + 1 : s_idx*block + 1 + block].astype(np.int64)[None]).cuda()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        nano_logits, _ = model(X, targets=Y)
    nano_logits = nano_logits[0]  # [T, V]

    # Both use same target and mask
    ce_ref = F.cross_entropy(ref_logits.float(), Y[0], reduction='none')
    ce_nano = F.cross_entropy(nano_logits.float(), Y[0], reduction='none')
    mask = X[0] != EOD  # mask on input EOD (ref-compatible)
    n = mask.sum().item()
    ref_ce_mean = (ce_ref * mask.float()).sum().item() / n
    nano_ce_mean = (ce_nano * mask.float()).sum().item() / n
    print(f"{name:10s}  {ref_ce_mean:>10.4f}  {nano_ce_mean:>10.4f}  {nano_ce_mean - ref_ce_mean:>+10.4f}  {n}")
    ref_sum_ce += (ce_ref * mask.float()).sum().item()
    ref_n += n
    nano_sum_ce += (ce_nano * mask.float()).sum().item()
    nano_n += n
    del X, Y, nano_logits
    torch.cuda.empty_cache()

print(f"\n8-sample ref mean CE:  {ref_sum_ce / ref_n:.6f}")
print(f"8-sample nano mean CE: {nano_sum_ce / nano_n:.6f}")
print(f"Δ: {(nano_sum_ce - ref_sum_ce)/nano_n:+.6f}")
