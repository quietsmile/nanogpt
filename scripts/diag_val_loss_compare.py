"""Compute val loss on nano retrain iter_7485 ckpt AND ref iter_7485 ckpt.
If val losses match, model quality is aligned even if per-step training
trajectory oscillates in ±0.15 nat."""
import torch, torch.nn.functional as F
import sys, numpy as np
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

def make_model(state_dict=None):
    cfg = GPTConfig(block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512,
        n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006, use_rope=True,
        rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536,
        qk_layernorm=True, tie_embeddings=False, disable_scaled_init_method=True, use_moe=True,
        moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160, moe_router_topk=8,
        moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True,
        moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160,
        moe_routing_type="greedy", eod_token_id=151643, mask_loss_id=160000,
        seq_aux_balance_alpha=0.0, use_eod_attn_mask=False)  # alpha=0 for pure LM loss eval
    m = GPT(cfg).cuda()
    if state_dict is not None:
        m.load_state_dict(state_dict, strict=False)
    m.eval()
    return m

def eval_val(model, n_samples=100):
    data = np.memmap('/root/nanogpt/data/cybertron_baseline/val.bin', dtype=np.int32, mode='r')
    block = 8192
    n_val = (len(data) - 1) // block
    if n_samples > n_val: n_samples = n_val
    EOD = 151643
    total_sum, total_n = 0.0, 0
    for i in range(n_samples):
        X = torch.from_numpy(data[i*block : i*block + block].astype(np.int64)[None]).cuda()
        Y = torch.from_numpy(data[i*block + 1 : i*block + 1 + block].astype(np.int64)[None]).cuda()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(X, targets=Y)
        ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), Y.view(-1),
                              ignore_index=-1, reduction='none')
        mask = (X.view(-1) != EOD).float()
        total_sum += (ce * mask).sum().item()
        total_n += int(mask.sum().item())
        del X, Y, logits
        torch.cuda.empty_cache()
    return total_sum / total_n

# --- nano retrain iter_7485 ---
print("Loading nano retrain iter_7485...")
ck = torch.load('/root/nanogpt/out-cybertron-moe-196-from0-fresh/ckpt.pt',
                 map_location='cuda', weights_only=False)
nano_sd = ck['model']
m_nano = make_model(nano_sd)
print(f"Evaluating nano on 100 val samples...")
nano_val = eval_val(m_nano, n_samples=100)
print(f"nano iter_7485 val loss = {nano_val:.6f}")
del m_nano
torch.cuda.empty_cache()

# --- ref iter_7485 ---
print("\nLoading ref iter_7485 (via convert)...")
meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0007485")
ref_sd = convert(meg)
m_ref = make_model(ref_sd)
print(f"Evaluating ref on 100 val samples...")
ref_val = eval_val(m_ref, n_samples=100)
print(f"ref iter_7485 val loss = {ref_val:.6f}")

print(f"\n=== VAL LOSS COMPARISON ===")
print(f"nano = {nano_val:.6f}")
print(f"ref  = {ref_val:.6f}")
print(f"Δ    = {nano_val - ref_val:+.6f} nat")
