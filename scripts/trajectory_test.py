"""Trajectory alignment test: load ref iter_0 weights into nano, run 2 optim steps, compare per-step loss to ref."""
import sys, os, torch, numpy as np
sys.path.insert(0, '/root/nanogpt')
from model import GPTConfig, GPT

CKPT_ITER0 = '/root/nanogpt/reports/megatron_iter0_ckpt.pt'
DATA = '/root/nanogpt/data/cybertron_baseline/train.bin'
BLOCK = 8192

# Ref trajectory from TB
REF = {1: 11.942984, 2: 11.942345, 3: 11.944288}

cfg = GPTConfig(
    block_size=BLOCK, vocab_size=152064, n_layer=9, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64,
    use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536,
    qk_layernorm=True, tie_embeddings=False, init_std=0.006, disable_scaled_init_method=True,
    use_moe=True, moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160,
    moe_router_topk=8, moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True,
    moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160,
    moe_routing_type='greedy',
    eod_token_id=151643, mask_loss_id=160000, seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False,
)
m = GPT(cfg).cuda().bfloat16()
m.load_state_dict(torch.load(CKPT_ITER0, map_location='cpu', weights_only=False), strict=False)

# Match ref optimizer: adam (β1=0.9, β2=0.95, eps=1e-15), wd=0.1, grad_clip=1.0
# ref yaml: accumulate_allreduce_grads_in_fp32=true — nano uses AdamW default (bf16 grad accum).
opt = torch.optim.AdamW(m.parameters(), lr=2.4e-6, betas=(0.9, 0.95), eps=1e-15, weight_decay=0.1, fused=True)

arr = np.memmap(DATA, dtype=np.int32, mode='r')
def get_batch(step_idx):
    """ref iter N sees samples [(N-1)*64, N*64). Return batched over 64 samples as one big batch."""
    idx_list, tgt_list = [], []
    start = (step_idx - 1) * 64
    for s in range(64):
        off = (start + s) * 8192
        tok = np.array(arr[off:off + 8193].astype(np.int64))
        idx_list.append(torch.from_numpy(tok[:8192]))
        tgt_list.append(torch.from_numpy(tok[1:8193]))
    return torch.stack(idx_list, dim=0), torch.stack(tgt_list, dim=0)

# Ref LR schedule: warmup 32000 samples = 500 iters, linear 0 → peak lr 1.2e-3
def lr_at(iter_num):
    warmup_iters = 500
    peak = 1.2e-3
    return iter_num / warmup_iters * peak

print(f"iter | nano_loss | ref_loss | diff | lr")
for step in range(1, 4):
    # Set lr for this step (ref logs lr AT iter N, so iter 1 uses lr=2.4e-6 for its step)
    cur_lr = lr_at(step)
    for g in opt.param_groups: g['lr'] = cur_lr

    x, y = get_batch(step)
    x, y = x.cuda(), y.cuda()
    # Process in chunks of 8 to fit memory (accumulate gradients)
    opt.zero_grad(set_to_none=True)
    total_loss = 0.0
    for c0 in range(0, 64, 8):
        _, loss = m(x[c0:c0+8], targets=y[c0:c0+8])
        loss_scaled = loss / 8  # 8 chunks
        loss_scaled.backward()
        total_loss += loss.item()
    avg_loss = total_loss / 8

    # Grad clip
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()

    ref_loss = REF.get(step, float('nan'))
    diff = avg_loss - ref_loss
    print(f"{step:4d} | {avg_loss:9.4f} | {ref_loss:8.4f} | {diff:+.4f} | {cur_lr:.3e}")
