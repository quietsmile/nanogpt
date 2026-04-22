"""One-step trajectory test.

- Load Megatron iter_0 weights into nano
- Run 1 optim step on iter-1 batch (samples 0..63) using matching hparams
- Compare resulting nano loss on iter-2 batch vs ref iter 2 loss (11.9423)

Memory-aggressive: micro_batch=2, empty_cache between micro-batches.
"""
import sys, os, gc, torch, numpy as np
sys.path.insert(0, '/root/nanogpt')
from model import GPTConfig, GPT

REF_LOSS = {1: 11.942984, 2: 11.942345, 3: 11.944288}

cfg = GPTConfig(
    block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64,
    use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5, use_swiglu=True, ffn_hidden_size=1536,
    qk_layernorm=True, tie_embeddings=False, init_std=0.006, disable_scaled_init_method=True,
    use_moe=True, moe_layer_freq=[0]+[1]*8, num_experts=144, moe_ffn_hidden_size=160,
    moe_router_topk=8, moe_n_group=8, moe_topk_group=1, moe_norm_topk_prob=True,
    moe_router_score_correction_coeff=0.001, moe_shared_expert_hidden_size=160,
    moe_routing_type='greedy',
    eod_token_id=151643, mask_loss_id=160000, seq_aux_balance_alpha=0.0001, use_eod_attn_mask=False,
)
m = GPT(cfg).cuda().bfloat16()
m.load_state_dict(torch.load('/root/nanogpt/reports/megatron_iter0_ckpt.pt', map_location='cpu', weights_only=False), strict=False)

# Reference optimizer: Adam(β=0.9,0.95, eps=1e-15), wd=0.1, grad_clip=1.0
# LR at iter 1 = 1/500 × 1.2e-3 = 2.4e-6 (ref TB confirms)
lr_iter1 = 2.4e-6
opt = torch.optim.AdamW(m.parameters(), lr=lr_iter1, betas=(0.9, 0.95), eps=1e-15, weight_decay=0.1, fused=True)

arr = np.memmap('/root/nanogpt/data/cybertron_baseline/train.bin', dtype=np.int32, mode='r')
def load_batch(step):
    start = (step - 1) * 64
    return [(np.array(arr[(start+s)*8192 : (start+s)*8192+8193].astype(np.int64))) for s in range(64)]

# --- Forward-only iter-1 loss (from iter_0 weights) ---
samples_iter1 = load_batch(1)
m.eval()
with torch.no_grad():
    losses = []
    for c0 in range(0, 64, 2):
        idx = torch.stack([torch.from_numpy(samples_iter1[c0+i][:8192]) for i in range(2)], 0).cuda()
        tgt = torch.stack([torch.from_numpy(samples_iter1[c0+i][1:8193]) for i in range(2)], 0).cuda()
        _, loss = m(idx, targets=tgt)
        losses.append(loss.item())
    nano_iter1 = sum(losses)/len(losses)
print(f'Forward iter_0 on iter-1 batch: nano={nano_iter1:.6f} ref={REF_LOSS[1]:.6f} Δ={nano_iter1-REF_LOSS[1]:+.6f}', flush=True)

# --- Backward + 1 optim step (gradient from iter-1 batch) ---
m.train()
opt.zero_grad(set_to_none=True)
acc_steps = 32  # 32 accums × micro=2 → effective 64 samples
total_loss = 0.0
for c0 in range(0, 64, 2):
    idx = torch.stack([torch.from_numpy(samples_iter1[c0+i][:8192]) for i in range(2)], 0).cuda()
    tgt = torch.stack([torch.from_numpy(samples_iter1[c0+i][1:8193]) for i in range(2)], 0).cuda()
    _, loss = m(idx, targets=tgt)
    total_loss += loss.item()
    (loss / acc_steps).backward()
    del idx, tgt, loss
    torch.cuda.empty_cache()
bwd_avg_loss = total_loss / acc_steps
print(f'Backward complete, avg training loss: {bwd_avg_loss:.6f}', flush=True)

# Grad clip + step
gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
print(f'grad_norm (after clip cap 1.0): {gnorm.item():.4f}', flush=True)
opt.step()
opt.zero_grad(set_to_none=True)

# --- Forward iter-2 batch with nano's post-step weights ---
del samples_iter1; gc.collect(); torch.cuda.empty_cache()
samples_iter2 = load_batch(2)
m.eval()
with torch.no_grad():
    losses = []
    for c0 in range(0, 64, 2):
        idx = torch.stack([torch.from_numpy(samples_iter2[c0+i][:8192]) for i in range(2)], 0).cuda()
        tgt = torch.stack([torch.from_numpy(samples_iter2[c0+i][1:8193]) for i in range(2)], 0).cuda()
        _, loss = m(idx, targets=tgt)
        losses.append(loss.item())
    nano_iter2 = sum(losses)/len(losses)
print(f'\nAfter 1 optim step, forward on iter-2 batch:')
print(f'  nano loss: {nano_iter2:.6f}')
print(f'  ref iter 2: {REF_LOSS[2]:.6f}')
print(f'  Δ: {nano_iter2-REF_LOSS[2]:+.6f}')
