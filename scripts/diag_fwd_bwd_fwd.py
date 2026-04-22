"""Isolate backward+optim by running forward -> optim step -> forward.

Setup:
  - Load iter_5988 ckpt (matches weights ref iter 5989 forward used)
  - Forward iter 5989's 64 samples, verify loss ≈ ref 3.057512 (expected ULP)
  - Backward + AdamW step with ref's hypers (lr=1.198156e-3)
  - Forward iter 5990's 64 samples, check loss vs ref 3.007788
    If matches within ULP → backward+optim aligned
    If systematic Δ → bug in backward/optim
"""
import torch, torch.nn.functional as F
import sys, numpy as np, math
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
    seq_aux_balance_alpha=0.0, use_eod_attn_mask=False)  # disable aux to isolate main-loss grad
model = GPT(cfg).cuda()
model.load_state_dict(sd, strict=False)
model.train()

# Match ref hyperparameters at iter 5989 (from train log + args):
#   lr = 1.198156e-3 (logged)
#   betas = (0.9, 0.95), eps = 1e-15
#   weight_decay = 0.1
#   grad_clip = 1.0
lr_5989 = 1.198156e-3
optimizer = model.configure_optimizers(
    weight_decay=0.1, learning_rate=lr_5989,
    betas=(0.9, 0.95), eps=1e-15, device_type='cuda')

data = np.memmap('/root/nanogpt/data/cybertron_baseline/train.bin', dtype=np.int32, mode='r')
block = 8192
GBS = 64
MBS = 1  # memory-limited: lm_head needs [mb, T, V]=18GB fp32 for mb=4
NUM_MB = GBS // MBS  # 64 microbatches
EOD = 151643

def gather_batch(start_offset):
    return [(data[(start_offset+i)*block : (start_offset+i)*block + block].astype(np.int64),
             data[(start_offset+i)*block + 1 : (start_offset+i)*block + 1 + block].astype(np.int64))
            for i in range(GBS)]

# --- STEP 1: forward iter 5989 data, record loss ---
print("=" * 60)
print("STEP 1: Forward iter 5989 (iter_5988 ckpt)")
print("=" * 60)
batch_5989 = gather_batch(5988 * 64)  # offsets 383232..383295

# Simulate ref-style: MBS=4 samples per microbatch, NUM_MB=16 microbatches
total_loss = 0.0
optimizer.zero_grad(set_to_none=True)
for mb_idx in range(NUM_MB):
    # Gather 4 samples for this microbatch
    x_list = [batch_5989[mb_idx*MBS + b][0] for b in range(MBS)]
    y_list = [batch_5989[mb_idx*MBS + b][1] for b in range(MBS)]
    X = torch.from_numpy(np.stack(x_list)).cuda()
    Y = torch.from_numpy(np.stack(y_list)).cuda()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, mb_loss = model(X, targets=Y)
    # Megatron: loss passed to backward = mb_loss / num_microbatches
    (mb_loss / NUM_MB).backward()
    total_loss += mb_loss.item()
    del X, Y
    torch.cuda.empty_cache()

iter5989_loss = total_loss / NUM_MB  # avg across microbatches (Megatron-style)
print(f"iter 5989 nano loss (avg per sample) = {iter5989_loss:.6f}")
print(f"iter 5989 ref  logged lm_loss       = 3.057512")
print(f"Δ = {iter5989_loss - 3.057512:+.6f}")

# --- STEP 2: grad clip + optim step ---
# Clip grad to max_norm=1.0 (matches Megatron's grad_clip)
# Compute grad_norm WITH and WITHOUT embedding/lm_head (which Megatron marks .shared=True and excludes)
import math as _math
def gn(params):
    return _math.sqrt(sum((p.grad.detach().float().pow(2).sum().item()) for p in params if p.grad is not None))

all_params = list(model.parameters())
wte_params = list(model.transformer.wte.parameters())
lm_head_params = list(model.lm_head.parameters())
non_wte_lm = [p for p in all_params if all(p is not e for e in wte_params + lm_head_params)]

gn_all = gn(all_params)
gn_wte = gn(wte_params)
gn_lm = gn(lm_head_params)
gn_no_wte = gn([p for p in all_params if all(p is not e for e in wte_params)])
gn_no_lm = gn([p for p in all_params if all(p is not e for e in lm_head_params)])
gn_no_both = gn(non_wte_lm)

print(f"\nSTEP 2 grad_norm breakdown (ref logged = 0.130):")
print(f"  ALL params       = {gn_all:.6f}")
print(f"  wte only         = {gn_wte:.6f}")
print(f"  lm_head only     = {gn_lm:.6f}")
print(f"  exclude wte      = {gn_no_wte:.6f}")
print(f"  exclude lm_head  = {gn_no_lm:.6f}")
print(f"  exclude both     = {gn_no_both:.6f}")

# Per-block grad norms
print(f"\nPer-block (transformer.h.N):")
for i in range(len(model.transformer.h)):
    bp = list(model.transformer.h[i].parameters())
    print(f"  block {i}: {gn(bp):.6f}")

# Apply standard clipping over all params (matches nano training)
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"  clipped to 1.0; computed (all): {grad_norm:.6f}")

# MoE expert_bias update
from model import MoERouter
for _m in model.modules():
    if isinstance(_m, MoERouter):
        _m.update_expert_bias()

optimizer.step()
optimizer.zero_grad(set_to_none=True)
print("optimizer.step() applied")

# --- STEP 3: forward iter 5990 data ---
model.eval()
print("\n" + "=" * 60)
print("STEP 3: Forward iter 5990 (post-optim-step weights)")
print("=" * 60)
batch_5990 = gather_batch(5989 * 64)  # offsets 383296..383359
total_loss_5990 = 0.0
for mb_idx in range(NUM_MB):
    x_list = [batch_5990[mb_idx*MBS + b][0] for b in range(MBS)]
    y_list = [batch_5990[mb_idx*MBS + b][1] for b in range(MBS)]
    X = torch.from_numpy(np.stack(x_list)).cuda()
    Y = torch.from_numpy(np.stack(y_list)).cuda()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, mb_loss = model(X, targets=Y)
    total_loss_5990 += mb_loss.item()
    del X, Y
    torch.cuda.empty_cache()

iter5990_loss = total_loss_5990 / NUM_MB
print(f"iter 5990 nano loss = {iter5990_loss:.6f}")
print(f"iter 5990 ref  logged = 3.007788")
print(f"Δ = {iter5990_loss - 3.007788:+.6f}")

print("\n=== Summary ===")
print(f"iter 5989 Δ = {iter5989_loss - 3.057512:+.6f} (measures forward alignment — should be ULP)")
print(f"iter 5990 Δ = {iter5990_loss - 3.007788:+.6f} (measures forward + 1 optim step — gap here = backward/optim drift)")
