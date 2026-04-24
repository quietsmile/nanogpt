"""Run N continuous training steps from iter_5988 ckpt + Adam state,
compare each step's loss to ref's logged value. Reveal drift pattern."""
import torch, torch.nn.functional as F
import sys, numpy as np
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

import sys as _sys; _sys.stdout.reconfigure(line_buffering=True)
# Ref loss + LR at iters 5989..5993 (5 steps for quick trajectory)
REF = {
    5989:(1.198156e-3, 3.057512), 5990:(1.196314e-3, 3.007788),
    5991:(1.194475e-3, 2.938776), 5992:(1.192640e-3, 3.058283),
    5993:(1.190807e-3, 3.000015),
}
N_STEPS = len(REF)

# Load model + optim state
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
model.train()

optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), 1e-15, 'cuda')
# Load Adam state
nano_optim = torch.load('/newcpfs/user/yuchen/karpathy/cybertron_dump/nano_optim_iter5988.pt',
                         map_location='cuda', weights_only=False)
ref_state = nano_optim['state']
for group in optimizer.param_groups:
    for p in group['params']:
        pname = None
        for n, param in model.named_parameters():
            if param is p: pname = n; break
        if pname in ref_state:
            st = ref_state[pname]
            optimizer.state[p] = {
                'step': torch.tensor(float(st['step']), dtype=torch.float32, device='cuda'),
                'exp_avg': st['exp_avg'].to('cuda'),
                'exp_avg_sq': st['exp_avg_sq'].to('cuda'),
            }
print(f"Adam state loaded for {sum(1 for g in optimizer.param_groups for _ in g['params'])} params")

data = np.memmap('/root/nanogpt/data/cybertron_baseline/train.bin', dtype=np.int32, mode='r')
block = 8192
GBS = 64
EOD = 151643
from model import MoERouter

print(f"\n{'iter':>5s} {'LR':>10s} {'nano_loss':>10s} {'ref_loss':>10s} {'Δ':>10s} {'grad_nano':>10s}")
iters = sorted(REF.keys())
for iter_n in iters:
    lr, ref_loss = REF[iter_n]
    # Set LR
    for g in optimizer.param_groups:
        g['lr'] = lr

    # Forward + backward on 64 samples of this iter
    start = (iter_n - 1) * 64  # iter N consumes samples [(N-1)*64, N*64)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for i in range(GBS):
        X = torch.from_numpy(data[(start+i)*block : (start+i)*block + block].astype(np.int64)[None]).cuda()
        Y = torch.from_numpy(data[(start+i)*block + 1 : (start+i)*block + 1 + block].astype(np.int64)[None]).cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, mb_loss = model(X, targets=Y)
        (mb_loss / GBS).backward()
        total_loss += mb_loss.item()
        del X, Y
        torch.cuda.empty_cache()
    nano_loss = total_loss / GBS

    # Update expert_bias
    for m in model.modules():
        if isinstance(m, MoERouter):
            m.update_expert_bias()

    # Grad clip + step
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
    optimizer.step()

    print(f"{iter_n:>5d} {lr:>10.4e} {nano_loss:>10.6f} {ref_loss:>10.6f} {nano_loss-ref_loss:>+10.6f} {grad_norm:>10.4f}")
