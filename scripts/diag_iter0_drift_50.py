"""50-step drift from iter_0 — extend iter0_drift.py to iter 50.
Tests if current nano (with all fixes) stays within 0.05 nat of ref over
50 iterations. Old retrain showed Δ=-0.14 at iter 50 pre-fixes."""
import torch, torch.nn.functional as F
import sys, numpy as np
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/root/nanogpt"); sys.path.insert(0, "/root/nanogpt/scripts")
from megatron_to_nano import load_all_megatron_shards, convert
from model import GPTConfig, GPT

REF_DATA = """
1:(2.400000e-06, 11.942980); 2:(4.800000e-06, 11.942340); 3:(7.200000e-06, 11.944290)
4:(9.600000e-06, 11.939070); 5:(1.200000e-05, 11.926620); 6:(1.440000e-05, 11.907280)
7:(1.680000e-05, 11.882210); 8:(1.920000e-05, 11.862550); 9:(2.160000e-05, 11.850610)
10:(2.400000e-05, 11.824230); 11:(2.640000e-05, 11.797130); 12:(2.880000e-05, 11.773700)
13:(3.120000e-05, 11.757690); 14:(3.360000e-05, 11.739360); 15:(3.600000e-05, 11.709860)
16:(3.840000e-05, 11.688530); 17:(4.080000e-05, 11.671060); 18:(4.320000e-05, 11.655720)
19:(4.560000e-05, 11.620240); 20:(4.800000e-05, 11.596600); 21:(5.040000e-05, 11.570150)
22:(5.280000e-05, 11.551390); 23:(5.520000e-05, 11.526250); 24:(5.760000e-05, 11.486590)
25:(6.000000e-05, 11.450220); 26:(6.240000e-05, 11.423200); 27:(6.480000e-05, 11.379510)
28:(6.720000e-05, 11.330630); 29:(6.960000e-05, 11.309140); 30:(7.200000e-05, 11.258410)
31:(7.440000e-05, 11.218500); 32:(7.680000e-05, 11.194470); 33:(7.920000e-05, 11.138960)
34:(8.160000e-05, 11.090280); 35:(8.400000e-05, 11.037210); 36:(8.640000e-05, 10.998710)
37:(8.880000e-05, 10.948130); 38:(9.120000e-05, 10.890210); 39:(9.360000e-05, 10.831430)
40:(9.600000e-05, 10.786210); 41:(9.840000e-05, 10.746440); 42:(1.008000e-04, 10.669940)
43:(1.032000e-04, 10.651740); 44:(1.056000e-04, 10.589740); 45:(1.080000e-04, 10.511550)
46:(1.104000e-04, 10.457160); 47:(1.128000e-04, 10.398630); 48:(1.152000e-04, 10.333830)
49:(1.176000e-04, 10.309890); 50:(1.200000e-04, 10.255080)
"""
REF = {}
for ln in REF_DATA.strip().split('\n'):
    for entry in ln.split(';'):
        e = entry.strip()
        if not e: continue
        k, v = e.split(':', 1)
        lr, loss = eval(v)
        REF[int(k)] = (lr, loss)

meg = load_all_megatron_shards("/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196_init/iter_0000000")
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
nano_optim = torch.load('/newcpfs/user/yuchen/karpathy/cybertron_dump/nano_optim_iter0.pt',
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
print(f"iter_0 Adam state loaded (step={nano_optim['step']})")

data = np.memmap('/root/nanogpt/data/cybertron_baseline/train.bin', dtype=np.int32, mode='r')
block = 8192
GBS = 64
EOD = 151643
from model import MoERouter

print(f"{'iter':>5s} {'LR':>10s} {'nano':>10s} {'ref':>10s} {'Δ':>10s} {'grad_nano':>10s}")
for iter_n in sorted(REF.keys()):
    lr, ref_loss = REF[iter_n]
    for g in optimizer.param_groups: g['lr'] = lr
    start = (iter_n - 1) * 64
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
    for m in model.modules():
        if isinstance(m, MoERouter): m.update_expert_bias()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
    optimizer.step()
    print(f"{iter_n:>5d} {lr:>10.4e} {nano_loss:>10.6f} {ref_loss:>10.6f} {nano_loss-ref_loss:>+10.6f} {grad_norm:>10.4f}")
