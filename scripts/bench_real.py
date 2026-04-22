"""Bench the real 00196 config on 1 GPU. Usage: CUDA_VISIBLE_DEVICES=1 python3 scripts/bench_real.py"""
import torch, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPTConfig, GPT

print(f"device: {torch.cuda.get_device_name()}", flush=True)
cfg = GPTConfig(
    block_size=8192, vocab_size=152064,
    n_layer=9, n_head=4, n_embd=512, n_kv_head=2, kv_channels=64,
    use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5,
    use_swiglu=True, ffn_hidden_size=1536, qk_layernorm=True,
    tie_embeddings=False, init_std=0.006, disable_scaled_init_method=True,
    use_moe=True, moe_layer_freq=[0] + [1] * 8, num_experts=144,
    moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1,
    moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001,
    moe_shared_expert_hidden_size=160,
    eod_token_id=151643, mask_loss_id=160000,
    seq_aux_balance_alpha=0.0001, use_eod_attn_mask=True,
)
m = GPT(cfg).cuda().bfloat16()
print("built", flush=True)
x = torch.randint(0, 152064, (1, 8192), device="cuda")
y = torch.randint(0, 152064, (1, 8192), device="cuda")

print("fwd0 start", flush=True)
torch.cuda.synchronize(); t=time.time()
_, loss = m(x, targets=y)
torch.cuda.synchronize()
print(f"fwd0: t={time.time()-t:.2f}s loss={loss.item():.4f} mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

print("bwd0 start", flush=True); t=time.time()
loss.backward()
torch.cuda.synchronize()
print(f"bwd0: t={time.time()-t:.2f}s mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
for i in range(3):
    torch.cuda.synchronize(); t=time.time()
    opt.zero_grad(set_to_none=True)
    _, loss = m(x, targets=y)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    print(f"full step {i}: t={time.time()-t:.3f}s", flush=True)
