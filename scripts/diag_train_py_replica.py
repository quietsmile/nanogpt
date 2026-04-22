"""Replicate train.py's EXACT iter-1 execution path, dump grad_norm.

Mimics train.py end-to-end:
  - load ckpt (with optim state)
  - rebuild model via configurator path
  - DDP wrap + configure_optimizers
  - use get_batch_sequential (prefetch + step advance)
  - scaler.scale(loss).backward() with ctx context manager
  - update_expert_bias + scaler.unscale_ + clip_grad_norm_

vs diag_per_param_grads.py which is a simplified path.
"""
import os, sys, math, pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPT_DIR)

# Simulate config
batch_size = 1
block_size = 8192
ddp_rank = int(os.environ.get('RANK', 0))
ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0))
ddp_world_size = int(os.environ.get('WORLD_SIZE', 1))
gradient_accumulation_steps = 64
data_dir = os.path.join(ROOT, 'data/cybertron_baseline')

torch.cuda.set_device(ddp_local_rank)
torch.distributed.init_process_group(backend='nccl')
device = f'cuda:{ddp_local_rank}'

# MIMIC train.py exactly:
# line 137: gradient_accumulation_steps //= ddp_world_size
gradient_accumulation_steps //= ddp_world_size
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
dtype = 'bfloat16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

# data iterator (matches train.py line 186-224)
_seq_data_pos = {'train': ddp_rank * batch_size, 'val': 0}
_seq_data = {}
def get_batch_sequential(split):
    global _seq_data, _seq_data_pos
    if split not in _seq_data:
        fname = 'train.bin' if split == 'train' else 'val.bin'
        _seq_data[split] = np.memmap(os.path.join(data_dir, fname), dtype=np.int32, mode='r')
    data = _seq_data[split]
    n_tokens = len(data)
    sample_stride = block_size
    n_samples = (n_tokens - 1) // sample_stride
    step = ddp_world_size * batch_size if split == 'train' else batch_size
    indices = []
    for i in range(batch_size):
        idx = (_seq_data_pos[split] + i) % n_samples
        indices.append(idx * sample_stride)
    _seq_data_pos[split] += step
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in indices])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in indices])
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

# Build model from ckpt
from model import GPTConfig, GPT, MoERouter
ckpt = torch.load('/root/nanogpt/out-cybertron-moe-196-from0/ckpt.pt', map_location=device, weights_only=False)
cfg_dict = dict(ckpt['model_args'])
cfg_dict.update({
    'moe_routing_type': 'greedy',
    'eod_token_id': 151643,
    'mask_loss_id': 160000,
    'seq_aux_balance_alpha': 0.0001,
    'use_eod_attn_mask': False,
})
cfg = GPTConfig(**cfg_dict)
model = GPT(cfg).to(device)
model.load_state_dict(ckpt['model'], strict=False)
raw_model = model
model = DDP(model, device_ids=[ddp_local_rank])

# Optimizer (matches train.py)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=1.2e-3,
                                            betas=(0.9, 0.95), eps=1e-15, device_type='cuda')
if ckpt['optimizer']:
    optimizer.load_state_dict(ckpt['optimizer'])
    # fix device mismatch for step tensors
    for state in optimizer.state.values():
        if 'step' in state and hasattr(state['step'], 'to'):
            state['step'] = state['step'].to(device)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# One training step mimicking train.py exactly
model.train()
X, Y = get_batch_sequential('train')
acc_loss_sum = 0.0
for micro_step in range(gradient_accumulation_steps):
    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
    with ctx:
        logits, loss = model(X, Y)
        acc_loss_sum += loss.detach().float().item()
        loss = loss / gradient_accumulation_steps
    X, Y = get_batch_sequential('train')
    scaler.scale(loss).backward()

# aux-free MoE bias update
for m in raw_model.modules():
    if isinstance(m, MoERouter):
        m.update_expert_bias()

# grad norm (pre-clip)
scaler.unscale_(optimizer)
total_sq = 0.0
for name, p in raw_model.named_parameters():
    if p.grad is None: continue
    total_sq += p.grad.float().norm().item() ** 2
gn_pre = total_sq ** 0.5
if ddp_rank == 0:
    print(f'[rank0] train.py-replica: iter 1 loss {acc_loss_sum/gradient_accumulation_steps:.4f}, gn (pre-clip) = {gn_pre:.4f}')
    print(f'[rank0] expected: ref gn=2.033, diag gn=2.034')

torch.distributed.destroy_process_group()
