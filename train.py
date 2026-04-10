"""
Training script supporting both single GPU and multi-GPU DDP.

Extends the original nanoGPT train.py with:
- Bitwise deterministic training mode (--deterministic)
- WSD-exp learning rate schedule (--lr_decay_style=wsd-exp)
- Sample-based LR scheduling (matching cybertron's convention)
- Configurable Adam eps (for cybertron alignment: eps=1e-15)
- Support for megatron/cybertron-format blended datasets

Usage examples:
  # Original GPT-2 style:
  python train.py --batch_size=32 --compile=False

  # Cybertron baseline (scaling_moe_00196 dense equivalent):
  python train.py config/cybertron_baseline.py

  # DDP:
  torchrun --standalone --nproc_per_node=8 train.py config/cybertron_baseline.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12       # micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
n_kv_head = None      # None = MHA; integer = GQA kv heads
kv_channels = None    # per-head dim; None → n_embd // n_head
use_rope = False
rotary_base = 10000
use_rmsnorm = False
norm_eps = 1e-5
use_swiglu = False
ffn_hidden_size = None
tie_embeddings = True
init_std = 0.02
qk_layernorm = False  # RMSNorm on Q/K per head before RoPE (cybertron qk_layernorm)
disable_scaled_init_method = False  # skip 1/sqrt(2*n_layer) scaling for residual projs
# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-8       # cybertron uses 1e-15
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
lr_decay_style = 'cosine'  # 'cosine' | 'wsd-exp'
warmup_iters = 2000         # used for cosine schedule (iters)
lr_decay_iters = 600000     # used for cosine schedule (iters)
min_lr = 6e-5
# WSD-exp schedule (sample-based, matching cybertron convention)
# These are ignored when lr_decay_style='cosine'
warmup_samples = 0         # warmup samples (lr goes from 0 to learning_rate)
decay_end_samples = 0      # total samples at which decay ends (== total training samples)
constant_samples = 0       # samples in the constant phase (after warmup, before decay)
global_batch_size = None   # if set, used to convert sample counts to iter counts
# DDP settings
backend = 'nccl'
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
# deterministic training
deterministic = False      # set True for bitwise reproducible training
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Deterministic mode: must be set before any CUDA ops
if deterministic:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# Seeding: all ranks use same seed when deterministic=True for reproducibility
# (data batches are already assigned per-rank, no need for different seeds)
if deterministic:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed + (int(os.environ.get('RANK', 0))))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)


def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# Sequential data loader for cybertron-aligned training
# When using cybertron data (dataset ends with '_cybertron'), we read sequentially
_seq_data_pos = {'train': 0, 'val': 0}
_seq_data = {}


def get_batch_sequential(split):
    """Read data sequentially, matching cybertron's data ordering exactly."""
    global _seq_data, _seq_data_pos
    if split not in _seq_data:
        fname = 'train.bin' if split == 'train' else 'val.bin'
        _seq_data[split] = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')

    data = _seq_data[split]
    n_tokens = len(data)

    # Each sample is block_size+1 tokens (block_size input + 1 target)
    sample_stride = block_size + 1
    n_samples = (n_tokens - 1) // sample_stride

    indices = []
    for _ in range(batch_size):
        idx = _seq_data_pos[split] % n_samples
        indices.append(idx * sample_stride)
        _seq_data_pos[split] += 1

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in indices])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in indices])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


use_sequential = dataset.endswith('_cybertron') or dataset.endswith('_sequential')
_get_batch = get_batch_sequential if use_sequential else get_batch

# -----------------------------------------------------------------------------
# Model init
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout,
    n_kv_head=n_kv_head, kv_channels=kv_channels,
    use_rope=use_rope, rotary_base=rotary_base,
    use_rmsnorm=use_rmsnorm, norm_eps=norm_eps,
    use_swiglu=use_swiglu, ffn_hidden_size=ffn_hidden_size,
    tie_embeddings=tie_embeddings, init_std=init_std,
    qk_layernorm=qk_layernorm,
    disable_scaled_init_method=disable_scaled_init_method,
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
              'n_kv_head', 'kv_channels', 'use_rope', 'rotary_base', 'use_rmsnorm', 'norm_eps',
              'use_swiglu', 'ffn_hidden_size', 'tie_embeddings', 'init_std',
              'qk_layernorm', 'disable_scaled_init_method']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# GradScaler for fp16
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), adam_eps, device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# Compile
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# -----------------------------------------------------------------------------
# LR scheduling
# -----------------------------------------------------------------------------
# Determine effective_global_batch_size for sample-based scheduling
_eff_gbs = global_batch_size if global_batch_size else (batch_size * gradient_accumulation_steps * ddp_world_size)


def get_lr(it):
    """Learning rate as a function of iteration count.

    Supports two modes:
    1. 'cosine': classic cosine decay with warmup (original nanogpt behavior)
    2. 'wsd-exp': warmup-stable-decay with exponential decay (cybertron behavior)
       - Uses sample-based counting internally
    """
    if lr_decay_style == 'cosine':
        # 1) linear warmup
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        # 2) beyond decay range
        if it > lr_decay_iters:
            return min_lr
        # 3) cosine decay
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    elif lr_decay_style == 'wsd-exp':
        # Convert iteration to consumed samples
        consumed_samples = (it + 1) * _eff_gbs

        _warmup = warmup_samples
        _constant = constant_samples
        _decay_end = decay_end_samples

        # Phase 1: warmup (samples 0 to warmup_samples)
        if consumed_samples <= _warmup:
            if _warmup == 0:
                return learning_rate
            return learning_rate * consumed_samples / _warmup

        # Phase 2: constant (warmup_samples to constant_samples)
        if consumed_samples < _constant:
            return learning_rate

        # Phase 3: exponential decay (constant_samples to decay_end_samples)
        if consumed_samples >= _decay_end:
            return min_lr

        decay_range = _decay_end - _constant
        progress = (consumed_samples - _constant) / decay_range  # 0 to 1

        # Exponential decay: lr * 0.5^(-progress * log2(min_lr/lr))
        # Equivalent to: lr * (min_lr/lr)^progress
        # At progress=0: lr; at progress=1: min_lr
        min_lr_ratio = max(1e-8 / learning_rate, min_lr / learning_rate)
        ratio = 0.5 ** (-progress * math.log2(min_lr_ratio))
        return ratio * learning_rate

    else:
        raise ValueError(f"Unknown lr_decay_style: {lr_decay_style}")


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = _get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
X, Y = _get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    # Set LR for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate and checkpoint
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        consumed_samples = (iter_num + 1) * _eff_gbs
        print(f"step {iter_num} (samples {consumed_samples:,}): "
              f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6e}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "consumed_samples": consumed_samples,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # Forward / backward with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = _get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        consumed_samples = (iter_num + 1) * _eff_gbs
        print(f"iter {iter_num}: loss {lossf:.4f}, samples {consumed_samples:,}, "
              f"time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
