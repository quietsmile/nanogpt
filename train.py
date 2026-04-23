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
from monitor import create_monitor

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
# MoE (DeepSeek aux-free grouped routing)
use_moe = False
moe_layer_freq = None   # list e.g. [0]+[1]*8; None = all dense. NOTE: list type is handled below
num_experts = 64
moe_ffn_hidden_size = 128
moe_router_topk = 2
moe_n_group = 1
moe_topk_group = 1
moe_norm_topk_prob = True
moe_router_score_correction_coeff = 0.001
moe_shared_expert_hidden_size = None
# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
adam_eps = 1e-8       # cybertron uses 1e-15
grad_clip = 1.0
# Muon (NorMuon + Polar Express). When use_muon=True, attention/MLP 2D weights
# and MoE 3D expert weights are routed to Muon; embeddings/router/norms stay on AdamW.
use_muon = False
muon_lr = None        # if None, defaults to learning_rate * 33 (typical Muon scale)
muon_momentum = 0.95
muon_beta2 = 0.95
muon_weight_decay = None  # if None, uses weight_decay
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
               if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None), list))]
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
    ddp_rank = 0
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

# TF32 breaks bitwise reproducibility across runs (kernel selection varies by
# shape/load). Disable under deterministic mode for strict DDP resume equality.
torch.backends.cuda.matmul.allow_tf32 = not deterministic
torch.backends.cudnn.allow_tf32 = not deterministic
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)


def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.int32, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.int32, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# Sequential data loader for cybertron-aligned training
# When using cybertron data (dataset ends with '_cybertron'), we read sequentially.
# DDP: rank r reads samples r, r+W, r+2W, ... (interleaved sharding, W = world_size).
# Val: all ranks read from position 0 (same data; losses are all-reduced later).
_seq_data_pos = {'train': ddp_rank * batch_size, 'val': 0}
_seq_data = {}


def get_batch_sequential(split):
    """Read data sequentially, matching cybertron's data ordering exactly."""
    global _seq_data, _seq_data_pos
    if split not in _seq_data:
        fname = 'train.bin' if split == 'train' else 'val.bin'
        _seq_data[split] = np.memmap(os.path.join(data_dir, fname), dtype=np.int32, mode='r')

    data = _seq_data[split]
    n_tokens = len(data)

    # prepare_cybertron_data.py stores each cybertron blended sample as exactly
    # block_size tokens (SEQ_LENGTH=8192). Input is sample[:], target = sample[1:]+next_sample[0].
    # The last target token (from the next sample) is negligible noise (1/8192 of loss budget).
    sample_stride = block_size
    n_samples = (n_tokens - 1) // sample_stride

    # Rank offset is applied at initialization (line 189: _seq_data_pos = {'train': ddp_rank * batch_size, ...}).
    # Each call advances by world_size*batch_size so ranks stay interleaved: rank r reads r, r+W, r+2W, ...
    # For val: all ranks read the same data (averaged in estimate_loss).
    step = ddp_world_size * batch_size if (ddp and split == 'train') else batch_size

    indices = []
    for i in range(batch_size):
        idx = (_seq_data_pos[split] + i) % n_samples
        indices.append(idx * sample_stride)
    _seq_data_pos[split] += step

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in indices])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in indices])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


use_sequential = (dataset.endswith('_cybertron') or dataset.endswith('_sequential')
                  or 'cybertron' in dataset)
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
    # MoE
    use_moe=use_moe,
    moe_layer_freq=moe_layer_freq,
    num_experts=num_experts,
    moe_ffn_hidden_size=moe_ffn_hidden_size,
    moe_router_topk=moe_router_topk,
    moe_n_group=moe_n_group,
    moe_topk_group=moe_topk_group,
    moe_norm_topk_prob=moe_norm_topk_prob,
    moe_router_score_correction_coeff=moe_router_score_correction_coeff,
    moe_shared_expert_hidden_size=moe_shared_expert_hidden_size,
    # Alignment-critical behavior knobs — must flow from config to the model so
    # resume+override works. If any of these are missing on the config namespace,
    # GPTConfig defaults apply.
    moe_routing_type=globals().get('moe_routing_type', 'group_limited_greedy'),
    eod_token_id=globals().get('eod_token_id', None),
    mask_loss_id=globals().get('mask_loss_id', None),
    seq_aux_balance_alpha=globals().get('seq_aux_balance_alpha', 0.0),
    use_eod_attn_mask=globals().get('use_eod_attn_mask', False),
)

_resume_batch = None  # set below when resuming from checkpoint
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
    # Architectural keys that MUST match the ckpt (shape-affecting). These get
    # overridden from the ckpt to guarantee state_dict compatibility.
    _arch_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
                  'n_kv_head', 'kv_channels', 'use_rope', 'rotary_base', 'use_rmsnorm',
                  'norm_eps', 'use_swiglu', 'ffn_hidden_size', 'tie_embeddings',
                  'init_std', 'qk_layernorm', 'disable_scaled_init_method',
                  'use_moe', 'moe_layer_freq', 'num_experts', 'moe_ffn_hidden_size',
                  'moe_router_topk', 'moe_n_group', 'moe_topk_group',
                  'moe_norm_topk_prob', 'moe_router_score_correction_coeff',
                  'moe_shared_expert_hidden_size']
    for k in _arch_keys:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # Alignment-critical non-arch keys (routing algorithm, attn masks, loss masks,
    # aux-loss coeffs). These MUST come from the CURRENT config file, not the ckpt
    # — otherwise enabling a code-gap fix in the config has no effect on resumed
    # training. Silently falling back to defaults is exactly what caused the v1/v2/v3
    # runs to quietly use group_limited_greedy routing + no EOD mask.
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    # strict=False so old ckpts (pre aux-free-bias fix) without the new
    # `local_tokens_per_expert` buffer still load; the buffer is initialized
    # to zeros in the MoERouter constructor, which is the correct start state.
    model.load_state_dict(state_dict, strict=False)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    # Restore pre-eval RNG + data state for bitwise-deterministic resume.
    # For DDP runs, ckpt may hold per-rank state under '_per_rank'; each rank
    # picks its own slot. Otherwise fall back to the single-rank state.
    # Stash RNG + data position state; apply AFTER DDP wrap so both scratch
    # and resume paths have identical pre-loop RNG (DDP __init__ may consume RNG).
    _pr = checkpoint.get('_per_rank')
    _per_rank_batch = None
    _pending_rng_cpu = None
    _pending_rng_cuda = None
    if _pr is not None and ddp_rank < len(_pr):
        _mine = _pr[ddp_rank]
        _pending_rng_cpu = _mine['rng_state_cpu'].cpu()
        _pending_rng_cuda = [s.cpu() for s in _mine['rng_state_cuda']]
        _seq_data_pos.update(_mine['seq_data_pos'])
        if 'X' in _mine:
            _per_rank_batch = (_mine['X'].to(device), _mine['Y'].to(device))
    else:
        if 'rng_state_cpu' in checkpoint:
            _pending_rng_cpu = checkpoint['rng_state_cpu'].cpu()
        if 'rng_state_cuda' in checkpoint:
            _pending_rng_cuda = [s.cpu() for s in checkpoint['rng_state_cuda']]
        if 'seq_data_pos' in checkpoint:
            _seq_data_pos.update(checkpoint['seq_data_pos'])
    # Restore the batch that was current when checkpoint was saved.
    # DDP: prefer per-rank X/Y from _per_rank slot; otherwise fall back to master's X/Y.
    if _per_rank_batch is not None:
        _resume_batch = _per_rank_batch
    else:
        _resume_batch = (checkpoint['X'].to(device), checkpoint['Y'].to(device)) \
            if 'X' in checkpoint else None
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
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), adam_eps, device_type,
    use_muon=use_muon, muon_lr=muon_lr, muon_momentum=muon_momentum,
    muon_beta2=muon_beta2, muon_weight_decay=muon_weight_decay,
)
# Tag each param_group with lr_mult so the LR schedule can scale heterogeneous groups
# by a single base. AdamW groups default to 1.0; Muon group multiplies by its base/AdamW
# base ratio so the WSD-exp / cosine schedule shape is shared across both.
for pg in optimizer.param_groups:
    if 'lr_mult' not in pg:
        pg['lr_mult'] = pg['lr'] / learning_rate if learning_rate > 0 else 1.0
if init_from == 'resume':
    _opt_state = checkpoint.get('optimizer')
    _has_state = bool(_opt_state) and ('param_groups' in _opt_state or _opt_state.get('_multi'))
    if _has_state:
        optimizer.load_state_dict(_opt_state)
    else:
        if master_process:
            print('resume: optimizer state empty/absent; using fresh state')
checkpoint = None

# Compile
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    # find_unused_parameters=False: all experts get touched through grouped_mm ops
    # (weights multiplied by 0 routing prob still flow through graph). Setting to True
    # hurts MFU ~40% with no benefit on this MoE impl.
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

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
        mean_loss = losses.mean()
        # All-reduce across DDP ranks so every rank reports the global average
        if ddp:
            mean_loss_t = torch.tensor(mean_loss, device=device)
            torch.distributed.all_reduce(mean_loss_t, op=torch.distributed.ReduceOp.AVG)
            mean_loss = mean_loss_t.item()
        out[split] = mean_loss
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
# On resume: restore RNG state AFTER DDP wrap (DDP __init__ may consume RNG in
# scratch path, so restoring before wrap leaves resume with stale state). Doing
# it here puts both paths on equal footing right before the training loop.
if init_from == 'resume':
    if _pending_rng_cpu is not None:
        torch.set_rng_state(_pending_rng_cpu)
    if _pending_rng_cuda is not None:
        torch.cuda.set_rng_state_all(_pending_rng_cuda)
# On resume, use the saved batch so training step iter_num uses identical data
if init_from == 'resume' and _resume_batch is not None:
    X, Y = _resume_batch
else:
    X, Y = _get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
monitor = create_monitor(raw_model, optimizer, out_dir=out_dir,
                         master=master_process, ddp=ddp)

while True:
    # Set LR for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group.get('lr_mult', 1.0)

    # Evaluate and checkpoint
    # All DDP ranks must participate in estimate_loss() since it contains dist.all_reduce.
    if iter_num % eval_interval == 0:
        # Snapshot state BEFORE estimate_loss() on ALL ranks (eval advances _seq_data_pos)
        pre_eval_seq_pos = dict(_seq_data_pos)
        pre_eval_rng_cpu = torch.get_rng_state()
        pre_eval_rng_cuda = torch.cuda.get_rng_state_all()
        # DDP: each rank has its own _seq_data_pos + CUDA RNG state. Collect them
        # all on rank 0 so the single ckpt can restore each rank's slot.
        pre_eval_per_rank = None
        if ddp:
            _local_snap = {
                'rng_state_cpu': pre_eval_rng_cpu,
                'rng_state_cuda': pre_eval_rng_cuda,
                'seq_data_pos': pre_eval_seq_pos,
                'X': X.detach().cpu(),
                'Y': Y.detach().cpu(),
            }
            _gather = [None] * ddp_world_size
            import torch.distributed as _dist
            _dist.all_gather_object(_gather, _local_snap)
            if master_process:
                pre_eval_per_rank = _gather
        if master_process:
            pre_eval_X = X.clone()
            pre_eval_Y = Y.clone()

        losses = estimate_loss()

        # Restore data position and RNG on ALL ranks so eval batches don't skip training data
        _seq_data_pos.update(pre_eval_seq_pos)
        torch.set_rng_state(pre_eval_rng_cpu)
        torch.cuda.set_rng_state_all(pre_eval_rng_cuda)

        if master_process:
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
                        # Pre-eval RNG + data state for bitwise-deterministic resume:
                        # On resume, eval re-runs from this exact state, then training
                        # step iter_num uses X/Y (= the batch pre-fetched after step iter_num-1).
                        'rng_state_cpu': pre_eval_rng_cpu,
                        'rng_state_cuda': pre_eval_rng_cuda,
                        'seq_data_pos': pre_eval_seq_pos,
                        '_per_rank': pre_eval_per_rank,
                        'X': pre_eval_X.cpu(),
                        'Y': pre_eval_Y.cpu(),
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # Forward / backward with gradient accumulation. We log LM cross-entropy
    # only — not the (CE + alpha*aux) combined value the model returns — so
    # the logged value matches Megatron's TB `lm loss` scalar. aux still flows
    # through backward via the combined loss, so the training signal is
    # unchanged; only the reporting scope is different.
    _acc_loss_sum = 0.0      # local sum of per-mb LM CE (for reporting)
    _acc_aux_sum = 0.0       # local sum of per-mb alpha*aux_contrib (diagnostic)
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            # Prefer the exposed lm/aux split (set by GPT.forward on the raw
            # model). Fall back to the combined loss for older checkpoints /
            # non-aux configs. raw_model handles the DDP unwrap.
            _lm = getattr(raw_model, 'last_lm_loss', None)
            _aux = getattr(raw_model, 'last_aux_contrib', None)
            if _lm is not None and _aux is not None:
                _acc_loss_sum += _lm.float().item()
                _acc_aux_sum += _aux.float().item()
            else:
                _acc_loss_sum += loss.detach().float().item()
            loss = loss / gradient_accumulation_steps
        X, Y = _get_batch('train')
        scaler.scale(loss).backward()

    # Capture per-iter MoE routing stats BEFORE update_expert_bias resets the
    # counter. Aggregates across layers: per-expert token counts averaged over
    # microbatches, then take global max/min/mean + maxVio = (max - mean)/mean.
    # Matches ref master log columns `tokens_per_expert/{max,min,mean}` and
    # `maxVio/{micro_batch,global_batch}`.
    from model import MoERouter
    _moe_counts = []  # one tensor [E] per MoE layer (fp32, local-rank counts)
    for _m in raw_model.modules():
        if isinstance(_m, MoERouter):
            _moe_counts.append(_m.local_tokens_per_expert.detach().clone().float())
    if _moe_counts:
        import torch.distributed as _dist
        _stacked = torch.stack(_moe_counts, dim=0)  # [L_moe, E]
        if ddp:
            _dist.all_reduce(_stacked, op=_dist.ReduceOp.SUM)
        # Per-microbatch scale: divide by gradient_accumulation_steps for a
        # per-mb average (ref's tokens_per_expert/mean is per-mb).
        _per_mb = _stacked / max(gradient_accumulation_steps, 1)
        _moe_max = float(_per_mb.max().item())
        _moe_min = float(_per_mb.min().item())
        _moe_mean = float(_per_mb.mean().item())
        _moe_maxvio = (_moe_max - _moe_mean) / _moe_mean if _moe_mean > 0 else 0.0
    else:
        _moe_max = _moe_min = _moe_mean = _moe_maxvio = None

    # Aux-free MoE bias: apply one sign() update per optim step using the token
    # counts accumulated over the grad_accum micro-steps (matches Megatron's
    # _update_router_expert_bias in finalize_model_grads.py).
    for _m in raw_model.modules():
        if isinstance(_m, MoERouter):
            _m.update_expert_bias()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        _grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else:
        _grad_norm = torch.tensor(0.0)

    # All-reduce loss BEFORE optimizer.step so monitor.step() (which consumes
    # grads + global_mean_loss) runs while grads are still present. Moving the
    # collective up by two lines changes no training state — it just brings the
    # logging value forward a hair. The collective itself is unchanged.
    _local_mean = _acc_loss_sum / gradient_accumulation_steps
    if ddp:
        _t = torch.tensor(_local_mean, device=device)
        torch.distributed.all_reduce(_t, op=torch.distributed.ReduceOp.AVG)
        _global_mean_loss = _t.item()
    else:
        _global_mean_loss = _local_mean

    scaler.step(optimizer)
    scaler.update()
    monitor.step(iter_num, loss=_global_mean_loss, grad_norm=_grad_norm, lr=lr,
                 samples=(iter_num + 1) * _eff_gbs)
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = _global_mean_loss
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        consumed_samples = (iter_num + 1) * _eff_gbs
        _gn = float(_grad_norm.item()) if hasattr(_grad_norm, 'item') else float(_grad_norm)
        print(f"iter {iter_num}: loss {lossf:.4f}, samples {consumed_samples:,}, "
              f"time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, gn {_gn:.4f}")
        # Structured JSON log for dashboard overlay (master only)
        if master_process:
            import json as _json
            _log_path = os.path.join(out_dir, 'train_log.jsonl')
            try:
                _entry = {
                    'iter': iter_num, 'loss': float(lossf),
                    'samples': int(consumed_samples),
                    'dt_ms': float(dt*1000), 'mfu': float(running_mfu),
                    'lr': float(lr), 'grad_norm': _gn,
                }
                # MoE routing stats captured pre-update_expert_bias (see above).
                # Comparable to ref master log: tokens_per_expert/{max,min,mean}
                # and maxVio/micro_batch.
                if _moe_max is not None:
                    _entry['tokens_per_expert_max'] = _moe_max
                    _entry['tokens_per_expert_min'] = _moe_min
                    _entry['tokens_per_expert_mean'] = _moe_mean
                    _entry['maxvio_micro_batch'] = _moe_maxvio
                # Aux contribution (diagnostic — not added to 'loss' above).
                if gradient_accumulation_steps > 0 and _acc_aux_sum != 0:
                    _entry['aux_contrib_mean'] = (
                        _acc_aux_sum / gradient_accumulation_steps)
                with open(_log_path, 'a') as _f:
                    _f.write(_json.dumps(_entry) + '\n')
            except Exception:
                pass
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

monitor.close()
if ddp:
    destroy_process_group()
