"""Build a nano-format train-resumable ckpt from a Megatron iter_N directory.

Combines:
  - Weight conversion (scripts/megatron_to_nano.py)
  - Optim state conversion (scripts/recover_megatron_optim.py + optim_megatron_to_nano.py)
  - Properly formatted {'model', 'model_args', 'optimizer', 'iter_num', ...} dict

Output is a ckpt.pt that train.py's init_from='resume' can load directly, producing
training that continues from iter N with full Adam history (step, exp_avg, exp_avg_sq).
"""
from __future__ import annotations
import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)        # for `from model import ...`
sys.path.insert(0, SCRIPT_DIR)  # for sibling script imports
from megatron_to_nano import load_all_megatron_shards, convert
from optim_megatron_to_nano import remap_optim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meg-dir', required=True, help='path to iter_NNNNNNN directory')
    ap.add_argument('--meg-optim', required=True, help='output of recover_megatron_optim.py')
    ap.add_argument('--out', required=True, help='output ckpt.pt')
    args = ap.parse_args()

    print('[1] Loading + converting Megatron weights...', flush=True)
    meg = load_all_megatron_shards(args.meg_dir)
    sd = convert(meg)
    print(f'    {len(sd)} nano params')

    print('[2] Loading + remapping optim state...', flush=True)
    payload = torch.load(args.meg_optim, map_location='cpu', weights_only=False)
    nano_optim = remap_optim(payload['state'], payload['step'])
    step = payload['step']
    print(f'    {len(nano_optim)} optim entries, step={step}')

    # Build nano's GPT model_args (matches cybertron_moe_196_resume.py)
    model_args = dict(
        block_size=8192, vocab_size=152064, n_layer=9, n_head=4, n_embd=512,
        n_kv_head=2, kv_channels=64, dropout=0.0, bias=False, init_std=0.006,
        use_rope=True, rotary_base=50000, use_rmsnorm=True, norm_eps=1e-5,
        use_swiglu=True, ffn_hidden_size=1536, tie_embeddings=False,
        qk_layernorm=True, disable_scaled_init_method=True,
        use_moe=True, moe_layer_freq=[0] + [1] * 8, num_experts=144,
        moe_ffn_hidden_size=160, moe_router_topk=8, moe_n_group=8, moe_topk_group=1,
        moe_norm_topk_prob=True, moe_router_score_correction_coeff=0.001,
        moe_shared_expert_hidden_size=160,
    )

    # Build PyTorch-fused-AdamW state_dict matching nano's configure_optimizers ordering.
    # nano's configure_optimizers creates 2 param groups: decay (dim>=2) then nodecay (dim<2).
    # Each group's 'params' list is a list of INTEGER indices. state is keyed by integer index.
    # We need the same ordering as model.named_parameters() filtered by dim.
    from model import GPTConfig, GPT
    cfg = GPTConfig(**model_args)
    model = GPT(cfg)

    decay_params = []
    nodecay_params = []
    name_to_param = dict(model.named_parameters())
    for n, p in model.named_parameters():
        if p.dim() >= 2:
            decay_params.append((n, p))
        else:
            nodecay_params.append((n, p))
    all_params = decay_params + nodecay_params  # order used by opt.param_groups

    # Build the optim.state_dict() format:
    # {'state': {i: {...} for i in indices}, 'param_groups': [{'params': [0,1,..], ...}, ...]}
    opt_state = {'state': {}, 'param_groups': []}
    for i, (n, p) in enumerate(all_params):
        if n in nano_optim:
            e = nano_optim[n]
            opt_state['state'][i] = {
                'step': torch.tensor(float(e['step'])),  # device will be handled on load
                'exp_avg': e['exp_avg'].to(torch.float32),
                'exp_avg_sq': e['exp_avg_sq'].to(torch.float32),
            }
        else:
            print(f'    WARN: no optim for {n} (will start fresh)')

    opt_state['param_groups'] = [
        {'params': list(range(len(decay_params))),
         'lr': 1.2e-3, 'betas': (0.9, 0.95), 'eps': 1e-15,
         'weight_decay': 0.1, 'amsgrad': False, 'maximize': False,
         'foreach': None, 'capturable': False, 'differentiable': False,
         'fused': True},
        {'params': list(range(len(decay_params), len(all_params))),
         'lr': 1.2e-3, 'betas': (0.9, 0.95), 'eps': 1e-15,
         'weight_decay': 0.0, 'amsgrad': False, 'maximize': False,
         'foreach': None, 'capturable': False, 'differentiable': False,
         'fused': True},
    ]

    # Use Megatron's `iteration` (= completed training steps). This differs from Adam
    # optimizer step when there's a zero-grad patch (PAI iter_0 save has Adam step=1 but
    # iteration=0). For nano's LR schedule to match ref, we want iter_num to mean
    # "completed training steps" — matches ref.
    meg_iteration = int(torch.load(
        f'{args.meg_dir}/mp_rank_00_000/model_optim_rng.pt',
        map_location='cpu', weights_only=False).get('iteration', step))
    print(f'    Megatron iteration (completed steps) = {meg_iteration}; Adam step = {step}')
    ckpt = {
        'model': sd,
        'model_args': model_args,
        'optimizer': opt_state,
        'iter_num': meg_iteration,
        'best_val_loss': 1e9,
        'config': {},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ckpt, args.out)
    sz = os.path.getsize(args.out) / 1e9
    print(f'[3] wrote {args.out}  ({sz:.2f} GB)')


if __name__ == '__main__':
    main()
