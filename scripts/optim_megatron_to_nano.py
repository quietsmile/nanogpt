"""Convert Megatron optim state (from recover_megatron_optim.py) → nano AdamW state.

Applies the SAME transformations as scripts/megatron_to_nano.py to exp_avg / exp_avg_sq:
  - linear_qkv.weight → q_proj + k_proj + v_proj (GQA interleaved split)
  - linear_fc1 → gate_proj + up_proj (SwiGLU fused split)
  - experts.linear_fc{12}.weight{I} → stacked gate_weight/up_weight/down_weight

Input:  reports/short_window/meg_optim_iter500.pt (output of recover_megatron_optim.py)
Output: reports/short_window/nano_optim_iter500.pt
"""
from __future__ import annotations
import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from megatron_to_nano import (
    N_LAYER, N_HEAD, N_KV_HEAD, HEAD_DIM, N_EMBD, N_REP,
    NUM_EXPERTS, EP_SIZE, split_qkv, split_gate_up,
)


def remap_optim(meg_state: dict, step: int) -> dict:
    """Apply same reshaping to exp_avg/exp_avg_sq as to weights.

    meg_state: {meg_param_name: {'exp_avg': Tensor, 'exp_avg_sq': Tensor}}
    returns:   {nano_param_name: {'exp_avg': Tensor, 'exp_avg_sq': Tensor, 'step': int}}
    """
    nano: dict = {}

    def _copy(src_key, dst_key):
        s = meg_state[src_key]
        nano[dst_key] = {'exp_avg': s['exp_avg'].clone(), 'exp_avg_sq': s['exp_avg_sq'].clone()}

    # Direct mappings (no shape change)
    _copy('embedding.word_embeddings.weight', 'transformer.wte.weight')
    _copy('decoder.final_layernorm.weight',   'transformer.ln_f.weight')
    _copy('output_layer.weight',              'lm_head.weight')

    for L in range(N_LAYER):
        pm = f'decoder.layers.{L}'
        pn = f'transformer.h.{L}'
        # Pre-attn norm
        _copy(f'{pm}.self_attention.linear_qkv.layer_norm_weight', f'{pn}.ln_1.weight')
        # QKV split — apply split_qkv to BOTH exp_avg and exp_avg_sq
        qkv_state = meg_state[f'{pm}.self_attention.linear_qkv.weight']
        for st_key in ('exp_avg', 'exp_avg_sq'):
            q, k, v = split_qkv(qkv_state[st_key])
            nano.setdefault(f'{pn}.attn.q_proj.weight', {})[st_key] = q
            nano.setdefault(f'{pn}.attn.k_proj.weight', {})[st_key] = k
            nano.setdefault(f'{pn}.attn.v_proj.weight', {})[st_key] = v
        _copy(f'{pm}.self_attention.q_layernorm.weight', f'{pn}.attn.q_layernorm.weight')
        _copy(f'{pm}.self_attention.k_layernorm.weight', f'{pn}.attn.k_layernorm.weight')
        _copy(f'{pm}.self_attention.linear_proj.weight', f'{pn}.attn.c_proj.weight')

        if L == 0:
            # Dense layer
            _copy(f'{pm}.mlp.linear_fc1.layer_norm_weight', f'{pn}.ln_2.weight')
            fc1_state = meg_state[f'{pm}.mlp.linear_fc1.weight']
            for st_key in ('exp_avg', 'exp_avg_sq'):
                gate, up = split_gate_up(fc1_state[st_key])
                nano.setdefault(f'{pn}.mlp.gate_proj.weight', {})[st_key] = gate
                nano.setdefault(f'{pn}.mlp.up_proj.weight',   {})[st_key] = up
            _copy(f'{pm}.mlp.linear_fc2.weight', f'{pn}.mlp.down_proj.weight')
        else:
            # MoE layer
            _copy(f'{pm}.pre_mlp_layernorm.weight', f'{pn}.ln_2.weight')
            _copy(f'{pm}.mlp.router.weight', f'{pn}.mlp.router.linear.weight')

            # Stack expert optim states into [E, C, H] / [E, H, C]
            gate_list_ea, up_list_ea, down_list_ea = [], [], []
            gate_list_easq, up_list_easq, down_list_easq = [], [], []
            for I in range(NUM_EXPERTS):
                fc1_state = meg_state[f'{pm}.mlp.experts.linear_fc1.weight{I}']
                fc2_state = meg_state[f'{pm}.mlp.experts.linear_fc2.weight{I}']
                # Split linear_fc1 state into gate+up, transpose into [C, H]
                for state_list_gate, state_list_up, state_list_down, st_key in [
                    (gate_list_ea,   up_list_ea,   down_list_ea,   'exp_avg'),
                    (gate_list_easq, up_list_easq, down_list_easq, 'exp_avg_sq'),
                ]:
                    g, u = split_gate_up(fc1_state[st_key])     # each [160, 512]
                    state_list_gate.append(g.t().contiguous())   # [512, 160]
                    state_list_up.append(u.t().contiguous())
                    # fc2 is [512, 160], transpose to [160, 512]
                    state_list_down.append(fc2_state[st_key].t().contiguous())
            nano[f'{pn}.mlp.gate_weight'] = {
                'exp_avg': torch.stack(gate_list_ea, dim=0),
                'exp_avg_sq': torch.stack(gate_list_easq, dim=0),
            }
            nano[f'{pn}.mlp.up_weight'] = {
                'exp_avg': torch.stack(up_list_ea, dim=0),
                'exp_avg_sq': torch.stack(up_list_easq, dim=0),
            }
            nano[f'{pn}.mlp.down_weight'] = {
                'exp_avg': torch.stack(down_list_ea, dim=0),
                'exp_avg_sq': torch.stack(down_list_easq, dim=0),
            }
            # Shared expert
            s_fc1_state = meg_state[f'{pm}.mlp.shared_experts.linear_fc1.weight']
            s_fc2_state = meg_state[f'{pm}.mlp.shared_experts.linear_fc2.weight']
            for st_key in ('exp_avg', 'exp_avg_sq'):
                sg, su = split_gate_up(s_fc1_state[st_key])
                nano.setdefault(f'{pn}.mlp.shared_expert.gate_proj.weight', {})[st_key] = sg
                nano.setdefault(f'{pn}.mlp.shared_expert.up_proj.weight',   {})[st_key] = su
            nano[f'{pn}.mlp.shared_expert.down_proj.weight'] = {
                'exp_avg': s_fc2_state['exp_avg'].clone(),
                'exp_avg_sq': s_fc2_state['exp_avg_sq'].clone(),
            }

    # Add the step counter to each entry
    for k in nano:
        nano[k]['step'] = step
    return nano


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meg-optim', required=True, help='output of recover_megatron_optim.py')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    payload = torch.load(args.meg_optim, map_location='cpu', weights_only=False)
    meg_state = payload['state']
    step = payload['step']

    print(f'loaded {len(meg_state)} meg params, step={step}')
    nano = remap_optim(meg_state, step)
    print(f'produced {len(nano)} nano param entries')

    # Spot check shapes
    for k in ['transformer.wte.weight', 'transformer.h.0.attn.q_proj.weight',
              'transformer.h.0.mlp.gate_proj.weight', 'transformer.h.1.mlp.gate_weight']:
        if k in nano:
            s = nano[k]
            print(f'  {k}: exp_avg shape={tuple(s["exp_avg"].shape)} step={s["step"]}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({'state': nano, 'step': step}, args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
