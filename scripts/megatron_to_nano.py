"""Convert Megatron scaling_moe_00196 checkpoint → nanogpt state_dict.

Megatron layout (from iter_0001497/mp_rank_00_{0..3}/model_optim_rng.pt):
  embedding.word_embeddings.weight              [152064, 512]
  decoder.final_layernorm.weight                [512]
  decoder.layers.{L}.self_attention.linear_qkv.weight         [512, 512]    fused GQA QKV
  decoder.layers.{L}.self_attention.linear_qkv.layer_norm_weight [512]     pre-attn RMSNorm
  decoder.layers.{L}.self_attention.q_layernorm.weight        [64]
  decoder.layers.{L}.self_attention.k_layernorm.weight        [64]
  decoder.layers.{L}.self_attention.linear_proj.weight        [512, 256]
  decoder.layers.0.mlp.linear_fc1.weight                      [3072, 512]  dense SwiGLU [gate;up]
  decoder.layers.0.mlp.linear_fc1.layer_norm_weight           [512]        pre-mlp RMSNorm (dense)
  decoder.layers.0.mlp.linear_fc2.weight                      [512, 1536]
  decoder.layers.{1..8}.pre_mlp_layernorm.weight              [512]        pre-mlp RMSNorm (MoE)
  decoder.layers.{1..8}.mlp.router.weight                     [144, 512]
  decoder.layers.{1..8}.mlp.router.e_score_correction_bias                [144]        load-balance bias
  decoder.layers.{1..8}.mlp.experts.linear_fc1.weight{I}      [320, 512]   per-expert fused [gate;up]  (split across 4 EP shards: rank r has I in [r*36, r*36+35])
  decoder.layers.{1..8}.mlp.experts.linear_fc2.weight{I}      [512, 160]   per-expert down-proj
  decoder.layers.{1..8}.mlp.shared_experts.linear_fc1.weight  [320, 512]
  decoder.layers.{1..8}.mlp.shared_experts.linear_fc2.weight  [512, 160]
  output_layer.weight                           [152064, 512]

Nano layout (from current model.py):
  transformer.wte.weight                        [152064, 512]
  transformer.ln_f.weight                       [512]
  transformer.h.{L}.ln_1.weight                 [512]
  transformer.h.{L}.attn.q_proj.weight          [256, 512]
  transformer.h.{L}.attn.k_proj.weight          [128, 512]
  transformer.h.{L}.attn.v_proj.weight          [128, 512]
  transformer.h.{L}.attn.q_layernorm.weight     [64]
  transformer.h.{L}.attn.k_layernorm.weight     [64]
  transformer.h.{L}.attn.c_proj.weight          [512, 256]
  transformer.h.{L}.ln_2.weight                 [512]
  Dense (layer 0):
    transformer.h.0.mlp.gate_proj.weight        [1536, 512]
    transformer.h.0.mlp.up_proj.weight          [1536, 512]
    transformer.h.0.mlp.down_proj.weight        [512, 1536]
  MoE (layers 1-8):
    transformer.h.{L}.mlp.router.linear.weight  [144, 512]
    transformer.h.{L}.mlp.router.e_score_correction_bias  [144]  (buffer)
    transformer.h.{L}.mlp.gate_weight           [144, 512, 160]  stacked [E, C, H]
    transformer.h.{L}.mlp.up_weight             [144, 512, 160]
    transformer.h.{L}.mlp.down_weight           [144, 160, 512]
    transformer.h.{L}.mlp.shared_expert.gate_proj.weight  [160, 512]
    transformer.h.{L}.mlp.shared_expert.up_proj.weight    [160, 512]
    transformer.h.{L}.mlp.shared_expert.down_proj.weight  [512, 160]
  lm_head.weight                                [152064, 512]

Key transformations:
  1. linear_qkv.weight [512,512] → q_proj/k_proj/v_proj:
     Megatron GQA interleaves per KV-group (n_rep=2 Q + 1 K + 1 V) × head_dim=64:
       rows 0..63:    Q_head_0 (group 0)
       rows 64..127:  Q_head_1 (group 0)
       rows 128..191: K_head_0 (group 0)
       rows 192..255: V_head_0 (group 0)
       rows 256..319: Q_head_2 (group 1)
       rows 320..383: Q_head_3 (group 1)
       rows 384..447: K_head_1 (group 1)
       rows 448..511: V_head_1 (group 1)
  2. linear_fc1 (SwiGLU fused [gate;up]):
     Megatron rows 0..H-1 = gate, rows H..2H-1 = up
     Split into nano gate_proj and up_proj
  3. Routed experts split across 4 EP shards — concat all 4 then stack into [E, C, H] tensor.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from typing import Dict

import torch

# Paths
MEGATRON_CKPT_DIR = '/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196/iter_0001497'  # default; overridable via --ckpt-dir
CYBERTRON_PY  = '/newcpfs/user/yuchen/llm/cybertron_dots3.0_swa'
MEGATRON_PY   = '/newcpfs/user/yuchen/llm/megatron_dots3.0_swa'

# Arch constants for 00196
N_LAYER = 9
N_HEAD = 4
N_KV_HEAD = 2
HEAD_DIM = 64
N_EMBD = 512
N_REP = N_HEAD // N_KV_HEAD  # 2
FFN_HIDDEN = 1536
MOE_FFN_HIDDEN = 160
NUM_EXPERTS = 144
EP_SIZE = 4


def load_all_megatron_shards(ckpt_dir: str = None) -> Dict[str, torch.Tensor]:
    ckpt_dir = ckpt_dir or MEGATRON_CKPT_DIR
    """Load the 4 EP shards and merge into a single dict keyed by Megatron names.

    Non-expert params are duplicated across shards → dedup (take rank-0 copy).
    Expert params are sharded by EP → rank r has experts [r*36..r*36+35] under
    names weight0..weight35 (LOCAL indices); remap to GLOBAL expert ids.
    """
    sys.path.insert(0, CYBERTRON_PY)
    sys.path.insert(0, MEGATRON_PY)
    merged: Dict[str, torch.Tensor] = {}
    for r in range(EP_SIZE):
        ck = torch.load(f'{ckpt_dir}/mp_rank_00_{r:03d}/model_optim_rng.pt',
                        map_location='cpu', weights_only=False)
        sd = ck['model']
        if hasattr(sd, 'state_dict'):
            sd = sd.state_dict()
        for k, v in sd.items():
            if '_extra_state' in k:
                continue
            if not isinstance(v, torch.Tensor):
                continue
            if '.experts.linear_fc' in k:
                # Rename e.g. decoder.layers.1.mlp.experts.linear_fc1.weight0 →
                # decoder.layers.1.mlp.experts.linear_fc1.weight{36*r + 0}
                import re
                m = re.match(r'(.+experts\.linear_fc[12])\.weight(\d+)$', k)
                if m:
                    prefix, local_i = m.group(1), int(m.group(2))
                    gi = r * 36 + local_i
                    new_k = f'{prefix}.weight{gi}'
                    merged[new_k] = v
                    continue
                # Fallback: keep as-is
            if k not in merged:
                merged[k] = v
    return merged


def split_qkv(qkv: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """Split Megatron GQA-interleaved QKV [512,512] into q [256,512], k [128,512], v [128,512]."""
    group_rows = (N_REP + 2) * HEAD_DIM  # 4*64 = 256 rows per KV group
    n_groups = N_KV_HEAD                  # 2 groups
    q_rows, k_rows, v_rows = [], [], []
    for g in range(n_groups):
        base = g * group_rows
        for h in range(N_REP):
            q_rows.append(qkv[base + h * HEAD_DIM : base + (h + 1) * HEAD_DIM])
        k_rows.append(qkv[base + N_REP * HEAD_DIM : base + (N_REP + 1) * HEAD_DIM])
        v_rows.append(qkv[base + (N_REP + 1) * HEAD_DIM : base + (N_REP + 2) * HEAD_DIM])
    q = torch.cat(q_rows, dim=0)  # [256, 512]
    k = torch.cat(k_rows, dim=0)  # [128, 512]
    v = torch.cat(v_rows, dim=0)  # [128, 512]
    return q, k, v


def split_gate_up(fc1: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Split Megatron SwiGLU fused [gate; up] along dim 0."""
    H = fc1.shape[0] // 2
    return fc1[:H].clone(), fc1[H:].clone()


def convert(meg: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert Megatron merged dict to nano state_dict."""
    nano: Dict[str, torch.Tensor] = {}

    # Embedding + final norm + output head
    nano['transformer.wte.weight'] = meg['embedding.word_embeddings.weight'].clone()
    nano['transformer.ln_f.weight'] = meg['decoder.final_layernorm.weight'].clone()
    nano['lm_head.weight'] = meg['output_layer.weight'].clone()

    # Per-layer
    for L in range(N_LAYER):
        prefix_meg = f'decoder.layers.{L}'
        prefix_nano = f'transformer.h.{L}'

        # Pre-attention RMSNorm (inside linear_qkv.layer_norm_weight in Megatron)
        nano[f'{prefix_nano}.ln_1.weight'] = meg[f'{prefix_meg}.self_attention.linear_qkv.layer_norm_weight'].clone()

        # QKV split
        q, k, v = split_qkv(meg[f'{prefix_meg}.self_attention.linear_qkv.weight'])
        nano[f'{prefix_nano}.attn.q_proj.weight'] = q
        nano[f'{prefix_nano}.attn.k_proj.weight'] = k
        nano[f'{prefix_nano}.attn.v_proj.weight'] = v
        nano[f'{prefix_nano}.attn.q_layernorm.weight'] = meg[f'{prefix_meg}.self_attention.q_layernorm.weight'].clone()
        nano[f'{prefix_nano}.attn.k_layernorm.weight'] = meg[f'{prefix_meg}.self_attention.k_layernorm.weight'].clone()
        nano[f'{prefix_nano}.attn.c_proj.weight'] = meg[f'{prefix_meg}.self_attention.linear_proj.weight'].clone()

        # Pre-MLP RMSNorm (different location depending on MoE vs dense)
        if L == 0:
            nano[f'{prefix_nano}.ln_2.weight'] = meg[f'{prefix_meg}.mlp.linear_fc1.layer_norm_weight'].clone()
            # Dense FFN
            gate, up = split_gate_up(meg[f'{prefix_meg}.mlp.linear_fc1.weight'])
            nano[f'{prefix_nano}.mlp.gate_proj.weight'] = gate
            nano[f'{prefix_nano}.mlp.up_proj.weight'] = up
            nano[f'{prefix_nano}.mlp.down_proj.weight'] = meg[f'{prefix_meg}.mlp.linear_fc2.weight'].clone()
        else:
            nano[f'{prefix_nano}.ln_2.weight'] = meg[f'{prefix_meg}.pre_mlp_layernorm.weight'].clone()
            # Router
            nano[f'{prefix_nano}.mlp.router.linear.weight'] = meg[f'{prefix_meg}.mlp.router.weight'].clone()
            nano[f'{prefix_nano}.mlp.router.e_score_correction_bias'] = meg[f'{prefix_meg}.mlp.router.e_score_correction_bias'].clone()
            # Routed experts — build stacked tensors [E, C, H] and [E, H, C]
            gate_list, up_list, down_list = [], [], []
            for I in range(NUM_EXPERTS):
                fc1 = meg[f'{prefix_meg}.mlp.experts.linear_fc1.weight{I}']  # [320, 512]
                fc2 = meg[f'{prefix_meg}.mlp.experts.linear_fc2.weight{I}']  # [512, 160]
                g_i, u_i = split_gate_up(fc1)                              # each [160, 512]
                # nano layout: gate_weight [E, C=512, H=160] — so each expert is [512, 160] (transpose)
                gate_list.append(g_i.t().contiguous())                      # [512, 160]
                up_list.append(u_i.t().contiguous())                        # [512, 160]
                # down_weight [E, H=160, C=512] — Megatron fc2 is [512, 160], transpose to [160, 512]
                down_list.append(fc2.t().contiguous())                      # [160, 512]
            nano[f'{prefix_nano}.mlp.gate_weight'] = torch.stack(gate_list, dim=0)  # [144, 512, 160]
            nano[f'{prefix_nano}.mlp.up_weight']   = torch.stack(up_list, dim=0)
            nano[f'{prefix_nano}.mlp.down_weight'] = torch.stack(down_list, dim=0)
            # Shared expert (same format as ExpertMLP: gate_proj [H, C], up_proj [H, C], down_proj [C, H])
            s_fc1 = meg[f'{prefix_meg}.mlp.shared_experts.linear_fc1.weight']  # [320, 512]
            s_fc2 = meg[f'{prefix_meg}.mlp.shared_experts.linear_fc2.weight']  # [512, 160]
            s_gate, s_up = split_gate_up(s_fc1)                                # [160, 512]
            nano[f'{prefix_nano}.mlp.shared_expert.gate_proj.weight'] = s_gate
            nano[f'{prefix_nano}.mlp.shared_expert.up_proj.weight']   = s_up
            nano[f'{prefix_nano}.mlp.shared_expert.down_proj.weight'] = s_fc2  # [512, 160] (C, H)
    return nano


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default=MEGATRON_CKPT_DIR, help='Megatron iter_XXXXXXX dir')
    ap.add_argument('--out', default='/home/claudeuser/nanogpt/reports/megatron_to_nano_ckpt.pt')
    ap.add_argument('--shape-only', action='store_true',
                    help='Skip writing the ckpt, just print shape map')
    args = ap.parse_args()

    print(f"Loading Megatron shards from {args.ckpt_dir}...", flush=True)
    meg = load_all_megatron_shards(args.ckpt_dir)
    print(f"  merged dict: {len(meg)} keys")

    print("Converting to nano layout...", flush=True)
    nano = convert(meg)
    print(f"  nano state_dict: {len(nano)} keys")

    # Summary
    total_elems = sum(v.numel() for v in nano.values())
    print(f"  total params: {total_elems:,} ({total_elems/1e6:.2f}M)")

    # Spot check a couple shapes
    for n in ['transformer.wte.weight', 'transformer.h.0.attn.q_proj.weight',
              'transformer.h.0.mlp.gate_proj.weight', 'transformer.h.1.mlp.gate_weight',
              'transformer.h.1.mlp.router.linear.weight']:
        if n in nano:
            print(f"  {n}: shape={tuple(nano[n].shape)}, dtype={nano[n].dtype}")

    if not args.shape_only:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save(nano, args.out)
        sz = os.path.getsize(args.out) / 1e9
        print(f"wrote {args.out} ({sz:.2f} GB)")


if __name__ == '__main__':
    main()
