"""nanogpt ↔ Megatron (cybertron_dots3.0_swa) parameter-name mapping for MoE 00196.

Handles:
  - single nanogpt name → one or multiple Megatron names (QKV splitting, expert shard splitting)
  - reverse lookup given Megatron state_dict
  - shape invariants for each mapping
"""
from dataclasses import dataclass
from typing import Callable, List, Optional
import re


@dataclass
class Mapping:
    # nanogpt param name template, e.g. "transformer.h.{L}.attn.q_proj.weight"
    nano: str
    # function: (nano_name, megatron_state_dict_keys) -> list[(megatron_name, slice_info)]
    # slice_info is None for 1:1; for split/concat it describes the row/col range or list index.
    resolve: Callable[[str, List[str]], list]
    # Optional shape checker (nano_shape, megatron_shapes_list) -> bool
    check: Optional[Callable] = None


# ---------- Helpers ----------
def _layer_idx(nano_name: str) -> int:
    m = re.search(r'transformer\.h\.(\d+)\.', nano_name)
    return int(m.group(1)) if m else -1


def _is_moe_layer(layer_idx: int, moe_layer_freq: list) -> bool:
    return bool(moe_layer_freq[layer_idx]) if layer_idx >= 0 and layer_idx < len(moe_layer_freq) else False


# ---------- Mapping builder ----------
def build_name_map(model_cfg, moe_layer_freq):
    """Return a dict: nano_name → list[(megatron_name, transform)] where transform is a label.

    Args:
        model_cfg: object with n_layer, n_head, n_kv_head, kv_channels, n_embd, num_experts, use_moe.
        moe_layer_freq: list like [0] + [1]*8.
    """
    n_layer = model_cfg.n_layer
    n_head = model_cfg.n_head
    n_kv_head = model_cfg.n_kv_head if model_cfg.n_kv_head else n_head
    head_dim = model_cfg.kv_channels if model_cfg.kv_channels else model_cfg.n_embd // n_head
    n_embd = model_cfg.n_embd
    num_experts = model_cfg.num_experts
    q_rows = n_head * head_dim       # 256
    k_rows = n_kv_head * head_dim    # 128
    v_rows = n_kv_head * head_dim    # 128

    m = {}

    # ---- Embedding / final norm / output head ----
    m['transformer.wte.weight'] = [('embedding.word_embeddings.weight', 'identity')]
    m['transformer.ln_f.weight'] = [('decoder.final_layernorm.weight', 'identity')]
    m['lm_head.weight'] = [('output_layer.weight', 'identity')]

    for L in range(n_layer):
        # Pre-attention RMSNorm lives INSIDE TE's LayerNormLinear as linear_qkv.layer_norm_weight.
        m[f'transformer.h.{L}.ln_1.weight'] = [
            (f'decoder.layers.{L}.self_attention.linear_qkv.layer_norm_weight', 'identity')
        ]
        # QKV split: Megatron linear_qkv.weight shape (q+k+v, n_embd) = (512,512).
        # Row layout is Megatron-interleaved: groups of (n_rep Q + 1 K + 1 V) × head_dim.
        # We expose three nano params → 3 slices of the Megatron weight; the test checks the row sum.
        m[f'transformer.h.{L}.attn.q_proj.weight'] = [
            (f'decoder.layers.{L}.self_attention.linear_qkv.weight', f'qkv_split:q[{q_rows}]')
        ]
        m[f'transformer.h.{L}.attn.k_proj.weight'] = [
            (f'decoder.layers.{L}.self_attention.linear_qkv.weight', f'qkv_split:k[{k_rows}]')
        ]
        m[f'transformer.h.{L}.attn.v_proj.weight'] = [
            (f'decoder.layers.{L}.self_attention.linear_qkv.weight', f'qkv_split:v[{v_rows}]')
        ]
        m[f'transformer.h.{L}.attn.q_layernorm.weight'] = [
            (f'decoder.layers.{L}.self_attention.q_layernorm.weight', 'identity')
        ]
        m[f'transformer.h.{L}.attn.k_layernorm.weight'] = [
            (f'decoder.layers.{L}.self_attention.k_layernorm.weight', 'identity')
        ]
        m[f'transformer.h.{L}.attn.c_proj.weight'] = [
            (f'decoder.layers.{L}.self_attention.linear_proj.weight', 'identity')
        ]
        # Pre-MLP RMSNorm inside linear_fc1.layer_norm_weight
        m[f'transformer.h.{L}.ln_2.weight'] = [
            (f'decoder.layers.{L}.mlp.linear_fc1.layer_norm_weight', 'identity')
            if not _is_moe_layer(L, moe_layer_freq) else
            (f'decoder.layers.{L}.pre_mlp_layernorm.weight', 'identity_moe_prenorm')
        ]

        if _is_moe_layer(L, moe_layer_freq):
            # Router: nano `mlp.router.linear.weight` (num_experts, n_embd) ↔ Megatron `mlp.router.weight`
            m[f'transformer.h.{L}.mlp.router.linear.weight'] = [
                (f'decoder.layers.{L}.mlp.router.weight', 'identity')
            ]
            # e_score_correction_bias — buffer in nano, also a buffer in Megatron
            m[f'transformer.h.{L}.mlp.router.e_score_correction_bias'] = [
                (f'decoder.layers.{L}.mlp.router.expert_bias', 'identity_buffer')
            ]
            # Routed experts are stacked: nano uses [E, C, H] / [E, H, C] tensors
            # mlp.gate_weight ↔ Megatron `mlp.experts.linear_fc1.weight{I}` (gate slice of fc1)
            m[f'transformer.h.{L}.mlp.gate_weight'] = [
                (f'decoder.layers.{L}.mlp.experts.linear_fc1.weight{{I}}', 'stacked_swiglu:gate')
            ]
            m[f'transformer.h.{L}.mlp.up_weight'] = [
                (f'decoder.layers.{L}.mlp.experts.linear_fc1.weight{{I}}', 'stacked_swiglu:up')
            ]
            m[f'transformer.h.{L}.mlp.down_weight'] = [
                (f'decoder.layers.{L}.mlp.experts.linear_fc2.weight{{I}}', 'stacked_identity')
            ]
            # Shared expert
            m[f'transformer.h.{L}.mlp.shared_expert.gate_proj.weight'] = [
                (f'decoder.layers.{L}.mlp.shared_experts.linear_fc1.weight', 'swiglu_split:gate')
            ]
            m[f'transformer.h.{L}.mlp.shared_expert.up_proj.weight'] = [
                (f'decoder.layers.{L}.mlp.shared_experts.linear_fc1.weight', 'swiglu_split:up')
            ]
            m[f'transformer.h.{L}.mlp.shared_expert.down_proj.weight'] = [
                (f'decoder.layers.{L}.mlp.shared_experts.linear_fc2.weight', 'identity')
            ]
        else:
            # Dense SwiGLU FFN
            m[f'transformer.h.{L}.mlp.gate_proj.weight'] = [
                (f'decoder.layers.{L}.mlp.linear_fc1.weight', 'swiglu_split:gate')
            ]
            m[f'transformer.h.{L}.mlp.up_proj.weight'] = [
                (f'decoder.layers.{L}.mlp.linear_fc1.weight', 'swiglu_split:up')
            ]
            m[f'transformer.h.{L}.mlp.down_proj.weight'] = [
                (f'decoder.layers.{L}.mlp.linear_fc2.weight', 'identity')
            ]
    return m


def find_megatron_expert_shape(meg_shape_json, layer_idx, expert_idx, which='fc1'):
    """Return shape of a particular expert's linear_fcX.weight{I}. Used by tests.

    meg_shape_json: the dict loaded from reference/megatron_state_dict_shapes.json
    """
    # EP=4; experts 0..35 on rank0, 36..71 on rank1, etc.
    rank = expert_idx // 36
    local = expert_idx  # Megatron keeps global indices in some EP impls; the dump we have keeps
                       # per-rank naming starting at 0. So local index in rank shard = expert_idx % 36.
    local = expert_idx - rank * 36
    rank_key = f'rank{rank:03d}'
    sd = meg_shape_json[rank_key]
    key = f'decoder.layers.{layer_idx}.mlp.experts.linear_{which}.weight{local}'
    if key not in sd:
        return None
    return sd[key]['shape']
