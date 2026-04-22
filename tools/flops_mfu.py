"""Param count / FLOPs / MFU tools for cybertron MoE alignment.

All conventions match Megatron's `get_num_parameters` and `num_floating_point_operations`.
The reference Megatron job (scaling_moe_00196) records `num_floating_point_operations_so_far`
in its checkpoint `args`, which gives an independent ground-truth.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class ParamBreakdown:
    total: int
    trainable: int
    embedding: int        # word_embeddings + output_layer
    non_embedding: int    # total - embedding
    attn_per_layer: int
    attn_norm_per_layer: int
    mlp_dense_per_layer: int    # layer 0 only if MoE config
    moe_routed_per_layer: int   # per MoE layer, routed experts only (shared excluded)
    moe_shared_per_layer: int
    moe_router_per_layer: int
    final_norm: int
    total_routed: int           # routed experts across all MoE layers
    total_shared: int
    active_per_token: int       # params seen by one token's forward pass

    def pretty(self):
        lines = [
            f"=== param breakdown ===",
            f"total:            {self.total:>15,}  ({self.total/1e6:.2f}M)",
            f"trainable:        {self.trainable:>15,}  ({self.trainable/1e6:.2f}M)",
            f"embedding+out:    {self.embedding:>15,}  ({self.embedding/1e6:.2f}M)",
            f"non-embedding:    {self.non_embedding:>15,}  ({self.non_embedding/1e6:.2f}M)",
            f"total routed:     {self.total_routed:>15,}  ({self.total_routed/1e6:.2f}M)",
            f"total shared:     {self.total_shared:>15,}  ({self.total_shared/1e6:.2f}M)",
            f"active/token:     {self.active_per_token:>15,}  ({self.active_per_token/1e6:.2f}M)",
            f"per-layer attn:   {self.attn_per_layer:,}  attn-norm: {self.attn_norm_per_layer}",
            f"per-layer MoE:    routed={self.moe_routed_per_layer:,}  "
                f"shared={self.moe_shared_per_layer:,}  router={self.moe_router_per_layer:,}",
            f"layer-0 dense:    {self.mlp_dense_per_layer:,}",
            f"final norm:       {self.final_norm}",
        ]
        return '\n'.join(lines)


def count_params_detailed(model) -> ParamBreakdown:
    """Count params on a nanogpt GPT instance built with cybertron_moe_196 config."""
    import torch.nn as nn

    total = 0
    trainable = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    # Break by structure using attribute access (works with nanogpt GPT naming)
    cfg = model.config
    L = cfg.n_layer
    moe_layer_freq = cfg.moe_layer_freq if (cfg.use_moe and cfg.moe_layer_freq is not None) else [0]*L

    embedding = model.transformer.wte.weight.numel()
    if getattr(model, 'lm_head', None) is not None and not cfg.tie_embeddings:
        embedding += model.lm_head.weight.numel()

    # Attention (identical on all layers)
    attn = model.transformer.h[0].attn
    attn_proj = 0
    for name in ['q_proj','k_proj','v_proj','c_proj']:
        if hasattr(attn, name):
            attn_proj += getattr(attn, name).weight.numel()
    attn_norm = 0
    for name in ['q_layernorm','k_layernorm']:
        m = getattr(attn, name, None)
        if m is not None:
            attn_norm += m.weight.numel()
    # Pre-norm per layer
    pre_norm = (model.transformer.h[0].ln_1.weight.numel()
                + model.transformer.h[0].ln_2.weight.numel())

    # MLP (dense and MoE)
    mlp_dense = 0
    moe_routed_pl = 0
    moe_shared_pl = 0
    moe_router_pl = 0
    total_routed = 0
    total_shared = 0
    for i, layer in enumerate(model.transformer.h):
        mlp = layer.mlp
        if hasattr(mlp, 'experts'):       # MoE
            r = sum(p.numel() for e in mlp.experts for p in e.parameters())
            s = sum(p.numel() for p in mlp.shared_expert.parameters()) if mlp.shared_expert is not None else 0
            rr = 0  # router
            if hasattr(mlp, 'router'):
                rr = sum(p.numel() for p in mlp.router.parameters() if p.requires_grad)
            moe_routed_pl = r
            moe_shared_pl = s
            moe_router_pl = rr
            total_routed += r
            total_shared += s
        else:                             # dense
            mlp_dense = sum(p.numel() for p in mlp.parameters())

    final_norm = model.transformer.ln_f.weight.numel()

    # Active per-token = embedding lookup + per-layer (norms + attn + (dense OR shared + topk experts + router))
    topk = cfg.moe_router_topk if cfg.use_moe else 0
    per_expert = moe_routed_pl // cfg.num_experts if cfg.use_moe and cfg.num_experts else 0
    active_mlp_moe = moe_shared_pl + topk * per_expert + moe_router_pl
    active_per_token = 0
    # Embedding (one row only, but embedding has full matrix; "active" counts lookup as 1 row)
    active_per_token += cfg.n_embd  # one row of wte
    # (lm_head similarly one row activation at output for CE, but to match Megatron "active params"
    #  we count full lm_head + full wte since both are used by every token via weights — actually
    #  Megatron counts them as params, not re-activated. Following convention: include embedding.)
    active_per_token += embedding
    active_per_token += final_norm
    for i in range(L):
        active_per_token += pre_norm + attn_proj + attn_norm
        if moe_layer_freq[i]:
            active_per_token += active_mlp_moe
        else:
            active_per_token += mlp_dense

    return ParamBreakdown(
        total=total,
        trainable=trainable,
        embedding=embedding,
        non_embedding=total - embedding,
        attn_per_layer=attn_proj,
        attn_norm_per_layer=attn_norm,
        mlp_dense_per_layer=mlp_dense,
        moe_routed_per_layer=moe_routed_pl,
        moe_shared_per_layer=moe_shared_pl,
        moe_router_per_layer=moe_router_pl,
        final_norm=final_norm,
        total_routed=total_routed,
        total_shared=total_shared,
        active_per_token=active_per_token,
    )


def compute_flops_per_step(cfg: Any, seq_len: int, global_bs: int) -> dict:
    """Return a FLOPs breakdown per training step (forward+backward = 3× forward).

    Follows Megatron's `num_floating_point_operations` convention for GQA+MoE:
      fwd = B * S * ( 2 * active_params   # linear ops
                     + attention_ops      # 4 * S * n_head * head_dim (Q*K + P*V)
                     )
      train = 3 * fwd  (bwd ≈ 2× fwd)
    """
    L = cfg.n_layer
    H = cfg.n_embd
    n_head = cfg.n_head
    n_kv_head = cfg.n_kv_head or n_head
    head_dim = cfg.kv_channels or (H // n_head)
    vocab = cfg.vocab_size_override or cfg.vocab_size
    ffn = cfg.ffn_hidden_size
    moe_ffn = cfg.moe_ffn_hidden_size
    num_experts = cfg.num_experts
    topk = cfg.moe_router_topk if cfg.use_moe else 0

    moe_layer_freq = cfg.moe_layer_freq or [0]*L
    n_moe = sum(moe_layer_freq)
    n_dense = L - n_moe

    # Per-token linear op params (flops = 2 * params per mac)
    # attention qkv proj: (2H*H + 2 * n_kv_head*head_dim * H) — GQA
    attn_linear_macs = H * (n_head*head_dim) + 2 * H * (n_kv_head*head_dim) + (n_head*head_dim) * H
    # dense SwiGLU: 3 * H * ffn
    dense_mlp_macs = 3 * H * ffn
    # MoE per token: shared (3*H*moe_ffn) + topk * (3*H*moe_ffn) + router (H*num_experts)
    moe_macs = 3 * H * moe_ffn + topk * 3 * H * moe_ffn + H * num_experts
    # embedding+lm_head are counted via lm_head only (wte is lookup, no mac in forward)
    lm_head_macs = H * vocab

    per_token_macs = (n_dense * (attn_linear_macs + dense_mlp_macs)
                      + n_moe * (attn_linear_macs + moe_macs)
                      + lm_head_macs)
    # Attention matmul (scales with S, not params):
    # flops per token per layer = 4 * S * n_head * head_dim  (Q*K + P*V, each 2*S*n_head*head_dim)
    attn_matmul_per_token = L * 4 * seq_len * n_head * head_dim

    fwd_flops_per_token = 2 * per_token_macs + attn_matmul_per_token
    tokens_per_step = global_bs * seq_len
    fwd_flops = tokens_per_step * fwd_flops_per_token
    train_flops = 3 * fwd_flops

    return {
        'tokens_per_step': tokens_per_step,
        'fwd_flops_per_token': fwd_flops_per_token,
        'fwd_flops_per_step': fwd_flops,
        'train_flops_per_step': train_flops,
        'attn_linear_macs': attn_linear_macs,
        'dense_mlp_macs': dense_mlp_macs,
        'moe_macs': moe_macs,
        'lm_head_macs': lm_head_macs,
        'attn_matmul_per_token': attn_matmul_per_token,
    }


def compute_mfu(train_flops_per_step: float, step_time_s: float, n_gpu: int,
                peak_flops_per_gpu: float) -> float:
    """MFU = actual / peak. Peak defaults per GPU in bf16."""
    peak = n_gpu * peak_flops_per_gpu
    achieved = train_flops_per_step / step_time_s
    return achieved / peak


# Common peaks (bf16 tensor core, vendor-spec):
PEAK_FLOPS = {
    'H100_SXM_bf16': 989e12,
    'H100_PCIe_bf16': 756e12,
    'A100_SXM_bf16': 312e12,
}
