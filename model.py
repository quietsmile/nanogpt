"""
Full definition of a GPT Language Model, supporting both original GPT-2 style
and Cybertron-aligned architecture (RMSNorm, RoPE, GQA, SwiGLU, qk_layernorm).

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """RMS normalization (no bias, no mean subtraction)."""

    def __init__(self, ndim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x):
        # Variance in fp32, but output must match x.dtype (under autocast, bf16 residual).
        # If we multiplied by self.weight (fp32) last, the widest-rule would promote output to
        # fp32, leaking fp32 into the residual stream — ref keeps bf16, so we match by casting
        # weight too.
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        normed = (x.float() * norm * self.weight.float()).to(x.dtype)
        return normed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Matches the cybertron/megatron implementation:
    - Applies to the first rotary_dim dimensions of Q and K
    - Uses the standard RoPE formula with base=50000 (matching rotary_base in config)
    """

    def __init__(self, dim, base=50000, max_seq_len=8192):
        super().__init__()
        self.dim = dim
        self.base = base
        # inv_freq MUST stay fp32; bf16 has 8-bit mantissa → positions > 256 round to multiples of 2
        # which wrecks RoPE at seq_len=8192. We recompute cos/sin fresh in fp32 every forward,
        # casting only the final output to q.dtype. Register as non-persistent, force-fp32 buffer.
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len=None, position_ids=None):
        # position_ids: [B, T] int tensor of per-token rotary positions. When None,
        # defaults to arange(seq_len). Use TE's fused_apply_rotary_pos_emb for bitwise
        # match to cybertron/Megatron ref (which uses apply_rope_fusion=True). Plain
        # Python cos/sin-based RoPE differs from fused kernel by L1=1.16e-3 on typical
        # inputs due to bf16 intermediate rounding inside rotate_half.
        if seq_len is None:
            seq_len = q.shape[-2]
        # Compute freqs (fp32 for long sequences)
        inv = self.inv_freq.float()
        if position_ids is None:
            t = torch.arange(seq_len, device=q.device, dtype=torch.float32)
            freqs = torch.outer(t, inv)                          # [T, dim/2]
        else:
            t = position_ids.float()                              # [B, T]
            freqs = t.unsqueeze(-1) * inv                         # [B, T, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)                   # [..., dim] fp32
        # TE's fused_apply_rotary_pos_emb expects [S, B, H, D] (sbhd) format with
        # freqs [S, 1, 1, D]. We receive q,k in [B, H, S, D] → permute to sbhd.
        # The fused kernel is CUDA-only and silently returns NaN on CPU tensors,
        # so also gate on q.is_cuda — lets CPU unit tests fall through to the
        # unfused path.
        try:
            from megatron.core.extensions.transformer_engine import fused_apply_rotary_pos_emb
            use_fused = (position_ids is None) and q.is_cuda
        except ImportError:
            use_fused = False
        if use_fused:
            freqs_sbhd = emb.view(seq_len, 1, 1, -1)
            q_sbhd = q.permute(2, 0, 1, 3).contiguous()
            k_sbhd = k.permute(2, 0, 1, 3).contiguous()
            q_out = fused_apply_rotary_pos_emb(q_sbhd, freqs_sbhd, interleaved=False).permute(1, 2, 0, 3).contiguous()
            k_out = fused_apply_rotary_pos_emb(k_sbhd, freqs_sbhd, interleaved=False).permute(1, 2, 0, 3).contiguous()
            return q_out, k_out
        # Fallback: unfused Python impl
        if position_ids is None:
            cos = emb.cos().to(q.dtype)[None, None, :, :]
            sin = emb.sin().to(q.dtype)[None, None, :, :]
        else:
            cos = emb.cos().to(q.dtype).unsqueeze(1)
            sin = emb.sin().to(q.dtype).unsqueeze(1)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # head_dim: use explicit kv_channels if set, else n_embd // n_head
        self.head_dim = config.kv_channels if config.kv_channels is not None else config.n_embd // config.n_head

        # GQA support
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        assert config.n_head % self.n_kv_head == 0
        self.n_rep = config.n_head // self.n_kv_head  # repetitions for GQA

        if config.use_rope:
            # Separate Q, K, V projections for GQA
            # Q: n_embd → n_head * head_dim
            # K, V: n_embd → n_kv_head * head_dim
            self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
            self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
            self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
            self.rotary_emb = RotaryEmbedding(
                self.head_dim, base=config.rotary_base, max_seq_len=config.block_size
            )
            # qk_layernorm: RMSNorm on Q and K per head, BEFORE RoPE
            # Applied to shape [B, heads, T, head_dim]
            if config.qk_layernorm:
                self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
                self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
            else:
                self.q_layernorm = None
                self.k_layernorm = None
        else:
            # Original combined QKV for standard MHA (GPT-2 style)
            assert self.n_kv_head == self.n_head, "GQA requires use_rope=True or matching heads"
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.q_layernorm = None
            self.k_layernorm = None

        # Output projection: n_head * head_dim → n_embd
        self.c_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.use_rope = config.use_rope

        # flash attention support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        self.attention_impl = getattr(config, 'attention_impl', 'sdpa')
        if self.attention_impl == 'te':
            import transformer_engine.pytorch as _te
            self._te_attn = _te.DotProductAttention(
                num_attention_heads=config.n_head,
                kv_channels=self.head_dim,
                num_gqa_groups=self.n_kv_head,
                attention_dropout=0.0,
                qkv_format='bshd',
                attn_mask_type='causal',
            )

    def forward(self, x, attn_mask=None, position_ids=None):
        B, T, C = x.size()

        if self.use_rope:
            q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

            # qk_layernorm: applied before RoPE, matching cybertron's order
            if self.q_layernorm is not None:
                q = self.q_layernorm(q)
            if self.k_layernorm is not None:
                k = self.k_layernorm(k)

            q, k = self.rotary_emb(q, k, seq_len=T, position_ids=position_ids)

            # Expand KV heads for GQA (SDPA/manual need full-head KV; TE handles GQA natively)
            if self.n_rep > 1 and self.attention_impl != 'te':
                k = k.unsqueeze(2).expand(B, self.n_kv_head, self.n_rep, T, self.head_dim)
                k = k.reshape(B, self.n_head, T, self.head_dim)
                v = v.unsqueeze(2).expand(B, self.n_kv_head, self.n_rep, T, self.head_dim)
                v = v.reshape(B, self.n_head, T, self.head_dim)
        else:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.attention_impl == 'te':
            # TE expects [B, S, H, D] with qkv_format='bshd'. We have q/k/v as [B, H, S, D].
            # Force matching dtype (all bf16 under autocast).
            common_dtype = torch.bfloat16 if torch.is_autocast_enabled() else q.dtype
            q_te = q.transpose(1, 2).contiguous().to(common_dtype)
            k_te = k.transpose(1, 2).contiguous().to(common_dtype)
            v_te = v.transpose(1, 2).contiguous().to(common_dtype)
            y = self._te_attn(q_te, k_te, v_te)  # [B, S, H*D]
            return self.c_proj(y)
        elif self.attention_impl == 'fp32_manual':
            orig_dtype = q.dtype
            qf = q.float()
            kf = k.float()
            vf = v.float()
            att = (qf @ kf.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            T_q = qf.size(-2)
            if attn_mask is None:
                causal = torch.ones(T_q, T_q, dtype=torch.bool, device=qf.device).tril()
                att = att.masked_fill(~causal, float('-inf'))
            else:
                att = att.masked_fill(~attn_mask, float('-inf'))
            att = torch.nn.functional.softmax(att, dim=-1)
            y = (att @ vf).to(orig_dtype)
        elif self.flash:
            if attn_mask is None:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True,
                )
            else:
                # attn_mask is a boolean [B, 1, T, T]: True = attend, False = mask.
                # SDPA expects is_causal=False when attn_mask is provided.
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False,
                )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attn_mask is None:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            else:
                att = att.masked_fill(~attn_mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Standard MLP with GELU (GPT-2 style)."""

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SwiGLUMLP(nn.Module):
    """SwiGLU FFN, matching cybertron's ffn_hidden_size convention.

    The cybertron config uses:
      ffn_hidden_size = 1536  (the hidden dim of each of the two gate projections)
    The output is: swish(gate(x)) * up(x), then projected down.
    """

    def __init__(self, config):
        super().__init__()
        ffn_hidden = config.ffn_hidden_size
        self.gate_proj = nn.Linear(config.n_embd, ffn_hidden, bias=config.bias)
        self.up_proj   = nn.Linear(config.n_embd, ffn_hidden, bias=config.bias)
        self.down_proj = nn.Linear(ffn_hidden, config.n_embd, bias=config.bias)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x):
        # fp32 silu*up then cast to bf16 — matches TE's fused SwiGLU kernel (single bf16
        # round at end). Pure `F.silu(g)*u` under autocast rounds silu output to bf16
        # before multiplication → loses 1 ULP on ~26% of positions.
        g = self.gate_proj(x)
        u = self.up_proj(x)
        h = (F.silu(g.float()) * u.float()).to(x.dtype)
        return self.dropout(self.down_proj(h))


class ExpertMLP(nn.Module):
    """Single routed expert: SwiGLU FFN (512→160→512 for MoE 198)."""

    def __init__(self, n_embd, hidden_size, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, hidden_size, bias=bias)
        self.up_proj   = nn.Linear(n_embd, hidden_size, bias=bias)
        self.down_proj = nn.Linear(hidden_size, n_embd, bias=bias)

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        h = (F.silu(g.float()) * u.float()).to(x.dtype)
        return self.down_proj(h)


class MoERouter(nn.Module):
    """Grouped sigmoid top-k router with expert score correction bias.

    Matches cybertron's router_scoring_func=sigmoid, n_group grouped routing,
    norm_topk_prob=True, and use_router_expert_score_correction.
    """

    def __init__(self, n_embd, num_experts, topk, n_group, topk_group,
                 norm_topk_prob, score_correction_coeff,
                 routing_type='group_limited_greedy'):
        super().__init__()
        assert num_experts % n_group == 0, "num_experts must be divisible by n_group"
        self.num_experts = num_experts
        self.topk = topk
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.score_correction_coeff = score_correction_coeff
        self.experts_per_group = num_experts // n_group
        self.routing_type = routing_type   # 'greedy' | 'group_limited_greedy'

        # Router linear (fp32 forward via moe_gating_fp32)
        self.linear = nn.Linear(n_embd, num_experts, bias=False)
        # Score correction bias: updated once per optim step via update_expert_bias(),
        # NOT via gradients. See Megatron router.py:444 and finalize_model_grads.py:270.
        self.register_buffer('e_score_correction_bias', torch.zeros(num_experts))
        # Load counts accumulated across grad_accum micro-steps; consumed + reset by
        # update_expert_bias() after the accumulation loop completes.
        self.register_buffer('local_tokens_per_expert',
                             torch.zeros(num_experts, dtype=torch.float32))

    def forward(self, x):  # x: [S, n_embd]
        S = x.shape[0]
        E = self.num_experts
        G = self.n_group
        K = self.topk

        # 1. Compute scores in fp32 (moe_gating_fp32=True). We MUST disable autocast here,
        # otherwise F.linear's output is downcast to bf16 regardless of .float() on inputs,
        # producing bf16-rounded logits that route 1-2% of tokens to wrong experts at
        # boundary cases. Matches cybertron's moe_gating_fp32=True path.
        with torch.amp.autocast('cuda', enabled=False):
            logits = F.linear(x.float(), self.linear.weight.float())  # [S, E], fp32
            scores = torch.sigmoid(logits)                            # [S, E], fp32
            # 2. Add correction bias for routing decision (NOT added to final weights)
            scores_biased = scores + self.e_score_correction_bias.float()

        # 3. Routing: flat top-K (cybertron routing_type='greedy') OR group-limited.
        if self.routing_type == 'greedy':
            # Flat top-K over all experts — matches cybertron's DeepSeekRoutingType.GREEDY
            topk_idx = scores_biased.topk(K, dim=-1).indices  # [S, K]
        else:
            # Group-limited top-K: sum top-(K//topk_group) per group → group score;
            # select topk_group groups; pick top-K among those groups' experts.
            epg = self.experts_per_group
            group_scores = scores_biased.view(S, G, epg).topk(K // self.topk_group, dim=-1).values.sum(dim=-1)
            group_idx = group_scores.topk(self.topk_group, dim=-1).indices  # [S, topk_group]
            group_mask = torch.zeros(S, G, dtype=scores_biased.dtype, device=x.device)
            group_mask.scatter_(1, group_idx, 1.0)
            score_mask = group_mask.unsqueeze(-1).expand(S, G, epg).reshape(S, E)
            scores_masked = scores_biased.masked_fill(score_mask == 0, float('-inf'))
            topk_idx = scores_masked.topk(K, dim=-1).indices  # [S, K]

        # 4. Final weights: ORIGINAL (unbiased) scores at selected positions. KEEP FP32
        # to match cybertron's path where weighted-sum of expert outputs is done in fp32.
        final_weights = scores.gather(1, topk_idx)  # [S, K], fp32
        if self.norm_topk_prob:
            final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-20)

        # 7. Accumulate token counts across grad_accum micro-steps. The actual bias
        #    update + all-reduce + reset happens in update_expert_bias(), called once
        #    per optim step from the training loop. This matches Megatron's behaviour:
        #    router.py:444-448 (accumulate) + finalize_model_grads.py:270-295 (update).
        if torch.is_grad_enabled():
            with torch.no_grad():
                # Keep counter in fp32 even if the module was .bfloat16()'d (mirrors
                # Megatron's _maintain_float32_expert_bias pattern).
                if self.local_tokens_per_expert.dtype != torch.float32:
                    self.local_tokens_per_expert.data = self.local_tokens_per_expert.data.float()
                self.local_tokens_per_expert.scatter_add_(
                    0, topk_idx.reshape(-1),
                    torch.ones(S * K, dtype=torch.float32, device=x.device)
                )

        # Return also raw sigmoid scores (pre-bias, pre-group-mask) for seq_aux balance loss
        return topk_idx, final_weights, scores  # [S, K], [S, K], [S, E]

    @torch.no_grad()
    def update_expert_bias(self):
        """Apply one aux-free bias update from the accumulated token counts.

        Must be called once per optim step, AFTER the grad_accum micro-steps finish
        and BEFORE optimizer.step(). Equivalent to Megatron's
        `_update_router_expert_bias` in finalize_model_grads.py.
        """
        import torch.distributed as dist
        # Keep both buffers in fp32 (mirrors Megatron's _maintain_float32_expert_bias).
        if self.local_tokens_per_expert.dtype != torch.float32:
            self.local_tokens_per_expert.data = self.local_tokens_per_expert.data.float()
        if self.e_score_correction_bias.dtype != torch.float32:
            self.e_score_correction_bias.data = self.e_score_correction_bias.data.float()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.local_tokens_per_expert, op=dist.ReduceOp.SUM)
        mean_load = self.local_tokens_per_expert.mean()
        self.e_score_correction_bias.add_(
            (mean_load - self.local_tokens_per_expert).sign()
            * self.score_correction_coeff
        )
        self.local_tokens_per_expert.zero_()


class MoEFFN(nn.Module):
    """MoE FFN layer: grouped routing + N experts + always-on shared expert.

    Matches cybertron's MoE structure: shared expert output + weighted sum of
    top-K routed expert outputs.

    Dispatch is a single `torch._grouped_mm` call per projection (gate/up/down),
    not a Python for-loop over experts. This removes the O(num_experts) dispatch
    overhead that crushed MFU on the naive impl (see perf notes in RUNBOOK).
    """

    def __init__(self, config):
        super().__init__()
        self.router = MoERouter(
            n_embd=config.n_embd,
            num_experts=config.num_experts,
            topk=config.moe_router_topk,
            n_group=config.moe_n_group,
            topk_group=config.moe_topk_group,
            norm_topk_prob=config.moe_norm_topk_prob,
            score_correction_coeff=config.moe_router_score_correction_coeff,
            routing_type=getattr(config, 'moe_routing_type', 'group_limited_greedy'),
        )
        E = config.num_experts
        C = config.n_embd
        H = config.moe_ffn_hidden_size
        # Stacked expert weights for grouped_mm. Total params == E × ExpertMLP.
        # Layout: grouped_mm(x [M,C], w [E,C,H], offs) -> [M,H].
        self.gate_weight = nn.Parameter(torch.empty(E, C, H))
        self.up_weight   = nn.Parameter(torch.empty(E, C, H))
        self.down_weight = nn.Parameter(torch.empty(E, H, C))
        nn.init.normal_(self.gate_weight, mean=0.0, std=config.init_std)
        nn.init.normal_(self.up_weight,   mean=0.0, std=config.init_std)
        nn.init.normal_(self.down_weight, mean=0.0, std=config.init_std)

        if config.moe_shared_expert_hidden_size is not None:
            self.shared_expert = ExpertMLP(
                config.n_embd, config.moe_shared_expert_hidden_size, bias=config.bias
            )
        else:
            self.shared_expert = None
        self.num_experts = config.num_experts
        self.topk = config.moe_router_topk
        self.seq_aux_alpha = getattr(config, 'seq_aux_balance_alpha', 0.0) or 0.0

    def forward(self, x):  # x: [B, T, C]
        B, T, C = x.shape
        x_flat = x.view(B * T, C)

        # Shared expert (always active, moe_shared_expert_overlap=False → sequential)
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)  # [B*T, C]
        else:
            shared_out = torch.zeros_like(x_flat)

        # Routed experts: grouped-GEMM dispatch (single CUDA op per projection)
        topk_idx, weights, raw_scores = self.router(x_flat)  # [B*T, K], [B*T, K], [B*T, E]
        S = x_flat.shape[0]
        K = self.topk
        E = self.num_experts

        # Optional TE fused path — matches cybertron's moe_permute_fusion=True exactly.
        import os as _os
        if _os.environ.get('NANO_TE_MOE', '0') == '1':
            import transformer_engine.pytorch as _te
            # Build routing_map [S, E] int32 and probs [S, E] fp32
            routing_map = torch.zeros(S, E, dtype=torch.int32, device=x.device)
            routing_map.scatter_(1, topk_idx, 1)
            probs = torch.zeros(S, E, dtype=torch.float32, device=x.device)
            probs.scatter_(1, topk_idx, weights.float())
            # Permute tokens
            permuted, _perm_probs, row_id_map = _te.moe_permute_with_probs(
                x_flat, probs, routing_map, num_out_tokens=S*K,
            )
            # Expert compute via grouped GEMM (gate/up fused)
            # Build per-expert fused weights [E, 2H, C] and [E, C, H] once per call —
            # TODO cache if perf matters. For now, reinstantiate.
            # Use te.GroupedLinear for experts (matches ref's fused grouped_gemm)
            H_ffn = self.gate_weight.shape[-1]
            if not hasattr(self, '_gl_fc1'):
                self._gl_fc1 = _te.GroupedLinear(
                    num_gemms=E, in_features=C, out_features=2*H_ffn, bias=False,
                    params_dtype=torch.float32, device=x.device,
                )
                self._gl_fc2 = _te.GroupedLinear(
                    num_gemms=E, in_features=H_ffn, out_features=C, bias=False,
                    params_dtype=torch.float32, device=x.device,
                )
                # Copy nano weights into per-expert weights (gate_weight [E,C,H], up_weight [E,C,H], down_weight [E,H,C])
                with torch.no_grad():
                    for e in range(E):
                        gate_e = self.gate_weight[e].T.contiguous()  # [H, C]
                        up_e = self.up_weight[e].T.contiguous()      # [H, C]
                        fc1_e = torch.cat([gate_e, up_e], dim=0)     # [2H, C]
                        getattr(self._gl_fc1, f'weight{e}').data.copy_(fc1_e.float())
                        getattr(self._gl_fc2, f'weight{e}').data.copy_(self.down_weight[e].T.contiguous().float())
            m_splits = routing_map.sum(dim=0).tolist()
            h12 = self._gl_fc1(permuted, m_splits=m_splits)
            gate, up = h12.chunk(2, dim=-1)
            h_act = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
            out_perm = self._gl_fc2(h_act, m_splits=m_splits)
            # Unpermute with fused probs-weighted merge
            routed_out = _te.moe_unpermute(
                out_perm, row_id_map, merging_probs=probs,
                restore_shape=x_flat.shape, map_type='mask',
            )
            # Fp32 sum then cast (matches nano's non-TE path)
            out = (shared_out.float() + routed_out.float()).to(x_flat.dtype).view(B, T, C)
            # Return per Block.forward expectation: (out, aux)
            if self.training and self.seq_aux_alpha > 0:
                # Skip aux for now when using TE path (only affects training)
                aux = x.new_zeros(())
            else:
                aux = x.new_zeros(())
            return out, aux

        # Expand tokens across topk → [S*K, C] and pair with flat (expert, weight)
        flat_tokens  = x_flat.unsqueeze(1).expand(S, K, C).reshape(S * K, C)
        flat_experts = topk_idx.reshape(-1)             # [S*K]
        # Keep weights in fp32 to match cybertron's weighted-sum precision
        flat_weights = weights.reshape(-1).float()      # [S*K], fp32

        # Sort so tokens for each expert are contiguous
        order          = flat_experts.argsort(stable=True)
        sorted_tokens  = flat_tokens[order].contiguous()
        sorted_weights = flat_weights[order]
        sorted_experts = flat_experts[order]

        # Per-expert counts
        counts = sorted_experts.bincount(minlength=E)                     # [E]
        offs   = counts.cumsum(0)                                          # [E] long
        orig_starts = torch.cat([offs.new_zeros(1), offs[:-1]])            # [E]
        # Each row's local offset within its expert bucket
        row_ids   = torch.arange(S * K, device=x.device, dtype=torch.int64)
        local_off = row_ids - orig_starts[sorted_experts]                  # [S*K]

        # Pad each bucket to max(counts) (aligned) for bmm — reliable autograd, no grouped_mm bugs
        M_per = int(counts.max().item())                                   # H2D sync, one per layer
        if M_per < 8:
            M_per = 8
        bucket = sorted_tokens.new_zeros(E, M_per, C)
        bucket[sorted_experts, local_off] = sorted_tokens
        # Three batched GEMMs: bucket [E, M_per, C] @ weight [E, C, H] → [E, M_per, H]
        gate = torch.bmm(bucket, self.gate_weight)
        up_  = torch.bmm(bucket, self.up_weight)
        # fp32 silu*up then cast — matches TE fused SwiGLU (single bf16 round)
        h = (F.silu(gate.float()) * up_.float()).to(gate.dtype)
        out_bucket = torch.bmm(h, self.down_weight)                        # [E, M_per, C]
        # Gather expert outputs in sorted (by expert) order, apply weights, then
        # scatter_add back to token positions. This matches Megatron's unpermute
        # path: `output.scatter_add_(0, sorted_indices, permuted_tokens * probs)`
        # (bf16-sum order differs from per-token K-wise sum, affects loss at ULP level).
        # Weighted sum in fp32 to match TE's moe_unpermute precision.
        out_flat = out_bucket[sorted_experts, local_off].float()           # [S*K, C] fp32
        out_flat = out_flat * sorted_weights.unsqueeze(1).float()          # fp32 * fp32
        sorted_token_ids = row_ids.div(K, rounding_mode='floor')[order]
        routed_out = torch.zeros(S, C, dtype=torch.float32, device=x.device)  # fp32 accumulator
        routed_out.scatter_add_(0, sorted_token_ids.unsqueeze(1).expand(-1, C), out_flat)
        # Cast to bf16 only at final combine
        out = (shared_out.float() + routed_out).to(x_flat.dtype).view(B, T, C)

        # Sequence-wise balance aux loss.
        # Exact match to cybertron_dots3.0_swa/cybertron/models/deepseek_v2/modules_deepseekv2.py:374-378:
        #   fii = fi.sum(0).div_(seq_len * K / E)   # [B, E] = count * E / (T*K)
        #   pii = pi.div(pi.sum(-1, keepdim=True) + 1e-20).mean(0)   # [B, E]  per-token normalized
        #   aux = (fii * pii).sum(dim=1).mean()
        if self.training and self.seq_aux_alpha > 0:
            E, K = self.num_experts, self.topk
            topk_b = topk_idx.view(B, T, K)                                   # [B, T, K]
            counts_bE = torch.zeros(B, E, device=x.device, dtype=torch.float32)
            src = torch.ones_like(topk_b, dtype=torch.float32)
            counts_bE.scatter_add_(1, topk_b.reshape(B, T * K), src.reshape(B, T * K))
            fii = counts_bE * (float(E) / (float(T) * float(K)))              # [B, E]
            pi = raw_scores.view(B, T, E).float()                             # [B, T, E]
            pii = (pi / (pi.sum(dim=-1, keepdim=True) + 1e-20)).mean(dim=1)   # [B, E]
            aux = (fii * pii).sum(dim=1).mean()                                # scalar
        else:
            aux = x.new_zeros(())
        return out, aux


class Block(nn.Module):

    def __init__(self, config, layer_idx=0):
        super().__init__()
        # Normalization
        if config.use_rmsnorm:
            self.ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
            self.ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        else:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        self.attn = CausalSelfAttention(config)

        # FFN: MoE or dense, based on moe_layer_freq
        use_moe = (
            config.use_moe
            and config.moe_layer_freq is not None
            and config.moe_layer_freq[layer_idx] == 1
        )
        if use_moe:
            self.mlp = MoEFFN(config)
        elif config.use_swiglu:
            self.mlp = SwiGLUMLP(config)
        else:
            self.mlp = MLP(config)
        self.is_moe = use_moe

    def forward(self, x, attn_mask=None, position_ids=None):
        # fp32 residual ablation: keep x in fp32 across blocks; each sublayer still
        # runs bf16 internally (under autocast), but the additive combine is fp32.
        # This removes the bf16 ULP truncation on every residual write that
        # otherwise compounds over L=9 layers × 7485 steps.
        fp32_residual = getattr(self.ln_1, '_fp32_residual', False)
        attn_out = self.attn(self.ln_1(x), attn_mask=attn_mask, position_ids=position_ids)
        if fp32_residual:
            x = x.float() + attn_out.float()
        else:
            x = x + attn_out
        mlp_in = self.ln_2(x)
        if self.is_moe:
            mlp_out, aux = self.mlp(mlp_in)
        else:
            mlp_out, aux = self.mlp(mlp_in), x.new_zeros(())
        if fp32_residual:
            x = x.float() + mlp_out.float()
        else:
            x = x + mlp_out
        return x, aux


@dataclass
class GPTConfig:
    """Configuration for both original GPT-2 and cybertron-aligned models."""
    block_size: int = 1024
    vocab_size: int = 50304   # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: better and faster

    # --- Cybertron-style options ---
    n_kv_head: Optional[int] = None  # None = MHA; set to < n_head for GQA
    kv_channels: Optional[int] = None  # per-head dimension; None → n_embd // n_head
    use_rope: bool = False            # Use RoPE instead of learned position embeddings
    rotary_base: int = 10000          # RoPE base frequency
    use_rmsnorm: bool = False         # Use RMSNorm instead of LayerNorm
    norm_eps: float = 1e-5            # Epsilon for normalization
    use_swiglu: bool = False          # Use SwiGLU FFN instead of standard GELU MLP
    ffn_hidden_size: Optional[int] = None  # SwiGLU hidden size; defaults to 4*n_embd if None
    tie_embeddings: bool = True       # Tie input and output embeddings (GPT-2 style)
    init_std: float = 0.02            # Weight initialization std
    qk_layernorm: bool = False        # Apply RMSNorm to Q and K per-head, before RoPE
    disable_scaled_init_method: bool = False  # If True, skip 1/sqrt(2*n_layer) scaling for residual projs

    # --- MoE options (DeepSeek aux-free grouped routing) ---
    use_moe: bool = False
    moe_layer_freq: Optional[list] = None  # e.g. [0,1,1,...,1]; None = all dense
    num_experts: int = 64              # routed experts per MoE layer
    moe_ffn_hidden_size: int = 128     # per-expert SwiGLU hidden size
    moe_router_topk: int = 2           # top-K experts selected per token
    moe_n_group: int = 1               # num groups for grouped routing
    moe_topk_group: int = 1            # num groups selected per token (currently only 1)
    moe_norm_topk_prob: bool = True    # normalize top-K scores to sum=1
    moe_router_score_correction_coeff: float = 0.001  # bias update step (aux-free load balance)
    moe_shared_expert_hidden_size: Optional[int] = None  # None = no shared expert
    moe_routing_type: str = 'group_limited_greedy'  # or 'greedy' (flat top-K over all experts)

    # --- Cybertron loss/attention extras (needed for scaling_moe_00196 alignment) ---
    eod_token_id: Optional[int] = None     # if set, loss at positions where target == this id is masked
    mask_loss_id: Optional[int] = None     # additional target id masked from loss (e.g. 160000 sentinel)
    seq_aux_balance_alpha: float = 0.0     # α for sequence-wise MoE balance aux loss; 0 = disabled
    use_eod_attn_mask: bool = False        # attention cannot cross EOD within a packed sequence
    attention_impl: str = 'sdpa'           # 'sdpa' | 'fp32_manual' | 'te' (TransformerEngine DotProductAttention, matches cybertron kernel)
    fp32_residual: bool = False            # keep residual stream in fp32; each block casts sublayer outputs
                                           # to fp32 before x + sublayer(x). Costs ~2x activation memory but
                                           # removes bf16 ULP noise per block that compounds over L layers × N steps.

    def __post_init__(self):
        if self.use_swiglu and self.ffn_hidden_size is None:
            # Default SwiGLU hidden size: 2/3 * 4 * n_embd, rounded to multiple of 64
            self.ffn_hidden_size = int(2 / 3 * 4 * self.n_embd)
            self.ffn_hidden_size = (self.ffn_hidden_size + 63) // 64 * 64
        if not self.use_swiglu:
            self.ffn_hidden_size = 4 * self.n_embd  # not used by MLP, but stored for reference


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        transformer_dict = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
        )
        # Propagate fp32_residual flag. Stored on ln_1 as a cheap per-block marker
        # readable from Block.forward without adding another signature arg.
        for _blk in transformer_dict['h']:
            _blk.ln_1._fp32_residual = getattr(config, 'fp32_residual', False)

        if config.use_rope:
            # No learned position embeddings when using RoPE
            transformer_dict['wpe'] = None
        else:
            transformer_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)

        if config.use_rmsnorm:
            transformer_dict['ln_f'] = RMSNorm(config.n_embd, eps=config.norm_eps)
        else:
            transformer_dict['ln_f'] = LayerNorm(config.n_embd, bias=config.bias)

        self.transformer = nn.ModuleDict(transformer_dict)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_embeddings:
            self.transformer.wte.weight = self.lm_head.weight

        # Init all weights
        self.apply(self._init_weights)
        # Apply special scaled init to residual projections, per GPT-2 paper.
        # cybertron sets disable_scaled_init_method=True, so skip this when requested.
        if not config.disable_scaled_init_method:
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight') or pn.endswith('down_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=config.init_std / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if self.config.use_rope:
                pass  # no wpe to subtract
            else:
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        # Cast embedding output to current autocast dtype so residual stream stays in bf16.
        # Without this, residual stream accumulates in fp32 (embed outputs fp32 + widest-rule adds)
        # which is MORE precision than ref's bf16 residual. When the fp32_residual
        # ablation is enabled (config.fp32_residual=True) we intentionally skip the
        # bf16 downcast so residual stays fp32 end-to-end.
        if torch.is_autocast_enabled() and not getattr(self.config, 'fp32_residual', False):
            tok_emb = tok_emb.to(torch.get_autocast_dtype('cuda'))

        if self.config.use_rope:
            x = self.transformer.drop(tok_emb)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)

        # Build EOD-aware attention mask + segment-local position_ids when enabled.
        # Matches Megatron's packed_seq_params / _apply_rotary_pos_emb_thd behaviour:
        # attention cannot cross EOD AND RoPE resets to position 0 at each EOD boundary.
        attn_mask = None
        position_ids = None
        eod_id = getattr(self.config, 'eod_token_id', None)
        if getattr(self.config, 'use_eod_attn_mask', False) and eod_id is not None:
            # segment_id[b, k] = number of EOD tokens strictly before position k
            # → positions 0..EOD_pos (inclusive) are in the same segment.
            is_eod = (idx == eod_id).to(torch.int32)
            seg = torch.nn.functional.pad(is_eod.cumsum(dim=1)[:, :-1], (1, 0), value=0)
            # same-segment mask & causal combined
            same_seg = seg.unsqueeze(2) == seg.unsqueeze(1)       # [B, T, T] bool
            causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))
            attn_mask = (same_seg & causal).unsqueeze(1)          # [B, 1, T, T]
            # Segment-local position ids: for each position k, offset is "start of
            # current segment". Since seg is monotonically non-decreasing across T,
            # prefix-max of (is_seg_start * arange) gives the per-position segment start.
            is_seg_start = torch.cat(
                [torch.ones(b, 1, dtype=torch.bool, device=device),
                 seg[:, 1:] != seg[:, :-1]],
                dim=1,
            )                                                     # [B, T] bool
            abs_pos = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
            seg_start = torch.where(is_seg_start, abs_pos, torch.zeros_like(abs_pos))
            seg_start = seg_start.cummax(dim=1).values            # [B, T] int
            position_ids = abs_pos - seg_start                    # [B, T] int, resets at EOD

        aux_sum = x.new_zeros(())
        n_moe_layers = 0
        for block in self.transformer.h:
            x, aux = block(x, attn_mask=attn_mask, position_ids=position_ids)
            if getattr(block, 'is_moe', False):
                aux_sum = aux_sum + aux
                n_moe_layers += 1
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Force lm_head in fp32. Ref `fp16_lm_cross_entropy: false` implies logits
            # should not be bf16 (bf16 has ~3 decimal digits precision on 152064-dim softmax).
            with torch.amp.autocast('cuda', enabled=False):
                logits = F.linear(x.float(), self.lm_head.weight.float())
            t_flat = targets.view(-1)
            # Mask positions where the INPUT token is EOD or mask_loss_id (matches
            # Megatron's `loss_mask[data == eod_token] = 0` semantics — mask is
            # keyed on INPUT at position i, NOT target. Previously nano masked
            # where target[i] == EOD, which is off-by-one (masks position before EOD
            # instead of EOD position itself). Fix: use idx (input) for masking.
            mask_ids = []
            if getattr(self.config, 'eod_token_id', None) is not None:
                mask_ids.append(self.config.eod_token_id)
            if getattr(self.config, 'mask_loss_id', None) is not None:
                mask_ids.append(self.config.mask_loss_id)
            if mask_ids:
                idx_flat = idx.view(-1)
                mask = torch.zeros_like(idx_flat, dtype=torch.bool)
                for mid in mask_ids:
                    mask = mask | (idx_flat == mid)
                t_flat = torch.where(mask, torch.full_like(t_flat, -1), t_flat)
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), t_flat, ignore_index=-1)
            # Sequence-wise MoE balance aux: stored as a tensor component of loss
            # (for backward) but kept separable so callers can log LM CE alone —
            # matches Megatron, which reports pure `lm loss` and routes aux through
            # MoEAuxLossAutoScaler independently.
            alpha = getattr(self.config, 'seq_aux_balance_alpha', 0.0) or 0.0
            if alpha > 0 and n_moe_layers > 0:
                aux_contrib = alpha * (aux_sum / n_moe_layers)
            else:
                aux_contrib = lm_loss.new_zeros(())
            loss = lm_loss + aux_contrib
            # Expose components on the module for training-loop logging. train.py
            # can subtract aux to log apples-to-apples with ref's TB `lm loss`.
            self.last_lm_loss = lm_loss.detach()
            self.last_aux_contrib = aux_contrib.detach()
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if not self.config.use_rope:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, device_type,
                             use_muon=False, muon_lr=None, muon_momentum=0.95, muon_beta2=0.95,
                             muon_weight_decay=None):
        """
        Returns either a stock AdamW (use_muon=False, the alignment baseline) or a
        MultiOptimizer holding both Muon and AdamW (use_muon=True).

        Param routing when use_muon=True:
          Muon  ← attention {q,k,v,c}_proj, shared_expert {gate,up,down}_proj,
                  MoE expert {gate,up,down}_weight (3D batched).
          AdamW ← embeddings (wte, lm_head), MoE router gate, all 1D params (norms),
                  any biases.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        if not use_muon:
            # Original alignment-baseline path: single AdamW with 2D-decay / 1D-nodecay split.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, eps=eps, **extra_args
            )
            print(f"using fused AdamW: {use_fused}")
            return optimizer

        # Muon path. Name-based routing (NOT dim-based: MoE 3D goes to Muon, embedding 2D
        # goes to AdamW). Within AdamW, 2D-decay vs 1D-nodecay matches the baseline path
        # so the only single-variable change vs baseline is which params get NorMuon.
        from muon import Muon, MultiOptimizer

        def _to_adamw(name: str, param) -> bool:
            return (
                name.endswith('wte.weight')
                or name.endswith('lm_head.weight')
                or '.router.linear' in name
                or param.dim() < 2
            )

        muon_names, muon_params = [], []
        adam_decay_names, adam_decay_params = [], []
        adam_nodecay_names, adam_nodecay_params = [], []

        for n, p in param_dict.items():
            if _to_adamw(n, p):
                if p.dim() >= 2:
                    adam_decay_names.append(n)
                    adam_decay_params.append(p)
                else:
                    adam_nodecay_names.append(n)
                    adam_nodecay_params.append(p)
            else:
                muon_names.append(n)
                muon_params.append(p)

        # Sanity: every param accounted for exactly once
        seen = set(adam_decay_names) | set(adam_nodecay_names) | set(muon_names)
        missing = set(param_dict.keys()) - seen
        if missing:
            raise RuntimeError(f"unrouted params: {sorted(missing)}")
        if any('.router.linear' in n for n in muon_names):
            raise RuntimeError("MoE router weight wrongly routed to Muon")

        muon_lr_eff = muon_lr if muon_lr is not None else (learning_rate * 33.0)
        muon_wd_eff = muon_weight_decay if muon_weight_decay is not None else weight_decay

        muon_total = sum(p.numel() for p in muon_params)
        adam_dec_total = sum(p.numel() for p in adam_decay_params)
        adam_nodec_total = sum(p.numel() for p in adam_nodecay_params)
        print(f"Muon: {len(muon_params)} tensors, {muon_total:,} params, lr={muon_lr_eff}")
        print(f"AdamW (decay):   {len(adam_decay_params)} tensors, {adam_dec_total:,} params")
        print(f"AdamW (nodecay): {len(adam_nodecay_params)} tensors, {adam_nodec_total:,} params")
        if len(muon_params) <= 4:
            print(f"  Muon param names: {muon_names}")
        if len(adam_decay_params) <= 4:
            print(f"  AdamW decay names: {adam_decay_names}")

        adam_groups = [
            {'params': adam_decay_params, 'weight_decay': weight_decay},
            {'params': adam_nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        adam_extra = dict(fused=True) if use_fused else dict()
        adamw = torch.optim.AdamW(
            adam_groups, lr=learning_rate, betas=betas, eps=eps, **adam_extra
        )
        muon = Muon(
            muon_params,
            lr=muon_lr_eff,
            momentum=muon_momentum,
            beta2=muon_beta2,
            weight_decay=muon_wd_eff,
        )
        return MultiOptimizer({'adamw': adamw, 'muon': muon})

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        N = self.get_num_params()
        cfg = self.config
        head_dim = cfg.kv_channels if cfg.kv_channels is not None else cfg.n_embd // cfg.n_head
        L, H, Q, T = cfg.n_layer, cfg.n_head, head_dim, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 989e12  # H100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
