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
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


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
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def _rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, seq_len=None):
        if seq_len is None:
            seq_len = q.shape[-2]
        if seq_len > self.cos_cached.shape[-2]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
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

    def forward(self, x):
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

            q, k = self.rotary_emb(q, k, seq_len=T)

            # Expand KV heads for GQA
            if self.n_rep > 1:
                k = k.unsqueeze(2).expand(B, self.n_kv_head, self.n_rep, T, self.head_dim)
                k = k.reshape(B, self.n_head, T, self.head_dim)
                v = v.unsqueeze(2).expand(B, self.n_kv_head, self.n_rep, T, self.head_dim)
                v = v.reshape(B, self.n_head, T, self.head_dim)
        else:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
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
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class ExpertMLP(nn.Module):
    """Single routed expert: SwiGLU FFN (512→160→512 for MoE 198)."""

    def __init__(self, n_embd, hidden_size, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, hidden_size, bias=bias)
        self.up_proj   = nn.Linear(n_embd, hidden_size, bias=bias)
        self.down_proj = nn.Linear(hidden_size, n_embd, bias=bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoERouter(nn.Module):
    """Grouped sigmoid top-k router with expert score correction bias.

    Matches cybertron's router_scoring_func=sigmoid, n_group grouped routing,
    norm_topk_prob=True, and use_router_expert_score_correction.
    """

    def __init__(self, n_embd, num_experts, topk, n_group, topk_group,
                 norm_topk_prob, score_correction_coeff):
        super().__init__()
        assert num_experts % n_group == 0, "num_experts must be divisible by n_group"
        self.num_experts = num_experts
        self.topk = topk
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.score_correction_coeff = score_correction_coeff
        self.experts_per_group = num_experts // n_group

        # Router linear (fp32 forward via moe_gating_fp32)
        self.linear = nn.Linear(n_embd, num_experts, bias=False)
        # Score correction bias: updated each step, NOT via gradients
        self.register_buffer('e_score_correction_bias', torch.zeros(num_experts))

    def forward(self, x):  # x: [S, n_embd]
        S = x.shape[0]
        E = self.num_experts
        G = self.n_group
        K = self.topk

        # 1. Compute scores in fp32 (moe_gating_fp32=True)
        logits = self.linear(x.float())             # [S, E], fp32
        scores = torch.sigmoid(logits).to(x.dtype)  # [S, E], bf16

        # 2. Add correction bias for routing decision (NOT added to final weights)
        scores_biased = scores + self.e_score_correction_bias

        # 3. Pick top-1 group: max score per group
        epg = self.experts_per_group
        group_scores = scores_biased.view(S, G, epg).max(dim=-1).values  # [S, G]
        _, selected_group = group_scores.topk(self.topk_group, dim=-1)   # [S, topk_group=1]

        # 4. Mask non-selected groups
        group_mask = torch.zeros(S, G, dtype=scores_biased.dtype, device=x.device)
        group_mask.scatter_(1, selected_group, 1.0)  # [S, G]
        expert_mask = group_mask.unsqueeze(-1).expand(S, G, epg).reshape(S, E)
        scores_masked = scores_biased.masked_fill(expert_mask == 0, float('-inf'))

        # 5. Top-K within the selected group
        topk_idx = scores_masked.topk(K, dim=-1).indices  # [S, K]

        # 6. Final weights: ORIGINAL (unbiased) scores at selected positions
        final_weights = scores.gather(1, topk_idx)  # [S, K]
        if self.norm_topk_prob:
            final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-20)

        # 7. Update correction bias during training (manual, not via gradient)
        if self.training:
            with torch.no_grad():
                tokens_per_expert = torch.zeros(E, dtype=torch.float32, device=x.device)
                tokens_per_expert.scatter_add_(
                    0, topk_idx.reshape(-1),
                    torch.ones(S * K, dtype=torch.float32, device=x.device)
                )
                mean_load = tokens_per_expert.mean()
                self.e_score_correction_bias.add_(
                    (mean_load - tokens_per_expert).sign().to(self.e_score_correction_bias.dtype)
                    * self.score_correction_coeff
                )

        return topk_idx, final_weights  # [S, K], [S, K]


class MoEFFN(nn.Module):
    """MoE FFN layer: grouped routing + N experts + always-on shared expert.

    Matches cybertron's MoE structure: shared expert output + weighted sum of
    top-K routed expert outputs.
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
        )
        self.experts = nn.ModuleList([
            ExpertMLP(config.n_embd, config.moe_ffn_hidden_size, bias=config.bias)
            for _ in range(config.num_experts)
        ])
        if config.moe_shared_expert_hidden_size is not None:
            self.shared_expert = ExpertMLP(
                config.n_embd, config.moe_shared_expert_hidden_size, bias=config.bias
            )
        else:
            self.shared_expert = None
        self.num_experts = config.num_experts
        self.topk = config.moe_router_topk

    def forward(self, x):  # x: [B, T, C]
        B, T, C = x.shape
        x_flat = x.view(B * T, C)

        # Shared expert (always active, moe_shared_expert_overlap=False → sequential)
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)  # [B*T, C]
        else:
            shared_out = torch.zeros_like(x_flat)

        # Routed experts: dispatch-and-combine
        topk_idx, weights = self.router(x_flat)  # [B*T, K], [B*T, K]
        routed_out = torch.zeros_like(x_flat)

        for k in range(self.topk):           # K=8 slots
            idx_k = topk_idx[:, k]           # [B*T] expert index per token for this slot
            w_k   = weights[:, k, None]      # [B*T, 1] weight per token for this slot
            for e in range(self.num_experts):  # 144 experts
                sel = (idx_k == e)
                if not sel.any():
                    continue
                routed_out[sel] = routed_out[sel] + w_k[sel] * self.experts[e](x_flat[sel])

        return (shared_out + routed_out).view(B, T, C)


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

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


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

        if self.config.use_rope:
            x = self.transformer.drop(tok_emb)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 2D tensors: weight matrices and embeddings → weight decay
        # 1D tensors: biases, norms → no weight decay
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
