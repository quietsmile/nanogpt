"""Attention — bitwise-preserved split of v1.0 CausalSelfAttention."""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .primitives import RMSNorm, RotaryEmbedding


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


