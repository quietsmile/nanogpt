"""Core primitives — unchanged from v1.0 model.py (bitwise-preserved)."""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class ChunkedLinearCrossEntropy(torch.autograd.Function):
    """Memory-efficient fused linear + cross-entropy.

    At mb=4, the full logits tensor [mb*T, V] = [32768, 152064] fp32 is 20 GB;
    forward + backward doubles that. Chunking iterates over rows of the hidden
    tensor, computing logits + loss + backward grads per-chunk, then freeing
    intermediates. Peak extra memory = ~(3 × chunk × V × 4 bytes) ≈ 3.7 GB at
    chunk=2048, instead of 40 GB.

    Returns per-token mean loss over non-ignored positions (reduction='mean',
    ignore_index=-1) — mathematically identical to
    F.cross_entropy(x @ W.T, targets, ignore_index=-1) with reduction='mean'.
    """

    @staticmethod
    def forward(ctx, x, weight, targets, chunk_size=2048, ignore_index=-1):
        # x: [N, C] (caller should pass fp32 for accuracy; we don't cast)
        # weight: [V, C] fp32
        # targets: [N] long, with `ignore_index` for masked positions
        assert x.dtype == torch.float32 and weight.dtype == torch.float32, \
            f"ChunkedLinearCrossEntropy expects fp32 inputs; got {x.dtype}/{weight.dtype}"
        assert x.dim() == 2 and weight.dim() == 2 and x.shape[1] == weight.shape[1]
        N, C = x.shape
        V = weight.shape[0]

        total_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        total_tokens = torch.zeros((), device=x.device, dtype=torch.float32)
        grad_x = torch.zeros_like(x)
        grad_W = torch.zeros_like(weight)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x[start:end]          # [chunk, C] fp32
            y_chunk = targets[start:end]    # [chunk] long
            mask = (y_chunk != ignore_index)  # [chunk] bool

            logits = x_chunk @ weight.T      # [chunk, V] fp32
            log_probs = F.log_softmax(logits, dim=-1)
            del logits
            safe_y = y_chunk.clamp(min=0)
            nll = -log_probs.gather(1, safe_y.unsqueeze(1)).squeeze(1)
            nll = torch.where(mask, nll, torch.zeros_like(nll))
            total_loss = total_loss + nll.sum()
            total_tokens = total_tokens + mask.float().sum()

            # Backward gradients: d(nll_sum)/d(logits) = softmax - onehot at valid rows.
            softmax = log_probs.exp()
            del log_probs
            one_hot = torch.zeros_like(softmax)
            one_hot.scatter_(1, safe_y.unsqueeze(1), 1.0)
            dlogits = softmax - one_hot
            del softmax, one_hot
            dlogits = dlogits * mask.float().unsqueeze(1)

            # Accumulate gradients (unscaled — divide by total_tokens after the loop)
            grad_x[start:end] = dlogits @ weight
            grad_W.addmm_(dlogits.T, x_chunk)
            del dlogits

        mean_loss = total_loss / total_tokens
        grad_x.div_(total_tokens)
        grad_W.div_(total_tokens)

        ctx.save_for_backward(grad_x, grad_W)
        return mean_loss

    @staticmethod
    def backward(ctx, grad_out):
        grad_x, grad_W = ctx.saved_tensors
        return grad_x * grad_out, grad_W * grad_out, None, None, None


def linear_cross_entropy(x, weight, targets, chunk_size=2048, ignore_index=-1,
                         use_chunked=True):
    """Convenience wrapper. When chunking is disabled, falls back to standard
    F.linear + F.cross_entropy for numerical parity testing.
    """
    if use_chunked:
        return ChunkedLinearCrossEntropy.apply(x, weight, targets, chunk_size, ignore_index)
    logits = F.linear(x, weight)
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


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


