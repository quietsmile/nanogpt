"""Block — bitwise-preserved split of v1.0."""
import torch
import torch.nn as nn
from .primitives import RMSNorm, LayerNorm
from .attention import CausalSelfAttention
from .mlp import MLP, SwiGLUMLP, MoEFFN


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


