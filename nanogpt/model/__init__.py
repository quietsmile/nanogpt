"""nanogpt v2 model package — bitwise-preserved split of v1.0 model.py."""
from .attention import CausalSelfAttention
from .block import Block
from .gpt import GPT, GPTConfig
from .mlp import MLP, MoEFFN, MoERouter, SwiGLUMLP
from .primitives import (
    ChunkedLinearCrossEntropy,
    LayerNorm,
    RMSNorm,
    RotaryEmbedding,
    linear_cross_entropy,
)

__all__ = [
    "GPT", "GPTConfig", "Block",
    "CausalSelfAttention",
    "MLP", "SwiGLUMLP", "MoERouter", "MoEFFN",
    "RMSNorm", "LayerNorm", "RotaryEmbedding",
    "ChunkedLinearCrossEntropy", "linear_cross_entropy",
]
