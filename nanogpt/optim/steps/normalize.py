"""Per-param normalization step — NorMuon variant (arxiv 2510.05491).

After momentum, before orth: divide grad by its elementwise 2nd-moment RMS
(EMA). Stabilizes updates for ill-conditioned gradients.

State: 'second_moment' fp32 tensor, EMA with beta2.
"""
from __future__ import annotations

import torch

from ..pipeline import OptimStep, StepContext


class PerParamNormalize(OptimStep):
    """NorMuon-style per-element normalization by RMS of 2nd moment EMA.

    Args:
        beta2   : 2nd-moment EMA coefficient
        eps     : added to denominator for stability
    """

    def __init__(self, beta2: float = 0.95, eps: float = 1e-8):
        self.beta2 = beta2
        self.eps = eps

    def __call__(self, ctx: StepContext) -> None:
        if "second_moment" not in ctx.state:
            ctx.state["second_moment"] = torch.zeros_like(ctx.grad, dtype=torch.float32)
        sq = ctx.state["second_moment"]
        g_fp32 = ctx.grad.to(torch.float32)
        sq.mul_(self.beta2).addcmul_(g_fp32, g_fp32, value=1.0 - self.beta2)
        ctx.grad = g_fp32 / (sq.sqrt() + self.eps)
        ctx.meta["second_moment_norm"] = float(sq.norm().item())
