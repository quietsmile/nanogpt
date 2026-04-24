"""Momentum accumulation steps.

Momentum   exp_avg ← β * exp_avg + grad           (no (1-β), Megatron style)
           Optionally Nesterov:  grad ← grad + β * exp_avg
"""
from __future__ import annotations

import torch

from ..pipeline import OptimStep, StepContext


class Momentum(OptimStep):
    """SGD-style momentum accumulation.

    Uses a fp32 `momentum_buffer` state per-param. Follows Megatron's
    convention of NO (1-β) rescaling on incoming grad — exp_avg grows linearly
    until orthogonalization balances it.

    Args:
        beta      : momentum coefficient (default 0.95 — Muon standard)
        nesterov  : if True, effective grad = grad + beta * exp_avg (after update)
    """

    def __init__(self, beta: float = 0.95, nesterov: bool = True):
        self.beta = beta
        self.nesterov = nesterov

    def __call__(self, ctx: StepContext) -> None:
        if "momentum_buffer" not in ctx.state:
            ctx.state["momentum_buffer"] = torch.zeros_like(ctx.grad, dtype=torch.float32)
        exp_avg = ctx.state["momentum_buffer"]

        grad_fp32 = ctx.grad.detach().to(torch.float32, copy=True)
        exp_avg.mul_(self.beta).add_(grad_fp32)

        if self.nesterov:
            ctx.grad = grad_fp32.add(exp_avg, alpha=self.beta)
        else:
            ctx.grad = exp_avg
        ctx.meta["momentum_buffer_norm"] = float(exp_avg.norm().item())
