"""Unified Muon optimizer — reads a pipeline (list of OptimStep).

Muon algorithm = series of stages:
    1. (optional) cache raw grad   (for CautiousWD)
    2. WeightDecay (decoupled or coupled)
    3. Momentum (with optional Nesterov)
    4. (optional) PerParamNormalize
    5. Orthogonalization (NewtonSchulz or PolarExpress)
    6. SpectralScale / ShapeScale / UnitRMSScale

Each stage is an OptimStep instance. The pipeline is a plain list:
    pipeline = [DecoupledWD(), Momentum(beta=0.95, nesterov=True),
                NewtonSchulz(coefs='quintic', steps=5), SpectralScale(rms=0.2)]
    opt = Muon(params, pipeline=pipeline, lr=1.2e-3)

Swap algorithm parts by editing the list; no new Optimizer class needed.

Per-group hyperparameters (defaults via `defaults`, override per group):
    lr, weight_decay

Steps read from ctx.group['lr'] / ctx.group['weight_decay'] — stateless steps
can be shared across groups. Per-param state (momentum_buffer, second_moment)
lives on optimizer.state[p].

Hooks: pass `hook=callable(step_name, ctx)` to observe per-stage tensors for
monitor / viz. Zero overhead if unset.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch.optim.optimizer import Optimizer

from .pipeline import OptimStep, StepContext


class Muon(Optimizer):
    def __init__(
        self,
        params,
        pipeline: list[OptimStep],
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        hook: Callable[[str, StepContext], None] | None = None,
    ):
        if not pipeline:
            raise ValueError("pipeline must contain at least one OptimStep")
        for step in pipeline:
            if not isinstance(step, OptimStep):
                raise TypeError(f"pipeline entry {step!r} is not an OptimStep")
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.pipeline = pipeline
        self.hook = hook

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() not in (2, 3):
                    raise ValueError(
                        f"Muon requires 2D or 3D params; got {tuple(p.shape)}. "
                        "Route 1D / scalar params to AdamW."
                    )
                ctx = StepContext(
                    param=p,
                    grad=p.grad,
                    state=self.state[p],
                    group=group,
                )
                for step in self.pipeline:
                    step(ctx)
                    if self.hook is not None:
                        self.hook(type(step).__name__, ctx)
                # Final update: requires ctx.update (set by Scale) or fall back to ctx.grad.
                if ctx.update is None:
                    # No Scale step present — treat ctx.grad as the update, lr_scale=1.
                    update = ctx.grad.to(p.dtype)
                else:
                    update = ctx.update
                p.data.add_(update, alpha=-group["lr"] * ctx.lr_scale)

        return loss
