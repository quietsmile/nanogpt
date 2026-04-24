"""Pipeline base — compose Muon-family optimizers from small, unit-testable steps.

Each step receives a StepContext holding (param, grad, momentum_state, group-
defaults, intermediate tensors) and may mutate ctx.grad / ctx.update or add
diagnostic fields under ctx.meta[*]. After all steps run, Muon.step() applies
`p.data.add_(ctx.update, alpha=-ctx.lr_scale)`.

Hook integration: Muon's optional `hook` callable receives
`hook(step_name, ctx)` after every step runs — used by monitor/probes to
capture per-stage norms without touching optimizer internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class StepContext:
    """Mutable per-param state passed through the step pipeline.

    Fields:
        param       : nn.Parameter being updated (2D or 3D).
        grad        : working gradient tensor. Steps MAY modify in-place or reassign.
        state       : optimizer's per-param state dict (e.g. momentum_buffer).
        group       : optimizer param_group dict (hyperparameters).
        update      : final orthogonalized/scaled direction Muon will subtract.
                      Defaults to `grad` if no step populates it; SpectralScale
                      typically sets this.
        lr_scale    : multiplier on top of group['lr'] for the final update.
                      Steps like SpectralScale write here.
        meta        : dict for diagnostics (e.g. `meta['ns_converged']`, per-step
                      grad norms). Never affects training state; used only by
                      observers / hooks.
    """
    param: torch.Tensor
    grad: torch.Tensor
    state: dict
    group: dict
    update: torch.Tensor | None = None
    lr_scale: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)


class OptimStep:
    """Abstract base — one algorithm stage (WD / Momentum / Orth / Scale / ...).

    Subclasses implement `__call__(ctx: StepContext) -> None` which mutates
    ctx in-place.

    Subclasses should be stateless (tiny config-only) so they can be shared
    across all params in a group. Per-param state lives on ctx.state.
    """

    def __call__(self, ctx: StepContext) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        body = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
        return f"{type(self).__name__}({body})"
