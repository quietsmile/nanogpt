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
from .steps.orthogonalize import NewtonSchulz, PolarExpress


_ORTHO_TYPES = (NewtonSchulz, PolarExpress)


class Muon(Optimizer):
    """Pipeline-based Muon.

    `fused_param_lists` (optional): list of lists of params that share an input
    dim and should be treated as a single virtual matrix during orthogonalize
    (e.g. nano's split q_proj/k_proj/v_proj corresponds to Megatron's fused
    linear_qkv). For each list, momentum/weight-decay run per-param as usual,
    then grads are vstack'd, one orthogonalize call runs on the fused tensor,
    the result is split back, and the post-ortho steps (spectral scale) run
    per-param. All params in a list must share `shape[1]`.
    """

    def __init__(
        self,
        params,
        pipeline: list[OptimStep],
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        hook: Callable[[str, StepContext], None] | None = None,
        fused_param_lists: list[list[torch.Tensor]] | None = None,
        per_head_split: dict[int, int] | None = None,
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

        # Locate the orthogonalize step (used by fused path); None if pipeline has none.
        self._ortho_idx = next(
            (i for i, s in enumerate(pipeline) if isinstance(s, _ORTHO_TYPES)),
            None,
        )

        # Per-head split: {id(p) → head_dim}. For each such param, ortho is run
        # independently on each [head_dim, in_dim] chunk along dim 0 (matches
        # Megatron's `enable_qkv_per_head=True` for Q/K/V Muon path).
        self._per_head_split: dict[int, int] = dict(per_head_split or {})

        # Build {id(p): (group_idx, group_params)} for fast lookup during step.
        self._fused_map: dict[int, tuple[int, list[torch.Tensor]]] = {}
        self._fused_lists: list[list[torch.Tensor]] = list(fused_param_lists or [])
        for gi, plist in enumerate(self._fused_lists):
            if len(plist) < 2:
                continue
            in_dim = plist[0].shape[1]
            for p in plist:
                if p.shape[1] != in_dim:
                    raise ValueError(
                        f"fused params must share shape[1]; group {gi} got "
                        f"{[tuple(x.shape) for x in plist]}"
                    )
                self._fused_map[id(p)] = (gi, plist)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            standalone = []
            seen_fused: dict[int, list[torch.Tensor]] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                key = id(p)
                if key in self._fused_map:
                    gi, plist = self._fused_map[key]
                    if gi not in seen_fused:
                        seen_fused[gi] = plist
                else:
                    standalone.append(p)

            for p in standalone:
                self._step_one(p, group)
            for plist in seen_fused.values():
                self._step_fused(plist, group)

        return loss

    # ---------- single-param path ----------
    def _step_one(self, p: torch.Tensor, group: dict) -> None:
        if p.dim() not in (2, 3):
            raise ValueError(
                f"Muon requires 2D or 3D params; got {tuple(p.shape)}. "
                "Route 1D / scalar params to AdamW."
            )
        head_dim = self._per_head_split.get(id(p))
        if head_dim is None or self._ortho_idx is None:
            # Standard path: full pipeline on the whole param.
            ctx = StepContext(
                param=p, grad=p.grad, state=self.state[p], group=group,
            )
            for step in self.pipeline:
                step(ctx)
                if self.hook is not None:
                    self.hook(type(step).__name__, ctx)
            update = ctx.update if ctx.update is not None else ctx.grad.to(p.dtype)
            p.data.add_(update, alpha=-group["lr"] * ctx.lr_scale)
            return

        # Per-head path: pre-ortho per-param (single momentum buffer for the
        # full grad), then split grad into [head_dim, in_dim] chunks along
        # dim 0, run ortho independently on each, concat back, continue with
        # post-ortho per-chunk (so spectral scale is computed against
        # head_dim × in_dim, matching Megatron exactly).
        ctx = StepContext(
            param=p, grad=p.grad, state=self.state[p], group=group,
        )
        for step in self.pipeline[: self._ortho_idx]:
            step(ctx)
            if self.hook is not None:
                self.hook(type(step).__name__, ctx)
        if ctx.grad.dim() != 2:
            raise ValueError(
                f"per_head_split requires 2D param; got {tuple(p.shape)}"
            )
        out_dim = ctx.grad.shape[0]
        if out_dim % head_dim != 0:
            raise ValueError(
                f"out_dim {out_dim} not divisible by head_dim {head_dim}"
            )
        n_heads = out_dim // head_dim
        ortho_step = self.pipeline[self._ortho_idx]

        # Apply post-ortho stages to a synthetic ctx per head, then sum
        # (since it's an additive update accumulation per head with
        # independent ortho results).
        head_updates = []
        head_lr_scale = 1.0
        for h in range(n_heads):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            head_grad = ctx.grad[sl].clone()
            head_ctx = StepContext(
                param=p[sl], grad=head_grad, state={}, group=group,
            )
            ortho_step(head_ctx)
            if self.hook is not None:
                self.hook(type(ortho_step).__name__ + ":head", head_ctx)
            for step in self.pipeline[self._ortho_idx + 1:]:
                step(head_ctx)
                if self.hook is not None:
                    self.hook(type(step).__name__, head_ctx)
            head_updates.append(
                head_ctx.update if head_ctx.update is not None else head_ctx.grad.to(p.dtype)
            )
            head_lr_scale = head_ctx.lr_scale
        update = torch.cat(head_updates, dim=0)
        p.data.add_(update, alpha=-group["lr"] * head_lr_scale)

    # ---------- fused-group path ----------
    def _step_fused(self, plist: list[torch.Tensor], group: dict) -> None:
        if self._ortho_idx is None:
            # No orthogonalize in pipeline — fused fusion has no effect; fall back per-param.
            for p in plist:
                if p.grad is not None:
                    self._step_one(p, group)
            return

        # Pre-ortho stages run per-param (momentum buffer is per-param).
        ctxs = []
        for p in plist:
            if p.grad is None:
                continue
            if p.dim() not in (2, 3):
                raise ValueError(
                    f"fused Muon supports 2D or 3D params; got {tuple(p.shape)}"
                )
            ctx = StepContext(
                param=p, grad=p.grad, state=self.state[p], group=group,
            )
            for step in self.pipeline[: self._ortho_idx]:
                step(ctx)
                if self.hook is not None:
                    self.hook(type(step).__name__, ctx)
            ctxs.append(ctx)
        if not ctxs:
            return

        # Concat grads along the "out" dim (dim 0 for 2D, dim 1 for 3D batch-style).
        grads = [c.grad for c in ctxs]
        ndim = grads[0].dim()
        if any(g.dim() != ndim for g in grads):
            raise ValueError(f"fused grads must share ndim; got {[g.dim() for g in grads]}")
        cat_dim = 0 if ndim == 2 else 1
        out_dims = [g.shape[cat_dim] for g in grads]
        ref_shape = list(grads[0].shape)
        for g in grads:
            for d in range(ndim):
                if d == cat_dim:
                    continue
                if g.shape[d] != ref_shape[d]:
                    raise ValueError(
                        f"fused grads must share non-cat dims; got {[tuple(x.shape) for x in grads]}"
                    )
        fused_grad = torch.cat(grads, dim=cat_dim)
        ortho_step = self.pipeline[self._ortho_idx]
        fused_ctx = StepContext(
            param=ctxs[0].param, grad=fused_grad,
            state={}, group=group,
        )
        ortho_step(fused_ctx)
        if self.hook is not None:
            self.hook(type(ortho_step).__name__ + ":fused", fused_ctx)

        offsets = [0]
        for d in out_dims:
            offsets.append(offsets[-1] + d)
        for i, c in enumerate(ctxs):
            sl = [slice(None)] * ndim
            sl[cat_dim] = slice(offsets[i], offsets[i + 1])
            c.grad = fused_ctx.grad[tuple(sl)]

        # Post-ortho stages (spectral scale, etc.) run per-param.
        for c in ctxs:
            for step in self.pipeline[self._ortho_idx + 1:]:
                step(c)
                if self.hook is not None:
                    self.hook(type(step).__name__, c)
            update = c.update if c.update is not None else c.grad.to(c.param.dtype)
            c.param.data.add_(update, alpha=-group["lr"] * c.lr_scale)
