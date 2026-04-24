"""MultiOptimizer — hold Muon + AdamW behind a single Optimizer-like interface.

Same contract as v1 muon_megatron.MultiOptimizer / muon.MultiOptimizer (both
had the same implementation). Lifted unchanged.
"""
from __future__ import annotations

from torch.optim.optimizer import Optimizer


class MultiOptimizer:
    def __init__(self, optimizers: dict[str, Optimizer]):
        for name, opt in optimizers.items():
            if not isinstance(opt, Optimizer):
                raise TypeError(f"optimizer {name!r} is not a torch.optim.Optimizer")
        self.inner: dict[str, Optimizer] = dict(optimizers)

    @property
    def param_groups(self):
        return [g for opt in self.inner.values() for g in opt.param_groups]

    @property
    def state(self):
        merged: dict = {}
        for opt in self.inner.values():
            merged.update(opt.state)
        return merged

    def step(self, closure=None):
        loss = None
        if closure is not None:
            import torch
            with torch.enable_grad():
                loss = closure()
        for opt in self.inner.values():
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.inner.values():
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"_multi": True, **{name: opt.state_dict() for name, opt in self.inner.items()}}

    def load_state_dict(self, state_dict: dict):
        if not state_dict.get("_multi"):
            raise ValueError(
                "load_state_dict expected a MultiOptimizer-format dict; "
                f"got keys {list(state_dict.keys())}."
            )
        for name, opt in self.inner.items():
            if name not in state_dict:
                raise KeyError(f"missing optimizer {name!r}; have {list(state_dict.keys())}")
            opt.load_state_dict(state_dict[name])
