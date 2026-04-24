"""LR schedules — pure functions, unit-testable.

Extracted bitwise from train.py `get_lr`. Two modes:
  cosine    classic cosine decay with iter-based warmup (nanoGPT style)
  wsd-exp   warmup / stable / exp-decay, sample-based (Cybertron style)

Each returns learning rate given iter + config. No global state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CosineSchedule:
    learning_rate: float
    min_lr: float
    warmup_iters: int
    lr_decay_iters: int

    def __call__(self, it: int) -> float:
        if it < self.warmup_iters:
            return self.learning_rate * (it + 1) / (self.warmup_iters + 1)
        if it > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)


@dataclass
class WSDExpSchedule:
    """Cybertron-style warmup → constant → exp-decay. Sample-based.

    Phase 1: samples in [0, warmup_samples]          linear up from 0 to lr
    Phase 2: samples in [warmup_samples, constant]   constant at lr
    Phase 3: samples in [constant, decay_end]        exp decay to min_lr

    `effective_gbs` maps iter → consumed_samples (= (iter+1) * gbs).
    """
    learning_rate: float
    min_lr: float
    warmup_samples: int
    constant_samples: int
    decay_end_samples: int
    effective_gbs: int

    def __call__(self, it: int) -> float:
        consumed = (it + 1) * self.effective_gbs

        if consumed <= self.warmup_samples:
            if self.warmup_samples == 0:
                return self.learning_rate
            return self.learning_rate * consumed / self.warmup_samples
        if consumed < self.constant_samples:
            return self.learning_rate
        if consumed >= self.decay_end_samples:
            return self.min_lr
        decay_range = self.decay_end_samples - self.constant_samples
        progress = (consumed - self.constant_samples) / decay_range
        min_lr_ratio = max(1e-8 / self.learning_rate, self.min_lr / self.learning_rate)
        ratio = 0.5 ** (-progress * math.log2(min_lr_ratio))
        return ratio * self.learning_rate


def lr_for_iter(
    it: int,
    *,
    lr_decay_style: str,
    learning_rate: float,
    min_lr: float,
    # cosine
    warmup_iters: int = 0,
    lr_decay_iters: int = 0,
    # wsd-exp
    warmup_samples: int = 0,
    constant_samples: int = 0,
    decay_end_samples: int = 0,
    effective_gbs: int = 1,
) -> float:
    """Functional entry point — pick schedule by name, compute lr."""
    if lr_decay_style == "cosine":
        return CosineSchedule(
            learning_rate=learning_rate, min_lr=min_lr,
            warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters,
        )(it)
    if lr_decay_style == "wsd-exp":
        return WSDExpSchedule(
            learning_rate=learning_rate, min_lr=min_lr,
            warmup_samples=warmup_samples,
            constant_samples=constant_samples,
            decay_end_samples=decay_end_samples,
            effective_gbs=effective_gbs,
        )(it)
    raise ValueError(f"unknown lr_decay_style {lr_decay_style!r}")
