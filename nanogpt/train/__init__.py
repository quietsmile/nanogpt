"""nanogpt.train — training subsystem.

Modules:
  schedule   LR schedules (cosine, wsd-exp) — pure functions, easy to unit test
  ckpt       save/load checkpoint with multi-opt state round-trip
  config     TrainConfig dataclass (replaces configurator.py globals hack)

The main training loop is still in repo-root train.py (for v2.0 compat).
A future nanogpt.train.loop will consolidate it.
"""
from .schedule import lr_for_iter

__all__ = ["lr_for_iter"]
