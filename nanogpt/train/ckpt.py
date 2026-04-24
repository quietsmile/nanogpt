"""Checkpoint save/load utilities.

Extracted from train.py. Handles single Optimizer + MultiOptimizer state_dicts
round-trip. Preserves RNG state (CPU + CUDA) for deterministic resume.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def build_checkpoint(
    model: torch.nn.Module,
    optimizer: Any,
    iter_num: int,
    best_val_loss: float,
    config: dict,
    extra: dict | None = None,
) -> dict:
    """Build checkpoint dict (serializable)."""
    ckpt = {
        "model": _strip_ddp_prefix(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
        "rng_state_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        ckpt["rng_state_cuda"] = torch.cuda.get_rng_state_all()
    if extra:
        ckpt.update(extra)
    return ckpt


def save_checkpoint(ckpt: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_rng(ckpt: dict) -> None:
    if "rng_state_cpu" in ckpt:
        torch.set_rng_state(ckpt["rng_state_cpu"])
    if "rng_state_cuda" in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng_state_cuda"])


def _strip_ddp_prefix(model):
    """Return underlying module of DDP wrapper (or model itself)."""
    return getattr(model, "module", model)
