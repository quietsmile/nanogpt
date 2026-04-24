"""nanogpt.monitor — observer hooks + probes + writers.

Design contract:
  - Hooks are a DAG-like event bus: train.loop emits events, probes subscribe.
  - Model / optim code NEVER imports monitor. Monitor is a pure observer.
  - Viz (dashboard) only reads reports/ — never imports nanogpt.* at runtime.

The legacy monitor/ package at repo root is the current implementation.
This package hosts the Hook registry and upcoming probes migration.
"""
from .registry import Hooks

__all__ = ["Hooks"]
