"""nanogpt v2 — bitwise-aligned refactor of v1.0.0.

Entry points:
    nanogpt.optim     Muon / AdamW / MultiOptimizer
    nanogpt.model     GPT + MoEFFN (bucket path)
    nanogpt.train     training loop + DDP + ckpt
    nanogpt.monitor   observer hooks (viz decoupled)

See docs/REFACTOR.md for v1→v2 migration.
"""
__version__ = "2.0.0-dev"
