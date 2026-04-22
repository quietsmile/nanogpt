"""Non-invasive training dynamics monitoring for nanogpt scaling experiments.

Design goal: capture rich learning-dynamics metrics (optimizer health, residual
stream, MoE routing, logit stats, etc.) without touching the training numerical
path. Achieved via:
  * forward hooks installed on existing modules (MoERouter, Block, lm_head)
  * read-only traversal of named_parameters() and optimizer.state
  * env-var gate (NANOGPT_MONITOR=1) — when disabled, hooks are NOT registered
    and all API calls are no-ops, i.e. zero overhead.

Integration is 3 lines in train.py (import + create_monitor + monitor.step).
No edits to model.py.
"""
import os


def create_monitor(model, optimizer, out_dir, master=True, ddp=False):
    """Factory. Returns a real Monitor iff NANOGPT_MONITOR=1, else a NullMonitor.

    The NullMonitor has identical method surface but every call is a no-op,
    so integrating the monitor never changes the cost or numerics of a
    disabled run.
    """
    if os.environ.get('NANOGPT_MONITOR', '0') != '1':
        from .core import NullMonitor
        return NullMonitor()
    from .core import Monitor
    return Monitor(model, optimizer, out_dir=out_dir, master=master, ddp=ddp)
