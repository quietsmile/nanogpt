"""Hook registry — lightweight event bus for training-time observers.

Usage in train loop:
    hooks = Hooks()
    # (observer code registers via hooks.on('post_backward', fn))
    ...
    hooks.emit('post_backward', grad_norm=gn, iter=it, model=model)

Subscribers (monitor/probes.py, viz builders) live OUTSIDE train code and
register whatever events they care about. If nothing registers, emit is O(1)
no-op — zero cost when observation is off.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


class Hooks:
    """Map event_name → list of callables. Callables receive **kwargs."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register `fn` for `event`. Returns `fn` so it can be used as a decorator.

        Example:
            @hooks.on('post_backward')
            def log_grad(grad_norm, iter, **_):
                print(f'iter {iter}: {grad_norm:.4f}')
        """
        self._subs[event].append(fn)
        return fn

    def off(self, event: str, fn: Callable) -> None:
        """Remove a previously registered callback."""
        if event in self._subs:
            try:
                self._subs[event].remove(fn)
            except ValueError:
                pass

    def emit(self, event: str, **kwargs: Any) -> None:
        """Fire all callbacks for event. Exceptions in callbacks are swallowed
        (observation must never break training)."""
        subs = self._subs.get(event)
        if not subs:
            return
        for fn in subs:
            try:
                fn(**kwargs)
            except Exception as e:  # noqa: BLE001
                import sys
                print(f"[hooks] {event} callback {fn.__name__} raised: {e!r}",
                      file=sys.stderr)

    def count(self, event: str | None = None) -> int:
        """Count registered callbacks (total if event is None)."""
        if event is None:
            return sum(len(v) for v in self._subs.values())
        return len(self._subs.get(event, []))

    def clear(self) -> None:
        self._subs.clear()
