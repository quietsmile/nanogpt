"""Hook registry tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanogpt.monitor import Hooks


def test_emit_no_subscribers_is_noop():
    h = Hooks()
    h.emit("never_fired", x=1, y=2)  # should not raise


def test_on_fires_callback():
    h = Hooks()
    received = []
    h.on("x", lambda v: received.append(v))
    h.emit("x", v=42)
    assert received == [42]


def test_multiple_callbacks_in_order():
    h = Hooks()
    order = []
    h.on("x", lambda: order.append(1))
    h.on("x", lambda: order.append(2))
    h.on("x", lambda: order.append(3))
    h.emit("x")
    assert order == [1, 2, 3]


def test_exception_in_callback_does_not_stop_training():
    h = Hooks()
    ran = []
    def bad(**_):
        raise RuntimeError("oops")
    def good(**_):
        ran.append(True)
    h.on("x", bad)
    h.on("x", good)
    h.emit("x")  # should swallow bad's error
    assert ran == [True]


def test_off_removes_callback():
    h = Hooks()
    log = []
    fn = lambda: log.append(1)
    h.on("x", fn)
    h.off("x", fn)
    h.emit("x")
    assert log == []


def test_count():
    h = Hooks()
    h.on("a", lambda: None)
    h.on("a", lambda: None)
    h.on("b", lambda: None)
    assert h.count("a") == 2
    assert h.count("b") == 1
    assert h.count() == 3


def test_on_returns_function():
    """on() returns the registered fn so code can keep a handle."""
    h = Hooks()
    def my_handler(val):
        return val * 2
    ret = h.on("x", my_handler)
    assert ret is my_handler  # for chainability
    assert h.count("x") == 1


def test_decoupling_viz_imports():
    """dashboard / viz must NEVER import nanogpt.{model,optim,train,monitor}."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    dashboard = root / "dashboard"
    if not dashboard.exists():
        return
    for p in dashboard.rglob("*.py"):
        text = p.read_text()
        for forbidden in ["from nanogpt.model", "from nanogpt.train",
                          "from nanogpt.optim", "import nanogpt.model",
                          "import nanogpt.train", "import nanogpt.optim"]:
            assert forbidden not in text, \
                f"{p.name} imports {forbidden} — viz must stay decoupled"
