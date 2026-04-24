"""Checkpoint save/load/RNG roundtrip tests."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanogpt.train.ckpt import build_checkpoint, load_checkpoint, restore_rng, save_checkpoint


def test_build_checkpoint_structure():
    m = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ckpt = build_checkpoint(m, opt, iter_num=7, best_val_loss=2.5, config={"x": 1})
    assert set(ckpt.keys()) >= {
        "model", "optimizer", "iter_num", "best_val_loss", "config", "rng_state_cpu"
    }
    assert ckpt["iter_num"] == 7
    assert ckpt["best_val_loss"] == 2.5
    assert ckpt["config"]["x"] == 1


def test_save_load_roundtrip(tmp_path):
    m = torch.nn.Linear(8, 8)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    m.weight.data.fill_(0.5)
    ckpt = build_checkpoint(m, opt, iter_num=3, best_val_loss=1.0, config={})
    save_checkpoint(ckpt, tmp_path / "ckpt.pt")
    restored = load_checkpoint(tmp_path / "ckpt.pt")
    m2 = torch.nn.Linear(8, 8)
    m2.load_state_dict(restored["model"])
    assert torch.equal(m2.weight.data, m.weight.data)
    assert restored["iter_num"] == 3


def test_restore_rng_reproduces_numbers():
    torch.manual_seed(0)
    _ = torch.randn(3)
    rng = torch.get_rng_state()
    a = torch.randn(4)
    restore_rng({"rng_state_cpu": rng})
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_extra_fields_included():
    m = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ckpt = build_checkpoint(m, opt, iter_num=0, best_val_loss=0.0, config={}, extra={"seq_pos": 42})
    assert ckpt["seq_pos"] == 42
