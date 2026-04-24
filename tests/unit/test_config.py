"""TrainConfig dataclass tests."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanogpt.train.config import TrainConfig


def test_defaults_validate():
    cfg = TrainConfig()
    cfg.validate()  # must not raise


def test_invalid_muon_impl_rejected():
    cfg = TrainConfig(use_muon=True, muon_impl="qqq")
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_lr_decay_rejected():
    cfg = TrainConfig(lr_decay_style="exponential")
    with pytest.raises(ValueError):
        cfg.validate()


def test_wsd_exp_bad_boundary_rejected():
    cfg = TrainConfig(lr_decay_style="wsd-exp",
                      warmup_samples=1000,
                      constant_samples=2000,
                      decay_end_samples=1500)  # decay_end < constant
    with pytest.raises(ValueError):
        cfg.validate()


def test_moe_layer_freq_length_mismatch_rejected():
    cfg = TrainConfig(use_moe=True, n_layer=9, moe_layer_freq=[0, 1, 1])
    with pytest.raises(ValueError):
        cfg.validate()


def test_from_python_file(tmp_path):
    p = tmp_path / "cfg.py"
    p.write_text("out_dir = 'test_out'\nlearning_rate = 1e-4\nn_layer = 6\n")
    cfg = TrainConfig.from_python_file(p)
    assert cfg.out_dir == "test_out"
    assert cfg.learning_rate == 1e-4
    assert cfg.n_layer == 6


def test_from_python_file_ignores_extraneous(tmp_path):
    p = tmp_path / "cfg.py"
    p.write_text("learning_rate = 2e-4\nnot_a_field = 123\nimport_stuff = 'foo'\n")
    cfg = TrainConfig.from_python_file(p)
    assert cfg.learning_rate == 2e-4
    assert not hasattr(cfg, "not_a_field")
