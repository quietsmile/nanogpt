"""Scaling config YAML load + validation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SCALING_DIR = Path(__file__).resolve().parents[2] / "configs" / "scaling"


def _load_yaml(p):
    # Minimal YAML loader — avoid adding PyYAML dependency unless already present.
    try:
        import yaml  # noqa: F401
        with open(p) as f:
            return yaml.safe_load(f)
    except ImportError:
        pytest.skip("PyYAML not available")


@pytest.mark.parametrize("tier", ["tier1_dense_107", "tier2_moe_196",
                                   "tier3_dense_500m", "tier4_moe_1b"])
def test_scaling_config_loads(tier):
    p = SCALING_DIR / f"{tier}.yaml"
    assert p.exists(), p
    cfg = _load_yaml(p)
    assert cfg["name"] == tier
    assert cfg["family"] in ("dense", "moe")
    assert "arch" in cfg
    assert "train" in cfg
    assert "schedule" in cfg


@pytest.mark.parametrize("tier,expected_family,expected_moe", [
    ("tier1_dense_107",  "dense", False),
    ("tier2_moe_196",    "moe",   True),
    ("tier3_dense_500m", "dense", False),
    ("tier4_moe_1b",     "moe",   True),
])
def test_tier_family_matches_arch(tier, expected_family, expected_moe):
    p = SCALING_DIR / f"{tier}.yaml"
    cfg = _load_yaml(p)
    assert cfg["family"] == expected_family
    assert cfg["arch"]["use_moe"] == expected_moe


def test_tier_schedule_samples_monotonic():
    for tier in ["tier1_dense_107", "tier2_moe_196", "tier3_dense_500m", "tier4_moe_1b"]:
        cfg = _load_yaml(SCALING_DIR / f"{tier}.yaml")
        s = cfg["schedule"]
        assert s["warmup_samples"] <= s["constant_samples"] <= s["decay_end_samples"], \
            f"{tier}: non-monotonic schedule {s}"


def test_tier_moe_layer_freq_length_matches_n_layer():
    for tier in ["tier2_moe_196", "tier4_moe_1b"]:
        cfg = _load_yaml(SCALING_DIR / f"{tier}.yaml")
        n_layer = cfg["arch"]["n_layer"]
        freq = cfg["arch"]["moe_layer_freq"]
        assert len(freq) == n_layer, f"{tier}: len(freq)={len(freq)} != n_layer={n_layer}"


def test_tier_ordering_by_n_layer():
    """Tiers 1→4 should be monotonically non-decreasing in n_layer."""
    n_layers = []
    for tier in ["tier1_dense_107", "tier2_moe_196", "tier3_dense_500m", "tier4_moe_1b"]:
        cfg = _load_yaml(SCALING_DIR / f"{tier}.yaml")
        n_layers.append(cfg["arch"]["n_layer"])
    assert n_layers == sorted(n_layers), f"n_layer not monotonic: {n_layers}"
