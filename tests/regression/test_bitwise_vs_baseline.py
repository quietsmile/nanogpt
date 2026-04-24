"""Regression: compare live PAI train_log.jsonl to pinned baseline YAML bitwise.

Run via `make bitwise` — requires a completed PAI training run's train_log.jsonl
whose path is provided via NANOGPT_REGRESSION_TRAIN_LOG env, or the default
CPFS path for v1.0.0 baseline reruns.

Marked `bitwise` so it's opt-in (doesn't run on every `make test`).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = ROOT / "tests" / "regression" / "baselines"


def _load_yaml(p):
    try:
        import yaml
        with open(p) as f:
            return yaml.safe_load(f)
    except ImportError:
        pytest.skip("PyYAML not available")


def _load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _find_baseline(name):
    p = BASELINES_DIR / f"{name}.yaml"
    if not p.exists():
        pytest.skip(f"baseline {name}.yaml not found")
    return p


@pytest.mark.bitwise
def test_v1_moe196_det1337_50iter():
    """Rerun with refactor-v2 code must match v1.0 baseline bitwise.

    Provide train_log.jsonl path via env NANOGPT_REGRESSION_TRAIN_LOG.
    """
    log_path = os.environ.get("NANOGPT_REGRESSION_TRAIN_LOG")
    if not log_path:
        pytest.skip("set NANOGPT_REGRESSION_TRAIN_LOG to the jsonl path")
    live = _load_jsonl(log_path)
    assert live, f"empty log: {log_path}"

    base = _load_yaml(_find_baseline("v1.0_moe196_det1337_50iter"))
    base_iters = {d["iter"]: d for d in base["iters"]}

    mismatches = []
    for rec in live:
        it = rec["iter"]
        if it not in base_iters:
            continue
        b = base_iters[it]
        if rec["loss"] != b["loss"]:
            mismatches.append((it, "loss", rec["loss"], b["loss"]))
        if rec["grad_norm"] != b["grad_norm"]:
            mismatches.append((it, "grad_norm", rec["grad_norm"], b["grad_norm"]))
    if mismatches:
        head = mismatches[:5]
        print("\nmismatches (first 5):")
        for it, field, got, exp in head:
            print(f"  iter {it} {field}: got {got} expected {exp} (Δ={got-exp:+.2e})")
    assert not mismatches, f"{len(mismatches)} bitwise diffs from baseline"


@pytest.mark.bitwise
def test_v1_moe196_muon_megatron_det1337_50iter():
    log_path = os.environ.get("NANOGPT_REGRESSION_MUON_TRAIN_LOG")
    if not log_path:
        pytest.skip("set NANOGPT_REGRESSION_MUON_TRAIN_LOG")
    live = _load_jsonl(log_path)
    base = _load_yaml(_find_baseline("v1.0_moe196_muon_megatron_det1337_50iter"))
    base_iters = {d["iter"]: d for d in base["iters"]}
    for rec in live:
        if rec["iter"] not in base_iters:
            continue
        b = base_iters[rec["iter"]]
        assert rec["loss"] == b["loss"], f"iter {rec['iter']} loss mismatch"
        assert rec["grad_norm"] == b["grad_norm"], f"iter {rec['iter']} grad_norm mismatch"


def test_baselines_exist():
    """Sanity: both canonical baselines committed to repo."""
    for name in ["v1.0_moe196_det1337_50iter", "v1.0_moe196_muon_megatron_det1337_50iter"]:
        p = BASELINES_DIR / f"{name}.yaml"
        assert p.exists(), f"{name}.yaml missing"
        cfg = _load_yaml(p)
        assert len(cfg["iters"]) == 51, f"{name}: expected 51 iters, got {len(cfg['iters'])}"
