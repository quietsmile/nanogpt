"""Sweep /prodcpfs/user/lien/eval_results/pretrain_v4_lite/{NAME}/{ITER}/summary/last.csv,
extract V4L aggregate + key metric groups, write to one big TSV/CSV.

Output: reports/eval_results_table.csv  (one row per (NAME, ITER))
"""
from __future__ import annotations
import csv
import os
import sys
from pathlib import Path

ROOT = Path("/prodcpfs/user/lien/eval_results/pretrain_v4_lite")
OUT = Path("/home/claudeuser/nanogpt/reports/eval_results_table.csv")

KEY_METRICS = [
    "--- PreTrainV4_Lite ---",
    "PreTrainV4.1",
    "--- 英文知识 ---",
    "mmlu_5shot",
    "mmlu_pro_5shot",
    "mmlu_redux_5shot",
    "ARC-challenge_25shot",
    "openbookqa_5shot",
    "--- 中文知识 ---",
    "ceval-test_5shot",
    "cmmlu_5shot",
    "C3_5shot",
    "--- 推理能力 ---",
    "winogrande_5shot",
    "hellaswag_10shot",
    "PIQA_0shot",
    "--- 数学 ---",
    "math",
    "gsm8k",
    "agieval-zh-mathqa_3~5shot",
    "--- 代码 ---",
    "humaneval",
    "mbpp",
]


def parse_last_csv(path: Path) -> tuple[dict, int]:
    """Return ({metric: value}, n_real). n_real = #rows where value is not '-'."""
    out = {}
    n_real = 0
    n_total = 0
    if not path.exists():
        return out, 0
    with open(path) as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) < 2:
        return out, 0
    # last column is the model score
    for r in rows[1:]:
        if not r:
            continue
        ds = r[0].strip()
        try:
            val = r[-1].strip()
        except IndexError:
            continue
        if val and val != "-":
            n_total += 1
            try:
                num = float(val)
                n_real += 1
            except ValueError:
                num = None
            out[ds] = val
    return out, n_real


def main():
    if not ROOT.exists():
        print(f"missing {ROOT}", file=sys.stderr)
        return 1
    rows = []
    exps = sorted(ROOT.iterdir())
    print(f"scanning {len(exps)} experiments...", file=sys.stderr)
    for exp in exps:
        if not exp.is_dir():
            continue
        for iter_dir in sorted(exp.iterdir(), key=lambda p: p.name):
            if not iter_dir.is_dir():
                continue
            csv_path = iter_dir / "summary" / "last.csv"
            metrics, n_real = parse_last_csv(csv_path)
            if not metrics:
                continue
            row = {"name": exp.name, "iter": iter_dir.name, "n_real": n_real}
            for m in KEY_METRICS:
                row[m] = metrics.get(m, "")
            rows.append(row)
    print(f"  {len(rows)} runs with last.csv", file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "iter", "n_real"] + KEY_METRICS
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
