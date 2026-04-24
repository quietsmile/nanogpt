"""Convert PAI Megatron TB tfevents → nano-style run JSON for dashboard."""
import argparse, json, os, glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb-dir", required=True, help="dir containing tfevents")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ref-tb-dir", default="/prodcpfs/user/yuchen/scaling_exp/auto_test/tensorboard/scaling_moe_00196_noise1_v2")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load all tfevents files in dir
    files = sorted(glob.glob(f"{args.tb_dir}/*tfevents*"))
    if not files:
        raise SystemExit(f"no tfevents in {args.tb_dir}")

    by_step = {}
    for f in files:
        ea = EventAccumulator(f, size_guidance={'scalars': 0}).Reload()
        if 'lm loss' not in ea.Tags()['scalars']: continue
        for e in ea.Scalars('lm loss'):
            by_step[int(e.step)] = e.value
    if not by_step:
        raise SystemExit("no lm loss scalars")

    steps = sorted(by_step.keys())
    train_loss = [[s, by_step[s]] for s in steps]

    # Ref baseline (ref_noise1_v2) for delta compute
    ref_files = glob.glob(f"{args.ref_tb_dir}/*tfevents*")
    ref_loss = {}
    for f in ref_files:
        ea = EventAccumulator(f, size_guidance={'scalars':0}).Reload()
        if 'lm loss' in ea.Tags()['scalars']:
            for e in ea.Scalars('lm loss'): ref_loss[int(e.step)] = e.value

    # Compute compare vs ref
    common = [s for s in steps if s in ref_loss]
    diffs = [by_step[s] - ref_loss[s] for s in common]
    compare = {}
    if diffs:
        abs_diffs = [abs(d) for d in diffs]
        compare = {
            "n_common_steps": len(common),
            "iter_offset": 0,
            "max_abs_diff": max(abs_diffs),
            "max_abs_diff_step": common[int(np.argmax(abs_diffs))],
            "first_diverge_step_1e4": next((s for s, d in zip(common, abs_diffs) if d > 1e-4), -1),
            "mean_abs_diff": float(np.mean(abs_diffs)),
            "early_1_50_mean_abs": float(np.mean([abs(d) for s, d in zip(common, diffs) if 1 <= s <= 50])) if any(1<=s<=50 for s in common) else 0.0,
            "mid_51_500_mean_abs": float(np.mean([abs(d) for s, d in zip(common, diffs) if 51 <= s <= 500])) if any(51<=s<=500 for s in common) else 0.0,
            "mid_501_2000_mean_abs": float(np.mean([abs(d) for s, d in zip(common, diffs) if 501 <= s <= 2000])) if any(501<=s<=2000 for s in common) else 0.0,
            "decay_6k_end_mean_abs": float(np.mean([abs(d) for s, d in zip(common, diffs) if 6000 <= s])) if any(6000<=s for s in common) else 0.0,
            "final_iter_diff": diffs[-1],
        }

    out = {
        "run_id": args.run_id, "label": args.label,
        "has_biasfix": True,
        "host": None,
        "config": "PAI Megatron TB",
        "started_at": "",
        "init_from": "",
        "notes": args.notes,
        "iters_completed": steps[-1],
        "global_batch_size": 64,
        "dp_world_size": 8,
        "train_loss_points": train_loss,
        "val_loss_points": [],
        "routing_stats_points": [],
        "compare": compare,
        "final_nano_loss": train_loss[-1][1],
        "final_ref_loss": ref_loss.get(steps[-1], None),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}  (n={len(train_loss)})")

if __name__ == "__main__":
    main()
