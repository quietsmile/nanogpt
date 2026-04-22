"""Pull latest training logs from GPU box and refresh run JSONs + runs_index + dashboard.

Usage:
  python3 dashboard/refresh_runs.py            # refresh all known runs
  python3 dashboard/refresh_runs.py --rebuild  # also rebuild dashboard HTML
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Each entry: (run_id, label, has_biasfix, host, remote_jsonl_path, remote_log_path, config)
RUNS = [
    {
        'run_id': 'nano-196-20260420_231055',
        'label': 'v1 (buggy bias) · 7485步 full run',
        'has_biasfix': False,
        'host': 'root@22.4.243.44',
        'remote_jsonl': '/root/nanogpt/out-cybertron-moe-196-resume/train_log.jsonl',
        'remote_log': '/root/nanogpt/logs/train_resume_full.log',
        'config': 'config/cybertron_moe_196_resume.py',
        'started_at': '2026-04-20 23:10:55 +0800',
        'init_from': 'resume (ref iter_0)',
    },
    {
        'run_id': 'nano-196-biasfix-20260421_123000',
        'label': 'v2 (aux-free bias fix) · 500步 sanity',
        'has_biasfix': True,
        'host': 'root@22.4.243.44',
        'remote_jsonl': '/root/nanogpt/out-cybertron-moe-196-resume-biasfix/train_log.jsonl',
        'remote_log': '/root/nanogpt/logs/train_biasfix_500_v2.log',
        'config': 'config/cybertron_moe_196_resume_test.py',
        'started_at': '2026-04-21 12:30:00 +0800',
        'init_from': 'resume (ref iter_0)',
        'notes': 'Aux-free bias 改为 per-optim-step 更新（match Megatron）。结论：对 loss 轨迹影响 < 0.01 nat，gap 仍 ~1 nat —— 不是主要 bug。',
    },
    {
        'run_id': 'nano-196-fullfix-20260421_130000',
        'label': 'v4 (resume-path fix, iter_0 init, fresh AdamW) · 200步',
        'has_biasfix': True,
        'host': 'root@22.4.243.44',
        'remote_jsonl': '/root/nanogpt/out-cybertron-moe-196-resume-eodmask/train_log.jsonl',
        'remote_log': '/root/nanogpt/logs/train_eodmask_v4.log',
        'config': 'config/cybertron_moe_196_eodmask.py',
        'started_at': '2026-04-21 13:00:00 +0800',
        'init_from': 'resume (ref iter_0, fresh optim state — STILL wrong)',
        'notes': 'v4 修了 resume-path bug（透传 greedy/eod_mask/loss_masks/seq_aux），但从 iter_0 起 fresh AdamW 仍有 step=0 vs ref step=1 错位 → 1 nat gap 持续。',
    },
    {
        'run_id': 'nano-196-from10-20260421_170900',
        'label': 'v5 (iter_10 + optim state, 8-GPU DDP) · 200步（随机数据 bug 仍在）',
        'has_biasfix': True,
        'host': 'root@22.4.243.44',
        'remote_jsonl': '/root/nanogpt/out-cybertron-moe-196-from10/train_log.jsonl',
        'remote_log': '/root/nanogpt/logs/train_from10_v2.log',
        'config': 'config/cybertron_moe_196_from10.py',
        'started_at': '2026-04-21 17:09:00 +0800',
        'init_from': 'resume (ref iter_10 weights + full Adam history step=10)',
        'notes': 'v5 时还没发现 use_sequential 字符串匹配 bug，数据其实是 random sampling，导致 Δ +0.7 nat 漂移。',
    },
    {
        'run_id': 'nano-196-aligned-20260421_200000',
        'label': 'v6 对齐版（use_sequential + LR + find_unused fix）· 7485步 full run (in progress)',
        'has_biasfix': True,
        'host': 'root@22.4.243.44',
        'remote_jsonl': '/root/nanogpt/out-cybertron-moe-196-from0/train_log.jsonl',
        'remote_log': '/root/nanogpt/logs/train_from0_200.log',
        'config': 'config/cybertron_moe_196_from0.py',
        'started_at': '2026-04-21 20:00:00 +0800',
        'init_from': 'resume (ref iter_0 weights + Adam step=1, iter_num=0)',
        'iter_offset': 1,  # nano iter N ↔ ref iter N+1 (nano 0-indexed, ref 1-indexed)
        'notes': '所有 bug 修完：use_sequential、LR off-by-one、optim state、greedy routing、loss masks、find_unused_parameters=False。iter 0 bitwise 对齐 ref iter 1。MFU ≈ 9.2% (mb=1, 受限于 lm_head logits tensor memory)。200 步后 Δ ~ -0.025 nat，iter 500 Δ = -0.014 nat，持续收敛到 bf16 ULP 振荡 ceiling。',
    },
]


def ssh_cat(host, path):
    try:
        return subprocess.check_output(
            ['ssh', '-o', 'StrictHostKeyChecking=no', host, f'cat {path}'],
            text=True, timeout=60, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return ''


def ssh_grep(host, path, pattern):
    try:
        return subprocess.check_output(
            ['ssh', '-o', 'StrictHostKeyChecking=no', host,
             f"grep -E '{pattern}' {path} 2>/dev/null || true"],
            text=True, timeout=30,
        )
    except subprocess.CalledProcessError:
        return ''


def load_ref():
    p = f'{ROOT}/reference/tb/key_scalars.json'
    tb = json.load(open(p))
    return {int(s): v for s, v in tb['lm loss']}


def downsample(iters, keep_early=100, mod=1, max_iter=None):
    # mod=1 → keep every iter. Dashboard can handle 7k+ points via Plotly.
    out = []
    for i in iters:
        if i <= keep_early or i % mod == 0 or i == max_iter:
            out.append(i)
    return out


def refresh_one(run_meta, ref):
    host = run_meta['host']
    jsonl = ssh_cat(host, run_meta['remote_jsonl'])
    if not jsonl.strip():
        print(f"[{run_meta['run_id']}] no log yet", file=sys.stderr)
        return None

    nano = {}
    for line in jsonl.splitlines():
        try:
            d = json.loads(line)
            nano[d['iter']] = d
        except Exception:
            pass
    if not nano:
        return None

    max_iter = max(nano)
    keeps = set(downsample(sorted(nano), max_iter=max_iter))
    train_pairs = [[i, nano[i]['loss']] for i in sorted(keeps)]

    # val points from text log
    val_pairs = []
    txt = ssh_grep(host, run_meta['remote_log'], 'val loss')
    for line in txt.splitlines():
        m = re.search(r'step (\d+).*train loss ([\d.]+), val loss ([\d.]+)', line)
        if m:
            val_pairs.append([int(m.group(1)), float(m.group(3))])

    # compare vs ref — use per-run iter_offset (nano iter N ↔ ref iter N+offset)
    offset = int(run_meta.get('iter_offset', 0))
    common_nano = sorted(n for n in nano if (n + offset) in ref)
    diffs = [abs(nano[n]['loss'] - ref[n + offset]) for n in common_nano]
    if diffs:
        max_d = max(diffs); max_idx = common_nano[diffs.index(max_d)]
    else:
        max_d = None; max_idx = None

    def bucket(lo, hi):
        vals = [abs(nano[n]['loss'] - ref[n + offset]) for n in common_nano if lo < n <= hi]
        return sum(vals) / len(vals) if vals else None

    compare = {
        'n_common_steps': len(common_nano),
        'iter_offset': offset,
        'max_abs_diff': max_d, 'max_abs_diff_step': max_idx,
        'first_diverge_step_1e4': next((n for n in common_nano if abs(nano[n]['loss'] - ref[n + offset]) > 1e-4), None),
        'mean_abs_diff': (sum(diffs) / len(diffs)) if diffs else None,
        'early_1_50_mean_abs':   bucket(0, 50),
        'mid_51_500_mean_abs':   bucket(50, 500),
        'mid_501_2000_mean_abs': bucket(500, 2000),
        'decay_6k_end_mean_abs': bucket(6000, 10**9),
        'final_iter_diff': (nano[max_iter]['loss'] - ref[max_iter + offset]) if (max_iter + offset) in ref else None,
    }

    run = dict(run_meta)
    run.pop('remote_jsonl', None); run.pop('remote_log', None)
    run.update({
        'iters_completed': max_iter,
        'global_batch_size': 64,
        'dp_world_size': 8,
        'train_loss_points': train_pairs,
        'val_loss_points': val_pairs,
        'compare': compare,
        'final_nano_loss': nano[max_iter]['loss'],
        'final_ref_loss': ref.get(max_iter),
    })
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true', help='also rebuild dashboard HTML')
    args = ap.parse_args()

    ref = load_ref()
    os.makedirs(f'{ROOT}/reports/runs', exist_ok=True)
    index = []
    for rm in RUNS:
        r = refresh_one(rm, ref)
        if r is None:
            continue
        path = f'{ROOT}/reports/runs/{r["run_id"]}.json'
        with open(path, 'w') as f:
            json.dump(r, f)
        index.append({
            'run_id': r['run_id'], 'label': r['label'],
            'iters_completed': r['iters_completed'],
            'has_biasfix': r['has_biasfix'],
            'started_at': r['started_at'],
            'file': f'runs/{r["run_id"]}.json',
        })
        print(f"[{r['run_id']}] {r['iters_completed']} iters, "
              f"last loss {r['final_nano_loss']:.4f}, "
              f"max|Δ|={r['compare']['max_abs_diff']:.4f}@{r['compare']['max_abs_diff_step']}")
    with open(f'{ROOT}/reports/runs_index.json', 'w') as f:
        json.dump(index, f, indent=2)
    print(f'updated runs_index.json: {len(index)} runs')

    if args.rebuild:
        subprocess.check_call([sys.executable, f'{ROOT}/dashboard/build_local.py'])


if __name__ == '__main__':
    main()
