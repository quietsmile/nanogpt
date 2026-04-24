"""Build reports/dense_ablation.json for the dashboard's dense-ablation panel.

Emits a flexible `runs` list so the dashboard can let the user pick any subset to
plot + compute pairwise Δ. Each run has: id, label, color, iter_offset, source, points.

Extend by appending a new entry to DENSE_RUNS below.

Convention: iter_offset is how to SHIFT the run's iter_0 to align with the canonical
"step 1 = first optim step" x-axis. Megatron's first logged iter is 1, nano's is 0,
so nano runs use offset=1 and Megatron uses offset=0.

Output schema (current):
{
  "runs": [
    {"id", "label", "color", "iter_offset", "source", "points": [[iter, loss, lr, grad_norm], ...]},
    ...
  ],
  "meta": {"generated_at": str, ...}
}
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NANO_HOST = 'root@22.4.243.44'

# Registry of all dense runs (ref + nano variants). Add new rows here; nothing
# else needs changing in build or dashboard code.
DENSE_RUNS = [
    # --- 107 config (canonical smallest-dense, n_layer=8, h=528, ffn=1824) ---
    {
        'id': 'ref_dense_107_v3',
        'label': 'ref-dense_107 (v3, Megatron)',
        'color': '#79c0ff',
        'iter_offset': 0,
        'source_type': 'megatron_log',
        'source': '/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_dense_00107_full_v3/logs/rank-0-1-scaling_dense_00107_full_v3-run.log',
    },
    {
        'id': 'ref_dense_107_v2',
        'label': 'ref-dense_107 (v2, earlier, stopped @~3.4k)',
        'color': '#58a6ff',
        'iter_offset': 0,
        'source_type': 'megatron_log',
        'source': '/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_dense_00107_full_v2/logs/rank-0-1-scaling_dense_00107_full_v2-run.log',
    },
    {
        'id': 'nano_dense_107_seed1337',
        'label': 'nano-dense_107 seed=1337 (current)',
        'color': '#ff7b72',
        'iter_offset': 1,
        'source_type': 'ssh_jsonl',
        'source': f'{NANO_HOST}:/root/nanogpt/out-cybertron-dense-107/train_log.jsonl',
    },
    {
        'id': 'nano_dense_107_seed1337_old',
        'label': 'nano-dense_107 seed=1337 (earlier, stopped @4050)',
        'color': '#f0883e',
        'iter_offset': 1,
        'source_type': 'ssh_jsonl',
        'source': f'{NANO_HOST}:/root/nanogpt/out-cybertron-dense-107_old_stopped/train_log.jsonl',
    },
    {
        'id': 'nano_dense_107_seed42',
        'label': 'nano-dense_107 seed=42 (planned)',
        'color': '#a5d6ff',
        'iter_offset': 1,
        'source_type': 'ssh_jsonl',
        'source': f'{NANO_HOST}:/root/nanogpt/out-cybertron-dense-107-seed42/train_log.jsonl',
    },
    {
        'id': 'nano_dense_107_from_ref',
        'label': 'nano-dense_107 from ref iter_2 (planned)',
        'color': '#d2a8ff',
        'iter_offset': 1,
        'source_type': 'ssh_jsonl',
        'source': f'{NANO_HOST}:/root/nanogpt/out-cybertron-dense-107-from-ref/train_log.jsonl',
    },
    # --- 196-dimensioned dense (MoE-off variant of scaling_moe_00196; n_layer=9, h=512, ffn=1536) ---
    #     Loss not directly comparable to 107 (different architecture) but useful as
    #     historical MoE-vs-dense ablation — nano and ref here share architecture.
    {
        'id': 'ref_dense_196_full',
        'label': 'ref-dense_196 (MoE-off variant, Megatron)',
        'color': '#bc8cff',
        'iter_offset': 0,
        'source_type': 'megatron_log',
        'source': '/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_dense_00196_full/logs/rank-0-1-scaling_dense_00196_full-run.log',
    },
    {
        'id': 'nano_dense_196_full',
        'label': 'nano-dense_196 (MoE-off variant)',
        'color': '#ff9bc7',
        'iter_offset': 1,
        'source_type': 'ssh_jsonl',
        'source': f'{NANO_HOST}:/root/nanogpt/out-cybertron-dense-196-full/train_log.jsonl',
    },
]

# --- 5 nano seed noise-floor runs (1000 iter each) + 5 PAI seed noise-floor runs ---
#     Auto-generated so adding a new seed is one-line change.
_NOISE_NANO_COLORS = ['#ff7b72', '#ff9f43', '#f78166', '#ffa940', '#fd7e14']
_NOISE_PAI_COLORS  = ['#79c0ff', '#58a6ff', '#468bff', '#3182ff', '#1f6feb']
for _s in (1, 2, 3, 4, 5):
    # box1 runs noise1-3, box2 runs noise4-5; all full 6711 iter.
    _nano_host = 'root@22.4.243.44' if _s <= 3 else 'root@22.1.6.211'
    DENSE_RUNS.append({
        'id': f'nano_dense_107_noise{_s}',
        'label': f'nano-107 noise{_s} (seed={_s*1000}, full 6711 iter)',
        'color': _NOISE_NANO_COLORS[_s-1],
        'iter_offset': 1,
        'source_type': 'ssh_jsonl',
        'source': f'{_nano_host}:/root/nanogpt/out-cybertron-dense-107-noise{_s}-full/train_log.jsonl',
    })
# Same seed=1000 as noise1 but with deterministic=False + TE attn + no chunked_ce
# → 1.78× faster. Loss curve should track noise1 closely (same seed, but different
# matmul/attention kernels mean non-bitwise at bf16 ULP).
DENSE_RUNS.append({
    'id': 'nano_dense_107_noise1_fast',
    'label': 'nano-107 noise1 FAST (seed=1000, 437ms/iter)',
    'color': '#d97706',
    'iter_offset': 1,
    'source_type': 'ssh_jsonl',
    'source': f'{NANO_HOST}:/root/nanogpt/out-cybertron-dense-107-noise1-fast/train_log.jsonl',
})
for _s in (1, 2, 3, 4, 5):
    # noise1 full run reuses _v3 path (bug: EXP_NAME wasn't bumped before resubmit,
    #   overwrote 1000-iter log; not worth fixing now since full run data is good).
    # noise2-5 full runs use clean _full paths.
    _suffix = 'v3' if _s == 1 else 'full'
    DENSE_RUNS.append({
        'id': f'ref_dense_107_noise{_s}',
        'label': f'ref-107 noise{_s} (seed={_s*1000}, full 6711 iter)',
        'color': _NOISE_PAI_COLORS[_s-1],
        'iter_offset': 0,
        'source_type': 'megatron_log',
        'source': f'/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_dense_00107_noise{_s}_{_suffix}/logs/rank-0-1-scaling_dense_00107_noise{_s}_{_suffix}-run.log',
    })

# --- MoE 196 noise fleet (7485 iter) ---
_MOE_NANO_COLORS = ['#ff5555', '#ff9500']
_MOE_PAI_COLORS = ['#56d4ff', '#3182ff', '#9ba0ff', '#be95ff', '#f778ba', '#79c0ff']
DENSE_RUNS.append({
    'id': 'nano_moe_196_noise1_mb1',
    'label': 'nano-moe_196 noise1 mb=1 fp32 TE (seed=1000)',
    'color': _MOE_NANO_COLORS[0],
    'iter_offset': 1,
    'source_type': 'ssh_jsonl',
    'source': f'{NANO_HOST}:/root/nanogpt/out-cybertron-moe-196-noise1/train_log.jsonl',
})
DENSE_RUNS.append({
    'id': 'nano_moe_196_noise2_mb4',
    'label': 'nano-moe_196 noise2 mb=4 bf16 TE (seed=2000)',
    'color': _MOE_NANO_COLORS[1],
    'iter_offset': 1,
    'source_type': 'ssh_jsonl',
    'source': 'root@22.1.6.211:/root/nanogpt/out-cybertron-moe-196-noise2-mb4/train_log.jsonl',
})
for _s in (1, 2, 3, 4, 5):
    DENSE_RUNS.append({
        'id': f'ref_moe_196_noise{_s}',
        'label': f'ref-moe_196 noise{_s} (seed={_s*1000})',
        'color': _MOE_PAI_COLORS[_s-1],
        'iter_offset': 0,
        'source_type': 'megatron_log',
        'source': f'/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196_noise{_s}_v2/logs/rank-0-1-scaling_moe_00196_noise{_s}_v2-run.log',
    })
DENSE_RUNS.append({
    'id': 'ref_moe_196_speedtest',
    'label': 'ref-moe_196 speedtest (seed=1000, ws 262162)',
    'color': _MOE_PAI_COLORS[5],
    'iter_offset': 0,
    'source_type': 'megatron_log',
    'source': '/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196_speedtest/logs/rank-0-1-scaling_moe_00196_speedtest-run.log',
})


def parse_ref_log(path):
    if not path or not os.path.exists(path):
        return []
    out = []
    iter_re = re.compile(r'iteration\s+(\d+)/')
    loss_re = re.compile(r'lm loss:\s*([-+0-9.Ee]+)')
    lr_re   = re.compile(r'learning rate:\s*([-+0-9.Ee]+)')
    gn_re   = re.compile(r'grad norm:\s*([-+0-9.Ee]+)')
    with open(path) as f:
        for line in f:
            m = iter_re.search(line)
            if not m:
                continue
            it = int(m.group(1))
            l = loss_re.search(line); lr = lr_re.search(line); gn = gn_re.search(line)
            if not (l and lr and gn):
                continue
            out.append([it, float(l.group(1)), float(lr.group(1)), float(gn.group(1))])
    return out


def fetch_ssh_jsonl(spec):
    host, _, path = spec.partition(':')
    if not host or not path:
        return []
    try:
        txt = subprocess.check_output(
            ['ssh', '-o', 'StrictHostKeyChecking=no', host, f'cat {path}'],
            text=True, timeout=60, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    out = []
    for line in txt.splitlines():
        try:
            d = json.loads(line)
            out.append([int(d['iter']), float(d['loss']),
                        float(d.get('lr', 0.0)), float(d.get('grad_norm', 0.0))])
        except Exception:
            continue
    return out


def fetch_points(run):
    st = run['source_type']
    if st == 'megatron_log':
        return parse_ref_log(run['source'])
    if st == 'ssh_jsonl':
        return fetch_ssh_jsonl(run['source'])
    raise ValueError(f'unknown source_type: {st}')


def _aggregate_fleet(runs_out, ids, label, color, iter_offset):
    """Compute mean + stderr across a set of runs at each aligned iter.
    Returns a synthetic run dict or None if no usable data."""
    fleet = [r for r in runs_out if r['id'] in ids and r['n_points']]
    if not fleet:
        return None
    # Find iters present in ALL fleet members for a reliable mean.
    shared = None
    for r in fleet:
        its = {p[0] for p in r['points']}
        shared = its if shared is None else shared & its
    if not shared:
        return None
    shared = sorted(shared)
    # For each shared iter, compute avg loss and avg grad_norm across fleet.
    from statistics import mean, stdev
    avg_pts = []
    ys_loss, ys_gn = [], []  # for stats
    for it in shared:
        losses = []
        gns = []
        for r in fleet:
            for p in r['points']:
                if p[0] == it:
                    losses.append(p[1])
                    gns.append(p[3])
                    break
        if len(losses) == len(fleet):
            mu = mean(losses)
            gmu = mean(gns)
            avg_pts.append([it, mu, 0.0, gmu])
            ys_loss.append(losses)
            ys_gn.append(gns)
    # Fleet-level variance: mean over iters of per-iter cross-run std
    per_iter_stds = [stdev(ys) if len(ys) > 1 else 0.0 for ys in ys_loss]
    avg_cross_run_std = sum(per_iter_stds) / len(per_iter_stds) if per_iter_stds else None
    return {
        'id': f'fleet_avg_{label}',
        'label': f'FLEET AVG · {label} (n={len(fleet)})',
        'color': color,
        'iter_offset': iter_offset,
        'source': f'synthetic avg of {len(fleet)} runs',
        'source_type': 'fleet_avg',
        'points': avg_pts,
        'n_points': len(avg_pts),
        'last_loss': avg_pts[-1][1] if avg_pts else None,
        'last_iter': avg_pts[-1][0] if avg_pts else None,
        'fleet_n_runs': len(fleet),
        'fleet_avg_cross_run_std_loss': avg_cross_run_std,
    }


def _ema_last(ys, alpha=0.02):
    if not ys: return None
    e = ys[0]
    for x in ys[1:]:
        e = alpha * x + (1 - alpha) * e
    return e


def _tail_stats(ys, n=100):
    if not ys: return None, None
    ys = ys[-n:]
    m = sum(ys) / len(ys)
    s = None
    if len(ys) > 1:
        s = (sum((x - m) ** 2 for x in ys) / (len(ys) - 1)) ** 0.5
    return m, s


def main():
    runs_out = []
    for run in DENSE_RUNS:
        pts = fetch_points(run)
        if not pts:
            print(f'[warn] no data for {run["id"]} ({run["source"]})', file=sys.stderr)
        losses = [p[1] for p in pts]
        tail100_mean, tail100_std = _tail_stats(losses, 100)
        r = {
            'id': run['id'],
            'label': run['label'],
            'color': run['color'],
            'iter_offset': run['iter_offset'],
            'source': run['source'],
            'source_type': run['source_type'],
            'points': pts,
            'n_points': len(pts),
            'last_loss': pts[-1][1] if pts else None,
            'last_iter': pts[-1][0] if pts else None,
            'ema_last': _ema_last(losses, 0.02),       # α=0.02, effective window ~50-100 iter
            'tail100_mean': tail100_mean,              # simple mean of last 100 iter losses
            'tail100_std': tail100_std,                # within-run std of last 100 iter losses
        }
        runs_out.append(r)

    # Append synthetic fleet averages (computed across 5-seed noise probes).
    nano_ids = {f'nano_dense_107_noise{i}' for i in range(1, 6)}
    pai_ids  = {f'ref_dense_107_noise{i}' for i in range(1, 6)}
    nano_fleet = _aggregate_fleet(runs_out, nano_ids, 'nano', '#f85149', iter_offset=1)
    pai_fleet  = _aggregate_fleet(runs_out, pai_ids,  'pai-ref', '#2f81f7', iter_offset=0)
    if nano_fleet: runs_out.append(nano_fleet)
    if pai_fleet:  runs_out.append(pai_fleet)

    payload = {
        'runs': runs_out,
        'meta': {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config_ref': 'scaling_dense_00107_nc.yaml (canonical smallest-dense node)',
            'n_layer': 8, 'hidden': 528, 'ffn': 1824, 'kv_channels': 128,
            'gbs': 48, 'lr': 0.001063, 'max_iters': 6711,
            'nano_fleet_cross_run_std_loss': nano_fleet['fleet_avg_cross_run_std_loss'] if nano_fleet else None,
            'pai_fleet_cross_run_std_loss': pai_fleet['fleet_avg_cross_run_std_loss'] if pai_fleet else None,
        },
    }

    out_path = os.path.join(ROOT, 'reports', 'dense_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f)

    print(f'wrote {out_path}  ({len(runs_out)} runs)')
    for r in runs_out:
        last = f'last_iter={r["last_iter"]} loss={r["last_loss"]:.4f}' if r['n_points'] else 'EMPTY'
        print(f'  {r["id"]:<35} {r["n_points"]:>5} pts  {last}')


if __name__ == '__main__':
    main()
