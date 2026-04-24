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
    {
        'run_id': 'nano-196-v10final-20260422',
        'label': 'v10 FINAL (fused_RoPE + te.GroupedLinear + NCCL determinism) · 7485步 full retrain',
        'has_biasfix': True,
        'host': 'root@22.4.243.44',
        'remote_jsonl': '/root/nanogpt/out-cybertron-moe-196-from0-fresh/train_log.jsonl',
        'remote_log': '/root/nanogpt/logs/train_from0_fresh.log',
        'config': 'config/cybertron_moe_196_from0.py',
        'started_at': '2026-04-22 00:00:00 +0800',
        'init_from': 'resume (ref iter_0 weights + Adam step=1, iter_num=0)',
        'iter_offset': 1,
        'notes': 'v10 FINAL：fused_apply_rotary_pos_emb (95% drift 源) + te.GroupedLinear + fp32 SwiGLU/MoE/silu + 输入端 EOD mask + NCCL strict determinism。last-100-mean nano=2.8539 / ref=2.8493 / Δ=+0.0047 nat（1/10 step stdev）。Δ 在 ±0.15 nat 震荡并收敛到 ULP。Val loss Δ=+0.023 nat。',
    },
    {
        'run_id': 'nano-196-moediag-20260423',
        'label': 'v10 moediag · 200步（iter 7000→7200, MoE routing stats logged）',
        'has_biasfix': True,
        'host': 'root@22.4.243.44',
        'remote_jsonl': '/root/nanogpt/out-cybertron-moe-196-moediag/train_log.jsonl',
        'remote_log': '/root/nanogpt/logs/moediag.log',
        'config': 'config/cybertron_moe_196_moediag.py',
        'started_at': '2026-04-23 01:24:00 +0800',
        'init_from': 'resume (v10-fresh iter_7000 ckpt)',
        'iter_offset': 1,
        'notes': '短跑，主要为了收集 per-iter MoE 路由 stats (maxvio, tok_per_expert)。 maxvio mean 0.54 vs ref 1.95 — nano routing 3.6× 更均衡。Loss Δ = +0.00541 (201 iter avg), 跟完整 7485 步 +0.00546 几乎 bitwise 吻合。',
    },
    {
        'run_id': 'nano-196-iter0diag-20260423',
        'label': 'v10 iter_0 diag · 500步（from iter_0 seed, 测 maxvio 早期是否对齐）',
        'has_biasfix': True,
        'host': 'root@22.4.243.44',
        'remote_jsonl': '/root/nanogpt/out-moediag-iter0/train_log.jsonl',
        'remote_log': '/root/nanogpt/logs/iter0diag.log',
        'config': 'config/cybertron_moe_196_iter0_diag.py',
        'started_at': '2026-04-23 02:30:00 +0800',
        'init_from': 'seed from ref iter_0000000 + meg_optim_iter0 (step=1)',
        'iter_offset': 1,
        'notes': '测试 maxvio 在 warmup 早期是否跟 ref 对齐；若对齐再在 optim 过程中 diverge，则定位到累积差异源。',
    },
    {
        'run_id': 'nano-196-v10repro-fixed-ab-s1337-20260424',
        'label': 'v10repro fixed-AB s1337 · TE+sync MoE fix + fast-path flags · 7485步',
        'has_biasfix': True,
        'host': 'root@22.4.243.44',
        'remote_jsonl': '/root/nanogpt/out-v10repro-fixed-ab-s1337/train_log.jsonl',
        'remote_log': '/root/nanogpt/out-v10repro-fixed-ab-s1337/train.log',
        'config': 'config/cybertron_moe_196_v10repro_fixed_ab_s1337.py',
        'started_at': '2026-04-24 11:13:00 +0800',
        'init_from': 'resume (ref iter_0 seed ckpt)',
        'notes': 'TE+sync-grad-back MoE fix from agent；A(fused all-reduce)+B(fast-path flags)。Killed at iter ~130 后切 bucket-path。',
    },
    {
        'run_id': 'nano-196-v10repro-fixed-ab-s4242-20260424',
        'label': 'v10repro fixed-AB s4242 · TE+sync · 7485步',
        'has_biasfix': True,
        'host': 'root@22.1.6.211',
        'remote_jsonl': '/root/nanogpt/out-v10repro-fixed-ab-s4242/train_log.jsonl',
        'remote_log': '/root/nanogpt/out-v10repro-fixed-ab-s4242/train.log',
        'config': 'config/cybertron_moe_196_v10repro_fixed_ab_s4242.py',
        'started_at': '2026-04-24 11:14:00 +0800',
        'init_from': 'resume (ref iter_0 seed ckpt)',
        'notes': 'Seed 4242 on box2，TE+sync MoE fix。Killed at iter ~40 后切 bucket-path。',
    },
    {
        'run_id': 'nano-196-v10repro-bucket-s1337-20260424',
        'label': 'v10repro bucket s1337 · NANO_TE_MOE=0 bucket-padding · 7485步',
        'has_biasfix': True,
        'host': 'root@22.4.243.44',
        'remote_jsonl': '/root/nanogpt/out-v10repro-bucket-s1337/train_log.jsonl',
        'remote_log': '/root/nanogpt/out-v10repro-bucket-s1337/train.log',
        'config': 'config/cybertron_moe_196_v10repro_bucket_s1337.py',
        'started_at': '2026-04-24 11:36:00 +0800',
        'init_from': 'resume (ref iter_0 seed ckpt)',
        'notes': 'Bucket-padding path (grad 直接流 self.gate_weight，无 sync overhead)。dt ~1280ms/iter，比 TE+sync 快 2.6×，比 ref Megatron 只慢 1.3×。对 v10 轨迹 bf16 ULP 匹配。',
    },
    {
        'run_id': 'nano-196-v10repro-bucket-s4242-20260424',
        'label': 'v10repro bucket s4242 · NANO_TE_MOE=0 · 7485步',
        'has_biasfix': True,
        'host': 'root@22.1.6.211',
        'remote_jsonl': '/root/nanogpt/out-v10repro-bucket-s4242/train_log.jsonl',
        'remote_log': '/root/nanogpt/out-v10repro-bucket-s4242/train.log',
        'config': 'config/cybertron_moe_196_v10repro_bucket_s4242.py',
        'started_at': '2026-04-24 11:36:00 +0800',
        'init_from': 'resume (ref iter_0 seed ckpt)',
        'notes': 'Seed 4242 on box2，bucket-padding path。Seed variance check with s1337。',
    },
    {
        'run_id': 'nano-196-pai-v10repro-bucket-full-20260424',
        'label': 'v10repro bucket PAI full · 7485步（PAI dlc1e59rt6tz8bnz）',
        'has_biasfix': True,
        'host': None,  # PAI — data already on CPFS
        'remote_jsonl': '/prodcpfs/user/yuchen/scaling_exp/auto_test/nano_moe_196_pai_v10repro_bucket_full/train_log.jsonl',
        'remote_log': '/prodcpfs/user/yuchen/scaling_exp/auto_test/nano_moe_196_pai_v10repro_bucket_full/logs/rank-0-1-nano_moe_196_pai_v10repro_bucket_full.log',
        'config': 'config/cybertron_moe_196_pai_v10repro_bucket_full.py',
        'started_at': '2026-04-24 11:43:00 +0800',
        'init_from': 'resume (ref iter_0 seed ckpt)',
        'notes': 'PAI ws 137902 quota quotadbz1mvpy1v5，1 pod × 8 GPU，bucket-padding + fast-path。dt ~1440ms/iter，~3h 到 iter 7485。',
    },
]


def ssh_cat(host, path):
    # host=None means local file (used for PAI runs whose data is on mounted CPFS)
    if host is None:
        try:
            with open(path) as f:
                return f.read()
        except Exception:
            return ''
    try:
        return subprocess.check_output(
            ['ssh', '-o', 'StrictHostKeyChecking=no', host, f'cat {path}'],
            text=True, timeout=60, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return ''


def ssh_grep(host, path, pattern):
    if host is None:
        try:
            return subprocess.check_output(
                ['sh', '-c', f"grep -E '{pattern}' {path} 2>/dev/null || true"],
                text=True, timeout=30,
            )
        except subprocess.CalledProcessError:
            return ''
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

    # Routing stats (only present for runs trained after d782fc3 added the
    # per-iter tokens_per_expert logging).
    # IMPORTANT aggregation fix (2026-04-23): the original
    # `maxvio_micro_batch` field was computed as max across [L, E] of
    # counts summed across all 64 mbs, which is NOT the same formula ref
    # uses. Ref's `maxVio/micro_batch` (cybertron modules_deepseekv2:547)
    # is (max - mean) / mean on a SINGLE microbatch's [E] tensor, averaged
    # across mbs × layers. So: when moe_per_layer is available, use the
    # per-layer-averaged max/mean (already mean-across-mbs) to compute
    # a per-layer maxvio and then mean across layers. This matches ref
    # within 0.3% at iter 1 with identical weights (verified via
    # /tmp/apples_maxvio.py).
    _dp = 8  # nano's DP world size (all runs here)
    routing_pairs = []
    for i in sorted(keeps):
        d = nano[i]
        if 'tokens_per_expert_max' in d or 'maxvio_mb4_apples' in d:
            # Priority 1: apples-to-apples synthetic mb=4 maxvio (matches ref's
            # mb=4 granularity exactly). Available in runs trained after this fix.
            if 'maxvio_mb4_apples' in d:
                mv = float(d['maxvio_mb4_apples'])
                mx = float(d.get('tok_max_mb4_apples', 0.0))
                me = float(d.get('tok_mean_mb4_apples', 0.0))
                mn = 0.0  # not emitted (rarely used)
            # Priority 2: per-layer stats (pre-synthetic-mb fix).
            elif d.get('moe_per_layer'):
                pl = d['moe_per_layer']
                layer_vios = [((L['max'] - L['mean']) / L['mean']) if L['mean'] > 0 else 0
                              for L in pl]
                mv = sum(layer_vios) / len(layer_vios) if layer_vios else 0.0
                mx = sum(L['max'] for L in pl) / len(pl)
                mn = sum(L['min'] for L in pl) / len(pl)
                me = sum(L['mean'] for L in pl) / len(pl)
            else:
                # Fallback: the old misaligned formula.
                mv = float(d.get('maxvio_micro_batch', 0.0))
                mx = float(d.get('tokens_per_expert_max', 0.0)) / _dp
                mn = float(d.get('tokens_per_expert_min', 0.0)) / _dp
                me = float(d.get('tokens_per_expert_mean', 0.0)) / _dp
            routing_pairs.append([i, mv, mx, mn, me])
    # (No trailing elements; we already built the full entry.)

    run = dict(run_meta)
    run.pop('remote_jsonl', None); run.pop('remote_log', None)
    run.update({
        'iters_completed': max_iter,
        'global_batch_size': 64,
        'dp_world_size': 8,
        'train_loss_points': train_pairs,
        'val_loss_points': val_pairs,
        'routing_stats_points': routing_pairs,  # [iter, maxvio, tok_max_per_mb, tok_min_per_mb, tok_mean_per_mb]
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
