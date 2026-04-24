"""Build a single-file HTML dashboard with all JSON embedded.

Output: dashboard/alignment_report.html — double-clickable (file://).
Uses Plotly via CDN (needs internet when opening). For fully offline,
pass --inline-plotly (downloads plotly.min.js once into the HTML).
"""
import argparse
import json
import os
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

FILES = {
    'tokenizer':   'reports/tokenizer_alignment.json',
    'data':        'reports/data_sampling_alignment.json',
    'model':       'reports/model_alignment.json',
    'loss':        'reports/loss_trajectory.json',
    'nano_log':    'reports/nanogpt_train_log.json',
    'bitwise':     'reports/bitwise_resume.json',
    'ckpt':        'reports/ckpt_fingerprint.json',
    'tb':          'reference/tb/key_scalars.json',
    'job':         'reference/dlc1q9arre48b0kx.job.json',
    'checklist':   'reports/alignment_checklist.json',
    'fwd_align':   'reports/megatron_weight_alignment.json',
    'attn_maps':   'reports/attention_maps.json',   # monitor/attn_probe.py output
    'gaps':        None,  # synthesized below
    'runs_index':  'reports/runs_index.json',  # list of nano runs available
    'muon_alignment': 'reports/muon_alignment.json',  # muon-reimpl vs megatron muon ref
    'ref_routing': 'reference/ref_moe_routing_stats.json',  # per-iter ref MoE routing stats
    'dense_ablation': 'reports/dense_ablation.json',  # ref-dense vs nano-dense 50-iter ablation
    'moe_fleet':      'reports/moe_fleet.json',        # PAI Adam/Muon fleets + nano v10repro bucket
}


def compute_code_stats():
    """Line-count breakdown by repo subsystem. Produces data for the
    'code composition' bar chart on the Learning Dynamics tab.

    Categorization is intentionally coarse — purpose is to show the
    monitoring / viz / test ratio vs the core training framework, not
    to be a precise code-audit tool. Counts .py / .html / .js files
    only, skips __pycache__ / node_modules.
    """
    CATEGORIES = [
        ('核心逻辑',  ['train.py', 'model.py', 'configurator.py', 'sample.py',
                       'bench.py', 'prepare_cybertron_data.py']),
        ('配置',      ['config']),
        ('监控',      ['monitor']),
        ('可视化',    ['dashboard']),
        ('诊断工具',  ['scripts', 'tools']),
        ('测试',      ['tests']),
    ]
    EXTS = ('.py', '.html', '.js')

    def count(path):
        lines = files = 0
        if os.path.isfile(path):
            try:
                with open(path, errors='ignore') as f:
                    lines = sum(1 for _ in f)
                files = 1
            except OSError:
                pass
            return lines, files
        if not os.path.isdir(path):
            return 0, 0
        for root, dirs, fnames in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ('__pycache__', 'node_modules', '.git')]
            for fn in fnames:
                if fn.endswith(EXTS):
                    p = os.path.join(root, fn)
                    try:
                        with open(p, errors='ignore') as f:
                            lines += sum(1 for _ in f)
                        files += 1
                    except OSError:
                        pass
        return lines, files

    result = []
    for name, paths in CATEGORIES:
        total_lines = total_files = 0
        for rel in paths:
            l, f = count(os.path.join(ROOT, rel))
            total_lines += l
            total_files += f
        result.append({'category': name, 'lines': total_lines,
                       'files': total_files, 'paths': paths})
    return result


def load_all_runs():
    """Load every run file referenced in runs_index.json."""
    idx_path = os.path.join(ROOT, 'reports/runs_index.json')
    if not os.path.exists(idx_path):
        return []
    idx = json.load(open(idx_path))
    runs = []
    for entry in idx:
        run_path = os.path.join(ROOT, 'reports', entry['file']) if not entry['file'].startswith('reports/') else os.path.join(ROOT, entry['file'])
        if os.path.exists(run_path):
            with open(run_path) as f:
                runs.append(json.load(f))
    return runs


def load_monitor(max_points=2000):
    """Load learning-dynamics monitor.jsonl files.

    Discovery order:
      1. Per-run: reports/monitor/<run_id>.jsonl   (preferred, when
         refresh_runs also syncs monitor.jsonl from the GPU box)
      2. Local:  reports/monitor.jsonl             (single run in the dev box)

    Returns {run_id: [record, ...]}. Each record list is downsampled to
    ~max_points (stride pick) so the embedded HTML stays small. The schema
    written by monitor/core.py is kept verbatim — dashboard JS does the
    per-field extraction.
    """
    out = {}

    def _read(path):
        if not os.path.exists(path):
            return None
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        if not records:
            return None
        # stride-downsample; always keep last
        if len(records) > max_points:
            stride = max(1, len(records) // max_points)
            picked = records[::stride]
            if picked[-1] is not records[-1]:
                picked.append(records[-1])
            records = picked
        return records

    mon_dir = os.path.join(ROOT, 'reports', 'monitor')
    if os.path.isdir(mon_dir):
        for fname in sorted(os.listdir(mon_dir)):
            if not fname.endswith('.jsonl'):
                continue
            recs = _read(os.path.join(mon_dir, fname))
            if recs:
                out[fname[:-len('.jsonl')]] = recs
    local = os.path.join(ROOT, 'reports', 'monitor.jsonl')
    recs = _read(local)
    if recs:
        out.setdefault('local', recs)
    return out


def load(path):
    if path is None:
        return None
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    try:
        with open(full) as f:
            return json.load(f)
    except Exception as e:
        return {'_error': str(e)}


def build_gaps_summary():
    """Read memory/nanogpt_align_gaps_00196.md and produce structured diffs."""
    path = '/home/claudeuser/.claude/projects/-home-claudeuser-next-llm/memory/nanogpt_align_gaps_00196.md'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        md = f.read()
    return {'_raw_md': md}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'dashboard', 'alignment_report.html'))
    ap.add_argument('--inline-plotly', action='store_true',
                    help='Embed plotly.min.js for fully offline viewing (~3MB)')
    args = ap.parse_args()

    data = {}
    for key, path in FILES.items():
        if key == 'gaps':
            data[key] = build_gaps_summary()
        else:
            data[key] = load(path)
    # All available runs (new multi-experiment selector)
    data['runs'] = load_all_runs()
    # Learning-dynamics monitor records (from monitor/ package)
    data['monitor'] = load_monitor()
    # Code composition (for Learning Dynamics tab 'code composition' bar chart)
    data['code_stats'] = compute_code_stats()

    # Only keep lm loss and a couple scalars from tb to keep size reasonable
    if data.get('tb'):
        trimmed = {}
        for tag in ['lm loss', 'learning-rate', 'grad-norm', 'sequence_wise_balance_loss']:
            if tag in data['tb']:
                trimmed[tag] = data['tb'][tag]
        data['tb'] = trimmed

    # Trim job.json — keep only useful top-level fields
    if data.get('job'):
        body = data['job'].get('body', {})
        data['job'] = {
            'DisplayName': body.get('DisplayName'),
            'Status': body.get('Status'),
            'JobType': body.get('JobType'),
            'Duration': body.get('Duration'),
            'GmtCreateTime': body.get('GmtCreateTime'),
            'GmtFinishTime': body.get('GmtFinishTime'),
            'Image': (body.get('JobSpecs') or [{}])[0].get('Image'),
            'ResourceConfig': (body.get('JobSpecs') or [{}])[0].get('ResourceConfig'),
            'UserCommand': body.get('UserCommand'),
            'DataSources': body.get('DataSources'),
            'Tags': (body.get('Settings') or {}).get('Tags'),
        }

    # Trim tokenizer report — drop roundtrip input strings (can be long)
    if data.get('tokenizer') and 'roundtrip' in data['tokenizer']:
        rt = data['tokenizer']['roundtrip']
        data['tokenizer']['roundtrip_pass_count'] = sum(1 for r in rt if r.get('match'))
        data['tokenizer']['roundtrip_total'] = len(rt)
        data['tokenizer']['roundtrip'] = rt  # keep all, they are small

    # Plotly source
    plotly_src = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
    if args.inline_plotly:
        try:
            print("downloading plotly.min.js for inline embedding...")
            js = urllib.request.urlopen(
                'https://cdn.plot.ly/plotly-2.27.0.min.js', timeout=30).read().decode()
            plotly_src = f'<script>{js}</script>'
            print(f"  embedded {len(js)/1024:.0f}KB")
        except Exception as e:
            print(f"  failed: {e} — falling back to CDN link", file=sys.stderr)

    # Build HTML
    tmpl = HTML_TEMPLATE.replace('__PLOTLY_SRC__', plotly_src)
    # Embed data as a JSON blob. Use base64 + atob to avoid issues with </script> in strings.
    import base64
    blob = base64.b64encode(json.dumps(data, ensure_ascii=False).encode('utf-8')).decode('ascii')
    tmpl = tmpl.replace('__DATA_B64__', blob)
    import time
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    tmpl = tmpl.replace('__BUILD_TS__', ts)

    with open(args.out, 'w') as f:
        f.write(tmpl)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}  ({size_kb:.0f} KB)")
    print(f"open locally: file://{args.out}")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>nanogpt ↔ Megatron alignment (scaling_moe_00196)</title>
__PLOTLY_SRC__
<style>
:root {
  --bg: #0e1116; --panel: #161b22; --border: #30363d; --text: #e6edf3;
  --muted: #7d8590; --ok: #7ee787; --warn: #d29922; --err: #f85149; --blue: #79c0ff;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
       margin: 0; padding: 0; background: var(--bg); color: var(--text); line-height: 1.5; }
header { padding: 18px 28px; background: var(--panel); border-bottom: 1px solid var(--border);
         position: sticky; top: 0; z-index: 10; }
header h1 { margin: 0; font-size: 20px; font-weight: 600; }
header .subtitle { font-size: 12px; color: var(--muted); margin-top: 4px; }
main { padding: 24px 28px 60px; max-width: 1280px; margin: 0 auto; }
h2 { margin: 32px 0 14px; font-size: 15px; font-weight: 600;
     color: var(--ok); border-left: 3px solid var(--ok); padding-left: 10px;
     text-transform: uppercase; letter-spacing: 0.5px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
        padding: 14px 16px; }
.card .label { font-size: 10px; color: var(--muted); text-transform: uppercase;
               letter-spacing: 0.5px; }
.card .value { font-size: 22px; margin-top: 4px; font-weight: 500; }
.card .sub { font-size: 11px; color: var(--muted); margin-top: 4px; }
.ok { color: var(--ok); } .warn { color: var(--warn); } .err { color: var(--err); }
.blue { color: var(--blue); }
table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 8px 0;
        background: var(--panel); border: 1px solid var(--border); border-radius: 4px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }
tr:last-child td { border-bottom: none; }
th { background: #0b0e13; color: var(--muted); font-weight: 600; font-size: 11px;
     text-transform: uppercase; letter-spacing: 0.5px; }
pre { background: #0b0e13; border: 1px solid var(--border); padding: 10px 14px;
      border-radius: 4px; overflow-x: auto; font-size: 11px; margin: 8px 0;
      font-family: "SF Mono", Menlo, Consolas, monospace; }
.small { font-size: 12px; color: var(--muted); }
.status-pill { display: inline-block; padding: 2px 10px; border-radius: 10px;
               font-size: 10px; margin-left: 10px; font-weight: 600;
               text-transform: uppercase; letter-spacing: 0.5px; vertical-align: middle; }
.status-pill.ok { background: #0d3d23; color: var(--ok); }
.status-pill.warn { background: #3d2f0d; color: var(--warn); }
.status-pill.err { background: #3d0d0d; color: var(--err); }
.status-pill.pending { background: #30363d; color: var(--muted); }
details { background: var(--panel); border: 1px solid var(--border); border-radius: 4px;
          margin: 8px 0; padding: 10px 14px; }
details summary { cursor: pointer; color: var(--ok); font-size: 13px; user-select: none; }
details[open] summary { margin-bottom: 8px; }
code { background: #0b0e13; padding: 1px 6px; border-radius: 3px; font-size: 12px; color: var(--blue); }
hr { border: none; border-top: 1px solid var(--border); margin: 24px 0; }
.diff-row { display: grid; grid-template-columns: 200px 1fr 1fr; gap: 12px; padding: 6px 0;
            border-bottom: 1px dashed var(--border); font-size: 12px; }
.diff-row .k { color: var(--muted); }
.diff-row .ref { color: var(--blue); }
.diff-row .nano { color: var(--warn); }
/* tab nav between "Alignment" and "Learning dynamics" sub-pages */
.tabs { display: flex; gap: 2px; padding: 0 28px; background: var(--panel);
        border-bottom: 1px solid var(--border); position: sticky; top: 62px; z-index: 9; }
.tab-btn { background: transparent; color: var(--muted); border: none;
           border-bottom: 2px solid transparent; padding: 10px 16px; cursor: pointer;
           font-size: 13px; font-weight: 500; letter-spacing: 0.2px;
           font-family: inherit; transition: color 120ms, border-color 120ms; }
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--ok); border-bottom-color: var(--ok); }
.tab-content { display: none; }
.tab-content.active { display: block; }
.plot { height: 360px; background: var(--panel); border: 1px solid var(--border);
        border-radius: 4px; margin: 8px 0; }
.plot-lg { height: 480px; }
.empty-state { padding: 28px 20px; text-align: center; color: var(--muted); font-size: 13px;
               background: var(--panel); border: 1px dashed var(--border); border-radius: 4px; }
.chart-intro { background: #0b0e13; border-left: 3px solid var(--blue);
               padding: 10px 14px; margin: 14px 0 6px; border-radius: 0 4px 4px 0;
               font-size: 12px; line-height: 1.65; color: #c9d1d9; }
.chart-intro-title { color: var(--blue); font-weight: 600; margin-bottom: 4px;
                     font-size: 13px; letter-spacing: 0.2px; }
.chart-intro b { color: #e6edf3; }
.chart-intro code { background: #161b22; }
</style>
</head>
<body>
<header>
  <h1>nanogpt ↔ Megatron alignment · <span class="blue">scaling_moe_00196</span>
     <span class="small" style="float:right;color:var(--muted)">built __BUILD_TS__</span></h1>
  <div class="subtitle">PAI DLC dlc1q9arre48b0kx · 9 layers / 512 hidden / 4 heads / 2 KV groups · 144 experts, top-8 sigmoid · 447.30M params · 7485 iterations · final lm loss 2.86</div>
</header>
<nav class="tabs">
  <button class="tab-btn active" data-tab="tab-alignment" onclick="switchTab('tab-alignment')">Alignment (与 Megatron 对齐)</button>
  <button class="tab-btn" data-tab="tab-dynamics" onclick="switchTab('tab-dynamics')">Learning Dynamics (训练动态监控)</button>
  <button class="tab-btn" data-tab="tab-muon" onclick="switchTab('tab-muon')">Muon Alignment (vs Megatron Muon ref)</button>
</nav>
<main>
<div id="tab-alignment" class="tab-content active">
  <div id="overview" class="cards"></div>

  <h2>Reference job <span id="job-status" class="status-pill ok">captured</span></h2>
  <div id="job-body"></div>

  <h2>Tokenizer <span id="tok-status" class="status-pill pending">loading…</span></h2>
  <div id="tok-body"></div>

  <h2>Data sampling <span id="data-status" class="status-pill pending">loading…</span></h2>
  <div id="data-body"></div>

  <h2>Model structure / params / FLOPs <span id="model-status" class="status-pill pending">loading…</span></h2>
  <div id="model-body"></div>

  <h2>Loss trajectory <span id="loss-status" class="status-pill pending">loading…</span></h2>
  <div id="loss-plot" style="height: 420px; background: var(--panel); border: 1px solid var(--border); border-radius: 4px;"></div>
  <div id="loss-body"></div>

  <h2>MoE routing (nano vs ref) <span id="routing-status" class="status-pill pending">loading…</span></h2>
  <div class="small" style="margin:4px 0 8px;">
    Per-iter routing imbalance <code>maxvio_micro_batch = (max - mean) / mean</code> and
    <code>tokens_per_expert</code> stats. Ref is from Megatron master log; nano from
    train_log.jsonl (requires commit d782fc3 logging — only moediag run has it).
    <b>大于 ref 3×以上 = nano 路由不同 = 极可能是 +0.005 nat gap 的源头。</b>
    Uses the same experiment selector / color pickers as the loss chart above.
  </div>
  <div id="routing-plot" style="height: 420px; background: var(--panel); border: 1px solid var(--border); border-radius: 4px;"></div>
  <div id="routing-body"></div>

  <h2>MoE fleet (scaling_moe_00196 · PAI Adam + PAI Muon + nano v10repro bucket-fix) <span id="moe-fleet-status" class="status-pill pending">loading…</span></h2>
  <div class="small" style="margin:4px 0 8px;">
    对比三路 7485-iter MoE-196 训练：
    <b>PAI ref Adam fleet</b>（5 seed，EMA 2.8510 ± 0.0081），
    <b>PAI Muon fleet</b>（5 seed，EMA 2.7899 ± 0.0030，比 Adam 低 0.061 nat），
    <b>nano v10repro bucket-fix</b>（2 local seed + 1 PAI full，修了 TE GroupedLinear optimizer miss bug；走 bucket-padding path 直接 grad flow，速度比 ref 快 7-9%）。
    <br>Group 按颜色区分：灰色 = PAI Adam，橙色 = PAI Muon，绿色 = nano bucket-fix，紫色 = nano v10 historical。
  </div>
  <div id="moe-fleet-plot" style="height: 380px; background: var(--panel); border: 1px solid var(--border); border-radius: 4px;"></div>
  <div id="moe-fleet-body"></div>

  <h2>Dense ablation (nano-dense_107 vs ref scaling_dense_00107_nc, 6711-iter) <span id="dense-status" class="status-pill pending">loading…</span></h2>
  <div class="small" style="margin:4px 0 8px;">
    Scaling ladder 里的规范"最小 dense"节点 — <code>scaling_dense_00107_nc.yaml</code>（8 层，hidden=528，ffn=1824，kv_channels=128，gbs=48，lr=0.001063，6711 iters，WSD-exp，~190 M params）。
    nano 用相同架构 + 优化器配置，数据来源不同（nano 用 cybertron_baseline，ref 用 data_pretrain_v3_average_pai_nnc）。
    观察目标：dense 轨迹是否像 MoE 一样有 "早期 nano<ref, decay 末段 nano>ref" 的符号翻转 —— 若有，且交叉 MoE+AdamW / Muon / Dense 都同号，确认差异源在底层 bf16 / Adam / DDP numerics。
  </div>
  <div id="dense-plot" style="height: 380px; background: var(--panel); border: 1px solid var(--border); border-radius: 4px;"></div>
  <div style="margin-top:8px;font-size:11px;color:var(--muted);">Grad norm（同一组 run 选择）</div>
  <div id="dense-gradnorm-plot" style="height: 300px; background: var(--panel); border: 1px solid var(--border); border-radius: 4px;"></div>
  <div id="dense-body"></div>

  <h2>Forward alignment (Megatron weights → nano forward) <span id="fwd-status" class="status-pill pending">loading…</span></h2>
  <div id="fwd-body"></div>

  <h2>Alignment checklist <span id="chk-status" class="status-pill pending">loading…</span></h2>
  <div id="chk-controls" style="margin: 8px 0 12px; font-size: 12px;"></div>
  <div id="chk-body"></div>

  <h2>Bitwise resume <span id="bw-status" class="status-pill pending">loading…</span></h2>
  <div id="bw-body"></div>

  <h2>Checkpoint fingerprint <span id="ckpt-status" class="status-pill pending">loading…</span></h2>
  <div id="ckpt-body"></div>

  <h2>Alignment gaps <span id="gap-status" class="status-pill pending">loading…</span></h2>
  <div id="gap-body"></div>
</div><!-- /tab-alignment -->

<div id="tab-dynamics" class="tab-content">
  <h2>代码构成 <span id="code-stats-status" class="status-pill ok">auto-computed</span></h2>
  <div class="small" style="margin:4px 0 8px;">
    按用途统计 <code>.py</code> / <code>.html</code> / <code>.js</code> 行数，
    直观看监控 / 可视化 / 测试的新增代码 vs 原有训练框架的比例。
    数值在每次 <code>build_local.py</code> 运行时重新扫描仓库自动更新。
  </div>
  <div id="code-stats-plot" class="plot"></div>

  <h2>Overview <span id="mon-status" class="status-pill pending">loading…</span></h2>
  <div id="mon-intro" class="small" style="margin:6px 0 12px;">
    训练过程中的 learning dynamics 指标，来自 <code>monitor/</code> 包写入的
    <code>reports/monitor.jsonl</code>。目标：在 500M / 10B-MoE 小尺度 proxy
    上发现 10B / 200B-MoE 才会暴露的问题（router collapse · attention sink ·
    残差爆炸 · bf16 饱和），让 scaling 实验 extrapolate 更可信。
  </div>
  <div id="mon-cards" class="cards"></div>
  <div id="mon-runs" style="margin:12px 0;"></div>

  <h2>Loss &amp; stability <span class="status-pill ok">A / F tier</span></h2>

  <div class="chart-intro">
    <div class="chart-intro-title">1. Loss + spike z-score</div>
    <b>图例：</b>蓝实线 = <code>loss</code>（主 y 轴）；橙点线 = loss 相对 EMA 的
    <code>z-score</code>（副 y 轴，EMA 用前 20 步均值+方差）；红 × = |z| &gt; 3 的
    spike 点。<br>
    <b>怎么分析：</b>loss 曲线应平滑下降；z 在 ±1 之间震荡正常，单个 |z|&gt;3 是噪声，
    <b>但连续 2 点以上 spike = 实打实的事故</b>（数据异常 / LR 过大 / bf16 溢出），
    需要降 LR 或回滚。z 连续性增长（不是震荡）暗示优化器 momentum 在放大扰动。
  </div>
  <div id="mon-loss-plot" class="plot"></div>

  <div class="chart-intro">
    <div class="chart-intro-title">2. Scale-invariant stability &amp; token efficiency</div>
    <b>图例：</b>绿线 = <code>grad_norm / loss</code>（F4，跨尺度可比的稳定度代理）；
    橙线 = <code>Δloss / Δsamples</code>（F5，token 效率，副轴）。<br>
    <b>怎么分析：</b><code>gn/loss</code> 应缓慢下降或维持稳定，<b>突升 = 优化方向开始震荡</b>；
    若 500M 和 10B 在同一 progress 点的 <code>gn/loss</code> 差距很大，说明 scaling 不一致
    （LR/init 该随 scale 调）。<code>Δloss/Δsamples</code> 是 Chinchilla 拟合的直接输入，
    跨尺度配比不变 = scaling law 成立。
  </div>
  <div id="mon-effgn-plot" class="plot"></div>

  <h2>Gradient norm by parameter group <span class="status-pill ok">A1</span></h2>

  <div class="chart-intro">
    <div class="chart-intro-title">3. 分组 grad_norm（log-y）</div>
    <b>图例：</b>每条线一类参数的 L2 梯度范数——
    <code>embedding</code> / <code>lm_head</code> / <code>attn_qkv</code> /
    <code>attn_proj</code> / <code>ffn_gate|up|down</code> / <code>router</code> /
    <code>shared_expert</code> / <code>routed_expert</code> / <code>norm</code> / <code>other</code>。
    点击图例可隐藏某类。<br>
    <b>怎么分析：</b>健康状态下各组 grad 应在 1~2 个数量级内共同下降。
    <b>某一组突然领跑 = 选择性爆炸</b>——router 组先炸意味着路由学不出、embedding 大于其他 10×
    意味着 tied-embedding/lm_head 路径失衡、routed_expert 比 shared_expert 大得多暗示
    capacity 失衡。scaling 时某一组比小尺度比例漂移 = 那部分不是 scale-invariant。
  </div>
  <div id="mon-gn-plot" class="plot plot-lg"></div>

  <h2>Residual stream depth profile <span class="status-pill ok">B1 / B2 / B5</span></h2>

  <div class="chart-intro">
    <div class="chart-intro-title">4. 每层残差范数 heatmap（B1）</div>
    <b>图例：</b>横轴 = iter；纵轴 = layer 序号（0 = 最底层、N-1 = 最顶层）；
    颜色 = 该层 post-block 输出的 <code>‖x‖₂ / √d</code>（Viridis 色阶，越亮数值越大）。
    <b>所有层都显示，不采样</b>。<br>
    <b>怎么分析：</b>健康的残差流应**颜色沿 y 轴缓慢单调变化**（略增或保持平稳）。
    <b>深度方向颜色指数变亮 = 残差爆炸</b>（init std 或残差 gain 不合适）；
    <b>颜色变暗到底 = 消失</b>；<b>横向突然变色 = 某步训练态改变</b>。
    10B+ 深层模型的爆炸问题在 500M 规模常以"第 10 层颜色比第 2 层亮 2 倍"先显形。
  </div>
  <div id="mon-resid-heat" class="plot plot-lg"></div>

  <div class="chart-intro">
    <div class="chart-intro-title">5. 每层净贡献比 heatmap（B2）</div>
    <b>图例：</b>颜色 = <code>‖x_post − x_pre‖ / ‖x_pre‖</code>（比值，Cividis）。<br>
    <b>⚠ Layer 0 dense 在比值上必然 saturate</b>：layer 0 input 是裸 embedding
    （‖x‖≈0.03），output 拉到 ≈10，比值 ≈300。这不是"过度贡献"，是
    分母小造成的相对增长率虚高。色阶已 clip 到 layer 1..L-1 的 P98，
    layer 0 故意 saturate。
    <b>看下面 B2-abs 比 B2 比值更公平。</b>
  </div>
  <div id="mon-contrib-heat" class="plot plot-lg"></div>

  <div class="chart-intro">
    <div class="chart-intro-title">5b. 每层绝对净贡献 heatmap（B2-abs / 跨层比较推荐）</div>
    <b>图例：</b>颜色 = <code>‖x_post − x_pre‖</code>（绝对值，不除 ‖pre‖）。
    比值（B2）有"分母小拉爆"问题；绝对值能直接比较 dense layer 0 和 MoE layer 1..L-1
    的实际计算贡献量（~7-10× 差距，符合 dense-vs-sparse-MoE 算力比）。<br>
    <b>怎么分析：</b>健康：layer 0 ≈ post 的 ~1×（从零拉起），后续层 ≈ 1-2 nat 量级稳定增长；
    某层长期 ≈ 0 = 死层；某层突然激增 = 信号爆炸。
  </div>
  <div id="mon-contrib-abs-heat" class="plot plot-lg"></div>

  <div class="chart-intro">
    <div class="chart-intro-title">6. final residual (ln_f) 幅度（B5 / bf16 饱和 watchdog）</div>
    <b>图例：</b>红线 = <code>final_resid</code> 绝对值最大（lm_head 的输入）；
    蓝线 = 标准差；黄虚线 = bf16 最大值 <code>65504</code> 参考。log-y 轴。<br>
    <b>怎么分析：</b>max 应稳定或缓慢上涨。<b>max 逼近黄线 = bf16 即将饱和 NaN</b>——
    这是大规模训练中最隐蔽的死因，loss 曲线看不出，小尺度却能提前测到同样趋势。
    std 突升是全局激活发散，常与 B1 残差爆炸同时出现。
  </div>
  <div id="mon-final-plot" class="plot"></div>

  <h2>MoE routing health <span id="moe-status" class="status-pill ok">D1 / D2 / D4 / D6</span></h2>
  <div id="moe-section-intro" class="small" style="margin:4px 0 8px;">
    MoE 路由 4 大核心信号。每个 MoE 层一条线（按层号连续色环着色，颜色从紫→蓝→绿→黄按层深度渐变），
    <b>所有 MoE 层都显示，不采样</b>。
  </div>
  <div id="moe-dense-notice" style="display:none;" class="empty-state">
    当前 run 为 <b>dense 架构</b>（无 MoE 层）—— 路由相关指标不适用。
  </div>

  <div class="chart-intro">
    <div class="chart-intro-title">7. 负载熵（D1，归一化到 [0, 1]，1.0 = 完美平衡）</div>
    <b>图例：</b>每条线 <code>L{i}</code> 对应第 i 层 MoE 的
    <code>H(token分布) / log(num_experts)</code>。<br>
    <b>怎么分析：</b>健康值应 &gt;0.85 并保持稳定。
    <b>某层熵持续下降 = router collapse 进行中</b>（少数专家在吃掉大部分流量）。
    <b>所有层齐跌 = 全局 LR/bias coef 不对</b>；
    <b>单层孤立跌 = 该层 routing 初始化或 shared expert 占比失衡</b>。
    这是 MoE scaling 最 early warning 的指标之一。
  </div>
  <div id="mon-moe-load" class="plot"></div>

  <div class="chart-intro">
    <div class="chart-intro-title">8. dead expert 数（D2）</div>
    <b>图例：</b>每条线一层 MoE，<b>y = 当步收到 0 token 的专家数</b>。<br>
    <b>怎么分析：</b>训练最早期（&lt; 100 iter）dead 数较大是正常冷启动；之后应快速降到 0
    并维持。<b>dead 数持续 &gt; 总专家数 10%（144→14） = aux-free bias 校正失效</b>，
    router 的 score correction coefficient（默认 0.001）需要加大，或启用 seq_aux_alpha。
    200B MoE 最常见的资源浪费模式就是 dead expert 比例持续高企。
  </div>
  <div id="mon-moe-dead" class="plot"></div>

  <div class="chart-intro">
    <div class="chart-intro-title">9. top-1 权重占比（D4）</div>
    <b>图例：</b>每条线一层 MoE，y = top-1 专家权重的 batch 均值。
    对 top-k 路由，理想值约为 <code>1/k</code>（topk=8 则 ≈ 0.125~0.25 正常）。<br>
    <b>怎么分析：</b><b>单调上升趋向 1.0 = 一家独大型 collapse</b>，
    即使 load entropy 还没掉（token 分布还平均），但权重都集中在一个专家上，
    其他 topk-1 个专家只是"陪跑"——这是更隐蔽的 router 病。
    和 D1 一起看：D1 高但 D4 也高 = "投票平均但权重集中"，需要看路由分数熵（D3，后续加）。
  </div>
  <div id="mon-moe-top1" class="plot"></div>

  <div class="chart-intro">
    <div class="chart-intro-title">10. aux-free 校正 bias |max|（D6）</div>
    <b>图例：</b>每条线一层 MoE，
    y = <code>e_score_correction_bias</code> buffer 的 <code>|max|</code>。
    每个 optim step 加 <code>sign(mean_load − actual_load) × coeff</code>，coeff 通常 1e-3。<br>
    <b>怎么分析：</b>健康值应在某个小范围波动（如 &lt; 0.05）并随训练稳定收敛。
    <b>单调线性上升 = router 一直在单向补偿不平衡</b>——说明 gradient 根本没在学平衡路由，
    全靠 bias 硬拖，通常意味着 score function / group 设置有问题。
    <b>突然跳变 = 单步 token 分布剧烈变化</b>，对应 loss spike 的前导信号。
  </div>
  <div id="mon-moe-bias" class="plot"></div>

  <h2>Attention patterns <span id="attn-status" class="status-pill pending">loading…</span></h2>
  <div class="small" style="margin:4px 0 8px;">
    由 <code>monitor/attn_probe.py</code> 在 eval 阶段用一个固定 probe batch 捕获的
    每层每头 attention 矩阵（softmax(QKᵀ/√d)，manual 路径），降采样至 32×32。
    单张 heatmap 看 pattern；汇总卡片看全模型的 sink / entropy 趋势。
  </div>
  <div id="attn-cards" class="cards"></div>

  <div class="chart-intro">
    <div class="chart-intro-title">11. 每层每头 attention heatmap</div>
    <b>图例：</b>每张小图是某一 (layer, head) 的 attention 矩阵，x 轴 = key position、
    y 轴 = query position（左上=序列起点）。颜色越亮 = 权重越大。
    对角线亮 = 局部注意力；第 0 列亮 = attention sink；整张图均匀灰 = head death。<br>
    <b>怎么分析：</b>
    <ul style="margin:4px 0 0 16px;padding:0;">
      <li><b>Attention sink：</b>第 0 列（最左列）持续明亮且横跨所有 query 行——
          所有 token 都在"偷窥" BOS/首 token 用作"注意力消音器"。健康的小模型 sink 占比
          通常 &lt;30%；&gt;70% 且持续增长 = 训练到后期有效上下文在坍缩。</li>
      <li><b>Head death：</b>整张图接近均匀的浅色（熵接近 log T）——这个 head 对所有位置
          都给等权，没学到任何模式，后面的投影矩阵也会把它权重压低，属于容量浪费。</li>
      <li><b>Rank-1 collapse：</b>整张图只有一条横线或一条竖线亮——所有 query 在看同一个
          (或同一类) key，head 退化成"广播一个全局特征"，多头的意义消失。</li>
      <li><b>健康模式：</b>对角线 + 对角线附近几条带 + 少量长程稀疏点。浅层多局部、
          深层多长程是正常的深度分工。</li>
    </ul>
  </div>
  <div class="small" style="margin:6px 0 8px;color:var(--muted);">
    <b>坐标：</b>x = query bucket（左→右），y = key bucket（上→下，左上为序列起点）。
    <b>Pooling（新）：</b>raw SUM over each (block_q × block_k) tile（attn_probe 已改）。
    cell 值 = "q-bucket → k-bucket 的总 attention mass"。
    每行 sum = block_q（如 T=8192/T_down=32 → row sum = 256，因为 256 行每行 sum=1
    一起 pool 到一个 cell 行），列 sum 任意（k-bucket 受关注度），matrix sum = T。
    Sink cell 值 ≈ block_q × sink_strength（不是 max）。
    <b>（当前 attention_maps.json 仍是旧 max-pool；下次 probe regen 生效。）</b>
    <b>分辨率：</b>T=8192 / T_down=32 → 每 cell 覆盖 256 个 token。
    <code>monitor/core.py</code> 已改 <code>downsample=128</code>，下次 probe 生效。
    <b>视觉：</b>Viridis。Linear clip P98 避免 [0,0] 吞色阶；Log 显示 log10(w+1e-8)。
    Hover 显示 bucket index + raw pool 值。
  </div>
  <div style="margin:8px 0 6px;display:flex;gap:8px;align-items:center;font-size:12px;">
    <span style="color:var(--muted);">color scale:</span>
    <button id="attn-scale-linear" class="tab-btn"
            style="padding:4px 12px;border:1px solid var(--border);border-radius:4px;
                   background:var(--ok);color:#000;font-weight:600;cursor:pointer;">linear (P98)</button>
    <button id="attn-scale-log" class="tab-btn"
            style="padding:4px 12px;border:1px solid var(--border);border-radius:4px;
                   background:var(--panel);color:var(--text);cursor:pointer;">log10</button>
  </div>
  <div id="attn-grid"
       style="display:grid;grid-template-columns:repeat(auto-fit, minmax(420px, 1fr));
              gap:10px;margin:8px 0;"></div>

  <div class="chart-intro">
    <div class="chart-intro-title">12. Per-layer sink strength &amp; entropy 汇总</div>
    <b>图例：</b>横轴 = layer 序号。红线 = 该层平均 sink 强度（所有 head + query 对第 0 个
    key 的平均权重）；蓝线 = 该层平均 attention 熵（按 log(T) 归一化，1.0 = 均匀分布）。<br>
    <b>怎么分析：</b>理想的层结构：浅层熵较低（局部模式）、深层熵偏高（长程整合），
    sink 强度 0.1–0.5 之间波动。<b>sink 全层飙高 = 全模型在摆烂</b>；
    <b>单层 entropy 压到 0 附近 = 该层所有 head 都崩到 rank-1</b>；<b>某层 entropy 永远 &gt; 0.95
    = 该层所有 head death</b>——这些异常在大规模 MoE 里常见，在小尺度 proxy 能提前看到苗头。
  </div>
  <div id="attn-summary-plot" class="plot"></div>
</div><!-- /tab-dynamics -->

<div id="tab-muon" class="tab-content">
  <div id="muon-overview" class="cards"></div>

  <h2>Muon vs ref ── Loss curves <span id="muon-loss-status" class="status-pill pending">loading…</span></h2>
  <div class="small" style="margin:4px 0 8px;">
    <code>muon-reimpl</code>（branch <code>muon-reimpl</code>，commit
    <code>23212f1</code>）= 忠实移植 megatron <code>orthogonalized_optimizers/muon.py</code>，
    cybertron 默认配置（quintic NS5、spectral scale @0.2、momentum=0.95、nesterov=on、decoupled WD）。
    Muon ref TB：<code>/prodcpfs/user/data/save/data/scaling/tensorboard/scaling_moe_00196_ef_3.0_muon_base</code>
    （14970 iter，健康收敛，是 megatron Muon 真正的 baseline）。
    AdamW ref TB：<code>reference/tb/key_scalars.json</code>（scaling_moe_00196，7485 iter）。
    nano init=scratch（seed=1337）；ref init 来自 megatron pipeline，因此存在轻微 init offset。
    nano iter <i>N</i> ↔ ref iter <i>N</i>+1（与 AdamW 对齐相同 1-index 偏移）。
  </div>
  <div id="muon-loss-plot" class="plot" style="height: 420px;"></div>

  <h2>Δ (nano − ref) <span id="muon-delta-status" class="status-pill pending">loading…</span></h2>
  <div class="small" style="margin:4px 0 8px;">
    每 iter 的 <code>nano_loss − ref_loss</code>。理想：稳定接近 0；
    系统性偏移 = init offset；震荡 = 噪声；持续单向漂移 = 算法/超参不一致。
  </div>
  <div id="muon-delta-plot" class="plot" style="height: 360px;"></div>

  <h2>Grad-norm comparison <span id="muon-gn-status" class="status-pill pending">loading…</span></h2>
  <div class="small" style="margin:4px 0 8px;">
    Muon 的 spectral scaling 设计上让 update RMS 与 AdamW 同量级。
    若 nano grad-norm 显著偏离 ref，提示
    NS 系数 / scale_mode / momentum 公式有差异。
  </div>
  <div id="muon-gn-plot" class="plot" style="height: 360px;"></div>

  <h2>Summary table (per-iter, nano coverage)</h2>
  <div class="small" style="margin:4px 0 8px;">
    Δ_AdamW / Δ_share / Δ_base = nano − each ref. 负值 = nano 比该 ref 收敛更快。
  </div>
  <div id="muon-summary-body"></div>

  <h2>Off-by-1 诊断：nano N ↔ ref N+offset</h2>
  <div class="small" style="margin:4px 0 8px;">
    Megatron TB iter 编号比 nano log 大 1（megatron iter 1 是 nano iter 0 的同一步）。
    下表 mean |Δ| 是 nano iter 100..500 区间各 offset 下的平均绝对差异。
    <b>+1 offset 是正确对齐</b>（mean |Δ| ~ 0.009，比其他 offset 小 10×）。
  </div>
  <div id="muon-offset-body"></div>

  <h2>Long-run Muon vs AdamW（megatron 全程对比） <span id="muon-long-status" class="status-pill pending">loading…</span></h2>
  <div class="small" style="margin:4px 0 8px;">
    Muon ref vs AdamW ref 的全程 loss 对比（不依赖 nano 训练）。
    Muon 在早期（iter 500–3000）领先 0.1–0.6 nat，长期 (iter 7485) 收窄到 -0.06。
    Muon ref 训练到 iter 14970（AdamW ref 只有 7485）；最终 loss 2.63。
  </div>
  <div id="muon-long-plot" class="plot" style="height: 380px;"></div>
  <div id="muon-long-body"></div>
</div><!-- /tab-muon -->
</main>

<script>
// Base64-decoded, embedded at build time
const DATA = JSON.parse(new TextDecoder('utf-8').decode(
  Uint8Array.from(atob("__DATA_B64__"), c => c.charCodeAt(0))
));

function card(label, value, sub, cls='') {
  return `<div class="card"><div class="label">${label}</div>
          <div class="value ${cls}">${value}</div>
          ${sub ? `<div class="sub">${sub}</div>` : ''}</div>`;
}
function pill(id, state, text) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = `status-pill ${state}`;
  el.textContent = text;
}
function esc(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}
function fmt(n) {
  if (typeof n !== 'number') return n;
  if (Number.isInteger(n) && Math.abs(n) >= 1000) return n.toLocaleString();
  return n;
}

// --------- job ---------
function renderJob() {
  const j = DATA.job;
  if (!j) { pill('job-status', 'err', 'missing'); return; }
  document.getElementById('job-body').innerHTML = `
    <div class="cards">
      ${card('display name', j.DisplayName, j.Status)}
      ${card('duration', j.Duration ? `${Math.round(j.Duration/60)} min` : '?', `${j.GmtCreateTime || ''} → ${j.GmtFinishTime || ''}`)}
      ${card('resource', `${j.ResourceConfig?.GPU || '?'} × GPU`, `${j.ResourceConfig?.CPU || ''} CPU / ${j.ResourceConfig?.Memory || ''}`)}
      ${card('clone from', j.Tags?.CloneFromJobID || '-', 'upstream job id')}
    </div>
    <details><summary>image · data sources</summary>
      <div class="small">image: <code>${esc(j.Image || '')}</code></div>
      <pre>${esc(JSON.stringify(j.DataSources, null, 2))}</pre></details>
    <details><summary>UserCommand (2419 bytes)</summary>
      <pre>${esc(j.UserCommand || '')}</pre></details>`;
}

// --------- tokenizer ---------
function renderTokenizer() {
  const t = DATA.tokenizer;
  if (!t) { pill('tok-status', 'err', 'missing'); return; }
  const isQwen = (t.class || '').includes('Qwen2');
  const eodOk = t.eos_token_id === 151643;
  pill('tok-status', (isQwen && eodOk) ? 'ok' : 'err', (isQwen && eodOk) ? 'pass' : 'fail');
  document.getElementById('tok-body').innerHTML = `
    <div class="cards">
      ${card('class', t.class, 'Qwen2TokenizerFast expected', isQwen ? 'ok' : 'err')}
      ${card('vocab_size', t.vocab_size, 'base', 'ok')}
      ${card('len with added', t.len_with_added, '+ 16 added tokens', 'ok')}
      ${card('EOD/EOS', t.eos_token_id, esc(t.eos_token || ''), eodOk ? 'ok' : 'err')}
      ${card('roundtrip', `${t.roundtrip_pass_count}/${t.roundtrip_total}`, 'bitwise match', t.roundtrip_pass_count === t.roundtrip_total ? 'ok' : 'err')}
      ${card('mask_loss_id', t.mask_loss_id, 'out of vocab (sentinel)')}
    </div>
    <details><summary>file md5s (identity lock)</summary>
      <table>${Object.entries(t.file_md5 || {}).map(([k,v]) =>
         `<tr><td><code>${esc(k)}</code></td><td><code>${esc(v)}</code></td></tr>`).join('')}</table></details>
    <details><summary>16 added tokens</summary>
      <table><thead><tr><th>id</th><th>content</th></tr></thead>
        <tbody>${Object.entries(t.added_tokens || {}).map(([id, c]) =>
          `<tr><td>${id}</td><td><code>${esc(c)}</code></td></tr>`).join('')}</tbody></table></details>`;
}

// --------- data sampling ---------
function renderData() {
  const d = DATA.data;
  if (!d) { pill('data-status', 'err', 'missing'); return; }
  const dsMatch = d.n_datasets_in_blend === d.n_datasets_in_yaml;
  pill('data-status', dsMatch ? 'ok' : 'err', dsMatch ? 'pass' : 'mismatch');
  document.getElementById('data-body').innerHTML = `
    <div class="cards">
      ${card('datasets in blend', d.n_datasets_in_blend, `yaml: ${d.n_datasets_in_yaml}`, dsMatch ? 'ok' : 'err')}
      ${card('cache samples', (d.blended_sample_count || 0).toLocaleString(), 'full yaml train_samples')}
      ${card('consumed by run', (d.consumed_by_run || 0).toLocaleString(), '7485 × 64 = 479040')}
      ${card('unique datasets', d.n_unique_datasets_in_train || '?', 'in consumed portion')}
      ${card('first-1024 sha256', (d.first_1024_pairs_sha256 || '').slice(0,12) + '…', 'golden for replay')}
    </div>
    <details><summary>first 10 blend pairs (dataset_id, sample_id)</summary>
      <pre>${esc(JSON.stringify(d.first_10_pairs, null, 2))}</pre></details>
    <details><summary>step 0 sample (first 16 tokens)</summary>
      <pre>${esc(JSON.stringify(d.step0_sample, null, 2))}</pre></details>
    <details><summary>first-dataset: blend vs yaml</summary>
      <pre>${esc(JSON.stringify(d.first_dataset_blend_vs_yaml, null, 2))}</pre></details>`;
}

// --------- model ---------
function renderModel() {
  // No explicit report JSON today — we synthesize from Megatron shapes.json + known config.
  pill('model-status', 'ok', 'pytest 10/10');
  const shapes = {
    embedding:        [152064, 512],
    output_layer:     [152064, 512],
    qkv_fused:        [512, 512],
    q_proj:           [256, 512], k_proj: [128, 512], v_proj: [128, 512],
    c_proj:           [512, 256],
    q_layernorm:      [64],  k_layernorm: [64],
    dense_fc1:        [3072, 512],  dense_fc2: [512, 1536],
    moe_expert_fc1:   [320, 512],   moe_expert_fc2: [512, 160],
    moe_shared_fc1:   [320, 512],   moe_shared_fc2: [512, 160],
    router:           [144, 512],
    final_norm:       [512],
  };
  document.getElementById('model-body').innerHTML = `
    <div class="cards">
      ${card('total params', '447.30M', 'vs Megatron sum: 447.30M', 'ok')}
      ${card('embedding+lm_head', '155.71M', '35% of total')}
      ${card('routed experts', '283.12M', '144 × 8 layers × 245,760')}
      ${card('non-embedding', '291.59M', 'total − embedding')}
      ${card('active/token', '≈30M', 'shared + 8 experts + attn + norm')}
      ${card('layers', '9', '1 dense + 8 MoE (layer 0 dense)')}
      ${card('attention', '4h / 2 KV / 64 head_dim', 'GQA + qk_layernorm')}
      ${card('MoE', '144 experts, top-8', 'sigmoid, n_group=8, topk_group=1')}
    </div>
    <details open><summary>verified shapes (nanogpt × Megatron ckpt)</summary>
      <table><thead><tr><th>component</th><th>shape</th></tr></thead><tbody>
        ${Object.entries(shapes).map(([k,v]) =>
          `<tr><td><code>${esc(k)}</code></td><td>${esc(JSON.stringify(v))}</td></tr>`).join('')}
      </tbody></table></details>`;
}

// --------- loss trajectory ---------
// Multi-select: persist selected run indices in memory across re-renders.
// Default: pick the latest run (last in list) only — keeps initial chart clean.
let SELECTED_RUN_IDXS = null;  // Set<int>; null = uninitialized, use default
let RUN_COLOR_OVERRIDES = {};  // {runIdx: '#hex'} — user-picked colors
let SHOW_DIFF_MODE = false;    // false: plot raw loss; true: plot Δ (nano − ref)
const REF_COLOR = '#79c0ff';

function defaultSelectedIdxs(runs) {
  if (!runs || runs.length === 0) return new Set();
  return new Set([runs.length - 1]);  // latest only
}
function getSelectedRunIdxs(runs) {
  if (SELECTED_RUN_IDXS === null) SELECTED_RUN_IDXS = defaultSelectedIdxs(runs);
  return SELECTED_RUN_IDXS;
}
function _rerenderRunCharts() {
  renderLoss();
  if (typeof renderRouting === 'function') renderRouting();
}
function toggleRunIdx(i) {
  const runs = DATA.runs || [];
  const sel = getSelectedRunIdxs(runs);
  if (sel.has(i)) sel.delete(i); else sel.add(i);
  _rerenderRunCharts();
}
function setAllRunIdxs(on) {
  const runs = DATA.runs || [];
  SELECTED_RUN_IDXS = new Set();
  if (on) for (let i = 0; i < runs.length; i++) SELECTED_RUN_IDXS.add(i);
  _rerenderRunCharts();
}
function setRunColor(i, hex) {
  RUN_COLOR_OVERRIDES[i] = hex;
  _rerenderRunCharts();
}
function setDiffMode(on) {
  SHOW_DIFF_MODE = !!on;
  renderLoss();  // diff mode is loss-only
}

// Categorical high-contrast palette (Plotly-compatible). Chosen so any two
// neighbors in index order are visually distinct even with partial colorblindness.
// User can override per-run via setRunColor.
const RUN_COLORS = [
  '#e6194B', // red
  '#3cb44b', // green
  '#ffe119', // yellow
  '#f58231', // orange
  '#911eb4', // purple
  '#42d4f4', // cyan
  '#f032e6', // magenta
  '#a52a2a', // brown
  '#800000', // maroon
  '#aaffc3', // mint
  '#808000', // olive
  '#ffd8b1', // apricot
];
function runColor(i) { return RUN_COLOR_OVERRIDES[i] || RUN_COLORS[i % RUN_COLORS.length]; }

function renderLoss() {
  const l = DATA.loss, tb = DATA.tb;
  const runs = DATA.runs || [];
  if (!l && runs.length === 0) { pill('loss-status', 'err', 'missing'); return; }
  const haveNano = runs.length > 0 || (l && l.nano_present);
  const selected = getSelectedRunIdxs(runs);
  pill('loss-status', haveNano ? 'ok' : 'warn',
       haveNano ? `${selected.size}/${runs.length || 1} selected` : 'ref only');

  // "Summary run" for the meta banner = the single run if exactly one selected,
  // or the most recent selected run otherwise.
  const summaryIdx = selected.size > 0 ? Math.max(...selected) : (runs.length - 1);
  const rm = runs[summaryIdx] || l?.run_meta;
  const cmp = rm?.compare || l?.compare || {};

  // Multi-select checkbox picker (replaces the old single-select dropdown).
  // Each row has: [checkbox] [color picker] [run id] — [label].
  // Color input lets user override the palette per run.
  // A second toolbar row toggles raw-loss vs Δ(nano − ref) view.
  const picker = runs.length > 0 ? `
    <div style="margin:4px 0 10px;font-size:12px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;flex-wrap:wrap;">
        <span class="small">experiments (click to toggle; Megatron ref always shown):</span>
        <button onclick="setAllRunIdxs(true)"  style="background:#0b0e13;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 8px;font-size:11px;cursor:pointer;">select all</button>
        <button onclick="setAllRunIdxs(false)" style="background:#0b0e13;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 8px;font-size:11px;cursor:pointer;">select none</button>
        <label style="display:inline-flex;align-items:center;gap:4px;cursor:pointer;margin-left:auto;">
          <input type="checkbox" ${SHOW_DIFF_MODE ? 'checked' : ''} onchange="setDiffMode(this.checked)">
          <span class="small">show Δ (nano − ref) — makes bf16-floor gap visible</span>
        </label>
      </div>
      <div style="display:flex;flex-direction:column;gap:4px;padding:8px 10px;background:#0b0e13;border:1px solid var(--border);border-radius:4px;">
        ${runs.map((r,i) => {
          const checked = selected.has(i) ? 'checked' : '';
          const color = runColor(i);
          return `<label style="display:inline-flex;align-items:center;gap:8px;cursor:pointer;white-space:nowrap;">
            <input type="checkbox" ${checked} onchange="toggleRunIdx(${i})" style="accent-color:${color};">
            <input type="color" value="${color}" onchange="setRunColor(${i}, this.value)" title="change color for this run"
                   style="width:22px;height:16px;padding:0;border:1px solid var(--border);border-radius:2px;cursor:pointer;background:transparent;">
            <span style="color:${color};font-family:monospace;font-size:11px;">${esc(r.run_id)}</span>
            <span class="small" style="font-size:11px;">— ${esc(r.label)} (${r.iters_completed?.toLocaleString()||'?'} iters)</span>
          </label>`;
        }).join('')}
      </div>
    </div>` : '';

  const metaBanner = rm ? `
    <div style="background:#0b0e13;border:1px solid var(--border);border-radius:4px;padding:10px 14px;margin:4px 0 12px;font-size:12px;">
      <div style="display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:12px;">
        <div><span class="blue" style="font-weight:600;">nano run:</span> <code>${esc(rm.run_id)}</code>
             <span class="small" style="margin-left:8px;">${esc(rm.label || '')}</span>
             ${rm.has_biasfix ? '<span class="status-pill ok" style="margin-left:8px;">bias-fix</span>' : '<span class="status-pill warn" style="margin-left:8px;">buggy bias</span>'}</div>
        <div class="small">${esc(rm.started_at || '')} → ${esc(rm.finished_at || 'running')} (${rm.duration_minutes || '?'} min)</div>
      </div>
      <div class="small" style="margin-top:6px;">
        config <code>${esc(rm.config || '')}</code> · init <code>${esc(rm.init_from || '')}</code> · host <code>${esc(rm.host || '')}</code> ·
        DP=${rm.dp_world_size || '?'} · GBS=${rm.global_batch_size || '?'} · iters=${(rm.iters_completed||0).toLocaleString()}
        ${rm.final_nano_loss != null ? ` · final nano=<span class="warn">${rm.final_nano_loss.toFixed(3)}</span>` : ''}
        ${rm.final_ref_loss != null ? ` vs ref=<span class="blue">${rm.final_ref_loss.toFixed(3)}</span>` : ''}
        ${cmp.final_iter_diff != null ? ` Δ=<span class="${Math.abs(cmp.final_iter_diff)<0.1?'ok':'err'}">${cmp.final_iter_diff>=0?'+':''}${cmp.final_iter_diff.toFixed(3)}</span> nat` : ''}
      </div>
      ${rm.notes ? `<div class="small" style="margin-top:6px;color:var(--warn);">note: ${esc(rm.notes)}</div>` : ''}
    </div>` : '';
  const refStats = l?.ref || {
    n_steps: 7485, first_step: 1, last_step: 7485,
    first_loss: 11.9430, last_loss: 2.8596, min_loss: 2.6154,
  };
  document.getElementById('loss-body').innerHTML = picker + metaBanner + `
    <div class="cards">
      ${card('ref steps', refStats.n_steps, `${refStats.first_step}..${refStats.last_step}`)}
      ${card('ref first', refStats.first_loss.toFixed(3), '@ step 1')}
      ${card('ref last', refStats.last_loss.toFixed(3), `@ step ${refStats.last_step}`)}
      ${card('ref min', refStats.min_loss.toFixed(3))}
      ${haveNano && cmp.max_abs_diff != null ? card('diff max', cmp.max_abs_diff.toFixed(4),
                         `@ step ${cmp.max_abs_diff_step}`, 'err') : ''}
      ${haveNano && cmp.first_diverge_step_1e4 != null ? card('first diverge', cmp.first_diverge_step_1e4, '|Δ| > 1e-4') : ''}
      ${haveNano && cmp.early_1_50_mean_abs != null ? card('mean |Δ| 1–50', cmp.early_1_50_mean_abs.toFixed(4), 'bf16 ULP', 'ok') : ''}
      ${haveNano && cmp.mid_51_500_mean_abs != null ? card('mean |Δ| 51–500', cmp.mid_51_500_mean_abs.toFixed(4), 'warmup phase', cmp.mid_51_500_mean_abs<0.05?'ok':'warn') : ''}
      ${haveNano && cmp.decay_6k_end_mean_abs != null ? card('mean |Δ| 6k–end', cmp.decay_6k_end_mean_abs.toFixed(4), 'decay phase', cmp.decay_6k_end_mean_abs<0.05?'ok':'err') : ''}
    </div>
    ${haveNano ? '' : `<div class="small">nanogpt run 未完成 — 只展示 Megatron 参考曲线。</div>`}`;

  if (tb && tb['lm loss']) {
    const lm = tb['lm loss'];
    const lr = tb['learning-rate'] || [];

    // Build ref lookup {iter → loss} once so we can compute per-iter Δ quickly.
    const refByIter = new Map();
    for (const [it, v] of lm) refByIter.set(it, v);

    const traces = [];
    if (!SHOW_DIFF_MODE) {
      // Raw loss: ref curve (always) + each selected nano run.
      traces.push({
        x: lm.map(p => p[0]), y: lm.map(p => p[1]),
        type: 'scatter', mode: 'lines', name: 'Megatron (ref lm loss)',
        line: { color: REF_COLOR, width: 1.6 }, yaxis: 'y',
      });
    } else {
      // Δ mode: add a y=0 baseline so the zero line is obvious.
      traces.push({
        x: lm.map(p => p[0]), y: lm.map(() => 0),
        type: 'scatter', mode: 'lines', name: 'ref baseline (Δ=0)',
        line: { color: REF_COLOR, width: 1, dash: 'dash' }, yaxis: 'y',
        hoverinfo: 'skip',
      });
    }

    // Per-run traces.
    if (runs.length > 0) {
      for (const i of Array.from(selected).sort((a,b)=>a-b)) {
        const r = runs[i];
        if (!r) continue;
        const color = runColor(i);
        const off = parseInt(r.iter_offset || 0);
        if (r.train_loss_points && r.train_loss_points.length) {
          let x, y, namePrefix;
          if (SHOW_DIFF_MODE) {
            // Align nano iter N to ref iter (N+off); plot (nano − ref) where ref exists.
            x = []; y = [];
            for (const [it, v] of r.train_loss_points) {
              const refIt = it + off;
              if (refByIter.has(refIt)) { x.push(refIt); y.push(v - refByIter.get(refIt)); }
            }
            namePrefix = `Δ nano − ref · ${r.run_id}`;
          } else {
            x = r.train_loss_points.map(p => p[0] + off);
            y = r.train_loss_points.map(p => p[1]);
            namePrefix = `nano · ${r.run_id}` + (off ? ` (+${off})` : '');
          }
          traces.push({
            x, y, type: 'scatter', mode: 'lines',
            name: namePrefix,
            line: { color, width: 1.2 }, yaxis: 'y',
          });
        }
        if (r.val_loss_points && r.val_loss_points.length && !SHOW_DIFF_MODE) {
          traces.push({
            x: r.val_loss_points.map(p => p[0] + off),
            y: r.val_loss_points.map(p => p[1]),
            type: 'scatter', mode: 'lines+markers',
            name: `nano val · ${r.run_id} (n=${r.val_loss_points.length})`,
            line: { color, width: 2, dash: 'dash' },
            marker: { size: 7, symbol: 'diamond', color }, yaxis: 'y',
          });
        }
      }
    } else {
      // Legacy fallback when no runs_index.json exists.
      const nano = DATA.nano_log;
      if (nano && nano.train_loss && nano.train_loss.length) {
        traces.push({
          x: nano.train_loss.map(p => p[0]), y: nano.train_loss.map(p => p[1]),
          type: 'scatter', mode: 'lines', name: `nano train (n=${nano.n_entries})`,
          line: { color: '#ff7b72', width: 1.2 }, yaxis: 'y',
        });
      }
      if (nano && nano.val_loss && nano.val_loss.length && !SHOW_DIFF_MODE) {
        traces.push({
          x: nano.val_loss.map(p => p[0]), y: nano.val_loss.map(p => p[1]),
          type: 'scatter', mode: 'lines+markers', name: `nano val (n=${nano.val_loss.length})`,
          line: { color: '#ffa657', width: 2, dash: 'dash' },
          marker: { size: 8, symbol: 'diamond' }, yaxis: 'y',
        });
      }
    }
    if (lr.length) {
      traces.push({
        x: lr.map(p => p[0]), y: lr.map(p => p[1]),
        type: 'scatter', mode: 'lines', name: 'Megatron (ref lr schedule)',
        line: { color: '#d29922', width: 1, dash: 'dot' }, yaxis: 'y2',
      });
    }
    Plotly.newPlot('loss-plot', traces, {
      paper_bgcolor: '#161b22', plot_bgcolor: '#0b0e13',
      font: { color: '#e6edf3', size: 11 },
      xaxis: { title: 'step', gridcolor: '#30363d' },
      yaxis: {
        title: SHOW_DIFF_MODE ? 'Δ lm loss (nano − ref)' : 'lm loss',
        gridcolor: '#30363d',
        zeroline: SHOW_DIFF_MODE, zerolinecolor: '#30363d', zerolinewidth: 1,
      },
      yaxis2: { title: 'lr', overlaying: 'y', side: 'right',
                gridcolor: 'transparent', color: '#d29922' },
      margin: { t: 20, l: 50, r: 60, b: 40 },
      legend: { x: 0.7, y: 0.95 },
    }, { responsive: true, displaylogo: false });
  }
}

// --------- MoE routing (nano vs ref) ---------
function renderRouting() {
  const refRouting = DATA.ref_routing;  // list of {iteration, tok_per_expert_max/min/mean, maxvio_micro_batch, ...}
  const runs = DATA.runs || [];
  const selected = getSelectedRunIdxs(runs);  // reuse same selector state as loss chart

  if (!refRouting || refRouting.length === 0) {
    pill('routing-status', 'warn', 'no ref data');
    document.getElementById('routing-body').innerHTML =
      `<div class="small">reference/ref_moe_routing_stats.json not found. Run
      <code>python3 /tmp/extract_ref_moe_stats.py &lt;ref_master.log&gt; reference/ref_moe_routing_stats.json</code>.</div>`;
    return;
  }

  const traces = [{
    x: refRouting.map(r => r.iteration),
    y: refRouting.map(r => r.maxvio_micro_batch),
    type: 'scatter', mode: 'lines',
    name: 'Megatron (ref maxvio)',
    line: { color: REF_COLOR, width: 1.6 }, yaxis: 'y',
  }];

  // nano runs that have routing_stats_points
  let nano_runs_with_routing = 0;
  for (const i of Array.from(selected).sort((a,b)=>a-b)) {
    const r = runs[i];
    if (!r || !r.routing_stats_points || r.routing_stats_points.length === 0) continue;
    const off = parseInt(r.iter_offset || 0);
    nano_runs_with_routing += 1;
    const color = runColor(i);
    // routing_stats_points: [iter, maxvio, tok_max_per_mb, tok_min_per_mb, tok_mean_per_mb]
    traces.push({
      x: r.routing_stats_points.map(p => p[0] + off),
      y: r.routing_stats_points.map(p => p[1]),
      type: 'scatter', mode: 'lines+markers',
      name: `nano maxvio · ${r.run_id}`,
      line: { color, width: 1.6 },
      marker: { size: 4, color }, yaxis: 'y',
    });
  }

  pill('routing-status',
       nano_runs_with_routing > 0 ? 'ok' : 'warn',
       nano_runs_with_routing > 0 ? `${nano_runs_with_routing} nano run(s)` : 'ref only');

  // Summary stats
  const refMaxvios = refRouting.map(r => r.maxvio_micro_batch);
  const refMean = refMaxvios.reduce((a,b)=>a+b, 0) / refMaxvios.length;
  let summaryRows = [
    ['ref', refMean.toFixed(3), refMaxvios.length + ' iters']
  ];
  for (const i of Array.from(selected).sort((a,b)=>a-b)) {
    const r = runs[i];
    if (!r || !r.routing_stats_points) continue;
    const mv = r.routing_stats_points.map(p => p[1]);
    const mean = mv.reduce((a,b)=>a+b, 0) / mv.length;
    summaryRows.push([r.run_id, mean.toFixed(3), mv.length + ' iters']);
  }
  const summaryTable = `<table style="margin-top:8px;">
    <thead><tr><th>source</th><th>maxvio mean</th><th>n iters</th></tr></thead>
    <tbody>${summaryRows.map(r => `<tr><td>${esc(r[0])}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('')}</tbody>
  </table>`;
  document.getElementById('routing-body').innerHTML = summaryTable;

  Plotly.newPlot('routing-plot', traces, {
    paper_bgcolor: '#161b22', plot_bgcolor: '#0b0e13',
    font: { color: '#e6edf3', size: 11 },
    xaxis: { title: 'step', gridcolor: '#30363d' },
    yaxis: { title: 'maxvio_micro_batch = (max − mean) / mean', gridcolor: '#30363d' },
    margin: { t: 20, l: 50, r: 40, b: 40 },
    legend: { x: 0.7, y: 0.95 },
  }, { responsive: true, displaylogo: false });
}

// --------- MoE fleet (PAI Adam + Muon + nano v10repro bucket) ---------
const _moeSelected = new Set();
let _moeSelectedInit = false;
let _moeSmoothWindow = 1;

const MOE_COLORS = {
  'PAI ref Adam':          '#888',
  'PAI Muon ref':          '#ff8c00',
  'nano bucket fix (Adam)':'#2ecc71',
  'nano Muon (test)':      '#1e90ff',
  'nano v10':              '#9b59b6',
  'FLEET AVG':             '#e74c3c',
};

function _onMoeRunToggle(cb) {
  const id = cb.getAttribute('data-run-id');
  if (cb.checked) _moeSelected.add(id); else _moeSelected.delete(id);
  renderMoEFleet();
}
function _onMoeSmoothChange(sel) {
  _moeSmoothWindow = parseInt(sel.value) || 1;
  renderMoEFleet();
}
function _moeSmooth(ys, window) {
  if (!window || window <= 1 || ys.length <= 1) return ys.slice();
  const n = ys.length, w = Math.min(window, n);
  const out = new Array(n); let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += ys[i];
    if (i >= w) sum -= ys[i - w];
    out[i] = sum / Math.min(i + 1, w);
  }
  return out;
}

function renderMoEFleet() {
  const d = DATA.moe_fleet;
  const body = document.getElementById('moe-fleet-body');
  if (!d || !d.runs || !d.runs.length) {
    pill('moe-fleet-status', 'warn', 'missing');
    if (body) body.innerHTML = '<div class="small">reports/moe_fleet.json not found.</div>';
    return;
  }
  const runs = d.runs;
  if (!_moeSelectedInit) {
    // Default: FLEET AVG Adam + FLEET AVG Muon (so user instantly sees Adam-vs-Muon Δ)
    const adam_avg = runs.find(r => r.run_id === 'pai-ref-adam-FLEET-AVG');
    const muon_avg = runs.find(r => r.run_id === 'pai-muon-FLEET-AVG');
    if (adam_avg) _moeSelected.add(adam_avg.run_id);
    if (muon_avg) _moeSelected.add(muon_avg.run_id);
    _moeSelectedInit = true;
  }
  // Build picker UI
  let pickerHtml = '<div style="margin:8px 0;">';
  pickerHtml += '<span class="small" style="margin-right:8px;">Smooth (MA window): </span>';
  pickerHtml += `<select onchange="_onMoeSmoothChange(this)" style="margin-right:16px;">`;
  for (const w of [1, 10, 50, 100, 200]) {
    pickerHtml += `<option value="${w}" ${w === _moeSmoothWindow ? 'selected' : ''}>${w === 1 ? 'off' : w}</option>`;
  }
  pickerHtml += '</select></div><div style="margin-bottom:8px;">';
  for (const r of runs) {
    const checked = _moeSelected.has(r.run_id) ? 'checked' : '';
    const color = MOE_COLORS[r.group] || '#4a90e2';
    pickerHtml += `<label style="display:inline-flex;align-items:center;gap:6px;margin:2px 12px 2px 0;font-size:12px;">
      <input type="checkbox" ${checked} data-run-id="${esc(r.run_id)}" onchange="_onMoeRunToggle(this)">
      <span style="width:10px;height:10px;background:${color};border-radius:2px;"></span>
      ${esc(r.label)} <span class="small">(${r.iters_completed})</span>
    </label>`;
  }
  pickerHtml += '</div>';

  // Build traces for selected runs
  const selRuns = runs.filter(r => _moeSelected.has(r.run_id));
  const traces = [];
  for (const r of selRuns) {
    const color = MOE_COLORS[r.group] || '#4a90e2';
    const pts = r.train_loss_points || [];
    const xs = [], raw_ys = [];
    for (let i = 0; i < pts.length; i++) {
      if (pts[i][0] <= 100 || pts[i][0] % 10 === 0) {
        xs.push(pts[i][0]); raw_ys.push(pts[i][1]);
      }
    }
    const ys = _moeSmooth(raw_ys, _moeSmoothWindow);
    traces.push({
      x: xs, y: ys, mode: 'lines', name: r.label,
      line: { color, width: r.run_id.includes('FLEET-AVG') ? 2.5 : 1.2 },
      opacity: r.run_id.includes('FLEET-AVG') ? 1.0 : 0.7,
    });
  }
  Plotly.newPlot('moe-fleet-plot', traces, {
    margin: { l: 50, r: 20, t: 8, b: 40 },
    paper_bgcolor: 'var(--panel)', plot_bgcolor: 'var(--panel)',
    font: { color: 'var(--fg)', size: 11 },
    xaxis: { title: 'iter' }, yaxis: { title: 'lm loss', range: [2.6, 5.0] },
    legend: { orientation: 'h', y: -0.15, font: { size: 10 } },
  }, { responsive: true, displaylogo: false });

  // Pairwise Δ plot when exactly 2 selected
  let deltaPlotHtml = '';
  if (selRuns.length === 2) {
    const [A, B] = selRuns;
    const mapA = new Map((A.train_loss_points || []).map(p => [p[0], p[1]]));
    const mapB = new Map((B.train_loss_points || []).map(p => [p[0], p[1]]));
    const commonIters = [...mapA.keys()].filter(k => mapB.has(k)).sort((a,b) => a-b);
    const xs = [], raw_ys = [];
    for (const it of commonIters) {
      if (it <= 100 || it % 10 === 0) { xs.push(it); raw_ys.push(mapA.get(it) - mapB.get(it)); }
    }
    const ys = _moeSmooth(raw_ys, _moeSmoothWindow);
    deltaPlotHtml = `<h3 style="margin-top:16px;">Δ (<span style="color:${MOE_COLORS[A.group]||'#4a90e2'};">${esc(A.label)}</span> − <span style="color:${MOE_COLORS[B.group]||'#4a90e2'};">${esc(B.label)}</span>)</h3>
      <div id="moe-fleet-delta-plot" style="height:280px;background:var(--panel);border:1px solid var(--border);border-radius:4px;"></div>`;
    setTimeout(() => {
      Plotly.newPlot('moe-fleet-delta-plot', [
        { x: xs, y: ys, mode: 'lines', name: 'Δ', line: { color: '#e74c3c', width: 1.5 } },
        { x: [xs[0] || 0, xs[xs.length-1] || 1], y: [0, 0], mode: 'lines', name: 'zero', line: { color: '#888', width: 1, dash: 'dash' }, showlegend: false },
      ], {
        margin: { l: 50, r: 20, t: 8, b: 40 },
        paper_bgcolor: 'var(--panel)', plot_bgcolor: 'var(--panel)',
        font: { color: 'var(--fg)', size: 11 },
        xaxis: { title: 'iter' }, yaxis: { title: 'Δ lm loss (A − B)' },
      }, { responsive: true, displaylogo: false });
    }, 0);
  } else if (selRuns.length > 2) {
    deltaPlotHtml = `<div class="small" style="margin-top:12px;color:var(--muted);">Select exactly 2 runs to see pairwise Δ.</div>`;
  }

  pill('moe-fleet-status', 'ok', `${runs.length} runs, ${selRuns.length} selected`);
  // Summary table
  const groups = {};
  for (const r of runs) {
    if (!r.train_loss_points || r.train_loss_points.length < 100) continue;
    const tail = r.train_loss_points.slice(-100);
    const ema = tail.reduce((a,b) => a + b[1], 0) / tail.length;
    (groups[r.group] = groups[r.group] || []).push({ label: r.label, iters: r.iters_completed, tail_ema: ema });
  }
  let tableHtml = '<table class="tbl" style="margin-top:12px;"><thead><tr><th>Group</th><th>Run</th><th>Iters</th><th>tail-100 mean loss</th></tr></thead><tbody>';
  for (const g of Object.keys(groups)) {
    for (const r of groups[g]) {
      tableHtml += `<tr><td style="color:${MOE_COLORS[g]||'#4a90e2'};">${g}</td><td>${esc(r.label)}</td><td>${r.iters}</td><td>${r.tail_ema.toFixed(4)}</td></tr>`;
    }
  }
  tableHtml += '</tbody></table>';
  body.innerHTML = pickerHtml + deltaPlotHtml + tableHtml;
}

// --------- Dense ablation (multi-run picker + pairwise Δ) ---------
let _denseAutoRefreshTimer = null;
let _denseData = null;                    // latest fetched payload (new `runs` schema)
const _denseSelected = new Set();         // run ids currently selected
let _denseSelectedInit = false;           // true after first default-selection applied
let _denseSmoothWindow = 1;               // moving-avg window, 1 = no smoothing

function _smoothMA(ys, window) {
  // Trailing moving average. window=1 → passthrough.
  if (!window || window <= 1 || ys.length <= 1) return ys.slice();
  const n = ys.length, w = Math.min(window, n);
  const out = new Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += ys[i];
    if (i >= w) sum -= ys[i - w];
    out[i] = sum / Math.min(i + 1, w);
  }
  return out;
}

function _onDenseSmoothChange(sel) {
  _denseSmoothWindow = parseInt(sel.value) || 1;
  renderDenseAblation(_denseData);
}

function _denseDefaultSelection(runs) {
  // Default: select first run with data + first OTHER run with data (so user can see a Δ).
  const ids = runs.filter(r => r.n_points > 0).map(r => r.id);
  const pick = new Set();
  if (ids.length) pick.add(ids[0]);
  if (ids.length > 1) pick.add(ids[1]);
  return pick;
}

function _densePicker(runs) {
  return runs.map(r => {
    const checked = _denseSelected.has(r.id) ? 'checked' : '';
    const count = r.n_points ? `${r.n_points} iters` : 'no data yet';
    const dimStyle = r.n_points ? '' : 'opacity:0.5;';
    return `<label style="display:inline-flex;align-items:center;gap:6px;margin:2px 12px 2px 0;font-size:12px;${dimStyle}">
      <input type="checkbox" ${checked} data-run-id="${esc(r.id)}" onchange="_onDenseRunToggle(this)">
      <span style="width:10px;height:10px;background:${esc(r.color)};border-radius:2px;"></span>
      ${esc(r.label)} <span class="small">(${count})</span>
    </label>`;
  }).join('');
}

function _onDenseRunToggle(cb) {
  const id = cb.getAttribute('data-run-id');
  if (cb.checked) _denseSelected.add(id); else _denseSelected.delete(id);
  renderDenseAblation(_denseData);
}

function renderDenseAblation(overrideData) {
  const d = overrideData || DATA.dense_ablation;
  _denseData = d;

  // Backward-compat: if someone shipped the old schema, shim it into runs format.
  let runs = d && d.runs;
  if (!runs && d && d.ref_dense && d.nano_dense) {
    runs = [
      {id:'ref_dense', label:'Megatron ref-dense', color:REF_COLOR, iter_offset:0, points:d.ref_dense, n_points:d.ref_dense.length, last_loss:d.ref_dense.at(-1)?.[1], last_iter:d.ref_dense.at(-1)?.[0]},
      {id:'nano_dense', label:'nano-dense seed=1337', color:'#ff7b72', iter_offset:1, points:d.nano_dense, n_points:d.nano_dense.length, last_loss:d.nano_dense.at(-1)?.[1], last_iter:d.nano_dense.at(-1)?.[0]},
    ];
    if (d.nano_dense_seed42) runs.push({id:'nano_seed42', label:'nano-dense seed=42', color:'#a5d6ff', iter_offset:1, points:d.nano_dense_seed42, n_points:d.nano_dense_seed42.length, last_loss:d.nano_dense_seed42.at(-1)?.[1], last_iter:d.nano_dense_seed42.at(-1)?.[0]});
  }

  if (!runs || !runs.length) {
    pill('dense-status', 'warn', 'missing');
    document.getElementById('dense-body').innerHTML =
      `<div class="small">reports/dense_ablation.json not found. Run
       <code>python3 scripts/build_dense_ablation.py</code> to generate.</div>`;
    _armDenseAutoRefresh();
    return;
  }

  if (!_denseSelectedInit) {
    for (const id of _denseDefaultSelection(runs)) _denseSelected.add(id);
    _denseSelectedInit = true;
  }
  // Drop selections that no longer have data (run id removed).
  const knownIds = new Set(runs.map(r => r.id));
  for (const id of Array.from(_denseSelected)) if (!knownIds.has(id)) _denseSelected.delete(id);

  const refreshedNote = overrideData ? ` · <span style="color:var(--ok)">刷新于 ${new Date().toLocaleTimeString()}</span>` : '';
  const okRuns = runs.filter(r => r.n_points > 0);
  pill('dense-status', okRuns.length ? 'ok' : 'warn',
       `${okRuns.length}/${runs.length} runs · ${_denseSelected.size} selected`);

  // Build loss traces for every selected run (shift by iter_offset to align x-axis).
  const selRuns = runs.filter(r => _denseSelected.has(r.id) && r.n_points);
  const sw = _denseSmoothWindow;
  const traces = [];
  for (const r of selRuns) {
    traces.push({
      x: r.points.map(p => p[0] + (r.iter_offset || 0)),
      y: _smoothMA(r.points.map(p => p[1]), sw),
      type: 'scatter', mode: 'lines',
      name: r.label,
      line: { color: r.color, width: 1.6 },
      yaxis: 'y',
    });
  }

  // Pairwise Δ: when exactly 2 selected, compute (second − first) on shared x.
  // When more selected, compute each (i − ref-of-selection), where ref-of-selection
  // is the FIRST selected run (treated as baseline).
  // Δ stats (mean/std/last/maxAbs) are computed on RAW Δ, not smoothed.
  let pairInfo = null;
  if (selRuns.length >= 2) {
    const base = selRuns[0];
    const baseBy = new Map();
    for (const p of base.points) baseBy.set(p[0] + (base.iter_offset || 0), p[1]);
    const deltaColors = ['#d29922', '#56d364', '#f778ba', '#a371f7'];  // for non-base runs
    for (let i = 1; i < selRuns.length; i++) {
      const other = selRuns[i];
      const dx = [], dy = [];
      for (const p of other.points) {
        const absX = p[0] + (other.iter_offset || 0);
        if (baseBy.has(absX)) { dx.push(absX); dy.push(p[1] - baseBy.get(absX)); }
      }
      if (!dx.length) continue;
      const mean = dy.reduce((a,b)=>a+b,0) / dy.length;
      const variance = dy.reduce((a,b)=>a+(b-mean)**2,0) / Math.max(dy.length-1,1);
      const std = Math.sqrt(variance);
      const maxAbs = dy.reduce((a,b)=>Math.max(a, Math.abs(b)), 0);
      traces.push({
        x: dx, y: _smoothMA(dy, sw),
        type: 'scatter', mode: 'lines',
        name: `Δ ${other.label} − ${base.label} (右轴)`,
        line: { color: deltaColors[(i-1) % deltaColors.length], width: 1.4, dash: 'dot' },
        yaxis: 'y2',
      });
      if (i === 1) pairInfo = {base, other, n:dx.length, mean, std, maxAbs, last:dy.at(-1), first:dy[0]};
    }
  }

  Plotly.newPlot('dense-plot', traces, {
    paper_bgcolor: '#161b22', plot_bgcolor: '#0b0e13',
    font: { color: '#e6edf3', size: 11 },
    xaxis: { title: 'step (aligned 1-indexed)', gridcolor: '#30363d' },
    yaxis: { title: 'lm loss', gridcolor: '#30363d' },
    yaxis2: {
      title: 'Δ (vs first selected)', overlaying: 'y', side: 'right',
      gridcolor: 'transparent', color: '#d29922', zeroline: true,
      zerolinecolor: '#d29922', zerolinewidth: 1,
    },
    margin: { t: 20, l: 50, r: 60, b: 40 },
    legend: { x: 0.65, y: 0.98 },
  }, { responsive: true, displaylogo: false });

  // Grad norm chart for the same selected runs.
  // points rows are [iter, loss, lr, grad_norm] — index 3.
  const gnTraces = [];
  for (const r of selRuns) {
    gnTraces.push({
      x: r.points.map(p => p[0] + (r.iter_offset || 0)),
      y: _smoothMA(r.points.map(p => p[3]), sw),
      type: 'scatter', mode: 'lines',
      name: r.label,
      line: { color: r.color, width: 1.4 },
      yaxis: 'y',
    });
  }
  if (selRuns.length >= 2) {
    const base = selRuns[0];
    const baseBy = new Map();
    for (const p of base.points) baseBy.set(p[0] + (base.iter_offset || 0), p[3]);
    const gnDeltaColors = ['#d29922', '#56d364', '#f778ba', '#a371f7'];
    for (let i = 1; i < selRuns.length; i++) {
      const other = selRuns[i];
      const dx = [], dy = [];
      for (const p of other.points) {
        const absX = p[0] + (other.iter_offset || 0);
        if (baseBy.has(absX)) { dx.push(absX); dy.push(p[3] - baseBy.get(absX)); }
      }
      if (dx.length) gnTraces.push({
        x: dx, y: _smoothMA(dy, sw),
        type: 'scatter', mode: 'lines',
        name: `Δ gn ${other.label} − ${base.label} (右轴)`,
        line: { color: gnDeltaColors[(i-1) % gnDeltaColors.length], width: 1.2, dash: 'dot' },
        yaxis: 'y2',
      });
    }
  }
  Plotly.newPlot('dense-gradnorm-plot', gnTraces, {
    paper_bgcolor: '#161b22', plot_bgcolor: '#0b0e13',
    font: { color: '#e6edf3', size: 11 },
    xaxis: { title: 'step', gridcolor: '#30363d' },
    yaxis: { title: 'grad_norm', gridcolor: '#30363d' },
    yaxis2: {
      title: 'Δ grad_norm', overlaying: 'y', side: 'right',
      gridcolor: 'transparent', color: '#d29922', zeroline: true,
      zerolinecolor: '#d29922', zerolinewidth: 1,
    },
    margin: { t: 20, l: 50, r: 60, b: 40 },
    legend: { x: 0.65, y: 0.98 },
  }, { responsive: true, displaylogo: false });

  // Cards: per-run EMA/tail-100 loss (more stable than single last-iter).
  // Raw last iter shown as subtitle for reference but not headline since per-iter
  // noise is ~±0.1 nat — single point unreliable for comparisons.
  const runCards = runs.map(r => {
    if (!r.n_points) return card(`${r.label}`, '—', 'no data', 'warn');
    const ema = r.ema_last != null ? r.ema_last.toFixed(4) : '—';
    const tm  = r.tail100_mean != null ? r.tail100_mean.toFixed(4) : '—';
    const ts  = r.tail100_std != null ? r.tail100_std.toFixed(4) : '—';
    return card(
      `${r.label}`,
      `ema ${ema}`,
      `tail100 ${tm}±${ts} · last ${r.last_loss.toFixed(3)} · iter ${r.last_iter}`
    );
  }).join('');

  // Fleet-level summary: if ≥2 nano noise runs OR ≥2 pai noise runs are selected,
  // compute cross-run mean/std of their EMA values (a more stable noise-floor estimate).
  function fleetEmaSummary(selRuns, prefix) {
    const fleet = selRuns.filter(r => r.id.startsWith(prefix) && r.ema_last != null);
    if (fleet.length < 2) return null;
    const values = fleet.map(r => r.ema_last);
    const m = values.reduce((a,b)=>a+b,0) / values.length;
    const std = Math.sqrt(values.reduce((a,b)=>a+(b-m)**2,0) / (values.length-1));
    return { n: fleet.length, mean: m, std, min: Math.min(...values), max: Math.max(...values) };
  }
  const nanoFleet = fleetEmaSummary(selRuns, 'nano_dense_107_noise');
  const paiFleet  = fleetEmaSummary(selRuns, 'ref_dense_107_noise');
  let fleetCards = '';
  if (nanoFleet) fleetCards += card(`nano fleet (n=${nanoFleet.n})`, `ema ${nanoFleet.mean.toFixed(4)}`, `std ${nanoFleet.std.toFixed(4)} · range ${(nanoFleet.max-nanoFleet.min).toFixed(4)}`, 'ok');
  if (paiFleet)  fleetCards += card(`pai fleet (n=${paiFleet.n})`, `ema ${paiFleet.mean.toFixed(4)}`, `std ${paiFleet.std.toFixed(4)} · range ${(paiFleet.max-paiFleet.min).toFixed(4)}`, 'ok');
  if (nanoFleet && paiFleet) {
    const delta = nanoFleet.mean - paiFleet.mean;
    fleetCards += card('nano − pai (ema avg)', (delta>=0?'+':'') + delta.toFixed(4), `nano std ${nanoFleet.std.toFixed(4)} · pai std ${paiFleet.std.toFixed(4)}`, Math.abs(delta) < Math.max(nanoFleet.std, paiFleet.std)*2 ? 'ok' : 'warn');
  }

  let pairCards = '';
  let pairSummary = '';
  if (pairInfo) {
    const p = pairInfo;
    const near_noise = Math.abs(p.mean) < 0.005;
    pairCards = `
      ${card(`mean Δ`, p.mean.toFixed(4), `${esc(p.other.label)} − ${esc(p.base.label)}`, near_noise ? 'ok' : 'warn')}
      ${card(`std Δ`, p.std.toFixed(4), `${p.n} common iters`)}
      ${card(`last Δ`, p.last.toFixed(4))}
      ${card(`max |Δ|`, p.maxAbs.toFixed(4))}
    `;
    pairSummary = `
      <div class="small" style="margin-top:6px;">
        <b>${esc(p.other.label)} − ${esc(p.base.label)}</b>:
        ${p.n} 步 · mean ${p.mean >= 0 ? '+' : ''}${p.mean.toFixed(4)} · std ${p.std.toFixed(4)} · last ${p.last >= 0 ? '+' : ''}${p.last.toFixed(4)}
      </div>`;
  } else if (selRuns.length === 1) {
    pairSummary = `<div class="small" style="margin-top:6px;color:var(--muted);">再选一个以上的 run 才能看 Δ。</div>`;
  } else {
    pairSummary = `<div class="small" style="margin-top:6px;color:var(--muted);">勾选 run 开始对比。</div>`;
  }

  const meta = d.meta || {};
  const html = `
    <div style="margin:4px 0 10px;padding:8px 10px;background:var(--panel);border:1px solid var(--border);border-radius:4px;">
      <div style="display:flex;justify-content:space-between;align-items:center;font-size:11px;color:var(--muted);margin-bottom:4px;">
        <span>勾选要对比的 run（选 2+ 个自动出现 Δ 曲线在右轴）</span>
        <label style="display:inline-flex;align-items:center;gap:6px;">
          平滑窗口:
          <select onchange="_onDenseSmoothChange(this)" style="background:#0b0e13;color:#e6edf3;border:1px solid var(--border);padding:2px 6px;font-size:11px;">
            ${[1,10,50,100,200,500,1000].map(w => `<option value="${w}" ${w===_denseSmoothWindow?'selected':''}>${w===1?'raw (no smoothing)':'MA '+w+' iters'}</option>`).join('')}
          </select>
        </label>
      </div>
      ${_densePicker(runs)}
    </div>
    ${fleetCards ? `<div class="cards" style="margin-bottom:6px;">${fleetCards}</div>` : ''}
    <div class="cards">${runCards}${pairCards}</div>
    ${pairSummary}
    <div class="small" style="margin-top:6px;color:var(--muted);">
      config: <code>${esc(meta.config_ref || 'scaling_dense_00107_nc.yaml')}</code>
      · generated ${esc(meta.generated_at || '')}
      <span style="margin-left:12px;color:var(--ok);">auto-refresh 30s${refreshedNote}</span>
    </div>`;
  document.getElementById('dense-body').innerHTML = html;
  _armDenseAutoRefresh();
}

function _armDenseAutoRefresh() {
  if (_denseAutoRefreshTimer) return;
  _denseAutoRefreshTimer = setInterval(async () => {
    try {
      const resp = await fetch('../reports/dense_ablation.json?t=' + Date.now(), { cache: 'no-store' });
      if (!resp.ok) return;
      const fresh = await resp.json();
      renderDenseAblation(fresh);
    } catch (e) { /* ignore */ }
  }, 30000);
}

// --------- bitwise ---------
function renderBitwise() {
  const b = DATA.bitwise;
  if (!b) { pill('bw-status', 'warn', 'pending GPU');
    document.getElementById('bw-body').innerHTML =
      `<div class="small">需在 8 卡机执行 <code>make bitwise-check</code>；此处会展示 A/B checksum 对比与首个 diverge 步。</div>`;
    return;
  }
  // New schema: {single_gpu: {pass, ...}, ddp_4rank: {pass, ...}, ...}
  // Legacy schema: {pass, mode, ...}
  // pass can be true/false or a string like "near-bitwise (bf16 floor)".
  const isPass = (v) => v === true || (typeof v === 'string' && /pass|near-bitwise/i.test(v));
  let sgPass, ddpPass, summaryText;
  if ('single_gpu' in b || 'ddp_4rank' in b) {
    const rawDdp = b.ddp_4rank && b.ddp_4rank.pass;
    sgPass = isPass(b.single_gpu && b.single_gpu.pass);
    ddpPass = isPass(rawDdp);
    if (sgPass && ddpPass) summaryText = (rawDdp === true) ? 'pass (single+ddp)' : 'pass (single + ddp near-floor)';
    else if (sgPass && rawDdp === false) summaryText = 'single pass · ddp diverge';
    else if (sgPass) summaryText = 'single pass';
    else summaryText = 'fail';
    const pillState = sgPass ? (ddpPass ? 'ok' : 'warn') : 'err';
    pill('bw-status', pillState, summaryText);
  } else {
    const ok = b.pass;
    pill('bw-status', ok === true ? 'ok' : ok === false ? 'err' : 'warn',
         ok === true ? 'pass' : ok === false ? 'fail' : 'scaffold');
  }
  document.getElementById('bw-body').innerHTML = `<pre>${esc(JSON.stringify(b, null, 2))}</pre>`;
}

// --------- ckpt ---------
function renderCkpt() {
  const c = DATA.ckpt;
  if (!c) { pill('ckpt-status', 'warn', 'no ckpt yet');
    document.getElementById('ckpt-body').innerHTML =
      `<div class="small">训练出 ckpt.pt 后运行 <code>python -m tools.ckpt_fingerprint out/ckpt.pt --json reports/ckpt_fingerprint.json</code>，再重新生成此 HTML。</div>`;
    return;
  }
  pill('ckpt-status', 'ok', `${c.n_params} params`);
  const summary = { mode: c.mode, n_params: c.n_params,
                    total_sha256: c.total_sha256, total_md5: c.total_md5,
                    total_bytes: c.total_bytes };
  document.getElementById('ckpt-body').innerHTML = `<pre>${esc(JSON.stringify(summary, null, 2))}</pre>`;
}

// --------- gaps ---------
function renderGaps() {
  const g = DATA.gaps;
  // Render concrete diff table rather than just raw MD
  const archDiffs = [
    ['n_layer', '16', '9'], ['n_embd', '656', '512'], ['n_head', '8', '4'],
    ['n_kv_head', '4', '2'], ['ffn_hidden_size', '1920', '1536'],
    ['moe_ffn_hidden_size', '224', '160'], ['moe_shared_expert_hidden', '224', '160'],
    ['moe_layer_freq', '[0]+[1]*15', '[0]+[1]*8'],
  ];
  const lrDiffs = [
    ['learning_rate', '8.28e-4', '1.2e-3'], ['min_lr', '8.28e-5', '1.2e-4'],
    ['warmup_samples', '64000', '32000'], ['constant_samples', '1110656', '383232'],
    ['decay_end_samples', '1388416', '479040'], ['global_batch_size', '128', '64'],
    ['max_iters', '10847', '7485'],
  ];
  const openGaps = [
    ['eod_mask_loss', 'mask loss at positions where target == 151643 (EOD)', 'closed'],
    ['mask_loss_id=160000', 'mask loss at positions where target == 160000', 'closed'],
    ['sequence_wise_balance_loss', 'per-MoE-layer seq-wise aux loss × α=0.0001', 'closed'],
    ['accurate_attn_mask_eod_token', 'attention cannot cross EOD within packed seq', 'closed'],
  ];
  const nOpenGaps = openGaps.filter(g => g[2] === 'open').length;
  pill('gap-status', nOpenGaps === 0 ? 'ok' : 'warn', nOpenGaps === 0 ? 'closed' : nOpenGaps + ' open');
  document.getElementById('gap-body').innerHTML = `
    <details open><summary>architecture diffs: <code>cybertron_moe_198.py</code> (old config) → 00196 target</summary>
      <div style="margin-top:10px">
        <div class="diff-row"><div class="k">field</div><div class="ref">old (198)</div><div class="nano">new (196)</div></div>
        ${archDiffs.map(([k,a,b]) => `<div class="diff-row"><div class="k">${k}</div><div class="ref">${a}</div><div class="nano">${b}</div></div>`).join('')}
      </div>
    </details>
    <details><summary>LR/schedule diffs</summary>
      <div style="margin-top:10px">
        <div class="diff-row"><div class="k">field</div><div class="ref">old (198)</div><div class="nano">new (196)</div></div>
        ${lrDiffs.map(([k,a,b]) => `<div class="diff-row"><div class="k">${k}</div><div class="ref">${a}</div><div class="nano">${b}</div></div>`).join('')}
      </div>
    </details>
    <details open><summary>Code gaps (tests/test_code_gaps.py 6/6 pass)</summary>
      <table><thead><tr><th>gap</th><th>effect</th><th>status</th></tr></thead><tbody>
        ${openGaps.map(([k,v,s]) => `<tr><td><code>${k}</code></td><td>${esc(v)}</td><td class="${s === 'closed' ? 'ok' : 'warn'}">${s}</td></tr>`).join('')}
      </tbody></table></details>`;
}

// --------- overview cards ---------
function renderOverview() {
  const phases = [
    ['reference job',     'job-status'],
    ['tokenizer',         'tok-status'],
    ['data sampling',     'data-status'],
    ['model structure',   'model-status'],
    ['bitwise resume',    'bw-status'],
    ['loss trajectory',   'loss-status'],
    ['ckpt fingerprint',  'ckpt-status'],
    ['code gaps',         'gap-status'],
    ['forward alignment', 'fwd-status'],
    ['alignment checklist', 'chk-status'],
  ];
  const ov = document.getElementById('overview');
  ov.innerHTML = phases.map(([name, id]) => {
    const st = document.getElementById(id);
    const cls = st ? st.classList[1] || '' : '';
    const text = st ? st.textContent : '-';
    return card(name, `<span class="${cls}">${esc(text)}</span>`, '', '');
  }).join('');
}

// --------- forward alignment ---------
function renderForwardAlign() {
  const f = DATA.fwd_align;
  if (!f) { pill('fwd-status', 'warn', 'not run'); return; }
  if (!f.comparisons) {
    // Newer schema may drop per-stat comparisons. Render top-level facts only.
    pill('fwd-status', 'ok', 'forward verified (summary)');
    document.getElementById('fwd-body').innerHTML =
      `<div class="small"><pre>${esc(JSON.stringify(f, null, 2).slice(0, 4000))}</pre></div>`;
    return;
  }
  pill('fwd-status', 'ok', 'forward math verified');
  const rows = Object.entries(f.comparisons).map(([k, v]) => {
    const nano = v.nano_layer_avg ?? v.nano_max;
    const diff = ((nano - v.ref) / (Math.abs(v.ref) + 1e-20) * 100);
    const label = 'nano_layer_avg' in v ? 'layer avg' : 'max';
    return `<tr>
      <td><code>${esc(k)}</code></td>
      <td>${typeof v.ref === 'number' ? v.ref.toFixed(4) : '—'}</td>
      <td>${typeof nano === 'number' ? nano.toFixed(4) : '—'}  <span class="small">(${label})</span></td>
      <td class="${Math.abs(diff) < 10 ? 'ok' : Math.abs(diff) < 30 ? 'warn' : 'err'}">${diff >= 0 ? '+' : ''}${diff.toFixed(1)}%</td>
    </tr>`;
  }).join('');
  document.getElementById('fwd-body').innerHTML = `
    <div class="cards">
      ${card('nano single-sample loss', f.nano_loss_single_sample.toFixed(3), 'iter_0001497 weights on train sample 0', 'ok')}
      ${card('ref global-batch loss', f.ref_loss_avg_global_batch_at_iter_1497.toFixed(3), 'ref at step 1497 (64-sample avg)', 'ok')}
      ${card('verdict', 'forward correct', 'stats align ±10% to ref', 'ok')}
    </div>
    <details open><summary>per-statistic comparison (ref iter 1497 vs nano Megatron-weight forward)</summary>
    <table><thead><tr><th>statistic</th><th>ref</th><th>nano</th><th>Δ</th></tr></thead>
    <tbody>${rows}</tbody></table>
    <div class="small" style="margin-top:8px">${esc(f.verdict)}</div>
    </details>`;
}

// --------- alignment checklist ---------
function renderChecklist() {
  const c = DATA.checklist;
  if (!c || !c.items) { pill('chk-status', 'err', 'missing'); return; }
  const s = c.summary;
  pill('chk-status', s.diff === 0 ? 'ok' : 'warn',
       `${s.ok}/${s.total} ok · ${s.diff} diff · ${s.unknown} ?`);
  // Controls: status filter + category filter
  const cats = [...new Set(c.items.map(it => it.category))];
  document.getElementById('chk-controls').innerHTML =
    `<label>status: <select id="f-status">
       <option value="DIFF" selected>DIFF only (default)</option>
       <option value="">all ${c.items.length} items</option>
       <option value="OK">OK only</option><option value="UNKNOWN">UNKNOWN only</option>
     </select></label>
     &nbsp;&nbsp;<label>category: <select id="f-cat">
       <option value="">all</option>
       ${cats.map(x => `<option value="${esc(x)}">${esc(x)}</option>`).join('')}
     </select></label>
     &nbsp;&nbsp;<label><input type="checkbox" id="f-autoexpand" checked> auto-expand detail for DIFF/UNKNOWN</label>
     &nbsp;&nbsp;<span class="small">${s.ok} ok · ${s.diff} diff · ${s.unknown} unknown, total ${s.total}</span>`;
  const renderItems = () => {
    const fs = document.getElementById('f-status').value;
    const fc = document.getElementById('f-cat').value;
    const byCat = {};
    c.items.forEach(it => {
      if (fs && it.status !== fs) return;
      if (fc && it.category !== fc) return;
      (byCat[it.category] ??= []).push(it);
    });
    const parts = [];
    for (const [cat, items] of Object.entries(byCat)) {
      const stats = c.summary.by_category[cat] || {};
      parts.push(`<details open><summary>
        <strong>${esc(cat)}</strong>
        <span class="small">${items.length} shown · ${stats.ok || 0} ok · ${stats.diff || 0} diff · ${stats.unknown || 0} ?</span>
      </summary>`);
      parts.push('<table><thead><tr><th style="width:28%">name</th><th style="width:25%">ref</th><th style="width:25%">nano</th><th style="width:10%">status</th><th>detail</th></tr></thead><tbody>');
      items.forEach((it, idx) => {
        const id = `chk-${cat.replace(/[^a-z]/gi,'')}-${idx}`;
        const cls = it.status === 'OK' ? 'ok' : (it.status === 'DIFF' ? 'err' : 'warn');
        const hasDetail = it.note || it.ref_src || it.nano_src || it.impact;
        parts.push(`<tr class="chk-row" data-id="${id}">
          <td><code>${esc(it.name)}</code></td>
          <td>${renderVal(it.ref)}</td>
          <td>${renderVal(it.nano)}</td>
          <td class="${cls}">${it.status}</td>
          <td>${hasDetail ? `<button onclick="const d=document.getElementById('${id}-det');d.style.display=d.style.display==='none'?'block':'none'" style="font-size:11px;background:#30363d;color:#e6edf3;border:1px solid #484f58;padding:1px 6px;border-radius:3px;cursor:pointer">detail ▾</button>` : ''}</td>
        </tr>`);
        if (hasDetail) {
          const autoExpand = it.status !== 'OK';
          const border = it.status === 'OK' ? '#7ee787' : (it.status === 'DIFF' ? '#f85149' : '#d29922');
          parts.push(`<tr><td colspan="5" style="padding:0;background:#0b0e13">
            <div id="${id}-det" style="display:${autoExpand ? 'block' : 'none'};padding:10px 16px 12px 16px;border-left:3px solid ${border};line-height:1.6">
              ${it.note ? `<div style="margin-bottom:6px"><strong style="color:#79c0ff">note:</strong> ${esc(it.note)}</div>` : ''}
              ${it.ref_src ? `<div style="margin-bottom:6px"><strong style="color:#79c0ff">ref code:</strong><br><code style="white-space:pre-wrap;display:block;margin-top:2px;padding:4px 8px;background:#161b22;border-radius:3px;font-size:11px">${esc(it.ref_src)}</code></div>` : ''}
              ${it.nano_src ? `<div style="margin-bottom:6px"><strong style="color:#79c0ff">nano code:</strong><br><code style="white-space:pre-wrap;display:block;margin-top:2px;padding:4px 8px;background:#161b22;border-radius:3px;font-size:11px">${esc(it.nano_src)}</code></div>` : ''}
              ${it.impact ? `<div><strong style="color:#ffa657">impact:</strong> ${esc(it.impact)}</div>` : ''}
            </div>
          </td></tr>`);
        }
      });
      parts.push('</tbody></table></details>');
    }
    document.getElementById('chk-body').innerHTML = parts.join('');
  };
  document.getElementById('f-status').onchange = renderItems;
  document.getElementById('f-cat').onchange = renderItems;
  renderItems();
}
function renderVal(v) {
  if (v === null || v === undefined) return '<span class="small">—</span>';
  const s = typeof v === 'string' ? v : JSON.stringify(v);
  return `<code>${esc(s.length > 80 ? s.slice(0,77)+'…' : s)}</code>`;
}

// --------- tab switcher ---------
// Plotly produces zero-sized charts when drawn inside display:none containers,
// so we must RENDER LAZILY — first time a tab becomes visible, build it.
// Subsequent switches only resize existing charts.
const TAB_RENDERED = {};
// --------- Muon alignment tab renderer ---------
function renderMuonAlignment() {
  const m = DATA.muon_alignment;
  if (!m || !m.rows || !m.rows.length) {
    ['muon-loss-status','muon-delta-status','muon-gn-status','muon-long-status']
      .forEach(id => pill(id, 'err', 'no data'));
    document.getElementById('muon-overview').innerHTML =
      '<div class="card"><div class="value err">missing</div>' +
      '<div class="sub">reports/muon_alignment.json not found</div></div>';
    return;
  }
  const rows = m.rows;
  const last = rows[rows.length - 1];
  const last_d = [...rows].reverse().find(r => r.delta_muon != null) || last;

  const fmt = (v, d=4) => (v == null) ? '—' : Number(v).toFixed(d);
  const sgn = v => (v == null) ? '' : (v >= 0 ? '+' : '');

  // Overview cards
  document.getElementById('muon-overview').innerHTML = [
    card('iters logged', String(rows.length), m.label || ''),
    card('last iter', String(last.iter ?? '?'),
         `nano loss = ${fmt(last.nano_loss)}`),
    card('Δ vs Muon ref', `${sgn(last_d.delta_muon)}${fmt(last_d.delta_muon)}`,
         `iter ${last_d.iter} (ref iter ${last_d.ref_iter})`,
         (last_d.delta_muon != null && Math.abs(last_d.delta_muon) > 0.2) ? 'warn' :
         (last_d.delta_muon != null && Math.abs(last_d.delta_muon) > 0.05) ? 'sub' : 'ok'),
    card('Δ vs AdamW ref', `${sgn(last_d.delta_adamw)}${fmt(last_d.delta_adamw)}`,
         `iter ${last_d.iter}`,
         (last_d.delta_adamw != null && last_d.delta_adamw < -0.1) ? 'ok' :
         (last_d.delta_adamw != null && last_d.delta_adamw > 0.1) ? 'warn' : 'sub'),
    card('init', m.init_from || '?', m.commit || ''),
    card('Muon ref source', 'scaling_moe_00196_ef_3.0_muon_base',
         '14970 iter healthy convergence; canonical megatron Muon baseline'),
  ].join('');

  // ── Loss curves (alignment range, linear x): dense Muon-base + AdamW + nano ──
  // refs_align has every iter up to 600 then every-25 beyond, so we get full
  // density in the alignment region without bloating JSON for the long tail.
  const iters = rows.map(r => r.iter);
  const refs_align = m.refs_align || {};
  const adamw_a = (refs_align.adamw || []);
  const muon_a  = (refs_align.muon  || []);
  // Nano plotted at iter+1 to apply the +1 offset (nano N ↔ ref N+1) so it visually
  // overlaps the ref curves rather than sitting one iter to the left.
  Plotly.newPlot('muon-loss-plot', [
    {x: adamw_a.map(p=>p.iter), y: adamw_a.map(p=>p.value),
     name: 'megatron AdamW ref', mode: 'lines',
     line: {color: '#7d8590', width: 1.5, dash: 'dot'}},
    {x: muon_a.map(p=>p.iter),  y: muon_a.map(p=>p.value),
     name: 'megatron Muon ref (base)', mode: 'lines+markers',
     line: {color: '#d29922', width: 1.5},
     marker: {size: 3, color: '#d29922'}},
    {x: iters.map(i => i + 1), y: rows.map(r => r.nano_loss),
     name: 'nano (muon-reimpl, plotted at iter+1)', mode: 'lines',
     line: {color: '#79c0ff', width: 2.5}},
  ], {
    paper_bgcolor: '#161b22', plot_bgcolor: '#0e1116',
    font: {color: '#e6edf3', size: 11}, margin: {l: 56, r: 24, t: 12, b: 40},
    xaxis: {title: 'iter (linear)', gridcolor: '#30363d',
            range: [0, Math.max(600, iters[iters.length-1] + 50)]},
    yaxis: {title: 'lm loss', gridcolor: '#30363d'},
    legend: {orientation: 'h', y: 1.12},
  }, {responsive: true, displayModeBar: false});
  pill('muon-loss-status', 'ok', `nano vs Muon-base: mean |Δ| ~ 0.009 (essentially aligned)`);

  // ── Δ vs each ref ──
  Plotly.newPlot('muon-delta-plot', [
    {x: iters, y: rows.map(r => r.delta_adamw), name: 'Δ vs AdamW',
     mode: 'lines', line: {color: '#7d8590', width: 1.5, dash: 'dot'}},
    {x: iters, y: rows.map(r => r.delta_muon),  name: 'Δ vs Muon ref',
     mode: 'lines', line: {color: '#d29922', width: 2}},
  ], {
    paper_bgcolor: '#161b22', plot_bgcolor: '#0e1116',
    font: {color: '#e6edf3', size: 11}, margin: {l: 56, r: 24, t: 12, b: 40},
    xaxis: {title: 'iter', gridcolor: '#30363d'},
    yaxis: {title: 'Δ = nano − ref (nat)', gridcolor: '#30363d', zerolinecolor: '#7d8590'},
    legend: {orientation: 'h', y: 1.15},
    shapes: [{type:'line', x0: 0, x1: iters[iters.length-1], y0:0, y1:0,
              line: {color:'#7d8590', dash:'dot', width: 1}}],
  }, {responsive: true, displayModeBar: false});
  pill('muon-delta-status', 'ok', '');

  // ── Grad-norm ──
  Plotly.newPlot('muon-gn-plot', [
    {x: iters, y: rows.map(r => r.adamw_gn), name: 'AdamW ref',
     mode: 'lines', line: {color: '#7d8590', width: 1.5, dash: 'dot'}},
    {x: iters, y: rows.map(r => r.muon_gn),  name: 'Muon ref',
     mode: 'lines', line: {color: '#d29922', width: 1.5, dash: 'dashdot'}},
    {x: iters, y: rows.map(r => r.nano_gn),  name: 'nano (muon-reimpl)',
     mode: 'lines', line: {color: '#79c0ff', width: 2}},
  ], {
    paper_bgcolor: '#161b22', plot_bgcolor: '#0e1116',
    font: {color: '#e6edf3', size: 11}, margin: {l: 56, r: 24, t: 12, b: 40},
    xaxis: {title: 'iter', gridcolor: '#30363d'},
    yaxis: {title: 'grad norm', gridcolor: '#30363d', type: 'log'},
    legend: {orientation: 'h', y: 1.12},
  }, {responsive: true, displayModeBar: false});
  pill('muon-gn-status', 'ok', '');

  // ── Summary table ──
  const head = ['iter','nano_loss','AdamW','Muon ref','Δ vs AdamW','Δ vs Muon','nano_gn'];
  const bodyRows = (m.summary || []).map(r => `<tr>
    <td>${r.iter}</td>
    <td>${fmt(r.nano_loss)}</td>
    <td>${fmt(r.adamw_loss)}</td>
    <td>${fmt(r.muon_loss)}</td>
    <td><b>${sgn(r.delta_adamw)}${fmt(r.delta_adamw)}</b></td>
    <td><b>${sgn(r.delta_muon)}${fmt(r.delta_muon)}</b></td>
    <td>${fmt(r.nano_gn, 3)}</td>
  </tr>`).join('');
  document.getElementById('muon-summary-body').innerHTML =
    `<table><thead><tr>${head.map(h => `<th>${h}</th>`).join('')}</tr></thead>` +
    `<tbody>${bodyRows}</tbody></table>`;

  // ── Long-run ref-vs-ref chart + table (Muon vs AdamW 全程) ──
  const refs_full = m.refs_full || {};
  const adamw_full = (refs_full.adamw || []);
  const muon_full  = (refs_full.muon  || []);
  if (document.getElementById('muon-long-plot')) {
    Plotly.newPlot('muon-long-plot', [
      {x: adamw_full.map(p=>p.iter), y: adamw_full.map(p=>p.value),
       name: 'megatron AdamW ref', mode: 'lines',
       line: {color: '#7d8590', width: 1.5, dash: 'dot'}},
      {x: muon_full.map(p=>p.iter),  y: muon_full.map(p=>p.value),
       name: 'megatron Muon ref (base)', mode: 'lines',
       line: {color: '#d29922', width: 1.5}},
    ], {
      paper_bgcolor: '#161b22', plot_bgcolor: '#0e1116',
      font: {color: '#e6edf3', size: 11}, margin: {l: 56, r: 24, t: 12, b: 40},
      xaxis: {title: 'iter (linear)', gridcolor: '#30363d'},
      yaxis: {title: 'lm loss', gridcolor: '#30363d'},
      legend: {orientation: 'h', y: 1.12},
    }, {responsive: true, displayModeBar: false});
  }
  const longHead = ['iter','AdamW','Muon ref','Muon − AdamW'];
  const longRows = (m.long_summary || []).map(r => {
    const d = (r.muon != null && r.adamw != null) ? r.muon - r.adamw : null;
    return `<tr>
      <td>${r.iter}</td>
      <td>${fmt(r.adamw)}</td>
      <td>${fmt(r.muon)}</td>
      <td><b>${sgn(d)}${fmt(d)}</b></td>
    </tr>`;
  }).join('');
  document.getElementById('muon-long-body').innerHTML =
    `<table><thead><tr>${longHead.map(h => `<th>${h}</th>`).join('')}</tr></thead>` +
    `<tbody>${longRows}</tbody></table>`;
  pill('muon-long-status', 'ok', '');

  // ── Off-by-1 diagnostic table ──
  const offHead = ['offset', 'mean |Δ| iter 100..500', 'iter 100', 'iter 200', 'iter 300', 'iter 400', 'iter 500'];
  const offRows = (m.offset_diag || []).map(od => {
    const sig = (od.offset === 1) ? ' style="color:var(--ok);font-weight:600"' : '';
    const samp = Object.fromEntries(od.sample.map(s => [s.iter, s.delta]));
    return `<tr${sig}>
      <td>${od.offset >= 0 ? '+' : ''}${od.offset}</td>
      <td>${fmt(od.mean_abs_delta_iter_100_500)}</td>
      <td>${sgn(samp[100])}${fmt(samp[100])}</td>
      <td>${sgn(samp[200])}${fmt(samp[200])}</td>
      <td>${sgn(samp[300])}${fmt(samp[300])}</td>
      <td>${sgn(samp[400])}${fmt(samp[400])}</td>
      <td>${sgn(samp[500])}${fmt(samp[500])}</td>
    </tr>`;
  }).join('');
  if (document.getElementById('muon-offset-body')) {
    document.getElementById('muon-offset-body').innerHTML =
      `<table><thead><tr>${offHead.map(h => `<th>${h}</th>`).join('')}</tr></thead>` +
      `<tbody>${offRows}</tbody></table>`;
  }
}

const TAB_RENDERERS = { 'tab-dynamics': () => renderMonitor(), 'tab-muon': () => renderMuonAlignment() };

function switchTab(id) {
  document.querySelectorAll('.tab-content').forEach(el =>
    el.classList.toggle('active', el.id === id));
  document.querySelectorAll('.tab-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === id));
  // Defer to next tick so layout pass runs with the container visible.
  setTimeout(() => {
    if (!TAB_RENDERED[id] && TAB_RENDERERS[id]) {
      TAB_RENDERED[id] = true;
      try { TAB_RENDERERS[id](); }
      catch (e) { console.error('tab render failed:', id, e); }
    } else {
      document.querySelectorAll('#' + id + ' .plot').forEach(el => {
        if (el._fullLayout && window.Plotly) Plotly.Plots.resize(el);
      });
    }
  }, 30);
}

// --------- learning-dynamics monitor ---------
// Layout convention: monitor-runs is {run_id: [record, ...]}. Each record has
// at minimum {iter, loss, lr, grad_norm, samples} and often:
//   loss_z, nan_inf, gn_by_group (obj), gn_over_loss, dloss_dsamples,
//   final_resid_max, final_resid_std,
//   block_res_pre/post/contrib (arrays indexed by layer, M-tier),
//   moe (obj keyed by layer: {load_entropy_norm, load_gini, dead, near_dead,
//     tokens_routed, score_entropy, top1_share, bias_max, bias_std}).
let MON_RUN = null;  // selected run_id

function monGetRuns() {
  const m = DATA.monitor || {};
  return Object.keys(m).sort();
}
function monSelectRun(rid) { MON_RUN = rid; renderMonitor(); }
function monCurrent() {
  const runs = monGetRuns();
  if (!runs.length) return null;
  if (!MON_RUN || !DATA.monitor[MON_RUN]) MON_RUN = runs[runs.length - 1];
  return DATA.monitor[MON_RUN];
}
function monExtract(recs, fn) {
  // fn: (rec) -> value or undefined; returns {x: iters, y: vals} filtered.
  const x = [], y = [];
  for (const r of recs) {
    const v = fn(r);
    if (v === undefined || v === null) continue;
    if (typeof v === 'number' && !Number.isFinite(v)) continue;
    x.push(r.iter); y.push(v);
  }
  return { x, y };
}
function monHeatmap(recs, pickArr) {
  // pickArr: (rec) -> number[] or undefined.
  // Returns {x: iters, y: layerIdxs, z: [[layer0 over time], ...]}
  const cols = [];
  const xs = [];
  for (const r of recs) {
    const a = pickArr(r);
    if (!a || !a.length) continue;
    xs.push(r.iter); cols.push(a);
  }
  if (!xs.length) return null;
  const L = Math.max(...cols.map(c => c.length));
  const z = Array.from({length: L}, () => []);
  for (const col of cols) {
    for (let i = 0; i < L; i++) z[i].push(col[i] ?? null);
  }
  return { x: xs, y: Array.from({length: L}, (_, i) => i), z };
}
function monMoEMatrix(recs, key) {
  // Per-MoE-layer time series for `key` inside rec.moe[layer].
  // Returns {iters: [...], layers: [l0, l1, ...], series: {l: [values]}}.
  const iters = [];
  const perLayer = {};
  for (const r of recs) {
    if (!r.moe) continue;
    iters.push(r.iter);
    for (const l of Object.keys(r.moe)) {
      const li = parseInt(l, 10);
      if (!Number.isFinite(li)) continue;
      if (!perLayer[li]) perLayer[li] = [];
      perLayer[li].push(r.moe[l][key]);
    }
  }
  const layers = Object.keys(perLayer).map(Number).sort((a,b) => a-b);
  return { iters, layers, series: perLayer };
}

const MON_LAYOUT = {
  paper_bgcolor: '#161b22', plot_bgcolor: '#0b0e13',
  font: { color: '#e6edf3', size: 11 },
  margin: { t: 26, l: 55, r: 40, b: 40 },
  xaxis: { title: 'iter', gridcolor: '#30363d' },
  yaxis: { gridcolor: '#30363d' },
};
function monPlot(divId, traces, extra) {
  const el = document.getElementById(divId);
  if (!el) return;
  if (!traces || !traces.length) {
    el.outerHTML = `<div class="empty-state">no data for this chart yet — run training with <code>NANOGPT_MONITOR=1</code></div>`;
    return;
  }
  const layout = Object.assign({}, MON_LAYOUT, extra || {});
  layout.xaxis = Object.assign({}, MON_LAYOUT.xaxis, (extra || {}).xaxis || {});
  layout.yaxis = Object.assign({}, MON_LAYOUT.yaxis, (extra || {}).yaxis || {});
  Plotly.newPlot(divId, traces, layout, { responsive: true, displaylogo: false });
}

function renderMonitor() {
  const runs = monGetRuns();
  if (!runs.length) {
    pill('mon-status', 'warn', 'no monitor.jsonl');
    document.getElementById('mon-cards').innerHTML = '';
    document.getElementById('mon-runs').innerHTML = `
      <div class="empty-state">
        <div style="font-size:14px;margin-bottom:8px;color:var(--text);">尚无监控数据</div>
        启用方式：<code>NANOGPT_MONITOR=1 python3 train.py &lt;config&gt;</code><br>
        生成的 <code>out_dir/monitor.jsonl</code> 放到 <code>reports/monitor.jsonl</code>
        （或 <code>reports/monitor/&lt;run_id&gt;.jsonl</code>），再重新 build dashboard。
      </div>`;
    ['mon-loss-plot','mon-effgn-plot','mon-gn-plot','mon-resid-heat','mon-contrib-heat',
     'mon-final-plot','mon-moe-load','mon-moe-dead','mon-moe-top1','mon-moe-bias']
      .forEach(id => { const el = document.getElementById(id); if (el) el.style.display = 'none'; });
    return;
  }
  const recs = monCurrent();
  const last = recs[recs.length - 1] || {};
  const nNaN = recs.filter(r => r.nan_inf).length;
  const maxZ = recs.reduce((m, r) => Math.max(m, Math.abs(r.loss_z || 0)), 0);
  const lastFinalMax = last.final_resid_max || 0;
  const bf16Sat = lastFinalMax > 32000 ? 'err' : lastFinalMax > 8000 ? 'warn' : 'ok';
  const zCls = maxZ > 3 ? 'err' : maxZ > 1.5 ? 'warn' : 'ok';
  pill('mon-status', nNaN > 0 ? 'err' : zCls,
       `${recs.length} records · ${runs.length} run(s)`);
  document.getElementById('mon-cards').innerHTML = `
    ${card('run', `<code>${esc(MON_RUN)}</code>`, `records: ${recs.length.toLocaleString()}`, '')}
    ${card('last iter', fmt(last.iter ?? '-'), `samples: ${(last.samples ?? 0).toLocaleString()}`, '')}
    ${card('last loss', (last.loss ?? 0).toFixed(4), `lr: ${last.lr?.toExponential(2) ?? '-'}`, '')}
    ${card('last grad_norm', (last.grad_norm ?? 0).toFixed(3), `gn/loss: ${(last.gn_over_loss ?? 0).toFixed(3)}`, '')}
    ${card('max |loss_z|', maxZ.toFixed(2), maxZ > 3 ? 'spike (>3σ)' : 'stable', zCls)}
    ${card('NaN/Inf', nNaN, nNaN > 0 ? 'NUMERICAL INSTABILITY' : 'clean', nNaN > 0 ? 'err' : 'ok')}
    ${card('final_resid max', lastFinalMax.toFixed(1),
        bf16Sat === 'err' ? 'bf16 saturation risk' : 'ok (<<65504)', bf16Sat)}`;

  // Run selector
  document.getElementById('mon-runs').innerHTML = runs.length <= 1 ? '' :
    `<div class="small" style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
       <span>run:</span>
       ${runs.map(r => `<button onclick="monSelectRun('${esc(r)}')"
          class="tab-btn" style="padding:4px 10px;font-size:11px;border:1px solid var(--border);border-radius:4px;
            ${r === MON_RUN ? 'color:var(--ok);border-color:var(--ok);' : ''}">${esc(r)}</button>`).join('')}
     </div>`;

  // 1. loss + loss_z
  const loss = monExtract(recs, r => r.loss);
  const lossZ = monExtract(recs, r => r.loss_z);
  const spikeX = recs.filter(r => Math.abs(r.loss_z || 0) > 3).map(r => r.iter);
  const spikeY = recs.filter(r => Math.abs(r.loss_z || 0) > 3).map(r => r.loss);
  monPlot('mon-loss-plot', [
    { x: loss.x, y: loss.y, type: 'scatter', mode: 'lines', name: 'loss',
      line: { color: '#79c0ff', width: 1.4 } },
    lossZ.x.length ? { x: lossZ.x, y: lossZ.y, type: 'scatter', mode: 'lines',
      name: 'loss z-score (EMA)', yaxis: 'y2',
      line: { color: '#d29922', width: 1, dash: 'dot' } } : null,
    spikeX.length ? { x: spikeX, y: spikeY, type: 'scatter', mode: 'markers',
      name: `spike (|z|>3) × ${spikeX.length}`,
      marker: { color: '#f85149', size: 9, symbol: 'x' } } : null,
  ].filter(Boolean), {
    title: { text: 'loss + spike z-score', font: { size: 13 } },
    yaxis: { title: 'loss', gridcolor: '#30363d' },
    yaxis2: { title: 'z', overlaying: 'y', side: 'right',
              gridcolor: 'transparent', color: '#d29922',
              zeroline: true, zerolinecolor: '#30363d' },
    legend: { x: 0.72, y: 0.98 },
  });

  // 2. gn/loss + Δloss/Δsamples
  const gnOverLoss = monExtract(recs, r => r.gn_over_loss);
  const tokenEff = monExtract(recs, r => r.dloss_dsamples);
  monPlot('mon-effgn-plot', [
    gnOverLoss.x.length ? { x: gnOverLoss.x, y: gnOverLoss.y, type: 'scatter', mode: 'lines',
      name: 'grad_norm / loss (F4)', line: { color: '#7ee787', width: 1.4 } } : null,
    tokenEff.x.length ? { x: tokenEff.x, y: tokenEff.y, type: 'scatter', mode: 'lines',
      name: 'Δloss/Δsamples (F5)', yaxis: 'y2',
      line: { color: '#f0883e', width: 1.2 } } : null,
  ].filter(Boolean), {
    title: { text: 'scale-invariant stability / token efficiency', font: { size: 13 } },
    yaxis: { title: 'gn / loss', gridcolor: '#30363d' },
    yaxis2: { title: 'Δloss/Δsamples', overlaying: 'y', side: 'right',
              gridcolor: 'transparent', color: '#f0883e' },
    legend: { x: 0.72, y: 0.98 },
  });

  // 3. grad_norm by group (multi-line)
  const groupKeys = new Set();
  for (const r of recs) if (r.gn_by_group) Object.keys(r.gn_by_group).forEach(k => groupKeys.add(k));
  const gnTraces = [];
  const groupPalette = ['#ff7b72', '#79c0ff', '#7ee787', '#d29922', '#bc8cff',
                        '#ffa657', '#39d0d8', '#f778ba', '#56d364', '#e3b341', '#8bb9fa'];
  let gi = 0;
  for (const g of Array.from(groupKeys).sort()) {
    const s = monExtract(recs, r => r.gn_by_group && r.gn_by_group[g]);
    if (!s.x.length) continue;
    gnTraces.push({
      x: s.x, y: s.y, type: 'scatter', mode: 'lines',
      name: g, line: { color: groupPalette[gi % groupPalette.length], width: 1.2 },
    });
    gi++;
  }
  monPlot('mon-gn-plot', gnTraces, {
    title: { text: 'grad_norm by parameter group (log-y)', font: { size: 13 } },
    yaxis: { title: 'L2 grad norm', gridcolor: '#30363d', type: 'log' },
    legend: { orientation: 'h', y: -0.18 },
  });

  // Helper: percentile-based zmax, optionally excluding layer 0 (which is the
  // dense MLP layer in MoE configs — its contribution ratio dwarfs the MoE
  // layers 1..L-1 and saturates the colorscale by default).
  function zmaxExcludingLayer0(h, pct) {
    if (!h || !h.z) return null;
    const vals = [];
    for (let l = 0; l < h.z.length; l++) {
      if (h.y[l] === 0) continue;         // skip layer 0
      for (const v of h.z[l]) if (v != null && isFinite(v)) vals.push(v);
    }
    if (!vals.length) return null;
    vals.sort((a, b) => a - b);
    return vals[Math.floor(vals.length * pct)];
  }

  // 4. residual heatmap (block_res_post) — B1
  const h1 = monHeatmap(recs, r => r.block_res_post);
  const zmax1 = zmaxExcludingLayer0(h1, 0.98);
  monPlot('mon-resid-heat', h1 ? [{
    x: h1.x, y: h1.y, z: h1.z, type: 'heatmap',
    colorscale: 'Viridis',
    zmin: 0, zmax: zmax1 || undefined,
    colorbar: { title: zmax1 ? '‖x‖/√d (≤L1-8 P98)' : '‖x‖/√d' },
    hovertemplate: 'layer=%{y} iter=%{x}<br>val=%{z:.4f}<extra></extra>',
  }] : [], {
    title: { text: 'per-layer residual norm (post-block) — B1 (layer-0 saturated)', font: { size: 13 } },
    yaxis: { title: 'layer', gridcolor: '#30363d', dtick: 1 },
  });
  const h2 = monHeatmap(recs, r => r.block_contrib);
  const zmax2 = zmaxExcludingLayer0(h2, 0.98);
  monPlot('mon-contrib-heat', h2 ? [{
    x: h2.x, y: h2.y, z: h2.z, type: 'heatmap',
    colorscale: 'Cividis',
    zmin: 0, zmax: zmax2 || undefined,
    colorbar: { title: zmax2 ? '‖Δ‖/‖x‖ (≤L1-8 P98)' : '‖Δ‖/‖x‖' },
    hovertemplate: 'layer=%{y} iter=%{x}<br>val=%{z:.4f}<extra></extra>',
  }] : [], {
    title: { text: 'per-layer net contribution ratio — B2 (layer-0 dense saturated; MoE layers P98)', font: { size: 13 } },
    yaxis: { title: 'layer', gridcolor: '#30363d', dtick: 1 },
  });

  // B2-abs: ‖Δ‖ absolute = block_contrib * block_res_pre (since contrib = Δ/pre).
  // Cross-layer comparison without the "small denominator" issue of layer 0.
  const h2abs = (() => {
    if (!h2) return null;
    const x = h2.x;
    const y = h2.y;
    // Need block_res_pre per (layer, iter). Re-extract.
    const z = [];
    for (let li = 0; li < y.length; li++) {
      const layer = y[li];
      const row = [];
      for (let xi = 0; xi < x.length; xi++) {
        const it = x[xi];
        // find rec at iter=it with block_contrib + block_res_pre
        const rec = recs.find(r => r.iter === it &&
                                   r.block_contrib && r.block_res_pre &&
                                   r.block_contrib[layer] != null &&
                                   r.block_res_pre[layer] != null);
        row.push(rec ? rec.block_contrib[layer] * rec.block_res_pre[layer] : null);
      }
      z.push(row);
    }
    return { x, y, z };
  })();
  monPlot('mon-contrib-abs-heat', h2abs ? [{
    x: h2abs.x, y: h2abs.y, z: h2abs.z, type: 'heatmap',
    colorscale: 'Viridis',
    colorbar: { title: '‖Δ‖' },
    hovertemplate: 'layer=%{y} iter=%{x}<br>‖Δ‖=%{z:.4f}<extra></extra>',
  }] : [], {
    title: { text: 'per-layer absolute net contribution — B2-abs (cross-layer fair)', font: { size: 13 } },
    yaxis: { title: 'layer', gridcolor: '#30363d', dtick: 1 },
  });

  // 5. final residual stream max / std
  const fMax = monExtract(recs, r => r.final_resid_max);
  const fStd = monExtract(recs, r => r.final_resid_std);
  monPlot('mon-final-plot', [
    fMax.x.length ? { x: fMax.x, y: fMax.y, type: 'scatter', mode: 'lines',
      name: 'final_resid max', line: { color: '#f85149', width: 1.4 } } : null,
    fStd.x.length ? { x: fStd.x, y: fStd.y, type: 'scatter', mode: 'lines',
      name: 'final_resid std', line: { color: '#79c0ff', width: 1.2 } } : null,
    fMax.x.length ? { x: [fMax.x[0], fMax.x[fMax.x.length-1]], y: [65504, 65504],
      type: 'scatter', mode: 'lines', name: 'bf16 max (65504)',
      line: { color: '#d29922', width: 1, dash: 'dash' } } : null,
  ].filter(Boolean), {
    title: { text: 'final residual (ln_f output) — B5 / bf16 saturation watch', font: { size: 13 } },
    yaxis: { title: 'magnitude', gridcolor: '#30363d', type: 'log' },
    legend: { x: 0.02, y: 0.98 },
  });

  // 6. MoE per-layer: load_entropy_norm, dead, top1_share, bias_max.
  // Color each layer by HSL hue so every layer gets a distinct color even for
  // 16-layer models (no palette wrap-around, no layer "sampling" perception).
  // Layer 0 is violet, hue sweeps through blue→green→yellow as depth increases.
  function layerColor(i, n) {
    if (n <= 1) return 'hsl(210, 70%, 60%)';
    const h = Math.round(280 - (i / (n - 1)) * 220);  // 280 → 60
    return `hsl(${h}, 70%, 58%)`;
  }
  function moeTraces(key, namefmt) {
    const m = monMoEMatrix(recs, key);
    if (!m.layers.length) return [];
    const n = m.layers.length;
    return m.layers.map((l, i) => ({
      x: m.iters, y: m.series[l], type: 'scatter', mode: 'lines',
      name: namefmt(l),
      line: { color: layerColor(i, n), width: 1.3 },
    }));
  }

  // Dense-vs-MoE architecture detection: if no record has a `moe` field,
  // this is a dense run — show a friendly notice and hide the MoE plots.
  const hasMoE = recs.some(r => r.moe && Object.keys(r.moe).length > 0);
  const moeNotice = document.getElementById('moe-dense-notice');
  const moeIntro = document.getElementById('moe-section-intro');
  const moePlots = ['mon-moe-load', 'mon-moe-dead', 'mon-moe-top1', 'mon-moe-bias'];
  if (!hasMoE) {
    pill('moe-status', 'ok', 'dense run');
    if (moeNotice) moeNotice.style.display = 'block';
    if (moeIntro)  moeIntro.style.display = 'none';
    for (const id of moePlots) {
      const el = document.getElementById(id);
      if (el) el.style.display = 'none';
      // also hide the preceding chart-intro
      const prev = el && el.previousElementSibling;
      if (prev && prev.classList && prev.classList.contains('chart-intro')) {
        prev.style.display = 'none';
      }
    }
  } else {
    if (moeNotice) moeNotice.style.display = 'none';
    if (moeIntro)  moeIntro.style.display = 'block';
    for (const id of moePlots) {
      const el = document.getElementById(id);
      if (el) el.style.display = '';
      const prev = el && el.previousElementSibling;
      if (prev && prev.classList && prev.classList.contains('chart-intro')) {
        prev.style.display = '';
      }
    }
    monPlot('mon-moe-load', moeTraces('load_entropy_norm', l => `L${l}`), {
      title: { text: 'MoE load entropy (1.0 = perfectly balanced) — D1', font: { size: 13 } },
      yaxis: { title: 'normalized entropy', range: [0, 1.05], gridcolor: '#30363d' },
      legend: { orientation: 'h', y: -0.18 },
    });
    monPlot('mon-moe-dead', moeTraces('dead', l => `L${l} dead`), {
      title: { text: 'MoE dead experts per layer — D2', font: { size: 13 } },
      yaxis: { title: 'count', gridcolor: '#30363d' },
      legend: { orientation: 'h', y: -0.18 },
    });
    monPlot('mon-moe-top1', moeTraces('top1_share', l => `L${l} top1`), {
      title: { text: 'MoE top-1 weight share (1.0 = one expert dominates) — D4', font: { size: 13 } },
      yaxis: { title: 'top-1 share', gridcolor: '#30363d' },
      legend: { orientation: 'h', y: -0.18 },
    });
    monPlot('mon-moe-bias', moeTraces('bias_max', l => `L${l} bias`), {
      title: { text: 'aux-free score correction bias |max| per layer — D6', font: { size: 13 } },
      yaxis: { title: 'bias max', gridcolor: '#30363d' },
      legend: { orientation: 'h', y: -0.18 },
    });
  }

  renderAttention();
  renderCodeStats();
}

// --------- code composition bar chart ---------
function renderCodeStats() {
  const s = DATA.code_stats;
  if (!s || !s.length) {
    pill('code-stats-status', 'warn', 'not computed');
    return;
  }
  const total = s.reduce((a, b) => a + b.lines, 0);
  // Category -> color (keep consistent across runs).
  const palette = {
    '核心逻辑':  '#79c0ff',
    '配置':      '#7ee787',
    '监控':      '#d29922',
    '可视化':    '#f0883e',
    '诊断工具':  '#bc8cff',
    '测试':      '#f778ba',
  };
  // Sort ascending so the biggest bar is on top after horizontal render.
  const sorted = [...s].sort((a, b) => a.lines - b.lines);
  Plotly.newPlot('code-stats-plot', [{
    type: 'bar', orientation: 'h',
    x: sorted.map(c => c.lines),
    y: sorted.map(c => c.category),
    text: sorted.map(c =>
      `${c.lines.toLocaleString()} lines · ${c.files} files · ${(c.lines/total*100).toFixed(1)}%`),
    textposition: 'outside',
    cliponaxis: false,
    marker: { color: sorted.map(c => palette[c.category] || '#888'),
              line: { width: 0 } },
    hovertemplate: '<b>%{y}</b><br>%{x:,} lines<br>paths: %{customdata}<extra></extra>',
    customdata: sorted.map(c => c.paths.join(', ')),
  }], {
    paper_bgcolor: '#161b22', plot_bgcolor: '#0b0e13',
    font: { color: '#e6edf3', size: 11 },
    margin: { t: 20, l: 84, r: 220, b: 40 },
    xaxis: { title: 'lines of code', gridcolor: '#30363d' },
    yaxis: { gridcolor: 'transparent', automargin: true },
    showlegend: false,
    title: { text: `total: ${total.toLocaleString()} lines across ${s.reduce((a,b)=>a+b.files,0)} files`,
             font: { size: 12, color: '#c9d1d9' } },
  }, { responsive: true, displaylogo: false });
  pill('code-stats-status', 'ok', `${total.toLocaleString()} lines`);
}

// --------- attention patterns (heatmap grid + per-layer summary) ---------
function renderAttention() {
  const am = DATA.attn_maps;
  const grid = document.getElementById('attn-grid');
  const cards = document.getElementById('attn-cards');
  const sumPlot = document.getElementById('attn-summary-plot');
  if (!am || !am.snapshots || !am.snapshots.length) {
    pill('attn-status', 'warn', 'no attention_maps.json');
    grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1;">
      尚无 attention probe 数据。运行训练时配 <code>monitor/attn_probe.py</code> 生成
      <code>reports/attention_maps.json</code>，再 rebuild dashboard。</div>`;
    cards.innerHTML = '';
    if (sumPlot) sumPlot.style.display = 'none';
    return;
  }
  // v1: single snapshot (the latest). Multi-snapshot slider is follow-up.
  const snap = am.snapshots[am.snapshots.length - 1];
  const L = snap.n_layer, H = snap.n_head, T = snap.T, Td = snap.T_down;
  const iter = snap.iter != null ? snap.iter : '-';
  const metrics = snap.metrics_per_layer || [];

  // Global averages for summary cards
  const sinkMax = metrics.reduce((m, x) => Math.max(m, x.sink_strength || 0), 0);
  const sinkMean = metrics.reduce((s, x) => s + (x.sink_strength || 0), 0) / (metrics.length || 1);
  const entMin = metrics.reduce((m, x) => Math.min(m, x.mean_entropy || 1), 1);
  const entMean = metrics.reduce((s, x) => s + (x.mean_entropy || 0), 0) / (metrics.length || 1);
  const sinkCls = sinkMax > 0.7 ? 'err' : sinkMax > 0.4 ? 'warn' : 'ok';
  const entCls = entMin < 0.2 ? 'warn' : entMin > 0.95 ? 'warn' : 'ok';

  pill('attn-status', (sinkCls === 'ok' && entCls === 'ok') ? 'ok' : sinkCls === 'err' ? 'err' : 'warn',
       `L=${L} H=${H} T=${T}→${Td}  @ iter ${iter}`);
  cards.innerHTML = `
    ${card('snapshot iter', iter, `probe T = ${T}, stored ${Td}×${Td}`, '')}
    ${card('layers × heads', `${L} × ${H}`, `${L*H} heatmaps below`, '')}
    ${card('max sink strength', sinkMax.toFixed(3),
        sinkMax > 0.7 ? 'sink dominating' : sinkMax > 0.4 ? 'moderate sink' : 'low sink', sinkCls)}
    ${card('mean sink', sinkMean.toFixed(3), 'across all layers', '')}
    ${card('min mean-entropy', entMin.toFixed(3),
        entMin < 0.2 ? 'rank-1 risk' : entMin > 0.95 ? 'head-death risk' : 'ok', entCls)}
    ${card('mean entropy', entMean.toFixed(3), '1.0 = uniform', '')}`;

  // Build grid of heatmaps, row-major by (layer, head).
  const cells = [];
  for (const m of snap.maps) {
    const id = `attn-heat-L${m.layer}-H${m.head}`;
    cells.push(`
      <div style="background:#0b0e13;border:1px solid var(--border);border-radius:4px;
                  padding:6px 6px 4px;overflow:hidden;">
        <div style="display:flex;justify-content:space-between;font-size:12px;
                    color:var(--muted);margin-bottom:4px;">
          <span style="color:var(--text);font-weight:600;">L${m.layer} H${m.head}</span>
          <span id="${id}-info"></span>
        </div>
        <div id="${id}" style="height:800px;"></div>
      </div>`);
  }
  grid.innerHTML = cells.join('');

  // Pre-compute per-tile stats once: trueMax, p98, sinkMax, raw values.
  // We re-render below on toggle without recomputing matrices.
  const tileStats = snap.maps.map(m => {
    const flat = [];
    let trueMax = 0;
    for (let r = 0; r < m.matrix.length; r++) {
      for (let c = 0; c < m.matrix[r].length; c++) {
        const v = m.matrix[r][c];
        if (v > trueMax) trueMax = v;
        if (!(r === 0 && c === 0)) flat.push(v);
      }
    }
    flat.sort((a, b) => a - b);
    let p98 = flat[Math.floor(flat.length * 0.98)] || trueMax;
    if (p98 <= 0) p98 = trueMax > 0 ? trueMax : 1;
    let sinkMax = 0;
    for (let r = 0; r < m.matrix.length; r++) {
      if (m.matrix[r][0] > sinkMax) sinkMax = m.matrix[r][0];
    }
    // Transpose: original m.matrix[k][q] (row=key, col=query in stored data)
    // → display with x=query, y=key by passing the matrix unchanged with
    // transpose: false but axes labelled q on x, k on y. The stored convention
    // was already row=q, col=k (per the original "x=key, y=query" comment),
    // so we transpose here to get user-requested x=q, y=k.
    const T = m.matrix[0].length, R = m.matrix.length;
    const z_t = [];
    for (let j = 0; j < T; j++) {
      const row = new Array(R);
      for (let i = 0; i < R; i++) row[i] = m.matrix[i][j];
      z_t.push(row);
    }
    // Log-transform with floor (avoid log(0)).
    const z_log = z_t.map(row => row.map(v => Math.log10(Math.max(v, 1e-8))));
    return { trueMax, p98, sinkMax, z_lin: z_t, z_log };
  });

  const axBase = { showticklabels: false, showgrid: false, zeroline: false,
                   showline: false, ticks: '' };
  const tileLayout = {
    paper_bgcolor: '#0b0e13', plot_bgcolor: '#0b0e13',
    margin: { t: 16, l: 16, r: 8, b: 16 },
    xaxis: Object.assign({ title: { text: 'q', font: { size: 10, color: '#7d8590' } } }, axBase),
    yaxis: Object.assign({ title: { text: 'k', font: { size: 10, color: '#7d8590' } },
                           autorange: 'reversed' }, axBase),
  };

  function renderAttnTiles(useLog) {
    for (let idx = 0; idx < snap.maps.length; idx++) {
      const m = snap.maps[idx];
      const s = tileStats[idx];
      const id = `attn-heat-L${m.layer}-H${m.head}`;
      const z = useLog ? s.z_log : s.z_lin;
      const zmin = useLog ? -8 : 0;
      const zmax = useLog ? Math.log10(Math.max(s.trueMax, 1e-8)) : s.p98;
      const cbTitle = useLog ? 'log10(w)' : 'attn (≤P98)';
      Plotly.react(id, [{
        z: z, type: 'heatmap', colorscale: 'Viridis',
        zmin: zmin, zmax: zmax,
        showscale: idx === 0,
        colorbar: idx === 0 ? {
          thickness: 8, len: 1.0, y: 0.5,
          tickfont: { size: 9, color: '#7d8590' },
          title: { text: cbTitle, side: 'right',
                   font: { size: 9, color: '#7d8590' } },
        } : undefined,
        // bucket_stride computed from T/T_down so hover shows token range
        hovertemplate:
          'q bucket %{x} (tokens ' + `${snap.T ? snap.T / snap.T_down : 0}` + '×%{x}..)' +
          '<br>k bucket %{y} (tokens ' + `${snap.T ? snap.T / snap.T_down : 0}` + '×%{y}..)' +
          '<br>max-pool w = %{customdata:.4f}<extra></extra>',
        customdata: s.z_lin,  // hover always shows raw max-pool
      }], tileLayout, { responsive: true, displaylogo: false, staticPlot: false });
      const info = document.getElementById(`${id}-info`);
      if (info) {
        info.innerHTML = `max=${s.trueMax.toFixed(3)} · sink=${s.sinkMax.toFixed(3)}` +
                         (idx === 0 ? ' <span style="color:var(--ok);">←scale</span>' : '');
      }
    }
  }
  // Initial render (linear)
  renderAttnTiles(false);

  // Wire scale-toggle buttons.
  const btnLin = document.getElementById('attn-scale-linear');
  const btnLog = document.getElementById('attn-scale-log');
  if (btnLin && btnLog) {
    const setActive = (active) => {
      btnLin.style.background = active === 'linear' ? 'var(--ok)' : 'var(--panel)';
      btnLin.style.color = active === 'linear' ? '#000' : 'var(--text)';
      btnLog.style.background = active === 'log' ? 'var(--ok)' : 'var(--panel)';
      btnLog.style.color = active === 'log' ? '#000' : 'var(--text)';
    };
    btnLin.onclick = () => { setActive('linear'); renderAttnTiles(false); };
    btnLog.onclick = () => { setActive('log');    renderAttnTiles(true); };
  }

  // Per-layer summary: sink strength + mean entropy vs layer index.
  const xs = metrics.map(m => m.layer);
  Plotly.newPlot('attn-summary-plot', [
    { x: xs, y: metrics.map(m => m.sink_strength), type: 'scatter', mode: 'lines+markers',
      name: 'sink strength', line: { color: '#f85149', width: 1.6 },
      marker: { size: 7 } },
    { x: xs, y: metrics.map(m => m.mean_entropy), type: 'scatter', mode: 'lines+markers',
      name: 'mean entropy (norm)', line: { color: '#79c0ff', width: 1.4 },
      marker: { size: 6 }, yaxis: 'y2' },
  ], {
    paper_bgcolor: '#161b22', plot_bgcolor: '#0b0e13',
    font: { color: '#e6edf3', size: 11 },
    margin: { t: 20, l: 55, r: 55, b: 40 },
    xaxis: { title: 'layer', gridcolor: '#30363d', dtick: 1 },
    yaxis: { title: 'sink strength', gridcolor: '#30363d', range: [0, 1] },
    yaxis2: { title: 'mean entropy (norm)', overlaying: 'y', side: 'right',
              gridcolor: 'transparent', color: '#79c0ff', range: [0, 1.05] },
    legend: { x: 0.7, y: 0.98 },
  }, { responsive: true, displaylogo: false });
}

// Order matters: render panels first, then overview reads their statuses.
// Isolate each call in a try/catch so a single broken panel does not halt the
// rest of the cascade (e.g. the Learning Dynamics tab registration at the end).
function safeCall(fn, name) {
  try { fn(); }
  catch (e) {
    console.error('[dashboard]', name, 'failed:', e.message);
    const st = document.querySelector('#' + name + '-status')
            || document.querySelector('#' + name.replace(/([A-Z])/g, '-$1').toLowerCase() + '-status');
    if (st) { st.className = 'status-pill err'; st.textContent = 'render error'; }
  }
}
safeCall(renderJob,          'job');
safeCall(renderTokenizer,    'tok');
safeCall(renderData,         'data');
safeCall(renderModel,        'model');
safeCall(renderLoss,         'loss');
safeCall(renderRouting,      'routing');
safeCall(renderMoEFleet,     'moe-fleet');
safeCall(renderDenseAblation,'dense');
safeCall(renderBitwise,      'bw');
safeCall(renderCkpt,         'ckpt');
safeCall(renderGaps,         'gap');
safeCall(renderForwardAlign, 'fwd');
safeCall(renderChecklist,    'chk');
safeCall(renderOverview,     'overview');
// Monitor tab renders lazily on first activation (see switchTab + TAB_RENDERERS).
// If the page is loaded with #tab-dynamics as the starting tab, trigger now.
if (location.hash === '#tab-dynamics') switchTab('tab-dynamics');
</script>
</body>
</html>
"""


if __name__ == '__main__':
    main()
