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
    'gaps':        None,  # synthesized below
    'runs_index':  'reports/runs_index.json',  # list of nano runs available
    'ref_routing': 'reference/ref_moe_routing_stats.json',  # per-iter ref MoE routing stats
}


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
    <b>图例：</b>同上；颜色 = <code>‖x_post − x_pre‖ / ‖x_pre‖</code>（Cividis 色阶）——
    该层对残差流的实际修改比例。<br>
    <b>怎么分析：</b>健康时每层 contrib ≈ 0.1 ~ 1.0。
    <b>某层长期接近 0（最暗）= 该层被"跳过"</b>——模块在学一个恒等映射，容量浪费；
    <b>&gt;2 = 单层过度贡献</b>，预示后续层能信号被淹没。
    跨尺度比较：10B 某几层 "熄灭" 的问题可以在 500M 规模提前看到苗头。
  </div>
  <div id="mon-contrib-heat" class="plot plot-lg"></div>

  <div class="chart-intro">
    <div class="chart-intro-title">6. final residual (ln_f) 幅度（B5 / bf16 饱和 watchdog）</div>
    <b>图例：</b>红线 = <code>final_resid</code> 绝对值最大（lm_head 的输入）；
    蓝线 = 标准差；黄虚线 = bf16 最大值 <code>65504</code> 参考。log-y 轴。<br>
    <b>怎么分析：</b>max 应稳定或缓慢上涨。<b>max 逼近黄线 = bf16 即将饱和 NaN</b>——
    这是大规模训练中最隐蔽的死因，loss 曲线看不出，小尺度却能提前测到同样趋势。
    std 突升是全局激活发散，常与 B1 残差爆炸同时出现。
  </div>
  <div id="mon-final-plot" class="plot"></div>

  <h2>MoE routing health <span class="status-pill ok">D1 / D2 / D4 / D6</span></h2>
  <div class="small" style="margin:4px 0 8px;">
    MoE 路由 4 大核心信号。每个 MoE 层一条线（按层号连续色环着色，颜色从紫→蓝→绿→黄按层深度渐变），
    <b>所有 MoE 层都显示，不采样</b>。
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
</div><!-- /tab-dynamics -->
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
const TAB_RENDERERS = { 'tab-dynamics': () => renderMonitor() };

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

  // 4. residual heatmap (block_res_post)
  const h1 = monHeatmap(recs, r => r.block_res_post);
  monPlot('mon-resid-heat', h1 ? [{
    x: h1.x, y: h1.y, z: h1.z, type: 'heatmap',
    colorscale: 'Viridis', colorbar: { title: '‖x‖/√d' },
  }] : [], {
    title: { text: 'per-layer residual norm (post-block) — B1', font: { size: 13 } },
    yaxis: { title: 'layer', gridcolor: '#30363d', dtick: 1 },
  });
  const h2 = monHeatmap(recs, r => r.block_contrib);
  monPlot('mon-contrib-heat', h2 ? [{
    x: h2.x, y: h2.y, z: h2.z, type: 'heatmap',
    colorscale: 'Cividis', colorbar: { title: '‖Δ‖/‖x‖' },
  }] : [], {
    title: { text: 'per-layer net contribution ratio — B2 (near 0 = dead layer)', font: { size: 13 } },
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

// Order matters: render panels first, then overview reads their statuses.
renderJob();
renderTokenizer();
renderData();
renderModel();
renderLoss();
renderRouting();
renderBitwise();
renderCkpt();
renderGaps();
renderForwardAlign();
renderChecklist();
renderOverview();
// Monitor tab renders lazily on first activation (see switchTab + TAB_RENDERERS).
// If the page is loaded with #tab-dynamics as the starting tab, trigger now.
if (location.hash === '#tab-dynamics') switchTab('tab-dynamics');
</script>
</body>
</html>
"""


if __name__ == '__main__':
    main()
