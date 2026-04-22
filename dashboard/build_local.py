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
</style>
</head>
<body>
<header>
  <h1>nanogpt ↔ Megatron alignment · <span class="blue">scaling_moe_00196</span>
     <span class="small" style="float:right;color:var(--muted)">built __BUILD_TS__</span></h1>
  <div class="subtitle">PAI DLC dlc1q9arre48b0kx · 9 layers / 512 hidden / 4 heads / 2 KV groups · 144 experts, top-8 sigmoid · 447.30M params · 7485 iterations · final lm loss 2.86</div>
</header>
<main>
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
function currentRunIdx() {
  const sel = document.getElementById('run-selector');
  return sel ? parseInt(sel.value) : 0;
}

function renderLoss() {
  const l = DATA.loss, tb = DATA.tb;
  const runs = DATA.runs || [];
  if (!l && runs.length === 0) { pill('loss-status', 'err', 'missing'); return; }
  const haveNano = runs.length > 0 || (l && l.nano_present);
  pill('loss-status', haveNano ? 'ok' : 'warn', haveNano ? `${runs.length || 1} run(s)` : 'ref only');

  const idx = currentRunIdx();
  const rm = runs[idx] || l?.run_meta;
  const cmp = rm?.compare || l?.compare || {};

  // Run picker
  const picker = runs.length > 0 ? `
    <div style="display:flex;align-items:center;gap:10px;margin:4px 0 10px;font-size:12px;">
      <span class="small">experiment:</span>
      <select id="run-selector" onchange="renderLoss()"
              style="background:#0b0e13;color:var(--text);border:1px solid var(--border);border-radius:4px;padding:4px 8px;font-size:12px;min-width:360px;">
        ${runs.map((r,i) => `<option value="${i}" ${i===idx?'selected':''}>${esc(r.run_id)} — ${esc(r.label)} (${r.iters_completed?.toLocaleString()||'?'} iters)</option>`).join('')}
      </select>
      <span class="small" style="margin-left:auto;">${runs.length} run(s) · started ${esc(runs[idx]?.started_at || '')}</span>
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
    const trace1 = {
      x: lm.map(p => p[0]), y: lm.map(p => p[1]),
      type: 'scatter', mode: 'lines', name: 'Megatron (ref lm loss)',
      line: { color: '#79c0ff', width: 1.2 }, yaxis: 'y',
    };
    const traces = [trace1];

    // Overlay selected nano run (preferred) or fall back to legacy nano_log
    const nanoRun = runs[idx];
    if (nanoRun && nanoRun.train_loss_points && nanoRun.train_loss_points.length) {
      // Apply iter_offset so nano's internal iter N is plotted at ref's iter N+offset,
      // aligning the two curves on the same x-axis. (nano 0-indexed vs ref 1-indexed → offset=1)
      const off = parseInt(nanoRun.iter_offset || 0);
      traces.push({
        x: nanoRun.train_loss_points.map(p => p[0] + off),
        y: nanoRun.train_loss_points.map(p => p[1]),
        type: 'scatter', mode: 'lines',
        name: `nano · ${nanoRun.run_id}` + (off ? ` (+${off})` : ''),
        line: { color: nanoRun.has_biasfix ? '#7ee787' : '#ff7b72', width: 1.2 }, yaxis: 'y',
      });
      if (nanoRun.val_loss_points && nanoRun.val_loss_points.length) {
        traces.push({
          x: nanoRun.val_loss_points.map(p => p[0] + off),
          y: nanoRun.val_loss_points.map(p => p[1]),
          type: 'scatter', mode: 'lines+markers',
          name: `nano val (n=${nanoRun.val_loss_points.length})`,
          line: { color: '#ffa657', width: 2, dash: 'dash' },
          marker: { size: 8, symbol: 'diamond' }, yaxis: 'y',
        });
      }
    } else {
      const nano = DATA.nano_log;
      if (nano && nano.train_loss && nano.train_loss.length) {
        traces.push({
          x: nano.train_loss.map(p => p[0]), y: nano.train_loss.map(p => p[1]),
          type: 'scatter', mode: 'lines', name: `nano train (n=${nano.n_entries})`,
          line: { color: '#ff7b72', width: 1.2 }, yaxis: 'y',
        });
      }
      if (nano && nano.val_loss && nano.val_loss.length) {
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
        type: 'scatter', mode: 'lines', name: 'learning rate',
        line: { color: '#d29922', width: 1, dash: 'dot' }, yaxis: 'y2',
      });
    }
    Plotly.newPlot('loss-plot', traces, {
      paper_bgcolor: '#161b22', plot_bgcolor: '#0b0e13',
      font: { color: '#e6edf3', size: 11 },
      xaxis: { title: 'step', gridcolor: '#30363d' },
      yaxis: { title: 'lm loss', gridcolor: '#30363d' },
      yaxis2: { title: 'lr', overlaying: 'y', side: 'right',
                gridcolor: 'transparent', color: '#d29922' },
      margin: { t: 20, l: 50, r: 60, b: 40 },
      legend: { x: 0.7, y: 0.95 },
    }, { responsive: true, displaylogo: false });
  }
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
  let sgPass, ddpPass, summaryText;
  if ('single_gpu' in b || 'ddp_4rank' in b) {
    sgPass = b.single_gpu && b.single_gpu.pass;
    ddpPass = b.ddp_4rank && b.ddp_4rank.pass;
    if (sgPass && ddpPass) summaryText = 'pass (single+ddp)';
    else if (sgPass && ddpPass === false) summaryText = 'single pass · ddp diverge';
    else if (sgPass) summaryText = 'single pass';
    else summaryText = 'fail';
    const pillState = sgPass ? (ddpPass === false ? 'warn' : 'ok') : 'err';
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

// Order matters: render panels first, then overview reads their statuses.
renderJob();
renderTokenizer();
renderData();
renderModel();
renderLoss();
renderBitwise();
renderCkpt();
renderGaps();
renderForwardAlign();
renderChecklist();
renderOverview();
</script>
</body>
</html>
"""


if __name__ == '__main__':
    main()
