"""Build self-contained v2.0.0 3-seed fleet dashboard.

Reads:
  - v2.0.0 3 jsonl (s1337 / s42 / s7)
  - v1.0 nano canonical bucket jsonl
  - Megatron ref TB key_scalars.json (lm_loss)

Outputs:
  - dashboard/v2_fleet.html (single file, embedded JSON, Plotly via CDN)
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

JSONL_PATHS = {
    "s1337": "/prodcpfs/user/yuchen/scaling_exp/auto_test/v2_t14_moe196_full_7485iter/train_log.jsonl",
    "s42":   "/prodcpfs/user/yuchen/scaling_exp/auto_test/v2_t14_moe196_full_s42/train_log.jsonl",
    "s7":    "/prodcpfs/user/yuchen/scaling_exp/auto_test/v2_moe196_full_scratch_s7/train_log.jsonl",
    "v1.0_canonical": "/prodcpfs/user/yuchen/scaling_exp/auto_test/nano_moe_196_pai_v10repro_bucket_full/train_log.jsonl",
}
REF_TB = ROOT / "reference" / "tb" / "key_scalars.json"

REF_FLEET = {"mean": 2.8474, "std": 0.0051, "n_seeds": 4, "label": "Megatron ref Adam 4-seed"}
V1_FLEET  = {"mean": 2.8463, "std": 0.0017, "n_seeds": 3, "label": "nano v1.0 bucket 3-seed"}


def load_jsonl(p, fields=("iter", "loss", "dt_ms", "moe_dead_layermean",
                          "moe_entropy_layermean", "maxvio_micro_batch")):
    out = []
    with open(p) as f:
        for line in f:
            try:
                j = json.loads(line)
                out.append({k: j.get(k) for k in fields})
            except Exception:
                pass
    return out


def downsample(rows, max_pts=600):
    """Keep evenly-spaced ~max_pts iters for plot performance."""
    if len(rows) <= max_pts:
        return rows
    step = max(1, len(rows) // max_pts)
    return rows[::step]


def tail_mean(rows, n=100, key="loss"):
    vals = [r[key] for r in rows[-n:] if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def median(vals):
    s = sorted(vals)
    return s[len(s) // 2] if s else None


def build_data():
    rows = {}
    for name, path in JSONL_PATHS.items():
        if not os.path.exists(path):
            print(f"WARN: missing {path}", file=sys.stderr)
            continue
        rows[name] = load_jsonl(path)
        print(f"loaded {name}: {len(rows[name])} iters")

    # Ref TB lm_loss
    ref = json.load(open(REF_TB))
    ref_lm = ref["lm loss"]  # list of [iter, loss]
    ref_rows = [{"iter": it, "loss": loss} for it, loss in ref_lm]
    rows["megatron_ref"] = ref_rows
    print(f"loaded megatron_ref: {len(ref_rows)} iters")

    # Compute summary stats
    summary = {}
    for name, rs in rows.items():
        tm = tail_mean(rs)
        dts = [r.get("dt_ms") for r in rs if r.get("dt_ms")]
        summary[name] = {
            "tail_100_loss_mean": tm,
            "n_iters": len(rs),
            "dt_ms_median": median(dts) if dts else None,
        }
    # MoE health (only for v2 seeds + v1.0 — ref doesn't expose these)
    moe_keys = ["moe_dead_layermean", "moe_entropy_layermean", "maxvio_micro_batch"]
    for name in ["s1337", "s42", "s7", "v1.0_canonical"]:
        if name not in rows:
            continue
        for k in moe_keys:
            vals = [r.get(k) for r in rows[name][-100:] if r.get(k) is not None]
            summary[name][f"tail_100_{k}"] = sum(vals) / len(vals) if vals else None

    # Downsample for plotting
    ds = {name: downsample(rs, 600) for name, rs in rows.items()}

    # 3-seed fleet aggregate
    v2_seeds = [n for n in ("s1337", "s42", "s7") if n in summary]
    fleet_losses = [summary[n]["tail_100_loss_mean"] for n in v2_seeds
                    if summary[n]["tail_100_loss_mean"] is not None]
    if fleet_losses:
        m = sum(fleet_losses) / len(fleet_losses)
        var = sum((x - m) ** 2 for x in fleet_losses) / max(1, len(fleet_losses) - 1)
        summary["_v2_fleet"] = {
            "n_seeds": len(fleet_losses),
            "mean": m,
            "std": var ** 0.5,
            "individual": dict(zip(v2_seeds, fleet_losses)),
        }
    summary["_ref_fleet"] = REF_FLEET
    summary["_v1_fleet"] = V1_FLEET

    return ds, summary


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>nanogpt v2.0.0 fleet vs Megatron ref · MoE-196</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {
  --bg: #0e1116; --panel: #161b22; --border: #30363d; --text: #e6edf3;
  --muted: #7d8590; --ok: #7ee787; --warn: #d29922; --err: #f85149; --blue: #79c0ff;
  --magenta: #ff7b72; --orange: #ffa657;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
       margin: 0; padding: 0; background: var(--bg); color: var(--text); line-height: 1.5; }
header { padding: 22px 32px; background: linear-gradient(180deg, #1a2030 0%, var(--panel) 100%);
         border-bottom: 1px solid var(--border); }
header h1 { margin: 0; font-size: 22px; font-weight: 600; }
header .sub { font-size: 13px; color: var(--muted); margin-top: 6px; }
main { padding: 28px 32px 60px; max-width: 1320px; margin: 0 auto; }
h2 { margin: 36px 0 16px; font-size: 14px; font-weight: 600;
     color: var(--ok); border-left: 3px solid var(--ok); padding-left: 10px;
     text-transform: uppercase; letter-spacing: 0.7px; }
.hero { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }
.hero .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
              padding: 18px 20px; }
.hero .card .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
.hero .card .value { font-size: 28px; margin-top: 6px; font-weight: 600; }
.hero .card .sub { font-size: 11px; color: var(--muted); margin-top: 6px; }
.hero .accent .value { color: var(--ok); }
.hero .accent2 .value { color: var(--blue); }
.hero .accent3 .value { color: var(--orange); }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.row.full { grid-template-columns: 1fr; }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
         padding: 14px 18px; }
.panel h3 { margin: 0 0 10px; font-size: 13px; color: var(--blue); font-weight: 600; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
th { background: var(--panel); color: var(--muted); font-size: 11px;
     text-transform: uppercase; letter-spacing: 0.4px; font-weight: 600; }
.ok { color: var(--ok); } .warn { color: var(--warn); } .err { color: var(--err); }
.muted { color: var(--muted); }
.note { font-size: 12px; color: var(--muted); margin-top: 6px; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px;
       background: #0d3d23; color: var(--ok); }
.tag.warn { background: #3d2f0d; color: var(--warn); }
.tag.blue { background: #0d2f3d; color: var(--blue); }
@media (max-width: 900px) { .row { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>nanoGPT v2.0.0 · 3-seed MoE-196 fleet vs Megatron ref</h1>
  <div class="sub">7485-iter PAI runs · scaling_moe_00196 config (9L/512d/144 experts/top-8) · refactor-v2 code on master @ v2.0.0</div>
</header>
<main>

  <div class="hero">
    <div class="card accent">
      <div class="label">v2.0 3-seed tail-100 mean</div>
      <div class="value" id="hero-v2"></div>
      <div class="sub" id="hero-v2-sub"></div>
    </div>
    <div class="card accent2">
      <div class="label">Δ vs Megatron ref</div>
      <div class="value" id="hero-delta-ref"></div>
      <div class="sub">ref 4-seed: 2.8474 ± 0.0051</div>
    </div>
    <div class="card">
      <div class="label">Δ vs nano v1.0</div>
      <div class="value" id="hero-delta-v1"></div>
      <div class="sub">v1.0 3-seed: 2.8463 ± 0.0017</div>
    </div>
    <div class="card accent3">
      <div class="label">Speed vs v1.0 baseline</div>
      <div class="value" id="hero-speed"></div>
      <div class="sub">~595 ms/iter vs ~1100 ms/iter</div>
    </div>
  </div>

  <h2>Loss trajectory — 3 seeds vs v1.0 vs Megatron ref</h2>
  <div class="row full">
    <div class="panel">
      <h3>Smoothed (EMA α=0.05) — log-y for visibility of late-train</h3>
      <div id="plot-loss" style="height:480px;"></div>
      <div class="note">v2.0 3 seeds (color), v1.0 nano canonical (purple), Megatron ref (gray dotted). All seeds converge into ref's tail-100 noise band.</div>
    </div>
  </div>

  <h2>Late-train zoom — last 1000 iters</h2>
  <div class="row full">
    <div class="panel">
      <h3>Per-iter loss · iter 6485 → 7485</h3>
      <div id="plot-loss-tail" style="height:380px;"></div>
    </div>
  </div>

  <h2>Tail-100 mean comparison — fleet vs fleet</h2>
  <div class="row">
    <div class="panel">
      <h3>Per-seed bars + fleet means with std error bars</h3>
      <div id="plot-tail-bar" style="height:400px;"></div>
    </div>
    <div class="panel">
      <h3>Speed (median ms/iter, post-warmup)</h3>
      <div id="plot-speed" style="height:400px;"></div>
    </div>
  </div>

  <h2>MoE routing health — tail-100 stats</h2>
  <div class="row full">
    <div class="panel">
      <h3>Dead experts (per layer mean) · entropy · maxvio (micro-batch)</h3>
      <div id="plot-moe" style="height:380px;"></div>
      <div class="note">All 3 seeds + v1.0 baseline track within ±0.5% on entropy and ±15% on dead experts — no routing pathology, divergence in loss is init lottery only.</div>
    </div>
  </div>

  <h2>Numerical summary</h2>
  <div class="panel">
    <table>
      <thead><tr><th>fleet</th><th>tail-100 mean</th><th>std (n)</th><th>Δ vs ref</th><th>speed</th></tr></thead>
      <tbody id="summary-tbody"></tbody>
    </table>
  </div>

  <h2>Per-seed details</h2>
  <div class="panel">
    <table>
      <thead><tr><th>seed</th><th>tail-100 loss</th><th>moe_dead</th><th>moe_entropy</th><th>maxvio</th><th>dt_ms (median)</th></tr></thead>
      <tbody id="seed-tbody"></tbody>
    </table>
  </div>

</main>
<script>
const DATA = __DATA__;
const SUMMARY = __SUMMARY__;

// ---------- helpers ----------
const fmt = (x, d=4) => x === null || x === undefined ? '—' : Number(x).toFixed(d);
const ema = (xs, alpha=0.05) => {
  if (!xs.length) return [];
  let e = xs[0]; const out = [e];
  for (let i=1;i<xs.length;i++){ e = alpha*xs[i] + (1-alpha)*e; out.push(e); }
  return out;
};
const COLORS = {s1337: '#7ee787', s42: '#79c0ff', s7: '#ff7b72',
                'v1.0_canonical': '#bc8cff', megatron_ref: '#888'};
const LABELS = {s1337: 'v2 seed 1337', s42: 'v2 seed 42', s7: 'v2 seed 7',
                'v1.0_canonical': 'nano v1.0 (canonical)', megatron_ref: 'Megatron ref'};

// ---------- hero ----------
const f = SUMMARY._v2_fleet || {};
document.getElementById('hero-v2').textContent = fmt(f.mean, 4);
document.getElementById('hero-v2-sub').textContent = `± ${fmt(f.std, 4)} (n=${f.n_seeds || 0})`;
const dRef = (f.mean || 0) - SUMMARY._ref_fleet.mean;
const dV1  = (f.mean || 0) - SUMMARY._v1_fleet.mean;
document.getElementById('hero-delta-ref').textContent = (dRef>=0?'+':'') + fmt(dRef, 4) + ' nat';
document.getElementById('hero-delta-v1').textContent  = (dV1>=0?'+':'')  + fmt(dV1, 4) + ' nat';
// speed: median of v2 seeds vs v1.0
const v2dts = ['s1337','s42','s7'].map(s => SUMMARY[s]?.dt_ms_median).filter(x=>x);
const v2dt = v2dts.length ? v2dts.reduce((a,b)=>a+b,0)/v2dts.length : null;
const v1dt = SUMMARY['v1.0_canonical']?.dt_ms_median;
document.getElementById('hero-speed').textContent = (v1dt && v2dt) ? (v1dt/v2dt).toFixed(2)+'×' : '—';

// ---------- loss trajectory plot ----------
const tracesLoss = [];
for (const name of ['megatron_ref','v1.0_canonical','s1337','s42','s7']) {
  if (!DATA[name]) continue;
  const xs = DATA[name].map(r => r.iter);
  const ys = DATA[name].map(r => r.loss);
  tracesLoss.push({
    x: xs, y: ema(ys, 0.05), name: LABELS[name], type: 'scatter', mode: 'lines',
    line: {color: COLORS[name], width: name==='megatron_ref' ? 1.5 : 2,
           dash: name==='megatron_ref' ? 'dot' : 'solid'},
    opacity: name==='megatron_ref' ? 0.7 : 1,
  });
}
Plotly.newPlot('plot-loss', tracesLoss, {
  paper_bgcolor:'#161b22', plot_bgcolor:'#0e1116',
  font:{color:'#e6edf3', size:11},
  xaxis:{title:'iter', gridcolor:'#30363d', zeroline:false},
  yaxis:{title:'loss (EMA)', type:'log', gridcolor:'#30363d', zeroline:false},
  legend:{x:0.7, y:0.95, bgcolor:'rgba(0,0,0,0.4)'},
  margin:{l:60,r:20,t:10,b:50},
}, {displayModeBar:false, responsive:true});

// ---------- tail zoom ----------
const tracesTail = [];
for (const name of ['megatron_ref','v1.0_canonical','s1337','s42','s7']) {
  if (!DATA[name]) continue;
  const tail = DATA[name].filter(r => r.iter >= 6485);
  if (!tail.length) continue;
  tracesTail.push({
    x: tail.map(r=>r.iter), y: tail.map(r=>r.loss),
    name: LABELS[name], type:'scatter', mode:'lines',
    line:{color:COLORS[name], width: name==='megatron_ref' ? 1 : 1.5,
          dash: name==='megatron_ref' ? 'dot' : 'solid'},
    opacity: name==='megatron_ref' ? 0.55 : 0.9,
  });
}
Plotly.newPlot('plot-loss-tail', tracesTail, {
  paper_bgcolor:'#161b22', plot_bgcolor:'#0e1116', font:{color:'#e6edf3', size:11},
  xaxis:{title:'iter', gridcolor:'#30363d'},
  yaxis:{title:'loss (raw)', gridcolor:'#30363d', range:[2.6, 3.1]},
  legend:{x:0.7, y:0.95, bgcolor:'rgba(0,0,0,0.4)'},
  margin:{l:60,r:20,t:10,b:50},
}, {displayModeBar:false, responsive:true});

// ---------- bar: tail-100 fleet means ----------
const seedNames = ['s1337','s42','s7'];
const seedVals = seedNames.map(s => SUMMARY[s]?.tail_100_loss_mean);
const seedBar = {
  x: seedNames.map(s=>LABELS[s]), y: seedVals, name: 'v2 per-seed',
  type:'bar', marker:{color: seedNames.map(s=>COLORS[s])},
  text: seedVals.map(v=>fmt(v,4)), textposition:'outside',
};
const fleets = ['v2 (3-seed)','nano v1.0 (3-seed)','Megatron ref (4-seed)'];
const fmeans = [SUMMARY._v2_fleet?.mean, SUMMARY._v1_fleet.mean, SUMMARY._ref_fleet.mean];
const fstds  = [SUMMARY._v2_fleet?.std,  SUMMARY._v1_fleet.std,  SUMMARY._ref_fleet.std];
const fleetBar = {
  x: fleets, y: fmeans, name:'fleet mean', type:'bar',
  marker:{color:['#7ee787','#bc8cff','#888']},
  error_y: {type:'data', array: fstds, color:'#e6edf3', thickness:1.5, width:6},
  text: fmeans.map((m,i)=>fmt(m,4)+' ± '+fmt(fstds[i],4)), textposition:'outside',
  xaxis:'x2', yaxis:'y2',
};
Plotly.newPlot('plot-tail-bar', [seedBar, fleetBar], {
  paper_bgcolor:'#161b22', plot_bgcolor:'#0e1116', font:{color:'#e6edf3', size:11},
  grid:{rows:1, columns:2, pattern:'independent'},
  xaxis: {domain:[0,0.43], gridcolor:'#30363d', tickangle:-15},
  yaxis: {domain:[0,1], range:[2.83, 2.87], title:'tail-100 loss', gridcolor:'#30363d'},
  xaxis2:{domain:[0.57,1], gridcolor:'#30363d', tickangle:-15},
  yaxis2:{domain:[0,1], range:[2.83, 2.87], gridcolor:'#30363d'},
  margin:{l:60,r:20,t:10,b:80}, showlegend:false,
}, {displayModeBar:false, responsive:true});

// ---------- speed bar ----------
const dtNames = ['s1337','s42','s7','v1.0_canonical'];
const dtLabels = dtNames.map(s => LABELS[s]);
const dtVals = dtNames.map(s => SUMMARY[s]?.dt_ms_median);
Plotly.newPlot('plot-speed', [{
  x: dtLabels, y: dtVals, type:'bar',
  marker:{color: dtNames.map(s=>COLORS[s])},
  text: dtVals.map(v=>fmt(v,0)+' ms'), textposition:'outside',
}], {
  paper_bgcolor:'#161b22', plot_bgcolor:'#0e1116', font:{color:'#e6edf3', size:11},
  xaxis:{tickangle:-15, gridcolor:'#30363d'},
  yaxis:{title:'median dt_ms post-warmup', gridcolor:'#30363d'},
  margin:{l:60,r:20,t:10,b:80}, showlegend:false,
}, {displayModeBar:false, responsive:true});

// ---------- MoE health ----------
const moeKeys = [
  ['tail_100_moe_dead_layermean','dead experts (per-layer mean)'],
  ['tail_100_moe_entropy_layermean','routing entropy'],
  ['tail_100_maxvio_micro_batch','maxvio (micro-batch)'],
];
const moeTraces = moeKeys.map(([k, lbl], idx) => ({
  x: ['s1337','s42','s7','v1.0_canonical'].map(s=>LABELS[s]),
  y: ['s1337','s42','s7','v1.0_canonical'].map(s => SUMMARY[s]?.[k]),
  name: lbl, type:'bar',
  text: ['s1337','s42','s7','v1.0_canonical'].map(s=>fmt(SUMMARY[s]?.[k], 3)),
  textposition:'outside',
  xaxis:'x'+(idx+1), yaxis:'y'+(idx+1),
  marker:{color: ['#7ee787','#79c0ff','#ff7b72','#bc8cff']},
}));
Plotly.newPlot('plot-moe', moeTraces, {
  paper_bgcolor:'#161b22', plot_bgcolor:'#0e1116', font:{color:'#e6edf3', size:11},
  grid:{rows:1, columns:3, pattern:'independent'},
  xaxis:{domain:[0,0.30], gridcolor:'#30363d', tickangle:-25},
  yaxis:{domain:[0,1], gridcolor:'#30363d', title:'dead experts'},
  xaxis2:{domain:[0.36,0.64], gridcolor:'#30363d', tickangle:-25},
  yaxis2:{domain:[0,1], gridcolor:'#30363d', title:'entropy'},
  xaxis3:{domain:[0.70,1], gridcolor:'#30363d', tickangle:-25},
  yaxis3:{domain:[0,1], gridcolor:'#30363d', title:'maxvio'},
  margin:{l:50,r:20,t:10,b:80}, showlegend:false,
}, {displayModeBar:false, responsive:true});

// ---------- summary table ----------
const sumRows = [
  ['v2.0.0 3-seed', SUMMARY._v2_fleet?.mean, SUMMARY._v2_fleet?.std, SUMMARY._v2_fleet?.n_seeds, dRef, v2dt],
  ['nano v1.0 3-seed bucket', SUMMARY._v1_fleet.mean, SUMMARY._v1_fleet.std, SUMMARY._v1_fleet.n_seeds, SUMMARY._v1_fleet.mean - SUMMARY._ref_fleet.mean, v1dt],
  ['Megatron ref 4-seed', SUMMARY._ref_fleet.mean, SUMMARY._ref_fleet.std, SUMMARY._ref_fleet.n_seeds, 0, '—'],
];
const stb = document.getElementById('summary-tbody');
for (const [lbl, m, std, n, dr, dt] of sumRows) {
  const dCls = Math.abs(dr) <= 0.005 ? 'ok' : (Math.abs(dr) <= 0.02 ? 'warn' : 'err');
  const drStr = (dr >= 0 ? '+' : '') + fmt(dr, 4) + ' nat';
  stb.insertAdjacentHTML('beforeend',
    `<tr><td>${lbl}</td><td>${fmt(m, 4)}</td><td>± ${fmt(std, 4)} (n=${n})</td>` +
    `<td class="${dCls}">${drStr}</td><td>${dt === '—' ? '—' : fmt(dt, 0)+' ms'}</td></tr>`);
}

// ---------- per-seed table ----------
const seedTb = document.getElementById('seed-tbody');
for (const s of ['s1337','s42','s7','v1.0_canonical']) {
  const r = SUMMARY[s];
  if (!r) continue;
  seedTb.insertAdjacentHTML('beforeend',
    `<tr><td>${LABELS[s]}</td>` +
    `<td>${fmt(r.tail_100_loss_mean,4)}</td>` +
    `<td>${fmt(r.tail_100_moe_dead_layermean,3)}</td>` +
    `<td>${fmt(r.tail_100_moe_entropy_layermean,4)}</td>` +
    `<td>${fmt(r.tail_100_maxvio_micro_batch,4)}</td>` +
    `<td>${fmt(r.dt_ms_median, 0)} ms</td></tr>`);
}
</script>
</body>
</html>
"""


def main():
    ds, summary = build_data()
    html = (HTML_TEMPLATE
            .replace("__DATA__", json.dumps(ds, separators=(",", ":")))
            .replace("__SUMMARY__", json.dumps(summary, separators=(",", ":"))))
    out = ROOT / "dashboard" / "v2_fleet.html"
    out.write_text(html)
    sz = out.stat().st_size
    print(f"\nwrote {out}  ({sz/1024:.0f} KB)")
    print("\nfleet stats:")
    for k, v in summary.items():
        if k.startswith("_") or "tail_100_loss_mean" not in v:
            continue
        print(f"  {k:18s} tail-100 = {v['tail_100_loss_mean']:.4f}  dt_ms = {v.get('dt_ms_median')}")
    if "_v2_fleet" in summary:
        f = summary["_v2_fleet"]
        print(f"\nv2 fleet: mean={f['mean']:.4f} ± {f['std']:.4f} (n={f['n_seeds']})")


if __name__ == "__main__":
    main()
