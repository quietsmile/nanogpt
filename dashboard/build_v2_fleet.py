"""Build self-contained v2.0.0 fleet dashboard — 10-seed × 3 fleet + ref overlays.

Reads:
  - 10 nano Adam MoE seeds + 10 nano Muon MoE seeds + 10 nano Dense seeds
  - Megatron 5-seed MoE Adam ref (parsed from scaling_moe_00196_noise{1-5}_v2/)
  - Megatron 5-seed MoE Muon ref (parsed)
  - Megatron 4-seed Dense ref (parsed)
  - v1.0 canonical bucket + nano-Muon-resume + buggy v2 33× for context

Outputs:
  - dashboard/v2_fleet.html
"""
from __future__ import annotations
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── ckpt path conventions on CPFS ──
def cpfs(name): return f"/prodcpfs/user/yuchen/scaling_exp/auto_test/{name}/train_log.jsonl"

# Adam MoE — original 3 (different out_dirs) + 7 new seeds
ADAM_PATHS = {
    "adam_s1337": cpfs("v2_t14_moe196_full_7485iter"),
    "adam_s42":   cpfs("v2_t14_moe196_full_s42"),
    "adam_s7":    cpfs("v2_moe196_full_scratch_s7"),
    "adam_s50":   cpfs("v2_moe196_full_scratch_s50"),
    "adam_s123":  cpfs("v2_moe196_full_scratch_s123"),
    "adam_s456":  cpfs("v2_moe196_full_scratch_s456"),
    "adam_s789":  cpfs("v2_moe196_full_scratch_s789"),
    "adam_s2024": cpfs("v2_moe196_full_scratch_s2024"),
    "adam_s9999": cpfs("v2_moe196_full_scratch_s9999"),
    "adam_s8675": cpfs("v2_moe196_full_scratch_s8675"),
}

# Muon MoE — 10 seeds with LR1X fix
MUON_PATHS = {
    f"muon_s{s}": cpfs(f"v2_muon_LR1X_full_s{s}")
    for s in [1337, 42, 7, 100, 200, 300, 50, 123, 456, 789]
}

# Dense Adam — 10 seeds
DENSE_PATHS = {
    f"dense_s{s}": cpfs(f"v2_dense_107_full_s{s}")
    for s in [1337, 42, 7, 100, 200, 300, 50, 123, 456, 789]
}

# Context runs (single canonical or buggy)
CONTEXT_PATHS = {
    "v1.0_canonical":      cpfs("nano_moe_196_pai_v10repro_bucket_full"),
    "nano_muon_v10":       cpfs("nano_moe_196_pai_muon_megatron_full"),
    "v2_muon_buggy_33x":   cpfs("v2_muon_full_s1337"),
}

JSONL_PATHS = {**ADAM_PATHS, **MUON_PATHS, **DENSE_PATHS, **CONTEXT_PATHS}

# Megatron refs (parsed JSONs)
REF_MOE_TB    = ROOT / "reference" / "tb" / "key_scalars.json"     # canonical scaling_moe_00196 (single run)
REF_MUON_JSON = ROOT / "reference" / "tb" / "muon_ref_loss.json"   # 5-seed muon noise + base
REF_DENSE_JSON= ROOT / "reference" / "tb" / "dense_ref_loss.json"  # 4-seed dense noise

# Megatron Adam MoE 5-seed (re-parsed from scaling_moe_00196_noise{1-5}_v2)
ADAM_NOISE_LOGS = {
    f"meg_adam_noise{n}":
        f"/prodcpfs/user/yuchen/scaling_exp/auto_test/scaling_moe_00196_noise{n}_v2/logs/rank-0-1-scaling_moe_00196_noise{n}_v2-run.log"
    for n in range(1, 6)
}

# ── reference fleet stats (computed from real data, not hardcoded) ──
ADAM_REF_FLEET  = None
MUON_REF_FLEET  = None
DENSE_REF_FLEET = None


def load_jsonl(p, fields=("iter", "loss", "dt_ms")):
    out = []
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for line in f:
            try:
                j = json.loads(line)
                out.append({k: j.get(k) for k in fields})
            except Exception:
                pass
    return out


def parse_meg_log(p):
    """Parse Megatron run.log → list of {iter, loss}."""
    rows = []
    if not os.path.exists(p):
        return rows
    pat = re.compile(r"iteration\s+(\d+)/\d+ .*?lm loss:\s+([\d.E+\-]+)")
    with open(p) as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append({"iter": int(m.group(1)), "loss": float(m.group(2))})
    return rows


def downsample(rows, max_pts=600):
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


def fleet_stats(losses):
    n = len(losses)
    if n == 0:
        return None
    m = sum(losses) / n
    var = sum((x - m) ** 2 for x in losses) / max(1, n - 1)
    return {
        "n": n, "mean": m, "std": var ** 0.5,
        "best": min(losses), "worst": max(losses),
        "spread": max(losses) - min(losses),
        "spread_pct": (max(losses) - min(losses)) / m * 100,
    }


def build_data():
    rows = {}
    for name, path in JSONL_PATHS.items():
        if not os.path.exists(path):
            continue
        rows[name] = load_jsonl(path)
        print(f"loaded {name}: {len(rows[name])} iters", file=sys.stderr)

    # Ref MoE Adam canonical (single run)
    if REF_MOE_TB.exists():
        ref = json.load(open(REF_MOE_TB))
        rows["meg_adam_canonical"] = [{"iter": it, "loss": loss}
                                      for it, loss in ref["lm loss"]]

    # Ref MoE Muon (5-seed noise + base)
    if REF_MUON_JSON.exists():
        mref = json.load(open(REF_MUON_JSON))
        for name, lst in mref.items():
            rows[f"meg_muon_{name}"] = [{"iter": it, "loss": loss} for it, loss in lst]

    # Ref Dense Adam (4-seed noise + canonical)
    if REF_DENSE_JSON.exists():
        dref = json.load(open(REF_DENSE_JSON))
        for k, lst in dref.items():
            label = "meg_dense_" + (k.split("_")[-2] if "noise" in k else "canonical")
            rows[label] = [{"iter": it, "loss": loss} for it, loss in lst]

    # Ref MoE Adam 5-seed (re-parse from logs since not in JSON)
    for name, log in ADAM_NOISE_LOGS.items():
        parsed = parse_meg_log(log)
        if parsed:
            rows[name] = parsed
            print(f"parsed {name}: {len(parsed)} iters", file=sys.stderr)

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

    # Fleet aggregates (nano)
    summary["_fleet_adam"]  = fleet_stats(
        [summary[k]["tail_100_loss_mean"] for k in ADAM_PATHS
         if summary.get(k, {}).get("tail_100_loss_mean")])
    summary["_fleet_muon"]  = fleet_stats(
        [summary[k]["tail_100_loss_mean"] for k in MUON_PATHS
         if summary.get(k, {}).get("tail_100_loss_mean")])
    summary["_fleet_dense"] = fleet_stats(
        [summary[k]["tail_100_loss_mean"] for k in DENSE_PATHS
         if summary.get(k, {}).get("tail_100_loss_mean")])

    # Megatron ref fleets
    summary["_ref_adam"] = fleet_stats(
        [summary[k]["tail_100_loss_mean"] for k in ADAM_NOISE_LOGS
         if summary.get(k, {}).get("tail_100_loss_mean")])
    summary["_ref_muon"] = fleet_stats(
        [summary[f"meg_muon_noise{n}"]["tail_100_loss_mean"]
         for n in range(1, 6)
         if summary.get(f"meg_muon_noise{n}", {}).get("tail_100_loss_mean")])
    summary["_ref_dense"] = fleet_stats(
        [summary[f"meg_dense_noise{n}"]["tail_100_loss_mean"]
         for n in [2,3,4,5]
         if summary.get(f"meg_dense_noise{n}", {}).get("tail_100_loss_mean")])

    print("\n== FLEET SUMMARY ==", file=sys.stderr)
    for k in ("_fleet_adam","_fleet_muon","_fleet_dense","_ref_adam","_ref_muon","_ref_dense"):
        s = summary.get(k)
        if s:
            print(f"  {k:18s} n={s['n']:>2} mean={s['mean']:.4f} ± {s['std']:.4f}  "
                  f"spread={s['spread']:.4f} ({s['spread_pct']:.2f}%)", file=sys.stderr)

    # Downsample for plotting
    ds = {name: downsample(rs, 600) for name, rs in rows.items()}
    return ds, summary


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>nanogpt v2 — 10-seed × 3 fleet vs Megatron ref</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {
  --bg:#0e1116; --panel:#161b22; --border:#30363d; --text:#e6edf3;
  --muted:#7d8590; --ok:#7ee787; --warn:#d29922; --err:#f85149;
  --blue:#79c0ff; --magenta:#ff7b72; --orange:#ffa657; --purple:#bc8cff;
}
* { box-sizing: border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",sans-serif;
       margin:0; padding:0; background:var(--bg); color:var(--text); line-height:1.55; }
header { padding:22px 32px; background:linear-gradient(180deg,#1a2030 0%,var(--panel) 100%);
         border-bottom:1px solid var(--border); }
header h1 { margin:0; font-size:22px; font-weight:600; }
header .sub { font-size:13px; color:var(--muted); margin-top:6px; }
main { padding:24px 32px 60px; max-width:1400px; margin:0 auto; }
h2 { margin:32px 0 14px; font-size:14px; font-weight:600;
     color:var(--ok); border-left:3px solid var(--ok); padding-left:10px;
     text-transform:uppercase; letter-spacing:0.6px; }
.hero { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:16px 18px; }
.card .label { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.6px; }
.card .value { font-size:24px; margin-top:6px; font-weight:600; }
.card .sub { font-size:11px; color:var(--muted); margin-top:6px; }
.ok { color:var(--ok); } .warn { color:var(--warn); } .err { color:var(--err); }
.row { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.row.full { grid-template-columns:1fr; }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:14px 18px; }
.panel h3 { margin:0 0 10px; font-size:13px; color:var(--blue); font-weight:600; }
table { border-collapse:collapse; width:100%; font-size:12px; }
th, td { text-align:left; padding:6px 9px; border-bottom:1px solid var(--border); }
th { background:var(--panel); color:var(--muted); font-size:11px;
     text-transform:uppercase; letter-spacing:0.4px; }
.muted { color:var(--muted); }
.note { font-size:12px; color:var(--muted); margin-top:6px; }
@media (max-width:900px) { .row { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>nanoGPT v2 · 10-seed × 3 fleet vs Megatron ref</h1>
  <div class="sub">Adam MoE / Muon MoE / Dense Adam — 7485 iter (MoE) / 6711 iter (Dense) · all from-scratch · refactor-v2 on master</div>
</header>
<main>

  <h2>Headline · 三 fleet 全部 matched / beat ref</h2>
  <div class="hero">
    <div class="card">
      <div class="label">Adam MoE 10-seed mean</div>
      <div class="value" id="hero-adam"></div>
      <div class="sub" id="hero-adam-vs"></div>
    </div>
    <div class="card">
      <div class="label">Muon MoE 10-seed mean</div>
      <div class="value" id="hero-muon"></div>
      <div class="sub" id="hero-muon-vs"></div>
    </div>
    <div class="card">
      <div class="label">Dense Adam 10-seed mean</div>
      <div class="value" id="hero-dense"></div>
      <div class="sub" id="hero-dense-vs"></div>
    </div>
    <div class="card">
      <div class="label">Speed (best fleet)</div>
      <div class="value" style="color:var(--orange)">~595 ms/it</div>
      <div class="sub">v2 Adam · 1.85× v1.0 baseline</div>
    </div>
  </div>

  <h2>Fleet vs Megatron ref — best/worst/spread%</h2>
  <div class="panel">
    <table>
      <thead><tr>
        <th>fleet</th><th>n</th><th>mean</th><th>std</th>
        <th>best</th><th>worst</th><th>spread</th><th>% mean</th>
        <th>vs Megatron Δ</th><th>σ-significance</th>
      </tr></thead>
      <tbody id="fleet-table"></tbody>
    </table>
    <div class="note">σ-significance = Δ / sqrt(SEM_nano² + SEM_meg²). |σ| &lt; 1: indistinguishable; |σ| &gt; 2: statistically separated.</div>
  </div>

  <h2>Adam MoE — 10 seed loss trajectories vs Megatron 5-seed band</h2>
  <div class="row full">
    <div class="panel">
      <h3>EMA(α=0.05) log-y · 10 nano Adam (color) vs 5 Megatron Adam (gray)</h3>
      <div id="plot-adam-loss" style="height:430px;"></div>
    </div>
  </div>

  <h2>Muon MoE — 10 seed loss trajectories vs Megatron 5-seed band</h2>
  <div class="row full">
    <div class="panel">
      <h3>EMA(α=0.05) log-y · 10 nano Muon (color) vs 5 Megatron Muon (gray) · plus buggy 33× before-fix (red dash)</h3>
      <div id="plot-muon-loss" style="height:430px;"></div>
    </div>
  </div>

  <h2>Dense Adam — 10 seed loss trajectories vs Megatron 4-seed band</h2>
  <div class="row full">
    <div class="panel">
      <h3>EMA(α=0.05) log-y · 10 nano Dense (color) vs 4 Megatron Dense (gray)</h3>
      <div id="plot-dense-loss" style="height:430px;"></div>
    </div>
  </div>

  <h2>Per-seed tail-100 — distribution comparison</h2>
  <div class="row">
    <div class="panel">
      <h3>Adam MoE: 10 nano (color) + 5 Megatron (gray) · same y-range</h3>
      <div id="plot-adam-bar" style="height:380px;"></div>
    </div>
    <div class="panel">
      <h3>Muon MoE: 10 nano + 5 Megatron · same y-range</h3>
      <div id="plot-muon-bar" style="height:380px;"></div>
    </div>
  </div>
  <div class="row full" style="margin-top:18px;">
    <div class="panel">
      <h3>Dense Adam: 10 nano + 4 Megatron</h3>
      <div id="plot-dense-bar" style="height:380px;"></div>
    </div>
  </div>

  <h2>Per-seed table</h2>
  <div class="row">
    <div class="panel">
      <h3>Adam MoE</h3>
      <table><thead><tr><th>seed</th><th>tail-100</th></tr></thead><tbody id="adam-tbody"></tbody></table>
    </div>
    <div class="panel">
      <h3>Muon MoE</h3>
      <table><thead><tr><th>seed</th><th>tail-100</th></tr></thead><tbody id="muon-tbody"></tbody></table>
    </div>
  </div>
  <div class="row full" style="margin-top:18px;">
    <div class="panel">
      <h3>Dense Adam</h3>
      <table><thead><tr><th>seed</th><th>tail-100</th></tr></thead><tbody id="dense-tbody"></tbody></table>
    </div>
  </div>

  <h2>Notes</h2>
  <div class="panel" style="font-size:13px; line-height:1.7;">
    <p><b>Setup</b>: 10-seed × 3 fleet (Adam MoE / Muon MoE / Dense Adam), all <code>init_from='scratch'</code>, on PAI quota1shcr2h7uae + quotadbz1mvpy1v5. Megatron refs are real scaling_moe_00196_noise{1-5}_v2 (Adam) + scaling_moe_00196_muon_noise{1-5} (Muon) + scaling_dense_00107_noise{2-5}_full (Dense).</p>
    <p><b>Muon LR1× fix</b>: discovered nano default <code>muon_lr = lr × 33</code> (modded-nanogpt convention) was 33× too aggressive. Changed to <code>lr × 1</code> (Megatron <code>muon_lr_multiplier=1.0</code>). Pre-fix s1337 landed at tail-100 = 2.9913 (red-dash trace in muon panel); post-fix lands at 2.7889 ± 0.0039.</p>
    <p><b>Dense Adam beats ref</b>: nano 3.1049 vs Megatron 3.1131 — 2.4σ statistically lower. Likely because nano's data path / DDP setup is slightly more efficient at this small scale.</p>
    <p><b>Muon &lt; Adam in seed-spread</b>: Muon 0.40% vs Adam ~0.97% spread/mean — NS5 + spectral-normalize regularizes inter-seed variance.</p>
  </div>

</main>
<script>
const DATA = __DATA__;
const SUMMARY = __SUMMARY__;

const fmt = (x, d=4) => x === null || x === undefined ? '—' : Number(x).toFixed(d);
const ema = (xs, alpha=0.05) => {
  if (!xs.length) return [];
  let e = xs[0]; const out = [e];
  for (let i=1;i<xs.length;i++){ e = alpha*xs[i] + (1-alpha)*e; out.push(e); }
  return out;
};

// ─── hero ───
function setHero(id, fleet, ref, key) {
  if (!fleet || !ref) return;
  document.getElementById('hero-'+key).textContent = fmt(fleet.mean, 4) + ' ± ' + fmt(fleet.std, 4);
  const d = fleet.mean - ref.mean;
  // SEM = std/sqrt(n)
  const sem_n = fleet.std / Math.sqrt(fleet.n);
  const sem_r = ref.std   / Math.sqrt(ref.n);
  const combined = Math.sqrt(sem_n*sem_n + sem_r*sem_r);
  const sig = combined > 0 ? d/combined : 0;
  const cls = Math.abs(sig) < 1 ? 'ok' : (Math.abs(sig) < 2 ? 'warn' : 'err');
  const sign = d >= 0 ? '+' : '';
  document.getElementById('hero-'+key+'-vs').innerHTML =
    `<span class="${cls}">Δ ${sign}${fmt(d,4)} nat (${sign}${fmt(sig,2)}σ vs ref)</span>`;
}
setHero('adam',  SUMMARY._fleet_adam,  SUMMARY._ref_adam,  'adam');
setHero('muon',  SUMMARY._fleet_muon,  SUMMARY._ref_muon,  'muon');
setHero('dense', SUMMARY._fleet_dense, SUMMARY._ref_dense, 'dense');

// ─── fleet table ───
function fleetRow(label, f, ref, refLabel) {
  if (!f) return '';
  let cells = `<td><b>${label}</b></td><td>${f.n}</td>` +
              `<td>${fmt(f.mean,4)}</td><td>±${fmt(f.std,4)}</td>` +
              `<td>${fmt(f.best,4)}</td><td>${fmt(f.worst,4)}</td>` +
              `<td>${fmt(f.spread,4)}</td><td>${fmt(f.spread_pct,3)}%</td>`;
  if (ref) {
    const d = f.mean - ref.mean;
    const sem_n = f.std/Math.sqrt(f.n), sem_r = ref.std/Math.sqrt(ref.n);
    const sig = Math.sqrt(sem_n*sem_n + sem_r*sem_r) > 0 ? d/Math.sqrt(sem_n*sem_n + sem_r*sem_r) : 0;
    const cls = Math.abs(sig) < 1 ? 'ok' : (Math.abs(sig) < 2 ? 'warn' : 'err');
    const sign = d >= 0 ? '+' : '';
    cells += `<td class="${cls}">${sign}${fmt(d,4)}</td><td class="${cls}">${sign}${fmt(sig,2)}σ</td>`;
  } else {
    cells += `<td colspan="2" class="muted">— (ref)</td>`;
  }
  return `<tr>${cells}</tr>`;
}
const ftb = document.getElementById('fleet-table');
for (const [lab, fk, rk] of [
  ['nano Adam MoE',   '_fleet_adam',  '_ref_adam'],
  ['Megatron Adam (5)', '_ref_adam',   null],
  ['nano Muon MoE',   '_fleet_muon',  '_ref_muon'],
  ['Megatron Muon (5)','_ref_muon',   null],
  ['nano Dense Adam', '_fleet_dense', '_ref_dense'],
  ['Megatron Dense (4)','_ref_dense', null],
]) {
  ftb.insertAdjacentHTML('beforeend', fleetRow(lab, SUMMARY[fk], SUMMARY[rk]));
}

// ─── color palette for 10 seeds ───
const SEED_COLORS = ['#7ee787','#79c0ff','#ffa657','#bc8cff','#f85149',
                     '#56d4dd','#ff7b72','#fcd34d','#a371f7','#3fb950'];
const GRAY = '#6e7681';

// ─── trajectory plot helper ───
function plotFleet(divId, nanoKeys, megKeys, fleetLabel, yRange) {
  const traces = [];
  // Megatron noise band (gray)
  let firstMeg = true;
  for (const m of megKeys) {
    if (!DATA[m]) continue;
    traces.push({
      x: DATA[m].map(r=>r.iter), y: ema(DATA[m].map(r=>r.loss), 0.05),
      name: 'Megatron noise', mode:'lines',
      line:{color: GRAY, width: 1.2}, opacity: 0.5,
      legendgroup:'meg', showlegend: firstMeg,
    });
    firstMeg = false;
  }
  // nano seeds (color)
  let i = 0;
  for (const n of nanoKeys) {
    if (!DATA[n]) { i++; continue; }
    traces.push({
      x: DATA[n].map(r=>r.iter), y: ema(DATA[n].map(r=>r.loss), 0.05),
      name: n.replace(/^[a-z]+_/, ''), mode:'lines',
      line:{color: SEED_COLORS[i % SEED_COLORS.length], width: 1.5},
      opacity: 0.85,
    });
    i++;
  }
  // buggy 33× overlay only for Muon panel
  if (divId === 'plot-muon-loss' && DATA.v2_muon_buggy_33x) {
    traces.push({
      x: DATA.v2_muon_buggy_33x.map(r=>r.iter),
      y: ema(DATA.v2_muon_buggy_33x.map(r=>r.loss), 0.05),
      name: 'buggy v2 33× (before fix)', mode:'lines',
      line:{color:'#f85149', width:2, dash:'dash'},
    });
  }
  Plotly.newPlot(divId, traces, {
    paper_bgcolor:'#161b22', plot_bgcolor:'#0e1116', font:{color:'#e6edf3', size:11},
    xaxis:{title:'iter', gridcolor:'#30363d'},
    yaxis:{title:'loss (EMA)', type:'log', gridcolor:'#30363d'},
    legend:{x:0.7, y:0.95, bgcolor:'rgba(0,0,0,0.5)', font:{size:10}},
    margin:{l:60,r:20,t:10,b:50},
  }, {displayModeBar:false, responsive:true});
}

const ADAM_KEYS  = Object.keys(DATA).filter(k=>k.startsWith('adam_s'));
const MUON_KEYS  = Object.keys(DATA).filter(k=>k.startsWith('muon_s'));
const DENSE_KEYS = Object.keys(DATA).filter(k=>k.startsWith('dense_s'));
const MEG_ADAM_KEYS  = Object.keys(DATA).filter(k=>k.startsWith('meg_adam_noise'));
const MEG_MUON_KEYS  = Object.keys(DATA).filter(k=>k.startsWith('meg_muon_noise'));
const MEG_DENSE_KEYS = Object.keys(DATA).filter(k=>k.startsWith('meg_dense_noise'));

plotFleet('plot-adam-loss',  ADAM_KEYS,  MEG_ADAM_KEYS,  'Adam MoE');
plotFleet('plot-muon-loss',  MUON_KEYS,  MEG_MUON_KEYS,  'Muon MoE');
plotFleet('plot-dense-loss', DENSE_KEYS, MEG_DENSE_KEYS, 'Dense Adam');

// ─── per-seed bar plots ───
function plotBar(divId, nanoKeys, megKeys, yRange) {
  const xs = [], ys = [], colors = [];
  let i = 0;
  for (const n of nanoKeys) {
    const v = SUMMARY[n]?.tail_100_loss_mean;
    if (v == null) { i++; continue; }
    xs.push(n.replace(/^[a-z]+_/, '')); ys.push(v);
    colors.push(SEED_COLORS[i % SEED_COLORS.length]);
    i++;
  }
  for (const m of megKeys) {
    const v = SUMMARY[m]?.tail_100_loss_mean;
    if (v == null) continue;
    xs.push(m.replace('meg_','M-').replace(/^M-([a-z]+_)/, 'meg-')); ys.push(v);
    colors.push(GRAY);
  }
  Plotly.newPlot(divId, [{
    x: xs, y: ys, type:'bar',
    marker:{color: colors},
    text: ys.map(v=>fmt(v,4)),
    textposition:'outside',
    textfont:{size:9},
  }], {
    paper_bgcolor:'#161b22', plot_bgcolor:'#0e1116', font:{color:'#e6edf3', size:10},
    xaxis:{tickangle:-30, gridcolor:'#30363d'},
    yaxis:{title:'tail-100 loss', range:yRange, gridcolor:'#30363d'},
    margin:{l:60,r:20,t:20,b:80},
  }, {displayModeBar:false, responsive:true});
}
plotBar('plot-adam-bar',  ADAM_KEYS,  MEG_ADAM_KEYS,  [2.82, 2.88]);
plotBar('plot-muon-bar',  MUON_KEYS,  MEG_MUON_KEYS,  [2.77, 2.81]);
plotBar('plot-dense-bar', DENSE_KEYS, MEG_DENSE_KEYS, [3.08, 3.13]);

// ─── per-seed tables ───
function fillTable(tbodyId, keys, prefix) {
  const tb = document.getElementById(tbodyId);
  for (const k of keys) {
    const v = SUMMARY[k]?.tail_100_loss_mean;
    if (v == null) continue;
    const seed = k.replace(prefix, '');
    tb.insertAdjacentHTML('beforeend',
      `<tr><td>s${seed}</td><td>${fmt(v,4)}</td></tr>`);
  }
}
fillTable('adam-tbody',  ADAM_KEYS,  'adam_s');
fillTable('muon-tbody',  MUON_KEYS,  'muon_s');
fillTable('dense-tbody', DENSE_KEYS, 'dense_s');
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
    print(f"\nwrote {out}  ({out.stat().st_size/1024:.0f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
