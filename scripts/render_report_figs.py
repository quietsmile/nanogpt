"""Render PNG charts for the RedDoc report from reports/dense_ablation.json.

Output to dashboard/figs/ (served by http.server on :8882).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(ROOT, 'dashboard', 'figs')
os.makedirs(OUT_DIR, exist_ok=True)

d = json.load(open(os.path.join(ROOT, 'reports/dense_ablation.json')))
runs = {r['id']: r for r in d['runs']}


def ema(xs, alpha=0.02):
    e = xs[0]; out = [e]
    for x in xs[1:]:
        e = alpha * x + (1 - alpha) * e
        out.append(e)
    return out


# --- Figure 1: 5 nano + 5 pai noise loss curves, with EMA-smoothed overlays. ---
fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for side, ax_i, prefix, color_base, label_prefix in [
    ('nano', ax[0], 'nano_dense_107_noise', '#d9534f', 'nano'),
    ('pai-ref', ax[1], 'ref_dense_107_noise', '#5bc0de', 'pai-ref'),
]:
    for s in range(1, 6):
        rid = f'{prefix}{s}'
        r = runs.get(rid)
        if not r or not r['n_points']: continue
        xs = [p[0] + r['iter_offset'] for p in r['points']]
        ys = [p[1] for p in r['points']]
        ax_i.plot(xs, ys, color=color_base, alpha=0.18, linewidth=0.9)
        ax_i.plot(xs, ema(ys, 0.02), color=color_base, alpha=0.95, linewidth=1.8,
                  label=f'seed {s*1000}' if s == 1 else None)
    ax_i.set_title(f'{side} fleet (5 seeds, 1000 iter) — raw + EMA α=0.02')
    ax_i.set_xlabel('step')
    ax_i.grid(alpha=0.3)
ax[0].set_ylabel('lm loss')
ax[0].set_xlim(0, 1000); ax[1].set_xlim(0, 1000)
ax[0].set_ylim(3.5, 6.5); ax[1].set_ylim(3.5, 6.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_fleet_curves.png'), dpi=110, bbox_inches='tight',
            facecolor='white')
plt.close()
print('wrote fig1_fleet_curves.png')

# --- Figure 2: Fleet-avg Δ over steps: mean(nano_fleet[i]) − mean(pai_fleet[i]). ---
fig, ax = plt.subplots(figsize=(10, 5))
nano_by = [{p[0] + r['iter_offset']: p[1] for p in r['points']}
           for r in [runs[f'nano_dense_107_noise{s}'] for s in range(1, 6)]]
pai_by  = [{p[0] + r['iter_offset']: p[1] for p in r['points']}
           for r in [runs[f'ref_dense_107_noise{s}'] for s in range(1, 6)]]
common = sorted(set.intersection(*(set(x) for x in nano_by + pai_by)))
nano_avg = [np.mean([m[i] for m in nano_by]) for i in common]
pai_avg  = [np.mean([m[i] for m in pai_by]) for i in common]
nano_std = [np.std([m[i] for m in nano_by], ddof=1) for i in common]
pai_std  = [np.std([m[i] for m in pai_by],  ddof=1) for i in common]
delta = np.array(nano_avg) - np.array(pai_avg)
delta_ema = ema(list(delta), 0.02)

ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.fill_between(common, -np.array(nano_std), np.array(nano_std),
                color='#d9534f', alpha=0.12, label='nano fleet ±1σ (cross-seed)')
ax.fill_between(common, -np.array(pai_std), np.array(pai_std),
                color='#5bc0de', alpha=0.12, label='pai-ref fleet ±1σ (cross-seed)')
ax.plot(common, delta, color='#888', alpha=0.35, linewidth=0.8, label='raw Δ')
ax.plot(common, delta_ema, color='#222', linewidth=2.0, label='EMA α=0.02')
ax.set_xlabel('step')
ax.set_ylabel('Δ loss = nano_fleet_mean − pai_fleet_mean')
ax.set_title('Fleet-mean Δ vs step (n=5 each side)')
ax.legend(loc='best')
ax.grid(alpha=0.3)
ax.set_xlim(0, 1000)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig2_fleet_delta.png'), dpi=110, bbox_inches='tight',
            facecolor='white')
plt.close()
print('wrote fig2_fleet_delta.png')

# --- Figure 3: Baseline vs Fast (same seed=1000) loss curves. ---
fig, ax = plt.subplots(figsize=(10, 5))
for rid, color, label in [
    ('nano_dense_107_noise1',      '#d9534f', 'baseline (780 ms/iter)'),
    ('nano_dense_107_noise1_fast', '#d97706', 'fast (432 ms/iter, 1.81×)'),
]:
    r = runs.get(rid)
    if not r or not r['n_points']: continue
    xs = [p[0] + r['iter_offset'] for p in r['points']]
    ys = [p[1] for p in r['points']]
    ax.plot(xs, ys, color=color, alpha=0.22, linewidth=0.8)
    ax.plot(xs, ema(ys, 0.02), color=color, linewidth=2, label=label)
ax.set_xlabel('step'); ax.set_ylabel('lm loss')
ax.set_title('Speed optimization — same seed=1000, 1000 iter\n'
             '3 config flips: deterministic=False, chunked_ce=False, attention_impl="te"')
ax.legend(loc='best')
ax.grid(alpha=0.3)
ax.set_xlim(0, 1000); ax.set_ylim(3.5, 6.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_speed_compare.png'), dpi=110, bbox_inches='tight',
            facecolor='white')
plt.close()
print('wrote fig3_speed_compare.png')

# --- Figure 4: fleet stats bar chart. ---
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = ['nano\nfleet EMA', 'pai-ref\nfleet EMA', 'nano\ntail-100', 'pai-ref\ntail-100']
nano_ema = [runs[f'nano_dense_107_noise{s}']['ema_last'] for s in range(1, 6)]
pai_ema  = [runs[f'ref_dense_107_noise{s}']['ema_last']  for s in range(1, 6)]
nano_tm  = [runs[f'nano_dense_107_noise{s}']['tail100_mean'] for s in range(1, 6)]
pai_tm   = [runs[f'ref_dense_107_noise{s}']['tail100_mean']  for s in range(1, 6)]
means = [np.mean(nano_ema), np.mean(pai_ema), np.mean(nano_tm), np.mean(pai_tm)]
stds  = [np.std(nano_ema, ddof=1), np.std(pai_ema, ddof=1),
         np.std(nano_tm, ddof=1),  np.std(pai_tm, ddof=1)]
colors = ['#d9534f', '#5bc0de', '#d9534f', '#5bc0de']
ax.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.75,
       edgecolor='black', linewidth=0.7)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + 0.003, f'{m:.4f}\n±{s:.4f}', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('loss value')
ax.set_title('Fleet-mean loss ± cross-seed std (n=5 each)')
ax.set_ylim(4.02, 4.09)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_fleet_bars.png'), dpi=110, bbox_inches='tight',
            facecolor='white')
plt.close()
print('wrote fig4_fleet_bars.png')

print(f'\nAll figs in {OUT_DIR}')
print('URL base: http://47.84.144.221:8882/dashboard/figs/')
