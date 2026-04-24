"""Headless-browser screenshots of the Dense Ablation panel for the RedDoc report.

Uses playwright to render http://localhost:8882/dashboard/alignment_report.html,
wait for dashboard JS to populate charts, then crop screenshots of:
  - dense-plot (loss curves + Δ)
  - dense-gradnorm-plot (grad_norm curves + Δ)
And a screenshot of the picker + cards block.

Output → dashboard/figs/ (same dir as matplotlib figures, served on :8882).
"""
import os
import time
from playwright.sync_api import sync_playwright

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = os.path.join(ROOT, 'dashboard', 'figs')
os.makedirs(OUT_DIR, exist_ok=True)

URL = 'http://localhost:8882/dashboard/alignment_report.html'

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(viewport={'width': 1600, 'height': 1200},
                              device_scale_factor=2)  # retina sharpness
    page = ctx.new_page()
    page.goto(URL, wait_until='networkidle', timeout=60000)
    # scroll to dense section first so Plotly lazily renders (it only sizes once visible)
    page.evaluate("document.getElementById('dense-plot').scrollIntoView({block:'start'})")
    time.sleep(2)
    page.wait_for_selector('#dense-plot .plot-container', state='attached', timeout=30000)
    # nudge render by dispatching a resize event — Plotly subscribes to these
    page.evaluate("window.dispatchEvent(new Event('resize'))")
    time.sleep(3)

    def set_selection(ids):
        # Manipulate _denseSelected state directly, then re-render once.
        js = (
            "const want = new Set(" + repr(list(ids)) + ");"
            "_denseSelected.clear();"
            "for (const id of want) _denseSelected.add(id);"
            "renderDenseAblation(_denseData);"
        )
        page.evaluate(js)
        time.sleep(2)
        # smoothing: also pin to MA 100 for cleaner report images
        page.evaluate("_denseSmoothWindow = 100; renderDenseAblation(_denseData);")
        time.sleep(2)

    # Two-way nano vs ref (one seed each, MA-100 smooth) — clean picture.
    set_selection(['nano_dense_107_noise1', 'ref_dense_107_noise1'])

    # --- Screenshot dense-plot (loss + Δ) ---
    plot = page.query_selector('#dense-plot')
    plot.screenshot(path=os.path.join(OUT_DIR, 'dash_dense_loss.png'))
    print('wrote dash_dense_loss.png')

    # --- Screenshot dense-gradnorm-plot ---
    gn = page.query_selector('#dense-gradnorm-plot')
    gn.screenshot(path=os.path.join(OUT_DIR, 'dash_dense_gradnorm.png'))
    print('wrote dash_dense_gradnorm.png')

    # --- Screenshot dense-body (picker + stat cards) showing the full 5+5 fleet ---
    desired_fleet = [f'nano_dense_107_noise{i}' for i in range(1, 6)] + \
                    [f'ref_dense_107_noise{i}'  for i in range(1, 6)]
    set_selection(desired_fleet)
    body = page.query_selector('#dense-body')
    body.screenshot(path=os.path.join(OUT_DIR, 'dash_dense_cards.png'))
    print('wrote dash_dense_cards.png')

    # --- For speed compare: switch to baseline + fast same-seed ---
    set_selection(['nano_dense_107_noise1', 'nano_dense_107_noise1_fast'])
    plot2 = page.query_selector('#dense-plot')
    plot2.screenshot(path=os.path.join(OUT_DIR, 'dash_speed_compare.png'))
    print('wrote dash_speed_compare.png')

    browser.close()

print(f'\nAll screenshots in {OUT_DIR}')
