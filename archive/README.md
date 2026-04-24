# archive/

Historical artifacts from the v10 alignment hunt (2026-04-20 ~ 2026-04-24).
Kept for reference but **not part of the main code path**.

## scripts_v10hunt/ (71 files)
One-off diagnostic scripts written during the nano↔ref alignment investigation.
Superseded by structured regression tests in `tests/regression/`.

Examples:
- `diag_activations.py` — per-sublayer act_std/mean/abs_max dump against ref master.log
- `diag_per_layer_stats.py` — hook every block's ln/attn/mlp input/output
- `diag_grad_diff.py` — compare nano per-param grad to ref's exp_avg-derived grad
- `trajectory_test.py` / `short_window_test.py` — iter-by-iter loss trajectory compare
- `load_megatron_forward_test.py` — load Megatron weights, 1-sample forward

## config_v10hunt/ (34 files)
Intermediate experiment configs, superseded by canonical tier configs:
- `cybertron_moe_196_{bare,resume,resume_test,eodmask,from0,from10,moediag,iter0_diag}.py`
- `cybertron_moe_196_noise{1-5}*.py` / `speedtest*.py` / `v10repro_*.py`
- `cybertron_dense_107_noise{1-5}*.py` / `seed42.py` / `from_ref.py`
- upstream nanoGPT examples: `eval_gpt2*.py`, `train_gpt2.py`, `train_shakespeare_char.py`

## reports_v10hunt/
Old experiment ckpts (meg_optim_iter*, megatron_iter*_ckpt), diag JSON dumps,
short_window/ window.

## Why archive not delete
- Sister agents may reference specific diag snippets when debugging similar issues
- Historical ckpts (meg_optim_iter*.pt) are reproducible but take ~hours to
  regenerate (recover_megatron_optim.py on 4-shard distrib_optim.pt)
- v10-era configs serve as worked examples when adding new experiment variants
