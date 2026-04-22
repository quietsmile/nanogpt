# Alignment dashboard

All panels read from `../reports/*.json` and `../reference/tb/key_scalars.json`.

## Run locally (DSW or any machine with Python)

```bash
cd /home/claudeuser/nanogpt
python3 -m http.server 8787 --bind 0.0.0.0
```

Open http://<host>:8787/dashboard/ — the HTML fetches JSON from `../reports/`
and `../reference/` relative to its own location, so the server must be
started from the repo root.

## Panels

1. **Tokenizer** — Qwen2 class, 151643+16 vocab, EOD id, file md5s.
2. **Data sampling** — blend dataset count, cache size, first-1024 pair sha256, step-0 sample tokens.
3. **Model structure** — pytest asserts today; dashboard shows a placeholder until a JSON is written.
4. **Bitwise resume** — shown after `make bitwise-check` on an 8-GPU box.
5. **Loss trajectory** — Megatron reference curve always visible (7485 steps); nanogpt overlay after training run.
6. **Ckpt fingerprint** — produced by `python -m tools.ckpt_fingerprint`.

## How to refresh a panel

Every panel reads one file. Rerun the corresponding pytest or script; reload the page.
