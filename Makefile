PY = python3

# ============================================================================
# Dev iteration loop — run every code change
# ============================================================================
#
#   make test      fast unit tests (<30s, CPU-only) — run on every save
#   make bitwise   bitwise regression vs baselines (GPU, <10min) — before merge
#   make smoke     short end-to-end smoke (GPU, <5min)
#   make lint      ruff lint
#   make ci        lint + test  (default pre-commit)
#   make all       lint + test + bitwise + smoke (pre-merge)

.PHONY: test bitwise smoke lint fmt ci all clean

test:
	$(PY) -m pytest tests/unit -x -q --timeout=15

bitwise:
	$(PY) -m pytest tests/regression -m bitwise -x --timeout=600

smoke:
	$(PY) -m pytest tests/integration -x --timeout=300

lint:
	ruff check . || echo "(ruff not installed, skipping)"

fmt:
	ruff check --fix . || echo "(ruff not installed, skipping)"

ci: lint test
	@echo "CI green. For pre-merge also run: make bitwise && make smoke"

all: lint test bitwise smoke
	@echo "ALL passed."

clean:
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name .pytest_cache -type d -exec rm -rf {} +
	find . -name '*.pyc' -delete

# ============================================================================
# Legacy alignment test suite (kept for backward-compat)
# ============================================================================

# ---- Reference pull (DSW-only, ALIBABA_CLOUD_CREDENTIALS_URI must be set) ----
JOB_ID ?= dlc1q9arre48b0kx
WORKSPACE_ID ?= 262162

reference-pull:
	@mkdir -p reference reference/tb
	ALIBABA_CLOUD_CREDENTIALS_URI=http://localhost:7002/api/v1/credentials/0 \
	$(PY) /home/claudeuser/.claude/skills/pai/scripts/pai_manage.py get-job \
	  --job-id $(JOB_ID) > reference/$(JOB_ID).job.json
	@echo "[ok] reference/$(JOB_ID).job.json"

# ---- Alignment test suite (runs on DSW, no GPU) ----
align: tokenizer-align data-align model-align code-gaps-align loss-align

tokenizer-align:
	$(PY) -m pytest tests/test_tokenizer_alignment.py -v

data-align:
	$(PY) -m pytest tests/test_data_sampling_alignment.py -v --timeout=120

model-align:
	$(PY) -m pytest tests/test_model_alignment.py -v

code-gaps-align:
	$(PY) -m pytest tests/test_code_gaps.py -v

loss-align:
	$(PY) -m pytest tests/test_loss_trajectory.py -v

# ---- Bitwise resume (requires CUDA / 8-GPU box) ----
bitwise-check:
	$(PY) -m pytest tests/test_bitwise_resume.py -v

# ---- Dashboard ----
dashboard:
	@echo "open http://$(shell hostname):8787/dashboard/"
	$(PY) -m http.server 8787 --bind 0.0.0.0

# ---- Data prep for 00196 (runs on any machine with CPFS + cache access) ----
prepare-data-196:
	$(PY) prepare_cybertron_data.py --exp 196

.PHONY: reference-pull align tokenizer-align data-align model-align \
        code-gaps-align loss-align bitwise-check dashboard prepare-data-196
