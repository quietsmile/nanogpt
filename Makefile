PY = python3

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
align: tokenizer-align data-align model-align loss-align

tokenizer-align:
	$(PY) -m pytest tests/test_tokenizer_alignment.py -v

data-align:
	$(PY) -m pytest tests/test_data_sampling_alignment.py -v --timeout=120

model-align:
	$(PY) -m pytest tests/test_model_alignment.py -v

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

.PHONY: reference-pull align tokenizer-align data-align model-align loss-align \
        bitwise-check dashboard prepare-data-196
