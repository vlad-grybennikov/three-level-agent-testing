# Task shortcuts `make help` lists everything (npm-run equivalent).
#
# Variables (override per invocation):
#   MODEL=qwen|opus|gpt          which predefined config to use (default qwen)
#   CONFIG=path/to/config.json   explicit config file (overrides MODEL)
#   K=5                          episodes per task for suite runs
#   FLAGS="--judge --variants"   extra runner flags for suite runs
#   MSG="..."                    instruction for `make episode`
#
# Examples:
#   make suite MODEL=opus K=5 FLAGS="--judge"
#   make variants MODEL=opus
#   make episode MODEL=opus MSG="Cancel Bruno Silva's internet order."

PY := .venv/bin/python
K ?= 3
FLAGS ?=

# Predefined model configs = MODEL resolves to a config file; an explicit
# CONFIG=... on the command line always wins.
MODEL ?= qwen
CONFIG.qwen := configs/local-qwen.json
CONFIG.opus := configs/claude.json
CONFIG.gpt  := configs/gpt.json
CONFIG ?= $(CONFIG.$(MODEL))
CFG_FLAG = --config $(CONFIG)

.DEFAULT_GOAL := help
.PHONY: help check-model test test-unit test-levels suite suite-qwen \
        suite-opus variants intents review-list chat ui episode notebook \
        export matrices

help: ## list available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-12s %s\n", $$1, $$2}'

check-model:
	@test -n "$(CONFIG)" || \
		(echo "unknown MODEL '$(MODEL)' please use qwen, opus, or gpt (or pass CONFIG=path)"; exit 1)

test: ## full offline test suite (no model needed)
	$(PY) -m pytest -q

test-unit: ## codebase unit tests only
	$(PY) -m pytest -q tests/unit

test-levels: ## three-level framework tests only
	$(PY) -m pytest -q approach_tests

suite: check-model ## live suite for MODEL -> report-$(MODEL).json  [MODEL, K, FLAGS]
	$(PY) -m telecom_aut.testing approach_tests/tasks --k $(K) \
		$(CFG_FLAG) --out report-$(MODEL).json $(FLAGS)

suite-qwen: MODEL := qwen
suite-qwen: suite ## alias: make suite MODEL=qwen

suite-opus: MODEL := opus
suite-opus: suite ## alias: make suite MODEL=opus

variants: check-model ## LLM instruction variants for all tasks (review after)  [MODEL]
	$(PY) -m telecom_aut.testing.variants approach_tests/tasks --n 4 $(CFG_FLAG)

intents: check-model ## LLM paraphrases for the intent dataset (review after)  [MODEL]
	$(PY) -m telecom_aut.testing.intent_augment \
		approach_tests/fixtures/level3_intents.json --n 3 $(CFG_FLAG)

review-list: ## show all files with still-unreviewed LLM generations
	@grep -rl '"reviewed": false' approach_tests || echo "nothing awaiting review"

NB ?= report_summary_opus.ipynb

notebook: ## open a Jupyter notebook (NB=report_summary_qwen.ipynb for the qwen one)
	$(PY) -m notebook $(NB)

export: ## export all tables (CSV/MD) + figures (PNG/PDF) from report-*.json to exports/
	$(PY) export_assets.py

matrices: ## regenerate paper Fig. 3 per-task matrices from exported CSVs to media/
	@mkdir -p media
	$(PY) final-matrices-generation.py exports/report-qwen/task_matrix-qwen.csv media/matrix_qwen.png
	$(PY) final-matrices-generation.py exports/report-opus/task_matrix-opus.csv media/matrix_opus.png

chat: check-model ## chat API server (pair with `make ui`)  [MODEL]
	$(PY) -m telecom_aut.chat $(CFG_FLAG)

ui: ## Next.js chat UI on :3011
	cd chat-ui && npm run dev

episode: check-model ## one live episode: make episode MSG="..."  [MODEL]
	@test -n "$(MSG)" || (echo 'usage: make episode MSG="..."' && exit 1)
	$(PY) -m telecom_aut.run "$(MSG)" $(CFG_FLAG)
