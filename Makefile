VENV ?= .venv
PY   := $(VENV)/bin/python
MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
REPO  ?= jayu1023/qwen2.5-0.5b-ticket-triage-lora

.PHONY: help venv install data train eval eval-fast baselines test push push-dry all clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

venv: ## create a python 3.12 venv (needs uv)
	uv venv -p 3.12 $(VENV)

install: ## install dev dependencies
	uv pip install --python $(PY) -r requirements-dev.txt

data: ## generate the dataset (deterministic)
	$(PY) src/make_dataset.py

train: ## fine-tune the LoRA adapter
	$(PY) src/train_lora.py --model $(MODEL)

baselines: ## the two prompting baselines only (run these BEFORE training)
	$(PY) src/evaluate.py --conditions base_zeroshot base_fewshot --out evals/baselines.json

eval: ## all three conditions on both held-out splits
	$(PY) src/evaluate.py --model $(MODEL)

eval-fast: ## 40 rows per split, for a smoke test
	$(PY) src/evaluate.py --model $(MODEL) --limit 40 --out evals/smoke.json

push-dry: ## generate the model card, upload nothing
	$(PY) src/push_to_hub.py --repo $(REPO) --dry-run

push: ## push adapter + card to the Hub (needs YOUR write token: hf auth login)
	$(PY) src/push_to_hub.py --repo $(REPO)

test: ## unit tests (no model download)
	$(PY) -m pytest tests

all: data train eval ## the whole experiment end to end

clean:
	rm -rf runs .pytest_cache evals/smoke.json
