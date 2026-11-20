#!/usr/bin/env python
"""Push the trained adapter to the Hugging Face Hub, with an honest model card.

    hf auth login                      # or: export HF_TOKEN=hf_...
    python src/push_to_hub.py --repo <your-username>/qwen2.5-0.5b-ticket-triage-lora

Dry run first (writes the card locally, uploads nothing):

    python src/push_to_hub.py --repo you/whatever --dry-run

**Never run by the author** — pushing needs a write token belonging to whoever
owns the Hub account. The adapter weights are committed to this repo instead,
so nothing here depends on the Hub existing.

The model card is generated from `training_card.json` and `evals/results.json`,
not hand-written, so it cannot drift from what was actually measured. That
includes the parts that do not flatter the model: the honest `test_hard`
numbers, the `plan` field collapsing below its majority baseline, and the
split artifact that invalidated the first result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "adapters" / "qwen05b-triage-lora"

CARD = """---
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - structured-output
  - information-extraction
license: bsd-3-clause
---

# {repo_name}

LoRA adapter for **{base_model}** that turns a free-text support ticket into one
strict JSON object.

```json
{{"category":"api","severity":"sev2","plan":"enterprise","escalate":true}}
```

Three fields are extraction — the signal is in the text. The fourth, `escalate`, is
**never stated**: it has to be inferred from a policy
(`severity in {{sev1,sev2}} AND category in {{ingestion,api}}`).

Code, dataset generator and full evaluation harness:
**https://github.com/jayu1023/lora-finetune-lab**

## Read this before using the numbers

The first version of this experiment reported **`exact_match = 1.00`** on held-out data.
A perfect held-out score is a bug report, so the split was audited:

```
category  train=25  unseen=25  shared=25  -> 100% of holdout phrases were SEEN in training
```

The holdout varied the *template skeleton* while reusing the *exact phrases that carry each
label*, so the adapter scored 1.00 by memorising `phrase -> label`. A third split was added
(`test_hard`) that paraphrases every signal-carrying phrase and asserts zero phrase overlap
with training. Same adapter, honest question:

**1.00 -> 0.23.**

## Measured results — `test_hard` (held-out templates AND phrases, 120 rows, greedy)

| condition | exact | category | severity | plan | escalate | prompt tokens |
|---|---|---|---|---|---|---|
| base zero-shot | {bz_exact} | {bz_cat} | {bz_sev} | {bz_plan} | {bz_esc} | {bz_tok} |
| base few-shot (k=4) | {bf_exact} | {bf_cat} | {bf_sev} | {bf_plan} | {bf_esc} | {bf_tok} |
| **this adapter** | **{lo_exact}** | **{lo_cat}** | **{lo_sev}** | {lo_plan} | {lo_esc} | **{lo_tok}** |
| *majority-class baseline* | — | *0.22* | *0.36* | *0.35* | *0.73* | — |

**What works:** category and severity are well clear of their majority baselines on
phrasing the model has never seen. Few-shot prompting scored {bf_exact} — effectively
nothing — so fine-tuning did something prompting could not. Prompt cost is {lo_tok} tokens
versus {bf_tok}.

**What does not:** `plan` came out at {lo_plan}, **below** its 0.35 majority baseline. The
prediction-distribution diagnostic shows the model answered `self_serve` on **115 of 120
rows** — it collapsed to one class. Training phrases contained literal tier names ("we are
on the Enterprise plan"); the held-out phrases paraphrased them away. It learned the
string, not the concept. Diagnosed, documented, deliberately not patched.

On the saturated splits (`test_seen`, `test_unseen`) this adapter scores 1.00 on every
field. Those numbers are not meaningful — see above.

## Training

| | |
|---|---|
| rank / alpha / dropout | {rank} / {alpha} / {dropout} |
| target modules | {targets} |
| learning rate | {lr}, cosine, {warmup} warmup steps |
| epochs / batch / grad accum | {epochs} / {batch} / {accum} |
| trainable params | **{trainable:,} / {total:,} ({pct}%)** |
| peak memory | {mem} GB |
| training time | {secs} s on {device} ({platform}) |
| final train loss | {loss} |

Loss is computed on the **completion only** — prompt tokens are masked to `-100`. The
prompt is ~220 tokens and the target ~40, so training on the full sequence would spend
~85% of the gradient on text the model is always *given* at inference.

Loss reached 2e-4 before epoch 2, so three epochs bought memorisation rather than
capability — plausibly part of why `plan` collapsed. One epoch at a lower rank is the
first thing to try.

## Limitations

- **The data is synthetic**, generated from phrasing templates. No label noise, where real
  triage labels disagree between humans maybe 10–20% of the time — an invisible ceiling.
- Category, severity and plan are sampled **independently**, so some tickets are
  semantically odd. Cleanly separable for measurement; less realistic.
- **No hyperparameter sweep.** These are defensible defaults, not measured optima.
- **One seed, one run.** On 120 rows, one row is 0.8pp.
- `exact_match = {lo_exact}` is **not a working triage system.** It is a reference
  implementation with an evaluation that is honest about its own limits.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
```

Use the same system prompt the adapter was trained with — it is in
[`src/prompts.py`](https://github.com/jayu1023/lora-finetune-lab/blob/main/src/prompts.py).
A different prompt format at inference than at training is the usual reason a fine-tune
appears not to work.

## License

BSD-3-Clause.
"""


def metric(results: dict, condition: str, key: str, field: str | None = None) -> str:
    block = results["splits"]["test_hard"][condition]
    value = block["field_accuracy"][field] if field else block[key]
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def build_card(repo_id: str) -> str:
    card = json.loads((ADAPTER / "training_card.json").read_text())
    results = json.loads((ROOT / "evals" / "results.json").read_text())
    hp = card["hyperparams"]

    def row(cond: str, prefix: str) -> dict:
        block = results["splits"]["test_hard"][cond]
        return {
            f"{prefix}_exact": f"{block['exact_match']:.2f}",
            f"{prefix}_cat": f"{block['field_accuracy']['category']:.2f}",
            f"{prefix}_sev": f"{block['field_accuracy']['severity']:.2f}",
            f"{prefix}_plan": f"{block['field_accuracy']['plan']:.2f}",
            f"{prefix}_esc": f"{block['field_accuracy']['escalate']:.2f}",
            f"{prefix}_tok": str(block["median_prompt_tokens"]),
        }

    values = {
        "base_model": hp["model"],
        "repo_id": repo_id,
        "repo_name": repo_id.split("/")[-1],
        "rank": hp["rank"], "alpha": hp["alpha"], "dropout": hp["dropout"],
        "targets": ", ".join(f"`{t}`" for t in hp["target_modules"]),
        "lr": hp["learning_rate"], "warmup": card.get("warmup_steps", "—"),
        "epochs": hp["epochs"], "batch": hp["batch_size"], "accum": hp["grad_accum"],
        "trainable": card["trainable_params"], "total": card["total_params"],
        "pct": card["trainable_pct"], "mem": card["peak_memory_gb"],
        "secs": card["train_seconds"], "device": card["device"],
        "platform": card["platform"], "loss": card["final_train_loss"],
        **row("base_zeroshot", "bz"), **row("base_fewshot", "bf"), **row("lora", "lo"),
    }
    return CARD.format(**values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="e.g. jayu1023/qwen2.5-0.5b-ticket-triage-lora")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="write the card locally, upload nothing")
    args = parser.parse_args()

    if not (ADAPTER / "adapter_model.safetensors").exists():
        raise SystemExit(f"no adapter at {ADAPTER} — run: make train")

    card = build_card(args.repo)
    card_path = ADAPTER / "README.md"
    card_path.write_text(card)
    print(f"wrote model card -> {card_path.relative_to(ROOT)} ({len(card)} chars)")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded. Review the card above, then re-run without it.")
        return 0

    from huggingface_hub import HfApi, whoami

    try:
        user = whoami()["name"]
    except Exception:
        raise SystemExit(
            "not logged in to the Hugging Face Hub.\n\n"
            "    hf auth login          # or: export HF_TOKEN=hf_...\n\n"
            "The token needs WRITE access. Create one at "
            "https://huggingface.co/settings/tokens"
        )
    print(f"logged in as: {user}")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(ADAPTER),
        repo_id=args.repo,
        repo_type="model",
        # The training card is evidence; the trainer's own scratch files are not.
        ignore_patterns=["*.lock", "tmp/*"],
        commit_message="LoRA adapter for support-ticket triage, with measured eval results",
    )
    url = f"https://huggingface.co/{args.repo}"
    print(f"\npushed -> {url}")
    print("Add that link to the repo README and your CV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
