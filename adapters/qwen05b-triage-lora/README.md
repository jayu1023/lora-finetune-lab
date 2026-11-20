---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - structured-output
  - information-extraction
license: bsd-3-clause
---

# qwen2.5-0.5b-ticket-triage-lora

LoRA adapter for **Qwen/Qwen2.5-0.5B-Instruct** that turns a free-text support ticket into one
strict JSON object.

```json
{"category":"api","severity":"sev2","plan":"enterprise","escalate":true}
```

Three fields are extraction — the signal is in the text. The fourth, `escalate`, is
**never stated**: it has to be inferred from a policy
(`severity in {sev1,sev2} AND category in {ingestion,api}`).

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
| base zero-shot | 0.02 | 0.22 | 0.35 | 0.28 | 0.64 | 175 |
| base few-shot (k=4) | 0.03 | 0.20 | 0.34 | 0.23 | 0.72 | 436 |
| **this adapter** | **0.23** | **0.76** | **0.83** | 0.33 | 0.82 | **175** |
| *majority-class baseline* | — | *0.22* | *0.36* | *0.35* | *0.73* | — |

**What works:** category and severity are well clear of their majority baselines on
phrasing the model has never seen. Few-shot prompting scored 0.03 — effectively
nothing — so fine-tuning did something prompting could not. Prompt cost is 175 tokens
versus 436.

**What does not:** `plan` came out at 0.33, **below** its 0.35 majority baseline. The
prediction-distribution diagnostic shows the model answered `self_serve` on **115 of 120
rows** — it collapsed to one class. Training phrases contained literal tier names ("we are
on the Enterprise plan"); the held-out phrases paraphrased them away. It learned the
string, not the concept. Diagnosed, documented, deliberately not patched.

On the saturated splits (`test_seen`, `test_unseen`) this adapter scores 1.00 on every
field. Those numbers are not meaningful — see above.

## Training

| | |
|---|---|
| rank / alpha / dropout | 16 / 32 / 0.05 |
| target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| learning rate | 0.0002, cosine, 6 warmup steps |
| epochs / batch / grad accum | 3 / 4 / 2 |
| trainable params | **8,798,208 / 502,830,976 (1.7497%)** |
| peak memory | 1.04 GB |
| training time | 2935.4 s on mps (macOS-26.5.2-arm64-arm-64bit (arm64)) |
| final train loss | 0.0253 |

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
- `exact_match = 0.23` is **not a working triage system.** It is a reference
  implementation with an evaluation that is honest about its own limits.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "jayu1023/qwen2.5-0.5b-ticket-triage-lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
```

Use the same system prompt the adapter was trained with — it is in
[`src/prompts.py`](https://github.com/jayu1023/lora-finetune-lab/blob/main/src/prompts.py).
A different prompt format at inference than at training is the usual reason a fine-tune
appears not to work.

## License

BSD-3-Clause.
