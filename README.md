# LoRA Fine-Tuning Lab

Fine-tunes Qwen2.5-0.5B-Instruct with PEFT + LoRA to turn a free-text support
ticket into one strict JSON object — and then tries hard to prove the result
wrong.

The headline is not the accuracy. It is that **the first version of this
experiment produced a perfect 1.00 and the perfect score was an artifact**, and
the repo now contains the harder split that exposed it, the diagnostic that
explains it, and the honest number that replaced it.

**Adapter on the Hub:** [`jayu1023/qwen2.5-0.5b-ticket-triage-lora`](https://huggingface.co/jayu1023/qwen2.5-0.5b-ticket-triage-lora) — 35 MB, with a
model card generated from the measured results rather than written by hand.

**Docs:** [`docs/EXPLAINER.md`](docs/EXPLAINER.md) — what LoRA is, what every
hyperparameter costs, why the eval is built this way.
[`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md) — how to present and defend it.

```bash
make venv install
make data          # deterministic synthetic dataset
make baselines     # the prompting baselines FIRST, before training
make train         # ~49 min on an M4 (MPS) / ~7 min on a Colab T4
make eval          # all 3 conditions x 3 splits
make test          # 43 unit tests, no model download
```

Notebook: [`notebooks/lora_finetune_lab.ipynb`](notebooks/lora_finetune_lab.ipynb)
— runs top to bottom on a free Colab T4 and calls into `src/`, so there is no
second copy of the training loop to drift.

---

## The task

```
Ticket: "Forwarding from our ops channel:
         > Signature verification on our callbacks fails every time
         > An important workflow is unusable although others still function
         Adding that our agreement includes round-the-clock coverage."

→ {"category":"api","severity":"sev2","plan":"enterprise","escalate":true}
```

Three fields are **extraction** — the signal is in the text. The fourth,
`escalate`, is **never stated**: it must be inferred from
`severity ∈ {sev1,sev2} AND category ∈ {ingestion,api}`. Keeping both kinds in
one schema is what lets the eval separate "learned to emit JSON" from "learned
the task".

---

## The experiment that was wrong, and how it got caught

The first run reported **`exact_match = 1.00` on the held-out split.** Every
field, both splits, perfect.

A perfect score on a held-out set is a bug report, not a result. The audit:

```
category  train= 25 unseen= 25 shared= 25  -> 100% of holdout phrases were SEEN in training
severity  train= 12 unseen= 12 shared= 12  -> 100%
plan      train= 12 unseen= 12 shared= 12  -> 100%
```

The split varied the **template skeleton** (a question vs. a forwarded quote vs.
a thread reply) while reusing the **exact phrases that carry each label**. So the
adapter scored 1.00 by memorising `phrase → label`, and the number measured
nothing about generalisation.

The fix is a third split, `test_hard`, which paraphrases every signal-carrying
phrase and **asserts zero phrase overlap with training** in the generator. The
training data did not change, so the same adapter is evaluated — only the
question being asked of it got honest.

| Split | Templates | Phrases | What it measures |
|---|---|---|---|
| `test_seen` | training | training | nothing much — the ceiling |
| `test_unseen` | **held out** | training | tolerance of unfamiliar sentence structure |
| **`test_hard`** | **held out** | **held out** | **generalisation. The only number worth quoting.** |

---

## Measured results

Qwen2.5-0.5B-Instruct, greedy decoding, 120 held-out rows per split.
`base_fewshot` = the same prompt with 4 worked examples — the baseline that
matters, because it is what a sensible engineer tries first.

### `test_hard` — held-out templates *and* phrases

| condition | exact | category | severity | plan | escalate | prompt tokens |
|---|---|---|---|---|---|---|
| `base_zeroshot` | 0.02 | 0.22 | 0.35 | 0.28 | 0.64 | 175 |
| `base_fewshot` | 0.03 | 0.20 | 0.34 | 0.23 | 0.72 | **436** |
| **`lora`** | **0.23** | **0.76** | **0.83** | 0.33 | 0.82 | **175** |
| *majority-class baseline* | — | *0.22* | *0.36* | *0.35* | *0.73* | — |

### `test_unseen` and `test_seen` — the flattering splits

| split | condition | exact | category | severity | plan | escalate |
|---|---|---|---|---|---|---|
| `test_unseen` | base_zeroshot | 0.01 | 0.15 | 0.40 | 0.60 | 0.69 |
| `test_unseen` | base_fewshot | 0.07 | 0.31 | 0.31 | 0.87 | 0.77 |
| `test_unseen` | **lora** | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 |
| `test_seen` | base_zeroshot | 0.01 | 0.23 | 0.30 | 0.59 | 0.80 |
| `test_seen` | base_fewshot | 0.10 | 0.41 | 0.37 | 0.91 | 0.64 |
| `test_seen` | **lora** | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 |

**The 1.00 → 0.23 drop between `test_unseen` and `test_hard` is the single most
informative number in this repo.** Same adapter, same model, same prompt. The
only thing that changed is whether the phrasing was one the model had seen. That
gap is the template-memorisation tax, and it is what a random train/test split
over synthetic data would have hidden completely.

### What got better

- **Exact match 0.02 → 0.23 on genuinely unseen phrasing**, roughly 11x, while
  few-shot prompting achieved 0.03 — i.e. nothing. For this task fine-tuning did
  something prompting could not.
- **`category` 0.22 → 0.76** and **`severity` 0.35 → 0.83** against majority
  baselines of 0.22 and 0.36. These are real: the model maps unfamiliar
  paraphrases to the right class.
- **`escalate` 0.82 vs a 0.73 majority baseline** — and this one needs the
  distribution to interpret, see below.
- **Prompt cost: 175 vs 436 median tokens.** 2.5x shorter on every single
  request, forever, versus a one-off 49-minute training cost.

### What got worse, or did not work

- **`plan` collapsed to 0.33, below its 0.35 majority baseline.** The
  prediction-distribution diagnostic says why: the adapter answered
  **`self_serve` on 115 of 120 rows**. It is not partially right; it has
  collapsed to one class. The training phrases contained literal tier names
  ("we are on the Enterprise plan"), the held-out phrases paraphrased them away
  ("our agreement includes round-the-clock coverage"), and the model had learned
  the *string*, not the concept.
- **`exact_match = 0.23` is low in absolute terms.** Three of four fields right
  is not a working triage system. The `plan` collapse caps it: most failures are
  a single wrong `plan` field.
- **`test_seen` and `test_unseen` are saturated at 1.00** and cannot regress, so
  they carry no signal. They are kept only to show the size of the gap.

### Why `base_fewshot` is worse than it looks

The distribution diagnostic also undercuts the baseline's `escalate = 0.72`:

```
base_fewshot  test_hard  most-predicted -> escalate: False 120/120
base_zeroshot test_hard  most-predicted -> category: ingestion 120/120  severity: sev2 120/120
lora          test_hard  most-predicted -> category: ingestion 41/120   severity: sev2 52/120
```

Few-shot answered `escalate: false` on **every single row**. Its 0.72 is just the
majority class, achieved by a constant — worth exactly nothing. The base model
is worse still: a constant `ingestion` and a constant `sev2`. The adapter is the
only condition whose predictions are actually distributed, which is what makes
its 0.76/0.83 meaningful and its `plan` number diagnosable.

This is why the repo reports the predicted-class distribution next to every
accuracy. An accuracy near a majority baseline and a collapsed distribution are
the same number with completely different meanings.

---

## Training run — the numbers to quote

Straight from [`adapters/qwen05b-triage-lora/training_card.json`](adapters/qwen05b-triage-lora/training_card.json),
written by the training script:

| | |
|---|---|
| base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| rank / alpha / dropout | **16 / 32 / 0.05** (α/r = 2) |
| target modules | `q,k,v,o_proj` + `gate,up,down_proj` — attention **and** MLP |
| learning rate / schedule | **2e-4**, cosine, 6 warmup steps |
| epochs / batch / grad accum | 3 / 4 / 2 (effective batch 8, 225 steps) |
| max length | 512, over-length examples **dropped not truncated** |
| trainable params | **8,798,208 / 502,830,976 = 1.75%** |
| peak memory | **1.04 GB** |
| training time | **2,935 s (49 min)** on Apple M4, MPS, bf16 |
| final train loss | 0.0253 (reached 2e-4 by epoch 1.7) |
| adapter size | **35 MB** (`adapter_model.safetensors`) |

**On the 49 minutes:** step time degraded from **2.0 s/it to ~30 s/it** over the
run — sustained MPS throughput on an M4 falls roughly 15x under thermal load.
The same job is ~7 minutes on a Colab T4. If you are benchmarking LoRA on Apple
silicon, measure the *end* of the run, not the first ten steps. (I initially
mis-diagnosed this as CPU contention; it is thermal.)

**On the loss:** it hit 2e-4 by epoch 1.7, so epoch 3 bought nothing except more
memorisation of the training phrasing — which is plausibly part of why `plan`
collapsed. One epoch was likely the right call and is not what was run.

---

## QLoRA vs LoRA, and 1 epoch vs 3

```bash
make train                              # LoRA, bf16 base
python src/train_lora.py --quantize 4bit --epochs 1   # QLoRA, NF4 + double quant
```

`--quantize 4bit` is the real QLoRA recipe: NF4 4-bit frozen base, double quantisation,
bf16 compute, `prepare_model_for_kbit_training`. Verified loading as `Linear4bit` layers on
Apple MPS — 451 MB footprint against ~1 GB for bf16.

### Matched 1-epoch comparison, `test_hard`

| run | exact | category | severity | plan | escalate | peak mem | final loss |
|---|---|---|---|---|---|---|---|
| **LoRA 1 epoch** | **0.23** | **0.92** | **0.85** | 0.35 | **0.91** | 1.04 GB | 0.0684 |
| QLoRA 1 epoch | 0.19 | 0.76 | 0.78 | 0.35 | 0.80 | **0.81 GB** | 0.0624 |
| LoRA 3 epochs | 0.23 | 0.76 | 0.83 | 0.33 | 0.82 | 1.04 GB | 0.0253 |

**Finding 1 — 4-bit cost accuracy here, and saved memory that was not scarce.**
QLoRA lost 4 points of exact match and 16 points of category accuracy against an otherwise
identical LoRA run, for 22% less peak memory. On a 0.5B model that is a bad trade: nothing
was close to not fitting. QLoRA earns its place when the base model *would not load
otherwise* — at 7B+ on a single consumer GPU — not as a default. Measuring it is the only
way to know which case you are in.

Note the training losses point the other way (0.0624 vs 0.0684). **Lower training loss, worse
held-out accuracy** is the ordinary shape of the thing: loss on the training distribution is
not the quantity anyone cares about.

**Finding 2 — 1 epoch generalises better than 3, which confirms a hypothesis this repo had
only asserted.** The 3-epoch README already said "loss hit 2e-4 before epoch 2, so epoch 3
bought memorisation rather than capability — plausibly part of why `plan` collapsed." That
was a guess. Trained for 1 epoch, the same config scores **category 0.92 against 0.76** and
**escalate 0.91 against 0.82** on held-out phrasing, with identical exact match. Training
loss is 27x *higher* (0.068 vs 0.0025) and the model is *better*. The extra epochs were
fitting the training phrasings.

`plan` stays at 0.33–0.35 in all three runs — still a near-constant `self_serve`. That
failure is in the data, not the training budget, which is consistent with the diagnosis and
is why more epochs and fewer epochs both fail to fix it.

### Why there is no timing claim here

QLoRA ran in 197 s and LoRA in 892 s, and **that comparison is not reported as a result**,
because the runs were not thermally matched — QLoRA ran on a cool machine and LoRA
immediately after it. Sustained MPS throughput on this M4 falls roughly 15x under load, which
is far larger than any real difference between the methods. A fair timing comparison needs
alternating runs with cooldowns, which was not done.

---

## Sample outputs — `lora` on `test_hard`

Every failure below is the real logged output. Note how often `plan` is the only
wrong field.

| Ticket (truncated) | Expected | Predicted | Wrong |
|---|---|---|---|
| "…could you point us at the documentation for adding a colleague…" | `how_to, sev3, enterprise, false` | `how_to, sev3, self_serve, false` | `plan` |
| "…signature verification on our callbacks fails every time / an important workflow is unusable…" | `api, sev2, enterprise, true` | `api, sev2, self_serve, true` | `plan` |
| "…a tile we built last month now comes up empty / purely a tidiness matter…" | `dashboards, sev3, business, false` | `dashboards, sev3, self_serve, false` | `plan` |
| "…there is a discrepancy between the statement and what we expected to pay / nobody in the organisation can do anything…" | `billing, sev1, self_serve, false` | `billing, **sev2**, self_serve, false` | `severity` |
| "…a tile we built last month now comes up empty…" | `dashboards, sev3, business, false` | `**ingestion**, sev3, self_serve, false` | `category`, `plan` |

Format compliance was never the problem: **`json_lenient` and `json_strict` are
1.00 for every condition on every split**, including the un-tuned base model.
Qwen2.5-0.5B-Instruct already emits clean JSON when asked. So unlike the usual
structured-output fine-tuning story, **none of the gain here is format
compliance** — it is all field accuracy, which is the harder and more honest
version of the result.

---

## Honest status

- **The data is synthetic**, generated from phrasing templates. This is the
  biggest limitation and no number here survives contact with real tickets
  unexamined. There is no label noise, whereas real triage labels disagree
  between humans maybe 10–20% of the time — an invisible ceiling.
- **Category, severity and plan are sampled independently**, so some tickets are
  semantically odd ("how do we rotate our API key" + "everything is down"). That
  makes fields cleanly separable for measurement and the data less realistic.
- **No hyperparameter sweep** over `r`, `alpha` or the learning rate — these are defensible
  defaults, not measured optima. Epochs and quantisation *were* ablated (above); rank was
  not, and `r=4` is the obvious next thing to try given what 1 epoch did.
- **One seed, one run.** On 120 rows, one row is 0.8pp; the 0.76 vs 0.83 gap
  between `category` and `severity` is not meaningfully distinguishable.
- **0.5B, not 1.5B.** The tracker suggested Qwen2.5-1.5B; the measured run is the
  0.5B because it trains in a sane time on an M4. `--model` switches it and the
  notebook defaults to 1.5B on a T4, unmeasured.
- **No constrained-decoding baseline.** A grammar constraint gets 100% valid JSON
  with no training at all. It is not a competitor here (every condition already
  scores 1.00 on format) but it would be on a model that emits messier JSON.
- **The adapter is on the Hub**: [`jayu1023/qwen2.5-0.5b-ticket-triage-lora`](https://huggingface.co/jayu1023/qwen2.5-0.5b-ticket-triage-lora),
  pushed by the repo owner with `make push`. The card is generated from
  `training_card.json` and `evals/results.json` rather than written by hand, so it cannot
  drift from what was measured — and it carries the unflattering parts too (the 1.00
  artifact, the `plan` collapse). The weights are also committed here, so nothing depends
  on the Hub existing.
- **No LLM-judge or human evaluation.** Every metric is exact-match against
  programmatically generated labels.

## What I would do next, in order

1. **Fix the `plan` collapse** — it is the one concrete, diagnosed defect.
   Train with the tier names paraphrased or removed from a fraction of examples,
   so the model cannot lean on the literal string.
2. ~~**One epoch**~~ — **done, and it worked**: category 0.76 → 0.92, escalate 0.82 → 0.91
   on held-out phrasing. `r=4` is still untested and is the remaining half of this.
3. **A fourth, harder split** where severity and category are *correlated* as they
   are in reality, to see whether the model learned the policy or the marginals.
4. Then, and only then, a larger base model. Scaling up before fixing a
   diagnosed collapse just buys a more expensive version of the same bug.

## License

BSD-3-Clause.
