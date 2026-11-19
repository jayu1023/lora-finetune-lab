# Explainer — LoRA Fine-Tuning Lab

What LoRA actually is, what every hyperparameter in this repo costs, and why the
evaluation is built the way it is. No prior fine-tuning knowledge assumed.

The measured numbers live in [`../README.md`](../README.md). This document is the
method; the README is the result.

---

## 1. When to fine-tune, and when not to

Three ways to make a model do something it does not do out of the box:

| Approach | What it changes | Good at | Bad at |
|---|---|---|---|
| **Prompting** (zero/few-shot) | nothing — the weights are untouched | fast iteration, no training, instant rollback | long prompts on every request; bounded by what the model can be talked into |
| **RAG** | what the model *knows* at inference time | injecting facts that change often, attribution | teaching behaviour or format |
| **Fine-tuning** | the weights (or an adapter) | **behaviour and format**; shortening prompts | injecting facts — expensive, and it hallucinates them back |

The one-line rule: **fine-tune to change *how* the model answers; retrieve to
change *what* it knows.** This repo's task — "always reply with exactly this JSON
schema" — is a format and behaviour task, which is squarely in fine-tuning's
lane. A question like "what is our SLA for Enterprise" is a retrieval task, and
belongs in the sibling `rag-service` project.

The trap worth naming: fine-tuning *feels* like the serious answer, so people
reach for it when a four-example prompt would have done. That is exactly why this
repo measures a few-shot baseline before it reports the adapter — see §6.

---

## 2. What LoRA is

A full fine-tune updates every weight. For a 0.5B model that means 0.5B
gradients, 0.5B optimizer moments (×2 for Adam), and a 1GB+ artifact per task.
For a 7B model it stops fitting on normal hardware at all.

**LoRA (Low-Rank Adaptation)** starts from an observation: the *update* a
fine-tune applies to a weight matrix tends to be low-rank — it does not need all
the degrees of freedom the matrix has. So instead of learning `ΔW` directly
(shape `d × k`), learn two thin matrices whose product has that shape:

```
          frozen                    trainable
  h  =  W₀ · x        +      (α / r) · B · A · x
        (d × k)                (d × r)  (r × k)
```

With `d = k = 1024` and `r = 16`: `ΔW` would be 1,048,576 parameters; `B·A` is
32,768. **A 32x reduction on that matrix**, and `W₀` never moves.

Three consequences that matter operationally:

1. **Only the adapter is trained**, so optimizer state shrinks by the same factor.
   This is what makes a 1.5B LoRA fit on a free T4.
2. **`A` is initialised random, `B` is initialised to zero**, so `B·A = 0` at step
   zero. Training starts *exactly* at the base model's behaviour — a LoRA run
   cannot be worse than the base model at initialisation, which full fine-tuning
   cannot promise.
3. **The adapter is a few MB**, and it can be merged into the base weights
   (`merge_and_unload()`) so inference costs exactly what the base model costs.
   One served base model + N task adapters is the deployment pattern this buys.

**QLoRA** is LoRA plus a 4-bit quantised frozen base. It cuts memory further at
some quality and speed cost, and it is what you reach for when the base model
will not fit in bf16. This repo uses plain bf16 LoRA because a 0.5B model fits
comfortably; the notebook notes where QLoRA would go for a 7B+ base.

---

## 3. Every hyperparameter, and what it costs

### `r = 16` (rank)

The width of the bottleneck — the one knob that sets adapter capacity.

- Too low (r=1–4): underfits anything structural. Fine for a style nudge.
- Too high (r=128+): more parameters, slower, and on a small dataset it
  overfits — you have handed it enough capacity to memorise.
- 8–32 is the usual working range for a task like this.

16 was chosen as the middle of that range for a format task on 600 examples. The
honest caveat: **this repo does not sweep `r`**, so 16 is a defensible default,
not a measured optimum. On a task this easy a sweep would likely show r=4 is
enough — and saying so is more useful than implying 16 was derived.

### `alpha = 32` (scaling)

The update is scaled by `α / r`. Here that is `32 / 16 = 2`.

The reason this parameterisation exists: it decouples the learning rate from the
rank. Without it, doubling `r` would also roughly double the update magnitude and
you would have to retune the LR every time you changed the rank. Keeping
`α = 2r` is the common convention and is what this repo does.

### `target_modules` — attention **and** MLP

```python
("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
```

The classic LoRA paper adapted only `q_proj` and `v_proj`. That is cheaper and
often enough for a *style* change.

This task changes the **output format**, and token-level formatting behaviour
lives substantially in the MLP (`gate/up/down`), not only in attention. So all
seven projections are adapted. The cost is directly visible: including the MLP is
most of why 1.75% of parameters are trainable here instead of ~0.3%.

If VRAM were the binding constraint, dropping to `q_proj, v_proj` is the first
cut to make.

### `learning_rate = 2e-4`

Roughly 10–100x a typical full-fine-tune LR (which would be 1e-5 to 5e-5). That
is normal and not a mistake: only the adapter moves, it starts from zero, and it
has far fewer parameters to coordinate. 1e-4 to 3e-4 is the usual LoRA band.

Cosine schedule with a short warmup. The warmup matters more than it looks —
`B` starts at zero, so the first steps produce large relative updates, and
warming up stops the adapter being thrown somewhere bad before the schedule
settles.

### `epochs = 3`, `batch_size = 4`, `grad_accum = 2`

Effective batch = 8. On 600 examples that is 75 optimizer steps per epoch.

Gradient accumulation exists to decouple the batch size the *optimizer* sees from
the one *memory* allows: four sequences fit, eight do not, so two forward/backward
passes are accumulated before stepping. The gradient is mathematically the same
as a batch of 8; only peak memory differs.

The README reports what the loss curve actually did, and whether 3 epochs was
more than this task needed.

### `max_length = 512`, with over-length examples **dropped, not truncated**

Truncating a training example cuts the end off the target — which teaches the
model to emit *malformed* JSON, the exact failure the fine-tune is supposed to
eliminate. Dropping is the correct handling for a structured-output target. The
script prints how many it dropped so the number is never silent.

---

## 4. The detail most fine-tuning scripts get wrong

**Loss is computed on the completion only.** The prompt tokens are masked to
`-100`:

```python
labels = [IGNORE] * len(prompt_ids) + completion_ids
```

Without this, the model is also trained to reproduce the system instruction and
the ticket text. That is not merely wasteful — it is *most of the sequence*. The
prompt here is ~220 tokens and the target is ~40, so training on the full
sequence spends roughly 85% of the gradient signal teaching the model to generate
text it will always be *given* at inference.

It still "works" — loss goes down, and the model learns the task eventually —
which is exactly why the mistake survives: nothing visibly breaks. It just makes
your fine-tune several times less sample-efficient than it should be.

A related detail in the collator: padding is masked out of the loss *as well as*
out attention. Padding tokens that reach the loss teach the model to emit pad
tokens.

---

## 5. The task, and the one field that is not extraction

Input: a free-text support ticket. Output:

```json
{"category":"ingestion","severity":"sev1","plan":"enterprise","escalate":true}
```

Three fields are **extraction** — the signal is somewhere in the text
("events stopped arriving" → `ingestion`).

`escalate` is **never stated**. It has to be inferred from a policy:

```python
escalate = severity in {"sev1", "sev2"} and category in {"ingestion", "api"}
```

That asymmetry is the point. A model can learn to copy three fields out of text
without learning anything about the task. `escalate` requires combining two
inferred values under a rule, so it is the field that separates pattern-matching
from learning — and it is reported separately for exactly that reason.

It also comes with a trap the README has to state: **`escalate` is ~73% one
class**, so "always answer `false`" scores 0.73. Any `escalate` number below that
is worse than a constant, and a number slightly above it means almost nothing.
The dataset generator prints every field's majority-class baseline so this can
never be quietly skipped.

---

## 6. Why there are three conditions, not two

The result "fine-tuned model beats the base model" is close to worthless for a
format task. A base instruct model asked for JSON will often produce
`Sure! Here's the JSON: {...}`, which fails a strict format check. Beating that is
automatic and proves nothing about whether fine-tuning was the right tool.

So there are three:

| Condition | Prompt | What it tells you |
|---|---|---|
| `base_zeroshot` | instruction only | the floor |
| `base_fewshot` | instruction + 4 worked examples | **the honest baseline** |
| `lora` | instruction only, adapter loaded | the fine-tune |

`base_fewshot` is the comparison that matters, because it is the thing a
sensible engineer would try *first* — it costs nothing, ships in minutes, and
rolls back instantly. If the adapter does not beat it on accuracy, then the
fine-tune has to justify itself on a different axis, and there is a real one:
**prompt tokens**. Few-shot pays for those examples on every single request,
forever; the adapter pays once at training time. The eval reports median prompt
length per condition so that trade is visible rather than asserted.

All three conditions share the same system prompt and the same target format,
from the same `src/prompts.py`. Training on one format and evaluating on another
is the most common way to produce a fine-tuning number that means nothing, and
one shared code path plus tests pinning it is the only real defence.

---

## 7. The split that makes the numbers honest

The data is **synthetic**, generated from phrasing templates. That is the biggest
limitation in this repo and it is stated plainly in the README.

A random train/test split over template-generated data measures almost nothing:
the test set is the same handful of skeletons with different nouns, so a model
that memorised the skeletons scores brilliantly.

So the split is by template family — and then, after that turned out not to be
enough, by phrase pool as well:

| Split | Templates | Phrases | Measures |
|---|---|---|---|
| `test_seen` | training | training | the ceiling; saturated |
| `test_unseen` | **held out** | training | tolerance of unfamiliar sentence structure |
| **`test_hard`** | **held out** | **held out** | **generalisation** |

### Why there are three and not two

The first version of this repo had only the first two, and reported
`exact_match = 1.00` on `test_unseen`. A perfect held-out score is a bug report,
so an audit was run:

```
category  train=25  unseen=25  shared=25  -> 100% of holdout phrases were SEEN in training
severity  train=12  unseen=12  shared=12  -> 100%
plan      train=12  unseen=12  shared=12  -> 100%
```

The split varied the **skeleton** of each sentence while reusing the **exact
phrases that carry each label**. "Events stopped arriving in the pipeline"
appeared in both, so the adapter scored 1.00 by memorising `phrase → label`.

`test_hard` paraphrases every signal-carrying phrase, and `make_dataset.py`
**asserts the overlap is zero** — a leak fails the build rather than inflating a
number. Training data was unchanged, so the same adapter is evaluated; only the
question got honest. `1.00 → 0.23`.

The lesson generalises past this repo: **the holdout has to withhold the thing
the model could cheat on.** A split that withholds sentence structure while
keeping the vocabulary measures almost nothing, and it fails in the flattering
direction, which is the direction nobody checks.

Two further honesty measures in the generator: tickets are deduplicated so a leak
cannot happen by collision, and an explicit assertion fails the build if any
ticket text appears in both train and test.

### What the synthetic data still cannot tell you

- Real tickets are messier, longer, multilingual, and contain contradictions.
- Category, severity and plan are sampled **independently**, so some generated
  tickets are semantically odd ("how do we rotate our API key" + "everything is
  down"). That makes the fields cleanly separable for measurement and makes the
  data less realistic.
- The label is generated *from* the text by construction, so there is no label
  noise. Real triage labels disagree between humans maybe 10–20% of the time, and
  that ceiling is invisible here.

The correct conclusion from a synthetic result is "the pipeline works and the
method is sound", never "this model is ready for production triage".

---

## 8. The metrics, and what each one hides

| Metric | Measures | Hides |
|---|---|---|
| `json_lenient` | a `{...}` could be extracted at all | whether the model is chatty around it |
| `json_strict` | output was exactly the four keys | nothing — this is the format number |
| `exact_match` | all four fields correct | which field failed |
| field accuracy | per field, unparsed counted wrong | whether failures are format or reasoning |
| **`accuracy given parsed`** | per field, over parsed rows only | — this is the one that *reveals* the above |
| **`predicted_distribution`** | how often each class was predicted | — this is the one that reveals a **collapse** |
| `median_prompt_tokens` | cost per request | — |

### The collapse detector

An accuracy sitting near its majority-class baseline has two completely different
explanations, and no accuracy number can separate them: the model is partially
right, or the model has collapsed to one answer. So the eval reports the
predicted-class distribution next to every accuracy.

It earned itself immediately. On `test_hard` the adapter's `plan` accuracy was
0.33 against a 0.35 majority baseline, which reads as "partially learned". The
distribution said it answered `self_serve` on **115 of 120 rows** — collapsed.
Same diagnostic showed the few-shot baseline's `escalate = 0.72` was a constant
`false` on all 120 rows, i.e. worth nothing.

**The `given parsed` column is the second most useful number in the table.** If the
adapter's `exact_match` is far above the baseline's but their `given parsed`
accuracies are similar, then the fine-tune taught the model *to emit JSON* and
taught it nothing about triage. That is a real, shippable win — format compliance
is why people fine-tune for structured output — but it is a different claim from
"the model got better at the task", and one aggregate accuracy number cannot tell
them apart.

### Deliberately lenient parsing

`parse_prediction` pulls the first `{...}` out of whatever the model said, strips
code fences, and `normalise` accepts `"Sev1"` and the string `"true"`. All of
that makes the **baseline look better**, which shrinks the reported fine-tuning
gain.

That is the right direction to be generous in. A strict parser would inflate the
result by scoring the base model wrong for capitalisation, and the number would
be measuring the parser instead of the model. `json_strict` is reported alongside
so the strict view is still visible.

### Greedy decoding

`do_sample=False`. Sampling would make the eval unreproducible, and for a
structured-extraction task there is no upside to creativity.

---

## 9. What this repo deliberately does not do

| Missing | Why | What adding it looks like |
|---|---|---|
| Hyperparameter sweep (`r`, `alpha`, LR) | one config, measured honestly, beats a sweep reported selectively | sweep `r ∈ {4,16,64}`, report all of it, not just the best |
| A fix for the `plan` collapse | it is diagnosed, not fixed — and a documented defect is more useful than a quiet patch | paraphrase or drop literal tier names in part of the training set |
| QLoRA / 4-bit | a 0.5B base fits in bf16 | `BitsAndBytesConfig(load_in_4bit=True)`; needed for a 7B+ base |
| Real data | no licensed ticket corpus to hand | the limitation that matters most; everything else is secondary |
| Constrained decoding baseline | a fourth condition, and a strong one | grammar-constrained generation gets 100% valid JSON with *no* training — the honest competitor to a format fine-tune |
| Human agreement ceiling | synthetic labels have none | relabel 100 rows twice; the disagreement rate is the real ceiling |
| Larger base model | 1.5B is in the notebook, not in the measured run | `--model Qwen/Qwen2.5-1.5B-Instruct` on a T4 |

The fourth row is the one a sharp interviewer will raise, so it is listed rather
than waited for: **if you only need valid JSON, constrained decoding gets you
there without training at all.** Fine-tuning earns its place when you also want
the *content* to be right and the prompt to be short.

---

## 10. Reading the code in 10 minutes

1. `src/prompts.py` — the shared contract. Read first; training and all three
   eval conditions go through it.
2. `src/make_dataset.py` — the escalation policy, and the template-family split.
3. `src/train_lora.py` → `ExtractionDataset.__init__` — the prompt masking, which
   is the one line that matters most.
4. `src/evaluate.py` → `score()` — where `given parsed` is computed.
5. `tests/test_scoring.py` — the metrics pinned by example, including the cases
   that demonstrate why `given parsed` and the collapse detector exist.
6. `tests/test_dataset.py`, the `test_hard` section — the assertions that stop the
   phrase leak coming back.
