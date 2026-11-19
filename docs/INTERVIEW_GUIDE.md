# Interview Guide — LoRA Fine-Tuning Lab

How to present this project and survive the follow-ups.

**Read this first.** Of the three repos, this one has the *weakest* result and
the *strongest* story. `exact_match = 0.23` is not impressive. What is impressive
is that the first version of this experiment reported **1.00**, and you are the
one who found out why that was wrong. Lead with that, every time. A candidate who
catches their own inflated metric is demonstrating the single hardest thing to
hire for in ML.

Do not lead with "I fine-tuned a model." Everyone has.

---

## 1. The 60-second version

> "It's a LoRA fine-tune of Qwen2.5-0.5B that turns a free-text support ticket
> into a strict JSON object — category, severity, plan, and whether to escalate.
> 1.75% of parameters trainable, 35 MB adapter, 49 minutes on my laptop.
>
> The interesting part is that my first result was wrong. I got `exact_match` of
> 1.00 on the held-out set — every field, perfect. A perfect score on held-out
> data is a bug report, not a result, so I audited the split and found that 100%
> of the phrases carrying the labels in my 'unseen' set had appeared in training.
> I'd only varied the sentence structure, not the phrasing. The model had
> memorised phrase-to-label.
>
> So I built a third split that paraphrases every signal-carrying phrase, with an
> assertion in the generator that phrase overlap with training is zero. Same
> adapter, same prompt, honest question: **1.00 dropped to 0.23.**
>
> And then the diagnostic told me exactly what broke, which I'd like to walk
> through."

Stop. They will ask.

---

## 2. The thing to walk through

> "Per-field on the hard split: category went from 0.22 to 0.76, severity 0.35 to
> 0.83 — both well above their majority-class baselines, so that's real learning.
> But `plan` came out at 0.33, which is *below* its 0.35 majority baseline.
>
> I'd added a prediction-distribution report alongside every accuracy, and it
> said the model answered `self_serve` on **115 of 120 rows**. It hadn't partly
> learned `plan` — it had collapsed to one class.
>
> The cause is visible in the data. My training phrases contained literal tier
> names: 'we are on the Enterprise plan'. The held-out phrases paraphrased those
> away: 'our agreement includes round-the-clock coverage and a named contact'.
> The model had learned the string `Enterprise`, not the concept of a tier. Remove
> the string and it has nothing, so it falls back to the prior.
>
> That's one concrete, diagnosed defect with an obvious fix — paraphrase or drop
> the tier names in a fraction of the training examples so the literal string
> can't be leaned on. It's the first thing in my Next section."

**Why this answer works:** a number, a diagnosis, a mechanism, and a fix. Most
candidates can produce the number.

---

## 3. The second finding, which is sharper than it sounds

> "The distribution report also undercut my *baseline*. Few-shot prompting scored
> 0.72 on `escalate` against a 0.73 majority baseline — which looks like it almost
> works. The distribution showed it answered `false` on **all 120 rows**. Its 0.72
> is a constant. It's worth nothing.
>
> Same for the base model zero-shot: constant `ingestion` for category, constant
> `sev2` for severity, on every single row.
>
> The adapter was the only condition whose predictions were actually distributed.
> That's what makes its 0.76 and 0.83 mean something — and it's why I now report
> the predicted-class distribution next to every accuracy. An accuracy near a
> majority baseline and a collapsed distribution are the same number with
> completely different meanings, and a metrics table alone can't tell them apart."

---

## 4. The questions you will get

### "Why LoRA instead of a full fine-tune?"

> "Memory and operations. A full fine-tune updates every weight, so you carry
> gradients and Adam's two moments for all 0.5B parameters, and the artifact is a
> gigabyte per task. LoRA freezes the base and learns a low-rank update — the
> insight being that the *update* a fine-tune applies is empirically low-rank, so
> you don't need all the degrees of freedom the matrix has.
>
> Concretely here: 8.8 million trainable parameters out of 503 million, 1.75%, a
> 35 MB adapter, 1.04 GB peak memory. That's what makes it run on a laptop.
>
> The operational argument is bigger though. `B` is initialised to zero, so the
> adapter starts at *exactly* the base model's behaviour — a LoRA run can't be
> worse than base at step zero, which a full fine-tune can't promise. And you can
> merge the adapter into the base weights, so inference costs what the base costs.
> One served base model plus N task adapters is the deployment pattern."

### "What do rank and alpha actually do? Why 16 and 32?"

> "Rank is the width of the bottleneck — it sets adapter capacity. Too low and it
> underfits anything structural; too high and on a small dataset you've handed it
> enough capacity to memorise. 8 to 32 is the usual working band.
>
> Alpha scales the update by alpha over rank — here 32/16, so a factor of 2. The
> reason that parameterisation exists is to decouple the learning rate from the
> rank: without it, doubling the rank roughly doubles the update magnitude and
> you'd retune the LR every time you changed the rank. Keeping alpha = 2r is the
> convention.
>
> Honest answer on why 16: it's the middle of the defensible range, not a measured
> optimum — I didn't sweep it. And given my loss hit 2e-4 before epoch 2, I'd
> expect r=4 to match it. Lower rank is also the standard remedy for the
> memorisation I actually observed, so it's on my list."

### "Which modules did you adapt, and why?"

> "All four attention projections plus the three MLP projections — q, k, v, o,
> gate, up, down.
>
> The original LoRA paper adapted only q and v, which is cheaper and often enough
> for a style change. But my task changes the output *format*, and token-level
> formatting behaviour lives substantially in the MLP, not just in attention. So I
> included it. The cost is directly visible: that's most of why 1.75% of params
> are trainable instead of about 0.3%.
>
> If VRAM were the binding constraint, dropping back to q and v is the first cut
> I'd make."

### "Why is the learning rate 2e-4? That's high."

> "It's roughly 10 to 100x a full-fine-tune rate, and that's normal for LoRA.
> Only the adapter moves, it starts from zero, and there are far fewer parameters
> to coordinate. 1e-4 to 3e-4 is the usual band.
>
> The warmup matters more than people expect — `B` starts at zero, so the earliest
> steps produce large relative updates, and warming up stops the adapter being
> thrown somewhere bad before the cosine schedule settles."

### "Walk me through your training script. What's the easiest thing to get wrong?"

The highest-signal question here. Have the answer cold.

> "Masking the prompt out of the loss. I set the prompt tokens' labels to -100 so
> loss is computed on the completion only.
>
> Without it you're also training the model to reproduce the system instruction
> and the ticket text — and that's *most of the sequence*. My prompt is about 220
> tokens and the target is about 40, so training on the full sequence spends
> roughly 85% of the gradient teaching the model to generate text it will always
> be given at inference.
>
> The reason the mistake survives in the wild is that nothing visibly breaks. Loss
> goes down, the model learns the task eventually. It just makes you several times
> less sample-efficient than you should be.
>
> Related one in the collator: pad tokens have to be masked out of the loss as
> well as out of attention, or you teach the model to emit padding. And I *drop*
> examples over the max length rather than truncate them — truncating cuts the end
> off a JSON target, which teaches malformed output, the exact failure the
> fine-tune is supposed to remove."

### "Why did you build a few-shot baseline? Isn't the comparison obviously fine-tuning vs. base?"

> "Because 'fine-tuned beats base' is nearly worthless for a format task, and
> because few-shot prompting is what a sensible engineer tries *first* — it costs
> nothing, ships in minutes, and rolls back instantly. If four examples in a
> prompt get you there, the fine-tune has to justify itself on a different axis.
>
> In my case it didn't get there: few-shot was 0.03 exact match on the hard split
> versus the adapter's 0.23, and its per-field numbers were collapsed constants.
> So fine-tuning genuinely did something prompting couldn't.
>
> But I also report the axis where few-shot loses even when it wins: prompt
> tokens. 436 median versus 175. Few-shot pays for those examples on every single
> request forever; the adapter paid 49 minutes once. If accuracy had been a wash,
> that's where I'd have made the argument."

### "When would you *not* fine-tune?"

> "When the thing you want to change is what the model *knows* rather than how it
> answers. Fine-tuning teaches behaviour and format well and injects facts badly —
> it'll learn to produce confident, fluent, wrong facts. Changing knowledge is a
> retrieval problem, which is what my `rag-service` repo is.
>
> I'd also not fine-tune when prompting hasn't been tried properly, when the data
> is under a few hundred examples, or when the requirement is only 'valid JSON' —
> grammar-constrained decoding gets you 100% valid JSON with no training at all.
> Fine-tuning earns its place when you want the *content* right and the prompt
> short, which is the case I measured."

### "What's QLoRA and would you have used it?"

> "LoRA plus a 4-bit quantised frozen base. It cuts memory further at some quality
> and speed cost, and you reach for it when the base won't fit in bf16.
>
> I didn't need it — 0.5B in bf16 peaked at 1.04 GB. At 7B-plus on a single
> consumer GPU, QLoRA is how you make it fit at all."

### "Your `exact_match` is 0.23. Is that good?"

Do not get defensive. This is the question you built the project to answer.

> "No. Three of four fields right isn't a working triage system, and I say so in
> the README.
>
> What I'd argue is useful: it's 0.23 against 0.02 for the base model and 0.03 for
> few-shot, on phrasing the model has genuinely never seen, and it's capped by one
> diagnosed failure — the `plan` field collapsed to a constant, so most of the
> misses are a single wrong field. That's a fixable defect, not a vague
> underperformance.
>
> I'd much rather hand you a 0.23 I can explain mechanically than the 1.00 I
> originally got and couldn't."

### "How would you make it better?"

> "In order, and the order is the point.
>
> First, fix the `plan` collapse — paraphrase or drop the literal tier names in
> part of the training set so the model can't lean on the string. That's the one
> diagnosed defect.
>
> Second, one epoch and r=4. The loss hit 2e-4 before epoch 2, so three epochs of
> capacity was buying memorisation, not capability — which is plausibly part of
> why `plan` collapsed in the first place.
>
> Third, a harder split where severity and category are *correlated* the way they
> are in reality, to check whether it learned the escalation policy or just the
> marginals.
>
> Only then a bigger model. Scaling up before fixing a diagnosed collapse buys a
> more expensive version of the same bug."

### "Your data is synthetic. Doesn't that invalidate everything?"

> "It limits it severely and I lead with that in the README. There's no label
> noise, where real triage labels disagree between humans maybe 10 to 20% of the
> time — that's an invisible ceiling. And my fields are sampled independently, so
> some tickets are semantically incoherent, which makes them cleanly separable for
> measurement and less realistic.
>
> What synthetic data *can* legitimately support is 'the pipeline is correct and
> the method is sound' — and specifically, it's what let me construct the
> phrase-level holdout that caught my own bug. With real data I wouldn't have been
> able to guarantee zero phrase overlap, so I'd have had a weaker honest number
> and a less obvious artifact.
>
> The conclusion I'd defend is about method. The conclusion I would not defend is
> that this model is ready for production triage."

---

## 5. Questions where the honest answer is "I didn't"

| Question | Answer |
|---|---|
| "Did you sweep hyperparameters?" | "No. r=16, alpha=32, lr=2e-4 are defensible defaults, not measured optima, and the README says that. The loss curve suggests r=4 and one epoch would match." |
| "Did you use 1.5B?" | "No. The measured run is 0.5B, because it trains in a sane time on an M4. `--model` switches it and the notebook defaults to 1.5B on a T4 — unmeasured." |
| "Is the adapter on the Hub?" | "No — that needs a write token I don't have. The push cell is in the notebook, ready. The adapter weights are committed to the repo instead." |
| "Multiple seeds? Error bars?" | "One seed, one run. On 120 rows one row is 0.8pp, so the 0.76 vs 0.83 gap between category and severity isn't meaningfully distinguishable, and I don't claim it is." |
| "LLM-as-judge or human eval?" | "Neither. Every metric is exact-match against programmatically generated labels. For this task that's appropriate; for a generation task it wouldn't be." |
| "Constrained decoding baseline?" | "Not implemented. It's the honest competitor for a pure format task — a grammar gets 100% valid JSON with no training. It happens not to bite here because *every* condition already scores 1.00 on JSON validity, including the base model." |
| "Did it beat the base model on format?" | "There was nothing to beat. Qwen2.5-0.5B-Instruct emits clean JSON when asked — `json_strict` is 1.00 for every condition on every split. So unlike the usual structured-output story, none of my gain is format compliance. It's all field accuracy, which is the harder result." |

---

## 6. Where this sits on your résumé

**Claims this repo makes defensible:** LLM fine-tuning, LoRA, PEFT, Hugging Face
Transformers, PyTorch, parameter-efficient training, evaluation design, held-out
split construction, metric interpretation, Apple MPS and CUDA training.

**Claims it does not back.** Don't let it imply: QLoRA (not used — know the
difference, don't claim the practice), RLHF/DPO/preference tuning, distributed or
multi-GPU training, production model serving, large-model training. And it is not
a RAG or agent project — those are the sibling repos.

**One sentence if someone opens it:** "My first result was a perfect 1.00; I found
out why that was wrong, built the split that exposed it, and the honest number is
0.23 with one diagnosed failure I can point at."

---

## 7. The mindset

This project's value is almost entirely methodological, and that is worth more
than a good score, because good scores are cheap and trustworthy scores are not.

Three habits it demonstrates, in the order an interviewer will value them:

1. **A perfect held-out score was treated as a bug.** That instinct is the
   difference between an engineer who reports metrics and one who can be trusted
   with them.
2. **Every accuracy is reported next to its majority-class baseline and its
   prediction distribution.** That is what turned "escalate is 0.72, nearly
   working" into "escalate is a constant, worth nothing."
3. **The baseline was measured before the fancy thing**, and the write-up states
   the axis where the fancy thing would have lost.

Avoid:
- **Overselling 0.23.** It isn't good. Say so first; the story is why you trust it.
- **Claiming QLoRA, 1.5B, or Hub hosting.** None were done.
- **Hiding the 1.00.** It is the best thing in the repo. The artifact *is* the
  finding.
