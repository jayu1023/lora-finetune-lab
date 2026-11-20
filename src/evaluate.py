#!/usr/bin/env python
"""Base vs few-shot vs LoRA, on two held-out sets.

    python src/evaluate.py                       # all three conditions
    python src/evaluate.py --conditions lora

Three conditions, because "fine-tuned beats zero-shot" is a result nobody should
be impressed by:

  base_zeroshot   the instruction only. The floor.
  base_fewshot    the same instruction + k worked examples. **The honest
                  baseline.** If prompting gets you here, the fine-tune has to
                  justify itself on something other than accuracy -- and the
                  prompt-token column is where it does.
  lora            the adapter, with the same instruction and no examples.

Two test sets:

  test_seen       training templates, training phrases
  test_unseen     held-out templates, training phrases -- isolates the effect of
                  an unfamiliar sentence structure
  test_hard       held-out templates AND held-out phrases -- **the generalisation
                  number, and the only one worth quoting**

  The third split exists because the second one was not enough: it reuses the
  exact phrases that carry each label, so a model can score 1.00 on it by
  memorising phrase -> label. test_hard paraphrases every signal-carrying phrase.

Metrics, and why each exists:

  json_lenient    a {...} could be extracted at all (a chatty wrapper is fine)
  json_strict     output was exactly the four expected keys, nothing else
  exact_match     every one of the four fields correct. The headline.
  field accuracy  per field, counted wrong when the output did not parse
  acc | parsed    per field, over parsed rows only. **This separates "learned
                  to emit JSON" from "learned the task"** -- the distinction a
                  single accuracy number hides completely.
  prompt_tokens   median prompt length. The cost axis few-shot loses on.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prompts import FIELDS, build_messages  # noqa: E402
from src.scoring import score  # noqa: E402  (pure maths; see src/scoring.py)

MAX_NEW_TOKENS = 64
FEW_SHOT_K = 4


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(model_name: str, adapter: Path | None, device: str):
    dtype = torch.bfloat16 if device in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        # Fold the adapter into the base weights: inference then costs exactly
        # what the base model costs, which is half the argument for LoRA.
        model = model.merge_and_unload()
    model.to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, messages_batch: list[list[dict]], device: str) -> tuple[list[str], list[int]]:
    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    # Left padding so every sequence's generation starts at the same offset --
    # right padding makes a batched decoder generate from inside the padding.
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    tokenizer.padding_side = original_side

    output = model.generate(
        **encoded,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,          # greedy: the eval must be reproducible
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output[:, encoded["input_ids"].shape[1] :]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    lengths = encoded["attention_mask"].sum(dim=1).tolist()
    return texts, lengths


def run_condition(
    name: str,
    rows: list[dict],
    model,
    tokenizer,
    device: str,
    *,
    few_shot: list[dict] | None,
    batch_size: int,
) -> dict:
    predictions: list[str] = []
    prompt_lengths: list[int] = []
    started = time.perf_counter()

    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        messages = [build_messages(r["ticket"], few_shot=few_shot) for r in chunk]
        texts, lengths = generate(model, tokenizer, messages, device)
        predictions.extend(texts)
        prompt_lengths.extend(lengths)
        print(f"    {min(start + batch_size, len(rows))}/{len(rows)}", end="\r", flush=True)

    result = score(rows, predictions, prompt_lengths)
    result["condition"] = name
    result["eval_seconds"] = round(time.perf_counter() - started, 1)
    print(f"    {len(rows)}/{len(rows)} done in {result['eval_seconds']}s")
    return result


def print_table(results: dict[str, dict[str, dict]]) -> None:
    for split, by_condition in results.items():
        print(f"\n=== {split} ===")
        header = (
            f"{'condition':<16}{'n':>4}{'json_len':>10}{'json_str':>10}"
            f"{'exact':>8}{'cat':>7}{'sev':>7}{'plan':>7}{'esc':>7}{'prompt_tok':>12}"
        )
        print(header)
        print("-" * len(header))
        for name, r in by_condition.items():
            fa = r["field_accuracy"]
            print(
                f"{name:<16}{r['n']:>4}{r['json_lenient']:>10.2f}{r['json_strict']:>10.2f}"
                f"{r['exact_match']:>8.2f}{fa['category']:>7.2f}{fa['severity']:>7.2f}"
                f"{fa['plan']:>7.2f}{fa['escalate']:>7.2f}{r['median_prompt_tokens']:>12}"
            )

        print("\n  predicted class distribution (collapse detector):")
        for name, r in by_condition.items():
            summary = []
            for field in FIELDS:
                top, count = next(iter(r["predicted_distribution"][field].items()))
                summary.append(f"{field}: {top} {count}/{r['n']}")
            print(f"    {name:<16} most-predicted -> " + "  ".join(summary))

        print("\n  accuracy given the output parsed (format vs task):")
        for name, r in by_condition.items():
            fa = r["field_accuracy_given_parsed"]
            print(
                f"    {name:<16} parsed={r['parsed_rows']:>3}/{r['n']:<4} "
                + "  ".join(f"{f}={fa[f]:.2f}" for f in FIELDS)
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", type=Path, default=ROOT / "adapters" / "qwen05b-triage-lora")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["base_zeroshot", "base_fewshot", "lora"],
        choices=["base_zeroshot", "base_fewshot", "lora"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="truncate each split (smoke test)")
    parser.add_argument("--out", type=Path, default=ROOT / "evals" / "results.json")
    args = parser.parse_args()

    device = pick_device()
    print(f"device={device}")

    splits = {
        "test_seen": read_jsonl(ROOT / "data" / "test_seen.jsonl"),
        "test_unseen": read_jsonl(ROOT / "data" / "test_unseen.jsonl"),
        "test_hard": read_jsonl(ROOT / "data" / "test_hard.jsonl"),
    }
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}

    # Few-shot examples come from TRAIN, never from a test split, and are fixed
    # by seed so the condition is reproducible.
    train = read_jsonl(ROOT / "data" / "train.jsonl")
    few_shot = random.Random(7).sample(train, FEW_SHOT_K)

    results: dict[str, dict[str, dict]] = {split: {} for split in splits}

    for condition in args.conditions:
        adapter = args.adapter if condition == "lora" else None
        if adapter is not None and not adapter.exists():
            print(f"skipping {condition}: no adapter at {adapter}")
            continue

        print(f"\n[{condition}] loading…")
        model, tokenizer = load(args.model, adapter, device)
        shots = few_shot if condition == "base_fewshot" else None

        for split, rows in splits.items():
            print(f"  {split}:")
            results[split][condition] = run_condition(
                condition, rows, model, tokenizer, device,
                few_shot=shots, batch_size=args.batch_size,
            )

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    print_table(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "device": device, "splits": results}, indent=2))
    try:
        shown = args.out.resolve().relative_to(ROOT)
    except ValueError:
        shown = args.out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
