"""Scoring maths. Deliberately free of torch, transformers and peft.

This lives apart from `evaluate.py` for one concrete reason: CI installs only
pytest, because downloading ~200MB of torch to test arithmetic is a waste of a
CI minute and the training/eval path needs a GPU to be worth running anyway.
`evaluate.py` imports torch at module level and uses `@torch.inference_mode()`
as a decorator, so a lazy import inside the functions would not have helped —
the module cannot be imported at all without torch present.

So the pure maths lives here, the model plumbing lives there, and
`tests/test_scoring.py` imports from here. If these metrics are wrong, every
number in the README is wrong, which is why they are the part that is tested
on every push.
"""

from __future__ import annotations

import statistics

from src.prompts import FIELDS, is_well_formed, normalise, parse_prediction


def score(rows: list[dict], predictions: list[str], prompt_lengths: list[int]) -> dict:
    n = len(rows)
    lenient = strict = exact = 0
    field_correct = {f: 0 for f in FIELDS}
    field_correct_parsed = {f: 0 for f in FIELDS}
    parsed_count = 0
    failures: list[dict] = []

    for row, raw, _ in zip(rows, predictions, prompt_lengths):
        label = row["label"]
        parsed = parse_prediction(raw)

        if parsed is not None:
            lenient += 1
            parsed_count += 1
            if is_well_formed(parsed):
                strict += 1
            prediction = normalise(parsed)
        else:
            prediction = {f: None for f in FIELDS}

        hits = {f: prediction[f] == label[f] for f in FIELDS}
        for field, hit in hits.items():
            field_correct[field] += int(hit)
            if parsed is not None:
                field_correct_parsed[field] += int(hit)

        if all(hits.values()):
            exact += 1
        elif len(failures) < 8:
            failures.append(
                {
                    "ticket": row["ticket"][:140].replace("\n", " "),
                    "expected": label,
                    "raw_output": raw.strip()[:160],
                    "wrong_fields": [f for f, hit in hits.items() if not hit],
                }
            )

    # Predicted-class distribution per field. This is the diagnostic that
    # explains a number rather than just reporting it: a field whose accuracy
    # sits near its majority-class baseline is usually a model that has
    # collapsed to one answer, and only the distribution shows that.
    distribution: dict[str, dict] = {f: {} for f in FIELDS}
    for raw in predictions:
        parsed = parse_prediction(raw)
        prediction = normalise(parsed) if parsed is not None else {f: None for f in FIELDS}
        for field in FIELDS:
            key = str(prediction[field])
            distribution[field][key] = distribution[field].get(key, 0) + 1
    for field in FIELDS:
        distribution[field] = dict(
            sorted(distribution[field].items(), key=lambda kv: kv[1], reverse=True)
        )

    denominator = max(parsed_count, 1)
    return {
        "predicted_distribution": distribution,
        "n": n,
        "json_lenient": lenient / n,
        "json_strict": strict / n,
        "exact_match": exact / n,
        "field_accuracy": {f: field_correct[f] / n for f in FIELDS},
        "field_accuracy_given_parsed": {f: field_correct_parsed[f] / denominator for f in FIELDS},
        "parsed_rows": parsed_count,
        "median_prompt_tokens": int(statistics.median(prompt_lengths)),
        "failures": failures,
    }
