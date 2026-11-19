"""Prompt construction and output parsing.

One module, because the single most common way to get a meaningless fine-tuning
result is to train on one prompt format and evaluate on another. Training and
every eval condition call the same functions here.
"""

from __future__ import annotations

import json
import re
from typing import Any

FIELDS = ("category", "severity", "plan", "escalate")

CATEGORIES = ["billing", "ingestion", "api", "dashboards", "how_to"]
SEVERITIES = ["sev1", "sev2", "sev3"]
PLANS = ["enterprise", "business", "self_serve"]

SYSTEM = (
    "You triage support tickets. Reply with one JSON object and nothing else, "
    "with exactly these keys:\n"
    f'  "category": one of {CATEGORIES}\n'
    f'  "severity": one of {SEVERITIES}\n'
    f'  "plan":     one of {PLANS}\n'
    '  "escalate": true or false\n'
    "Escalate only sev1 or sev2 tickets in the ingestion or api categories."
)


def target_json(label: dict[str, Any]) -> str:
    """The exact string the model is trained to produce.

    Key order is fixed and separators are compact, so the training target is
    byte-identical for identical labels. Letting `json.dumps` default to
    `", "` separators would teach the model to spend tokens on whitespace.
    """
    return json.dumps({k: label[k] for k in FIELDS}, separators=(",", ":"))


def user_message(ticket: str) -> str:
    return f"Ticket:\n{ticket}"


def build_messages(
    ticket: str,
    *,
    few_shot: list[dict] | None = None,
) -> list[dict[str, str]]:
    """Chat messages for one ticket.

    `few_shot` turns this into the prompted baseline: the same instruction and
    the same target format, with k worked examples as prior turns. Identical in
    every other respect to the fine-tuned path, so the comparison isolates
    fine-tuning rather than measuring two different prompts.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM}]
    for example in few_shot or []:
        messages.append({"role": "user", "content": user_message(example["ticket"])})
        messages.append({"role": "assistant", "content": target_json(example["label"])})
    messages.append({"role": "user", "content": user_message(ticket)})
    return messages


_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def parse_prediction(text: str) -> dict[str, Any] | None:
    """Extract a prediction from raw model output. Returns None if unusable.

    This is deliberately **lenient**: it pulls the first {...} out of whatever
    the model said, so a chatty "Sure! Here you go: {...}" still counts as a
    parse. A strict parser would make the base model look worse than it is and
    inflate the fine-tuning result -- the honest comparison needs the baseline
    measured at its best.

    `json_strict` in the eval reports the stricter number alongside it.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    candidate = text if text.startswith("{") else None
    if candidate is None:
        match = _JSON_OBJECT.search(text)
        candidate = match.group(0) if match else None
    if candidate is None:
        return None

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def normalise(prediction: dict[str, Any]) -> dict[str, Any]:
    """Coerce a parsed prediction toward the label space.

    Also generous on purpose: "Sev1" and "SEV1" count as sev1, and the strings
    "true"/"false" count as booleans. Scoring a correct answer wrong because of
    capitalisation measures the parser, not the model.
    """
    out: dict[str, Any] = {}
    for field in FIELDS:
        value = prediction.get(field)
        if field == "escalate":
            if isinstance(value, bool):
                out[field] = value
            elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                out[field] = value.strip().lower() == "true"
            else:
                out[field] = None
        else:
            out[field] = value.strip().lower().replace(" ", "_") if isinstance(value, str) else None
    return out


def is_well_formed(prediction: dict[str, Any] | None) -> bool:
    """Strict check: exactly the four keys, no extras, no missing."""
    return prediction is not None and set(prediction) == set(FIELDS)
