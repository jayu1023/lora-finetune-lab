#!/usr/bin/env python
"""Generate the structured-extraction dataset.

Task: a free-text support ticket in, one strict JSON object out.

    {"category": "...", "severity": "...", "plan": "...", "escalate": true|false}

Three of those four fields are *extraction* -- the signal is somewhere in the
text. The fourth, `escalate`, is never stated: it is a policy inference from
severity and category. Keeping both kinds of field in one schema is deliberate,
because it lets the eval separate "the model learned to emit JSON" from "the
model learned the task".

Three held-out splits, which is one more than this started with:

  test_seen     training templates, training phrases
  test_unseen   HELD-OUT templates, training phrases  -- isolates sentence structure
  test_hard     HELD-OUT templates AND held-out phrases -- the generalisation number

The third exists because the first version of this file was wrong. `test_unseen`
reuses the exact phrases that carry each label, so the adapter scored 1.00 on it
by memorising phrase -> label. An audit found 100% phrase overlap. `test_hard`
paraphrases every signal-carrying phrase and asserts zero overlap, so a correct
answer requires mapping meaning to a label rather than recalling a string.

**The data is synthetic, and that is the single biggest limitation of this repo.**
It is generated from phrasing templates, so it is cleaner and more regular than
real tickets. The mitigation is the split: the held-out set is built from
**template families the model never saw**, so a model that memorised phrasing
scores badly on it. Reporting only a random split would have hidden exactly that.

    python src/make_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"

CATEGORIES = ["billing", "ingestion", "api", "dashboards", "how_to"]
SEVERITIES = ["sev1", "sev2", "sev3"]
PLANS = ["enterprise", "business", "self_serve"]

# --------------------------------------------------------------------------- #
# the policy the model has to infer -- it is never written in the ticket text
# --------------------------------------------------------------------------- #
ESCALATE_CATEGORIES = {"ingestion", "api"}


def should_escalate(category: str, severity: str) -> bool:
    """Northwind's rule: only sev1/sev2 may be escalated, and only for the two
    categories engineering owns. Everything else is handled in support."""
    return severity in {"sev1", "sev2"} and category in ESCALATE_CATEGORIES


# --------------------------------------------------------------------------- #
# surface forms -- the signal the extraction fields come from
# --------------------------------------------------------------------------- #
CATEGORY_PHRASES = {
    "billing": [
        "the invoice total looks wrong",
        "we were charged twice this month",
        "the seat count on our bill does not match",
        "a credit from last cycle never appeared",
        "the proration on our upgrade looks off",
    ],
    "ingestion": [
        "events stopped arriving in the pipeline",
        "our event stream is dropping records",
        "the ingest endpoint is rejecting our payloads",
        "roughly a third of our events never land",
        "the schema registry is refusing valid events",
    ],
    "api": [
        "the REST endpoint returns 500 intermittently",
        "our API calls are timing out",
        "pagination is returning duplicate rows",
        "the SDK is getting 429s well under our quota",
        "the webhook signature check keeps failing",
    ],
    "dashboards": [
        "a saved dashboard will not load",
        "the chart renders blank for last week",
        "dashboard numbers do not match the export",
        "a shared dashboard link 404s for my team",
        "the date filter resets on every refresh",
    ],
    "how_to": [
        "we cannot work out how to invite a teammate",
        "where do we configure a custom field",
        "how do we export a report on a schedule",
        "is there a way to rename a workspace",
        "how do we rotate our API key",
    ],
}

SEVERITY_PHRASES = {
    "sev1": [
        "everything is down for all of our users",
        "this is a complete outage and we are losing data",
        "nothing works at all and it is affecting every customer we have",
        "production is fully broken right now",
    ],
    "sev2": [
        "it is broken for about half our team",
        "a major feature is unusable for a subset of our users",
        "we have a workaround but it is painful",
        "it is badly degraded but not completely down",
    ],
    "sev3": [
        "it is a minor annoyance, nothing urgent",
        "low priority, whenever you get to it",
        "not blocking anyone, just untidy",
        "it is cosmetic as far as we can tell",
    ],
}

PLAN_PHRASES = {
    "enterprise": [
        "we are on the Enterprise plan with 24/7 coverage",
        "our Enterprise contract is up for renewal shortly",
        "as an Enterprise customer with a dedicated account team",
        "we pay for Enterprise support",
    ],
    "business": [
        "we are on the Business tier",
        "our Business plan renews monthly",
        "we are a Business plan customer",
        "we upgraded to Business last quarter",
    ],
    "self_serve": [
        "we signed up with a credit card last week",
        "we are on the free-to-start self-serve plan",
        "we are just a small self-serve account",
        "we pay month to month on the self-serve tier",
    ],
}

# --------------------------------------------------------------------------- #
# template families. `train` families are what the model sees; `holdout`
# families are structurally different phrasings it never sees.
# --------------------------------------------------------------------------- #
TRAIN_TEMPLATES: list[tuple[str, str]] = [
    ("direct", "Hi — {category}. {severity}. For context, {plan}."),
    ("urgent_first", "{severity_cap}! {category}. {plan}."),
    ("plan_first", "{plan_cap}. We have an issue: {category}. {severity}."),
    ("report", "Ticket: {category}\nImpact: {severity}\nAccount: {plan}"),
    ("rambling", "Not sure if this is the right place, but {category}. "
                 "To be clear {severity}. {plan_cap}, if that matters."),
    ("apologetic", "Sorry to bother you. {category_cap}. {severity_cap}. {plan_cap}."),
    ("terse", "{category_cap}. {severity_cap}. {plan_cap}."),
]

# --------------------------------------------------------------------------- #
# Held-out PHRASE pools: paraphrases of the same labels, sharing as little
# vocabulary with the training phrases as English reasonably allows.
#
# These exist because the first version of this dataset was wrong. The holdout
# split varied only the template *skeleton* while reusing the exact phrases that
# carry the label -- so the adapter scored 1.00 on it by memorising
# phrase -> label, and the "generalisation" number measured nothing. An audit
# showed 100% of holdout phrases had been seen in training.
#
# `test_hard` uses held-out templates AND held-out phrases, so a correct answer
# requires actually mapping meaning to a label.
# --------------------------------------------------------------------------- #
HOLDOUT_CATEGORY_PHRASES = {
    "billing": [
        "there is a discrepancy between the statement and what we expected to pay",
        "our finance team is disputing a line item on the latest receipt",
        "the amount deducted from our card does not reconcile",
    ],
    "ingestion": [
        "telemetry we publish upstream is not showing up downstream",
        "the collector appears to be discarding a portion of what we emit",
        "our batches are being refused at the intake layer",
    ],
    "api": [
        "programmatic requests from our service are erroring out sporadically",
        "the client library raises throttling errors even when we are well under budget",
        "signature verification on our callbacks fails every time",
    ],
    "dashboards": [
        "one of our saved visualisations refuses to display anything",
        "the figures shown in the UI disagree with the downloaded file",
        "a tile we built last month now comes up empty",
    ],
    "how_to": [
        "we are struggling to find the setting that controls this",
        "could you point us at the documentation for adding a colleague",
        "we are unsure of the correct procedure for scheduling an extract",
    ],
}

HOLDOUT_SEVERITY_PHRASES = {
    "sev1": [
        "nobody in the organisation can do anything at the moment",
        "we are entirely offline and records appear to be disappearing",
        "this has taken out the whole business",
    ],
    "sev2": [
        "a significant slice of the department is blocked",
        "an important workflow is unusable although others still function",
        "we can limp along with a manual process but it hurts",
    ],
    "sev3": [
        "there is no rush on this one whatsoever",
        "purely a tidiness matter, nobody is held up",
        "trivial, and it affects nothing we rely on",
    ],
}

HOLDOUT_PLAN_PHRASES = {
    "enterprise": [
        "our agreement includes round-the-clock coverage and a named contact",
        "we sit on the top commercial tier with an annual commitment",
        "we hold the premium contract with the dedicated success manager",
    ],
    "business": [
        "we sit on the mid commercial tier",
        "our subscription is the middle of the three paid options",
        "we moved off the entry tier to the one above it last year",
    ],
    "self_serve": [
        "we onboarded ourselves online with no sales contact",
        "ours is a small pay-as-you-go account opened through the website",
        "we are on the entry-level plan we signed up for unaided",
    ],
}

HOLDOUT_TEMPLATES: list[tuple[str, str]] = [
    # Different structure: a question, a quoted complaint, and a thread reply.
    ("question", "Can someone look at this? {category_cap} — and to be clear, "
                 "{severity}. We are the account where {plan}."),
    ("forwarded", "Forwarding from our ops channel:\n\n> {category_cap}\n> {severity_cap}\n\n"
                  "Adding that {plan}."),
    ("followup", "Following up on my earlier message. Still the same problem: {category}. "
                 "Since then {severity}. (Account detail: {plan}.)"),
]


def capitalise(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def render(template: str, category_phrase: str, severity_phrase: str, plan_phrase: str) -> str:
    return template.format(
        category=category_phrase,
        category_cap=capitalise(category_phrase),
        severity=severity_phrase,
        severity_cap=capitalise(severity_phrase),
        plan=plan_phrase,
        plan_cap=capitalise(plan_phrase),
    )


def make_example(
    rng: random.Random,
    templates: list[tuple[str, str]],
    *,
    category_pool: dict | None = None,
    severity_pool: dict | None = None,
    plan_pool: dict | None = None,
) -> dict:
    category_pool = category_pool or CATEGORY_PHRASES
    severity_pool = severity_pool or SEVERITY_PHRASES
    plan_pool = plan_pool or PLAN_PHRASES

    category = rng.choice(CATEGORIES)
    severity = rng.choice(SEVERITIES)
    plan = rng.choice(PLANS)
    family, template = rng.choice(templates)

    text = render(
        template,
        rng.choice(category_pool[category]),
        rng.choice(severity_pool[severity]),
        rng.choice(plan_pool[plan]),
    )
    return {
        "ticket": text,
        "template_family": family,
        "label": {
            "category": category,
            "severity": severity,
            "plan": plan,
            "escalate": should_escalate(category, severity),
        },
    }


def generate(n: int, templates: list[tuple[str, str]], seed: int, **pools) -> list[dict]:
    """Deduplicated on ticket text, so a train/test leak cannot happen by
    collision -- with a small phrase pool, identical renders are likely."""
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[dict] = []
    attempts = 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        example = make_example(rng, templates, **pools)
        if example["ticket"] in seen:
            continue
        seen.add(example["ticket"])
        out.append(example)
    if len(out) < n:
        raise SystemExit(f"could only generate {len(out)} unique examples of {n}")
    return out


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"  {path.relative_to(ROOT)}: {len(rows)} rows")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=600)
    parser.add_argument("--test-seen", type=int, default=120)
    parser.add_argument("--test-unseen", type=int, default=120)
    parser.add_argument("--test-hard", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # Separate seeds per split, then an explicit overlap check -- a leak here
    # would make every number in the README meaningless.
    train = generate(args.train, TRAIN_TEMPLATES, args.seed)
    test_seen = generate(args.test_seen, TRAIN_TEMPLATES, args.seed + 1)
    test_unseen = generate(args.test_unseen, HOLDOUT_TEMPLATES, args.seed + 2)
    test_hard = generate(
        args.test_hard,
        HOLDOUT_TEMPLATES,
        args.seed + 3,
        category_pool=HOLDOUT_CATEGORY_PHRASES,
        severity_pool=HOLDOUT_SEVERITY_PHRASES,
        plan_pool=HOLDOUT_PLAN_PHRASES,
    )

    train_texts = {r["ticket"] for r in train}
    test_seen = [r for r in test_seen if r["ticket"] not in train_texts]
    overlap = train_texts & {r["ticket"] for r in test_seen}
    assert not overlap, f"train/test leak: {len(overlap)} shared tickets"
    assert not (train_texts & {r["ticket"] for r in test_unseen})
    assert not (train_texts & {r["ticket"] for r in test_hard})

    # Phrase-level audit. test_unseen deliberately REUSES training phrases (it
    # isolates the effect of an unfamiliar sentence structure); test_hard must
    # not, or it is not measuring generalisation either.
    def phrase_overlap(rows: list[dict], pools: list[dict]) -> float:
        train_phrases = {
            phrase for pool in (CATEGORY_PHRASES, SEVERITY_PHRASES, PLAN_PHRASES)
            for plist in pool.values() for phrase in plist
        }
        used = {
            phrase for pool in pools for plist in pool.values() for phrase in plist
            if any(phrase.lower() in r["ticket"].lower() for r in rows)
        }
        if not used:
            return 0.0
        return len(used & train_phrases) / len(used)

    hard_overlap = phrase_overlap(
        test_hard,
        [HOLDOUT_CATEGORY_PHRASES, HOLDOUT_SEVERITY_PHRASES, HOLDOUT_PLAN_PHRASES],
    )
    assert hard_overlap == 0.0, f"test_hard reuses {hard_overlap:.0%} of training phrases"

    print("wrote:")
    write(OUT / "train.jsonl", train)
    write(OUT / "test_seen.jsonl", test_seen)
    write(OUT / "test_unseen.jsonl", test_unseen)
    write(OUT / "test_hard.jsonl", test_hard)

    escalate_rate = sum(r["label"]["escalate"] for r in train) / len(train)
    print(f"\nlabel balance (train): escalate=True on {escalate_rate:.1%} of rows")
    print("majority-class baselines (what a model must beat to mean anything):")
    for field in ("category", "severity", "plan", "escalate"):
        counts: dict = {}
        for row in train:
            value = row["label"][field]
            counts[value] = counts.get(value, 0) + 1
        best = max(counts.values()) / len(train)
        print(f"  {field:<9} always-predict-most-common = {best:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
