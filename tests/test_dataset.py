"""The dataset, the policy rule, and the split that makes the eval mean anything."""

import json
from pathlib import Path

import pytest

from src.make_dataset import (
    CATEGORIES,
    HOLDOUT_TEMPLATES,
    PLANS,
    SEVERITIES,
    TRAIN_TEMPLATES,
    generate,
    should_escalate,
)

ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> list[dict]:
    path = ROOT / "data" / name
    if not path.exists():
        pytest.skip(f"{name} not generated; run python src/make_dataset.py")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# the policy the model has to infer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "category,severity,expected",
    [
        ("ingestion", "sev1", True),
        ("api", "sev2", True),
        ("ingestion", "sev3", False),   # right category, severity too low
        ("billing", "sev1", False),     # severe, but support owns billing
        ("dashboards", "sev2", False),
        ("how_to", "sev1", False),
    ],
)
def test_escalation_policy(category, severity, expected):
    assert should_escalate(category, severity) is expected


def test_escalate_is_not_derivable_from_one_field_alone():
    """If it were, the field would be measuring extraction, not the policy."""
    assert should_escalate("ingestion", "sev1") != should_escalate("ingestion", "sev3")
    assert should_escalate("ingestion", "sev1") != should_escalate("billing", "sev1")


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def test_generation_is_deterministic_for_a_seed():
    assert generate(20, TRAIN_TEMPLATES, 1) == generate(20, TRAIN_TEMPLATES, 1)


def test_generation_produces_unique_tickets():
    rows = generate(100, TRAIN_TEMPLATES, 3)
    assert len({r["ticket"] for r in rows}) == 100


def test_labels_are_always_inside_the_label_space():
    for row in generate(60, TRAIN_TEMPLATES, 4):
        label = row["label"]
        assert label["category"] in CATEGORIES
        assert label["severity"] in SEVERITIES
        assert label["plan"] in PLANS
        assert isinstance(label["escalate"], bool)


def test_the_escalate_label_matches_the_policy_on_every_row():
    for row in generate(100, TRAIN_TEMPLATES, 5):
        label = row["label"]
        assert label["escalate"] == should_escalate(label["category"], label["severity"])


# --------------------------------------------------------------------------- #
# the split — this is the part that makes the eval honest
# --------------------------------------------------------------------------- #
def test_train_and_holdout_template_families_are_disjoint():
    assert not ({f for f, _ in TRAIN_TEMPLATES} & {f for f, _ in HOLDOUT_TEMPLATES})


def test_generated_files_have_no_shared_tickets():
    train = {r["ticket"] for r in read("train.jsonl")}
    for name in ("test_seen.jsonl", "test_unseen.jsonl"):
        assert not (train & {r["ticket"] for r in read(name)}), f"leak into {name}"


def test_test_seen_uses_only_training_template_families():
    families = {f for f, _ in TRAIN_TEMPLATES}
    assert {r["template_family"] for r in read("test_seen.jsonl")} <= families


def test_test_unseen_uses_only_holdout_template_families():
    """The generalisation claim rests entirely on this."""
    families = {f for f, _ in HOLDOUT_TEMPLATES}
    assert {r["template_family"] for r in read("test_unseen.jsonl")} <= families


def test_every_label_value_appears_in_the_training_set():
    """A value absent from training cannot be learned, and its test rows would
    silently measure nothing."""
    train = read("train.jsonl")
    for field, space in (("category", CATEGORIES), ("severity", SEVERITIES), ("plan", PLANS)):
        assert {r["label"][field] for r in train} == set(space)


# --------------------------------------------------------------------------- #
# test_hard — the split that exposed the artifact
#
# The original holdout varied only the template skeleton and reused the exact
# phrases carrying each label, so the adapter scored 1.00 by memorising
# phrase -> label. These tests pin the fix so it cannot silently regress.
# --------------------------------------------------------------------------- #
from src.make_dataset import (  # noqa: E402
    CATEGORY_PHRASES,
    HOLDOUT_CATEGORY_PHRASES,
    HOLDOUT_PLAN_PHRASES,
    HOLDOUT_SEVERITY_PHRASES,
    PLAN_PHRASES,
    SEVERITY_PHRASES,
)


def all_phrases(*pools) -> set[str]:
    return {p for pool in pools for plist in pool.values() for p in plist}


def test_holdout_phrase_pools_share_no_phrase_with_training():
    train = all_phrases(CATEGORY_PHRASES, SEVERITY_PHRASES, PLAN_PHRASES)
    holdout = all_phrases(
        HOLDOUT_CATEGORY_PHRASES, HOLDOUT_SEVERITY_PHRASES, HOLDOUT_PLAN_PHRASES
    )
    assert not (train & holdout)


def test_holdout_pools_cover_every_label_value():
    assert set(HOLDOUT_CATEGORY_PHRASES) == set(CATEGORIES)
    assert set(HOLDOUT_SEVERITY_PHRASES) == set(SEVERITIES)
    assert set(HOLDOUT_PLAN_PHRASES) == set(PLANS)


def test_test_hard_contains_no_training_phrase():
    """The assertion the generator makes, re-made against the written file."""
    rows = read("test_hard.jsonl")
    for phrase in all_phrases(CATEGORY_PHRASES, SEVERITY_PHRASES, PLAN_PHRASES):
        assert not any(phrase.lower() in r["ticket"].lower() for r in rows), phrase


def test_test_hard_uses_holdout_templates_only():
    families = {f for f, _ in HOLDOUT_TEMPLATES}
    assert {r["template_family"] for r in read("test_hard.jsonl")} <= families


def test_test_unseen_still_reuses_training_phrases_on_purpose():
    """test_unseen isolates sentence structure, so it SHOULD reuse phrases.
    If this ever fails, the two splits have stopped measuring different things."""
    rows = read("test_unseen.jsonl")
    train_phrases = all_phrases(CATEGORY_PHRASES, SEVERITY_PHRASES, PLAN_PHRASES)
    assert any(
        phrase.lower() in r["ticket"].lower() for phrase in train_phrases for r in rows
    )


def test_test_hard_labels_still_obey_the_escalation_policy():
    for row in read("test_hard.jsonl"):
        label = row["label"]
        assert label["escalate"] == should_escalate(label["category"], label["severity"])
