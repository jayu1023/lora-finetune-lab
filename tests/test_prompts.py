"""Prompt construction and output parsing.

These matter more than they look. Training and evaluation share this module on
purpose — the classic way to get a meaningless fine-tuning number is to train on
one prompt format and evaluate on another, and the only defence is one code path
plus tests that pin it.
"""

import json

from src.prompts import (
    FIELDS,
    build_messages,
    is_well_formed,
    normalise,
    parse_prediction,
    target_json,
)

LABEL = {"category": "ingestion", "severity": "sev1", "plan": "enterprise", "escalate": True}


def test_target_json_is_compact_and_key_ordered():
    """Byte-identical targets for identical labels. Default separators would
    teach the model to spend tokens on whitespace."""
    assert target_json(LABEL) == (
        '{"category":"ingestion","severity":"sev1","plan":"enterprise","escalate":true}'
    )


def test_target_json_key_order_is_stable_regardless_of_input_order():
    shuffled = {"escalate": True, "plan": "enterprise", "severity": "sev1", "category": "ingestion"}
    assert target_json(shuffled) == target_json(LABEL)


def test_target_json_is_parseable_and_round_trips():
    assert json.loads(target_json(LABEL)) == LABEL


def test_zero_shot_messages_are_system_then_user():
    messages = build_messages("events stopped arriving")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "events stopped arriving" in messages[-1]["content"]


def test_few_shot_messages_interleave_examples_before_the_real_question():
    shots = [{"ticket": "a", "label": LABEL}, {"ticket": "b", "label": LABEL}]
    messages = build_messages("real one", few_shot=shots)
    assert [m["role"] for m in messages] == [
        "system", "user", "assistant", "user", "assistant", "user",
    ]
    assert messages[-1]["content"].endswith("real one")
    # The examples must show the exact target format the model is trained on.
    assert messages[2]["content"] == target_json(LABEL)


def test_the_system_prompt_states_the_label_space_and_the_policy():
    system = build_messages("x")[0]["content"]
    for value in ("ingestion", "sev1", "self_serve"):
        assert value in system
    assert "Escalate only" in system


def test_few_shot_and_zero_shot_share_the_same_system_prompt():
    """Otherwise the comparison measures two different prompts, not fine-tuning."""
    assert build_messages("x")[0] == build_messages("x", few_shot=[{"ticket": "a", "label": LABEL}])[0]


# --------------------------------------------------------------------------- #
# parsing: lenient on purpose
# --------------------------------------------------------------------------- #
def test_parses_bare_json():
    assert parse_prediction(target_json(LABEL)) == LABEL


def test_parses_json_wrapped_in_chat():
    """The baseline must be measured at its best. A strict parser would make the
    base model look worse than it is and inflate the fine-tuning delta."""
    assert parse_prediction(f"Sure! Here you go:\n{target_json(LABEL)}\nHope that helps.") == LABEL


def test_parses_a_fenced_code_block():
    assert parse_prediction(f"```json\n{target_json(LABEL)}\n```") == LABEL


def test_returns_none_for_prose_with_no_json():
    assert parse_prediction("This looks like an ingestion problem, probably sev1.") is None


def test_returns_none_for_malformed_json():
    assert parse_prediction('{"category": "ingestion", "severity":}') is None


def test_returns_none_for_a_json_array():
    assert parse_prediction('["ingestion"]') is None


# --------------------------------------------------------------------------- #
# normalising: generous, so the parser is not what is being scored
# --------------------------------------------------------------------------- #
def test_normalise_lowercases_and_underscores_enum_values():
    assert normalise({"category": "Ingestion", "severity": "SEV1", "plan": "self serve"}) == {
        "category": "ingestion", "severity": "sev1", "plan": "self_serve", "escalate": None,
    }


def test_normalise_accepts_stringified_booleans():
    assert normalise({"escalate": "True"})["escalate"] is True
    assert normalise({"escalate": "false"})["escalate"] is False


def test_normalise_rejects_a_non_boolean_escalate():
    assert normalise({"escalate": "maybe"})["escalate"] is None
    assert normalise({"escalate": 1})["escalate"] is None


def test_normalise_always_returns_every_field():
    assert set(normalise({})) == set(FIELDS)


def test_well_formed_requires_exactly_the_four_keys():
    assert is_well_formed(LABEL)
    assert not is_well_formed({**LABEL, "extra": 1})
    assert not is_well_formed({"category": "api"})
    assert not is_well_formed(None)
