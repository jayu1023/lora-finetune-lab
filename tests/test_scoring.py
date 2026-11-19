"""The metrics. If these are wrong, every number in the README is wrong."""

from src.evaluate import score
from src.prompts import target_json

LABEL = {"category": "api", "severity": "sev2", "plan": "business", "escalate": True}
ROW = {"ticket": "the api is timing out for half our team", "label": LABEL}


def run(outputs: list[str], rows=None):
    rows = rows or [ROW] * len(outputs)
    return score(rows, outputs, [100] * len(outputs))


def test_a_perfect_prediction_scores_one_everywhere():
    result = run([target_json(LABEL)])
    assert result["exact_match"] == 1.0
    assert result["json_strict"] == 1.0
    assert all(v == 1.0 for v in result["field_accuracy"].values())


def test_prose_scores_zero_and_counts_as_unparsed():
    result = run(["I think this is an api issue."])
    assert result["json_lenient"] == 0.0
    assert result["exact_match"] == 0.0
    assert result["parsed_rows"] == 0
    assert all(v == 0.0 for v in result["field_accuracy"].values())


def test_one_wrong_field_fails_exact_match_but_not_the_others():
    result = run([target_json({**LABEL, "severity": "sev3"})])
    assert result["exact_match"] == 0.0
    assert result["field_accuracy"]["severity"] == 0.0
    assert result["field_accuracy"]["category"] == 1.0


def test_extra_keys_pass_lenient_but_fail_strict():
    import json

    result = run([json.dumps({**LABEL, "confidence": 0.9})])
    assert result["json_lenient"] == 1.0
    assert result["json_strict"] == 0.0
    # The four real fields are still right, so exact_match holds.
    assert result["exact_match"] == 1.0


def test_field_accuracy_given_parsed_ignores_unparsed_rows():
    """The whole point of this metric: it separates "learned to emit JSON" from
    "learned the task". Here one row is perfect and one is prose — overall
    accuracy is 0.5, but accuracy among parsed rows is 1.0."""
    result = run([target_json(LABEL), "no json here at all"])
    assert result["field_accuracy"]["category"] == 0.5
    assert result["field_accuracy_given_parsed"]["category"] == 1.0
    assert result["parsed_rows"] == 1


def test_given_parsed_does_not_divide_by_zero_when_nothing_parsed():
    result = run(["prose", "more prose"])
    assert result["field_accuracy_given_parsed"]["category"] == 0.0


def test_capitalised_values_still_count_as_correct():
    import json

    result = run([json.dumps({"category": "API", "severity": "Sev2", "plan": "Business", "escalate": "true"})])
    assert result["exact_match"] == 1.0


def test_failures_are_captured_for_inspection_and_capped():
    result = run(["prose"] * 20)
    assert len(result["failures"]) == 8
    assert result["failures"][0]["wrong_fields"]


def test_a_correct_prediction_records_no_failure():
    assert run([target_json(LABEL)])["failures"] == []


# --------------------------------------------------------------------------- #
# the collapse detector
#
# An accuracy near a majority-class baseline and a collapsed prediction
# distribution are the same number with completely different meanings. This is
# the diagnostic that told us `plan` had collapsed to self_serve on 115/120 rows
# rather than being partially correct.
# --------------------------------------------------------------------------- #
def test_distribution_reveals_a_collapsed_prediction():
    import json

    constant = json.dumps({**LABEL, "plan": "self_serve"})
    result = run([constant] * 10)
    assert result["predicted_distribution"]["plan"] == {"self_serve": 10}


def test_distribution_is_sorted_most_predicted_first():
    import json

    outputs = [json.dumps({**LABEL, "plan": "self_serve"})] * 7
    outputs += [json.dumps({**LABEL, "plan": "enterprise"})] * 3
    distribution = run(outputs)["predicted_distribution"]["plan"]
    assert list(distribution) == ["self_serve", "enterprise"]
    assert list(distribution.values()) == [7, 3]


def test_distribution_counts_unparsed_rows_as_none():
    result = run(["not json at all"] * 4)
    assert result["predicted_distribution"]["category"] == {"None": 4}


def test_distribution_covers_every_field():
    assert set(run([target_json(LABEL)])["predicted_distribution"]) == set(
        ("category", "severity", "plan", "escalate")
    )
