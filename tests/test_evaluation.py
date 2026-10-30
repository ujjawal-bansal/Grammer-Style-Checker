"""Offline tests for the evaluation harness (no API calls)."""

import json

import pytest

from core.schemas import CheckOptions, CheckResult, SectionIssue, Suggestion
from evaluation.harness import (
    Case,
    contains_phrase,
    evaluate,
    load_cases,
    merge,
    normalise,
    outcomes_from_json,
    outcomes_to_json,
    parse_case,
    render_report,
    score,
)


def case(**overrides):
    base = {
        "id": "c1",
        "group": "G",
        "text": "Their going.",
        "expect": "change",
        "must_contain": ["They're going."],
    }
    base.update(overrides)
    return parse_case(base)


def result_for(text, replacements=(), issues=()):
    suggestions = []
    for i, (original, replacement) in enumerate(replacements, start=1):
        start = text.index(original)
        suggestions.append(
            Suggestion(
                f"s{i}", "grammar", original, replacement, "why", start, start + len(original), 0
            )
        )
    return CheckResult(text, CheckOptions(), tuple(suggestions), (), tuple(issues), 1)


# ---------------------------------------------------------------- the case file


def test_the_shipped_case_file_is_valid_and_covers_every_kind_of_case():
    cases = load_cases()
    assert len(cases) >= 25
    groups = {c.group for c in cases}
    for group in (
        "Broken sentences",
        "Contractions and possessives",
        "Subject-verb agreement",
        "Tense",
        "Pronoun case",
        "Spelling and punctuation",
        "Already correct",
        "Ambiguous but correct",
        "Tone",
        "Clarity",
    ):
        assert group in groups
    assert {c.expect for c in cases} == {"change", "unchanged"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expect": "maybe"}, "expect must be one of"),
        ({"checks": ["spelling"]}, "checks must be"),
        ({"checks": []}, "checks must be"),
        ({"tone": "angry"}, "unknown tone"),
        ({"must_contain": []}, "needs at least one must_contain"),
        ({"typo_field": 1}, "unknown fields"),
    ],
)
def test_invalid_cases_are_rejected_loudly(overrides, message):
    with pytest.raises(ValueError, match=message):
        case(**overrides)


def test_duplicate_ids_are_rejected(tmp_path):
    line = json.dumps({"id": "x", "group": "G", "text": "Fine.", "expect": "unchanged"})
    path = tmp_path / "cases.jsonl"
    path.write_text(f"{line}\n{line}\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


def test_options_come_from_the_case():
    c = case(checks=["tone"], tone="formal")
    assert c.options.categories == frozenset({"tone"}) and c.options.tone == "formal"


# ---------------------------------------------------------------- scoring


def test_unchanged_cases_must_come_back_exactly_as_written():
    c = case(expect="unchanged", text="I saw her duck.", must_contain=[])
    assert score(c, "I saw her duck.") == (True, ())
    passed, reasons = score(c, "I saw her pet duck.")
    assert not passed and "should stay as written" in reasons[0]


def test_change_cases_need_every_expected_fix_and_none_of_the_errors():
    c = case(text="Me and him goes.", must_contain=["He and I go."], must_not_contain=["him"])
    assert score(c, "He and I go.")[0]
    assert not score(c, "Me and him goes.")[0]  # untouched
    passed, reasons = score(c, "He and him go.")
    assert not passed and any("missing" in r for r in reasons)
    assert any("still contains 'him'" in r for r in reasons)


def test_forbidden_phrases_match_whole_words_only():
    # forbidding "am study" must not reject the correct "am studying"
    c = case(text="i am study", must_contain=["I am studying."], must_not_contain=["am study"])
    assert score(c, "I am studying.")[0]
    assert not score(c, "I am study.")[0]
    assert contains_phrase("park ; however", " ;")  # punctuation edges match as written
    assert not contains_phrase("park; however", " ;")


def test_no_forbidden_phrase_in_the_case_file_is_part_of_its_own_expected_fix():
    for c in load_cases():
        for bad in c.must_not_contain:
            for good in c.must_contain:
                assert not contains_phrase(good, bad), (c.id, bad, good)


def test_straight_and_curly_apostrophes_are_both_accepted():
    assert normalise("They’re “fine”") == 'They\'re "fine"'
    assert score(case(), "They’re going.")[0]


# ---------------------------------------------------------------- running cases


def test_evaluate_scores_the_text_after_accepting_every_suggestion():
    cases = [case(), case(id="c2", text="All good.", expect="unchanged", must_contain=[])]
    responses = {
        "Their going.": result_for("Their going.", [("Their", "They're")]),
        "All good.": result_for("All good."),
    }
    seen = []
    summary = evaluate(cases, lambda text, options: responses[text], on_outcome=seen.append)
    assert (summary.passed, summary.total) == (2, 2)
    assert (
        summary.outcomes[0].corrected == "They're going." and summary.outcomes[0].suggestions == 1
    )
    assert len(seen) == 2


def test_a_failing_or_partial_check_counts_as_a_failed_case_never_a_pass():
    def boom(text, options):
        raise RuntimeError("network down")

    summary = evaluate([case()], boom)
    assert summary.passed == 0 and summary.outcomes[0].error == "RuntimeError"

    issue = SectionIssue(0, 0, 5, "failed", "rate limit")
    partial = result_for("Their going.", [("Their", "They're")], issues=[issue])
    summary = evaluate([case()], lambda text, options: partial)
    assert summary.passed == 0 and summary.outcomes[0].error == "rate limit"


def test_report_lists_totals_groups_and_every_case():
    cases = [case(), case(id="c2", group="H", text="Fine.", expect="unchanged", must_contain=[])]
    responses = {
        "Their going.": result_for("Their going."),  # uncorrected -> fail
        "Fine.": result_for("Fine."),
    }
    report = render_report(evaluate(cases, lambda t, o: responses[t]), "Run note.")
    assert "**Result: 1 / 2 cases passed (50%).**" in report
    assert "| G | 0 / 1 |" in report and "| H | 1 / 1 |" in report
    assert "FAIL: left an error uncorrected" in report and "Run note." in report


def test_unfinished_cases_are_counted_and_explained_in_the_report():
    def boom(text, options):
        raise RuntimeError("rate limited")

    summary = evaluate([case()], boom)
    assert summary.incomplete == 1 and not summary.outcomes[0].completed
    assert "could not be checked because of API errors" in render_report(summary, "n")


def test_merge_replaces_outcomes_per_case_and_keeps_case_file_order():
    cases = [case(id="a"), case(id="b"), case(id="c")]
    fixed = result_for("Their going.", [("Their", "They're")])
    first = evaluate(cases[:2], lambda t, o: (_ for _ in ()).throw(RuntimeError("429")))
    second = evaluate([cases[1], cases[2]], lambda t, o: fixed)
    merged = merge(first, second, cases)
    assert [o.case.id for o in merged.outcomes] == ["a", "b", "c"]
    assert [o.completed for o in merged.outcomes] == [False, True, True]
    assert merged.passed == 2


def test_saved_outcomes_round_trip():
    cases = [case(id="a"), case(id="b")]
    responses = iter(
        [result_for("Their going.", [("Their", "They're")]), result_for("Their going.")]
    )
    summary = evaluate(cases, lambda t, o: next(responses))
    restored = outcomes_from_json(json.loads(json.dumps(outcomes_to_json(summary))), cases)
    assert [(o.case.id, o.passed, o.corrected) for o in restored.outcomes] == [
        (o.case.id, o.passed, o.corrected) for o in summary.outcomes
    ]


def test_saved_verdicts_are_dropped_when_the_case_expectations_change():
    old = case(id="a")
    summary = evaluate([old], lambda t, o: result_for("Their going.", [("Their", "They're")]))
    saved = outcomes_to_json(summary)
    stricter = case(id="a", must_contain=["They're going.", "never present"])
    assert outcomes_from_json(saved, [stricter]).outcomes == []  # re-run, never reuse
    assert outcomes_from_json(saved, []).outcomes == []  # case removed from the file


def test_report_escapes_table_separators():
    c = Case("p", "G", "a | b", "unchanged")
    report = render_report(evaluate([c], lambda t, o: result_for("a | b")), "note")
    assert "a \\| b" in report
