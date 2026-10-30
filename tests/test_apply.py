import html
import random
import re

import pytest

from core.apply import (
    apply_accepted,
    assign_ids,
    find_occurrences,
    locate_suggestions,
    render_html,
    resolve_overlaps,
)
from core.errors import StaleSuggestionError
from core.schemas import Chunk, RawSuggestion, Suggestion

OFFSET = 100


def raw(original, replacement="X", context_before="", category="grammar"):
    return RawSuggestion(
        category=category,
        original=original,
        replacement=replacement,
        explanation="why",
        context_before=context_before,
    )


def locate(text, *raws):
    return locate_suggestions(Chunk(0, OFFSET, OFFSET + len(text), text), raws)


def sug(id_, start, end, text, replacement="X", category="grammar"):
    return Suggestion(id_, category, text[start:end], replacement, "why", start, end, 0)


# ---------------------------------------------------------------- locating


def test_find_occurrences_includes_overlapping_matches():
    assert find_occurrences("aaaa", "aa") == [0, 1, 2]
    assert find_occurrences("abc", "") == []


def test_unique_match_gets_absolute_offsets():
    text = "I has a apple today."
    found, dropped = locate(text, raw("has", "have"))
    assert dropped == []
    s = found[0]
    assert (s.start, s.end) == (OFFSET + 2, OFFSET + 5)
    assert text[s.start - OFFSET : s.end - OFFSET] == "has"


def test_repeated_text_is_disambiguated_by_context_before():
    text = "the cat saw the dog"
    found, dropped = locate(text, raw("the", "a", context_before="saw"))
    assert dropped == [] and found[0].start - OFFSET == 12
    found, _ = locate(text, raw("the", "a", context_before="saw "))
    assert found[0].start - OFFSET == 12


@pytest.mark.parametrize("context", ["", "   ", "nothing matches", "a"])
def test_repeated_text_without_usable_context_is_reported_ambiguous(context):
    found, dropped = locate("a x. a x.", raw("x", "y", context_before=context))
    assert found == []
    assert [d.reason for d in dropped] == ["ambiguous"]


def test_context_matching_several_places_is_still_ambiguous():
    found, dropped = locate("go now. go now.", raw("now", "later", context_before="go"))
    assert found == [] and dropped[0].reason == "ambiguous"


def test_missing_original_is_reported_not_found():
    found, dropped = locate("hello world", raw("absent text"))
    assert found == [] and dropped[0].reason == "not_found"
    assert dropped[0].original == "absent text" and dropped[0].category == "grammar"


def test_match_inside_a_longer_word_is_rejected():
    found, dropped = locate("another cat", raw("the", "a"))
    assert found == [] and dropped[0].reason == "mid_word"


def test_standalone_word_is_preferred_over_mid_word_matches():
    text = "another the cat"
    found, dropped = locate(text, raw("the", "a"))
    assert dropped == [] and text[found[0].start - OFFSET : found[0].end - OFFSET] == "the"
    assert found[0].start - OFFSET == 8


def test_no_op_suggestions_are_reported_separately():
    found, dropped = locate("same text here", raw("same", "same"))
    assert found == [] and dropped[0].reason == "no_op"


def test_an_echoed_correction_that_is_not_in_the_text_is_unverified_not_a_no_op():
    # The model sometimes copies its corrected text into both fields. That is a failed
    # suggestion, so it is counted as unverified instead of being silently ignored.
    found, dropped = locate(
        "We went to the park however it rained.",
        raw("park; however, it rained", "park; however, it rained"),
    )
    assert found == [] and dropped[0].reason == "not_found"


def test_matching_is_exact_not_fuzzy():
    found, dropped = locate("It is  double spaced", raw("is double", "is"))
    assert found == [] and dropped[0].reason == "not_found"
    found, dropped = locate("Cafe", raw("cafe", "Cafe"))
    assert found == [] and dropped[0].reason == "not_found"


def test_unicode_and_punctuation_spans():
    text = "Le café naïve, dont bien."
    found, dropped = locate(text, raw("naïve,", "naïve;"), raw("dont", "don't"))
    assert dropped == [] and len(found) == 2


def test_long_originals_are_previewed_in_drop_reports():
    _, dropped = locate("text", raw("z" * 500))
    assert len(dropped[0].original) <= 60


# ---------------------------------------------------------------- overlaps / ids


def test_overlapping_suggestions_cannot_both_survive():
    text = "0123456789"
    a, b, c = sug("a", 0, 5, text), sug("b", 3, 7, text), sug("c", 7, 9, text)
    kept, dropped = resolve_overlaps([b, c, a])
    assert [s.id for s in kept] == ["a", "c"]
    assert [d.reason for d in dropped] == ["overlap"]


def test_adjacent_suggestions_are_both_kept():
    text = "0123456789"
    kept, dropped = resolve_overlaps([sug("a", 0, 5, text), sug("b", 5, 8, text)])
    assert len(kept) == 2 and dropped == []


def test_identical_spans_keep_only_one():
    text = "0123456789"
    kept, dropped = resolve_overlaps([sug("a", 2, 6, text), sug("b", 2, 6, text)])
    assert len(kept) == 1 and len(dropped) == 1


def test_overlap_resolution_is_deterministic_regardless_of_input_order():
    text = "0123456789" * 3
    items = [
        sug(f"x{i}", s, e, text)
        for i, (s, e) in enumerate([(0, 4), (2, 6), (5, 9), (8, 12), (11, 14)])
    ]
    expected = [s.id for s in resolve_overlaps(items)[0]]
    rng = random.Random(7)
    for _ in range(20):
        shuffled = items[:]
        rng.shuffle(shuffled)
        assert [s.id for s in resolve_overlaps(shuffled)[0]] == expected


def test_assign_ids_orders_by_position():
    text = "0123456789"
    out = assign_ids([sug("", 5, 7, text), sug("", 0, 2, text)])
    assert [(s.id, s.start) for s in out] == [("s1", 0), ("s2", 5)]


# ---------------------------------------------------------------- applying

TEXT = "I has a apple. She dont like it.\nThey was happy."


@pytest.fixture
def suggestions():
    def make(id_, original, replacement):
        start = TEXT.index(original)
        return Suggestion(
            id_, "grammar", original, replacement, "why", start, start + len(original), 0
        )

    return [
        make("s1", "has", "have"),
        make("s2", "a apple", "an apple"),
        make("s3", "dont", "don't"),
        make("s4", "was", "were"),
    ]


def test_no_accepted_suggestions_leaves_text_untouched(suggestions):
    assert apply_accepted(TEXT, suggestions, []) == TEXT


def test_all_accepted(suggestions):
    out = apply_accepted(TEXT, suggestions, ["s1", "s2", "s3", "s4"])
    assert out == "I have an apple. She don't like it.\nThey were happy."


def test_subset_accepted(suggestions):
    assert apply_accepted(TEXT, suggestions, ["s3"]) == TEXT.replace("dont", "don't")


def test_result_does_not_depend_on_acceptance_order(suggestions):
    ids = ["s1", "s2", "s3", "s4"]
    expected = apply_accepted(TEXT, suggestions, ids)
    rng = random.Random(3)
    for _ in range(10):
        rng.shuffle(ids)
        assert apply_accepted(TEXT, suggestions, ids) == expected
        assert apply_accepted(TEXT, list(reversed(suggestions)), ids) == expected


def test_unknown_ids_are_ignored(suggestions):
    assert apply_accepted(TEXT, suggestions, ["nope"]) == TEXT


def test_changed_text_is_detected_instead_of_corrupted(suggestions):
    with pytest.raises(StaleSuggestionError):
        apply_accepted("X" + TEXT, suggestions, ["s1"])
    with pytest.raises(StaleSuggestionError):
        apply_accepted(TEXT[:5], suggestions, ["s4"])


def test_overlapping_accepted_suggestions_are_refused_defensively():
    text = "0123456789"
    a, b = sug("a", 0, 5, text), sug("b", 3, 7, text)
    with pytest.raises(StaleSuggestionError):
        apply_accepted(text, [a, b], ["a", "b"])
    assert apply_accepted(text, [a, b], ["a"]) == "X56789"  # either one alone is fine


def test_matches_naive_reference_for_random_selections():
    rng = random.Random(1234)
    for _ in range(300):
        text = "".join(rng.choice("abcde \n.é") for _ in range(rng.randint(1, 120)))
        items, pos, n = [], 0, 0
        while pos < len(text):
            start = pos + rng.randint(0, 5)
            if start >= len(text):
                break
            end = min(len(text), start + rng.randint(1, 6))
            items.append(
                Suggestion(
                    f"s{n}",
                    "grammar",
                    text[start:end],
                    rng.choice(["", "X", "a longer one"]),
                    "e",
                    start,
                    end,
                    0,
                )
            )
            n += 1
            pos = end + rng.randint(0, 3)
        accepted = {s.id for s in items if rng.random() < 0.5}
        expected = text
        for s in sorted(
            (s for s in items if s.id in accepted), key=lambda s: s.start, reverse=True
        ):
            expected = expected[: s.start] + s.replacement + expected[s.end :]
        order = list(accepted)
        rng.shuffle(order)
        assert apply_accepted(text, items, order) == expected


# ---------------------------------------------------------------- html rendering


def strip_tags(markup):
    return html.unescape(re.sub(r"<[^>]+>", "", markup))


def test_render_escapes_everything_untrusted():
    text = 'Say <script>alert("x")</script> & "quote" teh end'
    start = text.index("teh")
    s = Suggestion(
        "s1",
        "grammar",
        "teh",
        '<img src=x onerror="evil()">',
        'bad "title" <b>',
        start,
        start + 3,
        0,
    )
    for status in ("pending", "accepted", "rejected"):
        out = render_html(text, [s], {"s1": status})
        assert "<script" not in out and "<img" not in out and "<b>" not in out
        assert "&lt;script&gt;" in out
        assert out.count("<") == out.count(">")


def test_render_pending_reconstructs_the_original_text_exactly():
    text = "I has a apple.\n\nShe dont <like> it & more."
    items = [
        Suggestion("s1", "grammar", "has", "have", "why", 2, 5, 0),
        Suggestion(
            "s2", "grammar", "dont", "don't", "why", text.index("dont"), text.index("dont") + 4, 0
        ),
    ]
    assert strip_tags(render_html(text, items, {})) == text


def test_render_marks_each_status_differently():
    text = "one two three"
    items = [
        Suggestion("s1", "grammar", "one", "1", "why", 0, 3, 0),
        Suggestion("s2", "grammar", "two", "2", "why", 4, 7, 0),
        Suggestion("s3", "grammar", "three", "3", "why", 8, 13, 0),
    ]
    out = render_html(text, items, {"s1": "accepted", "s2": "rejected"})
    assert "<del" in out and "<ins" in out and "<mark" in out


def test_render_refuses_stale_suggestions():
    s = Suggestion("s1", "grammar", "xyz", "q", "why", 0, 3, 0)
    with pytest.raises(StaleSuggestionError):
        render_html("abc", [s], {})
