import random

import groq
import httpx
import pytest

from core.apply import apply_accepted
from core.errors import InputValidationError
from core.llm_client import GroqClient
from core.pipeline import (
    MAX_SPLIT_DEPTH,
    chunk_text,
    count_words,
    run_check,
    split_chunk,
    validate_input,
)
from core.schemas import CheckOptions
from tests.helpers import FakeGroq, make_settings, ok_response, rate_limit, status_error, suggestion

REQ = httpx.Request("POST", "https://api.groq.com/x")
P1, P2, P3 = "I has a apple today.", "She dont like it.", "They was happy then."
TEXT = f"{P1}\n\n{P2}\n\n{P3}"  # 13 words, three paragraphs


def sug(original, replacement, context_before="", category="grammar"):
    return suggestion(original, replacement, category, "why", context_before)


class Setup:
    def __init__(self, script, **settings):
        self.settings = make_settings(**settings)
        self.fake = FakeGroq(script)
        self.sleeps: list[float] = []
        self.client = GroqClient(
            self.settings, client=self.fake, sleep=self.sleeps.append, rand=lambda: 0.0
        )

    def run(self, text=TEXT, options=None):
        return run_check(text, options or CheckOptions(), self.settings, self.client)

    @property
    def sent_texts(self):
        return [c["messages"][1]["content"] for c in self.fake.calls]


# ---------------------------------------------------------------- input validation


@pytest.mark.parametrize("text", ["", "   ", "\n\t \n"])
def test_blank_input_rejected(text):
    with pytest.raises(InputValidationError, match="enter some text"):
        validate_input(text, make_settings())


def test_non_string_and_control_characters_rejected():
    with pytest.raises(InputValidationError):
        validate_input(None, make_settings())  # type: ignore[arg-type]
    with pytest.raises(InputValidationError, match="control characters"):
        validate_input("hello\x00world", make_settings())


def test_word_limit_is_enforced_and_configurable():
    settings = make_settings(max_input_words=10)
    validate_input(" ".join(["w"] * 10), settings)
    with pytest.raises(InputValidationError, match="11 words; the limit is 10"):
        validate_input(" ".join(["w"] * 11), settings)
    validate_input(" ".join(["w"] * 11), make_settings(max_input_words=50))


def test_character_limit_catches_whitespace_free_input():
    with pytest.raises(InputValidationError, match="too long"):
        validate_input("x" * 1000, make_settings(max_input_words=10))


def test_invalid_input_never_calls_the_api():
    setup = Setup([], max_input_words=5)
    with pytest.raises(InputValidationError):
        setup.run("one two three four five six")
    assert setup.fake.calls == []


# ---------------------------------------------------------------- chunking


def assert_chunk_invariants(text, chunks, max_words):
    words_in_chunks = []
    previous_end = 0
    for c in chunks:
        assert text[c.start : c.end] == c.text
        assert c.text == c.text.strip() and c.text
        assert 1 <= count_words(c.text) <= max_words
        assert c.start >= previous_end
        previous_end = c.end
        words_in_chunks.extend(c.text.split())
    assert words_in_chunks == text.split()


def test_short_text_is_one_chunk():
    chunks = chunk_text("  Hello there world.  ", 400)
    assert len(chunks) == 1 and chunks[0].text == "Hello there world."
    assert chunks[0].start == 2


def test_empty_text_has_no_chunks():
    assert chunk_text("   ", 10) == []


def test_chunks_prefer_paragraph_boundaries():
    chunks = chunk_text(TEXT, 5)
    assert [c.text for c in chunks] == [P1, P2, P3]
    assert_chunk_invariants(TEXT, chunks, 5)


def test_chunks_prefer_sentence_boundaries_when_no_paragraphs():
    text = "One two three four five. Six seven eight nine ten eleven twelve."
    chunks = chunk_text(text, 8)
    assert chunks[0].text == "One two three four five."
    assert_chunk_invariants(text, chunks, 8)


def test_hard_split_when_there_is_no_natural_break():
    text = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_text(text, 10)
    assert [count_words(c.text) for c in chunks] == [10, 10, 5]
    assert_chunk_invariants(text, chunks, 10)


def test_chunking_invariants_hold_for_random_text():
    rng = random.Random(99)
    for _ in range(200):
        pieces = []
        for _ in range(rng.randint(1, 80)):
            pieces.append(rng.choice(["alpha", "beta.", "gamma!", "delta", "épsilon?", "zeta"]))
            pieces.append(rng.choice([" ", " ", " ", "  ", "\n", "\n\n", " \n \n "]))
        text = "".join(pieces)
        max_words = rng.randint(1, 15)
        assert_chunk_invariants(text, chunk_text(text, max_words), max_words)


def test_split_chunk_halves_with_absolute_offsets():
    text = "Pre. I has a apple. She dont like it."
    whole = chunk_text(text, 100)[0]
    parts = split_chunk(whole)
    assert len(parts) == 2 and {p.index for p in parts} == {whole.index}
    for p in parts:
        assert text[p.start : p.end] == p.text
    assert " ".join(p.text for p in parts) == whole.text


def test_single_word_chunk_cannot_be_split():
    chunk = chunk_text("word", 10)[0]
    assert split_chunk(chunk) == [chunk]


# ---------------------------------------------------------------- happy path


def three_chunk_script():
    return [
        ok_response([sug("has", "have", "I "), sug("a apple", "an apple")]),
        ok_response([sug("dont", "don't")]),
        ok_response([sug("was", "were", "They ")]),
    ]


def test_full_run_merges_chunks_with_correct_absolute_offsets():
    setup = Setup(three_chunk_script(), max_chunk_words=5)
    result = setup.run()
    assert result.total_chunks == 3 and not result.is_partial
    assert [s.id for s in result.suggestions] == ["s1", "s2", "s3", "s4"]
    for s in result.suggestions:
        assert TEXT[s.start : s.end] == s.original
    assert [s.original for s in result.suggestions] == ["has", "a apple", "dont", "was"]
    corrected = apply_accepted(TEXT, result.suggestions, [s.id for s in result.suggestions])
    assert corrected == "I have an apple today.\n\nShe don't like it.\n\nThey were happy then."
    assert result.unverified_count == 0
    assert len(setup.fake.calls) == 3
    # chunks are sent sequentially, in order
    assert P1 in setup.sent_texts[0] and P2 in setup.sent_texts[1] and P3 in setup.sent_texts[2]


def test_no_suggestions_is_a_normal_result():
    result = Setup([ok_response([])]).run("Perfectly fine text.")
    assert result.suggestions == () and not result.is_partial and result.unverified_count == 0


def test_only_the_configured_model_is_used_for_every_call():
    setup = Setup(three_chunk_script(), max_chunk_words=5, model="openai/gpt-oss-120b")
    setup.run()
    assert {c["model"] for c in setup.fake.calls} == {"openai/gpt-oss-120b"}


def test_options_reach_the_request():
    setup = Setup([ok_response([])])
    setup.run("Some text here.", CheckOptions(categories={"tone"}, tone="friendly"))
    call = setup.fake.calls[0]
    enum = call["response_format"]["json_schema"]["schema"]["properties"]["suggestions"]
    assert enum["items"]["properties"]["category"]["enum"] == ["tone"]
    assert "Target tone: friendly" in setup.sent_texts[0]


# ------------------------------------------------------- unverified suggestions are reported


def test_unverifiable_suggestions_are_counted_and_reported_not_hidden():
    items = [
        sug("has", "have", "I "),  # fine
        sug("not in the text", "x"),  # not found
        sug("today", "today"),  # no-op (reported separately)
        {"category": "grammar", "original": "x"},  # malformed item
        sug("apple", "apples"),
        sug("a apple", "an apple"),  # overlaps "apple"
    ]
    result = Setup([ok_response(items)]).run(P1)
    assert [s.original for s in result.suggestions] == ["has", "a apple"]
    counts = result.drop_counts()
    assert counts["not_found"] == 1 and counts["invalid"] == 1 and counts["no_op"] == 1
    assert counts["overlap"] == 1
    assert result.unverified_count == 3  # no-op excluded
    assert len(result.suggestions) + result.unverified_count + counts["no_op"] == 6


def test_ambiguous_suggestions_are_reported():
    text = "the cat and the dog"
    result = Setup([ok_response([sug("the", "a")])]).run(text)
    assert result.suggestions == () and result.drop_counts() == {"ambiguous": 1}


def test_overflowing_model_output_is_reported():
    items = [sug(f"w{i}", f"x{i}") for i in range(45)]
    result = Setup([ok_response(items)]).run(" ".join(f"w{i}" for i in range(45)))
    assert result.drop_counts().get("overflow") == 5


def test_overlaps_never_corrupt_the_corrected_text():
    # "a apple" (6-13) overlaps both other spans; "has a" (2-7) and "apple today" (8-19) do not.
    items = [sug("has a", "have an"), sug("a apple", "an apple"), sug("apple today", "apple now")]
    result = Setup([ok_response(items)]).run(P1)
    assert [s.original for s in result.suggestions] == ["has a", "apple today"]
    out = apply_accepted(P1, result.suggestions, [s.id for s in result.suggestions])
    assert out == "I have an apple now."
    assert result.drop_counts() == {"overlap": 1}


# ---------------------------------------------------------------- partial results


def test_daily_rate_limit_on_second_chunk_preserves_first_chunk_results():
    headers = {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "5m"}
    script = [ok_response([sug("has", "have", "I ")]), rate_limit(headers), ok_response([])]
    setup = Setup(script, max_chunk_words=5)
    result = setup.run()
    assert [s.original for s in result.suggestions] == ["has"]
    assert result.is_partial
    kinds = [(i.chunk_index, i.kind) for i in result.issues]
    assert kinds == [(1, "failed"), (2, "skipped")]
    assert "daily" in result.issues[0].message
    assert len(setup.fake.calls) == 2  # chunk 3 was never sent
    assert apply_accepted(TEXT, result.suggestions, ["s1"]).startswith("I have a apple")


def test_timeouts_stop_further_calls_but_keep_earlier_results():
    script = [ok_response([sug("has", "have", "I ")])] + [groq.APITimeoutError(request=REQ)] * 3
    setup = Setup(script, max_chunk_words=5, max_retries=2)
    result = setup.run()
    assert len(result.suggestions) == 1
    assert [i.kind for i in result.issues] == ["failed", "skipped"]
    assert len(setup.fake.calls) == 4


def test_server_error_is_recorded_without_raising():
    script = [ok_response([sug("has", "have", "I ")]), status_error(groq.InternalServerError, 503)]
    result = Setup(script, max_chunk_words=5, max_retries=0).run()
    assert result.is_partial and "HTTP 503" in result.issues[0].message


def test_invalid_output_in_one_chunk_does_not_stop_later_chunks():
    script = [
        ok_response([sug("has", "have", "I ")]),
        *[ok_response(content="garbage")] * 3,
        ok_response([sug("was", "were", "They ")]),
    ]
    setup = Setup(script, max_chunk_words=5, max_retries=2)
    result = setup.run()
    assert [s.original for s in result.suggestions] == ["has", "was"]
    assert [(i.chunk_index, i.kind) for i in result.issues] == [(1, "failed")]
    assert len(setup.fake.calls) == 5


def test_failure_on_the_first_chunk_returns_a_result_with_no_suggestions():
    setup = Setup([rate_limit({"retry-after": "60"})], max_chunk_words=5)
    result = setup.run()
    assert result.suggestions == ()
    assert [i.kind for i in result.issues] == ["failed", "skipped", "skipped"]
    assert len(setup.fake.calls) == 1


def test_short_rate_limit_is_waited_out_and_the_run_completes():
    script = [
        ok_response([sug("has", "have", "I ")]),
        rate_limit({"retry-after": "2"}),
        ok_response([sug("dont", "don't")]),
        ok_response([]),
    ]
    setup = Setup(script, max_chunk_words=5)
    result = setup.run()
    assert not result.is_partial and len(result.suggestions) == 2
    assert setup.sleeps == [2.0]


# ---------------------------------------------------------------- token-limit handling

TWO_SENTENCES = "I has a apple. She dont like it."  # 8 words, one chunk


def too_large():
    return status_error(groq.APIStatusError, 413)


def test_oversized_chunk_is_split_in_half_and_both_halves_are_checked():
    script = [
        too_large(),
        ok_response([sug("has", "have", "I ")]),
        ok_response([sug("dont", "don't")]),
    ]
    setup = Setup(script, max_chunk_words=8)
    result = setup.run(TWO_SENTENCES)
    assert not result.is_partial
    assert [s.original for s in result.suggestions] == ["has", "dont"]
    for s in result.suggestions:
        assert TWO_SENTENCES[s.start : s.end] == s.original
    assert len(setup.fake.calls) == 3
    assert "I has a apple." in setup.sent_texts[1] and "She dont" in setup.sent_texts[2]


def test_a_half_that_is_still_too_large_is_reported_but_the_other_half_survives():
    assert MAX_SPLIT_DEPTH == 1
    script = [too_large(), too_large(), ok_response([sug("dont", "don't")])]
    setup = Setup(script, max_chunk_words=8)
    result = setup.run(TWO_SENTENCES)
    assert [s.original for s in result.suggestions] == ["dont"]
    assert [i.kind for i in result.issues] == ["failed"] and "token limit" in result.issues[
        0
    ].message
    assert len(setup.fake.calls) == 3  # no endless splitting


def test_truncated_output_triggers_the_same_split_fallback():
    script = [
        ok_response(content='{"suggestions": [', finish_reason="length"),
        ok_response([]),
        ok_response([]),
    ]
    setup = Setup(script, max_chunk_words=8)
    assert not setup.run(TWO_SENTENCES).is_partial
    assert len(setup.fake.calls) == 3


def test_unsplittable_chunk_is_reported_as_failed():
    setup = Setup([too_large()])
    result = setup.run("Supercalifragilistic")
    assert [i.kind for i in result.issues] == ["failed"] and len(setup.fake.calls) == 1


# ---------------------------------------------------------------- progress reporting


def run_with_progress(setup, text=TEXT):
    seen: list[tuple[int, int]] = []
    result = run_check(
        text, CheckOptions(), setup.settings, setup.client, progress=lambda *a: seen.append(a)
    )
    return result, seen


def test_progress_is_reported_before_each_section_and_at_the_end():
    setup = Setup(three_chunk_script(), max_chunk_words=5)
    result, seen = run_with_progress(setup)
    assert seen == [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert len(result.suggestions) == 4  # reporting progress changes nothing else


def test_a_split_section_does_not_change_the_reported_total():
    script = [too_large(), ok_response([]), ok_response([])]
    _, seen = run_with_progress(Setup(script, max_chunk_words=8), TWO_SENTENCES)
    assert seen == [(0, 1), (1, 1)]


def test_progress_still_completes_when_a_section_fails():
    headers = {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "7m"}
    script = [ok_response([]), rate_limit(headers)]
    result, seen = run_with_progress(Setup(script, max_chunk_words=5))
    assert seen[-1] == (3, 3) and result.is_partial


def test_invalid_input_reports_no_progress():
    with pytest.raises(InputValidationError):
        run_with_progress(Setup([]), "   ")
