"""Unit tests for the HTML fragments in ui/components.py (no Streamlit runtime needed)."""

import re

import pytest

from core.schemas import CATEGORIES, CheckOptions, CheckResult, Suggestion
from ui import components as ui


def make(sid, category="grammar", original="teh", replacement="the", explanation="Spelling.", at=0):
    return Suggestion(sid, category, original, replacement, explanation, at, at + len(original), 0)


def result_of(*suggestions):
    return CheckResult("text", CheckOptions(), tuple(suggestions), (), (), 1)


def balanced(markup):
    return markup.count("<") == markup.count(">")


# ---------------------------------------------------------------- escaping


EVIL = '<img src=x onerror="evil()"> & <script>alert(1)</script>'


@pytest.mark.parametrize("status", ["pending", "accepted", "rejected", "bogus"])
def test_card_escapes_every_model_and_user_string(status):
    s = make("s1", original=EVIL, replacement=EVIL, explanation=EVIL)
    out = ui.card_html(s, status)
    assert "<img" not in out and "<script" not in out
    assert out.count("&lt;img") == 3 and balanced(out)


def test_summary_and_group_headers_escape_unknown_categories():
    s = make("s1", category='"><b>x</b>')
    out = ui.summary_html(result_of(s), {}) + ui.group_header_html('"><b>x</b>', 1)
    assert "<b>" not in out and balanced(out)


# ---------------------------------------------------------------- cards


def test_card_shows_an_explicit_status_label_not_only_a_colour():
    s = make("s1")
    assert "Not decided" in ui.card_html(s, "pending")
    assert "Accepted" in ui.card_html(s, "accepted")
    assert "Rejected" in ui.card_html(s, "rejected")
    assert "Not decided" in ui.card_html(s, "unexpected")  # unknown states fall back safely


def test_card_reads_as_a_sentence_for_screen_readers():
    text = re.sub(r"<[^>]+>", "", ui.card_html(make("s1"), "pending"))
    assert "Replace teh→ with the" in text  # arrow is aria-hidden; the words carry the meaning


def test_a_deletion_is_labelled_instead_of_showing_an_empty_box():
    out = ui.card_html(make("s1", original="very", replacement=""), "pending")
    assert "(remove)" in out


def test_card_carries_its_category_for_colour_coding():
    assert "gsc-card--tone" in ui.card_html(make("s1", category="tone"), "pending")


# ---------------------------------------------------------------- grouping and counts


def test_groups_follow_the_fixed_category_order_and_keep_text_order_inside():
    items = [
        make("s1", "clarity", at=0),
        make("s2", "grammar", at=10),
        make("s3", "tone", at=20),
        make("s4", "grammar", at=30),
    ]
    groups = ui.group_by_category(items)
    assert [c for c, _ in groups] == ["grammar", "tone", "clarity"]
    assert [s.id for s in groups[0][1]] == ["s2", "s4"]


def test_empty_categories_are_not_shown():
    assert [c for c, _ in ui.group_by_category([make("s1", "tone")])] == ["tone"]
    assert ui.group_by_category([]) == []


def test_decision_counts():
    items = [make(f"s{i}", at=i * 5) for i in range(1, 5)]
    counts = ui.decision_counts(items, {"s1": "accepted", "s2": "rejected", "zz": "accepted"})
    assert (counts.accepted, counts.rejected, counts.pending, counts.total) == (1, 1, 2, 4)


def test_summary_reports_totals_per_category_and_decisions():
    items = [make("s1", "grammar"), make("s2", "grammar", at=5), make("s3", "clarity", at=9)]
    out = ui.summary_html(result_of(*items), {"s1": "accepted"})
    assert "3 suggestions" in out
    assert re.search(r"Grammar\s*<span[^>]*>2</span>", out)
    assert re.search(r"Clarity\s*<span[^>]*>1</span>", out)
    assert "Tone" not in out  # no empty chips
    assert "1 accepted" in out and "0 rejected" in out and "2 to review" in out


def test_plural():
    assert ui.plural(1, "suggestion") == "1 suggestion"
    assert ui.plural(0, "suggestion") == "0 suggestions"
    assert ui.plural(1200, "word") == "1,200 words"


# ---------------------------------------------------------------- word counter and copy


@pytest.mark.parametrize(
    ("words", "state"), [(0, "empty"), (1, "ok"), (1000, "ok"), (1001, "over")]
)
def test_word_counter_states(words, state):
    out = ui.word_count_html(words, 1000)
    assert f"gsc-count--{state}" in out and f"{words:,} / 1,000 words" in out


def test_every_category_has_a_label_and_a_hint():
    for category in CATEGORIES:
        assert ui.CATEGORY_LABELS[category] and ui.CATEGORY_HINTS[category]


def test_example_text_fits_the_default_limits():
    words = len(ui.EXAMPLE_TEXT.split())
    assert 20 <= words <= 400  # one request with the default settings


def test_static_fragments_are_well_formed():
    for markup in (
        ui.header_html(),
        ui.empty_state_html(),
        ui.preview_legend_html(),
        ui.clean_result_html(),
    ):
        assert balanced(markup) and "<script" not in markup
