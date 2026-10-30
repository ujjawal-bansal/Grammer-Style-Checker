"""Presentation helpers for the Streamlit UI.

Pure functions that turn result data into small HTML fragments, so the page layout in
`app.py` stays readable and the markup can be tested without a browser. Every piece of user
or model text is HTML-escaped here; the only markup is what these functions emit.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from core.schemas import CATEGORIES, CheckResult, Suggestion

CATEGORY_LABELS: dict[str, str] = {"grammar": "Grammar", "tone": "Tone", "clarity": "Clarity"}
CATEGORY_HINTS: dict[str, str] = {
    "grammar": "Spelling, punctuation, agreement and tense",
    "tone": "Wording that suits the tone you choose",
    "clarity": "Wordy, vague or hard-to-follow phrasing",
}
STATUS_LABELS: dict[str, str] = {
    "pending": "Not decided",
    "accepted": "Accepted",
    "rejected": "Rejected",
}

# Short sample with one or two issues per category, so a first-time user can see every
# part of the workflow. Loading it never calls the API; the user still presses "Check text".
EXAMPLE_TEXT = (
    "Their going to present the quarterly results tomorrow, and me and him has prepared "
    "the slides. The figures was collected by the team over a period of several weeks in "
    "order to be able to show the trends. Honestly the numbers are kinda awesome however we "
    "still need to fix a few things before the meeting."
)


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


# ------------------------------------------------------------------ data shaping


def group_by_category(suggestions: Iterable[Suggestion]) -> list[tuple[str, list[Suggestion]]]:
    """Non-empty groups in the fixed category order; text order is kept inside a group."""
    groups: dict[str, list[Suggestion]] = {c: [] for c in CATEGORIES}
    for s in suggestions:
        groups.setdefault(s.category, []).append(s)
    return [(c, items) for c, items in groups.items() if items]


@dataclass(frozen=True)
class DecisionCounts:
    accepted: int
    rejected: int
    pending: int

    @property
    def total(self) -> int:
        return self.accepted + self.rejected + self.pending


def decision_counts(
    suggestions: Iterable[Suggestion], decisions: Mapping[str, str]
) -> DecisionCounts:
    statuses = [decisions.get(s.id, "pending") for s in suggestions]
    return DecisionCounts(
        accepted=statuses.count("accepted"),
        rejected=statuses.count("rejected"),
        pending=sum(1 for x in statuses if x not in ("accepted", "rejected")),
    )


# ------------------------------------------------------------------ fragments


def header_html() -> str:
    return (
        '<header class="gsc-header">'
        '<p class="gsc-eyebrow">Writing assistant</p>'
        '<h1 class="gsc-title">Intelligent Grammar &amp; Style Checker</h1>'
        '<p class="gsc-lede">Improves the grammar, tone and clarity of your writing with an LLM. '
        "Every change is a suggestion: you decide what to keep.</p>"
        "</header>"
    )


def word_count_html(words: int, limit: int) -> str:
    if words == 0:
        state = "empty"
    elif words > limit:
        state = "over"
    else:
        state = "ok"
    return (
        f'<p class="gsc-count gsc-count--{state}" aria-live="polite">'
        f"{words:,} / {limit:,} words</p>"
    )


def empty_state_html() -> str:
    steps = [
        ("Paste your text", "Up to the word limit shown under the box."),
        ("Choose what to check", "Grammar, tone and clarity, and the tone you are aiming for."),
        ("Review each suggestion", "Accept or reject changes, then copy or download the result."),
    ]
    items = "".join(
        f'<li class="gsc-step"><span class="gsc-step__num" aria-hidden="true">{i}</span>'
        f'<span><strong>{esc(title)}</strong><span class="gsc-step__body">{esc(body)}</span>'
        "</span></li>"
        for i, (title, body) in enumerate(steps, start=1)
    )
    return (
        '<section class="gsc-empty" aria-label="How it works">'
        f'<ol class="gsc-steps">{items}</ol>'
        '<p class="gsc-empty__note">Nothing is changed until you accept it. '
        "The original wording is kept wherever it is already correct.</p>"
        "</section>"
    )


def summary_html(result: CheckResult, decisions: Mapping[str, str]) -> str:
    counts = decision_counts(result.suggestions, decisions)
    chips = "".join(
        f'<span class="gsc-chip gsc-chip--{esc(c)}">{esc(CATEGORY_LABELS.get(c, c))} '
        f'<span class="gsc-chip__n">{len(items)}</span></span>'
        for c, items in group_by_category(result.suggestions)
    )
    status = (
        f'<span class="gsc-tally gsc-tally--accepted">{counts.accepted} accepted</span>'
        f'<span class="gsc-tally gsc-tally--rejected">{counts.rejected} rejected</span>'
        f'<span class="gsc-tally gsc-tally--pending">{counts.pending} to review</span>'
    )
    return (
        '<section class="gsc-summary" aria-label="Summary">'
        f'<p class="gsc-summary__head">{esc(plural(counts.total, "suggestion"))}</p>'
        f'<div class="gsc-summary__chips">{chips}</div>'
        f'<div class="gsc-summary__tally" aria-live="polite">{status}</div>'
        "</section>"
    )


def group_header_html(category: str, count: int) -> str:
    label = CATEGORY_LABELS.get(category, category)
    hint = CATEGORY_HINTS.get(category, "")
    return (
        f'<div class="gsc-group gsc-group--{esc(category)}">'
        f'<h3 class="gsc-group__title">{esc(label)} <span class="gsc-group__n">{count}</span></h3>'
        f'<p class="gsc-group__hint">{esc(hint)}</p>'
        "</div>"
    )


def card_html(s: Suggestion, status: str) -> str:
    status = status if status in STATUS_LABELS else "pending"
    replacement = (
        f"<ins>{esc(s.replacement)}</ins>"
        if s.replacement
        else '<ins class="gsc-diff__remove">(remove)</ins>'
    )
    return (
        f'<div class="gsc-card gsc-card--{esc(s.category)} gsc-card--{status}">'
        f'<p class="gsc-card__status gsc-status--{status}">{STATUS_LABELS[status]}</p>'
        '<p class="gsc-diff">'
        f'<span class="gsc-sr">Replace </span><del>{esc(s.original)}</del>'
        '<span class="gsc-diff__arrow" aria-hidden="true">→</span>'
        f'<span class="gsc-sr"> with </span>{replacement}'
        "</p>"
        f'<p class="gsc-why">{esc(s.explanation)}</p>'
        "</div>"
    )


def preview_legend_html() -> str:
    return (
        '<p class="gsc-legend">'
        '<span><mark class="gsc-legend__mark">highlighted</mark> not decided</span>'
        "<span><del>struck out</del> <ins>green</ins> accepted</span>"
        '<span><span class="gsc-legend__rejected">dotted</span> rejected (kept as written)</span>'
        "</p>"
    )


def clean_result_html() -> str:
    return (
        '<section class="gsc-clean" role="status">'
        '<span class="gsc-clean__icon" aria-hidden="true">✓</span>'
        "<div><p class='gsc-clean__title'>No issues found. Your text looks good.</p>"
        '<p class="gsc-clean__body">Correct text is left exactly as you wrote it.</p></div>'
        "</section>"
    )
