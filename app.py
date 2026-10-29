"""Streamlit UI for the Intelligent Grammar & Style Checker.

Thin by design: validation, API access, verification and text editing all live in `core/`.
The API is only called when the user presses "Check text"; accepting or rejecting a
suggestion just re-renders from the stored result.
"""

from __future__ import annotations

import html
import logging

import streamlit as st

from core import pipeline
from core.apply import apply_accepted, render_html
from core.config import Settings, load_settings
from core.errors import CheckerError, ConfigError
from core.schemas import (
    CATEGORIES,
    DROP_REASONS,
    TONES,
    CheckOptions,
    CheckResult,
    Suggestion,
)

logger = logging.getLogger("grammar_checker.app")

LABELS = {"grammar": "Grammar", "tone": "Tone", "clarity": "Clarity"}
GENERIC_ERROR = "Something went wrong while checking your text. Please try again."


# ------------------------------------------------------------------ state helpers


def _decide(suggestion_id: str, status: str) -> None:
    st.session_state.setdefault("decisions", {})[suggestion_id] = status


def _decide_all(status: str | None) -> None:
    result: CheckResult | None = st.session_state.get("result")
    if result is None:
        return
    st.session_state["decisions"] = (
        {s.id: status for s in result.suggestions} if status is not None else {}
    )


def _run_check(text: str, selected: list[str], tone: str, settings: Settings) -> None:
    st.session_state["ui_error"] = None
    st.session_state["result"] = None
    st.session_state["decisions"] = {}
    try:
        options = CheckOptions(categories=frozenset(selected), tone=tone)
        with st.spinner("Checking your text..."):
            st.session_state["result"] = pipeline.run_check(text, options, settings)
    except CheckerError as err:
        st.session_state["ui_error"] = err.user_message
    except Exception as err:  # never show raw exception details (they could echo secrets)
        logger.error("Unexpected error: %s", type(err).__name__)
        st.session_state["ui_error"] = GENERIC_ERROR


# ------------------------------------------------------------------ rendering


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _render_issues(result: CheckResult) -> None:
    if not result.issues:
        return
    failed = [i for i in result.issues if i.kind == "failed"]
    skipped = [i for i in result.issues if i.kind == "skipped"]
    lines = []
    for issue in result.issues:
        first = len(result.text[: issue.start].split()) + 1
        last = first + len(result.text[issue.start : issue.end].split()) - 1
        lines.append(f"- Words {first}-{last}: {issue.message}")
    if result.suggestions:
        parts = [f"{_plural(len(failed), 'section')} could not be checked"]
        if skipped:
            parts.append(f"{_plural(len(skipped), 'section')} skipped")
        headline = (
            "Partial result: "
            + " and ".join(parts)
            + ". The suggestions below come from the sections that were checked."
        )
        st.warning(headline + "\n\n" + "\n".join(lines))
    else:
        st.error("Your text could not be checked.\n\n" + "\n".join(lines))


def _render_unverified(result: CheckResult) -> None:
    counts = result.drop_counts()
    problems = [
        f"{n} because {DROP_REASONS[reason]}" for reason, n in counts.items() if reason != "no_op"
    ]
    if problems:
        st.info(
            f"{_plural(result.unverified_count, 'suggestion')} from the model could not be "
            "verified against your text and are not shown: " + "; ".join(problems) + "."
        )
    if counts.get("no_op"):
        st.caption(f"{_plural(counts['no_op'], 'suggestion')} proposed no change and was ignored.")


def _card_html(s: Suggestion) -> str:
    esc = html.escape
    replacement = esc(s.replacement) if s.replacement else "(remove)"
    return (
        f'<div style="font-size:0.75rem;opacity:0.7;text-transform:uppercase;">'
        f"{esc(LABELS.get(s.category, s.category))}</div>"
        f'<div style="margin:4px 0;"><del style="background:#ffd6d6;color:#7a0000;padding:0 2px;">'
        f"{esc(s.original)}</del> &rarr; "
        f'<ins style="background:#d4f5d4;color:#064d06;padding:0 2px;">{replacement}</ins></div>'
        f'<div style="opacity:0.85;">{esc(s.explanation)}</div>'
    )


def _render_suggestion(s: Suggestion, status: str) -> None:
    with st.container(border=True):
        text_col, action_col = st.columns([4, 2])
        with text_col:
            st.html(_card_html(s))
        with action_col:
            accept, reject = st.columns(2)
            accept.button(
                "Accept",
                key=f"accept_{s.id}",
                on_click=_decide,
                args=(s.id, "accepted"),
                type="primary" if status == "accepted" else "secondary",
                width="stretch",
            )
            reject.button(
                "Reject",
                key=f"reject_{s.id}",
                on_click=_decide,
                args=(s.id, "rejected"),
                type="primary" if status == "rejected" else "secondary",
                width="stretch",
            )


def _render_result(result: CheckResult, current_text: str) -> None:
    if current_text != result.text:
        st.info(
            "The text box changed after this check. The results below belong to the text that "
            "was checked; press **Check text** again to update them."
        )
    _render_issues(result)
    _render_unverified(result)

    if not result.suggestions:
        if not result.issues:
            st.success("No issues found. Your text looks good.")
        return

    decisions: dict[str, str] = st.session_state.get("decisions", {})
    accepted = [s.id for s in result.suggestions if decisions.get(s.id) == "accepted"]
    rejected = [s.id for s in result.suggestions if decisions.get(s.id) == "rejected"]

    st.subheader(f"Suggestions ({len(result.suggestions)})")
    bulk = st.columns([1, 1, 1, 3])
    bulk[0].button("Accept all", key="accept_all", on_click=_decide_all, args=("accepted",))
    bulk[1].button("Reject all", key="reject_all", on_click=_decide_all, args=("rejected",))
    bulk[2].button("Reset", key="reset_all", on_click=_decide_all, args=(None,))

    for s in result.suggestions:
        _render_suggestion(s, decisions.get(s.id, "pending"))

    st.subheader("Preview")
    st.caption("Yellow = pending, red/green = accepted change, dotted underline = rejected.")
    try:
        st.html(render_html(result.text, result.suggestions, decisions))
        corrected = apply_accepted(result.text, result.suggestions, accepted)
    except CheckerError as err:
        st.error(err.user_message)
        return

    pending = len(result.suggestions) - len(accepted) - len(rejected)
    st.subheader("Corrected text")
    st.caption(f"{len(accepted)} accepted, {len(rejected)} rejected, {pending} pending.")
    st.code(corrected, language=None, wrap_lines=True)
    st.download_button(
        "Download corrected text",
        data=corrected,
        file_name="corrected.txt",
        mime="text/plain",
        key="download_corrected",
    )


# ------------------------------------------------------------------ page


def main() -> None:
    st.set_page_config(page_title="Grammar & Style Checker", page_icon="✍️", layout="wide")
    st.title("Intelligent Grammar & Style Checker")
    st.caption(
        "Improves grammar, tone and clarity with an LLM (Groq). "
        "Review every suggestion before you accept it."
    )

    try:
        settings = load_settings()
    except ConfigError as err:
        st.error(err.user_message)
        st.info("Create a `.env` file from `.env.example` and restart the app.")
        st.stop()

    with st.sidebar:
        st.header("Options")
        selected = [
            c for c in CATEGORIES if st.checkbox(LABELS[c], value=True, key=f"category_{c}")
        ]
        tone = st.selectbox("Target tone", TONES, key="tone", disabled="tone" not in selected)
        st.divider()
        st.caption(f"Model: {settings.model}")
        st.caption(f"Limit: {settings.max_input_words} words per check")

    text = st.text_area(
        "Your text", height=260, key="input_text", placeholder="Paste or type your text here..."
    )
    words = len(text.split())
    too_long = words > settings.max_input_words
    st.caption(f"{words} / {settings.max_input_words} words")
    if too_long:
        st.warning(
            f"The text has {words} words; the limit is {settings.max_input_words}. "
            "Please shorten it or check it in parts."
        )
    if not selected:
        st.warning("Select at least one category in the sidebar.")

    disabled = not text.strip() or too_long or not selected
    if st.button("Check text", type="primary", disabled=disabled, key="check"):
        _run_check(text, selected, tone, settings)

    if st.session_state.get("ui_error"):
        st.error(st.session_state["ui_error"])
    result: CheckResult | None = st.session_state.get("result")
    if result is not None:
        _render_result(result, text)


main()
