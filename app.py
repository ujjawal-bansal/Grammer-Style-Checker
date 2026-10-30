"""Streamlit UI for the Intelligent Grammar & Style Checker.

Thin by design: validation, API access, verification and text editing all live in `core/`,
and the HTML fragments and styles live in `ui/`. The API is only called when the user asks
for a check; accepting or rejecting a suggestion just re-renders from the stored result.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from core import pipeline
from core.apply import apply_accepted, render_html
from core.config import Settings, load_settings
from core.errors import CheckerError, ConfigError
from core.schemas import CATEGORIES, DROP_REASONS, TONES, CheckOptions, CheckResult, Suggestion
from ui import components as ui

logger = logging.getLogger("grammar_checker.app")

STYLES = Path(__file__).resolve().parent / "ui" / "styles.css"
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


def _clear_results() -> None:
    st.session_state["result"] = None
    st.session_state["decisions"] = {}
    st.session_state["ui_error"] = None


def _request_check() -> None:
    st.session_state["check_requested"] = True


def _load_example() -> None:
    st.session_state["input_text"] = ui.EXAMPLE_TEXT
    _clear_results()


def _new_check() -> None:
    st.session_state["input_text"] = ""
    _clear_results()


def _run_check(text: str, selected: list[str], tone: str, settings: Settings) -> None:
    _clear_results()
    progress_slot = st.empty()

    def on_progress(done: int, total: int) -> None:
        if total > 1:  # a single request needs no progress bar, the spinner says enough
            label = f"Checking section {min(done + 1, total)} of {total}…"
            progress_slot.progress(done / total, text=label if done < total else "Finishing…")

    try:
        options = CheckOptions(categories=frozenset(selected), tone=tone)
        with st.spinner("Checking your text…", show_time=True):
            st.session_state["result"] = pipeline.run_check(
                text, options, settings, progress=on_progress
            )
    except CheckerError as err:
        st.session_state["ui_error"] = err.user_message
    except Exception as err:  # never show raw exception details (they could echo secrets)
        logger.error("Unexpected error: %s", type(err).__name__)
        st.session_state["ui_error"] = GENERIC_ERROR
    finally:
        progress_slot.empty()


# ------------------------------------------------------------------ composer


def _render_composer(settings: Settings) -> tuple[str, list[str], str, bool]:
    """Text box, check options and actions. Returns (text, categories, tone, can_check)."""
    with st.container(key="composer"):
        text = st.text_area(
            "Your text",
            key="input_text",
            height=240,
            placeholder="Paste or type the text you want to improve…",
        )
        options_col, tone_col = st.columns([3, 2], vertical_alignment="bottom")
        with options_col:
            selected = (
                st.pills(
                    "Checks",
                    options=list(CATEGORIES),
                    selection_mode="multi",
                    default=list(CATEGORIES),
                    format_func=lambda c: ui.CATEGORY_LABELS.get(c, c),
                    key="checks",
                    help="Grammar: spelling, punctuation, agreement and tense. "
                    "Tone: wording that suits the tone you choose. "
                    "Clarity: wordy, vague or hard-to-follow phrasing.",
                )
                or []
            )
        with tone_col:
            tone = st.selectbox(
                "Target tone",
                TONES,
                key="tone",
                format_func=str.capitalize,
                disabled="tone" not in selected,
                help="Used by the tone check only.",
            )

        words = len(text.split())
        too_long = words > settings.max_input_words
        can_check = bool(text.strip()) and not too_long and bool(selected)

        status_col, actions_col = st.columns([2, 3], vertical_alignment="center")
        with status_col:
            st.html(ui.word_count_html(words, settings.max_input_words))
        with actions_col, st.container(horizontal=True, horizontal_alignment="right"):
            st.button(
                "Try an example",
                key="example",
                type="tertiary",
                icon=":material/edit_note:",
                on_click=_load_example,
                help="Fill the box with a short sample text. Nothing is sent until you check it.",
            )
            has_something = bool(text) or st.session_state.get("result") is not None
            st.button(
                "New check",
                key="new_check",
                icon=":material/restart_alt:",
                on_click=_new_check,
                disabled=not (has_something or st.session_state.get("ui_error")),
                help="Clear the text and the results.",
            )
            st.button(
                "Check text",
                key="check",
                type="primary",
                icon=":material/spellcheck:",
                on_click=_request_check,
                disabled=not can_check,
            )

    if not text.strip():
        st.html('<p class="gsc-hint">Paste or type some text to enable checking.</p>')
    if too_long:
        st.warning(
            f"The text has {words} words; the limit is {settings.max_input_words}. "
            "Please shorten it or check it in parts.",
            icon=":material/warning:",
        )
    if not selected:
        st.warning(
            "Select at least one category to check: grammar, tone or clarity.",
            icon=":material/warning:",
        )
    return text, selected, tone, can_check


# ------------------------------------------------------------------ results


def _try_again_button(can_check: bool) -> None:
    st.button(
        "Try again",
        key="retry",
        icon=":material/refresh:",
        on_click=_request_check,
        disabled=not can_check,
    )


def _render_issues(result: CheckResult, can_check: bool) -> None:
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
        parts = [f"{ui.plural(len(failed), 'section')} could not be checked"]
        if skipped:
            parts.append(f"{ui.plural(len(skipped), 'section')} skipped")
        headline = (
            "Partial result: "
            + " and ".join(parts)
            + ". The suggestions below come from the sections that were checked."
        )
        st.warning(headline + "\n\n" + "\n".join(lines), icon=":material/warning:")
    else:
        st.error("Your text could not be checked.\n\n" + "\n".join(lines), icon=":material/error:")
        _try_again_button(can_check)


def _render_unverified(result: CheckResult) -> None:
    counts = result.drop_counts()
    problems = [
        f"{n} because {DROP_REASONS[reason]}" for reason, n in counts.items() if reason != "no_op"
    ]
    if problems:
        st.info(
            f"{ui.plural(result.unverified_count, 'suggestion')} from the model could not be "
            "verified against your text and are not shown: " + "; ".join(problems) + ".",
            icon=":material/info:",
        )
    if counts.get("no_op"):
        st.caption(
            f"{ui.plural(counts['no_op'], 'suggestion')} proposed no change and was ignored."
        )


def _render_suggestion(s: Suggestion, status: str) -> None:
    with st.container(border=True, key=f"card_{s.id}"):
        st.html(ui.card_html(s, status))
        with st.container(horizontal=True, gap="small"):
            st.button(
                "Accept",
                key=f"accept_{s.id}",
                on_click=_decide,
                args=(s.id, "accepted"),
                type="primary" if status == "accepted" else "secondary",
                icon=":material/check:",
                help="Use this change in the corrected text.",
            )
            st.button(
                "Reject",
                key=f"reject_{s.id}",
                on_click=_decide,
                args=(s.id, "rejected"),
                type="primary" if status == "rejected" else "secondary",
                icon=":material/close:",
                help="Keep your original wording.",
            )


def _render_result(result: CheckResult, current_text: str, can_check: bool) -> None:
    if current_text != result.text:
        st.info(
            "The text box changed after this check. The results below belong to the text that "
            "was checked; press **Check text** again to update them.",
            icon=":material/info:",
        )
    _render_issues(result, can_check)
    _render_unverified(result)

    if not result.suggestions:
        if result.unverified_count:
            # The model did report problems, they just could not be verified: never call the
            # text clean in that case.
            st.warning(
                "No suggestions can be shown for this text, so it has not been confirmed as "
                "correct. Checking again often helps.",
                icon=":material/warning:",
            )
            _try_again_button(can_check)
        elif not result.issues:
            st.html(ui.clean_result_html())
        return

    decisions: dict[str, str] = st.session_state.get("decisions", {})
    counts = ui.decision_counts(result.suggestions, decisions)
    accepted = [s.id for s in result.suggestions if decisions.get(s.id) == "accepted"]

    st.html(ui.summary_html(result, decisions))
    with st.container(horizontal=True, gap="small", key="bulk_actions"):
        st.button(
            "Accept all",
            key="accept_all",
            icon=":material/done_all:",
            on_click=_decide_all,
            args=("accepted",),
        )
        st.button(
            "Reject all",
            key="reject_all",
            icon=":material/block:",
            on_click=_decide_all,
            args=("rejected",),
        )
        st.button(
            "Reset decisions",
            key="reset_all",
            type="tertiary",
            icon=":material/undo:",
            on_click=_decide_all,
            args=(None,),
        )

    review_col, output_col = st.columns([11, 10], gap="large")
    with review_col:
        st.subheader("Suggestions", anchor=False)
        for category, items in ui.group_by_category(result.suggestions):
            st.html(ui.group_header_html(category, len(items)))
            for s in items:
                _render_suggestion(s, decisions.get(s.id, "pending"))

    with output_col:
        try:
            preview = render_html(result.text, result.suggestions, decisions)
            corrected = apply_accepted(result.text, result.suggestions, accepted)
        except CheckerError as err:
            st.error(err.user_message, icon=":material/error:")
            return
        with st.container(key="preview_panel"):
            st.subheader("Preview", anchor=False)
            st.html(ui.preview_legend_html())
            st.html(preview)
        with st.container(key="corrected_panel"):
            st.subheader("Corrected text", anchor=False)
            st.caption(
                f"{counts.accepted} accepted, {counts.rejected} rejected, "
                f"{counts.pending} not decided. Only accepted changes are applied."
            )
            st.code(corrected, language=None, wrap_lines=True)
            with st.container(horizontal=True, vertical_alignment="center"):
                st.download_button(
                    "Download .txt",
                    data=corrected,
                    file_name="corrected.txt",
                    mime="text/plain",
                    key="download_corrected",
                    icon=":material/download:",
                    on_click="ignore",
                )
                st.caption("Or use the copy button in the corner of the text box.")


# ------------------------------------------------------------------ page


def main() -> None:
    st.set_page_config(page_title="Grammar & Style Checker", page_icon="✍️", layout="wide")
    st.html(STYLES)
    st.html(ui.header_html())

    try:
        settings = load_settings()
    except ConfigError as err:
        st.error(err.user_message, icon=":material/key_off:")
        st.info("Create a `.env` file from `.env.example` and restart the app.")
        st.stop()

    text, selected, tone, can_check = _render_composer(settings)

    if st.session_state.pop("check_requested", False) and can_check:
        _run_check(text, selected, tone, settings)

    error = st.session_state.get("ui_error")
    result: CheckResult | None = st.session_state.get("result")
    if error:
        st.error(error, icon=":material/error:")
        _try_again_button(can_check)
    elif result is None:
        st.html(ui.empty_state_html())
    else:
        _render_result(result, text, can_check)

    st.html(
        f'<footer class="gsc-footer">Model: {ui.esc(settings.model)} via the Groq API · up to '
        f"{settings.max_input_words:,} words per check · your text is sent only when you press "
        "Check text.</footer>"
    )


main()
