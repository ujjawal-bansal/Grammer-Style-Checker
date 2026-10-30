"""Everything that touches text offsets: locating suggestions, resolving conflicts,
applying accepted edits deterministically, and rendering a safe HTML preview.

Safety properties (all covered by tests):
- A suggestion is only kept if its `original` text is found *exactly* in the text, on word
  boundaries, at exactly one candidate location.
- Overlapping suggestions can never both survive.
- Edits are applied left-to-right from the original string, never by re-searching, so the
  result does not depend on the order in which suggestions were accepted.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Mapping
from dataclasses import replace

from core.errors import StaleSuggestionError
from core.schemas import Chunk, DroppedSuggestion, RawSuggestion, Suggestion

PREVIEW_CHARS = 60


def _preview(value: str) -> str:
    value = " ".join(value.split())
    return value if len(value) <= PREVIEW_CHARS else value[: PREVIEW_CHARS - 1] + "…"


def find_occurrences(haystack: str, needle: str) -> list[int]:
    """Start offsets of every (possibly overlapping) occurrence of `needle`."""
    positions: list[int] = []
    if not needle:
        return positions
    pos = haystack.find(needle)
    while pos != -1:
        positions.append(pos)
        pos = haystack.find(needle, pos + 1)
    return positions


def _on_word_boundaries(text: str, start: int, end: int) -> bool:
    """False if the span cuts through a word (e.g. 'the' inside 'another')."""
    if start > 0 and text[start].isalnum() and text[start - 1].isalnum():
        return False
    return not (end < len(text) and text[end - 1].isalnum() and text[end].isalnum())


def _context_matches(text: str, start: int, context: str) -> bool:
    if not context.strip():
        return False
    prefix = text[:start]
    return prefix.endswith(context) or prefix.rstrip().endswith(context.strip())


def locate_in_text(text: str, raw: RawSuggestion) -> tuple[int, int] | str:
    """Return (start, end) offsets within `text`, or a drop-reason string."""
    candidates = find_occurrences(text, raw.original)
    if not candidates:
        return "not_found"
    n = len(raw.original)
    on_words = [s for s in candidates if _on_word_boundaries(text, s, s + n)]
    if not on_words:
        return "mid_word"
    if len(on_words) == 1:
        return on_words[0], on_words[0] + n
    by_context = [s for s in on_words if _context_matches(text, s, raw.context_before)]
    if len(by_context) == 1:
        return by_context[0], by_context[0] + n
    return "ambiguous"


def locate_suggestions(
    chunk: Chunk, raws: Iterable[RawSuggestion]
) -> tuple[list[Suggestion], list[DroppedSuggestion]]:
    """Turn raw model suggestions into verified suggestions with absolute offsets."""
    found: list[Suggestion] = []
    dropped: list[DroppedSuggestion] = []
    for raw in raws:
        if raw.original == raw.replacement:
            reason: str | None = "no_op"
            span: tuple[int, int] | None = None
        else:
            located = locate_in_text(chunk.text, raw)
            reason, span = (located, None) if isinstance(located, str) else (None, located)
        if reason is not None or span is None:
            dropped.append(
                DroppedSuggestion(
                    reason=reason or "not_found",
                    chunk_index=chunk.index,
                    category=raw.category,
                    original=_preview(raw.original),
                )
            )
            continue
        found.append(
            Suggestion(
                id="",
                category=raw.category,
                original=raw.original,
                replacement=raw.replacement,
                explanation=raw.explanation,
                start=chunk.start + span[0],
                end=chunk.start + span[1],
                chunk_index=chunk.index,
            )
        )
    return found, dropped


def resolve_overlaps(
    suggestions: Iterable[Suggestion],
) -> tuple[list[Suggestion], list[DroppedSuggestion]]:
    """Keep a non-overlapping subset, deterministically (earliest start, then shortest)."""
    kept: list[Suggestion] = []
    dropped: list[DroppedSuggestion] = []
    last_end = -1
    for s in sorted(suggestions, key=lambda x: (x.start, x.end)):
        if s.start < last_end:
            dropped.append(
                DroppedSuggestion("overlap", s.chunk_index, s.category, _preview(s.original))
            )
            continue
        kept.append(s)
        last_end = s.end
    return kept, dropped


def assign_ids(suggestions: Iterable[Suggestion]) -> tuple[Suggestion, ...]:
    ordered = sorted(suggestions, key=lambda x: (x.start, x.end))
    return tuple(replace(s, id=f"s{i}") for i, s in enumerate(ordered, start=1))


def apply_accepted(
    text: str, suggestions: Iterable[Suggestion], accepted_ids: Iterable[str]
) -> str:
    """Return `text` with exactly the accepted suggestions applied.

    Raises StaleSuggestionError if an accepted suggestion no longer matches the text or
    overlaps another accepted one, instead of silently producing corrupted output.
    """
    wanted = set(accepted_ids)
    chosen = sorted((s for s in suggestions if s.id in wanted), key=lambda s: (s.start, s.end))
    pieces: list[str] = []
    cursor = 0
    for s in chosen:
        if s.start < cursor or s.end > len(text) or text[s.start : s.end] != s.original:
            raise StaleSuggestionError()
        pieces.append(text[cursor : s.start])
        pieces.append(s.replacement)
        cursor = s.end
    pieces.append(text[cursor:])
    return "".join(pieces)


# Inline styles so the preview reads correctly even without the app's stylesheet. The colours
# match ui/styles.css and keep text contrast at WCAG AA or better on each background.
_STYLE_PENDING = "background:#fbecc4;color:#23201a;border-radius:3px;padding:0 2px;"
_STYLE_DEL = (
    "background:#f8e3df;color:#94382e;text-decoration:line-through;border-radius:3px;padding:0 2px;"
)
_STYLE_INS = (
    "background:#e3efe5;color:#2d6639;text-decoration:none;border-radius:3px;padding:0 2px;"
)
_STYLE_REJECTED = "border-bottom:1.5px dotted #665d4d;"


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def render_html(text: str, suggestions: Iterable[Suggestion], decisions: Mapping[str, str]) -> str:
    """HTML preview of the original text with suggestions marked. Every piece of user or
    model text is HTML-escaped; the only markup is what this function emits."""
    parts: list[str] = []
    cursor = 0
    for s in sorted(suggestions, key=lambda x: (x.start, x.end)):
        if s.start < cursor or text[s.start : s.end] != s.original:
            raise StaleSuggestionError()
        parts.append(_esc(text[cursor : s.start]))
        status = decisions.get(s.id, "pending")
        title = _esc(f"{s.category}: {s.explanation}")
        if status == "accepted":
            parts.append(
                f'<del style="{_STYLE_DEL}" title="{title}">{_esc(s.original)}</del>'
                f'<ins style="{_STYLE_INS}" title="{title}">{_esc(s.replacement)}</ins>'
            )
        elif status == "rejected":
            parts.append(f'<span style="{_STYLE_REJECTED}">{_esc(s.original)}</span>')
        else:
            parts.append(
                f'<mark style="{_STYLE_PENDING}" title="{title}">{_esc(s.original)}</mark>'
            )
        cursor = s.end
    parts.append(_esc(text[cursor:]))
    return f'<div style="white-space:pre-wrap;line-height:1.7;">{"".join(parts)}</div>'
