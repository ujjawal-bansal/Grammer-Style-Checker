"""Orchestration: validate input -> chunk -> call the model per chunk -> verify and locate
suggestions -> merge. Valid partial results are always preserved when a later chunk fails."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from core.apply import assign_ids, locate_suggestions, resolve_overlaps
from core.config import Settings
from core.errors import CheckerError, InputValidationError, InvalidOutputError, TokenLimitError
from core.llm_client import GroqClient
from core.schemas import (
    CheckOptions,
    CheckResult,
    Chunk,
    DroppedSuggestion,
    SectionIssue,
    Suggestion,
)

logger = logging.getLogger("grammar_checker.pipeline")

MAX_SPLIT_DEPTH = 1  # a chunk rejected for size is halved at most once
_MAX_CHARS_PER_WORD = 25  # guards against huge inputs without whitespace
_WORD_RE = re.compile(r"\S+")
_PARAGRAPH_GAP_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)\]]*$")


def count_words(text: str) -> int:
    return len(text.split())


def validate_input(text: str, settings: Settings) -> None:
    if not isinstance(text, str) or not text.strip():
        raise InputValidationError("Please enter some text to check.")
    if "\x00" in text:
        raise InputValidationError("The text contains unsupported control characters.")
    words = count_words(text)
    if words > settings.max_input_words:
        raise InputValidationError(
            f"The text has {words} words; the limit is {settings.max_input_words}. "
            "Please shorten it or check it in parts."
        )
    if len(text) > settings.max_input_words * _MAX_CHARS_PER_WORD:
        raise InputValidationError("The text is too long. Please shorten it or check it in parts.")


def _best_break(text: str, words: list[tuple[int, int]], i: int, j: int, max_words: int) -> int:
    """Pick where a chunk covering words[i:j] should end (exclusive index k <= j)."""
    floor = i + max(1, max_words // 2)  # avoid tiny chunks
    for k in range(j, floor, -1):  # prefer a paragraph break
        if _PARAGRAPH_GAP_RE.search(text[words[k - 1][1] : words[k][0]]):
            return k
    for k in range(j, floor, -1):  # then a sentence end
        if _SENTENCE_END_RE.search(text[words[k - 1][0] : words[k - 1][1]]):
            return k
    return j


def chunk_text(
    text: str, max_words: int, *, base_offset: int = 0, base_index: int = 0
) -> list[Chunk]:
    """Split into chunks of at most `max_words` words. Every chunk is an exact slice of
    `text`, and offsets are absolute (shifted by `base_offset`)."""
    words = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    chunks: list[Chunk] = []
    i = 0
    while i < len(words):
        j = min(i + max_words, len(words))
        if j < len(words):
            j = _best_break(text, words, i, j, max_words)
        start, end = words[i][0], words[j - 1][1]
        chunks.append(
            Chunk(base_index + len(chunks), base_offset + start, base_offset + end, text[start:end])
        )
        i = j
    return chunks


def split_chunk(chunk: Chunk) -> list[Chunk]:
    """Halve a chunk (used when the API reports it was too large). May return [chunk]."""
    n_words = count_words(chunk.text)
    if n_words < 2:
        return [chunk]
    parts = chunk_text(
        chunk.text,
        math.ceil(n_words / 2),
        base_offset=chunk.start,
        base_index=chunk.index,
    )
    # all parts belong to the same logical section, so they share its index
    return [Chunk(chunk.index, p.start, p.end, p.text) for p in parts]


class _Run:
    """State for one check. Chunks are processed sequentially (free-tier rate limits)."""

    def __init__(self, client: Any, options: CheckOptions) -> None:
        self.client = client
        self.options = options
        self.suggestions: list[Suggestion] = []
        self.dropped: list[DroppedSuggestion] = []
        self.issues: list[SectionIssue] = []
        self.stopped = False

    def _issue(self, chunk: Chunk, kind: str, message: str) -> None:
        self.issues.append(SectionIssue(chunk.index, chunk.start, chunk.end, kind, message))

    def process(self, chunk: Chunk, depth: int = 0) -> None:
        if self.stopped:
            self._issue(chunk, "skipped", "Not checked because an earlier section failed.")
            return
        try:
            parsed = self.client.request_suggestions(chunk.text, self.options)
        except TokenLimitError as err:
            parts = split_chunk(chunk) if depth < MAX_SPLIT_DEPTH else [chunk]
            if len(parts) > 1:
                for part in parts:
                    self.process(part, depth + 1)
            else:
                self._issue(chunk, "failed", err.user_message)
            return
        except InvalidOutputError as err:  # local to this chunk: keep going
            self._issue(chunk, "failed", err.user_message)
            return
        except CheckerError as err:  # rate limit, auth, timeout, server...: stop calling the API
            self._issue(chunk, "failed", err.user_message)
            self.stopped = True
            return

        found, dropped = locate_suggestions(chunk, parsed.valid)
        self.suggestions.extend(found)
        self.dropped.extend(dropped)
        for reason, count in (
            ("invalid", parsed.invalid_count),
            ("overflow", parsed.overflow_count),
        ):
            self.dropped.extend(DroppedSuggestion(reason, chunk.index) for _ in range(count))


def run_check(
    text: str,
    options: CheckOptions,
    settings: Settings,
    client: Any | None = None,
) -> CheckResult:
    """Check `text`. Raises InputValidationError for bad input (no API call is made).
    API failures never raise: they are recorded as section issues next to any valid
    suggestions obtained from the sections that did succeed."""
    validate_input(text, settings)
    client = client or GroqClient(settings)
    chunks = chunk_text(text, settings.max_chunk_words)

    run = _Run(client, options)
    for chunk in chunks:
        run.process(chunk)

    kept, overlaps = resolve_overlaps(run.suggestions)
    result = CheckResult(
        text=text,
        options=options,
        suggestions=assign_ids(kept),
        dropped=tuple(run.dropped + overlaps),
        issues=tuple(run.issues),
        total_chunks=len(chunks),
    )
    logger.info(
        "Checked %d chars in %d chunk(s): %d suggestions, %d unverified, %d section issue(s)",
        len(text),
        len(chunks),
        len(result.suggestions),
        result.unverified_count,
        len(result.issues),
    )
    return result
