"""Request options, the strict JSON schema sent to Groq, and validation of the model output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.errors import InputValidationError, InvalidOutputError

CATEGORIES: tuple[str, ...] = ("grammar", "tone", "clarity")
TONES: tuple[str, ...] = ("professional", "formal", "friendly", "casual", "confident")
DEFAULT_TONE = "professional"
MAX_SUGGESTIONS_PER_REQUEST = 40


@dataclass(frozen=True)
class CheckOptions:
    categories: frozenset[str] = frozenset(CATEGORIES)
    tone: str = DEFAULT_TONE

    def __post_init__(self) -> None:
        categories = frozenset(self.categories)
        object.__setattr__(self, "categories", categories)
        if not categories:
            raise InputValidationError("Select at least one category to check.")
        if not categories <= set(CATEGORIES):
            raise InputValidationError("Unknown category selected.")
        if self.tone not in TONES:
            raise InputValidationError("Unknown target tone selected.")

    @property
    def ordered_categories(self) -> tuple[str, ...]:
        return tuple(c for c in CATEGORIES if c in self.categories)


class RawSuggestion(BaseModel):
    """One suggestion exactly as returned by the model (not yet located in the text)."""

    model_config = ConfigDict(extra="forbid")

    category: Literal["grammar", "tone", "clarity"]
    original: str = Field(min_length=1, max_length=2000)
    replacement: str = Field(max_length=4000)
    explanation: str = Field(min_length=1, max_length=1000)
    context_before: str = Field(max_length=200)


_SUGGESTION_FIELDS = ("category", "original", "replacement", "explanation", "context_before")


def build_response_schema(categories: frozenset[str] | tuple[str, ...]) -> dict[str, Any]:
    """Strict-mode `json_schema` payload for Groq structured outputs.

    Follows the documented strict-mode rules: every property is required, every object has
    `additionalProperties: false`, and only plain types / enum / array are used.
    """
    allowed = [c for c in CATEGORIES if c in set(categories)]
    if not allowed:
        raise InputValidationError("Select at least one category to check.")
    properties: dict[str, Any] = {
        "category": {"type": "string", "enum": allowed},
        "original": {"type": "string"},
        "replacement": {"type": "string"},
        "explanation": {"type": "string"},
        "context_before": {"type": "string"},
    }
    return {
        "name": "style_suggestions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "suggestions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": properties,
                        "required": list(_SUGGESTION_FIELDS),
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["suggestions"],
            "additionalProperties": False,
        },
    }


@dataclass(frozen=True)
class ParsedOutput:
    valid: list[RawSuggestion]
    invalid_count: int = 0  # items that failed validation (reported to the user, not hidden)
    overflow_count: int = 0  # items beyond MAX_SUGGESTIONS_PER_REQUEST


def parse_llm_output(content: str, categories: frozenset[str]) -> ParsedOutput:
    """Validate the model's JSON. Whole-response problems raise `InvalidOutputError`
    (retryable); individual bad items are counted so they can be reported."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        raise InvalidOutputError() from None
    if not isinstance(data, dict) or not isinstance(data.get("suggestions"), list):
        raise InvalidOutputError()

    items: list[Any] = data["suggestions"]
    valid: list[RawSuggestion] = []
    invalid = 0
    for item in items[:MAX_SUGGESTIONS_PER_REQUEST]:
        try:
            suggestion = RawSuggestion.model_validate(item)
        except ValidationError:
            invalid += 1
            continue
        if suggestion.category not in categories:
            invalid += 1
            continue
        valid.append(suggestion)
    overflow = max(0, len(items) - MAX_SUGGESTIONS_PER_REQUEST)
    return ParsedOutput(valid=valid, invalid_count=invalid, overflow_count=overflow)


# --------------------------------------------------------------------------------------
# Result models (immutable). All offsets are absolute character offsets into the full text.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Suggestion:
    id: str
    category: str
    original: str
    replacement: str
    explanation: str
    start: int
    end: int
    chunk_index: int


# Why a model suggestion could not be turned into a verified suggestion.
DROP_REASONS: dict[str, str] = {
    "not_found": "the quoted text was not found in your text",
    "mid_word": "the quoted text only matched inside a longer word",
    "ambiguous": "the quoted text appears several times and the location is unclear",
    "overlap": "it overlapped another suggestion",
    "invalid": "the model's output for it was malformed",
    "overflow": "the model returned more suggestions than the app handles",
    "no_op": "it proposed no change",
}


@dataclass(frozen=True)
class DroppedSuggestion:
    reason: str
    chunk_index: int | None = None
    category: str | None = None
    original: str | None = None  # shortened preview, if known


@dataclass(frozen=True)
class SectionIssue:
    """A part of the text that was not (fully) checked."""

    chunk_index: int
    start: int
    end: int
    kind: str  # "failed" | "skipped"
    message: str


@dataclass(frozen=True)
class CheckResult:
    text: str
    options: CheckOptions
    suggestions: tuple[Suggestion, ...]
    dropped: tuple[DroppedSuggestion, ...]
    issues: tuple[SectionIssue, ...]
    total_chunks: int

    @property
    def is_partial(self) -> bool:
        return bool(self.issues)

    @property
    def unverified_count(self) -> int:
        """Suggestions that could not be verified or applied (no-ops are not counted)."""
        return sum(1 for d in self.dropped if d.reason != "no_op")

    def drop_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.dropped:
            counts[d.reason] = counts.get(d.reason, 0) + 1
        return counts
