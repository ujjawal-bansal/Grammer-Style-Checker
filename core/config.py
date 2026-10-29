"""Configuration from environment variables / a local `.env` file.

The real environment wins over `.env`. The `.env` file is read without mutating
`os.environ`. The API key lives only in `Settings.api_key`, which is excluded from
`repr()`/`str()` so it cannot leak into logs or tracebacks by accident.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from core.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_MODEL = "openai/gpt-oss-20b"
REASONING_EFFORTS = ("low", "medium", "high")

ENV_VARS = (
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "GROQ_TIMEOUT_SECONDS",
    "GROQ_MAX_RETRIES",
    "GROQ_MAX_RETRY_WAIT_SECONDS",
    "GROQ_REASONING_EFFORT",
    "GROQ_MAX_COMPLETION_TOKENS",
    "MAX_INPUT_WORDS",
    "MAX_CHUNK_WORDS",
)

_USE_DEFAULT: Any = object()  # sentinel: "use DEFAULT_ENV_FILE as it is at call time"
_MODEL_RE = re.compile(r"^[A-Za-z0-9._:/\-]{1,100}$")
_PLACEHOLDER_KEYS = {"your_groq_api_key_here", "your-api-key", "changeme", "xxx"}


@dataclass(frozen=True, repr=False)
class Settings:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 30.0
    max_retries: int = 2
    max_retry_wait_seconds: float = 10.0
    reasoning_effort: str | None = "low"
    max_completion_tokens: int = 2500
    max_input_words: int = 1000
    max_chunk_words: int = 400

    def safe_summary(self) -> dict[str, object]:
        """Everything except the API key; safe to log or display."""
        return {
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "max_retry_wait_seconds": self.max_retry_wait_seconds,
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_completion_tokens,
            "max_input_words": self.max_input_words,
            "max_chunk_words": self.max_chunk_words,
        }

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.safe_summary().items())
        return f"Settings({fields})"

    __str__ = __repr__


def _text(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _number(
    env: Mapping[str, str], name: str, default: float, lo: float, hi: float, *, integer: bool
) -> float:
    raw = _text(env, name)
    if raw is None:
        return default
    kind = "an integer" if integer else "a number"
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        raise ConfigError(
            f"{name} must be {kind} between {lo:g} and {hi:g} (got {raw!r})."
        ) from None
    if not lo <= value <= hi:
        raise ConfigError(f"{name} must be {kind} between {lo:g} and {hi:g} (got {raw!r}).")
    return value


def _reasoning_effort(env: Mapping[str, str]) -> str | None:
    if "GROQ_REASONING_EFFORT" not in env:
        return "low"
    raw = (env["GROQ_REASONING_EFFORT"] or "").strip().lower()
    if raw in ("", "none", "off"):
        return None
    if raw not in REASONING_EFFORTS:
        allowed = ", ".join(REASONING_EFFORTS)
        raise ConfigError(f"GROQ_REASONING_EFFORT must be one of: {allowed} (or empty to disable).")
    return raw


def load_settings(
    environ: Mapping[str, str] | None = None,
    env_file: Path | None | Any = _USE_DEFAULT,
) -> Settings:
    """Build validated settings.

    With `environ=None` (normal use) values come from the process environment, falling
    back to `env_file`. Passing an explicit `environ` (tests) uses only that mapping.
    """
    if env_file is _USE_DEFAULT:
        env_file = DEFAULT_ENV_FILE
    if environ is None:
        merged: dict[str, str] = {}
        if env_file is not None and Path(env_file).is_file():
            merged.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
        merged.update(os.environ)
        env: Mapping[str, str] = merged
    else:
        env = environ

    api_key = _text(env, "GROQ_API_KEY")
    if api_key is None:
        raise ConfigError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key."
        )
    if api_key.lower() in _PLACEHOLDER_KEYS or re.search(r"\s", api_key):
        raise ConfigError("GROQ_API_KEY looks invalid (placeholder text or contains whitespace).")

    model = _text(env, "GROQ_MODEL") or DEFAULT_MODEL
    if not _MODEL_RE.match(model):
        raise ConfigError("GROQ_MODEL contains unsupported characters.")

    return Settings(
        api_key=api_key,
        model=model,
        timeout_seconds=float(_number(env, "GROQ_TIMEOUT_SECONDS", 30.0, 1, 300, integer=False)),
        max_retries=int(_number(env, "GROQ_MAX_RETRIES", 2, 0, 5, integer=True)),
        max_retry_wait_seconds=float(
            _number(env, "GROQ_MAX_RETRY_WAIT_SECONDS", 10.0, 0, 120, integer=False)
        ),
        reasoning_effort=_reasoning_effort(env),
        max_completion_tokens=int(
            _number(env, "GROQ_MAX_COMPLETION_TOKENS", 2500, 64, 131072, integer=True)
        ),
        max_input_words=int(_number(env, "MAX_INPUT_WORDS", 1000, 1, 20000, integer=True)),
        max_chunk_words=int(_number(env, "MAX_CHUNK_WORDS", 400, 20, 5000, integer=True)),
    )
