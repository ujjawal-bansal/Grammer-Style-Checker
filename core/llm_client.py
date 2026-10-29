"""Groq integration: strict structured output, timeout, bounded retries, clear 429 handling.

Guarantees:
- Only the configured model is ever called. There is no fallback model or provider.
- The SDK's own automatic retries are disabled (`max_retries=0`); this module owns the retry
  policy so a rate limit is never retried blindly.
- The API key, request headers and the submitted text are never logged or put in messages.
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

import groq
from groq import Groq

from core.config import Settings
from core.errors import (
    ApiError,
    ApiUnavailableError,
    AuthError,
    CheckerError,
    InvalidOutputError,
    ModelUnavailableError,
    RateLimitError,
    RequestTimeoutError,
    TokenLimitError,
)
from core.prompts import build_messages
from core.schemas import CheckOptions, ParsedOutput, build_response_schema, parse_llm_output

logger = logging.getLogger("grammar_checker.llm")

TEMPERATURE = 0.2
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 8.0
_RESET_RE = re.compile(r"^[0-9A-Za-z.:+\- ]{1,32}$")
_RETRYABLE = (RequestTimeoutError, ApiUnavailableError, InvalidOutputError)


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if number >= 0 and number == number else None  # rejects NaN / negatives


def _safe_reset(value: str | None) -> str | None:
    return value if value is not None and _RESET_RE.match(value) else None


def rate_limit_from_headers(headers: Mapping[str, str]) -> RateLimitError:
    """Interpret Groq's documented rate-limit headers.

    Per the docs `x-ratelimit-*-requests` describe the per-day request quota and
    `x-ratelimit-*-tokens` the per-minute token quota; `retry-after` is only sent with 429.
    """
    retry_after = _to_float(headers.get("retry-after"))
    remaining_requests = _to_float(headers.get("x-ratelimit-remaining-requests"))
    if remaining_requests is not None and remaining_requests == 0:
        return RateLimitError(
            "day", retry_after, _safe_reset(headers.get("x-ratelimit-reset-requests"))
        )
    if retry_after is not None:
        return RateLimitError(
            "minute", retry_after, _safe_reset(headers.get("x-ratelimit-reset-tokens"))
        )
    return RateLimitError("unknown")


def map_groq_exception(exc: Exception, model: str) -> CheckerError:
    """Translate an SDK exception into a typed, user-presentable error (no raw payloads)."""
    if isinstance(exc, groq.APITimeoutError):
        return RequestTimeoutError()
    if isinstance(exc, groq.APIConnectionError):
        return ApiUnavailableError("Could not reach the Groq API. Check your network connection.")
    if isinstance(exc, groq.APIStatusError):
        status = exc.status_code
        text = f"{getattr(exc, 'message', '')} {exc}".lower()
        if status == 429:
            return rate_limit_from_headers(exc.response.headers)
        if status in (401, 403):
            return AuthError(status)
        if status == 404:
            return ModelUnavailableError(model)
        if status == 413:
            return TokenLimitError()
        if status == 408:
            return RequestTimeoutError()
        if status == 422:
            return InvalidOutputError()
        if status == 400:
            if "context length" in text or "context_length" in text or "too large" in text:
                return TokenLimitError()
            if "generated json" in text or "json_validate_failed" in text:
                return InvalidOutputError()
            return ApiError(
                "Groq rejected the request (HTTP 400). If you changed GROQ_MODEL, check that the "
                "model supports strict structured output and GROQ_REASONING_EFFORT."
            )
        if status >= 500:
            return ApiUnavailableError(f"The Groq service had a problem (HTTP {status}).")
        return ApiError(f"Groq returned an error (HTTP {status}).")
    return ApiError("An unexpected error occurred while calling the Groq API.")


class GroqClient:
    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rand: Callable[[], float] = random.random,
    ) -> None:
        self._settings = settings
        self._client = client
        self._sleep = sleep
        self._rand = rand

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = Groq(
                api_key=self._settings.api_key,
                timeout=self._settings.timeout_seconds,
                max_retries=0,  # retries are handled (and bounded) in request_suggestions()
            )
        return self._client

    def _single_call(self, text: str, options: CheckOptions) -> ParsedOutput:
        s = self._settings
        kwargs: dict[str, Any] = {
            "model": s.model,
            "messages": build_messages(text, options),
            "response_format": {
                "type": "json_schema",
                "json_schema": build_response_schema(options.categories),
            },
            "temperature": TEMPERATURE,
            "max_completion_tokens": s.max_completion_tokens,
        }
        if s.reasoning_effort:
            kwargs["reasoning_effort"] = s.reasoning_effort
            kwargs["include_reasoning"] = False

        try:
            response = self._get_client().chat.completions.create(**kwargs)
        except groq.APIError as exc:
            logger.warning(
                "Groq call failed: %s (HTTP %s)",
                type(exc).__name__,
                getattr(exc, "status_code", "n/a"),
            )
            raise map_groq_exception(exc, s.model) from None

        if not response.choices:
            raise InvalidOutputError()
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise TokenLimitError()
        content = getattr(choice.message, "content", None)
        if not content:
            raise InvalidOutputError()
        return parse_llm_output(content, options.categories)

    def _backoff(self, attempt: int) -> float:
        base = min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * (2**attempt))
        return base + self._rand() * 0.5

    def request_suggestions(self, text: str, options: CheckOptions) -> ParsedOutput:
        """One logical request with at most `1 + max_retries` API calls in total."""
        attempts_allowed = 1 + self._settings.max_retries
        waited_out_rate_limit = False

        for attempt in range(attempts_allowed):
            is_last = attempt + 1 >= attempts_allowed
            try:
                return self._single_call(text, options)
            except RateLimitError as err:
                can_wait = (
                    err.scope == "minute"
                    and err.retry_after is not None
                    and err.retry_after <= self._settings.max_retry_wait_seconds
                    and not waited_out_rate_limit
                )
                if is_last or not can_wait:
                    raise
                waited_out_rate_limit = True
                logger.info("Rate limited; waiting %.1fs before one retry", err.retry_after)
                self._sleep(err.retry_after or 0.0)
            except _RETRYABLE as err:
                if is_last:
                    raise
                delay = self._backoff(attempt)
                logger.info("Retrying after %s in %.1fs", type(err).__name__, delay)
                self._sleep(delay)

        raise ApiError("The request could not be completed.")  # pragma: no cover (unreachable)
