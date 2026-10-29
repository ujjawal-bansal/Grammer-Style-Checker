"""Typed errors. Every error carries a `user_message` that is safe to show in the UI:
it never contains the API key, request headers or the submitted text."""

from __future__ import annotations


class CheckerError(Exception):
    """Base class for all expected, user-presentable failures."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class ConfigError(CheckerError):
    """Missing or invalid configuration (environment / .env)."""


class InputValidationError(CheckerError):
    """The submitted text cannot be processed (empty, too long, ...)."""


class RateLimitError(CheckerError):
    """Groq answered HTTP 429. `scope` is 'day' (request quota exhausted), 'minute'
    (per-minute limits) or 'unknown'; derived from the documented rate-limit headers."""

    def __init__(
        self,
        scope: str = "unknown",
        retry_after: float | None = None,
        reset: str | None = None,
    ) -> None:
        self.scope = scope
        self.retry_after = retry_after
        self.reset = reset
        no_fallback = "The app never switches to another model or a paid service on its own."
        if scope == "day":
            when = f" (resets in {reset})" if reset else ""
            message = (
                f"The daily request limit for this Groq account appears to be used up{when}. "
                f"Try again after it resets. {no_fallback}"
            )
        elif scope == "minute":
            wait = f"about {max(1, round(retry_after))} seconds" if retry_after else "a minute"
            message = (
                f"Groq's per-minute rate limit was reached. Please wait {wait} and try again. "
                f"{no_fallback}"
            )
        else:
            message = (
                "Groq's rate limit was reached. Please wait a bit and try again. Your limits are "
                f"listed at https://console.groq.com/settings/limits. {no_fallback}"
            )
        super().__init__(message)


class AuthError(CheckerError):
    def __init__(self, status: int) -> None:
        super().__init__(
            f"Groq rejected the API key (HTTP {status}). Check GROQ_API_KEY in your .env file."
        )


class ModelUnavailableError(CheckerError):
    def __init__(self, model: str) -> None:
        super().__init__(
            f"The model '{model}' was not found or is not available to your account. "
            "Check GROQ_MODEL. The app does not fall back to another model."
        )


class RequestTimeoutError(CheckerError):
    def __init__(self) -> None:
        super().__init__("The request to Groq timed out. Please try again.")


class ApiUnavailableError(CheckerError):
    """Network failure or HTTP 5xx from Groq."""


class TokenLimitError(CheckerError):
    """The input or the generated output did not fit the model's token limits."""

    def __init__(self) -> None:
        super().__init__("The text was too large for a single request (token limit).")


class InvalidOutputError(CheckerError):
    """The model returned something that is not valid JSON / not the expected shape."""

    def __init__(self) -> None:
        super().__init__("The model returned a response that could not be validated.")


class ApiError(CheckerError):
    """Any other Groq API failure that is not worth retrying."""


class StaleSuggestionError(CheckerError):
    """A suggestion no longer matches the text (or two accepted suggestions overlap)."""

    def __init__(self) -> None:
        super().__init__("The suggestions no longer match the text. Please run the check again.")
