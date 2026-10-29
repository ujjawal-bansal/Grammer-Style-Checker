import inspect
import logging
from types import SimpleNamespace

import groq
import httpx
import pytest

from core.errors import (
    ApiError,
    ApiUnavailableError,
    AuthError,
    InvalidOutputError,
    ModelUnavailableError,
    RateLimitError,
    RequestTimeoutError,
    TokenLimitError,
)
from core.llm_client import GroqClient, rate_limit_from_headers
from core.schemas import CheckOptions
from tests.conftest import FAKE_API_KEY
from tests.helpers import (
    FakeGroq,
    make_settings,
    ok_response,
    rate_limit,
    status_error,
    suggestion,
)

REQ = httpx.Request("POST", "https://api.groq.com/x")
NO_CHOICES = SimpleNamespace(choices=[])
NONE_CONTENT = SimpleNamespace(
    choices=[SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="stop")]
)


class Harness:
    def __init__(self, script, **settings):
        self.fake = FakeGroq(script)
        self.sleeps: list[float] = []
        self.client = GroqClient(
            make_settings(**settings),
            client=self.fake,
            sleep=self.sleeps.append,
            rand=lambda: 0.0,
        )

    def run(self, text="teh cat", options=None):
        return self.client.request_suggestions(text, options or CheckOptions())


def test_success_and_request_shape():
    h = Harness([ok_response([suggestion()])])
    parsed = h.run()
    assert [s.replacement for s in parsed.valid] == ["the"]
    call = h.fake.calls[0]
    assert call["model"] == "openai/gpt-oss-20b"
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["strict"] is True
    assert call["max_completion_tokens"] == 2500
    assert call["reasoning_effort"] == "low" and call["include_reasoning"] is False
    assert "teh cat" in call["messages"][1]["content"]
    assert "stream" not in call and "tools" not in call  # unsupported with structured outputs
    assert h.sleeps == []


def test_reasoning_parameters_omitted_when_disabled():
    h = Harness([ok_response()], reasoning_effort=None)
    h.run()
    assert "reasoning_effort" not in h.fake.calls[0]
    assert "include_reasoning" not in h.fake.calls[0]


def test_configured_model_is_passed_through():
    h = Harness([ok_response()], model="openai/gpt-oss-120b")
    h.run()
    assert h.fake.calls[0]["model"] == "openai/gpt-oss-120b"


def test_model_not_found_does_not_fall_back_to_another_model():
    h = Harness([status_error(groq.NotFoundError, 404)], model="some/missing-model")
    with pytest.raises(ModelUnavailableError) as exc:
        h.run()
    assert "some/missing-model" in exc.value.user_message
    assert "does not fall back" in exc.value.user_message
    assert len(h.fake.calls) == 1 and h.fake.calls[0]["model"] == "some/missing-model"


@pytest.mark.parametrize("retries", [0, 1, 2, 3])
def test_server_errors_are_retried_a_bounded_number_of_times(retries):
    script = [status_error(groq.InternalServerError, 503)] * (retries + 5)
    h = Harness(script, max_retries=retries)
    with pytest.raises(ApiUnavailableError):
        h.run()
    assert len(h.fake.calls) == 1 + retries
    assert len(h.sleeps) == retries


def test_backoff_grows_exponentially_with_a_cap():
    h = Harness([status_error(groq.InternalServerError, 500)] * 6, max_retries=5)
    with pytest.raises(ApiUnavailableError):
        h.run()
    assert h.sleeps == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_timeout_then_success():
    h = Harness([groq.APITimeoutError(request=REQ), ok_response([suggestion()])])
    assert len(h.run().valid) == 1
    assert len(h.fake.calls) == 2


def test_timeout_exhausts_retries():
    h = Harness([groq.APITimeoutError(request=REQ)] * 3, max_retries=2)
    with pytest.raises(RequestTimeoutError):
        h.run()
    assert len(h.fake.calls) == 3


def test_connection_error_is_retryable_with_clear_message():
    h = Harness([groq.APIConnectionError(request=REQ)] * 2, max_retries=1)
    with pytest.raises(ApiUnavailableError) as exc:
        h.run()
    assert "network" in exc.value.user_message.lower()


@pytest.mark.parametrize(
    "bad",
    [
        ok_response(content="not json"),
        ok_response(content='{"nope": 1}'),
        ok_response(content=""),
        NO_CHOICES,
        NONE_CONTENT,
    ],
)
def test_invalid_output_is_retried_then_succeeds(bad):
    h = Harness([bad, ok_response([suggestion()])])
    assert len(h.run().valid) == 1
    assert len(h.fake.calls) == 2


def test_invalid_output_exhausts_retries():
    h = Harness([ok_response(content="garbage")] * 3, max_retries=2)
    with pytest.raises(InvalidOutputError):
        h.run()
    assert len(h.fake.calls) == 3


def test_422_is_retried_as_invalid_output():
    h = Harness([status_error(groq.UnprocessableEntityError, 422), ok_response()])
    h.run()
    assert len(h.fake.calls) == 2


def test_short_per_minute_rate_limit_is_waited_out_once():
    h = Harness([rate_limit({"retry-after": "3"}), ok_response([suggestion()])])
    assert len(h.run().valid) == 1
    assert h.sleeps == [3.0]
    assert len(h.fake.calls) == 2


def test_second_rate_limit_is_not_retried_again():
    h = Harness([rate_limit({"retry-after": "3"})] * 5, max_retries=5)
    with pytest.raises(RateLimitError) as exc:
        h.run()
    assert exc.value.scope == "minute"
    assert len(h.fake.calls) == 2
    assert h.sleeps == [3.0]


def test_long_retry_after_fails_immediately_without_sleeping():
    h = Harness([rate_limit({"retry-after": "60"}), ok_response()])
    with pytest.raises(RateLimitError) as exc:
        h.run()
    assert exc.value.retry_after == 60.0
    assert "60 seconds" in exc.value.user_message
    assert len(h.fake.calls) == 1 and h.sleeps == []


def test_daily_quota_exhaustion_is_reported_clearly_and_never_retried():
    headers = {
        "retry-after": "5",
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-reset-requests": "12m30s",
    }
    h = Harness([rate_limit(headers), ok_response()])
    with pytest.raises(RateLimitError) as exc:
        h.run()
    err = exc.value
    assert err.scope == "day" and err.reset == "12m30s"
    assert "daily" in err.user_message and "12m30s" in err.user_message
    assert "never switches to another model or a paid service" in err.user_message
    assert len(h.fake.calls) == 1 and h.sleeps == []


def test_rate_limit_without_headers_is_unknown_and_not_retried():
    h = Harness([rate_limit(), ok_response()])
    with pytest.raises(RateLimitError) as exc:
        h.run()
    assert exc.value.scope == "unknown"
    assert "console.groq.com/settings/limits" in exc.value.user_message
    assert len(h.fake.calls) == 1


def test_rate_limit_header_parsing_is_defensive():
    assert rate_limit_from_headers({"retry-after": "abc"}).scope == "unknown"
    assert rate_limit_from_headers({"retry-after": "-4"}).scope == "unknown"
    assert rate_limit_from_headers({"retry-after": "nan"}).scope == "unknown"
    err = rate_limit_from_headers(
        {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "<script>alert(1)"}
    )
    assert err.scope == "day" and err.reset is None  # unsafe reset text is dropped
    assert rate_limit_from_headers({"x-ratelimit-remaining-requests": "5"}).scope == "unknown"


@pytest.mark.parametrize(
    "cls,status", [(groq.AuthenticationError, 401), (groq.PermissionDeniedError, 403)]
)
def test_auth_errors_are_not_retried(cls, status):
    h = Harness([status_error(cls, status), ok_response()])
    with pytest.raises(AuthError) as exc:
        h.run()
    assert "GROQ_API_KEY" in exc.value.user_message
    assert len(h.fake.calls) == 1


def test_request_too_large_maps_to_token_limit_without_retry():
    h = Harness([status_error(groq.APIStatusError, 413), ok_response()])
    with pytest.raises(TokenLimitError):
        h.run()
    assert len(h.fake.calls) == 1


def test_truncated_output_maps_to_token_limit():
    h = Harness([ok_response(content='{"suggestions": [', finish_reason="length")])
    with pytest.raises(TokenLimitError):
        h.run()
    assert len(h.fake.calls) == 1


def test_bad_request_variants():
    ctx = status_error(groq.BadRequestError, 400, message="Please reduce the context length")
    with pytest.raises(TokenLimitError):
        Harness([ctx]).run()

    bad_json = status_error(
        groq.BadRequestError, 400, message="Generated JSON does not match the expected schema"
    )
    h = Harness([bad_json, ok_response()])
    h.run()
    assert len(h.fake.calls) == 2

    other = status_error(groq.BadRequestError, 400, message="reasoning_effort is not supported")
    h = Harness([other, ok_response()])
    with pytest.raises(ApiError) as exc:
        h.run()
    assert "GROQ_MODEL" in exc.value.user_message and len(h.fake.calls) == 1


def test_unexpected_sdk_error_becomes_api_error():
    h = Harness([groq.APIError("weird", request=REQ, body=None)])
    with pytest.raises(ApiError):
        h.run()


def test_secrets_and_text_never_reach_logs_or_user_messages(caplog):
    caplog.set_level(logging.DEBUG)
    secret_text = "my private diary entry xyz"
    scripts = [
        [rate_limit({"retry-after": "2"}), rate_limit({"retry-after": "2"})],
        [status_error(groq.AuthenticationError, 401, message=f"bad key {FAKE_API_KEY}")],
        [status_error(groq.InternalServerError, 500, message=secret_text)] * 3,
        [groq.APITimeoutError(request=REQ)] * 3,
    ]
    for script in scripts:
        h = Harness(script)
        with pytest.raises(Exception) as exc:
            h.run(text=secret_text)
        assert FAKE_API_KEY not in exc.value.user_message
        assert secret_text not in exc.value.user_message
    assert FAKE_API_KEY not in caplog.text
    assert secret_text not in caplog.text


def test_real_sdk_client_is_built_without_sdk_retries_and_with_configured_timeout():
    client = GroqClient(make_settings(timeout_seconds=12.5))
    sdk = client._get_client()
    assert isinstance(sdk, groq.Groq)
    assert sdk.max_retries == 0
    assert sdk.timeout == 12.5


def test_sdk_accepts_every_parameter_we_send():
    sdk = groq.Groq(api_key=FAKE_API_KEY)
    accepted = set(inspect.signature(sdk.chat.completions.create).parameters)
    h = Harness([ok_response()])
    h.run()
    assert set(h.fake.calls[0]) <= accepted
