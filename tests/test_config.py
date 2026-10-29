import socket

import pytest

from core.config import DEFAULT_MODEL, Settings, load_settings
from core.errors import ConfigError
from tests.conftest import FAKE_API_KEY


def env(**overrides):
    base = {"GROQ_API_KEY": FAKE_API_KEY}
    base.update(overrides)
    return base


def test_defaults():
    s = load_settings(env())
    assert s.model == DEFAULT_MODEL == "openai/gpt-oss-20b"
    assert s.timeout_seconds == 30.0
    assert s.max_retries == 2
    assert s.max_retry_wait_seconds == 10.0
    assert s.reasoning_effort == "low"
    assert s.max_completion_tokens == 2500
    assert s.max_input_words == 1000
    assert s.max_chunk_words == 400


def test_overrides_from_environment():
    s = load_settings(
        env(
            GROQ_MODEL="openai/gpt-oss-120b",
            GROQ_TIMEOUT_SECONDS="12.5",
            GROQ_MAX_RETRIES="0",
            GROQ_MAX_RETRY_WAIT_SECONDS="3",
            GROQ_REASONING_EFFORT="HIGH",
            GROQ_MAX_COMPLETION_TOKENS="4000",
            MAX_INPUT_WORDS="500",
            MAX_CHUNK_WORDS="100",
        )
    )
    assert s.model == "openai/gpt-oss-120b"
    assert s.timeout_seconds == 12.5
    assert s.max_retries == 0
    assert s.max_retry_wait_seconds == 3.0
    assert s.reasoning_effort == "high"
    assert s.max_completion_tokens == 4000
    assert (s.max_input_words, s.max_chunk_words) == (500, 100)


@pytest.mark.parametrize("raw", ["", "none", "off", "  "])
def test_reasoning_effort_can_be_disabled(raw):
    assert load_settings(env(GROQ_REASONING_EFFORT=raw)).reasoning_effort is None


def test_invalid_reasoning_effort():
    with pytest.raises(ConfigError, match="GROQ_REASONING_EFFORT"):
        load_settings(env(GROQ_REASONING_EFFORT="extreme"))


@pytest.mark.parametrize("environ", [{}, {"GROQ_API_KEY": ""}, {"GROQ_API_KEY": "   "}])
def test_missing_key(environ):
    with pytest.raises(ConfigError, match="GROQ_API_KEY is not set"):
        load_settings(environ)


@pytest.mark.parametrize("bad", ["your_groq_api_key_here", "CHANGEME", "has space inside"])
def test_placeholder_or_malformed_key(bad):
    with pytest.raises(ConfigError, match="looks invalid") as exc:
        load_settings({"GROQ_API_KEY": bad})
    assert bad not in exc.value.user_message


@pytest.mark.parametrize(
    "name,value",
    [
        ("GROQ_TIMEOUT_SECONDS", "abc"),
        ("GROQ_TIMEOUT_SECONDS", "0"),
        ("GROQ_MAX_RETRIES", "-1"),
        ("GROQ_MAX_RETRIES", "99"),
        ("GROQ_MAX_RETRIES", "1.5"),
        ("GROQ_MAX_COMPLETION_TOKENS", "10"),
        ("MAX_INPUT_WORDS", "0"),
        ("MAX_CHUNK_WORDS", "5"),
    ],
)
def test_invalid_numbers_name_the_variable(name, value):
    with pytest.raises(ConfigError, match=name):
        load_settings(env(**{name: value}))


def test_invalid_model_characters():
    with pytest.raises(ConfigError, match="GROQ_MODEL"):
        load_settings(env(GROQ_MODEL="bad model; rm -rf"))


def test_blank_model_uses_default():
    assert load_settings(env(GROQ_MODEL="  ")).model == DEFAULT_MODEL


def test_api_key_never_appears_in_repr_str_or_summary():
    s = load_settings(env())
    assert s.api_key == FAKE_API_KEY
    for rendered in (repr(s), str(s), f"{s}", repr(s.safe_summary())):
        assert FAKE_API_KEY not in rendered
    assert "api_key" not in s.safe_summary()


def test_error_messages_do_not_contain_the_key():
    with pytest.raises(ConfigError) as exc:
        load_settings(env(MAX_INPUT_WORDS="nope"))
    assert FAKE_API_KEY not in exc.value.user_message


def test_dotenv_file_is_used_and_real_environment_wins(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        f"GROQ_API_KEY={FAKE_API_KEY}\nGROQ_MODEL=from-file/model\nMAX_CHUNK_WORDS=250\n"
    )
    s = load_settings(env_file=dotenv)
    assert (s.api_key, s.model, s.max_chunk_words) == (FAKE_API_KEY, "from-file/model", 250)

    monkeypatch.setenv("GROQ_MODEL", "from-env/model")
    assert load_settings(env_file=dotenv).model == "from-env/model"


def test_missing_dotenv_file_is_fine_when_env_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_API_KEY)
    assert load_settings(env_file=tmp_path / "absent.env").api_key == FAKE_API_KEY


def test_loading_dotenv_does_not_mutate_os_environ(tmp_path):
    import os

    dotenv = tmp_path / ".env"
    dotenv.write_text(f"GROQ_API_KEY={FAKE_API_KEY}\n")
    load_settings(env_file=dotenv)
    assert "GROQ_API_KEY" not in os.environ


def test_settings_is_immutable():
    s = load_settings(env())
    assert isinstance(s, Settings)
    with pytest.raises(AttributeError):
        s.model = "other"  # type: ignore[misc]


def test_network_is_blocked_in_unit_tests():
    sock = socket.socket()
    with pytest.raises(RuntimeError, match="blocked"):
        sock.connect(("127.0.0.1", 9))
    sock.close()
