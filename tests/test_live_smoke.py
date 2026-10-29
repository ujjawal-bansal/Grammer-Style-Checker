"""Opt-in smoke test against the real Groq API. Skipped by default.

Run with:  RUN_LIVE_TESTS=1 pytest -m live
It sends one short request, so it uses a little of your own free-tier quota.
"""

import os

import pytest

from core.config import load_settings
from core.errors import ConfigError
from core.llm_client import GroqClient
from core.schemas import CheckOptions

pytestmark = pytest.mark.live


@pytest.fixture
def live_client():
    if os.environ.get("RUN_LIVE_TESTS") != "1":
        pytest.skip("set RUN_LIVE_TESTS=1 to run tests that call the real Groq API")
    try:
        settings = load_settings()
    except ConfigError as err:
        pytest.skip(err.user_message)
    return GroqClient(settings)


def test_live_roundtrip_returns_validated_suggestions(live_client):
    parsed = live_client.request_suggestions("I has a apple. She dont like it.", CheckOptions())
    assert parsed.invalid_count == 0
    assert all(s.original and s.explanation for s in parsed.valid)
