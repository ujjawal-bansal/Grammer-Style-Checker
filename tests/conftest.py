import socket

import pytest

FAKE_API_KEY = "fake-test-key-DO-NOT-USE-0123456789"

_CONFIG_VARS = (
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


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch, request):
    """Tests never see real credentials from the developer's shell (except the opt-in live test)."""
    if request.node.get_closest_marker("live"):
        return
    for name in _CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch, request):
    """Unit tests must never reach the network (or spend free-tier quota)."""
    if request.node.get_closest_marker("live"):
        return

    def _blocked(*args, **kwargs):
        raise RuntimeError("network access is blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)


@pytest.fixture
def fake_key():
    return FAKE_API_KEY
