"""Shared test doubles. No network: the Groq client is replaced by a scripted fake."""

from __future__ import annotations

import json
from types import SimpleNamespace

import groq
import httpx

from core.config import Settings
from tests.conftest import FAKE_API_KEY

_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def make_settings(**overrides) -> Settings:
    base = {"api_key": FAKE_API_KEY}
    base.update(overrides)
    return Settings(**base)


def suggestion(
    original="teh",
    replacement="the",
    category="grammar",
    explanation="Spelling.",
    context_before="",
):
    return {
        "category": category,
        "original": original,
        "replacement": replacement,
        "explanation": explanation,
        "context_before": context_before,
    }


def ok_response(suggestions=None, *, content=None, finish_reason="stop"):
    if content is None:
        content = json.dumps({"suggestions": suggestions if suggestions is not None else []})
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def status_error(cls, status, headers=None, message="boom"):
    response = httpx.Response(status, headers=headers or {}, request=_REQUEST)
    return cls(message, response=response, body=None)


def rate_limit(headers=None):
    return status_error(groq.RateLimitError, 429, headers)


class FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("unexpected extra API call")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FakeGroq:
    def __init__(self, script):
        self.completions = FakeCompletions(script)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self):
        return self.completions.calls
