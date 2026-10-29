import json

import pytest

from core.errors import InputValidationError, InvalidOutputError
from core.schemas import (
    CATEGORIES,
    MAX_SUGGESTIONS_PER_REQUEST,
    CheckOptions,
    build_response_schema,
    parse_llm_output,
)
from tests.helpers import suggestion

ALL = frozenset(CATEGORIES)
ALLOWED_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum"}


def walk(node):
    yield node
    if isinstance(node, dict):
        for key in ("properties",):
            for child in node.get(key, {}).values():
                yield from walk(child)
        if "items" in node:
            yield from walk(node["items"])


def test_schema_follows_documented_strict_mode_rules():
    payload = build_response_schema(ALL)
    assert payload["strict"] is True
    root = payload["schema"]
    for node in walk(root):
        assert set(node) <= ALLOWED_KEYWORDS, f"unexpected keywords: {set(node) - ALLOWED_KEYWORDS}"
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])  # all props required, no optionals


def test_schema_enum_matches_requested_categories_in_canonical_order():
    payload = build_response_schema(frozenset({"clarity", "grammar"}))
    item = payload["schema"]["properties"]["suggestions"]["items"]
    assert item["properties"]["category"]["enum"] == ["grammar", "clarity"]


def test_schema_rejects_no_categories():
    with pytest.raises(InputValidationError):
        build_response_schema(frozenset())


def test_options_validation():
    assert CheckOptions().ordered_categories == CATEGORIES
    assert CheckOptions(categories={"tone"}).ordered_categories == ("tone",)
    with pytest.raises(InputValidationError):
        CheckOptions(categories=frozenset())
    with pytest.raises(InputValidationError):
        CheckOptions(categories={"grammar", "spelling"})
    with pytest.raises(InputValidationError):
        CheckOptions(tone="sarcastic")


def test_parse_valid_output():
    content = json.dumps({"suggestions": [suggestion(), suggestion("cat", "dog", "clarity")]})
    parsed = parse_llm_output(content, ALL)
    assert [s.original for s in parsed.valid] == ["teh", "cat"]
    assert (parsed.invalid_count, parsed.overflow_count) == (0, 0)


def test_parse_empty_list_is_valid():
    assert parse_llm_output('{"suggestions": []}', ALL).valid == []


@pytest.mark.parametrize(
    "content",
    ["", "not json", "[]", '"x"', "{}", '{"suggestions": "nope"}', '{"other": []}', "null"],
)
def test_parse_rejects_malformed_whole_responses(content):
    with pytest.raises(InvalidOutputError):
        parse_llm_output(content, ALL)


def test_bad_items_are_counted_not_hidden():
    bad_items = [
        {**suggestion(), "category": "spelling"},  # unknown category
        {**suggestion(), "original": ""},  # empty original
        {**suggestion(), "extra": "field"},  # additional property
        {"original": "x"},  # missing fields
        "just a string",
        suggestion("ok", "fine"),
    ]
    parsed = parse_llm_output(json.dumps({"suggestions": bad_items}), ALL)
    assert [s.original for s in parsed.valid] == ["ok"]
    assert parsed.invalid_count == 5


def test_unrequested_category_is_counted_as_invalid():
    content = json.dumps({"suggestions": [suggestion(category="tone"), suggestion("a", "b")]})
    parsed = parse_llm_output(content, frozenset({"grammar"}))
    assert len(parsed.valid) == 1
    assert parsed.invalid_count == 1


def test_overflow_is_capped_and_reported():
    items = [suggestion(f"w{i}", f"x{i}") for i in range(MAX_SUGGESTIONS_PER_REQUEST + 7)]
    parsed = parse_llm_output(json.dumps({"suggestions": items}), ALL)
    assert len(parsed.valid) == MAX_SUGGESTIONS_PER_REQUEST
    assert parsed.overflow_count == 7
