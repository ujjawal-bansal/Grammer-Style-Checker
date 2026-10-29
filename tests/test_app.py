"""UI tests with Streamlit's AppTest. The Groq SDK client is replaced by a scripted fake,
so no network access happens and no free-tier quota is used."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import core.config
import core.pipeline
from core.llm_client import GroqClient as RealGroqClient
from tests.conftest import FAKE_API_KEY
from tests.helpers import FakeGroq, ok_response, rate_limit, suggestion

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TEXT = "I has a apple. She dont like it."
P1 = "I has a apple that I bought yesterday from the market near my house."
P2 = "She dont like it because the taste was very strange to her indeed."


def sug(original, replacement, context_before=""):
    return suggestion(original, replacement, "grammar", "why", context_before)


@pytest.fixture
def api(monkeypatch, tmp_path):
    """Valid configuration + a scripted fake instead of the real Groq client."""
    monkeypatch.setenv("GROQ_API_KEY", FAKE_API_KEY)
    monkeypatch.setattr(core.config, "DEFAULT_ENV_FILE", tmp_path / "absent.env")
    fake = FakeGroq([])
    monkeypatch.setattr(
        core.pipeline,
        "GroqClient",
        lambda settings: RealGroqClient(
            settings, client=fake, sleep=lambda s: None, rand=lambda: 0.0
        ),
    )
    return fake


def new_app():
    return AppTest.from_file(APP, default_timeout=20).run()


def check(at, text=TEXT):
    at.text_area(key="input_text").set_value(text).run()
    return at.button(key="check").click().run()


def all_text(at):
    values = []

    def walk(node):
        for child in getattr(node, "children", {}).values():
            value = getattr(child, "value", None)
            if isinstance(value, str):
                values.append(value)
            walk(child)

    walk(at.main)
    walk(at.sidebar)
    return values


def corrected(at):
    return at.code[0].value


def test_missing_api_key_shows_setup_help_and_stops(monkeypatch, tmp_path):
    monkeypatch.setattr(core.config, "DEFAULT_ENV_FILE", tmp_path / "absent.env")
    at = new_app()
    assert not at.exception
    assert "GROQ_API_KEY is not set" in at.error[0].value
    assert ".env.example" in at.info[0].value
    assert len(at.text_area) == 0  # the app stops before rendering the form


def test_initial_state(api):
    at = new_app()
    assert not at.exception
    assert at.button(key="check").disabled is True  # nothing to check yet
    assert at.sidebar.selectbox(key="tone").value == "professional"
    assert api.calls == []


def test_check_then_accept_reject_without_new_api_calls(api):
    api.completions.script = [ok_response([sug("has", "have", "I "), sug("dont", "don't")])]
    at = check(new_app())
    assert not at.exception
    assert len(api.calls) == 1
    assert corrected(at) == TEXT  # nothing accepted yet -> text unchanged

    at.button(key="accept_s1").click().run()
    assert corrected(at) == "I have a apple. She dont like it."
    at.button(key="accept_s2").click().run()
    assert corrected(at) == "I have a apple. She don't like it."
    at.button(key="reject_s1").click().run()
    assert corrected(at) == "I has a apple. She don't like it."
    assert at.session_state["decisions"] == {"s1": "rejected", "s2": "accepted"}

    at.button(key="accept_all").click().run()
    assert corrected(at) == "I have a apple. She don't like it."
    at.button(key="reject_all").click().run()
    assert corrected(at) == TEXT
    at.button(key="accept_all").click().run()
    at.button(key="reset_all").click().run()
    assert corrected(at) == TEXT and at.session_state["decisions"] == {}
    assert len(api.calls) == 1  # every click above only re-rendered stored results
    assert len(at.download_button) == 1


def test_running_the_check_again_calls_the_api_again_and_resets_decisions(api):
    api.completions.script = [
        ok_response([sug("has", "have")]),
        ok_response([sug("dont", "don't")]),
    ]
    at = check(new_app())
    at.button(key="accept_s1").click().run()
    at = check(at, "She dont like it.")
    assert len(api.calls) == 2
    assert at.session_state["decisions"] == {}
    assert corrected(at) == "She dont like it."


def test_no_issues_shows_success(api):
    api.completions.script = [ok_response([])]
    at = check(new_app())
    assert "No issues found" in at.success[0].value
    assert len(at.code) == 0


def test_selected_options_reach_the_request(api):
    api.completions.script = [ok_response([])]
    at = new_app()
    at.sidebar.checkbox(key="category_grammar").uncheck()
    at.sidebar.selectbox(key="tone").select("formal")
    at = check(at)
    request = api.calls[0]
    enum = request["response_format"]["json_schema"]["schema"]["properties"]["suggestions"]
    assert enum["items"]["properties"]["category"]["enum"] == ["tone", "clarity"]
    assert "Target tone: formal" in request["messages"][1]["content"]


def test_no_category_selected_disables_checking(api):
    at = new_app()
    for category in ("grammar", "tone", "clarity"):
        at.sidebar.checkbox(key=f"category_{category}").uncheck()
    at.text_area(key="input_text").set_value(TEXT).run()
    assert at.button(key="check").disabled is True
    assert any("at least one category" in w.value for w in at.warning)
    assert at.sidebar.selectbox(key="tone").disabled is True


def test_over_limit_input_is_blocked_before_any_api_call(api, monkeypatch):
    monkeypatch.setenv("MAX_INPUT_WORDS", "5")
    at = new_app()
    at.text_area(key="input_text").set_value("one two three four five six").run()
    assert at.button(key="check").disabled is True
    assert any("6 words; the limit is 5" in w.value for w in at.warning)
    assert api.calls == []


def test_unverifiable_suggestions_are_reported_to_the_user(api):
    api.completions.script = [
        ok_response([sug("has", "have", "I "), sug("not in the text", "x"), sug("same", "same")])
    ]
    at = check(new_app())
    message = at.info[0].value
    assert "1 suggestion" in message and "could not be verified" in message
    assert "not found in your text" in message
    assert any("proposed no change" in c.value for c in at.caption)
    assert at.button(key="accept_s1").value is False and len(at.button) >= 5


def test_partial_results_are_kept_and_explained(api, monkeypatch):
    monkeypatch.setenv("MAX_CHUNK_WORDS", "20")
    headers = {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "7m"}
    api.completions.script = [ok_response([sug("has", "have", "I ")]), rate_limit(headers)]
    at = check(new_app(), f"{P1}\n\n{P2}")
    assert not at.exception
    warning = at.warning[0].value
    assert "Partial result" in warning and "daily request limit" in warning and "7m" in warning
    assert "never switches to another model or a paid service" in warning
    assert len(api.calls) == 2
    at.button(key="accept_s1").click().run()  # the valid suggestion still works
    assert corrected(at).startswith("I have a apple")


def test_total_failure_shows_a_clear_error_and_no_suggestions(api):
    api.completions.script = [rate_limit()]
    at = check(new_app())
    assert "could not be checked" in at.error[0].value
    assert "rate limit" in at.error[0].value.lower()
    assert not any(b.key.startswith("accept_s") for b in at.button)
    assert len(api.calls) == 1


def test_editing_the_text_after_a_check_flags_stale_results(api):
    api.completions.script = [ok_response([sug("has", "have")])]
    at = check(new_app())
    at.text_area(key="input_text").set_value(TEXT + " More text.").run()
    assert any("text box changed" in i.value for i in at.info)
    assert corrected(at) == TEXT  # still based on the text that was checked


def test_model_output_is_html_escaped_in_every_html_element(api):
    api.completions.script = [ok_response([sug("has", "<img src=x onerror=alert(1)>")])]
    at = check(new_app())
    at.button(key="accept_s1").click().run()
    rendered = " ".join(h.value for h in at.html)
    assert "<img" not in rendered and "&lt;img" in rendered
    assert "<script" not in rendered


def test_user_text_is_not_rendered_as_markdown_or_html(api):
    text = "I has ![x](http://example.invalid/t.png) <script>alert(1)</script> done."
    api.completions.script = [ok_response([sug("has", "have", "I ")])]
    at = check(new_app(), text)
    rendered = " ".join(h.value for h in at.html)
    assert "<script" not in rendered and "&lt;script&gt;" in rendered
    assert not any("example.invalid" in m.value for m in at.markdown)


def test_unexpected_errors_are_shown_generically(api, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError(f"super secret detail {FAKE_API_KEY}")

    monkeypatch.setattr(core.pipeline, "run_check", boom)
    at = check(new_app())
    assert not at.exception
    assert at.error[0].value == "Something went wrong while checking your text. Please try again."
    assert not any("super secret" in t or FAKE_API_KEY in t for t in all_text(at))


def test_api_key_never_appears_in_the_rendered_page(api):
    api.completions.script = [rate_limit({"retry-after": "60"}), ok_response([sug("has", "have")])]
    at = check(new_app())
    assert all(FAKE_API_KEY not in t for t in all_text(at))
    at = check(at)
    assert all(FAKE_API_KEY not in t for t in all_text(at))
    assert not at.exception
