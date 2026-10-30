# Intelligent Grammar & Style Checker

Paste text, get **grammar, tone and clarity** suggestions from an LLM (Groq), then accept or
reject each suggestion individually. Built with Python, Streamlit and the Groq API.

## Features

- Three checks you can switch on/off right under the text box: grammar, tone (with a target
  tone), clarity.
- **Span-based suggestions**: every suggestion points at an exact piece of your text and shows
  the replacement and a short reason.
- **Per-suggestion Accept / Reject**, plus Accept all / Reject all / Reset. Nothing changes
  until you accept it.
- **Deterministic editing**: the corrected text is built by the app from the accepted
  suggestions, never rewritten wholesale by the model.
- Honest reporting: suggestions that cannot be verified against your text are counted and
  explained, not silently dropped. If a later part of a long text fails (rate limit, API
  error), the valid results obtained so far are kept and shown.

## Using the app

1. **Write.** Paste or type your text (or press **Try an example**). The counter under the box
   shows how many words you have against the limit; **Check text** stays disabled, with the
   reason shown, until the text and the selected checks are valid.
2. **Choose checks.** Toggle Grammar, Tone and Clarity, and pick a target tone (used by the
   tone check only).
3. **Check.** The text is sent to the API only now. Long texts are checked section by
   section, with a progress bar.
4. **Review.** A summary shows how many suggestions each check found and how many you have
   accepted, rejected or not yet decided. Suggestions are grouped by check; each card shows
   the original and the replacement, the reason, and its status in words (not only colour).
   Accept or reject each one, or use **Accept all / Reject all / Reset decisions**.
5. **Use the result.** The preview marks every suggestion in your text; the corrected text
   below it contains only the changes you accepted. Copy it with the button in the corner of
   the box, or **Download .txt**.
6. **Start again.** **New check** clears the text and the results. If a check fails, the
   error says why and **Try again** repeats it without retyping anything.

The interface uses a light theme with accessible contrast (WCAG AA for all text), visible
keyboard focus, and a single-column layout on narrow screens.

## Quick start

Requires **Python 3.11+**.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # add requirements-dev.txt to run the tests
cp .env.example .env                        # then put your Groq API key into .env
streamlit run app.py
```

`.env` is git-ignored. Never commit it, and never paste your key into the code or the UI.

## Configuration (`.env` or real environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | _(required)_ | Your Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Model ID; must support strict JSON-schema output on Groq |
| `GROQ_TIMEOUT_SECONDS` | `30` | Per-request timeout (passed to the SDK's HTTP client) |
| `GROQ_MAX_RETRIES` | `2` | Extra attempts after the first (timeouts, 5xx, malformed output) |
| `GROQ_MAX_RETRY_WAIT_SECONDS` | `10` | A per-minute rate limit is waited out once only if `retry-after` is at most this |
| `GROQ_REASONING_EFFORT` | `low` | `low`/`medium`/`high`; empty omits the parameter |
| `GROQ_MAX_COMPLETION_TOKENS` | `2500` | Output cap per request (includes reasoning tokens) |
| `MAX_INPUT_WORDS` | `1000` | Longest text accepted |
| `MAX_CHUNK_WORDS` | `400` | Text is sent to the API in chunks of at most this many words |

Real environment variables take precedence over `.env`.

## Free tier, limits and billing

- The app only uses the configured model through the Groq API. It has **no fallback model or
  provider**; when a limit is reached it shows a clear error and stops calling the API.
- Your actual limits are shown in the Groq console (https://console.groq.com/settings/limits).
  The rate-limit documentation lists example per-model limits but describes them as the
  base limits of the *Developer* plan, so do not assume those numbers apply to your plan.
- Groq's documentation says upgrading from the Free to the Developer tier requires a payment
  method. It does **not** explicitly state whether signing up for the Free tier requires one.
  Check this yourself in the console.
- **The app cannot enforce billing.** If your Groq organisation is on a paid plan, requests
  may be billed. Keep your plan on Free if you do not want to pay.

## How it works

```
text -> validate (words, characters) -> chunk (paragraph/sentence aware, <= MAX_CHUNK_WORDS)
     -> one Groq request per chunk, sequentially (strict JSON schema, bounded retries)
     -> validate every item (Pydantic) -> locate each suggestion in the text
     -> drop conflicts and report them -> merge -> Accept/Reject UI -> corrected text
```

Safety properties (each is covered by tests):

- A suggestion is kept only if its quoted `original` text is found **exactly**, on word
  boundaries, at exactly one location (the model also supplies `context_before` to
  disambiguate repeats). Otherwise it is reported as not found / mid-word / ambiguous.
- Overlapping suggestions can never both survive. Edits are applied left to right from the
  original string, so the result does not depend on the order of clicks, and a stale or
  overlapping selection raises an error instead of corrupting the text.
- Model output and your text are HTML-escaped before display; no markdown is rendered from them.
- The API key is held only in `Settings.api_key` (excluded from `repr`), is never logged,
  never rendered and never placed in error messages. Submitted text is not logged either.

### Error handling

| Situation | Behaviour |
|---|---|
| HTTP 429, per-day quota exhausted (`x-ratelimit-remaining-requests` is 0) | Clear "daily limit" message, no retry |
| HTTP 429, short `retry-after` | Waits once, retries once |
| HTTP 429, long or unknown `retry-after` | Clear message, no retry |
| Timeout, network error, 5xx, malformed output | Retried up to `GROQ_MAX_RETRIES` times with backoff |
| Request/response too large (HTTP 413, truncated output) | The chunk is halved once and retried; if that still fails the section is reported |
| 401/403, 404 (model), other 4xx | Immediate, specific message; no retry, no fallback |
| Failure in a later chunk | Earlier valid suggestions are kept; failed and skipped sections are listed |

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                 # offline: the Groq client is replaced by a scripted fake
RUN_LIVE_TESTS=1 pytest -m live        # optional: one real request (uses a little of your quota)
ruff check . && ruff format --check .
```

The offline suite never touches the network (sockets are blocked) and covers configuration,
schemas, the Groq client (retries, rate limits, error mapping), span location, overlap and
ambiguity handling, deterministic editing (including randomized checks against a reference
implementation), chunking, partial results and progress reporting, correction-quality
regressions, the Streamlit UI (via `streamlit.testing`: empty state, checks, accept/reject,
accept all, reset, new check, try again, download, long text, errors and HTML escaping), the
UI fragments in `ui/components.py`, and repository hygiene (`.env` ignored, no key-like
strings in tracked files).

## Layout

```
app.py              Streamlit UI: page layout, state and callbacks (thin)
ui/components.py    escaped HTML fragments: header, summary, cards, empty and clean states
ui/styles.css       style layer on top of the theme in .streamlit/config.toml
core/config.py      .env / environment -> validated Settings
core/errors.py      typed, user-presentable errors
core/schemas.py     options, strict JSON schema, output validation, result models
core/prompts.py     prompt construction (user text is passed as delimited data)
core/llm_client.py  Groq calls: timeout, bounded retries, 429 handling
core/pipeline.py    input validation, chunking, orchestration, partial results
core/apply.py       span location, conflict handling, deterministic edits, safe HTML
tests/              pytest suite
```

## Known limitations

- English-oriented prompts; matching of the model's quoted text is exact (no fuzzy matching),
  so a suggestion whose quote differs by even one character is reported as unverified.
- Chunks are processed one after another (free-tier limits are tight), so long texts take
  longer. The default limits (1,000 words, 400-word chunks) are starting points; tune them for
  your plan.
- Quality of the suggestions depends on the model. The default was selected from Groq's
  documentation (production model, strict structured output); its real-world quality on your
  texts is not benchmarked here.
- Single-user local app. If you deploy it publicly, all visitors share one API key and one
  quota. There is no authentication or per-user rate limiting.
