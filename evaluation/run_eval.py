"""Live correction-quality evaluation against the configured Groq model.

    python -m evaluation.run_eval                 # all cases
    python -m evaluation.run_eval --resume        # only cases that are missing or failed to run
    python -m evaluation.run_eval --only tone-01,clarity-02

This sends one request per case (more if a case is split or retried), so it uses some of your
API quota. Requests are paced to stay under free-tier per-minute limits. The API key is read
the same way as the app reads it and is never printed.

Outcomes are stored in evaluation/results.json and merged across runs as long as the model,
reasoning effort and prompt are unchanged; any change starts a fresh set. The Markdown report
(evaluation/RESULTS.md) is rebuilt from the stored outcomes after every run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

from core.config import load_settings
from core.errors import ConfigError
from core.llm_client import GroqClient
from core.pipeline import run_check
from core.prompts import SYSTEM_PROMPT
from evaluation.harness import (
    Outcome,
    Summary,
    evaluate,
    load_cases,
    merge,
    outcomes_from_json,
    outcomes_to_json,
    render_report,
)

HERE = Path(__file__).resolve().parent
DEFAULT_REPORT = HERE / "RESULTS.md"
DEFAULT_STORE = HERE / "results.json"


def fingerprint(model: str, reasoning_effort: str | None) -> dict:
    prompt = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]
    return {"model": model, "reasoning_effort": reasoning_effort, "prompt": prompt}


def load_store(path: Path, meta: dict) -> dict:
    """Saved outcomes for this exact configuration, or an empty store."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"meta": meta, "runs": [], "outcomes": []}
    if data.get("meta") != meta:
        print("Model, reasoning effort or prompt changed since the stored run: starting fresh.")
        return {"meta": meta, "runs": [], "outcomes": []}
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resume", action="store_true", help="only missing or unfinished cases")
    parser.add_argument("--only", default="", help="comma-separated case ids to run")
    parser.add_argument("--limit", type=int, default=None, help="run at most N cases")
    parser.add_argument("--pace", type=float, default=8.0, help="seconds between requests")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Markdown output")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE, help="saved outcomes")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as err:
        print(err.user_message, file=sys.stderr)
        return 2

    all_cases = load_cases()
    meta = fingerprint(settings.model, settings.reasoning_effort)
    store = load_store(args.store, meta)
    stored = outcomes_from_json(store["outcomes"], all_cases)

    cases = all_cases
    wanted = {i.strip() for i in args.only.split(",") if i.strip()}
    if wanted:
        unknown = wanted - {c.id for c in all_cases}
        if unknown:
            print(f"Unknown case ids: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c.id in wanted]
    if args.resume:
        done = {o.case.id for o in stored.outcomes if o.completed}
        cases = [c for c in cases if c.id not in done]
    cases = cases[: args.limit]

    print(f"Evaluating {len(cases)} cases with {settings.model} (uses API quota)…")
    client = GroqClient(settings)
    first = True

    def check(text, options):
        nonlocal first
        if not first:
            time.sleep(args.pace)
        first = False
        return run_check(text, options, settings, client)

    def show(o: Outcome) -> None:
        mark = "pass" if o.passed else "FAIL"
        print(f"[{mark}] {o.case.id}: {o.corrected if o.completed else o.error}")
        for reason in o.reasons:
            print(f"        - {reason}")

    fresh = evaluate(cases, check, on_outcome=show) if cases else Summary()
    combined = merge(stored, fresh, all_cases)
    if cases:
        store["runs"].append(date.today().isoformat())
    store["outcomes"] = outcomes_to_json(combined)
    args.store.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    runs = store["runs"]
    note = (
        f"Model `{settings.model}`, reasoning effort `{settings.reasoning_effort}`, prompt "
        f"fingerprint `{meta['prompt']}`. Each case was checked once; results combine "
        f"{len(runs)} run(s) on {', '.join(sorted(set(runs))) or 'no date'} (later runs only "
        "re-checked cases that had not completed). LLM output varies between runs, so treat "
        "this as a snapshot, not a guarantee."
    )
    args.report.write_text(render_report(combined, note), encoding="utf-8")
    print(
        f"\n{combined.passed}/{combined.total} passed, {combined.incomplete} not completed. "
        f"Report: {args.report}"
    )
    return 0 if combined.total and combined.passed == combined.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
