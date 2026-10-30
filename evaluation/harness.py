"""Correction-quality evaluation: a labelled set of sentences, an automatic scorer and a
Markdown report.

The scorer only looks at the corrected text that the app would produce if the user accepted
every suggestion, so it measures the whole pipeline (prompt, model, span verification,
overlap handling and deterministic editing), not the raw model output.

Everything here is pure; the live run that calls the API is in `evaluation/run_eval.py`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from core.apply import apply_accepted
from core.schemas import CATEGORIES, TONES, CheckOptions, CheckResult

CASES_FILE = Path(__file__).resolve().parent / "cases.jsonl"
EXPECTATIONS = ("change", "unchanged")


@dataclass(frozen=True)
class Case:
    id: str
    group: str
    text: str
    expect: str
    checks: tuple[str, ...] = ("grammar",)
    tone: str = "professional"
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()

    @property
    def options(self) -> CheckOptions:
        return CheckOptions(categories=frozenset(self.checks), tone=self.tone)


@dataclass(frozen=True)
class Outcome:
    case: Case
    passed: bool
    corrected: str | None
    reasons: tuple[str, ...] = ()
    suggestions: int = 0
    error: str | None = None

    @property
    def completed(self) -> bool:
        """False when the API never produced a result (rate limit, network, server error)."""
        return self.corrected is not None


@dataclass
class Summary:
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(o.passed for o in self.outcomes)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def incomplete(self) -> int:
        return sum(not o.completed for o in self.outcomes)

    def by_group(self) -> dict[str, tuple[int, int]]:
        groups: dict[str, tuple[int, int]] = {}
        for o in self.outcomes:
            ok, n = groups.get(o.case.group, (0, 0))
            groups[o.case.group] = (ok + o.passed, n + 1)
        return groups


# ------------------------------------------------------------------ cases


def parse_case(raw: dict) -> Case:
    """Validate one case so a typo in the data file fails loudly instead of skewing scores."""
    known = {"id", "group", "text", "expect", "checks", "tone", "must_contain", "must_not_contain"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"case {raw.get('id')!r}: unknown fields {sorted(unknown)}")
    case = Case(
        id=str(raw["id"]),
        group=str(raw["group"]),
        text=str(raw["text"]),
        expect=str(raw["expect"]),
        checks=tuple(raw.get("checks", ("grammar",))),
        tone=str(raw.get("tone", "professional")),
        must_contain=tuple(raw.get("must_contain", ())),
        must_not_contain=tuple(raw.get("must_not_contain", ())),
    )
    if case.expect not in EXPECTATIONS:
        raise ValueError(f"case {case.id!r}: expect must be one of {EXPECTATIONS}")
    if not case.checks or not set(case.checks) <= set(CATEGORIES):
        raise ValueError(f"case {case.id!r}: checks must be a non-empty subset of {CATEGORIES}")
    if case.tone not in TONES:
        raise ValueError(f"case {case.id!r}: unknown tone {case.tone!r}")
    if case.expect == "change" and not case.must_contain:
        raise ValueError(f"case {case.id!r}: a 'change' case needs at least one must_contain")
    return case


def load_cases(path: Path = CASES_FILE) -> list[Case]:
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                cases.append(parse_case(json.loads(line)))
            except (KeyError, ValueError) as err:
                raise ValueError(f"{path.name} line {number}: {err}") from None
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: duplicate case ids")
    return cases


# ------------------------------------------------------------------ scoring


def normalise(text: str) -> str:
    """Compare apostrophes and quotes loosely: straight and curly forms are both correct."""
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def contains_phrase(text: str, phrase: str) -> bool:
    """True if `phrase` occurs as whole words, so a forbidden "am study" does not match the
    correct "am studying". Edges that are punctuation or spaces match as written."""
    phrase = normalise(phrase)
    start = r"\b" if phrase[:1].isalnum() else ""
    end = r"\b" if phrase[-1:].isalnum() else ""
    return re.search(start + re.escape(phrase) + end, normalise(text)) is not None


def score(case: Case, corrected: str) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    fixed = normalise(corrected)
    if case.expect == "unchanged":
        if corrected != case.text:
            reasons.append("changed text that should stay as written")
    else:
        if corrected == case.text:
            reasons.append("left an error uncorrected")
        reasons += [f"missing {s!r}" for s in case.must_contain if normalise(s) not in fixed]
        reasons += [
            f"still contains {s!r}" for s in case.must_not_contain if contains_phrase(fixed, s)
        ]
    return not reasons, tuple(reasons)


def corrected_text(result: CheckResult) -> str:
    """The text the user gets after pressing Accept all."""
    return apply_accepted(result.text, result.suggestions, [s.id for s in result.suggestions])


def evaluate(
    cases: Iterable[Case],
    check: Callable[[str, CheckOptions], CheckResult],
    on_outcome: Callable[[Outcome], None] | None = None,
) -> Summary:
    """Run `check` on every case and score the result. A case whose check fails or returns a
    partial result counts as failed, so API problems can never inflate the score."""
    summary = Summary()
    for case in cases:
        try:
            result = check(case.text, case.options)
        except Exception as err:  # noqa: BLE001 - any failure is a failed case, never a crash
            outcome = Outcome(case, False, None, ("the check failed",), error=type(err).__name__)
        else:
            if result.issues:
                outcome = Outcome(
                    case,
                    False,
                    None,
                    ("the check did not complete",),
                    error=result.issues[0].message,
                )
            else:
                fixed = corrected_text(result)
                passed, reasons = score(case, fixed)
                outcome = Outcome(case, passed, fixed, reasons, len(result.suggestions))
        summary.outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(outcome)
    return summary


def merge(previous: Summary, fresh: Summary, cases: Iterable[Case]) -> Summary:
    """Replace earlier outcomes with fresh ones for the same case, in case-file order."""
    by_id = {o.case.id: o for o in previous.outcomes}
    by_id.update({o.case.id: o for o in fresh.outcomes})
    return Summary([by_id[c.id] for c in cases if c.id in by_id])


# ------------------------------------------------------------------ saved outcomes


def outcomes_to_json(summary: Summary) -> list[dict]:
    return [
        {
            "id": o.case.id,
            "passed": o.passed,
            "corrected": o.corrected,
            "reasons": list(o.reasons),
            "suggestions": o.suggestions,
            "error": o.error,
        }
        for o in summary.outcomes
    ]


def outcomes_from_json(items: list[dict], cases: Iterable[Case]) -> Summary:
    """Rebuild saved outcomes against the current case file; cases that no longer exist, or
    whose definition changed since they were scored, are dropped so they get re-run."""
    by_id = {c.id: c for c in cases}
    outcomes = []
    for item in items:
        case = by_id.get(item.get("id"))
        if case is None:
            continue
        corrected = item.get("corrected")
        if corrected is not None and score(case, corrected) != (
            bool(item.get("passed")),
            tuple(item.get("reasons", ())),
        ):
            continue  # the expectations changed: the old verdict no longer holds
        outcomes.append(
            Outcome(
                case,
                bool(item.get("passed")),
                corrected,
                tuple(item.get("reasons", ())),
                int(item.get("suggestions", 0)),
                item.get("error"),
            )
        )
    return Summary(outcomes)


# ------------------------------------------------------------------ report


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_report(summary: Summary, title_note: str) -> str:
    lines = [
        "# Correction-quality evaluation",
        "",
        title_note,
        "",
        f"**Result: {summary.passed} / {summary.total} cases passed "
        f"({summary.passed / summary.total:.0%}).**"
        if summary.total
        else "No cases were run.",
        "",
    ]
    if summary.incomplete:
        lines += [
            f"{summary.incomplete} case(s) could not be checked because of API errors and "
            "count as failed. Run `python -m evaluation.run_eval --resume` to retry only those.",
            "",
        ]
    lines += [
        "| Group | Passed |",
        "|---|---|",
    ]
    lines += [f"| {g} | {ok} / {n} |" for g, (ok, n) in summary.by_group().items()]
    lines += [
        "",
        "## Every case",
        "",
        "| Case | Input | Corrected text (all suggestions accepted) | Result |",
        "|---|---|---|---|",
    ]
    for o in summary.outcomes:
        verdict = "pass" if o.passed else "FAIL: " + "; ".join(o.reasons)
        shown = o.corrected if o.corrected is not None else f"(no result: {o.error})"
        lines.append(f"| {o.case.id} | {_cell(o.case.text)} | {_cell(shown)} | {_cell(verdict)} |")
    lines.append("")
    return "\n".join(lines)
