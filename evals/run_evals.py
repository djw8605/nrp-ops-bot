#!/usr/bin/env python
"""Score the agent against ``cases.yaml``.

This exists to answer one question with evidence rather than vibes: is a
self-hosted open-weight model good enough for the multi-step cases, or does the
multi-hop reasoning need a frontier model? Run the same suite against both and
compare -- the ``--model`` flag overrides the configured model per run.

    uv run python evals/run_evals.py --model llama-3.3-70b-instruct
    uv run python evals/run_evals.py --model llama-3.3-70b-instruct --dry-run

Scoring per case:

* ``namespace`` -- the expected namespace appeared in some tool call's arguments
* ``workload``  -- the expected workload/pod/node was named in the answer
* ``finding``   -- every expected keyword appeared in the answer
* ``tools``     -- all expected tools were called, and no forbidden tool was
* ``denied``    -- for out-of-scope cases, a namespace_denied result was produced

Tool count is reported, not scored: fewer calls is better, but a right answer in
six beats a wrong one in two.

**This runs against the live cluster and a live model.** It is read-only, but it
is not free, and it will show real cluster state. ``--dry-run`` validates the
case file and the configuration without calling anything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nrp_ops_agent.agent import Agent, AgentResult
from nrp_ops_agent.config import Settings, get_settings
from nrp_ops_agent.tools import registry

CASES_PATH = Path(__file__).resolve().parent / "cases.yaml"


@dataclass
class Case:
    id: str
    question: str
    namespace: str | None = None
    workload: str | None = None
    expect_tools: list[str] = field(default_factory=list)
    forbid_tools: list[str] = field(default_factory=list)
    finding_keywords: list[str] = field(default_factory=list)
    expect_denied: bool = False
    notes: str | None = None


@dataclass
class Score:
    case_id: str
    namespace_ok: bool
    workload_ok: bool
    finding_ok: bool
    tools_ok: bool
    denied_ok: bool
    tool_count: int
    latency_s: float
    stop_reason: str
    answer: str
    tools_called: list[str]
    missing_tools: list[str]
    forbidden_called: list[str]

    @property
    def passed(self) -> bool:
        return all(
            (self.namespace_ok, self.workload_ok, self.finding_ok, self.tools_ok, self.denied_ok)
        )

    def row(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        flags = "".join(
            letter if ok else letter.lower()
            for letter, ok in (
                ("N", self.namespace_ok),
                ("W", self.workload_ok),
                ("F", self.finding_ok),
                ("T", self.tools_ok),
                ("D", self.denied_ok),
            )
        )
        return (
            f"{mark}  {self.case_id:<26} {flags}  "
            f"{self.tool_count:>2} calls  {self.latency_s:>5.1f}s  {self.stop_reason}"
        )


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Case(**case) for case in raw["cases"]]


def score(case: Case, result: AgentResult, latency_s: float) -> Score:
    answer = result.text.lower()
    called = [call.name for call in result.tool_calls]

    namespace_ok = True
    if case.namespace:
        namespace_ok = (
            any(call.arguments.get("namespace") == case.namespace for call in result.tool_calls)
            or case.namespace in answer
        )

    workload_ok = case.workload.lower() in answer if case.workload else True
    finding_ok = all(keyword.lower() in answer for keyword in case.finding_keywords)

    missing = [tool for tool in case.expect_tools if tool not in called]
    forbidden = [tool for tool in case.forbid_tools if tool in called]
    tools_ok = not missing and not forbidden

    denied_ok = True
    if case.expect_denied:
        denied_ok = any(call.error == "namespace_denied" for call in result.tool_calls) or any(
            word in answer for word in ("scope", "not allowed", "denied")
        )

    return Score(
        case_id=case.id,
        namespace_ok=namespace_ok,
        workload_ok=workload_ok,
        finding_ok=finding_ok,
        tools_ok=tools_ok,
        denied_ok=denied_ok,
        tool_count=len(result.tool_calls),
        latency_s=latency_s,
        stop_reason=result.stop_reason,
        answer=result.text,
        tools_called=called,
        missing_tools=missing,
        forbidden_called=forbidden,
    )


def validate(cases: list[Case]) -> list[str]:
    """Catch case-file mistakes before spending a single model call."""
    known = set(registry())
    problems: list[str] = []
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            problems.append(f"{case.id}: duplicate id")
        seen.add(case.id)
        for tool in [*case.expect_tools, *case.forbid_tools]:
            if tool not in known:
                problems.append(f"{case.id}: unknown tool {tool!r}")
        if not case.question.strip():
            problems.append(f"{case.id}: empty question")
    return problems


async def run(cases: list[Case], settings: Settings, only: str | None) -> list[Score]:
    agent = Agent(settings)
    scores: list[Score] = []
    try:
        for case in cases:
            if only and only not in case.id:
                continue
            started = time.perf_counter()
            try:
                result = await agent.investigate(case.question)
            except Exception as exc:
                result = AgentResult(text=f"ERROR: {exc}", stop_reason="harness_error")
            elapsed = time.perf_counter() - started
            outcome = score(case, result, elapsed)
            scores.append(outcome)
            print(outcome.row(), flush=True)
    finally:
        await agent.aclose()
    return scores


def summarise(scores: list[Score], model: str) -> dict[str, Any]:
    passed = sum(1 for s in scores if s.passed)
    total = len(scores)
    return {
        "model": model,
        "cases": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "namespace_accuracy": _rate(scores, lambda s: s.namespace_ok),
        "workload_accuracy": _rate(scores, lambda s: s.workload_ok),
        "finding_accuracy": _rate(scores, lambda s: s.finding_ok),
        "tool_selection_accuracy": _rate(scores, lambda s: s.tools_ok),
        "mean_tool_calls": round(sum(s.tool_count for s in scores) / total, 2) if total else 0.0,
        "mean_latency_s": round(sum(s.latency_s for s in scores) / total, 2) if total else 0.0,
        "timeouts": sum(1 for s in scores if s.stop_reason == "timeout"),
        "model_errors": sum(1 for s in scores if s.stop_reason == "model_error"),
    }


def _rate(scores: list[Score], predicate: Any) -> float:
    return round(sum(1 for s in scores if predicate(s)) / len(scores), 3) if scores else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Override NRP_OPS_LLM_MODEL for this run.")
    parser.add_argument("--only", default=None, help="Substring filter on case id.")
    parser.add_argument("--json", type=Path, default=None, help="Write full results here.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the case file and configuration without calling anything.",
    )
    args = parser.parse_args(argv)

    cases = load_cases()
    problems = validate(cases)
    if problems:
        for problem in problems:
            print(f"invalid case: {problem}", file=sys.stderr)
        return 2

    settings = get_settings()
    if args.model:
        settings = settings.model_copy(update={"llm_model": args.model})

    if args.dry_run:
        print(f"{len(cases)} cases valid; {len(registry())} tools registered")
        print(f"model={settings.llm_model or '<unset>'} endpoint={settings.llm_base_url}")
        return 0

    if not settings.llm_model:
        print("NRP_OPS_LLM_MODEL is unset; pass --model", file=sys.stderr)
        return 2

    print(f"running {len(cases)} cases against {settings.llm_model}\n")
    scores = asyncio.run(run(cases, settings, args.only))

    summary = summarise(scores, settings.llm_model)
    print("\n" + json.dumps(summary, indent=2))

    if args.json:
        args.json.write_text(
            json.dumps(
                {"summary": summary, "scores": [vars(s) for s in scores]}, indent=2, default=str
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    return 0 if summary["passed"] == summary["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
