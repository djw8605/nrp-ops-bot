"""Tests for the eval harness itself.

The scorer is the instrument used to decide whether a cheaper model is good
enough, so a scorer that quietly passes everything would be worse than no evals
at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from run_evals import Case, load_cases, main, score, summarise, validate

from nrp_ops_agent.agent import AgentResult, ToolCallRecord


def call(
    name: str, args: dict[str, object] | None = None, error: str | None = None
) -> ToolCallRecord:
    return ToolCallRecord(
        name=name,
        arguments=args or {},
        ok=error is None,
        latency_ms=1.0,
        size_bytes=10,
        error=error,
    )


class TestCaseFile:
    def test_twenty_cases_are_defined(self) -> None:
        assert len(load_cases()) == 20

    def test_case_ids_are_unique_and_tools_are_real(self) -> None:
        assert validate(load_cases()) == []

    def test_every_case_has_a_question(self) -> None:
        assert all(case.question.strip() for case in load_cases())

    def test_validation_catches_an_unknown_tool(self) -> None:
        problems = validate([Case(id="x", question="q", expect_tools=["kubectl"])])
        assert problems and "kubectl" in problems[0]

    def test_validation_catches_duplicate_ids(self) -> None:
        cases = [Case(id="dup", question="a"), Case(id="dup", question="b")]
        assert any("duplicate" in p for p in validate(cases))


class TestScoring:
    def test_a_correct_run_passes(self) -> None:
        case = Case(
            id="coder",
            question="coder down?",
            namespace="coder",
            workload="coder-0",
            expect_tools=["list_pods"],
            finding_keywords=["oomkilled"],
        )
        result = AgentResult(
            text="coder-0 was OOMKilled twice in the last hour.",
            tool_calls=[call("list_pods", {"namespace": "coder"})],
        )
        assert score(case, result, 1.0).passed is True

    def test_wrong_namespace_fails(self) -> None:
        case = Case(id="c", question="q", namespace="coder", expect_tools=[])
        result = AgentResult(text="all good", tool_calls=[call("list_pods", {"namespace": "jhub"})])
        assert score(case, result, 1.0).namespace_ok is False

    def test_missing_keyword_fails_the_finding(self) -> None:
        case = Case(id="c", question="q", finding_keywords=["oomkilled"])
        assert score(case, AgentResult(text="it restarted"), 1.0).finding_ok is False

    def test_missing_expected_tool_is_reported(self) -> None:
        case = Case(id="c", question="q", expect_tools=["promql", "list_pods"])
        outcome = score(case, AgentResult(text="x", tool_calls=[call("list_pods")]), 1.0)
        assert outcome.tools_ok is False
        assert outcome.missing_tools == ["promql"]

    def test_forbidden_tool_is_reported(self) -> None:
        case = Case(id="c", question="q", forbid_tools=["get_logs"])
        outcome = score(case, AgentResult(text="x", tool_calls=[call("get_logs")]), 1.0)
        assert outcome.tools_ok is False
        assert outcome.forbidden_called == ["get_logs"]

    def test_out_of_scope_case_needs_a_denial_or_an_explanation(self) -> None:
        case = Case(id="c", question="q", expect_denied=True)
        by_denial = score(
            case,
            AgentResult(text="nope", tool_calls=[call("list_pods", error="namespace_denied")]),
            1.0,
        )
        by_words = score(case, AgentResult(text="kube-system is out of scope"), 1.0)
        silent = score(case, AgentResult(text="everything looks fine"), 1.0)
        assert by_denial.denied_ok is True
        assert by_words.denied_ok is True
        assert silent.denied_ok is False

    def test_namespace_named_in_the_answer_also_counts(self) -> None:
        """A run that concluded without a namespaced call, but named the right
        namespace, has still identified it."""
        case = Case(id="c", question="q", namespace="rook-ceph")
        assert score(case, AgentResult(text="rook-ceph is degraded"), 1.0).namespace_ok is True

    def test_tool_count_is_recorded_not_scored(self) -> None:
        case = Case(id="c", question="q", expect_tools=["list_pods"])
        outcome = score(case, AgentResult(text="x", tool_calls=[call("list_pods")] * 9), 1.0)
        assert outcome.tool_count == 9
        assert outcome.tools_ok is True

    def test_row_renders_flags_for_each_dimension(self) -> None:
        case = Case(id="c", question="q", namespace="coder", finding_keywords=["nope"])
        row = score(case, AgentResult(text="all fine"), 1.0).row()
        assert row.startswith("FAIL")
        assert "f" in row  # lowercase flag marks the failed dimension


class TestSummary:
    def test_summary_aggregates_each_dimension(self) -> None:
        case_ok = Case(id="a", question="q")
        case_bad = Case(id="b", question="q", finding_keywords=["missing"])
        scores = [
            score(case_ok, AgentResult(text="fine", tool_calls=[call("list_pods")]), 1.0),
            score(case_bad, AgentResult(text="fine", tool_calls=[call("list_pods")]), 3.0),
        ]
        summary = summarise(scores, "test-model")
        assert summary["cases"] == 2
        assert summary["passed"] == 1
        assert summary["pass_rate"] == 0.5
        assert summary["finding_accuracy"] == 0.5
        assert summary["mean_tool_calls"] == 1.0
        assert summary["mean_latency_s"] == 2.0

    def test_empty_scores_do_not_divide_by_zero(self) -> None:
        assert summarise([], "m")["pass_rate"] == 0.0


class TestCli:
    def test_dry_run_validates_without_calling_anything(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "20 cases valid" in out

    def test_missing_model_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import run_evals

        from nrp_ops_agent.config import Settings

        monkeypatch.setattr(
            run_evals, "get_settings", lambda: Settings(_env_file=None, llm_model="")
        )
        assert main([]) == 2
