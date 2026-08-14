"""Agent loop tests: budgets, error containment and message shape.

No real model is contacted; ``_complete`` is replaced with a scripted stand-in so
that the loop's control flow is what is under test.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from helpers import ns
from nrp_ops_agent.agent import Agent, ModelError
from nrp_ops_agent.config import Settings


def tool_turn(name: str, args: dict[str, Any], call_id: str = "c1") -> dict[str, Any]:
    return {
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def answer(text: str) -> dict[str, Any]:
    return {"content": text, "tool_calls": []}


class Scripted:
    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self.turns = list(turns)
        self.seen: list[list[dict[str, Any]]] = []
        self.calls = 0

    async def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls += 1
        self.seen.append([dict(m) for m in messages])
        return self.turns.pop(0) if self.turns else answer("out of script")


def make_agent(
    monkeypatch: pytest.MonkeyPatch, turns: list[dict[str, Any]], **cfg: Any
) -> tuple[Agent, Scripted]:
    settings = Settings(_env_file=None, slack_team_id="T1", llm_model="test-model", **cfg)
    agent = Agent(settings)
    model = Scripted(turns)
    monkeypatch.setattr(agent, "_complete", model)
    return agent, model


@pytest.mark.usefixtures("allow_all_namespaces")
class TestLoopControl:
    async def test_answers_without_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, model = make_agent(monkeypatch, [answer("nothing is wrong")])
        result = await agent.investigate("is anything broken?")
        assert result.text == "nothing is wrong"
        assert result.stop_reason == "answered"
        assert result.tool_calls == []
        assert model.calls == 1

    async def test_runs_a_tool_then_answers(
        self, monkeypatch: pytest.MonkeyPatch, fake_kube: Any
    ) -> None:
        fake_kube.core._responses["list_namespaced_pod"] = ns(items=[], metadata=ns(_continue=None))
        agent, _ = make_agent(
            monkeypatch,
            [tool_turn("list_pods", {"namespace": "coder"}), answer("coder has no pods")],
        )
        result = await agent.investigate("coder down?")
        assert result.stop_reason == "answered"
        assert [c.name for c in result.tool_calls] == ["list_pods"]
        assert result.tool_calls[0].ok is True

    async def test_tool_call_budget_is_enforced(
        self, monkeypatch: pytest.MonkeyPatch, fake_kube: Any
    ) -> None:
        fake_kube.core._responses["list_namespaced_pod"] = ns(items=[], metadata=ns(_continue=None))
        turns = [tool_turn("list_pods", {"namespace": "coder"}, f"c{i}") for i in range(8)]
        agent, _ = make_agent(monkeypatch, turns, agent_max_tool_calls=3, agent_max_iterations=8)
        result = await agent.investigate("coder down?")
        assert len(result.tool_calls) == 3
        assert result.stop_reason in ("tool_budget", "iteration_budget")

    async def test_iteration_budget_is_enforced(
        self, monkeypatch: pytest.MonkeyPatch, fake_kube: Any
    ) -> None:
        fake_kube.core._responses["list_namespaced_pod"] = ns(items=[], metadata=ns(_continue=None))
        turns = [tool_turn("list_pods", {"namespace": "coder"}, f"c{i}") for i in range(20)]
        agent, model = make_agent(
            monkeypatch, turns, agent_max_iterations=2, agent_max_tool_calls=12
        )
        result = await agent.investigate("coder down?")
        assert result.iterations == 2
        assert model.calls == 2
        assert result.stop_reason == "iteration_budget"
        assert "iteration limit" in result.text

    async def test_parallel_tool_calls_in_one_turn(
        self, monkeypatch: pytest.MonkeyPatch, fake_kube: Any
    ) -> None:
        fake_kube.core._responses["list_namespaced_pod"] = ns(items=[], metadata=ns(_continue=None))
        fake_kube.core._responses["list_namespaced_event"] = ns(items=[])
        turn = {
            "content": None,
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "list_pods", "arguments": '{"namespace":"coder"}'},
                },
                {
                    "id": "b",
                    "type": "function",
                    "function": {"name": "get_events", "arguments": '{"namespace":"coder"}'},
                },
            ],
        }
        agent, model = make_agent(monkeypatch, [turn, answer("both checked")])
        result = await agent.investigate("coder?")
        assert [c.name for c in result.tool_calls] == ["list_pods", "get_events"]
        tool_msgs = [m for m in model.seen[-1] if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b"]


@pytest.mark.usefixtures("allow_all_namespaces")
class TestMessageShape:
    async def test_operator_message_is_marked_trusted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, model = make_agent(monkeypatch, [answer("ok")])
        await agent.investigate("coder is down")
        user_msg = next(m for m in model.seen[0] if m["role"] == "user")
        assert str(user_msg["content"]).startswith("<trusted_operator_request>")

    async def test_playbook_is_injected_for_a_matching_service(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, model = make_agent(monkeypatch, [answer("ok")])
        await agent.investigate("coder is not responding")
        system = str(next(m for m in model.seen[0] if m["role"] == "system")["content"])
        assert "Playbook: coder" in system
        assert "get_workload" in system

    async def test_allowed_namespaces_are_stated_in_the_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, model = make_agent(monkeypatch, [answer("ok")])
        await agent.investigate("hello")
        system = str(next(m for m in model.seen[0] if m["role"] == "system")["content"])
        assert "Namespaces you may query" in system

    async def test_tool_results_are_wrapped(
        self, monkeypatch: pytest.MonkeyPatch, fake_kube: Any
    ) -> None:
        fake_kube.core._responses["list_namespaced_pod"] = ns(items=[], metadata=ns(_continue=None))
        agent, model = make_agent(
            monkeypatch, [tool_turn("list_pods", {"namespace": "coder"}), answer("ok")]
        )
        await agent.investigate("coder?")
        tool_msg = next(m for m in model.seen[-1] if m.get("role") == "tool")
        assert str(tool_msg["content"]).startswith('<untrusted_tool_output tool="list_pods">')


@pytest.mark.usefixtures("allow_all_namespaces")
class TestOutputBudget:
    async def test_oldest_results_are_dropped_and_the_model_is_told(
        self, monkeypatch: pytest.MonkeyPatch, fake_kube: Any
    ) -> None:
        # Each log call returns ~40k characters; a 10k-token budget is 40k chars.
        fake_kube.core._responses["read_namespaced_pod_log"] = "x" * 40_000
        turns = [
            tool_turn("get_logs", {"namespace": "coder", "pod": f"p{i}"}, f"c{i}") for i in range(3)
        ]
        turns.append(answer("done"))
        agent, model = make_agent(
            monkeypatch, turns, agent_tool_output_token_budget=10_000, agent_max_iterations=5
        )
        result = await agent.investigate("coder logs")
        assert result.dropped_results > 0

        final = model.seen[-1]
        dropped = [
            m for m in final if m.get("role") == "tool" and str(m["content"]).startswith("[dropped")
        ]
        assert dropped
        notes = [m for m in final if m.get("role") == "system" and "dropped" in str(m["content"])]
        assert notes, "the model must be told that results were dropped"

    async def test_small_results_are_never_dropped(
        self, monkeypatch: pytest.MonkeyPatch, fake_kube: Any
    ) -> None:
        fake_kube.core._responses["read_namespaced_pod_log"] = "short line"
        agent, _ = make_agent(
            monkeypatch,
            [tool_turn("get_logs", {"namespace": "coder", "pod": "p"}), answer("done")],
        )
        result = await agent.investigate("logs")
        assert result.dropped_results == 0


@pytest.mark.usefixtures("allow_all_namespaces")
class TestFailureContainment:
    async def test_model_error_becomes_a_reported_outcome(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, _ = make_agent(monkeypatch, [])

        async def _boom(messages: list[dict[str, Any]]) -> dict[str, Any]:
            raise ModelError("gateway exploded")

        monkeypatch.setattr(agent, "_complete", _boom)
        result = await agent.investigate("coder?")
        assert result.stop_reason == "model_error"
        assert result.ok is False
        assert "gateway exploded" in result.text

    async def test_tool_failure_does_not_end_the_turn(
        self, monkeypatch: pytest.MonkeyPatch, fake_kube: Any
    ) -> None:
        fake_kube.core._responses["list_namespaced_pod"] = RuntimeError("api on fire")
        agent, _ = make_agent(
            monkeypatch,
            [tool_turn("list_pods", {"namespace": "coder"}), answer("the API is unhealthy")],
        )
        result = await agent.investigate("coder?")
        assert result.tool_calls[0].ok is False
        assert result.stop_reason == "answered"
        assert result.text == "the API is unhealthy"

    async def test_empty_model_content_yields_a_useful_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, _ = make_agent(monkeypatch, [answer("")])
        result = await agent.investigate("coder?")
        assert "produced no answer" in result.text

    async def test_progress_callback_is_invoked(
        self, monkeypatch: pytest.MonkeyPatch, fake_kube: Any
    ) -> None:
        fake_kube.core._responses["list_namespaced_pod"] = ns(items=[], metadata=ns(_continue=None))
        agent, _ = make_agent(
            monkeypatch, [tool_turn("list_pods", {"namespace": "coder"}), answer("ok")]
        )
        seen: list[str] = []

        async def progress(line: str) -> None:
            seen.append(line)

        await agent.investigate("coder?", progress=progress)
        assert seen == ["checking: list_pods"]
