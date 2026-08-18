"""The tool-calling loop.

Hand-rolled against the OpenAI-compatible chat completions API on purpose. An
agent framework would put a large, changing dependency between attacker
controlled cluster text and the tool dispatcher; this file is short enough that
an auditor can read every path that text takes.

Security role:

* Every tool result is wrapped by :func:`~nrp_ops_agent.prompts.wrap_untrusted`
  before it is appended to the conversation, with delimiters escaped so a log
  line cannot break out.
* Tool names come from the registry only -- an unknown name is refused by
  :func:`~nrp_ops_agent.tools.dispatch`, never executed.
* Nothing here can raise into the Slack handler: model faults, tool faults and
  timeouts all become a reported outcome.

* Charts drawn by a tool are collected out of band (see
  :mod:`nrp_ops_agent.charts`) and returned on :class:`AgentResult`. Image bytes
  never enter the conversation -- the model sees only the numbers it plotted.

Budgets (all env-driven): max tool calls per turn, max loop iterations, wall
clock timeout, a cap on accumulated tool output, and a cap on charts per reply.
When the output cap is exceeded the oldest tool results are dropped and the
model is *told* they were dropped, so it does not silently reason from a
truncated picture.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import httpx

from nrp_ops_agent import audit, charts
from nrp_ops_agent.config import Settings, get_allowlist_loader, get_settings
from nrp_ops_agent.playbooks import get_playbooks
from nrp_ops_agent.prompts import (
    build_system_prompt,
    wrap_operator_message,
    wrap_untrusted,
    wrap_untrusted_thread_message,
)
from nrp_ops_agent.tools import ToolResult, dispatch, tool_schemas

#: Rough characters-per-token. Used only to enforce the output budget, where a
#: 25% error changes nothing that matters.
CHARS_PER_TOKEN: Final = 4

ProgressFn = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class ThreadMessage:
    """One earlier Slack message, already classified by the caller.

    The classification is the security-relevant part and is deliberately not
    inferred here: :mod:`nrp_ops_agent.slack_app` decides who is an operator by
    the same allowlist :mod:`nrp_ops_agent.authz` uses, so there is one answer to
    "is this person trusted" rather than two that can drift apart.
    """

    author: str
    text: str
    #: A message the bot itself posted. Replayed as an assistant turn.
    is_bot: bool = False
    #: Author is in ``operators``. Only these are replayed as instructions.
    is_operator: bool = False

    def render(self) -> dict[str, Any]:
        if self.is_bot:
            return {"role": "assistant", "content": self.text}
        if self.is_operator:
            return {"role": "user", "content": wrap_operator_message(self.text)}
        return {
            "role": "user",
            "content": wrap_untrusted_thread_message(self.author, self.text),
        }


@dataclass(frozen=True)
class ToolCallRecord:
    """One executed tool call, for the audit trail and the Slack tool trace."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    latency_ms: float
    size_bytes: int
    error: str | None = None

    def render(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        status = "" if self.ok else f"  -> {self.error}"
        return f"{self.name}({args}){status}"


@dataclass
class AgentResult:
    text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    stop_reason: str = "answered"
    redaction_rules: list[str] = field(default_factory=list)
    dropped_results: int = 0
    #: Images drawn during this turn, for the caller to upload. They travel
    #: beside the text rather than inside it: a tool result is JSON in the model
    #: context, and a PNG is neither.
    charts: list[charts.Chart] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.stop_reason in ("answered", "tool_budget", "iteration_budget")


class ModelError(Exception):
    pass


class Agent:
    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client

    # -------------------------------------------------------------- plumbing --

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            key = self._settings.llm_api_key.get_secret_value()
            if key:
                headers["Authorization"] = f"Bearer {key}"
            self._client = httpx.AsyncClient(
                base_url=self._settings.llm_base_url,
                headers=headers,
                timeout=self._settings.llm_timeout_s,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """One chat completion. Retries once on a transient transport failure."""
        payload = {
            "model": self._settings.llm_model,
            "messages": messages,
            "tools": tool_schemas(),
            "tool_choice": "auto",
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_tokens,
        }
        timer = audit.Timer()
        last_exc: Exception | None = None
        for attempt in (1, 2):
            try:
                resp = await self._http().post("/chat/completions", json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt == 1:
                    await asyncio.sleep(1.0)
                    continue
                break
            if resp.status_code >= 500 and attempt == 1:
                await asyncio.sleep(1.0)
                continue
            if resp.status_code >= 400:
                audit.emit(
                    "model_error",
                    model=self._settings.llm_model,
                    ok=False,
                    reason=f"http_{resp.status_code}",
                    latency_ms=timer.elapsed_ms,
                )
                raise ModelError(
                    f"LLM endpoint returned HTTP {resp.status_code}: {resp.text[:300]}"
                )
            try:
                body: dict[str, Any] = resp.json()
            except ValueError as exc:
                raise ModelError("LLM endpoint returned a non-JSON body.") from exc

            choices = body.get("choices") or []
            if not choices:
                raise ModelError("LLM endpoint returned no choices.")
            usage = body.get("usage") or {}
            audit.emit(
                "model_call",
                model=self._settings.llm_model,
                ok=True,
                latency_ms=timer.elapsed_ms,
                extra={
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                },
            )
            message: dict[str, Any] = choices[0].get("message") or {}
            return message

        audit.emit(
            "model_error",
            model=self._settings.llm_model,
            ok=False,
            reason=type(last_exc).__name__ if last_exc else "unreachable",
            latency_ms=timer.elapsed_ms,
        )
        raise ModelError(f"Could not reach the LLM endpoint: {last_exc}")

    # ------------------------------------------------------------ the loop --

    async def investigate(
        self,
        message: str,
        *,
        history: Sequence[ThreadMessage] | None = None,
        progress: ProgressFn | None = None,
    ) -> AgentResult:
        """One investigation, with any charts it drew attached to the result.

        The collector wraps the whole loop rather than each tool call so that a
        turn's chart budget is a turn's, not a tool's, and so that every exit
        path -- answered, timed out, budget exhausted -- carries the pictures
        drawn before it.
        """
        with charts.collecting(self._settings.charts_max_per_turn) as collector:
            result = await self._run(message, history=history, progress=progress)
        result.charts = list(collector.charts)
        return result

    async def _run(
        self,
        message: str,
        *,
        history: Sequence[ThreadMessage] | None = None,
        progress: ProgressFn | None = None,
    ) -> AgentResult:
        settings = self._settings
        deadline = time.monotonic() + settings.agent_wall_clock_timeout_s

        # Playbook selection reads the thread too: "restart it" on its own
        # matches nothing, while the operator's earlier "coder-0 is OOMKilling"
        # picks the right playbook for the follow-up.
        history = tuple(history or ())
        selection_text = "\n".join([*(m.text for m in history if not m.is_bot), message])
        playbook = get_playbooks().select(selection_text)
        namespaces = get_allowlist_loader().load().namespaces
        system = build_system_prompt(playbook, allowed_namespaces=namespaces)

        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages.extend(m.render() for m in history)
        messages.append({"role": "user", "content": wrap_operator_message(message)})

        result = AgentResult(text="")
        calls_made = 0

        for iteration in range(1, settings.agent_max_iterations + 1):
            result.iterations = iteration

            if time.monotonic() >= deadline:
                result.stop_reason = "timeout"
                result.text = self._partial(result, "ran out of time")
                return result

            try:
                reply = await asyncio.wait_for(
                    self._complete(messages), timeout=max(1.0, deadline - time.monotonic())
                )
            except TimeoutError:
                result.stop_reason = "timeout"
                result.text = self._partial(result, "ran out of time waiting for the model")
                return result
            except ModelError as exc:
                result.stop_reason = "model_error"
                result.text = f"I could not complete the investigation: {exc}"
                return result

            tool_calls = reply.get("tool_calls") or []
            content = (reply.get("content") or "").strip()

            if not tool_calls:
                result.text = content or self._partial(result, "produced no answer")
                result.stop_reason = "answered"
                return result

            # Keep the assistant turn verbatim: the tool_call ids must match the
            # tool messages that follow, or the next request is rejected.
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.get("content"),
                    "tool_calls": tool_calls,
                }
            )

            remaining = settings.agent_max_tool_calls - calls_made
            if remaining <= 0:
                result.stop_reason = "tool_budget"
                messages.append(
                    self._system_note(
                        "Tool call budget exhausted. Answer now from what you already have, "
                        "and say what you did not get to check."
                    )
                )
                continue

            accepted = tool_calls[:remaining]
            calls_made += len(accepted)

            if progress is not None:
                await progress(self._progress_line(accepted))

            budget_s = max(1.0, deadline - time.monotonic())
            executed = await asyncio.gather(*(self._run_one(call, budget_s) for call in accepted))

            for call, tool_result in executed:
                record = self._record(call, tool_result)
                result.tool_calls.append(record)
                result.redaction_rules.extend(tool_result.redaction_rules)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "name": record.name,
                        "content": wrap_untrusted(record.name, tool_result.json),
                    }
                )

            if len(tool_calls) > len(accepted):
                messages.append(
                    self._system_note(
                        f"{len(tool_calls) - len(accepted)} further tool call(s) were not run: "
                        "the per-turn budget is exhausted."
                    )
                )

            dropped = self._enforce_output_budget(messages)
            if dropped:
                result.dropped_results += dropped
                messages.append(
                    self._system_note(
                        f"{dropped} earlier tool result(s) were dropped from context to stay "
                        "within the output budget. Do not assume anything about their contents; "
                        "re-run a specific tool if you need it again."
                    )
                )

        result.stop_reason = "iteration_budget"
        result.text = self._partial(result, "hit the iteration limit")
        return result

    # ------------------------------------------------------------- helpers --

    async def _run_one(
        self, call: dict[str, Any], budget_s: float
    ) -> tuple[dict[str, Any], ToolResult]:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments")
        tool_result = await dispatch(name, arguments, timeout_s=min(30.0, budget_s))
        return call, tool_result

    @staticmethod
    def _record(call: dict[str, Any], tool_result: ToolResult) -> ToolCallRecord:
        function = call.get("function") or {}
        raw_args = function.get("arguments")
        parsed: dict[str, Any]
        if isinstance(raw_args, dict):
            parsed = raw_args
        else:
            try:
                loaded = json.loads(raw_args or "{}")
                parsed = loaded if isinstance(loaded, dict) else {"_raw": str(raw_args)}
            except (json.JSONDecodeError, TypeError):
                parsed = {"_raw": str(raw_args)[:200]}
        return ToolCallRecord(
            name=str(function.get("name") or "?"),
            arguments=parsed,
            ok=tool_result.ok,
            latency_ms=tool_result.latency_ms,
            size_bytes=tool_result.size_bytes,
            error=None if tool_result.ok else str(tool_result.content.get("error")),
        )

    @staticmethod
    def _system_note(text: str) -> dict[str, Any]:
        return {"role": "system", "content": text}

    @staticmethod
    def _progress_line(calls: list[dict[str, Any]]) -> str:
        names = [str((c.get("function") or {}).get("name") or "?") for c in calls]
        return "checking: " + ", ".join(names)

    def _enforce_output_budget(self, messages: list[dict[str, Any]]) -> int:
        """Drop the oldest tool results until the accumulated output fits.

        Returns how many were dropped. The corresponding assistant tool_calls are
        left in place and their content replaced, so the id pairing the API
        requires stays intact.
        """
        budget_chars = self._settings.agent_tool_output_token_budget * CHARS_PER_TOKEN
        total = sum(len(str(m.get("content") or "")) for m in messages if m.get("role") == "tool")
        dropped = 0
        for msg in messages:
            if total <= budget_chars:
                break
            if msg.get("role") != "tool":
                continue
            content = str(msg.get("content") or "")
            if content.startswith("[dropped"):
                continue
            total -= len(content)
            msg["content"] = "[dropped to stay within the tool output budget]"
            dropped += 1
        return dropped

    @staticmethod
    def _partial(result: AgentResult, why: str) -> str:
        checked = ", ".join(dict.fromkeys(c.name for c in result.tool_calls)) or "nothing"
        return (
            f"I {why} before reaching a conclusion. Tools run so far: {checked}. "
            "Ask again to narrow the question, or name the namespace directly."
        )
