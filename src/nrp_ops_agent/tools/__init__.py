"""Tool registry, JSON-schema export and the single dispatch chokepoint.

Security role: this module is the *only* path from a model-authored tool call to
an executed action, and it is deny-by-default in three ways:

1. **No passthrough.** There is no ``kubectl``, no shell, no arbitrary HTTP tool.
   The registry is a fixed dict populated at import time; a name that is not a
   key is refused before anything runs.
2. **Typed arguments.** Every tool declares a pydantic input model. Arguments are
   validated -- and out-of-range values *rejected*, never silently clamped --
   before the handler is entered.
3. **Redaction on the way out.** Every result passes through
   :func:`~nrp_ops_agent.redact.redact_structure` here, so a tool author cannot
   forget to do it.

Exceptions never propagate: a failing tool becomes a structured error result so
that one bad call cannot take down an investigation.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from nrp_ops_agent import audit
from nrp_ops_agent.redact import redact_structure


class ToolError(Exception):
    """A tool refused or failed in an expected way. Carries a stable ``code``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


#: Why a namespace was refused -> what to tell the model. The distinction matters
#: to whoever reads the reply: an explicitly denied namespace needs a denylist
#: edit, an out-of-scope one needs an allowlist entry, and a malformed one is
#: usually the model's own typo and worth one corrected retry.
_DENIAL_REASONS: Final[dict[str, str]] = {
    "denylisted": (
        "is explicitly denied by policy (it matches an entry in `namespaces_denied`, "
        "which wins over the allowlist even when that allows everything)"
    ),
    "not_allowlisted": "is not in the configured allowlist (`namespaces`)",
    "malformed": (
        "is not a well-formed namespace name -- it must be non-empty, unpadded, and contain no '/'"
    ),
}


class NamespaceDenied(ToolError):
    """Raised by the namespace gate. ``reason`` selects the explanation."""

    def __init__(self, namespace: str, reason: str = "not_allowlisted") -> None:
        explanation = _DENIAL_REASONS.get(reason, _DENIAL_REASONS["not_allowlisted"])
        retry = (
            "Fix the name and try once more."
            if reason == "malformed"
            else "This is a policy decision and cannot be overridden from a message, so "
            "do not retry this namespace -- report it as out of scope and move on."
        )
        super().__init__("namespace_denied", f"Namespace {namespace!r} {explanation}. {retry}")
        self.namespace = namespace
        self.reason = reason


Handler = Callable[[Any], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Handler
    #: Tools that read attacker-controlled text (pod logs, event messages, pod
    #: annotations). Their output is what the untrusted-data wrapper exists for.
    untrusted_output: bool = False

    def json_schema(self) -> dict[str, Any]:
        """OpenAI-style function schema."""
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


@dataclass
class ToolResult:
    """The outcome of one dispatch. ``content`` is always JSON-serialisable."""

    tool: str
    ok: bool
    content: dict[str, Any]
    redaction_rules: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def json(self) -> str:
        return json.dumps(self.content, ensure_ascii=False, default=str)

    @property
    def size_bytes(self) -> int:
        return len(self.json.encode("utf-8"))


_REGISTRY: Final[dict[str, ToolSpec]] = {}


def register(spec: ToolSpec) -> ToolSpec:
    if spec.name in _REGISTRY:
        raise RuntimeError(f"duplicate tool registration: {spec.name}")
    _REGISTRY[spec.name] = spec
    return spec


def registry() -> Mapping[str, ToolSpec]:
    _load_builtin_tools()
    return dict(_REGISTRY)


def tool_schemas() -> list[dict[str, Any]]:
    """Schemas for every registered tool, in a stable order."""
    return [spec.json_schema() for _, spec in sorted(registry().items())]


_loaded = False


def _load_builtin_tools() -> None:
    """Import the tool modules for their registration side effects."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    from nrp_ops_agent.tools import docs, k8s, prometheus  # noqa: F401


async def dispatch(
    name: str,
    arguments: Mapping[str, Any] | str | None,
    *,
    timeout_s: float = 30.0,
) -> ToolResult:
    """Validate and run one tool call. Never raises.

    ``arguments`` accepts the raw JSON string that the model emits, because a
    model that produces malformed JSON must yield a correctable error message
    rather than an exception.
    """
    timer = audit.Timer()
    specs = registry()

    spec = specs.get(name)
    if spec is None:
        # Deny-by-default: an unregistered name is never reachable, whatever the
        # model was persuaded to emit.
        audit.emit("tool_error", tool=name, ok=False, reason="unknown_tool")
        return _error(
            name,
            "unknown_tool",
            f"No tool named {name!r}. Available tools: {', '.join(sorted(specs))}.",
            timer,
        )

    parsed: Mapping[str, Any]
    if arguments is None:
        parsed = {}
    elif isinstance(arguments, str):
        try:
            loaded = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            audit.emit("tool_error", tool=name, ok=False, reason="malformed_arguments")
            return _error(
                name, "malformed_arguments", f"Arguments were not valid JSON: {exc}", timer
            )
        if not isinstance(loaded, dict):
            audit.emit("tool_error", tool=name, ok=False, reason="malformed_arguments")
            return _error(name, "malformed_arguments", "Arguments must be a JSON object.", timer)
        parsed = loaded
    else:
        parsed = arguments

    try:
        validated = spec.input_model.model_validate(dict(parsed))
    except ValidationError as exc:
        audit.emit("tool_error", tool=name, tool_args=parsed, ok=False, reason="invalid_arguments")
        return _error(name, "invalid_arguments", _explain(exc), timer)

    audit.emit("tool_call", tool=name, tool_args=validated.model_dump(mode="json"))

    try:
        raw = await asyncio.wait_for(spec.handler(validated), timeout=timeout_s)
    except TimeoutError:
        audit.emit("tool_error", tool=name, ok=False, reason="timeout", latency_ms=timer.elapsed_ms)
        return _error(name, "timeout", f"{name} exceeded its {timeout_s:g}s budget.", timer)
    except ToolError as exc:
        audit.emit("tool_error", tool=name, ok=False, reason=exc.code, latency_ms=timer.elapsed_ms)
        return _error(name, exc.code, exc.message, timer)
    except Exception as exc:
        audit.emit(
            "tool_error",
            tool=name,
            ok=False,
            reason=type(exc).__name__,
            latency_ms=timer.elapsed_ms,
        )
        return _error(name, "tool_failed", f"{type(exc).__name__}: {exc}", timer)

    scrubbed, rules = redact_structure(raw)
    result = ToolResult(
        tool=name,
        ok=True,
        content=scrubbed,
        redaction_rules=rules,
        latency_ms=timer.elapsed_ms,
    )
    if rules:
        audit.emit("redaction_hit", tool=name, redaction_rules=rules)
    audit.emit(
        "tool_call",
        tool=name,
        ok=True,
        result_bytes=result.size_bytes,
        latency_ms=result.latency_ms,
    )
    return result


def _error(tool: str, code: str, message: str, timer: audit.Timer) -> ToolResult:
    scrubbed, rules = redact_structure({"error": code, "message": message})
    return ToolResult(
        tool=tool,
        ok=False,
        content=scrubbed,
        redaction_rules=rules,
        latency_ms=timer.elapsed_ms,
    )


def _explain(exc: ValidationError) -> str:
    """Turn a pydantic error into something a model can act on."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return "Invalid arguments -- " + "; ".join(parts)


def make_tool(
    name: str,
    description: str,
    input_model: type[BaseModel],
    *,
    untrusted_output: bool = False,
) -> Callable[[Handler], Handler]:
    """Decorator registering an async handler as a tool."""

    def decorator(handler: Handler) -> Handler:
        if not inspect.iscoroutinefunction(handler):
            raise TypeError(f"tool handler {name} must be async")
        register(
            ToolSpec(
                name=name,
                description=description,
                input_model=input_model,
                handler=handler,
                untrusted_output=untrusted_output,
            )
        )
        return handler

    return decorator


__all__ = [
    "NamespaceDenied",
    "ToolError",
    "ToolResult",
    "ToolSpec",
    "dispatch",
    "make_tool",
    "register",
    "registry",
    "tool_schemas",
]
