"""Structured audit logging.

Security role: this is the compliance evidence trail. Every authorization
decision, model call, tool call and outbound reply is emitted here as a single
JSON object per line, with a stable schema (documented in the README).

Two invariants:

* **No secret values.** Tool arguments are passed through :mod:`redact` before
  emission, and tool *results* are never emitted -- only their size.
* **No full log bodies.** Pod logs can be megabytes and are attacker-controlled;
  putting them in the audit stream would make the audit stream itself an
  injection and exfiltration surface.

Emission is to stdout by default so that the container runtime owns shipping,
rotation and retention.
"""

from __future__ import annotations

import contextvars
import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Final, Literal, TextIO

from nrp_ops_agent.redact import redact_structure

AuditEvent = Literal[
    "authz_allow",
    "authz_deny",
    "model_call",
    "model_error",
    "tool_call",
    "tool_error",
    "redaction_hit",
    "reply_sent",
    "thread_history",
    "startup",
    "policy_reload",
]

SCHEMA_VERSION: Final = 1

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nrp_ops_correlation_id", default=None
)

_write_lock = threading.Lock()
_stream_override: TextIO | None = None

# Argument values long enough to be a pasted credential or an injection payload
# are truncated: the audit trail needs to show *which* pod was queried, not to
# archive arbitrary model-authored text.
_MAX_ARG_CHARS: Final = 512


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:16]


@contextmanager
def correlation(correlation_id: str | None = None) -> Iterator[str]:
    """Bind a correlation ID for the duration of one Slack request."""
    cid = correlation_id or new_correlation_id()
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


def current_correlation_id() -> str | None:
    return _correlation_id.get()


def set_stream(stream: TextIO | None) -> None:
    """Redirect audit output. Used by tests; production uses stdout/stderr."""
    global _stream_override
    _stream_override = stream


def _stream() -> TextIO:
    if _stream_override is not None:
        return _stream_override
    # Read the env directly rather than importing config: audit must work during
    # startup failures, including a failure to construct Settings.
    if os.environ.get("NRP_OPS_AUDIT_STREAM", "stdout") == "stderr":
        return sys.stderr
    return sys.stdout


def _truncate_args(args: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if args is None:
        return None
    scrubbed, _ = redact_structure(dict(args))
    out: dict[str, Any] = {}
    for key, value in scrubbed.items():
        if isinstance(value, str) and len(value) > _MAX_ARG_CHARS:
            out[key] = value[:_MAX_ARG_CHARS] + f"...<truncated {len(value)} chars>"
        else:
            out[key] = value
    return out


def emit(
    event: AuditEvent,
    *,
    slack_user: str | None = None,
    slack_team: str | None = None,
    slack_channel: str | None = None,
    thread_ts: str | None = None,
    text: str | None = None,
    tool: str | None = None,
    tool_args: Mapping[str, Any] | None = None,
    result_bytes: int | None = None,
    latency_ms: float | None = None,
    redaction_rules: list[str] | None = None,
    model: str | None = None,
    reason: str | None = None,
    ok: bool | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one audit record. Returns the record, for tests and for callers
    that want to attach it to a reply."""
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "schema": SCHEMA_VERSION,
        "correlation_id": _correlation_id.get(),
        "event": event,
    }
    optional: dict[str, Any] = {
        "slack_user": slack_user,
        "slack_team": slack_team,
        "slack_channel": slack_channel,
        "thread_ts": thread_ts,
        "text": text,
        "tool": tool,
        "tool_args": _truncate_args(tool_args),
        "result_bytes": result_bytes,
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        "redaction_rules": redaction_rules or None,
        "model": model,
        "reason": reason,
        "ok": ok,
    }
    record.update({k: v for k, v in optional.items() if v is not None})
    if extra:
        record.update(dict(extra))

    line = json.dumps(record, default=str, ensure_ascii=False)
    stream = _stream()
    with _write_lock:
        stream.write(line + "\n")
        stream.flush()
    return record


class Timer:
    """Wall-clock stopwatch in milliseconds, for ``latency_ms`` fields."""

    __slots__ = ("_start",)

    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0
