"""Tests for the audit trail's schema stability and its no-secrets invariant."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator

import pytest

from nrp_ops_agent import audit


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    buf = io.StringIO()
    audit.set_stream(buf)
    try:
        yield buf
    finally:
        audit.set_stream(None)


def _records(buf: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


def test_record_is_one_json_line_with_stable_keys(stream: io.StringIO) -> None:
    audit.emit("authz_deny", slack_user="U1", slack_team="T1", reason="unknown_user")
    (record,) = _records(stream)
    assert record["event"] == "authz_deny"
    assert record["schema"] == audit.SCHEMA_VERSION
    assert record["slack_user"] == "U1"
    assert record["reason"] == "unknown_user"
    assert isinstance(record["ts"], str) and record["ts"].endswith("+00:00")


def test_absent_fields_are_omitted_not_null(stream: io.StringIO) -> None:
    audit.emit("startup")
    (record,) = _records(stream)
    assert "tool" not in record
    assert "slack_user" not in record
    assert set(record) == {"ts", "schema", "correlation_id", "event"}


def test_correlation_id_binds_to_the_request(stream: io.StringIO) -> None:
    with audit.correlation() as cid:
        assert audit.current_correlation_id() == cid
        audit.emit("model_call", model="test-model")
        audit.emit("tool_call", tool="get_pod_status")
    assert audit.current_correlation_id() is None

    first, second = _records(stream)
    assert first["correlation_id"] == cid
    assert second["correlation_id"] == cid


def test_nested_correlations_restore_the_outer_id(stream: io.StringIO) -> None:
    with audit.correlation("outer"):
        with audit.correlation("inner"):
            audit.emit("tool_call", tool="a")
        audit.emit("tool_call", tool="b")
    inner, outer = _records(stream)
    assert inner["correlation_id"] == "inner"
    assert outer["correlation_id"] == "outer"


def test_tool_args_are_redacted(stream: io.StringIO) -> None:
    """Model-authored tool arguments reach the audit stream; a pasted credential
    must not survive the trip."""
    audit.emit(
        "tool_call",
        tool="promql",
        tool_args={"query": 'up{job="x"}', "header": "Bearer sk-abcdefghijklmnop0123"},
    )
    (record,) = _records(stream)
    args = record["tool_args"]
    assert isinstance(args, dict)
    assert args["query"] == 'up{job="x"}'
    assert "sk-abcdefghijklmnop0123" not in json.dumps(args)
    assert "[REDACTED:bearer_token]" in str(args["header"])


def test_long_tool_args_are_truncated(stream: io.StringIO) -> None:
    audit.emit("tool_call", tool="promql", tool_args={"query": "a" * 5000})
    (record,) = _records(stream)
    args = record["tool_args"]
    assert isinstance(args, dict)
    query = str(args["query"])
    assert len(query) < 700
    assert "truncated 5000 chars" in query


def test_result_bodies_are_never_emitted_only_sizes(stream: io.StringIO) -> None:
    audit.emit("tool_call", tool="get_logs", result_bytes=1_048_576, latency_ms=12.3456)
    (record,) = _records(stream)
    assert record["result_bytes"] == 1_048_576
    assert record["latency_ms"] == 12.35
    assert "result" not in record


def test_redaction_hits_are_recorded_by_rule_name(stream: io.StringIO) -> None:
    audit.emit("redaction_hit", tool="get_logs", redaction_rules=["jwt", "bearer_token"])
    (record,) = _records(stream)
    assert record["redaction_rules"] == ["jwt", "bearer_token"]


def test_timer_reports_milliseconds() -> None:
    timer = audit.Timer()
    assert timer.elapsed_ms >= 0.0
    assert timer.elapsed_ms < 5_000.0
