"""PromQL tool tests against a mocked HTTP transport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from nrp_ops_agent.config import Settings
from nrp_ops_agent.tools import dispatch
from nrp_ops_agent.tools import prometheus as prom
from nrp_ops_agent.tools.prometheus import PromqlArgs, promql

INSTANT_OK = {
    "status": "success",
    "data": {
        "resultType": "vector",
        "result": [
            {"metric": {"__name__": "up", "job": "coder"}, "value": [1755000000, "1"]},
            {"metric": {"__name__": "up", "job": "hub"}, "value": [1755000000, "0"]},
        ],
    },
}


def install(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
    **cfg: Any,
) -> list[httpx.Request]:
    """Point the tool at a mock transport and record the requests it makes."""
    seen: list[httpx.Request] = []

    def _handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    settings = Settings(_env_file=None, prometheus_base_url="https://thanos.test", **cfg)
    client = httpx.AsyncClient(
        base_url=settings.prometheus_base_url, transport=httpx.MockTransport(_handle)
    )
    monkeypatch.setattr(prom, "_client", client)
    monkeypatch.setattr(prom, "get_settings", lambda: settings)
    return seen


def ok(payload: dict[str, Any]) -> Any:
    return lambda request: httpx.Response(200, json=payload)


class TestInstantQueries:
    async def test_instant_query_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = install(monkeypatch, ok(INSTANT_OK))
        out = await promql(PromqlArgs(query="up"))
        assert out["query_type"] == "instant"
        assert out["series_count"] == 2
        assert out["series"][0]["value"] == [1755000000, "1"]
        assert seen[0].url.path == "/api/v1/query"
        assert "step" not in dict(seen[0].url.params)

    async def test_range_query_uses_query_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{"metric": {"job": "coder"}, "values": [[1, "1"], [2, "2"]]}],
            },
        }
        seen = install(monkeypatch, ok(payload))
        out = await promql(PromqlArgs(query="up", lookback="6h", step="5m"))
        assert out["query_type"] == "range"
        assert seen[0].url.path == "/api/v1/query_range"
        params = dict(seen[0].url.params)
        assert params["step"] == "5m"
        assert float(params["end"]) - float(params["start"]) == pytest.approx(21600, rel=0.01)


class TestGuards:
    async def test_series_cap_truncates_and_reports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [{"metric": {"i": str(i)}, "value": [0, "1"]} for i in range(50)],
            },
        }
        install(monkeypatch, ok(payload), prometheus_max_series=10)
        out = await promql(PromqlArgs(query="up"))
        assert out["series_count"] == 50
        assert out["truncated_series"] is True
        assert len(out["series"]) == 10

    async def test_sample_cap_keeps_both_ends(self, monkeypatch: pytest.MonkeyPatch) -> None:
        values = [[i, str(i)] for i in range(1000)]
        payload = {
            "status": "success",
            "data": {"resultType": "matrix", "result": [{"metric": {}, "values": values}]},
        }
        install(monkeypatch, ok(payload))
        out = await promql(PromqlArgs(query="up", step="1m"))
        assert out["truncated_samples"] is True
        kept = out["series"][0]["values"]
        assert len(kept) == prom.MAX_SAMPLES_PER_SERIES
        assert kept[0] == [0, "0"]
        assert kept[-1] == [999, "999"]

    async def test_absurd_range_is_refused_before_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = install(monkeypatch, ok(INSTANT_OK))
        result = await dispatch("promql", {"query": "up", "lookback": "30d", "step": "1s"})
        assert result.ok is False
        assert result.content["error"] == "range_too_large"
        assert seen == []

    @pytest.mark.parametrize("duration", ["1 h", "forever", "-5m", "5", "1hh", "abc"])
    async def test_malformed_durations_are_rejected(
        self, monkeypatch: pytest.MonkeyPatch, duration: str
    ) -> None:
        seen = install(monkeypatch, ok(INSTANT_OK))
        result = await dispatch("promql", {"query": "up", "lookback": duration, "step": "1m"})
        assert result.ok is False
        assert result.content["error"] == "invalid_arguments"
        assert seen == []

    async def test_overlong_query_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = install(monkeypatch, ok(INSTANT_OK))
        result = await dispatch("promql", {"query": "up" * 2000})
        assert result.ok is False
        assert seen == []


class TestErrorHandling:
    async def test_promql_syntax_error_is_reported_actionably(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = {"status": "error", "errorType": "bad_data", "error": "parse error at char 3"}
        install(monkeypatch, lambda r: httpx.Response(200, json=payload))
        result = await dispatch("promql", {"query": "up{"})
        assert result.ok is False
        assert result.content["error"] == "promql_error"
        assert "parse error" in result.content["message"]

    async def test_http_error_is_structured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, lambda r: httpx.Response(503, text="upstream down"))
        result = await dispatch("promql", {"query": "up"})
        assert result.ok is False
        assert result.content["error"] == "prometheus_error"

    async def test_timeout_is_structured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        install(monkeypatch, _timeout)
        result = await dispatch("promql", {"query": "up"})
        assert result.ok is False
        assert result.content["error"] == "timeout"

    async def test_non_json_body_is_structured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, lambda r: httpx.Response(200, text="<html>nope</html>"))
        result = await dispatch("promql", {"query": "up"})
        assert result.ok is False
        assert result.content["error"] == "prometheus_error"

    async def test_warnings_are_surfaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = dict(INSTANT_OK, warnings=["partial response from 1 of 3 stores"])
        install(monkeypatch, ok(payload))
        out = await promql(PromqlArgs(query="up"))
        assert out["warnings"] == ["partial response from 1 of 3 stores"]


class TestAuth:
    async def test_bearer_token_is_sent_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(prom, "_client", None)
        settings = Settings(
            _env_file=None,
            prometheus_base_url="https://thanos.test",
            prometheus_bearer_token="s3cret-token-value",
        )
        monkeypatch.setattr(prom, "get_settings", lambda: settings)
        client = prom._http()
        try:
            assert client.headers["Authorization"] == "Bearer s3cret-token-value"
        finally:
            await client.aclose()
            monkeypatch.setattr(prom, "_client", None)

    async def test_no_auth_header_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(prom, "_client", None)
        settings = Settings(_env_file=None, prometheus_base_url="https://thanos.test")
        monkeypatch.setattr(prom, "get_settings", lambda: settings)
        client = prom._http()
        try:
            assert "Authorization" not in client.headers
        finally:
            await client.aclose()
            monkeypatch.setattr(prom, "_client", None)

    async def test_the_token_never_reaches_the_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A Prometheus error body that echoes the request must not carry the
        credential back into context."""
        install(
            monkeypatch,
            lambda r: httpx.Response(401, text="Bearer s3cret-token-value-here rejected"),
            prometheus_bearer_token="s3cret-token-value-here",
        )
        result = await dispatch("promql", {"query": "up"})
        assert "s3cret-token-value-here" not in json.dumps(result.content)
