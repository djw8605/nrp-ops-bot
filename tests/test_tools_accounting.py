"""Accounting tool tests against a mocked MCP transport.

The interesting surface is the JSON-RPC envelope: the server is stateless, so
one POST is the whole conversation, and everything that can go wrong -- a tool
error inside a 200, an SSE-framed body, a truncated row set -- has to be handled
from that single response.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from nrp_ops_agent import charts
from nrp_ops_agent.config import Settings
from nrp_ops_agent.tools import accounting, dispatch

ENDPOINT = "https://accounting.test/"


def install(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
    **cfg: Any,
) -> list[httpx.Request]:
    """Point the tools at a mock transport and record the requests they make."""
    seen: list[httpx.Request] = []

    def _handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    cfg.setdefault("accounting_mcp_url", ENDPOINT)
    settings = Settings(_env_file=None, **cfg)
    client = httpx.AsyncClient(transport=httpx.MockTransport(_handle))
    monkeypatch.setattr(accounting, "_client", client)
    monkeypatch.setattr(accounting, "get_settings", lambda: settings)
    monkeypatch.setattr(charts, "get_settings", lambda: settings, raising=False)
    return seen


def result(payload: dict[str, Any]) -> Any:
    """A successful CallToolResult carrying ``payload`` as structured content."""

    def _respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload)}],
                    "structuredContent": payload,
                    "isError": False,
                },
            },
        )

    return _respond


def results(by_tool: dict[str, dict[str, Any]]) -> Any:
    """Route each MCP tool name to its own payload.

    Some tools make two calls -- the breakdown chart resolves its date window
    from ``get_latest_data_date`` before it queries anything -- so a single
    canned response would answer the wrong question.
    """

    def _respond(request: httpx.Request) -> httpx.Response:
        name = json.loads(request.content)["params"]["name"]
        assert name in by_tool, f"unexpected MCP tool call: {name}"
        return result(by_tool[name])(request)

    return _respond


def sent(request: httpx.Request) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(request.content)
    return body


TOP_ROWS = {
    "dimension": "namespace",
    "resource": "gpu",
    "start_date": "2026-07-01",
    "end_date": "2026-07-31",
    "row_count": 2,
    "rows": [
        {"namespace": "braingeneers", "unit": "gpu_hours", "usage": 184320.0},
        {"namespace": "coder", "unit": "gpu_hours", "usage": 71204.5},
    ],
}

TREND_ROWS = {
    "dimension": "namespace",
    "value": "coder",
    "resource": "gpu",
    "start_date": "2026-07-01",
    "end_date": "2026-07-03",
    "row_count": 3,
    "rows": [
        {"date": "2026-07-02", "unit": "gpu_hours", "usage": 240.0},
        {"date": "2026-07-01", "unit": "gpu_hours", "usage": 120.0},
        {"date": "2026-07-03", "unit": "gpu_hours", "usage": 300.0},
    ],
}


LLM_DAILY_ROWS = {
    "metric": "tokens_used",
    "start_date": "2026-08-11",
    "end_date": "2026-08-12",
    "rows": [
        {"date": "2026-08-11", "model": "deepseek-v4-flash", "tokens_used": 1000.0},
        {"date": "2026-08-12", "model": "deepseek-v4-flash", "tokens_used": 1600.0},
        {"date": "2026-08-11", "model": "gemma", "tokens_used": 40.0},
        {"date": "2026-08-12", "model": "gemma", "tokens_used": 57.0},
    ],
}


class TestEnvelope:
    async def test_call_is_a_stateless_jsonrpc_post(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = install(monkeypatch, result(TOP_ROWS))
        await accounting.accounting_top(
            accounting.TopArgs(dimension="namespace", resource="gpu", limit=2)
        )

        request = seen[0]
        assert request.method == "POST"
        assert str(request.url) == ENDPOINT
        body = sent(request)
        assert body["method"] == "tools/call"
        assert body["params"]["name"] == "top_resource_consumers"
        assert body["params"]["arguments"] == {
            "dimension": "namespace",
            "resource": "gpu",
            "limit": 2,
        }

    async def test_accept_header_carries_both_media_types(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The transport 406s a request that does not accept SSE, even when the
        # server is configured to answer in JSON.
        seen = install(monkeypatch, result(TOP_ROWS))
        await accounting.accounting_top(accounting.TopArgs(dimension="namespace", resource="gpu"))
        accept = seen[0].headers["accept"]
        assert "application/json" in accept
        assert "text/event-stream" in accept

    async def test_no_initialize_handshake(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = install(monkeypatch, result(TOP_ROWS))
        await accounting.accounting_top(accounting.TopArgs(dimension="namespace", resource="gpu"))
        assert len(seen) == 1

    async def test_none_arguments_are_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = install(monkeypatch, result(TOP_ROWS))
        await accounting.accounting_top(accounting.TopArgs(dimension="namespace", resource="gpu"))
        assert "institution" not in sent(seen[0])["params"]["arguments"]

    async def test_sse_framed_body_is_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        envelope = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [], "structuredContent": TOP_ROWS, "isError": False},
        }
        install(
            monkeypatch,
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"event: message\ndata: {json.dumps(envelope)}\n\n".encode(),
            ),
        )
        out = await accounting.accounting_top(
            accounting.TopArgs(dimension="namespace", resource="gpu")
        )
        assert out["rows"][0]["namespace"] == "braingeneers"

    async def test_text_content_is_parsed_when_unstructured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(
            monkeypatch,
            lambda request: httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": [{"type": "text", "text": json.dumps(TOP_ROWS)}]},
                },
            ),
        )
        out = await accounting.accounting_top(
            accounting.TopArgs(dimension="namespace", resource="gpu")
        )
        assert out["row_count"] == 2


class TestFailures:
    async def test_tool_error_inside_a_200_reaches_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The server reports its own validation failures this way. They are
        # correctable, so the text has to survive intact.
        install(
            monkeypatch,
            lambda request: httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "content": [{"type": "text", "text": "unknown resource: 'gpus'"}],
                        "isError": True,
                    },
                },
            ),
        )
        out = await dispatch("accounting_top", {"dimension": "namespace", "resource": "gpu"})
        assert not out.ok
        assert out.content["error"] == "accounting_rejected"
        assert "unknown resource" in out.content["message"]

    async def test_jsonrpc_error_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(
            monkeypatch,
            lambda request: httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "bad"}},
            ),
        )
        out = await dispatch("accounting_top", {"dimension": "namespace", "resource": "gpu"})
        assert not out.ok
        assert out.content["error"] == "accounting_error"

    async def test_http_error_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, lambda request: httpx.Response(502, text="bad gateway"))
        out = await dispatch("accounting_top", {"dimension": "namespace", "resource": "gpu"})
        assert not out.ok
        assert "502" in out.content["message"]

    async def test_transport_failure_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        install(monkeypatch, _boom)
        out = await dispatch("accounting_top", {"dimension": "namespace", "resource": "gpu"})
        assert not out.ok
        assert out.content["error"] == "accounting_unreachable"

    async def test_unconfigured_endpoint_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, result(TOP_ROWS), accounting_mcp_url="")
        out = await dispatch("accounting_top", {"dimension": "namespace", "resource": "gpu"})
        assert not out.ok
        assert out.content["error"] == "accounting_not_configured"


class TestArgumentValidation:
    """Out-of-range arguments are rejected before anything is sent."""

    @pytest.mark.parametrize(
        "args",
        [
            {"dimension": "namespace", "resource": "gpu_hours"},  # alias, not canonical
            {"dimension": "pod_name", "resource": "gpu"},  # not a rankable dimension
            {"dimension": "namespace", "resource": "gpu", "limit": 0},
            {"dimension": "namespace", "resource": "gpu", "limit": 500},
            {"dimension": "namespace", "resource": "gpu", "start_date": "July 2026"},
            {"dimension": "namespace", "resource": "gpu", "unknown": 1},
        ],
    )
    async def test_rejected(self, monkeypatch: pytest.MonkeyPatch, args: dict[str, Any]) -> None:
        seen = install(monkeypatch, result(TOP_ROWS))
        out = await dispatch("accounting_top", args)
        assert not out.ok
        assert out.content["error"] == "invalid_arguments"
        assert seen == []


class TestShaping:
    async def test_rows_are_capped_and_flagged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        wide = {
            "row_count": 10,
            "rows": [
                {"namespace": f"ns-{i}", "unit": "gpu_hours", "usage": 1.0} for i in range(10)
            ],
        }
        install(monkeypatch, result(wide), accounting_max_rows=4)
        out = await accounting.accounting_top(
            accounting.TopArgs(dimension="namespace", resource="gpu")
        )
        assert len(out["rows"]) == 4
        assert out["rows_truncated"] is True
        assert out["rows_returned"] == 4

    async def test_discover_asks_for_the_latest_date_and_the_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, result({"latest_date": "2026-08-16", "values": []}))
        out = await accounting.accounting_discover(
            accounting.DiscoverArgs(dimension="namespace", limit=5)
        )
        assert "latest_data_date" in out
        assert "values" in out

    async def test_discover_without_a_dimension_makes_one_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = install(monkeypatch, result({"latest_date": "2026-08-16"}))
        out = await accounting.accounting_discover(accounting.DiscoverArgs())
        assert len(seen) == 1
        assert sent(seen[0])["params"]["name"] == "get_latest_data_date"
        assert "values" not in out


class TestCharts:
    async def test_trend_chart_is_attached_and_returns_its_numbers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, result(TREND_ROWS))
        with charts.collecting(2) as collector:
            out = await accounting.accounting_chart(
                accounting.ChartArgs(
                    kind="trend", resource="gpu", dimension="namespace", value="coder"
                )
            )

        assert out["attached"] is True
        assert out["unit"] == "GPU hours"
        assert len(collector.charts) == 1
        assert collector.charts[0].png.startswith(b"\x89PNG")
        # Rows arrive unordered; the chart and its summary must be by date.
        assert {"first_date": "2026-07-01", "first": 120.0} in out["plotted"]
        assert {"last_date": "2026-07-03", "last": 300.0} in out["plotted"]

    async def test_ranking_chart_plots_every_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, result(TOP_ROWS))
        with charts.collecting(2) as collector:
            out = await accounting.accounting_chart(
                accounting.ChartArgs(kind="ranking", resource="gpu", dimension="namespace")
            )
        assert out["plotted"] == [
            {"namespace": "braingeneers", "usage": 184320.0},
            {"namespace": "coder", "usage": 71204.5},
        ]
        assert len(collector.charts) == 1

    async def test_breakdown_folds_the_tail_into_other(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = []
        for day in ("2026-07-01", "2026-07-02"):
            for index in range(9):
                rows.append(
                    {
                        "date": day,
                        "namespace": f"ns-{index}",
                        "unit": "gpu_hours",
                        "usage": float(100 - index * 5),
                    }
                )
        seen = install(
            monkeypatch,
            results(
                {
                    "get_latest_data_date": {"latest_data_date": "2026-07-02"},
                    "query_resource_usage": {
                        "rows": rows,
                        "start_date": "2026-06-03",
                        "end_date": "2026-07-02",
                    },
                }
            ),
        )
        with charts.collecting(2) as collector:
            out = await accounting.accounting_chart(
                accounting.ChartArgs(
                    kind="breakdown", resource="gpu", dimension="namespace", top_n=4
                )
            )
        # A breakdown with no dates must cover 30 days, not the single latest
        # day query_resource_usage would otherwise default to.
        window = sent(seen[1])["params"]["arguments"]
        assert window["start_date"] == "2026-06-03"
        assert window["end_date"] == "2026-07-02"
        labels = [row.get("namespace") for row in out["plotted"] if "namespace" in row]
        assert labels == ["ns-0", "ns-1", "ns-2", "ns-3", charts.OTHER_LABEL]
        assert {"folded_into_other": 5} in out["plotted"]
        assert len(collector.charts) == 1

    async def test_a_cluster_wide_trend_resolves_a_thirty_day_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = install(
            monkeypatch,
            results(
                {
                    "get_latest_data_date": {"latest_data_date": "2026-07-30"},
                    "query_resource_usage": {
                        "rows": [{"date": "2026-07-30", "unit": "gpu_hours", "usage": 9.0}],
                        "start_date": "2026-07-01",
                        "end_date": "2026-07-30",
                    },
                }
            ),
        )
        with charts.collecting(2):
            await accounting.accounting_chart(accounting.ChartArgs(kind="trend", resource="gpu"))
        window = sent(seen[1])["params"]["arguments"]
        assert window["start_date"] == "2026-07-01"
        assert window["end_date"] == "2026-07-30"

    async def test_explicit_dates_are_not_widened(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = install(
            monkeypatch,
            results(
                {
                    "query_resource_usage": {
                        "rows": [{"date": "2026-07-05", "unit": "gpu_hours", "usage": 9.0}],
                        "start_date": "2026-07-05",
                        "end_date": "2026-07-05",
                    }
                }
            ),
        )
        with charts.collecting(2):
            await accounting.accounting_chart(
                accounting.ChartArgs(
                    kind="trend", resource="gpu", start_date="2026-07-05", end_date="2026-07-05"
                )
            )
        # No latest-date lookup: the caller said what window they wanted.
        assert len(seen) == 1

    async def test_no_rows_draws_nothing_and_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(
            monkeypatch, result({"rows": [], "start_date": "2026-07-01", "end_date": "2026-07-01"})
        )
        with charts.collecting(2) as collector:
            out = await accounting.accounting_chart(
                accounting.ChartArgs(kind="ranking", resource="gpu", dimension="namespace")
            )
        assert out["attached"] is False
        assert out["rows"] == 0
        assert collector.charts == []

    async def test_budget_refuses_rather_than_promising_a_missing_picture(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, result(TOP_ROWS))
        with charts.collecting(1):
            first = await dispatch(
                "accounting_chart",
                {"kind": "ranking", "resource": "gpu", "dimension": "namespace"},
            )
            second = await dispatch(
                "accounting_chart",
                {"kind": "ranking", "resource": "gpu", "dimension": "namespace"},
            )
        assert first.ok
        assert not second.ok
        assert second.content["error"] == "chart_budget"

    async def test_disabled_charts_refuse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install(monkeypatch, result(TOP_ROWS), charts_enabled=False)
        with charts.collecting(2):
            out = await dispatch(
                "accounting_chart",
                {"kind": "ranking", "resource": "gpu", "dimension": "namespace"},
            )
        assert not out.ok
        assert out.content["error"] == "charts_disabled"

    async def test_breakdown_without_a_dimension_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = install(monkeypatch, result(TOP_ROWS))
        with charts.collecting(2):
            out = await dispatch("accounting_chart", {"kind": "breakdown", "resource": "gpu"})
        assert not out.ok
        assert seen == []

    async def test_a_secret_in_a_namespace_name_is_scrubbed_before_it_is_drawn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Redaction cannot reach a rasterised label, so it has to happen on this
        # side of the renderer -- the label the tool reports and the label drawn
        # are the same string.
        leaked = {
            "rows": [
                {
                    "namespace": "ns-AKIAIOSFODNN7EXAMPLE",
                    "unit": "gpu_hours",
                    "usage": 5.0,
                }
            ],
            "start_date": "2026-07-01",
            "end_date": "2026-07-01",
        }
        install(monkeypatch, result(leaked))
        with charts.collecting(2):
            out = await accounting.accounting_chart(
                accounting.ChartArgs(kind="ranking", resource="gpu", dimension="namespace")
            )
        drawn = out["plotted"][0]["namespace"]
        assert "AKIAIOSFODNN7EXAMPLE" not in drawn
        assert "REDACTED" in drawn


class TestLlmCharts:
    """A question like "plot the most used LLM models by day" was
    unsatisfiable: `dimension` accepted only the four rank dimensions, so `dimension='model'` failed
    validation and no chart could ever be drawn for it.

    Rows come from a different table with different column names -- the real
    payload shape is copied here: ``tokens_used`` rather than ``usage``, and no
    ``unit`` column at all.
    """

    async def test_a_breakdown_by_model_reads_the_llm_token_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = install(
            monkeypatch,
            results(
                {
                    "get_latest_data_date": {"latest_data_date": "2026-08-12"},
                    "query_llm_token_usage": LLM_DAILY_ROWS,
                }
            ),
        )
        with charts.collecting(2) as collector:
            out = await accounting.accounting_chart(
                accounting.ChartArgs(kind="breakdown", resource="llm", dimension="model")
            )

        # The model dimension lives only in query_llm_token_usage; asking
        # query_resource_usage for it returns rows with no such column.
        args = sent(seen[1])["params"]["arguments"]
        assert sent(seen[1])["params"]["name"] == "query_llm_token_usage"
        assert args["group_by"] == ["date", "model"]
        assert out["attached"] is True
        assert out["unit"] == "tokens"
        assert len(collector.charts) == 1
        # Read from tokens_used, not usage: the wrong column silently plots zeros.
        assert {"model": "deepseek-v4-flash", "total": 2600.0} in out["plotted"]

    async def test_a_ranking_of_models_reads_the_llm_token_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = install(
            monkeypatch,
            result(
                {
                    "metric": "tokens_used",
                    "start_date": "2026-08-12",
                    "end_date": "2026-08-12",
                    "rows": [
                        {"model": "gemma", "tokens_used": 57.0},
                        {"model": "deepseek-v4-flash", "tokens_used": 1600.0},
                    ],
                }
            ),
        )
        with charts.collecting(2) as collector:
            out = await accounting.accounting_chart(
                accounting.ChartArgs(
                    kind="ranking",
                    resource="llm",
                    dimension="model",
                    start_date="2026-08-12",
                    end_date="2026-08-12",
                )
            )
        assert sent(seen[0])["params"]["name"] == "query_llm_token_usage"
        # Rows arrive in whatever order the table returns them; a ranking must
        # be ranked.
        assert out["plotted"] == [
            {"model": "deepseek-v4-flash", "usage": 1600.0},
            {"model": "gemma", "usage": 57.0},
        ]
        assert len(collector.charts) == 1

    async def test_a_model_dimension_needs_the_llm_resource(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, result({}))
        with charts.collecting(2):
            out = await dispatch(
                "accounting_chart",
                {"kind": "breakdown", "resource": "gpu", "dimension": "model"},
            )
        assert out.ok is False
        assert out.content["error"] == "invalid_arguments"
        assert "resource='llm'" in out.content["message"]

    async def test_a_gpu_filter_on_an_llm_breakdown_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The token table has no node or GPU columns. Ignoring the filter would
        draw a cluster-wide chart under a title claiming it was filtered."""
        install(monkeypatch, result({}))
        with charts.collecting(2):
            out = await dispatch(
                "accounting_chart",
                {
                    "kind": "breakdown",
                    "resource": "llm",
                    "dimension": "model",
                    "gpu_model_regex": "A100",
                },
            )
        assert out.ok is False
        assert out.content["error"] == "invalid_arguments"
        assert "gpu_model_regex" in out.content["message"]

    async def test_a_model_trend_points_at_the_breakdown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, result({}))
        with charts.collecting(2):
            out = await dispatch(
                "accounting_chart",
                {"kind": "trend", "resource": "llm", "dimension": "model", "value": "qwen3"},
            )
        assert out.ok is False
        assert "breakdown" in out.content["message"]

    async def test_llm_tokens_by_namespace_still_use_the_usage_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Namespace, institution and node all exist for resource='llm' in the
        usage table, so only the token-table dimensions are rerouted."""
        seen = install(
            monkeypatch,
            results(
                {
                    "get_latest_data_date": {"latest_data_date": "2026-08-12"},
                    "query_resource_usage": {
                        "rows": [{"date": "2026-08-12", "namespace": "coder", "usage": 12.0}],
                        "start_date": "2026-07-14",
                        "end_date": "2026-08-12",
                    },
                }
            ),
        )
        with charts.collecting(2):
            out = await accounting.accounting_chart(
                accounting.ChartArgs(kind="breakdown", resource="llm", dimension="namespace")
            )
        assert sent(seen[1])["params"]["name"] == "query_resource_usage"
        # Neither table reports a unit for llm rows, but they are tokens, and an
        # axis reading "usage" says less than it could.
        assert out["unit"] == "tokens"


class TestBreakdownWindow:
    """A daily breakdown multiplies days by dimension values, so the row cap can
    be reached before the window is covered -- and the rows that arrive are the
    most recent ones. The chart used to be titled with the window that was
    *asked for* while drawing the shorter one it got: 92 days requested, 42
    drawn, nothing saying so.
    """

    @staticmethod
    def _daily(start: str, days: int, models: int = 3) -> dict[str, Any]:
        from datetime import date, timedelta

        first = date.fromisoformat(start)
        return {
            "metric": "tokens_used",
            "start_date": start,
            "end_date": (first + timedelta(days=days - 1)).isoformat(),
            "rows": [
                {
                    "date": (first + timedelta(days=day)).isoformat(),
                    "model": f"m-{index}",
                    "tokens_used": 100.0 + index,
                }
                for day in range(days)
                for index in range(models)
            ],
        }

    async def test_the_row_limit_is_sized_to_the_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen = install(
            monkeypatch,
            result(self._daily("2026-05-17", 92)),
            accounting_max_rows=500,
        )
        with charts.collecting(2):
            await accounting.accounting_chart(
                accounting.ChartArgs(
                    kind="breakdown",
                    resource="llm",
                    dimension="model",
                    start_date="2026-05-17",
                    end_date="2026-08-17",
                )
            )
        # 92 days of a dozen models does not fit in 500 rows, and asking for 500
        # returns the last 42 days of the quarter.
        assert sent(seen[0])["params"]["arguments"]["limit"] > 500

    async def test_a_short_row_set_is_titled_with_the_days_it_actually_has(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The server echoes the requested window but returns only recent rows.
        payload = self._daily("2026-07-07", 42)
        payload["start_date"] = "2026-05-17"
        install(monkeypatch, result(payload))
        with charts.collecting(2) as collector:
            out = await accounting.accounting_chart(
                accounting.ChartArgs(
                    kind="breakdown",
                    resource="llm",
                    dimension="model",
                    start_date="2026-05-17",
                    end_date="2026-08-17",
                )
            )
        assert out["window"] == "2026-07-07 to 2026-08-17"
        assert "2026-05-17" in out["window_requested"]
        assert "row limit" in out["note"]
        # The picture must not caption itself with a window it does not show.
        assert "2026-07-07" in collector.charts[0].alt_text

    async def test_a_complete_row_set_says_nothing_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install(monkeypatch, result(self._daily("2026-08-01", 5)))
        with charts.collecting(2):
            out = await accounting.accounting_chart(
                accounting.ChartArgs(
                    kind="breakdown",
                    resource="llm",
                    dimension="model",
                    start_date="2026-08-01",
                    end_date="2026-08-05",
                )
            )
        assert out["window"] == "2026-08-01 to 2026-08-05"
        assert "window_requested" not in out
        assert "row limit" not in out["note"]


class TestRegistration:
    def test_every_accounting_tool_is_registered_and_flagged_untrusted(self) -> None:
        from nrp_ops_agent.tools import registry

        names = {name for name in registry() if name.startswith("accounting_")}
        assert names == {
            "accounting_chart",
            "accounting_discover",
            "accounting_llm_usage",
            "accounting_top",
            "accounting_trend",
            "accounting_usage",
        }
        # Namespace names are user-chosen, so the wrapper applies to all of them.
        assert all(registry()[name].untrusted_output for name in names)
