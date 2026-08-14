"""PromQL tool.

Security role: minor -- Prometheus is read-only by construction and its output is
metric names and numbers, not free text. Two things still matter:

* the query is sent to a configured endpoint only, never a model-supplied host;
* results are capped by series count and by sample count, because a careless
  ``rate(...[1h])`` across NRP's node count returns tens of thousands of series
  and would evict the rest of the investigation from context.

Prometheus is the highest-value tool for availability questions, so the
playbooks in ``playbooks/`` carry known-good queries rather than expecting the
model to invent PromQL against metrics it cannot enumerate.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from nrp_ops_agent.config import get_settings
from nrp_ops_agent.tools import ToolError, make_tool

#: Prometheus duration literals, e.g. 5m, 1h, 7d.
_DURATION: Final = r"^[0-9]{1,6}(ms|s|m|h|d|w|y)$"

Duration = Annotated[str, StringConstraints(pattern=_DURATION, max_length=16)]

#: Samples returned per series in a range query before truncation.
MAX_SAMPLES_PER_SERIES: Final = 200

_client: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        settings = get_settings()
        headers = {"Accept": "application/json"}
        if settings.prometheus_bearer_token is not None:
            token = settings.prometheus_bearer_token.get_secret_value()
            headers["Authorization"] = f"Bearer {token}"
        _client = httpx.AsyncClient(
            base_url=settings.prometheus_base_url,
            headers=headers,
            timeout=settings.prometheus_timeout_s,
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


class PromqlArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(
        min_length=1,
        max_length=2048,
        description="A PromQL expression. Prefer the queries given in the playbook.",
    )
    lookback: Duration = Field(
        default="1h",
        description="How far back a range query covers. Ignored for instant queries.",
    )
    step: Duration | None = Field(
        default=None,
        description="Set this to run a RANGE query at this resolution (e.g. 1m) and see "
        "how a value changed over `lookback`. Leave unset for a single instant value.",
    )


def _duration_seconds(text: str) -> float:
    units = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}
    for suffix, factor in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * factor
    raise ToolError("invalid_duration", f"Not a Prometheus duration: {text!r}")


@make_tool(
    "promql",
    "Run a PromQL query against the NRP metrics endpoint. Omit `step` for an "
    "instant value ('what is it now'); set `step` for a range query over "
    "`lookback` ('when did it change'). Use this for error rates, saturation and "
    "capacity questions -- it answers 'is it actually down' far better than logs.",
    PromqlArgs,
)
async def promql(args: PromqlArgs) -> dict[str, Any]:
    settings = get_settings()
    client = _http()

    params: dict[str, str]
    if args.step is None:
        path = "/api/v1/query"
        params = {"query": args.query}
    else:
        now = time.time()
        span = _duration_seconds(args.lookback)
        step = _duration_seconds(args.step)
        if step <= 0 or span / step > 11_000:
            raise ToolError(
                "range_too_large",
                f"lookback={args.lookback} at step={args.step} would return "
                f"{int(span / step)} points per series. Increase step or shorten lookback.",
            )
        path = "/api/v1/query_range"
        params = {
            "query": args.query,
            "start": f"{now - span:.3f}",
            "end": f"{now:.3f}",
            "step": args.step,
        }

    try:
        resp = await client.get(path, params=params)
    except httpx.TimeoutException as exc:
        raise ToolError(
            "timeout", f"Prometheus did not respond within {settings.prometheus_timeout_s:g}s."
        ) from exc
    except httpx.HTTPError as exc:
        raise ToolError("prometheus_unreachable", f"{type(exc).__name__}: {exc}") from exc

    if resp.status_code >= 400:
        detail = resp.text[:500]
        raise ToolError("prometheus_error", f"HTTP {resp.status_code}: {detail}")

    try:
        payload = resp.json()
    except ValueError as exc:
        raise ToolError("prometheus_error", "Prometheus returned a non-JSON body.") from exc

    if payload.get("status") != "success":
        raise ToolError(
            "promql_error",
            f"{payload.get('errorType', 'error')}: {payload.get('error', 'unknown')}",
        )

    data = payload.get("data") or {}
    result = data.get("result") or []
    result_type = data.get("resultType")

    total_series = len(result)
    truncated_series = total_series > settings.prometheus_max_series
    result = result[: settings.prometheus_max_series]

    series: list[dict[str, Any]] = []
    truncated_samples = False
    for item in result:
        entry: dict[str, Any] = {"metric": item.get("metric") or {}}
        if "value" in item:
            entry["value"] = item["value"]
        if "values" in item:
            values = item["values"]
            if len(values) > MAX_SAMPLES_PER_SERIES:
                # Keep the ends: the shape of the change is what matters.
                head = MAX_SAMPLES_PER_SERIES // 2
                values = values[:head] + values[-head:]
                truncated_samples = True
            entry["values"] = values
        series.append(entry)

    return {
        "query": args.query,
        "query_type": "instant" if args.step is None else "range",
        "lookback": args.lookback if args.step is not None else None,
        "step": args.step,
        "result_type": result_type,
        "series_count": total_series,
        "truncated_series": truncated_series,
        "truncated_samples": truncated_samples,
        "warnings": payload.get("warnings") or [],
        "series": series,
    }
