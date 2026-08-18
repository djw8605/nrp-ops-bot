"""NRP accounting usage, over the read-only MCP server in front of ClickHouse.

The ETL in `nrp-clickhouse` rolls Prometheus and the NRP portal into daily
per-namespace and per-pod usage rows, and publishes a read-only MCP server over
them at `https://nrp-accounting-mcp.nrp-nautilus.io/`. That answers a class of
question the cluster itself cannot: Prometheus retention is short and its
`kube_pod_container_resource_*` series say what was *requested*, not what a
namespace was charged for over a quarter.

Security role: this is an outbound call to one configured endpoint, and it is
deny-by-default in the same two ways every other tool here is.

* **No passthrough.** The MCP server advertises fourteen tools; this module
  exposes five of them, each behind a pydantic model with its own literals and
  bounds. A model cannot reach an MCP tool this file does not name, and cannot
  pass an argument these models do not declare -- so the fact that the endpoint
  is unauthenticated and reachable by anything in the cluster costs nothing
  here.
* **Untrusted output.** Namespace, node and PI names are chosen by the people
  who run the workloads, so every result is flagged and wrapped. They are not
  as hostile a source as a pod log -- nothing here returns free text -- but
  "sum GPU hours by namespace" will happily return a namespace called
  ``ignore-previous-instructions``.

Deliberately *not* gated on the namespace allowlist, unlike the Kubernetes
tools. That allowlist bounds what can be read *out of the cluster* -- logs, pod
specs, events -- where the blast radius of a bad namespace is a credential in
someone's log line. Accounting rows are aggregate usage numbers that already
leave NRP for XDMoD, and the questions worth asking here ("who are the top ten
GPU consumers?") are cluster-wide by construction: an allowlist would answer
them wrongly rather than refusing them, which is the worse failure.

Transport: the server runs FastMCP in stateless mode with JSON responses, so a
call is one plain JSON-RPC POST with no ``initialize`` handshake and no session
to keep. That is why this speaks the protocol directly instead of pulling in an
MCP client -- the client's value is session and transport negotiation, and there
is neither.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Annotated, Any, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from nrp_ops_agent import charts
from nrp_ops_agent.config import get_settings
from nrp_ops_agent.tools import ToolError, make_tool

#: ``YYYY-MM-DD``. The accounting data is daily; anything finer is a mistake
#: about what this data set is, and is worth rejecting rather than truncating.
_ISO_DATE: Final = r"^\d{4}-\d{2}-\d{2}$"

IsoDate = Annotated[str, StringConstraints(pattern=_ISO_DATE)]

Name = Annotated[str, StringConstraints(min_length=1, max_length=253, strip_whitespace=True)]

Regex = Annotated[str, StringConstraints(min_length=1, max_length=200)]

#: Canonical resource names, mirroring the server's own list. It also accepts
#: aliases (``gpu_hours``, ``GPU hours``) and normalises them, but a closed
#: literal is a better instruction to the model than a free string plus a
#: paragraph of aliases -- and a wrong one is refused here rather than after a
#: round trip. ``wall`` is pod wall time, which is not billable: it is the
#: denominator that turns core-hours back into device counts.
Resource = Literal["gpu", "cpu", "memory", "storage", "network", "fpga", "llm", "wall", "other"]

#: What ``accounting_top`` can rank. Narrower than the group-by list because the
#: server's own ranking query supports only these four.
RankDimension = Literal["namespace", "institution", "node", "commercial"]

#: What ``accounting_trend`` can pin to a single value.
TrendDimension = Literal["namespace", "institution", "node", "node_institution", "commercial"]

#: What ``accounting_usage`` can group by, and ``accounting_discover`` can list.
GroupDimension = Literal[
    "date",
    "namespace",
    "institution",
    "node",
    "node_institution",
    "pi",
    "commercial",
    "created_by",
    "resource",
    "raw_resource",
    "gpu_model_name",
    "unit",
]

ListDimension = Literal[
    "namespace",
    "institution",
    "node",
    "node_institution",
    "pi",
    "commercial",
    "resource",
    "raw_resource",
    "gpu_model_name",
]

LlmDimension = Literal["date", "namespace", "token_alias", "model", "token_type"]

#: What a group-by defaults to when the model gives none. Namespace is the
#: dimension every accounting question is eventually about.
_DEFAULT_GROUP_BY: Final[list[GroupDimension]] = ["namespace"]
_DEFAULT_LLM_GROUP_BY: Final[list[LlmDimension]] = ["namespace"]

#: Human wording for the axis, keyed by the unit the server reports. Falls back
#: to the raw unit, which is already readable.
_UNIT_LABELS: Final[dict[str, str]] = {
    "gpu_hours": "GPU hours",
    "cpu_core_hours": "CPU core hours",
    "core_hours": "CPU core hours",
    "byte_hours": "byte hours",
    "gb_hours": "GB hours",
    "tokens": "tokens",
    "wall_hours": "wall hours",
}

_client: httpx.AsyncClient | None = None


#: Sent on every call rather than set once on the client. Streamable HTTP
#: requires *both* media types even when the server is configured to answer in
#: JSON -- the transport validates Accept before it looks at its own response
#: mode, and omitting the SSE type is a 406 rather than a fallback. Keeping them
#: on the request means the contract holds whatever client is in use.
_HEADERS: Final[dict[str, str]] = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=get_settings().accounting_timeout_s)
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _endpoint() -> str:
    url = get_settings().accounting_mcp_url
    if not url:
        raise ToolError(
            "accounting_not_configured",
            "NRP_OPS_ACCOUNTING_MCP_URL is unset, so there is no accounting server to "
            "query. Answer from the other tools and say accounting data was unavailable.",
        )
    return url


def _decode(response: httpx.Response) -> dict[str, Any]:
    """Read one JSON-RPC response body, JSON or SSE-framed.

    The deployment sets ``json_response=True``, but the same server speaks SSE
    when that flag is off. Handling both means a config change on the other side
    degrades to a slower path rather than to an unparseable body.
    """
    content_type = response.headers.get("content-type", "")
    text = response.text
    if content_type.startswith("text/event-stream"):
        for line in text.splitlines():
            if line.startswith("data:"):
                text = line[5:].strip()
                break
    try:
        body = json.loads(text)
    except ValueError as exc:
        raise ToolError(
            "accounting_error", "The accounting server returned a non-JSON body."
        ) from exc
    if not isinstance(body, dict):
        raise ToolError("accounting_error", "The accounting server returned a non-object body.")
    return body


async def call_mcp(tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Call one tool on the accounting MCP server and return its structured result.

    ``None`` arguments are dropped rather than sent: every filter on that server
    is optional, and an explicit null in the audit trail reads as a filter that
    was applied.
    """
    settings = get_settings()
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool,
            "arguments": {k: v for k, v in arguments.items() if v is not None},
        },
    }

    try:
        response = await _http().post(_endpoint(), json=payload, headers=_HEADERS)
    except httpx.TimeoutException as exc:
        raise ToolError(
            "timeout",
            f"The accounting server did not respond within {settings.accounting_timeout_s:g}s. "
            "A wide date range over pod granularity is the usual cause -- narrow it.",
        ) from exc
    except httpx.HTTPError as exc:
        raise ToolError("accounting_unreachable", f"{type(exc).__name__}: {exc}") from exc

    if response.status_code >= 400:
        raise ToolError("accounting_error", f"HTTP {response.status_code}: {response.text[:500]}")

    body = _decode(response)

    if "error" in body:
        error = body["error"] or {}
        raise ToolError(
            "accounting_error",
            f"{error.get('code', 'error')}: {str(error.get('message', 'unknown'))[:500]}",
        )

    result = body.get("result")
    if not isinstance(result, dict):
        raise ToolError("accounting_error", "The accounting server returned no result.")

    content = result.get("content") or []
    text = ""
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = str(block.get("text") or "")
            break

    if result.get("isError"):
        # The server's own validation errors arrive this way -- "unknown
        # resource", "unknown dimension". They are correctable, so they must
        # reach the model intact rather than as "the tool failed".
        raise ToolError("accounting_rejected", text[:500] or "The accounting server refused.")

    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    if text:
        try:
            parsed = json.loads(text)
        except ValueError:
            return {"text": text}
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    return {}


def _cap_rows(payload: dict[str, Any]) -> dict[str, Any]:
    """Bound one response to the configured row cap.

    The server's own ``limit`` already bounds most queries, but ``group_by`` of
    two high-cardinality dimensions multiplies out, and a thousand rows of
    namespace-by-date would evict the rest of the investigation from context.
    """
    maximum = get_settings().accounting_max_rows
    for key in ("rows", "values"):
        rows = payload.get(key)
        if isinstance(rows, list) and len(rows) > maximum:
            payload[key] = rows[:maximum]
            payload[f"{key}_truncated"] = True
            payload[f"{key}_returned"] = maximum
    return payload


def _unit_label(unit: str | None) -> str:
    if not unit:
        return "usage"
    return _UNIT_LABELS.get(str(unit), str(unit).replace("_", " "))


def _rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


class DiscoverArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: ListDimension | None = Field(
        default=None,
        description="Omit to get only the latest ingested date. Set it to also list the "
        "distinct values seen in usage rows for that dimension, which is how you find "
        "the exact spelling of a namespace, institution or GPU model.",
    )
    resource: Resource | None = Field(
        default=None,
        description="Restrict the listed values to rows for this resource. Pair with "
        "dimension='gpu_model_name' to enumerate GPU models.",
    )
    start_date: IsoDate | None = None
    end_date: IsoDate | None = Field(
        default=None,
        description="Omit both dates to look at the most recent ingested day only.",
    )
    prefix: Name | None = Field(default=None, description="Only values starting with this.")
    limit: int = Field(default=50, ge=1, le=500)


@make_tool(
    "accounting_discover",
    "What accounting data exists. Always returns the most recent ingested date -- "
    "accounting lags real time by a day or more, so check this before saying "
    "'today'. With `dimension` set it also lists the values actually observed in "
    "usage rows (namespaces, institutions, nodes, GPU models), which is how you "
    "get an exact spelling before querying it.",
    DiscoverArgs,
    untrusted_output=True,
)
async def accounting_discover(args: DiscoverArgs) -> dict[str, Any]:
    latest = await call_mcp("get_latest_data_date", {})
    out: dict[str, Any] = {"latest_data_date": latest}

    if args.dimension is not None:
        values = await call_mcp(
            "list_filter_values",
            {
                "dimension": args.dimension,
                "resource": args.resource,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "prefix": args.prefix,
                "limit": args.limit,
            },
        )
        out["values"] = _cap_rows(values)
    return out


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


class TopArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: RankDimension = Field(
        description="What to rank. 'commercial' splits the cluster into commercial and "
        "non-commercial namespaces.",
    )
    resource: Resource
    start_date: IsoDate | None = None
    end_date: IsoDate | None = Field(
        default=None,
        description="Omit both dates for the most recent ingested day. Supply only one "
        "for a single-day query.",
    )
    institution: Name | None = None
    node_regex: Regex | None = Field(
        default=None, description="ClickHouse regex matched against the node name."
    )
    gpu_model_regex: Regex | None = Field(
        default=None,
        description="ClickHouse regex against the GPU model, e.g. '(?i)A100'. Pair with "
        "resource='gpu'.",
    )
    limit: int = Field(default=10, ge=1, le=100)


@make_tool(
    "accounting_top",
    "Rank the biggest consumers of one resource over a date window: top namespaces, "
    "institutions or nodes by GPU hours, CPU core hours, storage or LLM tokens. This "
    "is the tool for 'who is using all the A100s' -- pair resource='gpu' with "
    "gpu_model_regex='(?i)A100'.",
    TopArgs,
    untrusted_output=True,
)
async def accounting_top(args: TopArgs) -> dict[str, Any]:
    payload = await call_mcp(
        "top_resource_consumers",
        {
            "dimension": args.dimension,
            "resource": args.resource,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "institution": args.institution,
            "node_regex": args.node_regex,
            "gpu_model_regex": args.gpu_model_regex,
            "limit": args.limit,
        },
    )
    return _cap_rows(payload)


# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #


class TrendArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: TrendDimension
    value: Name = Field(description="The exact namespace, institution or node name.")
    resource: Resource
    start_date: IsoDate | None = None
    end_date: IsoDate | None = Field(
        default=None, description="Omit both for the last 30 days of ingested data."
    )
    gpu_model_regex: Regex | None = None


@make_tool(
    "accounting_trend",
    "Daily usage of one resource for a single namespace, institution or node, "
    "defaulting to the last 30 days. Use this for 'when did their GPU usage jump' -- "
    "it returns one row per day. Call accounting_chart with the same arguments if a "
    "picture would carry the answer better than a list of numbers.",
    TrendArgs,
    untrusted_output=True,
)
async def accounting_trend(args: TrendArgs) -> dict[str, Any]:
    payload = await call_mcp(
        "get_usage_timeseries",
        {
            "dimension": args.dimension,
            "value": args.value,
            "resource": args.resource,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "gpu_model_regex": args.gpu_model_regex,
        },
    )
    return _cap_rows(payload)


# --------------------------------------------------------------------------- #
# The escape hatch
# --------------------------------------------------------------------------- #


class UsageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    group_by: list[GroupDimension] = Field(
        default_factory=lambda: list(_DEFAULT_GROUP_BY),
        max_length=4,
        description="Aggregate by these columns, e.g. ['date','namespace'] for a "
        "per-namespace daily breakdown or ['gpu_model_name'] for a fleet split.",
    )
    resource: Resource | None = None
    start_date: IsoDate | None = None
    end_date: IsoDate | None = Field(
        default=None, description="Omit both for the most recent ingested day."
    )
    namespace: Name | None = None
    institution: Name | None = None
    node: Name | None = None
    node_regex: Regex | None = None
    node_institution: Name | None = None
    raw_resource: Name | None = Field(
        default=None,
        description="The source label rather than the canonical resource, e.g. "
        "'nvidia_com_a100'. Discover these with accounting_discover.",
    )
    gpu_model_regex: Regex | None = None
    limit: int = Field(default=100, ge=1, le=500)


@make_tool(
    "accounting_usage",
    "Custom accounting aggregation: sum usage over a date window with any "
    "combination of filters and group-by columns. Use this only when "
    "accounting_top, accounting_trend and accounting_discover do not fit -- they "
    "are cheaper and their results are easier to read.",
    UsageArgs,
    untrusted_output=True,
)
async def accounting_usage(args: UsageArgs) -> dict[str, Any]:
    payload = await call_mcp(
        "query_resource_usage",
        {
            "group_by": list(args.group_by),
            "resource": args.resource,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "namespace": args.namespace,
            "institution": args.institution,
            "node": args.node,
            "node_regex": args.node_regex,
            "node_institution": args.node_institution,
            "raw_resource": args.raw_resource,
            "gpu_model_regex": args.gpu_model_regex,
            "limit": args.limit,
        },
    )
    return _cap_rows(payload)


# --------------------------------------------------------------------------- #
# LLM tokens
# --------------------------------------------------------------------------- #


class LlmArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    group_by: list[LlmDimension] = Field(
        default_factory=lambda: list(_DEFAULT_LLM_GROUP_BY),
        max_length=4,
        description="Aggregate token counts by these columns.",
    )
    start_date: IsoDate | None = None
    end_date: IsoDate | None = Field(
        default=None, description="Omit both for the most recent ingested day."
    )
    namespace: Name | None = None
    model: Name | None = Field(default=None, description="An exact served model name.")
    token_type: Name | None = Field(default=None, description="Usually 'prompt' or 'completion'.")
    limit: int = Field(default=100, ge=1, le=500)


@make_tool(
    "accounting_llm_usage",
    "LLM token accounting for the NRP inference gateway: tokens by namespace, served "
    "model, token alias and token type. Separate from accounting_usage because the "
    "tokens are counted per model rather than per node.",
    LlmArgs,
    untrusted_output=True,
)
async def accounting_llm_usage(args: LlmArgs) -> dict[str, Any]:
    payload = await call_mcp(
        "query_llm_token_usage",
        {
            "group_by": list(args.group_by),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "namespace": args.namespace,
            "model": args.model,
            "token_type": args.token_type,
            "limit": args.limit,
        },
    )
    return _cap_rows(payload)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


class ChartArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["trend", "ranking", "breakdown"] = Field(
        description="'trend' is one line over time; 'ranking' is horizontal bars for "
        "one date window; 'breakdown' is a stacked area showing how a total splits by "
        "dimension over time.",
    )
    resource: Resource
    dimension: RankDimension | None = Field(
        default=None,
        description="Required for 'ranking' and 'breakdown': what the bars or bands "
        "are. For 'trend' it names what `value` refers to.",
    )
    value: Name | None = Field(
        default=None,
        description="For 'trend': the one namespace, institution or node to plot. Omit "
        "it to plot the cluster-wide total.",
    )
    start_date: IsoDate | None = None
    end_date: IsoDate | None = Field(
        default=None,
        description="Omit both: 'trend' and 'breakdown' cover the last 30 days, "
        "'ranking' the most recent ingested day.",
    )
    namespace: Name | None = None
    institution: Name | None = None
    node_regex: Regex | None = None
    gpu_model_regex: Regex | None = None
    top_n: int = Field(
        default=8,
        ge=2,
        le=15,
        description="How many bars, or how many bands before the rest fold into 'Other'.",
    )
    title: Annotated[str, StringConstraints(max_length=120)] | None = Field(
        default=None, description="Overrides the generated title. Keep it factual."
    )


@make_tool(
    "accounting_chart",
    "Draw a chart of accounting usage and attach it to the reply as a PNG. Use it "
    "when the shape is the answer -- a trend over weeks, a ranking of the top "
    "consumers, or how a total splits between namespaces over time. It queries the "
    "data itself, so call it INSTEAD of the matching query tool, not after it: the "
    "numbers it plotted come back in the result for you to quote. Still write the "
    "finding in words; the picture supports the text, it does not replace it.",
    ChartArgs,
    untrusted_output=True,
)
async def accounting_chart(args: ChartArgs) -> dict[str, Any]:
    settings = get_settings()
    if not settings.charts_enabled:
        raise ToolError(
            "charts_disabled",
            "Chart rendering is disabled on this deployment. Use accounting_top, "
            "accounting_trend or accounting_usage and describe the numbers instead.",
        )
    if charts.budget_remaining() <= 0:
        raise ToolError(
            "chart_budget",
            f"This reply already carries {settings.charts_max_per_turn} chart(s), which "
            "is the per-answer limit. Query the numbers instead and describe them.",
        )

    if args.kind == "trend":
        return await _chart_trend(args)
    if args.kind == "ranking":
        return await _chart_ranking(args)
    return await _chart_breakdown(args)


def _window(payload: Mapping[str, Any]) -> str:
    start = str(payload.get("start_date") or "")
    end = str(payload.get("end_date") or "")
    if start and end and start != end:
        return f"{start} to {end}"
    return start or end or "latest ingested day"


def _footer() -> str:
    return "NRP accounting · daily aggregates from ClickHouse"


#: Matches the server's own trend default, so a chart drawn through
#: get_usage_timeseries and one drawn through query_resource_usage cover the
#: same window when neither is given dates.
DEFAULT_TREND_DAYS: Final = 30


async def _trend_window(
    start_date: str | None, end_date: str | None
) -> tuple[str | None, str | None]:
    """Fill in a 30-day window when the caller gave neither end of one.

    ``get_usage_timeseries`` defaults to the last 30 days on the server, but
    ``query_resource_usage`` -- which the cluster-wide and breakdown charts go
    through -- defaults to the *latest ingested day only*. Drawing a 30-day
    trend as a single point is the kind of wrong that looks like a working
    feature, so the window is resolved here rather than left to whichever query
    the chart happens to use.
    """
    if start_date is not None or end_date is not None:
        return start_date, end_date

    latest = await call_mcp("get_latest_data_date", {})
    raw = str(latest.get("latest_data_date") or "")
    try:
        end = date.fromisoformat(raw)
    except ValueError as exc:
        raise ToolError(
            "accounting_error",
            f"The accounting server reported an unusable latest date: {raw[:40]!r}.",
        ) from exc
    return (end - timedelta(days=DEFAULT_TREND_DAYS - 1)).isoformat(), end.isoformat()


def _attach(png: bytes, *, title: str, alt_text: str, slug: str) -> bool:
    return charts.emit(
        charts.Chart(
            filename=f"nrp-accounting-{slug}.png",
            title=title,
            alt_text=alt_text,
            png=png,
        )
    )


def _result(
    *,
    attached: bool,
    kind: str,
    title: str,
    unit: str,
    window: str,
    plotted: list[dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "chart": kind,
        "title": title,
        "unit": unit,
        "window": window,
        "attached": attached,
        "plotted": plotted,
    }
    if attached:
        out["note"] = (
            "The chart is attached to your reply as an image. Do not describe the "
            "picture -- state the finding and quote the numbers above."
        )
    else:
        out["note"] = (
            "The chart could NOT be attached, so the operator will see only your text. "
            "Give the numbers above in full and do not refer to a chart."
        )
    return out


async def _chart_trend(args: ChartArgs) -> dict[str, Any]:
    if args.value is not None:
        if args.dimension is None:
            raise ToolError(
                "invalid_arguments",
                "kind='trend' with a `value` also needs `dimension` to say what that "
                "value is (namespace, institution, node...).",
            )
        payload = await call_mcp(
            "get_usage_timeseries",
            {
                "dimension": args.dimension,
                "value": args.value,
                "resource": args.resource,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "gpu_model_regex": args.gpu_model_regex,
            },
        )
        subject = f"{args.dimension} {args.value}"
    else:
        start_date, end_date = await _trend_window(args.start_date, args.end_date)
        payload = await call_mcp(
            "query_resource_usage",
            {
                "group_by": ["date", "unit"],
                "resource": args.resource,
                "start_date": start_date,
                "end_date": end_date,
                "namespace": args.namespace,
                "institution": args.institution,
                "node_regex": args.node_regex,
                "gpu_model_regex": args.gpu_model_regex,
                "limit": 400,
            },
        )
        subject = "the cluster"

    rows = sorted(_rows(payload), key=lambda row: str(row.get("date") or ""))
    if not rows:
        return {
            "chart": "trend",
            "rows": 0,
            "attached": False,
            "window": _window(payload),
            "note": "No accounting rows matched, so nothing was drawn. Say there is no data.",
        }

    unit = _unit_label(str(rows[0].get("unit") or ""))
    x_labels = [charts.clean_label(row.get("date")) for row in rows]
    values = tuple(_number(row.get("usage")) for row in rows)
    window = _window(payload)

    title = args.title or f"{unit} for {subject}"
    title = charts.clean_label(title)
    png = charts.render_trend(
        title=title,
        subtitle=window,
        footer=_footer(),
        x_labels=x_labels,
        series=[charts.Series(label=unit, values=values)],
        y_label=unit,
    )
    attached = _attach(
        png, title=title, alt_text=f"Daily {unit} for {subject}, {window}.", slug="trend"
    )

    total = sum(values)
    peak_index = max(range(len(values)), key=lambda i: values[i])
    return _result(
        attached=attached,
        kind="trend",
        title=title,
        unit=unit,
        window=window,
        plotted=[
            {
                "days": len(rows),
                "total": round(total, 3),
                "mean_per_day": round(total / len(rows), 3),
            },
            {"peak_date": x_labels[peak_index], "peak": round(values[peak_index], 3)},
            {"first_date": x_labels[0], "first": round(values[0], 3)},
            {"last_date": x_labels[-1], "last": round(values[-1], 3)},
        ],
    )


async def _chart_ranking(args: ChartArgs) -> dict[str, Any]:
    if args.dimension is None:
        raise ToolError(
            "invalid_arguments", "kind='ranking' needs `dimension` -- what is being ranked."
        )
    payload = await call_mcp(
        "top_resource_consumers",
        {
            "dimension": args.dimension,
            "resource": args.resource,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "institution": args.institution,
            "node_regex": args.node_regex,
            "gpu_model_regex": args.gpu_model_regex,
            "limit": min(args.top_n, charts.MAX_BARS),
        },
    )
    rows = _rows(payload)
    if not rows:
        return {
            "chart": "ranking",
            "rows": 0,
            "attached": False,
            "window": _window(payload),
            "note": "No accounting rows matched, so nothing was drawn. Say there is no data.",
        }

    unit = _unit_label(str(rows[0].get("unit") or ""))
    labels = [charts.clean_label(row.get(args.dimension)) for row in rows]
    values = [_number(row.get("usage")) for row in rows]
    window = _window(payload)

    title = args.title or f"Top {args.dimension}s by {unit}"
    title = charts.clean_label(title)
    png = charts.render_ranking(
        title=title,
        subtitle=window,
        footer=_footer(),
        labels=labels,
        values=values,
        # Every bar carries its own value here, so an axis label repeating the
        # unit the title already names is ink with no reader.
        x_label="" if unit.lower() in title.lower() else unit,
    )
    attached = _attach(
        png,
        title=title,
        alt_text=f"Horizontal bar chart ranking {len(rows)} {args.dimension}s by {unit}, {window}.",
        slug=f"top-{args.dimension}",
    )
    return _result(
        attached=attached,
        kind="ranking",
        title=title,
        unit=unit,
        window=window,
        plotted=[
            {args.dimension: label, "usage": round(value, 3)}
            for label, value in zip(labels, values, strict=True)
        ],
    )


async def _chart_breakdown(args: ChartArgs) -> dict[str, Any]:
    if args.dimension is None:
        raise ToolError(
            "invalid_arguments",
            "kind='breakdown' needs `dimension` -- what the stacked bands are.",
        )
    start_date, end_date = await _trend_window(args.start_date, args.end_date)
    payload = await call_mcp(
        "query_resource_usage",
        {
            "group_by": ["date", args.dimension, "unit"],
            "resource": args.resource,
            "start_date": start_date,
            "end_date": end_date,
            "namespace": args.namespace,
            "institution": args.institution,
            "node_regex": args.node_regex,
            "gpu_model_regex": args.gpu_model_regex,
            "limit": get_settings().accounting_max_rows,
        },
    )
    rows = _rows(payload)
    if not rows:
        return {
            "chart": "breakdown",
            "rows": 0,
            "attached": False,
            "window": _window(payload),
            "note": "No accounting rows matched, so nothing was drawn. Say there is no data.",
        }

    unit = _unit_label(str(rows[0].get("unit") or ""))
    dates = sorted({str(row.get("date") or "") for row in rows})
    date_index = {date: i for i, date in enumerate(dates)}

    totals: dict[str, float] = {}
    grid: dict[str, list[float]] = {}
    for row in rows:
        key = charts.clean_label(row.get(args.dimension) or "unknown")
        usage = _number(row.get("usage"))
        totals[key] = totals.get(key, 0.0) + usage
        grid.setdefault(key, [0.0] * len(dates))[date_index[str(row.get("date") or "")]] += usage

    ranked = sorted(totals, key=lambda key: -totals[key])
    keep = ranked[: min(args.top_n, charts.MAX_SERIES)]
    folded = ranked[len(keep) :]

    series = [charts.Series(label=key, values=tuple(grid[key])) for key in keep]
    if folded:
        other = [0.0] * len(dates)
        for key in folded:
            for index, value in enumerate(grid[key]):
                other[index] += value
        series.append(charts.Series(label=charts.OTHER_LABEL, values=tuple(other)))

    window = _window(payload)
    title = args.title or f"{unit} by {args.dimension}"
    title = charts.clean_label(title)
    png = charts.render_trend(
        title=title,
        subtitle=window,
        footer=_footer(),
        x_labels=dates,
        series=series,
        y_label=unit,
        stacked=True,
    )
    attached = _attach(
        png,
        title=title,
        alt_text=(
            f"Stacked area chart of daily {unit} split across {len(series)} "
            f"{args.dimension}s, {window}."
        ),
        slug=f"breakdown-{args.dimension}",
    )
    return _result(
        attached=attached,
        kind="breakdown",
        title=title,
        unit=unit,
        window=window,
        plotted=[{args.dimension: item.label, "total": round(item.total, 3)} for item in series]
        + ([{"folded_into_other": len(folded)}] if folded else []),
    )
