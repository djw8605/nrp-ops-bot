"""Ask the agent one question from a terminal.

The Slack listener is the production entry point, but it needs a workspace, an
app token and a Socket Mode connection before it will answer anything. That is a
lot of moving parts to stand up when the thing you actually want to see is
whether the model drives the tool loop sensibly and whether the playbook it
picked was the right one.

This runs the *same* ``Agent.investigate`` the Slack app calls -- same tools,
same namespace allowlist, same redaction, same budgets. Only the transport
differs, so a question that works here works in Slack.

It is a development tool. It performs no Slack authorization because there is no
Slack user to authorize: whoever can run it already has whatever cluster access
their kubeconfig grants. The namespace allowlist still applies, because that
check lives inside each tool rather than at the edge.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from nrp_ops_agent import audit
from nrp_ops_agent.agent import Agent, AgentResult
from nrp_ops_agent.config import get_allowlist_loader, get_settings
from nrp_ops_agent.playbooks import get_playbooks
from nrp_ops_agent.tools.accounting import aclose as aclose_accounting
from nrp_ops_agent.tools.k8s import aclose_clients


def _preflight() -> list[str]:
    """Report the misconfigurations that otherwise fail as a confusing answer.

    Each of these produces a *plausible* wrong result rather than a crash: no
    model name looks like a transport error, an empty allowlist looks like an
    empty cluster, and a missing docs index looks like undocumented software.
    """
    settings = get_settings()
    problems: list[str] = []

    if not settings.llm_model:
        problems.append(
            "NRP_OPS_LLM_MODEL is unset. List the choices with:\n"
            "    curl -s https://ellm.nrp-nautilus.io/v1/models | python3 -m json.tool"
        )
    if not settings.llm_api_key.get_secret_value():
        problems.append("NRP_OPS_LLM_API_KEY is unset; the gateway answers 403 without it.")

    namespaces = get_allowlist_loader().load().namespaces
    if not namespaces:
        problems.append(
            f"No namespaces are allowed, so every Kubernetes tool will refuse before it\n"
            f"    calls the API -- which reads like a healthy, empty cluster. The policy file\n"
            f"    at {settings.allowlist_path} is missing or empty. Point at the dev copy:\n"
            f"    export NRP_OPS_ALLOWLIST_PATH=dev/allowlist.yaml"
        )
    if not settings.docs_db_path.exists():
        problems.append(
            f"No docs index at {settings.docs_db_path}, so search_docs will fail and the\n"
            f"    agent will answer without citations. Build one:\n"
            f"    python -m nrp_ops_agent.docs_index.sync --db-path ./docs.sqlite3"
        )
    return problems


def _render(result: AgentResult, *, show_tools: bool) -> str:
    lines: list[str] = []
    if show_tools and result.tool_calls:
        lines.append("--- tool calls " + "-" * 50)
        for index, call in enumerate(result.tool_calls, start=1):
            status = "ok" if call.ok else f"ERROR {call.error}"
            args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
            lines.append(
                f"{index:>2}. {call.name}({args}) -> {status} "
                f"[{call.latency_ms:.0f}ms, {call.size_bytes}B]"
            )
        lines.append("")

    lines.append(result.text)
    lines.append("")
    lines.append(
        f"--- {result.stop_reason} | {result.iterations} iteration(s) | "
        f"{len(result.tool_calls)} tool call(s)"
        + (f" | {result.dropped_results} result(s) dropped" if result.dropped_results else "")
        + (
            f" | redacted: {', '.join(sorted(set(result.redaction_rules)))}"
            if result.redaction_rules
            else ""
        )
    )
    return "\n".join(lines)


async def _investigate(question: str, *, quiet: bool, show_tools: bool) -> AgentResult:
    async def progress(note: str) -> None:
        if not quiet:
            print(f"  ... {note}", file=sys.stderr, flush=True)

    agent = Agent()
    try:
        return await agent.investigate(question, progress=None if quiet else progress)
    finally:
        await agent.aclose()
        await aclose_clients()
        await aclose_accounting()


def _save_charts(result: AgentResult, directory: Path) -> list[Path]:
    """Write any charts the turn drew, so they can be looked at.

    In Slack these are uploaded to the thread. Here there is nowhere to put an
    image, and a chart nobody opens is a feature that only appears to work --
    so the terminal path writes the file and prints where it went.
    """
    if not result.charts:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, chart in enumerate(result.charts, start=1):
        path = directory / f"{index:02d}-{chart.filename}"
        path.write_bytes(chart.png)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nrp-ops-ask",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question", nargs="*", help="The operator question, unquoted is fine.")
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="Print every tool call with its arguments, latency and outcome.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress the live progress lines on stderr."
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=Path("./charts"),
        help="Where to write any charts the answer drew. Default: ./charts",
    )
    parser.add_argument(
        "--which-playbook",
        action="store_true",
        help="Print the playbook this question selects and exit without calling the model.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run the preflight checks and exit. Does not call the model.",
    )
    args = parser.parse_args(argv)

    # Audit records go to stdout by default, which would interleave with the
    # answer. The transcript is the point of this tool, so send them to stderr.
    audit.set_stream(sys.stderr)

    problems = _preflight()
    if args.check:
        if problems:
            print("preflight found problems:\n", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}\n", file=sys.stderr)
            return 1
        print("preflight ok: model, allowlist and docs index are all configured.")
        return 0

    question = " ".join(args.question).strip()
    if not question:
        parser.error("no question given")

    if args.which_playbook:
        playbook = get_playbooks().select(question)
        if playbook is None:
            print("no playbook matched, and no _default is loaded")
            return 1
        print(playbook.render())
        return 0

    # Warn but continue: a partially configured run still shows the loop
    # working, and the banner explains any answer that comes back thin.
    for problem in problems:
        print(f"warning: {problem}\n", file=sys.stderr)

    result = asyncio.run(_investigate(question, quiet=args.quiet, show_tools=args.show_tools))
    print(_render(result, show_tools=args.show_tools))
    for path in _save_charts(result, args.charts_dir):
        print(f"--- chart written to {path}")
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
