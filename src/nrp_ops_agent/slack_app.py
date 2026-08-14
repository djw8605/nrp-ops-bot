"""Slack listener (Socket Mode).

Security role: the outbound redaction chokepoint lives here --
:func:`_post` and :func:`_update` are the only places this process writes to
Slack, and both scrub first. Nothing else in the package may call
``chat.postMessage`` directly.

Socket Mode means there is no public ingress, no request-signature endpoint and
no inbound network path to this pod at all; egress is restricted by
``deploy/networkpolicy.yaml``.

Denied messages get an emoji reaction and nothing else. See
:mod:`nrp_ops_agent.authz` for why silence is the correct response.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from nrp_ops_agent import audit
from nrp_ops_agent.agent import Agent, AgentResult
from nrp_ops_agent.authz import Authorizer, SlackEvent
from nrp_ops_agent.config import Settings, get_settings
from nrp_ops_agent.redact import redact

log = logging.getLogger("nrp_ops_agent.slack")

#: Slack hard-limits a message to 4000 characters; stay well under it.
MAX_MESSAGE_CHARS = 3500
ACK_TEXT = "on it -- investigating..."


class SlackOps:
    """Event handling, kept separate from Bolt wiring so it can be tested."""

    def __init__(
        self,
        settings: Settings | None = None,
        authorizer: Authorizer | None = None,
        agent: Agent | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._authz = authorizer or Authorizer(self._settings)
        self._agent = agent or Agent(self._settings)

    # ------------------------------------------------------------ outbound --

    async def _post(self, client: Any, *, channel: str, thread_ts: str, text: str) -> str | None:
        """Post to a thread. The outbound redaction chokepoint."""
        clean, rules = redact(text)
        if rules:
            audit.emit("redaction_hit", slack_channel=channel, redaction_rules=rules)
        response = await client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=clean[:MAX_MESSAGE_CHARS]
        )
        ts = response.get("ts")
        return str(ts) if ts else None

    async def _update(self, client: Any, *, channel: str, ts: str, text: str) -> None:
        clean, rules = redact(text)
        if rules:
            audit.emit("redaction_hit", slack_channel=channel, redaction_rules=rules)
        await client.chat_update(channel=channel, ts=ts, text=clean[:MAX_MESSAGE_CHARS])

    async def _react(self, client: Any, *, channel: str, ts: str) -> None:
        try:
            await client.reactions_add(
                channel=channel, timestamp=ts, name=self._settings.slack_deny_reaction
            )
        except Exception:
            log.debug("could not add deny reaction", exc_info=True)

    # ------------------------------------------------------------- handling --

    async def handle(self, event: SlackEvent, client: Any) -> AgentResult | None:
        with audit.correlation():
            decision = await self._authz.authorize(event)
            if not decision.allowed:
                # Silence plus a reaction: never confirm who is or is not an
                # operator, or that a channel is wired up.
                await self._react(client, channel=event.channel, ts=event.ts)
                return None

            ack_ts = await self._post(
                client, channel=event.channel, thread_ts=event.reply_thread_ts, text=ACK_TEXT
            )

            async def progress(line: str) -> None:
                if ack_ts:
                    try:
                        await self._update(
                            client, channel=event.channel, ts=ack_ts, text=f"{ACK_TEXT}\n_{line}_"
                        )
                    except Exception:
                        log.debug("progress update failed", exc_info=True)

            try:
                result = await self._agent.investigate(event.text, progress=progress)
            except Exception as exc:
                log.exception("investigation failed")
                text = f"The investigation failed: {type(exc).__name__}: {exc}"
                if ack_ts:
                    await self._update(client, channel=event.channel, ts=ack_ts, text=text)
                else:
                    await self._post(
                        client,
                        channel=event.channel,
                        thread_ts=event.reply_thread_ts,
                        text=text,
                    )
                return None

            reply = format_reply(result)
            if ack_ts:
                await self._update(client, channel=event.channel, ts=ack_ts, text=reply)
            else:
                ack_ts = await self._post(
                    client, channel=event.channel, thread_ts=event.reply_thread_ts, text=reply
                )

            audit.emit(
                "reply_sent",
                slack_user=event.user,
                slack_team=event.team_id,
                slack_channel=event.channel,
                thread_ts=event.reply_thread_ts,
                ok=result.ok,
                reason=result.stop_reason,
                result_bytes=len(reply.encode("utf-8")),
                extra={"tool_calls": len(result.tool_calls), "iterations": result.iterations},
            )
            return result


def format_reply(result: AgentResult) -> str:
    """Finding first, then the tool trace.

    The trace is not optional. Operators will not act on a diagnosis they cannot
    audit, and the trace is what lets them re-run the same queries by hand.
    """
    parts = [result.text.strip()]

    if result.tool_calls:
        lines = [f"  {index}. {call.render()}" for index, call in enumerate(result.tool_calls, 1)]
        parts.append(
            "*Tool calls* (" + str(len(result.tool_calls)) + ")\n```\n" + "\n".join(lines) + "\n```"
        )

    footnotes: list[str] = []
    if result.stop_reason not in ("answered",):
        footnotes.append(f"stopped: {result.stop_reason}")
    if result.dropped_results:
        footnotes.append(f"{result.dropped_results} tool result(s) dropped for context budget")
    if result.redaction_rules:
        unique = ", ".join(dict.fromkeys(result.redaction_rules))
        footnotes.append(f"redacted from tool output: {unique}")
    if footnotes:
        parts.append("_" + " | ".join(footnotes) + "_")

    return "\n\n".join(p for p in parts if p.strip())


def to_event(body: dict[str, Any], event: dict[str, Any]) -> SlackEvent:
    """Build a :class:`SlackEvent` from a Bolt payload.

    ``team_id`` is read from the envelope rather than the inner event: for
    Slack Connect messages the inner ``team`` is the *author's* workspace, and
    authorizing on it is exactly the federated-org hole the check exists to
    close. Taking the envelope's value means a message from another workspace
    fails the comparison.
    """
    return SlackEvent(
        user=str(event.get("user") or ""),
        team_id=str(body.get("team_id") or event.get("team") or ""),
        channel=str(event.get("channel") or ""),
        channel_type=str(event.get("channel_type") or ""),
        text=str(event.get("text") or ""),
        ts=str(event.get("ts") or ""),
        thread_ts=str(event["thread_ts"]) if event.get("thread_ts") else None,
        bot_id=str(event["bot_id"]) if event.get("bot_id") else None,
    )


def build_app(ops: SlackOps | None = None) -> Any:
    """Wire the Bolt async app. Imported lazily so tests need no Slack tokens."""
    from slack_bolt.app.async_app import AsyncApp

    settings = get_settings()
    ops = ops or SlackOps(settings)
    app = AsyncApp(token=settings.slack_bot_token.get_secret_value())

    @app.event("app_mention")
    async def _on_mention(body: dict[str, Any], event: dict[str, Any], client: Any) -> None:
        await ops.handle(to_event(body, event), client)

    @app.event("message")
    async def _on_message(body: dict[str, Any], event: dict[str, Any], client: Any) -> None:
        # Only direct messages; channel messages arrive as app_mention.
        if event.get("channel_type") != "im":
            return
        await ops.handle(to_event(body, event), client)

    return app


async def run() -> None:
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

    settings = get_settings()
    missing = [
        name
        for name, value in (
            ("NRP_OPS_SLACK_BOT_TOKEN", settings.slack_bot_token.get_secret_value()),
            ("NRP_OPS_SLACK_APP_TOKEN", settings.slack_app_token.get_secret_value()),
            ("NRP_OPS_SLACK_TEAM_ID", settings.slack_team_id),
            ("NRP_OPS_LLM_MODEL", settings.llm_model),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"missing required configuration: {', '.join(missing)}")

    audit.emit(
        "startup",
        model=settings.llm_model,
        extra={
            "allowlist_path": str(settings.allowlist_path),
            "kube_mode": settings.kube_mode,
            "prometheus": settings.prometheus_base_url,
        },
    )
    app = build_app()
    handler = AsyncSocketModeHandler(app, settings.slack_app_token.get_secret_value())
    await handler.start_async()  # type: ignore[no-untyped-call]


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("NRP_OPS_LOG_LEVEL", "INFO"),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    main()
