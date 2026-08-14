"""Authorization gate.

Security role: this is the front door. Nothing -- no model call, no tool call --
happens before :func:`authorize` returns an allow.

Checks run in order and short-circuit on the first failure:

1. **Team ID.** The event must come from the configured workspace. This is what
   blocks Slack Connect users from federated organisations, who can otherwise
   appear in a shared channel and mention the bot.
2. **User ID.** Membership is by Slack user ID (``U...``/``W...``) only. Display
   names and emails are user-editable, so authorizing on them would let anyone
   rename themselves into the allowlist.
3. **Channel.** An allowlisted channel, or a DM when DMs are enabled.
4. **Rate limit.** A per-user token bucket, refilled continuously.

Every decision, allow *and* deny, is audited with the user ID, team, channel and
the raw message text.

On deny the caller reacts with an emoji and posts nothing. Posting "you are not
authorized" would turn the bot into an enumeration oracle: an outsider could
learn who is an operator, which channels are wired up, and whether a user ID
exists, purely from which messages get a reply.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import StrEnum

from nrp_ops_agent import audit
from nrp_ops_agent.config import AllowlistLoader, Settings, get_allowlist_loader, get_settings


class DenyReason(StrEnum):
    WRONG_TEAM = "wrong_team"
    UNKNOWN_USER = "unknown_user"
    CHANNEL_NOT_ALLOWED = "channel_not_allowed"
    DMS_DISABLED = "dms_disabled"
    RATE_LIMITED = "rate_limited"
    BOT_MESSAGE = "bot_message"
    EMPTY_MESSAGE = "empty_message"


@dataclass(frozen=True)
class SlackEvent:
    """The fields authorization is allowed to see.

    Deliberately minimal: there is no display name or email field here, so no
    future edit can accidentally authorize on one.
    """

    user: str
    team_id: str
    channel: str
    channel_type: str
    text: str
    ts: str
    thread_ts: str | None = None
    bot_id: str | None = None

    @property
    def is_dm(self) -> bool:
        return self.channel_type == "im"

    @property
    def reply_thread_ts(self) -> str:
        """Always reply in a thread: the parent if there is one, else this message."""
        return self.thread_ts or self.ts


@dataclass(frozen=True)
class AuthzDecision:
    allowed: bool
    reason: DenyReason | None = None
    retry_after_s: float | None = None


class TokenBucket:
    """Per-user rate limiter, refilled continuously over one hour."""

    def __init__(self, capacity: int, refill_seconds: float = 3600.0) -> None:
        self._capacity = float(capacity)
        self._rate = capacity / refill_seconds
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, float]] = {}

    def consume(self, key: str, *, now: float | None = None) -> tuple[bool, float]:
        """Take one token. Returns ``(allowed, seconds_until_next_token)``."""
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._state.get(key, (self._capacity, now))
            tokens = min(self._capacity, tokens + (now - last) * self._rate)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                return True, 0.0
            self._state[key] = (tokens, now)
            return False, (1.0 - tokens) / self._rate

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._state.clear()
            else:
                self._state.pop(key, None)


class Authorizer:
    def __init__(
        self,
        settings: Settings | None = None,
        loader: AllowlistLoader | None = None,
        bucket: TokenBucket | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._loader = loader or get_allowlist_loader()
        self._bucket = bucket or TokenBucket(self._settings.rate_limit_per_hour)

    async def authorize(self, event: SlackEvent) -> AuthzDecision:
        decision = self._decide(event)
        audit.emit(
            "authz_allow" if decision.allowed else "authz_deny",
            slack_user=event.user,
            slack_team=event.team_id,
            slack_channel=event.channel,
            thread_ts=event.reply_thread_ts,
            text=event.text,
            ok=decision.allowed,
            reason=str(decision.reason) if decision.reason else None,
        )
        return decision

    def _decide(self, event: SlackEvent) -> AuthzDecision:
        # A bot's own messages must never re-enter the loop.
        if event.bot_id:
            return AuthzDecision(False, DenyReason.BOT_MESSAGE)

        if not event.text.strip():
            return AuthzDecision(False, DenyReason.EMPTY_MESSAGE)

        expected_team = self._settings.slack_team_id
        if not expected_team or event.team_id != expected_team:
            return AuthzDecision(False, DenyReason.WRONG_TEAM)

        allowlist = self._loader.load()
        if event.user not in allowlist.operators:
            return AuthzDecision(False, DenyReason.UNKNOWN_USER)

        if event.is_dm:
            if not self._settings.slack_allow_dms:
                return AuthzDecision(False, DenyReason.DMS_DISABLED)
        elif event.channel not in allowlist.channels:
            return AuthzDecision(False, DenyReason.CHANNEL_NOT_ALLOWED)

        allowed, retry_after = self._bucket.consume(event.user)
        if not allowed:
            return AuthzDecision(False, DenyReason.RATE_LIMITED, retry_after_s=retry_after)

        return AuthzDecision(True)
