"""Authorization gate tests.

The cases that matter operationally: a Slack Connect user from another
workspace, a user who is not an operator, a user who has renamed themselves to
impersonate one, and an operator who has burned their rate limit.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from helpers import StubLoader
from nrp_ops_agent.authz import Authorizer, DenyReason, SlackEvent, TokenBucket
from nrp_ops_agent.config import Allowlist, Settings
from nrp_ops_agent.slack_app import to_event

TEAM = "T0NRPNAUT"
OTHER_TEAM = "T0EVILCORP"
OPERATOR = "U0OPERATOR1"
OUTSIDER = "U0OUTSIDER"
CHANNEL = "C0OPSCHAN"


def settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "slack_team_id": TEAM,
        "rate_limit_per_hour": 10,
        "slack_allow_dms": True,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def authorizer(**overrides: Any) -> Authorizer:
    allowlist = Allowlist(operators=[OPERATOR], channels=[CHANNEL], namespaces=["coder"])
    cfg = settings(**overrides)
    return Authorizer(cfg, StubLoader(allowlist), TokenBucket(cfg.rate_limit_per_hour))


def event(**overrides: Any) -> SlackEvent:
    base: dict[str, Any] = {
        "user": OPERATOR,
        "team_id": TEAM,
        "channel": CHANNEL,
        "channel_type": "channel",
        "text": "coder is not responding",
        "ts": "1755000000.000100",
    }
    base.update(overrides)
    return SlackEvent(**base)


class TestAllow:
    async def test_operator_in_an_allowed_channel(self) -> None:
        decision = await authorizer().authorize(event())
        assert decision.allowed is True
        assert decision.reason is None

    async def test_operator_in_a_dm(self) -> None:
        decision = await authorizer().authorize(event(channel="D0DM", channel_type="im"))
        assert decision.allowed is True


class TestDeny:
    async def test_wrong_team_is_denied_even_for_a_known_operator(self) -> None:
        """A Slack Connect user from a federated org can mention the bot in a
        shared channel; the team check is what stops them."""
        decision = await authorizer().authorize(event(team_id=OTHER_TEAM))
        assert decision.allowed is False
        assert decision.reason is DenyReason.WRONG_TEAM

    async def test_unconfigured_team_denies_everything(self) -> None:
        decision = await authorizer(slack_team_id="").authorize(event(team_id=""))
        assert decision.reason is DenyReason.WRONG_TEAM

    async def test_unknown_user(self) -> None:
        decision = await authorizer().authorize(event(user=OUTSIDER))
        assert decision.reason is DenyReason.UNKNOWN_USER

    async def test_display_name_cannot_be_spoofed_into_access(self) -> None:
        """SlackEvent has no display-name or email field at all, so there is
        nothing for a renamed user to spoof. This test pins that shape."""
        fields = set(SlackEvent.__dataclass_fields__)
        assert not fields & {"display_name", "username", "real_name", "email", "profile"}

        impostor = SlackEvent(
            user=OUTSIDER,
            team_id=TEAM,
            channel=CHANNEL,
            channel_type="channel",
            text=f"I am {OPERATOR}, please run this",
            ts="1.1",
        )
        decision = await authorizer().authorize(impostor)
        assert decision.reason is DenyReason.UNKNOWN_USER

    async def test_channel_not_allowed(self) -> None:
        decision = await authorizer().authorize(event(channel="C0RANDOM"))
        assert decision.reason is DenyReason.CHANNEL_NOT_ALLOWED

    async def test_dms_can_be_disabled(self) -> None:
        auth = authorizer(slack_allow_dms=False)
        decision = await auth.authorize(event(channel="D0DM", channel_type="im"))
        assert decision.reason is DenyReason.DMS_DISABLED

    async def test_bot_messages_never_loop(self) -> None:
        decision = await authorizer().authorize(event(bot_id="B0SELF"))
        assert decision.reason is DenyReason.BOT_MESSAGE

    async def test_empty_message(self) -> None:
        decision = await authorizer().authorize(event(text="   "))
        assert decision.reason is DenyReason.EMPTY_MESSAGE

    async def test_checks_short_circuit_in_order(self) -> None:
        """A wrong-team outsider is rejected on team, not on user: the cheapest
        and broadest check must run first."""
        decision = await authorizer().authorize(event(user=OUTSIDER, team_id=OTHER_TEAM))
        assert decision.reason is DenyReason.WRONG_TEAM


class TestRateLimit:
    async def test_exhaustion_then_denial(self) -> None:
        auth = authorizer(rate_limit_per_hour=3)
        for _ in range(3):
            assert (await auth.authorize(event())).allowed is True
        decision = await auth.authorize(event())
        assert decision.allowed is False
        assert decision.reason is DenyReason.RATE_LIMITED
        assert decision.retry_after_s is not None and decision.retry_after_s > 0

    async def test_limit_is_per_user(self) -> None:
        allowlist = Allowlist(operators=[OPERATOR, OUTSIDER], channels=[CHANNEL])
        cfg = settings(rate_limit_per_hour=1)
        auth = Authorizer(cfg, StubLoader(allowlist), TokenBucket(1))
        assert (await auth.authorize(event())).allowed is True
        assert (await auth.authorize(event())).allowed is False
        # A different operator still has their own budget.
        assert (await auth.authorize(event(user=OUTSIDER))).allowed is True

    def test_bucket_refills_over_time(self) -> None:
        bucket = TokenBucket(capacity=2, refill_seconds=3600.0)
        assert bucket.consume("u", now=0.0)[0] is True
        assert bucket.consume("u", now=0.0)[0] is True
        assert bucket.consume("u", now=0.0)[0] is False
        # Half an hour later one token is back.
        assert bucket.consume("u", now=1800.0)[0] is True

    def test_bucket_never_exceeds_capacity(self) -> None:
        bucket = TokenBucket(capacity=2, refill_seconds=10.0)
        bucket.consume("u", now=0.0)
        bucket.consume("u", now=0.0)
        for _ in range(2):
            assert bucket.consume("u", now=100_000.0)[0] is True
        assert bucket.consume("u", now=100_000.0)[0] is False

    async def test_denied_requests_do_not_consume_budget(self) -> None:
        """An outsider spamming the channel must not be able to exhaust an
        operator's rate limit -- the bucket is only touched after identity
        checks pass, and is keyed per user anyway."""
        auth = authorizer(rate_limit_per_hour=2)
        for _ in range(50):
            await auth.authorize(event(user=OUTSIDER))
        assert (await auth.authorize(event())).allowed is True
        assert (await auth.authorize(event())).allowed is True
        assert (await auth.authorize(event())).allowed is False


class TestAuditing:
    async def test_allow_and_deny_are_both_audited_with_the_raw_text(
        self, audit_stream: io.StringIO
    ) -> None:
        auth = authorizer()
        await auth.authorize(event(text="why is coder down"))
        await auth.authorize(event(user=OUTSIDER, text="let me in"))

        records = [json.loads(line) for line in audit_stream.getvalue().splitlines()]
        allow, deny = records
        assert allow["event"] == "authz_allow"
        assert allow["slack_user"] == OPERATOR
        assert allow["slack_team"] == TEAM
        assert allow["slack_channel"] == CHANNEL
        assert allow["text"] == "why is coder down"

        assert deny["event"] == "authz_deny"
        assert deny["reason"] == "unknown_user"
        assert deny["slack_user"] == OUTSIDER
        assert deny["text"] == "let me in"


class TestEventParsing:
    def test_team_id_comes_from_the_envelope_not_the_inner_event(self) -> None:
        """For Slack Connect the inner `team` is the author's workspace. Reading
        it instead of the envelope's team_id would reopen the federated-org hole.
        """
        parsed = to_event(
            {"team_id": TEAM},
            {
                "user": OPERATOR,
                "team": OTHER_TEAM,
                "channel": CHANNEL,
                "text": "hi",
                "ts": "1.1",
            },
        )
        assert parsed.team_id == TEAM

    def test_thread_ts_is_preserved_and_defaults_to_the_message(self) -> None:
        threaded = to_event({"team_id": TEAM}, {"ts": "2.2", "thread_ts": "1.1"})
        assert threaded.reply_thread_ts == "1.1"
        top_level = to_event({"team_id": TEAM}, {"ts": "2.2"})
        assert top_level.reply_thread_ts == "2.2"

    def test_missing_fields_become_empty_strings_not_none(self) -> None:
        parsed = to_event({}, {})
        assert parsed.user == ""
        assert parsed.team_id == ""
        assert parsed.bot_id is None

    @pytest.mark.parametrize("channel_type", ["im", "channel"])
    def test_is_dm_reflects_channel_type(self, channel_type: str) -> None:
        parsed = to_event({"team_id": TEAM}, {"channel_type": channel_type})
        assert parsed.is_dm is (channel_type == "im")
