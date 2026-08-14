"""Fake Kubernetes API objects and stubs shared by the test modules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nrp_ops_agent.config import Allowlist


def ns(**kwargs: Any) -> SimpleNamespace:
    """Shorthand for a fake API object."""
    return SimpleNamespace(**kwargs)


def ago(seconds: int) -> datetime:
    return datetime.now(UTC) - timedelta(seconds=seconds)


class StubLoader:
    """Stands in for AllowlistLoader without touching the filesystem."""

    def __init__(self, allowlist: Allowlist) -> None:
        self._allowlist = allowlist
        self.path = Path("/dev/null")

    def load(self) -> Allowlist:
        return self._allowlist


class FakeApi:
    """Records calls and replays canned responses or raises canned errors."""

    def __init__(self, **responses: Any) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        async def _call(**kwargs: Any) -> Any:
            self.calls.append((name, kwargs))
            if name not in self._responses:
                raise AssertionError(f"unexpected API call: {name}")
            value = self._responses[name]
            if isinstance(value, Exception):
                raise value
            return value

        return _call

    def kwargs_for(self, method: str) -> dict[str, Any]:
        for name, kwargs in self.calls:
            if name == method:
                return kwargs
        raise AssertionError(f"{method} was never called")


class ApiException(Exception):
    """Mirrors kubernetes_asyncio.client.ApiException's shape."""

    def __init__(self, status: int, reason: str = "") -> None:
        super().__init__(f"{status}: {reason}")
        self.status = status
        self.reason = reason
