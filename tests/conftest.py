"""Shared fixtures.

The Kubernetes fixtures build fake API objects out of ``SimpleNamespace`` rather
than importing kubernetes-asyncio models, so the tests exercise the attribute
walk the tools actually perform and stay fast enough to run on every save.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

import pytest

from helpers import FakeApi, StubLoader
from nrp_ops_agent import audit
from nrp_ops_agent.config import Allowlist


def _install_allowlist(monkeypatch: pytest.MonkeyPatch, allowlist: Allowlist) -> None:
    """Override the policy everywhere it is consumed.

    ``get_allowlist_loader`` is imported by name into several modules, so each
    one holds its own binding. Patching only the tools module would leave the
    agent's system prompt reading the real (empty) policy.
    """
    from nrp_ops_agent import agent, config
    from nrp_ops_agent.tools import k8s

    loader = StubLoader(allowlist)
    for module in (config, k8s, agent):
        monkeypatch.setattr(module, "get_allowlist_loader", lambda: loader)


@pytest.fixture
def allow_all_namespaces(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_allowlist(monkeypatch, Allowlist(namespaces=["*"], namespaces_denied=["kube-system"]))


@pytest.fixture
def allow_coder_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_allowlist(monkeypatch, Allowlist(namespaces=["coder", "jhub-*"]))


@pytest.fixture
def fake_kube(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install fake core/apps clients; tests populate the canned responses."""
    from nrp_ops_agent.tools import k8s

    core = FakeApi()
    apps = FakeApi()
    clients = k8s.KubeClients(core=core, apps=apps)

    async def _fake_clients() -> k8s.KubeClients:
        return clients

    monkeypatch.setattr(k8s, "_clients", _fake_clients)
    return clients


@pytest.fixture
def audit_stream() -> Iterator[io.StringIO]:
    buf = io.StringIO()
    audit.set_stream(buf)
    try:
        yield buf
    finally:
        audit.set_stream(None)


@pytest.fixture(autouse=True)
def _quiet_audit(request: pytest.FixtureRequest) -> Iterator[None]:
    """Keep audit JSON out of pytest output unless a test asks for it."""
    if "audit_stream" in request.fixturenames:
        yield
        return
    audit.set_stream(io.StringIO())
    try:
        yield
    finally:
        audit.set_stream(None)
