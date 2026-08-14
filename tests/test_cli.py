"""Tests for the local development CLI.

The CLI's job is to make a misconfigured local run *say so*. Every condition it
checks otherwise produces a plausible-looking wrong answer rather than an error:
an empty namespace allowlist reads as an empty cluster, a missing docs index
reads as undocumented software. A preflight that passed silently in those
states would be worse than no preflight.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nrp_ops_agent import cli


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    from nrp_ops_agent.config import get_allowlist_loader, get_settings

    get_settings.cache_clear()
    get_allowlist_loader.cache_clear()


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    model: str = "a-model",
    key: str = "a-key",
    namespaces: str = "coder",
    docs: bool = True,
) -> None:
    allowlist = tmp_path / "allowlist.yaml"
    allowlist.write_text(f"namespaces: [{namespaces}]\n" if namespaces else "namespaces: []\n")

    db = tmp_path / "docs.sqlite3"
    if docs:
        db.write_text("")

    monkeypatch.setenv("NRP_OPS_LLM_MODEL", model)
    monkeypatch.setenv("NRP_OPS_LLM_API_KEY", key)
    monkeypatch.setenv("NRP_OPS_ALLOWLIST_PATH", str(allowlist))
    monkeypatch.setenv("NRP_OPS_DOCS_DB_PATH", str(db))


class TestPreflight:
    def test_a_fully_configured_run_reports_no_problems(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _configure(monkeypatch, tmp_path)
        assert cli._preflight() == []
        assert cli.main(["--check"]) == 0

    def test_an_empty_allowlist_is_reported_not_tolerated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The failure this prevents: every k8s tool refuses before calling the
        API, and the agent reports a healthy, empty namespace."""
        _configure(monkeypatch, tmp_path, namespaces="")
        problems = cli._preflight()
        assert any("No namespaces are allowed" in p for p in problems)
        assert cli.main(["--check"]) == 1

    def test_a_missing_docs_index_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _configure(monkeypatch, tmp_path, docs=False)
        assert any("No docs index" in p for p in cli._preflight())

    def test_a_missing_model_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _configure(monkeypatch, tmp_path, model="")
        assert any("NRP_OPS_LLM_MODEL" in p for p in cli._preflight())

    def test_a_missing_key_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _configure(monkeypatch, tmp_path, key="")
        assert any("NRP_OPS_LLM_API_KEY" in p for p in cli._preflight())


class TestNonModelPaths:
    def test_which_playbook_selects_without_calling_the_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No HTTP client is constructed on this path, so it stays usable when
        the gateway is unreachable."""
        _configure(monkeypatch, tmp_path)
        assert cli.main(["--which-playbook", "coder", "is", "down"]) == 0
        assert "Playbook: coder" in capsys.readouterr().out

    def test_an_empty_question_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _configure(monkeypatch, tmp_path)
        with pytest.raises(SystemExit):
            cli.main([])
