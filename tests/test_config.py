"""Tests for env settings and the hot-reloaded allowlist.

The behaviour under test is deny-by-default: an absent, empty or corrupt policy
file must never widen access.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nrp_ops_agent.config import Allowlist, AllowlistLoader, Settings


class TestSettings:
    def test_env_prefix_and_secret_masking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NRP_OPS_SLACK_BOT_TOKEN", "xoxb-test-token-value")
        monkeypatch.setenv("NRP_OPS_SLACK_TEAM_ID", "T0123456789")
        settings = Settings(_env_file=None)
        assert settings.slack_team_id == "T0123456789"
        assert settings.slack_bot_token.get_secret_value() == "xoxb-test-token-value"
        # The token must not appear in repr/str, which end up in tracebacks.
        assert "xoxb-test-token-value" not in repr(settings)
        assert "xoxb-test-token-value" not in str(settings.slack_bot_token)

    def test_base_urls_lose_trailing_slash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NRP_OPS_PROMETHEUS_BASE_URL", "https://thanos.example.org/")
        monkeypatch.setenv("NRP_OPS_LLM_BASE_URL", "https://llm.example.org/v1/")
        settings = Settings(_env_file=None)
        assert settings.prometheus_base_url == "https://thanos.example.org"
        assert settings.llm_base_url == "https://llm.example.org/v1"

    def test_out_of_range_values_are_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NRP_OPS_RATE_LIMIT_PER_HOUR", "0")
        with pytest.raises(ValueError, match="rate_limit_per_hour"):
            Settings(_env_file=None)

    def test_unknown_kube_mode_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NRP_OPS_KUBE_MODE", "sudo")
        with pytest.raises(ValueError, match="kube_mode"):
            Settings(_env_file=None)

    def test_settings_are_frozen(self) -> None:
        settings = Settings(_env_file=None)
        with pytest.raises(ValueError, match="frozen"):
            settings.slack_team_id = "T999"  # type: ignore[misc]


class TestAllowlistNamespaceMatching:
    def test_empty_policy_denies_everything(self) -> None:
        policy = Allowlist()
        assert policy.namespace_allowed("coder") is False
        assert policy.namespace_allowed("default") is False

    def test_exact_and_glob_matches(self) -> None:
        policy = Allowlist(namespaces=["coder", "jhub-*"])
        assert policy.namespace_allowed("coder") is True
        assert policy.namespace_allowed("jhub-astro") is True
        assert policy.namespace_allowed("coder-staging") is False
        assert policy.namespace_allowed("jhub") is False

    def test_denylist_wins_over_allowlist(self) -> None:
        policy = Allowlist(namespaces=["*"], namespaces_denied=["kube-system", "cert-manager"])
        assert policy.namespace_allowed("coder") is True
        assert policy.namespace_allowed("kube-system") is False
        assert policy.namespace_allowed("cert-manager") is False

    @pytest.mark.parametrize(
        "candidate",
        ["", " coder", "coder ", "coder/pods", "../kube-system", "CODER"],
        ids=["empty", "leading-space", "trailing-space", "slash", "traversal", "case"],
    )
    def test_malformed_namespaces_are_denied(self, candidate: str) -> None:
        policy = Allowlist(namespaces=["coder"])
        assert policy.namespace_allowed(candidate) is False

    def test_matching_is_case_sensitive(self) -> None:
        """fnmatchcase, not fnmatch: on a case-insensitive filesystem plain
        fnmatch would let ``KUBE-SYSTEM`` slip past a ``kube-system`` denial."""
        policy = Allowlist(namespaces=["*"], namespaces_denied=["kube-system"])
        assert policy.namespace_allowed("kube-system") is False
        assert policy.namespace_allowed("Kube-System") is True  # different namespace entirely

    def test_extra_keys_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="admins"):
            Allowlist.model_validate({"operators": ["U1"], "admins": ["U2"]})


class TestAllowlistLoader:
    def test_missing_file_denies_everything(self, tmp_path: Path) -> None:
        loader = AllowlistLoader(tmp_path / "absent.yaml")
        policy = loader.load()
        assert policy.operators == frozenset()
        assert policy.namespace_allowed("coder") is False

    def test_loads_and_caches(self, tmp_path: Path) -> None:
        path = tmp_path / "allowlist.yaml"
        path.write_text("operators: [U111]\nnamespaces: [coder]\n", encoding="utf-8")
        loader = AllowlistLoader(path)
        assert loader.load().operators == frozenset({"U111"})
        assert loader.load() is loader.load()  # cached, not re-parsed

    def test_hot_reload_on_mtime_change(self, tmp_path: Path) -> None:
        path = tmp_path / "allowlist.yaml"
        path.write_text("operators: [U111]\n", encoding="utf-8")
        loader = AllowlistLoader(path)
        assert loader.load().operators == frozenset({"U111"})

        path.write_text("operators: [U111, U222]\n", encoding="utf-8")
        os.utime(path, (0, 0))  # force a distinct mtime without sleeping
        assert loader.load().operators == frozenset({"U111", "U222"})

    def test_corrupt_file_retains_last_good_policy(self, tmp_path: Path) -> None:
        """A bad ConfigMap edit must not fail open, and must not fail closed
        either -- the running agent keeps the policy it already validated."""
        path = tmp_path / "allowlist.yaml"
        path.write_text("operators: [U111]\n", encoding="utf-8")
        loader = AllowlistLoader(path)
        assert loader.load().operators == frozenset({"U111"})

        path.write_text("operators: [U111\n  - broken: [", encoding="utf-8")
        os.utime(path, (0, 0))
        assert loader.load().operators == frozenset({"U111"})

    def test_non_mapping_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "allowlist.yaml"
        path.write_text("- U111\n- U222\n", encoding="utf-8")
        loader = AllowlistLoader(path)
        assert loader.load().operators == frozenset()

    def test_empty_file_denies_everything(self, tmp_path: Path) -> None:
        path = tmp_path / "allowlist.yaml"
        path.write_text("", encoding="utf-8")
        loader = AllowlistLoader(path)
        assert loader.load().operators == frozenset()
        assert loader.load().namespace_allowed("coder") is False
