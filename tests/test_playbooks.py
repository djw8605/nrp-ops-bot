"""Playbook loading, matching and shape validation.

The shipped playbooks are validated against the live tool registry: a runbook
step naming a tool that does not exist, or passing arguments that tool would
reject, is a bug that surfaces as the agent wasting calls at 3am.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nrp_ops_agent.playbooks import DEFAULT_PLAYBOOK_DIR, Playbook, load_playbooks
from nrp_ops_agent.tools import registry

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED = load_playbooks()


class TestShippedPlaybooks:
    def test_the_expected_playbooks_exist(self) -> None:
        assert set(SHIPPED.services) == {"coder", "jupyterhub", "ceph", "gpu"}
        assert SHIPPED.get("_default") is not None

    def test_playbook_directory_is_found_from_the_installed_package(self) -> None:
        assert DEFAULT_PLAYBOOK_DIR.is_dir()
        assert (DEFAULT_PLAYBOOK_DIR / "coder.yaml").is_file()

    @pytest.mark.parametrize("service", ["coder", "jupyterhub", "ceph", "gpu"])
    def test_every_step_names_a_registered_tool(self, service: str) -> None:
        playbook = SHIPPED.get(service)
        assert playbook is not None
        known = set(registry())
        for step in playbook.steps:
            assert step.tool in known, f"{service}: unknown tool {step.tool}"

    @pytest.mark.parametrize("service", ["coder", "jupyterhub", "ceph", "gpu"])
    def test_every_step_validates_against_its_tool_schema(self, service: str) -> None:
        """``<...>`` marks a value the agent must discover at runtime -- a hub
        pod carries a generated suffix, and Ceph has nine candidate namespaces,
        so hardcoding either ships a name that is stale or simply wrong. Those
        stand in as a valid dummy so every *other* argument is still checked.
        """
        playbook = SHIPPED.get(service)
        assert playbook is not None
        specs = registry()
        for step in playbook.steps:
            args = {
                k: ("placeholder" if isinstance(v, str) and v.startswith("<") else v)
                for k, v in step.args.items()
            }
            specs[step.tool].input_model.model_validate(args)

    def test_promql_states_whether_it_was_verified(self) -> None:
        """Any PromQL must say whether it was confirmed against the live
        cluster, so a reader never mistakes a guess for a checked query. Both
        answers are acceptable; silence is not."""
        for service in SHIPPED.services:
            playbook = SHIPPED.get(service)
            assert playbook is not None
            for step in playbook.steps:
                if step.tool == "promql":
                    note = step.note or ""
                    assert "TODO: verify" in note or "Verified" in note, f"{service}/{step.name}"

    def test_every_playbook_namespace_is_permitted_by_the_shipped_allowlist(self) -> None:
        """The two halves have to agree. They did not: the allowlist granted
        `jupyterhub` and `rook-ceph` while the cluster has `jupyterlab` and nine
        `rook-*` namespaces, so every Ceph and JupyterHub step was refused
        inside the tool before any API call -- a silent denial that looks like a
        healthy "nothing found" rather than a misconfiguration.
        """
        import fnmatch

        allowlist = yaml.safe_load((REPO_ROOT / "deploy" / "configmap-allowlist.yaml").read_text())
        policy = yaml.safe_load(allowlist["data"]["allowlist.yaml"])
        patterns: list[str] = policy["namespaces"]
        denied: list[str] = policy.get("namespaces_denied", [])

        for service in SHIPPED.services:
            playbook = SHIPPED.get(service)
            assert playbook is not None
            for namespace in playbook.namespaces:
                assert any(fnmatch.fnmatch(namespace, p) for p in patterns), (
                    f"{service}: playbook names namespace {namespace!r}, which the "
                    f"shipped allowlist does not permit"
                )
                assert not any(fnmatch.fnmatch(namespace, d) for d in denied), (
                    f"{service}: namespace {namespace!r} is explicitly denied"
                )

    def test_default_playbook_is_placeholder_shaped(self) -> None:
        """The fallback uses <placeholders> rather than invented namespaces, so
        the model must discover the real ones instead of copying a guess."""
        default = SHIPPED.get("_default")
        assert default is not None
        assert any("<" in str(v) for step in default.steps for v in step.args.values())


class TestMatching:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("users are reporting that coder is not responding", "coder"),
            ("code-server workspaces won't start", "coder"),
            ("jupyterhub spawn is timing out", "jupyterhub"),
            ("nobody can get a notebook", "jupyterhub"),
            ("ceph is degraded", "ceph"),
            ("PVC stuck pending", "ceph"),
            ("no GPUs available for my job", "gpu"),
            ("nvidia device plugin looks broken", "gpu"),
        ],
    )
    def test_keyword_selection(self, message: str, expected: str) -> None:
        selected = SHIPPED.select(message)
        assert selected is not None and selected.service == expected

    def test_unmatched_message_falls_back_to_default(self) -> None:
        selected = SHIPPED.select("the frobnicator is emitting sparks")
        assert selected is not None and selected.service == "_default"

    def test_matching_is_whole_word(self) -> None:
        """'gpu' must not fire on 'gpuest'; a substring match would route every
        question containing a common fragment to the wrong runbook."""
        selected = SHIPPED.select("the gpuest thing happened")
        assert selected is not None and selected.service == "_default"

    def test_longest_keyword_wins(self) -> None:
        """'code-server' is more specific than 'coder' and than 'hub'."""
        selected = SHIPPED.select("code-server in the hub namespace")
        assert selected is not None and selected.service == "coder"

    def test_matching_is_case_insensitive(self) -> None:
        selected = SHIPPED.select("CODER IS DOWN")
        assert selected is not None and selected.service == "coder"


class TestLoading:
    def test_malformed_playbook_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        (tmp_path / "good.yaml").write_text("service: good\nmatch: [good]\n", encoding="utf-8")
        (tmp_path / "bad.yaml").write_text("service: bad\nsteps: [[[", encoding="utf-8")
        (tmp_path / "wrongkeys.yaml").write_text("service: x\nnope: 1\n", encoding="utf-8")
        loaded = load_playbooks(tmp_path)
        assert loaded.services == ["good"]

    def test_missing_directory_yields_an_empty_set(self, tmp_path: Path) -> None:
        loaded = load_playbooks(tmp_path / "absent")
        assert loaded.services == []
        assert loaded.select("anything") is None

    def test_render_includes_order_namespaces_and_docs(self) -> None:
        playbook = SHIPPED.get("coder")
        assert playbook is not None
        rendered = playbook.render()
        assert "Playbook: coder" in rendered
        assert "Likely namespaces: coder" in rendered
        assert "admindocs/upgrades/coder" in rendered
        assert "deviate when the evidence points elsewhere" in rendered
        assert "1. workload health" in rendered

    def test_empty_playbook_renders_without_error(self) -> None:
        assert "Playbook: x" in Playbook(service="x").render()


class TestPackaging:
    """The container is the only place this breaks, and it breaks quietly.

    ``load_playbooks`` treats a missing directory as "no playbooks" rather than
    an error. A wheel that omitted the runbooks would therefore start cleanly,
    answer every question with no triage order, and never say why -- so the
    packaging is asserted here rather than discovered in production.
    """

    def test_wheel_declares_the_playbooks_as_force_included(self) -> None:
        import tomllib

        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
        assert force_include["playbooks"] == "nrp_ops_agent/playbooks"

    def test_packaged_location_wins_when_it_exists(self, tmp_path: Path) -> None:
        """An installed wheel puts the runbooks beside the module; the editable
        dev checkout does not and must fall back to the repository root."""
        from nrp_ops_agent import playbooks as module

        assert module._default_playbook_dir().is_dir()
        assert sorted(p.name for p in module._default_playbook_dir().glob("*.yaml")) == [
            "_default.yaml",
            "ceph.yaml",
            "coder.yaml",
            "gpu.yaml",
            "jupyterhub.yaml",
        ]
