"""Playbook loading, matching and shape validation.

The shipped playbooks are validated against the live tool registry: a runbook
step naming a tool that does not exist, or passing arguments that tool would
reject, is a bug that surfaces as the agent wasting calls at 3am.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nrp_ops_agent.playbooks import DEFAULT_PLAYBOOK_DIR, Playbook, load_playbooks
from nrp_ops_agent.tools import registry

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
        playbook = SHIPPED.get(service)
        assert playbook is not None
        specs = registry()
        for step in playbook.steps:
            specs[step.tool].input_model.model_validate(step.args)

    def test_unverified_promql_is_marked(self) -> None:
        """Any PromQL not confirmed against the live cluster must say so, so a
        reader never mistakes a guess for a checked query."""
        for service in SHIPPED.services:
            playbook = SHIPPED.get(service)
            assert playbook is not None
            for step in playbook.steps:
                if step.tool == "promql":
                    assert step.note and "TODO: verify" in step.note, f"{service}/{step.name}"

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
