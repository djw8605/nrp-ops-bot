"""MCP front-end tests.

The MCP server must be a second *front end* to the same dispatcher, never a
second tool surface. These tests pin that: the same tools, the same schemas, the
same namespace enforcement, the same untrusted-output wrapper.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from nrp_ops_agent import mcp_server
from nrp_ops_agent.prompts import CLOSE_TAG
from nrp_ops_agent.tools import registry


@pytest.fixture(scope="module", autouse=True)
def _registered() -> None:
    mcp_server.register_all()


async def _tool(name: str) -> Any:
    return await mcp_server.mcp.get_tool(name)


class TestSurfaceParity:
    async def test_exactly_the_registered_tools_are_exposed(self) -> None:
        for name in registry():
            assert await _tool(name) is not None

    @pytest.mark.parametrize("name", sorted(registry()))
    async def test_schema_matches_the_dispatcher_contract(self, name: str) -> None:
        """The advertised parameters must be the ones actually enforced --
        otherwise an MCP client is told a looser contract than it gets."""
        exposed = (await _tool(name)).parameters
        expected = registry()[name].input_model.model_json_schema()

        assert exposed.get("additionalProperties") is False
        assert set(exposed["properties"]) == set(expected["properties"])
        assert set(exposed.get("required", [])) == set(expected.get("required", []))

    async def test_constraints_survive_into_the_advertised_schema(self) -> None:
        properties = (await _tool("get_logs")).parameters["properties"]
        assert properties["tail_lines"]["maximum"] == 1000
        assert properties["namespace"]["pattern"].startswith("^[a-z0-9]")

    async def test_descriptions_come_from_the_registry(self) -> None:
        assert (await _tool("promql")).description == registry()["promql"].description

    def test_describe_emits_the_full_surface(self) -> None:
        surface = json.loads(mcp_server.describe())
        assert len(surface) == len(registry())
        assert {entry["function"]["name"] for entry in surface} == set(registry())


@pytest.mark.usefixtures("allow_coder_only")
class TestEnforcementIsShared:
    async def test_output_is_wrapped_as_untrusted(self, fake_kube: Any) -> None:
        fake_kube.core._responses["read_namespaced_pod_log"] = "hello"
        handler = mcp_server._make_handler("get_logs", registry()["get_logs"].input_model)
        out = await handler(namespace="coder", pod="coder-0")
        assert out.startswith('<untrusted_tool_output tool="get_logs">')
        assert out.count(CLOSE_TAG) == 1

    async def test_namespace_allowlist_applies(self, fake_kube: Any) -> None:
        handler = mcp_server._make_handler("list_pods", registry()["list_pods"].input_model)
        out = await handler(namespace="kube-system")
        assert "namespace_denied" in out
        assert fake_kube.core.calls == []

    async def test_redaction_applies(self, fake_kube: Any) -> None:
        fake_kube.core._responses["read_namespaced_pod_log"] = "password=hunter2hunter2"
        handler = mcp_server._make_handler("get_logs", registry()["get_logs"].input_model)
        out = await handler(namespace="coder", pod="coder-0")
        assert "hunter2hunter2" not in out
