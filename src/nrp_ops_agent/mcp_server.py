"""Expose the read-only tool surface over MCP.

Security role: this is a second *front end* to the same dispatcher, not a second
tool surface. Every call goes through :func:`nrp_ops_agent.tools.dispatch`, so
argument validation, the namespace allowlist, redaction and the audit trail all
apply identically. Adding a tool here without registering it in ``tools/`` is
impossible by construction: the server enumerates the registry.

Run with ``python -m nrp_ops_agent.mcp_server``. Note that MCP clients are not
subject to the Slack authorization gate, so this entry point must only be
exposed to operators who are already trusted with read access to the cluster.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel

from nrp_ops_agent.prompts import wrap_untrusted
from nrp_ops_agent.tools import dispatch, registry

mcp: FastMCP[None] = FastMCP("nrp-ops-agent")


def _signature_for(model: type[BaseModel]) -> inspect.Signature:
    """Derive a real call signature from a tool's pydantic input model.

    FastMCP introspects the handler rather than accepting a schema, and it
    rejects ``**kwargs``. Generating the signature from the same model the
    dispatcher validates against means the MCP-advertised parameters cannot
    drift from the ones actually enforced.
    """
    parameters = [
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=(
                inspect.Parameter.empty
                if field.is_required()
                else field.get_default(call_default_factory=True)
            ),
            # rebuild_annotation, not .annotation: the latter discards the
            # Field(...) constraints, which would advertise a looser schema than
            # the dispatcher actually enforces.
            annotation=field.rebuild_annotation(),
        )
        for name, field in model.model_fields.items()
    ]
    return inspect.Signature(parameters)


def _make_handler(name: str, model: type[BaseModel]) -> Any:
    async def handler(**kwargs: Any) -> str:
        # Straight through the same chokepoint: validation, the namespace
        # allowlist, redaction and the audit trail all still apply.
        result = await dispatch(name, kwargs)
        # Wrapped even here: an MCP client's model faces the same untrusted text.
        return wrap_untrusted(name, result.json)

    handler.__name__ = name
    handler.__signature__ = _signature_for(model)  # type: ignore[attr-defined]
    handler.__annotations__ = {
        field_name: field.rebuild_annotation() for field_name, field in model.model_fields.items()
    } | {"return": str}
    return handler


def register_all() -> None:
    for name, spec in sorted(registry().items()):
        mcp.tool(name=name, description=spec.description)(
            _make_handler(name, spec.input_model)
        )


def describe() -> str:
    """The tool surface as JSON, for documentation and review."""
    return json.dumps([spec.json_schema() for _, spec in sorted(registry().items())], indent=2)


def main() -> None:
    register_all()
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
