"""Runbook loader.

Playbooks give the agent a reproducible triage order per service instead of
letting it improvise a different investigation every time the same question is
asked. They are a *suggestion* injected into the system prompt, not a script:
the prompt explicitly permits deviation when the evidence points elsewhere.

They also carry known-good PromQL. The model cannot enumerate the metrics this
cluster actually exposes, so asking it to invent queries produces confident
nonsense; a curated query that returns a number is worth more than an invented
one that returns an empty series.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PLAYBOOK_DIR: Final = Path(__file__).resolve().parents[2] / "playbooks"
_DEFAULT_NAME: Final = "_default"


class PlaybookStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    note: str | None = None


class Playbook(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    service: str
    match: tuple[str, ...] = ()
    namespaces: tuple[str, ...] = ()
    docs_hints: tuple[str, ...] = ()
    steps: tuple[PlaybookStep, ...] = ()

    def render(self) -> str:
        """Format for injection into the system prompt."""
        lines = [f"Playbook: {self.service}"]
        if self.namespaces:
            lines.append(f"Likely namespaces: {', '.join(self.namespaces)}")
        if self.docs_hints:
            lines.append(f"Relevant docs paths: {', '.join(self.docs_hints)}")
        lines.append("Suggested triage order (deviate when the evidence points elsewhere):")
        for index, step in enumerate(self.steps, start=1):
            args = ", ".join(f"{k}={v!r}" for k, v in step.args.items())
            lines.append(f"  {index}. {step.name} -- {step.tool}({args})")
            if step.note:
                lines.append(f"     note: {step.note}")
        return "\n".join(lines)


class PlaybookSet:
    def __init__(self, playbooks: list[Playbook]) -> None:
        self._by_service = {p.service: p for p in playbooks}
        self._default = self._by_service.get(_DEFAULT_NAME)

    @property
    def services(self) -> list[str]:
        return sorted(s for s in self._by_service if s != _DEFAULT_NAME)

    def get(self, service: str) -> Playbook | None:
        return self._by_service.get(service)

    def select(self, message: str) -> Playbook | None:
        """Pick the playbook whose keywords best match the operator's message.

        Whole-word matching, with a tolerated plural: operators write "no GPUs
        available" and "workspaces won't start", so a bare word boundary would
        miss the common phrasing -- but "gpu" must still not fire on "gpuest".
        The longest matching keyword wins, so "code-server" beats "coder".
        """
        text = message.lower()
        best: tuple[int, Playbook] | None = None
        for playbook in self._by_service.values():
            if playbook.service == _DEFAULT_NAME:
                continue
            score = 0
            for keyword in playbook.match:
                pattern = rf"(?<![a-z0-9]){re.escape(keyword.lower())}s?(?![a-z0-9])"
                if re.search(pattern, text):
                    score = max(score, len(keyword))
            if score and (best is None or score > best[0]):
                best = (score, playbook)
        if best is not None:
            return best[1]
        return self._default


def load_playbooks(directory: Path | None = None) -> PlaybookSet:
    """Load every ``*.yaml`` in the playbook directory.

    A malformed playbook is skipped rather than fatal: a typo in one runbook must
    not take the bot offline for every other service.
    """
    directory = directory or DEFAULT_PLAYBOOK_DIR
    playbooks: list[Playbook] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.yaml")):
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    playbooks.append(Playbook.model_validate(raw))
            except (OSError, ValueError, yaml.YAMLError):
                continue
    return PlaybookSet(playbooks)


@lru_cache(maxsize=1)
def get_playbooks() -> PlaybookSet:
    return load_playbooks()
