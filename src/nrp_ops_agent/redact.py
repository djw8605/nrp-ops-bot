"""Secret scrubbing.

Security role: this is the last line of defence against credential exfiltration.

The agent's ServiceAccount is denied ``secrets`` in RBAC, but that is *not*
sufficient: ``pods/log`` is itself a secret-exfiltration channel, because
applications routinely log their own credentials -- database URIs with embedded
passwords, ``Authorization: Bearer`` headers at debug level, mounted
ServiceAccount tokens echoed by a misbehaving client library. An operator asking
"why is coder failing" can therefore pull a live credential into the model
context and into a Slack channel without anyone intending it.

This module is applied at two chokepoints:

1. tool output, *before* it is appended to the model context;
2. outbound Slack text, immediately before ``chat.postMessage``/``chat.update``.

Rule names that fire are audited; matched values never are.

These are regexes, so they are best-effort: a bespoke credential format will slip
through. That is an argument for keeping the namespace allowlist tight, not an
argument against this layer.
"""

from __future__ import annotations

import re
from typing import Any, Final

REDACTION_TEMPLATE: Final = "[REDACTED:{rule}]"

# A value already replaced by this module must never be re-matched by a later,
# broader rule; that would double-redact and report a bogus rule name.
_PLACEHOLDER_GUARD: Final = r"(?!\[REDACTED)"


def _placeholder(rule: str) -> str:
    return REDACTION_TEMPLATE.format(rule=rule)


class _Rule:
    """A named pattern plus the substitution that neutralises it."""

    __slots__ = ("_group", "name", "pattern")

    def __init__(self, name: str, pattern: re.Pattern[str], group: int = 0) -> None:
        self.name = name
        self.pattern = pattern
        self._group = group

    def apply(self, text: str) -> tuple[str, bool]:
        """Return ``(scrubbed, hit)``."""
        hit = False
        replacement = _placeholder(self.name)

        def _sub(match: re.Match[str]) -> str:
            nonlocal hit
            hit = True
            if self._group == 0:
                return replacement
            # Preserve everything outside the sensitive group so that the
            # surrounding log line stays diagnostically useful.
            whole = match.group(0)
            start, end = match.span(self._group)
            offset = match.start()
            return whole[: start - offset] + replacement + whole[end - offset :]

        return self.pattern.sub(_sub, text), hit


# Ordered: the most specific and most damaging first, so that a broad rule never
# claims a match that a precise rule would have labelled better.
_RULES: Final[tuple[_Rule, ...]] = (
    _Rule(
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"
            r"[\s\S]*?"
            r"-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"
        ),
    ),
    _Rule(
        "dockerconfigjson",
        re.compile(r"""(?i)"\.?dockerconfigjson"\s*:\s*"([A-Za-z0-9+/=]{8,})\"""", re.VERBOSE),
        group=1,
    ),
    _Rule(
        "docker_auth_entry",
        re.compile(r'(?i)"auth"\s*:\s*"([A-Za-z0-9+/=]{8,})"'),
        group=1,
    ),
    _Rule(
        "htpasswd_line",
        re.compile(r"(?m)^[\w.@+-]{1,64}:(\$(?:apr1|2[abxy]|1|5|6)\$[^\s:]{8,})"),
        group=1,
    ),
    _Rule(
        "uri_credentials",
        re.compile(
            r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s/:@]{1,128}:"
            rf"{_PLACEHOLDER_GUARD}([^\s/@]{{1,256}})@"
        ),
        group=1,
    ),
    _Rule(
        "basic_auth_header",
        re.compile(rf"(?i)\bbasic\s+{_PLACEHOLDER_GUARD}([A-Za-z0-9+/=]{{12,}})"),
        group=1,
    ),
    _Rule(
        "bearer_token",
        re.compile(rf"(?i)\bbearer\s+{_PLACEHOLDER_GUARD}([A-Za-z0-9._~+/=\-]{{12,}})"),
        group=1,
    ),
    # Three dot-separated base64url segments starting with the "{" header that
    # every JWT begins with. Covers Kubernetes ServiceAccount tokens verbatim.
    _Rule(
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_=\-]{4,}\.[A-Za-z0-9_=\-]{4,}\.[A-Za-z0-9_=\-]*"),
    ),
    _Rule("slack_token", re.compile(r"\bxox[baprseo]-[A-Za-z0-9\-]{8,}")),
    _Rule("slack_app_token", re.compile(r"\bxapp-\d-[A-Za-z0-9\-]{8,}")),
    _Rule("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b")),
    _Rule("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    _Rule("gitlab_token", re.compile(r"\bgl(?:pat|rt|dt|ft|soat)-[A-Za-z0-9_\-]{16,}")),
    _Rule(
        "aws_access_key_id",
        re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ABIA|ACCA)[A-Z0-9]{16}\b"),
    ),
    _Rule(
        "aws_secret_access_key",
        re.compile(
            r"(?i)\baws_?secret_?(?:access_?)?key\b\s*[:=]\s*"
            rf"[\"']?{_PLACEHOLDER_GUARD}([A-Za-z0-9/+=]{{40}})[\"']?"
        ),
        group=1,
    ),
    # Deliberately last and deliberately broad: catches the long tail of
    # `password=...` / `api_key: ...` that application logs are full of. Slight
    # over-redaction here is preferable to leaking a live credential into Slack.
    _Rule(
        "credential_assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|token|api[_\-]?key|access[_\-]?key"
            r"|client[_\-]?secret|private[_\-]?key)\b\s*[:=]\s*"
            rf"[\"']?{_PLACEHOLDER_GUARD}([^\s\"',;}}\]]{{6,}})[\"']?"
        ),
        group=1,
    ),
)

RULE_NAMES: Final[tuple[str, ...]] = tuple(rule.name for rule in _RULES)


def redact(text: str) -> tuple[str, list[str]]:
    """Scrub credentials from ``text``.

    Returns the scrubbed text and the names of the rules that fired, in the order
    they were applied. The matched *values* are never returned, logged or
    otherwise retained.
    """
    if not text:
        return text, []
    hits: list[str] = []
    for rule in _RULES:
        text, hit = rule.apply(text)
        if hit:
            hits.append(rule.name)
    return text, hits


def redact_structure(obj: Any) -> tuple[Any, list[str]]:
    """Recursively :func:`redact` every string inside a JSON-shaped object.

    Dict *keys* are scrubbed too: a Secret-derived key name can itself be
    sensitive, and a key is just as reachable by an operator reading the reply.
    """
    hits: list[str] = []

    def _walk(node: Any) -> Any:
        if isinstance(node, str):
            scrubbed, rules = redact(node)
            hits.extend(rules)
            return scrubbed
        if isinstance(node, dict):
            out: dict[Any, Any] = {}
            for key, value in node.items():
                new_key = _walk(key) if isinstance(key, str) else key
                out[new_key] = _walk(value)
            return out
        if isinstance(node, list):
            return [_walk(item) for item in node]
        if isinstance(node, tuple):
            return tuple(_walk(item) for item in node)
        return node

    result = _walk(obj)
    # Preserve first-seen order while de-duplicating.
    return result, list(dict.fromkeys(hits))
