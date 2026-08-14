"""Environment-driven configuration.

Security role: this module is the single place where endpoints, credentials and
policy knobs enter the process. Nothing else in the package may hardcode a URL,
a token, a namespace or a Slack ID. Credentials are held as
:class:`~pydantic.SecretStr` so that an accidental ``repr()`` in a log line or a
traceback does not leak them.

Two kinds of settings live here:

* :class:`Settings` -- immutable, read once from the environment at startup.
* :class:`AllowlistLoader` -- the *mutable* policy (operators, channels,
  namespaces) mounted from a ConfigMap and hot-reloaded on mtime change, so that
  adding an operator is a ``kubectl apply``, not a redeploy.
"""

from __future__ import annotations

import fnmatch
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

KubeMode = Literal["auto", "incluster", "kubeconfig"]

NamespaceDecision = Literal["allowed", "denylisted", "not_allowlisted", "malformed"]
"""Outcome of :meth:`Allowlist.namespace_decision`. Only ``allowed`` permits a call."""


class Settings(BaseSettings):
    """Static, env-driven configuration. Constructed once via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="NRP_OPS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # ---------------------------------------------------------------- Slack --
    slack_bot_token: SecretStr = Field(
        default=SecretStr(""),
        description="xoxb-... bot token. Scopes: app_mentions:read, chat:write, "
        "im:history, reactions:write, and channels:history/groups:history for "
        "reading thread context in channels.",
    )
    slack_app_token: SecretStr = Field(
        default=SecretStr(""),
        description="xapp-... app-level token with connections:write, for Socket Mode.",
    )
    slack_team_id: str = Field(
        default="",
        description="Workspace team ID (T...). Events from any other team are denied, "
        "which is what blocks Slack Connect users from federated orgs.",
    )
    slack_deny_reaction: str = Field(
        default="no_entry",
        description="Emoji reacted to denied messages. The bot posts no text on deny so "
        "that it cannot be used to enumerate the operator allowlist.",
    )
    slack_allow_dms: bool = Field(
        default=True,
        description="Whether message.im events from allowlisted operators are accepted.",
    )
    slack_thread_history_limit: int = Field(
        default=20,
        ge=0,
        le=200,
        description="How many earlier messages in the thread are read for context. 0 "
        "disables thread reading entirely, restoring the one-message-at-a-time "
        "behaviour. Needs channels:history/groups:history for channels; im:history "
        "already covers DMs.",
    )
    slack_thread_history_char_budget: int = Field(
        default=12_000,
        ge=0,
        description="Cap on total characters of thread history sent to the model. The "
        "oldest messages are dropped first, so the turns nearest the question survive.",
    )

    # ---------------------------------------------------------------- Authz --
    allowlist_path: Path = Field(
        default=Path("/etc/nrp-ops-agent/allowlist.yaml"),
        description="ConfigMap-mounted policy file. Hot-reloaded on mtime change.",
    )
    rate_limit_per_hour: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Per-user token bucket capacity, refilled over one hour.",
    )

    # ------------------------------------------------------------ Kubernetes --
    kube_mode: KubeMode = Field(
        default="auto",
        description="'incluster' uses the ServiceAccount token; 'kubeconfig' uses "
        "KUBECONFIG/~/.kube/config; 'auto' tries in-cluster then falls back.",
    )
    kubeconfig_path: Path | None = Field(
        default=None, description="Overrides KUBECONFIG when kube_mode allows kubeconfig."
    )
    kubeconfig_context: str | None = Field(default=None)
    kube_request_timeout_s: float = Field(default=15.0, gt=0, le=120)

    # ------------------------------------------------------------ Prometheus --
    # NRP runs both thanos.nrp-nautilus.io and prometheus.nrp-nautilus.io. Thanos is
    # the default because it fronts Prometheus with longer retention, which is what
    # "was this broken yesterday too?" questions need; point this at the plain
    # Prometheus if you want the freshest scrape instead. Both hosts are permitted
    # egress in deploy/networkpolicy.yaml.
    # Verified 2026-08-14 from outside the cluster: both hosts answer
    # /api/v1/query with HTTP 200 and no credentials, so the bearer token below
    # stays unset by default. Note Thanos fans in every NRP cluster at once --
    # `ceph_health_status` alone returns nine series, one per Ceph cluster -- so
    # aggregate without a `by (namespace)` and you get a number that describes
    # no single cluster.
    prometheus_base_url: str = Field(default="https://thanos.nrp-nautilus.io")
    prometheus_bearer_token: SecretStr | None = Field(
        default=None, description="Omit for an unauthenticated endpoint."
    )
    prometheus_timeout_s: float = Field(default=10.0, gt=0, le=60)
    prometheus_max_series: int = Field(
        default=200, ge=1, le=5000, description="Response series cap; excess is truncated."
    )

    # ------------------------------------------------------------------- LLM --
    # NRP's OpenAI-compatible gateway. NRP_OPS_LLM_MODEL has no default on purpose:
    # pointing at the wrong model silently is worse than failing to start, and the
    # gateway serves several. List them with GET {llm_base_url}/models.
    # Verified 2026-08-14: GET https://ellm.nrp-nautilus.io/v1/models returns 200
    # unauthenticated and lists 14 models, which confirms the /v1 suffix. POST to
    # /v1/chat/completions is refused with 403 from outside the cluster, so a key
    # (or in-cluster origin) is required -- llm_api_key is not optional in
    # practice. TODO: verify from inside the cluster that the chosen model
    # actually emits OpenAI-style tool_calls; the agent loop depends on it and it
    # could not be exercised from outside. Models offered at time of writing:
    # qwen3, qwen3-small, qwen3-4bit, gpt-oss, gemma, gemma-small, gemma-small-e4b,
    # gemma4-small, gemma4-12b, kimi, minimax-m2, deepseek-v4-flash, glm-5,
    # plus qwen3-embedding (embeddings only -- not a chat model).
    llm_base_url: str = Field(default="https://ellm.nrp-nautilus.io/v1")
    llm_api_key: SecretStr = Field(default=SecretStr(""))
    llm_model: str = Field(default="")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=1, le=32768)
    llm_timeout_s: float = Field(default=90.0, gt=0, le=600)

    # ----------------------------------------------------------------- Agent --
    agent_max_tool_calls: int = Field(default=12, ge=1, le=64)
    agent_max_iterations: int = Field(default=8, ge=1, le=32)
    agent_wall_clock_timeout_s: float = Field(default=120.0, gt=0, le=900)
    agent_tool_output_token_budget: int = Field(
        default=60_000,
        ge=1_000,
        description="Approximate cap on accumulated tool output held in context. When "
        "exceeded the oldest tool results are dropped and the model is told so.",
    )

    # ------------------------------------------------------------------ Docs --
    docs_db_path: Path = Field(default=Path("/var/lib/nrp-ops-agent/docs.sqlite3"))
    docs_repo_url: str = Field(default="https://gitlab.nrp-nautilus.io/prp/nrp-site.git")
    docs_site_base_url: str = Field(default="https://nrp.ai")

    # ----------------------------------------------------------------- Audit --
    audit_stream: Literal["stdout", "stderr"] = Field(default="stdout")

    @field_validator("prometheus_base_url", "llm_base_url", "docs_site_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


# --------------------------------------------------------------------------- #
# Mutable policy: operators, channels, namespaces
# --------------------------------------------------------------------------- #


class Allowlist(BaseModel):
    """The hot-reloadable half of policy, mounted from a ConfigMap.

    Every field is deny-by-default: an empty list means "nothing matches", never
    "everything matches".
    """

    model_config = {"frozen": True, "extra": "forbid"}

    operators: frozenset[str] = frozenset()
    """Slack user IDs (``U...``/``W...``). Never display names or emails, which
    users can change at will."""

    channels: frozenset[str] = frozenset()
    """Slack channel IDs (``C...``/``G...``) where mentions are honoured."""

    namespaces: tuple[str, ...] = ()
    """fnmatch globs, e.g. ``coder``, ``jhub-*``. Checked inside every k8s tool."""

    namespaces_denied: tuple[str, ...] = ()
    """Globs that win over :attr:`namespaces` even on an exact match."""

    @field_validator("operators", "channels", mode="before")
    @classmethod
    def _to_frozenset(cls, v: object) -> object:
        if isinstance(v, list | tuple | set):
            return frozenset(str(x).strip() for x in v if str(x).strip())
        return v

    @field_validator("namespaces", "namespaces_denied", mode="before")
    @classmethod
    def _to_tuple(cls, v: object) -> object:
        if isinstance(v, list | set):
            return tuple(str(x).strip() for x in v if str(x).strip())
        return v

    def namespace_decision(self, namespace: str) -> NamespaceDecision:
        """Deny-by-default namespace check, with *why* attached.

        The reason is not cosmetic: "denied by policy" and "not in scope" are
        different operator actions (remove a denylist entry vs. add an allowlist
        entry), and a single message for both made a self-deny on the bot's own
        namespace read as a missing allowlist entry.
        """
        if not namespace or "/" in namespace or namespace != namespace.strip():
            return "malformed"
        if any(fnmatch.fnmatchcase(namespace, pat) for pat in self.namespaces_denied):
            return "denylisted"
        if any(fnmatch.fnmatchcase(namespace, pat) for pat in self.namespaces):
            return "allowed"
        return "not_allowlisted"

    def namespace_allowed(self, namespace: str) -> bool:
        """Deny-by-default namespace check. Denylist wins over the allowlist."""
        return self.namespace_decision(namespace) == "allowed"


class AllowlistLoader:
    """Reads :class:`Allowlist` from disk, re-reading only when mtime changes.

    A malformed or missing file is *not* fatal and is *not* treated as "allow
    everything": the last good policy is retained, and if there has never been a
    good policy an empty (deny-everything) one is used.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._cached = Allowlist()
        self._mtime: float | None = None
        self._loaded_once = False

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> Allowlist:
        with self._lock:
            try:
                mtime = self._path.stat().st_mtime
            except OSError:
                if not self._loaded_once:
                    self._loaded_once = True
                return self._cached

            if self._loaded_once and mtime == self._mtime:
                return self._cached

            try:
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    raise ValueError("allowlist file must contain a YAML mapping")
                self._cached = Allowlist.model_validate(raw)
                self._mtime = mtime
            except (OSError, ValueError, yaml.YAMLError):
                # Keep the previous good policy; the caller audits the stale read.
                self._mtime = mtime
            self._loaded_once = True
            return self._cached


@lru_cache(maxsize=1)
def get_allowlist_loader() -> AllowlistLoader:
    return AllowlistLoader(get_settings().allowlist_path)
