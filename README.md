# nrp-ops-agent

A Slack bot that lets NRP (National Research Platform / Nautilus) operators diagnose cluster
problems in natural language. An operator writes

> @nrp-ops users are reporting that coder is not responding

and the agent investigates the Kubernetes cluster **read-only**, consults the NRP documentation,
and replies in-thread with a finding, cited evidence, and the exact tool calls it made.

> **Status: Phase 0 complete.** Configuration, audit logging and redaction are implemented and
> tested. The Kubernetes, Prometheus and docs tools, the agent loop and the Slack listener land in
> Phases 1a–1e. See [Build order](#build-order).

---

## Security model

These are hard requirements, enforced in RBAC and in code. None of them is enforced by asking the
language model nicely — any design that depends on the model *choosing* not to do something is a
bug in this repository.

| # | Invariant | Enforced by |
|---|-----------|-------------|
| 1 | No access to `secrets`, `pods/exec`, `pods/portforward`, `pods/attach`, `serviceaccounts/token`; no `create`/`update`/`patch`/`delete`/`deletecollection` anywhere | `deploy/clusterrole.yaml` (Phase 1a) |
| 2 | No generic `kubectl` or shell passthrough — every tool has typed, validated parameters | `tools/` pydantic schemas (Phase 1a) |
| 3 | Pod logs, annotations and event messages are attacker-controlled text and are labelled as untrusted before reaching the model | `agent.py` delimiter wrapping (Phase 1c) |
| 4 | Every tool result and every outbound Slack message passes through redaction | `redact.py` ✅ |
| 5 | Authorization is on Slack **user ID** and **team ID**, never display name or email | `authz.py` (Phase 1d) |
| 6 | Deny by default: unknown user, team, channel, namespace or tool is refused | `config.Allowlist` ✅ / `authz.py` |

### Why redaction matters even though `secrets` is denied

Denying the `secrets` resource in RBAC is necessary but **not sufficient**. `pods/log` is itself a
secret-exfiltration channel, because applications log their own credentials: database URIs with
embedded passwords, `Authorization: Bearer` headers at debug level, ServiceAccount tokens echoed by
a misbehaving client library, `.dockerconfigjson` blobs in an image-pull error. An operator asking a
routine question can therefore pull a live credential into the model context and into a Slack
channel with nobody intending it.

`redact.py` is the mitigation. It runs at two chokepoints — tool output before it enters the model
context, and outbound Slack text before it is posted — and it is regex-based, so it is best-effort.
A bespoke credential format will slip through. That is an argument for keeping the namespace
allowlist tight, not an argument against the layer.

Current rules: JWTs (including ServiceAccount tokens), `Bearer` and `Basic` authorization headers,
AWS access key IDs and secret access keys, PEM private key blocks (RSA/EC/OPENSSH/PKCS8),
`.dockerconfigjson` and Docker `auth` entries, htpasswd hashes, URIs with embedded credentials,
Slack bot/app tokens, GitHub tokens and fine-grained PATs, GitLab tokens, and a deliberately broad
`password=` / `api_key:` assignment catch-all.

Rule *names* that fire are audited. Matched *values* never are.

### Prompt injection

Any Nautilus user can write text into a place this agent reads: a pod log line, a pod annotation, a
Kubernetes event message. That text is hostile input. Tool results are wrapped as

```
<untrusted_tool_output tool="get_logs">…</untrusted_tool_output>
```

with the payload escaped so that a log line containing a literal `</untrusted_tool_output>` cannot
break out of the wrapper. The mitigation that actually matters, though, is capability: even a fully
successful injection can only cause read-only calls, only within allowlisted namespaces, only
through the registered tool set. `tests/test_injection.py` tests the plumbing, not the model's
judgment — that is the point.

---

## Configuration

Everything is environment-driven through `src/nrp_ops_agent/config.py`, prefixed `NRP_OPS_`. No
endpoint, credential, namespace or Slack ID is hardcoded anywhere else in the package.

| Variable | Default | Notes |
|---|---|---|
| `NRP_OPS_SLACK_BOT_TOKEN` | — | `xoxb-…`, from a Secret |
| `NRP_OPS_SLACK_APP_TOKEN` | — | `xapp-…`, Socket Mode |
| `NRP_OPS_SLACK_TEAM_ID` | — | events from any other team are denied |
| `NRP_OPS_SLACK_DENY_REACTION` | `no_entry` | denied messages get an emoji and nothing else |
| `NRP_OPS_SLACK_ALLOW_DMS` | `true` | |
| `NRP_OPS_ALLOWLIST_PATH` | `/etc/nrp-ops-agent/allowlist.yaml` | ConfigMap-mounted, hot-reloaded |
| `NRP_OPS_RATE_LIMIT_PER_HOUR` | `10` | per-user token bucket |
| `NRP_OPS_KUBE_MODE` | `auto` | `auto` \| `incluster` \| `kubeconfig` |
| `NRP_OPS_KUBECONFIG_PATH` / `_CONTEXT` | — | dev only |
| `NRP_OPS_PROMETHEUS_BASE_URL` | `https://thanos.nrp-nautilus.io` | **TODO: verify** |
| `NRP_OPS_PROMETHEUS_BEARER_TOKEN` | — | omit if unauthenticated |
| `NRP_OPS_PROMETHEUS_TIMEOUT_S` / `_MAX_SERIES` | `10` / `200` | |
| `NRP_OPS_LLM_BASE_URL` | `https://llm.nrp-nautilus.io/v1` | **TODO: verify** |
| `NRP_OPS_LLM_API_KEY` / `_MODEL` | — | `_MODEL` has no default on purpose |
| `NRP_OPS_LLM_TEMPERATURE` / `_MAX_TOKENS` | `0.0` / `2048` | |
| `NRP_OPS_AGENT_MAX_TOOL_CALLS` | `12` | per turn |
| `NRP_OPS_AGENT_MAX_ITERATIONS` | `8` | loop iterations |
| `NRP_OPS_AGENT_WALL_CLOCK_TIMEOUT_S` | `120` | |
| `NRP_OPS_AGENT_TOOL_OUTPUT_TOKEN_BUDGET` | `60000` | oldest results dropped past this |
| `NRP_OPS_DOCS_DB_PATH` | `/var/lib/nrp-ops-agent/docs.sqlite3` | on a PVC |
| `NRP_OPS_DOCS_REPO_URL` | `https://gitlab.nrp-nautilus.io/prp/nrp-site.git` | |
| `NRP_OPS_AUDIT_STREAM` | `stdout` | |

### Policy file (hot-reloaded)

Mounted from `deploy/configmap-allowlist.yaml`. Adding an operator is a `kubectl apply`, not a
redeploy: the loader watches mtime and re-reads. A corrupt edit retains the last known-good policy
rather than failing open *or* failing closed.

```yaml
operators:          # Slack user IDs only — never display names or emails
  - U01ABCDEF
channels:           # Slack channel IDs where mentions are honoured
  - C01GHIJKL
namespaces:         # fnmatch globs; deny-by-default, empty means nothing
  - coder
  - "jhub-*"
namespaces_denied:  # wins over `namespaces` even on an exact match
  - kube-system
```

---

## Audit schema

One JSON object per line on stdout. Stable keys; absent fields are omitted rather than emitted as
`null`. This is the compliance evidence trail.

| Key | Type | Present |
|---|---|---|
| `ts` | RFC 3339 UTC, ms precision | always |
| `schema` | int, currently `1` | always |
| `correlation_id` | 16 hex chars, one per Slack request | always (may be `null` outside a request) |
| `event` | see below | always |
| `slack_user`, `slack_team`, `slack_channel`, `thread_ts` | string | Slack-originated events |
| `text` | string | authz events — the raw message text |
| `tool`, `tool_args` | string, object | tool events; args are redacted and truncated at 512 chars |
| `result_bytes` | int | tool events — size only, never the body |
| `latency_ms` | float, 2 dp | timed events |
| `redaction_rules` | array of rule names | `redaction_hit`; never the matched values |
| `model` | string | model events |
| `reason` | string | denials and errors |
| `ok` | bool | outcome-bearing events |

`event` is one of `startup`, `policy_reload`, `authz_allow`, `authz_deny`, `model_call`,
`model_error`, `tool_call`, `tool_error`, `redaction_hit`, `reply_sent`.

Two things are deliberately absent: secret values (tool arguments are scrubbed before emission) and
full log bodies (pod logs are large and attacker-controlled — archiving them would make the audit
stream itself an injection and exfiltration surface).

---

## Slack app setup

Socket Mode only. There is no public ingress and no request-signature endpoint, so the bot needs no
inbound network path at all.

1. Create an app at <https://api.slack.com/apps> → *From scratch*.
2. **Socket Mode** → enable. Generate an app-level token with `connections:write` → `xapp-…`.
3. **OAuth & Permissions** → bot token scopes, and *nothing more*:
   - `app_mentions:read` — see mentions
   - `chat:write` — post and edit replies
   - `im:history` — read DMs the bot is in
   - `reactions:write` — react to denied messages
4. **Event Subscriptions** → subscribe to bot events: `app_mention`, `message.im`.
5. Install to the workspace → `xoxb-…`.
6. Put both tokens in a Kubernetes Secret; the Deployment consumes them via `envFrom`.

Not requested, deliberately: `channels:history`, `groups:history`, `users:read`, `files:read`. The
agent authorizes on user ID and does not need the directory.

---

## Development

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Stack: Python 3.12, `slack-bolt` (Socket Mode), `kubernetes-asyncio`, `fastmcp`, `httpx`,
`pydantic` v2, SQLite FTS5 for the docs index. No agent framework — the tool-calling loop is
hand-rolled against the OpenAI-compatible chat completions API so that it stays auditable.

## Build order

- **Phase 0** — scaffold, `config.py`, `audit.py`, `redact.py`, tests ✅
- **Phase 1a** — `tools/k8s.py`, `deploy/clusterrole.yaml`
- **Phase 1b** — `docs_index/sync.py`, `tools/docs.py`
- **Phase 1c** — `tools/prometheus.py`, `playbooks/`, `agent.py`, `prompts.py`
- **Phase 1d** — `slack_app.py`, `authz.py`, deploy manifests
- **Phase 1e** — evals
- **Phase 2** — write actions behind two-human approval. Not started; requires explicit sign-off.
