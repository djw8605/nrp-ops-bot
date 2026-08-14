# nrp-ops-agent

A Slack bot that lets NRP (National Research Platform / Nautilus) operators diagnose cluster
problems in natural language. An operator writes

> @nrp-ops users are reporting that coder is not responding

and the agent investigates the Kubernetes cluster **read-only**, consults the NRP documentation,
and replies in-thread with a finding, cited evidence, and the exact tool calls it made.

> **Status: Phase 1 complete.** Read-only Kubernetes tools, PromQL, the documentation index, the
> playbooks, the tool-calling loop, the Slack listener, the deploy manifests and the eval harness
> are all implemented and tested (358 tests, `ruff` and `mypy --strict` clean). Phase 2 — write
> actions behind two-human approval — has **not** been started and needs explicit sign-off.
>
> Several values are marked `TODO: verify` because they could not be checked from the build
> environment: see [Before you deploy](#before-you-deploy).

---

## Security model

These are hard requirements, enforced in RBAC and in code. None of them is enforced by asking the
language model nicely — any design that depends on the model *choosing* not to do something is a
bug in this repository.

| # | Invariant | Enforced by |
|---|-----------|-------------|
| 1 | No access to `secrets`, `pods/exec`, `pods/portforward`, `pods/attach`, `serviceaccounts/token`; no `create`/`update`/`patch`/`delete`/`deletecollection` anywhere | `deploy/clusterrole.yaml`, asserted by `tests/test_rbac.py` |
| 2 | No generic `kubectl` or shell passthrough — every tool has typed, validated parameters | `tools/__init__.py` registry + pydantic schemas |
| 3 | Pod logs, annotations and event messages are attacker-controlled text and are labelled as untrusted before reaching the model | `prompts.wrap_untrusted`, applied in `agent.py` |
| 4 | Every tool result and every outbound Slack message passes through redaction | `tools.dispatch` and `slack_app._post`/`_update` |
| 5 | Authorization is on Slack **user ID** and **team ID**, never display name or email | `authz.py` — `SlackEvent` has no name or email field at all |
| 6 | Deny by default: unknown user, team, channel, namespace or tool is refused | `config.Allowlist`, `authz.py`, `tools.dispatch` |

`tests/test_rbac.py` checks both directions: it parses `clusterrole.yaml` and asserts no forbidden
resource or write verb appears, *and* it scans the tool source for every Kubernetes method called
and asserts the role grants each one. Adding `read_namespaced_secret` to a tool fails the suite even
though the manifest is untouched. A live `SelfSubjectAccessReview` check runs with
`NRP_OPS_TEST_LIVE_CLUSTER=1`.

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
judgment — that is the point. Its strongest case scripts a model that *fully complies* with an
injected instruction to read `kube-system`, and asserts the call is refused and no API request is
made.

## Tools

Nine tools, all read-only. Out-of-range arguments are **rejected**, never silently clamped.

| Tool | Notes |
|---|---|
| `list_pods(namespace, label_selector?, field_selector?, limit=50)` | limit ≤ 200 |
| `get_pod_status(namespace, name)` | includes `lastState.terminated` reason and exit code; annotations included and flagged untrusted |
| `get_logs(namespace, pod, container?, tail_lines=200, since_seconds=1800, previous=false)` | `tail_lines` ≤ 1000; total output capped at 96 KB, keeping the tail |
| `get_events(namespace, since_seconds=3600, involved_object?, limit=50)` | newest first |
| `get_workload(kind, namespace, name)` | deployment/statefulset/daemonset/replicaset; reports `generation` vs `observedGeneration` |
| `get_node(name)` | conditions, taints, capacity vs allocatable. Labels deliberately omitted |
| `list_nodes(label_selector?, only_unhealthy=true)` | NRP has hundreds of nodes; healthy ones are never the answer |
| `promql(query, lookback="1h", step?)` | omit `step` for instant, set it for range. Series and sample caps enforced |
| `search_docs(query, limit=5, section="all")` | BM25 over the MDX index; every hit carries its `nrp.ai` URL |

Every returned object passes through `scrub_k8s_fields`, which drops `managedFields`, the
`kubectl.kubernetes.io/last-applied-configuration` annotation, all `env`/`envFrom` and
`imagePullSecrets`, and reduces Secret/ConfigMap references to their name. Handlers also build
summaries field by field rather than dumping objects, so the scrubber is a second line of defence
rather than the only one.

There is no ConfigMap tool and no Secret tool, and `configmaps` is absent from the ClusterRole.

## Documentation index

The docs site is Astro Starlight. `docs_index/sync.py` clones
`gitlab.nrp-nautilus.io/prp/nrp-site` and indexes the MDX under `src/content/docs` — it does **not**
crawl the rendered HTML, which loses heading structure and reflows code blocks.

- Chunked by heading, keeping the full heading path (`Rolling upgrade > Rolling back`)
- MDX/JSX component tags stripped; fenced code blocks preserved byte-identical, because operators
  search on exact identifiers
- FTS5 tokenizer keeps `nvidia.com/gpu` and `rook-ceph-mgr` as single terms
- Short subsections are deliberately *not* merged into their predecessor: folding a two-line
  "Rolling back" section under the previous heading would make the citation point at the wrong part
  of the page
- Each chunk maps to its published URL plus a heading anchor; `admindocs` and `userdocs` are both
  indexed, and the admin guide is the one that holds the actual runbooks
- Built atomically into `<db>.building` and renamed on success, so the agent never reads a
  half-built index. A checkout yielding zero chunks is a failure, not an empty index

`Retriever` is the seam for embeddings later. Phase 1 is BM25 only — exact-term matching beats pure
vector search on a corpus this full of identifiers. Retrieval quality is pinned by ten operator-style
queries in `tests/test_docs_index.py`.

## Playbooks

`playbooks/*.yaml` give each service a reproducible triage order, injected into the system prompt as
a suggestion with explicit permission to deviate. They also carry known-good PromQL, because the
model cannot enumerate this cluster's metrics and asking it to invent queries produces confident
nonsense.

`tests/test_playbooks.py` validates every step against the live tool registry — a step naming a
tool that does not exist, or passing arguments that tool would reject, fails the suite. It also
asserts that **every unverified PromQL query carries a `TODO: verify` note**, so a reader never
mistakes a guess for a checked query.

## Agent loop

Hand-rolled in `agent.py`, roughly 300 lines, no agent framework. Budgets, all env-driven: 12 tool
calls per turn, 8 loop iterations, 120 s wall clock, 60k tokens of accumulated tool output. When the
output cap is hit the oldest results are dropped **and the model is told they were dropped**, so it
does not silently reason from a truncated picture. Model faults, tool faults and timeouts all become
reported outcomes rather than exceptions.

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
| `NRP_OPS_PROMETHEUS_BASE_URL` | `https://thanos.nrp-nautilus.io` | or `https://prometheus.nrp-nautilus.io` |
| `NRP_OPS_PROMETHEUS_BEARER_TOKEN` | — | omit if unauthenticated |
| `NRP_OPS_PROMETHEUS_TIMEOUT_S` / `_MAX_SERIES` | `10` / `200` | |
| `NRP_OPS_LLM_BASE_URL` | `https://ellm.nrp-nautilus.io/v1` | **TODO: verify** the `/v1` suffix |
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

## Before you deploy

These could not be verified from the build environment and are marked `TODO: verify` in the source:

1. **`NRP_OPS_LLM_MODEL` has no default** and the pod refuses to start without it. List the
   available models with `curl https://ellm.nrp-nautilus.io/v1/models`. Confirm the chosen model
   supports OpenAI-style `tool_calls` — the loop depends on it.
2. **The `/v1` suffix** on the LLM base URL, and whether the gateway needs an API key.
3. **Prometheus auth** — whether Thanos or Prometheus needs a bearer token from inside the cluster.
4. **PromQL in the playbooks.** Every query the live cluster has not confirmed is marked
   `TODO: verify`; `ceph_*`, `kube_pod_resource_requests` and the nginx ingress label names are the
   likeliest to differ. Metric names are the one thing a model will invent convincingly.
5. **Slack team ID and operator/channel IDs** in `deploy/configmap-allowlist.yaml`, and the
   `NRP_OPS_SLACK_TEAM_ID` env in `deploy/deployment.yaml`.
6. **`deploy/networkpolicy.yaml`** — the API server CIDR, and the fact that the fallback egress rule
   permits any public HTTPS destination. A CiliumNetworkPolicy with FQDN rules is sketched in that
   file and is the version you want if Cilium is the CNI.
7. **Starlight slug casing** in `url_for()` — the whole path is lowercased, which matches the known
   `Documentation/` → `/documentation/` mapping but would be wrong for a genuinely mixed-case
   filename. Spot-check a few generated URLs against the live site.
8. **Storage class** for the docs PVC (`rook-cephfs` assumed) and the workload names in
   `jupyterhub.yaml` and `ceph.yaml`.

Deploy order:

```bash
kubectl apply -f deploy/namespace.yaml
kubectl -n nrp-ops-agent create secret generic nrp-ops-agent-tokens \
  --from-literal=NRP_OPS_SLACK_BOT_TOKEN=xoxb-... \
  --from-literal=NRP_OPS_SLACK_APP_TOKEN=xapp-... \
  --from-literal=NRP_OPS_LLM_API_KEY=...
kubectl apply -f deploy/serviceaccount.yaml -f deploy/clusterrole.yaml \
               -f deploy/clusterrolebinding.yaml -f deploy/configmap-allowlist.yaml
kubectl apply -f deploy/deployment.yaml -f deploy/networkpolicy.yaml
kubectl apply -f deploy/cronjob-docs-sync.yaml
kubectl -n nrp-ops-agent create job --from=cronjob/nrp-ops-agent-docs-sync docs-sync-initial
```

Confirm the RBAC does what it claims, against the real cluster:

```bash
kubectl auth can-i get secrets --as=system:serviceaccount:nrp-ops-agent:nrp-ops-agent   # want: no
kubectl auth can-i create pods/exec --as=system:serviceaccount:nrp-ops-agent:nrp-ops-agent  # no
kubectl auth can-i list pods --as=system:serviceaccount:nrp-ops-agent:nrp-ops-agent      # want: yes
```

## Evals

`evals/cases.yaml` holds 20 operator questions with expected namespaces, workloads, findings and
tool sets. `run_evals.py` scores each run on namespace accuracy, workload identification, finding
match and tool selection, and reports mean tool count and latency.

```bash
uv run python evals/run_evals.py --dry-run              # validate cases, call nothing
uv run python evals/run_evals.py --model <model-name>   # live cluster + live model
uv run python evals/run_evals.py --model <a> --json a.json
```

This is how you decide whether a self-hosted open-weight model is good enough for the multi-step
cases or whether they need a frontier model: run the same suite against both and compare. Tool count
is reported rather than scored — a right answer in six calls beats a wrong one in two.

The expected findings are written against plausible cluster states, not recorded incidents. Replace
them with real outages as they happen; that is what makes the suite worth running.

## MCP server

`python -m nrp_ops_agent.mcp_server` exposes the same nine tools over MCP. It is a second *front
end*, not a second tool surface: every call goes through the same dispatcher, so validation, the
namespace allowlist, redaction and the audit trail all apply, and the advertised schemas are
generated from the same pydantic models. MCP clients bypass the Slack authorization gate, so only
expose it to people already trusted with cluster read access.

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
- **Phase 1a** — `tools/k8s.py`, `deploy/clusterrole.yaml` ✅
- **Phase 1b** — `docs_index/sync.py`, `tools/docs.py` ✅
- **Phase 1c** — `tools/prometheus.py`, `playbooks/`, `agent.py`, `prompts.py` ✅
- **Phase 1d** — `slack_app.py`, `authz.py`, deploy manifests ✅
- **Phase 1e** — evals ✅
- **Phase 2** — write actions behind two-human approval: the agent never acts, it posts a proposal
  with interactive buttons, a *second* allowlisted operator approves, and a separate executor
  process with its own narrowly scoped ServiceAccount performs the single named action, with the
  model entirely out of the privilege path. **Not started — requires explicit sign-off.**

What Phase 1 has *not* been through: an end-to-end run against the live cluster, a live model, or a
real Slack workspace. Every component is tested against mocks and fakes; none of it has met
production.
