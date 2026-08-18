# nrp-ops-agent

A Slack bot that lets NRP (National Research Platform / Nautilus) operators diagnose cluster
problems in natural language. An operator writes

> @nrp-ops users are reporting that coder is not responding

and the agent investigates the Kubernetes cluster **read-only**, consults the NRP documentation,
and replies in-thread with a finding, cited evidence, and the exact tool calls it made.

> **Status: Phase 1 complete.** Read-only Kubernetes tools, PromQL, the documentation index, the
> playbooks, the tool-calling loop, the Slack listener, the deploy manifests and the eval harness
> are all implemented and tested (541 tests, `ruff` and `mypy --strict` clean). Phase 2 — write
> actions behind two-human approval — has **not** been started and needs explicit sign-off.
>
> The cluster-specific assumptions have since been checked against live Thanos, live
> kube-state-metrics and the live docs site, and **seven of them were wrong** — including a namespace
> allowlist that would have silently refused every Ceph and JupyterHub question. Those are fixed.
>
> **The loop has now run end to end against the live cluster and a live model.**
> `deepseek-v4-flash` on `ellm.nrp-nautilus.io` emits OpenAI-style `tool_calls` correctly; a
> question about failing pods in `coder` resolved in 4 tool calls to a correct, evidenced answer.
> See [Testing it locally](#testing-it-locally).
>
> Slack tokens are sealed into `deploy/sealedsecret-tokens.yaml` and verified against `auth.test`
> (workspace **NRP**, bot `nrp_ops_bot`). Still untested: the bot answering a real mention.

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

Sixteen tools, all read-only. Out-of-range arguments are **rejected**, never silently clamped.

| Tool | Notes |
|---|---|
| `list_pods(namespace, label_selector?, field_selector?, limit=50)` | limit ≤ 200 |
| `get_pod_status(namespace, name)` | includes `lastState.terminated` reason and exit code; annotations included and flagged untrusted |
| `get_logs(namespace, pod, container?, tail_lines=200, since_seconds=1800, previous=false)` | `tail_lines` ≤ 1000; total output capped at 96 KB, keeping the tail |
| `get_events(namespace, since_seconds=3600, involved_object?, limit=50)` | newest first |
| `get_workload(kind, namespace, name)` | deployment/statefulset/daemonset/replicaset; reports `generation` vs `observedGeneration` |
| `get_cronjob(namespace, name, job_limit=5)` | schedule, last success, and recent Jobs with their failure conditions and `restartPolicy` |
| `get_node(name)` | conditions, taints, capacity vs allocatable. Labels deliberately omitted |
| `list_nodes(label_selector?, only_unhealthy=true)` | NRP has hundreds of nodes; healthy ones are never the answer |
| `promql(query, lookback="1h", step?)` | omit `step` for instant, set it for range. Series and sample caps enforced |
| `search_docs(query, limit=5, section="all")` | BM25 over the MDX index; every hit carries its `nrp.ai` URL |
| `accounting_discover(dimension?, resource?, dates?, prefix?, limit=50)` | latest ingested date, plus the namespace / institution / node / GPU-model values actually observed |
| `accounting_top(dimension, resource, dates?, filters?, limit=10)` | ranks namespaces, institutions, nodes or the commercial flag by one resource |
| `accounting_trend(dimension, value, resource, dates?)` | daily usage for one namespace/institution/node; defaults to 30 days |
| `accounting_usage(group_by=[…], filters?, limit=100)` | the escape hatch: any group-by and filter combination |
| `accounting_llm_usage(group_by=[…], filters?, limit=100)` | LLM tokens by namespace, served model, alias and token type |
| `accounting_chart(kind, resource, dimension?, …)` | queries **and** draws: `trend`, `ranking` or `breakdown`, attached to the reply as a PNG |

Every returned object passes through `scrub_k8s_fields`, which drops `managedFields`, the
`kubectl.kubernetes.io/last-applied-configuration` annotation, all `env`/`envFrom` and
`imagePullSecrets`, and reduces Secret/ConfigMap references to their name. Handlers also build
summaries field by field rather than dumping objects, so the scrubber is a second line of defence
rather than the only one.

There is no ConfigMap tool and no Secret tool, and `configmaps` is absent from the ClusterRole.

## Accounting and charts

"Who is burning the A100s", "how much did that namespace use last quarter" and "is our GPU
demand still growing" are not cluster questions and Prometheus answers them badly: retention is
short, and `kube_pod_container_resource_*` records what was *requested*, not what was charged.

Those go to the accounting MCP server from
[`djw8605/nrp-clickhouse`](https://github.com/djw8605/nrp-clickhouse), which serves daily
per-namespace and per-pod usage rolled up from Prometheus and the NRP portal. In production that is
`https://nrp-accounting-mcp.nrp-nautilus.io/` (`NRP_OPS_ACCOUNTING_MCP_URL`).

- **The transport is one POST.** That server runs FastMCP stateless with JSON responses, so a call
  needs no `initialize` handshake and no session — which is why
  [`tools/accounting.py`](src/nrp_ops_agent/tools/accounting.py) speaks JSON-RPC directly instead
  of adding an MCP client dependency. `Accept` must list **both** `application/json` and
  `text/event-stream` or the transport answers 406 regardless of the server's response mode.
- **Five of its fourteen tools are exposed**, each behind a pydantic model with closed literals for
  resource and dimension. The endpoint is unauthenticated by design; what bounds this bot's reach
  into it is the registry, not the network.
- **Not gated on the namespace allowlist**, unlike the Kubernetes tools. That allowlist bounds what
  can be read *out of* the cluster — logs, pod specs, events — where a bad namespace means someone
  else's credential in a log line. Accounting rows are aggregates that already leave NRP for XDMoD,
  and "top ten GPU consumers" is cluster-wide by construction: an allowlist would answer it wrongly
  rather than refusing it.
- **Results are flagged untrusted.** Namespace, node and PI names are user-chosen, so
  `accounting_*` output is wrapped like any other tool output.

### Charts

`accounting_chart` runs the query itself, renders a PNG, and returns the numbers it plotted.
Call it *instead of* the matching query tool, not after it.

| `kind` | Form | Answers |
|---|---|---|
| `trend` | line with a 10% wash, end-labelled | "when did their GPU usage jump?" |
| `ranking` | horizontal bars, largest first, value at each tip | "who are the top consumers?" |
| `breakdown` | stacked areas, tail folded into *Other* | "how does the total split, and is that shifting?" |

Bars and bands split by `namespace`, `institution`, `node` or `commercial`, and — with
`resource='llm'` — by `model`, `token_alias` or `token_type`. Those last three live only in the LLM
token table, which is keyed differently (`tokens_used`, no `unit` column) and holds no node or GPU
columns, so `accounting_chart` reads them from `query_llm_token_usage` and refuses the filters that
table cannot apply rather than dropping them silently. Without them, "plot the most used LLM models
by day" had no answerable form at all: `dimension='model'` failed validation, and the model retried
it rather than finding another way.

A `breakdown` sizes its row request to the window it was asked for. The row cap is a *total* and a
daily breakdown multiplies days by values, so a quarter-long question used to come back as its most
recent six weeks — under a subtitle still naming the quarter, which inverted the ranking it was drawn
to show. The chart is now captioned with the days it actually has, and a short row set reports
`window_requested` so the reply can say what it does not cover.

The image travels **out of band**: a context-local collector in
[`charts.py`](src/nrp_ops_agent/charts.py) carries it from the tool to `AgentResult.charts`, and
`slack_app._upload_charts` posts it into the thread after the text. Image bytes never enter the
model's context — it sees only the numbers, and the prompt tells it to write the finding from those
and never to describe a picture it cannot see.

Three things follow from a PNG being unscrubbable:

1. **Labels are redacted before they are drawn.** `redact.py` cannot reach rasterised text, so
   `charts.clean_label` runs on every namespace name, date and title on the way *into* matplotlib.
   The title and alt text that ride beside the file go through the outbound chokepoint as well.
2. **The upload cannot cost the answer.** It happens after `chat.update`, and a failure — a
   workspace installed without `files:write`, most likely — is audited as `chart_upload ok=false`
   and otherwise swallowed. Set `NRP_OPS_CHARTS_ENABLED=false` to stop drawing them entirely.
3. **Two charts per reply** (`NRP_OPS_CHARTS_MAX_PER_TURN`). Past the budget the tool *refuses*
   rather than silently dropping, so the model never promises a picture that will not appear.

Style is one validated palette across every chart, so two in a thread read as one system: six
categorical hues assigned by slot and never cycled (worst adjacent pair 9.1 ΔE under simulated
colour-vision deficiency, 19.6 unsimulated), one blue for rankings because length already carries
magnitude, a legend whenever there is more than one series, and one number format per chart chosen
from its peak — `184.3k` above `9.1k`, never `15.8k` above `9,120`.

`nrp-ops-ask` writes any charts it drew to `--charts-dir` (default `./charts`) so they can be
looked at from a terminal.

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
- Rebuilt hourly by the `docs-sync` sidecar in `deploy/deployment.yaml` (`nrp-ops-docs-sync
  --every 3600`), so the worst case is a runbook edit the bot cites in its pre-edit form for under
  an hour. A rebuild is a shallow clone and a full rebuild — about a minute — so the interval is
  bounded by politeness to the docs repo, not by cost
- The index lives on an `emptyDir` shared by the two containers in the pod, so it is rebuilt on
  every pod start and does not survive one. That is deliberate: it was a CronJob writing to an RWX
  CephFS PVC until 2026-08-15, and it never produced an index once — the CSI driver did not apply
  the pod's `fsGroup`, the volume root stayed `root:root 0755`, and nothing running as uid 65532
  could fix that from inside the cluster. A minute of rebuild on restart buys the whole failure
  mode away

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
asserts that **every PromQL query states whether it was verified against the live cluster**, so a
reader never mistakes a guess for a checked query, and that **every namespace a playbook names is
permitted by the shipped allowlist** — the check that catches a runbook pointing at a namespace the
tool would refuse.

## Agent loop

Hand-rolled in `agent.py`, around 600 lines, no agent framework. Budgets, all env-driven: 12 tool
calls per turn, 8 loop iterations, 120 s wall clock, 60k tokens of accumulated tool output. When the
output cap is hit the oldest results are dropped **and the model is told they were dropped**, so it
does not silently reason from a truncated picture. Model faults, tool faults and timeouts all become
reported outcomes rather than exceptions.

### An answer cut off at the output limit

`finish_reason: length` means the endpoint stopped the model mid-sentence, and it is the one signal
that separates a finished finding from half of one. The loop reads it: the partial answer is
discarded — nothing has been posted yet, since the ack message is edited only once the turn ends —
and asked for again, shorter. Discarded rather than continued, because stitching a second generation
onto a sentence that stopped mid-word leaves a seam in the middle of the finding.

If the retry is cut off too, what was written is posted with `stopped: truncated` in the footnote and
`ok: false` in the audit. The failure that made this necessary was quieter than that: a reply ending
mid-word read as a complete answer, and an operator asking it to continue got a fresh investigation
of the same question.

A reasoning model can also spend its whole budget in `reasoning_content` and return empty `content`.
That is `empty_answer`, not `answered` — the model produced nothing, and the reply says so instead of
claiming an answer with no text in it.

`NRP_OPS_LLM_MAX_TOKENS` is the served model's own documented ceiling (100k for deepseek-v4, which
carries a 1M-token context), not a house style. It is a ceiling and not a target: what actually
bounds a long generation is `NRP_OPS_LLM_TIMEOUT_S` (90 s) and the 120 s wall clock, and the Slack
reply is capped at 3500 characters regardless. Raising it buys headroom for reasoning tokens, not
longer replies.

### A follow-up does not re-run the investigation

Thread history replays what was *said* — the bot's own text included — but not the tool results
behind it, so a second turn in a thread had the trace of its first turn with none of the data. The
answer was to query everything again. The last turn's tool exchange is carried per thread instead,
keyed by channel and parent timestamp, and replayed with a note saying it is from the previous turn
and may be stale.

In-process and deliberately not persisted: the Deployment runs one replica, a restart costs one
re-query, and a thread's results have no business outliving the pod. Bounded three ways — 20k
characters, 30 minutes, 32 threads — because a cache in a process that runs for weeks needs a lid,
and because replaying a half-hour-old pod list as though it were current is worse than looking again.
Carried results keep the `<untrusted_tool_output>` wrapper they were first stored with.

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
| `NRP_OPS_SLACK_THREAD_HISTORY_LIMIT` | `20` | earlier thread messages read for context; `0` disables |
| `NRP_OPS_SLACK_THREAD_HISTORY_CHAR_BUDGET` | `12000` | oldest thread turns dropped past this |
| `NRP_OPS_ALLOWLIST_PATH` | `/etc/nrp-ops-agent/allowlist.yaml` | ConfigMap-mounted, hot-reloaded |
| `NRP_OPS_RATE_LIMIT_PER_HOUR` | `10` | per-user token bucket |
| `NRP_OPS_KUBE_MODE` | `auto` | `auto` \| `incluster` \| `kubeconfig` |
| `NRP_OPS_KUBECONFIG_PATH` / `_CONTEXT` | — | dev only |
| `NRP_OPS_PROMETHEUS_BASE_URL` | `https://thanos.nrp-nautilus.io` | or `https://prometheus.nrp-nautilus.io` |
| `NRP_OPS_PROMETHEUS_BEARER_TOKEN` | — | verified unauthenticated; leave unset |
| `NRP_OPS_PROMETHEUS_TIMEOUT_S` / `_MAX_SERIES` | `10` / `200` | |
| `NRP_OPS_ACCOUNTING_MCP_URL` | `https://nrp-accounting-mcp.nrp-nautilus.io/` | whole endpoint, not a base path; empty disables the tools |
| `NRP_OPS_ACCOUNTING_TIMEOUT_S` | `20` | kept under the loop's 30s dispatch cap so a chart still has time to render |
| `NRP_OPS_ACCOUNTING_MAX_ROWS` | `500` | rows kept per response; excess truncated and flagged |
| `NRP_OPS_CHARTS_ENABLED` | `true` | needs the Slack `files:write` scope |
| `NRP_OPS_CHARTS_MAX_PER_TURN` | `2` | past this the chart tool refuses rather than dropping |
| `NRP_OPS_LLM_BASE_URL` | `https://ellm.nrp-nautilus.io/v1` | `/v1` verified; POST needs a key |
| `NRP_OPS_LLM_API_KEY` / `_MODEL` | — | `_MODEL` has no default on purpose |
| `NRP_OPS_LLM_TEMPERATURE` / `_MAX_TOKENS` | `0.0` / `100000` | the served model's own output ceiling; see below |
| `NRP_OPS_AGENT_MAX_TOOL_CALLS` | `12` | per turn |
| `NRP_OPS_AGENT_MAX_ITERATIONS` | `8` | loop iterations |
| `NRP_OPS_AGENT_MAX_TRUNCATION_RETRIES` | `1` | re-asks for a shorter answer when one is cut off; `0` posts the partial, marked |
| `NRP_OPS_AGENT_WALL_CLOCK_TIMEOUT_S` | `120` | the real bound on generation once `_MAX_TOKENS` is this high |
| `NRP_OPS_AGENT_TOOL_OUTPUT_TOKEN_BUDGET` | `60000` | oldest results dropped past this |
| `NRP_OPS_AGENT_THREAD_MEMO_CHAR_BUDGET` | `20000` | previous turn's tool results carried into a follow-up; `0` disables |
| `NRP_OPS_AGENT_THREAD_MEMO_TTL_S` | `1800` | past this a follow-up starts clean |
| `NRP_OPS_AGENT_THREAD_MEMO_MAX_THREADS` | `32` | threads held; least recently answered evicted first |
| `NRP_OPS_DOCS_DB_PATH` | `/var/lib/nrp-ops-agent/docs.sqlite3` | emptyDir shared with the sidecar |
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

A refused namespace says which of the two gates stopped it, because the fixes differ: *"explicitly
denied by policy"* means an entry in `namespaces_denied` matched and needs removing there, while
*"not in the configured allowlist"* means nothing in `namespaces` matched and needs adding. With the
shipped `namespaces: ["*"]`, only the first can happen.

The bot's own namespace, `system-nrp-ops-bot`, is readable — it was denied until 2026-08-14, which
made self-triage ("why am I crashlooping?") fail with a message that read like an out-of-scope
tenant. That means the agent can read its own pod logs (the audit trail, before `redact.py` scrubs
it on the way to Slack) and its own ConfigMap (the operator and channel IDs). Both stay behind
`operators`; re-add the namespace to `namespaces_denied` to reverse it.

#### Restricting the bot to specific people

`operators` is the control that does this. It is checked **before** the channel list and applies to
DMs and channel mentions alike, so a non-operator is refused even inside an allowlisted channel.
The order in [`authz.py`](src/nrp_ops_agent/authz.py) is: bot message → empty → wrong team →
**not an operator** → channel/DM → rate limit.

| Message | Allowed when |
|---|---|
| `@nrp-ops` in a channel | user ∈ `operators` **and** channel ∈ `channels` |
| DM to the bot | user ∈ `operators` **and** `NRP_OPS_SLACK_ALLOW_DMS` (default `true`) |

Set `NRP_OPS_SLACK_ALLOW_DMS=false` to force every request into an audited channel.

**Getting the IDs.** The bot cannot look them up — `users:read` and `channels:read` were
deliberately not requested, so it has no directory access. From the Slack UI:

- **User ID** (`U…`, or `W…` on Enterprise Grid) — click the person → *View full profile* → `…`
  (More) → *Copy member ID*.
- **Channel ID** (`C…`, or `G…` for older private channels) — click the channel name → *About* →
  bottom of the panel. Or copy a message link: it is the `/archives/C…` segment.

Always IDs, never names or emails: display names are mutable, and `SlackEvent` has no field for
either.

**Two failure modes that look like nothing happening:**

1. **Placeholder IDs behave exactly like an empty list.** The bot connects, logs `authz_deny`, and
   answers nobody. Shipping with `U00000000000` is a working deployment that ignores everyone.
2. **Listing a channel does not put the bot in it.** Slack only delivers `app_mention` for channels
   the bot has joined, so run `/invite @nrp-ops` there too — otherwise the event never arrives and
   the audit log shows *nothing at all*, not even a denial. That distinction is the fastest way to
   tell the two apart: a denial means the allowlist; silence means the bot is not in the channel.

Editing is a `kubectl apply`, not a redeploy — the loader watches mtime. Allow up to a minute for
kubelet to project the updated ConfigMap into the pod.

Each operator gets `NRP_OPS_RATE_LIMIT_PER_HOUR` (default 10) investigations per hour, tracked per
user ID.

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
`model_error`, `model_truncated`, `tool_call`, `tool_error`, `redaction_hit`, `thread_history`,
`chart_upload`, `reply_sent`.

`model_call` carries `finish_reason` alongside its token counts. `length` there is the endpoint
saying the answer was cut off, and `model_truncated` follows when that answer was discarded and
re-asked; a `reply_sent` with `reason: truncated` is one that survived the retry and reached Slack
marked as incomplete.

`chart_upload` records one attempted image upload (`filename`, plus `result_bytes` on success).
`ok: false` there is almost always a workspace installed without `files:write`; the reply itself
was already posted, so this is the only place that failure shows up. `reply_sent` carries a
`charts` count alongside `tool_calls` and `iterations`.

`thread_history` records how much prior conversation was read for a turn (`messages`, and
`untrusted` for the count written by non-operators). `ok: false` on it means the read failed —
almost always a missing history scope — and the turn proceeded without context.

Two things are deliberately absent: secret values (tool arguments are scrubbed before emission) and
full log bodies (pod logs are large and attacker-controlled — archiving them would make the audit
stream itself an injection and exfiltration surface).

---

## Slack app setup

Socket Mode only. There is no public ingress and no request-signature endpoint, so the bot needs no
inbound network path at all.

Fastest path — <https://api.slack.com/apps> → *Create New App* → **From an app manifest**, pick the
NRP workspace, and paste [`slack/app-manifest.yaml`](slack/app-manifest.yaml). That sets the scopes,
the event subscriptions and Socket Mode in one step.

Then collect the two tokens, **which live on two different pages**:

| Token | Where | Notes |
|---|---|---|
| `xapp-…` | **Basic Information → App-Level Tokens** → *Generate Token and Scopes*, scope `connections:write` | A manifest cannot mint tokens, so this step is always manual |
| `xoxb-…` | *Install to Workspace*, then **OAuth & Permissions → Bot User OAuth Token** | Does not exist until the app is installed |

> **If you cannot find the app-level token:** it is not under *OAuth & Permissions*, which is where
> every other Slack token lives — that is the usual reason it seems missing. It is on **Basic
> Information**, below *App Credentials*, and it does not exist until you generate one. Toggling
> **Settings → Socket Mode** on also offers to create it inline, which is the easier route. You can
> reopen the value later by clicking the token's name; it is not show-once.

Finally you need the workspace **team ID** (`T…`) for `NRP_OPS_SLACK_TEAM_ID`, and the operators'
Slack **user IDs** for `deploy/configmap-allowlist.yaml` — profile → `…` → *Copy member ID*. Both
tokens go into the sealed Secret; the Deployment consumes them via `envFrom`.

Doing it by hand instead: *From scratch*, then Socket Mode → enable and generate the app token;
**OAuth & Permissions** → bot scopes `app_mentions:read`, `chat:write`, `im:history`,
`reactions:write`, `channels:history`, `groups:history`, `files:write` and nothing more;
**Event Subscriptions** → bot events `app_mention` and `message.im`; install to the workspace.

### Thread context needs a reinstall

The bot reads the thread it is mentioned in so a follow-up like *"is it still broken?"* resolves
against what was already said. That calls `conversations.replies`, which needs a history scope for
the conversation type: `im:history` (already granted) covers DMs, `channels:history` and
`groups:history` cover public and private channels.

**Editing the manifest does not grant a scope to an app that is already installed.** Apply the
manifest at *App Manifest*, then **reinstall** from *Install App* — Slack will show the added
permissions and re-issue the `xoxb-…` token. Put the new token in the sealed Secret if it changed.

Until that happens the bot keeps working: `conversations.replies` fails with `missing_scope` in
channels, the failure is audited as `thread_history` with `ok: false`, and the turn is answered
without context — exactly the behaviour that made threads feel forgetful in the first place. Threads
in DMs gain context immediately, since `im:history` is already held.

`files:write` is in the same boat and fails the same way. Without it `files_upload_v2` raises
`missing_scope`, the failure is audited as `chart_upload` with `ok: false`, and the reply — already
posted — stands without its chart. Reinstall to grant it, or set `NRP_OPS_CHARTS_ENABLED=false` so
the agent stops drawing charts it cannot deliver.

Non-operators can reply inside a thread an operator started. Their messages are included as context
but wrapped in `<untrusted_thread_message>`, the same envelope used for pod logs, so they read as
data and never as instructions — the operator-only boundary in `authz.py` is not weakened by
reading the thread. Set `NRP_OPS_SLACK_THREAD_HISTORY_LIMIT=0` to turn thread reading off entirely.

Not requested, deliberately: `users:read`, `channels:read`, `files:read`. The agent authorizes on
user ID and does not need the directory, and it uploads charts rather than reading anyone else's
files. (`channels:history` and `groups:history` *are* requested — see above; this line predated
thread reading and named them by mistake.)

### Replies are converted to Slack mrkdwn

Slack does not render Markdown. Its `text` field takes *mrkdwn*, a much smaller syntax that overlaps
just enough to be misleading: `**bold**` arrives as literal asterisks, `## Heading` as a literal
hash, and `[the runbook](https://nrp.ai/x)` as literal brackets — so the doc links, the part an
operator is meant to click to check a finding, arrive unclickable.

Two things address this, and both are needed. The system prompt asks the model for mrkdwn, which
gets most of the way there; `slack_format.py` then converts whatever actually came back, because
models drift into Markdown on long answers and bulleted lists are exactly where they drift. The
conversion is idempotent, which is what lets both run without fighting each other.

| Markdown | Becomes |
|---|---|
| `**bold**`, `__bold__` | `*bold*` |
| `~~strike~~` | `~strike~` |
| `# Heading` | a bold line — mrkdwn has no headings |
| `- item` | `•` / `◦` / `▪` by depth |
| `[text](url)` | `<url\|text>` |
| a pipe table | a code block — mrkdwn has no tables, monospace is the only alignment there is |
| `---` | a rule of box-drawing characters |

A single `*text*` is left exactly as written. Markdown reads it as italic and Slack reads it as
bold, so no rendering satisfies both; since the prompt asks for mrkdwn, most of those runs were
meant as Slack bold, and rewriting them would demote the words the model chose to emphasise.

Code is why this is a segmenting parser and not a pile of regex substitutions. Operators get YAML,
PromQL and log lines back, and those are full of `*`, `_` and `|` — rewriting emphasis inside a
fenced block would corrupt the evidence the answer rests on. Fenced blocks and inline spans are
lifted out before any prose rule runs and put back afterwards, changed only by the `&`/`<`/`>`
escaping Slack requires everywhere. The tool trace is escaped the same way and never converted, so a
selector in it stays byte-identical to what an operator would re-run by hand.

Truncation to the 3500-character cap is fence-aware for the same reason: cutting mid-block would
leave an unterminated ` ``` `, and Slack renders that as one code block swallowing the rest of the
message — turning a merely-truncated answer into an unreadable one.

---

## Before you deploy

Most of the original unknowns were checked on 2026-08-14 against live Thanos, live
kube-state-metrics, the live docs site and the live model gateway. **Six of them were wrong**, and
the corrections are in the source. What that pass found, and what is still open:

### Corrected (was wrong, now verified)

1. **The namespace allowlist denied nearly everything it was meant to allow.** It granted
   `jupyterhub`, `rook-ceph` and `nvidia-device-plugin` — *none of which exist on this cluster*.
   Because the namespace check runs inside each tool before any API call, every Ceph and JupyterHub
   question would have been refused silently, which reads like a healthy "nothing found" rather
   than a misconfiguration. Real: `coder`, `coder-dev`, `coder-user-dev`, `jupyterlab`,
   `jupyterlab-east`, `rook`/`rook-*`, `gpu-operator`.
2. **NRP runs nine independent Ceph clusters**, one per namespace (`rook`, `rook-central`,
   `rook-east`, `rook-ucsd`, `rook-tide`, `rook-south-east`, `rook-pacific`, `rook-haosu`,
   `rook-fullerton`). "Is Ceph healthy?" has nine answers; every Ceph query is now grouped
   `by (namespace)` and every answer must name the cluster.
3. **JupyterHub lives in `jupyterlab`, not `jupyterhub`.** The deployments really are `hub` and
   `proxy`, but the hub is a Deployment, so its pod is `hub-<generated>` — the assumed `hub-0` was
   never going to resolve.
4. **`kube_pod_resource_requests` does not exist** (it is `kube_pod_container_resource_requests`),
   and it must be restricted to Running pods. Unfiltered it counted 2168 GPU requests against 1133
   allocatable and reported a *negative* number of free GPUs. Correct answer at the time of
   checking: 674 in use of 1133, **459 free**.
5. **There is no nginx ingress.** No `nginx_ingress_controller_*` metric exists; NRP fronts
   services with haproxy, backends named `<namespace>_<service>_<port>` (e.g. `coder_coder_http`).
   The old query returned an empty result rather than an error — indistinguishable from "no errors".
6. **The docs PVC storage class `rook-cephfs` does not exist.** The RWX CephFS class on this
   cluster is called `cephfs`. The PVC would have stayed Pending forever, leaving the agent
   with no docs index. Two rounds of chasing that claim later — `cephfs` was the wrong Ceph
   cluster, then `rook-cephfs-central` bound but was unwritable, its root `root:root 0755`
   because the CSI driver never applied `fsGroup` — **the PVC is gone entirely**. The index is
   an `emptyDir` rebuilt by a sidecar. Persisting a file that a one-minute `git clone`
   reconstructs was never worth a storage dependency this cluster could not satisfy.
7. **`ceph_pool_percent_used` is a fraction in 0..1, not a percent.** The fullest pool reads `0.84`,
   meaning **84% full**. Reported verbatim it sounds like 0.84% — reassuring, and exactly backwards,
   on the one metric that explains hanging writes.

Also fixed: docs search dropped no stopwords, so under the OR fallback a whole-sentence question
like *"how do I request a GPU in my pod"* was ranked by the words `how`/`my`/`in` and returned three
FPGA pages, never surfacing the GPU page. With stopwords removed, top-1 accuracy on ten real
operator questions went from 2/10 to 8/10.

### Verified correct (no change needed)

- **Prometheus and Thanos need no auth** — both answer `/api/v1/query` with HTTP 200 and no
  credentials. `prometheus_bearer_token` stays unset.
- **The `/v1` suffix is right**; `GET /v1/models` returns 200 and lists 14 models.
- **`ceph_health_status`, `ceph_osd_up`, `ceph_pool_percent_used` all exist** — the metrics flagged
  as *most* likely to be wrong were the ones that were right.
- **`gpu-operator/nvidia-device-plugin-daemonset`** exists exactly as assumed.
- **Sealed-secrets is available** — controller `sealed-secrets` in `sealed-secrets-operator`,
  so the one Secret this repo needs can live in git as a SealedSecret.
- **The model drives the loop.** `deepseek-v4-flash` returns well-formed OpenAI `tool_calls`;
  the gateway 403s unauthenticated POSTs but accepts a key from outside the cluster, so the
  earlier 403 was missing auth rather than an IP restriction.
- **`url_for()` slug casing is correct.** Indexing the real repo produced 146 page URLs and all 146
  appear in the live sitemap; all 234 sitemap entries are lowercase with no trailing slash.

### Still open

1. **The container image has never been built or run.** CI is its first exercise. The smoke test in
   the build job runs it under the deployment's own security context and asserts the playbooks and
   all nine tools load, so a broken image fails the pipeline rather than reaching ArgoCD — but no
   one has watched it happen yet.
2. **Operator and channel IDs** in `deploy/configmap-allowlist.yaml` — still `U00000000000` /
   `C00000000000`. The team ID and model are now set, and the tokens are sealed and verified
   against `auth.test`, but with placeholder operator IDs the bot starts cleanly and ignores
   everyone.
3. **`deploy/networkpolicy.yaml`** — the API server CIDR, and the fact that the fallback egress rule
   permits any public HTTPS destination. Needs cluster access to confirm. A CiliumNetworkPolicy with
   FQDN rules is sketched in that file and is the version you want if Cilium is the CNI.
4. **The ArgoCD instance and AppProject** for `argocd/application.yaml`. The cluster has the
   argoproj CRDs and an `argo` namespace, but which instance owns this app is local convention.
5. **A documentation gap, not a code bug:** no page on nrp.ai covers a PVC stuck `Pending` — no
   indexed chunk contains both "pvc" and "pending". `admindocs/storage/user-pvc-issues` is about XFS
   corruption. Since that is among the most common storage questions, the agent will answer it from
   cluster evidence alone, and the ceph playbook now warns against stretching that page to fit.

## Testing it locally

The Slack listener needs a workspace, an app token and a Socket Mode connection before it answers
anything. `nrp-ops-ask` skips all of that and calls the *same* `Agent.investigate` — same tools,
same allowlist, same redaction, same budgets — so a question that works here works in Slack.

`.env` supplies the model; three more settings have production defaults that are wrong on a laptop:

```bash
NRP_OPS_LLM_API_KEY=...            # you have this
NRP_OPS_LLM_MODEL=deepseek-v4-flash
NRP_OPS_ALLOWLIST_PATH=dev/allowlist.yaml
NRP_OPS_DOCS_DB_PATH=./docs.sqlite3
NRP_OPS_KUBE_MODE=kubeconfig
NRP_OPS_KUBECONFIG_PATH=dev/kubeconfig-proxy.yaml
```

The allowlist matters most. Its default is `/etc/nrp-ops-agent/allowlist.yaml`, which does not exist
locally, and **an absent policy file is deny-everything** — every Kubernetes tool then refuses
before making an API call, which reads exactly like a healthy, empty cluster. `--check` catches
that and the other two:

```bash
nrp-ops-ask --check
```

Build the docs index once (clones the docs repo, ~870 chunks):

```bash
python -m nrp_ops_agent.docs_index.sync --db-path ./docs.sqlite3
```

### Kubernetes access needs `kubectl proxy`

NRP's API server certificate has no Authority Key Identifier extension. OpenSSL 3.x — Python 3.13
on macOS ships 3.6 — rejects it outright:

```
SSLCertVerificationError: certificate verify failed: Missing Authority Key Identifier
```

`kubectl` on the same machine works fine, so this looks like a bug in the agent and is not one. It
also sidesteps a second problem: the `nautilus` context authenticates through OIDC, and kubectl is
the tool that definitely refreshes those tokens. Run the proxy and point the agent at it:

```bash
kubectl proxy --port=8001 &
```

`dev/kubeconfig-proxy.yaml` targets `http://127.0.0.1:8001`, so no TLS and no credentials are
involved on the agent's side. None of this applies in the cluster, where `NRP_OPS_KUBE_MODE` is
`incluster` and the ServiceAccount CA is used.

### Asking things

```bash
nrp-ops-ask --show-tools "are there any failing pods in the coder namespace?"
```

`--show-tools` prints every call with arguments, latency and outcome, which is the fastest way to
see whether the model is following the playbook or wandering. `--which-playbook` prints the
selected runbook and exits without calling the model, and `--quiet` drops the live progress lines.

To replay the scored suite instead of an ad-hoc question:

```bash
python evals/run_evals.py --only coder      # one case
python evals/run_evals.py                   # all 20, scored
```

> `nrp-ops-ask` is a development entry point. It performs no Slack authorization — there is no
> Slack user to authorize — so it is not in the container's `args` and has no place in the
> Deployment. The namespace allowlist still applies, because that check lives inside each tool
> rather than at the edge.

## Container build and GitOps

The application and the manifests that deploy it share this repository, so a code change and the
rollout that carries it land in one history. ArgoCD watches `deploy/`.

`.github/workflows/build-container.yml` runs on every push to `main`:

1. **test** — `ruff`, `mypy`, `pytest`. The image is rolled out unattended, so a failing suite must
   stop the pipeline before anything is published.
2. **build** — builds the image, runs it the way Kubernetes will (non-root `65532`, read-only root
   filesystem, `/tmp` the only writable mount) and asserts the playbooks and all nine tools load,
   then pushes to `ghcr.io/djw8605/nrp-ops-bot` tagged `sha-<short>` and `latest`.
3. **bump** — rewrites `newTag` in `deploy/kustomization.yaml` and commits it back. **That commit is
   the deploy trigger**: ArgoCD sees the diff and syncs.

The tag is derived from the commit on purpose. A moving tag such as `latest` would leave
`kustomization.yaml` byte-identical, ArgoCD would see no diff, and nothing would ever roll out.

The bump commit cannot re-trigger the workflow: it touches only `deploy/`, which is in
`paths-ignore`, and pushes made with `GITHUB_TOKEN` do not start workflow runs. Either alone would
be enough.

Pull requests build and smoke-test the image but never push it.

> **First run:** the GHCR package is created private. Grant the repository access under
> *Package settings → Manage Actions access* (or make it public) or the cluster cannot pull it.
> If `main` is protected, the `bump` job needs permission to push to it.

### The namespace is not created here

`deploy/` deliberately contains no `Namespace`. It is owned by ArgoCD's destination or by a cluster
admin.

> **Carry the Pod Security labels across.** The `Namespace` this repo used to ship set
> `pod-security.kubernetes.io/enforce|audit|warn: restricted`. **Nothing in `deploy/` sets them any
> more.** The workloads still satisfy `restricted`, but nothing enforces that they continue to —
> whoever creates the namespace should apply those labels:
>
> ```bash
> kubectl label namespace system-nrp-ops-bot \
>   pod-security.kubernetes.io/enforce=restricted \
>   pod-security.kubernetes.io/audit=restricted \
>   pod-security.kubernetes.io/warn=restricted
> ```

### Secrets: what needs kubeseal

**Exactly one Secret, `nrp-ops-agent-tokens`, with three keys.** Everything else the agent reads is
a ConfigMap or an env var and is already in git.

| Key | Where it comes from | Needed for |
|---|---|---|
| `NRP_OPS_SLACK_BOT_TOKEN` | Slack app → OAuth & Permissions (`xoxb-…`) | posting replies |
| `NRP_OPS_SLACK_APP_TOKEN` | Slack app → Basic Information → App-Level Tokens (`xapp-…`, scope `connections:write`) | the Socket Mode connection |
| `NRP_OPS_LLM_API_KEY` | the NRP LLM gateway | every model call |

The cluster runs the sealed-secrets controller (`sealed-secrets` in namespace
`sealed-secrets-operator`, verified 2026-08-14), which is what makes this repository GitOps-able: a
`Secret` must never be committed, but the `SealedSecret` it becomes is encrypted to *this cluster's*
key and is safe in a public repo.

**`deploy/sealedsecret-tokens.yaml` is already generated and committed**, and
`deploy/kustomization.yaml` references it. Nothing further is needed unless a token changes.

To regenerate — reading straight from `.env` and piping, so the plaintext Secret is never written to
disk and never reaches the API server (`--dry-run=client`):

```bash
set -a; . ./.env; set +a
kubectl create secret generic nrp-ops-agent-tokens --namespace system-nrp-ops-bot \
  --from-literal=NRP_OPS_SLACK_BOT_TOKEN="$NRP_OPS_SLACK_BOT_TOKEN" \
  --from-literal=NRP_OPS_SLACK_APP_TOKEN="$NRP_OPS_SLACK_APP_TOKEN" \
  --from-literal=NRP_OPS_LLM_API_KEY="$NRP_OPS_LLM_API_KEY" \
  --dry-run=client -o yaml \
| kubeseal --controller-name sealed-secrets --controller-namespace sealed-secrets-operator \
    --format yaml > deploy/sealedsecret-tokens.yaml
```

Seal **only** these three. `.env` also holds `NRP_OPS_LLM_MODEL` and `NRP_OPS_PROMETHEUS_BASE_URL`,
which are not secrets and are set in `deploy/deployment.yaml` — and note that an explicit `env`
entry there *overrides* the same key arriving through `envFrom`, so a value sealed into the Secret
would lose to the manifest anyway.

> **Strict scope.** The ciphertext is bound to both the name `nrp-ops-agent-tokens` and the
> namespace `system-nrp-ops-bot`. Rename either and the controller cannot decrypt it — and it reports
> that in its own log rather than failing the sync, so the symptom is a pod stuck without
> credentials rather than an obvious error. Re-seal instead of editing.

`.env`, `secret-tokens.yaml` and `*.secret.yaml` are all in `.gitignore`, so plaintext cannot be
committed by accident.

Rotating a token means repeating the above and pushing — ArgoCD applies the new SealedSecret, but
the Deployment does not restart on a Secret change, so finish with:

```bash
kubectl -n system-nrp-ops-bot rollout restart deployment/nrp-ops-agent
```

### Deploying

1. **Create the namespace with the Pod Security labels.** `deploy/` no longer ships a Namespace, and
   a namespace ArgoCD creates on its own carries no PSS labels:
   ```bash
   kubectl create namespace system-nrp-ops-bot
   kubectl label namespace system-nrp-ops-bot \
     pod-security.kubernetes.io/enforce=restricted \
     pod-security.kubernetes.io/audit=restricted \
     pod-security.kubernetes.io/warn=restricted
   ```
   `argocd/application.yaml` also sets these via `managedNamespaceMetadata`, which keeps them under
   GitOps rather than depending on whoever ran the command above.
2. ~~Seal the tokens~~ — done; `deploy/sealedsecret-tokens.yaml` is committed and wired in.
3. **Fill in the operator and channel IDs** in `deploy/configmap-allowlist.yaml`. Still
   `U00000000000` / `C00000000000`, and **the agent denies everyone until they are real** — a
   deployment with placeholders starts cleanly and ignores every message. `NRP_OPS_SLACK_TEAM_ID`
   and `NRP_OPS_LLM_MODEL` are already set.
4. **Make the GHCR package readable by the cluster.** It is created private on first push; either
   make it public or add a pull secret.
5. **Register the Application**, after checking with the NRP admins which ArgoCD instance and
   AppProject to use — the cluster has the argoproj CRDs and an `argo` namespace, but that choice is
   local convention this repo cannot infer:
   ```bash
   kubectl apply -f argocd/application.yaml
   ```
6. **Nothing to seed.** The `docs-sync` sidecar builds the index as soon as the pod starts and
   hourly after that. `search_docs` returns `docs_index_missing` for the first minute or so; if it
   still does, the sidecar is failing and its log says why:
   ```bash
   kubectl -n system-nrp-ops-bot logs deploy/nrp-ops-agent -c docs-sync
   ```

Render what ArgoCD will apply, without a cluster:

```bash
kubectl kustomize deploy/
```

To apply by hand instead of through ArgoCD:

```bash
kubectl apply -k deploy/
```

The docs index starts empty and the `docs-sync` sidecar fills it about a minute after the pod
starts, so `search_docs` answers nothing until then.

Confirm the RBAC does what it claims, against the real cluster:

```bash
kubectl auth can-i get secrets --as=system:serviceaccount:system-nrp-ops-bot:nrp-ops-agent   # want: no
kubectl auth can-i create pods/exec --as=system:serviceaccount:system-nrp-ops-bot:nrp-ops-agent  # no
kubectl auth can-i list pods --as=system:serviceaccount:system-nrp-ops-bot:nrp-ops-agent      # want: yes
```

## Evals

`evals/cases.yaml` holds 24 operator questions with expected namespaces, workloads, findings and
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

`python -m nrp_ops_agent.mcp_server` exposes the same sixteen tools over MCP. It is a second *front
end*, not a second tool surface: every call goes through the same dispatcher, so validation, the
namespace allowlist, redaction and the audit trail all apply, and the advertised schemas are
generated from the same pydantic models. MCP clients bypass the Slack authorization gate, so only
expose it to people already trusted with cluster read access.

Note that an MCP client has nowhere to put an image, so `accounting_chart` returns its numbers with
`attached: false` there rather than failing — the collector is installed by the agent loop, and
`charts.emit` reports that there is none.

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
`pydantic` v2, SQLite FTS5 for the docs index, `matplotlib` (Agg) for charts. No agent framework — the tool-calling loop is
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
