"""Read-only Kubernetes tools.

Security role: these are the only ways the agent can touch the cluster, and they
are read-only by construction as well as by RBAC.

* Every namespace argument is validated against the allowlist **inside the tool**
  (:func:`_require_namespace`), not in the caller, so there is no path that skips
  the check.
* Every returned payload goes through :func:`scrub_k8s_fields`, which removes
  ``managedFields``, the ``kubectl.kubernetes.io/last-applied-configuration``
  annotation, all ``env``/``envFrom``, ``imagePullSecrets``, and reduces
  ``secret``/``configMap`` volume references to their name. Handlers also build
  their summaries field-by-field rather than dumping objects, so the scrubber is
  the second line of defence rather than the only one.
* ``get_logs``, ``get_events`` and the annotations in ``get_pod_status`` return
  attacker-controlled text -- any Nautilus user can write into them. Those tools
  are flagged ``untrusted_output=True`` so the agent loop wraps them.

There is deliberately no ConfigMap tool and no Secret tool. See
``deploy/clusterrole.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from nrp_ops_agent.config import get_allowlist_loader, get_settings
from nrp_ops_agent.tools import NamespaceDenied, ToolError, make_tool

# --------------------------------------------------------------------------- #
# Shared argument types
# --------------------------------------------------------------------------- #

_DNS_NAME: Final = r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$"
# Label/field selectors have their own grammar; restrict the charset rather than
# trying to parse it, so that nothing exotic reaches the API server query string.
_SELECTOR: Final = r"^[A-Za-z0-9_.\-/=!,()\[\] *]{0,512}$"

Namespace = Annotated[str, StringConstraints(pattern=_DNS_NAME, max_length=253)]
ResourceName = Annotated[str, StringConstraints(pattern=_DNS_NAME, max_length=253)]
Selector = Annotated[str, StringConstraints(pattern=_SELECTOR, max_length=512)]

#: Total log bytes returned by one ``get_logs`` call. A single chatty pod can
#: emit megabytes per minute, which would evict everything else from context.
MAX_LOG_BYTES: Final = 96_000


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# Field scrubbing
# --------------------------------------------------------------------------- #

#: Dropped wherever they appear. ``env``/``envFrom`` are how credentials most
#: often reach a pod spec; ``managedFields`` is noise that also replays old spec
#: contents; ``imagePullSecrets`` names registry credentials.
_DROP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "managedFields",
        "managed_fields",
        "env",
        "envFrom",
        "env_from",
        "imagePullSecrets",
        "image_pull_secrets",
    }
)

#: Reduced to ``{"name": ...}`` wherever they appear, so the agent can say "the
#: coder-tls secret is mounted" without being able to read what is in it.
_NAME_ONLY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "secret",
        "configMap",
        "config_map",
        "secretRef",
        "secret_ref",
        "configMapRef",
        "config_map_ref",
        "secretKeyRef",
        "secret_key_ref",
        "configMapKeyRef",
        "config_map_key_ref",
    }
)

_LAST_APPLIED: Final = "kubectl.kubernetes.io/last-applied-configuration"


def scrub_k8s_fields(node: Any) -> Any:
    """Recursively remove sensitive Kubernetes fields from a payload.

    Applied to every tool result. Context-free on purpose: it does not need to
    know whether it is looking at a volume, a container or an owner reference,
    so a future tool cannot accidentally route around it.
    """
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _DROP_KEYS:
                continue
            if key in _NAME_ONLY_KEYS and isinstance(value, dict):
                name = value.get("name")
                out[key] = {"name": name} if name is not None else {}
                continue
            if key in ("annotations", "labels") and isinstance(value, dict):
                out[key] = {k: v for k, v in value.items() if k != _LAST_APPLIED}
                continue
            out[key] = scrub_k8s_fields(value)
        return out
    if isinstance(node, list):
        return [scrub_k8s_fields(item) for item in node]
    return node


# --------------------------------------------------------------------------- #
# Client plumbing
# --------------------------------------------------------------------------- #


@dataclass
class KubeClients:
    """Typed as ``Any`` deliberately: kubernetes-asyncio ships no stubs, and
    annotating with its classes would make every signature implicitly Any."""

    core: Any
    apps: Any
    batch: Any = None


_clients_singleton: KubeClients | None = None


async def _load_kube_config() -> None:
    from kubernetes_asyncio import config as kube_config

    settings = get_settings()
    mode = settings.kube_mode

    if mode in ("incluster", "auto"):
        try:
            # kubernetes-asyncio ships partial types; this loader is unannotated.
            kube_config.load_incluster_config()  # type: ignore[no-untyped-call]
            return
        except Exception:
            if mode == "incluster":
                raise
            # 'auto': fall through to kubeconfig for local development.

    await kube_config.load_kube_config(
        config_file=str(settings.kubeconfig_path) if settings.kubeconfig_path else None,
        context=settings.kubeconfig_context,
    )


async def _clients() -> KubeClients:
    """Process-wide client pair. Monkeypatched wholesale in tests."""
    global _clients_singleton
    if _clients_singleton is None:
        from kubernetes_asyncio import client as kube_client

        await _load_kube_config()
        api = kube_client.ApiClient()
        _clients_singleton = KubeClients(
            core=kube_client.CoreV1Api(api),
            apps=kube_client.AppsV1Api(api),
            batch=kube_client.BatchV1Api(api),
        )
    return _clients_singleton


async def aclose_clients() -> None:
    """Close the shared ApiClient and forget it.

    The long-running Slack process holds exactly one of these for its lifetime,
    so this is not leak cleanup -- it is for short-lived entry points such as
    ``nrp-ops-ask``, where an unclosed aiohttp session prints a warning at
    interpreter shutdown that looks like a fault in the answer just printed.
    """
    global _clients_singleton
    if _clients_singleton is None:
        return
    api = getattr(_clients_singleton.core, "api_client", None)
    _clients_singleton = None
    if api is not None:
        await api.close()


def _require_namespace(namespace: str) -> str:
    """Deny-by-default namespace gate, called inside every namespaced tool."""
    decision = get_allowlist_loader().load().namespace_decision(namespace)
    if decision != "allowed":
        raise NamespaceDenied(namespace, decision)
    return namespace


# --------------------------------------------------------------------------- #
# Small helpers for reading typed API objects
# --------------------------------------------------------------------------- #


def _g(obj: Any, *path: str) -> Any:
    """Attribute walk that tolerates ``None`` at any level."""
    for part in path:
        if obj is None:
            return None
        obj = getattr(obj, part, None)
    return obj


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat(timespec="seconds")
    return str(value)


def _age_seconds(value: Any) -> int | None:
    if not isinstance(value, datetime):
        return None
    return int((datetime.now(UTC) - value.astimezone(UTC)).total_seconds())


def _container_state(state: Any) -> dict[str, Any]:
    """Flatten a ContainerState into the bits that explain an outage."""
    if state is None:
        return {"state": "unknown"}
    if _g(state, "running") is not None:
        return {"state": "running", "started_at": _iso(_g(state, "running", "started_at"))}
    if _g(state, "waiting") is not None:
        return {
            "state": "waiting",
            "reason": _g(state, "waiting", "reason"),
            "message": _g(state, "waiting", "message"),
        }
    if _g(state, "terminated") is not None:
        return {
            "state": "terminated",
            "reason": _g(state, "terminated", "reason"),
            "exit_code": _g(state, "terminated", "exit_code"),
            "signal": _g(state, "terminated", "signal"),
            "started_at": _iso(_g(state, "terminated", "started_at")),
            "finished_at": _iso(_g(state, "terminated", "finished_at")),
            "message": _g(state, "terminated", "message"),
        }
    return {"state": "unknown"}


def _conditions(obj: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cond in _g(obj, "status", "conditions") or []:
        out.append(
            {
                "type": _g(cond, "type"),
                "status": _g(cond, "status"),
                "reason": _g(cond, "reason"),
                "message": _g(cond, "message"),
                "last_transition_time": _iso(_g(cond, "last_transition_time")),
            }
        )
    return out


def _pod_summary(pod: Any) -> dict[str, Any]:
    statuses = _g(pod, "status", "container_statuses") or []
    ready = sum(1 for cs in statuses if _g(cs, "ready"))
    restarts = sum(int(_g(cs, "restart_count") or 0) for cs in statuses)
    problems = [
        {
            "container": _g(cs, "name"),
            **_container_state(_g(cs, "state")),
        }
        for cs in statuses
        if not _g(cs, "ready")
    ]
    return {
        "name": _g(pod, "metadata", "name"),
        "namespace": _g(pod, "metadata", "namespace"),
        "phase": _g(pod, "status", "phase"),
        "ready": f"{ready}/{len(statuses)}" if statuses else "0/0",
        "restarts": restarts,
        "node": _g(pod, "spec", "node_name"),
        "age_seconds": _age_seconds(_g(pod, "metadata", "creation_timestamp")),
        "deletion_timestamp": _iso(_g(pod, "metadata", "deletion_timestamp")),
        "not_ready_containers": problems,
    }


def _wrap_api_error(exc: Exception) -> ToolError:
    """Turn a kubernetes-asyncio ApiException into something actionable.

    A 403 here is worth surfacing plainly: it usually means the ClusterRole is
    doing its job, and the model should stop retrying rather than hunt for a way
    around it.
    """
    status = getattr(exc, "status", None)
    reason = getattr(exc, "reason", None)
    if status == 404:
        return ToolError("not_found", f"Not found: {reason or exc}")
    if status == 403:
        return ToolError(
            "forbidden",
            "The agent's ServiceAccount is not permitted to read this. This is "
            "intentional and is not something to work around.",
        )
    if status is not None:
        return ToolError("api_error", f"Kubernetes API returned {status}: {reason or exc}")
    return ToolError("api_error", f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# list_pods
# --------------------------------------------------------------------------- #


class ListPodsArgs(_Args):
    namespace: Namespace
    label_selector: Selector | None = Field(
        default=None, description="e.g. app=coder or app in (coder,coder-proxy)"
    )
    field_selector: Selector | None = Field(default=None, description="e.g. status.phase=Running")
    limit: int = Field(default=50, ge=1, le=200)


@make_tool(
    "list_pods",
    "List pods in a namespace with phase, readiness, restart counts and the state "
    "of any container that is not ready. Start here for 'is X up?' questions.",
    ListPodsArgs,
)
async def list_pods(args: ListPodsArgs) -> dict[str, Any]:
    _require_namespace(args.namespace)
    clients = await _clients()
    try:
        resp = await clients.core.list_namespaced_pod(
            namespace=args.namespace,
            label_selector=args.label_selector,
            field_selector=args.field_selector,
            limit=args.limit,
            timeout_seconds=int(get_settings().kube_request_timeout_s),
        )
    except Exception as exc:
        raise _wrap_api_error(exc) from exc

    items = list(_g(resp, "items") or [])
    pods = [_pod_summary(pod) for pod in items[: args.limit]]
    result: dict[str, Any] = {
        "namespace": args.namespace,
        "returned": len(pods),
        "truncated": bool(_g(resp, "metadata", "_continue")) or len(items) > args.limit,
        "pods": pods,
    }
    return scrub_k8s_fields(result)  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# get_pod_status
# --------------------------------------------------------------------------- #


class GetPodStatusArgs(_Args):
    namespace: Namespace
    name: ResourceName


@make_tool(
    "get_pod_status",
    "Detailed status for one pod: phase, conditions, per-container current and "
    "previous state including termination reason and exit code, restart counts, "
    "node, QoS class and scheduling status. Use after list_pods narrows it down.",
    GetPodStatusArgs,
    # Pod annotations are writable by whoever owns the workload.
    untrusted_output=True,
)
async def get_pod_status(args: GetPodStatusArgs) -> dict[str, Any]:
    _require_namespace(args.namespace)
    clients = await _clients()
    try:
        pod = await clients.core.read_namespaced_pod(name=args.name, namespace=args.namespace)
    except Exception as exc:
        raise _wrap_api_error(exc) from exc

    containers: list[dict[str, Any]] = []
    for cs in _g(pod, "status", "container_statuses") or []:
        entry: dict[str, Any] = {
            "name": _g(cs, "name"),
            "image": _g(cs, "image"),
            "ready": _g(cs, "ready"),
            "started": _g(cs, "started"),
            "restart_count": _g(cs, "restart_count"),
            "current": _container_state(_g(cs, "state")),
        }
        last = _g(cs, "last_state")
        if _g(last, "terminated") is not None:
            entry["previous"] = _container_state(last)
        containers.append(entry)

    init_containers = [
        {
            "name": _g(cs, "name"),
            "ready": _g(cs, "ready"),
            "restart_count": _g(cs, "restart_count"),
            "current": _container_state(_g(cs, "state")),
        }
        for cs in _g(pod, "status", "init_container_statuses") or []
    ]

    conditions = _conditions(pod)
    scheduled = next((c for c in conditions if c["type"] == "PodScheduled"), None)

    result: dict[str, Any] = {
        "name": _g(pod, "metadata", "name"),
        "namespace": _g(pod, "metadata", "namespace"),
        "phase": _g(pod, "status", "phase"),
        "reason": _g(pod, "status", "reason"),
        "message": _g(pod, "status", "message"),
        "node": _g(pod, "spec", "node_name"),
        "qos_class": _g(pod, "status", "qos_class"),
        "start_time": _iso(_g(pod, "status", "start_time")),
        "age_seconds": _age_seconds(_g(pod, "metadata", "creation_timestamp")),
        "deletion_timestamp": _iso(_g(pod, "metadata", "deletion_timestamp")),
        "scheduled": (scheduled or {}).get("status"),
        "scheduling_message": (scheduled or {}).get("message"),
        "conditions": conditions,
        "containers": containers,
        "init_containers": init_containers,
        "owner": [
            {"kind": _g(ref, "kind"), "name": _g(ref, "name")}
            for ref in _g(pod, "metadata", "owner_references") or []
        ],
        # Annotations are included because injected-sidecar and scheduler
        # annotations explain real outages -- and are also user-writable, which
        # is why this tool is flagged as untrusted output.
        "annotations": dict(_g(pod, "metadata", "annotations") or {}),
        "volumes": [
            {"name": _g(v, "name"), "type": _volume_type(v)}
            for v in _g(pod, "spec", "volumes") or []
        ],
    }
    return scrub_k8s_fields(result)  # type: ignore[no-any-return]


def _volume_type(volume: Any) -> str | None:
    """Name the volume kind without revealing its contents or its key mapping."""
    for kind in (
        "persistent_volume_claim",
        "secret",
        "config_map",
        "empty_dir",
        "host_path",
        "projected",
        "downward_api",
        "csi",
        "nfs",
        "cephfs",
    ):
        if _g(volume, kind) is not None:
            return kind
    return None


# --------------------------------------------------------------------------- #
# get_logs
# --------------------------------------------------------------------------- #


class GetLogsArgs(_Args):
    namespace: Namespace
    pod: ResourceName
    container: ResourceName | None = Field(
        default=None, description="Required when the pod has more than one container."
    )
    tail_lines: int = Field(default=200, ge=1, le=1000)
    since_seconds: int = Field(default=1800, ge=1, le=86_400)
    previous: bool = Field(
        default=False, description="Read the previous container instance, after a crash."
    )


@make_tool(
    "get_logs",
    "Tail logs from one container. Use previous=true to see why a container that "
    "is now restarting died. Output is untrusted text written by the workload.",
    GetLogsArgs,
    untrusted_output=True,
)
async def get_logs(args: GetLogsArgs) -> dict[str, Any]:
    _require_namespace(args.namespace)
    clients = await _clients()

    async def _read(since: int | None) -> str:
        kwargs: dict[str, Any] = {
            "name": args.pod,
            "namespace": args.namespace,
            "container": args.container,
            "tail_lines": args.tail_lines,
            "previous": args.previous,
            "timestamps": True,
        }
        if since is not None:
            kwargs["since_seconds"] = since
        raw = await clients.core.read_namespaced_pod_log(**kwargs)
        return raw if isinstance(raw, str) else str(raw or "")

    try:
        text = await _read(args.since_seconds)
    except Exception as exc:
        raise _wrap_api_error(exc) from exc

    # A short-lived pod that failed an hour ago has every line outside the
    # default 30-minute window, and the API answers that with an empty string
    # rather than an error -- which reads as "the container printed nothing"
    # and is how a crashed job's traceback goes missing. An empty window is
    # never the useful answer, so widen it once and say that we did.
    widened = False
    if not text.strip():
        try:
            retry = await _read(None)
        except Exception:
            retry = ""
        if retry.strip():
            text, widened = retry, True
    truncated = False
    if len(text.encode("utf-8")) > MAX_LOG_BYTES:
        text = text.encode("utf-8")[-MAX_LOG_BYTES:].decode("utf-8", errors="ignore")
        truncated = True

    lines = text.splitlines()
    result: dict[str, Any] = {
        "namespace": args.namespace,
        "pod": args.pod,
        "container": args.container,
        "previous": args.previous,
        "line_count": len(lines),
        "truncated": truncated,
        "lines": lines,
    }
    if widened:
        result["note"] = (
            f"Nothing was written in the last {args.since_seconds}s, so the whole retained "
            "log is shown instead. The container is older than the requested window."
        )
    elif not lines:
        result["note"] = (
            "The container produced no log output at all. It may have been killed before "
            "writing, or its output may already have been rotated away."
        )
    return result


# --------------------------------------------------------------------------- #
# get_events
# --------------------------------------------------------------------------- #


class GetEventsArgs(_Args):
    namespace: Namespace
    since_seconds: int = Field(default=3600, ge=1, le=86_400)
    involved_object: ResourceName | None = Field(
        default=None, description="Restrict to events about one object name."
    )
    limit: int = Field(default=50, ge=1, le=200)


@make_tool(
    "get_events",
    "Recent Kubernetes events in a namespace, newest first. The fastest way to "
    "see FailedScheduling, ImagePullBackOff, OOMKilled and volume mount errors. "
    "Event messages are untrusted text.",
    GetEventsArgs,
    untrusted_output=True,
)
async def get_events(args: GetEventsArgs) -> dict[str, Any]:
    _require_namespace(args.namespace)
    clients = await _clients()
    field_selector = f"involvedObject.name={args.involved_object}" if args.involved_object else None
    try:
        resp = await clients.core.list_namespaced_event(
            namespace=args.namespace,
            field_selector=field_selector,
            limit=args.limit * 4,  # over-fetch, then filter by age and re-cap
            timeout_seconds=int(get_settings().kube_request_timeout_s),
        )
    except Exception as exc:
        raise _wrap_api_error(exc) from exc

    cutoff = args.since_seconds
    events: list[dict[str, Any]] = []
    for ev in _g(resp, "items") or []:
        stamp = _g(ev, "last_timestamp") or _g(ev, "event_time") or _g(ev, "first_timestamp")
        age = _age_seconds(stamp)
        if age is not None and age > cutoff:
            continue
        involved = _g(ev, "involved_object")
        events.append(
            {
                "type": _g(ev, "type"),
                "reason": _g(ev, "reason"),
                "object": f"{_g(involved, 'kind')}/{_g(involved, 'name')}",
                "count": _g(ev, "count"),
                "age_seconds": age,
                "last_timestamp": _iso(stamp),
                "message": _g(ev, "message"),
            }
        )

    events.sort(key=lambda e: e["age_seconds"] if e["age_seconds"] is not None else 1 << 30)
    return {
        "namespace": args.namespace,
        "since_seconds": args.since_seconds,
        "returned": min(len(events), args.limit),
        "truncated": len(events) > args.limit,
        "events": events[: args.limit],
    }


# --------------------------------------------------------------------------- #
# get_workload
# --------------------------------------------------------------------------- #

WorkloadKind = Literal["deployment", "statefulset", "daemonset", "replicaset"]

_READ_METHOD: Final[dict[str, str]] = {
    "deployment": "read_namespaced_deployment",
    "statefulset": "read_namespaced_stateful_set",
    "daemonset": "read_namespaced_daemon_set",
    "replicaset": "read_namespaced_replica_set",
}


class GetWorkloadArgs(_Args):
    kind: WorkloadKind
    namespace: Namespace
    name: ResourceName


@make_tool(
    "get_workload",
    "Replica health for a deployment, statefulset, daemonset or replicaset: "
    "desired vs ready vs available vs updated, generation vs observedGeneration "
    "(a mismatch means the controller has not acted on the latest spec), and "
    "rollout conditions.",
    GetWorkloadArgs,
)
async def get_workload(args: GetWorkloadArgs) -> dict[str, Any]:
    _require_namespace(args.namespace)
    clients = await _clients()
    method = getattr(clients.apps, _READ_METHOD[args.kind])
    try:
        obj = await method(name=args.name, namespace=args.namespace)
    except Exception as exc:
        raise _wrap_api_error(exc) from exc

    status = _g(obj, "status")
    if args.kind == "daemonset":
        replicas = {
            "desired": _g(status, "desired_number_scheduled"),
            "current": _g(status, "current_number_scheduled"),
            "ready": _g(status, "number_ready"),
            "available": _g(status, "number_available"),
            "updated": _g(status, "updated_number_scheduled"),
            "misscheduled": _g(status, "number_misscheduled"),
        }
    else:
        replicas = {
            "desired": _g(obj, "spec", "replicas"),
            "current": _g(status, "replicas"),
            "ready": _g(status, "ready_replicas"),
            "available": _g(status, "available_replicas"),
            "updated": _g(status, "updated_replicas"),
        }

    generation = _g(obj, "metadata", "generation")
    observed = _g(status, "observed_generation")
    result: dict[str, Any] = {
        "kind": args.kind,
        "namespace": args.namespace,
        "name": args.name,
        "replicas": replicas,
        "generation": generation,
        "observed_generation": observed,
        "rollout_in_progress": generation is not None
        and observed is not None
        and generation != observed,
        "age_seconds": _age_seconds(_g(obj, "metadata", "creation_timestamp")),
        "conditions": _conditions(obj),
        "images": [
            {"container": _g(c, "name"), "image": _g(c, "image")}
            for c in _g(obj, "spec", "template", "spec", "containers") or []
        ],
    }
    return scrub_k8s_fields(result)  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# CronJobs
# --------------------------------------------------------------------------- #


class GetCronJobArgs(_Args):
    namespace: Namespace
    name: ResourceName
    job_limit: Annotated[int, Field(ge=1, le=20)] = 5


@make_tool(
    "get_cronjob",
    "Schedule and recent run outcomes for a CronJob: the cron expression, last "
    "schedule and successful times, whether it is suspended, and the most recent "
    "Jobs it created with their succeeded/failed counts and the pod names to pull "
    "logs from. Use this for 'did the cron job run' and 'why did it fail' -- "
    "get_workload does not cover CronJobs.",
    GetCronJobArgs,
)
async def get_cronjob(args: GetCronJobArgs) -> dict[str, Any]:
    _require_namespace(args.namespace)
    clients = await _clients()
    try:
        obj = await clients.batch.read_namespaced_cron_job(name=args.name, namespace=args.namespace)
    except Exception as exc:
        raise _wrap_api_error(exc) from exc

    spec, status = _g(obj, "spec"), _g(obj, "status")
    uid = _g(obj, "metadata", "uid")

    # Jobs are matched by ownerReference rather than by name prefix: a Job named
    # <cronjob>-29779157 looks like a child but nothing stops an unrelated Job
    # from being named that way, and the owner reference is what Kubernetes
    # itself uses.
    jobs: list[dict[str, Any]] = []
    try:
        listed = await clients.batch.list_namespaced_job(namespace=args.namespace)
    except Exception as exc:
        raise _wrap_api_error(exc) from exc

    owned = [
        job
        for job in _g(listed, "items") or []
        if any(_g(ref, "uid") == uid for ref in _g(job, "metadata", "owner_references") or [])
    ]
    owned.sort(key=lambda j: str(_g(j, "metadata", "creation_timestamp") or ""), reverse=True)

    for job in owned[: args.job_limit]:
        job_status = _g(job, "status")
        jobs.append(
            {
                "name": _g(job, "metadata", "name"),
                "age_seconds": _age_seconds(_g(job, "metadata", "creation_timestamp")),
                "active": _g(job_status, "active") or 0,
                "succeeded": _g(job_status, "succeeded") or 0,
                "failed": _g(job_status, "failed") or 0,
                "conditions": _conditions(job),
            }
        )

    result: dict[str, Any] = {
        "kind": "cronjob",
        "namespace": args.namespace,
        "name": args.name,
        "schedule": _g(spec, "schedule"),
        "suspended": bool(_g(spec, "suspend")),
        "concurrency_policy": _g(spec, "concurrency_policy"),
        # A Job whose pods are gone takes its logs with it. Surfacing the policy
        # here is what turns "logs return not_found" from a mystery into an
        # explanation.
        "restart_policy": _g(spec, "job_template", "spec", "template", "spec", "restart_policy"),
        "last_schedule_time": _g(status, "last_schedule_time"),
        "last_successful_time": _g(status, "last_successful_time"),
        "active_jobs": len(_g(status, "active") or []),
        "recent_jobs": jobs,
        "images": [
            {"container": _g(c, "name"), "image": _g(c, "image")}
            for c in _g(spec, "job_template", "spec", "template", "spec", "containers") or []
        ],
    }
    return scrub_k8s_fields(result)  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

_PRESSURE_CONDITIONS: Final = (
    "MemoryPressure",
    "DiskPressure",
    "PIDPressure",
    "NetworkUnavailable",
)


def _node_summary(node: Any, *, include_taints: bool) -> dict[str, Any]:
    conditions = {
        str(_g(c, "type")): {"status": _g(c, "status"), "reason": _g(c, "reason")}
        for c in _g(node, "status", "conditions") or []
    }
    ready = conditions.get("Ready", {}).get("status")
    pressure = {
        name: conditions[name]["status"] for name in _PRESSURE_CONDITIONS if name in conditions
    }
    summary: dict[str, Any] = {
        "name": _g(node, "metadata", "name"),
        "ready": ready,
        "unschedulable": bool(_g(node, "spec", "unschedulable")),
        "pressure": pressure,
        "conditions": conditions,
    }
    if include_taints:
        summary["taints"] = [
            {"key": _g(t, "key"), "value": _g(t, "value"), "effect": _g(t, "effect")}
            for t in _g(node, "spec", "taints") or []
        ]
        summary["capacity"] = dict(_g(node, "status", "capacity") or {})
        summary["allocatable"] = dict(_g(node, "status", "allocatable") or {})
        summary["kubelet_version"] = _g(node, "status", "node_info", "kubelet_version")
    return summary


def _node_is_unhealthy(summary: dict[str, Any]) -> bool:
    if summary.get("ready") != "True":
        return True
    if summary.get("unschedulable"):
        return True
    return any(status == "True" for status in (summary.get("pressure") or {}).values())


class GetNodeArgs(_Args):
    name: ResourceName


@make_tool(
    "get_node",
    "Conditions, taints, capacity vs allocatable and pressure flags for one node. "
    "Use when pods on a specific node are unhealthy.",
    GetNodeArgs,
)
async def get_node(args: GetNodeArgs) -> dict[str, Any]:
    clients = await _clients()
    try:
        node = await clients.core.read_node(name=args.name)
    except Exception as exc:
        raise _wrap_api_error(exc) from exc
    # Node labels are deliberately not returned: on Nautilus they are a large
    # inventory dump that crowds out the actual answer.
    return scrub_k8s_fields(_node_summary(node, include_taints=True))  # type: ignore[no-any-return]


class ListNodesArgs(_Args):
    label_selector: Selector | None = Field(
        default=None, description="e.g. nvidia.com/gpu.product=NVIDIA-A100"
    )
    only_unhealthy: bool = Field(
        default=True,
        description="Default true: NRP has hundreds of nodes, and the healthy ones "
        "are never the answer.",
    )


@make_tool(
    "list_nodes",
    "Cluster node health. By default returns only nodes that are NotReady, "
    "cordoned, or reporting memory/disk/PID pressure.",
    ListNodesArgs,
)
async def list_nodes(args: ListNodesArgs) -> dict[str, Any]:
    clients = await _clients()
    try:
        resp = await clients.core.list_node(
            label_selector=args.label_selector,
            timeout_seconds=int(get_settings().kube_request_timeout_s),
        )
    except Exception as exc:
        raise _wrap_api_error(exc) from exc

    items = list(_g(resp, "items") or [])
    summaries = [_node_summary(node, include_taints=False) for node in items]
    selected = [s for s in summaries if _node_is_unhealthy(s)] if args.only_unhealthy else summaries

    limit = 100
    return scrub_k8s_fields(  # type: ignore[no-any-return]
        {
            "total_nodes": len(summaries),
            "unhealthy_count": sum(1 for s in summaries if _node_is_unhealthy(s)),
            "only_unhealthy": args.only_unhealthy,
            "returned": min(len(selected), limit),
            "truncated": len(selected) > limit,
            "nodes": selected[:limit],
        }
    )
