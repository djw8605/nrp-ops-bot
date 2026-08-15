"""Tests for the read-only Kubernetes tools against a mocked API client.

The load-bearing assertions are the negative ones: environment variables,
managedFields, imagePullSecrets and Secret volume contents must never appear in
tool output, and no namespace outside the allowlist may be reached.
"""

from __future__ import annotations

from typing import Any

import pytest

from helpers import ApiException, ago, ns
from nrp_ops_agent.tools import dispatch, registry
from nrp_ops_agent.tools import k8s as k8s_tools
from nrp_ops_agent.tools.k8s import (
    GetCronJobArgs,
    GetEventsArgs,
    GetLogsArgs,
    GetNodeArgs,
    GetPodStatusArgs,
    GetWorkloadArgs,
    ListNodesArgs,
    ListPodsArgs,
    get_cronjob,
    get_events,
    get_logs,
    get_node,
    get_pod_status,
    get_workload,
    list_nodes,
    list_pods,
    scrub_k8s_fields,
)

# Applied per class rather than per module: TestNamespaceEnforcement installs its
# own narrower policy, and two competing usefixtures marks have no defined order.
allow_all = pytest.mark.usefixtures("allow_all_namespaces")


# --------------------------------------------------------------------------- #
# Fixtures for fake objects
# --------------------------------------------------------------------------- #


def make_pod(
    name: str = "coder-0",
    namespace: str = "coder",
    phase: str = "Running",
    ready: bool = True,
    restarts: int = 0,
    waiting_reason: str | None = None,
    terminated_reason: str | None = None,
) -> Any:
    state = ns(running=ns(started_at=ago(600)), waiting=None, terminated=None)
    if waiting_reason:
        state = ns(
            running=None,
            waiting=ns(reason=waiting_reason, message="back-off restarting"),
            terminated=None,
        )
    last_state = ns(running=None, waiting=None, terminated=None)
    if terminated_reason:
        last_state = ns(
            running=None,
            waiting=None,
            terminated=ns(
                reason=terminated_reason,
                exit_code=137,
                signal=9,
                started_at=ago(900),
                finished_at=ago(700),
                message="Container was OOM killed",
            ),
        )
    return ns(
        metadata=ns(
            name=name,
            namespace=namespace,
            creation_timestamp=ago(3600),
            deletion_timestamp=None,
            annotations={
                "coder.com/session": "abc",
                "kubectl.kubernetes.io/last-applied-configuration": '{"spec":{"env":"SECRET"}}',
            },
            owner_references=[ns(kind="StatefulSet", name="coder")],
            managed_fields=[{"manager": "kubectl", "fieldsV1": {"f:spec": {}}}],
        ),
        spec=ns(
            node_name="node-gpu-04",
            volumes=[
                ns(
                    name="tls",
                    secret=ns(secret_name="coder-tls"),
                    persistent_volume_claim=None,
                    config_map=None,
                    empty_dir=None,
                    host_path=None,
                    projected=None,
                    downward_api=None,
                    csi=None,
                    nfs=None,
                    cephfs=None,
                ),
            ],
        ),
        status=ns(
            phase=phase,
            reason=None,
            message=None,
            qos_class="Burstable",
            start_time=ago(3500),
            conditions=[
                ns(
                    type="PodScheduled",
                    status="True",
                    reason=None,
                    message=None,
                    last_transition_time=ago(3500),
                ),
                ns(
                    type="Ready",
                    status="True" if ready else "False",
                    reason=None,
                    message=None,
                    last_transition_time=ago(3400),
                ),
            ],
            container_statuses=[
                ns(
                    name="coder",
                    image="nrp/coder:v2.9.1",
                    ready=ready,
                    started=ready,
                    restart_count=restarts,
                    state=state,
                    last_state=last_state,
                )
            ],
            init_container_statuses=[],
        ),
    )


# --------------------------------------------------------------------------- #
# scrub_k8s_fields
# --------------------------------------------------------------------------- #


class TestScrubber:
    def test_drops_env_managed_fields_and_pull_secrets(self) -> None:
        raw = {
            "metadata": {
                "name": "coder-0",
                "managedFields": [{"manager": "kubectl"}],
                "annotations": {
                    "keep": "yes",
                    "kubectl.kubernetes.io/last-applied-configuration": "{...}",
                },
            },
            "spec": {
                "imagePullSecrets": [{"name": "regcred"}],
                "containers": [
                    {
                        "name": "coder",
                        "env": [{"name": "DB_PASSWORD", "value": "hunter2"}],
                        "envFrom": [{"secretRef": {"name": "coder-env"}}],
                    }
                ],
            },
        }
        out = scrub_k8s_fields(raw)
        flat = repr(out)
        assert "managedFields" not in flat
        assert "DB_PASSWORD" not in flat
        assert "hunter2" not in flat
        assert "imagePullSecrets" not in flat
        assert "envFrom" not in flat
        assert "last-applied-configuration" not in flat
        assert out["metadata"]["annotations"] == {"keep": "yes"}

    def test_secret_and_configmap_refs_keep_only_the_name(self) -> None:
        raw = {
            "volumes": [
                {
                    "name": "tls",
                    "secret": {"secretName": "coder-tls", "items": [{"key": "tls.key"}]},
                },
                {"name": "cfg", "configMap": {"name": "coder-config", "items": [{"key": "x"}]}},
            ]
        }
        out = scrub_k8s_fields(raw)
        # secretName is not `name`, so nothing at all survives -- correct: the
        # scrubber must not leak the key mapping either.
        assert out["volumes"][0]["secret"] == {}
        assert out["volumes"][1]["configMap"] == {"name": "coder-config"}
        assert "tls.key" not in repr(out)

    def test_nested_lists_are_walked(self) -> None:
        raw = {"a": [{"b": [{"env": [{"name": "X", "value": "s3cret"}]}]}]}
        assert "s3cret" not in repr(scrub_k8s_fields(raw))

    def test_ordinary_fields_survive(self) -> None:
        raw = {"phase": "Running", "restartCount": 3, "ready": True}
        assert scrub_k8s_fields(raw) == raw


# --------------------------------------------------------------------------- #
# Namespace enforcement
# --------------------------------------------------------------------------- #


class TestNamespaceEnforcement:
    @pytest.mark.usefixtures("allow_coder_only")
    async def test_namespace_outside_allowlist_is_refused(self, fake_kube: Any) -> None:
        result = await dispatch("list_pods", {"namespace": "kube-system"})
        assert result.ok is False
        assert result.content["error"] == "namespace_denied"
        # And crucially: the API was never called.
        assert fake_kube.core.calls == []

    @pytest.mark.usefixtures("allow_coder_only")
    async def test_glob_namespace_is_permitted(self, fake_kube: Any) -> None:
        fake_kube.core._responses["list_namespaced_pod"] = ns(items=[], metadata=ns(_continue=None))
        result = await dispatch("list_pods", {"namespace": "jhub-astro"})
        assert result.ok is True

    @pytest.mark.usefixtures("allow_all_namespaces")
    async def test_denylist_wins(self, fake_kube: Any) -> None:
        """The wildcard policy allows '*' but explicitly denies kube-system."""
        result = await dispatch("list_pods", {"namespace": "kube-system"})
        assert result.ok is False
        assert result.content["error"] == "namespace_denied"
        assert fake_kube.core.calls == []

    @pytest.mark.usefixtures("allow_all_namespaces")
    async def test_denial_message_names_the_denylist(self, fake_kube: Any) -> None:
        """A denylisted namespace must not read as a missing allowlist entry --
        the two need different fixes from whoever edits the ConfigMap."""
        result = await dispatch("list_pods", {"namespace": "kube-system"})
        assert "namespaces_denied" in result.content["message"]
        assert "not in the configured allowlist" not in result.content["message"]

    @pytest.mark.usefixtures("allow_coder_only")
    async def test_out_of_scope_message_points_at_the_allowlist(self, fake_kube: Any) -> None:
        result = await dispatch("list_pods", {"namespace": "some-tenant"})
        assert "not in the configured allowlist" in result.content["message"]
        assert "namespaces_denied" not in result.content["message"]

    @pytest.mark.parametrize("tool", ["list_pods", "get_events"])
    @pytest.mark.usefixtures("allow_coder_only")
    async def test_every_namespaced_tool_checks(self, fake_kube: Any, tool: str) -> None:
        result = await dispatch(tool, {"namespace": "rook-ceph"})
        assert result.content["error"] == "namespace_denied"
        assert fake_kube.core.calls == []


# --------------------------------------------------------------------------- #
# list_pods
# --------------------------------------------------------------------------- #


@allow_all
class TestListPods:
    async def test_summarises_pods(self, fake_kube: Any) -> None:
        fake_kube.core._responses["list_namespaced_pod"] = ns(
            items=[make_pod(), make_pod(name="coder-1", ready=False, restarts=5)],
            metadata=ns(_continue=None),
        )
        out = await list_pods(ListPodsArgs(namespace="coder"))
        assert out["returned"] == 2
        assert out["pods"][0]["ready"] == "1/1"
        assert out["pods"][1]["ready"] == "0/1"
        assert out["pods"][1]["restarts"] == 5
        assert out["pods"][0]["node"] == "node-gpu-04"

    async def test_no_env_or_managed_fields_leak(self, fake_kube: Any) -> None:
        fake_kube.core._responses["list_namespaced_pod"] = ns(
            items=[make_pod()], metadata=ns(_continue=None)
        )
        blob = repr(await list_pods(ListPodsArgs(namespace="coder")))
        assert "managed_fields" not in blob and "managedFields" not in blob
        assert "coder-tls" not in blob
        assert "last-applied-configuration" not in blob

    async def test_selectors_are_forwarded(self, fake_kube: Any) -> None:
        fake_kube.core._responses["list_namespaced_pod"] = ns(items=[], metadata=ns(_continue=None))
        await list_pods(ListPodsArgs(namespace="coder", label_selector="app=coder", limit=10))
        kwargs = fake_kube.core.kwargs_for("list_namespaced_pod")
        assert kwargs["label_selector"] == "app=coder"
        assert kwargs["limit"] == 10

    async def test_limit_above_max_is_rejected_not_clamped(self, fake_kube: Any) -> None:
        result = await dispatch("list_pods", {"namespace": "coder", "limit": 5000})
        assert result.ok is False
        assert result.content["error"] == "invalid_arguments"
        assert "limit" in result.content["message"]
        assert fake_kube.core.calls == []

    async def test_forbidden_is_reported_as_intentional(self, fake_kube: Any) -> None:
        fake_kube.core._responses["list_namespaced_pod"] = ApiException(403, "Forbidden")
        result = await dispatch("list_pods", {"namespace": "coder"})
        assert result.ok is False
        assert result.content["error"] == "forbidden"
        assert "not something to work around" in result.content["message"]


# --------------------------------------------------------------------------- #
# get_pod_status
# --------------------------------------------------------------------------- #


@allow_all
class TestGetPodStatus:
    async def test_reports_previous_termination(self, fake_kube: Any) -> None:
        fake_kube.core._responses["read_namespaced_pod"] = make_pod(
            ready=False,
            restarts=4,
            waiting_reason="CrashLoopBackOff",
            terminated_reason="OOMKilled",
        )
        out = await get_pod_status(GetPodStatusArgs(namespace="coder", name="coder-0"))
        container = out["containers"][0]
        assert container["current"]["reason"] == "CrashLoopBackOff"
        assert container["previous"]["reason"] == "OOMKilled"
        assert container["previous"]["exit_code"] == 137
        assert container["restart_count"] == 4
        assert out["qos_class"] == "Burstable"
        assert out["scheduled"] == "True"

    async def test_running_pod_has_no_previous_key(self, fake_kube: Any) -> None:
        fake_kube.core._responses["read_namespaced_pod"] = make_pod()
        out = await get_pod_status(GetPodStatusArgs(namespace="coder", name="coder-0"))
        assert "previous" not in out["containers"][0]

    async def test_annotations_lose_last_applied_configuration(self, fake_kube: Any) -> None:
        fake_kube.core._responses["read_namespaced_pod"] = make_pod()
        out = await get_pod_status(GetPodStatusArgs(namespace="coder", name="coder-0"))
        assert out["annotations"] == {"coder.com/session": "abc"}

    async def test_volumes_are_names_and_types_only(self, fake_kube: Any) -> None:
        fake_kube.core._responses["read_namespaced_pod"] = make_pod()
        out = await get_pod_status(GetPodStatusArgs(namespace="coder", name="coder-0"))
        assert out["volumes"] == [{"name": "tls", "type": "secret"}]
        assert "coder-tls" not in repr(out)

    async def test_not_found_is_structured(self, fake_kube: Any) -> None:
        fake_kube.core._responses["read_namespaced_pod"] = ApiException(404, "pods 'x' not found")
        result = await dispatch("get_pod_status", {"namespace": "coder", "name": "x"})
        assert result.ok is False
        assert result.content["error"] == "not_found"


# --------------------------------------------------------------------------- #
# get_logs
# --------------------------------------------------------------------------- #


@allow_all
class TestGetLogs:
    async def test_returns_lines(self, fake_kube: Any) -> None:
        fake_kube.core._responses["read_namespaced_pod_log"] = "line one\nline two\n"
        out = await get_logs(GetLogsArgs(namespace="coder", pod="coder-0"))
        assert out["lines"] == ["line one", "line two"]
        assert out["line_count"] == 2
        assert out["truncated"] is False

    async def test_tail_lines_above_1000_is_rejected(self, fake_kube: Any) -> None:
        result = await dispatch(
            "get_logs", {"namespace": "coder", "pod": "coder-0", "tail_lines": 5000}
        )
        assert result.ok is False
        assert result.content["error"] == "invalid_arguments"
        assert fake_kube.core.calls == []

    async def test_tail_lines_at_the_boundary_is_accepted(self, fake_kube: Any) -> None:
        fake_kube.core._responses["read_namespaced_pod_log"] = "ok"
        result = await dispatch(
            "get_logs", {"namespace": "coder", "pod": "coder-0", "tail_lines": 1000}
        )
        assert result.ok is True

    async def test_previous_is_forwarded(self, fake_kube: Any) -> None:
        fake_kube.core._responses["read_namespaced_pod_log"] = ""
        await get_logs(GetLogsArgs(namespace="coder", pod="coder-0", previous=True))
        assert fake_kube.core.kwargs_for("read_namespaced_pod_log")["previous"] is True

    async def test_huge_logs_are_truncated_to_the_tail(self, fake_kube: Any) -> None:
        fake_kube.core._responses["read_namespaced_pod_log"] = (
            "x" * (k8s_tools.MAX_LOG_BYTES + 5_000) + "\nTHE-INTERESTING-LAST-LINE"
        )
        out = await get_logs(GetLogsArgs(namespace="coder", pod="coder-0"))
        assert out["truncated"] is True
        assert out["lines"][-1] == "THE-INTERESTING-LAST-LINE"
        assert len("\n".join(out["lines"]).encode()) <= k8s_tools.MAX_LOG_BYTES

    async def test_credentials_in_logs_are_redacted_by_dispatch(self, fake_kube: Any) -> None:
        """pods/log is the exfiltration channel the redaction layer exists for."""
        fake_kube.core._responses["read_namespaced_pod_log"] = (
            "connecting to postgres://coder:hunter2hunter2@db:5432/coder"
        )
        result = await dispatch("get_logs", {"namespace": "coder", "pod": "coder-0"})
        assert result.ok is True
        assert "hunter2hunter2" not in result.json
        assert "uri_credentials" in result.redaction_rules


# --------------------------------------------------------------------------- #
# get_events
# --------------------------------------------------------------------------- #


@allow_all
class TestGetEvents:
    def _events(self) -> Any:
        return ns(
            items=[
                ns(
                    type="Warning",
                    reason="FailedScheduling",
                    involved_object=ns(kind="Pod", name="coder-0"),
                    count=12,
                    last_timestamp=ago(120),
                    event_time=None,
                    first_timestamp=ago(600),
                    message="0/412 nodes are available: 8 Insufficient nvidia.com/gpu",
                ),
                ns(
                    type="Normal",
                    reason="Pulled",
                    involved_object=ns(kind="Pod", name="coder-1"),
                    count=1,
                    last_timestamp=ago(50_000),
                    event_time=None,
                    first_timestamp=None,
                    message="stale event",
                ),
            ]
        )

    async def test_filters_by_age_and_sorts_newest_first(self, fake_kube: Any) -> None:
        fake_kube.core._responses["list_namespaced_event"] = self._events()
        out = await get_events(GetEventsArgs(namespace="coder", since_seconds=3600))
        assert out["returned"] == 1
        assert out["events"][0]["reason"] == "FailedScheduling"
        assert "stale event" not in repr(out)

    async def test_involved_object_becomes_a_field_selector(self, fake_kube: Any) -> None:
        fake_kube.core._responses["list_namespaced_event"] = ns(items=[])
        await get_events(GetEventsArgs(namespace="coder", involved_object="coder-0"))
        kwargs = fake_kube.core.kwargs_for("list_namespaced_event")
        assert kwargs["field_selector"] == "involvedObject.name=coder-0"


# --------------------------------------------------------------------------- #
# get_workload
# --------------------------------------------------------------------------- #


@allow_all
class TestGetWorkload:
    async def test_deployment_replica_health(self, fake_kube: Any) -> None:
        fake_kube.apps._responses["read_namespaced_deployment"] = ns(
            metadata=ns(generation=9, creation_timestamp=ago(86400)),
            spec=ns(
                replicas=3,
                template=ns(
                    spec=ns(
                        containers=[ns(name="coder", image="nrp/coder:v2.9.1")],
                        image_pull_secrets=[ns(name="regcred")],
                    )
                ),
            ),
            status=ns(
                replicas=3,
                ready_replicas=1,
                available_replicas=1,
                updated_replicas=3,
                observed_generation=8,
                conditions=[
                    ns(
                        type="Available",
                        status="False",
                        reason="MinimumReplicasUnavailable",
                        message="Deployment does not have minimum availability.",
                        last_transition_time=ago(300),
                    )
                ],
            ),
        )
        out = await get_workload(
            GetWorkloadArgs(kind="deployment", namespace="coder", name="coder")
        )
        assert out["replicas"] == {
            "desired": 3,
            "current": 3,
            "ready": 1,
            "available": 1,
            "updated": 3,
        }
        assert out["rollout_in_progress"] is True
        assert out["conditions"][0]["reason"] == "MinimumReplicasUnavailable"
        assert "regcred" not in repr(out)

    async def test_daemonset_uses_its_own_status_fields(self, fake_kube: Any) -> None:
        fake_kube.apps._responses["read_namespaced_daemon_set"] = ns(
            metadata=ns(generation=2, creation_timestamp=ago(3600)),
            spec=ns(replicas=None, template=ns(spec=ns(containers=[]))),
            status=ns(
                desired_number_scheduled=412,
                current_number_scheduled=410,
                number_ready=404,
                number_available=404,
                updated_number_scheduled=412,
                number_misscheduled=0,
                observed_generation=2,
                conditions=[],
            ),
        )
        out = await get_workload(
            GetWorkloadArgs(kind="daemonset", namespace="coder", name="node-exporter")
        )
        assert out["replicas"]["desired"] == 412
        assert out["replicas"]["ready"] == 404
        assert out["rollout_in_progress"] is False

    async def test_unknown_kind_is_rejected(self, fake_kube: Any) -> None:
        result = await dispatch(
            "get_workload", {"kind": "cronjob", "namespace": "coder", "name": "x"}
        )
        assert result.ok is False
        assert result.content["error"] == "invalid_arguments"
        assert fake_kube.apps.calls == []
        # A CronJob is not a workload kind here -- it has its own tool, because
        # it has no replicas and no observedGeneration to report.
        assert "get_cronjob" in registry()


@allow_all
class TestGetCronJob:
    """`get_workload(kind='cronjob')` was the only thing on offer and it just
    returned invalid_arguments, so "did the docs sync run?" was unanswerable."""

    def _cronjob(self, **status: Any) -> Any:
        base = {"last_schedule_time": ago(600), "last_successful_time": None, "active": []}
        base.update(status)
        return ns(
            metadata=ns(uid="uid-1", creation_timestamp=ago(86400)),
            spec=ns(
                schedule="17 * * * *",
                suspend=False,
                concurrency_policy="Forbid",
                job_template=ns(
                    spec=ns(
                        template=ns(
                            spec=ns(
                                restart_policy="Never",
                                containers=[ns(name="docs-sync", image="nrp/agent:1")],
                            )
                        )
                    )
                ),
            ),
            status=ns(**base),
        )

    def _job(self, name: str, *, uid: str = "uid-1", **status: Any) -> Any:
        base: dict[str, Any] = {"active": 0, "succeeded": 0, "failed": 0, "conditions": []}
        base.update(status)
        return ns(
            metadata=ns(
                name=name,
                creation_timestamp=ago(600),
                owner_references=[ns(uid=uid, kind="CronJob")],
            ),
            status=ns(**base),
        )

    async def test_reports_schedule_and_failed_runs(self, fake_kube: Any) -> None:
        fake_kube.batch._responses["read_namespaced_cron_job"] = self._cronjob()
        fake_kube.batch._responses["list_namespaced_job"] = ns(
            items=[
                self._job(
                    "docs-sync-29779157",
                    failed=4,
                    conditions=[
                        ns(
                            type="Failed",
                            status="True",
                            reason="BackoffLimitExceeded",
                            message="Job has reached the specified backoff limit",
                            last_transition_time=ago(300),
                        )
                    ],
                )
            ]
        )
        out = await get_cronjob(GetCronJobArgs(namespace="coder", name="docs-sync"))

        assert out["schedule"] == "17 * * * *"
        assert out["suspended"] is False
        assert out["last_successful_time"] is None
        assert out["recent_jobs"][0]["failed"] == 4
        assert out["recent_jobs"][0]["conditions"][0]["reason"] == "BackoffLimitExceeded"
        # Surfaced so "the logs are gone" has a visible cause rather than
        # looking like the pod never existed.
        assert out["restart_policy"] == "Never"

    async def test_only_jobs_this_cronjob_owns_are_reported(self, fake_kube: Any) -> None:
        """Name prefixes are not ownership: an unrelated Job can be named to
        look like a child, and the owner reference is what actually decides."""
        fake_kube.batch._responses["read_namespaced_cron_job"] = self._cronjob()
        fake_kube.batch._responses["list_namespaced_job"] = ns(
            items=[
                self._job("docs-sync-29779157", succeeded=1),
                self._job("docs-sync-imposter", uid="uid-other", succeeded=1),
            ]
        )
        out = await get_cronjob(GetCronJobArgs(namespace="coder", name="docs-sync"))
        assert [j["name"] for j in out["recent_jobs"]] == ["docs-sync-29779157"]

    async def test_namespace_allowlist_applies(self, fake_kube: Any, allow_coder_only: Any) -> None:
        result = await dispatch("get_cronjob", {"namespace": "kube-system", "name": "x"})
        assert result.content["error"] == "namespace_denied"
        assert fake_kube.batch.calls == []


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


def make_node(name: str, ready: str = "True", memory_pressure: str = "False") -> Any:
    return ns(
        metadata=ns(name=name, labels={"kubernetes.io/hostname": name}),
        spec=ns(
            unschedulable=False, taints=[ns(key="nvidia.com/gpu", value="", effect="NoSchedule")]
        ),
        status=ns(
            conditions=[
                ns(type="Ready", status=ready, reason="KubeletReady"),
                ns(type="MemoryPressure", status=memory_pressure, reason=None),
            ],
            capacity={"cpu": "128", "nvidia.com/gpu": "8"},
            allocatable={"cpu": "126", "nvidia.com/gpu": "8"},
            node_info=ns(kubelet_version="v1.31.4"),
        ),
    )


@allow_all
class TestNodes:
    async def test_list_nodes_returns_only_unhealthy_by_default(self, fake_kube: Any) -> None:
        fake_kube.core._responses["list_node"] = ns(
            items=[
                make_node("healthy-1"),
                make_node("sick-1", ready="Unknown"),
                make_node("sick-2", memory_pressure="True"),
            ]
        )
        out = await list_nodes(ListNodesArgs())
        assert out["total_nodes"] == 3
        assert out["unhealthy_count"] == 2
        assert {n["name"] for n in out["nodes"]} == {"sick-1", "sick-2"}

    async def test_list_nodes_can_return_everything(self, fake_kube: Any) -> None:
        fake_kube.core._responses["list_node"] = ns(items=[make_node("a"), make_node("b")])
        out = await list_nodes(ListNodesArgs(only_unhealthy=False))
        assert out["returned"] == 2

    async def test_list_nodes_omits_the_label_inventory(self, fake_kube: Any) -> None:
        fake_kube.core._responses["list_node"] = ns(items=[make_node("a", ready="False")])
        out = await list_nodes(ListNodesArgs())
        assert "kubernetes.io/hostname" not in repr(out)

    async def test_get_node_includes_capacity_and_taints(self, fake_kube: Any) -> None:
        fake_kube.core._responses["read_node"] = make_node("node-gpu-04")
        out = await get_node(GetNodeArgs(name="node-gpu-04"))
        assert out["capacity"]["nvidia.com/gpu"] == "8"
        assert out["allocatable"]["cpu"] == "126"
        assert out["taints"][0]["key"] == "nvidia.com/gpu"
        assert out["kubelet_version"] == "v1.31.4"

    async def test_node_tools_need_no_namespace(self, fake_kube: Any) -> None:
        """Nodes are cluster-scoped, so the namespace allowlist does not apply --
        but they also expose nothing namespace-specific."""
        fake_kube.core._responses["read_node"] = make_node("n1")
        result = await dispatch("get_node", {"name": "n1"})
        assert result.ok is True


# --------------------------------------------------------------------------- #
# Registry-level invariants
# --------------------------------------------------------------------------- #


class TestRegistryInvariants:
    def test_untrusted_output_is_flagged_on_the_right_tools(self) -> None:
        specs = registry()
        assert specs["get_logs"].untrusted_output is True
        assert specs["get_events"].untrusted_output is True
        assert specs["get_pod_status"].untrusted_output is True
        assert specs["get_workload"].untrusted_output is False

    def test_there_is_no_secret_or_configmap_tool(self) -> None:
        names = set(registry())
        for forbidden in ("get_secret", "list_secrets", "get_configmap", "list_configmaps"):
            assert forbidden not in names

    def test_every_tool_forbids_extra_arguments(self) -> None:
        for name, spec in registry().items():
            schema = spec.input_model.model_json_schema()
            assert schema.get("additionalProperties") is False, name

    @pytest.mark.usefixtures("allow_all_namespaces")
    async def test_unknown_tool_is_refused(self) -> None:
        result = await dispatch("kubectl", {"args": "get secrets -A"})
        assert result.ok is False
        assert result.content["error"] == "unknown_tool"
