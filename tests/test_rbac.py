"""Tests that assert the ClusterRole actually encodes the security invariants.

This is the "verify the ServiceAccount genuinely cannot read secrets" step. A
live cluster is not available in CI, so the verification is done two ways that
together are stronger than a single ``kubectl auth can-i``:

1. **The grant.** Parse ``deploy/clusterrole.yaml`` and assert that no forbidden
   resource and no write verb appears anywhere in it.
2. **The demand.** Scan the tool source for every Kubernetes API method it calls,
   map those back to resources, and assert the ClusterRole grants each one --
   and, in the other direction, that no tool calls a method for a resource the
   role withholds. Adding ``read_namespaced_secret`` to a tool fails this test
   even though the ClusterRole is untouched.

``test_live_cluster_denies_secrets`` does the real SelfSubjectAccessReview and is
skipped unless ``NRP_OPS_TEST_LIVE_CLUSTER=1``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTERROLE = REPO_ROOT / "deploy" / "clusterrole.yaml"
K8S_TOOLS = REPO_ROOT / "src" / "nrp_ops_agent" / "tools" / "k8s.py"

WRITE_VERBS = {"create", "update", "patch", "delete", "deletecollection", "*"}

FORBIDDEN_RESOURCES = {
    "secrets",
    "configmaps",
    "pods/exec",
    "pods/attach",
    "pods/portforward",
    "serviceaccounts/token",
    "*",
}


@pytest.fixture(scope="module")
def rules() -> list[dict[str, Any]]:
    doc = yaml.safe_load(CLUSTERROLE.read_text(encoding="utf-8"))
    assert doc["kind"] == "ClusterRole"
    return list(doc["rules"])


class TestClusterRoleGrant:
    def test_no_write_verbs_anywhere(self, rules: list[dict[str, Any]]) -> None:
        for rule in rules:
            offending = {v.lower() for v in rule["verbs"]} & WRITE_VERBS
            assert not offending, f"write verb {offending} granted on {rule.get('resources')}"

    def test_only_read_verbs_are_granted(self, rules: list[dict[str, Any]]) -> None:
        for rule in rules:
            assert set(rule["verbs"]) <= {"get", "list", "watch"}, rule

    @pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_RESOURCES))
    def test_forbidden_resources_are_absent(
        self, rules: list[dict[str, Any]], forbidden: str
    ) -> None:
        for rule in rules:
            assert forbidden not in rule["resources"], (
                f"{forbidden} must never be granted; found in rule {rule}"
            )

    def test_no_wildcard_api_groups(self, rules: list[dict[str, Any]]) -> None:
        for rule in rules:
            assert "*" not in rule.get("apiGroups", []), rule

    def test_pods_log_is_granted_and_explicitly_documented(
        self, rules: list[dict[str, Any]]
    ) -> None:
        """pods/log is a genuine exfiltration channel, so its presence must be a
        deliberate, commented decision rather than an accident."""
        granted = {r for rule in rules for r in rule["resources"]}
        assert "pods/log" in granted
        text = CLUSTERROLE.read_text(encoding="utf-8")
        assert "exfiltration" in text.lower()
        assert "redact.py" in text


# --------------------------------------------------------------------------- #
# The demand side: what the code actually calls
# --------------------------------------------------------------------------- #

#: kubernetes-asyncio method name -> the RBAC resource it needs.
METHOD_TO_RESOURCE = {
    "list_namespaced_pod": "pods",
    "read_namespaced_pod": "pods",
    "read_namespaced_pod_log": "pods/log",
    "list_namespaced_event": "events",
    "list_node": "nodes",
    "read_node": "nodes",
    "read_namespaced_deployment": "deployments",
    "read_namespaced_stateful_set": "statefulsets",
    "read_namespaced_daemon_set": "daemonsets",
    "read_namespaced_replica_set": "replicasets",
    "read_namespaced_service": "services",
    "read_namespaced_ingress": "ingresses",
    "read_namespaced_cron_job": "cronjobs",
    "list_namespaced_job": "jobs",
}

_CALL = re.compile(r"clients\.(?:core|apps|batch)\.(\w+)|\"(read_namespaced_\w+)\"")


def _called_methods() -> set[str]:
    source = K8S_TOOLS.read_text(encoding="utf-8")
    found: set[str] = set()
    for match in _CALL.finditer(source):
        found.add(match.group(1) or match.group(2))
    return found


class TestToolDemandMatchesGrant:
    def test_every_api_call_maps_to_a_known_resource(self) -> None:
        unknown = _called_methods() - set(METHOD_TO_RESOURCE)
        assert not unknown, (
            f"tools call Kubernetes methods this test cannot classify: {sorted(unknown)}. "
            "Add them to METHOD_TO_RESOURCE and confirm the ClusterRole grants them."
        )

    def test_every_api_call_is_granted_by_the_clusterrole(
        self, rules: list[dict[str, Any]]
    ) -> None:
        granted = {r for rule in rules for r in rule["resources"]}
        for method in sorted(_called_methods()):
            resource = METHOD_TO_RESOURCE[method]
            assert resource in granted, f"{method} needs {resource}, which is not granted"

    def test_no_tool_calls_a_forbidden_resource(self) -> None:
        for method in _called_methods():
            assert METHOD_TO_RESOURCE[method] not in FORBIDDEN_RESOURCES

    def test_no_tool_mutates(self) -> None:
        """No create/patch/replace/delete call may appear in the tool source."""
        source = K8S_TOOLS.read_text(encoding="utf-8")
        for verb in (
            "create_namespaced",
            "patch_namespaced",
            "replace_namespaced",
            "delete_namespaced",
        ):
            assert verb not in source, f"{verb} appears in the read-only tool module"

    def test_no_exec_or_portforward_helpers_are_imported(self) -> None:
        source = K8S_TOOLS.read_text(encoding="utf-8")
        for forbidden in ("stream", "ws_client", "connect_get_namespaced_pod_exec", "portforward"):
            assert forbidden not in source


@pytest.mark.skipif(
    os.environ.get("NRP_OPS_TEST_LIVE_CLUSTER") != "1",
    reason="set NRP_OPS_TEST_LIVE_CLUSTER=1 to run SelfSubjectAccessReview against a real cluster",
)
class TestLiveCluster:
    async def test_live_cluster_denies_secrets(self) -> None:
        """Ask the API server directly whether this identity may read secrets."""
        from kubernetes_asyncio import client as kube_client

        from nrp_ops_agent.tools.k8s import _load_kube_config

        await _load_kube_config()
        api = kube_client.ApiClient()
        auth = kube_client.AuthorizationV1Api(api)
        try:
            for group, resource, verb in [
                ("", "secrets", "get"),
                ("", "secrets", "list"),
                ("", "pods/exec", "create"),
                ("", "pods/portforward", "create"),
                ("", "pods", "delete"),
                ("apps", "deployments", "patch"),
            ]:
                review = kube_client.V1SelfSubjectAccessReview(
                    spec=kube_client.V1SelfSubjectAccessReviewSpec(
                        resource_attributes=kube_client.V1ResourceAttributes(
                            group=group, resource=resource, verb=verb
                        )
                    )
                )
                result = await auth.create_self_subject_access_review(review)
                assert result.status.allowed is False, f"{verb} {resource} was ALLOWED"
        finally:
            await api.close()

    async def test_live_cluster_allows_reading_pods(self) -> None:
        from kubernetes_asyncio import client as kube_client

        from nrp_ops_agent.tools.k8s import _load_kube_config

        await _load_kube_config()
        api = kube_client.ApiClient()
        auth = kube_client.AuthorizationV1Api(api)
        try:
            review = kube_client.V1SelfSubjectAccessReview(
                spec=kube_client.V1SelfSubjectAccessReviewSpec(
                    resource_attributes=kube_client.V1ResourceAttributes(
                        group="", resource="pods", verb="list"
                    )
                )
            )
            result = await auth.create_self_subject_access_review(review)
            assert result.status.allowed is True
        finally:
            await api.close()
