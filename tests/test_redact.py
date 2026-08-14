"""Table-driven tests for the redaction chokepoint.

Positive cases assert the credential is gone *and* that the right rule name was
reported. Negative cases assert ordinary cluster output survives untouched --
over-redaction that mangles pod names or image digests would make the agent
useless, so both directions matter.
"""

from __future__ import annotations

import pytest

from nrp_ops_agent.redact import RULE_NAMES, redact, redact_structure

# A structurally valid but entirely fabricated ServiceAccount-shaped JWT.
FAKE_JWT = (
    "eyJhbGciOiJSUzI1NiIsImtpZCI6IkZBS0UifQ"
    ".eyJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6Y29kZXI6Y29kZXIifQ"
    ".QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbg"
)

POSITIVE: list[tuple[str, str, str]] = [
    (
        "jwt",
        f'time=12:00 level=debug msg="using token {FAKE_JWT}"',
        "eyJhbGciOiJSUzI1NiIsImtpZCI6IkZBS0UifQ",
    ),
    (
        "bearer_token",
        "GET /api/v1/pods Authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz012345",
        "sk-abcdefghijklmnopqrstuvwxyz012345",
    ),
    (
        "basic_auth_header",
        "authorization: Basic YWRtaW46aHVudGVyMjAyMjIy",
        "YWRtaW46aHVudGVyMjAyMjIy",
    ),
    (
        "private_key_block",
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA1234\nabcd\n"
        "-----END RSA PRIVATE KEY-----",
        "MIIEowIBAAKCAQEA1234",
    ),
    (
        "private_key_block",
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "b3BlbnNzaC1rZXktdjEA\n"
        "-----END OPENSSH PRIVATE KEY-----",
        "b3BlbnNzaC1rZXktdjEA",
    ),
    (
        "dockerconfigjson",
        '{".dockerconfigjson":"eyJhdXRocyI6e319QUJDREVG"}',
        "eyJhdXRocyI6e319QUJDREVG",
    ),
    (
        "docker_auth_entry",
        '{"auths":{"gitlab.nrp-nautilus.io":{"auth":"cm9vdDpodW50ZXIyMjIy"}}}',
        "cm9vdDpodW50ZXIyMjIy",
    ),
    (
        "htpasswd_line",
        "operator:$apr1$Zx91kQ2p$Q1mBqM0oQZ4l8xTt0dK9r1",
        "$apr1$Zx91kQ2p$Q1mBqM0oQZ4l8xTt0dK9r1",
    ),
    (
        "uri_credentials",
        "could not connect to postgres://coder:sup3rs3cret@db.coder.svc:5432/coder",
        "sup3rs3cret",
    ),
    ("slack_token", "SLACK_BOT_TOKEN was xoxb-1234567890-abcdefghij", "xoxb-1234567890-abcdefghij"),
    ("slack_app_token", "xapp-1-A012345-9876543210-abcdef", "xapp-1-A012345-9876543210-abcdef"),
    ("github_token", "cloning with ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123", "ghp_ABCDEFGHIJKLMNOPQRST"),
    (
        "github_pat",
        "token github_pat_11ABCDEFG0abcdefghijklmnopqrstuvwxyz012345",
        "github_pat_11ABCDEFG0",
    ),
    ("gitlab_token", "registry login glpat-AbCdEfGhIjKlMnOpQrSt", "glpat-AbCdEfGhIjKlMnOpQrSt"),
    ("aws_access_key_id", "s3 backend key AKIAIOSFODNN7EXAMPLE failed", "AKIAIOSFODNN7EXAMPLE"),
    (
        "aws_secret_access_key",
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    ),
    (
        "credential_assignment",
        'ceph mount failed: password="rAdOsGwSecret99"',
        "rAdOsGwSecret99",
    ),
    (
        "credential_assignment",
        "启动失败 api_key: nrp-llm-abcdef123456",
        "nrp-llm-abcdef123456",
    ),
]

# Ordinary Kubernetes/Nautilus output that must pass through byte-identical.
NEGATIVE: list[str] = [
    "pod coder-7d9f8b6c4-x2klm restarted 3 times (CrashLoopBackOff)",
    "image nrp/coder@sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "Normal Scheduled  Successfully assigned coder/coder-0 to node-gpu-04.nrp-nautilus.io",
    "mountPath: /var/run/secrets/kubernetes.io/serviceaccount",
    "secretName: coder-tls",
    "the per-user token bucket refilled at 12:00",
    "https://nrp.ai/documentation/userdocs/coder/deploy",
    'level=info msg="starting server" addr=:8080 replicas=3',
    "Warning FailedScheduling 0/412 nodes are available: 8 Insufficient nvidia.com/gpu",
    "MountVolume.SetUp failed for volume ceph-fs : rpc error: code = Internal",
    "",
]


@pytest.mark.parametrize(("rule", "text", "secret"), POSITIVE, ids=[c[1][:40] for c in POSITIVE])
def test_positive_cases_are_redacted(rule: str, text: str, secret: str) -> None:
    scrubbed, hits = redact(text)
    assert secret not in scrubbed, f"{rule}: secret survived redaction"
    assert rule in hits, f"{rule}: expected rule not reported, got {hits}"
    assert f"[REDACTED:{rule}]" in scrubbed


@pytest.mark.parametrize("text", NEGATIVE, ids=[t[:40] or "empty" for t in NEGATIVE])
def test_negative_cases_are_untouched(text: str) -> None:
    scrubbed, hits = redact(text)
    assert scrubbed == text
    assert hits == []


def test_surrounding_context_is_preserved() -> None:
    """A redacted line must stay diagnostically useful."""
    scrubbed, _ = redact("could not connect to postgres://coder:sup3rs3cret@db.coder.svc:5432/x")
    assert scrubbed.startswith("could not connect to postgres://coder:")
    assert scrubbed.endswith("@db.coder.svc:5432/x")


@pytest.mark.parametrize(("_rule", "text", "_secret"), POSITIVE, ids=[c[0] for c in POSITIVE])
def test_redaction_is_idempotent(_rule: str, text: str, _secret: str) -> None:
    """Running the scrubber twice must not re-match its own placeholders.

    Both chokepoints (tool output, outbound Slack text) can see the same string,
    so a non-idempotent rule would double-redact and report phantom hits.
    """
    once, first_hits = redact(text)
    twice, second_hits = redact(once)
    assert twice == once
    assert second_hits == []
    assert first_hits


def test_rule_names_are_unique() -> None:
    assert len(set(RULE_NAMES)) == len(RULE_NAMES)


def test_multiple_rules_in_one_blob() -> None:
    blob = f"Bearer {FAKE_JWT}\nAKIAIOSFODNN7EXAMPLE\npassword=hunter2222"
    scrubbed, hits = redact(blob)
    assert set(hits) == {"bearer_token", "aws_access_key_id", "credential_assignment"}
    assert "eyJ" not in scrubbed
    assert "AKIA" not in scrubbed
    assert "hunter2222" not in scrubbed


class TestRedactStructure:
    def test_nested_values_are_scrubbed(self) -> None:
        obj = {
            "pod": "coder-0",
            "containers": [
                {"name": "coder", "log": f"token {FAKE_JWT}"},
                {"name": "sidecar", "log": "healthy"},
            ],
        }
        scrubbed, hits = redact_structure(obj)
        assert hits == ["jwt"]
        assert scrubbed["containers"][0]["log"] == "token [REDACTED:jwt]"
        assert scrubbed["containers"][1]["log"] == "healthy"
        assert scrubbed["pod"] == "coder-0"

    def test_dict_keys_are_scrubbed(self) -> None:
        scrubbed, hits = redact_structure({f"Bearer {FAKE_JWT}": 1})
        assert "bearer_token" in hits
        assert "eyJ" not in next(iter(scrubbed))

    def test_non_string_scalars_survive(self) -> None:
        obj = {"restarts": 7, "ready": False, "ts": None, "ratio": 0.5}
        scrubbed, hits = redact_structure(obj)
        assert scrubbed == obj
        assert hits == []

    def test_hits_are_deduplicated_in_order(self) -> None:
        obj = ["password=aaaaaaaa", f"tok {FAKE_JWT}", "password=bbbbbbbb"]
        _, hits = redact_structure(obj)
        assert hits == ["credential_assignment", "jwt"]
