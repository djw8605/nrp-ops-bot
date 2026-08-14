"""nrp-ops-agent: a read-only Slack operations agent for the NRP/Nautilus cluster.

Security role: this package as a whole is designed to be incapable of mutating
cluster state. Phase 1 exposes read verbs only, enforced in RBAC (see
``deploy/clusterrole.yaml``) and again in code (see ``tools/``). Nothing here may
rely on the language model declining to do something.
"""

__version__ = "0.1.0"
