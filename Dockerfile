# Two stages so the build toolchain never reaches the runtime image.
#
# The runtime constraints come from deploy/deployment.yaml, which runs this with
# runAsNonRoot: true, runAsUser: 65532, readOnlyRootFilesystem: true and all
# capabilities dropped. Anything here that expects to write outside /tmp, or to
# run as root, fails at admission rather than at build time -- so the choices
# below are load-bearing, not stylistic.

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only what the wheel needs. playbooks/ is force-included into the package
# by pyproject.toml, so it must be present at build time even though it lives
# outside src/.
COPY pyproject.toml README.md ./
COPY src ./src
COPY playbooks ./playbooks

RUN pip install .


FROM python:3.12-slim

# git is a runtime dependency, not a build one: nrp-ops-docs-sync clones the
# docs repository on every CronJob run. ca-certificates is needed for the HTTPS
# clone and for the Prometheus/LLM calls.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# HOME=/tmp because both workloads mount an emptyDir at /tmp and it is the only
# writable path under readOnlyRootFilesystem. git resolves HOME even for an
# anonymous clone, so pointing it there keeps the docs-sync job from failing on
# a read-only filesystem.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

# 65532 ("nonroot") matches runAsUser in the Deployment. Declared here as well
# so the image is correct on its own, rather than only under that one manifest.
USER 65532:65532

# No ENTRYPOINT on purpose: both workloads select their console script through
# `args` (nrp-ops-agent, nrp-ops-docs-sync), which Kubernetes maps to the
# container command. Setting an ENTRYPOINT here would silently turn those into
# arguments to it.
CMD ["nrp-ops-agent"]
