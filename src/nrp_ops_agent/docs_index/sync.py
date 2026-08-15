"""Clone the NRP docs site, chunk its MDX, and build a SQLite FTS5 index.

Runs as the ``docs-sync`` sidecar in ``deploy/deployment.yaml``: one long-lived
container calling :func:`sync_forever`, writing into an ``emptyDir`` shared with
the agent container beside it. It used to be a CronJob writing to an RWX CephFS
PVC, which spent more time broken than working -- the volume root was created
root-owned and the CSI driver never applied ``fsGroup``, so every run died on
"Permission denied" and the index was never built at all. An ``emptyDir`` gets
the pod's ``fsGroup`` unconditionally, with no CSI driver in the path.

The cost of that trade is that the index is rebuilt on every pod start rather
than surviving one, which for ~870 chunks is about a minute of one CPU.

Design notes:

* **Source, not HTML.** Crawling the rendered site loses heading structure and
  reflows code blocks; the MDX keeps both.
* **Chunk by heading.** Each chunk carries its full heading path, so a hit can be
  cited as "Admin Guide > Coder > Upgrades > Rolling back" rather than as a bare
  page.
* **Code blocks survive intact.** Operators search for exact identifiers --
  ``nvidia.com/gpu``, ``rook-ceph-mgr``, a specific kubectl invocation -- and
  those live inside fences. JSX/MDX component tags are stripped around them.
* **Every chunk keeps its published URL** so a reply is verifiable.

Phase 1 ranks with FTS5 BM25 only. :class:`~nrp_ops_agent.tools.docs.Retriever`
is the seam for adding embeddings later; exact-term matching outperforms pure
vector search on a corpus this full of identifiers, so it is not built yet.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from nrp_ops_agent.config import get_settings

DOCS_SUBDIR: Final = Path("src/content/docs")
SCHEMA_VERSION: Final = 1

#: Clone attempts before giving up. The docs mirror being briefly unreachable is
#: a transient the job can absorb; the schedule is hourly, but a run that fails
#: leaves the index a full hour staler for no good reason.
CLONE_ATTEMPTS: Final = 3

#: Seconds between rebuilds when running under ``--every`` with no value given.
REBUILD_INTERVAL: Final = 3600

#: How long to wait after a *failed* rebuild, rather than the full interval. The
#: run that matters most is the first one of a pod's life -- until it lands
#: ``search_docs`` has nothing to answer from -- and sitting out an hour over a
#: docs mirror that blinked is the wrong trade.
RETRY_AFTER_FAILURE: Final = 300

_FRONTMATTER: Final = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_TITLE_LINE: Final = re.compile(r"^title:\s*(.+?)\s*$", re.MULTILINE)
_FENCE: Final = re.compile(r"(^```[^\n]*\n.*?^```[ \t]*$)", re.MULTILINE | re.DOTALL)
_HEADING: Final = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
_IMPORT_LINE: Final = re.compile(r"^\s*import\s+.+?from\s+['\"][^'\"]+['\"];?\s*$", re.MULTILINE)
_EXPORT_LINE: Final = re.compile(r"^\s*export\s+(?:const|let|default)\s.+$", re.MULTILINE)
#: Self-closing or paired JSX component tags: <Card ...>, </Tabs>, <Aside/>.
_JSX_TAG: Final = re.compile(r"</?[A-Z][A-Za-z0-9]*(?:\s[^<>]*?)?/?>")
#: Plain HTML tags Starlight authors sometimes drop inline.
_HTML_TAG: Final = re.compile(r"</?(?:br|hr|div|span|p|img|a|em|strong)\b[^<>]*>")
_MD_LINK: Final = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")

#: Long sections are split on paragraph boundaries rather than truncated --
#: dropping the tail of a runbook is exactly how a reply ends up wrong.
MAX_CHUNK_CHARS: Final = 6_000


@dataclass(frozen=True)
class Chunk:
    path: str
    url: str
    section: str
    page_title: str
    heading_path: str
    body: str


# --------------------------------------------------------------------------- #
# Text processing
# --------------------------------------------------------------------------- #


def split_frontmatter(text: str) -> tuple[str, str]:
    """Return ``(title, body)``, stripping YAML frontmatter."""
    match = _FRONTMATTER.match(text)
    if not match:
        return "", text
    block = match.group(1)
    title_match = _TITLE_LINE.search(block)
    title = title_match.group(1).strip().strip("\"'") if title_match else ""
    return title, text[match.end() :]


def strip_mdx(text: str) -> str:
    """Remove MDX/JSX scaffolding while leaving fenced code blocks byte-identical."""
    parts = _FENCE.split(text)
    out: list[str] = []
    for index, part in enumerate(parts):
        if index % 2 == 1:  # a captured fence
            out.append(part)
            continue
        cleaned = _IMPORT_LINE.sub("", part)
        cleaned = _EXPORT_LINE.sub("", cleaned)
        cleaned = _JSX_TAG.sub("", cleaned)
        cleaned = _HTML_TAG.sub("", cleaned)
        # Keep link text and the target: NRP docs cross-reference heavily and the
        # target path is itself a searchable identifier.
        cleaned = _MD_LINK.sub(r"\1 (\2)", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        out.append(cleaned)
    return "".join(out)


def _mask_fences(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _FENCE.finditer(text)]


def chunk_by_heading(text: str) -> list[tuple[str, str]]:
    """Split into ``(heading_path, body)`` pairs.

    Headings inside fenced blocks are ignored -- a shell comment starting with
    ``#`` is not a section boundary.
    """
    fences = _mask_fences(text)

    def in_fence(pos: int) -> bool:
        return any(start <= pos < end for start, end in fences)

    matches = [m for m in _HEADING.finditer(text) if not in_fence(m.start())]
    if not matches:
        body = text.strip()
        return [("", body)] if body else []

    chunks: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        chunks.append(("", preamble))

    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_path = " > ".join(t for _, t in stack)

        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        chunks.append((heading_path, body))

    return _finalize(chunks)


def _finalize(chunks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop empty sections and split oversized ones.

    Short sections are deliberately *not* merged into their predecessor: a
    two-line "Rolling back" subsection is precisely the answer to a real
    question, and folding it under the previous heading would make the citation
    point at the wrong part of the page.
    """
    out: list[tuple[str, str]] = []
    for heading, body in chunks:
        stripped = body.strip()
        if not stripped:
            # A heading whose content is entirely subsections; those subsections
            # carry it in their own heading path already.
            continue
        for piece in _split_long(stripped):
            out.append((heading, piece))
    return out


def _split_long(body: str) -> list[str]:
    if len(body) <= MAX_CHUNK_CHARS:
        return [body]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for para in body.split("\n\n"):
        if current and size + len(para) > MAX_CHUNK_CHARS:
            pieces.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para) + 2
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def url_for(rel_path: Path, site_base_url: str) -> str:
    """Map a docs source path to its published URL.

    ``src/content/docs/Documentation/admindocs/upgrades/coder.mdx``
    becomes ``https://nrp.ai/documentation/admindocs/upgrades/coder``.

    Verified 2026-08-14 against the live site: indexing the real docs repo
    produced 146 distinct page URLs, every one of which appears in
    https://nrp.ai/sitemap-0.xml. All 234 sitemap entries are lowercase and
    none carries a trailing slash, so the wholesale lowercasing below is right
    and there is no mixed-case filename to get wrong.
    """
    parts = list(rel_path.with_suffix("").parts)
    if parts and parts[-1].lower() == "index":
        parts.pop()
    slug = "/".join(p.lower() for p in parts)
    return f"{site_base_url}/{slug}".rstrip("/")


def section_for(rel_path: Path) -> str:
    lowered = [p.lower() for p in rel_path.parts]
    if "admindocs" in lowered:
        return "admindocs"
    if "userdocs" in lowered:
        return "userdocs"
    return "other"


def iter_chunks(docs_root: Path, site_base_url: str) -> Iterator[Chunk]:
    for path in sorted(docs_root.rglob("*")):
        if path.suffix.lower() not in (".md", ".mdx") or not path.is_file():
            continue
        rel = path.relative_to(docs_root)
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        title, body = split_frontmatter(raw)
        cleaned = strip_mdx(body)
        url = url_for(rel, site_base_url)
        section = section_for(rel)
        for heading_path, chunk_body in chunk_by_heading(cleaned):
            yield Chunk(
                path=str(rel),
                url=url if not heading_path else f"{url}#{_anchor(heading_path)}",
                section=section,
                page_title=title or rel.stem,
                heading_path=heading_path,
                body=chunk_body,
            )


def _anchor(heading_path: str) -> str:
    leaf = heading_path.split(" > ")[-1]
    slug = re.sub(r"[^a-z0-9\s-]", "", leaf.lower()).strip()
    return re.sub(r"\s+", "-", slug)


# --------------------------------------------------------------------------- #
# Index build
# --------------------------------------------------------------------------- #

_DDL: Final = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE chunks (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL,
    url          TEXT NOT NULL,
    section      TEXT NOT NULL,
    page_title   TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    body         TEXT NOT NULL
);
CREATE INDEX chunks_section ON chunks(section);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    page_title, heading_path, body,
    content='chunks', content_rowid='id',
    tokenize="unicode61 tokenchars '-_./'"
);
"""

# Identifiers like nvidia.com/gpu and rook-ceph-mgr must tokenize as one term,
# which is what the tokenchars above buy; the default tokenizer would shred them.


class IndexDirectoryUnwritable(RuntimeError):
    """The index directory exists but this process cannot create files in it."""


def _describe_permissions(directory: Path) -> str:
    """Owner, mode and the identity we are trying to write as.

    `sqlite3.connect` reports every one of these as "unable to open database
    file", which is indistinguishable from a missing path. On a CephFS PVC the
    answer is almost always that the volume root is owned by root with mode
    0755 and fsGroup was never applied, so both halves of the comparison have to
    be in the message for it to be actionable.
    """
    try:
        st = directory.stat()
        owner = f"uid={st.st_uid} gid={st.st_gid} mode={st.st_mode & 0o7777:04o}"
    except OSError as exc:
        owner = f"could not stat ({exc})"
    running = f"uid={os.getuid()} gid={os.getgid()} groups={sorted(os.getgroups())}"
    return f"{directory} is {owner}; this process runs as {running}"


def _probe_writable(directory: Path) -> OSError | None:
    """Actually try to create a file. Returns the error, or None on success.

    ``os.access(W_OK)`` is not used: it answers from the mode bits alone and
    lies under exactly the conditions that matter here -- root squash, a
    read-only mount, and ACLs that disagree with the mode.
    """
    probe = directory / f".docs-sync-write-probe.{os.getpid()}"
    try:
        probe.touch()
    except OSError as exc:
        return exc
    finally:
        with contextlib.suppress(OSError):
            probe.unlink()
    return None


def require_writable_index_dir(db_path: Path) -> None:
    """Make the index directory usable, or explain precisely why it is not.

    Called before the clone rather than after it: a permissions problem is not
    worth a 30-second checkout to discover, and the previous ordering buried the
    real fault under a traceback from deep inside build_index.

    Everything here is best-effort repair before giving up, because the common
    failure is a volume whose root was created by someone else and only needs a
    directory or a mode bit to become usable.
    """
    directory = db_path.parent

    # Create the whole tree, not just the leaf: a subPath mount or a fresh
    # volume can present an empty root with nothing underneath it.
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except FileExistsError as exc:
        raise IndexDirectoryUnwritable(
            f"{directory} exists but is not a directory ({exc}); the mount path is wrong"
        ) from exc
    except OSError as exc:
        raise IndexDirectoryUnwritable(
            f"cannot create the index directory: {_describe_permissions(directory)} ({exc})"
        ) from exc

    failure = _probe_writable(directory)
    if failure is None:
        return

    # Repair attempt: if this process owns the directory, or shares its group,
    # the mode alone may be the problem and is ours to widen. A volume chowned
    # to another uid entirely is not, and falls through to the error below.
    with contextlib.suppress(OSError):
        mode = directory.stat().st_mode & 0o7777
        directory.chmod(mode | 0o770)
        if _probe_writable(directory) is None:
            print(
                f"index directory was not writable; relaxed its mode to "
                f"{directory.stat().st_mode & 0o7777:04o}",
                file=sys.stderr,
                flush=True,
            )
            return

    raise IndexDirectoryUnwritable(
        f"cannot write the docs index: {_describe_permissions(directory)} ({failure}). "
        "The directory is mounted but not writable by this user, and relaxing the mode did "
        "not help, so it belongs to another uid. In the cluster this path is an emptyDir "
        "inside the agent pod and should always be writable, so suspect either a "
        "securityContext whose fsGroup no longer matches runAsUser, or a "
        "NRP_OPS_DOCS_DB_PATH pointing at something else -- a PVC root is root-owned "
        "unless its CSI driver applies fsGroup, which is the failure that moved this "
        "index off a PVC in the first place. Confirm with: kubectl -n system-nrp-ops-bot "
        "exec deploy/nrp-ops-agent -c docs-sync -- ls -lnd /var/lib/nrp-ops-agent"
    )


def build_index(chunks: list[Chunk], db_path: Path) -> int:
    require_writable_index_dir(db_path)

    # A unique name per run. The fixed ".building" path was fine until a run
    # died holding it: a leftover owned by another uid cannot be unlinked, and
    # every subsequent run then failed on a file that had nothing to do with it.
    # Same directory, so the replace below is still atomic.
    tmp_path = db_path.with_suffix(f"{db_path.suffix}.building.{os.getpid()}")
    with contextlib.suppress(OSError):
        db_path.with_suffix(db_path.suffix + ".building").unlink(missing_ok=True)

    try:
        conn = sqlite3.connect(tmp_path)
    except sqlite3.OperationalError as exc:
        # SQLite says "unable to open database file" for a missing directory, a
        # missing file and a denial alike. The preflight above should have
        # caught this; if it did not, do not let the vaguest of the three
        # possible meanings be the whole error.
        raise IndexDirectoryUnwritable(
            f"could not create {tmp_path}: {exc}. {_describe_permissions(db_path.parent)}"
        ) from exc

    try:
        conn.executescript(_DDL)
        conn.executemany(
            "INSERT INTO chunks (path, url, section, page_title, heading_path, body) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(c.path, c.url, c.section, c.page_title, c.heading_path, c.body) for c in chunks],
        )
        conn.execute(
            "INSERT INTO chunks_fts (rowid, page_title, heading_path, body) "
            "SELECT id, page_title, heading_path, body FROM chunks"
        )
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [("schema_version", str(SCHEMA_VERSION)), ("chunk_count", str(len(chunks)))],
        )
        conn.commit()
    except BaseException:
        # Never leave a partial build behind for the next run to trip over.
        conn.close()
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise
    else:
        conn.close()

    # The agent reads this file from the container next door. That is the same
    # uid today, but the two containers are only guaranteed to share the pod's
    # fsGroup, and SQLite creates the file 0600 under the default umask -- which
    # would leave the reader with an index it can see and cannot open.
    with contextlib.suppress(OSError):
        tmp_path.chmod(0o664)

    # Atomic swap: readers either see the old index or the new one, never a
    # half-built one. The CronJob and the agent share this file on a PVC.
    try:
        tmp_path.replace(db_path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise IndexDirectoryUnwritable(
            f"built the index but could not move it into place at {db_path}: {exc}. "
            f"{_describe_permissions(db_path.parent)}"
        ) from exc
    return len(chunks)


class CloneFailed(RuntimeError):
    """git clone failed, carrying git's own stderr rather than just a status."""


def clone_docs(repo_url: str, dest: Path, ref: str | None = None) -> Path:
    """Shallow-clone the docs repo.

    ``check=True`` used to raise ``CalledProcessError``, whose message is only
    "returned non-zero exit status 128" -- git's stderr was captured and then
    thrown away with the exception. In the CronJob that meant a failed run left
    a traceback naming no cause: not the DNS failure, not the TLS error, not the
    404. Everything below exists to put git's own words in the log.
    """
    cmd = ["git", "clone", "--depth", "1", "--single-branch"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [repo_url, str(dest)]

    last = ""
    for attempt in range(1, CLONE_ATTEMPTS + 1):
        print(
            f"cloning {repo_url}" + (f" at {ref}" if ref else "") + f" (attempt {attempt})",
            file=sys.stderr,
            flush=True,
        )
        # git refuses a non-empty destination, so a retry has to start clean.
        with contextlib.suppress(OSError):
            shutil.rmtree(dest, ignore_errors=True)
        try:
            proc = subprocess.run(  # noqa: S603
                cmd, check=False, capture_output=True, timeout=600, text=True
            )
        except subprocess.TimeoutExpired:
            # A drop rather than a refusal: almost always egress policy. A
            # refusal would have come back as a fast non-zero exit instead.
            last = (
                "timed out after 600s with no response. This is what a blocked egress path "
                "looks like -- check the NetworkPolicy allows HTTPS to the docs host, and "
                "that the host resolves."
            )
        else:
            if proc.returncode == 0:
                return dest
            last = f"exit {proc.returncode}:\n" + (
                (proc.stderr or proc.stdout or "").strip() or "(git wrote nothing to stderr)"
            )

        # A docs mirror that is briefly unreachable should cost a retry, not a
        # failed run and an index an hour staler than it needs to be.
        if attempt < CLONE_ATTEMPTS:
            delay = 2**attempt
            print(f"clone failed, retrying in {delay}s: {last}", file=sys.stderr, flush=True)
            time.sleep(delay)

    raise CloneFailed(f"git clone of {repo_url} failed after {CLONE_ATTEMPTS} attempts -- {last}")


def sync(
    *,
    repo_url: str | None = None,
    db_path: Path | None = None,
    site_base_url: str | None = None,
    local_checkout: Path | None = None,
) -> int:
    settings = get_settings()
    repo_url = repo_url or settings.docs_repo_url
    db_path = db_path or settings.docs_db_path
    site_base_url = (site_base_url or settings.docs_site_base_url).rstrip("/")

    # Before the clone: a volume this job cannot write to fails the run whatever
    # the checkout produces, and finding that out first makes the log say so.
    require_writable_index_dir(db_path)

    tmpdir: str | None = None
    try:
        if local_checkout is not None:
            checkout = local_checkout
        else:
            tmpdir = tempfile.mkdtemp(prefix="nrp-docs-")
            checkout = clone_docs(repo_url, Path(tmpdir) / "nrp-site")

        docs_root = checkout / DOCS_SUBDIR
        if not docs_root.is_dir():
            present = sorted(p.name for p in checkout.iterdir())[:20] if checkout.is_dir() else []
            raise FileNotFoundError(
                f"{DOCS_SUBDIR} not found in checkout at {checkout}. Top level holds: "
                f"{', '.join(present) or '(nothing)'}. The docs repo layout may have moved."
            )

        chunks = list(iter_chunks(docs_root, site_base_url))
        if not chunks:
            raise RuntimeError(
                "docs checkout produced zero chunks; refusing to publish an empty index"
            )
        print(f"chunked {len(chunks)} sections, writing {db_path}", file=sys.stderr, flush=True)
        return build_index(chunks, db_path)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def sync_forever(every: float, **kwargs: Any) -> None:
    """Rebuild the index every ``every`` seconds, forever. Never returns.

    This is how docs-sync runs in production now that it is a sidecar rather
    than a CronJob: there is no controller left to re-run it on a schedule, so
    the schedule lives here.

    Nothing in this loop is allowed to be fatal. A CronJob run that failed was
    retried by the Job controller and, at worst, showed up as a failed Job; a
    sidecar that exits takes its restart budget with it and eventually
    CrashLoopBackOffs next to a perfectly healthy agent. A failed rebuild
    therefore logs and waits, and the previous index -- if there is one --
    stays in place and keeps answering.
    """
    while True:
        started = time.monotonic()
        try:
            count = sync(**kwargs)
        # Broad on purpose -- see the docstring: nothing here may be fatal.
        except Exception as exc:
            delay = min(every, RETRY_AFTER_FAILURE)
            print(
                f"docs-sync failed: {type(exc).__name__}: {exc}; retrying in {delay:.0f}s",
                file=sys.stderr,
                flush=True,
            )
        else:
            # Measure from the start of the run, so a rebuild that takes two
            # minutes does not push the next one to interval + 2 minutes and
            # walk the rebuild time around the clock.
            delay = max(0.0, every - (time.monotonic() - started))
            print(
                f"indexed {count} chunks; next rebuild in {delay:.0f}s",
                file=sys.stderr,
                flush=True,
            )
        time.sleep(delay)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the NRP docs FTS5 index.")
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--site-base-url", default=None)
    parser.add_argument(
        "--local-checkout",
        type=Path,
        default=None,
        help="Index an existing nrp-site checkout instead of cloning.",
    )
    parser.add_argument(
        "--every",
        nargs="?",
        type=float,
        const=float(REBUILD_INTERVAL),
        default=None,
        metavar="SECONDS",
        help=(
            f"Rebuild on a loop instead of once, every SECONDS "
            f"(default {REBUILD_INTERVAL}). How the sidecar runs; a bare "
            f"invocation still builds once and exits, which is what the "
            f"development path and the seed command want."
        ),
    )
    args = parser.parse_args(argv)

    if args.every is not None:
        if args.every <= 0:
            parser.error("--every must be greater than zero")
        # Does not return, and does not fail: sync_forever swallows the
        # per-rebuild failures a long-lived container must survive.
        sync_forever(
            args.every,
            repo_url=args.repo_url,
            db_path=args.db_path,
            site_base_url=args.site_base_url,
            local_checkout=args.local_checkout,
        )

    try:
        count = sync(
            repo_url=args.repo_url,
            db_path=args.db_path,
            site_base_url=args.site_base_url,
            local_checkout=args.local_checkout,
        )
    except Exception as exc:
        # The last line of a crashlooping container's log is the one anybody
        # actually reads, so put the reason there rather than at the top of a
        # traceback. Non-zero exit still fails the Job.
        print(f"docs-sync failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1

    print(f"indexed {count} chunks", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
