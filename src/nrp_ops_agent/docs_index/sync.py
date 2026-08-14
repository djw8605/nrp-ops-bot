"""Clone the NRP docs site, chunk its MDX, and build a SQLite FTS5 index.

Run by ``deploy/cronjob-docs-sync.yaml`` every 6 hours into a PVC.

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
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from nrp_ops_agent.config import get_settings

DOCS_SUBDIR: Final = Path("src/content/docs")
SCHEMA_VERSION: Final = 1

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

    TODO: verify -- Starlight slug casing. This lowercases the whole path, which
    matches the known ``Documentation/`` -> ``/documentation/`` mapping, but a
    page whose filename is genuinely mixed-case would get the wrong URL.
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


def build_index(chunks: list[Chunk], db_path: Path) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(db_path.suffix + ".building")
    tmp_path.unlink(missing_ok=True)

    conn = sqlite3.connect(tmp_path)
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
    finally:
        conn.close()

    # Atomic swap: readers either see the old index or the new one, never a
    # half-built one. The CronJob and the agent share this file on a PVC.
    tmp_path.replace(db_path)
    return len(chunks)


def clone_docs(repo_url: str, dest: Path, ref: str | None = None) -> Path:
    cmd = ["git", "clone", "--depth", "1", "--single-branch"]
    if ref:
        cmd += ["--branch", ref]
    cmd += [repo_url, str(dest)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)  # noqa: S603
    return dest


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

    tmpdir: str | None = None
    try:
        if local_checkout is not None:
            checkout = local_checkout
        else:
            tmpdir = tempfile.mkdtemp(prefix="nrp-docs-")
            checkout = clone_docs(repo_url, Path(tmpdir) / "nrp-site")

        docs_root = checkout / DOCS_SUBDIR
        if not docs_root.is_dir():
            raise FileNotFoundError(f"{DOCS_SUBDIR} not found in checkout at {checkout}")

        chunks = list(iter_chunks(docs_root, site_base_url))
        if not chunks:
            raise RuntimeError(
                "docs checkout produced zero chunks; refusing to publish an empty index"
            )
        return build_index(chunks, db_path)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


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
    args = parser.parse_args(argv)

    count = sync(
        repo_url=args.repo_url,
        db_path=args.db_path,
        site_base_url=args.site_base_url,
        local_checkout=args.local_checkout,
    )
    print(f"indexed {count} chunks", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
