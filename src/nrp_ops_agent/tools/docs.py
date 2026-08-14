"""NRP documentation search.

Security role: low, but not zero -- the query string reaches SQLite's FTS5 MATCH
parser, so it is tokenised into quoted terms here rather than passed through.
That removes both the injection surface and the class of "unrecognized token"
errors that a model would otherwise burn turns on.

Ranking is BM25 over an index built by :mod:`nrp_ops_agent.docs_index.sync`.
:class:`Retriever` is the seam for adding embeddings later; it is not
implemented in Phase 1 because exact-term matching wins on a corpus whose useful
content is identifiers (``nvidia.com/gpu``, ``rook-ceph-mgr``, kubectl lines).
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from nrp_ops_agent.config import get_settings
from nrp_ops_agent.tools import ToolError, make_tool

Section = Literal["userdocs", "admindocs", "all"]

#: Terms are matched on word characters plus the identifier punctuation the
#: index tokenizer keeps together.
_TERM: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")

#: Body text returned per hit. Enough to judge relevance and quote from,
#: not enough to crowd out cluster evidence.
SNIPPET_CHARS: Final = 700


class Retriever(Protocol):
    """The seam for a future embedding-based retriever."""

    async def search(self, query: str, *, limit: int, section: Section) -> list[dict[str, Any]]: ...


def _fts_query(query: str) -> str:
    """Build a safe FTS5 MATCH expression from free text.

    Each term is double-quoted, which makes FTS5 treat it as a literal and
    neutralises ``NEAR``, ``*``, column filters and unbalanced quotes.
    """
    terms = _TERM.findall(query)[:12]
    if not terms:
        raise ToolError("empty_query", "The query contained no searchable terms.")
    return " ".join('"' + t.replace('"', "") + '"' for t in terms)


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise ToolError(
            "docs_index_missing",
            f"No docs index at {db_path}. The docs-sync CronJob has not run yet.",
        )
    # Read-only URI: this process must never be able to write the shared index.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _run_search(
    db_path: Path, query: str, limit: int, section: Section
) -> tuple[list[dict[str, Any]], bool]:
    """Return ``(rows, relaxed)``. AND first for precision, OR as a fallback."""
    match_all = _fts_query(query)
    conn = _connect(db_path)
    try:
        for relaxed, expression in ((False, match_all), (True, match_all.replace('" "', '" OR "'))):
            sql = (
                "SELECT c.url, c.section, c.page_title, c.heading_path, c.body, "
                "       bm25(chunks_fts, 4.0, 3.0, 1.0) AS score "
                "FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
                "WHERE chunks_fts MATCH ? "
            )
            params: list[Any] = [expression]
            if section != "all":
                sql += "AND c.section = ? "
                params.append(section)
            sql += "ORDER BY score LIMIT ?"
            params.append(limit)

            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as exc:
                raise ToolError("docs_index_error", f"Docs index query failed: {exc}") from exc
            if rows:
                return [_row_to_hit(r) for r in rows], relaxed
        return [], True
    finally:
        conn.close()


def _row_to_hit(row: sqlite3.Row) -> dict[str, Any]:
    body = row["body"]
    return {
        "url": row["url"],
        "section": row["section"],
        "page": row["page_title"],
        "heading": row["heading_path"] or row["page_title"],
        "score": round(float(row["score"]), 3),
        "excerpt": body[:SNIPPET_CHARS] + ("..." if len(body) > SNIPPET_CHARS else ""),
    }


class SearchDocsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=512)
    limit: int = Field(default=5, ge=1, le=20)
    section: Section = Field(
        default="all",
        description="'admindocs' holds the operator runbooks (upgrades, cluster, "
        "storage); 'userdocs' describes what users are told to do.",
    )


@make_tool(
    "search_docs",
    "Search the NRP documentation (nrp.ai) for runbooks, configuration and known "
    "procedures. Every hit returns its published URL -- cite them, so the "
    "operator can verify the answer. Prefer section='admindocs' for how to fix "
    "something and 'userdocs' for what users expect to work.",
    SearchDocsArgs,
)
async def search_docs(args: SearchDocsArgs) -> dict[str, Any]:
    db_path = get_settings().docs_db_path
    hits, relaxed = await asyncio.to_thread(
        _run_search, db_path, args.query, args.limit, args.section
    )
    return {
        "query": args.query,
        "section": args.section,
        "match_mode": "any-term" if relaxed else "all-terms",
        "hit_count": len(hits),
        "hits": hits,
    }


class Fts5Retriever:
    """Concrete :class:`Retriever` over the FTS5 index."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or get_settings().docs_db_path

    async def search(
        self, query: str, *, limit: int = 5, section: Section = "all"
    ) -> list[dict[str, Any]]:
        hits, _ = await asyncio.to_thread(_run_search, self._db_path, query, limit, section)
        return hits
