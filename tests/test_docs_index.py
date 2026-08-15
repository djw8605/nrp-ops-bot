"""Tests for the MDX chunker, the FTS5 index and search_docs.

The corpus below imitates the real nrp-site layout closely enough to exercise
what actually matters: Starlight frontmatter, JSX components wrapped around
fenced code, deep heading nesting, and the identifiers operators search on
(``nvidia.com/gpu``, ``rook-ceph-mgr``, ``coder-tls``).
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pytest

from helpers import ns
from nrp_ops_agent.docs_index import sync as sync_module
from nrp_ops_agent.docs_index.sync import (
    CloneFailed,
    IndexDirectoryUnwritable,
    build_index,
    chunk_by_heading,
    clone_docs,
    iter_chunks,
    require_writable_index_dir,
    section_for,
    split_frontmatter,
    strip_mdx,
    sync,
    url_for,
)
from nrp_ops_agent.tools import dispatch
from nrp_ops_agent.tools import docs as docs_tool
from nrp_ops_agent.tools.docs import Fts5Retriever, SearchDocsArgs, _fts_query, search_docs

SITE = "https://nrp.ai"

CODER_ADMIN = """---
title: Upgrading Coder
sidebar:
  order: 3
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

The Coder deployment on Nautilus runs in the `coder` namespace.

## Rolling upgrade

<Aside type="caution">
Drain workspaces before upgrading.
</Aside>

Bump the image and watch the rollout:

```bash
kubectl -n coder set image deployment/coder coder=nrp/coder:v2.9.1
kubectl -n coder rollout status deployment/coder
```

### Rolling back

If `coder` pods enter CrashLoopBackOff, roll back:

```bash
kubectl -n coder rollout undo deployment/coder
```

## TLS

The ingress terminates TLS with the `coder-tls` secret.
"""

GPU_USER = """---
title: Using GPUs
---

Request a GPU with the `nvidia.com/gpu` resource limit.

## Requesting a GPU

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

If your pod stays Pending, the cluster is out of that GPU type.
"""

CEPH_ADMIN = """---
title: Ceph storage
---

## Health checks

Run `ceph status` from the toolbox pod. The `rook-ceph-mgr` deployment must be
Running for the dashboard to answer.

## PVC stuck in Pending

Check the `rook-ceph-operator` logs.
"""


@pytest.fixture
def docs_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "nrp-site" / "src" / "content" / "docs" / "Documentation"
    (root / "admindocs" / "upgrades").mkdir(parents=True)
    (root / "admindocs" / "storage").mkdir(parents=True)
    (root / "userdocs" / "gpu").mkdir(parents=True)
    (root / "admindocs" / "upgrades" / "coder.mdx").write_text(CODER_ADMIN, encoding="utf-8")
    (root / "admindocs" / "storage" / "ceph.md").write_text(CEPH_ADMIN, encoding="utf-8")
    (root / "userdocs" / "gpu" / "index.mdx").write_text(GPU_USER, encoding="utf-8")
    return tmp_path / "nrp-site"


@pytest.fixture
def docs_db(tmp_path: Path, docs_checkout: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "docs.sqlite3"
    count = sync(db_path=db, site_base_url=SITE, local_checkout=docs_checkout)
    assert count > 0
    monkeypatch.setattr(docs_tool, "get_settings", lambda: _FakeSettings(db))
    return db


class _FakeSettings:
    def __init__(self, db_path: Path) -> None:
        self.docs_db_path = db_path


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


class TestFrontmatter:
    def test_title_is_extracted_and_frontmatter_removed(self) -> None:
        title, body = split_frontmatter(CODER_ADMIN)
        assert title == "Upgrading Coder"
        assert "sidebar:" not in body
        assert body.lstrip().startswith("import {")

    def test_missing_frontmatter_is_tolerated(self) -> None:
        title, body = split_frontmatter("# Just a heading\n")
        assert title == ""
        assert body == "# Just a heading\n"


class TestStripMdx:
    def test_code_fences_survive_byte_identical(self) -> None:
        out = strip_mdx(CODER_ADMIN)
        assert "kubectl -n coder set image deployment/coder coder=nrp/coder:v2.9.1" in out
        assert "nvidia.com/gpu" in strip_mdx(GPU_USER)

    def test_jsx_components_and_imports_are_removed(self) -> None:
        out = strip_mdx(CODER_ADMIN)
        assert "import {" not in out
        assert "<Aside" not in out
        assert "</Aside>" not in out
        # The prose inside the component is kept -- it is often the warning.
        assert "Drain workspaces before upgrading." in out

    def test_markdown_links_keep_text_and_target(self) -> None:
        out = strip_mdx("See [the GPU guide](/documentation/userdocs/gpu) first.")
        assert "the GPU guide" in out
        assert "/documentation/userdocs/gpu" in out

    def test_jsx_lookalikes_inside_fences_are_preserved(self) -> None:
        text = "```html\n<Card>literal</Card>\n```\n<Card>stripped</Card>\n"
        out = strip_mdx(text)
        assert "<Card>literal</Card>" in out
        assert "<Card>stripped</Card>" not in out


class TestChunking:
    def test_heading_path_is_hierarchical(self) -> None:
        chunks = chunk_by_heading(strip_mdx(split_frontmatter(CODER_ADMIN)[1]))
        paths = [h for h, _ in chunks]
        assert "Rolling upgrade > Rolling back" in paths
        assert "TLS" in paths

    def test_comments_inside_fences_are_not_headings(self) -> None:
        text = (
            "## Real\n\n```bash\n# not a heading\necho hi\n```\n\n"
            "body text here that is long enough\n"
        )
        paths = [h for h, _ in chunk_by_heading(text)]
        assert paths == ["Real"]

    def test_body_lands_with_its_heading(self) -> None:
        chunks = dict(chunk_by_heading(strip_mdx(split_frontmatter(CODER_ADMIN)[1])))
        assert "rollout undo" in chunks["Rolling upgrade > Rolling back"]

    def test_empty_document_yields_nothing(self) -> None:
        assert chunk_by_heading("   \n\n  ") == []


class TestUrlMapping:
    @pytest.mark.parametrize(
        ("rel", "expected"),
        [
            (
                "Documentation/admindocs/upgrades/coder.mdx",
                f"{SITE}/documentation/admindocs/upgrades/coder",
            ),
            ("Documentation/userdocs/gpu/index.mdx", f"{SITE}/documentation/userdocs/gpu"),
            (
                "Documentation/userdocs/coder/deploy.md",
                f"{SITE}/documentation/userdocs/coder/deploy",
            ),
        ],
    )
    def test_source_path_maps_to_published_url(self, rel: str, expected: str) -> None:
        assert url_for(Path(rel), SITE) == expected

    @pytest.mark.parametrize(
        ("rel", "expected"),
        [
            ("Documentation/admindocs/storage/ceph.md", "admindocs"),
            ("Documentation/userdocs/gpu/index.mdx", "userdocs"),
            ("Documentation/other/thing.md", "other"),
        ],
    )
    def test_section_detection(self, rel: str, expected: str) -> None:
        assert section_for(Path(rel)) == expected

    def test_chunks_carry_an_anchor_to_their_heading(self, docs_checkout: Path) -> None:
        chunks = list(iter_chunks(docs_checkout / "src/content/docs", SITE))
        rollback = next(c for c in chunks if c.heading_path.endswith("Rolling back"))
        assert rollback.url.endswith("/documentation/admindocs/upgrades/coder#rolling-back")


# --------------------------------------------------------------------------- #
# Index build
# --------------------------------------------------------------------------- #


class TestIndexBuild:
    def test_index_is_populated(self, docs_db: Path) -> None:
        conn = sqlite3.connect(docs_db)
        try:
            (count,) = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            (version,) = conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
        finally:
            conn.close()
        assert count >= 6
        assert version == "1"

    def test_build_is_atomic(self, tmp_path: Path, docs_checkout: Path) -> None:
        """A rebuild replaces the file wholesale; no .building artefact is left."""
        db = tmp_path / "docs.sqlite3"
        sync(db_path=db, site_base_url=SITE, local_checkout=docs_checkout)
        sync(db_path=db, site_base_url=SITE, local_checkout=docs_checkout)
        assert db.exists()
        assert not db.with_suffix(db.suffix + ".building").exists()

    def test_empty_checkout_refuses_to_publish(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        (empty / "src" / "content" / "docs").mkdir(parents=True)
        with pytest.raises(RuntimeError, match="zero chunks"):
            sync(db_path=tmp_path / "x.sqlite3", site_base_url=SITE, local_checkout=empty)

    def test_missing_docs_dir_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            sync(db_path=tmp_path / "x.sqlite3", local_checkout=tmp_path / "nope")

    def test_build_index_accepts_an_empty_list(self, tmp_path: Path) -> None:
        assert build_index([], tmp_path / "empty.sqlite3") == 0


class TestCloneFailureIsLegible:
    """The CronJob crashlooped and left nothing to read: `check=True` raised
    CalledProcessError, whose message is only an exit status, and git's stderr
    went into the exception and never to the log."""

    def test_git_stderr_reaches_the_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(cmd: list[str], **kwargs: Any) -> Any:
            return ns(
                returncode=128,
                stdout="",
                stderr="fatal: unable to access 'https://gitlab...': Could not resolve host",
            )

        monkeypatch.setattr(sync_module.subprocess, "run", fake_run)
        with pytest.raises(CloneFailed, match="Could not resolve host"):
            clone_docs("https://gitlab.nrp-nautilus.io/prp/nrp-site.git", tmp_path / "co")

    def test_a_timeout_names_the_likely_cause(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(cmd: list[str], **kwargs: Any) -> Any:
            raise subprocess.TimeoutExpired(cmd, 600)

        monkeypatch.setattr(sync_module.subprocess, "run", fake_run)
        with pytest.raises(CloneFailed, match="egress"):
            clone_docs("https://gitlab.nrp-nautilus.io/prp/nrp-site.git", tmp_path / "co")

    def test_main_exits_nonzero_with_the_reason_on_the_last_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def boom(**kwargs: Any) -> int:
            raise CloneFailed("Could not resolve host gitlab.nrp-nautilus.io")

        monkeypatch.setattr(sync_module, "sync", boom)
        assert sync_module.main(["--db-path", str(tmp_path / "x.sqlite3")]) == 1
        assert "Could not resolve host" in capsys.readouterr().err.strip().splitlines()[-1]

    @pytest.mark.skipif(os.getuid() == 0, reason="root bypasses file permission checks")
    def test_an_unwritable_index_dir_names_both_sides_of_the_mismatch(self, tmp_path: Path) -> None:
        """sqlite3 reports this as "unable to open database file", which reads
        like a missing path. The error has to name the directory's owner *and*
        the uid trying to write, or there is nothing to act on."""
        locked = tmp_path / "pvc"
        locked.mkdir()
        locked.chmod(0o555)
        try:
            with pytest.raises(IndexDirectoryUnwritable) as exc:
                require_writable_index_dir(locked / "docs.sqlite3")
        finally:
            locked.chmod(0o755)
        message = str(exc.value)
        assert str(locked) in message
        assert "mode=0555" in message
        assert f"uid={os.getuid()}" in message
        assert "fsGroup" in message

    def test_an_uncreatable_index_dir_is_reported(self, tmp_path: Path) -> None:
        """Same failure, reached without depending on mode bits so it holds when
        the suite runs as root."""
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("x", encoding="utf-8")
        with pytest.raises(IndexDirectoryUnwritable, match="index directory"):
            require_writable_index_dir(blocker / "sub" / "docs.sqlite3")

    def test_the_check_runs_before_the_clone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A volume that cannot be written fails the run whatever the checkout
        holds, so spending a clone to discover it is wasted time and a more
        confusing log."""

        def fake_clone(*args: Any, **kwargs: Any) -> Path:
            raise AssertionError("clone should not have been attempted")

        monkeypatch.setattr(sync_module, "clone_docs", fake_clone)
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("x", encoding="utf-8")
        with pytest.raises(IndexDirectoryUnwritable):
            sync(db_path=blocker / "sub" / "docs.sqlite3", site_base_url=SITE)

    def test_a_writable_dir_passes(self, tmp_path: Path) -> None:
        require_writable_index_dir(tmp_path / "docs.sqlite3")
        # The probe must not survive the check.
        assert list(tmp_path.iterdir()) == []

    def test_a_missing_docs_dir_names_what_was_there_instead(self, tmp_path: Path) -> None:
        checkout = tmp_path / "co"
        (checkout / "website").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="website"):
            sync(db_path=tmp_path / "x.sqlite3", local_checkout=checkout)


# --------------------------------------------------------------------------- #
# Query construction
# --------------------------------------------------------------------------- #


class TestFtsQueryBuilding:
    @pytest.mark.parametrize(
        "hostile",
        [
            'coder" OR chunks_fts MATCH "',
            "coder NEAR/5 secret",
            "coder*",
            'body:"coder"',
            "coder AND (secret OR token)",
        ],
    )
    def test_fts_syntax_in_the_query_is_neutralised(self, hostile: str) -> None:
        """Every term comes back as its own double-quoted literal, so FTS5 sees
        no operators, no column filters and no unbalanced quotes."""
        built = _fts_query(hostile)
        assert re.fullmatch(r'"[^"]+"(?: "[^"]+")*', built), built
        assert ":" not in built  # no column filter survived
        assert "*" not in built  # no prefix wildcard survived

    @pytest.mark.parametrize(
        "hostile",
        [
            'coder" OR chunks_fts MATCH "',
            "coder NEAR/5 secret",
            "coder*",
            'body:"coder"',
            "coder AND (secret OR token)",
        ],
    )
    async def test_hostile_query_runs_without_error(self, docs_db: Path, hostile: str) -> None:
        """The real proof: SQLite parses it and returns rows rather than raising."""
        result = await dispatch("search_docs", {"query": hostile})
        assert result.ok is True

    def test_identifiers_stay_whole(self) -> None:
        assert _fts_query("nvidia.com/gpu pending") == '"nvidia.com/gpu" "pending"'

    def test_query_with_no_terms_is_refused(self) -> None:
        from nrp_ops_agent.tools import ToolError

        with pytest.raises(ToolError):
            _fts_query("!!! ???")

    def test_stopwords_are_dropped(self) -> None:
        """Operators type whole questions. The filler words in them are the
        reason the OR fallback used to rank a long FPGA FAQ above the GPU page:
        every chunk contains "how"/"my"/"in", so those terms carry the score.
        """
        assert _fts_query("how do I request a GPU in my pod") == '"request" "GPU" "pod"'

    def test_an_all_stopword_query_keeps_its_terms(self) -> None:
        """Stripping every term would turn a findable query into an error, so
        the filter only applies when something survives it."""
        assert _fts_query("how do I") == '"how" "do" "I"'


# --------------------------------------------------------------------------- #
# Retrieval quality
# --------------------------------------------------------------------------- #

#: The ten operator-style queries used to sanity-check retrieval, each with the
#: URL fragment the top hit must contain.
RETRIEVAL_CASES = [
    ("how do I upgrade coder", "admindocs/upgrades/coder"),
    ("coder rollout undo rollback", "admindocs/upgrades/coder"),
    ("coder-tls ingress certificate", "admindocs/upgrades/coder"),
    ("kubectl set image deployment coder", "admindocs/upgrades/coder"),
    ("nvidia.com/gpu limit", "userdocs/gpu"),
    ("pod pending no gpu available", "userdocs/gpu"),
    ("requesting a gpu", "userdocs/gpu"),
    ("rook-ceph-mgr not running", "admindocs/storage/ceph"),
    ("pvc stuck pending ceph", "admindocs/storage/ceph"),
    ("ceph status toolbox", "admindocs/storage/ceph"),
]


class TestRetrievalQuality:
    @pytest.mark.parametrize(("query", "expected_url_fragment"), RETRIEVAL_CASES)
    async def test_top_hit_is_the_right_page(
        self, docs_db: Path, query: str, expected_url_fragment: str
    ) -> None:
        out = await search_docs(SearchDocsArgs(query=query, limit=3))
        assert out["hit_count"] > 0, f"no hits for {query!r}"
        assert expected_url_fragment in out["hits"][0]["url"], (
            f"{query!r} -> {out['hits'][0]['url']}"
        )

    async def test_every_hit_carries_a_verifiable_url(self, docs_db: Path) -> None:
        out = await search_docs(SearchDocsArgs(query="coder"))
        assert out["hits"]
        for hit in out["hits"]:
            assert hit["url"].startswith(f"{SITE}/documentation/")
            assert hit["heading"]
            assert hit["excerpt"]

    async def test_section_filter_restricts_results(self, docs_db: Path) -> None:
        admin = await search_docs(SearchDocsArgs(query="gpu coder", section="admindocs"))
        user = await search_docs(SearchDocsArgs(query="gpu coder", section="userdocs"))
        assert all(h["section"] == "admindocs" for h in admin["hits"])
        assert all(h["section"] == "userdocs" for h in user["hits"])

    async def test_all_terms_preferred_then_any_term_fallback(self, docs_db: Path) -> None:
        precise = await search_docs(SearchDocsArgs(query="rolling back"))
        assert precise["match_mode"] == "all-terms"
        loose = await search_docs(SearchDocsArgs(query="coder quasar tachyon"))
        assert loose["match_mode"] == "any-term"
        assert loose["hit_count"] > 0

    async def test_no_match_returns_empty_not_an_error(self, docs_db: Path) -> None:
        out = await search_docs(SearchDocsArgs(query="zzzqqqxxx"))
        assert out["hit_count"] == 0
        assert out["hits"] == []

    async def test_retriever_protocol_implementation(self, docs_db: Path) -> None:
        hits = await Fts5Retriever(docs_db).search("upgrade coder", limit=2)
        assert hits and "coder" in hits[0]["url"]


class TestDocsToolPlumbing:
    async def test_missing_index_is_a_structured_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            docs_tool, "get_settings", lambda: _FakeSettings(tmp_path / "absent.db")
        )
        result = await dispatch("search_docs", {"query": "coder"})
        assert result.ok is False
        assert result.content["error"] == "docs_index_missing"

    async def test_index_is_opened_read_only(self, docs_db: Path) -> None:
        conn = sqlite3.connect(f"file:{docs_db}?mode=ro", uri=True)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("DELETE FROM chunks")
        finally:
            conn.close()

    async def test_limit_above_max_is_rejected(self, docs_db: Path) -> None:
        result = await dispatch("search_docs", {"query": "coder", "limit": 500})
        assert result.ok is False
        assert result.content["error"] == "invalid_arguments"

    async def test_search_via_dispatch(self, docs_db: Path) -> None:
        result: Any = await dispatch("search_docs", {"query": "upgrade coder", "limit": 2})
        assert result.ok is True
        assert result.content["hit_count"] >= 1
