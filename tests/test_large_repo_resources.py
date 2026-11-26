"""Disk and memory bounds for large repositories, with unchanged output.

Each test runs the real code path on real SQLite + sqlite-vec with
realistic sizes, and samples real file sizes / allocations while the
operation runs. Every resource bound here is paired with an equivalence
check against the previous algorithm, because the fix must only change
resource use, never results.

Run: python -m pytest tests/test_large_repo_resources.py -v
"""

from __future__ import annotations

import gc
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugin_implementation.unified_db import UnifiedWikiDB, _serialize_float32_vec  # noqa: E402

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
DIM = 1536

# Real source text as node content (average ~1.4 KB, like the calibration repo).
_CORPUS = (PLUGIN_ROOT / "plugin_implementation" / "unified_db.py").read_text()


def _text(i: int, size: int = 1400) -> str:
    off = (i * 7919) % (len(_CORPUS) - size)
    return _CORPUS[off: off + size]


def _vec(text: str) -> List[float]:
    """Deterministic 1536-d vector derived from the text."""
    digest = hashlib.sha256(text.encode("utf-8", "replace")).digest()
    return [b / 255.0 for b in digest] * (DIM // len(digest))


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _vec_or_skip(db: UnifiedWikiDB) -> None:
    if not db.vec_available:
        pytest.skip("sqlite-vec extension unavailable")


# ═══════════════════════════════════════════════════════════════════════════
# A1 — populate_embeddings keeps the WAL bounded
# ═══════════════════════════════════════════════════════════════════════════

def _seed_nodes(db: UnifiedWikiDB, n: int) -> None:
    rows = []
    for i in range(n):
        if i % 97 == 0:
            text = ""            # skipped: empty
        elif i % 101 == 0:
            text = " \n\t\r "    # skipped: whitespace only
        else:
            text = _text(i)
        rows.append({"node_id": f"python::pkg{i % 40}/m{i}.py::f{i}", "source_text": text,
                     "rel_path": f"pkg{i % 40}/m{i}.py", "symbol_name": f"f{i}",
                     "symbol_type": "function"})
        if len(rows) == 5000:
            db.upsert_nodes_batch(rows)
            rows = []
    db.upsert_nodes_batch(rows)
    db.conn.commit()


def _reference_embedded(db: UnifiedWikiDB) -> Dict[str, List[float]]:
    """What the previous single-pass algorithm embedded: same filter, scan order."""
    ws = " ' ' || char(9) || char(10) || char(13) "
    rows = db.conn.execute(
        "SELECT node_id, source_text FROM repo_nodes "
        f"WHERE source_text IS NOT NULL AND length(trim(source_text, {ws})) > 0"
    ).fetchall()
    return {r["node_id"]: _vec(r["source_text"]) for r in rows}


def test_populate_embeddings_wal_bounded(tmp_path):
    db = UnifiedWikiDB(tmp_path / "emb.wiki.db", embedding_dim=DIM)
    _vec_or_skip(db)
    n = 20_000
    _seed_nodes(db, n)
    db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    wal = f"{db.db_path}-wal"
    samples: List[int] = []
    calls: List[List[str]] = []

    def embed(texts: List[str]) -> List[List[float]]:
        samples.append(_size(wal))
        calls.append(list(texts))
        return [_vec(t) for t in texts]

    count = db.populate_embeddings(embed, batch_size=64)
    samples.append(_size(wal))

    expected = _reference_embedded(db)
    assert count == len(expected)
    # The batches are the same as one pass over every row (same order, 64 per call).
    flat = [t for c in calls for t in c]
    assert len(flat) == len(expected) and all(len(c) == 64 for c in calls[:-1])
    stored = {
        r[0]: r[1]
        for r in db.conn.execute("SELECT node_id, embedding FROM repo_vec").fetchall()
    }
    assert set(stored) == set(expected)
    for nid, vec in expected.items():
        assert stored[nid] == _serialize_float32_vec(vec)

    # Previous code: one transaction, WAL ~6.4 KB/node -> ~128 MB here.
    assert max(samples) < 16 * 1024 * 1024, f"WAL peaked at {max(samples)/1e6:.1f} MB"
    db.close()


def test_populate_embeddings_failed_batch_skipped_and_budget_raises(tmp_path):
    """Per-batch failure: log and continue; budget errors still abort."""
    from plugin_implementation.budget_errors import BudgetExceededError

    db = UnifiedWikiDB(tmp_path / "fail.wiki.db", embedding_dim=DIM)
    _vec_or_skip(db)
    _seed_nodes(db, 3000)
    expected = _reference_embedded(db)
    order = list(expected)
    state = {"n": 0}

    def flaky(texts):
        state["n"] += 1
        if state["n"] == 20:  # a batch in the second page
            raise RuntimeError("rate limited")
        return [_vec(t) for t in texts]

    count = db.populate_embeddings(flaky, batch_size=64)
    lost = set(order[19 * 64: 20 * 64])
    stored = {r[0] for r in db.conn.execute("SELECT node_id FROM repo_vec")}
    assert count == len(expected) - 64
    assert stored == set(expected) - lost

    def budget(_texts):
        raise BudgetExceededError("project_budget_exceeded", "budget")

    with pytest.raises(BudgetExceededError):
        db.populate_embeddings(budget, batch_size=64)
    db.close()


# ═══════════════════════════════════════════════════════════════════════════
# A2 — full unified DB build stays within the disk budget
# ═══════════════════════════════════════════════════════════════════════════

class _Embeddings:
    def embed_documents(self, texts):
        return [_vec(t) for t in texts]

    def embed_query(self, text):
        return _vec(text)


def _synthetic_graph(n: int) -> nx.MultiDiGraph:
    import random

    G = nx.MultiDiGraph()
    for i in range(n):
        d = f"pkg{i % 50}/sub{i % 7}"
        G.add_node(
            f"python::{d}/m{i}.py::f{i}", symbol_name=f"f{i}", symbol_type="function",
            rel_path=f"{d}/m{i}.py", file_name=f"m{i}.py", language="python",
            source_text=_text(i), docstring="doc " * 10, start_line=1, end_line=40,
        )
    ids = list(G.nodes)
    rnd = random.Random(1)
    for i in range(1, n):
        if i % 200:  # leave 0.5% orphans so Phase 2 runs its passes
            G.add_edge(ids[rnd.randrange(0, i)], ids[i], relationship_type="calls")
    return G


def _indexer(tmp_path, graph):
    from plugin_implementation.filesystem_indexer import FilesystemRepositoryIndexer

    idx = FilesystemRepositoryIndexer(
        cache_dir=str(tmp_path / "cache"), model_cache_dir=str(tmp_path / "hf"),
        embeddings=_Embeddings(), max_workers=1,
    )
    idx.relationship_graph = graph
    idx._last_commit_hash = "cafebabedeadbeef"
    return idx


def test_unified_db_build_disk_budget(tmp_path):
    probe = UnifiedWikiDB(tmp_path / "probe.wiki.db", embedding_dim=DIM)
    vec_ok = probe.vec_available
    probe.close()
    if not vec_ok:
        pytest.skip("sqlite-vec extension unavailable")

    idx = _indexer(tmp_path, _synthetic_graph(25_000))
    key = idx.vectorstore_manager._generate_repo_hash("/repo", [], "cafebabedeadbeef")
    db_path = str(tmp_path / "cache" / f"{key}.wiki.db")
    peak = {"total": 0, "wal": 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            wal = _size(db_path + "-wal")
            total = _size(db_path) + wal + _size(db_path + "-shm")
            peak["total"] = max(peak["total"], total)
            peak["wal"] = max(peak["wal"], wal)
            time.sleep(0.002)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    try:
        assert idx._write_unified_db("/repo", "o/r:main:cafebabe") == key
    finally:
        stop.set()
        th.join()

    final = _size(db_path) + _size(db_path + "-wal") + _size(db_path + "-shm")
    assert final > 200e6  # the build really happened
    # Previous code: ~1.5x final (WAL kept the size of the embedding transaction).
    assert peak["total"] <= 1.1 * final + 64 * 1024 * 1024, (
        f"peak {peak['total']/1e6:.0f} MB vs final {final/1e6:.0f} MB "
        f"(WAL {peak['wal']/1e6:.0f} MB)"
    )
    assert peak["wal"] < 64 * 1024 * 1024


def test_readonly_reader_opens_built_db(tmp_path):
    path = tmp_path / "ro.wiki.db"
    db = UnifiedWikiDB(path, embedding_dim=DIM)
    _vec_or_skip(db)
    db.from_networkx(_synthetic_graph(2000))
    db.populate_embeddings(_Embeddings().embed_documents)
    db.close()
    assert not os.path.exists(f"{path}-wal") or _size(f"{path}-wal") == 0

    ro = UnifiedWikiDB(path, embedding_dim=DIM, readonly=True)
    assert ro.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert ro.node_count() == 2000
    assert ro.search_fts5("populate_embeddings", limit=5)
    hits = ro.search_vec(_vec(_text(7)), k=3)
    assert hits and hits[0]["node_id"].endswith("::f7")
    ro.close()


def test_fts_rebuild_contents_unchanged(tmp_path):
    """Chunked FTS rebuild == single INSERT…SELECT (rowids and contents)."""
    db = UnifiedWikiDB(tmp_path / "fts.wiki.db", embedding_dim=DIM)
    G = _synthetic_graph(12_000)
    db.from_networkx(G)
    # Re-import a smaller graph: the stale-node purge leaves rowid gaps.
    G.remove_nodes_from([n for k, n in enumerate(list(G.nodes)) if k % 13 == 0])
    db.from_networkx(G)
    assert db.conn.execute("SELECT max(rowid) - count(*) FROM repo_nodes").fetchone()[0] > 0
    db._populate_fts5()
    got = db.conn.execute("SELECT rowid, * FROM repo_fts ORDER BY rowid").fetchall()

    db.conn.execute("DELETE FROM repo_fts")
    db.conn.execute(
        "INSERT INTO repo_fts (node_id, symbol_name, signature, docstring, source_text) "
        "SELECT node_id, symbol_name, signature, docstring, source_text FROM repo_nodes"
    )
    db.conn.commit()
    want = db.conn.execute("SELECT rowid, * FROM repo_fts ORDER BY rowid").fetchall()
    assert [tuple(r) for r in got] == [tuple(r) for r in want]
    assert len(got) == db.node_count()
    db.close()


# ═══════════════════════════════════════════════════════════════════════════
# A4 — orphan resolution: same output, memory not linear in orphans
# ═══════════════════════════════════════════════════════════════════════════

def _orphan_db(tmp_path, name: str, n_orphans: int, n_anchors: int = 400):
    """DB + graph with doc-like orphans and anchors sharing names/dirs/vectors."""
    db = UnifiedWikiDB(tmp_path / f"{name}.wiki.db", embedding_dim=DIM)
    G = nx.MultiDiGraph()
    for i in range(n_anchors):
        d = f"svc{i % 20}"
        G.add_node(f"python::{d}/a{i}.py::Anchor{i}", symbol_name=f"Anchor{i}",
                   symbol_type="class", rel_path=f"{d}/a{i}.py", language="python",
                   source_text=f"class Anchor{i}:\n    '''Handles topic{i % 50}.'''\n" + _text(i, 400))
    anchors = list(G.nodes)
    for i in range(1, n_anchors):
        G.add_edge(anchors[i - 1], anchors[i], relationship_type="calls")
    for j in range(n_orphans):
        d = f"svc{j % 20}"
        G.add_node(f"markdown::{d}/doc{j}.md::section_topic{j % 50}",
                   symbol_name=f"topic{j % 50} Anchor{j % n_anchors}",
                   symbol_type="markdown_section", rel_path=f"{d}/doc{j}.md",
                   language="markdown",
                   source_text=f"See Anchor{j % n_anchors} for topic{j % 50}. " + _text(j, 300))
    db.from_networkx(G)
    rows = db.conn.execute("SELECT node_id, source_text FROM repo_nodes").fetchall()
    db.upsert_embeddings_batch(
        [(r["node_id"], _vec(r["source_text"][:60])) for r in rows]
    )
    db.conn.commit()
    return db, G


def _edge_set(G):
    return sorted(
        (u, v, d.get("relationship_type"), d.get("created_by"), repr(d.get("raw_similarity")))
        for u, v, d in G.edges(data=True)
    )


def _old_resolve_orphans(db, G):
    """Previous behaviour: pre-fetch every pending orphan vector into a dict."""
    from plugin_implementation import graph_orphan_cascade_v2 as cascade
    from plugin_implementation import graph_orphan_hybrid as hybrid
    from plugin_implementation import graph_topology as topo

    original = hybrid.resolve_orphans_hybrid
    cache: Dict[str, Optional[List[float]]] = {}

    def prefetched(db_, G_, nid, *, orphan_embeddings=None, **kw):
        if not cache:
            pending = [n for n in topo.find_orphans(G_)]
            cache.update(cascade.collect_orphan_embeddings(db_, pending))
        return original(db_, G_, nid, orphan_embeddings=cache, **kw)

    topo.resolve_orphans_hybrid = prefetched
    try:
        return topo.resolve_orphans(db, G)
    finally:
        topo.resolve_orphans_hybrid = original


def test_orphan_resolution_unchanged_and_bounded(tmp_path):
    from plugin_implementation import graph_topology as topo

    db_old, G_old = _orphan_db(tmp_path, "old", 1500)
    _vec_or_skip(db_old)
    db_new, G_new = _orphan_db(tmp_path, "new", 1500)
    stats_old = _old_resolve_orphans(db_old, G_old)
    stats_new = topo.resolve_orphans(db_new, G_new)
    assert stats_new == stats_old
    assert stats_new["hybrid"] > 0  # the vector pass actually ran
    assert _edge_set(G_new) == _edge_set(G_old)
    db_old.close()
    db_new.close()

    # Peak traced memory of resolve_orphans: small vs 4x orphans.
    peaks = {}
    for n in (1000, 4000):
        db, G = _orphan_db(tmp_path, f"mem{n}", n)
        gc.collect()
        tracemalloc.start()
        topo.resolve_orphans(db, G)
        peaks[n] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        db.close()
    # Previously ~62 KB per orphan (3000 more orphans -> +185 MB).
    growth = peaks[4000] - peaks[1000]
    assert growth < 3000 * 8 * 1024, f"peak grew {growth/1e6:.0f} MB for 3000 orphans"


# ═══════════════════════════════════════════════════════════════════════════
# A3 — git object store removed after repository info is read
# ═══════════════════════════════════════════════════════════════════════════

def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_dir_removed_after_info(tmp_path, monkeypatch):
    from plugin_implementation import filesystem_indexer as fi
    from plugin_implementation.repo_providers import GitCloneConfig

    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "README.md").write_text("# Demo\n\nA small demo repository.\n")
    (src / "pkg" / "core.py").write_text("class Core:\n    def run(self):\n        return 1\n")
    # Non-trivial pack so the size drop is measurable.
    (src / "pkg" / "blob.txt").write_bytes(os.urandom(256 * 1024).hex().encode())
    _git("init", "-q", "-b", "trunk", cwd=src)
    _git("add", ".", cwd=src)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init", cwd=src)
    head = _git("rev-parse", "HEAD", cwd=src)

    cfg = GitCloneConfig(clone_url=f"file://{src}", repo_identifier="demo/repo", branch="trunk")
    idx = fi.FilesystemRepositoryIndexer(
        cache_dir=str(tmp_path / "cache"), model_cache_dir=str(tmp_path / "hf"),
        clone_config=cfg, embeddings=_Embeddings(), max_workers=1,
    )
    seen = {}

    def fake_build(self, repo_identifier, repo_path, max_files, force_rebuild=False):
        seen["objects"] = sum(f.stat().st_size for f in (Path(repo_path) / ".git" / "objects").rglob("*") if f.is_file())
        seen["files"] = sorted(p.name for p in Path(repo_path).rglob("*") if p.is_file() and ".git" not in p.parts)
        return {"repository_identifier": repo_identifier}

    monkeypatch.setattr(fi.FilesystemRepositoryIndexer, "_build_filesystem_index", fake_build)
    result = idx.index_repository("demo/repo", branch="trunk", force_rebuild=True)
    repo_path = result["repository_path"]

    assert idx.get_commit_hash() == head
    assert result["actual_branch"] == "trunk"
    assert idx.current_repo_info["remote_url"] == f"file://{src}"
    assert seen["objects"] == 0                      # pack is gone before indexing
    assert seen["files"] == ["README.md", "blob.txt", "core.py"]  # checkout kept

    # Reusing the clone path (same commit) still reports the same info.
    again = idx.repo_manager.get_repository_info(repo_path)
    assert again["commit_hash"] == head and again["branch"] == "trunk"


# ═══════════════════════════════════════════════════════════════════════════
# A5 — repository-analysis inputs unchanged, retained documents bounded
# ═══════════════════════════════════════════════════════════════════════════

def _docs(n: int):
    from langchain_core.documents import Document

    out = []
    for i in range(n):
        kind = i % 10
        if i in (5, 6, 400):
            src, ct, st = "README.md", "text", "markdown_section"
        elif kind == 0:
            src, ct, st = f"conf/app{i}.yaml", "text", "yaml_document"
        elif kind == 1:
            src, ct, st = f"docs/guide{i}.md", "text", "markdown_section"
        else:
            src, ct, st = f"pkg{i % 30}/mod{i}.py", "symbol", "function"
        out.append(Document(
            page_content=_text(i, 1200),
            metadata={"source": src, "rel_path": src, "file_path": "/repo/" + src,
                      "chunk_type": ct, "symbol_type": st, "symbol_name": f"s{i}",
                      "language": "python" if ct == "symbol" else "markdown",
                      "section_id": i},
        ))
    return out


def _analysis_inputs(indexer, stats):
    from plugin_implementation.agents.wiki_graph_optimized import (
        OptimizedWikiGenerationAgent as Agent,
    )

    docs = indexer.get_all_documents()
    count_fn = getattr(indexer, "get_document_count", None)
    sources_fn = getattr(indexer, "get_document_sources", None)
    sources = sources_fn() if callable(sources_fn) else [
        d.metadata.get("source", "") for d in docs if d.metadata.get("source")
    ]
    norm = dict(stats)
    norm["code_analysis"] = dict(stats["code_analysis"], languages=sorted(stats["code_analysis"]["languages"]))
    norm["filter_stats"] = None
    summary = indexer.export_index_summary()
    summary["files"]["files"] = sorted(summary["files"]["files"])
    summary["languages"] = sorted(summary["languages"])
    return {
        "readme": Agent._extract_full_readme_content_from_docs(None, docs),
        "samples": Agent._extract_representative_code_samples(None, docs),
        "sources": set(sources),
        "count": count_fn() if callable(count_fn) else len(docs),
        "stats": norm,
        "summary": summary,
    }


def _run_materialize(tmp_path, n, keep_all: bool, monkeypatch):
    from plugin_implementation import filesystem_indexer as fi

    class Analysis:
        def __init__(self):
            self.documents = []
            self.documents_iter = iter(_docs(n))
            self.unified_graph = nx.MultiDiGraph()
            self.cross_language_relationships = []

    if keep_all:
        monkeypatch.setenv("DEEPWIKI_KEEP_DOC_SPLITS", "1")
    else:
        monkeypatch.delenv("DEEPWIKI_KEEP_DOC_SPLITS", raising=False)
    repo = tmp_path / ("full" if keep_all else "lean")
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    idx = fi.FilesystemRepositoryIndexer(
        cache_dir=str(tmp_path / "cache"), model_cache_dir=str(tmp_path / "hf"),
        embeddings=_Embeddings(), max_workers=1,
    )
    idx.current_repo_path = str(repo)
    idx.current_repo_info = {"commit_hash": "c0ffee", "branch": "main", "remote_url": "u"}
    monkeypatch.setattr(idx.graph_builder, "analyze_repository", lambda *a, **k: Analysis())
    gc.collect()
    tracemalloc.start()
    stats = idx._build_filesystem_index("o/r:main:c0ffee", str(repo), None)
    retained = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    return idx, stats, retained


def test_document_consumers_unchanged_and_retention_bounded(tmp_path, monkeypatch):
    n = 20_000
    full_idx, full_stats, full_mem = _run_materialize(tmp_path, n, True, monkeypatch)
    lean_idx, lean_stats, lean_mem = _run_materialize(tmp_path, n, False, monkeypatch)

    assert _analysis_inputs(lean_idx, lean_stats) == _analysis_inputs(full_idx, full_stats)
    assert lean_stats["documents"]["total"] == n
    # README chunks + 3 .py + 2 config + 2 .md; nothing else is read later.
    assert len(lean_idx.get_all_documents()) <= 10
    # Previously every Document stayed alive (~1.2 KB text + metadata each).
    assert lean_mem < full_mem / 5, f"retained {lean_mem/1e6:.1f} MB vs {full_mem/1e6:.1f} MB"


# ═══════════════════════════════════════════════════════════════════════════
# A6 — configured filters reach the graph builder
# ═══════════════════════════════════════════════════════════════════════════

def _filter_repo(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "generated").mkdir()
    (root / "vendor" / "lib").mkdir(parents=True)
    for i in range(14):
        (root / "src" / f"m{i:02d}.py").write_text(f"def f{i}():\n    return {i}\n")
    (root / "generated" / "big_gen.py").write_text("def g():\n    return 0\n")
    (root / "vendor" / "lib" / "v.py").write_text("def v():\n    return 0\n")
    (root / "src" / "huge.py").write_text("x = 1\n" * (400 * 1024))  # ~2.3 MB
    (root / "README.md").write_text("# Repo\n")


def _builder_files(idx, repo: Path, max_files=None) -> set:
    seen = set()
    builder = idx.graph_builder
    original = builder._discover_files_by_language

    def spy(*a, **k):
        res = original(*a, **k)
        for lang, paths in res.items():
            if lang != "unknown":
                seen.update(os.path.relpath(p, repo) for p in paths)
        raise RuntimeError("stop after discovery")

    builder._discover_files_by_language = spy
    with pytest.raises(RuntimeError):
        idx._build_filesystem_index("o/r:main:c0ffee", str(repo), max_files)
    return seen


def test_filter_manager_applied_to_graph_builder(tmp_path):
    from plugin_implementation.filesystem_indexer import FilesystemRepositoryIndexer

    repo = tmp_path / "repo"
    _filter_repo(repo)
    idx = FilesystemRepositoryIndexer(
        cache_dir=str(tmp_path / "cache"), model_cache_dir=str(tmp_path / "hf"),
        api_filters={"excluded_dirs": ["generated"], "max_file_size_mb": 1},
        embeddings=_Embeddings(), max_workers=1,
    )
    approved = idx._discover_files(str(repo))[:10]
    assert "src/huge.py" not in idx._discover_files(str(repo))
    assert not any(p.startswith("generated/") for p in approved)
    assert _builder_files(idx, repo, max_files=10) == set(approved)


def test_unfiltered_graph_builder_walk_unchanged(tmp_path):
    """No configured filters, no max_files: the builder still walks the repo itself."""
    from plugin_implementation.filesystem_indexer import FilesystemRepositoryIndexer

    repo = tmp_path / "repo"
    _filter_repo(repo)
    (repo / ".editorconfig").write_text("root = true\n")  # builder-only file today
    idx = FilesystemRepositoryIndexer(
        cache_dir=str(tmp_path / "cache"), model_cache_dir=str(tmp_path / "hf"),
        embeddings=_Embeddings(), max_workers=1,
    )
    own = idx.graph_builder._discover_files_by_language(str(repo))
    expected = {os.path.relpath(p, repo) for lang, ps in own.items() if lang != "unknown" for p in ps}
    assert ".editorconfig" in expected and "src/huge.py" in expected
    assert _builder_files(idx, repo) == expected
