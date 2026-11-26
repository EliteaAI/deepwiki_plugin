"""Scaling and correctness tests for orphan resolution, Phase 2, and Phase 3.

Covers:
1. FTS5 query sanitization (XML tags, punctuation, keywords, Unicode, multi-word).
2. Indexed O(k) embedding blob fetch (`get_embedding_blobs_by_ids`).
3. Scoped `search_vec` correctness when global top-k sits in other dirs/clusters.
4. `resolve_orphans_hybrid` short-circuit and `text_chunk` skipping.
5. End-to-end 15k+ orphan resolution speed, no duplicate Pass-4 links, bounded WAL.
6. `_extract_proximity_edges` exact equivalence + O(N) scaling.
7. `bridge_disconnected_components` exact equivalence + scaling.
8. Phase 3 `_consolidate_sections`, `_consolidate_pages`, `_cross_edges_count`,
   and isolated-file assignment correctness + scaling.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple
from unittest.mock import MagicMock

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugin_implementation.feature_flags import get_feature_flags  # noqa: E402
from plugin_implementation.graph_clustering import (  # noqa: E402
    _consolidate_pages,
    _consolidate_sections,
    _cross_edges_count,
    hierarchical_leiden_cluster,
)
from plugin_implementation.graph_orphan_hybrid import (  # noqa: E402
    _DirHybridContext,
    resolve_orphans_hybrid,
)
from plugin_implementation.graph_topology import (  # noqa: E402
    _build_path_index,
    _extract_proximity_edges,
    bridge_disconnected_components,
    resolve_orphans,
)
from plugin_implementation.unified_db import (  # noqa: E402
    UnifiedWikiDB,
    _sanitize_fts5_query,
)

DIM = 1536


def _det_vec(seed_text: str, dim: int = DIM) -> List[float]:
    digest = hashlib.sha256(seed_text.encode("utf-8", "replace")).digest()
    return [b / 255.0 for b in digest] * (dim // len(digest))


def _vec_or_skip(db: UnifiedWikiDB) -> None:
    if not db.vec_available:
        pytest.skip("sqlite-vec extension unavailable")


# ═══════════════════════════════════════════════════════════════════════════
# 1. FTS5 query sanitization
# ═══════════════════════════════════════════════════════════════════════════


class TestFTS5Sanitization:
    def test_sanitize_tokens_and_special_inputs(self):
        assert _sanitize_fts5_query('<property name="foo">') == '"property" "name" "foo"'
        assert _sanitize_fts5_query('<?xml version="1.0"?>') == '"xml" "version" "1" "0"'
        assert _sanitize_fts5_query("file-name.xml") == '"file" "name" "xml"'
        assert _sanitize_fts5_query("a:b") == '"a" "b"'
        assert _sanitize_fts5_query("AND OR NOT NEAR") == '"AND" "OR" "NOT" "NEAR"'
        assert _sanitize_fts5_query("topic0 Anchor0") == '"topic0" "Anchor0"'
        assert _sanitize_fts5_query("認証サービス ユーザー設定") == '"認証サービス" "ユーザー設定"'
        assert _sanitize_fts5_query("<<<>>> ---") == ""
        assert _sanitize_fts5_query("") == ""

    def test_search_fts5_handles_xml_colons_keywords_unicode(self, tmp_path):
        db = UnifiedWikiDB(tmp_path / "fts_san.wiki.db", embedding_dim=4)
        db.upsert_nodes_batch([
            {
                "node_id": "n_xml",
                "symbol_name": '<property name="timeout">',
                "symbol_type": "xml_document",
                "rel_path": "cfg/app.xml",
                "source_text": '<property name="timeout" value="30"/>',
            },
            {
                "node_id": "n_col",
                "symbol_name": "a:b",
                "symbol_type": "function",
                "rel_path": "src/a.py",
                "source_text": "def handle_and_or(): pass # AND OR NOT a:b",
            },
            {
                "node_id": "n_uni",
                "symbol_name": "認証サービス",
                "symbol_type": "class",
                "rel_path": "src/auth.py",
                "source_text": "class 認証サービス: pass",
            },
        ])
        db.conn.commit()
        db._populate_fts5()

        assert db.search_fts5('<property name="timeout">')[0]["node_id"] == "n_xml"
        assert db.search_fts5('<?xml version="1.0"?>') == []
        assert db.search_fts5("file-name.xml") == []
        assert db.search_fts5("a:b")[0]["node_id"] == "n_col"
        assert db.search_fts5("AND OR NOT")[0]["node_id"] == "n_col"
        assert db.search_fts5("認証サービス")[0]["node_id"] == "n_uni"
        assert db.search_fts5("<<<>>> ---") == []
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Indexed batch embedding blob fetch
# ═══════════════════════════════════════════════════════════════════════════


def test_get_embedding_blobs_by_ids_speed_and_correctness(tmp_path):
    db = UnifiedWikiDB(tmp_path / "blobs.wiki.db", embedding_dim=DIM)
    _vec_or_skip(db)
    n = 10_000
    nodes = [
        {"node_id": f"n_{i}", "rel_path": f"d{i % 20}/f{i}.py", "symbol_name": f"f{i}"}
        for i in range(n)
    ]
    for s in range(0, n, 2500):
        db.upsert_nodes_batch(nodes[s : s + 2500])
    db.conn.commit()

    base_vec = _det_vec("base")
    for s in range(0, n, 1000):
        db.upsert_embeddings_batch([(f"n_{i}", base_vec) for i in range(s, s + 1000)])
    db.conn.commit()

    query_ids = [f"n_{i}" for i in range(100, 300)] + ["missing_id", "n_100"]
    t0 = time.perf_counter()
    blobs = db.get_embedding_blobs_by_ids(query_ids)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, f"get_embedding_blobs_by_ids took {elapsed:.2f}s"
    assert len(blobs) == 200
    assert "missing_id" not in blobs
    assert len(blobs["n_150"]) == DIM * 4
    db.close()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Scoped search_vec avoids global top-k post-filter loss
# ═══════════════════════════════════════════════════════════════════════════


def test_scoped_search_vec_finds_local_hits_behind_global_top_k(tmp_path):
    db = UnifiedWikiDB(tmp_path / "scoped_vec.wiki.db", embedding_dim=4)
    _vec_or_skip(db)

    nodes = []
    embs = []
    # 25 global nodes in "other/" with distance 0.0 to query [1, 0, 0, 0]
    for i in range(25):
        nid = f"other_{i}"
        nodes.append({
            "node_id": nid,
            "rel_path": f"other/m{i}.py",
            "symbol_name": nid,
            "macro_cluster": 9,
        })
        embs.append((nid, [1.0, 0.0, 0.0, 0.0]))

    # Local node in "target/" with non-zero distance — previously dropped
    # because global top-k=5 was 100% saturated by "other/" nodes.
    nodes.append({
        "node_id": "local_hit",
        "rel_path": "target/sub/service.py",
        "symbol_name": "LocalService",
        "macro_cluster": 3,
    })
    embs.append(("local_hit", [0.8, 0.2, 0.0, 0.0]))

    db.upsert_nodes_batch(nodes)
    db.upsert_embeddings_batch(embs)
    db.conn.commit()

    hits_path = db.search_vec([1.0, 0.0, 0.0, 0.0], k=5, path_prefix="target")
    assert [h["node_id"] for h in hits_path] == ["local_hit"]
    assert hits_path[0]["vec_distance"] > 0.0

    hits_cluster = db.search_vec([1.0, 0.0, 0.0, 0.0], k=5, cluster_id=3)
    assert [h["node_id"] for h in hits_cluster] == ["local_hit"]
    db.close()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Hybrid short-circuit, text_chunk skip, and RRF parity
# ═══════════════════════════════════════════════════════════════════════════


class TestHybridShortCircuit:
    def test_short_circuits_without_vec_when_fts_empty_and_unreachable(self):
        db = MagicMock()
        db.vec_available = True
        db.get_node.return_value = {
            "node_id": "o1",
            "symbol_name": "LonelyClass",
            "symbol_type": "class",
            "rel_path": "pkg/a.py",
        }
        db.search_fts_with_path.return_value = []

        G = nx.MultiDiGraph()
        G.add_node("o1", rel_path="pkg/a.py", symbol_name="LonelyClass", symbol_type="class")

        hits = resolve_orphans_hybrid(db, G, "o1", k=60, threshold=0.02)
        assert hits == []
        db.get_embedding_by_id.assert_not_called()
        db.search_vec.assert_not_called()

    def test_skips_text_chunk_immediately(self):
        db = MagicMock()
        db.get_node.return_value = {
            "node_id": "o_chunk",
            "symbol_name": "<tag>foo</tag>",
            "symbol_type": "text_chunk",
            "rel_path": "data/big.xml",
        }
        G = nx.MultiDiGraph()
        G.add_node("o_chunk", symbol_type="text_chunk", rel_path="data/big.xml")

        hits = resolve_orphans_hybrid(db, G, "o_chunk")
        assert hits == []
        db.search_fts_with_path.assert_not_called()
        db.search_vec.assert_not_called()

    def test_resolves_with_and_without_dir_context(self, tmp_path):
        db = UnifiedWikiDB(tmp_path / "hyb_ctx.wiki.db", embedding_dim=4)
        _vec_or_skip(db)
        db.upsert_nodes_batch([
            {
                "node_id": "anchor",
                "symbol_name": "OrderProcessor",
                "symbol_type": "class",
                "rel_path": "orders/proc.py",
                "source_text": "class OrderProcessor: process orders",
            },
            {
                "node_id": "orphan",
                "symbol_name": "OrderProcessor",
                "symbol_type": "markdown_section",
                "rel_path": "orders/README.md",
                "source_text": "Docs for OrderProcessor",
            },
        ])
        db.upsert_embeddings_batch([
            ("anchor", [1.0, 0.0, 0.0, 0.0]),
            ("orphan", [0.99, 0.01, 0.0, 0.0]),
        ])
        db.conn.commit()
        db._populate_fts5()

        G = nx.MultiDiGraph()
        G.add_node("anchor", rel_path="orders/proc.py", symbol_name="OrderProcessor", symbol_type="class")
        G.add_node("orphan", rel_path="orders/README.md", symbol_name="OrderProcessor", symbol_type="markdown_section")

        flags = get_feature_flags()
        hits_direct = resolve_orphans_hybrid(db, G, "orphan", flags=flags)
        ctx = _DirHybridContext(db, G, "orders", total_nodes=2)
        try:
            hits_ctx = resolve_orphans_hybrid(db, G, "orphan", flags=flags, _dir_ctx=ctx)
        finally:
            ctx.close()

        assert hits_direct and hits_direct[0]["node_id"] == "anchor"
        assert hits_ctx and hits_ctx[0]["node_id"] == "anchor"
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# 5. End-to-end 15k+ orphan resolution scaling & invariants
# ═══════════════════════════════════════════════════════════════════════════


def test_resolve_orphans_15k_mixed_xml_giant_dir_and_root_fast(tmp_path, caplog):
    db = UnifiedWikiDB(tmp_path / "scale15k.wiki.db", embedding_dim=DIM)
    _vec_or_skip(db)

    G = nx.MultiDiGraph()
    nodes = []

    # Connected code anchors across 25 normal dirs + 1 giant flat dir + root
    for d_idx in range(25):
        d = f"svc_{d_idx}"
        a1 = f"python::{d}/a.py::Service{d_idx}"
        a2 = f"python::{d}/b.py::Helper{d_idx}"
        for nid, name, rp in (
            (a1, f"Service{d_idx}", f"{d}/a.py"),
            (a2, f"Helper{d_idx}", f"{d}/b.py"),
        ):
            attrs = {
                "node_id": nid,
                "symbol_name": name,
                "symbol_type": "class",
                "rel_path": rp,
                "language": "python",
                "source_text": f"class {name}: handles domain {d_idx}",
            }
            nodes.append(attrs)
            G.add_node(nid, **attrs)
        G.add_edge(a1, a2, relationship_type="calls")

    # Giant flat directory (> 2000 nodes) with 1 connected anchor + 2600 orphans
    giant_anchor = "python::giant_dir/index.py::GiantAnchor"
    g_attrs = {
        "node_id": giant_anchor,
        "symbol_name": "GiantAnchor",
        "symbol_type": "class",
        "rel_path": "giant_dir/index.py",
        "language": "python",
        "source_text": "class GiantAnchor: root hub in giant directory",
    }
    nodes.append(g_attrs)
    G.add_node(giant_anchor, **g_attrs)
    G.add_edge(giant_anchor, "python::svc_0/a.py::Service0", relationship_type="calls")

    for i in range(2600):
        nid = f"xml::giant_dir/item_{i}.xml::chunk"
        attrs = {
            "node_id": nid,
            "symbol_name": f'<record id="{i}" name="GiantAnchor">' if i < 20 else f'<tag_{i}>',
            "symbol_type": "xml_document" if i < 20 else "text_chunk",
            "rel_path": f"giant_dir/item_{i}.xml",
            "language": "xml",
            "source_text": f'<record id="{i}">GiantAnchor config</record>',
        }
        nodes.append(attrs)
        G.add_node(nid, **attrs)

    # 12,400 orphans across normal dirs (mix of text_chunk XMLs, xml_document, markdown, code)
    for i in range(12_400):
        d_idx = i % 25
        d = f"svc_{d_idx}"
        if i % 10 == 0:
            nid = f"markdown::{d}/doc_{i}.md::sec"
            stype = "markdown_section"
            sname = f"Service{d_idx}"
            rp = f"{d}/doc_{i}.md"
            text = f"Documentation for Service{d_idx} endpoint {i}"
        elif i % 10 == 1:
            nid = f"python::{d}/mod_{i}.py::fn_{i}"
            stype = "function"
            sname = f"Helper{d_idx}" if i % 20 == 1 else f"unique_fn_{i}"
            rp = f"{d}/mod_{i}.py"
            text = f"def {sname}(): calls Helper{d_idx}"
        else:
            nid = f"xml::{d}/data_{i}.xml::c{i}"
            stype = "text_chunk"
            sname = f'<entry key="k{i}">'
            rp = f"{d}/data_{i}.xml"
            text = f'<entry key="k{i}">payload {i}</entry>'
        attrs = {
            "node_id": nid,
            "symbol_name": sname,
            "symbol_type": stype,
            "rel_path": rp,
            "language": "xml" if stype == "text_chunk" else "python",
            "source_text": text,
        }
        nodes.append(attrs)
        G.add_node(nid, **attrs)

    # Root-level orphans
    for i in range(30):
        nid = f"markdown::README_{i}.md::top"
        attrs = {
            "node_id": nid,
            "symbol_name": "Service0" if i < 5 else f"RootNote{i}",
            "symbol_type": "markdown_section",
            "rel_path": f"README_{i}.md",
            "language": "markdown",
            "source_text": f"Root readme {i} mentioning Service0",
        }
        nodes.append(attrs)
        G.add_node(nid, **attrs)

    for s in range(0, len(nodes), 5000):
        db.upsert_nodes_batch(nodes[s : s + 5000])
    db.conn.commit()
    db._populate_fts5()

    # Persist embeddings (same vector for Service{d_idx} pairs so hybrid RRF matches)
    for s in range(0, len(nodes), 2000):
        chunk = nodes[s : s + 2000]
        pairs = []
        for row in chunk:
            key = row["symbol_name"] if row["symbol_name"].startswith("Service") else row["node_id"]
            pairs.append((row["node_id"], _det_vec(key)))
        db.upsert_embeddings_batch(pairs)
    db.conn.commit()
    db.checkpoint_wal()

    caplog.set_level(logging.INFO)
    t0 = time.perf_counter()
    stats = resolve_orphans(db, G)
    elapsed = time.perf_counter() - t0

    assert elapsed < 15.0, f"15k orphan resolution took {elapsed:.2f}s"
    assert stats["orphan_count"] > 15_000
    assert stats["hybrid"] > 0
    assert stats["directory"] > 0
    assert stats["orphans_remaining"] == 0

    # No orphan resolved via hybrid/lexical should also receive a duplicate directory_link
    for u in G.nodes():
        edge_classes = {d.get("edge_class") for _, _, d in G.out_edges(u, data=True)}
        if "semantic" in edge_classes or "lexical" in edge_classes:
            assert "directory" not in edge_classes, f"{u} got duplicate directory_link"

    # WAL remained bounded
    wal_path = f"{db.db_path}-wal"
    wal_size = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
    assert wal_size < 64 * 1024 * 1024
    assert any("[ORPHAN] Pass 2" in r.message for r in caplog.records)
    db.close()


# ═══════════════════════════════════════════════════════════════════════════
# 6. _extract_proximity_edges equivalence & scaling
# ═══════════════════════════════════════════════════════════════════════════


def _naive_proximity_edges(G: nx.MultiDiGraph) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    _DOC_PREFIXES = ("docs/", "doc/", "documentation/")
    dir_to_code: Dict[str, List[str]] = {}
    for nid, data in G.nodes(data=True):
        if data.get("is_doc") == 1 or data.get("symbol_type") in {"file_doc", "markdown_section", "text_chunk"}:
            continue
        rp = data.get("rel_path", "")
        if rp:
            d = os.path.dirname(rp).replace("\\", "/")
            if d:
                dir_to_code.setdefault(d, []).append(nid)

    for nid, data in G.nodes(data=True):
        if not (data.get("is_doc") == 1 or data.get("symbol_type") in {"file_doc", "markdown_section", "text_chunk"}):
            continue
        rp = data.get("rel_path", "")
        if not rp:
            continue
        doc_dir = os.path.dirname(rp).replace("\\", "/")
        if not doc_dir:
            seen_top: Set[str] = set()
            doc_hash = int.from_bytes(hashlib.md5(nid.encode("utf-8")).digest()[:8], "big")
            for code_dir, code_nodes in dir_to_code.items():
                top = code_dir.split("/", 1)[0]
                if not top or top in seen_top or not code_nodes:
                    continue
                cnid = code_nodes[doc_hash % len(code_nodes)]
                if cnid != nid:
                    edges.append((nid, cnid))
                seen_top.add(top)
                if len(seen_top) >= 20:
                    break
            continue
        topic_dir = doc_dir
        for prefix in _DOC_PREFIXES:
            if topic_dir.startswith(prefix):
                topic_dir = topic_dir[len(prefix):]
                break
        if not topic_dir:
            continue
        for code_dir, code_nodes in dir_to_code.items():
            if code_dir == topic_dir or code_dir.endswith("/" + topic_dir):
                for cnid in code_nodes[:5]:
                    if cnid != nid:
                        edges.append((nid, cnid))
    return edges


def test_extract_proximity_edges_equivalence_and_scaling():
    # Equivalence on mixed root + docs/ + nested dirs
    G_eq = nx.MultiDiGraph()
    for i in range(25):
        G_eq.add_node(f"code_{i}", rel_path=f"top{i}/sub/api/m{i}.py", symbol_type="function")
    G_eq.add_node("code_flat", rel_path="lib/api/util.py", symbol_type="function")
    G_eq.add_node("doc_root_1", rel_path="README.md", is_doc=1, symbol_type="file_doc")
    G_eq.add_node("doc_root_2", rel_path="LICENSE.txt", is_doc=1, symbol_type="file_doc")
    G_eq.add_node("doc_api", rel_path="docs/api/overview.md", is_doc=1, symbol_type="markdown_section")
    G_eq.add_node("doc_sub", rel_path="documentation/sub/api/ref.md", is_doc=1, symbol_type="markdown_section")

    assert _extract_proximity_edges(G_eq, _build_path_index(G_eq)) == _naive_proximity_edges(G_eq)

    # Scaling: 10,000 doc nodes x 2,000 code dirs
    G_big = nx.MultiDiGraph()
    for i in range(2_000):
        G_big.add_node(f"c_{i}", rel_path=f"pkg_{i // 50}/mod_{i}/f.py", symbol_type="function")
    for i in range(10_000):
        rp = f"README_{i}.md" if i < 200 else f"docs/mod_{i % 2000}/d_{i}.xml"
        G_big.add_node(f"d_{i}", rel_path=rp, is_doc=1, symbol_type="text_chunk")

    path_idx = _build_path_index(G_big)
    t0 = time.perf_counter()
    edges = _extract_proximity_edges(G_big, path_idx)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"_extract_proximity_edges took {elapsed:.2f}s"
    assert len(edges) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. bridge_disconnected_components equivalence & scaling
# ═══════════════════════════════════════════════════════════════════════════


def test_bridge_disconnected_components_equivalence_and_scaling():
    # Equivalence against O(C^2) reference on single-dir and multi-dir components
    G1 = nx.MultiDiGraph()
    G2 = nx.MultiDiGraph()
    comps = [
        [("a1", "src/core/a.py"), ("a2", "src/core/b.py"), ("a3", "lib/x.py")],
        [("b1", "src/core/c.py")],  # single-dir matching comp 0
        [("c1", "lib/y.py"), ("c2", "lib/z.py")],  # multi-file single-dir
        [("d1", "src/core/d.py"), ("d2", "lib/w.py")],  # multi-dir
        [("e1", "other/q.py")],
    ]
    for comp in comps:
        for nid, rp in comp:
            G1.add_node(nid, rel_path=rp)
            G2.add_node(nid, rel_path=rp)
        for i in range(len(comp) - 1):
            G1.add_edge(comp[i][0], comp[i + 1][0], edge_class="structural")
            G2.add_edge(comp[i][0], comp[i + 1][0], edge_class="structural")

    # Naive reference on G2
    raw_comps = sorted(list(nx.weakly_connected_components(G2)), key=len, reverse=True)
    def _hist(comp):
        c: Counter = Counter()
        for nid in comp:
            rp = G2.nodes[nid]["rel_path"]
            c[rp.rsplit("/", 1)[0] if "/" in rp else "<root>"] += 1
        return c
    chists = [_hist(c) for c in raw_comps]
    creps = [max(c, key=lambda n: G2.in_degree(n) + G2.out_degree(n)) for c in raw_comps]
    naive_pairs = set()
    for i in range(1, len(raw_comps)):
        bt = 0
        bs = sum(min(chists[i][d], chists[0][d]) for d in chists[i] if d in chists[0])
        for j in range(1, i):
            sim = sum(min(chists[i][d], chists[j][d]) for d in chists[i] if d in chists[j])
            if sim > bs:
                bs = sim
                bt = j
        naive_pairs.add((creps[i], creps[bt]))

    db = MagicMock()
    stats = bridge_disconnected_components(db, G1, skip_db=True)
    actual_pairs = {
        (u, v) for u, v, d in G1.edges(data=True)
        if d.get("edge_class") == "bridge"
    }
    expected_bidir = naive_pairs | {(v, u) for u, v in naive_pairs}
    assert actual_pairs == expected_bidir
    assert stats["components_after"] == 1

    # Scaling: 10,000 components (5,000 in same dir + 5,000 across dirs)
    G_big = nx.MultiDiGraph()
    for i in range(10_000):
        d = "same/dir" if i < 5_000 else f"dir_{i % 250}"
        G_big.add_node(f"n_{i}", rel_path=f"{d}/f_{i}.py")

    t0 = time.perf_counter()
    stats_big = bridge_disconnected_components(MagicMock(), G_big, skip_db=True)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"bridge_disconnected_components took {elapsed:.2f}s"
    assert stats_big["components_after"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 8. Phase 3 consolidation & _cross_edges_count scaling
# ═══════════════════════════════════════════════════════════════════════════


def test_phase3_consolidation_and_cross_edges_scaling():
    # _cross_edges_count exactness with asymmetric set sizes and parallel edges
    G = nx.MultiDiGraph()
    for i in range(100):
        G.add_node(f"big_{i}", rel_path="big/a.py")
    G.add_node("small_0", rel_path="small/a.py")
    G.add_edge("small_0", "big_5", relationship_type="calls")
    G.add_edge("small_0", "big_5", relationship_type="imports")
    G.add_edge("big_10", "small_0", relationship_type="calls")
    assert _cross_edges_count(G, {"small_0"}, {f"big_{i}" for i in range(100)}) == 3
    assert _cross_edges_count(G, {f"big_{i}" for i in range(100)}, {"small_0"}) == 3

    # _consolidate_sections scaling: 3,000 sections
    G_sec = nx.MultiDiGraph()
    sections = {}
    macro_assign = {}
    micro_assign = {}
    for sid in range(3_000):
        nid = f"s_{sid}"
        G_sec.add_node(nid, rel_path=f"pkg_{sid % 100}/f_{sid}.py")
        if sid > 0 and sid % 3 == 0:
            G_sec.add_edge(f"s_{sid}", f"s_{sid - 1}", relationship_type="calls")
        sections[sid] = {"pages": {0: [nid]}}
        macro_assign[nid] = sid
        micro_assign[sid] = {nid: 0}

    leiden_sec = {
        "sections": sections,
        "macro_assignments": macro_assign,
        "micro_assignments": micro_assign,
    }
    t0 = time.perf_counter()
    _consolidate_sections(leiden_sec, G_sec, n_files=3_000)
    sec_elapsed = time.perf_counter() - t0
    assert sec_elapsed < 3.0, f"_consolidate_sections took {sec_elapsed:.2f}s"
    assert len(leiden_sec["macro_assignments"]) == 3_000
    for nid, sid in leiden_sec["macro_assignments"].items():
        assert sid in leiden_sec["sections"]

    # _consolidate_pages scaling: 5,000 pages across 10 sections
    G_pg = nx.MultiDiGraph()
    pg_sections = {}
    pg_macro = {}
    pg_micro = {}
    for i in range(5_000):
        sid = i % 10
        pid = i // 10
        nid = f"p_{i}"
        G_pg.add_node(nid, rel_path=f"sec_{sid}/d_{pid % 30}/f_{i}.py")
        if i >= 10:
            G_pg.add_edge(nid, f"p_{i - 10}", relationship_type="calls")
        pg_sections.setdefault(sid, {"pages": {}})["pages"][pid] = [nid]
        pg_macro[nid] = sid
        pg_micro.setdefault(sid, {})[nid] = pid

    leiden_pg = {
        "sections": pg_sections,
        "macro_assignments": pg_macro,
        "micro_assignments": pg_micro,
    }
    t1 = time.perf_counter()
    _consolidate_pages(leiden_pg, G_pg)
    pg_elapsed = time.perf_counter() - t1
    assert pg_elapsed < 3.0, f"_consolidate_pages took {pg_elapsed:.2f}s"
    all_assigned = set()
    for sid, sec in leiden_pg["sections"].items():
        for pid, nids in sec["pages"].items():
            for nid in nids:
                assert leiden_pg["micro_assignments"][sid][nid] == pid
                all_assigned.add(nid)
    assert len(all_assigned) == 5_000


def test_hierarchical_leiden_isolated_files_scaling():
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")

    G = nx.MultiDiGraph()
    # 100 connected file pairs in distinct directories -> multiple sections
    for d in range(100):
        u = f"c_{d}_0"
        v = f"c_{d}_1"
        G.add_node(u, rel_path=f"dir_{d}/a.py", symbol_type="function")
        G.add_node(v, rel_path=f"dir_{d}/b.py", symbol_type="function")
        G.add_edge(u, v, relationship_type="calls", weight=1.0)
    # 15,000 isolated files distributed across those directories
    for i in range(15_000):
        G.add_node(f"iso_{i}", rel_path=f"dir_{i % 100}/iso_{i}.xml", symbol_type="xml_document")

    t0 = time.perf_counter()
    res = hierarchical_leiden_cluster(G, hubs=set())
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"hierarchical_leiden_cluster took {elapsed:.2f}s"
    assert res["algorithm_metadata"]["isolated_files"] == 15_000
    assert len(res["macro_assignments"]) == 15_200
