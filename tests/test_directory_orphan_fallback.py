"""
Tests for directory-proximity orphan resolution fallback (Pass 3) and
synthetic edge weight handling.

Validates that _resolve_orphans_by_directory() connects isolated nodes
to the highest-degree anchor in the same or parent directory when both
FTS5 lexical and vector semantic passes are unavailable.

Also validates that apply_edge_weights() uses structural-only in-degree
and applies SYNTHETIC_WEIGHT_FLOOR to synthetic edge classes.
"""

import math
import os
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from plugin_implementation.graph_topology import (
    SYNTHETIC_WEIGHT_FLOOR,
    _resolve_orphans_by_directory,
    apply_edge_weights,
    find_orphans,
    resolve_orphans,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_graph_with_edges(nodes_with_paths, edges=None):
    """Build a MultiDiGraph with rel_path attributes and optional edges.

    nodes_with_paths: [(node_id, rel_path)]
    edges: [(src, tgt)] — structural edges
    """
    G = nx.MultiDiGraph()
    for nid, path in nodes_with_paths:
        G.add_node(nid, rel_path=path, symbol_name=nid,
                   symbol_type="function", file_name=path)
    for src, tgt in (edges or []):
        G.add_edge(src, tgt, relationship_type="calls")
    return G


def _make_db_mock(nodes_by_id):
    """Create a minimal DB mock with get_node, search_fts5, vec_available."""
    db = MagicMock()
    db.vec_available = False

    def get_node(nid):
        return nodes_by_id.get(nid)

    def search_fts5(query, limit=3):
        return []  # No FTS5 matches — forces fallback to directory pass

    db.get_node = get_node
    db.search_fts5 = search_fts5
    return db


# ═══════════════════════════════════════════════════════════════════════════
# find_orphans
# ═══════════════════════════════════════════════════════════════════════════

class TestFindOrphans:

    def test_isolated_nodes_are_orphans(self):
        G = _make_graph_with_edges([
            ("a", "src/a.cpp"),
            ("b", "src/b.cpp"),
            ("c", "src/c.cpp"),
        ], edges=[("a", "b")])
        orphans = find_orphans(G)
        assert orphans == ["c"]

    def test_no_orphans(self):
        G = _make_graph_with_edges([
            ("a", "src/a.cpp"),
            ("b", "src/b.cpp"),
        ], edges=[("a", "b")])
        orphans = find_orphans(G)
        assert orphans == []

    def test_all_orphans(self):
        G = _make_graph_with_edges([
            ("a", "src/a.cpp"),
            ("b", "lib/b.cpp"),
        ])
        orphans = find_orphans(G)
        assert set(orphans) == {"a", "b"}


# ═══════════════════════════════════════════════════════════════════════════
# _resolve_orphans_by_directory
# ═══════════════════════════════════════════════════════════════════════════

class TestDirectoryOrphanResolution:

    def test_same_directory_connection(self):
        """Orphan connects to highest-degree node in same directory."""
        G = _make_graph_with_edges([
            ("anchor1", "src/core/anchor1.cpp"),
            ("anchor2", "src/core/anchor2.cpp"),
            ("orphan", "src/core/orphan.cpp"),
        ], edges=[
            ("anchor1", "anchor2"),
            ("anchor2", "anchor1"),  # anchor2 has higher total degree
        ])
        nodes_db = {
            "orphan": {"rel_path": "src/core/orphan.cpp", "symbol_name": "orphan"},
        }
        db = _make_db_mock(nodes_db)

        edges_added = _resolve_orphans_by_directory(db, G, ["orphan"])

        assert edges_added == 1
        assert G.has_edge("orphan", "anchor1") or G.has_edge("orphan", "anchor2")
        assert find_orphans(G) == []  # no longer orphaned

    def test_parent_directory_fallback(self):
        """Orphan climbs to parent directory if no anchor in same dir."""
        G = _make_graph_with_edges([
            ("anchor", "src/core/utils/anchor.cpp"),
            ("orphan", "src/core/detail/orphan.cpp"),
        ], edges=[
            ("anchor", "anchor"),  # self-loop to give anchor a degree
        ])
        nodes_db = {
            "orphan": {"rel_path": "src/core/detail/orphan.cpp",
                       "symbol_name": "orphan"},
        }
        db = _make_db_mock(nodes_db)

        edges_added = _resolve_orphans_by_directory(db, G, ["orphan"])

        # No anchor in src/core/detail/ → climbs to src/core/ → still none
        # → climbs to src/ → still none → no match (anchor is in src/core/utils/)
        # Actually _expanding_prefixes("src/core/detail/orphan.cpp") gives:
        # ["src/core/detail", "src/core", "src", ""]
        # anchor is in "src/core/utils" — not matching any of those dirs.
        # So this actually won't connect. Let me fix the test.
        assert edges_added == 0

    def test_parent_directory_match(self):
        """Orphan finds anchor in parent directory."""
        G = _make_graph_with_edges([
            ("anchor", "src/core/anchor.cpp"),
            ("orphan", "src/core/detail/orphan.cpp"),
        ], edges=[
            ("anchor", "anchor"),
        ])
        nodes_db = {
            "orphan": {"rel_path": "src/core/detail/orphan.cpp",
                       "symbol_name": "orphan"},
        }
        db = _make_db_mock(nodes_db)

        edges_added = _resolve_orphans_by_directory(db, G, ["orphan"])

        assert edges_added == 1
        assert G.has_edge("orphan", "anchor")

    def test_multiple_orphans_same_directory(self):
        """Multiple orphans in same directory all connect to the anchor."""
        G = _make_graph_with_edges([
            ("anchor", "src/core/anchor.cpp"),
            ("o1", "src/core/o1.cpp"),
            ("o2", "src/core/o2.cpp"),
            ("o3", "src/core/o3.cpp"),
        ], edges=[
            ("anchor", "anchor"),
        ])
        nodes_db = {
            "o1": {"rel_path": "src/core/o1.cpp", "symbol_name": "o1"},
            "o2": {"rel_path": "src/core/o2.cpp", "symbol_name": "o2"},
            "o3": {"rel_path": "src/core/o3.cpp", "symbol_name": "o3"},
        }
        db = _make_db_mock(nodes_db)

        edges_added = _resolve_orphans_by_directory(db, G, ["o1", "o2", "o3"])

        assert edges_added == 3
        assert G.has_edge("o1", "anchor")
        assert G.has_edge("o2", "anchor")
        assert G.has_edge("o3", "anchor")

    def test_highest_degree_preferred(self):
        """Among multiple anchors, the highest-degree one is chosen."""
        G = _make_graph_with_edges([
            ("a_low", "src/a_low.cpp"),
            ("a_high", "src/a_high.cpp"),
            ("other1", "lib/o1.cpp"),
            ("other2", "lib/o2.cpp"),
            ("orphan", "src/orphan.cpp"),
        ], edges=[
            ("a_low", "other1"),        # a_low degree = 1
            ("a_high", "other1"),       # a_high degree = 3
            ("a_high", "other2"),
            ("other2", "a_high"),
        ])
        nodes_db = {
            "orphan": {"rel_path": "src/orphan.cpp", "symbol_name": "orphan"},
        }
        db = _make_db_mock(nodes_db)

        _resolve_orphans_by_directory(db, G, ["orphan"])

        assert G.has_edge("orphan", "a_high")

    def test_orphan_without_path_skipped(self):
        """Orphan with no rel_path in DB is gracefully skipped."""
        G = _make_graph_with_edges([
            ("anchor", "src/a.cpp"),
            ("orphan", ""),
        ], edges=[("anchor", "anchor")])
        nodes_db = {
            "orphan": {"rel_path": "", "symbol_name": "orphan"},
        }
        db = _make_db_mock(nodes_db)

        edges_added = _resolve_orphans_by_directory(db, G, ["orphan"])
        assert edges_added == 0

    def test_orphan_not_in_db_skipped(self):
        """Orphan missing from DB is skipped silently."""
        G = _make_graph_with_edges([
            ("anchor", "src/a.cpp"),
            ("orphan", "src/b.cpp"),
        ], edges=[("anchor", "anchor")])
        db = _make_db_mock({})  # empty DB

        edges_added = _resolve_orphans_by_directory(db, G, ["orphan"])
        assert edges_added == 0

    def test_no_anchors_in_any_directory(self):
        """When ALL nodes are orphans, no edges can be added."""
        G = _make_graph_with_edges([
            ("o1", "src/a.cpp"),
            ("o2", "lib/b.cpp"),
        ])
        nodes_db = {
            "o1": {"rel_path": "src/a.cpp", "symbol_name": "o1"},
            "o2": {"rel_path": "lib/b.cpp", "symbol_name": "o2"},
        }
        db = _make_db_mock(nodes_db)

        edges_added = _resolve_orphans_by_directory(db, G, ["o1", "o2"])
        assert edges_added == 0

    def test_empty_orphan_list(self):
        """No-op on empty orphan list."""
        G = nx.MultiDiGraph()
        db = _make_db_mock({})
        assert _resolve_orphans_by_directory(db, G, []) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Integration: resolve_orphans with directory fallback
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveOrphansIntegration:

    def test_directory_fallback_when_fts5_fails(self):
        """Full pipeline: FTS5 returns nothing → directory fallback kicks in."""
        G = _make_graph_with_edges([
            ("anchor", "src/core/anchor.cpp"),
            ("orphan", "src/core/orphan.cpp"),
        ], edges=[("anchor", "anchor")])

        nodes_db = {
            "anchor": {
                "node_id": "anchor",
                "rel_path": "src/core/anchor.cpp",
                "symbol_name": "anchor",
                "source_text": "void anchor() {}",
            },
            "orphan": {
                "node_id": "orphan",
                "rel_path": "src/core/orphan.cpp",
                "symbol_name": "xyz_unique_name_no_fts_match",
                "source_text": "void orphan() {}",
            },
        }
        db = _make_db_mock(nodes_db)

        stats = resolve_orphans(db, G, embedding_fn=None)

        assert stats["orphan_count"] == 1
        assert stats["directory_edges_added"] >= 1
        assert stats["orphans_remaining"] == 0

    def test_fts5_resolves_before_directory(self):
        """If FTS5 resolves an orphan, directory fallback is not needed."""
        G = _make_graph_with_edges([
            ("anchor", "src/core/anchor.cpp"),
            ("orphan", "src/core/orphan.cpp"),
        ], edges=[("anchor", "anchor")])

        nodes_db = {
            "anchor": {
                "node_id": "anchor",
                "rel_path": "src/core/anchor.cpp",
                "symbol_name": "anchor",
            },
            "orphan": {
                "node_id": "orphan",
                "rel_path": "src/core/orphan.cpp",
                "symbol_name": "findme",
            },
        }
        db = MagicMock()
        db.vec_available = False

        def get_node(nid):
            return nodes_db.get(nid)

        def search_fts5(query, limit=3):
            if query == "findme":
                return [{"node_id": "anchor"}]
            return []

        db.get_node = get_node
        db.search_fts5 = search_fts5

        with patch.dict(
            os.environ,
            {
                "DEEPWIKI_ORPHAN_CASCADE_V2": "0",
                "DEEPWIKI_ORPHAN_LEXICAL_TIERED": "0",
                "DEEPWIKI_FTS_STOPWORD_GATE": "0",
            },
        ):
            stats = resolve_orphans(db, G, embedding_fn=None)

        assert stats["lexical_edges_added"] >= 1
        assert stats["directory_edges_added"] == 0
        assert stats["orphans_remaining"] == 0

    def test_mixed_resolution(self):
        """Some orphans resolved by FTS5, others by directory fallback."""
        G = _make_graph_with_edges([
            ("anchor1", "src/core/a.cpp"),
            ("anchor2", "lib/util/b.cpp"),
            ("orphan_fts", "src/core/fts.cpp"),
            ("orphan_dir", "lib/util/dir.cpp"),
        ], edges=[
            ("anchor1", "anchor2"),
        ])

        nodes_db = {
            "anchor1": {
                "node_id": "anchor1",
                "rel_path": "src/core/a.cpp",
                "symbol_name": "anchor1",
            },
            "anchor2": {
                "node_id": "anchor2",
                "rel_path": "lib/util/b.cpp",
                "symbol_name": "anchor2",
            },
            "orphan_fts": {
                "node_id": "orphan_fts",
                "rel_path": "src/core/fts.cpp",
                "symbol_name": "fts_match",
            },
            "orphan_dir": {
                "node_id": "orphan_dir",
                "rel_path": "lib/util/dir.cpp",
                "symbol_name": "xyzzy_no_fts_match",
            },
        }

        db = MagicMock()
        db.vec_available = False

        def get_node(nid):
            return nodes_db.get(nid)

        def search_fts5(query, limit=3):
            if query == "fts_match":
                return [{"node_id": "anchor1"}]
            return []

        db.get_node = get_node
        db.search_fts5 = search_fts5

        with patch.dict(
            os.environ,
            {
                "DEEPWIKI_ORPHAN_CASCADE_V2": "0",
                "DEEPWIKI_ORPHAN_LEXICAL_TIERED": "0",
                "DEEPWIKI_FTS_STOPWORD_GATE": "0",
            },
        ):
            stats = resolve_orphans(db, G, embedding_fn=None)

        assert stats["orphan_count"] == 2
        assert stats["lexical_edges_added"] >= 1
        assert stats["directory_edges_added"] >= 1
        assert stats["orphans_remaining"] == 0

    def test_large_scale_directory_resolution(self):
        """Simulate fmtlib-like scenario: many orphans, no embeddings."""
        # 100 connected nodes across 5 directories
        # 50 orphans across same directories
        connected_nodes = []
        edges = []
        for d_idx, d in enumerate(["include/fmt", "src", "test", "doc", "include/fmt/detail"]):
            for i in range(20):
                nid = f"c_{d_idx}_{i}"
                connected_nodes.append((nid, f"{d}/file{i}.cpp"))
                if i > 0:
                    prev = f"c_{d_idx}_{i-1}"
                    edges.append((nid, prev))

        orphan_nodes = []
        for d_idx, d in enumerate(["include/fmt", "src", "test", "doc", "include/fmt/detail"]):
            for i in range(10):
                nid = f"o_{d_idx}_{i}"
                orphan_nodes.append((nid, f"{d}/orphan{i}.cpp"))

        G = _make_graph_with_edges(connected_nodes + orphan_nodes, edges)

        nodes_db = {}
        for nid, path in orphan_nodes:
            nodes_db[nid] = {
                "node_id": nid,
                "rel_path": path,
                "symbol_name": f"unique_{nid}",
            }
        db = _make_db_mock(nodes_db)

        orphan_ids = [nid for nid, _ in orphan_nodes]
        edges_added = _resolve_orphans_by_directory(db, G, orphan_ids)

        assert edges_added == 50  # all 50 orphans should be connected
        # No orphans from our list remain
        remaining = [n for n in orphan_ids if
                     G.in_degree(n) == 0 and G.out_degree(n) == 0]
        assert len(remaining) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic edge weight handling in apply_edge_weights
# ═══════════════════════════════════════════════════════════════════════════

class TestSyntheticEdgeWeights:
    """Validate that synthetic edges use structural-only in-degree and
    receive the SYNTHETIC_WEIGHT_FLOOR."""

    def test_structural_edges_use_structural_in_degree_only(self):
        """Synthetic edges pointing AT a node should NOT inflate its
        in-degree for weight calculation of structural edges."""
        G = nx.MultiDiGraph()
        G.add_node("target")
        G.add_node("caller")

        # 1 structural edge → target (structural in-degree = 1)
        G.add_edge("caller", "target", relationship_type="calls",
                   edge_class="structural")

        # 50 directory_link edges → target (should NOT inflate in-degree)
        for i in range(50):
            oid = f"dir_orphan_{i}"
            G.add_node(oid)
            G.add_edge(oid, "target", relationship_type="directory_link",
                       edge_class="directory", created_by="dir_proximity_fallback")

        apply_edge_weights(G)

        # The structural edge weight should be based on in-degree=1
        # (not 51), so weight = 1/log(1+2) = 1/log(3) ≈ 0.9102
        structural_edge = G.edges["caller", "target", 0]
        expected = 1.0 / math.log(1 + 2)
        assert abs(structural_edge["weight"] - expected) < 0.001, (
            f"Structural weight {structural_edge['weight']:.4f} != "
            f"expected {expected:.4f} (in_degree should be 1, not 51)"
        )

    def test_synthetic_edges_get_weight_floor(self):
        """directory_link edges get at least SYNTHETIC_WEIGHT_FLOOR."""
        G = nx.MultiDiGraph()
        G.add_node("anchor")
        G.add_node("orphan")

        # Many structural edges to make anchor high in-degree
        for i in range(200):
            src = f"caller_{i}"
            G.add_node(src)
            G.add_edge(src, "anchor", relationship_type="calls",
                       edge_class="structural")

        # One directory_link edge
        G.add_edge("orphan", "anchor", relationship_type="directory_link",
                   edge_class="directory", created_by="dir_proximity_fallback")

        apply_edge_weights(G)

        dir_edge = G.edges["orphan", "anchor", 0]
        assert dir_edge["weight"] >= SYNTHETIC_WEIGHT_FLOOR, (
            f"Directory edge weight {dir_edge['weight']:.4f} < "
            f"floor {SYNTHETIC_WEIGHT_FLOOR}"
        )

    def test_lexical_edges_get_weight_floor(self):
        """lexical_link edges also get the weight floor."""
        G = nx.MultiDiGraph()
        G.add_node("a")
        G.add_node("b")

        for i in range(100):
            src = f"s{i}"
            G.add_node(src)
            G.add_edge(src, "b", relationship_type="calls",
                       edge_class="structural")

        G.add_edge("a", "b", relationship_type="lexical_link",
                   edge_class="lexical", created_by="fts5_lexical")

        apply_edge_weights(G)

        lex_edge = G.edges["a", "b", 0]
        assert lex_edge["weight"] >= SYNTHETIC_WEIGHT_FLOOR

    def test_doc_edges_get_weight_floor(self):
        """Doc edges (hyperlink, proximity) get the weight floor."""
        G = nx.MultiDiGraph()
        G.add_node("doc")
        G.add_node("code")

        for i in range(50):
            src = f"c{i}"
            G.add_node(src)
            G.add_edge(src, "code", relationship_type="calls",
                       edge_class="structural")

        G.add_edge("doc", "code", relationship_type="hyperlink",
                   edge_class="doc", created_by="md_hyperlink")

        apply_edge_weights(G)

        doc_edge = G.edges["doc", "code", 0]
        assert doc_edge["weight"] >= SYNTHETIC_WEIGHT_FLOOR

    def test_stats_include_synthetic_count(self):
        """Stats dict should report how many edges got the floor."""
        G = nx.MultiDiGraph()
        G.add_node("a")
        G.add_node("b")
        G.add_node("c")

        G.add_edge("a", "b", edge_class="structural")
        G.add_edge("c", "b", edge_class="directory")

        stats = apply_edge_weights(G)

        assert stats["synthetic_floored"] == 1
        assert stats["edges_weighted"] == 2

    def test_empty_graph_safe(self):
        """Empty graph returns zero stats."""
        G = nx.MultiDiGraph()
        stats = apply_edge_weights(G)
        assert stats["edges_weighted"] == 0

    def test_pure_structural_graph_unchanged(self):
        """On a graph with no synthetic edges, behaviour is identical
        to the old formula."""
        G = nx.MultiDiGraph()
        for i in range(10):
            G.add_node(f"n{i}")
        # n0 has in-degree 5 (from n1..n5)
        for i in range(1, 6):
            G.add_edge(f"n{i}", "n0", edge_class="structural",
                       relationship_type="calls")

        apply_edge_weights(G)

        expected_w = 1.0 / math.log(5 + 2)  # ~0.5145
        for u, v, data in G.edges(data=True):
            assert abs(data["weight"] - expected_w) < 0.001

    def test_high_degree_anchor_with_mixed_edges(self):
        """Realistic scenario: anchor with structural + directory edges.
        Structural weight based only on structural in-degree;
        directory edges get the floor."""
        G = nx.MultiDiGraph()
        G.add_node("anchor")

        # 20 structural callers
        for i in range(20):
            n = f"caller_{i}"
            G.add_node(n)
            G.add_edge(n, "anchor", edge_class="structural",
                       relationship_type="calls")

        # 100 orphans via directory_link
        for i in range(100):
            n = f"orphan_{i}"
            G.add_node(n)
            G.add_edge(n, "anchor", edge_class="directory",
                       relationship_type="directory_link")

        apply_edge_weights(G)

        # Structural edges: w = 1/log(20+2) ≈ 0.3234
        expected_structural = 1.0 / math.log(20 + 2)
        for i in range(20):
            e = G.edges[f"caller_{i}", "anchor", 0]
            assert abs(e["weight"] - expected_structural) < 0.001

        # Directory edges: floor dominates (0.5 > 0.3234)
        for i in range(100):
            e = G.edges[f"orphan_{i}", "anchor", 0]
            assert e["weight"] >= SYNTHETIC_WEIGHT_FLOOR
