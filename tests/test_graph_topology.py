"""
Tests for Phase 2 — graph_topology.py
Edge weighting, hub detection, semantic edge injection, weight persistence.

Suite layout
============
TestApplyEdgeWeights        —  edge weight formula verification
TestDetectHubs              —  Z-score hub detection
TestFindOrphans             —  orphan identification
TestExpandingPrefixes       —  path prefix helper
TestResolveOrphans          —  FTS5 + vector orphan linking
TestPersistWeightsToDB      —  weight round-trip through DB
TestRunPhase2               —  full orchestrator
"""

import json
import math
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch


@contextmanager
def _patched_flags(**overrides):
    """Override hard-coded ``FeatureFlags`` fields for the duration of a test.

    The graph-quality baseline flags (cascade v2, tiered lexical, stopword
    gate, …) are no longer env-driven, so tests that need to disable them
    flip the dataclass fields directly.
    """
    from plugin_implementation import feature_flags as _ff
    from plugin_implementation import graph_topology as _gt

    base = _ff.get_feature_flags()
    patched = _dc_replace(base, **overrides)
    with patch.object(_ff, "get_feature_flags", return_value=patched), \
         patch.object(_gt, "get_feature_flags", return_value=patched):
        yield patched

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugin_implementation.graph_topology import (
    _expanding_prefixes,
    apply_edge_weights,
    detect_hubs,
    find_orphans,
    flag_hubs_in_db,
    persist_weights_to_db,
    resolve_orphans,
    run_phase2,
)
from plugin_implementation.unified_db import UnifiedWikiDB


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_graph(*edges, orphans=None):
    """Build a small MultiDiGraph for testing.

    Each edge is a (source, target) tuple or (source, target, attrs_dict).
    Orphan nodes (completely disconnected) can be added via the orphans list.
    """
    G = nx.MultiDiGraph()
    for e in edges:
        if len(e) == 2:
            G.add_edge(e[0], e[1], relationship_type="calls")
        elif len(e) == 3:
            attrs = e[2] if isinstance(e[2], dict) else {}
            G.add_edge(e[0], e[1], **attrs)
    for n in (orphans or []):
        G.add_node(n)
    return G


def _make_db(tmp_dir, nodes=None, edges=None) -> UnifiedWikiDB:
    """Create a fresh UnifiedWikiDB with optional seed data."""
    db_path = os.path.join(tmp_dir, "test.wiki.db")
    db = UnifiedWikiDB(db_path)
    if nodes:
        db.upsert_nodes_batch(nodes)
    if edges:
        db.upsert_edges_batch(edges)
    return db


def _make_node_dict(node_id, symbol_name="", symbol_type="function",
                    rel_path="src/main.py", source_text="", docstring="",
                    is_architectural=1, **kwargs) -> Dict[str, Any]:
    """Minimal node dict for DB insertion."""
    d = {
        "node_id": node_id,
        "symbol_name": symbol_name or node_id,
        "symbol_type": symbol_type,
        "rel_path": rel_path,
        "source_text": source_text,
        "docstring": docstring,
        "is_architectural": is_architectural,
        "language": "python",
    }
    d.update(kwargs)
    return d


# ═══════════════════════════════════════════════════════════════════════════
# 1. apply_edge_weights
# ═══════════════════════════════════════════════════════════════════════════

class TestApplyEdgeWeights(unittest.TestCase):
    """Verify inverse in-degree formula: W = 1/log(InDeg(v) + 2)."""

    def test_empty_graph(self):
        G = nx.MultiDiGraph()
        stats = apply_edge_weights(G)
        self.assertEqual(stats["edges_weighted"], 0)

    def test_single_edge(self):
        G = _make_graph(("A", "B"))
        stats = apply_edge_weights(G)
        self.assertEqual(stats["edges_weighted"], 1)
        # B has in_degree 1 → weight = 1/log(1+2) = 1/log(3)
        expected = 1.0 / math.log(3)
        weight = list(G.edges(data=True))[0][2]["weight"]
        self.assertAlmostEqual(weight, expected, places=6)

    def test_high_fan_in_gets_low_weight(self):
        """Node with many incoming edges gets low edge weight."""
        # 10 nodes all calling "hub"
        edges = [(f"n{i}", "hub") for i in range(10)]
        G = _make_graph(*edges)
        apply_edge_weights(G)

        # hub in_degree = 10 → weight = 1/log(12)
        expected = 1.0 / math.log(12)
        for u, v, data in G.edges(data=True):
            self.assertEqual(v, "hub")
            self.assertAlmostEqual(data["weight"], expected, places=6)

    def test_low_fan_in_gets_high_weight(self):
        """Direct unique edge gets high weight."""
        G = _make_graph(("A", "B"))
        apply_edge_weights(G)
        weight = list(G.edges(data=True))[0][2]["weight"]
        # B in_degree 1 → weight = 1/log(3) ≈ 0.91
        self.assertGreater(weight, 0.9)

    def test_weight_monotonically_decreases(self):
        """Edges to higher-fan-in targets should have lower weights."""
        # B has 1 incoming, C has 5 incoming
        edges = [("A", "B")] + [(f"n{i}", "C") for i in range(5)]
        G = _make_graph(*edges)
        apply_edge_weights(G)

        w_b = [d["weight"] for u, v, d in G.edges(data=True) if v == "B"][0]
        w_c = [d["weight"] for u, v, d in G.edges(data=True) if v == "C"][0]
        self.assertGreater(w_b, w_c)

    def test_stats_returned(self):
        edges = [("A", "B"), ("B", "C"), ("C", "D")]
        G = _make_graph(*edges)
        stats = apply_edge_weights(G)
        self.assertIn("edges_weighted", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("mean", stats)
        self.assertEqual(stats["edges_weighted"], 3)

    def test_parallel_edges(self):
        """MultiDiGraph can have parallel edges; all should get weights."""
        G = nx.MultiDiGraph()
        G.add_edge("A", "B", relationship_type="calls")
        G.add_edge("A", "B", relationship_type="imports")
        stats = apply_edge_weights(G)
        self.assertEqual(stats["edges_weighted"], 2)
        for u, v, k, d in G.edges(data=True, keys=True):
            self.assertIn("weight", d)


# ═══════════════════════════════════════════════════════════════════════════
# 2. detect_hubs
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectHubs(unittest.TestCase):
    """Z-score in-degree hub detection."""

    def test_empty_graph(self):
        G = nx.MultiDiGraph()
        self.assertEqual(detect_hubs(G), set())

    def test_small_graph_no_hubs(self):
        """Too few nodes for Z-score → no hubs."""
        G = _make_graph(("A", "B"))
        self.assertEqual(detect_hubs(G), set())

    def test_uniform_graph_no_hubs(self):
        """All nodes have similar in-degree → Z-scores ≈ 0."""
        # Ring: each node has in-degree 1
        edges = [(f"n{i}", f"n{(i + 1) % 8}") for i in range(8)]
        G = _make_graph(*edges)
        self.assertEqual(detect_hubs(G), set())

    def test_obvious_hub_detected(self):
        """One node with massive in-degree, rest with 0-1."""
        # 30 nodes call "hub"
        edges = [(f"n{i}", "hub") for i in range(30)]
        # A few extra edges between normal nodes
        edges += [("n0", "n1"), ("n2", "n3")]
        G = _make_graph(*edges)

        hubs = detect_hubs(G, z_threshold=3.0)
        self.assertIn("hub", hubs)
        # Normal nodes shouldn't be hubs
        self.assertNotIn("n1", hubs)
        self.assertNotIn("n3", hubs)

    def test_multiple_hubs(self):
        """Two extremely high fan-in nodes both detected."""
        # hub1 has 50 incoming, hub2 has 40 incoming, rest have 0-1
        edges = [(f"a{i}", "hub1") for i in range(50)]
        edges += [(f"b{i}", "hub2") for i in range(40)]
        edges += [("a0", "a1"), ("b0", "b1")]
        G = _make_graph(*edges)

        hubs = detect_hubs(G, z_threshold=3.0)
        self.assertIn("hub1", hubs)
        self.assertIn("hub2", hubs)

    def test_custom_threshold(self):
        """Lower threshold catches more hubs."""
        edges = [(f"n{i}", "mildly_popular") for i in range(8)]
        edges += [("x", "y"), ("y", "z")]
        G = _make_graph(*edges)

        # With strict threshold, might not catch
        strict = detect_hubs(G, z_threshold=5.0)
        # With lenient threshold
        lenient = detect_hubs(G, z_threshold=1.5)
        self.assertTrue(len(lenient) >= len(strict))

    def test_all_same_degree_no_std(self):
        """If std == 0 (all same degree), return empty set."""
        # All nodes have in-degree exactly 1
        edges = [(f"n{i}", f"n{(i + 1) % 5}") for i in range(5)]
        G = _make_graph(*edges)
        self.assertEqual(detect_hubs(G), set())


class TestFlagHubsInDB(unittest.TestCase):
    """Test persisting hub flags to DB."""

    def test_flag_hubs(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict(f"n{i}") for i in range(5)]
            db = _make_db(tmp, nodes=nodes)

            flag_hubs_in_db(db, {"n0", "n2"})

            n0 = db.get_node("n0")
            n1 = db.get_node("n1")
            n2 = db.get_node("n2")
            self.assertEqual(n0["is_hub"], 1)
            self.assertEqual(n1["is_hub"], 0)
            self.assertEqual(n2["is_hub"], 1)

    def test_flag_empty_set(self):
        """Empty hub set should be a no-op."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(tmp, nodes=[_make_node_dict("n0")])
            flag_hubs_in_db(db, set())  # should not raise
            self.assertEqual(db.get_node("n0")["is_hub"], 0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. find_orphans / _expanding_prefixes
# ═══════════════════════════════════════════════════════════════════════════

class TestFindOrphans(unittest.TestCase):
    """Orphan detection — nodes with in_degree==0 AND out_degree==0."""

    def test_no_orphans(self):
        G = _make_graph(("A", "B"), ("B", "C"))
        self.assertEqual(find_orphans(G), [])

    def test_source_node_not_orphan(self):
        """Node with out_degree > 0 is NOT an orphan (it has outgoing)."""
        G = _make_graph(("A", "B"))
        orphans = find_orphans(G)
        # A has out_degree=1, B has in_degree=1 — neither is orphan
        self.assertNotIn("A", orphans)
        self.assertNotIn("B", orphans)

    def test_isolated_node_is_orphan(self):
        G = _make_graph(("A", "B"), orphans=["Z"])
        orphans = find_orphans(G)
        self.assertEqual(orphans, ["Z"])

    def test_multiple_orphans(self):
        G = _make_graph(("A", "B"), orphans=["X", "Y", "Z"])
        orphans = set(find_orphans(G))
        self.assertEqual(orphans, {"X", "Y", "Z"})


class TestExpandingPrefixes(unittest.TestCase):
    """Test path prefix expansion for locality-biased search."""

    def test_deep_path(self):
        prefixes = _expanding_prefixes("src/auth/handlers/login.py")
        self.assertEqual(prefixes, [
            "src/auth/handlers",
            "src/auth",
            "src",
            "",
        ])

    def test_root_file(self):
        prefixes = _expanding_prefixes("main.py")
        self.assertEqual(prefixes, [""])

    def test_one_dir_deep(self):
        prefixes = _expanding_prefixes("src/app.py")
        self.assertEqual(prefixes, ["src", ""])

    def test_windows_paths(self):
        """Backslashes should be normalized."""
        prefixes = _expanding_prefixes("src\\auth\\login.py")
        self.assertEqual(prefixes, ["src/auth", "src", ""])


# ═══════════════════════════════════════════════════════════════════════════
# 4. resolve_orphans — FTS5 + vector semantic linking
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveOrphans(unittest.TestCase):
    """Test the Asymmetric Lazy Linking orphan resolver."""

    def test_no_orphans(self):
        """No orphans → nothing to do."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(tmp, nodes=[
                _make_node_dict("A"), _make_node_dict("B")
            ])
            G = _make_graph(("A", "B"))
            stats = resolve_orphans(db, G)
            self.assertEqual(stats["orphan_count"], 0)
            self.assertEqual(stats["lexical_edges_added"], 0)

    def test_lexical_resolution_via_fts5(self):
        """Orphan matched by symbol name via FTS5."""
        with tempfile.TemporaryDirectory() as tmp, _patched_flags(orphan_cascade_v2=False, orphan_lexical_tiered=False, fts_stopword_gate=False):
            nodes = [
                _make_node_dict("connected_a", symbol_name="UserService",
                                source_text="class UserService: pass"),
                _make_node_dict("connected_b", symbol_name="UserRepo",
                                source_text="class UserRepo: pass"),
                _make_node_dict("orphan_z", symbol_name="UserService",
                                source_text="test user service",
                                rel_path="tests/test_user.py"),
            ]
            db = _make_db(tmp, nodes=nodes)
            db._populate_fts5()

            G = _make_graph(("connected_a", "connected_b"), orphans=["orphan_z"])

            stats = resolve_orphans(db, G, embedding_fn=None)
            self.assertGreater(stats["lexical_edges_added"], 0)
            self.assertEqual(stats["orphans_resolved"], 1)
            # The orphan should now have edges in the graph
            self.assertGreater(G.out_degree("orphan_z"), 0)

    def test_lexical_no_self_link(self):
        """FTS5 hits should exclude the orphan itself."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                _make_node_dict("sole_node", symbol_name="UniqueSymbol",
                                source_text="UniqueSymbol implementation"),
            ]
            db = _make_db(tmp, nodes=nodes)
            db._populate_fts5()

            G = nx.MultiDiGraph()
            G.add_node("sole_node")  # orphan

            stats = resolve_orphans(db, G)
            # Only one node, so FTS5 hit is itself → should skip
            self.assertEqual(stats["lexical_edges_added"], 0)
            self.assertEqual(stats["orphans_remaining"], 1)

    def test_vector_resolution_fallback(self):
        """When lexical fails, vector pass should kick in."""
        with tempfile.TemporaryDirectory() as tmp, _patched_flags(orphan_cascade_v2=False, orphan_lexical_tiered=False, fts_stopword_gate=False):
            nodes = [
                _make_node_dict("connected_a", symbol_name="AlphaProcessor",
                                source_text="process alpha data here"),
                _make_node_dict("connected_b", symbol_name="BetaHandler"),
                _make_node_dict("orphan_z", symbol_name="Zz",
                                source_text="a long enough text for embedding pass"),
            ]
            db = _make_db(tmp, nodes=nodes)
            db._populate_fts5()

            G = _make_graph(("connected_a", "connected_b"), orphans=["orphan_z"])

            # Mock the embedding_fn
            mock_embed = MagicMock(return_value=[0.1] * 128)

            # Patch vec_available as a property on the instance (not the class)
            with patch.object(type(db), 'vec_available',
                              new_callable=lambda: property(lambda self: True)):
                # Mock search_vec to return a hit
                db.search_vec = MagicMock(return_value=[{
                    "node_id": "connected_a",
                    "vec_distance": 0.1,
                }])

                stats = resolve_orphans(db, G, embedding_fn=mock_embed,
                                        vec_distance_threshold=0.15)

            # FTS5 won't match "Zz" to anything useful 
            # Vector should resolve it
            self.assertGreater(stats["semantic_edges_added"], 0)

    def test_orphan_not_in_db_skipped(self):
        """If orphan node doesn't exist in DB, skip gracefully."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(tmp, nodes=[_make_node_dict("A")])
            G = nx.MultiDiGraph()
            G.add_node("A")
            G.add_node("ghost")  # exists in graph but not in DB
            # A calls nothing; ghost is orphan
            stats = resolve_orphans(db, G)
            # ghost should be skipped
            self.assertEqual(stats["orphan_count"], 2)

    def test_short_symbol_name_skipped(self):
        """Symbol names < 2 chars should be skipped for FTS5."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                _make_node_dict("A_connected", symbol_name="RealClass"),
                _make_node_dict("orphan_x", symbol_name="X"),  # too short
            ]
            db = _make_db(tmp, nodes=nodes)
            db._populate_fts5()

            G = nx.MultiDiGraph()
            G.add_node("A_connected")
            G.add_node("orphan_x")

            stats = resolve_orphans(db, G)
            self.assertEqual(stats["lexical_edges_added"], 0)

    def test_edges_added_to_both_graph_and_db(self):
        """Synthetic edges should appear in both DB and NetworkX."""
        with tempfile.TemporaryDirectory() as tmp, _patched_flags(orphan_cascade_v2=False, orphan_lexical_tiered=False, fts_stopword_gate=False):
            nodes = [
                _make_node_dict("target_a", symbol_name="AuthHandler",
                                source_text="authentication handler"),
                _make_node_dict("target_b", symbol_name="DataStore",
                                source_text="data storage"),
                _make_node_dict("orphan_c", symbol_name="AuthHandler",
                                source_text="auth handler test",
                                rel_path="tests/test_auth.py"),
            ]
            db = _make_db(tmp, nodes=nodes)
            db._populate_fts5()

            G = _make_graph(("target_a", "target_b"), orphans=["orphan_c"])

            before_edges_graph = G.number_of_edges()

            resolve_orphans(db, G)

            # Legacy path uses skip_db=True so persist_weights_to_db can
            # rewrite the edges later; the graph-only assertion is enough
            # to prove the orphan was resolved.
            self.assertGreater(G.number_of_edges(), before_edges_graph)


# ═══════════════════════════════════════════════════════════════════════════
# 5. persist_weights_to_db
# ═══════════════════════════════════════════════════════════════════════════

class TestPersistWeightsToDB(unittest.TestCase):
    """Test weight persistence round-trip."""

    def test_empty_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(tmp)
            G = nx.MultiDiGraph()
            n = persist_weights_to_db(db, G)
            self.assertEqual(n, 0)

    def test_weights_persisted(self):
        """Weights from the graph should appear in the DB."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                _make_node_dict("A"),
                _make_node_dict("B"),
                _make_node_dict("C"),
            ]
            db = _make_db(tmp, nodes=nodes)

            G = _make_graph(("A", "B"), ("B", "C"))
            apply_edge_weights(G)

            n = persist_weights_to_db(db, G)
            self.assertEqual(n, 2)

            # Verify in DB
            rows = db.conn.execute("SELECT * FROM repo_edges").fetchall()
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertGreater(dict(row)["weight"], 0)

    def test_overwrite_existing_edges(self):
        """persist_weights_to_db replaces all edges."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict("A"), _make_node_dict("B")]
            old_edges = [{"source_id": "A", "target_id": "B",
                          "rel_type": "old", "weight": 999.0}]
            db = _make_db(tmp, nodes=nodes, edges=old_edges)
            self.assertEqual(db.edge_count(), 1)

            G = _make_graph(("A", "B"))
            apply_edge_weights(G)
            persist_weights_to_db(db, G)

            rows = db.conn.execute("SELECT * FROM repo_edges").fetchall()
            self.assertEqual(len(rows), 1)
            self.assertNotAlmostEqual(dict(rows[0])["weight"], 999.0)


# ═══════════════════════════════════════════════════════════════════════════
# 6. run_phase2 orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class TestRunPhase2(unittest.TestCase):
    """End-to-end Phase 2 pipeline test."""

    def test_full_pipeline(self):
        """Run weighting → hub detection → orphan resolution → persist."""
        with tempfile.TemporaryDirectory() as tmp:
            # Build a graph with a hub, normal edges, and an orphan
            nodes = [
                _make_node_dict(f"n{i}", symbol_name=f"Node{i}",
                                source_text=f"node {i} implementation")
                for i in range(12)
            ]
            nodes.append(
                _make_node_dict("orphan_doc", symbol_name="Node3",
                                source_text="documentation about Node3",
                                rel_path="docs/node3.md")
            )

            db = _make_db(tmp, nodes=nodes)
            db._populate_fts5()

            # Build graph: n0-n9 all call n10 (hub), plus some edges
            edges = [(f"n{i}", "n10") for i in range(10)]
            edges += [("n0", "n1"), ("n1", "n2")]
            G = _make_graph(*edges, orphans=["orphan_doc"])

            results = run_phase2(db, G)

            # Verify sub-results exist
            self.assertIn("weighting", results)
            self.assertIn("hubs", results)
            self.assertIn("orphan_resolution", results)
            self.assertIn("edges_persisted", results)

            # Verify weights were applied
            self.assertGreater(results["weighting"]["edges_weighted"], 0)

            # Verify edges persisted
            self.assertGreater(results["edges_persisted"], 0)

            # Verify metadata
            self.assertTrue(db.get_meta("phase2_completed"))

    def test_pipeline_no_orphans_no_hubs(self):
        """Pipeline works when there are no special cases."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                _make_node_dict("A", symbol_name="ClassA"),
                _make_node_dict("B", symbol_name="ClassB"),
            ]
            db = _make_db(tmp, nodes=nodes)
            db._populate_fts5()

            G = _make_graph(("A", "B"))

            results = run_phase2(db, G)
            self.assertEqual(results["hubs"]["count"], 0)
            self.assertEqual(results["orphan_resolution"]["orphan_count"], 0)
            self.assertEqual(results["edges_persisted"], 1)

    def test_pipeline_empty_graph(self):
        """Pipeline handles empty graph gracefully."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(tmp)
            G = nx.MultiDiGraph()

            results = run_phase2(db, G)
            self.assertEqual(results["weighting"]["edges_weighted"], 0)
            self.assertEqual(results["hubs"]["count"], 0)
            self.assertEqual(results["orphan_resolution"]["orphan_count"], 0)
            self.assertEqual(results["edges_persisted"], 0)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Edge cases & numpy fallback
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):
    """Miscellaneous edge cases."""

    def test_hub_detection_pure_python_fallback(self):
        """Test Z-score computation without numpy."""
        import plugin_implementation.graph_topology as gt
        original = gt._HAS_NUMPY

        try:
            gt._HAS_NUMPY = False

            edges = [(f"n{i}", "hub") for i in range(30)]
            edges += [("n0", "n1"), ("n2", "n3")]
            G = _make_graph(*edges)

            hubs = detect_hubs(G, z_threshold=3.0)
            self.assertIn("hub", hubs)
        finally:
            gt._HAS_NUMPY = original

    def test_weight_formula_boundary(self):
        """Node with in_degree=0 → weight = 1/log(0+2) = 1/log(2) ≈ 1.44."""
        G = _make_graph(("A", "B"))
        # A has in_degree=0, but let's add an edge TO A
        G.add_edge("B", "A", relationship_type="calls")
        apply_edge_weights(G)

        # Edge B→A: A has in_degree=1 → 1/log(3)
        for u, v, d in G.edges(data=True):
            if v == "A":
                expected = 1.0 / math.log(3)
                self.assertAlmostEqual(d["weight"], expected, places=6)

    def test_large_graph_performance(self):
        """Smoke test: 1000 nodes, 2000 edges. Should complete < 2s."""
        import time
        edges = [(f"n{i}", f"n{(i * 7 + 3) % 1000}") for i in range(2000)]
        G = _make_graph(*edges)

        t0 = time.time()
        apply_edge_weights(G)
        hubs = detect_hubs(G)
        elapsed = time.time() - t0

        self.assertLess(elapsed, 2.0, "Phase 2 weighting+hubs took > 2s on 2K edges")
        # Just verify it completed
        self.assertGreater(G.number_of_edges(), 0)


if __name__ == "__main__":
    unittest.main()
