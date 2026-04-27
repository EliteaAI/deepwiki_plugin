"""
Tests for Phase 3 — graph_clustering.py
Macro-clustering, micro-clustering, page sizing, hub re-integration.

Suite layout
============
TestAutoResolution           —  γ tuning by graph size
TestMacroClustering          —  Louvain macro-level partitioning
TestMicroClustering          —  Sub-Louvain within macro-clusters
TestDynamicPageSizing        —  Merge small / split oversized
TestHubReintegration         —  Majority-vote hub assignment
TestPersistClusters          —  DB round-trip
TestRunPhase3                —  Full orchestrator
TestEdgeCases                —  Empty graphs, single-node, etc.
"""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugin_implementation.graph_clustering import (
    MICRO_CLUSTER_RULES,
    _find_nearest_cluster,
    _recursive_split,
    _to_weighted_undirected,
    apply_page_sizing,
    auto_resolution,
    macro_cluster,
    micro_cluster,
    micro_cluster_all,
    persist_clusters,
    reintegrate_hubs,
    run_phase3,
)
from plugin_implementation.graph_topology import apply_edge_weights, detect_hubs
from plugin_implementation.unified_db import UnifiedWikiDB


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_graph(*edges, orphans=None):
    """Build a small MultiDiGraph. Each edge = (src, tgt) or (src, tgt, attrs)."""
    G = nx.MultiDiGraph()
    for e in edges:
        if len(e) == 2:
            G.add_edge(e[0], e[1], relationship_type="calls", weight=1.0)
        elif len(e) == 3:
            attrs = e[2] if isinstance(e[2], dict) else {}
            attrs.setdefault("weight", 1.0)
            G.add_edge(e[0], e[1], **attrs)
    for n in (orphans or []):
        G.add_node(n)
    return G


def _make_db(tmp_dir, nodes=None, edges=None) -> UnifiedWikiDB:
    db_path = os.path.join(tmp_dir, "test.wiki.db")
    db = UnifiedWikiDB(db_path)
    if nodes:
        db.upsert_nodes_batch(nodes)
    if edges:
        db.upsert_edges_batch(edges)
    return db


def _make_node_dict(node_id, **kwargs) -> Dict[str, Any]:
    d = {
        "node_id": node_id,
        "symbol_name": kwargs.get("symbol_name", node_id),
        "symbol_type": kwargs.get("symbol_type", "function"),
        "rel_path": kwargs.get("rel_path", "src/main.py"),
        "source_text": kwargs.get("source_text", ""),
        "is_architectural": kwargs.get("is_architectural", 1),
        "language": kwargs.get("language", "python"),
    }
    d.update(kwargs)
    return d


def _build_clusterable_graph():
    """Build a graph with two clear communities connected by a thin bridge.

    Community A: a0-a1-a2-a3-a4 (dense intra-edges)
    Community B: b0-b1-b2-b3-b4 (dense intra-edges)
    Bridge: a0 → b0 (single weak link)
    """
    G = nx.MultiDiGraph()
    # Community A — dense
    for i in range(5):
        for j in range(5):
            if i != j:
                G.add_edge(f"a{i}", f"a{j}", weight=1.0, relationship_type="calls")
    # Community B — dense
    for i in range(5):
        for j in range(5):
            if i != j:
                G.add_edge(f"b{i}", f"b{j}", weight=1.0, relationship_type="calls")
    # Thin bridge
    G.add_edge("a0", "b0", weight=0.1, relationship_type="calls")
    return G


# ═══════════════════════════════════════════════════════════════════════════
# 1. auto_resolution
# ═══════════════════════════════════════════════════════════════════════════

class TestAutoResolution(unittest.TestCase):

    def test_small_repo(self):
        # 100 nodes: max(0.3, 1.0 - 0.2*2.0) = 0.6
        self.assertAlmostEqual(auto_resolution(100), 0.6, places=1)

    def test_medium_repo(self):
        # 1000 nodes: max(0.3, 1.0 - 0.2*3.0) = 0.4
        r = auto_resolution(1000)
        self.assertGreaterEqual(r, 0.3)
        self.assertLessEqual(r, 0.5)

    def test_large_repo(self):
        # 50000 nodes: max(0.3, 1.0 - 0.2*4.7) = max(0.3, 0.06) = 0.3
        r = auto_resolution(50000)
        self.assertAlmostEqual(r, 0.3, places=1)

    def test_minimum_clamped(self):
        self.assertEqual(auto_resolution(10**10), 0.3)

    def test_single_node(self):
        self.assertEqual(auto_resolution(1), 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Macro-Clustering
# ═══════════════════════════════════════════════════════════════════════════

class TestMacroClustering(unittest.TestCase):

    def test_two_communities_separated(self):
        """Two dense clusters connected by a thin bridge → separate macro-clusters."""
        G = _build_clusterable_graph()
        assignments = macro_cluster(G, hubs=set(), resolution=1.0)

        # All a-nodes should be in one cluster, all b-nodes in another
        a_clusters = {assignments[f"a{i}"] for i in range(5)}
        b_clusters = {assignments[f"b{i}"] for i in range(5)}
        self.assertEqual(len(a_clusters), 1, "A-nodes should be in one cluster")
        self.assertEqual(len(b_clusters), 1, "B-nodes should be in one cluster")
        self.assertNotEqual(a_clusters, b_clusters, "Clusters should differ")

    def test_hubs_excluded(self):
        """Hub nodes should not appear in macro-clustering output."""
        G = _build_clusterable_graph()
        assignments = macro_cluster(G, hubs={"a0"})
        self.assertNotIn("a0", assignments)

    def test_empty_after_hub_removal(self):
        """If all nodes are hubs, return empty."""
        G = _make_graph(("A", "B"))
        assignments = macro_cluster(G, hubs={"A", "B"})
        self.assertEqual(assignments, {})

    def test_single_community(self):
        """Fully connected graph → one cluster."""
        edges = [(f"n{i}", f"n{j}") for i in range(5) for j in range(5) if i != j]
        G = _make_graph(*edges)
        assignments = macro_cluster(G, hubs=set(), resolution=1.0)
        self.assertEqual(len(set(assignments.values())), 1)

    def test_resolution_affects_cluster_count(self):
        """Lower resolution → fewer clusters."""
        G = _build_clusterable_graph()
        high_res = macro_cluster(G, hubs=set(), resolution=2.0)
        low_res = macro_cluster(G, hubs=set(), resolution=0.3)
        self.assertGreaterEqual(
            len(set(high_res.values())),
            len(set(low_res.values())),
        )


class TestToWeightedUndirected(unittest.TestCase):

    def test_collapse_parallel_edges(self):
        """Parallel directed edges should sum weights."""
        G = nx.MultiDiGraph()
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("A", "B", weight=2.0)
        G.add_edge("B", "A", weight=0.5)
        U = _to_weighted_undirected(G)
        self.assertAlmostEqual(U["A"]["B"]["weight"], 3.5)

    def test_preserves_all_nodes(self):
        G = nx.MultiDiGraph()
        G.add_node("orphan")
        G.add_edge("A", "B", weight=1.0)
        U = _to_weighted_undirected(G)
        self.assertIn("orphan", U.nodes())


# ═══════════════════════════════════════════════════════════════════════════
# 3. Micro-Clustering
# ═══════════════════════════════════════════════════════════════════════════

class TestMicroClustering(unittest.TestCase):

    def test_small_cluster_single_page(self):
        """Cluster < 4 nodes → all in micro_id=0."""
        G = _make_graph(("A", "B"), ("B", "C"))
        result = micro_cluster(G, {"A", "B", "C"})
        self.assertTrue(all(v == 0 for v in result.values()))

    def test_large_cluster_splits(self):
        """Dense 10-node cluster with two sub-communities should split."""
        G = nx.MultiDiGraph()
        # sub-community 1: n0-n4
        for i in range(5):
            for j in range(5):
                if i != j:
                    G.add_edge(f"n{i}", f"n{j}", weight=2.0)
        # sub-community 2: n5-n9
        for i in range(5, 10):
            for j in range(5, 10):
                if i != j:
                    G.add_edge(f"n{i}", f"n{j}", weight=2.0)
        # weak bridge
        G.add_edge("n0", "n5", weight=0.01)
        nodes = {f"n{i}" for i in range(10)}
        result = micro_cluster(G, nodes, resolution=1.5)
        # Should find at least 2 micro-clusters
        self.assertGreaterEqual(len(set(result.values())), 2)

    def test_micro_cluster_all(self):
        """micro_cluster_all processes every macro-cluster."""
        G = _build_clusterable_graph()
        macro_a = {f"a{i}": 0 for i in range(5)}
        macro_b = {f"b{i}": 1 for i in range(5)}
        macro_all = {**macro_a, **macro_b}
        result = micro_cluster_all(G, macro_all)
        self.assertIn(0, result)
        self.assertIn(1, result)
        self.assertEqual(len(result[0]), 5)
        self.assertEqual(len(result[1]), 5)

    def test_no_edges_cluster(self):
        """Cluster with no internal edges → all micro_id=0."""
        G = nx.MultiDiGraph()
        for i in range(6):
            G.add_node(f"n{i}")
        result = micro_cluster(G, {f"n{i}" for i in range(6)})
        self.assertTrue(all(v == 0 for v in result.values()))


# ═══════════════════════════════════════════════════════════════════════════
# 4. Dynamic Page Sizing
# ═══════════════════════════════════════════════════════════════════════════

class TestDynamicPageSizing(unittest.TestCase):

    def test_merge_tiny_clusters(self):
        """Clusters below merge_threshold should be absorbed."""
        G = nx.MultiDiGraph()
        # 1 tiny cluster (1 node), 1 normal cluster (5 nodes)
        for i in range(5):
            for j in range(5):
                if i != j:
                    G.add_edge(f"n{i}", f"n{j}", weight=1.0)
        G.add_edge("n0", "lone", weight=0.5)
        G.add_node("lone")

        micro = {f"n{i}": 0 for i in range(5)}
        micro["lone"] = 1  # tiny cluster

        result = apply_page_sizing(G, 0, micro, rules={
            "min_page_size": 3,
            "max_page_size": 25,
            "merge_threshold": 2,
        })

        # "lone" should have been merged into the main cluster
        clusters = set(result.values())
        self.assertEqual(len(clusters), 1, "Tiny cluster should be merged")

    def test_split_oversized_clusters(self):
        """Clusters > max_page_size should be split."""
        G = nx.MultiDiGraph()
        n = 30
        for i in range(n):
            for j in range(i + 1, min(i + 3, n)):
                G.add_edge(f"n{i}", f"n{j}", weight=1.0)

        micro = {f"n{i}": 0 for i in range(n)}
        result = apply_page_sizing(G, 0, micro, rules={
            "min_page_size": 2,
            "max_page_size": 10,
            "merge_threshold": 2,
        })
        max_cluster_size = max(
            sum(1 for v in result.values() if v == cid)
            for cid in set(result.values())
        )
        self.assertLessEqual(max_cluster_size, 15,
                             "Oversized cluster should be split (allowing some slack)")

    def test_normal_clusters_unchanged(self):
        """Clusters within bounds should not be modified."""
        G = _make_graph(("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"))
        micro = {"A": 0, "B": 0, "C": 0, "D": 0}
        result = apply_page_sizing(G, 0, micro)
        self.assertEqual(len(set(result.values())), 1)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Hub Re-Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestHubReintegration(unittest.TestCase):

    def test_majority_assignment(self):
        """Hub with >60% edges to one cluster → assigned there."""
        G = nx.MultiDiGraph()
        # hub connects to 4 nodes in cluster 0, 1 node in cluster 1
        for i in range(4):
            G.add_edge("hub", f"c0_{i}", weight=1.0)
        G.add_edge("hub", "c1_0", weight=1.0)

        macro = {f"c0_{i}": 0 for i in range(4)}
        macro["c1_0"] = 1

        result = reintegrate_hubs(G, {"hub"}, macro)
        self.assertEqual(result["hub"][0], 0)  # macro_id

    def test_dispersed_hub_gets_plurality(self):
        """Hub with edges evenly spread → assigned to cluster with most edges."""
        G = nx.MultiDiGraph()
        G.add_edge("hub", "c0_a", weight=1.0)
        G.add_edge("hub", "c0_b", weight=1.0)
        G.add_edge("hub", "c1_a", weight=1.0)
        G.add_edge("hub", "c2_a", weight=1.0)

        macro = {"c0_a": 0, "c0_b": 0, "c1_a": 1, "c2_a": 2}
        result = reintegrate_hubs(G, {"hub"}, macro)
        self.assertEqual(result["hub"][0], 0)  # cluster 0 has most edges (2)

    def test_disconnected_hub_gets_largest_cluster(self):
        """Hub with no edges → assigned to largest cluster."""
        G = nx.MultiDiGraph()
        G.add_node("hub")
        G.add_node("n0")
        G.add_node("n1")
        G.add_node("n2")
        macro = {"n0": 0, "n1": 0, "n2": 1}
        result = reintegrate_hubs(G, {"hub"}, macro)
        self.assertEqual(result["hub"][0], 0)  # cluster 0 is largest

    def test_incoming_edges_count(self):
        """Incoming edges to hub should also be counted."""
        G = nx.MultiDiGraph()
        for i in range(5):
            G.add_edge(f"c0_{i}", "hub", weight=1.0)
        G.add_edge("c1_0", "hub", weight=1.0)

        macro = {f"c0_{i}": 0 for i in range(5)}
        macro["c1_0"] = 1

        result = reintegrate_hubs(G, {"hub"}, macro)
        self.assertEqual(result["hub"][0], 0)  # macro_id

    def test_multiple_hubs(self):
        """Each hub gets independent assignment (plurality vote)."""
        G = nx.MultiDiGraph()
        for i in range(4):
            G.add_edge("hub1", f"c0_{i}", weight=1.0)
        G.add_edge("hub2", "c1_0", weight=1.0)
        G.add_edge("hub2", "c2_0", weight=1.0)

        macro = {f"c0_{i}": 0 for i in range(4)}
        macro["c1_0"] = 1
        macro["c2_0"] = 2

        result = reintegrate_hubs(G, {"hub1", "hub2"}, macro)
        self.assertEqual(result["hub1"][0], 0)
        # hub2 has tie (1 edge each to cluster 1 and 2) — gets whichever is most_common
        self.assertIn(result["hub2"][0], [1, 2])


# ═══════════════════════════════════════════════════════════════════════════
# 6. Persist Clusters to DB
# ═══════════════════════════════════════════════════════════════════════════

class TestPersistClusters(unittest.TestCase):

    def test_basic_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict(f"n{i}") for i in range(6)]
            db = _make_db(tmp, nodes=nodes)

            macro = {f"n{i}": 0 for i in range(3)}
            macro.update({f"n{i}": 1 for i in range(3, 6)})
            micro = {
                0: {f"n{i}": 0 for i in range(3)},
                1: {f"n{i}": 0 for i in range(3, 6)},
            }
            hub_assign = {}

            stats = persist_clusters(db, macro, micro, hub_assign)
            self.assertEqual(stats["nodes_clustered"], 6)
            self.assertEqual(stats["macro_clusters"], 2)

            # Verify DB state
            all_clusters = db.get_all_clusters()
            self.assertEqual(len(all_clusters), 2)

    def test_with_hubs(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict("n0"), _make_node_dict("hub1")]
            db = _make_db(tmp, nodes=nodes)

            macro = {"n0": 0}
            micro = {0: {"n0": 0}}
            hub_assign = {"hub1": (0, 0)}  # (macro_id, micro_id)

            persist_clusters(db, macro, micro, hub_assign)

            hub = db.get_node("hub1")
            self.assertEqual(hub["is_hub"], 1)
            self.assertEqual(hub["hub_assignment"], "0")
            self.assertEqual(hub["macro_cluster"], 0)
            self.assertEqual(hub["micro_cluster"], 0)

    def test_resets_stale_clusters(self):
        """persist_clusters clears pre-existing cluster assignments before writing."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create nodes with stale cluster assignments from a "previous run"
            nodes = [
                _make_node_dict("n0", macro_cluster=9, micro_cluster=9),
                _make_node_dict("n1", macro_cluster=9, micro_cluster=9),
                _make_node_dict("stale", macro_cluster=5, micro_cluster=5),
            ]
            db = _make_db(tmp, nodes=nodes)

            # Only assign n0 and n1 in the new run — stale is NOT assigned
            macro = {"n0": 0, "n1": 0}
            micro = {0: {"n0": 0, "n1": 0}}
            hub_assign = {}

            persist_clusters(db, macro, micro, hub_assign)

            # n0/n1 should have new assignments
            self.assertEqual(db.get_node("n0")["macro_cluster"], 0)
            self.assertEqual(db.get_node("n1")["macro_cluster"], 0)

            # stale node should have NULL clusters (reset)
            stale = db.get_node("stale")
            self.assertIsNone(stale["macro_cluster"])
            self.assertIsNone(stale["micro_cluster"])


# ═══════════════════════════════════════════════════════════════════════════
# 7. Full Pipeline — run_phase3
# ═══════════════════════════════════════════════════════════════════════════

class TestRunPhase3(unittest.TestCase):

    def test_end_to_end(self):
        """Full Phase 3 pipeline with two-community graph."""
        with tempfile.TemporaryDirectory() as tmp:
            # Build graph with two communities and a hub
            G = nx.MultiDiGraph()
            nodes_data = []

            # Community A
            for i in range(6):
                nid = f"a{i}"
                nodes_data.append(_make_node_dict(nid, rel_path=f"src/auth/{nid}.py"))
                for j in range(6):
                    if i != j:
                        G.add_edge(f"a{i}", f"a{j}", weight=1.0, relationship_type="calls")

            # Community B
            for i in range(6):
                nid = f"b{i}"
                nodes_data.append(_make_node_dict(nid, rel_path=f"src/data/{nid}.py"))
                for j in range(6):
                    if i != j:
                        G.add_edge(f"b{i}", f"b{j}", weight=1.0, relationship_type="calls")

            # Hub — connected to both
            nodes_data.append(_make_node_dict("hub_logger"))
            for i in range(6):
                G.add_edge(f"a{i}", "hub_logger", weight=0.5, relationship_type="calls")
                G.add_edge(f"b{i}", "hub_logger", weight=0.5, relationship_type="calls")

            # Thin bridge
            G.add_edge("a0", "b0", weight=0.01, relationship_type="calls")

            db = _make_db(tmp, nodes=nodes_data)

            # Phase 2 prerequisites
            apply_edge_weights(G)
            hubs = detect_hubs(G, z_threshold=2.0)

            results = run_phase3(db, G, hubs=hubs)

            self.assertGreaterEqual(results["macro"]["cluster_count"], 1)
            self.assertGreaterEqual(results["micro"]["total_pages"], 1)
            self.assertTrue(db.get_meta("phase3_completed"))

    def test_empty_graph(self):
        """Empty graph → no clusters."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(tmp)
            G = nx.MultiDiGraph()
            results = run_phase3(db, G, hubs=set())
            self.assertEqual(results["macro"]["cluster_count"], 0)
            self.assertEqual(results["micro"]["total_pages"], 0)

    def test_single_node_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(tmp, nodes=[_make_node_dict("only")])
            G = nx.MultiDiGraph()
            G.add_node("only")
            results = run_phase3(db, G, hubs=set())
            self.assertEqual(results["macro"]["nodes_assigned"], 1)

    def test_all_hubs(self):
        """If all nodes are hubs → no macro, just hub re-integration."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict(f"h{i}") for i in range(3)]
            db = _make_db(tmp, nodes=nodes)

            G = _make_graph(("h0", "h1"), ("h1", "h2"))
            hubs = {"h0", "h1", "h2"}

            results = run_phase3(db, G, hubs=hubs)
            self.assertEqual(results["macro"]["cluster_count"], 0)
            self.assertEqual(results["hubs"]["total"], 3)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):

    def test_disconnected_components(self):
        """Multiple disconnected components should become separate clusters."""
        G = nx.MultiDiGraph()
        for i in range(4):
            G.add_edge(f"a{i}", f"a{(i + 1) % 4}", weight=1.0)
        for i in range(4):
            G.add_edge(f"b{i}", f"b{(i + 1) % 4}", weight=1.0)

        assignments = macro_cluster(G, hubs=set(), resolution=1.0)
        a_clusters = {assignments[f"a{i}"] for i in range(4)}
        b_clusters = {assignments[f"b{i}"] for i in range(4)}
        self.assertEqual(len(a_clusters), 1)
        self.assertEqual(len(b_clusters), 1)
        self.assertNotEqual(a_clusters, b_clusters)

    def test_recursive_split_force_chunk(self):
        """Force-chunk when Louvain can't split (fully connected)."""
        G = nx.MultiDiGraph()
        for i in range(30):
            for j in range(30):
                if i != j:
                    G.add_edge(f"n{i}", f"n{j}", weight=1.0)
        chunks = _recursive_split(G, {f"n{i}" for i in range(30)}, max_size=10)
        # Should produce roughly 3 chunks
        self.assertTrue(all(len(c) <= 10 for c in chunks))
        total_nodes = sum(len(c) for c in chunks)
        self.assertEqual(total_nodes, 30)

    def test_deterministic_seed(self):
        """Same input → same clusters (seed=42)."""
        G = _build_clusterable_graph()
        r1 = macro_cluster(G, hubs=set(), resolution=1.0)
        r2 = macro_cluster(G, hubs=set(), resolution=1.0)
        self.assertEqual(r1, r2)

    def test_large_graph_smoke(self):
        """Smoke test: 500 nodes, 3 communities."""
        import time
        G = nx.MultiDiGraph()
        for c in range(3):
            base = c * 150
            for i in range(150):
                for j in range(i + 1, min(i + 5, 150)):
                    G.add_edge(f"n{base + i}", f"n{base + j}", weight=1.0)
            # thin bridge to next community
            if c < 2:
                G.add_edge(f"n{base + 149}", f"n{base + 150}", weight=0.01)

        t0 = time.time()
        assignments = macro_cluster(G, hubs=set())
        elapsed = time.time() - t0

        self.assertGreaterEqual(len(set(assignments.values())), 2)
        self.assertLess(elapsed, 5.0, f"Clustering 500 nodes took {elapsed:.1f}s")


if __name__ == "__main__":
    unittest.main()
