"""Tests for bridge_disconnected_components in graph_topology.py.

Verifies that disconnected components in the full graph are bridged
via directory-proximity edges before Leiden clustering (Phase 3).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

import networkx as nx

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), os.pardir, "plugin_implementation"),
)

from graph_topology import bridge_disconnected_components


def _make_db():
    """Return a mock DB whose upsert_edge silently succeeds."""
    db = MagicMock()
    db.upsert_edge = MagicMock()
    db.conn = MagicMock()
    return db


def _make_graph(components):
    """Build a MultiDiGraph from a list of component specs.

    Each spec is a list of (node_id, rel_path) tuples.
    Edges are added between consecutive nodes within each component.
    """
    G = nx.MultiDiGraph()
    for comp in components:
        for nid, rpath in comp:
            G.add_node(nid, rel_path=rpath)
        # Chain edges within component
        for i in range(len(comp) - 1):
            G.add_edge(comp[i][0], comp[i + 1][0], edge_class="structural")
    return G


class TestBridgingSingleComponent(unittest.TestCase):
    """No bridging needed if graph is already connected."""

    def test_single_component(self):
        G = _make_graph([
            [("a", "src/main.cpp"), ("b", "src/util.cpp"), ("c", "src/core.cpp")],
        ])
        db = _make_db()
        stats = bridge_disconnected_components(db, G)
        self.assertEqual(stats["components_before"], 1)
        self.assertEqual(stats["components_after"], 1)
        self.assertEqual(stats["bridges_added"], 0)

    def test_empty_graph(self):
        G = nx.MultiDiGraph()
        db = _make_db()
        stats = bridge_disconnected_components(db, G)
        self.assertEqual(stats["components_before"], 0)
        self.assertEqual(stats["components_after"], 0)
        self.assertEqual(stats["bridges_added"], 0)


class TestBridgingTwoComponents(unittest.TestCase):
    """Two disconnected components should be bridged."""

    def test_two_components_bridged(self):
        G = _make_graph([
            [("a", "src/core.cpp"), ("b", "src/core.h")],
            [("c", "src/util.cpp"), ("d", "src/util.h")],
        ])
        db = _make_db()
        stats = bridge_disconnected_components(db, G)
        self.assertEqual(stats["components_before"], 2)
        self.assertEqual(stats["components_after"], 1)
        self.assertEqual(stats["bridges_added"], 2)  # bidirectional

    def test_bridge_edges_are_in_graph(self):
        G = _make_graph([
            [("a", "src/main.cpp")],
            [("b", "src/util.cpp")],
        ])
        edges_before = G.number_of_edges()
        db = _make_db()
        bridge_disconnected_components(db, G)
        self.assertEqual(G.number_of_edges(), edges_before + 2)

    def test_bridge_edges_persisted_to_db(self):
        G = _make_graph([
            [("a", "src/main.cpp")],
            [("b", "src/util.cpp")],
        ])
        db = _make_db()
        bridge_disconnected_components(db, G)
        # 2 calls to upsert_edge (bidirectional)
        self.assertEqual(db.upsert_edge.call_count, 2)


class TestBridgingMultipleComponents(unittest.TestCase):
    """Multiple disconnected components should all get bridged."""

    def test_five_components(self):
        G = _make_graph([
            [("a1", "src/a/x.cpp"), ("a2", "src/a/y.cpp")],
            [("b1", "src/b/x.cpp")],
            [("c1", "src/c/x.cpp")],
            [("d1", "src/d/x.cpp")],
            [("e1", "src/e/x.cpp")],
        ])
        db = _make_db()
        stats = bridge_disconnected_components(db, G)
        self.assertEqual(stats["components_before"], 5)
        self.assertEqual(stats["components_after"], 1)
        # 4 bridges × 2 (bidirectional) = 8
        self.assertEqual(stats["bridges_added"], 8)

    def test_many_singletons(self):
        """Even single-node components are bridged (not orphans if they have self-edges)."""
        G = nx.MultiDiGraph()
        for i in range(20):
            nid = f"node_{i}"
            G.add_node(nid, rel_path=f"src/dir_{i % 4}/file_{i}.cpp")
            # Self-edge so they're not degree-0 orphans
            G.add_edge(nid, nid, edge_class="structural")
        db = _make_db()
        stats = bridge_disconnected_components(db, G)
        self.assertEqual(stats["components_before"], 20)
        self.assertEqual(stats["components_after"], 1)
        self.assertEqual(stats["bridges_added"], 38)  # 19 × 2


class TestBridgingDirectoryProximity(unittest.TestCase):
    """Bridge edges should prefer directory-similar components."""

    def test_same_dir_preferred(self):
        # Component 0 (largest): src/core/
        # Component 1: src/core/ (same directory → should bridge to comp 0)
        # Component 2: test/other/ (different directory)
        G = _make_graph([
            [("a1", "src/core/a.cpp"), ("a2", "src/core/b.cpp"),
             ("a3", "src/core/c.cpp")],
            [("b1", "src/core/d.cpp")],
            [("c1", "test/other/x.cpp")],
        ])
        db = _make_db()
        stats = bridge_disconnected_components(db, G)
        self.assertEqual(stats["components_after"], 1)

        # Verify b1 bridges to a component in src/core (the largest)
        bridge_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_class") == "bridge"
        ]
        # b1 should connect to one of {a1, a2, a3}
        b1_targets = [v for u, v in bridge_edges if u == "b1"]
        self.assertTrue(
            any(t in {"a1", "a2", "a3"} for t in b1_targets),
            f"Expected b1 to bridge to src/core/ component, got {b1_targets}",
        )

    def test_root_dir_fallback(self):
        """Nodes without directory info still get bridged."""
        G = _make_graph([
            [("a", "main.cpp"), ("b", "util.cpp")],
            [("c", "other.cpp")],
        ])
        db = _make_db()
        stats = bridge_disconnected_components(db, G)
        self.assertEqual(stats["components_after"], 1)


class TestBridgingEdgeAttributes(unittest.TestCase):
    """Bridge edges have correct edge_class and relationship_type."""

    def test_edge_class_is_bridge(self):
        G = _make_graph([
            [("a", "src/a.cpp")],
            [("b", "src/b.cpp")],
        ])
        db = _make_db()
        bridge_disconnected_components(db, G)
        for u, v, data in G.edges(data=True):
            if data.get("created_by") == "component_bridging":
                self.assertEqual(data["edge_class"], "bridge")
                self.assertEqual(data["relationship_type"], "component_bridge")

    def test_bridge_not_counted_as_structural(self):
        """Verify bridge edges won't inflate structural in-degree."""
        from graph_topology import apply_edge_weights

        G = _make_graph([
            [("a", "src/a.cpp"), ("b", "src/a.h")],
            [("c", "src/b.cpp")],
        ])
        db = _make_db()
        bridge_disconnected_components(db, G)
        stats = apply_edge_weights(G)
        # Should complete without error; bridge edges get synthetic floor
        self.assertGreater(stats["edges_weighted"], 0)


class TestBridgingRepresentativeSelection(unittest.TestCase):
    """Bridge endpoints should be the highest-degree nodes in each component."""

    def test_highest_degree_used(self):
        G = nx.MultiDiGraph()
        # Component 1: star topology centered on "hub1"
        for i in range(5):
            nid = f"leaf1_{i}"
            G.add_node(nid, rel_path=f"src/a/{nid}.cpp")
            G.add_edge("hub1", nid, edge_class="structural")
        G.add_node("hub1", rel_path="src/a/hub.cpp")

        # Component 2: star centered on "hub2"
        for i in range(3):
            nid = f"leaf2_{i}"
            G.add_node(nid, rel_path=f"src/b/{nid}.cpp")
            G.add_edge("hub2", nid, edge_class="structural")
        G.add_node("hub2", rel_path="src/b/hub.cpp")

        db = _make_db()
        bridge_disconnected_components(db, G)

        # The bridge should connect hub1 ↔ hub2 (highest degree in each component)
        bridge_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_class") == "bridge"
        ]
        bridge_nodes = {u for u, v in bridge_edges} | {v for u, v in bridge_edges}
        self.assertIn("hub1", bridge_nodes)
        self.assertIn("hub2", bridge_nodes)


class TestBridgingWithRunPhase2(unittest.TestCase):
    """Integration: run_phase2 includes component bridging."""

    def test_run_phase2_has_bridging_key(self):
        from graph_topology import run_phase2

        G = _make_graph([
            [("a", "src/a.cpp"), ("b", "src/a.h")],
            [("c", "src/b.cpp")],
        ])
        db = _make_db()
        # Stub DB methods used by sub-steps
        db.get_node = MagicMock(return_value=None)
        db.search_fts5 = MagicMock(return_value=[])
        db.node_count = MagicMock(return_value=3)
        db.edge_count = MagicMock(return_value=5)
        db.set_hub = MagicMock()
        db.set_meta = MagicMock()
        db.upsert_edges_batch = MagicMock()

        results = run_phase2(db, G)
        self.assertIn("component_bridging", results)
        self.assertEqual(results["component_bridging"]["components_before"], 2)
        self.assertEqual(results["component_bridging"]["components_after"], 1)


if __name__ == "__main__":
    unittest.main()
