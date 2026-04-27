"""
Tests for page-level centroid detection.

Validates that _detect_page_centroids() correctly identifies the most
significant nodes in a page community using degree + architectural
type priority scoring.
"""

import math

import networkx as nx
import pytest

from plugin_implementation.graph_clustering import (
    CENTROID_ARCHITECTURAL_MIN_PRIORITY,
    MAX_CENTROIDS,
    MIN_CENTROIDS,
    _detect_page_centroids,
)
from plugin_implementation.constants import SYMBOL_TYPE_PRIORITY


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_graph(nodes_with_attrs, edges=None):
    """Build a MultiDiGraph for centroid tests.

    nodes_with_attrs: list of (nid, symbol_type, symbol_name)
    edges: optional list of (src, dst)
    """
    G = nx.MultiDiGraph()
    for nid, stype, sname in nodes_with_attrs:
        G.add_node(nid, symbol_type=stype, symbol_name=sname,
                   rel_path=f"src/{sname}.py", file_name=f"src/{sname}.py")
    for src, dst in (edges or []):
        G.add_edge(src, dst, weight=1.0, edge_class="calls")
    return G


# ═══════════════════════════════════════════════════════════════════════════
# Test: Empty / trivial inputs
# ═══════════════════════════════════════════════════════════════════════════

class TestCentroidEdgeCases:

    def test_empty_page(self):
        G = nx.MultiDiGraph()
        assert _detect_page_centroids(G, []) == []

    def test_single_node(self):
        G = _make_graph([("a", "class", "MyClass")])
        centroids = _detect_page_centroids(G, ["a"])
        assert len(centroids) == 1
        assert centroids[0]["node_id"] == "a"
        assert centroids[0]["score"] == 1.0

    def test_two_nodes_same_type(self):
        """Higher-degree node should rank first."""
        G = _make_graph([
            ("a", "function", "funcA"),
            ("b", "function", "funcB"),
        ], edges=[("a", "b")])
        centroids = _detect_page_centroids(G, ["a", "b"])
        assert len(centroids) >= 1
        # 'a' has out-degree 1 + in-degree 0 = 1; 'b' has in-degree 1
        # Both have degree 1, so either can be first


# ═══════════════════════════════════════════════════════════════════════════
# Test: Architectural priority boost
# ═══════════════════════════════════════════════════════════════════════════

class TestCentroidArchitecturalPriority:

    def test_class_beats_method(self):
        """A class should rank above a method even with lower degree."""
        G = _make_graph([
            ("cls", "class", "MyClass"),
            ("m1", "method", "method1"),
            ("m2", "method", "method2"),
            ("m3", "method", "method3"),
        ], edges=[
            # Methods have more connections
            ("m1", "m2"), ("m2", "m3"), ("m3", "m1"),
            ("cls", "m1"),
        ])
        centroids = _detect_page_centroids(G, ["cls", "m1", "m2", "m3"])
        # Class should be the top centroid (method has priority < 5,
        # so only cls is architectural)
        assert centroids[0]["node_id"] == "cls"
        assert centroids[0]["type"] == "class"

    def test_function_beats_variable(self):
        """A standalone function should rank above a variable."""
        G = _make_graph([
            ("fn", "function", "process_data"),
            ("v1", "variable", "counter"),
        ])
        centroids = _detect_page_centroids(G, ["fn", "v1"])
        assert centroids[0]["node_id"] == "fn"

    def test_interface_ranked_high(self):
        """Interfaces have top-tier priority."""
        G = _make_graph([
            ("iface", "interface", "IService"),
            ("fn", "function", "helper"),
            ("const", "constant", "MAX_SIZE"),
        ])
        centroids = _detect_page_centroids(G, ["iface", "fn", "const"])
        assert centroids[0]["node_id"] == "iface"

    def test_only_architectural_in_centroids(self):
        """When architectural nodes exist, non-architectural nodes are excluded."""
        G = _make_graph([
            ("cls", "class", "Handler"),
            ("m1", "method", "handle"),
            ("m2", "method", "process"),
            ("prop", "property", "value"),
        ])
        centroids = _detect_page_centroids(G, ["cls", "m1", "m2", "prop"])
        # Only cls has priority >= CENTROID_ARCHITECTURAL_MIN_PRIORITY
        centroid_ids = {c["node_id"] for c in centroids}
        assert "cls" in centroid_ids
        # methods and properties should NOT appear as centroids
        assert "m1" not in centroid_ids
        assert "prop" not in centroid_ids

    def test_fallback_when_no_architectural(self):
        """When no architectural nodes exist, all nodes are candidates."""
        G = _make_graph([
            ("m1", "method", "methodA"),
            ("m2", "method", "methodB"),
        ], edges=[("m1", "m2")])
        centroids = _detect_page_centroids(G, ["m1", "m2"])
        # Should still return centroids (fallback to all nodes)
        assert len(centroids) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Test: Degree scoring
# ═══════════════════════════════════════════════════════════════════════════

class TestCentroidDegreeScoring:

    def test_high_degree_class_wins(self):
        """Among multiple classes, the one with more connections wins."""
        G = _make_graph([
            ("hub_cls", "class", "HubClass"),
            ("leaf_cls", "class", "LeafClass"),
            ("fn1", "function", "func1"),
            ("fn2", "function", "func2"),
            ("fn3", "function", "func3"),
        ], edges=[
            ("hub_cls", "fn1"), ("hub_cls", "fn2"), ("hub_cls", "fn3"),
            ("fn1", "hub_cls"), ("fn2", "hub_cls"),
            ("leaf_cls", "fn1"),
        ])
        centroids = _detect_page_centroids(
            G, ["hub_cls", "leaf_cls", "fn1", "fn2", "fn3"]
        )
        assert centroids[0]["node_id"] == "hub_cls"

    def test_isolated_node_low_score(self):
        """A node with zero edges should have lower score."""
        G = _make_graph([
            ("connected", "class", "ConnectedClass"),
            ("isolated", "class", "IsolatedClass"),
            ("fn", "function", "helper"),
        ], edges=[
            ("connected", "fn"), ("fn", "connected"),
        ])
        centroids = _detect_page_centroids(
            G, ["connected", "isolated", "fn"]
        )
        # connected class has degree 2, isolated has 0
        assert centroids[0]["node_id"] == "connected"


# ═══════════════════════════════════════════════════════════════════════════
# Test: top_k parameter
# ═══════════════════════════════════════════════════════════════════════════

class TestCentroidTopK:

    def test_default_k_via_sqrt(self):
        """Default k = sqrt(N), clamped to [MIN, MAX]."""
        nodes = [(f"n{i}", "class", f"Class{i}") for i in range(25)]
        G = _make_graph(nodes)
        nids = [f"n{i}" for i in range(25)]
        centroids = _detect_page_centroids(G, nids)
        expected_k = min(MAX_CENTROIDS, max(MIN_CENTROIDS, round(math.sqrt(25))))
        assert len(centroids) == expected_k

    def test_explicit_top_k(self):
        """Explicit top_k limits the result."""
        nodes = [(f"n{i}", "function", f"func{i}") for i in range(20)]
        G = _make_graph(nodes)
        nids = [f"n{i}" for i in range(20)]
        centroids = _detect_page_centroids(G, nids, top_k=3)
        assert len(centroids) == 3

    def test_top_k_larger_than_pool(self):
        """top_k > available nodes returns all available."""
        G = _make_graph([("a", "class", "A"), ("b", "class", "B")])
        centroids = _detect_page_centroids(G, ["a", "b"], top_k=100)
        assert len(centroids) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Test: Score normalization
# ═══════════════════════════════════════════════════════════════════════════

class TestCentroidScores:

    def test_top_centroid_score_is_one(self):
        """The best centroid should have score 1.0."""
        G = _make_graph([
            ("a", "class", "ClassA"),
            ("b", "function", "funcB"),
        ], edges=[("a", "b")])
        centroids = _detect_page_centroids(G, ["a", "b"])
        assert centroids[0]["score"] == 1.0

    def test_scores_descending(self):
        """Scores should be in descending order."""
        nodes = [(f"n{i}", "class", f"Class{i}") for i in range(10)]
        edges = [(f"n0", f"n{i}") for i in range(1, 10)]
        G = _make_graph(nodes, edges)
        nids = [f"n{i}" for i in range(10)]
        centroids = _detect_page_centroids(G, nids)
        scores = [c["score"] for c in centroids]
        assert scores == sorted(scores, reverse=True)

    def test_centroid_has_required_fields(self):
        """Each centroid dict must have node_id, name, type, score."""
        G = _make_graph([("a", "class", "MyClass")])
        centroids = _detect_page_centroids(G, ["a"])
        c = centroids[0]
        assert "node_id" in c
        assert "name" in c
        assert "type" in c
        assert "score" in c


# ═══════════════════════════════════════════════════════════════════════════
# Test: Integration with hierarchical_leiden_cluster
# ═══════════════════════════════════════════════════════════════════════════

class TestCentroidInLeiden:

    def test_leiden_result_has_centroids(self):
        """hierarchical_leiden_cluster should produce centroids per page."""
        try:
            import igraph
            import leidenalg
        except ImportError:
            pytest.skip("leidenalg/igraph not installed")

        from plugin_implementation.graph_clustering import (
            hierarchical_leiden_cluster,
        )

        # Build a small connected graph
        G = nx.MultiDiGraph()
        for i in range(20):
            stype = "class" if i < 5 else "function" if i < 15 else "method"
            G.add_node(f"n{i}", symbol_type=stype, symbol_name=f"Sym{i}",
                       rel_path=f"src/mod{i % 4}.py")
        # Connect in a ring + random cross-edges
        for i in range(20):
            G.add_edge(f"n{i}", f"n{(i + 1) % 20}", weight=1.0)
        for i in range(0, 20, 3):
            G.add_edge(f"n{i}", f"n{(i + 7) % 20}", weight=0.5)

        result = hierarchical_leiden_cluster(G, set())

        # Every section should have per-page centroids
        for sec_id, sec_data in result["sections"].items():
            assert "centroids" in sec_data, f"Section {sec_id} missing centroids"
            for pg_id, page_centroids in sec_data["centroids"].items():
                # Each page should have at least 1 centroid
                assert len(page_centroids) >= 1
                for c in page_centroids:
                    assert "node_id" in c
                    assert "name" in c
                    assert "type" in c
                    assert "score" in c
                    assert 0.0 <= c["score"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Test: Full-graph Leiden (no projection)
# ═══════════════════════════════════════════════════════════════════════════

class TestFullGraphLeiden:
    """Verify Leiden on the full graph produces connected communities."""

    def test_full_graph_fewer_communities_than_projected(self):
        """Running Leiden on full graph should produce fewer raw communities
        than running on a projected (fragmented) graph."""
        try:
            import igraph
            import leidenalg
        except ImportError:
            pytest.skip("leidenalg/igraph not installed")

        from plugin_implementation.graph_clustering import (
            architectural_projection,
            hierarchical_leiden_cluster,
        )

        # Build a graph where methods connect two classes
        G = nx.MultiDiGraph()
        for i in range(10):
            G.add_node(f"cls{i}", symbol_type="class", symbol_name=f"Class{i}",
                       rel_path=f"src/mod{i}.py")
            for j in range(3):
                meth = f"cls{i}.m{j}"
                G.add_node(meth, symbol_type="method", symbol_name=f"m{j}",
                           rel_path=f"src/mod{i}.py")
                G.add_edge(f"cls{i}", meth, weight=1.0, edge_class="contains")
                # Cross-class method calls — these link classes through methods
                target_cls = (i + 1) % 10
                G.add_edge(meth, f"cls{target_cls}.m0",
                           weight=1.0, edge_class="calls")

        # Full graph: should be well-connected → few communities
        full_result = hierarchical_leiden_cluster(G, set())
        n_full = full_result["algorithm_metadata"]["sections"]

        # Projected graph: methods removed, many cross-class edges lost
        P = architectural_projection(G)
        proj_result = hierarchical_leiden_cluster(P, set())
        n_proj = proj_result["algorithm_metadata"]["sections"]

        # Full graph should produce ≤ projected community count
        assert n_full <= n_proj, (
            f"Full graph ({n_full}) should have ≤ communities than "
            f"projected ({n_proj})"
        )

    def test_methods_cluster_with_parent_class(self):
        """Methods should naturally cluster with their parent class."""
        try:
            import igraph
            import leidenalg
        except ImportError:
            pytest.skip("leidenalg/igraph not installed")

        from plugin_implementation.graph_clustering import (
            hierarchical_leiden_cluster,
        )

        G = nx.MultiDiGraph()
        # Two isolated class clusters with strong internal connectivity
        for cls_id in range(2):
            cls = f"cls{cls_id}"
            G.add_node(cls, symbol_type="class", symbol_name=f"Class{cls_id}",
                       rel_path=f"src/pkg{cls_id}/main.py")
            for m in range(5):
                meth = f"{cls}.m{m}"
                G.add_node(meth, symbol_type="method", symbol_name=f"m{m}",
                           rel_path=f"src/pkg{cls_id}/main.py")
                G.add_edge(cls, meth, weight=2.0, edge_class="contains")
                G.add_edge(meth, cls, weight=1.0, edge_class="calls")
                # Methods call each other within same class
                if m > 0:
                    G.add_edge(meth, f"{cls}.m{m-1}", weight=1.0, edge_class="calls")

        result = hierarchical_leiden_cluster(G, set())

        # Check that methods are in the same section as their class
        macro = result["macro_assignments"]
        for cls_id in range(2):
            cls = f"cls{cls_id}"
            cls_section = macro.get(cls)
            for m in range(5):
                meth = f"{cls}.m{m}"
                assert macro.get(meth) == cls_section, (
                    f"{meth} should be in same section as {cls}"
                )
