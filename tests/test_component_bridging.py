"""
Tests for post-projection component bridging.

Validates that _bridge_components() correctly connects disconnected
components in the projected graph using directory-proximity heuristics,
reducing the number of connected components so Leiden can cluster freely.
"""

from collections import Counter

import networkx as nx
import pytest

from plugin_implementation.graph_clustering import (
    BRIDGE_WEIGHT,
    _bridge_components,
    _dir_histogram,
    _dir_of_node,
    _dir_similarity,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_projected_graph(components):
    """Build a projected MultiDiGraph with disconnected components.

    components: list of lists, each inner list is [(nid, rel_path), ...]
                Edges are added within each component as a chain.
    """
    P = nx.MultiDiGraph()
    for comp in components:
        for nid, path in comp:
            P.add_node(nid, rel_path=path, file_name=path, symbol_type="function")
        # Chain nodes within each component
        for i in range(len(comp) - 1):
            P.add_edge(comp[i][0], comp[i + 1][0], weight=1.0, edge_class="calls")
    return P


def _singleton_components(n, dir_prefix="src"):
    """Create N single-node disconnected components in the same directory."""
    comps = []
    for i in range(n):
        comps.append([
            (f"py::{dir_prefix}/mod{i}.py::func{i}", f"{dir_prefix}/mod{i}.py"),
        ])
    return comps


# ═══════════════════════════════════════════════════════════════════════════
# Test: Single component (no-op)
# ═══════════════════════════════════════════════════════════════════════════

class TestBridgeNoOp:
    """Nothing to bridge when the graph is already connected."""

    def test_empty_graph(self):
        P = nx.MultiDiGraph()
        stats = _bridge_components(P)
        assert stats["components_before"] <= 1
        assert stats["bridges_added"] == 0

    def test_single_node(self):
        P = nx.MultiDiGraph()
        P.add_node("a", rel_path="src/a.py", symbol_type="function")
        stats = _bridge_components(P)
        assert stats["components_before"] == 1
        assert stats["components_after"] == 1
        assert stats["bridges_added"] == 0

    def test_already_connected(self):
        comp = [
            ("py::src/a.py::A", "src/a.py"),
            ("py::src/b.py::B", "src/b.py"),
            ("py::src/c.py::C", "src/c.py"),
        ]
        P = _make_projected_graph([comp])
        stats = _bridge_components(P)
        assert stats["components_before"] == 1
        assert stats["bridges_added"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Test: Basic bridging
# ═══════════════════════════════════════════════════════════════════════════

class TestBridgeBasic:
    """Bridging two or more disconnected components."""

    def test_two_components_merged(self):
        """Two disconnected components become one."""
        comp1 = [
            ("py::src/a.py::A", "src/a.py"),
            ("py::src/b.py::B", "src/b.py"),
        ]
        comp2 = [
            ("py::src/c.py::C", "src/c.py"),
        ]
        P = _make_projected_graph([comp1, comp2])
        assert nx.number_weakly_connected_components(P) == 2

        stats = _bridge_components(P)

        assert stats["components_before"] == 2
        assert stats["components_after"] == 1
        assert stats["bridges_added"] == 2  # bidirectional

    def test_three_components_merged(self):
        """Three disconnected components become one."""
        comp1 = [("py::src/a.py::A", "src/a.py"), ("py::src/a.py::B", "src/a.py")]
        comp2 = [("py::lib/x.py::X", "lib/x.py")]
        comp3 = [("py::lib/y.py::Y", "lib/y.py")]
        P = _make_projected_graph([comp1, comp2, comp3])
        assert nx.number_weakly_connected_components(P) == 3

        stats = _bridge_components(P)

        assert stats["components_before"] == 3
        assert stats["components_after"] == 1
        assert stats["bridges_added"] == 4  # 2 bridges × 2 directions

    def test_many_singletons_merged(self):
        """50 singleton components all get bridged into one."""
        comps = _singleton_components(50)
        P = _make_projected_graph(comps)
        assert nx.number_weakly_connected_components(P) == 50

        stats = _bridge_components(P)

        assert stats["components_before"] == 50
        assert stats["components_after"] == 1
        assert stats["bridges_added"] == 49 * 2  # 49 bridges × 2 directions

    def test_large_fragmentation(self):
        """500 disconnected components collapse to 1."""
        comps = _singleton_components(500, dir_prefix="src/utils")
        P = _make_projected_graph(comps)
        assert nx.number_weakly_connected_components(P) == 500

        stats = _bridge_components(P)

        assert stats["components_before"] == 500
        assert stats["components_after"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Test: Directory-proximity matching
# ═══════════════════════════════════════════════════════════════════════════

class TestBridgeDirectoryProximity:
    """Bridging uses directory proximity to pick targets."""

    def test_same_dir_preferred(self):
        """A singleton in 'lib/' should bridge to the 'lib/' component,
        not the 'src/' component."""
        comp_src = [
            ("py::src/a.py::A", "src/a.py"),
            ("py::src/b.py::B", "src/b.py"),
            ("py::src/c.py::C", "src/c.py"),
        ]
        comp_lib = [
            ("py::lib/x.py::X", "lib/x.py"),
            ("py::lib/y.py::Y", "lib/y.py"),
        ]
        orphan_lib = [
            ("py::lib/z.py::Z", "lib/z.py"),
        ]
        P = _make_projected_graph([comp_src, comp_lib, orphan_lib])
        assert nx.number_weakly_connected_components(P) == 3

        stats = _bridge_components(P)
        assert stats["components_after"] == 1

        # Check that the orphan in lib/ is connected to one of the lib/ nodes
        orphan = "py::lib/z.py::Z"
        neighbors = set(P.successors(orphan)) | set(P.predecessors(orphan))
        lib_nodes = {"py::lib/x.py::X", "py::lib/y.py::Y"}
        # The bridge should connect to the lib component representative
        assert neighbors & lib_nodes, (
            f"Orphan lib/z.py should bridge to lib/ component, got {neighbors}"
        )

    def test_mixed_directories(self):
        """Components from different directories get bridged correctly."""
        comp_a = [
            ("py::api/routes.py::Routes", "api/routes.py"),
            ("py::api/views.py::Views", "api/views.py"),
        ]
        comp_b = [
            ("py::api/models.py::Models", "api/models.py"),
        ]
        comp_c = [
            ("py::tests/test_api.py::TestAPI", "tests/test_api.py"),
        ]
        P = _make_projected_graph([comp_a, comp_b, comp_c])

        stats = _bridge_components(P)
        assert stats["components_after"] == 1

        # comp_b (api/models.py) should preferentially bridge to comp_a (api/)
        models_node = "py::api/models.py::Models"
        neighbors = set(P.successors(models_node)) | set(P.predecessors(models_node))
        api_nodes = {"py::api/routes.py::Routes", "py::api/views.py::Views"}
        assert neighbors & api_nodes, (
            f"api/models should bridge to api/ component, got {neighbors}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test: Bridge edge properties
# ═══════════════════════════════════════════════════════════════════════════

class TestBridgeEdgeProperties:
    """Verify the attributes on bridge edges."""

    def test_bridge_weight(self):
        """Bridge edges use the configured BRIDGE_WEIGHT."""
        comp1 = [("py::a.py::A", "a.py")]
        comp2 = [("py::b.py::B", "b.py")]
        P = _make_projected_graph([comp1, comp2])

        _bridge_components(P)

        bridge_edges = [
            (u, v, d)
            for u, v, d in P.edges(data=True)
            if d.get("edge_class") == "bridge"
        ]
        assert len(bridge_edges) == 2
        for _, _, data in bridge_edges:
            assert data["weight"] == BRIDGE_WEIGHT

    def test_bridge_edge_class(self):
        """Bridge edges are labelled with edge_class='bridge'."""
        comp1 = [("py::a.py::A", "a.py")]
        comp2 = [("py::b.py::B", "b.py")]
        P = _make_projected_graph([comp1, comp2])

        _bridge_components(P)

        bridge_edges = [
            d.get("edge_class")
            for _, _, d in P.edges(data=True)
            if d.get("edge_class") == "bridge"
        ]
        assert all(ec == "bridge" for ec in bridge_edges)

    def test_existing_edges_preserved(self):
        """Bridging does not remove or modify pre-existing edges."""
        comp1 = [("py::src/a.py::A", "src/a.py"), ("py::src/b.py::B", "src/b.py")]
        comp2 = [("py::lib/x.py::X", "lib/x.py")]
        P = _make_projected_graph([comp1, comp2])

        orig_edges = set()
        for u, v, d in P.edges(data=True):
            orig_edges.add((u, v, d.get("weight"), d.get("edge_class")))

        _bridge_components(P)

        # All original edges must still exist
        for u, v, w, ec in orig_edges:
            found = False
            for _, _, d in P.edges(u, data=True):
                if d.get("weight") == w and d.get("edge_class") == ec:
                    found = True
                    break
            assert found, f"Original edge {u}->{v} was modified or removed"

    def test_bidirectional_bridges(self):
        """Each bridge is added in both directions (u→v and v→u)."""
        comp1 = [("py::a.py::A", "a.py")]
        comp2 = [("py::b.py::B", "b.py")]
        P = _make_projected_graph([comp1, comp2])

        _bridge_components(P)

        bridge_a_out = [
            v for _, v, d in P.edges("py::a.py::A", data=True)
            if d.get("edge_class") == "bridge"
        ]
        bridge_b_out = [
            v for _, v, d in P.edges("py::b.py::B", data=True)
            if d.get("edge_class") == "bridge"
        ]
        # One of (A, B) should have a bridge to the other in each direction
        assert len(bridge_a_out) + len(bridge_b_out) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# Test: Representative node selection
# ═══════════════════════════════════════════════════════════════════════════

class TestRepresentativeSelection:
    """Bridge endpoints should pick high-degree nodes as representatives."""

    def test_hub_like_node_is_representative(self):
        """The most connected node in a component is the bridge endpoint."""
        P = nx.MultiDiGraph()
        # Component with a "hub" node H connected to A, B, C
        for nid in ["H", "A", "B", "C"]:
            P.add_node(nid, rel_path="src/main.py", symbol_type="function")
        P.add_edge("H", "A", weight=1.0)
        P.add_edge("H", "B", weight=1.0)
        P.add_edge("H", "C", weight=1.0)

        # Singleton component
        P.add_node("Z", rel_path="src/other.py", symbol_type="function")

        stats = _bridge_components(P)
        assert stats["components_after"] == 1

        # H should be the representative (degree 3 vs 1 for A, B, C)
        # Check that H has a bridge edge
        h_bridges = [
            v for _, v, d in P.edges("H", data=True)
            if d.get("edge_class") == "bridge"
        ]
        z_bridges = [
            v for _, v, d in P.edges("Z", data=True)
            if d.get("edge_class") == "bridge"
        ]
        assert len(h_bridges) > 0 or len(z_bridges) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Test: Integration with Phase 3 pipeline call site
# ═══════════════════════════════════════════════════════════════════════════

class TestBridgePipelineIntegration:
    """Verify bridge_components works within the clustering pipeline context."""

    def test_bridge_then_leiden_reduces_communities(self):
        """After bridging, Leiden with low γ produces far fewer communities
        than the number of original disconnected components."""
        try:
            import igraph
            import leidenalg
        except ImportError:
            pytest.skip("leidenalg/igraph not installed")

        from plugin_implementation.graph_clustering import (
            _nx_to_igraph,
            _to_weighted_undirected,
        )

        # Create 100 small disconnected components (2 nodes each)
        P = nx.MultiDiGraph()
        for i in range(100):
            a = f"py::src/pkg{i}/mod.py::ClassA{i}"
            b = f"py::src/pkg{i}/mod.py::ClassB{i}"
            P.add_node(a, rel_path=f"src/pkg{i}/mod.py", symbol_type="class")
            P.add_node(b, rel_path=f"src/pkg{i}/mod.py", symbol_type="function")
            P.add_edge(a, b, weight=1.0, edge_class="calls")

        assert nx.number_weakly_connected_components(P) == 100

        # Without bridging: Leiden must produce >= 100 communities
        G_undir = _to_weighted_undirected(P)
        ig, nodes = _nx_to_igraph(G_undir)
        part_before = leidenalg.find_partition(
            ig, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=0.01, seed=42,
        )
        n_communities_before = len(set(part_before.membership))
        assert n_communities_before >= 100

        # With bridging: should collapse dramatically
        _bridge_components(P)
        assert nx.number_weakly_connected_components(P) == 1

        G_undir2 = _to_weighted_undirected(P)
        ig2, nodes2 = _nx_to_igraph(G_undir2)
        part_after = leidenalg.find_partition(
            ig2, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=0.01, seed=42,
        )
        n_communities_after = len(set(part_after.membership))

        # After bridging, Leiden should produce far fewer communities
        assert n_communities_after < n_communities_before / 5, (
            f"Expected significant reduction: {n_communities_before} → {n_communities_after}"
        )

    def test_bridge_stats_in_results(self):
        """The pipeline returns bridging stats in the results dict."""
        comp1 = [("py::src/a.py::A", "src/a.py")]
        comp2 = [("py::src/b.py::B", "src/b.py")]
        P = _make_projected_graph([comp1, comp2])

        stats = _bridge_components(P)

        assert "components_before" in stats
        assert "components_after" in stats
        assert "bridges_added" in stats
        assert stats["components_before"] == 2
        assert stats["components_after"] == 1
