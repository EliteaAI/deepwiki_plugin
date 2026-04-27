"""
Tests for file-level graph contraction and file-contracted Leiden clustering.

Validates that:
- _contract_to_file_graph() correctly aggregates node-level edges into
  file-level edges with summed weights.
- hierarchical_leiden_cluster() uses file contraction for sections and
  per-section node-level Leiden for pages.
- Isolated files are assigned to nearest section by directory proximity.
- The return format is identical to the previous implementation.
"""

from collections import Counter
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from plugin_implementation.graph_clustering import (
    LEIDEN_FILE_SECTION_RESOLUTION,
    LEIDEN_PAGE_RESOLUTION,
    _contract_to_file_graph,
    _detect_page_centroids,
    hierarchical_leiden_cluster,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_graph(nodes_by_file, cross_file_edges=None, same_file_edges=None):
    """Build a MultiDiGraph with rel_path attributes and weighted edges.

    Args:
        nodes_by_file: {rel_path: [(node_id, symbol_type), ...]}
        cross_file_edges: [(src, tgt, weight), ...]
        same_file_edges: [(src, tgt, weight), ...]
    """
    G = nx.MultiDiGraph()
    for rel_path, nodes in nodes_by_file.items():
        for nid, sym_type in nodes:
            G.add_node(nid, rel_path=rel_path, file_name=rel_path,
                       symbol_type=sym_type)
    for src, tgt, w in (cross_file_edges or []):
        G.add_edge(src, tgt, weight=w, edge_class="structural",
                   rel_type="references")
    for src, tgt, w in (same_file_edges or []):
        G.add_edge(src, tgt, weight=w, edge_class="structural",
                   rel_type="defines")
    return G


def _simple_3file_graph():
    """3 files, 2 have strong cross-file edges, 1 is weakly connected.

    File layout:
      src/core.py: ClassA, func_a (both class)
      src/util.py: ClassB (class)
      lib/helper.py: ClassC (class)

    Cross edges: core→util strong (weight 5), core→helper weak (weight 0.5)
    """
    nodes = {
        "src/core.py": [("core_A", "class"), ("core_func_a", "function")],
        "src/util.py": [("util_B", "class")],
        "lib/helper.py": [("helper_C", "class")],
    }
    cross = [
        ("core_A", "util_B", 5.0),
        ("core_func_a", "util_B", 3.0),  # core→util total = 8.0
        ("core_A", "helper_C", 0.5),      # core→helper total = 0.5
    ]
    same = [
        ("core_A", "core_func_a", 2.0),   # within core.py — should be dropped
    ]
    return _make_graph(nodes, cross, same)


def _multi_cluster_graph():
    """6 files forming 2 natural clusters + 1 isolated file.

    Cluster 1 (src/): core.py ↔ util.py ↔ api.py (strong cross-file)
    Cluster 2 (lib/): math.py ↔ stats.py (strong cross-file)
    Isolated: docs/readme.md (no cross-file edges)
    """
    nodes = {
        "src/core.py": [("core_A", "class"), ("core_B", "function")],
        "src/util.py": [("util_A", "class"), ("util_B", "function")],
        "src/api.py": [("api_A", "class")],
        "lib/math.py": [("math_A", "class"), ("math_B", "function")],
        "lib/stats.py": [("stats_A", "class")],
        "docs/readme.md": [("doc_A", "markdown_section")],
    }
    cross = [
        # Cluster 1: strong intra-cluster
        ("core_A", "util_A", 10.0),
        ("core_B", "util_B", 8.0),
        ("util_A", "api_A", 6.0),
        ("core_A", "api_A", 4.0),
        # Cluster 2: strong intra-cluster
        ("math_A", "stats_A", 12.0),
        ("math_B", "stats_A", 5.0),
        # Weak inter-cluster bridge
        ("core_A", "math_A", 0.3),
    ]
    same = [
        ("core_A", "core_B", 3.0),
        ("util_A", "util_B", 2.0),
        ("math_A", "math_B", 4.0),
    ]
    return _make_graph(nodes, cross, same)


# ═══════════════════════════════════════════════════════════════════════════
# _contract_to_file_graph tests
# ═══════════════════════════════════════════════════════════════════════════

class TestContractToFileGraph:
    """Tests for _contract_to_file_graph()."""

    def test_basic_contraction(self):
        """File nodes created, same-file edges dropped, cross-file summed."""
        G = _simple_3file_graph()
        FG, file_to_nodes = _contract_to_file_graph(G)

        # 3 file nodes
        assert set(FG.nodes()) == {"src/core.py", "src/util.py", "lib/helper.py"}

        # File-to-nodes mapping
        assert len(file_to_nodes["src/core.py"]) == 2
        assert len(file_to_nodes["src/util.py"]) == 1
        assert len(file_to_nodes["lib/helper.py"]) == 1

        # Cross-file edges with summed weights
        assert FG.has_edge("src/core.py", "src/util.py")
        assert FG.has_edge("src/core.py", "lib/helper.py")
        # core→util: 5.0 + 3.0 = 8.0
        w_cu = FG["src/core.py"]["src/util.py"]["weight"]
        assert abs(w_cu - 8.0) < 0.01
        # core→helper: 0.5
        w_ch = FG["src/core.py"]["lib/helper.py"]["weight"]
        assert abs(w_ch - 0.5) < 0.01

    def test_same_file_edges_dropped(self):
        """Edges within the same file are not in the contracted graph."""
        G = _simple_3file_graph()
        FG, _ = _contract_to_file_graph(G)
        # No self-loops
        for u, v in FG.edges():
            assert u != v

    def test_no_edge_between_unconnected_files(self):
        """util.py and helper.py have no direct cross-file edges."""
        G = _simple_3file_graph()
        FG, _ = _contract_to_file_graph(G)
        assert not FG.has_edge("src/util.py", "lib/helper.py")

    def test_empty_graph(self):
        """Empty graph produces empty file graph."""
        G = nx.MultiDiGraph()
        FG, ftm = _contract_to_file_graph(G)
        assert FG.number_of_nodes() == 0
        assert FG.number_of_edges() == 0
        assert ftm == {}

    def test_single_file(self):
        """Single-file graph: one file node, no edges."""
        G = _make_graph(
            {"main.py": [("a", "class"), ("b", "function")]},
            same_file_edges=[("a", "b", 1.0)],
        )
        FG, ftm = _contract_to_file_graph(G)
        assert FG.number_of_nodes() == 1
        assert FG.number_of_edges() == 0
        assert "main.py" in ftm
        assert set(ftm["main.py"]) == {"a", "b"}

    def test_multi_cluster_contraction(self):
        """6-file graph contracts to 6 file nodes with correct edges."""
        G = _multi_cluster_graph()
        FG, ftm = _contract_to_file_graph(G)

        assert FG.number_of_nodes() == 6
        # Cross-file edges: core↔util, util↔api, core↔api, math↔stats, core↔math
        assert FG.number_of_edges() == 5

        # Strong intra-cluster edges
        assert FG["src/core.py"]["src/util.py"]["weight"] == 18.0  # 10+8
        assert FG["src/util.py"]["src/api.py"]["weight"] == 6.0
        assert FG["src/core.py"]["src/api.py"]["weight"] == 4.0
        assert FG["lib/math.py"]["lib/stats.py"]["weight"] == 17.0  # 12+5

        # Weak inter-cluster edge
        assert abs(FG["src/core.py"]["lib/math.py"]["weight"] - 0.3) < 0.01

    def test_bidirectional_edges_summed(self):
        """Directed edges A→B and B→A both contribute to undirected file edge."""
        G = _make_graph(
            {
                "a.py": [("a1", "class")],
                "b.py": [("b1", "class")],
            },
            cross_file_edges=[("a1", "b1", 2.0), ("b1", "a1", 3.0)],
        )
        FG, _ = _contract_to_file_graph(G)
        assert FG["a.py"]["b.py"]["weight"] == 5.0

    def test_nodes_without_rel_path_use_unknown(self):
        """Nodes without rel_path are grouped under '<unknown>'."""
        G = nx.MultiDiGraph()
        G.add_node("orphan_a", symbol_type="function")
        G.add_node("orphan_b", symbol_type="class")
        G.add_edge("orphan_a", "orphan_b", weight=1.0)

        FG, ftm = _contract_to_file_graph(G)
        assert "<unknown>" in ftm
        assert len(ftm["<unknown>"]) == 2
        # Same "file" → no edge in file graph
        assert FG.number_of_edges() == 0

    def test_parallel_edges_between_same_files(self):
        """Multiple parallel edges between same node-pairs sum correctly."""
        G = _make_graph(
            {
                "a.py": [("a1", "class")],
                "b.py": [("b1", "class")],
            },
        )
        # Add 3 parallel edges
        G.add_edge("a1", "b1", weight=1.0, rel_type="calls")
        G.add_edge("a1", "b1", weight=2.0, rel_type="references")
        G.add_edge("a1", "b1", weight=0.5, rel_type="creates")

        FG, _ = _contract_to_file_graph(G)
        assert abs(FG["a.py"]["b.py"]["weight"] - 3.5) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# hierarchical_leiden_cluster tests
# ═══════════════════════════════════════════════════════════════════════════

# Gate: skip if igraph/leidenalg not installed
try:
    import igraph
    import leidenalg
    _HAS_LEIDEN = True
except ImportError:
    _HAS_LEIDEN = False

leiden_required = pytest.mark.skipif(
    not _HAS_LEIDEN,
    reason="igraph and leidenalg required",
)


@leiden_required
class TestHierarchicalLeidenFileContracted:
    """Tests for the file-contracted hierarchical Leiden."""

    def test_return_format(self):
        """Return dict has all required keys."""
        G = _multi_cluster_graph()
        result = hierarchical_leiden_cluster(G, hubs=set())

        assert "sections" in result
        assert "macro_assignments" in result
        assert "micro_assignments" in result
        assert "algorithm_metadata" in result

        meta = result["algorithm_metadata"]
        assert meta["algorithm"] == "hierarchical_leiden_file_contracted"
        assert "file_nodes" in meta
        assert "connected_files" in meta
        assert "isolated_files" in meta

    def test_all_nodes_assigned(self):
        """Every non-hub node gets a section and page assignment."""
        G = _multi_cluster_graph()
        result = hierarchical_leiden_cluster(G, hubs=set())

        all_nodes = set(G.nodes())
        assigned = set(result["macro_assignments"].keys())
        assert assigned == all_nodes

        # Every node also has a page assignment
        for nid, sec_id in result["macro_assignments"].items():
            assert nid in result["micro_assignments"][sec_id]

    def test_hubs_excluded(self):
        """Hub nodes are excluded from clustering."""
        G = _multi_cluster_graph()
        hubs = {"core_A"}
        result = hierarchical_leiden_cluster(G, hubs=hubs)

        assert "core_A" not in result["macro_assignments"]
        all_pages_nodes = set()
        for sec in result["sections"].values():
            for nids in sec["pages"].values():
                all_pages_nodes.update(nids)
        assert "core_A" not in all_pages_nodes

    def test_two_clusters_detected(self):
        """Graph with 2 natural clusters produces ≥2 sections."""
        G = _multi_cluster_graph()
        result = hierarchical_leiden_cluster(G, hubs=set())

        n_sections = len(result["sections"])
        # Should detect at least 2 clusters (src/ vs lib/)
        # The docs file might go to either or become its own section
        assert n_sections >= 2, f"Expected ≥2 sections, got {n_sections}"

    def test_cluster_quality_src_vs_lib(self):
        """src/ nodes and lib/ nodes should be in different sections."""
        G = _multi_cluster_graph()
        result = hierarchical_leiden_cluster(G, hubs=set())

        macro = result["macro_assignments"]
        src_sections = {macro[n] for n in ["core_A", "core_B", "util_A",
                                            "util_B", "api_A"]
                        if n in macro}
        lib_sections = {macro[n] for n in ["math_A", "math_B", "stats_A"]
                        if n in macro}

        # src and lib should be in different sections
        assert src_sections.isdisjoint(lib_sections), (
            f"src sections {src_sections} overlap with lib sections {lib_sections}"
        )

    def test_isolated_file_assigned(self):
        """Isolated file (no cross-file edges) still gets assigned."""
        G = _multi_cluster_graph()
        result = hierarchical_leiden_cluster(G, hubs=set())

        assert "doc_A" in result["macro_assignments"]

    def test_pages_within_sections(self):
        """Each section has at least one page."""
        G = _multi_cluster_graph()
        result = hierarchical_leiden_cluster(G, hubs=set())

        for sec_id, sec in result["sections"].items():
            assert len(sec["pages"]) >= 1
            # Pages have centroids
            assert "centroids" in sec

    def test_pages_cover_all_section_nodes(self):
        """All nodes in a section appear in exactly one page."""
        G = _multi_cluster_graph()
        result = hierarchical_leiden_cluster(G, hubs=set())

        for sec_id, sec in result["sections"].items():
            page_nodes = set()
            for pg_id, nids in sec["pages"].items():
                for nid in nids:
                    assert nid not in page_nodes, f"{nid} in multiple pages"
                    page_nodes.add(nid)

    def test_empty_graph(self):
        """Empty graph produces empty result."""
        G = nx.MultiDiGraph()
        result = hierarchical_leiden_cluster(G, hubs=set())
        assert result["sections"] == {}
        assert result["macro_assignments"] == {}

    def test_all_hubs_empty_result(self):
        """If all nodes are hubs, result is empty."""
        G = _make_graph({"a.py": [("a1", "class")]})
        result = hierarchical_leiden_cluster(G, hubs={"a1"})
        assert result["sections"] == {}

    def test_single_file_single_section(self):
        """Single-file graph → single section."""
        G = _make_graph(
            {"main.py": [("a", "class"), ("b", "function"), ("c", "method")]},
            same_file_edges=[("a", "b", 1.0), ("b", "c", 1.0)],
        )
        result = hierarchical_leiden_cluster(G, hubs=set())
        assert len(result["sections"]) == 1

    def test_file_metadata_in_result(self):
        """Algorithm metadata includes file contraction stats."""
        G = _multi_cluster_graph()
        result = hierarchical_leiden_cluster(G, hubs=set())

        meta = result["algorithm_metadata"]
        assert meta["file_nodes"] == 6  # 6 files
        assert meta["connected_files"] + meta["isolated_files"] == 6

    def test_large_graph_reasonable_sections(self):
        """Larger graph (20 files, 2 clusters) produces reasonable sections."""
        nodes = {}
        cross = []
        # Cluster A: 10 files in src/
        for i in range(10):
            fpath = f"src/module_{i}.py"
            nodes[fpath] = [(f"src_cls_{i}", "class"),
                            (f"src_fn_{i}", "function")]
        # Cluster B: 10 files in lib/
        for i in range(10):
            fpath = f"lib/helper_{i}.py"
            nodes[fpath] = [(f"lib_cls_{i}", "class"),
                            (f"lib_fn_{i}", "function")]

        # Strong intra-cluster cross-file edges
        for i in range(9):
            cross.append((f"src_cls_{i}", f"src_cls_{i+1}", 5.0))
            cross.append((f"lib_cls_{i}", f"lib_cls_{i+1}", 5.0))
        # Weak inter-cluster edge
        cross.append(("src_cls_0", "lib_cls_0", 0.1))

        G = _make_graph(nodes, cross)
        result = hierarchical_leiden_cluster(G, hubs=set())

        n_sections = len(result["sections"])
        assert 2 <= n_sections <= 6, f"Expected 2-6 sections, got {n_sections}"

    def test_resolution_affects_section_count(self):
        """Higher section resolution produces more sections."""
        G = _multi_cluster_graph()
        result_low = hierarchical_leiden_cluster(
            G, hubs=set(), section_resolution=0.5)
        result_high = hierarchical_leiden_cluster(
            G, hubs=set(), section_resolution=5.0)

        # Higher γ should produce at least as many sections
        n_low = len(result_low["sections"])
        n_high = len(result_high["sections"])
        assert n_high >= n_low

    def test_page_centroids_populated(self):
        """Every page has centroids detected."""
        G = _multi_cluster_graph()
        result = hierarchical_leiden_cluster(G, hubs=set())

        for sec in result["sections"].values():
            for pg_id in sec["pages"]:
                assert pg_id in sec["centroids"]
