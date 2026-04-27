"""Tests for hierarchical Leiden clustering (Phase 1 of revised plan).

Validates:
- Section ⊃ page nesting (no cross-section splits)
- Domain coherence on synthetic graphs
- Hub reintegration
- Feature flag gating
- Algorithm metadata recording
- Edge cases (single node, disconnected, all-hub)
"""

import random
from collections import Counter
from typing import Dict, List, Set, Tuple
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from plugin_implementation.graph_clustering import (
    architectural_projection,
    hierarchical_leiden_cluster,
    persist_clusters,
    reintegrate_hubs,
    run_phase3,
)
from plugin_implementation.feature_flags import FeatureFlags


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_domain_graph(
    n_domains: int = 4,
    n_subgroups: int = 3,
    nodes_per_subgroup: int = 8,
    intra_weight: float = 5.0,
    cross_sub_weight: float = 0.8,
    cross_domain_weight: float = 0.05,
    seed: int = 42,
) -> Tuple[nx.MultiDiGraph, Dict[int, List[str]]]:
    """Build a synthetic graph with known domain/subgroup structure.

    Returns:
        (G, domain_nodes) where domain_nodes maps domain_id → node_ids.
    """
    rng = random.Random(seed)
    G = nx.MultiDiGraph()
    domain_nodes: Dict[int, List[str]] = {}
    node_id = 0

    for d in range(n_domains):
        domain_nodes[d] = []
        for s in range(n_subgroups):
            rel_path = f"domain_{d}/module_{s}.py"
            sub_nodes = []
            for _ in range(nodes_per_subgroup):
                nid = f"py::mod{d}_{s}::Cls{node_id}"
                G.add_node(nid, symbol_type="class", rel_path=rel_path)
                sub_nodes.append(nid)
                node_id += 1
            # Dense intra-subgroup edges
            for i, u in enumerate(sub_nodes):
                for j, v in enumerate(sub_nodes):
                    if i < j:
                        G.add_edge(u, v, weight=intra_weight)
                        G.add_edge(v, u, weight=intra_weight)
            # Cross-subgroup edges within domain
            if s > 0:
                prev_sub = domain_nodes[d][-(nodes_per_subgroup):]
                for _ in range(3):
                    u = sub_nodes[rng.randint(0, len(sub_nodes)-1)]
                    v = prev_sub[rng.randint(0, len(prev_sub)-1)]
                    G.add_edge(u, v, weight=cross_sub_weight)
                    G.add_edge(v, u, weight=cross_sub_weight)
            domain_nodes[d].extend(sub_nodes)

    # Weak cross-domain edges
    for d in range(n_domains - 1):
        u = domain_nodes[d][0]
        v = domain_nodes[d+1][0]
        G.add_edge(u, v, weight=cross_domain_weight)

    return G, domain_nodes


def _mock_db():
    """Create a mock DB that remembers set_meta and set_clusters_batch calls."""
    db = MagicMock()
    db.conn = MagicMock()
    db._meta_store = {}

    def _set_meta(key, value):
        db._meta_store[key] = value

    db.set_meta = MagicMock(side_effect=_set_meta)
    db.set_clusters_batch = MagicMock()
    db.set_hub = MagicMock()

    # _load_hubs_from_db reads from conn.execute
    db.conn.execute.return_value.fetchall.return_value = []
    return db


# ═════════════════════════════════════════════════════════════════════════════
# Core Hierarchical Leiden
# ═════════════════════════════════════════════════════════════════════════════

class TestHierarchicalLeidenCluster:
    """Test the hierarchical_leiden_cluster() function directly."""

    def test_4_domain_graph(self):
        """4 domains × 3 subgroups × 8 nodes = 96 nodes."""
        G, domain_nodes = _build_domain_graph(4, 3, 8)
        result = hierarchical_leiden_cluster(G, hubs=set())

        assert result["algorithm_metadata"]["algorithm"] == "hierarchical_leiden_file_contracted"
        n_sections = result["algorithm_metadata"]["sections"]
        n_pages = result["algorithm_metadata"]["pages"]

        # Should find ≈4 sections and reasonable pages
        assert 2 <= n_sections <= 8, f"Expected ~4 sections, got {n_sections}"
        assert n_pages >= 4, f"Expected ≥4 pages, got {n_pages}"

    def test_6_domain_graph(self):
        """6 domains × 5 subgroups × 15 nodes = 450 nodes."""
        G, domain_nodes = _build_domain_graph(6, 5, 15)
        result = hierarchical_leiden_cluster(G, hubs=set())

        n_sections = result["algorithm_metadata"]["sections"]
        n_pages = result["algorithm_metadata"]["pages"]
        assert 3 <= n_sections <= 12
        assert n_pages >= 6

    def test_10_domain_graph(self):
        """10 domains × 4 subgroups × 10 nodes = 400 nodes."""
        G, domain_nodes = _build_domain_graph(10, 4, 10)
        result = hierarchical_leiden_cluster(G, hubs=set())

        n_sections = result["algorithm_metadata"]["sections"]
        n_pages = result["algorithm_metadata"]["pages"]
        assert 4 <= n_sections <= 16
        assert n_pages >= 10

    def test_nesting_no_cross_section_splits(self):
        """Every page must fall entirely within one section."""
        G, _ = _build_domain_graph(4, 3, 8)
        result = hierarchical_leiden_cluster(G, hubs=set())

        # Verify each page sits in exactly one section
        for sec_id, sec_data in result["sections"].items():
            for pg_id, node_ids in sec_data["pages"].items():
                for nid in node_ids:
                    assert result["macro_assignments"][nid] == sec_id, (
                        f"Node {nid} in page {pg_id} belongs to section "
                        f"{result['macro_assignments'][nid]}, expected {sec_id}"
                    )

    def test_domain_coherence(self):
        """Nodes from the same domain should mostly land in the same section."""
        G, domain_nodes = _build_domain_graph(4, 3, 8)
        result = hierarchical_leiden_cluster(G, hubs=set())

        coherences = []
        for d, nodes in domain_nodes.items():
            sections = [result["macro_assignments"][n] for n in nodes]
            counts = Counter(sections)
            majority = max(counts.values())
            coherences.append(majority / len(nodes))

        avg = sum(coherences) / len(coherences)
        assert avg >= 0.8, f"Average domain coherence {avg:.0%} < 80%"

    def test_all_nodes_assigned(self):
        """Every non-hub node must have both macro and micro assignments."""
        G, _ = _build_domain_graph(4, 3, 8)
        result = hierarchical_leiden_cluster(G, hubs=set())

        for nid in G.nodes():
            assert nid in result["macro_assignments"], f"Missing macro: {nid}"
            sec_id = result["macro_assignments"][nid]
            assert nid in result["micro_assignments"][sec_id], (
                f"Missing micro: {nid}"
            )

    def test_hub_exclusion(self):
        """Hub nodes should NOT appear in assignments."""
        G, domain_nodes = _build_domain_graph(4, 3, 8)
        hubs = {domain_nodes[0][0], domain_nodes[1][0]}
        result = hierarchical_leiden_cluster(G, hubs=hubs)

        for hub in hubs:
            assert hub not in result["macro_assignments"]

    def test_metadata_fields(self):
        """Algorithm metadata should carry all required fields."""
        G, _ = _build_domain_graph(4, 3, 8)
        result = hierarchical_leiden_cluster(G, hubs=set())
        meta = result["algorithm_metadata"]

        assert meta["algorithm"] == "hierarchical_leiden_file_contracted"
        assert "section_resolution" in meta
        assert "page_resolution" in meta
        assert "sections" in meta
        assert "pages" in meta
        assert "seed" in meta
        assert "file_nodes" in meta
        assert "connected_files" in meta
        assert "isolated_files" in meta

    def test_custom_resolution(self):
        """Custom resolution parameters should be respected."""
        G, _ = _build_domain_graph(4, 3, 8)
        # Very low page resolution → fewer pages
        result_coarse = hierarchical_leiden_cluster(
            G, hubs=set(), page_resolution=0.05,
        )
        # Default page resolution
        result_fine = hierarchical_leiden_cluster(
            G, hubs=set(), page_resolution=1.0,
        )
        # Coarse should produce fewer or equal pages
        assert (result_coarse["algorithm_metadata"]["pages"]
                <= result_fine["algorithm_metadata"]["pages"])


class TestHierarchicalLeidenEdgeCases:
    """Edge cases and degenerate graphs."""

    def test_single_node(self):
        """Single-node graph → 1 section, 1 page."""
        G = nx.MultiDiGraph()
        G.add_node("py::a::Cls", symbol_type="class")
        result = hierarchical_leiden_cluster(G, hubs=set())

        assert result["algorithm_metadata"]["sections"] == 1
        assert result["algorithm_metadata"]["pages"] == 1

    def test_two_disconnected_components(self):
        """Two disconnected cliques in different files → 2 sections."""
        G = nx.MultiDiGraph()
        for i in range(5):
            nid = f"py::a::Cls{i}"
            G.add_node(nid, symbol_type="class", rel_path="pkg_a/module.py")
            for j in range(i):
                G.add_edge(f"py::a::Cls{i}", f"py::a::Cls{j}", weight=5.0)
                G.add_edge(f"py::a::Cls{j}", f"py::a::Cls{i}", weight=5.0)
        for i in range(5):
            nid = f"py::b::Cls{i}"
            G.add_node(nid, symbol_type="class", rel_path="pkg_b/module.py")
            for j in range(i):
                G.add_edge(f"py::b::Cls{i}", f"py::b::Cls{j}", weight=5.0)
                G.add_edge(f"py::b::Cls{j}", f"py::b::Cls{i}", weight=5.0)

        result = hierarchical_leiden_cluster(G, hubs=set())
        assert result["algorithm_metadata"]["sections"] == 2

    def test_all_hubs(self):
        """Every node is a hub → empty sections."""
        G = nx.MultiDiGraph()
        for i in range(5):
            G.add_node(f"py::a::Cls{i}", symbol_type="class")
        hubs = set(G.nodes())
        result = hierarchical_leiden_cluster(G, hubs=hubs)

        assert result["algorithm_metadata"]["sections"] == 0
        assert result["algorithm_metadata"]["pages"] == 0

    def test_no_edges(self):
        """Nodes in different files with no edges → each file is its own section."""
        G = nx.MultiDiGraph()
        for i in range(10):
            G.add_node(f"py::a::Cls{i}", symbol_type="class",
                       rel_path=f"pkg/module_{i}.py")
        result = hierarchical_leiden_cluster(G, hubs=set())

        # Each file is isolated → each gets its own section
        assert result["algorithm_metadata"]["sections"] == 10
        assert result["algorithm_metadata"]["pages"] == 10

    def test_star_graph(self):
        """Star topology: one center connected to all leafs."""
        G = nx.MultiDiGraph()
        G.add_node("center", symbol_type="class")
        for i in range(20):
            nid = f"leaf_{i}"
            G.add_node(nid, symbol_type="function")
            G.add_edge("center", nid, weight=1.0)
            G.add_edge(nid, "center", weight=1.0)
        result = hierarchical_leiden_cluster(G, hubs=set())
        # Should produce at least 1 section
        assert result["algorithm_metadata"]["sections"] >= 1


# ═════════════════════════════════════════════════════════════════════════════
# Feature Flag Gating
# ═════════════════════════════════════════════════════════════════════════════

class TestFeatureFlagGating:
    """run_phase3 dispatches to the correct pipeline based on flags."""

    def test_legacy_when_flag_disabled(self):
        """When hierarchical_leiden=False, legacy pipeline runs."""
        G, _ = _build_domain_graph(2, 2, 5)
        db = _mock_db()
        flags = FeatureFlags(hierarchical_leiden=False)

        result = run_phase3(db, G, feature_flags=flags)

        # Legacy pipeline key indicators: no algorithm_metadata
        assert "macro" in result
        assert "micro" in result
        # Legacy doesn't store algorithm_metadata at top level
        assert result.get("algorithm_metadata") is None

    def test_hierarchical_when_flag_enabled(self):
        """When hierarchical_leiden=True, hierarchical pipeline runs."""
        G, _ = _build_domain_graph(2, 2, 5)
        db = _mock_db()
        flags = FeatureFlags(hierarchical_leiden=True)

        result = run_phase3(db, G, feature_flags=flags)

        assert "algorithm_metadata" in result
        assert result["algorithm_metadata"]["algorithm"] == "hierarchical_leiden_file_contracted"

    def test_flag_default_uses_legacy(self):
        """Default flags → legacy pipeline."""
        G, _ = _build_domain_graph(2, 2, 5)
        db = _mock_db()
        flags = FeatureFlags()  # all defaults = False

        result = run_phase3(db, G, feature_flags=flags)
        assert result.get("algorithm_metadata") is None


# ═════════════════════════════════════════════════════════════════════════════
# Full Pipeline (hierarchical path)
# ═════════════════════════════════════════════════════════════════════════════

class TestHierarchicalPipeline:
    """End-to-end through _run_phase3_hierarchical_leiden."""

    def test_persists_to_db(self):
        """Clusters are written to DB."""
        G, _ = _build_domain_graph(3, 2, 6)
        db = _mock_db()
        flags = FeatureFlags(hierarchical_leiden=True)

        run_phase3(db, G, feature_flags=flags)

        db.set_clusters_batch.assert_called_once()
        db.conn.commit.assert_called()

    def test_stores_metadata_in_db(self):
        """Algorithm metadata stored via set_meta."""
        G, _ = _build_domain_graph(3, 2, 6)
        db = _mock_db()
        flags = FeatureFlags(hierarchical_leiden=True)

        run_phase3(db, G, feature_flags=flags)

        assert db._meta_store.get("phase3_completed") is True
        assert "phase3_stats" in db._meta_store
        assert "phase3_algorithm" in db._meta_store
        assert db._meta_store["phase3_algorithm"]["algorithm"] == "hierarchical_leiden_file_contracted"

    def test_hub_reintegration(self):
        """Hubs are reintegrated after clustering."""
        G, domain_nodes = _build_domain_graph(3, 2, 6)
        # Mark two nodes as hubs
        hub_nodes = {domain_nodes[0][0], domain_nodes[1][0]}
        db = _mock_db()
        # Return hub nodes from DB — use dicts to match sqlite3.Row behavior
        db.conn.execute.return_value.fetchall.return_value = [
            {"node_id": h} for h in hub_nodes
        ]
        flags = FeatureFlags(hierarchical_leiden=True)

        result = run_phase3(db, G, feature_flags=flags)

        assert result["hubs"]["total"] > 0

    def test_child_propagation(self):
        """Child nodes (methods) inherit parent's cluster."""
        G = nx.MultiDiGraph()
        # Create a class and its method
        G.add_node("py::mod::MyClass", symbol_type="class")
        G.add_node("py::mod::MyClass.do_stuff", symbol_type="method",
                    parent_symbol="MyClass")
        G.add_edge("py::mod::MyClass", "py::mod::MyClass.do_stuff", weight=1.0)
        # Add another class for at least 2 nodes
        G.add_node("py::mod::Other", symbol_type="class")
        G.add_edge("py::mod::MyClass", "py::mod::Other", weight=2.0)

        db = _mock_db()
        flags = FeatureFlags(hierarchical_leiden=True)

        result = run_phase3(db, G, feature_flags=flags)

        # The method should have been propagated
        batch_call = db.set_clusters_batch.call_args
        batch = batch_call[0][0]
        node_ids_in_batch = {t[0] for t in batch}
        assert "py::mod::MyClass.do_stuff" in node_ids_in_batch

    def test_stable_page_counts(self):
        """Running twice with same seed gives identical results."""
        G, _ = _build_domain_graph(4, 3, 8)
        db1 = _mock_db()
        db2 = _mock_db()
        flags = FeatureFlags(hierarchical_leiden=True)

        r1 = run_phase3(db1, G, feature_flags=flags)
        r2 = run_phase3(db2, G, feature_flags=flags)

        assert r1["macro"]["cluster_count"] == r2["macro"]["cluster_count"]
        assert r1["micro"]["total_pages"] == r2["micro"]["total_pages"]


# ═════════════════════════════════════════════════════════════════════════════
# Regression: Legacy pipeline unchanged
# ═════════════════════════════════════════════════════════════════════════════

class TestLegacyPipelineUnchanged:
    """Verify the legacy path hasn't regressed."""

    def test_legacy_produces_results(self):
        G, _ = _build_domain_graph(3, 2, 6)
        db = _mock_db()
        flags = FeatureFlags(hierarchical_leiden=False)

        result = run_phase3(db, G, feature_flags=flags)

        assert result["macro"]["cluster_count"] > 0
        assert result["micro"]["total_pages"] > 0
        db.set_clusters_batch.assert_called_once()

    def test_legacy_persists_phase3_completed(self):
        G, _ = _build_domain_graph(2, 2, 5)
        db = _mock_db()
        flags = FeatureFlags(hierarchical_leiden=False)

        run_phase3(db, G, feature_flags=flags)

        assert db._meta_store.get("phase3_completed") is True
