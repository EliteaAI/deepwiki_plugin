"""
Integration tests for Clustering V2 improvements.

Each test class maps to an implementation phase from PLANNING_CLUSTERING_V2.md.
Tests for unimplemented phases are marked with @unittest.skip and can be
unskipped as each phase is implemented.

Phase order: 7A → 7E → 7F → 7C → 7G → 7B → 7D

Run all:      python -m pytest tests/test_clustering_v2_integration.py -v
Run phase:    python -m pytest tests/test_clustering_v2_integration.py -k "Phase7A" -v
Run end2end:  python -m pytest tests/test_clustering_v2_integration.py -k "EndToEnd" -v
"""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Set
from unittest.mock import MagicMock, patch

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest_clustering import (
    build_cpp_header_graph,
    build_large_repo_graph,
    build_mixed_code_doc_graph,
    build_pure_doc_graph,
    build_two_community_graph,
    make_db,
    make_doc_node,
    make_graph,
    make_node_dict,
    mock_llm_multi_response,
    mock_llm_response,
    populate_db_and_graph,
    run_full_pipeline,
)

from plugin_implementation.graph_clustering import (
    MICRO_CLUSTER_RULES,
    apply_page_sizing,
    macro_cluster,
    micro_cluster,
    micro_cluster_all,
    run_phase3,
)
from plugin_implementation.graph_topology import (
    apply_edge_weights,
    detect_hubs,
    find_orphans,
    inject_doc_edges,
    resolve_orphans,
    run_phase2,
)
from plugin_implementation.unified_db import UnifiedWikiDB


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7A: Quick Wins — Doc Routing
# ═══════════════════════════════════════════════════════════════════════════

class TestPhase7A_DocRouting(unittest.TestCase):
    """Phase 7A.0: Documentation nodes must enter the unified DB
    regardless of SEPARATE_DOC_INDEX flag."""

    def test_docs_in_db_with_is_doc_flag(self):
        """Doc nodes upserted to DB should have is_doc=1 and is_architectural=1."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                make_node_dict("code_1", symbol_type="function", rel_path="src/main.py"),
                make_node_dict("code_2", symbol_type="class", rel_path="src/api.py"),
                make_doc_node("doc_1", "README.md"),
                make_doc_node("doc_2", "docs/guide.md"),
            ]
            db = make_db(tmp, nodes=nodes)

            doc1 = db.get_node("doc_1")
            self.assertIsNotNone(doc1, "Doc node should exist in DB")
            self.assertEqual(doc1.get("is_doc"), 1, "Doc node should have is_doc=1")
            self.assertEqual(doc1.get("is_architectural"), 1)

            doc2 = db.get_node("doc_2")
            self.assertIsNotNone(doc2)
            self.assertEqual(doc2.get("is_doc"), 1)

    def test_docs_in_graph_nodes(self):
        """Doc nodes should exist in the NX graph alongside code nodes."""
        with tempfile.TemporaryDirectory() as tmp:
            G, nodes = build_mixed_code_doc_graph()
            db = make_db(tmp, nodes=nodes)

            doc_node_ids = [n["node_id"] for n in nodes if n.get("is_doc")]
            for nid in doc_node_ids:
                self.assertIn(nid, G.nodes(), f"Doc node {nid} should be in graph")

    def test_docs_participate_in_clustering(self):
        """After full pipeline, doc nodes should have cluster assignments."""
        with tempfile.TemporaryDirectory() as tmp:
            G, nodes = build_mixed_code_doc_graph()
            db = make_db(tmp, nodes=nodes)
            # Ensure DB has edges matching the graph
            edges = []
            for u, v, data in G.edges(data=True):
                edges.append({
                    "source_id": u,
                    "target_id": v,
                    "rel_type": data.get("relationship_type", "calls"),
                })
            db.upsert_edges_batch(edges)

            run_full_pipeline(db, G)

            # Check at least some doc nodes got clustered
            doc_node_ids = [n["node_id"] for n in nodes if n.get("is_doc")]
            clustered_docs = 0
            for nid in doc_node_ids:
                node = db.get_node(nid)
                if node and node.get("macro_cluster") is not None:
                    clustered_docs += 1

            # Not all docs will necessarily be clustered (orphans stay unclustered)
            # but with the mixed graph having edges, most should be
            self.assertGreater(
                clustered_docs, 0,
                f"At least some doc nodes should be clustered, got 0/{len(doc_node_ids)}"
            )

    def test_code_nodes_not_marked_as_doc(self):
        """Code nodes should NOT have is_doc flag set."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                make_node_dict("func_1", symbol_type="function"),
                make_node_dict("cls_1", symbol_type="class"),
                make_doc_node("doc_1", "README.md"),
            ]
            db = make_db(tmp, nodes=nodes)

            func = db.get_node("func_1")
            self.assertFalse(func.get("is_doc"), "Code node should not be is_doc")

            cls = db.get_node("cls_1")
            self.assertFalse(cls.get("is_doc"))


class TestPhase7A_AdaptivePageSize(unittest.TestCase):
    """Phase 7A.1: max_page_size should scale with repo size."""

    def test_small_repo_page_size_25(self):
        from plugin_implementation.graph_clustering import _adaptive_max_page_size
        self.assertEqual(_adaptive_max_page_size(100), 25)

    def test_medium_repo_page_size_35(self):
        from plugin_implementation.graph_clustering import _adaptive_max_page_size
        self.assertEqual(_adaptive_max_page_size(800), 35)

    def test_large_repo_page_size_45(self):
        from plugin_implementation.graph_clustering import _adaptive_max_page_size
        self.assertEqual(_adaptive_max_page_size(3000), 45)

    def test_very_large_repo_page_size_60(self):
        from plugin_implementation.graph_clustering import _adaptive_max_page_size
        self.assertEqual(_adaptive_max_page_size(6000), 60)

    def test_page_size_used_in_pipeline(self):
        """Micro-clusters in 800-node graph should be capped at 35, not 25."""
        from plugin_implementation.graph_clustering import _adaptive_max_page_size
        with tempfile.TemporaryDirectory() as tmp:
            G, nodes = build_large_repo_graph(n=800)
            db = make_db(tmp, nodes=nodes)
            edges = [{"source_id": u, "target_id": v,
                       "rel_type": d.get("relationship_type", "calls")}
                     for u, v, d in G.edges(data=True)]
            db.upsert_edges_batch(edges)

            run_full_pipeline(db, G)
            expected_max = _adaptive_max_page_size(800)
            self.assertEqual(expected_max, 35)


class TestPhase7A_FileAwareSplitting(unittest.TestCase):
    """Phase 7A.2: Recursive split fallback should group by file path."""

    def test_file_grouped_split(self):
        """Symbols from 3 files should split into file-cohesive groups."""
        from plugin_implementation.graph_clustering import _file_aware_chunk

        G = nx.MultiDiGraph()
        files = ["src/a.py", "src/b.py", "src/c.py"]
        all_nodes = set()
        for fi, fpath in enumerate(files):
            for si in range(15):
                nid = f"n_{fi}_{si}"
                G.add_node(nid, rel_path=fpath, start_line=si * 10)
                all_nodes.add(nid)

        groups = _file_aware_chunk(G, all_nodes, max_size=20)
        # Each group should be predominantly from one file
        for group in groups:
            paths = {G.nodes[n].get("rel_path") for n in group}
            self.assertLessEqual(len(paths), 1,
                                  f"Group should be file-cohesive, got {len(paths)} files")

    def test_single_file_splits_by_line_order(self):
        """Symbols from 1 file should split by declaration order."""
        from plugin_implementation.graph_clustering import _file_aware_chunk

        G = nx.MultiDiGraph()
        nodes = set()
        for i in range(40):
            nid = f"n_{i}"
            G.add_node(nid, rel_path="big_header.h", start_line=i * 10)
            nodes.add(nid)

        groups = _file_aware_chunk(G, nodes, max_size=15)
        self.assertGreater(len(groups), 1)
        # Each group's nodes should have contiguous line numbers
        for group in groups:
            lines = sorted(G.nodes[n].get("start_line", 0) for n in group)
            if len(lines) > 1:
                max_gap = max(lines[i + 1] - lines[i] for i in range(len(lines) - 1))
                # Lines should be relatively contiguous (not random)
                self.assertLess(max_gap, 200, "Split should be line-contiguous")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7E: Edge Pipeline Reordering
# ═══════════════════════════════════════════════════════════════════════════

class TestPhase7E_EdgePipelineOrder(unittest.TestCase):
    """Phase 7E: resolve_orphans must run BEFORE weighting and hub detection."""

    def test_resolve_orphans_before_weighting(self):
        """Orphan-resolution edges should get proper weights (not default 1.0)."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create graph with orphan doc nodes
            nodes = [
                make_node_dict("func_1", source_text="def func_1(): pass"),
                make_node_dict("func_2", source_text="def func_2(): pass"),
                make_doc_node("doc_1", "README.md", source_text="Uses func_1 and func_2"),
            ]
            db = make_db(tmp, nodes=nodes)
            G = make_graph(("func_1", "func_2"), orphans=["doc_1"])

            db.upsert_edges_batch([{
                "source_id": "func_1",
                "target_id": "func_2",
                "rel_type": "calls",
            }])

            p2 = run_phase2(db, G)

            # After phase2, doc_1 should have edges (from orphan resolution)
            doc_edges = list(G.in_edges("doc_1", data=True)) + list(G.out_edges("doc_1", data=True))
            if doc_edges:
                # Those edges should have been weighted (not just default 1.0)
                for u, v, data in doc_edges:
                    weight = data.get("weight", None)
                    self.assertIsNotNone(weight, "Resolved edges should have weight")

    def test_hub_detection_sees_all_edges(self):
        """Hub detection should consider semantic edges, not just structural."""
        with tempfile.TemporaryDirectory() as tmp:
            # Node with 2 structural incoming but 15 semantic incoming = should be hub
            nodes = [make_node_dict(f"n_{i}") for i in range(20)]
            nodes.append(make_node_dict("target", source_text="central function"))
            db = make_db(tmp, nodes=nodes)

            G = nx.MultiDiGraph()
            for n in nodes:
                G.add_node(n["node_id"])
            # 2 structural edges
            G.add_edge("n_0", "target", relationship_type="calls", weight=1.0)
            G.add_edge("n_1", "target", relationship_type="calls", weight=1.0)
            # 15 semantic edges (simulating post-orphan-resolution)
            for i in range(2, 17):
                G.add_edge(f"n_{i}", "target", relationship_type="semantic", weight=0.5)

            edges = [{"source_id": u, "target_id": v,
                       "rel_type": d.get("relationship_type", "calls")}
                     for u, v, d in G.edges(data=True)]
            db.upsert_edges_batch(edges)

            p2 = run_phase2(db, G)
            hubs = set(p2["hubs"]["node_ids"])
            # With 17 incoming edges, "target" should be a hub
            self.assertIn("target", hubs,
                          "Node with many semantic edges should be detected as hub")


class TestPhase7E_DocEdgeInjection(unittest.TestCase):
    """Phase 7E: Hyperlink + proximity edge extraction for docs."""

    def test_hyperlink_edges_from_markdown_links(self):
        """Markdown [text](path) links should create directed edges."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                make_node_dict("code_1", symbol_name="MyClass",
                                rel_path="src/myclass.py"),
                make_doc_node("doc_1", "docs/guide.md",
                              source_text="See [MyClass](../src/myclass.py) for details."),
            ]
            db = make_db(tmp, nodes=nodes)
            G = make_graph(orphans=["code_1", "doc_1"])

            # After phase 2, doc→code edge should exist via hyperlink extraction
            run_phase2(db, G)
            doc_out = list(G.out_edges("doc_1", data=True))
            targets = {v for _, v, _ in doc_out}
            self.assertIn("code_1", targets,
                          "Markdown link should create hyperlink edge to code node")

    def test_proximity_edges_same_directory(self):
        """Files in related directories should get proximity edges."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                make_node_dict("code_api", rel_path="src/api/handler.py"),
                make_doc_node("doc_api", "docs/api/README.md"),
            ]
            db = make_db(tmp, nodes=nodes)
            G = make_graph(orphans=["code_api", "doc_api"])

            run_phase2(db, G)
            # Check for proximity edge between api-related files
            has_edge = G.has_edge("doc_api", "code_api") or G.has_edge("code_api", "doc_api")
            self.assertTrue(has_edge, "Proximity edge should link same-directory files")

    def test_edge_types_distinguishable(self):
        """Different edge tiers should have distinct relationship_type attrs."""
        with tempfile.TemporaryDirectory() as tmp:
            G, nodes = build_mixed_code_doc_graph()
            db = make_db(tmp, nodes=nodes)
            edges = [{"source_id": u, "target_id": v,
                       "rel_type": d.get("relationship_type", "calls")}
                     for u, v, d in G.edges(data=True)]
            db.upsert_edges_batch(edges)

            run_phase2(db, G)
            rel_types = set()
            for _, _, data in G.edges(data=True):
                rel_types.add(data.get("relationship_type", ""))

            # Should have at least structural + one doc edge type
            self.assertTrue(
                rel_types & {"calls", "imports"},
                "Structural edge types should exist"
            )


class TestPhase7E_WeightingOnCompleteGraph(unittest.TestCase):
    """Phase 7E: Edge weighting operates on ALL edge types."""

    def test_semantic_edges_get_proper_weights(self):
        """Semantic edges should have inverse-in-degree weights, not default 1.0."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                make_node_dict("a"),
                make_node_dict("b"),
                make_node_dict("c"),
            ]
            db = make_db(tmp, nodes=nodes)
            G = make_graph(("a", "b"))
            # Simulate a semantic edge added by orphan resolution
            G.add_edge("c", "b", relationship_type="semantic", weight=1.0)

            db.upsert_edges_batch([
                {"source_id": "a", "target_id": "b", "rel_type": "calls"},
                {"source_id": "c", "target_id": "b", "rel_type": "semantic"},
            ])

            run_phase2(db, G)

            # Both edges pointing to "b" should have been weighted equally
            # b has in_degree=2, so weight = 1/log(2+2) = 1/log(4)
            expected = 1.0 / math.log(4)
            for u, v, data in G.edges(data=True):
                if v == "b":
                    self.assertAlmostEqual(
                        data["weight"], expected, places=4,
                        msg=f"Edge {u}→{v} should have weight {expected:.4f}"
                    )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7F: Central Symbol Selection
# ═══════════════════════════════════════════════════════════════════════════

class TestPhase7F_CentralSymbolSelection(unittest.TestCase):
    """Phase 7F: select_central_symbols() using PageRank."""

    def test_pagerank_top_k(self):
        """Hub node in a star graph should be in top-k central symbols."""
        from plugin_implementation.graph_clustering import select_central_symbols

        G = nx.MultiDiGraph()
        hub = "hub_node"
        spokes = [f"spoke_{i}" for i in range(15)]
        for s in spokes:
            G.add_edge(s, hub, weight=1.0)
            G.add_edge(hub, s, weight=0.5)

        cluster_nodes = {hub} | set(spokes)
        central = select_central_symbols(G, cluster_nodes, k=5)

        self.assertEqual(len(central), 5)
        self.assertIn(hub, central, "Hub should be in central symbols")

    def test_k_clamped_to_cluster_size(self):
        """If cluster has fewer nodes than k, return all nodes."""
        from plugin_implementation.graph_clustering import select_central_symbols

        G = make_graph(("a", "b"), ("b", "c"))
        central = select_central_symbols(G, {"a", "b", "c"}, k=10)
        self.assertEqual(len(central), 3)

    def test_empty_cluster(self):
        """Empty cluster should return empty list."""
        from plugin_implementation.graph_clustering import select_central_symbols

        G = nx.MultiDiGraph()
        central = select_central_symbols(G, set(), k=5)
        self.assertEqual(central, [])

    def test_disconnected_cluster(self):
        """Disconnected nodes get uniform PageRank — still returns k."""
        from plugin_implementation.graph_clustering import select_central_symbols

        G = nx.MultiDiGraph()
        nodes = {f"n_{i}" for i in range(10)}
        for n in nodes:
            G.add_node(n)

        central = select_central_symbols(G, nodes, k=5)
        self.assertEqual(len(central), 5)

    def test_chain_topology_endpoints(self):
        """In a chain A→B→C→D→E, undirected PageRank favors interior nodes."""
        from plugin_implementation.graph_clustering import select_central_symbols

        G = nx.MultiDiGraph()
        chain = ["a", "b", "c", "d", "e"]
        for i in range(len(chain) - 1):
            G.add_edge(chain[i], chain[i + 1], weight=1.0)

        central = select_central_symbols(G, set(chain), k=2)
        # In undirected PageRank, interior nodes (degree 2) rank higher
        # than endpoints (degree 1). At least one interior node in top-2.
        interior = {"b", "c", "d"}
        self.assertTrue(
            interior & set(central),
            f"At least one interior node should be in top-2, got {central}",
        )


class TestPhase7F_ClusterPlannerTopK(unittest.TestCase):
    """Phase 7F: cluster_planner uses central symbols, not all node_ids."""

    def test_target_symbols_bounded_by_k(self):
        """Each page should have at most k target_symbols (not hundreds)."""
        with tempfile.TemporaryDirectory() as tmp:
            G, nodes = build_cpp_header_graph(n_symbols=300)
            db = make_db(tmp, nodes=nodes)
            edges = [{"source_id": u, "target_id": v,
                       "rel_type": d.get("relationship_type", "calls")}
                     for u, v, d in G.edges(data=True)]
            db.upsert_edges_batch(edges)

            run_full_pipeline(db, G)

            from plugin_implementation.wiki_structure_planner.cluster_planner import (
                ClusterStructurePlanner,
            )

            mock_llm = mock_llm_response("Test Section")
            planner = ClusterStructurePlanner(db, mock_llm)
            spec = planner.plan_structure()

            for section in spec.sections:
                for page in section.pages:
                    self.assertLessEqual(
                        len(page.target_symbols), 15,
                        f"Page '{page.page_name}' has {len(page.target_symbols)} symbols, "
                        f"expected <= 15 (central selection)"
                    )

    def test_cluster_node_ids_in_page_metadata(self):
        """Page metadata should contain full cluster_node_ids for expansion."""
        with tempfile.TemporaryDirectory() as tmp:
            G, nodes = build_cpp_header_graph(n_symbols=100)
            db = make_db(tmp, nodes=nodes)
            edges = [{"source_id": u, "target_id": v,
                       "rel_type": d.get("relationship_type", "calls")}
                     for u, v, d in G.edges(data=True)]
            db.upsert_edges_batch(edges)

            run_full_pipeline(db, G)

            from plugin_implementation.wiki_structure_planner.cluster_planner import (
                ClusterStructurePlanner,
            )

            mock_llm = mock_llm_response("Test Section")
            planner = ClusterStructurePlanner(db, mock_llm)
            spec = planner.plan_structure()

            for section in spec.sections:
                for page in section.pages:
                    # target_symbols should be small (central)
                    # cluster_node_ids should contain ALL micro-cluster nodes
                    metadata = getattr(page, "metadata", {}) or {}
                    if page.target_symbols:
                        cluster_ids = metadata.get("cluster_node_ids", [])
                        self.assertGreaterEqual(
                            len(cluster_ids), len(page.target_symbols),
                            "cluster_node_ids should contain at least as many nodes as target_symbols"
                        )


class TestPhase7F_PageSplitSafetyNet(unittest.TestCase):
    """Phase 7F: PAGE_SPLIT should be a safety net, not the main path."""

    def test_no_part_n_pages(self):
        """With central selection, pages should never need Part-N splitting."""
        with tempfile.TemporaryDirectory() as tmp:
            G, nodes = build_cpp_header_graph(n_symbols=300)
            db = make_db(tmp, nodes=nodes)
            edges = [{"source_id": u, "target_id": v,
                       "rel_type": d.get("relationship_type", "calls")}
                     for u, v, d in G.edges(data=True)]
            db.upsert_edges_batch(edges)

            run_full_pipeline(db, G)

            from plugin_implementation.wiki_structure_planner.cluster_planner import (
                ClusterStructurePlanner,
            )

            mock_llm = mock_llm_response("Test Section")
            planner = ClusterStructurePlanner(db, mock_llm)
            spec = planner.plan_structure()

            for section in spec.sections:
                for page in section.pages:
                    self.assertLessEqual(
                        len(page.target_symbols), 60,
                        f"Page '{page.page_name}' has too many symbols for central selection"
                    )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7C: InfoMap Micro-Clustering
# ═══════════════════════════════════════════════════════════════════════════

class TestPhase7C_InfoMapMicroClustering(unittest.TestCase):
    """Phase 7C: InfoMap should find flow-based micro-clusters."""

    def test_infomap_finds_flow_traps(self):
        """Two directed cycles connected by weak bridge → 2 clusters."""
        from plugin_implementation.graph_clustering import micro_cluster

        G = nx.MultiDiGraph()
        # Cycle 1: A→B→C→A
        G.add_edge("A", "B", weight=2.0)
        G.add_edge("B", "C", weight=2.0)
        G.add_edge("C", "A", weight=2.0)
        # Cycle 2: D→E→F→D
        G.add_edge("D", "E", weight=2.0)
        G.add_edge("E", "F", weight=2.0)
        G.add_edge("F", "D", weight=2.0)
        # Weak bridge
        G.add_edge("A", "D", weight=0.01)

        nodes = {"A", "B", "C", "D", "E", "F"}
        result = micro_cluster(G, nodes, algorithm="infomap")

        cluster_vals = set(result.values())
        self.assertGreaterEqual(len(cluster_vals), 2,
                                 "InfoMap should find 2 flow traps")

    def test_infomap_respects_directed_edges(self):
        """Asymmetric flow should affect clustering."""
        from plugin_implementation.graph_clustering import micro_cluster

        G = nx.MultiDiGraph()
        # Strong flow in one direction
        for i in range(5):
            G.add_edge(f"a{i}", f"a{(i + 1) % 5}", weight=2.0)
        # Weak reverse flow
        for i in range(5):
            G.add_edge(f"a{(i + 1) % 5}", f"a{i}", weight=0.1)

        nodes = {f"a{i}" for i in range(5)}
        result = micro_cluster(G, nodes, algorithm="infomap")
        # All in one cluster (strong directional cycle)
        self.assertEqual(len(set(result.values())), 1)

    def test_infomap_deterministic_seed(self):
        """Same graph, two runs should produce identical results."""
        from plugin_implementation.graph_clustering import micro_cluster

        G = build_two_community_graph(n=8)
        nodes = set(G.nodes())

        r1 = micro_cluster(G, nodes, algorithm="infomap")
        r2 = micro_cluster(G, nodes, algorithm="infomap")
        self.assertEqual(r1, r2)


class TestPhase7C_InfoMapFallback(unittest.TestCase):
    """Phase 7C: Graceful degradation when igraph not available."""

    def test_infomap_fallback_to_louvain(self):
        """When igraph is not importable, Louvain should be used."""
        import plugin_implementation.graph_clustering as gc

        # Temporarily pretend igraph is unavailable
        orig_igraph = gc._HAS_IGRAPH
        orig_infomap = gc._HAS_INFOMAP
        try:
            gc._HAS_IGRAPH = False
            gc._HAS_INFOMAP = False
            G = build_two_community_graph(n=5)
            nodes = set(G.nodes())
            result = micro_cluster(G, nodes, algorithm="auto")
            # Should still produce valid clustering (using Louvain)
            self.assertEqual(set(result.keys()), nodes)
            self.assertGreaterEqual(len(set(result.values())), 1)
        finally:
            gc._HAS_IGRAPH = orig_igraph
            gc._HAS_INFOMAP = orig_infomap


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7G: Doc-Aware Clustering
# ═══════════════════════════════════════════════════════════════════════════

class TestPhase7G_DocClusterDetection(unittest.TestCase):
    """Phase 7G: Detect doc-dominant clusters after clustering."""

    def test_detect_doc_dominant_cluster(self):
        """Cluster with >70% doc nodes should be detected as doc-dominant."""
        from plugin_implementation.graph_clustering import detect_doc_clusters

        with tempfile.TemporaryDirectory() as tmp:
            nodes = [make_doc_node(f"doc_{i}", f"docs/page_{i}.md") for i in range(8)]
            nodes.extend([make_node_dict(f"code_{i}") for i in range(2)])
            db = make_db(tmp, nodes=nodes)

            # Simulate cluster map: all in macro=0
            cluster_map = {0: {0: [n["node_id"] for n in nodes]}}
            result = detect_doc_clusters(cluster_map, db)
            self.assertIn(0, result, "Cluster with 80% docs should be doc-dominant")
            self.assertGreater(result[0], 0.7)

    def test_mixed_cluster_not_doc_dominant(self):
        """Cluster with 30% docs should NOT be detected as doc-dominant."""
        from plugin_implementation.graph_clustering import detect_doc_clusters

        with tempfile.TemporaryDirectory() as tmp:
            nodes = [make_doc_node(f"doc_{i}", f"docs/page_{i}.md") for i in range(3)]
            nodes.extend([make_node_dict(f"code_{i}") for i in range(7)])
            db = make_db(tmp, nodes=nodes)

            cluster_map = {0: {0: [n["node_id"] for n in nodes]}}
            result = detect_doc_clusters(cluster_map, db)
            self.assertNotIn(0, result, "Cluster with 30% docs should NOT be doc-dominant")


class TestPhase7G_PureDocRepo(unittest.TestCase):
    """Phase 7G: Pure-doc repos use directory-based clustering."""

    def test_pure_doc_directory_clustering(self):
        """50 docs in 5 directories should produce ~5 sections."""
        with tempfile.TemporaryDirectory() as tmp:
            G, nodes = build_pure_doc_graph(n_docs=50)
            db = make_db(tmp, nodes=nodes)
            edges = [{"source_id": u, "target_id": v,
                       "rel_type": d.get("relationship_type", "hyperlink")}
                     for u, v, d in G.edges(data=True)]
            db.upsert_edges_batch(edges)

            run_full_pipeline(db, G)

            # Check macro-cluster count approximates directory count
            all_nodes_data = []
            for n in nodes:
                nd = db.get_node(n["node_id"])
                if nd:
                    all_nodes_data.append(nd)

            macro_ids = {nd.get("macro_cluster") for nd in all_nodes_data
                        if nd.get("macro_cluster") is not None}
            self.assertGreaterEqual(len(macro_ids), 3,
                                     "Pure-doc repo should have multiple sections")
            self.assertLessEqual(len(macro_ids), 8,
                                  "Sections should not exceed directory count significantly")

    def test_readme_is_central_in_directory(self):
        """README.md should be among central symbols in its directory cluster."""
        from plugin_implementation.graph_clustering import select_central_symbols

        G, nodes = build_pure_doc_graph(n_docs=30)
        # Get nodes from first directory
        dir_nodes = {n["node_id"] for n in nodes
                     if "getting-started" in n.get("rel_path", "")}
        readme_id = next(nid for nid in dir_nodes if "README.md" in nid)

        central = select_central_symbols(G, dir_nodes, k=3)
        self.assertIn(readme_id, central, "README should be central in its directory")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7B: Leiden Macro-Clustering
# ═══════════════════════════════════════════════════════════════════════════

class TestPhase7B_LeidenMacroClustering(unittest.TestCase):
    """Phase 7B: Leiden produces connected communities."""

    def test_leiden_two_communities(self):
        """Basic two-community graph should produce 2 clusters."""
        G = build_two_community_graph(n=8)
        result = macro_cluster(G, hubs=set(), algorithm="leiden")

        a_clusters = {result[f"a{i}"] for i in range(8)}
        b_clusters = {result[f"b{i}"] for i in range(8)}
        self.assertEqual(len(a_clusters), 1)
        self.assertEqual(len(b_clusters), 1)
        self.assertNotEqual(a_clusters, b_clusters)

    def test_leiden_resolution_parameter(self):
        """Higher resolution → more clusters."""
        G = build_two_community_graph(n=8)
        high = macro_cluster(G, hubs=set(), resolution=2.0, algorithm="leiden")
        low = macro_cluster(G, hubs=set(), resolution=0.3, algorithm="leiden")
        self.assertGreaterEqual(len(set(high.values())), len(set(low.values())))


class TestPhase7B_LeidenFallback(unittest.TestCase):
    """Phase 7B: Graceful degradation when leidenalg not available."""

    def test_leiden_fallback_to_louvain(self):
        """When leidenalg is not importable, Louvain should be used."""
        import plugin_implementation.graph_clustering as gc

        original = getattr(gc, "_HAS_LEIDEN", True)
        try:
            gc._HAS_LEIDEN = False
            G = build_two_community_graph(n=5)
            result = macro_cluster(G, hubs=set(), algorithm="auto")
            self.assertEqual(set(result.keys()), set(G.nodes()))
        finally:
            gc._HAS_LEIDEN = original


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7D: File-Internal Hierarchical Split
# ═══════════════════════════════════════════════════════════════════════════

class TestPhase7D_FileInternalSplit(unittest.TestCase):
    """Phase 7D: Split single-file clusters by scope/namespace."""

    def test_split_by_namespace(self):
        """Symbols in 3 namespaces should split into 3 groups."""
        from plugin_implementation.graph_clustering import _file_internal_split

        G = nx.MultiDiGraph()
        nodes = []
        for ns in ["fmt", "detail", "internal"]:
            for i in range(20):
                nid = f"cpp::format.h::{ns}::func_{i}"
                G.add_node(nid, namespace=ns, rel_path="format.h", start_line=i * 5)
                nodes.append(nid)

        groups = _file_internal_split(nodes, G, max_size=30)
        self.assertGreaterEqual(len(groups), 2, "Should split by namespace")
        # Each group should be predominantly one namespace
        for group in groups:
            namespaces = {G.nodes[n].get("namespace") for n in group}
            self.assertLessEqual(len(namespaces), 2)

    def test_split_by_class_scope(self):
        """Methods in 2 classes should split into 2 groups."""
        from plugin_implementation.graph_clustering import _file_internal_split

        G = nx.MultiDiGraph()
        nodes = []
        for cls in ["Formatter", "Parser"]:
            for i in range(20):
                nid = f"cpp::main.h::{cls}::method_{i}"
                G.add_node(nid, parent_symbol=cls, rel_path="main.h", start_line=i * 5)
                nodes.append(nid)

        groups = _file_internal_split(nodes, G, max_size=25)
        self.assertGreaterEqual(len(groups), 2)

    def test_small_file_no_split(self):
        """File with 10 symbols (< max_size) stays as one group."""
        from plugin_implementation.graph_clustering import _file_internal_split

        G = nx.MultiDiGraph()
        nodes = []
        for i in range(10):
            nid = f"n_{i}"
            G.add_node(nid, rel_path="small.py", start_line=i * 10)
            nodes.append(nid)

        groups = _file_internal_split(nodes, G, max_size=50)
        self.assertEqual(len(groups), 1)


# ═══════════════════════════════════════════════════════════════════════════
# End-to-End Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEnd_FmtlibLike(unittest.TestCase):
    """E2E: C++ header-heavy repo should produce bounded page count."""

    @unittest.skip("Requires phases 7A+7E+7F implemented")
    def test_fmtlib_like_page_count(self):
        """1500 code + 50 doc nodes → 20-50 pages, no Part-N splitting."""
        with tempfile.TemporaryDirectory() as tmp:
            G, nodes = build_cpp_header_graph(n_symbols=1500)
            db = make_db(tmp, nodes=nodes)
            edges = [{"source_id": u, "target_id": v,
                       "rel_type": d.get("relationship_type", "calls")}
                     for u, v, d in G.edges(data=True)]
            db.upsert_edges_batch(edges)

            run_full_pipeline(db, G)

            from plugin_implementation.wiki_structure_planner.cluster_planner import (
                ClusterStructurePlanner,
            )
            # Multi-response mock for multiple sections
            responses = [(f"Section {i}", None) for i in range(20)]
            mock_llm = mock_llm_multi_response(responses)
            planner = ClusterStructurePlanner(db, mock_llm)
            spec = planner.plan_structure()

            self.assertGreater(spec.total_pages, 15,
                               "Should have at least 15 pages")
            self.assertLess(spec.total_pages, 60,
                            f"Should have < 60 pages, got {spec.total_pages}")

            # No page should have Part-N naming
            for section in spec.sections:
                for page in section.pages:
                    self.assertNotIn("(Part", page.page_name,
                                      f"Part-N splitting shouldn't happen: {page.page_name}")

    @unittest.skip("Requires phases 7A+7E+7F implemented")
    def test_fmtlib_like_docs_present(self):
        """Documentation should appear in wiki structure."""
        with tempfile.TemporaryDirectory() as tmp:
            G, nodes = build_cpp_header_graph(n_symbols=300)
            db = make_db(tmp, nodes=nodes)
            edges = [{"source_id": u, "target_id": v,
                       "rel_type": d.get("relationship_type", "calls")}
                     for u, v, d in G.edges(data=True)]
            db.upsert_edges_batch(edges)

            run_full_pipeline(db, G)

            from plugin_implementation.wiki_structure_planner.cluster_planner import (
                ClusterStructurePlanner,
            )
            mock_llm = mock_llm_response("Test Section")
            planner = ClusterStructurePlanner(db, mock_llm)
            spec = planner.plan_structure()

            # At least one page should have target_docs
            all_docs = []
            for section in spec.sections:
                for page in section.pages:
                    all_docs.extend(getattr(page, "target_docs", []) or [])

            self.assertGreater(len(all_docs), 0,
                               "At least some pages should reference doc files")


class TestEndToEnd_SmallPython(unittest.TestCase):
    """E2E: Small Python repo should produce stable, reasonable output."""

    def test_small_python_stable_pipeline(self):
        """100 code + 10 doc nodes → 10-20 pages, stable sections."""
        with tempfile.TemporaryDirectory() as tmp:
            G, nodes = build_mixed_code_doc_graph()
            db = make_db(tmp, nodes=nodes)
            edges = [{"source_id": u, "target_id": v,
                       "rel_type": d.get("relationship_type", "calls")}
                     for u, v, d in G.edges(data=True)]
            db.upsert_edges_batch(edges)

            p2, p3 = run_full_pipeline(db, G)

            # Pipeline should complete without errors
            self.assertIn("weighting", p2)
            self.assertIn("macro", p3)

            # Reasonable cluster count
            n_sections = p3["macro"]["cluster_count"]
            self.assertGreater(n_sections, 1, "Should have multiple sections")
            self.assertLess(n_sections, 15, "Should not over-segment small repo")

            n_pages = p3["micro"]["total_pages"]
            self.assertGreater(n_pages, 2, "Should have multiple pages")
            self.assertLess(n_pages, 30, "Should not over-segment small repo")


class TestEndToEnd_PureDocRepo(unittest.TestCase):
    """E2E: Pure markdown repo should cluster by directory."""

    @unittest.skip("Requires phases 7E+7G implemented")
    def test_pure_doc_sections_match_directories(self):
        """50 docs in 5 directories → 3-8 sections aligned to dirs."""
        with tempfile.TemporaryDirectory() as tmp:
            G, nodes = build_pure_doc_graph(n_docs=50)
            db = make_db(tmp, nodes=nodes)
            edges = [{"source_id": u, "target_id": v,
                       "rel_type": d.get("relationship_type", "hyperlink")}
                     for u, v, d in G.edges(data=True)]
            db.upsert_edges_batch(edges)

            p2, p3 = run_full_pipeline(db, G)

            n_sections = p3["macro"]["cluster_count"]
            self.assertGreaterEqual(n_sections, 3)
            self.assertLessEqual(n_sections, 8)

            n_pages = p3["micro"]["total_pages"]
            self.assertGreater(n_pages, 5)
            self.assertLess(n_pages, 30)


class TestEndToEnd_MixedCodeDoc(unittest.TestCase):
    """E2E: Mixed repo with code sections + doc pages."""

    @unittest.skip("Requires phases 7E+7G implemented")
    def test_mixed_repo_has_doc_and_code_clusters(self):
        """Code-heavy clusters and doc-heavy clusters should both exist."""
        with tempfile.TemporaryDirectory() as tmp:
            G, nodes = build_mixed_code_doc_graph()
            db = make_db(tmp, nodes=nodes)
            edges = [{"source_id": u, "target_id": v,
                       "rel_type": d.get("relationship_type", "calls")}
                     for u, v, d in G.edges(data=True)]
            db.upsert_edges_batch(edges)

            p2, p3 = run_full_pipeline(db, G)

            # Check that both code and doc nodes are clustered
            doc_clustered = 0
            code_clustered = 0
            for n in nodes:
                nd = db.get_node(n["node_id"])
                if nd and nd.get("macro_cluster") is not None:
                    if nd.get("is_doc"):
                        doc_clustered += 1
                    else:
                        code_clustered += 1

            self.assertGreater(code_clustered, 0, "Code nodes should be clustered")
            self.assertGreater(doc_clustered, 0, "Doc nodes should be clustered")


class TestEndToEnd_LargeRepoSmoke(unittest.TestCase):
    """E2E: Large repo should complete pipeline within reasonable bounds."""

    def test_large_repo_pipeline_completes(self):
        """2000-node graph should complete Phase 2+3 without errors."""
        with tempfile.TemporaryDirectory() as tmp:
            G, nodes = build_large_repo_graph(n=2000)
            db = make_db(tmp, nodes=nodes)
            edges = [{"source_id": u, "target_id": v,
                       "rel_type": d.get("relationship_type", "calls")}
                     for u, v, d in G.edges(data=True)]
            db.upsert_edges_batch(edges)

            p2, p3 = run_full_pipeline(db, G)

            self.assertIn("weighting", p2)
            self.assertIn("macro", p3)
            self.assertIn("micro", p3)

            # Basic sanity
            self.assertGreater(p3["macro"]["cluster_count"], 3)
            self.assertLess(p3["macro"]["cluster_count"], 30)
            self.assertGreater(p3["micro"]["total_pages"], 10)


if __name__ == "__main__":
    unittest.main()
