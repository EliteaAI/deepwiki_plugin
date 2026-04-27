"""
Phase 3 tests — Code-First Page Identity.

Verifies that:
1. In mixed clusters, code symbols (class, interface, function) are seeds
2. In function-only clusters, functions are seeds
3. In doc-dominant clusters (>70% docs), docs drive identity
4. type_alias is a valid seed (SUPPORTING tier)
5. Graceful fallback when no code symbols exist
"""

import pytest
from unittest.mock import MagicMock, patch

import networkx as nx

from plugin_implementation.constants import (
    PAGE_IDENTITY_SYMBOLS,
    SUPPORTING_CODE_SYMBOLS,
    DOC_CLUSTER_SYMBOLS,
    SYMBOL_TYPE_PRIORITY,
)
from plugin_implementation.wiki_structure_planner.cluster_planner import (
    ClusterStructurePlanner,
)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _make_mock_db(nodes):
    """Create a minimal mock DB for planner tests."""
    db = MagicMock()
    node_map = {n["node_id"]: n for n in nodes}

    def _get_node(nid):
        return node_map.get(nid)

    db.get_node = MagicMock(side_effect=_get_node)
    db.get_meta = MagicMock(return_value=None)
    db.conn = MagicMock()
    db.conn.execute.return_value.fetchall.return_value = []
    return db


def _make_node(nid, name, stype, is_arch=True, is_doc=False):
    return {
        "node_id": nid,
        "symbol_name": name,
        "symbol_type": stype,
        "is_architectural": is_arch,
        "is_doc": is_doc,
        "rel_path": f"src/{name.lower()}.py",
    }


def _make_planner_with_graph(db, nodes, edges=None):
    """Create a planner with a pre-built graph."""
    llm = MagicMock()
    planner = ClusterStructurePlanner(db, llm, wiki_title="Test")

    G = nx.MultiDiGraph()
    for n in nodes:
        G.add_node(n["node_id"], **n)
    for e in (edges or []):
        G.add_edge(e[0], e[1], weight=1.0, relationship_type="calls")

    planner._cluster_graph = G
    return planner


# ═════════════════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestCodeFirstIdentity:
    """Code symbols drive page identity in mixed clusters."""

    def test_mixed_cluster_classes_are_seeds(self):
        """3 classes + 5 docs → classes should be selected first."""
        nodes = [
            _make_node("c1", "AuthService", "class"),
            _make_node("c2", "AuthHandler", "class"),
            _make_node("c3", "TokenManager", "class"),
            _make_node("d1", "README", "module_doc", is_doc=True),
            _make_node("d2", "AUTH_GUIDE", "file_doc", is_doc=True),
            _make_node("d3", "SETUP", "module_doc", is_doc=True),
            _make_node("d4", "DEPLOY", "file_doc", is_doc=True),
            _make_node("d5", "API_REF", "module_doc", is_doc=True),
        ]
        # Edges: classes connected to each other (high PageRank)
        edges = [("c1", "c2"), ("c2", "c3"), ("c3", "c1")]

        db = _make_mock_db(nodes)
        planner = _make_planner_with_graph(db, nodes, edges)

        node_ids = [n["node_id"] for n in nodes]
        result = planner._select_central_node_ids(node_ids, k=3)

        # All 3 seeds should be classes
        for nid in result:
            node = db.get_node(nid)
            assert node["symbol_type"] in PAGE_IDENTITY_SYMBOLS, \
                f"Expected class/interface/function, got {node['symbol_type']} for {nid}"

    def test_function_only_cluster(self):
        """Functions-only cluster → functions are seeds."""
        nodes = [
            _make_node("f1", "parse_config", "function"),
            _make_node("f2", "validate_input", "function"),
            _make_node("f3", "format_output", "function"),
            _make_node("f4", "log_event", "function"),
            _make_node("f5", "send_notification", "function"),
            _make_node("f6", "retry_request", "function"),
        ]
        edges = [("f1", "f2"), ("f2", "f3"), ("f3", "f4"), ("f4", "f5")]

        db = _make_mock_db(nodes)
        planner = _make_planner_with_graph(db, nodes, edges)

        node_ids = [n["node_id"] for n in nodes]
        result = planner._select_central_node_ids(node_ids, k=3)

        for nid in result:
            node = db.get_node(nid)
            assert node["symbol_type"] == "function"

    def test_doc_dominant_cluster(self):
        """1 class + 10 docs → docs-only page."""
        nodes = [
            _make_node("c1", "SmallClass", "class"),
        ] + [
            _make_node(f"d{i}", f"Doc{i}", "module_doc", is_doc=True)
            for i in range(10)
        ]

        db = _make_mock_db(nodes)
        planner = _make_planner_with_graph(db, nodes)

        node_ids = [n["node_id"] for n in nodes]
        result = planner._select_central_node_ids(node_ids, k=5)

        # Check at least some docs are seeds (doc-dominant path)
        doc_seeds = [nid for nid in result
                     if (db.get_node(nid) or {}).get("symbol_type") in DOC_CLUSTER_SYMBOLS]
        assert len(doc_seeds) > 0, "Doc-dominant cluster should have doc seeds"

    def test_type_alias_valid_seed(self):
        """type_alias-only cluster → type_alias is valid seed."""
        nodes = [
            _make_node("t1", "UserId", "type_alias"),
            _make_node("t2", "SessionToken", "type_alias"),
            _make_node("t3", "ConfigMap", "type_alias"),
            _make_node("t4", "RequestId", "type_alias"),
            _make_node("t5", "ResponseBody", "type_alias"),
            _make_node("t6", "ErrorCode", "type_alias"),
        ]
        edges = [("t1", "t2"), ("t2", "t3"), ("t3", "t4")]

        db = _make_mock_db(nodes)
        planner = _make_planner_with_graph(db, nodes, edges)

        node_ids = [n["node_id"] for n in nodes]
        result = planner._select_central_node_ids(node_ids, k=3)

        for nid in result:
            node = db.get_node(nid)
            assert node["symbol_type"] in SUPPORTING_CODE_SYMBOLS

    def test_small_cluster_returns_all(self):
        """Cluster smaller than k → return all."""
        nodes = [
            _make_node("c1", "ClassA", "class"),
            _make_node("c2", "ClassB", "class"),
        ]

        db = _make_mock_db(nodes)
        planner = _make_planner_with_graph(db, nodes)

        node_ids = [n["node_id"] for n in nodes]
        result = planner._select_central_node_ids(node_ids, k=10)

        assert set(result) == {"c1", "c2"}

    def test_priority_order_identity_before_supporting(self):
        """When cluster has both identity + supporting, identity fills first."""
        nodes = [
            _make_node("c1", "MainClass", "class"),
            _make_node("c2", "Helper", "class"),
            _make_node("t1", "TypeA", "type_alias"),
            _make_node("t2", "TypeB", "type_alias"),
            _make_node("k1", "CONST", "constant"),
        ]
        edges = [("c1", "c2"), ("c1", "t1"), ("t1", "t2")]

        db = _make_mock_db(nodes)
        planner = _make_planner_with_graph(db, nodes, edges)

        node_ids = [n["node_id"] for n in nodes]
        result = planner._select_central_node_ids(node_ids, k=2)

        # Should pick classes (identity tier) over type_alias/constant
        for nid in result:
            node = db.get_node(nid)
            assert node["symbol_type"] in PAGE_IDENTITY_SYMBOLS, \
                f"Expected identity symbol, got {node['symbol_type']}"


class TestDocDominanceThreshold:
    """Doc-dominant detection with the 70% threshold."""

    def test_exactly_70_pct(self):
        """70% docs = doc-dominant → docs are seeds."""
        # 7 docs + 3 code = 70% docs
        nodes = [
            _make_node(f"d{i}", f"Doc{i}", "module_doc", is_doc=True)
            for i in range(7)
        ] + [
            _make_node("c1", "ClassA", "class"),
            _make_node("c2", "ClassB", "class"),
            _make_node("c3", "ClassC", "class"),
        ]

        db = _make_mock_db(nodes)
        planner = _make_planner_with_graph(db, nodes)

        node_ids = [n["node_id"] for n in nodes]
        result = planner._select_central_node_ids(node_ids, k=3)

        # At 70%, the cluster is doc-dominant
        doc_count = sum(
            1 for nid in result
            if (db.get_node(nid) or {}).get("symbol_type") in DOC_CLUSTER_SYMBOLS
        )
        assert doc_count > 0

    def test_below_70_pct_code_first(self):
        """60% docs = NOT doc-dominant → code symbols first."""
        # 6 docs + 4 code = 60% docs
        nodes = [
            _make_node(f"d{i}", f"Doc{i}", "module_doc", is_doc=True)
            for i in range(6)
        ] + [
            _make_node("c1", "ClassA", "class"),
            _make_node("c2", "ClassB", "class"),
            _make_node("c3", "ClassC", "class"),
            _make_node("c4", "ClassD", "class"),
        ]
        edges = [("c1", "c2"), ("c2", "c3"), ("c3", "c4"), ("c4", "c1")]

        db = _make_mock_db(nodes)
        planner = _make_planner_with_graph(db, nodes, edges)

        node_ids = [n["node_id"] for n in nodes]
        result = planner._select_central_node_ids(node_ids, k=3)

        # Code symbols should dominate (code-first path)
        code_count = sum(
            1 for nid in result
            if (db.get_node(nid) or {}).get("symbol_type") in PAGE_IDENTITY_SYMBOLS
        )
        assert code_count >= 2, f"Expected ≥2 code seeds, got {code_count}"
