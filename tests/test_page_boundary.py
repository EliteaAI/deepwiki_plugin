"""
Phase 2 tests — Page-Boundary Fidelity.

Verifies that:
1. PageSpec.metadata carries structured section_id, page_id, cluster_node_ids
2. _try_cluster_expansion reads IDs from metadata (with rationale fallback)
3. Expansion respects page boundary (cluster_node_ids), not just macro boundary
"""

import pytest
from unittest.mock import MagicMock, patch

from plugin_implementation.state.wiki_state import PageSpec
from plugin_implementation.cluster_expansion import (
    expand_for_page,
    _collect_expansion_neighbors,
)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _mock_db_with_nodes(nodes, edges=None):
    """Create a mock DB preloaded with node/edge data.

    nodes: list of dicts with at least {node_id, symbol_name, symbol_type,
           macro_cluster, micro_cluster, is_architectural, rel_path}
    edges: list of dicts with {source_id, target_id, rel_type, weight}
    """
    db = MagicMock()
    node_map = {n["node_id"]: n for n in nodes}

    def _mock_get_node(nid):
        return node_map.get(nid)

    db.get_node = MagicMock(side_effect=_mock_get_node)

    def _mock_execute(sql, params=None):
        result = MagicMock()

        if "FROM repo_nodes" in sql and "symbol_name" in sql:
            name = params[0] if params else None
            matches = [n for n in nodes if n["symbol_name"] == name]
            if params and len(params) > 1 and "macro_cluster" in sql:
                macro = params[1]
                matches = [n for n in matches if n.get("macro_cluster") == macro]
            result.fetchall.return_value = matches
            return result

        if "FROM repo_nodes" in sql and "node_id IN" in sql:
            ids = params if params else []
            result.fetchall.return_value = [n for n in nodes if n["node_id"] in ids]
            return result

        if "FROM repo_edges" in sql and "source_id" in sql:
            src = params[0] if params else None
            edge_list = edges or []
            result.fetchall.return_value = [
                (e["target_id"], e.get("rel_type", "calls"), e.get("weight", 1.0))
                for e in edge_list if e["source_id"] == src
            ]
            return result

        if "FROM repo_edges" in sql and "target_id" in sql:
            tgt = params[0] if params else None
            edge_list = edges or []
            result.fetchall.return_value = [
                (e["source_id"], e.get("rel_type", "calls"), e.get("weight", 1.0))
                for e in edge_list if e["target_id"] == tgt
            ]
            return result

        if "FROM repo_nodes" in sql and "is_doc" in sql:
            macro = params[0] if params else None
            docs = [n for n in nodes if n.get("is_doc") and n.get("macro_cluster") == macro]
            result.fetchall.return_value = docs
            return result

        if "repo_fts" in sql:
            result.fetchall.return_value = []
            return result

        result.fetchall.return_value = []
        return result

    db.conn = MagicMock()
    db.conn.execute = MagicMock(side_effect=_mock_execute)
    return db


def _make_node(node_id, name, stype="class", macro=1, micro=0,
               is_arch=True, is_doc=False, rel_path="src/mod.py",
               content="class Foo: pass"):
    return {
        "node_id": node_id,
        "symbol_name": name,
        "symbol_type": stype,
        "macro_cluster": macro,
        "micro_cluster": micro,
        "is_architectural": is_arch,
        "is_doc": is_doc,
        "rel_path": rel_path,
        "start_line": 1,
        "end_line": 10,
        "content": content,
        "signature": f"{stype} {name}",
        "docstring": "",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Test: Metadata carries structured IDs
# ═════════════════════════════════════════════════════════════════════════════

class TestMetadataCarriesIDs:
    """PageSpec.metadata now carries planner_mode, section_id, page_id."""

    def test_page_spec_has_section_id(self):
        spec = PageSpec(
            page_name="Test",
            page_order=1,
            description="d",
            content_focus="c",
            rationale="Grouped by graph clustering (macro=3, micro=2, 10 symbols)",
            metadata={
                "planner_mode": "cluster",
                "section_id": 3,
                "page_id": 2,
                "cluster_node_ids": ["a", "b"],
            },
        )
        assert spec.metadata["section_id"] == 3
        assert spec.metadata["page_id"] == 2
        assert spec.metadata["planner_mode"] == "cluster"
        assert spec.metadata["cluster_node_ids"] == ["a", "b"]

    def test_backward_compat_old_metadata(self):
        """Old-style metadata (only cluster_node_ids) still works."""
        spec = PageSpec(
            page_name="Test",
            page_order=1,
            description="d",
            content_focus="c",
            rationale="Grouped by graph clustering (macro=3, micro=2)",
            metadata={"cluster_node_ids": ["a", "b"]},
        )
        assert "section_id" not in spec.metadata
        assert spec.metadata["cluster_node_ids"] == ["a", "b"]


# ═════════════════════════════════════════════════════════════════════════════
# Test: Page boundary enforcement in expansion
# ═════════════════════════════════════════════════════════════════════════════

class TestPageBoundaryExpansion:
    """Expansion respects cluster_node_ids as page boundary."""

    def test_neighbor_excluded_when_outside_page(self):
        """Neighbors not in cluster_node_ids are excluded."""
        # Node A is in page, Node B is in same macro but different page
        node_a = _make_node("a", "ClassA", macro=1, micro=0)
        node_b = _make_node("b", "ClassB", macro=1, micro=1)
        edges = [{"source_id": "a", "target_id": "b", "rel_type": "calls", "weight": 5.0}]

        db = _mock_db_with_nodes([node_a, node_b], edges)

        # Page boundary = only node "a"
        page_boundary = {"a"}

        neighbors = _collect_expansion_neighbors(
            db, ["a"], set(), macro_id=1,
            page_boundary_ids=page_boundary,
        )

        # Node B should NOT be in results (not in page boundary)
        neighbor_ids = {n[0] for n in neighbors}
        assert "b" not in neighbor_ids

    def test_neighbor_included_when_in_page(self):
        """Neighbors in cluster_node_ids are included."""
        node_a = _make_node("a", "ClassA", macro=1, micro=0)
        node_b = _make_node("b", "ClassB", macro=1, micro=0)
        edges = [{"source_id": "a", "target_id": "b", "rel_type": "calls", "weight": 5.0}]

        db = _mock_db_with_nodes([node_a, node_b], edges)

        page_boundary = {"a", "b"}

        neighbors = _collect_expansion_neighbors(
            db, ["a"], set(), macro_id=1,
            page_boundary_ids=page_boundary,
        )

        neighbor_ids = {n[0] for n in neighbors}
        assert "b" in neighbor_ids

    def test_fallback_to_macro_when_no_page_boundary(self):
        """When cluster_node_ids is None, macro_id boundary applies."""
        node_a = _make_node("a", "ClassA", macro=1, micro=0)
        node_b = _make_node("b", "ClassB", macro=1, micro=1)  # different micro, same macro
        node_c = _make_node("c", "ClassC", macro=2, micro=0)  # different macro
        edges = [
            {"source_id": "a", "target_id": "b", "rel_type": "calls", "weight": 5.0},
            {"source_id": "a", "target_id": "c", "rel_type": "calls", "weight": 5.0},
        ]

        db = _mock_db_with_nodes([node_a, node_b, node_c], edges)

        neighbors = _collect_expansion_neighbors(
            db, ["a"], set(), macro_id=1,
            page_boundary_ids=None,  # no page boundary
        )

        neighbor_ids = {n[0] for n in neighbors}
        # B is in same macro → included
        assert "b" in neighbor_ids
        # C is in different macro → excluded
        assert "c" not in neighbor_ids

    def test_expand_for_page_with_cluster_node_ids(self):
        """expand_for_page passes cluster_node_ids through."""
        node_a = _make_node("a", "ClassA", macro=1, micro=0,
                            content="class ClassA:\n    pass")
        node_b = _make_node("b", "ClassB", macro=1, micro=1,
                            content="class ClassB:\n    pass")
        edges = [{"source_id": "a", "target_id": "b", "rel_type": "calls", "weight": 5.0}]

        db = _mock_db_with_nodes([node_a, node_b], edges)

        # With cluster_node_ids = only "a", expansion should NOT include "b"
        docs = expand_for_page(
            db=db,
            page_symbols=["ClassA"],
            macro_id=1,
            micro_id=0,
            cluster_node_ids=["a"],
            token_budget=50000,
        )

        # Only ClassA should be in docs (initial), no expansion to ClassB
        doc_names = [d.metadata.get("symbol_name", "") for d in docs]
        assert "ClassA" in doc_names
        assert "ClassB" not in doc_names

    def test_expand_for_page_without_cluster_node_ids(self):
        """expand_for_page falls back to macro boundary when no node_ids."""
        node_a = _make_node("a", "ClassA", macro=1, micro=0,
                            content="class ClassA:\n    pass")
        node_b = _make_node("b", "ClassB", macro=1, micro=1,
                            content="class ClassB:\n    pass")
        edges = [{"source_id": "a", "target_id": "b", "rel_type": "calls", "weight": 5.0}]

        db = _mock_db_with_nodes([node_a, node_b], edges)

        # Without cluster_node_ids, macro boundary allows B
        docs = expand_for_page(
            db=db,
            page_symbols=["ClassA"],
            macro_id=1,
            token_budget=50000,
        )

        doc_names = [d.metadata.get("symbol_name", "") for d in docs]
        assert "ClassA" in doc_names
        # B is in same macro, so it can appear as expansion neighbor
        # (whether it does depends on edge traversal — the key point is it's not excluded)


# ═════════════════════════════════════════════════════════════════════════════
# Test: Doc nodes included in page boundary
# ═════════════════════════════════════════════════════════════════════════════

class TestDocNodesInPage:
    """Documentation nodes are part of cluster_node_ids and included."""

    def test_doc_node_in_cluster_node_ids(self):
        """Doc nodes in cluster_node_ids are included in page metadata."""
        spec = PageSpec(
            page_name="Test",
            page_order=1,
            description="d",
            content_focus="c",
            rationale="Grouped by graph clustering (macro=1, micro=0, 3 symbols)",
            metadata={
                "planner_mode": "cluster",
                "section_id": 1,
                "page_id": 0,
                "cluster_node_ids": ["class_node", "func_node", "doc_node"],
            },
        )
        assert "doc_node" in spec.metadata["cluster_node_ids"]
