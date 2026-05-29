"""K2 — edge provenance fields on ``RelationshipResult``.

``get_relationships`` on both query-service backends must surface the
edge's taxonomy bucket (``edge_class``), the producer's self-reported
``confidence``, and any line-precise ``via`` anchors. These let
downstream consumers (ask / research tools, the writer) cite the exact
source location and reason about edge trustworthiness rather than
treating every relationship as opaque and equal.
"""

from __future__ import annotations

import networkx as nx
import pytest

from plugin_implementation.code_graph.graph_query_service import (
    GraphQueryService,
    RelationshipResult,
    _edge_confidence,
    _edge_via,
)
from plugin_implementation.code_graph.storage_query_service import StorageQueryService
from plugin_implementation.unified_db import UnifiedWikiDB


# ─── 1. _edge_via extraction across shapes ──────────────────────────────

class TestEdgeViaExtraction:

    def test_dict_annotations_via_list(self):
        edata = {"annotations": {"via": ["route=/api/users", "callsite@L9"]}}
        assert _edge_via(edata) == ["route=/api/users", "callsite@L9"]

    def test_json_string_annotations(self):
        edata = {"annotations": '{"via": ["src=email@L42"]}'}
        assert _edge_via(edata) == ["src=email@L42"]

    def test_top_level_via_scalar(self):
        assert _edge_via({"via": "route=/api/x"}) == ["route=/api/x"]

    def test_no_via_returns_empty(self):
        assert _edge_via({}) == []
        assert _edge_via({"annotations": "not-json {["}) == []
        assert _edge_via({"annotations": {"other": 1}}) == []

    def test_blank_entries_dropped(self):
        edata = {"annotations": {"via": ["  ", "route=/x", ""]}}
        assert _edge_via(edata) == ["route=/x"]


class TestEdgeConfidence:

    def test_top_level_confidence_preferred(self):
        assert _edge_confidence({"confidence": "high"}) == "high"

    def test_falls_back_to_annotations(self):
        assert _edge_confidence({"annotations": {"confidence": "low"}}) == "low"

    def test_json_string_annotations(self):
        assert _edge_confidence({"annotations": '{"confidence": "med"}'}) == "med"

    def test_absent_returns_empty(self):
        assert _edge_confidence({}) == ""
        assert _edge_confidence({"annotations": {"via": ["x"]}}) == ""


# ─── 2. RelationshipResult dataclass defaults ───────────────────────────

def test_relationship_result_provenance_defaults():
    r = RelationshipResult(source_name="a", target_name="b", relationship_type="calls")
    assert r.edge_class == ""
    assert r.confidence == ""
    assert r.via == []


# ─── 3. NetworkX-backed GraphQueryService ───────────────────────────────

def _nx_graph_with_provenance() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node("svc", symbol_name="OrderService", symbol_type="class")
    g.add_node("ct", symbol_name="POST /api/orders", symbol_type="contract")
    # In-memory index hook used by GraphQueryService.
    g._node_index = {}
    g.add_edge(
        "svc", "ct",
        relationship_type="defines",
        edge_class="structural",
        confidence="high",
        annotations={"via": ["route=/api/orders"]},
    )
    return g


def test_graph_query_service_surfaces_provenance():
    g = _nx_graph_with_provenance()
    svc = GraphQueryService(g)
    rels = svc.get_relationships("svc", direction="outgoing", max_depth=1)
    defines = [r for r in rels if r.relationship_type == "defines"]
    assert defines, "defines edge missing"
    r = defines[0]
    assert r.edge_class == "structural"
    assert r.confidence == "high"
    assert r.via == ["route=/api/orders"]


# ─── 4. UnifiedWikiDB-backed StorageQueryService ────────────────────────

@pytest.fixture
def provenance_db(tmp_path):
    db = UnifiedWikiDB(str(tmp_path / "prov.wiki.db"), embedding_dim=4)
    db._upsert_nodes_batch([
        {
            "node_id": "svc", "rel_path": "src/svc.py", "file_name": "svc.py",
            "language": "python", "symbol_name": "OrderService",
            "symbol_type": "class", "source_text": "class OrderService: ...",
            "is_architectural": 1, "is_doc": 0,
        },
        {
            "node_id": "ct", "rel_path": "src/svc.py", "file_name": "svc.py",
            "language": "python", "symbol_name": "POST /api/orders",
            "symbol_type": "contract", "source_text": "",
            "signature": "rest_route", "is_architectural": 1, "is_doc": 0,
        },
    ])
    db.upsert_edge(
        "svc", "ct", "defines",
        edge_class="structural",
        annotations='{"via": ["route=/api/orders"], "confidence": "high"}',
    )
    db.conn.commit()
    yield db
    db.close()


def test_storage_query_service_surfaces_provenance(provenance_db):
    svc = StorageQueryService(provenance_db)
    rels = svc.get_relationships("svc", direction="outgoing", max_depth=1)
    defines = [r for r in rels if r.relationship_type == "defines"]
    assert defines, "defines edge missing"
    r = defines[0]
    assert r.edge_class == "structural"
    assert r.confidence == "high"
    assert r.via == ["route=/api/orders"]
