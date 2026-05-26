"""Tests for Phase B2+B3+B5: contract nodes, CONSUMES edge, api_surface promotion.

Pins:
- SymbolType.CONTRACT exists and is architectural;
- RelationshipType.CONSUMES exists;
- materialize_contract_nodes creates stable-id nodes with DEFINES edges;
- dedup semantics (same (kind, surface) → single contract node);
- cross_language_linker L1 discovers contract nodes and emits cross-lang edges;
- writer expansion pool includes contract nodes via 'structural' edge class;
- contract_kind is stored in `signature` attribute (flows through _nx_node_to_dict).
"""

from __future__ import annotations

import math
from unittest.mock import patch

import networkx as nx
import pytest

from plugin_implementation.code_graph.api_surface_extractor import (
    APISurface,
    _match_grpc,
    materialize_contract_nodes,
    _contract_node_id,
)
from plugin_implementation.code_graph.cross_language_linker import (
    link_l1_api_surface,
    _link_l1_via_contract_nodes,
)
from plugin_implementation.constants import (
    ARCHITECTURAL_SYMBOLS,
    EXPANSION_SYMBOL_TYPES,
    SYMBOL_TYPE_PRIORITY,
    WRITER_ALLOWED_EDGE_CLASSES,
)
from plugin_implementation.parsers.base_parser import RelationshipType, SymbolType


# ─── 1. Enum existence ──────────────────────────────────────────────────

class TestEnumAdditions:

    def test_contract_symbol_type_exists(self):
        assert SymbolType.CONTRACT.value == "contract"

    def test_consumes_relationship_type_exists(self):
        assert RelationshipType.CONSUMES.value == "consumes"

    def test_contract_is_architectural(self):
        assert "contract" in ARCHITECTURAL_SYMBOLS

    def test_contract_in_expansion_types(self):
        assert "contract" in EXPANSION_SYMBOL_TYPES

    def test_contract_has_priority(self):
        assert "contract" in SYMBOL_TYPE_PRIORITY
        assert SYMBOL_TYPE_PRIORITY["contract"] > SYMBOL_TYPE_PRIORITY["method"]


# ─── 2. Node ID stability ───────────────────────────────────────────────

class TestContractNodeId:

    def test_deterministic(self):
        a = _contract_node_id("rest_route", "POST /api/users")
        b = _contract_node_id("rest_route", "POST /api/users")
        assert a == b

    def test_different_surfaces_produce_different_ids(self):
        a = _contract_node_id("rest_route", "POST /api/users")
        b = _contract_node_id("rest_route", "GET /api/users")
        assert a != b

    def test_different_kinds_produce_different_ids(self):
        a = _contract_node_id("rest_route", "POST /api/users")
        b = _contract_node_id("grpc_service", "POST /api/users")
        assert a != b

    def test_format(self):
        nid = _contract_node_id("rest_route", "GET /health")
        assert nid == "contract::rest_route::GET /health"


# ─── 3. materialize_contract_nodes ──────────────────────────────────────

def _make_graph_with_surfaces():
    """Build a graph with two nodes that expose overlapping surfaces."""
    g = nx.MultiDiGraph()
    g.add_node("py_handler", symbol_name="UserHandler", symbol_type="class",
               rel_path="src/handlers.py", file_name="handlers.py",
               language="python", start_line=10, end_line=50,
               source_text="class UserHandler: ...")
    g.add_node("ts_client", symbol_name="UserClient", symbol_type="class",
               rel_path="src/client.ts", file_name="client.ts",
               language="typescript", start_line=1, end_line=30,
               source_text="class UserClient { ... }")
    return g


class TestMaterializeContractNodes:

    def test_creates_contract_node(self):
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [APISurface(
                kind="rest_route", surface="POST /api/users",
                weight_hint=0.8, metadata={"method": "POST", "path": "/api/users"},
            )],
        }
        added = materialize_contract_nodes(g, surfaces)
        assert added == 1
        nid = _contract_node_id("rest_route", "POST /api/users")
        assert g.has_node(nid)
        data = g.nodes[nid]
        assert data["symbol_type"] == "contract"
        assert data["symbol_name"] == "POST /api/users"
        assert data["signature"] == "rest_route"
        assert data["is_architectural"] is True

    def test_creates_defines_edge(self):
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [APISurface(
                kind="rest_route", surface="POST /api/users",
                weight_hint=0.8, metadata={"method": "POST", "path": "/api/users"},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("rest_route", "POST /api/users")
        edges = list(g.edges("py_handler", data=True))
        defines_edges = [
            (u, v, d) for u, v, d in edges
            if v == nid and d.get("relationship_type") == "defines"
        ]
        assert len(defines_edges) == 1
        _, _, d = defines_edges[0]
        assert d["edge_class"] == "structural"

    def test_via_annotations_on_defines_edge(self):
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [APISurface(
                kind="rest_route", surface="POST /api/users",
                weight_hint=0.8, metadata={"method": "POST", "path": "/api/users"},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("rest_route", "POST /api/users")
        edges = list(g.edges("py_handler", data=True))
        defines_edges = [(u, v, d) for u, v, d in edges if v == nid]
        _, _, d = defines_edges[0]
        via = d.get("annotations", {}).get("via", [])
        assert "dispatch=POST" in via
        assert "route=/api/users" in via

    def test_dedup_same_surface(self):
        """Two nodes exposing the same surface share one contract node."""
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [APISurface(
                kind="rest_route", surface="POST /api/users",
                weight_hint=0.8, metadata={},
            )],
            "ts_client": [APISurface(
                kind="rest_route", surface="POST /api/users",
                weight_hint=0.6, metadata={},
            )],
        }
        added = materialize_contract_nodes(g, surfaces)
        assert added == 1  # only one contract node created
        nid = _contract_node_id("rest_route", "POST /api/users")
        # But two defines edges (one from each owner).
        preds = list(g.predecessors(nid))
        assert set(preds) == {"py_handler", "ts_client"}

    def test_multiple_surfaces_per_node(self):
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [
                APISurface(kind="rest_route", surface="POST /api/users",
                           weight_hint=0.8, metadata={}),
                APISurface(kind="rest_route", surface="GET /api/users",
                           weight_hint=0.7, metadata={}),
            ],
        }
        added = materialize_contract_nodes(g, surfaces)
        assert added == 2

    def test_missing_owner_node_skipped(self):
        g = nx.MultiDiGraph()
        surfaces = {
            "nonexistent": [APISurface(
                kind="rest_route", surface="GET /x",
                weight_hint=0.5, metadata={},
            )],
        }
        added = materialize_contract_nodes(g, surfaces)
        assert added == 0

    def test_contract_node_inherits_owner_file_info(self):
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [APISurface(
                kind="rest_route", surface="DELETE /api/users/{id}",
                weight_hint=0.8, metadata={},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("rest_route", "DELETE /api/users/{id}")
        data = g.nodes[nid]
        assert data["rel_path"] == "src/handlers.py"
        assert data["language"] == "python"


# ─── 4. Cross-language linker L1 via contract nodes ─────────────────────

class TestL1ViaContractNodes:

    def _setup_graph_with_contracts(self):
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [APISurface(
                kind="rest_route", surface="POST /api/users",
                weight_hint=0.8, metadata={},
            )],
            "ts_client": [APISurface(
                kind="rest_route", surface="POST /api/users",
                weight_hint=0.6, metadata={},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        return g

    def test_l1_discovers_cross_language_pair(self):
        g = self._setup_graph_with_contracts()
        edges = link_l1_api_surface(g)
        assert len(edges) == 1
        src, tgt, attrs = edges[0]
        assert {src, tgt} == {"py_handler", "ts_client"}
        assert attrs["relationship_type"] == "cross_language_L1"
        assert attrs["edge_class"] == "cross_language"

    def test_l1_via_annotation_references_contract(self):
        g = self._setup_graph_with_contracts()
        edges = link_l1_api_surface(g)
        _, _, attrs = edges[0]
        via = attrs.get("annotations", {}).get("via", [])
        contract_id = _contract_node_id("rest_route", "POST /api/users")
        assert f"contract={contract_id}" in via

    def test_l1_same_language_no_edge(self):
        """Two nodes in the same language should NOT get a cross-language edge."""
        g = nx.MultiDiGraph()
        g.add_node("py_a", symbol_type="class", language="python",
                   rel_path="a.py", file_name="a.py")
        g.add_node("py_b", symbol_type="class", language="python",
                   rel_path="b.py", file_name="b.py")
        surfaces = {
            "py_a": [APISurface(kind="rest_route", surface="GET /x",
                                weight_hint=0.5, metadata={})],
            "py_b": [APISurface(kind="rest_route", surface="GET /x",
                                weight_hint=0.5, metadata={})],
        }
        materialize_contract_nodes(g, surfaces)
        edges = link_l1_api_surface(g)
        assert edges == []

    def test_l1_falls_back_to_legacy_when_no_contracts(self):
        """Without contract nodes, legacy dict-based path activates."""
        g = _make_graph_with_surfaces()
        # No materialize_contract_nodes called → no contract nodes.
        surfaces_by_node = {
            "py_handler": [APISurface(
                kind="rest_route", surface="POST /api/users",
                weight_hint=0.8, metadata={},
            )],
            "ts_client": [APISurface(
                kind="rest_route", surface="POST /api/users",
                weight_hint=0.6, metadata={},
            )],
        }
        edges = link_l1_api_surface(g, surfaces_by_node)
        assert len(edges) == 1
        src, tgt, _ = edges[0]
        assert {src, tgt} == {"py_handler", "ts_client"}

    def test_l1_specificity_decreases_with_more_implementors(self):
        """More implementors → lower specificity → lower weight."""
        g = nx.MultiDiGraph()
        for lang in ("python", "typescript", "java", "go"):
            nid = f"{lang}_node"
            g.add_node(nid, symbol_type="class", language=lang,
                       rel_path=f"src/{lang}.ext", file_name=f"{lang}.ext")

        surfaces = {
            f"{lang}_node": [APISurface(
                kind="rest_route", surface="GET /shared",
                weight_hint=0.5, metadata={},
            )]
            for lang in ("python", "typescript", "java", "go")
        }
        materialize_contract_nodes(g, surfaces)
        many_edges = link_l1_api_surface(g)

        # Now a pair-only scenario for comparison.
        g2 = nx.MultiDiGraph()
        g2.add_node("a", symbol_type="class", language="python",
                    rel_path="a.py", file_name="a.py")
        g2.add_node("b", symbol_type="class", language="typescript",
                    rel_path="b.ts", file_name="b.ts")
        surfaces2 = {
            "a": [APISurface(kind="rest_route", surface="GET /unique",
                             weight_hint=0.5, metadata={})],
            "b": [APISurface(kind="rest_route", surface="GET /unique",
                             weight_hint=0.5, metadata={})],
        }
        materialize_contract_nodes(g2, surfaces2)
        few_edges = link_l1_api_surface(g2)

        max_many_weight = max(e[2]["weight"] for e in many_edges)
        pair_weight = few_edges[0][2]["weight"]
        assert pair_weight > max_many_weight


# ─── 5. Writer edge-class compatibility ─────────────────────────────────

class TestWriterCompatibility:

    def test_defines_edges_use_allowed_class(self):
        """DEFINES edges from materialize use 'structural' → writer can walk them."""
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [APISurface(
                kind="rest_route", surface="GET /api/health",
                weight_hint=0.5, metadata={},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("rest_route", "GET /api/health")
        for _, _, d in g.edges("py_handler", data=True):
            if d.get("relationship_type") == "defines":
                assert d["edge_class"] in WRITER_ALLOWED_EDGE_CLASSES


# ─── 6. Persistence round-trip (signature stores contract_kind) ─────────

class TestPersistenceRoundTrip:

    def test_signature_stores_contract_kind(self):
        """The signature field on _nx_node_to_dict flows contract_kind."""
        from plugin_implementation.unified_db import UnifiedWikiDB
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [APISurface(
                kind="grpc_service", surface="grpc:UserService/GetUser",
                weight_hint=0.9, metadata={},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("grpc_service", "grpc:UserService/GetUser")

        db = UnifiedWikiDB.__new__(UnifiedWikiDB)
        result = db._nx_node_to_dict(nid, g.nodes[nid])
        assert result["signature"] == "grpc_service"
        assert result["symbol_type"] == "contract"
        assert result["symbol_name"] == "grpc:UserService/GetUser"


# ─── 7. Proto gRPC cross-product regression ───────────────────────────

class TestProtoNoCrossProduct:

    def test_multi_service_proto_no_cross_product(self):
        """Multiple services in one proto file must NOT cross-multiply RPCs."""
        proto_text = (
            "service CartService {\n"
            "  rpc AddItem (AddItemReq) returns (Empty) {}\n"
            "  rpc GetCart (GetCartReq) returns (Cart) {}\n"
            "}\n\n"
            "service AdService {\n"
            "  rpc GetAds (AdReq) returns (AdResp) {}\n"
            "}\n"
        )
        surfaces = _match_grpc(proto_text, "schema")
        surface_keys = {s["surface"] for s in surfaces}
        assert surface_keys == {
            "grpc:CartService/AddItem",
            "grpc:CartService/GetCart",
            "grpc:AdService/GetAds",
        }

    def test_single_service_proto_still_works(self):
        proto_text = (
            "service Greeter {\n"
            "  rpc SayHello (HelloReq) returns (HelloResp) {}\n"
            "}\n"
        )
        surfaces = _match_grpc(proto_text, "proto")
        assert len(surfaces) == 1
        assert surfaces[0]["surface"] == "grpc:Greeter/SayHello"
