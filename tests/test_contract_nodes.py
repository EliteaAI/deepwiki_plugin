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

import networkx as nx

from plugin_implementation.code_graph.api_surface_extractor import (
    APISurface,
    _match_grpc,
    _match_grpc_client,
    _match_rest_typescript,
    extract_api_surfaces,
    extract_api_surfaces_for_graph,
    extract_grpc_stub_bindings,
    materialize_contract_nodes,
    _contract_node_id,
)
from plugin_implementation.code_graph.cross_language_linker import (
    link_l1_api_surface,
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


# ─── 7. Phase B3: defines vs consumes role discrimination ───────────────

class TestConsumesRole:
    """A server route registration *defines* a contract; an outbound client
    call *consumes* it. The shared contract node couples both directions."""

    def test_client_surface_emits_consumes_edge(self):
        g = _make_graph_with_surfaces()
        surfaces = {
            "ts_client": [APISurface(
                kind="rest_route", surface="POST /api/users",
                weight_hint=0.7,
                metadata={"method": "POST", "path": "/api/users", "role": "client"},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("rest_route", "POST /api/users")
        edges = [d for _, v, d in g.edges("ts_client", data=True) if v == nid]
        assert len(edges) == 1
        assert edges[0]["relationship_type"] == "consumes"
        assert edges[0]["edge_class"] == "structural"

    def test_server_role_emits_defines_edge(self):
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [APISurface(
                kind="rest_route", surface="POST /api/users",
                weight_hint=0.8,
                metadata={"method": "POST", "path": "/api/users", "role": "server"},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("rest_route", "POST /api/users")
        edges = [d for _, v, d in g.edges("py_handler", data=True) if v == nid]
        assert edges[0]["relationship_type"] == "defines"

    def test_missing_role_defaults_to_defines(self):
        """Non-REST surfaces (gRPC/GraphQL/FFI) carry no role and must keep
        their historical ``defines`` semantics."""
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [APISurface(
                kind="grpc", surface="grpc:UserSvc/Get",
                weight_hint=0.8, metadata={"service": "UserSvc", "method": "Get"},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("grpc", "grpc:UserSvc/Get")
        edges = [d for _, v, d in g.edges("py_handler", data=True) if v == nid]
        assert edges[0]["relationship_type"] == "defines"

    def test_ts_client_matcher_tags_role_client(self):
        surfaces = _match_rest_typescript('axios.get("/api/users")')
        assert surfaces, "expected the axios client call to be matched"
        assert all(s["metadata"].get("role") == "client" for s in surfaces)

    def test_ts_fetch_matcher_tags_role_client(self):
        surfaces = _match_rest_typescript('fetch("/api/items", {method: "POST"})')
        assert surfaces
        assert all(s["metadata"].get("role") == "client" for s in surfaces)

    def test_ts_decorator_matcher_tags_role_server(self):
        surfaces = _match_rest_typescript('@Post("/api/users")\nfn() {}')
        assert surfaces
        assert all(s["metadata"].get("role") == "server" for s in surfaces)

    def test_l1_pairs_consumer_with_definer(self):
        """A Python handler defining ``POST /api/users`` and a TS client
        consuming it must be cross-language coupled through the contract."""
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [APISurface(
                kind="rest_route", surface="POST /api/users",
                weight_hint=0.8,
                metadata={"method": "POST", "path": "/api/users", "role": "server"},
            )],
            "ts_client": [APISurface(
                kind="rest_route", surface="POST /api/users",
                weight_hint=0.7,
                metadata={"method": "POST", "path": "/api/users", "role": "client"},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        edges = link_l1_api_surface(g)
        pairs = {frozenset((u, v)) for u, v, _ in edges}
        assert frozenset(("py_handler", "ts_client")) in pairs
        attrs = next(a for u, v, a in edges
                     if {u, v} == {"py_handler", "ts_client"})
        assert attrs["relationship_type"] == "cross_language_L1"
        assert attrs["edge_class"] == "cross_language"


# ─── 8. Phase B3: multi-language gRPC consumer (client) detection ────────

# Real call-site shapes taken verbatim (modulo whitespace) from
# GoogleCloudPlatform/microservices-demo so the matcher is pinned against
# production code rather than synthetic stubs.

class TestGrpcClientMatcher:
    """Unit-level coverage of ``_match_grpc_client`` per language.

    The proto-canonical RPC name (PascalCase) is what the server-side
    matcher emits, so each client surface must normalise to it — otherwise
    consumer and provider would never dedupe onto a shared contract node.
    """

    def test_go_chained_construct_and_call(self):
        # frontend/rpc.go: pb.NewProductCatalogServiceClient(conn).ListProducts(...)
        text = (
            "resp, err := pb.NewProductCatalogServiceClient(fe.productCatalogSvcConn)"
            ".ListProducts(ctx, &pb.Empty{})"
        )
        surfaces = _match_grpc_client(text, "go")
        assert {s["surface"] for s in surfaces} == {
            "grpc:ProductCatalogService/ListProducts"
        }
        assert all(s["metadata"]["role"] == "client" for s in surfaces)

    def test_go_chained_call_on_next_line(self):
        # frontend/handlers.go: constructor and RPC split across lines.
        text = (
            "order, err := pb.NewCheckoutServiceClient(fe.checkoutSvcConn).\n"
            "\t\tPlaceOrder(ctx, req)"
        )
        surfaces = _match_grpc_client(text, "go")
        assert {s["surface"] for s in surfaces} == {
            "grpc:CheckoutService/PlaceOrder"
        }

    def test_go_generated_stub_definition_not_matched(self):
        """The generated ``func New<Svc>Client`` definition and the stub's
        own method definitions must never be mistaken for consumer calls."""
        text = (
            "func NewCartServiceClient(cc grpc.ClientConnInterface) CartServiceClient {\n"
            "\treturn &cartServiceClient{cc}\n"
            "}\n"
            "func (c *cartServiceClient) AddItem(ctx context.Context) {}\n"
        )
        assert _match_grpc_client(text, "go") == []

    def test_python_bound_stub_resolves_call(self):
        # recommendationservice: module-scope bind + in-method call.
        bindings = extract_grpc_stub_bindings(
            "product_catalog_stub = demo_pb2_grpc.ProductCatalogServiceStub(channel)",
            "python",
        )
        assert bindings == {"product_catalog_stub": "ProductCatalogService"}
        call_text = "        response = product_catalog_stub.ListProducts(demo_pb2.Empty())"
        surfaces = _match_grpc_client(call_text, "python", bindings)
        assert {s["surface"] for s in surfaces} == {
            "grpc:ProductCatalogService/ListProducts"
        }
        assert surfaces[0]["metadata"]["role"] == "client"

    def test_java_lowercamel_call_normalises_to_pascal(self):
        # adservice/AdServiceClient.java: blockingStub.getAds(...)
        bindings = extract_grpc_stub_bindings(
            "blockingStub = AdServiceGrpc.newBlockingStub(channel);", "java"
        )
        assert bindings == {"blockingStub": "AdService"}
        surfaces = _match_grpc_client(
            "AdResponse response = blockingStub.getAds(request);", "java", bindings
        )
        assert {s["surface"] for s in surfaces} == {"grpc:AdService/GetAds"}

    def test_java_type_declaration_binds_stub(self):
        """The field type decl alone is enough to bind the variable."""
        bindings = extract_grpc_stub_bindings(
            "private AdServiceGrpc.AdServiceBlockingStub blockingStub;", "java"
        )
        assert bindings.get("blockingStub") == "AdService"

    def test_csharp_strips_async_suffix(self):
        # cartservice/tests: cartClient.GetCartAsync(request)
        bindings = extract_grpc_stub_bindings(
            "var cartClient = new CartServiceClient(channel);", "csharp"
        )
        assert bindings == {"cartClient": "CartService"}
        surfaces = _match_grpc_client(
            "var cart = await cartClient.GetCartAsync(request);", "csharp", bindings
        )
        assert {s["surface"] for s in surfaces} == {"grpc:CartService/GetCart"}

    def test_unbound_call_site_is_ignored(self):
        """A method call on a variable with no stub binding must not emit."""
        assert _match_grpc_client("logger.Info(\"hi\")", "go") == []
        assert _match_grpc_client("self.helper.compute(x)", "python") == []


def _grpc_server_client_graph(server_lang, server_text, client_lang, client_text):
    """Two-node graph: a gRPC provider + a consumer in another language."""
    g = nx.MultiDiGraph()
    g.add_node("server", symbol_name="ServiceImpl", symbol_type="class",
               rel_path=f"server.{server_lang}", file_name=f"server.{server_lang}",
               language=server_lang, start_line=1, end_line=40,
               source_text=server_text)
    g.add_node("client", symbol_name="ClientCaller", symbol_type="function",
               rel_path=f"client.{client_lang}", file_name=f"client.{client_lang}",
               language=client_lang, start_line=1, end_line=40,
               source_text=client_text)
    return g


class TestGrpcConsumesIntegration:
    """End-to-end: orchestrator extraction → contract materialization, with a
    provider and a consumer landing on the *same* contract node via
    ``defines`` and ``consumes`` edges respectively."""

    def _assert_defines_and_consumes(self, g, surface):
        surfaces = extract_api_surfaces_for_graph(g)
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("grpc", surface)
        assert g.has_node(nid), f"missing contract node for {surface}"
        # Provider → defines
        server_edges = [d for _, v, d in g.edges("server", data=True) if v == nid]
        assert server_edges, "provider produced no edge into the contract"
        assert any(d["relationship_type"] == "defines" for d in server_edges)
        # Consumer → consumes
        client_edges = [d for _, v, d in g.edges("client", data=True) if v == nid]
        assert client_edges, "consumer produced no edge into the contract"
        assert any(d["relationship_type"] == "consumes" for d in client_edges)
        return nid

    def test_go_consumer_links_to_go_provider(self):
        server = (
            "type CartServiceServer interface {\n"
            "    AddItem(context.Context, *AddItemRequest) (*Empty, error)\n"
            "}\n"
        )
        client = (
            "func (fe *frontendServer) addToCart(ctx context.Context) error {\n"
            "    _, err := pb.NewCartServiceClient(fe.cartSvcConn).AddItem(ctx, req)\n"
            "    return err\n"
            "}\n"
        )
        g = _grpc_server_client_graph("go", server, "go", client)
        self._assert_defines_and_consumes(g, "grpc:CartService/AddItem")

    def test_java_consumer_links_to_java_provider(self):
        server = (
            "public class AdServiceImpl extends AdServiceGrpc.AdServiceImplBase {\n"
            "    @Override public void getAds(AdRequest r, StreamObserver<AdResponse> o) {}\n"
            "}\n"
        )
        client = (
            "AdServiceGrpc.AdServiceBlockingStub blockingStub = "
            "AdServiceGrpc.newBlockingStub(channel);\n"
            "AdResponse response = blockingStub.getAds(request);\n"
        )
        g = _grpc_server_client_graph("java", server, "java", client)
        self._assert_defines_and_consumes(g, "grpc:AdService/GetAds")

    def test_csharp_consumer_links_to_csharp_provider(self):
        server = (
            "public class CartServiceImpl : CartService.CartServiceBase {\n"
            "    public override Task<Cart> GetCart(GetCartRequest r, "
            "ServerCallContext c) {}\n"
            "}\n"
        )
        client = (
            "var cartClient = new CartServiceClient(channel);\n"
            "var cart = await cartClient.GetCartAsync(request);\n"
        )
        g = _grpc_server_client_graph("csharp", server, "csharp", client)
        self._assert_defines_and_consumes(g, "grpc:CartService/GetCart")

    def test_python_cross_slice_consumer(self, tmp_path):
        """The Python binding lives in a different symbol/slice than the call,
        so the orchestrator must read the *whole file* via ``repo_root`` to
        resolve the stub variable. Mirrors recommendationservice."""
        src = (
            "import demo_pb2\n"
            "import demo_pb2_grpc\n"
            "\n"
            "channel = grpc.insecure_channel('localhost:3550')\n"
            "product_catalog_stub = demo_pb2_grpc.ProductCatalogServiceStub(channel)\n"
            "\n"
            "class RecommendationService(demo_pb2_grpc.RecommendationServiceServicer):\n"
            "    def ListRecommendations(self, request, context):\n"
            "        response = product_catalog_stub.ListProducts(demo_pb2.Empty())\n"
            "        return response\n"
        )
        repo_root = tmp_path
        (repo_root / "recommendation_server.py").write_text(src, encoding="utf-8")

        g = nx.MultiDiGraph()
        # Provider node: the gRPC servicer (a different proto service).
        g.add_node(
            "provider", symbol_name="ProductCatalogServiceServicer",
            symbol_type="class", rel_path="product_catalog_server.py",
            file_name="product_catalog_server.py", language="python",
            start_line=1, end_line=20,
            source_text=(
                "class ProductCatalogServiceServicer(object):\n"
                "    def ListProducts(self, request, context):\n"
                "        pass\n"
            ),
        )
        # Consumer node: ONLY the method body (binding is in another slice).
        g.add_node(
            "consumer", symbol_name="ListRecommendations", symbol_type="method",
            rel_path="recommendation_server.py", file_name="recommendation_server.py",
            language="python", start_line=8, end_line=10,
            source_text=(
                "    def ListRecommendations(self, request, context):\n"
                "        response = product_catalog_stub.ListProducts(demo_pb2.Empty())\n"
                "        return response\n"
            ),
        )

        surfaces = extract_api_surfaces_for_graph(g, repo_root=str(repo_root))
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("grpc", "grpc:ProductCatalogService/ListProducts")
        assert g.has_node(nid)
        provider_edges = [d for _, v, d in g.edges("provider", data=True) if v == nid]
        assert any(d["relationship_type"] == "defines" for d in provider_edges)
        consumer_edges = [d for _, v, d in g.edges("consumer", data=True) if v == nid]
        assert consumer_edges, "cross-slice consumer failed to resolve the stub"
        assert any(d["relationship_type"] == "consumes" for d in consumer_edges)

    def test_python_consumer_without_repo_root_does_not_resolve_cross_slice(self):
        """Without ``repo_root`` the file-level binding pre-pass can't run, so
        a call site whose binding lives in another slice yields no surface —
        documents why the orchestrator pre-pass is required."""
        g = nx.MultiDiGraph()
        g.add_node(
            "consumer", symbol_name="ListRecommendations", symbol_type="method",
            rel_path="recommendation_server.py", file_name="recommendation_server.py",
            language="python", start_line=8, end_line=10,
            source_text=(
                "    def ListRecommendations(self, request, context):\n"
                "        response = product_catalog_stub.ListProducts(demo_pb2.Empty())\n"
                "        return response\n"
            ),
        )
        surfaces = extract_api_surfaces_for_graph(g)  # no repo_root
        assert "consumer" not in surfaces

    def test_consumes_edge_carries_dispatch_via(self):
        """The consumes edge records the RPC method in its ``via`` trail."""
        server = (
            "type CartServiceServer interface {\n"
            "    AddItem(context.Context, *AddItemRequest) (*Empty, error)\n"
            "}\n"
        )
        client = (
            "func f() { pb.NewCartServiceClient(conn).AddItem(ctx, req) }\n"
        )
        g = _grpc_server_client_graph("go", server, "go", client)
        surfaces = extract_api_surfaces_for_graph(g)
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("grpc", "grpc:CartService/AddItem")
        client_edges = [d for _, v, d in g.edges("client", data=True) if v == nid]
        via = client_edges[0].get("annotations", {}).get("via", [])
        assert "dispatch=AddItem" in via


# ─── 9. Phase B3 (rich parsers): TypeScript/JS, C++, Rust gRPC consumers ─

class TestGrpcClientMatcherRichLangs:
    """``_match_grpc_client`` coverage for the remaining rich-parser
    languages: JavaScript/TypeScript (grpc-js), C++ (gRPC ``NewStub``) and
    Rust (tonic). Each must normalise its call-site method name to the
    proto-canonical PascalCase the server matchers emit."""

    def test_typescript_camelcase_call_normalises_to_pascal(self):
        # grpc-js: new pkg.<Svc>Client(addr, creds); client.listProducts(...)
        bindings = extract_grpc_stub_bindings(
            "const client = new shop.ProductCatalogServiceClient(addr, creds);",
            "typescript",
        )
        assert bindings == {"client": "ProductCatalogService"}
        surfaces = _match_grpc_client(
            "client.listProducts(req, (err, res) => {});", "typescript", bindings
        )
        assert {s["surface"] for s in surfaces} == {
            "grpc:ProductCatalogService/ListProducts"
        }
        assert surfaces[0]["metadata"]["role"] == "client"

    def test_javascript_shares_typescript_path(self):
        bindings = extract_grpc_stub_bindings(
            "const c = new demo.CartServiceClient(addr, creds);", "javascript"
        )
        surfaces = _match_grpc_client("c.getCart(req, cb);", "javascript", bindings)
        assert {s["surface"] for s in surfaces} == {"grpc:CartService/GetCart"}

    def test_cpp_newstub_bind_and_arrow_call(self):
        # auto stub = <Svc>::NewStub(channel); stub->ListProducts(...)
        text = (
            "auto stub = ProductCatalogService::NewStub(channel);\n"
            "Status s = stub->ListProducts(&ctx, req, &resp);\n"
        )
        surfaces = _match_grpc_client(text, "cpp")
        assert {s["surface"] for s in surfaces} == {
            "grpc:ProductCatalogService/ListProducts"
        }
        assert surfaces[0]["metadata"]["role"] == "client"

    def test_cpp_unique_ptr_type_declaration_binds_stub(self):
        text = (
            "std::unique_ptr<CartService::Stub> cart = CartService::NewStub(ch);\n"
            "cart->GetCart(&ctx, req, &resp);\n"
        )
        surfaces = _match_grpc_client(text, "cpp")
        assert {s["surface"] for s in surfaces} == {"grpc:CartService/GetCart"}

    def test_rust_snake_case_call_normalises_to_pascal(self):
        # tonic: let mut c = <Svc>Client::connect(addr); c.list_products(req)
        text = (
            "let mut client = ProductCatalogServiceClient::connect(addr).await?;\n"
            "let resp = client.list_products(request).await?;\n"
        )
        surfaces = _match_grpc_client(text, "rust")
        assert {s["surface"] for s in surfaces} == {
            "grpc:ProductCatalogService/ListProducts"
        }
        assert surfaces[0]["metadata"]["role"] == "client"

    def test_rust_new_constructor_binds_stub(self):
        text = (
            "let mut cart = CartServiceClient::new(channel);\n"
            "cart.get_cart(req).await?;\n"
        )
        surfaces = _match_grpc_client(text, "rust")
        assert {s["surface"] for s in surfaces} == {"grpc:CartService/GetCart"}

    def test_cpp_unbound_arrow_call_is_ignored(self):
        assert _match_grpc_client("stub->ListProducts(&ctx, req, &resp);", "cpp") == []

    def test_rust_unbound_call_is_ignored(self):
        assert _match_grpc_client("client.list_products(request).await?;", "rust") == []


def _assert_grpc_defines_and_consumes(g, surface):
    """Module-level mirror of the integration assert (no inheritance, so the
    parent class's tests aren't re-collected under this class)."""
    surfaces = extract_api_surfaces_for_graph(g)
    materialize_contract_nodes(g, surfaces)
    nid = _contract_node_id("grpc", surface)
    assert g.has_node(nid), f"missing contract node for {surface}"
    server_edges = [d for _, v, d in g.edges("server", data=True) if v == nid]
    assert server_edges, "provider produced no edge into the contract"
    assert any(d["relationship_type"] == "defines" for d in server_edges)
    client_edges = [d for _, v, d in g.edges("client", data=True) if v == nid]
    assert client_edges, "consumer produced no edge into the contract"
    assert any(d["relationship_type"] == "consumes" for d in client_edges)
    return nid


class TestGrpcConsumesIntegrationRichLangs:
    """End-to-end provider+consumer dedupe for the rich-parser languages,
    reusing the same orchestrator → materialize assertions."""

    def test_typescript_consumer_links_to_ts_provider(self):
        server = (
            "server.addService(shop.ProductCatalogService.service, {\n"
            "    listProducts: listProductsHandler,\n"
            "});\n"
        )
        client = (
            "const client = new shop.ProductCatalogServiceClient(addr, creds);\n"
            "client.listProducts(req, (err, res) => {});\n"
        )
        g = _grpc_server_client_graph("typescript", server, "typescript", client)
        _assert_grpc_defines_and_consumes(g, "grpc:ProductCatalogService/ListProducts")

    def test_cpp_consumer_links_to_cpp_provider(self):
        server = (
            "class CartServiceImpl final : public CartService::Service {\n"
            "    Status GetCart(ServerContext* c, const Req* r, Resp* w) override {}\n"
            "};\n"
        )
        client = (
            "auto stub = CartService::NewStub(channel);\n"
            "Status s = stub->GetCart(&ctx, req, &resp);\n"
        )
        g = _grpc_server_client_graph("cpp", server, "cpp", client)
        _assert_grpc_defines_and_consumes(g, "grpc:CartService/GetCart")

    def test_rust_consumer_links_to_rust_provider(self):
        server = (
            "#[tonic::async_trait]\n"
            "impl CartService for MyCart {\n"
            "    async fn get_cart(&self, request: Request<Req>) "
            "-> Result<Response<Resp>, Status> {}\n"
            "}\n"
        )
        client = (
            "let mut cart = CartServiceClient::connect(addr).await?;\n"
            "let resp = cart.get_cart(request).await?;\n"
        )
        g = _grpc_server_client_graph("rust", server, "rust", client)
        _assert_grpc_defines_and_consumes(g, "grpc:CartService/GetCart")



# ─── Contract-edge provenance (language / confidence / obj_kind) ─────────

class TestContractEdgeProvenance:
    """Phase-2 follow-up: the contract edge must carry the three fields that
    were previously dropped — ``language`` (first-class repo_edges column),
    ``confidence`` and ``obj_kind`` (in the annotations blob)."""

    def test_defines_edge_carries_language(self):
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [APISurface(
                kind="rest_route", surface="POST /api/users",
                weight_hint=0.8, metadata={"method": "POST", "path": "/api/users"},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("rest_route", "POST /api/users")
        d = next(d for _, v, d in g.edges("py_handler", data=True) if v == nid)
        # owner py_handler.language == "python"
        assert d["language"] == "python"

    def test_consumes_edge_carries_language(self):
        g = _make_graph_with_surfaces()
        surfaces = {
            "ts_client": [APISurface(
                kind="rest_route", surface="POST /api/users", weight_hint=0.7,
                metadata={"method": "POST", "path": "/api/users", "role": "client"},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("rest_route", "POST /api/users")
        d = next(d for _, v, d in g.edges("ts_client", data=True) if v == nid)
        assert d["language"] == "typescript"

    def test_defines_edge_confidence_is_extracted(self):
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [APISurface(
                kind="rest_route", surface="POST /api/users", weight_hint=0.8,
                metadata={"method": "POST", "path": "/api/users", "role": "server"},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("rest_route", "POST /api/users")
        d = next(d for _, v, d in g.edges("py_handler", data=True) if v == nid)
        assert d["annotations"]["confidence"] == "EXTRACTED"

    def test_consumes_edge_confidence_is_inferred(self):
        g = _make_graph_with_surfaces()
        surfaces = {
            "ts_client": [APISurface(
                kind="rest_route", surface="POST /api/users", weight_hint=0.7,
                metadata={"method": "POST", "path": "/api/users", "role": "client"},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("rest_route", "POST /api/users")
        d = next(d for _, v, d in g.edges("ts_client", data=True) if v == nid)
        assert d["annotations"]["confidence"] == "INFERRED"

    def test_edge_obj_kind_matches_contract_kind(self):
        g = _make_graph_with_surfaces()
        surfaces = {
            "py_handler": [APISurface(
                kind="grpc", surface="grpc:UserSvc/Get",
                weight_hint=0.8, metadata={"service": "UserSvc", "method": "Get"},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("grpc", "grpc:UserSvc/Get")
        d = next(d for _, v, d in g.edges("py_handler", data=True) if v == nid)
        assert d["annotations"]["obj_kind"] == "grpc"

    def test_confidence_round_trips_through_query_helper(self):
        """graph_query_service._edge_confidence must read back the label the
        materializer stashed in the annotations blob (the canonical channel —
        repo_edges has no confidence column)."""
        from plugin_implementation.code_graph.graph_query_service import (
            _edge_confidence,
        )
        import json

        g = _make_graph_with_surfaces()
        surfaces = {
            "ts_client": [APISurface(
                kind="rest_route", surface="POST /api/users", weight_hint=0.7,
                metadata={"method": "POST", "path": "/api/users", "role": "client"},
            )],
        }
        materialize_contract_nodes(g, surfaces)
        nid = _contract_node_id("rest_route", "POST /api/users")
        d = next(d for _, v, d in g.edges("ts_client", data=True) if v == nid)
        assert _edge_confidence(d) == "INFERRED"
        # Also survives the JSON-string form the storage backend produces.
        serialized = {"annotations": json.dumps(d["annotations"])}
        assert _edge_confidence(serialized) == "INFERRED"
