"""K3 — contract consumption is discoverable through ask/research tools.

Phase B emits ``defines`` (server) and ``consumes`` (client) edges from
code symbols to a shared ``contract`` node, each carrying a line-precise
``via`` anchor (route / callsite). The ``get_relationships`` research tool
must surface both the edge type and the anchor so an agent can answer
"who consumes POST /api/orders and where?" without reading every file.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from plugin_implementation.code_graph.storage_query_service import StorageQueryService
from plugin_implementation.deep_research.research_tools import create_codebase_tools
from plugin_implementation.unified_db import UnifiedWikiDB


def _tool(tools, name):
    return next(tool for tool in tools if tool.name == name)


@pytest.fixture
def contract_db(tmp_path):
    """DB with a REST contract node, a server definer, and a client consumer."""
    db = UnifiedWikiDB(str(tmp_path / "contract.wiki.db"), embedding_dim=4)
    db._upsert_nodes_batch([
        {
            "node_id": "py::OrderService",
            "rel_path": "src/orders.py", "file_name": "orders.py",
            "language": "python", "symbol_name": "OrderService",
            "symbol_type": "class",
            "source_text": "class OrderService:\n    def create(self): ...",
            "signature": "class OrderService",
            "is_architectural": 1, "is_doc": 0,
        },
        {
            "node_id": "ts::OrderClient",
            "rel_path": "web/orderClient.ts", "file_name": "orderClient.ts",
            "language": "typescript", "symbol_name": "OrderClient",
            "symbol_type": "class",
            "source_text": "class OrderClient {\n  create() { axios.post('/api/orders') }\n}",
            "signature": "class OrderClient",
            "is_architectural": 1, "is_doc": 0,
        },
        {
            "node_id": "contract::rest::POST /api/orders",
            "rel_path": "src/orders.py", "file_name": "orders.py",
            "language": "python", "symbol_name": "POST /api/orders",
            "symbol_type": "contract", "source_text": "",
            "signature": "rest_route", "is_architectural": 1, "is_doc": 0,
        },
    ])
    # Server defines the contract; client consumes it. Both carry anchors.
    db.upsert_edge(
        "py::OrderService", "contract::rest::POST /api/orders", "defines",
        edge_class="structural",
        annotations='{"via": ["route=/api/orders"]}',
    )
    db.upsert_edge(
        "ts::OrderClient", "contract::rest::POST /api/orders", "consumes",
        edge_class="structural",
        annotations='{"via": ["callsite=orderClient.ts@L2"]}',
    )
    db.conn.commit()
    db._populate_fts5()
    yield db
    db.close()


def test_contract_relationships_show_consumes_and_via(contract_db):
    with patch.dict(os.environ, {"DEEPWIKI_PROGRESSIVE_TOOLS": "1"}):
        tools = create_codebase_tools(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            query_service=StorageQueryService(contract_db),
        )
    result = _tool(tools, "get_relationships_tool").invoke(
        {"symbol_name": "POST /api/orders", "direction": "both", "max_depth": 1}
    )

    # Both the server definer and the client consumer must be reachable.
    assert "OrderService" in result
    assert "OrderClient" in result
    # Edge types are surfaced verbatim.
    assert "defines" in result
    assert "consumes" in result
    # Line-precise provenance anchors are rendered.
    assert "route=/api/orders" in result
    assert "callsite=orderClient.ts@L2" in result


def test_consumer_relationship_carries_anchor(contract_db):
    """Querying from the client side surfaces its consumes anchor."""
    svc = StorageQueryService(contract_db)
    node_id, rels = svc.resolve_and_traverse(
        "OrderClient", direction="outgoing", max_depth=1, max_results=20,
    )
    consumes = [r for r in rels if r.relationship_type == "consumes"]
    assert consumes, "consumes edge missing from client traversal"
    assert consumes[0].via == ["callsite=orderClient.ts@L2"]
    assert consumes[0].edge_class == "structural"
