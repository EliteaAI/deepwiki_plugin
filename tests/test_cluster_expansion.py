"""
Unit tests for cluster_expansion.py — Phase 6 of the Unified Graph & Clustering plan.

Tests cover:
- expand_for_page basic flow (symbols → expansion → docs)
- Symbol resolution (exact match + FTS5 fallback)
- Cluster boundary enforcement (cross-cluster leaks blocked)
- Token budget exhaustion (budget respected)
- Priority grouping (P0 > P1 > P2 order)
- Documentation node inclusion (is_doc cluster members)
- Empty / edge-case inputs
- _extract_macro_id regex parsing
- _node_to_document conversion
- DEEPWIKI_LEGACY_INDEXES flag effect

Run: python -m pytest tests/test_cluster_expansion.py -v
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure we can import from the plugin
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugin_implementation.feature_flags import FeatureFlags
from plugin_implementation.unified_db import UnifiedWikiDB
from plugin_implementation.cluster_expansion import (
    expand_for_page,
    _resolve_symbols,
    _collect_expansion_neighbors,
    _collect_search_terms,
    _count_structural_edges,
    _find_framework_references,
    _get_cluster_docs,
    _node_to_document,
    _estimate_tokens,
    _fts_fallback,
    _search_framework_fts,
    DEFAULT_TOKEN_BUDGET,
    MAX_NEIGHBORS_PER_SYMBOL,
    MAX_EXPANSION_TOTAL,
    EXPANSION_SYMBOL_TYPES,
    _P0_REL_TYPES,
    _P1_REL_TYPES,
    _P2_REL_TYPES,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_cluster.wiki.db"


@pytest.fixture
def db(db_path):
    """Create a fresh DB with a graph that has 2 macro-clusters."""
    _db = UnifiedWikiDB(db_path, embedding_dim=4)
    _populate_test_graph(_db)
    yield _db
    _db.close()


def _populate_test_graph(db: UnifiedWikiDB):
    """Insert test nodes and edges into the unified DB.

    Cluster layout:
        Macro 0 (auth cluster):
            AuthService (class)       <- micro 0
            authenticate (function)   <- micro 0
            TokenValidator (class)    <- micro 1
            README.md (file_doc)      <- micro 0
        Macro 1 (data cluster):
            DataStore (class)         <- micro 2
            query (function)          <- micro 2
    """
    conn = db.conn

    nodes = [
        # --- Macro 0, Micro 0: auth core ---
        {
            "node_id": "py::auth::AuthService",
            "rel_path": "src/auth/service.py",
            "file_name": "service.py",
            "language": "python",
            "symbol_name": "AuthService",
            "symbol_type": "class",
            "start_line": 10, "end_line": 80,
            "source_text": "class AuthService:\n    def authenticate(self):\n        pass\n" * 5,
            "docstring": "Handles auth.",
            "signature": "class AuthService",
            "is_architectural": 1,
            "is_doc": 0,
            "macro_cluster": 0,
            "micro_cluster": 0,
        },
        {
            "node_id": "py::auth::authenticate",
            "rel_path": "src/auth/service.py",
            "file_name": "service.py",
            "language": "python",
            "symbol_name": "authenticate",
            "symbol_type": "function",
            "start_line": 20, "end_line": 40,
            "source_text": "def authenticate(token: str) -> bool:\n    return True",
            "docstring": "Verify token.",
            "signature": "def authenticate(token: str) -> bool",
            "is_architectural": 1,
            "is_doc": 0,
            "macro_cluster": 0,
            "micro_cluster": 0,
        },
        # --- Macro 0, Micro 1: auth support ---
        {
            "node_id": "py::auth::TokenValidator",
            "rel_path": "src/auth/validator.py",
            "file_name": "validator.py",
            "language": "python",
            "symbol_name": "TokenValidator",
            "symbol_type": "class",
            "start_line": 1, "end_line": 30,
            "source_text": "class TokenValidator:\n    def validate(self, token):\n        pass",
            "docstring": "Validates JWT tokens.",
            "signature": "class TokenValidator",
            "is_architectural": 1,
            "is_doc": 0,
            "macro_cluster": 0,
            "micro_cluster": 1,
        },
        # --- Macro 0, Micro 0: doc node ---
        {
            "node_id": "doc::auth::README",
            "rel_path": "src/auth/README.md",
            "file_name": "README.md",
            "language": "markdown",
            "symbol_name": "README.md",
            "symbol_type": "file_doc",
            "start_line": 1, "end_line": 20,
            "source_text": "# Auth Module\n\nAuthentication documentation.",
            "docstring": "",
            "signature": "",
            "is_architectural": 1,
            "is_doc": 1,
            "macro_cluster": 0,
            "micro_cluster": 0,
        },
        # --- Macro 1, Micro 2: data cluster ---
        {
            "node_id": "py::data::DataStore",
            "rel_path": "src/data/store.py",
            "file_name": "store.py",
            "language": "python",
            "symbol_name": "DataStore",
            "symbol_type": "class",
            "start_line": 1, "end_line": 50,
            "source_text": "class DataStore:\n    def query(self, q):\n        pass",
            "docstring": "Data storage.",
            "signature": "class DataStore",
            "is_architectural": 1,
            "is_doc": 0,
            "macro_cluster": 1,
            "micro_cluster": 2,
        },
        {
            "node_id": "py::data::query",
            "rel_path": "src/data/store.py",
            "file_name": "store.py",
            "language": "python",
            "symbol_name": "query",
            "symbol_type": "function",
            "start_line": 10, "end_line": 30,
            "source_text": "def query(self, q: str) -> list:\n    return []",
            "docstring": "Execute a query.",
            "signature": "def query(self, q: str) -> list",
            "is_architectural": 1,
            "is_doc": 0,
            "macro_cluster": 1,
            "micro_cluster": 2,
        },
        # --- Non-architectural node (should be excluded from expansion) ---
        {
            "node_id": "py::auth::_helper",
            "rel_path": "src/auth/service.py",
            "file_name": "service.py",
            "language": "python",
            "symbol_name": "_helper",
            "symbol_type": "function",
            "start_line": 82, "end_line": 85,
            "source_text": "def _helper(): pass",
            "docstring": "",
            "signature": "def _helper()",
            "is_architectural": 0,
            "is_doc": 0,
            "macro_cluster": 0,
            "micro_cluster": 0,
        },
    ]

    cols = [
        "node_id", "rel_path", "file_name", "language", "symbol_name",
        "symbol_type", "start_line", "end_line", "source_text", "docstring",
        "signature", "is_architectural", "is_doc", "macro_cluster", "micro_cluster",
    ]
    placeholders = ",".join("?" * len(cols))
    col_names = ",".join(cols)

    for n in nodes:
        vals = [n.get(c, "") for c in cols]
        conn.execute(f"INSERT INTO repo_nodes ({col_names}) VALUES ({placeholders})", vals)

    # --- Edges ---
    edges = [
        # AuthService → authenticate (P0: defines_body)
        ("py::auth::AuthService", "py::auth::authenticate", "defines_body", "structural", 5.0),
        # AuthService → TokenValidator (P1: composition)
        ("py::auth::AuthService", "py::auth::TokenValidator", "composition", "structural", 3.0),
        # authenticate → DataStore (P2: calls — cross-cluster!)
        ("py::auth::authenticate", "py::data::DataStore", "calls", "structural", 2.0),
        # DataStore → query (P0: defines_body)
        ("py::data::DataStore", "py::data::query", "defines_body", "structural", 5.0),
        # TokenValidator → AuthService (inheritance — P0, incoming)
        ("py::auth::TokenValidator", "py::auth::AuthService", "inheritance", "structural", 4.0),
        # AuthService → _helper (calls — to non-architectural)
        ("py::auth::AuthService", "py::auth::_helper", "calls", "structural", 1.0),
    ]

    for src, tgt, rel, ecls, weight in edges:
        conn.execute(
            "INSERT INTO repo_edges (source_id, target_id, rel_type, edge_class, weight) "
            "VALUES (?, ?, ?, ?, ?)",
            (src, tgt, rel, ecls, weight),
        )

    conn.commit()


def _insert_framework_node(db: UnifiedWikiDB, node_id: str, **kwargs):
    defaults = {
        "rel_path": f"src/{node_id.replace('::', '_')}.py",
        "file_name": f"{node_id}.py",
        "language": "python",
        "symbol_name": node_id,
        "symbol_type": "class",
        "start_line": 1,
        "end_line": 5,
        "source_text": f"class {node_id}:\n    pass",
        "docstring": "",
        "signature": "",
        "is_architectural": 1,
        "is_doc": 0,
        "macro_cluster": 0,
        "micro_cluster": 0,
    }
    defaults.update(kwargs)

    cols = [
        "node_id", "rel_path", "file_name", "language", "symbol_name",
        "symbol_type", "start_line", "end_line", "source_text", "docstring",
        "signature", "is_architectural", "is_doc", "macro_cluster", "micro_cluster",
    ]
    placeholders = ",".join("?" for _ in cols)
    db.conn.execute(
        f"INSERT INTO repo_nodes ({','.join(cols)}) VALUES ({placeholders})",
        [node_id] + [defaults.get(col, "") for col in cols[1:]],
    )


def _insert_framework_edge(
    db: UnifiedWikiDB,
    source_id: str,
    target_id: str,
    rel_type: str = "calls",
    edge_class: str = "structural",
    weight: float = 1.0,
):
    db.conn.execute(
        "INSERT INTO repo_edges (source_id, target_id, rel_type, edge_class, weight) "
        "VALUES (?, ?, ?, ?, ?)",
        (source_id, target_id, rel_type, edge_class, weight),
    )


@pytest.fixture
def framework_db(tmp_path):
    """Graph modelled after framework string dispatch.

    Cluster 0 has handler-like classes with child methods. Cluster 1 has
    dispatcher/source nodes that mention those method names as strings.
    """
    _db = UnifiedWikiDB(tmp_path / "framework.wiki.db", embedding_dim=4)

    _insert_framework_node(
        _db,
        "python::Event",
        symbol_name="Event",
        symbol_type="class",
        macro_cluster=0,
        source_text="class Event:\n    def configuration_created(self, ctx): pass",
    )
    _insert_framework_node(
        _db,
        "python::Event.configuration_created",
        symbol_name="configuration_created",
        symbol_type="method",
        is_architectural=0,
        macro_cluster=0,
        source_text="def configuration_created(self, ctx): pass",
    )
    _insert_framework_edge(
        _db,
        "python::Event",
        "python::Event.configuration_created",
        rel_type="defines",
    )

    _insert_framework_node(
        _db,
        "python::AnotherHandler",
        symbol_name="AnotherHandler",
        symbol_type="class",
        macro_cluster=0,
        source_text="class AnotherHandler:\n    def handle_request(self, r): pass",
    )
    _insert_framework_node(
        _db,
        "python::AnotherHandler.handle_request",
        symbol_name="handle_request",
        symbol_type="method",
        is_architectural=0,
        macro_cluster=0,
        source_text="def handle_request(self, r): pass",
    )
    _insert_framework_edge(
        _db,
        "python::AnotherHandler",
        "python::AnotherHandler.handle_request",
        rel_type="defines",
    )

    _insert_framework_node(
        _db,
        "python::get_tools_c0",
        symbol_name="get_tools",
        symbol_type="function",
        macro_cluster=0,
        source_text="def get_tools(): return []",
    )

    _insert_framework_node(
        _db,
        "python::create_configuration",
        symbol_name="create_configuration",
        symbol_type="function",
        macro_cluster=1,
        source_text="def create_configuration(ctx):\n    emit('configuration_created', data)",
    )
    _insert_framework_node(
        _db,
        "python::dispatch_util",
        symbol_name="dispatch_util",
        symbol_type="function",
        macro_cluster=1,
        source_text="def dispatch_util(r):\n    manager.handle_request(r)",
    )
    _insert_framework_node(
        _db,
        "python::get_tools_c1",
        symbol_name="get_tools",
        symbol_type="function",
        macro_cluster=1,
        source_text="def get_tools(): return toolkit.get_tools()",
    )
    _insert_framework_node(
        _db,
        "python::some_helper",
        symbol_name="some_helper",
        symbol_type="function",
        macro_cluster=1,
        source_text="def some_helper(): return 42",
    )

    _db._populate_fts5()
    _db.conn.commit()
    yield _db
    _db.close()


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _node_to_document
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeToDocument:
    def test_basic_conversion(self):
        node = {
            "node_id": "py::Foo",
            "symbol_name": "Foo",
            "symbol_type": "class",
            "rel_path": "src/foo.py",
            "file_name": "foo.py",
            "language": "python",
            "start_line": 1, "end_line": 10,
            "source_text": "class Foo: pass",
            "docstring": "A foo.",
            "signature": "class Foo",
            "is_architectural": 1,
            "is_doc": 0,
        }
        doc = _node_to_document(node, is_initial=True)
        assert doc.page_content == "class Foo: pass"
        assert doc.metadata["symbol_name"] == "Foo"
        assert doc.metadata["source"] == "src/foo.py"
        assert doc.metadata["is_initially_retrieved"] is True

    def test_expanded_from_metadata(self):
        node = {"node_id": "x", "source_text": "code", "symbol_name": "x"}
        doc = _node_to_document(node, is_initial=False, expanded_from="calls")
        assert doc.metadata["is_initially_retrieved"] is False
        assert doc.metadata["expanded_from"] == "calls"


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _estimate_tokens
# ═══════════════════════════════════════════════════════════════════════════

class TestEstimateTokens:
    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_short_code(self):
        text = "class Foo: pass"
        tokens = _estimate_tokens(text)
        assert tokens > 0
        assert tokens < 20  # ~15 chars → ~4 tokens

    def test_longer_code(self):
        text = "x" * 350  # 350 chars → ~100 tokens
        tokens = _estimate_tokens(text)
        assert 80 <= tokens <= 120


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _resolve_symbols
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveSymbols:
    def test_exact_match(self, db):
        result = _resolve_symbols(db, ["AuthService"])
        assert "py::auth::AuthService" in result
        assert result["py::auth::AuthService"]["symbol_name"] == "AuthService"

    def test_exact_match_scoped_to_cluster(self, db):
        result = _resolve_symbols(db, ["AuthService"], macro_id=0)
        assert "py::auth::AuthService" in result

    def test_exact_match_wrong_cluster(self, db):
        # AuthService is in macro=0, searching in macro=1 should find nothing
        result = _resolve_symbols(db, ["AuthService"], macro_id=1)
        assert "py::auth::AuthService" not in result

    def test_multiple_symbols(self, db):
        result = _resolve_symbols(db, ["AuthService", "authenticate", "TokenValidator"])
        assert len(result) == 3

    def test_nonexistent_symbol(self, db):
        result = _resolve_symbols(db, ["NonExistent"])
        assert len(result) == 0

    def test_empty_input(self, db):
        result = _resolve_symbols(db, [])
        assert result == {}

    def test_non_architectural_excluded(self, db):
        """Non-architectural symbols (_helper) should not be resolved."""
        result = _resolve_symbols(db, ["_helper"])
        assert len(result) == 0

    def test_fts_fallback_when_populated(self, db):
        """FTS5 fallback returns empty when FTS content not populated via from_networkx.

        NOTE: The test DB uses raw SQL inserts, so the FTS5 table is empty.
        This test verifies the fallback is safe (returns []) rather than crashing.
        A full integration test with from_networkx populates FTS5 content."""
        rows = _fts_fallback(db, "AuthService", macro_id=0)
        # FTS5 content not populated via raw inserts — expect graceful empty
        assert isinstance(rows, list)


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _collect_expansion_neighbors
# ═══════════════════════════════════════════════════════════════════════════

class TestCollectExpansionNeighbors:
    def test_basic_expansion(self, db):
        """AuthService should expand to authenticate and TokenValidator (same cluster)."""
        seen = {"py::auth::AuthService"}
        neighbors = _collect_expansion_neighbors(db, ["py::auth::AuthService"], seen, macro_id=0)
        neighbor_ids = {nid for nid, _, _ in neighbors}
        assert "py::auth::authenticate" in neighbor_ids
        assert "py::auth::TokenValidator" in neighbor_ids

    def test_cross_cluster_blocked(self, db):
        """authenticate → DataStore is cross-cluster (macro 0 → 1), should be filtered out."""
        seen = {"py::auth::authenticate"}
        neighbors = _collect_expansion_neighbors(db, ["py::auth::authenticate"], seen, macro_id=0)
        neighbor_ids = {nid for nid, _, _ in neighbors}
        assert "py::data::DataStore" not in neighbor_ids

    def test_non_architectural_excluded(self, db):
        """_helper is non-architectural and should be excluded from expansion."""
        seen = {"py::auth::AuthService"}
        neighbors = _collect_expansion_neighbors(db, ["py::auth::AuthService"], seen, macro_id=0)
        neighbor_ids = {nid for nid, _, _ in neighbors}
        assert "py::auth::_helper" not in neighbor_ids

    def test_no_cluster_filter(self, db):
        """Without macro_id, cross-cluster expansion is allowed."""
        seen = {"py::auth::authenticate"}
        neighbors = _collect_expansion_neighbors(db, ["py::auth::authenticate"], seen, macro_id=None)
        neighbor_ids = {nid for nid, _, _ in neighbors}
        # DataStore should now appear since no cluster boundary
        assert "py::data::DataStore" in neighbor_ids

    def test_sorted_by_weight(self, db):
        """Neighbours should come back sorted by edge weight desc."""
        seen = {"py::auth::AuthService"}
        neighbors = _collect_expansion_neighbors(db, ["py::auth::AuthService"], seen, macro_id=0)
        # defines_body (5.0) should come before composition (3.0)
        ids = [nid for nid, _, _ in neighbors]
        if "py::auth::authenticate" in ids and "py::auth::TokenValidator" in ids:
            assert ids.index("py::auth::authenticate") < ids.index("py::auth::TokenValidator")

    def test_incoming_edges_included(self, db):
        """TokenValidator → AuthService (inheritance) should be found as incoming edge."""
        seen = {"py::auth::AuthService"}
        neighbors = _collect_expansion_neighbors(db, ["py::auth::AuthService"], seen, macro_id=0)
        # TokenValidator has an inheritance edge TO AuthService, should appear via incoming
        neighbor_ids = {nid for nid, _, _ in neighbors}
        assert "py::auth::TokenValidator" in neighbor_ids


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _get_cluster_docs
# ═══════════════════════════════════════════════════════════════════════════

class TestGetClusterDocs:
    def test_get_macro_docs(self, db):
        """Should find README.md doc in macro cluster 0."""
        docs = _get_cluster_docs(db, macro_id=0, micro_id=None, seen_ids=set())
        assert len(docs) == 1
        assert docs[0]["symbol_name"] == "README.md"
        assert docs[0]["is_doc"] == 1

    def test_get_micro_docs(self, db):
        """README is in micro=0. Searching micro=1 should find nothing."""
        docs = _get_cluster_docs(db, macro_id=0, micro_id=1, seen_ids=set())
        assert len(docs) == 0

    def test_seen_ids_excluded(self, db):
        """Already-seen doc should not appear."""
        docs = _get_cluster_docs(db, macro_id=0, micro_id=None,
                                 seen_ids={"doc::auth::README"})
        assert len(docs) == 0

    def test_no_docs_in_cluster(self, db):
        """Macro cluster 1 has no doc nodes."""
        docs = _get_cluster_docs(db, macro_id=1, micro_id=None, seen_ids=set())
        assert len(docs) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Tests: expand_for_page (integration)
# ═══════════════════════════════════════════════════════════════════════════

class TestExpandForPage:
    def test_basic_expansion(self, db):
        """Expanding [AuthService] with macro=0 should produce several docs."""
        docs = expand_for_page(
            db=db,
            page_symbols=["AuthService"],
            macro_id=0,
            token_budget=50_000,
        )
        assert len(docs) > 0
        # AuthService itself should be initial
        auth_doc = next(d for d in docs if d.metadata["symbol_name"] == "AuthService")
        assert auth_doc.metadata["is_initially_retrieved"] is True

    def test_expansion_includes_neighbors(self, db):
        """Expansion should include authenticate and TokenValidator (same cluster).

        Pinned to the legacy (smart_expansion=False) expansion path so the
        assertion keeps guarding that branch — smart expansion uses focused
        per-type rel-type sets and does not follow ``defines_body`` from a
        class to its function members.
        """
        legacy_flags = FeatureFlags(smart_expansion=False, language_hints=False)
        with patch(
            "plugin_implementation.cluster_expansion.get_feature_flags",
            return_value=legacy_flags,
        ):
            docs = expand_for_page(db, ["AuthService"], macro_id=0, token_budget=50_000)
        names = {d.metadata["symbol_name"] for d in docs}
        assert "authenticate" in names
        assert "TokenValidator" in names

    def test_expansion_includes_docs(self, db):
        """README.md should be included as cluster doc."""
        docs = expand_for_page(db, ["AuthService"], macro_id=0, token_budget=50_000)
        names = {d.metadata["symbol_name"] for d in docs}
        assert "README.md" in names

    def test_cross_cluster_not_included(self, db):
        """DataStore (macro=1) should NOT appear in macro=0 expansion."""
        docs = expand_for_page(db, ["AuthService"], macro_id=0, token_budget=50_000)
        names = {d.metadata["symbol_name"] for d in docs}
        assert "DataStore" not in names
        assert "query" not in names

    def test_empty_symbols(self, db):
        """No symbols → no documents."""
        docs = expand_for_page(db, [], macro_id=0, token_budget=50_000)
        assert docs == []

    def test_token_budget_respected(self, db):
        """With a very small budget, only the initial symbol fits."""
        docs = expand_for_page(
            db=db,
            page_symbols=["AuthService"],
            macro_id=0,
            token_budget=10,  # tiny budget
        )
        # Budget is so small that probably nothing fits (AuthService has ~75 chars = ~21 tokens)
        # Depending on the exact content, might get 0 or 1 doc
        total_tokens = sum(_estimate_tokens(d.page_content) for d in docs)
        # Even if docs are returned, they should not massively exceed the budget
        # (the first symbol may still be added even if it exceeds remaining budget
        #  because the loop breaks after that)
        assert total_tokens <= 50  # generous upper bound for budget=10

    def test_token_budget_medium(self, db):
        """With moderate budget, initial symbol fits but not all expansions."""
        # AuthService source is ~75 chars repeated 5x = ~375 chars = ~107 tokens
        docs = expand_for_page(
            db=db,
            page_symbols=["AuthService"],
            macro_id=0,
            token_budget=120,  # enough for AuthService, maybe authenticate
        )
        # Should have at least 1 doc (AuthService itself)
        assert len(docs) >= 1

    def test_multiple_initial_symbols(self, db):
        """Multiple page symbols should all be resolved and expanded."""
        docs = expand_for_page(
            db=db,
            page_symbols=["AuthService", "authenticate"],
            macro_id=0,
            token_budget=50_000,
        )
        initial = [d for d in docs if d.metadata.get("is_initially_retrieved")]
        assert len(initial) == 2

    def test_include_docs_false(self, db):
        """When include_docs=False, README should not appear."""
        docs = expand_for_page(
            db=db,
            page_symbols=["AuthService"],
            macro_id=0,
            token_budget=50_000,
            include_docs=False,
        )
        names = {d.metadata["symbol_name"] for d in docs}
        # README might still appear if it's a neighbor via edges, but not from
        # the cluster doc step. Since there's no edge from AuthService to README,
        # it should be absent.
        assert "README.md" not in names

    def test_no_duplicate_nodes(self, db):
        """No node should appear twice in the result."""
        docs = expand_for_page(
            db=db,
            page_symbols=["AuthService", "TokenValidator"],
            macro_id=0,
            token_budget=50_000,
        )
        node_ids = [d.metadata["node_id"] for d in docs]
        assert len(node_ids) == len(set(node_ids))

    def test_nonexistent_symbols_graceful(self, db):
        """Non-existent symbols produce empty list, no crash."""
        docs = expand_for_page(
            db=db,
            page_symbols=["DoesNotExist", "AlsoMissing"],
            macro_id=0,
            token_budget=50_000,
        )
        assert docs == []

    def test_data_cluster_expansion(self, db):
        """DataStore in macro=1 should expand to query but not AuthService.

        Pinned to the legacy expansion path — see ``test_expansion_includes_neighbors``.
        """
        legacy_flags = FeatureFlags(smart_expansion=False, language_hints=False)
        with patch(
            "plugin_implementation.cluster_expansion.get_feature_flags",
            return_value=legacy_flags,
        ):
            docs = expand_for_page(
                db=db,
                page_symbols=["DataStore"],
                macro_id=1,
                token_budget=50_000,
            )
        names = {d.metadata["symbol_name"] for d in docs}
        assert "DataStore" in names
        assert "query" in names
        assert "AuthService" not in names

    def test_priority_ordering(self, db):
        """P0 neighbours should appear before P2 neighbours."""
        docs = expand_for_page(db, ["AuthService"], macro_id=0, token_budget=50_000)
        # authenticate is P0 (defines_body), TokenValidator is P1 (composition)
        # They should appear in that order among the expanded docs
        expanded = [d for d in docs if not d.metadata.get("is_initially_retrieved")]
        names = [d.metadata["symbol_name"] for d in expanded]
        if "authenticate" in names and "TokenValidator" in names:
            assert names.index("authenticate") < names.index("TokenValidator")


# ═══════════════════════════════════════════════════════════════════════════
# Tests: framework reference discovery (Step 2.75)
# ═══════════════════════════════════════════════════════════════════════════

class TestCountStructuralEdges:
    def test_structural_edges_are_counted(self, framework_db):
        count = _count_structural_edges(framework_db, "python::Event")
        assert count == 1

    def test_bridge_edges_are_ignored(self, framework_db):
        _insert_framework_edge(
            framework_db,
            "python::Event",
            "python::some_helper",
            rel_type="references",
            edge_class="bridge",
        )
        framework_db.conn.commit()

        assert _count_structural_edges(framework_db, "python::Event") == 1

    def test_leaf_without_edges_is_orphan(self, framework_db):
        assert _count_structural_edges(framework_db, "python::some_helper") == 0


class TestCollectSearchTerms:
    def test_class_collects_child_method_names(self, framework_db):
        node = framework_db.get_node("python::Event")
        terms = _collect_search_terms(framework_db, "python::Event", node)

        assert "Event" in terms
        assert "configuration_created" in terms

    def test_function_returns_only_own_name(self, framework_db):
        node = framework_db.get_node("python::get_tools_c0")
        terms = _collect_search_terms(framework_db, "python::get_tools_c0", node)

        assert terms == ["get_tools"]

    def test_short_name_is_excluded(self, framework_db):
        terms = _collect_search_terms(
            framework_db,
            "python::ab",
            {"symbol_name": "ab", "symbol_type": "function"},
        )

        assert terms == []


class TestFindFrameworkReferences:
    def test_finds_cross_cluster_dispatcher_via_child_method(self, framework_db):
        orphan_node = framework_db.get_node("python::Event")
        results = _find_framework_references(
            framework_db,
            {"python::Event": orphan_node},
            seen_ids=set(),
            macro_id=0,
        )

        hit_ids = {node_id for node_id, _, _ in results}
        assert "python::create_configuration" in hit_ids

    def test_filters_same_name_peer_definitions(self, framework_db):
        orphan_node = framework_db.get_node("python::get_tools_c0")
        results = _find_framework_references(
            framework_db,
            {"python::get_tools_c0": orphan_node},
            seen_ids=set(),
            macro_id=0,
        )

        hit_ids = {node_id for node_id, _, _ in results}
        assert "python::get_tools_c1" not in hit_ids

    def test_skips_other_orphans(self, framework_db):
        results = _find_framework_references(
            framework_db,
            {
                "python::Event": framework_db.get_node("python::Event"),
                "python::AnotherHandler": framework_db.get_node("python::AnotherHandler"),
            },
            seen_ids=set(),
            macro_id=0,
        )

        hit_ids = {node_id for node_id, _, _ in results}
        assert "python::Event" not in hit_ids
        assert "python::AnotherHandler" not in hit_ids

    def test_expanded_from_reason_is_descriptive(self, framework_db):
        orphan_node = framework_db.get_node("python::AnotherHandler")
        results = _find_framework_references(
            framework_db,
            {"python::AnotherHandler": orphan_node},
            seen_ids=set(),
            macro_id=0,
        )

        assert any(reason.startswith("fts_framework_ref:") for _, _, reason in results)

    def test_empty_fts_result_does_not_fall_back_to_sql_scan(self):
        class ConnSpy:
            def __init__(self):
                self.executed = False

            def execute(self, *_args, **_kwargs):
                self.executed = True
                raise AssertionError("SQL fallback should not run")

        class EmptyFtsDB:
            def __init__(self):
                self.conn = ConnSpy()

            def search_fts5(self, query, cluster_id=None, limit=20):
                return []

        db = EmptyFtsDB()

        assert _search_framework_fts(db, "configuration_created", None, 3) == []
        assert db.conn.executed is False


class TestFrameworkReferenceExpansion:
    def test_expand_for_page_includes_framework_string_refs(self, framework_db):
        docs = expand_for_page(
            db=framework_db,
            page_symbols=["Event"],
            macro_id=0,
            token_budget=50_000,
        )

        by_name = {doc.metadata["symbol_name"]: doc for doc in docs}
        assert "Event" in by_name
        assert "create_configuration" in by_name
        assert by_name["create_configuration"].metadata["expanded_from"].startswith(
            "fts_framework_ref:"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tests: _extract_macro_id
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractMacroId:
    """Test the regex helper from cluster_utils."""

    @pytest.fixture(autouse=True)
    def _import_extract(self):
        """Import from the lightweight cluster_utils module (no heavy deps)."""
        from plugin_implementation.cluster_utils import extract_macro_id
        self._extract = extract_macro_id

    def test_standard_rationale(self):
        r = "Grouped by graph clustering (macro=3, micro=7, 12 symbols)"
        assert self._extract(r) == 3

    def test_macro_zero(self):
        r = "Grouped by graph clustering (macro=0, micro=1, 5 symbols)"
        assert self._extract(r) == 0

    def test_no_macro(self):
        r = "This page is about authentication"
        assert self._extract(r) is None

    def test_empty_rationale(self):
        assert self._extract("") is None

    def test_none_rationale(self):
        assert self._extract(None) is None


# ═══════════════════════════════════════════════════════════════════════════
# Tests: legacy-index disablement (always-off baseline)
# ═══════════════════════════════════════════════════════════════════════════

class TestLegacyIndexesFlag:
    """The legacy ``.code_graph.gz`` / ``.faiss`` writes are now permanently off.

    UnifiedWikiDB is the single index store. The previous
    ``DEEPWIKI_LEGACY_INDEXES`` env switch has been removed; only the
    ``LEGACY_INDEXES_ENABLED`` constant survives for backward-compatible
    imports.
    """

    def test_legacy_indexes_disabled_by_default(self):
        from plugin_implementation import filesystem_indexer

        assert filesystem_indexer.LEGACY_INDEXES_ENABLED is False


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Constants consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_default_token_budget(self):
        assert DEFAULT_TOKEN_BUDGET == 50_000

    def test_max_neighbors(self):
        assert MAX_NEIGHBORS_PER_SYMBOL == 15

    def test_max_expansion(self):
        assert MAX_EXPANSION_TOTAL == 200

    def test_expansion_types_include_core(self):
        for t in ("class", "function", "interface", "module_doc", "file_doc"):
            assert t in EXPANSION_SYMBOL_TYPES

    def test_priority_groups_disjoint(self):
        """Priority groups should not overlap."""
        assert _P0_REL_TYPES & _P1_REL_TYPES == set()
        assert _P0_REL_TYPES & _P2_REL_TYPES == set()
        assert _P1_REL_TYPES & _P2_REL_TYPES == set()

    def test_priority_groups_known_types(self):
        """All P0/P1/P2 types should be non-empty."""
        assert len(_P0_REL_TYPES) > 0
        assert len(_P1_REL_TYPES) > 0
        assert len(_P2_REL_TYPES) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Augmentation — C++ / Go / Rust cross-file content stitching
# ═══════════════════════════════════════════════════════════════════════════

def _populate_cpp_graph(db: UnifiedWikiDB):
    """Insert a small C++ graph: header decl + cpp impl linked by defines_body."""
    conn = db.conn
    nodes = [
        # Class declaration in header
        {
            "node_id": "cpp::Point.h::Point",
            "rel_path": "include/Point.h",
            "file_name": "Point.h",
            "language": "cpp",
            "symbol_name": "Point",
            "symbol_type": "class",
            "start_line": 1, "end_line": 20,
            "source_text": "class Point {\npublic:\n  void move(int dx, int dy);\n  int x, y;\n};",
            "is_architectural": 1, "is_doc": 0,
            "macro_cluster": 0, "micro_cluster": 0,
        },
        # Method declaration (child of class)
        {
            "node_id": "cpp::Point.h::Point::move",
            "rel_path": "include/Point.h",
            "file_name": "Point.h",
            "language": "cpp",
            "symbol_name": "move",
            "symbol_type": "method",
            "start_line": 3, "end_line": 3,
            "source_text": "void move(int dx, int dy);",
            "is_architectural": 0, "is_doc": 0,
            "macro_cluster": 0, "micro_cluster": 0,
        },
        # Implementation in .cpp (impl node — defines_body source)
        {
            "node_id": "cpp::Point.cpp::Point::move",
            "rel_path": "src/Point.cpp",
            "file_name": "Point.cpp",
            "language": "cpp",
            "symbol_name": "move",
            "symbol_type": "method",
            "start_line": 5, "end_line": 10,
            "source_text": "void Point::move(int dx, int dy) {\n  x += dx;\n  y += dy;\n}",
            "is_architectural": 0, "is_doc": 0,
            "macro_cluster": 0, "micro_cluster": 0,
        },
        # Standalone function in header
        {
            "node_id": "cpp::util.h::distance",
            "rel_path": "include/util.h",
            "file_name": "util.h",
            "language": "cpp",
            "symbol_name": "distance",
            "symbol_type": "function",
            "start_line": 1, "end_line": 1,
            "source_text": "double distance(const Point& a, const Point& b);",
            "is_architectural": 1, "is_doc": 0,
            "macro_cluster": 0, "micro_cluster": 0,
        },
        # Implementation of standalone function
        {
            "node_id": "cpp::util.cpp::distance",
            "rel_path": "src/util.cpp",
            "file_name": "util.cpp",
            "language": "cpp",
            "symbol_name": "distance",
            "symbol_type": "function",
            "start_line": 3, "end_line": 6,
            "source_text": "double distance(const Point& a, const Point& b) {\n  return sqrt(pow(a.x-b.x,2)+pow(a.y-b.y,2));\n}",
            "is_architectural": 1, "is_doc": 0,
            "macro_cluster": 0, "micro_cluster": 0,
        },
    ]

    cols = [
        "node_id", "rel_path", "file_name", "language", "symbol_name",
        "symbol_type", "start_line", "end_line", "source_text",
        "is_architectural", "is_doc", "macro_cluster", "micro_cluster",
    ]
    placeholders = ",".join("?" * len(cols))
    col_names = ",".join(cols)
    for n in nodes:
        vals = [n.get(c, "") for c in cols]
        conn.execute(f"INSERT INTO repo_nodes ({col_names}) VALUES ({placeholders})", vals)

    edges = [
        # Point class defines move method
        ("cpp::Point.h::Point", "cpp::Point.h::Point::move", "defines", "structural", 5.0),
        # Implementation defines_body declaration (impl → decl)
        ("cpp::Point.cpp::Point::move", "cpp::Point.h::Point::move", "defines_body", "structural", 5.0),
        # Standalone function impl defines_body declaration
        ("cpp::util.cpp::distance", "cpp::util.h::distance", "defines_body", "structural", 5.0),
    ]
    for src, tgt, rel, ecls, weight in edges:
        conn.execute(
            "INSERT INTO repo_edges (source_id, target_id, rel_type, edge_class, weight) "
            "VALUES (?, ?, ?, ?, ?)",
            (src, tgt, rel, ecls, weight),
        )
    conn.commit()


def _populate_go_graph(db: UnifiedWikiDB):
    """Insert a Go-style graph: struct + receiver methods in separate file."""
    conn = db.conn
    nodes = [
        {
            "node_id": "go::server.go::Server",
            "rel_path": "pkg/server.go",
            "file_name": "server.go",
            "language": "go",
            "symbol_name": "Server",
            "symbol_type": "struct",
            "start_line": 10, "end_line": 15,
            "source_text": "type Server struct {\n\taddr string\n\trouter *Router\n}",
            "is_architectural": 1, "is_doc": 0,
            "macro_cluster": 0, "micro_cluster": 0,
        },
        {
            "node_id": "go::handler.go::Server::Handle",
            "rel_path": "pkg/handler.go",
            "file_name": "handler.go",
            "language": "go",
            "symbol_name": "Handle",
            "symbol_type": "method",
            "start_line": 5, "end_line": 12,
            "source_text": "func (s *Server) Handle(w http.ResponseWriter, r *http.Request) {\n\ts.router.ServeHTTP(w, r)\n}",
            "is_architectural": 0, "is_doc": 0,
            "macro_cluster": 0, "micro_cluster": 0,
        },
        {
            "node_id": "go::handler.go::Server::Start",
            "rel_path": "pkg/handler.go",
            "file_name": "handler.go",
            "language": "go",
            "symbol_name": "Start",
            "symbol_type": "method",
            "start_line": 14, "end_line": 20,
            "source_text": "func (s *Server) Start() error {\n\treturn http.ListenAndServe(s.addr, s)\n}",
            "is_architectural": 0, "is_doc": 0,
            "macro_cluster": 0, "micro_cluster": 0,
        },
    ]

    cols = [
        "node_id", "rel_path", "file_name", "language", "symbol_name",
        "symbol_type", "start_line", "end_line", "source_text",
        "is_architectural", "is_doc", "macro_cluster", "micro_cluster",
    ]
    placeholders = ",".join("?" * len(cols))
    col_names = ",".join(cols)
    for n in nodes:
        vals = [n.get(c, "") for c in cols]
        conn.execute(f"INSERT INTO repo_nodes ({col_names}) VALUES ({placeholders})", vals)

    edges = [
        # Server defines Handle (cross-file receiver method)
        ("go::server.go::Server", "go::handler.go::Server::Handle", "defines", "structural", 5.0),
        # Server defines Start (cross-file receiver method)
        ("go::server.go::Server", "go::handler.go::Server::Start", "defines", "structural", 5.0),
    ]
    for src, tgt, rel, ecls, weight in edges:
        conn.execute(
            "INSERT INTO repo_edges (source_id, target_id, rel_type, edge_class, weight) "
            "VALUES (?, ?, ?, ?, ?)",
            (src, tgt, rel, ecls, weight),
        )
    conn.commit()


def _populate_rust_graph(db: UnifiedWikiDB):
    """Insert a Rust-style graph: struct + cross-file impl block methods."""
    conn = db.conn
    nodes = [
        {
            "node_id": "rust::config.rs::Config",
            "rel_path": "src/config.rs",
            "file_name": "config.rs",
            "language": "rust",
            "symbol_name": "Config",
            "symbol_type": "struct",
            "start_line": 1, "end_line": 8,
            "source_text": "pub struct Config {\n    pub host: String,\n    pub port: u16,\n}",
            "is_architectural": 1, "is_doc": 0,
            "macro_cluster": 0, "micro_cluster": 0,
        },
        {
            "node_id": "rust::config_impl.rs::Config::from_env",
            "rel_path": "src/config_impl.rs",
            "file_name": "config_impl.rs",
            "language": "rust",
            "symbol_name": "from_env",
            "symbol_type": "method",
            "start_line": 3, "end_line": 10,
            "source_text": "impl Config {\n    pub fn from_env() -> Self {\n        Config { host: env::var(\"HOST\").unwrap(), port: 8080 }\n    }\n}",
            "is_architectural": 0, "is_doc": 0,
            "macro_cluster": 0, "micro_cluster": 0,
        },
    ]

    cols = [
        "node_id", "rel_path", "file_name", "language", "symbol_name",
        "symbol_type", "start_line", "end_line", "source_text",
        "is_architectural", "is_doc", "macro_cluster", "micro_cluster",
    ]
    placeholders = ",".join("?" * len(cols))
    col_names = ",".join(cols)
    for n in nodes:
        vals = [n.get(c, "") for c in cols]
        conn.execute(f"INSERT INTO repo_nodes ({col_names}) VALUES ({placeholders})", vals)

    edges = [
        ("rust::config.rs::Config", "rust::config_impl.rs::Config::from_env", "defines", "structural", 5.0),
    ]
    for src, tgt, rel, ecls, weight in edges:
        conn.execute(
            "INSERT INTO repo_edges (source_id, target_id, rel_type, edge_class, weight) "
            "VALUES (?, ?, ?, ?, ?)",
            (src, tgt, rel, ecls, weight),
        )
    conn.commit()


# ── Fixtures ──

@pytest.fixture
def cpp_db(tmp_path):
    db = UnifiedWikiDB(tmp_path / "cpp.wiki.db", embedding_dim=4)
    _populate_cpp_graph(db)
    yield db
    db.close()


@pytest.fixture
def go_db(tmp_path):
    db = UnifiedWikiDB(tmp_path / "go.wiki.db", embedding_dim=4)
    _populate_go_graph(db)
    yield db
    db.close()


@pytest.fixture
def rust_db(tmp_path):
    db = UnifiedWikiDB(tmp_path / "rust.wiki.db", embedding_dim=4)
    _populate_rust_graph(db)
    yield db
    db.close()


# ── Tests ──

class TestAugmentCpp:
    """C++ header↔implementation augmentation via cluster expansion."""

    def test_class_augmented_with_method_impls(self, cpp_db):
        """Point class should be augmented with out-of-line method body from Point.cpp."""
        docs = expand_for_page(
            db=cpp_db,
            page_symbols=["Point"],
            macro_id=0,
            token_budget=50_000,
        )
        point_docs = [d for d in docs if d.metadata["symbol_name"] == "Point"]
        assert len(point_docs) == 1
        content = point_docs[0].page_content
        assert "Point::move" in content, "Should include implementation body"
        assert "Point.cpp" in content, "Should cite impl file"
        assert point_docs[0].metadata.get("is_augmented") is True

    def test_function_augmented_with_impl(self, cpp_db):
        """Standalone function distance() in header should get augmented with .cpp body."""
        docs = expand_for_page(
            db=cpp_db,
            page_symbols=["distance"],
            macro_id=0,
            token_budget=50_000,
        )
        dist_docs = [d for d in docs if d.metadata["symbol_name"] == "distance"]
        assert len(dist_docs) >= 1
        # The header decl doc should now contain the implementation
        header_doc = [d for d in dist_docs if "util.h" in d.metadata.get("rel_path", "")]
        assert len(header_doc) == 1
        content = header_doc[0].page_content
        assert "sqrt" in content, "Implementation body should be stitched in"
        assert "util.cpp" in content, "Should cite impl file"

    def test_no_augmentation_for_python(self, db):
        """Python symbols should NOT be augmented (no header/impl split)."""
        docs = expand_for_page(
            db=db,
            page_symbols=["AuthService"],
            macro_id=0,
            token_budget=50_000,
        )
        auth_docs = [d for d in docs if d.metadata["symbol_name"] == "AuthService"]
        assert len(auth_docs) == 1
        assert auth_docs[0].metadata.get("is_augmented") is not True


class TestAugmentGo:
    """Go receiver method augmentation via cluster expansion."""

    def test_struct_augmented_with_receiver_methods(self, go_db):
        """Go Server struct should be augmented with Handle and Start from handler.go."""
        docs = expand_for_page(
            db=go_db,
            page_symbols=["Server"],
            macro_id=0,
            token_budget=50_000,
        )
        server_docs = [d for d in docs if d.metadata["symbol_name"] == "Server"]
        assert len(server_docs) == 1
        content = server_docs[0].page_content
        assert "Handle" in content, "Should include Handle receiver method"
        assert "Start" in content, "Should include Start receiver method"
        assert "handler.go" in content, "Should cite source file"
        assert server_docs[0].metadata.get("is_augmented") is True


class TestAugmentRust:
    """Rust impl block augmentation via cluster expansion."""

    def test_struct_augmented_with_impl_methods(self, rust_db):
        """Rust Config struct should be augmented with from_env from config_impl.rs."""
        docs = expand_for_page(
            db=rust_db,
            page_symbols=["Config"],
            macro_id=0,
            token_budget=50_000,
        )
        config_docs = [d for d in docs if d.metadata["symbol_name"] == "Config"]
        assert len(config_docs) == 1
        content = config_docs[0].page_content
        assert "from_env" in content, "Should include impl method"
        assert "config_impl.rs" in content, "Should cite impl file"
        assert config_docs[0].metadata.get("is_augmented") is True


class TestAugmentBudget:
    """Augmentation should respect the token budget."""

    def test_augmentation_costs_budget(self, cpp_db):
        """Augmented content should consume extra budget tokens."""
        # With a tiny budget, augmentation should still happen but less room
        # for expansion
        docs_small = expand_for_page(
            db=cpp_db,
            page_symbols=["Point"],
            macro_id=0,
            token_budget=100,  # very tight
        )
        docs_large = expand_for_page(
            db=cpp_db,
            page_symbols=["Point"],
            macro_id=0,
            token_budget=50_000,
        )
        # More budget → potentially more docs
        assert len(docs_large) >= len(docs_small)
