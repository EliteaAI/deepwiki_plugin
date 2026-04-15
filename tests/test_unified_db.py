"""
Unit tests for UnifiedWikiDB — Phase 1 of the Unified Graph & Clustering spec.

Tests cover:
- Schema creation and connection
- Node CRUD (single, batch, queries)
- Edge CRUD (single, batch, queries, neighbors)
- FTS5 search (basic, filtered, prefix fallback)
- NetworkX roundtrip (from_networkx → to_networkx)
- Metadata storage
- Cluster assignments
- Statistics
- Hybrid search (FTS5-only when vec unavailable)
- Edge cases (empty graph, missing nodes, special characters)

Run: python -m pytest tests/test_unified_db.py -v
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import networkx as nx
import pytest

# Ensure we can import from the plugin
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugin_implementation.unified_db import (
    UnifiedWikiDB,
    UNIFIED_DB_ENABLED,
    _serialize_float32_vec,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def db_path(tmp_path):
    """Return a temporary .wiki.db path."""
    return tmp_path / "test_repo.wiki.db"


@pytest.fixture
def db(db_path):
    """Create and return a fresh UnifiedWikiDB instance."""
    _db = UnifiedWikiDB(db_path, embedding_dim=4)  # dim=4 for test vectors
    yield _db
    _db.close()


@pytest.fixture
def sample_graph():
    """Create a small NetworkX MultiDiGraph mimicking graph_builder output."""
    G = nx.MultiDiGraph()
    G.graph["analysis_type"] = "multi_tier"

    # --- Rich nodes (comprehensive analysis) ---
    G.add_node("python::auth::AuthService", **{
        "rel_path": "src/auth/service.py",
        "file_name": "service.py",
        "language": "python",
        "symbol_type": "class",
        "symbol_name": "AuthService",
        "start_line": 10,
        "end_line": 80,
        "parent_symbol": None,
        "analysis_level": "comprehensive",
        "symbol": type("Symbol", (), {
            "source_text": "class AuthService:\n    def authenticate(self):\n        pass",
            "docstring": "Handles authentication logic.",
            "signature": "class AuthService",
            "parameters": [],
            "return_type": "",
        })(),
    })

    G.add_node("python::auth::AuthService::authenticate", **{
        "rel_path": "src/auth/service.py",
        "file_name": "service.py",
        "language": "python",
        "symbol_type": "function",
        "symbol_name": "authenticate",
        "start_line": 20,
        "end_line": 40,
        "parent_symbol": "python::auth::AuthService",
        "analysis_level": "comprehensive",
        "symbol": type("Symbol", (), {
            "source_text": "def authenticate(self, token: str) -> bool:\n    return True",
            "docstring": "Verify authentication token.",
            "signature": "def authenticate(self, token: str) -> bool",
            "parameters": ["self", "token: str"],
            "return_type": "bool",
        })(),
    })

    # --- Basic nodes (basic analysis) ---
    G.add_node("python::utils::hash_password", **{
        "name": "hash_password",
        "type": "function",
        "symbol_name": "hash_password",
        "symbol_type": "function",
        "file_path": "/repo/src/utils/crypto.py",
        "rel_path": "src/utils/crypto.py",
        "file_name": "crypto.py",
        "language": "python",
        "start_line": 5,
        "end_line": 15,
        "analysis_level": "basic",
        "source_text": "def hash_password(password: str) -> str:\n    return hashlib.sha256(password.encode()).hexdigest()",
        "docstring": "Hash a password using SHA-256.",
        "parameters": ["password: str"],
        "return_type": "str",
    })

    G.add_node("python::models::User", **{
        "name": "User",
        "type": "class",
        "symbol_name": "User",
        "symbol_type": "class",
        "rel_path": "src/models/user.py",
        "file_name": "user.py",
        "language": "python",
        "start_line": 1,
        "end_line": 30,
        "analysis_level": "basic",
        "source_text": "class User:\n    def __init__(self, name):\n        self.name = name",
        "docstring": "User data model.",
        "parameters": [],
        "return_type": "",
    })

    # --- A doc node ---
    G.add_node("python::readme::file_doc", **{
        "name": "README.md",
        "type": "file_doc",
        "symbol_name": "README.md",
        "symbol_type": "file_doc",
        "rel_path": "README.md",
        "file_name": "README.md",
        "language": "markdown",
        "start_line": 0,
        "end_line": 0,
        "analysis_level": "basic",
        "source_text": "# Auth Service\n\nA service for authentication and authorization.",
        "docstring": "",
    })

    # --- Edges ---
    G.add_edge("python::auth::AuthService", "python::auth::AuthService::authenticate", **{
        "relationship_type": "CONTAINS",
        "source_file": "src/auth/service.py",
        "target_file": "src/auth/service.py",
        "analysis_level": "comprehensive",
        "annotations": {"member_type": "method"},
    })

    G.add_edge("python::auth::AuthService::authenticate", "python::utils::hash_password", **{
        "relationship_type": "CALLS",
        "source_file": "src/auth/service.py",
        "target_file": "src/utils/crypto.py",
        "analysis_level": "comprehensive",
        "annotations": {"cross_file": True},
    })

    G.add_edge("python::auth::AuthService", "python::models::User", **{
        "relationship_type": "USES",
        "source_file": "src/auth/service.py",
        "target_file": "src/models/user.py",
        "analysis_level": "comprehensive",
        "annotations": {},
    })

    # Parallel edge (same source→target, different annotations)
    G.add_edge("python::auth::AuthService", "python::models::User", **{
        "relationship_type": "IMPORTS",
        "source_file": "src/auth/service.py",
        "target_file": "src/models/user.py",
        "analysis_level": "comprehensive",
        "annotations": {"import_style": "from_import"},
    })

    return G


# ═══════════════════════════════════════════════════════════════════════════
# DB-1: Schema and connection
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemaCreation:
    """DB-1: Verify schema creation and pragmas."""

    def test_db_file_created(self, db_path):
        """DB-1.1: .wiki.db file is created on disk."""
        _db = UnifiedWikiDB(db_path)
        assert db_path.exists()
        _db.close()

    def test_tables_exist(self, db):
        """DB-1.2: All expected tables are created."""
        tables = {row[0] for row in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'table')"
        ).fetchall()}

        assert "repo_nodes" in tables
        assert "repo_edges" in tables
        assert "wiki_meta" in tables

    def test_fts5_exists(self, db):
        """DB-1.3: FTS5 virtual table exists."""
        tables = {row[0] for row in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()}
        assert "repo_fts" in tables

    def test_wal_mode(self, db):
        """DB-1.4: WAL journal mode is active."""
        mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_on(self, db):
        """DB-1.5: Foreign keys are enabled."""
        fk = db.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_indexes_created(self, db):
        """DB-1.6: Expected indexes exist."""
        indexes = {row[0] for row in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()}
        assert "idx_nodes_path" in indexes
        assert "idx_edges_source" in indexes
        assert "idx_edges_target" in indexes

    def test_readonly_mode(self, db_path):
        """DB-1.7: Read-only mode works on existing DB."""
        # Create DB first
        _db = UnifiedWikiDB(db_path)
        _db.upsert_node("test::node", symbol_name="test")
        _db.conn.commit()
        _db.close()

        # Open read-only
        ro_db = UnifiedWikiDB(db_path, readonly=True)
        node = ro_db.get_node("test::node")
        assert node is not None
        assert node["symbol_name"] == "test"
        ro_db.close()

    def test_schema_idempotent(self, db_path):
        """DB-1.8: Creating schema twice doesn't error."""
        db1 = UnifiedWikiDB(db_path)
        db1.upsert_node("n1", symbol_name="A")
        db1.conn.commit()
        db1.close()

        # Re-open same path → schema DDL re-runs
        db2 = UnifiedWikiDB(db_path)
        assert db2.get_node("n1") is not None
        db2.close()


# ═══════════════════════════════════════════════════════════════════════════
# DB-2: Node CRUD
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeCRUD:
    """DB-2: Node insert, upsert, query operations."""

    def test_upsert_and_get_node(self, db):
        """DB-2.1: Insert and retrieve a single node."""
        db.upsert_node(
            "python::main::MyClass",
            rel_path="src/main.py",
            file_name="main.py",
            language="python",
            symbol_name="MyClass",
            symbol_type="class",
            start_line=10,
            end_line=50,
            source_text="class MyClass:\n    pass",
            docstring="A test class.",
        )
        db.conn.commit()

        node = db.get_node("python::main::MyClass")
        assert node is not None
        assert node["symbol_name"] == "MyClass"
        assert node["symbol_type"] == "class"
        assert node["rel_path"] == "src/main.py"
        assert node["start_line"] == 10
        assert node["end_line"] == 50
        assert node["is_architectural"] == 1

    def test_upsert_replaces(self, db):
        """DB-2.2: Upsert with same ID replaces existing node."""
        db.upsert_node("n1", symbol_name="Original")
        db.conn.commit()
        assert db.get_node("n1")["symbol_name"] == "Original"

        db.upsert_node("n1", symbol_name="Updated")
        db.conn.commit()
        assert db.get_node("n1")["symbol_name"] == "Updated"
        assert db.node_count() == 1

    def test_batch_upsert(self, db):
        """DB-2.3: Batch insert multiple nodes."""
        nodes = [
            {"node_id": f"node_{i}", "symbol_name": f"Fn{i}", "symbol_type": "function"}
            for i in range(100)
        ]
        db.upsert_nodes_batch(nodes)
        db.conn.commit()
        assert db.node_count() == 100

    def test_get_missing_node_returns_none(self, db):
        """DB-2.4: get_node for absent ID returns None."""
        assert db.get_node("nonexistent::id") is None

    def test_get_nodes_by_path_prefix(self, db):
        """DB-2.5: Path prefix query works."""
        db.upsert_nodes_batch([
            {"node_id": "n1", "rel_path": "src/auth/login.py", "symbol_name": "login"},
            {"node_id": "n2", "rel_path": "src/auth/session.py", "symbol_name": "session"},
            {"node_id": "n3", "rel_path": "src/models/user.py", "symbol_name": "user"},
            {"node_id": "n4", "rel_path": "tests/test_auth.py", "symbol_name": "test"},
        ])
        db.conn.commit()

        auth_nodes = db.get_nodes_by_path_prefix("src/auth")
        assert len(auth_nodes) == 2
        names = {n["symbol_name"] for n in auth_nodes}
        assert names == {"login", "session"}

    def test_get_architectural_nodes(self, db):
        """DB-2.6: Architectural filter works."""
        db.upsert_nodes_batch([
            {"node_id": "n1", "symbol_type": "class", "symbol_name": "Cls"},
            {"node_id": "n2", "symbol_type": "function", "symbol_name": "fn"},
            {"node_id": "n3", "symbol_type": "variable", "symbol_name": "var"},
            {"node_id": "n4", "symbol_type": "interface", "symbol_name": "iface"},
        ])
        db.conn.commit()

        arch = db.get_architectural_nodes()
        # class, function, interface are architectural; variable is not
        names = {n["symbol_name"] for n in arch}
        assert "Cls" in names
        assert "fn" in names
        assert "iface" in names
        assert "var" not in names

    def test_is_doc_flag(self, db):
        """DB-2.7: Doc symbol types set is_doc=1."""
        db.upsert_node("doc1", symbol_type="file_doc", symbol_name="README")
        db.upsert_node("doc2", symbol_type="module_doc", symbol_name="init")
        db.upsert_node("code1", symbol_type="class", symbol_name="Cls")
        db.conn.commit()

        doc1 = db.get_node("doc1")
        doc2 = db.get_node("doc2")
        code1 = db.get_node("code1")
        assert doc1["is_doc"] == 1
        assert doc2["is_doc"] == 1
        assert code1["is_doc"] == 0

    def test_parameters_list_serialization(self, db):
        """DB-2.8: Parameters stored as JSON list survives roundtrip."""
        db.upsert_node("fn1", parameters=["self", "x: int", "y: str"])
        db.conn.commit()

        node = db.get_node("fn1")
        params = json.loads(node["parameters"])
        assert params == ["self", "x: int", "y: str"]

    def test_symbol_type_enum_normalization(self, db):
        """DB-2.9: SymbolType enum-like objects are normalized to lowercase string."""
        # Simulate a SymbolType enum with a .value attribute
        class FakeEnum:
            value = "CLASS"

        db.upsert_node("n1", symbol_type=FakeEnum())
        db.conn.commit()
        node = db.get_node("n1")
        assert node["symbol_type"] == "class"


# ═══════════════════════════════════════════════════════════════════════════
# DB-3: Edge CRUD
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCRUD:
    """DB-3: Edge insert, query, neighbor operations."""

    def test_upsert_edge(self, db):
        """DB-3.1: Insert a single edge."""
        db.upsert_node("a", symbol_name="A")
        db.upsert_node("b", symbol_name="B")
        db.upsert_edge("a", "b", "CALLS", source_file="a.py", target_file="b.py")
        db.conn.commit()

        assert db.edge_count() == 1
        edges = db.get_edges_from("a")
        assert len(edges) == 1
        assert edges[0]["rel_type"] == "CALLS"

    def test_batch_edges(self, db):
        """DB-3.2: Batch insert edges."""
        db.upsert_nodes_batch([
            {"node_id": "a"},
            {"node_id": "b"},
            {"node_id": "c"},
        ])
        db.upsert_edges_batch([
            {"source_id": "a", "target_id": "b", "rel_type": "CALLS"},
            {"source_id": "a", "target_id": "c", "rel_type": "USES"},
            {"source_id": "b", "target_id": "c", "rel_type": "IMPORTS"},
        ])
        db.conn.commit()

        assert db.edge_count() == 3

    def test_parallel_edges_allowed(self, db):
        """DB-3.3: Parallel edges (same endpoints, different type) are allowed."""
        db.upsert_nodes_batch([{"node_id": "a"}, {"node_id": "b"}])
        db.upsert_edges_batch([
            {"source_id": "a", "target_id": "b", "rel_type": "CALLS"},
            {"source_id": "a", "target_id": "b", "rel_type": "IMPORTS"},
            {"source_id": "a", "target_id": "b", "rel_type": "CALLS",
             "annotations": {"variant": "2"}},
        ])
        db.conn.commit()

        assert db.edge_count() == 3
        edges = db.get_edges_from("a")
        assert len(edges) == 3

    def test_get_edges_filtered_by_type(self, db):
        """DB-3.4: Filter edges by rel_type."""
        db.upsert_nodes_batch([{"node_id": "a"}, {"node_id": "b"}, {"node_id": "c"}])
        db.upsert_edges_batch([
            {"source_id": "a", "target_id": "b", "rel_type": "CALLS"},
            {"source_id": "a", "target_id": "c", "rel_type": "USES"},
        ])
        db.conn.commit()

        calls_only = db.get_edges_from("a", rel_types=["CALLS"])
        assert len(calls_only) == 1
        assert calls_only[0]["target_id"] == "b"

    def test_get_edges_to(self, db):
        """DB-3.5: Incoming edges query."""
        db.upsert_nodes_batch([{"node_id": "a"}, {"node_id": "b"}, {"node_id": "c"}])
        db.upsert_edges_batch([
            {"source_id": "a", "target_id": "c", "rel_type": "CALLS"},
            {"source_id": "b", "target_id": "c", "rel_type": "USES"},
        ])
        db.conn.commit()

        incoming = db.get_edges_to("c")
        assert len(incoming) == 2

    def test_get_neighbors_1hop(self, db):
        """DB-3.6: 1-hop outgoing neighbors."""
        db.upsert_nodes_batch([
            {"node_id": "a"}, {"node_id": "b"}, {"node_id": "c"}, {"node_id": "d"},
        ])
        db.upsert_edges_batch([
            {"source_id": "a", "target_id": "b", "rel_type": "CALLS"},
            {"source_id": "a", "target_id": "c", "rel_type": "CALLS"},
            {"source_id": "b", "target_id": "d", "rel_type": "CALLS"},
        ])
        db.conn.commit()

        neighbors = db.get_neighbors("a", hops=1, direction="out")
        assert neighbors == {"b", "c"}

    def test_get_neighbors_2hop(self, db):
        """DB-3.7: 2-hop outgoing neighbors."""
        db.upsert_nodes_batch([
            {"node_id": "a"}, {"node_id": "b"}, {"node_id": "c"}, {"node_id": "d"},
        ])
        db.upsert_edges_batch([
            {"source_id": "a", "target_id": "b", "rel_type": "CALLS"},
            {"source_id": "b", "target_id": "c", "rel_type": "CALLS"},
            {"source_id": "c", "target_id": "d", "rel_type": "CALLS"},
        ])
        db.conn.commit()

        neighbors = db.get_neighbors("a", hops=2, direction="out")
        assert neighbors == {"b", "c"}

    def test_get_neighbors_bidirectional(self, db):
        """DB-3.8: Bidirectional neighbor traversal."""
        db.upsert_nodes_batch([
            {"node_id": "a"}, {"node_id": "b"}, {"node_id": "c"},
        ])
        db.upsert_edges_batch([
            {"source_id": "a", "target_id": "b", "rel_type": "CALLS"},
            {"source_id": "c", "target_id": "a", "rel_type": "USES"},
        ])
        db.conn.commit()

        neighbors = db.get_neighbors("a", hops=1, direction="both")
        assert neighbors == {"b", "c"}

    def test_edge_annotations_roundtrip(self, db):
        """DB-3.9: Edge annotations dict survives JSON roundtrip."""
        db.upsert_nodes_batch([{"node_id": "a"}, {"node_id": "b"}])
        db.upsert_edge("a", "b", "CALLS", annotations={"cross_file": True, "count": 3})
        db.conn.commit()

        edges = db.get_edges_from("a")
        ann = json.loads(edges[0]["annotations"])
        assert ann["cross_file"] is True
        assert ann["count"] == 3


# ═══════════════════════════════════════════════════════════════════════════
# DB-4: FTS5 search
# ═══════════════════════════════════════════════════════════════════════════

class TestFTS5Search:
    """DB-4: Full-text search via FTS5."""

    def _insert_and_index(self, db):
        """Helper: insert nodes and rebuild FTS5."""
        db.upsert_nodes_batch([
            {"node_id": "n1", "symbol_name": "authenticate", "symbol_type": "function",
             "rel_path": "src/auth/login.py",
             "source_text": "def authenticate(token): verify_token(token)",
             "docstring": "Verify authentication token and return user."},
            {"node_id": "n2", "symbol_name": "hash_password", "symbol_type": "function",
             "rel_path": "src/utils/crypto.py",
             "source_text": "def hash_password(pw): return sha256(pw)",
             "docstring": "Hash password using SHA-256."},
            {"node_id": "n3", "symbol_name": "UserModel", "symbol_type": "class",
             "rel_path": "src/models/user.py",
             "source_text": "class UserModel: pass",
             "docstring": "User data model with authentication support."},
        ])
        db.conn.commit()
        db._populate_fts5()

    def test_basic_search(self, db):
        """DB-4.1: Basic FTS5 search returns ranked results."""
        self._insert_and_index(db)
        results = db.search_fts5("authenticate")
        assert len(results) > 0
        # The "authenticate" function should be ranked highest
        assert results[0]["node_id"] == "n1"

    def test_search_by_docstring(self, db):
        """DB-4.2: Search matches docstring content."""
        self._insert_and_index(db)
        # Note: FTS5 porter tokenizer strips hyphens, so "SHA-256" becomes "sha" "256"
        results = db.search_fts5("password")
        assert any(r["node_id"] == "n2" for r in results)

    def test_search_with_path_filter(self, db):
        """DB-4.3: Path prefix filter restricts results."""
        self._insert_and_index(db)
        results = db.search_fts5("authentication", path_prefix="src/auth")
        # Only n1 is under src/auth, n3 mentions "authentication" but is under src/models
        node_ids = {r["node_id"] for r in results}
        assert "n1" in node_ids
        assert "n3" not in node_ids

    def test_empty_query_returns_empty(self, db):
        """DB-4.4: Empty query returns no results."""
        self._insert_and_index(db)
        assert db.search_fts5("") == []
        assert db.search_fts5("   ") == []

    def test_search_no_matches(self, db):
        """DB-4.5: Non-matching query returns empty."""
        self._insert_and_index(db)
        results = db.search_fts5("xyznonexistent")
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════
# DB-5: NetworkX roundtrip
# ═══════════════════════════════════════════════════════════════════════════

class TestNetworkXRoundtrip:
    """DB-5: from_networkx and to_networkx."""

    def test_from_networkx_node_count(self, db, sample_graph):
        """DB-5.1: All nodes imported."""
        db.from_networkx(sample_graph)
        assert db.node_count() == sample_graph.number_of_nodes()

    def test_from_networkx_edge_count(self, db, sample_graph):
        """DB-5.2: All edges imported (including parallel edges)."""
        db.from_networkx(sample_graph)
        assert db.edge_count() == sample_graph.number_of_edges()

    def test_rich_node_source_text_extraction(self, db, sample_graph):
        """DB-5.3: Rich nodes have source_text extracted from symbol object."""
        db.from_networkx(sample_graph)
        node = db.get_node("python::auth::AuthService")
        assert "class AuthService" in node["source_text"]
        assert "Handles authentication logic." in node["docstring"]
        assert node["signature"] == "class AuthService"

    def test_basic_node_source_text(self, db, sample_graph):
        """DB-5.4: Basic nodes have source_text stored directly."""
        db.from_networkx(sample_graph)
        node = db.get_node("python::utils::hash_password")
        assert "def hash_password" in node["source_text"]
        assert "SHA-256" in node["docstring"]

    def test_roundtrip_preserves_nodes(self, db, sample_graph):
        """DB-5.5: from_networkx → to_networkx preserves all nodes."""
        db.from_networkx(sample_graph)
        G2 = db.to_networkx()

        assert G2.number_of_nodes() == sample_graph.number_of_nodes()
        for nid in sample_graph.nodes:
            assert nid in G2.nodes, f"Node {nid} missing after roundtrip"

    def test_roundtrip_preserves_edges(self, db, sample_graph):
        """DB-5.6: from_networkx → to_networkx preserves all edges."""
        db.from_networkx(sample_graph)
        G2 = db.to_networkx()

        assert G2.number_of_edges() == sample_graph.number_of_edges()

    def test_roundtrip_node_attrs(self, db, sample_graph):
        """DB-5.7: Node attributes survive roundtrip."""
        db.from_networkx(sample_graph)
        G2 = db.to_networkx()

        n = G2.nodes["python::auth::AuthService"]
        assert n["symbol_name"] == "AuthService"
        assert n["symbol_type"] == "class"
        assert n["language"] == "python"
        assert n["start_line"] == 10

    def test_roundtrip_edge_attrs(self, db, sample_graph):
        """DB-5.8: Edge attributes survive roundtrip."""
        db.from_networkx(sample_graph)
        G2 = db.to_networkx()

        # Check edges from AuthService to authenticate
        src = "python::auth::AuthService"
        tgt = "python::auth::AuthService::authenticate"
        edges = [(u, v, d) for u, v, d in G2.edges(src, data=True) if v == tgt]
        assert len(edges) >= 1
        edge_data = edges[0][2]
        assert edge_data["relationship_type"] == "CONTAINS"

    def test_roundtrip_parallel_edges(self, db, sample_graph):
        """DB-5.9: Parallel edges preserved in roundtrip."""
        db.from_networkx(sample_graph)
        G2 = db.to_networkx()

        # AuthService→User has both USES and IMPORTS edges
        src = "python::auth::AuthService"
        tgt = "python::models::User"
        edges = [(u, v, d) for u, v, d in G2.edges(src, data=True) if v == tgt]
        rel_types = {e[2]["relationship_type"] for e in edges}
        assert "USES" in rel_types
        assert "IMPORTS" in rel_types

    def test_roundtrip_doc_node(self, db, sample_graph):
        """DB-5.10: Doc nodes are flagged correctly."""
        db.from_networkx(sample_graph)
        doc = db.get_node("python::readme::file_doc")
        assert doc is not None
        assert doc["is_doc"] == 1
        assert doc["symbol_type"] == "file_doc"

    def test_fts5_populated_after_import(self, db, sample_graph):
        """DB-5.11: FTS5 is populated after from_networkx."""
        db.from_networkx(sample_graph)
        # Search for text that's in a node's source_text
        results = db.search_fts5("authenticate")
        assert len(results) > 0

    def test_empty_graph(self, db):
        """DB-5.12: Empty graph import is a no-op."""
        G = nx.MultiDiGraph()
        db.from_networkx(G)
        assert db.node_count() == 0
        assert db.edge_count() == 0

    def test_none_graph(self, db):
        """DB-5.13: None graph import doesn't crash."""
        db.from_networkx(None)
        assert db.node_count() == 0


# ═══════════════════════════════════════════════════════════════════════════
# DB-6: Metadata
# ═══════════════════════════════════════════════════════════════════════════

class TestMetadata:
    """DB-6: Wiki metadata storage."""

    def test_set_get_string(self, db):
        """DB-6.1: Store and retrieve a string value."""
        db.set_meta("repo_name", "myproject")
        assert db.get_meta("repo_name") == "myproject"

    def test_set_get_dict(self, db):
        """DB-6.2: Store and retrieve a dict value."""
        config = {"dim": 1536, "model": "text-embedding-3-small"}
        db.set_meta("embedding_config", config)
        result = db.get_meta("embedding_config")
        assert result == config

    def test_set_get_list(self, db):
        """DB-6.3: Store and retrieve a list value."""
        langs = ["python", "javascript", "rust"]
        db.set_meta("languages", langs)
        assert db.get_meta("languages") == langs

    def test_get_missing_returns_default(self, db):
        """DB-6.4: Missing key returns default."""
        assert db.get_meta("nonexistent") is None
        assert db.get_meta("nonexistent", "fallback") == "fallback"

    def test_upsert_meta(self, db):
        """DB-6.5: Setting same key overwrites."""
        db.set_meta("version", 1)
        assert db.get_meta("version") == 1
        db.set_meta("version", 2)
        assert db.get_meta("version") == 2


# ═══════════════════════════════════════════════════════════════════════════
# DB-7: Cluster operations
# ═══════════════════════════════════════════════════════════════════════════

class TestClusterOperations:
    """DB-7: Cluster assignment and retrieval."""

    def test_set_and_get_cluster(self, db):
        """DB-7.1: Assign and query cluster."""
        db.upsert_node("n1", symbol_name="A")
        db.set_cluster("n1", macro=0, micro=1)
        db.conn.commit()

        nodes = db.get_nodes_by_cluster(macro=0, micro=1)
        assert len(nodes) == 1
        assert nodes[0]["node_id"] == "n1"

    def test_batch_clusters(self, db):
        """DB-7.2: Batch cluster assignment."""
        db.upsert_nodes_batch([
            {"node_id": f"n{i}", "symbol_name": f"S{i}"}
            for i in range(10)
        ])
        db.conn.commit()

        assignments = [
            (f"n{i}", i // 3, i % 3) for i in range(10)
        ]
        db.set_clusters_batch(assignments)
        db.conn.commit()

        cluster_0 = db.get_nodes_by_cluster(macro=0)
        assert len(cluster_0) == 3  # n0, n1, n2

    def test_get_all_clusters(self, db):
        """DB-7.3: Get nested cluster structure."""
        db.upsert_nodes_batch([
            {"node_id": "a"}, {"node_id": "b"}, {"node_id": "c"}, {"node_id": "d"},
        ])
        db.set_clusters_batch([
            ("a", 0, 0), ("b", 0, 1), ("c", 1, 0), ("d", 1, 0),
        ])
        db.conn.commit()

        clusters = db.get_all_clusters()
        assert 0 in clusters
        assert 1 in clusters
        assert len(clusters[1][0]) == 2  # c, d in macro=1, micro=0

    def test_set_hub(self, db):
        """DB-7.4: Hub flagging."""
        db.upsert_node("hub1", symbol_name="Logger")
        db.set_hub("hub1", is_hub=True, assignment="logging_macro_0")
        db.conn.commit()

        node = db.get_node("hub1")
        assert node["is_hub"] == 1
        assert node["hub_assignment"] == "logging_macro_0"


# ═══════════════════════════════════════════════════════════════════════════
# DB-8: Statistics
# ═══════════════════════════════════════════════════════════════════════════

class TestStatistics:
    """DB-8: Stats and diagnostics."""

    def test_stats_empty_db(self, db):
        """DB-8.1: Stats on empty DB."""
        s = db.stats()
        assert s["node_count"] == 0
        assert s["edge_count"] == 0
        assert s["db_size_mb"] >= 0

    def test_stats_with_data(self, db, sample_graph):
        """DB-8.2: Stats reflect imported data."""
        db.from_networkx(sample_graph)
        s = db.stats()
        assert s["node_count"] == sample_graph.number_of_nodes()
        assert s["edge_count"] == sample_graph.number_of_edges()
        assert "python" in s["languages"]
        assert "class" in s["symbol_types"]
        assert s["embedding_dim"] == 4  # from fixture


# ═══════════════════════════════════════════════════════════════════════════
# DB-9: Hybrid search
# ═══════════════════════════════════════════════════════════════════════════

class TestHybridSearch:
    """DB-9: Hybrid RRF search (FTS5-only when vec unavailable)."""

    def test_hybrid_fts_only(self, db, sample_graph):
        """DB-9.1: Hybrid search falls back to FTS5 when no embedding provided."""
        db.from_networkx(sample_graph)
        results = db.search_hybrid("authenticate", embedding=None, limit=5)
        assert len(results) > 0
        # Should find the authenticate function
        ids = {r["node_id"] for r in results}
        assert "python::auth::AuthService::authenticate" in ids

    def test_hybrid_combined_score(self, db, sample_graph):
        """DB-9.2: Results have combined_score field."""
        db.from_networkx(sample_graph)
        results = db.search_hybrid("authentication token")
        for r in results:
            assert "combined_score" in r
            assert r["combined_score"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# DB-10: Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """DB-10: Edge cases and robustness."""

    def test_special_chars_in_node_id(self, db):
        """DB-10.1: Node IDs with special chars work."""
        weird_id = "rust::impl<T>::Display::fmt"
        db.upsert_node(weird_id, symbol_name="fmt", symbol_type="function")
        db.conn.commit()
        assert db.get_node(weird_id) is not None

    def test_unicode_in_source_text(self, db):
        """DB-10.2: Unicode in source text survives roundtrip."""
        db.upsert_node("u1", source_text='# Ünïcödé: "画蛇添足" — 한국어')
        db.conn.commit()
        node = db.get_node("u1")
        assert "Ünïcödé" in node["source_text"]
        assert "画蛇添足" in node["source_text"]

    def test_large_source_text(self, db):
        """DB-10.3: Large source text (100KB) stored correctly."""
        big_text = "x" * 100_000
        db.upsert_node("big", source_text=big_text)
        db.conn.commit()
        assert len(db.get_node("big")["source_text"]) == 100_000

    def test_context_manager(self, db_path):
        """DB-10.4: Context manager usage."""
        with UnifiedWikiDB(db_path) as _db:
            _db.upsert_node("n1", symbol_name="ctx_test")
            _db.conn.commit()
            assert _db.node_count() == 1

    def test_concurrent_reads(self, db_path):
        """DB-10.5: WAL allows concurrent reads."""
        db1 = UnifiedWikiDB(db_path)
        db1.upsert_node("n1", symbol_name="A")
        db1.conn.commit()

        # Open second connection (simulating concurrent reader)
        db2 = UnifiedWikiDB(db_path, readonly=True)
        assert db2.get_node("n1") is not None
        db2.close()
        db1.close()

    def test_vec_serialize_float32(self):
        """DB-10.6: Float32 serialization produces correct length."""
        vec = [1.0, 2.0, 3.0, 4.0]
        blob = _serialize_float32_vec(vec)
        assert len(blob) == 4 * 4  # 4 floats * 4 bytes each

    def test_empty_batch_operations(self, db):
        """DB-10.7: Empty batches are no-ops."""
        db.upsert_nodes_batch([])
        db.upsert_edges_batch([])
        db.upsert_embeddings_batch([])
        db.set_clusters_batch([])
        assert db.node_count() == 0

    def test_edge_weight_update(self, db):
        """DB-10.8: Edge weight updates work."""
        db.upsert_nodes_batch([{"node_id": "a"}, {"node_id": "b"}])
        db.upsert_edge("a", "b", "CALLS", weight=1.0)
        db.conn.commit()

        edges = db.get_edges_from("a")
        edge_id = edges[0]["id"]

        db.update_edge_weights_batch([{"id": edge_id, "weight": 0.5}])
        edges = db.get_edges_from("a")
        assert edges[0]["weight"] == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# 11. populate_embeddings
# ═══════════════════════════════════════════════════════════════════════════

class TestPopulateEmbeddings:
    """Tests for bulk embedding population via populate_embeddings()."""

    def test_basic_populate(self, db):
        """DB-11.1: populate_embeddings embeds all non-empty nodes."""
        if not db.vec_available:
            pytest.skip("sqlite-vec not available (extensions not loadable)")
        db.upsert_nodes_batch([
            {"node_id": "a", "source_text": "class Foo:\n    pass"},
            {"node_id": "b", "source_text": "class Bar:\n    pass"},
            {"node_id": "c", "source_text": ""},  # empty — should be skipped
        ])
        db.conn.commit()

        def mock_embed_docs(texts):
            return [[float(len(t) % 10) / 10, 0.5, 0.3, 0.1] for t in texts]

        count = db.populate_embeddings(mock_embed_docs, batch_size=2)
        assert count == 2  # only a and b (c is empty)

    def test_populate_enables_vec_search(self, db):
        """DB-11.2: After populate, search_vec returns results."""
        if not db.vec_available:
            pytest.skip("sqlite-vec not available (extensions not loadable)")
        db.upsert_nodes_batch([
            {"node_id": "x", "source_text": "hello world", "symbol_name": "X"},
        ])
        db.conn.commit()
        db._populate_fts5()

        def mock_embed_docs(texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        db.populate_embeddings(mock_embed_docs)

        results = db.search_vec([0.1, 0.2, 0.3, 0.4], k=5)
        assert len(results) > 0
        assert results[0]["node_id"] == "x"

    def test_populate_handles_batch_failure(self, db):
        """DB-11.3: One failing batch doesn't stop the rest."""
        if not db.vec_available:
            pytest.skip("sqlite-vec not available (extensions not loadable)")
        db.upsert_nodes_batch([
            {"node_id": "a", "source_text": "text a"},
            {"node_id": "b", "source_text": "text b"},
            {"node_id": "c", "source_text": "text c"},
        ])
        db.conn.commit()

        call_count = [0]

        def failing_embed(texts):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("rate limited")
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        # batch_size=1 so each node is its own batch
        count = db.populate_embeddings(failing_embed, batch_size=1)
        assert count == 2  # first batch fails, other 2 succeed

    def test_populate_returns_zero_without_vec(self, tmp_path):
        """DB-11.4: Returns 0 when sqlite-vec is unavailable."""
        db = UnifiedWikiDB(tmp_path / "test.wiki.db", embedding_dim=4)
        db.upsert_nodes_batch([{"node_id": "a", "source_text": "hello"}])
        db.conn.commit()

        # Force vec unavailable
        original = db._vec_available
        db._vec_available = False
        count = db.populate_embeddings(lambda texts: [[0.0] * 4] * len(texts))
        assert count == 0
        db._vec_available = original
        db.close()

    def test_populate_skips_whitespace_only_text(self, db):
        """DB-11.5: Nodes with whitespace-only source_text are skipped."""
        if not db.vec_available:
            pytest.skip("sqlite-vec not available (extensions not loadable)")
        db.upsert_nodes_batch([
            {"node_id": "a", "source_text": "   \n\t  "},
            {"node_id": "b", "source_text": "real content"},
        ])
        db.conn.commit()

        count = db.populate_embeddings(
            lambda texts: [[0.1, 0.2, 0.3, 0.4] for _ in texts]
        )
        assert count == 1  # only b
