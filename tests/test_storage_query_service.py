"""Regression tests for UnifiedWikiDB-backed research query tools."""

import os
from unittest.mock import patch

import pytest

from plugin_implementation.deep_research.research_tools import create_codebase_tools
from plugin_implementation.repo_resolution import resolve_unified_db_path, save_cache_index_atomic
from plugin_implementation.unified_db import UnifiedWikiDB


@pytest.fixture
def query_db(tmp_path):
    """Create a small unified DB with code, method, field, doc and edge rows."""
    db = UnifiedWikiDB(str(tmp_path / "query.wiki.db"), embedding_dim=4)
    db._upsert_nodes_batch([
        {
            "node_id": "auth::AuthService",
            "rel_path": "src/auth/service.py",
            "file_name": "service.py",
            "language": "python",
            "start_line": 10,
            "end_line": 80,
            "symbol_name": "AuthService",
            "symbol_type": "class",
            "source_text": "class AuthService:\n    def login(self): ...",
            "docstring": "Coordinates authentication tokens.",
            "signature": "class AuthService",
            "is_architectural": 1,
            "is_doc": 0,
        },
        {
            "node_id": "auth::TokenManager",
            "rel_path": "src/auth/tokens.py",
            "file_name": "tokens.py",
            "language": "python",
            "start_line": 5,
            "end_line": 50,
            "symbol_name": "TokenManager",
            "symbol_type": "class",
            "source_text": "class TokenManager:\n    def create_token(self): ...",
            "docstring": "Creates authentication tokens.",
            "signature": "class TokenManager",
            "is_architectural": 1,
            "is_doc": 0,
        },
        {
            "node_id": "auth::login",
            "rel_path": "src/auth/service.py",
            "file_name": "service.py",
            "language": "python",
            "start_line": 20,
            "end_line": 30,
            "symbol_name": "login",
            "symbol_type": "method",
            "source_text": "def login(self, user): return self.tokens.create(user)",
            "docstring": "Logs in a user with an authentication token.",
            "signature": "def login(self, user)",
            "is_architectural": 0,
            "is_doc": 0,
        },
        {
            "node_id": "auth::token_value",
            "rel_path": "src/auth/service.py",
            "file_name": "service.py",
            "language": "python",
            "start_line": 32,
            "end_line": 32,
            "symbol_name": "token_value",
            "symbol_type": "field",
            "source_text": "token_value = raw_token",
            "docstring": "Low-level token storage field.",
            "signature": "token_value",
            "is_architectural": 0,
            "is_doc": 0,
        },
        {
            "node_id": "docs::auth",
            "rel_path": "README.md",
            "file_name": "README.md",
            "language": "markdown",
            "symbol_name": "Authentication Guide",
            "symbol_type": "markdown_document",
            "source_text": "# Authentication Guide\nToken docs.",
            "docstring": "",
            "signature": "",
            "is_architectural": 0,
            "is_doc": 1,
        },
    ])
    db.upsert_edge("auth::AuthService", "auth::TokenManager", "calls")
    db.upsert_edge("auth::AuthService", "auth::login", "defines")
    db.conn.commit()
    db._populate_fts5()

    yield db
    db.close()


def _tool(tools, name):
    return next(tool for tool in tools if tool.name == name)


def test_resolve_unified_db_path_uses_cache_index_not_latest_mtime(tmp_path):
    requested = tmp_path / "requested.wiki.db"
    unrelated = tmp_path / "unrelated_doconly.wiki.db"
    requested.write_text("requested", encoding="utf-8")
    unrelated.write_text("unrelated", encoding="utf-8")
    os.utime(requested, (100, 100))
    os.utime(unrelated, (200, 200))

    save_cache_index_atomic(tmp_path, {
        "unified_db": {
            "epameliteatest/TestProject/TestProject:mb-java:6f785643": "requested",
            "some/other/repo:main:abcdef12": "unrelated_doconly",
        },
    })

    resolved = resolve_unified_db_path(
        canonical_repo_id="epameliteatest/TestProject/TestProject:mb-java:6f785643",
        cache_dir=tmp_path,
    )

    assert resolved == str(requested)


def test_query_graph_uses_unified_db_query_service(query_db):
    from plugin_implementation.code_graph.storage_query_service import StorageQueryService

    tools = create_codebase_tools(
        retriever_stack=None,
        graph_manager=None,
        code_graph=None,
        query_service=StorageQueryService(query_db),
    )

    with patch.dict(os.environ, {"DEEPWIKI_PROGRESSIVE_TOOLS": "1"}):
        progressive_tools = create_codebase_tools(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            query_service=StorageQueryService(query_db),
        )

    result = _tool(progressive_tools, "query_graph").invoke({"expression": "type:class text:auth limit:5"})

    assert "Graph query service not available" not in result
    assert "AuthService" in result


def test_search_graph_uses_unified_db_relationships(query_db):
    from plugin_implementation.code_graph.storage_query_service import StorageQueryService

    tools = create_codebase_tools(
        retriever_stack=None,
        graph_manager=None,
        code_graph=None,
        query_service=StorageQueryService(query_db),
    )
    result = _tool(tools, "search_graph").invoke({"query": "AuthService", "k": 3})

    assert "neighbors unavailable" not in result
    assert "Relationships" in result
    assert "TokenManager" in result


def test_search_symbols_limits_discovery_to_arch_symbols_and_methods(query_db):
    from plugin_implementation.code_graph.storage_query_service import StorageQueryService

    with patch.dict(os.environ, {"DEEPWIKI_PROGRESSIVE_TOOLS": "1"}):
        tools = create_codebase_tools(
            retriever_stack=None,
            graph_manager=None,
            code_graph=None,
            query_service=StorageQueryService(query_db),
        )

    search_symbols = _tool(tools, "search_symbols")

    token_result = search_symbols.invoke({"query": "token", "k": 10})
    assert "TokenManager" in token_result
    assert "token_value" not in token_result
    assert "Authentication Guide" not in token_result

    method_result = search_symbols.invoke({"query": "login", "k": 10})
    assert "login" in method_result
