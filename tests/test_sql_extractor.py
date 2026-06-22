"""Tests for the SQL/DDL graph extractor (roadmap C1/§3.1).

These exercise ``build_sql_graph`` directly: typed nodes (schema/table/view/
column/index/function/trigger) and edges (defines/references/triggered_by/
calls), plus the lightweight DDL parser's robustness to comments, dollar-quoted
function bodies, quoted identifiers, multi-file FK/view resolution, and the
feature flag.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from plugin_implementation.code_graph.sql_extractor import build_sql_graph
from plugin_implementation.feature_flags import FeatureFlags


def _write(tmpdir: str, name: str, content: str) -> str:
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def _edges_by_type(graph, rel_type):
    """Return list of (source_name, target_name) for the given relationship_type."""
    out = []
    for u, v, data in graph.edges(data=True):
        if data.get("relationship_type") == rel_type:
            out.append((graph.nodes[u]["name"], graph.nodes[v]["name"]))
    return out


def _nodes_by_type(graph, symbol_type):
    return sorted(
        data["name"]
        for _, data in graph.nodes(data=True)
        if data.get("symbol_type") == symbol_type
    )


# ──────────────────────────────────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────────────────────────────────


def test_table_columns_and_inline_fk():
    sql = """
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        email TEXT UNIQUE
    );

    CREATE TABLE orders (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        total NUMERIC(10, 2)
    );
    """
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "schema.sql", sql)
        graph, stats = build_sql_graph([path], d)

    assert _nodes_by_type(graph, "sql_table") == ["orders", "users"]
    # columns defined under their tables
    defines = _edges_by_type(graph, "defines")
    assert ("users", "id") in defines
    assert ("users", "name") in defines
    assert ("users", "email") in defines
    assert ("orders", "user_id") in defines
    # inline FK column → column
    refs = _edges_by_type(graph, "references")
    assert ("user_id", "id") in refs
    assert stats["tables"] == 2
    assert stats["columns"] == 6


def test_table_level_foreign_key_constraint():
    sql = """
    CREATE TABLE users (id INTEGER PRIMARY KEY);
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "schema.sql", sql)
        graph, _ = build_sql_graph([path], d)

    refs = _edges_by_type(graph, "references")
    assert ("user_id", "id") in refs
    # the table-level constraint must NOT become a column named "CONSTRAINT"/"fk_user"
    assert "fk_user" not in _nodes_by_type(graph, "sql_column")


def test_view_references_tables():
    sql = """
    CREATE TABLE users (id INTEGER);
    CREATE TABLE orders (id INTEGER, user_id INTEGER);
    CREATE VIEW user_orders AS
        SELECT u.id, o.id FROM users u JOIN orders o ON o.user_id = u.id;
    """
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "schema.sql", sql)
        graph, _ = build_sql_graph([path], d)

    assert _nodes_by_type(graph, "sql_view") == ["user_orders"]
    refs = _edges_by_type(graph, "references")
    assert ("user_orders", "users") in refs
    assert ("user_orders", "orders") in refs


def test_index_defines_edge():
    sql = """
    CREATE TABLE orders (id INTEGER, user_id INTEGER);
    CREATE INDEX idx_orders_user ON orders (user_id);
    """
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "schema.sql", sql)
        graph, _ = build_sql_graph([path], d)

    assert _nodes_by_type(graph, "sql_index") == ["idx_orders_user"]
    assert ("orders", "idx_orders_user") in _edges_by_type(graph, "defines")


def test_trigger_triggered_by_table():
    sql = """
    CREATE TABLE orders (id INTEGER);
    CREATE TRIGGER trg_orders AFTER INSERT ON orders
        FOR EACH ROW EXECUTE FUNCTION audit_fn();
    """
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "schema.sql", sql)
        graph, _ = build_sql_graph([path], d)

    assert _nodes_by_type(graph, "sql_trigger") == ["trg_orders"]
    assert ("trg_orders", "orders") in _edges_by_type(graph, "triggered_by")


def test_function_calls_function():
    sql = """
    CREATE FUNCTION helper_sum(uid integer) RETURNS numeric AS $$
        SELECT 1
    $$ LANGUAGE sql;

    CREATE OR REPLACE FUNCTION total_for(uid integer) RETURNS numeric AS $$
    BEGIN
        RETURN helper_sum(uid);
    END;
    $$ LANGUAGE plpgsql;
    """
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "schema.sql", sql)
        graph, _ = build_sql_graph([path], d)

    assert sorted(_nodes_by_type(graph, "sql_function")) == ["helper_sum", "total_for"]
    assert ("total_for", "helper_sum") in _edges_by_type(graph, "calls")


def test_schema_defines_table():
    sql = """
    CREATE SCHEMA app;
    CREATE TABLE app.users (id INTEGER PRIMARY KEY);
    """
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "schema.sql", sql)
        graph, _ = build_sql_graph([path], d)

    assert _nodes_by_type(graph, "sql_schema") == ["app"]
    assert ("app", "users") in _edges_by_type(graph, "defines")


# ──────────────────────────────────────────────────────────────────────────
# Robustness
# ──────────────────────────────────────────────────────────────────────────


def test_comments_are_ignored():
    sql = """
    -- a leading comment mentioning CREATE TABLE ghost (x int);
    CREATE TABLE users (
        id INTEGER, /* inline block REFERENCES nothing(id) */
        name TEXT  -- trailing comment
    );
    """
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "schema.sql", sql)
        graph, _ = build_sql_graph([path], d)

    assert _nodes_by_type(graph, "sql_table") == ["users"]
    assert "ghost" not in _nodes_by_type(graph, "sql_table")
    assert _edges_by_type(graph, "references") == []


def test_dollar_quoted_body_does_not_break_statement_split():
    sql = """
    CREATE FUNCTION f1() RETURNS void AS $$
        -- this body contains a semicolon; and the word CREATE TABLE fake (x int);
        BEGIN END;
    $$ LANGUAGE plpgsql;

    CREATE TABLE real_table (id INTEGER);
    """
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "schema.sql", sql)
        graph, _ = build_sql_graph([path], d)

    assert "real_table" in _nodes_by_type(graph, "sql_table")
    assert "fake" not in _nodes_by_type(graph, "sql_table")
    assert "f1" in _nodes_by_type(graph, "sql_function")


def test_quoted_identifiers_are_normalized():
    sql = '''
    CREATE TABLE "User" ("Id" INTEGER PRIMARY KEY);
    CREATE TABLE `order` (id INTEGER, owner INTEGER REFERENCES "User"("Id"));
    '''
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "schema.sql", sql)
        graph, _ = build_sql_graph([path], d)

    tables = _nodes_by_type(graph, "sql_table")
    assert "User" in tables
    assert "order" in tables
    assert ("owner", "Id") in _edges_by_type(graph, "references")


def test_cross_file_foreign_key_resolution():
    users_sql = "CREATE TABLE users (id INTEGER PRIMARY KEY);"
    orders_sql = """
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER REFERENCES users(id)
    );
    """
    with tempfile.TemporaryDirectory() as d:
        p1 = _write(d, "users.sql", users_sql)
        p2 = _write(d, "orders.sql", orders_sql)
        graph, _ = build_sql_graph([p1, p2], d)

    assert ("user_id", "id") in _edges_by_type(graph, "references")


# ──────────────────────────────────────────────────────────────────────────
# Edge metadata + flag
# ──────────────────────────────────────────────────────────────────────────


def test_edge_and_node_metadata():
    sql = "CREATE TABLE users (id INTEGER PRIMARY KEY);"
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "schema.sql", sql)
        graph, _ = build_sql_graph([path], d)

    assert graph.graph.get("language") == "sql"
    # node provenance
    for _, data in graph.nodes(data=True):
        assert data["language"] == "sql"
        assert data["analysis_level"] == "code"
    # edge provenance
    for _, _, data in graph.edges(data=True):
        assert data["edge_class"] == "structural"
        assert data["language"] == "sql"
        assert data["created_by"] == "sql_extractor"
        assert data["analysis_level"] == "code"
        assert data["annotations"]["confidence"] == "EXTRACTED"


def test_feature_flag_disabled_returns_empty():
    sql = "CREATE TABLE users (id INTEGER PRIMARY KEY);"
    with tempfile.TemporaryDirectory() as d:
        path = _write(d, "schema.sql", sql)
        flags = FeatureFlags(sql_extraction=False)
        graph, stats = build_sql_graph([path], d, flags=flags)

    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0
    assert stats.get("tables", 0) == 0


def test_empty_input_returns_empty_graph():
    with tempfile.TemporaryDirectory() as d:
        graph, stats = build_sql_graph([], d)
    assert graph.number_of_nodes() == 0
    assert stats.get("sql_files", 0) == 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
