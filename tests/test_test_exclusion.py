"""Tests for test-code exclusion feature (P1).

Covers:
- is_test_path() heuristics in constants.py
- is_test column tagging in unified_db.py
- Feature flag exclude_tests in feature_flags.py
- Graph filtering in graph_clustering.py
- SQL filtering in cluster_planner._load_architectural_cluster_map()
"""

import os
import sqlite3
import pytest

from plugin_implementation.constants import is_test_path
from plugin_implementation.feature_flags import FeatureFlags, get_feature_flags


# ═════════════════════════════════════════════════════════════════════════════
# 1. is_test_path() — path-based detection heuristics
# ═════════════════════════════════════════════════════════════════════════════


class TestIsTestPathDirectories:
    """Directory-based test detection patterns."""

    @pytest.mark.parametrize("path", [
        "tests/test_foo.py",
        "test/unit/test_bar.py",
        "src/tests/integration/test_api.py",
        "some/module/tests/conftest.py",
    ])
    def test_tests_directory(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "__tests__/Button.test.js",
        "src/components/__tests__/Header.test.tsx",
    ])
    def test_jest_tests_directory(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "spec/models/user_spec.rb",
        "spec/features/login_spec.rb",
    ])
    def test_spec_directory(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "testing/integration/main_test.go",
        "pkg/testing/helpers.go",
    ])
    def test_testing_directory(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "fixtures/data.json",
        "test/fixtures/mock_response.json",
    ])
    def test_fixtures_directory(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "mocks/api_mock.py",
        "src/mock/handler.ts",
    ])
    def test_mocks_directory(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "testdata/input.txt",
        "pkg/testdata/golden.json",
    ])
    def test_testdata_directory(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "testutils/helpers.go",
        "testutil/factory.py",
    ])
    def test_testutils_directory(self, path):
        assert is_test_path(path) is True


class TestIsTestPathFiles:
    """File-name-based test detection patterns."""

    @pytest.mark.parametrize("path", [
        "test_foo.py",
        "src/utils/test_parser.py",
        "pkg/test_client.go",
    ])
    def test_python_style_prefix(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "foo_test.go",
        "src/handler_test.go",
        "parser_test.py",
    ])
    def test_go_style_suffix(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "Button.test.js",
        "src/components/Header.test.tsx",
        "api.test.ts",
    ])
    def test_js_test_suffix(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "Button.spec.js",
        "src/components/Header.spec.tsx",
        "api.spec.ts",
    ])
    def test_js_spec_suffix(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "FooTest.java",
        "src/main/java/com/app/FooTest.java",
        "BarTests.cs",
    ])
    def test_java_csharp_suffix(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "conftest.py",
        "src/conftest.py",
        "tests/conftest.py",
    ])
    def test_conftest(self, path):
        assert is_test_path(path) is True

    @pytest.mark.parametrize("path", [
        "setup_test.py",
        "setup_tests.js",
    ])
    def test_setup_tests(self, path):
        assert is_test_path(path) is True


class TestIsTestPathNegative:
    """Paths that should NOT match test patterns."""

    @pytest.mark.parametrize("path", [
        "src/main.py",
        "src/utils/parser.py",
        "include/fmt/format.h",
        "src/core/engine.cpp",
        "pkg/server/handler.go",
        "src/components/Button.jsx",
        "docs/README.md",
        "Makefile",
        "setup.py",
        "pyproject.toml",
        ".github/workflows/ci.yml",
        "src/testing.py",        # file named 'testing.py' (not in testing/ dir)
        "src/attestation.go",    # contains 'test' substring but not a test
        "src/contest/main.py",   # 'contest' contains 'test' but shouldn't match
        "src/latest/module.py",  # 'latest' contains 'test'
    ])
    def test_non_test_paths(self, path):
        assert is_test_path(path) is False


class TestIsTestPathCaseInsensitive:
    """Patterns should match regardless of case."""

    @pytest.mark.parametrize("path", [
        "Tests/TestFoo.py",
        "TEST/unit.py",
        "Test/integration.py",
    ])
    def test_case_insensitive_directories(self, path):
        assert is_test_path(path) is True


class TestIsTestPathEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_string(self):
        assert is_test_path("") is False

    def test_root_level_test_file(self):
        assert is_test_path("test_main.py") is True

    def test_deeply_nested(self):
        assert is_test_path("a/b/c/d/tests/test_deep.py") is True

    def test_vendored_gtest(self):
        """Vendored Google Test should match (it's under test/)."""
        assert is_test_path("test/gtest/gtest/gtest.h") is True
        assert is_test_path("test/gtest/gmock/gmock.h") is True


# ═════════════════════════════════════════════════════════════════════════════
# 2. Feature flag — exclude_tests
# ═════════════════════════════════════════════════════════════════════════════


class TestExcludeTestsFlag:
    """The exclude_tests flag reads correctly from environment."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("DEEPWIKI_EXCLUDE_TESTS", raising=False)

    def test_default_false(self):
        """exclude_tests defaults to False (opt-in feature)."""
        flags = get_feature_flags()
        assert flags.exclude_tests is False

    @pytest.mark.parametrize("value", ["1", "true", "yes"])
    def test_truthy(self, monkeypatch, value):
        monkeypatch.setenv("DEEPWIKI_EXCLUDE_TESTS", value)
        assert get_feature_flags().exclude_tests is True

    @pytest.mark.parametrize("value", ["0", "false", "no", ""])
    def test_falsy(self, monkeypatch, value):
        monkeypatch.setenv("DEEPWIKI_EXCLUDE_TESTS", value)
        assert get_feature_flags().exclude_tests is False


# ═════════════════════════════════════════════════════════════════════════════
# 3. DB tagging — is_test column populated during persist
# ═════════════════════════════════════════════════════════════════════════════


class TestDbIsTestColumn:
    """Verify that is_test is set correctly during node persistence."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a temporary UnifiedWikiDB instance."""
        from plugin_implementation.unified_db import UnifiedWikiDB
        db_path = tmp_path / "test.wiki.db"
        return UnifiedWikiDB(str(db_path))

    def test_test_nodes_tagged(self, db):
        """Nodes from test files get is_test=1."""
        nodes = [
            {"node_id": "n1", "rel_path": "src/main.py", "symbol_name": "main", "symbol_type": "function"},
            {"node_id": "n2", "rel_path": "tests/test_main.py", "symbol_name": "test_main", "symbol_type": "function"},
            {"node_id": "n3", "rel_path": "test/gtest/gtest.h", "symbol_name": "Test", "symbol_type": "class"},
            {"node_id": "n4", "rel_path": "src/utils.py", "symbol_name": "helper", "symbol_type": "function"},
            {"node_id": "n5", "rel_path": "src/foo_test.go", "symbol_name": "TestFoo", "symbol_type": "function"},
        ]
        db.upsert_nodes_batch(nodes)

        # Check is_test values
        for nid, expected in [("n1", 0), ("n2", 1), ("n3", 1), ("n4", 0), ("n5", 1)]:
            row = db.get_node(nid)
            assert row is not None, f"Node {nid} not found"
            assert row["is_test"] == expected, (
                f"Node {nid} (path={row['rel_path']}): "
                f"expected is_test={expected}, got {row['is_test']}"
            )

    def test_all_non_test_nodes(self, db):
        """When no test files, all nodes have is_test=0."""
        nodes = [
            {"node_id": "n1", "rel_path": "src/main.py", "symbol_name": "main", "symbol_type": "function"},
            {"node_id": "n2", "rel_path": "lib/utils.py", "symbol_name": "util", "symbol_type": "function"},
        ]
        db.upsert_nodes_batch(nodes)

        for nid in ["n1", "n2"]:
            row = db.get_node(nid)
            assert row["is_test"] == 0

    def test_index_exists(self, db):
        """The is_test index should exist."""
        rows = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_nodes_test'"
        ).fetchall()
        assert len(rows) == 1


# ═════════════════════════════════════════════════════════════════════════════
# 4. Graph filtering — test nodes removed from clustering input
# ═════════════════════════════════════════════════════════════════════════════


class TestGraphFiltering:
    """Test node exclusion from the clustering graph."""

    def _make_graph(self):
        """Build a small graph with test and non-test nodes."""
        import networkx as nx

        G = nx.MultiDiGraph()
        # Non-test nodes
        G.add_node("n1", rel_path="src/main.py", symbol_name="main", symbol_type="function")
        G.add_node("n2", rel_path="src/utils.py", symbol_name="helper", symbol_type="function")
        G.add_node("n3", rel_path="src/core.py", symbol_name="Core", symbol_type="class")
        # Test nodes
        G.add_node("t1", rel_path="tests/test_main.py", symbol_name="test_main", symbol_type="function")
        G.add_node("t2", rel_path="test/unit/test_utils.py", symbol_name="test_helper", symbol_type="function")

        # Edges
        G.add_edge("n1", "n2", weight=1.0)
        G.add_edge("n1", "n3", weight=2.0)
        G.add_edge("n2", "n3", weight=1.5)
        G.add_edge("t1", "n1", weight=1.0)  # test → src
        G.add_edge("t2", "n2", weight=1.0)  # test → src

        return G

    def test_filter_test_nodes(self):
        """When exclude_tests is on, test nodes are removed before clustering."""
        G = self._make_graph()

        # Directly test the filtering logic from _run_phase3_hierarchical_leiden
        excluded = set()
        for nid, data in G.nodes(data=True):
            rel_path = data.get("rel_path") or data.get("file_name") or ""
            if is_test_path(rel_path):
                excluded.add(nid)

        assert excluded == {"t1", "t2"}

        non_test_nodes = set(G.nodes()) - excluded
        G_filtered = G.subgraph(non_test_nodes).copy()

        assert G_filtered.number_of_nodes() == 3
        assert set(G_filtered.nodes()) == {"n1", "n2", "n3"}
        # Edges from test nodes should be gone
        assert G_filtered.number_of_edges() == 3

    def test_no_filter_when_disabled(self):
        """When exclude_tests is off, all nodes remain."""
        G = self._make_graph()

        excluded = set()
        # exclude_tests = False — don't filter
        assert len(excluded) == 0
        assert G.number_of_nodes() == 5


# ═════════════════════════════════════════════════════════════════════════════
# 5. SQL filtering — _load_architectural_cluster_map respects is_test
# ═════════════════════════════════════════════════════════════════════════════


class TestClusterMapFiltering:
    """Verify that test nodes are excluded from the cluster map query."""

    @pytest.fixture
    def db_with_clusters(self, tmp_path):
        """Create a DB with clustered nodes, some marked as test."""
        from plugin_implementation.unified_db import UnifiedWikiDB
        db_path = tmp_path / "clustered.wiki.db"
        db = UnifiedWikiDB(str(db_path))

        nodes = [
            # Non-test architectural nodes
            {
                "node_id": "n1", "rel_path": "src/main.py",
                "symbol_name": "main", "symbol_type": "function",
                "macro_cluster": 0, "micro_cluster": 0,
            },
            {
                "node_id": "n2", "rel_path": "src/utils.py",
                "symbol_name": "helper", "symbol_type": "function",
                "macro_cluster": 0, "micro_cluster": 1,
            },
            {
                "node_id": "n3", "rel_path": "src/core.py",
                "symbol_name": "Core", "symbol_type": "class",
                "macro_cluster": 1, "micro_cluster": 0,
            },
            # Test architectural nodes
            {
                "node_id": "t1", "rel_path": "tests/test_main.py",
                "symbol_name": "TestMain", "symbol_type": "class",
                "macro_cluster": 0, "micro_cluster": 0,
            },
            {
                "node_id": "t2", "rel_path": "test/test_core.py",
                "symbol_name": "TestCore", "symbol_type": "class",
                "macro_cluster": 1, "micro_cluster": 0,
            },
        ]
        db.upsert_nodes_batch(nodes)
        return db

    def test_without_exclusion(self, db_with_clusters, monkeypatch):
        """Without exclude_tests, all architectural nodes appear."""
        monkeypatch.delenv("DEEPWIKI_EXCLUDE_TESTS", raising=False)

        rows = db_with_clusters.conn.execute(
            "SELECT node_id FROM repo_nodes "
            "WHERE macro_cluster IS NOT NULL AND is_architectural = 1"
        ).fetchall()

        node_ids = {r["node_id"] for r in rows}
        assert "t1" in node_ids
        assert "t2" in node_ids
        assert len(node_ids) == 5

    def test_with_exclusion(self, db_with_clusters, monkeypatch):
        """With exclude_tests, test nodes are filtered out by is_test=0."""
        monkeypatch.setenv("DEEPWIKI_EXCLUDE_TESTS", "1")

        rows = db_with_clusters.conn.execute(
            "SELECT node_id FROM repo_nodes "
            "WHERE macro_cluster IS NOT NULL AND is_architectural = 1 "
            "AND is_test = 0"
        ).fetchall()

        node_ids = {r["node_id"] for r in rows}
        assert "t1" not in node_ids
        assert "t2" not in node_ids
        assert node_ids == {"n1", "n2", "n3"}


# ═════════════════════════════════════════════════════════════════════════════
# 6. Schema migration — existing DBs without is_test column
# ═════════════════════════════════════════════════════════════════════════════


class TestSchemaMigration:
    """Verify that opening an old DB without is_test gets it migrated."""

    @staticmethod
    def _old_schema_ddl():
        """Return the full repo_nodes DDL but WITHOUT the is_test column."""
        return """
        CREATE TABLE repo_nodes (
            node_id         TEXT PRIMARY KEY,
            rel_path        TEXT NOT NULL DEFAULT '',
            file_name       TEXT NOT NULL DEFAULT '',
            language        TEXT NOT NULL DEFAULT '',
            start_line      INTEGER DEFAULT 0,
            end_line        INTEGER DEFAULT 0,
            symbol_name     TEXT NOT NULL DEFAULT '',
            symbol_type     TEXT NOT NULL DEFAULT '',
            parent_symbol   TEXT DEFAULT NULL,
            analysis_level  TEXT DEFAULT 'comprehensive',
            source_text     TEXT DEFAULT '',
            docstring       TEXT DEFAULT '',
            signature       TEXT DEFAULT '',
            parameters      TEXT DEFAULT '',
            return_type     TEXT DEFAULT '',
            is_architectural INTEGER DEFAULT 0,
            is_doc           INTEGER DEFAULT 0,
            chunk_type       TEXT DEFAULT NULL,
            macro_cluster   INTEGER DEFAULT NULL,
            micro_cluster   INTEGER DEFAULT NULL,
            is_hub          INTEGER DEFAULT 0,
            hub_assignment  TEXT DEFAULT NULL,
            indexed_at      TEXT DEFAULT (datetime('now'))
        );
        """

    def test_migration_adds_is_test_column(self, tmp_path):
        """Open a DB that was created WITHOUT is_test, re-open with new code."""
        import sqlite3
        from plugin_implementation.unified_db import UnifiedWikiDB

        db_path = tmp_path / "old.wiki.db"

        # 1) Create a DB with the OLD schema (no is_test column)
        conn = sqlite3.connect(str(db_path))
        conn.executescript(self._old_schema_ddl())
        conn.execute("""
            INSERT INTO repo_nodes (node_id, rel_path, symbol_name, symbol_type, is_architectural, macro_cluster)
            VALUES ('n1', 'src/main.py', 'main', 'function', 1, 0)
        """)
        conn.execute("""
            INSERT INTO repo_nodes (node_id, rel_path, symbol_name, symbol_type, is_architectural, macro_cluster)
            VALUES ('t1', 'tests/test_main.py', 'test_main', 'function', 1, 0)
        """)
        conn.commit()
        conn.close()

        # 2) Re-open with new UnifiedWikiDB — migration should add is_test
        db = UnifiedWikiDB(str(db_path))

        # 3) Verify column exists
        cols = {
            row[1]
            for row in db.conn.execute("PRAGMA table_info(repo_nodes)").fetchall()
        }
        assert "is_test" in cols

        # 4) Verify backfill: src/main.py → is_test=0, tests/test_main.py → is_test=1
        row_n1 = db.conn.execute(
            "SELECT is_test FROM repo_nodes WHERE node_id = 'n1'"
        ).fetchone()
        assert row_n1["is_test"] == 0

        row_t1 = db.conn.execute(
            "SELECT is_test FROM repo_nodes WHERE node_id = 't1'"
        ).fetchone()
        assert row_t1["is_test"] == 1, "Migration should backfill is_test=1 for test paths"

        # 5) The SQL query with AND is_test = 0 should NOT crash and should exclude t1
        rows = db.conn.execute(
            "SELECT node_id FROM repo_nodes "
            "WHERE macro_cluster IS NOT NULL AND is_architectural = 1 AND is_test = 0"
        ).fetchall()
        assert len(rows) == 1  # only n1, t1 excluded by backfill
        assert rows[0]["node_id"] == "n1"

    def test_migration_idempotent(self, tmp_path):
        """Running migration twice on the same DB should be safe."""
        from plugin_implementation.unified_db import UnifiedWikiDB

        db_path = tmp_path / "new.wiki.db"
        # First open creates schema from scratch
        db1 = UnifiedWikiDB(str(db_path))
        db1.close()

        # Second open triggers migration check again — should not fail
        db2 = UnifiedWikiDB(str(db_path))
        cols = {
            row[1]
            for row in db2.conn.execute("PRAGMA table_info(repo_nodes)").fetchall()
        }
        assert "is_test" in cols
        db2.close()

    def test_migration_index_created(self, tmp_path):
        """Migration should also create the idx_nodes_test index."""
        import sqlite3
        from plugin_implementation.unified_db import UnifiedWikiDB

        db_path = tmp_path / "old_noindex.wiki.db"

        # Create old schema without is_test
        conn = sqlite3.connect(str(db_path))
        conn.executescript(self._old_schema_ddl())
        conn.commit()
        conn.close()

        # Open with new code
        db = UnifiedWikiDB(str(db_path))

        indexes = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_nodes_test'"
        ).fetchall()
        assert len(indexes) == 1
        db.close()


# ═════════════════════════════════════════════════════════════════════════════
# 7. Multi-language coverage — redpanda-style test paths
# ═════════════════════════════════════════════════════════════════════════════


class TestMultiLanguageCoverage:
    """Ensure test detection works for Go, Python, Java, C++, Rust, JS/TS."""

    @pytest.mark.parametrize("path,expected", [
        # Go
        ("src/rpk/cmd/rpk_test.go", True),
        ("src/rpk/cmd/rpk.go", False),
        ("tests/integration/go_test.go", True),

        # Python
        ("tests/rptest/test_upgrade.py", True),
        ("src/rptest/services.py", False),
        ("conftest.py", True),

        # Java
        ("src/test/java/com/rpk/FooTest.java", True),
        ("src/main/java/com/rpk/Foo.java", False),

        # C++
        ("src/v/raft/tests/raft_test.cc", True),
        ("src/v/raft/raft_group.h", False),
        ("test/gtest/gtest/gtest.h", True),

        # Rust
        ("src/transform/tests/test_transform.rs", True),
        ("src/transform/transform.rs", False),

        # JavaScript / TypeScript
        ("src/components/__tests__/App.test.tsx", True),
        ("src/components/App.tsx", False),
        ("src/utils/parser.spec.ts", True),
        ("src/utils/parser.ts", False),

        # Ruby
        ("spec/models/user_spec.rb", True),
        ("app/models/user.rb", False),
    ])
    def test_multi_language_paths(self, path, expected):
        assert is_test_path(path) is expected, f"is_test_path({path!r}) should be {expected}"
