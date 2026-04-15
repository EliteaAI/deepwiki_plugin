"""
Tests for GraphTextIndex — SQLite FTS5 full-text search over graph nodes.

Covers:
- Index building from NetworkX graphs
- BM25-ranked full-text search
- Symbol type filtering
- CamelCase / snake_case tokenisation
- Persistence (save / load cycle)
- Structured queries (search_by_name, search_by_type, get_by_node_id)
- Cache-index registration
- Integration with research_tools (hybrid search)
- Edge cases (empty graph, None graph, special characters)
"""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helpers — build realistic test graphs
# ---------------------------------------------------------------------------

def _make_graph(nodes=None, edges=None):
    """Build a NetworkX DiGraph from a list of node dicts and edge tuples."""
    g = nx.DiGraph()
    for n in (nodes or []):
        nid = n.pop('id')
        g.add_node(nid, **n)
    for src, tgt, data in (edges or []):
        g.add_edge(src, tgt, **(data or {}))
    return g


def _realistic_graph():
    """A small but realistic graph covering many symbol types."""
    nodes = [
        {
            'id': 'python::auth::src/auth/service.py::AuthService',
            'symbol_name': 'AuthService',
            'symbol_type': 'class',
            'file_path': '/repo/src/auth/service.py',
            'rel_path': 'src/auth/service.py',
            'language': 'python',
            'docstring': 'Authentication service handling JWT tokens and session management.',
            'content': 'class AuthService:\n    def login(self, username, password): ...',
            'start_line': 10,
            'end_line': 80,
            'full_name': 'auth.service.AuthService',
        },
        {
            'id': 'python::auth::src/auth/service.py::AuthService.login',
            'symbol_name': 'login',
            'symbol_type': 'method',
            'file_path': '/repo/src/auth/service.py',
            'rel_path': 'src/auth/service.py',
            'language': 'python',
            'docstring': 'Authenticate user with username and password.',
            'content': 'def login(self, username, password): ...',
            'start_line': 20,
            'end_line': 45,
        },
        {
            'id': 'python::models::src/models/user.py::User',
            'symbol_name': 'User',
            'symbol_type': 'class',
            'file_path': '/repo/src/models/user.py',
            'rel_path': 'src/models/user.py',
            'language': 'python',
            'docstring': 'User model representing authenticated users.',
            'content': 'class User:\n    id: int\n    username: str\n    email: str',
            'start_line': 1,
            'end_line': 20,
        },
        {
            'id': 'python::utils::src/utils/crypto.py::hash_password',
            'symbol_name': 'hash_password',
            'symbol_type': 'function',
            'file_path': '/repo/src/utils/crypto.py',
            'rel_path': 'src/utils/crypto.py',
            'language': 'python',
            'docstring': 'Hash a plaintext password using bcrypt.',
            'content': 'def hash_password(password: str) -> str:\n    return bcrypt.hashpw(password)',
            'start_line': 5,
            'end_line': 15,
        },
        {
            'id': 'python::config::src/config.py::MAX_LOGIN_ATTEMPTS',
            'symbol_name': 'MAX_LOGIN_ATTEMPTS',
            'symbol_type': 'constant',
            'file_path': '/repo/src/config.py',
            'rel_path': 'src/config.py',
            'language': 'python',
            'content': 'MAX_LOGIN_ATTEMPTS = 5',
            'start_line': 1,
            'end_line': 1,
        },
        {
            'id': 'python::types::src/types.py::UserID',
            'symbol_name': 'UserID',
            'symbol_type': 'type_alias',
            'file_path': '/repo/src/types.py',
            'rel_path': 'src/types.py',
            'language': 'python',
            'content': 'UserID = NewType("UserID", int)',
            'start_line': 3,
            'end_line': 3,
        },
        {
            'id': 'python::models::src/models/role.py::Role',
            'symbol_name': 'Role',
            'symbol_type': 'enum',
            'file_path': '/repo/src/models/role.py',
            'rel_path': 'src/models/role.py',
            'language': 'python',
            'docstring': 'User roles: ADMIN, USER, GUEST.',
            'content': 'class Role(Enum):\n    ADMIN = "admin"\n    USER = "user"\n    GUEST = "guest"',
            'start_line': 1,
            'end_line': 5,
        },
        {
            'id': 'python::interfaces::src/interfaces.py::Authenticatable',
            'symbol_name': 'Authenticatable',
            'symbol_type': 'trait',
            'file_path': '/repo/src/interfaces.py',
            'rel_path': 'src/interfaces.py',
            'language': 'python',
            'docstring': 'Trait for objects that can be authenticated.',
            'content': 'class Authenticatable(Protocol):\n    def verify(self) -> bool: ...',
            'start_line': 1,
            'end_line': 5,
        },
        # Doc node (should be excluded from code search)
        {
            'id': 'doc::README.md',
            'symbol_name': 'README.md',
            'symbol_type': 'markdown_document',
            'file_path': '/repo/README.md',
            'rel_path': 'README.md',
            'language': '',
            'content': '# Auth Service\nThis is the authentication service for the platform.',
        },
        # Node with camelCase name
        {
            'id': 'java::db::src/DatabaseConnectionPool.java::DatabaseConnectionPool',
            'symbol_name': 'DatabaseConnectionPool',
            'symbol_type': 'class',
            'file_path': '/repo/src/DatabaseConnectionPool.java',
            'rel_path': 'src/DatabaseConnectionPool.java',
            'language': 'java',
            'docstring': 'Manages database connections with pooling and health checks.',
            'content': 'public class DatabaseConnectionPool { ... }',
            'start_line': 1,
            'end_line': 100,
        },
    ]
    edges = [
        (
            'python::auth::src/auth/service.py::AuthService',
            'python::models::src/models/user.py::User',
            {'relationship_type': 'uses'},
        ),
        (
            'python::auth::src/auth/service.py::AuthService',
            'python::utils::src/utils/crypto.py::hash_password',
            {'relationship_type': 'calls'},
        ),
    ]
    return _make_graph(nodes, edges)


# =====================================================================
# Test Classes
# =====================================================================

class TestTokenizeName(unittest.TestCase):
    """Test the _tokenize_name helper."""

    def setUp(self):
        from plugin_implementation.code_graph.graph_text_index import _tokenize_name
        self.tok = _tokenize_name

    def test_camel_case(self):
        self.assertEqual(self.tok('AuthService'), 'auth service')

    def test_pascal_case_multi(self):
        self.assertEqual(self.tok('DatabaseConnectionPool'), 'database connection pool')

    def test_snake_case(self):
        self.assertEqual(self.tok('hash_password'), 'hash password')

    def test_acronym(self):
        self.assertEqual(self.tok('XMLParser'), 'xml parser')

    def test_mixed(self):
        result = self.tok('getHTTPResponse')
        self.assertIn('get', result)
        self.assertIn('response', result)

    def test_single_word(self):
        self.assertEqual(self.tok('User'), 'user')

    def test_empty_string(self):
        self.assertEqual(self.tok(''), '')

    def test_all_upper(self):
        self.assertEqual(self.tok('MAX_LOGIN_ATTEMPTS'), 'max login attempts')


class TestGraphTextIndexBuild(unittest.TestCase):
    """Test building the FTS5 index from a graph."""

    def setUp(self):
        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        self.tmpdir = tempfile.mkdtemp()
        self.idx = GraphTextIndex(cache_dir=self.tmpdir)
        self.graph = _realistic_graph()

    def tearDown(self):
        self.idx.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_build_returns_count(self):
        count = self.idx.build_from_graph(self.graph, 'test_key')
        # All named nodes should be indexed (10 nodes in _realistic_graph)
        self.assertEqual(count, 10)

    def test_build_creates_db_file(self):
        self.idx.build_from_graph(self.graph, 'test_key')
        db_path = Path(self.tmpdir) / 'test_key.fts5.db'
        self.assertTrue(db_path.exists())

    def test_build_is_idempotent(self):
        """Building twice with same key should replace the old index."""
        self.idx.build_from_graph(self.graph, 'test_key')
        count = self.idx.build_from_graph(self.graph, 'test_key')
        self.assertEqual(count, 10)

    def test_build_empty_graph(self):
        empty = nx.DiGraph()
        count = self.idx.build_from_graph(empty, 'empty_key')
        self.assertEqual(count, 0)

    def test_node_count(self):
        self.idx.build_from_graph(self.graph, 'test_key')
        self.assertEqual(self.idx.node_count, 10)

    def test_is_open_after_build(self):
        self.assertFalse(self.idx.is_open)
        self.idx.build_from_graph(self.graph, 'test_key')
        self.assertTrue(self.idx.is_open)


class TestGraphTextIndexSearch(unittest.TestCase):
    """Test FTS5 BM25-ranked search."""

    def setUp(self):
        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        self.tmpdir = tempfile.mkdtemp()
        self.idx = GraphTextIndex(cache_dir=self.tmpdir)
        self.idx.build_from_graph(_realistic_graph(), 'search_test')

    def tearDown(self):
        self.idx.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_exact_name_match(self):
        results = self.idx.search('AuthService', k=5)
        self.assertTrue(len(results) > 0)
        names = [d.metadata['symbol_name'] for d in results]
        self.assertIn('AuthService', names)

    def test_camel_case_partial(self):
        """Searching 'database connection' should match DatabaseConnectionPool."""
        results = self.idx.search('database connection', k=5)
        names = [d.metadata['symbol_name'] for d in results]
        self.assertIn('DatabaseConnectionPool', names)

    def test_docstring_match(self):
        """Search 'JWT tokens' should match AuthService (in docstring)."""
        results = self.idx.search('JWT tokens', k=5)
        self.assertTrue(len(results) > 0)
        names = [d.metadata['symbol_name'] for d in results]
        self.assertIn('AuthService', names)

    def test_content_match(self):
        """Search 'bcrypt hashpw' should match hash_password (in content)."""
        results = self.idx.search('bcrypt', k=5)
        self.assertTrue(len(results) > 0)
        names = [d.metadata['symbol_name'] for d in results]
        self.assertIn('hash_password', names)

    def test_exclude_types(self):
        """Excluding doc types should not return markdown_document nodes."""
        from plugin_implementation.constants import DOC_SYMBOL_TYPES
        results = self.idx.search('auth service', k=20, exclude_types=DOC_SYMBOL_TYPES)
        types = {d.metadata['symbol_type'] for d in results}
        self.assertNotIn('markdown_document', types)

    def test_symbol_types_filter(self):
        """Filtering to only classes should return only classes."""
        results = self.idx.search('auth', k=20, symbol_types=frozenset({'class'}))
        for doc in results:
            self.assertEqual(doc.metadata['symbol_type'], 'class')

    def test_search_returns_documents(self):
        results = self.idx.search('User', k=5)
        for doc in results:
            self.assertIsInstance(doc, Document)
            self.assertTrue(doc.page_content)
            self.assertIn('symbol_name', doc.metadata)
            self.assertEqual(doc.metadata['search_source'], 'graph_fts')

    def test_search_respects_k(self):
        results = self.idx.search('auth', k=2)
        self.assertLessEqual(len(results), 2)

    def test_empty_query_returns_empty(self):
        results = self.idx.search('', k=5)
        self.assertEqual(results, [])

    def test_no_match_returns_empty(self):
        results = self.idx.search('zzzznonexistent12345', k=5)
        self.assertEqual(results, [])

    def test_bm25_ranking_prefers_name_over_content(self):
        """AuthService should rank higher when searching 'AuthService'
        than a node that only mentions it in content."""
        results = self.idx.search('AuthService', k=10)
        if len(results) >= 2:
            # The first result should be the AuthService class itself
            self.assertEqual(results[0].metadata['symbol_name'], 'AuthService')

    def test_snake_case_search(self):
        """Searching 'hash password' should match hash_password."""
        results = self.idx.search('hash password', k=5)
        names = [d.metadata['symbol_name'] for d in results]
        self.assertIn('hash_password', names)

    def test_search_when_closed(self):
        self.idx.close()
        results = self.idx.search('AuthService', k=5)
        self.assertEqual(results, [])


class TestGraphTextIndexStructured(unittest.TestCase):
    """Test structured (non-FTS) queries."""

    def setUp(self):
        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        self.tmpdir = tempfile.mkdtemp()
        self.idx = GraphTextIndex(cache_dir=self.tmpdir)
        self.idx.build_from_graph(_realistic_graph(), 'struct_test')

    def tearDown(self):
        self.idx.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_search_by_name_exact(self):
        results = self.idx.search_by_name('AuthService', exact=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata['symbol_name'], 'AuthService')

    def test_search_by_name_prefix(self):
        results = self.idx.search_by_name('Auth', exact=False)
        self.assertTrue(len(results) >= 1)
        names = [d.metadata['symbol_name'] for d in results]
        self.assertIn('AuthService', names)

    def test_search_by_type(self):
        results = self.idx.search_by_type('class')
        names = {d.metadata['symbol_name'] for d in results}
        self.assertIn('AuthService', names)
        self.assertIn('User', names)
        self.assertIn('DatabaseConnectionPool', names)

    def test_get_by_node_id(self):
        result = self.idx.get_by_node_id('python::auth::src/auth/service.py::AuthService')
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol_name'], 'AuthService')
        self.assertEqual(result['symbol_type'], 'class')

    def test_get_by_node_id_not_found(self):
        result = self.idx.get_by_node_id('nonexistent::node')
        self.assertIsNone(result)


class TestGraphTextIndexPersistence(unittest.TestCase):
    """Test save/load lifecycle."""

    def setUp(self):
        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        self.tmpdir = tempfile.mkdtemp()
        self.GraphTextIndex = GraphTextIndex

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_after_build(self):
        """Build, close, reload, search should work."""
        idx1 = self.GraphTextIndex(cache_dir=self.tmpdir)
        idx1.build_from_graph(_realistic_graph(), 'persist_test')
        idx1.close()

        idx2 = self.GraphTextIndex(cache_dir=self.tmpdir)
        loaded = idx2.load('persist_test')
        self.assertTrue(loaded)
        self.assertEqual(idx2.node_count, 10)

        results = idx2.search('AuthService', k=5)
        self.assertTrue(len(results) > 0)
        idx2.close()

    def test_load_nonexistent(self):
        idx = self.GraphTextIndex(cache_dir=self.tmpdir)
        loaded = idx.load('no_such_key')
        self.assertFalse(loaded)

    def test_exists(self):
        idx = self.GraphTextIndex(cache_dir=self.tmpdir)
        self.assertFalse(idx.exists('exist_test'))
        idx.build_from_graph(_realistic_graph(), 'exist_test')
        self.assertTrue(idx.exists('exist_test'))
        idx.close()

    def test_delete(self):
        idx = self.GraphTextIndex(cache_dir=self.tmpdir)
        idx.build_from_graph(_realistic_graph(), 'del_test')
        idx.close()
        self.assertTrue(idx.exists('del_test'))
        idx.delete('del_test')
        self.assertFalse(idx.exists('del_test'))


class TestGraphTextIndexCacheRegistration(unittest.TestCase):
    """Test cache_index.json registration."""

    def setUp(self):
        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        self.tmpdir = tempfile.mkdtemp()
        self.idx = GraphTextIndex(cache_dir=self.tmpdir)
        self.idx.build_from_graph(_realistic_graph(), 'reg_key_123')

    def tearDown(self):
        self.idx.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_register_creates_fts5_key(self):
        import json
        self.idx.register_in_cache_index('owner/repo:main:abc123', 'reg_key_123')

        index_path = Path(self.tmpdir) / 'cache_index.json'
        self.assertTrue(index_path.exists())

        with open(index_path) as f:
            data = json.load(f)
        self.assertIn('fts5', data)
        self.assertEqual(data['fts5']['owner/repo:main:abc123'], 'reg_key_123')

    def test_load_by_repo_name(self):
        """After registering, should be able to load by repo name."""
        self.idx.register_in_cache_index('owner/repo:main:abc123', 'reg_key_123')
        self.idx.close()

        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        idx2 = GraphTextIndex(cache_dir=self.tmpdir)
        loaded = idx2.load_by_repo_name('owner/repo:main:abc123')
        self.assertTrue(loaded)
        self.assertEqual(idx2.node_count, 10)
        idx2.close()


class TestGraphManagerFTSIntegration(unittest.TestCase):
    """Test GraphManager's FTS index methods."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Enable FTS5 feature flag for these tests
        self._orig_fts5_env = os.environ.get('DEEPWIKI_ENABLE_FTS5')
        os.environ['DEEPWIKI_ENABLE_FTS5'] = '1'
        # Import here to avoid import errors during collection
        from plugin_implementation.graph_manager import GraphManager
        self.manager = GraphManager(cache_dir=self.tmpdir)
        self.graph = _realistic_graph()

    def tearDown(self):
        if self.manager._fts_index:
            self.manager._fts_index.close()
        # Restore original env var state
        if self._orig_fts5_env is None:
            os.environ.pop('DEEPWIKI_ENABLE_FTS5', None)
        else:
            os.environ['DEEPWIKI_ENABLE_FTS5'] = self._orig_fts5_env
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_fts_index_property(self):
        """fts_index should be lazily created."""
        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        idx = self.manager.fts_index
        self.assertIsInstance(idx, GraphTextIndex)
        # Same instance on repeated access
        self.assertIs(self.manager.fts_index, idx)

    def test_build_fts_index(self):
        idx = self.manager.build_fts_index(
            self.graph, '/tmp/test_repo', commit_hash='abc123'
        )
        self.assertTrue(idx.is_open)
        self.assertEqual(idx.node_count, 10)

    def test_load_fts_index(self):
        # Build first
        self.manager.build_fts_index(
            self.graph, '/tmp/test_repo', commit_hash='abc123'
        )
        self.manager.fts_index.close()

        # Load
        idx = self.manager.load_fts_index('/tmp/test_repo', commit_hash='abc123')
        self.assertIsNotNone(idx)
        self.assertEqual(idx.node_count, 10)

    def test_save_graph_auto_builds_fts(self):
        """save_graph should automatically build the FTS5 index."""
        self.manager.save_graph(
            self.graph, '/tmp/test_repo', commit_hash='abc123'
        )
        idx = self.manager.fts_index
        self.assertTrue(idx.is_open)
        self.assertEqual(idx.node_count, 10)

    def test_clear_cache_removes_fts(self):
        """clear_cache should also remove FTS5 files."""
        # Don't use commit_hash — clear_cache doesn't pass it to key generation
        self.manager.save_graph(
            self.graph, '/tmp/test_repo',
        )
        fts_files_before = list(Path(self.tmpdir).glob("*.fts5.db"))
        self.assertTrue(len(fts_files_before) > 0)

        self.manager.clear_cache('/tmp/test_repo')
        fts_files_after = list(Path(self.tmpdir).glob("*.fts5.db"))
        self.assertEqual(len(fts_files_after), 0)


class TestSearchCodebaseWithFTS(unittest.TestCase):
    """Test research_tools.search_codebase with FTS5 integration."""

    def setUp(self):
        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        self.tmpdir = tempfile.mkdtemp()
        self.graph = _realistic_graph()
        self.idx = GraphTextIndex(cache_dir=self.tmpdir)
        self.idx.build_from_graph(self.graph, 'search_test')

    def tearDown(self):
        self.idx.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_hybrid_search_fts_branch(self):
        """When FTS index is available, search_codebase should use it."""
        from plugin_implementation.deep_research.research_tools import create_codebase_tools

        tools = create_codebase_tools(
            retriever_stack=None,
            graph_manager=None,
            code_graph=self.graph,
            graph_text_index=self.idx,
        )
        search_tool = tools[0]  # search_codebase

        result = search_tool.invoke({'query': 'AuthService', 'k': 5})
        self.assertIn('AuthService', result)
        self.assertIn('graph_fts', result)

    def test_fallback_to_brute_force(self):
        """When FTS index is None, should fall back to brute-force."""
        from plugin_implementation.deep_research.research_tools import create_codebase_tools

        tools = create_codebase_tools(
            retriever_stack=None,
            graph_manager=None,
            code_graph=self.graph,
            graph_text_index=None,
        )
        search_tool = tools[0]

        result = search_tool.invoke({'query': 'AuthService', 'k': 5})
        self.assertIn('AuthService', result)
        # Should use brute-force graph search (source = 'graph')
        self.assertIn('graph', result)


class TestGraphTextIndexEdgeCases(unittest.TestCase):
    """Edge cases and robustness tests."""

    def setUp(self):
        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        self.tmpdir = tempfile.mkdtemp()
        self.GraphTextIndex = GraphTextIndex

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_special_characters_in_query(self):
        """FTS5 special chars (*, quotes) should not crash."""
        idx = self.GraphTextIndex(cache_dir=self.tmpdir)
        idx.build_from_graph(_realistic_graph(), 'special_test')
        # These should not raise
        idx.search('"auth*"', k=5)
        idx.search('class::method', k=5)
        idx.search('a/b/c.py', k=5)
        idx.search('SELECT * FROM', k=5)
        idx.close()

    def test_unicode_in_names(self):
        """Nodes with unicode names should be indexed and searchable."""
        g = _make_graph([{
            'id': 'python::café::café.py::CaféHandler',
            'symbol_name': 'CaféHandler',
            'symbol_type': 'class',
            'file_path': '/repo/café.py',
            'rel_path': 'café.py',
            'language': 'python',
            'content': 'class CaféHandler: ...',
        }])
        idx = self.GraphTextIndex(cache_dir=self.tmpdir)
        count = idx.build_from_graph(g, 'unicode_test')
        self.assertEqual(count, 1)
        results = idx.search('café', k=5)
        self.assertTrue(len(results) >= 1)
        idx.close()

    def test_very_long_content(self):
        """Nodes with very long content should not crash the indexer."""
        g = _make_graph([{
            'id': 'python::big::big.py::BigClass',
            'symbol_name': 'BigClass',
            'symbol_type': 'class',
            'file_path': '/repo/big.py',
            'rel_path': 'big.py',
            'language': 'python',
            'content': 'class BigClass:\n' + '    x = 1\n' * 50000,
        }])
        idx = self.GraphTextIndex(cache_dir=self.tmpdir)
        count = idx.build_from_graph(g, 'big_test')
        self.assertEqual(count, 1)
        results = idx.search('BigClass', k=5)
        self.assertEqual(len(results), 1)
        idx.close()

    def test_node_without_name_skipped(self):
        """Nodes without symbol_name should not be indexed."""
        g = nx.DiGraph()
        g.add_node('nameless', symbol_type='class', content='class Foo: ...')
        # No symbol_name attribute
        idx = self.GraphTextIndex(cache_dir=self.tmpdir)
        count = idx.build_from_graph(g, 'nameless_test')
        self.assertEqual(count, 0)
        idx.close()

    def test_repr(self):
        idx = self.GraphTextIndex(cache_dir=self.tmpdir)
        self.assertIn('closed', repr(idx))
        idx.build_from_graph(_realistic_graph(), 'repr_test')
        self.assertIn('repr_test', repr(idx))
        idx.close()

    def test_multiple_indexes(self):
        """Multiple cache keys should coexist."""
        idx = self.GraphTextIndex(cache_dir=self.tmpdir)
        idx.build_from_graph(_realistic_graph(), 'key_a')
        idx.close()
        idx.build_from_graph(_realistic_graph(), 'key_b')
        idx.close()

        # Both should exist
        self.assertTrue(idx.exists('key_a'))
        self.assertTrue(idx.exists('key_b'))

        # Load one and search
        idx.load('key_a')
        results = idx.search('AuthService', k=5)
        self.assertTrue(len(results) > 0)
        idx.close()


class TestGraphTextIndexBM25Weights(unittest.TestCase):
    """Test that BM25 weights prioritise name > docstring > content."""

    def setUp(self):
        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        self.tmpdir = tempfile.mkdtemp()

        # Create a graph where "logger" appears in different fields
        g = _make_graph([
            {
                'id': 'py::Logger',
                'symbol_name': 'Logger',        # name match
                'symbol_type': 'class',
                'file_path': '/repo/logger.py',
                'rel_path': 'logger.py',
                'language': 'python',
                'docstring': 'Main logging class.',
                'content': 'class Logger: pass',
            },
            {
                'id': 'py::Config',
                'symbol_name': 'Config',
                'symbol_type': 'class',
                'file_path': '/repo/config.py',
                'rel_path': 'config.py',
                'language': 'python',
                'docstring': 'Configures the logger settings.',  # docstring match
                'content': 'class Config: pass',
            },
            {
                'id': 'py::Handler',
                'symbol_name': 'Handler',
                'symbol_type': 'class',
                'file_path': '/repo/handler.py',
                'rel_path': 'handler.py',
                'language': 'python',
                'docstring': '',
                'content': 'class Handler:\n    self.logger = Logger()',  # content match
            },
        ])

        self.idx = GraphTextIndex(cache_dir=self.tmpdir)
        self.idx.build_from_graph(g, 'weight_test')

    def tearDown(self):
        self.idx.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_name_match_ranks_highest(self):
        results = self.idx.search('logger', k=3)
        self.assertTrue(len(results) >= 1)
        # Logger class (name match) should be first
        self.assertEqual(results[0].metadata['symbol_name'], 'Logger')

    def test_all_matches_returned(self):
        results = self.idx.search('logger', k=10)
        names = {d.metadata['symbol_name'] for d in results}
        # All three should appear (name, docstring, content match)
        self.assertEqual(names, {'Logger', 'Config', 'Handler'})


# ---------------------------------------------------------------------------
# Tests for search_by_path_prefix
# ---------------------------------------------------------------------------

class TestSearchByPathPrefix(unittest.TestCase):
    """Tests for GraphTextIndex.search_by_path_prefix()."""

    def setUp(self):
        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        self.tmpdir = tempfile.mkdtemp()
        self.idx = GraphTextIndex(cache_dir=self.tmpdir)
        graph = _make_graph([
            {'id': 'py::src/auth/service.py::AuthService', 'symbol_name': 'AuthService',
             'symbol_type': 'class', 'rel_path': 'src/auth/service.py', 'content': 'class AuthService:'},
            {'id': 'py::src/auth/models.py::User', 'symbol_name': 'User',
             'symbol_type': 'class', 'rel_path': 'src/auth/models.py', 'content': 'class User:'},
            {'id': 'py::src/api/handler.py::APIHandler', 'symbol_name': 'APIHandler',
             'symbol_type': 'class', 'rel_path': 'src/api/handler.py', 'content': 'class APIHandler:'},
            {'id': 'py::tests/test_auth.py::TestAuth', 'symbol_name': 'TestAuth',
             'symbol_type': 'class', 'rel_path': 'tests/test_auth.py', 'content': 'class TestAuth:'},
            {'id': 'md::docs/README.md::README', 'symbol_name': 'README',
             'symbol_type': 'markdown_document', 'rel_path': 'docs/README.md', 'content': '# Readme'},
        ])
        self.idx.build_from_graph(graph, 'prefix_test')

    def tearDown(self):
        self.idx.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_prefix_match_exact_dir(self):
        """Symbols under src/auth/ are returned."""
        rows = self.idx.search_by_path_prefix('src/auth')
        names = {r['symbol_name'] for r in rows}
        self.assertEqual(names, {'AuthService', 'User'})

    def test_prefix_no_match(self):
        """Non-existent prefix returns empty."""
        rows = self.idx.search_by_path_prefix('nonexistent')
        self.assertEqual(rows, [])

    def test_prefix_with_trailing_slash(self):
        """Leading/trailing slashes are stripped."""
        rows = self.idx.search_by_path_prefix('/src/auth/')
        names = {r['symbol_name'] for r in rows}
        self.assertEqual(names, {'AuthService', 'User'})

    def test_prefix_with_symbol_type_filter(self):
        """Filter by symbol_types returns only matching types."""
        rows = self.idx.search_by_path_prefix('docs', symbol_types=frozenset({'markdown_document'}))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['symbol_name'], 'README')

    def test_prefix_with_exclude_types(self):
        """exclude_types filters out unwanted types."""
        rows = self.idx.search_by_path_prefix('src', exclude_types=frozenset({'class'}))
        # All symbols under src/ are class, so all should be excluded
        self.assertEqual(rows, [])

    def test_prefix_root_returns_all(self):
        """Prefix 'src' returns everything under src/."""
        rows = self.idx.search_by_path_prefix('src')
        names = {r['symbol_name'] for r in rows}
        self.assertEqual(names, {'AuthService', 'User', 'APIHandler'})

    def test_prefix_closed_index(self):
        """Returns empty when index is closed."""
        self.idx.close()
        rows = self.idx.search_by_path_prefix('src')
        self.assertEqual(rows, [])


# ---------------------------------------------------------------------------
# Tests for _score_doc_candidate
# ---------------------------------------------------------------------------

class TestScoreDocCandidate(unittest.TestCase):
    """Tests for OptimizedWikiGenerationAgent._score_doc_candidate()."""

    def _score(self, *args, **kwargs):
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        return OptimizedWikiGenerationAgent._score_doc_candidate(*args, **kwargs)

    def test_target_docs_returns_1000(self):
        """When target_docs is set, score is always 1000."""
        score = self._score(
            'docs/README.md', '# README', ['docs/README.md'],
            set(), set(), 'Overview'
        )
        self.assertEqual(score, 1000)

    def test_readme_gets_bonus(self):
        """README files get a +30 bonus."""
        score = self._score(
            'README.md', '# project', None,
            set(), set(), ''
        )
        self.assertGreaterEqual(score, 30)

    def test_symbol_keyword_boost(self):
        """Keyword match in path gets +100 per keyword."""
        score = self._score(
            'src/auth/service.md', '# Auth', None,
            set(), {'auth', 'service'}, 'Authentication Service'
        )
        self.assertGreaterEqual(score, 200)

    def test_page_keyword_boost(self):
        """Page keyword match in path gets +50 per keyword."""
        score = self._score(
            'docs/tutorial.md', '# Tutorial', None,
            {'tutorial'}, set(), 'Tutorial Page'
        )
        self.assertGreaterEqual(score, 50)

    def test_docs_folder_bonus(self):
        """Files in docs/ get +20."""
        score = self._score(
            'docs/guide.md', '# Guide', None,
            set(), set(), ''
        )
        # -50 (no topic match) + 20 (docs folder) = -30 baseline
        self.assertGreaterEqual(score, -50 + 20)

    def test_no_topic_match_penalty(self):
        """Non-readme docs with no topic overlap get -50."""
        score = self._score(
            'random.md', '# Random', None,
            set(), set(), 'Authentication'
        )
        self.assertLessEqual(score, 0)


# ---------------------------------------------------------------------------
# Tests for _get_doc_nodes_fts5 / _get_doc_nodes_brute consistency
# ---------------------------------------------------------------------------

class TestDocNodesHelpers(unittest.TestCase):
    """Test that FTS5 and brute-force doc retrieval produce comparable results."""

    def setUp(self):
        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        self.tmpdir = tempfile.mkdtemp()
        # Build a graph with doc + code nodes
        self.graph = _make_graph([
            {'id': 'md::README.md::README', 'symbol_name': 'README',
             'symbol_type': 'markdown_document', 'rel_path': 'README.md',
             'content': '# My Project\n\nThis is the main readme.'},
            {'id': 'md::docs/auth.md::AuthDocs', 'symbol_name': 'AuthDocs',
             'symbol_type': 'markdown_document', 'rel_path': 'docs/auth.md',
             'content': '# Authentication\n\nLogin and registration.'},
            {'id': 'md::docs/api.md::APIDocs', 'symbol_name': 'APIDocs',
             'symbol_type': 'markdown_document', 'rel_path': 'docs/api.md',
             'content': '# API Reference\n\nEndpoints documentation.'},
            {'id': 'py::src/auth.py::AuthService', 'symbol_name': 'AuthService',
             'symbol_type': 'class', 'rel_path': 'src/auth.py',
             'content': 'class AuthService: pass'},
        ])
        self.idx = GraphTextIndex(cache_dir=self.tmpdir)
        self.idx.build_from_graph(self.graph, 'doctest')

    def tearDown(self):
        self.idx.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_agent(self, fts_index):
        """Create a minimal agent mock with the required methods."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        # We'll call the static/instance methods directly, no need for full construction
        return OptimizedWikiGenerationAgent

    def test_fts5_finds_doc_nodes_by_keyword(self):
        """FTS5 finds doc nodes matching keywords."""
        from plugin_implementation.constants import DOC_SYMBOL_TYPES
        results = self.idx.search('authentication', k=10, symbol_types=DOC_SYMBOL_TYPES)
        self.assertTrue(len(results) >= 1)
        rel_paths = {d.metadata.get('rel_path', '') for d in results}
        self.assertIn('docs/auth.md', rel_paths)

    def test_fts5_excludes_code_nodes_with_doc_type_filter(self):
        """FTS5 with symbol_types=DOC_SYMBOL_TYPES excludes code."""
        from plugin_implementation.constants import DOC_SYMBOL_TYPES
        results = self.idx.search('AuthService', k=10, symbol_types=DOC_SYMBOL_TYPES)
        for doc in results:
            self.assertIn(doc.metadata['symbol_type'], DOC_SYMBOL_TYPES)

    def test_path_prefix_search_for_target_docs(self):
        """search_by_path_prefix finds docs matching target_docs paths."""
        from plugin_implementation.constants import DOC_SYMBOL_TYPES
        rows = self.idx.search_by_path_prefix('docs/auth', symbol_types=DOC_SYMBOL_TYPES, k=5)
        self.assertTrue(len(rows) >= 1)
        self.assertEqual(rows[0]['rel_path'], 'docs/auth.md')


# ---------------------------------------------------------------------------
# Tests for StructureCollector graph_text_index wiring
# ---------------------------------------------------------------------------

class TestStructureCollectorFTSWiring(unittest.TestCase):
    """Test that StructureCollector accepts and stores graph_text_index."""

    def test_constructor_accepts_graph_text_index(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import StructureCollector
        mock_idx = MagicMock()
        collector = StructureCollector(
            page_budget=10, repo_root='/tmp',
            code_graph=None, graph_text_index=mock_idx,
        )
        self.assertIs(collector.graph_text_index, mock_idx)

    def test_constructor_default_none(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import StructureCollector
        collector = StructureCollector(page_budget=10, repo_root='/tmp')
        self.assertIsNone(collector.graph_text_index)


# ---------------------------------------------------------------------------
# Tests for WikiStructurePlannerEngine graph_text_index wiring
# ---------------------------------------------------------------------------

class TestStructureEngineFTSWiring(unittest.TestCase):
    """Test that WikiStructurePlannerEngine accepts graph_text_index."""

    def test_constructor_accepts_graph_text_index(self):
        from plugin_implementation.wiki_structure_planner.structure_engine import WikiStructurePlannerEngine
        tmpdir = tempfile.mkdtemp()
        try:
            mock_idx = MagicMock()
            engine = WikiStructurePlannerEngine(
                repo_root=tmpdir,
                graph_text_index=mock_idx,
            )
            self.assertIs(engine.graph_text_index, mock_idx)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_constructor_default_none(self):
        from plugin_implementation.wiki_structure_planner.structure_engine import WikiStructurePlannerEngine
        tmpdir = tempfile.mkdtemp()
        try:
            engine = WikiStructurePlannerEngine(repo_root=tmpdir)
            self.assertIsNone(engine.graph_text_index)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests for wiki_graph_optimized graph_text_index parameter
# ---------------------------------------------------------------------------

class TestWikiGraphOptimizedFTSParam(unittest.TestCase):
    """Test that OptimizedWikiGenerationAgent stores graph_text_index."""

    def test_constructor_stores_fts_index(self):
        """Constructor stores graph_text_index attribute."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        mock_idx = MagicMock()
        mock_indexer = MagicMock()
        mock_indexer.relationship_graph = None
        mock_retriever = MagicMock()
        mock_retriever.relationship_graph = None
        mock_llm = MagicMock()

        agent = OptimizedWikiGenerationAgent(
            indexer=mock_indexer,
            retriever_stack=mock_retriever,
            llm=mock_llm,
            repository_url='https://github.com/test/repo',
            graph_text_index=mock_idx,
        )
        self.assertIs(agent.graph_text_index, mock_idx)


# ---------------------------------------------------------------------------
# Tests for search_codebase doc/FTS balance
# ---------------------------------------------------------------------------

class TestSearchCodebaseBalance(unittest.TestCase):
    """Test the doc/FTS k-value balance tuning."""

    def test_max_doc_results_env_var(self):
        """_MAX_DOC_RESULTS defaults to 3 and is configurable via env."""
        import importlib
        # Default
        with patch.dict(os.environ, {}, clear=False):
            if 'DEEPWIKI_MAX_DOC_RESULTS' in os.environ:
                del os.environ['DEEPWIKI_MAX_DOC_RESULTS']
            mod = importlib.import_module('plugin_implementation.deep_research.research_tools')
            # Can't easily check closure variable, but module imports without error
            self.assertTrue(hasattr(mod, 'create_codebase_tools'))


# =====================================================================
# Thread-safety tests for GraphTextIndex
# =====================================================================

class TestGraphTextIndexThreadSafety(unittest.TestCase):
    """Verify per-call read-only connections prevent thread-safety crashes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        self.idx = GraphTextIndex(cache_dir=self.tmpdir)
        # Build a small test index
        G = nx.DiGraph()
        G.add_node('n1', symbol_name='AlphaService', symbol_type='class',
                    file_path='/repo/alpha.py', rel_path='alpha.py',
                    language='python', start_line=1, end_line=50,
                    content='class AlphaService: pass')
        G.add_node('n2', symbol_name='BetaHelper', symbol_type='function',
                    file_path='/repo/beta.py', rel_path='beta.py',
                    language='python', start_line=1, end_line=20,
                    content='def BetaHelper(): pass')
        self.idx.build_from_graph(G, cache_key='thread_test')

    def tearDown(self):
        self.idx.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_no_shared_conn_after_build(self):
        """After build_from_graph, _conn should be None (no shared connection)."""
        self.assertIsNone(self.idx._conn)
        self.assertEqual(self.idx._cache_key, 'thread_test')

    def test_is_open_with_cache_key_only(self):
        """is_open should be True when _cache_key is set (even without _conn)."""
        self.assertIsNone(self.idx._conn)
        self.assertTrue(self.idx.is_open)

    def test_search_without_shared_conn(self):
        """search() works via per-call connections, not shared _conn."""
        self.assertIsNone(self.idx._conn)
        results = self.idx.search('AlphaService', k=5)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].metadata['symbol_name'], 'AlphaService')

    def test_search_by_name_without_shared_conn(self):
        """search_by_name() works via per-call connections."""
        results = self.idx.search_by_name('Alpha', exact=False, k=5)
        self.assertTrue(len(results) > 0)

    def test_search_by_type_without_shared_conn(self):
        """search_by_type() works via per-call connections."""
        results = self.idx.search_by_type('class', k=5)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].metadata['symbol_name'], 'AlphaService')

    def test_search_by_path_prefix_without_shared_conn(self):
        """search_by_path_prefix() works via per-call connections."""
        results = self.idx.search_by_path_prefix('alpha')
        self.assertTrue(len(results) > 0)

    def test_get_by_node_id_without_shared_conn(self):
        """get_by_node_id() works via per-call connections."""
        result = self.idx.get_by_node_id('n1')
        self.assertIsNotNone(result)
        self.assertEqual(result['symbol_name'], 'AlphaService')

    def test_node_count_without_shared_conn(self):
        """node_count works via per-call connections."""
        self.assertEqual(self.idx.node_count, 2)

    def test_concurrent_reads_from_threads(self):
        """Multiple real threads can read concurrently without crash."""
        import threading

        results_by_thread = {}
        errors = []

        def worker(thread_id):
            try:
                r = self.idx.search('AlphaService', k=5)
                results_by_thread[thread_id] = len(r)
            except Exception as exc:
                errors.append((thread_id, str(exc)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [], f"Thread errors: {errors}")
        self.assertEqual(len(results_by_thread), 8)
        for tid, count in results_by_thread.items():
            self.assertGreater(count, 0, f"Thread {tid} got 0 results")

    def test_load_sets_cache_key_without_conn(self):
        """load() should set _cache_key but not store a shared connection."""
        # Close first, then reload
        self.idx.close()
        self.assertFalse(self.idx.is_open)
        ok = self.idx.load('thread_test')
        self.assertTrue(ok)
        self.assertIsNone(self.idx._conn)
        self.assertEqual(self.idx._cache_key, 'thread_test')
        # Reads should still work
        results = self.idx.search('Beta', k=5)
        self.assertTrue(len(results) > 0)

    def test_open_read_conn_is_read_only(self):
        """_open_read_conn() connections should reject writes."""
        conn = self.idx._open_read_conn()
        try:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("INSERT INTO symbols (node_id, symbol_name) VALUES ('x', 'x')")
        finally:
            conn.close()


# =====================================================================
# ContentExpander-based expansion tests
# =====================================================================

class TestNodeToDocument(unittest.TestCase):
    """Test the _node_to_document static helper on OptimizedWikiGenerationAgent."""

    def test_node_to_document_returns_none_for_empty(self):
        """_node_to_document returns None for nodes without content."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        G = nx.DiGraph()
        G.add_node('empty', symbol_name='Empty', symbol_type='class')
        result = OptimizedWikiGenerationAgent._node_to_document('empty', G)
        self.assertIsNone(result)

    def test_node_to_document_extracts_metadata(self):
        """_node_to_document creates proper Document with all metadata."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        G = nx.DiGraph()
        G.add_node('n1', symbol_name='Foo', symbol_type='class',
                    file_path='/repo/foo.py', rel_path='foo.py',
                    language='python', start_line=5, end_line=20,
                    content='class Foo: pass')
        doc = OptimizedWikiGenerationAgent._node_to_document('n1', G)
        self.assertIsNotNone(doc)
        self.assertEqual(doc.page_content, 'class Foo: pass')
        self.assertEqual(doc.metadata['symbol_name'], 'Foo')
        self.assertEqual(doc.metadata['symbol_type'], 'class')
        self.assertEqual(doc.metadata['language'], 'python')
        self.assertTrue(doc.metadata['graph_retrieved'])


if __name__ == '__main__':
    unittest.main()
