"""
Tests for SPEC-2: JQL Structured Queries.

Covers:
  - jql_parser: tokenization, clause parsing, query splitting
  - GraphQueryService.query(): index routing, post-filters, related traversal
"""

import sys
import os
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

import networkx as nx

# ---------------------------------------------------------------------------
# Path setup — allow running standalone
# ---------------------------------------------------------------------------
_PLUGIN_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'))
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from plugin_implementation.code_graph.jql_parser import (
    ClauseOp, JQLClause, JQLQuery,
    parse_jql, is_jql_expression,
)
from plugin_implementation.code_graph.graph_query_service import (
    GraphQueryService, SymbolResult,
)


# ---------------------------------------------------------------------------
# Helper: build a small mock graph
# ---------------------------------------------------------------------------

def _build_test_graph():
    """Create a minimal DiGraph with in-memory indexes for testing."""
    G = nx.DiGraph()

    # Nodes
    nodes = [
        ('auth::AuthService', {
            'symbol_name': 'AuthService',
            'symbol_type': 'class',
            'rel_path': 'src/auth/service.py',
            'file_path': '/repo/src/auth/service.py',
            'language': 'python',
            'layer': 'core_type',
            'docstring': 'Handles authentication and authorization',
            'content': 'class AuthService:\n    ...',
        }),
        ('auth::login', {
            'symbol_name': 'login',
            'symbol_type': 'function',
            'rel_path': 'src/auth/handlers.py',
            'file_path': '/repo/src/auth/handlers.py',
            'language': 'python',
            'layer': 'public_api',
            'docstring': 'Login endpoint handler',
            'content': 'def login(request):\n    ...',
        }),
        ('auth::BaseHandler', {
            'symbol_name': 'BaseHandler',
            'symbol_type': 'class',
            'rel_path': 'src/auth/base.py',
            'file_path': '/repo/src/auth/base.py',
            'language': 'python',
            'layer': 'internal',
            'docstring': 'Base request handler',
            'content': 'class BaseHandler:\n    ...',
        }),
        ('db::DatabasePool', {
            'symbol_name': 'DatabasePool',
            'symbol_type': 'class',
            'rel_path': 'src/db/pool.py',
            'file_path': '/repo/src/db/pool.py',
            'language': 'python',
            'layer': 'infrastructure',
            'docstring': 'Connection pool for database',
            'content': 'class DatabasePool:\n    ...',
        }),
        ('db::MAX_CONNECTIONS', {
            'symbol_name': 'MAX_CONNECTIONS',
            'symbol_type': 'constant',
            'rel_path': 'src/db/config.py',
            'file_path': '/repo/src/db/config.py',
            'language': 'python',
            'layer': 'constant',
            'docstring': '',
            'content': 'MAX_CONNECTIONS = 100',
        }),
        ('api::UserController', {
            'symbol_name': 'UserController',
            'symbol_type': 'class',
            'rel_path': 'src/api/controllers.py',
            'file_path': '/repo/src/api/controllers.py',
            'language': 'python',
            'layer': 'public_api',
            'docstring': 'REST controller for user operations',
            'content': 'class UserController:\n    ...',
        }),
    ]
    for nid, data in nodes:
        G.add_node(nid, **data)

    # Edges
    edges = [
        ('auth::AuthService', 'auth::BaseHandler', {'relationship_type': 'inherits'}),
        ('auth::login', 'auth::AuthService', {'relationship_type': 'calls'}),
        ('auth::AuthService', 'db::DatabasePool', {'relationship_type': 'uses'}),
        ('api::UserController', 'auth::AuthService', {'relationship_type': 'calls'}),
        ('api::UserController', 'auth::login', {'relationship_type': 'calls'}),
        ('db::DatabasePool', 'db::MAX_CONNECTIONS', {'relationship_type': 'uses'}),
    ]
    for src, tgt, data in edges:
        G.add_edge(src, tgt, **data)

    # In-memory indexes (mimic graph_builder output)
    G._node_index = {}
    G._simple_name_index = {}
    G._full_name_index = {}
    G._name_index = defaultdict(list)
    G._suffix_index = defaultdict(list)

    for nid, data in G.nodes(data=True):
        name = data.get('symbol_name', '')
        lang = data.get('language', '')
        fpath = data.get('rel_path', '')

        G._node_index[(name, fpath, lang)] = nid
        G._full_name_index[nid] = nid
        G._name_index[name].append(nid)

        simple = name.rsplit('.', 1)[-1].rsplit('::', 1)[-1]
        G._simple_name_index[(simple, fpath, lang)] = nid

    return G


# ===================================================================
# Test: JQL Parser
# ===================================================================

class TestJQLParser(unittest.TestCase):
    """Tests for jql_parser.parse_jql() and helpers."""

    def test_empty_query(self):
        q = parse_jql('')
        self.assertTrue(q.is_empty)
        q2 = parse_jql('   ')
        self.assertTrue(q2.is_empty)

    def test_single_type_clause(self):
        q = parse_jql('type:class')
        self.assertEqual(q.type_values, ['class'])
        self.assertFalse(q.is_or)

    def test_multiple_type_clauses_or(self):
        q = parse_jql('type:class OR type:interface')
        self.assertEqual(sorted(q.type_values), ['class', 'interface'])
        self.assertTrue(q.is_or)

    def test_file_glob(self):
        q = parse_jql('file:src/auth/*.py')
        self.assertEqual(q.file_value, 'src/auth/*.py')

    def test_quoted_value(self):
        q = parse_jql('related:"Base Handler"')
        self.assertEqual(q.related_value, 'Base Handler')

    def test_text_clause(self):
        q = parse_jql('text:authentication')
        self.assertTrue(q.has_text_clause)
        self.assertEqual(q.text_value, 'authentication')

    def test_layer_clause(self):
        q = parse_jql('layer:core_type')
        self.assertEqual(q.layer_value, 'core_type')

    def test_name_clause(self):
        q = parse_jql('name:Auth*')
        self.assertEqual(q.name_value, 'Auth*')

    def test_connections_gt(self):
        q = parse_jql('connections:>5')
        clause = q.connections_clause
        self.assertIsNotNone(clause)
        self.assertEqual(clause.op, ClauseOp.GT)
        self.assertTrue(clause.matches_numeric(6))
        self.assertFalse(clause.matches_numeric(5))
        self.assertFalse(clause.matches_numeric(3))

    def test_connections_gte(self):
        q = parse_jql('connections:>=5')
        clause = q.connections_clause
        self.assertEqual(clause.op, ClauseOp.GTE)
        self.assertTrue(clause.matches_numeric(5))
        self.assertFalse(clause.matches_numeric(4))

    def test_connections_lt(self):
        q = parse_jql('connections:<10')
        clause = q.connections_clause
        self.assertEqual(clause.op, ClauseOp.LT)
        self.assertTrue(clause.matches_numeric(9))
        self.assertFalse(clause.matches_numeric(10))

    def test_limit_clause(self):
        q = parse_jql('type:class limit:5')
        self.assertEqual(q.limit, 5)
        self.assertEqual(q.type_values, ['class'])

    def test_limit_clamped(self):
        q = parse_jql('limit:9999')
        self.assertEqual(q.limit, 500)

    def test_direction_clause(self):
        q = parse_jql('related:AuthService dir:outgoing')
        self.assertEqual(q.direction_value, 'outgoing')
        self.assertEqual(q.related_value, 'AuthService')

    def test_has_rel_clause(self):
        q = parse_jql('has_rel:inherits')
        self.assertEqual(q.has_rel_values, ['inherits'])

    def test_combined_query(self):
        q = parse_jql('type:class file:src/auth/* text:handler limit:10')
        self.assertEqual(q.type_values, ['class'])
        self.assertEqual(q.file_value, 'src/auth/*')
        self.assertEqual(q.text_value, 'handler')
        self.assertEqual(q.limit, 10)
        self.assertFalse(q.is_or)

    def test_index_vs_postfilter_split(self):
        q = parse_jql('type:class related:Base dir:incoming connections:>3')
        # Index clauses
        self.assertEqual(len(q.index_clauses), 1)  # type:class
        self.assertEqual(q.index_clauses[0].field, 'type')
        # Post-filters
        self.assertEqual(len(q.post_filters), 3)  # related, dir, connections
        fields = {c.field for c in q.post_filters}
        self.assertEqual(fields, {'related', 'dir', 'connections'})

    def test_unknown_field_ignored(self):
        q = parse_jql('type:class unknown:value')
        self.assertEqual(len(q.index_clauses), 1)

    def test_implicit_and(self):
        q = parse_jql('type:class file:src/*')
        self.assertFalse(q.is_or)
        self.assertEqual(len(q.index_clauses), 2)

    def test_explicit_and(self):
        q = parse_jql('type:class AND file:src/*')
        self.assertFalse(q.is_or)
        self.assertEqual(len(q.index_clauses), 2)


class TestIsJQLExpression(unittest.TestCase):
    """Tests for is_jql_expression() heuristic."""

    def test_jql_detected(self):
        self.assertTrue(is_jql_expression('type:class'))
        self.assertTrue(is_jql_expression('file:src/*.py'))
        self.assertTrue(is_jql_expression('related:AuthService connections:>5'))

    def test_plain_text_not_jql(self):
        self.assertFalse(is_jql_expression('authentication handlers'))
        self.assertFalse(is_jql_expression('how does login work'))
        self.assertFalse(is_jql_expression(''))

    def test_unknown_field_not_jql(self):
        self.assertFalse(is_jql_expression('foo:bar'))


class TestClauseMatching(unittest.TestCase):
    """Tests for JQLClause matching helpers."""

    def test_eq_exact_match(self):
        c = JQLClause(field='type', op=ClauseOp.EQ, value='class')
        self.assertTrue(c.matches_value('class'))
        self.assertTrue(c.matches_value('Class'))  # case-insensitive
        self.assertFalse(c.matches_value('function'))

    def test_eq_glob_match(self):
        c = JQLClause(field='file', op=ClauseOp.EQ, value='src/auth/*.py')
        self.assertTrue(c.matches_value('src/auth/service.py'))
        self.assertFalse(c.matches_value('src/db/pool.py'))

    def test_match_substring(self):
        c = JQLClause(field='text', op=ClauseOp.MATCH, value='auth')
        self.assertTrue(c.matches_value('authentication'))
        self.assertFalse(c.matches_value('database'))


# ===================================================================
# Test: GraphQueryService.query()
# ===================================================================

class TestGraphQueryServiceQuery(unittest.TestCase):
    """Tests for GraphQueryService.query() — SPEC-2 executor."""

    def setUp(self):
        self.graph = _build_test_graph()
        # No FTS5 — all queries will use full-scan fallback
        self.svc = GraphQueryService(self.graph, fts_index=None)

    def test_type_filter(self):
        results = self.svc.query('type:class')
        names = {r.symbol_name for r in results}
        self.assertIn('AuthService', names)
        self.assertIn('BaseHandler', names)
        self.assertIn('DatabasePool', names)
        self.assertIn('UserController', names)
        # Should NOT include functions or constants
        self.assertNotIn('login', names)
        self.assertNotIn('MAX_CONNECTIONS', names)

    def test_type_filter_function(self):
        results = self.svc.query('type:function')
        names = {r.symbol_name for r in results}
        self.assertEqual(names, {'login'})

    def test_file_glob_filter(self):
        results = self.svc.query('file:src/auth/*')
        names = {r.symbol_name for r in results}
        self.assertIn('AuthService', names)
        self.assertIn('login', names)
        self.assertIn('BaseHandler', names)
        self.assertNotIn('DatabasePool', names)

    def test_combined_type_and_file(self):
        results = self.svc.query('type:class file:src/auth/*')
        names = {r.symbol_name for r in results}
        self.assertIn('AuthService', names)
        self.assertIn('BaseHandler', names)
        self.assertNotIn('login', names)  # function, not class

    def test_name_filter_exact(self):
        results = self.svc.query('name:AuthService')
        names = {r.symbol_name for r in results}
        self.assertIn('AuthService', names)
        self.assertEqual(len(names), 1)

    def test_name_filter_glob(self):
        results = self.svc.query('name:*Controller')
        names = {r.symbol_name for r in results}
        self.assertIn('UserController', names)
        self.assertNotIn('AuthService', names)

    def test_connections_gt(self):
        # AuthService: in_degree=2 (login→, UserController→), out_degree=2 (→BaseHandler, →DatabasePool)
        # = 4 connections
        # UserController: in_degree=0, out_degree=2 = 2 connections
        results = self.svc.query('type:class connections:>3')
        names = {r.symbol_name for r in results}
        self.assertIn('AuthService', names)
        self.assertNotIn('UserController', names)

    def test_related_filter(self):
        # Symbols reachable from AuthService outgoing
        results = self.svc.query('related:AuthService dir:outgoing')
        names = {r.symbol_name for r in results}
        self.assertIn('BaseHandler', names)
        self.assertIn('DatabasePool', names)

    def test_related_incoming(self):
        # Symbols reaching AuthService
        results = self.svc.query('related:AuthService dir:incoming')
        names = {r.symbol_name for r in results}
        self.assertIn('login', names)
        self.assertIn('UserController', names)

    def test_has_rel_inherits(self):
        results = self.svc.query('type:class has_rel:inherits')
        names = {r.symbol_name for r in results}
        # AuthService inherits BaseHandler, BaseHandler is inherited
        self.assertTrue(
            'AuthService' in names or 'BaseHandler' in names,
            f"Expected ≥1 class with 'inherits' edge, got: {names}"
        )

    def test_limit(self):
        results = self.svc.query('type:class limit:2')
        self.assertLessEqual(len(results), 2)

    def test_empty_query(self):
        results = self.svc.query('')
        self.assertEqual(results, [])

    def test_no_matches(self):
        results = self.svc.query('type:interface')
        self.assertEqual(results, [])

    def test_related_nonexistent_symbol(self):
        results = self.svc.query('related:NonExistentClass')
        self.assertEqual(results, [])

    def test_related_with_has_rel(self):
        # AuthService → outgoing → only 'calls' edges from login/UserController
        results = self.svc.query('related:AuthService dir:outgoing has_rel:uses')
        names = {r.symbol_name for r in results}
        self.assertIn('DatabasePool', names)

    def test_constant_type(self):
        results = self.svc.query('type:constant')
        names = {r.symbol_name for r in results}
        self.assertEqual(names, {'MAX_CONNECTIONS'})


# ===================================================================
# Test: GraphQueryService._matches_path
# ===================================================================

class TestMatchesPath(unittest.TestCase):
    """Tests for the static _matches_path helper."""

    def test_prefix_match(self):
        self.assertTrue(GraphQueryService._matches_path('src/auth/service.py', 'src/auth'))
        self.assertTrue(GraphQueryService._matches_path('src/auth/service.py', 'src/auth/'))

    def test_glob_match(self):
        self.assertTrue(GraphQueryService._matches_path('src/auth/service.py', 'src/auth/*.py'))
        self.assertFalse(GraphQueryService._matches_path('src/db/pool.py', 'src/auth/*.py'))

    def test_exact_match(self):
        self.assertTrue(GraphQueryService._matches_path('src/auth', 'src/auth'))

    def test_empty_path(self):
        self.assertFalse(GraphQueryService._matches_path('', 'src/auth'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
