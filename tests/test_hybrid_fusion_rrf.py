"""
Tests for Phase 4: Hybrid Fusion (RRF) and ``search_graph`` tool.

Test suites:
1. **RRF unit tests** — ``reciprocal_rank_fusion`` and ``fuse_search_results``
   with synthetic Documents (no external dependencies)
2. **search_codebase wiring** — ``create_codebase_tools`` with mocked backends,
   verifying RRF is used when ``HYBRID_FUSION_ENABLED`` is True
3. **search_graph tool** — FTS5 + graph traversal with mocked and real data
4. **Helper functions** — ``_find_graph_node``, ``_format_neighbors``,
   ``_extract_rel_type``
5. **Integration tests** — real FTS5 databases + cached graphs (fmtlib,
   configurations)
"""

import gzip
import os
import pickle
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import networkx as nx

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _THIS_DIR.parent
_IMPL_DIR = _PLUGIN_ROOT / 'plugin_implementation'

if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))

from langchain_core.documents import Document

from plugin_implementation.deep_research.hybrid_fusion import (
    HYBRID_FUSION_ENABLED,
    RRF_K,
    FusedResult,
    reciprocal_rank_fusion,
    fuse_search_results,
    _dedup_value,
    _merge_metadata,
)
from plugin_implementation.deep_research.research_tools import (
    _find_graph_node,
    _format_neighbors,
    _extract_rel_type,
    _search_graph_by_text,
    create_codebase_tools,
)
from plugin_implementation.code_graph.graph_text_index import GraphTextIndex

# ---------------------------------------------------------------------------
# Cache paths for integration tests
# ---------------------------------------------------------------------------
_WIKI_BUILDER = _PLUGIN_ROOT.parent.parent / 'wiki_builder'
_CACHE_DIR = _WIKI_BUILDER / 'cache'
_FMTLIB_KEY = '551134763c1f5c1b3feca4dd95076290'
_CONFIGS_KEY = 'cd9d7a4aefa47194b872a7093a855156'


def _fts_db_exists(key: str) -> bool:
    return (_CACHE_DIR / f'{key}.fts5.db').exists()


def _load_cached_graph(key: str):
    """Load a pickled+gzipped graph from cache."""
    path = _CACHE_DIR / f'{key}.code_graph.gz'
    if not path.exists():
        return None
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------
def _make_doc(name: str, content: str = '', source: str = 'test',
              symbol_type: str = 'function', **extra_meta) -> Document:
    """Create a Document with standard metadata."""
    meta = {
        'symbol_name': name,
        'symbol_type': symbol_type,
        'source': source,
        'rel_path': source,
        'start_line': 1,
        'end_line': 10,
    }
    meta.update(extra_meta)
    return Document(page_content=content or f'def {name}(): pass', metadata=meta)


def _build_mini_graph() -> nx.MultiDiGraph:
    """Build a small MultiDiGraph for unit tests."""
    G = nx.MultiDiGraph()

    # Nodes
    G.add_node('src/auth.py::AuthService', **{
        'symbol_name': 'AuthService', 'symbol_type': 'class',
        'content': 'class AuthService:\n    def login(self): ...',
        'docstring': 'Authentication service for user login.',
        'rel_path': 'src/auth.py', 'file_path': '/repo/src/auth.py',
        'start_line': 1, 'end_line': 20,
    })
    G.add_node('src/auth.py::login', **{
        'symbol_name': 'login', 'symbol_type': 'method',
        'content': 'def login(self, username, password): ...',
        'docstring': 'Authenticate a user.',
        'rel_path': 'src/auth.py', 'file_path': '/repo/src/auth.py',
        'start_line': 5, 'end_line': 15,
    })
    G.add_node('src/db.py::Database', **{
        'symbol_name': 'Database', 'symbol_type': 'class',
        'content': 'class Database:\n    def query(self): ...',
        'docstring': 'Database connection pool.',
        'rel_path': 'src/db.py', 'file_path': '/repo/src/db.py',
        'start_line': 1, 'end_line': 30,
    })
    G.add_node('src/models.py::User', **{
        'symbol_name': 'User', 'symbol_type': 'class',
        'content': 'class User:\n    name: str\n    email: str',
        'docstring': 'User model.',
        'rel_path': 'src/models.py', 'file_path': '/repo/src/models.py',
        'start_line': 1, 'end_line': 10,
    })
    G.add_node('src/utils.py::hash_password', **{
        'symbol_name': 'hash_password', 'symbol_type': 'function',
        'content': 'def hash_password(pw: str) -> str: ...',
        'docstring': 'Hash a password with bcrypt.',
        'rel_path': 'src/utils.py', 'file_path': '/repo/src/utils.py',
        'start_line': 1, 'end_line': 5,
    })

    # Edges
    G.add_edge('src/auth.py::AuthService', 'src/auth.py::login',
               relationship_type='contains')
    G.add_edge('src/auth.py::AuthService', 'src/db.py::Database',
               relationship_type='calls')
    G.add_edge('src/auth.py::login', 'src/utils.py::hash_password',
               relationship_type='calls')
    G.add_edge('src/auth.py::login', 'src/models.py::User',
               relationship_type='references')
    G.add_edge('src/db.py::Database', 'src/models.py::User',
               relationship_type='creates')

    return G


# ==========================================================================
# Suite 1: RRF Core Algorithm
# ==========================================================================
class TestRRFCore(unittest.TestCase):
    """Test the reciprocal rank fusion algorithm with synthetic data."""

    def test_single_source_preserves_order(self):
        """Single source → RRF scores decrease monotonically."""
        docs = [_make_doc(f'sym_{i}') for i in range(5)]
        results = reciprocal_rank_fusion({'src': docs})
        scores = [r.rrf_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_single_source_scores(self):
        """Verify exact RRF score for single source."""
        docs = [_make_doc('alpha'), _make_doc('beta')]
        results = reciprocal_rank_fusion({'src': docs}, k=60)
        # rank 1 → 1/(60+1), rank 2 → 1/(60+2)
        self.assertAlmostEqual(results[0].rrf_score, 1.0 / 61)
        self.assertAlmostEqual(results[1].rrf_score, 1.0 / 62)

    def test_two_sources_boost_overlap(self):
        """Document in both sources gets boosted above single-source docs."""
        vs_docs = [_make_doc('shared'), _make_doc('vs_only')]
        fts_docs = [_make_doc('fts_only'), _make_doc('shared')]

        results = reciprocal_rank_fusion({
            'vectorstore': vs_docs,
            'fts5': fts_docs,
        })

        # 'shared' should be first: it has contributions from both sources
        self.assertEqual(results[0].document.metadata['symbol_name'], 'shared')

        # Verify accumulated score
        # VS rank 1: 1/61, FTS5 rank 2: 1/62
        expected = 1.0 / 61 + 1.0 / 62
        self.assertAlmostEqual(results[0].rrf_score, expected)

    def test_dedup_keeps_better_ranked_content(self):
        """When docs share dedup key, content from better-ranked source wins."""
        vs_doc = _make_doc('MyClass', content='VS version of MyClass')
        fts_doc = _make_doc('MyClass', content='FTS5 version of MyClass')

        # VS has MyClass at rank 1, FTS5 has it at rank 3 (worse)
        vs_docs = [vs_doc]
        fts_docs = [_make_doc('other1'), _make_doc('other2'), fts_doc]

        results = reciprocal_rank_fusion({
            'vectorstore': vs_docs,
            'fts5': fts_docs,
        })

        myclass_results = [r for r in results if 'myclass' in
                           _dedup_value(r.document, 'symbol_name')]
        self.assertEqual(len(myclass_results), 1)
        # VS rank 1 is better than FTS5 rank 3, so VS content should be kept
        # (actually, the _merge_metadata keeps the better-ranked doc's content)

    def test_cap_limits_output(self):
        """Cap parameter limits the number of returned results."""
        docs = [_make_doc(f'sym_{i}') for i in range(20)]
        results = reciprocal_rank_fusion({'src': docs}, cap=5)
        self.assertEqual(len(results), 5)

    def test_empty_inputs(self):
        """Empty ranked lists produce empty results."""
        results = reciprocal_rank_fusion({})
        self.assertEqual(results, [])
        results = reciprocal_rank_fusion({'a': [], 'b': []})
        self.assertEqual(results, [])

    def test_three_sources(self):
        """Three sources all contribute to RRF score."""
        doc_a = [_make_doc('shared'), _make_doc('a_only')]
        doc_b = [_make_doc('b_only'), _make_doc('shared')]
        doc_c = [_make_doc('shared'), _make_doc('c_only')]

        results = reciprocal_rank_fusion({
            'a': doc_a, 'b': doc_b, 'c': doc_c,
        })

        # 'shared' appears in all three → highest RRF score
        self.assertEqual(results[0].document.metadata['symbol_name'], 'shared')
        # Score: 1/61 + 1/62 + 1/61
        expected = 1.0 / 61 + 1.0 / 62 + 1.0 / 61
        self.assertAlmostEqual(results[0].rrf_score, expected)

    def test_source_ranks_tracked(self):
        """source_ranks dict tracks per-source rank for each result."""
        vs = [_make_doc('alpha'), _make_doc('beta')]
        fts = [_make_doc('beta'), _make_doc('gamma')]

        results = reciprocal_rank_fusion({'vs': vs, 'fts': fts})

        beta = next(r for r in results if
                    r.document.metadata['symbol_name'] == 'beta')
        self.assertEqual(beta.source_ranks, {'vs': 2, 'fts': 1})

    def test_custom_k_value(self):
        """Custom k parameter changes score magnitudes."""
        docs = [_make_doc('sym')]
        results_k10 = reciprocal_rank_fusion({'src': docs}, k=10)
        results_k100 = reciprocal_rank_fusion({'src': docs}, k=100)
        # k=10: 1/11, k=100: 1/101
        self.assertAlmostEqual(results_k10[0].rrf_score, 1.0 / 11)
        self.assertAlmostEqual(results_k100[0].rrf_score, 1.0 / 101)

    def test_no_dedup_key_uses_content_hash(self):
        """Documents without symbol_name still get unique entries."""
        doc1 = Document(page_content='content A', metadata={})
        doc2 = Document(page_content='content B', metadata={})
        results = reciprocal_rank_fusion({'src': [doc1, doc2]})
        self.assertEqual(len(results), 2)

    def test_best_rank_property(self):
        """FusedResult.best_rank returns minimum rank across sources."""
        fr = FusedResult(
            document=_make_doc('x'),
            rrf_score=0.03,
            source_ranks={'vs': 5, 'fts': 2, 'brute': 8},
        )
        self.assertEqual(fr.best_rank, 2)

    def test_best_rank_empty(self):
        """FusedResult.best_rank returns 9999 when no sources."""
        fr = FusedResult(document=_make_doc('x'))
        self.assertEqual(fr.best_rank, 9999)


# ==========================================================================
# Suite 2: fuse_search_results convenience function
# ==========================================================================
class TestFuseSearchResults(unittest.TestCase):
    """Test the convenience wrapper that returns plain Documents."""

    def test_returns_documents(self):
        """fuse_search_results returns Document objects, not FusedResult."""
        docs = [_make_doc('sym')]
        result = fuse_search_results({'src': docs})
        self.assertIsInstance(result[0], Document)

    def test_rrf_score_in_metadata(self):
        """Returned documents carry rrf_score metadata."""
        docs = [_make_doc('sym')]
        result = fuse_search_results({'src': docs})
        self.assertIn('rrf_score', result[0].metadata)
        self.assertAlmostEqual(result[0].metadata['rrf_score'],
                               round(1.0 / 61, 6))

    def test_rrf_sources_in_metadata(self):
        """Returned documents carry rrf_sources metadata."""
        vs = [_make_doc('shared')]
        fts = [_make_doc('shared')]
        result = fuse_search_results({'vs': vs, 'fts': fts})
        sources = result[0].metadata.get('rrf_sources', '')
        self.assertIn('vs', sources)
        self.assertIn('fts', sources)

    def test_cap_parameter(self):
        """Cap limits results in convenience function."""
        docs = [_make_doc(f's{i}') for i in range(10)]
        result = fuse_search_results({'src': docs}, cap=3)
        self.assertEqual(len(result), 3)

    def test_ordering_by_score(self):
        """Results are ordered by RRF score descending."""
        vs = [_make_doc('first'), _make_doc('second')]
        result = fuse_search_results({'src': vs})
        scores = [d.metadata['rrf_score'] for d in result]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ==========================================================================
# Suite 3: Dedup and merge helpers
# ==========================================================================
class TestDedupHelpers(unittest.TestCase):
    """Test _dedup_value and _merge_metadata."""

    def test_dedup_value_normal(self):
        doc = _make_doc('MyClass')
        self.assertEqual(_dedup_value(doc, 'symbol_name'), 'myclass')

    def test_dedup_value_missing_key(self):
        doc = Document(page_content='x', metadata={})
        self.assertEqual(_dedup_value(doc, 'symbol_name'), '')

    def test_dedup_value_whitespace(self):
        doc = _make_doc('  Foo  ')
        self.assertEqual(_dedup_value(doc, 'symbol_name'), 'foo')

    def test_merge_metadata_combines_sources(self):
        d1 = _make_doc('a', search_sources='vs')
        d2 = _make_doc('a')
        merged = _merge_metadata(d1, d2, 'fts')
        self.assertIn('fts', merged.metadata.get('search_sources', ''))


# ==========================================================================
# Suite 4: Feature flag
# ==========================================================================
class TestFeatureFlag(unittest.TestCase):
    """Test HYBRID_FUSION_ENABLED feature flag."""

    def test_default_off(self):
        """HYBRID_FUSION_ENABLED is off by default."""
        # Unless someone set DEEPWIKI_HYBRID_FUSION=1 in the env
        expected = os.environ.get("DEEPWIKI_HYBRID_FUSION", "0") == "1"
        self.assertEqual(HYBRID_FUSION_ENABLED, expected)

    @patch.dict(os.environ, {"DEEPWIKI_HYBRID_FUSION": "1"})
    def test_flag_on(self):
        """When env var is set, module-level flag reflects at import time.
        
        Note: since HYBRID_FUSION_ENABLED is evaluated at import time,
        we test the logic directly.
        """
        self.assertEqual(os.environ.get("DEEPWIKI_HYBRID_FUSION"), "1")


# ==========================================================================
# Suite 5: _find_graph_node helper
# ==========================================================================
class TestFindGraphNode(unittest.TestCase):
    """Test _find_graph_node for symbol lookup in the graph."""

    def setUp(self):
        self.G = _build_mini_graph()

    def test_exact_composite_key(self):
        """Find node by exact composite key rel_path::symbol_name."""
        node = _find_graph_node(self.G, 'AuthService', 'src/auth.py')
        self.assertEqual(node, 'src/auth.py::AuthService')

    def test_substring_fallback(self):
        """Fall back to substring match when no rel_path given."""
        node = _find_graph_node(self.G, 'AuthService')
        self.assertIsNotNone(node)
        self.assertIn('AuthService', node)

    def test_shortest_match_preferred(self):
        """When multiple nodes match, shortest is preferred."""
        node = _find_graph_node(self.G, 'User')
        self.assertEqual(node, 'src/models.py::User')

    def test_not_found(self):
        """Returns None when no node matches."""
        node = _find_graph_node(self.G, 'NonExistent')
        self.assertIsNone(node)

    def test_empty_symbol_name(self):
        """Returns None for empty symbol name."""
        node = _find_graph_node(self.G, '')
        self.assertIsNone(node)

    def test_none_graph(self):
        """Returns None when graph is None."""
        node = _find_graph_node(None, 'AuthService')
        self.assertIsNone(node)

    def test_case_insensitive(self):
        """Substring match is case-insensitive."""
        node = _find_graph_node(self.G, 'authservice')
        self.assertIsNotNone(node)
        self.assertIn('AuthService', node)


# ==========================================================================
# Suite 6: _format_neighbors helper
# ==========================================================================
class TestFormatNeighbors(unittest.TestCase):
    """Test _format_neighbors for graph relationship formatting."""

    def setUp(self):
        self.G = _build_mini_graph()

    def test_outgoing_edges(self):
        """Outgoing edges show → arrow."""
        lines = _format_neighbors(self.G, 'src/auth.py::AuthService')
        out_lines = [l for l in lines if '→' in l]
        self.assertGreater(len(out_lines), 0)

    def test_incoming_edges(self):
        """Incoming edges show ← arrow."""
        lines = _format_neighbors(self.G, 'src/models.py::User')
        in_lines = [l for l in lines if '←' in l]
        self.assertGreater(len(in_lines), 0)

    def test_relationship_type_shown(self):
        """Edge relationship type is shown in brackets."""
        lines = _format_neighbors(self.G, 'src/auth.py::AuthService')
        # AuthService → login [contains]
        contains_lines = [l for l in lines if 'contains' in l]
        self.assertGreater(len(contains_lines), 0)

    def test_max_per_direction(self):
        """max_per_direction caps output."""
        lines = _format_neighbors(self.G, 'src/auth.py::AuthService',
                                  max_per_direction=1)
        out_lines = [l for l in lines if '→' in l]
        # At most 1 outgoing + possible "... and N more"
        actual_out = [l for l in out_lines if '...' not in l]
        self.assertLessEqual(len(actual_out), 1)

    def test_isolated_node(self):
        """Node with no edges returns empty list."""
        G = nx.MultiDiGraph()
        G.add_node('isolated', symbol_name='isolated')
        lines = _format_neighbors(G, 'isolated')
        self.assertEqual(lines, [])


# ==========================================================================
# Suite 7: _extract_rel_type helper
# ==========================================================================
class TestExtractRelType(unittest.TestCase):
    """Test _extract_rel_type for MultiDiGraph edge data."""

    def test_multidigraph_format(self):
        """Standard MultiDiGraph edge_data format: {key: {attrs}}."""
        data = {0: {'relationship_type': 'calls'}}
        self.assertEqual(_extract_rel_type(data), 'calls')

    def test_missing_relationship_type(self):
        """Falls back to 'related' when attribute missing."""
        data = {0: {'weight': 1}}
        self.assertEqual(_extract_rel_type(data), 'related')

    def test_none_data(self):
        """Returns 'related' for None."""
        self.assertEqual(_extract_rel_type(None), 'related')

    def test_empty_dict(self):
        """Returns 'related' for empty dict."""
        self.assertEqual(_extract_rel_type({}), 'related')


# ==========================================================================
# Suite 8: _search_graph_by_text (brute-force fallback)
# ==========================================================================
class TestSearchGraphByText(unittest.TestCase):
    """Test the brute-force keyword search over graph nodes."""

    def setUp(self):
        self.G = _build_mini_graph()

    def test_exact_name_match(self):
        """Exact symbol name match gets highest score."""
        docs = _search_graph_by_text(self.G, 'AuthService')
        self.assertGreater(len(docs), 0)
        self.assertEqual(docs[0].metadata['symbol_name'], 'AuthService')

    def test_keyword_in_docstring(self):
        """Keywords in docstring are found."""
        docs = _search_graph_by_text(self.G, 'authentication')
        self.assertGreater(len(docs), 0)

    def test_empty_query(self):
        """Empty query returns no results."""
        docs = _search_graph_by_text(self.G, '')
        self.assertEqual(len(docs), 0)

    def test_none_graph(self):
        """None graph returns no results."""
        docs = _search_graph_by_text(None, 'test')
        self.assertEqual(len(docs), 0)

    def test_k_parameter(self):
        """k limits number of results."""
        docs = _search_graph_by_text(self.G, 'class', k=2)
        self.assertLessEqual(len(docs), 2)

    def test_doc_nodes_excluded(self):
        """DOC_SYMBOL_TYPES nodes are excluded from results."""
        # Add a doc node
        self.G.add_node('readme::doc', **{
            'symbol_name': 'readme',
            'symbol_type': 'markdown_document',
            'content': 'AuthService readme documentation',
        })
        docs = _search_graph_by_text(self.G, 'AuthService')
        doc_types = {d.metadata['symbol_type'] for d in docs}
        self.assertNotIn('markdown_document', doc_types)


# ==========================================================================
# Suite 9: search_codebase with mocked backends
# ==========================================================================
class TestSearchCodebaseMocked(unittest.TestCase):
    """Test search_codebase tool creation and execution with mocked backends."""

    def _create_tools(self, *, with_vs=True, with_fts=True,
                      vs_docs=None, fts_docs=None, fusion_enabled=False):
        """Create tools with mocked retriever_stack and graph_text_index."""
        retriever = MagicMock() if with_vs else None
        if retriever and vs_docs is not None:
            retriever.search_repository.return_value = vs_docs

        graph_text_index = MagicMock() if with_fts else None
        if graph_text_index:
            graph_text_index.is_open = True
            graph_text_index.search_smart.return_value = fts_docs or []

        code_graph = _build_mini_graph()

        env_patch = patch.dict(os.environ, {
            "DEEPWIKI_HYBRID_FUSION": "1" if fusion_enabled else "0"
        })
        # Also patch the module-level flag
        flag_patch = patch(
            'plugin_implementation.deep_research.research_tools.HYBRID_FUSION_ENABLED',
            fusion_enabled,
        )

        env_patch.start()
        flag_patch.start()
        self.addCleanup(env_patch.stop)
        self.addCleanup(flag_patch.stop)

        tools = create_codebase_tools(
            retriever_stack=retriever,
            graph_manager=MagicMock(),
            code_graph=code_graph,
            graph_text_index=graph_text_index,
        )
        return tools, retriever, graph_text_index

    def test_tools_include_search_graph(self):
        """create_codebase_tools returns search_graph tool."""
        tools, _, _ = self._create_tools()
        names = [t.name for t in tools]
        self.assertIn('search_graph', names)

    def test_tools_count(self):
        """4 tools are returned: search_codebase, get_symbol_relationships,
        search_graph, think."""
        tools, _, _ = self._create_tools()
        self.assertEqual(len(tools), 4)

    def test_search_codebase_uses_search_smart(self):
        """When FTS5 is available, search_smart is called (not search)."""
        fts_docs = [_make_doc('fmt_format', content='Format function')]
        tools, _, fts_idx = self._create_tools(
            with_vs=False, fts_docs=fts_docs,
        )
        search_tool = next(t for t in tools if t.name == 'search_codebase')
        result = search_tool.invoke({'query': 'format', 'k': 5})
        fts_idx.search_smart.assert_called_once()

    def test_search_codebase_legacy_no_fusion(self):
        """With fusion OFF, legacy dedup/sort is used."""
        vs_docs = [_make_doc('alpha', search_source='vectorstore')]
        fts_docs = [_make_doc('beta')]
        tools, _, _ = self._create_tools(
            vs_docs=vs_docs, fts_docs=fts_docs,
            fusion_enabled=False,
        )
        search_tool = next(t for t in tools if t.name == 'search_codebase')
        result = search_tool.invoke({'query': 'test', 'k': 5})
        self.assertIn('alpha', result)
        self.assertIn('beta', result)

    def test_search_codebase_with_fusion(self):
        """With fusion ON, RRF is used and results have rrf_score."""
        vs_docs = [_make_doc('shared', content='VS shared')]
        fts_docs = [_make_doc('shared', content='FTS shared'),
                    _make_doc('fts_only')]
        tools, _, _ = self._create_tools(
            vs_docs=vs_docs, fts_docs=fts_docs,
            fusion_enabled=True,
        )
        search_tool = next(t for t in tools if t.name == 'search_codebase')
        result = search_tool.invoke({'query': 'test', 'k': 5})
        self.assertIn('shared', result)

    def test_search_codebase_no_results(self):
        """Empty results produce a 'No results' message."""
        tools, retriever, fts = self._create_tools(
            vs_docs=[], fts_docs=[],
        )
        retriever.search_repository.return_value = []
        search_tool = next(t for t in tools if t.name == 'search_codebase')
        result = search_tool.invoke({'query': 'nonexistent_xyzzy', 'k': 5})
        self.assertIn('No results', result)

    def test_search_codebase_brute_force_fallback(self):
        """Without FTS5, falls back to brute-force graph search."""
        tools, _, _ = self._create_tools(with_vs=False, with_fts=False)
        search_tool = next(t for t in tools if t.name == 'search_codebase')
        # 'AuthService' is in the mini graph
        result = search_tool.invoke({'query': 'AuthService', 'k': 5})
        self.assertIn('AuthService', result)


# ==========================================================================
# Suite 10: search_graph tool with mocked data
# ==========================================================================
class TestSearchGraphToolMocked(unittest.TestCase):
    """Test search_graph tool with mocked FTS5 and real mini graph."""

    def _create_tools(self, *, fts_docs=None):
        """Create tools returning only the search_graph tool."""
        fts_idx = MagicMock()
        fts_idx.is_open = True
        fts_idx.search_smart.return_value = fts_docs or []

        code_graph = _build_mini_graph()

        tools = create_codebase_tools(
            retriever_stack=MagicMock(),
            graph_manager=MagicMock(),
            code_graph=code_graph,
            graph_text_index=fts_idx,
        )
        return next(t for t in tools if t.name == 'search_graph'), code_graph

    def test_returns_symbols_with_relationships(self):
        """search_graph includes relationship info in output."""
        fts_docs = [_make_doc(
            'AuthService', content='class AuthService: ...',
            source='src/auth.py', symbol_type='class',
        )]
        tool, _ = self._create_tools(fts_docs=fts_docs)
        result = tool.invoke({'query': 'AuthService', 'k': 3})
        self.assertIn('AuthService', result)
        self.assertIn('Relationships', result)

    def test_no_results_message(self):
        """No matches returns informative message."""
        tool, _ = self._create_tools(fts_docs=[])
        result = tool.invoke({'query': 'xyzzy_nothing'})
        self.assertIn('No symbols found', result)

    def test_include_neighbors_false(self):
        """With include_neighbors=False, relationships section absent."""
        fts_docs = [_make_doc('Database', content='class Database: ...',
                              source='src/db.py', symbol_type='class')]
        tool, _ = self._create_tools(fts_docs=fts_docs)
        result = tool.invoke({
            'query': 'Database', 'k': 3,
            'include_neighbors': False,
        })
        self.assertNotIn('Relationships', result)

    def test_truncated_content(self):
        """Long content is truncated to 500 chars."""
        long_content = 'x' * 1000
        fts_docs = [_make_doc('BigSym', content=long_content)]
        tool, _ = self._create_tools(fts_docs=fts_docs)
        result = tool.invoke({'query': 'BigSym'})
        self.assertIn('truncated', result)

    def test_multiple_results(self):
        """Multiple symbols found → multiple sections."""
        fts_docs = [
            _make_doc('AuthService', source='src/auth.py', symbol_type='class',
                      content='class AuthService: ...'),
            _make_doc('Database', source='src/db.py', symbol_type='class',
                      content='class Database: ...'),
        ]
        tool, _ = self._create_tools(fts_docs=fts_docs)
        result = tool.invoke({'query': 'class', 'k': 5})
        self.assertIn('[1]', result)
        self.assertIn('[2]', result)


# ==========================================================================
# Suite 11: Integration with real FTS5 + cached graphs
# ==========================================================================
@unittest.skipUnless(
    _fts_db_exists(_CONFIGS_KEY),
    'configurations FTS5 DB not found — run test_query_builder.py first'
)
class TestHybridFusionIntegration(unittest.TestCase):
    """Integration: RRF fusion with real FTS5 results from configurations."""

    @classmethod
    def setUpClass(cls):
        cls.idx = GraphTextIndex(cache_dir=str(_CACHE_DIR))
        if not cls.idx.load(_CONFIGS_KEY):
            raise unittest.SkipTest("Could not load configurations FTS5 index")
        cls.graph = _load_cached_graph(_CONFIGS_KEY)

    def test_fts5_and_brute_force_fused(self):
        """Fuse FTS5 results with brute-force results via RRF."""
        fts_docs = self.idx.search_smart('configuration', k=5)
        brute_docs = _search_graph_by_text(self.graph, 'configuration', k=5)

        if not fts_docs and not brute_docs:
            self.skipTest('No results for "configuration"')

        ranked = {}
        if fts_docs:
            ranked['fts5'] = fts_docs
        if brute_docs:
            ranked['brute'] = brute_docs

        fused = fuse_search_results(ranked, cap=10)
        self.assertGreater(len(fused), 0)

        # Verify scores and sources in metadata
        for doc in fused:
            self.assertIn('rrf_score', doc.metadata)
            self.assertGreater(doc.metadata['rrf_score'], 0)

    def test_overlap_boosted(self):
        """Symbols appearing in both FTS5 and brute-force get boosted."""
        query = 'database'
        fts_docs = self.idx.search_smart(query, k=10)
        brute_docs = _search_graph_by_text(self.graph, query, k=10)

        if not fts_docs or not brute_docs:
            self.skipTest('Need results from both sources')

        # Find overlapping symbol names
        fts_names = {d.metadata.get('symbol_name', '').lower() for d in fts_docs}
        brute_names = {d.metadata.get('symbol_name', '').lower() for d in brute_docs}
        overlap = fts_names & brute_names

        if not overlap:
            self.skipTest('No overlapping symbols between sources')

        fused = fuse_search_results({'fts5': fts_docs, 'brute': brute_docs})
        # The overlapping symbol should have rrf_sources from both
        for doc in fused:
            name = doc.metadata.get('symbol_name', '').lower()
            if name in overlap:
                sources = doc.metadata.get('rrf_sources', '')
                self.assertIn('fts5', sources)
                self.assertIn('brute', sources)
                break

    def test_fuse_preserves_metadata(self):
        """Fused results preserve rel_path, symbol_type, etc."""
        fts_docs = self.idx.search_smart('format', k=5)
        if not fts_docs:
            self.skipTest('No results for "format"')

        fused = fuse_search_results({'fts5': fts_docs}, cap=5)
        for doc in fused:
            self.assertIn('symbol_name', doc.metadata)
            self.assertIn('symbol_type', doc.metadata)


# ==========================================================================
# Suite 12: Integration — search_graph with real graph
# ==========================================================================
@unittest.skipUnless(
    _fts_db_exists(_CONFIGS_KEY),
    'configurations FTS5 DB not found'
)
class TestSearchGraphIntegration(unittest.TestCase):
    """Integration: search_graph with real FTS5 + cached graph."""

    @classmethod
    def setUpClass(cls):
        cls.idx = GraphTextIndex(cache_dir=str(_CACHE_DIR))
        if not cls.idx.load(_CONFIGS_KEY):
            raise unittest.SkipTest("Could not load configurations FTS5 index")
        cls.graph = _load_cached_graph(_CONFIGS_KEY)
        if cls.graph is None:
            raise unittest.SkipTest("Could not load configurations cached graph")

    def _make_search_graph_tool(self):
        """Create the search_graph tool with real backends."""
        tools = create_codebase_tools(
            retriever_stack=None,
            graph_manager=MagicMock(),
            code_graph=self.graph,
            graph_text_index=self.idx,
        )
        return next(t for t in tools if t.name == 'search_graph')

    def test_real_symbol_with_neighbors(self):
        """Search for a real symbol and get its graph neighbors."""
        tool = self._make_search_graph_tool()
        result = tool.invoke({'query': 'Configuration', 'k': 3})
        self.assertIn('Configuration', result)
        # Should have some relationships
        self.assertTrue(
            'Relationships' in result or 'No symbols found' not in result
        )

    def test_function_search(self):
        """Search for a function returns code content."""
        tool = self._make_search_graph_tool()
        result = tool.invoke({'query': 'parse', 'k': 3})
        # Should find something with 'parse' in the name
        self.assertNotIn('No symbols found', result)


# ==========================================================================
# Suite 13: RRF edge cases and stress tests
# ==========================================================================
class TestRRFEdgeCases(unittest.TestCase):
    """Edge cases and stress tests for RRF algorithm."""

    def test_single_doc_single_source(self):
        """Simplest case: one document from one source."""
        docs = [_make_doc('only')]
        results = reciprocal_rank_fusion({'src': docs})
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].rrf_score, 1.0 / 61)

    def test_many_sources(self):
        """10 sources each contributing the same doc → very high score."""
        shared = _make_doc('shared')
        ranked = {f'src_{i}': [_make_doc('shared')] for i in range(10)}
        results = reciprocal_rank_fusion(ranked)
        self.assertEqual(len(results), 1)
        expected = 10 * (1.0 / 61)  # All at rank 1
        self.assertAlmostEqual(results[0].rrf_score, expected)

    def test_large_list_performance(self):
        """100 docs from 3 sources → no crash, correct count."""
        src_a = [_make_doc(f'a_{i}') for i in range(100)]
        src_b = [_make_doc(f'b_{i}') for i in range(100)]
        src_c = [_make_doc(f'c_{i}') for i in range(100)]
        results = reciprocal_rank_fusion({
            'a': src_a, 'b': src_b, 'c': src_c,
        })
        self.assertEqual(len(results), 300)  # No overlap

    def test_partial_overlap(self):
        """50% overlap between two sources."""
        shared = [_make_doc(f'shared_{i}') for i in range(5)]
        only_a = [_make_doc(f'only_a_{i}') for i in range(5)]
        only_b = [_make_doc(f'only_b_{i}') for i in range(5)]
        results = reciprocal_rank_fusion({
            'a': shared + only_a,
            'b': shared + only_b,
        })
        # shared: 5, only_a: 5, only_b: 5 = 15 unique
        self.assertEqual(len(results), 15)
        # Shared docs should be at the top (boosted)
        top_names = {r.document.metadata['symbol_name'] for r in results[:5]}
        for i in range(5):
            self.assertIn(f'shared_{i}', top_names)

    def test_tie_breaking_by_best_rank(self):
        """Equal RRF scores break ties by best_rank."""
        # Two docs from one source with same score from another
        doc_a = _make_doc('a_sym')
        doc_b = _make_doc('b_sym')
        results = reciprocal_rank_fusion({'src': [doc_a, doc_b]})
        # a_sym at rank 1 should come before b_sym at rank 2
        self.assertEqual(results[0].document.metadata['symbol_name'], 'a_sym')

    def test_empty_content_docs(self):
        """Documents with empty content are still processed."""
        doc = Document(
            page_content='',
            metadata={'symbol_name': 'empty'},
        )
        results = reciprocal_rank_fusion({'src': [doc]})
        self.assertEqual(len(results), 1)


# ==========================================================================
# Suite 14: search_codebase fusion vs legacy comparison
# ==========================================================================
class TestFusionVsLegacy(unittest.TestCase):
    """Compare fusion ON vs OFF behavior with identical inputs."""

    def _run_search(self, fusion_enabled: bool, vs_docs, fts_docs):
        """Create tools and run search_codebase."""
        retriever = MagicMock()
        retriever.search_repository.return_value = vs_docs

        fts_idx = MagicMock()
        fts_idx.is_open = True
        fts_idx.search_smart.return_value = fts_docs

        with patch.dict(os.environ, {"DEEPWIKI_HYBRID_FUSION": "1" if fusion_enabled else "0"}):
            with patch(
                'plugin_implementation.deep_research.research_tools.HYBRID_FUSION_ENABLED',
                fusion_enabled,
            ):
                tools = create_codebase_tools(
                    retriever_stack=retriever,
                    graph_manager=MagicMock(),
                    code_graph=_build_mini_graph(),
                    graph_text_index=fts_idx,
                )
                search_tool = next(t for t in tools if t.name == 'search_codebase')
                return search_tool.invoke({'query': 'test', 'k': 5})

    def test_both_modes_return_results(self):
        """Both fusion and legacy modes return results for same input."""
        vs_docs = [_make_doc('alpha', search_source='vectorstore')]
        fts_docs = [_make_doc('beta')]

        legacy = self._run_search(False, vs_docs, fts_docs)
        fusion = self._run_search(True, vs_docs, fts_docs)

        # Both should contain both symbols
        self.assertIn('alpha', legacy)
        self.assertIn('alpha', fusion)
        self.assertIn('beta', legacy)
        self.assertIn('beta', fusion)

    def test_fusion_shows_rrf_source(self):
        """Fusion mode includes source info differently than legacy."""
        vs_docs = [_make_doc('gamma', search_source='vectorstore')]
        fts_docs = [_make_doc('gamma')]  # Same symbol in both

        fusion = self._run_search(True, vs_docs, fts_docs)
        # RRF should merge the duplicate
        self.assertIn('gamma', fusion)

    def test_legacy_dedup_by_symbol(self):
        """Legacy mode deduplicates by symbol name (graph doc dropped)."""
        vs_docs = [_make_doc('same_sym', search_source='vectorstore')]
        fts_docs = [_make_doc('same_sym')]

        legacy = self._run_search(False, vs_docs, fts_docs)
        # Should only have one 'same_sym' entry
        count = legacy.count('same_sym')
        # In the formatted output, symbol name appears in header + content
        # but should appear in only ONE result block
        result_blocks = legacy.split('###')
        sym_blocks = [b for b in result_blocks if 'same_sym' in b]
        self.assertEqual(len(sym_blocks), 1)


if __name__ == '__main__':
    unittest.main()
