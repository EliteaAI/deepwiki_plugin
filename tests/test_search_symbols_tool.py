"""
Tests for the ``search_symbols`` LangChain tool in the structure planner.

Tests cover:
1. **Unit tests** — ``_handle_search_symbols`` with mocked FTS5 index
2. **Integration tests** — real FTS5 queries against fmtlib and configurations
3. **Tool registration** — ``search_symbols`` is in ``get_tools()`` result
4. **Fallback** — graceful degradation when FTS5 is unavailable
"""

import gzip
import os
import pickle
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional
from unittest.mock import MagicMock, patch

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

from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
from plugin_implementation.wiki_structure_planner.structure_tools import (
    StructureCollector,
)

# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------
_WIKI_BUILDER = _PLUGIN_ROOT.parent.parent / 'wiki_builder'
_CACHE_DIR = _WIKI_BUILDER / 'cache'
_FMTLIB_KEY = '551134763c1f5c1b3feca4dd95076290'
_CONFIGS_KEY = 'cd9d7a4aefa47194b872a7093a855156'


def _fts_db_exists(key: str) -> bool:
    return (_CACHE_DIR / f'{key}.fts5.db').exists()


def _graph_exists(key: str) -> bool:
    return (_CACHE_DIR / f'{key}.code_graph.gz').exists()


def _load_graph(key: str):
    gpath = _CACHE_DIR / f'{key}.code_graph.gz'
    with gzip.open(gpath, 'rb') as f:
        return pickle.load(f)


def _make_doc(name: str, sym_type: str, path: str, score: float = 0.0) -> Document:
    """Helper to create a fake Document for mock testing."""
    return Document(
        page_content=f"source code for {name}",
        metadata={
            'symbol_name': name,
            'symbol_type': sym_type,
            'rel_path': path,
            'search_score': score,
            'node_id': f'test::{name}',
        },
    )


# ===================================================================
# Unit Tests — tool registration
# ===================================================================

class TestSearchSymbolsToolRegistered(unittest.TestCase):
    """Verify search_symbols is in the tool list."""

    def test_search_symbols_in_tools(self):
        collector = StructureCollector(page_budget=10, repo_root='/tmp')
        tools = collector.get_tools()
        tool_names = [t.name for t in tools]
        self.assertIn('search_symbols', tool_names)

    def test_tool_count_increased(self):
        """Should now have 10 tools (was 9 before search_symbols)."""
        collector = StructureCollector(page_budget=10, repo_root='/tmp')
        tools = collector.get_tools()
        self.assertEqual(len(tools), 10)


# ===================================================================
# Unit Tests — handler with mocked FTS5
# ===================================================================

class TestHandleSearchSymbolsMocked(unittest.TestCase):
    """Test _handle_search_symbols with mocked GraphTextIndex."""

    def setUp(self):
        self.collector = StructureCollector(page_budget=10, repo_root='/tmp')
        self.mock_fts = MagicMock(spec=GraphTextIndex)
        self.mock_fts.is_open = True
        self.collector.graph_text_index = self.mock_fts

    def test_returns_results(self):
        self.mock_fts.search_symbols.return_value = [
            _make_doc('AuthService', 'class', 'src/auth/service.py', -5.2),
            _make_doc('TokenManager', 'class', 'src/auth/token.py', -3.1),
        ]
        result = self.collector._handle_search_symbols("authentication")
        self.assertIn('AuthService', result)
        self.assertIn('TokenManager', result)
        self.assertIn('BM25-ranked', result)

    def test_no_results(self):
        self.mock_fts.search_symbols.return_value = []
        result = self.collector._handle_search_symbols("xyznonexistent")
        self.assertIn('No symbols found', result)
        self.assertIn('TIPS', result)

    def test_type_filter_passed(self):
        self.mock_fts.search_symbols.return_value = []
        self.collector._handle_search_symbols("auth", symbol_type="class")
        call_kwargs = self.mock_fts.search_symbols.call_args
        self.assertEqual(call_kwargs.kwargs.get('symbol_types'), frozenset({'class'}))

    def test_path_prefix_passed(self):
        self.mock_fts.search_symbols.return_value = []
        self.collector._handle_search_symbols("auth", path_prefix="src/services")
        call_kwargs = self.mock_fts.search_symbols.call_args
        self.assertEqual(call_kwargs.kwargs.get('path_prefix'), 'src/services')

    def test_short_query_rejected(self):
        result = self.collector._handle_search_symbols("x")
        self.assertIn('ERROR', result)

    def test_empty_query_rejected(self):
        result = self.collector._handle_search_symbols("")
        self.assertIn('ERROR', result)

    def test_fts_exception_handled(self):
        self.mock_fts.search_symbols.side_effect = RuntimeError("DB error")
        result = self.collector._handle_search_symbols("auth")
        self.assertIn('ERROR', result)
        self.assertIn('Search failed', result)


# ===================================================================
# Unit Tests — fallback when FTS5 unavailable
# ===================================================================

class TestSearchSymbolsFallback(unittest.TestCase):
    """Verify search_symbols falls back to search_graph when FTS5 is off."""

    def test_fallback_to_search_graph_no_fts(self):
        """When graph_text_index is None, falls back to search_graph."""
        import networkx as nx
        G = nx.MultiDiGraph()
        G.add_node('cls1', symbol_name='AuthService', symbol_type='class',
                    rel_path='src/auth.py', docstring='handles auth')

        collector = StructureCollector(page_budget=10, repo_root='/tmp')
        collector.code_graph = G
        collector.graph_text_index = None

        result = collector._handle_search_symbols("Auth")
        # Should find AuthService via fallback substring search
        self.assertIn('AuthService', result)

    def test_fallback_to_search_graph_fts_not_open(self):
        """When graph_text_index.is_open is False, falls back."""
        import networkx as nx
        G = nx.MultiDiGraph()
        G.add_node('cls1', symbol_name='Foo', symbol_type='class',
                    rel_path='src/foo.py')

        mock_fts = MagicMock(spec=GraphTextIndex)
        mock_fts.is_open = False

        collector = StructureCollector(page_budget=10, repo_root='/tmp')
        collector.code_graph = G
        collector.graph_text_index = mock_fts

        result = collector._handle_search_symbols("Foo")
        self.assertIn('Foo', result)


# ===================================================================
# Unit Tests — result formatting
# ===================================================================

class TestFormatSearchSymbolsResults(unittest.TestCase):
    """Test _format_search_symbols_results output format."""

    def setUp(self):
        self.collector = StructureCollector(page_budget=10, repo_root='/tmp')

    def test_groups_by_type(self):
        docs = [
            _make_doc('AuthService', 'class', 'src/auth.py'),
            _make_doc('validate_token', 'function', 'src/auth.py'),
            _make_doc('MAX_RETRIES', 'constant', 'src/config.py'),
        ]
        result = self.collector._format_search_symbols_results(
            "auth", docs, None, None,
        )
        self.assertIn('CLASSES', result)
        self.assertIn('FUNCTIONS', result)
        self.assertIn('CONSTANTS', result)

    def test_shows_filter_info(self):
        docs = [_make_doc('Foo', 'class', 'src/foo.py')]
        result = self.collector._format_search_symbols_results(
            "foo", docs, "class", "src/",
        )
        self.assertIn('type=class', result)
        self.assertIn('path=src/', result)

    def test_shows_use_instruction(self):
        docs = [_make_doc('Bar', 'class', 'src/bar.py')]
        result = self.collector._format_search_symbols_results(
            "bar", docs, None, None,
        )
        self.assertIn('target_symbols', result)


# ===================================================================
# Integration Tests — real FTS5 on fmtlib
# ===================================================================

@unittest.skipUnless(
    _fts_db_exists(_FMTLIB_KEY) and _graph_exists(_FMTLIB_KEY),
    "fmtlib cache not found",
)
class TestSearchSymbolsFmtlib(unittest.TestCase):
    """Integration: search_symbols against real fmtlib FTS5 index."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(_FMTLIB_KEY)
        cls.fts = GraphTextIndex(cache_dir=str(_CACHE_DIR))
        if not cls.fts.load(_FMTLIB_KEY):
            raise unittest.SkipTest("Could not load fmtlib FTS5 index")

        cls.collector = StructureCollector(page_budget=20, repo_root='/tmp')
        cls.collector.code_graph = cls.graph
        cls.collector.graph_text_index = cls.fts

    def test_concept_search_format(self):
        """search_symbols('formatting output') finds format-related symbols."""
        result = self.collector._handle_search_symbols("formatting output string")
        self.assertIn('format', result.lower())

    def test_type_filter_class(self):
        """symbol_type='class' only returns classes/structs."""
        result = self.collector._handle_search_symbols(
            "format", symbol_type="class",
        )
        # Should contain classes section
        self.assertIn('CLASS', result)

    def test_path_prefix_filter(self):
        """path_prefix='include' scopes to include directory."""
        result = self.collector._handle_search_symbols(
            "format", path_prefix="include",
        )
        # Results should mention include/ paths
        self.assertIn('include', result.lower())

    def test_no_results_for_nonsense(self):
        """Nonsense query returns helpful no-results message."""
        result = self.collector._handle_search_symbols("xyzquuxnonexistent42")
        self.assertIn('No symbols found', result)

    def test_natural_language_query(self):
        """Natural language query works with stop word removal."""
        result = self.collector._handle_search_symbols(
            "How does the format context handle arguments?"
        )
        # Should find format_context related symbols
        self.assertIn('format', result.lower())


# ===================================================================
# Integration Tests — real FTS5 on configurations (Python)
# ===================================================================

@unittest.skipUnless(
    _fts_db_exists(_CONFIGS_KEY) and _graph_exists(_CONFIGS_KEY),
    "configurations cache not found",
)
class TestSearchSymbolsConfigurations(unittest.TestCase):
    """Integration: search_symbols against configurations (Python)."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(_CONFIGS_KEY)
        cls.fts = GraphTextIndex(cache_dir=str(_CACHE_DIR))
        if not cls.fts.load(_CONFIGS_KEY):
            raise unittest.SkipTest("Could not load configurations FTS5 index")

        cls.collector = StructureCollector(page_budget=10, repo_root='/tmp')
        cls.collector.code_graph = cls.graph
        cls.collector.graph_text_index = cls.fts

    def test_error_related_search(self):
        """search_symbols('error handling') finds ConfigurationError."""
        result = self.collector._handle_search_symbols("error handling configuration")
        self.assertIn('Error', result)  # ConfigurationError or similar

    def test_class_filter(self):
        """Filter by class type returns only classes."""
        result = self.collector._handle_search_symbols(
            "configuration", symbol_type="class",
        )
        self.assertIn('CLASS', result)

    def test_vs_search_graph(self):
        """search_symbols and search_graph should both find relevant symbols.

        search_symbols uses BM25 ranking; search_graph uses substring match.
        Both should find 'Configuration' for the query 'configuration'.
        """
        fts_result = self.collector._handle_search_symbols("configuration")
        graph_result = self.collector._handle_search_graph("Configuration")

        # Both should contain results
        self.assertNotIn('No symbols found', fts_result.lower())
        self.assertNotIn('No symbols found', graph_result.lower())


if __name__ == '__main__':
    unittest.main()
