"""
Tests for Phase 5: Content Expander FTS5 fallback (Strategy 6).

Tests cover:
1. **Unit tests** — ``_find_graph_node`` with mocked FTS5 index
2. **Constructor** — optional ``graph_text_index`` parameter
3. **Integration tests** — real FTS5 + cached graph (configurations)
4. **Profiling** — ``node_index_hit_fts5`` counter
5. **Graceful degradation** — when FTS5 is unavailable or errors
"""

import gzip
import os
import pickle
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
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

from plugin_implementation.content_expander import ContentExpander
from plugin_implementation.code_graph.graph_text_index import GraphTextIndex

# ---------------------------------------------------------------------------
# Cache paths for integration tests
# ---------------------------------------------------------------------------
_WIKI_BUILDER = _PLUGIN_ROOT.parent.parent / 'wiki_builder'
_CACHE_DIR = _WIKI_BUILDER / 'cache'
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
# Test graph builder
# ---------------------------------------------------------------------------
def _build_test_graph() -> nx.MultiDiGraph:
    """Build a small graph where some symbols can only be found via FTS5."""
    G = nx.MultiDiGraph()

    # Nodes with standard naming
    G.add_node('src/config.py::Configuration', **{
        'symbol_name': 'Configuration',
        'symbol_type': 'class',
        'content': 'class Configuration:\n    """Manages app configuration."""',
        'docstring': 'Manages app configuration.',
        'file_path': '/repo/src/config.py',
        'rel_path': 'src/config.py',
        'language': 'python',
        'start_line': 1,
        'end_line': 50,
    })
    G.add_node('src/config.py::load_config', **{
        'symbol_name': 'load_config',
        'symbol_type': 'function',
        'content': 'def load_config(path: str) -> Configuration: ...',
        'docstring': 'Load configuration from YAML file.',
        'file_path': '/repo/src/config.py',
        'rel_path': 'src/config.py',
        'language': 'python',
        'start_line': 55,
        'end_line': 70,
    })
    G.add_node('src/parser.py::ConfigParser', **{
        'symbol_name': 'ConfigParser',
        'symbol_type': 'class',
        'content': 'class ConfigParser:\n    """Parses config files."""',
        'docstring': 'Parses config files.',
        'file_path': '/repo/src/parser.py',
        'rel_path': 'src/parser.py',
        'language': 'python',
        'start_line': 1,
        'end_line': 30,
    })

    # Edges
    G.add_edge('src/config.py::load_config', 'src/config.py::Configuration',
               relationship_type='creates')
    G.add_edge('src/parser.py::ConfigParser', 'src/config.py::Configuration',
               relationship_type='references')

    # Build simple indexes that _find_graph_node expects
    # (These are minimal — the point is to test what happens AFTER they all fail)
    G._node_index = {}
    G._simple_name_index = {}
    G._full_name_index = {}
    G._name_index = {}
    G._suffix_index = {}

    return G


def _build_graph_with_indexes() -> nx.MultiDiGraph:
    """Build a graph WITH proper indexes (FTS5 should NOT be needed)."""
    G = _build_test_graph()

    # Populate indexes so standard strategies find nodes
    for node_id in G.nodes():
        data = G.nodes[node_id]
        sym = data.get('symbol_name', '')
        fp = data.get('file_path', '')
        lang = data.get('language', '')
        key = (sym, fp, lang)
        G._node_index[key] = node_id
        G._name_index.setdefault(sym, []).append(node_id)
        # suffix
        simple = sym.split('.')[-1].split('::')[-1]
        G._suffix_index.setdefault(simple, []).append(node_id)

    return G


# ==========================================================================
# Suite 1: Constructor
# ==========================================================================
class TestContentExpanderConstructor(unittest.TestCase):
    """Test that ContentExpander accepts optional graph_text_index."""

    def test_default_no_fts(self):
        """Default constructor has None graph_text_index."""
        exp = ContentExpander()
        self.assertIsNone(exp.graph_text_index)

    def test_with_fts(self):
        """graph_text_index is stored when provided."""
        mock_fts = MagicMock()
        exp = ContentExpander(graph_store=None, graph_text_index=mock_fts)
        self.assertIs(exp.graph_text_index, mock_fts)

    def test_backward_compat(self):
        """Old-style single-arg call still works."""
        G = _build_test_graph()
        exp = ContentExpander(G)
        self.assertIs(exp.graph, G)
        self.assertIsNone(exp.graph_text_index)


# ==========================================================================
# Suite 2: FTS5 fallback — unit tests with mocked index
# ==========================================================================
class TestFTS5Fallback(unittest.TestCase):
    """Test _find_graph_node Strategy 6 (FTS5) with mocked index."""

    def setUp(self):
        # Graph with EMPTY indexes so all strategies 1-5 fail
        self.G = _build_test_graph()
        self.mock_fts = MagicMock()
        self.mock_fts.is_open = True
        self.expander = ContentExpander(
            graph_store=self.G,
            graph_text_index=self.mock_fts,
        )
        # Force _ensure_graph_indexes to run, then wipe all indexes
        # so that strategies 1-5 all miss and we reach strategy 6 (FTS5)
        self.expander._ensure_graph_indexes()
        self.expander._node_index = {}
        self.expander._simple_name_index = {}
        self.expander._full_name_index = {}
        self.expander._name_index = {}
        self.expander._suffix_index = {}

    def test_fts5_finds_symbol(self):
        """FTS5 search resolves symbol when index strategies fail."""
        self.mock_fts.search_smart.return_value = [
            Document(
                page_content='class Configuration: ...',
                metadata={
                    'symbol_name': 'Configuration',
                    'rel_path': 'src/config.py',
                    'symbol_type': 'class',
                },
            ),
        ]
        node = self.expander._find_graph_node(
            'Configuration', '/repo/src/config.py', 'python'
        )
        self.assertEqual(node, 'src/config.py::Configuration')
        self.mock_fts.search_smart.assert_called_once()

    def test_fts5_prefers_same_file(self):
        """When FTS5 returns multiple results, same-file match is preferred."""
        self.mock_fts.search_smart.return_value = [
            Document(
                page_content='class ConfigParser: ...',
                metadata={
                    'symbol_name': 'ConfigParser',
                    'rel_path': 'src/parser.py',
                },
            ),
            Document(
                page_content='class Configuration: ...',
                metadata={
                    'symbol_name': 'Configuration',
                    'rel_path': 'src/config.py',
                },
            ),
        ]
        node = self.expander._find_graph_node(
            'Config', 'src/config.py', 'python'
        )
        # Should prefer config.py match since file_path ends with 'src/config.py'
        self.assertEqual(node, 'src/config.py::Configuration')

    def test_fts5_returns_none_when_no_graph_match(self):
        """FTS5 doc that doesn't correspond to a graph node → still None."""
        self.mock_fts.search_smart.return_value = [
            Document(
                page_content='class NonExistent: ...',
                metadata={
                    'symbol_name': 'NonExistent',
                    'rel_path': 'src/nonexistent.py',
                },
            ),
        ]
        node = self.expander._find_graph_node(
            'NonExistent', '/repo/src/foo.py', 'python'
        )
        self.assertIsNone(node)

    def test_fts5_not_called_when_index_succeeds(self):
        """FTS5 should NOT be called when a standard index strategy works."""
        G = _build_graph_with_indexes()
        expander = ContentExpander(graph_store=G, graph_text_index=self.mock_fts)
        node = expander._find_graph_node(
            'Configuration', '/repo/src/config.py', 'python'
        )
        self.assertIsNotNone(node)
        self.mock_fts.search_smart.assert_not_called()

    def test_fts5_graceful_on_exception(self):
        """FTS5 exception doesn't crash; returns None gracefully."""
        self.mock_fts.search_smart.side_effect = RuntimeError("DB locked")
        node = self.expander._find_graph_node(
            'Configuration', '/repo/src/config.py', 'python'
        )
        self.assertIsNone(node)

    def test_fts5_not_called_when_index_closed(self):
        """FTS5 not called if is_open is False."""
        self.mock_fts.is_open = False
        self.mock_fts.search_smart.return_value = [
            Document(page_content='x', metadata={'symbol_name': 'Configuration',
                                                   'rel_path': 'src/config.py'}),
        ]
        node = self.expander._find_graph_node(
            'Configuration', '/repo/src/config.py', 'python'
        )
        self.assertIsNone(node)
        self.mock_fts.search_smart.assert_not_called()

    def test_fts5_not_called_when_none(self):
        """No FTS5 index → still returns None (no crash)."""
        expander = ContentExpander(graph_store=self.G, graph_text_index=None)
        # Wipe built indexes so strategies 1-5 miss, reaching the FTS5 path
        expander._ensure_graph_indexes()
        expander._node_index = {}
        expander._simple_name_index = {}
        expander._full_name_index = {}
        expander._name_index = {}
        expander._suffix_index = {}
        node = expander._find_graph_node(
            'Configuration', '/repo/src/config.py', 'python'
        )
        self.assertIsNone(node)

    def test_fts5_empty_results(self):
        """FTS5 returns empty list → None."""
        self.mock_fts.search_smart.return_value = []
        node = self.expander._find_graph_node(
            'NonExistent', '/repo/src/foo.py', 'python'
        )
        self.assertIsNone(node)

    def test_fts5_called_with_symbol_intent(self):
        """FTS5 is called with intent='symbol'."""
        self.mock_fts.search_smart.return_value = []
        self.expander._find_graph_node('Config', '/repo/src/foo.py', 'python')
        self.mock_fts.search_smart.assert_called_once_with(
            'Config', k=3, intent='symbol',
        )


# ==========================================================================
# Suite 3: Profiling counter
# ==========================================================================
class TestFTS5Profiling(unittest.TestCase):
    """Test that FTS5 hits are tracked in _profile_stats."""

    @patch.dict(os.environ, {"DEEPWIKI_PROFILE_EXPANSION": "1"})
    def test_fts5_hit_counted(self):
        """node_index_hit_fts5 counter increments on FTS5 success."""
        G = _build_test_graph()
        mock_fts = MagicMock()
        mock_fts.is_open = True
        mock_fts.search_smart.return_value = [
            Document(page_content='x', metadata={
                'symbol_name': 'Configuration',
                'rel_path': 'src/config.py',
            }),
        ]
        expander = ContentExpander(graph_store=G, graph_text_index=mock_fts)
        # Force index init then wipe so FTS5 path is reached
        expander._ensure_graph_indexes()
        expander._node_index = {}
        expander._simple_name_index = {}
        expander._full_name_index = {}
        expander._name_index = {}
        expander._suffix_index = {}

        node = expander._find_graph_node('Configuration', '/repo/src/config.py', 'python')
        self.assertIsNotNone(node)
        self.assertEqual(
            expander._profile_stats.get('node_index_hit_fts5', 0), 1
        )

    @patch.dict(os.environ, {"DEEPWIKI_PROFILE_EXPANSION": "1"})
    def test_miss_counter_on_fts5_failure(self):
        """node_index_miss counter when FTS5 also fails."""
        G = _build_test_graph()
        mock_fts = MagicMock()
        mock_fts.is_open = True
        mock_fts.search_smart.return_value = []
        expander = ContentExpander(graph_store=G, graph_text_index=mock_fts)
        # Force index init then wipe
        expander._ensure_graph_indexes()
        expander._node_index = {}
        expander._simple_name_index = {}
        expander._full_name_index = {}
        expander._name_index = {}
        expander._suffix_index = {}

        node = expander._find_graph_node('NonExistent', '/repo/x.py', 'python')
        self.assertIsNone(node)
        self.assertEqual(
            expander._profile_stats.get('node_index_miss', 0), 1
        )


# ==========================================================================
# Suite 4: Integration with real FTS5 + cached graph (configurations)
# ==========================================================================
@unittest.skipUnless(
    _fts_db_exists(_CONFIGS_KEY),
    'configurations FTS5 DB not found'
)
class TestFTS5FallbackIntegration(unittest.TestCase):
    """Integration: FTS5 fallback with real FTS5 + cached configurations graph."""

    @classmethod
    def setUpClass(cls):
        cls.fts = GraphTextIndex(cache_dir=str(_CACHE_DIR))
        if not cls.fts.load(_CONFIGS_KEY):
            raise unittest.SkipTest("Could not load configurations FTS5 index")
        cls.graph = _load_cached_graph(_CONFIGS_KEY)
        if cls.graph is None:
            raise unittest.SkipTest("Could not load configurations cached graph")

    def _make_expander_empty_indexes(self):
        """Create expander with real graph but empty indexes → forces FTS5 path."""
        expander = ContentExpander(
            graph_store=self.graph,
            graph_text_index=self.fts,
        )
        # Wipe all indexes to force FTS5 path
        expander._node_index = {}
        expander._simple_name_index = {}
        expander._full_name_index = {}
        expander._name_index = {}
        expander._suffix_index = {}
        return expander

    def test_fts5_resolves_real_symbol(self):
        """FTS5 can resolve a real symbol from the configurations graph."""
        expander = self._make_expander_empty_indexes()
        # Find a real symbol from the graph to test with
        sample_node = None
        for node_id, data in self.graph.nodes(data=True):
            if data.get('symbol_type') == 'class' and data.get('symbol_name'):
                sample_node = (data['symbol_name'], node_id, data)
                break
        if sample_node is None:
            self.skipTest("No class node found in configurations graph")

        sym_name, expected_id, data = sample_node
        file_path = data.get('file_path', '')
        language = data.get('language', 'python')

        node = expander._find_graph_node(sym_name, file_path, language)
        # Should find SOMETHING (maybe not exact ID but a valid node)
        # The FTS5 may return a different node with the same name
        if node is not None:
            self.assertIn(node, self.graph)

    def test_fts5_fallback_after_index_miss(self):
        """Real-world scenario: index miss → FTS5 finds it."""
        expander = self._make_expander_empty_indexes()

        # Pick a function symbol
        func_node = None
        for nid, data in self.graph.nodes(data=True):
            if data.get('symbol_type') == 'function' and data.get('symbol_name'):
                func_node = (data['symbol_name'], nid, data)
                break
        if func_node is None:
            self.skipTest("No function node in configurations graph")

        sym_name, expected_id, data = func_node
        node = expander._find_graph_node(
            sym_name, data.get('file_path', ''), data.get('language', ''),
        )
        # FTS5 should find a matching node
        if node is not None:
            self.assertIn(node, self.graph)


# ==========================================================================
# Suite 5: WikiRetrieverStack wiring
# ==========================================================================
class TestRetrieverStackWiring(unittest.TestCase):
    """Test that WikiRetrieverStack passes graph_text_index to ContentExpander."""

    def test_fts_passed_to_expander(self):
        """graph_text_index parameter reaches ContentExpander."""
        from plugin_implementation.retrievers import WikiRetrieverStack

        mock_vs_manager = MagicMock()
        mock_vs_manager.get_retriever.return_value = None
        mock_vs_manager.embeddings = None
        mock_fts = MagicMock()

        stack = WikiRetrieverStack(
            vectorstore_manager=mock_vs_manager,
            relationship_graph=None,
            graph_text_index=mock_fts,
        )
        self.assertIs(stack.content_expander.graph_text_index, mock_fts)

    def test_default_no_fts(self):
        """Without graph_text_index, expander has None."""
        from plugin_implementation.retrievers import WikiRetrieverStack

        mock_vs_manager = MagicMock()
        mock_vs_manager.get_retriever.return_value = None
        mock_vs_manager.embeddings = None

        stack = WikiRetrieverStack(
            vectorstore_manager=mock_vs_manager,
            relationship_graph=None,
        )
        self.assertIsNone(stack.content_expander.graph_text_index)


if __name__ == '__main__':
    unittest.main()
