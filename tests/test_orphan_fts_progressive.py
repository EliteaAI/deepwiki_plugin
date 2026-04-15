"""
Tests for the orphan FTS fallback in progressive disclosure tools and
get_code empty-content fixes.

Covers:
1. _orphan_fts_fallback helper — method extraction, FTS search, formatting
2. get_relationships_tool integration — orphan fallback triggers when rels=[]
3. Container type coverage — class, interface, struct, enum, trait
4. Edge cases — no FTS index, no methods, non-container types, dunder methods
5. get_relationships defines-only — FTS fallback also triggers for containers
   whose only edges are structural 'defines' (effectively orphan)
6. get_code empty content — FTS5 fallback, symbol.source_text, informative msg
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# Helpers for building synthetic graphs
# ---------------------------------------------------------------------------

def _node(name: str, sym_type: str, **extra) -> dict:
    return {'symbol_name': name, 'name': name, 'symbol_type': sym_type, **extra}


def _container(name: str, sym_type: str = 'class') -> dict:
    return _node(name, sym_type, rel_path=f'src/{name.lower()}.py')


def _method(name: str) -> dict:
    return _node(name, 'method')


def _function(name: str) -> dict:
    return _node(name, 'function', rel_path=f'src/{name.lower()}.py')


@dataclass
class FakeRel:
    """Mimics RelationshipResult from GraphQueryService."""
    source_name: str
    target_name: str
    relationship_type: str
    source_type: str = ''
    target_type: str = ''
    hop_distance: int = 1


def _add_defines(G: nx.MultiDiGraph, parent: str, child: str):
    G.add_edge(parent, child, relationship_type='defines')


def _make_orphan_graph(sym_name='Event', sym_type='class', methods=None):
    """Build a graph with a container that has methods but no external edges."""
    G = nx.MultiDiGraph()
    G.add_node(sym_name, **_container(sym_name, sym_type))
    for m in (methods or ['on_startup', 'on_shutdown']):
        G.add_node(f'{sym_name}::{m}', **_method(m))
        _add_defines(G, sym_name, f'{sym_name}::{m}')
    return G


class _FakeFTSDoc:
    """Minimal FTS result doc."""
    def __init__(self, node_id, symbol_name, symbol_type='class', rel_path='src/foo.py'):
        self.metadata = {
            'node_id': node_id,
            'symbol_name': symbol_name,
            'symbol_type': symbol_type,
            'rel_path': rel_path,
        }


class _FakeFTSIndex:
    """Minimal FTS5 index mock with configurable search results."""
    def __init__(self, results_by_query=None):
        self.is_open = True
        self._results = results_by_query or {}

    def search(self, query, k=3, symbol_types=None):
        return self._results.get(query, [])


# ===================================================================
# 1. _orphan_fts_fallback unit tests
# ===================================================================

class TestOrphanFtsFallback(unittest.TestCase):
    """Direct tests for _orphan_fts_fallback helper."""

    def _get_fallback(self):
        """Import the helper from research_tools closure.

        Since _orphan_fts_fallback is a closure-local function inside
        create_codebase_tools, we test it indirectly through
        get_relationships_tool. But we can also extract the logic
        by importing the module-level CONTAINER_SYMBOL_TYPES constant.
        """
        from plugin_implementation.deep_research.research_tools import CONTAINER_SYMBOL_TYPES
        return CONTAINER_SYMBOL_TYPES

    def test_container_types_complete(self):
        """All five container types should be included."""
        types = self._get_fallback()
        for t in ('class', 'interface', 'struct', 'enum', 'trait'):
            self.assertIn(t, types, f"'{t}' missing from CONTAINER_SYMBOL_TYPES")

    def test_non_container_excluded(self):
        """Functions, methods, constants should NOT be containers."""
        types = self._get_fallback()
        for t in ('function', 'method', 'constant', 'macro', 'module_doc'):
            self.assertNotIn(t, types, f"'{t}' should not be in CONTAINER_SYMBOL_TYPES")


# ===================================================================
# 2. Orphan FTS via get_relationships integration tests
# ===================================================================

class TestOrphanFtsInGetRelationships(unittest.TestCase):
    """Test orphan FTS fallback triggers inside get_relationships_tool."""

    def _build_tools(self, code_graph, fts_index=None, query_service_mock=None):
        """Build progressive tools with test graph and FTS index."""
        # We need to force progressive tools on
        with patch.dict(os.environ, {'DEEPWIKI_PROGRESSIVE_TOOLS': '1'}):
            from plugin_implementation.deep_research.research_tools import create_codebase_tools
            tools = create_codebase_tools(
                retriever_stack=MagicMock(),
                graph_manager=MagicMock(),
                code_graph=code_graph,
                repo_analysis=None,
                event_callback=lambda x: None,
                graph_text_index=fts_index,
            )
        # Find get_relationships tool
        for t in tools:
            if t.name in ('get_relationships_tool', 'get_relationships'):
                return t
        self.fail("get_relationships tool not found in progressive tools")

    def test_orphan_class_with_fts_hits(self):
        """Orphan class triggers FTS fallback and returns string-ref results."""
        G = _make_orphan_graph('Event', 'class', ['on_startup', 'on_shutdown'])

        fts = _FakeFTSIndex({
            'on_startup': [
                _FakeFTSDoc('AppBoot', 'AppBootstrap', 'class', 'src/boot.py'),
            ],
            'on_shutdown': [
                _FakeFTSDoc('Cleanup', 'CleanupHandler', 'function', 'src/cleanup.py'),
            ],
        })

        # Mock the query_service to resolve the symbol but return no rels
        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = ('Event', [])
            mock_qs.resolve_symbol.return_value = 'Event'
            mock_qs.search.return_value = []

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'Event'})

        self.assertIn('no direct graph relationships', result)
        self.assertIn('on_startup', result)
        self.assertIn('AppBootstrap', result)
        self.assertIn('on_shutdown', result)
        self.assertIn('CleanupHandler', result)
        self.assertIn('string', result.lower())

    def test_orphan_struct_with_fts_hits(self):
        """Orphan struct also triggers FTS fallback."""
        G = _make_orphan_graph('Config', 'struct', ['validate', 'apply'])

        fts = _FakeFTSIndex({
            'validate': [
                _FakeFTSDoc('Loader', 'ConfigLoader', 'class', 'src/loader.py'),
            ],
        })

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = ('Config', [])

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'Config'})

        self.assertIn('no direct graph relationships', result)
        self.assertIn('validate', result)
        self.assertIn('ConfigLoader', result)

    def test_orphan_trait_with_fts_hits(self):
        """Orphan trait also triggers FTS fallback."""
        G = _make_orphan_graph('Serializable', 'trait', ['serialize', 'deserialize'])

        fts = _FakeFTSIndex({
            'serialize': [
                _FakeFTSDoc('JsonWriter', 'JsonWriter', 'class', 'src/json.rs'),
            ],
        })

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = ('Serializable', [])

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'Serializable'})

        self.assertIn('no direct graph relationships', result)
        self.assertIn('serialize', result)
        self.assertIn('JsonWriter', result)

    def test_orphan_enum_with_fts_hits(self):
        """Orphan enum also triggers FTS fallback."""
        G = _make_orphan_graph('Status', 'enum', ['active', 'inactive'])

        fts = _FakeFTSIndex({
            'active': [
                _FakeFTSDoc('Filter', 'StatusFilter', 'function', 'src/filter.py'),
            ],
        })

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = ('Status', [])

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'Status'})

        self.assertIn('no direct graph relationships', result)
        self.assertIn('active', result)
        self.assertIn('StatusFilter', result)

    def test_orphan_interface_with_fts_hits(self):
        """Orphan interface also triggers FTS fallback."""
        G = _make_orphan_graph('Handler', 'interface', ['handle', 'process'])

        fts = _FakeFTSIndex({
            'handle': [
                _FakeFTSDoc('Router', 'RequestRouter', 'class', 'src/router.py'),
            ],
        })

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = ('Handler', [])

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'Handler'})

        self.assertIn('no direct graph relationships', result)
        self.assertIn('handle', result)
        self.assertIn('RequestRouter', result)

    def test_orphan_class_no_fts_hits_returns_no_rels(self):
        """No FTS hits means we get the standard 'no relationships' message."""
        G = _make_orphan_graph('Lonely', 'class', ['do_nothing'])

        fts = _FakeFTSIndex({})  # No results for anything

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = ('Lonely', [])

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'Lonely'})

        self.assertIn('no relationships', result)
        self.assertNotIn('referenced as strings', result)

    def test_non_container_no_fallback(self):
        """A standalone function should NOT trigger orphan FTS fallback."""
        G = nx.MultiDiGraph()
        G.add_node('my_func', **_function('my_func'))

        fts = _FakeFTSIndex({
            'my_func': [_FakeFTSDoc('Caller', 'Caller', 'class', 'src/caller.py')],
        })

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = ('my_func', [])

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'my_func'})

        # Should get standard no-rels message, NOT orphan FTS output
        self.assertIn('no relationships', result)
        self.assertNotIn('referenced as strings', result)

    def test_no_fts_index_skips_fallback(self):
        """When no FTS5 index is available, orphan fallback is skipped."""
        G = _make_orphan_graph('Event', 'class', ['on_startup'])

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = ('Event', [])

            tool = self._build_tools(G, fts_index=None)
            result = tool.invoke({'symbol_name': 'Event'})

        self.assertIn('no relationships', result)

    def test_dunder_methods_skipped(self):
        """Dunder methods (__init__, __str__) should be skipped in FTS search."""
        G = nx.MultiDiGraph()
        G.add_node('Widget', **_container('Widget', 'class'))
        # Only dunder methods
        for m in ['__init__', '__str__', '__repr__']:
            G.add_node(f'Widget::{m}', **_method(m))
            _add_defines(G, 'Widget', f'Widget::{m}')

        fts = _FakeFTSIndex({})

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = ('Widget', [])

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'Widget'})

        # Should get standard no-rels, not orphan output (no non-dunder methods)
        self.assertIn('no relationships', result)
        self.assertNotIn('referenced as strings', result)

    def test_self_references_excluded(self):
        """FTS hits pointing back to the orphan node itself are excluded."""
        G = _make_orphan_graph('Event', 'class', ['fire'])

        fts = _FakeFTSIndex({
            'fire': [
                _FakeFTSDoc('Event', 'Event', 'class', 'src/event.py'),  # Self!
                _FakeFTSDoc('Dispatcher', 'Dispatcher', 'class', 'src/dispatch.py'),
            ],
        })

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = ('Event', [])

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'Event'})

        self.assertIn('Dispatcher', result)
        # "Event" appears as the subject but should not be listed as a referencing symbol
        # Count occurrences of "referenced in `Event`" — should be 0
        self.assertNotIn("referenced in `Event`", result)

    def test_result_contains_get_code_tip(self):
        """Orphan FTS result includes a tip to use get_code."""
        G = _make_orphan_graph('Event', 'class', ['on_startup'])

        fts = _FakeFTSIndex({
            'on_startup': [
                _FakeFTSDoc('Boot', 'AppBoot', 'class', 'src/boot.py'),
            ],
        })

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = ('Event', [])

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'Event'})

        self.assertIn('get_code', result)

    def test_symbol_with_normal_rels_no_fallback(self):
        """When symbol has normal relationships, orphan FTS is NOT triggered."""
        G = nx.MultiDiGraph()
        G.add_node('AuthService', **_container('AuthService', 'class'))
        G.add_node('BaseService', **_container('BaseService', 'class'))
        G.add_edge('AuthService', 'BaseService', relationship_type='inheritance')

        fts = _FakeFTSIndex({})

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = (
                'AuthService',
                [FakeRel('AuthService', 'BaseService', 'inheritance')],
            )

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'AuthService'})

        # Should show normal relationships, NOT orphan output
        self.assertIn('BaseService', result)
        self.assertIn('inheritance', result)
        self.assertNotIn('no direct graph relationships', result)


# ===================================================================
# 3. Wiki graph orphan type coverage (regression)
# ===================================================================

class TestWikiGraphOrphanTypes(unittest.TestCase):
    """Verify wiki_graph_optimized also covers all container types."""

    def test_orphan_types_include_enum_and_trait(self):
        """The orphan detection in wiki_graph_optimized should include enum and trait."""
        # Read the source to verify the types
        import inspect
        from plugin_implementation.agents import wiki_graph_optimized
        source = inspect.getsource(wiki_graph_optimized)
        # The orphan check line should include all container types
        self.assertIn("'enum'", source)
        self.assertIn("'trait'", source)
        self.assertIn("'class'", source)
        self.assertIn("'interface'", source)
        self.assertIn("'struct'", source)


# ===================================================================
# 4. Defines-only containers → FTS fallback also triggers
# ===================================================================

class TestDefinesOnlyFallback(unittest.TestCase):
    """When all rels are structural 'defines' edges, the container should
    also get orphan FTS treatment (appended after the defines lines)."""

    def _build_tools(self, code_graph, fts_index=None):
        with patch.dict(os.environ, {'DEEPWIKI_PROGRESSIVE_TOOLS': '1'}):
            from plugin_implementation.deep_research.research_tools import create_codebase_tools
            tools = create_codebase_tools(
                retriever_stack=MagicMock(),
                graph_manager=MagicMock(),
                code_graph=code_graph,
                repo_analysis=None,
                event_callback=lambda x: None,
                graph_text_index=fts_index,
            )
        for t in tools:
            if t.name in ('get_relationships_tool', 'get_relationships'):
                return t
        self.fail("get_relationships tool not found")

    def test_defines_only_triggers_fts_fallback(self):
        """Class with only 'defines' rels should also get orphan FTS hits appended."""
        G = _make_orphan_graph('Event', 'class', ['configuration_created'])

        fts = _FakeFTSIndex({
            'configuration_created': [
                _FakeFTSDoc('Utils', 'create_configuration', 'function', 'utils.py'),
            ],
        })

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            # Return defines rels (NOT empty — this is the key difference)
            mock_qs.resolve_and_traverse.return_value = (
                'Event',
                [FakeRel('Event', 'configuration_created', 'defines')],
            )

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'Event'})

        # Should contain BOTH the defines relationship AND the FTS hits
        self.assertIn('defines', result)
        self.assertIn('configuration_created', result)
        # The orphan FTS output should be appended
        self.assertIn('referenced as strings', result.lower())
        self.assertIn('create_configuration', result)

    def test_defines_only_no_fts_hits_shows_defines(self):
        """Defines-only with no FTS hits still shows the defines relationships."""
        G = _make_orphan_graph('Widget', 'class', ['draw'])

        fts = _FakeFTSIndex({})  # No results

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = (
                'Widget',
                [FakeRel('Widget', 'draw', 'defines')],
            )

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'Widget'})

        # Shows defines rels normally, no orphan section
        self.assertIn('defines', result)
        self.assertIn('draw', result)
        self.assertNotIn('referenced as strings', result.lower())

    def test_mixed_rels_no_orphan_fallback(self):
        """When rels include non-defines types, orphan FTS should NOT trigger."""
        G = _make_orphan_graph('Service', 'class', ['handle'])
        G.add_node('BaseService', **_container('BaseService', 'class'))
        G.add_edge('Service', 'BaseService', relationship_type='inheritance')

        fts = _FakeFTSIndex({
            'handle': [_FakeFTSDoc('Router', 'Router', 'class', 'r.py')],
        })

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = (
                'Service',
                [
                    FakeRel('Service', 'handle', 'defines'),
                    FakeRel('Service', 'BaseService', 'inheritance'),
                ],
            )

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'Service'})

        # Normal rels shown, no orphan FTS
        self.assertIn('inheritance', result)
        self.assertNotIn('referenced as strings', result.lower())

    def test_defines_only_non_container_no_fallback(self):
        """Non-container types with defines-only rels should NOT trigger FTS."""
        G = nx.MultiDiGraph()
        G.add_node('my_module', symbol_name='my_module', name='my_module',
                    symbol_type='module_doc', rel_path='mod.py')
        G.add_node('helper', symbol_name='helper', name='helper',
                    symbol_type='function', rel_path='mod.py')
        _add_defines(G, 'my_module', 'helper')

        fts = _FakeFTSIndex({
            'helper': [_FakeFTSDoc('X', 'X', 'class', 'x.py')],
        })

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_and_traverse.return_value = (
                'my_module',
                [FakeRel('my_module', 'helper', 'defines')],
            )

            tool = self._build_tools(G, fts)
            result = tool.invoke({'symbol_name': 'my_module'})

        # Shows defines but no orphan FTS (module_doc is not a container type)
        self.assertIn('defines', result)
        self.assertNotIn('referenced as strings', result.lower())


# ===================================================================
# 5. get_code empty content handling
# ===================================================================

class TestGetCodeEmptyContent(unittest.TestCase):
    """Test get_code improvements: FTS fallback, source_text, informative msg."""

    def _build_get_code(self, code_graph, fts_index=None):
        with patch.dict(os.environ, {'DEEPWIKI_PROGRESSIVE_TOOLS': '1'}):
            from plugin_implementation.deep_research.research_tools import create_codebase_tools
            tools = create_codebase_tools(
                retriever_stack=MagicMock(),
                graph_manager=MagicMock(),
                code_graph=code_graph,
                repo_analysis=None,
                event_callback=lambda x: None,
                graph_text_index=fts_index,
            )
        for t in tools:
            if t.name == 'get_code':
                return t
        self.fail("get_code tool not found")

    def test_empty_content_returns_informative_message(self):
        """When content is empty and no FTS, return a helpful message instead of empty block."""
        G = nx.MultiDiGraph()
        G.add_node('Event', symbol_name='Event', name='Event',
                    symbol_type='class', rel_path='events/configuration_created.py',
                    content='', docstring='', start_line='', end_line='')

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_symbol.return_value = 'Event'

            tool = self._build_get_code(G, fts_index=None)
            result = tool.invoke({'symbol_name': 'Event'})

        self.assertIn('Source code not available', result)
        self.assertIn('Event', result)
        self.assertIn('class', result)
        self.assertNotIn('```\n\n```', result)  # No empty code block

    def test_fts_fallback_retrieves_content(self):
        """When graph node has empty content, FTS index content is used."""
        G = nx.MultiDiGraph()
        G.add_node('Event', symbol_name='Event', name='Event',
                    symbol_type='class', rel_path='events/event.py',
                    content='', docstring='', start_line=1, end_line=10)

        class FakeFTSWithLookup:
            is_open = True
            def get_by_node_id(self, node_id):
                if node_id == 'Event':
                    return {
                        'content': 'class Event:\n    def fire(self):\n        pass',
                        'docstring': '',
                    }
                return None
            def search(self, *a, **kw):
                return []

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_symbol.return_value = 'Event'

            tool = self._build_get_code(G, fts_index=FakeFTSWithLookup())
            result = tool.invoke({'symbol_name': 'Event'})

        self.assertIn('class Event:', result)
        self.assertIn('def fire(self):', result)
        self.assertIn('```', result)  # Should have a proper code block

    def test_symbol_source_text_preferred(self):
        """symbol.source_text should be preferred over data.get('content')."""
        G = nx.MultiDiGraph()

        class FakeSymbol:
            source_text = 'class Event:\n    """From symbol.source_text."""'

        G.add_node('Event', symbol_name='Event', name='Event',
                    symbol_type='class', rel_path='events/event.py',
                    content='', symbol=FakeSymbol(),
                    start_line=1, end_line=5)

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_symbol.return_value = 'Event'

            tool = self._build_get_code(G, fts_index=None)
            result = tool.invoke({'symbol_name': 'Event'})

        self.assertIn('From symbol.source_text', result)
        self.assertIn('```', result)

    def test_source_text_attr_used(self):
        """data.get('source_text') should be tried before docstring."""
        G = nx.MultiDiGraph()
        G.add_node('helper', symbol_name='helper', name='helper',
                    symbol_type='function', rel_path='utils.py',
                    content='', source_text='def helper():\n    return 42',
                    start_line=1, end_line=2)

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_symbol.return_value = 'helper'

            tool = self._build_get_code(G, fts_index=None)
            result = tool.invoke({'symbol_name': 'helper'})

        self.assertIn('def helper():', result)
        self.assertIn('return 42', result)

    def test_existing_content_unchanged(self):
        """When content already exists, behavior is unchanged (no FTS lookup)."""
        G = nx.MultiDiGraph()
        G.add_node('Event', symbol_name='Event', name='Event',
                    symbol_type='class', rel_path='events/event.py',
                    content='class Event:\n    pass',
                    start_line=1, end_line=2)

        class FakeFTSNeverCalled:
            is_open = True
            def get_by_node_id(self, node_id):
                raise AssertionError("FTS should not be called when content exists")
            def search(self, *a, **kw):
                return []

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_symbol.return_value = 'Event'

            tool = self._build_get_code(G, fts_index=FakeFTSNeverCalled())
            result = tool.invoke({'symbol_name': 'Event'})

        self.assertIn('class Event:', result)
        self.assertIn('```', result)

    def test_fts_fallback_empty_still_shows_message(self):
        """FTS fallback with empty content still returns informative message."""
        G = nx.MultiDiGraph()
        G.add_node('Ghost', symbol_name='Ghost', name='Ghost',
                    symbol_type='class', rel_path='ghost.py',
                    content='', docstring='')

        class FakeFTSEmpty:
            is_open = True
            def get_by_node_id(self, node_id):
                return {'content': '', 'docstring': ''}
            def search(self, *a, **kw):
                return []

        with patch('plugin_implementation.deep_research.research_tools.GraphQueryService') as MockQS:
            mock_qs = MockQS.return_value
            mock_qs.resolve_symbol.return_value = 'Ghost'

            tool = self._build_get_code(G, fts_index=FakeFTSEmpty())
            result = tool.invoke({'symbol_name': 'Ghost'})

        self.assertIn('Source code not available', result)
        self.assertNotIn('```\n\n```', result)


if __name__ == '__main__':
    unittest.main()
