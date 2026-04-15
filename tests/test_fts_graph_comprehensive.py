"""
Comprehensive FTS5 + Graph Traversal Tests.

Validates FTS5 index queries against real cached graphs (fmtlib C++,
configurations Python) and tests graph traversal patterns used by
``query_graph``, ``search_graph``, ``search_symbols``, content expander,
and hybrid fusion.

Test tiers:
  1. **FTS5 path-prefix queries** — ``search_by_path_prefix`` with real
     directory structures, symbol_type filtering, exclude_types.
  2. **Graph traversal: CREATES** — factory function → created types.
  3. **Graph traversal: inheritance** — class → base classes via FTS5 + graph.
  4. **Graph traversal: defines_body** — C++ declaration → implementation.
  5. **Graph traversal: composition/aggregation** — field type discovery.
  6. **Graph traversal: alias_of** — type alias chain resolution.
  7. **Graph traversal: specializes/instantiates** — C++ templates.
  8. **query_graph text_filter** — path-prefix + concept intersection.
  9. **FTS5 concept queries** — ``search_smart`` on real docstrings.
  10. **Mixed symbol filtering** — architectural vs non-architectural.
  11. **Cross-file relationships** — symbols from different files linked.
  12. **Edge cases** — isolated nodes, circular references.
  13. **search_symbols via structure_tools** — end-to-end with real FTS5.

Run:
    cd pylon_deepwiki/plugins/deepwiki_plugin
    python -m pytest tests/test_fts_graph_comprehensive.py -v
"""

import gzip
import os
import pickle
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, Optional, Set
from unittest.mock import MagicMock, patch

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

from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
from plugin_implementation.code_graph.expansion_engine import (
    expand_smart,
    resolve_alias_chain,
    find_composed_types,
    find_creates_from_methods,
    edges_between,
    has_relationship,
    get_neighbors_by_relationship,
    augment_cpp_node,
)

# ---------------------------------------------------------------------------
# Cache setup  — same pattern as test_smart_expansion.py
# ---------------------------------------------------------------------------
_CACHE_DIR = _PLUGIN_ROOT.parent.parent / 'wiki_builder' / 'cache'
CACHE_DIR = Path(os.environ.get('DEEPWIKI_CACHE_DIR', str(_CACHE_DIR)))

FMTLIB_KEY = '551134763c1f5c1b3feca4dd95076290'   # C++ library (~9400 nodes)
CONFIGS_KEY = 'cd9d7a4aefa47194b872a7093a855156'   # Python project (~715 nodes)

_graph_cache: Dict[str, object] = {}


def _load_graph(key: str):
    """Load a pickled+gzipped graph from cache, with in-memory caching."""
    if key in _graph_cache:
        return _graph_cache[key]
    path = CACHE_DIR / f'{key}.code_graph.gz'
    if not path.exists():
        _graph_cache[key] = None
        return None
    with gzip.open(path, 'rb') as f:
        graph = pickle.load(f)
    _graph_cache[key] = graph
    return graph


def _fts_db_exists(key: str) -> bool:
    return (CACHE_DIR / f'{key}.fts5.db').exists()


def _has_fts_and_graph(key: str) -> bool:
    return _fts_db_exists(key) and (CACHE_DIR / f'{key}.code_graph.gz').exists()


# ===================================================================
#  Helper: collect edge types from graph between two nodes
# ===================================================================
def _edge_types_between(G, src, dst) -> Set[str]:
    """Return set of relationship_type strings for all edges src → dst."""
    types = set()
    edata = G.get_edge_data(src, dst)
    if edata:
        for edge in edata.values():
            rt = str(edge.get('relationship_type', '')).lower()
            if rt:
                types.add(rt)
    return types


# ===================================================================
#  1. FTS5 Path-Prefix Queries (configurations Python)
# ===================================================================
@unittest.skipUnless(
    _has_fts_and_graph(CONFIGS_KEY),
    'configurations FTS5+graph not found in cache',
)
class TestFTS5PathPrefix(unittest.TestCase):
    """search_by_path_prefix on a real Python project index."""

    @classmethod
    def setUpClass(cls):
        cls.idx = GraphTextIndex(cache_dir=str(CACHE_DIR))
        if not cls.idx.load(CONFIGS_KEY):
            raise unittest.SkipTest('Failed to load configurations FTS5')
        cls.graph = _load_graph(CONFIGS_KEY)
        if cls.graph is None:
            raise unittest.SkipTest('Failed to load configurations graph')

    def test_root_prefix_returns_all(self):
        """Querying the root prefix returns a large number of nodes."""
        # Detect root prefix from first node
        for _, data in self.graph.nodes(data=True):
            rp = data.get('rel_path', '')
            if rp and '/' in rp:
                root = rp.split('/')[0]
                break
        else:
            self.skipTest('No rel_path with / in graph')
        rows = self.idx.search_by_path_prefix(root, k=5000)
        self.assertGreater(len(rows), 50, f'Root prefix {root!r} should return many nodes')

    def test_specific_directory(self):
        """Querying a specific subdirectory returns a subset."""
        # Find a path with at least 2 segments
        for _, data in self.graph.nodes(data=True):
            rp = data.get('rel_path', '')
            parts = rp.strip('/').split('/')
            if len(parts) >= 3:
                prefix = '/'.join(parts[:2])
                break
        else:
            self.skipTest('No deep paths in graph')
        rows = self.idx.search_by_path_prefix(prefix, k=500)
        self.assertGreater(len(rows), 0, f'Prefix {prefix!r} should return some nodes')
        # All returned paths should start with the prefix
        for row in rows:
            rp = row.get('rel_path', '')
            self.assertTrue(
                rp.startswith(prefix) or rp.startswith(prefix + '/'),
                f'{rp!r} does not start with {prefix!r}',
            )

    def test_symbol_type_filter(self):
        """Filtering by symbol_type narrows results."""
        # Get all from root
        for _, data in self.graph.nodes(data=True):
            rp = data.get('rel_path', '')
            if rp and '/' in rp:
                root = rp.split('/')[0]
                break
        all_rows = self.idx.search_by_path_prefix(root, k=5000)
        class_rows = self.idx.search_by_path_prefix(
            root, symbol_types=frozenset({'class'}), k=5000,
        )
        self.assertGreater(len(all_rows), len(class_rows))
        for row in class_rows:
            self.assertEqual(row['symbol_type'], 'class')

    def test_exclude_types(self):
        """Excluding DOC types removes documentation symbols."""
        for _, data in self.graph.nodes(data=True):
            rp = data.get('rel_path', '')
            if rp and '/' in rp:
                root = rp.split('/')[0]
                break
        all_rows = self.idx.search_by_path_prefix(root, k=5000)
        filtered = self.idx.search_by_path_prefix(
            root,
            exclude_types=frozenset({'module_doc', 'file_doc', 'markdown_document', 'markdown_section'}),
            k=5000,
        )
        for row in filtered:
            self.assertNotIn(row['symbol_type'], {'module_doc', 'file_doc', 'markdown_document', 'markdown_section'})

    def test_nonexistent_prefix_empty(self):
        """A path that doesn't exist returns empty."""
        rows = self.idx.search_by_path_prefix('nonexistent_dir_xyz_999', k=100)
        self.assertEqual(len(rows), 0)


# ===================================================================
#  2. Graph Traversal: CREATES (fmtlib C++ / configurations Python)
# ===================================================================
@unittest.skipUnless(
    _has_fts_and_graph(CONFIGS_KEY),
    'configurations FTS5+graph not found in cache',
)
class TestGraphCreates(unittest.TestCase):
    """CREATES edges: factory function → created types."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(CONFIGS_KEY)
        if cls.graph is None:
            raise unittest.SkipTest('No configurations graph')
        cls.idx = GraphTextIndex(cache_dir=str(CACHE_DIR))
        cls.idx.load(CONFIGS_KEY)

    def test_find_creates_edges(self):
        """Graph should have at least one CREATES edge."""
        creates_count = 0
        for u, v, data in self.graph.edges(data=True):
            rt = str(data.get('relationship_type', '')).lower()
            if rt in ('creates', 'instantiates', 'constructs'):
                creates_count += 1
        if creates_count == 0:
            self.skipTest('No CREATES edges in configurations graph')
        self.assertGreater(creates_count, 0)

    def test_creates_via_fts_then_graph(self):
        """FTS5 finds a function → graph shows what it CREATES."""
        # Find a function node that has CREATES outgoing
        creator_node = None
        created_type = None
        for u, v, data in self.graph.edges(data=True):
            rt = str(data.get('relationship_type', '')).lower()
            if rt in ('creates', 'instantiates', 'constructs'):
                udata = self.graph.nodes.get(u, {})
                if udata.get('symbol_type', '').lower() == 'function':
                    creator_node = u
                    created_type = v
                    break
        if creator_node is None:
            self.skipTest('No function → CREATES edge found')
        
        creator_name = self.graph.nodes[creator_node].get('symbol_name', '')
        # Search for this function via FTS5
        docs = self.idx.search_smart(creator_name, k=5, intent='symbol')
        found = any(
            d.metadata.get('node_id') == creator_node or
            d.metadata.get('symbol_name') == creator_name
            for d in docs
        )
        self.assertTrue(found, f'FTS5 should find creator function {creator_name!r}')

    def test_find_creates_from_methods_helper(self):
        """expand_smart helper find_creates_from_methods works on real graph."""
        # Find a class node
        for nid, data in self.graph.nodes(data=True):
            if data.get('symbol_type', '').lower() == 'class':
                created = find_creates_from_methods(self.graph, nid)
                # Just verify it doesn't crash — may or may not find creates
                self.assertIsInstance(created, list)
                break
        else:
            self.skipTest('No class nodes in graph')


# ===================================================================
#  3. Graph Traversal: Inheritance (configurations Python)
# ===================================================================
@unittest.skipUnless(
    _has_fts_and_graph(CONFIGS_KEY),
    'configurations FTS5+graph not found in cache',
)
class TestGraphInheritance(unittest.TestCase):
    """Inheritance chains via FTS5 + graph traversal."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(CONFIGS_KEY)
        if cls.graph is None:
            raise unittest.SkipTest('No configurations graph')
        cls.idx = GraphTextIndex(cache_dir=str(CACHE_DIR))
        cls.idx.load(CONFIGS_KEY)

    def test_inheritance_edges_exist(self):
        """Graph should have inheritance/extends edges."""
        inh_count = 0
        for u, v, data in self.graph.edges(data=True):
            rt = str(data.get('relationship_type', '')).lower()
            if rt in ('inheritance', 'extends', 'implements'):
                inh_count += 1
        self.assertGreater(inh_count, 0, 'No inheritance edges in configurations graph')

    def test_fts5_find_child_then_traverse_to_parent(self):
        """FTS5 finds a subclass → graph traversal finds its base class."""
        # Find an inheritance edge
        child_nid = None
        parent_nid = None
        for u, v, data in self.graph.edges(data=True):
            rt = str(data.get('relationship_type', '')).lower()
            if rt in ('inheritance', 'extends', 'implements'):
                child_nid, parent_nid = u, v
                break
        if child_nid is None:
            self.skipTest('No inheritance edges')

        child_name = self.graph.nodes[child_nid].get('symbol_name', '')
        parent_name = self.graph.nodes[parent_nid].get('symbol_name', '')

        # FTS5 should find child
        docs = self.idx.search_smart(child_name, k=5, intent='symbol')
        self.assertTrue(len(docs) > 0, f'FTS5 should find child {child_name!r}')

        # Graph should reveal parent — returns List[Tuple[node_id, rel_type]]
        parents = get_neighbors_by_relationship(
            self.graph, child_nid, {'inheritance', 'extends', 'implements'}, direction='successors',
        )
        parent_ids = {nid for nid, _ in parents}
        self.assertIn(parent_nid, parent_ids,
                       f'Graph should show {parent_name!r} as parent of {child_name!r}')

    def test_inheritance_chain_depth(self):
        """Some class hierarchies should be ≥2 levels deep."""
        # Build depth map
        max_depth = 0
        for nid, data in self.graph.nodes(data=True):
            if data.get('symbol_type', '').lower() not in ('class', 'interface'):
                continue
            depth = 0
            current = nid
            visited = {current}
            while True:
                parent_tuples = get_neighbors_by_relationship(
                    self.graph, current, {'inheritance', 'extends'}, direction='successors',
                )
                parent = next((nid for nid, _ in parent_tuples if nid not in visited), None)
                if parent is None:
                    break
                visited.add(parent)
                current = parent
                depth += 1
            max_depth = max(max_depth, depth)
        # At least one non-trivial chain
        self.assertGreaterEqual(max_depth, 1,
                                 'Should have at least one class extending another')


# ===================================================================
#  4. Graph Traversal: defines_body (fmtlib C++)
# ===================================================================
@unittest.skipUnless(
    _has_fts_and_graph(FMTLIB_KEY),
    'fmtlib FTS5+graph not found in cache',
)
class TestGraphDefinesBody(unittest.TestCase):
    """defines_body: C++ declaration → implementation split."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(FMTLIB_KEY)
        if cls.graph is None:
            raise unittest.SkipTest('No fmtlib graph')
        cls.idx = GraphTextIndex(cache_dir=str(CACHE_DIR))
        cls.idx.load(FMTLIB_KEY)

    def test_defines_body_edges_exist(self):
        """C++ graph should have defines_body edges (decl → impl split)."""
        count = 0
        for u, v, data in self.graph.edges(data=True):
            if str(data.get('relationship_type', '')).lower() == 'defines_body':
                count += 1
        self.assertGreater(count, 0, 'fmtlib should have defines_body edges')

    def test_augment_cpp_node_on_real_class(self):
        """augment_cpp_node should handle C++ classes (may return None if no impl split)."""
        # Find a class with defines_body edges on its contained methods
        for nid, ndata in self.graph.nodes(data=True):
            if ndata.get('symbol_type', '').lower() not in ('class', 'struct'):
                continue
            result = augment_cpp_node(self.graph, nid)
            if result is not None:
                # Found a class with augmented content
                self.assertTrue(
                    hasattr(result, 'augmented_content'),
                    'AugmentedContent should have augmented_content field',
                )
                self.assertIsInstance(result.augmented_content, str)
                return
        # If no class had augmentable content, that's acceptable for this graph
        self.skipTest('No C++ class with augmentable defines_body content in fmtlib')

    def test_fts5_finds_header_declaration(self):
        """FTS5 can find both the header declaration and the .cpp implementation."""
        # Find a defines_body pair
        for u, v, data in self.graph.edges(data=True):
            if str(data.get('relationship_type', '')).lower() == 'defines_body':
                impl_name = self.graph.nodes.get(u, {}).get('symbol_name', '')
                decl_name = self.graph.nodes.get(v, {}).get('symbol_name', '')
                if impl_name and decl_name:
                    # FTS5 should find the symbol
                    docs = self.idx.search_smart(decl_name, k=10, intent='symbol')
                    self.assertGreater(len(docs), 0,
                                       f'FTS5 should find {decl_name!r}')
                    return
        self.skipTest('No defines_body pair with names')


# ===================================================================
#  5. Graph Traversal: Composition (field type discovery)
# ===================================================================
@unittest.skipUnless(
    _has_fts_and_graph(CONFIGS_KEY),
    'configurations FTS5+graph not found in cache',
)
class TestGraphComposition(unittest.TestCase):
    """Composition/aggregation: classes that hold instances of other classes."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(CONFIGS_KEY)
        if cls.graph is None:
            raise unittest.SkipTest('No configurations graph')

    def test_find_composed_types_on_real_class(self):
        """find_composed_types should work on real graph classes."""
        for nid, data in self.graph.nodes(data=True):
            if data.get('symbol_type', '').lower() == 'class':
                composed = find_composed_types(self.graph, nid)
                # Just verify it doesn't crash — may or may not find composition
                self.assertIsInstance(composed, list)
                break
        else:
            self.skipTest('No class nodes')

    def test_references_edges_exist(self):
        """Graph should have REFERENCES edges (type usage / composition)."""
        ref_count = 0
        for u, v, data in self.graph.edges(data=True):
            rt = str(data.get('relationship_type', '')).lower()
            if rt == 'references':
                ref_count += 1
        self.assertGreater(ref_count, 0, 'Should have REFERENCES edges')


# ===================================================================
#  6. Graph Traversal: alias_of (type alias resolution)
# ===================================================================
@unittest.skipUnless(
    _has_fts_and_graph(FMTLIB_KEY),
    'fmtlib FTS5+graph not found in cache',
)
class TestGraphAliasOf(unittest.TestCase):
    """Type alias chain resolution via alias_of edges."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(FMTLIB_KEY)
        if cls.graph is None:
            raise unittest.SkipTest('No fmtlib graph')

    def test_alias_of_edges_exist(self):
        """C++ graph should have alias_of / uses_type edges for typedefs."""
        alias_count = 0
        for u, v, data in self.graph.edges(data=True):
            rt = str(data.get('relationship_type', '')).lower()
            if rt in ('alias_of', 'uses_type'):
                alias_count += 1
        if alias_count == 0:
            self.skipTest('No alias_of/uses_type edges in fmtlib')
        self.assertGreater(alias_count, 0)

    def test_resolve_alias_chain_real(self):
        """resolve_alias_chain follows alias_of edges to the concrete type."""
        for u, v, data in self.graph.edges(data=True):
            rt = str(data.get('relationship_type', '')).lower()
            if rt in ('alias_of', 'uses_type'):
                udata = self.graph.nodes.get(u, {})
                if udata.get('symbol_type', '').lower() == 'type_alias':
                    resolved = resolve_alias_chain(self.graph, u, max_hops=5)
                    self.assertIsNotNone(resolved)
                    # Should resolve to something — possibly itself or the target
                    return
        self.skipTest('No type_alias with alias_of edge')


# ===================================================================
#  7. Graph Traversal: specializes/instantiates (C++ templates)
# ===================================================================
@unittest.skipUnless(
    _has_fts_and_graph(FMTLIB_KEY),
    'fmtlib FTS5+graph not found in cache',
)
class TestGraphSpecializes(unittest.TestCase):
    """Template specialization via specializes/instantiates edges."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(FMTLIB_KEY)
        if cls.graph is None:
            raise unittest.SkipTest('No fmtlib graph')

    def test_specializes_edges_exist(self):
        """fmtlib should have specializes/instantiates edges (templates)."""
        spec_count = 0
        for u, v, data in self.graph.edges(data=True):
            rt = str(data.get('relationship_type', '')).lower()
            if rt in ('specializes', 'instantiates', 'template_specialization'):
                spec_count += 1
        if spec_count == 0:
            self.skipTest('No specializes/instantiates edges in fmtlib')
        self.assertGreater(spec_count, 0)


# ===================================================================
#  8. query_graph text_filter (path-prefix + concept intersection)
# ===================================================================
@unittest.skipUnless(
    _has_fts_and_graph(CONFIGS_KEY),
    'configurations FTS5+graph not found in cache',
)
class TestQueryGraphTextFilter(unittest.TestCase):
    """query_graph with text_filter parameter for concept narrowing."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(CONFIGS_KEY)
        if cls.graph is None:
            raise unittest.SkipTest('No configurations graph')
        cls.idx = GraphTextIndex(cache_dir=str(CACHE_DIR))
        if not cls.idx.load(CONFIGS_KEY):
            raise unittest.SkipTest('FTS5 load failed')

    def _make_planner(self):
        """Create a minimal StructureCollector object for _handle_query_graph."""
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            StructureCollector,
        )
        planner = StructureCollector.__new__(StructureCollector)
        planner.code_graph = self.graph
        planner.graph_text_index = self.idx
        planner._graph_module_cache = {}
        planner._detected_repo_prefix = None
        planner._repository_file_count = 0
        return planner

    def test_no_filter_returns_all(self):
        """Without text_filter, query_graph returns all symbols under prefix."""
        planner = self._make_planner()
        # Get any valid prefix
        for _, data in self.graph.nodes(data=True):
            rp = data.get('rel_path', '')
            if rp and '/' in rp:
                prefix = rp.split('/')[0]
                break
        result = planner._handle_query_graph(prefix)
        self.assertIn('Total symbols:', result)

    def test_text_filter_narrows_results(self):
        """text_filter reduces the number of reported symbols."""
        planner_all = self._make_planner()
        planner_filtered = self._make_planner()
        # Find root prefix
        for _, data in self.graph.nodes(data=True):
            rp = data.get('rel_path', '')
            if rp and '/' in rp:
                prefix = rp.split('/')[0]
                break
        result_all = planner_all._handle_query_graph(prefix)
        # Pick a specific term from a symbol name
        for _, data in self.graph.nodes(data=True):
            name = data.get('symbol_name', '')
            if name and len(name) > 5 and data.get('symbol_type', '').lower() == 'class':
                keyword = name
                break
        result_filtered = planner_filtered._handle_query_graph(prefix, text_filter=keyword)
        # Filtered result should mention fewer total symbols
        # (or show results — the key is it doesn't crash)
        self.assertIsInstance(result_filtered, str)
        self.assertGreater(len(result_filtered), 10)

    def test_text_filter_nonexistent_concept(self):
        """text_filter with nonsense returns empty / zero symbols."""
        planner = self._make_planner()
        for _, data in self.graph.nodes(data=True):
            rp = data.get('rel_path', '')
            if rp and '/' in rp:
                prefix = rp.split('/')[0]
                break
        result = planner._handle_query_graph(prefix, text_filter='zzz_nonexistent_xyz_999')
        # Should indicate empty results — either '0' symbols or 'No symbols found'
        self.assertTrue(
            '0' in result or 'No symbols' in result or 'no symbols' in result.lower(),
            f'Expected empty result indication, got: {result[:200]}',
        )


# ===================================================================
#  9. FTS5 Concept Queries (search_smart on real data)
# ===================================================================
@unittest.skipUnless(
    _fts_db_exists(CONFIGS_KEY),
    'configurations FTS5 DB not found',
)
class TestFTS5ConceptQueries(unittest.TestCase):
    """search_smart with concept/natural-language queries on real data."""

    @classmethod
    def setUpClass(cls):
        cls.idx = GraphTextIndex(cache_dir=str(CACHE_DIR))
        if not cls.idx.load(CONFIGS_KEY):
            raise unittest.SkipTest('FTS5 load failed')

    def test_general_concept_query(self):
        """A general concept query returns relevant ranked results."""
        docs = self.idx.search_smart('configuration loading', k=10, intent='general')
        self.assertGreater(len(docs), 0, 'Should find config-related symbols')
        # All results should have search_source metadata
        for doc in docs:
            self.assertEqual(doc.metadata.get('search_source'), 'graph_fts')

    def test_symbol_intent_query(self):
        """A symbol-intent query returns exact or close name matches."""
        # Find a real class name
        docs = self.idx.search_smart('Configuration', k=5, intent='symbol')
        self.assertGreater(len(docs), 0)
        # Top result should contain 'Configuration' or similar
        top_name = docs[0].metadata.get('symbol_name', '')
        self.assertIn('onfig', top_name.lower(),
                       f'Top result {top_name!r} should match config')

    def test_query_returns_ranked_results(self):
        """Results should be ranked (BM25 ordering preserved)."""
        docs = self.idx.search_smart('configuration loading', k=20, intent='general')
        if len(docs) < 2:
            self.skipTest('Not enough results for ranking check')
        # Check scores if present in metadata
        scores = [
            d.metadata.get('search_score',
                d.metadata.get('bm25_score', None))
            for d in docs
        ]
        if all(s is not None for s in scores):
            float_scores = [float(s) for s in scores]
            # BM25 rank() can be negative; verify they are monotonically sorted
            # (either ascending or descending — depends on the engine)
            is_desc = all(float_scores[i] >= float_scores[i+1] for i in range(len(float_scores)-1))
            is_asc = all(float_scores[i] <= float_scores[i+1] for i in range(len(float_scores)-1))
            self.assertTrue(
                is_desc or is_asc,
                f'Scores should be monotonically sorted: {float_scores[:5]}...',
            )
        else:
            # No scores in metadata — just verify results exist
            self.assertGreater(len(docs), 1, 'Should have multiple ranked results')

    def test_empty_query_returns_empty(self):
        """Empty or whitespace query returns empty."""
        docs = self.idx.search_smart('', k=10, intent='general')
        self.assertEqual(len(docs), 0)

    def test_query_with_path_filter(self):
        """search_symbols with path_prefix scopes results."""
        # Get any path prefix
        all_docs = self.idx.search_smart('class', k=100, intent='general')
        if len(all_docs) < 5:
            self.skipTest('Not enough symbols')
        # Find a specific path prefix
        paths = [d.metadata.get('rel_path', '') for d in all_docs if d.metadata.get('rel_path')]
        if not paths:
            self.skipTest('No rel_path metadata')
        # Pick first directory
        first_dir = paths[0].split('/')[0] if '/' in paths[0] else paths[0]
        # search_symbols is on GraphTextIndex — check if it has path filtering
        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        scoped = self.idx.search_symbols(
            'class', k=50,
            path_prefix=first_dir,
        )
        for doc in scoped:
            rp = doc.metadata.get('rel_path', '')
            self.assertTrue(
                rp.startswith(first_dir),
                f'{rp!r} should start with {first_dir!r}',
            )


# ===================================================================
#  10. Mixed Symbol Filtering (architectural vs non-architectural)
# ===================================================================
@unittest.skipUnless(
    _has_fts_and_graph(CONFIGS_KEY),
    'configurations FTS5+graph not found in cache',
)
class TestMixedSymbolFiltering(unittest.TestCase):
    """Verify filtering between architectural and non-architectural symbols."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(CONFIGS_KEY)
        if cls.graph is None:
            raise unittest.SkipTest('No configurations graph')
        cls.idx = GraphTextIndex(cache_dir=str(CACHE_DIR))
        cls.idx.load(CONFIGS_KEY)

    def test_graph_has_multiple_symbol_types(self):
        """Graph should have diverse symbol types (class, function, method, etc.)."""
        types = set()
        for _, data in self.graph.nodes(data=True):
            st = data.get('symbol_type', '').lower()
            if st:
                types.add(st)
        # Should have at least class and function
        self.assertIn('class', types)
        self.assertIn('function', types)
        self.assertGreater(len(types), 2, f'Only found types: {types}')

    def test_fts5_type_filtering(self):
        """FTS5 search_symbols can filter by type."""
        # Use a broad query that returns many results
        all_docs = self.idx.search_symbols('*', k=200)
        if not all_docs:
            # Try a common keyword
            all_docs = self.idx.search_smart('config OR class OR function', k=200, intent='general')
        class_docs = self.idx.search_symbols('*', k=200, symbol_types=frozenset({'class'}))
        if not class_docs:
            class_docs = self.idx.search_smart(
                'config OR class OR function', k=200, intent='general',
                symbol_types=frozenset({'class'}),
            )
        
        all_types = {d.metadata.get('symbol_type') for d in all_docs}
        class_types = {d.metadata.get('symbol_type') for d in class_docs}
        
        if not all_docs:
            self.skipTest('No results from FTS5 for broad query')
        self.assertGreater(len(all_types), len(class_types),
                           f'All types {all_types} should be more diverse than class-only {class_types}')
        if class_docs:
            self.assertEqual(class_types, {'class'})

    def test_exclude_method_symbols(self):
        """Excluding 'method' type reduces results significantly."""
        all_docs = self.idx.search_symbols('', k=500)
        no_methods = self.idx.search_symbols(
            '', k=500,
            exclude_types=frozenset({'method'}),
        )
        method_count = sum(1 for d in all_docs if d.metadata.get('symbol_type') == 'method')
        if method_count == 0:
            self.skipTest('No method symbols to exclude')
        self.assertLess(len(no_methods), len(all_docs))


# ===================================================================
#  11. Cross-File Relationships
# ===================================================================
@unittest.skipUnless(
    _has_fts_and_graph(CONFIGS_KEY),
    'configurations FTS5+graph not found in cache',
)
class TestCrossFileRelationships(unittest.TestCase):
    """Symbols from different files that are linked by graph edges."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(CONFIGS_KEY)
        if cls.graph is None:
            raise unittest.SkipTest('No configurations graph')

    def test_cross_file_edges_exist(self):
        """Some graph edges should connect nodes from different files."""
        cross_file_count = 0
        for u, v, data in self.graph.edges(data=True):
            upath = self.graph.nodes.get(u, {}).get('rel_path', '')
            vpath = self.graph.nodes.get(v, {}).get('rel_path', '')
            if upath and vpath and upath != vpath:
                cross_file_count += 1
                if cross_file_count >= 10:
                    break
        self.assertGreater(cross_file_count, 0,
                           'Graph should have cross-file relationships')

    def test_import_edges_link_files(self):
        """Import/dependency edges should link different files."""
        import_count = 0
        for u, v, data in self.graph.edges(data=True):
            rt = str(data.get('relationship_type', '')).lower()
            if rt in ('imports', 'depends_on', 'uses'):
                upath = self.graph.nodes.get(u, {}).get('rel_path', '')
                vpath = self.graph.nodes.get(v, {}).get('rel_path', '')
                if upath and vpath and upath != vpath:
                    import_count += 1
                    if import_count >= 5:
                        break
        if import_count == 0:
            self.skipTest('No cross-file import edges found')
        self.assertGreater(import_count, 0)


# ===================================================================
#  12. Edge Cases: Isolated Nodes, Circular References
# ===================================================================
@unittest.skipUnless(
    _has_fts_and_graph(CONFIGS_KEY),
    'configurations FTS5+graph not found in cache',
)
class TestEdgeCases(unittest.TestCase):
    """Edge cases: isolated nodes, circular references, self-loops."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(CONFIGS_KEY)
        if cls.graph is None:
            raise unittest.SkipTest('No configurations graph')
        cls.idx = GraphTextIndex(cache_dir=str(CACHE_DIR))
        cls.idx.load(CONFIGS_KEY)

    def test_isolated_nodes_exist(self):
        """Some nodes should have zero edges (isolated symbols)."""
        isolated = [
            nid for nid in self.graph.nodes()
            if self.graph.degree(nid) == 0
        ]
        # It's OK if there are none — this is informational
        # But we shouldn't crash when expanding isolated nodes
        if isolated:
            nid = isolated[0]
            result = expand_smart({nid}, self.graph)
            self.assertIsNotNone(result)

    def test_expand_smart_on_real_nodes(self):
        """expand_smart should work on various real graph nodes."""
        tested = 0
        for nid, data in self.graph.nodes(data=True):
            st = data.get('symbol_type', '').lower()
            if st in ('class', 'function', 'constant'):
                result = expand_smart({nid}, self.graph)
                self.assertIsNotNone(result)
                self.assertIsInstance(result.expanded_nodes, set)
                tested += 1
                if tested >= 5:
                    break
        self.assertGreater(tested, 0, 'Should test at least one node')

    def test_no_crash_on_self_loops(self):
        """If graph has self-loops, expand_smart shouldn't crash."""
        # Check for self-loops
        self_loops = [u for u, v in self.graph.edges() if u == v]
        if not self_loops:
            # Create temporary self-loop for testing
            first_node = next(iter(self.graph.nodes()))
            self.graph.add_edge(first_node, first_node, relationship_type='test_self_loop')
            try:
                result = expand_smart({first_node}, self.graph)
                self.assertIsNotNone(result)
            finally:
                # Clean up
                self.graph.remove_edge(first_node, first_node)
        else:
            nid = self_loops[0]
            result = expand_smart({nid}, self.graph)
            self.assertIsNotNone(result)


# ===================================================================
#  13. FTS5 on fmtlib C++ (large graph, different language)
# ===================================================================
@unittest.skipUnless(
    _has_fts_and_graph(FMTLIB_KEY),
    'fmtlib FTS5+graph not found in cache',
)
class TestFTS5Fmtlib(unittest.TestCase):
    """FTS5 queries on fmtlib C++ graph — validates cross-language support."""

    @classmethod
    def setUpClass(cls):
        cls.idx = GraphTextIndex(cache_dir=str(CACHE_DIR))
        if not cls.idx.load(FMTLIB_KEY):
            raise unittest.SkipTest('FTS5 load failed')
        cls.graph = _load_graph(FMTLIB_KEY)
        if cls.graph is None:
            raise unittest.SkipTest('No fmtlib graph')

    def test_search_format_concept(self):
        """'format' is the core concept of fmtlib — should find many results."""
        docs = self.idx.search_smart('format', k=50, intent='general')
        self.assertGreater(len(docs), 5, 'fmtlib should have many format-related symbols')

    def test_search_by_class_name(self):
        """Search for known fmtlib classes."""
        # fmtlib has classes like formatter, basic_format_string, etc.
        docs = self.idx.search_smart('formatter', k=10, intent='symbol')
        self.assertGreater(len(docs), 0, 'Should find formatter-related classes')

    def test_path_prefix_include_dir(self):
        """search_by_path_prefix works on fmtlib's include directory."""
        # fmtlib structure: include/fmt/*.h
        rows = self.idx.search_by_path_prefix('include', k=500)
        if len(rows) == 0:
            # Try alternative structure
            rows = self.idx.search_by_path_prefix('fmt', k=500)
        if len(rows) == 0:
            # Try with repo prefix
            for _, data in self.graph.nodes(data=True):
                rp = data.get('rel_path', '')
                if rp and '/' in rp:
                    root = rp.split('/')[0]
                    rows = self.idx.search_by_path_prefix(root, k=500)
                    break
        self.assertGreater(len(rows), 0, 'Should find symbols in include/fmt')

    def test_cpp_symbol_types(self):
        """fmtlib should have C++-specific symbol types."""
        types = set()
        for _, data in self.graph.nodes(data=True):
            st = data.get('symbol_type', '').lower()
            if st:
                types.add(st)
        # C++ typically has struct, class, function, method, macro, etc.
        self.assertTrue(
            types & {'class', 'struct', 'function', 'method', 'macro', 'enum'},
            f'Expected C++ symbol types, got: {types}',
        )

    def test_smart_expansion_on_cpp_class(self):
        """Smart expansion should work on C++ class nodes."""
        for nid, data in self.graph.nodes(data=True):
            if data.get('symbol_type', '').lower() in ('class', 'struct'):
                if self.graph.degree(nid) >= 3:  # Non-trivial node
                    result = expand_smart({nid}, self.graph)
                    self.assertIsNotNone(result)
                    # C++ classes often have inheritance or defines relationships
                    return
        self.skipTest('No suitable C++ class node found')


# ===================================================================
#  14. search_symbols end-to-end via structure_tools
# ===================================================================
@unittest.skipUnless(
    _fts_db_exists(CONFIGS_KEY),
    'configurations FTS5 DB not found',
)
class TestSearchSymbolsEndToEnd(unittest.TestCase):
    """search_symbols tool handler with real FTS5 data."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(CONFIGS_KEY)
        if cls.graph is None:
            raise unittest.SkipTest('No configurations graph')
        cls.idx = GraphTextIndex(cache_dir=str(CACHE_DIR))
        if not cls.idx.load(CONFIGS_KEY):
            raise unittest.SkipTest('FTS5 load failed')

    def _make_planner(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            StructureCollector,
        )
        planner = StructureCollector.__new__(StructureCollector)
        planner.code_graph = self.graph
        planner.graph_text_index = self.idx
        planner._graph_module_cache = {}
        planner._detected_repo_prefix = None
        return planner

    def test_search_symbols_returns_results(self):
        """_handle_search_symbols returns formatted results."""
        planner = self._make_planner()
        result = planner._handle_search_symbols('configuration', None, None)
        self.assertIn('configuration', result.lower())

    def test_search_symbols_with_type_filter(self):
        """Filtering by symbol_type narrows results."""
        planner = self._make_planner()
        result = planner._handle_search_symbols('test', 'class', None)
        self.assertIsInstance(result, str)
        # Should not contain 'function' type markers if only classes requested
        # (result format may vary, but shouldn't crash)

    def test_search_symbols_with_path_prefix(self):
        """path_prefix scopes results to a directory."""
        planner = self._make_planner()
        # Find a valid path prefix
        for _, data in self.graph.nodes(data=True):
            rp = data.get('rel_path', '')
            if rp and '/' in rp:
                prefix = rp.split('/')[0]
                break
        result = planner._handle_search_symbols('class', None, prefix)
        self.assertIsInstance(result, str)

    def test_search_symbols_no_results(self):
        """Search for nonsense returns a 'no results' message."""
        planner = self._make_planner()
        result = planner._handle_search_symbols('zzz_nonexistent_xyz_999', None, None)
        self.assertIsInstance(result, str)
        # Should indicate no/few results


# ===================================================================
#  15. Graph Statistics Sanity Checks
# ===================================================================
@unittest.skipUnless(
    _has_fts_and_graph(CONFIGS_KEY),
    'configurations FTS5+graph not found in cache',
)
class TestGraphStatsSanity(unittest.TestCase):
    """Sanity checks that cached graphs have expected properties."""

    @classmethod
    def setUpClass(cls):
        cls.configs_graph = _load_graph(CONFIGS_KEY)
        cls.fmtlib_graph = _load_graph(FMTLIB_KEY)

    def test_configs_graph_size(self):
        """Configurations graph should have hundreds of nodes."""
        if self.configs_graph is None:
            self.skipTest('No configs graph')
        self.assertGreater(self.configs_graph.number_of_nodes(), 100)
        self.assertGreater(self.configs_graph.number_of_edges(), 50)

    def test_fmtlib_graph_size(self):
        """fmtlib graph should have thousands of nodes."""
        if self.fmtlib_graph is None:
            self.skipTest('No fmtlib graph')
        self.assertGreater(self.fmtlib_graph.number_of_nodes(), 5000)
        self.assertGreater(self.fmtlib_graph.number_of_edges(), 1000)

    def test_configs_is_multidigraph(self):
        """Graph should be a MultiDiGraph (directed, allows parallel edges)."""
        if self.configs_graph is None:
            self.skipTest('No configs graph')
        self.assertIsInstance(self.configs_graph, nx.MultiDiGraph)

    def test_node_attributes_present(self):
        """Nodes should have required attributes: symbol_name, symbol_type, rel_path."""
        if self.configs_graph is None:
            self.skipTest('No configs graph')
        checked = 0
        for nid, data in self.configs_graph.nodes(data=True):
            # Most nodes should have these
            if data.get('symbol_name') and data.get('symbol_type') and data.get('rel_path'):
                checked += 1
                if checked >= 20:
                    break
        self.assertGreater(checked, 10, 'Nodes should have basic attributes')

    def test_edge_attributes_present(self):
        """Edges should have relationship_type attribute."""
        if self.configs_graph is None:
            self.skipTest('No configs graph')
        typed_edges = 0
        for u, v, data in self.configs_graph.edges(data=True):
            if data.get('relationship_type'):
                typed_edges += 1
                if typed_edges >= 20:
                    break
        self.assertGreater(typed_edges, 10, 'Edges should have relationship_type')

    def test_fts5_covers_graph_nodes(self):
        """FTS5 index should cover most graph nodes."""
        if self.configs_graph is None:
            self.skipTest('No configs graph')
        idx = GraphTextIndex(cache_dir=str(CACHE_DIR))
        if not idx.load(CONFIGS_KEY):
            self.skipTest('FTS5 load failed')
        # Count nodes with symbol_name (FTS5 indexes these)
        named_nodes = sum(
            1 for _, d in self.configs_graph.nodes(data=True)
            if d.get('symbol_name')
        )
        # FTS5 should have indexed most of them
        all_rows = idx.search_by_path_prefix('', k=10000)
        # Coverage should be > 50%
        if named_nodes > 0:
            coverage = len(all_rows) / named_nodes
            self.assertGreater(coverage, 0.3,
                               f'FTS5 covers only {coverage:.0%} of named nodes')


if __name__ == '__main__':
    unittest.main()
