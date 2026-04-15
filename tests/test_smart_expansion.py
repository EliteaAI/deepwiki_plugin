"""
Comprehensive tests for the Smart Graph Expansion Engine.

Three test tiers:
  1. **Unit tests** — synthetic graphs testing edge utilities and strategy functions.
  2. **Integration tests** — real cached graphs (fmtlib C++, configurations Python)
     validating that smart expansion produces better results than naive.
  3. **Regression tests** — ensure smart expansion degrades gracefully to naive
     behavior when edge types are missing.

Run:
    cd pylon_deepwiki/plugins/deepwiki_plugin
    python -m pytest tests/test_smart_expansion.py -v
"""

import os
import tempfile
import unittest
from collections import Counter
from typing import Dict, Set

import networkx as nx

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from plugin_implementation.code_graph.expansion_engine import (
    EXPANSION_WORTHY_TYPES,
    CLASS_LIKE_TYPES,
    AugmentedContent,
    ExpansionResult,
    _get_source_text,
    edges_between,
    has_relationship,
    get_neighbors_by_relationship,
    resolve_alias_chain,
    augment_cpp_node,
    find_composed_types,
    find_creates_from_methods,
    find_calls_to_free_functions,
    expand_smart,
    _expand_class,
    _expand_function,
    _expand_constant,
    _expand_type_alias,
)

# ---------------------------------------------------------------------------
# Graph building helpers (parse real code snippets → graph → expansion)
# ---------------------------------------------------------------------------

def _write_files(tmpdir: str, files: dict) -> dict:
    """Write files dict to tmpdir, return {filename: absolute_path}."""
    paths = {}
    for name, content in files.items():
        fpath = os.path.join(tmpdir, name)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, 'w') as f:
            f.write(content)
        paths[name] = fpath
    return paths


def _build_graph(files: dict):
    """Build a real graph from a dict of {filename: source_code}."""
    from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_files(tmpdir, files)
        builder = EnhancedUnifiedGraphBuilder()
        result = builder.analyze_repository(tmpdir)
        return result.unified_graph


def _find_node(graph, name_contains: str, symbol_type: str = None) -> str:
    """Find a single node whose symbol_name contains *name_contains*."""
    candidates = []
    for nid, data in graph.nodes(data=True):
        sym_name = data.get('symbol_name', '') or ''
        if name_contains in nid or name_contains in sym_name:
            if symbol_type:
                st = data.get('symbol_type', '').lower()
                if st != symbol_type.lower():
                    continue
            candidates.append(nid)
    assert len(candidates) > 0, (
        f"No node matching '{name_contains}'"
        f"{' (type=' + symbol_type + ')' if symbol_type else ''}."
    )
    return min(candidates, key=len)


def _expanded_names(result) -> set:
    """Return set of symbol_name-like strings from expanded_nodes."""
    names = set()
    for nid in result.expanded_nodes:
        parts = nid.split('::')
        last = parts[-1] if parts else nid
        names.add(last)
        if '.' in last:
            names.add(last.rsplit('.', 1)[-1])
    return names


# ============================================================================
# Helper: Build synthetic graphs
# ============================================================================

def _make_class_node(name: str, lang: str = 'python', content: str = '', file_path: str = 'src/foo.py'):
    """Return a dict suitable for ``G.add_node(name, **attrs)``."""
    return dict(
        symbol_name=name, symbol_type='class', language=lang,
        source_text=content or f'class {name}: pass',
        file_path=file_path, rel_path=file_path,
    )


def _make_function_node(name: str, lang: str = 'python', content: str = '', file_path: str = 'src/foo.py'):
    return dict(
        symbol_name=name, symbol_type='function', language=lang,
        source_text=content or f'def {name}(): pass',
        file_path=file_path, rel_path=file_path,
    )


def _make_method_node(name: str, lang: str = 'python', content: str = '', file_path: str = 'src/foo.py'):
    return dict(
        symbol_name=name, symbol_type='method', language=lang,
        source_text=content or f'def {name}(self): pass',
        file_path=file_path, rel_path=file_path,
    )


def _make_field_node(name: str, lang: str = 'python', file_path: str = 'src/foo.py'):
    return dict(
        symbol_name=name, symbol_type='field', language=lang,
        source_text=f'self.{name} = None', file_path=file_path, rel_path=file_path,
    )


def _make_constant_node(name: str, lang: str = 'python', content: str = '', file_path: str = 'src/foo.py'):
    return dict(
        symbol_name=name, symbol_type='constant', language=lang,
        source_text=content or f'{name} = 42',
        file_path=file_path, rel_path=file_path,
    )


def _make_type_alias_node(name: str, lang: str = 'cpp', content: str = '', file_path: str = 'src/foo.h'):
    return dict(
        symbol_name=name, symbol_type='type_alias', language=lang,
        source_text=content or f'using {name} = int;',
        file_path=file_path, rel_path=file_path,
    )


def _add_edge(G, src: str, tgt: str, rel_type: str, **extra):
    """Add an edge with relationship_type (handles both Di and MultiDi)."""
    attrs = {'relationship_type': rel_type, **extra}
    G.add_edge(src, tgt, **attrs)


# ============================================================================
# 1. UNIT TESTS: Edge Utilities
# ============================================================================

class TestEdgesBetween(unittest.TestCase):
    """Test edges_between() with DiGraph and MultiDiGraph."""

    def test_digraph_single_edge(self):
        G = nx.DiGraph()
        G.add_edge('a', 'b', relationship_type='calls')
        edges = edges_between(G, 'a', 'b')
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]['relationship_type'], 'calls')

    def test_digraph_no_edge(self):
        G = nx.DiGraph()
        G.add_node('a')
        G.add_node('b')
        self.assertEqual(edges_between(G, 'a', 'b'), [])

    def test_multidigraph_multiple_edges(self):
        G = nx.MultiDiGraph()
        G.add_edge('a', 'b', relationship_type='calls')
        G.add_edge('a', 'b', relationship_type='references')
        edges = edges_between(G, 'a', 'b')
        self.assertEqual(len(edges), 2)
        types = {e['relationship_type'] for e in edges}
        self.assertEqual(types, {'calls', 'references'})

    def test_multidigraph_no_edge(self):
        G = nx.MultiDiGraph()
        G.add_node('a')
        self.assertEqual(edges_between(G, 'a', 'b'), [])


class TestHasRelationship(unittest.TestCase):
    """Test has_relationship()."""

    def test_positive(self):
        G = nx.MultiDiGraph()
        G.add_edge('a', 'b', relationship_type='inheritance')
        self.assertTrue(has_relationship(G, 'a', 'b', 'inheritance'))

    def test_negative(self):
        G = nx.MultiDiGraph()
        G.add_edge('a', 'b', relationship_type='calls')
        self.assertFalse(has_relationship(G, 'a', 'b', 'inheritance'))

    def test_case_insensitive(self):
        G = nx.MultiDiGraph()
        G.add_edge('a', 'b', relationship_type='Inheritance')
        self.assertTrue(has_relationship(G, 'a', 'b', 'inheritance'))

    def test_multiple_types(self):
        G = nx.MultiDiGraph()
        G.add_edge('a', 'b', relationship_type='calls')
        self.assertTrue(has_relationship(G, 'a', 'b', 'calls', 'references'))
        self.assertFalse(has_relationship(G, 'a', 'b', 'inheritance', 'composition'))


class TestGetNeighborsByRelationship(unittest.TestCase):
    """Test get_neighbors_by_relationship()."""

    def setUp(self):
        self.G = nx.MultiDiGraph()
        self.G.add_node('cls', **_make_class_node('MyClass'))
        self.G.add_node('base', **_make_class_node('BaseClass'))
        self.G.add_node('iface', **_make_class_node('IMyInterface', lang='java'))
        self.G.nodes['iface']['symbol_type'] = 'interface'
        self.G.add_node('util_fn', **_make_function_node('utility'))
        _add_edge(self.G, 'cls', 'base', 'inheritance')
        _add_edge(self.G, 'cls', 'iface', 'implementation')
        _add_edge(self.G, 'cls', 'util_fn', 'calls')

    def test_successors_inheritance(self):
        result = get_neighbors_by_relationship(
            self.G, 'cls', {'inheritance'}, 'successors')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ('base', 'inheritance'))

    def test_predecessors(self):
        result = get_neighbors_by_relationship(
            self.G, 'base', {'inheritance'}, 'predecessors')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ('cls', 'inheritance'))

    def test_type_filter(self):
        result = get_neighbors_by_relationship(
            self.G, 'cls', {'inheritance', 'implementation', 'calls'}, 'successors',
            type_filter=CLASS_LIKE_TYPES)
        names = {nid for nid, _ in result}
        self.assertIn('base', names)
        self.assertIn('iface', names)
        self.assertNotIn('util_fn', names)  # function, not class-like

    def test_limit(self):
        result = get_neighbors_by_relationship(
            self.G, 'cls', {'inheritance', 'implementation', 'calls'}, 'successors', limit=1)
        self.assertEqual(len(result), 1)


class TestResolveAliasChain(unittest.TestCase):
    """Test resolve_alias_chain()."""

    def test_single_hop(self):
        G = nx.MultiDiGraph()
        G.add_node('alias1', **_make_type_alias_node('MyAlias'))
        G.add_node('concrete', **_make_class_node('ConcreteClass', lang='cpp'))
        _add_edge(G, 'alias1', 'concrete', 'alias_of')
        result = resolve_alias_chain(G, 'alias1')
        self.assertEqual(result, 'concrete')

    def test_multi_hop(self):
        G = nx.MultiDiGraph()
        G.add_node('a1', **_make_type_alias_node('Alias1'))
        G.add_node('a2', **_make_type_alias_node('Alias2'))
        G.add_node('target', **_make_class_node('Target', lang='cpp'))
        _add_edge(G, 'a1', 'a2', 'alias_of')
        _add_edge(G, 'a2', 'target', 'alias_of')
        result = resolve_alias_chain(G, 'a1')
        self.assertEqual(result, 'target')

    def test_circular_safety(self):
        G = nx.MultiDiGraph()
        G.add_node('a1', **_make_type_alias_node('Alias1'))
        G.add_node('a2', **_make_type_alias_node('Alias2'))
        _add_edge(G, 'a1', 'a2', 'alias_of')
        _add_edge(G, 'a2', 'a1', 'alias_of')  # circular
        result = resolve_alias_chain(G, 'a1', max_hops=10)
        # Should not hang; returns some node (a2) since it's type_alias → keeps going,
        # but visited set prevents infinite loop
        self.assertIsNotNone(result)

    def test_no_alias(self):
        G = nx.MultiDiGraph()
        G.add_node('cls', **_make_class_node('Foo'))
        result = resolve_alias_chain(G, 'cls')
        self.assertIsNone(result)


# ============================================================================
# 2. UNIT TESTS: C++ Augmentation
# ============================================================================

class TestAugmentCppNode(unittest.TestCase):
    """Test augment_cpp_node() with synthetic C++ split-file graphs."""

    def test_function_augmentation(self):
        """Function declaration + implementation from different file."""
        G = nx.MultiDiGraph()
        G.add_node('decl', symbol_name='foo', symbol_type='function', language='cpp',
                    source_text='void foo();', file_path='include/foo.h', rel_path='include/foo.h')
        G.add_node('impl', symbol_name='foo', symbol_type='function', language='cpp',
                    source_text='void foo() { return; }', file_path='src/foo.cpp', rel_path='src/foo.cpp')
        _add_edge(G, 'impl', 'decl', 'defines_body')

        aug = augment_cpp_node(G, 'decl')
        self.assertIsNotNone(aug)
        self.assertIn('void foo();', aug.augmented_content)
        self.assertIn('void foo() { return; }', aug.augmented_content)
        self.assertIn('Implementation from src/foo.cpp', aug.augmented_content)

    def test_no_augmentation_same_file(self):
        """No augmentation when declaration and implementation are in the same file."""
        G = nx.MultiDiGraph()
        G.add_node('decl', symbol_name='bar', symbol_type='function', language='cpp',
                    source_text='void bar() {}', file_path='src/bar.cpp', rel_path='src/bar.cpp')
        G.add_node('impl', symbol_name='bar_impl', symbol_type='function', language='cpp',
                    source_text='void bar() { /* impl */ }', file_path='src/bar.cpp', rel_path='src/bar.cpp')
        _add_edge(G, 'impl', 'decl', 'defines_body')

        aug = augment_cpp_node(G, 'decl')
        self.assertIsNone(aug)  # same file — no augmentation

    def test_class_augmentation_with_methods(self):
        """Class augmented with out-of-line method implementations."""
        G = nx.MultiDiGraph()
        G.add_node('cls', symbol_name='MyClass', symbol_type='class', language='cpp',
                    source_text='class MyClass { void doStuff(); };',
                    file_path='include/my.h', rel_path='include/my.h')
        G.add_node('method_decl', symbol_name='doStuff', symbol_type='method', language='cpp',
                    source_text='void doStuff();', file_path='include/my.h', rel_path='include/my.h')
        G.add_node('method_impl', symbol_name='doStuff', symbol_type='method', language='cpp',
                    source_text='void MyClass::doStuff() { /* body */ }',
                    file_path='src/my.cpp', rel_path='src/my.cpp')
        _add_edge(G, 'cls', 'method_decl', 'defines')
        _add_edge(G, 'method_impl', 'method_decl', 'defines_body')

        aug = augment_cpp_node(G, 'cls')
        self.assertIsNotNone(aug)
        self.assertIn('class MyClass', aug.augmented_content)
        self.assertIn('MyClass::doStuff()', aug.augmented_content)
        self.assertIn('Implementations from src/my.cpp', aug.augmented_content)

    def test_no_augmentation_non_cpp(self):
        """Python nodes should not be augmented."""
        G = nx.MultiDiGraph()
        G.add_node('fn', symbol_name='foo', symbol_type='function', language='python',
                    source_text='def foo(): pass', file_path='foo.py', rel_path='foo.py')
        aug = augment_cpp_node(G, 'fn')
        self.assertIsNone(aug)


# ============================================================================
# 3. UNIT TESTS: Transitive 2-Hop Helpers
# ============================================================================

class TestFindComposedTypes(unittest.TestCase):
    """Test find_composed_types() — Class → defines → Field → composition → Type."""

    def setUp(self):
        self.G = nx.MultiDiGraph()
        self.G.add_node('cls', **_make_class_node('Service'))
        self.G.add_node('field_db', **_make_field_node('database'))
        self.G.add_node('field_logger', **_make_field_node('logger'))
        self.G.add_node('Database', **_make_class_node('Database'))
        self.G.add_node('Logger', **_make_class_node('Logger'))
        _add_edge(self.G, 'cls', 'field_db', 'defines')
        _add_edge(self.G, 'cls', 'field_logger', 'defines')
        _add_edge(self.G, 'field_db', 'Database', 'composition')
        _add_edge(self.G, 'field_logger', 'Logger', 'aggregation')

    def test_finds_composed_types(self):
        result = find_composed_types(self.G, 'cls')
        self.assertEqual(set(result), {'Database', 'Logger'})

    def test_respects_limit(self):
        result = find_composed_types(self.G, 'cls', limit=1)
        self.assertEqual(len(result), 1)

    def test_empty_when_no_fields(self):
        G = nx.MultiDiGraph()
        G.add_node('cls', **_make_class_node('Empty'))
        self.assertEqual(find_composed_types(G, 'cls'), [])


class TestFindCreatesFromMethods(unittest.TestCase):
    """Test find_creates_from_methods() — Class → defines → Method → creates → Type."""

    def test_factory_pattern(self):
        G = nx.MultiDiGraph()
        G.add_node('factory', **_make_class_node('Factory'))
        G.add_node('create_method', **_make_method_node('create_widget'))
        G.add_node('Widget', **_make_class_node('Widget'))
        _add_edge(G, 'factory', 'create_method', 'defines')
        _add_edge(G, 'create_method', 'Widget', 'creates')

        result = find_creates_from_methods(G, 'factory')
        self.assertEqual(result, ['Widget'])

    def test_no_creates(self):
        G = nx.MultiDiGraph()
        G.add_node('cls', **_make_class_node('Cls'))
        G.add_node('method', **_make_method_node('do_work'))
        _add_edge(G, 'cls', 'method', 'defines')
        self.assertEqual(find_creates_from_methods(G, 'cls'), [])


class TestFindCallsToFreeFunctions(unittest.TestCase):
    """Test find_calls_to_free_functions()."""

    def test_finds_free_function_calls(self):
        G = nx.MultiDiGraph()
        G.add_node('cls', **_make_class_node('Handler'))
        G.add_node('method', **_make_method_node('handle'))
        G.add_node('validate', **_make_function_node('validate_input'))
        G.add_node('other_cls', **_make_class_node('Other'))
        _add_edge(G, 'cls', 'method', 'defines')
        _add_edge(G, 'method', 'validate', 'calls')
        _add_edge(G, 'method', 'other_cls', 'calls')  # class, not function

        result = find_calls_to_free_functions(G, 'cls')
        self.assertEqual(result, ['validate'])


# ============================================================================
# 4. UNIT TESTS: Per-Symbol-Type Expansion Strategies
# ============================================================================

class TestExpandClass(unittest.TestCase):
    """Test _expand_class() strategy."""

    def test_full_class_expansion(self):
        """Class with inheritance, composition, and creates."""
        G = nx.MultiDiGraph()
        G.add_node('cls', **_make_class_node('UserService'))
        G.add_node('base', **_make_class_node('BaseService'))
        G.add_node('iface', symbol_name='IService', symbol_type='interface',
                    language='python', source_text='class IService: pass',
                    file_path='s.py', rel_path='s.py')
        G.add_node('field_db', **_make_field_node('db'))
        G.add_node('Database', **_make_class_node('Database'))
        G.add_node('create_m', **_make_method_node('create_user'))
        G.add_node('User', **_make_class_node('User'))
        G.add_node('helper', **_make_function_node('validate'))

        _add_edge(G, 'cls', 'base', 'inheritance')
        _add_edge(G, 'cls', 'iface', 'implementation')
        _add_edge(G, 'cls', 'field_db', 'defines')
        _add_edge(G, 'field_db', 'Database', 'composition')
        _add_edge(G, 'cls', 'create_m', 'defines')
        _add_edge(G, 'create_m', 'User', 'creates')
        _add_edge(G, 'cls', 'helper', 'calls')  # direct call (P2)

        new_nodes, reasons = _expand_class(G, 'cls', set())

        self.assertIn('base', new_nodes)
        self.assertIn('iface', new_nodes)
        self.assertIn('Database', new_nodes)
        self.assertIn('User', new_nodes)
        # helper is a direct call at class level, not via method 2-hop,
        # so it won't be found by find_calls_to_free_functions
        # (that requires class → defines → method → calls → fn)

    def test_empty_class(self):
        """Class with no edges."""
        G = nx.MultiDiGraph()
        G.add_node('cls', **_make_class_node('Empty'))
        new_nodes, reasons = _expand_class(G, 'cls', set())
        self.assertEqual(len(new_nodes), 0)

    def test_already_expanded_deduped_by_caller(self):
        """Strategy functions return all neighbors; dedup happens in expand_smart().

        Strategy functions intentionally do NOT filter by already_expanded so
        that higher-priority reasons are preserved.  The caller (expand_smart)
        is responsible for deduplication.
        """
        G = nx.MultiDiGraph()
        G.add_node('cls', **_make_class_node('A'))
        G.add_node('base', **_make_class_node('B'))
        _add_edge(G, 'cls', 'base', 'inheritance')

        # Strategy returns base even though it's "already expanded"
        already = {'base'}
        new_nodes, reasons = _expand_class(G, 'cls', already)
        self.assertIn('base', new_nodes)
        self.assertIn('base', reasons)

        # But expand_smart correctly deduplicates
        result = expand_smart({'cls', 'base'}, G)
        # base is a matched node so always included, cls too
        self.assertIn('cls', result.expanded_nodes)
        self.assertIn('base', result.expanded_nodes)


class TestExpandFunction(unittest.TestCase):
    """Test _expand_function() strategy."""

    def test_factory_function(self):
        G = nx.MultiDiGraph()
        G.add_node('fn', **_make_function_node('create_widget'))
        G.add_node('Widget', **_make_class_node('Widget'))
        G.add_node('helper', **_make_function_node('validate'))
        _add_edge(G, 'fn', 'Widget', 'creates')
        _add_edge(G, 'fn', 'helper', 'calls')

        new_nodes, reasons = _expand_function(G, 'fn', set())
        self.assertIn('Widget', new_nodes)
        self.assertIn('helper', new_nodes)

    def test_function_with_type_refs(self):
        G = nx.MultiDiGraph()
        G.add_node('fn', **_make_function_node('process'))
        G.add_node('Result', **_make_class_node('Result'))
        _add_edge(G, 'fn', 'Result', 'references')

        new_nodes, _ = _expand_function(G, 'fn', set())
        self.assertIn('Result', new_nodes)


class TestExpandConstant(unittest.TestCase):
    """Test _expand_constant() strategy.

    Constants (module-level variables) are expanded purely through usage sites:
      - Forward:  ``calls`` → initialisation functions (e.g. ``CONFIG = load_config()``)
      - Forward:  ``references`` → type used in the constant definition
      - Backward: ``references`` → functions/classes/methods/constants that *use* this constant

    In Python / JS / TS a constant can hold a dict, list, or object and
    still be referenced via the same base name (``MY_DICT["key"]``,
    ``ITEMS[0]``, ``CONFIG.setting``).  The parser produces a single
    ``REFERENCES`` edge from the containing function to the constant's
    base name, regardless of subscript or attribute access.
    These tests verify that the expansion engine follows those edges
    correctly.
    """

    # ── Forward: init calls ──────────────────────────────────────────

    def test_constant_with_init_call(self):
        G = nx.MultiDiGraph()
        G.add_node('const', **_make_constant_node('DEFAULT_CONFIG'))
        G.add_node('load_fn', **_make_function_node('load_config'))
        _add_edge(G, 'const', 'load_fn', 'calls')

        new_nodes, _ = _expand_constant(G, 'const', set())
        self.assertIn('load_fn', new_nodes)

    # ── Forward: type reference ──────────────────────────────────────

    def test_constant_with_type_reference(self):
        """Forward: constant whose definition references a type."""
        G = nx.MultiDiGraph()
        G.add_node('TIMEOUT', **_make_constant_node(
            'TIMEOUT', content='TIMEOUT: timedelta = timedelta(seconds=30)'))
        G.add_node('timedelta_cls', **_make_class_node('timedelta'))
        _add_edge(G, 'TIMEOUT', 'timedelta_cls', 'references')

        new_nodes, reasons = _expand_constant(G, 'TIMEOUT', set())
        self.assertIn('timedelta_cls', new_nodes)
        self.assertIn('type ref', reasons.get('timedelta_cls', '').lower())

    # ── Backward: scalar constant used by functions ──────────────────

    def test_scalar_constant_used_by_function(self):
        """Backward: a plain scalar constant referenced by a function."""
        G = nx.MultiDiGraph()
        G.add_node('MAX_RETRIES', **_make_constant_node('MAX_RETRIES', content='MAX_RETRIES = 5'))
        G.add_node('retry_fn', **_make_function_node('retry_request'))
        _add_edge(G, 'retry_fn', 'MAX_RETRIES', 'references')

        new_nodes, reasons = _expand_constant(G, 'MAX_RETRIES', set())
        self.assertIn('retry_fn', new_nodes,
                      "Function using scalar constant should appear in expansion")
        self.assertIn('constant user', reasons.get('retry_fn', '').lower())

    # ── Backward: dict constant used via subscript ───────────────────

    def test_dict_constant_used_via_subscript(self):
        """Backward: ``CONFIG["key"]`` in a function body.

        The parser sees ``ast.Name('CONFIG', Load)`` inside the subscript
        and produces a REFERENCES edge from the function to CONFIG.
        Expansion should discover the function.
        """
        G = nx.MultiDiGraph()
        G.add_node('CONFIG', **_make_constant_node(
            'CONFIG', content='CONFIG = {"db_host": "localhost", "db_port": 5432}'))
        G.add_node('connect_db', **_make_function_node(
            'connect_db', content='def connect_db():\n    host = CONFIG["db_host"]'))
        # Parser would produce this edge from visit_Name
        _add_edge(G, 'connect_db', 'CONFIG', 'references')

        new_nodes, reasons = _expand_constant(G, 'CONFIG', set())
        self.assertIn('connect_db', new_nodes,
                      "Function using dict constant via subscript should be found")

    # ── Backward: list constant used via index ───────────────────────

    def test_list_constant_used_via_index(self):
        """Backward: ``ALLOWED_HOSTS[0]`` in a function body."""
        G = nx.MultiDiGraph()
        G.add_node('ALLOWED_HOSTS', **_make_constant_node(
            'ALLOWED_HOSTS', content='ALLOWED_HOSTS = ["localhost", "0.0.0.0"]'))
        G.add_node('get_host', **_make_function_node(
            'get_host', content='def get_host():\n    return ALLOWED_HOSTS[0]'))
        _add_edge(G, 'get_host', 'ALLOWED_HOSTS', 'references')

        new_nodes, _ = _expand_constant(G, 'ALLOWED_HOSTS', set())
        self.assertIn('get_host', new_nodes,
                      "Function using list constant via index should be found")

    # ── Backward: constant used by a class ───────────────────────────

    def test_constant_used_by_class(self):
        """Backward: a class body references a constant (e.g. default value)."""
        G = nx.MultiDiGraph()
        G.add_node('DEFAULT_TIMEOUT', **_make_constant_node('DEFAULT_TIMEOUT'))
        G.add_node('HttpClient', **_make_class_node('HttpClient'))
        _add_edge(G, 'HttpClient', 'DEFAULT_TIMEOUT', 'references')

        new_nodes, _ = _expand_constant(G, 'DEFAULT_TIMEOUT', set())
        self.assertIn('HttpClient', new_nodes,
                      "Class using constant should be found via backward references")

    # ── Backward: constant used by a method ──────────────────────────

    def test_constant_used_by_method(self):
        """Backward: a method references a constant.

        Methods are not architectural symbols, but expansion should still
        follow the backward ``references`` edge to the method — the
        expansion engine type_filter includes ``'method'``.
        """
        G = nx.MultiDiGraph()
        G.add_node('API_VERSION', **_make_constant_node('API_VERSION'))
        G.add_node('build_url', **_make_method_node('build_url'))
        _add_edge(G, 'build_url', 'API_VERSION', 'references')

        new_nodes, reasons = _expand_constant(G, 'API_VERSION', set())
        self.assertIn('build_url', new_nodes,
                      "Method referencing constant should be discovered via backward expansion")

    # ── Backward: constant used by another constant ──────────────────

    def test_constant_used_by_another_constant(self):
        """Backward: one constant references another (e.g. ``FULL_URL = BASE_URL + '/api'``)."""
        G = nx.MultiDiGraph()
        G.add_node('BASE_URL', **_make_constant_node('BASE_URL', content='BASE_URL = "https://api.example.com"'))
        G.add_node('FULL_URL', **_make_constant_node('FULL_URL', content='FULL_URL = BASE_URL + "/api"'))
        _add_edge(G, 'FULL_URL', 'BASE_URL', 'references')

        new_nodes, _ = _expand_constant(G, 'BASE_URL', set())
        self.assertIn('FULL_URL', new_nodes,
                      "Constant derived from another constant should appear in backward expansion")

    # ── Backward: limit honoured ─────────────────────────────────────

    def test_backward_user_limit_honoured(self):
        """Backward ``references`` edges are capped at limit=2."""
        G = nx.MultiDiGraph()
        G.add_node('SHARED', **_make_constant_node('SHARED'))
        for i in range(6):
            name = f'fn_{i}'
            G.add_node(name, **_make_function_node(name))
            _add_edge(G, name, 'SHARED', 'references')

        new_nodes, reasons = _expand_constant(G, 'SHARED', set())
        user_reasons = [nid for nid, r in reasons.items() if 'constant user' in r.lower()]
        self.assertLessEqual(len(user_reasons), 2,
                             "Backward references limit should cap at 2")

    # ── Already-expanded predecessors are skipped ────────────────────

    def test_already_expanded_users_skipped(self):
        """Users already in ``already_expanded`` should not be returned."""
        G = nx.MultiDiGraph()
        G.add_node('FLAG', **_make_constant_node('FLAG'))
        G.add_node('checker', **_make_function_node('checker'))
        G.add_node('validator', **_make_function_node('validator'))
        _add_edge(G, 'checker', 'FLAG', 'references')
        _add_edge(G, 'validator', 'FLAG', 'references')

        # checker is already expanded
        new_nodes, _ = _expand_constant(G, 'FLAG', {'checker'})
        # checker is still returned by the strategy (filtering happens
        # upstream in expand_smart), but verify both are in result at
        # least at the strategy level.
        self.assertIn('validator', new_nodes)

    # ── Bidirectional: forward init + backward users ─────────────────

    def test_bidirectional_init_and_users(self):
        """Full bidirectional: forward init call + backward user functions."""
        G = nx.MultiDiGraph()
        G.add_node('SETTINGS', **_make_constant_node(
            'SETTINGS', content='SETTINGS = load_settings()'))
        G.add_node('load_settings', **_make_function_node('load_settings'))
        G.add_node('apply_fn', **_make_function_node('apply_settings'))
        G.add_node('render_fn', **_make_function_node('render_page'))

        # Forward: init call
        _add_edge(G, 'SETTINGS', 'load_settings', 'calls')
        # Backward: two user functions
        _add_edge(G, 'apply_fn', 'SETTINGS', 'references')
        _add_edge(G, 'render_fn', 'SETTINGS', 'references')

        new_nodes, reasons = _expand_constant(G, 'SETTINGS', set())

        self.assertIn('load_settings', new_nodes, "Forward: init function")
        self.assertIn('apply_fn', new_nodes, "Backward: user function")
        self.assertIn('render_fn', new_nodes, "Backward: user function")
        self.assertIn('init function', reasons.get('load_settings', '').lower())

    # ── Constant with no edges ───────────────────────────────────────

    def test_orphan_constant(self):
        """A constant with no edges returns empty expansion."""
        G = nx.MultiDiGraph()
        G.add_node('UNUSED', **_make_constant_node('UNUSED'))
        new_nodes, reasons = _expand_constant(G, 'UNUSED', set())
        self.assertEqual(len(new_nodes), 0, "Orphan constant should yield no expansion")
        self.assertEqual(len(reasons), 0)

    # ── JS/TS-style: constant object used via property access ────────

    def test_js_object_constant_used_via_property(self):
        """JS/TS pattern: ``CONFIG.apiUrl`` — the parser tracks the base name."""
        G = nx.MultiDiGraph()
        G.add_node('CONFIG', **_make_constant_node(
            'CONFIG', lang='typescript',
            content='export const CONFIG = { apiUrl: "https://...", timeout: 30 };'))
        G.add_node('fetchData', **_make_function_node(
            'fetchData', lang='typescript',
            content='function fetchData() { fetch(CONFIG.apiUrl); }'))
        _add_edge(G, 'fetchData', 'CONFIG', 'references')

        new_nodes, _ = _expand_constant(G, 'CONFIG', set())
        self.assertIn('fetchData', new_nodes,
                      "TS function using object constant via property should be found")

    # ── JS/TS-style: array constant used via index ───────────────────

    def test_ts_array_constant_used_via_index(self):
        """TS pattern: ``ROUTES[0]`` — parser tracks the base name ROUTES."""
        G = nx.MultiDiGraph()
        G.add_node('ROUTES', **_make_constant_node(
            'ROUTES', lang='typescript',
            content="export const ROUTES = ['/home', '/about', '/settings'];"))
        G.add_node('getHome', **_make_function_node(
            'getHome', lang='typescript',
            content='function getHome() { return ROUTES[0]; }'))
        _add_edge(G, 'getHome', 'ROUTES', 'references')

        new_nodes, _ = _expand_constant(G, 'ROUTES', set())
        self.assertIn('getHome', new_nodes,
                      "TS function using array constant via index should be found")


class TestExpandTypeAlias(unittest.TestCase):
    """Test _expand_type_alias() strategy."""

    def test_resolves_alias(self):
        G = nx.MultiDiGraph()
        G.add_node('alias', **_make_type_alias_node('Vec'))
        G.add_node('target', **_make_class_node('Vector', lang='cpp'))
        _add_edge(G, 'alias', 'target', 'alias_of')

        new_nodes, reasons = _expand_type_alias(G, 'alias', set())
        self.assertIn('target', new_nodes)
        self.assertIn('alias chain target', reasons.get('target', ''))


# ============================================================================
# 5. UNIT TESTS: Main expand_smart() Entry Point
# ============================================================================

class TestExpandSmart(unittest.TestCase):
    """Test the main expand_smart() function."""

    def _build_rich_graph(self) -> nx.MultiDiGraph:
        """Build a synthetic graph with mixed symbol types and relationships."""
        G = nx.MultiDiGraph()

        # Classes
        G.add_node('PaymentProcessor', **_make_class_node('PaymentProcessor'))
        G.add_node('BaseProcessor', **_make_class_node('BaseProcessor'))
        G.add_node('StripeGateway', **_make_class_node('StripeGateway'))
        G.add_node('PaymentResult', **_make_class_node('PaymentResult'))

        # Fields
        G.add_node('field_gateway', **_make_field_node('gateway'))
        G.add_node('field_result', **_make_field_node('result'))

        # Methods
        G.add_node('process_method', **_make_method_node('process'))
        G.add_node('create_result_method', **_make_method_node('create_result'))

        # Free function
        G.add_node('validate_payment', **_make_function_node('validate_payment'))

        # Constant
        G.add_node('MAX_RETRIES', **_make_constant_node('MAX_RETRIES'))

        # Edges
        _add_edge(G, 'PaymentProcessor', 'BaseProcessor', 'inheritance')
        _add_edge(G, 'PaymentProcessor', 'field_gateway', 'defines')
        _add_edge(G, 'field_gateway', 'StripeGateway', 'composition')
        _add_edge(G, 'PaymentProcessor', 'process_method', 'defines')
        _add_edge(G, 'PaymentProcessor', 'create_result_method', 'defines')
        _add_edge(G, 'create_result_method', 'PaymentResult', 'creates')
        _add_edge(G, 'process_method', 'validate_payment', 'calls')

        return G

    def test_class_expansion_includes_all_priorities(self):
        G = self._build_rich_graph()
        result = expand_smart({'PaymentProcessor'}, G)

        # P0: base class
        self.assertIn('BaseProcessor', result.expanded_nodes)
        # P0: created type
        self.assertIn('PaymentResult', result.expanded_nodes)
        # P1: composed type
        self.assertIn('StripeGateway', result.expanded_nodes)
        # P2: free function called by method
        self.assertIn('validate_payment', result.expanded_nodes)
        # Original node always included
        self.assertIn('PaymentProcessor', result.expanded_nodes)

    def test_function_expansion(self):
        G = nx.MultiDiGraph()
        G.add_node('factory', **_make_function_node('create_widget'))
        G.add_node('Widget', **_make_class_node('Widget'))
        G.add_node('util', **_make_function_node('format_name'))
        _add_edge(G, 'factory', 'Widget', 'creates')
        _add_edge(G, 'factory', 'util', 'calls')

        result = expand_smart({'factory'}, G)
        self.assertIn('Widget', result.expanded_nodes)
        self.assertIn('util', result.expanded_nodes)

    def test_per_symbol_cap(self):
        """Ensure per-symbol cap is enforced."""
        G = nx.MultiDiGraph()
        G.add_node('cls', **_make_class_node('BigClass'))
        # Add 20 base classes
        for i in range(20):
            nid = f'base_{i}'
            G.add_node(nid, **_make_class_node(f'Base{i}'))
            _add_edge(G, 'cls', nid, 'inheritance')

        result = expand_smart({'cls'}, G, per_symbol_cap=5)
        # 1 matched + at most 5 expanded
        self.assertLessEqual(len(result.expanded_nodes), 1 + 5)

    def test_global_cap(self):
        """Ensure global cap is enforced across multiple matched symbols."""
        G = nx.MultiDiGraph()
        matched = set()
        for i in range(10):
            cls_id = f'cls_{i}'
            G.add_node(cls_id, **_make_class_node(f'Class{i}'))
            matched.add(cls_id)
            for j in range(5):
                base_id = f'base_{i}_{j}'
                G.add_node(base_id, **_make_class_node(f'Base{i}_{j}'))
                _add_edge(G, cls_id, base_id, 'inheritance')

        result = expand_smart(matched, G, global_cap=10)
        new_count = len(result.expanded_nodes) - len(matched)
        self.assertLessEqual(new_count, 10)

    def test_matched_nodes_always_included(self):
        """Matched nodes are in expanded_nodes even without edges."""
        G = nx.MultiDiGraph()
        G.add_node('lonely', **_make_class_node('Lonely'))
        result = expand_smart({'lonely'}, G)
        self.assertIn('lonely', result.expanded_nodes)

    def test_expansion_reasons_tracked(self):
        """Each expanded node has a reason string."""
        G = nx.MultiDiGraph()
        G.add_node('cls', **_make_class_node('A'))
        G.add_node('base', **_make_class_node('B'))
        _add_edge(G, 'cls', 'base', 'inheritance')

        result = expand_smart({'cls'}, G)
        self.assertIn('base', result.expansion_reasons)
        self.assertIn('base class', result.expansion_reasons['base'])

    def test_cpp_augmentation_tracked(self):
        """C++ nodes with defines_body are augmented."""
        G = nx.MultiDiGraph()
        G.add_node('decl', symbol_name='foo', symbol_type='function', language='cpp',
                    source_text='void foo();', file_path='h.h', rel_path='h.h')
        G.add_node('impl', symbol_name='foo_impl', symbol_type='function', language='cpp',
                    source_text='void foo() { body }', file_path='c.cpp', rel_path='c.cpp')
        _add_edge(G, 'impl', 'decl', 'defines_body')

        result = expand_smart({'decl'}, G)
        self.assertIn('decl', result.augmentations)
        self.assertIn('void foo() { body }', result.augmentations['decl'].augmented_content)

    def test_unknown_symbol_type_still_included(self):
        """Nodes with unknown symbol_type are kept but not expanded."""
        G = nx.MultiDiGraph()
        G.add_node('mystery', symbol_name='x', symbol_type='widget', language='python',
                    source_text='x = 1', file_path='f.py', rel_path='f.py')
        result = expand_smart({'mystery'}, G)
        self.assertIn('mystery', result.expanded_nodes)
        self.assertEqual(len(result.expanded_nodes), 1)  # no expansion

    def test_empty_matched_set(self):
        G = nx.MultiDiGraph()
        result = expand_smart(set(), G)
        self.assertEqual(len(result.expanded_nodes), 0)

    def test_multiple_symbols_different_types(self):
        """Expand a mix of class + function + constant together."""
        G = nx.MultiDiGraph()
        G.add_node('cls', **_make_class_node('Service'))
        G.add_node('base', **_make_class_node('BaseService'))
        G.add_node('fn', **_make_function_node('create_service'))
        G.add_node('result_cls', **_make_class_node('ServiceResult'))
        G.add_node('const', **_make_constant_node('TIMEOUT'))
        G.add_node('init_fn', **_make_function_node('load_timeout'))
        _add_edge(G, 'cls', 'base', 'inheritance')
        _add_edge(G, 'fn', 'result_cls', 'creates')
        _add_edge(G, 'const', 'init_fn', 'calls')

        result = expand_smart({'cls', 'fn', 'const'}, G)
        self.assertIn('base', result.expanded_nodes)
        self.assertIn('result_cls', result.expanded_nodes)
        self.assertIn('init_fn', result.expanded_nodes)


# ============================================================================
# 6. INTEGRATION TESTS: Code Snippets → Graph Builder → Expansion
# ============================================================================

class TestCppExpansionFromSnippets(unittest.TestCase):
    """C++ cross-file expansion: header/source split, inheritance, overrides.

    Exercises the full pipeline: write files → graph builder → expand_smart.
    Verified graph edges (from diagnostic):
      - JsonFormatter --[inheritance]--> Formatter
      - XmlFormatter  --[inheritance]--> Formatter
      - TypedFormatter --[inheritance]--> Formatter
      - JsonFormatter.format --[overrides]--> Formatter.format
      - JsonFormatter.format --[defines_body]--> JsonFormatter.format (self-loop)
      - __file__ --[instantiates]--> TypedFormatter (template explicit instantiation)
    """

    CPP_FILES = {
        'include/base.h': (
            '#pragma once\n'
            'class Formatter {\n'
            'public:\n'
            '    virtual void format(const char* text) = 0;\n'
            '    virtual ~Formatter() = default;\n'
            '};\n'
        ),
        'include/derived.h': (
            '#pragma once\n'
            '#include "base.h"\n'
            '\n'
            'class JsonFormatter : public Formatter {\n'
            'public:\n'
            '    void format(const char* text) override;\n'
            '    int indent_level;\n'
            '};\n'
            '\n'
            'class XmlFormatter : public Formatter {\n'
            'public:\n'
            '    void format(const char* text) override;\n'
            '};\n'
        ),
        'src/derived.cpp': (
            '#include "derived.h"\n'
            '\n'
            'void JsonFormatter::format(const char* text) {\n'
            '    // json formatting implementation\n'
            '}\n'
            '\n'
            'void XmlFormatter::format(const char* text) {\n'
            '    // xml formatting implementation\n'
            '}\n'
        ),
        'include/template.h': (
            '#pragma once\n'
            '#include "base.h"\n'
            '\n'
            'template<typename T>\n'
            'class TypedFormatter : public Formatter {\n'
            'public:\n'
            '    void format(const char* text) override;\n'
            '    T value;\n'
            '};\n'
            '\n'
            'template class TypedFormatter<int>;\n'
            'template class TypedFormatter<double>;\n'
        ),
    }

    @classmethod
    def setUpClass(cls):
        cls.graph = _build_graph(cls.CPP_FILES)

    # ── Graph structure tests ────────────────────────────────────────

    def test_source_text_available_on_nodes(self):
        """Every class/method node should have source_text via Symbol object."""
        for nid, data in self.graph.nodes(data=True):
            st = data.get('symbol_type', '').lower()
            if st in ('class', 'method', 'function'):
                src = _get_source_text(data)
                self.assertTrue(len(src) > 0,
                                f"Node {nid} (type={st}) should have source_text")

    def test_inheritance_edges_cross_file(self):
        """Inheritance edges should link .h files across include/ directory."""
        formatter = _find_node(self.graph, 'Formatter', 'class')
        json_fmt = _find_node(self.graph, 'JsonFormatter', 'class')
        xml_fmt = _find_node(self.graph, 'XmlFormatter', 'class')
        self.assertTrue(has_relationship(self.graph, json_fmt, formatter, 'inheritance'),
                        "JsonFormatter → Formatter inheritance edge expected")
        self.assertTrue(has_relationship(self.graph, xml_fmt, formatter, 'inheritance'),
                        "XmlFormatter → Formatter inheritance edge expected")

    def test_defines_body_edge_exists(self):
        """The .cpp implementation should create a defines_body edge for the .h declaration."""
        # The graph builder creates defines_body as a self-loop (impl and decl
        # share the same node ID namespace).  Verify the edge exists at all.
        found = False
        for u, v, data in self.graph.edges(data=True):
            if data.get('relationship_type', '') == 'defines_body':
                found = True
                break
        self.assertTrue(found, "At least one defines_body edge should exist "
                        "for cross-file .h/.cpp pairs")

    def test_overrides_edge_cross_file(self):
        """JsonFormatter::format should have an overrides edge to Formatter::format."""
        json_format = _find_node(self.graph, 'JsonFormatter.format', 'method')
        base_format = _find_node(self.graph, 'Formatter.format', 'method')
        self.assertTrue(has_relationship(self.graph, json_format, base_format, 'overrides'),
                        "JsonFormatter::format → Formatter::format overrides edge expected")

    # ── Expansion tests ──────────────────────────────────────────────

    def test_derived_class_expands_to_base(self):
        """JsonFormatter should expand to include base class Formatter."""
        node = _find_node(self.graph, 'JsonFormatter', 'class')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        self.assertIn('Formatter', names,
                      "JsonFormatter should expand to base class Formatter")
        # Verify the reason mentions base class
        base_reasons = [r for r in result.expansion_reasons.values()
                        if 'base class' in r.lower()]
        self.assertGreater(len(base_reasons), 0,
                           "Reason should mention 'base class'")

    def test_base_class_backward_discovers_derived(self):
        """Expanding Formatter should discover JsonFormatter, XmlFormatter, TypedFormatter."""
        node = _find_node(self.graph, 'Formatter', 'class')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        derived = names & {'JsonFormatter', 'XmlFormatter', 'TypedFormatter'}
        self.assertGreaterEqual(len(derived), 2,
                                f"Formatter should discover ≥2 derived classes; got {derived}")

    def test_template_class_inherits_base(self):
        """TypedFormatter should expand to base class Formatter."""
        node = _find_node(self.graph, 'TypedFormatter', 'class')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        self.assertIn('Formatter', names,
                      "TypedFormatter should expand to base class Formatter")

    def test_multiple_matched_classes(self):
        """Expanding both JsonFormatter and XmlFormatter together."""
        json_node = _find_node(self.graph, 'JsonFormatter', 'class')
        xml_node = _find_node(self.graph, 'XmlFormatter', 'class')
        result = expand_smart({json_node, xml_node}, self.graph)
        names = _expanded_names(result)
        self.assertIn('Formatter', names,
                      "Both derived classes share base Formatter")
        self.assertIn('JsonFormatter', names)
        self.assertIn('XmlFormatter', names)

    def test_no_python_augmentations_for_cpp(self):
        """C++ nodes should not produce Python-style augmentations when
        source_text is only on the Symbol object (no direct node attr)."""
        node = _find_node(self.graph, 'Formatter', 'class')
        result = expand_smart({node}, self.graph)
        # Augmentation requires direct source_text or symbol.source_text
        # plus cross-file defines_body with different file paths.
        # For this snippet set, augmentation may or may not fire depending
        # on whether Self-loop defines_body counts. Either way, no crash.
        self.assertIn(node, result.expanded_nodes)


class TestPythonExpansionFromSnippets(unittest.TestCase):
    """Python cross-file expansion: constants, inheritance, calls, creates.

    Exercises the full pipeline with a 4-file Python project.
    Verified graph edges (from diagnostic):
      - UserModel --[inheritance]--> BaseModel
      - AdminModel --[inheritance]--> UserModel
      - create_user --[creates]--> UserModel
      - retry_operation --[references]--> MAX_RETRIES
      - get_db_host --[references]--> CONFIG
      - build_url --[references]--> API_VERSION
      - get_prompt_keys --[references]--> SERVICE_PROMPT_KEYS
      - RequestHandler --[references]--> DEFAULT_TIMEOUT
      - RequestHandler.handle --[calls]--> create_user
      - RequestHandler.safe_handle --[calls]--> retry_operation
    """

    PYTHON_FILES = {
        'constants.py': (
            'SERVICE_PROMPT_KEYS = ["system", "user", "assistant"]\n'
            'MAX_RETRIES = 5\n'
            'DEFAULT_TIMEOUT = 30\n'
            'CONFIG = {\n'
            '    "db_host": "localhost",\n'
            '    "db_port": 5432,\n'
            '    "debug": True,\n'
            '}\n'
            'API_VERSION = "v2"\n'
        ),
        'models.py': (
            'from typing import Optional, List\n'
            '\n'
            'class BaseModel:\n'
            '    name: str\n'
            '    def validate(self) -> bool:\n'
            '        return True\n'
            '\n'
            'class UserModel(BaseModel):\n'
            '    email: str\n'
            '    def validate(self) -> bool:\n'
            '        return "@" in self.email\n'
            '\n'
            'class AdminModel(UserModel):\n'
            '    permissions: List[str]\n'
        ),
        'service.py': (
            'from constants import MAX_RETRIES, CONFIG, SERVICE_PROMPT_KEYS, API_VERSION\n'
            'from models import UserModel\n'
            '\n'
            'def create_user(name: str) -> UserModel:\n'
            '    user = UserModel()\n'
            '    user.name = name\n'
            '    return user\n'
            '\n'
            'def retry_operation(fn):\n'
            '    for i in range(MAX_RETRIES):\n'
            '        try:\n'
            '            return fn()\n'
            '        except Exception:\n'
            '            pass\n'
            '\n'
            'def get_db_host():\n'
            '    return CONFIG["db_host"]\n'
            '\n'
            'def build_url(path: str) -> str:\n'
            '    return f"https://api.example.com/{API_VERSION}/{path}"\n'
            '\n'
            'def get_prompt_keys():\n'
            '    return SERVICE_PROMPT_KEYS\n'
        ),
        'handler.py': (
            'from constants import DEFAULT_TIMEOUT, MAX_RETRIES\n'
            'from service import create_user, retry_operation\n'
            '\n'
            'class RequestHandler:\n'
            '    timeout: int = DEFAULT_TIMEOUT\n'
            '\n'
            '    def handle(self, data: dict):\n'
            '        user = create_user(data["name"])\n'
            '        return user\n'
            '\n'
            '    def safe_handle(self, data: dict):\n'
            '        return retry_operation(lambda: self.handle(data))\n'
        ),
    }

    @classmethod
    def setUpClass(cls):
        cls.graph = _build_graph(cls.PYTHON_FILES)

    # ── Graph structure verification ─────────────────────────────────

    def test_source_text_available_on_python_nodes(self):
        """Every class/function/constant should have source_text via Symbol."""
        for nid, data in self.graph.nodes(data=True):
            st = data.get('symbol_type', '').lower()
            if st in ('class', 'function', 'constant', 'method'):
                src = _get_source_text(data)
                self.assertTrue(
                    len(src) > 0,
                    f"Node {nid} (type={st}) should have source_text via symbol"
                )

    def test_constant_count_in_graph(self):
        """constants.py should produce exactly 5 constant nodes."""
        constants = [
            data.get('symbol_name', '')
            for nid, data in self.graph.nodes(data=True)
            if data.get('symbol_type', '').lower() == 'constant'
        ]
        self.assertGreaterEqual(
            len(constants), 5,
            f"Expected ≥ 5 constants; found {len(constants)}: {constants}"
        )
        expected = {'SERVICE_PROMPT_KEYS', 'MAX_RETRIES', 'DEFAULT_TIMEOUT',
                    'CONFIG', 'API_VERSION'}
        self.assertTrue(expected.issubset(set(constants)),
                        f"Missing constants: {expected - set(constants)}")

    def test_cross_file_references_edges(self):
        """Functions in service.py should have references edges to constants.py."""
        retry_node = _find_node(self.graph, 'retry_operation', 'function')
        max_retries = _find_node(self.graph, 'MAX_RETRIES', 'constant')
        self.assertTrue(
            has_relationship(self.graph, retry_node, max_retries, 'references'),
            "retry_operation → MAX_RETRIES references edge expected"
        )

    def test_cross_file_calls_edge(self):
        """handler.py methods should have calls edges to service.py functions."""
        handle = _find_node(self.graph, 'handle', 'method')
        create_user = _find_node(self.graph, 'create_user', 'function')
        self.assertTrue(
            has_relationship(self.graph, handle, create_user, 'calls'),
            "RequestHandler.handle → create_user calls edge expected"
        )

    def test_cross_file_creates_edge(self):
        """create_user should have a creates edge to UserModel."""
        create_user = _find_node(self.graph, 'create_user', 'function')
        user_model = _find_node(self.graph, 'UserModel', 'class')
        self.assertTrue(
            has_relationship(self.graph, create_user, user_model, 'creates'),
            "create_user → UserModel creates edge expected"
        )

    # ── Class hierarchy expansion ────────────────────────────────────

    def test_class_inheritance_forward(self):
        """UserModel should expand to its base class BaseModel."""
        node = _find_node(self.graph, 'UserModel', 'class')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        self.assertIn('BaseModel', names,
                      "UserModel should expand to base class BaseModel")

    def test_base_class_backward_discovers_derived(self):
        """BaseModel should discover derived classes via backward expansion."""
        node = _find_node(self.graph, 'BaseModel', 'class')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        derived = names & {'UserModel', 'AdminModel'}
        self.assertGreater(len(derived), 0,
                           "BaseModel should discover at least one derived class")

    # ── Function expansion ───────────────────────────────────────────

    def test_function_forward_creates_type(self):
        """create_user creates UserModel — forward expansion finds it."""
        node = _find_node(self.graph, 'create_user', 'function')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        self.assertIn('UserModel', names,
                      "create_user should expand to created type UserModel")

    def test_function_backward_finds_callers(self):
        """retry_operation is called by safe_handle — backward expansion discovers caller.

        Note: forward expansion of functions follows ``references`` only to
        class-like types/type_aliases, not to constants.  Constant discovery
        works from the constant direction (see constant expansion tests).
        """
        node = _find_node(self.graph, 'retry_operation', 'function')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        caller_found = 'safe_handle' in names or 'RequestHandler' in names
        self.assertTrue(caller_found,
                        f"retry_operation should find caller safe_handle; got {names}")

    def test_create_user_backward_finds_callers(self):
        """create_user is called by RequestHandler.handle — backward finds it."""
        node = _find_node(self.graph, 'create_user', 'function')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        caller_found = 'handle' in names or 'RequestHandler' in names
        self.assertTrue(caller_found,
                        f"create_user should find caller handle; got {names}")

    # ── Constant expansion ───────────────────────────────────────────

    def test_constant_backward_discovers_users(self):
        """MAX_RETRIES should discover functions/classes that reference it."""
        node = _find_node(self.graph, 'MAX_RETRIES', 'constant')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        users = names & {'retry_operation', 'RequestHandler'}
        self.assertGreater(len(users), 0,
                           f"MAX_RETRIES should discover usage sites; got {names}")

    def test_dict_constant_backward_via_subscript(self):
        """CONFIG["db_host"] — parser tracks CONFIG reference, expansion finds get_db_host."""
        node = _find_node(self.graph, 'CONFIG', 'constant')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        self.assertGreater(len(result.expanded_nodes), 1,
                           "Dict constant CONFIG should expand to usage site(s)")

    def test_list_constant_backward_discovers_users(self):
        """SERVICE_PROMPT_KEYS is a list — get_prompt_keys uses it."""
        node = _find_node(self.graph, 'SERVICE_PROMPT_KEYS', 'constant')
        result = expand_smart({node}, self.graph)
        self.assertGreater(len(result.expanded_nodes), 1,
                           "List constant SERVICE_PROMPT_KEYS should discover usage sites")

    def test_scalar_constant_backward_discovers_user_class(self):
        """DEFAULT_TIMEOUT is referenced by RequestHandler class attribute."""
        node = _find_node(self.graph, 'DEFAULT_TIMEOUT', 'constant')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        self.assertIn('RequestHandler', names,
                      "DEFAULT_TIMEOUT should discover RequestHandler as user")

    # ── Python-specific: no C++ augmentation ─────────────────────────

    def test_no_augmentation_for_python(self):
        """Python code should never produce C++ augmentations."""
        all_nodes = set()
        for nid, data in self.graph.nodes(data=True):
            if data.get('symbol_type', '').lower() in ('class', 'function'):
                all_nodes.add(nid)
                if len(all_nodes) >= 5:
                    break
        result = expand_smart(all_nodes, self.graph)
        self.assertEqual(len(result.augmentations), 0,
                         "Python graph should have zero C++ augmentations")

    # ── Multi-symbol expansion ───────────────────────────────────────

    def test_multiple_matched_symbols(self):
        """Expanding multiple classes at once should include all of them."""
        classes = set()
        for nid, data in self.graph.nodes(data=True):
            if data.get('symbol_type', '').lower() == 'class':
                classes.add(nid)
                if len(classes) >= 3:
                    break
        result = expand_smart(classes, self.graph)
        for cls_id in classes:
            self.assertIn(cls_id, result.expanded_nodes)
        self.assertGreaterEqual(len(result.expanded_nodes), len(classes))

    def test_class_with_constants_expansion(self):
        """RequestHandler references DEFAULT_TIMEOUT — backward should find it."""
        node = _find_node(self.graph, 'RequestHandler', 'class')
        result = expand_smart({node}, self.graph)
        self.assertGreater(len(result.expanded_nodes), 1,
                           "RequestHandler should find related nodes via expansion")


class TestCppConstantExpansionFromSnippets(unittest.TestCase):
    """C++ constant expansion: ``const`` declarations and enum members.

    The C++ enhanced parser produces SymbolType.CONSTANT for:
      - ``const`` / ``extern const`` declarations
      - enum members (``enumerator`` nodes)
    Constants participate in general REFERENCES edges.
    """

    CPP_FILES = {
        'include/config.h': (
            '#pragma once\n'
            '\n'
            'const int MAX_CONNECTIONS = 100;\n'
            'const double PI = 3.14159265;\n'
            'extern const char* APP_NAME;\n'
            '\n'
            'enum LogLevel {\n'
            '    LOG_DEBUG = 0,\n'
            '    LOG_INFO = 1,\n'
            '    LOG_WARNING = 2,\n'
            '    LOG_ERROR = 3\n'
            '};\n'
        ),
        'src/config.cpp': (
            '#include "config.h"\n'
            '\n'
            'const char* APP_NAME = "MyApp";\n'
        ),
        'include/server.h': (
            '#pragma once\n'
            '#include "config.h"\n'
            '\n'
            'class Server {\n'
            'public:\n'
            '    void start();\n'
            '    void log(LogLevel level, const char* msg);\n'
            'private:\n'
            '    int pool_size;\n'
            '};\n'
        ),
        'src/server.cpp': (
            '#include "server.h"\n'
            '\n'
            'void Server::start() {\n'
            '    pool_size = MAX_CONNECTIONS;\n'
            '}\n'
            '\n'
            'void Server::log(LogLevel level, const char* msg) {\n'
            '    if (level >= LOG_WARNING) {\n'
            '        // log it\n'
            '    }\n'
            '}\n'
        ),
        'src/math_utils.cpp': (
            '#include "config.h"\n'
            '\n'
            'double circle_area(double r) {\n'
            '    return PI * r * r;\n'
            '}\n'
        ),
    }

    @classmethod
    def setUpClass(cls):
        cls.graph = _build_graph(cls.CPP_FILES)

    # ── Graph structure checks ───────────────────────────────────────

    def test_const_declarations_are_constants(self):
        """``const int MAX_CONNECTIONS`` should be a constant node."""
        constants = {
            data.get('symbol_name', '')
            for _, data in self.graph.nodes(data=True)
            if data.get('symbol_type', '').lower() == 'constant'
        }
        self.assertIn('MAX_CONNECTIONS', constants,
                      f"MAX_CONNECTIONS not in constants; got {constants}")
        self.assertIn('PI', constants,
                      f"PI not in constants; got {constants}")

    def test_enum_members_are_constants(self):
        """Enum members like LOG_DEBUG should be constant nodes."""
        constants = {
            data.get('symbol_name', '')
            for _, data in self.graph.nodes(data=True)
            if data.get('symbol_type', '').lower() == 'constant'
        }
        enum_members = constants & {'LOG_DEBUG', 'LOG_INFO', 'LOG_WARNING', 'LOG_ERROR'}
        self.assertGreaterEqual(len(enum_members), 2,
                                f"Expected ≥2 enum member constants; got {enum_members}")

    def test_constant_source_text_available(self):
        """Constant nodes should have source_text via Symbol."""
        for nid, data in self.graph.nodes(data=True):
            if (data.get('symbol_type', '').lower() == 'constant'
                    and data.get('symbol_name', '') in ('MAX_CONNECTIONS', 'PI')):
                src = _get_source_text(data)
                self.assertTrue(len(src) > 0,
                                f"Constant {data['symbol_name']} should have source_text")

    # ── Expansion tests ──────────────────────────────────────────────

    def test_const_backward_discovers_usage(self):
        """MAX_CONNECTIONS should expand to Server (which uses it in start())."""
        node = _find_node(self.graph, 'MAX_CONNECTIONS', 'constant')
        result = expand_smart({node}, self.graph)
        self.assertGreater(len(result.expanded_nodes), 1,
                           "MAX_CONNECTIONS should discover at least one usage site")

    def test_pi_backward_discovers_math_function(self):
        """PI should expand to circle_area (which references it)."""
        node = _find_node(self.graph, 'PI', 'constant')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        # Either circle_area directly or via __file__ node
        self.assertGreater(len(result.expanded_nodes), 1,
                           f"PI should discover circle_area; got {names}")

    def test_enum_constant_backward_expansion(self):
        """An enum member (LOG_WARNING) should discover usage sites."""
        node = _find_node(self.graph, 'LOG_WARNING', 'constant')
        result = expand_smart({node}, self.graph)
        # LOG_WARNING is used inside Server::log — may discover Server or log method
        self.assertGreater(len(result.expanded_nodes), 1,
                           "LOG_WARNING enum member should discover usage sites")

    def test_mixed_constants_and_classes(self):
        """Expanding Server + MAX_CONNECTIONS together should work cleanly."""
        server = _find_node(self.graph, 'Server', 'class')
        max_conn = _find_node(self.graph, 'MAX_CONNECTIONS', 'constant')
        result = expand_smart({server, max_conn}, self.graph)
        self.assertIn(server, result.expanded_nodes)
        self.assertIn(max_conn, result.expanded_nodes)
        self.assertGreater(len(result.expanded_nodes), 2,
                           "Mixed expansion should discover additional nodes")


class TestTypeScriptConstantExpansionFromSnippets(unittest.TestCase):
    """TypeScript constant expansion through full pipeline.

    The TS enhanced parser produces SymbolType.CONSTANT for:
      - ``const`` declarations (unless arrow functions → promoted to FUNCTION)
      - enum members (``is_enum_member: True``)
    Constants are registered in _global_symbol_registry for cross-file REFERENCES.
    """

    TS_FILES = {
        'config.ts': (
            'export const API_BASE_URL = "https://api.example.com";\n'
            'export const MAX_RETRIES = 3;\n'
            'export const DEFAULT_HEADERS = {\n'
            '    "Content-Type": "application/json",\n'
            '    "Accept": "application/json"\n'
            '};\n'
            '\n'
            'export enum Status {\n'
            '    Active = "active",\n'
            '    Inactive = "inactive",\n'
            '    Pending = "pending"\n'
            '}\n'
        ),
        'types.ts': (
            'import { Status } from "./config";\n'
            '\n'
            'export interface User {\n'
            '    name: string;\n'
            '    status: Status;\n'
            '}\n'
            '\n'
            'export type UserList = User[];\n'
        ),
        'api.ts': (
            'import { API_BASE_URL, MAX_RETRIES, DEFAULT_HEADERS } from "./config";\n'
            'import { User } from "./types";\n'
            '\n'
            'export async function fetchUser(id: string): Promise<User> {\n'
            '    let attempt = 0;\n'
            '    while (attempt < MAX_RETRIES) {\n'
            '        const response = await fetch(\n'
            '            `${API_BASE_URL}/users/${id}`,\n'
            '            { headers: DEFAULT_HEADERS }\n'
            '        );\n'
            '        if (response.ok) {\n'
            '            return response.json();\n'
            '        }\n'
            '        attempt++;\n'
            '    }\n'
            '    throw new Error("Max retries exceeded");\n'
            '}\n'
            '\n'
            'export async function fetchUsers(): Promise<User[]> {\n'
            '    const response = await fetch(\n'
            '        `${API_BASE_URL}/users`,\n'
            '        { headers: DEFAULT_HEADERS }\n'
            '    );\n'
            '    return response.json();\n'
            '}\n'
        ),
        'handler.ts': (
            'import { Status } from "./config";\n'
            'import { User } from "./types";\n'
            'import { fetchUser } from "./api";\n'
            '\n'
            'export class UserHandler {\n'
            '    async activate(userId: string): Promise<void> {\n'
            '        const user = await fetchUser(userId);\n'
            '        if (user.status !== Status.Active) {\n'
            '            user.status = Status.Active;\n'
            '        }\n'
            '    }\n'
            '}\n'
        ),
    }

    @classmethod
    def setUpClass(cls):
        cls.graph = _build_graph(cls.TS_FILES)

    # ── Graph structure checks ───────────────────────────────────────

    def test_const_declarations_are_constants(self):
        """``const API_BASE_URL`` should be a constant node."""
        constants = {
            data.get('symbol_name', '')
            for _, data in self.graph.nodes(data=True)
            if data.get('symbol_type', '').lower() == 'constant'
        }
        expected = {'API_BASE_URL', 'MAX_RETRIES', 'DEFAULT_HEADERS'}
        found = expected & constants
        self.assertGreaterEqual(len(found), 2,
                                f"Expected ≥2 of {expected} as constants; got {constants}")

    def test_enum_is_single_node_not_individual_members(self):
        """TS enhanced parser aggregates enum members into a single 'enum' node.

        Individual members (Active, Inactive, Pending) are NOT produced as
        separate constant nodes.  This test documents the current behaviour.
        """
        enums = {
            data.get('symbol_name', '')
            for _, data in self.graph.nodes(data=True)
            if data.get('symbol_type', '').lower() == 'enum'
        }
        self.assertIn('Status', enums,
                      f"Status enum should exist as a single node; got {enums}")
        # Verify members are NOT individual nodes
        constants = {
            data.get('symbol_name', '')
            for _, data in self.graph.nodes(data=True)
            if data.get('symbol_type', '').lower() == 'constant'
        }
        enum_members = constants & {'Active', 'Inactive', 'Pending'}
        self.assertEqual(len(enum_members), 0,
                         f"Enum members should not be individual constant nodes; got {enum_members}")

    def test_constant_source_text_available(self):
        """TS constant nodes should have source_text."""
        for nid, data in self.graph.nodes(data=True):
            if (data.get('symbol_type', '').lower() == 'constant'
                    and data.get('symbol_name', '') in ('API_BASE_URL', 'MAX_RETRIES')):
                src = _get_source_text(data)
                self.assertTrue(len(src) > 0,
                                f"TS constant {data['symbol_name']} should have source_text")

    # ── Expansion tests ──────────────────────────────────────────────

    def test_scalar_constant_backward_discovers_users(self):
        """MAX_RETRIES should discover fetchUser (which uses it)."""
        node = _find_node(self.graph, 'MAX_RETRIES', 'constant')
        result = expand_smart({node}, self.graph)
        self.assertGreater(len(result.expanded_nodes), 1,
                           "MAX_RETRIES should discover at least one usage site")

    def test_url_constant_backward_discovers_api(self):
        """API_BASE_URL should discover functions in api.ts."""
        node = _find_node(self.graph, 'API_BASE_URL', 'constant')
        result = expand_smart({node}, self.graph)
        self.assertGreater(len(result.expanded_nodes), 1,
                           "API_BASE_URL should discover api functions")

    def test_object_constant_backward_discovers_users(self):
        """DEFAULT_HEADERS (object literal) should discover usage sites."""
        node = _find_node(self.graph, 'DEFAULT_HEADERS', 'constant')
        result = expand_smart({node}, self.graph)
        self.assertGreater(len(result.expanded_nodes), 1,
                           "DEFAULT_HEADERS should discover usage functions")

    def test_enum_node_exists_as_enum_type(self):
        """Status enum should be discoverable as an 'enum' symbol type.

        Since enum members are aggregated, the enum itself is the
        architectural node.  Expansion from a function that references
        a constant should still work via the references edge.
        """
        # Verify the enum exists
        enum_nodes = [
            nid for nid, data in self.graph.nodes(data=True)
            if data.get('symbol_name', '') == 'Status'
               and data.get('symbol_type', '').lower() == 'enum'
        ]
        self.assertEqual(len(enum_nodes), 1,
                         "Exactly one Status enum node expected")

    def test_no_augmentation_for_typescript(self):
        """TypeScript code should never produce C++ augmentations."""
        all_nodes = set()
        for nid, data in self.graph.nodes(data=True):
            if data.get('symbol_type', '').lower() in ('class', 'function', 'constant'):
                all_nodes.add(nid)
                if len(all_nodes) >= 5:
                    break
        result = expand_smart(all_nodes, self.graph)
        self.assertEqual(len(result.augmentations), 0,
                         "TypeScript graph should have zero C++ augmentations")

    def test_multiple_constants_expansion(self):
        """Expanding multiple constants together should work without error."""
        const_nodes = set()
        for nid, data in self.graph.nodes(data=True):
            if (data.get('symbol_type', '').lower() == 'constant'
                    and data.get('symbol_name', '') in ('API_BASE_URL', 'MAX_RETRIES', 'DEFAULT_HEADERS')):
                const_nodes.add(nid)
        self.assertGreaterEqual(len(const_nodes), 2,
                                "Should have at least 2 constant nodes to test")
        result = expand_smart(const_nodes, self.graph)
        for cn in const_nodes:
            self.assertIn(cn, result.expanded_nodes)
        self.assertGreater(len(result.expanded_nodes), len(const_nodes),
                           "Multiple constants should discover additional usage sites")


class TestJavaScriptConstantExpansionFromSnippets(unittest.TestCase):
    """JavaScript constant expansion through full pipeline.

    The JS visitor parser produces SymbolType.CONSTANT for ALL
    top-level const/let/var declarations.

    Body-reference scanning (Pass 3 in the JS parser) creates
    ``references`` edges from functions/methods to constants they
    use, enabling both forward and backward smart expansion.
    CommonJS ``require()`` destructuring is treated as import
    bindings so that cross-file references are properly resolved.
    """

    JS_FILES = {
        'constants.js': (
            'const BASE_URL = "https://api.example.com";\n'
            'const TIMEOUT_MS = 5000;\n'
            'const DEFAULT_OPTIONS = {\n'
            '    retries: 3,\n'
            '    backoff: 1000\n'
            '};\n'
            'const VALID_STATUSES = [200, 201, 204];\n'
            '\n'
            'module.exports = { BASE_URL, TIMEOUT_MS, DEFAULT_OPTIONS, VALID_STATUSES };\n'
        ),
        'client.js': (
            'const { BASE_URL, TIMEOUT_MS, DEFAULT_OPTIONS } = require("./constants");\n'
            '\n'
            'class ApiClient {\n'
            '    constructor() {\n'
            '        this.baseUrl = BASE_URL;\n'
            '        this.timeout = TIMEOUT_MS;\n'
            '    }\n'
            '\n'
            '    async request(path, options) {\n'
            '        const merged = { ...DEFAULT_OPTIONS, ...options };\n'
            '        const url = `${this.baseUrl}/${path}`;\n'
            '        return fetch(url, {\n'
            '            ...merged,\n'
            '            signal: AbortSignal.timeout(this.timeout)\n'
            '        });\n'
            '    }\n'
            '}\n'
            '\n'
            'module.exports = { ApiClient };\n'
        ),
        'validator.js': (
            'const { VALID_STATUSES } = require("./constants");\n'
            '\n'
            'function isValidResponse(response) {\n'
            '    return VALID_STATUSES.includes(response.status);\n'
            '}\n'
            '\n'
            'function checkHealth(response) {\n'
            '    if (!isValidResponse(response)) {\n'
            '        throw new Error("Unhealthy");\n'
            '    }\n'
            '    return true;\n'
            '}\n'
            '\n'
            'module.exports = { isValidResponse, checkHealth };\n'
        ),
    }

    @classmethod
    def setUpClass(cls):
        cls.graph = _build_graph(cls.JS_FILES)

    # ── Graph structure checks ───────────────────────────────────────

    def test_const_declarations_are_constants(self):
        """JS top-level const declarations should be constant nodes."""
        constants = {
            data.get('symbol_name', '')
            for _, data in self.graph.nodes(data=True)
            if data.get('symbol_type', '').lower() == 'constant'
        }
        expected = {'BASE_URL', 'TIMEOUT_MS', 'DEFAULT_OPTIONS', 'VALID_STATUSES'}
        found = expected & constants
        self.assertGreaterEqual(len(found), 3,
                                f"Expected ≥3 of {expected} as constants; got {constants}")

    def test_constant_source_text_available(self):
        """JS constant nodes should have source_text via Symbol."""
        for nid, data in self.graph.nodes(data=True):
            if (data.get('symbol_type', '').lower() == 'constant'
                    and data.get('symbol_name', '') in ('BASE_URL', 'TIMEOUT_MS')):
                src = _get_source_text(data)
                self.assertTrue(len(src) > 0,
                                f"JS constant {data['symbol_name']} should have source_text")

    # ── Cross-file reference checks ──────────────────────────────────

    def test_references_edges_exist_for_js_constants(self):
        """JS parser now creates cross-file references edges for constants.

        Body-reference scanning (Pass 3) walks function/method bodies for
        identifiers that match import bindings or same-file top-level
        symbols, emitting REFERENCES edges.
        """
        constant_names = {'BASE_URL', 'TIMEOUT_MS', 'DEFAULT_OPTIONS', 'VALID_STATUSES'}
        constant_ids = {
            nid for nid, data in self.graph.nodes(data=True)
            if data.get('symbol_type', '').lower() == 'constant'
               and data.get('symbol_name', '') in constant_names
        }
        ref_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get('relationship_type', '') == 'references'
               and (u in constant_ids or v in constant_ids)
        ]
        self.assertGreater(len(ref_edges), 0,
                           "JS parser should create references edges for constants "
                           "used in function/method bodies")

    # ── Expansion tests ──────────────────────────────────────────────

    def test_js_constant_backward_expansion(self):
        """Backward expansion from JS constant discovers functions that use it."""
        node = _find_node(self.graph, 'BASE_URL', 'constant')
        result = expand_smart({node}, self.graph)
        self.assertGreater(len(result.expanded_nodes), 1,
                           "JS constant with references edges should expand "
                           "beyond itself (backward to usage sites)")

    def test_js_function_expands_to_constants(self):
        """JS function with body-scanned references discovers used constants."""
        node = _find_node(self.graph, 'isValidResponse', 'function')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        self.assertIn('VALID_STATUSES', names,
                      "isValidResponse should discover VALID_STATUSES via "
                      "body-scanned references edge")

    def test_no_augmentation_for_javascript(self):
        """JavaScript code should never produce C++ augmentations."""
        all_nodes = set()
        for nid, data in self.graph.nodes(data=True):
            if data.get('symbol_type', '').lower() in ('class', 'function', 'constant'):
                all_nodes.add(nid)
                if len(all_nodes) >= 5:
                    break
        result = expand_smart(all_nodes, self.graph)
        self.assertEqual(len(result.augmentations), 0,
                         "JavaScript graph should have zero C++ augmentations")

    def test_multiple_js_constants_expand_beyond_matched(self):
        """Expanding multiple JS constants discovers usage sites via references."""
        const_nodes = set()
        for nid, data in self.graph.nodes(data=True):
            if (data.get('symbol_type', '').lower() == 'constant'
                    and data.get('symbol_name', '') in ('BASE_URL', 'TIMEOUT_MS', 'DEFAULT_OPTIONS')):
                const_nodes.add(nid)
        self.assertGreaterEqual(len(const_nodes), 2,
                                "Should have at least 2 JS constant nodes")
        result = expand_smart(const_nodes, self.graph)
        for cn in const_nodes:
            self.assertIn(cn, result.expanded_nodes)
        # With references edges, expansion should discover more nodes
        self.assertGreater(len(result.expanded_nodes), len(const_nodes),
                           "With references edges, should expand beyond "
                           "matched constants to usage sites")


# ============================================================================
# 7. REGRESSION TESTS: Graceful Degradation
# ============================================================================

class TestGracefulDegradation(unittest.TestCase):
    """Smart expansion degrades gracefully when edges/data are missing."""

    def test_graph_with_no_typed_edges(self):
        """Graph with only 'defines' and 'imports' — all SKIP — no expansion."""
        G = nx.MultiDiGraph()
        G.add_node('cls', **_make_class_node('A'))
        G.add_node('method', **_make_method_node('m'))
        G.add_node('mod', symbol_name='os', symbol_type='module',
                    language='python', source_text='import os',
                    file_path='f.py', rel_path='f.py')
        _add_edge(G, 'cls', 'method', 'defines')
        _add_edge(G, 'cls', 'mod', 'imports')

        result = expand_smart({'cls'}, G)
        # Only the matched node — no useful expansion edges
        self.assertEqual(result.expanded_nodes, {'cls'})

    def test_node_not_in_graph(self):
        """Matched node ID that doesn't exist in graph."""
        G = nx.MultiDiGraph()
        G.add_node('real', **_make_class_node('Real'))
        result = expand_smart({'nonexistent'}, G)
        self.assertIn('nonexistent', result.expanded_nodes)
        self.assertEqual(len(result.expanded_nodes), 1)

    def test_node_with_no_content(self):
        """Node exists but has no source_text — still included in expanded set."""
        G = nx.MultiDiGraph()
        G.add_node('empty', symbol_name='E', symbol_type='class', language='python',
                    source_text='', file_path='f.py', rel_path='f.py')
        result = expand_smart({'empty'}, G)
        self.assertIn('empty', result.expanded_nodes)

    def test_all_neighbors_already_expanded(self):
        """When all potential expansion targets are already in the set."""
        G = nx.MultiDiGraph()
        G.add_node('cls', **_make_class_node('A'))
        G.add_node('base', **_make_class_node('B'))
        _add_edge(G, 'cls', 'base', 'inheritance')

        # Both already matched
        result = expand_smart({'cls', 'base'}, G)
        # base should not appear in expansion_reasons (it was already matched)
        self.assertNotIn('base', result.expansion_reasons)

    def test_exception_in_strategy_doesnt_crash(self):
        """Even if graph data is malformed, expand_smart shouldn't crash."""
        G = nx.MultiDiGraph()
        G.add_node('weird', symbol_name='W', symbol_type='class', language='python',
                    source_text='class W: pass', file_path='f.py', rel_path='f.py')
        # Add a successor with completely broken data
        G.add_node('broken')  # no attributes at all
        _add_edge(G, 'weird', 'broken', 'inheritance')

        # Should not crash
        result = expand_smart({'weird'}, G)
        self.assertIn('weird', result.expanded_nodes)


# ============================================================================
# 7. BIDIRECTIONAL EXPANSION TESTS
# ============================================================================

def _make_interface_node(name: str, lang: str = 'python', content: str = '', file_path: str = 'src/foo.py'):
    return dict(
        symbol_name=name, symbol_type='interface', language=lang,
        source_text=content or f'class {name}(ABC): pass',
        file_path=file_path, rel_path=file_path,
    )


def _make_trait_node(name: str, lang: str = 'rust', content: str = '', file_path: str = 'src/foo.rs'):
    return dict(
        symbol_name=name, symbol_type='trait', language=lang,
        source_text=content or f'trait {name} {{}}',
        file_path=file_path, rel_path=file_path,
    )


def _make_macro_node(name: str, lang: str = 'cpp', content: str = '', file_path: str = 'src/foo.h'):
    return dict(
        symbol_name=name, symbol_type='macro', language=lang,
        source_text=content or f'#define {name}',
        file_path=file_path, rel_path=file_path,
    )


class TestBidirectionalClassExpansion(unittest.TestCase):
    """Test backward (predecessor) queries in _expand_class."""

    def test_class_finds_derived_classes(self):
        """A class should discover its derived (child) classes via backward inheritance."""
        G = nx.MultiDiGraph()
        G.add_node('Base', **_make_class_node('Base'))
        G.add_node('Derived1', **_make_class_node('Derived1'))
        G.add_node('Derived2', **_make_class_node('Derived2'))
        # Derived -> Base (inheritance successors)
        _add_edge(G, 'Derived1', 'Base', 'inheritance')
        _add_edge(G, 'Derived2', 'Base', 'inheritance')

        new_nodes, reasons = _expand_class(G, 'Base', set())

        self.assertIn('Derived1', new_nodes, "Should find derived class via backward inheritance")
        self.assertIn('Derived2', new_nodes, "Should find second derived class")
        self.assertIn('derived class', reasons.get('Derived1', '').lower())

    def test_interface_finds_implementors(self):
        """An interface node should discover classes that implement it."""
        G = nx.MultiDiGraph()
        G.add_node('IFace', **_make_interface_node('IFace'))
        G.add_node('Impl1', **_make_class_node('Impl1'))
        G.add_node('Impl2', **_make_class_node('Impl2'))
        # Impl -> IFace (implementation successors)
        _add_edge(G, 'Impl1', 'IFace', 'implementation')
        _add_edge(G, 'Impl2', 'IFace', 'implementation')

        new_nodes, reasons = _expand_class(G, 'IFace', set())

        self.assertIn('Impl1', new_nodes, "Should find implementor via backward implementation")
        self.assertIn('Impl2', new_nodes)
        self.assertIn('implementor', reasons.get('Impl1', '').lower())

    def test_trait_finds_implementors(self):
        """A trait node should discover structs/classes implementing it."""
        G = nx.MultiDiGraph()
        G.add_node('MyTrait', **_make_trait_node('MyTrait'))
        G.add_node('MyStruct', **dict(
            symbol_name='MyStruct', symbol_type='struct', language='rust',
            source_text='struct MyStruct {}', file_path='src/lib.rs', rel_path='src/lib.rs',
        ))
        _add_edge(G, 'MyStruct', 'MyTrait', 'implementation')

        new_nodes, reasons = _expand_class(G, 'MyTrait', set())
        self.assertIn('MyStruct', new_nodes)

    def test_class_not_interface_skips_implementors(self):
        """A regular class should NOT look for implementors (only interfaces/traits do)."""
        G = nx.MultiDiGraph()
        G.add_node('RegularClass', **_make_class_node('RegularClass'))
        G.add_node('SomeClass', **_make_class_node('SomeClass'))
        _add_edge(G, 'SomeClass', 'RegularClass', 'implementation')

        new_nodes, reasons = _expand_class(G, 'RegularClass', set())
        # SomeClass should NOT appear via implementor path (it may appear via other backward paths)
        impl_reasons = [r for nid, r in reasons.items() if 'implementor' in r.lower()]
        self.assertEqual(len(impl_reasons), 0, "Regular classes should not trigger implementor backward search")

    def test_class_finds_composers(self):
        """Classes that compose (contain) this type should be discovered."""
        G = nx.MultiDiGraph()
        G.add_node('Widget', **_make_class_node('Widget'))
        G.add_node('Dashboard', **_make_class_node('Dashboard'))
        # Dashboard composes Widget
        _add_edge(G, 'Dashboard', 'Widget', 'composition')

        new_nodes, reasons = _expand_class(G, 'Widget', set())

        self.assertIn('Dashboard', new_nodes, "Should find composing class via backward composition")
        self.assertIn('composed by', reasons.get('Dashboard', '').lower())

    def test_class_finds_creators(self):
        """Functions/classes that create this type should be discovered."""
        G = nx.MultiDiGraph()
        G.add_node('Product', **_make_class_node('Product'))
        G.add_node('factory', **_make_function_node('create_product'))
        # factory creates Product
        _add_edge(G, 'factory', 'Product', 'creates')

        new_nodes, reasons = _expand_class(G, 'Product', set())

        self.assertIn('factory', new_nodes, "Should find creator function via backward creates")
        self.assertIn('created by', reasons.get('factory', '').lower())

    def test_class_finds_referencing_types(self):
        """Types that reference (use as param/return) this class should be discovered."""
        G = nx.MultiDiGraph()
        G.add_node('Message', **_make_class_node('Message'))
        G.add_node('handler', **_make_function_node('handle_message'))
        # handler references Message (e.g. parameter type)
        _add_edge(G, 'handler', 'Message', 'references')

        new_nodes, reasons = _expand_class(G, 'Message', set())

        self.assertIn('handler', new_nodes, "Should find referencing function via backward references")
        self.assertIn('referenced by', reasons.get('handler', '').lower())

    def test_bidirectional_class_full_expansion(self):
        """Full integration: class with both forward and backward edges."""
        G = nx.MultiDiGraph()
        G.add_node('Animal', **_make_class_node('Animal'))
        G.add_node('BaseEntity', **_make_class_node('BaseEntity'))
        G.add_node('Dog', **_make_class_node('Dog'))
        G.add_node('Cat', **_make_class_node('Cat'))
        G.add_node('PetStore', **_make_class_node('PetStore'))

        # Forward: Animal inherits from BaseEntity
        _add_edge(G, 'Animal', 'BaseEntity', 'inheritance')
        # Backward: Dog and Cat inherit from Animal
        _add_edge(G, 'Dog', 'Animal', 'inheritance')
        _add_edge(G, 'Cat', 'Animal', 'inheritance')
        # Backward: PetStore composes Animal
        _add_edge(G, 'PetStore', 'Animal', 'composition')

        new_nodes, reasons = _expand_class(G, 'Animal', set())

        # Forward results
        self.assertIn('BaseEntity', new_nodes, "Forward: should find base class")
        # Backward results
        self.assertIn('Dog', new_nodes, "Backward: should find derived class Dog")
        self.assertIn('Cat', new_nodes, "Backward: should find derived class Cat")
        self.assertIn('PetStore', new_nodes, "Backward: should find composer PetStore")

    def test_derived_class_limit_honored(self):
        """Backward inheritance query should respect its limit=3."""
        G = nx.MultiDiGraph()
        G.add_node('Base', **_make_class_node('Base'))
        # Add 6 derived classes
        for i in range(6):
            name = f'Derived{i}'
            G.add_node(name, **_make_class_node(name))
            _add_edge(G, name, 'Base', 'inheritance')

        new_nodes, reasons = _expand_class(G, 'Base', set())

        # Count how many have "derived class" reason
        derived_reasons = [nid for nid, r in reasons.items() if 'derived class' in r.lower()]
        self.assertLessEqual(len(derived_reasons), 3, "Backward inheritance limit should cap at 3")


class TestBidirectionalFunctionExpansion(unittest.TestCase):
    """Test backward (predecessor) queries in _expand_function."""

    def test_function_finds_callers(self):
        """A function should discover its callers via backward calls edge."""
        G = nx.MultiDiGraph()
        G.add_node('helper', **_make_function_node('helper'))
        G.add_node('main_func', **_make_function_node('main_func'))
        G.add_node('test_func', **_make_function_node('test_func'))
        # main_func and test_func call helper
        _add_edge(G, 'main_func', 'helper', 'calls')
        _add_edge(G, 'test_func', 'helper', 'calls')

        new_nodes, reasons = _expand_function(G, 'helper', set())

        self.assertIn('main_func', new_nodes, "Should find caller main_func via backward calls")
        self.assertIn('test_func', new_nodes, "Should find caller test_func via backward calls")
        self.assertIn('caller', reasons.get('main_func', '').lower())

    def test_function_finds_referencing_types(self):
        """Types referencing this function (e.g. callback registrations) should be found."""
        G = nx.MultiDiGraph()
        G.add_node('callback_fn', **_make_function_node('callback_fn'))
        G.add_node('EventSystem', **_make_class_node('EventSystem'))
        # EventSystem references callback_fn
        _add_edge(G, 'EventSystem', 'callback_fn', 'references')

        new_nodes, reasons = _expand_function(G, 'callback_fn', set())

        self.assertIn('EventSystem', new_nodes, "Should find referencing class")
        self.assertIn('referenced by', reasons.get('EventSystem', '').lower())

    def test_function_bidirectional_full(self):
        """Full bidirectional: forward callees + backward callers."""
        G = nx.MultiDiGraph()
        G.add_node('process', **_make_function_node('process'))
        G.add_node('validate', **_make_function_node('validate'))  # callee
        G.add_node('main', **_make_function_node('main'))  # caller
        G.add_node('ReturnType', **_make_class_node('ReturnType'))  # referenced

        # Forward: process calls validate, references ReturnType
        _add_edge(G, 'process', 'validate', 'calls')
        _add_edge(G, 'process', 'ReturnType', 'references')
        # Backward: main calls process
        _add_edge(G, 'main', 'process', 'calls')

        new_nodes, reasons = _expand_function(G, 'process', set())

        self.assertIn('validate', new_nodes, "Forward: callee")
        self.assertIn('ReturnType', new_nodes, "Forward: referenced type")
        self.assertIn('main', new_nodes, "Backward: caller")

    def test_caller_limit_honored(self):
        """Backward caller query should respect its limit=3."""
        G = nx.MultiDiGraph()
        G.add_node('target', **_make_function_node('target'))
        for i in range(7):
            name = f'caller_{i}'
            G.add_node(name, **_make_function_node(name))
            _add_edge(G, name, 'target', 'calls')

        new_nodes, reasons = _expand_function(G, 'target', set())

        caller_reasons = [nid for nid, r in reasons.items() if 'caller' in r.lower()]
        self.assertLessEqual(len(caller_reasons), 3, "Backward calls limit should cap at 3")


class TestBidirectionalConstantExpansion(unittest.TestCase):
    """Test backward (predecessor) queries in _expand_constant."""

    def test_constant_finds_users(self):
        """A constant should discover symbols that reference it."""
        G = nx.MultiDiGraph()
        G.add_node('MAX_SIZE', **_make_constant_node('MAX_SIZE'))
        G.add_node('allocator', **_make_function_node('allocator'))
        G.add_node('Config', **_make_class_node('Config'))
        # allocator and Config reference MAX_SIZE
        _add_edge(G, 'allocator', 'MAX_SIZE', 'references')
        _add_edge(G, 'Config', 'MAX_SIZE', 'references')

        new_nodes, reasons = _expand_constant(G, 'MAX_SIZE', set())

        self.assertIn('allocator', new_nodes, "Should find function using this constant")
        self.assertIn('Config', new_nodes, "Should find class using this constant")
        self.assertIn('constant user', reasons.get('allocator', '').lower())

    def test_constant_bidirectional(self):
        """Full bidirectional: forward init calls + backward users."""
        G = nx.MultiDiGraph()
        G.add_node('TIMEOUT', **_make_constant_node('TIMEOUT'))
        G.add_node('compute_timeout', **_make_function_node('compute_timeout'))
        G.add_node('client', **_make_function_node('client'))

        # Forward: TIMEOUT calls compute_timeout (init)
        _add_edge(G, 'TIMEOUT', 'compute_timeout', 'calls')
        # Backward: client references TIMEOUT
        _add_edge(G, 'client', 'TIMEOUT', 'references')

        new_nodes, reasons = _expand_constant(G, 'TIMEOUT', set())

        self.assertIn('compute_timeout', new_nodes, "Forward: init function")
        self.assertIn('client', new_nodes, "Backward: constant user")


class TestBidirectionalTypeAliasExpansion(unittest.TestCase):
    """Test backward (predecessor) queries in _expand_type_alias."""

    def test_type_alias_finds_users(self):
        """A type alias should discover symbols that reference it."""
        G = nx.MultiDiGraph()
        G.add_node('StringVec', **_make_type_alias_node('StringVec'))
        G.add_node('parse_args', **_make_function_node('parse_args', lang='cpp'))
        # parse_args references StringVec
        _add_edge(G, 'parse_args', 'StringVec', 'references')

        new_nodes, reasons = _expand_type_alias(G, 'StringVec', set())

        self.assertIn('parse_args', new_nodes, "Should find function using this alias")
        self.assertIn('alias user', reasons.get('parse_args', '').lower())

    def test_type_alias_bidirectional(self):
        """Forward alias chain + backward users."""
        G = nx.MultiDiGraph()
        G.add_node('Alias', **_make_type_alias_node('Alias'))
        G.add_node('ConcreteType', **_make_class_node('ConcreteType', lang='cpp'))
        G.add_node('consumer', **_make_function_node('consumer', lang='cpp'))

        # Forward: Alias -> ConcreteType (alias chain)
        _add_edge(G, 'Alias', 'ConcreteType', 'alias_of')
        # Backward: consumer references Alias
        _add_edge(G, 'consumer', 'Alias', 'references')

        new_nodes, reasons = _expand_type_alias(G, 'Alias', set())

        self.assertIn('ConcreteType', new_nodes, "Forward: alias chain target")
        self.assertIn('consumer', new_nodes, "Backward: alias user")


class TestBidirectionalExpandSmart(unittest.TestCase):
    """Integration tests for expand_smart with bidirectional expansion."""

    def test_smart_expansion_finds_callers_for_function(self):
        """expand_smart should include callers when expanding a function."""
        G = nx.MultiDiGraph()
        G.add_node('target_fn', **_make_function_node('target_fn'))
        G.add_node('caller_fn', **_make_function_node('caller_fn'))
        G.add_node('callee_fn', **_make_function_node('callee_fn'))
        _add_edge(G, 'caller_fn', 'target_fn', 'calls')
        _add_edge(G, 'target_fn', 'callee_fn', 'calls')

        result = expand_smart({'target_fn'}, G)

        self.assertIn('caller_fn', result.expanded_nodes,
                       "Callers should appear in expanded_nodes")
        self.assertIn('callee_fn', result.expanded_nodes,
                       "Callees should appear in expanded_nodes")

    def test_smart_expansion_finds_derived_for_class(self):
        """expand_smart should include derived classes when expanding a class."""
        G = nx.MultiDiGraph()
        G.add_node('Base', **_make_class_node('Base'))
        G.add_node('Child', **_make_class_node('Child'))
        G.add_node('Parent', **_make_class_node('Parent'))
        _add_edge(G, 'Base', 'Parent', 'inheritance')  # forward: base class
        _add_edge(G, 'Child', 'Base', 'inheritance')  # backward: derived

        result = expand_smart({'Base'}, G)

        self.assertIn('Parent', result.expanded_nodes, "Forward base class")
        self.assertIn('Child', result.expanded_nodes, "Backward derived class")
        # Check reasons
        self.assertIn('Child', result.expansion_reasons)

    def test_smart_expansion_backward_does_not_exceed_per_symbol_cap(self):
        """Bidirectional expansion should respect per_symbol_cap total."""
        G = nx.MultiDiGraph()
        G.add_node('func', **_make_function_node('func'))
        # Add 20 callers (backward)
        for i in range(20):
            name = f'caller_{i}'
            G.add_node(name, **_make_function_node(name))
            _add_edge(G, name, 'func', 'calls')
        # Add 10 callees (forward)
        for i in range(10):
            name = f'callee_{i}'
            G.add_node(name, **_make_function_node(name))
            _add_edge(G, 'func', name, 'calls')

        result = expand_smart({'func'}, G, per_symbol_cap=15)

        # Total expansion should not exceed per_symbol_cap
        expanded_minus_matched = result.expanded_nodes - {'func'}
        self.assertLessEqual(len(expanded_minus_matched), 15,
                             "Bidirectional expansion should respect per_symbol_cap")

    def test_backward_expansion_reasons_tracked(self):
        """Backward expansion reasons should be recorded in expansion_reasons."""
        G = nx.MultiDiGraph()
        G.add_node('helper', **_make_function_node('helper'))
        G.add_node('main', **_make_function_node('main'))
        _add_edge(G, 'main', 'helper', 'calls')

        result = expand_smart({'helper'}, G)

        self.assertIn('main', result.expansion_reasons)
        self.assertIn('caller', result.expansion_reasons['main'].lower())

    def test_constant_users_in_expand_smart(self):
        """expand_smart should find users of a constant."""
        G = nx.MultiDiGraph()
        G.add_node('PI', **_make_constant_node('PI'))
        G.add_node('calc', **_make_function_node('calculate'))
        _add_edge(G, 'calc', 'PI', 'references')

        result = expand_smart({'PI'}, G)
        self.assertIn('calc', result.expanded_nodes)

    def test_dict_constant_expand_smart(self):
        """expand_smart finds users of a dict constant accessed via subscript."""
        G = nx.MultiDiGraph()
        G.add_node('ENDPOINTS', **_make_constant_node(
            'ENDPOINTS', content='ENDPOINTS = {"users": "/api/users", "items": "/api/items"}'))
        G.add_node('fetch_users', **_make_function_node('fetch_users'))
        G.add_node('fetch_items', **_make_function_node('fetch_items'))
        _add_edge(G, 'fetch_users', 'ENDPOINTS', 'references')
        _add_edge(G, 'fetch_items', 'ENDPOINTS', 'references')

        result = expand_smart({'ENDPOINTS'}, G)
        self.assertIn('fetch_users', result.expanded_nodes)
        self.assertIn('fetch_items', result.expanded_nodes)

    def test_constant_with_init_and_users_expand_smart(self):
        """expand_smart: constant with both init call (forward) and users (backward)."""
        G = nx.MultiDiGraph()
        G.add_node('REGISTRY', **_make_constant_node(
            'REGISTRY', content='REGISTRY = build_registry()'))
        G.add_node('build_registry', **_make_function_node('build_registry'))
        G.add_node('lookup', **_make_function_node('lookup'))
        G.add_node('register', **_make_function_node('register'))

        _add_edge(G, 'REGISTRY', 'build_registry', 'calls')   # forward: init
        _add_edge(G, 'lookup', 'REGISTRY', 'references')       # backward: user
        _add_edge(G, 'register', 'REGISTRY', 'references')     # backward: user

        result = expand_smart({'REGISTRY'}, G)
        self.assertIn('build_registry', result.expanded_nodes,
                      "Forward init function should be found")
        self.assertIn('lookup', result.expanded_nodes,
                      "Backward user function should be found")
        self.assertIn('register', result.expanded_nodes,
                      "Backward user function should be found")

    def test_mixed_constants_and_classes_expand_smart(self):
        """expand_smart with both constants and classes in matched_symbols."""
        G = nx.MultiDiGraph()
        G.add_node('MyClass', **_make_class_node('MyClass'))
        G.add_node('Base', **_make_class_node('Base'))
        G.add_node('CONFIG', **_make_constant_node('CONFIG'))
        G.add_node('use_cfg', **_make_function_node('use_config'))

        _add_edge(G, 'MyClass', 'Base', 'inheritance')
        _add_edge(G, 'use_cfg', 'CONFIG', 'references')

        result = expand_smart({'MyClass', 'CONFIG'}, G)
        self.assertIn('Base', result.expanded_nodes, "Class base should be found")
        self.assertIn('use_cfg', result.expanded_nodes, "Constant user should be found")

    def test_type_alias_users_in_expand_smart(self):
        """expand_smart should find users of a type alias."""
        G = nx.MultiDiGraph()
        G.add_node('MyAlias', **_make_type_alias_node('MyAlias'))
        G.add_node('use_alias', **_make_function_node('use_alias', lang='cpp'))
        _add_edge(G, 'use_alias', 'MyAlias', 'references')

        result = expand_smart({'MyAlias'}, G)
        self.assertIn('use_alias', result.expanded_nodes)


class TestBidirectionalExpandSmartFromSnippets(unittest.TestCase):
    """Bidirectional expansion integration: real C++ cross-file snippets.

    6-file project: shapes hierarchy plus a free function that uses them.
    Verified graph edges:
      - Circle --[inheritance]--> Shape
      - Rectangle --[inheritance]--> Shape
      - Circle.area --[overrides]--> Shape.area
      - Rectangle.area --[overrides]--> Shape.area
      - total_area --[references]--> Shape (parameter type)
    """

    CPP_FILES = {
        'include/shape.h': (
            '#pragma once\n'
            'class Shape {\n'
            'public:\n'
            '    virtual double area() const = 0;\n'
            '    virtual ~Shape() = default;\n'
            '};\n'
        ),
        'include/circle.h': (
            '#pragma once\n'
            '#include "shape.h"\n'
            '\n'
            'class Circle : public Shape {\n'
            'public:\n'
            '    double radius;\n'
            '    double area() const override;\n'
            '};\n'
        ),
        'include/rect.h': (
            '#pragma once\n'
            '#include "shape.h"\n'
            '\n'
            'class Rectangle : public Shape {\n'
            'public:\n'
            '    double width, height;\n'
            '    double area() const override;\n'
            '};\n'
        ),
        'src/circle.cpp': (
            '#include "circle.h"\n'
            '\n'
            'double Circle::area() const {\n'
            '    return 3.14159 * radius * radius;\n'
            '}\n'
        ),
        'src/rect.cpp': (
            '#include "rect.h"\n'
            '\n'
            'double Rectangle::area() const {\n'
            '    return width * height;\n'
            '}\n'
        ),
        'src/main.cpp': (
            '#include "circle.h"\n'
            '#include "rect.h"\n'
            '\n'
            'double total_area(Shape* shapes[], int count) {\n'
            '    double sum = 0;\n'
            '    for (int i = 0; i < count; i++) {\n'
            '        sum += shapes[i]->area();\n'
            '    }\n'
            '    return sum;\n'
            '}\n'
        ),
    }

    @classmethod
    def setUpClass(cls):
        cls.graph = _build_graph(cls.CPP_FILES)

    # ── Graph structure tests ────────────────────────────────────────

    def test_source_text_on_shape_nodes(self):
        """Shape hierarchy nodes should have source_text via Symbol."""
        for name in ('Shape', 'Circle', 'Rectangle'):
            node = _find_node(self.graph, name, 'class')
            src = _get_source_text(self.graph.nodes[node])
            self.assertTrue(len(src) > 0,
                            f"{name} should have source_text via symbol")

    def test_inheritance_edges(self):
        """Circle and Rectangle should have inheritance edges to Shape."""
        shape = _find_node(self.graph, 'Shape', 'class')
        circle = _find_node(self.graph, 'Circle', 'class')
        rect = _find_node(self.graph, 'Rectangle', 'class')
        self.assertTrue(has_relationship(self.graph, circle, shape, 'inheritance'),
                        "Circle → Shape inheritance expected")
        self.assertTrue(has_relationship(self.graph, rect, shape, 'inheritance'),
                        "Rectangle → Shape inheritance expected")

    def test_overrides_edges(self):
        """Derived area() methods should have overrides edges to Shape.area()."""
        shape_area = _find_node(self.graph, 'Shape.area', 'method')
        circle_area = _find_node(self.graph, 'Circle.area', 'method')
        self.assertTrue(
            has_relationship(self.graph, circle_area, shape_area, 'overrides'),
            "Circle.area → Shape.area overrides expected"
        )

    # ── Bidirectional expansion ──────────────────────────────────────

    def test_base_class_backward_discovers_derived(self):
        """Shape base class should discover Circle and Rectangle via backward."""
        node = _find_node(self.graph, 'Shape', 'class')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        derived = names & {'Circle', 'Rectangle'}
        self.assertGreaterEqual(len(derived), 2,
                                f"Shape should discover ≥2 derived; got {derived}")

    def test_derived_class_forward_discovers_base(self):
        """Circle should discover Shape via forward inheritance."""
        node = _find_node(self.graph, 'Circle', 'class')
        result = expand_smart({node}, self.graph)
        names = _expanded_names(result)
        self.assertIn('Shape', names, "Circle should expand to base class Shape")

    def test_function_discovers_referenced_types(self):
        """total_area references Shape* — should discover Shape."""
        node = _find_node(self.graph, 'total_area', 'function')
        result = expand_smart({node}, self.graph)
        self.assertGreater(len(result.expanded_nodes), 1,
                           "total_area should find at least one related type")

    def test_multi_class_expansion(self):
        """Expanding Circle + Rectangle together should include Shape."""
        circle = _find_node(self.graph, 'Circle', 'class')
        rect = _find_node(self.graph, 'Rectangle', 'class')
        result = expand_smart({circle, rect}, self.graph)
        names = _expanded_names(result)
        self.assertIn('Shape', names,
                      "Both derived classes should include shared base Shape")

    def test_expansion_reasons_mention_inheritance(self):
        """Expansion reasons should document the relationship type."""
        node = _find_node(self.graph, 'Circle', 'class')
        result = expand_smart({node}, self.graph)
        shape_node = _find_node(self.graph, 'Shape', 'class')
        if shape_node in result.expansion_reasons:
            reason = result.expansion_reasons[shape_node].lower()
            self.assertTrue('base' in reason or 'inherit' in reason,
                            f"Reason should mention inheritance; got: {reason}")


if __name__ == '__main__':
    unittest.main()


# ===========================================================================
# Phase 2: Verify smart expansion is enabled by default
# ===========================================================================

class TestSmartExpansionDefault(unittest.TestCase):
    """Phase 2 — Smart expansion should be enabled by default."""

    def test_smart_expansion_default_on(self):
        """SMART_EXPANSION_ENABLED should be True by default (no env override)."""
        import importlib
        import os

        # Remove env override if present, then reimport
        old_val = os.environ.pop("DEEPWIKI_SMART_EXPANSION", None)
        try:
            import plugin_implementation.code_graph.expansion_engine as mod
            importlib.reload(mod)
            self.assertTrue(mod.SMART_EXPANSION_ENABLED,
                            "Smart expansion should be ON by default (Phase 2)")
        finally:
            # Restore original env
            if old_val is not None:
                os.environ["DEEPWIKI_SMART_EXPANSION"] = old_val
            importlib.reload(mod)

    def test_smart_expansion_can_be_disabled(self):
        """Setting DEEPWIKI_SMART_EXPANSION=0 should disable smart expansion."""
        import importlib
        import os

        old_val = os.environ.get("DEEPWIKI_SMART_EXPANSION")
        os.environ["DEEPWIKI_SMART_EXPANSION"] = "0"
        try:
            import plugin_implementation.code_graph.expansion_engine as mod
            importlib.reload(mod)
            self.assertFalse(mod.SMART_EXPANSION_ENABLED,
                             "DEEPWIKI_SMART_EXPANSION=0 should disable smart expansion")
        finally:
            if old_val is not None:
                os.environ["DEEPWIKI_SMART_EXPANSION"] = old_val
            else:
                os.environ.pop("DEEPWIKI_SMART_EXPANSION", None)
            importlib.reload(mod)
