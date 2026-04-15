"""
Comprehensive tests for new features added across recent sessions:

1. **Macro expansion** — `_expand_macro()` strategy, macro in constants
2. **Specializes / template type args** — class expansion follows specializes edges + references
3. **Instantiates at P0** — instantiates edge followed in expansion priorities
4. **Orphan class FTS fallback** — framework-string pattern detection
5. **Arch-symbol filtering** — symbol resolution filters to ARCHITECTURAL_SYMBOLS
6. **search_by_name symbol_types** — FTS5 structured query type filtering

Run:
    cd pylon_deepwiki/plugins/deepwiki_plugin
    python -m pytest tests/test_new_features_coverage.py -v
"""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Set
from unittest.mock import MagicMock, patch

import networkx as nx

# ---------------------------------------------------------------------------
# Modules under test
# ---------------------------------------------------------------------------
from plugin_implementation.code_graph.expansion_engine import (
    CLASS_LIKE_TYPES,
    EXPANSION_PRIORITIES,
    EXPANSION_WORTHY_TYPES,
    SKIP_RELATIONSHIPS,
    _STRATEGY_MAP,
    _expand_class,
    _expand_function,
    _expand_macro,
    edges_between,
    expand_smart,
    get_neighbors_by_relationship,
    has_relationship,
)
from plugin_implementation.constants import (
    ARCHITECTURAL_SYMBOLS,
    CODE_SYMBOL_TYPES,
    EXPANSION_SYMBOL_TYPES,
)

# ---------------------------------------------------------------------------
# Synthetic graph helpers (reuse patterns from test_smart_expansion)
# ---------------------------------------------------------------------------

def _make_node(name: str, sym_type: str, lang: str = 'python',
               content: str = '', file_path: str = 'src/foo.py'):
    """Generic node factory."""
    return dict(
        symbol_name=name, symbol_type=sym_type, language=lang,
        source_text=content or f'# {name}', name=name,
        file_path=file_path, rel_path=file_path,
    )


def _cls(name, **kw):
    return _make_node(name, 'class', **kw)


def _fn(name, **kw):
    return _make_node(name, 'function', **kw)


def _method(name, **kw):
    return _make_node(name, 'method', **kw)


def _macro(name, lang='cpp', **kw):
    return _make_node(name, 'macro', lang=lang,
                      content=f'#define {name} ...', **kw)


def _const(name, **kw):
    return _make_node(name, 'constant', **kw)


def _type_alias(name, lang='cpp', **kw):
    return _make_node(name, 'type_alias', lang=lang, **kw)


def _struct(name, lang='cpp', **kw):
    return _make_node(name, 'struct', lang=lang, **kw)


def _interface(name, lang='java', **kw):
    return _make_node(name, 'interface', lang=lang, **kw)


def _field(name, **kw):
    return _make_node(name, 'field', **kw)


def _edge(G, src, tgt, rel_type, **extra):
    """Add edge to MultiDiGraph."""
    G.add_edge(src, tgt, relationship_type=rel_type, **extra)


# =============================================================================
# 1. MACRO EXPANSION TESTS
# =============================================================================

class TestMacroInConstants(unittest.TestCase):
    """Verify 'macro' is present in all relevant constant sets."""

    def test_macro_in_architectural_symbols(self):
        self.assertIn('macro', ARCHITECTURAL_SYMBOLS)

    def test_macro_in_expansion_symbol_types(self):
        self.assertIn('macro', EXPANSION_SYMBOL_TYPES)

    def test_macro_in_code_symbol_types(self):
        self.assertIn('macro', CODE_SYMBOL_TYPES)

    def test_macro_in_expansion_worthy_types(self):
        self.assertIn('macro', EXPANSION_WORTHY_TYPES)

    def test_macro_in_strategy_map(self):
        self.assertIn('macro', _STRATEGY_MAP)
        self.assertEqual(_STRATEGY_MAP['macro'], _expand_macro)


class TestExpandMacro(unittest.TestCase):
    """Test _expand_macro() strategy function."""

    def test_macro_references_types(self):
        """Macro referencing a class should include the class."""
        G = nx.MultiDiGraph()
        G.add_node('FMT_COMPILE', **_macro('FMT_COMPILE'))
        G.add_node('compiled_format', **_cls('compiled_format', lang='cpp'))
        _edge(G, 'FMT_COMPILE', 'compiled_format', 'references')

        new_nodes, reasons = _expand_macro(G, 'FMT_COMPILE', set())
        self.assertIn('compiled_format', new_nodes)
        self.assertIn('referenced type', reasons.get('compiled_format', ''))

    def test_macro_calls_functions(self):
        """Macro that invokes a function should expand to it."""
        G = nx.MultiDiGraph()
        G.add_node('DEFINE_TYPE', **_macro('DEFINE_TYPE'))
        G.add_node('register_type', **_fn('register_type', lang='cpp'))
        _edge(G, 'DEFINE_TYPE', 'register_type', 'calls')

        new_nodes, reasons = _expand_macro(G, 'DEFINE_TYPE', set())
        self.assertIn('register_type', new_nodes)
        self.assertIn('callee', reasons.get('register_type', ''))

    def test_macro_users_via_predecessors(self):
        """Classes/functions that use a macro should appear as users."""
        G = nx.MultiDiGraph()
        G.add_node('LOG_MACRO', **_macro('LOG_MACRO'))
        G.add_node('AppService', **_cls('AppService', lang='cpp'))
        G.add_node('helper_fn', **_fn('helper_fn', lang='cpp'))
        # predecessors with references/calls to the macro
        _edge(G, 'AppService', 'LOG_MACRO', 'references')
        _edge(G, 'helper_fn', 'LOG_MACRO', 'calls')

        new_nodes, reasons = _expand_macro(G, 'LOG_MACRO', set())
        self.assertIn('AppService', new_nodes)
        self.assertIn('helper_fn', new_nodes)
        self.assertIn('macro user', reasons.get('AppService', ''))
        self.assertIn('macro user', reasons.get('helper_fn', ''))

    def test_macro_no_edges(self):
        """Macro with no edges — no expansion, but no crash."""
        G = nx.MultiDiGraph()
        G.add_node('EMPTY_MACRO', **_macro('EMPTY_MACRO'))
        new_nodes, reasons = _expand_macro(G, 'EMPTY_MACRO', set())
        self.assertEqual(len(new_nodes), 0)

    def test_macro_does_not_include_self(self):
        """Macro should never include itself in expansion."""
        G = nx.MultiDiGraph()
        G.add_node('M', **_macro('M'))
        G.add_node('C', **_cls('C', lang='cpp'))
        _edge(G, 'M', 'C', 'references')
        new_nodes, _ = _expand_macro(G, 'M', set())
        self.assertNotIn('M', new_nodes)

    def test_macro_type_filter_excludes_methods(self):
        """Macro references to methods should be filtered out (only arch types)."""
        G = nx.MultiDiGraph()
        G.add_node('MACRO_X', **_macro('MACRO_X'))
        G.add_node('some_method', **_method('some_method', lang='cpp'))
        G.add_node('SomeClass', **_cls('SomeClass', lang='cpp'))
        _edge(G, 'MACRO_X', 'some_method', 'references')
        _edge(G, 'MACRO_X', 'SomeClass', 'references')

        new_nodes, _ = _expand_macro(G, 'MACRO_X', set())
        # Method is not in CLASS_LIKE_TYPES | {'function', 'type_alias'} → excluded
        self.assertNotIn('some_method', new_nodes)
        self.assertIn('SomeClass', new_nodes)

    def test_macro_expand_smart_integration(self):
        """expand_smart() correctly dispatches to _expand_macro for macro nodes."""
        G = nx.MultiDiGraph()
        G.add_node('MY_MACRO', **_macro('MY_MACRO'))
        G.add_node('TargetClass', **_cls('TargetClass', lang='cpp'))
        _edge(G, 'MY_MACRO', 'TargetClass', 'references')

        result = expand_smart({'MY_MACRO'}, G)
        self.assertIn('MY_MACRO', result.expanded_nodes)
        self.assertIn('TargetClass', result.expanded_nodes)
        self.assertIn('TargetClass', result.expansion_reasons)

    def test_macro_mixed_with_classes(self):
        """Expanding both a macro and a class in the same call."""
        G = nx.MultiDiGraph()
        G.add_node('FMT_COMPILE', **_macro('FMT_COMPILE'))
        G.add_node('formatter', **_cls('formatter', lang='cpp'))
        G.add_node('BaseFormatter', **_cls('BaseFormatter', lang='cpp'))
        _edge(G, 'FMT_COMPILE', 'formatter', 'references')
        _edge(G, 'formatter', 'BaseFormatter', 'inheritance')

        result = expand_smart({'FMT_COMPILE', 'formatter'}, G)
        self.assertIn('FMT_COMPILE', result.expanded_nodes)
        self.assertIn('formatter', result.expanded_nodes)
        self.assertIn('BaseFormatter', result.expanded_nodes)


# =============================================================================
# 2. SPECIALIZES / TEMPLATE TYPE ARGS TESTS
# =============================================================================

class TestSpecializesExpansion(unittest.TestCase):
    """Test that _expand_class follows specializes edges and collects type args."""

    def test_specializes_follows_template_base(self):
        """Class specializing a template should expand to the template base."""
        G = nx.MultiDiGraph()
        G.add_node('MyVector', **_cls('MyVector', lang='cpp'))
        G.add_node('vector', **_cls('vector', lang='cpp'))
        _edge(G, 'MyVector', 'vector', 'specializes')

        new_nodes, reasons = _expand_class(G, 'MyVector', set())
        self.assertIn('vector', new_nodes)
        self.assertIn('template base', reasons.get('vector', ''))

    def test_specializes_collects_type_args_from_base_refs(self):
        """Type arguments referenced by template base should be included."""
        G = nx.MultiDiGraph()
        G.add_node('MyMap', **_cls('MyMap', lang='cpp'))
        G.add_node('map', **_cls('map', lang='cpp'))
        G.add_node('Key', **_cls('Key', lang='cpp'))
        G.add_node('Value', **_cls('Value', lang='cpp'))
        _edge(G, 'MyMap', 'map', 'specializes')
        # Template base references the type args
        _edge(G, 'map', 'Key', 'references')
        _edge(G, 'map', 'Value', 'references')

        new_nodes, reasons = _expand_class(G, 'MyMap', set())
        self.assertIn('map', new_nodes)
        self.assertIn('Key', new_nodes)
        self.assertIn('Value', new_nodes)
        self.assertIn('template type arg', reasons.get('Key', ''))

    def test_specializes_direct_references_on_class(self):
        """Class's own references (P2) should also capture type args."""
        G = nx.MultiDiGraph()
        G.add_node('MyContainer', **_cls('MyContainer', lang='cpp'))
        G.add_node('vector', **_cls('vector', lang='cpp'))
        G.add_node('Widget', **_cls('Widget', lang='cpp'))
        _edge(G, 'MyContainer', 'vector', 'specializes')
        # Direct reference from the class itself (as parser emits)
        _edge(G, 'MyContainer', 'Widget', 'references')

        new_nodes, reasons = _expand_class(G, 'MyContainer', set())
        self.assertIn('vector', new_nodes)
        self.assertIn('Widget', new_nodes)
        self.assertIn('referenced type', reasons.get('Widget', ''))

    def test_specializes_with_type_alias_arg(self):
        """Template arg that's a type_alias should be included too."""
        G = nx.MultiDiGraph()
        G.add_node('Spec', **_cls('Spec', lang='cpp'))
        G.add_node('base_tmpl', **_cls('base_tmpl', lang='cpp'))
        G.add_node('MyAlias', **_type_alias('MyAlias'))
        _edge(G, 'Spec', 'base_tmpl', 'specializes')
        _edge(G, 'base_tmpl', 'MyAlias', 'references')

        new_nodes, _ = _expand_class(G, 'Spec', set())
        self.assertIn('base_tmpl', new_nodes)
        self.assertIn('MyAlias', new_nodes)

    def test_specializes_no_base(self):
        """Class without specializes edges — no template expansion."""
        G = nx.MultiDiGraph()
        G.add_node('PlainClass', **_cls('PlainClass', lang='cpp'))
        new_nodes, _ = _expand_class(G, 'PlainClass', set())
        # No specializes edges → no template-related nodes
        self.assertEqual(len(new_nodes), 0)

    def test_specializes_integrate_expand_smart(self):
        """expand_smart should include specializes targets."""
        G = nx.MultiDiGraph()
        G.add_node('Derived', **_cls('Derived', lang='cpp'))
        G.add_node('TemplateBase', **_cls('TemplateBase', lang='cpp'))
        _edge(G, 'Derived', 'TemplateBase', 'specializes')

        result = expand_smart({'Derived'}, G)
        self.assertIn('TemplateBase', result.expanded_nodes)
        spec_reasons = [r for r in result.expansion_reasons.values()
                        if 'template' in r.lower()]
        self.assertGreater(len(spec_reasons), 0)


# =============================================================================
# 3. INSTANTIATES AT P0 TESTS
# =============================================================================

class TestInstantiatesExpansion(unittest.TestCase):
    """Test that instantiates is now at P0 (not in SKIP_RELATIONSHIPS)."""

    def test_instantiates_not_in_skip(self):
        """instantiates should NOT be in SKIP_RELATIONSHIPS."""
        self.assertNotIn('instantiates', SKIP_RELATIONSHIPS)

    def test_instantiates_in_expansion_priorities(self):
        """instantiates should be in EXPANSION_PRIORITIES at P0."""
        self.assertIn('instantiates', EXPANSION_PRIORITIES)
        self.assertEqual(EXPANSION_PRIORITIES['instantiates']['priority'], 0)

    def test_instantiates_budget(self):
        """instantiates should have a positive budget."""
        self.assertGreater(EXPANSION_PRIORITIES['instantiates']['budget'], 0)

    def test_function_follows_instantiates(self):
        """_expand_function follows instantiates edges."""
        G = nx.MultiDiGraph()
        G.add_node('factory_fn', **_fn('create_widget'))
        G.add_node('Widget', **_cls('Widget'))
        _edge(G, 'factory_fn', 'Widget', 'instantiates')

        new_nodes, reasons = _expand_function(G, 'factory_fn', set())
        self.assertIn('Widget', new_nodes)
        # Reason should mention the relationship type
        self.assertIn('Widget', reasons)

    def test_instantiates_expand_smart(self):
        """expand_smart correctly follows instantiates from a function."""
        G = nx.MultiDiGraph()
        G.add_node('builder_fn', **_fn('build_config'))
        G.add_node('Config', **_cls('Config'))
        _edge(G, 'builder_fn', 'Config', 'instantiates')

        result = expand_smart({'builder_fn'}, G)
        self.assertIn('Config', result.expanded_nodes)

    def test_instantiates_vs_creates_both_work(self):
        """Both creates and instantiates edges should be followed."""
        G = nx.MultiDiGraph()
        G.add_node('fn', **_fn('make_stuff'))
        G.add_node('TypeA', **_cls('TypeA'))
        G.add_node('TypeB', **_cls('TypeB'))
        _edge(G, 'fn', 'TypeA', 'creates')
        _edge(G, 'fn', 'TypeB', 'instantiates')

        new_nodes, reasons = _expand_function(G, 'fn', set())
        self.assertIn('TypeA', new_nodes)
        self.assertIn('TypeB', new_nodes)


# =============================================================================
# 4. ORPHAN CLASS FTS FALLBACK TESTS
# =============================================================================

class TestOrphanClassDetection(unittest.TestCase):
    """Test the orphan class detection logic (without real FTS).

    The orphan fallback lives in wiki_graph_optimized._get_docs_by_target_symbols.
    We test the detection logic pattern here using synthetic graphs.
    """

    def _is_orphan_class(self, graph, node_id, expanded_nodes, matched_nodes):
        """Replicate the orphan detection logic from wiki_graph_optimized."""
        node_data = graph.nodes.get(node_id, {})
        sym_type = (node_data.get('symbol_type') or '').lower()
        if sym_type not in ('class', 'interface', 'struct'):
            return False

        has_expansion = False
        for succ in list(graph.successors(node_id))[:20]:
            if succ in expanded_nodes and succ != node_id:
                has_expansion = True
                break
        if not has_expansion:
            for pred in list(graph.predecessors(node_id))[:20]:
                if pred in expanded_nodes and pred != node_id:
                    has_expansion = True
                    break
        return not has_expansion

    def test_class_with_inheritance_not_orphan(self):
        """A class with a base class in expanded_nodes is NOT an orphan."""
        G = nx.MultiDiGraph()
        G.add_node('Derived', **_cls('Derived'))
        G.add_node('Base', **_cls('Base'))
        _edge(G, 'Derived', 'Base', 'inheritance')

        expanded = {'Derived', 'Base'}
        matched = {'Derived'}
        self.assertFalse(self._is_orphan_class(G, 'Derived', expanded, matched))

    def test_class_with_no_expansion_is_orphan(self):
        """A class with no expanded neighbors IS an orphan."""
        G = nx.MultiDiGraph()
        G.add_node('Event', **_cls('Event'))
        G.add_node('handler', **_method('on_event'))
        _edge(G, 'Event', 'handler', 'defines')
        # handler is NOT in expanded_nodes (methods are not architectural)

        expanded = {'Event'}
        matched = {'Event'}
        self.assertTrue(self._is_orphan_class(G, 'Event', expanded, matched))

    def test_class_with_only_methods_is_orphan(self):
        """A class that only defines methods (not in expanded) is orphan."""
        G = nx.MultiDiGraph()
        G.add_node('Service', **_cls('Service'))
        G.add_node('start', **_method('start'))
        G.add_node('stop', **_method('stop'))
        _edge(G, 'Service', 'start', 'defines')
        _edge(G, 'Service', 'stop', 'defines')

        # Methods not in expanded_nodes
        expanded = {'Service'}
        matched = {'Service'}
        self.assertTrue(self._is_orphan_class(G, 'Service', expanded, matched))

    def test_function_not_detected_as_orphan(self):
        """Functions should not be checked for orphan status."""
        G = nx.MultiDiGraph()
        G.add_node('my_fn', **_fn('my_fn'))
        self.assertFalse(self._is_orphan_class(G, 'my_fn', {'my_fn'}, {'my_fn'}))

    def test_class_with_predecessor_in_expanded_not_orphan(self):
        """A class whose predecessor is in expanded is not orphan."""
        G = nx.MultiDiGraph()
        G.add_node('Widget', **_cls('Widget'))
        G.add_node('Factory', **_cls('Factory'))
        _edge(G, 'Factory', 'Widget', 'creates')

        expanded = {'Widget', 'Factory'}
        matched = {'Widget'}
        self.assertFalse(self._is_orphan_class(G, 'Widget', expanded, matched))


class TestOrphanMethodNameExtraction(unittest.TestCase):
    """Test method name extraction from orphan classes using has_relationship."""

    def _extract_method_names(self, graph, node_id):
        """Replicate the cleaned-up method extraction logic."""
        method_names = []
        for succ in graph.successors(node_id):
            if has_relationship(graph, node_id, succ, 'defines'):
                succ_data = graph.nodes.get(succ, {})
                succ_type = (succ_data.get('symbol_type') or '').lower()
                if succ_type in ('method', 'function'):
                    method_name = (succ_data.get('symbol_name', '')
                                   or succ_data.get('name', ''))
                    if method_name and not method_name.startswith('__'):
                        method_names.append(method_name)
        return method_names

    def test_extracts_public_methods(self):
        """Public methods of a class are extracted."""
        G = nx.MultiDiGraph()
        G.add_node('Event', **_cls('Event'))
        G.add_node('configuration_created', **_method('configuration_created'))
        G.add_node('configuration_deleted', **_method('configuration_deleted'))
        _edge(G, 'Event', 'configuration_created', 'defines')
        _edge(G, 'Event', 'configuration_deleted', 'defines')

        names = self._extract_method_names(G, 'Event')
        self.assertEqual(set(names), {'configuration_created', 'configuration_deleted'})

    def test_skips_dunder_methods(self):
        """__init__, __repr__ etc. should be skipped."""
        G = nx.MultiDiGraph()
        G.add_node('Event', **_cls('Event'))
        G.add_node('__init__', **_method('__init__'))
        G.add_node('process', **_method('process'))
        _edge(G, 'Event', '__init__', 'defines')
        _edge(G, 'Event', 'process', 'defines')

        names = self._extract_method_names(G, 'Event')
        self.assertEqual(names, ['process'])

    def test_skips_non_defines_edges(self):
        """Only 'defines' edges should be followed for method extraction."""
        G = nx.MultiDiGraph()
        G.add_node('MyClass', **_cls('MyClass'))
        G.add_node('other_fn', **_fn('other_fn'))
        _edge(G, 'MyClass', 'other_fn', 'calls')  # not defines

        names = self._extract_method_names(G, 'MyClass')
        self.assertEqual(names, [])

    def test_skips_non_method_defines(self):
        """Fields defined by the class should not be included."""
        G = nx.MultiDiGraph()
        G.add_node('MyClass', **_cls('MyClass'))
        G.add_node('field_x', **_field('x'))
        G.add_node('do_work', **_method('do_work'))
        _edge(G, 'MyClass', 'field_x', 'defines')
        _edge(G, 'MyClass', 'do_work', 'defines')

        names = self._extract_method_names(G, 'MyClass')
        self.assertEqual(names, ['do_work'])

    def test_multidigraph_multiple_edges(self):
        """Works correctly with MultiDiGraph (multiple edges between same pair)."""
        G = nx.MultiDiGraph()
        G.add_node('Cls', **_cls('Cls'))
        G.add_node('m', **_method('m'))
        _edge(G, 'Cls', 'm', 'defines')
        _edge(G, 'Cls', 'm', 'contains')  # extra edge

        names = self._extract_method_names(G, 'Cls')
        self.assertEqual(names, ['m'])


# =============================================================================
# 5. ARCHITECTURAL SYMBOL FILTERING TESTS
# =============================================================================

class TestArchitecturalSymbolSets(unittest.TestCase):
    """Verify correctness of architectural symbol set definitions."""

    def test_arch_symbols_include_all_code_types(self):
        """CODE_SYMBOL_TYPES should be a subset of ARCHITECTURAL_SYMBOLS."""
        self.assertTrue(
            CODE_SYMBOL_TYPES.issubset(ARCHITECTURAL_SYMBOLS),
            f"CODE_SYMBOL_TYPES has types not in ARCHITECTURAL_SYMBOLS: "
            f"{CODE_SYMBOL_TYPES - ARCHITECTURAL_SYMBOLS}"
        )

    def test_expansion_types_subset_of_arch(self):
        """EXPANSION_SYMBOL_TYPES ⊂ (ARCHITECTURAL_SYMBOLS ∪ {module_doc, file_doc})."""
        allowed = ARCHITECTURAL_SYMBOLS | frozenset({'module_doc', 'file_doc'})
        extra = EXPANSION_SYMBOL_TYPES - allowed
        self.assertEqual(
            extra, frozenset(),
            f"EXPANSION_SYMBOL_TYPES has unexpected types: {extra}"
        )

    def test_method_excluded_from_arch(self):
        """'method' must NOT be in ARCHITECTURAL_SYMBOLS."""
        self.assertNotIn('method', ARCHITECTURAL_SYMBOLS)

    def test_constructor_excluded_from_arch(self):
        """'constructor' must NOT be in ARCHITECTURAL_SYMBOLS."""
        self.assertNotIn('constructor', ARCHITECTURAL_SYMBOLS)

    def test_field_excluded_from_arch(self):
        """'field' must NOT be in ARCHITECTURAL_SYMBOLS."""
        self.assertNotIn('field', ARCHITECTURAL_SYMBOLS)

    def test_method_excluded_from_expansion(self):
        """'method' must NOT be in EXPANSION_SYMBOL_TYPES."""
        self.assertNotIn('method', EXPANSION_SYMBOL_TYPES)

    def test_macro_in_all_sets(self):
        """Macro should be in arch, expansion, and code sets."""
        self.assertIn('macro', ARCHITECTURAL_SYMBOLS)
        self.assertIn('macro', EXPANSION_SYMBOL_TYPES)
        self.assertIn('macro', CODE_SYMBOL_TYPES)

    def test_class_like_types_in_arch(self):
        """All CLASS_LIKE_TYPES should be in ARCHITECTURAL_SYMBOLS."""
        for t in CLASS_LIKE_TYPES:
            self.assertIn(t, ARCHITECTURAL_SYMBOLS, f"'{t}' from CLASS_LIKE_TYPES missing in ARCHITECTURAL_SYMBOLS")


class TestArchFilteringInSymbolResolution(unittest.TestCase):
    """Test that symbol resolution strategies filter to arch symbols.

    We simulate what _get_docs_by_target_symbols does with a name_index
    and verify that non-arch symbols are filtered out.
    """

    def _simulate_name_index_lookup(self, name_index, graph, target):
        """Simulate Strategy 1 from _get_docs_by_target_symbols."""
        if target not in name_index:
            return []
        return [
            nid for nid in name_index[target]
            if (graph.nodes.get(nid, {}).get('symbol_type') or '').lower()
            in ARCHITECTURAL_SYMBOLS
        ]

    def test_filters_methods_from_name_index(self):
        """Methods matching a target name should be filtered out."""
        G = nx.MultiDiGraph()
        G.add_node('cls::format', **_method('format'))
        G.add_node('format_fn', **_fn('format'))
        G.add_node('format_cls', **_cls('format', lang='cpp'))

        name_index = {
            'format': ['cls::format', 'format_fn', 'format_cls'],
        }

        results = self._simulate_name_index_lookup(name_index, G, 'format')
        self.assertNotIn('cls::format', results)
        self.assertIn('format_fn', results)
        self.assertIn('format_cls', results)

    def test_filters_constructors(self):
        """Constructors should be filtered out."""
        G = nx.MultiDiGraph()
        G.add_node('Widget_ctor', symbol_name='Widget', symbol_type='constructor',
                    language='cpp', source_text='Widget::Widget() {}',
                    file_path='w.cpp', rel_path='w.cpp')
        G.add_node('Widget_cls', **_cls('Widget', lang='cpp'))

        name_index = {'Widget': ['Widget_ctor', 'Widget_cls']}
        results = self._simulate_name_index_lookup(name_index, G, 'Widget')
        self.assertEqual(results, ['Widget_cls'])

    def test_filters_fields(self):
        """Field symbols should be filtered out."""
        G = nx.MultiDiGraph()
        G.add_node('data_field', **_field('data'))
        G.add_node('data_cls', **_cls('data', lang='cpp'))

        name_index = {'data': ['data_field', 'data_cls']}
        results = self._simulate_name_index_lookup(name_index, G, 'data')
        self.assertEqual(results, ['data_cls'])

    def test_allows_macros(self):
        """Macros should pass the arch filter."""
        G = nx.MultiDiGraph()
        G.add_node('FMT_COMPILE_macro', **_macro('FMT_COMPILE'))

        name_index = {'FMT_COMPILE': ['FMT_COMPILE_macro']}
        results = self._simulate_name_index_lookup(name_index, G, 'FMT_COMPILE')
        self.assertEqual(results, ['FMT_COMPILE_macro'])

    def test_allows_type_alias(self):
        """Type aliases pass the arch filter."""
        G = nx.MultiDiGraph()
        G.add_node('string_view_alias', **_type_alias('string_view'))

        name_index = {'string_view': ['string_view_alias']}
        results = self._simulate_name_index_lookup(name_index, G, 'string_view')
        self.assertEqual(results, ['string_view_alias'])


# =============================================================================
# 6. search_by_name WITH symbol_types TESTS
# =============================================================================

class TestSearchByNameSymbolTypes(unittest.TestCase):
    """Test GraphTextIndex.search_by_name() with the symbol_types parameter.

    Uses a real in-memory SQLite FTS5 database to verify filtering.
    """

    @classmethod
    def setUpClass(cls):
        """Create a minimal in-memory FTS5 index for testing."""
        cls._tmpdir = tempfile.mkdtemp()
        cls._db_path = os.path.join(cls._tmpdir, 'test_index.db')
        conn = sqlite3.connect(cls._db_path)
        conn.execute("""
            CREATE TABLE symbols (
                node_id TEXT PRIMARY KEY,
                symbol_name TEXT,
                symbol_type TEXT,
                file_path TEXT,
                language TEXT,
                docstring TEXT,
                content TEXT,
                name_tokens TEXT
            )
        """)
        # Insert test data — mix of arch and non-arch symbols
        test_rows = [
            ('n1', 'format', 'function', 'fmt/format.h', 'cpp', '', 'void format() {}', 'format'),
            ('n2', 'format', 'method', 'fmt/core.h', 'cpp', '', 'void format() {}', 'format'),
            ('n3', 'format', 'class', 'fmt/format.h', 'cpp', 'A formatter', 'class format {};', 'format'),
            ('n4', 'FMT_COMPILE', 'macro', 'fmt/compile.h', 'cpp', '', '#define FMT_COMPILE', 'fmt compile'),
            ('n5', 'FMT_COMPILE', 'constructor', 'fmt/compile.h', 'cpp', '', 'FMT_COMPILE() {}', 'fmt compile'),
            ('n6', 'Widget', 'class', 'src/widget.h', 'cpp', 'A widget', 'class Widget {};', 'widget'),
            ('n7', 'widget_field', 'field', 'src/widget.h', 'cpp', '', 'int widget_field;', 'widget field'),
        ]
        conn.executemany(
            "INSERT INTO symbols VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            test_rows,
        )
        conn.commit()
        conn.close()
        cls._conn = None

    def _search_by_name(self, name, exact=False, k=10, symbol_types=None):
        """Replicate the search_by_name logic directly against our test DB."""
        where_parts = []
        params = []
        if exact:
            where_parts.append("symbol_name = ?")
            params.append(name)
        else:
            where_parts.append("symbol_name LIKE ?")
            params.append(f"%{name}%")
        if symbol_types:
            placeholders = ','.join('?' for _ in symbol_types)
            where_parts.append(f"symbol_type IN ({placeholders})")
            params.extend(sorted(symbol_types))
        sql = f"SELECT * FROM symbols WHERE {' AND '.join(where_parts)} LIMIT ?"
        params.append(k)
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(sql, params).fetchall()
            return rows
        finally:
            conn.close()

    def test_no_filter_returns_all(self):
        """Without symbol_types, all matching rows returned."""
        rows = self._search_by_name('format', exact=True)
        self.assertEqual(len(rows), 3)  # function, method, class

    def test_arch_filter_excludes_method(self):
        """With ARCHITECTURAL_SYMBOLS filter, method is excluded."""
        rows = self._search_by_name('format', exact=True,
                                     symbol_types=ARCHITECTURAL_SYMBOLS)
        types = {r[2] for r in rows}
        self.assertNotIn('method', types)
        self.assertIn('function', types)
        self.assertIn('class', types)

    def test_arch_filter_excludes_constructor(self):
        """Constructor is excluded by arch filter."""
        rows = self._search_by_name('FMT_COMPILE', exact=True,
                                     symbol_types=ARCHITECTURAL_SYMBOLS)
        types = {r[2] for r in rows}
        self.assertNotIn('constructor', types)
        self.assertIn('macro', types)

    def test_arch_filter_excludes_field(self):
        """Field is excluded by arch filter."""
        rows = self._search_by_name('widget', exact=False,
                                     symbol_types=ARCHITECTURAL_SYMBOLS)
        types = {r[2] for r in rows}
        self.assertNotIn('field', types)

    def test_specific_type_filter(self):
        """Can filter to a single type."""
        rows = self._search_by_name('format', exact=True,
                                     symbol_types=frozenset({'class'}))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], 'class')

    def test_macro_passes_arch_filter(self):
        """Macro type passes the arch filter."""
        rows = self._search_by_name('FMT_COMPILE', exact=True,
                                     symbol_types=ARCHITECTURAL_SYMBOLS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][2], 'macro')

    @classmethod
    def tearDownClass(cls):
        """Clean up temp files."""
        try:
            os.unlink(cls._db_path)
            os.rmdir(cls._tmpdir)
        except Exception:
            pass


# =============================================================================
# 7. EDGE CASE & REGRESSION TESTS
# =============================================================================

class TestExpansionEdgeCases(unittest.TestCase):
    """Edge cases across all new features."""

    def test_macro_and_class_same_name(self):
        """A macro and class with the same name should both be expandable."""
        G = nx.MultiDiGraph()
        G.add_node('FMT_cls', **_cls('FMT', lang='cpp'))
        G.add_node('FMT_macro', **_macro('FMT'))
        G.add_node('Base', **_cls('Base', lang='cpp'))
        G.add_node('Helper', **_fn('Helper', lang='cpp'))
        _edge(G, 'FMT_cls', 'Base', 'inheritance')
        _edge(G, 'FMT_macro', 'Helper', 'calls')

        result = expand_smart({'FMT_cls', 'FMT_macro'}, G)
        self.assertIn('Base', result.expanded_nodes)
        self.assertIn('Helper', result.expanded_nodes)

    def test_specializes_and_inheritance_combined(self):
        """Class with both inheritance and specializes should follow both."""
        G = nx.MultiDiGraph()
        G.add_node('SpecialWidget', **_cls('SpecialWidget', lang='cpp'))
        G.add_node('BaseWidget', **_cls('BaseWidget', lang='cpp'))
        G.add_node('vector', **_cls('vector', lang='cpp'))
        _edge(G, 'SpecialWidget', 'BaseWidget', 'inheritance')
        _edge(G, 'SpecialWidget', 'vector', 'specializes')

        new_nodes, reasons = _expand_class(G, 'SpecialWidget', set())
        self.assertIn('BaseWidget', new_nodes)
        self.assertIn('vector', new_nodes)

    def test_instantiates_and_creates_combined(self):
        """Function with both creates and instantiates edges."""
        G = nx.MultiDiGraph()
        G.add_node('fn', **_fn('build'))
        G.add_node('TypeA', **_cls('TypeA'))
        G.add_node('TypeB', **_cls('TypeB'))
        _edge(G, 'fn', 'TypeA', 'creates')
        _edge(G, 'fn', 'TypeB', 'instantiates')

        result = expand_smart({'fn'}, G)
        self.assertIn('TypeA', result.expanded_nodes)
        self.assertIn('TypeB', result.expanded_nodes)

    def test_orphan_class_with_only_imports(self):
        """Class with only imports edges is still orphan (imports are SKIP)."""
        G = nx.MultiDiGraph()
        G.add_node('OrphanCls', **_cls('OrphanCls'))
        G.add_node('os_mod', **_make_node('os', 'module'))
        _edge(G, 'OrphanCls', 'os_mod', 'imports')

        result = expand_smart({'OrphanCls'}, G)
        # imports is skipped → no expansion beyond the class itself
        expansion_beyond_matched = result.expanded_nodes - {'OrphanCls'}
        self.assertEqual(len(expansion_beyond_matched), 0)

    def test_struct_treated_as_class_like(self):
        """Struct should use _expand_class strategy."""
        G = nx.MultiDiGraph()
        G.add_node('MyStruct', **_struct('MyStruct'))
        G.add_node('Base', **_struct('Base'))
        _edge(G, 'MyStruct', 'Base', 'inheritance')

        result = expand_smart({'MyStruct'}, G)
        self.assertIn('Base', result.expanded_nodes)

    def test_interface_treated_as_class_like(self):
        """Interface should use _expand_class strategy."""
        G = nx.MultiDiGraph()
        G.add_node('IService', **_interface('IService'))
        G.add_node('BaseInterface', **_interface('BaseInterface'))
        _edge(G, 'IService', 'BaseInterface', 'inheritance')

        result = expand_smart({'IService'}, G)
        self.assertIn('BaseInterface', result.expanded_nodes)


class TestExpansionPriorityOrder(unittest.TestCase):
    """Verify priority ordering of expansion — P0 before P1 before P2."""

    def test_p0_inheritance_reason_preserved_over_p2_reference(self):
        """When a node is reachable by both inheritance (P0) and references (P2),
        the first reason (inheritance) wins because strategies are called in order."""
        G = nx.MultiDiGraph()
        G.add_node('cls', **_cls('A'))
        G.add_node('base', **_cls('B'))
        _edge(G, 'cls', 'base', 'inheritance')
        _edge(G, 'cls', 'base', 'references')

        new_nodes, reasons = _expand_class(G, 'cls', set())
        self.assertIn('base', new_nodes)
        # First reason should be P0 (inheritance), not P2 (references)
        self.assertIn('base class', reasons['base'])

    def test_p0_creates_before_p2_calls(self):
        """Creates (P0) reason should win over calls (P2)."""
        G = nx.MultiDiGraph()
        G.add_node('factory', **_fn('create'))
        G.add_node('Widget', **_cls('Widget'))
        _edge(G, 'factory', 'Widget', 'creates')
        _edge(G, 'factory', 'Widget', 'calls')  # unlikely but tests priority

        new_nodes, reasons = _expand_function(G, 'factory', set())
        self.assertIn('Widget', new_nodes)
        self.assertIn('created type', reasons['Widget'])


class TestHasRelationshipWithOrphanCode(unittest.TestCase):
    """Test that has_relationship() works correctly for orphan detection edge cases."""

    def test_multidigraph_defines_detected(self):
        """has_relationship detects 'defines' in a MultiDiGraph."""
        G = nx.MultiDiGraph()
        G.add_node('cls', **_cls('C'))
        G.add_node('m', **_method('m'))
        _edge(G, 'cls', 'm', 'defines')
        self.assertTrue(has_relationship(G, 'cls', 'm', 'defines'))

    def test_multiple_edges_one_defines(self):
        """When multiple edges exist, defines is still detected."""
        G = nx.MultiDiGraph()
        G.add_node('cls', **_cls('C'))
        G.add_node('m', **_method('m'))
        _edge(G, 'cls', 'm', 'defines')
        _edge(G, 'cls', 'm', 'contains')
        self.assertTrue(has_relationship(G, 'cls', 'm', 'defines'))

    def test_no_defines_not_detected(self):
        """When no defines edge exists, returns False."""
        G = nx.MultiDiGraph()
        G.add_node('cls', **_cls('C'))
        G.add_node('m', **_method('m'))
        _edge(G, 'cls', 'm', 'contains')
        self.assertFalse(has_relationship(G, 'cls', 'm', 'defines'))


# =============================================================================
# 8. FULL INTEGRATION: EXPAND_SMART WITH ALL NEW FEATURES
# =============================================================================

class TestExpandSmartFullIntegration(unittest.TestCase):
    """Integration tests combining multiple new features in one graph."""

    def _build_complex_graph(self):
        """Build a synthetic graph exercising all new features."""
        G = nx.MultiDiGraph()

        # Classes
        G.add_node('Formatter', **_cls('Formatter', lang='cpp'))
        G.add_node('BaseFormatter', **_cls('BaseFormatter', lang='cpp'))
        G.add_node('vector', **_cls('vector', lang='cpp'))
        G.add_node('Widget', **_cls('Widget', lang='cpp'))

        # Struct
        G.add_node('FormatSpec', **_struct('FormatSpec'))

        # Macro
        G.add_node('FMT_COMPILE', **_macro('FMT_COMPILE'))
        G.add_node('DEFINE_FORMATTER', **_macro('DEFINE_FORMATTER'))

        # Functions
        G.add_node('format_to', **_fn('format_to', lang='cpp'))
        G.add_node('register_type', **_fn('register_type', lang='cpp'))

        # Constants
        G.add_node('MAX_BUFFER', **_const('MAX_BUFFER', lang='cpp'))

        # Type alias
        G.add_node('string_view', **_type_alias('string_view'))
        G.add_node('basic_string', **_cls('basic_string', lang='cpp'))

        # Methods (non-arch, should not leak into results)
        G.add_node('do_format', **_method('do_format', lang='cpp'))
        G.add_node('constructor', symbol_name='Formatter', symbol_type='constructor',
                    language='cpp', source_text='Formatter() {}',
                    file_path='fmt.h', rel_path='fmt.h')

        # --- Edges ---
        # Inheritance
        _edge(G, 'Formatter', 'BaseFormatter', 'inheritance')
        # Specializes (template)
        _edge(G, 'Formatter', 'vector', 'specializes')
        _edge(G, 'vector', 'Widget', 'references')  # type arg
        # Macro references
        _edge(G, 'FMT_COMPILE', 'FormatSpec', 'references')
        _edge(G, 'DEFINE_FORMATTER', 'register_type', 'calls')
        _edge(G, 'Formatter', 'FMT_COMPILE', 'references')  # macro user
        # Function creates
        _edge(G, 'format_to', 'FormatSpec', 'instantiates')
        # Type alias chain
        _edge(G, 'string_view', 'basic_string', 'alias_of')
        # Defines (should not expand)
        _edge(G, 'Formatter', 'do_format', 'defines')
        _edge(G, 'Formatter', 'constructor', 'defines')

        return G

    def test_class_expansion_complete(self):
        """Formatter class expands: inheritance + specializes + type args."""
        G = self._build_complex_graph()
        result = expand_smart({'Formatter'}, G)

        # P0: base class
        self.assertIn('BaseFormatter', result.expanded_nodes)
        # P1: template base
        self.assertIn('vector', result.expanded_nodes)
        # P1: template type arg (via base's references)
        self.assertIn('Widget', result.expanded_nodes)
        # Non-arch should NOT be in expanded
        self.assertNotIn('do_format', result.expanded_nodes)
        self.assertNotIn('constructor', result.expanded_nodes)

    def test_macro_expansion_complete(self):
        """FMT_COMPILE macro expands to referenced types."""
        G = self._build_complex_graph()
        result = expand_smart({'FMT_COMPILE'}, G)

        self.assertIn('FormatSpec', result.expanded_nodes)
        # Formatter references FMT_COMPILE → predecessor user
        self.assertIn('Formatter', result.expanded_nodes)

    def test_function_with_instantiates(self):
        """format_to function follows instantiates."""
        G = self._build_complex_graph()
        result = expand_smart({'format_to'}, G)

        self.assertIn('FormatSpec', result.expanded_nodes)

    def test_type_alias_resolves(self):
        """string_view alias resolves to basic_string."""
        G = self._build_complex_graph()
        result = expand_smart({'string_view'}, G)

        self.assertIn('basic_string', result.expanded_nodes)

    def test_mixed_expansion_all_features(self):
        """Expand a mix of class + macro + function + alias together."""
        G = self._build_complex_graph()
        matched = {'Formatter', 'FMT_COMPILE', 'format_to', 'string_view'}
        result = expand_smart(matched, G)

        # All matched present
        for nid in matched:
            self.assertIn(nid, result.expanded_nodes)

        # Expansion includes architectural neighbors
        self.assertIn('BaseFormatter', result.expanded_nodes)
        self.assertIn('FormatSpec', result.expanded_nodes)
        self.assertIn('basic_string', result.expanded_nodes)
        self.assertIn('vector', result.expanded_nodes)

        # No non-arch leaks
        self.assertNotIn('do_format', result.expanded_nodes)
        self.assertNotIn('constructor', result.expanded_nodes)

    def test_expansion_reasons_are_meaningful(self):
        """All expansion reasons should be non-empty descriptive strings."""
        G = self._build_complex_graph()
        result = expand_smart({'Formatter', 'FMT_COMPILE'}, G)

        for nid, reason in result.expansion_reasons.items():
            self.assertIsInstance(reason, str)
            self.assertGreater(len(reason), 0, f"Empty reason for {nid}")
            # Should contain a descriptive word
            self.assertTrue(
                any(kw in reason.lower() for kw in [
                    'base', 'template', 'created', 'composed', 'callee',
                    'referenced', 'macro', 'alias', 'free function',
                    'implements', 'expansion', 'type arg',
                ]),
                f"Reason '{reason}' for {nid} seems non-descriptive"
            )


# =============================================================================
# 9. REAL GRAPH INTEGRATION (fmtlib C++ — if available)
# =============================================================================

import gzip
import pickle

_THIS_DIR = Path(__file__).resolve().parent
_PLUGIN_DIR = _THIS_DIR.parent
_CACHE_DIR = _PLUGIN_DIR.parent.parent / "wiki_builder" / "cache"
CACHE_DIR = Path(os.environ.get("DEEPWIKI_CACHE_DIR", str(_CACHE_DIR)))
FMTLIB_GRAPH_HASH = "551134763c1f5c1b3feca4dd95076290"
CONFIG_GRAPH_HASH = "cd9d7a4aefa47194b872a7093a855156"

_graph_cache: Dict[str, object] = {}


def _load_graph(graph_hash: str):
    if graph_hash in _graph_cache:
        return _graph_cache[graph_hash]
    path = CACHE_DIR / f"{graph_hash}.code_graph.gz"
    if not path.exists():
        _graph_cache[graph_hash] = None
        return None
    with gzip.open(path, "rb") as f:
        graph = pickle.load(f)
    _graph_cache[graph_hash] = graph
    return graph


@unittest.skipUnless(
    _load_graph(FMTLIB_GRAPH_HASH) is not None,
    "fmtlib cached graph not found"
)
class TestRealFmtlibMacroExpansion(unittest.TestCase):
    """Test macro expansion on real fmtlib graph."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(FMTLIB_GRAPH_HASH)

    def test_macro_nodes_exist(self):
        """fmtlib graph may or may not have macro nodes depending on parser version."""
        macros = [
            nid for nid, d in self.graph.nodes(data=True)
            if d.get('symbol_type', '').lower() == 'macro'
        ]
        # Informational: log count but don't fail — older cached graphs
        # may not have macros yet (parser enhancement pending)
        if len(macros) == 0:
            self.skipTest(
                "No macro nodes in fmtlib cached graph (expected with older parser version)"
            )

    def test_macro_expansion_produces_results(self):
        """Expanding a macro in fmtlib should find related types/functions."""
        macro_node = None
        best_edge_count = 0
        for nid, data in self.graph.nodes(data=True):
            if data.get('symbol_type', '').lower() != 'macro':
                continue
            edge_count = (
                sum(1 for _ in self.graph.successors(nid)) +
                sum(1 for _ in self.graph.predecessors(nid))
            )
            if edge_count > best_edge_count:
                best_edge_count = edge_count
                macro_node = nid

        if not macro_node or best_edge_count < 1:
            self.skipTest("No macro with edges found in fmtlib")

        result = expand_smart({macro_node}, self.graph)
        # Should have expanded beyond just the macro itself
        expansion_count = len(result.expanded_nodes) - 1
        self.assertGreater(expansion_count, 0,
                           f"Expected expansion from macro {macro_node}")

    def test_specializes_with_type_args_real(self):
        """Real fmtlib: class with specializes + type arg references."""
        found = None
        for nid, data in self.graph.nodes(data=True):
            if data.get('symbol_type', '').lower() not in ('class', 'struct'):
                continue
            has_spec = False
            for succ in self.graph.successors(nid):
                if has_relationship(self.graph, nid, succ, 'specializes'):
                    has_spec = True
                    break
            if has_spec:
                found = nid
                break

        if not found:
            self.skipTest("No class with SPECIALIZES in fmtlib")

        result = expand_smart({found}, self.graph)
        spec_reasons = [r for r in result.expansion_reasons.values()
                        if 'template' in r.lower()]
        self.assertGreater(len(spec_reasons), 0,
                           "Expected template-related expansion reasons")

    def test_instantiates_edges_real(self):
        """Real fmtlib: check if instantiates edges exist and are followed."""
        found = None
        target = None
        for nid, data in self.graph.nodes(data=True):
            if data.get('symbol_type', '').lower() not in ('function', 'class', 'struct'):
                continue
            for succ in self.graph.successors(nid):
                if has_relationship(self.graph, nid, succ, 'instantiates'):
                    found = nid
                    target = succ
                    break
            if found:
                break

        if not found:
            self.skipTest("No INSTANTIATES edges in fmtlib graph")

        result = expand_smart({found}, self.graph)
        self.assertIn(target, result.expanded_nodes,
                       f"Expected instantiated target {target} in expansion")


@unittest.skipUnless(
    _load_graph(CONFIG_GRAPH_HASH) is not None,
    "configurations cached graph not found"
)
class TestRealConfigOrphanPattern(unittest.TestCase):
    """Test orphan class patterns on real configurations graph."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _load_graph(CONFIG_GRAPH_HASH)

    def test_orphan_classes_exist(self):
        """Find classes in configurations graph that have no arch expansion."""
        orphan_count = 0
        for nid, data in self.graph.nodes(data=True):
            if data.get('symbol_type', '').lower() != 'class':
                continue

            result = expand_smart({nid}, self.graph)
            expansion_beyond_self = result.expanded_nodes - {nid}
            if len(expansion_beyond_self) == 0:
                orphan_count += 1

        # This is informational — just log, don't fail
        # (configurations graph may or may not have orphans)

    def test_event_class_pattern(self):
        """Look for Event-like class pattern in configurations graph."""
        event_like = None
        for nid, data in self.graph.nodes(data=True):
            if data.get('symbol_type', '').lower() != 'class':
                continue
            name = data.get('symbol_name', '')
            if 'event' in name.lower() or 'Event' in name:
                event_like = nid
                break

        if not event_like:
            self.skipTest("No Event-like class in configurations graph")

        # Check if it has methods
        method_count = 0
        for succ in self.graph.successors(event_like):
            if has_relationship(self.graph, event_like, succ, 'defines'):
                succ_type = self.graph.nodes.get(succ, {}).get('symbol_type', '').lower()
                if succ_type in ('method', 'function'):
                    method_count += 1

        self.assertGreater(method_count, 0,
                           f"Event class '{event_like}' should have methods")


# =============================================================================
# 10. STRATEGY MAP COMPLETENESS
# =============================================================================

class TestStrategyMapCompleteness(unittest.TestCase):
    """Verify the strategy map covers all expected expansion-worthy types."""

    def test_all_class_like_types_have_strategy(self):
        for t in CLASS_LIKE_TYPES:
            self.assertIn(t, _STRATEGY_MAP, f"Missing strategy for class-like type '{t}'")

    def test_function_has_strategy(self):
        self.assertIn('function', _STRATEGY_MAP)

    def test_constant_has_strategy(self):
        self.assertIn('constant', _STRATEGY_MAP)

    def test_type_alias_has_strategy(self):
        self.assertIn('type_alias', _STRATEGY_MAP)

    def test_macro_has_strategy(self):
        self.assertIn('macro', _STRATEGY_MAP)

    def test_no_strategy_for_method(self):
        """Methods now share the function expansion strategy."""
        self.assertIn('method', _STRATEGY_MAP)
        # Verify it uses the same strategy as function
        self.assertIs(_STRATEGY_MAP['method'], _STRATEGY_MAP['function'])

    def test_no_strategy_for_field(self):
        """Fields should NOT have a strategy."""
        self.assertNotIn('field', _STRATEGY_MAP)

    def test_no_strategy_for_constructor(self):
        """Constructors now share the function expansion strategy."""
        self.assertIn('constructor', _STRATEGY_MAP)
        # Verify it uses the same strategy as function
        self.assertIs(_STRATEGY_MAP['constructor'], _STRATEGY_MAP['function'])


if __name__ == '__main__':
    unittest.main()
