"""
Test C++ parser macro emission.

Tests that the C++ parser correctly emits SymbolType.MACRO for:
1. Simple value macros  (#define FOO 42)
2. Empty/flag macros    (#define FOO)
3. Function-like macros (#define FOO(x) ...)
4. Include guards
5. Macro metadata (value, function_macro flag, body, parameters)
6. Macro interactions with other symbol types
7. Graph builder treats macros as architectural symbols

This verifies the fix where visit_preproc_def and visit_preproc_function_def
were incorrectly emitting SymbolType.CONSTANT and SymbolType.FUNCTION
instead of SymbolType.MACRO.
"""

import os
import tempfile

import pytest
from plugin_implementation.parsers.cpp_enhanced_parser import CppEnhancedParser
from plugin_implementation.parsers.base_parser import SymbolType
from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
from plugin_implementation.constants import ARCHITECTURAL_SYMBOLS


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def parser():
    return CppEnhancedParser()


def _symbols_by_type(result, symbol_type_value):
    """Return symbols matching a specific symbol_type.value."""
    return [s for s in result.symbols
            if (s.symbol_type.value if hasattr(s.symbol_type, 'value')
                else str(s.symbol_type)) == symbol_type_value]


def _build_graph(code, filename='test.h'):
    """Build a graph from C++ source code using the standard builder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        with open(filepath, 'w') as f:
            f.write(code)
        builder = EnhancedUnifiedGraphBuilder()
        analysis = builder.analyze_repository(tmpdir)
        return analysis.unified_graph


# ── Simple Value Macros ───────────────────────────────────────────────

class TestSimpleValueMacros:
    """Test that #define NAME value macros emit SymbolType.MACRO."""

    SIMPLE_MACROS = """
#define MAX_BUFFER_SIZE 1024
#define FMT_VERSION 90100
#define PI 3.14159
#define GREETING "hello world"
"""

    def test_simple_macros_have_macro_type(self, parser):
        """Simple value macros must be typed as MACRO, not CONSTANT."""
        result = parser.parse_file('test.h', self.SIMPLE_MACROS)
        macros = _symbols_by_type(result, 'macro')
        macro_names = {s.name for s in macros}

        assert 'MAX_BUFFER_SIZE' in macro_names
        assert 'FMT_VERSION' in macro_names
        assert 'PI' in macro_names
        assert 'GREETING' in macro_names

    def test_simple_macros_not_typed_as_constant(self, parser):
        """Ensure no macro leaks into the constant bucket."""
        result = parser.parse_file('test.h', self.SIMPLE_MACROS)
        constants = _symbols_by_type(result, 'constant')
        constant_names = {s.name for s in constants}

        assert 'MAX_BUFFER_SIZE' not in constant_names
        assert 'FMT_VERSION' not in constant_names

    def test_simple_macro_metadata_value(self, parser):
        """Simple value macros should have metadata['value'] set."""
        result = parser.parse_file('test.h', self.SIMPLE_MACROS)
        macros = _symbols_by_type(result, 'macro')
        by_name = {s.name: s for s in macros}

        meta = getattr(by_name['MAX_BUFFER_SIZE'], 'metadata', {}) or {}
        assert meta.get('value') == '1024', f"Expected '1024', got {meta.get('value')}"
        assert meta.get('macro') is True

    def test_simple_macro_not_function_macro(self, parser):
        """Simple macros should NOT have function_macro flag."""
        result = parser.parse_file('test.h', self.SIMPLE_MACROS)
        macros = _symbols_by_type(result, 'macro')
        for m in macros:
            meta = getattr(m, 'metadata', {}) or {}
            assert not meta.get('function_macro'), \
                f"Simple macro {m.name} should not be a function_macro"


# ── Empty / Flag Macros ──────────────────────────────────────────────

class TestFlagMacros:
    """Test that #define NAME (no value) macros emit SymbolType.MACRO."""

    FLAG_MACROS = """
#define FMT_HEADER_ONLY
#define NDEBUG
#define __MY_CUSTOM_GUARD_H__
"""

    def test_flag_macros_have_macro_type(self, parser):
        """Flag macros (no value) should be MACRO."""
        result = parser.parse_file('test.h', self.FLAG_MACROS)
        macros = _symbols_by_type(result, 'macro')
        macro_names = {s.name for s in macros}

        assert 'FMT_HEADER_ONLY' in macro_names
        assert 'NDEBUG' in macro_names

    def test_flag_macro_value_is_none(self, parser):
        """Flag macros should have metadata value=None."""
        result = parser.parse_file('test.h', self.FLAG_MACROS)
        macros = _symbols_by_type(result, 'macro')
        by_name = {s.name: s for s in macros}

        meta = getattr(by_name['FMT_HEADER_ONLY'], 'metadata', {}) or {}
        # value may be None or empty string
        assert meta.get('value') is None or meta.get('value') == '', \
            f"Flag macro value should be None/empty, got {meta.get('value')}"


# ── Function-Like Macros ─────────────────────────────────────────────

class TestFunctionLikeMacros:
    """Test that #define NAME(args) body macros emit SymbolType.MACRO."""

    FUNC_MACROS = """
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define FMT_COMPILE(s) detail::compile<decltype(s)>(s)
#define ASSERT_EQ(expected, actual) do { if ((expected) != (actual)) abort(); } while(0)
#define STRINGIFY(x) #x
"""

    def test_function_macros_have_macro_type(self, parser):
        """Function-like macros must be typed as MACRO, not FUNCTION."""
        result = parser.parse_file('test.h', self.FUNC_MACROS)
        macros = _symbols_by_type(result, 'macro')
        macro_names = {s.name for s in macros}

        assert 'MIN' in macro_names
        assert 'MAX' in macro_names
        assert 'FMT_COMPILE' in macro_names
        assert 'ASSERT_EQ' in macro_names
        assert 'STRINGIFY' in macro_names

    def test_function_macros_not_typed_as_function(self, parser):
        """Ensure no function-like macro leaks into the function bucket."""
        result = parser.parse_file('test.h', self.FUNC_MACROS)
        functions = _symbols_by_type(result, 'function')
        function_names = {s.name for s in functions}

        assert 'MIN' not in function_names
        assert 'FMT_COMPILE' not in function_names

    def test_function_macro_metadata_flag(self, parser):
        """Function-like macros should have metadata['function_macro']=True."""
        result = parser.parse_file('test.h', self.FUNC_MACROS)
        macros = _symbols_by_type(result, 'macro')
        by_name = {s.name: s for s in macros}

        meta = getattr(by_name['MIN'], 'metadata', {}) or {}
        assert meta.get('function_macro') is True
        assert meta.get('macro') is True

    def test_function_macro_has_body(self, parser):
        """Function-like macros should have metadata['body'] with the expansion text."""
        result = parser.parse_file('test.h', self.FUNC_MACROS)
        macros = _symbols_by_type(result, 'macro')
        by_name = {s.name: s for s in macros}

        meta = getattr(by_name['MIN'], 'metadata', {}) or {}
        body = meta.get('body', '')
        assert body, f"MIN should have a body, got: {body!r}"
        # The body should contain the ternary expression
        assert '?' in body or '<' in body, f"Unexpected body content: {body}"

    def test_function_macro_parameters_as_child_symbols(self, parser):
        """Function-like macros should have parameter child symbols."""
        result = parser.parse_file('test.h', self.FUNC_MACROS)
        params = _symbols_by_type(result, 'parameter')

        # MIN(a, b) should produce 2 parameters
        min_params = [p for p in params if 'MIN' in (p.full_name or '')]
        assert len(min_params) == 2, \
            f"Expected 2 params for MIN, got {len(min_params)}: {[p.name for p in min_params]}"
        param_names = {p.name for p in min_params}
        assert 'a' in param_names
        assert 'b' in param_names


# ── Mixed Macros and Non-Macros ──────────────────────────────────────

class TestMixedSymbols:
    """Test that macros coexist correctly with other symbol types."""

    MIXED_CODE = """
#define VERSION 42
#define MAKE_FOO(x) new Foo(x)

const int REAL_CONSTANT = 100;

class Foo {
public:
    explicit Foo(int x);
    void bar();
};

void standalone();
"""

    def test_macro_vs_real_constant(self, parser):
        """#define is MACRO; const int is CONSTANT — they must not collide."""
        result = parser.parse_file('test.h', self.MIXED_CODE)

        macros = _symbols_by_type(result, 'macro')
        constants = _symbols_by_type(result, 'constant')

        macro_names = {s.name for s in macros}
        constant_names = {s.name for s in constants}

        # VERSION is #define → macro
        assert 'VERSION' in macro_names
        assert 'VERSION' not in constant_names

        # REAL_CONSTANT is const int → constant
        assert 'REAL_CONSTANT' in constant_names
        assert 'REAL_CONSTANT' not in macro_names

    def test_function_macro_vs_real_function(self, parser):
        """#define MAKE_FOO(x) is MACRO; void standalone() is FUNCTION."""
        result = parser.parse_file('test.h', self.MIXED_CODE)

        macros = _symbols_by_type(result, 'macro')
        functions = _symbols_by_type(result, 'function')

        macro_names = {s.name for s in macros}
        function_names = {s.name for s in functions}

        assert 'MAKE_FOO' in macro_names
        assert 'MAKE_FOO' not in function_names

        assert 'standalone' in function_names
        assert 'standalone' not in macro_names

    def test_class_unaffected_by_nearby_macros(self, parser):
        """Class parsing should not be disrupted by surrounding macros."""
        result = parser.parse_file('test.h', self.MIXED_CODE)

        classes = _symbols_by_type(result, 'class')
        class_names = {s.name for s in classes}
        assert 'Foo' in class_names

    def test_total_macro_count(self, parser):
        """Only the two #define lines should produce macro symbols."""
        result = parser.parse_file('test.h', self.MIXED_CODE)
        macros = _symbols_by_type(result, 'macro')
        assert len(macros) == 2, \
            f"Expected 2 macros, got {len(macros)}: {[m.name for m in macros]}"


# ── Graph Builder Integration ────────────────────────────────────────

class TestMacroInGraphBuilder:
    """Test that macro symbols flow into the graph as architectural nodes."""

    GRAPH_CODE = """
#define BUFFER_SIZE 4096
#define FMT_THROW(x) throw x

class MyClass {
public:
    void method();
};
"""

    def test_macro_nodes_in_graph(self, parser):
        """Macros should appear as nodes in the graph."""
        graph = _build_graph(self.GRAPH_CODE)

        # Find macro nodes — the graph may store symbol as a Symbol object or dict
        macro_nodes = []
        for nid, data in graph.nodes(data=True):
            sym = data.get('symbol')
            st = data.get('symbol_type', '')
            if st == 'macro':
                macro_nodes.append(nid)
            elif sym is not None:
                kind = getattr(sym, 'kind', None) or (sym.get('kind') if isinstance(sym, dict) else None)
                if kind == 'macro':
                    macro_nodes.append(nid)
        assert len(macro_nodes) >= 1, \
            f"Expected at least 1 macro node in graph, got {len(macro_nodes)}"

    def test_macro_is_architectural(self, parser):
        """Macro symbol_type 'macro' is in ARCHITECTURAL_SYMBOLS."""
        assert 'macro' in ARCHITECTURAL_SYMBOLS, \
            f"'macro' not in ARCHITECTURAL_SYMBOLS: {ARCHITECTURAL_SYMBOLS}"

    def test_macro_node_has_correct_kind(self, parser):
        """Graph nodes for macros should have symbol_type='macro' or symbol.kind='macro'."""
        graph = _build_graph(self.GRAPH_CODE)

        found = False
        for nid, data in graph.nodes(data=True):
            if 'BUFFER_SIZE' not in nid:
                continue
            st = data.get('symbol_type', '')
            sym = data.get('symbol')
            kind = None
            if sym is not None:
                kind = getattr(sym, 'kind', None) or (sym.get('kind') if isinstance(sym, dict) else None)
            assert st == 'macro' or kind == 'macro', \
                f"BUFFER_SIZE node should be macro, got symbol_type={st!r}, kind={kind!r}"
            found = True
            break
        assert found, "BUFFER_SIZE macro not found in graph at all"


# ── Realistic Header File ────────────────────────────────────────────

class TestRealisticHeader:
    """Test with a realistic C++ header combining all macro forms."""

    REALISTIC_HEADER = """
#ifndef MY_LIB_H_
#define MY_LIB_H_

#include <string>

// Version info
#define MY_LIB_VERSION_MAJOR 2
#define MY_LIB_VERSION_MINOR 1
#define MY_LIB_VERSION_PATCH 0

// Feature flags
#define MY_LIB_ENABLE_LOGGING
#define MY_LIB_USE_EXCEPTIONS

// Utility macros
#define MY_LIB_STRINGIFY(x) #x
#define MY_LIB_CONCAT(a, b) a ## b
#define MY_LIB_ASSERT(cond) do { if(!(cond)) abort(); } while(0)

// API export
#ifdef _WIN32
  #define MY_LIB_API __declspec(dllexport)
#else
  #define MY_LIB_API __attribute__((visibility("default")))
#endif

namespace mylib {

class Widget {
public:
    Widget();
    ~Widget();
    void render();
private:
    std::string name_;
};

int compute(int x, int y);

} // namespace mylib

#endif // MY_LIB_H_
"""

    def test_all_macros_detected(self, parser):
        """All macro definitions in a realistic header should be detected."""
        result = parser.parse_file('my_lib.h', self.REALISTIC_HEADER)
        macros = _symbols_by_type(result, 'macro')
        macro_names = {s.name for s in macros}

        # Include guard
        assert 'MY_LIB_H_' in macro_names

        # Version macros
        assert 'MY_LIB_VERSION_MAJOR' in macro_names
        assert 'MY_LIB_VERSION_MINOR' in macro_names

        # Feature flags
        assert 'MY_LIB_ENABLE_LOGGING' in macro_names

        # Function-like macros
        assert 'MY_LIB_STRINGIFY' in macro_names
        assert 'MY_LIB_ASSERT' in macro_names

    def test_no_macros_as_constants(self, parser):
        """None of the #define macros should appear as constants."""
        result = parser.parse_file('my_lib.h', self.REALISTIC_HEADER)
        constants = _symbols_by_type(result, 'constant')
        constant_names = {s.name for s in constants}

        # None of these should be typed as constant
        for macro_name in ['MY_LIB_H_', 'MY_LIB_VERSION_MAJOR',
                           'MY_LIB_ENABLE_LOGGING', 'MY_LIB_API']:
            assert macro_name not in constant_names, \
                f"{macro_name} should be MACRO, not CONSTANT"

    def test_no_function_macros_as_functions(self, parser):
        """None of the function-like macros should appear as functions."""
        result = parser.parse_file('my_lib.h', self.REALISTIC_HEADER)
        functions = _symbols_by_type(result, 'function')
        function_names = {s.name for s in functions}

        for macro_name in ['MY_LIB_STRINGIFY', 'MY_LIB_CONCAT', 'MY_LIB_ASSERT']:
            assert macro_name not in function_names, \
                f"{macro_name} should be MACRO, not FUNCTION"

    def test_real_class_and_function_still_parsed(self, parser):
        """Regular class and function should be unaffected by macros."""
        result = parser.parse_file('my_lib.h', self.REALISTIC_HEADER)

        classes = _symbols_by_type(result, 'class')
        class_names = {s.name for s in classes}
        assert 'Widget' in class_names

        functions = _symbols_by_type(result, 'function')
        function_names = {s.name for s in functions}
        assert 'compute' in function_names

    def test_macro_count_reasonable(self, parser):
        """The realistic header should produce a reasonable number of macro symbols."""
        result = parser.parse_file('my_lib.h', self.REALISTIC_HEADER)
        macros = _symbols_by_type(result, 'macro')
        # We have: MY_LIB_H_, VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH,
        #          ENABLE_LOGGING, USE_EXCEPTIONS, STRINGIFY, CONCAT, ASSERT,
        #          MY_LIB_API (conditional, at least one branch)
        assert len(macros) >= 8, \
            f"Expected at least 8 macros, got {len(macros)}: {[m.name for m in macros]}"


# ── SymbolType Enum Consistency ──────────────────────────────────────

class TestSymbolTypeEnum:
    """Test that SymbolType.MACRO exists and is 'macro'."""

    def test_symbol_type_macro_exists(self):
        """SymbolType.MACRO must exist in the enum."""
        assert hasattr(SymbolType, 'MACRO'), "SymbolType missing MACRO member"

    def test_symbol_type_macro_value(self):
        """SymbolType.MACRO.value must be 'macro'."""
        assert SymbolType.MACRO.value == 'macro'

    def test_cpp_parser_capabilities_include_macro(self):
        """C++ parser's LanguageCapabilities should list MACRO."""
        parser = CppEnhancedParser()
        caps = parser.get_capabilities()
        supported = {st.value if hasattr(st, 'value') else str(st)
                     for st in caps.supported_symbols}
        assert 'macro' in supported, \
            f"'macro' not in C++ parser capabilities: {supported}"
