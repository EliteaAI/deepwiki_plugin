"""
Test cross-file namespace resolution end-to-end:
  Parser → Graph Builder → Expansion Engine

Covers the scenario where C++ namespace-scoped symbols (structs, classes,
functions, template types) defined in one file are used in other files via
qualified names like ``detail::color_type`` or ``detail::parse_format_string``.

The tests verify:
1. Parser produces correct symbol names with namespace prefix
2. Parser emits correct cross-file reference/calls relationships
3. Graph builder resolves those references into real graph edges
4. Expansion engine follows those edges across files
"""
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

from plugin_implementation.parsers.cpp_enhanced_parser import CppEnhancedParser
from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
from plugin_implementation.code_graph.expansion_engine import expand_smart


# ═══════════════════════════════════════════════════════════════════════════════
# Shared fixture: three-file C++ project exercising namespace cross-file refs
# ═══════════════════════════════════════════════════════════════════════════════

DETAIL_H = r"""
namespace detail {

struct color_type {
    int r, g, b;
};

template <typename T>
struct streamed_view {
    T value;
};

class format_context {
    const char* data;
public:
    void advance(int n);
};

void parse_format_string(const char* fmt);

template <typename Char>
auto format_handler(Char c) -> int;

int compute_width(const char* s, int len);

}  // namespace detail
"""

COLOR_CPP = r"""
namespace fmt {

class text_style {
    detail::color_type foreground;
    detail::streamed_view<int> cached;
};

void render() {
    detail::parse_format_string("hello");
    auto x = detail::format_handler<char>('a');
    detail::color_type c;
    c.r = 255;
    int w = detail::compute_width("abc", 3);
}

}  // namespace fmt
"""

UTILS_CPP = r"""
void process(detail::color_type& color) {
    detail::streamed_view<double> sv;
    sv.value = 3.14;
}

detail::color_type make_color(int r, int g, int b) {
    detail::color_type c;
    c.r = r;
    return c;
}

void format(detail::format_context& ctx) {
    ctx.advance(1);
}
"""


@pytest.fixture(scope="module")
def cpp_project(tmp_path_factory):
    """Create a temporary three-file C++ project."""
    d = tmp_path_factory.mktemp("ns_crossfile")
    files = {
        "detail.h": DETAIL_H,
        "color.cpp": COLOR_CPP,
        "utils.cpp": UTILS_CPP,
    }
    paths = {}
    for name, content in files.items():
        p = d / name
        p.write_text(content)
        paths[name] = str(p)
    return str(d), paths


@pytest.fixture(scope="module")
def parse_results(cpp_project):
    """Parse all three files and return dict of filename → ParseResult."""
    _, paths = cpp_project
    parser = CppEnhancedParser()
    results = {}
    for name, path in paths.items():
        results[name] = parser.parse_file(path)
    return results


@pytest.fixture(scope="module")
def graph_and_analysis(cpp_project):
    """Build the unified graph from the three-file project."""
    repo_dir, _ = cpp_project
    builder = EnhancedUnifiedGraphBuilder()
    analysis = builder.analyze_repository(repo_dir)
    return analysis.unified_graph


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _symbol_names(parse_result) -> Dict[str, str]:
    """Return {full_name: symbol_type} for all symbols in a ParseResult."""
    return {
        s.full_name: s.symbol_type.value
        for s in parse_result.symbols
        if s.full_name
    }


def _rel_targets(parse_result, source_contains: str, rel_type: str) -> List[str]:
    """Return target_symbol list for edges matching source/type."""
    return [
        r.target_symbol
        for r in parse_result.relationships
        if source_contains in r.source_symbol and r.relationship_type.value == rel_type
    ]


def _graph_successors(graph, node_substring: str, rel_type: str) -> Set[str]:
    """Find all successor node IDs for edges matching type from a node."""
    result = set()
    for nid in graph.nodes():
        if node_substring in nid:
            for _, target, edata in graph.out_edges(nid, data=True):
                if edata.get("relationship_type") == rel_type:
                    result.add(target)
    return result


def _find_node(graph, substring: str) -> str:
    """Find a single node ID containing the substring."""
    for nid in graph.nodes():
        if substring in nid:
            return nid
    return None


def _expanded_names(graph, node_id: str) -> Set[str]:
    """Run expand_smart and return symbol names of expanded nodes."""
    result = expand_smart({node_id}, graph)
    names = set()
    for nid in result.expanded_nodes:
        ndata = graph.nodes.get(nid, {})
        names.add(ndata.get("symbol_name", ndata.get("name", nid)))
    return names


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Parser — symbol classification
# ═══════════════════════════════════════════════════════════════════════════════

class TestParserNamespaceSymbols:
    """Verify that namespace-scoped symbols get correct type and qualified name."""

    def test_detail_h_struct_symbol(self, parse_results):
        names = _symbol_names(parse_results["detail.h"])
        assert "detail.detail.color_type" in names
        assert names["detail.detail.color_type"] == "struct"

    def test_detail_h_template_struct(self, parse_results):
        names = _symbol_names(parse_results["detail.h"])
        assert "detail.detail.streamed_view" in names
        assert names["detail.detail.streamed_view"] == "struct"

    def test_detail_h_class_symbol(self, parse_results):
        names = _symbol_names(parse_results["detail.h"])
        assert "detail.detail.format_context" in names
        assert names["detail.detail.format_context"] == "class"

    def test_detail_h_function(self, parse_results):
        names = _symbol_names(parse_results["detail.h"])
        assert "detail.detail.parse_format_string" in names
        assert names["detail.detail.parse_format_string"] == "function"

    def test_detail_h_trailing_return_function(self, parse_results):
        """template <Char> auto format_handler(Char) -> int should be FUNCTION not METHOD."""
        names = _symbol_names(parse_results["detail.h"])
        assert "detail.detail.format_handler" in names
        assert names["detail.detail.format_handler"] == "function"

    def test_detail_h_plain_function(self, parse_results):
        names = _symbol_names(parse_results["detail.h"])
        assert "detail.detail.compute_width" in names
        assert names["detail.detail.compute_width"] == "function"

    def test_color_cpp_class_in_namespace(self, parse_results):
        names = _symbol_names(parse_results["color.cpp"])
        assert "color.fmt.text_style" in names
        assert names["color.fmt.text_style"] == "class"

    def test_color_cpp_function_in_namespace(self, parse_results):
        names = _symbol_names(parse_results["color.cpp"])
        assert "color.fmt.render" in names
        assert names["color.fmt.render"] == "function"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Parser — cross-file relationship edges
# ═══════════════════════════════════════════════════════════════════════════════

class TestParserCrossFileRelationships:
    """Verify that the parser emits correct edges for namespace-qualified references."""

    def test_field_type_reference(self, parse_results):
        """text_style fields should reference detail.color_type and detail.streamed_view."""
        targets = _rel_targets(parse_results["color.cpp"], "text_style", "references")
        assert "detail.color_type" in targets
        assert "detail.streamed_view" in targets

    def test_function_call_to_namespace_function(self, parse_results):
        """render() should have 'calls' edges to detail.parse_format_string."""
        targets = _rel_targets(parse_results["color.cpp"], "render", "calls")
        assert "detail.parse_format_string" in targets

    def test_function_call_to_template_function(self, parse_results):
        """render() should have 'calls' edge to detail.format_handler."""
        targets = _rel_targets(parse_results["color.cpp"], "render", "calls")
        assert "detail.format_handler" in targets

    def test_variable_declaration_reference(self, parse_results):
        """render() creates detail.color_type via local variable."""
        targets = _rel_targets(parse_results["color.cpp"], "render", "creates")
        assert "detail.color_type" in targets

    def test_parameter_type_reference(self, parse_results):
        """process(detail::color_type&) should reference detail.color_type."""
        targets = _rel_targets(parse_results["utils.cpp"], "process", "references")
        assert "detail.color_type" in targets

    def test_return_type_reference(self, parse_results):
        """make_color returning detail::color_type should reference it."""
        targets = _rel_targets(parse_results["utils.cpp"], "make_color", "references")
        assert "detail.color_type" in targets

    def test_template_instantiation_in_body(self, parse_results):
        """process() using detail::streamed_view<double> should reference it."""
        targets = _rel_targets(parse_results["utils.cpp"], "process", "references")
        assert "detail.streamed_view" in targets

    def test_class_parameter_reference(self, parse_results):
        """format(detail::format_context&) should reference the class."""
        targets = _rel_targets(parse_results["utils.cpp"], "format", "references")
        assert "detail.format_context" in targets

    def test_composition_edges(self, parse_results):
        """Field nodes should have composition edges to namespace types."""
        targets = _rel_targets(parse_results["color.cpp"], "foreground", "composition")
        assert "detail.color_type" in targets

        targets2 = _rel_targets(parse_results["color.cpp"], "cached", "composition")
        assert "detail.streamed_view" in targets2

    def test_function_call_to_plain_function(self, parse_results):
        """render() calls detail::compute_width — should produce a calls edge."""
        targets = _rel_targets(parse_results["color.cpp"], "render", "calls")
        assert "detail.compute_width" in targets


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Graph builder — cross-file edge resolution
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphCrossFileResolution:
    """Verify graph edges connect across file boundaries via namespace."""

    def test_graph_has_detail_struct_node(self, graph_and_analysis):
        """detail::color_type should be in the graph."""
        assert _find_node(graph_and_analysis, "detail.color_type") is not None

    def test_graph_has_detail_template_struct(self, graph_and_analysis):
        assert _find_node(graph_and_analysis, "detail.streamed_view") is not None

    def test_graph_has_detail_class(self, graph_and_analysis):
        assert _find_node(graph_and_analysis, "detail.format_context") is not None

    def test_graph_has_detail_functions(self, graph_and_analysis):
        assert _find_node(graph_and_analysis, "parse_format_string") is not None
        assert _find_node(graph_and_analysis, "format_handler") is not None
        assert _find_node(graph_and_analysis, "compute_width") is not None

    def test_graph_has_color_class(self, graph_and_analysis):
        assert _find_node(graph_and_analysis, "text_style") is not None

    def test_graph_has_color_function(self, graph_and_analysis):
        assert _find_node(graph_and_analysis, "fmt.render") is not None

    def test_text_style_references_detail_struct(self, graph_and_analysis):
        """text_style --references--> detail.color_type (cross-file)."""
        targets = _graph_successors(graph_and_analysis, "text_style", "references")
        detail_nodes = {t for t in targets if "color_type" in t and "detail" in t}
        assert len(detail_nodes) >= 1, f"Expected cross-file reference to detail.color_type, got {targets}"

    def test_text_style_references_template_struct(self, graph_and_analysis):
        """text_style --references--> detail.streamed_view (cross-file)."""
        targets = _graph_successors(graph_and_analysis, "text_style", "references")
        detail_nodes = {t for t in targets if "streamed_view" in t and "detail" in t}
        assert len(detail_nodes) >= 1

    def test_render_calls_detail_function(self, graph_and_analysis):
        """render --calls--> detail.parse_format_string (cross-file)."""
        targets = _graph_successors(graph_and_analysis, "fmt.render", "calls")
        assert any("parse_format_string" in t for t in targets), f"Calls: {targets}"

    def test_render_calls_template_function(self, graph_and_analysis):
        """render --calls--> detail.format_handler (cross-file)."""
        targets = _graph_successors(graph_and_analysis, "fmt.render", "calls")
        assert any("format_handler" in t for t in targets), f"Calls: {targets}"

    def test_render_calls_plain_function(self, graph_and_analysis):
        """render --calls--> detail.compute_width (cross-file)."""
        targets = _graph_successors(graph_and_analysis, "fmt.render", "calls")
        assert any("compute_width" in t for t in targets), f"Calls: {targets}"

    def test_process_references_detail_struct(self, graph_and_analysis):
        """process --references--> detail.color_type via parameter."""
        targets = _graph_successors(graph_and_analysis, "utils::process", "references")
        assert any("color_type" in t and "detail" in t for t in targets), f"Refs: {targets}"

    def test_make_color_references_return_type(self, graph_and_analysis):
        """make_color --references--> detail.color_type via return type."""
        targets = _graph_successors(graph_and_analysis, "make_color", "references")
        assert any("color_type" in t and "detail" in t for t in targets), f"Refs: {targets}"

    def test_format_references_class(self, graph_and_analysis):
        """format(detail::format_context&) should reference format_context cross-file."""
        # The function 'format' in utils.cpp references detail.format_context
        process_refs = set()
        for nid in graph_and_analysis.nodes():
            if "utils" in nid and "format" in nid and "color_type" not in nid:
                for _, target, edata in graph_and_analysis.out_edges(nid, data=True):
                    if edata.get("relationship_type") == "references":
                        process_refs.add(target)
        assert any("format_context" in t for t in process_refs), f"Refs: {process_refs}"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Expansion engine — cross-file namespace expansion
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpansionCrossFileNamespace:
    """Verify that expand_smart follows namespace edges across files."""

    def test_expand_text_style_pulls_detail_struct(self, graph_and_analysis):
        """Expanding text_style should pull in detail::color_type via field composition."""
        nid = _find_node(graph_and_analysis, "text_style")
        assert nid, "text_style node not found"
        names = _expanded_names(graph_and_analysis, nid)
        assert "color_type" in names, f"Expected color_type in expansion, got {names}"

    def test_expand_text_style_pulls_template_struct(self, graph_and_analysis):
        """Expanding text_style should pull in detail::streamed_view via field composition."""
        nid = _find_node(graph_and_analysis, "text_style")
        names = _expanded_names(graph_and_analysis, nid)
        assert "streamed_view" in names, f"Expected streamed_view in expansion, got {names}"

    def test_expand_render_pulls_called_function(self, graph_and_analysis):
        """Expanding render should pull in detail::parse_format_string via calls."""
        nid = _find_node(graph_and_analysis, "fmt.render")
        assert nid, "render node not found"
        names = _expanded_names(graph_and_analysis, nid)
        assert "parse_format_string" in names, f"Expected parse_format_string in expansion, got {names}"

    def test_expand_render_pulls_template_function(self, graph_and_analysis):
        """Expanding render should pull in detail::format_handler via calls."""
        nid = _find_node(graph_and_analysis, "fmt.render")
        names = _expanded_names(graph_and_analysis, nid)
        assert "format_handler" in names, f"Expected format_handler in expansion, got {names}"

    def test_expand_render_pulls_created_type(self, graph_and_analysis):
        """Expanding render should pull in detail::color_type via creates."""
        nid = _find_node(graph_and_analysis, "fmt.render")
        names = _expanded_names(graph_and_analysis, nid)
        assert "color_type" in names, f"Expected color_type in expansion, got {names}"

    def test_expand_process_pulls_parameter_type(self, graph_and_analysis):
        """Expanding process should pull in detail::color_type via parameter reference."""
        nid = _find_node(graph_and_analysis, "utils::process")
        assert nid, "process node not found"
        names = _expanded_names(graph_and_analysis, nid)
        assert "color_type" in names, f"Expected color_type in expansion, got {names}"

    def test_expand_process_pulls_template_instantiation(self, graph_and_analysis):
        """Expanding process should pull in detail::streamed_view via creates."""
        nid = _find_node(graph_and_analysis, "utils::process")
        names = _expanded_names(graph_and_analysis, nid)
        assert "streamed_view" in names, f"Expected streamed_view in expansion, got {names}"

    def test_expand_make_color_pulls_return_type(self, graph_and_analysis):
        """Expanding make_color should pull in detail::color_type via return type reference."""
        nid = _find_node(graph_and_analysis, "make_color")
        assert nid, "make_color node not found"
        names = _expanded_names(graph_and_analysis, nid)
        assert "color_type" in names, f"Expected color_type in expansion, got {names}"

    def test_backward_expand_struct_shows_users(self, graph_and_analysis):
        """Expanding detail::color_type (backward) should show classes/functions that use it."""
        nid = _find_node(graph_and_analysis, "detail.color_type")
        assert nid, "detail.color_type node not found"
        result = expand_smart({nid}, graph_and_analysis)
        expanded_ids = result.expanded_nodes
        # Should pull in text_style (which composes it) or process/make_color (which reference it)
        expanded_files = set()
        for eid in expanded_ids:
            ndata = graph_and_analysis.nodes.get(eid, {})
            rel = ndata.get("rel_path", "")
            if rel:
                expanded_files.add(rel)
        # At minimum, nodes from other files should be pulled in
        assert len(expanded_ids) > 1, f"Expected backward expansion, got {expanded_ids}"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Nested namespaces
# ═══════════════════════════════════════════════════════════════════════════════

NESTED_NS_H = r"""
namespace outer {
namespace inner {

struct Widget {
    int id;
};

void configure(Widget& w);

}  // namespace inner
}  // namespace outer
"""

NESTED_NS_CPP = r"""
void setup() {
    outer::inner::Widget w;
    w.id = 42;
    outer::inner::configure(w);
}
"""


class TestNestedNamespaceCrossFile:
    """Test deeply nested namespaces (outer::inner::Widget) across files."""

    @pytest.fixture(autouse=True)
    def _build_nested(self, tmp_path):
        d = tmp_path / "nested_ns"
        d.mkdir()
        (d / "widget.h").write_text(NESTED_NS_H)
        (d / "setup.cpp").write_text(NESTED_NS_CPP)
        builder = EnhancedUnifiedGraphBuilder()
        analysis = builder.analyze_repository(str(d))
        self.graph = analysis.unified_graph

    def test_nested_ns_widget_in_graph(self):
        """outer::inner::Widget should be in graph."""
        assert _find_node(self.graph, "Widget") is not None

    def test_nested_ns_configure_in_graph(self):
        """outer::inner::configure should be in graph."""
        assert _find_node(self.graph, "configure") is not None

    def test_setup_calls_configure_cross_file(self):
        """setup() should call outer::inner::configure across files."""
        targets = _graph_successors(self.graph, "setup", "calls")
        assert any("configure" in t for t in targets), f"Calls from setup: {targets}"

    def test_setup_creates_widget_cross_file(self):
        """setup() should create outer::inner::Widget across files."""
        targets = _graph_successors(self.graph, "setup", "creates")
        assert any("Widget" in t for t in targets), f"Creates from setup: {targets}"

    def test_expand_setup_pulls_nested_ns_symbols(self):
        """Expanding setup should pull in Widget and configure from other file."""
        nid = _find_node(self.graph, "setup")
        assert nid, "setup node not found"
        result = expand_smart({nid}, self.graph)
        names = set()
        for eid in result.expanded_nodes:
            ndata = self.graph.nodes.get(eid, {})
            sname = ndata.get("symbol_name", ndata.get("name", eid))
            names.add(sname)
        assert "Widget" in names or "configure" in names, \
            f"Expected nested-ns symbols in expansion, got {names}"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Method inside class inside namespace (true method, not free function)
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_IN_NS_H = r"""
namespace mylib {

class Engine {
public:
    void start();
    template <typename T>
    auto process(T item) -> bool;
};

}  // namespace mylib
"""

CLASS_IN_NS_USER_CPP = r"""
void run_engine() {
    mylib::Engine e;
    e.start();
}
"""


class TestMethodInsideNamespaceClass:
    """Template methods inside a class inside a namespace should be METHOD not FUNCTION."""

    @pytest.fixture(autouse=True)
    def _build(self, tmp_path):
        d = tmp_path / "cls_ns"
        d.mkdir()
        (d / "engine.h").write_text(CLASS_IN_NS_H)
        (d / "user.cpp").write_text(CLASS_IN_NS_USER_CPP)
        self.parser = CppEnhancedParser()
        self.result_h = self.parser.parse_file(str(d / "engine.h"))
        builder = EnhancedUnifiedGraphBuilder()
        analysis = builder.analyze_repository(str(d))
        self.graph = analysis.unified_graph

    def test_engine_class_in_namespace(self):
        """mylib::Engine should be parsed as class."""
        names = _symbol_names(self.result_h)
        engine_key = [k for k in names if "Engine" in k and "method" not in k.lower()]
        assert engine_key, f"Engine not found in {names}"
        assert names[engine_key[0]] == "class"

    def test_template_method_in_class_is_method(self):
        """template <T> auto process(T) -> bool inside class should be METHOD."""
        names = _symbol_names(self.result_h)
        process_key = [k for k in names if "process" in k]
        assert process_key, f"process not found in {names}"
        # Inside a class, it should be method
        assert names[process_key[0]] == "method"

    def test_run_engine_creates_engine_cross_file(self):
        """run_engine() creates mylib::Engine cross-file."""
        targets = _graph_successors(self.graph, "run_engine", "creates")
        assert any("Engine" in t for t in targets), f"Creates: {targets}"

    def test_expand_run_engine_pulls_class(self):
        """Expanding run_engine should pull in Engine from the namespace."""
        nid = _find_node(self.graph, "run_engine")
        assert nid
        names = _expanded_names(self.graph, nid)
        assert "Engine" in names, f"Expected Engine, got {names}"
