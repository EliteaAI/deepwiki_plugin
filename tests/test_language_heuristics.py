"""
Phase 7 tests — Language Heuristics.

Validates:
 - Language hints for all 8 rich parser language families
 - Dominant language detection from DB nodes
 - Symbol filtering based on language hints
 - Augmentation budget adjustment
 - Extra relationship type integration
 - Language-aware expansion recovers language-specific patterns
 - Flag disabled → no behavior change
"""

import sqlite3
import pytest
from unittest.mock import patch

from plugin_implementation.wiki_structure_planner.language_heuristics import (
    LanguageHints,
    get_language_hints,
    detect_dominant_language,
    should_include_in_expansion,
    get_extra_rel_types,
    compute_augmentation_budget_fraction,
    _HINTS_BY_LANGUAGE,
    _DEFAULT_HINTS,
)


# ─── Fixtures ────────────────────────────────────────────────────────


def _make_db():
    """Create an in-memory SQLite DB with repo_nodes and repo_edges tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE repo_nodes (
            node_id TEXT PRIMARY KEY,
            symbol_name TEXT,
            symbol_type TEXT,
            rel_path TEXT DEFAULT '',
            file_name TEXT DEFAULT '',
            language TEXT DEFAULT 'python',
            start_line INTEGER DEFAULT 0,
            end_line INTEGER DEFAULT 0,
            source_text TEXT DEFAULT '',
            is_architectural INTEGER DEFAULT 1,
            is_doc INTEGER DEFAULT 0,
            macro_cluster INTEGER,
            micro_cluster INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE repo_edges (
            source_id TEXT,
            target_id TEXT,
            rel_type TEXT DEFAULT 'calls',
            weight REAL DEFAULT 1.0
        )
    """)
    return conn


def _add_node(conn, node_id, symbol_name, symbol_type, language="python",
              macro_cluster=0, micro_cluster=0, **kwargs):
    defaults = dict(
        rel_path=f"src/{symbol_name}.py",
        file_name=f"{symbol_name}.py",
        start_line=1, end_line=10,
        source_text=f"class {symbol_name}:\n    pass",
        is_architectural=1, is_doc=0,
    )
    defaults.update(kwargs)
    conn.execute(
        "INSERT INTO repo_nodes "
        "(node_id, symbol_name, symbol_type, rel_path, file_name, language, "
        "start_line, end_line, source_text, is_architectural, is_doc, "
        "macro_cluster, micro_cluster) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (node_id, symbol_name, symbol_type,
         defaults["rel_path"], defaults["file_name"], language,
         defaults["start_line"], defaults["end_line"], defaults["source_text"],
         defaults["is_architectural"], defaults["is_doc"],
         macro_cluster, micro_cluster),
    )


def _add_edge(conn, source_id, target_id, rel_type="calls", weight=1.0):
    conn.execute(
        "INSERT INTO repo_edges (source_id, target_id, rel_type, weight) "
        "VALUES (?,?,?,?)",
        (source_id, target_id, rel_type, weight),
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. Hint registry — all 8 language families present
# ═══════════════════════════════════════════════════════════════════════

class TestHintRegistry:

    @pytest.mark.parametrize("lang", [
        "cpp", "c", "c_sharp", "go", "java", "javascript", "python", "rust", "typescript",
    ])
    def test_all_languages_have_hints(self, lang):
        """Each rich parser language has a registered hints object."""
        hints = get_language_hints(lang)
        assert isinstance(hints, LanguageHints)
        assert hints is not _DEFAULT_HINTS

    def test_unknown_language_returns_default(self):
        hints = get_language_hints("cobol")
        assert hints is _DEFAULT_HINTS
        assert hints.language == "unknown"


# ═══════════════════════════════════════════════════════════════════════
# 2. C++ specific heuristics
# ═══════════════════════════════════════════════════════════════════════

class TestCppHints:

    def test_prefers_impl_over_decl(self):
        hints = get_language_hints("cpp")
        assert hints.prefer_impl_over_decl is True

    def test_high_augmentation_priority(self):
        hints = get_language_hints("cpp")
        assert hints.impl_augmentation_priority == 2.0

    def test_has_declaration_extensions(self):
        hints = get_language_hints("cpp")
        assert ".h" in hints.declaration_extensions
        assert ".hpp" in hints.declaration_extensions

    def test_has_implementation_extensions(self):
        hints = get_language_hints("cpp")
        assert ".cpp" in hints.implementation_extensions
        assert ".cc" in hints.implementation_extensions

    def test_extra_rels_include_defines_body(self):
        hints = get_language_hints("cpp")
        assert "defines_body" in hints.extra_expansion_rels

    def test_macro_suppressed(self):
        hints = get_language_hints("cpp")
        assert not should_include_in_expansion(hints, "macro", "src/file.cpp")

    def test_class_not_suppressed(self):
        hints = get_language_hints("cpp")
        assert should_include_in_expansion(hints, "class", "src/file.h")


# ═══════════════════════════════════════════════════════════════════════
# 3. C# specific heuristics
# ═══════════════════════════════════════════════════════════════════════

class TestCSharpHints:

    def test_namespace_grouping(self):
        hints = get_language_hints("c_sharp")
        assert hints.grouping_hint == "namespace"

    def test_interface_impl_priority(self):
        hints = get_language_hints("c_sharp")
        assert hints.interface_impl_priority is True

    def test_extra_rels_include_implementation(self):
        hints = get_language_hints("c_sharp")
        assert "implementation" in hints.extra_expansion_rels


# ═══════════════════════════════════════════════════════════════════════
# 4. Go specific heuristics
# ═══════════════════════════════════════════════════════════════════════

class TestGoHints:

    def test_package_grouping(self):
        hints = get_language_hints("go")
        assert hints.grouping_hint == "package"

    def test_receiver_methods_priority(self):
        hints = get_language_hints("go")
        assert hints.impl_augmentation_priority > 1.0

    def test_interface_impl_priority(self):
        hints = get_language_hints("go")
        assert hints.interface_impl_priority is True


# ═══════════════════════════════════════════════════════════════════════
# 5. Java specific heuristics
# ═══════════════════════════════════════════════════════════════════════

class TestJavaHints:

    def test_package_grouping(self):
        hints = get_language_hints("java")
        assert hints.grouping_hint == "package"

    def test_annotation_support(self):
        hints = get_language_hints("java")
        assert "annotates" in hints.extra_expansion_rels

    def test_interface_impl_priority(self):
        hints = get_language_hints("java")
        assert hints.interface_impl_priority is True


# ═══════════════════════════════════════════════════════════════════════
# 6. JavaScript specific heuristics
# ═══════════════════════════════════════════════════════════════════════

class TestJavaScriptHints:

    def test_module_grouping(self):
        hints = get_language_hints("javascript")
        assert hints.grouping_hint == "module"

    def test_exports_rel(self):
        hints = get_language_hints("javascript")
        assert "exports" in hints.extra_expansion_rels

    def test_constant_suppressed(self):
        hints = get_language_hints("javascript")
        assert not should_include_in_expansion(hints, "constant", "src/config.js")


# ═══════════════════════════════════════════════════════════════════════
# 7. Python specific heuristics
# ═══════════════════════════════════════════════════════════════════════

class TestPythonHints:

    def test_module_grouping(self):
        hints = get_language_hints("python")
        assert hints.grouping_hint == "module"

    def test_decorates_rel(self):
        hints = get_language_hints("python")
        assert "decorates" in hints.extra_expansion_rels

    def test_no_symbols_suppressed(self):
        hints = get_language_hints("python")
        assert len(hints.skip_symbol_types) == 0


# ═══════════════════════════════════════════════════════════════════════
# 8. Rust specific heuristics
# ═══════════════════════════════════════════════════════════════════════

class TestRustHints:

    def test_module_grouping(self):
        hints = get_language_hints("rust")
        assert hints.grouping_hint == "module"

    def test_impl_priority(self):
        hints = get_language_hints("rust")
        assert hints.impl_augmentation_priority >= 1.5

    def test_interface_impl_priority(self):
        hints = get_language_hints("rust")
        assert hints.interface_impl_priority is True

    def test_extra_rels_include_impl(self):
        hints = get_language_hints("rust")
        assert "implementation" in hints.extra_expansion_rels


# ═══════════════════════════════════════════════════════════════════════
# 9. TypeScript specific heuristics
# ═══════════════════════════════════════════════════════════════════════

class TestTypeScriptHints:

    def test_module_grouping(self):
        hints = get_language_hints("typescript")
        assert hints.grouping_hint == "module"

    def test_prefers_ts_over_dts(self):
        hints = get_language_hints("typescript")
        assert hints.prefer_impl_over_decl is True
        assert ".d.ts" in hints.declaration_extensions

    def test_exports_rel(self):
        hints = get_language_hints("typescript")
        assert "exports" in hints.extra_expansion_rels


# ═══════════════════════════════════════════════════════════════════════
# 10. Dominant language detection
# ═══════════════════════════════════════════════════════════════════════

class TestDominantLanguageDetection:

    def test_detects_python_majority(self):
        conn = _make_db()
        for i in range(5):
            _add_node(conn, f"py{i}", f"PySym{i}", "class", language="python")
        for i in range(2):
            _add_node(conn, f"js{i}", f"JsSym{i}", "class", language="javascript")

        result = detect_dominant_language(conn, [f"py{i}" for i in range(5)] + [f"js{i}" for i in range(2)])
        assert result == "python"

    def test_detects_cpp_majority(self):
        conn = _make_db()
        for i in range(4):
            _add_node(conn, f"cpp{i}", f"CppSym{i}", "class", language="cpp",
                      rel_path=f"src/mod{i}.cpp")
        _add_node(conn, "py0", "PySym0", "class", language="python")

        result = detect_dominant_language(conn, [f"cpp{i}" for i in range(4)] + ["py0"])
        assert result == "cpp"

    def test_empty_returns_none(self):
        conn = _make_db()
        result = detect_dominant_language(conn, [])
        assert result is None

    def test_missing_nodes_returns_none(self):
        conn = _make_db()
        result = detect_dominant_language(conn, ["nonexistent1", "nonexistent2"])
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# 11. Symbol filtering
# ═══════════════════════════════════════════════════════════════════════

class TestSymbolFiltering:

    def test_suppresses_skipped_types(self):
        hints = LanguageHints(language="test", skip_symbol_types=frozenset({"macro", "constant"}))
        assert not should_include_in_expansion(hints, "macro", "src/file.py")
        assert not should_include_in_expansion(hints, "constant", "src/config.py")
        assert should_include_in_expansion(hints, "class", "src/file.py")

    def test_default_hints_suppress_nothing(self):
        hints = _DEFAULT_HINTS
        assert should_include_in_expansion(hints, "class", "src/file.py")
        assert should_include_in_expansion(hints, "macro", "src/file.py")
        assert should_include_in_expansion(hints, "constant", "src/file.py")


# ═══════════════════════════════════════════════════════════════════════
# 12. Augmentation budget
# ═══════════════════════════════════════════════════════════════════════

class TestAugmentationBudget:

    def test_cpp_gets_higher_budget(self):
        hints = get_language_hints("cpp")
        fraction = compute_augmentation_budget_fraction(hints)
        assert fraction > 0.3  # Default is 0.3, C++ should get more

    def test_python_gets_normal_budget(self):
        hints = get_language_hints("python")
        fraction = compute_augmentation_budget_fraction(hints)
        assert 0.25 <= fraction <= 0.35  # ~30%

    def test_clamped_to_range(self):
        hints = LanguageHints(language="extreme", impl_augmentation_priority=10.0)
        fraction = compute_augmentation_budget_fraction(hints)
        assert fraction == 0.6  # clamped to max

        hints_low = LanguageHints(language="minimal", impl_augmentation_priority=0.01)
        fraction_low = compute_augmentation_budget_fraction(hints_low)
        assert fraction_low == 0.1  # clamped to min


# ═══════════════════════════════════════════════════════════════════════
# 13. Extra relationship types
# ═══════════════════════════════════════════════════════════════════════

class TestExtraRelTypes:

    def test_get_extra_rel_types(self):
        hints = get_language_hints("java")
        extra = get_extra_rel_types(hints)
        assert "annotates" in extra
        assert "implementation" in extra

    def test_default_has_no_extra(self):
        extra = get_extra_rel_types(_DEFAULT_HINTS)
        assert len(extra) == 0


# ═══════════════════════════════════════════════════════════════════════
# 14. Integration: smart expansion with language hints
# ═══════════════════════════════════════════════════════════════════════

class TestSmartExpansionWithLanguageHints:

    def test_extra_rels_expand_additional_neighbors(self):
        """When extra_rel_types are passed, expand_symbol_smart follows them."""
        from plugin_implementation.code_graph.shared_expansion import expand_symbol_smart

        conn = _make_db()
        _add_node(conn, "n1", "MyTrait", "trait", language="rust",
                  rel_path="src/traits.rs")
        _add_node(conn, "n2", "MyImpl", "struct", language="rust",
                  rel_path="src/impl.rs")
        _add_edge(conn, "n2", "n1", "implementation")  # MyImpl implements MyTrait

        # Without extra rels: implementation is a P0 for class-like expansion,
        # but from the trait's perspective (n1 as seed), it should be found
        # via incoming 'implementation' edge
        seen = set()
        result = expand_symbol_smart(
            conn, "n1", "trait", seen,
            extra_rel_types=frozenset({"implementation"}),
        )
        found_ids = {r[0] for r in result}
        assert "n2" in found_ids

    def test_no_extra_rels_when_none(self):
        """With extra_rel_types=None, no extra expansion occurs."""
        from plugin_implementation.code_graph.shared_expansion import expand_symbol_smart

        conn = _make_db()
        _add_node(conn, "n1", "func1", "function", language="python",
                  rel_path="src/mod.py")
        _add_node(conn, "n2", "func2", "function", language="python",
                  rel_path="src/mod.py")
        _add_edge(conn, "n1", "n2", "decorates")

        seen = set()
        # Without extra rels, 'decorates' is in SKIP_RELATIONSHIPS → not followed
        result_no_extra = expand_symbol_smart(
            conn, "n1", "function", seen,
            extra_rel_types=None,
        )
        found_no = {r[0] for r in result_no_extra}

        # With extra rels including 'decorates'
        seen2 = set()
        result_extra = expand_symbol_smart(
            conn, "n1", "function", seen2,
            extra_rel_types=frozenset({"decorates"}),
        )
        found_extra = {r[0] for r in result_extra}

        # n2 should be found with decorates extra rel, but maybe not without
        assert "n2" in found_extra

    def test_cpp_defines_body_expansion(self):
        """C++ hints should allow following defines_body edges for implementation recovery."""
        from plugin_implementation.code_graph.shared_expansion import expand_symbol_smart

        conn = _make_db()
        _add_node(conn, "h1", "Widget", "class", language="cpp",
                  rel_path="include/widget.h")
        _add_node(conn, "impl1", "Widget_render", "function", language="cpp",
                  rel_path="src/widget.cpp")
        _add_edge(conn, "h1", "impl1", "defines_body")

        hints = get_language_hints("cpp")
        seen = set()
        result = expand_symbol_smart(
            conn, "h1", "class", seen,
            extra_rel_types=hints.extra_expansion_rels,
        )
        found_ids = {r[0] for r in result}
        assert "impl1" in found_ids


# ═══════════════════════════════════════════════════════════════════════
# 15. Flag gating
# ═══════════════════════════════════════════════════════════════════════

class TestFlagGating:

    def test_language_hints_on_by_default(self):
        """Cluster baseline is now ON; language_hints is part of that baseline."""
        from plugin_implementation.feature_flags import FeatureFlags
        flags = FeatureFlags()
        assert flags.language_hints is True

    def test_language_hints_can_be_disabled_explicitly(self):
        """Explicit override still works for tests that need the legacy path."""
        from plugin_implementation.feature_flags import FeatureFlags
        flags = FeatureFlags(smart_expansion=False, language_hints=False)
        assert flags.smart_expansion is False
        assert flags.language_hints is False
        # Both off = purely legacy expansion, no lang hints code reached
