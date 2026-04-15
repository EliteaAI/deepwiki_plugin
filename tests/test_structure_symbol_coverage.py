"""
Phase 3 tests — Symbol-level coverage tracking in StructureCollector.

Tests that:
- Symbol coverage is tracked when pages define target_symbols
- Coverage feedback includes percentage and unassigned symbol names
- Batch mode also tracks symbols and warns about unknown ones
- _ensure_architectural_symbols_loaded filters by symbol_type
"""

import pytest
import networkx as nx

from plugin_implementation.wiki_structure_planner.structure_tools import StructureCollector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _build_mock_graph(symbols):
    """Build a minimal nx.DiGraph that StructureCollector can inspect.

    Args:
        symbols: List of dicts with keys 'name', 'symbol_type', 'rel_path'.
    """
    g = nx.DiGraph()
    for sym in symbols:
        node_id = f"{sym['rel_path']}::{sym['name']}"
        g.add_node(node_id,
                   symbol_name=sym['name'],
                   symbol_type=sym['symbol_type'],
                   rel_path=sym['rel_path'])
    return g


@pytest.fixture
def sample_graph():
    """Graph with 5 architectural symbols and 2 non-architectural (method, module_doc)."""
    return _build_mock_graph([
        {'name': 'UserService',    'symbol_type': 'class',      'rel_path': 'src/user.py'},
        {'name': 'OrderService',   'symbol_type': 'class',      'rel_path': 'src/orders.py'},
        {'name': 'PaymentHandler', 'symbol_type': 'class',      'rel_path': 'src/pay.py'},
        {'name': 'calculate_tax',  'symbol_type': 'function',   'rel_path': 'src/tax.py'},
        {'name': 'MAX_RETRIES',    'symbol_type': 'constant',   'rel_path': 'src/constants.py'},
        # Non-architectural — should NOT be counted
        {'name': 'get_name',       'symbol_type': 'method',     'rel_path': 'src/user.py'},
        {'name': 'module overview','symbol_type': 'module_doc', 'rel_path': 'src/user.py'},
    ])


@pytest.fixture
def collector(sample_graph):
    """StructureCollector wired to the sample graph."""
    c = StructureCollector(page_budget=10, repo_root="/fake", code_graph=sample_graph)
    # Pre-create a section so define_page doesn't complain
    c.sections["Core"] = {
        "section_name": "Core",
        "section_order": 1,
        "description": "Core section",
        "rationale": "test",
        "pages": [],
    }
    return c


# ---------------------------------------------------------------------------
# Tests — _ensure_architectural_symbols_loaded
# ---------------------------------------------------------------------------

class TestArchitecturalSymbolLoading:
    """Verify the lazy loader only picks architectural symbols."""

    def test_loads_only_architectural_types(self, collector):
        collector._ensure_architectural_symbols_loaded()
        assert collector._all_architectural_symbols == {
            'UserService', 'OrderService', 'PaymentHandler',
            'calculate_tax', 'MAX_RETRIES',
        }

    def test_skips_methods_and_module_doc(self, collector):
        collector._ensure_architectural_symbols_loaded()
        assert 'get_name' not in collector._all_architectural_symbols
        assert 'module overview' not in collector._all_architectural_symbols

    def test_loads_once(self, collector):
        """Second call should be a no-op (set already populated)."""
        collector._ensure_architectural_symbols_loaded()
        first_set_id = id(collector._all_architectural_symbols)
        collector._ensure_architectural_symbols_loaded()
        assert id(collector._all_architectural_symbols) == first_set_id

    def test_no_graph_returns_empty(self):
        """Without a code_graph, set stays empty."""
        c = StructureCollector(page_budget=5, repo_root="/x")
        c._ensure_architectural_symbols_loaded()
        assert c._all_architectural_symbols == set()


# ---------------------------------------------------------------------------
# Tests — symbol coverage via define_page
# ---------------------------------------------------------------------------

class TestDefinePageSymbolCoverage:
    """define_page should register validated symbols and report coverage."""

    def test_single_page_covers_symbols(self, collector):
        result = collector._handle_define_page(
            section_name="Core",
            page_name="User Management",
            page_order=1,
            description="Handles users",
            content_focus="User lifecycle",
            rationale="users",
            target_symbols=["UserService"],
            target_docs=[],
            target_folders=["src"],
            key_files=["src/user.py"],
        )
        # UserService should now be covered
        assert "UserService" in collector._covered_symbols
        assert collector._symbol_to_page.get("UserService") == "User Management"
        # Coverage feedback should mention symbol coverage
        assert "Symbol coverage" in result

    def test_multiple_pages_accumulate_coverage(self, collector):
        collector._handle_define_page(
            section_name="Core", page_name="P1", page_order=1,
            description="d", content_focus="c", rationale="r",
            target_symbols=["UserService", "OrderService"],
            target_docs=[], target_folders=["src"], key_files=[],
        )
        collector._handle_define_page(
            section_name="Core", page_name="P2", page_order=2,
            description="d", content_focus="c", rationale="r",
            target_symbols=["PaymentHandler", "calculate_tax"],
            target_docs=[], target_folders=["src"], key_files=[],
        )
        covered, total, uncovered = collector._compute_symbol_coverage()
        assert covered == 4
        assert total == 5
        assert uncovered == ["MAX_RETRIES"]

    def test_full_coverage_no_unassigned(self, collector):
        collector._handle_define_page(
            section_name="Core", page_name="All", page_order=1,
            description="d", content_focus="c", rationale="r",
            target_symbols=[
                "UserService", "OrderService", "PaymentHandler",
                "calculate_tax", "MAX_RETRIES",
            ],
            target_docs=[], target_folders=["src"], key_files=[],
        )
        covered, total, uncovered = collector._compute_symbol_coverage()
        assert covered == total == 5
        assert uncovered == []

    def test_hallucinated_symbol_not_counted(self, collector):
        """Symbols not in the graph should not appear in _covered_symbols."""
        result = collector._handle_define_page(
            section_name="Core", page_name="Fake", page_order=1,
            description="d", content_focus="c", rationale="r",
            target_symbols=["NonExistentClass"],
            target_docs=[], target_folders=["src"], key_files=[],
        )
        assert "NonExistentClass" not in collector._covered_symbols
        assert "not found in graph" in result

    def test_coverage_feedback_format(self, collector):
        """Feedback should show fraction and percentage."""
        collector._handle_define_page(
            section_name="Core", page_name="P1", page_order=1,
            description="d", content_focus="c", rationale="r",
            target_symbols=["UserService"],
            target_docs=[], target_folders=["src"], key_files=[],
        )
        feedback = collector._format_symbol_coverage_feedback()
        assert "1/5" in feedback
        assert "20%" in feedback
        assert "Unassigned symbols" in feedback


# ---------------------------------------------------------------------------
# Tests — symbol coverage via batch_define_pages
# ---------------------------------------------------------------------------

class TestBatchDefinePageSymbolCoverage:
    """batch_define_pages should also track symbols and warn about unknowns."""

    def test_batch_registers_symbols(self, collector):
        pages = [
            {
                "section_name": "Core",
                "page_name": "Users",
                "description": "User module",
                "target_symbols": ["UserService"],
                "target_folders": ["src"],
            },
            {
                "section_name": "Core",
                "page_name": "Orders",
                "description": "Order module",
                "target_symbols": ["OrderService", "calculate_tax"],
                "target_folders": ["src"],
            },
        ]
        # Trigger the case-insensitive index build first (normally done by define_page)
        collector._handle_define_page(
            section_name="Core", page_name="Init", page_order=0,
            description="d", content_focus="c", rationale="r",
            target_symbols=["MAX_RETRIES"],
            target_docs=[], target_folders=["src"], key_files=[],
        )
        result = collector._handle_batch_define_pages(pages)
        # Check symbol registration
        assert "UserService" in collector._covered_symbols
        assert "OrderService" in collector._covered_symbols
        assert "calculate_tax" in collector._covered_symbols
        assert "Symbol coverage" in result

    def test_batch_warns_unknown_symbols(self, collector):
        """Per-page validation should flag symbols not in the graph."""
        # Build the index first
        collector._handle_define_page(
            section_name="Core", page_name="Init", page_order=0,
            description="d", content_focus="c", rationale="r",
            target_symbols=["UserService"],
            target_docs=[], target_folders=["src"], key_files=[],
        )
        pages = [
            {
                "section_name": "Core",
                "page_name": "Ghosts",
                "description": "Ghost module",
                "target_symbols": ["GhostClass", "SpookyFunction"],
                "target_folders": ["src"],
            },
        ]
        result = collector._handle_batch_define_pages(pages)
        assert "not in graph" in result
        assert "GhostClass" in result


# ---------------------------------------------------------------------------
# Tests — _compute_symbol_coverage edge cases
# ---------------------------------------------------------------------------

class TestSymbolCoverageEdgeCases:
    """Edge cases for coverage computation."""

    def test_empty_graph(self):
        """No graph → (0, 0, [])."""
        c = StructureCollector(page_budget=5, repo_root="/x")
        assert c._compute_symbol_coverage() == (0, 0, [])

    def test_graph_with_only_methods(self):
        """If graph has only methods, coverage should be 0/0."""
        g = _build_mock_graph([
            {'name': 'do_stuff', 'symbol_type': 'method', 'rel_path': 'a.py'},
        ])
        c = StructureCollector(page_budget=5, repo_root="/x", code_graph=g)
        assert c._compute_symbol_coverage() == (0, 0, [])

    def test_reset_clears_tracking(self, collector):
        """reset() should wipe symbol tracking."""
        collector._covered_symbols.add("UserService")
        collector._all_architectural_symbols.add("UserService")
        collector._symbol_to_page["UserService"] = "P1"
        collector.reset()
        assert collector._covered_symbols == set()
        assert collector._all_architectural_symbols == set()
        assert collector._symbol_to_page == {}

    def test_format_feedback_empty_graph(self):
        """No graph → empty feedback string."""
        c = StructureCollector(page_budget=5, repo_root="/x")
        assert c._format_symbol_coverage_feedback() == ""
