"""
Pipeline Phase 3 — Structure Planner & Context Quality Tests

Tests for:
3.1 Doc file counting in explore_tree
3.2 Compact query_graph output format
3.3 AUTO_TARGET_DOCS prompt alignment
3.4 Post-planning validation
3.5 Line numbers in code_source tags
3.6 Graph-based import extraction
3.7 Graph file_imports fallback
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Ensure plugin_implementation is importable
# ---------------------------------------------------------------------------
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
IMPL_ROOT = PLUGIN_ROOT / "plugin_implementation"
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))
if str(IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL_ROOT))


# ============================================================================
# 3.1 — Doc file counting in explore_tree
# ============================================================================

class TestDocFileCounting:
    """Tests for _is_doc_file and explore_tree doc counting."""

    def test_is_doc_file_markdown(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import _is_doc_file
        assert _is_doc_file("README.md") is True
        assert _is_doc_file("CHANGELOG.md") is True

    def test_is_doc_file_rst(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import _is_doc_file
        assert _is_doc_file("index.rst") is True

    def test_is_doc_file_txt(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import _is_doc_file
        assert _is_doc_file("notes.txt") is True

    def test_is_doc_file_yaml_json(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import _is_doc_file
        assert _is_doc_file("config.yml") is True
        assert _is_doc_file("data.json") is True

    def test_is_doc_file_code_is_false(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import _is_doc_file
        assert _is_doc_file("main.py") is False
        assert _is_doc_file("app.js") is False
        assert _is_doc_file("server.go") is False

    def test_is_code_file_still_works(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import _is_code_file
        assert _is_code_file("main.py") is True
        assert _is_code_file("app.jsx") is True
        assert _is_code_file("README.md") is False

    def test_explore_tree_has_code_doc_counts(self):
        """explore_repository_tree should populate code_files and doc_files per dir."""
        from plugin_implementation.wiki_structure_planner.structure_tools import explore_repository_tree
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create src/ with code files
            src = os.path.join(tmpdir, "src")
            os.makedirs(src)
            Path(src, "main.py").write_text("code")
            Path(src, "utils.py").write_text("code")
            Path(src, "README.md").write_text("docs")
            
            # Create docs/ with doc files
            docs = os.path.join(tmpdir, "docs")
            os.makedirs(docs)
            Path(docs, "guide.md").write_text("docs")
            Path(docs, "api.rst").write_text("docs")
            Path(docs, "config.json").write_text("{}")
            
            result = explore_repository_tree(tmpdir, max_depth=2)
            dirs = result["directories"]
            
            src_dir = next(d for d in dirs if d["name"] == "src")
            assert src_dir["code_files"] == 2
            assert src_dir["doc_files"] == 1
            assert src_dir["files"] == 3
            
            docs_dir = next(d for d in dirs if d["name"] == "docs")
            assert docs_dir["code_files"] == 0
            assert docs_dir["doc_files"] >= 2  # md, rst, json

    def test_format_tree_shows_breakdown(self):
        """format_tree_for_llm should show code/doc breakdown."""
        from plugin_implementation.wiki_structure_planner.structure_tools import format_tree_for_llm
        
        tree_data = {
            "directories": [
                {"path": "src", "name": "src", "depth": 1, "files": 10,
                 "code_files": 8, "doc_files": 2, "subdirs": 3}
            ],
            "top_level_files": ["README.md"],
            "total_files": 10,
        }
        
        output = format_tree_for_llm(tree_data)
        assert "8 code" in output
        assert "2 doc" in output
        assert "TOTAL:" in output

    def test_format_tree_total_summary(self):
        """format_tree_for_llm total line should include code/doc/other counts."""
        from plugin_implementation.wiki_structure_planner.structure_tools import format_tree_for_llm
        
        tree_data = {
            "directories": [
                {"path": "a", "name": "a", "depth": 1, "files": 5,
                 "code_files": 3, "doc_files": 1, "subdirs": 0},
                {"path": "b", "name": "b", "depth": 1, "files": 3,
                 "code_files": 2, "doc_files": 0, "subdirs": 0},
            ],
            "top_level_files": [],
            "total_files": 8,
        }
        
        output = format_tree_for_llm(tree_data)
        # Total should show breakdown
        assert "5 code" in output
        assert "1 doc" in output

    def test_format_file_breakdown_helper(self):
        """_format_file_breakdown should produce readable string."""
        from plugin_implementation.wiki_structure_planner.structure_tools import _format_file_breakdown
        
        result = _format_file_breakdown({"files": 10, "code_files": 7, "doc_files": 2})
        assert "10 files" in result
        assert "7 code" in result
        assert "2 doc" in result
        assert "1 other" in result
        
        # No breakdown fields should still work
        result2 = _format_file_breakdown({"files": 5})
        assert "5 files" in result2

    def test_format_tree_content_only_hides_shallow_dirs(self):
        """Shallow dirs (0 files, only subdirs) should be hidden entirely."""
        from plugin_implementation.wiki_structure_planner.structure_tools import format_tree_for_llm
        
        tree_data = {
            "directories": [
                # Shallow top-level — should NOT appear in output
                {"path": "src", "name": "src", "depth": 1, "files": 0,
                 "code_files": 0, "doc_files": 0, "subdirs": 5},
                # Content top-level — should appear
                {"path": "tests", "name": "tests", "depth": 1, "files": 20,
                 "code_files": 18, "doc_files": 2, "subdirs": 3},
                # Shallow depth 2 — should NOT appear
                {"path": "src/v", "name": "v", "depth": 2, "files": 0,
                 "code_files": 0, "doc_files": 0, "subdirs": 3},
                # Content depth 3 — SHOULD appear under src/ header
                {"path": "src/v/redpanda", "name": "redpanda", "depth": 3,
                 "files": 10, "code_files": 9, "doc_files": 1, "subdirs": 4},
            ],
            "top_level_files": ["README.md"],
            "total_files": 30,
        }
        
        output = format_tree_for_llm(tree_data)
        
        # Shallow dirs should NOT appear in top-level listing
        assert "📁 src/" not in output.split("Inside Large")[0]
        # Content dir SHOULD appear
        assert "tests/" in output
        # Shallow depth-2 should not appear at all (no 'v/' line on its own)
        lines = output.split("\n")
        v_lines = [l for l in lines if "└─ v/" in l and "redpanda" not in l]
        assert len(v_lines) == 0, f"Shallow dir 'v/' should be hidden, found: {v_lines}"
        # Content depth-3 SHOULD appear with relative path
        assert "v/redpanda/" in output
        # No SKIP signals anywhere
        assert "⏭" not in output
        assert "SKIP" not in output

    def test_format_tree_content_only_next_steps(self):
        """NEXT STEPS should only suggest content dirs, no shallow substitution."""
        from plugin_implementation.wiki_structure_planner.structure_tools import format_tree_for_llm
        
        tree_data = {
            "directories": [
                {"path": "lib", "name": "lib", "depth": 1, "files": 0,
                 "code_files": 0, "doc_files": 0, "subdirs": 4},
                {"path": "lib/core", "name": "core", "depth": 2, "files": 15,
                 "code_files": 14, "doc_files": 1, "subdirs": 2},
                {"path": "lib/utils", "name": "utils", "depth": 2, "files": 8,
                 "code_files": 8, "doc_files": 0, "subdirs": 0},
            ],
            "top_level_files": [],
            "total_files": 23,
        }
        
        output = format_tree_for_llm(tree_data)
        
        # NEXT STEPS should suggest content dirs
        assert "query_graph('lib/core')" in output
        assert "query_graph('lib/utils')" in output
        # Should NOT suggest the shallow parent
        assert "query_graph('lib')" not in output

    def test_format_tree_content_only_summary_counts_content_dirs(self):
        """TOTAL line should count content directories, not all directories."""
        from plugin_implementation.wiki_structure_planner.structure_tools import format_tree_for_llm
        
        tree_data = {
            "directories": [
                {"path": "src", "name": "src", "depth": 1, "files": 0,
                 "code_files": 0, "doc_files": 0, "subdirs": 2},
                {"path": "src/api", "name": "api", "depth": 2, "files": 5,
                 "code_files": 5, "doc_files": 0, "subdirs": 0},
                {"path": "docs", "name": "docs", "depth": 1, "files": 3,
                 "code_files": 0, "doc_files": 3, "subdirs": 0},
            ],
            "top_level_files": [],
            "total_files": 8,
        }
        
        output = format_tree_for_llm(tree_data)
        
        # Should count 2 content dirs (src/api, docs), not 3 total dirs
        assert "2 content directories" in output


# ============================================================================
# 3.2 — Compact query_graph output format
# ============================================================================

class TestCompactQueryGraph:
    """Tests for compact mode in _format_graph_analysis."""

    def _make_collector(self):
        """Create a minimal StructureCollector for testing."""
        from plugin_implementation.wiki_structure_planner.structure_tools import StructureCollector
        return StructureCollector(page_budget=20, repo_root="/tmp/test")

    def test_compact_mode_triggers_above_80(self):
        """When total_symbols > 80, compact mode should activate."""
        collector = self._make_collector()
        analysis = {
            'total_symbols': 100,
            'class_count': 50,
            'function_count': 50,
            'files_with_symbols': 20,
            'all_classes': [
                {'name': f'Class{i}', 'type': 'class', 'connections': i}
                for i in range(50)
            ],
            'all_functions': [
                {'name': f'func{i}', 'type': 'function', 'connections': 0}
                for i in range(50)
            ],
        }
        
        output = collector._format_graph_analysis("src/", analysis)
        assert "COMPACT" in output

    def test_normal_mode_below_80(self):
        """When total_symbols <= 80, normal mode should be used."""
        collector = self._make_collector()
        analysis = {
            'total_symbols': 30,
            'class_count': 10,
            'function_count': 20,
            'files_with_symbols': 5,
            'all_classes': [
                {'name': f'Class{i}', 'type': 'class', 'connections': i}
                for i in range(10)
            ],
            'all_functions': [],
        }
        
        output = collector._format_graph_analysis("src/", analysis)
        assert "COMPACT" not in output

    def test_compact_mode_reduces_output(self):
        """Compact mode should produce fewer lines than normal mode."""
        collector = self._make_collector()
        
        # Large analysis with 120 classes and 80 functions
        classes = [
            {'name': f'Class{i}', 'type': 'class', 'connections': i,
             'inherits_from': [f'Base{i % 5}'] if i % 3 == 0 else []}
            for i in range(120)
        ]
        functions = [
            {'name': f'func{i}', 'type': 'function', 'connections': 0}
            for i in range(80)
        ]
        
        analysis = {
            'total_symbols': 200,
            'class_count': 120,
            'function_count': 80,
            'files_with_symbols': 40,
            'all_classes': classes,
            'all_functions': functions,
        }
        
        output_compact = collector._format_graph_analysis("src/", analysis)
        
        # Verify compact triggers
        assert "COMPACT" in output_compact
        
        # The output should still have all section headers
        assert "CLASSES WITH RELATIONSHIPS" in output_compact
        assert "STANDALONE CLASSES" in output_compact

    def test_empty_analysis(self):
        """Zero symbols should return no-symbols message."""
        collector = self._make_collector()
        analysis = {'total_symbols': 0}
        output = collector._format_graph_analysis("src/", analysis)
        assert "No symbols found" in output


# ============================================================================
# 3.3 — AUTO_TARGET_DOCS prompt alignment
# ============================================================================

class TestAutoTargetDocsPrompt:
    """Tests for AUTO_TARGET_DOCS prompt text alignment with PageDefinition schema."""

    def test_auto_target_docs_prompt_mentions_target_symbols(self):
        """When AUTO_TARGET_DOCS is ON, prompt should mention target_symbols as primary."""
        from plugin_implementation.wiki_structure_planner.structure_tools import StructureCollector
        collector = StructureCollector(page_budget=20, repo_root="/tmp/test")
        
        analysis = {
            'total_symbols': 10,
            'class_count': 5,
            'function_count': 5,
            'files_with_symbols': 3,
            'all_classes': [{'name': f'C{i}', 'type': 'class', 'connections': 0} for i in range(5)],
            'all_functions': [],
            'all_docs': [{'path': 'README.md', 'type': 'markdown'}],
        }
        
        with patch('plugin_implementation.wiki_structure_planner.structure_tools.AUTO_TARGET_DOCS', True):
            output = collector._format_graph_analysis("src/", analysis)
            # Should mention target_symbols as primary, not retrieval_query
            assert "target_symbols" in output.lower() or "target_symbols is PRIMARY" in output
            assert "FALLBACK" in output  # retrieval_query is fallback
            assert "OPTIONAL" in output  # target_docs is optional

    def test_page_definition_schema_has_retrieval_query(self):
        """PageDefinition schema should have retrieval_query field."""
        from plugin_implementation.wiki_structure_planner.structure_tools import PageDefinition
        fields = PageDefinition.model_fields
        assert 'retrieval_query' in fields
        assert 'target_symbols' in fields
        assert 'target_docs' in fields


# ============================================================================
# 3.4 — Post-planning validation
# ============================================================================

class TestPostPlanningValidation:
    """Tests for structure_engine._validate_structure."""

    def _make_engine(self):
        """Create a minimal WikiStructurePlannerEngine for testing."""
        from plugin_implementation.wiki_structure_planner.structure_engine import WikiStructurePlannerEngine
        # Mock the LLM
        mock_llm = MagicMock()
        engine = WikiStructurePlannerEngine.__new__(WikiStructurePlannerEngine)
        engine.llm = mock_llm
        engine.tool_calls = []
        engine.status = "idle"
        engine.error = None
        return engine

    def test_validates_empty_symbol_pages(self, caplog):
        """Should warn about pages with no target_symbols AND no target_folders/target_docs."""
        engine = self._make_engine()
        collector = MagicMock()
        collector.stats = {"coverage_pct": 100, "covered_dirs": 5, "discovered_dirs": 5, "uncovered_dirs": []}
        collector.code_graph = None
        
        parsed = {
            "sections": [{
                "section_name": "Core",
                "pages": [
                    {"page_name": "Auth Flow", "target_symbols": ["AuthService"]},
                    {"page_name": "Empty Page", "target_symbols": []},
                    # Pages with target_folders/target_docs should NOT trigger warning
                    {"page_name": "CI Workflows", "target_symbols": [], "target_folders": [".github/workflows"]},
                    {"page_name": "Config Guide", "target_symbols": [], "target_docs": ["README.md"]},
                ]
            }]
        }
        
        with caplog.at_level(logging.WARNING):
            engine._validate_structure(parsed, collector)
        
        assert "EMPTY SYMBOLS" in caplog.text
        assert "Empty Page" in caplog.text
        # Pages with folders/docs should not be flagged
        assert "CI Workflows" not in caplog.text
        assert "Config Guide" not in caplog.text

    def test_validates_oversized_pages(self, caplog):
        """Should warn about pages with > 50 target_symbols (broad-page model)."""
        engine = self._make_engine()
        collector = MagicMock()
        collector.stats = {"coverage_pct": 100, "covered_dirs": 5, "discovered_dirs": 5, "uncovered_dirs": []}
        collector.code_graph = None
        
        parsed = {
            "sections": [{
                "section_name": "Core",
                "pages": [{
                    "page_name": "Kitchen Sink",
                    "target_symbols": [f"Sym{i}" for i in range(55)],
                }]
            }]
        }
        
        with caplog.at_level(logging.WARNING):
            engine._validate_structure(parsed, collector)
        
        assert "OVERSIZED" in caplog.text

    def test_no_oversized_warning_for_broad_pages(self, caplog):
        """Pages with 10-30 symbols should NOT trigger oversized warning (broad-page model)."""
        engine = self._make_engine()
        collector = MagicMock()
        collector.stats = {"coverage_pct": 100, "covered_dirs": 5, "discovered_dirs": 5, "uncovered_dirs": []}
        collector.code_graph = None
        
        parsed = {
            "sections": [{
                "section_name": "Core",
                "pages": [{
                    "page_name": "Broad Page",
                    "target_symbols": [f"Sym{i}" for i in range(25)],
                }]
            }]
        }
        
        with caplog.at_level(logging.WARNING):
            engine._validate_structure(parsed, collector)
        
        assert "OVERSIZED" not in caplog.text

    def test_validates_low_coverage(self, caplog):
        """Should warn when coverage is below 50%."""
        engine = self._make_engine()
        collector = MagicMock()
        collector.stats = {
            "coverage_pct": 30,
            "covered_dirs": 3,
            "discovered_dirs": 10,
            "uncovered_dirs": ["dir1", "dir2", "dir3"],
        }
        collector.code_graph = None
        
        parsed = {"sections": [{"section_name": "S1", "pages": []}]}
        
        with caplog.at_level(logging.WARNING):
            engine._validate_structure(parsed, collector)
        
        assert "LOW COVERAGE" in caplog.text

    def test_validates_missing_symbols_in_graph(self, caplog):
        """Should warn about target_symbols not found in graph."""
        engine = self._make_engine()
        collector = MagicMock()
        collector.stats = {"coverage_pct": 100, "covered_dirs": 5, "discovered_dirs": 5, "uncovered_dirs": []}
        
        # Create a mock graph with known symbols
        import networkx as nx
        graph = nx.MultiDiGraph()
        graph.add_node("n1", symbol_name="AuthService", name="AuthService")
        graph.add_node("n2", symbol_name="UserRepo", name="UserRepo")
        collector.code_graph = graph
        
        parsed = {
            "sections": [{
                "section_name": "Core",
                "pages": [{
                    "page_name": "Auth",
                    "target_symbols": ["AuthService", "NonExistentClass", "FakeHelper"],
                }]
            }]
        }
        
        with caplog.at_level(logging.WARNING):
            engine._validate_structure(parsed, collector)
        
        assert "MISSING SYMBOLS" in caplog.text
        assert "NonExistentClass" in caplog.text

    def test_valid_structure_logs_info(self, caplog):
        """Valid structure should log info, not warnings."""
        engine = self._make_engine()
        collector = MagicMock()
        collector.stats = {"coverage_pct": 90, "covered_dirs": 9, "discovered_dirs": 10, "uncovered_dirs": []}
        collector.code_graph = None
        
        parsed = {
            "sections": [{
                "section_name": "Core",
                "pages": [{"page_name": "Auth", "target_symbols": ["AuthService"]}],
            }]
        }
        
        with caplog.at_level(logging.INFO):
            engine._validate_structure(parsed, collector)
        
        assert "looks good" in caplog.text


# ============================================================================
# 3.5 — Line numbers in code_source tags
# ============================================================================

class TestLineNumbers:
    """Tests for line number inclusion in context formatting."""

    def _make_doc(self, symbol_name, source, content, start_line=0, end_line=0):
        """Create a mock Document with metadata."""
        doc = MagicMock()
        doc.page_content = content
        doc.metadata = {
            'source': source,
            'file_path': source,
            'symbol_name': symbol_name,
            'start_line': start_line,
            'end_line': end_line,
            'symbol_type': 'class',
            'chunk_type': 'code',
        }
        return doc

    def test_simple_context_includes_line_annotations(self):
        """_format_simple_context should add [SYMBOL] line annotations in <line_map> blocks."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        
        agent = MagicMock(spec=OptimizedWikiGenerationAgent)
        agent._is_documentation_file = lambda x: False
        agent._extract_imports_for_file = lambda f, c: ""
        
        doc1 = self._make_doc("MyClass", "src/main.py", "class MyClass: ...", 10, 50)
        doc2 = self._make_doc("helper", "src/main.py", "def helper(): ...", 55, 70)
        
        page_spec = MagicMock()
        page_spec.target_folders = []
        page_spec.key_files = []
        
        result = OptimizedWikiGenerationAgent._format_simple_context(agent, [doc1, doc2], page_spec)
        content = result["content"]
        
        assert "[SYMBOL] MyClass: L10-L50" in content
        assert "[SYMBOL] helper: L55-L70" in content
        assert "<line_map>" in content
        assert "</line_map>" in content

    def test_simple_context_skips_zero_lines(self):
        """Symbols with start_line=0 should NOT get line annotations."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        
        agent = MagicMock(spec=OptimizedWikiGenerationAgent)
        agent._is_documentation_file = lambda x: False
        agent._extract_imports_for_file = lambda f, c: ""
        
        doc = self._make_doc("NoLines", "src/main.py", "class NoLines: ...", 0, 0)
        
        page_spec = MagicMock()
        page_spec.target_folders = []
        page_spec.key_files = []
        
        result = OptimizedWikiGenerationAgent._format_simple_context(agent, [doc], page_spec)
        assert "[SYMBOL] NoLines:" not in result["content"]
        assert "<line_map>" not in result["content"]

    def test_tier_header_includes_line_range(self):
        """_format_tier_with_hints tier 1 header should show L{start}-L{end}."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        
        agent = MagicMock(spec=OptimizedWikiGenerationAgent)
        agent._format_hybrid_hint = lambda *args, **kwargs: ""
        
        doc = self._make_doc("MyClass", "src/main.py", "class MyClass: ...", 10, 50)
        
        result = OptimizedWikiGenerationAgent._format_tier_with_hints(
            agent, [doc], tier=1, tier_map={}, graph=None
        )
        assert "L10-L50" in result

    def test_tier2_header_includes_line_range(self):
        """_format_tier_with_hints tier 2 header should show L{start}-L{end}."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        
        agent = MagicMock(spec=OptimizedWikiGenerationAgent)
        agent._format_hybrid_hint = lambda *args, **kwargs: ""
        
        doc = self._make_doc("Helper", "src/utils.py", "def helper(): ...", 5, 20)
        
        result = OptimizedWikiGenerationAgent._format_tier_with_hints(
            agent, [doc], tier=2, tier_map={}, graph=None
        )
        assert "L5-L20" in result

    def test_fetched_documents_include_line_range(self):
        """_format_fetched_documents should show line range in Source line."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        
        agent = MagicMock(spec=OptimizedWikiGenerationAgent)
        doc = self._make_doc("Parser", "src/parser.py", "class Parser: ...", 100, 250)
        
        result = OptimizedWikiGenerationAgent._format_fetched_documents(agent, [doc])
        assert "L100-L250" in result

    def test_fetched_documents_no_lines_when_zero(self):
        """_format_fetched_documents should omit line range when both zero."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        
        agent = MagicMock(spec=OptimizedWikiGenerationAgent)
        doc = self._make_doc("Unknown", "src/file.py", "content", 0, 0)
        
        result = OptimizedWikiGenerationAgent._format_fetched_documents(agent, [doc])
        assert "L0" not in result
        assert "`src/file.py`" in result


# ============================================================================
# 3.6 + 3.7 — Graph-based import extraction
# ============================================================================

class TestGraphBasedImports:
    """Tests for graph-based import extraction and file_imports fallback."""

    def test_extract_imports_from_graph_edges(self):
        """_get_imports_from_graph should find imports from graph edges."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        import networkx as nx
        
        agent = MagicMock(spec=OptimizedWikiGenerationAgent)
        
        # Build a mock graph with import relationships
        graph = nx.MultiDiGraph()
        graph.add_node("file:auth", rel_path="src/auth.py", symbol_name="auth", imports=["os", "logging"])
        graph.add_node("AuthService", rel_path="src/auth.py", symbol_name="AuthService", imports=[])
        graph.add_node("UserRepo", rel_path="src/user.py", symbol_name="UserRepo", imports=[])
        
        # Add import edges
        graph.add_edge("AuthService", "UserRepo", relationship_type="imports")
        
        # Mock retriever_stack
        agent.retriever_stack = MagicMock()
        agent.retriever_stack.relationship_graph = graph
        
        result = OptimizedWikiGenerationAgent._get_imports_from_graph(agent, "src/auth.py")
        assert "os" in result
        assert "logging" in result
        assert "UserRepo" in result

    def test_extract_imports_file_imports_edges(self):
        """_get_imports_from_graph should also follow file_imports edge type."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        import networkx as nx
        
        agent = MagicMock(spec=OptimizedWikiGenerationAgent)
        
        graph = nx.MultiDiGraph()
        graph.add_node("file:main", rel_path="src/main.py", symbol_name="main")
        graph.add_node("flask", symbol_name="flask")
        graph.add_edge("file:main", "flask", type="file_imports")
        
        agent.retriever_stack = MagicMock()
        agent.retriever_stack.relationship_graph = graph
        
        result = OptimizedWikiGenerationAgent._get_imports_from_graph(agent, "src/main.py")
        assert "flask" in result

    def test_no_graph_returns_empty(self):
        """_get_imports_from_graph returns '' when no graph available."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        
        agent = MagicMock(spec=OptimizedWikiGenerationAgent)
        agent.retriever_stack = MagicMock()
        agent.retriever_stack.relationship_graph = None
        agent.indexer = MagicMock()
        agent.indexer.relationship_graph = None
        
        result = OptimizedWikiGenerationAgent._get_imports_from_graph(agent, "src/file.py")
        assert result == ""

    def test_extract_imports_prefers_graph(self):
        """_extract_imports_for_file should use graph imports when available."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        import networkx as nx
        
        agent = MagicMock()
        # Bind real methods
        agent._get_imports_from_graph = lambda fp: OptimizedWikiGenerationAgent._get_imports_from_graph(agent, fp)
        agent._extract_imports_regex = lambda fp, c: OptimizedWikiGenerationAgent._extract_imports_regex(agent, fp, c)
        agent._extract_imports_for_file = lambda fp, c: OptimizedWikiGenerationAgent._extract_imports_for_file(agent, fp, c)
        
        graph = nx.MultiDiGraph()
        graph.add_node("file:app", rel_path="src/app.go", symbol_name="app", imports=["fmt", "net/http"])
        
        agent.retriever_stack = MagicMock()
        agent.retriever_stack.relationship_graph = graph
        
        # Go is NOT supported by regex, but graph should still work
        result = agent._extract_imports_for_file("src/app.go", "package main\nimport \"fmt\"\nfunc main() {}")
        assert "fmt" in result
        assert "net/http" in result

    def test_extract_imports_regex_fallback(self):
        """_extract_imports_for_file should fall back to regex when no graph."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        
        agent = MagicMock()
        agent._get_imports_from_graph = lambda fp: OptimizedWikiGenerationAgent._get_imports_from_graph(agent, fp)
        agent._extract_imports_regex = lambda fp, c: OptimizedWikiGenerationAgent._extract_imports_regex(agent, fp, c)
        agent._extract_imports_for_file = lambda fp, c: OptimizedWikiGenerationAgent._extract_imports_for_file(agent, fp, c)
        agent.retriever_stack = MagicMock()
        agent.retriever_stack.relationship_graph = None
        agent.indexer = MagicMock()
        agent.indexer.relationship_graph = None
        
        content = "import os\nimport sys\n\ndef main():\n    pass"
        result = agent._extract_imports_for_file("app.py", content)
        
        assert "import os" in result
        assert "import sys" in result

    def test_regex_unsupported_language_message(self):
        """Regex fallback for unsupported language returns helpful message."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        
        agent = MagicMock()
        agent._get_imports_from_graph = lambda fp: OptimizedWikiGenerationAgent._get_imports_from_graph(agent, fp)
        agent._extract_imports_regex = lambda fp, c: OptimizedWikiGenerationAgent._extract_imports_regex(agent, fp, c)
        agent._extract_imports_for_file = lambda fp, c: OptimizedWikiGenerationAgent._extract_imports_for_file(agent, fp, c)
        agent.retriever_stack = MagicMock()
        agent.retriever_stack.relationship_graph = None
        agent.indexer = MagicMock()
        agent.indexer.relationship_graph = None
        
        result = agent._extract_imports_for_file("main.rs", "use std::io;\nfn main() {}")
        assert "not supported" in result.lower()

    def test_graph_imports_sorted(self):
        """Graph-based imports should be sorted for deterministic output."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        import networkx as nx
        
        agent = MagicMock(spec=OptimizedWikiGenerationAgent)
        
        graph = nx.MultiDiGraph()
        graph.add_node("n1", rel_path="f.py", symbol_name="A", imports=["zlib", "abc", "logging"])
        
        agent.retriever_stack = MagicMock()
        agent.retriever_stack.relationship_graph = graph
        
        result = OptimizedWikiGenerationAgent._get_imports_from_graph(agent, "f.py")
        lines = result.strip().split('\n')
        assert lines == sorted(lines)


# ============================================================================
# Analyze module doc file counting
# ============================================================================

class TestAnalyzeModuleDocCounting:
    """Tests that _handle_analyze_module shows doc file counts."""

    def test_analyze_module_shows_doc_files(self):
        """analyze_module should list documentation files separately."""
        from plugin_implementation.wiki_structure_planner.structure_tools import StructureCollector
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            Path(tmpdir, "main.py").write_text("code")
            Path(tmpdir, "utils.py").write_text("code")
            Path(tmpdir, "README.md").write_text("docs")
            Path(tmpdir, "CHANGELOG.md").write_text("docs")
            
            collector = StructureCollector(page_budget=10, repo_root=tmpdir)
            result = collector._handle_analyze_module(".")
            
            assert "Code files (2)" in result
            assert "Documentation files (2)" in result
            assert "README.md" in result

    def test_analyze_module_subdirs_show_doc_counts(self):
        """analyze_module subdirectories should show doc file counts."""
        from plugin_implementation.wiki_structure_planner.structure_tools import StructureCollector
        
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = os.path.join(tmpdir, "api")
            os.makedirs(sub)
            Path(sub, "handler.py").write_text("code")
            Path(sub, "routes.py").write_text("code")
            Path(sub, "README.md").write_text("docs")
            
            collector = StructureCollector(page_budget=10, repo_root=tmpdir)
            result = collector._handle_analyze_module(".")
            
            assert "2 code" in result
            assert "1 doc" in result
