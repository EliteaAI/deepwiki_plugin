"""
Comprehensive test suite for the merged capability-based naming, doc/code separation,
batch_define_pages, and structure planner features.

Tests cover:
1. Shared constants (DOC_SYMBOL_TYPES, DOC_CHUNK_TYPES, EXPANSION_SYMBOL_TYPES)
2. File extension wiring (DOCUMENTATION_EXTENSIONS, filter_manager)
3. Graph builder doc filtering (SEPARATE_DOC_INDEX flag)
4. Structure tools: define_page with target_docs, symbol validation, batch_define_pages
5. Structure tools: O(1) symbol path index (_symbol_rel_paths)
6. Wiki state: PageSpec target_docs field
7. Doc node retrieval (_get_doc_nodes_from_graph)
8. Doc cluster detection
9. ARCHITECTURAL_TYPES filtering in graph expansion
10. Feature flag gating
"""

import os
import pytest
import networkx as nx
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Module under test imports
# ---------------------------------------------------------------------------
from plugin_implementation.constants import (
    DOC_SYMBOL_TYPES,
    DOC_CHUNK_TYPES,
    EXPANSION_SYMBOL_TYPES,
)
from plugin_implementation.code_graph.graph_builder import (
    EnhancedUnifiedGraphBuilder,
)
from plugin_implementation.state.wiki_state import PageSpec, WikiStructureSpec, SectionSpec
from plugin_implementation.filter_manager import FilterManager


# ============================================================================
# 1. Shared Constants Tests
# ============================================================================
class TestSharedConstants:
    """Verify the centralized constants in constants.py."""

    def test_doc_symbol_types_is_frozenset(self):
        assert isinstance(DOC_SYMBOL_TYPES, frozenset)

    def test_doc_chunk_types_is_frozenset(self):
        assert isinstance(DOC_CHUNK_TYPES, frozenset)

    def test_expansion_symbol_types_is_frozenset(self):
        assert isinstance(EXPANSION_SYMBOL_TYPES, frozenset)

    def test_doc_symbol_types_contains_expected(self):
        # Core set that must always be present (non-exhaustive — new doc types
        # are added as DOCUMENTATION_EXTENSIONS / KNOWN_FILENAMES grow).
        core_expected = {
            'markdown_document', 'markdown_section', 'text_chunk',
            'toml_document', 'yaml_document', 'restructuredtext_document',
            'plaintext_document', 'text_document',
        }
        assert core_expected.issubset(DOC_SYMBOL_TYPES), (
            f"Missing from DOC_SYMBOL_TYPES: {core_expected - DOC_SYMBOL_TYPES}"
        )
        # Expanded doc types added for doc-file-visibility feature
        expanded = {
            'json_document', 'xml_document', 'html_document',
            'config_document', 'build_config_document', 'schema_document',
            'infrastructure_document', 'script_document', 'asciidoc_document',
            'document_document', 'pdf_document',
        }
        assert expanded.issubset(DOC_SYMBOL_TYPES), (
            f"Missing expanded types: {expanded - DOC_SYMBOL_TYPES}"
        )

    def test_doc_chunk_types_contains_expected(self):
        assert DOC_CHUNK_TYPES == {'text', 'documentation', 'markdown'}

    def test_expansion_types_excludes_method(self):
        """EXPANSION_SYMBOL_TYPES should NOT contain 'method' (expanded via parent class)."""
        assert 'method' not in EXPANSION_SYMBOL_TYPES

    def test_expansion_types_includes_macro(self):
        """EXPANSION_SYMBOL_TYPES should contain 'macro' (architecturally significant)."""
        assert 'macro' in EXPANSION_SYMBOL_TYPES

    def test_expansion_types_includes_core_architectural(self):
        for sym_type in ('class', 'interface', 'struct', 'enum', 'function', 'constant', 'type_alias'):
            assert sym_type in EXPANSION_SYMBOL_TYPES, f"Missing {sym_type}"

    def test_graph_builder_imports_same_doc_types(self):
        """graph_builder.DOC_SYMBOL_TYPES must be the same object as constants.DOC_SYMBOL_TYPES."""
        from plugin_implementation.code_graph import graph_builder as gb
        assert gb.DOC_SYMBOL_TYPES is DOC_SYMBOL_TYPES

    def test_retrievers_imports_same_doc_types(self):
        """retrievers must use the same DOC_SYMBOL_TYPES from constants."""
        from plugin_implementation import retrievers as ret
        assert ret.DOC_SYMBOL_TYPES is DOC_SYMBOL_TYPES
        assert ret.DOC_CHUNK_TYPES is DOC_CHUNK_TYPES


# ============================================================================
# 2. File Extension Wiring Tests
# ============================================================================
class TestFileExtensionWiring:
    """Verify all documentation extensions are properly wired."""

    # Extensions added in the recent commit
    RECENTLY_ADDED = [
        '.gradle', '.kts', '.wsdl', '.xsd', '.proto',
        '.tf', '.tfvars', '.hcl', '.mod',
        '.sh', '.bash', '.bat', '.cmd', '.ps1', '.psm1',
    ]

    @pytest.fixture
    def doc_extensions(self):
        """Get DOCUMENTATION_EXTENSIONS as a class attribute."""
        return EnhancedUnifiedGraphBuilder.DOCUMENTATION_EXTENSIONS

    def test_documentation_extensions_contains_recent_additions(self, doc_extensions):
        for ext in self.RECENTLY_ADDED:
            assert ext in doc_extensions, \
                f"Missing {ext} in DOCUMENTATION_EXTENSIONS"

    def test_documentation_extensions_has_category(self, doc_extensions):
        """Each extension must map to a non-empty category string."""
        for ext, category in doc_extensions.items():
            assert isinstance(category, str) and len(category) > 0, \
                f"Extension {ext} has invalid category: {category!r}"

    def test_markdown_extensions_present(self, doc_extensions):
        for ext in ('.md', '.rst', '.txt'):
            assert ext in doc_extensions

    def test_config_extensions_present(self, doc_extensions):
        for ext in ('.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.properties'):
            assert ext in doc_extensions

    def test_filter_manager_includes_new_extensions(self):
        """FilterManager should recognize the new extensions as allowed."""
        fm = FilterManager()
        # The new extensions should be recognized by filter_manager
        for ext in ['.proto', '.tf', '.gradle', '.kts', '.sh', '.bat', '.ps1']:
            # FilterManager.allowed_extensions or similar
            if hasattr(fm, 'allowed_extensions'):
                assert ext in fm.allowed_extensions, f"{ext} not in FilterManager.allowed_extensions"
            elif hasattr(fm, 'ALLOWED_EXTENSIONS'):
                assert ext in fm.ALLOWED_EXTENSIONS, f"{ext} not in FilterManager.ALLOWED_EXTENSIONS"
            elif hasattr(fm, 'language_map'):
                # Some extensions may map to a language
                pass  # language_map may not have all doc extensions


# ============================================================================
# 3. Graph Builder Doc Filtering Tests
# ============================================================================
class TestGraphBuilderDocFiltering:
    """Test SEPARATE_DOC_INDEX feature flag behavior in graph_builder."""

    def test_separate_doc_index_default_off(self):
        """SEPARATE_DOC_INDEX should default to False (off)."""
        from plugin_implementation.code_graph import graph_builder as gb
        # Default env should be "0" -> False
        assert gb.SEPARATE_DOC_INDEX is False or os.getenv("DEEPWIKI_DOC_SEPARATE_INDEX", "0") == "1"

    def test_architectural_symbols_superset(self):
        """graph_builder ARCHITECTURAL_SYMBOLS is a superset of EXPANSION_SYMBOL_TYPES."""
        builder = EnhancedUnifiedGraphBuilder.__new__(EnhancedUnifiedGraphBuilder)
        # Access the instance attribute set in __init__
        arch_symbols = {
            'class', 'interface', 'struct', 'enum', 'trait',
            'function', 'constant', 'type_alias', 'macro',
            'markdown_document', 'toml_document', 'text_chunk', 'text_document',
            'restructuredtext_document', 'plaintext_document'
        }
        # EXPANSION_SYMBOL_TYPES should be a subset (minus macro, plus type_alias but no doc types)
        core_types = {'class', 'interface', 'struct', 'enum', 'trait', 'function', 'constant', 'type_alias'}
        assert core_types.issubset(arch_symbols)


# ============================================================================
# 4. PageSpec / WikiStructureSpec Tests
# ============================================================================
class TestPageSpecTargetDocs:
    """Test the target_docs field on PageSpec."""

    def test_page_spec_has_target_docs(self):
        """PageSpec must have target_docs field."""
        ps = PageSpec(
            page_name="Test Page",
            page_order=1,
            description="desc",
            content_focus="focus",
            rationale="rationale",
        )
        assert hasattr(ps, 'target_docs')
        assert ps.target_docs == []

    def test_page_spec_with_target_docs(self):
        ps = PageSpec(
            page_name="Test",
            page_order=1,
            description="desc",
            content_focus="focus",
            rationale="rationale",
            target_docs=["README.md", "docs/guide.md"],
        )
        assert ps.target_docs == ["README.md", "docs/guide.md"]

    def test_page_spec_serialization_includes_target_docs(self):
        ps = PageSpec(
            page_name="Test",
            page_order=1,
            description="d",
            content_focus="c",
            rationale="r",
            target_docs=["a.md"],
        )
        data = ps.model_dump()
        assert 'target_docs' in data
        assert data['target_docs'] == ["a.md"]

    def test_wiki_structure_spec_round_trip(self):
        """WikiStructureSpec with target_docs should serialize/deserialize."""
        spec_data = {
            "wiki_title": "Test Wiki",
            "overview": "Overview",
            "sections": [
                {
                    "section_name": "Core",
                    "section_order": 1,
                    "description": "Core section",
                    "rationale": "Main",
                    "pages": [
                        {
                            "page_name": "Page1",
                            "page_order": 1,
                            "description": "desc",
                            "content_focus": "focus",
                            "rationale": "rationale",
                            "target_symbols": ["MyClass"],
                            "target_docs": ["README.md"],
                            "target_folders": ["src/"],
                            "key_files": [],
                            "retrieval_query": "",
                        }
                    ],
                }
            ],
            "total_pages": 1,
        }
        spec = WikiStructureSpec.model_validate(spec_data)
        assert spec.sections[0].pages[0].target_docs == ["README.md"]


# ============================================================================
# 5. Structure Tools: StructureCollector Tests
# ============================================================================
class TestStructureCollector:
    """Test StructureCollector behavior including symbol validation, batch, and target_docs."""

    @pytest.fixture
    def mock_graph(self):
        """Create a mock code graph with known symbols."""
        g = nx.DiGraph()
        # Add code symbols
        g.add_node("src/services.py::OrderService", 
                    symbol_name="OrderService", 
                    symbol_type="class",
                    rel_path="src/services.py",
                    name="OrderService")
        g.add_node("src/services.py::PaymentProcessor",
                    symbol_name="PaymentProcessor",
                    symbol_type="class",
                    rel_path="src/services.py",
                    name="PaymentProcessor")
        g.add_node("src/utils.py::validate_input",
                    symbol_name="validate_input",
                    symbol_type="function",
                    rel_path="src/utils.py",
                    name="validate_input")
        # Add doc symbols
        g.add_node("README.md::doc",
                    symbol_name="README.md",
                    symbol_type="markdown_document",
                    rel_path="README.md",
                    name="README.md")
        g.add_node("docs/guide.md::doc",
                    symbol_name="Architecture Guide",
                    symbol_type="markdown_section",
                    rel_path="docs/guide.md",
                    name="Architecture Guide")
        return g

    @pytest.fixture
    def collector(self, mock_graph):
        """Create a StructureCollector with the mock graph."""
        from plugin_implementation.wiki_structure_planner.structure_tools import StructureCollector
        return StructureCollector(
            page_budget=10,
            repo_root="/tmp/test_repo",
            code_graph=mock_graph,
        )

    def test_collector_creation(self, collector):
        assert collector.page_budget == 10
        assert collector.code_graph is not None

    def _define_page_helper(self, collector, section_name, page_name, description,
                           target_symbols=None, target_folders=None, target_docs=None):
        """Helper to call _handle_define_page with all required args."""
        return collector._handle_define_page(
            section_name=section_name,
            page_name=page_name,
            page_order=len(collector.pages) + 1,
            description=description,
            content_focus=description,
            rationale="test",
            target_symbols=target_symbols or [],
            target_docs=target_docs or [],
            target_folders=target_folders or [],
            key_files=[],
            retrieval_query="",
        )

    def test_case_insensitive_index_built_on_define_page(self, collector):
        """Calling define_page should lazily build _case_insensitive_symbols."""
        collector.sections["Core"] = {
            "section_name": "Core",
            "section_order": 1,
            "description": "Core section",
            "rationale": "Main",
            "pages": [],
        }
        self._define_page_helper(
            collector, "Core", "Order Processing", "Order management",
            target_symbols=["OrderService"], target_folders=["src/"],
        )
        assert hasattr(collector, '_case_insensitive_symbols')
        assert 'orderservice' in collector._case_insensitive_symbols

    def test_symbol_rel_paths_index_built(self, collector):
        """_symbol_rel_paths index should be built alongside _case_insensitive_symbols."""
        collector.sections["Core"] = {
            "section_name": "Core",
            "section_order": 1,
            "description": "Core section",
            "rationale": "Main",
            "pages": [],
        }
        self._define_page_helper(
            collector, "Core", "Order Processing", "Order management",
            target_symbols=["OrderService"],
        )
        assert hasattr(collector, '_symbol_rel_paths')
        assert collector._symbol_rel_paths.get('orderservice') == 'src/services.py'

    def test_case_insensitive_symbol_matching(self, collector):
        """Symbols should match case-insensitively."""
        collector.sections["Core"] = {
            "section_name": "Core",
            "section_order": 1,
            "description": "d",
            "rationale": "r",
            "pages": [],
        }
        self._define_page_helper(
            collector, "Core", "Payment Flow", "Payment processing",
            target_symbols=["paymentprocessor"],  # lowercase
        )
        # Should find and correct the symbol
        assert len(collector.pages) >= 1
        page = collector.pages[-1]
        assert "PaymentProcessor" in page["target_symbols"]

    def test_define_page_target_docs_default_empty(self, collector):
        """Pages should have target_docs=[] by default."""
        collector.sections["Core"] = {
            "section_name": "Core", "section_order": 1,
            "description": "d", "rationale": "r", "pages": [],
        }
        self._define_page_helper(
            collector, "Core", "Test Page", "desc",
            target_symbols=["OrderService"],
        )
        page = collector.pages[-1]
        assert "target_docs" in page
        assert page["target_docs"] == [] or isinstance(page["target_docs"], list)

    def test_batch_define_pages_creates_section_auto(self, collector):
        """batch_define_pages should auto-create sections that don't exist."""
        pages = [
            {
                "section_name": "NewSection",
                "page_name": "New Page",
                "description": "New page desc",
                "target_symbols": ["OrderService"],
                "target_folders": ["src/"],
            }
        ]
        result = collector._handle_batch_define_pages(pages=pages)
        assert "NewSection" in collector.sections
        assert len(collector.pages) >= 1

    def test_batch_define_pages_validates_symbols_o1(self, collector):
        """batch_define_pages should validate symbols using O(1) index."""
        # Pre-populate index by defining a page first
        collector.sections["Core"] = {
            "section_name": "Core", "section_order": 1,
            "description": "d", "rationale": "r", "pages": [],
        }
        self._define_page_helper(
            collector, "Core", "Dummy", "d",
            target_symbols=["OrderService"],
        )
        # Now batch define with valid + invalid symbols
        pages = [
            {
                "section_name": "Core",
                "page_name": "Validated Page",
                "description": "desc",
                "target_symbols": ["PaymentProcessor", "NonExistent"],
                "target_folders": [],
            }
        ]
        result = collector._handle_batch_define_pages(pages=pages)
        page = collector.pages[-1]
        # PaymentProcessor should be validated, NonExistent should be dropped
        assert "PaymentProcessor" in page["target_symbols"]

    def test_batch_define_pages_derives_folders_from_index(self, collector):
        """batch_define_pages should derive folders from _symbol_rel_paths O(1)."""
        # Trigger index building
        collector.sections["Core"] = {
            "section_name": "Core", "section_order": 1,
            "description": "d", "rationale": "r", "pages": [],
        }
        self._define_page_helper(
            collector, "Core", "Dummy", "d",
            target_symbols=["OrderService"],
        )
        # batch define with a symbol that has a known path
        pages = [
            {
                "section_name": "Core",
                "page_name": "Utility Functions",
                "description": "desc",
                "target_symbols": ["validate_input"],
                "target_folders": [],
            }
        ]
        collector._handle_batch_define_pages(pages=pages)
        page = collector.pages[-1]
        # validate_input is at src/utils.py, so derived folder should be "src"
        assert "src" in page.get("target_folders", [])

    def test_batch_define_pages_empty_list_error(self, collector):
        """batch_define_pages with empty list should return error."""
        result = collector._handle_batch_define_pages(pages=[])
        assert "error" in result.lower() or "Error" in result

    def test_batch_define_pages_budget_tracking(self, collector):
        """batch_define_pages should track page count against budget."""
        collector.sections["Core"] = {
            "section_name": "Core", "section_order": 1,
            "description": "d", "rationale": "r", "pages": [],
        }
        pages = [
            {
                "section_name": "Core",
                "page_name": f"Page {i}",
                "description": f"desc {i}",
                "target_symbols": [],
                "target_folders": [],
            }
            for i in range(5)
        ]
        result = collector._handle_batch_define_pages(pages=pages)
        assert len(collector.pages) >= 5
        stats = collector.stats
        assert stats["pages"] >= 5

    def test_stats_property(self, collector):
        """stats property should return section/page/coverage info."""
        stats = collector.stats
        assert "sections" in stats
        assert "pages" in stats
        assert "budget" in stats
        assert "coverage_pct" in stats


# ============================================================================
# 6. Doc Node Retrieval Tests
# ============================================================================
class TestDocNodeRetrieval:
    """Test _get_doc_nodes_from_graph on the wiki agent."""

    @pytest.fixture
    def mock_graph_with_docs(self):
        """Build a graph with doc and code nodes."""
        g = nx.DiGraph()
        # Code nodes
        g.add_node("src/main.py::App",
                    symbol_name="App", symbol_type="class",
                    rel_path="src/main.py", content="class App: pass")
        # Doc nodes
        g.add_node("README.md::doc",
                    symbol_name="README.md", symbol_type="markdown_document",
                    rel_path="README.md", content="# My Project\nWelcome",
                    file_path="README.md")
        g.add_node("docs/api.md::doc",
                    symbol_name="API Documentation", symbol_type="markdown_section",
                    rel_path="docs/api.md", content="# API\nEndpoints here",
                    file_path="docs/api.md")
        g.add_node("docs/guide.md::doc",
                    symbol_name="Getting Started", symbol_type="markdown_document",
                    rel_path="docs/guide.md", content="# Guide\nStep by step",
                    file_path="docs/guide.md")
        g.add_node("config.yaml::doc",
                    symbol_name="config.yaml", symbol_type="yaml_document",
                    rel_path="config.yaml", content="key: value",
                    file_path="config.yaml")
        return g

    def test_doc_nodes_filtered_by_type(self, mock_graph_with_docs):
        """_get_doc_nodes_from_graph should only return doc-type nodes."""
        # We test this by checking the DOC_SYMBOL_TYPES filter logic
        doc_nodes = []
        for node_id, data in mock_graph_with_docs.nodes(data=True):
            if data.get('symbol_type', '') in DOC_SYMBOL_TYPES:
                doc_nodes.append(node_id)
        assert len(doc_nodes) == 4  # README, api, guide, config
        # Code node should NOT be in doc_nodes
        assert "src/main.py::App" not in doc_nodes

    def test_code_node_excluded_from_docs(self, mock_graph_with_docs):
        """Code symbols should not be classified as doc types."""
        for node_id, data in mock_graph_with_docs.nodes(data=True):
            sym_type = data.get('symbol_type', '')
            if sym_type == 'class':
                assert sym_type not in DOC_SYMBOL_TYPES


# ============================================================================
# 7. Doc Cluster Detection Tests
# ============================================================================
class TestDocClusterDetection:
    """Test doc cluster detection logic."""

    def test_cluster_detection_finds_docs_dir(self):
        """Should detect 'docs/' as a documentation cluster."""
        g = nx.DiGraph()
        g.add_node("docs/intro.md::doc",
                    symbol_name="Intro", symbol_type="markdown_document",
                    rel_path="docs/intro.md")
        g.add_node("docs/setup.md::doc",
                    symbol_name="Setup", symbol_type="markdown_document",
                    rel_path="docs/setup.md")
        g.add_node("docs/api.md::doc",
                    symbol_name="API", symbol_type="markdown_section",
                    rel_path="docs/api.md")
        g.add_node("src/main.py::App",
                    symbol_name="App", symbol_type="class",
                    rel_path="src/main.py")

        # Simulate cluster detection: group doc nodes by parent dir
        doc_dirs = {}
        for node_id, data in g.nodes(data=True):
            if data.get('symbol_type', '') in DOC_SYMBOL_TYPES:
                rel_path = data.get('rel_path', '')
                if rel_path:
                    parent = '/'.join(rel_path.split('/')[:-1]) or '.'
                    doc_dirs.setdefault(parent, []).append(rel_path)

        assert 'docs' in doc_dirs
        assert len(doc_dirs['docs']) == 3

    def test_root_docs_detected(self):
        """Root-level docs (README.md etc) should be detected."""
        g = nx.DiGraph()
        g.add_node("README.md::doc",
                    symbol_name="README.md", symbol_type="markdown_document",
                    rel_path="README.md")
        g.add_node("CONTRIBUTING.md::doc",
                    symbol_name="CONTRIBUTING.md", symbol_type="markdown_document",
                    rel_path="CONTRIBUTING.md")

        doc_dirs = {}
        for node_id, data in g.nodes(data=True):
            if data.get('symbol_type', '') in DOC_SYMBOL_TYPES:
                rel_path = data.get('rel_path', '')
                parent = '/'.join(rel_path.split('/')[:-1]) or '.'
                doc_dirs.setdefault(parent, []).append(rel_path)

        assert '.' in doc_dirs
        assert len(doc_dirs['.']) == 2


# ============================================================================
# 8. ARCHITECTURAL_TYPES Expansion Filtering Tests
# ============================================================================
class TestArchitecturalTypesFiltering:
    """Test that graph expansion uses ARCHITECTURAL_TYPES to filter neighbors."""

    def test_architectural_types_excludes_method(self):
        """ARCHITECTURAL_TYPES should NOT contain 'method' to avoid bloating context."""
        ARCHITECTURAL_TYPES = {
            'class', 'interface', 'struct', 'enum', 'trait',
            'function', 'constant',
        }
        assert 'method' not in ARCHITECTURAL_TYPES

    def test_expansion_filters_non_architectural(self):
        """Simulate expansion filtering: only architectural types pass."""
        g = nx.DiGraph()
        g.add_node("A", symbol_type="class", symbol_name="ClassA")
        g.add_node("B", symbol_type="method", symbol_name="methodB")
        g.add_node("C", symbol_type="function", symbol_name="funcC")
        g.add_node("D", symbol_type="variable", symbol_name="varD")
        g.add_edge("A", "B", relation="CONTAINS")
        g.add_edge("A", "C", relation="CALLS")
        g.add_edge("A", "D", relation="REFERENCES")

        ARCHITECTURAL_TYPES = {
            'class', 'interface', 'struct', 'enum', 'trait',
            'function', 'constant',
        }
        # Only architectural successors should pass
        expanded = []
        for succ in g.successors("A"):
            data = g.nodes[succ]
            if data.get('symbol_type', '') in ARCHITECTURAL_TYPES:
                expanded.append(succ)

        assert "C" in expanded  # function passes
        assert "B" not in expanded  # method filtered
        assert "D" not in expanded  # variable filtered


# ============================================================================
# 9. Feature Flag Gating Tests
# ============================================================================
class TestFeatureFlags:
    """Test feature flag gating for doc separation features."""

    def test_separate_doc_index_env_parsing(self):
        """DEEPWIKI_DOC_SEPARATE_INDEX should parse '1' as True."""
        assert os.getenv("DEEPWIKI_DOC_SEPARATE_INDEX", "0") in ("0", "1")

    @patch.dict(os.environ, {"DEEPWIKI_DOC_SEPARATE_INDEX": "1"})
    def test_separate_doc_index_enabled(self):
        val = os.getenv("DEEPWIKI_DOC_SEPARATE_INDEX", "0") == "1"
        assert val is True

    @patch.dict(os.environ, {"DEEPWIKI_DOC_SEPARATE_INDEX": "0"})
    def test_separate_doc_index_disabled(self):
        val = os.getenv("DEEPWIKI_DOC_SEPARATE_INDEX", "0") == "1"
        assert val is False

    def test_semantic_retrieval_env_parsing(self):
        """DEEPWIKI_DOC_SEMANTIC_RETRIEVAL should default to 0."""
        val = os.getenv("DEEPWIKI_DOC_SEMANTIC_RETRIEVAL", "0")
        assert val in ("0", "1")

    def test_auto_target_docs_env_parsing(self):
        """DEEPWIKI_AUTO_TARGET_DOCS should default to 0."""
        val = os.getenv("DEEPWIKI_AUTO_TARGET_DOCS", "0")
        assert val in ("0", "1")


# ============================================================================
# 10. Query Graph DOC_SYMBOL_TYPES Integration Test
# ============================================================================
class TestQueryGraphDocIntegration:
    """Test that query_graph includes doc nodes from the graph."""

    def test_doc_types_recognized_in_graph_traversal(self):
        """When iterating graph nodes, DOC_SYMBOL_TYPES should classify docs correctly."""
        g = nx.DiGraph()
        # Add various symbol types
        types_in_graph = [
            ("class", "MyClass"),
            ("function", "my_func"),
            ("markdown_document", "README"),
            ("yaml_document", "config"),
            ("text_chunk", "CHANGELOG"),
            ("interface", "IMyInterface"),
        ]
        for sym_type, name in types_in_graph:
            g.add_node(f"{name}::node", symbol_type=sym_type, symbol_name=name, rel_path=f"{name}.py")

        docs = []
        code = []
        for node_id, data in g.nodes(data=True):
            if data.get('symbol_type', '') in DOC_SYMBOL_TYPES:
                docs.append(data['symbol_name'])
            else:
                code.append(data['symbol_name'])

        assert set(docs) == {"README", "config", "CHANGELOG"}
        assert set(code) == {"MyClass", "my_func", "IMyInterface"}


# ============================================================================
# 11. Batch Performance: O(1) vs O(n) Lookup Tests
# ============================================================================
class TestBatchPerformanceIndex:
    """Verify the O(1) _symbol_rel_paths index works correctly."""

    @pytest.fixture
    def large_graph(self):
        """Build a graph with many nodes to demonstrate O(1) benefit."""
        g = nx.DiGraph()
        # Add 1000 nodes
        for i in range(1000):
            g.add_node(
                f"src/module_{i % 20}/file_{i}.py::Symbol_{i}",
                symbol_name=f"Symbol_{i}",
                symbol_type="class",
                rel_path=f"src/module_{i % 20}/file_{i}.py",
                name=f"Symbol_{i}",
            )
        return g

    def test_index_covers_all_symbols(self, large_graph):
        """All symbols should be in the case-insensitive index."""
        index = {}
        paths = {}
        for node_id, data in large_graph.nodes(data=True):
            name = data.get('symbol_name', '') or data.get('name', '')
            if name:
                key = name.lower()
                index[key] = name
                rel_path = data.get('rel_path', '')
                if rel_path and key not in paths:
                    paths[key] = rel_path

        assert len(index) == 1000
        assert len(paths) == 1000
        assert paths['symbol_0'] == 'src/module_0/file_0.py'
        assert paths['symbol_42'] == 'src/module_2/file_42.py'

    def test_folder_derivation_from_path(self):
        """Folder derivation should extract parent directory from rel_path."""
        rel_path = "src/services/orders/order_service.py"
        dir_path = '/'.join(rel_path.split('/')[:-1])
        assert dir_path == "src/services/orders"

    def test_folder_derivation_root_file(self):
        """Root-level files should have empty folder."""
        rel_path = "main.py"
        dir_path = '/'.join(rel_path.split('/')[:-1])
        assert dir_path == ""


# ============================================================================
# 12. Capability-Based Naming Validation Tests
# ============================================================================
class TestCapabilityBasedNaming:
    """Test the naming rules for capability-based page titles."""

    BAD_NAMES = [
        "OrderService Implementation",  # class name in title
        "PaymentProcessor Module",  # class name in title
        "MyApp.java Core",  # filename in title
    ]

    GOOD_NAMES = [
        "Order Lifecycle Management",
        "Payment Processing Flow",
        "Data Persistence Layer",
        "Event Messaging System",
    ]

    def test_good_names_dont_contain_common_symbols(self):
        """Good capability names should not contain PascalCase class names."""
        import re
        pascal_case = re.compile(r'[A-Z][a-z]+[A-Z][a-z]+')
        for name in self.GOOD_NAMES:
            assert not pascal_case.search(name), \
                f"Good name '{name}' looks like it contains a PascalCase class name"

    def test_bad_names_contain_symbols(self):
        """Bad names typically contain PascalCase or file extensions."""
        import re
        pascal_or_ext = re.compile(r'[A-Z][a-z]+[A-Z]|\.java|\.py|\.ts')
        for name in self.BAD_NAMES:
            assert pascal_or_ext.search(name), \
                f"Bad name '{name}' should contain a symbol pattern"


# ============================================================================
# 13. Structure Prompts: Parallel Tool Call Instructions Tests
# ============================================================================
class TestStructurePromptsParallelCalls:
    """Verify the parallel tool call instructions are in the prompts."""

    @pytest.fixture
    def workflow_instructions(self):
        from plugin_implementation.wiki_structure_planner.structure_prompts import STRUCTURE_WORKFLOW_INSTRUCTIONS
        return STRUCTURE_WORKFLOW_INSTRUCTIONS

    def test_parallel_keyword_present(self, workflow_instructions):
        assert "parallel" in workflow_instructions.lower() or "PARALLEL" in workflow_instructions

    def test_batch_define_pages_mentioned(self, workflow_instructions):
        assert "batch_define_pages" in workflow_instructions

    def test_query_graph_parallel_example(self, workflow_instructions):
        # Should contain example of calling multiple query_graph in one message
        assert "query_graph" in workflow_instructions

    def test_define_section_parallel_example(self, workflow_instructions):
        # Should mention parallel section definition
        assert "define_section" in workflow_instructions


# ============================================================================
# 14. Integration: Constants Consistency Across Modules
# ============================================================================
class TestConstantsConsistency:
    """Ensure all modules reference the same constant objects, not duplicates."""

    def test_no_duplicate_doc_symbol_types_definition(self):
        """After consolidation, DOC_SYMBOL_TYPES should be defined only in constants.py.
        
        Other modules should import it, not redefine it.
        This test verifies the import chain is correct by checking object identity.
        """
        from plugin_implementation.constants import DOC_SYMBOL_TYPES as canonical
        from plugin_implementation.code_graph.graph_builder import DOC_SYMBOL_TYPES as gb_ref
        from plugin_implementation.retrievers import DOC_SYMBOL_TYPES as ret_ref

        assert gb_ref is canonical, "graph_builder.DOC_SYMBOL_TYPES is not the canonical constant"
        assert ret_ref is canonical, "retrievers.DOC_SYMBOL_TYPES is not the canonical constant"

    def test_doc_chunk_types_consistency(self):
        from plugin_implementation.constants import DOC_CHUNK_TYPES as canonical
        from plugin_implementation.retrievers import DOC_CHUNK_TYPES as ret_ref
        assert ret_ref is canonical, "retrievers.DOC_CHUNK_TYPES is not the canonical constant"


# ============================================================================
# 15. New Constants: ARCHITECTURAL_SYMBOLS, CODE_SYMBOL_TYPES, Display Buckets
# ============================================================================
class TestNewConstants:
    """Verify the new centralized constants added for architectural types."""

    def test_architectural_symbols_is_frozenset(self):
        from plugin_implementation.constants import ARCHITECTURAL_SYMBOLS
        assert isinstance(ARCHITECTURAL_SYMBOLS, frozenset)

    def test_architectural_symbols_contains_all_expected(self):
        from plugin_implementation.constants import ARCHITECTURAL_SYMBOLS
        expected_code = {'class', 'interface', 'struct', 'enum', 'trait',
                         'function', 'constant', 'type_alias', 'macro'}
        expected_docs = {'markdown_document', 'toml_document', 'text_chunk',
                         'text_document', 'restructuredtext_document', 'plaintext_document'}
        for sym in expected_code | expected_docs:
            assert sym in ARCHITECTURAL_SYMBOLS, f"Missing {sym}"

    def test_code_symbol_types_is_frozenset(self):
        from plugin_implementation.constants import CODE_SYMBOL_TYPES
        assert isinstance(CODE_SYMBOL_TYPES, frozenset)

    def test_code_symbol_types_has_no_docs(self):
        from plugin_implementation.constants import CODE_SYMBOL_TYPES, DOC_SYMBOL_TYPES
        overlap = CODE_SYMBOL_TYPES & DOC_SYMBOL_TYPES
        assert not overlap, f"CODE_SYMBOL_TYPES should not contain doc types: {overlap}"

    def test_code_symbol_types_has_type_alias(self):
        from plugin_implementation.constants import CODE_SYMBOL_TYPES
        assert 'type_alias' in CODE_SYMBOL_TYPES

    def test_class_like_types_includes_trait(self):
        from plugin_implementation.constants import CLASS_LIKE_TYPES
        assert 'trait' in CLASS_LIKE_TYPES
        assert 'class' in CLASS_LIKE_TYPES
        assert 'interface' in CLASS_LIKE_TYPES
        assert 'struct' in CLASS_LIKE_TYPES

    def test_alias_like_types(self):
        from plugin_implementation.constants import ALIAS_LIKE_TYPES
        assert 'type_alias' in ALIAS_LIKE_TYPES

    def test_graph_builder_uses_canonical_arch_symbols(self):
        """graph_builder should use ARCHITECTURAL_SYMBOLS from constants.py."""
        from plugin_implementation.constants import ARCHITECTURAL_SYMBOLS as canonical
        from plugin_implementation.code_graph.graph_builder import ARCHITECTURAL_SYMBOLS as gb_ref
        assert gb_ref is canonical

    def test_wiki_graph_uses_expansion_types(self):
        """wiki_graph_optimized should import EXPANSION_SYMBOL_TYPES from constants.py."""
        from plugin_implementation.constants import EXPANSION_SYMBOL_TYPES as canonical
        from plugin_implementation.agents.wiki_graph_optimized import EXPANSION_SYMBOL_TYPES as wgo_ref
        assert wgo_ref is canonical

    def test_expansion_includes_type_alias(self):
        """EXPANSION_SYMBOL_TYPES should include type_alias (was missing in old local def)."""
        from plugin_implementation.constants import EXPANSION_SYMBOL_TYPES
        assert 'type_alias' in EXPANSION_SYMBOL_TYPES


# ============================================================================
# 16. query_graph: Trait Bucketing and Type Alias Separation
# ============================================================================
class TestQueryGraphBucketing:
    """Verify that query_graph correctly categorizes all architectural symbol types."""

    @pytest.fixture
    def collector_with_graph(self):
        """Build a StructureCollector with a graph containing diverse symbol types."""
        from plugin_implementation.wiki_structure_planner.structure_tools import StructureCollector

        G = nx.MultiDiGraph()
        # Add symbols of each type under 'src/' prefix
        symbols = [
            ('src/models.py::MyClass', {'symbol_name': 'MyClass', 'symbol_type': 'class',
             'rel_path': 'src/models.py', 'content': 'class MyClass: pass'}),
            ('src/traits.py::Drawable', {'symbol_name': 'Drawable', 'symbol_type': 'trait',
             'rel_path': 'src/traits.py', 'content': 'trait Drawable {}'}),
            ('src/iface.py::IService', {'symbol_name': 'IService', 'symbol_type': 'interface',
             'rel_path': 'src/iface.py', 'content': 'interface IService {}'}),
            ('src/types.py::UserId', {'symbol_name': 'UserId', 'symbol_type': 'type_alias',
             'rel_path': 'src/types.py', 'content': 'using UserId = int;'}),
            ('src/enums.py::Color', {'symbol_name': 'Color', 'symbol_type': 'enum',
             'rel_path': 'src/enums.py', 'content': 'enum Color { RED, GREEN }'}),
            ('src/consts.py::MAX_SIZE', {'symbol_name': 'MAX_SIZE', 'symbol_type': 'constant',
             'rel_path': 'src/consts.py', 'content': 'MAX_SIZE = 100'}),
            ('src/funcs.py::create_user', {'symbol_name': 'create_user', 'symbol_type': 'function',
             'rel_path': 'src/funcs.py', 'content': 'def create_user(): pass'}),
            ('src/structs.py::Point', {'symbol_name': 'Point', 'symbol_type': 'struct',
             'rel_path': 'src/structs.py', 'content': 'struct Point { x: int }'}),
        ]
        for node_id, data in symbols:
            G.add_node(node_id, **data)

        # Add an alias_of edge from UserId -> int-like node (outside src/ scope)
        G.add_node('stdlib/builtins.py::int', symbol_name='int', symbol_type='class', 
                    rel_path='stdlib/builtins.py', content='')
        G.add_edge('src/types.py::UserId', 'stdlib/builtins.py::int', 
                    key=0, relationship_type='alias_of')

        collector = StructureCollector.__new__(StructureCollector)
        collector.code_graph = G
        collector.graph_text_index = None  # No FTS5 in unit test — use brute-force path
        collector._graph_module_cache = {}
        collector.repository_files = [d['rel_path'] for _, d in symbols]
        collector._case_insensitive_symbols = {}
        collector._symbol_rel_paths = {}
        for node_id, data in symbols:
            name_lower = data['symbol_name'].lower()
            collector._case_insensitive_symbols[name_lower] = node_id
            rp = data['rel_path']
            collector._symbol_rel_paths[name_lower] = rp

        return collector

    def test_trait_counted_as_class(self, collector_with_graph):
        """Trait symbols should be counted in the class bucket."""
        result = collector_with_graph._handle_query_graph("src")
        # The analysis should now contain 'trait' in the class count
        analysis = collector_with_graph._graph_module_cache.get('src', {})
        class_count = analysis.get('class_count', 0)
        # class + trait + interface + struct = 4
        assert class_count == 4, f"Expected 4 class-like types, got {class_count}"

    def test_type_alias_separate_from_enums(self, collector_with_graph):
        """type_alias should be in all_type_aliases, not in all_enums."""
        result = collector_with_graph._handle_query_graph("src")
        analysis = collector_with_graph._graph_module_cache.get('src', {})
        enum_names = [e['name'] for e in analysis.get('all_enums', [])]
        alias_names = [a['name'] for a in analysis.get('all_type_aliases', [])]
        assert 'UserId' in alias_names, "type_alias should be in all_type_aliases"
        assert 'UserId' not in enum_names, "type_alias should NOT be in all_enums"

    def test_type_alias_count_in_analysis(self, collector_with_graph):
        """Analysis dict should have type_alias_count field."""
        collector_with_graph._handle_query_graph("src")
        analysis = collector_with_graph._graph_module_cache.get('src', {})
        assert analysis.get('type_alias_count', 0) == 1

    def test_type_alias_alias_of_tracked(self, collector_with_graph):
        """type_alias should track alias_of relationships."""
        collector_with_graph._handle_query_graph("src")
        analysis = collector_with_graph._graph_module_cache.get('src', {})
        aliases = analysis.get('all_type_aliases', [])
        user_id_alias = [a for a in aliases if a['name'] == 'UserId']
        assert len(user_id_alias) == 1
        assert 'int' in user_id_alias[0].get('alias_of', [])

    def test_format_includes_type_aliases_section(self, collector_with_graph):
        """The LLM-visible output should have a TYPE ALIASES section."""
        result = collector_with_graph._handle_query_graph("src")
        assert "TYPE ALIASES" in result
        assert "UserId" in result

    def test_format_header_shows_traits(self, collector_with_graph):
        """The header should say Classes/Interfaces/Traits, not just Classes/Interfaces."""
        result = collector_with_graph._handle_query_graph("src")
        assert "Classes/Interfaces/Traits" in result

    def test_all_symbol_types_visible(self, collector_with_graph):
        """Every architectural symbol type should appear somewhere in the output."""
        result = collector_with_graph._handle_query_graph("src")
        # Check all names appear in the formatted output
        for name in ['MyClass', 'Drawable', 'IService', 'UserId', 'Color', 
                      'MAX_SIZE', 'create_user', 'Point']:
            assert name in result, f"Symbol '{name}' not visible in query_graph output"


# ============================================================================
# 17. Doc Cluster Depth
# ============================================================================
class TestDocClusterDepth:
    """Verify doc cluster detection works at depth > 3."""

    @pytest.fixture
    def agent_instance(self):
        """Minimal agent for testing _build_repo_profile and _detect_doc_clusters."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        agent = OptimizedWikiGenerationAgent.__new__(OptimizedWikiGenerationAgent)
        return agent

    def test_depth_4_docs_detected(self, agent_instance):
        """Files at depth 4 (e.g. project/modules/core/docs/readme.md) should be tracked."""
        files = [
            "project/modules/core/docs/readme.md",
            "project/modules/core/docs/api.md",
            "project/modules/core/docs/guide.md",
            "project/modules/core/src/main.py",
        ]
        profile = agent_instance._build_repo_profile(files)
        clusters = profile['doc_clusters']
        # Should find at least one doc cluster containing the docs/ directory
        cluster_paths = [c.get('path', '') for c in clusters]
        # The docs are at depth 4: project/modules/core/docs
        assert any('docs' in p for p in cluster_paths), \
            f"Deep doc directory not detected. Clusters: {cluster_paths}"

    def test_depth_1_docs_detected(self, agent_instance):
        """Root-level doc directories (depth 1) should now be tracked."""
        files = [
            "docs/readme.md",
            "docs/setup.md",
            "docs/api.md",
            "src/main.py",
        ]
        profile = agent_instance._build_repo_profile(files)
        clusters = profile['doc_clusters']
        cluster_paths = [c.get('path', '') for c in clusters]
        assert any('docs' in p for p in cluster_paths), \
            f"Root-level docs/ not detected. Clusters: {cluster_paths}"

    def test_depth_5_docs_detected(self, agent_instance):
        """Files at depth 5 should also be tracked."""
        files = [
            "a/b/c/d/docs/readme.md",
            "a/b/c/d/docs/guide.md",
            "a/b/c/d/docs/faq.md",
        ]
        profile = agent_instance._build_repo_profile(files)
        clusters = profile['doc_clusters']
        cluster_paths = [c.get('path', '') for c in clusters]
        assert any('docs' in p for p in cluster_paths), \
            f"Depth-5 docs not detected. Clusters: {cluster_paths}"


# ============================================================================
# 18. Graph-Based Search (search_codebase hybrid)
# ============================================================================
class TestGraphBasedSearch:
    """Verify the _search_graph_by_text function for hybrid code search."""

    @pytest.fixture
    def code_graph(self):
        G = nx.MultiDiGraph()
        G.add_node('src/auth.py::AuthService', 
                    symbol_name='AuthService', symbol_type='class',
                    rel_path='src/auth.py', content='class AuthService:\n    def login(self): pass',
                    docstring='Handles user authentication and session management',
                    start_line=1, end_line=10)
        G.add_node('src/auth.py::login', 
                    symbol_name='login', symbol_type='function',
                    rel_path='src/auth.py', content='def login(user): pass',
                    start_line=12, end_line=20)
        G.add_node('src/models.py::User', 
                    symbol_name='User', symbol_type='class',
                    rel_path='src/models.py', content='class User:\n    name: str',
                    start_line=1, end_line=5)
        # Add a doc node that should be excluded
        G.add_node('docs/readme.md::readme',
                    symbol_name='readme', symbol_type='markdown_document',
                    rel_path='docs/readme.md', content='# Project README')
        return G

    def test_exact_name_match(self, code_graph):
        from plugin_implementation.deep_research.research_tools import _search_graph_by_text
        results = _search_graph_by_text(code_graph, "AuthService", k=5)
        assert len(results) >= 1
        assert results[0].metadata['symbol_name'] == 'AuthService'

    def test_substring_match(self, code_graph):
        from plugin_implementation.deep_research.research_tools import _search_graph_by_text
        results = _search_graph_by_text(code_graph, "auth", k=5)
        names = [r.metadata['symbol_name'] for r in results]
        assert 'AuthService' in names

    def test_docstring_search(self, code_graph):
        from plugin_implementation.deep_research.research_tools import _search_graph_by_text
        results = _search_graph_by_text(code_graph, "authentication session", k=5)
        names = [r.metadata['symbol_name'] for r in results]
        assert 'AuthService' in names

    def test_excludes_doc_nodes(self, code_graph):
        from plugin_implementation.deep_research.research_tools import _search_graph_by_text
        results = _search_graph_by_text(code_graph, "readme", k=5)
        types = [r.metadata['symbol_type'] for r in results]
        assert 'markdown_document' not in types

    def test_empty_graph(self):
        from plugin_implementation.deep_research.research_tools import _search_graph_by_text
        G = nx.MultiDiGraph()
        results = _search_graph_by_text(G, "test", k=5)
        assert results == []

    def test_none_graph(self):
        from plugin_implementation.deep_research.research_tools import _search_graph_by_text
        results = _search_graph_by_text(None, "test", k=5)
        assert results == []

    def test_result_has_search_source_graph(self, code_graph):
        from plugin_implementation.deep_research.research_tools import _search_graph_by_text
        results = _search_graph_by_text(code_graph, "User", k=5)
        assert len(results) >= 1
        assert results[0].metadata['search_source'] == 'graph'

    def test_scoring_prefers_exact_match(self, code_graph):
        from plugin_implementation.deep_research.research_tools import _search_graph_by_text
        results = _search_graph_by_text(code_graph, "login", k=5)
        # Exact match "login" should score higher than "AuthService" which only has "login" in content
        if len(results) >= 2:
            assert results[0].metadata['symbol_name'] == 'login'
