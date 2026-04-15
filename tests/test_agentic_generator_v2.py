"""
Tests for AgenticDocGeneratorV2 (Planning + Tool Calling)

Tests the new agentic generator that combines:
1. LLM Planning Phase: Semantic section structure generation
2. Tool Calling Phase: Section-by-section generation with context swapping
"""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.documents import Document

from plugin_implementation.agents.agentic_doc_generator_v2 import (
    AgenticDocGeneratorV2,
    PlannedSection,
    DocumentationState,
    should_use_agentic_mode,
    ARCHITECTURAL_SYMBOL_TYPES,
)
from plugin_implementation.state.wiki_state import (
    PageStructure,
    DetailSectionSpec,
    OverviewSection,
    SymbolSummary,
)


class TestDocumentationStateV2:
    """Tests for V2 DocumentationState (simplified from V1)"""
    
    def test_empty_state_toc(self):
        """Empty state returns default message"""
        state = DocumentationState()
        toc = state.get_toc_for_prompt()
        assert "first detail section" in toc.lower()
    
    def test_add_section_updates_toc(self):
        """Adding section updates ToC"""
        state = DocumentationState()
        state.add_section("Authentication", "Handles user login")
        
        assert len(state.toc) == 1
        assert "Authentication" in state.get_toc_for_prompt()
        assert "Handles user login" in state.get_toc_for_prompt()
    
    def test_completed_sections_tracked(self):
        """Completed sections are tracked"""
        state = DocumentationState()
        state.add_section("Auth", "Auth summary")
        state.add_section("Data", "Data summary")
        
        assert "Auth" in state.completed_sections
        assert "Data" in state.completed_sections


class TestPlannedSection:
    """Tests for PlannedSection dataclass"""
    
    def test_planned_section_creation(self):
        """PlannedSection can be created with required fields"""
        section = PlannedSection(
            section_id="auth_impl",
            title="Authentication Implementation",
            description="Core authentication logic",
            primary_symbols=["AuthService", "TokenFactory"],
            supporting_symbols=["UserRepository"],
            symbol_paths=["src/auth/service.py"],
            retrieval_queries=["authentication service"],
            intent="implementation",
            suggested_elements=["sequence diagram", "code example"]
        )
        
        assert section.section_id == "auth_impl"
        assert section.title == "Authentication Implementation"
        assert len(section.primary_symbols) == 2
        assert "AuthService" in section.primary_symbols
    
    def test_planned_section_defaults(self):
        """PlannedSection has default empty full_code and signatures"""
        section = PlannedSection(
            section_id="test",
            title="Test",
            description="Test",
            primary_symbols=[],
            supporting_symbols=[],
            symbol_paths=[],
            retrieval_queries=[],
            intent=None,
            suggested_elements=[]
        )
        
        assert section.full_code == ""
        assert section.signatures == ""


class TestArchitecturalSymbolTypes:
    """Tests for architectural symbol type filtering"""
    
    def test_classes_are_architectural(self):
        """Classes, interfaces, structs are architectural"""
        for sym_type in ['class', 'interface', 'struct', 'enum', 'trait']:
            assert sym_type in ARCHITECTURAL_SYMBOL_TYPES
    
    def test_functions_are_architectural(self):
        """Standalone functions are architectural"""
        assert 'function' in ARCHITECTURAL_SYMBOL_TYPES
    
    def test_methods_not_architectural(self):
        """Methods are NOT architectural (belong to parent class)"""
        assert 'method' not in ARCHITECTURAL_SYMBOL_TYPES
        assert 'constructor' not in ARCHITECTURAL_SYMBOL_TYPES
    
    def test_docs_are_architectural(self):
        """Documentation files are architectural"""
        assert 'markdown_document' in ARCHITECTURAL_SYMBOL_TYPES
        assert 'documentation' in ARCHITECTURAL_SYMBOL_TYPES


class TestSymbolIndexBuilding:
    """Tests for symbol index building"""
    
    @pytest.fixture
    def mock_llm(self):
        llm = Mock()
        return llm
    
    @pytest.fixture
    def mock_page_spec(self):
        spec = Mock()
        spec.page_name = "Test Page"
        spec.description = "Test description"
        spec.content_focus = "Test focus"
        spec.key_files = []
        spec.target_folders = []
        return spec
    
    def test_builds_symbol_index_from_docs(self, mock_llm, mock_page_spec):
        """Symbol index is built from documents"""
        docs = [
            Document(
                page_content="class AuthService: pass",
                metadata={"symbol_name": "AuthService", "symbol_type": "class", "source": "auth.py"}
            ),
            Document(
                page_content="def helper(): pass",
                metadata={"symbol_name": "helper", "symbol_type": "function", "source": "utils.py"}
            ),
        ]
        
        generator = AgenticDocGeneratorV2(
            llm=mock_llm,
            expanded_docs=docs,
            page_spec=mock_page_spec,
            repo_context="Test repo",
            repository_url="https://github.com/test/repo",
            wiki_style="comprehensive",
            target_audience="developers"
        )
        
        symbol_index = generator._build_symbol_index()
        
        assert len(symbol_index) == 2
        names = [s.name for s in symbol_index]
        assert "AuthService" in names
        assert "helper" in names
    
    def test_filters_non_architectural_symbols(self, mock_llm, mock_page_spec):
        """Methods and variables are filtered out"""
        docs = [
            Document(
                page_content="class Service: pass",
                metadata={"symbol_name": "Service", "symbol_type": "class", "source": "service.py"}
            ),
            Document(
                page_content="def method(): pass",
                metadata={"symbol_name": "method", "symbol_type": "method", "source": "service.py"}
            ),
            Document(
                page_content="x = 1",
                metadata={"symbol_name": "x", "symbol_type": "variable", "source": "config.py"}
            ),
        ]
        
        generator = AgenticDocGeneratorV2(
            llm=mock_llm,
            expanded_docs=docs,
            page_spec=mock_page_spec,
            repo_context="Test repo",
            repository_url="https://github.com/test/repo",
            wiki_style="comprehensive",
            target_audience="developers"
        )
        
        symbol_index = generator._build_symbol_index()
        
        # Only class should be included, not method or variable
        assert len(symbol_index) == 1
        assert symbol_index[0].name == "Service"


class TestV2GeneratorInit:
    """Tests for V2 generator initialization"""
    
    @pytest.fixture
    def mock_llm(self):
        return Mock()
    
    @pytest.fixture
    def mock_page_spec(self):
        spec = Mock()
        spec.page_name = "Test Page"
        spec.description = "Test description"
        spec.content_focus = "Test focus"
        spec.key_files = []
        spec.target_folders = []
        return spec
    
    def test_initializes_with_docs(self, mock_llm, mock_page_spec):
        """Generator initializes correctly with documents"""
        docs = [
            Document(
                page_content="class Test: pass",
                metadata={"symbol_name": "Test", "source": "test.py"}
            )
        ]
        
        generator = AgenticDocGeneratorV2(
            llm=mock_llm,
            expanded_docs=docs,
            page_spec=mock_page_spec,
            repo_context="Test repo",
            repository_url="https://github.com/test/repo",
            wiki_style="comprehensive",
            target_audience="developers"
        )
        
        assert len(generator.all_docs) == 1
        assert "test.py" in generator.all_rel_paths
    
    def test_collects_all_file_paths(self, mock_llm, mock_page_spec):
        """Generator collects all unique file paths"""
        docs = [
            Document(page_content="a", metadata={"symbol_name": "A", "rel_path": "src/a.py"}),
            Document(page_content="b", metadata={"symbol_name": "B", "rel_path": "src/b.py"}),
            Document(page_content="c", metadata={"symbol_name": "C", "rel_path": "src/a.py"}),  # Duplicate
        ]
        
        generator = AgenticDocGeneratorV2(
            llm=mock_llm,
            expanded_docs=docs,
            page_spec=mock_page_spec,
            repo_context="Test repo",
            repository_url="https://github.com/test/repo",
            wiki_style="comprehensive",
            target_audience="developers"
        )
        
        # Should have 2 unique paths
        assert len(generator.all_rel_paths) == 2
        assert "src/a.py" in generator.all_rel_paths
        assert "src/b.py" in generator.all_rel_paths


class TestSectionContextBuilding:
    """Tests for building section context"""
    
    @pytest.fixture
    def generator_with_docs(self):
        mock_llm = Mock()
        mock_page_spec = Mock()
        mock_page_spec.page_name = "Test"
        mock_page_spec.description = "Test"
        mock_page_spec.content_focus = "Test"
        mock_page_spec.key_files = []
        mock_page_spec.target_folders = []
        
        docs = [
            Document(
                page_content="class AuthService:\n    def login(self): pass",
                metadata={"symbol_name": "AuthService", "symbol_type": "class", "source": "auth.py"}
            ),
            Document(
                page_content="class TokenFactory:\n    def create(self): pass",
                metadata={"symbol_name": "TokenFactory", "symbol_type": "class", "source": "token.py"}
            ),
            Document(
                page_content="class UserRepo:\n    def get(self): pass",
                metadata={"symbol_name": "UserRepo", "symbol_type": "class", "source": "user.py"}
            ),
        ]
        
        return AgenticDocGeneratorV2(
            llm=mock_llm,
            expanded_docs=docs,
            page_spec=mock_page_spec,
            repo_context="Test",
            repository_url="https://github.com/test/repo",
            wiki_style="comprehensive",
            target_audience="developers"
        )
    
    def test_builds_full_code_for_primary_symbols(self, generator_with_docs):
        """Primary symbols get full code"""
        section = PlannedSection(
            section_id="auth",
            title="Auth",
            description="Auth section",
            primary_symbols=["AuthService"],
            supporting_symbols=["TokenFactory"],
            symbol_paths=[],
            retrieval_queries=[],
            intent=None,
            suggested_elements=[]
        )
        
        generator_with_docs._build_section_context(section)
        
        assert "AuthService" in section.full_code
        assert "class AuthService" in section.full_code
        assert "login" in section.full_code
    
    def test_builds_signatures_for_other_symbols(self, generator_with_docs):
        """Non-primary symbols get signatures"""
        section = PlannedSection(
            section_id="auth",
            title="Auth",
            description="Auth section",
            primary_symbols=["AuthService"],
            supporting_symbols=["TokenFactory", "UserRepo"],
            symbol_paths=[],
            retrieval_queries=[],
            intent=None,
            suggested_elements=[]
        )
        
        generator_with_docs._build_section_context(section)
        
        # TokenFactory and UserRepo should be in signatures
        assert "TokenFactory" in section.signatures or "UserRepo" in section.signatures


class TestPlanningPhaseModels:
    """Tests for planning phase Pydantic models"""
    
    def test_page_structure_creation(self):
        """PageStructure can be created with all fields"""
        structure = PageStructure(
            title="Authentication Guide",
            introduction="This guide covers auth...",
            overview_sections=[
                OverviewSection(title="Architecture", content="The auth system...")
            ],
            detail_sections=[
                DetailSectionSpec(
                    section_id="auth_impl",
                    title="Authentication Implementation",
                    description="Core auth logic",
                    primary_symbols=["AuthService"],
                    supporting_symbols=["TokenFactory"]
                )
            ],
            conclusion="In summary..."
        )
        
        assert structure.title == "Authentication Guide"
        assert len(structure.overview_sections) == 1
        assert len(structure.detail_sections) == 1
    
    def test_symbol_summary_creation(self):
        """SymbolSummary can be created"""
        summary = SymbolSummary(
            name="AuthService",
            qualified_name="src/auth::AuthService",
            symbol_type="class",
            file_path="src/auth/service.py",
            brief="Handles authentication",
            relationships=["extends BaseService"],
            estimated_tokens=500
        )
        
        assert summary.name == "AuthService"
        assert summary.symbol_type == "class"


class TestToolCallHandling:
    """Tests for tool call handling in V2"""
    
    @pytest.fixture
    def generator(self):
        mock_llm = Mock()
        mock_page_spec = Mock()
        mock_page_spec.page_name = "Test"
        mock_page_spec.description = "Test"
        mock_page_spec.content_focus = "Test"
        mock_page_spec.key_files = []
        mock_page_spec.target_folders = []
        
        docs = [
            Document(
                page_content="class Test: pass",
                metadata={"symbol_name": "Test", "source": "test.py"}
            )
        ]
        
        gen = AgenticDocGeneratorV2(
            llm=mock_llm,
            expanded_docs=docs,
            page_spec=mock_page_spec,
            repo_context="Test",
            repository_url="https://github.com/test/repo",
            wiki_style="comprehensive",
            target_audience="developers"
        )
        
        # Set up minimal planned sections
        gen.planned_sections = [
            PlannedSection(
                section_id="sec1",
                title="Section 1",
                description="First section",
                primary_symbols=["Test"],
                supporting_symbols=[],
                symbol_paths=[],
                retrieval_queries=[],
                intent=None,
                suggested_elements=[]
            )
        ]
        
        return gen
    
    def test_handle_commit_section(self, generator):
        """commit_section tool call adds content"""
        section = generator.planned_sections[0]
        
        # Mock response with tool calls
        mock_response = Mock()
        mock_response.tool_calls = [
            {"name": "commit_section", "args": {"content": "# Section 1\n\nContent", "summary": "First section"}}
        ]
        
        generator._handle_tool_calls(mock_response, section)
        
        assert len(generator.committed_content) == 1
        assert "Section 1" in generator.committed_content[0]
        assert "Section 1" in generator.doc_state.completed_sections
    
    def test_handle_request_next_section(self, generator):
        """request_next_section with name returns section name"""
        # Add another section to jump to
        generator.planned_sections.append(
            PlannedSection(
                section_id="sec2",
                title="Section 2",
                description="Second",
                primary_symbols=[],
                supporting_symbols=[],
                symbol_paths=[],
                retrieval_queries=[],
                intent=None,
                suggested_elements=[]
            )
        )
        
        section = generator.planned_sections[0]
        
        mock_response = Mock()
        mock_response.tool_calls = [
            {"name": "commit_section", "args": {"content": "Content", "summary": "Done"}},
            {"name": "request_next_section", "args": {"section_name": "Section 2"}}
        ]
        
        next_section = generator._handle_tool_calls(mock_response, section)
        
        assert next_section == "Section 2"


class TestShouldUseAgenticMode:
    """Tests for the agentic mode decision helper"""
    
    def test_small_context_not_agentic(self):
        """Small context should not use agentic mode"""
        docs = [Document(page_content="x" * 1000, metadata={})]
        assert not should_use_agentic_mode(docs, token_budget=50_000)
    
    def test_large_context_uses_agentic(self):
        """Large context should use agentic mode"""
        # Token counter uses ~8 chars per token
        # 200K chars = ~25K tokens per doc, 3 docs = ~75K tokens
        docs = [Document(page_content="x" * 200000, metadata={}) for _ in range(3)]
        assert should_use_agentic_mode(docs, token_budget=50_000)
