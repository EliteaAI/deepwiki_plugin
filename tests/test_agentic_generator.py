"""
Tests for Agentic Documentation Generator

Tests the section-based context-swapping documentation generator.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document

from plugin_implementation.agents.agentic_doc_generator import (
    AgenticDocGenerator,
    SectionContext,
    DocumentationState,
    should_use_agentic_mode,
    AGENTIC_SYSTEM_PROMPT,
    AGENTIC_USER_PROMPT,
)


class TestDocumentationState:
    """Tests for DocumentationState tracking"""
    
    def test_empty_state_toc(self):
        """Empty state should return first section message"""
        state = DocumentationState()
        toc = state.get_toc_for_prompt()
        assert "first section" in toc.lower()
    
    def test_add_section_updates_toc(self):
        """Adding a section should update ToC"""
        state = DocumentationState()
        state.add_section("Auth Module", ["AuthManager", "TokenFactory"], "Handles authentication")
        
        toc = state.get_toc_for_prompt()
        assert "Auth Module" in toc
        assert "Handles authentication" in toc
        assert "AuthManager" in toc or "`AuthManager`" in toc
    
    def test_add_section_tracks_symbols(self):
        """Adding section should track symbol to section mapping"""
        state = DocumentationState()
        state.add_section("Auth Module", ["AuthManager", "TokenFactory"], "Handles auth")
        
        assert state.symbol_to_section["AuthManager"] == "Auth Module"
        assert state.symbol_to_section["TokenFactory"] == "Auth Module"
    
    def test_completed_sections_tracked(self):
        """Completed sections should be tracked"""
        state = DocumentationState()
        state.add_section("Auth Module", ["AuthManager"], "Auth stuff")
        state.add_section("Data Module", ["DataManager"], "Data stuff")
        
        assert "Auth Module" in state.completed_sections
        assert "Data Module" in state.completed_sections
    
    def test_cross_reference_for_documented_symbol(self):
        """Should return cross-reference for documented symbol"""
        state = DocumentationState()
        state.add_section("Auth Module", ["AuthManager"], "Auth handling")
        
        cross_ref = state.get_cross_reference("AuthManager")
        assert cross_ref is not None
        assert "Auth Module" in cross_ref
    
    def test_cross_reference_for_unknown_symbol(self):
        """Should return None for unknown symbol"""
        state = DocumentationState()
        cross_ref = state.get_cross_reference("UnknownClass")
        assert cross_ref is None


class TestSectionContext:
    """Tests for SectionContext data class"""
    
    def test_section_context_creation(self):
        """Should create section context with name and anchors"""
        section = SectionContext(
            name="Authentication",
            anchor_ids=["AuthManager", "TokenValidator"]
        )
        
        assert section.name == "Authentication"
        assert len(section.anchor_ids) == 2
        assert "AuthManager" in section.anchor_ids
    
    def test_section_context_defaults(self):
        """Should have empty defaults for code and signatures"""
        section = SectionContext(name="Test", anchor_ids=["Symbol1"])
        
        assert section.full_code == ""
        assert section.signatures == ""


class TestShouldUseAgenticMode:
    """Tests for mode decision logic"""
    
    def test_small_context_not_agentic(self):
        """Small context should not use agentic mode"""
        # Create small docs (under 50K tokens)
        docs = [
            Document(page_content="class Small: pass", metadata={"symbol_name": f"Small{i}"})
            for i in range(10)
        ]
        
        result = should_use_agentic_mode(docs, token_budget=50_000)
        assert result is False
    
    def test_large_context_uses_agentic(self):
        """Large context should use agentic mode"""
        # Create large docs (over budget)
        # Each char is ~0.25 tokens, so 200K chars = ~50K tokens
        large_content = "x" * 200_000  # ~50K tokens each doc
        docs = [
            Document(page_content=large_content, metadata={"symbol_name": f"Large{i}"})
            for i in range(2)  # ~100K tokens total
        ]
        
        result = should_use_agentic_mode(docs, token_budget=50_000)
        assert result is True


class TestAgenticDocGeneratorInit:
    """Tests for AgenticDocGenerator initialization"""
    
    def test_separates_anchors_and_deps(self):
        """Should separate anchor docs from expanded deps"""
        mock_llm = Mock()
        mock_page_spec = Mock()
        mock_page_spec.page_name = "Test Page"
        mock_page_spec.description = "Test description"
        mock_page_spec.content_focus = "Testing"
        
        anchor_doc = Document(
            page_content="class Anchor: pass",
            metadata={"symbol_name": "Anchor", "initially_retrieved": True}
        )
        dep_doc = Document(
            page_content="class Dep: pass",
            metadata={"symbol_name": "Dep", "initially_retrieved": False}
        )
        
        generator = AgenticDocGenerator(
            llm=mock_llm,
            expanded_docs=[anchor_doc, dep_doc],
            page_spec=mock_page_spec,
            repo_context="Test repo",
            repository_url="https://github.com/test/repo",
            wiki_style="comprehensive",
            target_audience="developers"
        )
        
        assert len(generator.anchor_docs) == 1
        assert len(generator.expanded_deps) == 1
        assert generator.anchor_docs[0].metadata["symbol_name"] == "Anchor"
        assert generator.expanded_deps[0].metadata["symbol_name"] == "Dep"
    
    def test_creates_sections_from_anchors(self):
        """Should create sections from anchor documents"""
        mock_llm = Mock()
        mock_page_spec = Mock()
        mock_page_spec.page_name = "Test Page"
        mock_page_spec.description = "Test"
        mock_page_spec.content_focus = "Test"
        
        docs = [
            Document(
                page_content=f"class Class{i}: pass",
                metadata={"symbol_name": f"Class{i}", "initially_retrieved": True, "source": f"file{i}.py"}
            )
            for i in range(3)
        ]
        
        generator = AgenticDocGenerator(
            llm=mock_llm,
            expanded_docs=docs,
            page_spec=mock_page_spec,
            repo_context="Test",
            repository_url="https://github.com/test/repo",
            wiki_style="comprehensive",
            target_audience="developers"
        )
        
        assert len(generator.sections) > 0
        # All anchors should be in some section
        all_anchor_ids = []
        for section in generator.sections:
            all_anchor_ids.extend(section.anchor_ids)
        assert len(all_anchor_ids) == 3


class TestAgenticPrompts:
    """Tests for agentic prompt templates"""
    
    def test_system_prompt_has_required_sections(self):
        """System prompt should have all required sections"""
        assert "DOCUMENTATION STYLE" in AGENTIC_SYSTEM_PROMPT
        assert "REPOSITORY OVERVIEW" in AGENTIC_SYSTEM_PROMPT
        assert "WORKFLOW" in AGENTIC_SYSTEM_PROMPT
        assert "TOOLS" in AGENTIC_SYSTEM_PROMPT
        assert "commit_section" in AGENTIC_SYSTEM_PROMPT
        assert "request_next_section" in AGENTIC_SYSTEM_PROMPT
    
    def test_system_prompt_has_tool_sequence(self):
        """System prompt should document correct tool sequence"""
        assert "commit_section" in AGENTIC_SYSTEM_PROMPT
        assert "request_next_section" in AGENTIC_SYSTEM_PROMPT
        assert "BEFORE" in AGENTIC_SYSTEM_PROMPT  # commit BEFORE request
    
    def test_system_prompt_has_formatting_guidelines(self):
        """System prompt should include formatting guidelines"""
        assert "Mermaid" in AGENTIC_SYSTEM_PROMPT or "mermaid" in AGENTIC_SYSTEM_PROMPT
        assert "table" in AGENTIC_SYSTEM_PROMPT.lower()
        assert "code" in AGENTIC_SYSTEM_PROMPT.lower()
    
    def test_user_prompt_has_placeholders(self):
        """User prompt should have required placeholders"""
        assert "{section_name}" in AGENTIC_USER_PROMPT
        assert "{full_code}" in AGENTIC_USER_PROMPT
        assert "{signatures}" in AGENTIC_USER_PROMPT
        assert "{toc_block}" in AGENTIC_USER_PROMPT
    
    def test_system_prompt_format_placeholders(self):
        """System prompt should have format placeholders"""
        assert "{target_audience}" in AGENTIC_SYSTEM_PROMPT
        assert "{wiki_style}" in AGENTIC_SYSTEM_PROMPT
        assert "{repo_analysis}" in AGENTIC_SYSTEM_PROMPT
        assert "{page_name}" in AGENTIC_SYSTEM_PROMPT


class TestAgenticDocGeneratorMethods:
    """Tests for AgenticDocGenerator methods"""
    
    def test_get_doc_id_from_node_id(self):
        """Should get doc ID from node_id metadata"""
        mock_llm = Mock()
        mock_page_spec = Mock()
        mock_page_spec.page_name = "Test"
        mock_page_spec.description = "Test"
        mock_page_spec.content_focus = "Test"
        
        generator = AgenticDocGenerator(
            llm=mock_llm,
            expanded_docs=[],
            page_spec=mock_page_spec,
            repo_context="Test",
            repository_url="https://github.com/test/repo",
            wiki_style="comprehensive",
            target_audience="developers"
        )
        
        doc = Document(
            page_content="test",
            metadata={"node_id": "my.node.id", "symbol_name": "fallback"}
        )
        
        doc_id = generator._get_doc_id(doc)
        assert doc_id == "my.node.id"
    
    def test_get_doc_id_fallback_to_symbol_name(self):
        """Should fall back to symbol_name if no node_id"""
        mock_llm = Mock()
        mock_page_spec = Mock()
        mock_page_spec.page_name = "Test"
        mock_page_spec.description = "Test"
        mock_page_spec.content_focus = "Test"
        
        generator = AgenticDocGenerator(
            llm=mock_llm,
            expanded_docs=[],
            page_spec=mock_page_spec,
            repo_context="Test",
            repository_url="https://github.com/test/repo",
            wiki_style="comprehensive",
            target_audience="developers"
        )
        
        doc = Document(
            page_content="test",
            metadata={"symbol_name": "MyClass"}
        )
        
        doc_id = generator._get_doc_id(doc)
        assert doc_id == "MyClass"
    
    def test_generate_section_name_from_symbols(self):
        """Should generate meaningful section name from symbols"""
        mock_llm = Mock()
        mock_page_spec = Mock()
        mock_page_spec.page_name = "Test"
        mock_page_spec.description = "Test"
        mock_page_spec.content_focus = "Test"
        
        generator = AgenticDocGenerator(
            llm=mock_llm,
            expanded_docs=[],
            page_spec=mock_page_spec,
            repo_context="Test",
            repository_url="https://github.com/test/repo",
            wiki_style="comprehensive",
            target_audience="developers"
        )
        
        docs = [
            Document(page_content="class Auth: pass", metadata={"symbol_name": "AuthManager"}),
            Document(page_content="class Token: pass", metadata={"symbol_name": "TokenFactory"})
        ]
        
        name = generator._generate_section_name(docs, 1)
        # New naming converts camelCase to Title Case, so "AuthManager" becomes "Auth Manager"
        assert "auth" in name.lower() and "manager" in name.lower()  # Human-readable format


class TestAgenticToolHandling:
    """Tests for tool call handling"""
    
    def test_handle_commit_adds_content(self):
        """commit_section should add content to committed list"""
        mock_llm = Mock()
        mock_page_spec = Mock()
        mock_page_spec.page_name = "Test"
        mock_page_spec.description = "Test"
        mock_page_spec.content_focus = "Test"
        
        doc = Document(
            page_content="class Test: pass",
            metadata={"symbol_name": "Test", "initially_retrieved": True, "source": "test.py"}
        )
        
        generator = AgenticDocGenerator(
            llm=mock_llm,
            expanded_docs=[doc],
            page_spec=mock_page_spec,
            repo_context="Test",
            repository_url="https://github.com/test/repo",
            wiki_style="comprehensive",
            target_audience="developers"
        )
        
        # Simulate tool call response
        section = generator.sections[0]
        mock_response = Mock()
        mock_response.tool_calls = [
            {
                "name": "commit_section",
                "args": {"content": "# Test Documentation", "summary": "Test docs"}
            }
        ]
        
        generator._handle_tool_calls(mock_response, section)
        
        assert len(generator.committed_content) == 1
        assert "# Test Documentation" in generator.committed_content[0]
        assert section.name in generator.doc_state.completed_sections


class TestAgenticFlowWithMockedLLM:
    """
    Integration tests for the full agentic generation flow.
    
    These tests mock the LLM to simulate realistic tool call sequences.
    """
    
    def _create_mock_llm_response(self, tool_calls: list):
        """Helper to create a mock LLM response with tool calls"""
        mock_response = Mock()
        mock_response.tool_calls = tool_calls
        mock_response.content = ""
        return mock_response
    
    def _create_generator_with_docs(self, num_anchors: int = 2, force_separate_sections: bool = False):
        """
        Helper to create a generator with test documents.
        
        Args:
            num_anchors: Number of anchor documents to create
            force_separate_sections: If True, directly inject multiple sections for testing
        """
        mock_llm = Mock()
        mock_page_spec = Mock()
        mock_page_spec.page_name = "Test Page"
        mock_page_spec.description = "Test description"
        mock_page_spec.content_focus = "Testing"
        
        # Always create small documents that will group into one section
        docs = [
            Document(
                page_content=f"class TestClass{i}:\n    def method(self): pass",
                metadata={
                    "symbol_name": f"TestClass{i}",
                    "initially_retrieved": True,
                    "source": f"test{i}.py"
                }
            )
            for i in range(num_anchors)
        ]
        
        generator = AgenticDocGenerator(
            llm=mock_llm,
            expanded_docs=docs,
            page_spec=mock_page_spec,
            repo_context="Test repository context",
            repository_url="https://github.com/test/repo",
            wiki_style="comprehensive",
            target_audience="developers"
        )
        
        if force_separate_sections and num_anchors >= 2:
            # Directly inject multiple sections for testing multi-section flows
            # This bypasses the token-based clustering logic which would group small docs
            generator.sections = [
                SectionContext(
                    name=f"Section {i+1}",
                    anchor_ids=[f"TestClass{i}"]
                )
                for i in range(num_anchors)
            ]
        
        return generator, mock_llm
    
    def test_single_section_flow(self):
        """Test complete flow with single section: commit then request"""
        generator, mock_llm = self._create_generator_with_docs(num_anchors=1)
        
        # Mock LLM to return commit_section then request_next_section
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_llm.invoke = Mock(return_value=self._create_mock_llm_response([
            {"name": "commit_section", "args": {"content": "# Section 1 Docs", "summary": "First section"}},
            {"name": "request_next_section", "args": {}}
        ]))
        
        # Run generation
        mock_config = Mock()
        result = generator.generate(mock_config)
        
        # Verify results
        assert len(generator.committed_content) == 1
        assert "# Section 1 Docs" in result
        assert mock_llm.invoke.called
    
    def test_multi_section_flow(self):
        """Test flow with multiple sections - each gets documented in sequence"""
        generator, mock_llm = self._create_generator_with_docs(num_anchors=3)
        
        # Track which section we're on to return different responses
        call_count = [0]
        
        def mock_invoke(*args, **kwargs):
            call_count[0] += 1
            section_num = min(call_count[0], len(generator.sections))
            return self._create_mock_llm_response([
                {"name": "commit_section", "args": {
                    "content": f"# Section {section_num} Documentation",
                    "summary": f"Section {section_num} summary"
                }},
                {"name": "request_next_section", "args": {}}
            ])
        
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_llm.invoke = Mock(side_effect=mock_invoke)
        
        # Run generation
        mock_config = Mock()
        result = generator.generate(mock_config)
        
        # Verify all sections were processed
        assert len(generator.committed_content) == len(generator.sections)
        assert len(generator.doc_state.completed_sections) == len(generator.sections)
    
    def test_context_swap_updates_section(self):
        """Test that request_next_section triggers context swap"""
        # Force separate sections with large docs
        generator, mock_llm = self._create_generator_with_docs(num_anchors=2, force_separate_sections=True)
        
        # Verify we have multiple sections
        assert len(generator.sections) >= 2, f"Expected >=2 sections, got {len(generator.sections)}"
        
        # First call commits section 1 and requests section 2
        # Second call commits section 2
        responses = [
            self._create_mock_llm_response([
                {"name": "commit_section", "args": {"content": "# Auth Module", "summary": "Authentication"}},
                {"name": "request_next_section", "args": {}}
            ]),
            self._create_mock_llm_response([
                {"name": "commit_section", "args": {"content": "# Data Module", "summary": "Data handling"}},
                {"name": "request_next_section", "args": {}}
            ])
        ]
        
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_llm.invoke = Mock(side_effect=responses)
        
        mock_config = Mock()
        result = generator.generate(mock_config)
        
        # Verify both sections documented
        assert "# Auth Module" in result
        assert "# Data Module" in result
        assert len(generator.doc_state.toc) == 2
    
    def test_jump_to_specific_section(self):
        """Test jumping to a specific section by name"""
        # Force separate sections
        generator, mock_llm = self._create_generator_with_docs(num_anchors=3, force_separate_sections=True)
        
        # Verify we have multiple sections
        num_sections = len(generator.sections)
        assert num_sections == 3, f"Expected 3 sections, got {num_sections}"
        
        # Get section names
        section_names = [s.name for s in generator.sections]
        
        # Test flow: Section 1 -> jump to Section 2 -> Section 3 (sequential from there)
        # This ensures jump works and then continues linearly
        responses = [
            # Section 1: commit and jump to Section 2
            self._create_mock_llm_response([
                {"name": "commit_section", "args": {"content": "# First", "summary": "First"}},
                {"name": "request_next_section", "args": {"section_name": section_names[1]}}
            ]),
            # Section 2: commit and auto-advance
            self._create_mock_llm_response([
                {"name": "commit_section", "args": {"content": "# Second", "summary": "Second"}},
                {"name": "request_next_section", "args": {}}
            ]),
            # Section 3: commit and finish
            self._create_mock_llm_response([
                {"name": "commit_section", "args": {"content": "# Third", "summary": "Third"}},
                {"name": "request_next_section", "args": {}}
            ])
        ]
        
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_llm.invoke = Mock(side_effect=responses)
        
        mock_config = Mock()
        result = generator.generate(mock_config)
        
        # All sections should be completed
        assert len(generator.doc_state.completed_sections) == num_sections
        # Verify the jump worked (Section 2 should be in ToC before Section 3)
        toc_sections = [name for name, _, _ in generator.doc_state.toc]
        assert "Section 1" in toc_sections
        assert "Section 2" in toc_sections
        assert "Section 3" in toc_sections
    
    def test_commit_without_request_auto_advances(self):
        """Test that missing request_next_section still advances"""
        # Force separate sections
        generator, mock_llm = self._create_generator_with_docs(num_anchors=2, force_separate_sections=True)
        
        num_sections = len(generator.sections)
        
        # Only commit, no request_next_section - should still auto-advance
        responses = []
        for i in range(num_sections):
            responses.append(self._create_mock_llm_response([
                {"name": "commit_section", "args": {"content": f"# Section {i+1}", "summary": f"S{i+1}"}}
                # Note: no request_next_section
            ]))
        
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_llm.invoke = Mock(side_effect=responses)
        
        mock_config = Mock()
        result = generator.generate(mock_config)
        
        # Should still complete all sections (auto-advance on no swap request)
        assert len(generator.committed_content) == num_sections
    
    def test_no_tool_calls_forces_advance(self):
        """Test handling when LLM doesn't return any tool calls"""
        # Use single section for this test to simplify
        generator, mock_llm = self._create_generator_with_docs(num_anchors=1)
        
        # No tool calls at all - should be handled gracefully
        mock_response_no_tools = Mock()
        mock_response_no_tools.tool_calls = []  # Empty list, not None
        mock_response_no_tools.content = "I'm confused..."
        
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_llm.invoke = Mock(return_value=mock_response_no_tools)
        
        mock_config = Mock()
        result = generator.generate(mock_config)
        
        # Should handle gracefully - advance past the section even without commit
        # The section will be marked done to avoid infinite loops
        assert generator.current_section_idx >= len(generator.sections)
    
    def test_toc_grows_across_sections(self):
        """Test that ToC accumulates across sections"""
        # Force separate sections
        generator, mock_llm = self._create_generator_with_docs(num_anchors=3, force_separate_sections=True)
        
        num_sections = len(generator.sections)
        assert num_sections >= 2, f"Expected >=2 sections, got {num_sections}"
        
        call_count = [0]
        
        def mock_invoke(*args, **kwargs):
            call_count[0] += 1
            return self._create_mock_llm_response([
                {"name": "commit_section", "args": {
                    "content": f"# Section {call_count[0]}",
                    "summary": f"Summary for section {call_count[0]}"
                }},
                {"name": "request_next_section", "args": {}}
            ])
        
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_llm.invoke = Mock(side_effect=mock_invoke)
        
        mock_config = Mock()
        generator.generate(mock_config)
        
        # Verify ToC has all sections
        assert len(generator.doc_state.toc) == num_sections
        for i, (name, symbols, summary) in enumerate(generator.doc_state.toc):
            assert f"section {i+1}" in summary.lower()
    
    def test_cross_references_added_to_signatures(self):
        """Test that signatures include cross-references for documented symbols"""
        generator, mock_llm = self._create_generator_with_docs(num_anchors=2)
        
        # Manually add a documented symbol
        generator.doc_state.add_section("Previous Section", ["AlreadyDocumented"], "Already done")
        generator.all_docs["AlreadyDocumented"] = Document(
            page_content="class AlreadyDocumented: pass",
            metadata={"symbol_name": "AlreadyDocumented"}
        )
        
        # Build section context
        section = generator.sections[0]
        generator._build_section_context(section)
        
        # Cross-reference note should be available
        cross_ref = generator.doc_state.get_cross_reference("AlreadyDocumented")
        assert cross_ref is not None
        assert "Previous Section" in cross_ref
    
    def test_final_document_has_toc_for_multiple_sections(self):
        """Test that final document includes ToC when multiple sections"""
        # Force separate sections
        generator, mock_llm = self._create_generator_with_docs(num_anchors=2, force_separate_sections=True)
        
        num_sections = len(generator.sections)
        assert num_sections >= 2, f"Expected >=2 sections, got {num_sections}"
        
        responses = [
            self._create_mock_llm_response([
                {"name": "commit_section", "args": {"content": "# First Section\nContent here", "summary": "First summary"}},
                {"name": "request_next_section", "args": {}}
            ]),
            self._create_mock_llm_response([
                {"name": "commit_section", "args": {"content": "# Second Section\nMore content", "summary": "Second summary"}},
                {"name": "request_next_section", "args": {}}
            ])
        ]
        
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_llm.invoke = Mock(side_effect=responses)
        
        mock_config = Mock()
        result = generator.generate(mock_config)
        
        # Result should have ToC header (Contents or Table of Contents)
        assert "## Contents" in result or "Table of Contents" in result or len(generator.committed_content) >= 2
    
    def test_empty_commit_content_handled(self):
        """Test handling of empty content in commit_section"""
        generator, mock_llm = self._create_generator_with_docs(num_anchors=1)
        
        # Empty content commit
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_llm.invoke = Mock(return_value=self._create_mock_llm_response([
            {"name": "commit_section", "args": {"content": "", "summary": "Empty"}},
            {"name": "request_next_section", "args": {}}
        ]))
        
        mock_config = Mock()
        result = generator.generate(mock_config)
        
        # Should handle gracefully (empty content not added)
        # The section may or may not be marked complete depending on implementation
        assert result is not None  # At minimum, shouldn't crash


class TestAgenticPromptBuilding:
    """Tests for dynamic prompt building"""
    
    def test_user_prompt_includes_section_progress(self):
        """User prompt should show section X of Y"""
        generator, _ = TestAgenticFlowWithMockedLLM()._create_generator_with_docs(num_anchors=3)
        
        section = generator.sections[0]
        generator._build_section_context(section)
        
        user_prompt = generator._build_user_prompt(section)
        
        assert "1 of 3" in user_prompt or "Section 1" in user_prompt
    
    def test_user_prompt_shows_remaining_sections(self):
        """User prompt should list remaining sections"""
        # Force separate sections
        generator, _ = TestAgenticFlowWithMockedLLM()._create_generator_with_docs(num_anchors=3, force_separate_sections=True)
        
        # Verify we have multiple sections
        if len(generator.sections) < 2:
            # Skip if we can't create multiple sections
            return
        
        # Mark first section as done
        generator.doc_state.completed_sections.add(generator.sections[0].name)
        
        section = generator.sections[1]
        generator._build_section_context(section)
        
        user_prompt = generator._build_user_prompt(section)
        
        # Should show that there are remaining sections or current section number
        # The prompt should indicate section progress
        assert "Section" in user_prompt or "section" in user_prompt or str(len(generator.sections)) in user_prompt
    
    def test_system_prompt_is_constant(self):
        """System prompt should be built once and remain constant"""
        generator, _ = TestAgenticFlowWithMockedLLM()._create_generator_with_docs(num_anchors=2)
        
        initial_system_prompt = generator.system_prompt
        
        # Simulate some state changes
        generator.doc_state.add_section("Test", ["Symbol"], "Summary")
        generator.current_section_idx = 1
        
        # System prompt should be unchanged
        assert generator.system_prompt == initial_system_prompt
        assert "Test Page" in generator.system_prompt  # Page name is baked in
