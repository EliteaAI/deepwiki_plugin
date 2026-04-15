"""
Test C++ declaration/implementation split handling.

Tests that:
1. extern const declarations in headers are extracted with source_text
2. Function declarations without body are extracted with source_text
3. Content expander can augment declarations with implementations
"""

import pytest
import tempfile
import os
from pathlib import Path

from plugin_implementation.parsers.cpp_enhanced_parser import CppEnhancedParser
from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
from plugin_implementation.content_expander import ContentExpander


# Test files content
COLORS_H = """#pragma once
#include <raylib.h>
#include <vector>

extern const Color darkGrey;
extern const Color green;
extern const Color red;

std::vector<Color> GetCellColors();

int CalculateScore(int level, int lines);
"""

COLORS_CPP = """#include "colors.h"

const Color darkGrey = {26, 31, 40, 255};
const Color green = {47, 230, 23, 255};
const Color red = {232, 18, 18, 255};

std::vector<Color> GetCellColors()
{
    return {darkGrey, green, red};
}

int CalculateScore(int level, int lines)
{
    return level * lines * 100;
}
"""


class TestCppDeclarationParsing:
    """Test that C++ declarations are properly extracted with source_text."""
    
    @pytest.fixture
    def parser(self):
        return CppEnhancedParser()
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary C++ files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            header_path = os.path.join(tmpdir, "colors.h")
            impl_path = os.path.join(tmpdir, "colors.cpp")
            
            with open(header_path, 'w') as f:
                f.write(COLORS_H)
            with open(impl_path, 'w') as f:
                f.write(COLORS_CPP)
            
            yield {
                'dir': tmpdir,
                'header': header_path,
                'impl': impl_path
            }
    
    def test_extern_const_declarations_extracted(self, parser, temp_files):
        """Test that extern const declarations are extracted as CONSTANT symbols."""
        result = parser.parse_file(temp_files['header'])
        
        # Find constant symbols
        constants = [s for s in result.symbols if s.symbol_type.value == 'constant']
        constant_names = [s.name for s in constants]
        
        # Should have the extern const declarations
        assert 'darkGrey' in constant_names, f"darkGrey not found in {constant_names}"
        assert 'green' in constant_names, f"green not found in {constant_names}"
        assert 'red' in constant_names, f"red not found in {constant_names}"
    
    def test_extern_const_has_source_text(self, parser, temp_files):
        """Test that extern const declarations have non-empty source_text."""
        result = parser.parse_file(temp_files['header'])
        
        constants = [s for s in result.symbols if s.symbol_type.value == 'constant']
        
        for const in constants:
            assert const.source_text, f"Constant {const.name} has no source_text"
            assert 'extern' in const.source_text or 'const' in const.source_text, \
                f"Constant {const.name} source_text doesn't contain declaration: {const.source_text}"
    
    def test_function_declarations_extracted(self, parser, temp_files):
        """Test that function declarations without body are extracted."""
        result = parser.parse_file(temp_files['header'])
        
        # Find function symbols
        functions = [s for s in result.symbols 
                    if s.symbol_type.value == 'function']
        function_names = [s.name for s in functions]
        
        # Should have the function declarations
        assert 'GetCellColors' in function_names, f"GetCellColors not found in {function_names}"
        assert 'CalculateScore' in function_names, f"CalculateScore not found in {function_names}"
    
    def test_function_declarations_have_source_text(self, parser, temp_files):
        """Test that function declarations have non-empty source_text."""
        result = parser.parse_file(temp_files['header'])
        
        functions = [s for s in result.symbols 
                    if s.symbol_type.value == 'function']
        
        for func in functions:
            assert func.source_text, f"Function {func.name} has no source_text"
            # Should contain the function name and parentheses
            assert func.name in func.source_text, \
                f"Function {func.name} not in source_text: {func.source_text}"
            assert '(' in func.source_text and ')' in func.source_text, \
                f"Function {func.name} source_text missing parentheses: {func.source_text}"
    
    def test_function_declarations_marked_no_body(self, parser, temp_files):
        """Test that function declarations are marked as having no body."""
        result = parser.parse_file(temp_files['header'])
        
        functions = [s for s in result.symbols 
                    if s.symbol_type.value == 'function']
        
        for func in functions:
            metadata = getattr(func, 'metadata', {}) or {}
            assert metadata.get('has_body') == False, \
                f"Function {func.name} should be marked has_body=False"
    
    def test_impl_file_has_definitions(self, parser, temp_files):
        """Test that implementation file has definitions with bodies."""
        result = parser.parse_file(temp_files['impl'])
        
        # Find constants with initializers
        constants = [s for s in result.symbols if s.symbol_type.value == 'constant']
        constant_names = [s.name for s in constants]
        
        assert 'darkGrey' in constant_names, "darkGrey definition not found"
        
        # Check the constant has the initializer in source_text
        darkGrey = next(s for s in constants if s.name == 'darkGrey')
        assert '{' in darkGrey.source_text, \
            f"darkGrey should have initializer in source_text: {darkGrey.source_text}"
    
    def test_impl_functions_have_body(self, parser, temp_files):
        """Test that implementation functions are marked as having body."""
        result = parser.parse_file(temp_files['impl'])
        
        functions = [s for s in result.symbols 
                    if s.symbol_type.value == 'function']
        
        for func in functions:
            metadata = getattr(func, 'metadata', {}) or {}
            # Implementation should have body
            assert metadata.get('has_body') == True, \
                f"Function {func.name} in impl should have has_body=True"


class TestGraphBuilderCppFiltering:
    """Test that graph builder properly indexes C++ declarations."""
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository with C++ files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create colors.h
            header_path = os.path.join(tmpdir, "colors.h")
            with open(header_path, 'w') as f:
                f.write(COLORS_H)
            
            # Create colors.cpp
            impl_path = os.path.join(tmpdir, "colors.cpp")
            with open(impl_path, 'w') as f:
                f.write(COLORS_CPP)
            
            yield tmpdir
    
    def test_declarations_not_filtered_as_empty(self, temp_repo):
        """Test that declarations with source_text are not filtered as empty."""
        builder = EnhancedUnifiedGraphBuilder()
        analysis = builder.analyze_repository(temp_repo)
        
        # Check documents were created
        assert len(analysis.documents) > 0, "No documents created"
        
        # Find constant documents
        constant_docs = [d for d in analysis.documents 
                        if d.metadata.get('symbol_type') == 'constant']
        
        # Should have constants from BOTH header and impl
        constant_names = [d.metadata.get('symbol_name') for d in constant_docs]
        
        # darkGrey should appear (from header declaration and/or impl definition)
        assert 'darkGrey' in constant_names, \
            f"darkGrey not in indexed documents. Found: {constant_names}"
    
    def test_function_declarations_indexed(self, temp_repo):
        """Test that function declarations are indexed."""
        builder = EnhancedUnifiedGraphBuilder()
        analysis = builder.analyze_repository(temp_repo)
        
        # Find function documents
        function_docs = [d for d in analysis.documents 
                        if d.metadata.get('symbol_type') == 'function']
        function_names = [d.metadata.get('symbol_name') for d in function_docs]
        
        # Should have functions
        assert 'GetCellColors' in function_names, \
            f"GetCellColors not indexed. Found: {function_names}"
        assert 'CalculateScore' in function_names, \
            f"CalculateScore not indexed. Found: {function_names}"
    
    def test_both_declaration_and_impl_in_graph(self, temp_repo):
        """Test that both declaration and implementation are in the graph."""
        builder = EnhancedUnifiedGraphBuilder()
        analysis = builder.analyze_repository(temp_repo)
        
        graph = analysis.unified_graph
        
        # Find nodes for darkGrey
        darkGrey_nodes = [n for n in graph.nodes() if 'darkGrey' in n]
        
        # Should have at least one node (could be 2 if from different files)
        assert len(darkGrey_nodes) >= 1, \
            f"Expected darkGrey nodes in graph, found: {darkGrey_nodes}"


class TestContentExpanderCppAugmentation:
    """Test that content expander augments declarations with implementations."""
    
    @pytest.fixture
    def temp_repo_with_graph(self):
        """Create temporary repo and build graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create colors.h
            header_path = os.path.join(tmpdir, "colors.h")
            with open(header_path, 'w') as f:
                f.write(COLORS_H)
            
            # Create colors.cpp  
            impl_path = os.path.join(tmpdir, "colors.cpp")
            with open(impl_path, 'w') as f:
                f.write(COLORS_CPP)
            
            # Build graph
            builder = EnhancedUnifiedGraphBuilder()
            analysis = builder.analyze_repository(tmpdir)
            
            yield {
                'dir': tmpdir,
                'analysis': analysis,
                'graph': analysis.unified_graph,
                'documents': analysis.documents
            }
    
    def test_constant_augmentation(self, temp_repo_with_graph):
        """Test that constant declarations get augmented with definitions."""
        graph = temp_repo_with_graph['graph']
        expander = ContentExpander(graph_store=graph)

        # Find the header constant node
        header_const_node = None
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('symbol_type') == 'constant' and node_data.get('symbol_name') == 'darkGrey':
                file_path = node_data.get('file_path', '')
                if file_path.endswith('colors.h'):
                    header_const_node = node_id
                    break

        assert header_const_node, "Header constant node for darkGrey not found"

        header_doc = expander._create_document_from_graph_node(header_const_node)
        assert header_doc, "Failed to build header constant document"

        augmented_doc = expander._augment_cpp_constant_with_definition(header_const_node, header_doc)
        assert augmented_doc is not None, "Expected constant augmentation for darkGrey"

        assert augmented_doc.metadata.get('has_split_definition') is True
        assert augmented_doc.metadata.get('replaces_original') is True
        assert 'colors.cpp' in (augmented_doc.metadata.get('definition_file') or '')
        assert 'Definition from' in augmented_doc.page_content
        assert 'const Color darkGrey' in augmented_doc.page_content


class TestArchitecturalFiltering:
    """
    Test that content expander filters out non-architectural symbols from output.
    
    Non-architectural symbols (methods, constructors, fields, variables) are used
    INTERNALLY during graph traversal to find transitive relationships, but should
    not appear in the final expansion output.
    """
    
    def test_architectural_filter_excludes_methods(self):
        """Test that methods are filtered from expansion output."""
        from plugin_implementation.content_expander import ContentExpander
        
        expander = ContentExpander(graph_store=None)
        
        # Verify ARCHITECTURAL_SYMBOL_TYPES doesn't include methods/constructors/fields
        assert 'method' not in expander.ARCHITECTURAL_SYMBOL_TYPES
        assert 'constructor' not in expander.ARCHITECTURAL_SYMBOL_TYPES
        assert 'destructor' not in expander.ARCHITECTURAL_SYMBOL_TYPES
        assert 'field' not in expander.ARCHITECTURAL_SYMBOL_TYPES
        assert 'variable' not in expander.ARCHITECTURAL_SYMBOL_TYPES
        
        # Verify architectural types ARE included
        assert 'class' in expander.ARCHITECTURAL_SYMBOL_TYPES
        assert 'struct' in expander.ARCHITECTURAL_SYMBOL_TYPES
        assert 'function' in expander.ARCHITECTURAL_SYMBOL_TYPES
        assert 'constant' in expander.ARCHITECTURAL_SYMBOL_TYPES
        assert 'enum' in expander.ARCHITECTURAL_SYMBOL_TYPES
    
    def test_filter_to_architectural_symbols(self):
        """Test _filter_to_architectural_symbols method directly."""
        from plugin_implementation.content_expander import ContentExpander
        from langchain_core.documents import Document
        
        expander = ContentExpander(graph_store=None)
        
        # Create mix of architectural and non-architectural documents
        docs = [
            # Architectural - should be kept
            Document(page_content="class Foo {}", metadata={'symbol_type': 'class', 'symbol_name': 'Foo'}),
            Document(page_content="void bar() {}", metadata={'symbol_type': 'function', 'symbol_name': 'bar'}),
            Document(page_content="const X = 1", metadata={'symbol_type': 'constant', 'symbol_name': 'X'}),
            Document(page_content="struct Data {}", metadata={'symbol_type': 'struct', 'symbol_name': 'Data'}),
            # Non-architectural - should be filtered
            Document(page_content="def method() {}", metadata={'symbol_type': 'method', 'symbol_name': 'method'}),
            Document(page_content="__init__()", metadata={'symbol_type': 'constructor', 'symbol_name': '__init__'}),
            Document(page_content="int x;", metadata={'symbol_type': 'field', 'symbol_name': 'x'}),
            Document(page_content="int y = 5;", metadata={'symbol_type': 'variable', 'symbol_name': 'y'}),
        ]
        
        filtered = expander._filter_to_architectural_symbols(docs)
        
        # Should have 4 architectural symbols
        assert len(filtered) == 4, f"Expected 4 architectural symbols, got {len(filtered)}"
        
        # Check that only architectural types remain
        symbol_types = {d.metadata.get('symbol_type') for d in filtered}
        assert symbol_types == {'class', 'function', 'constant', 'struct'}
        
        # Verify no non-architectural types
        for doc in filtered:
            symbol_type = doc.metadata.get('symbol_type', '').lower()
            assert symbol_type not in {'method', 'constructor', 'field', 'variable', 'destructor'}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
