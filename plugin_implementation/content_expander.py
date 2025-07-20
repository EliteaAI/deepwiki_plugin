"""
Content Expander for enhancing retrieved documents with related code context.

This module provides post-retrieval content expansion by analyzing code relationships
like inheritance, composition, method calls, and parent-child relationships using
the relationship code_graph from Enhanced Unified Graph Builder.
"""

import logging
import os
from typing import List, Dict, Set, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ContentExpander:
    """
    Expands retrieved documents with related code context using code_graph traversal.
    
    Key features:
    - Expands small symbols (variables, fields) to their parent containers (methods, classes)
    - Analyzes inheritance and composition relationships
    - Intelligent deduplication (prefers classes over individual methods)
    - Code-focused filtering for better retrieval quality
    """
    
    def __init__(self, graph_store=None):
        """
        Initialize content expander.
        
        Args:
            graph_store: NetworkX code_graph from Enhanced Unified Graph Builder
        """
        self.graph = graph_store
        self.logger = logging.getLogger(__name__)
        
        # Symbol type hierarchies for parent expansion
        self.symbol_hierarchy = {
            # Small symbols that need parent expansion
            'variable': ['method', 'function', 'class'],
            'field': ['class', 'interface'],
            'parameter': ['method', 'function', 'constructor'],
            'local_variable': ['method', 'function'],
            
            # Medium symbols that might need class context
            'method': ['class', 'interface'],
            'constructor': ['class'],  # Constructors expand to their enclosing class
            'property': ['class', 'interface'],
            
            # Large symbols that are complete containers - no member expansion needed
            'class': [],  # Classes expand to architectural context, not members
            'interface': [],  # Interfaces expand to architectural context, not members
            'function': [],  # Top-level functions are complete
            'module': []
        }
    
    def expand_retrieved_documents(self, documents: List[Document], apply_code_filters: bool = True) -> List[Document]:
        """
        Main method to expand retrieved documents with related context using code_graph traversal.
        
        Only operates on code documents - documentation, config, and other non-code documents
        are preserved as-is without modification.
        
        Args:
            documents: List of retrieved documents
            apply_code_filters: Whether to apply code-specific filtering and deduplication
            
        Returns:
            List of expanded documents with related context
        """
        if not documents:
            return documents
            
        try:
            # Separate code documents from non-code documents
            code_docs = []
            non_code_docs = []
            
            for doc in documents:
                if self._is_code_document(doc):
                    code_docs.append(doc)
                else:
                    non_code_docs.append(doc)
            
            self.logger.info(f"Document separation: {len(code_docs)} code, {len(non_code_docs)} non-code")
            
            # Only process code documents for expansion
            if code_docs:
                # Start with original code documents
                all_docs = code_docs.copy()
                processed_nodes = set()
                
                # For interfaces, we want to expand them to find implementing classes
                # So we don't pre-mark interface documents as processed
                
                # Don't pre-mark any original documents as processed
                # We want to expand all symbols to find their context
                # The deduplication will handle avoiding duplicate original documents
                
                # Phase 1: Expand each code document with comprehensive context (without original)
                for doc in code_docs:
                    expanded_context = self._expand_document_comprehensively(doc, processed_nodes)
                    all_docs.extend(expanded_context)
                
                # Phase 2: Apply deduplication and filtering to all documents
                if apply_code_filters:
                    final_code_docs = self._apply_code_filters(all_docs)
                else:
                    final_code_docs = self._deduplicate_documents(all_docs)
                
                # Combine deduplicated code documents with untouched non-code documents
                final_docs = final_code_docs + non_code_docs
            else:
                # No code documents to process, return all as-is
                final_docs = non_code_docs
            
            self.logger.info(f"Content expansion: {len(documents)} -> {len(final_docs)} documents")
            return final_docs
            
        except Exception as e:
            self.logger.error(f"Error during content expansion: {e}")
            return documents
    
    def _is_code_document(self, doc: Document) -> bool:
        """
        Determine if a document contains code that should be expanded.
        
        Returns True for code symbols, False for documentation/config/etc.
        Uses chunk_type metadata field for simple and reliable detection.
        """
        chunk_type = doc.metadata.get('chunk_type', '').lower()
        return chunk_type == 'symbol'
    
    def _expand_document_comprehensively(self, doc: Document, processed_nodes: Set[str]) -> List[Document]:
        """
        Expand a single code document with comprehensive context for best-in-class documentation.
        
        Important: Does NOT include the original document - only adds expanded context.
        The original document is already in the retrieved set, so we only add related context.
        
        For each symbol type, provides the most complete context:
        - Classes: Include inheritance hierarchy, constructors, key methods
        - Methods/Functions: Include parent class, called functions, parameters
        - Variables/Fields: Include containing class/method context
        - Small symbols: Expand to meaningful parents
        
        Args:
            doc: Document to expand
            processed_nodes: Set of already processed code_graph nodes to avoid duplicates
            
        Returns:
            List of documents with expanded context (NOT including the original)
        """
        result_docs = []
        
        if not self.graph:
            return result_docs
            
        # Extract symbol information from document metadata
        symbol_info = self._extract_symbol_info_from_document(doc)
        if not symbol_info:
            return result_docs
            
        symbol_name, symbol_type, file_path, language = symbol_info
        
        # Find the code_graph node for this symbol
        graph_node = self._find_graph_node(symbol_name, file_path, language)
        if not graph_node or graph_node in processed_nodes:
            return result_docs
            
        processed_nodes.add(graph_node)
        
        # Comprehensive expansion based on symbol type
        if symbol_type in ['class', 'interface']:
            # For classes/interfaces: provide architectural context (inheritance, composition)
            # NOT member expansion since classes already contain all their members
            self.logger.debug(f"Calling _expand_class_comprehensively for {symbol_name}")
            context_docs = self._expand_class_comprehensively(graph_node, processed_nodes)
            self.logger.debug(f"_expand_class_comprehensively returned {len(context_docs)} documents")
            result_docs.extend(context_docs)
            
        elif symbol_type in ['method', 'function']:
            # For methods/functions: include parent class and related functions
            self.logger.debug(f"Calling _expand_method_comprehensively for {symbol_name}")
            context_docs = self._expand_method_comprehensively(graph_node, processed_nodes, doc)
            self.logger.debug(f"_expand_method_comprehensively returned {len(context_docs)} documents")
            result_docs.extend(context_docs)
            
        elif symbol_type in ['constructor', 'constructor_call']:
            # For constructors and constructor calls: expand appropriately
            self.logger.debug(f"Calling _expand_constructor_comprehensively for {symbol_name}")
            context_docs = self._expand_constructor_comprehensively(graph_node, processed_nodes)
            self.logger.debug(f"_expand_constructor_comprehensively returned {len(context_docs)} documents")
            result_docs.extend(context_docs)
            
        elif symbol_type in ['variable', 'field', 'property']:
            # For small symbols: expand to meaningful containing context
            context_docs = self._expand_small_symbol_comprehensively(graph_node, processed_nodes)
            result_docs.extend(context_docs)
            
        elif symbol_type in ['enum', 'annotation']:
            # For type definitions: include usage context
            context_docs = self._expand_type_definition_comprehensively(graph_node, processed_nodes)
            result_docs.extend(context_docs)
        
        return result_docs
    
    def _expand_document_with_graph(self, doc: Document, processed_nodes: Set[str]) -> List[Document]:
        """
        Expand a single document using code_graph traversal to find meaningful context.
        
        Args:
            doc: Document to expand
            processed_nodes: Set of already processed code_graph nodes to avoid duplicates
            
        Returns:
            List of documents including the original and expanded context
        """
        result_docs = []
        
        # Always include the original document
        result_docs.append(doc)
        
        if not self.graph:
            return result_docs
            
        # Extract symbol information from document metadata
        symbol_info = self._extract_symbol_info_from_document(doc)
        if not symbol_info:
            return result_docs
            
        symbol_name, symbol_type, file_path, language = symbol_info
        
        # Find the code_graph node for this symbol
        graph_node = self._find_graph_node(symbol_name, file_path, language)
        if not graph_node or graph_node in processed_nodes:
            return result_docs
            
        processed_nodes.add(graph_node)
        
        # Check if this symbol needs parent expansion
        if self._needs_parent_expansion(symbol_type):
            parent_docs = self._find_parent_context(graph_node, processed_nodes)
            result_docs.extend(parent_docs)
        
        # For classes and interfaces, also find related content
        if symbol_type in ['class', 'interface']:
            related_docs = self._find_related_content(graph_node, processed_nodes)
            result_docs.extend(related_docs)
        
        return result_docs
    
    def _extract_symbol_info_from_document(self, doc: Document) -> Optional[Tuple[str, str, str, str]]:
        """Extract symbol information from document metadata."""
        metadata = doc.metadata
        
        # Try different metadata formats
        symbol_name = (metadata.get('symbol_name') or 
                      metadata.get('name') or 
                      metadata.get('title', '').split('/')[-1])
        
        symbol_type = (metadata.get('symbol_type') or 
                      metadata.get('type', '').lower())
        
        file_path = (metadata.get('file_path') or 
                    metadata.get('source', ''))
        
        language = (metadata.get('language') or 
                   metadata.get('lang', ''))
        
        if symbol_name and file_path:
            return symbol_name, symbol_type, file_path, language
        
        return None
    
    def _find_graph_node(self, symbol_name: str, file_path: str, language: str) -> Optional[str]:
        """
        Find the code_graph node ID for a symbol.
        
        The enhanced code_graph builder uses the format: language::file_name::symbol_name
        """
        if not self.graph:
            return None
            
        # Try different node ID formats used by enhanced code_graph builder
        file_name = file_path.split('/')[-1].split('.')[0] if file_path else ''
        
        # Handle fully qualified names like com.example.Vehicle -> Vehicle
        simple_name = symbol_name.split('.')[-1] if '.' in symbol_name else symbol_name
        
        possible_node_ids = [
            f"{language}::{file_name}::{symbol_name}",      # exact match
            f"{language}::{file_name}::{simple_name}",      # simple name in same file
            f"{language}::{symbol_name}",                   # exact match without file
            f"{language}::{simple_name}",                   # simple name without file
            symbol_name,                                    # raw symbol name
            simple_name                                     # raw simple name
        ]
        
        # Debug logging to see what we're looking for
        self.logger.debug(f"Looking for code_graph node: symbol='{symbol_name}', file='{file_path}', language='{language}'")
        self.logger.debug(f"File name extracted: '{file_name}'")
        self.logger.debug(f"Trying node IDs: {possible_node_ids}")
        
        for node_id in possible_node_ids:
            if self.graph.has_node(node_id):
                self.logger.debug(f"Found exact node: {node_id}")
                return node_id
                
        # Try partial matching for symbols with qualified names
        for node_id in self.graph.nodes():
            if node_id.endswith(f"::{symbol_name}") or node_id.endswith(f"::{simple_name}"):
                self.logger.debug(f"Found partial match: {node_id}")
                return node_id
                
        # Debug: show available nodes for troubleshooting
        available_nodes = list(self.graph.nodes())[:10]  # First 10 nodes
        self.logger.debug(f"Available nodes (first 10): {available_nodes}")
        
        self.logger.debug(f"No code_graph node found for symbol: {symbol_name}")
        return None
    
    def _needs_parent_expansion(self, symbol_type: str) -> bool:
        """Check if a symbol type needs parent context expansion."""
        return symbol_type.lower() in self.symbol_hierarchy and len(self.symbol_hierarchy[symbol_type.lower()]) > 0
    
    def _find_parent_context(self, graph_node: str, processed_nodes: Set[str]) -> List[Document]:
        """
        Find parent context for a symbol by traversing the code_graph.
        
        Uses both parent_symbol metadata and code_graph edges to find containing symbols.
        """
        parent_docs = []
        
        if not self.graph or not self.graph.has_node(graph_node):
            return parent_docs
            
        node_data = self.graph.nodes[graph_node]
        symbol_type = node_data.get('symbol_type', '').lower()
        
        # Strategy 1: Use parent_symbol metadata
        parent_symbol = node_data.get('parent_symbol')
        if parent_symbol:
            parent_node = self._find_graph_node(
                parent_symbol, 
                node_data.get('file_path', ''), 
                node_data.get('language', '')
            )
            if parent_node and parent_node not in processed_nodes:
                parent_doc = self._create_document_from_graph_node(parent_node)
                if parent_doc:
                    parent_docs.append(parent_doc)
                    processed_nodes.add(parent_node)
        
        # Strategy 2: Find parents through code_graph relationships
        if not parent_docs:
            parent_docs.extend(self._find_parents_through_edges(graph_node, symbol_type, processed_nodes))
        
        # Strategy 3: Find by hierarchical symbol analysis (for variables in methods, etc.)
        if not parent_docs and symbol_type in ['variable', 'field', 'parameter']:
            parent_docs.extend(self._find_containing_symbol(graph_node, processed_nodes))
        
        return parent_docs
    
    def _find_parents_through_edges(self, graph_node: str, symbol_type: str, processed_nodes: Set[str]) -> List[Document]:
        """Find parent symbols through code_graph edges (contains, defined_in relationships)."""
        parent_docs = []
        
        # Look for incoming edges that indicate containment
        for pred in self.graph.predecessors(graph_node):
            if pred in processed_nodes:
                continue
                
            edge_data = self.graph.get_edge_data(pred, graph_node)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get('relationship_type', '').lower()
                    if rel_type in ['contains', 'defines', 'has_member']:
                        parent_doc = self._create_document_from_graph_node(pred)
                        if parent_doc:
                            parent_docs.append(parent_doc)
                            processed_nodes.add(pred)
                            break
        
        return parent_docs
    
    def _find_containing_symbol(self, graph_node: str, processed_nodes: Set[str]) -> List[Document]:
        """Find containing symbol by analyzing node names and file structure."""
        containing_docs = []
        
        node_data = self.graph.nodes[graph_node]
        file_path = node_data.get('file_path', '')
        language = node_data.get('language', '')
        
        # Look for symbols in the same file that could be containers
        file_name = file_path.split('/')[-1].split('.')[0] if file_path else ''
        
        for node_id in self.graph.nodes():
            if node_id in processed_nodes or node_id == graph_node:
                continue
                
            other_node_data = self.graph.nodes[node_id]
            
            # Check if it's in the same file
            if (other_node_data.get('file_path', '') == file_path and
                other_node_data.get('symbol_type', '').lower() in ['class', 'method', 'function']):
                
                # Add as potential container
                container_doc = self._create_document_from_graph_node(node_id)
                if container_doc:
                    containing_docs.append(container_doc)
                    processed_nodes.add(node_id)
        
        return containing_docs[:2]  # Limit to avoid too many results
    
    def _create_document_from_graph_node(self, node_id: str) -> Optional[Document]:
        """Create a Document from a code_graph node."""
        if not self.graph or not self.graph.has_node(node_id):
            logger.debug(f"Graph or node not found for {node_id}")
            return None
            
        node_data = self.graph.nodes[node_id]
        symbol = node_data.get('symbol')
        
        if symbol and hasattr(symbol, 'source_text') and symbol.source_text:
            content = symbol.source_text
        else:
            # Fallback to node data or create placeholder content
            content = node_data.get('content')
            if not content:
                symbol_name = node_data.get('symbol_name', node_id)
                symbol_type = node_data.get('symbol_type', 'symbol')
                content = f"Symbol: {symbol_name} ({symbol_type})"
        
        # Ensure content is not None
        if not content:
            content = f"Graph node: {node_id}"
        
        metadata = {
            'symbol_name': node_data.get('symbol_name', ''),
            'symbol_type': node_data.get('symbol_type', ''),
            'file_path': node_data.get('file_path', ''),
            'language': node_data.get('language', ''),
            'chunk_type': 'symbol',
            'source': 'graph_expansion'
        }
        
        logger.debug(f"Created document from node {node_id}: {metadata['symbol_name']} ({metadata['symbol_type']})")
        return Document(page_content=content, metadata=metadata)
        
        return Document(page_content=content, metadata=metadata)
    
    def _find_related_content(self, graph_node: str, processed_nodes: Set[str]) -> List[Document]:
        """Find related content for classes and interfaces (inheritance, composition) - Legacy method."""
        # This method is now replaced by the comprehensive expansion methods above
        return []
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """
        Deduplicate documents, prioritizing higher-level containers over their members.
        
        Hierarchy (highest to lowest priority):
        1. Classes, Interfaces, Enums, Structs (containers)
        2. Top-level Functions
        3. Methods, Fields, Properties (members)
        4. Variables, Parameters (small symbols)
        
        Logic:
        - If a container (class/interface/enum) and its members are both present, keep only the container
        - If multiple containers are present, keep all of them
        - Keep documentation and other non-code documents
        """
        result = []
        
        # Categorize documents by priority level
        categories = {
            'containers': [],      # classes, interfaces, enums, structs
            'functions': [],       # top-level functions
            'members': [],         # methods, fields, properties
            'small_symbols': [],   # variables, parameters
            'documentation': [],   # markdown, text
            'other': []           # everything else
        }
        
        for doc in documents:
            doc_type = doc.metadata.get('symbol_type', 'other').lower()
            
            if doc_type in ['class', 'interface', 'enum', 'struct', 'annotation']:
                categories['containers'].append(doc)
            elif doc_type in ['function']:
                # Check if it's truly a top-level function (no parent)
                if not (doc.metadata.get('parent_class') or doc.metadata.get('parent_symbol')):
                    categories['functions'].append(doc)
                else:
                    categories['members'].append(doc)  # It's actually a method
            elif doc_type in ['method', 'field', 'property', 'constructor']:
                categories['members'].append(doc)
            elif doc_type in ['variable', 'parameter', 'local_variable']:
                categories['small_symbols'].append(doc)
            elif doc_type in ['markdown', 'text', 'documentation']:
                categories['documentation'].append(doc)
            else:
                categories['other'].append(doc)
        
        # Get container names for deduplication
        container_names = set()
        container_files = set()
        for container_doc in categories['containers']:
            container_name = container_doc.metadata.get('symbol_name')
            container_file = container_doc.metadata.get('file_path', '')
            if container_name:
                container_names.add(container_name)
                # Also track file-level containers for better deduplication
                if container_file:
                    container_files.add((container_name, container_file))
        
        # Always add all containers (highest priority)
        result.extend(categories['containers'])
        self.logger.debug(f"Added {len(categories['containers'])} containers")
        
        # Add top-level functions (no parent container)
        result.extend(categories['functions'])
        self.logger.debug(f"Added {len(categories['functions'])} top-level functions")
        
        # Add members only if their parent container is NOT already included
        included_member_signatures = set()
        for member_doc in categories['members']:
            member_name = member_doc.metadata.get('symbol_name', '')
            member_file = member_doc.metadata.get('file_path', '')
            parent_class = (member_doc.metadata.get('parent_class') or 
                           member_doc.metadata.get('parent_symbol', ''))
            
            # Skip if parent container is already included
            if parent_class in container_names:
                self.logger.debug(f"Skipping member {member_name} - parent {parent_class} already included")
                continue
                
            # Also check file-level container matching
            skip_due_to_file_container = False
            for container_name, container_file in container_files:
                if member_file == container_file and container_name in parent_class:
                    self.logger.debug(f"Skipping member {member_name} - file-level container {container_name} already included")
                    skip_due_to_file_container = True
                    break
            
            if skip_due_to_file_container:
                continue
            
            # Avoid duplicate members (same signature from different containers)
            member_signature = f"{member_name}:{member_doc.metadata.get('symbol_type', '')}"
            if member_signature in included_member_signatures:
                self.logger.debug(f"Skipping duplicate member {member_name}")
                continue
            
            result.append(member_doc)
            included_member_signatures.add(member_signature)
        
        self.logger.debug(f"Added {len([doc for doc in result if doc.metadata.get('symbol_type', '').lower() in ['method', 'field', 'property', 'constructor']])} members after deduplication")
        
        # Add small symbols only if their containing methods/classes are NOT already included
        for small_symbol_doc in categories['small_symbols']:
            symbol_name = small_symbol_doc.metadata.get('symbol_name', '')
            symbol_file = small_symbol_doc.metadata.get('file_path', '')
            parent_symbol = (small_symbol_doc.metadata.get('parent_class') or 
                            small_symbol_doc.metadata.get('parent_symbol', ''))
            
            # Check if parent is already in result
            parent_already_included = False
            for existing_doc in result:
                existing_name = existing_doc.metadata.get('symbol_name', '')
                existing_file = existing_doc.metadata.get('file_path', '')
                if (existing_name == parent_symbol or 
                    (existing_file == symbol_file and existing_name in parent_symbol)):
                    parent_already_included = True
                    break
            
            if not parent_already_included:
                result.append(small_symbol_doc)
        
        # Always add all documentation and other documents
        result.extend(categories['documentation'])
        result.extend(categories['other'])
        
        self.logger.debug(f"Deduplication: {len(documents)} -> {len(result)} documents")
        self.logger.debug(f"  Containers: {len(categories['containers'])}")
        self.logger.debug(f"  Functions: {len(categories['functions'])}")
        self.logger.debug(f"  Members: {len([doc for doc in result if doc.metadata.get('symbol_type', '').lower() in ['method', 'field', 'property', 'constructor']])}")
        self.logger.debug(f"  Documentation: {len(categories['documentation'])}")
        self.logger.debug(f"  Other: {len(categories['other'])}")
        
        return result
    
    def _apply_code_filters(self, documents: List[Document]) -> List[Document]:
        """
        Apply code-specific filtering and deduplication.
        
        Separates code documents from documentation and applies intelligent
        deduplication logic (prefer classes over methods).
        """
        code_docs = []
        doc_docs = []
        other_docs = []
        
        # Separate document types
        for doc in documents:
            doc_type = doc.metadata.get('symbol_type', '').lower()
            if doc_type in ['class', 'interface', 'function', 'method', 'variable', 'field']:
                code_docs.append(doc)
            elif doc_type in ['markdown', 'text', 'documentation']:
                doc_docs.append(doc)
            else:
                other_docs.append(doc)
        
        # Apply deduplication to code documents
        deduplicated_code = self._deduplicate_documents(code_docs)
        
        self.logger.debug(f"Code filtering: {len(documents)} -> {len(deduplicated_code + doc_docs + other_docs)} documents")
        self.logger.debug(f"  Code docs: {len(code_docs)} -> {len(deduplicated_code)}")
        self.logger.debug(f"  Doc docs: {len(doc_docs)}")
        self.logger.debug(f"  Other docs: {len(other_docs)}")
        
        return deduplicated_code + doc_docs + other_docs
    
    def _expand_class_comprehensively(self, class_node: str, processed_nodes: Set[str]) -> List[Document]:
        """
        Provide comprehensive architectural context for a class.
        
        Classes are complete containers that already include all their members (methods, fields, constructors).
        Instead, we expand classes to provide broader architectural understanding:
        - Parent classes (inheritance hierarchy) 
        - Implementing classes (for interfaces)
        - Related classes (composition relationships)
        
        This gives architectural context without duplicating what's already in the class.
        """
        context_docs = []
        
        if not self.graph or not self.graph.has_node(class_node):
            self.logger.debug(f"Graph or node not found for {class_node}")
            return context_docs
            
        node_data = self.graph.nodes[class_node]
        class_name = node_data.get('symbol_name', '')
        symbol_type = node_data.get('symbol_type', '').lower()
        
        self.logger.debug(f"Expanding {symbol_type} {class_name} for architectural context")
        
        # 1. Find inheritance context (parent classes, implemented interfaces, implementing classes)
        inheritance_docs = self._find_inheritance_context(class_node, processed_nodes)
        self.logger.debug(f"Found {len(inheritance_docs)} inheritance/implementation docs")
        context_docs.extend(inheritance_docs)
        
        # 2. Find composition relationships (classes this class uses)
        composition_docs = self._find_composition_context(class_node, processed_nodes, limit=3)
        self.logger.debug(f"Found {len(composition_docs)} composition docs")
        context_docs.extend(composition_docs)
        
        self.logger.debug(f"{symbol_type.title()} {class_name} expanded with {len(context_docs)} architectural context documents")
        return context_docs
    
    def _expand_method_comprehensively(self, method_node: str, processed_nodes: Set[str], original_doc: Document = None) -> List[Document]:
        """
        Provide comprehensive context for a method including parent class and related functions.
        
        Best-in-class documentation for a method should include:
        - Parent class context
        - Overridden methods from parent classes
        - Methods this method calls
        - Related methods in the same class
        
        Args:
            method_node: Graph node ID for the method
            processed_nodes: Set of already processed nodes to avoid duplicates
            original_doc: Original document containing metadata about specific method instance
        """
        context_docs = []
        
        if not self.graph or not self.graph.has_node(method_node):
            return context_docs
            
        node_data = self.graph.nodes[method_node]
        method_name = node_data.get('symbol_name', '')
        
        self.logger.debug(f"Expanding method {method_name} comprehensively")
        
        # 1. Find parent class (essential context) - use original doc metadata if available
        parent_docs = self._find_parent_context_for_method(method_node, processed_nodes, original_doc)
        context_docs.extend(parent_docs)
        
        # 2. Find methods this method calls
        called_methods = self._find_called_methods(method_node, processed_nodes, limit=2)
        logger.debug(f"Found {len(called_methods)} called methods")
        context_docs.extend(called_methods)
        
        # 3. Find methods/classes that call this method (callers)
        caller_docs = self._find_method_callers(method_node, processed_nodes, limit=2)
        logger.debug(f"Found {len(caller_docs)} method callers")
        context_docs.extend(caller_docs)
        
        # 4. Find overridden methods in parent classes
        overridden_docs = self._find_overridden_methods(method_node, processed_nodes)
        logger.debug(f"Found {len(overridden_docs)} overridden methods")
        context_docs.extend(overridden_docs)
        
        logger.debug(f"Method {method_name} expanded with {len(context_docs)} additional documents")
        return context_docs
    
    def _expand_constructor_comprehensively(self, constructor_node: str, processed_nodes: Set[str]) -> List[Document]:
        """
        Expand constructors to provide comprehensive context.
        
        Handles two cases:
        1. Constructor Definition (def __init__(self):) → expand to enclosing class
        2. Constructor Call/Invocation (MyClass() in method) → expand to:
           - The class being instantiated 
           - The enclosing method containing the call
           - The class containing that method
        
        Args:
            constructor_node: Graph node ID for the constructor
            processed_nodes: Set of already processed nodes to avoid duplicates
            
        Returns:
            List of documents providing comprehensive constructor context
        """
        context_docs = []
        
        if not self.graph or not self.graph.has_node(constructor_node):
            return context_docs
            
        constructor_data = self.graph.nodes[constructor_node]
        constructor_name = constructor_data.get('symbol_name', 'Unknown')
        symbol_type = constructor_data.get('symbol_type', '').lower()
        
        # Determine if this is a constructor call or constructor definition
        is_constructor_call = self._is_constructor_call(constructor_node, constructor_data)
        
        self.logger.debug(f"Constructor node: {constructor_node}")
        self.logger.debug(f"Constructor data: {constructor_data}")
        self.logger.debug(f"Is constructor call: {is_constructor_call}")
        
        if is_constructor_call:
            # Handle constructor call: MyClass() inside another method
            self.logger.debug(f"Expanding constructor call {constructor_name}")
            context_docs.extend(self._expand_constructor_call(constructor_node, processed_nodes))
        else:
            # Handle constructor definition: def __init__(self):
            self.logger.debug(f"Expanding constructor definition {constructor_name}")
            context_docs.extend(self._expand_constructor_definition(constructor_node, processed_nodes))
        
        self.logger.debug(f"Constructor {constructor_name} expanded with {len(context_docs)} additional documents")
        return context_docs
    
    def _is_constructor_call(self, constructor_node: str, constructor_data: Dict) -> bool:
        """
        Determine if this is a constructor call vs constructor definition.
        
        Based on our investigation, constructor detection should use:
        - Nodes ending with .<init> suffix are constructor definitions
        - Methods calling constructor definitions via 'calls' relationship are constructor calls
        """
        symbol_name = constructor_data.get('symbol_name', '')
        symbol_type = constructor_data.get('symbol_type', '').lower()
        
        # Check if this is a constructor definition (ends with .<init>)
        if symbol_name.endswith('.<init>'):
            return False  # This is a constructor definition, not a call
        
        # Direct check for constructor_call type
        if 'constructor_call' in symbol_type:
            return True
        
        # Check for call-related types
        if any(keyword in symbol_type for keyword in ['call', 'invocation', 'instantiation']):
            return True
            
        # Check if this method calls any constructor definitions (nodes ending with .<init>)
        # Look for outgoing edges to .<init> nodes
        for succ in self.graph.successors(constructor_node):
            succ_data = self.graph.nodes.get(succ, {})
            succ_name = succ_data.get('symbol_name', '')
            if succ_name.endswith('.<init>'):
                # Check edge type
                edge_data = self.graph.get_edge_data(constructor_node, succ)
                if edge_data:
                    for edge in edge_data.values():
                        rel_type = edge.get('relationship_type', '').lower()
                        if rel_type in ['calls', 'invokes', 'instantiates']:
                            return True
        
        return False
    
    def _expand_constructor_call(self, call_node: str, processed_nodes: Set[str]) -> List[Document]:
        """
        Expand constructor call like MyClass() inside OtherClass.__init__().
        
        Should expand to:
        1. MyClass (the class being instantiated)
        2. __init__ method containing the call (enclosing method)  
        3. OtherClass (class containing the enclosing method)
        """
        context_docs = []
        
        call_data = self.graph.nodes[call_node]
        
        # 1. Find the class being instantiated (target of the constructor call)
        instantiated_class_docs = self._find_instantiated_class(call_node, processed_nodes)
        context_docs.extend(instantiated_class_docs)
        
        # 2. Find the enclosing method that contains this constructor call
        enclosing_method_docs = self._find_parent_context(call_node, processed_nodes)
        context_docs.extend(enclosing_method_docs)
        
        # 3. Find the class containing the enclosing method (go up one more level)
        for method_doc in enclosing_method_docs:
            method_symbol_info = self._extract_symbol_info_from_document(method_doc)
            if method_symbol_info:
                method_name, method_type, file_path, language = method_symbol_info
                if method_type.lower() in ['method', 'function', 'constructor']:
                    method_node = self._find_graph_node(method_name, file_path, language)
                    if method_node:
                        # Find the class containing this method
                        containing_class_docs = self._find_parent_context(method_node, processed_nodes)
                        context_docs.extend(containing_class_docs)
                        break  # Only need one containing class
        
        return context_docs
    
    def _expand_constructor_definition(self, constructor_node: str, processed_nodes: Set[str]) -> List[Document]:
        """
        Expand constructor definition like def __init__(self): to its enclosing class.
        
        This is the original logic for constructor definitions.
        """
        context_docs = []
        
        # 1. Find the enclosing class - this is the primary context for constructor definitions
        parent_class_docs = self._find_parent_context(constructor_node, processed_nodes)
        context_docs.extend(parent_class_docs)
        
        # 2. Find other constructors in the same class (constructor overloading)
        if parent_class_docs:
            class_node = None
            # Find the class node from parent containers
            for doc in parent_class_docs:
                # Extract symbol info and find code_graph node
                symbol_info = self._extract_symbol_info_from_document(doc)
                if symbol_info:
                    symbol_name, symbol_type, file_path, language = symbol_info
                    if symbol_type.lower() == 'class':
                        class_node = self._find_graph_node(symbol_name, file_path, language)
                        break
            
            if class_node:
                # Find other constructors in the same class
                other_constructors = self._find_constructors(class_node, processed_nodes)
                # Filter out the current constructor to avoid duplicates
                for ctor_doc in other_constructors:
                    ctor_symbol_info = self._extract_symbol_info_from_document(ctor_doc)
                    if ctor_symbol_info:
                        ctor_name, ctor_type, ctor_file, ctor_lang = ctor_symbol_info
                        ctor_node = self._find_graph_node(ctor_name, ctor_file, ctor_lang)
                        if ctor_node != constructor_node:
                            context_docs.append(ctor_doc)
        
        return context_docs
    
    def _find_instantiated_class(self, call_node: str, processed_nodes: Set[str]) -> List[Document]:
        """
        Find the class that is being instantiated by a constructor call.
        
        For MyClass() call, find the MyClass class definition.
        """
        instantiated_docs = []
        
        # Look for outgoing edges to classes that this call instantiates
        for succ in self.graph.successors(call_node):
            if succ in processed_nodes:
                continue
                
            edge_data = self.graph.get_edge_data(call_node, succ)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get('relationship_type', '').lower()
                    if rel_type in ['instantiates', 'creates', 'calls']:
                        succ_data = self.graph.nodes.get(succ, {})
                        succ_type = succ_data.get('symbol_type', '').lower()
                        if succ_type in ['class', 'interface']:
                            instantiated_doc = self._create_document_from_graph_node(succ)
                            if instantiated_doc:
                                instantiated_docs.append(instantiated_doc)
                                processed_nodes.add(succ)
        
        # Alternative: look for class with same name as constructor call
        if not instantiated_docs:
            call_data = self.graph.nodes[call_node]
            class_name = call_data.get('symbol_name', '')
            if class_name:
                class_node = self._find_graph_node(
                    class_name, 
                    call_data.get('file_path', ''), 
                    call_data.get('language', '')
                )
                if class_node and class_node not in processed_nodes:
                    class_doc = self._create_document_from_graph_node(class_node)
                    if class_doc:
                        instantiated_docs.append(class_doc)
                        processed_nodes.add(class_node)
        
        return instantiated_docs
    
    def _expand_small_symbol_comprehensively(self, symbol_node: str, processed_nodes: Set[str]) -> List[Document]:
        """
        Expand small symbols (variables, fields) to their most meaningful containing context.
        
        For variables: method + class context
        For fields: class context + related methods that use this field
        """
        context_docs = []
        
        if not self.graph or not self.graph.has_node(symbol_node):
            return context_docs
            
        node_data = self.graph.nodes[symbol_node]
        symbol_name = node_data.get('symbol_name', '')
        symbol_type = node_data.get('symbol_type', '').lower()
        
        self.logger.debug(f"Expanding small symbol {symbol_name} ({symbol_type}) comprehensively")
        
        if symbol_type in ['variable', 'parameter']:
            # Variables need method and class context
            context_docs.extend(self._find_parent_context(symbol_node, processed_nodes))
            
        elif symbol_type in ['field', 'property']:
            # Fields need class context + methods that use them
            context_docs.extend(self._find_parent_context(symbol_node, processed_nodes))
            context_docs.extend(self._find_methods_using_field(symbol_node, processed_nodes, limit=2))
        
        self.logger.debug(f"Small symbol {symbol_name} expanded with {len(context_docs)} additional documents")
        return context_docs
    
    def _expand_type_definition_comprehensively(self, type_node: str, processed_nodes: Set[str]) -> List[Document]:
        """
        Expand type definitions (enums, annotations) with usage context.
        """
        context_docs = []
        
        if not self.graph or not self.graph.has_node(type_node):
            return context_docs
            
        node_data = self.graph.nodes[type_node]
        type_name = node_data.get('symbol_name', '')
        
        self.logger.debug(f"Expanding type definition {type_name} comprehensively")
        
        # Find classes/methods that use this type
        usage_docs = self._find_type_usage_context(type_node, processed_nodes, limit=2)
        context_docs.extend(usage_docs)
        
        return context_docs
    
    def _find_inheritance_context(self, class_node: str, processed_nodes: Set[str]) -> List[Document]:
        """Find parent classes, implemented interfaces, and child classes."""
        inheritance_docs = []
        
        # Get file path for this class to check file-level relationships
        node_data = self.graph.nodes[class_node]
        class_file_path = node_data.get('file_path', '')
        class_language = node_data.get('language', '')
        
        # Find the file-level node for this class
        file_node = None
        if class_file_path:
            file_name = os.path.splitext(os.path.basename(class_file_path))[0]
            possible_file_nodes = [
                f"{class_language}::{file_name}::__file__",
                f"{class_language}::{file_name}::__file__".replace("::", "_"),
            ]
            for possible_node in possible_file_nodes:
                if self.graph.has_node(possible_node):
                    file_node = possible_node
                    break
        
        # Function to check relationships from a given node
        def check_relationships_from_node(source_node):
            local_docs = []
            # Look for inheritance and implements relationships (outgoing from node)
            for succ in self.graph.successors(source_node):
                if succ in processed_nodes:
                    continue
                    
                edge_data = self.graph.get_edge_data(source_node, succ)
                if edge_data:
                    for edge in edge_data.values():
                        rel_type = edge.get('relationship_type', '').lower()
                        if rel_type in ['inheritance', 'extends', 'implements', 'implementation']:
                            parent_doc = self._create_document_from_graph_node(succ)
                            if parent_doc:
                                local_docs.append(parent_doc)
                                processed_nodes.add(succ)
            return local_docs
        
        # Check relationships from the class node itself
        inheritance_docs.extend(check_relationships_from_node(class_node))
        
        # Also check relationships from the file node (for interface implementations)
        if file_node and file_node != class_node:
            inheritance_docs.extend(check_relationships_from_node(file_node))
        
        # Also look for incoming inheritance relationships (classes that inherit from this class)
        for pred in self.graph.predecessors(class_node):
            if pred in processed_nodes or len(inheritance_docs) >= 6:
                continue
                
            edge_data = self.graph.get_edge_data(pred, class_node)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get('relationship_type', '').lower()
                    if rel_type in ['inheritance', 'extends']:
                        child_doc = self._create_document_from_graph_node(pred)
                        if child_doc:
                            inheritance_docs.append(child_doc)
                            processed_nodes.add(pred)
        
        # For interfaces: also look for implementation relationships (incoming to interface)
        symbol_type = node_data.get('symbol_type', '').lower()
        if symbol_type == 'interface':
            for pred in self.graph.predecessors(class_node):
                if len(inheritance_docs) >= 6:  # Limit for implementations but don't skip processed nodes
                    continue
                    
                edge_data = self.graph.get_edge_data(pred, class_node)
                if edge_data:
                    for edge in edge_data.values():
                        rel_type = edge.get('relationship_type', '').lower()
                        if rel_type in ['implementation', 'implements']:
                            pred_node_data = self.graph.nodes[pred]
                            pred_symbol_type = pred_node_data.get('symbol_type', '').lower()
                            
                            # For interfaces, we want implementing classes even if they're processed elsewhere
                            # Only skip if we've already added this specific node in this expansion
                            if pred in processed_nodes:
                                # Check if this implementing class is already in our inheritance_docs
                                already_added = False
                                for existing_doc in inheritance_docs:
                                    existing_symbol_info = self._extract_symbol_info_from_document(existing_doc)
                                    if existing_symbol_info:
                                        existing_name, existing_type, existing_file, existing_lang = existing_symbol_info
                                        existing_node = self._find_graph_node(existing_name, existing_file, existing_lang)
                                        if existing_node == pred:
                                            already_added = True
                                            break
                                
                                if already_added:
                                    continue
                            
                            # If this is a file-level implementation, find classes within that file
                            if pred_symbol_type in ['module', 'file']:
                                pred_file_path = pred_node_data.get('file_path', '')
                                # Find classes in the same file that implement this interface
                                for node_id in self.graph.nodes():
                                    # Skip if we already have too many implementations
                                    if len(inheritance_docs) >= 6:
                                        break
                                        
                                    other_node_data = self.graph.nodes[node_id]
                                    other_symbol_type = other_node_data.get('symbol_type', '').lower()
                                    other_file_path = other_node_data.get('file_path', '')
                                    
                                    # Find classes in the same file
                                    if (other_symbol_type == 'class' and 
                                        other_file_path == pred_file_path):
                                        
                                        # Check if we already added this class
                                        already_added = False
                                        for existing_doc in inheritance_docs:
                                            existing_symbol_info = self._extract_symbol_info_from_document(existing_doc)
                                            if existing_symbol_info:
                                                existing_name, existing_type, existing_file, existing_lang = existing_symbol_info
                                                existing_node = self._find_graph_node(existing_name, existing_file, existing_lang)
                                                if existing_node == node_id:
                                                    already_added = True
                                                    break
                                        
                                        if not already_added:
                                            class_doc = self._create_document_from_graph_node(node_id)
                                            if class_doc:
                                                inheritance_docs.append(class_doc)
                                                processed_nodes.add(node_id)
                            else:
                                # Direct class implementation
                                impl_doc = self._create_document_from_graph_node(pred)
                                if impl_doc:
                                    inheritance_docs.append(impl_doc)
                                    processed_nodes.add(pred)
        
        return inheritance_docs[:4]  # Increased limit for interfaces with implementations
    
    def _find_constructors(self, class_node: str, processed_nodes: Set[str]) -> List[Document]:
        """Find constructors for a class."""
        constructor_docs = []
        
        node_data = self.graph.nodes[class_node]
        class_name = node_data.get('symbol_name', '')
        file_path = node_data.get('file_path', '')
        language = node_data.get('language', '')
        
        logger.debug(f"Looking for constructors for class {class_name} in file {file_path}")
        
        # Look for constructor nodes
        found_candidates = 0
        for node_id in self.graph.nodes():
            if node_id in processed_nodes:
                continue
                
            other_node_data = self.graph.nodes[node_id]
            other_symbol_type = other_node_data.get('symbol_type', '').lower()
            other_file_path = other_node_data.get('file_path', '')
            other_parent_symbol = other_node_data.get('parent_symbol')
            
            # Check for constructor patterns
            is_constructor = (other_symbol_type == 'constructor' or 
                             node_id.endswith('.<init>') or
                             other_symbol_type == 'method' and other_node_data.get('symbol_name', '') == class_name)
            
            is_same_file = other_file_path == file_path
            is_parent_match = (other_parent_symbol == class_name or class_name in node_id)
            
            if is_constructor and is_same_file:
                found_candidates += 1
                logger.debug(f"Constructor candidate: {node_id}, symbol_type: {other_symbol_type}, parent: {other_parent_symbol}")
                
                if is_parent_match:
                    constructor_doc = self._create_document_from_graph_node(node_id)
                    if constructor_doc:
                        constructor_docs.append(constructor_doc)
                        processed_nodes.add(node_id)
                        logger.debug(f"Added constructor: {node_id}")
        
        logger.debug(f"Found {found_candidates} constructor candidates, added {len(constructor_docs)} constructors")
        return constructor_docs[:2]  # Limit to 2 constructors
    
    def _find_key_methods(self, class_node: str, processed_nodes: Set[str], limit: int = 3) -> List[Document]:
        """Find key public methods of a class."""
        method_docs = []
        
        node_data = self.graph.nodes[class_node]
        class_name = node_data.get('symbol_name', '')
        file_path = node_data.get('file_path', '')
        
        # Get fully qualified class name from parent_symbol
        class_parent = node_data.get('parent_symbol', '')
        if class_parent:
            fully_qualified_class = f"{class_parent}.{class_name}"
        else:
            fully_qualified_class = class_name
        
        logger.debug(f"Looking for methods with parent_symbol: {fully_qualified_class} or {class_name}")
        
        # Look for method nodes in the same class
        found_candidates = 0
        for node_id in self.graph.nodes():
            if node_id in processed_nodes or len(method_docs) >= limit:
                continue
                
            other_node_data = self.graph.nodes[node_id]
            other_symbol_type = other_node_data.get('symbol_type', '').lower()
            other_file_path = other_node_data.get('file_path', '')
            other_parent_symbol = other_node_data.get('parent_symbol', '')
            
            # Check if it's a method in the same file with matching parent
            is_method = other_symbol_type == 'method'
            is_same_file = other_file_path == file_path
            is_parent_match = (other_parent_symbol == fully_qualified_class or 
                              other_parent_symbol == class_name)
            
            if is_method and is_same_file:
                found_candidates += 1
                logger.debug(f"Method candidate: {node_id}, parent: {other_parent_symbol}")
                
                if is_parent_match:
                    method_doc = self._create_document_from_graph_node(node_id)
                    if method_doc:
                        method_docs.append(method_doc)
                        processed_nodes.add(node_id)
                        logger.debug(f"Added method: {node_id}")
        
        logger.debug(f"Found {found_candidates} method candidates, added {len(method_docs)} methods")
        return method_docs
    
    def _find_important_fields(self, class_node: str, processed_nodes: Set[str], limit: int = 2) -> List[Document]:
        """Find important fields/properties of a class."""
        field_docs = []
        
        node_data = self.graph.nodes[class_node]
        class_name = node_data.get('symbol_name', '')
        file_path = node_data.get('file_path', '')
        
        # Get fully qualified class name from parent_symbol
        class_parent = node_data.get('parent_symbol', '')
        if class_parent:
            fully_qualified_class = f"{class_parent}.{class_name}"
        else:
            fully_qualified_class = class_name
        
        logger.debug(f"Looking for fields with parent_symbol: {fully_qualified_class} or {class_name}")
        
        # Look for field nodes in the same class
        found_candidates = 0
        for node_id in self.graph.nodes():
            if node_id in processed_nodes or len(field_docs) >= limit:
                continue
                
            other_node_data = self.graph.nodes[node_id]
            other_symbol_type = other_node_data.get('symbol_type', '').lower()
            other_file_path = other_node_data.get('file_path', '')
            other_parent_symbol = other_node_data.get('parent_symbol', '')
            
            # Check if it's a field in the same file with matching parent
            is_field = other_symbol_type in ['field', 'property']
            is_same_file = other_file_path == file_path
            is_parent_match = (other_parent_symbol == fully_qualified_class or 
                              other_parent_symbol == class_name)
            
            if is_field and is_same_file:
                found_candidates += 1
                logger.debug(f"Field candidate: {node_id}, parent: {other_parent_symbol}")
                
                if is_parent_match:
                    field_doc = self._create_document_from_graph_node(node_id)
                    if field_doc:
                        field_docs.append(field_doc)
                        processed_nodes.add(node_id)
                        logger.debug(f"Added field: {node_id}")
        
        logger.debug(f"Found {found_candidates} field candidates, added {len(field_docs)} fields")
        return field_docs
    
    def _find_composition_context(self, class_node: str, processed_nodes: Set[str], limit: int = 2) -> List[Document]:
        """Find classes that this class uses (composition relationships)."""
        composition_docs = []
        
        # Look for 'uses' or 'calls' relationships to other classes
        for succ in self.graph.successors(class_node):
            if succ in processed_nodes or len(composition_docs) >= limit:
                continue
                
            edge_data = self.graph.get_edge_data(class_node, succ)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get('relationship_type', '').lower()
                    if rel_type in ['uses', 'composition', 'aggregation']:
                        comp_doc = self._create_document_from_graph_node(succ)
                        if comp_doc and comp_doc.metadata.get('type', '').lower() in ['class', 'interface']:
                            composition_docs.append(comp_doc)
                            processed_nodes.add(succ)
        
        return composition_docs
    
    def _find_called_methods(self, method_node: str, processed_nodes: Set[str], limit: int = 2) -> List[Document]:
        """Find methods that this method calls."""
        called_docs = []
        
        # Look for 'calls' relationships
        for succ in self.graph.successors(method_node):
            if succ in processed_nodes or len(called_docs) >= limit:
                continue
                
            edge_data = self.graph.get_edge_data(method_node, succ)
            if edge_data:
                for edge in edge_data.values():
                    rel_type = edge.get('relationship_type', '').lower()
                    if rel_type in ['calls', 'invokes']:
                        called_doc = self._create_document_from_graph_node(succ)
                        if called_doc:
                            called_docs.append(called_doc)
                            processed_nodes.add(succ)
        
        return called_docs
    
    def _find_method_callers(self, method_node: str, processed_nodes: Set[str], limit: int = 2) -> List[Document]:
        """Find methods and classes that call this method."""
        caller_docs = []
        
        if not self.graph or not self.graph.has_node(method_node):
            return caller_docs
        
        node_data = self.graph.nodes[method_node]
        method_name = node_data.get('symbol_name', '')
        method_file = node_data.get('file_path', '')
        
        # Get the class that owns this method
        method_parent = node_data.get('parent_symbol', '')
        method_class_name = method_parent.split('.')[-1] if method_parent else ''
        
        logger.debug(f"Looking for callers of method {method_name} in class {method_class_name}")
        
        # Strategy 1: Look for direct calls to this method
        for source, target, edge_data in self.graph.in_edges(method_node, data=True):
            if len(caller_docs) >= limit:
                break
                
            relationship_type = edge_data.get('relationship_type', '')
            
            if relationship_type == 'calls':
                source_node_data = self.graph.nodes[source]
                source_symbol_name = source_node_data.get('symbol_name', '')
                source_symbol_type = source_node_data.get('symbol_type', '')
                
                logger.debug(f"Found direct caller: {source_symbol_name} ({source_symbol_type}) calls {method_name}")
                
                if source not in processed_nodes:
                    caller_doc = self._create_document_from_graph_node(source)
                    if caller_doc:
                        caller_docs.append(caller_doc)
                        processed_nodes.add(source)
        
        # Strategy 2: Look for method calls through object references
        # Pattern: method --[calls]--> object_var, where object_var is of type that has our method
        for source, target, edge_data in self.graph.edges(data=True):
            if len(caller_docs) >= limit:
                break
                
            relationship_type = edge_data.get('relationship_type', '')
            
            if relationship_type == 'calls':
                source_node_data = self.graph.nodes[source]
                target_node_data = self.graph.nodes[target]
                
                source_symbol_name = source_node_data.get('symbol_name', '')
                source_symbol_type = source_node_data.get('symbol_type', '')
                target_symbol_name = target_node_data.get('symbol_name', '')
                target_symbol_type = target_node_data.get('symbol_type', '')
                
                # Check if source method calls a variable/parameter that could be our class type
                if (source_symbol_type == 'method' and 
                    target_symbol_type in ['variable', 'parameter'] and
                    source not in processed_nodes):
                    
                    # Look for patterns where the variable name suggests it's our class type
                    # or where there are relationships showing object instantiation
                    is_potential_caller = False
                    
                    # Pattern 1: Variable named like our class (e.g., 'app' for 'Application')
                    if (target_symbol_name.lower() in method_class_name.lower() or 
                        method_class_name.lower() in target_symbol_name.lower()):
                        is_potential_caller = True
                        logger.debug(f"Found potential caller by name pattern: {source_symbol_name} calls {target_symbol_name} (matches {method_class_name})")
                    
                    # Pattern 2: Look for instantiation relationships
                    # Check if the target variable has relationships to our method's class
                    for var_source, var_target, var_edge_data in self.graph.edges(data=True):
                        var_rel_type = var_edge_data.get('relationship_type', '')
                        if (var_target == target and var_rel_type in ['calls', 'references']):
                            var_source_data = self.graph.nodes[var_source]
                            var_source_name = var_source_data.get('symbol_name', '')
                            if var_source_name == method_class_name:
                                is_potential_caller = True
                                logger.debug(f"Found caller by instantiation: {source_symbol_name} calls {target_symbol_name} of type {var_source_name}")
                                break
                    
                    if is_potential_caller:
                        caller_doc = self._create_document_from_graph_node(source)
                        if caller_doc:
                            caller_docs.append(caller_doc)
                            processed_nodes.add(source)
                            logger.debug(f"Added caller: {source_symbol_name}")
                            
                            # Also add the class that contains this calling method
                            parent_class_docs = self._find_parent_context(source, processed_nodes)
                            caller_docs.extend(parent_class_docs)
        
        logger.debug(f"Found {len(caller_docs)} caller documents for method {method_name}")
        return caller_docs
    
    def _find_overridden_methods(self, method_node: str, processed_nodes: Set[str]) -> List[Document]:
        """Find methods in parent classes that this method overrides."""
        # TODO: Implement overridden method detection
        # For now, return empty list to avoid errors
        return []
    
    def _find_type_usage_context(self, type_node: str, processed_nodes: Set[str], limit: int = 2) -> List[Document]:
        """Find classes and methods that use this type definition."""
        usage_docs = []
        
        if not self.graph or not self.graph.has_node(type_node):
            return usage_docs
            
        type_data = self.graph.nodes[type_node]
        type_name = type_data.get('symbol_name', '')
        symbol_type = type_data.get('symbol_type', '').lower()
        
        # Look for incoming relationships where other symbols reference this type
        for source, target, edge_data in self.graph.in_edges(type_node, data=True):
            if len(usage_docs) >= limit:
                break
                
            relationship_type = edge_data.get('relationship_type', '')
            
            # For interfaces, include implementation relationships
            if symbol_type == 'interface':
                if relationship_type in ['references', 'uses', 'field_type', 'implementation', 'implements']:
                    if source not in processed_nodes:
                        usage_doc = self._create_document_from_graph_node(source)
                        if usage_doc:
                            usage_docs.append(usage_doc)
                            processed_nodes.add(source)
            else:
                # For other types (enums, annotations), use standard relationships
                if relationship_type in ['references', 'uses', 'field_type']:
                    if source not in processed_nodes:
                        usage_doc = self._create_document_from_graph_node(source)
                        if usage_doc:
                            usage_docs.append(usage_doc)
                            processed_nodes.add(source)
        
        return usage_docs
    
    def _find_methods_using_field(self, field_node: str, processed_nodes: Set[str], limit: int = 2) -> List[Document]:
        """Find methods that access or modify this field."""
        method_docs = []
        
        if not self.graph or not self.graph.has_node(field_node):
            return method_docs
            
        field_data = self.graph.nodes[field_node]
        field_name = field_data.get('symbol_name', '')
        
        # Look for incoming relationships where methods reference this field
        for source, target, edge_data in self.graph.in_edges(field_node, data=True):
            if len(method_docs) >= limit:
                break
                
            relationship_type = edge_data.get('relationship_type', '')
            
            if relationship_type in ['references', 'uses', 'accesses', 'modifies']:
                source_data = self.graph.nodes.get(source, {})
                source_type = source_data.get('symbol_type', '').lower()
                
                if source_type == 'method' and source not in processed_nodes:
                    method_doc = self._create_document_from_graph_node(source)
                    if method_doc:
                        method_docs.append(method_doc)
                        processed_nodes.add(source)
        
        return method_docs
    
    def _find_parent_context_for_method(self, method_node: str, processed_nodes: Set[str], original_doc: Document = None) -> List[Document]:
        """
        Find parent context for a method using both code_graph and document metadata.
        
        When multiple methods with the same name exist (e.g., start method in Engine, Vehicle, Car),
        the code_graph may have a single unified node. Use the original document's metadata to
        determine which specific class contains this method instance.
        """
        parent_docs = []
        
        if not self.graph or not self.graph.has_node(method_node):
            return parent_docs
        
        # Strategy 1: Use original document metadata to find containing class
        if original_doc and original_doc.metadata:
            metadata = original_doc.metadata
            file_path = metadata.get('file_path', '')
            language = metadata.get('language', '')
            
            # Look for containing class information in the document content or nearby lines
            content = original_doc.page_content
            start_line = metadata.get('start_line', 0)
            
            # Try to infer containing class from content structure
            containing_class = self._infer_containing_class_from_content(content, file_path, start_line)
            if containing_class:
                # Find the class node in the code_graph
                class_node = self._find_graph_node(containing_class, file_path, language)
                if class_node and class_node not in processed_nodes:
                    class_doc = self._create_document_from_graph_node(class_node)
                    if class_doc:
                        parent_docs.append(class_doc)
                        processed_nodes.add(class_node)
                        self.logger.debug(f"Found containing class from metadata: {containing_class}")
                        return parent_docs
        
        # Strategy 2: Fall back to original parent finding logic
        if not parent_docs:
            parent_docs = self._find_parent_context(method_node, processed_nodes)
        
        return parent_docs
    
    def _infer_containing_class_from_content(self, content: str, file_path: str, start_line: int) -> str:
        """
        Infer the containing class from method content and file structure.
        
        For Python, look for class definition patterns above the method.
        """
        if not content:
            return None
            
        # For Python methods, look for class definition pattern
        lines = content.split('\n')
        
        # If this is just the method content, we need to look at file structure
        # Look for class keyword in the content or use heuristics
        
        # Simple heuristic: if method calls self.something, it's an instance method
        if 'self.' in content:
            # Try to read the file and find the class containing this method
            try:
                if file_path and os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        file_lines = f.readlines()
                    
                    # Find the class that contains the method at start_line
                    current_class = None
                    for i, line in enumerate(file_lines):
                        line_num = i + 1
                        if line_num >= start_line:
                            break
                        
                        stripped = line.strip()
                        if stripped.startswith('class ') and ':' in stripped:
                            # Extract class name
                            class_def = stripped[6:].split(':')[0].strip()
                            if '(' in class_def:
                                current_class = class_def.split('(')[0].strip()
                            else:
                                current_class = class_def
                    
                    return current_class
            except:
                pass
        
        return None

def expand_content_simple(documents: List[Document]) -> List[Document]:
    """
    Simple content expansion function that can be used without a code_graph store.
    
    Args:
        documents: List of retrieved documents
        
    Returns:
        List of documents with basic deduplication applied
    """
    expander = ContentExpander()
    return expander.expand_retrieved_documents(documents)
