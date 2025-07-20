"""
GitHub-aware Code Splitter

Extends the existing GraphAwareCodeSplitter to work with GitHub-sourced content.
Handles content from file paths and uses the filter manager for language detection.
"""

import logging
from typing import Dict, List, Tuple, Set

import networkx as nx
from langchain.docstore.document import Document

from .code_splitter import GraphAwareCodeSplitter, SymbolInfo

logger = logging.getLogger(__name__)


class GitHubCodeSplitter(GraphAwareCodeSplitter):
    """Extended code splitter that works with GitHub-sourced content"""
    
    def __init__(self, 
                 filter_manager,
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200):
        """
        Initialize GitHub code splitter
        
        Args:
            filter_manager: FilterManager instance for language detection
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.filter_manager = filter_manager
    
    def split_github_repository(self, 
                               text_files: List[str],
                               code_files: List[str],
                               repository_traverser) -> Tuple[List[Document], List[Document], nx.DiGraph]:
        """
        Split GitHub repository files into documents and build relationship code_graph
        
        Args:
            text_files: List of text file paths
            code_files: List of code file paths
            repository_traverser: RepositoryTraverser for content retrieval
            
        Returns:
            - Text documents (README, docs)
            - Code documents (with metadata)
            - Combined relationship code_graph
        """
        text_docs = []
        code_docs = []
        
        logger.info(f"Processing {len(text_files)} text files and {len(code_files)} code files")
        
        # Process text files
        for file_path in text_files:
            try:
                docs = self._process_github_text_file(file_path, repository_traverser)
                text_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to process text file {file_path}: {e}")
        
        # Process code files
        for file_path in code_files:
            try:
                docs = self._process_github_code_file(file_path, repository_traverser)
                code_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to process code file {file_path}: {e}")
        
        # Build combined code_graph
        self._build_combined_graph()
        
        logger.info(f"Processed {len(text_docs)} text documents and {len(code_docs)} code documents")
        logger.info(f"Built code_graph with {self.combined_graph.number_of_nodes()} nodes and {self.combined_graph.number_of_edges()} edges")
        
        return text_docs, code_docs, self.combined_graph
    
    def _process_github_text_file(self, 
                                 file_path: str, 
                                 repository_traverser) -> List[Document]:
        """Process a GitHub text/documentation file"""
        
        # Get file content
        content = repository_traverser.get_file_content(file_path)
        if content is None:
            logger.warning(f"Could not read content for {file_path}")
            return []
        
        # Get base metadata
        metadata = repository_traverser.get_file_metadata(file_path, content)
        metadata['chunk_type'] = 'text'
        
        # Handle different text formats
        if file_path.lower().endswith(('.md', '.markdown')):
            return self._process_markdown_content(content, metadata)
        else:
            return self._process_plain_text_content(content, metadata)
    
    def _process_markdown_content(self, content: str, base_metadata: Dict) -> List[Document]:
        """Process markdown content with header-based splitting"""
        try:
            # Use markdown splitter from parent class
            docs = self.markdown_splitter.split_text(content)
            
            result_docs = []
            for i, doc in enumerate(docs):
                metadata = base_metadata.copy()
                metadata.update({
                    'chunk_index': i,
                    'header': doc.metadata.get('Header 1', ''),
                    'section': doc.metadata.get('Header 2', ''),
                    'subsection': doc.metadata.get('Header 3', ''),
                    'format': 'markdown'
                })
                
                result_docs.append(Document(
                    page_content=doc.page_content,
                    metadata=metadata
                ))
            
            return result_docs
            
        except Exception as e:
            logger.warning(f"Failed to split markdown content: {e}")
            # Fallback to plain text processing
            return self._process_plain_text_content(content, base_metadata)
    
    def _process_plain_text_content(self, content: str, base_metadata: Dict) -> List[Document]:
        """Process plain text content with simple chunking"""
        # Simple text chunking
        chunks = []
        
        if len(content) <= self.chunk_size:
            chunks = [content]
        else:
            # Split by lines first, then by size
            lines = content.split('\n')
            current_chunk = ""
            
            for line in lines:
                if len(current_chunk) + len(line) + 1 <= self.chunk_size:
                    current_chunk += line + '\n'
                else:
                    if current_chunk:
                        chunks.append(current_chunk.rstrip())
                    current_chunk = line + '\n'
            
            if current_chunk:
                chunks.append(current_chunk.rstrip())
        
        # Create documents
        result_docs = []
        for i, chunk in enumerate(chunks):
            metadata = base_metadata.copy()
            metadata.update({
                'chunk_index': i,
                'format': 'plain_text'
            })
            
            result_docs.append(Document(
                page_content=chunk,
                metadata=metadata
            ))
        
        return result_docs
    
    def _process_github_code_file(self, 
                                 file_path: str, 
                                 repository_traverser) -> List[Document]:
        """Process a GitHub code file with AST parsing"""
        
        # Get file content
        content = repository_traverser.get_file_content(file_path)
        if content is None:
            logger.warning(f"Could not read content for {file_path}")
            return []
        
        # Get language from filter manager
        language = self.filter_manager.get_file_language(file_path)
        if not language:
            logger.warning(f"Unknown language for {file_path}")
            return []
        
        # Get base metadata
        metadata = repository_traverser.get_file_metadata(file_path, content)
        metadata['chunk_type'] = 'code'
        
        # Check if we can use tree-sitter parsing
        if self._can_parse_with_tree_sitter(language):
            return self._process_code_with_ast(content, file_path, language, metadata)
        else:
            return self._process_code_without_ast(content, file_path, language, metadata)
    
    def _can_parse_with_tree_sitter(self, language: str) -> bool:
        """Check if we can parse the language with tree-sitter"""
        # Check if tree-sitter is available and parsers are initialized
        try:
            import tree_sitter_languages
            if not hasattr(self, 'parsers') or not self.parsers:
                return False
            return language in self.parsers
        except ImportError:
            return False
    
    def _process_code_with_ast(self, 
                              content: str, 
                              file_path: str, 
                              language: str, 
                              base_metadata: Dict) -> List[Document]:
        """Process code file with AST parsing using tree-sitter"""
        try:
            # Use parent class method for AST parsing
            # This will populate self.symbol_table and graphs
            symbols = self._extract_symbols_from_content(content, file_path, language)
            
            # Create documents for each symbol
            code_docs = []
            
            for symbol in symbols:
                # Get symbol content
                lines = content.split('\n')
                symbol_lines = lines[symbol.start_line:symbol.end_line + 1]
                symbol_content = '\n'.join(symbol_lines)
                
                # Create metadata
                metadata = base_metadata.copy()
                metadata.update({
                    'symbol': symbol.name,
                    'node_type': symbol.node_type,
                    'start_line': symbol.start_line,
                    'end_line': symbol.end_line,
                    'docstring': symbol.docstring,
                    'parameters': symbol.parameters,
                    'return_type': symbol.return_type
                })
                
                # Add symbol info to document
                doc = Document(
                    page_content=symbol_content,
                    metadata=metadata
                )
                code_docs.append(doc)
            
            # If no symbols found, create chunks from the whole file
            if not code_docs:
                code_docs = self._create_code_chunks(content, file_path, language, base_metadata)
            
            return code_docs
            
        except Exception as e:
            logger.warning(f"AST parsing failed for {file_path}: {e}")
            # Fallback to simple chunking
            return self._process_code_without_ast(content, file_path, language, base_metadata)
    
    def _process_code_without_ast(self, 
                                 content: str, 
                                 file_path: str, 
                                 language: str, 
                                 base_metadata: Dict) -> List[Document]:
        """Process code file without AST parsing using simple chunking"""
        return self._create_code_chunks(content, file_path, language, base_metadata)
    
    def _create_code_chunks(self, 
                           content: str, 
                           file_path: str, 
                           language: str, 
                           base_metadata: Dict) -> List[Document]:
        """Create code chunks using simple text splitting"""
        chunks = []
        
        if len(content) <= self.chunk_size:
            chunks = [content]
        else:
            # Split by lines and maintain indentation
            lines = content.split('\n')
            current_chunk = ""
            current_lines = []
            
            for i, line in enumerate(lines):
                if len(current_chunk) + len(line) + 1 <= self.chunk_size:
                    current_chunk += line + '\n'
                    current_lines.append(i)
                else:
                    if current_chunk:
                        chunks.append((current_chunk.rstrip(), current_lines))
                    current_chunk = line + '\n'
                    current_lines = [i]
            
            if current_chunk:
                chunks.append((current_chunk.rstrip(), current_lines))
        
        # Create documents
        code_docs = []
        for i, (chunk_content, line_numbers) in enumerate(chunks):
            metadata = base_metadata.copy()
            metadata.update({
                'chunk_index': i,
                'start_line': line_numbers[0] if line_numbers else 0,
                'end_line': line_numbers[-1] if line_numbers else 0,
                'chunk_type': 'code'
            })
            
            code_docs.append(Document(
                page_content=chunk_content,
                metadata=metadata
            ))
        
        return code_docs
    
    def _extract_symbols_from_content(self, 
                                    content: str, 
                                    file_path: str, 
                                    language: str) -> List[SymbolInfo]:
        """Extract symbols from code content using tree-sitter"""
        try:
            # Check if we have parsers available and can parse this language
            if not self._can_parse_with_tree_sitter(language):
                logger.debug(f"Tree-sitter parser not available for language {language}")
                return []
            
            # Call the parent's _process_with_ast method and extract symbols from the result
            documents = self._process_with_ast(content, file_path, language)
            
            # Extract symbol info from the documents
            symbols = []
            for doc in documents:
                if doc.metadata.get('symbol'):
                    symbol_info = SymbolInfo(
                        name=doc.metadata['symbol'],
                        node_type=doc.metadata.get('node_type', 'unknown'),
                        start_line=doc.metadata.get('start_line', 0),
                        end_line=doc.metadata.get('end_line', 0),
                        file_path=file_path,
                        language=language
                    )
                    symbol_info.code_content = doc.page_content
                    symbol_info.docstring = doc.metadata.get('docstring')
                    symbol_info.parameters = doc.metadata.get('parameters', [])
                    symbol_info.return_type = doc.metadata.get('return_type')
                    symbol_info.imports = set(doc.metadata.get('imports', []))
                    symbol_info.calls = set(doc.metadata.get('calls', []))
                    symbols.append(symbol_info)
            
            return symbols
        except Exception as e:
            logger.warning(f"Symbol extraction failed for {file_path}: {e}")
            return []
    
    def get_supported_languages(self) -> Set[str]:
        """Get set of supported languages for AST parsing"""
        if hasattr(self, 'parsers'):
            return set(self.parsers.keys())
        else:
            return set()
    
    def get_processing_stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        return {
            'total_symbols': len(self.symbol_table),
            'call_graph_nodes': self.call_graph.number_of_nodes(),
            'call_graph_edges': self.call_graph.number_of_edges(),
            'import_graph_nodes': self.import_graph.number_of_nodes(),
            'import_graph_edges': self.import_graph.number_of_edges(),
            'combined_graph_nodes': self.combined_graph.number_of_nodes(),
            'combined_graph_edges': self.combined_graph.number_of_edges()
        }
