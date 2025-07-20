"""
GitHub-native Indexer

Main indexer that orchestrates GitHub-based repository indexing with filtering,
extension-aware chunking, and proper caching. Replaces the file-system based indexer.

Compatible with both the full Alita SDK GitHubClient and the standalone GitHubClient.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING
from datetime import datetime

import networkx as nx
from langchain.docstore.document import Document

from .code_graph.graph_builder import EnhancedUnifiedGraphBuilder
from .filter_manager import FilterManager
from .repository_traverser import GitHubRepositoryTraverser
from .github_code_splitter import GitHubCodeSplitter
from .vectorstore import VectorStoreManager
from .graph_manager import GraphManager

if TYPE_CHECKING:
    from .github_client import StandaloneGitHubClient

logger = logging.getLogger(__name__)


class GitHubIndexer:
    """GitHub-native indexer for repository content with filtering and caching"""
    
    def __init__(self, 
                 github_client: Union[Any, "StandaloneGitHubClient"],
                 cache_dir: Optional[str] = None,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 model_cache_dir: Optional[str] = None,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 repo_config_path: Optional[str] = None,
                 api_filters: Optional[Dict[str, Any]] = None,
                 rate_limit_delay: float = 0.1,
                 use_enhanced_graph_builder: bool = True,
                 parallel_file_workers: int = 10,
                 parallel_parse_workers: int = 8):
        """
        Initialize GitHub indexer
        
        Args:
            github_client: GitHubClient instance (from Alita SDK or standalone)
            cache_dir: Directory for caching vector stores and graphs
            model_name: Model for embeddings (single model for both text and code)
            model_cache_dir: Directory for caching HuggingFace models
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            repo_config_path: Path to repo.json configuration
            api_filters: Additional filters from API level
            rate_limit_delay: Delay between GitHub API calls
            use_enhanced_graph_builder: Whether to use Enhanced Unified Graph Builder
            parallel_file_workers: Number of parallel workers for file content fetching
            parallel_parse_workers: Number of parallel workers for file parsing (CPU-bound)
        """
        self.github_client = github_client
        self.use_enhanced_graph_builder = use_enhanced_graph_builder
        self.parallel_file_workers = parallel_file_workers
        self.parallel_parse_workers = parallel_parse_workers
        
        # Initialize components
        self.filter_manager = FilterManager(
            repo_config_path=repo_config_path,
            api_filters=api_filters
        )
        
        self.repository_traverser = GitHubRepositoryTraverser(
            github_client=github_client,
            filter_manager=self.filter_manager,
            rate_limit_delay=rate_limit_delay
        )
        
        if use_enhanced_graph_builder:
            logger.info("Initializing GitHubIndexer with Enhanced Unified Graph Builder")
            self.graph_builder = EnhancedUnifiedGraphBuilder(max_workers=self.parallel_parse_workers)
            self.code_splitter = None
        else:
            logger.info("Initializing GitHubIndexer with legacy GitHubCodeSplitter")
            self.code_splitter = GitHubCodeSplitter(
                filter_manager=self.filter_manager,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            self.graph_builder = None
        
        self.vectorstore_manager = VectorStoreManager(
            cache_dir=cache_dir,
            model_name=model_name,
            model_cache_dir=model_cache_dir
        )
        
        self.graph_manager = GraphManager(cache_dir=cache_dir)
        
        # State
        self.text_documents: List[Document] = []
        self.code_documents: List[Document] = []
        self.all_documents: List[Document] = []
        self.relationship_graph: Optional[nx.DiGraph] = None
        self.vectorstore = None
        self.last_index_stats = None
        
        # Statistics
        self.indexing_stats = {}
    
    def index_repository(self, 
                        repository_name: Optional[str] = None,
                        branch: Optional[str] = None,
                        base_path: str = "",
                        force_rebuild: bool = False,
                        max_files: Optional[int] = None,
                        max_depth: Optional[int] = None) -> Dict[str, Any]:
        """
        Index GitHub repository with caching
        
        Args:
            repository_name: GitHub repository name (owner/repo) or None for current
            branch: Branch to index or None for active branch
            base_path: Starting directory path (empty for root)
            force_rebuild: Whether to force rebuild even if cache exists
            max_files: Maximum number of files to index
            max_depth: Maximum directory depth to traverse
            
        Returns:
            Dictionary with indexing results and statistics
        """
        start_time = datetime.now()
        
        # Set repository and branch if provided
        if repository_name:
            original_repo = self.github_client.github_repository
            self.github_client.github_repository = repository_name
            # Note: Would need to reinitialize github_repo_instance
        
        if branch:
            original_branch = self.github_client.active_branch
            self.github_client.active_branch = branch
        
        try:
            repo_identifier = f"{self.github_client.github_repository}:{self.github_client.active_branch}"
            logger.info(f"Starting GitHub repository indexing: {repo_identifier}")
            
            # Check if we can load from cache
            if not force_rebuild:
                cached_result = self._try_load_from_cache(repo_identifier)
                if cached_result:
                    cached_result['indexing_time'] = (datetime.now() - start_time).total_seconds()
                    return cached_result
            
            # Build index from scratch
            result = self._build_index(repo_identifier, base_path, max_files, max_depth)
            result['indexing_time'] = (datetime.now() - start_time).total_seconds()
            
            return result
            
        finally:
            # Restore original settings
            if repository_name:
                self.github_client.github_repository = original_repo
            if branch:
                self.github_client.active_branch = original_branch
    
    def _try_load_from_cache(self, repo_identifier: str) -> Optional[Dict[str, Any]]:
        """Try to load complete index from cache"""
        try:
            # Check if code_graph exists
            if not self.graph_manager.graph_exists(repo_identifier):
                logger.debug("Graph cache not found")
                return None
            
            # Load code_graph
            self.relationship_graph = self.graph_manager.load_graph(repo_identifier)
            if not self.relationship_graph:
                return None
            
            # Try to load vector store
            dummy_docs = []  # Empty for cache key generation
            self.vectorstore, self.all_documents = self.vectorstore_manager.load_or_build(
                dummy_docs, repo_identifier, force_rebuild=False
            )
            
            if not self.vectorstore or not self.all_documents:
                return None
            
            # Separate documents by type
            if self.use_enhanced_graph_builder:
                # EUGB uses different metadata structure
                self.text_documents = [doc for doc in self.all_documents 
                                     if doc.metadata.get('chunk_type') == 'text']
                self.code_documents = [doc for doc in self.all_documents 
                                     if doc.metadata.get('chunk_type') in ['symbol', 'code']]
            else:
                # Legacy code splitter metadata
                self.text_documents = [doc for doc in self.all_documents 
                                     if doc.metadata.get('chunk_type') == 'text']
                self.code_documents = [doc for doc in self.all_documents 
                                     if doc.metadata.get('chunk_type') == 'code']
            
            logger.info(f"Loaded complete index from cache for {repo_identifier}")
            return self._generate_index_stats(repo_identifier, from_cache=True)
            
        except Exception as e:
            logger.debug(f"Failed to load from cache: {e}")
            return None
    
    def _build_index(self, 
                    repo_identifier: str,
                    base_path: str,
                    max_files: Optional[int],
                    max_depth: Optional[int]) -> Dict[str, Any]:
        """Build index from scratch"""
        logger.info("Building index from scratch...")
        
        # Step 1: Discover files
        logger.info("Discovering repository files...")
        discovered_files = self.repository_traverser.discover_files(
            base_path=base_path,
            max_files=max_files,
            max_depth=max_depth
        )
        
        if not discovered_files:
            logger.warning("No files discovered for indexing")
            return self._generate_empty_stats(repo_identifier)
        
        # Step 2: Categorize files
        logger.info("Categorizing files by type...")
        text_files, code_files = self.repository_traverser.get_files_by_type(discovered_files)
        
        # Step 3: Process files and build code_graph
        logger.info("Processing files and building relationship code_graph...")
        
        if self.use_enhanced_graph_builder:
            # Use Enhanced Unified Graph Builder for comprehensive analysis
            logger.info("Using Enhanced Unified Graph Builder for file processing")
            
            # Convert GitHub files to file paths for EUGB
            all_file_paths = []
            
            # Process code files in parallel to speed up GitHub API calls
            logger.info(f"Fetching content for {len(code_files)} code files in parallel...")
            github_content_map = self._fetch_files_parallel(code_files, max_workers=self.parallel_file_workers)
            
            # Convert to EUGB format with temporary paths
            file_content_map = {}
            for file_path, file_content in github_content_map.items():
                if file_content:
                    # Create temporary file path representation
                    temp_file_path = f"/tmp/github_repo/{file_path}"
                    all_file_paths.append(temp_file_path)
                    file_content_map[temp_file_path] = file_content
            
            # Also process text files if they exist
            if text_files:
                logger.info(f"Fetching content for {len(text_files)} text files in parallel...")
                text_content_map = self._fetch_files_parallel(text_files, max_workers=self.parallel_file_workers)
                
                # Add text files to the content map for processing
                for file_path, content in text_content_map.items():
                    if content:
                        temp_file_path = f"/tmp/github_repo/{file_path}"
                        all_file_paths.append(temp_file_path)
                        file_content_map[temp_file_path] = content
                        all_file_paths.append(temp_file_path)
                        file_content_map[temp_file_path] = content
            
            # For now, use EUGB with local file simulation
            # TODO: Implement proper GitHub file integration in EUGB
            if all_file_paths:
                # Create a temporary directory structure in memory
                import tempfile
                import os
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Write files to temp directory
                    for temp_path, content in file_content_map.items():
                        actual_path = os.path.join(temp_dir, temp_path.replace("/tmp/github_repo/", ""))
                        os.makedirs(os.path.dirname(actual_path), exist_ok=True)
                        
                        # Handle binary content
                        if isinstance(content, str):
                            with open(actual_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                        else:
                            with open(actual_path, 'wb') as f:
                                f.write(content)
                    
                    # Run EUGB analysis on temp directory
                    analysis = self.graph_builder.analyze_repository(temp_dir)
                    
                    # Extract results
                    self.all_documents = analysis.documents
                    self.relationship_graph = analysis.unified_graph
                    
                    # Separate documents by type
                    self.text_documents = [doc for doc in self.all_documents 
                                         if doc.metadata.get('chunk_type') == 'text']
                    self.code_documents = [doc for doc in self.all_documents 
                                         if doc.metadata.get('chunk_type') in ['symbol', 'code']]
                    
                    # Update document metadata with GitHub info
                    for doc in self.all_documents:
                        # Update file paths to reflect GitHub paths
                        if 'file_path' in doc.metadata:
                            original_path = doc.metadata['file_path']
                            github_path = original_path.replace(temp_dir + "/", "")
                            doc.metadata['file_path'] = github_path
                            doc.metadata['source'] = github_path
                            doc.metadata['repository'] = self.github_client.github_repository
                            doc.metadata['branch'] = self.github_client.active_branch
            else:
                logger.warning("No valid code files found for EUGB processing")
                self.text_documents = []
                self.code_documents = []
                self.all_documents = []
                self.relationship_graph = None
                
        else:
            # Use legacy GitHub code splitter
            self.text_documents, self.code_documents, self.relationship_graph = self.code_splitter.split_github_repository(
                text_files, code_files, self.repository_traverser
            )
            
            # Combine all documents
            self.all_documents = self.text_documents + self.code_documents
        
        if not self.all_documents:
            logger.warning("No documents created from repository files")
            return self._generate_empty_stats(repo_identifier)
        
        # Combine all documents for single embedding model
        all_documents = self.text_documents + self.code_documents
        
        # Step 4: Build vector store
        logger.info("Building vector store with single embedding model...")
        self.vectorstore, self.all_documents = self.vectorstore_manager.load_or_build(
            all_documents, 
            repo_identifier, 
            force_rebuild=True
        )
        
        # Step 5: Save code_graph to cache
        if self.relationship_graph:
            logger.info("Saving relationship code_graph to cache...")
            self.graph_manager.save_graph(self.relationship_graph, repo_identifier)
        
        logger.info("Index building complete")
        index_stats = self._generate_index_stats(repo_identifier, from_cache=False)
        self.last_index_stats = index_stats
        return index_stats
    
    def _fetch_files_parallel(self, file_paths: List[str], max_workers: int = 10) -> Dict[str, str]:
        """
        Fetch file contents in parallel to speed up GitHub API calls
        
        Args:
            file_paths: List of file paths to fetch
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping file path to content
        """
        import concurrent.futures
        import time
        
        file_content_map = {}
        
        def fetch_single_file(file_path: str) -> Tuple[str, Optional[str]]:
            """Fetch content for a single file"""
            try:
                content = self.repository_traverser.get_file_content(file_path)
                return file_path, content
            except Exception as e:
                logger.warning(f"Failed to get content for {file_path}: {e}")
                return file_path, None
        
        # Use ThreadPoolExecutor for I/O bound GitHub API calls
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(fetch_single_file, file_path): file_path 
                             for file_path in file_paths}
            
            # Collect results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_file):
                file_path, content = future.result()
                if content is not None:
                    file_content_map[file_path] = content
                
                completed += 1
                if completed % 50 == 0:  # Log progress every 50 files
                    logger.info(f"Fetched content for {completed}/{len(file_paths)} files...")
        
        fetch_time = time.time() - start_time
        successful_files = len(file_content_map)
        logger.info(f"Parallel file fetching complete: {successful_files}/{len(file_paths)} files fetched in {fetch_time:.2f}s")
        
        return file_content_map

    def _generate_empty_stats(self, repo_identifier: str) -> Dict[str, Any]:
        """Generate stats for empty repository"""
        return {
            'repository_identifier': repo_identifier,
            'loaded_from_cache': False,
            'documents': {
                'total': 0,
                'text_documents': 0,
                'code_documents': 0,
                'file_types': {}
            },
            'code_analysis': {
                'languages': [],
                'symbols_by_type': {},
                'files_with_symbols': 0,
                'total_symbols': 0
            },
            'relationship_graph': {
                'nodes': 0,
                'edges': 0
            },
            'vector_store': {
                'has_vectorstore': False,
                'embedding_dimensions': None
            },
            'github_stats': self.repository_traverser.get_traversal_stats(),
            'filter_stats': self.filter_manager.get_filter_summary(),
            'processing_stats': self._get_processing_stats()
        }
    
    def _generate_index_stats(self, repo_identifier: str, from_cache: bool = False) -> Dict[str, Any]:
        """Generate comprehensive indexing statistics"""
        # Basic document analysis
        file_types = {}
        languages = set()
        
        for doc in self.all_documents:
            file_type = doc.metadata.get('file_type') or doc.metadata.get('language', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
            
            if doc.metadata.get('language'):
                languages.add(doc.metadata['language'])
        
        # Symbol analysis for code documents
        symbols_by_type = {}
        files_with_symbols = set()
        
        for doc in self.code_documents:
            if self.use_enhanced_graph_builder:
                # EUGB uses symbol_name and symbol_type in metadata
                if doc.metadata.get('symbol_name'):
                    symbol_type = doc.metadata.get('symbol_type', 'unknown')
                    symbols_by_type[symbol_type] = symbols_by_type.get(symbol_type, 0) + 1
                    files_with_symbols.add(doc.metadata.get('file_path', ''))
            else:
                # Legacy code splitter uses symbol and node_type
                if doc.metadata.get('symbol'):
                    node_type = doc.metadata.get('node_type', 'unknown')
                    symbols_by_type[node_type] = symbols_by_type.get(node_type, 0) + 1
                    files_with_symbols.add(doc.metadata.get('source', ''))
        
        # Graph analysis
        graph_analysis = {}
        if self.relationship_graph:
            graph_analysis = self.graph_manager.analyze_graph(self.relationship_graph)
        
        stats = {
            'repository_identifier': repo_identifier,
            'loaded_from_cache': from_cache,
            'documents': {
                'total': len(self.all_documents),
                'text_documents': len(self.text_documents),
                'code_documents': len(self.code_documents),
                'file_types': file_types
            },
            'code_analysis': {
                'languages': list(languages),
                'symbols_by_type': symbols_by_type,
                'files_with_symbols': len(files_with_symbols),
                'total_symbols': sum(symbols_by_type.values())
            },
            'relationship_graph': graph_analysis,
            'vector_store': {
                'has_vectorstore': self.vectorstore is not None,
                'embedding_dimensions': getattr(self.vectorstore, 'd', None) if self.vectorstore else None
            },
            'github_stats': self.repository_traverser.get_traversal_stats(),
            'filter_stats': self.filter_manager.get_filter_summary(),
            'processing_stats': self._get_processing_stats()
        }
        
        return stats
    
    def _get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics from appropriate processor"""
        if self.use_enhanced_graph_builder and hasattr(self.graph_builder, 'last_analysis'):
            # Return EUGB statistics
            analysis = self.graph_builder.last_analysis
            if analysis:
                return {
                    'processor_type': 'enhanced_unified_graph_builder',
                    'language_stats': analysis.language_stats,
                    'rich_languages': list(analysis.rich_languages),
                    'basic_languages': list(analysis.basic_languages),
                    'cross_language_relationships': len(analysis.cross_language_relationships)
                }
        elif self.code_splitter and hasattr(self.code_splitter, 'get_processing_stats'):
            # Return legacy code splitter statistics
            stats = self.code_splitter.get_processing_stats()
            stats['processor_type'] = 'github_code_splitter'
            return stats
        
        return {'processor_type': 'unknown'}
    
    # Keep all the existing interface methods for compatibility
    
    def get_vectorstore(self):
        """Get the vector store"""
        return self.vectorstore
    
    def get_all_documents(self) -> List[Document]:
        """Get all indexed documents"""
        return self.all_documents
    
    def get_relationship_graph(self) -> Optional[nx.DiGraph]:
        """Get the relationship code_graph"""
        return self.relationship_graph
    
    def search_documents(self, query: str, k: int = 20, include_scores: bool = False) -> List[Document]:
        """Search documents using vector similarity"""
        if not self.vectorstore:
            logger.warning("No vector store available for search")
            return []
        
        if include_scores:
            return self.vectorstore_manager.search_with_score(query, k=k)
        else:
            return self.vectorstore_manager.search_combined(query, k=k)
    
    def search_by_type(self, query: str, doc_type: str = 'all', k: int = 20) -> List[Document]:
        """Search documents by type (text, code, or all)"""
        if not self.vectorstore:
            return []
        
        if doc_type == 'text':
            return self.vectorstore_manager.search_text(query, k=k)
        elif doc_type == 'code':
            return self.vectorstore_manager.search_code(query, k=k)
        else:
            return self.vectorstore_manager.search_combined(query, k=k)
    
    def find_related_symbols(self, symbol_name: str, hops: int = 2) -> List[str]:
        """Find symbols related to given symbol through code_graph traversal"""
        if not self.relationship_graph:
            return []
        
        # Find symbol keys that match the name
        matching_symbols = []
        for node_id in self.relationship_graph.nodes():
            if symbol_name in node_id or node_id.endswith(f":{symbol_name}"):
                matching_symbols.append(node_id)
        
        # Get neighbors for all matching symbols
        all_neighbors = set()
        for symbol_key in matching_symbols:
            neighbors = self.graph_manager.get_node_neighbors(
                self.relationship_graph, symbol_key, hops=hops
            )
            all_neighbors.update(neighbors)
        
        return list(all_neighbors)
    
    def get_symbol_document(self, symbol_key: str) -> Optional[Document]:
        """Get document for specific symbol"""
        for doc in self.code_documents:
            doc_symbol_key = f"{doc.metadata.get('source')}:{doc.metadata.get('symbol')}"
            if doc_symbol_key == symbol_key:
                return doc
        return None
    
    def get_file_documents(self, file_path: str) -> List[Document]:
        """Get all documents from specific file"""
        return [
            doc for doc in self.all_documents
            if doc.metadata.get('source') == file_path
        ]
    
    def get_documents_by_language(self, language: str) -> List[Document]:
        """Get documents by programming language"""
        return [
            doc for doc in self.code_documents
            if doc.metadata.get('language') == language
        ]
    
    def clear_cache(self):
        """Clear all caches"""
        self.vectorstore_manager.clear_cache()
        self.graph_manager.clear_cache()
        self.repository_traverser.clear_cache()
        logger.info("Cleared all index caches")
    
    def export_index_summary(self) -> Dict[str, Any]:
        """Export comprehensive index summary"""
        if not self.all_documents:
            return {"error": "No documents indexed"}
        
        # Document summary
        doc_summary = {
            'total_documents': len(self.all_documents),
            'text_documents': len(self.text_documents),
            'code_documents': len(self.code_documents)
        }
        
        # File summary
        files = set(doc.metadata.get('source') for doc in self.all_documents)
        file_summary = {
            'total_files': len(files),
            'files': list(files)
        }
        
        # Language summary
        languages = set()
        for doc in self.code_documents:
            if doc.metadata.get('language'):
                languages.add(doc.metadata['language'])
        
        # Graph summary
        graph_summary = {}
        if self.relationship_graph:
            graph_summary = {
                'nodes': self.relationship_graph.number_of_nodes(),
                'edges': self.relationship_graph.number_of_edges(),
                'files_in_graph': len(self.graph_manager.get_files_in_graph(self.relationship_graph)),
                'languages_in_graph': list(self.graph_manager.get_languages_in_graph(self.relationship_graph))
            }
        
        return {
            'documents': doc_summary,
            'files': file_summary,
            'languages': list(languages),
            'code_graph': graph_summary,
            'has_vectorstore': self.vectorstore is not None,
            'github_stats': self.repository_traverser.get_traversal_stats(),
            'filter_stats': self.filter_manager.get_filter_summary()
        }
