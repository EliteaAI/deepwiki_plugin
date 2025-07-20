"""
GitHub Repository Traverser

Handles repository traversal using the GitHub API instead of local file system.
Provides filtered file discovery and content retrieval with caching.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Set, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class GitHubRepositoryTraverser:
    """Simplified traverser working directly with file paths and content"""
    
    def __init__(self, 
                 github_client,
                 filter_manager,
                 cache_enabled: bool = True,
                 rate_limit_delay: float = 0.1):
        """
        Initialize repository traverser
        
        Args:
            github_client: GitHubClient instance
            filter_manager: FilterManager instance for filtering
            cache_enabled: Whether to cache file content
            rate_limit_delay: Delay between API calls to avoid rate limits
        """
        self.github_client = github_client
        self.filter_manager = filter_manager
        self.cache_enabled = cache_enabled
        self.rate_limit_delay = rate_limit_delay
        
        # Simplified caches - just content cache
        self._content_cache: Dict[str, str] = {}
        self._file_paths_cache: Optional[List[str]] = None
        
        # Statistics
        self.stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'files_discovered': 0,
            'files_filtered': 0,
            'total_size_bytes': 0
        }
    
    def discover_files(self, 
                      base_path: str = "",
                      max_files: Optional[int] = None,
                      max_depth: Optional[int] = None) -> List[str]:
        """
        Discover all processable files in the repository
        
        Args:
            base_path: Starting directory path (empty for root)
            max_files: Maximum number of files to discover
            max_depth: Maximum directory depth to traverse (ignored since _get_files handles this)
            
        Returns:
            List of file paths as strings
        """
        logger.info(f"Starting file discovery from {base_path or 'root'}")
        
        try:
            # Use cached file paths if available
            if self._file_paths_cache is not None:
                all_file_paths = self._file_paths_cache
            else:
                # Get all files from repository
                all_file_paths = self._get_all_files(base_path)
                if self.cache_enabled:
                    self._file_paths_cache = all_file_paths
            
            # Filter files
            discovered_files = []
            for file_path in all_file_paths:
                if self._should_process_file(file_path):
                    discovered_files.append(file_path)
                    self.stats['files_discovered'] += 1
                else:
                    self.stats['files_filtered'] += 1
                    
                # Check file limit
                if max_files is not None and len(discovered_files) >= max_files:
                    break
            
            logger.info(f"File discovery complete: {len(discovered_files)} files found")
            self._log_discovery_stats()
            
            return discovered_files
            
        except Exception as e:
            logger.error(f"Error during file discovery: {e}")
            return []
    
    def _get_all_files(self, base_path: str = "") -> List[str]:
        """Get all file paths from repository"""
        # Rate limiting
        time.sleep(self.rate_limit_delay)
        
        try:
            # Use GitHub client to get all files
            file_paths = self.github_client._get_files(
                base_path, 
                self.github_client.active_branch, 
                self.github_client.github_repository
            )
            
            if isinstance(file_paths, str) and file_paths.startswith("Error"):
                logger.error(f"GitHub API error: {file_paths}")
                return []
            
            self.stats['api_calls'] += 1
            return file_paths
            
        except Exception as e:
            logger.error(f"Failed to get files from repository: {e}")
            return []
    
    def _should_process_file(self, file_path: str) -> bool:
        """Check if a file should be processed using filter manager"""
        return self.filter_manager.should_process_file(file_path)
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Get content of a file with caching
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string, or None if failed
        """
        # Check content cache
        if self.cache_enabled and file_path in self._content_cache:
            self.stats['cache_hits'] += 1
            return self._content_cache[file_path]
        
        # Rate limiting
        time.sleep(self.rate_limit_delay)
        
        try:
            # Use GitHub client to read file
            content = self.github_client._read_file(
                file_path, 
                self.github_client.active_branch,
                self.github_client.github_repository
            )
            
            self.stats['api_calls'] += 1
            
            # Check if it's an error message
            if content.startswith("File not found") or content.startswith("Error"):
                logger.warning(f"Failed to read file {file_path}: {content}")
                return None
            
            # Cache content
            if self.cache_enabled:
                self._content_cache[file_path] = content
            
            # Update size stats
            self.stats['total_size_bytes'] += len(content.encode('utf-8'))
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None
    
    def get_files_by_type(self, file_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        Separate files into text and code files
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Tuple of (text_files, code_files)
        """
        text_files = []
        code_files = []
        
        for file_path in file_paths:
            if self.filter_manager.is_text_file(file_path):
                text_files.append(file_path)
            elif self.filter_manager.is_code_file(file_path):
                code_files.append(file_path)
            # Files that are neither text nor code are ignored
        
        logger.info(f"Categorized files: {len(text_files)} text, {len(code_files)} code")
        return text_files, code_files
    
    def get_file_metadata(self, file_path: str, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a file
        
        Args:
            file_path: Path to the file
            content: Optional file content (to avoid re-reading)
            
        Returns:
            Dictionary with file metadata
        """
        metadata = {
            'source': file_path,
            'name': Path(file_path).name,
            'type': 'file'
        }
        
        # Add size if content is provided
        if content is not None:
            metadata['size'] = len(content.encode('utf-8'))
        
        # Add language information for code files
        language = self.filter_manager.get_file_language(file_path)
        if language:
            metadata['language'] = language
            metadata['file_type'] = 'code'
        elif self.filter_manager.is_text_file(file_path):
            metadata['file_type'] = 'text'
            # Determine text format
            ext = Path(file_path).suffix.lower()
            if ext in {'.md', '.markdown'}:
                metadata['format'] = 'markdown'
            elif ext in {'.rst'}:
                metadata['format'] = 'restructuredtext'
            elif ext in {'.txt'}:
                metadata['format'] = 'plain_text'
            else:
                metadata['format'] = 'text'
        
        return metadata
    
    def clear_cache(self) -> None:
        """Clear all caches"""
        self._content_cache.clear()
        self._file_paths_cache = None
        logger.info("Cleared all caches")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'content_cache_size': len(self._content_cache),
            'file_paths_cached': self._file_paths_cache is not None,
            'cache_enabled': self.cache_enabled
        }
    
    def get_traversal_stats(self) -> Dict[str, Any]:
        """Get traversal statistics"""
        return {
            **self.stats,
            'cache_stats': self.get_cache_stats()
        }
    
    def _log_discovery_stats(self) -> None:
        """Log discovery statistics"""
        logger.info(f"Discovery stats: "
                   f"API calls: {self.stats['api_calls']}, "
                   f"Cache hits: {self.stats['cache_hits']}, "
                   f"Files discovered: {self.stats['files_discovered']}, "
                   f"Files filtered: {self.stats['files_filtered']}, "
                   f"Total size: {self.stats['total_size_bytes'] / (1024*1024):.2f}MB")
