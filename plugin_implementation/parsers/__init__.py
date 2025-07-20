"""
Language-specific AST parsers for Wiki Toolkit Graph Enhancement

This package contains enhanced AST parsers for multiple programming languages,
providing comprehensive relationship detection for code_graph-based code analysis.

Each parser focuses on language-specific features while maintaining a consistent
interface for code_graph construction and relationship tracking.

Available parsers:
- Enhanced Python Parser: Comprehensive Python AST analysis
- Java Parser: Full Java language support with OOP features
- Base Parser: Abstract interface for all language parsers

Parser initialization and registration system provides automatic discovery
and registration of all available language parsers.
"""

import logging
from typing import Optional

# Import new modular architecture components
from .base_parser import BaseParser, parser_registry, ParserRegistry

logger = logging.getLogger(__name__)


def initialize_parsers() -> None:
    """
    Initialize and register all available language parsers
    """
    try:
        # Import and register Python parser
        from .python_parser import PythonParser
        python_parser = PythonParser()
        parser_registry.register_parser(python_parser)
        logger.info("Registered enhanced Python parser")
    except ImportError as e:
        logger.warning(f"Failed to import Python parser: {e}")
    
    try:
        # Import and register Java parser (tree-sitter based)
        from .java_parser_treesitter import JavaTreeSitterParser
        java_parser = JavaTreeSitterParser()
        parser_registry.register_parser(java_parser)
        logger.info("Registered tree-sitter Java parser")
    except ImportError as e:
        logger.warning(f"Failed to import tree-sitter Java parser: {e}")
        raise e
    except Exception as e:
        logger.warning(f"Failed to initialize tree-sitter Java parser: {e}")
        raise e

    supported_languages = parser_registry.get_supported_languages()
    logger.info(f"Initialized parsers for {len(supported_languages)} languages: {supported_languages}")


def get_parser_for_language(language: str) -> Optional[BaseParser]:
    """
    Get parser for a specific language
    
    Args:
        language: Programming language name
        
    Returns:
        Parser instance or None if not supported
    """
    return parser_registry.get_parser(language)


def get_parser_for_file(file_path: str) -> Optional[BaseParser]:
    """
    Get appropriate parser for a file based on its extension
    
    Args:
        file_path: Path to the source file
        
    Returns:
        Parser instance or None if no suitable parser found
    """
    return parser_registry.get_parser_for_file(file_path)


def list_supported_languages() -> list[str]:
    """
    Get list of all supported programming languages
    
    Returns:
        List of language names
    """
    return parser_registry.get_supported_languages()


def list_supported_extensions() -> set[str]:
    """
    Get set of all supported file extensions
    
    Returns:
        Set of file extensions
    """
    return parser_registry.get_supported_extensions()


# Auto-initialize parsers when module is imported
try:
    initialize_parsers()
except Exception as e:
    logger.error(f"Failed to initialize parsers: {e}")


# Export key components
__all__ = [
    # New modular architecture
    'ParserRegistry',
    'parser_registry',
    'initialize_parsers',
    'get_parser_for_language',
    'get_parser_for_file',
    'list_supported_languages',
    'list_supported_extensions',
    'BaseParser'
]
