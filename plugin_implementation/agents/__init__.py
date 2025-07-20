"""
Wiki Toolkit Agents Package

This package provides LangGraph-based agents for wiki generation and research.
"""

from .wiki_graph_optimized import OptimizedWikiGenerationAgent

# Backward compatibility alias
WikiGenerationAgent = OptimizedWikiGenerationAgent

__all__ = [
    "OptimizedWikiGenerationAgent",
]

# Version info
__version__ = "1.0.0"
__author__ = "Wiki Toolkit Team"
__description__ = "LangGraph agents for wiki generation and research"
