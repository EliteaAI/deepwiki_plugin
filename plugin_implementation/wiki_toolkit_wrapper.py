"""
Optimized API Wrapper for Wiki Toolkit with GitHub integration and parallel processing
"""

import logging
import traceback
import os
from typing import Any, Optional
from datetime import datetime

from langchain_core.callbacks import dispatch_custom_event

from .agents import OptimizedWikiGenerationAgent
# Remove BaseToolApiWrapper dependency for plugin - use simple class

from .wiki_indexer import GitHubIndexer
from .retrievers import WikiRetrieverStack
from .github_client import StandaloneGitHubClient

logger = logging.getLogger(__name__)


class OptimizedWikiToolkitWrapper:
    """Optimized API Wrapper for building comprehensive wikis from repository analysis"""
    
    def __init__(
        self,
        github_repository: str,
        github_access_token: Optional[str] = None,
        github_base_branch: str = "main",
        active_branch: Optional[str] = None,
        github_base_url: str = "https://api.github.com",
        max_files: Optional[int] = None,
        max_depth: Optional[int] = None,
        rate_limit_delay: float = 0.1,
        max_file_size: int = 1000000,
        parallel_processing: bool = True,
        enable_semantic_chunking: bool = True,
        cache_dir: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        force_rebuild_index: bool = False,
        llm: Any = None,
        alita: Any = None,
        artifact_callback: Any = None,  # Plugin artifact callback
        **kwargs
    ):
        """Initialize the optimized wrapper with simple constructor"""
        
        # Store configuration
        self.github_repository = StandaloneGitHubClient.clean_repository_name(github_repository) if github_repository else None
        self.github_access_token = github_access_token or os.getenv("GITHUB_ACCESS_TOKEN")
        self.github_base_branch = github_base_branch
        self.active_branch = active_branch or github_base_branch
        self.github_base_url = github_base_url
        self.max_files = max_files
        self.max_depth = max_depth
        self.rate_limit_delay = rate_limit_delay
        self.max_file_size = max_file_size
        self.parallel_processing = parallel_processing
        self.enable_semantic_chunking = enable_semantic_chunking
        self.cache_dir = cache_dir or f"/tmp/wiki_builder/cache"
        self.model_cache_dir = model_cache_dir or f"/tmp/wiki_builder/huggingface_cache"
        self.force_rebuild_index = force_rebuild_index
        
        # Core components
        self.llm = llm
        self.alita = alita
        self.artifact_callback = artifact_callback  # Plugin artifact callback
        self.indexer = None
        self.retriever_stack = None
        self.wiki_agent = None
        self.research_agent = None
        
        # Output configuration
        self.wiki_output_path = "wiki_output"
        self.bucket_name = "wiki_artifacts"
        
        # Initialize GitHub client
        if self.github_repository:
            self.github_client = StandaloneGitHubClient(
                repository=self.github_repository,
                access_token=self.github_access_token,
                branch=self.active_branch,
                base_url=self.github_base_url
            )
        else:
            self.github_client = None
            logger.warning("No GitHub repository specified")
        
        logger.info(f"Initialized WikiToolkitWrapper for repository: {self.github_repository}")

    def _save_artifact(self, filename: str, content: Any) -> bool:
        """Save artifact using either AlitaClient or plugin artifact callback"""
        try:
            if self.artifact_callback:
                # Plugin mode - use artifact callback
                if isinstance(content, str):
                    content = content.encode('utf-8')
                self.artifact_callback.create(filename, content)
                return True
            elif self.alita:
                # Toolkit mode - use AlitaClient
                if isinstance(content, str):
                    content = content.encode('utf-8')
                self.alita.create_artifact(self.bucket_name, filename, content)
                return True
            else:
                logger.warning("No artifact storage method available")
                return False
        except Exception as e:
            logger.error(f"Failed to save artifact {filename}: {e}")
            return False

    def _initialize_components_sync(self, workspace_path: str = "."):
        """Initialize indexing components and agents (sync version)"""
        if self.indexer is None and self.github_client:
            dispatch_custom_event(
                name="thinking_step",
                data={
                    "message": "Initializing repository indexer and retrieval components...",
                    "tool_name": "_initialize_components",
                    "toolkit": "wiki_toolkit"
                }
            )
            
            # Initialize GitHub-native indexer
            self.indexer = GitHubIndexer(
                github_client=self.github_client,
                cache_dir=self.cache_dir,
                model_cache_dir=self.model_cache_dir,
                api_filters=None,
                rate_limit_delay=self.rate_limit_delay
            )
            
            # Build repository index using GitHub API
            index_stats = self.indexer.index_repository(
                repository_name=self.github_repository,
                branch=self.active_branch,
                force_rebuild=self.force_rebuild_index,
                max_files=self.max_files,
                max_depth=self.max_depth
            )
            
            dispatch_custom_event(
                name="indexing_complete",
                data={
                    "message": f"Indexed {index_stats.get('documents', {}).get('total', 0)} documents",
                    "stats": index_stats,
                    "tool_name": "github_indexer",
                    "toolkit": "wiki_toolkit"
                }
            )
            
            # Initialize retriever stack
            self.retriever_stack = WikiRetrieverStack(
                vectorstore_manager=self.indexer.vectorstore_manager,
                relationship_graph=self.indexer.relationship_graph,
                llm_client=self.llm
            )


    def generate_wiki(
        self,
        query: str,
        wiki_title: Optional[str] = None,
        include_research: bool = True,
        include_diagrams: bool = True,
        output_format: str = "json"
    ) -> str:
        """Generate comprehensive wiki from repository analysis with optimized processing"""
        try:
            if not self.github_repository:
                raise ValueError("GitHub repository not specified")
                
            repo_path = self.github_repository

            if not wiki_title:
                wiki_title = f"{repo_path.split('/')[-1]} Wiki"

            dispatch_custom_event(
                name="thinking_step",
                data={
                    "message": f"Starting wiki generation for {repo_path} with query: {query}",
                    "tool_name": "generate_wiki",
                    "toolkit": "wiki_toolkit"
                }
            )

            # Initialize components
            self._initialize_components_sync()
            
            # Execute wiki generation using LangGraph agent
            dispatch_custom_event(
                name="thinking_step",
                data={
                    "message": "Executing wiki generation agent...",
                    "tool_name": "generate_wiki",
                    "toolkit": "wiki_toolkit"
                }
            )
            
            # Create wiki configuration
            
            # Create agent with configuration parameters directly
            from plugin_implementation.state.wiki_state import WikiStyle
            from plugin_implementation.state.wiki_state import TargetAudience
            wiki_agent = OptimizedWikiGenerationAgent(
                indexer=self.indexer,
                retriever_stack=self.retriever_stack,
                llm=self.llm,
                repository_url=self.github_repository,
                branch=self.active_branch,
                alita_client=self.alita if self.alita else None,  # Use AlitaClient only in toolkit mode
                bucket_name=self.bucket_name,
                wiki_style=WikiStyle.COMPREHENSIVE,
                target_audience=TargetAudience.MIXED,
                require_diagrams=include_diagrams,
                enable_progress_tracking=False,
                enable_quality_enhancement=True
            )
            
            # Generate wiki with user message using standard LangGraph pattern
            wiki_result = wiki_agent.generate_wiki(query)
            
            if wiki_result and wiki_result.get('success'):
                generation_summary = wiki_result.get('generation_summary', {})
                total_pages = generation_summary.get('total_pages', len(wiki_result.get('generated_pages', {})))
                total_sections = generation_summary.get('total_sections', 0)
                total_diagrams = generation_summary.get('total_diagrams', 0)
                
                result_message = f"""
**🚀 Wiki Generation Complete!**

Repository: {repo_path}
Title: {wiki_title}
Query: {query}

**📊 Processing Results:**
- Include Research: {include_research}
- Include Diagrams: {include_diagrams}
- Output Format: {output_format}
- Status: ✅ Complete

**🎯 Generated Content:**
- Wiki Pages: {total_pages}
- Sections: {total_sections}
- Diagrams: {total_diagrams}

**📈 Processing Statistics:**
- Execution Time: {wiki_result.get('execution_time', 0):.2f}s
- Messages: {len(wiki_result.get('messages', []))}

The comprehensive wiki has been successfully generated.
"""
                return result_message
            else:
                error_msg = wiki_result.get('error', 'Unknown error') if wiki_result else 'No result returned'
                return f"**❌ Wiki Generation Failed**\n\nRepository: {repo_path}\nError: {error_msg}"

        except Exception as e:
            logger.error(f"Failed to generate wiki: {e}")
            logger.error(traceback.format_exc())
            return f"Error generating wiki: {str(e)}"
