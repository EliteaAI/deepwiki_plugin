"""
Wiki Toolkit Wrapper (filesystem indexing only).
Supports multi-provider repositories: GitHub, GitLab, Bitbucket, Azure DevOps.
"""

import logging
import os
import traceback
from typing import Any, Dict, Optional, Union
try:  # Optional thinking emitter (safe no-op if task context not present)
    from tools import this  # type: ignore
except Exception:  # pragma: no cover
    this = None  # type: ignore

from .agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
from .github_client import StandaloneGitHubClient
from .retrievers import WikiRetrieverStack
from .filesystem_indexer import FilesystemRepositoryIndexer
from .repo_providers import GitCloneConfig
from .unified_retriever import UNIFIED_RETRIEVER_ENABLED

logger = logging.getLogger(__name__)


class HybridWikiToolkitWrapper:
    """
    Wrapper for filesystem-based indexing with multi-provider support.
    
    Supports: GitHub, GitLab, Bitbucket, Azure DevOps.
    """
    
    def __init__(
        self,
        # New multi-provider configuration
        repo_config: Optional[Dict[str, Any]] = None,
        clone_config: Optional[GitCloneConfig] = None,
        # Legacy GitHub-specific fields (for backward compatibility)
        github_repository: Optional[str] = None,
        github_access_token: Optional[str] = None,
        github_username: Optional[str] = None,
        github_base_branch: str = "main",
        active_branch: Optional[str] = None,
        github_base_url: str = "https://api.github.com",
        # Common configuration
        max_files: Optional[int] = None,
        max_depth: Optional[int] = None,
        rate_limit_delay: float = 0.1,
        cache_dir: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        force_rebuild_index: bool = False,
        llm: Any = None,
        embeddings: Any = None,
        indexing_method: str = "filesystem",  # deprecated (filesystem only)
        cleanup_repos_on_exit: bool = True,
        **kwargs
    ):
        """Initialize the wrapper with configurable multi-provider support"""
        
        # Store new multi-provider configuration
        self.repo_config = repo_config or {}
        self.clone_config = clone_config
        
        # Extract values from repo_config or fall back to legacy parameters
        if self.clone_config:
            # Use clone_config if available (preferred)
            self.repository = self.clone_config.repo_identifier
            self.branch = self.clone_config.branch
            self.provider_type = self.clone_config.provider.value
        elif self.repo_config:
            # Use repo_config dict
            self.repository = self.repo_config.get('repository') or github_repository
            self.branch = self.repo_config.get('branch') or github_base_branch
            self.provider_type = self.repo_config.get('provider_type', 'github')
        else:
            # Legacy GitHub-only mode
            self.repository = github_repository
            self.branch = github_base_branch
            self.provider_type = 'github'
        
        # Clean repository name (for GitHub compatibility)
        if self.provider_type == 'github' and self.repository:
            self.repository = StandaloneGitHubClient.clean_repository_name(self.repository)
        
        # Legacy GitHub fields (maintained for backward compatibility)
        self.github_repository = self.repository  # Alias for existing code
        self.github_access_token = github_access_token or os.getenv("GITHUB_ACCESS_TOKEN")
        self.github_base_branch = self.branch
        self.github_username = github_username or os.getenv("GITHUB_USERNAME")
        self.active_branch = active_branch or self.branch
        self.github_base_url = github_base_url
        
        # Common configuration
        self.max_files = max_files
        self.max_depth = max_depth
        self.rate_limit_delay = rate_limit_delay
        self.cache_dir = cache_dir or f"/tmp/wiki_builder/cache"
        self.model_cache_dir = model_cache_dir or f"/tmp/wiki_builder/huggingface_cache"
        self.force_rebuild_index = force_rebuild_index
        self.indexing_method = "filesystem"  # Only filesystem is supported
        self.cleanup_repos_on_exit = cleanup_repos_on_exit

        # Core components
        self.llm = llm
        self.indexer = None  # type: ignore[assignment]
        self.retriever_stack = None
        self.wiki_agent = None
        self.research_agent = None
        self.embeddings = embeddings
        
        # Output configuration
        self.wiki_output_path = "wiki_output"
        self.bucket_name = "wiki_artifacts"
        
        if not self.repository:
            logger.warning("No repository specified")
            if this and getattr(this, 'module', None):
                this.module.invocation_thinking(
                    "I am on phase initialization\nHybrid wiki toolkit initialized\n"
                    "Reasoning: Core components prepared (indexer placeholder, retriever stack deferred).\n"
                    "Next: Build or load repository index before content generation."
                )

        logger.info(
            f"Initialized HybridWikiToolkitWrapper for repository: {self.repository} "
            f"(provider: {self.provider_type}) using {self.indexing_method} indexing method"
        )

    def _initialize_components_sync(self, workspace_path: str = "."):
        """Initialize indexing components and agents (sync version)"""
        if self.indexer is None:
            self._initialize_filesystem_indexer()

            # Phase 5+: when unified retriever is active AND legacy indexes
            # were skipped, go straight to UnifiedRetriever — there is no
            # FAISS vectorstore to wrap in WikiRetrieverStack.
            if UNIFIED_RETRIEVER_ENABLED:
                self._try_upgrade_to_unified_retriever()

            # Fallback: build legacy WikiRetrieverStack when unified
            # retriever is disabled or upgrade failed.
            if self.retriever_stack is None:
                self.retriever_stack = WikiRetrieverStack(
                    vectorstore_manager=self.indexer.vectorstore_manager,
                    relationship_graph=self.indexer.relationship_graph,
                    llm_client=self.llm
                )

    def _try_upgrade_to_unified_retriever(self):
        """Replace WikiRetrieverStack with UnifiedRetriever if .wiki.db exists.

        Uses the same ``md5(repo_path + commit_hash)`` cache key as FAISS /
        BM25 / docstore to locate ``{cache_key}.wiki.db``.
        """
        try:
            from .unified_retriever import UnifiedRetriever
            from .unified_db import UnifiedWikiDB
            import glob

            cache_dir = str(
                getattr(
                    getattr(self.indexer, 'graph_manager', None), 'cache_dir', None
                ) or "/tmp/wiki_builder/cached_graphs"
            )

            db_path = None

            # 1. Direct hash (same key as FAISS / BM25)
            repo_path = getattr(self.indexer, 'current_repo_path', None)
            commit = getattr(self.indexer, '_last_commit_hash', None)
            if repo_path:
                vs_mgr = getattr(self.indexer, 'vectorstore_manager', None)
                if vs_mgr:
                    cache_key = vs_mgr._generate_repo_hash(repo_path, [], commit)
                    candidate = os.path.join(cache_dir, f"{cache_key}.wiki.db")
                    if os.path.exists(candidate):
                        db_path = candidate

            # 2. cache_index.json lookup
            if not db_path:
                try:
                    from .repo_resolution import load_cache_index
                    idx = load_cache_index(cache_dir)
                    udb_map = idx.get("unified_db", {})
                    if isinstance(udb_map, dict):
                        for _rid, ck in udb_map.items():
                            candidate = os.path.join(cache_dir, f"{ck}.wiki.db")
                            if os.path.exists(candidate):
                                db_path = candidate
                                break
                except Exception:
                    pass

            # 3. Glob fallback
            if not db_path:
                all_dbs = glob.glob(os.path.join(cache_dir, "*.wiki.db"))
                if all_dbs:
                    all_dbs.sort(key=os.path.getmtime, reverse=True)
                    db_path = all_dbs[0]

            if not db_path:
                logger.info("[UNIFIED_RETRIEVER] No .wiki.db found, keeping legacy retriever")
                return

            db = UnifiedWikiDB(db_path, readonly=True)

            # Build embedding_fn from the embeddings object
            embeddings_obj = self.embeddings or getattr(
                getattr(self.indexer, 'vectorstore_manager', None), 'embeddings', None
            )
            embedding_fn = None
            if embeddings_obj and hasattr(embeddings_obj, 'embed_query'):
                embedding_fn = embeddings_obj.embed_query

            self.retriever_stack = UnifiedRetriever(
                db=db,
                embedding_fn=embedding_fn,
                embeddings=embeddings_obj,
            )
            logger.info("[UNIFIED_RETRIEVER] Upgraded to UnifiedRetriever from %s", db_path)
        except Exception as exc:
            logger.warning("[UNIFIED_RETRIEVER] Upgrade failed, keeping legacy: %s", exc)

    def _initialize_filesystem_indexer(self):
        """Initialize filesystem-based indexer with multi-provider support"""
        logger.info(f"Initializing filesystem-based repository indexer (provider: {self.provider_type})...")
        
        self.indexer = FilesystemRepositoryIndexer(
            cache_dir=self.cache_dir,
            model_cache_dir=self.model_cache_dir,
            api_filters=None,
            max_workers=8,
            cleanup_repos_on_exit=self.cleanup_repos_on_exit,
            # Pass clone_config for multi-provider support
            clone_config=self.clone_config,
            # Legacy GitHub credentials (fallback if clone_config not available)
            github_access_token=self.github_access_token,
            github_username=self.github_username,
            embeddings=self.embeddings
        )
        
        # Build repository index using filesystem approach
        index_stats = self.indexer.index_repository(
            repository_url=self.repository,
            branch=self.branch,
            force_rebuild=self.force_rebuild_index,
            force_reclone=False,  # Only reclone if explicitly requested
            max_files=self.max_files
        )
        
        # Update active_branch from actual cloned branch (handles master vs main, etc.)
        actual_branch = index_stats.get('actual_branch')
        if actual_branch:
            logger.info(f"Using actual branch from clone: {actual_branch} (requested: {self.branch})")
            self.active_branch = actual_branch
            self.branch = actual_branch
            self.github_base_branch = actual_branch
        
        logger.info(f"Filesystem indexing complete: {index_stats.get('documents', {}).get('total', 0)} documents")
    
    def generate_wiki(
        self,
        query: str,
        wiki_title: Optional[str] = None,
        include_research: bool = True,
        include_diagrams: bool = True,
        output_format: str = "json"
    ):
        """Generate comprehensive wiki from repository analysis with hybrid processing"""
        try:
            if not self.github_repository:
                raise ValueError("GitHub repository not specified")
                
            repo_path = self.github_repository

            if not wiki_title:
                wiki_title = f"{repo_path.split('/')[-1]} Wiki"

            # Initialize components
            if this and getattr(this, 'module', None):
                this.module.invocation_thinking(f"I am on phase initialization\nStarting wiki generation for {repo_path} (query: {query})\nReasoning: Kick off end-to-end pipeline (index -> analysis -> structure -> pages).\nNext: Initialize indexer & retrieval stack.")
            self._initialize_components_sync()
            if this and getattr(this, 'module', None):
                this.module.invocation_thinking("I am on phase initialization\nIndexing complete – preparing generation agent\nReasoning: Repository artifacts available for higher-level semantic synthesis.\nNext: Dispatch structure + page generation workflow.")

            # Import optimized agent components
            from .state.wiki_state import WikiStyle, TargetAudience

            # Initialize wiki generation agent with optimized configuration
            wiki_agent = OptimizedWikiGenerationAgent(
                indexer=self.indexer,
                retriever_stack=self.retriever_stack,
                llm=self.llm,
                repository_url=self.github_repository,
                branch=self.active_branch,
                bucket_name=self.bucket_name,
                wiki_style=WikiStyle.COMPREHENSIVE,
                target_audience=TargetAudience.MIXED,
                require_diagrams=include_diagrams,
                enable_progress_tracking=False,
                enable_quality_enhancement=True,
                graph_text_index=getattr(
                    getattr(self.indexer, 'graph_manager', None),
                    'fts_index', None,
                ),
            )
            
            # Generate wiki with user message using standard LangGraph pattern
            if this and getattr(this, 'module', None):
                this.module.invocation_thinking("I am on phase page_generation\nDispatching page generation tasks\nReasoning: Parallel drafting accelerates throughput while leveraging shared repository context.\nNext: For each page, retrieve focused context then invoke LLM.")
            wiki_result = wiki_agent.generate_wiki(query)
            if this and getattr(this, 'module', None):
                this.module.invocation_thinking("I am on phase completion\nWiki generation workflow finished\nReasoning: All requested phases executed successfully.\nNext: Return artifacts & summaries to caller.")

            if wiki_result and wiki_result.get('success'):
                generation_summary = wiki_result.get('generation_summary') or {}
                generated_pages = wiki_result.get('generated_pages') or {}
                total_pages = generation_summary.get('total_pages', len(generated_pages))
                total_sections = generation_summary.get('total_sections', 0)

                workflow_errors = wiki_result.get('errors') if isinstance(wiki_result, dict) else None
                if not isinstance(workflow_errors, list):
                    workflow_errors = []
                failed_pages = wiki_result.get('failed_pages') if isinstance(wiki_result, dict) else None
                if not isinstance(failed_pages, list):
                    failed_pages = []

                result_message = f"""
            **🚀 Wiki Generation Complete!**

            Repository: {repo_path}
            Title: {wiki_title}
            Query: {query}

            **📊 Processing Results:**
            - Status: ✅ Complete

            **🎯 Generated Content:**
            - Wiki Pages: {total_pages}
            - Sections: {total_sections}

            **📈 Processing Statistics:**
            - Execution Time: {wiki_result.get('execution_time', 0):.2f}s

            The comprehensive wiki has been successfully generated.
            """

                if workflow_errors or failed_pages:
                    # Keep this short; detailed errors are returned in structured fields below.
                    result_message += "\n\n⚠️ Note: Some steps reported warnings/errors during generation."
                
                # Get commit hash from indexer for cache key generation
                commit_hash = None
                if self.indexer and hasattr(self.indexer, '_last_commit_hash'):
                    commit_hash = self.indexer._last_commit_hash
                
                return {
                    'success': True,
                    'result': result_message,
                    'artifacts': wiki_result.get('artifacts', []),
                    'execution_time': wiki_result.get('execution_time', 0),
                    'errors': workflow_errors,
                    'failed_pages': failed_pages,
                    # Pass through repository_context for Ask tool
                    'repository_context': wiki_result.get('repository_context'),
                    # Pass commit hash for cache key consistency
                    'commit_hash': commit_hash,
                    # Pass actual branch from clone (handles master vs main, etc.)
                    'branch': self.active_branch,
                    'provider_type': self.provider_type,
                }
            else:
                return {
                    'success': False,
                    'error': "Failed to generate wiki. Wiki generation agent failed to generate wiki."
                }

        except Exception as e:
            logger.error(f"Failed to generate wiki: {e}")
            logger.error(traceback.format_exc())
            raise e

    def get_index_summary(self) -> dict:
        """Get summary of current index"""
        if not self.indexer:
            return {"error": "No indexer initialized"}
        
        summary = self.indexer.export_index_summary()
        summary["indexing_method"] = self.indexing_method
        return summary

    def clear_cache(self):
        """Clear all caches"""
        if self.indexer:
            self.indexer.clear_cache()

    def cleanup_repositories(self):
        """Clean up repository clones (filesystem indexer only)"""
        if hasattr(self.indexer, 'cleanup_repositories'):
            self.indexer.cleanup_repositories()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        if self.cleanup_repos_on_exit:
            self.cleanup_repositories()


# Convenience functions for backward compatibility
def create_wiki_toolkit_wrapper(*args, **kwargs):
    """Create wiki toolkit wrapper with default filesystem indexing"""
    kwargs.setdefault('indexing_method', 'filesystem')
    return HybridWikiToolkitWrapper(*args, **kwargs)

def create_github_api_wrapper(*args, **kwargs):
    """Create wiki toolkit wrapper with GitHub API indexing (legacy)"""
    kwargs['indexing_method'] = 'github_api'
    return HybridWikiToolkitWrapper(*args, **kwargs)

def create_filesystem_wrapper(*args, **kwargs):
    """Create wiki toolkit wrapper with filesystem indexing"""
    kwargs['indexing_method'] = 'filesystem'
    return HybridWikiToolkitWrapper(*args, **kwargs)
