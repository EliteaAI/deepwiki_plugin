#!/usr/bin/env python3
# coding=utf-8

"""
Subprocess worker for Deep Research tool.

This module is executed in a separate process to isolate the Deep Research operation
from the main web worker process, following the same pattern as wiki/ask generation.

Unlike wiki generation, this does NOT use worker slot limits since Deep Research
needs to run independently.

Contract:
- Reads JSON from --input
- Writes JSON to --output
- Emits progress logs to stdout (parsed by parent for thinking steps)
- Emits special [THINKING_STEP] markers for structured thinking events
- Emits [TODO_UPDATE] markers for todo list updates

Input JSON schema:
{
    "base_path": "/path/to/cache",
    "question": "Research question",
    "llm_settings": {...},
    "embedding_model": {...},
    "github_repository": "owner/repo",
    "github_branch": "main",
    "research_type": "general" | "architecture" | "implementation" | "security" | "integration",
    "max_iterations": 15
}

Output JSON schema:
{
    "success": bool,
    "report": str,
    "todos": [...],
    "findings": [...],
    "thinking_steps": [...],
    "error": str (when success=false)
}
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from typing import Any, Dict, List, Optional


def _configure_logging() -> None:
    """Configure logging for the subprocess."""
    level_name = os.getenv("DEEPWIKI_WORKER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s -- %(message)s")
    )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Keep noisy libraries at WARNING
    if level > logging.DEBUG:
        for noisy in ["httpx", "urllib3", "openai", "langchain", "langgraph", "sentence_transformers"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _print(msg: str) -> None:
    """Print with flush for real-time streaming."""
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _emit_thinking_step(step_type: str, title: str, content: str, metadata: Optional[Dict] = None) -> None:
    """Emit a structured thinking step for the UI."""
    step = {
        "type": step_type,
        "title": title,
        "content": content[:500] if content else "",
        "metadata": metadata or {}
    }
    _print(f"[THINKING_STEP] {json.dumps(step)}")


def _emit_todo_update(todos: List[Dict]) -> None:
    """Emit a todo list update for the UI."""
    _print(f"[TODO_UPDATE] {json.dumps(todos)}")


def _build_llm_and_embeddings(llm_settings: Dict[str, Any], embedding_model: Any):
    """Build LLM and embeddings from settings."""
    from langchain_openai import ChatOpenAI
    from langchain_openai.embeddings import OpenAIEmbeddings

    # Detect provider
    provider = llm_settings.get("provider", "openai")
    
    # Normalize field names
    api_base = llm_settings.get("api_base") or llm_settings.get("openai_api_base")
    api_key = llm_settings.get("api_key") or llm_settings.get("openai_api_key")
    model_name = llm_settings.get("model_name", "gpt-4o-mini")
    organization = llm_settings.get("organization")
    max_retries = llm_settings.get("max_retries", 2)
    max_tokens = llm_settings.get("max_tokens", 8192)  # Higher for research
    default_headers = llm_settings.get("default_headers", {})

    if not api_base:
        raise ValueError("llm_settings.api_base is required")
    if not api_key:
        raise ValueError("llm_settings.api_key is required")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        anthropic_base_url = api_base.rstrip("/")
        if anthropic_base_url.endswith("/v1"):
            anthropic_base_url = anthropic_base_url[:-3]

        if not default_headers:
            default_headers = {
                "openai-organization": str(organization) if organization else "",
                "Authorization": f"Bearer {api_key}"
            }

        llm = ChatAnthropic(
            model=model_name,
            api_key=api_key,
            base_url=anthropic_base_url,
            max_tokens=max_tokens,
            temperature=0.1,
            max_retries=max_retries,
            streaming=False,
            default_headers=default_headers,
        )
        embeddings_base_url = api_base if api_base.endswith("/v1") else api_base.rstrip("/") + "/v1"
    else:
        # OpenAI
        temperature = 1.0 if str(model_name).startswith("o") else 0.1
        
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=api_base,
            organization=organization,
            max_retries=max_retries,
            streaming=False,
            max_tokens=max_tokens,
        )
        embeddings_base_url = api_base

    # Build embeddings
    embedding_model_name = embedding_model if isinstance(embedding_model, str) else "text-embedding-3-large"
    if isinstance(embedding_model, dict):
        embedding_model_name = embedding_model.get("model_name", embedding_model_name)

    embeddings = OpenAIEmbeddings(
        model=embedding_model_name,
        openai_api_key=api_key,
        openai_api_base=embeddings_base_url,
        openai_organization=organization,
    )

    return llm, embeddings


async def run_deep_research_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the Deep Research tool with the given payload.
    
    Args:
        payload: Input configuration
        
    Returns:
        Result dictionary
    """
    from .vectorstore import VectorStoreManager
    from .retrievers import WikiRetrieverStack
    from .repository_analysis_store import RepositoryAnalysisStore
    from .graph_manager import GraphManager
    from .deep_research import create_deep_research_engine
    
    logger = logging.getLogger(__name__)
    
    base_path = payload.get("base_path", "")
    question = payload.get("question", "")
    llm_settings = payload.get("llm_settings", {})
    embedding_model = payload.get("embedding_model")
    
    # Support new multi-provider repo_config with legacy fallback
    repo_config = payload.get("repo_config", {})
    if repo_config:
        provider_type = repo_config.get("provider_type", "github")
        repository = repo_config.get("repository", "")
        branch = repo_config.get("branch", "main")
    else:
        # Legacy GitHub-only mode
        provider_type = "github"
        repository = payload.get("github_repository", "")
        branch = payload.get("github_branch", "main")
    
    research_type = payload.get("research_type", "general")
    max_iterations = payload.get("max_iterations", 15)
    repo_identifier_override = payload.get("repo_identifier_override")
    analysis_key_override = payload.get("analysis_key_override")
    
    if not question:
        return {"success": False, "error": "No research question provided"}
    
    if not repository:
        return {"success": False, "error": "No repository specified"}
    
    # Caller inputs are repo:branch; we will resolve to commit-scoped id via cache refs.
    repo_identifier = f"{repository}:{branch}"
    
    _print(f"Starting deep research for repository: {repo_identifier} (provider: {provider_type})")
    _emit_thinking_step("start", "Research Started", f"Question: {question[:200]}...")
    
    # Set up cache directories
    cache_dir = os.path.join(base_path, "cache") if base_path else os.path.expanduser("~/.elitea/wiki_indexes")
    model_cache_dir = os.path.join(base_path, "huggingface_cache") if base_path else None
    
    # Set HuggingFace cache env vars
    if model_cache_dir:
        os.environ['TRANSFORMERS_CACHE'] = model_cache_dir
        os.environ['HF_HOME'] = model_cache_dir
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = model_cache_dir
    
    collected_events = []
    final_report = ""
    all_todos = []
    
    try:
        # In K8s Jobs mode, ensure indexes are available locally.
        # Jobs+API: downloads from platform bucket using llm_settings creds.
        # In Docker mode: ArtifactManager returns True immediately.
        try:
            from .artifact_manager import ArtifactManager, is_jobs_mode
            if is_jobs_mode():
                _print("[deep_research] K8s Jobs mode — verifying indexes are available locally...")
                from .artifacts_platform_client import (
                    create_platform_client_from_llm_settings, get_artifact_bucket,
                )
                from .registry_manager import normalize_wiki_id
                _art_client = create_platform_client_from_llm_settings(llm_settings) if llm_settings else None
                art_mgr = ArtifactManager(
                    cache_dir=cache_dir,
                    artifacts_client=_art_client,
                    bucket=get_artifact_bucket() if _art_client else "",
                )
                _wiki_id = normalize_wiki_id(repo_identifier)
                ok = art_mgr.ensure_indexes_for_wiki(wiki_id=_wiki_id)
                if ok:
                    _print("[deep_research] Index verification: OK")
                else:
                    _print("[deep_research] Warning: index verification returned False — will try loading anyway")
        except Exception as _art_err:
            _print(f"[deep_research] Warning: ArtifactManager check failed: {_art_err}")

        # Build LLM and embeddings
        _print("Initializing LLM and embeddings...")
        _emit_thinking_step("init", "Initializing", "Building LLM and embeddings...")
        llm, embeddings = _build_llm_and_embeddings(llm_settings, embedding_model)

        # Resolve canonical commit-scoped identifier and pick matching clone dir (or use override).
        repositories_dir = os.path.join(cache_dir, "repositories")
        canonical_repo_identifier = repo_identifier
        filesystem_root_dir = None
        try:
            from .repo_resolution import resolve_canonical_repo_identifier, resolve_clone_dir_for_canonical_id
            from .repo_resolution import load_cache_index

            if isinstance(repo_identifier_override, str) and repo_identifier_override.strip():
                # Validate override exists in cache before using it
                override_candidate = repo_identifier_override.strip()
                idx = load_cache_index(cache_dir)
                if idx.get(override_candidate):
                    canonical_repo_identifier = override_candidate
                    filesystem_root_dir = resolve_clone_dir_for_canonical_id(
                        canonical_repo_id=canonical_repo_identifier,
                        cache_dir=cache_dir,
                    )
                else:
                    # Override not in cache, fall back to resolution from repo_identifier
                    _print(f"Override '{override_candidate}' not found in cache, resolving from repo_identifier...")
                    canonical_repo_identifier = resolve_canonical_repo_identifier(
                        repo_identifier=repo_identifier,
                        cache_dir=cache_dir,
                        repositories_dir=repositories_dir,
                    )
                    filesystem_root_dir = resolve_clone_dir_for_canonical_id(
                        canonical_repo_id=canonical_repo_identifier,
                        cache_dir=cache_dir,
                    )
            else:
                canonical_repo_identifier = resolve_canonical_repo_identifier(
                    repo_identifier=repo_identifier,
                    cache_dir=cache_dir,
                    repositories_dir=repositories_dir,
                )
                filesystem_root_dir = resolve_clone_dir_for_canonical_id(
                    canonical_repo_id=canonical_repo_identifier,
                    cache_dir=cache_dir,
                )

            # Emit explicit cache keys used for deterministic testing.
            try:
                idx = load_cache_index(cache_dir)
                faiss_key = idx.get(canonical_repo_identifier)
                graphs_idx = idx.get('graphs', {}) if isinstance(idx.get('graphs', {}), dict) else {}
                graph_key = graphs_idx.get(f"{canonical_repo_identifier}:combined")

                lines = [
                    f"canonical_repo_identifier={canonical_repo_identifier}",
                    f"filesystem_root_dir={filesystem_root_dir or '(not found)'}",
                ]
                if isinstance(faiss_key, str):
                    lines.append(f"faiss_key={faiss_key} files={faiss_key}.faiss,{faiss_key}.docs.pkl")
                else:
                    lines.append("faiss_key=(not found)")

                if isinstance(graph_key, str):
                    lines.append(f"graph_key={graph_key} file={graph_key}.code_graph.gz")
                else:
                    lines.append("graph_key=(not found)")

                _emit_thinking_step("cache", "Cache Selection", "\n".join(lines))
            except Exception:
                pass
        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "ValueError",
                "error_category": "invalid_input",
            }
        except Exception:
            canonical_repo_identifier = repo_identifier
            filesystem_root_dir = None

        if filesystem_root_dir and os.path.isdir(filesystem_root_dir):
            _print(f"Canonical repo id: {canonical_repo_identifier}")
            _print(f"Filesystem root (commit-matched): {filesystem_root_dir}")
        else:
            # No commit-matched clone found. Try on-demand shallow clone.
            # This is needed in K8s Jobs mode when:
            #  - PVC-shared: the Job wrote the clone to PVC, but the path isn't
            #    resolved correctly (rare: resolve_clone_dir may return None for
            #    clones that don't embed commit hash in directory name).
            #  - PVC-free (future): no clone exists on the controller at all.
            repo_cloned = False
            try:
                from .artifact_manager import is_jobs_mode
                repo_config = payload.get("repo_config", {})
                provider_type = repo_config.get("provider_type", "github")
                provider_config = repo_config.get("provider_config", {})
                project = repo_config.get("project")

                if is_jobs_mode() and provider_config:
                    _print("[deep_research] Repo clone not found locally — attempting on-demand shallow clone...")
                    _emit_thinking_step(
                        "clone", "Cloning Repository",
                        f"Shallow-cloning {repository} @ {branch} for filesystem tools..."
                    )
                    from .repo_providers import RepoProviderFactory
                    clone_config = RepoProviderFactory.from_toolkit_config(
                        provider_type=provider_type,
                        config=provider_config,
                        repository=repository,
                        branch=branch,
                        project=project,
                    )

                    # Build a deterministic clone path under repositories/
                    import hashlib
                    safe_repo = repository.replace("/", "_").replace("\\", "_")
                    clone_dest = os.path.join(
                        repositories_dir,
                        f"{safe_repo}_{branch}",
                    )

                    if not os.path.isdir(clone_dest):
                        os.makedirs(os.path.dirname(clone_dest), exist_ok=True)
                        import subprocess as sp
                        cmd = [
                            "git", "clone",
                            "--branch", clone_config.branch,
                            "--depth", "1",
                            "--single-branch",
                            clone_config.clone_url,
                            clone_dest,
                        ]
                        _print(f"  git clone --depth 1 --branch {branch} {clone_config.safe_url} ...")
                        result = sp.run(cmd, capture_output=True, text=True, check=False, timeout=300)
                        if result.returncode == 0:
                            filesystem_root_dir = clone_dest
                            repo_cloned = True
                            _print(f"  Clone complete: {clone_dest}")
                        else:
                            sanitized = clone_config.sanitize_output(result.stderr)
                            _print(f"  Git clone failed: {sanitized}")
                    else:
                        filesystem_root_dir = clone_dest
                        repo_cloned = True
                        _print(f"  Existing clone found: {clone_dest}")
            except Exception as clone_err:
                _print(f"[deep_research] On-demand clone failed: {clone_err}")

            if not repo_cloned:
                # Final fallback: use repositories/ or cache/ as root.
                # FilesystemBackend will still work but with limited coverage
                # (only files referenced by vector store documents).
                filesystem_root_dir = repositories_dir if os.path.isdir(repositories_dir) else cache_dir
                _print(f"Canonical repo id: {canonical_repo_identifier}")
                _print(f"Filesystem root fallback: {filesystem_root_dir}")
            else:
                _print(f"Canonical repo id: {canonical_repo_identifier}")
                _print(f"Filesystem root (on-demand clone): {filesystem_root_dir}")
        
        # Phase 6: Check for unified DB mode before loading legacy indexes
        _unified_retriever_enabled = True  # always-on — UnifiedWikiDB is the only retrieval store
        _unified_retriever_active = False
        retriever_stack = None

        if _unified_retriever_enabled:
            try:
                import glob as _glob
                from .unified_retriever import UnifiedRetriever
                from .unified_db import UnifiedWikiDB

                _udb_path = None
                _all_dbs = _glob.glob(os.path.join(cache_dir, "*.wiki.db"))
                if _all_dbs:
                    _all_dbs.sort(key=os.path.getmtime, reverse=True)
                    _udb_path = _all_dbs[0]

                if _udb_path and os.path.isfile(_udb_path):
                    _udb = UnifiedWikiDB(_udb_path, readonly=True)
                    _embed_fn = None
                    if embeddings and hasattr(embeddings, 'embed_query'):
                        _embed_fn = embeddings.embed_query
                    retriever_stack = UnifiedRetriever(
                        db=_udb,
                        embedding_fn=_embed_fn,
                        embeddings=embeddings,
                    )
                    _unified_retriever_active = True
                    _print(f"[UNIFIED_RETRIEVER] Deep Research using UnifiedRetriever from {_udb_path}")
                else:
                    _print("[UNIFIED_RETRIEVER] No .wiki.db found — falling back to legacy retriever")
            except Exception as _ur_exc:
                _print(f"[UNIFIED_RETRIEVER] Upgrade failed, using legacy: {_ur_exc}")

        # Initialize vector store manager (skip hard failure when UnifiedRetriever is active)
        vectorstore_manager = None
        vectorstore = None
        if not _unified_retriever_active:
            _print("Loading vector store...")
            vectorstore_manager = VectorStoreManager(
                cache_dir=cache_dir,
                model_cache_dir=model_cache_dir,
                embeddings=embeddings
            )
            vectorstore = vectorstore_manager.load_by_repo_name(canonical_repo_identifier)
            if vectorstore is None:
                return {
                    "success": False,
                    "error": f"No wiki index found for {repo_identifier}. Please generate a wiki first."
                }
            _print(f"Vector store loaded with {vectorstore.index.ntotal} vectors")
        else:
            _print("[UNIFIED_RETRIEVER] Skipping legacy FAISS vectorstore loading")
        
        # Load relationship graph (always try — tools benefit from it even in unified mode)
        _print("Loading relationship graph...")
        graph_manager = GraphManager(cache_dir=cache_dir)
        relationship_graph = graph_manager.load_graph_by_repo_name(canonical_repo_identifier)
        if relationship_graph:
            _print(f"Relationship graph loaded: {relationship_graph.number_of_nodes()} nodes")
        else:
            _print("No relationship graph found - proceeding without graph analysis")
        
        # Load repository analysis
        _print("Loading repository analysis...")
        analysis_store = RepositoryAnalysisStore(cache_dir=cache_dir)
        repository_analysis = ""
        if isinstance(analysis_key_override, str) and analysis_key_override.strip():
            repository_analysis = analysis_store.get_analysis_for_prompt(
                canonical_repo_identifier,
                analysis_key_override=analysis_key_override,
            )
            if not repository_analysis:
                _print(
                    f"Pinned analysis not found for key '{analysis_key_override}'. "
                    "Falling back to latest analysis for canonical repo id."
                )

        if not repository_analysis:
            repository_analysis = analysis_store.get_analysis_for_prompt(canonical_repo_identifier)
        if not repository_analysis:
            repository_analysis = analysis_store.get_analysis_for_prompt(repository)
        
        if repository_analysis:
            _print(f"Repository analysis loaded ({len(repository_analysis):,} chars)")
        else:
            _print("No repository analysis found")
        
        # Initialize retriever stack (legacy fallback if UnifiedRetriever not active)
        if retriever_stack is None:
            _print("Initializing legacy retriever stack...")
            retriever_stack = WikiRetrieverStack(
                vectorstore_manager=vectorstore_manager,
                relationship_graph=relationship_graph,
                use_enhanced_graph=True
            )
        
        # Safety net: if the retrieved docs point outside the chosen clone root,
        # widen to repositories/ so filesystem tools can still operate.
        try:
            if vectorstore_manager and vectorstore_manager.documents:
                sample = vectorstore_manager.documents[0]
                sample_source = (sample.metadata or {}).get('source')
                if isinstance(sample_source, str) and os.path.isabs(sample_source):
                    norm_root = os.path.normpath(filesystem_root_dir)
                    norm_source = os.path.normpath(sample_source)
                    if not norm_source.startswith(norm_root + os.sep) and norm_source != norm_root:
                        if os.path.isdir(repositories_dir):
                            _print(
                                "Warning: retrieved doc source does not match commit root; "
                                "widening filesystem root to repositories/ for compatibility."
                            )
                            filesystem_root_dir = repositories_dir
        except Exception:
            pass

        backend_factory = None
        try:
            from deepagents.backends import FilesystemBackend

            def _filesystem_backend_factory(rt, root_dir=filesystem_root_dir):
                _ = rt  # runtime intentionally unused
                try:
                    return FilesystemBackend(root_dir=root_dir, virtual_mode=True)
                except TypeError:
                    return FilesystemBackend(root_dir)

            backend_factory = _filesystem_backend_factory
        except Exception as e:
            _print(f"Failed to initialize FilesystemBackend; falling back to StateBackend: {e}")

        # Create deep research engine using DeepAgents
        _print("Creating deep research engine (using DeepAgents)...")
        _emit_thinking_step("engine", "Research Engine", "Initializing DeepAgents agent with subagents...")
        
        # Parse repository analysis into structured format
        repo_analysis_dict = None
        if repository_analysis:
            repo_analysis_dict = {"summary": repository_analysis}
        
        engine = create_deep_research_engine(
            retriever_stack=retriever_stack,
            graph_manager=graph_manager,
            code_graph=relationship_graph,
            repo_analysis=repo_analysis_dict,
            llm_client=llm,
            backend=backend_factory,
            llm_settings=llm_settings,
            max_iterations=max_iterations,
            research_type=research_type,
            enable_graph_analysis=relationship_graph is not None,
            enable_subagents=True
        )
        
        # Run research and collect events
        _print("Starting research loop...")
        _emit_thinking_step("loop", "Research Loop", "Beginning iterative research...")
        
        async for event in engine.research(question):
            event_type = event.get('event_type', '')
            data = event.get('data', {})
            
            collected_events.append(event)
            
            # Emit appropriate thinking steps based on event type
            if event_type == 'thinking_step':
                step = data
                step_type = step.get('type', 'step')
                
                if step_type == 'tool_call':
                    tool_name = step.get('tool', 'unknown')
                    tool_input = step.get('input', '')
                    # Use tool_call_id if available, otherwise generate from step + tool name
                    call_id = step.get('tool_call_id') or step.get('call_id') or f"{step.get('step', 0)}-{tool_name}"
                    _emit_thinking_step(
                        'tool_call',
                        f"Calling: {tool_name}",
                        tool_input[:500],
                        {"tool": tool_name, "step": step.get('step', 0), "call_id": call_id}
                    )
                elif step_type == 'tool_result':
                    tool_name = step.get('tool', 'unknown')
                    preview = step.get('output_preview', step.get('output', '')[:500])
                    output_length = step.get('output_length', len(str(preview)))
                    # Use matching call_id to link with tool_call
                    call_id = step.get('tool_call_id') or step.get('call_id') or f"{step.get('step', 0)}-{tool_name}"
                    _emit_thinking_step(
                        'tool_result',
                        f"Result ({output_length} chars)",
                        preview[:500] if preview else '',
                        {"tool": tool_name, "step": step.get('step', 0), "call_id": call_id}
                    )
                else:
                    # Generic step
                    _emit_thinking_step(
                        step_type,
                        step.get('title', f'Step {step.get("step", "")}'),
                        step.get('content', '')[:500],
                        step.get('metadata', {})
                    )
            
            elif event_type == 'todo_update':
                all_todos = data.get('todos', [])
                _emit_todo_update(all_todos)
            
            elif event_type == 'research_complete':
                final_report = data.get('report', '')
                all_todos = data.get('todos', [])
                _print(f"Research complete: {len(final_report)} chars report")
            
            elif event_type == 'research_error':
                error = data.get('error', 'Unknown error')
                _emit_thinking_step('error', 'Error', error)
                return {
                    "success": False,
                    "error": error,
                    "thinking_steps": [e for e in collected_events if e.get('event_type') == 'thinking_step']
                }
        
        _emit_thinking_step("complete", "Research Complete", f"Generated {len(final_report)} character report")
        
        return {
            "success": True,
            "report": final_report,
            "todos": all_todos,
            "thinking_steps": [
                e.get('data', {}) for e in collected_events 
                if e.get('event_type') == 'thinking_step'
            ],
            "total_events": len(collected_events)
        }
        
    except Exception as e:
        logger.exception("Deep research failed")
        _emit_thinking_step('error', 'Research Failed', str(e))
        return {
            "success": False,
            "error": f"Deep research error: {str(e)}\n{traceback.format_exc()}"
        }


def run_deep_research(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Synchronous wrapper for deep research."""
    return asyncio.run(run_deep_research_async(payload))


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for subprocess."""
    _configure_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Deep Research subprocess worker")
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args(argv)
    
    # Read input
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read input: {e}")
        result = {"success": False, "error": f"Failed to read input: {e}"}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f)
        return 1
    
    # Run deep research
    try:
        result = run_deep_research(payload)
    except Exception as e:
        logger.exception("Unhandled exception in deep research worker")
        result = {"success": False, "error": str(e)}
    
    # Write output
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write output: {e}")
        return 1
    
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
