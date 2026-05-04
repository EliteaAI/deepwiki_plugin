#!/usr/bin/env python3
# coding=utf-8

"""Subprocess worker for DeepWiki generation.

This module is executed in a separate process to isolate heavy work (indexing,
parsing, graph building) from the main web worker process.

Contract:
- Reads JSON from --input
- Writes JSON to --output
- Emits progress logs to stdout

Input JSON schema (loosely):
{
  "base_path": "/tmp/wiki_builder",
  "query": "...",
  "llm_settings": {...},
  "embedding_model": "..." | null,
  "github_configuration": {...},
  "github_repository": "owner/repo",
  "github_base_branch": "main",
  "active_branch": "main",
  "force_rebuild_index": true,
  "indexing_method": "filesystem" | "github"
}

Output JSON schema:
{
  "success": bool,
  "result": str,
  "artifacts": list,
  "error": str (when success=false)
}
"""

import argparse
import logging
import json
import os
import sys
import traceback
from typing import Any, Dict
from datetime import datetime, timezone
import uuid


def _configure_logging() -> None:
    """Ensure stdlib logging emits in the subprocess.

    In the platform runtime, the parent process may configure logging, but this
    subprocess starts fresh; without basicConfig, INFO logs are often dropped and
    only WARNING+ appears.
    """
    level_name = os.getenv("DEEPWIKI_WORKER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    # Keep console logs verbose; thinking cleanup happens in the parent process.
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s -- %(message)s")
    )

    root = logging.getLogger()
    # Replace handlers to avoid duplicate logs if something preconfigured.
    root.handlers = [handler]
    root.setLevel(level)

    # Keep noisy libraries at WARNING unless explicitly requested.
    if level > logging.DEBUG:
        for noisy in ["httpx", "urllib3", "openai", "langchain", "langgraph"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _print(msg: str) -> None:
    # Make sure logs are flushed for real-time streaming.
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _build_llm_and_embeddings(llm_settings: Dict[str, Any], embedding_model: Any):
    # Detect provider - default to openai for backwards compatibility
    provider = llm_settings.get("provider", "openai")

    # Use normalized field names with fallback to legacy OpenAI-specific names
    api_base = llm_settings.get("api_base") or llm_settings.get("openai_api_base")
    api_key = llm_settings.get("api_key") or llm_settings.get("openai_api_key")
    model_name = llm_settings.get("model_name", "gpt-4o-mini")
    organization = llm_settings.get("organization")
    max_retries = llm_settings.get("max_retries", 2)
    max_tokens = llm_settings.get("max_tokens", 64000)
    default_headers = llm_settings.get("default_headers", {})
    # Honor platform streaming configuration (defaults to enabled).
    streaming = llm_settings.get("streaming", True)

    if not api_base:
        raise ValueError("llm_settings.api_base is required")
    if not api_key:
        raise ValueError("llm_settings.api_key is required")

    from langchain_openai import ChatOpenAI
    from langchain_openai.embeddings import OpenAIEmbeddings

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Anthropic models don't use the /v1 suffix
        anthropic_base_url = api_base.rstrip("/")
        if anthropic_base_url.endswith("/v1"):
            anthropic_base_url = anthropic_base_url[:-3]

        # Temperature: use 0.1 as default for Anthropic
        temperature = 0.1

        # Build default_headers if not provided
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
            temperature=temperature,
            max_retries=max_retries,
            streaming=streaming,
            default_headers=default_headers,
        )

        # For embeddings, we still need to use OpenAI-compatible endpoint
        # Embeddings typically go through /llm/v1 path
        embeddings_base_url = api_base
        if not embeddings_base_url.endswith("/v1"):
            embeddings_base_url = embeddings_base_url.rstrip("/") + "/v1"
    else:
        # OpenAI models
        temperature = 1.0 if str(model_name).startswith("o") else 0.1

        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=api_base,
            organization=organization,
            max_retries=max_retries,
            streaming=streaming,
            max_tokens=max_tokens,
        )
        embeddings_base_url = api_base

    embedding_model_name = embedding_model if embedding_model else "text-embedding-3-large"
    embeddings = OpenAIEmbeddings(
        model=embedding_model_name,
        api_key=api_key,
        base_url=embeddings_base_url,
        organization=organization,
        max_retries=max_retries,
    )

    return llm, embeddings, embedding_model_name


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        _configure_logging()
        # Best-effort line buffering for immediate streaming.
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(line_buffering=True)
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(line_buffering=True)
        except Exception:
            pass

        with open(args.input, "r", encoding="utf-8") as f:
            payload = json.load(f)

        base_path = os.path.abspath(payload.get("base_path") or "/tmp/wiki_builder")
        os.makedirs(base_path, exist_ok=True)

        # HF caches
        model_cache_dir = os.path.join(base_path, "huggingface_cache")
        os.makedirs(model_cache_dir, exist_ok=True)
        os.environ["TRANSFORMERS_CACHE"] = model_cache_dir
        os.environ["HF_HOME"] = model_cache_dir
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = model_cache_dir

        # Bridge user-facing wiki-generation knobs from the payload to the
        # environment so feature_flags.get_feature_flags() and the agent's
        # _resolve_planner_choice helper can pick them up without an
        # additional plumbing layer.
        _planner_mode = payload.get("planner_mode") or payload.get("planner_type")
        if _planner_mode:
            os.environ["DEEPWIKI_STRUCTURE_PLANNER"] = str(_planner_mode).strip().lower()
        if "exclude_tests" in payload and payload.get("exclude_tests") is not None:
            os.environ["DEEPWIKI_EXCLUDE_TESTS"] = "1" if payload.get("exclude_tests") else "0"

        query = payload.get("query")
        if not query:
            raise ValueError("query is required")

        llm_settings = payload.get("llm_settings") or {}
        embedding_model = payload.get("embedding_model")
        
        # New multi-provider repo_config approach
        repo_config = payload.get("repo_config") or {}
        
        # Legacy fallback for backward compatibility
        github_configuration = payload.get("github_configuration") or {}
        
        # Extract repository info from repo_config or legacy fields
        if repo_config:
            provider_type = repo_config.get("provider_type", "github")
            provider_config = repo_config.get("provider_config", {})
            repository = repo_config.get("repository")
            branch = repo_config.get("branch", "main")
            project = repo_config.get("project")
            active_branch = payload.get("active_branch", branch)
        else:
            # Legacy GitHub-only mode
            provider_type = "github"
            provider_config = github_configuration
            repository = payload.get("github_repository")
            branch = payload.get("github_base_branch", "main")
            project = None
            active_branch = payload.get("active_branch", branch)
        
        if not repository:
            raise ValueError("repository is required (either in repo_config or github_repository)")
        
        force_rebuild_index = bool(payload.get("force_rebuild_index", True))
        indexing_method = payload.get("indexing_method", "filesystem")

        _print(f"[worker] Building LLM+embeddings (repo={repository}, branch={active_branch}, provider={provider_type})")
        llm, embeddings, embedding_model_name = _build_llm_and_embeddings(llm_settings, embedding_model)
        _print(f"[worker] Embeddings: {embedding_model_name}")

        # Build clone configuration using the multi-provider factory
        from plugin_implementation.repo_providers import RepoProviderFactory, GitCloneConfig
        
        try:
            clone_config = RepoProviderFactory.from_toolkit_config(
                provider_type=provider_type,
                config=provider_config,
                repository=repository,
                branch=branch,
                project=project,
            )
            _print(f"[worker] Clone config built: {clone_config.provider.value} - {clone_config.repo_identifier} @ {clone_config.branch}")
        except Exception as e:
            _print(f"[worker] Failed to build clone config: {e}. Will fall back to direct credentials if available.")
            clone_config = None

        from plugin_implementation.hybrid_wiki_toolkit_wrapper import HybridWikiToolkitWrapper

        wrapper = HybridWikiToolkitWrapper(
            # Multi-provider configuration
            repo_config=repo_config,
            clone_config=clone_config,
            # Legacy GitHub-specific fields for backward compatibility
            github_repository=repository,
            github_base_branch=branch,
            active_branch=active_branch,
            cache_dir=os.path.join(base_path, "cache"),
            model_cache_dir=model_cache_dir,
            force_rebuild_index=force_rebuild_index,
            llm=llm,
            embeddings=embeddings,
            indexing_method=indexing_method,
        )

        _print(f"[worker] Starting wiki generation: {query}")
        result = wrapper.generate_wiki(query=query)

        # Save repository analysis for Ask tool if wiki generation succeeded
        if result and result.get("success"):
            try:
                from plugin_implementation.repository_analysis_store import RepositoryAnalysisStore
                
                cache_dir = os.path.join(base_path, "cache")
                analysis_store = RepositoryAnalysisStore(cache_dir)
                
                # Use actual branch from clone result (handles master vs main, etc.)
                actual_branch = result.get("branch") or active_branch
                if actual_branch != active_branch:
                    _print(f"[worker] Using actual branch from clone: {actual_branch} (requested: {active_branch})")
                    active_branch = actual_branch
                
                # Save only the LLM-generated repository analysis text
                # This is the comprehensive analysis from _llm_analyze_repository
                # based on README.md and directory structure (bottom-up approach)
                repo_context = result.get("repository_context", "")
                commit_hash = result.get("commit_hash")
                
                # Use identifier format with commit hash for cache isolation: "{repo}:{branch}:{commit_short}"
                from plugin_implementation.repository_identity import (
                    build_repo_identifier,
                    canonical_repository_path,
                    rebase_artifact_name,
                )

                canonical_repository = canonical_repository_path(repository, clone_config)
                repo_identifier = build_repo_identifier(
                    repository=repository,
                    branch=actual_branch,
                    commit_hash=commit_hash,
                    clone_config=clone_config,
                )

                wiki_version_id = (
                    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    + "-"
                    + uuid.uuid4().hex[:8]
                )

                analysis_key_override = f"{repo_identifier}@{wiki_version_id}"

                if repo_context:
                    # Backward compatible: keep the legacy analysis record (newest wins).
                    analysis_store.save_analysis(
                        repo_identifier=repo_identifier,
                        analysis=repo_context,  # Just the LLM analysis text
                        commit_hash=commit_hash,
                        metadata={
                            "branch": active_branch,
                            "indexing_method": indexing_method,
                            "commit_hash": commit_hash,
                            "provider_type": provider_type,
                        }
                    )

                    # Versioned analysis record: addressed by analysis_key_override, so UI can select.
                    analysis_store.save_analysis(
                        repo_identifier=repo_identifier,
                        analysis=repo_context,
                        commit_hash=commit_hash,
                        metadata={
                            "branch": active_branch,
                            "indexing_method": indexing_method,
                            "commit_hash": commit_hash,
                            "wiki_version_id": wiki_version_id,
                        },
                        analysis_key_override=analysis_key_override,
                    )
                    _print(f"[worker] Saved repository analysis for Ask tool ({len(repo_context)} chars) as {repo_identifier}")
                else:
                    _print("[worker] Warning: No repository_context in result, skipping analysis save")

                # Emit a manifest artifact listing the exact wiki pages for this run.
                # Context7-style: all artifacts are prefixed with wiki_id folder
                from plugin_implementation.registry_manager import normalize_wiki_id
                
                wiki_id = normalize_wiki_id(repo_identifier)
                _print(f"[worker] Wiki ID for folder structure: {wiki_id}")
                
                pages = []
                try:
                    for art in (result.get("artifacts") or []):
                        name = art.get("name")
                        if art.get("type") == "text/markdown" and isinstance(name, str) and name.endswith(".md"):
                            prefixed_name = rebase_artifact_name(name, wiki_id=wiki_id, subfolder="wiki_pages")
                            art["name"] = prefixed_name
                            pages.append(prefixed_name)
                except Exception:
                    pages = []
                
                # Extract wiki_title from wiki_structure, build description from repository_context
                wiki_title = ""
                wiki_description = ""
                try:
                    for art in (result.get("artifacts") or []):
                        name = art.get("name", "")
                        if art.get("type") == "application/json" and "wiki_structure" in name:
                            data = art.get("data", "")
                            if isinstance(data, str):
                                ws = json.loads(data)
                                wiki_title = ws.get("wiki_title", "")
                            break
                except Exception as ws_err:
                    _print(f"[worker] Warning: Failed to extract wiki title: {ws_err}")
                
                # Build description from repository_context JSON for LLM resolution
                # Uses executive_summary + core_purpose for rich semantic matching
                try:
                    if repo_context and repo_context.strip().startswith("{"):
                        ctx_json = json.loads(repo_context)
                        exec_summary = ctx_json.get("executive_summary", "")
                        core_purpose = ctx_json.get("core_purpose", "")
                        if exec_summary or core_purpose:
                            wiki_description = f"{exec_summary} {core_purpose}".strip()
                except Exception as ctx_err:
                    _print(f"[worker] Warning: Failed to parse repository_context for description: {ctx_err}")
                
                # Also prefix JSON artifacts (wiki_structure) with wiki_id folder
                try:
                    for art in (result.get("artifacts") or []):
                        name = art.get("name")
                        if art.get("type") == "application/json" and isinstance(name, str) and "wiki_structure" in name:
                            art["name"] = rebase_artifact_name(name, wiki_id=wiki_id, subfolder="analysis")
                except Exception:
                    pass

                manifest = {
                    "schema_version": 2,  # Version 2 = folder structure
                    "wiki_id": wiki_id,
                    "wiki_title": wiki_title,
                    "description": wiki_description,
                    "wiki_version_id": wiki_version_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "canonical_repo_identifier": repo_identifier,
                    "repository": canonical_repository,
                    "branch": active_branch,
                    "commit_hash": commit_hash,
                    "analysis_key": analysis_key_override,
                    "pages": pages,
                    "provider_type": provider_type,
                }

                # Attach cache keys for deterministic artifact resolution (vectorstore/graph/docstore).
                # Context7-style: paths are relative to wiki_id folder
                try:
                    from plugin_implementation.repo_resolution import load_cache_index

                    cache_dir = os.path.join(base_path, "cache")
                    idx = load_cache_index(cache_dir)
                    faiss_key = idx.get(repo_identifier) if isinstance(idx, dict) else None
                    graphs_idx = idx.get("graphs", {}) if isinstance(idx.get("graphs", {}), dict) else {}
                    docs_idx = idx.get("docs", {}) if isinstance(idx.get("docs", {}), dict) else {}
                    bm25_idx = idx.get("bm25", {}) if isinstance(idx.get("bm25", {}), dict) else {}

                    graph_key = graphs_idx.get(f"{repo_identifier}:combined")
                    docstore_key = docs_idx.get(repo_identifier)
                    bm25_key = bm25_idx.get(repo_identifier)

                    if isinstance(faiss_key, str):
                        manifest["faiss_cache_key"] = faiss_key
                    if isinstance(graph_key, str):
                        manifest["graph_cache_key"] = graph_key
                    if isinstance(docstore_key, str):
                        manifest["docstore_cache_key"] = docstore_key
                        manifest["docstore_files"] = [
                            f"{docstore_key}.docstore.bin",
                            f"{docstore_key}.doc_index.json",
                        ]
                    if isinstance(bm25_key, str):
                        manifest["bm25_cache_key"] = bm25_key
                        manifest["bm25_files"] = [
                            f"{bm25_key}.bm25.sqlite",
                        ]

                    # Unified DB key (.wiki.db) — Phase 6 unified graph+cluster DB
                    udb_idx = idx.get("unified_db", {}) if isinstance(idx.get("unified_db", {}), dict) else {}
                    udb_key = udb_idx.get(repo_identifier)
                    if not udb_key:
                        # Fallback: look for the .wiki.db file directly
                        import glob as _glob
                        _udb_matches = _glob.glob(os.path.join(cache_dir, "*.wiki.db"))
                        if _udb_matches:
                            _udb_matches.sort(key=os.path.getmtime, reverse=True)
                            udb_key = os.path.splitext(os.path.basename(_udb_matches[0]))[0]
                    if isinstance(udb_key, str):
                        manifest["unified_db_key"] = udb_key
                        manifest["unified_db_files"] = [f"{udb_key}.wiki.db"]

                    # Store the hashed analysis cache key so manifest_index_files()
                    # can resolve the actual filename ({md5}_analysis.json).
                    if isinstance(analysis_key_override, str) and analysis_key_override:
                        import hashlib
                        analysis_hash = hashlib.md5(analysis_key_override.encode()).hexdigest()
                        manifest["analysis_cache_key"] = analysis_hash
                except Exception as mf_cache_err:
                    _print(f"[worker] Warning: Failed to attach cache keys to manifest: {mf_cache_err}")

                try:
                    artifacts = result.get("artifacts")
                    if not isinstance(artifacts, list):
                        artifacts = []
                        result["artifacts"] = artifacts
                    
                    # Store manifest at {wiki_id}/wiki_manifest_{version}.json
                    artifacts.append(
                        {
                            "name": f"{wiki_id}/wiki_manifest_{wiki_version_id}.json",
                            "object_type": "wiki_manifest",
                            "type": "application/json",
                            "data": json.dumps(manifest, indent=2, ensure_ascii=False),
                        }
                    )

                    # Also surface metadata at the top-level result for debugging and registry.
                    result["wiki_version_id"] = wiki_version_id
                    result["wiki_id"] = wiki_id
                    result["analysis_key"] = analysis_key_override
                    result["canonical_repo_identifier"] = repo_identifier
                    result["branch"] = active_branch
                    result["commit_hash"] = commit_hash
                    result["provider_type"] = provider_type
                    result["wiki_title"] = wiki_title
                    result["wiki_description"] = wiki_description
                except Exception as mf_err:
                    _print(f"[worker] Warning: Failed to append wiki manifest artifact: {mf_err}")

                # Upload index artifacts to platform bucket (only in Jobs+API mode).
                # In Docker mode or Jobs+PVC mode, this is a no-op.
                try:
                    from plugin_implementation.artifact_manager import ArtifactManager
                    art_mgr = ArtifactManager(
                        cache_dir=os.path.join(base_path, "cache"),
                    )
                    if art_mgr._needs_transport:
                        _print("[worker] Jobs+API mode: uploading index artifacts to platform bucket...")
                        art_mgr.upload_indexes(wiki_id=wiki_id, manifest=manifest)

                        # --- Index cleanup (best-effort) ---
                        try:
                            idx_stats = art_mgr.cleanup_stale_indexes(
                                wiki_id=wiki_id,
                                current_manifest=manifest,
                            )
                            if idx_stats.get("indexes_deleted"):
                                _print(
                                    f"[worker] Index cleanup for {wiki_id}: "
                                    f"{idx_stats['indexes_deleted']} orphaned indexes deleted"
                                )
                            elif idx_stats.get("indexes_up_to_date"):
                                _print(f"[worker] Index cleanup for {wiki_id}: indexes already up to date")
                        except Exception as cleanup_err:
                            _print(f"[worker] Index cleanup failed (non-fatal): {cleanup_err}")

                        # --- Content version retention (best-effort) ---
                        try:
                            cv_stats = art_mgr.cleanup_old_content_versions(
                                wiki_id=wiki_id,
                                current_manifest=manifest,
                            )
                            if cv_stats.get("manifests_deleted") or cv_stats.get("pages_deleted"):
                                _print(
                                    f"[worker] Content cleanup for {wiki_id}: "
                                    f"{cv_stats.get('manifests_deleted', 0)} manifests, "
                                    f"{cv_stats.get('pages_deleted', 0)} pages removed"
                                )
                        except Exception as cleanup_err:
                            _print(f"[worker] Content version cleanup failed (non-fatal): {cleanup_err}")
                    else:
                        _print("[worker] Docker mode or no transport — skipping index upload")

                except Exception as upload_err:
                    _print(f"[worker] Warning: Failed to upload index artifacts: {upload_err}")
            except Exception as save_err:
                _print(f"[worker] Warning: Failed to save repository analysis: {save_err}")

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f)

        _print("[worker] Done")
        return 0

    except Exception as ex:  # pylint: disable=W0718
        # Log full traceback for debugging (server-side)
        full_traceback = traceback.format_exc()
        _print(f"[worker] Error traceback:\n{full_traceback}")
        
        # Extract clean error message for user (no technical details)
        user_message = str(ex)
        
        # Categorize the error for structured handling
        error_category = "unknown_error"
        if "Authentication failed" in user_message or "Permission denied" in user_message:
            error_category = "authentication_error"
        elif "Branch" in user_message and "does not exist" in user_message:
            error_category = "branch_not_found"
        elif "Repository not found" in user_message:
            error_category = "repository_not_found"
        elif "Repository indexing failed" in user_message:
            error_category = "indexing_failed"
        elif "No documents found" in user_message:
            error_category = "empty_repository"
        elif "rate limit" in user_message.lower():
            error_category = "rate_limit"
        elif "timeout" in user_message.lower():
            error_category = "timeout"
        
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({
                    "success": False, 
                    "error": user_message,  # Clean message only
                    "error_category": error_category
                }, f)
        except Exception:
            pass
        _print("[worker] Failed: " + user_message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
