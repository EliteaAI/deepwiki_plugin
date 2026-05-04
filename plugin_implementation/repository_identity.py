"""Helpers for canonical repository and wiki artifact identity."""

from __future__ import annotations

from typing import Any, Dict, Optional


def canonical_repository_path(repository: str, clone_config: Optional[Any] = None) -> str:
    """Return the provider-normalized repository path used for cache keys."""
    repo_identifier = getattr(clone_config, "repo_identifier", None) if clone_config is not None else None
    if isinstance(repo_identifier, str) and repo_identifier.strip():
        return repo_identifier.strip().strip("/")
    return (repository or "").strip().strip("/")


def build_repo_identifier(
    *,
    repository: str,
    branch: str,
    commit_hash: Optional[str] = None,
    clone_config: Optional[Any] = None,
) -> str:
    """Build the canonical cache identifier for a repository generation."""
    repo_path = canonical_repository_path(repository, clone_config)
    active_branch = (branch or "main").strip() or "main"
    if commit_hash:
        return f"{repo_path}:{active_branch}:{commit_hash[:8]}"
    return f"{repo_path}:{active_branch}"


def clone_config_from_repo_config(repo_config: Optional[Dict[str, Any]], repository: str, branch: str) -> Optional[Any]:
    """Build a provider clone config from runtime repo_config when enough data exists."""
    if not isinstance(repo_config, dict) or not repository:
        return None

    provider_type = repo_config.get("provider_type", "github")
    provider_config = repo_config.get("provider_config") or {}
    project = repo_config.get("project")

    try:
        from .repo_providers import RepoProviderFactory

        return RepoProviderFactory.from_toolkit_config(
            provider_type=provider_type,
            config=provider_config,
            repository=repository,
            branch=branch,
            project=project,
        )
    except Exception:
        return None


def build_query_repo_identifier(
    *,
    repository: str,
    branch: str,
    repo_config: Optional[Dict[str, Any]] = None,
    clone_config: Optional[Any] = None,
) -> str:
    """Build the repo:branch cache lookup key used by ask/research.

    The generation path keys caches by provider-normalized repository paths.
    For Azure DevOps this is org/project/repo, while the toolkit's raw
    repository field can be only the repo name. Ask/research must therefore
    derive the same provider-aware path before resolving commit-scoped caches.
    """
    effective_clone_config = clone_config or clone_config_from_repo_config(repo_config, repository, branch)
    return build_repo_identifier(
        repository=repository,
        branch=branch,
        clone_config=effective_clone_config,
    )


def rebase_artifact_name(name: str, *, wiki_id: str, subfolder: str) -> str:
    """Place an artifact under the canonical wiki_id/subfolder path.

    Older exports can already contain a different wiki_id prefix, for example
    ``old-wiki/wiki_pages/page.md``. In that case we keep only the path below
    ``wiki_pages`` and rebase it under the canonical wiki folder.
    """
    artifact_name = (name or "").lstrip("/")
    if artifact_name.startswith(f"{wiki_id}/"):
        return artifact_name

    subfolder_prefix = f"{subfolder}/"
    nested_marker = f"/{subfolder}/"

    if artifact_name.startswith(subfolder_prefix):
        suffix = artifact_name[len(subfolder_prefix):]
    elif nested_marker in artifact_name:
        suffix = artifact_name.split(nested_marker, 1)[1]
    else:
        suffix = artifact_name

    return f"{wiki_id}/{subfolder}/{suffix.lstrip('/')}"