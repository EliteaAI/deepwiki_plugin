"""Helpers for canonical repository and wiki artifact identity."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def normalize_repositories(value: Any) -> List[str]:
    """Coerce a toolkit repository setting into a de-duplicated list of names.

    Mirrors elitea_sdk ADO Repos ``normalize_repositories``: legacy toolkits
    store a single string, multi-repo toolkits (EL-6610) store a list that may
    be empty.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    elif not isinstance(value, (list, tuple)):
        value = [value]
    repositories: List[str] = []
    for item in value:
        if item is None:
            continue
        item = str(item).strip()
        if item and item not in repositories:
            repositories.append(item)
    return repositories


def select_single_repository(value: Any) -> Optional[str]:
    """Pick the one repository DeepWiki works on from a toolkit setting.

    DeepWiki indexes exactly one repository. With several configured, the
    first one is used — the same default the ADO Repos toolkit binds per call.
    Returns None when nothing is configured.
    """
    repositories = normalize_repositories(value)
    return repositories[0] if repositories else None


def repository_from_identifier(repo_identifier: Any, provider_type: Optional[str] = None) -> Optional[str]:
    """Recover the toolkit-level repository from a canonical ``repo:branch[:commit]`` id.

    ADO canonical ids are ``org/project/repo`` while the toolkit field holds the
    repository name only.
    """
    if not isinstance(repo_identifier, str):
        return None
    path = repo_identifier.strip().split(":", 1)[0].strip("/")
    if not path:
        return None
    if provider_type in ("ado_repos", "ado", "azure_devops"):
        return path.rsplit("/", 1)[-1] or None
    return path


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