"""Branch resolution helpers (#7273).

Since EL-6610 the ADO Repos toolkit can hold several repositories, so users
commonly leave the base/active branch empty. An empty/unset branch must
resolve to the repository's *real* default branch — never a guessed
``'main'`` — otherwise the clone, cache keys, registry entry and manifest
``wiki_id`` drift apart from what the UI expects.

Pure helpers plus thin ``git`` subprocess wrappers. No pylon imports here:
this module must import cleanly in subprocess workers and standalone tests.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)

_REFS_HEADS_PREFIX = "refs/heads/"


def normalize_branch(value) -> Optional[str]:
    """Normalize a configured branch value.

    Strips whitespace and a leading ``refs/heads/`` prefix. Returns ``None``
    for ``None``, non-string values, or an empty/whitespace-only string.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.startswith(_REFS_HEADS_PREFIX):
        stripped = stripped[len(_REFS_HEADS_PREFIX):].strip()
    return stripped or None


def select_configured_branch(*values) -> Optional[str]:
    """Return the first non-empty normalized branch among ``values``."""
    for value in values:
        normalized = normalize_branch(value)
        if normalized:
            return normalized
    return None


def parse_symref_head(output: str) -> Optional[str]:
    """Parse ``git ls-remote --symref <url> HEAD`` output for the default branch.

    Expected symref line shape: ``ref: refs/heads/master\\tHEAD``.
    """
    if not isinstance(output, str):
        return None
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("ref:"):
            continue
        rest = line[len("ref:"):].strip()
        ref = rest.split()[0] if rest else ""
        if ref.startswith(_REFS_HEADS_PREFIX):
            branch = ref[len(_REFS_HEADS_PREFIX):].strip()
            if branch:
                return branch
    return None


def resolve_remote_default_branch(clone_url: str, timeout: int = 30) -> Optional[str]:
    """Resolve a repository's default branch via ``git ls-remote --symref``.

    Returns ``None`` on any failure (non-zero exit, timeout, missing ``git``
    binary, bad input). Never logs ``clone_url`` — it can carry embedded
    credentials; callers should log a safe label instead.
    """
    if not clone_url:
        return None

    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        result = subprocess.run(
            ["git", "ls-remote", "--symref", clone_url, "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Timed out resolving remote default branch")
        return None
    except FileNotFoundError:
        logger.warning("git binary not found; cannot resolve remote default branch")
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to resolve remote default branch: %s", exc)
        return None

    if result.returncode != 0:
        return None

    return parse_symref_head(result.stdout)


def read_checked_out_branch(local_path: str) -> Optional[str]:
    """Return the checked-out branch name of a local clone.

    Returns ``None`` when detached or on any failure.
    """
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "--short", "-q", "HEAD"],
            cwd=local_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to read checked-out branch for %s: %s", local_path, exc)
        return None

    if result.returncode != 0:
        return None

    branch = result.stdout.strip()
    return branch or None


def indexed_branches(cache_dir, repo_path: str) -> List[str]:
    """Return the distinct branches indexed for ``repo_path`` in ``cache_dir``.

    Reads ``cache_index.json``'s ``refs`` keys (``repo:branch``), as used by
    ``repo_resolution.load_cache_index`` / ``split_repo_identifier`` /
    ``repo_branch_key``.
    """
    try:
        from .repo_resolution import load_cache_index, split_repo_identifier
    except ImportError:  # pragma: no cover - standalone import mode
        from repo_resolution import load_cache_index, split_repo_identifier

    index = load_cache_index(cache_dir)
    refs = index.get("refs")
    if not isinstance(refs, dict):
        return []

    normalized_repo_path = (repo_path or "").strip().strip("/")
    if not normalized_repo_path:
        return []

    branches: List[str] = []
    for key in refs.keys():
        if not isinstance(key, str):
            continue
        repo, branch, _commit = split_repo_identifier(key)
        if repo and repo.strip().strip("/") == normalized_repo_path and branch:
            if branch not in branches:
                branches.append(branch)
    return branches
