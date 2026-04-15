"""
Artifact Manager — Storage + transport layer for wiki indexes.

Supports two deployment modes, automatically detected:

1. Docker mode (default, DEEPWIKI_JOBS_ENABLED=false):
   - All artifacts live on local filesystem at cache_dir.
   - generate_wiki writes indexes locally (same process).
   - ask/deep_research read indexes from local cache_dir.
   - No artifact sync — local fs IS the persistent store.
   - ArtifactManager methods are no-ops (return True immediately).

2. K8s Jobs + platform artifacts (DEEPWIKI_JOBS_ENABLED=true):
   - generate_wiki runs as a K8s Job with **emptyDir**.
   - Worker uploads index files to the "wiki_artifacts" bucket.
   - Versioned manifest (wiki_manifest_{version}.json) is uploaded separately
     by the invoke.py result pipeline as a normal artifact to the same bucket.
   - Controller lists versioned manifests (newest = latest) and downloads
     indexes from platform bucket → local PVC cache.
   - "Keep latest only" eviction keeps controller cache bounded.
   - Eager pre-download after generation eliminates cold-start for ask.
   - Workers are fully stateless (no PVC dependency).

Mode detection:
   DEEPWIKI_JOBS_ENABLED=true → Jobs+API (production K8s)
   Otherwise                  → Docker   (local development)

Key design principle:
    Both modes share the same filesystem paths (cache_dir/).
    The difference is only the TRANSPORT:
    - Docker:    local-only (everything in one process/pod)
    - Jobs+API:  platform artifact HTTP API (multi-pod, no shared filesystem)
    Core generation/query code is 100% shared — only this module differs.
"""

from __future__ import annotations

import json
import gzip
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature flags and deployment mode detection
# ---------------------------------------------------------------------------

_JOBS_ENABLED_ENV = "DEEPWIKI_JOBS_ENABLED"


def is_jobs_mode() -> bool:
    """Check if K8s Jobs mode is enabled (wiki generation dispatched to Jobs)."""
    return os.environ.get(_JOBS_ENABLED_ENV, "false").lower() == "true"


def _has_platform_transport() -> bool:
    """Check if platform artifact transport is configured (Jobs+API mode).

    Workers get DEEPWIKI_ARTIFACT_BASE_URL and DEEPWIKI_ARTIFACT_BUCKET
    injected by the controller.  On the controller these are not set, but
    the controller constructs its ArtifactManager with an explicit client,
    so this check is only relevant for worker auto-setup.
    """
    from .artifacts_platform_client import (
        ARTIFACT_BUCKET_ENV, ARTIFACT_BASE_URL_ENV,
    )
    return bool(
        os.environ.get(ARTIFACT_BUCKET_ENV, "").strip()
        or os.environ.get(ARTIFACT_BASE_URL_ENV, "").strip()
    )


def needs_artifact_transport() -> bool:
    """
    Check if platform artifact transport is needed.

    Transport is needed when:
    - Jobs mode is enabled (generate_wiki dispatched to ephemeral pods)
    - AND platform artifact bucket/API is configured
    """
    return is_jobs_mode() and _has_platform_transport()


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

# Files referenced by cache keys in the manifest
_INDEX_FILE_PATTERNS = {
    "faiss": ["{key}.faiss"],
    "docstore": ["{key}.docstore.bin", "{key}.doc_index.json"],
    "graph": ["{key}.code_graph.gz"],
    "bm25": ["{key}.bm25.sqlite"],
    "fts5": ["{key}.fts5.db"],
    "unified_db": ["{key}.wiki.db"],
    "analysis": ["{key}_analysis.json"],
}


def manifest_index_files(manifest: Dict[str, Any]) -> List[str]:
    """
    Extract all local index filenames from a manifest.

    Returns relative filenames (no directory prefix) that should exist in cache_dir.
    """
    files: List[str] = []

    for key_field, patterns in [
        ("faiss_cache_key", _INDEX_FILE_PATTERNS["faiss"]),
        ("docstore_cache_key", _INDEX_FILE_PATTERNS["docstore"]),
        ("graph_cache_key", _INDEX_FILE_PATTERNS["graph"]),
        ("graph_cache_key", _INDEX_FILE_PATTERNS["fts5"]),
        ("bm25_cache_key", _INDEX_FILE_PATTERNS["bm25"]),
        ("unified_db_key", _INDEX_FILE_PATTERNS["unified_db"]),
    ]:
        cache_key = manifest.get(key_field)
        if isinstance(cache_key, str) and cache_key:
            for pattern in patterns:
                files.append(pattern.format(key=cache_key))

    # BM25 files may be listed explicitly
    for f in manifest.get("bm25_files", []):
        if isinstance(f, str) and f not in files:
            files.append(f)

    # Docstore files may be listed explicitly
    for f in manifest.get("docstore_files", []):
        if isinstance(f, str) and f not in files:
            files.append(f)

    # Analysis — uses hashed cache key (md5), NOT the human-readable analysis_key
    analysis_cache_key = manifest.get("analysis_cache_key")
    if isinstance(analysis_cache_key, str) and analysis_cache_key:
        for pattern in _INDEX_FILE_PATTERNS["analysis"]:
            files.append(pattern.format(key=analysis_cache_key))

    return files


# ---------------------------------------------------------------------------
# ArtifactManager
# ---------------------------------------------------------------------------

class ArtifactManager:
    """
    Manages artifact storage and retrieval across deployment modes.

    Behavior by mode:
    - Docker mode: No-op. Everything is local filesystem.
    - Jobs + API: Full upload/download via platform artifact HTTP API.

    Usage in ask/deep_research subprocess workers::

        mgr = ArtifactManager(cache_dir=cache_dir, artifacts_client=client, bucket=bucket)
        mgr.ensure_indexes_for_wiki(wiki_id=wiki_id, manifest=manifest)
        # Now VectorStoreManager / GraphManager / etc. can load from cache_dir as usual.

    Usage in wiki_job_worker after generation::

        mgr = ArtifactManager(cache_dir=cache_dir)
        mgr.upload_indexes(wiki_id=wiki_id, manifest=manifest)
        # Uploads index files to platform bucket (no-op if not API mode).
        # Versioned manifest is uploaded separately by invoke.py result pipeline.

    Usage after generate_wiki completes (controller side)::

        mgr.eager_preload(wiki_id=wiki_id)
        # Downloads indexes from platform bucket to local cache.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        artifacts_client=None,
        bucket: str = "",
        max_cached_repos: int = 10,
    ):
        """
        Args:
            cache_dir: Local filesystem cache directory (e.g. /data/wiki_builder/cache).
            artifacts_client: PlatformArtifactClient (has download_artifact/upload_artifact).
                              Only needed in Jobs+API mode (when no shared PVC).
            bucket: Platform bucket name (e.g. 'wiki_artifacts').
            max_cached_repos: Max repos to keep in local cache ("keep latest only").
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = artifacts_client
        self.bucket = bucket
        self.max_cached_repos = max_cached_repos
        self._jobs_mode = is_jobs_mode()
        self._needs_transport = needs_artifact_transport()

        # Auto-create platform artifact client from env when in Jobs+API mode
        # and no explicit client was provided.
        if self.client is None and self._needs_transport:
            from .artifacts_platform_client import (
                create_platform_client_from_env, get_artifact_bucket,
            )
            self.client = create_platform_client_from_env()
            if self.client and not self.bucket:
                self.bucket = get_artifact_bucket()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_indexes_for_wiki(
        self,
        wiki_id: str,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Ensure all index files for a wiki are available in cache_dir.

        Behavior by mode:
        - Docker: Always True (files are already local).
        - Jobs+API: Download missing files from platform bucket.

        Args:
            wiki_id: Wiki identifier (e.g. "owner--repo--main").
            manifest: Parsed manifest dict. If None, will be fetched from bucket.

        Returns:
            True if core indexes are available locally, False on failure.
        """
        logger.info(
            "ensure_indexes_for_wiki called: wiki_id=%s jobs_mode=%s client=%s bucket=%s cache_dir=%s",
            wiki_id, self._jobs_mode,
            type(self.client).__name__ if self.client else None,
            self.bucket, self.cache_dir,
        )
        if not self._jobs_mode:
            return True  # Docker mode — local fs is the source of truth

        # --- Jobs+API mode: download from platform bucket ---
        if not self.client:
            logger.warning(
                "ArtifactManager: Jobs+API mode but no artifacts_client configured — "
                "cannot download indexes. Pass artifacts_client to ArtifactManager."
            )
            return False

        # Fetch manifest if not provided
        if manifest is None:
            manifest = self._fetch_latest_manifest(wiki_id)
            if manifest is None:
                logger.warning(f"No manifest found for wiki_id={wiki_id}")
                return False

        # Determine which files need downloading
        needed_files = manifest_index_files(manifest)
        missing = [f for f in needed_files if not (self.cache_dir / f).exists()]

        if not missing:
            logger.info(f"All {len(needed_files)} index files already cached for {wiki_id}")
            # Even when files are present, ensure cache_index.json is up to date
            # (may have been lost on pod restart while PVC files survived).
            self._register_manifest_in_cache_index(manifest)
            return True

        logger.info(f"Downloading {len(missing)}/{len(needed_files)} missing index files for {wiki_id}")

        # Download missing files from bucket
        ok = True
        failed_files: List[str] = []
        for filename in missing:
            # Files in bucket are stored under {wiki_id}/indexes/{filename}
            # or directly under {wiki_id}/{filename} depending on upload format
            downloaded = self._download_index_file(wiki_id, filename)
            if not downloaded:
                logger.error(f"Failed to download index file: {filename}")
                ok = False
                failed_files.append(filename)

        # Register cache_index.json even if non-critical files (e.g. analysis) failed.
        # Core indexes (faiss, graph, docstore, bm25) are what matter for ask/deep_research.
        # If ALL downloads failed, don't register.
        downloaded_count = len(missing) - len(failed_files)
        if downloaded_count > 0:
            self._register_manifest_in_cache_index(manifest)
            self._evict_old_repos()
            if failed_files:
                logger.warning(
                    f"Registered cache_index despite {len(failed_files)} failed downloads "
                    f"(non-critical): {failed_files}"
                )

        return ok

    def eager_preload(self, wiki_id: str) -> bool:
        """
        Pre-download all indexes for a wiki immediately after generation.

        Called by the controller after a generate_wiki Job succeeds.
        Downloads indexes from platform bucket to local cache.
        This eliminates cold-start latency for the first ask/deep_research.

        Returns True if successful.
        """
        if not self._jobs_mode:
            return True  # Docker mode — nothing to preload

        logger.info(f"Eager pre-loading indexes for {wiki_id}")
        return self.ensure_indexes_for_wiki(wiki_id=wiki_id, manifest=None)

    def upload_indexes(
        self,
        wiki_id: str,
        manifest: Dict[str, Any],
    ) -> bool:
        """
        Upload index files from local cache_dir to platform bucket.

        Behavior by mode:
        - Docker: No-op (local fs is the persistent store).
        - Jobs+API: Upload all index files to bucket under {wiki_id}/indexes/.

        Returns True if all uploads succeeded (or if no upload needed).
        """
        if not self._jobs_mode:
            return True  # Docker mode — no upload needed

        if not self.client:
            logger.warning(
                "ArtifactManager: Jobs+API mode but no artifacts_client — cannot upload"
            )
            return False

        files_to_upload = manifest_index_files(manifest)
        logger.info(f"Uploading {len(files_to_upload)} index files for {wiki_id}")

        ok = True
        total_bytes = 0
        t0 = time.monotonic()

        for filename in files_to_upload:
            local_path = self.cache_dir / filename
            if not local_path.exists():
                logger.warning(f"Index file not found locally, skipping upload: {filename}")
                continue

            bucket_path = f"{wiki_id}/indexes/{filename}"
            try:
                data = local_path.read_bytes()
                total_bytes += len(data)
                self.client.upload_artifact(
                    self.bucket,
                    bucket_path,
                    data,
                )
                logger.debug(f"Uploaded {filename} ({len(data):,} bytes) → {bucket_path}")
            except Exception as e:
                logger.error(f"Failed to upload {filename}: {e}")
                ok = False

        # NOTE: Content artifacts (pages, manifest, structure JSON) are also
        # uploaded directly by wiki_job_worker.py — see the content artifact
        # upload block that runs immediately after upload_indexes().
        # The versioned wiki_manifest_{version}.json and wiki pages are uploaded
        # directly to the bucket with correct folder paths to avoid Pylon's
        # result_objects stripping directory prefixes from artifact names.

        elapsed = time.monotonic() - t0
        logger.info(
            f"Upload complete: {len(files_to_upload)} index files, "
            f"{total_bytes / (1024 * 1024):.1f} MB in {elapsed:.1f}s"
        )
        return ok

    def list_cached_repos(self) -> List[str]:
        """
        List all repo identifiers that have indexes in local cache.

        Reads from cache_index.json (the same index used by VectorStoreManager, etc.).
        """
        from .repo_resolution import load_cache_index, _RESERVED_TOP_LEVEL_KEYS

        idx = load_cache_index(self.cache_dir)
        repos = []
        for key in idx:
            if key not in _RESERVED_TOP_LEVEL_KEYS and isinstance(idx[key], str):
                repos.append(key)
        return repos

    # ------------------------------------------------------------------
    # Private: download
    # ------------------------------------------------------------------

    def _fetch_latest_manifest(self, wiki_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest versioned manifest for *wiki_id* from platform bucket.

        Primary path: list artifacts and find the newest
        ``{wiki_id}/wiki_manifest_{version}.json``.  These versioned manifests
        are uploaded through the normal invoke.py artifact pipeline.

        Legacy fallback: try ``{wiki_id}/manifest.json`` for older wikis that
        were generated before the consolidation to versioned manifests.
        """
        logger.info(
            "Fetching manifest for wiki_id=%s bucket=%s client=%s",
            wiki_id, self.bucket, type(self.client).__name__ if self.client else None,
        )
        # --- Primary: find newest wiki_manifest_*.json by modified date ---
        try:
            artifacts = self._list_bucket_artifacts(wiki_id)
            logger.info(
                "Listed %d artifacts with prefix=%s", len(artifacts), wiki_id,
            )
            manifest_files = [
                a for a in artifacts
                if isinstance(a.get("name"), str)
                and "wiki_manifest_" in a["name"]
                and a["name"].endswith(".json")
            ]
            if manifest_files:
                # Sort by modified date descending — pick latest
                manifest_files.sort(
                    key=lambda a: a.get("modified", ""),
                    reverse=True,
                )
                data = self.client.download_artifact(
                    self.bucket,
                    manifest_files[0]["name"],
                )
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                logger.debug("Loaded versioned manifest: %s", manifest_files[0]["name"])
                return json.loads(data)
        except Exception as e:
            logger.debug(f"Versioned manifest lookup failed for {wiki_id}: {e}")

        # --- Legacy fallback: {wiki_id}/manifest.json ---
        try:
            data = self.client.download_artifact(
                self.bucket,
                f"{wiki_id}/manifest.json",
            )
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            logger.debug("Loaded legacy manifest.json for %s", wiki_id)
            return json.loads(data)
        except Exception as e:
            logger.debug(f"No legacy manifest.json for {wiki_id}: {e}")

        logger.warning("No manifest found for %s", wiki_id)
        return None

    def _list_bucket_artifacts(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List artifacts in bucket, optionally filtered by prefix."""
        try:
            if hasattr(self.client, "list_artifacts"):
                return self.client.list_artifacts(self.bucket, prefix=prefix)
        except Exception as e:
            logger.debug(f"list_artifacts failed: {e}")
        return []

    def _download_index_file(self, wiki_id: str, filename: str) -> bool:
        """
        Download a single index file from bucket to local cache_dir.

        Tries multiple bucket paths:
        1. {wiki_id}/indexes/{filename} (new structured layout)
        2. {wiki_id}/{filename} (flat layout)
        """
        local_path = self.cache_dir / filename

        for bucket_path in [
            f"{wiki_id}/indexes/{filename}",
            f"{wiki_id}/{filename}",
        ]:
            try:
                data = self.client.download_artifact(self.bucket, bucket_path)
                local_path.parent.mkdir(parents=True, exist_ok=True)

                if isinstance(data, str):
                    local_path.write_text(data, encoding="utf-8")
                else:
                    local_path.write_bytes(data)

                logger.debug(f"Downloaded {bucket_path} → {local_path} ({local_path.stat().st_size:,} bytes)")
                return True
            except Exception:
                continue

        logger.warning(f"Could not download index file {filename} for {wiki_id} from any path")
        return False

    # ------------------------------------------------------------------
    # Private: cache index registration
    # ------------------------------------------------------------------

    def _register_manifest_in_cache_index(self, manifest: Dict[str, Any]) -> None:
        """
        Register manifest cache keys in local cache_index.json.

        This makes the downloaded indexes discoverable by VectorStoreManager,
        GraphManager, etc. (which all read cache_index.json).
        """
        from .repo_resolution import load_cache_index, save_cache_index_atomic, ensure_refs, repo_branch_key, split_repo_identifier

        idx = load_cache_index(self.cache_dir)

        # canonical_repo_identifier may already be commit-scoped (e.g. "owner/repo:branch:commit8").
        # Split it to extract the repo-only part so we don't double-append branch:commit.
        raw_id = manifest.get("canonical_repo_identifier", "")
        repo_only, branch_from_id, commit_from_id = split_repo_identifier(raw_id)

        # Prefer branch/commit from the manifest's dedicated fields, fall back to parsed parts.
        branch = manifest.get("branch") or branch_from_id or "main"
        commit = manifest.get("commit_hash", "")

        if not repo_only:
            logger.warning("Manifest missing canonical_repo_identifier — skipping cache_index registration")
            return

        # Build canonical identifier
        if commit:
            commit8 = commit[:8]
            canonical = f"{repo_only}:{branch}:{commit8}"
        else:
            canonical = f"{repo_only}:{branch}"

        # Register FAISS vectorstore key
        faiss_key = manifest.get("faiss_cache_key")
        if isinstance(faiss_key, str):
            idx[canonical] = faiss_key

        # Register graph key
        graph_key = manifest.get("graph_cache_key")
        if isinstance(graph_key, str):
            if "graphs" not in idx:
                idx["graphs"] = {}
            idx["graphs"][f"{canonical}:combined"] = graph_key

        # Register docstore key
        docstore_key = manifest.get("docstore_cache_key")
        if isinstance(docstore_key, str):
            if "docs" not in idx:
                idx["docs"] = {}
            idx["docs"][canonical] = docstore_key

        # Register BM25 key
        bm25_key = manifest.get("bm25_cache_key")
        if isinstance(bm25_key, str):
            if "bm25" not in idx:
                idx["bm25"] = {}
            idx["bm25"][canonical] = bm25_key

        # Register FTS5 key (companion to graph, uses same cache key)
        if isinstance(graph_key, str):
            if "fts5" not in idx:
                idx["fts5"] = {}
            idx["fts5"][canonical] = graph_key

        # Register unified DB key (.wiki.db)
        unified_db_key = manifest.get("unified_db_key")
        if isinstance(unified_db_key, str):
            if "unified_db" not in idx:
                idx["unified_db"] = {}
            idx["unified_db"][canonical] = unified_db_key

        # Register ref pointer: repo:branch → repo:branch:commit8
        if commit:
            refs = ensure_refs(idx)
            refs[repo_branch_key(repo_only, branch)] = canonical

        save_cache_index_atomic(self.cache_dir, idx)
        logger.info(f"Registered manifest indexes in cache_index.json: {canonical}")

    # ------------------------------------------------------------------
    # Bucket cleanup helpers
    # ------------------------------------------------------------------

    def _partition_bucket_artifacts(
        self, wiki_id: str
    ) -> Dict[str, Any]:
        """Partition all bucket artifacts for *wiki_id* by kind.

        Returns dict with keys: manifests, indexes, pages, legacy_manifest, other,
        and the raw list under 'all'.
        """
        result: Dict[str, Any] = {
            "manifests": [],
            "indexes": [],
            "pages": [],
            "legacy_manifest": None,
            "other": [],
            "all": [],
        }

        try:
            all_artifacts = self._list_bucket_artifacts(wiki_id)
        except Exception as exc:
            logger.warning("_partition_bucket_artifacts: list failed: %s", exc)
            return result

        result["all"] = all_artifacts
        wiki_prefix = f"{wiki_id}/"

        for a in all_artifacts:
            name = a.get("name", "") if isinstance(a, dict) else str(a)
            if not name.startswith(wiki_prefix):
                continue

            rel = name[len(wiki_prefix):]

            if "wiki_manifest_" in rel and rel.endswith(".json"):
                result["manifests"].append(a)
            elif rel == "manifest.json":
                result["legacy_manifest"] = a
            elif rel.startswith("indexes/"):
                result["indexes"].append(a)
            elif rel.startswith("wiki_pages/"):
                result["pages"].append(a)
            else:
                result["other"].append(a)

        return result

    # ------------------------------------------------------------------
    # Index cleanup: keep only the current generation's indexes
    # ------------------------------------------------------------------

    def cleanup_stale_indexes(
        self,
        wiki_id: str,
        current_manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Remove orphaned index files from the platform bucket.

        Compares index artifacts in ``{wiki_id}/indexes/`` against the
        filenames referenced by *current_manifest*.  Indexes that belong to
        the current generation are kept; everything else is deleted.

        If all current indexes are already present in the bucket (i.e. the
        indexes haven't changed), logs "indexes up to date" and returns
        without deleting anything.

        This is best-effort: failures are logged but do not propagate.

        Args:
            wiki_id:          Wiki identifier (e.g. "fmtlib--fmt--master").
            current_manifest: The manifest dict for the generation that was
                              just completed.

        Returns:
            Summary dict with counts.
        """
        stats: Dict[str, Any] = {
            "wiki_id": wiki_id,
            "indexes_in_bucket": 0,
            "indexes_current": 0,
            "indexes_deleted": 0,
            "indexes_up_to_date": False,
            "errors": [],
        }

        if not self.client:
            logger.debug("cleanup_stale_indexes: no client — skipping")
            return stats

        parts = self._partition_bucket_artifacts(wiki_id)
        index_arts = parts["indexes"]
        stats["indexes_in_bucket"] = len(index_arts)

        # Build the set of index basenames for the current generation
        current_basenames: Set[str] = set()
        for fn in manifest_index_files(current_manifest):
            current_basenames.add(fn)
        stats["indexes_current"] = len(current_basenames)

        if not index_arts:
            logger.debug(
                "cleanup_stale_indexes(%s): no index artifacts in bucket", wiki_id
            )
            return stats

        # Classify bucket indexes as current or orphaned
        orphaned: List[Dict[str, Any]] = []
        matched = 0
        for ia in index_arts:
            ia_name = ia.get("name", "") if isinstance(ia, dict) else str(ia)
            basename = ia_name.split("/")[-1] if "/" in ia_name else ia_name
            if basename in current_basenames:
                matched += 1
            else:
                orphaned.append(ia)

        if not orphaned:
            stats["indexes_up_to_date"] = True
            logger.info(
                "cleanup_stale_indexes(%s): indexes are up to date "
                "(%d files match current generation, 0 orphaned)",
                wiki_id, matched,
            )
            return stats

        logger.info(
            "cleanup_stale_indexes(%s): %d current, %d orphaned — deleting orphaned indexes",
            wiki_id, matched, len(orphaned),
        )

        for ia in orphaned:
            ia_name = ia.get("name", "") if isinstance(ia, dict) else str(ia)
            try:
                self.client.delete_artifact(self.bucket, ia_name)
                stats["indexes_deleted"] += 1
            except Exception as exc:
                stats["errors"].append(f"delete {ia_name}: {exc}")

        logger.info(
            "cleanup_stale_indexes(%s): deleted %d orphaned indexes (%d errors)",
            wiki_id, stats["indexes_deleted"], len(stats["errors"]),
        )
        return stats

    # ------------------------------------------------------------------
    # Content version retention: keep N most recent manifest+page sets
    # ------------------------------------------------------------------

    # How many manifest versions (with their pages) to retain.
    # The platform uploads the new manifest *after* the worker finishes,
    # so we reserve one extra slot.  With CONTENT_VERSIONS_TO_KEEP = 4
    # the bucket holds at most 5 versions (4 kept + 1 incoming).
    CONTENT_VERSIONS_TO_KEEP = 4

    def cleanup_old_content_versions(
        self,
        wiki_id: str,
        keep_versions: int = CONTENT_VERSIONS_TO_KEEP,
        current_manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retain the *keep_versions* most recent manifest+page sets and delete
        the rest.

        In Jobs+API mode the worker uploads content artifacts (pages, manifest,
        structure JSON) directly to the bucket *before* calling this method.
        So the current generation's versioned manifest is already present in
        the bucket by the time cleanup runs.  The *current_manifest* parameter
        provides an additional safety net: its pages are always treated as live,
        regardless of whether the versioned JSON has been listed yet.

        With ``keep_versions=4`` the bucket holds up to 4 manifest+page sets
        after cleanup.

        Strategy:
        1. List ``{wiki_id}/wiki_manifest_*.json`` — sort newest-first.
        2. Keep the *keep_versions* most recent manifests.
        3. Build a "live pages" set from kept manifests + *current_manifest*.
        4. Delete excess manifests.
        5. Delete wiki pages not referenced by any live manifest.
        6. Delete legacy ``{wiki_id}/manifest.json`` if present.

        This is best-effort: failures are logged but do not propagate.

        Args:
            wiki_id:          Wiki identifier.
            keep_versions:    Number of previous versions to retain (default 4).
            current_manifest: Manifest for the generation just completed.  Its
                              pages are always treated as live even if the
                              versioned JSON hasn't been listed from the bucket
                              yet (eventual consistency).

        Returns:
            Summary dict with counts.
        """
        stats: Dict[str, Any] = {
            "wiki_id": wiki_id,
            "manifests_found": 0,
            "manifests_kept": 0,
            "manifests_deleted": 0,
            "pages_deleted": 0,
            "other_deleted": 0,
            "errors": [],
        }

        if not self.client:
            logger.debug("cleanup_old_content_versions: no client — skipping")
            return stats

        parts = self._partition_bucket_artifacts(wiki_id)
        manifest_arts = parts["manifests"]
        page_arts = parts["pages"]
        legacy_manifest = parts["legacy_manifest"]

        stats["manifests_found"] = len(manifest_arts)

        if not manifest_arts:
            logger.debug(
                "cleanup_old_content_versions(%s): no versioned manifests — nothing to clean",
                wiki_id,
            )
            return stats

        # -- 1. Sort manifests newest-first, split into keep / excess --
        manifest_arts.sort(
            key=lambda a: a.get("modified", "") if isinstance(a, dict) else "",
            reverse=True,
        )
        keep = manifest_arts[:keep_versions]
        excess = manifest_arts[keep_versions:]
        stats["manifests_kept"] = len(keep)

        if not excess and not legacy_manifest:
            logger.info(
                "cleanup_old_content_versions(%s): %d manifests found, all within "
                "retention limit (%d) — nothing to clean",
                wiki_id, len(manifest_arts), keep_versions,
            )
            return stats

        # -- 2. Build "live pages" set from kept manifests + current gen --
        live_pages: Set[str] = set()

        # Pages from the current generation (not yet in bucket as manifest).
        if current_manifest:
            for p in current_manifest.get("pages", []):
                if isinstance(p, str):
                    live_pages.add(p)

        for ma in keep:
            ma_name = ma.get("name", "") if isinstance(ma, dict) else str(ma)
            try:
                raw = self.client.download_artifact(self.bucket, ma_name)
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                m = json.loads(raw)
                for p in m.get("pages", []):
                    if isinstance(p, str):
                        live_pages.add(p)
            except Exception as exc:
                logger.debug("Could not parse manifest %s: %s", ma_name, exc)
                stats["errors"].append(f"parse {ma_name}: {exc}")

        # -- 3. Delete excess manifests --
        for ma in excess:
            ma_name = ma.get("name", "") if isinstance(ma, dict) else str(ma)
            try:
                self.client.delete_artifact(self.bucket, ma_name)
                stats["manifests_deleted"] += 1
                logger.debug("Deleted excess manifest: %s", ma_name)
            except Exception as exc:
                stats["errors"].append(f"delete {ma_name}: {exc}")

        # -- 4. Delete wiki pages not in any live manifest --
        if live_pages and page_arts:
            orphaned_pages = [
                pa for pa in page_arts
                if (pa.get("name", "") if isinstance(pa, dict) else str(pa))
                not in live_pages
            ]
            if orphaned_pages:
                logger.info(
                    "cleanup_old_content_versions(%s): deleting %d orphaned pages "
                    "(keeping %d live pages across %d manifests)",
                    wiki_id, len(orphaned_pages), len(live_pages), len(keep),
                )
                for pa in orphaned_pages:
                    pa_name = pa.get("name", "") if isinstance(pa, dict) else str(pa)
                    try:
                        self.client.delete_artifact(self.bucket, pa_name)
                        stats["pages_deleted"] += 1
                    except Exception as exc:
                        stats["errors"].append(f"delete {pa_name}: {exc}")

        # -- 5. Delete legacy manifest.json (superseded by versioned) --
        if legacy_manifest:
            lm_name = (
                legacy_manifest.get("name", "")
                if isinstance(legacy_manifest, dict)
                else str(legacy_manifest)
            )
            try:
                self.client.delete_artifact(self.bucket, lm_name)
                stats["other_deleted"] += 1
                logger.debug("Deleted legacy manifest: %s", lm_name)
            except Exception:
                pass

        total_deleted = (
            stats["manifests_deleted"]
            + stats["pages_deleted"]
            + stats["other_deleted"]
        )
        if total_deleted:
            logger.info(
                "cleanup_old_content_versions(%s): deleted %d manifests, "
                "%d pages, %d other (%d errors)",
                wiki_id,
                stats["manifests_deleted"],
                stats["pages_deleted"],
                stats["other_deleted"],
                len(stats["errors"]),
            )
        else:
            logger.info(
                "cleanup_old_content_versions(%s): nothing to delete", wiki_id
            )

        return stats

    # ------------------------------------------------------------------
    # Private: eviction
    # ------------------------------------------------------------------

    def _evict_old_repos(self) -> None:
        """
        "Keep latest only" eviction: remove old repo indexes when cache exceeds limit.

        Eviction strategy:
        - Read cache_index.json to find all repo keys
        - Keep the N most recently referenced repos
        - Delete index files for evicted repos
        """
        from .repo_resolution import load_cache_index, save_cache_index_atomic, _RESERVED_TOP_LEVEL_KEYS

        idx = load_cache_index(self.cache_dir)

        # Collect all repo keys (non-reserved top-level keys with string values)
        repo_keys = [k for k in idx if k not in _RESERVED_TOP_LEVEL_KEYS and isinstance(idx[k], str)]

        if len(repo_keys) <= self.max_cached_repos:
            return  # Within limits

        # Sort by file modification time of the FAISS index (proxy for recency)
        def _faiss_mtime(key: str) -> float:
            faiss_hash = idx.get(key, "")
            if isinstance(faiss_hash, str):
                faiss_path = self.cache_dir / f"{faiss_hash}.faiss"
                if faiss_path.exists():
                    return faiss_path.stat().st_mtime
            return 0.0

        repo_keys.sort(key=_faiss_mtime, reverse=True)

        # Keep top N, evict the rest
        to_evict = repo_keys[self.max_cached_repos:]
        if not to_evict:
            return

        logger.info(f"Evicting {len(to_evict)} old repo indexes (keeping {self.max_cached_repos})")

        evicted_hashes: Set[str] = set()
        graphs_idx = idx.get("graphs", {})
        docs_idx = idx.get("docs", {})
        bm25_idx = idx.get("bm25", {})

        for key in to_evict:
            # Collect all hash values for this repo
            faiss_hash = idx.pop(key, None)
            if isinstance(faiss_hash, str):
                evicted_hashes.add(faiss_hash)

            graph_hash = graphs_idx.pop(f"{key}:combined", None)
            if isinstance(graph_hash, str):
                evicted_hashes.add(graph_hash)

            doc_hash = docs_idx.pop(key, None)
            if isinstance(doc_hash, str):
                evicted_hashes.add(doc_hash)

            bm25_hash = bm25_idx.pop(key, None)
            if isinstance(bm25_hash, str):
                evicted_hashes.add(bm25_hash)

            logger.debug(f"Evicted repo from cache_index: {key}")

        # Delete files matching evicted hashes
        deleted_count = 0
        deleted_bytes = 0
        for h in evicted_hashes:
            for ext in [".faiss", ".code_graph.gz", ".docstore.bin", ".doc_index.json",
                        ".bm25.sqlite", ".fts5.sqlite", ".docs.pkl", "_analysis.json"]:
                fpath = self.cache_dir / f"{h}{ext}"
                if fpath.exists():
                    try:
                        sz = fpath.stat().st_size
                        fpath.unlink()
                        deleted_count += 1
                        deleted_bytes += sz
                    except Exception as e:
                        logger.warning(f"Failed to delete {fpath}: {e}")

        save_cache_index_atomic(self.cache_dir, idx)
        logger.info(
            f"Eviction complete: removed {deleted_count} files "
            f"({deleted_bytes / (1024 * 1024):.1f} MB) for {len(to_evict)} repos"
        )
