"""
Platform Artifact Client for DeepWiki artifact transport.

Uses the ELITEA platform's artifact HTTP API (the same API that invoke.py's
MiniArtifactClient and elitea-sdk's EliteAClient use) instead of direct S3/boto3
connections.

In ELITEA, all artifact storage goes through the platform's artifact endpoints:
    - Upload:   POST  /api/v2/artifacts/artifacts/default/{project_id}/{bucket}
    - Download: GET   /api/v2/artifacts/artifact/default/{project_id}/{bucket}/{name}
    - List:     GET   /api/v2/artifacts/artifacts/default/{project_id}/{bucket}
    - Delete:   DELETE /api/v2/artifacts/artifact/default/{project_id}/{bucket}/{name}

The platform internally routes these to Minio/S3 with proper bucket prefixing,
access control, and project scoping.

Credential flow:
    1. Controller receives ``llm_settings`` from the platform on each invocation.
    2. ``extract_artifact_settings(llm_settings)`` derives base_url, api_key, project_id.
    3. For K8s Jobs, the controller injects artifact settings as env vars into the
       worker pod: DEEPWIKI_ARTIFACT_BASE_URL, DEEPWIKI_ARTIFACT_API_KEY,
       DEEPWIKI_ARTIFACT_PROJECT_ID.
    4. Worker reads those env vars to build its own PlatformArtifactClient.

Bucket key layout (inside the "wiki-artifacts" bucket):
    _jobs/{job_id}/input.json                  Job input  (controller -> worker)
    _jobs/{job_id}/result.json                 Job result (worker -> controller)
    {wiki_id}/indexes/{file}                   Index files (worker uploads)
    {wiki_id}/wiki_manifest_{version}.json     Versioned manifest (worker uploads)
    {wiki_id}/wiki_pages/{section}/{page}.md   Wiki pages (worker uploads)
    {wiki_id}/analysis/wiki_structure_*.json   Structure analysis (worker uploads)
    _registry/wikis.json                       Global wiki registry (controller uploads)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable names for artifact API passthrough to worker pods
# ---------------------------------------------------------------------------

ARTIFACT_BASE_URL_ENV = "DEEPWIKI_ARTIFACT_BASE_URL"
ARTIFACT_API_KEY_ENV = "DEEPWIKI_ARTIFACT_API_KEY"
ARTIFACT_PROJECT_ID_ENV = "DEEPWIKI_ARTIFACT_PROJECT_ID"
ARTIFACT_BUCKET_ENV = "DEEPWIKI_ARTIFACT_BUCKET"
ARTIFACT_X_SECRET_ENV = "DEEPWIKI_ARTIFACT_X_SECRET"
ARTIFACT_COOKIE_ENV = "DEEPWIKI_ARTIFACT_COOKIE"

# api_key value the runtime sends when it authenticates with the user's
# session cookie instead of a token (the cookie rides in default_headers).
SESSION_API_KEY_PLACEHOLDER = "session"

# All env var names that should be passed to worker Job pods
ARTIFACT_ENV_VARS = [
    ARTIFACT_BASE_URL_ENV,
    ARTIFACT_API_KEY_ENV,
    ARTIFACT_PROJECT_ID_ENV,
    ARTIFACT_BUCKET_ENV,
    ARTIFACT_X_SECRET_ENV,
    ARTIFACT_COOKIE_ENV,
]

DEFAULT_BUCKET = "wiki-artifacts"
DEFAULT_API_PATH = "/api/v2"


# ---------------------------------------------------------------------------
# Chunked artifact transport (#6991)
#
# The platform rejects multipart upload bodies over 150 MB (HTTP 413). Index
# files like {key}.wiki.db / {key}.fts5.db can far exceed that, so large
# uploads/downloads are transparently split into parts here — the platform
# itself is never changed.
# ---------------------------------------------------------------------------

ARTIFACT_PART_SIZE_MB_ENV = "DEEPWIKI_ARTIFACT_PART_SIZE_MB"
DEFAULT_ARTIFACT_PART_SIZE_MB = 64
# Must stay comfortably under the platform's 150 MB request-body limit.
MAX_ARTIFACT_PART_SIZE_MB = 140

CHUNK_MANIFEST_SUFFIX = ".chunks.json"

# Matches the ".chunk-<12 hex>-<5 digits>" suffix appended to a logical
# artifact name for each uploaded part (generation = uuid.uuid4().hex[:12]).
_CHUNK_PART_RE = re.compile(r"\.chunk-[0-9a-f]{12}-\d{5}$")


def get_artifact_part_size() -> int:
    """
    Resolve the configured chunk size (bytes) for chunked artifact transport.

    Reads ``DEEPWIKI_ARTIFACT_PART_SIZE_MB`` (an integer, in MB). Missing,
    invalid or non-positive values fall back to the default with a warning.
    The result is clamped to ``[1, MAX_ARTIFACT_PART_SIZE_MB]`` MB.
    """
    raw = os.environ.get(ARTIFACT_PART_SIZE_MB_ENV, "").strip()
    mb = DEFAULT_ARTIFACT_PART_SIZE_MB
    if raw:
        try:
            parsed = int(raw)
            if parsed <= 0:
                raise ValueError(f"must be positive, got {parsed}")
            mb = parsed
        except ValueError as exc:
            logger.warning(
                "Invalid %s=%r (%s) — falling back to default %d MB",
                ARTIFACT_PART_SIZE_MB_ENV, raw, exc, DEFAULT_ARTIFACT_PART_SIZE_MB,
            )
            mb = DEFAULT_ARTIFACT_PART_SIZE_MB

    mb = max(1, min(mb, MAX_ARTIFACT_PART_SIZE_MB))
    return mb * 1024 * 1024


def logical_artifact_name(name: str) -> str:
    """
    Strip a trailing chunk-part suffix or the chunk-manifest suffix from *name*.

    Returns *name* unchanged if it isn't a chunked-artifact name. Lets
    callers (e.g. stale-index cleanup) classify chunk parts/manifests by the
    logical artifact they belong to.
    """
    if name.endswith(CHUNK_MANIFEST_SUFFIX):
        return name[: -len(CHUNK_MANIFEST_SUFFIX)]
    match = _CHUNK_PART_RE.search(name)
    if match:
        return name[: match.start()]
    return name


class ArtifactHTTPError(RuntimeError):
    """
    RuntimeError raised by ``_post_object``/``_get_object`` on an HTTP error,
    carrying the response ``status_code``.

    Subclasses RuntimeError so existing ``except RuntimeError`` callers are
    unaffected; part-upload retry logic inspects ``.status_code`` to tell a
    retryable 5xx from a non-retryable 4xx.
    """

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Settings extraction (same logic as invoke.py's extract_artifact_settings)
# ---------------------------------------------------------------------------

def extract_artifact_settings(llm_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract artifact API settings from ``llm_settings``.

    ``llm_settings`` is the dict provided by the platform on each invocation,
    containing ``api_base``, ``api_key``, and ``organization`` (project_id).

    Strips LLM-specific suffixes from ``api_base`` to get the platform base URL.
    """
    openai_api_base = llm_settings.get("api_base", "") or llm_settings.get("openai_api_base", "")
    openai_api_key = llm_settings.get("api_key", "") or llm_settings.get("openai_api_key", "")
    # project_id lives under several possible keys depending on model type:
    #   "organization"        — OpenAI models (mapped from openai_organization)
    #   "openai_organization" — legacy / direct field name
    #   "project_id"          — always added by provider_worker from EliteAClient
    project_id = (
        llm_settings.get("organization")
        or llm_settings.get("openai_organization")
        or llm_settings.get("project_id")
        or ""
    )
    if not project_id:
        logger.warning(
            "extract_artifact_settings: project_id is empty! "
            "llm_settings keys=%s",
            list(llm_settings.keys()),
        )

    # Strip LLM-related suffixes:
    #   /llm/v1, /llm/v2, /llm/api/v1, /llm (Anthropic)
    base_url = re.sub(r'/llm(/api)?(/v\d+)?/?$', '', openai_api_base)

    return {
        "base_url": base_url,
        "api_key": openai_api_key,
        "project_id": str(project_id),
        "api_path": DEFAULT_API_PATH,
        "x_secret": llm_settings.get("x_secret", "secret"),
        "cookie": _session_cookie(llm_settings),
    }


def _session_cookie(llm_settings: Dict[str, Any]) -> str:
    """Return the session ``Cookie`` header the runtime sends to the platform."""
    headers = llm_settings.get("default_headers")
    if not isinstance(headers, dict):
        return ""
    for name, value in headers.items():
        if str(name).lower() == "cookie" and value:
            return str(value)
    return ""


def extract_artifact_settings_from_env() -> Optional[Dict[str, Any]]:
    """
    Reconstruct artifact settings from worker pod environment variables.

    This is the counterpart of ``inject_artifact_env_vars``: the controller
    writes settings into env vars, the worker reads them back.

    Returns None if the required env vars are not set.
    """
    base_url = os.environ.get(ARTIFACT_BASE_URL_ENV, "").strip()
    api_key = os.environ.get(ARTIFACT_API_KEY_ENV, "").strip()
    project_id = os.environ.get(ARTIFACT_PROJECT_ID_ENV, "").strip()

    if not base_url or not api_key:
        return None

    return {
        "base_url": base_url,
        "api_key": api_key,
        "project_id": project_id,
        "api_path": DEFAULT_API_PATH,
        "x_secret": os.environ.get(ARTIFACT_X_SECRET_ENV, "secret"),
        "cookie": os.environ.get(ARTIFACT_COOKIE_ENV, "").strip(),
    }


def inject_artifact_env_vars(
    artifact_settings: Dict[str, Any],
    bucket: str = DEFAULT_BUCKET,
) -> Dict[str, str]:
    """
    Build a dict of env var name → value for injecting into a worker pod spec.

    The controller calls this before creating a Job to make the artifact
    settings available to the worker via environment variables.
    """
    return {
        ARTIFACT_BASE_URL_ENV: artifact_settings.get("base_url", ""),
        ARTIFACT_API_KEY_ENV: artifact_settings.get("api_key", ""),
        ARTIFACT_PROJECT_ID_ENV: artifact_settings.get("project_id", ""),
        ARTIFACT_BUCKET_ENV: bucket,
        ARTIFACT_X_SECRET_ENV: artifact_settings.get("x_secret", "secret"),
        ARTIFACT_COOKIE_ENV: artifact_settings.get("cookie", ""),
    }


def is_artifact_transport_configured() -> bool:
    """
    Check if platform artifact transport is configured via environment.

    Returns True if the DEEPWIKI_ARTIFACT_BASE_URL env var is set,
    indicating that a controller has injected artifact credentials
    for this worker pod.
    """
    return bool(os.environ.get(ARTIFACT_BASE_URL_ENV, "").strip())


def get_artifact_bucket() -> str:
    """Get the artifact bucket name from env or default."""
    return os.environ.get(ARTIFACT_BUCKET_ENV, DEFAULT_BUCKET).strip()


# ---------------------------------------------------------------------------
# Platform Artifact Client
# ---------------------------------------------------------------------------

def artifact_auth_headers(api_key: str, x_secret: str, cookie: str = "") -> Dict[str, str]:
    """
    Auth headers for platform artifact calls.

    Session-auth runtimes pass a placeholder api_key and the user's session
    cookie; the placeholder is not a token, so it is not sent as one.
    """
    headers = {"X-SECRET": x_secret}
    if api_key and api_key != SESSION_API_KEY_PLACEHOLDER:
        headers["Authorization"] = f"Bearer {api_key}"
    if cookie:
        headers["Cookie"] = cookie
    return headers


class PlatformArtifactClient:
    """
    HTTP client for the ELITEA platform artifact API.

    Uses the same endpoints as elitea-sdk's ``EliteAClient`` and deepwiki's
    ``MiniArtifactClient`` in invoke.py.

    All requests authenticated via ``Authorization: Bearer {api_key}``
    and ``X-SECRET`` header.
    """

    def __init__(self, artifact_settings: Dict[str, Any]):
        """
        Args:
            artifact_settings: Dict with base_url, api_key, project_id, api_path, x_secret.
                              Typically from ``extract_artifact_settings()`` or env vars.
        """
        self.base_url = artifact_settings.get("base_url", "")
        self.api_key = artifact_settings.get("api_key", "")
        self.project_id = artifact_settings.get("project_id", "")
        self.api_path = artifact_settings.get("api_path", DEFAULT_API_PATH)
        self.x_secret = artifact_settings.get("x_secret", "secret")
        self.cookie = artifact_settings.get("cookie", "")
        self.part_size = get_artifact_part_size()

        if not self.base_url:
            raise ValueError("artifact_settings.base_url is required")
        if not self.api_key:
            raise ValueError("artifact_settings.api_key is required")

    def _headers(self) -> Dict[str, str]:
        return artifact_auth_headers(self.api_key, self.x_secret, self.cookie)

    def _artifact_url(self, bucket: str) -> str:
        """URL base for artifact operations (upload / list)."""
        bucket_segment = quote(str(bucket).lower(), safe="")
        return (
            f"{self.base_url}{self.api_path}/artifacts/artifacts"
            f"/default/{self.project_id}/{bucket_segment}"
        )

    def _single_artifact_url(self, bucket: str, name: str) -> str:
        """URL for a specific artifact (download / delete)."""
        bucket_segment = quote(str(bucket).lower(), safe="")
        name_segment = quote(str(name), safe="/")
        return (
            f"{self.base_url}{self.api_path}/artifacts/artifact"
            f"/default/{self.project_id}/{bucket_segment}/{name_segment}"
        )

    # ------------------------------------------------------------------
    # Public API (matches the interface expected by ArtifactManager)
    # ------------------------------------------------------------------

    def _post_object(self, bucket: str, name: str, data: bytes) -> Dict[str, Any]:
        """
        POST raw bytes as a single multipart object to the bucket.

        Same behavior/logging/errors as the original single-shot upload —
        callers (``upload_artifact``, ``_post_part``) decide whether the
        body needs to be chunked first.
        """
        import requests

        url = self._artifact_url(bucket)
        logger.info(
            "Uploading artifact: %s to bucket %s (url=%s, project=%s)",
            name, bucket, url, self.project_id,
        )

        files = {"file": (name, data)}
        response = requests.post(
            url, headers=self._headers(), files=files, verify=False, timeout=300
        )

        if response.status_code == 403:
            logger.error(
                "Artifact upload 403 Forbidden — url=%s, project=%s, "
                "key_prefix=%s, response=%s",
                url, self.project_id,
                self.api_key[:8] + "..." if self.api_key else "(empty)",
                response.text[:500],
            )
            raise ArtifactHTTPError(
                f"Not authorized to upload artifact (HTTP 403): {response.text[:200]}",
                403,
            )
        if response.status_code not in (200, 201):
            logger.error(
                "Artifact upload failed — url=%s, status=%d, response=%s",
                url, response.status_code, response.text[:500],
            )
            raise ArtifactHTTPError(
                f"Failed to upload artifact: HTTP {response.status_code} — {response.text[:500]}",
                response.status_code,
            )

        logger.debug("Uploaded artifact: %s/%s (%d bytes)", bucket, name, len(data))
        try:
            return response.json()
        except Exception:
            return {"status": "ok"}

    def _post_part(self, bucket: str, name: str, data: bytes) -> Dict[str, Any]:
        """
        Upload one chunk part via ``_post_object``, retrying transient failures.

        Up to 3 attempts total, retrying on ``requests.ConnectionError`` /
        ``requests.Timeout`` or a 5xx ``ArtifactHTTPError``, with a 1s then
        2s backoff between attempts. 403/413/other 4xx errors are never
        retried — they're a durable rejection, not a transient blip.
        """
        import requests

        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                return self._post_object(bucket, name, data)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
            except ArtifactHTTPError as exc:
                if 500 <= exc.status_code < 600:
                    last_exc = exc
                else:
                    raise

            if attempt < 3:
                logger.warning(
                    "Chunk part upload failed (attempt %d/3) for %s/%s: %s — retrying in %ds",
                    attempt, bucket, name, last_exc, attempt,
                )
                time.sleep(attempt)

        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _iter_bytes_parts(data: bytes, part_size: int) -> Iterable[bytes]:
        """Slice in-memory *data* into ``part_size`` chunks via a memoryview."""
        mv = memoryview(data)
        for i in range(0, len(mv), part_size):
            yield bytes(mv[i:i + part_size])

    @staticmethod
    def _iter_file_parts(path: Path, part_size: int) -> Iterable[bytes]:
        """Read *path* ``part_size`` bytes at a time (bounded memory use)."""
        with open(path, "rb") as f:
            while True:
                part = f.read(part_size)
                if not part:
                    break
                yield part

    def _upload_chunked(
        self,
        bucket: str,
        name: str,
        total_size: int,
        parts_iter: Iterable[bytes],
    ) -> Dict[str, Any]:
        """
        Upload *parts_iter* as numbered chunk parts, then commit a manifest.

        The manifest is uploaded LAST — it is the commit marker. If a part
        upload fails, this raises without writing a manifest, so any
        previous manifest (and its parts) stays intact and still readable.
        Already-uploaded new parts are deliberately left in place (not
        rolled back): orphans like these are cleaned up by wiki deletion or
        the next successful overwrite, which keeps this path simple.
        """
        t0 = time.monotonic()

        # Best-effort: a corrupt previous manifest must not block overwriting
        # it with fresh data, so we only lose the old-generation part names
        # (-> their cleanup below is skipped, not the upload itself).
        try:
            previous_manifest = self._get_chunk_manifest(bucket, name)
        except Exception as exc:
            logger.warning(
                "Could not read previous chunk manifest for %s/%s before overwrite: %s",
                bucket, name, exc,
            )
            previous_manifest = None
        old_part_names = (
            {p["name"] for p in previous_manifest["parts"]}
            if previous_manifest else set()
        )

        generation = uuid.uuid4().hex[:12]
        parts_meta: List[Dict[str, Any]] = []
        whole_sha256 = hashlib.sha256()
        uploaded_size = 0

        for index, part_bytes in enumerate(parts_iter):
            part_key = f"{name}.chunk-{generation}-{index:05d}"
            self._post_part(bucket, part_key, part_bytes)
            whole_sha256.update(part_bytes)
            uploaded_size += len(part_bytes)
            parts_meta.append({
                "name": part_key,
                "size": len(part_bytes),
                "sha256": hashlib.sha256(part_bytes).hexdigest(),
            })

        if uploaded_size != total_size:
            raise RuntimeError(
                f"Chunked upload size mismatch for {bucket}/{name}: "
                f"uploaded {uploaded_size} bytes, expected {total_size}"
            )

        manifest = {
            "format": "deepwiki-chunked-artifact",
            "version": 1,
            "name": name,
            "size": uploaded_size,
            "sha256": whole_sha256.hexdigest(),
            "part_size": self.part_size,
            "generation": generation,
            "parts": parts_meta,
        }
        self._post_object(
            bucket, name + CHUNK_MANIFEST_SUFFIX, json.dumps(manifest).encode("utf-8")
        )

        # Commit succeeded. Remove any stale single object of the same name:
        # reads try the single object first, so a leftover one would shadow
        # the new chunked data.
        try:
            self.delete_artifact(bucket, name)
        except Exception as exc:
            raise RuntimeError(
                f"Chunked upload for {bucket}/{name} committed but failed to delete "
                f"the stale single object: {exc}"
            ) from exc

        # Best-effort: drop old-generation parts no longer referenced. Never
        # raise — a failed cleanup here must not fail the upload.
        new_part_names = {p["name"] for p in parts_meta}
        for old_name in old_part_names - new_part_names:
            try:
                self.delete_artifact(bucket, old_name)
            except Exception as exc:
                logger.warning(
                    "Failed to delete stale chunk part %s/%s: %s", bucket, old_name, exc
                )

        elapsed = time.monotonic() - t0
        logger.info(
            "Chunked upload complete: %s (%.1f MB, %d parts) in %.1fs",
            name, uploaded_size / (1024 * 1024), len(parts_meta), elapsed,
        )
        return {
            "status": "ok",
            "chunked": True,
            "parts": len(parts_meta),
            "size": uploaded_size,
        }

    def upload_artifact(
        self, bucket: str, name: str, data: bytes | str
    ) -> Dict[str, Any]:
        """
        Upload a file to the platform bucket.

        Transparently splits bodies over ``self.part_size`` into chunk parts
        plus a commit manifest, to stay under the platform's request-body
        limit. Most uploads are small, so the common case is unchanged: one
        request, zero extra round trips.

        Args:
            bucket: Bucket name.
            name: File key/path (may contain '/' for nested paths).
            data: File content (bytes or str).

        Returns:
            Platform response dict.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        if len(data) <= self.part_size:
            return self._post_object(bucket, name, data)

        return self._upload_chunked(
            bucket, name, len(data), self._iter_bytes_parts(data, self.part_size)
        )

    def upload_file(self, bucket: str, name: str, path: str | Path) -> Dict[str, Any]:
        """
        Upload the file at *path* to the platform bucket.

        For files <= ``self.part_size`` this is just ``upload_artifact`` on
        the file's bytes. Larger files are streamed part-by-part straight
        from disk (``part_size`` bytes at a time), so memory use is bounded
        to one part regardless of file size.
        """
        file_path = Path(path)
        size = file_path.stat().st_size

        if size <= self.part_size:
            return self.upload_artifact(bucket, name, file_path.read_bytes())

        return self._upload_chunked(
            bucket, name, size, self._iter_file_parts(file_path, self.part_size)
        )

    def _get_object(self, bucket: str, name: str, stream: bool = False):
        """
        GET a single object from the bucket, returning the raw ``requests.Response``.

        Keeps the original 403 logging/raising behavior. Callers decide how
        to handle 200/400/404/other statuses (single object vs. chunk
        fallback).
        """
        import requests

        url = self._single_artifact_url(bucket, name)
        logger.debug("Fetching artifact: %s/%s (stream=%s)", bucket, name, stream)

        response = requests.get(
            url, headers=self._headers(), verify=False, timeout=300, stream=stream
        )

        if response.status_code == 403:
            logger.error(
                "Artifact download 403 Forbidden — url=%s, project=%s, "
                "key_prefix=%s, response=%s",
                url, self.project_id,
                self.api_key[:8] + "..." if self.api_key else "(empty)",
                response.text[:500],
            )
            raise RuntimeError(
                f"Not authorized to access artifact (HTTP 403): {response.text[:200]}"
            )
        return response

    def _get_chunk_manifest(self, bucket: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch and validate the chunk manifest for *name*, if any.

        Returns None when no manifest exists (non-200 GET) — that's the
        normal case for every artifact that was never chunked. Raises
        RuntimeError if a manifest exists but is structurally invalid,
        since that indicates corruption rather than absence.
        """
        response = self._get_object(bucket, name + CHUNK_MANIFEST_SUFFIX)
        if response.status_code != 200:
            return None

        try:
            manifest = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"Corrupt chunk manifest for {bucket}/{name}: {exc}"
            ) from exc

        if (
            not isinstance(manifest, dict)
            or manifest.get("format") != "deepwiki-chunked-artifact"
            or manifest.get("version") != 1
            or not isinstance(manifest.get("size"), int)
            or not isinstance(manifest.get("parts"), list)
            or not manifest["parts"]
        ):
            raise RuntimeError(
                f"Corrupt chunk manifest for {bucket}/{name}: unexpected structure"
            )
        for part in manifest["parts"]:
            if (
                not isinstance(part, dict)
                or not isinstance(part.get("name"), str)
                or not isinstance(part.get("size"), int)
                or not isinstance(part.get("sha256"), str)
            ):
                raise RuntimeError(
                    f"Corrupt chunk manifest for {bucket}/{name}: invalid part entry"
                )

        return manifest

    def _download_chunked_bytes(
        self, bucket: str, name: str, manifest: Dict[str, Any]
    ) -> bytes:
        """Download every part in *manifest* into memory, verifying integrity."""
        buf = bytearray()
        whole_sha256 = hashlib.sha256()

        for part in manifest["parts"]:
            part_name = part["name"]
            response = self._get_object(bucket, part_name)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to download chunk part {bucket}/{part_name}: "
                    f"HTTP {response.status_code} — {response.text[:300]}"
                )
            part_bytes = response.content
            if len(part_bytes) != part["size"]:
                raise RuntimeError(
                    f"Chunk part size mismatch for {bucket}/{part_name}: "
                    f"got {len(part_bytes)}, expected {part['size']}"
                )
            if hashlib.sha256(part_bytes).hexdigest() != part["sha256"]:
                raise RuntimeError(f"Chunk part checksum mismatch for {bucket}/{part_name}")
            whole_sha256.update(part_bytes)
            buf.extend(part_bytes)

        if len(buf) != manifest["size"]:
            raise RuntimeError(
                f"Chunked artifact size mismatch for {bucket}/{name}: "
                f"got {len(buf)}, expected {manifest['size']}"
            )
        if whole_sha256.hexdigest() != manifest["sha256"]:
            raise RuntimeError(f"Chunked artifact checksum mismatch for {bucket}/{name}")

        return bytes(buf)

    def download_artifact(self, bucket: str, name: str) -> bytes:
        """
        Download a file from the platform bucket.

        Falls back to chunked reassembly when the single object is missing
        (HTTP 400 or 404 — the platform uses both for "not found") and a
        valid chunk manifest exists.

        Args:
            bucket: Bucket name.
            name: File key/path.

        Returns:
            Raw file content as bytes.
        """
        response = self._get_object(bucket, name)

        if response.status_code == 200:
            content = response.content
            logger.debug(
                "Downloaded artifact: %s/%s (%d bytes)", bucket, name, len(content)
            )
            return content

        if response.status_code in (400, 404):
            manifest = self._get_chunk_manifest(bucket, name)
            if manifest is not None:
                return self._download_chunked_bytes(bucket, name, manifest)
            if response.status_code == 404:
                raise RuntimeError(f"Artifact not found: {bucket}/{name}")

        raise RuntimeError(
            f"Failed to download artifact: HTTP {response.status_code} — {response.text[:300]}"
        )

    def _download_chunked_to_file(
        self, bucket: str, name: str, manifest: Dict[str, Any], tmp_path: Path
    ) -> int:
        """Stream every manifest part into *tmp_path*, verifying integrity."""
        whole_sha256 = hashlib.sha256()
        written = 0

        with open(tmp_path, "wb") as f:
            for part in manifest["parts"]:
                part_name = part["name"]
                response = self._get_object(bucket, part_name, stream=True)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to download chunk part {bucket}/{part_name}: "
                        f"HTTP {response.status_code} — {response.text[:300]}"
                    )
                part_sha256 = hashlib.sha256()
                part_size = 0
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    part_sha256.update(chunk)
                    whole_sha256.update(chunk)
                    part_size += len(chunk)
                    written += len(chunk)

                if part_size != part["size"]:
                    raise RuntimeError(
                        f"Chunk part size mismatch for {bucket}/{part_name}: "
                        f"got {part_size}, expected {part['size']}"
                    )
                if part_sha256.hexdigest() != part["sha256"]:
                    raise RuntimeError(f"Chunk part checksum mismatch for {bucket}/{part_name}")

        if written != manifest["size"]:
            raise RuntimeError(
                f"Chunked artifact size mismatch for {bucket}/{name}: "
                f"got {written}, expected {manifest['size']}"
            )
        if whole_sha256.hexdigest() != manifest["sha256"]:
            raise RuntimeError(f"Chunked artifact checksum mismatch for {bucket}/{name}")

        return written

    def download_artifact_to_file(
        self, bucket: str, name: str, dest_path: str | Path
    ) -> int:
        """
        Download *name* straight to *dest_path* (bounded memory use).

        Writes to a ``<dest>.part-download`` temp file first, then
        ``os.replace``s it into place atomically — callers that check
        ``exists()`` on *dest_path* never observe a partially-downloaded
        file. On any failure the temp file is removed and the error raised.

        Falls back to chunked reassembly the same way ``download_artifact``
        does.

        Returns the number of bytes written.
        """
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest.with_name(dest.name + ".part-download")

        try:
            response = self._get_object(bucket, name, stream=True)

            if response.status_code == 200:
                written = 0
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        written += len(chunk)
                os.replace(tmp_path, dest)
                logger.debug(
                    "Downloaded artifact to file: %s/%s (%d bytes) -> %s",
                    bucket, name, written, dest,
                )
                return written

            if response.status_code in (400, 404):
                manifest = self._get_chunk_manifest(bucket, name)
                if manifest is not None:
                    written = self._download_chunked_to_file(bucket, name, manifest, tmp_path)
                    os.replace(tmp_path, dest)
                    logger.debug(
                        "Downloaded chunked artifact to file: %s/%s (%d bytes) -> %s",
                        bucket, name, written, dest,
                    )
                    return written
                if response.status_code == 404:
                    raise RuntimeError(f"Artifact not found: {bucket}/{name}")

            raise RuntimeError(
                f"Failed to download artifact: HTTP {response.status_code} — {response.text[:300]}"
            )
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    def _delete_artifact_url(self, bucket: str) -> str:
        """URL for DELETE artifact (filename goes as query param)."""
        bucket_segment = quote(str(bucket).lower(), safe="")
        return (
            f"{self.base_url}{self.api_path}/artifacts/artifact"
            f"/default/{self.project_id}/{bucket_segment}"
        )

    def delete_artifact(self, bucket: str, name: str) -> None:
        """Delete a file from the platform bucket.

        Platform DELETE endpoint expects filename as a query parameter,
        not as a path segment (unlike GET which uses path).

        Raises:
            RuntimeError: If the platform returns an error status.
        """
        import requests

        url = self._delete_artifact_url(bucket)
        logger.debug("Deleting artifact: %s/%s", bucket, name)

        response = requests.delete(
            url,
            headers=self._headers(),
            params={"filename": name},
            verify=False,
            timeout=60,
        )
        if response.status_code in (200, 204):
            return
        if response.status_code == 404:
            logger.debug("Artifact already absent: %s/%s", bucket, name)
            return
        raise RuntimeError(
            f"Delete artifact failed HTTP {response.status_code} for {bucket}/{name}"
        )

    def list_artifacts(
        self, bucket: str, prefix: str = ""
    ) -> List[Dict[str, Any]]:
        """
        List files in a platform bucket.

        Args:
            bucket: Bucket name.
            prefix: Optional prefix to filter results.

        Returns:
            List of dicts with 'name', 'size', 'modified' keys.
        """
        import requests

        url = self._artifact_url(bucket)
        logger.debug("Listing artifacts in bucket %s (prefix=%s)", bucket, prefix)

        response = requests.get(
            url, headers=self._headers(), verify=False, timeout=60
        )

        if response.status_code == 404:
            return []
        if response.status_code != 200:
            logger.warning("List artifacts returned HTTP %d", response.status_code)
            return []

        try:
            data = response.json()
            # Platform response may be a list directly or wrapped in a key
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("rows", data.get("items", []))
            else:
                items = []

            result = []
            for item in items:
                name = item.get("name", item.get("file_name", ""))
                if prefix and not name.startswith(prefix):
                    continue
                result.append({
                    "name": name,
                    "size": item.get("size", 0),
                    "modified": item.get("modified", item.get("upload_date", "")),
                })
            return result
        except Exception as e:
            logger.warning("Failed to parse artifact list: %s", e)
            return []

    def artifact_exists(self, bucket: str, name: str) -> bool:
        """Check whether a file exists in the platform bucket."""
        try:
            self.download_artifact(bucket, name)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Convenience: matches MiniArtifactClient interface (for registries)
    # ------------------------------------------------------------------

    def create_artifact(
        self, bucket_name: str, artifact_name: str, data: str
    ) -> Dict[str, Any]:
        """Alias for upload_artifact (matches MiniArtifactClient interface)."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return self.upload_artifact(bucket_name, artifact_name, data)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_platform_client_from_settings(
    artifact_settings: Dict[str, Any],
) -> Optional[PlatformArtifactClient]:
    """
    Create a PlatformArtifactClient from artifact settings dict.

    Returns None if settings are incomplete.
    """
    if not artifact_settings:
        return None

    base_url = artifact_settings.get("base_url", "")
    api_key = artifact_settings.get("api_key", "")

    if not base_url or not api_key:
        logger.debug("Incomplete artifact settings — cannot create platform client")
        return None

    try:
        return PlatformArtifactClient(artifact_settings)
    except Exception as exc:
        logger.warning("Failed to create PlatformArtifactClient: %s", exc)
        return None


def create_platform_client_from_env() -> Optional[PlatformArtifactClient]:
    """
    Create a PlatformArtifactClient from environment variables.

    Reads DEEPWIKI_ARTIFACT_BASE_URL, DEEPWIKI_ARTIFACT_API_KEY, etc.
    Returns None if env vars are not set.
    """
    settings = extract_artifact_settings_from_env()
    if not settings:
        return None
    return create_platform_client_from_settings(settings)


def create_platform_client_from_llm_settings(
    llm_settings: Dict[str, Any],
) -> Optional[PlatformArtifactClient]:
    """
    Create a PlatformArtifactClient from the llm_settings dict
    provided by the platform on each invocation.
    """
    if not llm_settings:
        return None
    settings = extract_artifact_settings(llm_settings)
    return create_platform_client_from_settings(settings)
