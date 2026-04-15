"""
Platform Artifact Client for DeepWiki artifact transport.

Uses the ELITEA platform's artifact HTTP API (the same API that invoke.py's
MiniArtifactClient and elitea-sdk's EliteAClient use) instead of direct S3/boto3
connections.

In ELITEA, all artifact storage goes through the platform's artifact endpoints:
    - Upload:   POST  /api/v1/artifacts/artifacts/default/{project_id}/{bucket}
    - Download: GET   /api/v1/artifacts/artifact/default/{project_id}/{bucket}/{name}
    - List:     GET   /api/v1/artifacts/artifacts/default/{project_id}/{bucket}
    - Delete:   DELETE /api/v1/artifacts/artifact/default/{project_id}/{bucket}/{name}

The platform internally routes these to Minio/S3 with proper bucket prefixing,
access control, and project scoping.

Credential flow:
    1. Controller receives ``llm_settings`` from the platform on each invocation.
    2. ``extract_artifact_settings(llm_settings)`` derives base_url, api_key, project_id.
    3. For K8s Jobs, the controller injects artifact settings as env vars into the
       worker pod: DEEPWIKI_ARTIFACT_BASE_URL, DEEPWIKI_ARTIFACT_API_KEY,
       DEEPWIKI_ARTIFACT_PROJECT_ID.
    4. Worker reads those env vars to build its own PlatformArtifactClient.

Bucket key layout (inside the "wiki_artifacts" bucket):
    _jobs/{job_id}/input.json                  Job input  (controller -> worker)
    _jobs/{job_id}/result.json                 Job result (worker -> controller)
    {wiki_id}/indexes/{file}                   Index files (worker uploads)
    {wiki_id}/wiki_manifest_{version}.json     Versioned manifest (worker uploads)
    {wiki_id}/wiki_pages/{section}/{page}.md   Wiki pages (worker uploads)
    {wiki_id}/analysis/wiki_structure_*.json   Structure analysis (worker uploads)
    _registry/wikis.json                       Global wiki registry (controller uploads)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
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

# All env var names that should be passed to worker Job pods
ARTIFACT_ENV_VARS = [
    ARTIFACT_BASE_URL_ENV,
    ARTIFACT_API_KEY_ENV,
    ARTIFACT_PROJECT_ID_ENV,
    ARTIFACT_BUCKET_ENV,
    ARTIFACT_X_SECRET_ENV,
]

DEFAULT_BUCKET = "wiki_artifacts"
DEFAULT_API_PATH = "/api/v1"


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
    }


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

        if not self.base_url:
            raise ValueError("artifact_settings.base_url is required")
        if not self.api_key:
            raise ValueError("artifact_settings.api_key is required")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-SECRET": self.x_secret,
        }

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

    def upload_artifact(
        self, bucket: str, name: str, data: bytes | str
    ) -> Dict[str, Any]:
        """
        Upload a file to the platform bucket.

        Args:
            bucket: Bucket name.
            name: File key/path (may contain '/' for nested paths).
            data: File content (bytes or str).

        Returns:
            Platform response dict.
        """
        import requests

        if isinstance(data, str):
            data = data.encode("utf-8")

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
            raise RuntimeError(
                f"Not authorized to upload artifact (HTTP 403): {response.text[:200]}"
            )
        if response.status_code not in (200, 201):
            logger.error(
                "Artifact upload failed — url=%s, status=%d, response=%s",
                url, response.status_code, response.text[:500],
            )
            raise RuntimeError(
                f"Failed to upload artifact: HTTP {response.status_code} — {response.text[:500]}"
            )

        logger.debug("Uploaded artifact: %s/%s (%d bytes)", bucket, name, len(data))
        try:
            return response.json()
        except Exception:
            return {"status": "ok"}

    def download_artifact(self, bucket: str, name: str) -> bytes:
        """
        Download a file from the platform bucket.

        Args:
            bucket: Bucket name.
            name: File key/path.

        Returns:
            Raw file content as bytes.
        """
        import requests

        url = self._single_artifact_url(bucket, name)
        logger.debug("Downloading artifact: %s/%s", bucket, name)

        response = requests.get(
            url, headers=self._headers(), verify=False, timeout=300
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
        if response.status_code == 404:
            raise RuntimeError(f"Artifact not found: {bucket}/{name}")
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download artifact: HTTP {response.status_code} — {response.text[:300]}"
            )

        content = response.content
        logger.debug(
            "Downloaded artifact: %s/%s (%d bytes)", bucket, name, len(content)
        )
        return content

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
