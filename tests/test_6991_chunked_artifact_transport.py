"""Transparent chunked artifact transport for uploads/downloads over the
platform's 150 MB request-body limit (#6991).

Uses an in-memory fake HTTP layer (monkeypatched ``requests.post/get/delete``)
that mimics the platform's artifact API: POST multipart upload, GET single
object (400 for "not found"), DELETE with filename as a query param.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import pytest
import requests

from plugin_implementation.artifacts_platform_client import (
    ARTIFACT_PART_SIZE_MB_ENV,
    CHUNK_MANIFEST_SUFFIX,
    DEFAULT_ARTIFACT_PART_SIZE_MB,
    MAX_ARTIFACT_PART_SIZE_MB,
    PlatformArtifactClient,
    get_artifact_part_size,
    logical_artifact_name,
)
from plugin_implementation.artifact_manager import ArtifactManager

BASE_URL = "https://platform.example"
API_PATH = "/api/v2"
PROJECT_ID = "7"
BUCKET = "wiki-artifacts"


# ---------------------------------------------------------------------------
# Fake HTTP layer
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", json_data=None, text: str | None = None):
        self.status_code = status_code
        self.content = content
        self._json_data = json_data
        if text is not None:
            self.text = text
        elif json_data is not None:
            self.text = json.dumps(json_data)
        elif isinstance(content, bytes):
            self.text = content.decode("utf-8", errors="replace")
        else:
            self.text = str(content)

    def json(self):
        if self._json_data is not None:
            return self._json_data
        # Mirrors requests.Response.json(): parse the raw body on demand —
        # covers GET responses that return stored bytes (e.g. a manifest
        # round-tripped through the fake bucket) rather than a canned dict.
        return json.loads(self.content)

    def iter_content(self, chunk_size: int = 1024 * 1024):
        data = self.content
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]


def _bucket_only(url: str) -> str:
    """Parse the bucket out of a bucket-only artifact URL (list/upload/delete)."""
    marker = "/default/"
    idx = url.index(marker) + len(marker)
    _project_id, bucket = url[idx:].split("/", 1)
    return bucket


def _bucket_and_name(url: str) -> Tuple[str, str]:
    """Parse bucket + name out of a single-artifact URL (GET with name)."""
    marker = "/default/"
    idx = url.index(marker) + len(marker)
    _project_id, bucket, name = url[idx:].split("/", 2)
    return bucket, name


class FakePlatform:
    """In-memory stand-in for the platform's artifact HTTP API."""

    def __init__(self, limit: int = 10_000_000):
        self.store: Dict[Tuple[str, str], bytes] = {}
        self.limit = limit
        self.post_log: list = []
        self.get_log: list = []
        self.delete_log: list = []
        self.fail_times: Dict[Tuple[str, str], int] = {}
        self.force_413: set = set()

    # -- requests.post --
    def post(self, url, headers=None, files=None, verify=False, timeout=None, **kw):
        name, data = files["file"]
        bucket = _bucket_only(url)
        key = (bucket, name)
        self.post_log.append(key)

        if key in self.force_413:
            self.force_413.discard(key)
            return FakeResponse(413, json_data={"error": "File exceeds the 150 MB limit"})
        if self.fail_times.get(key, 0) > 0:
            self.fail_times[key] -= 1
            return FakeResponse(500, text="Internal Server Error")
        if len(data) > self.limit:
            return FakeResponse(413, json_data={"error": "File exceeds the 150 MB limit"})

        self.store[key] = bytes(data)
        return FakeResponse(201, json_data={"status": "ok"})

    # -- requests.get --
    def get(self, url, headers=None, verify=False, timeout=None, stream=False, **kw):
        self.get_log.append(url)
        if "/artifacts/artifacts/" in url:
            bucket = _bucket_only(url)
            items = [
                {"name": n, "size": len(d), "modified": "2026-01-01T00:00:00Z"}
                for (b, n), d in self.store.items() if b == bucket
            ]
            return FakeResponse(200, json_data=items)

        bucket, name = _bucket_and_name(url)
        key = (bucket, name)
        if key not in self.store:
            return FakeResponse(400, json_data={"error": "File was not found"})
        return FakeResponse(200, content=self.store[key])

    # -- requests.delete --
    def delete(self, url, headers=None, params=None, verify=False, timeout=None, **kw):
        bucket = _bucket_only(url)
        name = params["filename"]
        key = (bucket, name)
        self.delete_log.append(key)
        if key in self.store:
            del self.store[key]
            return FakeResponse(200)
        return FakeResponse(404)


@pytest.fixture
def platform():
    return FakePlatform(limit=10_000_000)


def install_fake(monkeypatch, platform: FakePlatform) -> None:
    monkeypatch.setattr(requests, "post", platform.post)
    monkeypatch.setattr(requests, "get", platform.get)
    monkeypatch.setattr(requests, "delete", platform.delete)


def make_client(part_size: int) -> PlatformArtifactClient:
    client = PlatformArtifactClient({
        "base_url": BASE_URL,
        "api_key": "test-key",
        "project_id": PROJECT_ID,
        "api_path": API_PATH,
        "x_secret": "test-secret",
    })
    client.part_size = part_size
    return client


# ---------------------------------------------------------------------------
# 1. Small upload — no regression
# ---------------------------------------------------------------------------

def test_small_upload_is_single_post_no_chunking(monkeypatch, platform):
    install_fake(monkeypatch, platform)
    client = make_client(part_size=256)

    client.upload_artifact(BUCKET, "small.json", b"x" * 50)

    assert platform.post_log == [(BUCKET, "small.json")]
    assert platform.delete_log == []
    assert (BUCKET, "small.json" + CHUNK_MANIFEST_SUFFIX) not in platform.store
    assert client.download_artifact(BUCKET, "small.json") == b"x" * 50


# ---------------------------------------------------------------------------
# 2. Large upload over the fake platform limit → chunked
# ---------------------------------------------------------------------------

def test_large_upload_is_chunked_and_round_trips(monkeypatch, tmp_path):
    # limit comfortably above part_size so every *part* passes, but well
    # below the full 2600-byte payload so the single-shot path would 413.
    platform = FakePlatform(limit=2000)
    install_fake(monkeypatch, platform)
    client = make_client(part_size=256)

    data = bytes((i % 251) for i in range(2600))  # not a multiple of part_size
    name = "big.wiki.db"

    result = client.upload_artifact(BUCKET, name, data)
    assert result["chunked"] is True
    assert result["size"] == len(data)

    # Every stored object respects the fake platform's limit.
    for (bucket, stored_name), stored_bytes in platform.store.items():
        if bucket == BUCKET and stored_name.startswith(name):
            assert len(stored_bytes) <= platform.limit

    # Manifest committed last.
    assert platform.post_log[-1] == (BUCKET, name + CHUNK_MANIFEST_SUFFIX)

    assert client.download_artifact(BUCKET, name) == data

    dest = tmp_path / "download" / "big.wiki.db"
    written = client.download_artifact_to_file(BUCKET, name, dest)
    assert written == len(data)
    assert dest.read_bytes() == data
    assert not dest.with_name(dest.name + ".part-download").exists()


# ---------------------------------------------------------------------------
# 3. Boundary: exactly part_size vs part_size + 1
# ---------------------------------------------------------------------------

def test_boundary_exact_part_size_is_single_post(monkeypatch, platform):
    install_fake(monkeypatch, platform)
    client = make_client(part_size=100)

    client.upload_artifact(BUCKET, "exact.bin", b"a" * 100)

    assert platform.post_log == [(BUCKET, "exact.bin")]
    assert (BUCKET, "exact.bin" + CHUNK_MANIFEST_SUFFIX) not in platform.store


def test_boundary_part_size_plus_one_is_chunked_two_parts(monkeypatch, platform):
    install_fake(monkeypatch, platform)
    client = make_client(part_size=100)

    data = b"b" * 101
    client.upload_artifact(BUCKET, "over.bin", data)

    manifest = client._get_chunk_manifest(BUCKET, "over.bin")
    assert manifest is not None
    assert len(manifest["parts"]) == 2
    assert manifest["parts"][0]["size"] == 100
    assert manifest["parts"][1]["size"] == 1
    assert client.download_artifact(BUCKET, "over.bin") == data


# ---------------------------------------------------------------------------
# 4. upload_file reads bounded parts from disk
# ---------------------------------------------------------------------------

def test_upload_file_large_path_round_trips(monkeypatch, platform, tmp_path):
    install_fake(monkeypatch, platform)
    client = make_client(part_size=200)

    data = os.urandom(200 * 3 + 37)
    src = tmp_path / "source.wiki.db"
    src.write_bytes(data)

    client.upload_file(BUCKET, "from_file.wiki.db", src)

    manifest = client._get_chunk_manifest(BUCKET, "from_file.wiki.db")
    assert manifest is not None
    assert manifest["size"] == len(data)
    assert client.download_artifact(BUCKET, "from_file.wiki.db") == data


def test_upload_file_small_path_is_single_post(monkeypatch, platform, tmp_path):
    install_fake(monkeypatch, platform)
    client = make_client(part_size=200)

    data = b"small file contents"
    src = tmp_path / "small_source.json"
    src.write_bytes(data)

    client.upload_file(BUCKET, "small_from_file.json", src)

    assert platform.post_log == [(BUCKET, "small_from_file.json")]
    assert client.download_artifact(BUCKET, "small_from_file.json") == data


# ---------------------------------------------------------------------------
# 5. Corrupted part → download raises, no partial file left behind
# ---------------------------------------------------------------------------

def test_corrupted_part_fails_download_cleanly(monkeypatch, platform, tmp_path):
    install_fake(monkeypatch, platform)
    client = make_client(part_size=50)

    data = b"c" * 170
    client.upload_artifact(BUCKET, "corrupt.bin", data)

    # Mutate a stored part so its checksum no longer matches the manifest.
    part_key = (BUCKET, "corrupt.bin.chunk-" + client._get_chunk_manifest(BUCKET, "corrupt.bin")["generation"] + "-00001")
    assert part_key in platform.store
    original = platform.store[part_key]
    platform.store[part_key] = bytes((b ^ 0xFF) for b in original)

    with pytest.raises(RuntimeError):
        client.download_artifact(BUCKET, "corrupt.bin")

    dest = tmp_path / "out" / "corrupt.bin"
    with pytest.raises(RuntimeError):
        client.download_artifact_to_file(BUCKET, "corrupt.bin", dest)

    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part-download").exists()


# ---------------------------------------------------------------------------
# 6. Overwrite semantics
# ---------------------------------------------------------------------------

def test_overwrite_chunked_deletes_old_generation_parts(monkeypatch, platform):
    install_fake(monkeypatch, platform)
    client = make_client(part_size=64)

    v1 = b"1" * 200
    client.upload_artifact(BUCKET, "over.wiki.db", v1)
    manifest_v1 = client._get_chunk_manifest(BUCKET, "over.wiki.db")
    old_part_names = {p["name"] for p in manifest_v1["parts"]}
    assert old_part_names  # sanity: there were parts

    v2 = b"2" * 300
    client.upload_artifact(BUCKET, "over.wiki.db", v2)

    for part_name in old_part_names:
        assert (BUCKET, part_name) not in platform.store

    assert client.download_artifact(BUCKET, "over.wiki.db") == v2


def test_overwrite_deletes_stale_single_object(monkeypatch, platform):
    install_fake(monkeypatch, platform)
    client = make_client(part_size=64)

    # Seed a stale single (non-chunked) object under the target name.
    client.upload_artifact(BUCKET, "stale.wiki.db", b"stale-single-object")
    assert (BUCKET, "stale.wiki.db") in platform.store

    big = b"3" * 300
    client.upload_artifact(BUCKET, "stale.wiki.db", big)

    # Stale single object removed after the chunked commit.
    assert (BUCKET, "stale.wiki.db") not in platform.store
    assert client.download_artifact(BUCKET, "stale.wiki.db") == big


# ---------------------------------------------------------------------------
# 7. Retry behavior for part uploads
# ---------------------------------------------------------------------------

def test_part_upload_retries_on_5xx_then_succeeds(monkeypatch, platform):
    install_fake(monkeypatch, platform)
    client = make_client(part_size=64)

    data = b"4" * 150

    # The chunk generation id is random, so fail the *first* part-upload
    # attempt this test observes, exactly once, rather than targeting a
    # specific precomputed part name.
    seen = {"failed_once": False}
    real_post = platform.post

    def flaky_post(url, headers=None, files=None, verify=False, timeout=None, **kw):
        name, _data = files["file"]
        if not seen["failed_once"] and ".chunk-" in name:
            seen["failed_once"] = True
            platform.post_log.append((_bucket_only(url), name))
            return FakeResponse(500, text="Internal Server Error")
        return real_post(url, headers=headers, files=files, verify=verify, timeout=timeout, **kw)

    monkeypatch.setattr(requests, "post", flaky_post)

    client.upload_artifact(BUCKET, "retry.bin", data)

    assert seen["failed_once"] is True
    assert client.download_artifact(BUCKET, "retry.bin") == data
    # 3 parts for 150 bytes / 64-byte parts, plus exactly one retried attempt.
    chunk_attempts = [k for k in platform.post_log if ".chunk-" in k[1]]
    assert len(chunk_attempts) == 4
    assert len(set(chunk_attempts)) == 3


def test_part_upload_413_is_not_retried_and_previous_manifest_untouched(monkeypatch, platform):
    install_fake(monkeypatch, platform)
    client = make_client(part_size=64)

    v1 = b"5" * 150
    client.upload_artifact(BUCKET, "keep.bin", v1)
    manifest_before = client._get_chunk_manifest(BUCKET, "keep.bin")

    v2 = b"6" * 150

    attempts = {"count": 0}
    real_post = platform.post

    def reject_second_part(url, headers=None, files=None, verify=False, timeout=None, **kw):
        name, _data = files["file"]
        if ".chunk-" in name:
            attempts["count"] += 1
            if attempts["count"] == 2:
                platform.post_log.append((_bucket_only(url), name))
                return FakeResponse(413, json_data={"error": "File exceeds the 150 MB limit"})
        return real_post(url, headers=headers, files=files, verify=verify, timeout=timeout, **kw)

    monkeypatch.setattr(requests, "post", reject_second_part)

    with pytest.raises(RuntimeError):
        client.upload_artifact(BUCKET, "keep.bin", v2)

    # Exactly one attempt at the rejecting part — no retry on 413.
    assert attempts["count"] == 2

    # Previous manifest/data is untouched and still readable.
    manifest_after = client._get_chunk_manifest(BUCKET, "keep.bin")
    assert manifest_after == manifest_before
    assert client.download_artifact(BUCKET, "keep.bin") == v1


# ---------------------------------------------------------------------------
# 8. Missing object, no manifest — unchanged error messages
# ---------------------------------------------------------------------------

def test_download_missing_400_no_manifest(monkeypatch, platform):
    install_fake(monkeypatch, platform)
    client = make_client(part_size=64)

    with pytest.raises(RuntimeError) as exc_info:
        client.download_artifact(BUCKET, "nothing-here")

    assert "Failed to download artifact: HTTP 400" in str(exc_info.value)


def test_download_missing_404_no_manifest(monkeypatch):
    def fake_get_404(url, headers=None, verify=False, timeout=None, stream=False, **kw):
        return FakeResponse(404, text="Not Found")

    monkeypatch.setattr(requests, "get", fake_get_404)
    monkeypatch.setattr(requests, "post", lambda *a, **kw: FakeResponse(500))
    monkeypatch.setattr(requests, "delete", lambda *a, **kw: FakeResponse(500))

    client = make_client(part_size=64)

    with pytest.raises(RuntimeError) as exc_info:
        client.download_artifact(BUCKET, "nothing-here")

    assert str(exc_info.value) == f"Artifact not found: {BUCKET}/nothing-here"


# ---------------------------------------------------------------------------
# 9. logical_artifact_name + get_artifact_part_size
# ---------------------------------------------------------------------------

def test_logical_artifact_name_unchanged_for_plain_name():
    assert logical_artifact_name("foo.wiki.db") == "foo.wiki.db"


def test_logical_artifact_name_strips_chunk_part_suffix():
    assert logical_artifact_name("foo.wiki.db.chunk-aabbccddeeff-00007") == "foo.wiki.db"


def test_logical_artifact_name_strips_manifest_suffix():
    assert logical_artifact_name("foo.wiki.db.chunks.json") == "foo.wiki.db"


def test_get_artifact_part_size_default(monkeypatch):
    monkeypatch.delenv(ARTIFACT_PART_SIZE_MB_ENV, raising=False)
    assert get_artifact_part_size() == DEFAULT_ARTIFACT_PART_SIZE_MB * 1024 * 1024


def test_get_artifact_part_size_clamps_to_max(monkeypatch):
    monkeypatch.setenv(ARTIFACT_PART_SIZE_MB_ENV, "500")
    assert get_artifact_part_size() == MAX_ARTIFACT_PART_SIZE_MB * 1024 * 1024


def test_get_artifact_part_size_valid_value(monkeypatch):
    monkeypatch.setenv(ARTIFACT_PART_SIZE_MB_ENV, "10")
    assert get_artifact_part_size() == 10 * 1024 * 1024


@pytest.mark.parametrize("raw", ["0", "-5", "not-a-number", ""])
def test_get_artifact_part_size_invalid_falls_back_to_default(monkeypatch, raw):
    if raw == "":
        monkeypatch.delenv(ARTIFACT_PART_SIZE_MB_ENV, raising=False)
    else:
        monkeypatch.setenv(ARTIFACT_PART_SIZE_MB_ENV, raw)
    assert get_artifact_part_size() == DEFAULT_ARTIFACT_PART_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# 10. ArtifactManager integration
# ---------------------------------------------------------------------------

@pytest.fixture
def jobs_mode_env(monkeypatch):
    monkeypatch.setenv("DEEPWIKI_JOBS_ENABLED", "true")


def test_artifact_manager_upload_indexes_uses_upload_file(monkeypatch, platform, tmp_path, jobs_mode_env):
    install_fake(monkeypatch, platform)
    client = make_client(part_size=64)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    index_bytes = b"7" * 200  # > part_size, forces chunking
    (cache_dir / "abc123.wiki.db").write_bytes(index_bytes)

    manifest = {
        "canonical_repo_identifier": "owner/repo:main",
        "branch": "main",
        "unified_db_key": "abc123",
    }

    mgr = ArtifactManager(cache_dir=cache_dir, artifacts_client=client, bucket=BUCKET)
    ok = mgr.upload_indexes(wiki_id="owner--repo--main", manifest=manifest)

    assert ok is True
    uploaded_name = "owner--repo--main/indexes/abc123.wiki.db"
    chunk_manifest = client._get_chunk_manifest(BUCKET, uploaded_name)
    assert chunk_manifest is not None
    assert chunk_manifest["size"] == len(index_bytes)


def test_artifact_manager_cleanup_stale_indexes_keeps_current_chunk_parts(
    monkeypatch, platform, tmp_path, jobs_mode_env
):
    install_fake(monkeypatch, platform)
    client = make_client(part_size=64)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    index_bytes = b"8" * 200
    (cache_dir / "abc123.wiki.db").write_bytes(index_bytes)

    manifest = {
        "canonical_repo_identifier": "owner/repo:main",
        "branch": "main",
        "unified_db_key": "abc123",
    }
    wiki_id = "owner--repo--main"

    mgr = ArtifactManager(cache_dir=cache_dir, artifacts_client=client, bucket=BUCKET)
    assert mgr.upload_indexes(wiki_id=wiki_id, manifest=manifest) is True

    # A genuinely stale index file from a previous generation, not referenced
    # by the current manifest's cache keys.
    client.upload_artifact(BUCKET, f"{wiki_id}/indexes/old999.wiki.db", b"stale-data")

    before_keys = {k for k in platform.store if k[0] == BUCKET and k[1].startswith(f"{wiki_id}/indexes/")}
    assert any("abc123.wiki.db" in k[1] for k in before_keys)
    assert (BUCKET, f"{wiki_id}/indexes/old999.wiki.db") in before_keys

    stats = mgr.cleanup_stale_indexes(wiki_id=wiki_id, current_manifest=manifest)

    assert stats["indexes_deleted"] == 1
    after_keys = {k for k in platform.store if k[0] == BUCKET and k[1].startswith(f"{wiki_id}/indexes/")}
    assert (BUCKET, f"{wiki_id}/indexes/old999.wiki.db") not in after_keys
    # Current generation's chunk parts and manifest survive cleanup.
    assert any("abc123.wiki.db" in k[1] for k in after_keys)
    assert client._get_chunk_manifest(BUCKET, f"{wiki_id}/indexes/abc123.wiki.db") is not None


def test_artifact_manager_ensure_indexes_downloads_chunked_index(
    monkeypatch, platform, tmp_path, jobs_mode_env
):
    install_fake(monkeypatch, platform)
    client = make_client(part_size=64)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    index_bytes = b"9" * 200
    (cache_dir / "abc123.wiki.db").write_bytes(index_bytes)

    manifest = {
        "canonical_repo_identifier": "owner/repo:main",
        "branch": "main",
        "unified_db_key": "abc123",
    }
    wiki_id = "owner--repo--main"

    mgr = ArtifactManager(cache_dir=cache_dir, artifacts_client=client, bucket=BUCKET)
    assert mgr.upload_indexes(wiki_id=wiki_id, manifest=manifest) is True

    # Simulate a fresh pod: local cache file is gone.
    (cache_dir / "abc123.wiki.db").unlink()

    ok = mgr.ensure_indexes_for_wiki(wiki_id=wiki_id, manifest=manifest)

    assert ok is True
    assert (cache_dir / "abc123.wiki.db").read_bytes() == index_bytes
