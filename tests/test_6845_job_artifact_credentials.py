"""Job workers must use the artifact client that uploaded their input (#6845)."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from plugin_implementation.artifacts_platform_client import ARTIFACT_ENV_VARS
from plugin_implementation.k8s_job_manager import K8sJobManager
from plugin_implementation.wiki_job_worker import load_input


def test_worker_script_starts_with_kubernetes_pythonpath():
    plugin_root = Path(__file__).resolve().parents[1]
    worker_script = plugin_root / "plugin_implementation" / "wiki_job_worker.py"

    result = subprocess.run(
        [sys.executable, str(worker_script), "--help"],
        env={**os.environ, "PYTHONPATH": str(plugin_root)},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "--job-id" in result.stdout


@pytest.fixture
def job_manager(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("DEEPWIKI_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("DEEPWIKI_ARTIFACT_BASE_URL", "https://platform.example")
    monkeypatch.setenv("DEEPWIKI_ARTIFACT_API_KEY", "platform-token")
    monkeypatch.setenv("DEEPWIKI_ARTIFACT_PROJECT_ID", "7")
    monkeypatch.setenv("DEEPWIKI_ARTIFACT_X_SECRET", "platform-secret")
    monkeypatch.setenv("DEEPWIKI_ARTIFACT_BUCKET", "job-inputs")
    monkeypatch.setenv("DEEPWIKI_EXCLUDE_TESTS", "1")
    monkeypatch.setenv("DEEPWIKI_LICENSE_USERNAME", "test-user")
    manager = K8sJobManager(base_path=str(tmp_path / "controller"))
    monkeypatch.setattr(manager, "get_slot_availability", lambda: {"can_start": True})
    monkeypatch.setattr(manager, "_build_init_container", lambda _client: None)
    batch_api = SimpleNamespace(create_namespaced_job=Mock())
    monkeypatch.setattr(manager, "_get_batch_api", lambda: batch_api)
    return manager, batch_api


@pytest.mark.parametrize("source", ["environment_fallback", "request", "empty_scope"])
def test_worker_downloads_input_with_upload_credentials(
    source, job_manager, tmp_path, monkeypatch
):
    manager, batch_api = job_manager
    llm_settings = {
        "provider": "openai",
        "model_name": "gpt-5-mini-2025-08-07",
        "api_key": "dial-model-token",
        "project_id": "99",
    }
    if source != "environment_fallback":
        llm_settings.update(
            api_base="https://request-platform.example/llm/v1",
            organization="12" if source == "request" else "",
            project_id="12" if source == "request" else "",
            x_secret="request-secret" if source == "request" else "",
        )
    payload = {"llm_settings": llm_settings, "query": "GO"}
    uploaded = {}

    def upload(url, **kwargs):
        uploaded.update(url=url, headers=kwargs["headers"], file=kwargs["files"]["file"])
        return SimpleNamespace(status_code=201, json=lambda: {"status": "ok"})

    def download(url, **kwargs):
        expected_url = uploaded["url"].replace("/artifacts/artifacts/", "/artifacts/artifact/")
        expected_url += "/" + uploaded["file"][0]
        if url != expected_url or kwargs["headers"] != uploaded["headers"]:
            return SimpleNamespace(status_code=403, text="Incorrect artifact credentials")
        return SimpleNamespace(status_code=200, content=uploaded["file"][1])

    monkeypatch.setattr(requests, "post", upload)
    monkeypatch.setattr(requests, "get", download)
    assert manager.create_job("test-job", payload)["success"]
    pod = batch_api.create_namespaced_job.call_args.kwargs["body"].spec.template.spec
    worker_env = pod.containers[0].env

    assert next(entry.value for entry in worker_env if entry.name == "DEEPWIKI_EXCLUDE_TESTS") == "1"
    assert pod.volumes[0].empty_dir is not None

    for name in ARTIFACT_ENV_VARS:
        monkeypatch.delenv(name)
    for entry in worker_env:
        monkeypatch.setenv(entry.name, entry.value)
    # Simulate the worker's emptyDir, separate from the controller's local input.
    monkeypatch.setenv("DEEPWIKI_BASE_PATH", str(tmp_path / "worker"))
    assert load_input("test-job") == payload
    cached = tmp_path / "worker" / "jobs" / "test-job" / "input.json"
    assert json.loads(cached.read_text()) == payload
    # Kubernetes receives a single authoritative value for each artifact setting.
    for name in ARTIFACT_ENV_VARS:
        assert sum(entry.name == name for entry in worker_env) == 1


def test_failed_input_upload_does_not_start_worker(job_manager, monkeypatch):
    manager, batch_api = job_manager
    monkeypatch.setattr(
        requests, "post", lambda *args, **kwargs: SimpleNamespace(status_code=503, text="Unavailable")
    )

    result = manager.create_job("test-job", {"llm_settings": {}, "query": "GO"})

    assert result["error_category"] == "platform_upload_failed"
    batch_api.create_namespaced_job.assert_not_called()
