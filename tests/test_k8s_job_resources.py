"""Storage resources must reach the generated Job independently of its volume cap."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

from kubernetes import client
import pytest

from plugin_implementation.k8s_job_manager import K8sJobManager


@pytest.mark.parametrize(
    "storage_request,storage_limit",
    [(None, None), ("", ""), ("2Gi", None), (None, "25Gi"), ("2Gi", "25Gi")],
)
def test_generated_job_storage_resources(
    tmp_path, monkeypatch, storage_request, storage_limit
):
    for name in list(os.environ):
        if name.startswith("DEEPWIKI_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("DEEPWIKI_LICENSE_USERNAME", "test-user")
    monkeypatch.setenv("DEEPWIKI_JOB_EMPTY_DIR_SIZE_LIMIT", "20Gi")
    monkeypatch.setenv("DEEPWIKI_JOB_CPU_REQUEST", "2")
    if storage_request is not None:
        monkeypatch.setenv("DEEPWIKI_JOB_EPHEMERAL_STORAGE_REQUEST", storage_request)
    if storage_limit is not None:
        monkeypatch.setenv("DEEPWIKI_JOB_EPHEMERAL_STORAGE_LIMIT", storage_limit)

    manager = K8sJobManager(base_path=str(tmp_path))
    monkeypatch.setattr(manager, "get_slot_availability", lambda: {"can_start": True})
    monkeypatch.setattr(manager, "_build_init_container", lambda _: None)
    monkeypatch.setattr(manager, "_upload_job_input", lambda *args, **kwargs: True)
    batch_api = SimpleNamespace(create_namespaced_job=Mock())
    monkeypatch.setattr(manager, "_get_batch_api", lambda: batch_api)

    assert manager.create_job(
        "storage-test",
        {
            "query": "GO",
            "llm_settings": {
                "api_base": "https://platform.example/llm/v1",
                "api_key": "test-token",
                "project_id": "7",
            },
        },
    )["success"]
    job = batch_api.create_namespaced_job.call_args.kwargs["body"]
    pod = client.ApiClient().sanitize_for_serialization(job)["spec"]["template"]["spec"]
    expected_requests = {"memory": "2Gi", "cpu": "2"}
    expected_limits = {"memory": "8Gi", "cpu": "4"}
    if storage_request:
        expected_requests["ephemeral-storage"] = storage_request
    if storage_limit:
        expected_limits["ephemeral-storage"] = storage_limit
    assert pod["containers"][0]["resources"] == {
        "requests": expected_requests,
        "limits": expected_limits,
    }
    assert pod["volumes"][0]["emptyDir"] == {"sizeLimit": "20Gi"}
