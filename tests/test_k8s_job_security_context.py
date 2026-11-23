"""The generated Job's securityContext comes from the controller's environment."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

from kubernetes import client
import pytest

from plugin_implementation import k8s_job_manager as jm
from plugin_implementation.k8s_job_manager import K8sJobManager

SECURITY_ENVS = (
    jm.JOB_RUN_AS_USER_ENV,
    jm.JOB_RUN_AS_GROUP_ENV,
    jm.JOB_FS_GROUP_ENV,
    jm.JOB_RUN_AS_NON_ROOT_ENV,
    jm.JOB_ALLOW_PRIVILEGE_ESCALATION_ENV,
    jm.JOB_READ_ONLY_ROOT_FILESYSTEM_ENV,
    jm.JOB_CAPABILITIES_DROP_ENV,
    jm.JOB_SECCOMP_PROFILE_ENV,
)


def _pod_spec(tmp_path, monkeypatch, with_init=True):
    monkeypatch.setenv("DEEPWIKI_LICENSE_USERNAME", "test-user")
    manager = K8sJobManager(base_path=str(tmp_path))
    monkeypatch.setattr(manager, "get_slot_availability", lambda: {"can_start": True})
    if not with_init:
        monkeypatch.setattr(manager, "_build_init_container", lambda _: None)
    else:
        init = client.V1Container(
            name="plugin-bootstrap",
            image="x",
            security_context=jm.build_container_security_context(client, manager.security),
        )
        monkeypatch.setattr(manager, "_build_init_container", lambda _: init)
    monkeypatch.setattr(manager, "_upload_job_input", lambda *args, **kwargs: True)
    batch_api = SimpleNamespace(create_namespaced_job=Mock())
    monkeypatch.setattr(manager, "_get_batch_api", lambda: batch_api)

    assert manager.create_job(
        "security-test",
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
    return client.ApiClient().sanitize_for_serialization(job)["spec"]["template"]["spec"]


@pytest.fixture
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("DEEPWIKI_"):
            monkeypatch.delenv(name)


def test_defaults_are_hardened(tmp_path, monkeypatch, clean_env):
    pod = _pod_spec(tmp_path, monkeypatch)

    assert pod["securityContext"] == {
        "runAsUser": 33,
        "runAsGroup": 33,
        "fsGroup": 33,
        "runAsNonRoot": True,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    expected = {
        "runAsUser": 33,
        "runAsGroup": 33,
        "runAsNonRoot": True,
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": False,
        "capabilities": {"drop": ["ALL"]},
    }
    assert pod["containers"][0]["securityContext"] == expected
    assert pod["initContainers"][0]["securityContext"] == expected


def test_real_init_container_uses_the_same_settings(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv("DEEPWIKI_LICENSE_USERNAME", "test-user")
    monkeypatch.setenv("DEEPWIKI_LICENSE_PASSWORD", "test-pass")
    monkeypatch.setenv(jm.JOB_RUN_AS_USER_ENV, "1000")
    manager = K8sJobManager(base_path=str(tmp_path))

    init = manager._build_init_container(client)
    if init is None:
        pytest.skip("init container disabled in this configuration")
    sc = client.ApiClient().sanitize_for_serialization(init)["securityContext"]
    assert sc["runAsUser"] == 1000
    assert sc["allowPrivilegeEscalation"] is False
    assert sc["capabilities"] == {"drop": ["ALL"]}


def test_values_come_from_env(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv(jm.JOB_RUN_AS_USER_ENV, "1000")
    monkeypatch.setenv(jm.JOB_RUN_AS_GROUP_ENV, "1000")
    monkeypatch.setenv(jm.JOB_FS_GROUP_ENV, "2000")
    monkeypatch.setenv(jm.JOB_READ_ONLY_ROOT_FILESYSTEM_ENV, "true")
    monkeypatch.setenv(jm.JOB_CAPABILITIES_DROP_ENV, "ALL, NET_RAW")
    monkeypatch.setenv(jm.JOB_SECCOMP_PROFILE_ENV, "Localhost")

    pod = _pod_spec(tmp_path, monkeypatch, with_init=False)

    assert pod["securityContext"]["fsGroup"] == 2000
    assert pod["securityContext"]["seccompProfile"] == {"type": "Localhost"}
    sc = pod["containers"][0]["securityContext"]
    assert (sc["runAsUser"], sc["runAsGroup"]) == (1000, 1000)
    assert sc["readOnlyRootFilesystem"] is True
    assert sc["capabilities"] == {"drop": ["ALL", "NET_RAW"]}


def test_blank_env_leaves_field_unset(tmp_path, monkeypatch, clean_env):
    for name in SECURITY_ENVS:
        monkeypatch.setenv(name, "")

    pod = _pod_spec(tmp_path, monkeypatch, with_init=False)

    assert "securityContext" not in pod or pod["securityContext"] == {}
    assert pod["containers"][0].get("securityContext", {}) == {}


@pytest.mark.parametrize(
    "name,value",
    [(jm.JOB_RUN_AS_USER_ENV, "www-data"), (jm.JOB_ALLOW_PRIVILEGE_ESCALATION_ENV, "maybe")],
)
def test_invalid_env_fails_at_controller_start(tmp_path, monkeypatch, clean_env, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        K8sJobManager(base_path=str(tmp_path))
