"""Regression tests for GitHub Security tab fixes (CodeQL alerts)."""

from __future__ import annotations

import importlib
import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from plugin_implementation.artifacts_platform_client import PlatformArtifactClient
from plugin_implementation.repo_providers.factory import RepoProviderFactory
from plugin_implementation.repo_providers.models import (
    ProviderConfig,
    ProviderType,
    extract_hostname,
    host_matches_domain,
)
from plugin_implementation.repo_providers.providers import BitbucketProvider


class _FakeResponse:
    def __init__(self, body: str, status_code: int):
        self._body = body
        self.status_code = status_code
        self.headers = {}

    def get_data(self, as_text: bool = False):
        return self._body if as_text else self._body.encode("utf-8")


def _load_ui_route_module(monkeypatch):
    """Import routes.ui with minimal Pylon and Flask stubs."""
    pylon = types.ModuleType("pylon")
    core = types.ModuleType("pylon.core")
    tools = types.ModuleType("pylon.core.tools")

    class _Web:
        def route(self, *args, **kwargs):
            return lambda fn: fn

    tools.web = _Web()

    fake_flask = types.ModuleType("flask")
    fake_flask.request = SimpleNamespace(headers={})
    fake_flask.make_response = lambda body, status=200: _FakeResponse(body, status)
    fake_flask.send_from_directory = lambda *args, **kwargs: _FakeResponse("", 200)

    monkeypatch.setitem(sys.modules, "pylon", pylon)
    monkeypatch.setitem(sys.modules, "pylon.core", core)
    monkeypatch.setitem(sys.modules, "pylon.core.tools", tools)
    monkeypatch.setitem(sys.modules, "flask", fake_flask)
    sys.modules.pop("routes.ui", None)
    return importlib.import_module("routes.ui"), fake_flask


def test_artifact_403_logs_do_not_leak_api_key_prefix(caplog, monkeypatch):
    secret_key = "sk-super-secret-token-value-123456"
    client = PlatformArtifactClient(
        {
            "base_url": "https://platform.example",
            "api_key": secret_key,
            "project_id": "7",
        }
    )

    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: SimpleNamespace(status_code=403, text="Forbidden"),
    )
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(status_code=403, text="Forbidden"),
    )

    with caplog.at_level(logging.ERROR, logger="plugin_implementation.artifacts_platform_client"):
        with pytest.raises(Exception):
            client._post_object("bucket", "file.txt", b"data")
        with pytest.raises(RuntimeError):
            client._get_object("bucket", "file.txt")

    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert secret_key not in messages
    assert secret_key[:8] not in messages
    assert "key_prefix=" not in messages
    assert "auth=configured" in messages


def test_artifact_403_logs_missing_auth_when_placeholder_or_empty(caplog, monkeypatch):
    client = PlatformArtifactClient(
        {
            "base_url": "https://platform.example",
            "api_key": "valid-temp-key",
            "project_id": "7",
        }
    )
    client.api_key = ""

    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(status_code=403, text="Forbidden"),
    )

    with caplog.at_level(logging.ERROR, logger="plugin_implementation.artifacts_platform_client"):
        with pytest.raises(RuntimeError):
            client._get_object("bucket", "file.txt")

    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "auth=missing" in messages


def test_ui_route_validates_project_id_header_and_blocks_xss(tmp_path, monkeypatch):
    ui_module, fake_flask = _load_ui_route_module(monkeypatch)

    dist_dir = tmp_path / "plugin" / "static" / "ui" / "dist"
    dist_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text(
        '<html><head><!-- deepwiki_ui_config -->'
        '<script type="module" src="./assets/index.js"></script>'
        '<link rel="stylesheet" href="./assets/index.css">'
        "</head><body></body></html>",
        encoding="utf-8",
    )
    fake_routes_file = tmp_path / "plugin" / "routes" / "ui.py"
    fake_routes_file.parent.mkdir(parents=True)
    fake_routes_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(ui_module, "__file__", str(fake_routes_file))

    route = ui_module.Route()

    # Direct route with project_id
    fake_flask.request.headers = {}
    resp = route.ui_route(project_id=2, toolkit_id=15)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert '"base_uri": "/ui/2/15"' in body
    assert 'src="/ui/2/15/assets/index.js"' in body
    assert 'href="/ui/2/15/assets/index.css"' in body

    # Proxy route with valid numeric X-Project-Id
    fake_flask.request.headers = {"X-Project-Id": "42"}
    resp = route.ui_route(project_id=None, toolkit_id=15)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert '"base_uri": "/app/ui_host/wikis/ui/42/15"' in body
    assert 'src="/app/ui_host/wikis/ui/42/15/assets/index.js"' in body

    # Proxy route with empty X-Project-Id falls back cleanly
    fake_flask.request.headers = {"X-Project-Id": ""}
    resp = route.ui_route(project_id=None, toolkit_id=15)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert json.loads(body.split("window.deepwiki_ui_config = ")[1].split(";</script>")[0]) == {
        "base_uri": "/ui/15"
    }

    # Malicious non-numeric X-Project-Id values are rejected with 400
    for bad_header in [
        '1"></script><script>alert(1)</script>',
        "12/../../evil",
        "abc",
        "-1",
        "1.5",
    ]:
        fake_flask.request.headers = {"X-Project-Id": bad_header}
        resp_or_tuple = route.ui_route(project_id=None, toolkit_id=15)
        assert isinstance(resp_or_tuple, tuple)
        body, status = resp_or_tuple
        assert status == 400
        assert bad_header not in body


def test_extract_hostname_and_host_matches_domain():
    assert extract_hostname("https://GitHub.COM:443/owner/repo.git") == "github.com"
    assert extract_hostname("git@github.com:owner/repo.git") == "github.com"
    assert extract_hostname("api.github.com.") == "api.github.com"
    assert extract_hostname("https://user:pass@bitbucket.example.com:7990/scm/p/r") == "bitbucket.example.com"
    assert extract_hostname("") == ""

    assert host_matches_domain("https://github.com/o/r", "github.com") is True
    assert host_matches_domain("https://api.github.com/repos", "github.com") is True
    assert host_matches_domain("https://github.com.evil.com/o/r", "github.com") is False
    assert host_matches_domain("https://evil-github.com/o/r", "github.com") is False
    assert host_matches_domain("https://evil.com/github.com/o/r", "github.com") is False


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://github.com/owner/repo", ProviderType.GITHUB),
        ("git@github.com:owner/repo.git", ProviderType.GITHUB),
        ("https://github.corp.example/owner/repo", ProviderType.GITHUB),
        ("https://gitlab.com/group/project", ProviderType.GITLAB),
        ("https://gitlab.corp.example/group/project", ProviderType.GITLAB),
        ("https://bitbucket.org/workspace/repo", ProviderType.BITBUCKET),
        ("https://bitbucket.corp.example:7990/scm/proj/repo.git", ProviderType.BITBUCKET),
        ("https://dev.azure.com/myorg/MyProject/_git/repo", ProviderType.AZURE_DEVOPS),
        ("https://myorg.visualstudio.com/MyProject/_git/repo", ProviderType.AZURE_DEVOPS),
        # Spoofed hosts / path substrings must NOT match
        ("https://evil.com/github.com/owner/repo", None),
        ("https://github.com.evil.com/owner/repo", None),
        ("https://evil.com/gitlab.com/group/project", None),
        ("https://gitlab.com.evil.com/group/project", None),
        ("https://evil.com/bitbucket.org/workspace/repo", None),
        ("https://bitbucket.org.evil.com/workspace/repo", None),
        ("https://evil.com/dev.azure.com/myorg/proj/_git/repo", None),
        ("https://evil.com/visualstudio.com/proj/_git/repo", None),
    ],
)
def test_provider_detect_from_url_checks_hostname_not_substring(url, expected):
    assert ProviderType.detect_from_url(url) is expected


def test_factory_and_bitbucket_cloud_detection_rejects_url_substring_spoofing():
    gh_cloud = RepoProviderFactory._normalize_config(
        ProviderType.GITHUB, {"base_url": "https://api.github.com"}, "o/r", "main", None
    )
    assert gh_cloud.is_cloud is True
    for spoofed in ("https://evil.com/api.github.com", "https://api.github.com.evil.com"):
        gh_spoof = RepoProviderFactory._normalize_config(
            ProviderType.GITHUB, {"base_url": spoofed}, "o/r", "main", None
        )
        assert gh_spoof.is_cloud is False

    gl_cloud = RepoProviderFactory._normalize_config(
        ProviderType.GITLAB, {"url": "https://gitlab.com"}, "g/p", "main", None
    )
    assert gl_cloud.is_cloud is True
    for spoofed in ("https://evil.com/gitlab.com", "https://gitlab.com.evil.com"):
        gl_spoof = RepoProviderFactory._normalize_config(
            ProviderType.GITLAB, {"url": spoofed}, "g/p", "main", None
        )
        assert gl_spoof.is_cloud is False

    bb_cloud = RepoProviderFactory._normalize_config(
        ProviderType.BITBUCKET, {"url": "https://bitbucket.org"}, "r", "main", "ws"
    )
    assert bb_cloud.is_cloud is True
    for spoofed in ("https://evil.com/bitbucket.org", "https://bitbucket.org.evil.com"):
        bb_spoof = RepoProviderFactory._normalize_config(
            ProviderType.BITBUCKET, {"url": spoofed}, "r", "main", "PROJ"
        )
        assert bb_spoof.is_cloud is False

    provider = BitbucketProvider()
    cfg_cloud = provider.build_clone_config(
        ProviderConfig(
            provider=ProviderType.BITBUCKET,
            host="bitbucket.org",
            api_url="https://bitbucket.org",
            repository="repo",
            project="ws",
        )
    )
    assert cfg_cloud.metadata["is_cloud"] is True
    assert "/scm/" not in cfg_cloud.clone_url

    cfg_spoof = provider.build_clone_config(
        ProviderConfig(
            provider=ProviderType.BITBUCKET,
            host="bitbucket.org.evil.com",
            api_url="https://bitbucket.org.evil.com",
            repository="repo",
            project="PROJ",
        )
    )
    assert cfg_spoof.metadata["is_cloud"] is False
    assert "/scm/PROJ/repo.git" in cfg_spoof.clone_url
