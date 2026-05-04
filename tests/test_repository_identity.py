"""Regression tests for canonical repository and artifact identity."""

import sys
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _THIS_DIR.parent

if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))


def test_multisegment_repo_identifier_normalizes_to_wiki_id():
    from plugin_implementation.artifact_export import normalize_wiki_id as export_normalize_wiki_id
    from plugin_implementation.registry_manager import normalize_wiki_id

    repo_identifier = "epameliteatest/TestProject/mb-java:mb-java:6f785643"

    assert normalize_wiki_id(repo_identifier) == "epameliteatest--testproject--mb-java--mb-java"
    assert (
        export_normalize_wiki_id(repository="epameliteatest/TestProject/mb-java", branch="mb-java")
        == "epameliteatest--testproject--mb-java--mb-java"
    )


def test_visualstudio_url_normalizes_with_org_project_repo():
    from plugin_implementation.artifact_export import normalize_wiki_id as export_normalize_wiki_id

    assert (
        export_normalize_wiki_id(
            repository="https://epameliteatest.visualstudio.com/TestProject/_git/TestRepo",
            branch="main",
        )
        == "epameliteatest--testproject--testrepo--main"
    )
    assert (
        export_normalize_wiki_id(
            repository="https://token@epameliteatest.visualstudio.com/TestProject/_git/TestRepo",
            branch="main",
        )
        == "epameliteatest--testproject--testrepo--main"
    )


def test_ado_clone_config_repo_identifier_keeps_org_project_repo():
    from plugin_implementation.repo_providers.factory import RepoProviderFactory
    from tests.test_invoke_repo_config import _make_ado_config

    clone_config = RepoProviderFactory.from_toolkit_config(
        provider_type="ado_repos",
        config=_make_ado_config(),
        repository="mb-java",
        branch="mb-java",
        project="TestProject",
    )

    assert clone_config.repo_identifier == "epameliteatest/TestProject/mb-java"
    assert clone_config.provider.value == "ado"


def test_build_repo_identifier_prefers_clone_config_identifier():
    from plugin_implementation.repository_identity import build_repo_identifier
    from plugin_implementation.repo_providers.factory import RepoProviderFactory
    from tests.test_invoke_repo_config import _make_ado_config

    clone_config = RepoProviderFactory.from_toolkit_config(
        provider_type="ado_repos",
        config=_make_ado_config(),
        repository="mb-java",
        branch="mb-java",
        project="TestProject",
    )

    repo_identifier = build_repo_identifier(
        repository="TestProject",
        branch="mb-java",
        commit_hash="6f785643ffff0000",
        clone_config=clone_config,
    )

    assert repo_identifier == "epameliteatest/TestProject/mb-java:mb-java:6f785643"


def test_build_query_repo_identifier_prefers_provider_normalized_repo_config():
    from plugin_implementation.repository_identity import build_query_repo_identifier
    from tests.test_invoke_repo_config import _make_ado_config

    repo_identifier = build_query_repo_identifier(
        repository="mb-java",
        branch="mb-java",
        repo_config={
            "provider_type": "ado_repos",
            "provider_config": _make_ado_config(),
            "project": "TestProject",
        },
    )

    assert repo_identifier == "epameliteatest/TestProject/mb-java:mb-java"


def test_rebase_artifact_name_removes_stale_wiki_prefix():
    from plugin_implementation.repository_identity import rebase_artifact_name

    wiki_id = "epameliteatest--testproject--mb-java--mb-java"

    assert (
        rebase_artifact_name(
            "epameliteatest--testproject--mb-java/wiki_pages/README.md",
            wiki_id=wiki_id,
            subfolder="wiki_pages",
        )
        == f"{wiki_id}/wiki_pages/README.md"
    )
    assert (
        rebase_artifact_name("README.md", wiki_id=wiki_id, subfolder="wiki_pages")
        == f"{wiki_id}/wiki_pages/README.md"
    )