"""Regression tests for #7273: empty branch must resolve to the repository's
real default branch, never a guessed ``'main'``.

Since EL-6610 the ADO Repos toolkit can hold several repositories, so users
commonly leave the base/active branch empty. These tests cover:
  - the pure helpers in ``plugin_implementation.branch_resolution``;
  - ``methods.invoke._extract_repo_config_from_toolkit`` no longer defaulting
    branch to 'main';
  - ``methods.invoke._resolve_branch`` priority order;
  - dispatch-level fail-fast behaviour for ask/deep_research and the
    generate_wiki pass-through;
  - a real (local, no-network) git clone whose default branch is 'master',
    proving the clone/path/identifier never contain 'None' or a guessed
    'main'.
"""

import json
import os
import subprocess
import sys
import types

import pytest

from test_invoke_repo_config import _load_invoke_module, _make_ado_config


# ---------------------------------------------------------------------------
# branch_resolution unit tests
# ---------------------------------------------------------------------------

from plugin_implementation.branch_resolution import (
    normalize_branch,
    parse_symref_head,
    select_configured_branch,
)


@pytest.mark.parametrize(
    'value, expected',
    [
        (None, None),
        ('', None),
        ('   ', None),
        ('master', 'master'),
        ('  master  ', 'master'),
        ('refs/heads/master', 'master'),
        ('refs/heads/feature/x', 'feature/x'),
        (123, None),
        ([], None),
        (['main'], None),
    ],
)
def test_normalize_branch(value, expected):
    assert normalize_branch(value) == expected


def test_select_configured_branch_picks_first_non_empty():
    assert select_configured_branch(None, '', '  ', 'dev') == 'dev'
    assert select_configured_branch('main', 'dev') == 'main'
    assert select_configured_branch(None, '', '   ') is None
    assert select_configured_branch() is None


@pytest.mark.parametrize(
    'output, expected',
    [
        ('ref: refs/heads/master\tHEAD\nabc123\tHEAD\n', 'master'),
        ('ref: refs/heads/feature/x\tHEAD\n', 'feature/x'),
        ('abc123\tHEAD\n', None),
        ('', None),
        (None, None),
        ('ref: refs/tags/v1\tHEAD\n', None),
    ],
)
def test_parse_symref_head(output, expected):
    assert parse_symref_head(output) == expected


def test_resolve_remote_default_branch_returns_none_on_missing_git(monkeypatch):
    from plugin_implementation import branch_resolution

    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(branch_resolution.subprocess, 'run', _raise)
    assert branch_resolution.resolve_remote_default_branch('https://example.com/repo.git') is None


def test_resolve_remote_default_branch_returns_none_on_timeout(monkeypatch):
    from plugin_implementation import branch_resolution

    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd='git', timeout=30)

    monkeypatch.setattr(branch_resolution.subprocess, 'run', _raise)
    assert branch_resolution.resolve_remote_default_branch('https://example.com/repo.git') is None


def test_resolve_remote_default_branch_returns_none_for_empty_url():
    from plugin_implementation.branch_resolution import resolve_remote_default_branch

    assert resolve_remote_default_branch('') is None
    assert resolve_remote_default_branch(None) is None


def test_read_checked_out_branch_returns_none_on_detached_head(tmp_path):
    from plugin_implementation.branch_resolution import read_checked_out_branch

    # Not even a git repo - simulates any failure path.
    assert read_checked_out_branch(str(tmp_path)) is None


def test_indexed_branches_reads_refs_for_exact_repo_path(tmp_path):
    from plugin_implementation.branch_resolution import indexed_branches
    from plugin_implementation.repo_resolution import save_cache_index_atomic

    index = {
        "refs": {
            "owner/repo:master": "owner/repo:master:aaaaaaaa",
            "owner/repo:dev": "owner/repo:dev:bbbbbbbb",
            "owner/other:main": "owner/other:main:cccccccc",
        }
    }
    save_cache_index_atomic(tmp_path, index)

    branches = indexed_branches(str(tmp_path), "owner/repo")
    assert sorted(branches) == ["dev", "master"]

    assert indexed_branches(str(tmp_path), "owner/other") == ["main"]
    assert indexed_branches(str(tmp_path), "owner/missing") == []


def test_indexed_branches_handles_missing_cache_index(tmp_path):
    from plugin_implementation.branch_resolution import indexed_branches

    assert indexed_branches(str(tmp_path), "owner/repo") == []


# ---------------------------------------------------------------------------
# _extract_repo_config_from_toolkit: old-style configs unchanged
# ---------------------------------------------------------------------------


def _code_toolkit_settings(**settings):
    return {'code_toolkit': {'settings': settings}}


def test_extract_repo_config_github_branch_configured_unchanged():
    invoke = _load_invoke_module()
    params = _code_toolkit_settings(
        github_configuration={'base_url': 'https://api.github.com', 'access_token': 't'},
        repository='owner/repo',
        active_branch='develop',
    )
    repo_config = invoke._extract_repo_config_from_toolkit(params)
    assert repo_config['provider_type'] == 'github'
    assert repo_config['repository'] == 'owner/repo'
    assert repo_config['branch'] == 'develop'


def test_extract_repo_config_gitlab_branch_configured_unchanged():
    invoke = _load_invoke_module()
    params = _code_toolkit_settings(
        gitlab_configuration={'url': 'https://gitlab.com', 'private_token': 't'},
        repository='group/project',
        branch='release',
    )
    repo_config = invoke._extract_repo_config_from_toolkit(params)
    assert repo_config['provider_type'] == 'gitlab'
    assert repo_config['branch'] == 'release'


def test_extract_repo_config_bitbucket_branch_configured_unchanged():
    invoke = _load_invoke_module()
    params = _code_toolkit_settings(
        bitbucket_configuration={'url': 'https://bitbucket.org'},
        repository='ws/repo',
        active_branch='main',
    )
    repo_config = invoke._extract_repo_config_from_toolkit(params)
    assert repo_config['provider_type'] == 'bitbucket'
    assert repo_config['branch'] == 'main'


def test_extract_repo_config_ado_string_repository_with_branch_unchanged():
    invoke = _load_invoke_module()
    params = _code_toolkit_settings(
        ado_configuration=_make_ado_config(),
        repository_id='TestRepo',
        base_branch='main',
    )
    repo_config = invoke._extract_repo_config_from_toolkit(params)
    assert repo_config['provider_type'] == 'ado_repos'
    assert repo_config['repository'] == 'TestRepo'
    assert repo_config['branch'] == 'main'


@pytest.mark.parametrize('repository_id', ['', None, '   ', [], [None], ['  ']])
def test_extract_repo_config_ado_list_repository_no_branch_yields_none(repository_id):
    invoke = _load_invoke_module()
    params = _code_toolkit_settings(
        ado_configuration=_make_ado_config(),
        repository_id=repository_id,
    )
    repo_config = invoke._extract_repo_config_from_toolkit(params)
    assert repo_config['branch'] is None


@pytest.mark.parametrize(
    'branch_value',
    ['', None, '   ', 'refs/heads/'],
)
def test_extract_repo_config_ado_empty_branch_variants_yield_none(branch_value):
    invoke = _load_invoke_module()
    params = _code_toolkit_settings(
        ado_configuration=_make_ado_config(),
        repository_id='TestRepo',
        active_branch=branch_value,
        base_branch=branch_value,
        branch=branch_value,
    )
    repo_config = invoke._extract_repo_config_from_toolkit(params)
    assert repo_config['branch'] is None


def test_extract_repo_config_toolkit_configuration_prefixed_branch_variants():
    invoke = _load_invoke_module()
    params = {
        'code_toolkit': {
            'toolkit_configuration_ado_configuration': _make_ado_config(),
            'toolkit_configuration_repository_id': ['TestRepo'],
            'toolkit_configuration_base_branch': 'release/2.0',
        }
    }
    repo_config = invoke._extract_repo_config_from_toolkit(params)
    assert repo_config['branch'] == 'release/2.0'


def test_extract_repo_config_legacy_direct_params_no_branch_yields_none():
    invoke = _load_invoke_module()
    params = {'github_repository': 'owner/repo'}
    repo_config = invoke._extract_repo_config_from_toolkit(params)
    assert repo_config['branch'] is None
    assert repo_config['repository'] == 'owner/repo'


def test_extract_repo_config_legacy_direct_params_branch_configured_unchanged():
    invoke = _load_invoke_module()
    params = {'github_repository': 'owner/repo', 'github_base_branch': 'develop'}
    repo_config = invoke._extract_repo_config_from_toolkit(params)
    assert repo_config['branch'] == 'develop'


def test_extract_repo_config_fallback_github_structure_no_branch_yields_none():
    invoke = _load_invoke_module()
    # No *_configuration key present -> goes through the "Fallback: assume
    # GitHub with legacy structure" branch inside the "else" clause.
    params = {
        'code_toolkit': {
            'settings': {
                'repository': 'owner/repo',
            }
        }
    }
    repo_config = invoke._extract_repo_config_from_toolkit(params)
    assert repo_config['provider_type'] == 'github'
    assert repo_config['branch'] is None


# ---------------------------------------------------------------------------
# _resolve_branch priority: configured > override > remote > single cached
# ---------------------------------------------------------------------------


def _repo_config(repository='owner/repo', branch=None, provider_type='github'):
    return {
        'provider_type': provider_type,
        'provider_config': {'base_url': 'https://api.github.com'},
        'repository': repository,
        'branch': branch,
        'project': None,
    }


def test_resolve_branch_prefers_already_configured_branch(monkeypatch):
    invoke = _load_invoke_module()

    called = {'ls_remote': False}
    monkeypatch.setattr(
        invoke, 'resolve_remote_default_branch',
        lambda *a, **k: called.update(ls_remote=True) or 'should-not-be-used',
    )

    repo_config = _repo_config(branch='configured-branch')
    invoke._resolve_branch(repo_config)

    assert repo_config['branch'] == 'configured-branch'
    assert called['ls_remote'] is False


def test_resolve_branch_uses_override_branch_when_present(monkeypatch):
    invoke = _load_invoke_module()

    called = {'ls_remote': False}
    monkeypatch.setattr(
        invoke, 'resolve_remote_default_branch',
        lambda *a, **k: called.update(ls_remote=True) or 'should-not-be-used',
    )

    repo_config = _repo_config(branch=None)
    invoke._resolve_branch(repo_config, repo_identifier_override='owner/repo:override-branch:abcd1234')

    assert repo_config['branch'] == 'override-branch'
    assert called['ls_remote'] is False


def test_resolve_branch_falls_back_to_remote_default(monkeypatch):
    invoke = _load_invoke_module()

    monkeypatch.setattr(invoke, 'resolve_remote_default_branch', lambda *a, **k: 'remote-default')

    repo_config = _repo_config(branch=None)
    invoke._resolve_branch(repo_config)

    assert repo_config['branch'] == 'remote-default'


def test_resolve_branch_falls_back_to_single_indexed_branch(monkeypatch, tmp_path):
    invoke = _load_invoke_module()

    monkeypatch.setattr(invoke, 'resolve_remote_default_branch', lambda *a, **k: None)
    monkeypatch.setattr(invoke, 'indexed_branches', lambda cache_dir, repo_path: ['only-indexed-branch'])

    repo_config = _repo_config(branch=None)
    invoke._resolve_branch(repo_config, cache_dir=str(tmp_path))

    assert repo_config['branch'] == 'only-indexed-branch'


def test_resolve_branch_stays_none_with_two_cached_branches(monkeypatch, tmp_path):
    invoke = _load_invoke_module()

    monkeypatch.setattr(invoke, 'resolve_remote_default_branch', lambda *a, **k: None)
    monkeypatch.setattr(invoke, 'indexed_branches', lambda cache_dir, repo_path: ['branch-a', 'branch-b'])

    repo_config = _repo_config(branch=None)
    invoke._resolve_branch(repo_config, cache_dir=str(tmp_path))

    assert repo_config['branch'] is None


def test_resolve_branch_stays_none_without_repository():
    invoke = _load_invoke_module()

    repo_config = _repo_config(repository=None, branch=None)
    invoke._resolve_branch(repo_config)

    assert repo_config['branch'] is None


# ---------------------------------------------------------------------------
# Dispatch-level tests (perform_invoke_request)
# ---------------------------------------------------------------------------


class _FakeDispatchMethod:
    """Minimal ``self`` for ``perform_invoke_request`` dispatch tests."""

    _create_error_response = _load_invoke_module().Method._create_error_response

    def __init__(self, base_path):
        self._base_path = base_path
        self.generate_wiki_calls = []
        self.ask_calls = []
        self.deep_research_calls = []

    def invocation_stop_checkpoint(self):
        pass

    def runtime_config(self):
        return {"base_path": self._base_path}

    def generate_wiki(self, **kwargs):
        self.generate_wiki_calls.append(kwargs)
        return {'success': True, 'result': 'ok'}

    def ask(self, **kwargs):
        self.ask_calls.append(kwargs)
        return {'success': True, 'answer': 'ok'}

    def deep_research(self, **kwargs):
        self.deep_research_calls.append(kwargs)
        return {'success': True, 'report': 'ok'}


@pytest.fixture
def dispatch_fixture(monkeypatch, tmp_path):
    invoke = _load_invoke_module()
    monkeypatch.setitem(sys.modules, 'tasknode_task', types.SimpleNamespace(id='invocation-7273'))
    fake = _FakeDispatchMethod(str(tmp_path))
    fake._create_error_response = types.MethodType(invoke.Method._create_error_response, fake)
    return invoke, fake


def _ado_params(repository_id='TestRepo', branch=None, **tool_params):
    settings = {
        'ado_configuration': _make_ado_config(),
        'repository_id': repository_id,
    }
    if branch is not None:
        settings['base_branch'] = branch
    return {
        'code_toolkit': {'settings': settings},
        **tool_params,
    }


def test_dispatch_ask_fails_fast_when_branch_unresolvable(dispatch_fixture, monkeypatch):
    invoke, fake = dispatch_fixture
    monkeypatch.setattr(invoke, 'resolve_remote_default_branch', lambda *a, **k: None)

    request_data = {
        'configuration': {'parameters': _ado_params(repository_id='TestRepo')},
        'parameters': {'question': 'What does it do?'},
    }

    response = invoke.Method.perform_invoke_request(fake, 'wikis', 'ask', request_data)

    assert response['status'] == 'Error'
    message = json.loads(response['result'])[0]['data']
    assert "Could not determine the branch of repository 'TestRepo'" in message
    assert 'Specify a branch in the toolkit' in message
    assert fake.ask_calls == []


def test_dispatch_deep_research_fails_fast_when_branch_unresolvable(dispatch_fixture, monkeypatch):
    invoke, fake = dispatch_fixture
    monkeypatch.setattr(invoke, 'resolve_remote_default_branch', lambda *a, **k: None)

    request_data = {
        'configuration': {'parameters': _ado_params(repository_id='TestRepo')},
        'parameters': {'question': 'Explain the architecture'},
    }

    response = invoke.Method.perform_invoke_request(fake, 'wikis', 'deep_research', request_data)

    assert response['status'] == 'Error'
    message = json.loads(response['result'])[0]['data']
    assert "Could not determine the branch of repository 'TestRepo'" in message
    assert fake.deep_research_calls == []


def test_dispatch_ask_resolves_branch_via_remote_default(dispatch_fixture, monkeypatch):
    invoke, fake = dispatch_fixture
    monkeypatch.setattr(invoke, 'resolve_remote_default_branch', lambda *a, **k: 'master')

    request_data = {
        'configuration': {'parameters': _ado_params(repository_id='TestRepo')},
        'parameters': {'question': 'What does it do?'},
    }

    response = invoke.Method.perform_invoke_request(fake, 'wikis', 'ask', request_data)

    assert response['status'] == 'Completed'
    assert len(fake.ask_calls) == 1
    assert fake.ask_calls[0]['repo_config']['branch'] == 'master'


def test_dispatch_generate_wiki_passes_resolved_branch_and_active_branch(dispatch_fixture, monkeypatch):
    invoke, fake = dispatch_fixture
    monkeypatch.setattr(invoke, 'resolve_remote_default_branch', lambda *a, **k: 'master')

    request_data = {
        'configuration': {'parameters': _ado_params(repository_id='TestRepo')},
        'parameters': {'query': 'Build wiki'},
    }

    response = invoke.Method.perform_invoke_request(fake, 'wikis', 'generate_wiki', request_data)

    assert response['status'] == 'Completed'
    assert len(fake.generate_wiki_calls) == 1
    call = fake.generate_wiki_calls[0]
    assert call['repo_config']['branch'] == 'master'
    assert call['active_branch'] == 'master'


def test_dispatch_generate_wiki_leaves_branch_none_when_unresolvable(dispatch_fixture, monkeypatch):
    invoke, fake = dispatch_fixture
    monkeypatch.setattr(invoke, 'resolve_remote_default_branch', lambda *a, **k: None)

    request_data = {
        'configuration': {'parameters': _ado_params(repository_id='TestRepo')},
        'parameters': {'query': 'Build wiki'},
    }

    response = invoke.Method.perform_invoke_request(fake, 'wikis', 'generate_wiki', request_data)

    # generate_wiki is allowed to proceed with branch=None; the clone-time
    # fallback (local_repository_manager) resolves it from the remote.
    assert response['status'] == 'Completed'
    assert len(fake.generate_wiki_calls) == 1
    call = fake.generate_wiki_calls[0]
    assert call['repo_config']['branch'] is None
    assert call['active_branch'] is None


def test_dispatch_configured_branch_never_triggers_ls_remote(dispatch_fixture, monkeypatch):
    invoke, fake = dispatch_fixture

    called = {'ls_remote': False}

    def _fail_if_called(*args, **kwargs):
        called['ls_remote'] = True
        return 'should-not-be-called'

    monkeypatch.setattr(invoke, 'resolve_remote_default_branch', _fail_if_called)

    request_data = {
        'configuration': {'parameters': _ado_params(repository_id='TestRepo', branch='already-configured')},
        'parameters': {'question': 'What does it do?'},
    }

    response = invoke.Method.perform_invoke_request(fake, 'wikis', 'ask', request_data)

    assert response['status'] == 'Completed'
    assert called['ls_remote'] is False
    assert fake.ask_calls[0]['repo_config']['branch'] == 'already-configured'


# ---------------------------------------------------------------------------
# Real git, no network: local bare repo with default branch 'master'
# ---------------------------------------------------------------------------


def _run_git(args, cwd=None):
    result = subprocess.run(
        ['git'] + args, cwd=cwd, capture_output=True, text=True, check=True,
    )
    return result.stdout


@pytest.fixture
def local_master_repo(tmp_path):
    """A local bare repo whose default branch is 'master', seeded from a
    temp work repo with one commit. Returns the bare repo's ``file://`` URL.
    """
    work_dir = tmp_path / "work"
    bare_dir = tmp_path / "bare.git"
    work_dir.mkdir()

    _run_git(['init', '-q'], cwd=str(work_dir))
    _run_git(['config', 'user.email', 'test@example.com'], cwd=str(work_dir))
    _run_git(['config', 'user.name', 'Test'], cwd=str(work_dir))
    (work_dir / "README.md").write_text("hello\n")
    _run_git(['add', 'README.md'], cwd=str(work_dir))
    _run_git(['commit', '-q', '-m', 'init'], cwd=str(work_dir))
    _run_git(['branch', '-M', 'master'], cwd=str(work_dir))

    bare_url = f"file://{bare_dir}"
    _run_git(['init', '-q', '--bare', '-b', 'master', str(bare_dir)])
    _run_git(['remote', 'add', 'origin', bare_url], cwd=str(work_dir))
    _run_git(['push', '-q', 'origin', 'master'], cwd=str(work_dir))

    return bare_url


def test_resolve_remote_default_branch_real_git_master(local_master_repo):
    from plugin_implementation.branch_resolution import resolve_remote_default_branch

    assert resolve_remote_default_branch(local_master_repo) == 'master'


def test_local_repository_manager_clones_none_branch_and_reports_master(tmp_path, local_master_repo):
    from plugin_implementation.local_repository_manager import LocalRepositoryManager
    from plugin_implementation.repo_providers.models import GitCloneConfig, ProviderType

    cache_dir = tmp_path / "repo_cache"
    clone_config = GitCloneConfig(
        clone_url=local_master_repo,
        repo_identifier="owner/repo",
        branch=None,
        provider=ProviderType.GITHUB,
    )
    manager = LocalRepositoryManager(cache_dir=str(cache_dir), clone_config=clone_config)

    local_path = manager.get_repository_local_path("owner/repo", branch=None)

    assert os.path.isdir(local_path)
    assert "None" not in local_path
    assert "master" in local_path
    assert "_main_" not in local_path

    info = manager.get_repository_info(local_path)
    assert info['branch'] == 'master'


def test_local_repository_manager_no_branch_clone_when_ls_remote_unresolvable(tmp_path, local_master_repo, monkeypatch):
    """When ls-remote resolution is monkeypatched to None, the manager must
    clone WITHOUT --branch and still report the real checked-out branch via
    read_checked_out_branch.
    """
    from plugin_implementation import local_repository_manager as lrm_module
    from plugin_implementation.local_repository_manager import LocalRepositoryManager
    from plugin_implementation.repo_providers.models import GitCloneConfig, ProviderType

    monkeypatch.setattr(lrm_module, 'resolve_remote_default_branch', lambda *a, **k: None)

    cache_dir = tmp_path / "repo_cache_2"
    clone_config = GitCloneConfig(
        clone_url=local_master_repo,
        repo_identifier="owner/repo2",
        branch=None,
        provider=ProviderType.GITHUB,
    )
    manager = LocalRepositoryManager(cache_dir=str(cache_dir), clone_config=clone_config)

    local_path = manager.get_repository_local_path("owner/repo2", branch=None)

    assert os.path.isdir(local_path)
    assert "None" not in local_path
    # Path is named after HEAD since the branch could not be pre-resolved.
    assert "_HEAD_" in local_path

    info = manager.get_repository_info(local_path)
    # The clone (without --branch) still checks out the repo's real default
    # branch; read_checked_out_branch must report it correctly.
    assert info['branch'] == 'master'
    assert info['branch'] != 'main'
