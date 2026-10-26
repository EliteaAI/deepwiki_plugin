"""Regression tests for #7186: ADO Repos list-type / empty ``repository_id``.

Since EL-6610 the ADO Repos toolkit stores ``repository_id`` as an optional
list. DeepWiki still works on exactly one repository: the first configured one
is used, and an empty value fails fast with an actionable message.
"""

import json
import sys
import types

import pytest

from test_invoke_repo_config import _load_invoke_module, _make_ado_config

from plugin_implementation.repo_providers.factory import create_clone_config_from_expanded_toolkit
from plugin_implementation.repository_identity import (
    normalize_repositories,
    repository_from_identifier,
    select_single_repository,
)


MISSING_MESSAGE = "The selected ADO Repos toolkit has no repository configured. Specify a repository in the toolkit"


def _ado_params(repository_id, **tool_params):
    return {
        'code_toolkit': {
            'settings': {
                'ado_configuration': _make_ado_config(),
                'repository_id': repository_id,
                'base_branch': 'main',
            }
        },
        **tool_params,
    }


@pytest.mark.parametrize(
    'value, expected',
    [
        (None, []),
        ('', []),
        ('  repo  ', ['repo']),
        ([], []),
        (['repo'], ['repo']),
        (['a', ' ', None, 'b', 'a'], ['a', 'b']),
        (('a', 'b'), ['a', 'b']),
    ],
)
def test_normalize_repositories_accepts_legacy_and_list_shapes(value, expected):
    assert normalize_repositories(value) == expected


def test_select_single_repository_uses_first_configured():
    assert select_single_repository(['first', 'second']) == 'first'
    assert select_single_repository('legacy') == 'legacy'
    assert select_single_repository([]) is None
    assert select_single_repository(None) is None


@pytest.mark.parametrize(
    'identifier, provider_type, expected',
    [
        ('epameliteatest/TestProject/mb-java:main', 'ado_repos', 'mb-java'),
        ('epameliteatest/TestProject/mb-java:main:abcd1234', 'ado_repos', 'mb-java'),
        ('owner/repo:main', 'github', 'owner/repo'),
        ('  ', 'ado_repos', None),
        (None, 'ado_repos', None),
        (['not-a-string'], 'ado_repos', None),
    ],
)
def test_repository_from_identifier(identifier, provider_type, expected):
    assert repository_from_identifier(identifier, provider_type) == expected


@pytest.mark.parametrize(
    'repository_id, expected',
    [
        ('TestRepo', 'TestRepo'),
        (['TestRepo'], 'TestRepo'),
        (['TestRepo', 'Other'], 'TestRepo'),
        ([' ', 'TestRepo'], 'TestRepo'),
        ([], None),
        (None, None),
    ],
)
def test_extract_repo_config_normalizes_ado_repository_id(repository_id, expected):
    invoke = _load_invoke_module()

    repo_config = invoke._extract_repo_config_from_toolkit(_ado_params(repository_id))

    assert repo_config['provider_type'] == 'ado_repos'
    assert repo_config['repository'] == expected


def test_extract_repo_config_normalizes_prefixed_list_repository_id():
    invoke = _load_invoke_module()

    params = {
        'code_toolkit': {
            'toolkit_configuration_ado_configuration': _make_ado_config(),
            'toolkit_configuration_repository_id': ['TestRepo'],
        }
    }

    assert invoke._extract_repo_config_from_toolkit(params)['repository'] == 'TestRepo'


def test_list_repository_builds_same_clone_identity_as_legacy_string():
    from plugin_implementation.repo_providers.factory import RepoProviderFactory

    invoke = _load_invoke_module()
    repo_config = invoke._extract_repo_config_from_toolkit(_ado_params(['TestRepo']))

    clone_config = RepoProviderFactory.from_toolkit_config(
        provider_type=repo_config['provider_type'],
        config=repo_config['provider_config'],
        repository=repo_config['repository'],
        branch=repo_config['branch'],
        project=repo_config['project'],
    )

    assert clone_config.repo_identifier == 'epameliteatest/TestProject/TestRepo'


@pytest.mark.parametrize('repository_id', ['TestRepo', ['TestRepo'], ['TestRepo', 'Other']])
def test_expanded_toolkit_factory_accepts_list_repository_id(repository_id):
    settings = {'ado_configuration': _make_ado_config(), 'repository_id': repository_id}

    clone_config = create_clone_config_from_expanded_toolkit('ado_repos', settings)

    assert clone_config.repo_identifier == 'epameliteatest/TestProject/TestRepo'


def test_missing_repository_error_message_offers_override():
    invoke = _load_invoke_module()

    error = invoke._missing_repository_error({'provider_type': 'ado_repos', 'repository': None})

    assert isinstance(error, ValueError)
    assert str(error) == MISSING_MESSAGE + " or pass `repo_identifier_override`."


def test_missing_repository_error_for_generation_omits_override():
    invoke = _load_invoke_module()

    error = invoke._missing_repository_error(
        {'provider_type': 'ado_repos', 'repository': None}, allow_override=False
    )

    assert str(error) == MISSING_MESSAGE + "."


def test_missing_repository_resolved_from_override():
    invoke = _load_invoke_module()
    repo_config = {'provider_type': 'ado_repos', 'repository': None}

    error = invoke._missing_repository_error(
        repo_config, 'epameliteatest/TestProject/mb-java:main'
    )

    assert error is None
    assert repo_config['repository'] == 'mb-java'


def test_configured_repository_passes_guard():
    invoke = _load_invoke_module()

    assert invoke._missing_repository_error({'provider_type': 'ado_repos', 'repository': 'r'}) is None


class _FakeMethod:
    """Minimal ``self`` for ``perform_invoke_request``; any tool execution is a failure."""

    _create_error_response = _load_invoke_module().Method._create_error_response

    def invocation_stop_checkpoint(self):
        pass

    def _fail(self, *args, **kwargs):
        raise AssertionError("tool must not run when no repository is configured")

    generate_wiki = ask = deep_research = _fail


@pytest.fixture
def invoke_with_task(monkeypatch):
    invoke = _load_invoke_module()
    monkeypatch.setitem(sys.modules, 'tasknode_task', types.SimpleNamespace(id='invocation-7186'))
    fake = _FakeMethod()
    fake._create_error_response = types.MethodType(invoke.Method._create_error_response, fake)
    return invoke, fake


@pytest.mark.parametrize(
    'tool_name, tool_params, expected_suffix',
    [
        ('ask', {'question': 'What does it do?'}, " or pass `repo_identifier_override`."),
        ('deep_research', {'question': 'Explain the architecture'}, " or pass `repo_identifier_override`."),
        ('generate_wiki', {'query': 'Build wiki'}, "."),
    ],
)
@pytest.mark.parametrize('repository_id', [[], None, ['  ']])
def test_tools_fail_fast_without_repository(invoke_with_task, tool_name, tool_params, expected_suffix, repository_id):
    invoke, fake = invoke_with_task
    request_data = {
        'configuration': {'parameters': _ado_params(repository_id)},
        'parameters': tool_params,
    }

    response = invoke.Method.perform_invoke_request(fake, 'wikis', tool_name, request_data)

    assert response['status'] == 'Error'
    assert response['error_category'] == 'invalid_input'
    message = json.loads(response['result'])[0]['data']
    assert message.endswith(MISSING_MESSAGE + expected_suffix)
    assert 'Traceback' not in message
    assert "'list' object has no attribute 'strip'" not in message


@pytest.mark.parametrize('repository_id', [['TestRepo'], ['TestRepo', 'Other'], 'TestRepo'])
def test_ask_receives_single_string_repository(invoke_with_task, repository_id):
    invoke, fake = invoke_with_task
    captured = {}

    def _ask(**kwargs):
        captured.update(kwargs)
        return {'success': True, 'answer': 'ok'}

    fake.ask = _ask
    request_data = {
        'configuration': {'parameters': _ado_params(repository_id)},
        'parameters': {'question': 'What does it do?'},
    }

    response = invoke.Method.perform_invoke_request(fake, 'wikis', 'ask', request_data)

    assert response['status'] == 'Completed'
    assert captured['repo_config']['repository'] == 'TestRepo'
