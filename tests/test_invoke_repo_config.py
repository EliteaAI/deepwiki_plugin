"""Regression tests for repo config extraction in methods.invoke."""

import importlib
import sys
import types
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _THIS_DIR.parent

if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))


def _load_invoke_module():
    """Import methods.invoke with minimal Pylon stubs."""
    pylon = types.ModuleType('pylon')
    core = types.ModuleType('pylon.core')
    tools = types.ModuleType('pylon.core.tools')

    class _Log:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    class _Web:
        def method(self, *args, **kwargs):
            return lambda fn: fn

        def init(self, *args, **kwargs):
            return lambda fn: fn

        def route(self, *args, **kwargs):
            return lambda fn: fn

    tools.log = _Log()
    tools.web = _Web()

    sys.modules['pylon'] = pylon
    sys.modules['pylon.core'] = core
    sys.modules['pylon.core.tools'] = tools
    sys.modules.pop('methods.invoke', None)
    return importlib.import_module('methods.invoke')


def _make_ado_config():
    return {
        'private': True,
        'alita_title': 'ado_eliteatest',
        'token': 'token_value',
        'project': 'TestProject',
        'organization_url': 'https://dev.azure.com/epameliteatest/',
        'configuration_uuid': 'e976a057-0f27-43b8-9767-853e00d57eff',
        'configuration_project_id': 2,
        'configuration_type': 'ado',
    }


def test_extract_repo_config_from_nested_ado_toolkit_settings():
    invoke = _load_invoke_module()

    params = {
        'code_toolkit': {
            'settings': {
                'ado_configuration': _make_ado_config(),
                'repository_id': 'TestRepo',
                'base_branch': 'main',
                'active_branch': 'develop',
            }
        }
    }

    repo_config = invoke._extract_repo_config_from_toolkit(params)

    assert repo_config['provider_type'] == 'ado_repos'
    assert repo_config['provider_config']['organization_url'] == 'https://dev.azure.com/epameliteatest/'
    assert repo_config['repository'] == 'TestRepo'
    assert repo_config['branch'] == 'develop'
    assert repo_config['project'] == 'TestProject'


def test_extract_repo_config_from_flat_top_level_ado_payload():
    invoke = _load_invoke_module()

    params = {
        'ado_configuration': _make_ado_config(),
        'repository_id': 'TestRepo',
        'base_branch': 'main',
        'active_branch': 'develop',
    }

    repo_config = invoke._extract_repo_config_from_toolkit(params)

    assert repo_config['provider_type'] == 'ado_repos'
    assert repo_config['provider_config']['organization_url'] == 'https://dev.azure.com/epameliteatest/'
    assert repo_config['repository'] == 'TestRepo'
    assert repo_config['branch'] == 'develop'
    assert repo_config['project'] == 'TestProject'


def test_extract_repo_config_from_prefixed_ado_toolkit_payload():
    invoke = _load_invoke_module()

    params = {
        'toolkit_configuration_ado_configuration': _make_ado_config(),
        'toolkit_configuration_repository_id': 'mb-java',
        'toolkit_configuration_base_branch': 'mb-java',
    }

    repo_config = invoke._extract_repo_config_from_toolkit(params)

    assert repo_config['provider_type'] == 'ado_repos'
    assert repo_config['provider_config']['organization_url'] == 'https://dev.azure.com/epameliteatest/'
    assert repo_config['repository'] == 'mb-java'
    assert repo_config['branch'] == 'mb-java'
    assert repo_config['project'] == 'TestProject'


def test_extract_repo_config_from_flat_code_toolkit_payload_without_settings_wrapper():
    invoke = _load_invoke_module()

    params = {
        'code_toolkit': {
            'ado_configuration': _make_ado_config(),
            'repository_id': 'TestRepo',
            'base_branch': 'main',
            'active_branch': 'develop',
        }
    }

    repo_config = invoke._extract_repo_config_from_toolkit(params)

    assert repo_config['provider_type'] == 'ado_repos'
    assert repo_config['provider_config']['organization_url'] == 'https://dev.azure.com/epameliteatest/'
    assert repo_config['repository'] == 'TestRepo'
    assert repo_config['branch'] == 'develop'
    assert repo_config['project'] == 'TestProject'


def test_extract_repo_config_merges_split_ado_settings_and_toolkit_config():
    invoke = _load_invoke_module()

    params = {
        'code_toolkit': {
            'toolkit_config': {
                'ado_configuration': _make_ado_config(),
            },
            'settings': {
                'ado_configuration': {
                    'elitea_title': 'Ado_test',
                    'private': True,
                },
                'repository_id': 'TestRepo',
                'active_branch': 'develop',
            },
        }
    }

    repo_config = invoke._extract_repo_config_from_toolkit(params)

    assert repo_config['provider_type'] == 'ado_repos'
    assert repo_config['provider_config']['organization_url'] == 'https://dev.azure.com/epameliteatest/'
    assert repo_config['provider_config']['elitea_title'] == 'Ado_test'
    assert repo_config['repository'] == 'TestRepo'
    assert repo_config['branch'] == 'develop'
    assert repo_config['project'] == 'TestProject'


def test_extract_repo_config_from_legacy_toolkit_configuration_code_repository_payload():
    invoke = _load_invoke_module()

    params = {
        'toolkit_configuration_code_repository': {
            'ado_configuration': _make_ado_config(),
            'repository_id': 'TestRepo',
            'active_branch': 'develop',
        }
    }

    repo_config = invoke._extract_repo_config_from_toolkit(params)

    assert repo_config['provider_type'] == 'ado_repos'
    assert repo_config['provider_config']['organization_url'] == 'https://dev.azure.com/epameliteatest/'
    assert repo_config['repository'] == 'TestRepo'
    assert repo_config['branch'] == 'develop'
    assert repo_config['project'] == 'TestProject'


def test_repo_provider_factory_accepts_ado_repos_alias():
    from plugin_implementation.repo_providers.factory import RepoProviderFactory

    clone_config = RepoProviderFactory.from_toolkit_config(
        provider_type='ado_repos',
        config=_make_ado_config(),
        repository='TestProject',
        branch='main',
        project='TestProject',
    )

    assert clone_config.provider.value == 'ado'
    assert clone_config.repo_identifier == 'epameliteatest/TestProject/TestProject'
    assert clone_config.branch == 'main'