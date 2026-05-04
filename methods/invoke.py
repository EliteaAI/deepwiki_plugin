import json
import re
import traceback
from typing import List, Optional, Dict, Any
from urllib.parse import quote

import requests

from pylon.core.tools import log  # pylint: disable=E0611,E0401,W0611
from pylon.core.tools import web  # pylint: disable=E0611,E0401,W0611

default_bucket = 'wiki_artifacts'

_TOOLKIT_PROVIDER_KEYS = (
    'github_configuration',
    'gitlab_configuration',
    'bitbucket_configuration',
    'ado_configuration',
)
def _payload_contains_provider_key(params: Dict[str, Any], provider_key: str) -> bool:
    """Check known toolkit payload locations for a provider-specific config key."""
    sources = [
        params,
        params.get('code_toolkit'),
        params.get('toolkit_configuration_code_toolkit'),
        params.get('toolkit_configuration_code_repository'),
        params.get('code_repository'),
    ]

    for source in sources:
        if not isinstance(source, dict):
            continue
        if provider_key in source:
            return True
        for nested in (source.get('settings'), source.get('toolkit_config'), _extract_configuration_parameters(source)):
            if isinstance(nested, dict) and provider_key in nested:
                return True

    return False


def _merge_dicts(*values: Any) -> Dict[str, Any]:
    """Shallow-merge dict values, ignoring non-dicts."""
    merged: Dict[str, Any] = {}
    for value in values:
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _extract_configuration_parameters(source: Dict[str, Any]) -> Dict[str, Any]:
    """Extract toolkit configuration payload from known wrapper shapes."""
    configuration = source.get('configuration')
    if isinstance(configuration, dict) and isinstance(configuration.get('parameters'), dict):
        return configuration.get('parameters') or {}
    return {}


def _merge_provider_configs(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Merge prefixed/unprefixed provider configs that may be split by UI/API shape."""
    merged = dict(settings)
    for provider_key in _TOOLKIT_PROVIDER_KEYS:
        prefixed_key = f'toolkit_configuration_{provider_key}'
        provider_config = _merge_dicts(settings.get(provider_key), settings.get(prefixed_key))
        if provider_config:
            merged[provider_key] = provider_config
            merged[prefixed_key] = provider_config
    return merged


def _merge_toolkit_payload(source: Any) -> Dict[str, Any]:
    """Merge toolkit_config/configuration/settings into one repo-config payload."""
    if not isinstance(source, dict):
        return {}

    wrapper_fields = {key: value for key, value in source.items() if key not in ('settings', 'toolkit_config', 'configuration')}
    sources = (
        wrapper_fields,
        source.get('toolkit_config'),
        _extract_configuration_parameters(source),
        source.get('settings'),
    )
    merged = _merge_dicts(*sources)

    for provider_key in _TOOLKIT_PROVIDER_KEYS:
        prefixed_key = f'toolkit_configuration_{provider_key}'
        provider_config = _merge_dicts(
            *(candidate.get(provider_key) for candidate in sources if isinstance(candidate, dict)),
            *(candidate.get(prefixed_key) for candidate in sources if isinstance(candidate, dict)),
        )
        if provider_config:
            merged[provider_key] = provider_config
            merged[prefixed_key] = provider_config

    return merged


def create_llm(
    provider: str,
    model_name: str,
    api_key: str,
    api_base: str,
    organization: str = None,
    default_headers: Dict = None,
    max_tokens: int = 4000,
    temperature: float = 0,
    max_retries: int = 2,
    streaming: bool = False,
):
    """Create an LLM instance that works for any provider via the ELITEA proxy.

    The ELITEA platform proxies all LLM requests through a gateway.
    - OpenAI models: ``api_base`` already ends with ``/v1``
      -> ChatOpenAI -> POST ``<api_base>/chat/completions``
    - Anthropic models: ``api_base`` is e.g. ``http://host/llm`` (no ``/v1``)
      -> ChatAnthropic -> POST ``<api_base>/v1/messages``

    Using the wrong client class results in 403 because the proxy path
    doesn't match.
    """
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # ChatAnthropic appends /v1/messages itself; strip /v1 if present.
        anthropic_base_url = api_base.rstrip("/")
        if anthropic_base_url.endswith("/v1"):
            anthropic_base_url = anthropic_base_url[:-3]

        if not default_headers:
            default_headers = {
                "openai-organization": str(organization) if organization else "",
                "Authorization": f"Bearer {api_key}",
            }

        return ChatAnthropic(
            model=model_name,
            api_key=api_key,
            base_url=anthropic_base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            streaming=streaming,
            default_headers=default_headers,
        )
    else:
        from langchain_openai import ChatOpenAI

        # Ensure api_base ends with /v1 for the OpenAI client.
        openai_base_url = api_base.rstrip("/")
        if not openai_base_url.endswith("/v1"):
            openai_base_url += "/v1"

        # o-series models require temperature=1
        if str(model_name).startswith("o"):
            temperature = 1.0

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=openai_base_url,
            organization=organization,
            max_retries=max_retries,
            streaming=streaming,
            max_tokens=max_tokens,
        )


def extract_artifact_settings(llm_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract artifact API settings from llm_settings (syngen-style approach).
    
    llm_settings contains:
    - api_base: e.g., 'http://<host_name>/llm/v1' or 'http://<host_name>/llm' (Anthropic)
    - api_key: API key (same key works for artifacts)
    - organization: project_id (e.g., '2')
    
    Returns dict with base_url, api_key, project_id for artifact API calls.
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
    log.info(f"Extracting artifact settings from llm_settings: api_base={openai_api_base}, project_id={project_id}")
    if not project_id:
        log.warning(f"project_id is empty! llm_settings keys: {list(llm_settings.keys())}")
    #
    # Strip LLM-related suffixes from openai_api_base to get base URL:
    # - '/llm/v1', '/llm/v2' (OpenAI models)
    # - '/llm/api/v1' (some configurations)
    # - '/llm' (Anthropic models)
    # e.g., 'http://<host_name>/llm/v1' -> 'http://<host_name>'
    # e.g., 'http://<host_name>/llm' -> 'http://<host_name>'
    base_url = re.sub(r'/llm(/api)?(/v\d+)?/?$', '', openai_api_base)
    #
    return {
        "base_url": base_url,
        "api_key": openai_api_key,
        "project_id": str(project_id),
        "api_path": "/api/v1",
        "x_secret": llm_settings.get("x_secret", "secret"),
    }


def download_artifact(artifact_settings: Dict, bucket_name: str, artifact_name: str) -> bytes:
    """Download artifact from platform bucket (syngen-style raw HTTP)."""
    base_url = artifact_settings.get("base_url", "")
    api_path = artifact_settings.get("api_path", "/api/v1")
    project_id = artifact_settings.get("project_id", "")
    api_key = artifact_settings.get("api_key", "")
    #
    artifact_url = f"{base_url}{api_path}/artifacts/artifact/default/{project_id}"
    # Artifact names may contain spaces or other characters; encode path segments.
    bucket_segment = quote(str(bucket_name).lower(), safe="")
    name_segment = quote(str(artifact_name), safe="")
    url = f"{artifact_url}/{bucket_segment}/{name_segment}"
    #
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-SECRET": artifact_settings.get("x_secret", "secret"),
    }
    #
    log.info("Downloading artifact: %s", url)
    #
    response = requests.get(url, headers=headers, verify=False, timeout=300)
    #
    if response.status_code == 403:
        raise RuntimeError("Not authorized to access artifact")
    elif response.status_code == 404:
        raise RuntimeError(f"Artifact not found: {bucket_name}/{artifact_name}")
    elif response.status_code != 200:
        raise RuntimeError(f"Failed to download artifact: {response.status_code}")
    #
    # Platform returns raw bytes directly (no base64 encoding)
    content = response.content
    log.info("Downloaded artifact: %d bytes", len(content))
    #
    return content


def upload_artifact(artifact_settings: Dict, bucket_name: str, artifact_name: str, artifact_data: bytes) -> Dict:
    """Upload artifact to platform bucket (syngen-style raw HTTP)."""
    base_url = artifact_settings.get("base_url", "")
    api_path = artifact_settings.get("api_path", "/api/v1")
    project_id = artifact_settings.get("project_id", "")
    api_key = artifact_settings.get("api_key", "")
    #
    artifacts_url = f"{base_url}{api_path}/artifacts/artifacts/default/{project_id}"
    url = f"{artifacts_url}/{bucket_name.lower()}"
    #
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-SECRET": artifact_settings.get("x_secret", "secret"),
    }
    #
    log.info("Uploading artifact: %s to bucket %s", artifact_name, bucket_name)
    #
    files = {'file': (artifact_name, artifact_data)}
    response = requests.post(url, headers=headers, files=files, verify=False, timeout=300)
    #
    if response.status_code == 403:
        raise RuntimeError("Not authorized to upload artifact")
    elif response.status_code not in [200, 201]:
        raise RuntimeError(f"Failed to upload artifact: {response.status_code}")
    #
    log.info("Uploaded artifact successfully")
    try:
        return response.json()
    except Exception:
        return {"status": "ok"}


class MiniArtifactClient:
    """
    Mini artifact client for WikiRegistryManager.
    Implements the interface expected by registry_manager: download_artifact, create_artifact.
    Uses raw HTTP requests like syngen_plugin.
    """
    
    def __init__(self, artifact_settings: Dict[str, Any]):
        self.settings = artifact_settings
    
    def download_artifact(self, bucket_name: str, artifact_name: str) -> bytes:
        """Download artifact from bucket."""
        return download_artifact(self.settings, bucket_name, artifact_name)
    
    def create_artifact(self, bucket_name: str, artifact_name: str, data: str) -> Dict:
        """Create/upload artifact to bucket."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return upload_artifact(self.settings, bucket_name, artifact_name, data)


def _extract_repo_config_from_toolkit(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract repository configuration from expanded code_toolkit.
    
    Supports: github, gitlab, bitbucket, ado_repos
    
    Returns a normalized repo_config dict with:
        - provider_type: str (github, gitlab, bitbucket, ado_repos)
        - provider_config: dict (the provider-specific configuration)
        - repository: str (repository identifier)
        - branch: str (branch name)
        - project: str or None (for Bitbucket/ADO)
        - is_cloud: bool or None (for Bitbucket)
    """
    code_toolkit = (
        params.get('code_toolkit')
        or params.get('toolkit_configuration_code_toolkit')
        or params.get('toolkit_configuration_code_repository')
        or params.get('code_repository')
        or {}
    )
    
    # Initialize with defaults
    repo_config = {
        'provider_type': 'github',
        'provider_config': {},
        'repository': None,
        'branch': 'main',
        'project': None,
        'is_cloud': None,
    }
    
    repo_settings = {}
    if isinstance(code_toolkit, dict):
        repo_settings = _merge_toolkit_payload(code_toolkit)

    if not repo_settings and isinstance(params, dict):
        repo_settings = _merge_provider_configs(params)

    if isinstance(repo_settings, dict) and any(
        key in repo_settings
        for key in (
            'github_configuration', 'gitlab_configuration', 'bitbucket_configuration', 'ado_configuration',
            'toolkit_configuration_github_configuration', 'toolkit_configuration_gitlab_configuration',
            'toolkit_configuration_bitbucket_configuration', 'toolkit_configuration_ado_configuration',
        )
    ):
        if 'github_configuration' in repo_settings or 'toolkit_configuration_github_configuration' in repo_settings:
            github_config = repo_settings.get('github_configuration') or repo_settings.get('toolkit_configuration_github_configuration') or {}
            repo_config['provider_type'] = 'github'
            repo_config['provider_config'] = github_config
            repo_config['repository'] = (
                repo_settings.get('repository')
                or repo_settings.get('github_repository')
                or repo_settings.get('toolkit_configuration_github_repository')
            )
            repo_config['branch'] = (
                repo_settings.get('active_branch')
                or repo_settings.get('toolkit_configuration_active_branch')
                or repo_settings.get('base_branch')
                or repo_settings.get('toolkit_configuration_base_branch')
                or repo_settings.get('branch', 'main')
            )
        elif 'gitlab_configuration' in repo_settings or 'toolkit_configuration_gitlab_configuration' in repo_settings:
            gitlab_config = repo_settings.get('gitlab_configuration') or repo_settings.get('toolkit_configuration_gitlab_configuration') or {}
            repo_config['provider_type'] = 'gitlab'
            repo_config['provider_config'] = gitlab_config
            repo_config['repository'] = repo_settings.get('repository') or repo_settings.get('toolkit_configuration_repository')
            repo_config['branch'] = (
                repo_settings.get('branch')
                or repo_settings.get('toolkit_configuration_branch')
                or repo_settings.get('active_branch')
                or repo_settings.get('toolkit_configuration_active_branch')
                or repo_settings.get('base_branch')
                or repo_settings.get('toolkit_configuration_base_branch', 'main')
            )
        elif 'bitbucket_configuration' in repo_settings or 'toolkit_configuration_bitbucket_configuration' in repo_settings:
            bitbucket_config = repo_settings.get('bitbucket_configuration') or repo_settings.get('toolkit_configuration_bitbucket_configuration') or {}
            repo_config['provider_type'] = 'bitbucket'
            repo_config['provider_config'] = bitbucket_config
            repo_config['repository'] = repo_settings.get('repository') or repo_settings.get('toolkit_configuration_repository')
            repo_config['branch'] = (
                repo_settings.get('branch')
                or repo_settings.get('toolkit_configuration_branch')
                or repo_settings.get('active_branch')
                or repo_settings.get('toolkit_configuration_active_branch')
                or repo_settings.get('base_branch')
                or repo_settings.get('toolkit_configuration_base_branch', 'main')
            )
            repo_config['project'] = repo_settings.get('project') or repo_settings.get('toolkit_configuration_project')
            repo_config['is_cloud'] = repo_settings.get('cloud') or repo_settings.get('toolkit_configuration_cloud')
        elif 'ado_configuration' in repo_settings or 'toolkit_configuration_ado_configuration' in repo_settings:
            ado_config = repo_settings.get('ado_configuration') or repo_settings.get('toolkit_configuration_ado_configuration') or {}
            repo_config['provider_type'] = 'ado_repos'
            repo_config['provider_config'] = ado_config
            repo_config['repository'] = (
                repo_settings.get('repository_id')
                or repo_settings.get('toolkit_configuration_repository_id')
                or repo_settings.get('repository')
                or repo_settings.get('toolkit_configuration_repository')
            )
            repo_config['branch'] = (
                repo_settings.get('active_branch')
                or repo_settings.get('toolkit_configuration_active_branch')
                or repo_settings.get('base_branch')
                or repo_settings.get('toolkit_configuration_base_branch')
                or repo_settings.get('branch', 'main')
            )
            repo_config['project'] = ado_config.get('project') or repo_settings.get('project') or repo_settings.get('toolkit_configuration_project')
        else:
            # Fallback: assume GitHub with legacy structure
            repo_config['provider_type'] = 'github'
            repo_config['provider_config'] = repo_settings.get('github_configuration', {})
            repo_config['repository'] = repo_settings.get('repository') or repo_settings.get('github_repository')
            repo_config['branch'] = repo_settings.get('base_branch') or repo_settings.get('active_branch', 'main')
    else:
        # Legacy fallback - direct parameters
        repo_config['provider_type'] = 'github'
        repo_config['provider_config'] = params.get('github_configuration', {})
        repo_config['repository'] = params.get('github_repository')
        repo_config['branch'] = params.get('github_base_branch') or params.get('github_branch', 'main')

    if _payload_contains_provider_key(params, 'ado_configuration') and (
        repo_config.get('provider_type') != 'ado_repos' or not repo_config.get('repository')
    ):
        log.warning(
            'Suspicious ADO repo config extraction: provider_type=%s repository=%s branch=%s project=%s',
            repo_config.get('provider_type'),
            repo_config.get('repository'),
            repo_config.get('branch'),
            repo_config.get('project'),
        )

    return repo_config

class Method:  # pylint: disable=E1101,R0903,W0201
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def _create_error_response(self, invocation_id, operation, model_name, exception, include_traceback=True):
        """Create a structured error response with optional stack trace.

        Mirrors Syngen's error contract:
        - status: "Error"
        - result: JSON string of list_of_objects (at least one message)
        - error_category / error_type fields for programmatic handling
        """
        error_type = type(exception).__name__
        error_category = "unknown_error"
        exception_str = str(exception)

        try:
            lower = exception_str.lower()
        except Exception:
            lower = ""

        if "not found" in lower or isinstance(exception, FileNotFoundError):
            error_category = "resource_not_found"
        elif "[service_busy]" in lower or "service is busy" in lower or "deepwiki service is busy" in lower:
            error_category = "service_busy"
        elif "download" in lower or "artifact" in lower:
            error_category = "artifact_error"
        elif "memory" in lower or isinstance(exception, MemoryError):
            error_category = "out_of_memory"
        elif "timeout" in lower:
            error_category = "timeout_error"
        elif isinstance(exception, RuntimeError):
            if "training" in lower:
                error_category = "training_failed"
            elif "inference" in lower or "generat" in lower:
                error_category = "inference_failed"
            else:
                error_category = "runtime_error"
        elif isinstance(exception, ValueError):
            error_category = "invalid_input"

        model_context = f" for model '{model_name}'" if model_name else ""
        
        # Build user-facing error message - keep it clean and actionable
        if include_traceback:
            # Internal/debug mode - include technical details
            error_message = f"{str(operation).capitalize()} failed{model_context}\n\n"
            error_message += f"Error: {exception_str}\n"
            error_message += f"Type: {error_type}\n"
            error_message += f"Category: {error_category}"
            
            tb_lines = traceback.format_exception(type(exception), exception, exception.__traceback__)
            stack_trace = "".join(tb_lines)
            error_message += f"\n\nStack Trace:\n{stack_trace}"
        else:
            # User-facing mode - just the error message, no technical noise
            error_message = f"{str(operation).capitalize()} failed{model_context}: {exception_str}"

        result_objects = [
            {
                "object_type": "message",
                "result_target": "response",
                "result_encoding": "plain",
                "data": error_message,
            }
        ]

        return {
            "invocation_id": invocation_id,
            "status": "Error",
            "result": json.dumps(result_objects),
            "result_type": "String",
            "error_category": error_category,
            "error_type": error_type,
        }

    @web.method()
    def _transform_deepwiki_query_request(self, request_data):
        """Transform deepwiki_query request to standard deepwiki request format.

        Extracts configuration from the referenced deepwiki toolkit and creates
        a request that can be processed by the standard tool handlers.

        Returns:
            tuple: (transformed_request_data, error_response) - error_response is None on success
        """
        log.info(f"[deepwiki_query] Transforming request")

        # Get toolkit parameters
        toolkit_params = request_data.get("configuration", {}).get("parameters", {})

        # Get the referenced deepwiki toolkit - can be an ID or a full toolkit object
        deepwiki_toolkit_ref = toolkit_params.get("deepwiki_toolkit")
        if not deepwiki_toolkit_ref:
            return None, "deepwiki_toolkit parameter is required - specify which DeepWiki toolkit to use"

        # Extract toolkit ID and settings - handle both int and dict (full toolkit object)
        deepwiki_settings = {}

        if isinstance(deepwiki_toolkit_ref, dict):
            deepwiki_toolkit_id = deepwiki_toolkit_ref.get("id")
            deepwiki_settings = deepwiki_toolkit_ref.get("settings", {})
            log.info(f"[deepwiki_query] Extracted toolkit ID from object: {deepwiki_toolkit_id}")
        else:
            deepwiki_toolkit_id = deepwiki_toolkit_ref
            log.info(f"[deepwiki_query] Using toolkit ID directly: {deepwiki_toolkit_id}")

        if not deepwiki_toolkit_id:
            return None, "Could not extract toolkit ID from deepwiki_toolkit parameter"

        # Get project_id from request_data root
        project_id = request_data.get("project_id")
        if not project_id:
            return None, "project_id not found in request - required context for deepwiki_query toolkit"

        log.info(f"[deepwiki_query] Using deepwiki toolkit {deepwiki_toolkit_id} from project {project_id}")

        # Get LLM and embedding config from deepwiki_query's OWN parameters (expanded by platform)
        # These are configured directly on deepwiki_query, not inherited from referenced toolkit
        llm_settings = toolkit_params.get("llm_settings")  # Expanded from llm_model
        embedding_model = toolkit_params.get("embedding_model")  # Expanded from embedding_model

        log.info(f"[deepwiki_query] llm_settings type: {type(llm_settings).__name__}")
        log.info(f"[deepwiki_query] embedding_model type: {type(embedding_model).__name__}")

        # Get code_toolkit from the REFERENCED deepwiki toolkit (for repo config)
        code_toolkit = deepwiki_settings.get("toolkit_configuration_code_toolkit") or deepwiki_settings.get("code_toolkit")
        log.info(f"[deepwiki_query] code_toolkit from referenced toolkit: {type(code_toolkit).__name__}")

        # Build mapped params combining:
        # - llm_settings, embedding_model from deepwiki_query's own config (expanded)
        # - code_toolkit from referenced deepwiki toolkit
        mapped_params = {
            "llm_settings": llm_settings or {},
            "embedding_model": embedding_model,
            "code_toolkit": code_toolkit,
        }

        log.info(f"[deepwiki_query] Mapped params keys: {mapped_params.keys()}")

        # Build transformed request with mapped configuration
        transformed_request = {
            "project_id": project_id,
            "parameters": request_data.get("parameters", {}),
            "configuration": {
                "parameters": mapped_params
            }
        }

        return transformed_request, None

    @web.method()
    def _handle_wiki_query_tool(self, invocation_id, tool_name, request_data):
        """Handle wiki_query toolkit tools.
        
        These tools provide Context7-style multi-wiki access:
        - list_wikis: List all available wikis in the registry
        - resolve_and_ask: Resolve which wiki to query and answer the question
        - resolve_and_deep_research: Resolve which wiki and perform deep research
        - delete_wiki: Delete a wiki and all its artifacts
        """
        try:
            toolkit_params = request_data.get("configuration", {}).get("parameters", {})
            tool_params = request_data.get("parameters", {})
            
            # Merge toolkit and tool params
            params = toolkit_params.copy()
            for key, value in tool_params.items():
                if key not in params or value:
                    params[key] = value
            
            # Extract llm_settings for artifact access (expanded by platform)
            llm_settings = toolkit_params.get("llm_settings", {})
            if not llm_settings:
                # Try to get from top-level configuration
                llm_settings = request_data.get("configuration", {}).get("llm_settings", {})
            
            log.info(f"[wiki_query] Tool: {tool_name}, llm_settings present: {bool(llm_settings)}")
            
            # Hardcoded bucket for wiki artifacts
            bucket_name = default_bucket
            
            if tool_name == "list_wikis":
                return self._list_wikis(invocation_id, params, bucket_name, llm_settings)
            elif tool_name == "resolve_and_ask":
                return self._resolve_and_ask(invocation_id, params, bucket_name, llm_settings)
            elif tool_name == "resolve_and_deep_research":
                return self._resolve_and_deep_research(invocation_id, params, bucket_name, llm_settings)
            elif tool_name == "delete_wiki":
                return self._delete_wiki(invocation_id, params, bucket_name, llm_settings)
            else:
                return self._create_error_response(
                    invocation_id=invocation_id,
                    operation=tool_name,
                    model_name=None,
                    exception=ValueError(f"Unknown wiki_query tool: {tool_name}"),
                    include_traceback=False,
                )
        except Exception as e:
            return self._create_error_response(
                invocation_id=invocation_id,
                operation=tool_name,
                model_name=None,
                exception=e,
                include_traceback=True,
            )

    @web.method()
    def _list_wikis(self, invocation_id, params, bucket_name, llm_settings=None):
        """List all wikis in the registry."""
        from ..plugin_implementation.registry_manager import WikiRegistryManager
        
        include_metadata = params.get("include_metadata", False)
        
        # Create artifacts client using llm_settings (syngen-style)
        try:
            if not llm_settings:
                log.warning("No llm_settings provided for list_wikis, cannot access artifacts")
                result_text = "No wikis found (artifacts settings not configured)"
                return {
                    "invocation_id": invocation_id,
                    "status": "Completed",
                    "result": json.dumps([{
                        "object_type": "wiki_list",
                        "result_target": "response",
                        "result_encoding": "plain",
                        "data": result_text
                    }]),
                    "result_type": "String",
                }
            
            # Use extract_artifact_settings to get proper artifact API settings
            artifact_settings = extract_artifact_settings(llm_settings)
            
            if not artifact_settings.get("base_url"):
                log.warning(f"No base_url in artifact_settings. llm_settings keys: {list(llm_settings.keys())}")
                result_text = "No wikis found (artifacts base_url not configured)"
                return {
                    "invocation_id": invocation_id,
                    "status": "Completed",
                    "result": json.dumps([{
                        "object_type": "wiki_list",
                        "result_target": "response",
                        "result_encoding": "plain",
                        "data": result_text
                    }]),
                    "result_type": "String",
                }
            
            artifacts_client = MiniArtifactClient(artifact_settings)
            log.info(f"[list_wikis] Created MiniArtifactClient for project {artifact_settings.get('project_id')}, base_url: {artifact_settings.get('base_url')}")
        except Exception as e:
            log.warning(f"Could not create artifacts client: {e}")
            result_text = f"No wikis found (artifacts client error: {e})"
            return {
                "invocation_id": invocation_id,
                "status": "Completed",
                "result": json.dumps([{
                    "object_type": "wiki_list",
                    "result_target": "response",
                    "result_encoding": "plain",
                    "data": result_text
                }]),
                "result_type": "String",
            }
        
        registry = WikiRegistryManager(artifacts_client, bucket_name)
        wikis = registry.list_wikis()
        
        if not wikis:
            result_text = "No wikis have been generated yet. Use the Deepwiki toolkit to generate wikis first."
        elif include_metadata:
            # Full metadata format
            lines = ["# Available Wikis\n"]
            for wiki in wikis:
                lines.append(f"## {wiki.get('id', 'unknown')}")
                lines.append(f"- **Repository**: {wiki.get('repo', 'N/A')}")
                lines.append(f"- **Branch**: {wiki.get('branch', 'N/A')}")
                lines.append(f"- **Title**: {wiki.get('display_name', 'N/A')}")
                lines.append(f"- **Description**: {wiki.get('description', 'N/A')}")
                lines.append(f"- **Created**: {wiki.get('created_at', 'N/A')}")
                lines.append("")
            result_text = "\n".join(lines)
        else:
            # Compact format for LLM resolution
            lines = ["Available wikis:"]
            for wiki in wikis:
                wiki_id = wiki.get("id", "unknown")
                title = wiki.get("display_name", "")
                desc = wiki.get("description", "")[:100] if wiki.get("description") else ""
                lines.append(f"- {wiki_id}: {title}" + (f" - {desc}..." if desc else ""))
            result_text = "\n".join(lines)
        
        return {
            "invocation_id": invocation_id,
            "status": "Completed",
            "result": json.dumps([{
                "object_type": "wiki_list",
                "result_target": "response",
                "result_encoding": "plain",
                "data": result_text
            }]),
            "result_type": "String",
        }

    @web.method()
    def _resolve_and_ask(self, invocation_id, params, bucket_name, llm_settings=None):
        """Resolve which wiki to query and answer the question.
        
        Uses LLM to match the question to the best wiki based on
        repository names and descriptions.
        """
        from ..plugin_implementation.registry_manager import WikiRegistryManager
        
        question = params.get("question", "")
        wiki_id_hint = params.get("wiki_id_hint")
        
        if not question:
            return self._create_error_response(
                invocation_id=invocation_id,
                operation="resolve_and_ask",
                model_name=None,
                exception=ValueError("Question is required"),
                include_traceback=False,
            )
        
        # Get llm_settings from params if not passed directly
        if not llm_settings:
            llm_settings = params.get("llm_settings", {})
        
        if not llm_settings:
            return self._create_error_response(
                invocation_id=invocation_id,
                operation="resolve_and_ask",
                model_name=None,
                exception=ValueError("llm_settings is required"),
                include_traceback=False,
            )
        
        # Create artifacts client using llm_settings
        try:
            artifact_settings = extract_artifact_settings(llm_settings)
            if not artifact_settings.get("base_url"):
                log.warning(f"No base_url in artifact_settings for resolve_and_ask")
                return self._create_error_response(
                    invocation_id=invocation_id,
                    operation="resolve_and_ask",
                    model_name=None,
                    exception=RuntimeError("Artifacts base_url not configured"),
                    include_traceback=False,
                )
            artifacts_client = MiniArtifactClient(artifact_settings)
            log.info(f"[resolve_and_ask] Created MiniArtifactClient for project {artifact_settings.get('project_id')}")
        except Exception as e:
            log.warning(f"Could not create artifacts client: {e}")
            return self._create_error_response(
                invocation_id=invocation_id,
                operation="resolve_and_ask",
                model_name=None,
                exception=RuntimeError(f"Could not access wiki registry: {e}"),
                include_traceback=False,
            )
        
        registry = WikiRegistryManager(artifacts_client, bucket_name)
        
        # If wiki_id_hint provided, use it directly
        if wiki_id_hint:
            resolved_wiki_id = wiki_id_hint
            log.info(f"[wiki_query] Using provided wiki_id_hint: {wiki_id_hint}")
        else:
            # Use LLM to resolve which wiki to query
            resolution_data = registry.get_wikis_for_resolution()
            
            if not resolution_data:
                return {
                    "invocation_id": invocation_id,
                    "status": "Completed",
                    "result": json.dumps([{
                        "object_type": "answer",
                        "result_target": "response",
                        "result_encoding": "plain",
                        "data": "No wikis available. Please generate wikis using the Deepwiki toolkit first."
                    }]),
                    "result_type": "String",
                }
            
            # Use LLM to resolve which wiki (llm_settings already validated above)
            resolved_wiki_id = self._resolve_wiki_with_llm(question, resolution_data, llm_settings)
            
            if not resolved_wiki_id:
                return {
                    "invocation_id": invocation_id,
                    "status": "Completed",
                    "result": json.dumps([{
                        "object_type": "answer",
                        "result_target": "response",
                        "result_encoding": "plain",
                        "data": f"Could not determine which wiki to query for: '{question}'. Available wikis: {', '.join([w['wiki_id'] for w in resolution_data])}"
                    }]),
                    "result_type": "String",
                }
        
        log.info(f"[wiki_query] Resolved wiki_id: {resolved_wiki_id}")
        
        # Get wiki metadata to construct repo_config for the ask tool
        wiki_entry = registry.get_wiki(resolved_wiki_id)
        if not wiki_entry:
            return {
                "invocation_id": invocation_id,
                "status": "Completed",
                "result": json.dumps([{
                    "object_type": "answer",
                    "result_target": "response",
                    "result_encoding": "plain",
                    "data": f"Wiki '{resolved_wiki_id}' not found in registry."
                }]),
                "result_type": "String",
            }
        
        # Build repo_config from wiki entry for the ask tool
        provider = wiki_entry.get("provider", "github")
        
        # Extract base repo (owner/repo) - handle legacy entries that may have full canonical identifier
        # Legacy format: "owner/repo:branch:commit" -> extract "owner/repo"
        wiki_repo = wiki_entry.get("repo", "")
        if ":" in wiki_repo:
            # Legacy format with branch/commit embedded - extract base repo
            base_repo = wiki_repo.split(":")[0]
            log.info(f"[resolve_and_ask] Extracted base repo '{base_repo}' from '{wiki_repo}'")
        else:
            base_repo = wiki_repo
        
        # Get branch from wiki entry, or parse from wiki_id (e.g., 'fmtlib--fmt--master' -> 'master')
        wiki_branch = wiki_entry.get("branch")
        if not wiki_branch:
            from ..plugin_implementation.registry_manager import parse_wiki_id
            parsed = parse_wiki_id(resolved_wiki_id)
            wiki_branch = parsed.get("branch", "main")
            log.info(f"[resolve_and_ask] Parsed branch '{wiki_branch}' from wiki_id '{resolved_wiki_id}'")
        
        repo_config = {
            'provider_type': provider,
            'provider_config': {},  # Will use cached artifacts, no auth needed
            'repository': base_repo,
            'branch': wiki_branch,
        }
        
        repo_identifier_override = wiki_entry.get("canonical_repo_identifier") or f"{base_repo}:{wiki_branch}"
        log.info(f"[resolve_and_ask] Using repo_identifier_override: {repo_identifier_override}")
        
        # Call the ask tool with resolved wiki (llm_settings already validated at start)
        try:
            result = self.ask(
                question=question,
                llm_settings=llm_settings,
                embedding_model=params.get('embedding_model'),
                repo_config=repo_config,
                chat_history=params.get('chat_history', []),
                k=params.get('k', 15),
                repo_identifier_override=repo_identifier_override,
                analysis_key_override=wiki_entry.get("analysis_key"),
            )
            
            if isinstance(result, dict) and result.get("success"):
                answer_text = result.get("answer", "Question answered successfully")
                # Prepend resolution info
                answer_with_context = f"*Querying wiki: {resolved_wiki_id}*\n\n{answer_text}"
                return {
                    "invocation_id": invocation_id,
                    "status": "Completed",
                    "result": json.dumps([{
                        "object_type": "answer",
                        "result_target": "response",
                        "result_encoding": "plain",
                        "data": answer_with_context
                    }]),
                    "result_type": "String",
                }
            else:
                error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
                return {
                    "invocation_id": invocation_id,
                    "status": "Completed",
                    "result": json.dumps([{
                        "object_type": "answer",
                        "result_target": "response",
                        "result_encoding": "plain",
                        "data": f"Failed to query wiki {resolved_wiki_id}: {error_msg}"
                    }]),
                    "result_type": "String",
                }
        except Exception as ask_err:
            log.error(f"[wiki_query] Ask failed: {ask_err}")
            return self._create_error_response(
                invocation_id=invocation_id,
                operation="resolve_and_ask",
                model_name=None,
                exception=ask_err,
                include_traceback=True,
            )

    @web.method()
    def _resolve_and_deep_research(self, invocation_id, params, bucket_name, llm_settings=None):
        """Resolve which wiki to query and perform deep research.
        
        Similar to resolve_and_ask but uses deep_research for complex queries.
        """
        from ..plugin_implementation.registry_manager import WikiRegistryManager
        
        question = params.get("question", "")
        wiki_id_hint = params.get("wiki_id_hint")
        research_type = params.get("research_type", "general")
        
        if not question:
            return self._create_error_response(
                invocation_id=invocation_id,
                operation="resolve_and_deep_research",
                model_name=None,
                exception=ValueError("Question is required"),
                include_traceback=False,
            )
        
        # Get llm_settings from params if not passed directly
        if not llm_settings:
            llm_settings = params.get("llm_settings", {})
        
        if not llm_settings:
            return self._create_error_response(
                invocation_id=invocation_id,
                operation="resolve_and_deep_research",
                model_name=None,
                exception=ValueError("llm_settings is required"),
                include_traceback=False,
            )
        
        # Create artifacts client using llm_settings
        try:
            artifact_settings = extract_artifact_settings(llm_settings)
            if not artifact_settings.get("base_url"):
                log.warning(f"No base_url in artifact_settings for resolve_and_deep_research")
                return self._create_error_response(
                    invocation_id=invocation_id,
                    operation="resolve_and_deep_research",
                    model_name=None,
                    exception=RuntimeError("Artifacts base_url not configured"),
                    include_traceback=False,
                )
            artifacts_client = MiniArtifactClient(artifact_settings)
            log.info(f"[resolve_and_deep_research] Created MiniArtifactClient for project {artifact_settings.get('project_id')}")
        except Exception as e:
            log.warning(f"Could not create artifacts client: {e}")
            return self._create_error_response(
                invocation_id=invocation_id,
                operation="resolve_and_deep_research",
                model_name=None,
                exception=RuntimeError(f"Could not access wiki registry: {e}"),
                include_traceback=False,
            )
        
        registry = WikiRegistryManager(artifacts_client, bucket_name)
        
        # Resolve wiki_id
        if wiki_id_hint:
            resolved_wiki_id = wiki_id_hint
        else:
            resolution_data = registry.get_wikis_for_resolution()
            if not resolution_data:
                return {
                    "invocation_id": invocation_id,
                    "status": "Completed",
                    "result": json.dumps([{
                        "object_type": "report",
                        "result_target": "response",
                        "result_encoding": "plain",
                        "data": "No wikis available. Please generate wikis using the Deepwiki toolkit first."
                    }]),
                    "result_type": "String",
                }
            
            # Use LLM to resolve which wiki (llm_settings already validated above)
            resolved_wiki_id = self._resolve_wiki_with_llm(question, resolution_data, llm_settings)
            
            if not resolved_wiki_id:
                return {
                    "invocation_id": invocation_id,
                    "status": "Completed",
                    "result": json.dumps([{
                        "object_type": "report",
                        "result_target": "response",
                        "result_encoding": "plain",
                        "data": f"Could not determine which wiki to query for: '{question}'"
                    }]),
                    "result_type": "String",
                }
        
        log.info(f"[wiki_query] Resolved wiki_id for deep_research: {resolved_wiki_id}")
        
        # Get wiki metadata
        wiki_entry = registry.get_wiki(resolved_wiki_id)
        if not wiki_entry:
            return {
                "invocation_id": invocation_id,
                "status": "Completed",
                "result": json.dumps([{
                    "object_type": "report",
                    "result_target": "response",
                    "result_encoding": "plain",
                    "data": f"Wiki '{resolved_wiki_id}' not found in registry."
                }]),
                "result_type": "String",
            }
        
        # Build repo_config - handle legacy entries with full canonical identifier
        # Legacy format: "owner/repo:branch:commit" -> extract "owner/repo"
        wiki_repo = wiki_entry.get("repo", "")
        if ":" in wiki_repo:
            base_repo = wiki_repo.split(":")[0]
            log.info(f"[resolve_and_deep_research] Extracted base repo '{base_repo}' from '{wiki_repo}'")
        else:
            base_repo = wiki_repo
        
        # Get branch from wiki entry, or parse from wiki_id (e.g., 'fmtlib--fmt--master' -> 'master')
        wiki_branch = wiki_entry.get("branch")
        if not wiki_branch:
            from ..plugin_implementation.registry_manager import parse_wiki_id
            parsed = parse_wiki_id(resolved_wiki_id)
            wiki_branch = parsed.get("branch", "main")
            log.info(f"[resolve_and_deep_research] Parsed branch '{wiki_branch}' from wiki_id '{resolved_wiki_id}'")
        
        repo_config = {
            'provider_type': wiki_entry.get("provider", "github"),
            'provider_config': {},
            'repository': base_repo,
            'branch': wiki_branch,
        }
        
        repo_identifier_override = wiki_entry.get("canonical_repo_identifier") or f"{base_repo}:{wiki_branch}"
        log.info(f"[resolve_and_deep_research] Using repo_identifier_override: {repo_identifier_override}")
        
        # Ensure max_tokens is set for deep_research - wiki_query doesn't have toolkit config,
        # so we hardcode a generous value for comprehensive reports
        if "max_tokens" not in llm_settings:
            llm_settings = {**llm_settings, "max_tokens": 16384}
            log.info("[resolve_and_deep_research] Set max_tokens=16384 for deep_research")
        
        # Call deep_research (llm_settings already validated at start)
        try:
            result = self.deep_research(
                question=question,
                llm_settings=llm_settings,
                embedding_model=params.get('embedding_model'),
                repo_config=repo_config,
                chat_history=params.get('chat_history', []),
                k=params.get('k', 15),
                research_type=research_type,
                enable_subagents=params.get('enable_subagents', True),
                repo_identifier_override=repo_identifier_override,
                analysis_key_override=wiki_entry.get("analysis_key"),
            )
            
            if isinstance(result, dict) and result.get("success"):
                report_text = result.get("report", result.get("answer", "Research completed"))
                report_with_context = f"*Deep research on wiki: {resolved_wiki_id}*\n\n{report_text}"
                return {
                    "invocation_id": invocation_id,
                    "status": "Completed",
                    "result": json.dumps([{
                        "object_type": "report",
                        "result_target": "response",
                        "result_encoding": "plain",
                        "data": report_with_context
                    }]),
                    "result_type": "String",
                }
            else:
                error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
                return {
                    "invocation_id": invocation_id,
                    "status": "Completed",
                    "result": json.dumps([{
                        "object_type": "report",
                        "result_target": "response",
                        "result_encoding": "plain",
                        "data": f"Failed deep research on wiki {resolved_wiki_id}: {error_msg}"
                    }]),
                    "result_type": "String",
                }
        except Exception as research_err:
            log.error(f"[wiki_query] Deep research failed: {research_err}")
            return self._create_error_response(
                invocation_id=invocation_id,
                operation="resolve_and_deep_research",
                model_name=None,
                exception=research_err,
                include_traceback=True,
            )

    @web.method()
    def _delete_wiki(self, invocation_id, params, bucket_name, llm_settings=None):
        """Delete a wiki and all its artifacts.
        
        Removes all artifacts with the wiki_id prefix and unregisters from registry.
        """
        from ..plugin_implementation.registry_manager import WikiRegistryManager
        
        wiki_id = params.get("wiki_id", "")
        
        if not wiki_id:
            return self._create_error_response(
                invocation_id=invocation_id,
                operation="delete_wiki",
                model_name=None,
                exception=ValueError("wiki_id is required"),
                include_traceback=False,
            )
        
        # Get llm_settings from params if not passed directly
        if not llm_settings:
            llm_settings = params.get("llm_settings", {})
        
        if not llm_settings:
            return self._create_error_response(
                invocation_id=invocation_id,
                operation="delete_wiki",
                model_name=None,
                exception=ValueError("llm_settings is required for artifact access"),
                include_traceback=False,
            )
        
        # Create artifacts client using llm_settings
        try:
            artifact_settings = extract_artifact_settings(llm_settings)
            if not artifact_settings.get("base_url"):
                log.warning(f"No base_url in artifact_settings for delete_wiki")
                return self._create_error_response(
                    invocation_id=invocation_id,
                    operation="delete_wiki",
                    model_name=None,
                    exception=RuntimeError("Artifacts base_url not configured"),
                    include_traceback=False,
                )
            artifacts_client = MiniArtifactClient(artifact_settings)
            log.info(f"[delete_wiki] Created MiniArtifactClient for project {artifact_settings.get('project_id')}")
        except Exception as e:
            log.warning(f"Could not create artifacts client: {e}")
            return self._create_error_response(
                invocation_id=invocation_id,
                operation="delete_wiki",
                model_name=None,
                exception=RuntimeError(f"Could not access wiki registry: {e}"),
                include_traceback=False,
            )
        
        registry = WikiRegistryManager(artifacts_client, bucket_name)
        
        # Check if wiki exists
        wiki_entry = registry.get_wiki(wiki_id)
        if not wiki_entry:
            return {
                "invocation_id": invocation_id,
                "status": "Completed",
                "result": json.dumps([{
                    "object_type": "message",
                    "result_target": "response",
                    "result_encoding": "plain",
                    "data": f"Wiki '{wiki_id}' not found in registry."
                }]),
                "result_type": "String",
            }
        
        # Delete wiki and all artifacts
        result = registry.delete_wiki_with_artifacts(wiki_id)
        
        # Format result message
        if result.get("errors"):
            error_list = "\n".join([f"- {e}" for e in result["errors"]])
            message = f"Wiki '{wiki_id}' deletion completed with errors:\n{error_list}\n\nDeleted {result['deleted_count']} artifacts."
        else:
            message = f"Wiki '{wiki_id}' successfully deleted.\n- Artifacts removed: {result['deleted_count']}\n- Registry updated: {'Yes' if result['registry_removed'] else 'No'}"
        
        return {
            "invocation_id": invocation_id,
            "status": "Completed",
            "result": json.dumps([{
                "object_type": "message",
                "result_target": "response",
                "result_encoding": "plain",
                "data": message
            }]),
            "result_type": "String",
        }

    @web.method()
    def _resolve_wiki_with_llm(self, question: str, wikis: List[Dict], llm_settings: Dict) -> Optional[str]:
        """Use LLM to resolve which wiki best matches the question.
        
        Args:
            question: The user's question
            wikis: List of wiki metadata dicts with wiki_id, wiki_title, description
            llm_settings: LLM configuration from platform
            
        Returns:
            Resolved wiki_id or None if resolution fails
        """
        # Build wiki list for prompt
        wiki_list = []
        for w in wikis:
            wiki_id = w.get('wiki_id', '') if isinstance(w, dict) else str(w)
            wiki_title = w.get('wiki_title', '') if isinstance(w, dict) else ''
            description = w.get('description', '') if isinstance(w, dict) else ''
            desc_preview = description[:150] if description else ''
            wiki_list.append(f"- {wiki_id}: {wiki_title} - {desc_preview}")
        wiki_list_text = "\n".join(wiki_list)
        
        resolution_prompt = f"""Given the following question and list of available code repositories, determine which repository the question is most likely about.

Question: {question}

Available repositories:
{wiki_list_text}

Respond with ONLY the wiki_id (e.g., "owner--repo--branch") of the most relevant repository, or "NONE" if none are relevant.
Do not include any explanation or other text."""

        try:
            # Extract LLM settings - platform uses api_key/api_base
            provider = llm_settings.get("provider", "openai")
            api_base = llm_settings.get("api_base") or llm_settings.get("openai_api_base", "")
            api_key = llm_settings.get("api_key") or llm_settings.get("openai_api_key", "")
            model_name = llm_settings.get("model_name", "gpt-4o-mini")
            organization = llm_settings.get("organization")
            default_headers = llm_settings.get("default_headers", {})
            max_tokens = llm_settings.get("max_tokens", 4000)
            
            log.info(f"[resolve_wiki] provider={provider}, model={model_name}, api_base={api_base[:50] if api_base else 'None'}...")
            
            llm = create_llm(
                provider=provider,
                model_name=model_name,
                api_key=api_key,
                api_base=api_base,
                organization=organization,
                default_headers=default_headers,
                max_tokens=max_tokens,
                temperature=0,
            )
            
            response = llm.invoke(resolution_prompt)
            resolved_id = response.content.strip().strip('"').strip("'")
            
            # Validate the resolved ID exists
            valid_ids = [w.get("wiki_id", "") if isinstance(w, dict) else str(w) for w in wikis]
            if resolved_id in valid_ids:
                return resolved_id
            elif resolved_id == "NONE":
                return None
            else:
                # Try fuzzy matching
                for valid_id in valid_ids:
                    if resolved_id.lower() in valid_id.lower() or valid_id.lower() in resolved_id.lower():
                        return valid_id
                return None
                
        except Exception as e:
            log.error(f"Wiki resolution failed: {e}")
            return None

    @web.method()
    def perform_invoke_request(self, toolkit_name, tool_name, request_data):  # pylint: disable=R0912,R0914,R0915
        """ Invoke: perform """
        """ Handle tool invocation """

        import tasknode_task  # pylint: disable=E0401,C0415
        invocation_id = tasknode_task.id

        # Validate toolkit - supports main deepwiki toolkit, deepwiki_query, and wiki_query
        valid_main_toolkits = ["WikiBuilderToolkit", "deepwiki", "Deepwiki", "wiki", "DeepWikiToolkit", "DeepWiki", "Wiki"]
        valid_query_toolkits = ["deepwiki_query", "DeepwikiQuery", "deepwiki-query"]
        valid_wiki_query_toolkits = ["wiki_query", "WikiQuery", "wiki-query"]

        all_valid_toolkits = valid_main_toolkits + valid_query_toolkits + valid_wiki_query_toolkits
        if toolkit_name not in all_valid_toolkits:
            return self._create_error_response(
                invocation_id=invocation_id,
                operation=tool_name,
                model_name=None,
                exception=FileNotFoundError(f"Unknown toolkit: {toolkit_name}. Expected: one of {all_valid_toolkits}"),
                include_traceback=True,
            )

        # Handle wiki_query toolkit - Context7-style multi-wiki queries
        if toolkit_name in valid_wiki_query_toolkits:
            # Only allow wiki discovery, query, and management tools
            valid_wiki_query_tools = ["list_wikis", "resolve_and_ask", "resolve_and_deep_research", "delete_wiki"]
            if tool_name not in valid_wiki_query_tools:
                return self._create_error_response(
                    invocation_id=invocation_id,
                    operation=tool_name,
                    model_name=None,
                    exception=ValueError(f"Tool '{tool_name}' not available in wiki_query toolkit. Available: {', '.join(valid_wiki_query_tools)}"),
                    include_traceback=False,
                )

            # Process wiki_query tools directly (they don't reference other toolkits)
            return self._handle_wiki_query_tool(invocation_id, tool_name, request_data)

        # Handle deepwiki_query toolkit - transform request and use referenced toolkit's config
        if toolkit_name in valid_query_toolkits:
            # Only allow read-only tools for deepwiki_query
            if tool_name not in ["ask", "deep_research"]:
                return self._create_error_response(
                    invocation_id=invocation_id,
                    operation=tool_name,
                    model_name=None,
                    exception=ValueError(f"Tool '{tool_name}' not available in deepwiki_query toolkit. Available: ask, deep_research"),
                    include_traceback=False,
                )

            # Transform request to use referenced toolkit's configuration
            transformed_request, error = self._transform_deepwiki_query_request(request_data)
            if error:
                return self._create_error_response(
                    invocation_id=invocation_id,
                    operation=tool_name,
                    model_name=None,
                    exception=ValueError(error),
                    include_traceback=False,
                )

            # Use the transformed request for the rest of the handler
            request_data = transformed_request
            log.info(f"[deepwiki_query] Delegating {tool_name} to standard handler with transformed request")

        try:
            # Get request data
            # return {
            #     "invocation_id": str(uuid.uuid4()),
            #     "status": "Completed",
            #     "result": json.dumps(request_data),
            #     "result_type": "String",
            # }
            toolkit_params = request_data.get("configuration", {}).get("parameters", {})
            tool_params = request_data.get("parameters", {})
            #
            params = toolkit_params.copy()
            for key, value in tool_params.items():
                if key not in params or value:
                    params[key] = value

            # Route to appropriate tool
            if tool_name == "generate_wiki":
                log.info("Generating Wiki artifacts. Invoking the generate wiki tool")

                self.invocation_stop_checkpoint()

                llm_settings = params.get('llm_settings') or {}
                model_name = None
                if isinstance(llm_settings, dict):
                    model_name = llm_settings.get('model_name')

                # Extract repository configuration from expanded code_toolkit
                repo_config = _extract_repo_config_from_toolkit(params)

                result = self.generate_wiki(
                    query=params["query"],
                    llm_settings=params.get('llm_settings') or {},
                    embedding_model=params.get('embedding_model'),
                    repo_config=repo_config,
                    active_branch=params.get('active_branch', 'main'),
                    force_rebuild_index=params.get('force_rebuild_index', True),
                    indexing_method=params.get('indexing_method', 'filesystem'),
                    # Optional user-facing knobs surfaced from the UI/tool params.
                    planner_mode=params.get('planner_mode') or params.get('planner_type'),
                    exclude_tests=params.get('exclude_tests'),
                    run_in_subprocess=True,
                )
            elif tool_name == "ask":
                log.info("Ask tool invoked. Answering question about repository.")

                self.invocation_stop_checkpoint()

                llm_settings = params.get('llm_settings') or {}

                # Extract repository configuration from expanded code_toolkit
                repo_config = _extract_repo_config_from_toolkit(params)

                # Get question from tool_params
                question = params.get('question', '')
                chat_history = params.get('chat_history', [])

                result = self.ask(
                    question=question,
                    llm_settings=llm_settings,
                    embedding_model=params.get('embedding_model'),
                    repo_config=repo_config,
                    chat_history=chat_history,
                    k=params.get('k', 15),
                    repo_identifier_override=params.get('repo_identifier_override'),
                    analysis_key_override=params.get('analysis_key_override'),
                )
            elif tool_name == "deep_research":
                log.info("Deep research tool invoked. Starting multi-step analysis.")

                self.invocation_stop_checkpoint()

                llm_settings = params.get('llm_settings') or {}

                # Extract repository configuration from expanded code_toolkit
                repo_config = _extract_repo_config_from_toolkit(params)

                # Get question and research params from tool_params
                question = params.get('question', '')
                chat_history = params.get('chat_history', [])
                research_type = params.get('research_type', 'general')
                enable_subagents = params.get('enable_subagents', True)

                result = self.deep_research(
                    question=question,
                    llm_settings=llm_settings,
                    embedding_model=params.get('embedding_model'),
                    repo_config=repo_config,
                    chat_history=chat_history,
                    k=params.get('k', 15),
                    research_type=research_type,
                    enable_subagents=enable_subagents,
                    repo_identifier_override=params.get('repo_identifier_override'),
                    analysis_key_override=params.get('analysis_key_override'),
                )
            else:
                return self._create_error_response(
                    invocation_id=invocation_id,
                    operation=tool_name,
                    model_name=None,
                    exception=FileNotFoundError(f"Unknown tool: {tool_name}"),
                    include_traceback=True,
                )

            # generate output artifacts
            if isinstance(result, dict) and result.get("success"):
                result_objects = []
                
                # For ask tool, use answer; for generate_wiki, use result; for deep_research, use report
                if tool_name == "ask":
                    answer_text = result.get("answer", "Question answered successfully")
                    result_objects.append({
                        "object_type": "message",
                        "result_target": "response",
                        "result_encoding": "plain",
                        "data": answer_text
                    })
                    # Add sources if available
                    sources = result.get("sources", [])
                    if sources:
                        sources_text = "\n\nSources:\n" + "\n".join([f"- {s.get('source', 'unknown')}" for s in sources[:5]])
                        result_objects.append({
                            "object_type": "message",
                            "result_target": "response",
                            "result_encoding": "plain",
                            "data": sources_text
                        })
                elif tool_name == "deep_research":
                    # deep_research response - use report field
                    report_text = result.get("report", result.get("answer", "Deep research completed successfully"))
                    result_objects.append({
                        "object_type": "message",
                        "result_target": "response",
                        "result_encoding": "plain",
                        "data": report_text
                    })
                else:
                    # generate_wiki response
                    result_objects.append({
                        "object_type": "message",
                        "result_target": "response",
                        "result_encoding": "plain",
                        "data": result.get("result", "Wiki generation completed successfully")
                    })

                # Surface partial failures / warnings in-band.
                workflow_errors = result.get("errors") if isinstance(result.get("errors"), list) else []
                failed_pages = result.get("failed_pages") if isinstance(result.get("failed_pages"), list) else []
                if workflow_errors or failed_pages:
                    summary_lines = []
                    if failed_pages:
                        summary_lines.append(f"Failed pages: {len(failed_pages)}")
                    if workflow_errors:
                        summary_lines.append(f"Errors: {len(workflow_errors)}")
                    result_objects.append({
                        "object_type": "message",
                        "result_target": "response",
                        "result_encoding": "plain",
                        "data": "⚠️ Partial issues detected:\n" + "\n".join(summary_lines),
                    })

                    if failed_pages:
                        lines = []
                        for item in failed_pages:
                            if isinstance(item, dict):
                                pid = item.get("page_id") or "(unknown)"
                                title = item.get("title") or ""
                                status = item.get("status") or ""
                                lines.append(f"- {pid} {title} ({status})".strip())
                            else:
                                lines.append(f"- {str(item)}")
                        result_objects.append({
                            "object_type": "message",
                            "result_target": "response",
                            "result_encoding": "plain",
                            "data": "Failed pages:\n" + "\n".join(lines),
                        })

                    if workflow_errors:
                        err_lines = [f"- {str(e)}" for e in workflow_errors]
                        result_objects.append({
                            "object_type": "message",
                            "result_target": "response",
                            "result_encoding": "plain",
                            "data": "Errors:\n" + "\n".join(err_lines),
                        })

                # Add artifacts as objects if available
                artifacts = result.get("artifacts", [])
                for artifact in artifacts:
                    # In Jobs+API mode the worker uploads content artifacts
                    # (pages, manifest, structure) directly to the platform
                    # bucket with correct folder paths.  Pylon's result_objects
                    # mechanism strips directory prefixes from artifact names,
                    # so we must skip re-uploading artifacts the worker already
                    # placed correctly.  The flag ``_uploaded_directly`` is set
                    # by wiki_job_worker.py after a successful direct upload.
                    if artifact.get("_uploaded_directly"):
                        continue

                    # Artifacts are now dictionaries with name, type, and base64 data
                    if artifact.get("type") == "application/json":
                        raw_name = artifact.get("name")
                        name = raw_name if isinstance(raw_name, str) else ""

                        # Determine object type and ensure a stable, non-empty filename.
                        object_type = "wiki_structure"
                        manifest_version_id = None
                        if isinstance(name, str) and "wiki_manifest_" in name:
                            object_type = "wiki_manifest"
                        else:
                            # If name is missing/empty, attempt to detect a manifest by content.
                            data = artifact.get("data")
                            if isinstance(data, str) and data.strip().startswith("{"):
                                try:
                                    parsed = json.loads(data)
                                    if isinstance(parsed, dict) and parsed.get("wiki_version_id") and isinstance(parsed.get("pages"), list):
                                        object_type = "wiki_manifest"
                                        manifest_version_id = str(parsed.get("wiki_version_id"))
                                except Exception:
                                    pass

                        if not (isinstance(name, str) and name.strip()):
                            if object_type == "wiki_manifest":
                                suffix = manifest_version_id or "unknown"
                                name = f"wiki_manifest_{suffix}.json"
                            else:
                                name = "wiki_structure.json"

                        result_objects.append({
                            "name": name,
                            "object_type": object_type,
                            "result_target": "artifact",
                            "result_extension": "json",
                            "result_encoding": "plain",
                            "result_bucket": default_bucket,
                            "data": artifact.get("data", "")
                        })

                    elif artifact.get("type") == "text/markdown":
                        result_objects.append({
                            "name": artifact.get("name", None),
                            "object_type": "wiki_page",
                            "result_target": "artifact",
                            "result_extension": "md",
                            "result_encoding": "plain",
                            "result_bucket": default_bucket,
                            "data": artifact.get("data", "")
                        })

                # Add repository_context as artifact for Ask tool (generate_wiki only)
                if tool_name == "generate_wiki":
                    repository_context = result.get("repository_context")
                    wiki_id = result.get("wiki_id")  # Added by wiki_subprocess_worker
                    if repository_context:
                        # Store in wiki folder: {wiki_id}/repository_context.txt
                        ctx_name = f"{wiki_id}/repository_context.txt" if wiki_id else "repository_context.txt"

                        # In Jobs+API mode, upload directly to avoid Pylon
                        # stripping the directory prefix from result_objects.
                        ctx_uploaded = False
                        try:
                            from ..plugin_implementation.artifact_manager import is_jobs_mode as _is_jobs
                            if _is_jobs() and wiki_id:
                                artifact_settings = extract_artifact_settings(llm_settings)
                                _ctx_client = MiniArtifactClient(artifact_settings)
                                _ctx_client.create_artifact(default_bucket, ctx_name, repository_context)
                                ctx_uploaded = True
                                log.info("Uploaded repository_context directly: %s", ctx_name)
                        except Exception as ctx_up_err:
                            log.debug("Direct context upload failed, falling back to result_objects: %s", ctx_up_err)

                        if not ctx_uploaded:
                            result_objects.append({
                                "name": ctx_name,
                                "object_type": "repository_context",
                                "result_target": "artifact",
                                "result_extension": "txt",
                                "result_encoding": "plain",
                                "result_bucket": default_bucket,
                                "data": repository_context
                            })
                    
                    # Register wiki in global registry (_registry/wikis.json)
                    if wiki_id:
                        try:
                            from ..plugin_implementation.registry_manager import WikiRegistryManager
                            
                            # Create mini artifact client using llm_settings (syngen-style)
                            # llm_settings is available from params at the start of generate_wiki handling
                            artifact_settings = extract_artifact_settings(llm_settings)
                            artifacts_client = MiniArtifactClient(artifact_settings)
                            registry_manager = WikiRegistryManager(artifacts_client, default_bucket)
                            
                            # Extract metadata from result (passed from wiki_subprocess_worker)
                            # canonical_repo_identifier format: "owner/repo:branch:commit"
                            # We need to extract just "owner/repo" for the registry
                            canonical_repo_full = result.get("canonical_repo_identifier", "")
                            # Parse: split by : and take the first part (owner/repo)
                            canonical_repo = canonical_repo_full.split(":")[0] if canonical_repo_full else ""
                            
                            commit_hash = result.get("commit_hash")
                            provider_type = result.get("provider_type", "github")
                            wiki_title = result.get("wiki_title", "")
                            wiki_description = result.get("wiki_description", "")
                            
                            # Count pages from artifacts
                            page_count = sum(
                                1 for art in result.get("artifacts", [])
                                if art.get("type") == "text/markdown" and art.get("name", "").endswith(".md")
                            )
                            
                            registry_manager.register_wiki(
                                wiki_id=wiki_id,
                                repo=canonical_repo,
                                branch=result.get("branch", "main"),
                                provider=provider_type,
                                host=f"{provider_type}.com" if provider_type in ("github", "gitlab") else provider_type,
                                display_name=wiki_title or canonical_repo,
                                description=wiki_description,
                                commit_hash=commit_hash,
                                canonical_repo_identifier=canonical_repo_full or None,
                                analysis_key=result.get("analysis_key"),
                                stats={"page_count": page_count} if page_count else None,
                            )
                            log.info(f"Registered wiki in registry: {wiki_id}")
                        except Exception as reg_err:
                            log.warning(f"Failed to register wiki in registry: {reg_err}")

                return {
                    "invocation_id": invocation_id,
                    "status": "Completed",
                    "result": json.dumps(result_objects),
                    "result_type": "String",
                }
            else:
                err = None
                err_type = None
                err_category = None
                if isinstance(result, dict):
                    err = result.get("error")
                    err_type = result.get("error_type")
                    err_category = result.get("error_category")
                if isinstance(err, str) and err.strip().startswith("[SERVICE_BUSY]"):
                    clean = err.strip()[len("[SERVICE_BUSY]"):].strip()
                    return self._create_error_response(
                        invocation_id=invocation_id,
                        operation=tool_name,
                        model_name=model_name if tool_name == "generate_wiki" else None,
                        exception=RuntimeError(clean or "DeepWiki service is busy. Please try again later."),
                        include_traceback=False,
                    )

                # Preserve explicit error types/categories emitted by subprocess workers.
                exc: Exception
                if isinstance(err_category, str) and err_category == "invalid_input":
                    exc = ValueError(err or "Invalid input")
                elif isinstance(err_type, str) and err_type == "ValueError":
                    exc = ValueError(err or "Invalid input")
                else:
                    exc = RuntimeError(err or "Unknown error")

                # Don't include traceback for user-facing errors - the error message is already clear.
                # Tracebacks are logged server-side for debugging.
                return self._create_error_response(
                    invocation_id=invocation_id,
                    operation=tool_name,
                    model_name=model_name if tool_name == "generate_wiki" else None,
                    exception=exc,
                    include_traceback=False,
                )

        except Exception as e:
            log.exception(f"Tool invocation failed: {toolkit_name}:{tool_name}")
            model_name = None
            try:
                llm_settings = request_data.get("configuration", {}).get("parameters", {}).get("llm_settings") or {}
                if isinstance(llm_settings, dict):
                    model_name = llm_settings.get("model_name")
            except Exception:
                pass
            return self._create_error_response(
                invocation_id=invocation_id,
                operation=tool_name,
                model_name=model_name,
                exception=e,
                include_traceback=True,
            )