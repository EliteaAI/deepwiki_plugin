"""
Repository Providers - Multi-provider support for git-based wiki generation

This module provides a unified interface for converting platform toolkit configurations
(from elitea-sdk) into git-clonable URLs that can be consumed by our filesystem indexer.

Supported Providers:
- GitHub (github.com + Enterprise)
- GitLab (gitlab.com + self-hosted)
- Bitbucket (Cloud + Server/Data Center)
- Azure DevOps (dev.azure.com + visualstudio.com)

Usage:
    from repo_providers import RepoProviderFactory, GitCloneConfig
    
    # From toolkit configuration dict
    clone_config = RepoProviderFactory.from_toolkit_config(
        provider_type="github",
        config={
            "base_url": "https://api.github.com",
            "access_token": "ghp_xxxx"
        },
        repository="owner/repo",
        branch="main"
    )
    
    # Use the clone config
    clone_url = clone_config.clone_url  # https://token@github.com/owner/repo.git
    repo_identifier = clone_config.repo_identifier  # owner/repo
"""

from .models import GitCloneConfig, ProviderType
from .factory import RepoProviderFactory, create_clone_config_from_expanded_toolkit
from .providers import (
    BaseRepoProvider,
    GitHubProvider,
    GitLabProvider,
    BitbucketProvider,
    AzureDevOpsProvider,
)

__all__ = [
    "GitCloneConfig",
    "ProviderType",
    "RepoProviderFactory",
    "create_clone_config_from_expanded_toolkit",
    "BaseRepoProvider",
    "GitHubProvider",
    "GitLabProvider",
    "BitbucketProvider",
    "AzureDevOpsProvider",
]
