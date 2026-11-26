"""
Data models for repository providers.

These models define the configuration and output structures for converting
platform toolkit configs to git-clonable URLs.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from urllib.parse import urlparse


_SCP_SSH_HOST_RE = re.compile(r"^[^/@\s]+@([^:/\s]+):")


def extract_hostname(url_or_host: str) -> str:
    """Extract normalized lowercase hostname (no credentials or port) from a URL or host string."""
    if not url_or_host:
        return ""
    value = url_or_host.strip()
    if not value:
        return ""
    if "://" in value:
        hostname = urlparse(value).hostname or ""
    else:
        ssh_match = _SCP_SSH_HOST_RE.match(value)
        if ssh_match:
            hostname = ssh_match.group(1)
        else:
            hostname = urlparse(f"https://{value}").hostname or ""
    return hostname.lower().rstrip(".")


def host_matches_domain(url_or_host: str, domain: str) -> bool:
    """Return True iff the parsed hostname equals *domain* or is a subdomain of *domain*."""
    hostname = extract_hostname(url_or_host)
    target = domain.lower().rstrip(".")
    if not hostname or not target:
        return False
    return hostname == target or hostname.endswith(f".{target}")


class ProviderType(str, Enum):
    """Supported repository provider types."""
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    AZURE_DEVOPS = "ado"
    
    @classmethod
    def from_string(cls, value: str) -> "ProviderType":
        """Convert string to ProviderType, with common aliases."""
        aliases = {
            "github": cls.GITHUB,
            "gh": cls.GITHUB,
            "gitlab": cls.GITLAB,
            "gl": cls.GITLAB,
            "bitbucket": cls.BITBUCKET,
            "bb": cls.BITBUCKET,
            "ado": cls.AZURE_DEVOPS,
            "ado_repos": cls.AZURE_DEVOPS,
            "azure": cls.AZURE_DEVOPS,
            "azure_devops": cls.AZURE_DEVOPS,
            "azuredevops": cls.AZURE_DEVOPS,
            "vsts": cls.AZURE_DEVOPS,
        }
        normalized = value.lower().strip()
        if normalized in aliases:
            return aliases[normalized]
        raise ValueError(f"Unknown provider type: {value}")
    
    @classmethod
    def detect_from_url(cls, url: str) -> Optional["ProviderType"]:
        """Detect provider type from a URL's parsed hostname."""
        hostname = extract_hostname(url)
        if not hostname:
            return None
        labels = hostname.split(".")

        if host_matches_domain(hostname, "github.com") or (
            len(labels) >= 2 and labels[0] == "github" and labels[1] != "com"
        ):
            return cls.GITHUB
        if host_matches_domain(hostname, "gitlab.com") or (
            len(labels) >= 2 and labels[0] == "gitlab" and labels[1] != "com"
        ):
            return cls.GITLAB
        if host_matches_domain(hostname, "bitbucket.org") or (
            len(labels) >= 2 and labels[0] == "bitbucket" and labels[1] != "org"
        ):
            return cls.BITBUCKET
        if (
            host_matches_domain(hostname, "dev.azure.com")
            or host_matches_domain(hostname, "visualstudio.com")
        ):
            return cls.AZURE_DEVOPS

        return None


@dataclass
class GitCloneConfig:
    """
    Configuration for cloning a repository via git.
    
    This is the output of provider conversion - contains everything needed
    to clone a repository using the git binary.
    
    Attributes:
        clone_url: The authenticated HTTPS URL for git clone
        repo_identifier: Canonical identifier (e.g., "owner/repo" for GitHub)
        branch: Branch to clone (default: main)
        provider: The provider type
        host: The git host (e.g., github.com, gitlab.example.com)
        api_url: Optional API base URL for the provider
        requires_auth: Whether authentication is required
        auth_method: Authentication method used (token, password, ssh, etc.)
        metadata: Additional provider-specific metadata
    """
    clone_url: str
    repo_identifier: str
    # None means "unresolved at clone time" (#7273): callers must resolve the
    # repository's real default branch, never default this to 'main'.
    branch: Optional[str] = "main"
    provider: ProviderType = ProviderType.GITHUB
    host: str = ""
    api_url: str = ""
    requires_auth: bool = False
    auth_method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize fields after initialization."""
        if not self.clone_url:
            raise ValueError("clone_url is required")
        if not self.repo_identifier:
            raise ValueError("repo_identifier is required")
        
        # Extract host from clone_url if not provided
        if not self.host:
            parsed = urlparse(self.clone_url.replace("git@", "https://").split(":")[0])
            self.host = parsed.netloc or ""
    
    @property
    def safe_clone_url(self) -> str:
        """Return clone URL with credentials redacted for logging."""
        parsed = urlparse(self.clone_url)
        if parsed.password:
            # Has explicit password
            safe = f"{parsed.scheme}://{parsed.username}:***@{parsed.netloc.split('@')[-1]}{parsed.path}"
            return safe
        elif "@" in self.clone_url and parsed.username:
            # Has token as username
            return f"{parsed.scheme}://***@{parsed.netloc.split('@')[-1]}{parsed.path}"
        return self.clone_url
    
    @property
    def safe_url(self) -> str:
        """Alias for safe_clone_url for convenience."""
        return self.safe_clone_url
    
    def sanitize_output(self, text: str) -> str:
        """
        Remove any credentials from output text (e.g., error messages).
        
        This extracts credentials from the clone_url and replaces any occurrence
        in the provided text with redacted placeholders.
        """
        if not text:
            return text
        
        result = text
        parsed = urlparse(self.clone_url)
        
        # Remove password if present
        if parsed.password:
            result = result.replace(parsed.password, '<redacted>')
        
        # Remove username/token if present and looks like a token
        if parsed.username:
            # Only redact if it looks like a token (not a regular username)
            # Tokens are typically long alphanumeric strings
            if len(parsed.username) > 20 or any(c.isdigit() for c in parsed.username):
                result = result.replace(parsed.username, '<token>')
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "clone_url": self.safe_clone_url,  # Always use safe URL in dict
            "repo_identifier": self.repo_identifier,
            "branch": self.branch,
            "provider": self.provider.value,
            "host": self.host,
            "api_url": self.api_url,
            "requires_auth": self.requires_auth,
            "auth_method": self.auth_method,
            "metadata": self.metadata,
        }


@dataclass
class ProviderConfig:
    """
    Normalized provider configuration.
    
    This is the input format after normalizing toolkit configs from various sources.
    """
    provider: ProviderType
    host: str  # Base host (e.g., github.com, gitlab.example.com)
    api_url: str  # API base URL
    
    # Repository identification
    repository: str  # Repository path/identifier
    # None means "unresolved" (#7273) — never defaulted to 'main' here.
    branch: Optional[str] = "main"
    
    # Authentication (various methods supported)
    token: Optional[str] = None  # Access token / PAT
    username: Optional[str] = None  # For username/password auth
    password: Optional[str] = None  # Password or app password
    
    # Provider-specific
    project: Optional[str] = None  # ADO project, Bitbucket workspace
    organization: Optional[str] = None  # ADO organization
    
    # Flags
    is_cloud: bool = True  # True for SaaS, False for self-hosted
    
    def has_auth(self) -> bool:
        """Check if any authentication method is configured."""
        return bool(self.token or (self.username and self.password))
    
    def get_auth_method(self) -> str:
        """Determine which authentication method is configured."""
        if self.token:
            return "token"
        elif self.username and self.password:
            return "password"
        return "anonymous"
