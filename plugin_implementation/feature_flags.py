"""
Feature flags for gradual rollout of clustering improvements.

All flags default to **disabled** so that existing behavior is unchanged
until a flag is explicitly turned on via environment variable.

Usage::

    from plugin_implementation.feature_flags import get_feature_flags

    flags = get_feature_flags()
    if flags.hierarchical_leiden:
        ...  # new path
    else:
        ...  # legacy path
"""

import os
from dataclasses import dataclass

__all__ = ["FeatureFlags", "get_feature_flags"]


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean from an environment variable (``1`` / ``true`` / ``yes``)."""
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes")


@dataclass(frozen=True)
class FeatureFlags:
    """Immutable snapshot of clustering feature flags.

    Every flag is ``False`` by default — the legacy pipeline runs unless
    the corresponding environment variable is set to ``1`` / ``true``.
    """

    #: Replace Louvain two-pass pipeline with hierarchical Leiden.
    hierarchical_leiden: bool = False

    #: Enable deterministic candidate builder + page-quality validator.
    capability_validation: bool = False

    #: Enable shared smart-expansion layer in cluster expansion.
    smart_expansion: bool = False

    #: Enable explicit coverage tracking via the coverage ledger.
    coverage_ledger: bool = False

    #: Enable language-specific heuristics for page shaping.
    language_hints: bool = False

    #: Exclude test code from wiki structure (clustering / page formation).
    #: Test nodes are still indexed and available for vector retrieval,
    #: but they do not participate in clustering or form wiki pages.
    exclude_tests: bool = False


def get_feature_flags() -> FeatureFlags:
    """Build a ``FeatureFlags`` instance from the current environment."""
    return FeatureFlags(
        hierarchical_leiden=_env_bool("DEEPWIKI_CLUSTER_HIERARCHICAL_LEIDEN"),
        capability_validation=_env_bool("DEEPWIKI_CLUSTER_CAPABILITY_VALIDATION"),
        smart_expansion=_env_bool("DEEPWIKI_CLUSTER_SMART_EXPANSION"),
        coverage_ledger=_env_bool("DEEPWIKI_CLUSTER_COVERAGE_LEDGER"),
        language_hints=_env_bool("DEEPWIKI_CLUSTER_LANGUAGE_HINTS"),
        exclude_tests=_env_bool("DEEPWIKI_EXCLUDE_TESTS"),
    )
