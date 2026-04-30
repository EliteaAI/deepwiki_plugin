"""
Feature flags for the deepwiki plugin.

Most of the graph-quality knobs that used to be gated behind individual
``DEEPWIKI_*`` environment variables are now hard-coded on. They have been
running stably with their default values long enough that exposing them as
toggles only adds noise — see the graph-quality back-port roadmap.

Flags that remain env-driven are the ones that intentionally change the
*content* of the wiki (test inclusion / linker) or are gated by an explicit
user choice on the request side.

Usage::

    from plugin_implementation.feature_flags import get_feature_flags

    flags = get_feature_flags()
    if flags.exclude_tests:
        ...
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
    """Immutable snapshot of plugin feature flags.

    Hard-coded fields are always-on baseline behaviour and are intentionally
    not configurable via env vars any more. Only ``exclude_tests`` and
    ``test_linker`` remain env-driven; everything else takes its default
    from the dataclass.
    """

    # ── Legacy clustering flags (still env-driven, default off) ────────
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

    # ── User-facing test handling ──────────────────────────────────────
    #: Exclude test code from wiki structure (clustering / page formation).
    #: Test nodes are still indexed and available for vector retrieval,
    #: but they do not participate in clustering or form wiki pages.
    #: Driven by request payload (``exclude_tests``) or DEEPWIKI_EXCLUDE_TESTS.
    exclude_tests: bool = False

    #: Run the test linker (test_node ↔ production_node) over the in-memory graph.
    #: Off by default; opt-in via DEEPWIKI_TEST_LINKER=1.
    test_linker: bool = False

    # ── Hard-coded graph-quality baseline (no env overrides) ───────────
    #: Drop FTS hits below this normalised BM25 score.
    fts_min_score_norm: float = 0.15
    #: Use the v2 orphan cascade (explicit refs → hybrid → tiered → directory).
    orphan_cascade_v2: bool = True
    #: Reuse persisted embeddings via ``get_embedding_by_id`` instead of re-embedding.
    orphan_reuse_embeddings: bool = True
    #: Run hybrid FTS+Vec RRF Pass 2 inside the v2 cascade.
    orphan_hybrid_search: bool = True
    #: RRF constant (TREC default 60).
    orphan_rrf_k: int = 60
    #: Drop fused candidates below this RRF score.
    orphan_rrf_threshold: float = 0.02
    #: Top-N candidates to keep after fusion.
    orphan_hybrid_top_n: int = 20
    #: Node-id construction style; ``"rel_path"`` is collision-safe.
    node_id_style: str = "rel_path"
    #: Build qualified-name and FQN indexes alongside the simple-name index.
    qualified_name_index: bool = True
    #: Run the cross-language linker (L0–L3) over the in-memory graph.
    cross_language_linking: bool = True
    #: Extract API surfaces (REST/gRPC/GraphQL/FFI/...) per node.
    api_surface_extraction: bool = True
    #: Skip FTS lookups for short / generic stop-token queries.
    fts_stopword_gate: bool = True
    #: Use the tiered T1–T4 lexical cascade inside orphan resolution.
    orphan_lexical_tiered: bool = True
    #: Compute IDF on the orphan symbol name and gate eligible tiers by it.
    orphan_lexical_idf_gate: bool = True
    #: Detect generic REST classes and rewrite the FTS query to use the file stem.
    orphan_rest_disambig: bool = True


def get_feature_flags() -> FeatureFlags:
    """Build a ``FeatureFlags`` instance, reading the few remaining env knobs."""
    return FeatureFlags(
        hierarchical_leiden=_env_bool("DEEPWIKI_CLUSTER_HIERARCHICAL_LEIDEN"),
        capability_validation=_env_bool("DEEPWIKI_CLUSTER_CAPABILITY_VALIDATION"),
        smart_expansion=_env_bool("DEEPWIKI_CLUSTER_SMART_EXPANSION"),
        coverage_ledger=_env_bool("DEEPWIKI_CLUSTER_COVERAGE_LEDGER"),
        language_hints=_env_bool("DEEPWIKI_CLUSTER_LANGUAGE_HINTS"),
        exclude_tests=_env_bool("DEEPWIKI_EXCLUDE_TESTS"),
        test_linker=_env_bool("DEEPWIKI_TEST_LINKER"),
    )
