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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


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

    # ── Graph-quality back-port (Phases 0–6) ───────────────────────────
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

    #: ``"stem"`` (legacy: ``language::file_name::qualified_name``) or
    #: ``"rel_path"`` (collision-safe: ``language::relpath::qualified_name``).
    node_id_style: str = "rel_path"

    #: Build qualified-name and FQN indexes alongside the simple-name index.
    qualified_name_index: bool = True

    #: Run the cross-language linker (L0–L3) over the in-memory graph.
    cross_language_linking: bool = True

    #: Run the test linker (test_node ↔ production_node) over the in-memory graph.
    test_linker: bool = False

    #: Extract API surfaces (REST/gRPC/GraphQL/FFI/...) per node.
    api_surface_extraction: bool = True

    # ── Phase 1/2 lexical hardening (graph-quality roadmap) ────────────
    #: Skip FTS lookups for short / generic stop-token queries.
    fts_stopword_gate: bool = True

    #: Use the tiered T1–T4 lexical cascade (with directory partitioning
    #: and ARCH-target filtering) inside orphan resolution. When off, the
    #: legacy flat ``search_fts5`` path is used (still gated by stopwords
    #: and ``fts_min_score_norm``).
    orphan_lexical_tiered: bool = True

    #: Compute IDF on the orphan symbol name and gate eligible tiers by
    #: it. Disable to keep the same behaviour without IDF gating.
    orphan_lexical_idf_gate: bool = True

    #: Detect generic REST classes (``API`` / ``Resource`` / ...) and
    #: rewrite the FTS query to ``"<file_stem> <symbol>"``.
    orphan_rest_disambig: bool = True


def get_feature_flags() -> FeatureFlags:
    """Build a ``FeatureFlags`` instance from the current environment."""
    return FeatureFlags(
        hierarchical_leiden=_env_bool("DEEPWIKI_CLUSTER_HIERARCHICAL_LEIDEN"),
        capability_validation=_env_bool("DEEPWIKI_CLUSTER_CAPABILITY_VALIDATION"),
        smart_expansion=_env_bool("DEEPWIKI_CLUSTER_SMART_EXPANSION"),
        coverage_ledger=_env_bool("DEEPWIKI_CLUSTER_COVERAGE_LEDGER"),
        language_hints=_env_bool("DEEPWIKI_CLUSTER_LANGUAGE_HINTS"),
        exclude_tests=_env_bool("DEEPWIKI_EXCLUDE_TESTS"),
        fts_min_score_norm=_env_float("DEEPWIKI_FTS_MIN_SCORE_NORM", 0.15),
        orphan_cascade_v2=_env_bool("DEEPWIKI_ORPHAN_CASCADE_V2", True),
        orphan_reuse_embeddings=_env_bool("DEEPWIKI_ORPHAN_REUSE_EMBEDDINGS", True),
        orphan_hybrid_search=_env_bool("DEEPWIKI_ORPHAN_HYBRID_SEARCH", True),
        orphan_rrf_k=_env_int("DEEPWIKI_ORPHAN_RRF_K", 60),
        orphan_rrf_threshold=_env_float("DEEPWIKI_ORPHAN_RRF_THRESHOLD", 0.02),
        orphan_hybrid_top_n=_env_int("DEEPWIKI_ORPHAN_HYBRID_TOP_N", 20),
        node_id_style=(os.environ.get("DEEPWIKI_NODE_ID_STYLE", "rel_path").strip().lower() or "rel_path"),
        qualified_name_index=_env_bool("DEEPWIKI_QUALIFIED_NAME_INDEX", True),
        cross_language_linking=_env_bool("DEEPWIKI_CROSS_LANGUAGE_LINKING", True),
        test_linker=_env_bool("DEEPWIKI_TEST_LINKER", False),
        api_surface_extraction=_env_bool("DEEPWIKI_API_SURFACE_EXTRACTION", True),
        fts_stopword_gate=_env_bool("DEEPWIKI_FTS_STOPWORD_GATE", True),
        orphan_lexical_tiered=_env_bool("DEEPWIKI_ORPHAN_LEXICAL_TIERED", True),
        orphan_lexical_idf_gate=_env_bool("DEEPWIKI_ORPHAN_LEXICAL_IDF_GATE", True),
        orphan_rest_disambig=_env_bool("DEEPWIKI_ORPHAN_REST_DISAMBIG", True),
    )
