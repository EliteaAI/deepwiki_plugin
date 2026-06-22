"""
Feature flags for the deepwiki plugin.

Most of the graph-quality and clustering knobs that used to be gated behind
individual ``DEEPWIKI_*`` environment variables are now hard-coded on. They
have been running stably with their default values long enough that exposing
them as toggles only adds noise — see the graph-quality back-port roadmap.

The cluster planner owns its own set of always-on quality phases
(``hierarchical_leiden``, ``capability_validation``, ``smart_expansion``,
``coverage_ledger``, ``language_hints``). Picking the cluster planner from
the request side is enough to enable them — no env vars required.

Flags that remain env-driven are the ones that intentionally change the
*content* of the wiki (``exclude_tests`` / ``test_linker``) and are gated
by an explicit user choice on the request side.

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


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    """Read a string env var restricted to ``allowed`` values; fall back to ``default``."""
    val = os.environ.get(name, "").strip().lower()
    if val and val in allowed:
        return val
    return default


@dataclass(frozen=True)
class FeatureFlags:
    """Immutable snapshot of plugin feature flags.

    Hard-coded fields are always-on baseline behaviour and are intentionally
    not configurable via env vars any more. Only ``exclude_tests`` and
    ``test_linker`` remain env-driven; everything else takes its default
    from the dataclass.
    """

    # ── User-facing test handling ──────────────────────────────────────
    #: Exclude test code from wiki structure (clustering / page formation).
    #: Test nodes are still indexed and available for vector retrieval,
    #: but they do not participate in clustering or form wiki pages.
    #: Driven by request payload (``exclude_tests``) or DEEPWIKI_EXCLUDE_TESTS.
    exclude_tests: bool = False

    #: Run the test linker (test_node ↔ production_node) over the in-memory graph.
    #: Off by default; opt-in via DEEPWIKI_TEST_LINKER=1.
    test_linker: bool = False

    # ── Cluster planner baseline (always-on; consumed only by cluster code) ──
    #: Replace Louvain two-pass pipeline with hierarchical Leiden.
    hierarchical_leiden: bool = True
    #: Enable deterministic candidate builder + page-quality validator.
    capability_validation: bool = True
    #: Enable shared smart-expansion layer in cluster expansion.
    smart_expansion: bool = True
    #: Enable explicit coverage tracking via the coverage ledger.
    coverage_ledger: bool = True
    #: Enable language-specific heuristics for page shaping.
    language_hints: bool = True

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
    #: Wire markdown document structure: synthesize ``contains`` edges from a
    #: parent ``markdown_document`` node to its ``markdown_section`` children
    #: and ``references`` edges from sections to the code symbols they mention
    #: (roadmap B6/§3.5).
    markdown_structure: bool = True
    #: Extract SQL/DDL schema structure: parse ``.sql`` / ``.ddl`` files into
    #: ``sql_table`` / ``sql_view`` / ``sql_column`` / ``sql_index`` /
    #: ``sql_function`` / ``sql_trigger`` / ``sql_schema`` nodes plus
    #: ``defines`` / ``references`` / ``triggered_by`` / ``calls`` edges
    #: (roadmap C1/§3.1).
    sql_extraction: bool = True
    #: Link ORM model classes (SQLAlchemy ``Column`` / Django ``Field`` /
    #: Hibernate ``@Entity``) to their ``sql_table`` counterparts via a
    #: ``models_table`` cross-language edge (roadmap C4). Runs after SQL
    #: extraction merges ``sql_table`` nodes.
    orm_linking: bool = True
    #: Skip FTS lookups for short / generic stop-token queries.
    fts_stopword_gate: bool = True
    #: Use the tiered T1–T4 lexical cascade inside orphan resolution.
    orphan_lexical_tiered: bool = True
    #: Compute IDF on the orphan symbol name and gate eligible tiers by it.
    orphan_lexical_idf_gate: bool = True
    #: Detect generic REST classes and rewrite the FTS query to use the file stem.
    orphan_rest_disambig: bool = True

    # ── Noise contraction (Path A pilot) ───────────────────────────────
    #: Drop ``variable`` / ``parameter`` / ``field`` nodes after AST parsing
    #: and rewire their edges onto the containing arch node (class / method /
    #: function / struct / interface / module / constructor) with a
    #: ``via=<orig_name>@L<line>`` annotation. The graph stays one hop richer
    #: per arch node and Leiden no longer co-clusters unrelated symbols
    #: through shared parameter/variable names.
    #:
    #: Default ON. Disable with ``DEEPWIKI_CONTRACT_NOISE=0`` to fall back
    #: to the pre-contraction shape (every variable/parameter/field is its
    #: own node). The empirical baseline that motivated this:
    #:
    #: - configurations (Python): 71% nodes / 73% edges removed; top section
    #:   dominance 54.6% → 20.8% on the contracted graph.
    #: - microservices-demo (Go-heavy polyglot): 22% nodes / 53% edges
    #:   removed; top dominance 56.5% → 14.9%.
    contract_noise_nodes: bool = True

    # ── Writer evidence gating (B7) ────────────────────────────────────
    #: Restrict ``_collect_expansion_neighbors`` to ``WRITER_ALLOWED_EDGE_CLASSES``.
    #:
    #: Without this filter, the writer's 1-hop expansion pool includes
    #: edges added by topology enrichment (lexical / directory / doc
    #: proximity / bridge) — synthetic glue that exists for clustering
    #: connectivity but does not represent real architectural coupling.
    #: Citing such an edge looks like grounded reasoning to the LLM but
    #: produces a phantom claim ("X uses Y because the parser noticed
    #: their names share three letters" — not a real relationship).
    #:
    #: Default ON. Disable with ``DEEPWIKI_WRITER_EDGE_CLASS_FILTER=0`` to
    #: fall back to the pre-B7 inclusive expansion pool. See
    #: ``_graph_audit/GAP_ANALYSIS_AND_ROADMAP.md`` §B7.
    writer_edge_class_filter: bool = True

    # ── Weight calibration (A.12 pilot) ────────────────────────────────
    #: Selects how synthetic-edge weights are floored in ``apply_edge_weights``.
    #:
    #: - ``"legacy"``: uniform ``SYNTHETIC_WEIGHT_FLOOR = 0.5`` for
    #:   every synthetic edge class — the pre-Phase-B behaviour.
    #: - ``"calibrated"`` (default): per-class floor table scaled by
    #:   ``raw_similarity`` when available. Higher floor for embedding-derived
    #:   edges, lower for pure heuristics. Documented in
    #:   ``_graph_audit/GAP_ANALYSIS_AND_ROADMAP.md`` §A.12. Promoted to the
    #:   default as the Phase B merge gate — the contract-node algebra and
    #:   downstream phases assume calibrated weights.
    #:
    #: Env: ``DEEPWIKI_WEIGHT_CALIBRATION_PROFILE=legacy|calibrated``.
    weight_calibration_profile: str = "calibrated"


def get_feature_flags() -> FeatureFlags:
    """Build a ``FeatureFlags`` instance, reading the few remaining env knobs."""
    return FeatureFlags(
        exclude_tests=_env_bool("DEEPWIKI_EXCLUDE_TESTS"),
        test_linker=_env_bool("DEEPWIKI_TEST_LINKER"),
        contract_noise_nodes=_env_bool("DEEPWIKI_CONTRACT_NOISE", default=True),
        writer_edge_class_filter=_env_bool(
            "DEEPWIKI_WRITER_EDGE_CLASS_FILTER", default=True,
        ),
        weight_calibration_profile=_env_choice(
            "DEEPWIKI_WEIGHT_CALIBRATION_PROFILE",
            default="calibrated",
            allowed=("legacy", "calibrated"),
        ),
    )
