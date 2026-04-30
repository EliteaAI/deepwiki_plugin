"""
Phase 2 — Edge Weighting, Hub Detection & Semantic Edge Injection

Takes the unified graph (from Phase 1 ``unified_db.py``) and reshapes its
topology so that downstream clustering (Phase 3) produces meaningful,
capability-based partitions rather than noisy, hub-dominated groups.

Three transformations applied in order:

1. **Inverse in-degree weighting** — Every edge (u, v) gets
   ``weight = 1 / log(InDegree(v) + 2)``. High-fan-in utility nodes
   (loggers, base classes, config) receive tiny weights; direct
   capability edges stay strong.

2. **Hub quarantine** — Nodes with in-degree Z-score > 3.0 are flagged
   ``is_hub = 1`` and will be *excluded* from Louvain clustering
   (Phase 3), then reattached via majority-vote assignment.

3. **Semantic edge injection** — Orphan nodes (in-degree == 0 AND
   out-degree == 0) are linked to the graph via:
   a. FTS5 lexical matching (symbol name → code symbols)
   b. Vector KNN (embedding similarity with expanding path prefix)
   This is the key innovation: docs ↔ code, event handlers ↔ emitters,
   cross-language bindings all become first-class edges.

Feature-flagged via ``DEEPWIKI_UNIFIED_DB=1`` (same flag as Phase 1).

Usage::

    from .graph_topology import apply_edge_weights, detect_hubs, resolve_orphans

    apply_edge_weights(G)           # mutates G in-place
    hubs = detect_hubs(G)           # returns Set[str]
    edges_added = resolve_orphans(  # injects lexical/semantic edges
        db, G, embedding_fn=embed_text
    )
    persist_weights_to_db(db, G)    # syncs all weights back to SQLite
"""

import logging
import math
import os
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx

# Optional sibling-module imports — guarded so this module remains
# usable when imported via a sys.path insertion (legacy tests) rather
# than as part of the ``plugin_implementation`` package.
try:
    from .feature_flags import FeatureFlags, get_feature_flags  # type: ignore
    from .graph_orphan_cascade_v2 import (  # type: ignore
        collect_orphan_embeddings,
        resolve_orphans_explicit_refs,
    )
    from .graph_orphan_hybrid import resolve_orphans_hybrid  # type: ignore
except ImportError:  # pragma: no cover — non-package import fallback
    try:
        from feature_flags import FeatureFlags, get_feature_flags  # type: ignore
        from graph_orphan_cascade_v2 import (  # type: ignore
            collect_orphan_embeddings,
            resolve_orphans_explicit_refs,
        )
        from graph_orphan_hybrid import resolve_orphans_hybrid  # type: ignore
    except ImportError:
        FeatureFlags = None  # type: ignore[assignment]
        get_feature_flags = None  # type: ignore[assignment]
        collect_orphan_embeddings = None  # type: ignore[assignment]
        resolve_orphans_explicit_refs = None  # type: ignore[assignment]
        resolve_orphans_hybrid = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# We intentionally avoid importing numpy as a hard dependency.
# The hub detection Z-score math only needs mean/std over a list of ints.
# If numpy is available we use it for speed; otherwise pure Python.
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# Weight floor for synthetic edges (orphan resolution, doc injection).
# Equivalent to a structural edge pointing at a node with in-degree ~5.
# Ensures these edges are strong enough for Leiden to group on.
SYNTHETIC_WEIGHT_FLOOR = 0.5


# ═══════════════════════════════════════════════════════════════════════════
# 1. Inverse In-Degree Edge Weighting
# ═══════════════════════════════════════════════════════════════════════════

def apply_edge_weights(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """Assign inverse in-degree weights to every edge in *G* (in-place).

    For each edge (u, v):

        weight = 1 / log(InDegree(v) + 2)

    **In-degree is computed from structural edges only** — synthetic edges
    injected by orphan resolution (``directory_link``, ``lexical_link``,
    ``semantic_link``) and doc edges are excluded from the degree count.
    This prevents orphan anchors from having their genuine structural
    edges crushed by inflated in-degree.

    Synthetic edges receive a **weight floor** of ``SYNTHETIC_WEIGHT_FLOOR``
    so that Leiden treats them as meaningful community signals rather than
    noise.

    Returns:
        Stats dict with min/max/mean weight and distribution info.
    """
    if G.number_of_edges() == 0:
        return {"edges_weighted": 0, "min": 0, "max": 0, "mean": 0}

    # Compute in-degree using ONLY structural edges (AST-derived).
    # Synthetic edges (orphan resolution, doc injection) are excluded
    # so they don't inflate anchor degrees and crush real edges.
    _SYNTHETIC_CLASSES = frozenset({"directory", "lexical", "semantic", "doc", "bridge"})
    structural_in: Dict[str, int] = {}
    for _u, v, data in G.edges(data=True):
        if data.get("edge_class", "structural") in _SYNTHETIC_CLASSES:
            continue
        structural_in[v] = structural_in.get(v, 0) + 1

    weights = []
    synthetic_count = 0
    for u, v, key, data in G.edges(data=True, keys=True):
        in_deg = structural_in.get(v, 0)
        w = 1.0 / math.log(in_deg + 2)

        # Synthetic recovery edges get a weight floor so Leiden
        # actually groups orphan nodes with their anchors.
        if data.get("edge_class", "structural") in _SYNTHETIC_CLASSES:
            w = max(w, SYNTHETIC_WEIGHT_FLOOR)
            synthetic_count += 1

        data["weight"] = w
        weights.append(w)

    stats = {
        "edges_weighted": len(weights),
        "synthetic_floored": synthetic_count,
        "min": round(min(weights), 4),
        "max": round(max(weights), 4),
        "mean": round(sum(weights) / len(weights), 4),
    }

    logger.info(
        "Edge weighting complete: %d edges (%d synthetic floored at %.2f), "
        "weight range [%.4f, %.4f], mean %.4f",
        stats["edges_weighted"], synthetic_count, SYNTHETIC_WEIGHT_FLOOR,
        stats["min"], stats["max"], stats["mean"],
    )
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# 2. Hub Detection (Z-Score on In-Degree)
# ═══════════════════════════════════════════════════════════════════════════

def detect_hubs(
    G: nx.MultiDiGraph,
    z_threshold: float = 3.0,
) -> Set[str]:
    """Return node IDs whose in-degree Z-score exceeds *z_threshold*.

    A hub is a node that is imported/called by a disproportionate number
    of other nodes — e.g. loggers, base classes, config singletons.
    Including them in Louvain clustering distorts communities because
    they pull unrelated nodes together.

    Args:
        G: The graph (after weighting).
        z_threshold: Standard deviations above mean to flag as hub.
            Default 3.0 (≈ top 0.13% under normal distribution).

    Returns:
        Set of hub node IDs.
    """
    if G.number_of_nodes() < 3:
        return set()

    nodes_list = list(G.nodes())
    degrees = [G.in_degree(n) for n in nodes_list]

    if _HAS_NUMPY:
        arr = np.array(degrees, dtype=float)
        mean = arr.mean()
        std = arr.std()
    else:
        n = len(degrees)
        mean = sum(degrees) / n
        variance = sum((d - mean) ** 2 for d in degrees) / n
        std = math.sqrt(variance)

    if std == 0:
        return set()

    hubs = set()
    for i, deg in enumerate(degrees):
        z = (deg - mean) / std
        if z > z_threshold:
            hubs.add(nodes_list[i])

    if hubs:
        logger.info(
            "Hub detection: %d hubs found (Z > %.1f) out of %d nodes. "
            "Mean in-degree=%.1f, std=%.1f",
            len(hubs), z_threshold, G.number_of_nodes(), mean, std,
        )
        # Log the top-5 hubs by degree for debugging
        hub_degs = sorted(
            [(n, G.in_degree(n)) for n in hubs],
            key=lambda x: x[1], reverse=True,
        )
        for nid, deg in hub_degs[:5]:
            logger.debug("  Hub: %s (in-degree=%d)", nid, deg)
    else:
        logger.info(
            "Hub detection: no hubs found (Z > %.1f). "
            "Mean in-degree=%.1f, std=%.1f",
            z_threshold, mean, std,
        )

    return hubs


def flag_hubs_in_db(db, hubs: Set[str]) -> None:
    """Persist hub flags into the unified DB.

    Args:
        db: ``UnifiedWikiDB`` instance.
        hubs: Set of node IDs flagged as hubs.
    """
    if not hubs:
        return

    for nid in hubs:
        db.set_hub(nid, is_hub=True)
    db.conn.commit()

    logger.info("Flagged %d hubs in unified DB", len(hubs))


# ═══════════════════════════════════════════════════════════════════════════
# 3. Semantic Edge Injection (Orphan Resolution)
# ═══════════════════════════════════════════════════════════════════════════

def _expanding_prefixes(rel_path: str) -> List[str]:
    """Generate expanding path prefixes for locality-biased search.

    Given ``src/auth/handlers/login.py`` returns:
    - ``src/auth/handlers``   (same directory)
    - ``src/auth``            (parent directory)
    - ``src``                 (grandparent)
    - ``""``                  (global / root)

    The caller searches each prefix in order, stopping at the first
    one that yields results. This ensures locality: same-directory
    matches are preferred over distant ones.
    """
    parts = rel_path.replace("\\", "/").split("/")
    # Drop the filename itself
    dir_parts = parts[:-1] if len(parts) > 1 else []

    prefixes = []
    while dir_parts:
        prefixes.append("/".join(dir_parts))
        dir_parts = dir_parts[:-1]

    prefixes.append("")  # global fallback (no prefix filter)
    return prefixes


def find_orphans(G: nx.MultiDiGraph) -> List[str]:
    """Return node IDs with both in-degree == 0 and out-degree == 0.

    These are completely disconnected nodes — documentation files,
    standalone scripts, event handlers, cross-language stubs, etc.
    They need synthetic edges to participate in clustering.
    """
    return [
        n for n in G.nodes()
        if G.in_degree(n) == 0 and G.out_degree(n) == 0
    ]


def _resolve_orphans_by_directory(
    db,
    G: nx.MultiDiGraph,
    orphan_ids: List[str],
) -> int:
    """Connect remaining orphans to the best anchor in same/parent directory.

    For each orphan:
    1. Look up its ``rel_path`` from the DB.
    2. Walk up the directory tree (same dir → parent → grandparent → root).
    3. At each level, find the connected (non-orphan) node with the
       highest total degree — the best "anchor" to attach to.
    4. Add a ``directory_link`` edge (orphan → anchor).

    This is the last-resort fallback when semantic embeddings are
    unavailable.  It ensures that files in the same directory end up
    in the same Leiden community rather than creating thousands of
    singleton components.

    Returns:
        Number of edges added.
    """
    if not orphan_ids:
        return 0

    orphan_set = set(orphan_ids)

    # Build directory → [(node_id, total_degree)] for NON-orphan nodes
    dir_index: Dict[str, List[Tuple[str, int]]] = {}
    for nid, data in G.nodes(data=True):
        if nid in orphan_set:
            continue
        rp = data.get("rel_path", "")
        if not rp:
            row = db.get_node(nid)
            rp = row.get("rel_path", "") if row else ""
        if not rp:
            continue
        d = os.path.dirname(rp).replace("\\", "/") or "<root>"
        deg = G.in_degree(nid) + G.out_degree(nid)
        dir_index.setdefault(d, []).append((nid, deg))

    # Sort each directory bucket by degree descending (best anchors first)
    for d in dir_index:
        dir_index[d].sort(key=lambda x: x[1], reverse=True)

    edges_added = 0

    for orphan_id in orphan_ids:
        node = db.get_node(orphan_id)
        if not node:
            continue
        rp = node.get("rel_path", "")
        if not rp:
            continue

        # Walk up the directory tree
        for prefix in _expanding_prefixes(rp):
            d = prefix if prefix else "<root>"
            candidates = dir_index.get(d, [])
            if candidates:
                anchor_id = candidates[0][0]  # highest-degree node
                _add_edge(
                    db, G,
                    source=orphan_id,
                    target=anchor_id,
                    rel_type="directory_link",
                    edge_class="directory",
                    created_by="dir_proximity_fallback",
                )
                edges_added += 1
                break  # resolved — stop climbing

    logger.info(
        "Directory proximity fallback: %d/%d orphans connected",
        edges_added, len(orphan_ids),
    )
    return edges_added


def _resolve_orphans_v2(
    db,
    G: nx.MultiDiGraph,
    orphans: List[str],
    resolved: Set[str],
    stats: Dict[str, Any],
    flags,
    t_start: float,
    *,
    embedding_fn: Optional[Callable] = None,
    embed_batch_fn: Optional[Callable] = None,
    fts_limit: int = 3,
    vec_k: int = 3,
    vec_distance_threshold: float = 0.15,
    max_lexical_edges: int = 2,
    embed_batch_size: int = 64,
    vec_prefix_depth: Optional[int] = None,
    embed_max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """4-pass cascade: explicit-ref → hybrid RRF → tiered lexical → directory.

    Owns all edge writes; the building-block modules
    (``graph_orphan_cascade_v2``, ``graph_orphan_hybrid``,
    ``graph_lexical_v2``) are pure and only return hit lists.
    """
    import time as _time

    stats.setdefault("explicit_ref", 0)

    # ── Pass 1 — explicit references (markdown links, backticks, imports)
    pass1_t = _time.monotonic()
    try:
        explicit_hits = resolve_orphans_explicit_refs(
            db, G, orphans, flags=flags,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("explicit-ref pass failed: %s", exc)
        explicit_hits = {}

    for nid, hits in explicit_hits.items():
        if not hits:
            continue
        for hit in hits:
            tgt = hit.get("node_id")
            if not tgt or tgt == nid or not G.has_node(tgt):
                continue
            matcher = hit.get("_matcher", "explicit_ref")
            edge_class = "doc" if matcher == "md_link" else "lexical"
            _add_edge(
                db, G,
                source=nid, target=tgt,
                rel_type="explicit_ref",
                edge_class=edge_class,
                created_by="explicit_ref_v2",
                raw_similarity=float(hit.get("_raw_score", 0.95)),
                skip_db=True,
                provenance={
                    "source": "explicit_ref",
                    "matcher": matcher,
                    "raw_score": hit.get("_raw_score", 0.95),
                },
            )
            stats["explicit_ref"] += 1
        resolved.add(nid)

    logger.info(
        "[ORPHAN] Pass 1 (explicit refs): %d edges, %d resolved in %.1fs",
        stats["explicit_ref"], len(resolved), _time.monotonic() - pass1_t,
    )

    # ── Pass 2 — hybrid FTS+Vec via RRF
    stats.setdefault("hybrid", 0)
    pass2_t = _time.monotonic()
    pending = [nid for nid in orphans if nid not in resolved]
    orphan_embs: Optional[Dict[str, Optional[List[float]]]] = None
    if flags.orphan_reuse_embeddings and pending:
        try:
            orphan_embs = collect_orphan_embeddings(db, pending)
        except Exception as exc:  # noqa: BLE001
            logger.debug("collect_orphan_embeddings failed: %s", exc)
            orphan_embs = None

    if pending and flags.orphan_hybrid_search:
        for nid in pending:
            try:
                hits = resolve_orphans_hybrid(
                    db, G, nid,
                    orphan_embeddings=orphan_embs,
                    embed_fn=embedding_fn,
                    flags=flags,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("hybrid pass failed for %s: %s", nid, exc)
                continue
            if not hits:
                continue
            added = 0
            for hit in hits[:max_lexical_edges]:
                tgt = hit.get("node_id")
                if not tgt or tgt == nid or not G.has_node(tgt):
                    continue
                _add_edge(
                    db, G,
                    source=nid, target=tgt,
                    rel_type="hybrid_link",
                    edge_class="semantic",
                    created_by="hybrid_rrf",
                    raw_similarity=float(hit.get("rrf_score", 0.0)),
                    skip_db=True,
                    provenance={
                        "source": "hybrid_rrf",
                        "rrf_score": hit.get("rrf_score"),
                        "fts_rank": hit.get("fts_rank"),
                        "vec_rank": hit.get("vec_rank"),
                    },
                )
                stats["hybrid"] += 1
                added += 1
            if added:
                resolved.add(nid)

    logger.info(
        "[ORPHAN] Pass 2 (hybrid RRF): %d edges, %d resolved in %.1fs",
        stats["hybrid"], len(resolved), _time.monotonic() - pass2_t,
    )

    # ── Pass 3 — tiered lexical T1–T4 (only on still-unresolved orphans)
    pass3_t = _time.monotonic()
    pending = [nid for nid in orphans if nid not in resolved]
    if pending:
        try:
            from .graph_lexical_v2 import resolve_orphans_lexical_tiered as _tiered
        except Exception:  # noqa: BLE001
            try:
                from graph_lexical_v2 import resolve_orphans_lexical_tiered as _tiered  # type: ignore
            except Exception as exc:  # noqa: BLE001
                logger.debug("graph_lexical_v2 unavailable: %s", exc)
                _tiered = None  # type: ignore[assignment]
        if _tiered is not None:
            for nid in pending:
                try:
                    hits = _tiered(
                        db, G, nid,
                        fts_limit=max(fts_limit, 8),
                        max_lexical_edges=max_lexical_edges,
                        flags=flags,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("tiered lexical failed for %s: %s", nid, exc)
                    continue
                if not hits:
                    continue
                for hit in hits:
                    tgt = hit.get("node_id")
                    if not tgt or tgt == nid or not G.has_node(tgt):
                        continue
                    _add_edge(
                        db, G,
                        source=nid, target=tgt,
                        rel_type="lexical_link",
                        edge_class="lexical",
                        created_by="fts5_lexical_v2",
                        skip_db=True,
                        provenance={
                            "source": "fts_lexical_tiered",
                            "tier": hit.get("_tier"),
                            "score_norm": hit.get("score_norm"),
                        },
                    )
                    stats["lexical"] += 1
                resolved.add(nid)

    logger.info(
        "[ORPHAN] Pass 3 (tiered lexical): %d edges, %d resolved in %.1fs",
        stats["lexical"], len(resolved), _time.monotonic() - pass3_t,
    )

    # ── Pass 4 — directory proximity fallback
    pass4_t = _time.monotonic()
    remaining = find_orphans(G)
    if remaining:
        stats["directory"] = _resolve_orphans_by_directory(db, G, remaining)

    stats["orphans_remaining"] = len(find_orphans(G))
    stats["resolved"] = stats["orphans_found"] - stats["orphans_remaining"]

    logger.info(
        "[ORPHAN] v2 total: %d/%d resolved in %.1fs "
        "(explicit=%d, hybrid=%d, lexical=%d, directory=%d), %d remaining",
        stats["resolved"], stats["orphans_found"],
        _time.monotonic() - t_start,
        stats["explicit_ref"], stats["hybrid"], stats["lexical"],
        stats["directory"], stats["orphans_remaining"],
    )

    return stats


# ─── Phase 1 lexical query gating ────────────────────────────────────────────

#: Tokens that are too generic to drive lexical orphan resolution.
_LEXICAL_STOPWORDS: frozenset = frozenset({
    "init", "main", "test", "tests", "setup", "config", "utils", "helper",
    "helpers", "common", "base", "abstract", "interface", "impl", "data",
    "model", "service", "factory", "manager", "handler", "controller",
    "view", "router", "routes", "api", "lib", "src", "app", "core",
})

#: Minimum query length (after stripping) for an FTS lookup to be issued.
_MIN_FTS_QUERY_LEN: int = 4


def _is_low_value_lexical_query(query: str) -> bool:
    """Return True when *query* should be skipped by lexical lookups.

    Skipped when shorter than ``_MIN_FTS_QUERY_LEN`` *or* lower-case
    form is in ``_LEXICAL_STOPWORDS``. Gated by
    :attr:`FeatureFlags.fts_stopword_gate`.
    """
    flags = get_feature_flags() if get_feature_flags is not None else None
    if flags is not None and not flags.fts_stopword_gate:
        return False
    if not query:
        return True
    cleaned = query.strip()
    if len(cleaned) < _MIN_FTS_QUERY_LEN:
        return True
    return cleaned.lower() in _LEXICAL_STOPWORDS


def resolve_orphans(
    db,
    G: nx.MultiDiGraph,
    embedding_fn: Optional[Callable] = None,
    embed_batch_fn: Optional[Callable] = None,
    fts_limit: int = 3,
    vec_k: int = 3,
    vec_distance_threshold: float = 0.15,
    max_lexical_edges: int = 2,
    embed_batch_size: int = 64,
    vec_prefix_depth: Optional[int] = None,
    embed_max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Mode-dependent orphan resolution cascade (batched for performance).

    Mirrors the wikis ``graph_topology.resolve_orphans`` exactly. The
    enabled feature flags determine which passes run and in what order:

    * Mode A (``orphan_cascade_v2`` on, default) — 4-pass cascade:
      explicit-ref → hybrid RRF → tiered lexical → directory
      proximity.
    * Mode B (cascade off, ``orphan_lexical_tiered`` on) — 3-pass
      cascade: tiered lexical T1–T4 → batched semantic vector →
      directory proximity.
    * Mode C (both off) — 3-pass cascade: legacy flat FTS (stopword
      gate + ``fts_min_score_norm``) → batched semantic vector →
      directory proximity.

    Edge writes are graph-only (skip_db=True); persist_weights_to_db
    rewrites all edges at the end of Phase 2.
    """
    import time as _time

    if vec_prefix_depth is None:
        try:
            vec_prefix_depth = max(0, int(os.getenv("DEEPWIKI_VEC_PREFIX_DEPTH", "2")))
        except ValueError:
            vec_prefix_depth = 2
    if embed_max_workers is None:
        try:
            embed_max_workers = max(1, int(os.getenv("DEEPWIKI_VEC_CONCURRENCY", "1")))
        except ValueError:
            embed_max_workers = 1

    orphans = find_orphans(G)

    stats: Dict[str, Any] = {
        "orphans_found": len(orphans),
        "orphan_count": len(orphans),  # legacy alias
        "lexical": 0,
        "semantic": 0,
        "directory": 0,
        "explicit_ref": 0,
        "hybrid": 0,
        "resolved": 0,
        "orphans_remaining": 0,
    }

    if not orphans:
        logger.info("Orphan resolution: no orphans found")
        _sync_legacy_aliases(stats)
        return stats

    t0 = _time.monotonic()
    logger.info("Orphan resolution: processing %d orphan nodes", len(orphans))

    resolved: Set[str] = set()
    flags = get_feature_flags() if get_feature_flags is not None else None

    # ── Mode A: v2 4-pass cascade (default) ──────────────────
    if flags is not None and flags.orphan_cascade_v2:
        out = _resolve_orphans_v2(
            db, G, orphans, resolved, stats, flags, t0,
            embedding_fn=embedding_fn,
            embed_batch_fn=embed_batch_fn,
            fts_limit=fts_limit,
            vec_k=vec_k,
            vec_distance_threshold=vec_distance_threshold,
            max_lexical_edges=max_lexical_edges,
            embed_batch_size=embed_batch_size,
            vec_prefix_depth=vec_prefix_depth,
            embed_max_workers=embed_max_workers,
        )
        _sync_legacy_aliases(out)
        return out

    # ── Pass 1: lexical (tiered T1–T4 OR legacy flat FTS) ────
    if flags is not None and flags.orphan_lexical_tiered:
        try:
            from .graph_lexical_v2 import resolve_orphans_lexical_tiered as _tiered
        except Exception:  # noqa: BLE001
            try:
                from graph_lexical_v2 import resolve_orphans_lexical_tiered as _tiered  # type: ignore
            except Exception as exc:  # noqa: BLE001
                logger.debug("graph_lexical_v2 unavailable: %s", exc)
                _tiered = None  # type: ignore[assignment]
        if _tiered is not None:
            for node_id in orphans:
                try:
                    hits = _tiered(
                        db, G, node_id,
                        fts_limit=max(fts_limit, 8),
                        max_lexical_edges=max_lexical_edges,
                        flags=flags,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("tiered lexical resolve failed for %s: %s", node_id, exc)
                    continue
                if not hits:
                    continue
                for hit in hits:
                    tgt = hit.get("node_id")
                    if not tgt or tgt == node_id or not G.has_node(tgt):
                        continue
                    _add_edge(
                        db, G,
                        source=node_id,
                        target=tgt,
                        rel_type="lexical_link",
                        edge_class="lexical",
                        created_by="fts5_lexical_v2",
                        skip_db=True,
                        provenance={
                            "source": "fts_lexical_tiered",
                            "tier": hit.get("_tier"),
                            "score_norm": hit.get("score_norm"),
                        },
                    )
                    stats["lexical"] += 1
                resolved.add(node_id)
    else:
        min_norm = flags.fts_min_score_norm if flags is not None else 0.0
        for node_id in orphans:
            node = db.get_node(node_id) if db is not None else None

            symbol_name = ""
            if node:
                symbol_name = node.get("symbol_name", "") or ""
            if not symbol_name:
                node_data = G.nodes.get(node_id, {}) or {}
                sym = node_data.get("symbol")
                if isinstance(sym, dict):
                    symbol_name = sym.get("name", "") or ""
                elif sym is not None:
                    symbol_name = getattr(sym, "name", "") or ""
                if not symbol_name:
                    symbol_name = node_data.get("symbol_name", "") or ""
            if not symbol_name or len(symbol_name) < 2:
                continue

            if _is_low_value_lexical_query(symbol_name):
                continue

            try:
                fts_hits = db.search_fts5(query=symbol_name, limit=fts_limit) or []
            except Exception as exc:  # noqa: BLE001
                logger.debug("legacy FTS lookup for %s failed: %s", node_id, exc)
                continue
            fts_hits = [h for h in fts_hits if h.get("node_id") != node_id]
            if min_norm > 0:
                fts_hits = [
                    h for h in fts_hits
                    if h.get("score_norm") is None
                    or float(h.get("score_norm", 0.0)) >= min_norm
                ]
            if not fts_hits:
                continue
            for hit in fts_hits[:max_lexical_edges]:
                tgt = hit.get("node_id")
                if not tgt or not G.has_node(tgt):
                    continue
                _add_edge(
                    db, G,
                    source=node_id,
                    target=tgt,
                    rel_type="lexical_link",
                    edge_class="lexical",
                    created_by="fts5_lexical",
                    skip_db=True,
                    provenance={
                        "source": "fts_lexical",
                        "query": symbol_name,
                        "score_norm": hit.get("score_norm"),
                    },
                )
                stats["lexical"] += 1
            resolved.add(node_id)

    t1 = _time.monotonic()
    logger.info(
        "[ORPHAN] Pass 1 (FTS lexical): %d edges, %d orphans resolved in %.1fs",
        stats["lexical"], len(resolved), t1 - t0,
    )

    # ── Pass 2: Semantic Vector (batched embeddings) ─────────
    _effective_embed = embed_batch_fn or embedding_fn
    if _effective_embed is not None and getattr(db, "vec_available", False):
        # Collect candidates (skip already-resolved orphans)
        vec_candidates: List[Tuple[str, str, str]] = []  # (node_id, text, rel_path)
        for node_id in orphans:
            if node_id in resolved:
                continue

            node = db.get_node(node_id) if db is not None else None
            text = ""
            rel_path = ""
            if node:
                text = node.get("source_text", "") or node.get("docstring", "") or ""
                rel_path = node.get("rel_path", "") or ""
            if not text:
                node_data = G.nodes.get(node_id, {}) or {}
                text = node_data.get("source_text", "") or node_data.get("docstring", "") or ""
                if not rel_path:
                    rel_path = (
                        (node_data.get("location") or {}).get("rel_path", "")
                        or node_data.get("rel_path", "")
                    )
            if not text or len(text.strip()) < 10:
                continue
            vec_candidates.append((node_id, text, rel_path))

        t2 = _time.monotonic()

        batch_specs = [
            vec_candidates[i:i + embed_batch_size]
            for i in range(0, len(vec_candidates), embed_batch_size)
        ]
        n_batches = len(batch_specs)
        n_candidates = len(vec_candidates)

        logger.info(
            "[ORPHAN] Pass 2 starting: %d candidates, %d batches "
            "(batch_size=%d, workers=%d, vec_prefix_depth=%d)",
            n_candidates, n_batches, embed_batch_size, embed_max_workers,
            vec_prefix_depth,
        )

        def _embed_batch(batch):
            texts = [t for _, t, _ in batch]
            if embed_batch_fn:
                return batch, embed_batch_fn(texts)
            return batch, [embedding_fn(t) for t in texts]

        _last_log = _time.monotonic()
        _log_interval = 10.0

        def _process_result(batch, embeddings, idx: int) -> None:
            nonlocal _last_log
            for (nid, _text, rp), emb in zip(batch, embeddings):
                if nid in resolved:
                    continue

                prefixes = _expanding_prefixes(rp)
                if vec_prefix_depth > 0:
                    capped: List[str] = prefixes[: max(1, vec_prefix_depth - 1)]
                    if "" not in capped:
                        capped.append("")
                    prefixes = capped

                for prefix in prefixes:
                    vec_hits = db.search_vec(
                        embedding=emb,
                        k=vec_k,
                        path_prefix=prefix if prefix else None,
                    )
                    vec_hits = [
                        h for h in vec_hits
                        if h.get("node_id") != nid
                        and h.get("vec_distance", 1.0) < vec_distance_threshold
                    ]

                    if vec_hits:
                        for hit in vec_hits:
                            tgt = hit.get("node_id")
                            if not tgt or not G.has_node(tgt):
                                continue
                            _add_edge(
                                db, G,
                                source=nid, target=tgt,
                                rel_type="semantic_link",
                                edge_class="semantic",
                                created_by="vec_semantic",
                                raw_similarity=1.0 - hit.get("vec_distance", 0),
                                skip_db=True,
                            )
                            stats["semantic"] += 1
                        resolved.add(nid)
                        break

            now = _time.monotonic()
            if now - _last_log >= _log_interval or idx == n_batches:
                logger.info(
                    "[ORPHAN] Pass 2 progress: batch %d/%d, %d edges, "
                    "%d/%d resolved, elapsed %.1fs",
                    idx, n_batches, stats["semantic"],
                    len(resolved), n_candidates, now - t2,
                )
                _last_log = now

        if embed_max_workers > 1 and n_batches > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=embed_max_workers) as pool:
                futures = [pool.submit(_embed_batch, b) for b in batch_specs]
                for i, fut in enumerate(as_completed(futures), start=1):
                    batch, embeddings = fut.result()
                    _process_result(batch, embeddings, i)
        else:
            for i, spec in enumerate(batch_specs, start=1):
                batch, embeddings = _embed_batch(spec)
                _process_result(batch, embeddings, i)

        t3 = _time.monotonic()
        logger.info(
            "[ORPHAN] Pass 2 (semantic vector): %d edges from %d candidates in %.1fs",
            stats["semantic"], n_candidates, t3 - t2,
        )

    # ── Pass 3: Directory proximity fallback ─────────────────
    t4 = _time.monotonic()
    remaining = find_orphans(G)
    if remaining:
        dir_edges = _resolve_orphans_by_directory(db, G, remaining)
        stats["directory"] = dir_edges

    t5 = _time.monotonic()
    logger.info(
        "[ORPHAN] Pass 3 (directory): %d resolved in %.1fs",
        stats["directory"], t5 - t4,
    )

    stats["orphans_remaining"] = len(find_orphans(G))
    stats["resolved"] = stats["orphans_found"] - stats["orphans_remaining"]

    logger.info(
        "[ORPHAN] Total: %d/%d orphans resolved in %.1fs "
        "(lexical=%d, semantic=%d, directory=%d), %d remaining",
        stats["resolved"], stats["orphans_found"], t5 - t0,
        stats["lexical"], stats["semantic"],
        stats["directory"],
        stats["orphans_remaining"],
    )

    _sync_legacy_aliases(stats)
    return stats


def _sync_legacy_aliases(stats: Dict[str, Any]) -> None:
    """Mirror the new stat keys onto legacy ``*_edges_added`` aliases.

    Existing callers (``filesystem_indexer``, legacy unit tests) still
    read ``orphan_count`` / ``lexical_edges_added`` etc.; the wikis
    code-base dropped those keys but deepwiki keeps them as aliases for
    backwards compatibility.
    """
    stats["orphan_count"] = stats.get("orphans_found", 0)
    stats["lexical_edges_added"] = stats.get("lexical", 0)
    stats["semantic_edges_added"] = stats.get("semantic", 0)
    stats["directory_edges_added"] = stats.get("directory", 0)
    stats["explicit_ref_edges_added"] = stats.get("explicit_ref", 0)
    stats["hybrid_edges_added"] = stats.get("hybrid", 0)
    stats["orphans_resolved"] = stats.get("resolved", 0)


def _add_edge(
    db, G: nx.MultiDiGraph,
    source: str, target: str,
    rel_type: str, edge_class: str, created_by: str,
    raw_similarity: Optional[float] = None,
    *,
    skip_db: bool = False,
    provenance: Optional[Dict[str, Any]] = None,
) -> None:
    """Add an edge to both the DB and the in-memory graph.

    When ``skip_db`` is True the edge is added only to the in-memory
    graph; the v2 orphan cascade uses this path because
    :func:`persist_weights_to_db` rewrites every edge at the end of
    Phase 2 with the final weight.
    """
    if not skip_db and db is not None:
        kwargs: Dict[str, Any] = {
            "edge_class": edge_class,
            "created_by": created_by,
        }
        if raw_similarity is not None:
            kwargs["raw_similarity"] = raw_similarity
        try:
            db.upsert_edge(source, target, rel_type, **kwargs)
        except Exception as exc:  # noqa: BLE001 — non-fatal
            logger.debug("upsert_edge(%s -> %s) failed: %s", source, target, exc)

    # In-memory graph insert
    attrs: Dict[str, Any] = {
        "relationship_type": rel_type,
        "edge_class": edge_class,
        "created_by": created_by,
    }
    if raw_similarity is not None:
        attrs["raw_similarity"] = raw_similarity
    if provenance is not None:
        attrs["provenance"] = provenance
    G.add_edge(source, target, **attrs)


# ═══════════════════════════════════════════════════════════════════════════
# 3b. Doc Edge Injection — Hyperlink (Tier 1) + Proximity (Tier 2)
# ═══════════════════════════════════════════════════════════════════════════

# Regex for Markdown links: [text](target_path)
_MD_LINK_RE = re.compile(r'\[(?:[^\]]*)\]\(([^)]+)\)')

# Regex for inline code references: `SymbolName`
_BACKTICK_REF_RE = re.compile(r'`([A-Za-z_]\w+(?:\.\w+)*)`')

# Doc-related symbol kinds (mirror constants.py DOC_SYMBOL_TYPES + legacy types)
_DOC_KINDS = frozenset({
    "module_doc", "file_doc", "doc", "readme", "documentation",
    "markdown", "rst", "text",
})


def _is_doc_node(G: nx.MultiDiGraph, node_id: str) -> bool:
    """Check if a node is a documentation node."""
    data = G.nodes.get(node_id, {})
    if data.get("is_doc") == 1:
        return True
    sym_type = data.get("symbol_type", "") or data.get("kind", "")
    if sym_type in _DOC_KINDS:
        return True
    if sym_type.endswith("_document") or sym_type.endswith("_section"):
        return True
    return False


def _normalize_path(ref_path: str, source_dir: str) -> str:
    """Resolve a relative path reference against the source file's directory.

    Normalises ``../src/foo.py`` relative to ``docs/guide/`` into
    ``src/foo.py``.
    """
    # Skip URLs, anchors, and email links
    if ref_path.startswith(("http://", "https://", "mailto:", "#")):
        return ""
    # Strip optional leading ./ (but not ../)
    if ref_path.startswith("./"):
        ref_path = ref_path[2:]
    joined = os.path.normpath(os.path.join(source_dir, ref_path))
    # normpath on pure relative gives us the cleaned relative path
    return joined.replace("\\", "/")


def _build_path_index(G: nx.MultiDiGraph) -> Dict[str, str]:
    """Build a mapping from rel_path → node_id for all graph nodes."""
    index: Dict[str, str] = {}
    for nid, data in G.nodes(data=True):
        rp = data.get("rel_path", "")
        if rp:
            index[rp] = nid
    return index


def _build_name_index(G: nx.MultiDiGraph) -> Dict[str, List[str]]:
    """Build mapping from symbol_name → [node_id, ...] for code nodes."""
    index: Dict[str, List[str]] = {}
    for nid, data in G.nodes(data=True):
        if _is_doc_node(G, nid):
            continue
        name = data.get("symbol_name", "")
        if name:
            index.setdefault(name, []).append(nid)
    return index


def _extract_hyperlink_edges(
    G: nx.MultiDiGraph,
    path_index: Dict[str, str],
    name_index: Dict[str, List[str]],
) -> List[Tuple[str, str]]:
    """Tier 1: Extract edges from Markdown links and backtick references.

    For each doc node, parse its ``source_text`` for:
    - ``[text](relative/path)`` → directed edge doc → target file
    - `` `SymbolName` `` → directed edge doc → code symbol

    Returns list of (source_id, target_id) pairs.
    """
    edges: List[Tuple[str, str]] = []

    for nid, data in G.nodes(data=True):
        if not _is_doc_node(G, nid):
            continue
        source_text = data.get("source_text", "") or ""
        if not source_text:
            continue

        rel_path = data.get("rel_path", "")
        source_dir = os.path.dirname(rel_path) if rel_path else ""

        # Parse markdown links
        for match in _MD_LINK_RE.finditer(source_text):
            ref = match.group(1).split("#")[0].strip()  # strip anchors
            if not ref:
                continue
            resolved = _normalize_path(ref, source_dir)
            if not resolved:
                continue
            target_nid = path_index.get(resolved)
            if target_nid and target_nid != nid:
                edges.append((nid, target_nid))

        # Parse backtick code references
        for match in _BACKTICK_REF_RE.finditer(source_text):
            symbol = match.group(1)
            # Try exact match first, then last segment (e.g. module.Class → Class)
            targets = name_index.get(symbol, [])
            if not targets and "." in symbol:
                targets = name_index.get(symbol.rsplit(".", 1)[-1], [])
            for tid in targets[:2]:  # Cap at 2 matches per reference
                if tid != nid:
                    edges.append((nid, tid))

    return edges


def _extract_proximity_edges(
    G: nx.MultiDiGraph,
    path_index: Dict[str, str],
) -> List[Tuple[str, str]]:
    """Tier 2: Extract edges between files in related directories.

    Heuristic: ``docs/api/README.md`` relates to ``src/api/*.py``.
    We strip common doc prefixes (docs/, doc/) and match the remaining
    directory structure against code node paths.

    Returns list of (source_id, target_id) pairs.
    """
    edges: List[Tuple[str, str]] = []
    _DOC_PREFIXES = ("docs/", "doc/", "documentation/")

    # Build directory → [node_id] mapping for code nodes
    dir_to_code: Dict[str, List[str]] = {}
    for nid, data in G.nodes(data=True):
        if _is_doc_node(G, nid):
            continue
        rp = data.get("rel_path", "")
        if rp:
            d = os.path.dirname(rp).replace("\\", "/")
            if d:
                dir_to_code.setdefault(d, []).append(nid)

    for nid, data in G.nodes(data=True):
        if not _is_doc_node(G, nid):
            continue
        rp = data.get("rel_path", "")
        if not rp:
            continue

        doc_dir = os.path.dirname(rp).replace("\\", "/")
        if not doc_dir:
            continue

        # Strip doc prefix to get the "topic" directory
        topic_dir = doc_dir
        for prefix in _DOC_PREFIXES:
            if topic_dir.startswith(prefix):
                topic_dir = topic_dir[len(prefix):]
                break

        if not topic_dir:
            continue

        # Look for code in matching directories:
        # src/api, lib/api, api, etc.
        for code_dir, code_nodes in dir_to_code.items():
            if code_dir == topic_dir or code_dir.endswith("/" + topic_dir):
                for cnid in code_nodes[:5]:  # Cap to avoid hairballs
                    if cnid != nid:
                        edges.append((nid, cnid))

    return edges


def _enrich_graph_nodes_from_db(db, G: nx.MultiDiGraph) -> None:
    """Populate missing graph node attributes from the DB.

    When graph nodes lack ``rel_path``, ``source_text``, or ``symbol_type``,
    look them up from the unified DB. This ensures doc edge extraction works
    even when graph nodes were created with minimal attributes.
    """
    _NEEDED = {"rel_path", "source_text", "symbol_type"}
    to_enrich = [
        nid for nid in G.nodes()
        if not (_NEEDED <= set(G.nodes[nid].keys()))
    ]
    if not to_enrich:
        return

    for nid in to_enrich:
        row = db.get_node(nid)
        if not row:
            continue
        node_data = G.nodes[nid]
        for key in ("rel_path", "source_text", "symbol_type", "symbol_name",
                     "is_doc", "language"):
            if key not in node_data and key in row:
                node_data[key] = row[key]


def inject_doc_edges(
    db,
    G: nx.MultiDiGraph,
) -> Dict[str, Any]:
    """Inject Tier 1 (hyperlink) and Tier 2 (proximity) doc edges.

    Must be called AFTER orphan resolution but BEFORE edge weighting,
    so that these edges get proper inverse in-degree weights.

    Returns:
        Stats dict with counts by tier.
    """
    stats = {
        "hyperlink_edges_added": 0,
        "proximity_edges_added": 0,
        "total_edges_added": 0,
    }

    # Ensure graph nodes have attributes needed for doc edge extraction
    _enrich_graph_nodes_from_db(db, G)

    path_index = _build_path_index(G)
    name_index = _build_name_index(G)

    # Tier 1: Hyperlink edges
    seen: Set[Tuple[str, str]] = set()
    for src, tgt in _extract_hyperlink_edges(G, path_index, name_index):
        if (src, tgt) not in seen and not G.has_edge(src, tgt):
            _add_edge(
                db, G,
                source=src, target=tgt,
                rel_type="hyperlink",
                edge_class="doc",
                created_by="md_hyperlink",
            )
            seen.add((src, tgt))
            stats["hyperlink_edges_added"] += 1

    # Tier 2: Proximity edges
    for src, tgt in _extract_proximity_edges(G, path_index):
        if (src, tgt) not in seen and not G.has_edge(src, tgt):
            _add_edge(
                db, G,
                source=src, target=tgt,
                rel_type="proximity",
                edge_class="doc",
                created_by="dir_proximity",
            )
            seen.add((src, tgt))
            stats["proximity_edges_added"] += 1

    stats["total_edges_added"] = (
        stats["hyperlink_edges_added"] + stats["proximity_edges_added"]
    )

    logger.info(
        "Doc edge injection: %d hyperlink + %d proximity = %d total edges",
        stats["hyperlink_edges_added"],
        stats["proximity_edges_added"],
        stats["total_edges_added"],
    )

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# 4. Persist Weights Back to DB
# ═══════════════════════════════════════════════════════════════════════════

def persist_weights_to_db(db, G: nx.MultiDiGraph) -> int:
    """Write computed edge weights from NetworkX back to the unified DB.

    This re-reads *all* edges from the DB, matches them to the in-memory
    graph by (source, target, rel_type) ordering, and batch-updates
    weights.

    For large graphs we do a full overwrite of edge weights rather than
    trying to match by edge PK (the DB edges have autoincrement IDs that
    don't correspond to NetworkX edge keys).

    Returns:
        Number of edges updated.
    """
    if G.number_of_edges() == 0:
        return 0

    # Strategy: delete all edges, re-insert with weights from the graph.
    # This is simple and correct for the dual-write scenario (the DB is
    # fully reconstructed from the graph each run anyway in Phase 1-2).
    db.conn.execute("DELETE FROM repo_edges")

    edge_batch = []
    BATCH = 5000

    for u, v, key, data in G.edges(data=True, keys=True):
        annotations = data.get("annotations", {})
        if isinstance(annotations, dict):
            import json
            annotations = json.dumps(annotations)

        edge_batch.append({
            "source_id": str(u),
            "target_id": str(v),
            "rel_type": data.get("relationship_type", ""),
            "edge_class": data.get("edge_class", "structural"),
            "analysis_level": data.get("analysis_level", "comprehensive"),
            "weight": data.get("weight", 1.0),
            "raw_similarity": data.get("raw_similarity"),
            "source_file": data.get("source_file", ""),
            "target_file": data.get("target_file", ""),
            "language": data.get("language", ""),
            "annotations": annotations,
            "created_by": data.get("created_by", "ast"),
        })

        if len(edge_batch) >= BATCH:
            db.upsert_edges_batch(edge_batch)
            edge_batch.clear()

    if edge_batch:
        db.upsert_edges_batch(edge_batch)

    db.conn.commit()

    count = db.edge_count()
    logger.info("Persisted %d weighted edges to unified DB", count)
    return count


# ═══════════════════════════════════════════════════════════════════════════
# 4b. Component Bridging — connect disconnected components
# ═══════════════════════════════════════════════════════════════════════════

def bridge_disconnected_components(
    db,
    G: nx.MultiDiGraph,
) -> Dict[str, Any]:
    """Connect disconnected components via directory-proximity bridge edges.

    After orphan resolution and doc edge injection, the graph may still
    contain many small disconnected components — clusters of 2+ nodes
    with internal edges but no edges to the rest of the graph.  Leiden
    produces at minimum one community per connected component regardless
    of the resolution parameter, so bridging is essential.

    Algorithm:
    1. Find all weakly connected components.
    2. Sort largest-first.
    3. For each non-largest component, find the most directory-similar
       larger (earlier) component using min-intersection of directory
       histograms.
    4. Add a lightweight bridge edge between each pair's highest-degree
       representative nodes.

    Bridge edges get ``edge_class='bridge'`` so the weighting step
    treats them as synthetic (excluded from structural in-degree and
    floored at ``SYNTHETIC_WEIGHT_FLOOR``).

    Mutates *G* in place and persists edges to *db*.

    Returns:
        Stats dict with component counts and bridges added.
    """
    components = list(nx.weakly_connected_components(G))
    n_before = len(components)

    stats = {
        "components_before": n_before,
        "components_after": n_before,
        "bridges_added": 0,
    }

    if n_before <= 1:
        logger.info("Component bridging: graph is fully connected (1 component)")
        return stats

    # Sort largest-first so index 0 is the biggest component
    components.sort(key=len, reverse=True)

    # Build directory histogram for each component
    def _dir_hist(comp):
        c: Counter = Counter()
        for nid in comp:
            data = G.nodes.get(nid, {})
            rp = data.get("rel_path") or data.get("file_name") or ""
            d = rp.rsplit("/", 1)[0] if "/" in rp else "<root>"
            c[d] += 1
        return c

    comp_dirs = [_dir_hist(comp) for comp in components]

    # Representative node per component (highest total degree)
    comp_reps = [
        max(comp, key=lambda n: G.in_degree(n) + G.out_degree(n))
        for comp in components
    ]

    def _dir_sim(a: Counter, b: Counter) -> int:
        return sum(min(a[d], b[d]) for d in a if d in b)

    bridges_added = 0
    for i in range(1, n_before):
        # Find the most directory-similar earlier (larger) component
        best_target = 0
        best_sim = _dir_sim(comp_dirs[i], comp_dirs[0])

        for j in range(1, i):
            sim = _dir_sim(comp_dirs[i], comp_dirs[j])
            if sim > best_sim:
                best_sim = sim
                best_target = j

        src = comp_reps[i]
        dst = comp_reps[best_target]

        # Bidirectional bridge edge
        _add_edge(
            db, G, source=src, target=dst,
            rel_type="component_bridge",
            edge_class="bridge",
            created_by="component_bridging",
        )
        _add_edge(
            db, G, source=dst, target=src,
            rel_type="component_bridge",
            edge_class="bridge",
            created_by="component_bridging",
        )
        bridges_added += 2

    n_after = nx.number_weakly_connected_components(G)
    stats["components_after"] = n_after
    stats["bridges_added"] = bridges_added

    logger.info(
        "Component bridging: %d → %d components (%d bridge edges added)",
        n_before, n_after, bridges_added,
    )

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# 5. Orchestrator — run full Phase 2 pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_phase2(
    db,
    G: nx.MultiDiGraph,
    embedding_fn: Optional[Callable] = None,
    z_threshold: float = 3.0,
    vec_distance_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Execute the complete Phase 2 pipeline on a graph + unified DB.

    Steps (corrected order — 7F):
        1. Resolve orphans (FTS5 lexical + vector semantic)
        2. Inject doc edges (hyperlink + proximity)
        3. Bridge disconnected components (directory proximity)
        4. Apply inverse in-degree edge weights on ALL edges
        5. Detect and flag hubs on complete degree distribution
        6. Persist final weights back to DB

    Args:
        db: ``UnifiedWikiDB`` instance (already populated by Phase 1).
        G: In-memory ``nx.MultiDiGraph`` (same graph as in DB).
        embedding_fn: Optional text → embedding callable.
        z_threshold: Hub detection Z-score threshold.
        vec_distance_threshold: Max distance for semantic links.

    Returns:
        Combined stats dict from all sub-steps.
    """
    results: Dict[str, Any] = {}

    # Step 1: Orphan resolution — inject semantic/lexical edges FIRST
    # so that weighting and hub detection see the COMPLETE edge set.
    results["orphan_resolution"] = resolve_orphans(
        db, G,
        embedding_fn=embedding_fn,
        vec_distance_threshold=vec_distance_threshold,
    )

    # Step 2: Doc edge injection (hyperlink + proximity)
    results["doc_edges"] = inject_doc_edges(db, G)

    # Note: cross-language linking + API-surface extraction + test-linker
    # run in Phase 1c (``FilesystemRepositoryIndexer._write_unified_db``)
    # *before* the graph is persisted, so by the time Phase 2 starts the
    # graph already carries cross-language edges with proper weights.

    # Step 3: Bridge disconnected components — BEFORE weighting so
    # bridge edges receive proper weights in step 4.
    results["component_bridging"] = bridge_disconnected_components(db, G)

    # Step 4: Edge weighting — on ALL edges (structural + semantic + lexical + doc + bridge)
    results["weighting"] = apply_edge_weights(G)

    # Step 5: Hub detection — on complete degree distribution
    hubs = detect_hubs(G, z_threshold=z_threshold)
    flag_hubs_in_db(db, hubs)
    results["hubs"] = {
        "count": len(hubs),
        "node_ids": sorted(hubs)[:20],  # Cap at 20 for readability
    }

    # Step 6: Persist weights
    results["edges_persisted"] = persist_weights_to_db(db, G)

    # Store phase 2 metadata
    db.set_meta("phase2_completed", True)
    db.set_meta("phase2_stats", results)

    logger.info(
        "Phase 2 complete: %d orphans resolved, %d doc edges injected, "
        "%d→%d components bridged, %d edges weighted, %d hubs, %d edges persisted",
        results["orphan_resolution"]["orphans_resolved"],
        results["doc_edges"].get("total_edges_added", 0),
        results["component_bridging"]["components_before"],
        results["component_bridging"]["components_after"],
        results["weighting"]["edges_weighted"],
        results["hubs"]["count"],
        results["edges_persisted"],
    )

    return results
