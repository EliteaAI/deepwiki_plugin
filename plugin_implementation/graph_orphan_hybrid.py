"""Hybrid orphan-resolution Pass 2 (FTS+Vec fused via Reciprocal Rank Fusion).

Back-port from ``wikis/backend/app/core/graph_orphan_hybrid.py``. Pure
helpers — the v2 cascade in :mod:`graph_topology` is responsible for
applying the returned hits as edges.

Public surface
--------------
* :func:`rrf_fuse` — pure helper; combines any number of ranked result
  lists into a single ranked list using ``score = Σ 1 / (k + rank_i)``.
* :func:`resolve_orphans_hybrid` — orphan-level Pass 2 used by the v2
  cascade. Returns ``[hit, ...]`` ranked by ``rrf_score`` desc.

Embeddings are preferred from the persisted ``repo_vec`` table via
``db.get_embedding_by_id``. If no persisted embedding is available,
the cascade may fall back to synchronous embedding when ``embed_fn``
is provided.
"""

from __future__ import annotations

import logging
import math
import struct
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx

try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3  # type: ignore[no-redef]

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

try:
    from .feature_flags import FeatureFlags, get_feature_flags
    from .unified_db import (
        UnifiedWikiDB,
        _attach_score_norm,
        _path_prefix_bounds,
        _sanitize_fts5_query,
        _serialize_float32_vec,
    )
except ImportError:  # pragma: no cover — non-package import fallback
    from feature_flags import FeatureFlags, get_feature_flags  # type: ignore
    from unified_db import (  # type: ignore
        UnifiedWikiDB,
        _attach_score_norm,
        _path_prefix_bounds,
        _sanitize_fts5_query,
        _serialize_float32_vec,
    )

logger = logging.getLogger(__name__)

#: Maximum directory node count for building full in-memory FTS5 + vector context.
MAX_LOCAL_DIR_NODES = 2000
#: Maximum candidate pool size for bounded vector KNN in giant dirs / repo root.
MAX_BOUNDED_VEC_POOL = 1000
#: Global FTS match cap above which giant-dir queries skip expensive posting scans.
MAX_GIANT_DIR_GLOBAL_FTS_HITS = 5000
#: Symbol types skipped in hybrid Pass 2 (line-split text fragments handled in Pass 4).
_SKIP_HYBRID_SYMBOL_TYPES = frozenset({"text_chunk"})


# ──────────────────────────────────────────────────────────────────────────
# RRF fusion
# ──────────────────────────────────────────────────────────────────────────


def rrf_fuse(
    *ranked_lists: Sequence[Dict[str, Any]],
    k: int = 60,
    id_key: str = "node_id",
) -> List[Dict[str, Any]]:
    """Fuse N ranked result lists via Reciprocal Rank Fusion.

    For each candidate ``id``, the fused score is::

        rrf_score(id) = Σ_i 1 / (k + rank_i(id))

    where ``rank_i`` is the 1-based position of ``id`` in the *i*-th
    input list (skipping lists where the candidate is absent).

    The returned dicts merge the hit payloads from every input list
    where the candidate appears (later lists win on key conflict),
    plus an attached ``rrf_score`` field. Order is descending by
    ``rrf_score``.

    Args:
        *ranked_lists: One or more lists of result dicts. Each dict
            must carry the ``id_key`` field. Order matters — index
            inside the list is the rank.
        k: RRF constant; larger ``k`` flattens the contribution of
            the top-ranked items. Default 60 (TREC standard).
        id_key: Field used to identify candidates. Default
            ``"node_id"`` (matches our storage layer).

    Returns:
        Single fused list, ordered by ``rrf_score`` desc. Each dict
        has ``rrf_score`` plus the merged payload.
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    scores: Dict[Any, float] = {}
    payloads: Dict[Any, Dict[str, Any]] = {}
    for lst in ranked_lists:
        if not lst:
            continue
        for rank, hit in enumerate(lst, start=1):
            cid = hit.get(id_key)
            if cid is None:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            existing = payloads.get(cid)
            if existing is None:
                payloads[cid] = dict(hit)
            else:
                merged = dict(existing)
                for key, val in hit.items():
                    if key != "rrf_score":
                        merged[key] = val
                payloads[cid] = merged

    fused: List[Dict[str, Any]] = []
    for cid, score in scores.items():
        item = dict(payloads[cid])
        item["rrf_score"] = score
        fused.append(item)
    fused.sort(key=lambda h: (h["rrf_score"], h.get(id_key, "")), reverse=True)
    return fused


# ──────────────────────────────────────────────────────────────────────────
# Orphan helpers
# ──────────────────────────────────────────────────────────────────────────


def _orphan_symbol_name(
    G: nx.MultiDiGraph,
    db_node: Optional[Dict[str, Any]],
    node_id: str,
) -> str:
    if db_node:
        name = db_node.get("symbol_name", "") or ""
        if name:
            return name
    data = G.nodes.get(node_id, {}) or {}
    sym = data.get("symbol")
    if isinstance(sym, dict):
        return sym.get("name", "") or ""
    if sym is not None:
        return getattr(sym, "name", "") or ""
    return data.get("symbol_name", "") or ""


def _orphan_rel_path(
    G: nx.MultiDiGraph,
    db_node: Optional[Dict[str, Any]],
    node_id: str,
) -> str:
    if db_node:
        rp = db_node.get("rel_path", "") or ""
        if rp:
            return rp
    data = G.nodes.get(node_id, {}) or {}
    rp = data.get("rel_path", "") or ""
    if not rp:
        loc = data.get("location") or {}
        if isinstance(loc, dict):
            rp = loc.get("rel_path", "") or ""
    return rp


def _local_dir(rel_path: str) -> str:
    if not rel_path:
        return ""
    parts = rel_path.rsplit("/", 1)
    return parts[0] if len(parts) == 2 else ""


# ──────────────────────────────────────────────────────────────────────────
# Directory-scoped hybrid context (O(dir_size) FTS + Vector KNN)
# ──────────────────────────────────────────────────────────────────────────


class _LocalVecIndex:
    """In-memory L2 KNN index over float32 embedding blobs for a single scope."""

    def __init__(self, node_ids: Sequence[str], blobs: Dict[str, bytes]) -> None:
        self._ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        self._dim: int = 0
        self._mat: Any = None
        self._vecs: List[Tuple[float, ...]] = []

        valid_items: List[Tuple[str, bytes]] = []
        expected_bytes = 0
        for nid in node_ids:
            blob = blobs.get(nid)
            if not blob:
                continue
            if expected_bytes == 0:
                dim = len(blob) // 4
                if dim <= 0 or len(blob) != dim * 4:
                    continue
                self._dim = dim
                expected_bytes = dim * 4
            if len(blob) == expected_bytes:
                valid_items.append((nid, blob))

        if not valid_items:
            return

        self._ids = [nid for nid, _ in valid_items]
        self._id_to_idx = {nid: idx for idx, nid in enumerate(self._ids)}

        if _HAS_NUMPY:
            self._mat = np.stack(
                [np.frombuffer(blob, dtype=np.float32) for _, blob in valid_items]
            )
        else:
            fmt = f"{self._dim}f"
            self._vecs = [struct.unpack(fmt, blob) for _, blob in valid_items]

    def contains(self, node_id: str) -> bool:
        return node_id in self._id_to_idx

    def get_vector(self, node_id: str) -> Optional[Any]:
        idx = self._id_to_idx.get(node_id)
        if idx is None:
            return None
        if _HAS_NUMPY and self._mat is not None:
            return self._mat[idx]
        if self._vecs:
            return list(self._vecs[idx])
        return None

    def query(
        self,
        query_vec: Any,
        *,
        top_n: int,
        exclude_id: str,
        extra_blobs: Optional[Dict[str, bytes]] = None,
    ) -> List[Dict[str, Any]]:
        if top_n <= 0:
            return []

        dim = len(query_vec)
        if dim == 0 or (self._dim and dim != self._dim):
            return []
        expected_bytes = dim * 4
        scored: List[Tuple[float, str]] = []

        if _HAS_NUMPY:
            q = np.asarray(query_vec, dtype=np.float32)
            if self._mat is not None and len(self._ids) > 0:
                diff = self._mat - q
                dists = np.sqrt(np.sum(diff * diff, axis=1))
                n_rows = len(self._ids)
                if n_rows > top_n + 1:
                    k_part = min(n_rows, top_n + 1)
                    idxs = np.argpartition(dists, k_part - 1)[:k_part]
                    for idx in idxs:
                        nid = self._ids[int(idx)]
                        if nid != exclude_id:
                            scored.append((float(dists[int(idx)]), nid))
                else:
                    for idx in range(n_rows):
                        nid = self._ids[idx]
                        if nid != exclude_id:
                            scored.append((float(dists[idx]), nid))

            if extra_blobs:
                extra_items = [
                    (nid, blob)
                    for nid, blob in extra_blobs.items()
                    if nid != exclude_id
                    and nid not in self._id_to_idx
                    and blob
                    and len(blob) == expected_bytes
                ]
                if extra_items:
                    emat = np.stack(
                        [np.frombuffer(blob, dtype=np.float32) for _, blob in extra_items]
                    )
                    ediff = emat - q
                    edists = np.sqrt(np.sum(ediff * ediff, axis=1))
                    for i, (nid, _) in enumerate(extra_items):
                        scored.append((float(edists[i]), nid))
        else:
            q_tuple = struct.unpack(f"{dim}f", _serialize_float32_vec(list(query_vec)))
            for idx, nid in enumerate(self._ids):
                if nid == exclude_id:
                    continue
                vec = self._vecs[idx]
                dist = math.sqrt(sum((a - b) * (a - b) for a, b in zip(vec, q_tuple)))
                scored.append((dist, nid))
            if extra_blobs:
                fmt = f"{dim}f"
                for nid, blob in extra_blobs.items():
                    if nid == exclude_id or nid in self._id_to_idx or not blob or len(blob) != expected_bytes:
                        continue
                    vec = struct.unpack(fmt, blob)
                    dist = math.sqrt(sum((a - b) * (a - b) for a, b in zip(vec, q_tuple)))
                    scored.append((dist, nid))

        scored.sort(key=lambda item: (item[0], item[1]))
        return [
            {"node_id": nid, "vec_distance": dist}
            for dist, nid in scored[:top_n]
        ]


class _DirHybridContext:
    """Scoped FTS + Vector context for one ``local_dir`` during Pass 2."""

    def __init__(
        self,
        db: UnifiedWikiDB,
        G: nx.MultiDiGraph,
        local_dir: str,
        *,
        total_nodes: int = 0,
    ) -> None:
        self.db = db
        self.G = G
        self.local_dir = local_dir
        self.total_nodes = total_nodes
        self.is_singleton = False
        self.is_giant = False
        self.db_nodes: Optional[Dict[str, Dict[str, Any]]] = None

        self._fts_conn: Optional[sqlite3.Connection] = None
        self._fts_built = False
        self._fts_cache: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        self._fts_count_cache: Dict[str, int] = {}

        self._vec_index: Optional[_LocalVecIndex] = None
        self._vec_built = False

        if local_dir:
            bounds = _path_prefix_bounds(local_dir)
            if bounds is not None:
                where = "rel_path >= ? AND rel_path < ?"
                params: list = [bounds[0], bounds[1], MAX_LOCAL_DIR_NODES + 1]
            else:
                where = "rel_path GLOB ?"
                params = [
                    local_dir.rstrip("/") + "/*" if not local_dir.endswith("*") else local_dir,
                    MAX_LOCAL_DIR_NODES + 1,
                ]
            rows = db.conn.execute(
                "SELECT node_id, rel_path, symbol_name, symbol_type, "
                "signature, docstring, source_text "
                f"FROM repo_nodes WHERE {where} LIMIT ?",
                params,
            ).fetchall()
            if len(rows) <= 1:
                self.is_singleton = True
            elif len(rows) <= MAX_LOCAL_DIR_NODES:
                self.db_nodes = {r["node_id"]: dict(r) for r in rows}
            else:
                self.is_giant = True
        else:
            if total_nodes > MAX_LOCAL_DIR_NODES:
                self.is_giant = True

    def get_db_node(self, orphan_id: str) -> Optional[Dict[str, Any]]:
        if self.db_nodes is not None:
            return self.db_nodes.get(orphan_id)
        return self.db.get_node(orphan_id)

    def _ensure_local_fts(self) -> None:
        if self._fts_built or self.db_nodes is None:
            return
        self._fts_built = True
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE VIRTUAL TABLE local_fts USING fts5("
            "node_id UNINDEXED, symbol_name, signature, docstring, source_text, "
            "tokenize='porter unicode61')"
        )
        conn.executemany(
            "INSERT INTO local_fts (node_id, symbol_name, signature, docstring, source_text) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (
                    nid,
                    row.get("symbol_name") or "",
                    row.get("signature") or "",
                    row.get("docstring") or "",
                    row.get("source_text") or "",
                )
                for nid, row in self.db_nodes.items()
            ],
        )
        conn.commit()
        self._fts_conn = conn

    def search_fts(self, query: str, limit: int) -> List[Dict[str, Any]]:
        sanitized = _sanitize_fts5_query(query)
        if not sanitized:
            return []
        cache_key = (sanitized, limit)
        cached = self._fts_cache.get(cache_key)
        if cached is not None:
            return cached

        res: List[Dict[str, Any]] = []
        if self.db_nodes is not None:
            self._ensure_local_fts()
            if self._fts_conn is not None:
                sql = (
                    "SELECT node_id, rank AS fts_rank FROM local_fts "
                    "WHERE local_fts MATCH ? ORDER BY rank LIMIT ?"
                )
                try:
                    rows = self._fts_conn.execute(sql, (sanitized, limit)).fetchall()
                except sqlite3.OperationalError:
                    if sanitized.endswith("*"):
                        rows = []
                    else:
                        try:
                            rows = self._fts_conn.execute(sql, (sanitized + "*", limit)).fetchall()
                        except sqlite3.OperationalError:
                            rows = []
                for row in rows:
                    nid = row["node_id"]
                    base = self.db_nodes.get(nid)
                    if base is None:
                        continue
                    item = dict(base)
                    item["fts_rank"] = row["fts_rank"]
                    res.append(_attach_score_norm(item))
        elif self.is_giant and self.local_dir:
            cnt = self._fts_count_cache.get(sanitized)
            if cnt is None:
                cnt = self.db.count_fts_matches(query, exact_match=False)
                self._fts_count_cache[sanitized] = cnt
            if cnt <= MAX_GIANT_DIR_GLOBAL_FTS_HITS:
                res = self.db.search_fts_with_path(
                    query, path_prefix=self.local_dir, limit=limit,
                ) or []
        else:
            if self.local_dir:
                res = self.db.search_fts_with_path(
                    query, path_prefix=self.local_dir, limit=limit,
                ) or []
            else:
                res = self.db.search_fts5(query=query, limit=limit) or []

        self._fts_cache[cache_key] = res
        return res

    def _ensure_vec_index(self) -> None:
        if self._vec_built or not self.db.vec_available:
            return
        self._vec_built = True
        if self.db_nodes is not None:
            node_ids = list(self.db_nodes.keys())
            blobs = self.db.get_embedding_blobs_by_ids(node_ids)
            self._vec_index = _LocalVecIndex(node_ids, blobs)
            return

        # Giant directory or giant repo root: build a bounded anchor pool.
        sample_limit = MAX_BOUNDED_VEC_POOL * 4
        if self.local_dir:
            bounds = _path_prefix_bounds(self.local_dir)
            if bounds is not None:
                rows = self.db.conn.execute(
                    "SELECT node_id FROM repo_nodes WHERE rel_path >= ? AND rel_path < ? LIMIT ?",
                    (bounds[0], bounds[1], sample_limit),
                ).fetchall()
            else:
                pat = self.local_dir.rstrip("/") + "/*" if not self.local_dir.endswith("*") else self.local_dir
                rows = self.db.conn.execute(
                    "SELECT node_id FROM repo_nodes WHERE rel_path GLOB ? LIMIT ?",
                    (pat, sample_limit),
                ).fetchall()
        else:
            rows = self.db.conn.execute(
                "SELECT node_id FROM repo_nodes WHERE is_architectural = 1 LIMIT ?",
                (sample_limit,),
            ).fetchall()
            if not rows:
                rows = self.db.conn.execute(
                    "SELECT node_id FROM repo_nodes LIMIT ?",
                    (sample_limit,),
                ).fetchall()

        cand_ids = [r["node_id"] for r in rows]
        if len(cand_ids) > MAX_BOUNDED_VEC_POOL:
            cand_ids.sort(
                key=lambda nid: (
                    (self.G.in_degree(nid) + self.G.out_degree(nid))
                    if self.G.has_node(nid) else -1
                ),
                reverse=True,
            )
            cand_ids = cand_ids[:MAX_BOUNDED_VEC_POOL]
        blobs = self.db.get_embedding_blobs_by_ids(cand_ids)
        self._vec_index = _LocalVecIndex(cand_ids, blobs)

    def get_vector(self, orphan_id: str) -> Optional[Any]:
        if self.db_nodes is not None:
            self._ensure_vec_index()
            if self._vec_index is not None:
                return self._vec_index.get_vector(orphan_id)
            return None
        if self._vec_built and self._vec_index is not None:
            vec = self._vec_index.get_vector(orphan_id)
            if vec is not None:
                return vec
        return self.db.get_embedding_by_id(orphan_id)

    def search_vec(
        self,
        orphan_id: str,
        embedding: Any,
        *,
        top_n: int,
        fts_candidate_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.db.vec_available:
            return []
        if self.db_nodes is not None or self.is_giant:
            self._ensure_vec_index()
            if self._vec_index is None:
                return []
            extra_blobs: Optional[Dict[str, bytes]] = None
            if self.is_giant and fts_candidate_ids:
                extra_ids = [
                    cid for cid in fts_candidate_ids
                    if cid != orphan_id and not self._vec_index.contains(cid)
                ]
                if extra_ids:
                    extra_blobs = self.db.get_embedding_blobs_by_ids(extra_ids)
            return self._vec_index.query(
                embedding,
                top_n=top_n,
                exclude_id=orphan_id,
                extra_blobs=extra_blobs,
            )

        vec_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        return self.db.search_vec(vec_list, k=top_n, path_prefix=self.local_dir or None) or []

    def close(self) -> None:
        if self._fts_conn is not None:
            try:
                self._fts_conn.close()
            except Exception:
                pass
            self._fts_conn = None
        self.db_nodes = None
        self._vec_index = None
        self._fts_cache.clear()
        self._fts_count_cache.clear()


# ──────────────────────────────────────────────────────────────────────────
# Hybrid Pass 2
# ──────────────────────────────────────────────────────────────────────────


def resolve_orphans_hybrid(
    db,
    G: nx.MultiDiGraph,
    orphan_id: str,
    *,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    top_n: Optional[int] = None,
    orphan_embeddings: Optional[Dict[str, Optional[List[float]]]] = None,
    embed_fn: Optional[Any] = None,
    flags: Optional[FeatureFlags] = None,
    _dir_ctx: Optional[_DirHybridContext] = None,
) -> List[Dict[str, Any]]:
    """Run hybrid FTS+Vec Pass 2 for ``orphan_id``.

    Returns the top ``top_n`` hits, RRF-fused and filtered by
    ``rrf_score >= threshold``. Side-effect free — the caller writes
    edges and attaches provenance (``"source": "hybrid_rrf"``).

    Embedding source order:

    1. ``orphan_embeddings[orphan_id]`` (Phase-3 reuse cache).
    2. ``_dir_ctx.get_vector(orphan_id)`` / ``db.get_embedding_by_id(orphan_id)``.
    3. ``embed_fn(text)`` (last resort — only when caller passes one).

    When all sources miss (or when FTS returned 0 hits and a vec-only
    rank-1 match cannot reach ``threshold``), the hybrid pass aborts so
    the v2 cascade can fall through without scanning ``repo_vec``.
    """
    if _dir_ctx is not None and _dir_ctx.is_singleton:
        return []

    flags = flags or get_feature_flags()
    k = k if k is not None else flags.orphan_rrf_k
    threshold = threshold if threshold is not None else flags.orphan_rrf_threshold
    top_n = top_n if top_n is not None else flags.orphan_hybrid_top_n

    if _dir_ctx is not None:
        db_node = _dir_ctx.get_db_node(orphan_id)
    else:
        db_node = db.get_node(orphan_id) if db is not None else None

    stype = (
        (db_node.get("symbol_type") if db_node else None)
        or (G.nodes.get(orphan_id, {}) or {}).get("symbol_type")
        or ""
    ).lower()
    if stype in _SKIP_HYBRID_SYMBOL_TYPES:
        return []

    name = _orphan_symbol_name(G, db_node, orphan_id)
    if not name or len(name) < 2:
        return []

    rel_path = _orphan_rel_path(G, db_node, orphan_id)
    local_dir = _local_dir(rel_path)

    # ── FTS branch ────────────────────────────────────────────────────
    fts_hits: List[Dict[str, Any]] = []
    try:
        if _dir_ctx is not None:
            fts_hits = _dir_ctx.search_fts(name, top_n)
        elif local_dir and hasattr(db, "search_fts_with_path"):
            fts_hits = db.search_fts_with_path(
                name, path_prefix=local_dir, limit=top_n,
            ) or []
        else:
            fts_hits = db.search_fts5(query=name, limit=top_n) or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("hybrid FTS for %s failed: %s", orphan_id, exc)
        fts_hits = []

    # Drop self-hits + sub-threshold BM25.
    if fts_hits:
        min_norm = flags.fts_min_score_norm
        cleaned: List[Dict[str, Any]] = []
        for h in fts_hits:
            if h.get("node_id") == orphan_id:
                continue
            if min_norm > 0:
                sn = h.get("score_norm")
                if sn is not None and float(sn) < min_norm:
                    continue
            cleaned.append(h)
        fts_hits = cleaned

    # Mathematically exact short-circuit: when FTS yields zero hits, the
    # highest possible RRF score from vec_hits alone (rank 1) is 1/(k+1).
    # If that cannot reach threshold, skip embedding lookup and vec scan.
    if not fts_hits and (1.0 / (k + 1)) < threshold:
        return []

    # ── Vec branch ────────────────────────────────────────────────────
    embedding: Any = None
    if orphan_embeddings is not None:
        embedding = orphan_embeddings.get(orphan_id)
    if embedding is None and _dir_ctx is not None:
        try:
            embedding = _dir_ctx.get_vector(orphan_id)
        except Exception:  # noqa: BLE001
            embedding = None
    elif embedding is None and db is not None and hasattr(db, "get_embedding_by_id"):
        try:
            embedding = db.get_embedding_by_id(orphan_id)
        except Exception:  # noqa: BLE001
            embedding = None
    if embedding is None and embed_fn is not None:
        # Last resort — embed the orphan's source text on the fly. The
        # caller decides whether the cost is acceptable; the v2 cascade
        # passes ``embed_fn`` only when persisted embeddings are missing.
        text = ""
        try:
            if db_node:
                text = db_node.get("source_text", "") or db_node.get("docstring", "") or ""
            if not text:
                node_data = G.nodes.get(orphan_id, {}) or {}
                text = node_data.get("source_text", "") or node_data.get("docstring", "") or ""
            if text and len(text.strip()) >= 10:
                embedding = embed_fn(text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("embed_fn fallback failed for %s: %s", orphan_id, exc)
            embedding = None
    if embedding is None or len(embedding) == 0:
        return []

    vec_hits: List[Dict[str, Any]] = []
    if db is not None and (_dir_ctx is not None or hasattr(db, "search_vec")):
        try:
            if _dir_ctx is not None:
                vec_hits = _dir_ctx.search_vec(
                    orphan_id,
                    embedding,
                    top_n=top_n,
                    fts_candidate_ids=[h["node_id"] for h in fts_hits if h.get("node_id")],
                )
            else:
                vec_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
                vec_hits = db.search_vec(
                    vec_list, k=top_n, path_prefix=local_dir or None,
                ) or []
        except TypeError:
            try:
                vec_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
                vec_hits = db.search_vec(vec_list, k=top_n) or []
            except Exception as exc:  # noqa: BLE001
                logger.debug("hybrid vec for %s failed: %s", orphan_id, exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("hybrid vec for %s failed: %s", orphan_id, exc)

    if vec_hits:
        vec_hits = [h for h in vec_hits if h.get("node_id") != orphan_id]

    # ── Fuse ──────────────────────────────────────────────────────────
    fused = rrf_fuse(fts_hits, vec_hits, k=k)
    if not fused:
        return []
    return [h for h in fused if h["rrf_score"] >= threshold][:top_n]


__all__ = ["rrf_fuse", "resolve_orphans_hybrid", "_DirHybridContext", "_LocalVecIndex"]
