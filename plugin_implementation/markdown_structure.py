"""Markdown document-structure edge synthesis (roadmap B6 / §3.5).

The documentation parser (:meth:`graph_builder._chunk_markdown_content`)
splits every markdown file into ``markdown_section`` nodes — one per
``H1``–``H4`` heading — but emits no edges between them and no parent
``markdown_document`` node for headered files. As a result the section
nodes are structurally disconnected: a reader of the graph cannot tell
which sections belong to the same document, nor which code symbols a
section talks about.

This module runs as a post-parse, pre-persist pass over the in-memory
relationship graph and closes that gap:

* **contains** — group ``markdown_section`` nodes by ``rel_path``,
  locate (or synthesize) the file's parent ``markdown_document`` node,
  and add a ``contains`` edge ``document → section`` for each child.
* **references** — for every markdown node, resolve the inline code
  references in its body (markdown links + ``\\`backtick\\``` symbol
  mentions) to real code nodes and add a ``references`` edge
  ``section → code symbol``.

Both edge kinds use ``edge_class="doc"`` so they feed Leiden
connectivity without being treated as architectural coupling by the
writer-evidence gate. Confidence is ``"EXTRACTED"`` — the relationships
are observed directly from the file structure and body text, not
inferred by name-only heuristics.

The reference-resolution machinery is reused verbatim from
:mod:`graph_orphan_cascade_v2` (the orphan cascade resolves the same
markdown-link / backtick references, but only for *orphan* doc nodes —
this pass applies it to *every* section).

The pass is side-effecting on the passed graph and returns a stats dict.
It must run before :func:`unified_db.UnifiedWikiDB.from_networkx` so the
new nodes/edges persist through the normal path.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import networkx as nx

from .feature_flags import FeatureFlags, get_feature_flags
from .graph_orphan_cascade_v2 import (
    build_path_index as _build_path_index,
    build_simple_name_index as _build_simple_name_index,
    resolve_doc_orphan_links as _resolve_doc_orphan_links,
)

logger = logging.getLogger(__name__)

#: Symbol types emitted by the markdown chunker.
_SECTION_TYPE = "markdown_section"
_DOCUMENT_TYPE = "markdown_document"
_MARKDOWN_TYPES = frozenset({_SECTION_TYPE, _DOCUMENT_TYPE})

#: Synthetic parent-node id prefix. Kept distinct from the parser's
#: ``{language}::{file_stem}::{symbol}`` scheme so it can never collide
#: with a parsed node.
_SYNTHETIC_DOC_PREFIX = "markdown_document::"


def _node_rel_path(data: Dict[str, Any]) -> str:
    """Return a node's ``rel_path``, falling back to a nested location."""
    rp = data.get("rel_path", "")
    if not rp:
        loc = data.get("location") or {}
        if isinstance(loc, dict):
            rp = loc.get("rel_path", "")
    return rp or ""


def _has_edge_of_type(
    G: nx.MultiDiGraph, u: str, v: str, rel_type: str
) -> bool:
    """True if a ``u → v`` edge with ``relationship_type == rel_type`` exists."""
    if not G.has_edge(u, v):
        return False
    bundle = G.get_edge_data(u, v) or {}
    for attrs in bundle.values():
        if (attrs or {}).get("relationship_type") == rel_type:
            return True
    return False


def wire_markdown_structure(
    G: nx.MultiDiGraph,
    *,
    flags: Optional[FeatureFlags] = None,
) -> Dict[str, int]:
    """Add ``contains`` + ``references`` edges for markdown documents.

    Args:
        G: The in-memory relationship graph (mutated in place).
        flags: Feature flags; defaults to :func:`get_feature_flags`.

    Returns:
        Stats dict with keys ``markdown_nodes``, ``documents_synthesized``,
        ``contains_edges``, ``references_edges``.
    """
    stats = {
        "markdown_nodes": 0,
        "documents_synthesized": 0,
        "contains_edges": 0,
        "references_edges": 0,
    }

    if flags is None:
        flags = get_feature_flags()
    if not getattr(flags, "markdown_structure", True):
        return stats

    md_nodes: List[tuple] = [
        (nid, data)
        for nid, data in G.nodes(data=True)
        if (data.get("symbol_type") or "").lower() in _MARKDOWN_TYPES
    ]
    if not md_nodes:
        return stats
    stats["markdown_nodes"] = len(md_nodes)

    # ── contains: parent markdown_document → markdown_section ───────────
    sections_by_path: Dict[str, List[str]] = defaultdict(list)
    document_by_path: Dict[str, str] = {}
    for nid, data in md_nodes:
        rel_path = _node_rel_path(data)
        sym_type = (data.get("symbol_type") or "").lower()
        if sym_type == _DOCUMENT_TYPE:
            document_by_path.setdefault(rel_path, nid)
        else:
            sections_by_path[rel_path].append(nid)

    for rel_path, section_ids in sections_by_path.items():
        if not rel_path:
            continue
        parent_id = document_by_path.get(rel_path)
        if parent_id is None:
            parent_id = f"{_SYNTHETIC_DOC_PREFIX}{rel_path}"
            if not G.has_node(parent_id):
                file_name = os.path.basename(rel_path)
                G.add_node(
                    parent_id,
                    name=file_name or rel_path,
                    symbol_name=file_name or rel_path,
                    symbol_type=_DOCUMENT_TYPE,
                    rel_path=rel_path,
                    file_name=file_name,
                    language="markdown",
                    start_line=0,
                    end_line=0,
                    analysis_level="documentation",
                    source_text="",
                    docstring="",
                    parameters=[],
                    return_type="markdown",
                )
                stats["documents_synthesized"] += 1
            document_by_path[rel_path] = parent_id

        for section_id in section_ids:
            if section_id == parent_id:
                continue
            if _has_edge_of_type(G, parent_id, section_id, "contains"):
                continue
            G.add_edge(
                parent_id,
                section_id,
                relationship_type="contains",
                edge_class="doc",
                weight=1.0,
                language="markdown",
                created_by="markdown_structure",
                annotations={"confidence": "EXTRACTED"},
            )
            stats["contains_edges"] += 1

    # ── references: markdown node → code symbol ─────────────────────────
    path_index = _build_path_index(G)
    name_index = _build_simple_name_index(G)

    for nid, data in md_nodes:
        source_text = data.get("source_text", "") or ""
        if not source_text:
            continue
        rel_path = _node_rel_path(data)
        source_dir = os.path.dirname(rel_path) if rel_path else ""
        hits = _resolve_doc_orphan_links(
            G, nid, source_text, source_dir, path_index, name_index,
        )
        for hit in hits:
            target = hit.get("node_id")
            if not target or target == nid or not G.has_node(target):
                continue
            if _has_edge_of_type(G, nid, target, "references"):
                continue
            G.add_edge(
                nid,
                target,
                relationship_type="references",
                edge_class="doc",
                weight=1.0,
                language="markdown",
                created_by="markdown_structure",
                raw_similarity=float(hit.get("_raw_score", 0.95)),
                annotations={
                    "confidence": "EXTRACTED",
                    "matcher": hit.get("_matcher", "explicit_ref"),
                },
            )
            stats["references_edges"] += 1

    logger.info(
        "Markdown structure: %d md nodes, %d documents synthesized, "
        "%d contains edges, %d references edges",
        stats["markdown_nodes"],
        stats["documents_synthesized"],
        stats["contains_edges"],
        stats["references_edges"],
    )
    return stats
