"""
Phase 4 — Cluster-Based Structure Planner

Converts pre-computed math clusters (Phases 1-3) into a full
``WikiStructureSpec`` with one cheap LLM call per macro-cluster.

The LLM no longer *discovers* structure — it only *names and describes*
what the graph algorithms already found.  This replaces 50-74 DeepAgents
tool calls with ~8-15 lightweight naming calls.

Activation::

    DEEPWIKI_STRUCTURE_PLANNER=cluster

Usage::

    from .cluster_planner import ClusterStructurePlanner

    planner = ClusterStructurePlanner(db, llm)
    spec = planner.plan_structure()
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

import networkx as nx

from ..graph_clustering import select_central_symbols
from ..state.wiki_state import PageSpec, SectionSpec, WikiStructureSpec
from ..unified_db import UnifiedWikiDB

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

#: Maximum dominant symbols to send to LLM per macro-cluster
MAX_DOMINANT_SYMBOLS = 10

#: Maximum symbols summarised per micro-cluster in the prompt
MAX_MICRO_SUMMARY_SYMBOLS = 8

#: Truncation length for signatures and docstrings in prompts
SIGNATURE_TRUNC = 200
DOCSTRING_TRUNC = 200

#: Label for the hub/core section
GLOBAL_CORE_SECTION_NAME = "System Core & Utilities"
GLOBAL_CORE_SECTION_DESC = (
    "Cross-cutting utilities, shared infrastructure, and high-connectivity "
    "hub symbols that are referenced by multiple subsystems."
)

# ═══════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════

CLUSTER_NAMING_SYSTEM = """\
You are a senior technical documentation architect.  You are given a \
mathematically verified cluster of code symbols from a software repository.  \
These symbols were grouped together because they have strong structural \
and semantic relationships (they call each other, import each other, or \
share data types).

Your task is strictly limited to **naming and describing** the cluster — \
the grouping itself is fixed and correct."""

CLUSTER_NAMING_USER = """\
CLUSTER #{cluster_id} — {node_count} symbols across {file_count} files

DOMINANT SYMBOLS (most representative by connectivity & documentation):
{dominant_symbols_json}

MICRO-CLUSTERS (sub-groupings within this cluster):
{micro_summaries_json}

YOUR TASK:
1. Give this Section a concise, capability-based name (e.g., \
"Authentication & Session Management", NOT "auth_service.py and related files").
2. Write a one-sentence section description.
3. For each micro-cluster, give it a concise Page name describing its \
specific functionality.
4. For each page, write a one-sentence description and a short \
retrieval_query that could be used to find relevant documentation.

Output ONLY valid JSON (no markdown fences):
{{
  "section_name": "...",
  "section_description": "...",
  "pages": [
    {{"micro_id": 0, "page_name": "...", "description": "...", "retrieval_query": "..."}},
    ...
  ]
}}"""


# ═══════════════════════════════════════════════════════════════════════════
# Planner
# ═══════════════════════════════════════════════════════════════════════════

class ClusterStructurePlanner:
    """Build a WikiStructureSpec from pre-computed clusters + cheap LLM naming.

    Parameters
    ----------
    db : UnifiedWikiDB
        Unified DB with Phase 3 cluster assignments already persisted.
    llm : BaseLanguageModel
        LangChain LLM used exclusively for naming (cheap, small prompts).
    wiki_title : str, optional
        Override wiki title; if *None*, derived from repo metadata.
    """

    def __init__(
        self,
        db: UnifiedWikiDB,
        llm: BaseLanguageModel,
        wiki_title: Optional[str] = None,
    ):
        self.db = db
        self.llm = llm
        self.wiki_title = wiki_title or self._derive_wiki_title()
        self._cluster_graph: Optional[nx.MultiDiGraph] = None
        self._central_k: Optional[int] = None

    # ── adaptive central-k ────────────────────────────────────────────

    def _adaptive_central_k(self) -> int:
        """Scale central symbol count with repo size.

        Smaller repos need fewer representative symbols per page;
        larger repos need more to capture the breadth of each cluster.

        | Nodes      | k (central symbols) |
        |------------|---------------------|
        | < 200      | 5                   |
        | 200–999    | 8                   |
        | 1000–4999  | 12                  |
        | 5000+      | 15                  |
        """
        if self._central_k is not None:
            return self._central_k

        G = self._get_cluster_graph()
        n = G.number_of_nodes()
        if n >= 5000:
            k = 15
        elif n >= 1000:
            k = 12
        elif n >= 200:
            k = 8
        else:
            k = 5
        self._central_k = k
        return k

    # ── public API ────────────────────────────────────────────────────

    def plan_structure(self) -> WikiStructureSpec:
        """Plan the full wiki structure from clusters.

        Returns a ``WikiStructureSpec`` ready for the page-generation graph.
        """
        t0 = time.time()

        # 1. Load cluster map from DB: macro → micro → [node_ids]
        #    Filter to architectural nodes only — methods/fields were propagated
        #    into clusters by Phase 3 but they inflate the cluster count and
        #    prompt size.  The expansion module already filters to architectural
        #    nodes (is_architectural = 1), so the planner should do the same.
        cluster_map = self._load_architectural_cluster_map()
        if not cluster_map:
            logger.warning("ClusterStructurePlanner: no clusters in DB — returning fallback")
            return self._fallback_spec()

        logger.info(
            "ClusterStructurePlanner: %d macro-clusters to name",
            len(cluster_map),
        )

        # 2. Load hub nodes assigned to global_core
        global_core_nodes = self._load_global_core_nodes()

        # 3. Name each macro-cluster via one LLM call
        sections: List[SectionSpec] = []
        section_order = 1

        for macro_id in sorted(cluster_map.keys()):
            micro_map = cluster_map[macro_id]
            try:
                section = self._name_macro_cluster(
                    macro_id=macro_id,
                    micro_map=micro_map,
                    section_order=section_order,
                )
                sections.append(section)
                section_order += 1
            except Exception as exc:
                logger.warning(
                    "LLM naming failed for macro %d, using fallback: %s",
                    macro_id, exc,
                )
                sections.append(self._fallback_section(
                    macro_id, micro_map, section_order,
                ))
                section_order += 1

        # 4. Add global_core section if there are hub nodes
        if global_core_nodes:
            core_section = self._build_global_core_section(
                global_core_nodes, section_order,
            )
            sections.append(core_section)

        total_pages = sum(len(s.pages) for s in sections)
        elapsed = time.time() - t0

        logger.info(
            "ClusterStructurePlanner: %d sections, %d pages in %.1fs",
            len(sections), total_pages, elapsed,
        )

        return WikiStructureSpec(
            wiki_title=self.wiki_title,
            overview=self._build_overview(sections),
            sections=sections,
            total_pages=total_pages,
        )

    # ── Architectural cluster map ──────────────────────────────────

    def _load_architectural_cluster_map(
        self,
    ) -> Dict[int, Dict[int, List[str]]]:
        """Load cluster map filtered to architectural nodes only.

        Phase 3 propagates assignments to child nodes (methods, fields, etc.)
        so that the unified DB has a complete mapping.  However, the structure
        planner should only consider top-level *architectural* symbols — the
        same set used by the content expansion module — to avoid inflating
        section and page counts.

        Returns ``{macro_id: {micro_id: [node_ids]}}`` where every node_id
        belongs to an architectural symbol.
        """
        rows = self.db.conn.execute(
            "SELECT node_id, macro_cluster, micro_cluster "
            "FROM repo_nodes "
            "WHERE macro_cluster IS NOT NULL AND is_architectural = 1"
        ).fetchall()

        result: Dict[int, Dict[int, List[str]]] = {}
        for row in rows:
            macro = row["macro_cluster"]
            micro = row["micro_cluster"] or 0
            result.setdefault(macro, {}).setdefault(micro, []).append(row["node_id"])

        # Remove empty micro-clusters (all their arch nodes were in a
        # different micro).  This shouldn't happen often, but guard anyway.
        for macro in list(result.keys()):
            for micro in list(result[macro].keys()):
                if not result[macro][micro]:
                    del result[macro][micro]
            if not result[macro]:
                del result[macro]

        total_nodes = sum(
            len(nids) for mm in result.values() for nids in mm.values()
        )
        logger.info(
            "Loaded architectural cluster map: %d macro-clusters, %d nodes "
            "(filtered from full cluster map)",
            len(result), total_nodes,
        )
        return result

    # ── LLM naming per macro-cluster ─────────────────────────────────

    def _name_macro_cluster(
        self,
        macro_id: int,
        micro_map: Dict[int, List[str]],
        section_order: int,
    ) -> SectionSpec:
        """Issue one LLM call to name a macro-cluster and its micro-clusters."""

        # Build compact payloads
        dominant = self._get_dominant_symbols(macro_id)
        micro_summaries = self._get_micro_summaries(micro_map)

        all_node_ids = set()
        for nids in micro_map.values():
            all_node_ids.update(nids)

        file_count = len({
            (self.db.get_node(nid) or {}).get("rel_path", "")
            for nid in list(all_node_ids)[:200]
        } - {""})

        prompt_text = CLUSTER_NAMING_USER.format(
            cluster_id=macro_id,
            node_count=len(all_node_ids),
            file_count=file_count,
            dominant_symbols_json=json.dumps(dominant, indent=2),
            micro_summaries_json=json.dumps(micro_summaries, indent=2),
        )

        messages = [
            SystemMessage(content=CLUSTER_NAMING_SYSTEM),
            HumanMessage(content=prompt_text),
        ]

        response = self.llm.invoke(messages)
        raw = _extract_text(response)

        naming = _parse_json_response(raw)

        # Build pages from micro-clusters + LLM naming
        pages: List[PageSpec] = []
        page_order = 1

        llm_pages = {p["micro_id"]: p for p in naming.get("pages", [])}

        for micro_id in sorted(micro_map.keys()):
            node_ids = micro_map[micro_id]
            llm_page = llm_pages.get(micro_id, {})

            page_name = llm_page.get("page_name") or f"Section {macro_id} — Page {micro_id}"
            page_desc = llm_page.get("description") or f"Page covering {len(node_ids)} symbols"
            retrieval_query = llm_page.get("retrieval_query") or page_name

            # Select central symbols via PageRank (Phase 7F)
            central_ids = self._select_central_node_ids(
                node_ids, k=self._adaptive_central_k(),
            )
            target_symbols = self._node_ids_to_symbol_names(central_ids)
            # Collect file paths for key_files
            key_files = self._node_ids_to_paths(node_ids)
            # Collect target folders (unique directory prefixes)
            target_folders = self._node_ids_to_folders(node_ids)
            # Collect documentation file paths for mixed retrieval
            target_docs = self._node_ids_to_doc_paths(node_ids)

            pages.append(PageSpec(
                page_name=page_name,
                page_order=page_order,
                description=page_desc,
                content_focus=page_desc,
                rationale=f"Grouped by graph clustering (macro={macro_id}, micro={micro_id}, {len(node_ids)} symbols)",
                target_symbols=target_symbols,
                target_docs=target_docs,
                target_folders=target_folders,
                key_files=key_files,
                retrieval_query=retrieval_query,
                metadata={"cluster_node_ids": list(node_ids)},
            ))
            page_order += 1

        section_name = naming.get("section_name") or f"Section {macro_id}"
        section_desc = naming.get("section_description") or f"Section covering {len(all_node_ids)} symbols"

        return SectionSpec(
            section_name=section_name,
            section_order=section_order,
            description=section_desc,
            rationale=f"Mathematical clustering: macro={macro_id}, {len(micro_map)} pages, {len(all_node_ids)} symbols",
            pages=pages,
        )

    # ── Dominant symbols for prompt ──────────────────────────────────

    def _get_dominant_symbols(self, macro_id: int) -> List[Dict[str, Any]]:
        """Get top-N most representative *architectural* symbols from a macro-cluster.

        Scoring: min(edge_count, 10) + has_docstring(2)
        Only architectural nodes are considered — methods, fields, etc. are excluded.
        """
        nodes = self.db.get_nodes_by_cluster(macro=macro_id)
        if not nodes:
            return []

        # Filter to architectural nodes only
        nodes = [n for n in nodes if n.get("is_architectural")]

        scored = []
        for n in nodes:
            # Count outgoing edges as proxy for connectivity
            edge_count = len(self.db.get_edges_from(n["node_id"]))
            score = (
                min(edge_count, 10)
                + (2 if n.get("docstring") else 0)
            )
            scored.append((score, n))

        scored.sort(key=lambda x: -x[0])

        return [
            {
                "name": n["symbol_name"],
                "type": n["symbol_type"],
                "path": n["rel_path"],
                "signature": (n.get("signature") or "")[:SIGNATURE_TRUNC],
                "docstring": (n.get("docstring") or "")[:DOCSTRING_TRUNC],
            }
            for _, n in scored[:MAX_DOMINANT_SYMBOLS]
        ]

    def _get_micro_summaries(
        self, micro_map: Dict[int, List[str]]
    ) -> List[Dict[str, Any]]:
        """Build compact summaries of each micro-cluster for the LLM prompt."""
        summaries = []
        for micro_id in sorted(micro_map.keys()):
            node_ids = micro_map[micro_id]
            # Fetch a sample of architectural nodes for the summary
            nodes = [
                self.db.get_node(nid)
                for nid in node_ids[:MAX_MICRO_SUMMARY_SYMBOLS * 3]
            ]
            nodes = [n for n in nodes if n and n.get("is_architectural")]
            nodes = nodes[:MAX_MICRO_SUMMARY_SYMBOLS]

            symbols = [
                {
                    "name": n["symbol_name"],
                    "type": n["symbol_type"],
                    "path": n["rel_path"],
                }
                for n in nodes
            ]

            # Unique directories
            dirs = sorted({
                n["rel_path"].rsplit("/", 1)[0]
                for n in nodes
                if "/" in n.get("rel_path", "")
            })

            summaries.append({
                "micro_id": micro_id,
                "symbol_count": len(node_ids),
                "symbols": symbols,
                "directories": dirs[:5],
            })

        return summaries

    # ── Global Core section (hub nodes) ──────────────────────────────

    def _load_global_core_nodes(self) -> List[Dict[str, Any]]:
        """Load hub nodes assigned to global_core."""
        rows = self.db.conn.execute(
            "SELECT * FROM repo_nodes WHERE is_hub = 1 AND hub_assignment = 'global_core'"
        ).fetchall()
        return [dict(r) for r in rows]

    def _build_global_core_section(
        self,
        core_nodes: List[Dict[str, Any]],
        section_order: int,
    ) -> SectionSpec:
        """Build a dedicated section for global-core hub symbols."""
        # Group into a single page (or split if too many)
        page_size = 25
        pages: List[PageSpec] = []

        for i in range(0, len(core_nodes), page_size):
            chunk = core_nodes[i : i + page_size]
            symbols = [n["symbol_name"] for n in chunk]
            paths = sorted({n["rel_path"] for n in chunk if n.get("rel_path")})
            folders = sorted({
                p.rsplit("/", 1)[0] for p in paths if "/" in p
            })
            doc_paths = sorted({n["rel_path"] for n in chunk if n.get("is_doc")})

            page_num = (i // page_size) + 1
            page_name = (
                "Core Utilities & Shared Infrastructure"
                if page_num == 1
                else f"Core Utilities — Part {page_num}"
            )

            pages.append(PageSpec(
                page_name=page_name,
                page_order=page_num,
                description="High-connectivity hub symbols referenced across multiple subsystems",
                content_focus="Shared utilities, base classes, and cross-cutting infrastructure",
                rationale=f"Hub nodes assigned to global_core ({len(chunk)} symbols)",
                target_symbols=symbols,
                target_docs=doc_paths,
                target_folders=folders[:10],
                key_files=paths[:20],
                retrieval_query="core utilities shared infrastructure base classes helpers",
            ))

        return SectionSpec(
            section_name=GLOBAL_CORE_SECTION_NAME,
            section_order=section_order,
            description=GLOBAL_CORE_SECTION_DESC,
            rationale=f"Hub re-integration: {len(core_nodes)} high-connectivity symbols",
            pages=pages,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _node_ids_to_symbol_names(self, node_ids: List[str]) -> List[str]:
        """Convert node_ids to symbol names (for target_symbols field).

        Only architectural *code* nodes are included — methods/fields are
        excluded because the expansion module resolves symbols with
        ``is_architectural = 1`` filter.  Documentation nodes (``is_doc = 1``)
        are also excluded here — they go into ``target_docs`` via
        :meth:`_node_ids_to_doc_paths` instead.
        """
        names = []
        for nid in node_ids:
            node = self.db.get_node(nid)
            if node and node.get("is_architectural") and not node.get("is_doc"):
                names.append(node["symbol_name"])
        return names

    def _node_ids_to_paths(self, node_ids: List[str]) -> List[str]:
        """Extract unique file paths from node_ids."""
        paths = set()
        for nid in node_ids:
            node = self.db.get_node(nid)
            if node and node.get("rel_path"):
                paths.add(node["rel_path"])
        return sorted(paths)[:20]  # Cap at 20

    def _node_ids_to_folders(self, node_ids: List[str]) -> List[str]:
        """Extract unique directory prefixes from node_ids."""
        folders = set()
        for nid in node_ids:
            node = self.db.get_node(nid)
            if node and "/" in (node.get("rel_path") or ""):
                folders.add(node["rel_path"].rsplit("/", 1)[0])
        return sorted(folders)[:10]

    def _node_ids_to_doc_paths(self, node_ids: List[str]) -> List[str]:
        """Extract documentation file paths from node_ids.

        Returns ``rel_path`` for every node that has ``is_doc = 1``.
        These paths are used as ``target_docs`` on :class:`PageSpec`,
        enabling the MIXED retrieval path (code + docs) during content
        expansion.
        """
        paths = set()
        for nid in node_ids:
            node = self.db.get_node(nid)
            if node and node.get("is_doc") and node.get("rel_path"):
                paths.add(node["rel_path"])
        return sorted(paths)[:20]

    # ── Central symbol selection (Phase 7F) ──────────────────────────

    def _get_cluster_graph(self) -> nx.MultiDiGraph:
        """Lazily reconstruct an NX graph from DB edges for PageRank.

        The graph is cached for the lifetime of the planner instance to
        avoid rebuilding it for every micro-cluster.
        """
        if self._cluster_graph is not None:
            return self._cluster_graph

        G = nx.MultiDiGraph()

        # Add all architectural nodes
        rows = self.db.conn.execute(
            "SELECT node_id FROM repo_nodes WHERE is_architectural = 1"
        ).fetchall()
        for row in rows:
            G.add_node(row["node_id"])

        # Add all edges with weights
        edge_rows = self.db.conn.execute(
            "SELECT source_id, target_id, rel_type, weight FROM repo_edges"
        ).fetchall()
        for row in edge_rows:
            src, tgt = row["source_id"], row["target_id"]
            if src in G and tgt in G:
                G.add_edge(
                    src, tgt,
                    relationship_type=row["rel_type"],
                    weight=row["weight"] or 1.0,
                )

        self._cluster_graph = G
        logger.debug(
            "Built cluster graph from DB: %d nodes, %d edges",
            G.number_of_nodes(), G.number_of_edges(),
        )
        return G

    def _select_central_node_ids(
        self, node_ids: List[str], k: int = 10,
    ) -> List[str]:
        """Select the top-k most central node IDs from a micro-cluster.

        Uses PageRank on the reconstructed graph. Falls back to the full
        list if the graph has too few edges for meaningful ranking.
        """
        G = self._get_cluster_graph()
        cluster_set = set(node_ids) & set(G.nodes())

        if len(cluster_set) <= k:
            return list(cluster_set)

        central = select_central_symbols(G, cluster_set, k=k)
        if not central:
            # Fallback: return first k node_ids
            return node_ids[:k]
        return central

    def _derive_wiki_title(self) -> str:
        """Try to derive the wiki title from DB metadata."""
        repo_id = self.db.get_meta("repo_identifier")
        if repo_id:
            name = repo_id.split("/")[-1].split(":")[0] if "/" in repo_id else repo_id
            return f"{name} — Technical Documentation"
        return "Repository Technical Documentation"

    def _build_overview(self, sections: List[SectionSpec]) -> str:
        """Generate a brief overview from section names."""
        section_list = ", ".join(s.section_name for s in sections)
        return (
            f"This wiki covers {sum(len(s.pages) for s in sections)} pages "
            f"across {len(sections)} sections: {section_list}."
        )

    def _fallback_spec(self) -> WikiStructureSpec:
        """Minimal spec when no clusters are available."""
        return WikiStructureSpec(
            wiki_title=self.wiki_title,
            overview="No clusters found — using fallback structure.",
            sections=[
                SectionSpec(
                    section_name="Repository Overview",
                    section_order=1,
                    description="General repository documentation",
                    rationale="Fallback: no clusters available",
                    pages=[
                        PageSpec(
                            page_name="Overview",
                            page_order=1,
                            description="General overview of the repository",
                            content_focus="Repository structure and key components",
                            rationale="Fallback page",
                            retrieval_query="repository overview main components",
                        )
                    ],
                )
            ],
            total_pages=1,
        )

    def _fallback_section(
        self,
        macro_id: int,
        micro_map: Dict[int, List[str]],
        section_order: int,
    ) -> SectionSpec:
        """Fallback section when LLM naming fails for a cluster."""
        all_node_ids = []
        for nids in micro_map.values():
            all_node_ids.extend(nids)

        # Try to derive a name from dominant file paths
        paths = self._node_ids_to_paths(all_node_ids[:50])
        if paths:
            common = os.path.commonpath(paths) if len(paths) > 1 else paths[0]
            section_name = f"Module: {common}" if common else f"Section {macro_id}"
        else:
            section_name = f"Section {macro_id}"

        pages = []
        page_order = 1
        for micro_id in sorted(micro_map.keys()):
            node_ids = micro_map[micro_id]
            # Central symbol selection (Phase 7F)
            central_ids = self._select_central_node_ids(
                node_ids, k=self._adaptive_central_k(),
            )
            target_symbols = self._node_ids_to_symbol_names(central_ids)
            key_files = self._node_ids_to_paths(node_ids)

            pages.append(PageSpec(
                page_name=f"{section_name} — Page {micro_id}",
                page_order=page_order,
                description=f"Page with {len(node_ids)} symbols",
                content_focus=f"Symbols in cluster {macro_id}/{micro_id}",
                rationale=f"Fallback naming for macro={macro_id}, micro={micro_id}",
                target_symbols=target_symbols,
                target_docs=self._node_ids_to_doc_paths(node_ids),
                target_folders=self._node_ids_to_folders(node_ids),
                key_files=key_files,
                retrieval_query=" ".join(target_symbols[:5]),
                metadata={"cluster_node_ids": list(node_ids)},
            ))
            page_order += 1

        return SectionSpec(
            section_name=section_name,
            section_order=section_order,
            description=f"Auto-generated section with {len(all_node_ids)} symbols",
            rationale=f"Fallback: LLM naming failed for macro={macro_id}",
            pages=pages,
        )


# ═══════════════════════════════════════════════════════════════════════════
# JSON parsing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_text(response: Any) -> str:
    """Pull plain text from a LangChain AI message."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "\n".join(parts)
    return str(content)


def _parse_json_response(raw: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from LLM output.

    Handles markdown fences, leading text, trailing commas, etc.
    """
    import re

    text = raw.strip()

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first { ... last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Remove trailing commas before } or ]
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse LLM JSON output (length %d)", len(raw))
    return {}
