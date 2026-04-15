"""
Phase 3 — Mathematical Pre-Clustering

Deterministically partitions the codebase into sections (macro-clusters)
and pages (micro-clusters) using Louvain community detection, eliminating
the LLM's expensive exploratory tool calls.

Pipeline:

1. **Macro-clustering** — Louvain on the weighted, hub-free undirected
   projection.  Resolution γ auto-tuned by graph size.
2. **Micro-clustering** — Sub-Louvain per macro-cluster at higher γ.
3. **Dynamic page sizing** — merge tiny clusters, split oversized ones.
4. **Hub re-integration** — majority-vote assignment or "global_core".
5. **Persist** — write all assignments to the unified DB.

Feature-flagged via ``DEEPWIKI_UNIFIED_DB=1`` (same flag as Phases 1-2).

Usage::

    from .graph_clustering import run_phase3

    results = run_phase3(db, G, hubs)
"""

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from .constants import ARCHITECTURAL_SYMBOLS

try:
    import igraph as ig
    _HAS_IGRAPH = True
except ImportError:
    _HAS_IGRAPH = False

try:
    import leidenalg
    _HAS_LEIDEN = True
except ImportError:
    _HAS_LEIDEN = False

# Composite flags for algorithm auto-selection
_HAS_INFOMAP = _HAS_IGRAPH  # igraph ships community_infomap()

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

MICRO_CLUSTER_RULES = {
    "min_page_size": 5,     # Merge clusters smaller than this
    "max_page_size": 25,    # Default; overridden by _adaptive_max_page_size()
    "merge_threshold": 4,   # Hard minimum — below this, always merge
}


def _adaptive_max_page_size(node_count: int) -> int:
    """Scale max page size based on repo complexity.

    Larger repos need bigger pages to avoid 100+ page wikis.
    This aligns micro-cluster caps with the planner's
    ``MAX_SYMBOLS_PER_PAGE`` heuristic.

    | Nodes      | Max page size |
    |------------|---------------|
    | < 500      | 25            |
    | 500–1999   | 35            |
    | 2000–4999  | 45            |
    | 5000+      | 60            |
    """
    if node_count >= 5000:
        return 60
    elif node_count >= 2000:
        return 45
    elif node_count >= 500:
        return 35
    return 25


def select_central_symbols(
    G: nx.MultiDiGraph,
    cluster_nodes: Set[str],
    k: int = 5,
) -> List[str]:
    """Select the top-*k* most central symbols in a cluster using PageRank.

    The returned node IDs represent the "representative" symbols that best
    characterise the cluster.  These are used as ``target_symbols`` in the
    wiki structure planner instead of sending ALL node IDs (which can be
    hundreds for large micro-clusters).

    Uses PageRank on the **undirected** projection so that hub nodes (high
    out-degree, e.g. README linking many pages) rank equally with authority
    nodes (high in-degree, e.g. utility functions called by many).

    Args:
        G: Full graph (we extract the subgraph for *cluster_nodes*).
        cluster_nodes: Set of node IDs belonging to this cluster.
        k: Number of central symbols to return.

    Returns:
        List of node IDs sorted by descending PageRank, length ≤ k.
    """
    if not cluster_nodes:
        return []

    k = min(k, len(cluster_nodes))

    # Extract subgraph and convert to undirected for symmetric centrality
    sub = G.subgraph(cluster_nodes)
    undirected = sub.to_undirected()

    try:
        pr = nx.pagerank(undirected, weight="weight", max_iter=100, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        # Fallback: uniform rank
        pr = {n: 1.0 / len(cluster_nodes) for n in cluster_nodes}

    ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in ranked[:k]]


# Hub re-integration: if >60% edges go to one cluster, assign there
HUB_MAJORITY_THRESHOLD = 0.6
GLOBAL_CORE_LABEL = "global_core"

# Symbol types that are always sub-components of a parent (never standalone
# cluster members).  If their parent is an architectural node the edge is
# "promoted" to that parent; otherwise the node is kept as-is.
_CHILD_SYMBOL_TYPES = frozenset({
    'method', 'constructor', 'destructor', 'field', 'variable',
    'property', 'getter', 'setter',
})


# ═══════════════════════════════════════════════════════════════════════════
# 0. Architectural Projection
# ═══════════════════════════════════════════════════════════════════════════

def architectural_projection(
    G: nx.MultiDiGraph,
) -> nx.MultiDiGraph:
    """Project *G* down to architectural-only nodes for community detection.

    Methods, constructors, fields, and variables inflate the node count by
    5-15× compared to the top-level symbols that actually define code
    structure.  Running Louvain on the full graph produces far too many
    communities for small-to-medium repos.

    This function:

    1. Keeps every node whose ``symbol_type`` is in
       :data:`ARCHITECTURAL_SYMBOLS` (classes, standalone functions,
       constants, doc chunks, …).
    2. For each *child* node (method, constructor, field, variable)
       resolves its parent architectural node using the ``node_id``
       naming convention (``lang::file::Parent.child`` → ``lang::file::Parent``).
    3. Re-maps every edge so that child endpoints are replaced by their
       parent.  Edges that become self-loops (both ends map to the same
       parent) are dropped.  Weights are summed.

    The result is a much smaller graph that still captures inter-module
    connectivity and produces sensible Louvain communities.
    """
    # ── 1. Classify nodes ──
    arch_nodes: Set[str] = set()
    child_to_parent: Dict[str, str] = {}   # child_node_id → arch parent_node_id

    # Build a fast lookup: (file_name, symbol_name) → node_id for arch nodes
    _arch_by_file_sym: Dict[tuple, str] = {}

    for nid, data in G.nodes(data=True):
        stype = (data.get("symbol_type") or "").lower()
        if stype in ARCHITECTURAL_SYMBOLS:
            arch_nodes.add(nid)
            fname = data.get("file_name", "")
            sname = data.get("symbol_name", "")
            if fname and sname:
                _arch_by_file_sym[(fname, sname)] = nid
        # Nodes with empty/unknown type are kept as-is (they usually come
        # from inferred edge targets and are already top-level).

    for nid, data in G.nodes(data=True):
        if nid in arch_nodes:
            continue
        stype = (data.get("symbol_type") or "").lower()
        if stype not in _CHILD_SYMBOL_TYPES:
            # Keep non-child, non-architectural nodes as-is (e.g. 'inferred',
            # 'unknown', 'module').  They still participate in clustering.
            arch_nodes.add(nid)
            continue

        # Try to resolve parent via node_id convention:
        #   lang::file::Parent.child  →  lang::file::Parent
        parts = nid.split("::", 2)
        if len(parts) == 3:
            lang, fname, qualified = parts
            if "." in qualified:
                parent_name = qualified.rsplit(".", 1)[0]
                parent_nid = f"{lang}::{fname}::{parent_name}"
                if parent_nid in arch_nodes or G.has_node(parent_nid):
                    child_to_parent[nid] = parent_nid
                    # Ensure parent is considered architectural
                    arch_nodes.add(parent_nid)
                    continue

            # Fallback: look up parent_symbol attribute
            psym = data.get("parent_symbol")
            if psym:
                candidate = _arch_by_file_sym.get((fname, psym))
                if candidate:
                    child_to_parent[nid] = candidate
                    continue

        # No resolvable parent — keep the node as a standalone cluster member
        arch_nodes.add(nid)

    projected = len(child_to_parent)
    remaining = len(arch_nodes)
    logger.info(
        "Architectural projection: %d → %d nodes (%d children promoted to parents)",
        G.number_of_nodes(), remaining, projected,
    )

    # ── 2. Build projected graph ──
    P = nx.MultiDiGraph()
    for nid in arch_nodes:
        if G.has_node(nid):
            P.add_node(nid, **G.nodes[nid])

    # Remap helper — resolves children to their parent arch node
    def _remap(nid: str) -> Optional[str]:
        if nid in arch_nodes:
            return nid
        return child_to_parent.get(nid)

    for u, v, data in G.edges(data=True):
        mu = _remap(u)
        mv = _remap(v)
        if mu is None or mv is None:
            continue  # orphaned child with no parent — skip
        if mu == mv:
            continue  # internal edge (method→method in same class)
        P.add_edge(mu, mv, **data)

    return P


# ═══════════════════════════════════════════════════════════════════════════
# 1. Resolution Auto-Tuning
# ═══════════════════════════════════════════════════════════════════════════

def auto_resolution(node_count: int) -> float:
    """Pick Louvain resolution γ based on graph size.

    Lower γ → fewer, larger communities.  Tuned so that architectural
    projections of typical repos land in a reasonable section range
    *before* the merge-cap step further consolidates.

    | Nodes       | γ    | Expected sections |
    |-------------|------|-------------------|
    | < 100       | ≥0.6 | 3–6               |
    | 100–500     | 0.6–0.5 | 5–10           |
    | 500–2,000   | 0.5–0.3 | 8–15           |
    | > 2,000     | 0.3  | 12–25             |

    Formula: ``γ = max(0.3, 1.0 − 0.2 × log₁₀(node_count))``
    """
    if node_count < 2:
        return 1.0
    return max(0.3, 1.0 - 0.2 * math.log10(node_count))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Macro-Clustering (Wiki Sections)
# ═══════════════════════════════════════════════════════════════════════════

def macro_cluster(
    G: nx.MultiDiGraph,
    hubs: Set[str],
    resolution: Optional[float] = None,
    algorithm: str = "louvain",
) -> Dict[str, int]:
    """Partition non-hub nodes into macro-clusters (wiki sections).

    Creates an undirected projection (Louvain requires undirected),
    removes hub nodes, runs community detection with weighted edges.

    Args:
        G: Weighted ``MultiDiGraph`` (after Phase 2 ``apply_edge_weights``).
        hubs: Set of hub node IDs to exclude from clustering.
        resolution: Louvain/Leiden γ. If None, auto-tuned by graph size.
        algorithm: ``"louvain"`` (default), ``"leiden"``, or ``"auto"``
            (Leiden if available, else Louvain).

    Returns:
        Mapping ``{node_id: macro_cluster_id}``.
    """
    # Build hub-free working copy
    work_nodes = set(G.nodes()) - hubs
    if not work_nodes:
        return {}

    G_work = G.subgraph(work_nodes).copy()

    if resolution is None:
        resolution = auto_resolution(G_work.number_of_nodes())

    # Resolve algorithm
    algo = algorithm
    if algo == "auto":
        algo = "leiden" if _HAS_LEIDEN else "louvain"

    logger.info(
        "Macro-clustering: %d nodes (excl %d hubs), resolution=%.2f, algo=%s",
        G_work.number_of_nodes(), len(hubs), resolution, algo,
    )

    if algo == "leiden" and _HAS_LEIDEN:
        return _leiden_macro(G_work, resolution)

    # Default: Louvain
    return _louvain_macro(G_work, resolution)


def _louvain_macro(
    G_work: nx.MultiDiGraph,
    resolution: float,
) -> Dict[str, int]:
    """Louvain macro-clustering (original implementation)."""
    G_undirected = _to_weighted_undirected(G_work)

    if G_undirected.number_of_nodes() < 2:
        return {n: 0 for n in G_work.nodes()}

    communities = nx.community.louvain_communities(
        G_undirected, weight="weight", resolution=resolution, seed=42,
    )

    assignment: Dict[str, int] = {}
    for cluster_id, members in enumerate(sorted(communities, key=len, reverse=True)):
        for node_id in members:
            assignment[node_id] = cluster_id

    logger.info(
        "Macro-clustering produced %d sections from %d nodes",
        len(communities), len(assignment),
    )
    return assignment


def _leiden_macro(
    G_work: nx.MultiDiGraph,
    resolution: float,
) -> Dict[str, int]:
    """Leiden macro-clustering via igraph bridge.

    Leiden guarantees all communities are internally connected (unlike
    Louvain which can produce disconnected communities).
    """
    G_undirected = _to_weighted_undirected(G_work)

    if G_undirected.number_of_nodes() < 2:
        return {n: 0 for n in G_work.nodes()}

    ig_graph, node_list = _nx_to_igraph(G_undirected)

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
    )

    # Map igraph partition back to node IDs, sort by size descending
    community_map: Dict[int, List[str]] = {}
    for idx, comm_id in enumerate(partition.membership):
        community_map.setdefault(comm_id, []).append(node_list[idx])

    assignment: Dict[str, int] = {}
    for new_id, (_, members) in enumerate(
        sorted(community_map.items(), key=lambda kv: -len(kv[1]))
    ):
        for node_id in members:
            assignment[node_id] = new_id

    logger.info(
        "Leiden macro-clustering produced %d sections from %d nodes",
        len(community_map), len(assignment),
    )
    return assignment


# ───────────────────────────────────────────────────────────────────────────
# 2b. Section-Count Cap (merge small macro-clusters)
# ───────────────────────────────────────────────────────────────────────────

def _max_sections(arch_node_count: int) -> int:
    """Target upper bound on macro-cluster (section) count.

    | Arch nodes | Max sections |
    |------------|--------------|
    | < 50       | 3–4          |
    | 100        | 6            |
    | 200        | 7            |
    | 500        | 8            |
    | 1,000      | 9            |
    | 5,000      | 11           |
    | 10,000+    | 12–15        |
    """
    if arch_node_count < 10:
        return 3
    return max(3, min(15, round(math.log10(arch_node_count) * 3)))


def _max_pages_per_section(section_node_count: int) -> int:
    """Target upper bound on micro-clusters (pages) per macro-cluster.

    Analogous to :func:`_max_sections` but for within-section granularity.
    Prevents one large section from producing dozens of pages.

    | Section nodes | Max pages |
    |---------------|-----------|
    | < 10          | 2–3       |
    | 20            | 4         |
    | 50            | 5         |
    | 100           | 6         |
    | 200+          | 7–8       |
    """
    if section_node_count < 6:
        return 2
    return max(2, min(8, round(math.log10(section_node_count) * 3)))


def merge_macro_clusters(
    G: nx.MultiDiGraph,
    assignments: Dict[str, int],
    max_sections: int,
) -> Dict[str, int]:
    """Iteratively merge the smallest macro-cluster into its nearest neighbour.

    Repeats until ``len(clusters) ≤ max_sections``.  Inter-cluster affinity
    is measured by sum of edge weights crossing the boundary (computed once,
    then updated incrementally on each merge).
    """
    n_initial = len(set(assignments.values()))
    if n_initial <= max_sections:
        return assignments

    # cluster_id → set-of-node-ids
    clusters: Dict[int, Set[str]] = {}
    for nid, cid in assignments.items():
        clusters.setdefault(cid, set()).add(nid)

    # Pre-compute pairwise inter-cluster affinity
    node_to_cluster: Dict[str, int] = dict(assignments)
    affinity: Dict[Tuple[int, int], float] = {}
    for u, v, data in G.edges(data=True):
        cu = node_to_cluster.get(u)
        cv = node_to_cluster.get(v)
        if cu is not None and cv is not None and cu != cv:
            key = (min(cu, cv), max(cu, cv))
            affinity[key] = affinity.get(key, 0.0) + data.get("weight", 1.0)

    while len(clusters) > max_sections:
        # Pick the smallest cluster
        smallest = min(clusters, key=lambda c: len(clusters[c]))

        # Find most-connected neighbour (highest affinity)
        best: Optional[int] = None
        best_w = -1.0
        for cid in clusters:
            if cid == smallest:
                continue
            key = (min(smallest, cid), max(smallest, cid))
            w = affinity.get(key, 0.0)
            if w > best_w or (
                w == best_w
                and best is not None
                and len(clusters[cid]) > len(clusters[best])
            ):
                best_w = w
                best = cid

        if best is None:
            break  # safety — should never happen

        # Merge smallest into best
        clusters[best] |= clusters.pop(smallest)
        for nid in clusters[best]:
            node_to_cluster[nid] = best

        # Fold smallest's affinity entries into best
        updated: Dict[Tuple[int, int], float] = {}
        for (c1, c2), w in affinity.items():
            nc1 = best if c1 == smallest else c1
            nc2 = best if c2 == smallest else c2
            if nc1 == nc2:
                continue  # now intra-cluster
            key = (min(nc1, nc2), max(nc1, nc2))
            updated[key] = updated.get(key, 0.0) + w
        affinity = updated

    # Renumber 0..N-1, largest cluster first
    result: Dict[str, int] = {}
    for new_id, (_, nodes) in enumerate(
        sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    ):
        for nid in nodes:
            result[nid] = new_id

    logger.info(
        "Merged macro-clusters: %d → %d sections (cap=%d)",
        n_initial, len(set(result.values())), max_sections,
    )
    return result


def _to_weighted_undirected(G: nx.MultiDiGraph) -> nx.Graph:
    """Collapse directed multi-edges into an undirected weighted graph.

    For each pair (u, v), sums all parallel edge weights.
    """
    U = nx.Graph()
    U.add_nodes_from(G.nodes())

    edge_weights: Dict[tuple, float] = {}
    for u, v, data in G.edges(data=True):
        key = (min(u, v), max(u, v))
        edge_weights[key] = edge_weights.get(key, 0.0) + data.get("weight", 1.0)

    for (u, v), w in edge_weights.items():
        U.add_edge(u, v, weight=w)

    return U


def _nx_to_igraph(G_undirected: nx.Graph) -> Tuple["ig.Graph", List[str]]:
    """Convert a NetworkX undirected graph to an igraph Graph.

    Returns:
        (ig_graph, node_list) where node_list maps igraph vertex indices
        back to NetworkX node IDs.
    """
    node_list = sorted(G_undirected.nodes())
    node_to_idx = {nid: i for i, nid in enumerate(node_list)}

    ig_graph = ig.Graph(n=len(node_list), directed=False)
    edges = []
    weights = []
    for u, v, data in G_undirected.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(data.get("weight", 1.0))

    ig_graph.add_edges(edges)
    ig_graph.es["weight"] = weights
    return ig_graph, node_list


def _nx_directed_to_igraph(
    G: nx.MultiDiGraph,
    nodes: Set[str],
) -> Tuple["ig.Graph", List[str]]:
    """Convert a directed NetworkX subgraph to an igraph Graph.

    Used by InfoMap which operates on directed, weighted graphs.

    Returns:
        (ig_graph, node_list) where node_list maps igraph vertex indices
        back to NetworkX node IDs.
    """
    node_list = sorted(nodes)
    node_to_idx = {nid: i for i, nid in enumerate(node_list)}

    subgraph = G.subgraph(nodes)
    edges = []
    weights = []
    for u, v, data in subgraph.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(data.get("weight", 1.0))

    ig_graph = ig.Graph(n=len(node_list), directed=True)
    ig_graph.add_edges(edges)
    ig_graph.es["weight"] = weights
    return ig_graph, node_list


# ═══════════════════════════════════════════════════════════════════════════
# 3. Micro-Clustering (Wiki Pages)
# ═══════════════════════════════════════════════════════════════════════════

def micro_cluster(
    G: nx.MultiDiGraph,
    macro_nodes: Set[str],
    resolution: float = 1.5,
    algorithm: str = "louvain",
) -> Dict[str, int]:
    """Sub-partition a macro-cluster into micro-clusters (wiki pages).

    Args:
        G: Full graph (only the subgraph of ``macro_nodes`` is used).
        macro_nodes: Node IDs in this macro-cluster.
        resolution: Higher γ = finer granularity.
        algorithm: ``"auto"`` (InfoMap if available, else Louvain),
            ``"infomap"``, or ``"louvain"`` (default).

    Returns:
        Mapping ``{node_id: micro_cluster_id}`` (IDs local to this macro).
    """
    if len(macro_nodes) < 4:
        return {n: 0 for n in macro_nodes}

    # Resolve algorithm
    algo = algorithm
    if algo == "auto":
        algo = "infomap" if _HAS_IGRAPH else "louvain"

    if algo == "infomap" and _HAS_IGRAPH:
        return _infomap_micro(G, macro_nodes)
    if algo == "leiden" and _HAS_LEIDEN:
        return _leiden_micro(G, macro_nodes, resolution)

    # Default: Louvain on undirected projection
    return _louvain_micro(G, macro_nodes, resolution)


def _louvain_micro(
    G: nx.MultiDiGraph,
    macro_nodes: Set[str],
    resolution: float = 1.5,
) -> Dict[str, int]:
    """Louvain micro-clustering (original implementation)."""
    subgraph = G.subgraph(macro_nodes).copy()
    G_undirected = _to_weighted_undirected(subgraph)

    if G_undirected.number_of_edges() == 0:
        return {n: 0 for n in macro_nodes}

    communities = nx.community.louvain_communities(
        G_undirected, weight="weight", resolution=resolution, seed=42,
    )

    assignment: Dict[str, int] = {}
    for micro_id, members in enumerate(sorted(communities, key=len, reverse=True)):
        for node_id in members:
            assignment[node_id] = micro_id

    return assignment


def _infomap_micro(
    G: nx.MultiDiGraph,
    macro_nodes: Set[str],
) -> Dict[str, int]:
    """InfoMap micro-clustering via igraph's community_infomap().

    Uses igraph's C-level InfoMap implementation which operates on
    *directed* weighted graphs.  Regions where a random walk gets
    "trapped" (high internal flow) become clusters — these correspond
    to functional execution paths, ideal for wiki pages.
    """
    subgraph = G.subgraph(macro_nodes).copy()

    if subgraph.number_of_edges() == 0:
        return {n: 0 for n in macro_nodes}

    ig_graph, node_list = _nx_directed_to_igraph(G, macro_nodes)

    communities = ig_graph.community_infomap(edge_weights="weight")

    # Map igraph membership back to node IDs
    assignment: Dict[str, int] = {}
    for idx, comm_id in enumerate(communities.membership):
        assignment[node_list[idx]] = comm_id

    # Renumber 0..N-1, largest cluster first
    cluster_sizes: Dict[int, int] = Counter(assignment.values())
    ranked = sorted(cluster_sizes.keys(), key=lambda c: -cluster_sizes[c])
    id_map = {old: new for new, old in enumerate(ranked)}
    return {nid: id_map[mid] for nid, mid in assignment.items()}


def _leiden_micro(
    G: nx.MultiDiGraph,
    macro_nodes: Set[str],
    resolution: float = 1.5,
) -> Dict[str, int]:
    """Leiden micro-clustering via igraph bridge."""
    subgraph = G.subgraph(macro_nodes).copy()
    G_undirected = _to_weighted_undirected(subgraph)

    if G_undirected.number_of_edges() == 0:
        return {n: 0 for n in macro_nodes}

    ig_graph, node_list = _nx_to_igraph(G_undirected)

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
    )

    assignment: Dict[str, int] = {}
    for idx, comm_id in enumerate(partition.membership):
        assignment[node_list[idx]] = comm_id

    # Renumber 0..N-1, largest cluster first
    cluster_sizes: Dict[int, int] = Counter(assignment.values())
    ranked = sorted(cluster_sizes.keys(), key=lambda c: -cluster_sizes[c])
    id_map = {old: new for new, old in enumerate(ranked)}
    return {nid: id_map[mid] for nid, mid in assignment.items()}


def micro_cluster_all(
    G: nx.MultiDiGraph,
    macro_assignments: Dict[str, int],
    resolution: float = 1.5,
    algorithm: str = "louvain",
) -> Dict[int, Dict[str, int]]:
    """Run micro-clustering on every macro-cluster.

    Returns:
        ``{macro_id: {node_id: micro_id}}``.
    """
    # Invert: macro_id → set of node_ids
    clusters: Dict[int, Set[str]] = {}
    for nid, mid in macro_assignments.items():
        clusters.setdefault(mid, set()).add(nid)

    result: Dict[int, Dict[str, int]] = {}
    for macro_id, nodes in sorted(clusters.items()):
        result[macro_id] = micro_cluster(G, nodes, resolution, algorithm=algorithm)

    total_micros = sum(
        len(set(m.values())) for m in result.values()
    )
    logger.info(
        "Micro-clustering: %d macro-clusters → %d total micro-clusters",
        len(result), total_micros,
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. Dynamic Page Sizing
# ═══════════════════════════════════════════════════════════════════════════

def apply_page_sizing(
    G: nx.MultiDiGraph,
    macro_id: int,
    micro_assignments: Dict[str, int],
    rules: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Enforce min/max page sizes by merging/splitting micro-clusters.

    Args:
        G: Full graph.
        macro_id: ID of the macro-cluster being processed.
        micro_assignments: ``{node_id: micro_id}`` within this macro.
        rules: Override ``MICRO_CLUSTER_RULES``.

    Returns:
        Updated ``{node_id: micro_id}`` after merges/splits.
    """
    r = rules or MICRO_CLUSTER_RULES
    min_size = r["min_page_size"]
    max_size = r["max_page_size"]
    merge_thresh = r["merge_threshold"]

    # Build micro_id → [node_ids]
    clusters: Dict[int, List[str]] = {}
    for nid, mid in micro_assignments.items():
        clusters.setdefault(mid, []).append(nid)

    # ── Merge small clusters ──
    changed = True
    while changed:
        changed = False
        small = [mid for mid, nodes in clusters.items() if len(nodes) < merge_thresh]
        if not small or len(clusters) <= 1:
            break

        for s_mid in small:
            if s_mid not in clusters:
                continue
            s_nodes = clusters[s_mid]
            best_target = _find_nearest_cluster(G, s_nodes, clusters, s_mid)
            if best_target is not None:
                clusters[best_target].extend(s_nodes)
                del clusters[s_mid]
                changed = True

    # ── Split oversized clusters ──
    new_clusters: Dict[int, List[str]] = {}
    next_id = 0
    for mid, nodes in sorted(clusters.items()):
        if len(nodes) <= max_size:
            new_clusters[next_id] = nodes
            next_id += 1
        else:
            # Recursively split with higher resolution
            sub_assignments = _recursive_split(G, set(nodes), max_size)
            for sub_nodes in sub_assignments:
                new_clusters[next_id] = sub_nodes
                next_id += 1

    # Rebuild node → micro_id mapping
    result: Dict[str, int] = {}
    for mid, nodes in new_clusters.items():
        for nid in nodes:
            result[nid] = mid

    return result


def _find_nearest_cluster(
    G: nx.MultiDiGraph,
    nodes: List[str],
    clusters: Dict[int, List[str]],
    exclude_mid: int,
) -> Optional[int]:
    """Find the cluster with the highest edge-weight sum to *nodes*."""
    node_set = set(nodes)
    scores: Counter = Counter()

    for nid in nodes:
        for _, neighbor, data in G.edges(nid, data=True):
            w = data.get("weight", 1.0)
            for mid, cnodes in clusters.items():
                if mid == exclude_mid:
                    continue
                if neighbor in set(cnodes):
                    scores[mid] += w
        # Also check incoming edges
        for predecessor in G.predecessors(nid):
            if predecessor in node_set:
                continue
            for mid, cnodes in clusters.items():
                if mid == exclude_mid:
                    continue
                if predecessor in set(cnodes):
                    for _, _, data in G.edges(predecessor, data=True):
                        scores[mid] += data.get("weight", 1.0)

    if scores:
        return scores.most_common(1)[0][0]
    # No edges found — just pick the largest cluster
    largest = max(
        ((mid, len(cnodes)) for mid, cnodes in clusters.items() if mid != exclude_mid),
        key=lambda x: x[1],
        default=None,
    )
    return largest[0] if largest else None


def _recursive_split(
    G: nx.MultiDiGraph,
    nodes: Set[str],
    max_size: int,
    resolution: float = 2.0,
) -> List[List[str]]:
    """Recursively split a set of nodes until all sub-clusters ≤ max_size.

    When Louvain cannot split further (common for dense C++ headers),
    falls back to file-path grouping so that nodes from the same source
    file stay together.  Within a single file, nodes are ordered by
    ``start_line`` to maintain declaration locality.
    """
    if len(nodes) <= max_size:
        return [list(nodes)]

    subgraph = G.subgraph(nodes).copy()
    G_undirected = _to_weighted_undirected(subgraph)

    if G_undirected.number_of_edges() == 0:
        return _file_aware_chunk(G, nodes, max_size)

    communities = nx.community.louvain_communities(
        G_undirected, weight="weight", resolution=resolution, seed=42,
    )

    if len(communities) <= 1:
        # Louvain couldn't split — fall back to file-aware grouping
        return _file_aware_chunk(G, nodes, max_size)

    result = []
    for members in communities:
        if len(members) <= max_size:
            result.append(list(members))
        else:
            result.extend(
                _recursive_split(G, members, max_size, resolution=resolution + 0.5)
            )
    return result


def _file_aware_chunk(
    G: nx.MultiDiGraph,
    nodes: Set[str],
    max_size: int,
) -> List[List[str]]:
    """Group *nodes* by file path, then chunk each file by declaration order.

    This produces semantically meaningful pages even when graph community
    detection fails (e.g. dense C++ headers where every symbol calls
    every other symbol).
    """
    from collections import defaultdict

    # Group by rel_path (or file_path as fallback)
    by_file: Dict[str, List[str]] = defaultdict(list)
    for nid in nodes:
        data = G.nodes.get(nid, {})
        fpath = data.get("rel_path") or data.get("file_path") or ""
        by_file[fpath].append(nid)

    # Sort each file's nodes by start_line for declaration locality
    for fpath in by_file:
        by_file[fpath].sort(
            key=lambda n: G.nodes.get(n, {}).get("start_line", 0)
        )

    # Pack file groups into chunks ≤ max_size
    result: List[List[str]] = []
    for fpath in sorted(by_file):
        file_nodes = by_file[fpath]
        if len(file_nodes) <= max_size:
            result.append(file_nodes)
        else:
            # Single file too large — split by line order
            for i in range(0, len(file_nodes), max_size):
                result.append(file_nodes[i:i + max_size])

    return result


def _file_internal_split(
    nodes: List[str],
    G: nx.MultiDiGraph,
    max_size: int = 50,
) -> List[List[str]]:
    """Split symbols within a single file by scope hierarchy.

    For header-only libraries (e.g. ``stb_image.h``, ``nlohmann/json.hpp``)
    where the entire codebase lives in one file, this produces meaningful
    sub-groups based on namespace/class scope.

    Strategy:
        1. If total nodes ≤ max_size → return single group (no split needed)
        2. Group by ``parent_symbol`` or ``namespace`` scope
        3. If scopes produce ≥ 2 groups → return scope groups
        4. Fallback: split by ``start_line`` order (preserves locality)
    """
    if len(nodes) <= max_size:
        return [list(nodes)]

    # Group by parent scope (namespace, class, struct)
    scopes: Dict[str, List[str]] = {}
    for nid in nodes:
        data = G.nodes.get(nid, {})
        scope = (
            data.get("parent_symbol", "")
            or data.get("namespace", "")
            or "__toplevel__"
        )
        scopes.setdefault(scope, []).append(nid)

    # If scopes produce reasonable groups, use them
    if len(scopes) > 1:
        result = []
        for scope_name in sorted(scopes):
            group = scopes[scope_name]
            if len(group) <= max_size:
                result.append(group)
            else:
                # Scope still too large — split by line number
                group.sort(
                    key=lambda n: G.nodes.get(n, {}).get("start_line", 0)
                )
                for i in range(0, len(group), max_size):
                    result.append(group[i:i + max_size])
        return result

    # Fallback: split by line number (preserves logical grouping)
    sorted_nodes = sorted(
        nodes, key=lambda n: G.nodes.get(n, {}).get("start_line", 0)
    )
    return [sorted_nodes[i:i + max_size]
            for i in range(0, len(sorted_nodes), max_size)]


def _merge_micro_to_cap(
    G: nx.MultiDiGraph,
    micro_assignments: Dict[str, int],
    max_pages: int,
) -> Dict[str, int]:
    """Merge the smallest micro-clusters until page count ≤ *max_pages*.

    Same merge-by-affinity approach as :func:`merge_macro_clusters`
    but applied at the per-section (micro) level.
    """
    # cluster_id → list of node_ids
    clusters: Dict[int, List[str]] = {}
    for nid, mid in micro_assignments.items():
        clusters.setdefault(mid, []).append(nid)

    if len(clusters) <= max_pages:
        return micro_assignments

    # Build node→cluster lookup
    node_to_micro: Dict[str, int] = dict(micro_assignments)

    # Pre-compute pairwise affinity
    affinity: Dict[Tuple[int, int], float] = {}
    for u, v, data in G.edges(data=True):
        cu = node_to_micro.get(u)
        cv = node_to_micro.get(v)
        if cu is not None and cv is not None and cu != cv:
            key = (min(cu, cv), max(cu, cv))
            affinity[key] = affinity.get(key, 0.0) + data.get("weight", 1.0)

    while len(clusters) > max_pages:
        smallest = min(clusters, key=lambda c: len(clusters[c]))

        best: Optional[int] = None
        best_w = -1.0
        for cid in clusters:
            if cid == smallest:
                continue
            key = (min(smallest, cid), max(smallest, cid))
            w = affinity.get(key, 0.0)
            if w > best_w or (w == best_w and best is not None and len(clusters[cid]) > len(clusters.get(best, []))):
                best_w = w
                best = cid

        if best is None:
            break

        clusters[best].extend(clusters.pop(smallest))
        for nid in clusters[best]:
            node_to_micro[nid] = best

        # Fold affinity entries
        updated: Dict[Tuple[int, int], float] = {}
        for (c1, c2), w in affinity.items():
            nc1 = best if c1 == smallest else c1
            nc2 = best if c2 == smallest else c2
            if nc1 == nc2:
                continue
            key = (min(nc1, nc2), max(nc1, nc2))
            updated[key] = updated.get(key, 0.0) + w
        affinity = updated

    # Renumber 0..N-1
    result: Dict[str, int] = {}
    for new_id, (_, nodes) in enumerate(
        sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    ):
        for nid in nodes:
            result[nid] = new_id

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5. Hub Re-Integration
# ═══════════════════════════════════════════════════════════════════════════

def reintegrate_hubs(
    G: nx.MultiDiGraph,
    hubs: Set[str],
    macro_assignments: Dict[str, int],
    majority_threshold: float = HUB_MAJORITY_THRESHOLD,
) -> Dict[str, str]:
    """Assign quarantined hubs to their majority cluster or global_core.

    For each hub, count edges (in + out) to each macro-cluster.
    If >60% go to one cluster, assign there. Otherwise → ``global_core``.

    Returns:
        ``{hub_node_id: cluster_assignment_str}``.
        Cluster-assigned hubs get ``str(macro_id)``, rest get ``"global_core"``.
    """
    hub_assignments: Dict[str, str] = {}

    for hub in hubs:
        cluster_edges: Counter = Counter()

        # Outgoing edges
        for _, neighbor in G.out_edges(hub):
            if neighbor in macro_assignments:
                cluster_edges[macro_assignments[neighbor]] += 1

        # Incoming edges
        for predecessor, _ in G.in_edges(hub):
            if predecessor in macro_assignments:
                cluster_edges[macro_assignments[predecessor]] += 1

        if not cluster_edges:
            hub_assignments[hub] = GLOBAL_CORE_LABEL
        else:
            total = sum(cluster_edges.values())
            top_cluster, top_count = cluster_edges.most_common(1)[0]
            if top_count > total * majority_threshold:
                hub_assignments[hub] = str(top_cluster)
            else:
                hub_assignments[hub] = GLOBAL_CORE_LABEL

    assigned = sum(1 for v in hub_assignments.values() if v != GLOBAL_CORE_LABEL)
    logger.info(
        "Hub re-integration: %d hubs — %d assigned to clusters, %d → global_core",
        len(hubs), assigned, len(hubs) - assigned,
    )
    return hub_assignments


# ═══════════════════════════════════════════════════════════════════════════
# 6. Persist Clusters to DB
# ═══════════════════════════════════════════════════════════════════════════

def persist_clusters(
    db,
    macro_assignments: Dict[str, int],
    micro_assignments: Dict[int, Dict[str, int]],
    hub_assignments: Dict[str, str],
) -> Dict[str, Any]:
    """Write all cluster assignments to the unified DB.

    Returns:
        Stats dict.
    """
    # Batch-build (node_id, macro, micro) tuples
    batch: List[Tuple[str, int, Optional[int]]] = []
    for node_id, macro_id in macro_assignments.items():
        micro_id = micro_assignments.get(macro_id, {}).get(node_id)
        batch.append((node_id, macro_id, micro_id))

    db.set_clusters_batch(batch)

    # Hub assignments
    for hub_id, assignment in hub_assignments.items():
        db.set_hub(hub_id, is_hub=True, assignment=assignment)

    db.conn.commit()

    stats = {
        "nodes_clustered": len(batch),
        "macro_clusters": len(set(macro_assignments.values())),
        "micro_clusters": sum(
            len(set(m.values())) for m in micro_assignments.values()
        ),
        "hubs_assigned": len(hub_assignments),
        "hubs_in_clusters": sum(
            1 for v in hub_assignments.values() if v != GLOBAL_CORE_LABEL
        ),
        "hubs_global_core": sum(
            1 for v in hub_assignments.values() if v == GLOBAL_CORE_LABEL
        ),
    }

    logger.info(
        "Persisted clusters: %d nodes → %d sections × %d pages, "
        "%d hubs (%d→cluster, %d→global_core)",
        stats["nodes_clustered"], stats["macro_clusters"],
        stats["micro_clusters"], stats["hubs_assigned"],
        stats["hubs_in_clusters"], stats["hubs_global_core"],
    )
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# 6b. Doc-Cluster Detection (Phase 7G)
# ═══════════════════════════════════════════════════════════════════════════

DOC_DOMINANT_THRESHOLD = 0.7  # >70% doc nodes → doc-dominant cluster


def detect_doc_clusters(
    cluster_map: Dict[int, Dict[int, List[str]]],
    db,
) -> Dict[int, float]:
    """Identify macro-clusters that are predominantly documentation.

    A cluster is "doc-dominant" when more than :data:`DOC_DOMINANT_THRESHOLD`
    of its nodes have ``is_doc=1`` in the database.

    Args:
        cluster_map: ``{macro_id: {micro_id: [node_ids]}}``.
        db: Unified DB handle (supports ``get_node(node_id)``).

    Returns:
        ``{macro_id: doc_ratio}`` for each doc-dominant cluster.
    """
    result: Dict[int, float] = {}

    for macro_id, micro_map in cluster_map.items():
        total = 0
        doc_count = 0
        for node_ids in micro_map.values():
            for nid in node_ids:
                total += 1
                node = db.get_node(nid)
                if node and node.get("is_doc"):
                    doc_count += 1

        if total > 0:
            ratio = doc_count / total
            if ratio > DOC_DOMINANT_THRESHOLD:
                result[macro_id] = ratio

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 7. Orchestrator — run full Phase 3 pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_phase3(
    db,
    G: nx.MultiDiGraph,
    hubs: Optional[Set[str]] = None,
    macro_resolution: Optional[float] = None,
    micro_resolution: float = 1.5,
    page_sizing_rules: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Execute the complete Phase 3 clustering pipeline.

    Steps:
        0. Project graph to architectural-only nodes (collapse methods → parent class)
        1. Macro-cluster (Louvain, hub-free, weighted, auto-γ)
        1b. Merge cap — consolidate to ≤ max_sections (log₁₀-based target)
        2. Micro-cluster (sub-Louvain per macro-cluster)
        3. Dynamic page sizing (merge small, split oversized)
        4. Hub re-integration (majority-vote)
        5. Propagate assignments back to child nodes
        6. Persist to unified DB

    Args:
        db: ``UnifiedWikiDB`` instance (populated by Phases 1-2).
        G: In-memory ``nx.MultiDiGraph`` (weighted from Phase 2).
        hubs: Set of hub node IDs. If None, reads from DB.
        macro_resolution: Override Louvain γ for macro pass.
        micro_resolution: Louvain γ for micro pass (default 1.5).
        page_sizing_rules: Override MICRO_CLUSTER_RULES.

    Returns:
        Combined stats dict.
    """
    results: Dict[str, Any] = {}

    # Get hubs from DB if not provided
    if hubs is None:
        hubs = _load_hubs_from_db(db)

    # Step 0: Architectural projection — collapse methods/fields into
    # their parent class so Louvain sees only top-level structure.
    P = architectural_projection(G)

    # Adjust hub set to only include nodes that survived projection
    projected_hubs = hubs & set(P.nodes())

    results["projection"] = {
        "original_nodes": G.number_of_nodes(),
        "projected_nodes": P.number_of_nodes(),
    }

    # Adaptive page sizing: scale max_page_size with repo complexity
    if page_sizing_rules is None:
        adaptive_max = _adaptive_max_page_size(P.number_of_nodes())
        page_sizing_rules = {
            **MICRO_CLUSTER_RULES,
            "max_page_size": adaptive_max,
        }
        logger.info(
            "Adaptive page sizing: %d projected nodes → max_page_size=%d",
            P.number_of_nodes(), adaptive_max,
        )

    # Step 1: Macro-clustering (on projected graph)
    macro_assignments = macro_cluster(P, projected_hubs, resolution=macro_resolution)

    # Step 1b: Merge cap — ensure reasonable section count for the repo size.
    cap = _max_sections(P.number_of_nodes())
    macro_assignments = merge_macro_clusters(P, macro_assignments, cap)

    results["macro"] = {
        "cluster_count": len(set(macro_assignments.values())) if macro_assignments else 0,
        "nodes_assigned": len(macro_assignments),
    }

    # Step 2: Micro-clustering (on projected graph)
    micro_assignments = micro_cluster_all(P, macro_assignments, resolution=micro_resolution)

    # Step 3: Dynamic page sizing (on projected graph)
    for macro_id in list(micro_assignments.keys()):
        micro_assignments[macro_id] = apply_page_sizing(
            P, macro_id, micro_assignments[macro_id], rules=page_sizing_rules,
        )

    # Step 3b: Per-section page cap — merge tiny pages until under cap
    total_before_cap = sum(len(set(m.values())) for m in micro_assignments.values())
    for macro_id in list(micro_assignments.keys()):
        section_nodes = set(micro_assignments[macro_id].keys())
        page_cap = _max_pages_per_section(len(section_nodes))
        n_pages = len(set(micro_assignments[macro_id].values()))
        if n_pages > page_cap:
            micro_assignments[macro_id] = _merge_micro_to_cap(
                P, micro_assignments[macro_id], page_cap,
            )

    total_after_cap = sum(len(set(m.values())) for m in micro_assignments.values())
    if total_before_cap != total_after_cap:
        logger.info(
            "Per-section page cap: %d → %d total pages",
            total_before_cap, total_after_cap,
        )

    results["micro"] = {
        "total_pages": sum(
            len(set(m.values())) for m in micro_assignments.values()
        ),
    }

    # Step 4: Hub re-integration (on original graph — hubs need full edge info)
    hub_assignments = reintegrate_hubs(G, projected_hubs, macro_assignments)
    results["hubs"] = {
        "total": len(hub_assignments),
        "assigned_to_cluster": sum(
            1 for v in hub_assignments.values() if v != GLOBAL_CORE_LABEL
        ),
        "global_core": sum(
            1 for v in hub_assignments.values() if v == GLOBAL_CORE_LABEL
        ),
    }

    # Step 5: Propagate assignments to child nodes.
    # Children inherit their parent's (macro, micro) assignment.
    propagated = 0
    for nid, data in G.nodes(data=True):
        if nid in macro_assignments:
            continue  # already assigned (architectural node)
        if nid in projected_hubs:
            continue  # hub — handled separately

        stype = (data.get("symbol_type") or "").lower()
        if stype not in _CHILD_SYMBOL_TYPES:
            continue

        # Resolve parent via node_id convention
        parent_nid = _resolve_parent(nid, data, set(macro_assignments.keys()))
        if parent_nid and parent_nid in macro_assignments:
            mid = macro_assignments[parent_nid]
            macro_assignments[nid] = mid
            # Inherit micro assignment from parent
            micro_map = micro_assignments.get(mid, {})
            if parent_nid in micro_map:
                micro_map[nid] = micro_map[parent_nid]
            propagated += 1

    logger.info("Propagated cluster assignments to %d child nodes", propagated)

    # Step 6: Persist
    results["persistence"] = persist_clusters(
        db, macro_assignments, micro_assignments, hub_assignments,
    )

    # Store metadata
    db.set_meta("phase3_completed", True)
    db.set_meta("phase3_stats", results)

    logger.info(
        "Phase 3 complete: %d sections, %d pages, %d hubs reintegrated",
        results["macro"]["cluster_count"],
        results["micro"]["total_pages"],
        results["hubs"]["total"],
    )
    return results


def _resolve_parent(
    node_id: str,
    node_data: Dict[str, Any],
    arch_node_ids: Set[str],
) -> Optional[str]:
    """Resolve a child node to its parent architectural node.

    Uses the ``lang::file::Parent.child`` node_id convention first,
    then falls back to the ``parent_symbol`` attribute.
    """
    parts = node_id.split("::", 2)
    if len(parts) == 3:
        lang, fname, qualified = parts
        if "." in qualified:
            parent_name = qualified.rsplit(".", 1)[0]
            parent_nid = f"{lang}::{fname}::{parent_name}"
            if parent_nid in arch_node_ids:
                return parent_nid

        psym = node_data.get("parent_symbol")
        if psym:
            candidate = f"{lang}::{fname}::{psym}"
            if candidate in arch_node_ids:
                return candidate

    return None


def _load_hubs_from_db(db) -> Set[str]:
    """Load hub node IDs from the unified DB."""
    rows = db.conn.execute(
        "SELECT node_id FROM repo_nodes WHERE is_hub = 1"
    ).fetchall()
    return {row["node_id"] for row in rows}
