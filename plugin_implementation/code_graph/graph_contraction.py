"""Graph contraction: drop noise nodes (variable / parameter / field) and
rewire their edges onto the containing arch node.

Empirically the AST graph is dominated by intra-method "noise plumbing":
on a 790-node Python repo, 71% of nodes and 65% of edges had at least one
noise endpoint, and most of those edges collapsed to self-loops once the
endpoints were rewritten onto their containing arch — proof they carried
no architectural information beyond what the arch-to-arch graph already
encodes.

This module performs that collapse on the in-memory ``nx.MultiDiGraph``
in-place, before clustering. Edges that survived rewriting carry a
``via=<orig_name>@L<line>`` entry in their ``annotations`` dict so the
context provided by the dropped intermediate node is preserved on the
edge itself.

Resolver handles three observed ``parent_symbol`` formats produced by
the parsers:

- **Dotted** (``module.Class.method``) — Python, Java, C#, TypeScript,
  JavaScript, and C++ (after C++ parser's ``_normalize_full_name``
  normalises ``::`` to ``.``).
- **Bare** (``ContainingType``) — Go, Rust. The containing type lives
  in the same module as the noise node; the lookup combines the noise
  node's own module (from its ``node_id``) with the bare parent name.
- **Empty** — Go package-level variables. Promoted to the module node.

Feature flag: ``FeatureFlags.contract_noise_nodes`` (env
``DEEPWIKI_CONTRACT_NOISE``), default on.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

#: Closed set of symbol types whose nodes carry no architectural information
#: beyond what their containing arch already encodes. The contractor drops
#: these and rewires their edges onto the containing arch (with the dropped
#: node's identity preserved as a ``via=`` annotation).
#:
#: Derived from ``graph_builder._get_type_priority``:
#: - priority 1 (``parameter``, ``variable``, ``local_variable``,
#:   ``argument``) → noise (sub-method scope).
#: - priority 2 (``field``, ``property``) → noise (data, not code).
#: - priority 2 (``constructor``) → **arch**, NOT noise — it is a callable
#:   with source code and a wiki-page-worthy unit.
#: - priority ≥ 3 (``method``, ``function``, ``class``, ``interface``,
#:   ``trait``, ``protocol``, ``struct``, ``enum``, ``record``,
#:   ``data_class``, ``object``, ``module``, ``namespace``, ``constant``,
#:   ``type_alias``, ``annotation``, ``decorator``, ``macro``) → arch.
#: - All doc types (``text_chunk``, ``markdown_document``, ``yaml_document``,
#:   ``infrastructure_document``, ...) → not noise; stay as terminals.
#: - Unknown types → not in this set, treated as terminals (safe default —
#:   contraction only ever DROPS nodes from this list, never others).
#:
#: Keep in sync with ``graph_builder._get_type_priority``: new noise types
#: must be added here explicitly. We intentionally don't auto-derive from
#: priority (priority is for disambiguation, a different concern; coupling
#: would make a priority tweak silently change contraction behaviour).
NOISE_TYPES: Set[str] = frozenset({
    "parameter",        # priority 1 — function/method parameter
    "variable",         # priority 1 — generic variable node
    "local_variable",   # priority 1 — explicit local-scope variable
    "argument",         # priority 1 — call-site argument node
    "field",            # priority 2 — struct/class data member
    "property",         # priority 2 — getter/setter-backed field
})


def _coerce_annotations(value: Any) -> dict:
    """Normalise the ``annotations`` edge attribute.

    In live graph_builder output it is a ``dict``; after a round-trip
    through the unified DB it is a JSON string. Both shapes need to
    survive contraction without raising. Anything else falls back to
    an empty dict.
    """
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _resolve_arch_parent(
    noise_nid: str,
    parent_symbol: str,
    by_pkg_full: Dict[Tuple[str, str], str],
) -> str:
    """Resolve a noise node to the ``node_id`` of its containing arch.

    Returns ``""`` if the parent cannot be located in ``by_pkg_full``.
    Caller decides what to do with unresolved nodes (current policy:
    leave them in the graph, flag in metrics).
    """
    own_parts = noise_nid.split("::", 2)
    own_module = own_parts[1] if len(own_parts) == 3 else ""

    if not parent_symbol:
        # Package-level vars (Go's bare `var x = ...` outside any function):
        # promote to the module node so the edge still has a real anchor.
        return by_pkg_full.get((own_module, own_module), "")

    if "." in parent_symbol:
        # Dotted family. Try (qualified module, qualified rest) first;
        # fall back to (own module, full parent) which handles cases
        # where the parser emitted the parent already-qualified inside
        # the same module.
        module_stem, rest = parent_symbol.split(".", 1)
        return (
            by_pkg_full.get((module_stem, rest))
            or by_pkg_full.get((own_module, parent_symbol))
            or ""
        )

    # Bare family — Go/Rust. The containing type lives in the same
    # module/package as the noise node by construction.
    return by_pkg_full.get((own_module, parent_symbol), "")


def _walk_to_arch(
    nid: str,
    nodes_data: Dict[str, dict],
    by_pkg_full: Dict[Tuple[str, str], str],
    depth: int = 0,
) -> str:
    """Follow ``parent_symbol`` chains until we hit an arch node.

    Caps recursion at depth 5 — beyond that we treat the chain as
    unresolvable rather than risk an infinite loop on circular data.
    """
    if depth > 5:
        return ""
    data = nodes_data.get(nid)
    if not data:
        return ""
    sym_type = data.get("symbol_type", "")
    if sym_type not in NOISE_TYPES:
        # Arch (class/method/struct/trait/type_alias/...) or doc node or
        # unknown — terminal in the walk. The current node IS the contraction
        # target; caller will rewrite edges onto it.
        return nid
    parent_symbol = (data.get("parent_symbol") or "").strip()
    target = _resolve_arch_parent(nid, parent_symbol, by_pkg_full)
    if not target:
        return ""
    target_type = nodes_data.get(target, {}).get("symbol_type", "")
    if target_type in NOISE_TYPES:
        return _walk_to_arch(target, nodes_data, by_pkg_full, depth + 1)
    return target


def contract_graph_inplace(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    """Drop noise nodes from ``graph`` and rewire their edges onto the
    containing arch node, preserving context as ``via=<name>@L<line>``
    annotations.

    Mutates ``graph`` in place. Returns metrics:

    - ``nodes_removed``: count of noise nodes successfully contracted.
    - ``edges_rewritten``: count of edges that had at least one endpoint
      rewritten (includes self-loops that were dropped after rewiring).
    - ``self_loops_dropped``: subset of ``edges_rewritten`` that became
      self-loops post-contraction (carried no cross-arch information).
    - ``unresolved``: noise nodes whose ``parent_symbol`` could not be
      resolved to a graph node — these are left in the graph and the
      count is logged per-language for monitoring.
    - ``by_language``: ``{lang: contracted_count}`` so unfamiliar
      languages with format quirks can be spotted in production logs.
    """
    # Snapshot node attrs so we can read them while mutating the graph.
    nodes_data: Dict[str, dict] = {nid: dict(d) for nid, d in graph.nodes(data=True)}

    # Build (module, full_name) → node_id index for parent_symbol lookups.
    by_pkg_full: Dict[Tuple[str, str], str] = {}
    for nid in nodes_data:
        parts = nid.split("::", 2)
        if len(parts) == 3:
            by_pkg_full[(parts[1], parts[2])] = nid

    # Resolve noise → arch.
    contract_map: Dict[str, str] = {}
    by_language: Dict[str, int] = defaultdict(int)
    unresolved: Dict[str, int] = defaultdict(int)

    for nid, data in nodes_data.items():
        if data.get("symbol_type", "") not in NOISE_TYPES:
            continue
        target = _walk_to_arch(nid, nodes_data, by_pkg_full)
        language = data.get("language", "?") or "?"
        if target and target != nid:
            contract_map[nid] = target
            by_language[language] += 1
        else:
            unresolved[language] += 1

    if not contract_map:
        logger.info("Contraction: no noise nodes to contract")
        return {
            "nodes_removed": 0,
            "edges_rewritten": 0,
            "self_loops_dropped": 0,
            "unresolved": sum(unresolved.values()),
            "by_language": {},
        }

    # Rewire edges that touch noise endpoints.
    edges_to_remove: List[Tuple[str, str, Any]] = []
    edges_to_add: List[Tuple[str, str, dict]] = []
    edges_rewritten = 0
    self_loops_dropped = 0

    for u, v, key, data in list(graph.edges(keys=True, data=True)):
        new_u = contract_map.get(u, u)
        new_v = contract_map.get(v, v)
        if new_u == u and new_v == v:
            continue
        edges_rewritten += 1
        edges_to_remove.append((u, v, key))
        if new_u == new_v:
            # Pure noise plumbing — method touching its own parameter, etc.
            self_loops_dropped += 1
            continue
        # Carry forward the original endpoint's identity as a ``via=`` annotation
        # so the context that was on the dropped node still lives somewhere.
        new_data = dict(data)
        anns = _coerce_annotations(new_data.get("annotations"))
        via_list = anns.get("via")
        if via_list is None:
            via_list = []
        elif not isinstance(via_list, list):
            via_list = [str(via_list)]
        if u != new_u:
            label = nodes_data[u].get("symbol_name") or u.rsplit("::", 1)[-1]
            line = nodes_data[u].get("start_line") or 0
            via_list.append(f"src={label}@L{line}")
        if v != new_v:
            label = nodes_data[v].get("symbol_name") or v.rsplit("::", 1)[-1]
            line = nodes_data[v].get("start_line") or 0
            via_list.append(f"tgt={label}@L{line}")
        anns["via"] = via_list
        new_data["annotations"] = anns
        edges_to_add.append((new_u, new_v, new_data))

    # Apply edge changes.
    for u, v, key in edges_to_remove:
        if graph.has_edge(u, v, key=key):
            graph.remove_edge(u, v, key=key)

    # Track existing (u, v, rel_type) tuples to merge ``via`` lists across
    # duplicate-edge collisions instead of double-adding.
    existing: Dict[Tuple[str, str, str], Tuple[Any, dict]] = {}
    for u, v, key, data in graph.edges(keys=True, data=True):
        existing[(u, v, str(data.get("relationship_type", "")))] = (key, data)

    for new_u, new_v, new_data in edges_to_add:
        rel = str(new_data.get("relationship_type", ""))
        seen = existing.get((new_u, new_v, rel))
        if seen is not None:
            # Merge ``via`` annotations into the existing edge instead of
            # duplicating; keeps edge counts honest.
            _, existing_data = seen
            anns = _coerce_annotations(existing_data.get("annotations"))
            existing_via = anns.get("via") or []
            if not isinstance(existing_via, list):
                existing_via = [str(existing_via)]
            new_via = _coerce_annotations(new_data.get("annotations")).get("via") or []
            if not isinstance(new_via, list):
                new_via = [str(new_via)]
            merged = list(dict.fromkeys(existing_via + new_via))  # dedup, keep order
            anns["via"] = merged
            existing_data["annotations"] = anns
            continue
        graph.add_edge(new_u, new_v, **new_data)
        # Re-key — graph.add_edge auto-assigns a key; re-fetch isn't strictly
        # needed for ``existing`` because subsequent collisions on the same
        # (u, v, rel) will find this entry on the next iteration. Make sure
        # by re-reading the latest key for this triple.
        for k, d in graph[new_u][new_v].items():
            if str(d.get("relationship_type", "")) == rel:
                existing[(new_u, new_v, rel)] = (k, d)
                break

    # Drop the now-orphaned noise nodes.
    nodes_removed = 0
    for nid in contract_map:
        if graph.has_node(nid):
            graph.remove_node(nid)
            nodes_removed += 1

    # Per-language summary line. Format matches the existing logging style
    # ("X edges, Y resolved, ..."), so dashboards/grep stay grep-friendly.
    by_lang_str = ", ".join(
        f"{lang}={n}" for lang, n in sorted(by_language.items())
    ) or "—"
    unresolved_str = ", ".join(
        f"{lang}={n}" for lang, n in sorted(unresolved.items())
    ) or "—"
    logger.info(
        "Contraction: %d nodes removed [%s], %d edges rewritten "
        "(%d self-loops dropped), %d unresolved [%s]",
        nodes_removed, by_lang_str, edges_rewritten, self_loops_dropped,
        sum(unresolved.values()), unresolved_str,
    )

    return {
        "nodes_removed": nodes_removed,
        "edges_rewritten": edges_rewritten,
        "self_loops_dropped": self_loops_dropped,
        "unresolved": sum(unresolved.values()),
        "by_language": dict(by_language),
    }


__all__ = ["contract_graph_inplace", "NOISE_TYPES"]
