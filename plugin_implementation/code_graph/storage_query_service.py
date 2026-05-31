"""UnifiedWikiDB-backed symbol search and relationship traversal service."""

from __future__ import annotations

import logging
from collections import deque
from fnmatch import fnmatch
from typing import Any, Iterable, Optional

from ..constants import classify_symbol_layer
from .graph_query_builder import EDGE_TYPE_ALIASES
from .graph_query_service import RelationshipResult, SymbolResult, _edge_via, _edge_confidence, _merge_edge_provenance

logger = logging.getLogger(__name__)


class StorageQueryService:
    """Query-service facade over ``UnifiedWikiDB``.

    This intentionally mirrors the small surface used by research tools:
    ``get_node``, ``resolve_symbol``, ``search``, ``get_relationships``,
    ``resolve_and_traverse`` and ``query``. It keeps ask/research on the
    unified DB path when no legacy NetworkX graph artifact exists.
    """

    def __init__(self, storage: Any):
        self.storage = storage
        self.graph = None

    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        if not node_id:
            return None
        try:
            return self.storage.get_node(node_id)
        except Exception:
            return None

    def resolve_symbol(self, symbol_name: str, file_path: str = '', language: str = '') -> Optional[str]:
        name = (symbol_name or '').strip()
        if not name:
            return None
        if self.get_node(name):
            return name

        rows = self._name_rows(name, exact=True, limit=15)
        best = self._select_best(rows, file_path=file_path, language=language)
        if best:
            return best.get('node_id')

        simple = name.rsplit('.', 1)[-1].rsplit('::', 1)[-1]
        if simple != name:
            rows = self._name_rows(simple, exact=True, limit=15)
            best = self._select_best(rows, file_path=file_path, language=language)
            if best:
                return best.get('node_id')

        rows = self._name_rows(name, exact=False, limit=15)
        best = self._select_best(rows, file_path=file_path, language=language)
        if best:
            return best.get('node_id')

        try:
            rows = self.storage.search_fts5(query=name, limit=5)
        except Exception:
            rows = []
        best = self._select_best(rows, file_path=file_path, language=language)
        return best.get('node_id') if best else None

    def search(
        self,
        query: str,
        k: int = 20,
        symbol_types: Optional[frozenset[str]] = None,
        exclude_types: Optional[frozenset[str]] = None,
        layer: Optional[str] = None,
        path_prefix: Optional[str] = None,
    ) -> list[SymbolResult]:
        if not query or not query.strip():
            return []

        try:
            rows = self.storage.search_fts5(
                query=query,
                path_prefix=path_prefix if path_prefix and '*' not in path_prefix else None,
                symbol_types=sorted(symbol_types) if symbol_types else None,
                limit=max(k * 3, k),
            )
        except Exception as exc:
            logger.debug("Unified DB FTS search failed for %r: %s", query, exc)
            rows = []

        if not rows:
            rows = self._name_rows(query, exact=False, limit=max(k * 3, k))

        results: list[SymbolResult] = []
        for row in rows:
            result = self._row_to_result(row, match_source='storage_fts')
            if not self._matches_filters(result, symbol_types, exclude_types, layer, path_prefix):
                continue
            results.append(result)
            if len(results) >= k:
                break

        connection_counts = self._connection_counts(result.node_id for result in results)
        for result in results:
            result.connections = connection_counts.get(result.node_id, 0)
        return results

    def get_relationships(
        self,
        node_id: str,
        direction: str = 'both',
        max_depth: int = 2,
        max_results: int = 50,
    ) -> list[RelationshipResult]:
        if not node_id or not self.get_node(node_id):
            return []

        results: list[RelationshipResult] = []
        visited = {node_id}
        seen_edges: dict[tuple[str, str, str], RelationshipResult] = {}
        frontier: deque[tuple[str, int]] = deque([(node_id, 0)])

        while frontier and len(results) < max_results:
            current, depth = frontier.popleft()
            if depth >= max_depth:
                continue

            edges: list[tuple[str, str, str, str, dict[str, Any]]] = []
            if direction in ('outgoing', 'out', 'both'):
                for edge in self.storage.get_edges_from(current):
                    target = edge.get('target_id') or ''
                    if target:
                        edges.append((current, target, self._edge_type(edge), target, edge))
            if direction in ('incoming', 'in', 'both'):
                for edge in self.storage.get_edges_to(current):
                    source = edge.get('source_id') or ''
                    if source:
                        edges.append((source, current, self._edge_type(edge), source, edge))

            node_rows = self._nodes_by_id({src for src, *_ in edges} | {tgt for _, tgt, *_ in edges})
            for source, target, rel_type, other, edge in edges:
                if len(results) >= max_results:
                    break
                edge_key = (source, target, rel_type)
                if edge_key in seen_edges:
                    # Parallel edge: merge its provenance into the first result
                    # so distinct via/confidence are not silently dropped.
                    _merge_edge_provenance(seen_edges[edge_key], edge)
                    continue

                source_row = node_rows.get(source, {})
                target_row = node_rows.get(target, {})
                result = RelationshipResult(
                    source_name=source_row.get('symbol_name') or source,
                    target_name=target_row.get('symbol_name') or target,
                    relationship_type=rel_type,
                    source_type=(source_row.get('symbol_type') or '').lower(),
                    target_type=(target_row.get('symbol_type') or '').lower(),
                    hop_distance=depth + 1,
                    edge_class=str(edge.get('edge_class', '') or ''),
                    confidence=_edge_confidence(edge),
                    via=_edge_via(edge),
                )
                results.append(result)
                seen_edges[edge_key] = result

                if other not in visited:
                    visited.add(other)
                    frontier.append((other, depth + 1))

        results.sort(key=lambda rel: rel.hop_distance)
        return results

    def resolve_and_traverse(
        self,
        symbol_name: str,
        direction: str = 'both',
        max_depth: int = 2,
        max_results: int = 50,
        file_path: str = '',
        language: str = '',
    ) -> tuple[Optional[str], list[RelationshipResult]]:
        node_id = self.resolve_symbol(symbol_name, file_path=file_path, language=language)
        if not node_id:
            return None, []
        return node_id, self.get_relationships(
            node_id,
            direction=direction,
            max_depth=max_depth,
            max_results=max_results,
        )

    def query(self, expression: str) -> list[SymbolResult]:
        from .jql_parser import parse_jql

        jql = parse_jql(expression)
        if jql.is_empty:
            return []

        results = self._execute_index_clauses(jql)
        results = self._apply_post_filters(results, jql)
        return results[:jql.limit]

    def _execute_index_clauses(self, jql: Any) -> list[SymbolResult]:
        symbol_types = frozenset(jql.type_values) if jql.type_values else None
        path_prefix = jql.file_value
        fetch_k = max(jql.limit * 5, 20)

        if jql.text_value:
            results = self.search(
                jql.text_value,
                k=fetch_k,
                symbol_types=symbol_types,
                layer=jql.layer_value,
                path_prefix=path_prefix,
            )
            if jql.name_value:
                name_filter = jql.name_value.lower()
                results = [r for r in results if name_filter in r.symbol_name.lower()]
            return results

        if jql.name_value:
            rows = self._name_rows(jql.name_value, exact='*' not in jql.name_value and '?' not in jql.name_value, limit=fetch_k)
            return [r for r in (self._row_to_result(row) for row in rows)
                    if self._matches_filters(r, symbol_types, None, jql.layer_value, path_prefix)]

        rows = self._scan_rows(limit=fetch_k, symbol_types=symbol_types, path_prefix=path_prefix)
        return [r for r in (self._row_to_result(row) for row in rows)
                if self._matches_filters(r, symbol_types, None, jql.layer_value, path_prefix)]

    def _apply_post_filters(self, results: list[SymbolResult], jql: Any) -> list[SymbolResult]:
        if jql.related_value:
            results = self._filter_related(
                results,
                related_name=jql.related_value,
                direction=jql.direction_value,
                edge_types=jql.has_rel_values,
            )
        elif jql.has_rel_values:
            results = self._filter_has_relationship(results, jql.has_rel_values)

        if jql.connections_clause:
            connection_counts = self._connection_counts(result.node_id for result in results)
            for result in results:
                result.connections = connection_counts.get(result.node_id, 0)
            results = [r for r in results if jql.connections_clause.matches_numeric(r.connections)]

        return results

    def _filter_related(
        self,
        results: list[SymbolResult],
        related_name: str,
        direction: str = 'both',
        edge_types: Optional[list[str]] = None,
        max_depth: int = 3,
    ) -> list[SymbolResult]:
        anchor = self.resolve_symbol(related_name)
        if not anchor:
            return []

        allowed = self._normalise_edge_types(edge_types) if edge_types else None
        reachable: set[str] = set()
        visited = {anchor}
        frontier: deque[tuple[str, int]] = deque([(anchor, 0)])

        while frontier:
            current, depth = frontier.popleft()
            if depth >= max_depth:
                continue
            edges: list[tuple[str, str]] = []
            if direction in ('outgoing', 'out', 'both'):
                edges.extend((self._edge_type(edge), edge.get('target_id') or '') for edge in self.storage.get_edges_from(current))
            if direction in ('incoming', 'in', 'both'):
                edges.extend((self._edge_type(edge), edge.get('source_id') or '') for edge in self.storage.get_edges_to(current))

            for rel_type, other in edges:
                if not other:
                    continue
                if allowed is not None and rel_type not in allowed:
                    continue
                reachable.add(other)
                if other not in visited:
                    visited.add(other)
                    frontier.append((other, depth + 1))

        if not results:
            rows = self._nodes_by_id(reachable).values()
            materialized = [self._row_to_result(row) for row in rows]
            connection_counts = self._connection_counts(result.node_id for result in materialized)
            for result in materialized:
                result.connections = connection_counts.get(result.node_id, 0)
            materialized.sort(key=lambda item: (-item.connections, item.symbol_name))
            return materialized
        return [result for result in results if result.node_id in reachable]

    def _filter_has_relationship(self, results: list[SymbolResult], edge_types: list[str]) -> list[SymbolResult]:
        allowed = self._normalise_edge_types(edge_types)
        kept: list[SymbolResult] = []
        for result in results:
            rels = self.storage.get_edges_from(result.node_id) + self.storage.get_edges_to(result.node_id)
            if any(self._edge_type(edge) in allowed for edge in rels):
                kept.append(result)
        return kept

    def _name_rows(self, name: str, exact: bool, limit: int) -> list[dict[str, Any]]:
        if not name:
            return []
        operator = '=' if exact else 'LIKE'
        value = name.lower() if exact else f'%{name.lower()}%'
        rows = self.storage.conn.execute(
            f"SELECT * FROM repo_nodes WHERE LOWER(symbol_name) {operator} ? LIMIT ?",
            (value, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def _scan_rows(
        self,
        limit: int,
        symbol_types: Optional[frozenset[str]] = None,
        path_prefix: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        conditions = []
        params: list[Any] = []
        if symbol_types:
            placeholders = ','.join('?' for _ in symbol_types)
            conditions.append(f"symbol_type IN ({placeholders})")
            params.extend(sorted(symbol_types))
        if path_prefix and '*' not in path_prefix and '?' not in path_prefix:
            conditions.append("rel_path GLOB ?")
            params.append(path_prefix.rstrip('/') + '/*')
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ''
        rows = self.storage.conn.execute(
            f"SELECT * FROM repo_nodes {where} LIMIT ?",
            [*params, limit],
        ).fetchall()
        return [dict(row) for row in rows]

    def _nodes_by_id(self, node_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        ids = [node_id for node_id in node_ids if node_id]
        if not ids:
            return {}
        try:
            rows = self.storage.get_nodes_by_ids(ids)
        except Exception:
            rows = [self.storage.get_node(node_id) for node_id in ids]
        return {row.get('node_id'): row for row in rows if row}

    def _connection_count(self, node_id: str) -> int:
        if not node_id:
            return 0
        return len(self.storage.get_edges_from(node_id)) + len(self.storage.get_edges_to(node_id))

    def _connection_counts(self, node_ids: Iterable[str]) -> dict[str, int]:
        ids = list(dict.fromkeys(node_id for node_id in node_ids if node_id))
        if not ids:
            return {}

        placeholders = ','.join('?' for _ in ids)
        try:
            rows = self.storage.conn.execute(
                f"""
                SELECT node_id, SUM(connection_count) AS connection_count
                FROM (
                    SELECT source_id AS node_id, COUNT(*) AS connection_count
                    FROM repo_edges
                    WHERE source_id IN ({placeholders})
                    GROUP BY source_id
                    UNION ALL
                    SELECT target_id AS node_id, COUNT(*) AS connection_count
                    FROM repo_edges
                    WHERE target_id IN ({placeholders})
                    GROUP BY target_id
                )
                GROUP BY node_id
                """,
                [*ids, *ids],
            ).fetchall()
            counts = {node_id: 0 for node_id in ids}
            for row in rows:
                values = dict(row)
                counts[values.get('node_id') or ''] = int(values.get('connection_count') or 0)
            return counts
        except Exception:
            return {node_id: self._connection_count(node_id) for node_id in ids}

    @staticmethod
    def _edge_type(edge: dict[str, Any]) -> str:
        return str(edge.get('rel_type') or edge.get('relationship_type') or edge.get('type') or 'related').lower()

    @staticmethod
    def _normalise_edge_types(edge_types: Optional[list[str]]) -> frozenset[str]:
        expanded: set[str] = set()
        for edge_type in edge_types or []:
            lowered = edge_type.lower()
            expanded.add(lowered)
            if lowered in EDGE_TYPE_ALIASES:
                expanded.add(EDGE_TYPE_ALIASES[lowered])
            for alias, canonical in EDGE_TYPE_ALIASES.items():
                if canonical == lowered:
                    expanded.add(alias)
        return frozenset(expanded)

    @staticmethod
    def _matches_path(path: str, pattern: str) -> bool:
        if not path or not pattern:
            return False
        if '*' in pattern or '?' in pattern:
            return fnmatch(path, pattern)
        prefix = pattern.rstrip('/')
        return path == prefix or path.startswith(prefix + '/')

    def _matches_filters(
        self,
        result: SymbolResult,
        symbol_types: Optional[frozenset[str]],
        exclude_types: Optional[frozenset[str]],
        layer: Optional[str],
        path_prefix: Optional[str],
    ) -> bool:
        if symbol_types and result.symbol_type not in symbol_types:
            return False
        if exclude_types and result.symbol_type in exclude_types:
            return False
        if layer and result.layer != layer:
            return False
        if path_prefix and not self._matches_path(result.rel_path, path_prefix):
            return False
        return True

    @staticmethod
    def _select_best(
        rows: list[dict[str, Any]],
        file_path: str = '',
        language: str = '',
    ) -> Optional[dict[str, Any]]:
        if not rows:
            return None
        if file_path or language:
            for row in rows:
                rel_path = row.get('rel_path') or ''
                row_language = (row.get('language') or '').lower()
                if file_path and rel_path == file_path:
                    return row
                if language and row_language == language.lower():
                    return row

        def key(row: dict[str, Any]) -> tuple[int, int]:
            is_arch = int(row.get('is_architectural') or 0)
            try:
                span = int(row.get('end_line') or 0) - int(row.get('start_line') or 0)
            except (TypeError, ValueError):
                span = 0
            return is_arch, span

        return max(rows, key=key)

    @staticmethod
    def _row_to_result(row: dict[str, Any], match_source: str = 'storage') -> SymbolResult:
        symbol_type = (row.get('symbol_type') or '').lower()
        rel_path = row.get('rel_path') or ''
        layer = row.get('layer') or classify_symbol_layer(
            symbol_type=symbol_type,
            symbol_name=row.get('symbol_name') or '',
            parent_symbol=row.get('parent_symbol') or '',
            file_path=rel_path,
            docstring=row.get('docstring') or '',
        )
        return SymbolResult(
            node_id=row.get('node_id') or '',
            symbol_name=row.get('symbol_name') or '',
            symbol_type=symbol_type,
            layer=layer,
            file_path=rel_path,
            rel_path=rel_path,
            connections=0,
            score=StorageQueryService._score_from_row(row),
            match_source=match_source,
            docstring=row.get('docstring') or '',
        )

    @staticmethod
    def _score_from_row(row: dict[str, Any]) -> float:
        score = row.get('score_norm')
        if score is None:
            score = row.get('fts_rank')
        if score is None:
            return 0.0
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    def stats(self) -> dict[str, Any]:
        return {
            'graph_nodes': self.storage.node_count(),
            'graph_edges': self.storage.edge_count(),
            'fts5_available': True,
            'backend': 'unified_db',
        }