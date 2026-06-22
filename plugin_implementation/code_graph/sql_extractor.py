"""SQL / DDL graph extractor (roadmap C1/§3.1).

SQL files were previously dropped by the indexer (no ``.sql`` / ``.ddl`` entry
in ``DOCUMENTATION_EXTENSIONS``). This module gives database-backed services a
first-class place in the unified graph by parsing DDL into typed nodes and
edges — without an LLM and without a heavyweight dependency.

A deliberately lightweight, dialect-tolerant regex parser is used instead of
``simple-ddl-parser`` so the feature ships with zero new runtime requirements
(no container rebuild). It targets the common, standards-ish DDL shapes:

* ``CREATE SCHEMA``
* ``CREATE TABLE`` (columns, inline + table-level ``FOREIGN KEY`` constraints)
* ``CREATE VIEW`` / ``CREATE MATERIALIZED VIEW`` (``FROM`` / ``JOIN`` tables)
* ``CREATE INDEX``
* ``CREATE TRIGGER``
* ``CREATE FUNCTION`` / ``CREATE PROCEDURE``

Emitted node types (``symbol_type``):
    sql_schema, sql_table, sql_view, sql_column, sql_index,
    sql_function, sql_trigger

Emitted edges (``relationship_type``):
    defines       schema → table, table → column, table → index
    references    column → column (FK), view → table
    triggered_by  trigger → table
    calls         function → function

Resolution is global across all SQL files in the repository so foreign keys
and view references can point at tables declared in a different file.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

from ..feature_flags import FeatureFlags, get_feature_flags

logger = logging.getLogger(__name__)

# ── Edge / node provenance constants ──────────────────────────────────────
_LANGUAGE = "sql"
_CREATED_BY = "sql_extractor"
_EDGE_CLASS = "structural"
_ANALYSIS_LEVEL = "code"
_CONFIDENCE = "EXTRACTED"

# ── Statement-classifying regexes (case-insensitive) ──────────────────────
_IDENT = r'[\w".`\[\]]+'

_RE_SCHEMA = re.compile(
    r'^\s*CREATE\s+(?:OR\s+REPLACE\s+)?SCHEMA\s+(?:IF\s+NOT\s+EXISTS\s+)?'
    r'(?:AUTHORIZATION\s+\w+\s+)?(?P<name>' + _IDENT + r')',
    re.IGNORECASE,
)
_RE_TABLE = re.compile(
    r'^\s*CREATE\s+(?:(?:GLOBAL|LOCAL)\s+)?(?:(?:TEMP|TEMPORARY|UNLOGGED)\s+)?'
    r'TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>' + _IDENT + r')\s*'
    r'\((?P<body>.*)\)\s*[^)]*$',
    re.IGNORECASE | re.DOTALL,
)
_RE_VIEW = re.compile(
    r'^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:MATERIALIZED\s+)?VIEW\s+'
    r'(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>' + _IDENT + r')\b.*?\bAS\b'
    r'(?P<query>.*)$',
    re.IGNORECASE | re.DOTALL,
)
_RE_INDEX = re.compile(
    r'^\s*CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?'
    r'(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>' + _IDENT + r')\s+ON\s+'
    r'(?:ONLY\s+)?(?P<table>' + _IDENT + r')',
    re.IGNORECASE | re.DOTALL,
)
_RE_TRIGGER = re.compile(
    r'^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:CONSTRAINT\s+)?TRIGGER\s+'
    r'(?P<name>' + _IDENT + r')\b.*?\bON\s+(?P<table>' + _IDENT + r')',
    re.IGNORECASE | re.DOTALL,
)
_RE_FUNCTION = re.compile(
    r'^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE)\s+'
    r'(?P<name>' + _IDENT + r')\s*\(',
    re.IGNORECASE | re.DOTALL,
)

# Inline / table-level foreign key: REFERENCES other(col)
_RE_REFERENCES = re.compile(
    r'\bREFERENCES\s+(?P<table>' + _IDENT + r')\s*'
    r'(?:\(\s*(?P<col>' + _IDENT + r')\s*\))?',
    re.IGNORECASE,
)
_RE_FK_CONSTRAINT = re.compile(
    r'\bFOREIGN\s+KEY\s*\(\s*(?P<local>' + _IDENT + r')\s*\)\s*'
    r'REFERENCES\s+(?P<table>' + _IDENT + r')\s*(?:\(\s*(?P<col>' + _IDENT + r')\s*\))?',
    re.IGNORECASE,
)
# FROM / JOIN <table> inside a view query.
_RE_FROM_JOIN = re.compile(
    r'\b(?:FROM|JOIN)\s+(?P<table>' + _IDENT + r')',
    re.IGNORECASE,
)
# Any identifier immediately followed by '(' — a candidate function call.
_RE_CALL = re.compile(r'\b(?P<name>[A-Za-z_]\w*)\s*\(')

# Column-definition lines starting with one of these keywords are table-level
# constraints, not columns.
_CONSTRAINT_LEADERS = frozenset({
    "constraint", "primary", "foreign", "unique", "check", "key",
    "index", "exclude", "like", "period",
})


def _clean_ident(raw: str) -> str:
    """Strip quoting (`"`, `` ` ``, ``[]``) from a single identifier part."""
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    elif raw.startswith('`') and raw.endswith('`'):
        raw = raw[1:-1]
    elif raw.startswith('[') and raw.endswith(']'):
        raw = raw[1:-1]
    return raw


def _split_qualified(name: str) -> Tuple[Optional[str], str]:
    """Split a possibly schema-qualified name into ``(schema, bare)``."""
    parts = [_clean_ident(p) for p in re.split(r'\.', name.strip()) if p.strip()]
    if not parts:
        return None, name.strip()
    if len(parts) >= 2:
        return parts[-2], parts[-1]
    return None, parts[-1]


def _blank_comments(sql: str) -> str:
    """Replace comments with same-length whitespace, preserving offsets/newlines."""
    out = list(sql)
    n = len(sql)
    i = 0
    while i < n:
        ch = sql[i]
        # Line comment: -- ... EOL
        if ch == '-' and i + 1 < n and sql[i + 1] == '-':
            j = i
            while j < n and sql[j] != '\n':
                out[j] = ' '
                j += 1
            i = j
            continue
        # Block comment: /* ... */
        if ch == '/' and i + 1 < n and sql[i + 1] == '*':
            j = i
            while j < n and not (sql[j] == '*' and j + 1 < n and sql[j + 1] == '/'):
                if sql[j] != '\n':
                    out[j] = ' '
                j += 1
            # blank the closing '*/'
            if j < n:
                out[j] = ' '
                if j + 1 < n:
                    out[j + 1] = ' '
                j += 2
            i = j
            continue
        # Single-quoted string — skip (don't blank, but step over to ignore ';')
        if ch == "'":
            i += 1
            while i < n:
                if sql[i] == "'" and i + 1 < n and sql[i + 1] == "'":
                    i += 2
                    continue
                if sql[i] == "'":
                    i += 1
                    break
                i += 1
            continue
        i += 1
    return ''.join(out)


def _split_statements(sql: str) -> List[Tuple[int, str]]:
    """Split DDL into ``(start_offset, text)`` statements.

    Respects parenthesis depth, single/double-quoted strings, and PostgreSQL
    dollar-quoted bodies (``$$ ... $$`` / ``$tag$ ... $tag$``) so function
    bodies containing ``;`` are not split.
    """
    statements: List[Tuple[int, str]] = []
    n = len(sql)
    i = 0
    depth = 0
    start = 0
    while i < n:
        ch = sql[i]
        if ch == "'" or ch == '"':
            quote = ch
            i += 1
            while i < n:
                if sql[i] == quote and quote == "'" and i + 1 < n and sql[i + 1] == "'":
                    i += 2
                    continue
                if sql[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if ch == '$':
            m = re.match(r'\$(\w*)\$', sql[i:])
            if m:
                tag = m.group(0)
                end = sql.find(tag, i + len(tag))
                if end == -1:
                    i = n
                else:
                    i = end + len(tag)
                continue
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        elif ch == ';' and depth == 0:
            text = sql[start:i]
            if text.strip():
                statements.append((start, text))
            start = i + 1
        i += 1
    tail = sql[start:]
    if tail.strip():
        statements.append((start, tail))
    return statements


def _split_top_level_commas(body: str) -> List[str]:
    """Split a table body on top-level commas (ignoring nested parens)."""
    parts: List[str] = []
    depth = 0
    cur = []
    for ch in body:
        if ch == '(':
            depth += 1
            cur.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append(''.join(cur))
    return parts


class _SqlGraph:
    """Accumulates SQL nodes/edges, resolving references in a second pass."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.graph.graph["language"] = _LANGUAGE
        self.graph.graph["analysis_level"] = _ANALYSIS_LEVEL
        # name (lower) → node_id; stores both qualified (schema.table) and bare names
        self.tables: Dict[str, str] = {}
        self.columns: Dict[str, str] = {}     # "table.col" lower → node_id
        self.functions: Dict[str, str] = {}
        self.schemas: Dict[str, str] = {}     # bare_lower → sql_schema node_id (O(1) lookup)
        self.qualified_tables: Dict[str, str] = {}  # "schema.table" lower → node_id (for schema-qualified refs)
        # Deferred references resolved after all nodes are created.
        self._pending_fk: List[Tuple[str, str, Optional[str]]] = []   # (col_node, ref_table, ref_col)
        self._pending_view: List[Tuple[str, List[str]]] = []          # (view_node, [tables])
        self._pending_trigger: List[Tuple[str, str]] = []             # (trigger_node, table)
        self._pending_calls: List[Tuple[str, str]] = []               # (func_node, body)
        self.stats = {
            "sql_files": 0,
            "tables": 0,
            "views": 0,
            "columns": 0,
            "indexes": 0,
            "functions": 0,
            "triggers": 0,
            "schemas": 0,
            "defines_edges": 0,
            "references_edges": 0,
            "triggered_by_edges": 0,
            "calls_edges": 0,
        }

    # -- node helpers ------------------------------------------------------
    def _add_node(
        self,
        node_id: str,
        name: str,
        symbol_type: str,
        rel_path: str,
        file_path: str,
        start_line: int,
        end_line: int,
        source_text: str,
    ) -> None:
        if node_id in self.graph:
            return
        file_name = Path(rel_path).stem
        self.graph.add_node(
            node_id,
            name=name,
            type=symbol_type,
            symbol_name=name,
            symbol_type=symbol_type,
            file_path=file_path,
            rel_path=rel_path,
            file_name=file_name,
            language=_LANGUAGE,
            start_line=start_line,
            end_line=end_line,
            analysis_level=_ANALYSIS_LEVEL,
            source_text=source_text,
            docstring="",
            parameters=[],
            return_type="",
        )

    def _add_edge(self, src: str, dst: str, rel_type: str, *, raw_score: float = 1.0) -> None:
        if src not in self.graph or dst not in self.graph:
            return
        self.graph.add_edge(
            src,
            dst,
            key=f"sql::{rel_type}",
            relationship_type=rel_type,
            edge_class=_EDGE_CLASS,
            analysis_level=_ANALYSIS_LEVEL,
            language=_LANGUAGE,
            weight=1.0,
            raw_similarity=raw_score,
            source_file=self.graph.nodes[src].get("rel_path", ""),
            target_file=self.graph.nodes[dst].get("rel_path", ""),
            created_by=_CREATED_BY,
            annotations={"confidence": _CONFIDENCE},
        )

    # -- pass 1: parse one file -------------------------------------------
    def add_file(self, file_path: str, rel_path: str, text: str) -> None:
        self.stats["sql_files"] += 1
        cleaned = _blank_comments(text)
        for start, stmt in _split_statements(cleaned):
            start_line = text[:start].count("\n") + 1
            end_line = start_line + stmt.count("\n")
            orig_stmt = text[start:start + len(stmt)]
            self._classify(stmt, orig_stmt, rel_path, file_path, start_line, end_line)

    def _node_id(self, kind: str, rel_path: str, qualified: str) -> str:
        return f"sql::{rel_path}::{kind}:{qualified}"

    def _classify(
        self,
        stmt: str,
        orig_stmt: str,
        rel_path: str,
        file_path: str,
        start_line: int,
        end_line: int,
    ) -> None:
        # SCHEMA
        m = _RE_SCHEMA.match(stmt)
        if m:
            _, bare = _split_qualified(m.group("name"))
            nid = self._node_id("schema", rel_path, bare)
            self._add_node(nid, bare, "sql_schema", rel_path, file_path,
                           start_line, end_line, orig_stmt.strip()[:2000])
            self.schemas[bare.lower()] = nid
            self.stats["schemas"] += 1
            return

        # TABLE
        m = _RE_TABLE.match(stmt)
        if m:
            self._handle_table(m, rel_path, file_path, start_line, end_line, orig_stmt)
            return

        # VIEW
        m = _RE_VIEW.match(stmt)
        if m:
            schema, bare = _split_qualified(m.group("name"))
            # Use qualified name in node ID to avoid collisions with same view names in different schemas.
            qualified = f"{schema}.{bare}" if schema else bare
            nid = self._node_id("view", rel_path, qualified)
            self._add_node(nid, bare, "sql_view", rel_path, file_path,
                           start_line, end_line, orig_stmt.strip()[:2000])
            # Store by qualified name for disambiguation; views are FROM-able like tables.
            self.qualified_tables[qualified.lower()] = nid
            # Store by bare name only if not already present (first occurrence wins).
            if bare.lower() not in self.tables:
                self.tables[bare.lower()] = nid
            self.stats["views"] += 1
            refs = [_split_qualified(t.group("table"))[1]
                    for t in _RE_FROM_JOIN.finditer(m.group("query"))]
            self._pending_view.append((nid, refs))
            return

        # INDEX
        m = _RE_INDEX.match(stmt)
        if m:
            schema, bare = _split_qualified(m.group("name"))
            _, tbl_bare = _split_qualified(m.group("table"))
            nid = self._node_id("index", rel_path, f"{tbl_bare}.{bare}")
            self._add_node(nid, bare, "sql_index", rel_path, file_path,
                           start_line, end_line, orig_stmt.strip()[:1000])
            self.stats["indexes"] += 1
            # table → index (defines) — deferred so the table can be in another file
            self._pending_fk.append((nid, tbl_bare, "__index__"))
            return

        # TRIGGER
        m = _RE_TRIGGER.match(stmt)
        if m:
            schema, bare = _split_qualified(m.group("name"))
            _, tbl_bare = _split_qualified(m.group("table"))
            nid = self._node_id("trigger", rel_path, bare)
            self._add_node(nid, bare, "sql_trigger", rel_path, file_path,
                           start_line, end_line, orig_stmt.strip()[:1500])
            self.stats["triggers"] += 1
            self._pending_trigger.append((nid, tbl_bare))
            return

        # FUNCTION / PROCEDURE
        m = _RE_FUNCTION.match(stmt)
        if m:
            schema, bare = _split_qualified(m.group("name"))
            nid = self._node_id("function", rel_path, bare)
            self._add_node(nid, bare, "sql_function", rel_path, file_path,
                           start_line, end_line, orig_stmt.strip()[:4000])
            self.functions[bare.lower()] = nid
            self.stats["functions"] += 1
            self._pending_calls.append((nid, stmt))
            return

    def _handle_table(self, m, rel_path, file_path, start_line, end_line, orig_stmt):
        schema, bare = _split_qualified(m.group("name"))
        # Use qualified name in node ID when schema is present to avoid collisions
        # (e.g., tenant_a.users and tenant_b.users are different tables).
        qualified = f"{schema}.{bare}" if schema else bare
        tbl_nid = self._node_id("table", rel_path, qualified)
        self._add_node(tbl_nid, bare, "sql_table", rel_path, file_path,
                       start_line, end_line, orig_stmt.strip()[:4000])
        # Store by qualified name for disambiguation when resolving schema-qualified refs.
        self.qualified_tables[qualified.lower()] = tbl_nid
        # Store by bare name only if not already present (first occurrence wins)
        # to avoid overwrites when multiple schemas have the same table name.
        if bare.lower() not in self.tables:
            self.tables[bare.lower()] = tbl_nid
        self.stats["tables"] += 1

        # schema → table (defines), if schema named and present later
        if schema:
            self._pending_fk.append((tbl_nid, schema, "__schema__"))

        body = m.group("body")
        for raw_part in _split_top_level_commas(body):
            part = raw_part.strip()
            if not part:
                continue
            first = re.split(r'\s+', part, 1)[0].lower().strip('"`[]')
            if first in _CONSTRAINT_LEADERS:
                # Table-level constraint — look for FOREIGN KEY.
                fk = _RE_FK_CONSTRAINT.search(part)
                if fk:
                    _, local_col = _split_qualified(fk.group("local"))
                    _, ref_tbl = _split_qualified(fk.group("table"))
                    ref_col = fk.group("col")
                    ref_col = _split_qualified(ref_col)[1] if ref_col else None
                    col_nid = self._node_id("column", rel_path, f"{bare}.{local_col}")
                    # The local column node is created in the column loop; defer.
                    self._pending_fk.append((col_nid, ref_tbl, ref_col))
                continue

            # Column definition: first token is the column name.
            col_name = _clean_ident(re.split(r'\s+', part, 1)[0])
            if not col_name:
                continue
            col_nid = self._node_id("column", rel_path, f"{bare}.{col_name}")
            self._add_node(col_nid, col_name, "sql_column", rel_path, file_path,
                           start_line, end_line, part[:500])
            self.columns[f"{bare}.{col_name}".lower()] = col_nid
            self.stats["columns"] += 1
            # table → column (defines)
            self._add_edge(tbl_nid, col_nid, "defines")
            self.stats["defines_edges"] += 1

            # Inline REFERENCES (FK)
            ref = _RE_REFERENCES.search(part)
            if ref:
                # Preserve qualified table name for disambiguation in resolve()
                ref_tbl = ref.group("table")  # may be "schema.table" or "table"
                ref_col = ref.group("col")
                ref_col = _split_qualified(ref_col)[1] if ref_col else None
                self._pending_fk.append((col_nid, ref_tbl, ref_col))

    # -- pass 2: resolve deferred references ------------------------------
    def resolve(self) -> None:
        # schema→table, table→index, column FKs (all share the _pending_fk list,
        # discriminated by the sentinel ref_col value).
        for src_nid, ref_table, ref_col in self._pending_fk:
            # Extract bare name for fallback lookups (e.g., "users" from "schema.users")
            _, bare_table = _split_qualified(ref_table)
            # Try qualified lookup first, then bare-name fallback
            tbl_nid = self.qualified_tables.get(ref_table.lower()) or self.tables.get(bare_table.lower())
            if ref_col == "__schema__":
                # src is the table; resolve the schema node by bare name (O(1)).
                schema_nid = self.schemas.get(bare_table.lower())
                if schema_nid:
                    self._add_edge(schema_nid, src_nid, "defines")
                    self.stats["defines_edges"] += 1
                continue
            if ref_col == "__index__":
                # src is the index; table → index defines.
                if tbl_nid:
                    self._add_edge(tbl_nid, src_nid, "defines")
                    self.stats["defines_edges"] += 1
                continue
            # Foreign key: column → column, else column → table.
            target = None
            if ref_col:
                target = self.columns.get(f"{bare_table}.{ref_col}".lower())
            if target is None:
                target = tbl_nid
            if target is not None:
                self._add_edge(src_nid, target, "references", raw_score=0.95)
                self.stats["references_edges"] += 1

        for view_nid, tables in self._pending_view:
            for t in tables:
                tbl_nid = self.tables.get(t.lower())
                if tbl_nid and tbl_nid != view_nid:
                    self._add_edge(view_nid, tbl_nid, "references", raw_score=0.9)
                    self.stats["references_edges"] += 1

        for trig_nid, table in self._pending_trigger:
            tbl_nid = self.tables.get(table.lower())
            if tbl_nid:
                self._add_edge(trig_nid, tbl_nid, "triggered_by")
                self.stats["triggered_by_edges"] += 1

        for func_nid, body in self._pending_calls:
            seen = set()
            for cm in _RE_CALL.finditer(body):
                callee = cm.group("name").lower()
                target = self.functions.get(callee)
                if target and target != func_nid and target not in seen:
                    self._add_edge(func_nid, target, "calls")
                    self.stats["calls_edges"] += 1
                    seen.add(target)


def build_sql_graph(
    file_paths: List[str],
    repo_path: str,
    *,
    flags: Optional[FeatureFlags] = None,
) -> Tuple[nx.MultiDiGraph, Dict[str, int]]:
    """Parse ``.sql`` / ``.ddl`` files into a typed SQL subgraph.

    Args:
        file_paths: Absolute paths of SQL files to parse.
        repo_path: Repository root, used to compute ``rel_path``.
        flags: Optional feature flags; falls back to ``get_feature_flags()``.

    Returns:
        ``(graph, stats)`` where ``graph`` is a ``MultiDiGraph`` of SQL nodes
        and edges (empty when the feature is disabled or no files parse) and
        ``stats`` is a count breakdown.
    """
    flags = flags or get_feature_flags()
    empty_stats = {
        "sql_files": 0, "tables": 0, "views": 0, "columns": 0, "indexes": 0,
        "functions": 0, "triggers": 0, "schemas": 0, "defines_edges": 0,
        "references_edges": 0, "triggered_by_edges": 0, "calls_edges": 0,
    }
    if not getattr(flags, "sql_extraction", True):
        return nx.MultiDiGraph(), empty_stats
    if not file_paths:
        return nx.MultiDiGraph(), empty_stats

    repo = Path(repo_path)
    builder = _SqlGraph()
    for fp in file_paths:
        try:
            text = Path(fp).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - IO edge
            logger.warning("sql_extractor: cannot read %s: %s", fp, exc)
            continue
        try:
            rel_path = str(Path(fp).relative_to(repo))
        except ValueError:
            rel_path = Path(fp).name
        try:
            builder.add_file(fp, rel_path, text)
        except Exception as exc:  # pragma: no cover - parser robustness
            logger.warning("sql_extractor: failed to parse %s: %s", fp, exc)

    builder.resolve()
    logger.info(
        "sql_extractor: %d files → %d tables, %d views, %d columns, "
        "%d functions, %d triggers; edges: %d defines, %d references, "
        "%d triggered_by, %d calls",
        builder.stats["sql_files"], builder.stats["tables"],
        builder.stats["views"], builder.stats["columns"],
        builder.stats["functions"], builder.stats["triggers"],
        builder.stats["defines_edges"], builder.stats["references_edges"],
        builder.stats["triggered_by_edges"], builder.stats["calls_edges"],
    )
    return builder.graph, builder.stats
