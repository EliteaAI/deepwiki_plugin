"""ORM model → SQL table linker (roadmap C4).

When a class in application code is an ORM model (SQLAlchemy ``Column`` /
Django ``models.Field`` / Hibernate ``@Entity``), this post-pass emits a
``models_table`` edge from the model-class node to the matching ``sql_table``
node produced by :mod:`code_graph.sql_extractor`.

The pass runs *after* the SQL subgraph has been merged into the unified graph
(so ``sql_table`` nodes exist) and is intentionally self-contained: it reads
only ``source_text`` / ``name`` / ``type`` node attributes already present on
the graph, so it needs no parser changes.

Phase D will collapse ``models_table`` into the generic ``defines`` / ``consumes``
contract algebra; until then it is a first-class ``cross_language`` edge so the
writer (which filters to ``cross_language`` edges) can surface the coupling.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# ORM detection signals
# ──────────────────────────────────────────────────────────────────────

# SQLAlchemy: ``Column(...)`` declarations and the declarative base / Table().
_SQLALCHEMY_SIGNALS = (
    re.compile(r"\bColumn\s*\("),
    re.compile(r"\b__tablename__\b"),
    re.compile(r"\bmapped_column\s*\("),
)
# Django: ``models.XxxField(...)`` declarations.
_DJANGO_SIGNAL = re.compile(r"\bmodels\.\w*Field\s*\(")
# Hibernate / JPA: ``@Entity`` (optionally with ``@Table(name="...")``).
_HIBERNATE_SIGNAL = re.compile(r"@Entity\b")

# Explicit table-name declarations (highest priority when present).
_SQLALCHEMY_TABLENAME = re.compile(
    r"__tablename__\s*=\s*['\"]([A-Za-z_][\w]*)['\"]"
)
_DJANGO_DB_TABLE = re.compile(
    r"db_table\s*=\s*['\"]([A-Za-z_][\w]*)['\"]"
)
_HIBERNATE_TABLE = re.compile(
    r"@Table\s*\(\s*(?:[^)]*\b)?name\s*=\s*\"([A-Za-z_][\w]*)\""
)

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")  # Handles consecutive capitals


def _to_snake_case(name: str) -> str:
    """Convert camelCase to snake_case, treating consecutive capitals as units.
    Examples: ``BlogPost`` → ``blog_post``, ``APIKey`` → ``api_key``, ``HTTPSConnection`` → ``https_connection``.
    """
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def _detect_orm(source_text: str, language: str) -> bool:
    """Return True when ``source_text`` looks like an ORM model definition."""
    if not source_text:
        return False
    lang = (language or "").lower()
    if lang in ("python", "py"):
        if _DJANGO_SIGNAL.search(source_text):
            return True
        return any(sig.search(source_text) for sig in _SQLALCHEMY_SIGNALS)
    if lang in ("java", "kotlin"):
        return bool(_HIBERNATE_SIGNAL.search(source_text))
    return False


def _candidate_table_names(class_name: str, source_text: str, language: str) -> List[str]:
    """Derive plausible table names for an ORM model, best match first."""
    names: List[str] = []
    lang = (language or "").lower()

    if lang in ("python", "py"):
        m = _SQLALCHEMY_TABLENAME.search(source_text)
        if m:
            names.append(m.group(1))
        m = _DJANGO_DB_TABLE.search(source_text)
        if m:
            names.append(m.group(1))
        # Django default: lowercased class name; SQLAlchemy convention: snake_case.
        names.append(class_name.lower())
        names.append(_to_snake_case(class_name))
    elif lang in ("java", "kotlin"):
        m = _HIBERNATE_TABLE.search(source_text)
        if m:
            names.append(m.group(1))
        # JPA default: the entity (class) name.
        names.append(class_name)
        names.append(class_name.lower())
        names.append(_to_snake_case(class_name))

    # De-dupe preserving order.
    seen: set = set()
    ordered: List[str] = []
    for n in names:
        key = n.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(n)
    return ordered


def _build_table_index(g: nx.MultiDiGraph) -> Dict[str, str]:
    """Map ``lower(table_name) -> sql_table node_id`` for matching."""
    index: Dict[str, str] = {}
    for node_id, data in g.nodes(data=True):
        stype = data.get("symbol_type") or data.get("type") or ""
        if hasattr(stype, "value"):
            stype = stype.value
        if str(stype).lower() != "sql_table":
            continue
        name = data.get("name") or data.get("symbol_name") or ""
        if name:
            index.setdefault(name.lower(), node_id)
    return index


def _build_view_index(g: nx.MultiDiGraph) -> Dict[str, str]:
    """Map ``lower(view_name) -> sql_view node_id`` for ORM→view matching.

    Some ORM classes legitimately map to database views (e.g. SQLAlchemy
    ``selectable`` mappings).  Kept separate from the table index so
    the caller can distinguish ``models_table`` from ``models_view``
    edges and avoid mixing the two concepts.
    """
    index: Dict[str, str] = {}
    for node_id, data in g.nodes(data=True):
        stype = data.get("symbol_type") or data.get("type") or ""
        if hasattr(stype, "value"):
            stype = stype.value
        if str(stype).lower() != "sql_view":
            continue
        name = data.get("name") or data.get("symbol_name") or ""
        if name:
            index.setdefault(name.lower(), node_id)
    return index


def link_orm_models(g: nx.MultiDiGraph) -> int:
    """Add ``models_table`` / ``models_view`` edges from ORM model classes.

    * ``models_table`` — ORM class maps to a ``sql_table`` node.
    * ``models_view`` — ORM class maps to a ``sql_view`` node (e.g. SQLAlchemy
      ``selectable`` mappings).

    Returns the number of edges added. No-op (returns 0) when the graph has no
    ``sql_table`` or ``sql_view`` nodes.
    """
    table_index = _build_table_index(g)
    view_index = _build_view_index(g)
    if not table_index and not view_index:
        return 0

    added = 0
    for node_id, data in list(g.nodes(data=True)):
        ntype = data.get("type") or data.get("symbol_type") or ""
        if hasattr(ntype, "value"):
            ntype = ntype.value
        if str(ntype).lower() != "class":
            continue

        source_text = data.get("source_text") or ""
        language = data.get("language") or ""
        if not _detect_orm(source_text, language):
            continue

        class_name = data.get("name") or data.get("symbol_name") or ""
        if not class_name:
            continue

        # First try to match a sql_table; then fall back to sql_view.
        target_id: Optional[str] = None
        matched_name: Optional[str] = None
        edge_key = "models_table"
        for cand in _candidate_table_names(class_name, source_text, language):
            tid = table_index.get(cand.lower())
            if tid is not None:
                target_id = tid
                matched_name = cand
                edge_key = "models_table"
                break

        if target_id is None:
            for cand in _candidate_table_names(class_name, source_text, language):
                vid = view_index.get(cand.lower())
                if vid is not None:
                    target_id = vid
                    matched_name = cand
                    edge_key = "models_view"
                    break

        if target_id is None or g.has_edge(node_id, target_id, key=edge_key):
            continue

        g.add_edge(
            node_id,
            target_id,
            key=edge_key,
            relationship_type=edge_key,
            type=edge_key,
            edge_class="cross_language",
            weight=0.85,
            language=language,
            created_by="orm_linker",
            annotations={"confidence": "INFERRED", "table_name": matched_name},
            provenance={
                "source": "orm_linker",
                "matcher": "orm_table_name",
            },
        )
        added += 1

    if added:
        logger.info("ORM linker: added %d models_table/models_view edge(s)", added)
    return added
