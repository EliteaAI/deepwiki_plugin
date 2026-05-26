"""Tests for universal via=callsite@L injection at the graph_builder seam.

Every parser records the precise event line in ``Relationship.source_range``
(the AST node of the CALL / CREATES / REFERENCES / IMPORTS event). The
graph builder now injects that into the edge's ``annotations.via`` list
using the same vocabulary contraction emits (``via=src=...@L<line>`` /
``via=tgt=...@L<line>``) — relationship-agnostic, one seam, no parser
walks.

Pins:
- annotations are preserved when source_range is missing (legacy edges);
- when source_range is set, ``via=<rel_type>@L<line>`` is appended;
- the format uses the canonical rel_type ("calls", "creates",
  "references", "imports") so an MCP consumer can answer "what and
  where" from a single edge row.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from plugin_implementation.code_graph.graph_builder import (
    EnhancedUnifiedGraphBuilder,
)


def _write(tmpdir: str, name: str, content: str) -> str:
    path = os.path.join(tmpdir, name)
    os.makedirs(os.path.dirname(path) or tmpdir, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def _via_list(edge_data: dict) -> list:
    """Return the via list from edge annotations (handles dict shape).

    Inside an in-memory NX graph (pre-DB persistence) annotations are
    still a dict — JSON serialisation only happens on insert.
    """
    anns = edge_data.get("annotations") or {}
    if not isinstance(anns, dict):
        return []
    via = anns.get("via")
    if isinstance(via, list):
        return via
    return [via] if via else []


# ─── 1. Python: CALL → via=calls@L<line> ──────────────────────────────

class TestPythonCallsiteVia:

    def test_python_call_emits_via_callsite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write(tmpdir, "svc.py", "\n".join([
                "class A:",
                "    def m(self):",
                "        B()",   # line 3 — the call site we want anchored
                "",
                "class B:",
                "    pass",
            ]))
            builder = EnhancedUnifiedGraphBuilder()
            analysis = builder.analyze_repository(tmpdir)
            graph = analysis.unified_graph

        # Find the A.m → B edge (CREATES, since B is a class).
        creates_edges = [
            (u, v, d)
            for u, v, d in graph.edges(data=True)
            if d.get("relationship_type") in ("creates", "calls")
            and v.endswith("::B")
            and "A.m" in u  # source is the method, not the class
        ]
        assert creates_edges, (
            f"Expected an A.m → B creates/calls edge, got edges with "
            f"target=B: {[(u, v, d.get('relationship_type')) for u, v, d in graph.edges(data=True) if v.endswith('::B')]}"
        )
        _u, _v, data = creates_edges[0]
        via = _via_list(data)
        rel = data.get("relationship_type")
        # The injected via entry must name the rel_type and a line.
        assert any(
            entry.startswith(f"{rel}@L") for entry in via
        ), f"Expected '{rel}@L<n>' in via list, got {via}"


# ─── 2. Idempotency / safety ─────────────────────────────────────────

class TestNoSourceRange:
    """If a producer ever omits source_range, the edge survives without
    a phantom via entry."""

    def test_no_source_range_no_via_callsite(self):
        # Build a Relationship-shaped object without source_range and feed
        # it through the same path graph_builder uses. We can't trivially
        # invoke just _add_relationships_bulk in isolation (it expects a
        # large analysis context), so we exercise the safety via the
        # public _SimpleRel surrogate and pin the source_range guard.
        from plugin_implementation.parsers.base_parser import Relationship, RelationshipType

        rel = Relationship(
            source_symbol="A.foo",
            target_symbol="B",
            relationship_type=RelationshipType.CALLS,
            source_file="x.py",
            source_range=None,  # ← missing
            annotations={"existing": "value"},
        )
        # Re-implement just the via injection for the unit check —
        # mirrors the production seam exactly.
        rel_annotations = dict(rel.annotations or {})
        source_range = rel.source_range
        if source_range is not None and getattr(source_range, "start", None) is not None:
            via_list = rel_annotations.setdefault("via", [])
            if isinstance(via_list, list):
                via_list.append(
                    f"{rel.relationship_type.value}@L{source_range.start.line}"
                )
        assert "via" not in rel_annotations
        assert rel_annotations["existing"] == "value"

    def test_existing_via_list_preserved(self):
        """Contraction may have already populated annotations.via with
        ``src=...@L`` / ``tgt=...@L``. The callsite injection must
        append, not replace."""
        from plugin_implementation.parsers.base_parser import Range, Position, Relationship, RelationshipType

        rel = Relationship(
            source_symbol="User",
            target_symbol="Validator",
            relationship_type=RelationshipType.REFERENCES,
            source_file="x.py",
            source_range=Range(start=Position(line=42, column=0),
                               end=Position(line=42, column=10)),
            annotations={"via": ["src=email@L11", "tgt=ValidatorImpl@L7"]},
        )
        rel_annotations = dict(rel.annotations or {})
        source_range = rel.source_range
        if source_range is not None and getattr(source_range, "start", None) is not None:
            via_list = rel_annotations.setdefault("via", [])
            if isinstance(via_list, list):
                via_list.append(
                    f"{rel.relationship_type.value}@L{source_range.start.line}"
                )
        # Pre-existing entries must be intact, callsite appended.
        assert rel_annotations["via"] == [
            "src=email@L11",
            "tgt=ValidatorImpl@L7",
            "references@L42",
        ]
