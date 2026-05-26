"""Tests for via= context propagation through the writer expansion pool.

The contraction module attaches an ``annotations.via`` list to every
rewired edge so the dropped intermediate field/parameter still anchors
the edge to a line in source. Enhancement 1 (Phase B) flows that
context through ``_collect_expansion_neighbors`` and surfaces it on
each expanded ``Document`` as ``metadata['expansion_via']``.

These tests pin:
- the ``_extract_via_from_annotations`` formatter on every shape it
  is realistically asked to handle (JSON string, dict, malformed,
  empty);
- end-to-end propagation from a real edge row → expansion candidate
  → ``_node_to_document`` metadata.
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import patch

import pytest

from plugin_implementation.cluster_expansion import (
    _collect_expansion_neighbors,
    _extract_via_from_annotations,
    _node_to_document,
)
from plugin_implementation.feature_flags import FeatureFlags


# ─── 1. Formatter contract ─────────────────────────────────────────────

class TestExtractVia:

    def test_json_string_with_list(self):
        raw = json.dumps({"via": ["src=email@L42", "tgt=username@L11"]})
        assert _extract_via_from_annotations(raw) == (
            "via src=email@L42; tgt=username@L11"
        )

    def test_dict_short_circuit(self):
        raw = {"via": ["src=foo@L7"]}
        assert _extract_via_from_annotations(raw) == "via src=foo@L7"

    def test_via_as_scalar_string(self):
        # Some legacy producers stored a single via string (not a list).
        raw = json.dumps({"via": "src=foo@L7"})
        assert _extract_via_from_annotations(raw) == "via src=foo@L7"

    def test_no_via_key(self):
        raw = json.dumps({"field_name": "x", "cross_file": True})
        assert _extract_via_from_annotations(raw) == ""

    def test_empty_inputs(self):
        assert _extract_via_from_annotations("") == ""
        assert _extract_via_from_annotations(None) == ""
        assert _extract_via_from_annotations({}) == ""

    def test_malformed_json_returns_empty(self):
        assert _extract_via_from_annotations("not-json {[") == ""

    def test_via_with_blank_entries_dropped(self):
        raw = json.dumps({"via": ["src=foo@L1", "", "  "]})
        assert _extract_via_from_annotations(raw) == "via src=foo@L1"


# ─── 2. End-to-end via propagation ─────────────────────────────────────

def _make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE repo_nodes (
            node_id TEXT PRIMARY KEY,
            symbol_name TEXT,
            symbol_type TEXT,
            rel_path TEXT,
            file_name TEXT,
            language TEXT DEFAULT 'python',
            start_line INTEGER DEFAULT 0,
            end_line INTEGER DEFAULT 0,
            source_text TEXT DEFAULT '',
            is_architectural INTEGER DEFAULT 1,
            is_doc INTEGER DEFAULT 0,
            macro_cluster INTEGER,
            micro_cluster INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE repo_edges (
            source_id TEXT,
            target_id TEXT,
            rel_type TEXT,
            edge_class TEXT NOT NULL DEFAULT 'structural',
            annotations TEXT DEFAULT '',
            weight REAL DEFAULT 1.0
        )
    """)
    return conn


def _add_node(conn, nid, name, sym_type="class", macro=0):
    conn.execute(
        "INSERT INTO repo_nodes "
        "(node_id, symbol_name, symbol_type, rel_path, file_name, "
        "is_architectural, macro_cluster) VALUES (?,?,?,?,?,?,?)",
        (nid, name, sym_type, f"src/{name}.py", f"{name}.py", 1, macro),
    )


def _add_edge(conn, src, tgt, rel_type, *, annotations=None,
              edge_class="structural"):
    conn.execute(
        "INSERT INTO repo_edges "
        "(source_id, target_id, rel_type, edge_class, annotations, weight) "
        "VALUES (?,?,?,?,?,?)",
        (src, tgt, rel_type, edge_class,
         json.dumps(annotations) if annotations else "", 1.0),
    )


class _DBStub:
    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def db_with_via():
    conn = _make_db()
    _add_node(conn, "n_seed", "User")
    _add_node(conn, "n_validator", "EmailValidator")
    _add_node(conn, "n_plain", "AuthService")
    # Edge with contraction's via= context (User.email field rewired
    # onto User class after contraction).
    _add_edge(
        conn, "n_seed", "n_validator", "references",
        annotations={"via": ["src=email@L42", "tgt=ValidatorImpl@L11"]},
    )
    # Edge without via= (plain AST relationship).
    _add_edge(conn, "n_seed", "n_plain", "calls")
    conn.commit()
    return conn


class TestViaPropagation:

    def test_via_attached_to_node_dict(self, db_with_via):
        with patch(
            "plugin_implementation.cluster_expansion.get_feature_flags",
            return_value=FeatureFlags(),
        ):
            pool = _collect_expansion_neighbors(
                _DBStub(db_with_via),
                seed_ids=["n_seed"],
                seen_ids={"n_seed"},
                macro_id=0,
            )
        # Pool returns (nid, node, rel_type) 3-tuples — weight is
        # dropped in the final dedup/return.
        validator_entries = [t for t in pool if t[0] == "n_validator"]
        assert validator_entries, "Validator neighbour missing from pool"
        _nid, node, _rel = validator_entries[0]
        assert node.get("_via_context") == (
            "via src=email@L42; tgt=ValidatorImpl@L11"
        )

    def test_no_via_when_annotations_empty(self, db_with_via):
        with patch(
            "plugin_implementation.cluster_expansion.get_feature_flags",
            return_value=FeatureFlags(),
        ):
            pool = _collect_expansion_neighbors(
                _DBStub(db_with_via),
                seed_ids=["n_seed"],
                seen_ids={"n_seed"},
                macro_id=0,
            )
        plain_entries = [t for t in pool if t[0] == "n_plain"]
        assert plain_entries, "Plain neighbour missing from pool"
        _nid, node, _rel = plain_entries[0]
        # No via= on the edge → no _via_context key on the node.
        assert "_via_context" not in node

    def test_node_to_document_surfaces_expansion_via(self):
        node = {
            "node_id": "n_x", "symbol_name": "X", "symbol_type": "class",
            "rel_path": "src/x.py", "file_name": "x.py",
            "source_text": "class X: ...",
            "_via_context": "via src=email@L42",
        }
        doc = _node_to_document(node, is_initial=False, expanded_from="references")
        assert doc.metadata.get("expansion_via") == "via src=email@L42"
        assert doc.metadata.get("expanded_from") == "references"

    def test_node_without_via_does_not_set_expansion_via_key(self):
        node = {
            "node_id": "n_y", "symbol_name": "Y", "symbol_type": "class",
            "rel_path": "src/y.py", "file_name": "y.py",
            "source_text": "class Y: ...",
        }
        doc = _node_to_document(node, is_initial=False, expanded_from="calls")
        assert "expansion_via" not in doc.metadata
