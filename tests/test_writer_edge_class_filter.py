"""Tests for the writer evidence edge-class gate (Phase B7).

The writer's 1-hop expansion pool is built by
``cluster_expansion._collect_expansion_neighbors``. Phase B7 gates this
pool by ``edge_class`` so synthetic glue (lexical / directory / doc-
proximity / bridge) cannot leak nodes into the LLM prompt context.

These tests pin:
- the canonical edge-class vocabularies in ``constants``,
- the producer audit (every name in ``WRITER_ALLOWED_EDGE_CLASSES`` is
  actually emitted by some producer in this codebase),
- the filter behaviour on a small in-memory DB, with the feature flag
  on (default) and off (legacy fallback).
"""

from __future__ import annotations

import os
import sqlite3
from unittest.mock import patch

import pytest

from plugin_implementation.cluster_expansion import _collect_expansion_neighbors
from plugin_implementation.constants import (
    EXPANSION_SYMBOL_TYPES,
    SYNTHETIC_EDGE_CLASSES,
    WRITER_ALLOWED_EDGE_CLASSES,
)
from plugin_implementation.feature_flags import FeatureFlags


# ─── 1. Vocabulary contracts ──────────────────────────────────────────

class TestEdgeClassVocabulary:

    def test_synthetic_set_is_disjoint_from_allow_set(self):
        """Synthetic edges must never accidentally end up in the allow-list."""
        overlap = SYNTHETIC_EDGE_CLASSES & WRITER_ALLOWED_EDGE_CLASSES
        assert not overlap, f"Synthetic classes leaked into allow-set: {overlap}"

    def test_allow_set_only_contains_emitted_classes(self):
        """Every name in the allow-set must have a real producer in this repo.

        The roadmap mentions ``cross_repo`` and ``member_uses`` as future
        producers; they are intentionally NOT in the allow-set until
        their producers land. This test guards against re-introducing
        ghost names by accident.
        """
        # Verified by ``grep edge_class= --include='*.py'`` on the
        # plugin_implementation tree — see constants.py docstring.
        emitted_in_repo = {
            "structural",
            "cross_language",
            "test_link",
        }
        unverified = WRITER_ALLOWED_EDGE_CLASSES - emitted_in_repo
        assert not unverified, (
            f"Allow-set contains classes with no producer in this repo: "
            f"{unverified}. Either add a producer or remove from the set."
        )

    def test_synthetic_set_matches_topology_emitters(self):
        """The synthetic set must match what ``graph_topology`` actually emits."""
        # graph_topology.py emit sites: directory (458), lexical (518/631/839/896),
        # semantic (576/1000), doc (518/1409/1422), bridge (1596+).
        expected = {"directory", "lexical", "semantic", "doc", "bridge"}
        assert SYNTHETIC_EDGE_CLASSES == expected


# ─── 2. Filter behaviour on a synthetic DB ────────────────────────────

def _make_db():
    """Minimal in-memory DB matching the slice of unified_db used by
    ``_collect_expansion_neighbors``: ``repo_nodes`` + ``repo_edges`` with
    an ``edge_class`` column."""
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


def _add_node(conn, nid, name, sym_type, macro=0):
    conn.execute(
        "INSERT INTO repo_nodes "
        "(node_id, symbol_name, symbol_type, rel_path, file_name, "
        "is_architectural, macro_cluster) VALUES (?,?,?,?,?,?,?)",
        (nid, name, sym_type, f"src/{name}.py", f"{name}.py", 1, macro),
    )


def _add_edge(conn, src, tgt, rel_type, edge_class):
    conn.execute(
        "INSERT INTO repo_edges (source_id, target_id, rel_type, edge_class, weight) "
        "VALUES (?,?,?,?,?)",
        (src, tgt, rel_type, edge_class, 1.0),
    )


class _DBStub:
    def __init__(self, conn):
        self.conn = conn


@pytest.fixture
def synthetic_db():
    conn = _make_db()
    # Seed: AuthService — 5 neighbours wired with mixed edge classes.
    _add_node(conn, "n_seed", "AuthService", "class")
    # Allowed-class neighbours (writer should keep these).
    _add_node(conn, "n_struct", "LoginHandler", "function")
    _add_edge(conn, "n_seed", "n_struct", "calls", "structural")
    _add_node(conn, "n_xlang", "AuthClient", "class")
    _add_edge(conn, "n_seed", "n_xlang", "implements", "cross_language")
    _add_node(conn, "n_test", "AuthServiceTest", "class")
    _add_edge(conn, "n_seed", "n_test", "tests", "test_link")
    # Synthetic-class neighbours (writer must drop these).
    _add_node(conn, "n_lex", "Authenticator", "class")
    _add_edge(conn, "n_seed", "n_lex", "lexical_match", "lexical")
    _add_node(conn, "n_dir", "DirNeighbour", "class")
    _add_edge(conn, "n_seed", "n_dir", "same_dir", "directory")
    _add_node(conn, "n_bridge", "BridgeNeighbour", "class")
    _add_edge(conn, "n_seed", "n_bridge", "bridge", "bridge")
    conn.commit()
    return conn


def _ids_in_pool(pool):
    return {nid for nid, _node, _rel, *_ in pool}


class TestEdgeClassFilter:

    def test_default_filter_drops_synthetic_neighbours(self, synthetic_db):
        with patch(
            "plugin_implementation.cluster_expansion.get_feature_flags",
            return_value=FeatureFlags(writer_edge_class_filter=True),
        ):
            pool = _collect_expansion_neighbors(
                _DBStub(synthetic_db),
                seed_ids=["n_seed"],
                seen_ids={"n_seed"},
                macro_id=0,
            )
        ids = _ids_in_pool(pool)
        assert {"n_struct", "n_xlang", "n_test"}.issubset(ids), (
            f"Allowed neighbours missing from pool: got {ids}"
        )
        for synth in ("n_lex", "n_dir", "n_bridge"):
            assert synth not in ids, (
                f"Synthetic neighbour {synth} leaked through allow-set"
            )

    def test_filter_off_includes_everything(self, synthetic_db):
        # Legacy fallback — flag off restores the pre-B7 inclusive pool.
        with patch(
            "plugin_implementation.cluster_expansion.get_feature_flags",
            return_value=FeatureFlags(writer_edge_class_filter=False),
        ):
            pool = _collect_expansion_neighbors(
                _DBStub(synthetic_db),
                seed_ids=["n_seed"],
                seen_ids={"n_seed"},
                macro_id=0,
            )
        ids = _ids_in_pool(pool)
        assert {
            "n_struct", "n_xlang", "n_test", "n_lex", "n_dir", "n_bridge",
        }.issubset(ids), (
            f"Legacy mode should include synthetic neighbours, got {ids}"
        )

    def test_omitted_edge_class_defaults_to_structural(self, synthetic_db):
        """Producers that don't specify ``edge_class`` end up with the
        schema default ``'structural'`` (unified_db.py:221) — the writer
        must therefore include those neighbours."""
        synthetic_db.execute(
            "INSERT INTO repo_nodes "
            "(node_id, symbol_name, symbol_type, rel_path, file_name, "
            "is_architectural, macro_cluster) "
            "VALUES (?,?,?,?,?,?,?)",
            ("n_default", "DefaultNode", "class", "src/default.py", "default.py", 1, 0),
        )
        # Omit edge_class — INSERT relies on the column DEFAULT.
        synthetic_db.execute(
            "INSERT INTO repo_edges (source_id, target_id, rel_type, weight) "
            "VALUES (?,?,?,?)",
            ("n_seed", "n_default", "calls", 1.0),
        )
        synthetic_db.commit()
        with patch(
            "plugin_implementation.cluster_expansion.get_feature_flags",
            return_value=FeatureFlags(writer_edge_class_filter=True),
        ):
            pool = _collect_expansion_neighbors(
                _DBStub(synthetic_db),
                seed_ids=["n_seed"],
                seen_ids={"n_seed"},
                macro_id=0,
            )
        assert "n_default" in _ids_in_pool(pool)


# ─── 3. Env-flag wiring ───────────────────────────────────────────────

class TestEnvOverride:

    def test_env_var_disables_filter(self, monkeypatch):
        from plugin_implementation.feature_flags import get_feature_flags
        monkeypatch.setenv("DEEPWIKI_WRITER_EDGE_CLASS_FILTER", "0")
        flags = get_feature_flags()
        assert flags.writer_edge_class_filter is False

    def test_env_var_unset_defaults_on(self, monkeypatch):
        from plugin_implementation.feature_flags import get_feature_flags
        monkeypatch.delenv("DEEPWIKI_WRITER_EDGE_CLASS_FILTER", raising=False)
        flags = get_feature_flags()
        assert flags.writer_edge_class_filter is True
