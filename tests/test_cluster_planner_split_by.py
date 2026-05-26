"""Tests for the SPLIT_BY pass in ClusterStructurePlanner (Phase A).

The SPLIT_BY validator decision used to be logged-and-ignored. Phase A
implements it as a per-rel_path partition pass, capped at
``_MAX_SPLIT_SUBPAGES``. These tests pin the partitioning rule directly,
without driving the full planner pipeline (which requires an LLM).

Targets ``ClusterStructurePlanner._apply_split_by_pass`` — the seam
extracted from ``plan_structure`` for unit testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pytest

from plugin_implementation.wiki_structure_planner.cluster_planner import (
    ClusterStructurePlanner,
    _MAX_SPLIT_SUBPAGES,
)
from plugin_implementation.wiki_structure_planner.page_validator import (
    KEEP, SPLIT_BY,
)


# ─── helpers ──────────────────────────────────────────────────────────

@dataclass
class _StubCandidate:
    macro_id: int
    micro_id: int


@dataclass
class _StubValidation:
    candidate: _StubCandidate
    shape_decision: str


class _NodeMapDB:
    """Minimal stub: only ``get_node`` is exercised by SPLIT_BY."""

    def __init__(self, rel_paths: Dict[str, str]):
        # node_id → rel_path
        self._rel = rel_paths

    def get_node(self, nid: str) -> Dict[str, str]:
        return {"rel_path": self._rel.get(nid, "")}


def _planner(db) -> ClusterStructurePlanner:
    """Build a planner without invoking ``__init__`` (avoids LLM/db deps).

    SPLIT_BY only touches ``self.db`` — bypassing the constructor keeps
    the test focused on the partition rule.
    """
    p = ClusterStructurePlanner.__new__(ClusterStructurePlanner)
    p.db = db
    return p


# ─── 1. Multi-file split ──────────────────────────────────────────────

class TestSplitByMultiFile:

    def test_three_files_partition_into_three_subpages(self):
        rel_paths = {
            "n1": "src/auth.py", "n2": "src/auth.py",
            "n3": "src/session.py", "n4": "src/session.py",
            "n5": "src/tokens.py",
        }
        cluster_map = {0: {0: list(rel_paths.keys())}}
        validations = [
            _StubValidation(_StubCandidate(0, 0), SPLIT_BY),
        ]

        split_count = _planner(_NodeMapDB(rel_paths))._apply_split_by_pass(
            cluster_map, validations,
        )

        assert split_count == 1
        # Three sub-pages, one per rel_path. Note: the planner reuses
        # the freed micro_id (next_pid = max(empty)+1 = 0) so the
        # original pid identity may be reassigned to a sub-page — the
        # contract is on partition shape, not pid uniqueness.
        assert len(cluster_map[0]) == 3
        # Each sub-page's nids share one rel_path.
        observed_paths = set()
        for _pid, nids in cluster_map[0].items():
            paths = {rel_paths[n] for n in nids}
            assert len(paths) == 1, f"Sub-page mixes files: {paths}"
            observed_paths.update(paths)
        assert observed_paths == {"src/auth.py", "src/session.py", "src/tokens.py"}

    def test_subpages_ordered_by_descending_size(self):
        # auth has 3 nodes, session has 2, tokens has 1.
        rel_paths = {
            "a1": "auth.py", "a2": "auth.py", "a3": "auth.py",
            "s1": "session.py", "s2": "session.py",
            "t1": "tokens.py",
        }
        cluster_map = {0: {0: list(rel_paths.keys())}}
        validations = [_StubValidation(_StubCandidate(0, 0), SPLIT_BY)]

        _planner(_NodeMapDB(rel_paths))._apply_split_by_pass(
            cluster_map, validations,
        )

        sizes = [len(nids) for _, nids in sorted(cluster_map[0].items())]
        assert sizes == sorted(sizes, reverse=True), (
            f"Sub-pages should be assigned in descending-size order, got {sizes}"
        )


# ─── 2. Single-file SPLIT_BY is a no-op ───────────────────────────────

class TestSplitBySingleFile:

    def test_single_file_split_left_untouched(self):
        rel_paths = {f"n{i}": "src/big_utility.py" for i in range(8)}
        cluster_map = {0: {0: list(rel_paths.keys())}}
        validations = [_StubValidation(_StubCandidate(0, 0), SPLIT_BY)]

        split_count = _planner(_NodeMapDB(rel_paths))._apply_split_by_pass(
            cluster_map, validations,
        )

        assert split_count == 0
        # Unchanged.
        assert cluster_map[0] == {0: list(rel_paths.keys())}


# ─── 3. _MAX_SPLIT_SUBPAGES cap ───────────────────────────────────────

class TestSplitByCap:

    def test_more_files_than_cap_merge_into_largest(self):
        # 7 distinct files; cap is 5 → smallest 2 merge into the largest.
        # Largest: f0 (5 nodes). Smaller: f1..f6 (1 node each).
        rel_paths: Dict[str, str] = {}
        for i in range(5):
            rel_paths[f"f0_{i}"] = "f0.py"
        for j in range(1, 7):  # f1..f6
            rel_paths[f"f{j}_0"] = f"f{j}.py"
        cluster_map = {0: {0: list(rel_paths.keys())}}
        validations = [_StubValidation(_StubCandidate(0, 0), SPLIT_BY)]

        _planner(_NodeMapDB(rel_paths))._apply_split_by_pass(
            cluster_map, validations,
        )

        # Capped at _MAX_SPLIT_SUBPAGES.
        assert len(cluster_map[0]) == _MAX_SPLIT_SUBPAGES
        # Largest sub-page absorbs the overflow files (5 + 2 = 7 nids).
        sizes = sorted([len(nids) for nids in cluster_map[0].values()],
                       reverse=True)
        # Top sub-page has at least the 5 original f0 nodes plus the 2
        # overflow nodes from f5, f6 (the smallest by ordering).
        assert sizes[0] >= 5 + (7 - _MAX_SPLIT_SUBPAGES)
        # Total nid count must be conserved across all sub-pages.
        total_after = sum(len(nids) for nids in cluster_map[0].values())
        assert total_after == len(rel_paths)


# ─── 4. Validations that are not SPLIT_BY are ignored ─────────────────

class TestValidationFiltering:

    def test_keep_decision_does_not_split(self):
        rel_paths = {
            "n1": "a.py", "n2": "b.py", "n3": "c.py",
        }
        cluster_map = {0: {0: list(rel_paths.keys())}}
        validations = [_StubValidation(_StubCandidate(0, 0), KEEP)]

        split_count = _planner(_NodeMapDB(rel_paths))._apply_split_by_pass(
            cluster_map, validations,
        )

        assert split_count == 0
        assert cluster_map[0] == {0: list(rel_paths.keys())}

    def test_validation_for_missing_cluster_silently_skipped(self):
        cluster_map: Dict[int, Dict[int, List[str]]] = {0: {0: ["x"]}}
        # Validation references a nonexistent (1, 1) cluster.
        validations = [_StubValidation(_StubCandidate(1, 1), SPLIT_BY)]

        split_count = _planner(_NodeMapDB({"x": "x.py"}))._apply_split_by_pass(
            cluster_map, validations,
        )

        assert split_count == 0
        assert cluster_map == {0: {0: ["x"]}}

    def test_node_with_unknown_rel_path_assigned_unknown_bucket(self):
        # Two distinct files; node n3 has no rel_path → goes into
        # ``_unknown`` bucket, which still counts as a third partition.
        rel_paths = {"n1": "a.py", "n2": "b.py"}  # n3 omitted
        cluster_map = {0: {0: ["n1", "n2", "n3"]}}
        validations = [_StubValidation(_StubCandidate(0, 0), SPLIT_BY)]

        _planner(_NodeMapDB(rel_paths))._apply_split_by_pass(
            cluster_map, validations,
        )

        # Three sub-pages: a.py, b.py, _unknown.
        assert len(cluster_map[0]) == 3
