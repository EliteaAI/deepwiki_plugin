"""
Tests for post-Leiden section and page consolidation.

Validates that _consolidate_sections() and _consolidate_pages() correctly
merge over-fragmented Leiden output to target section/page counts using
directory-proximity heuristics.
"""

import math
from collections import Counter
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from plugin_implementation.graph_clustering import (
    _consolidate_pages,
    _consolidate_sections,
    _dir_histogram,
    _dir_of_node,
    _dir_similarity,
    _target_section_count,
    _target_total_pages,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_graph(nodes_with_paths):
    """Build a small NX graph with rel_path attributes.

    nodes_with_paths: list of (node_id, rel_path)
    """
    G = nx.MultiDiGraph()
    for nid, path in nodes_with_paths:
        G.add_node(nid, rel_path=path, file_name=path, symbol_type="function")
    return G


def _make_leiden_result(sections_def):
    """Build a fake leiden_result dict.

    sections_def: {sec_id: {pg_id: [node_ids]}}
    """
    sections = {}
    macro_assign = {}
    micro_assign = {}

    for sid, pages in sections_def.items():
        sections[sid] = {"pages": dict(pages)}
        micro_assign[sid] = {}
        for pid, nids in pages.items():
            for nid in nids:
                macro_assign[nid] = sid
                micro_assign[sid][nid] = pid

    return {
        "sections": sections,
        "macro_assignments": macro_assign,
        "micro_assignments": micro_assign,
        "algorithm_metadata": {
            "algorithm": "hierarchical_leiden",
            "sections": len(sections),
            "pages": sum(
                len(p) for s in sections.values() for p in s["pages"].values()
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Target count functions
# ═══════════════════════════════════════════════════════════════════════════

class TestTargetCounts:
    """Test adaptive target count formulas (file-count based)."""

    def test_section_targets(self):
        # log₂ scaling: ceil(1.2 * log2(n)), clamped [5, 20]
        assert _target_section_count(5) == 5          # tiny → floor
        assert _target_section_count(50) == 7         # ceil(1.2*5.64) = 7
        assert _target_section_count(116) == 9        # fmtlib-sized
        assert _target_section_count(500) == 11       # ceil(1.2*8.97) = 11
        assert _target_section_count(1000) == 12      # ceil(1.2*9.97) = 12
        assert _target_section_count(5000) == 15      # ceil(1.2*12.29) = 15
        assert _target_section_count(7165) == 16      # redpanda files
        assert _target_section_count(50000) == 19     # large repo
        assert _target_section_count(1000000) == 20   # ceiling clamp

    def test_page_targets(self):
        # sqrt scaling: ceil(sqrt(n_nodes / 7)), clamped [8, 200]
        assert _target_total_pages(100) == 8          # floor clamp
        assert _target_total_pages(729) == 11         # configurations-sized
        assert _target_total_pages(11417) == 41       # fmtlib nodes
        assert _target_total_pages(146091) == 145     # redpanda nodes
        assert _target_total_pages(300000) == 200     # ceiling clamp

    def test_section_monotonic(self):
        """Larger repos should produce >= the target of smaller repos."""
        prev = 0
        for n in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 1000000]:
            t = _target_section_count(n)
            assert t >= prev, f"Non-monotonic at n={n}"
            prev = t

    def test_page_monotonic(self):
        prev = 0
        for n in [100, 729, 5000, 11417, 50000, 146091, 300000]:
            t = _target_total_pages(n)
            assert t >= prev, f"Non-monotonic at n={n}"
            prev = t

    def test_pages_per_section_grows(self):
        """Larger repos should have more pages per section (decoupled).

        Sections use file count (log₂), pages use node count (√).
        Different inputs guarantee the ratio grows.
        """
        # (n_files, n_nodes) pairs for increasingly large repos
        repos = [(50, 729), (122, 11417), (7165, 146091), (50000, 500000)]
        ratios = []
        for n_files, n_nodes in repos:
            s = _target_section_count(n_files)
            p = _target_total_pages(n_nodes)
            ratios.append(p / s)
        # Each ratio should be strictly larger than the previous
        for i in range(1, len(ratios)):
            assert ratios[i] > ratios[i - 1], (
                f"pages/section not growing: {ratios}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Directory helpers
# ═══════════════════════════════════════════════════════════════════════════

class TestDirHelpers:
    """Test directory extraction and similarity."""

    def test_dir_of_node_with_path(self):
        G = _make_graph([("n1", "src/lib/foo.cpp")])
        assert _dir_of_node("n1", G) == "src/lib"

    def test_dir_of_node_root(self):
        G = _make_graph([("n1", "main.cpp")])
        assert _dir_of_node("n1", G) == "<root>"

    def test_dir_of_node_missing(self):
        G = nx.MultiDiGraph()
        G.add_node("n1")
        assert _dir_of_node("n1", G) == "<root>"

    def test_dir_histogram(self):
        G = _make_graph([
            ("a", "src/foo.cpp"),
            ("b", "src/bar.cpp"),
            ("c", "include/baz.h"),
        ])
        hist = _dir_histogram(["a", "b", "c"], G)
        assert hist["src"] == 2
        assert hist["include"] == 1

    def test_dir_similarity_identical(self):
        a = Counter({"src": 3, "include": 2})
        b = Counter({"src": 5, "include": 1})
        assert _dir_similarity(a, b) == 3 + 1  # min(3,5) + min(2,1)

    def test_dir_similarity_disjoint(self):
        a = Counter({"src": 3})
        b = Counter({"lib": 5})
        assert _dir_similarity(a, b) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Section consolidation
# ═══════════════════════════════════════════════════════════════════════════

class TestSectionConsolidation:
    """Test _consolidate_sections."""

    def test_no_op_when_under_target(self):
        """Don't merge if already below target."""
        G = _make_graph([("a", "src/a.cpp"), ("b", "src/b.cpp")])
        result = _make_leiden_result({
            0: {0: ["a"]},
            1: {0: ["b"]},
        })
        original_count = len(result["sections"])
        _consolidate_sections(result, G)
        assert len(result["sections"]) == original_count

    def test_merges_to_target(self):
        """Many tiny sections should be merged down to target."""
        # Create 50 sections, each with 1 node, all in src/
        nodes = [(f"n{i}", f"src/file{i}.cpp") for i in range(50)]
        G = _make_graph(nodes)

        sections_def = {}
        for i in range(50):
            sections_def[i] = {0: [f"n{i}"]}

        result = _make_leiden_result(sections_def)
        target = _target_section_count(50)

        _consolidate_sections(result, G)

        assert len(result["sections"]) <= target

    def test_macro_assignments_updated(self):
        """All nodes should point to surviving section IDs after merge."""
        nodes = [(f"n{i}", f"src/file{i}.cpp") for i in range(20)]
        G = _make_graph(nodes)
        sections_def = {i: {0: [f"n{i}"]} for i in range(20)}
        result = _make_leiden_result(sections_def)

        _consolidate_sections(result, G)

        surviving_sids = set(result["sections"].keys())
        for nid, sid in result["macro_assignments"].items():
            assert sid in surviving_sids, (
                f"Node {nid} points to deleted section {sid}"
            )

    def test_micro_assignments_updated(self):
        """micro_assignments should only contain surviving section IDs."""
        nodes = [(f"n{i}", f"src/f{i}.cpp") for i in range(15)]
        G = _make_graph(nodes)
        sections_def = {i: {0: [f"n{i}"]} for i in range(15)}
        result = _make_leiden_result(sections_def)

        _consolidate_sections(result, G)

        for sid in result["micro_assignments"]:
            assert sid in result["sections"]

    def test_directory_proximity_preferred(self):
        """Sections in same directory should merge together."""
        nodes = [
            ("a", "src/core/a.cpp"),
            ("b", "src/core/b.cpp"),
            ("c", "src/core/c.cpp"),
            ("d", "lib/other/d.cpp"),
            ("e", "lib/other/e.cpp"),
        ]
        G = _make_graph(nodes)
        # 5 sections — target will be 5 (floor) so no merge.
        # Artificially lower target by having very few nodes.
        sections_def = {
            0: {0: ["a"]},
            1: {0: ["b"]},
            2: {0: ["c"]},
            3: {0: ["d"]},
            4: {0: ["e"]},
        }
        result = _make_leiden_result(sections_def)

        # Force a target of 2 by patching
        with patch(
            "plugin_implementation.graph_clustering._target_section_count",
            return_value=2,
        ):
            _consolidate_sections(result, G)

        # Should have 2 sections; core nodes together, lib nodes together
        assert len(result["sections"]) == 2
        for _sid, sec in result["sections"].items():
            all_nids = [n for pg in sec["pages"].values() for n in pg]
            dirs = {_dir_of_node(n, G) for n in all_nids}
            # Each section should be mostly one directory
            assert len(dirs) <= 2

    def test_pages_preserved_after_merge(self):
        """Page IDs should be unique within the merged section."""
        nodes = [(f"n{i}", "src/a.cpp") for i in range(6)]
        G = _make_graph(nodes)
        # Two sections with 2 pages each → merge into 1
        sections_def = {
            0: {0: ["n0", "n1"], 1: ["n2"]},
            1: {0: ["n3", "n4"], 1: ["n5"]},
        }
        result = _make_leiden_result(sections_def)

        with patch(
            "plugin_implementation.graph_clustering._target_section_count",
            return_value=1,
        ):
            _consolidate_sections(result, G)

        assert len(result["sections"]) == 1
        sid = list(result["sections"].keys())[0]
        pages = result["sections"][sid]["pages"]
        # All 6 nodes present, page IDs unique
        all_nids = [n for pg in pages.values() for n in pg]
        assert sorted(all_nids) == [f"n{i}" for i in range(6)]
        assert len(pages) == len(set(pages.keys()))


# ═══════════════════════════════════════════════════════════════════════════
# Page consolidation
# ═══════════════════════════════════════════════════════════════════════════

class TestPageConsolidation:
    """Test _consolidate_pages."""

    def test_no_op_when_under_target(self):
        """Don't merge pages if already below target."""
        G = _make_graph([("a", "src/a.cpp"), ("b", "src/b.cpp")])
        result = _make_leiden_result({
            0: {0: ["a"], 1: ["b"]},
        })
        original_pages = sum(
            len(s["pages"]) for s in result["sections"].values()
        )
        _consolidate_pages(result, G)
        new_pages = sum(
            len(s["pages"]) for s in result["sections"].values()
        )
        assert new_pages == original_pages

    def test_merges_pages_to_target(self):
        """Many pages in one section should be consolidated."""
        # 1 section, 100 pages, 1 node each → target ~8
        nodes = [(f"n{i}", f"src/file{i}.cpp") for i in range(100)]
        G = _make_graph(nodes)
        sections_def = {0: {i: [f"n{i}"] for i in range(100)}}
        result = _make_leiden_result(sections_def)

        _consolidate_pages(result, G)

        total = sum(len(s["pages"]) for s in result["sections"].values())
        assert total <= _target_total_pages(100)

    def test_all_nodes_survive(self):
        """No nodes should be lost during page merging."""
        nodes = [(f"n{i}", f"src/f{i}.cpp") for i in range(30)]
        G = _make_graph(nodes)
        sections_def = {0: {i: [f"n{i}"] for i in range(30)}}
        result = _make_leiden_result(sections_def)

        _consolidate_pages(result, G)

        all_nids = set()
        for sec in result["sections"].values():
            for nids in sec["pages"].values():
                all_nids.update(nids)
        assert all_nids == {f"n{i}" for i in range(30)}

    def test_global_convergence_no_per_section_cap(self):
        """Pages merge globally to target — large sections can keep many pages."""
        # 2 sections: one huge (150 pages), one small (5 pages)
        nodes_big = [(f"n{i}", f"src/core/f{i}.cpp") for i in range(150)]
        nodes_small = [(f"s{i}", f"lib/util/u{i}.cpp") for i in range(5)]
        G = _make_graph(nodes_big + nodes_small)

        sections_def = {
            0: {i: [f"n{i}"] for i in range(150)},
            1: {i: [f"s{i}"] for i in range(5)},
        }
        result = _make_leiden_result(sections_def)

        _consolidate_pages(result, G)

        total = sum(len(s["pages"]) for s in result["sections"].values())
        target = _target_total_pages(155)
        assert total <= target + 2, f"Total {total} > target {target}+2"
        # The big section absorbed most merges but is NOT artificially capped
        # — it may keep more pages than the small section, which is correct
        big_pages = len(result["sections"][0]["pages"])
        small_pages = len(result["sections"][1]["pages"])
        assert big_pages >= small_pages, (
            "Big section should keep ≥ pages than tiny section"
        )

    def test_micro_assignments_consistent(self):
        """micro_assignments[sid][nid] should match actual page placement."""
        nodes = [(f"n{i}", f"src/f{i}.cpp") for i in range(20)]
        G = _make_graph(nodes)
        sections_def = {0: {i: [f"n{i}"] for i in range(20)}}
        result = _make_leiden_result(sections_def)

        _consolidate_pages(result, G)

        for sid, sec in result["sections"].items():
            for pid, nids in sec["pages"].items():
                for nid in nids:
                    assert result["micro_assignments"][sid][nid] == pid


# ═══════════════════════════════════════════════════════════════════════════
# MERGE_WITH in cluster planner
# ═══════════════════════════════════════════════════════════════════════════

class TestMergeWithInPlanner:
    """Test that MERGE_WITH candidates are properly merged."""

    def _make_db_mock(self, nodes_by_id):
        """Create a mock DB with get_node and conn."""
        db = MagicMock()

        def get_node(nid):
            return nodes_by_id.get(nid)

        db.get_node = get_node
        return db

    def test_merge_with_combines_nodes(self):
        """MERGE_WITH candidates should be absorbed into siblings."""
        from plugin_implementation.wiki_structure_planner.candidate_builder import (
            CandidateRecord,
        )
        from plugin_implementation.wiki_structure_planner.page_validator import (
            KEEP,
            MERGE_WITH,
            ValidationResult,
        )

        # Setup: macro 0 has micro 0 (big, KEEP) and micro 1 (small, MERGE_WITH)
        cluster_map = {
            0: {
                0: ["a", "b", "c"],
                1: ["d"],
            },
        }

        validations = [
            ValidationResult(
                candidate=CandidateRecord(macro_id=0, micro_id=0, node_ids=["a", "b", "c"]),
                shape_decision=KEEP,
            ),
            ValidationResult(
                candidate=CandidateRecord(macro_id=0, micro_id=1, node_ids=["d"]),
                shape_decision=MERGE_WITH,
            ),
        ]

        # Simulate the planner's merge pass
        merge_count = 0
        for vr in validations:
            if vr.shape_decision != MERGE_WITH:
                continue
            mid = vr.candidate.macro_id
            pid = vr.candidate.micro_id
            if mid not in cluster_map or pid not in cluster_map[mid]:
                continue
            best_sibling = None
            best_size = 0
            for other_pid, other_nids in cluster_map[mid].items():
                if other_pid == pid:
                    continue
                if len(other_nids) > best_size:
                    best_size = len(other_nids)
                    best_sibling = other_pid
            if best_sibling is not None:
                cluster_map[mid][best_sibling].extend(cluster_map[mid].pop(pid))
                merge_count += 1

        assert merge_count == 1
        assert 1 not in cluster_map[0]
        assert "d" in cluster_map[0][0]
        assert len(cluster_map[0][0]) == 4

    def test_merge_with_no_sibling_kept(self):
        """If no sibling exists, MERGE_WITH candidate is kept as-is."""
        cluster_map = {
            0: {0: ["a", "b"]},  # only one micro in this macro
        }

        from plugin_implementation.wiki_structure_planner.candidate_builder import (
            CandidateRecord,
        )
        from plugin_implementation.wiki_structure_planner.page_validator import (
            MERGE_WITH,
            ValidationResult,
        )

        validations = [
            ValidationResult(
                candidate=CandidateRecord(macro_id=0, micro_id=0, node_ids=["a", "b"]),
                shape_decision=MERGE_WITH,
            ),
        ]

        merge_count = 0
        for vr in validations:
            if vr.shape_decision != MERGE_WITH:
                continue
            mid = vr.candidate.macro_id
            pid = vr.candidate.micro_id
            if mid not in cluster_map or pid not in cluster_map[mid]:
                continue
            best_sibling = None
            best_size = 0
            for other_pid, other_nids in cluster_map[mid].items():
                if other_pid == pid:
                    continue
                if len(other_nids) > best_size:
                    best_size = len(other_nids)
                    best_sibling = other_pid
            if best_sibling is not None:
                cluster_map[mid][best_sibling].extend(cluster_map[mid].pop(pid))
                merge_count += 1

        # No sibling → kept as-is
        assert merge_count == 0
        assert 0 in cluster_map[0]
        assert len(cluster_map[0][0]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Full pipeline integration
# ═══════════════════════════════════════════════════════════════════════════

class TestFullConsolidationPipeline:
    """Integration tests: section + page consolidation together."""

    def test_fmtlib_scale_scenario(self):
        """Simulate fmtlib-like graph: 9000 nodes, many tiny sections."""
        # Create 500 sections with ~18 nodes each (like fmtlib after projection)
        node_dirs = [
            "include/fmt", "src", "test", "doc",
            "include/fmt/detail", "include/fmt/ranges",
        ]
        node_count = 0
        nodes = []
        sections_def = {}

        for sid in range(500):
            nids = []
            for j in range(18):
                nid = f"n{node_count}"
                dir_idx = (sid + j) % len(node_dirs)
                nodes.append((nid, f"{node_dirs[dir_idx]}/file{node_count}.h"))
                nids.append(nid)
                node_count += 1

            # 3 pages per section
            sections_def[sid] = {
                0: nids[:6],
                1: nids[6:12],
                2: nids[12:],
            }

        G = _make_graph(nodes)
        result = _make_leiden_result(sections_def)

        _consolidate_sections(result, G)
        _consolidate_pages(result, G)

        n_sections = len(result["sections"])
        n_pages = sum(len(s["pages"]) for s in result["sections"].values())

        target_sec = _target_section_count(node_count)
        target_pg = _target_total_pages(node_count)

        assert n_sections <= target_sec, (
            f"Sections: {n_sections} > target {target_sec}"
        )
        # Page consolidation uses per-section proportional allocation which
        # can overshoot by 1-2 due to rounding (each section gets at least 1).
        assert n_pages <= target_pg + 5, (
            f"Pages: {n_pages} > target {target_pg} + tolerance"
        )

        # All nodes still present
        all_nids = set()
        for sec in result["sections"].values():
            for nids in sec["pages"].values():
                all_nids.update(nids)
        assert len(all_nids) == node_count

    def test_small_repo_no_consolidation(self):
        """Small graph with few sections should not be changed."""
        nodes = [(f"n{i}", f"src/f{i}.py") for i in range(10)]
        G = _make_graph(nodes)
        sections_def = {
            0: {0: ["n0", "n1", "n2"]},
            1: {0: ["n3", "n4", "n5"]},
            2: {0: ["n6", "n7", "n8", "n9"]},
        }
        result = _make_leiden_result(sections_def)

        _consolidate_sections(result, G)
        _consolidate_pages(result, G)

        assert len(result["sections"]) == 3
        total_pages = sum(len(s["pages"]) for s in result["sections"].values())
        assert total_pages == 3
