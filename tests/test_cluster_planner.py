"""                                                                              
Tests for Phase 4 — cluster_planner.py
Cluster-based wiki structure planning with two-pass LLM naming.

Suite layout
============
TestDominantSymbols          — scoring and ranking of representative symbols
TestMicroSummaries           — compact summaries for section-naming prompt
TestPageSymbols              — enriched symbols for page-naming prompt
TestGlobalCoreSection        — hub node section building
TestClusterNaming            — two-pass LLM naming with mock
TestPlanStructure            — full plan_structure() pipeline
TestFallbacks                — fallback spec, fallback section, LLM failure
TestJsonParsing              — _parse_json_response edge cases
TestEdgeCases                — empty DB, single cluster, etc.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugin_implementation.graph_clustering import run_phase3
from plugin_implementation.graph_topology import run_phase2
from plugin_implementation.state.wiki_state import (
    PageSpec,
    SectionSpec,
    WikiStructureSpec,
)
from plugin_implementation.unified_db import UnifiedWikiDB
from plugin_implementation.wiki_structure_planner.cluster_planner import (
    CLUSTER_NAMING_SYSTEM,
    CLUSTER_NAMING_USER,
    SECTION_NAMING_SYSTEM,
    PAGE_NAMING_SYSTEM,
    ClusterStructurePlanner,
    _extract_text,
    _parse_json_response,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_node_dict(node_id: str, **kwargs) -> Dict[str, Any]:
    d = {
        "node_id": node_id,
        "symbol_name": kwargs.get("symbol_name", node_id),
        "symbol_type": kwargs.get("symbol_type", "function"),
        "rel_path": kwargs.get("rel_path", "src/main.py"),
        "source_text": kwargs.get("source_text", f"def {node_id}(): pass"),
        "is_architectural": kwargs.get("is_architectural", 1),
        "language": kwargs.get("language", "python"),
        "docstring": kwargs.get("docstring", ""),
        "signature": kwargs.get("signature", f"def {node_id}()"),
    }
    d.update(kwargs)
    return d


def _make_db(tmp_dir: str, nodes=None, edges=None) -> UnifiedWikiDB:
    db_path = os.path.join(tmp_dir, "test.wiki.db")
    db = UnifiedWikiDB(db_path)
    if nodes:
        db.upsert_nodes_batch(nodes)
    if edges:
        db.upsert_edges_batch(edges)
    return db


def _build_two_community_db(tmp_dir: str):
    """Build a DB with two communities (6 nodes each) + a hub, fully clustered."""
    G = nx.MultiDiGraph()
    nodes_data = []

    # Community A — auth module
    for i in range(6):
        nid = f"auth_{i}"
        nodes_data.append(_make_node_dict(
            nid, symbol_name=f"AuthClass{i}", rel_path=f"src/auth/{nid}.py",
            symbol_type="class", docstring=f"Authentication class {i}",
        ))
        for j in range(6):
            if i != j:
                G.add_edge(f"auth_{i}", f"auth_{j}", weight=1.0, relationship_type="calls")

    # Community B — data module
    for i in range(6):
        nid = f"data_{i}"
        nodes_data.append(_make_node_dict(
            nid, symbol_name=f"DataHandler{i}", rel_path=f"src/data/{nid}.py",
            symbol_type="class", docstring=f"Data handler {i}",
        ))
        for j in range(6):
            if i != j:
                G.add_edge(f"data_{i}", f"data_{j}", weight=1.0, relationship_type="calls")

    # Hub — logger
    nodes_data.append(_make_node_dict(
        "hub_logger", symbol_name="Logger", rel_path="src/utils/logger.py",
        symbol_type="class", docstring="Central logging utility",
    ))
    for i in range(6):
        G.add_edge(f"auth_{i}", "hub_logger", weight=0.5, relationship_type="calls")
        G.add_edge(f"data_{i}", "hub_logger", weight=0.5, relationship_type="calls")

    # Thin bridge
    G.add_edge("auth_0", "data_0", weight=0.01, relationship_type="calls")

    db = _make_db(tmp_dir, nodes=nodes_data)
    db.set_meta("repo_identifier", "test-org/test-repo")

    # Run Phases 2 + 3
    from plugin_implementation.graph_topology import apply_edge_weights, detect_hubs
    apply_edge_weights(G)
    hubs = detect_hubs(G, z_threshold=2.0)
    p2 = run_phase2(db, G, embedding_fn=None)
    hub_ids = set(p2.get("hubs", {}).get("node_ids", []))
    run_phase3(db, G, hubs=hub_ids)

    return db, G


def _mock_llm_response(section_name="Authentication", pages=None):
    """Create a mock LLM that handles two-pass naming.

    The first call returns section naming; subsequent calls return page naming.
    If ``pages`` is given, each entry is returned for successive page-naming calls.
    """
    if pages is None:
        pages = [{"page_name": "Auth Core", "description": "Core auth", "retrieval_query": "auth core"}]

    call_count = [0]

    def mock_invoke(messages, **kwargs):
        call_count[0] += 1
        resp = MagicMock()
        if call_count[0] == 1:
            # Pass 1: section naming
            resp.content = json.dumps({
                "section_name": section_name,
                "section_description": f"{section_name} module",
            })
        else:
            # Pass 2: page naming (call_count-2 = page index)
            page_idx = call_count[0] - 2
            if page_idx < len(pages):
                resp.content = json.dumps(pages[page_idx])
            else:
                resp.content = json.dumps({
                    "page_name": f"Page {page_idx}",
                    "description": "Auto-generated",
                    "retrieval_query": "auto",
                })
        return resp

    mock = MagicMock()
    mock.invoke.side_effect = mock_invoke
    return mock


# ═══════════════════════════════════════════════════════════════════════════
# 1. Dominant Symbols
# ═══════════════════════════════════════════════════════════════════════════

class TestDominantSymbols(unittest.TestCase):

    def test_top_n_symbols_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                _make_node_dict(f"n{i}", docstring=f"doc {i}" if i < 5 else "")
                for i in range(15)
            ]
            db = _make_db(tmp, nodes=nodes)
            # Assign all to macro 0
            for i in range(15):
                db.set_cluster(f"n{i}", macro=0, micro=0)
            db.conn.commit()

            planner = ClusterStructurePlanner(db, _mock_llm_response())
            dominant = planner._get_dominant_symbols(0)

            self.assertLessEqual(len(dominant), 10)
            self.assertTrue(all("name" in d for d in dominant))
            self.assertTrue(all("type" in d for d in dominant))

    def test_architectural_nodes_ranked_higher(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                _make_node_dict("arch", is_architectural=1, symbol_type="class"),
                _make_node_dict("non_arch", is_architectural=0, symbol_type="variable"),
            ]
            db = _make_db(tmp, nodes=nodes)
            db.set_cluster("arch", macro=0, micro=0)
            db.set_cluster("non_arch", macro=0, micro=0)
            db.conn.commit()

            planner = ClusterStructurePlanner(db, _mock_llm_response())
            dominant = planner._get_dominant_symbols(0)

            self.assertEqual(dominant[0]["name"], "arch")

    def test_empty_cluster(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(tmp)
            planner = ClusterStructurePlanner(db, _mock_llm_response())
            dominant = planner._get_dominant_symbols(99)
            self.assertEqual(dominant, [])


# ═══════════════════════════════════════════════════════════════════════════
# 2. Micro Summaries
# ═══════════════════════════════════════════════════════════════════════════

class TestMicroSummaries(unittest.TestCase):

    def test_summary_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict(f"n{i}", rel_path=f"src/mod/{i}.py") for i in range(4)]
            db = _make_db(tmp, nodes=nodes)

            planner = ClusterStructurePlanner(db, _mock_llm_response())
            micro_map = {0: ["n0", "n1"], 1: ["n2", "n3"]}
            summaries = planner._get_micro_summaries(micro_map)

            self.assertEqual(len(summaries), 2)
            self.assertEqual(summaries[0]["micro_id"], 0)
            self.assertEqual(summaries[0]["symbol_count"], 2)
            self.assertTrue(len(summaries[0]["symbols"]) > 0)

    def test_caps_symbol_count(self):
        """Should not send more than MAX_MICRO_SUMMARY_SYMBOLS per micro."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict(f"n{i}") for i in range(20)]
            db = _make_db(tmp, nodes=nodes)

            planner = ClusterStructurePlanner(db, _mock_llm_response())
            micro_map = {0: [f"n{i}" for i in range(20)]}
            summaries = planner._get_micro_summaries(micro_map)

            self.assertEqual(summaries[0]["symbol_count"], 20)
            self.assertLessEqual(len(summaries[0]["symbols"]), 8)


# ═══════════════════════════════════════════════════════════════════════════
# 2b. Page Symbols (enriched, for Pass 2)
# ═══════════════════════════════════════════════════════════════════════════

class TestPageSymbols(unittest.TestCase):

    def test_returns_enriched_symbols(self):
        """Page symbols include signature and docstring."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                _make_node_dict(f"n{i}", docstring=f"Docstring {i}",
                                signature=f"def func{i}()")
                for i in range(5)
            ]
            db = _make_db(tmp, nodes=nodes)

            planner = ClusterStructurePlanner(db, _mock_llm_response())
            page_syms = planner._get_page_symbols([f"n{i}" for i in range(5)])

            self.assertGreater(len(page_syms), 0)
            self.assertIn("signature", page_syms[0])
            self.assertIn("docstring", page_syms[0])
            self.assertIn("name", page_syms[0])
            self.assertIn("type", page_syms[0])

    def test_caps_at_max(self):
        """Should not return more than MAX_PAGE_NAMING_SYMBOLS."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict(f"n{i}") for i in range(20)]
            db = _make_db(tmp, nodes=nodes)

            planner = ClusterStructurePlanner(db, _mock_llm_response())
            page_syms = planner._get_page_symbols([f"n{i}" for i in range(20)])

            self.assertLessEqual(len(page_syms), planner.MAX_PAGE_NAMING_SYMBOLS)

    def test_empty_node_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(tmp)
            planner = ClusterStructurePlanner(db, _mock_llm_response())
            page_syms = planner._get_page_symbols([])
            self.assertEqual(page_syms, [])

    def test_only_architectural_nodes(self):
        """Non-architectural nodes are excluded."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                _make_node_dict("arch", is_architectural=1, symbol_type="class"),
                _make_node_dict("non_arch", is_architectural=0, symbol_type="variable"),
            ]
            db = _make_db(tmp, nodes=nodes)
            planner = ClusterStructurePlanner(db, _mock_llm_response())
            page_syms = planner._get_page_symbols(["arch", "non_arch"])
            self.assertEqual(len(page_syms), 1)
            self.assertEqual(page_syms[0]["name"], "arch")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Cluster Naming (LLM mock)
# ═══════════════════════════════════════════════════════════════════════════

class TestClusterNaming(unittest.TestCase):

    def test_naming_produces_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict(f"n{i}", rel_path=f"src/auth/{i}.py") for i in range(5)]
            db = _make_db(tmp, nodes=nodes)
            for i in range(5):
                db.set_cluster(f"n{i}", macro=0, micro=0)
            db.conn.commit()

            mock_llm = _mock_llm_response(
                section_name="Authentication",
                pages=[
                    {"page_name": "Auth Core", "description": "Core auth logic", "retrieval_query": "auth core"},
                ],
            )

            planner = ClusterStructurePlanner(db, mock_llm)
            micro_map = {0: [f"n{i}" for i in range(5)]}
            section = planner._name_macro_cluster(0, micro_map, section_order=1)

            self.assertEqual(section.section_name, "Authentication")
            self.assertEqual(len(section.pages), 1)
            self.assertEqual(section.pages[0].page_name, "Auth Core")
            self.assertTrue(len(section.pages[0].target_symbols) > 0)

    def test_two_pass_calls_llm_separately(self):
        """Section naming and page naming are separate LLM calls."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict("n0")]
            db = _make_db(tmp, nodes=nodes)
            db.set_cluster("n0", macro=0, micro=0)
            db.conn.commit()

            mock_llm = _mock_llm_response()
            planner = ClusterStructurePlanner(db, mock_llm)
            planner._name_macro_cluster(0, {0: ["n0"]}, section_order=1)

            # 1 micro-cluster → 1 section call + 1 page call = 2 total
            self.assertEqual(mock_llm.invoke.call_count, 2)

            # First call should use SECTION_NAMING_SYSTEM
            section_msgs = mock_llm.invoke.call_args_list[0][0][0]
            self.assertIn("capability name", section_msgs[0].content.lower())

            # Second call should use PAGE_NAMING_SYSTEM
            page_msgs = mock_llm.invoke.call_args_list[1][0][0]
            self.assertIn("page titles", page_msgs[0].content.lower())

    def test_page_naming_gets_only_page_symbols(self):
        """Pass 2 should receive only the micro-cluster's own symbols, not macro dominants."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                _make_node_dict(f"auth_{i}", symbol_name=f"AuthClass{i}",
                                rel_path=f"src/auth/{i}.py", symbol_type="class")
                for i in range(3)
            ] + [
                _make_node_dict(f"data_{i}", symbol_name=f"DataHandler{i}",
                                rel_path=f"src/data/{i}.py", symbol_type="class")
                for i in range(3)
            ]
            db = _make_db(tmp, nodes=nodes)
            for i in range(3):
                db.set_cluster(f"auth_{i}", macro=0, micro=0)
                db.set_cluster(f"data_{i}", macro=0, micro=1)
            db.conn.commit()

            mock_llm = _mock_llm_response(
                section_name="Auth & Data",
                pages=[
                    {"page_name": "Authentication", "description": "Auth", "retrieval_query": "auth"},
                    {"page_name": "Data Handling", "description": "Data", "retrieval_query": "data"},
                ],
            )
            planner = ClusterStructurePlanner(db, mock_llm)
            micro_map = {
                0: [f"auth_{i}" for i in range(3)],
                1: [f"data_{i}" for i in range(3)],
            }
            section = planner._name_macro_cluster(0, micro_map, section_order=1)

            # 1 section call + 2 page calls = 3 total
            self.assertEqual(mock_llm.invoke.call_count, 3)

            # Check page-naming call for micro=0 has auth symbols, not data symbols
            page0_msg = mock_llm.invoke.call_args_list[1][0][0][1].content
            self.assertIn("AuthClass", page0_msg)
            self.assertNotIn("DataHandler", page0_msg)

            # Check page-naming call for micro=1 has data symbols, not auth symbols
            page1_msg = mock_llm.invoke.call_args_list[2][0][0][1].content
            self.assertIn("DataHandler", page1_msg)
            self.assertNotIn("AuthClass", page1_msg)

    def test_multiple_micro_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict(f"n{i}") for i in range(6)]
            db = _make_db(tmp, nodes=nodes)
            for i in range(3):
                db.set_cluster(f"n{i}", macro=0, micro=0)
            for i in range(3, 6):
                db.set_cluster(f"n{i}", macro=0, micro=1)
            db.conn.commit()

            mock_llm = _mock_llm_response(
                pages=[
                    {"page_name": "Page A", "description": "A", "retrieval_query": "a"},
                    {"page_name": "Page B", "description": "B", "retrieval_query": "b"},
                ],
            )

            planner = ClusterStructurePlanner(db, mock_llm)
            micro_map = {0: ["n0", "n1", "n2"], 1: ["n3", "n4", "n5"]}
            section = planner._name_macro_cluster(0, micro_map, section_order=1)

            self.assertEqual(len(section.pages), 2)
            self.assertEqual(section.pages[0].page_name, "Page A")
            self.assertEqual(section.pages[1].page_name, "Page B")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Full plan_structure()
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanStructure(unittest.TestCase):

    def test_end_to_end(self):
        """Full pipeline with two communities + hub → sections (hubs reintegrated)."""
        with tempfile.TemporaryDirectory() as tmp:
            db, G = _build_two_community_db(tmp)

            # Two-pass: each macro-cluster gets 1 section call + N page calls
            section_count = [0]

            def mock_invoke(messages, **kwargs):
                system_content = messages[0].content.lower()
                resp = MagicMock()
                if "capability name" in system_content:
                    # Section naming (Pass 1)
                    section_count[0] += 1
                    resp.content = json.dumps({
                        "section_name": f"Section {section_count[0]}",
                        "section_description": f"Description {section_count[0]}",
                    })
                else:
                    # Page naming (Pass 2)
                    resp.content = json.dumps({
                        "page_name": f"Page {section_count[0]}.auto",
                        "description": "desc",
                        "retrieval_query": "query",
                    })
                return resp

            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = mock_invoke

            planner = ClusterStructurePlanner(db, mock_llm)
            spec = planner.plan_structure()

            self.assertIsInstance(spec, WikiStructureSpec)
            # At least 2 sections from the 2 communities (hubs now reintegrated)
            self.assertGreaterEqual(len(spec.sections), 2)
            self.assertGreater(spec.total_pages, 0)
            self.assertTrue(spec.wiki_title)
            self.assertTrue(spec.overview)

    def test_returns_valid_wiki_structure_spec(self):
        """Spec should pass Pydantic validation."""
        with tempfile.TemporaryDirectory() as tmp:
            db, _ = _build_two_community_db(tmp)

            def mock_invoke(messages, **kwargs):
                system_content = messages[0].content.lower()
                resp = MagicMock()
                if "capability name" in system_content:
                    resp.content = json.dumps({
                        "section_name": "Test Section",
                        "section_description": "Test",
                    })
                else:
                    resp.content = json.dumps({
                        "page_name": "P1",
                        "description": "d",
                        "retrieval_query": "q",
                    })
                return resp

            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = mock_invoke

            planner = ClusterStructurePlanner(db, mock_llm)
            spec = planner.plan_structure()

            # Validate it round-trips through Pydantic
            validated = WikiStructureSpec.model_validate(spec.model_dump())
            self.assertEqual(validated.total_pages, spec.total_pages)

    def test_page_fields_populated(self):
        """Pages should have target_symbols, key_files, target_folders."""
        with tempfile.TemporaryDirectory() as tmp:
            db, _ = _build_two_community_db(tmp)

            def mock_invoke(messages, **kwargs):
                system_content = messages[0].content.lower()
                resp = MagicMock()
                if "capability name" in system_content:
                    resp.content = json.dumps({
                        "section_name": "Auth",
                        "section_description": "Auth module",
                    })
                else:
                    resp.content = json.dumps({
                        "page_name": "Auth Core",
                        "description": "core",
                        "retrieval_query": "auth",
                    })
                return resp

            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = mock_invoke

            planner = ClusterStructurePlanner(db, mock_llm)
            spec = planner.plan_structure()

            # All sections are regular (no more global_core synthetic section)
            self.assertTrue(spec.sections)
            page = spec.sections[0].pages[0]
            self.assertTrue(page.target_symbols)
            self.assertTrue(page.key_files)
            self.assertTrue(page.retrieval_query)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Fallbacks
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbacks(unittest.TestCase):

    def test_fallback_spec_on_empty_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(tmp)
            planner = ClusterStructurePlanner(db, _mock_llm_response())
            spec = planner.plan_structure()

            self.assertIsInstance(spec, WikiStructureSpec)
            self.assertEqual(len(spec.sections), 1)
            self.assertEqual(spec.sections[0].section_name, "Repository Overview")

    def test_fallback_section_on_llm_failure(self):
        """If LLM raises, fallback section is used with path-derived name."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict(f"n{i}", rel_path=f"src/api/{i}.py") for i in range(5)]
            db = _make_db(tmp, nodes=nodes)
            for i in range(5):
                db.set_cluster(f"n{i}", macro=0, micro=0)
            db.set_meta("phase3_completed", "1")  # Not used here, but realistic
            db.conn.commit()

            planner = ClusterStructurePlanner(db, _mock_llm_response())
            fallback = planner._fallback_section(0, {0: [f"n{i}" for i in range(5)]}, 1)

            self.assertIn("src/api", fallback.section_name)
            self.assertEqual(len(fallback.pages), 1)
            self.assertTrue(fallback.pages[0].target_symbols)

    def test_llm_exception_handled_gracefully(self):
        """If LLM throws during naming, plan_structure still succeeds with fallbacks."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict(f"n{i}") for i in range(4)]
            db = _make_db(tmp, nodes=nodes)
            for i in range(4):
                db.set_cluster(f"n{i}", macro=0, micro=0)
            db.conn.commit()

            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("LLM is down")

            planner = ClusterStructurePlanner(db, mock_llm)
            spec = planner.plan_structure()

            self.assertIsInstance(spec, WikiStructureSpec)
            self.assertGreaterEqual(len(spec.sections), 1)
            self.assertIn("Fallback", spec.sections[0].rationale)


# ═══════════════════════════════════════════════════════════════════════════
# 7. JSON Parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestJsonParsing(unittest.TestCase):

    def test_clean_json(self):
        raw = '{"section_name": "Auth", "pages": []}'
        result = _parse_json_response(raw)
        self.assertEqual(result["section_name"], "Auth")

    def test_markdown_fenced(self):
        raw = '```json\n{"section_name": "Auth"}\n```'
        result = _parse_json_response(raw)
        self.assertEqual(result["section_name"], "Auth")

    def test_leading_text(self):
        raw = 'Here is the result:\n{"section_name": "Auth", "pages": []}'
        result = _parse_json_response(raw)
        self.assertEqual(result["section_name"], "Auth")

    def test_trailing_comma(self):
        raw = '{"section_name": "Auth", "pages": [{"micro_id": 0,},],}'
        result = _parse_json_response(raw)
        self.assertEqual(result["section_name"], "Auth")

    def test_garbage_returns_empty(self):
        raw = "This is not JSON at all."
        result = _parse_json_response(raw)
        self.assertEqual(result, {})


class TestExtractText(unittest.TestCase):

    def test_string_content(self):
        mock = MagicMock()
        mock.content = "hello"
        self.assertEqual(_extract_text(mock), "hello")

    def test_list_content(self):
        mock = MagicMock()
        mock.content = [{"text": "part1"}, "part2"]
        result = _extract_text(mock)
        self.assertIn("part1", result)
        self.assertIn("part2", result)

    def test_plain_string(self):
        self.assertEqual(_extract_text("hello"), "hello")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):

    def test_single_cluster_single_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict("only")]
            db = _make_db(tmp, nodes=nodes)
            db.set_cluster("only", macro=0, micro=0)
            db.conn.commit()

            planner = ClusterStructurePlanner(db, _mock_llm_response())
            spec = planner.plan_structure()

            self.assertEqual(len(spec.sections), 1)
            self.assertEqual(spec.total_pages, 1)

    def test_wiki_title_derived_from_repo_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(tmp)
            db.set_meta("repo_identifier", "myorg/cool-project")
            planner = ClusterStructurePlanner(db, _mock_llm_response())
            self.assertIn("cool-project", planner.wiki_title)

    def test_wiki_title_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(tmp)
            planner = ClusterStructurePlanner(db, _mock_llm_response())
            self.assertIn("Technical Documentation", planner.wiki_title)

    def test_overview_contains_section_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [_make_node_dict(f"n{i}") for i in range(4)]
            db = _make_db(tmp, nodes=nodes)
            for i in range(4):
                db.set_cluster(f"n{i}", macro=0, micro=0)
            db.conn.commit()

            mock_llm = _mock_llm_response(section_name="My Section")
            planner = ClusterStructurePlanner(db, mock_llm)
            spec = planner.plan_structure()

            self.assertIn("My Section", spec.overview)

    def test_node_ids_to_helpers(self):
        """Test _node_ids_to_symbol_names, _node_ids_to_paths, _node_ids_to_folders."""
        with tempfile.TemporaryDirectory() as tmp:
            nodes = [
                _make_node_dict("n0", symbol_name="Foo", rel_path="src/a/foo.py"),
                _make_node_dict("n1", symbol_name="Bar", rel_path="src/b/bar.py"),
            ]
            db = _make_db(tmp, nodes=nodes)

            planner = ClusterStructurePlanner(db, _mock_llm_response())

            symbols = planner._node_ids_to_symbol_names(["n0", "n1"])
            self.assertEqual(symbols, ["Foo", "Bar"])

            paths = planner._node_ids_to_paths(["n0", "n1"])
            self.assertIn("src/a/foo.py", paths)
            self.assertIn("src/b/bar.py", paths)

            folders = planner._node_ids_to_folders(["n0", "n1"])
            self.assertIn("src/a", folders)
            self.assertIn("src/b", folders)

    def test_missing_node_ids_handled(self):
        """Non-existent node IDs should be silently skipped."""
        with tempfile.TemporaryDirectory() as tmp:
            db = _make_db(tmp)
            planner = ClusterStructurePlanner(db, _mock_llm_response())

            symbols = planner._node_ids_to_symbol_names(["nonexistent"])
            self.assertEqual(symbols, [])

            paths = planner._node_ids_to_paths(["nonexistent"])
            self.assertEqual(paths, [])


if __name__ == "__main__":
    unittest.main()
