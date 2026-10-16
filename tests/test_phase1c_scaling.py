"""Phase 1c scaling guards — regression tests for the superlinear stall.

Context: a K8s worker indexing a repo with 589,552 graph nodes (553,601 of
them documentation/XML chunk nodes, plus a generated minified JS bundle)
logged "UnifiedWikiDB opened" and then produced nothing for hours — CPU
busy, no OOM. Root cause was Phase 1c (``extract_api_surfaces_for_graph`` /
``run_cross_language_linker``): the empty-``source_text`` disk-slice
fallback re-scanned a ~200KB minified line once per symbol on that line
(O(symbols * line_length)), several gRPC-client regexes had an unanchored
leading ``\\w+`` group that goes superlinear on a long non-matching
identifier, 553K doc nodes were run through the full matcher set for no
benefit, and L1 cross-language linking paired every member of a
ubiquitous surface (O(k^2)).

Every test below would take minutes (or, for the fan-out case, longer) on
the pre-fix code and must now complete within the stated generous
wall-clock budget.
"""

from __future__ import annotations

import time

import networkx as nx

from plugin_implementation.code_graph.api_surface_extractor import (
    APISurface,
    extract_api_surfaces,
    extract_api_surfaces_for_graph,
)
from plugin_implementation.code_graph.cross_language_linker import (
    link_l1_api_surface,
)


# ─── 1. Minified-file slice fallback ────────────────────────────────────


class TestMinifiedSliceFallback:
    """5,000 nodes all slicing the same huge minified line must not
    re-scan that line per symbol — the slice should be skipped outright."""

    def test_minified_bundle_completes_fast_and_yields_no_slice_surfaces(self, tmp_path):
        # One ~350KB line: lots of minified function tokens + a real
        # fetch() call, so this would have produced a REST surface if the
        # slice fallback had been allowed to run on it.
        tokens = [f"function f{i}(){{}};" for i in range(20_000)]
        tokens.append('fetch("/api/x");')
        line = "".join(tokens)
        assert len(line) > 300_000

        bundle = tmp_path / "bundle.js"
        bundle.write_text(line, encoding="utf-8")

        g = nx.MultiDiGraph()
        for i in range(5_000):
            g.add_node(
                f"js_{i}",
                symbol_name=f"f{i}",
                symbol_type="function",
                rel_path="bundle.js",
                file_name="bundle.js",
                language="javascript",
                start_line=1,
                end_line=1,
                source_text="",
            )

        t0 = time.perf_counter()
        surfaces = extract_api_surfaces_for_graph(g, repo_root=str(tmp_path))
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"minified-slice scan took {elapsed:.1f}s"
        # The slice is longer than MAX_SLICE_CHARS, so every one of the
        # 5,000 nodes gets "" from _file_slice and produces no surface —
        # including the real fetch() call that lives on that line.
        assert surfaces == {}


# ─── 2. Doc-node flood gating ───────────────────────────────────────────


class TestDocNodeGating:
    """50,000 plain documentation nodes must be skipped outright, while
    schema (proto) and BDD-bearing doc file types still get scanned."""

    def test_doc_flood_is_fast_and_silent(self, tmp_path):
        g = nx.MultiDiGraph()
        for i in range(50_000):
            g.add_node(
                f"doc_{i}",
                symbol_name=f"chunk_{i}",
                symbol_type="text_chunk",
                rel_path=f"docs/page_{i}.xml",
                file_name=f"page_{i}.xml",
                language="xml",
                start_line=1,
                end_line=1,
                source_text="<note>hi</note>",
                analysis_level="documentation",
            )

        # Schema (proto/IDL) doc node — _match_grpc needs analysis_level
        # == "documentation" schema files to still be scanned.
        g.add_node(
            "schema_doc",
            symbol_name="cart.proto",
            symbol_type="text_chunk",
            rel_path="protos/cart.proto",
            file_name="cart.proto",
            language="schema",
            start_line=1,
            end_line=3,
            source_text="service Foo { rpc Bar(X) returns (Y); }",
            analysis_level="documentation",
        )

        # BDD .feature doc node — allow-listed by file suffix.
        g.add_node(
            "feature_doc",
            symbol_name="login.feature",
            symbol_type="text_chunk",
            rel_path="features/login.feature",
            file_name="login.feature",
            language="gherkin",
            start_line=1,
            end_line=1,
            source_text="Given a user\n",
            analysis_level="documentation",
        )

        t0 = time.perf_counter()
        surfaces = extract_api_surfaces_for_graph(g)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"doc flood scan took {elapsed:.1f}s"
        # None of the 50,000 plain xml doc nodes produced a surface.
        assert not any(k.startswith("doc_") for k in surfaces)

        assert "schema_doc" in surfaces
        assert {s["surface"] for s in surfaces["schema_doc"]} == {"grpc:Foo/Bar"}

        assert "feature_doc" in surfaces
        assert any(s["kind"] == "bdd" for s in surfaces["feature_doc"])


# ─── 3. Per-node text truncation ────────────────────────────────────────


class TestSurfaceScanTruncation:
    """A route decorator past MAX_SURFACE_SCAN_CHARS is invisible to the
    matchers; one within the cap is still found."""

    def test_decorator_past_cap_is_not_found(self):
        filler = "x" * 300_000
        node = {
            "language": "python",
            "rel_path": "api/big.py",
            "symbol_name": "get_x",
            "source_text": filler + '\n@app.get("/x")\ndef get_x(): return {}\n',
        }
        surfaces = extract_api_surfaces(node)
        assert surfaces == []

    def test_decorator_before_cap_is_found(self):
        filler = "x" * 300_000
        node = {
            "language": "python",
            "rel_path": "api/big.py",
            "symbol_name": "get_x",
            "source_text": '@app.get("/x")\ndef get_x(): return {}\n' + filler,
        }
        surfaces = extract_api_surfaces(node)
        rest = [s for s in surfaces if s["kind"] == "rest"]
        assert rest, "expected a REST surface from the decorator inside the scan cap"
        assert any("/x" in s["surface"] for s in rest)


# ─── 4. Regex hardening (superlinear patterns) ──────────────────────────


class TestRegexHardening:
    """Patterns that previously had an unanchored leading ``\\w+`` group
    must return quickly on a long non-matching identifier, while real
    matches are unaffected."""

    def test_go_fullmethodname_long_non_match_is_fast(self):
        node = {
            "language": "go",
            "rel_path": "x.go",
            "symbol_name": "x",
            "source_text": "a_" * 5_000,
        }
        t0 = time.perf_counter()
        surfaces = extract_api_surfaces(node)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"go FullMethodName scan took {elapsed:.2f}s"
        assert surfaces == []

    def test_go_fullmethodname_real_match_still_works(self):
        node = {
            "language": "go",
            "rel_path": "x.go",
            "symbol_name": "x",
            "source_text": (
                'CartService_AddItem_FullMethodName = '
                '"/hipstershop.CartService/AddItem"'
            ),
        }
        surfaces = extract_api_surfaces(node)
        assert any(s["surface"] == "grpc:CartService/AddItem" for s in surfaces)

    def test_long_identifier_is_fast_across_languages(self):
        text = "a_" * 10_000  # 20KB, no literal match for any matcher
        for language in (
            "python", "java", "csharp", "javascript", "cpp", "rust",
        ):
            node = {
                "language": language,
                "rel_path": f"x.{language}",
                "symbol_name": "x",
                "source_text": text,
            }
            t0 = time.perf_counter()
            extract_api_surfaces(node)
            elapsed = time.perf_counter() - t0
            assert elapsed < 1.0, f"{language} scan took {elapsed:.2f}s"


# ─── 5. Cross-language L1 fan-out cap ────────────────────────────────────


class TestFanoutCap:
    """A surface shared by more than MAX_L1_SURFACE_FANOUT nodes is
    skipped outright (O(k^2) pairing is too expensive for near-zero
    specificity); a surface under the cap still pairs normally."""

    @staticmethod
    def _contract_graph(n: int, contract_id: str) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        g.add_node(
            contract_id, symbol_type="contract", signature="rest_route",
            symbol_name="GET /x",
        )
        for i in range(n):
            lang = "python" if i % 2 == 0 else "typescript"
            nid = f"n{i}"
            g.add_node(nid, language=lang)
            g.add_edge(nid, contract_id, relationship_type="defines")
        return g

    def test_contract_nodes_path_skips_above_cap(self):
        g = self._contract_graph(200, "contract::rest_route::GET /x")
        t0 = time.perf_counter()
        edges = link_l1_api_surface(g)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"fan-out skip took {elapsed:.1f}s"
        assert edges == []

    def test_contract_nodes_path_pairs_under_cap(self):
        # 5 python + 5 typescript predecessors -> 5*5 = 25 cross-language
        # pairs (same-language pairs are excluded).
        g = self._contract_graph(10, "contract::rest_route::GET /y")
        edges = link_l1_api_surface(g)
        assert len(edges) == 25
        assert all(a["relationship_type"] == "cross_language_L1" for _, _, a in edges)

    @staticmethod
    def _legacy_surfaces(n: int, surface: str):
        g = nx.MultiDiGraph()
        surfaces_by_node = {}
        for i in range(n):
            lang = "python" if i % 2 == 0 else "typescript"
            nid = f"m{i}"
            g.add_node(nid, language=lang)
            surfaces_by_node[nid] = [APISurface(
                kind="rest", surface=surface, weight_hint=0.7, metadata={},
            )]
        return g, surfaces_by_node

    def test_legacy_path_skips_above_cap(self):
        g, surfaces_by_node = self._legacy_surfaces(200, "GET /z")
        t0 = time.perf_counter()
        edges = link_l1_api_surface(g, surfaces_by_node)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"legacy fan-out skip took {elapsed:.1f}s"
        assert edges == []

    def test_legacy_path_pairs_under_cap(self):
        g, surfaces_by_node = self._legacy_surfaces(10, "GET /w")
        edges = link_l1_api_surface(g, surfaces_by_node)
        assert len(edges) == 25
