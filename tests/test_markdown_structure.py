"""Tests for markdown document-structure synthesis (roadmap B6 / §3.5).

Pins:
- ``wire_markdown_structure`` adds ``contains`` edges from a parent
  ``markdown_document`` node to its ``markdown_section`` children;
- a parent document node is synthesized when the chunker emitted only
  sections (the common headered-file case);
- an existing ``markdown_document`` node is reused (not duplicated);
- ``references`` edges are added from a section to the code symbols it
  mentions (markdown links + backtick references), across every rich
  (deep-tier) parser language: C++, C#, Go, Java, Python, JS, TS, Rust;
- edges carry ``edge_class="doc"``, ``relationship_type`` of
  ``contains`` / ``references`` and ``confidence="EXTRACTED"`` in their
  annotations;
- the pass is idempotent (re-running adds no duplicate edges);
- the feature flag gates the whole pass.
"""

from __future__ import annotations

import networkx as nx

from plugin_implementation.feature_flags import FeatureFlags
from plugin_implementation.markdown_structure import (
    _SYNTHETIC_DOC_PREFIX,
    wire_markdown_structure,
)


# Rich (deep-tier) parser languages and a representative code symbol each.
_RICH_LANGS = {
    "cpp": ("cpp", "HttpServer"),
    "csharp": ("csharp", "OrderController"),
    "go": ("go", "PaymentService"),
    "java": ("java", "UserRepository"),
    "python": ("python", "CartHandler"),
    "javascript": ("javascript", "checkoutRouter"),
    "typescript": ("typescript", "AdClient"),
    "rust": ("rust", "ShippingActor"),
}


def _add_code_node(G: nx.MultiDiGraph, language: str, symbol: str) -> str:
    """Add a parsed code node (mirrors graph_builder basic-graph attrs)."""
    node_id = f"{language}::module::{symbol}"
    G.add_node(
        node_id,
        name=symbol,
        symbol_name=symbol,
        symbol_type="class",
        rel_path=f"src/{language}/module.{language}",
        file_name="module",
        language=language,
        start_line=1,
        end_line=10,
        analysis_level="basic",
        source_text=f"class {symbol} {{}}",
        docstring="",
    )
    return node_id


def _add_section(
    G: nx.MultiDiGraph,
    rel_path: str,
    name: str,
    body: str,
    *,
    order: int = 0,
) -> str:
    """Add a markdown_section node (mirrors _parse_documentation_files)."""
    node_id = f"markdown::{rel_path}::{name}#{order}"
    G.add_node(
        node_id,
        name=name,
        symbol_name=name,
        symbol_type="markdown_section",
        rel_path=rel_path,
        file_name="README",
        language="markdown",
        start_line=order,
        end_line=order + 5,
        analysis_level="documentation",
        source_text=f"[File: {rel_path}]\n# {name}\n{body}",
        docstring="",
    )
    return node_id


def _make_graph_all_langs() -> nx.MultiDiGraph:
    """README with one section per rich-parser language, each backticking
    that language's code symbol; plus all code nodes present."""
    G = nx.MultiDiGraph()
    for language, symbol in _RICH_LANGS.values():
        _add_code_node(G, language, symbol)
    rel_path = "README.md"
    for language, symbol in _RICH_LANGS.values():
        _add_section(
            G,
            rel_path,
            f"{language} section",
            f"This section describes the `{symbol}` component.",
            order=hash(language) % 1000,
        )
    return G


def _edges_of_type(G: nx.MultiDiGraph, rel_type: str):
    return [
        (u, v, d)
        for u, v, d in G.edges(data=True)
        if d.get("relationship_type") == rel_type
    ]


# ─── contains ───────────────────────────────────────────────────────────

class TestContainsEdges:

    def test_parent_document_is_synthesized(self):
        G = _make_graph_all_langs()
        stats = wire_markdown_structure(G)
        assert stats["documents_synthesized"] == 1
        parent_id = f"{_SYNTHETIC_DOC_PREFIX}README.md"
        assert G.has_node(parent_id)
        assert G.nodes[parent_id]["symbol_type"] == "markdown_document"

    def test_contains_edge_per_section(self):
        G = _make_graph_all_langs()
        stats = wire_markdown_structure(G)
        # One contains edge for each of the 8 language sections.
        assert stats["contains_edges"] == len(_RICH_LANGS)
        contains = _edges_of_type(G, "contains")
        assert len(contains) == len(_RICH_LANGS)
        for _u, _v, d in contains:
            assert d["edge_class"] == "doc"
            assert d["annotations"]["confidence"] == "EXTRACTED"

    def test_existing_document_node_reused(self):
        G = _make_graph_all_langs()
        # Pre-create the parent document node the parser would have emitted.
        existing = "markdown::README.md::doc"
        G.add_node(
            existing,
            symbol_name="README",
            symbol_type="markdown_document",
            rel_path="README.md",
            language="markdown",
            source_text="",
        )
        stats = wire_markdown_structure(G)
        assert stats["documents_synthesized"] == 0
        assert not G.has_node(f"{_SYNTHETIC_DOC_PREFIX}README.md")
        contains = _edges_of_type(G, "contains")
        assert all(u == existing for u, _v, _d in contains)


# ─── references (across all rich-parser languages) ──────────────────────

class TestReferencesEdges:

    def test_references_resolved_for_every_rich_language(self):
        G = _make_graph_all_langs()
        wire_markdown_structure(G)
        ref_targets = {v for _u, v, _d in _edges_of_type(G, "references")}
        for language, symbol in _RICH_LANGS.values():
            node_id = f"{language}::module::{symbol}"
            assert node_id in ref_targets, f"missing reference for {language}"

    def test_reference_edge_provenance(self):
        G = _make_graph_all_langs()
        wire_markdown_structure(G)
        refs = _edges_of_type(G, "references")
        assert refs
        for _u, _v, d in refs:
            assert d["edge_class"] == "doc"
            assert d["annotations"]["confidence"] == "EXTRACTED"
            assert d["annotations"]["matcher"] == "backtick"

    def test_markdown_link_reference_resolves_to_file_node(self):
        G = nx.MultiDiGraph()
        code_id = _add_code_node(G, "go", "PaymentService")
        # Section links to the Go source file by relative path.
        _add_section(
            G,
            "docs/overview.md",
            "Payments",
            "See [the service](../src/go/module.go) for details.",
        )
        wire_markdown_structure(G)
        ref_targets = {v for _u, v, _d in _edges_of_type(G, "references")}
        assert code_id in ref_targets


# ─── idempotency + gating ───────────────────────────────────────────────

class TestPassBehaviour:

    def test_idempotent(self):
        G = _make_graph_all_langs()
        first = wire_markdown_structure(G)
        n_edges = G.number_of_edges()
        second = wire_markdown_structure(G)
        assert G.number_of_edges() == n_edges
        assert second["contains_edges"] == 0
        assert second["references_edges"] == 0
        assert second["documents_synthesized"] == 0
        # Sanity: the first run did real work.
        assert first["contains_edges"] > 0

    def test_flag_disables_pass(self):
        G = _make_graph_all_langs()
        stats = wire_markdown_structure(
            G, flags=FeatureFlags(markdown_structure=False),
        )
        assert stats == {
            "markdown_nodes": 0,
            "documents_synthesized": 0,
            "contains_edges": 0,
            "references_edges": 0,
        }
        assert not _edges_of_type(G, "contains")
        assert not _edges_of_type(G, "references")

    def test_no_markdown_nodes_is_noop(self):
        G = nx.MultiDiGraph()
        _add_code_node(G, "python", "Foo")
        stats = wire_markdown_structure(G)
        assert stats["markdown_nodes"] == 0
        assert G.number_of_edges() == 0
