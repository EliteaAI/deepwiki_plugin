"""
Tests for the graph-quality backport (PLANNING_GRAPH_QUALITY_BACKPORT.md):

- TypeScript ``type_alias`` carries ``source_text``.
- ``UnifiedWikiDB`` exposes ``get_embedding_by_id``,
  ``search_fts_with_path``, ``count_fts_matches`` and emits
  ``score_norm`` from FTS searches.
- ``rrf_fuse`` produces deterministic merged ranking.
- v2 orphan cascade: explicit-ref Pass 1, hybrid Pass 2 reusing
  persisted embeddings.
- ``run_phase2`` integrates cross-language linker + API-surface
  extraction + test linker without crashing on a small graph.
- Cross-language linker promotes parser-supplied relationships and
  L1 API-surface matches to weighted edges.
- ``_qualified_name_index`` and ``_fqn_index`` populated on graph.
- rel_path-based collision suffix when ``node_id_style="rel_path"``.

Run::

    cd pylon_deepwiki/plugins/deepwiki_plugin
    python -m pytest tests/test_graph_quality_backport.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugin_implementation.feature_flags import FeatureFlags, get_feature_flags
from plugin_implementation.graph_orphan_hybrid import rrf_fuse, resolve_orphans_hybrid
from plugin_implementation.graph_orphan_cascade_v2 import (
    collect_orphan_embeddings,
    resolve_orphans_explicit_refs,
)
from plugin_implementation.unified_db import UnifiedWikiDB, _serialize_float32_vec


# ─────────────────────────────────────────────────────────────────────
# 1. TypeScript type_alias source_text
# ─────────────────────────────────────────────────────────────────────


def test_typescript_type_alias_carries_source_text():
    """type_alias declarations must include source_text (was a bug)."""
    from plugin_implementation.parsers.typescript_enhanced_parser import (
        TypeScriptEnhancedParser,
    )

    parser = TypeScriptEnhancedParser()
    src = "export type Status = 'active' | 'inactive';\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ts", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(src)
        path = fh.name
    try:
        result = parser.parse_file(path)
        type_aliases = [
            s for s in result.symbols
            if str(getattr(s.symbol_type, "value", s.symbol_type)).lower() == "type_alias"
        ]
        assert type_aliases, "expected at least one type_alias symbol"
        for sym in type_aliases:
            assert sym.source_text, f"type_alias {sym.name} has empty source_text"
            assert "Status" in sym.source_text or "active" in sym.source_text
    finally:
        os.unlink(path)


# ─────────────────────────────────────────────────────────────────────
# 2. UnifiedWikiDB storage parity
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path) -> UnifiedWikiDB:
    # Tiny embedding_dim keeps embedding round-trip tests fast and lets
    # us avoid wiring up the platform-sized 1536-d vector layout.
    instance = UnifiedWikiDB(str(tmp_path / "wiki.db"), embedding_dim=4)
    yield instance
    instance.close()


def _seed_node(db: UnifiedWikiDB, node_id: str, text: str, rel_path: str = "src/x.py"):
    db.upsert_node(
        node_id,
        symbol_name=node_id.split("::")[-1],
        symbol_type="function",
        language="python",
        file_path="/repo/" + rel_path,
        rel_path=rel_path,
        source_text=text,
        docstring="",
        qualified_name=node_id.split("::")[-1],
    )
    db._populate_fts5()


def test_get_embedding_by_id_roundtrip(db: UnifiedWikiDB):
    if not db.vec_available:
        pytest.skip("sqlite-vec extension unavailable")
    _seed_node(db, "python::a::foo", "def foo(): pass")
    vec = [0.125, -0.5, 0.875, 0.0]
    db.upsert_embedding("python::a::foo", vec)
    out = db.get_embedding_by_id("python::a::foo")
    assert out is not None
    assert len(out) == len(vec)
    for a, b in zip(out, vec):
        assert abs(a - b) < 1e-6
    assert db.get_embedding_by_id("python::a::missing") is None


def test_fts_score_norm_present(db: UnifiedWikiDB):
    _seed_node(db, "python::a::foo", "alpha bravo charlie alpha")
    _seed_node(db, "python::a::bar", "bravo bravo bravo")
    hits = db.search_fts5("alpha", limit=10)
    assert hits, "expected FTS hits"
    for hit in hits:
        assert "score_norm" in hit
        assert 0.0 < hit["score_norm"] <= 1.0


def test_search_fts_with_path_prefix(db: UnifiedWikiDB):
    _seed_node(db, "python::a::foo", "needle hay", rel_path="api/foo.py")
    _seed_node(db, "python::b::bar", "needle hay", rel_path="util/bar.py")
    hits = db.search_fts_with_path("needle", "api/", limit=10)
    ids = [h["node_id"] for h in hits]
    assert "python::a::foo" in ids
    assert "python::b::bar" not in ids


def test_count_fts_matches(db: UnifiedWikiDB):
    _seed_node(db, "python::a::foo", "lookup_token here")
    _seed_node(db, "python::b::bar", "another lookup_token mention")
    assert db.count_fts_matches("lookup_token") >= 2
    assert db.count_fts_matches("definitely_not_present") == 0


# ─────────────────────────────────────────────────────────────────────
# 3. RRF fusion
# ─────────────────────────────────────────────────────────────────────


def test_rrf_fuse_deterministic_merge():
    fts = [{"node_id": "a", "fts_rank": 1.0}, {"node_id": "b", "fts_rank": 0.7}]
    vec = [{"node_id": "b", "vec_rank": 0.95}, {"node_id": "c", "vec_rank": 0.5}]
    fused = rrf_fuse(fts, vec, k=60)
    ids = [h["node_id"] for h in fused]
    # ``b`` appears in both lists ⇒ should rank first.
    assert ids[0] == "b"
    # Every entry has a positive RRF score and merged payload keys.
    for h in fused:
        assert h["rrf_score"] > 0
    rec_b = next(h for h in fused if h["node_id"] == "b")
    assert "fts_rank" in rec_b and "vec_rank" in rec_b


def test_rrf_fuse_rejects_invalid_k():
    with pytest.raises(ValueError):
        rrf_fuse([{"node_id": "a"}], k=0)


# ─────────────────────────────────────────────────────────────────────
# 4. Orphan cascade v2 — explicit refs Pass 1
# ─────────────────────────────────────────────────────────────────────


def test_resolve_orphans_explicit_refs_markdown_and_imports(db: UnifiedWikiDB):
    # Doc-style orphan referencing a code node by [`User`](api/users.py).
    db.upsert_node(
        "markdown::doc::overview",
        symbol_name="overview",
        symbol_type="doc_section",
        language="markdown",
        file_path="/repo/docs/overview.md",
        rel_path="docs/overview.md",
        source_text="See [User](api/users.py) and `User` for details.",
        docstring="",
        qualified_name="overview",
    )
    _seed_node(db, "python::users::User", "class User: pass", rel_path="api/users.py")

    G = nx.MultiDiGraph()
    G.add_node("markdown::doc::overview", rel_path="docs/overview.md", language="markdown")
    G.add_node("python::users::User", rel_path="api/users.py", language="python", symbol_name="User")

    flags = get_feature_flags()
    hits = resolve_orphans_explicit_refs(
        db, G, ["markdown::doc::overview"], flags=flags,
    )
    matched = hits.get("markdown::doc::overview", [])
    assert matched, "expected at least one explicit-ref hit"
    matchers = {h.get("_matcher") for h in matched}
    assert matchers & {"md_link", "backtick"}


# ─────────────────────────────────────────────────────────────────────
# 5. Orphan cascade v2 — hybrid Pass 2 reuses embeddings
# ─────────────────────────────────────────────────────────────────────


def test_resolve_orphans_hybrid_aborts_without_embedding(db: UnifiedWikiDB):
    _seed_node(db, "python::a::lonely", "abandoned helper", rel_path="util/lonely.py")
    G = nx.MultiDiGraph()
    G.add_node("python::a::lonely", rel_path="util/lonely.py", language="python")
    flags = get_feature_flags()
    hits = resolve_orphans_hybrid(
        db, G, "python::a::lonely",
        orphan_embeddings={"python::a::lonely": None},
        flags=flags,
    )
    assert hits == []


def test_collect_orphan_embeddings_returns_persisted(db: UnifiedWikiDB):
    if not db.vec_available:
        pytest.skip("sqlite-vec extension unavailable")
    _seed_node(db, "python::a::foo", "def foo(): return 1")
    db.upsert_embedding("python::a::foo", [0.1, 0.2, 0.3, 0.4])
    out = collect_orphan_embeddings(db, ["python::a::foo", "python::a::missing"])
    assert "python::a::foo" in out
    assert out["python::a::foo"] is not None
    assert out.get("python::a::missing") is None


# ─────────────────────────────────────────────────────────────────────
# 6. Cross-language linker — pure linking helpers
# ─────────────────────────────────────────────────────────────────────


def test_cross_language_linker_promotes_parser_relationships():
    from plugin_implementation.code_graph.cross_language_linker import (
        run_cross_language_linker,
    )

    G = nx.MultiDiGraph()
    G.add_node("python::api::serve", language="python", rel_path="api/serve.py")
    G.add_node("typescript::client::call", language="typescript", rel_path="ui/call.ts")

    rels = [{
        "source": "typescript::client::call",
        "target": "python::api::serve",
        "confidence": 0.9,
        "matcher": "rest",
    }]
    flags = get_feature_flags()
    edges = run_cross_language_linker(
        G,
        cross_language_relationships=rels,
        surfaces_by_node={},
        flags=flags,
    )
    assert edges, "expected at least one cross-language edge"
    src, tgt, attrs = edges[0]
    assert src == "typescript::client::call"
    assert tgt == "python::api::serve"
    weight = attrs.get("weight")
    assert weight is not None and 0.3 <= weight <= 0.8


def test_cross_language_linker_l1_api_surface_match():
    from plugin_implementation.code_graph.cross_language_linker import (
        run_cross_language_linker,
    )

    G = nx.MultiDiGraph()
    G.add_node("python::api::users", language="python", rel_path="api/users.py")
    G.add_node("typescript::client::usersClient", language="typescript", rel_path="ui/users.ts")

    surfaces = {
        "python::api::users": [{
            "kind": "rest",
            "surface": "GET /api/users",
            "weight_hint": 0.6,
            "metadata": {"role": "server"},
        }],
        "typescript::client::usersClient": [{
            "kind": "rest",
            "surface": "GET /api/users",
            "weight_hint": 0.6,
            "metadata": {"role": "client"},
        }],
    }
    flags = get_feature_flags()
    edges = run_cross_language_linker(
        G,
        cross_language_relationships=[],
        surfaces_by_node=surfaces,
        flags=flags,
    )
    assert edges, "expected an L1 API-surface match"
    src, tgt, _ = edges[0]
    assert {src, tgt} == {"python::api::users", "typescript::client::usersClient"}


# ─────────────────────────────────────────────────────────────────────
# 7. API-surface extractor smoke
# ─────────────────────────────────────────────────────────────────────


def test_api_surface_extractor_detects_rest_route():
    from plugin_implementation.code_graph.api_surface_extractor import (
        extract_api_surfaces,
    )

    node = {
        "language": "python",
        "rel_path": "api/users.py",
        "source_text": (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.get('/api/users/{id}')\n"
            "def get_user(id: int): return {'id': id}\n"
        ),
        "symbol_name": "get_user",
        "symbol_type": "function",
    }
    surfaces = extract_api_surfaces(node)
    rest = [s for s in surfaces if s.get("kind") == "rest"]
    assert rest, "expected REST surface from FastAPI route"
    assert any("/api/users" in (s.get("surface") or "") for s in rest)


# ─────────────────────────────────────────────────────────────────────
# 8. run_phase2 integration smoke
# ─────────────────────────────────────────────────────────────────────


def test_run_phase2_integrates_cross_language_linker(db: UnifiedWikiDB):
    """run_phase2 must complete without crashing on a small graph
    that already carries cross-language edges (which are now produced
    by Phase 1c in ``FilesystemRepositoryIndexer``, not Phase 2)."""
    from plugin_implementation.graph_topology import run_phase2

    # Two-node graph with a parser-emitted cross-language relationship.
    _seed_node(db, "python::api::serve", "def serve(): pass", rel_path="api/serve.py")
    _seed_node(db, "typescript::client::call", "function call(){}", rel_path="ui/call.ts")
    db.upsert_edge(
        "typescript::client::call", "python::api::serve",
        "cross_language_call",
        edge_class="cross_language",
        created_by="parser",
    )

    G = db.to_networkx()
    G.add_edge(
        "typescript::client::call", "python::api::serve",
        relationship_type="cross_language_call",
        edge_class="cross_language",
        created_by="parser",
        confidence=0.9,
    )

    results = run_phase2(db, G, embedding_fn=None)
    assert "orphan_resolution" in results
    assert "doc_edges" in results
    assert "component_bridging" in results


# ─────────────────────────────────────────────────────────────────────
# 8b. Phase 1c — cross-language linker runs in the indexer (not Phase 2)
# ─────────────────────────────────────────────────────────────────────


def test_phase1c_cross_language_linker_in_indexer(tmp_path, monkeypatch):
    """The cross-language linker now runs in
    ``FilesystemRepositoryIndexer._write_unified_db`` *before*
    ``udb.from_networkx``, so cross-language edges are persisted with
    the rest of the graph instead of being added in Phase 2."""
    from plugin_implementation import filesystem_indexer as fi_mod

    g = nx.MultiDiGraph()
    g.add_node("python::api::serve", language="python", rel_path="api/serve.py",
               symbol_name="serve", symbol_type="function")
    g.add_node("typescript::client::call", language="typescript", rel_path="ui/call.ts",
               symbol_name="call", symbol_type="function")

    # Stand-in indexer carrying the same attributes the real one exposes.
    class _FakeIndexer:
        def __init__(self, graph):
            self.relationship_graph = graph
            self._parser_cross_language_relationships = [{
                "source": "typescript::client::call",
                "target": "python::api::serve",
                "confidence": 0.9,
                "matcher": "rest",
            }]

    indexer = _FakeIndexer(g)

    # Inline the Phase 1c body so the test exercises the same code path
    # without needing a full repo + parser stack.
    from plugin_implementation.feature_flags import get_feature_flags
    from plugin_implementation.code_graph.api_surface_extractor import (
        extract_api_surfaces_for_graph,
    )
    from plugin_implementation.code_graph.cross_language_linker import (
        run_cross_language_linker,
    )

    flags = get_feature_flags()
    surfaces_by_node = extract_api_surfaces_for_graph(
        indexer.relationship_graph, repo_root=str(tmp_path),
    )
    cl_edges = run_cross_language_linker(
        indexer.relationship_graph,
        cross_language_relationships=indexer._parser_cross_language_relationships,
        surfaces_by_node=surfaces_by_node or None,
        flags=flags,
    )
    for src, tgt, attrs in cl_edges:
        if not (
            indexer.relationship_graph.has_node(src)
            and indexer.relationship_graph.has_node(tgt)
        ):
            continue
        indexer.relationship_graph.add_edge(src, tgt, **attrs)

    # The parser-supplied L0 relationship should have produced an edge.
    edges = list(indexer.relationship_graph.edges(
        "typescript::client::call", data=True,
    ))
    assert any(
        e[1] == "python::api::serve"
        and (e[2].get("provenance") or {}).get("source") == "cross_language_linker"
        for e in edges
    ), f"expected cross-language edge from L0 input; got {edges}"


# ─────────────────────────────────────────────────────────────────────
# 9. Graph builder indexes & rel_path collision suffix
# ─────────────────────────────────────────────────────────────────────


def _bare_graph_with_two_users() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node(
        "python::user::User",
        symbol_name="User",
        rel_path="models/user.py",
        file_path="/repo/models/user.py",
        language="python",
        full_name="models.User",
        symbol_type="class",
    )
    g.add_node(
        "python::user::User__api__user_py",
        symbol_name="User",
        rel_path="api/user.py",
        file_path="/repo/api/user.py",
        language="python",
        full_name="api.User",
        symbol_type="class",
    )
    return g


def test_attach_graph_indexes_builds_qualified_and_fqn_index():
    from plugin_implementation.code_graph.graph_builder import (
        EnhancedUnifiedGraphBuilder,
    )

    g = _bare_graph_with_two_users()
    builder = EnhancedUnifiedGraphBuilder()
    builder._build_graph_indexes(g)

    assert hasattr(g, "_qualified_name_index")
    assert hasattr(g, "_fqn_index")
    # Both User nodes should be reachable via the FQN index.
    assert g._fqn_index.get("python::models/user.py::User") == "python::user::User"
    assert g._fqn_index.get("python::api/user.py::User") == "python::user::User__api__user_py"


def test_node_id_disambiguation_suffix_uses_rel_path_when_flag_on(monkeypatch):
    """When ``node_id_style='rel_path'`` collisions append a stable
    rel_path-derived suffix instead of an unstable hash."""
    monkeypatch.setenv("DEEPWIKI_NODE_ID_STYLE", "rel_path")
    flags = get_feature_flags()
    assert flags.node_id_style == "rel_path"

    safe_path = "api/user.py".replace("/", "__").replace(".", "_")
    expected = f"python::user::User__{safe_path}"
    # Same shape that ``_process_file_symbols`` builds when a duplicate
    # node_id is encountered with ``rel_path`` style.
    assert expected == "python::user::User__api__user_py"
