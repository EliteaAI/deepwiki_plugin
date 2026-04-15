"""
Shared test fixtures for Clustering V2 integration tests.

Consolidates helper functions previously duplicated across
test_graph_clustering.py, test_graph_topology.py, and test_cluster_planner.py.

Provides realistic graph topologies for integration testing:
- Two-community (basic)
- C++ header-heavy (dense, models fmtlib)
- Mixed code+doc (Python SDK)
- Pure-doc (documentation repo)
- Large repo (smoke/perf)
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import networkx as nx

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugin_implementation.unified_db import UnifiedWikiDB


# ═══════════════════════════════════════════════════════════════════════════
# Core Helpers (previously copy-pasted across test files)
# ═══════════════════════════════════════════════════════════════════════════

def make_graph(*edges, orphans=None):
    """Build a small MultiDiGraph.

    Each edge is (src, tgt) or (src, tgt, attrs_dict).
    Orphan nodes are added disconnected.
    """
    G = nx.MultiDiGraph()
    for e in edges:
        if len(e) == 2:
            G.add_edge(e[0], e[1], relationship_type="calls", weight=1.0)
        elif len(e) == 3:
            attrs = e[2] if isinstance(e[2], dict) else {}
            attrs.setdefault("weight", 1.0)
            G.add_edge(e[0], e[1], **attrs)
    for n in (orphans or []):
        G.add_node(n)
    return G


def make_db(tmp_dir, nodes=None, edges=None) -> UnifiedWikiDB:
    """Create a fresh UnifiedWikiDB with optional seed data."""
    db_path = os.path.join(tmp_dir, "test.wiki.db")
    db = UnifiedWikiDB(db_path)
    if nodes:
        db.upsert_nodes_batch(nodes)
    if edges:
        db.upsert_edges_batch(edges)
    return db


def make_node_dict(node_id, **kwargs) -> Dict[str, Any]:
    """Minimal node dict for DB insertion with sensible defaults."""
    d = {
        "node_id": node_id,
        "symbol_name": kwargs.pop("symbol_name", node_id),
        "symbol_type": kwargs.pop("symbol_type", "function"),
        "rel_path": kwargs.pop("rel_path", "src/main.py"),
        "source_text": kwargs.pop("source_text", f"def {node_id}(): pass"),
        "docstring": kwargs.pop("docstring", ""),
        "is_architectural": kwargs.pop("is_architectural", 1),
        "is_doc": kwargs.pop("is_doc", 0),
        "language": kwargs.pop("language", "python"),
        "signature": kwargs.pop("signature", f"def {node_id}()"),
    }
    d.update(kwargs)
    return d


def make_doc_node(node_id, rel_path, **kwargs) -> Dict[str, Any]:
    """Create a documentation node dict."""
    return make_node_dict(
        node_id,
        symbol_name=os.path.basename(rel_path),
        symbol_type=kwargs.pop("symbol_type", "markdown_document"),
        rel_path=rel_path,
        source_text=kwargs.pop("source_text", f"# {os.path.basename(rel_path)}\n\nDocumentation content."),
        is_architectural=1,
        is_doc=1,
        language="markdown",
        signature="",
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Graph Topologies
# ═══════════════════════════════════════════════════════════════════════════

def build_two_community_graph(n=5):
    """Two dense N-node communities connected by a thin bridge.

    Community A: a0..a(n-1) with dense intra-edges
    Community B: b0..b(n-1) with dense intra-edges
    Bridge: a0 → b0 (single weak link)
    """
    G = nx.MultiDiGraph()
    for prefix in ("a", "b"):
        for i in range(n):
            for j in range(n):
                if i != j:
                    G.add_edge(f"{prefix}{i}", f"{prefix}{j}",
                               relationship_type="calls", weight=1.0)
    G.add_edge("a0", "b0", relationship_type="calls", weight=0.1)
    return G


def build_cpp_header_graph(n_symbols=100):
    """Dense header-heavy graph modeling a C++ library like fmtlib.

    - 3 "header" files with n_symbols/3 functions each
    - Dense intra-file edges (every function may call ~30% of its neighbors)
    - Sparse inter-file edges (a few cross-header calls)
    - 5 doc nodes (README, CONTRIBUTING, etc.)
    - 3 config nodes (CMakeLists.txt, .clang-format, etc.)

    Returns (G, nodes_list) where nodes_list is ready for db.upsert_nodes_batch().
    """
    G = nx.MultiDiGraph()
    nodes = []
    rng = random.Random(42)

    headers = ["include/fmt/format.h", "include/fmt/core.h", "include/fmt/chrono.h"]
    per_file = n_symbols // len(headers)
    code_node_ids = []

    for fi, header in enumerate(headers):
        file_nodes = []
        for si in range(per_file):
            nid = f"cpp::{header}::func_{fi}_{si}"
            file_nodes.append(nid)
            code_node_ids.append(nid)
            nodes.append(make_node_dict(
                nid,
                symbol_name=f"func_{fi}_{si}",
                symbol_type="function",
                rel_path=header,
                language="cpp",
                source_text=f"void func_{fi}_{si}() {{}}",
                signature=f"void func_{fi}_{si}()",
                docstring=f"Function {si} in {header}" if si < 3 else "",
                start_line=si * 10,
            ))
            G.add_node(nid)

        # Dense intra-file edges (~30% connectivity)
        for i, src in enumerate(file_nodes):
            for j, tgt in enumerate(file_nodes):
                if i != j and rng.random() < 0.3:
                    G.add_edge(src, tgt, relationship_type="calls", weight=1.0)

    # Sparse inter-file edges (5% of first file calls second, etc.)
    for i in range(len(headers) - 1):
        src_nodes = code_node_ids[i * per_file:(i + 1) * per_file]
        tgt_nodes = code_node_ids[(i + 1) * per_file:(i + 2) * per_file]
        for src in src_nodes[:max(1, len(src_nodes) // 20)]:
            tgt = rng.choice(tgt_nodes)
            G.add_edge(src, tgt, relationship_type="calls", weight=1.0)

    # Doc nodes
    doc_files = [
        ("doc::README.md", "README.md"),
        ("doc::CONTRIBUTING.md", "CONTRIBUTING.md"),
        ("doc::docs/format.md", "docs/format.md"),
        ("doc::docs/chrono.md", "docs/chrono.md"),
        ("doc::CHANGELOG.md", "CHANGELOG.md"),
    ]
    for nid, rpath in doc_files:
        nodes.append(make_doc_node(nid, rpath))
        G.add_node(nid)

    # Config nodes
    config_files = [
        ("doc::CMakeLists.txt", "CMakeLists.txt", "infrastructure_document"),
        ("doc::.clang-format", ".clang-format", "yaml_document"),
        ("doc::.github/workflows/ci.yml", ".github/workflows/ci.yml", "yaml_document"),
    ]
    for nid, rpath, stype in config_files:
        nodes.append(make_doc_node(nid, rpath, symbol_type=stype))
        G.add_node(nid)

    return G, nodes


def build_mixed_code_doc_graph():
    """Python SDK with rich documentation (60% code, 40% docs).

    - 50 code nodes across 5 packages (10 per package)
    - 25 doc nodes (READMEs per package + top-level docs)
    - 5 config nodes (setup.py, tox.ini, CI files)
    - Structural edges within packages
    - Doc→code hyperlink-like edges (README references classes)
    - Cross-package import edges

    Returns (G, nodes_list).
    """
    G = nx.MultiDiGraph()
    nodes = []
    rng = random.Random(42)

    packages = ["auth", "api", "models", "utils", "cli"]

    # Code nodes: 10 per package
    code_by_pkg = {}
    for pkg in packages:
        pkg_nodes = []
        for i in range(10):
            nid = f"python::src/{pkg}/module_{i}.py::Class_{pkg}_{i}"
            pkg_nodes.append(nid)
            nodes.append(make_node_dict(
                nid,
                symbol_name=f"Class_{pkg}_{i}",
                symbol_type="class",
                rel_path=f"src/{pkg}/module_{i}.py",
                docstring=f"Class {i} in {pkg} package",
            ))
            G.add_node(nid)
        code_by_pkg[pkg] = pkg_nodes

        # Intra-package edges (chain: 0→1→2→...→9)
        for i in range(9):
            G.add_edge(pkg_nodes[i], pkg_nodes[i + 1],
                        relationship_type="calls", weight=1.0)
        # A couple reverse edges for density
        G.add_edge(pkg_nodes[5], pkg_nodes[0], relationship_type="imports", weight=1.0)

    # Cross-package edges (api→auth, cli→api, models→utils)
    cross = [("api", "auth"), ("cli", "api"), ("models", "utils"), ("auth", "models")]
    for src_pkg, tgt_pkg in cross:
        src = rng.choice(code_by_pkg[src_pkg])
        tgt = rng.choice(code_by_pkg[tgt_pkg])
        G.add_edge(src, tgt, relationship_type="imports", weight=1.0)

    # Doc nodes: per-package README + top-level docs
    for pkg in packages:
        nid = f"doc::src/{pkg}/README.md"
        nodes.append(make_doc_node(nid, f"src/{pkg}/README.md",
                                    source_text=f"# {pkg.title()} Package\n\nUses Class_{pkg}_0."))
        G.add_node(nid)

    top_docs = [
        ("doc::README.md", "README.md", "# My SDK\n\nOverview of the project."),
        ("doc::CONTRIBUTING.md", "CONTRIBUTING.md", "# Contributing\n\nHow to contribute."),
        ("doc::docs/quickstart.md", "docs/quickstart.md", "# Quick Start\n\nGetting started guide."),
        ("doc::docs/api-reference.md", "docs/api-reference.md", "# API Reference\n\nClass_api_0 usage."),
        ("doc::docs/auth-guide.md", "docs/auth-guide.md", "# Auth Guide\n\nClass_auth_0 details."),
    ]
    for nid, rpath, text in top_docs:
        nodes.append(make_doc_node(nid, rpath, source_text=text))
        G.add_node(nid)

    # Config nodes
    configs = [
        ("doc::setup.py", "setup.py", "infrastructure_document"),
        ("doc::tox.ini", "tox.ini", "infrastructure_document"),
        ("doc::.github/workflows/ci.yml", ".github/workflows/ci.yml", "yaml_document"),
        ("doc::pyproject.toml", "pyproject.toml", "infrastructure_document"),
        ("doc::requirements.txt", "requirements.txt", "infrastructure_document"),
    ]
    for nid, rpath, stype in configs:
        nodes.append(make_doc_node(nid, rpath, symbol_type=stype))
        G.add_node(nid)

    return G, nodes


def build_pure_doc_graph(n_docs=50):
    """Documentation-only repository (no code nodes).

    - n_docs markdown nodes across 5 directories (n_docs/5 each)
    - README.md in each directory
    - Some cross-reference hyperlink edges between docs
    - No code nodes, no structural call edges

    Returns (G, nodes_list).
    """
    G = nx.MultiDiGraph()
    nodes = []
    rng = random.Random(42)

    directories = ["getting-started", "api", "guides", "reference", "contributing"]
    per_dir = n_docs // len(directories)

    all_doc_ids = []
    dir_doc_ids = {}

    for di, dirname in enumerate(directories):
        dir_nodes = []
        for fi in range(per_dir):
            fname = "README.md" if fi == 0 else f"page_{fi}.md"
            rpath = f"docs/{dirname}/{fname}"
            nid = f"doc::{rpath}"
            dir_nodes.append(nid)
            all_doc_ids.append(nid)
            nodes.append(make_doc_node(
                nid, rpath,
                source_text=f"# {dirname.replace('-', ' ').title()} - {fname}\n\nContent for {fname}.",
            ))
            G.add_node(nid)

        dir_doc_ids[dirname] = dir_nodes

        # Intra-directory links (README references other pages)
        readme_nid = dir_nodes[0]
        for other in dir_nodes[1:]:
            G.add_edge(readme_nid, other, relationship_type="hyperlink", weight=1.0)

    # Cross-directory links (a few READMEs reference pages in other dirs)
    cross_links = [
        ("getting-started", "api"),
        ("guides", "reference"),
        ("api", "reference"),
    ]
    for src_dir, tgt_dir in cross_links:
        src = dir_doc_ids[src_dir][0]  # README of source dir
        tgt = rng.choice(dir_doc_ids[tgt_dir])
        G.add_edge(src, tgt, relationship_type="hyperlink", weight=1.0)

    return G, nodes


def build_large_repo_graph(n=2000):
    """Large multi-community graph for smoke/performance tests.

    - n nodes across n/100 communities (100 nodes each)
    - Dense intra-community edges, sparse bridges
    - 10% doc nodes (n/10)

    Returns (G, nodes_list).
    """
    G = nx.MultiDiGraph()
    nodes = []
    rng = random.Random(42)

    communities = max(2, n // 100)
    per_community = n // communities

    all_code_ids = []
    community_ids = {}

    for ci in range(communities):
        comm_nodes = []
        for ni in range(per_community):
            nid = f"python::pkg_{ci}/mod_{ni}.py::Func_{ci}_{ni}"
            comm_nodes.append(nid)
            all_code_ids.append(nid)
            nodes.append(make_node_dict(
                nid,
                symbol_name=f"Func_{ci}_{ni}",
                rel_path=f"pkg_{ci}/mod_{ni}.py",
                docstring=f"Function {ni}" if ni < 2 else "",
            ))
            G.add_node(nid)

        community_ids[ci] = comm_nodes

        # Intra-community edges (~15% connectivity)
        for i in range(len(comm_nodes)):
            for j in range(i + 1, min(i + 20, len(comm_nodes))):
                if rng.random() < 0.3:
                    G.add_edge(comm_nodes[i], comm_nodes[j],
                                relationship_type="calls", weight=1.0)
                    if rng.random() < 0.3:
                        G.add_edge(comm_nodes[j], comm_nodes[i],
                                    relationship_type="calls", weight=1.0)

    # Inter-community bridges (1-2 edges between adjacent communities)
    for ci in range(communities - 1):
        src = rng.choice(community_ids[ci])
        tgt = rng.choice(community_ids[ci + 1])
        G.add_edge(src, tgt, relationship_type="imports", weight=1.0)

    # Doc nodes (10% of total)
    n_docs = n // 10
    for di in range(n_docs):
        rpath = f"docs/section_{di % 10}/page_{di}.md"
        nid = f"doc::{rpath}"
        nodes.append(make_doc_node(nid, rpath))
        G.add_node(nid)

    return G, nodes


# ═══════════════════════════════════════════════════════════════════════════
# LLM Mocking
# ═══════════════════════════════════════════════════════════════════════════

def mock_llm_response(section_name="Test Section", pages=None):
    """Create a MagicMock LLM that returns a JSON naming response.

    The mock's .invoke() returns a MagicMock with .content containing JSON.
    """
    import json

    if pages is None:
        pages = [{"page_name": f"{section_name} Overview", "page_description": "Overview page."}]

    response_json = json.dumps({
        "section_name": section_name,
        "section_description": f"Description for {section_name}",
        "pages": pages,
    })

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=response_json)
    return mock_llm


def mock_llm_multi_response(responses):
    """Create a MagicMock LLM that returns different responses per call.

    responses: list of (section_name, pages_list) tuples.
    """
    import json

    side_effects = []
    for section_name, pages in responses:
        if pages is None:
            pages = [{"page_name": f"{section_name} Overview", "page_description": "Overview."}]
        response_json = json.dumps({
            "section_name": section_name,
            "section_description": f"Description for {section_name}",
            "pages": pages,
        })
        side_effects.append(MagicMock(content=response_json))

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = side_effects
    return mock_llm


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Runners
# ═══════════════════════════════════════════════════════════════════════════

def run_full_pipeline(db, G, embedding_fn=None):
    """Execute Phase 2 → Phase 3 pipeline on a db+graph pair.

    Returns (phase2_stats, phase3_stats).
    """
    from plugin_implementation.graph_topology import run_phase2
    from plugin_implementation.graph_clustering import run_phase3

    p2 = run_phase2(db, G, embedding_fn=embedding_fn)
    p3 = run_phase3(db, G)
    return p2, p3


def populate_db_and_graph(tmp_dir, nodes, graph=None):
    """Create a db from nodes and optionally import a pre-built graph.

    Returns (db, G) where G is either the passed graph or reconstructed from db.
    """
    db = make_db(tmp_dir, nodes=nodes)
    if graph is not None:
        return db, graph
    G = db.to_networkx()
    return db, G
