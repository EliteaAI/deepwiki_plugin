"""Tests for noise-node contraction (Phase A pilot).

Covers ``contract_graph_inplace`` end-to-end on synthetic graphs:
- parameter / variable / field nodes are removed and edges rewired onto
  the containing arch node;
- ``via=src=...`` / ``via=tgt=...`` annotations capture the dropped
  endpoint so context survives;
- self-loops (method ↔ its own parameter) collapse and don't pollute
  the graph;
- duplicate edges that arise post-rewriting merge their ``via`` lists
  instead of stacking;
- unresolved noise nodes (parent_symbol pointing nowhere) stay in the
  graph and are reported in the metrics.
"""

from __future__ import annotations

import networkx as nx
import pytest

from plugin_implementation.code_graph.graph_contraction import (
    NOISE_TYPES,
    contract_graph_inplace,
)


def _add_node(g: nx.MultiDiGraph, nid: str, *, symbol_type: str,
              parent_symbol: str = "", language: str = "python",
              symbol_name: str | None = None,
              start_line: int = 1) -> None:
    g.add_node(
        nid,
        symbol_type=symbol_type,
        parent_symbol=parent_symbol,
        language=language,
        symbol_name=symbol_name or nid.rsplit("::", 1)[-1],
        start_line=start_line,
    )


# ─── 1. NOISE_TYPES contract ──────────────────────────────────────────

class TestNoiseTypeSet:

    def test_noise_set_contains_priority_1_and_2_data(self):
        # parameter / variable / local_variable / argument (priority 1)
        # field / property (priority 2)
        for t in {"parameter", "variable", "local_variable",
                  "argument", "field", "property"}:
            assert t in NOISE_TYPES, f"{t} should be a noise type"

    def test_arch_types_not_in_noise(self):
        for t in {"class", "method", "function", "interface",
                  "constant", "constructor", "type_alias",
                  "macro", "module", "namespace"}:
            assert t not in NOISE_TYPES, f"{t} must NOT be classified as noise"

    def test_doc_types_not_in_noise(self):
        for t in {"markdown_section", "markdown_document",
                  "yaml_document", "infrastructure_document"}:
            assert t not in NOISE_TYPES


# ─── 2. Node removal + edge rewiring ──────────────────────────────────

class TestContractionDropsNoiseAndRewires:

    def test_parameter_collapses_to_method(self):
        """A method ↔ parameter edge becomes a self-loop and is dropped."""
        g = nx.MultiDiGraph()
        _add_node(g, "py::svc::AuthService", symbol_type="class")
        _add_node(g, "py::svc::AuthService.login",
                  symbol_type="method", parent_symbol="svc.AuthService")
        _add_node(g, "py::svc::AuthService.login.username",
                  symbol_type="parameter",
                  parent_symbol="svc.AuthService.login")
        # Edge: method → its own parameter (pure noise plumbing)
        g.add_edge("py::svc::AuthService.login",
                   "py::svc::AuthService.login.username",
                   relationship_type="uses")

        metrics = contract_graph_inplace(g)

        assert "py::svc::AuthService.login.username" not in g.nodes
        assert metrics["nodes_removed"] == 1
        assert metrics["edges_rewritten"] == 1
        assert metrics["self_loops_dropped"] == 1
        # The edge collapsed into a self-loop and was dropped.
        assert g.number_of_edges() == 0

    def test_field_to_external_class_rewires_to_owner(self):
        """User.email_field → EmailValidator becomes User → EmailValidator."""
        g = nx.MultiDiGraph()
        _add_node(g, "py::svc::User", symbol_type="class")
        _add_node(g, "py::val::EmailValidator", symbol_type="class")
        _add_node(
            g, "py::svc::User.email",
            symbol_type="field",
            parent_symbol="svc.User",
            symbol_name="email",
            start_line=42,
        )
        g.add_edge("py::svc::User.email",
                   "py::val::EmailValidator",
                   relationship_type="references")

        metrics = contract_graph_inplace(g)

        assert "py::svc::User.email" not in g.nodes
        assert metrics["nodes_removed"] == 1

        # The edge survives, rewired src=User instead of User.email.
        edges = list(g.out_edges("py::svc::User", data=True))
        assert len(edges) == 1
        u, v, data = edges[0]
        assert v == "py::val::EmailValidator"

        # ``via=`` annotation preserves the dropped endpoint identity.
        anns = data.get("annotations") or {}
        via = anns.get("via") or []
        assert any("email" in entry and "@L42" in entry for entry in via), (
            f"expected via=src=email@L42, got {via}"
        )

    def test_go_bare_parent_resolves_via_own_module(self):
        """Go ``parent_symbol`` is a bare type name; resolver pairs it
        with the noise node's own module."""
        g = nx.MultiDiGraph()
        _add_node(g, "go::user::User",
                  symbol_type="struct", language="go")
        _add_node(g, "go::user::User.Name",
                  symbol_type="field",
                  parent_symbol="User",  # bare, no dot
                  language="go")
        _add_node(g, "go::format::Stringer",
                  symbol_type="interface", language="go")
        g.add_edge("go::user::User.Name",
                   "go::format::Stringer",
                   relationship_type="implements")

        metrics = contract_graph_inplace(g)

        assert "go::user::User.Name" not in g.nodes
        assert metrics["by_language"].get("go") == 1
        # Edge rewired onto the User struct node.
        out = list(g.out_edges("go::user::User", data=True))
        assert len(out) == 1 and out[0][1] == "go::format::Stringer"

    def test_go_package_level_var_promotes_to_module(self):
        """Empty parent_symbol (Go package-level var) anchors to the
        module node ``go::pkg::pkg``."""
        g = nx.MultiDiGraph()
        _add_node(g, "go::config::config",
                  symbol_type="module", language="go")
        _add_node(g, "go::config::DefaultTimeout",
                  symbol_type="variable",
                  parent_symbol="",  # package-level
                  language="go")
        _add_node(g, "go::http::Client", symbol_type="class", language="go")
        g.add_edge("go::config::DefaultTimeout",
                   "go::http::Client",
                   relationship_type="references")

        contract_graph_inplace(g)

        assert "go::config::DefaultTimeout" not in g.nodes
        out = list(g.out_edges("go::config::config", data=True))
        assert len(out) == 1


# ─── 3. Idempotency / safety ──────────────────────────────────────────

class TestContractionSafety:

    def test_no_noise_no_changes(self):
        """Pure-arch graph is invariant under contraction."""
        g = nx.MultiDiGraph()
        _add_node(g, "py::a::A", symbol_type="class")
        _add_node(g, "py::a::B", symbol_type="class")
        g.add_edge("py::a::A", "py::a::B", relationship_type="extends")

        before = (g.number_of_nodes(), g.number_of_edges())
        metrics = contract_graph_inplace(g)
        after = (g.number_of_nodes(), g.number_of_edges())

        assert before == after
        assert metrics["nodes_removed"] == 0
        assert metrics["edges_rewritten"] == 0

    def test_unresolved_noise_remains(self):
        """Noise node whose parent can't be resolved stays in the graph
        and counts as ``unresolved`` in metrics."""
        g = nx.MultiDiGraph()
        _add_node(g, "py::a::Orphan",
                  symbol_type="parameter",
                  parent_symbol="nonexistent.parent")
        _add_node(g, "py::a::Other", symbol_type="class")
        g.add_edge("py::a::Orphan", "py::a::Other",
                   relationship_type="uses")

        metrics = contract_graph_inplace(g)

        assert "py::a::Orphan" in g.nodes
        assert metrics["nodes_removed"] == 0
        assert metrics["unresolved"] == 1

    def test_duplicate_edges_after_rewriting_merge_via(self):
        """Two field nodes referencing the same external class collapse
        to one edge whose ``via`` list captures both fields."""
        g = nx.MultiDiGraph()
        _add_node(g, "py::svc::User", symbol_type="class")
        _add_node(g, "py::val::Validator", symbol_type="class")
        _add_node(g, "py::svc::User.email",
                  symbol_type="field",
                  parent_symbol="svc.User",
                  symbol_name="email", start_line=10)
        _add_node(g, "py::svc::User.username",
                  symbol_type="field",
                  parent_symbol="svc.User",
                  symbol_name="username", start_line=11)
        g.add_edge("py::svc::User.email",
                   "py::val::Validator",
                   relationship_type="references")
        g.add_edge("py::svc::User.username",
                   "py::val::Validator",
                   relationship_type="references")

        contract_graph_inplace(g)

        edges = list(g.out_edges("py::svc::User", data=True))
        # Both rewired edges share the same (u, v, rel_type) tuple →
        # merged into one with combined via list.
        ref_edges = [e for e in edges
                     if e[2].get("relationship_type") == "references"]
        assert len(ref_edges) == 1
        anns = ref_edges[0][2].get("annotations") or {}
        via = anns.get("via") or []
        labels = " ".join(via)
        assert "email" in labels and "username" in labels

    def test_metrics_shape(self):
        """Return value carries the documented keys."""
        g = nx.MultiDiGraph()
        _add_node(g, "py::a::A", symbol_type="class")
        metrics = contract_graph_inplace(g)
        for key in ("nodes_removed", "edges_rewritten",
                    "self_loops_dropped", "unresolved", "by_language"):
            assert key in metrics
