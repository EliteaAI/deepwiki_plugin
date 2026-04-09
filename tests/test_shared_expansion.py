"""
Phase 4 tests — Smart-expansion parity (shared_expansion module).

Validates:
 - Alias chain resolution via DB
 - Per-symbol-type bidirectional expansion strategies
 - Cluster boundary enforcement in shared_expansion layer
 - Smart expansion integration into cluster_expansion.expand_for_page
 - Implementation grounding (cross-file context recovery)
"""

import sqlite3
import pytest
from unittest.mock import patch

from plugin_implementation.code_graph.shared_expansion import (
    resolve_alias_chain_db,
    expand_symbol_smart,
    _expand_class_db,
    _expand_function_db,
    _expand_constant_db,
    _expand_type_alias_db,
    _expand_macro_db,
    _expand_generic_db,
    _is_valid_expansion,
    _fetch_node,
)


# ─── Fixtures ────────────────────────────────────────────────────────


def _make_db():
    """Create an in-memory SQLite DB with repo_nodes and repo_edges tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE repo_nodes (
            node_id TEXT PRIMARY KEY,
            symbol_name TEXT,
            symbol_type TEXT,
            rel_path TEXT,
            file_name TEXT,
            language TEXT,
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
            weight REAL DEFAULT 1.0
        )
    """)
    return conn


def _add_node(conn, node_id, symbol_name, symbol_type,
              macro_cluster=0, micro_cluster=0, **kwargs):
    defaults = dict(
        rel_path=f"{symbol_name}.py",
        file_name=f"{symbol_name}.py",
        language="python",
        start_line=1, end_line=10,
        source_text=f"# {symbol_name}",
        is_architectural=1, is_doc=0,
    )
    defaults.update(kwargs)
    conn.execute(
        "INSERT INTO repo_nodes "
        "(node_id, symbol_name, symbol_type, rel_path, file_name, language, "
        "start_line, end_line, source_text, is_architectural, is_doc, "
        "macro_cluster, micro_cluster) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (node_id, symbol_name, symbol_type,
         defaults["rel_path"], defaults["file_name"], defaults["language"],
         defaults["start_line"], defaults["end_line"], defaults["source_text"],
         defaults["is_architectural"], defaults["is_doc"],
         macro_cluster, micro_cluster),
    )


def _add_edge(conn, src, tgt, rel_type, weight=1.0):
    conn.execute(
        "INSERT INTO repo_edges (source_id, target_id, rel_type, weight) "
        "VALUES (?,?,?,?)",
        (src, tgt, rel_type, weight),
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. Alias chain resolution
# ═══════════════════════════════════════════════════════════════════════

class TestAliasChainResolution:

    def test_single_hop_alias(self):
        conn = _make_db()
        _add_node(conn, "alias1", "MyAlias", "type_alias")
        _add_node(conn, "concrete1", "ConcreteClass", "class")
        _add_edge(conn, "alias1", "concrete1", "alias_of")

        result = resolve_alias_chain_db(conn, "alias1")
        assert result == "concrete1"

    def test_multi_hop_alias(self):
        conn = _make_db()
        _add_node(conn, "a1", "AliasOne", "type_alias")
        _add_node(conn, "a2", "AliasTwo", "type_alias")
        _add_node(conn, "a3", "AliasThree", "type_alias")
        _add_node(conn, "concrete", "FinalClass", "class")
        _add_edge(conn, "a1", "a2", "alias_of")
        _add_edge(conn, "a2", "a3", "alias_of")
        _add_edge(conn, "a3", "concrete", "alias_of")

        result = resolve_alias_chain_db(conn, "a1")
        assert result == "concrete"

    def test_alias_cycle_detected(self):
        conn = _make_db()
        _add_node(conn, "c1", "CycleA", "type_alias")
        _add_node(conn, "c2", "CycleB", "type_alias")
        _add_edge(conn, "c1", "c2", "alias_of")
        _add_edge(conn, "c2", "c1", "alias_of")

        result = resolve_alias_chain_db(conn, "c1")
        # Should not infinite-loop; returns last non-starting node
        assert result is not None

    def test_no_alias_edge(self):
        conn = _make_db()
        _add_node(conn, "standalone", "X", "type_alias")

        result = resolve_alias_chain_db(conn, "standalone")
        # No alias_of edges → returns None
        assert result is None

    def test_max_hops_limit(self):
        conn = _make_db()
        for i in range(10):
            _add_node(conn, f"hop{i}", f"Alias{i}", "type_alias")
        for i in range(9):
            _add_edge(conn, f"hop{i}", f"hop{i+1}", "alias_of")

        result = resolve_alias_chain_db(conn, "hop0", max_hops=3)
        # Should stop after 3 hops → hop3
        assert result == "hop3"


# ═══════════════════════════════════════════════════════════════════════
# 2. Class expansion (bidirectional)
# ═══════════════════════════════════════════════════════════════════════

class TestClassExpansion:

    def _build_class_graph(self):
        conn = _make_db()
        _add_node(conn, "base", "BaseClass", "class", macro_cluster=1)
        _add_node(conn, "child", "ChildClass", "class", macro_cluster=1)
        _add_node(conn, "composed", "ComposedType", "class", macro_cluster=1)
        _add_node(conn, "user", "UserFunc", "function", macro_cluster=1)
        _add_node(conn, "outside", "OutsideClass", "class", macro_cluster=2)

        # child inherits base
        _add_edge(conn, "child", "base", "inheritance", 3.0)
        # child creates composed
        _add_edge(conn, "child", "composed", "creates", 2.0)
        # child composes base
        _add_edge(conn, "child", "base", "composition", 1.5)
        # user references child
        _add_edge(conn, "user", "child", "references", 1.0)
        # child references outside (cross-cluster)
        _add_edge(conn, "child", "outside", "references", 1.0)
        return conn

    def test_class_finds_base_and_derived(self):
        conn = self._build_class_graph()
        results = _expand_class_db(conn, "child", set(), None, macro_id=1)
        nids = {r[0] for r in results}
        assert "base" in nids, "Should find base class via inheritance"

    def test_class_finds_creates(self):
        conn = self._build_class_graph()
        results = _expand_class_db(conn, "child", set(), None, macro_id=1)
        nids = {r[0] for r in results}
        assert "composed" in nids, "Should find created types"

    def test_class_respects_cluster_boundary(self):
        conn = self._build_class_graph()
        results = _expand_class_db(conn, "child", set(), None, macro_id=1)
        nids = {r[0] for r in results}
        assert "outside" not in nids, "Should not cross macro-cluster boundary"

    def test_class_respects_page_boundary(self):
        conn = self._build_class_graph()
        page_ids = {"child", "base"}  # only these in the page
        results = _expand_class_db(conn, "child", set(), page_ids, macro_id=1)
        nids = {r[0] for r in results}
        assert "composed" not in nids, "Should not include node outside page boundary"
        assert "base" in nids or len(nids) == 0  # base is in page boundary

    def test_class_backward_inheritance(self):
        conn = self._build_class_graph()
        # Expand base → should find child inheriting it
        results = _expand_class_db(conn, "base", set(), None, macro_id=1)
        nids = {r[0] for r in results}
        assert "child" in nids, "Should find derived classes via backward inheritance"


# ═══════════════════════════════════════════════════════════════════════
# 3. Function expansion
# ═══════════════════════════════════════════════════════════════════════

class TestFunctionExpansion:

    def _build_func_graph(self):
        conn = _make_db()
        _add_node(conn, "f1", "processData", "function", macro_cluster=1)
        _add_node(conn, "f2", "helperFunc", "function", macro_cluster=1)
        _add_node(conn, "f3", "callerFunc", "function", macro_cluster=1)
        _add_node(conn, "cls", "DataStore", "class", macro_cluster=1)
        _add_node(conn, "outside_f", "outsideFunc", "function", macro_cluster=2)

        # f1 calls f2
        _add_edge(conn, "f1", "f2", "calls", 2.0)
        # f3 calls f1 (f1 has caller)
        _add_edge(conn, "f3", "f1", "calls", 2.0)
        # f1 creates cls
        _add_edge(conn, "f1", "cls", "creates", 1.5)
        # f1 calls outside (cross-cluster)
        _add_edge(conn, "f1", "outside_f", "calls", 1.0)
        return conn

    def test_function_finds_callees(self):
        conn = self._build_func_graph()
        results = _expand_function_db(conn, "f1", set(), None, macro_id=1)
        nids = {r[0] for r in results}
        assert "f2" in nids, "Should find callee"

    def test_function_finds_callers(self):
        conn = self._build_func_graph()
        results = _expand_function_db(conn, "f1", set(), None, macro_id=1)
        nids = {r[0] for r in results}
        assert "f3" in nids, "Should find caller"

    def test_function_respects_cluster_boundary(self):
        conn = self._build_func_graph()
        results = _expand_function_db(conn, "f1", set(), None, macro_id=1)
        nids = {r[0] for r in results}
        assert "outside_f" not in nids

    def test_function_finds_creates(self):
        conn = self._build_func_graph()
        results = _expand_function_db(conn, "f1", set(), None, macro_id=1)
        nids = {r[0] for r in results}
        assert "cls" in nids, "Should find created types"


# ═══════════════════════════════════════════════════════════════════════
# 4. Constant expansion
# ═══════════════════════════════════════════════════════════════════════

class TestConstantExpansion:

    def test_constant_finds_referencers(self):
        conn = _make_db()
        _add_node(conn, "const1", "MAX_SIZE", "constant", macro_cluster=1)
        _add_node(conn, "func1", "validate", "function", macro_cluster=1)
        _add_edge(conn, "func1", "const1", "references", 1.0)

        results = _expand_constant_db(conn, "const1", set(), None, macro_id=1)
        nids = {r[0] for r in results}
        assert "func1" in nids, "Should find who references this constant"


# ═══════════════════════════════════════════════════════════════════════
# 5. Type alias expansion
# ═══════════════════════════════════════════════════════════════════════

class TestTypeAliasExpansion:

    def test_type_alias_resolves_chain(self):
        conn = _make_db()
        _add_node(conn, "alias1", "MyType", "type_alias", macro_cluster=1)
        _add_node(conn, "concrete", "RealClass", "class", macro_cluster=1)
        _add_edge(conn, "alias1", "concrete", "alias_of")

        results = _expand_type_alias_db(conn, "alias1", set(), None, macro_id=1)
        nids = {r[0] for r in results}
        assert "concrete" in nids, "Should resolve alias to concrete type"

    def test_type_alias_finds_usage_sites(self):
        conn = _make_db()
        _add_node(conn, "alias", "Handler", "type_alias", macro_cluster=1)
        _add_node(conn, "user", "Router", "class", macro_cluster=1)
        _add_edge(conn, "user", "alias", "references", 1.0)

        results = _expand_type_alias_db(conn, "alias", set(), None, macro_id=1)
        nids = {r[0] for r in results}
        assert "user" in nids, "Should find usage sites"


# ═══════════════════════════════════════════════════════════════════════
# 6. Macro expansion
# ═══════════════════════════════════════════════════════════════════════

class TestMacroExpansion:

    def test_macro_finds_usage_sites(self):
        conn = _make_db()
        _add_node(conn, "mac", "DEFINE_FLAG", "macro", macro_cluster=1)
        _add_node(conn, "user", "FlagHandler", "function", macro_cluster=1)
        _add_edge(conn, "user", "mac", "calls", 1.0)

        results = _expand_macro_db(conn, "mac", set(), None, macro_id=1)
        nids = {r[0] for r in results}
        assert "user" in nids


# ═══════════════════════════════════════════════════════════════════════
# 7. Dispatcher (expand_symbol_smart)
# ═══════════════════════════════════════════════════════════════════════

class TestExpandSymbolSmart:

    def _build_mixed_graph(self):
        conn = _make_db()
        _add_node(conn, "cls", "MyService", "class", macro_cluster=1)
        _add_node(conn, "func", "handler", "function", macro_cluster=1)
        _add_node(conn, "const", "TIMEOUT", "constant", macro_cluster=1)
        _add_node(conn, "alias", "Config", "type_alias", macro_cluster=1)
        _add_node(conn, "macro", "LOG", "macro", macro_cluster=1)
        _add_node(conn, "unknown", "mysterious", "widget", macro_cluster=1)
        _add_node(conn, "outside", "ExternalSvc", "class", macro_cluster=2)

        _add_edge(conn, "cls", "func", "references", 2.0)
        _add_edge(conn, "func", "const", "references", 1.0)
        _add_edge(conn, "alias", "cls", "alias_of")
        _add_edge(conn, "macro", "func", "references", 1.0)
        _add_edge(conn, "cls", "outside", "references", 1.0)
        return conn

    def test_dispatches_class(self):
        conn = self._build_mixed_graph()
        results = expand_symbol_smart(
            conn, "cls", "class", set(), macro_id=1)
        nids = {r[0] for r in results}
        assert "func" in nids, "Class should expand to its callees"
        assert "outside" not in nids, "Should respect cluster boundary"

    def test_dispatches_function(self):
        conn = self._build_mixed_graph()
        results = expand_symbol_smart(
            conn, "func", "function", set(), macro_id=1)
        nids = {r[0] for r in results}
        assert "const" in nids, "Function should find referenced constants"

    def test_dispatches_type_alias(self):
        conn = self._build_mixed_graph()
        results = expand_symbol_smart(
            conn, "alias", "type_alias", set(), macro_id=1)
        nids = {r[0] for r in results}
        assert "cls" in nids, "Type alias should resolve to concrete class"

    def test_dispatches_unknown_type(self):
        conn = self._build_mixed_graph()
        # "widget" is not a known type → generic expansion
        _add_edge(conn, "unknown", "func", "calls", 1.0)
        results = expand_symbol_smart(
            conn, "unknown", "widget", set(), macro_id=1)
        nids = {r[0] for r in results}
        assert "func" in nids, "Generic expansion should find neighbors"

    def test_respects_seen_ids(self):
        conn = self._build_mixed_graph()
        results = expand_symbol_smart(
            conn, "cls", "class", {"func"}, macro_id=1)
        nids = {r[0] for r in results}
        assert "func" not in nids, "Should not include already-seen nodes"

    def test_per_symbol_budget(self):
        conn = _make_db()
        _add_node(conn, "root", "Root", "class", macro_cluster=1)
        for i in range(20):
            nid = f"child{i}"
            _add_node(conn, nid, f"Child{i}", "class", macro_cluster=1)
            _add_edge(conn, "root", nid, "inheritance", float(20 - i))

        results = expand_symbol_smart(
            conn, "root", "class", set(), macro_id=1, per_symbol_budget=5)
        assert len(results) <= 5, "Should respect per_symbol_budget"


# ═══════════════════════════════════════════════════════════════════════
# 8. Validation helper
# ═══════════════════════════════════════════════════════════════════════

class TestValidation:

    def test_non_architectural_rejected(self):
        node = {"node_id": "x", "is_architectural": 0, "symbol_type": "class"}
        assert not _is_valid_expansion(node, None, None)

    def test_non_expansion_type_rejected(self):
        node = {"node_id": "x", "is_architectural": 1, "symbol_type": "method"}
        assert not _is_valid_expansion(node, None, None)

    def test_page_boundary_enforced(self):
        node = {"node_id": "x", "is_architectural": 1, "symbol_type": "class",
                "macro_cluster": 1}
        assert not _is_valid_expansion(node, page_boundary_ids={"y"}, macro_id=1)
        assert _is_valid_expansion(node, page_boundary_ids={"x"}, macro_id=1)

    def test_macro_cluster_enforced(self):
        node = {"node_id": "x", "is_architectural": 1, "symbol_type": "class",
                "macro_cluster": 2}
        assert not _is_valid_expansion(node, None, macro_id=1)

    def test_none_node_rejected(self):
        assert not _is_valid_expansion(None, None, None)


# ═══════════════════════════════════════════════════════════════════════
# 9. Integration: smart expansion via feature flag in cluster_expansion
# ═══════════════════════════════════════════════════════════════════════

class TestSmartExpansionIntegration:
    """Verify that expand_for_page uses smart expansion when the flag is on."""

    def _build_db_with_wrapper(self):
        """Create a mock UnifiedWikiDB with a real sqlite3 connection."""
        conn = _make_db()

        # Build a small cluster: AuthService → LoginHandler → SessionStore
        _add_node(conn, "auth", "AuthService", "class",
                  macro_cluster=1, micro_cluster=1,
                  source_text="class AuthService:\n    pass")
        _add_node(conn, "login", "LoginHandler", "function",
                  macro_cluster=1, micro_cluster=1,
                  source_text="def LoginHandler():\n    pass")
        _add_node(conn, "session", "SessionStore", "class",
                  macro_cluster=1, micro_cluster=1,
                  source_text="class SessionStore:\n    pass")
        _add_node(conn, "outside", "PaymentService", "class",
                  macro_cluster=2, micro_cluster=5,
                  source_text="class PaymentService:\n    pass")

        # auth → login (calls), auth → session (creates)
        _add_edge(conn, "auth", "login", "calls", 2.0)
        _add_edge(conn, "auth", "session", "creates", 2.0)
        # login → outside (cross-cluster)
        _add_edge(conn, "login", "outside", "references", 1.0)

        class MockDB:
            def __init__(self, connection):
                self.conn = connection

        return MockDB(conn)

    def test_smart_expansion_enabled(self):
        from plugin_implementation.cluster_expansion import expand_for_page
        from plugin_implementation.feature_flags import FeatureFlags

        db = self._build_db_with_wrapper()

        with patch("plugin_implementation.cluster_expansion.get_feature_flags") as mock_flags:
            mock_flags.return_value = FeatureFlags(smart_expansion=True)
            docs = expand_for_page(
                db,
                page_symbols=["AuthService"],
                macro_id=1,
                micro_id=1,
                cluster_node_ids=["auth", "login", "session"],
                token_budget=50000,
            )

        # Initial symbol should be present
        initial = [d for d in docs if d.metadata.get("is_initially_retrieved")]
        expanded = [d for d in docs if not d.metadata.get("is_initially_retrieved")]

        assert len(initial) >= 1, "Should have at least AuthService as initial"

        expanded_names = {d.metadata.get("symbol_name") for d in expanded}
        # Smart expansion should find LoginHandler and SessionStore
        # via calls and creates edges
        assert "LoginHandler" in expanded_names or "SessionStore" in expanded_names, \
            f"Smart expansion should find neighbors, got: {expanded_names}"

        all_names = {d.metadata.get("symbol_name") for d in docs}
        assert "PaymentService" not in all_names, \
            "Should NOT include cross-cluster node"

    def test_smart_expansion_disabled_uses_legacy(self):
        from plugin_implementation.cluster_expansion import expand_for_page
        from plugin_implementation.feature_flags import FeatureFlags

        db = self._build_db_with_wrapper()

        with patch("plugin_implementation.cluster_expansion.get_feature_flags") as mock_flags:
            mock_flags.return_value = FeatureFlags(smart_expansion=False)
            docs = expand_for_page(
                db,
                page_symbols=["AuthService"],
                macro_id=1,
                micro_id=1,
                cluster_node_ids=["auth", "login", "session"],
                token_budget=50000,
            )

        # Legacy path should also find neighbors (via generic 1-hop)
        all_names = {d.metadata.get("symbol_name") for d in docs}
        assert "PaymentService" not in all_names, \
            "Legacy path should also respect page boundaries"

    def test_smart_expansion_page_boundary_strict(self):
        """Smart expansion must not include nodes outside cluster_node_ids."""
        from plugin_implementation.cluster_expansion import expand_for_page
        from plugin_implementation.feature_flags import FeatureFlags

        db = self._build_db_with_wrapper()

        with patch("plugin_implementation.cluster_expansion.get_feature_flags") as mock_flags:
            mock_flags.return_value = FeatureFlags(smart_expansion=True)
            # Only auth and login in page, NOT session
            docs = expand_for_page(
                db,
                page_symbols=["AuthService"],
                macro_id=1,
                micro_id=1,
                cluster_node_ids=["auth", "login"],
                token_budget=50000,
            )

        all_names = {d.metadata.get("symbol_name") for d in docs}
        assert "SessionStore" not in all_names, \
            "Session is outside page boundary (cluster_node_ids)"
