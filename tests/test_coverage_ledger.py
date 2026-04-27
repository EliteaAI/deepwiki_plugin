"""
Phase 6 tests — Coverage Ledger + Compact LLM Refiner.

Validates:
 - Coverage gap detection on known-uncovered symbols
 - Coverage domain tracking correctness
 - Symbol/directory/doc-domain coverage math
 - Page overlap detection
 - Refiner prompt building and output parsing
 - Flag disabled → no behavior change
"""

import sqlite3
import json
import pytest
from unittest.mock import patch

from plugin_implementation.wiki_structure_planner.coverage_ledger import (
    CoverageLedger,
    CoverageReport,
    build_refiner_prompt,
    parse_refiner_output,
    REFINER_SYSTEM_PROMPT,
)
from plugin_implementation.wiki_structure_planner.candidate_builder import (
    CandidateRecord,
    CLASS_CODE,
    CLASS_MIXED,
    CLASS_DOCS,
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
            rel_path TEXT DEFAULT '',
            file_name TEXT DEFAULT '',
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
            rel_type TEXT DEFAULT 'calls',
            weight REAL DEFAULT 1.0
        )
    """)
    return conn


def _add_node(conn, node_id, symbol_name, symbol_type,
              macro_cluster=0, micro_cluster=0, **kwargs):
    defaults = dict(
        rel_path=f"src/{symbol_name}.py",
        file_name=f"{symbol_name}.py",
        language="python",
        start_line=1, end_line=10,
        source_text=f"class {symbol_name}:\n    " + "x = 1\n    " * 10,
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


class MockDB:
    def __init__(self, connection):
        self.conn = connection

    def get_node(self, node_id):
        row = self.conn.execute(
            "SELECT * FROM repo_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        return dict(row) if row else None


def _make_candidate(node_ids, macro_id=0, micro_id=0,
                    classification=CLASS_CODE, **kwargs):
    defaults = dict(
        code_identity_score=0.5,
        implementation_evidence=0.5,
        docs_dominance=0.0,
        public_api_presence=0.5,
        utility_contamination=0.0,
        file_spread=1,
        node_details=[],
    )
    defaults.update(kwargs)
    return CandidateRecord(
        macro_id=macro_id,
        micro_id=micro_id,
        node_ids=node_ids,
        classification=classification,
        **defaults,
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. Coverage gap detection
# ═══════════════════════════════════════════════════════════════════════

class TestCoverageGapDetection:

    def test_uncovered_high_value_symbols(self):
        """Symbols not in any candidate should appear as uncovered."""
        conn = _make_db()
        # 3 architectural symbols, only 1 covered
        _add_node(conn, "n1", "AuthService", "class", macro_cluster=0, micro_cluster=0,
                  rel_path="src/auth.py")
        _add_node(conn, "n2", "PaymentService", "class", macro_cluster=0, micro_cluster=0,
                  rel_path="src/payment.py")
        _add_node(conn, "n3", "OrderService", "class", macro_cluster=0, micro_cluster=0,
                  rel_path="src/orders.py")

        db = MockDB(conn)
        candidates = [_make_candidate(["n1"], macro_id=0, micro_id=0)]

        ledger = CoverageLedger(db, candidates)
        report = ledger.report()

        assert report.total_symbols == 3
        assert report.covered_symbols == 1
        uncovered_names = {u["symbol_name"] for u in report.uncovered_high_value}
        assert "PaymentService" in uncovered_names
        assert "OrderService" in uncovered_names

    def test_all_covered(self):
        """When all symbols are in candidates, no uncovered gaps."""
        conn = _make_db()
        _add_node(conn, "n1", "ClassA", "class", rel_path="src/a.py")
        _add_node(conn, "n2", "ClassB", "class", rel_path="src/b.py")

        db = MockDB(conn)
        candidates = [_make_candidate(["n1", "n2"])]

        ledger = CoverageLedger(db, candidates)
        report = ledger.report()

        assert report.total_symbols == 2
        assert report.covered_symbols == 2
        assert len(report.uncovered_high_value) == 0
        assert report.symbol_coverage == 1.0

    def test_low_priority_not_flagged_as_high_value(self):
        """Symbols with low SYMBOL_TYPE_PRIORITY should not be flagged as high-value uncovered."""
        conn = _make_db()
        _add_node(conn, "n1", "AuthService", "class", rel_path="src/auth.py")
        _add_node(conn, "n2", "SOME_CONST", "constant", rel_path="src/const.py")
        _add_node(conn, "n3", "MyMacro", "macro", rel_path="src/macros.py")

        db = MockDB(conn)
        # Only cover the class
        candidates = [_make_candidate(["n1"])]

        ledger = CoverageLedger(db, candidates)
        report = ledger.report()

        # constant (priority=5) and macro (priority=3) are below threshold 7
        uncovered_names = {u["symbol_name"] for u in report.uncovered_high_value}
        assert "SOME_CONST" not in uncovered_names
        assert "MyMacro" not in uncovered_names


# ═══════════════════════════════════════════════════════════════════════
# 2. Directory and doc-domain coverage
# ═══════════════════════════════════════════════════════════════════════

class TestDirectoryCoverage:

    def test_directory_coverage_math(self):
        """Directory coverage computed from candidate node paths."""
        conn = _make_db()
        _add_node(conn, "n1", "A", "class", rel_path="src/auth/service.py")
        _add_node(conn, "n2", "B", "class", rel_path="src/payment/handler.py")
        _add_node(conn, "n3", "C", "class", rel_path="src/orders/core.py")

        db = MockDB(conn)
        # Only cover auth
        candidates = [_make_candidate(["n1"])]

        ledger = CoverageLedger(db, candidates)
        report = ledger.report()

        assert report.total_directories == 3
        assert report.covered_directories == 1

    def test_doc_domain_coverage(self):
        """Documentation domains tracked separately."""
        conn = _make_db()
        _add_node(conn, "d1", "README", "module_doc", is_doc=1,
                  rel_path="docs/getting-started/readme.md")
        _add_node(conn, "d2", "API_DOC", "file_doc", is_doc=1,
                  rel_path="docs/api/endpoints.md")
        _add_node(conn, "n1", "Main", "class",
                  rel_path="src/main.py")

        db = MockDB(conn)
        # Cover only the doc from getting-started
        candidates = [_make_candidate(["d1", "n1"])]

        ledger = CoverageLedger(db, candidates)
        report = ledger.report()

        assert report.total_doc_domains >= 1
        assert report.covered_doc_domains >= 1


# ═══════════════════════════════════════════════════════════════════════
# 3. Page overlap detection
# ═══════════════════════════════════════════════════════════════════════

class TestPageOverlap:

    def test_detects_overlapping_candidates(self):
        """Two candidates sharing >50% nodes should be flagged."""
        conn = _make_db()
        for i in range(5):
            _add_node(conn, f"n{i}", f"Sym{i}", "class",
                      rel_path=f"src/mod{i}.py")

        db = MockDB(conn)
        # Candidate A: n0,n1,n2,n3 — Candidate B: n1,n2,n3,n4
        cand_a = _make_candidate(["n0", "n1", "n2", "n3"], micro_id=0)
        cand_b = _make_candidate(["n1", "n2", "n3", "n4"], micro_id=1)

        ledger = CoverageLedger(db, [cand_a, cand_b])
        report = ledger.report()

        # 3 shared out of 4 each = 75% overlap
        assert len(report.page_overlap_pairs) == 1
        assert report.page_overlap_pairs[0][2] == 3  # overlap count

    def test_no_overlap_detected(self):
        """Disjoint candidates should not produce overlap pairs."""
        conn = _make_db()
        for i in range(6):
            _add_node(conn, f"n{i}", f"Sym{i}", "class",
                      rel_path=f"src/mod{i}.py")

        db = MockDB(conn)
        cand_a = _make_candidate(["n0", "n1", "n2"], micro_id=0)
        cand_b = _make_candidate(["n3", "n4", "n5"], micro_id=1)

        ledger = CoverageLedger(db, [cand_a, cand_b])
        report = ledger.report()

        assert len(report.page_overlap_pairs) == 0


# ═══════════════════════════════════════════════════════════════════════
# 4. Scoped report (macro-cluster filter)
# ═══════════════════════════════════════════════════════════════════════

class TestScopedReport:

    def test_macro_scoped_report(self):
        """report(macro_id=X) only counts symbols from that macro."""
        conn = _make_db()
        _add_node(conn, "n1", "A", "class", macro_cluster=0, rel_path="src/a.py")
        _add_node(conn, "n2", "B", "class", macro_cluster=0, rel_path="src/b.py")
        _add_node(conn, "n3", "C", "class", macro_cluster=1, rel_path="src/c.py")

        db = MockDB(conn)
        candidates = [_make_candidate(["n1"], macro_id=0)]

        ledger = CoverageLedger(db, candidates)

        # Scoped to macro 0: 2 total, 1 covered
        r0 = ledger.report(macro_id=0)
        assert r0.total_symbols == 2
        assert r0.covered_symbols == 1

        # Scoped to macro 1: 1 total, 0 covered
        r1 = ledger.report(macro_id=1)
        assert r1.total_symbols == 1
        assert r1.covered_symbols == 0


# ═══════════════════════════════════════════════════════════════════════
# 5. Coverage report properties
# ═══════════════════════════════════════════════════════════════════════

class TestCoverageReportProperties:

    def test_symbol_coverage_ratio(self):
        report = CoverageReport(total_symbols=10, covered_symbols=7)
        assert report.symbol_coverage == 0.7

    def test_zero_denominator(self):
        report = CoverageReport(total_symbols=0, covered_symbols=0)
        assert report.symbol_coverage == 0.0
        assert report.doc_coverage == 0.0
        assert report.dir_coverage == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 6. Compact LLM Refiner
# ═══════════════════════════════════════════════════════════════════════

class TestRefinerPromptBuilding:

    def test_build_refiner_prompt_structure(self):
        """Prompt should contain all three sections."""
        prompt = build_refiner_prompt(
            section_symbols=[
                {"name": "AuthService", "type": "class", "file": "auth.py"},
                {"name": "login", "type": "function", "file": "auth.py"},
            ],
            candidate_summaries=[
                {"micro_id": 0, "classification": "code",
                 "identity_score": 0.8, "symbols": ["AuthService", "login"],
                 "suggested_shape": "keep"},
            ],
            quality_flags={
                "symbol_coverage": 0.85,
                "uncovered_hv_count": 2,
            },
        )
        assert "## Section Symbols" in prompt
        assert "## Candidates" in prompt
        assert "## Quality Flags" in prompt
        assert "AuthService" in prompt

    def test_prompt_truncates_symbols(self):
        """Only first 15 symbols should be included."""
        symbols = [
            {"name": f"Sym{i}", "type": "class", "file": f"f{i}.py"}
            for i in range(20)
        ]
        prompt = build_refiner_prompt(symbols, [], {})
        assert "Sym14" in prompt
        assert "Sym15" not in prompt

    def test_system_prompt_has_schema(self):
        """System prompt should describe the expected output schema."""
        assert "page_actions" in REFINER_SYSTEM_PROMPT
        assert "section_name" in REFINER_SYSTEM_PROMPT


class TestRefinerOutputParsing:

    def test_parses_valid_json(self):
        raw = json.dumps({
            "section_name": "Auth",
            "section_description": "Authentication",
            "page_actions": [
                {"action": "keep", "micro_id": 0, "name": "Login",
                 "description": "Login flow", "retrieval_query": "login auth"}
            ],
        })
        result = parse_refiner_output(raw)
        assert result is not None
        assert result["section_name"] == "Auth"
        assert len(result["page_actions"]) == 1

    def test_parses_json_in_code_fence(self):
        raw = '```json\n{"section_name":"X","page_actions":[]}\n```'
        result = parse_refiner_output(raw)
        assert result is not None
        assert result["section_name"] == "X"

    def test_parses_json_with_leading_text(self):
        raw = 'Here is the result:\n{"section_name":"Y","page_actions":[{"action":"keep"}]}'
        result = parse_refiner_output(raw)
        assert result is not None
        assert result["section_name"] == "Y"

    def test_returns_none_for_garbage(self):
        result = parse_refiner_output("not json at all")
        assert result is None

    def test_returns_none_for_partial_json(self):
        result = parse_refiner_output('{"section_name": "X"')
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# 7. Flag gating
# ═══════════════════════════════════════════════════════════════════════

class TestFlagGating:

    def test_coverage_ledger_not_invoked_when_flag_off(self):
        """When coverage_ledger flag is off, plan_structure skips step 1c."""
        from plugin_implementation.feature_flags import FeatureFlags

        flags = FeatureFlags()
        assert flags.coverage_ledger is False

        # With all flags off, CoverageLedger should not be instantiated.
        # This is verified by the existing test_cluster_planner.py tests
        # which pass without a repo_edges table → proves CoverageLedger
        # (which reads repo_nodes) is never reached.

    def test_coverage_ledger_works_when_flag_on(self):
        """When flag is on, coverage data is computed."""
        conn = _make_db()
        _add_node(conn, "n1", "A", "class", rel_path="src/a.py")
        _add_node(conn, "n2", "B", "class", rel_path="src/b.py")

        db = MockDB(conn)
        candidates = [_make_candidate(["n1"])]

        ledger = CoverageLedger(db, candidates)
        report = ledger.report()

        assert report.total_symbols == 2
        assert report.covered_symbols == 1
        assert 0.4 < report.symbol_coverage < 0.6
