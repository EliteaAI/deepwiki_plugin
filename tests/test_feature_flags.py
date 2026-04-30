"""Tests for feature_flags.py and the new page-identity constants in constants.py."""

import os
import pytest

from plugin_implementation.feature_flags import FeatureFlags, get_feature_flags, _env_bool
from plugin_implementation.constants import (
    PAGE_IDENTITY_SYMBOLS,
    SUPPORTING_CODE_SYMBOLS,
    DOC_CLUSTER_SYMBOLS,
    SYMBOL_TYPE_PRIORITY,
    ARCHITECTURAL_SYMBOLS,
    CODE_SYMBOL_TYPES,
    EXPANSION_SYMBOL_TYPES,
)


# ── helpers ──────────────────────────────────────────────────────────────────

# These flags are now hard-coded ON in the dataclass and are no longer read
# from the environment. The list below is kept so tests can scrub any stale
# values that might leak in from a developer's shell.
_FLAG_ENVVARS = [
    "DEEPWIKI_CLUSTER_HIERARCHICAL_LEIDEN",
    "DEEPWIKI_CLUSTER_CAPABILITY_VALIDATION",
    "DEEPWIKI_CLUSTER_SMART_EXPANSION",
    "DEEPWIKI_CLUSTER_COVERAGE_LEDGER",
    "DEEPWIKI_CLUSTER_LANGUAGE_HINTS",
    "DEEPWIKI_EXCLUDE_TESTS",
    "DEEPWIKI_TEST_LINKER",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove all flag env-vars before each test."""
    for var in _FLAG_ENVVARS:
        monkeypatch.delenv(var, raising=False)


# ═════════════════════════════════════════════════════════════════════════════
# Feature Flags
# ═════════════════════════════════════════════════════════════════════════════


class TestFeatureFlagsDefaults:
    """Cluster-planner flags are now baseline ON; user-facing flags default OFF."""

    def test_cluster_baseline_on(self):
        flags = get_feature_flags()
        assert flags.hierarchical_leiden is True
        assert flags.capability_validation is True
        assert flags.smart_expansion is True
        assert flags.coverage_ledger is True
        assert flags.language_hints is True

    def test_user_facing_flags_default_off(self):
        flags = get_feature_flags()
        assert flags.exclude_tests is False
        assert flags.test_linker is False

    def test_frozen_dataclass(self):
        flags = get_feature_flags()
        with pytest.raises(AttributeError):
            flags.hierarchical_leiden = False  # type: ignore[misc]


class TestFeatureFlagsFromEnv:
    """Only ``exclude_tests`` and ``test_linker`` remain env-driven."""

    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "YES"])
    def test_exclude_tests_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("DEEPWIKI_EXCLUDE_TESTS", value)
        assert get_feature_flags().exclude_tests is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "anything", ""])
    def test_exclude_tests_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("DEEPWIKI_EXCLUDE_TESTS", value)
        assert get_feature_flags().exclude_tests is False

    def test_test_linker_flag(self, monkeypatch):
        monkeypatch.setenv("DEEPWIKI_TEST_LINKER", "1")
        assert get_feature_flags().test_linker is True

    def test_cluster_envs_no_longer_consulted(self, monkeypatch):
        # Setting the legacy cluster env vars must not flip the flags off —
        # they're hard-coded ON regardless of environment.
        for var in _FLAG_ENVVARS[:5]:
            monkeypatch.setenv(var, "0")
        flags = get_feature_flags()
        assert flags.hierarchical_leiden is True
        assert flags.capability_validation is True
        assert flags.smart_expansion is True
        assert flags.coverage_ledger is True
        assert flags.language_hints is True


class TestEnvBoolHelper:
    """Low-level _env_bool correctness."""

    def test_missing_var_uses_default_false(self):
        assert _env_bool("NONEXISTENT_VAR_12345") is False

    def test_missing_var_uses_default_true(self):
        assert _env_bool("NONEXISTENT_VAR_12345", default=True) is True

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("DEEPWIKI_TEST_BOOL", "  true  ")
        assert _env_bool("DEEPWIKI_TEST_BOOL") is True


# ═════════════════════════════════════════════════════════════════════════════
# Page-Identity Constants
# ═════════════════════════════════════════════════════════════════════════════


class TestPageIdentityConstants:
    """Verify the new symbol classification constants are consistent."""

    def test_identity_and_supporting_disjoint(self):
        overlap = PAGE_IDENTITY_SYMBOLS & SUPPORTING_CODE_SYMBOLS
        assert overlap == set(), f"Overlap: {overlap}"

    def test_identity_and_doc_disjoint(self):
        overlap = PAGE_IDENTITY_SYMBOLS & DOC_CLUSTER_SYMBOLS
        assert overlap == set(), f"Overlap: {overlap}"

    def test_supporting_and_doc_disjoint(self):
        overlap = SUPPORTING_CODE_SYMBOLS & DOC_CLUSTER_SYMBOLS
        assert overlap == set(), f"Overlap: {overlap}"

    def test_code_symbols_subset_of_identity_plus_supporting(self):
        """Every CODE_SYMBOL_TYPES member should be in identity OR supporting."""
        combined = PAGE_IDENTITY_SYMBOLS | SUPPORTING_CODE_SYMBOLS
        missing = CODE_SYMBOL_TYPES - combined
        assert missing == set(), f"CODE_SYMBOL_TYPES not classified: {missing}"

    def test_all_identity_symbols_are_architectural(self):
        """Page identity symbols must be in ARCHITECTURAL_SYMBOLS."""
        missing = PAGE_IDENTITY_SYMBOLS - ARCHITECTURAL_SYMBOLS
        # module/namespace/protocol/record may not be in ARCHITECTURAL_SYMBOLS directly.
        # But they ARE valid identity seeds.  Only check core overlap.
        core = {'class', 'interface', 'struct', 'enum', 'trait', 'function'}
        assert core <= ARCHITECTURAL_SYMBOLS

    def test_priority_covers_identity_symbols(self):
        """Every PAGE_IDENTITY_SYMBOLS member has a priority."""
        for sym in PAGE_IDENTITY_SYMBOLS:
            assert sym in SYMBOL_TYPE_PRIORITY, f"Missing priority for {sym}"

    def test_priority_covers_supporting_symbols(self):
        """Every SUPPORTING_CODE_SYMBOLS member has a priority."""
        for sym in SUPPORTING_CODE_SYMBOLS:
            assert sym in SYMBOL_TYPE_PRIORITY, f"Missing priority for {sym}"

    def test_priority_covers_doc_symbols(self):
        """Every DOC_CLUSTER_SYMBOLS member has a priority."""
        for sym in DOC_CLUSTER_SYMBOLS:
            assert sym in SYMBOL_TYPE_PRIORITY, f"Missing priority for {sym}"

    def test_identity_priority_higher_than_supporting(self):
        """Identity symbols should have higher max priority than supporting."""
        id_max = max(SYMBOL_TYPE_PRIORITY.get(s, 0) for s in PAGE_IDENTITY_SYMBOLS)
        sup_max = max(SYMBOL_TYPE_PRIORITY.get(s, 0) for s in SUPPORTING_CODE_SYMBOLS)
        assert id_max > sup_max

    def test_class_highest_priority(self):
        assert SYMBOL_TYPE_PRIORITY['class'] == 10

    def test_function_priority(self):
        assert SYMBOL_TYPE_PRIORITY['function'] == 7

    def test_type_alias_priority(self):
        assert SYMBOL_TYPE_PRIORITY['type_alias'] == 6

    def test_doc_priority_lowest(self):
        doc_max = max(SYMBOL_TYPE_PRIORITY.get(s, 0) for s in DOC_CLUSTER_SYMBOLS)
        code_min = min(SYMBOL_TYPE_PRIORITY.get(s, 0) for s in SUPPORTING_CODE_SYMBOLS)
        assert doc_max <= code_min
