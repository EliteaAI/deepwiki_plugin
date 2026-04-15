"""
Tests for symbol prioritisation in the Graph-First structure refiner.

Verifies:
- _score_symbol returns correct layer-based + connection-based scores
- prioritize_symbols returns all symbols when below cap
- prioritize_symbols ranks and truncates when above cap
- Deduplication of same-name symbols from different files
- Environment variable override for max_symbols cap
"""

import os

import pytest

from plugin_implementation.wiki_structure_planner.structure_skeleton import SymbolInfo
from plugin_implementation.wiki_structure_planner.structure_refiner import (
    _score_symbol,
    prioritize_symbols,
    _LAYER_WEIGHTS,
    _DEFAULT_MAX_TARGET_SYMBOLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sym(name, layer, connections, sym_type='class', rel_path='src/core.py'):
    """Shorthand factory for SymbolInfo."""
    return SymbolInfo(
        name=name,
        type=sym_type,
        rel_path=rel_path,
        layer=layer,
        connections=connections,
        docstring='',
    )


# ---------------------------------------------------------------------------
# _score_symbol
# ---------------------------------------------------------------------------

class TestScoreSymbol:
    """Validate the scoring formula: layer_weight * 10 + min(connections, 100)."""

    def test_entry_point_highest(self):
        ep = _sym('main', 'entry_point', 5)
        pub = _sym('Foo', 'public_api', 5)
        assert _score_symbol(ep) > _score_symbol(pub)

    def test_layer_dominates_connections(self):
        """Even a low-connection entry_point beats a high-connection utility."""
        ep = _sym('main', 'entry_point', 0)
        const = _sym('MAX', 'constant', 999)
        assert _score_symbol(ep) > _score_symbol(const)

    def test_connections_cap_at_100(self):
        sym100 = _sym('A', 'public_api', 100)
        sym200 = _sym('B', 'public_api', 200)
        assert _score_symbol(sym100) == _score_symbol(sym200)

    def test_same_layer_higher_connections_wins(self):
        a = _sym('A', 'core_type', 10)
        b = _sym('B', 'core_type', 50)
        assert _score_symbol(b) > _score_symbol(a)

    def test_unknown_layer_defaults_to_2(self):
        sym = _sym('X', 'unknown_layer', 0)
        expected = 2 * 200 + 0
        assert _score_symbol(sym) == expected

    @pytest.mark.parametrize('layer,weight', _LAYER_WEIGHTS.items())
    def test_layer_weights_exact(self, layer, weight):
        sym = _sym('Sym', layer, 0)
        assert _score_symbol(sym) == weight * 200


# ---------------------------------------------------------------------------
# prioritize_symbols — below cap
# ---------------------------------------------------------------------------

class TestPrioritizeBelowCap:
    """When #symbols ≤ cap, all symbols are returned (sorted by score)."""

    def test_all_returned(self):
        syms = [_sym('A', 'public_api', 10), _sym('B', 'entry_point', 2)]
        result = prioritize_symbols(syms, max_symbols=10)
        assert set(result) == {'A', 'B'}

    def test_sorted_by_score(self):
        syms = [
            _sym('Low', 'constant', 1),
            _sym('High', 'entry_point', 50),
            _sym('Mid', 'public_api', 20),
        ]
        result = prioritize_symbols(syms, max_symbols=10)
        assert result[0] == 'High'
        assert result[-1] == 'Low'

    def test_empty_list(self):
        assert prioritize_symbols([], max_symbols=10) == []


# ---------------------------------------------------------------------------
# prioritize_symbols — above cap
# ---------------------------------------------------------------------------

class TestPrioritizeAboveCap:
    """When #symbols > cap, only top-N are returned."""

    def test_truncation(self):
        syms = [_sym(f'Sym{i}', 'public_api', i) for i in range(100)]
        result = prioritize_symbols(syms, max_symbols=5)
        assert len(result) == 5

    def test_top_symbols_selected(self):
        syms = [
            _sym('EntryMain', 'entry_point', 80),
            _sym('ApiService', 'public_api', 60),
            _sym('CoreEnum', 'core_type', 40),
            _sym('InfraUtil', 'infrastructure', 20),
            _sym('InternalHelper', 'internal', 10),
            _sym('CONST_X', 'constant', 5),
        ]
        result = prioritize_symbols(syms, max_symbols=3)
        assert result == ['EntryMain', 'ApiService', 'CoreEnum']

    def test_deduplication(self):
        """Same symbol name in two files → only one entry."""
        syms = [
            _sym('Logger', 'infrastructure', 30, rel_path='src/a.py'),
            _sym('Logger', 'infrastructure', 25, rel_path='src/b.py'),
            _sym('Entry', 'entry_point', 10),
        ]
        result = prioritize_symbols(syms, max_symbols=2)
        # Entry outranks Logger, then Logger appears once
        assert result == ['Entry', 'Logger']

    def test_cap_equals_one(self):
        syms = [
            _sym('A', 'public_api', 10),
            _sym('B', 'entry_point', 5),
        ]
        result = prioritize_symbols(syms, max_symbols=1)
        assert result == ['B']  # entry_point beats public_api


# ---------------------------------------------------------------------------
# Environment variable override
# ---------------------------------------------------------------------------

class TestEnvVarOverride:
    """DEEPWIKI_MAX_TARGET_SYMBOLS env var controls the cap."""

    def test_env_var_lowers_cap(self, monkeypatch):
        monkeypatch.setenv('DEEPWIKI_MAX_TARGET_SYMBOLS', '2')
        syms = [_sym(f'S{i}', 'public_api', i) for i in range(10)]
        result = prioritize_symbols(syms)  # no explicit max_symbols
        assert len(result) == 2

    def test_env_var_raises_cap(self, monkeypatch):
        monkeypatch.setenv('DEEPWIKI_MAX_TARGET_SYMBOLS', '999')
        syms = [_sym(f'S{i}', 'public_api', i) for i in range(50)]
        result = prioritize_symbols(syms)
        assert len(result) == 50  # all returned since 50 < 999

    def test_default_cap(self):
        assert _DEFAULT_MAX_TARGET_SYMBOLS == 30
