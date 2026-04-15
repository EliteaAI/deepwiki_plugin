"""
Pipeline Phase 2 — FTS5 Enablement
Tests for:
  - FTS5 default ON behaviour (no env var → enabled)
  - Opt-out via DEEPWIKI_ENABLE_FTS5=0
  - clear_cache None-guard safety
  - GraphManager.fts_index always returns an instance when enabled
"""

import os
import tempfile
import shutil

import pytest
import networkx as nx


# ============================================================================
# _is_fts5_enabled() — default ON behaviour
# ============================================================================

class TestFTS5DefaultOn:
    """Verify FTS5 is ON by default and can be opted out with =0."""

    def _call(self):
        from plugin_implementation.graph_manager import _is_fts5_enabled
        return _is_fts5_enabled()

    def test_default_no_env_var(self, monkeypatch):
        """When DEEPWIKI_ENABLE_FTS5 is not set at all, FTS5 should be ON."""
        monkeypatch.delenv('DEEPWIKI_ENABLE_FTS5', raising=False)
        assert self._call() is True

    def test_explicit_1_enabled(self, monkeypatch):
        """DEEPWIKI_ENABLE_FTS5=1 → ON."""
        monkeypatch.setenv('DEEPWIKI_ENABLE_FTS5', '1')
        assert self._call() is True

    def test_explicit_0_disabled(self, monkeypatch):
        """DEEPWIKI_ENABLE_FTS5=0 → OFF (opt-out)."""
        monkeypatch.setenv('DEEPWIKI_ENABLE_FTS5', '0')
        assert self._call() is False

    def test_random_value_enabled(self, monkeypatch):
        """Any value other than '0' should be treated as ON."""
        monkeypatch.setenv('DEEPWIKI_ENABLE_FTS5', 'yes')
        assert self._call() is True

    def test_empty_string_enabled(self, monkeypatch):
        """Empty string is not '0', so FTS5 should be ON."""
        monkeypatch.setenv('DEEPWIKI_ENABLE_FTS5', '')
        assert self._call() is True


# ============================================================================
# GraphManager.fts_index — always returns instance when enabled
# ============================================================================

class TestFtsIndexProperty:
    """Verify fts_index returns a real index by default."""

    @pytest.fixture()
    def manager(self, tmp_path, monkeypatch):
        monkeypatch.delenv('DEEPWIKI_ENABLE_FTS5', raising=False)
        from plugin_implementation.graph_manager import GraphManager
        mgr = GraphManager(cache_dir=str(tmp_path))
        yield mgr
        if mgr._fts_index:
            mgr._fts_index.close()

    @pytest.fixture()
    def disabled_manager(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DEEPWIKI_ENABLE_FTS5', '0')
        from plugin_implementation.graph_manager import GraphManager
        mgr = GraphManager(cache_dir=str(tmp_path))
        yield mgr
        if mgr._fts_index:
            mgr._fts_index.close()

    def test_fts_index_returns_instance_default(self, manager):
        """By default, fts_index should return a GraphTextIndex."""
        from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
        idx = manager.fts_index
        assert idx is not None
        assert isinstance(idx, GraphTextIndex)

    def test_fts_index_returns_none_when_disabled(self, disabled_manager):
        """When FTS5 is disabled, fts_index should return None."""
        idx = disabled_manager.fts_index
        assert idx is None

    def test_fts_index_is_lazy_singleton(self, manager):
        """Multiple accesses should return the same instance."""
        idx1 = manager.fts_index
        idx2 = manager.fts_index
        assert idx1 is idx2


# ============================================================================
# build_fts_index — builds automatically when enabled
# ============================================================================

class TestBuildFtsIndex:
    """Verify build_fts_index builds by default and skips when disabled."""

    @staticmethod
    def _small_graph():
        """Graph with flat node attributes as produced by graph_builder."""
        g = nx.DiGraph()
        g.add_node('Foo', **{
            'symbol_name': 'Foo',
            'symbol_type': 'class',
            'file_path': 'src/foo.py',
            'rel_path': 'src/foo.py',
            'language': 'python',
            'start_line': 1,
            'end_line': 10,
            'content': 'class Foo:\n    pass\n',
        })
        g.add_node('bar', **{
            'symbol_name': 'bar',
            'symbol_type': 'function',
            'file_path': 'src/bar.py',
            'rel_path': 'src/bar.py',
            'language': 'python',
            'start_line': 1,
            'end_line': 5,
            'content': 'def bar():\n    return 42\n',
        })
        return g

    @pytest.fixture()
    def manager(self, tmp_path, monkeypatch):
        monkeypatch.delenv('DEEPWIKI_ENABLE_FTS5', raising=False)
        from plugin_implementation.graph_manager import GraphManager
        mgr = GraphManager(cache_dir=str(tmp_path))
        yield mgr
        if mgr._fts_index:
            mgr._fts_index.close()

    def test_build_returns_index(self, manager):
        g = self._small_graph()
        idx = manager.build_fts_index(g, '/tmp/test', commit_hash='abc')
        assert idx is not None
        assert idx.is_open
        assert idx.node_count == 2

    def test_build_skips_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DEEPWIKI_ENABLE_FTS5', '0')
        from plugin_implementation.graph_manager import GraphManager
        mgr = GraphManager(cache_dir=str(tmp_path))
        g = self._small_graph()
        idx = mgr.build_fts_index(g, '/tmp/test', commit_hash='abc')
        assert idx is None


# ============================================================================
# clear_cache — None guard fix
# ============================================================================

class TestClearCacheNoneGuard:
    """Verify clear_cache does NOT crash when FTS5 is disabled."""

    def test_clear_specific_repo_fts_disabled(self, tmp_path, monkeypatch):
        """clear_cache(repo_path=...) must not crash when FTS5 is OFF."""
        monkeypatch.setenv('DEEPWIKI_ENABLE_FTS5', '0')
        from plugin_implementation.graph_manager import GraphManager
        mgr = GraphManager(cache_dir=str(tmp_path))
        # Should NOT raise AttributeError
        mgr.clear_cache(repo_path='/tmp/nonexistent')

    def test_clear_all_fts_disabled(self, tmp_path, monkeypatch):
        """clear_cache() (all) must not crash when FTS5 is OFF."""
        monkeypatch.setenv('DEEPWIKI_ENABLE_FTS5', '0')
        from plugin_implementation.graph_manager import GraphManager
        mgr = GraphManager(cache_dir=str(tmp_path))
        # Should NOT raise
        mgr.clear_cache()

    def test_clear_specific_repo_fts_enabled(self, tmp_path, monkeypatch):
        """clear_cache(repo_path=...) should work when FTS5 is ON."""
        monkeypatch.delenv('DEEPWIKI_ENABLE_FTS5', raising=False)
        from plugin_implementation.graph_manager import GraphManager
        mgr = GraphManager(cache_dir=str(tmp_path))
        # Should NOT raise
        mgr.clear_cache(repo_path='/tmp/nonexistent')
        if mgr._fts_index:
            mgr._fts_index.close()


# ============================================================================
# load_fts_index — returns None when disabled, index when enabled
# ============================================================================

class TestLoadFtsIndex:
    """Verify load methods respect the flag."""

    def test_load_returns_none_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DEEPWIKI_ENABLE_FTS5', '0')
        from plugin_implementation.graph_manager import GraphManager
        mgr = GraphManager(cache_dir=str(tmp_path))
        result = mgr.load_fts_index('/tmp/test')
        assert result is None

    def test_load_by_name_returns_none_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DEEPWIKI_ENABLE_FTS5', '0')
        from plugin_implementation.graph_manager import GraphManager
        mgr = GraphManager(cache_dir=str(tmp_path))
        result = mgr.load_fts_index_by_repo_name('test-repo')
        assert result is None
