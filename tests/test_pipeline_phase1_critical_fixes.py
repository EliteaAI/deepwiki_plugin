"""
Pipeline Phase 1 — Critical Fixes
Tests for Gap 1 (.github exclusion), Gap 6 (unified exclusion lists),
and Gap 10 (prompt contradiction).

Covers:
  - constants.py: DOT_DIR_WHITELIST, COVERAGE_IGNORE_DIRS, DEFAULT_EXCLUDE_PATTERNS
  - graph_builder.py: _discover_files_by_language uses DEFAULT_EXCLUDE_PATTERNS
  - structure_tools.py: IGNORED_DIRS import, _should_ignore_dir, register_discovered_dirs
  - structure_prompts.py: No contradictory naming rules
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ============================================================================
# constants.py — DOT_DIR_WHITELIST
# ============================================================================

class TestDotDirWhitelist:
    """Verify DOT_DIR_WHITELIST contains the expected CI/CD directories."""

    def test_github_in_whitelist(self):
        from plugin_implementation.constants import DOT_DIR_WHITELIST
        assert '.github' in DOT_DIR_WHITELIST

    def test_gitlab_in_whitelist(self):
        from plugin_implementation.constants import DOT_DIR_WHITELIST
        assert '.gitlab' in DOT_DIR_WHITELIST

    def test_circleci_in_whitelist(self):
        from plugin_implementation.constants import DOT_DIR_WHITELIST
        assert '.circleci' in DOT_DIR_WHITELIST

    def test_husky_in_whitelist(self):
        from plugin_implementation.constants import DOT_DIR_WHITELIST
        assert '.husky' in DOT_DIR_WHITELIST

    def test_git_not_in_whitelist(self):
        """The .git VCS internal must NOT be whitelisted."""
        from plugin_implementation.constants import DOT_DIR_WHITELIST
        assert '.git' not in DOT_DIR_WHITELIST

    def test_vscode_not_in_whitelist(self):
        from plugin_implementation.constants import DOT_DIR_WHITELIST
        assert '.vscode' not in DOT_DIR_WHITELIST

    def test_whitelist_is_frozenset(self):
        from plugin_implementation.constants import DOT_DIR_WHITELIST
        assert isinstance(DOT_DIR_WHITELIST, frozenset)


# ============================================================================
# constants.py — COVERAGE_IGNORE_DIRS
# ============================================================================

class TestCoverageIgnoreDirs:
    """Verify COVERAGE_IGNORE_DIRS does NOT contain whitelisted dirs."""

    def test_github_not_in_coverage_ignore(self):
        from plugin_implementation.constants import COVERAGE_IGNORE_DIRS
        assert '.github' not in COVERAGE_IGNORE_DIRS

    def test_gitlab_not_in_coverage_ignore(self):
        from plugin_implementation.constants import COVERAGE_IGNORE_DIRS
        assert '.gitlab' not in COVERAGE_IGNORE_DIRS

    def test_git_in_coverage_ignore(self):
        from plugin_implementation.constants import COVERAGE_IGNORE_DIRS
        assert '.git' in COVERAGE_IGNORE_DIRS

    def test_node_modules_in_coverage_ignore(self):
        from plugin_implementation.constants import COVERAGE_IGNORE_DIRS
        assert 'node_modules' in COVERAGE_IGNORE_DIRS

    def test_pycache_in_coverage_ignore(self):
        from plugin_implementation.constants import COVERAGE_IGNORE_DIRS
        assert '__pycache__' in COVERAGE_IGNORE_DIRS

    def test_coverage_ignore_is_frozenset(self):
        from plugin_implementation.constants import COVERAGE_IGNORE_DIRS
        assert isinstance(COVERAGE_IGNORE_DIRS, frozenset)

    def test_no_overlap_with_whitelist(self):
        """Whitelisted dirs must NEVER appear in COVERAGE_IGNORE_DIRS."""
        from plugin_implementation.constants import (
            COVERAGE_IGNORE_DIRS,
            DOT_DIR_WHITELIST,
        )
        overlap = COVERAGE_IGNORE_DIRS & DOT_DIR_WHITELIST
        assert overlap == set(), f"Overlap found: {overlap}"


# ============================================================================
# constants.py — DEFAULT_EXCLUDE_PATTERNS
# ============================================================================

class TestDefaultExcludePatterns:
    """Verify DEFAULT_EXCLUDE_PATTERNS does NOT contain '**/.*'."""

    def test_no_blanket_dot_pattern(self):
        """The '**/.*' pattern must NOT be present — it blocks .github etc."""
        from plugin_implementation.constants import DEFAULT_EXCLUDE_PATTERNS
        assert '**/.*' not in DEFAULT_EXCLUDE_PATTERNS

    def test_git_dir_excluded(self):
        from plugin_implementation.constants import DEFAULT_EXCLUDE_PATTERNS
        assert '**/.git/**' in DEFAULT_EXCLUDE_PATTERNS

    def test_node_modules_excluded(self):
        from plugin_implementation.constants import DEFAULT_EXCLUDE_PATTERNS
        assert '**/node_modules/**' in DEFAULT_EXCLUDE_PATTERNS

    def test_class_files_excluded(self):
        from plugin_implementation.constants import DEFAULT_EXCLUDE_PATTERNS
        assert '**/*.class' in DEFAULT_EXCLUDE_PATTERNS

    def test_vscode_excluded(self):
        from plugin_implementation.constants import DEFAULT_EXCLUDE_PATTERNS
        assert '**/.vscode/**' in DEFAULT_EXCLUDE_PATTERNS

    def test_idea_excluded(self):
        from plugin_implementation.constants import DEFAULT_EXCLUDE_PATTERNS
        assert '**/.idea/**' in DEFAULT_EXCLUDE_PATTERNS

    def test_is_tuple(self):
        from plugin_implementation.constants import DEFAULT_EXCLUDE_PATTERNS
        assert isinstance(DEFAULT_EXCLUDE_PATTERNS, tuple)


# ============================================================================
# graph_builder.py — _discover_files_by_language exclusion fix
# ============================================================================

class TestGraphBuilderExclusion:
    """Verify graph_builder uses centralized patterns and allows .github."""

    @pytest.fixture()
    def repo_dir(self, tmp_path):
        """Create a minimal repo with .github/workflows and regular source."""
        # .github/workflows/ci.yml
        wf = tmp_path / '.github' / 'workflows'
        wf.mkdir(parents=True)
        (wf / 'ci.yml').write_text('name: CI\non: push\n')
        # .github/CODEOWNERS
        (tmp_path / '.github' / 'CODEOWNERS').write_text('* @team\n')
        # .git (should be excluded)
        git_dir = tmp_path / '.git'
        git_dir.mkdir()
        (git_dir / 'HEAD').write_text('ref: refs/heads/main\n')
        # Regular source
        src = tmp_path / 'src'
        src.mkdir()
        (src / 'main.py').write_text('def main(): pass\n')
        # .vscode (should be excluded)
        vscode = tmp_path / '.vscode'
        vscode.mkdir()
        (vscode / 'settings.json').write_text('{}')
        return tmp_path

    def test_github_workflows_discovered(self, repo_dir):
        """Files under .github/ must be discovered (not blocked by **/.*) ."""
        from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
        builder = EnhancedUnifiedGraphBuilder()
        files = builder._discover_files_by_language(str(repo_dir))
        all_paths = []
        for lang_files in files.values():
            all_paths.extend(lang_files)
        github_files = [p for p in all_paths if '.github' in p]
        assert len(github_files) >= 1, (
            f"Expected .github files to be discovered, got: {all_paths}"
        )

    def test_git_dir_still_excluded(self, repo_dir):
        """Files under .git/ must still be excluded."""
        from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
        builder = EnhancedUnifiedGraphBuilder()
        files = builder._discover_files_by_language(str(repo_dir))
        all_paths = []
        for lang_files in files.values():
            all_paths.extend(lang_files)
        git_files = [p for p in all_paths if '/.git/' in p or p.endswith('/.git')]
        assert git_files == [], f"Expected .git files to be excluded, got: {git_files}"

    def test_vscode_dir_excluded(self, repo_dir):
        """Files under .vscode/ must still be excluded."""
        from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
        builder = EnhancedUnifiedGraphBuilder()
        files = builder._discover_files_by_language(str(repo_dir))
        all_paths = []
        for lang_files in files.values():
            all_paths.extend(lang_files)
        vscode_files = [p for p in all_paths if '.vscode' in p]
        assert vscode_files == [], f"Expected .vscode to be excluded, got: {vscode_files}"

    def test_regular_source_discovered(self, repo_dir):
        """Regular source files must still be discovered."""
        from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
        builder = EnhancedUnifiedGraphBuilder()
        files = builder._discover_files_by_language(str(repo_dir))
        assert 'python' in files
        assert any('main.py' in p for p in files['python'])


# ============================================================================
# structure_tools.py — IGNORED_DIRS import & _should_ignore_dir
# ============================================================================

class TestStructureToolsIgnoredDirs:
    """Verify IGNORED_DIRS is imported from constants and excludes correctly."""

    def test_ignored_dirs_equals_coverage_ignore(self):
        """IGNORED_DIRS should be the same object as COVERAGE_IGNORE_DIRS."""
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            IGNORED_DIRS,
        )
        from plugin_implementation.constants import COVERAGE_IGNORE_DIRS
        assert IGNORED_DIRS is COVERAGE_IGNORE_DIRS

    def test_github_not_in_ignored_dirs(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            IGNORED_DIRS,
        )
        assert '.github' not in IGNORED_DIRS

    def test_gitlab_not_in_ignored_dirs(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            IGNORED_DIRS,
        )
        assert '.gitlab' not in IGNORED_DIRS


class TestShouldIgnoreDir:
    """Verify _should_ignore_dir uses DOT_DIR_WHITELIST correctly."""

    def test_github_not_ignored(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            _should_ignore_dir,
        )
        assert _should_ignore_dir('.github') is False

    def test_gitlab_not_ignored(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            _should_ignore_dir,
        )
        assert _should_ignore_dir('.gitlab') is False

    def test_circleci_not_ignored(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            _should_ignore_dir,
        )
        assert _should_ignore_dir('.circleci') is False

    def test_husky_not_ignored(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            _should_ignore_dir,
        )
        assert _should_ignore_dir('.husky') is False

    def test_git_ignored(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            _should_ignore_dir,
        )
        assert _should_ignore_dir('.git') is True

    def test_vscode_ignored(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            _should_ignore_dir,
        )
        assert _should_ignore_dir('.vscode') is True

    def test_node_modules_ignored(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            _should_ignore_dir,
        )
        assert _should_ignore_dir('node_modules') is True

    def test_pycache_ignored(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            _should_ignore_dir,
        )
        assert _should_ignore_dir('__pycache__') is True

    def test_random_dot_dir_ignored(self):
        """Arbitrary dot-dirs not in whitelist should be ignored."""
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            _should_ignore_dir,
        )
        assert _should_ignore_dir('.random_hidden') is True

    def test_underscore_dir_ignored(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            _should_ignore_dir,
        )
        assert _should_ignore_dir('_internal') is True

    def test_init_not_ignored(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            _should_ignore_dir,
        )
        assert _should_ignore_dir('__init__.py') is False

    def test_egg_info_ignored(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            _should_ignore_dir,
        )
        assert _should_ignore_dir('mypackage.egg-info') is True

    def test_regular_dir_not_ignored(self):
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            _should_ignore_dir,
        )
        assert _should_ignore_dir('src') is False


# ============================================================================
# structure_tools.py — register_discovered_dirs
# ============================================================================

class TestRegisterDiscoveredDirs:
    """Verify register_discovered_dirs accepts whitelisted dot-dirs."""

    def _make_collector(self):
        """Create a minimal StructureCollector for testing."""
        from plugin_implementation.wiki_structure_planner.structure_tools import (
            StructureCollector,
        )
        collector = StructureCollector.__new__(StructureCollector)
        collector.discovered_dirs = set()
        collector.covered_dirs = set()
        collector._covered_symbols = set()
        collector._all_architectural_symbols = set()
        collector._symbol_to_page = {}
        collector.sections = {}
        collector.pages = []
        collector.errors = []
        collector.metadata = None
        collector._tree_analysis = None
        collector._thinking_log = []
        collector._graph_module_cache = {}
        collector.page_budget = 10
        collector.page_budget_overflow = 2
        collector.soft_page_budget = 12
        return collector

    def test_github_dir_registered(self):
        collector = self._make_collector()
        collector.register_discovered_dirs(['.github/'])
        assert '.github' in collector.discovered_dirs

    def test_gitlab_dir_registered(self):
        collector = self._make_collector()
        collector.register_discovered_dirs(['.gitlab/'])
        assert '.gitlab' in collector.discovered_dirs

    def test_circleci_dir_registered(self):
        collector = self._make_collector()
        collector.register_discovered_dirs(['.circleci/'])
        assert '.circleci' in collector.discovered_dirs

    def test_random_dot_dir_not_registered(self):
        """Non-whitelisted dot-dirs should still be filtered out."""
        collector = self._make_collector()
        collector.register_discovered_dirs(['.hidden/'])
        assert '.hidden' not in collector.discovered_dirs

    def test_git_dir_not_registered(self):
        collector = self._make_collector()
        collector.register_discovered_dirs(['.git/'])
        assert '.git' not in collector.discovered_dirs

    def test_regular_dir_registered(self):
        collector = self._make_collector()
        collector.register_discovered_dirs(['src/'])
        assert 'src' in collector.discovered_dirs

    def test_node_modules_not_registered(self):
        collector = self._make_collector()
        collector.register_discovered_dirs(['node_modules/'])
        assert 'node_modules' not in collector.discovered_dirs


# ============================================================================
# structure_prompts.py — No contradictory naming rules
# ============================================================================

class TestPromptNoContradiction:
    """Verify the structure planner prompt has consistent naming rules."""

    def test_no_must_include_class_names_rule(self):
        """The old contradictory rule 'Page names MUST include actual
        class/function names' must NOT be present."""
        from plugin_implementation.wiki_structure_planner.structure_prompts import (
            get_structure_task_prompt,
        )
        prompt = get_structure_task_prompt(
            repo_name="test-repo",
            page_budget=20,
        )
        assert 'Page names MUST include actual class/function names' not in prompt

    def test_capability_based_rule_present(self):
        """The correct rule about capability-based naming must be present
        in the system instructions (not the task prompt)."""
        from plugin_implementation.wiki_structure_planner.structure_prompts import (
            get_structure_planner_instructions,
        )
        prompt = get_structure_planner_instructions()
        assert 'capability-based' in prompt.lower()

    def test_page_granularity_guide_present(self):
        """The page granularity table must be in the system instructions."""
        from plugin_implementation.wiki_structure_planner.structure_prompts import (
            get_structure_planner_instructions,
        )
        prompt = get_structure_planner_instructions()
        # SPEC-3 compacted the heading to '## Page Granularity'
        assert 'Page Granularity' in prompt

    def test_rule1_in_system_prompt_consistent(self):
        """Rule 1 (Capability-based page names) forbids symbol names
        in page titles.  SPEC-3 compacted the phrasing."""
        from plugin_implementation.wiki_structure_planner.structure_prompts import (
            get_structure_planner_instructions,
        )
        prompt = get_structure_planner_instructions()
        # SPEC-3 compacted: 'NO symbol/class names in titles'
        assert 'NO symbol/class names in titles' in prompt
