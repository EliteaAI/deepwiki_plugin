"""
Shared constants for DeepWiki plugin implementation.

Centralizes symbol type definitions used across graph_builder, retrievers,
wiki_graph_optimized agent, and structure_tools to avoid duplication drift.

ALL symbol-type sets are defined here as the single source of truth.
"""

# =============================================================================
# Embedding Configuration
# =============================================================================

# Default embedding dimension for the platform-provided models.
# The EliteA platform uses OpenAI text-embedding-ada-002 (or equivalent)
# which produces 1536-dimensional vectors.  This is the dimension used
# for the ``repo_vec`` table in ``unified_db.py``.
DEFAULT_EMBEDDING_DIM = 1536

# Batch size for embedding generation during index build.
# Balanced between throughput and rate-limit safety.
EMBEDDING_BATCH_SIZE = 64

# =============================================================================
# Documentation File Extensions (single source of truth)
# =============================================================================
# Extension → doc_type mapping.  Used by graph_builder to classify files and by
# the retriever / wiki agent to recognise documentation sources.
# The doc_type value becomes part of the symbol_type: "{doc_type}_document".
DOCUMENTATION_EXTENSIONS = {
    # Prose / markup
    '.md': 'markdown',
    '.rst': 'restructuredtext',
    '.txt': 'plaintext',
    '.adoc': 'asciidoc',
    '.doc': 'document',
    '.docx': 'document',
    '.pdf': 'pdf',
    # Web / data interchange
    '.html': 'html',
    '.htm': 'html',
    '.xml': 'xml',
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.toml': 'toml',
    # Config files
    '.ini': 'config',
    '.cfg': 'config',
    '.conf': 'config',
    '.properties': 'config',
    '.env': 'config',
    # Build scripts / build configs
    '.gradle': 'build_config',
    '.kts': 'build_config',
    # Schema / IDL files
    '.wsdl': 'schema',
    '.xsd': 'schema',
    '.proto': 'schema',
    # Infrastructure as Code
    '.tf': 'infrastructure',
    '.tfvars': 'infrastructure',
    '.hcl': 'infrastructure',
    # Module definition files
    '.mod': 'config',
    # Shell / batch scripts (indexed as text, no tree-sitter)
    '.sh': 'script',
    '.bash': 'script',
    '.bat': 'script',
    '.cmd': 'script',
    '.ps1': 'script',
    '.psm1': 'script',
}

# Flat frozenset of documentation extensions for quick membership checks
# (used by retriever and wiki agent instead of hard-coded inline sets).
DOCUMENTATION_EXTENSIONS_SET = frozenset(DOCUMENTATION_EXTENSIONS.keys())

# =============================================================================
# Known Filenames (extensionless / special-name files)
# =============================================================================
# Files that cannot be classified by extension alone.  Checked by
# _discover_files_by_language() and _parse_documentation_files() when
# file_path.suffix is empty or not in DOCUMENTATION_EXTENSIONS.
KNOWN_FILENAMES = {
    # Build systems
    'Makefile': 'build_config',
    'makefile': 'build_config',
    'GNUmakefile': 'build_config',
    'CMakeLists.txt': 'build_config',
    'Justfile': 'build_config',
    'justfile': 'build_config',
    'Taskfile': 'build_config',
    'Earthfile': 'build_config',
    'Snakefile': 'build_config',
    'SConstruct': 'build_config',
    'SConscript': 'build_config',
    'Rakefile': 'build_config',
    'BUILD': 'build_config',
    'BUILD.bazel': 'build_config',
    'WORKSPACE': 'build_config',
    'WORKSPACE.bazel': 'build_config',
    'Tiltfile': 'build_config',
    # Container / DevOps
    'Dockerfile': 'infrastructure',
    'Containerfile': 'infrastructure',
    'Vagrantfile': 'infrastructure',
    'Procfile': 'infrastructure',
    # Package managers
    'Gemfile': 'config',
    'Brewfile': 'config',
    'Pipfile': 'config',
    # CI/CD
    'Jenkinsfile': 'script',
    # Documentation / repo metadata
    'Doxyfile': 'config',
    'LICENSE': 'plaintext',
    'CHANGELOG': 'plaintext',
    'CONTRIBUTING': 'plaintext',
    'AUTHORS': 'plaintext',
    'CODEOWNERS': 'config',
    # Editor / tool config (dot-prefixed but commonly not hidden)
    '.editorconfig': 'config',
    '.gitattributes': 'config',
    '.gitignore': 'config',
    '.dockerignore': 'config',
    '.npmignore': 'config',
    '.eslintignore': 'config',
    '.prettierignore': 'config',
    '.env': 'config',
    '.env.example': 'config',
    '.env.local': 'config',
    '.env.development': 'config',
    '.env.production': 'config',
}

# =============================================================================
# Documentation Symbol Types
# =============================================================================
# Symbol types emitted by parsers for documentation files.
# Must include "{doc_type}_document" for EVERY doc_type value that appears
# in DOCUMENTATION_EXTENSIONS or KNOWN_FILENAMES, plus section / chunk types.
DOC_SYMBOL_TYPES = frozenset({
    # Prose / markup
    'markdown_document',
    'markdown_section',
    'restructuredtext_document',
    'asciidoc_document',
    'plaintext_document',
    'document_document',      # .doc / .docx
    'pdf_document',
    # Web / data interchange
    'html_document',
    'xml_document',
    'json_document',
    'yaml_document',
    'toml_document',
    # Config
    'config_document',        # .ini, .cfg, .conf, .properties, .env, .mod
    # Build
    'build_config_document',  # .gradle, .kts, Makefile, CMakeLists.txt, etc.
    # Schema / IDL
    'schema_document',        # .wsdl, .xsd, .proto
    # Infrastructure as Code
    'infrastructure_document',# .tf, .tfvars, .hcl, Dockerfile, Vagrantfile
    # Scripts
    'script_document',        # .sh, .bash, .bat, .cmd, .ps1, .psm1, Jenkinsfile
    # Generic chunk types (used by _chunk_generic_text fallback)
    'text_chunk',
    'text_document',
})

# Chunk types used by the vector store / retriever layer for doc filtering.
# These are metadata['chunk_type'] values on Document objects in the store.
DOC_CHUNK_TYPES = frozenset({
    'text',
    'documentation',
    'markdown',
})

# =============================================================================
# Architectural Symbol Types
# =============================================================================

# The FULL set of architectural symbols kept in the graph / vector store.
# Used by graph_builder._is_architectural_symbol() to decide which parsed
# symbols become graph nodes and which chunks enter the vector store.
# Includes macro and ALL doc types so documentation chunks pass the gate.
ARCHITECTURAL_SYMBOLS = frozenset({
    # Top-level code symbols
    'class', 'interface', 'struct', 'enum', 'trait',
    'function',      # Standalone functions only — methods excluded
    'constant',      # Module-level constants
    'type_alias',    # Type aliases (using/typedef)
    'macro',         # C/C++ preprocessor definitions
    'contract',      # API surface contracts (rest_route, grpc_service, …)
    # SQL / DDL entity symbols (sql_extractor)
    'sql_schema', 'sql_table', 'sql_view', 'sql_function',
    'sql_column', 'sql_index', 'sql_trigger',
}) | DOC_SYMBOL_TYPES  # ← every doc type passes _is_architectural_symbol()

# Subset used when expanding graph context for wiki page generation.
# Excludes 'method' (part of parent class — expanded via class).
# Includes module_doc/file_doc for documentation context.
EXPANSION_SYMBOL_TYPES = frozenset({
    'class', 'interface', 'struct', 'enum', 'trait',
    'function',
    'constant',
    'type_alias',
    'macro',         # C/C++ macros are architecturally significant
    'contract',      # API surface contracts (rest_route, grpc_service, …)
    'module_doc', 'file_doc',
    # SQL / DDL entity symbols (sql_extractor)
    'sql_schema', 'sql_table', 'sql_view', 'sql_function',
})

# =============================================================================
# Page Identity Classification (capability-first clustering)
# =============================================================================
# These sets partition code symbols by their page-identity role.
# Used by cluster_planner seed selection and candidate_builder classification.

# Symbols that can anchor a page (drive page identity in mixed clusters).
# Ordered by SYMBOL_TYPE_PRIORITY when selecting seeds.
PAGE_IDENTITY_SYMBOLS = frozenset({
    'class', 'interface', 'trait', 'protocol',
    'enum', 'struct', 'record',
    'module', 'namespace',
    'function',
    # SQL / DDL entities that can anchor a page
    'sql_table', 'sql_view', 'sql_function',
})

# Symbols that support page content but do not drive page identity
# in mixed clusters; eligible as seeds only when no identity symbol exists.
SUPPORTING_CODE_SYMBOLS = frozenset({
    'constant', 'type_alias', 'macro',
    'method', 'property',
    # SQL / DDL supporting members
    'sql_column', 'sql_index', 'sql_trigger', 'sql_schema',
})

# Documentation symbol kinds used for docs-only page detection.
#
# Must stay in sync with the symbol_type values that
# ``code_graph/graph_builder.py:_parse_documentation_files`` actually
# emits. The previous {module_doc, file_doc} set was a dead synonym —
# no parser ever produced those types, so candidate_builder always
# saw doc_count == 0 and never classified anything as CLASS_DOCS.
# Empirical types observed in the live unified DB on configurations +
# microservices: markdown_document, plaintext_document, yaml_document,
# json_document, config_document, infrastructure_document,
# script_document, html_document, schema_document, text_chunk.
#
# When adding a new doc parser, audit ``_chunk_text_content`` and
# ``DOCUMENTATION_EXTENSIONS`` and add the new symbol_type here.
DOC_CLUSTER_SYMBOLS = frozenset({
    # Per-file blob doc nodes (one per source file unless chunked)
    'markdown_document',
    'plaintext_document',
    'yaml_document',
    'json_document',
    'config_document',
    'infrastructure_document',
    'script_document',
    'html_document',
    'schema_document',
    # Markdown header-split sections — the dominant type after Phase A's
    # always-split-by-header chunker (graph_builder._chunk_markdown_content
    # emits section_type='markdown_section', which becomes symbol_type on
    # the BasicSymbol node). Without this entry candidate_builder counts
    # README sections as code and never classifies pure-doc clusters.
    'markdown_section',
    # Generic chunks produced by _chunk_generic_text fallback
    'text_chunk',
    # Legacy aliases — kept for compatibility if any parser still emits them
    'module_doc', 'file_doc',
})

# Numeric priority for seed ordering — aligned with graph_builder._get_type_priority().
# All DOC_CLUSTER_SYMBOLS share priority 1 (lowest, tie-broken by name) so they
# never out-rank code seeds in mixed clusters but remain valid fallbacks for
# pure-doc clusters. Test contract:
# tests/test_feature_flags.py::test_priority_covers_doc_symbols.
SYMBOL_TYPE_PRIORITY = {
    'class': 10, 'interface': 10, 'trait': 10, 'protocol': 10,
    'enum': 9, 'struct': 9, 'record': 9,
    'module': 8, 'namespace': 8,
    'function': 7,
    'constant': 6, 'type_alias': 6, 'macro': 6,
    'contract': 5,
    'method': 3, 'property': 2,
    # SQL / DDL entities — tables/views anchor like structs; functions like funcs.
    'sql_table': 9, 'sql_view': 9, 'sql_schema': 8, 'sql_function': 7,
    'sql_index': 3, 'sql_trigger': 3, 'sql_column': 2,
    # Doc symbols — must cover every DOC_CLUSTER_SYMBOLS member.
    'markdown_document': 1,
    'markdown_section': 1,
    'plaintext_document': 1,
    'yaml_document': 1,
    'json_document': 1,
    'config_document': 1,
    'infrastructure_document': 1,
    'script_document': 1,
    'html_document': 1,
    'schema_document': 1,
    'text_chunk': 1,
    'module_doc': 1, 'file_doc': 1,
}

# =============================================================================
# Code-Only Symbol Types
# =============================================================================

# Code-only architectural types used by query_graph and graph expansion
# when we need only code symbols (no docs).
# This is the set shown in the structure planner's query_graph output.
CODE_SYMBOL_TYPES = frozenset({
    'class', 'interface', 'struct', 'enum', 'trait',
    'function',
    'constant',
    'type_alias',
    'macro',         # C/C++ macros are architecturally significant
    # SQL / DDL entities
    'sql_schema', 'sql_table', 'sql_view', 'sql_function',
    'sql_column', 'sql_index', 'sql_trigger',
})

# =============================================================================
# Edge-Class Vocabularies (Phase B7 — writer evidence gating)
# =============================================================================
# All ``edge_class`` values that appear on ``repo_edges.edge_class``. The
# default is ``"structural"`` (AST-derived). Synthetic classes are added
# by topology enrichment (orphan resolution, doc proximity, bridge edges)
# and exist primarily so Leiden can group otherwise-disconnected nodes
# into reasonable communities — they do not represent real architectural
# coupling and must NOT be cited by the writer.
#
# Producer audit (deepwiki_plugin, ``grep edge_class= --include='*.py'``):
#
# - ``structural``     — default, set by unified_db.py:221 and assumed
#                        whenever a parser/linker omits ``edge_class``.
# - ``cross_language`` — cross_language_linker.py:92, 141.
# - ``test_link``      — test_linker.py:108, 143.
# - ``directory``      — graph_topology.py:458 (directory-proximity glue).
# - ``lexical``        — graph_topology.py:518, 631, 839, 896 (FTS / name match).
# - ``semantic``       — graph_topology.py:576, 1000 (embedding cosine).
# - ``doc``            — graph_topology.py:518, 1409, 1422 (markdown link / proximity).
# - ``bridge``         — graph_topology.py:1596+; graph_clustering.py:1560+.
#
# Names referenced in the GAP_ANALYSIS_AND_ROADMAP but NOT yet emitted
# in this repo: ``cross_repo`` (planned for Phase D's contract-node
# algebra), ``member_uses`` (mentioned in cross_language_linker L3
# docstring but never set as edge_class). Add them to the allow-set
# below when their producers land — until then they would be silent
# placeholders.

#: Synthetic edges added by topology enrichment. Bypassed when computing
#: structural in-degree (``apply_edge_weights``) and when assembling
#: writer evidence (``_collect_expansion_neighbors``).
SYNTHETIC_EDGE_CLASSES = frozenset({
    "directory",
    "lexical",
    "semantic",
    "doc",
    "bridge",
})

#: Edge classes the writer is allowed to walk when expanding a page's
#: content. Default-allow set per ``GAP_ANALYSIS_AND_ROADMAP.md`` §B7:
#: only architecturally-real edges contribute to the prompt context, so
#: the LLM cannot ground a citation on a lexical/directory/bridge edge.
#:
#: Sweep this set explicitly when adding a new producer ``edge_class``.
#: ``_collect_expansion_neighbors`` blocks unknown classes by default,
#: so a typo in the producer makes its edges invisible to the writer
#: (loud silence, not silent inclusion).
WRITER_ALLOWED_EDGE_CLASSES = frozenset({
    "structural",      # AST-derived; the canonical default
    "cross_language",  # cross-language API surface match
    "test_link",       # test_node ↔ production_node
})

# =============================================================================
# Display Categories for query_graph
# =============================================================================
# Mapping from symbol types to display buckets in the structure planner.
# Used for both log lines and LLM-visible output.
CLASS_LIKE_TYPES = frozenset({'class', 'interface', 'protocol', 'struct', 'trait'})
FUNCTION_LIKE_TYPES = frozenset({'function', 'method', 'def'})
ENUM_LIKE_TYPES = frozenset({'enum', 'constant'})
ALIAS_LIKE_TYPES = frozenset({'type_alias'})

# =============================================================================
# Deterministic Symbol Layer Classification (SPEC-4)
# =============================================================================
# 6 layers assigned at index-build time — zero LLM cost.
# Used for prioritized search, structure planning, and display grouping.

SYMBOL_LAYERS = frozenset({
    'entry_point',     # main, run, cli, app — top-level entry functions
    'public_api',      # exported classes/interfaces with public surface
    'core_type',       # structs, enums, type aliases — data definitions
    'internal',        # nested/private classes and functions
    'infrastructure',  # config, utility, helper, middleware, logging
    'constant',        # module-level constants and macros
})

# Name patterns that signal an entry-point function.
_ENTRY_POINT_NAMES = frozenset({
    'main', 'run', 'execute', 'start', 'cli', 'app',
    'serve', 'setup', 'bootstrap', 'entrypoint', 'entry_point',
})

# Path segments that signal infrastructure code.
_INFRA_PATH_SEGMENTS = frozenset({
    'config', 'configs', 'configuration',
    'util', 'utils', 'utility', 'utilities',
    'helper', 'helpers',
    'middleware',
    'logging', 'logger',
    'common', 'shared', 'base',
})

# Path segments that signal internal/private code.
_INTERNAL_PATH_SEGMENTS = frozenset({
    'internal', '_internal', 'private', '_private',
    'impl', '_impl', 'detail', '_detail',
})

# Name suffixes that signal a public API class.
_PUBLIC_API_SUFFIXES = (
    'Service', 'Manager', 'Handler', 'Controller', 'Provider',
    'Client', 'Factory', 'Builder', 'Processor', 'Executor',
    'Router', 'Resolver', 'Adapter', 'Gateway', 'Repository',
    'API', 'Endpoint', 'Resource', 'Middleware',
)


def classify_symbol_layer(
    symbol_type: str,
    symbol_name: str,
    parent_symbol: str = '',
    file_path: str = '',
    docstring: str = '',
) -> str:
    """Classify a symbol into one of 6 deterministic layers.

    Priority order (highest → lowest):
      1. entry_point — function with entry-point name pattern
      2. constant — constants and macros (always assigned)
      3. core_type — structs, enums, type aliases (always assigned)
      4. infrastructure — path matches infra patterns
      5. internal — nested symbol or private path
      6. public_api — default for classes/interfaces/traits/functions

    Args:
        symbol_type: Lowercased symbol type (class, function, constant, etc.)
        symbol_name: Symbol name as-is.
        parent_symbol: Parent symbol name (non-empty → nested).
        file_path: Relative file path for path-based classification.
        docstring: Symbol docstring (presence signals public API).

    Returns:
        One of the 6 layer strings.
    """
    symbol_type = symbol_type.lower()
    name_lower = symbol_name.lower()

    # --- Constants / macros: always 'constant' ---
    if symbol_type in ('constant', 'macro'):
        return 'constant'

    # --- Core data types: always 'core_type' ---
    if symbol_type in ('struct', 'enum', 'type_alias'):
        return 'core_type'

    # --- Entry-point detection (functions only) ---
    if symbol_type == 'function':
        if name_lower in _ENTRY_POINT_NAMES:
            return 'entry_point'
        # Also match patterns like run_server, start_app
        for ep in _ENTRY_POINT_NAMES:
            if name_lower.startswith(ep + '_') or name_lower.endswith('_' + ep):
                return 'entry_point'

    # --- Path-based classification ---
    path_parts = set(file_path.lower().replace('\\', '/').split('/'))

    # Infrastructure: path contains config/, util/, helper/, etc.
    if path_parts & _INFRA_PATH_SEGMENTS:
        return 'infrastructure'

    # Internal: nested symbol or private path
    if parent_symbol:
        return 'internal'
    if path_parts & _INTERNAL_PATH_SEGMENTS:
        return 'internal'

    # --- Public API: classes with public patterns ---
    if symbol_type in ('class', 'interface', 'trait', 'protocol'):
        # Name suffix heuristic: FooService, BarManager, etc.
        if any(symbol_name.endswith(suffix) for suffix in _PUBLIC_API_SUFFIXES):
            return 'public_api'
        # Has docstring → likely public
        if docstring and len(docstring.strip()) > 10:
            return 'public_api'
        # Default class with no infra/internal signal → public_api
        return 'public_api'

    # --- Functions default to public_api ---
    if symbol_type == 'function':
        return 'public_api'

    # --- Fallback for unknown types ---
    return 'public_api'


# =============================================================================
# Directory Exclusion Constants (single source of truth)
# =============================================================================

# Dot-prefixed directories that should NOT be excluded from file discovery
# or tree exploration.  These contain CI/CD configs, workflow definitions,
# and templates that are architecturally significant.
DOT_DIR_WHITELIST = frozenset({
    '.github',      # GitHub Actions workflows, issue templates, dependabot
    '.gitlab',      # GitLab CI configs
    '.circleci',    # CircleCI configs
    '.husky',       # Git hooks manager
    '.elitea',       # EliteA SDK agent definitions, toolkit configs, test pipelines
})

# Directories ignored during tree exploration and coverage tracking.
# Used by structure_tools._should_ignore_dir() and coverage calculations.
# NOTE: Does NOT include .github/.gitlab — those are in DOT_DIR_WHITELIST.
COVERAGE_IGNORE_DIRS = frozenset({
    '.git', '.svn', '.hg',                                            # VCS internals
    '.vscode', '.idea', '.vs',                                         # IDE config
    '__pycache__', '.pytest_cache', '.mypy_cache', '.tox', '.eggs',    # Python caches
    'egg-info', '.egg-info',                                           # Python packaging
    'venv', '.venv', 'env', '.env',                                    # Virtual environments
    'build', 'dist', 'target',                                         # Build outputs
    'node_modules',                                                     # JS dependencies
    'htmlcov', '.coverage', 'coverage',                                 # Coverage reports
    '.cache', 'tmp', 'temp',                                            # Temporary files
    '.DS_Store', 'Thumbs.db',                                          # OS artifacts
    'logs', 'log',                                                      # Log directories
})

# Default file-exclusion glob patterns for graph_builder file discovery.
# Used as the default exclude_patterns in _discover_files_by_language().
# NOTE: Do NOT add '**/.*' — it blocks whitelisted directories like
# .github and known dot-files such as .editorconfig / .gitignore.
DEFAULT_EXCLUDE_PATTERNS = (
    # Dependency / build directories
    '**/node_modules/**', '**/build/**', '**/dist/**', '**/target/**',
    # VCS internals
    '**/.git/**', '**/.svn/**', '**/.hg/**',
    # Python caches
    '**/__pycache__/**', '**/.pytest_cache/**', '**/.mypy_cache/**',
    '**/.tox/**', '**/.eggs/**',
    # IDE configuration
    '**/.vscode/**', '**/.idea/**', '**/.vs/**',
    # Virtual environments
    '**/.venv/**',
    # Coverage / caching
    '**/.coverage/**', '**/.cache/**',
    # Compiled artifacts
    '**/*.class', '**/*.jar', '**/*.war',
)


# ── Test code detection ──────────────────────────────────────────────
# Path-based heuristics for identifying test files.
# Used when ``exclude_tests`` feature flag is enabled.
# Matches are case-insensitive (compiled with re.IGNORECASE).

import re as _re  # noqa: E402 — local import to keep at bottom of file

_TEST_PATH_PATTERNS = [
    # ── Directory-based ──
    _re.compile(r'(^|/)tests?/', _re.IGNORECASE),              # tests/, test/
    _re.compile(r'(^|/)__tests__/', _re.IGNORECASE),           # JavaScript __tests__/
    _re.compile(r'(^|/)specs?/', _re.IGNORECASE),              # Ruby/JS spec/
    _re.compile(r'(^|/)testing/', _re.IGNORECASE),             # Go testing/
    _re.compile(r'(^|/)test_?helpers?/', _re.IGNORECASE),      # test_helper/, testhelpers/
    _re.compile(r'(^|/)fixtures/', _re.IGNORECASE),            # Test fixtures
    _re.compile(r'(^|/)mocks?/', _re.IGNORECASE),              # Mock directories
    _re.compile(r'(^|/)testdata/', _re.IGNORECASE),            # Go test data
    _re.compile(r'(^|/)testutils?/', _re.IGNORECASE),          # Test utilities

    # ── File-based ──
    _re.compile(r'(^|/)test_[^/]+\.\w+$', _re.IGNORECASE),         # test_foo.py
    _re.compile(r'(^|/)[^/]+_test\.\w+$', _re.IGNORECASE),         # foo_test.go
    _re.compile(r'(^|/)[^/]+\.test\.\w+$', _re.IGNORECASE),        # foo.test.js
    _re.compile(r'(^|/)[^/]+\.spec\.\w+$', _re.IGNORECASE),        # foo.spec.js
    _re.compile(r'(^|/)[^/]+Tests?\.\w+$', _re.IGNORECASE),        # FooTest.java
    _re.compile(r'(^|/)conftest\.py$', _re.IGNORECASE),             # pytest conftest
    _re.compile(r'(^|/)setup_tests?\.\w+$', _re.IGNORECASE),       # Test setup files
]


def is_test_path(rel_path: str) -> bool:
    """Check if a relative file path matches test code patterns.

    Uses path-based heuristics (directory names and file suffixes) to
    identify test files.  This is deterministic, language-agnostic,
    and covers the vast majority of real-world test layouts.

    Returns ``True`` if the path matches any test pattern.
    """
    return any(p.search(rel_path) for p in _TEST_PATH_PATTERNS)
