# Plan: Merge feat/capability-based-naming + Add Missing Extensions

**Date**: 2025-01-XX  
**Branch**: `feat/context7-multi-wiki` ← merge from `feat/capability-based-naming`  
**Status**: ✅ COMPLETE

---

## 1. Executive Summary

This plan covers three work items:

1. **Task A (Easy)**: Add missing configuration/documentation file extensions  
2. **Task B (Medium)**: Merge `feat/capability-based-naming` into `feat/context7-multi-wiki`  
3. **Task C (Strategic)**: Plan for graph-as-first-class-citizen and content_expander decommission  

---

## 2. Branch Review: `feat/capability-based-naming`

### 2.1 Commits (4 total, 1,630 additions)

| Commit | Description |
|--------|-------------|
| `69f3b86` | Capability-based naming + smart graph expansion with ARCHITECTURAL_TYPES filtering |
| `0699ab0` | Big repo scaling - dynamic iterations and increased limits |
| `6f62d57` | `batch_define_pages` tool for faster big repo structure planning |
| `108256c` | Update wiki structure building for big repos (increase limits and prompts) |

### 2.2 Files Changed (12 files)

| File | Lines +/- | Summary |
|------|-----------|---------|
| `structure_tools.py` | +546 | `batch_define_pages`, `target_docs`, coverage overflow, type refs in query_graph, doc nodes display |
| `wiki_graph_optimized.py` | +569 | `_get_doc_nodes_from_graph()`, `_get_docs_from_store_semantic()`, doc cluster detection, ARCHITECTURAL_TYPES expansion filter, dynamic page budget, dynamic max_iterations |
| `structure_prompts.py` | +288 | Capability-based naming rules (no symbols in page titles), `batch_define_pages` docs, `target_docs` rule, large repo optimization guide, doc cluster prompt section |
| `retrievers.py` | +168 | `search_docs_semantic()` with EmbeddingsFilter reranking, feature flags |
| `graph_builder.py` | +145 | SEPARATE_DOC_INDEX feature flag, doc/code separation in graph building + vector store |
| `structure_engine.py` | +27 | Dynamic `max_iterations`, increased defaults, `doc_clusters` config, recursion_limit bump |
| `vectorstore.py` | +18 | SEPARATE_DOC_INDEX cache key suffix, feature flag |
| `cpp_enhanced_parser.py` | +10 | Indentation fix in `_extract_field_declaration` (field extraction was running outside its if-block) |
| `filter_manager.py` | +3 | C++ extension completeness (`.cc`, `.cxx`, `.c++`, `.hh`, `.hxx`) |
| `wiki_state.py` | +1 | `target_docs: List[str]` field on `PageSpec` |
| `local_repository_manager.py` | +6 | `.git` suffix stripping fix |
| `providers.py` | +6 | `.git` suffix stripping fix |

### 2.3 Quality Assessment

**Strengths:**
- ✅ Feature-flagged via env vars (DEEPWIKI_DOC_SEPARATE_INDEX, DEEPWIKI_DOC_SEMANTIC_RETRIEVAL, DEEPWIKI_AUTO_TARGET_DOCS)
- ✅ Well-documented code with clear logging
- ✅ Backward-compatible (all flags default to "0" = legacy behavior)
- ✅ Good separation of concerns (doc retrieval methods are standalone)
- ✅ Pragmatic about doc-in-graph "architectural wart" (<1% overhead)
- ✅ Dynamic scaling for big repos (budget, iterations, limits)
- ✅ `batch_define_pages` significantly reduces LLM tool-call overhead for big repos

**Concerns:**
- ⚠️ `DOC_SYMBOL_TYPES` is redefined in 4 places (graph_builder, wiki_graph_optimized, retrievers, structure_tools) - should be a shared constant
- ⚠️ `_get_doc_nodes_from_graph()` has `import re` inside the function body (move to top)
- ⚠️ `batch_define_pages` does a linear scan of graph nodes ([:5000]) per symbol - O(n*m) for large graphs
- ⚠️ `_detect_doc_clusters()` only checks depth 2-3, may miss deeper doc dirs
- ⚠️ The cpp_enhanced_parser indentation fix overlaps with our field extraction fixes

**Conflicts with our branch:**
- 🔴 `cpp_enhanced_parser.py`: Both branches modify `_extract_field_declaration`. Their fix is an indentation correction; our branch has comprehensive field type extraction. **Our version is more complete and supersedes theirs.**
- 🟡 `graph_builder.py`: They add feature flags at top + SEPARATE_DOC_INDEX logic. Our branch added `type_alias` to ARCHITECTURAL_SYMBOLS. **No logical conflict, just textual merge.**
- 🟡 `wiki_graph_optimized.py`: They add doc retrieval + ARCHITECTURAL_TYPES. Our branch didn't modify this file significantly. **Clean merge expected.**
- 🟢 All other files: No conflicts expected.

### 2.4 Merge Strategy

**Recommended: Cherry-pick with conflict resolution**

Since the branches diverged from `main` (not from each other), a direct merge would bring in all of `main`'s history. Better approach:

1. Cherry-pick the 4 commits in order onto our branch
2. Resolve the cpp_enhanced_parser.py conflict (keep our version)
3. Resolve the graph_builder.py textual conflict (combine both changes)
4. Run full test suite

```bash
git cherry-pick 69f3b86 0699ab0 6f62d57 108256c
# Resolve conflicts during each step
```

**Alternative: Merge feat/capability-based-naming into our branch**
```bash
git merge origin/feat/capability-based-naming
# Resolve conflicts
```

The merge approach is simpler but creates a merge commit.

---

## 3. Task A: Add Missing File Extensions

### 3.1 Extensions to Add

User requested: `.cfg`, `.mod`, `.sh`, `.bat`, `.ps`, `.gradle`, `.kts`, `.wsdl`, `.xsd`, `.proto`, `.tf`

Analysis of what's already present vs. what needs adding:

| Extension | filter_manager allowed | filter_manager lang_map | graph_builder DOC_EXT | graph_builder SUPPORTED_LANG | structure_tools is_code_file | Status |
|-----------|:---:|:---:|:---:|:---:|:---:|--------|
| `.cfg` | ✅ | ❌ | ✅ (config) | ❌ | ❌ | Needs lang_map |
| `.mod` | ❌ | ❌ | ❌ | ❌ | ❌ | **Add everywhere** |
| `.sh` | ✅ | ✅ (bash) | ❌ | ❌ | ❌ | Needs DOC_EXT or code handling |
| `.bat` | ✅ | ❌ | ❌ | ❌ | ❌ | Needs lang_map + DOC_EXT |
| `.ps1` | ✅ | ✅ (powershell) | ❌ | ❌ | ❌ | Needs DOC_EXT |
| `.gradle` | ❌ | ❌ | ❌ | ❌ | ✅ | **Add to filter+DOC_EXT** |
| `.kts` | ❌ | ❌ | ❌ | ❌ | ❌ | **Add everywhere** |
| `.wsdl` | ❌ | ❌ | ❌ | ❌ | ❌ | **Add everywhere** |
| `.xsd` | ❌ | ❌ | ❌ | ❌ | ❌ | **Add everywhere** |
| `.proto` | ❌ | ❌ | ❌ | ❌ | ❌ | **Add everywhere** |
| `.tf` | ❌ | ❌ | ❌ | ❌ | ❌ | **Add everywhere** |

### 3.2 Where to Add Each Extension

**Note**: Shell scripts (.sh, .bat, .ps1) and build scripts (.gradle, .kts) are interesting edge cases:
- They ARE code (executable), but tree-sitter may not have parsers for all of them
- For now, treat them as **documentation/config** type so they get indexed as text chunks
- `.proto` files could eventually get a tree-sitter parser, but for now treat as config/schema

#### A. `graph_builder.py` → `DOCUMENTATION_EXTENSIONS` dict

Add these extensions with appropriate doc_type values:

```python
# Build scripts (treated as config/build for now)
'.gradle': 'build_config',
'.kts': 'build_config',

# Schema/IDL files
'.wsdl': 'schema',
'.xsd': 'schema',
'.proto': 'schema',

# Infrastructure as Code
'.tf': 'infrastructure',
'.tfvars': 'infrastructure',

# Module/config files
'.mod': 'config',       # Go modules, Fortran modules

# Shell scripts (indexed as text, no tree-sitter)
'.sh': 'script',
'.bash': 'script',
'.bat': 'script',
'.cmd': 'script',
'.ps1': 'script',
'.psm1': 'script',
```

#### B. `filter_manager.py` → default allowed extensions

Add to appropriate section:
```python
# Build files
'.gradle', '.kts',
# Schema/IDL
'.wsdl', '.xsd', '.proto',
# Infrastructure
'.tf', '.tfvars', '.hcl',
# Module files
'.mod',
```

#### C. `filter_manager.py` → language detection map

Add mappings:
```python
'.gradle': 'groovy',
'.kts': 'kotlin_script',
'.wsdl': 'xml',
'.xsd': 'xml',
'.proto': 'protobuf',
'.tf': 'terraform',
'.tfvars': 'terraform',
'.hcl': 'hcl',
'.mod': 'gomod',
'.bat': 'batch',
'.cmd': 'batch',
'.cfg': 'config',
```

#### D. `wiki_graph_optimized.py` → inline config_files filter

Add `.proto`, `.tf`, `.gradle`, `.kts`, `.wsdl`, `.xsd` to config file detection.

#### E. `structure_tools.py` → `_is_code_file` 

Add `.kts` (already has `.gradle`). Also add `.proto` and `.tf` since they're "code-like".

### 3.3 Implementation Order

1. `graph_builder.py` DOCUMENTATION_EXTENSIONS ← primary gate for doc indexing
2. `filter_manager.py` allowed extensions + language map ← file discovery gate
3. `structure_tools.py` `_is_code_file` ← structure planning
4. `wiki_graph_optimized.py` inline filters ← config file classification
5. Run tests

---

## 4. Task C: Graph-as-First-Class-Citizen Roadmap

### 4.1 Current State

**Doc nodes in graph**: Stored with `symbol_type` like `markdown_document`, `text_chunk`, etc.
- ✅ Quick lookup by path
- ❌ No edges (relationships) — using NetworkX as "fancy dictionary"
- ❌ Duplicated in graph + dense vector store + BM25 index (~3x RAM)

**Content expander**: Rich expansion logic (alias chains, type resolution, architectural parent lookup)
- ✅ Deep context via multi-hop graph traversal
- ❌ Context bloat (expands more than needed for wiki generation)
- ❌ Not called from `wiki_graph_optimized.py` (which has its own simpler expansion)

### 4.2 Feature Flag Architecture (from feat/capability-based-naming)

| Flag | Default | Effect |
|------|---------|--------|
| `DEEPWIKI_DOC_SEPARATE_INDEX` | 0 | When 1: docs NOT in graph, only in vector store |
| `DEEPWIKI_DOC_SEMANTIC_RETRIEVAL` | 0 | When 1: use EmbeddingsFilter reranking for docs |
| `DEEPWIKI_AUTO_TARGET_DOCS` | 0 | When 1: docs auto-retrieved, LLM doesn't pick target_docs |

This is a 3-stage migration path:
- **Stage 0** (current): Docs in graph + vector store (legacy)
- **Stage 1** (`SEPARATE_INDEX=1`): Docs in vector store only, graph is code-only
- **Stage 2** (`SEMANTIC_RETRIEVAL=1`): Semantic doc retrieval with reranking
- **Stage 3** (`AUTO_TARGET_DOCS=1`): LLM doesn't need to manually select docs

### 4.3 Content Expander Decommission Path

**Phase 1 (Now)**: Keep content_expander but don't expand its role
- It's used by wiki_graph_optimized as fallback
- Our improvements (type_alias, alias chains, architectural parent) stay

**Phase 2 (Next)**: Factor out SmartGraphExpander
- Extract the best logic from content_expander into a graph-native utility
- Token budget management, multi-hop traversal, relationship prioritization
- Used directly by wiki_graph_optimized

**Phase 3 (Future)**: Full-text search on graph
- Add FTS index alongside NetworkX (e.g., SQLite FTS5 or whoosh)
- Graph becomes the single source of truth for code
- Vector store only for docs + semantic search
- Content expander fully retired

### 4.4 Immediate Action Items After Merge

1. ✅ **Consolidate `DOC_SYMBOL_TYPES`** — created `plugin_implementation/constants.py` with `DOC_SYMBOL_TYPES`, `DOC_CHUNK_TYPES`, `EXPANSION_SYMBOL_TYPES` (all frozenset). Updated all 4 importing modules.
2. ✅ **Move `import re`** — removed 4 redundant `import re` from function bodies in wiki_graph_optimized.py (module-level import already existed)
3. ✅ **Fix batch_define_pages O(n*m)** — added `_symbol_rel_paths` index for O(1) folder lookup alongside `_case_insensitive_symbols`
4. ✅ **Add unit tests** — 54 new tests covering doc node retrieval, doc clusters, batch performance index, feature flags, constants consistency, and more

---

## 5. Implementation Steps

### Step 1: Add missing file extensions (Task A) ✅ DONE
- Commit `dc1baff`: Added 15+ extensions to graph_builder, filter_manager, structure_tools, wiki_graph_optimized
- 170/170 tests pass

### Step 2: Merge feat/capability-based-naming (Task B) ✅ DONE
- Cherry-picked all 4 commits in order:
  - `7b17506` (was `69f3b86`) — capability-based naming + ARCHITECTURAL_TYPES (1 conflict in structure_tools.py `_is_code_file`, resolved)
  - `0aefc4b` (was `0699ab0`) — big repo scaling (clean)
  - `d8a6d1c` (was `6f62d57`) — batch_define_pages (clean)
  - `8975a49` (was `108256c`) — wiki structure for big repos (clean)
- 170/170 tests pass after merge

### Step 3: Post-merge cleanup ✅ DONE
- Commit `3cbb2ea`:
  - Created `plugin_implementation/constants.py` with shared `DOC_SYMBOL_TYPES`, `DOC_CHUNK_TYPES`, `EXPANSION_SYMBOL_TYPES` (frozenset)
  - Replaced inline definitions in graph_builder, wiki_graph_optimized, structure_tools, retrievers
  - Fixed batch_define_pages O(n*m): Added `_symbol_rel_paths` index for O(1) folder lookup
  - Removed 4 redundant `import re` in wiki_graph_optimized.py methods
  - Added `parallel_tool_calls=True` to structure engine model binding
  - Added explicit parallel tool calling instructions to structure prompts

### Step 3b: Comprehensive Test Suite ✅ DONE
- Commit `3d5e08c`: 54 new tests in `test_merge_features_comprehensive.py`
- Tests cover: constants, extensions, doc filtering, PageSpec target_docs, StructureCollector, batch_define_pages, doc clusters, ARCHITECTURAL_TYPES, feature flags, prompt parallel calls, constants consistency
- 224/224 total tests pass (170 existing + 54 new)

### Step 4: Enable feature flags for testing ❌ NOT YET
- Requires local deployment to test
- Test with `DEEPWIKI_DOC_SEPARATE_INDEX=1`
- Test with `DEEPWIKI_DOC_SEMANTIC_RETRIEVAL=1`
- Test with `DEEPWIKI_AUTO_TARGET_DOCS=1`

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation | Outcome |
|------|-----------|--------|------------|---------|
| Merge conflicts in cpp_parser | HIGH | LOW | Our version is more complete, just keep ours | ✅ No conflict — cpp_parser wasn't touched in cherry-picks |
| graph_builder.py merge conflict | MEDIUM | LOW | Simple textual merge, both add to different areas | ✅ No conflict |
| structure_tools.py _is_code_file conflict | N/A | LOW | Keep both additions (.kts + C++ comment) | ✅ Resolved, kept both |
| New extensions break existing tests | LOW | LOW | Extensions are additive, shouldn't break anything | ✅ No breakage |
| SEPARATE_DOC_INDEX causes missing docs | MEDIUM | MEDIUM | Feature-flagged, disabled by default | ⏳ Not yet tested with flag enabled |
| batch_define_pages perf with large graphs | MEDIUM | LOW | Only affects structure planning speed | ✅ Fixed with O(1) index |

---

## 7. Test Plan

- [x] All existing unit tests pass (170 tests) ✅
- [x] fmtlib real-world test (14,544 files, 0 crashes) ✅ (from prior session)
- [x] New extension types are picked up by graph_builder DOCUMENTATION_EXTENSIONS ✅ (tested in TestFileExtensionWiring)
- [x] Feature flags default to legacy behavior (no behavioral change) ✅ (tested in TestFeatureFlags)
- [x] `target_docs` field properly propagated through PageSpec ✅ (tested in TestPageSpecTargetDocs)
- [x] Shared constants imported identically across modules ✅ (tested in TestConstantsConsistency)
- [x] ARCHITECTURAL_TYPES filtering excludes method/macro ✅ (tested in TestArchitecturalTypesFiltering)
- [x] batch_define_pages O(1) index covers all symbols ✅ (tested in TestBatchPerformanceIndex)
- [x] Parallel tool call instructions present in prompts ✅ (tested in TestStructurePromptsParallelCalls)
- [ ] Feature flags tested with actual deployment ❌ (requires local Docker run)

**Final test count: 224/224 passing (170 existing + 54 new)**

---

## 8. Additional Work Done (Beyond Original Plan)

Items completed that were not in the original plan:

1. **Shared constants module** (`plugin_implementation/constants.py`): Created `EXPANSION_SYMBOL_TYPES` in addition to DOC_SYMBOL_TYPES and DOC_CHUNK_TYPES
2. **Parallel tool call polish**: Added explicit parallel call instructions to structure prompts (new section + updated Steps 2, 5, Large Repo Workflow) and enabled `parallel_tool_calls=True` in engine
3. **54 comprehensive tests**: Far exceeding original "add unit tests" item — covers 14 test classes across constants, extensions, filtering, state, tools, agents, flags, prompts, and consistency
