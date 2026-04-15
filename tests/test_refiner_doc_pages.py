"""
Tests for LLM-driven doc page creation in assemble_wiki_structure.

Verifies:
- LLM-created doc pages (cluster_id = "doc_0", "doc_1", ...) are processed
- Doc clusters assigned to code pages via doc_files are marked as covered
- Uncovered doc clusters fall back to mechanical _add_doc_pages
- No duplicate doc pages when LLM covers all clusters
- Doc-only repo structure still works via _build_doc_only_structure
"""

import pytest

from plugin_implementation.wiki_structure_planner.structure_refiner import (
    assemble_wiki_structure,
    _add_doc_pages,
)
from plugin_implementation.wiki_structure_planner.structure_skeleton import (
    DirCluster,
    DocCluster,
    StructureSkeleton,
    SymbolInfo,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sym(name, layer='internal', connections=2, sym_type='class',
         rel_path='src/core.py'):
    return SymbolInfo(
        name=name, type=sym_type, rel_path=rel_path,
        layer=layer, connections=connections, docstring='',
    )


def _skeleton(code_clusters=None, doc_clusters=None):
    """Build a minimal StructureSkeleton."""
    code_clusters = code_clusters or []
    doc_clusters = doc_clusters or []
    total_syms = sum(c.total_symbols for c in code_clusters)
    return StructureSkeleton(
        code_clusters=code_clusters,
        doc_clusters=doc_clusters,
        total_arch_symbols=total_syms,
        total_dirs_covered=sum(len(c.dirs) for c in code_clusters),
        total_dirs_in_repo=10,
        repo_languages={'python': 100},
        effective_depth=2,
        repo_name='test-repo',
    )


def _code_cluster(cid, dirs, symbols):
    return DirCluster(
        cluster_id=cid,
        dirs=dirs,
        symbols=symbols,
        total_symbols=len(symbols),
        primary_languages=['python'],
        depth_range=(1, 2),
    )


def _doc_cluster(dir_path, doc_files):
    return DocCluster(
        dir_path=dir_path,
        file_count=len(doc_files),
        doc_files=doc_files,
        doc_types=['md'],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLLMDocPages:
    """LLM creates standalone doc pages via cluster_id = 'doc_N'."""

    def test_llm_creates_doc_page(self):
        """Doc cluster handled by LLM as standalone page."""
        dc = _doc_cluster('docs/guides', ['docs/guides/intro.md', 'docs/guides/setup.md'])
        cc = _code_cluster(0, ['src/core'], [_sym('CoreClass')])
        skeleton = _skeleton(code_clusters=[cc], doc_clusters=[dc])

        llm_items = [
            {'cluster_id': 0, 'page_name': 'Core Logic', 'section_name': 'Core',
             'description': 'Core', 'retrieval_query': 'core'},
            {'cluster_id': 'doc_0', 'page_name': 'Getting Started Guides',
             'section_name': 'Documentation', 'description': 'Intro & setup guides',
             'retrieval_query': 'guides intro setup'},
        ]

        result = assemble_wiki_structure(llm_items, skeleton, 'test-repo')

        # Find the doc page
        all_pages = [p for s in result['sections'] for p in s['pages']]
        doc_pages = [p for p in all_pages if 'doc_0' in p.get('rationale', '')]
        assert len(doc_pages) == 1

        dp = doc_pages[0]
        assert dp['page_name'] == 'Getting Started Guides'
        assert dp['target_symbols'] == []
        assert dp['target_docs'] == ['docs/guides/intro.md', 'docs/guides/setup.md']
        assert dp['target_folders'] == ['docs/guides']

    def test_no_fallback_when_llm_covers_all(self):
        """No safety-net pages when LLM covers all doc clusters."""
        dc = _doc_cluster('docs', ['docs/README.md'])
        cc = _code_cluster(0, ['src'], [_sym('App')])
        skeleton = _skeleton(code_clusters=[cc], doc_clusters=[dc])

        llm_items = [
            {'cluster_id': 0, 'page_name': 'Application', 'section_name': 'Core',
             'description': 'App', 'retrieval_query': 'app'},
            {'cluster_id': 'doc_0', 'page_name': 'Project Docs',
             'section_name': 'Docs', 'description': 'Main docs',
             'retrieval_query': 'docs'},
        ]

        result = assemble_wiki_structure(llm_items, skeleton, 'test-repo')
        all_pages = [p for s in result['sections'] for p in s['pages']]

        # Should have exactly 2 pages (1 code + 1 doc), no fallback
        assert len(all_pages) == 2
        assert not any('fallback' in p.get('rationale', '').lower() for p in all_pages)


class TestDocAssignedToCodeCluster:
    """Doc files assigned to code clusters via doc_files still mark cluster as covered."""

    def test_assigned_docs_prevent_fallback(self):
        """When LLM assigns doc files to a code cluster, no fallback page created.

        Even though the refiner no longer requests doc assignment to code
        clusters, the LLM *might* still do it.  The assembly logic must
        recognise coverage and NOT create a duplicate fallback page.
        Code pages themselves won't carry the docs (target_docs stays empty).
        """
        dc = _doc_cluster('docs/api', ['docs/api/reference.md'])
        cc = _code_cluster(0, ['src/api'], [_sym('ApiHandler')])
        skeleton = _skeleton(code_clusters=[cc], doc_clusters=[dc])

        llm_items = [
            {'cluster_id': 0, 'page_name': 'API Layer', 'section_name': 'API',
             'description': 'API', 'retrieval_query': 'api',
             'doc_files': ['docs/api/reference.md']},
        ]

        result = assemble_wiki_structure(llm_items, skeleton, 'test-repo')
        all_pages = [p for s in result['sections'] for p in s['pages']]

        # Only 1 code page, no fallback doc page
        assert len(all_pages) == 1
        assert all_pages[0]['page_name'] == 'API Layer'
        # Code pages no longer carry doc_files as target_docs
        assert all_pages[0]['target_docs'] == []


class TestFallbackDocPages:
    """Uncovered doc clusters get safety-net pages."""

    def test_uncovered_cluster_gets_fallback(self):
        """Doc cluster NOT handled by LLM gets a fallback page."""
        dc0 = _doc_cluster('docs/tutorials', ['docs/tutorials/t1.md'])
        dc1 = _doc_cluster('examples', ['examples/demo.md', 'examples/quickstart.md'])
        cc = _code_cluster(0, ['src'], [_sym('Main')])
        skeleton = _skeleton(code_clusters=[cc], doc_clusters=[dc0, dc1])

        # LLM covers doc_0 but NOT doc_1
        llm_items = [
            {'cluster_id': 0, 'page_name': 'Main Module', 'section_name': 'Core',
             'description': 'Main', 'retrieval_query': 'main'},
            {'cluster_id': 'doc_0', 'page_name': 'Tutorials',
             'section_name': 'Learning', 'description': 'Tutorials',
             'retrieval_query': 'tutorials'},
        ]

        result = assemble_wiki_structure(llm_items, skeleton, 'test-repo')
        all_pages = [p for s in result['sections'] for p in s['pages']]

        # 3 pages: 1 code + 1 LLM doc + 1 fallback doc
        assert len(all_pages) == 3
        fallback = [p for p in all_pages if 'Doc-only directory' in p.get('rationale', '')]
        assert len(fallback) == 1
        assert fallback[0]['target_folders'] == ['examples']

    def test_all_uncovered_gets_all_fallbacks(self):
        """When LLM ignores all doc clusters, all get fallback pages."""
        dc = _doc_cluster('docs', ['docs/README.md'])
        cc = _code_cluster(0, ['src'], [_sym('App')])
        skeleton = _skeleton(code_clusters=[cc], doc_clusters=[dc])

        llm_items = [
            {'cluster_id': 0, 'page_name': 'App', 'section_name': 'Core',
             'description': 'App', 'retrieval_query': 'app'},
            # LLM ignores doc clusters entirely
        ]

        result = assemble_wiki_structure(llm_items, skeleton, 'test-repo')
        all_pages = [p for s in result['sections'] for p in s['pages']]

        assert len(all_pages) == 2  # 1 code + 1 fallback
        fallback = [p for p in all_pages if 'Doc-only directory' in p.get('rationale', '')]
        assert len(fallback) == 1

    def test_no_doc_clusters_no_fallback(self):
        """When there are no doc clusters, no fallback pages are created."""
        cc = _code_cluster(0, ['src'], [_sym('App')])
        skeleton = _skeleton(code_clusters=[cc], doc_clusters=[])

        llm_items = [
            {'cluster_id': 0, 'page_name': 'App', 'section_name': 'Core',
             'description': 'App', 'retrieval_query': 'app'},
        ]

        result = assemble_wiki_structure(llm_items, skeleton, 'test-repo')
        all_pages = [p for s in result['sections'] for p in s['pages']]
        assert len(all_pages) == 1


class TestInvalidDocClusterIds:
    """Invalid doc_N cluster_ids are gracefully skipped."""

    def test_invalid_doc_id_format(self):
        """Non-numeric suffix in doc_X is skipped."""
        dc = _doc_cluster('docs', ['docs/a.md'])
        cc = _code_cluster(0, ['src'], [_sym('X')])
        skeleton = _skeleton(code_clusters=[cc], doc_clusters=[dc])

        llm_items = [
            {'cluster_id': 0, 'page_name': 'X', 'section_name': 'S',
             'description': 'd', 'retrieval_query': 'q'},
            {'cluster_id': 'doc_abc', 'page_name': 'Bad', 'section_name': 'S',
             'description': 'd', 'retrieval_query': 'q'},
        ]

        result = assemble_wiki_structure(llm_items, skeleton, 'test-repo')
        all_pages = [p for s in result['sections'] for p in s['pages']]

        # 1 code + 1 fallback doc (since doc_abc is invalid)
        assert len(all_pages) == 2
        fallback = [p for p in all_pages if 'Doc-only directory' in p.get('rationale', '')]
        assert len(fallback) == 1

    def test_out_of_range_doc_id(self):
        """Doc index beyond range is skipped and cluster gets fallback."""
        dc = _doc_cluster('docs', ['docs/a.md'])
        cc = _code_cluster(0, ['src'], [_sym('X')])
        skeleton = _skeleton(code_clusters=[cc], doc_clusters=[dc])

        llm_items = [
            {'cluster_id': 0, 'page_name': 'X', 'section_name': 'S',
             'description': 'd', 'retrieval_query': 'q'},
            {'cluster_id': 'doc_99', 'page_name': 'Out of range',
             'section_name': 'S', 'description': 'd', 'retrieval_query': 'q'},
        ]

        result = assemble_wiki_structure(llm_items, skeleton, 'test-repo')
        all_pages = [p for s in result['sections'] for p in s['pages']]

        # 1 code + 1 fallback doc (since doc_99 doesn't exist)
        assert len(all_pages) == 2
