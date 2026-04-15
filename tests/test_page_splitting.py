"""Tests for post-structure page splitting when pages have too many target_symbols."""

import unittest
from unittest.mock import MagicMock, patch
from plugin_implementation.state.wiki_state import PageSpec, SectionSpec, WikiStructureSpec


def _make_page(name: str, symbols: list, **kwargs) -> PageSpec:
    """Helper to create a PageSpec with given symbols."""
    return PageSpec(
        page_name=name,
        page_order=1,
        description=f"Test page {name}",
        content_focus="test",
        rationale="test",
        target_symbols=symbols,
        target_docs=kwargs.get("target_docs", []),
        target_folders=kwargs.get("target_folders", []),
        key_files=kwargs.get("key_files", []),
        retrieval_query=kwargs.get("retrieval_query", ""),
    )


def _make_structure(pages: list) -> WikiStructureSpec:
    """Helper to wrap pages into a WikiStructureSpec."""
    section = SectionSpec(
        section_name="Test Section",
        section_order=1,
        description="Test",
        rationale="Test",
        pages=pages,
    )
    return WikiStructureSpec(
        wiki_title="Test Wiki",
        overview="Test",
        sections=[section],
        total_pages=len(pages),
    )


class TestPageSplitting(unittest.TestCase):
    """Test _split_overloaded_pages in OptimizedWikiGenerationAgent."""

    def _get_agent(self, max_syms=25):
        """Create a minimal mock agent with the real splitting method."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent

        agent = MagicMock(spec=OptimizedWikiGenerationAgent)
        agent._split_overloaded_pages = (
            OptimizedWikiGenerationAgent._split_overloaded_pages.__get__(agent)
        )
        agent._get_env_int = MagicMock(return_value=max_syms)
        agent.MAX_SYMBOLS_PER_PAGE = max_syms
        # No graph — symbols will be chunked linearly
        agent.retriever_stack = MagicMock()
        agent.retriever_stack.relationship_graph = None
        return agent

    def test_no_split_under_threshold(self):
        """Pages with ≤ max symbols should not be split."""
        agent = self._get_agent(max_syms=25)
        page = _make_page("Small Page", [f"Sym{i}" for i in range(10)])
        structure = _make_structure([page])

        agent._split_overloaded_pages(structure)

        self.assertEqual(len(structure.sections[0].pages), 1)
        self.assertEqual(structure.sections[0].pages[0].page_name, "Small Page")

    def test_split_over_threshold(self):
        """Pages with > max symbols should be split into sub-pages."""
        agent = self._get_agent(max_syms=10)
        symbols = [f"Sym{i}" for i in range(25)]
        page = _make_page("Big Page", symbols)
        structure = _make_structure([page])

        agent._split_overloaded_pages(structure)

        pages = structure.sections[0].pages
        self.assertEqual(len(pages), 3)  # 25 / 10 = 3 pages (10, 10, 5)
        self.assertEqual(pages[0].page_name, "Big Page (Part 1)")
        self.assertEqual(pages[1].page_name, "Big Page (Part 2)")
        self.assertEqual(pages[2].page_name, "Big Page (Part 3)")
        # All symbols accounted for
        all_syms = []
        for p in pages:
            all_syms.extend(p.target_symbols)
        self.assertEqual(sorted(all_syms), sorted(symbols))

    def test_split_preserves_metadata(self):
        """Split sub-pages should inherit parent's docs, folders, etc."""
        agent = self._get_agent(max_syms=10)
        page = _make_page(
            "Big Page",
            [f"Sym{i}" for i in range(15)],
            target_docs=["docs/readme.md"],
            target_folders=["src/main"],
            key_files=["src/main/app.py"],
            retrieval_query="app main",
        )
        structure = _make_structure([page])

        agent._split_overloaded_pages(structure)

        for sub_page in structure.sections[0].pages:
            self.assertEqual(sub_page.target_docs, ["docs/readme.md"])
            self.assertEqual(sub_page.target_folders, ["src/main"])
            self.assertEqual(sub_page.key_files, ["src/main/app.py"])
            self.assertEqual(sub_page.retrieval_query, "app main")

    def test_total_pages_updated(self):
        """total_pages should be updated after splitting."""
        agent = self._get_agent(max_syms=10)
        small_page = _make_page("Small", [f"S{i}" for i in range(5)])
        big_page = _make_page("Big", [f"B{i}" for i in range(30)])
        structure = _make_structure([small_page, big_page])

        agent._split_overloaded_pages(structure)

        # small stays 1, big splits into 3 (30/10) → total = 4
        self.assertEqual(len(structure.sections[0].pages), 4)
        self.assertEqual(structure.total_pages, 4)

    def test_exact_threshold_no_split(self):
        """Page with exactly max_syms symbols should NOT be split."""
        agent = self._get_agent(max_syms=25)
        page = _make_page("Exact", [f"Sym{i}" for i in range(25)])
        structure = _make_structure([page])

        agent._split_overloaded_pages(structure)

        self.assertEqual(len(structure.sections[0].pages), 1)

    def test_no_symbols_no_split(self):
        """Doc-only pages with no symbols should not be split."""
        agent = self._get_agent(max_syms=10)
        page = _make_page("Doc Page", [], target_docs=["docs/guide.md"])
        structure = _make_structure([page])

        agent._split_overloaded_pages(structure)

        self.assertEqual(len(structure.sections[0].pages), 1)

    def test_split_with_graph_groups_by_file(self):
        """When graph is available, symbols should be grouped by file path."""
        import networkx as nx

        agent = self._get_agent(max_syms=10)

        # Create a mock graph with name_index and node data
        graph = nx.DiGraph()
        name_index = {}
        # 15 symbols: 8 from file_a.py, 7 from file_b.py
        for i in range(8):
            nid = f"py::file_a.py::SymA{i}"
            graph.add_node(nid, rel_path="src/file_a.py", symbol_name=f"SymA{i}")
            name_index[f"SymA{i}"] = [nid]
        for i in range(7):
            nid = f"py::file_b.py::SymB{i}"
            graph.add_node(nid, rel_path="src/file_b.py", symbol_name=f"SymB{i}")
            name_index[f"SymB{i}"] = [nid]
        graph._name_index = name_index

        agent.retriever_stack.relationship_graph = graph

        symbols = [f"SymA{i}" for i in range(8)] + [f"SymB{i}" for i in range(7)]
        page = _make_page("Grouped Page", symbols)
        structure = _make_structure([page])

        agent._split_overloaded_pages(structure)

        pages = structure.sections[0].pages
        self.assertEqual(len(pages), 2)  # 8+7=15, max=10 → 2 pages
        # File_a symbols should stay together in one page
        page1_syms = set(pages[0].target_symbols)
        page2_syms = set(pages[1].target_symbols)
        # All SymA* should be in one page
        syma_set = {f"SymA{i}" for i in range(8)}
        symb_set = {f"SymB{i}" for i in range(7)}
        self.assertTrue(
            syma_set.issubset(page1_syms) or syma_set.issubset(page2_syms),
            "All SymA* symbols should be in the same sub-page",
        )


if __name__ == "__main__":
    unittest.main()
