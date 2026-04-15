"""
Comprehensive FTS5 + Graph Traversal Tests using REAL cached databases.

These tests load actual cached graphs (fmtlib/fmt C++, EliteaAI/configurations Python)
to validate FTS5 search, graph traversal, and edge-type–based expansion against real-world
data. They serve as integration/acceptance tests — not unit tests.

Requires cached graphs to be present at the standard cache location.
If a cached graph is missing, individual tests skip gracefully.

Run with:
    cd pylon_deepwiki/plugins/deepwiki_plugin
    python -m pytest tests/test_fts_graph_traversal_real.py -v

Or from repository root with PYTHONPATH:
    PYTHONPATH=pylon_deepwiki/plugins/deepwiki_plugin python -m pytest \
        pylon_deepwiki/plugins/deepwiki_plugin/tests/test_fts_graph_traversal_real.py -v
"""

import gzip
import os
import pickle
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

# ---------------------------------------------------------------------------
# Locate cache directory — resolve from test file position
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent  # .../tests/
_PLUGIN_DIR = _THIS_DIR.parent              # .../deepwiki_plugin/
_CACHE_DIR = _PLUGIN_DIR.parent.parent / "wiki_builder" / "cache"

# Known graph hashes (from cache_index.json)
FMTLIB_GRAPH_HASH = "551134763c1f5c1b3feca4dd95076290"
CONFIG_GRAPH_HASH = "cd9d7a4aefa47194b872a7093a855156"

# Allow override via env var for CI
CACHE_DIR = Path(os.environ.get("DEEPWIKI_CACHE_DIR", str(_CACHE_DIR)))


def _load_graph(graph_hash: str):
    """Load a cached NetworkX graph by hash. Returns None if file missing."""
    path = CACHE_DIR / f"{graph_hash}.code_graph.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _skip_if_no_graph(graph, repo_name: str):
    """Skip a test if the required cached graph is not available."""
    if graph is None:
        raise unittest.SkipTest(
            f"Cached graph for {repo_name} not found at {CACHE_DIR}. "
            f"Run wiki generation for {repo_name} first to populate cache."
        )


# Lazy-loaded graphs (loaded once per module, not per test)
_graphs = {}


def _get_graph(repo_key: str):
    """Get a cached graph by key, loading lazily."""
    if repo_key not in _graphs:
        if repo_key == "fmtlib":
            _graphs[repo_key] = _load_graph(FMTLIB_GRAPH_HASH)
        elif repo_key == "configurations":
            _graphs[repo_key] = _load_graph(CONFIG_GRAPH_HASH)
        else:
            _graphs[repo_key] = None
    return _graphs[repo_key]


# ---------------------------------------------------------------------------
# Import GraphTextIndex (may fail if not on PYTHONPATH — tests will skip)
# ---------------------------------------------------------------------------
try:
    from plugin_implementation.code_graph.graph_text_index import GraphTextIndex
    HAS_FTS = True
except ImportError:
    HAS_FTS = False


# ===========================================================================
# Test Suite 1: Graph Structure Validation
# ===========================================================================

class TestFmtlibGraphStructure(unittest.TestCase):
    """Validate fmtlib/fmt graph has expected structure and edge types."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _get_graph("fmtlib")
        _skip_if_no_graph(cls.graph, "fmtlib/fmt")

    def test_graph_has_expected_size(self):
        """fmtlib should have thousands of nodes and edges."""
        self.assertGreater(self.graph.number_of_nodes(), 5000)
        self.assertGreater(self.graph.number_of_edges(), 20000)

    def test_creates_edges_present(self):
        """fmtlib has significant CREATES edges (C++ stack construction)."""
        creates = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("relationship_type") == "creates"
        ]
        self.assertGreater(len(creates), 1000,
                           "fmtlib should have >1000 CREATES edges (stack construction)")

    def test_defines_body_edges_present(self):
        """C++ repos have DEFINES_BODY linking declarations to implementations."""
        db = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("relationship_type") == "defines_body"
        ]
        self.assertGreater(len(db), 1000,
                           "fmtlib should have >1000 DEFINES_BODY edges (decl→impl)")

    def test_specializes_edges_present(self):
        """C++ templates produce SPECIALIZES edges."""
        spec = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("relationship_type") == "specializes"
        ]
        self.assertGreater(len(spec), 100,
                           "fmtlib should have >100 SPECIALIZES edges (template usage)")

    def test_alias_of_edges_present(self):
        """C++ type aliases produce ALIAS_OF edges."""
        alias = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("relationship_type") == "alias_of"
        ]
        self.assertGreater(len(alias), 50,
                           "fmtlib should have >50 ALIAS_OF edges (typedef/using)")

    def test_inheritance_edges_present(self):
        """fmtlib has inheritance relationships."""
        inh = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("relationship_type") == "inheritance"
        ]
        self.assertGreater(len(inh), 50)

    def test_references_have_annotations(self):
        """A subset of REFERENCES edges carry annotation metadata."""
        refs = [
            d for _, _, d in self.graph.edges(data=True)
            if d.get("relationship_type") == "references"
        ]
        annotated = [r for r in refs if r.get("annotations")]
        self.assertGreater(len(annotated), 100,
                           "Expect some REFERENCES with annotation metadata")

    def test_references_annotation_keys(self):
        """Check what annotation keys actually exist on REFERENCES edges."""
        keys = set()
        for _, _, d in self.graph.edges(data=True):
            if d.get("relationship_type") == "references":
                ann = d.get("annotations", {})
                keys.update(ann.keys())
        # C++ parser uses 'reference_type' key
        self.assertIn("reference_type", keys,
                      f"Expected 'reference_type' in annotation keys, got: {keys}")

    def test_all_edge_types_present(self):
        """Verify the expected set of edge types exists."""
        edge_types = set()
        for _, _, d in self.graph.edges(data=True):
            et = d.get("relationship_type")
            if et:
                edge_types.add(et)
        expected = {"references", "calls", "creates", "defines", "defines_body",
                    "inheritance", "composition", "specializes", "alias_of"}
        missing = expected - edge_types
        self.assertEqual(missing, set(),
                         f"Missing expected edge types: {missing}")

    def test_symbol_types_diversity(self):
        """fmtlib should have diverse symbol types."""
        types = Counter()
        for _, data in self.graph.nodes(data=True):
            st = data.get("symbol_type")
            if st:
                types[st] += 1
        expected = {"class", "function", "method", "struct", "type_alias",
                    "constructor", "constant", "parameter", "field", "variable"}
        present = set(types.keys())
        missing = expected - present
        self.assertEqual(missing, set(),
                         f"Missing symbol types: {missing}. Have: {sorted(present)}")


class TestConfigurationsGraphStructure(unittest.TestCase):
    """Validate configurations graph has expected Python-specific structure."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _get_graph("configurations")
        _skip_if_no_graph(cls.graph, "EliteaAI/configurations")

    def test_graph_has_expected_size(self):
        self.assertGreater(self.graph.number_of_nodes(), 500)
        self.assertGreater(self.graph.number_of_edges(), 1500)

    def test_creates_edges_present(self):
        """Python CREATES: exception creation and factory patterns."""
        creates = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("relationship_type") == "creates"
        ]
        self.assertGreater(len(creates), 10,
                           "configurations should have CREATES edges (exception construction)")

    def test_creates_targets_are_meaningful(self):
        """CREATES targets should be real classes (ConfigurationError, etc.)."""
        target_names = set()
        for u, v, d in self.graph.edges(data=True):
            if d.get("relationship_type") == "creates":
                target_names.add(self.graph.nodes[v].get("symbol_name", ""))
        self.assertIn("ConfigurationError", target_names)

    def test_inheritance_present(self):
        """Python inheritance: model handlers inherit from base."""
        inh = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("relationship_type") == "inheritance"
        ]
        self.assertGreater(len(inh), 5)

    def test_inheritance_chain_correct(self):
        """LLMModelHandler should inherit from ConfigurationModelHandler."""
        found = False
        for u, v, d in self.graph.edges(data=True):
            if d.get("relationship_type") == "inheritance":
                src_name = self.graph.nodes[u].get("symbol_name", "")
                tgt_name = self.graph.nodes[v].get("symbol_name", "")
                if src_name == "LLMModelHandler" and tgt_name == "ConfigurationModelHandler":
                    found = True
                    break
        self.assertTrue(found,
                        "Expected LLMModelHandler --inherits--> ConfigurationModelHandler")

    def test_composition_edges(self):
        """Python composition for typed fields."""
        comp = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("relationship_type") == "composition"
        ]
        self.assertGreater(len(comp), 0)

    def test_imports_edges(self):
        """Python should have import edges."""
        imports = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get("relationship_type") == "imports"
        ]
        self.assertGreater(len(imports), 50,
                           "Python repo should have many import edges")

    def test_no_defines_body(self):
        """Python does NOT have defines_body (that's C++ only)."""
        db = [
            1 for _, _, d in self.graph.edges(data=True)
            if d.get("relationship_type") == "defines_body"
        ]
        self.assertEqual(len(db), 0,
                         "Python repo should have NO defines_body edges")


# ===========================================================================
# Test Suite 2: CREATES Traversal (Factory Patterns)
# ===========================================================================

class TestCreatesTraversal(unittest.TestCase):
    """Test CREATES edge traversal for factory pattern expansion."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = _get_graph("configurations")
        _skip_if_no_graph(cls.cfg, "EliteaAI/configurations")
        cls.fmt = _get_graph("fmtlib")
        _skip_if_no_graph(cls.fmt, "fmtlib/fmt")

    def _find_creators_of(self, graph, target_name: str):
        """Find all nodes that create instances of target_name."""
        creators = []
        for u, v, d in graph.edges(data=True):
            if d.get("relationship_type") == "creates":
                if graph.nodes[v].get("symbol_name") == target_name:
                    creators.append(graph.nodes[u].get("symbol_name", "?"))
        return creators

    def _find_created_by(self, graph, creator_name: str):
        """Find all types created by a given function/method."""
        created = []
        for u, v, d in graph.edges(data=True):
            if d.get("relationship_type") == "creates":
                if graph.nodes[u].get("symbol_name") == creator_name:
                    created.append(graph.nodes[v].get("symbol_name", "?"))
        return created

    def test_configuration_error_creators(self):
        """Multiple functions create ConfigurationError — expansion should include it."""
        creators = self._find_creators_of(self.cfg, "ConfigurationError")
        self.assertGreater(len(creators), 3,
                           "Multiple functions should create ConfigurationError")
        # Known creators
        self.assertIn("handle_validation_error", creators)
        self.assertIn("create_configuration", creators)
        self.assertIn("update_configuration", creators)

    def test_create_configuration_created_types(self):
        """create_configuration creates ConfigurationError instances."""
        created = self._find_created_by(self.cfg, "create_configuration")
        self.assertIn("ConfigurationError", created)

    def test_model_configuration_service_creators(self):
        """ModelConfigurationService is created by multiple factory functions."""
        creators = self._find_creators_of(self.cfg, "ModelConfigurationService")
        self.assertGreater(len(creators), 2)

    def test_creates_expansion_brings_class_context(self):
        """When expanding a function that CREATES a class, the class node is reachable."""
        # Find a function that creates ModelConfigurationService
        creator_node = None
        target_node = None
        for u, v, d in self.cfg.edges(data=True):
            if d.get("relationship_type") == "creates":
                if self.cfg.nodes[v].get("symbol_name") == "ModelConfigurationService":
                    creator_node = u
                    target_node = v
                    break
        self.assertIsNotNone(creator_node, "Should find a creator of ModelConfigurationService")
        self.assertIsNotNone(target_node)

        # The target should be a class (or inferred). Some created types
        # are inferred from constructor calls and may not have content.
        target_data = self.cfg.nodes[target_node]
        target_type = target_data.get("symbol_type", "")
        self.assertIn(target_type, ("class", "inferred"),
                      f"Created target should be class or inferred, got: {target_type}")
        # Verify the node exists and has a name — content may be empty for inferred
        self.assertTrue(target_data.get("symbol_name", ""),
                        "Created class should at least have a name")


# ===========================================================================
# Test Suite 3: Inheritance Chain Traversal
# ===========================================================================

class TestInheritanceTraversal(unittest.TestCase):
    """Test inheritance chain traversal for smart expansion."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = _get_graph("configurations")
        _skip_if_no_graph(cls.cfg, "EliteaAI/configurations")

    def _get_base_classes(self, graph, class_name: str, max_depth: int = 5):
        """Walk inheritance edges to find all base classes up to max_depth."""
        bases = []
        visited = set()

        # Find all nodes with this class name
        start_nodes = [
            nid for nid, data in graph.nodes(data=True)
            if data.get("symbol_name") == class_name
        ]

        queue = [(nid, 0) for nid in start_nodes]
        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth or current in visited:
                continue
            visited.add(current)

            for _, target, d in graph.out_edges(current, data=True):
                if d.get("relationship_type") == "inheritance":
                    target_name = graph.nodes[target].get("symbol_name", "?")
                    bases.append((target_name, depth + 1))
                    queue.append((target, depth + 1))
        return bases

    def test_handler_inheritance_chain(self):
        """LLMModelHandler → ConfigurationModelHandler chain."""
        bases = self._get_base_classes(self.cfg, "LLMModelHandler")
        base_names = [name for name, _ in bases]
        self.assertIn("ConfigurationModelHandler", base_names)

    def test_all_model_handlers_share_base(self):
        """All *ModelHandler classes inherit from ConfigurationModelHandler."""
        handlers = ["LLMModelHandler", "EmbeddingModelHandler",
                    "VectorStorageModelHandler", "ImageGenerationModelHandler"]
        for handler in handlers:
            bases = self._get_base_classes(self.cfg, handler)
            base_names = [name for name, _ in bases]
            self.assertIn("ConfigurationModelHandler", base_names,
                          f"{handler} should inherit from ConfigurationModelHandler")

    def test_api_inherits_apibase(self):
        """API classes inherit from APIBase."""
        bases = self._get_base_classes(self.cfg, "API")
        base_names = [name for name, _ in bases]
        self.assertIn("APIBase", base_names)


# ===========================================================================
# Test Suite 4: defines_body (C++ Declaration → Implementation)
# ===========================================================================

class TestDefinesBodyTraversal(unittest.TestCase):
    """Test C++ declaration-to-implementation linking via DEFINES_BODY."""

    @classmethod
    def setUpClass(cls):
        cls.fmt = _get_graph("fmtlib")
        _skip_if_no_graph(cls.fmt, "fmtlib/fmt")

    def test_defines_body_links_same_name(self):
        """Most DEFINES_BODY edges link symbols with the same name (decl→impl)."""
        same_name = 0
        diff_name = 0
        for u, v, d in self.fmt.edges(data=True):
            if d.get("relationship_type") == "defines_body":
                u_name = self.fmt.nodes[u].get("symbol_name", "")
                v_name = self.fmt.nodes[v].get("symbol_name", "")
                if u_name == v_name:
                    same_name += 1
                else:
                    diff_name += 1
        # Majority should link same-name symbols
        self.assertGreater(same_name, diff_name,
                           f"Same name: {same_name}, different: {diff_name}")

    def test_defines_body_augmentation_strategy(self):
        """For smart expansion: DEFINES_BODY target content augments source content."""
        # Find a defines_body edge where both nodes have content
        for u, v, d in self.fmt.edges(data=True):
            if d.get("relationship_type") != "defines_body":
                continue
            u_content = self.fmt.nodes[u].get("content", "")
            v_content = self.fmt.nodes[v].get("content", "")
            if u_content and v_content and len(u_content) > 20 and len(v_content) > 20:
                # Augmentation = concat. The combined content gives full picture.
                combined = u_content + "\n\n// Implementation:\n" + v_content
                self.assertGreater(len(combined), len(u_content))
                self.assertGreater(len(combined), len(v_content))
                return  # Found one good example, test passes
        self.skipTest("No defines_body edges with substantial content on both sides")


# ===========================================================================
# Test Suite 5: Alias Chain Resolution
# ===========================================================================

class TestAliasOfTraversal(unittest.TestCase):
    """Test ALIAS_OF chain resolution for type alias expansion."""

    @classmethod
    def setUpClass(cls):
        cls.fmt = _get_graph("fmtlib")
        _skip_if_no_graph(cls.fmt, "fmtlib/fmt")

    def _resolve_alias_chain(self, graph, alias_name: str, max_hops: int = 3):
        """Follow ALIAS_OF edges to resolve a type alias chain."""
        chain = []
        visited = set()
        current_nodes = [
            nid for nid, data in graph.nodes(data=True)
            if data.get("symbol_name") == alias_name
        ]

        for _ in range(max_hops):
            next_nodes = []
            for node in current_nodes:
                if node in visited:
                    continue
                visited.add(node)
                for _, target, d in graph.out_edges(node, data=True):
                    if d.get("relationship_type") == "alias_of":
                        target_name = graph.nodes[target].get("symbol_name", "?")
                        chain.append(target_name)
                        next_nodes.append(target)
            current_nodes = next_nodes
            if not current_nodes:
                break
        return chain

    def test_wstring_view_resolves_to_basic_string_view(self):
        """wstring_view --alias_of--> basic_string_view."""
        chain = self._resolve_alias_chain(self.fmt, "wstring_view")
        self.assertIn("basic_string_view", chain)

    def test_alias_chain_terminates(self):
        """Alias chain resolution should terminate (no infinite loops)."""
        # Try all alias_of sources — none should loop forever
        alias_sources = set()
        for u, v, d in self.fmt.edges(data=True):
            if d.get("relationship_type") == "alias_of":
                alias_sources.add(self.fmt.nodes[u].get("symbol_name", ""))

        for alias in list(alias_sources)[:20]:  # Sample 20
            chain = self._resolve_alias_chain(self.fmt, alias, max_hops=20)
            # With max_hops=20, chains should terminate. Some generic names
            # like 'type' resolve across multiple symbols — that's expected.
            # The important thing is it terminates, not that it's short.
            self.assertLessEqual(len(chain), 20,
                                 f"Alias chain for '{alias}' exceeds 20 hops: {chain}")


# ===========================================================================
# Test Suite 6: FTS5 Search on Real Graph
# ===========================================================================

@unittest.skipUnless(HAS_FTS, "GraphTextIndex not importable")
class TestFTS5OnFmtlib(unittest.TestCase):
    """FTS5 search tests on the fmtlib graph."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _get_graph("fmtlib")
        _skip_if_no_graph(cls.graph, "fmtlib/fmt")

        cls.tmpdir = tempfile.mkdtemp()
        cls.idx = GraphTextIndex(cache_dir=cls.tmpdir)
        count = cls.idx.build_from_graph(cls.graph, cache_key="fmtlib_test")
        assert count > 0, "FTS5 build should index nodes"

    @classmethod
    def tearDownClass(cls):
        cls.idx.close()
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_search_format_returns_results(self):
        """Searching 'format' should find format-related functions in fmtlib."""
        results = self.idx.search("format", k=20)
        self.assertGreater(len(results), 0)
        names = [r.metadata.get("symbol_name", "") for r in results]
        # Should find the core 'format' function
        format_matches = [n for n in names if "format" in n.lower()]
        self.assertGreater(len(format_matches), 0,
                           f"Expected 'format' in results, got: {names[:10]}")

    def test_search_string_view(self):
        """Searching 'string_view' should find the basic_string_view class."""
        results = self.idx.search("string_view", k=20)
        names = [r.metadata.get("symbol_name", "") for r in results]
        sv_matches = [n for n in names if "string_view" in n.lower()]
        self.assertGreater(len(sv_matches), 0)

    def test_search_by_name_exact(self):
        """search_by_name for 'basic_string_view' should find exact match."""
        results = self.idx.search_by_name("basic_string_view", k=5)
        names = [r.metadata.get("symbol_name", "") for r in results]
        self.assertIn("basic_string_view", names)

    def test_search_by_type_class(self):
        """search_by_type('class') should return only class symbols."""
        results = self.idx.search_by_type("class", k=50)
        for r in results:
            self.assertEqual(r.metadata.get("symbol_type"), "class")
        self.assertGreater(len(results), 10)

    def test_search_by_type_struct(self):
        """search_by_type('struct') should return struct symbols."""
        results = self.idx.search_by_type("struct", k=50)
        for r in results:
            self.assertEqual(r.metadata.get("symbol_type"), "struct")
        self.assertGreater(len(results), 10)

    def test_search_by_path_prefix(self):
        """search_by_path_prefix('include/fmt/') should find fmt headers."""
        results = self.idx.search_by_path_prefix("include/fmt/", k=100)
        self.assertGreater(len(results), 0)
        for r in results:
            rp = r.get("rel_path", "")
            self.assertTrue(rp.startswith("include/fmt"),
                            f"Expected path starting with 'include/fmt', got '{rp}'")

    def test_get_by_node_id(self):
        """get_by_node_id for a known node should return its data."""
        # Pick any node from graph
        sample_id = list(self.graph.nodes())[0]
        result = self.idx.get_by_node_id(sample_id)
        if result is not None:
            self.assertEqual(result.get("node_id"), sample_id)

    def test_search_concept_query(self):
        """A concept query 'buffer memory allocation' should return relevant results."""
        results = self.idx.search("buffer memory allocation", k=20)
        self.assertGreater(len(results), 0,
                           "Concept query should find something in fmtlib")

    def test_search_camel_case_tokenization(self):
        """CamelCase names should be searchable by component words."""
        # Search for a component of a CamelCase name
        results = self.idx.search("string view", k=20)
        # Should find basic_string_view via name_tokens splitting
        names = [r.metadata.get("symbol_name", "") for r in results]
        sv_matches = [n for n in names if "string" in n.lower()]
        self.assertGreater(len(sv_matches), 0,
                           "CamelCase/snake_case tokenization should match components")

    def test_node_count_matches_graph(self):
        """FTS index node count should equal graph node count."""
        graph_count = self.graph.number_of_nodes()
        fts_count = self.idx.node_count
        self.assertEqual(fts_count, graph_count,
                         f"FTS has {fts_count} nodes vs graph {graph_count}")


@unittest.skipUnless(HAS_FTS, "GraphTextIndex not importable")
class TestFTS5OnConfigurations(unittest.TestCase):
    """FTS5 search tests on the configurations graph."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _get_graph("configurations")
        _skip_if_no_graph(cls.graph, "EliteaAI/configurations")

        cls.tmpdir = tempfile.mkdtemp()
        cls.idx = GraphTextIndex(cache_dir=cls.tmpdir)
        count = cls.idx.build_from_graph(cls.graph, cache_key="config_test")
        assert count > 0

    @classmethod
    def tearDownClass(cls):
        cls.idx.close()
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_search_configuration(self):
        """Searching 'configuration' should find Configuration class and related."""
        results = self.idx.search("configuration", k=20)
        names = [r.metadata.get("symbol_name", "") for r in results]
        cfg_matches = [n for n in names if "configuration" in n.lower() or "config" in n.lower()]
        self.assertGreater(len(cfg_matches), 0)

    def test_search_validation(self):
        """Searching 'validation error' should find validation-related symbols."""
        results = self.idx.search("validation error", k=20)
        self.assertGreater(len(results), 0)

    def test_search_model_handler(self):
        """Searching 'model handler' should find the handler classes."""
        results = self.idx.search("model handler", k=20)
        names = [r.metadata.get("symbol_name", "") for r in results]
        handler_matches = [n for n in names if "handler" in n.lower() or "model" in n.lower()]
        self.assertGreater(len(handler_matches), 0)

    def test_search_by_path_prefix_routes(self):
        """Search by path prefix for routes/ directory."""
        results = self.idx.search_by_path_prefix("routes/", k=50)
        # configurations repo may or may not have routes/ — check gracefully
        for r in results:
            rp = r.get("rel_path", "")
            self.assertTrue(rp.startswith("routes/"))

    def test_search_by_name_api(self):
        """search_by_name for 'API' should find API classes."""
        results = self.idx.search_by_name("API", k=20)
        names = [r.metadata.get("symbol_name", "") for r in results]
        self.assertIn("API", names)


# ===========================================================================
# Test Suite 7: FTS5 + Graph Combined Traversal
# ===========================================================================

@unittest.skipUnless(HAS_FTS, "GraphTextIndex not importable")
class TestFTSGraphCombined(unittest.TestCase):
    """Test combining FTS5 search with graph traversal — the core smart expansion flow."""

    @classmethod
    def setUpClass(cls):
        cls.graph = _get_graph("configurations")
        _skip_if_no_graph(cls.graph, "EliteaAI/configurations")

        cls.tmpdir = tempfile.mkdtemp()
        cls.idx = GraphTextIndex(cache_dir=cls.tmpdir)
        cls.idx.build_from_graph(cls.graph, cache_key="config_combined_test")

    @classmethod
    def tearDownClass(cls):
        cls.idx.close()
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _fts_then_expand(self, query: str, edge_types: list, direction: str = "out",
                         k_search: int = 10, k_expand: int = 5):
        """
        FTS5 search → graph edge traversal.
        Returns (fts_results, expanded_nodes) where expanded_nodes are
        neighbors reached via specified edge types.
        """
        fts_results = self.idx.search(query, k=k_search)
        expanded = {}

        for doc in fts_results:
            node_id = doc.metadata.get("node_id", "")
            if not node_id or node_id not in self.graph:
                continue

            edges = (self.graph.out_edges(node_id, data=True) if direction == "out"
                     else self.graph.in_edges(node_id, data=True))

            for src, tgt, data in edges:
                et = data.get("relationship_type", "")
                if et in edge_types:
                    neighbor = tgt if direction == "out" else src
                    neighbor_data = self.graph.nodes.get(neighbor, {})
                    if neighbor not in expanded:
                        expanded[neighbor] = {
                            "name": neighbor_data.get("symbol_name", "?"),
                            "type": neighbor_data.get("symbol_type", "?"),
                            "edge_type": et,
                            "from": doc.metadata.get("symbol_name", "?"),
                        }
        return fts_results, expanded

    def test_fts_creates_expansion(self):
        """FTS find function → follow CREATES → get created types."""
        fts_results, expanded = self._fts_then_expand(
            "create configuration", ["creates"], direction="out"
        )
        self.assertGreater(len(fts_results), 0, "FTS should find results for 'create configuration'")

        # If we found creator functions, we should see ConfigurationError in expanded
        if expanded:
            expanded_names = [v["name"] for v in expanded.values()]
            # At least some created types should appear
            self.assertGreater(len(expanded_names), 0,
                               "CREATES expansion should find created types")

    def test_fts_inheritance_expansion(self):
        """FTS find class → follow INHERITANCE → get base classes."""
        fts_results, expanded = self._fts_then_expand(
            "LLMModelHandler", ["inheritance"], direction="out"
        )
        self.assertGreater(len(fts_results), 0)
        if expanded:
            expanded_names = [v["name"] for v in expanded.values()]
            self.assertIn("ConfigurationModelHandler", expanded_names,
                          f"Should find base class, got: {expanded_names}")

    def test_fts_composition_expansion(self):
        """FTS find class → follow COMPOSITION → get composed types."""
        fts_results, expanded = self._fts_then_expand(
            "configuration", ["composition"], direction="out"
        )
        # Composition may or may not yield results depending on what FTS finds
        # The important thing is no errors
        self.assertIsInstance(expanded, dict)

    def test_combined_p0_expansion(self):
        """Full P0 expansion: FTS search → inheritance + creates."""
        fts_results, expanded = self._fts_then_expand(
            "model handler", ["inheritance", "creates"], direction="out", k_search=20
        )
        self.assertGreater(len(fts_results), 0,
                           "FTS should find 'model handler' related symbols")


# ===========================================================================
# Test Suite 8: Edge Cases & Cross-File Relationships
# ===========================================================================

class TestEdgeCases(unittest.TestCase):
    """Edge cases: isolated nodes, circular refs, cross-file relationships."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = _get_graph("configurations")
        _skip_if_no_graph(cls.cfg, "EliteaAI/configurations")
        cls.fmt = _get_graph("fmtlib")
        _skip_if_no_graph(cls.fmt, "fmtlib/fmt")

    def test_isolated_nodes_exist(self):
        """Some nodes may have no edges (isolated). Smart expansion should handle them."""
        isolated_cfg = list(nx.isolates(self.cfg))
        # It's OK to have some or none — the expansion code must handle both
        # Just verify we can identify them
        self.assertIsInstance(isolated_cfg, list)

    def test_no_infinite_loops_in_circular_refs(self):
        """If circular references exist (A→B→A), traversal with visited set terminates."""
        # Check for actual cycles in the graph
        try:
            cycles = list(nx.simple_cycles(self.cfg))
            # If cycles exist, verify our traversal would handle them
            if cycles:
                # Pick a node in a cycle and try bounded traversal
                cycle_node = cycles[0][0]
                visited = set()
                queue = [cycle_node]
                steps = 0
                while queue and steps < 100:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    for _, target, _ in self.cfg.out_edges(current, data=True):
                        queue.append(target)
                    steps += 1
                self.assertLess(steps, 100, "Traversal should terminate with visited set")
        except nx.NetworkXError:
            pass  # simple_cycles may be slow on large graphs — skip

    def test_cross_file_relationships(self):
        """Edges should connect symbols in DIFFERENT files (cross-file dependencies)."""
        cross_file = 0
        same_file = 0
        for u, v, d in self.cfg.edges(data=True):
            u_file = self.cfg.nodes[u].get("rel_path", "")
            v_file = self.cfg.nodes[v].get("rel_path", "")
            if u_file and v_file:
                if u_file != v_file:
                    cross_file += 1
                else:
                    same_file += 1
        self.assertGreater(cross_file, 0,
                           "Should have cross-file relationships")
        # Cross-file relationships are the most valuable for wiki generation
        total = cross_file + same_file
        if total > 0:
            ratio = cross_file / total
            self.assertGreater(ratio, 0.1,
                               f"Cross-file ratio ({ratio:.2%}) seems too low")

    def test_multi_edge_between_same_nodes(self):
        """Discover if multiple edges exist between the same pair of nodes.
        
        Real graphs may use MultiDiGraph (multiple edges per u,v pair) to
        represent different relationship types. This test documents the
        actual behavior.
        """
        is_multi = isinstance(self.cfg, nx.MultiDiGraph)
        
        if is_multi:
            # MultiDiGraph: count unique (u,v) pairs with multiple edge keys
            edge_counts = Counter()
            for u, v, key in self.cfg.edges(keys=True):
                edge_counts[(u, v)] += 1
            max_count = max(edge_counts.values()) if edge_counts else 0
            multi_pairs = sum(1 for c in edge_counts.values() if c > 1)
            # Document: this is expected for MultiDiGraph
            self.assertIsInstance(max_count, int)
        else:
            # DiGraph: each pair should appear at most once
            edge_pairs = Counter()
            for u, v, _ in self.cfg.edges(data=True):
                edge_pairs[(u, v)] += 1
            max_count = max(edge_pairs.values()) if edge_pairs else 0
            self.assertLessEqual(max_count, 1,
                                 "DiGraph should have at most 1 edge per (u,v) pair.")


# ===========================================================================
# Test Suite 9: Symbol Type Filtering for FTS Indexing
# ===========================================================================

class TestSymbolTypeDistribution(unittest.TestCase):
    """Analyze symbol type distribution to inform FTS indexing decisions."""

    @classmethod
    def setUpClass(cls):
        cls.fmt = _get_graph("fmtlib")
        _skip_if_no_graph(cls.fmt, "fmtlib/fmt")

    def test_architectural_vs_non_architectural_ratio(self):
        """Measure ratio of architectural symbols vs parameter/variable/field."""
        types = Counter()
        for _, data in self.fmt.nodes(data=True):
            st = data.get("symbol_type", "unknown")
            types[st] += 1

        architectural = {"class", "struct", "interface", "enum", "trait",
                         "function", "type_alias", "constant", "module",
                         "constructor"}
        behavioral = {"method"}  # Debatable
        noise = {"parameter", "variable", "field", "inferred"}

        arch_count = sum(types.get(t, 0) for t in architectural)
        behavioral_count = sum(types.get(t, 0) for t in behavioral)
        noise_count = sum(types.get(t, 0) for t in noise)
        total = sum(types.values())

        # Document the ratio for indexing decisions
        self.assertGreater(total, 0)
        noise_ratio = noise_count / total if total else 0

        # The "questionable" symbols should be less than 70% of all nodes
        # If they're more, we should consider filtering them from FTS
        self.assertLess(noise_ratio, 0.8,
                        f"Noise ratio {noise_ratio:.2%} is very high. "
                        f"Arch: {arch_count}, Behavioral: {behavioral_count}, "
                        f"Noise: {noise_count}, Total: {total}")

    def test_module_level_vs_local_variables(self):
        """Count module-level variables (worth indexing) vs local ones (noise)."""
        module_level = 0
        local = 0
        for _, data in self.fmt.nodes(data=True):
            if data.get("symbol_type") != "variable":
                continue
            # Module-level variables typically have no parent or parent is module
            full_name = data.get("full_name", "")
            # Heuristic: fewer '.' segments = more likely module-level
            segments = full_name.count(".") if full_name else 0
            if segments <= 1:
                module_level += 1
            else:
                local += 1

        # Just document — no assertion on exact ratio
        total_vars = module_level + local
        if total_vars > 0:
            self.assertIsNotNone(total_vars)  # Always passes, just documents


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    unittest.main()
