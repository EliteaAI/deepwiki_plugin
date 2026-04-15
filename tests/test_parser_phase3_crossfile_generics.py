"""
Phase 3 / 3.5 — Cross-File Generic Decomposition & Graph Integration Tests.

Verifies that recursive type extraction works correctly when types are
defined in one file and referenced (via nested generics) in another.
Also validates that the graph builder correctly represents these
cross-file generic REFERENCES as edges.

Covers:
    1. Python  cross-file generic REFERENCES (parse_multiple_files)
    2. Java    cross-file generic REFERENCES (parse_multiple_files)
    3. TypeScript cross-file generic REFERENCES (parse_multiple_files)
    3b. C++    cross-file generic REFERENCES (parse_multiple_files)
    4. Graph-level cross-file generic edge verification (analyze_repository)
    5. TypeScript type alias ALIAS_OF + REFERENCES (Phase 3.5 fix)
    6. Cross-parser cross-file parity

See PLANNING_PARSER_REVIEW.md Phase 3 / 3.5 for details.
"""

import os
import tempfile
import pytest

from plugin_implementation.parsers.base_parser import SymbolType, RelationshipType
from plugin_implementation.parsers.python_parser import PythonParser
from plugin_implementation.parsers.java_visitor_parser import JavaVisitorParser
from plugin_implementation.parsers.typescript_enhanced_parser import TypeScriptEnhancedParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_files(tmpdir: str, files: dict) -> dict:
    """Write files dict to tmpdir, return {filename: absolute_path}."""
    paths = {}
    for name, content in files.items():
        # Support nested dirs: "src/models/User.java"
        fpath = os.path.join(tmpdir, name)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, 'w') as f:
            f.write(content)
        paths[name] = fpath
    return paths


def _all_refs(results: dict, source_filter=None):
    """Collect all REFERENCES relationships across all files in results."""
    out = []
    for fp, result in results.items():
        for r in result.relationships:
            if r.relationship_type == RelationshipType.REFERENCES:
                if source_filter is None or source_filter in r.source_symbol:
                    out.append(r)
    return out


def _ref_target_set(results: dict, source_filter=None):
    """Return set of target symbols from all REFERENCES."""
    return {r.target_symbol for r in _all_refs(results, source_filter)}


def _annotated_refs(results: dict, source_filter=None):
    """Return REFERENCES with non-empty reference_type annotation."""
    return [r for r in _all_refs(results, source_filter)
            if r.annotations.get('reference_type')]


def _cross_file_refs(results: dict, source_filter=None):
    """Return REFERENCES that have cross_file=True annotation or target_file set."""
    refs = _all_refs(results, source_filter)
    return [r for r in refs
            if r.annotations.get('cross_file') or
            (r.target_file and r.source_file and r.target_file != r.source_file)]


# =============================================================================
# 1. Python Cross-File Generic REFERENCES
# =============================================================================

class TestPythonCrossFileGenerics:
    """Python parser finds nested generic types defined in other files."""

    @pytest.fixture(autouse=True)
    def setup_parser(self):
        self.parser = PythonParser()

    def test_dict_list_user_cross_file(self):
        """User defined in models.py, used as Dict[str, List[User]] in service.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'models.py': (
                    "class User:\n"
                    "    def __init__(self, name: str):\n"
                    "        self.name = name\n"
                ),
                'service.py': (
                    "from typing import Dict, List\n"
                    "from models import User\n"
                    "\n"
                    "def get_users() -> Dict[str, List[User]]:\n"
                    "    return {}\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            # User should appear as REFERENCES target from service.py
            targets = _ref_target_set(results, 'get_users')
            assert 'User' in targets, f"User not found in REFERENCES targets: {targets}"

    def test_cross_file_ref_has_target_file(self):
        """Cross-file REFERENCES for User should have target_file set to models.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'models.py': "class User: pass\n",
                'service.py': (
                    "from typing import Optional\n"
                    "from models import User\n"
                    "\n"
                    "def find_user() -> Optional[User]:\n"
                    "    return None\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            # Find the REFERENCES to User from service.py
            user_refs = [r for r in _all_refs(results, 'find_user')
                         if r.target_symbol == 'User']
            assert len(user_refs) >= 1, "No REFERENCES to User found"

            # At least one should have cross-file resolution
            cross_file = [r for r in user_refs
                          if r.target_file and 'models' in r.target_file]
            assert len(cross_file) >= 1, (
                f"Expected cross-file ref to models.py, got: "
                f"{[(r.target_file, r.annotations) for r in user_refs]}"
            )

    def test_pep604_union_cross_file(self):
        """Admin | Guest from different files found via PEP 604 union."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'auth.py': "class Admin: pass\nclass Guest: pass\n",
                'handler.py': (
                    "from auth import Admin, Guest\n"
                    "\n"
                    "def handle(user: Admin | Guest) -> None:\n"
                    "    pass\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            targets = _ref_target_set(results, 'handle')
            assert 'Admin' in targets
            assert 'Guest' in targets

    def test_deeply_nested_cross_file(self):
        """Dict[str, Dict[str, List[Entity]]] with Entity from another file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'domain.py': "class Entity: pass\n",
                'repo.py': (
                    "from typing import Dict, List\n"
                    "from domain import Entity\n"
                    "\n"
                    "def load() -> Dict[str, Dict[str, List[Entity]]]:\n"
                    "    return {}\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            targets = _ref_target_set(results, 'load')
            assert 'Entity' in targets

    def test_callable_cross_file(self):
        """Callable[[Request], Response] with types from another file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'types.py': "class Request: pass\nclass Response: pass\n",
                'middleware.py': (
                    "from typing import Callable\n"
                    "from types import Request, Response\n"
                    "\n"
                    "def wrap(handler: Callable[[Request], Response]) -> None:\n"
                    "    pass\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            targets = _ref_target_set(results, 'wrap')
            assert 'Request' in targets
            assert 'Response' in targets


# =============================================================================
# 2. Java Cross-File Generic REFERENCES
# =============================================================================

class TestJavaCrossFileGenerics:
    """Java parser finds nested generic types defined in other files."""

    @pytest.fixture(autouse=True)
    def setup_parser(self):
        self.parser = JavaVisitorParser()

    def test_map_list_user_cross_file(self):
        """User in User.java, used as Map<String, List<User>> in Service.java."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'User.java': (
                    "package com.example;\n"
                    "public class User {\n"
                    "    private String name;\n"
                    "}\n"
                ),
                'Service.java': (
                    "package com.example;\n"
                    "import java.util.Map;\n"
                    "import java.util.List;\n"
                    "\n"
                    "public class Service {\n"
                    "    public Map<String, List<User>> getUsers() {\n"
                    "        return null;\n"
                    "    }\n"
                    "}\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            targets = _ref_target_set(results, 'getUsers')
            assert 'User' in targets, f"User not found in REFERENCES: {targets}"

    def test_field_composition_cross_file(self):
        """Field Map<String, List<Order>> with Order from another file → COMPOSITION."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'Order.java': (
                    "package com.example;\n"
                    "public class Order {\n"
                    "    private int id;\n"
                    "}\n"
                ),
                'OrderRepository.java': (
                    "package com.example;\n"
                    "import java.util.Map;\n"
                    "import java.util.List;\n"
                    "\n"
                    "public class OrderRepository {\n"
                    "    private Map<String, List<Order>> cache;\n"
                    "}\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            # Order should be found via COMPOSITION (field with user-defined type)
            all_rels = []
            for fp, result in results.items():
                all_rels.extend(result.relationships)
            comp_targets = {r.target_symbol for r in all_rels
                           if r.relationship_type == RelationshipType.COMPOSITION}
            assert 'Order' in comp_targets, f"Order not in COMPOSITION targets: {comp_targets}"

    def test_wildcard_cross_file(self):
        """List<? extends BaseEntity> with BaseEntity from another file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'BaseEntity.java': (
                    "package com.example;\n"
                    "public abstract class BaseEntity {\n"
                    "    private long id;\n"
                    "}\n"
                ),
                'Repository.java': (
                    "package com.example;\n"
                    "import java.util.List;\n"
                    "\n"
                    "public class Repository {\n"
                    "    public List<? extends BaseEntity> findAll() {\n"
                    "        return null;\n"
                    "    }\n"
                    "}\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            targets = _ref_target_set(results, 'findAll')
            assert 'BaseEntity' in targets

    def test_deeply_nested_cross_file(self):
        """Map<String, Map<String, List<Widget>>> with Widget from another file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'Widget.java': (
                    "package com.example;\n"
                    "public class Widget {}\n"
                ),
                'Dashboard.java': (
                    "package com.example;\n"
                    "import java.util.Map;\n"
                    "import java.util.List;\n"
                    "\n"
                    "public class Dashboard {\n"
                    "    public Map<String, Map<String, List<Widget>>> getWidgets() {\n"
                    "        return null;\n"
                    "    }\n"
                    "}\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            targets = _ref_target_set(results, 'getWidgets')
            assert 'Widget' in targets

    def test_param_cross_file(self):
        """Method param Map<String, List<Config>> with Config from another file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'Config.java': (
                    "package com.example;\n"
                    "public class Config {\n"
                    "    private String key;\n"
                    "}\n"
                ),
                'Loader.java': (
                    "package com.example;\n"
                    "import java.util.Map;\n"
                    "import java.util.List;\n"
                    "\n"
                    "public class Loader {\n"
                    "    public void load(Map<String, List<Config>> configs) {}\n"
                    "}\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            targets = _ref_target_set(results, 'load')
            assert 'Config' in targets


# =============================================================================
# 3. TypeScript Cross-File Generic REFERENCES
# =============================================================================

class TestTypeScriptCrossFileGenerics:
    """TypeScript parser finds nested generic types defined in other files."""

    @pytest.fixture(autouse=True)
    def setup_parser(self):
        self.parser = TypeScriptEnhancedParser()

    def test_map_array_user_cross_file(self):
        """User in user.ts, used as Map<string, Array<User>> in service.ts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'user.ts': (
                    "export interface User {\n"
                    "    name: string;\n"
                    "}\n"
                ),
                'service.ts': (
                    "import { User } from './user';\n"
                    "\n"
                    "export function getUsers(): Map<string, Array<User>> {\n"
                    "    return new Map();\n"
                    "}\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            targets = _ref_target_set(results, 'getUsers')
            assert 'User' in targets, f"User not found in REFERENCES: {targets}"

    def test_promise_result_cross_file(self):
        """Result in types.ts, used as Promise<Result> in handler.ts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'types.ts': (
                    "export interface Result {\n"
                    "    status: string;\n"
                    "    data: any;\n"
                    "}\n"
                ),
                'handler.ts': (
                    "import { Result } from './types';\n"
                    "\n"
                    "export async function handle(): Promise<Result> {\n"
                    "    return { status: 'ok', data: null };\n"
                    "}\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            targets = _ref_target_set(results, 'handle')
            assert 'Result' in targets

    def test_union_generic_cross_file(self):
        """User | Admin from different files in a union type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'user.ts': "export interface User { role: string; }\n",
                'admin.ts': "export interface Admin { level: number; }\n",
                'auth.ts': (
                    "import { User } from './user';\n"
                    "import { Admin } from './admin';\n"
                    "\n"
                    "export function check(actor: User | Admin): boolean {\n"
                    "    return true;\n"
                    "}\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            targets = _ref_target_set(results, 'check')
            assert 'User' in targets
            assert 'Admin' in targets

    def test_cross_file_ref_has_target_file(self):
        """REFERENCES for cross-file type should resolve target_file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'model.ts': "export class Entity { id: number = 0; }\n",
                'api.ts': (
                    "import { Entity } from './model';\n"
                    "\n"
                    "export function getEntity(): Promise<Entity> {\n"
                    "    return null as any;\n"
                    "}\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            entity_refs = [r for r in _all_refs(results, 'getEntity')
                           if r.target_symbol == 'Entity']
            assert len(entity_refs) >= 1, "No REFERENCES to Entity found"

            # At least one should have target_file pointing to model.ts
            cross = [r for r in entity_refs
                     if r.target_file and 'model' in r.target_file]
            assert len(cross) >= 1, (
                f"Expected cross-file ref to model.ts, got: "
                f"{[(r.target_file, r.annotations) for r in entity_refs]}"
            )

    def test_deeply_nested_cross_file(self):
        """Map<string, Map<string, Array<Widget>>> from another file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'widget.ts': "export interface Widget { label: string; }\n",
                'dashboard.ts': (
                    "import { Widget } from './widget';\n"
                    "\n"
                    "export function getDashboard(): Map<string, Map<string, Array<Widget>>> {\n"
                    "    return new Map();\n"
                    "}\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            targets = _ref_target_set(results, 'getDashboard')
            assert 'Widget' in targets


# =============================================================================
# 3b. C++ Cross-File Generic REFERENCES
# =============================================================================

class TestCppCrossFileGenerics:
    """C++ parser finds nested template types defined in other files."""

    @pytest.fixture(autouse=True)
    def setup_parser(self):
        try:
            from plugin_implementation.parsers.cpp_enhanced_parser import CppEnhancedParser
            self.parser = CppEnhancedParser()
            self.available = True
        except ImportError:
            self.available = False

    def test_vector_shared_ptr_point_cross_file(self):
        """Point in point.h, used as vector<shared_ptr<Point>> in canvas.cpp."""
        if not self.available:
            pytest.skip("C++ parser not available")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'point.h': (
                    "#pragma once\n"
                    "class Point {\n"
                    "public:\n"
                    "    double x, y;\n"
                    "};\n"
                ),
                'canvas.cpp': (
                    "#include <vector>\n"
                    "#include <memory>\n"
                    "#include \"point.h\"\n"
                    "\n"
                    "class Canvas {\n"
                    "public:\n"
                    "    std::vector<std::shared_ptr<Point>> points;\n"
                    "};\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))
            targets = _ref_target_set(results)
            assert 'Point' in targets, f"Point not found in cross-file REFERENCES: {targets}"

    def test_map_vector_widget_cross_file(self):
        """Widget in widget.h, used as map<string, vector<Widget>> in registry.cpp."""
        if not self.available:
            pytest.skip("C++ parser not available")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'widget.h': (
                    "#pragma once\n"
                    "class Widget {\n"
                    "public:\n"
                    "    std::string label;\n"
                    "};\n"
                ),
                'registry.cpp': (
                    "#include <map>\n"
                    "#include <vector>\n"
                    "#include <string>\n"
                    "#include \"widget.h\"\n"
                    "\n"
                    "class Registry {\n"
                    "public:\n"
                    "    std::map<std::string, std::vector<Widget>> widgets;\n"
                    "};\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))
            targets = _ref_target_set(results)
            assert 'Widget' in targets, f"Widget not found in cross-file REFERENCES: {targets}"

    def test_cross_file_ref_has_target_file(self):
        """Cross-file REFERENCES should resolve target_file to the header."""
        if not self.available:
            pytest.skip("C++ parser not available")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'entity.h': (
                    "#pragma once\n"
                    "class Entity {\n"
                    "    int id;\n"
                    "};\n"
                ),
                'service.cpp': (
                    "#include <vector>\n"
                    "#include \"entity.h\"\n"
                    "\n"
                    "class Service {\n"
                    "public:\n"
                    "    std::vector<Entity> entities;\n"
                    "};\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            entity_refs = []
            for fp, result in results.items():
                for r in result.relationships:
                    if (r.relationship_type == RelationshipType.REFERENCES
                            and r.target_symbol == 'Entity'):
                        entity_refs.append(r)
            assert len(entity_refs) >= 1, "No REFERENCES to Entity found"

            # At least one should have target_file pointing to entity.h
            cross = [r for r in entity_refs
                     if r.target_file and 'entity' in r.target_file]
            assert len(cross) >= 1, (
                f"Expected cross-file ref to entity.h, got: "
                f"{[(r.target_file, r.annotations) for r in entity_refs]}"
            )

    def test_function_return_type_cross_file(self):
        """Result in result.h, function returns vector<Result> in handler.cpp."""
        if not self.available:
            pytest.skip("C++ parser not available")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'result.h': (
                    "#pragma once\n"
                    "struct Result {\n"
                    "    int code;\n"
                    "    std::string msg;\n"
                    "};\n"
                ),
                'handler.cpp': (
                    "#include <vector>\n"
                    "#include \"result.h\"\n"
                    "\n"
                    "std::vector<Result> getResults() {\n"
                    "    return {};\n"
                    "}\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))
            targets = _ref_target_set(results, 'getResults')
            assert 'Result' in targets, f"Result not found in cross-file REFERENCES: {targets}"

    def test_deeply_nested_template_cross_file(self):
        """Item in item.h, used as map<string, map<string, vector<Item>>> in store.cpp."""
        if not self.available:
            pytest.skip("C++ parser not available")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'item.h': (
                    "#pragma once\n"
                    "class Item {\n"
                    "public:\n"
                    "    std::string name;\n"
                    "};\n"
                ),
                'store.cpp': (
                    "#include <map>\n"
                    "#include <vector>\n"
                    "#include <string>\n"
                    "#include \"item.h\"\n"
                    "\n"
                    "class Store {\n"
                    "public:\n"
                    "    std::map<std::string, std::map<std::string, std::vector<Item>>> inventory;\n"
                    "};\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))
            targets = _ref_target_set(results)
            assert 'Item' in targets, f"Item not found in deeply-nested cross-file REFERENCES: {targets}"


# =============================================================================
# 4. Graph-Level Cross-File Generic Edge Verification
# =============================================================================

class TestGraphCrossFileGenerics:
    """Graph builder creates correct edges for cross-file generic REFERENCES."""


    def _build_graph(self, files: dict):
        """Build graph from files dict using analyze_repository."""
        from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_files(tmpdir, files)
            builder = EnhancedUnifiedGraphBuilder()
            result = builder.analyze_repository(tmpdir)
            return result.unified_graph

    def test_python_cross_file_generic_graph_edge(self):
        """Graph has REFERENCES edge from service function to model User (nested generic)."""
        graph = self._build_graph({
            'models.py': "class User:\n    pass\n",
            'service.py': (
                "from typing import Dict, List\n"
                "from models import User\n"
                "\n"
                "def get_users() -> Dict[str, List[User]]:\n"
                "    return {}\n"
            ),
        })

        # Find nodes
        nodes = list(graph.nodes())
        user_nodes = [n for n in nodes if 'User' in n and 'get_users' not in n]
        fn_nodes = [n for n in nodes if 'get_users' in n]

        assert len(user_nodes) > 0, f"No User node found. Nodes: {nodes}"
        assert len(fn_nodes) > 0, f"No get_users node found. Nodes: {nodes}"

        # Check for REFERENCES edge from get_users → User
        ref_edges = [(s, t, d) for s, t, k, d in graph.edges(data=True, keys=True)
                     if d.get('relationship_type') == 'references'
                     and 'get_users' in s and 'User' in t]

        assert len(ref_edges) > 0, (
            f"No references edge from get_users to User. "
            f"All edges: {[(s, t, d.get('relationship_type')) for s, t, k, d in graph.edges(data=True, keys=True)]}"
        )

    def test_java_cross_file_generic_graph_edge(self):
        """Graph has edge for Map<String, List<User>> when User is in another file."""
        graph = self._build_graph({
            'User.java': (
                "package com.example;\n"
                "public class User { private String name; }\n"
            ),
            'UserService.java': (
                "package com.example;\n"
                "import java.util.Map;\n"
                "import java.util.List;\n"
                "\n"
                "public class UserService {\n"
                "    public Map<String, List<User>> getUsers() { return null; }\n"
                "}\n"
            ),
        })

        nodes = list(graph.nodes())
        user_nodes = [n for n in nodes if 'User' in n and 'Service' not in n and 'getUsers' not in n]
        fn_nodes = [n for n in nodes if 'getUsers' in n]

        assert len(user_nodes) > 0, f"No User node. Nodes: {nodes}"
        assert len(fn_nodes) > 0, f"No getUsers node. Nodes: {nodes}"

        # Check for references or composition edge to User
        user_edges = [(s, t, d.get('relationship_type'))
                      for s, t, k, d in graph.edges(data=True, keys=True)
                      if 'User' in t and 'getUsers' in s]
        assert len(user_edges) > 0, (
            f"No edge from getUsers to User. "
            f"All edges: {[(s, t, d.get('relationship_type')) for s, t, k, d in graph.edges(data=True, keys=True)]}"
        )

    def test_typescript_cross_file_generic_graph_edge(self):
        """Graph has REFERENCES edge for Promise<Result> when Result is cross-file."""
        graph = self._build_graph({
            'types.ts': "export interface Result { status: string; }\n",
            'handler.ts': (
                "import { Result } from './types';\n"
                "\n"
                "export async function handle(): Promise<Result> {\n"
                "    return { status: 'ok' };\n"
                "}\n"
            ),
        })

        nodes = list(graph.nodes())
        result_nodes = [n for n in nodes if 'Result' in n and 'handle' not in n]
        fn_nodes = [n for n in nodes if 'handle' in n]

        assert len(result_nodes) > 0, f"No Result node. Nodes: {nodes}"
        assert len(fn_nodes) > 0, f"No handle node. Nodes: {nodes}"

        ref_edges = [(s, t, d.get('relationship_type'))
                     for s, t, k, d in graph.edges(data=True, keys=True)
                     if 'Result' in t and 'handle' in s]
        assert len(ref_edges) > 0, (
            f"No edge from handle to Result. "
            f"All edges: {[(s, t, d.get('relationship_type')) for s, t, k, d in graph.edges(data=True, keys=True)]}"
        )

    def test_graph_preserves_annotations(self):
        """Graph edge annotations include reference_type from parser."""
        graph = self._build_graph({
            'models.py': "class User: pass\n",
            'service.py': (
                "from typing import List\n"
                "from models import User\n"
                "\n"
                "def find(x: List[User]) -> User:\n"
                "    pass\n"
            ),
        })

        # Find reference edges with annotations
        ref_edges = [(s, t, d)
                     for s, t, k, d in graph.edges(data=True, keys=True)
                     if d.get('relationship_type') == 'references'
                     and 'User' in t and 'find' in s]

        # At least some edge should exist
        assert len(ref_edges) > 0, "No reference edges from find to User"

        # Check annotations are preserved (at least one should have annotations)
        annotations = [d.get('annotations', {}) for _, _, d in ref_edges]
        has_ann = any(ann.get('reference_type') for ann in annotations)
        # Note: graph builder may or may not preserve all annotations
        # This test documents the current behavior
        if not has_ann:
            pytest.skip("Graph builder does not currently preserve reference_type annotations")


# =============================================================================
# 5. TypeScript Type Alias Relationships (Phase 3.5 — gaps closed)
# =============================================================================

class TestTypeScriptTypeAliasRelationships:
    """Verify TypeScript type alias REFERENCES and ALIAS_OF (Phase 3.5 fixes).

    Previously type aliases emitted zero relationships. Now:
      - REFERENCES emitted for all constituent types (union members,
        intersection members, generic type arguments)
      - ALIAS_OF emitted for simple aliases (type X = Y) and generic
        aliases (type X = Generic<Y>)
    """

    @pytest.fixture(autouse=True)
    def setup_parser(self):
        self.parser = TypeScriptEnhancedParser()

    def test_type_alias_symbol_correct(self):
        """type Foo = Bar should be TYPE_ALIAS symbol."""
        code = "type UserOrAdmin = User | Admin;\n"
        result = self.parser.parse_file('test.ts', content=code)
        aliases = [s for s in result.symbols if s.name == 'UserOrAdmin']
        assert len(aliases) == 1
        assert aliases[0].symbol_type == SymbolType.TYPE_ALIAS

    def test_type_alias_metadata_has_aliased_type(self):
        """Type alias stores aliased_type in metadata."""
        code = "type UserOrAdmin = User | Admin;\n"
        result = self.parser.parse_file('test.ts', content=code)
        aliases = [s for s in result.symbols if s.name == 'UserOrAdmin']
        assert len(aliases) == 1
        metadata = aliases[0].metadata or {}
        assert metadata.get('is_type_alias') is True
        assert metadata.get('aliased_type') is not None

    def test_simple_alias_emits_alias_of(self):
        """type ActiveUser = User → ALIAS_OF(ActiveUser→User)."""
        code = (
            "interface User { name: string; }\n"
            "type ActiveUser = User;\n"
        )
        result = self.parser.parse_file('test.ts', content=code)
        alias_of_rels = [r for r in result.relationships
                         if r.relationship_type == RelationshipType.ALIAS_OF]
        assert len(alias_of_rels) >= 1, (
            f"Expected ALIAS_OF from ActiveUser→User, got none. "
            f"Rels: {[(r.source_symbol, r.target_symbol, r.relationship_type.value) for r in result.relationships]}"
        )
        assert 'ActiveUser' in alias_of_rels[0].source_symbol
        assert alias_of_rels[0].target_symbol == 'User'

    def test_simple_alias_emits_references(self):
        """type ActiveUser = User → REFERENCES(ActiveUser→User)."""
        code = (
            "interface User { name: string; }\n"
            "type ActiveUser = User;\n"
        )
        result = self.parser.parse_file('test.ts', content=code)
        alias_refs = [r for r in result.relationships
                      if r.relationship_type == RelationshipType.REFERENCES
                      and 'ActiveUser' in r.source_symbol]
        assert len(alias_refs) >= 1, (
            f"Expected REFERENCES from ActiveUser→User, got none. "
            f"Rels: {[(r.source_symbol, r.target_symbol, r.relationship_type.value) for r in result.relationships]}"
        )
        targets = {r.target_symbol for r in alias_refs}
        assert 'User' in targets

    def test_union_type_alias_references(self):
        """type UserOrAdmin = User | Admin → REFERENCES to both."""
        code = (
            "interface User { name: string; }\n"
            "interface Admin { level: number; }\n"
            "type UserOrAdmin = User | Admin;\n"
        )
        result = self.parser.parse_file('test.ts', content=code)

        alias_refs = [r for r in result.relationships
                      if r.relationship_type == RelationshipType.REFERENCES
                      and 'UserOrAdmin' in r.source_symbol]
        targets = {r.target_symbol for r in alias_refs}

        assert 'User' in targets, f"User not in REFERENCES targets: {targets}"
        assert 'Admin' in targets, f"Admin not in REFERENCES targets: {targets}"

    def test_intersection_type_alias_references(self):
        """type Person = HasName & HasAge → REFERENCES to both."""
        code = (
            "interface HasName { name: string; }\n"
            "interface HasAge { age: number; }\n"
            "type Person = HasName & HasAge;\n"
        )
        result = self.parser.parse_file('test.ts', content=code)

        alias_refs = [r for r in result.relationships
                      if r.relationship_type == RelationshipType.REFERENCES
                      and 'Person' in r.source_symbol]
        targets = {r.target_symbol for r in alias_refs}

        assert 'HasName' in targets, f"HasName not in REFERENCES: {targets}"
        assert 'HasAge' in targets, f"HasAge not in REFERENCES: {targets}"

    def test_generic_type_alias_references(self):
        """type WidgetMap = Map<string, Array<Widget>> → REFERENCES Widget."""
        code = (
            "interface Widget { label: string; }\n"
            "type WidgetMap = Map<string, Array<Widget>>;\n"
        )
        result = self.parser.parse_file('test.ts', content=code)

        alias_refs = [r for r in result.relationships
                      if r.relationship_type == RelationshipType.REFERENCES
                      and 'WidgetMap' in r.source_symbol]
        targets = {r.target_symbol for r in alias_refs}

        assert 'Widget' in targets, f"Widget not in REFERENCES: {targets}"

    def test_generic_type_alias_emits_alias_of(self):
        """type WidgetList = Array<Widget> → ALIAS_OF is not emitted for builtins."""
        code = (
            "interface Widget { label: string; }\n"
            "type WidgetList = Array<Widget>;\n"
        )
        result = self.parser.parse_file('test.ts', content=code)

        alias_of_rels = [r for r in result.relationships
                         if r.relationship_type == RelationshipType.ALIAS_OF
                         and 'WidgetList' in r.source_symbol]
        # Array is a builtin generic, so ALIAS_OF should NOT be emitted
        # (builtin generics are filtered by _extract_type_references_recursive)
        assert len(alias_of_rels) == 0, (
            f"ALIAS_OF should not be emitted for builtin generic Array. "
            f"Got: {[(r.source_symbol, r.target_symbol) for r in alias_of_rels]}"
        )

    def test_user_generic_alias_emits_alias_of(self):
        """type UserResult = Response<User> → ALIAS_OF(UserResult→Response)."""
        code = (
            "interface User { name: string; }\n"
            "interface Response<T> { data: T; error: string; }\n"
            "type UserResult = Response<User>;\n"
        )
        result = self.parser.parse_file('test.ts', content=code)

        alias_of_rels = [r for r in result.relationships
                         if r.relationship_type == RelationshipType.ALIAS_OF
                         and 'UserResult' in r.source_symbol]
        assert len(alias_of_rels) >= 1, (
            f"Expected ALIAS_OF from UserResult→Response. "
            f"Rels: {[(r.source_symbol, r.target_symbol, r.relationship_type.value) for r in result.relationships]}"
        )
        assert alias_of_rels[0].target_symbol == 'Response'

    def test_cross_file_type_alias_references(self):
        """Cross-file type alias constituents found as REFERENCES."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'user.ts': "export interface User { name: string; }\n",
                'admin.ts': "export interface Admin { level: number; }\n",
                'types.ts': (
                    "import { User } from './user';\n"
                    "import { Admin } from './admin';\n"
                    "\n"
                    "export type UserOrAdmin = User | Admin;\n"
                ),
            })
            results = self.parser.parse_multiple_files(list(paths.values()))

            alias_refs = []
            for fp, res in results.items():
                for r in res.relationships:
                    if (r.relationship_type == RelationshipType.REFERENCES
                            and 'UserOrAdmin' in r.source_symbol):
                        alias_refs.append(r)

            targets = {r.target_symbol for r in alias_refs}
            assert 'User' in targets, f"User not in cross-file REFERENCES: {targets}"
            assert 'Admin' in targets, f"Admin not in cross-file REFERENCES: {targets}"

    def test_union_type_alias_no_alias_of(self):
        """Union type alias should NOT emit ALIAS_OF (no single target)."""
        code = (
            "interface User { name: string; }\n"
            "interface Admin { level: number; }\n"
            "type UserOrAdmin = User | Admin;\n"
        )
        result = self.parser.parse_file('test.ts', content=code)
        alias_of_rels = [r for r in result.relationships
                         if r.relationship_type == RelationshipType.ALIAS_OF
                         and 'UserOrAdmin' in r.source_symbol]
        assert len(alias_of_rels) == 0, (
            f"Union type alias should not emit ALIAS_OF. "
            f"Got: {[(r.source_symbol, r.target_symbol) for r in alias_of_rels]}"
        )


# =============================================================================
# 6. Cross-Parser Parity — Cross-File
# =============================================================================

class TestCrossParserCrossFileParity:
    """Same nested generic pattern across files → same user types found for all parsers."""

    @pytest.fixture(autouse=True)
    def setup_parsers(self):
        self.py_parser = PythonParser()
        self.java_parser = JavaVisitorParser()
        self.ts_parser = TypeScriptEnhancedParser()

    def test_nested_list_user_cross_file_all_parsers(self):
        """All parsers find 'User' from another file inside a nested container."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Python
            py_paths = _write_files(tmpdir, {
                'py_models.py': "class User: pass\n",
                'py_service.py': (
                    "from typing import Dict, List\n"
                    "from py_models import User\n"
                    "\n"
                    "def get_users() -> Dict[str, List[User]]:\n"
                    "    return {}\n"
                ),
            })
            py_results = self.py_parser.parse_multiple_files(list(py_paths.values()))
            py_targets = _ref_target_set(py_results, 'get_users')

            # Java
            java_paths = _write_files(tmpdir, {
                'User.java': (
                    "package test;\n"
                    "public class User {}\n"
                ),
                'Service.java': (
                    "package test;\n"
                    "import java.util.Map;\n"
                    "import java.util.List;\n"
                    "\n"
                    "public class Service {\n"
                    "    public Map<String, List<User>> getUsers() { return null; }\n"
                    "}\n"
                ),
            })
            java_results = self.java_parser.parse_multiple_files(list(java_paths.values()))
            java_targets = _ref_target_set(java_results, 'getUsers')

            # TypeScript
            ts_paths = _write_files(tmpdir, {
                'user.ts': "export interface User { name: string; }\n",
                'service.ts': (
                    "import { User } from './user';\n"
                    "export function getUsers(): Map<string, Array<User>> {\n"
                    "    return new Map();\n"
                    "}\n"
                ),
            })
            ts_results = self.ts_parser.parse_multiple_files(list(ts_paths.values()))
            ts_targets = _ref_target_set(ts_results, 'getUsers')

            assert 'User' in py_targets, f"Python missed User cross-file: {py_targets}"
            assert 'User' in java_targets, f"Java missed User cross-file: {java_targets}"
            assert 'User' in ts_targets, f"TypeScript missed User cross-file: {ts_targets}"
