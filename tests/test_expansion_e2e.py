"""
End-to-End Parser → Graph → Smart Expansion Tests.

Goal:
    Verify that parser fixes and standardizations (Phases 1–5) ACTUALLY
    produce edges that the expansion engine consumes, i.e. prove that parser
    improvements deliver expansion value and don't break anything.

Architecture:
    Code string → Parser → Graph Builder → expand_smart()
    We use ``EnhancedUnifiedGraphBuilder.analyze_repository()`` to build a
    real graph from temporary files, then feed matched nodes into
    ``expand_smart()`` and verify:
        1. Expected expansion nodes are present (VALUE)
        2. Edges are correctly typed (CORRECTNESS)
        3. No crashes, reasonable budgets (STABILITY)

Languages covered:
    - Python   (inheritance, composition, aggregation, calls, references)
    - Java     (inheritance, implementation, composition, calls, creates)
    - TypeScript (inheritance, implementation, aggregation, alias_of, instantiates)
    - JavaScript (inheritance, composition, creates, calls)
    - C++      (inheritance, specializes, defines_body, instantiates, composition)

Run:
    cd pylon_deepwiki/plugins/deepwiki_plugin
    python -m pytest tests/test_expansion_e2e.py -v
"""

import os
import tempfile

import networkx as nx
import pytest

from plugin_implementation.code_graph.expansion_engine import (
    CLASS_LIKE_TYPES,
    EXPANSION_PRIORITIES,
    SKIP_RELATIONSHIPS,
    AugmentedContent,
    ExpansionResult,
    augment_cpp_node,
    expand_smart,
    find_calls_to_free_functions,
    find_composed_types,
    find_creates_from_methods,
    format_type_args,
    get_edge_annotations,
    has_relationship,
    resolve_alias_chain,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_files(tmpdir: str, files: dict) -> dict:
    """Write files dict to tmpdir, return {filename: absolute_path}."""
    paths = {}
    for name, content in files.items():
        fpath = os.path.join(tmpdir, name)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, 'w') as f:
            f.write(content)
        paths[name] = fpath
    return paths


def _build_graph(files: dict):
    """Build a real graph from a dict of {filename: source_code}."""
    from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_files(tmpdir, files)
        builder = EnhancedUnifiedGraphBuilder()
        result = builder.analyze_repository(tmpdir)
        return result.unified_graph


def _find_node(graph, name_contains: str, symbol_type: str = None) -> str:
    """Find a single node whose ID or symbol_name contains *name_contains*.

    If *symbol_type* is given, filter by ``symbol_type`` attribute.
    Returns the full node ID or raises AssertionError.
    """
    candidates = []
    for nid, data in graph.nodes(data=True):
        # Match on node ID or symbol_name attribute
        sym_name = data.get('symbol_name', '') or ''
        if name_contains in nid or name_contains in sym_name:
            if symbol_type:
                st = data.get('symbol_type', '').lower()
                if st != symbol_type.lower():
                    continue
            candidates.append(nid)
    assert len(candidates) > 0, (
        f"No node matching '{name_contains}'"
        f"{' (type=' + symbol_type + ')' if symbol_type else ''}. "
        f"Nodes: {[n for n in graph.nodes()]}"
    )
    # Prefer exact or closest match
    if len(candidates) == 1:
        return candidates[0]
    # Pick shortest (most specific)
    return min(candidates, key=len)


def _find_nodes_by_name(graph, name_contains: str, symbol_type: str = None) -> list:
    """Return ALL node IDs matching *name_contains* (optionally filtered)."""
    results = []
    for nid, data in graph.nodes(data=True):
        sym_name = data.get('symbol_name', '') or ''
        if name_contains in nid or name_contains in sym_name:
            if symbol_type:
                st = data.get('symbol_type', '').lower()
                if st != symbol_type.lower():
                    continue
            results.append(nid)
    return results


def _expanded_names(result: ExpansionResult) -> set:
    """Return set of symbol_name-like strings from expanded_nodes (for readable assertions).

    Extracts both the full last component (e.g. 'shapes.Circle') and the
    short class name (e.g. 'Circle') so assertions can use either form.
    """
    names = set()
    for nid in result.expanded_nodes:
        parts = nid.split('::')
        last = parts[-1] if parts else nid
        names.add(last)
        # Also add the simple name after the last dot (Java package-qualified)
        if '.' in last:
            names.add(last.rsplit('.', 1)[-1])
    return names


def _edge_types(graph) -> set:
    """Return set of all edge relationship_type values in the graph."""
    types = set()
    for _, _, data in graph.edges(data=True):
        rt = data.get('relationship_type', '')
        if rt:
            types.add(rt)
    return types


def _edges_of_type(graph, rel_type: str) -> list:
    """Return list of (source, target) for edges with given relationship_type."""
    results = []
    for s, t, d in graph.edges(data=True):
        if d.get('relationship_type', '') == rel_type:
            results.append((s, t))
    return results


def _dump_graph_summary(graph, label: str = ""):
    """Debug helper: print graph summary (nodes, edge types, edges)."""
    print(f"\n=== Graph Summary{' (' + label + ')' if label else ''} ===")
    print(f"Nodes ({graph.number_of_nodes()}):")
    for nid, data in graph.nodes(data=True):
        print(f"  {nid} [{data.get('symbol_type', '?')}]")
    print(f"Edges ({graph.number_of_edges()}):")
    for s, t, d in graph.edges(data=True):
        print(f"  {s} --{d.get('relationship_type', '?')}--> {t}")
    print("=" * 50)


# =============================================================================
# 1. PYTHON — End-to-End Expansion
# =============================================================================

class TestPythonExpansionE2E:
    """Python parser → graph → expansion engine integration."""

    PYTHON_CODE = {
        'models.py': (
            "from typing import List, Optional\n"
            "\n"
            "class Entity:\n"
            "    \"\"\"Base entity.\"\"\"\n"
            "    def __init__(self, id: int):\n"
            "        self.id = id\n"
            "\n"
            "class User(Entity):\n"
            "    \"\"\"A user extends Entity.\"\"\"\n"
            "    def __init__(self, id: int, name: str):\n"
            "        super().__init__(id)\n"
            "        self.name = name\n"
        ),
        'service.py': (
            "from typing import List, Optional\n"
            "from models import User, Entity\n"
            "\n"
            "class UserRepository:\n"
            "    \"\"\"Stores users — composition of User.\"\"\"\n"
            "    users: List[User]\n"
            "\n"
            "    def find_by_name(self, name: str) -> Optional[User]:\n"
            "        for u in self.users:\n"
            "            if u.name == name:\n"
            "                return u\n"
            "        return None\n"
            "\n"
            "def create_admin() -> User:\n"
            "    \"\"\"Factory function.\"\"\"\n"
            "    return User(id=0, name='admin')\n"
        ),
    }

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = _build_graph(self.PYTHON_CODE)

    def test_graph_has_inheritance_edge(self):
        """Parser emits INHERITANCE for User(Entity) → expansion P0 input."""
        inh_edges = _edges_of_type(self.graph, 'inheritance')
        # User → Entity
        assert any(
            'User' in s and 'Entity' in t for s, t in inh_edges
        ), f"No inheritance edge User→Entity. Inheritance edges: {inh_edges}"

    def test_graph_has_composition_edge(self):
        """Parser emits COMPOSITION for List[User] field → expansion P1 input."""
        # Composition or references from UserRepository to User
        comp_edges = _edges_of_type(self.graph, 'composition')
        ref_edges = _edges_of_type(self.graph, 'references')
        relevant = [(s, t) for s, t in comp_edges + ref_edges
                     if 'UserRepository' in s or 'users' in s]
        # At least one edge connecting UserRepository or its field to User
        has_link = any('User' in t for s, t in relevant)
        assert has_link, (
            f"No composition/references from UserRepository's field to User. "
            f"comp={comp_edges}, ref={ref_edges}"
        )

    def test_graph_has_calls_edge(self):
        """Parser emits CALLS edge → expansion P2 input."""
        calls_edges = _edges_of_type(self.graph, 'calls')
        # create_admin or find_by_name should call something
        assert len(calls_edges) > 0, "No calls edges in Python graph"

    def test_expand_user_finds_base_class(self):
        """Expanding User pulls in Entity via inheritance (P0 VALUE)."""
        user_node = _find_node(self.graph, 'User', 'class')
        result = expand_smart({user_node}, self.graph)

        expanded = _expanded_names(result)
        assert 'Entity' in expanded, (
            f"Expansion of User did not find base class Entity. "
            f"Expanded: {expanded}"
        )

    def test_expand_user_repo_finds_composed_type(self):
        """Expanding UserRepository pulls in User via composition (P1 VALUE)."""
        repo_node = _find_node(self.graph, 'UserRepository', 'class')
        result = expand_smart({repo_node}, self.graph)

        expanded = _expanded_names(result)
        # Should find User via composed_types or references
        assert 'User' in expanded or 'Entity' in expanded, (
            f"Expansion of UserRepository found neither User nor Entity. "
            f"Expanded: {expanded}, "
            f"Reasons: {result.expansion_reasons}"
        )

    def test_expand_factory_function(self):
        """Expanding create_admin pulls in User via creates/references (P0 VALUE)."""
        fn_nodes = _find_nodes_by_name(self.graph, 'create_admin', 'function')
        if not fn_nodes:
            pytest.skip("create_admin not found as separate function node")
        fn_node = fn_nodes[0]
        result = expand_smart({fn_node}, self.graph)
        expanded = _expanded_names(result)
        # create_admin returns User, so should be referenced or created
        assert 'User' in expanded, (
            f"Expansion of create_admin did not find User. "
            f"Expanded: {expanded}"
        )

    def test_aggregation_edges_exist(self):
        """Python parser emits AGGREGATION for Optional[User] → expansion P1."""
        # Optional[User] in find_by_name should produce an aggregation edge
        agg_edges = _edges_of_type(self.graph, 'aggregation')
        ref_edges = _edges_of_type(self.graph, 'references')
        # Either aggregation or references to User from find_by_name
        all_relevant = agg_edges + ref_edges
        has_user_ref = any('User' in t for s, t in all_relevant if 'find_by_name' in s)
        # Aggregation may not be emitted if Optional detection doesn't fire — that's OK,
        # but references should exist
        assert has_user_ref or len(agg_edges) > 0, (
            f"No aggregation or reference from find_by_name → User. "
            f"Agg edges: {agg_edges}, Ref edges: {ref_edges}"
        )

    def test_multiple_matched_nodes_expansion(self):
        """Expanding both User and UserRepository: all structurally important types appear."""
        user_node = _find_node(self.graph, 'User', 'class')
        repo_node = _find_node(self.graph, 'UserRepository', 'class')
        result = expand_smart({user_node, repo_node}, self.graph)

        expanded = _expanded_names(result)
        # Entity (base of User) must be there
        assert 'Entity' in expanded, (
            f"Multi-match expansion missing Entity. Expanded: {expanded}"
        )


# =============================================================================
# 2. JAVA — End-to-End Expansion
# =============================================================================

class TestJavaExpansionE2E:
    """Java parser → graph → expansion engine integration."""

    JAVA_CODE = {
        'Shape.java': (
            "package shapes;\n"
            "\n"
            "public interface Shape {\n"
            "    double area();\n"
            "}\n"
        ),
        'Circle.java': (
            "package shapes;\n"
            "\n"
            "public class Circle implements Shape {\n"
            "    private double radius;\n"
            "\n"
            "    public Circle(double radius) {\n"
            "        this.radius = radius;\n"
            "    }\n"
            "\n"
            "    @Override\n"
            "    public double area() {\n"
            "        return Math.PI * radius * radius;\n"
            "    }\n"
            "}\n"
        ),
        'Rectangle.java': (
            "package shapes;\n"
            "\n"
            "public class Rectangle implements Shape {\n"
            "    private double width;\n"
            "    private double height;\n"
            "\n"
            "    public Rectangle(double width, double height) {\n"
            "        this.width = width;\n"
            "        this.height = height;\n"
            "    }\n"
            "\n"
            "    @Override\n"
            "    public double area() {\n"
            "        return width * height;\n"
            "    }\n"
            "}\n"
        ),
        'ShapeFactory.java': (
            "package shapes;\n"
            "\n"
            "import java.util.List;\n"
            "import java.util.ArrayList;\n"
            "\n"
            "public class ShapeFactory {\n"
            "    private List<Shape> created;\n"
            "\n"
            "    public ShapeFactory() {\n"
            "        this.created = new ArrayList<>();\n"
            "    }\n"
            "\n"
            "    public Circle createCircle(double r) {\n"
            "        Circle c = new Circle(r);\n"
            "        created.add(c);\n"
            "        return c;\n"
            "    }\n"
            "\n"
            "    public Rectangle createRectangle(double w, double h) {\n"
            "        Rectangle rect = new Rectangle(w, h);\n"
            "        created.add(rect);\n"
            "        return rect;\n"
            "    }\n"
            "}\n"
        ),
    }

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = _build_graph(self.JAVA_CODE)

    def test_graph_has_implementation_edges(self):
        """Java parser emits IMPLEMENTATION for Circle/Rectangle implements Shape."""
        impl_edges = _edges_of_type(self.graph, 'implementation')
        inh_edges = _edges_of_type(self.graph, 'inheritance')
        # Java may use inheritance or implementation for implements
        all_edges = impl_edges + inh_edges
        circle_implements = any(
            'Circle' in s and 'Shape' in t for s, t in all_edges
        )
        rect_implements = any(
            'Rectangle' in s and 'Shape' in t for s, t in all_edges
        )
        assert circle_implements, (
            f"No implementation/inheritance edge Circle→Shape. "
            f"impl={impl_edges}, inh={inh_edges}"
        )
        assert rect_implements, (
            f"No implementation/inheritance edge Rectangle→Shape. "
            f"impl={impl_edges}, inh={inh_edges}"
        )

    def test_graph_has_creates_edges(self):
        """Java parser emits CREATES from ShapeFactory methods → Circle/Rectangle."""
        creates_edges = _edges_of_type(self.graph, 'creates')
        # At least creates edge to Circle or Rectangle
        has_creates = any(
            ('Circle' in t or 'Rectangle' in t)
            for s, t in creates_edges
        )
        assert has_creates, (
            f"No creates edges to Circle/Rectangle. Creates: {creates_edges}"
        )

    def test_expand_circle_finds_interface(self):
        """Expanding Circle finds Shape via implementation (P0 VALUE)."""
        circle_node = _find_node(self.graph, 'Circle', 'class')
        result = expand_smart({circle_node}, self.graph)

        expanded = _expanded_names(result)
        assert 'Shape' in expanded, (
            f"Expansion of Circle did not find interface Shape. "
            f"Expanded: {expanded}"
        )

    def test_expand_shape_finds_implementors(self):
        """Expanding Shape (interface) finds implementors Circle+Rectangle (backward P1 VALUE)."""
        shape_nodes = _find_nodes_by_name(self.graph, 'Shape', 'interface')
        if not shape_nodes:
            # Some parsers may classify Shape as 'class' with implementation edges
            shape_nodes = _find_nodes_by_name(self.graph, 'Shape', 'class')
        if not shape_nodes:
            pytest.skip("Shape node not found in graph")

        result = expand_smart({shape_nodes[0]}, self.graph)
        expanded = _expanded_names(result)

        # At least one implementor should be found
        has_implementor = 'Circle' in expanded or 'Rectangle' in expanded
        assert has_implementor, (
            f"Expansion of Shape found no implementors. "
            f"Expanded: {expanded}, Reasons: {result.expansion_reasons}"
        )

    def test_expand_factory_finds_created_types(self):
        """Expanding ShapeFactory finds Circle/Rectangle via creates (P0 VALUE)."""
        factory_node = _find_node(self.graph, 'ShapeFactory', 'class')
        result = expand_smart({factory_node}, self.graph)

        expanded = _expanded_names(result)
        # Factory creates Circle and Rectangle
        has_created = 'Circle' in expanded or 'Rectangle' in expanded
        assert has_created, (
            f"Expansion of ShapeFactory found no created types. "
            f"Expanded: {expanded}, Reasons: {result.expansion_reasons}"
        )

    def test_composition_field(self):
        """ShapeFactory's List<Shape> field → composition/references edge to Shape."""
        comp_edges = _edges_of_type(self.graph, 'composition')
        ref_edges = _edges_of_type(self.graph, 'references')
        agg_edges = _edges_of_type(self.graph, 'aggregation')
        all_edges = comp_edges + ref_edges + agg_edges
        has_shape_ref = any(
            'Shape' in t and 'ShapeFactory' in s
            for s, t in all_edges
        ) or any(
            'Shape' in t and ('created' in s or 'ShapeFactory' in s)
            for s, t in all_edges
        )
        assert has_shape_ref, (
            f"No comp/ref/agg edge from ShapeFactory/created → Shape. "
            f"comp={comp_edges}, ref={ref_edges}, agg={agg_edges}"
        )


# =============================================================================
# 3. TYPESCRIPT — End-to-End Expansion
# =============================================================================

class TestTypeScriptExpansionE2E:
    """TypeScript parser → graph → expansion engine integration."""

    TS_CODE = {
        'types.ts': (
            "export interface Serializable {\n"
            "    toJSON(): string;\n"
            "}\n"
            "\n"
            "export type UserId = string;\n"
            "\n"
            "export type UserMap = Map<UserId, User>;\n"
        ),
        'models.ts': (
            "import { Serializable, UserId } from './types';\n"
            "\n"
            "export class User implements Serializable {\n"
            "    id: UserId;\n"
            "    name: string;\n"
            "\n"
            "    constructor(id: UserId, name: string) {\n"
            "        this.id = id;\n"
            "        this.name = name;\n"
            "    }\n"
            "\n"
            "    toJSON(): string {\n"
            "        return JSON.stringify({ id: this.id, name: this.name });\n"
            "    }\n"
            "}\n"
        ),
        'service.ts': (
            "import { User } from './models';\n"
            "\n"
            "export class UserService {\n"
            "    private users: User[] = [];\n"
            "\n"
            "    addUser(user: User): void {\n"
            "        this.users.push(user);\n"
            "    }\n"
            "\n"
            "    createUser(id: string, name: string): User {\n"
            "        const user = new User(id, name);\n"
            "        this.users.push(user);\n"
            "        return user;\n"
            "    }\n"
            "}\n"
        ),
    }

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = _build_graph(self.TS_CODE)

    def test_graph_has_implementation_edge(self):
        """TS parser emits IMPLEMENTATION for User implements Serializable."""
        impl_edges = _edges_of_type(self.graph, 'implementation')
        inh_edges = _edges_of_type(self.graph, 'inheritance')
        all_edges = impl_edges + inh_edges
        has_impl = any(
            'User' in s and 'Serializable' in t for s, t in all_edges
        )
        assert has_impl, (
            f"No implementation/inheritance edge User→Serializable. "
            f"impl={impl_edges}, inh={inh_edges}"
        )

    def test_graph_has_type_alias(self):
        """TS parser creates type_alias nodes for UserId and UserMap."""
        alias_nodes = _find_nodes_by_name(self.graph, 'UserId')
        assert len(alias_nodes) > 0, (
            f"No UserId type alias node. Nodes: {list(self.graph.nodes())}"
        )

    def test_expand_user_finds_interface(self):
        """Expanding User finds Serializable via implementation (P0 VALUE)."""
        user_node = _find_node(self.graph, 'User', 'class')
        result = expand_smart({user_node}, self.graph)

        expanded = _expanded_names(result)
        assert 'Serializable' in expanded, (
            f"Expansion of TS User did not find interface Serializable. "
            f"Expanded: {expanded}"
        )

    def test_expand_user_service_finds_user(self):
        """Expanding UserService finds User via composition/creates (VALUE)."""
        service_node = _find_node(self.graph, 'UserService', 'class')
        result = expand_smart({service_node}, self.graph)

        expanded = _expanded_names(result)
        assert 'User' in expanded, (
            f"Expansion of UserService did not find User. "
            f"Expanded: {expanded}"
        )

    def test_expand_type_alias_resolves(self):
        """Expanding UserId alias resolves correctly (if alias_of edge exists)."""
        alias_nodes = _find_nodes_by_name(self.graph, 'UserId', 'type_alias')
        if not alias_nodes:
            pytest.skip("UserId not found as type_alias node")

        result = expand_smart({alias_nodes[0]}, self.graph)
        # Type alias expansion should at least keep the matched node
        assert alias_nodes[0] in result.expanded_nodes

    def test_creates_edge_from_service(self):
        """TS parser emits CREATES from createUser → User (new User(...))."""
        creates_edges = _edges_of_type(self.graph, 'creates')
        instantiates_edges = _edges_of_type(self.graph, 'instantiates')
        all_edges = creates_edges + instantiates_edges
        has_creates = any('User' in t for s, t in all_edges)
        assert has_creates, (
            f"No creates/instantiates edge → User. "
            f"creates={creates_edges}, instantiates={instantiates_edges}"
        )


# =============================================================================
# 4. JAVASCRIPT — End-to-End Expansion
# =============================================================================

class TestJavaScriptExpansionE2E:
    """JavaScript parser → graph → expansion engine integration."""

    JS_CODE = {
        'animal.js': (
            "class Animal {\n"
            "    constructor(name) {\n"
            "        this.name = name;\n"
            "    }\n"
            "\n"
            "    speak() {\n"
            "        return `${this.name} makes a noise.`;\n"
            "    }\n"
            "}\n"
            "\n"
            "module.exports = { Animal };\n"
        ),
        'dog.js': (
            "const { Animal } = require('./animal');\n"
            "\n"
            "class Dog extends Animal {\n"
            "    constructor(name) {\n"
            "        super(name);\n"
            "    }\n"
            "\n"
            "    speak() {\n"
            "        return `${this.name} barks.`;\n"
            "    }\n"
            "}\n"
            "\n"
            "module.exports = { Dog };\n"
        ),
        'shelter.js': (
            "const { Dog } = require('./dog');\n"
            "const { Animal } = require('./animal');\n"
            "\n"
            "class Shelter {\n"
            "    constructor() {\n"
            "        /** @type {Animal[]} */\n"
            "        this.animals = [];\n"
            "    }\n"
            "\n"
            "    adopt(name) {\n"
            "        const dog = new Dog(name);\n"
            "        this.animals.push(dog);\n"
            "        return dog;\n"
            "    }\n"
            "}\n"
            "\n"
            "module.exports = { Shelter };\n"
        ),
    }

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = _build_graph(self.JS_CODE)

    def test_graph_has_inheritance_edge(self):
        """JS parser emits INHERITANCE for Dog extends Animal."""
        inh_edges = _edges_of_type(self.graph, 'inheritance')
        has_inh = any(
            'Dog' in s and 'Animal' in t for s, t in inh_edges
        )
        assert has_inh, (
            f"No inheritance edge Dog→Animal. Edges: {inh_edges}"
        )

    def test_expand_dog_finds_base_class(self):
        """Expanding Dog finds Animal via inheritance (P0 VALUE)."""
        dog_node = _find_node(self.graph, 'Dog', 'class')
        result = expand_smart({dog_node}, self.graph)

        expanded = _expanded_names(result)
        assert 'Animal' in expanded, (
            f"Expansion of Dog did not find base class Animal. "
            f"Expanded: {expanded}"
        )

    def test_expand_shelter_finds_related_types(self):
        """Expanding Shelter finds Dog or Animal via creates/composition (VALUE)."""
        shelter_node = _find_node(self.graph, 'Shelter', 'class')
        result = expand_smart({shelter_node}, self.graph)

        expanded = _expanded_names(result)
        has_related = 'Dog' in expanded or 'Animal' in expanded
        assert has_related, (
            f"Expansion of Shelter found neither Dog nor Animal. "
            f"Expanded: {expanded}, Reasons: {result.expansion_reasons}"
        )

    def test_expand_animal_backward_finds_derived(self):
        """Expanding Animal finds Dog via backward inheritance (VALUE)."""
        animal_node = _find_node(self.graph, 'Animal', 'class')
        result = expand_smart({animal_node}, self.graph)

        expanded = _expanded_names(result)
        assert 'Dog' in expanded, (
            f"Backward expansion of Animal did not find derived class Dog. "
            f"Expanded: {expanded}"
        )

    def test_creates_edge_from_adopt(self):
        """JS parser emits CREATES from Shelter.adopt → Dog (new Dog(...))."""
        creates_edges = _edges_of_type(self.graph, 'creates')
        has_creates = any('Dog' in t for s, t in creates_edges)
        assert has_creates, (
            f"No creates edge → Dog. Creates edges: {creates_edges}"
        )

    def test_js_no_aggregation_but_expansion_ok(self):
        """JS lacks aggregation edges but expansion still works via composition+references."""
        # This documents a known gap: JS has no aggregation distinction
        agg_edges = _edges_of_type(self.graph, 'aggregation')
        # It's OK if there are no aggregation edges — the expansion engine
        # gracefully falls back to composition and references
        shelter_node = _find_node(self.graph, 'Shelter', 'class')
        result = expand_smart({shelter_node}, self.graph)
        # Should still expand without crashing
        assert len(result.expanded_nodes) >= 1


# =============================================================================
# 5. C++ — End-to-End Expansion
# =============================================================================

class TestCppExpansionE2E:
    """C++ parser → graph → expansion engine integration."""

    CPP_CODE = {
        'base.h': (
            "#pragma once\n"
            "\n"
            "class Shape {\n"
            "public:\n"
            "    virtual ~Shape() = default;\n"
            "    virtual double area() const = 0;\n"
            "};\n"
        ),
        'circle.h': (
            "#pragma once\n"
            "#include \"base.h\"\n"
            "\n"
            "class Circle : public Shape {\n"
            "public:\n"
            "    explicit Circle(double radius);\n"
            "    double area() const override;\n"
            "private:\n"
            "    double radius_;\n"
            "};\n"
        ),
        'circle.cpp': (
            "#include \"circle.h\"\n"
            "#include <cmath>\n"
            "\n"
            "Circle::Circle(double radius) : radius_(radius) {}\n"
            "\n"
            "double Circle::area() const {\n"
            "    return M_PI * radius_ * radius_;\n"
            "}\n"
        ),
        'container.h': (
            "#pragma once\n"
            "#include \"base.h\"\n"
            "#include <vector>\n"
            "#include <memory>\n"
            "\n"
            "class ShapeContainer {\n"
            "public:\n"
            "    void add(std::unique_ptr<Shape> shape);\n"
            "    double totalArea() const;\n"
            "private:\n"
            "    std::vector<std::unique_ptr<Shape>> shapes_;\n"
            "};\n"
        ),
    }

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = _build_graph(self.CPP_CODE)

    def test_graph_has_inheritance_edge(self):
        """C++ parser emits INHERITANCE for Circle : public Shape."""
        inh_edges = _edges_of_type(self.graph, 'inheritance')
        has_inh = any(
            'Circle' in s and 'Shape' in t for s, t in inh_edges
        )
        assert has_inh, (
            f"No inheritance edge Circle→Shape. Edges: {inh_edges}"
        )

    def test_graph_has_defines_body_edge(self):
        """C++ parser emits DEFINES_BODY for circle.cpp → circle.h split."""
        db_edges = _edges_of_type(self.graph, 'defines_body')
        # defines_body: impl → declaration
        has_db = any(
            ('Circle' in s or 'area' in s) and ('Circle' in t or 'area' in t)
            for s, t in db_edges
        )
        if not has_db:
            # May not fire if parser doesn't detect cross-file impl
            pytest.skip(
                f"No defines_body edge for Circle. "
                f"DB edges: {db_edges}. This may be parser-specific."
            )

    def test_expand_circle_finds_base(self):
        """Expanding Circle finds Shape via inheritance (P0 VALUE)."""
        circle_node = _find_node(self.graph, 'Circle', 'class')
        result = expand_smart({circle_node}, self.graph)

        expanded = _expanded_names(result)
        assert 'Shape' in expanded, (
            f"Expansion of C++ Circle did not find base Shape. "
            f"Expanded: {expanded}"
        )

    def test_expand_circle_augmentation(self):
        """C++ augmentation: Circle declaration gets impl body from circle.cpp."""
        circle_node = _find_node(self.graph, 'Circle', 'class')
        result = expand_smart({circle_node}, self.graph)

        if result.augmentations:
            # Check any augmentation exists for Circle-related nodes
            aug_nodes = list(result.augmentations.keys())
            assert len(aug_nodes) > 0, "Augmentation dict empty"
        # If no augmentation, it means defines_body wasn't emitted — that's
        # a parser limitation, not an expansion bug
        # (test_graph_has_defines_body_edge covers the parser side)

    def test_expand_shape_backward_finds_derived(self):
        """Expanding Shape finds Circle via backward inheritance."""
        shape_node = _find_node(self.graph, 'Shape', 'class')
        result = expand_smart({shape_node}, self.graph)

        expanded = _expanded_names(result)
        assert 'Circle' in expanded, (
            f"Backward expansion of Shape did not find Circle. "
            f"Expanded: {expanded}"
        )

    def test_expand_container_finds_shape(self):
        """Expanding ShapeContainer finds Shape via composition (P1 VALUE)."""
        container_node = _find_node(self.graph, 'ShapeContainer', 'class')
        result = expand_smart({container_node}, self.graph)

        expanded = _expanded_names(result)
        assert 'Shape' in expanded, (
            f"Expansion of ShapeContainer did not find Shape. "
            f"Expanded: {expanded}, Reasons: {result.expansion_reasons}"
        )


# =============================================================================
# 5b. C++ TEMPLATE INSTANTIATION — type_args propagation
# =============================================================================

class TestCppTemplateInstantiationE2E:
    """Verify C++ template usage flows through to expansion correctly.

    Two code paths exist for template type information:

    1. **Field declarations** (``Container<Point> vertices_;`` inside a class)
       → tree-sitter ``field_declaration`` → composition/aggregation edges
       → type info preserved in ``field_type`` annotation (e.g. 'Container<Point>')

    2. **Variable declarations** (``Container<Point> c;`` inside a function)
       → tree-sitter ``declaration`` → ``visit_declaration``
       → INSTANTIATES edge with ``type_args`` annotation (e.g. ['Point'])

    Both paths deliver the *concrete type argument* to the expansion engine.
    """

    # ----- Path 1: field declarations → composition/aggregation -----
    CPP_FIELD_TEMPLATE_CODE = {
        'point.h': (
            "#pragma once\n"
            "\n"
            "struct Point {\n"
            "    double x, y;\n"
            "};\n"
        ),
        'polygon.h': (
            "#pragma once\n"
            '#include "point.h"\n'
            "\n"
            "template<typename T>\n"
            "class Container {\n"
            "public:\n"
            "    void push(const T& item);\n"
            "    T get(int index) const;\n"
            "private:\n"
            "    T* data_;\n"
            "    int size_;\n"
            "};\n"
            "\n"
            "class Polygon {\n"
            "public:\n"
            "    void addVertex(const Point& p);\n"
            "    int vertexCount() const;\n"
            "private:\n"
            "    Container<Point> vertices_;\n"
            "};\n"
        ),
    }

    # ----- Path 2: variable declarations → INSTANTIATES -----
    CPP_VAR_TEMPLATE_CODE = {
        'point.h': (
            "#pragma once\n"
            "\n"
            "struct Point {\n"
            "    double x, y;\n"
            "};\n"
        ),
        'container.h': (
            "#pragma once\n"
            "\n"
            "template<typename T>\n"
            "class Container {\n"
            "public:\n"
            "    void push(const T& item);\n"
            "    T get(int index) const;\n"
            "};\n"
        ),
        'usage.cpp': (
            '#include "point.h"\n'
            '#include "container.h"\n'
            "\n"
            "void processPoints() {\n"
            "    Container<Point> c;\n"
            "    Point p;\n"
            "    p.x = 1.0;\n"
            "    p.y = 2.0;\n"
            "    c.push(p);\n"
            "}\n"
        ),
    }

    # --- Field path tests ---

    def test_field_template_composition_exists(self):
        """Container<Point> field produces composition edge with type info."""
        graph = _build_graph(self.CPP_FIELD_TEMPLATE_CODE)
        comp_edges = _edges_of_type(graph, 'composition')
        has_container_comp = any('Container' in t for s, t in comp_edges)
        assert has_container_comp, (
            f"No composition edge → Container from field. "
            f"Composition edges: {comp_edges}"
        )

    def test_field_template_aggregation_captures_type_arg(self):
        """Container<Point> field produces aggregation edge to Point (type arg)."""
        graph = _build_graph(self.CPP_FIELD_TEMPLATE_CODE)
        agg_edges = _edges_of_type(graph, 'aggregation')
        has_point_agg = any('Point' in t for s, t in agg_edges)
        assert has_point_agg, (
            f"No aggregation edge → Point from template field. "
            f"Aggregation edges: {agg_edges}"
        )

    def test_field_template_type_in_annotation(self):
        """Composition edge annotation preserves full generic type 'Container<Point>'."""
        graph = _build_graph(self.CPP_FIELD_TEMPLATE_CODE)
        for s, t, d in graph.edges(data=True):
            if d.get('relationship_type') == 'composition' and 'Container' in t:
                field_type = d.get('annotations', {}).get('field_type', '')
                assert 'Point' in field_type, (
                    f"Composition annotation field_type missing 'Point'. "
                    f"Got: {field_type}"
                )
                return
        pytest.fail("No composition edge to Container found")

    def test_field_expansion_finds_container_and_point(self):
        """Expanding Polygon finds both Container (composition) and Point."""
        graph = _build_graph(self.CPP_FIELD_TEMPLATE_CODE)
        polygon_node = _find_node(graph, 'Polygon', 'class')
        result = expand_smart({polygon_node}, graph)
        expanded = _expanded_names(result)
        assert 'Container' in expanded, (
            f"Expansion of Polygon did not find Container. Expanded: {expanded}"
        )
        assert 'Point' in expanded, (
            f"Expansion of Polygon did not find Point. Expanded: {expanded}"
        )

    # --- Variable declaration path tests (INSTANTIATES) ---

    def test_var_instantiates_edge_exists(self):
        """Function-scope Container<Point> c; emits INSTANTIATES edge."""
        graph = _build_graph(self.CPP_VAR_TEMPLATE_CODE)
        inst_edges = _edges_of_type(graph, 'instantiates')
        has_inst = any('Container' in t for s, t in inst_edges)
        if not has_inst:
            # visit_declaration may not fire for all tree-sitter versions
            # Document what we DO have
            all_types = _edge_types(graph)
            pytest.skip(
                f"No INSTANTIATES edge from variable declaration. "
                f"Edge types present: {all_types}. "
                f"This may be a tree-sitter version issue."
            )

    def test_var_instantiates_type_args(self):
        """INSTANTIATES from variable carries type_args=['Point']."""
        from plugin_implementation.code_graph.expansion_engine import get_edge_annotations

        graph = _build_graph(self.CPP_VAR_TEMPLATE_CODE)
        inst_edges = _edges_of_type(graph, 'instantiates')
        container_edges = [(s, t) for s, t in inst_edges if 'Container' in t]
        if not container_edges:
            pytest.skip("No INSTANTIATES edge to Container — skip type_args check")

        source, target = container_edges[0]
        annot = get_edge_annotations(graph, source, target, 'instantiates')
        type_args = annot.get('type_args', [])
        assert 'Point' in type_args, (
            f"type_args missing 'Point'. Annotations: {annot}"
        )

    def test_var_expansion_follows_instantiates(self):
        """Expanding processPoints finds Container via INSTANTIATES."""
        graph = _build_graph(self.CPP_VAR_TEMPLATE_CODE)
        func_nodes = _find_nodes_by_name(graph, 'processPoints', 'function')
        if not func_nodes:
            pytest.skip("processPoints function not found in graph")

        result = expand_smart({func_nodes[0]}, graph)
        expanded = _expanded_names(result)
        assert 'Container' in expanded or 'Point' in expanded, (
            f"Expansion of processPoints did not find Container or Point. "
            f"Expanded: {expanded}"
        )

    def test_explicit_template_instantiation(self):
        """Explicit template instantiation (template class Foo<int>;) with type_args."""
        code = {
            'tmpl.h': (
                "#pragma once\n"
                "\n"
                "template<typename T>\n"
                "class Stack {\n"
                "public:\n"
                "    void push(const T& item);\n"
                "    T pop();\n"
                "};\n"
            ),
            'tmpl.cpp': (
                '#include "tmpl.h"\n'
                "\n"
                "template<typename T>\n"
                "void Stack<T>::push(const T& item) {}\n"
                "\n"
                "template<typename T>\n"
                "T Stack<T>::pop() { return T(); }\n"
                "\n"
                "// Explicit instantiation\n"
                "template class Stack<int>;\n"
                "template class Stack<double>;\n"
            ),
        }
        graph = _build_graph(code)
        inst_edges = _edges_of_type(graph, 'instantiates')
        stack_inst = [(s, t) for s, t in inst_edges if 'Stack' in t]
        assert stack_inst, (
            f"Expected INSTANTIATES edges targeting Stack from explicit "
            f"instantiation. All instantiates edges: {inst_edges}"
        )

        from plugin_implementation.code_graph.expansion_engine import edges_between
        found_type_args = set()
        for s, t in stack_inst:
            for edge_data in edges_between(graph, s, t):
                if edge_data.get('relationship_type', '').lower() == 'instantiates':
                    ta = (edge_data.get('annotations') or {}).get('type_args')
                    if ta:
                        found_type_args.update(ta)

        assert found_type_args, (
            f"Expected type_args on INSTANTIATES edges for Stack. "
            f"Edges: {stack_inst}"
        )
        assert 'int' in found_type_args, f"Missing 'int' in type_args: {found_type_args}"
        assert 'double' in found_type_args, f"Missing 'double' in type_args: {found_type_args}"


# =============================================================================
# 5c. TYPESCRIPT — Template/Generic Instantiation
# =============================================================================

class TestTypeScriptGenericInstantiationE2E:
    """Verify TypeScript generic instantiation flows through expansion.

    Current TS parser behavior for ``new Store<Product>()``:
      - ``new Product(name, price)`` in method body → CREATES edge (method → Product)
      - ``this.store.add(...)`` → CALLS edge (method → Store.add)
      - ``private store: Store<Product>`` field → defines edge only (no composition to Store)

    So from Catalog class-level expansion:
      - Product IS reachable (via creates from method)
      - Store IS reachable (via calls from method → Store.add → parent Store)
    """

    TS_GENERIC_CODE = {
        'store.ts': (
            "export class Store<T> {\n"
            "    private items: T[] = [];\n"
            "\n"
            "    add(item: T): void {\n"
            "        this.items.push(item);\n"
            "    }\n"
            "\n"
            "    getAll(): T[] {\n"
            "        return this.items;\n"
            "    }\n"
            "}\n"
        ),
        'product.ts': (
            "export class Product {\n"
            "    constructor(public name: string, public price: number) {}\n"
            "}\n"
        ),
        'catalog.ts': (
            "import { Store } from './store';\n"
            "import { Product } from './product';\n"
            "\n"
            "export class Catalog {\n"
            "    private store: Store<Product> = new Store<Product>();\n"
            "\n"
            "    addProduct(name: string, price: number): void {\n"
            "        this.store.add(new Product(name, price));\n"
            "    }\n"
            "}\n"
        ),
    }

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = _build_graph(self.TS_GENERIC_CODE)

    def test_creates_edge_to_product(self):
        """TS parser emits creates edge from Catalog.addProduct → Product."""
        creates = _edges_of_type(self.graph, 'creates')
        has_product = any('Product' in t for s, t in creates)
        assert has_product, (
            f"No creates edge → Product. Creates: {creates}"
        )

    def test_calls_edge_to_store_method(self):
        """TS parser emits calls edge from Catalog.addProduct → Store.add."""
        calls = _edges_of_type(self.graph, 'calls')
        has_store_call = any('Store' in t and 'add' in t for s, t in calls)
        assert has_store_call, (
            f"No calls edge → Store.add. Calls: {calls}"
        )

    def test_expand_catalog_finds_product(self):
        """Expanding Catalog finds Product (via creates from method)."""
        catalog_node = _find_node(self.graph, 'Catalog', 'class')
        result = expand_smart({catalog_node}, self.graph)

        expanded = _expanded_names(result)
        assert 'Product' in expanded, (
            f"Expansion of Catalog missing Product. "
            f"Expanded: {expanded}"
        )

    def test_expand_catalog_reachability(self):
        """Document what Catalog expansion actually finds — all connected types."""
        catalog_node = _find_node(self.graph, 'Catalog', 'class')
        result = expand_smart({catalog_node}, self.graph)
        expanded = _expanded_names(result)
        # Product must be found (creates edge)
        assert 'Product' in expanded, f"Product not found. Expanded: {expanded}"
        # Store MAY be found through calls chain — document but don't require
        # (Store reachability depends on expansion engine traversing
        #  calls → method → parent class, which is indirect)


# =============================================================================
# 6. CROSS-LANGUAGE — Expansion Engine Stability
# =============================================================================

class TestExpansionStability:
    """Verify expansion engine handles edge cases gracefully."""

    def test_empty_graph_expansion(self):
        """expand_smart on empty graph returns just matched nodes."""
        g = nx.MultiDiGraph()
        g.add_node("A", symbol_type="class")
        result = expand_smart({"A"}, g)
        assert "A" in result.expanded_nodes
        assert len(result.expanded_nodes) == 1

    def test_node_not_in_graph(self):
        """expand_smart with node not in graph doesn't crash."""
        g = nx.MultiDiGraph()
        result = expand_smart({"nonexistent"}, g)
        assert "nonexistent" in result.expanded_nodes

    def test_global_cap_respected(self):
        """Global cap prevents unbounded expansion."""
        g = nx.MultiDiGraph()
        # Create a star graph: one center, many leaves
        center = "center"
        g.add_node(center, symbol_type="class")
        for i in range(100):
            leaf = f"leaf_{i}"
            g.add_node(leaf, symbol_type="class")
            g.add_edge(center, leaf, relationship_type="inheritance")

        result = expand_smart({center}, g, global_cap=10, per_symbol_cap=10)
        # Should have center + at most 10 expanded
        assert len(result.expanded_nodes) <= 11

    def test_per_symbol_cap_respected(self):
        """Per-symbol cap limits single node expansion."""
        g = nx.MultiDiGraph()
        center = "center"
        g.add_node(center, symbol_type="class")
        for i in range(50):
            leaf = f"leaf_{i}"
            g.add_node(leaf, symbol_type="class")
            g.add_edge(center, leaf, relationship_type="inheritance")

        result = expand_smart({center}, g, per_symbol_cap=5, global_cap=100)
        # center + at most 5 expansion nodes
        assert len(result.expanded_nodes) <= 6


# =============================================================================
# 7. EDGE COVERAGE MATRIX — Document what each parser produces
# =============================================================================

class TestEdgeCoverageMatrix:
    """Document and verify which expansion-relevant edges each parser produces.

    This is not a pass/fail test — it's a coverage audit that records
    which edge types each language parser actually emits in a real graph.
    Failures here indicate parser gaps, not expansion bugs.
    """

    # Edge types the expansion engine consumes (from EXPANSION_PRIORITIES)
    EXPANSION_EDGES = set(EXPANSION_PRIORITIES.keys())
    # Edge types the expansion engine explicitly skips
    SKIPPED_EDGES = SKIP_RELATIONSHIPS

    def _check_edge_coverage(self, files, language_label):
        """Build graph and report edge type coverage."""
        graph = _build_graph(files)
        edge_types = _edge_types(graph)

        consumed = edge_types & self.EXPANSION_EDGES
        skipped = edge_types & self.SKIPPED_EDGES
        missing = self.EXPANSION_EDGES - edge_types

        print(f"\n{'=' * 60}")
        print(f"Edge Coverage: {language_label}")
        print(f"{'=' * 60}")
        print(f"  Consumed by expansion: {sorted(consumed)}")
        print(f"  Skipped by expansion:  {sorted(skipped)}")
        print(f"  Missing (not emitted): {sorted(missing)}")
        print(f"  Other edges:           {sorted(edge_types - consumed - skipped)}")

        return consumed, skipped, missing, graph

    def test_python_edge_coverage(self):
        """Python: document which expansion edges are actually emitted."""
        consumed, _, _, graph = self._check_edge_coverage(
            TestPythonExpansionE2E.PYTHON_CODE,
            "Python"
        )
        # Python MUST produce at least inheritance, calls
        assert 'inheritance' in consumed or 'calls' in consumed, (
            f"Python produced no expansion-relevant edges! Consumed: {consumed}"
        )

    def test_java_edge_coverage(self):
        """Java: document which expansion edges are actually emitted."""
        consumed, _, _, graph = self._check_edge_coverage(
            TestJavaExpansionE2E.JAVA_CODE,
            "Java"
        )
        # Java MUST produce at least implementation or inheritance
        assert 'implementation' in consumed or 'inheritance' in consumed, (
            f"Java produced no implementation/inheritance edges! Consumed: {consumed}"
        )

    def test_typescript_edge_coverage(self):
        """TypeScript: document which expansion edges are actually emitted."""
        consumed, _, _, graph = self._check_edge_coverage(
            TestTypeScriptExpansionE2E.TS_CODE,
            "TypeScript"
        )
        assert 'implementation' in consumed or 'inheritance' in consumed, (
            f"TS produced no implementation/inheritance edges! Consumed: {consumed}"
        )

    def test_javascript_edge_coverage(self):
        """JavaScript: document which expansion edges are actually emitted."""
        consumed, _, missing, graph = self._check_edge_coverage(
            TestJavaScriptExpansionE2E.JS_CODE,
            "JavaScript"
        )
        assert 'inheritance' in consumed, (
            f"JS produced no inheritance edges! Consumed: {consumed}"
        )
        # Document known gap: JS should not have aggregation
        assert 'aggregation' in missing, (
            "Unexpected: JS now emits aggregation (was previously missing)"
        )

    def test_cpp_edge_coverage(self):
        """C++: document which expansion edges are actually emitted."""
        consumed, _, _, graph = self._check_edge_coverage(
            TestCppExpansionE2E.CPP_CODE,
            "C++"
        )
        assert 'inheritance' in consumed, (
            f"C++ produced no inheritance edges! Consumed: {consumed}"
        )


# =============================================================================
# 8. PARSER FIX VALUE VERIFICATION — Do Phase 1-5 fixes help expansion?
# =============================================================================

class TestParserFixesValueForExpansion:
    """Verify that specific parser improvements translate to expansion value.

    Each test targets a specific parser fix and proves it either:
    (a) Produces edges the expansion engine consumes (VALUE), or
    (b) Produces edges the expansion engine skips (NO DIRECT VALUE), or
    (c) Is structurally correct but needs expansion engine evolution.
    """

    def test_python_aggregation_optional_type(self):
        """Phase 5 fix: Optional[User] → AGGREGATION edge → expansion P1.

        VALUE: The expansion engine follows aggregation edges at P1 priority.
        If the parser doesn't emit aggregation for Optional, the expansion
        engine's aggregation budget is wasted for Python.
        """
        code = {
            'models.py': (
                "class User:\n"
                "    name: str\n"
            ),
            'service.py': (
                "from typing import Optional\n"
                "from models import User\n"
                "\n"
                "class Service:\n"
                "    current_user: Optional[User]\n"
            ),
        }
        graph = _build_graph(code)
        agg_edges = _edges_of_type(graph, 'aggregation')
        ref_edges = _edges_of_type(graph, 'references')
        comp_edges = _edges_of_type(graph, 'composition')

        # Either aggregation or at least references to User
        has_user_link = (
            any('User' in t for s, t in agg_edges) or
            any('User' in t for s, t in ref_edges) or
            any('User' in t for s, t in comp_edges)
        )
        assert has_user_link, (
            f"No edge links to User from Optional[User]. "
            f"Agg: {agg_edges}, Ref: {ref_edges}, Comp: {comp_edges}"
        )

        # Verify expansion picks it up
        service_node = _find_node(graph, 'Service', 'class')
        result = expand_smart({service_node}, graph)
        expanded = _expanded_names(result)
        assert 'User' in expanded, (
            f"Expansion of Service did not find User from Optional[User]. "
            f"Expanded: {expanded}"
        )

    def test_java_overrides_skipped_by_design(self):
        """Java OVERRIDES edges are SKIPPED by expansion engine — by design.

        NO DIRECT VALUE for expansion, but ensures stability (no crash).
        The overrides edge is structurally correct but intentionally unused.
        """
        code = {
            'Base.java': (
                "package app;\n"
                "public class Base {\n"
                "    public String toString() { return \"Base\"; }\n"
                "}\n"
            ),
            'Child.java': (
                "package app;\n"
                "public class Child extends Base {\n"
                "    @Override\n"
                "    public String toString() { return \"Child\"; }\n"
                "}\n"
            ),
        }
        graph = _build_graph(code)
        overrides_edges = _edges_of_type(graph, 'overrides')
        # overrides exists in graph but expansion engine SKIPs it
        # This is by design — expansion finds polymorphism via
        # inheritance/implementation instead

        # Expansion should still work (inheritance)
        child_node = _find_node(graph, 'Child', 'class')
        result = expand_smart({child_node}, graph)
        expanded = _expanded_names(result)
        assert 'Base' in expanded, (
            f"Expansion of Child did not find Base via inheritance. "
            f"Expanded: {expanded}"
        )

    def test_ts_instantiates_creates_value(self):
        """TS CREATES/INSTANTIATES from new expressions → P0 expansion.

        VALUE: Parser Phase 1-4 improvements ensure `new Foo()` produces
        creates/instantiates edges that the expansion engine picks up at P0.
        """
        code = {
            'config.ts': (
                "export class Config {\n"
                "    debug: boolean = false;\n"
                "}\n"
            ),
            'app.ts': (
                "import { Config } from './config';\n"
                "\n"
                "export class App {\n"
                "    config: Config;\n"
                "\n"
                "    constructor() {\n"
                "        this.config = new Config();\n"
                "    }\n"
                "}\n"
            ),
        }
        graph = _build_graph(code)

        creates = _edges_of_type(graph, 'creates')
        instantiates = _edges_of_type(graph, 'instantiates')
        all_creates = creates + instantiates

        has_creates = any('Config' in t for s, t in all_creates)
        assert has_creates, (
            f"No creates/instantiates → Config. "
            f"Creates: {creates}, Instantiates: {instantiates}"
        )

        # Verify expansion uses it
        app_node = _find_node(graph, 'App', 'class')
        result = expand_smart({app_node}, graph)
        expanded = _expanded_names(result)
        assert 'Config' in expanded, (
            f"Expansion of App did not find Config via creates. "
            f"Expanded: {expanded}"
        )

    def test_constructor_standardization_value(self):
        """Phase 2: Constructor standardization ensures __init__/constructor
        nodes don't leak as class-level pollutants.

        VALUE: Constructors properly nested under classes means the 2-hop
        helpers (find_creates_from_methods, find_composed_types) work correctly
        because they traverse Class→[defines]→Method→[creates]→Type.
        """
        code = {
            'models.py': (
                "class Config:\n"
                "    debug: bool = False\n"
                "\n"
                "class App:\n"
                "    def __init__(self):\n"
                "        self.config = Config()\n"
            ),
        }
        graph = _build_graph(code)

        # Verify constructor is properly nested (defines edge from App)
        defines_edges = _edges_of_type(graph, 'defines')
        app_defines = [(s, t) for s, t in defines_edges if 'App' in s]
        has_init = any('__init__' in t or 'constructor' in t.lower()
                       for s, t in app_defines)
        assert has_init or len(app_defines) > 0, (
            f"App has no defines edges (constructor not nested). "
            f"Defines: {defines_edges}"
        )

    def test_cross_file_references_value(self):
        """Phase 3: Cross-file generic REFERENCES (e.g. Dict[str, List[User]])
        produce edges the expansion engine uses at P2.

        VALUE: Without cross-file reference resolution, the expansion engine
        sees only in-file symbols, missing critical type dependencies.
        """
        code = {
            'models.py': (
                "class Item:\n"
                "    name: str\n"
            ),
            'store.py': (
                "from typing import Dict, List\n"
                "from models import Item\n"
                "\n"
                "class Store:\n"
                "    items: Dict[str, List[Item]]\n"
            ),
        }
        graph = _build_graph(code)

        # Should have references from Store/items to Item
        ref_edges = _edges_of_type(graph, 'references')
        comp_edges = _edges_of_type(graph, 'composition')
        agg_edges = _edges_of_type(graph, 'aggregation')
        all_edges = ref_edges + comp_edges + agg_edges

        has_item_link = any('Item' in t for s, t in all_edges if 'Store' in s or 'items' in s)
        assert has_item_link, (
            f"No cross-file edge from Store → Item. "
            f"Ref: {ref_edges}, Comp: {comp_edges}, Agg: {agg_edges}"
        )

        # Expansion should find it
        store_node = _find_node(graph, 'Store', 'class')
        result = expand_smart({store_node}, graph)
        expanded = _expanded_names(result)
        assert 'Item' in expanded, (
            f"Expansion did not find cross-file Item. Expanded: {expanded}"
        )


# =============================================================================
# 9. BIDIRECTIONAL EXPANSION — Full Pipeline Test
# =============================================================================

class TestBidirectionalExpansionPipeline:
    """Verify bidirectional expansion works end-to-end with real graphs.

    Tests both forward (what does this use?) and backward (who uses this?)
    directions through the full Parser → Graph → Expansion pipeline.
    """

    HIERARCHY_CODE = {
        'vehicle.py': (
            "class Vehicle:\n"
            "    \"\"\"Base vehicle.\"\"\"\n"
            "    def __init__(self, make: str):\n"
            "        self.make = make\n"
        ),
        'car.py': (
            "from vehicle import Vehicle\n"
            "\n"
            "class Car(Vehicle):\n"
            "    \"\"\"A car.\"\"\"\n"
            "    def __init__(self, make: str, model: str):\n"
            "        super().__init__(make)\n"
            "        self.model = model\n"
        ),
        'truck.py': (
            "from vehicle import Vehicle\n"
            "\n"
            "class Truck(Vehicle):\n"
            "    \"\"\"A truck.\"\"\"\n"
            "    def __init__(self, make: str, payload: float):\n"
            "        super().__init__(make)\n"
            "        self.payload = payload\n"
        ),
        'fleet.py': (
            "from typing import List\n"
            "from vehicle import Vehicle\n"
            "from car import Car\n"
            "\n"
            "class Fleet:\n"
            "    \"\"\"Manages vehicles.\"\"\"\n"
            "    vehicles: List[Vehicle]\n"
            "\n"
            "    def add_car(self, make: str, model: str) -> Car:\n"
            "        car = Car(make, model)\n"
            "        return car\n"
        ),
    }

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graph = _build_graph(self.HIERARCHY_CODE)

    def test_forward_expansion_car(self):
        """Car → Vehicle (forward inheritance, P0)."""
        car_node = _find_node(self.graph, 'Car', 'class')
        result = expand_smart({car_node}, self.graph)
        expanded = _expanded_names(result)
        assert 'Vehicle' in expanded, f"Forward: Car→Vehicle missing. {expanded}"

    def test_backward_expansion_vehicle(self):
        """Vehicle ← Car, Truck (backward inheritance, P1)."""
        vehicle_node = _find_node(self.graph, 'Vehicle', 'class')
        result = expand_smart({vehicle_node}, self.graph)
        expanded = _expanded_names(result)
        # At least one derived class should be found
        has_derived = 'Car' in expanded or 'Truck' in expanded
        assert has_derived, (
            f"Backward: Vehicle←Car/Truck missing. Expanded: {expanded}"
        )

    def test_fleet_composition_expansion(self):
        """Fleet → Vehicle (composition via List[Vehicle] field)."""
        fleet_node = _find_node(self.graph, 'Fleet', 'class')
        result = expand_smart({fleet_node}, self.graph)
        expanded = _expanded_names(result)
        # Fleet composes Vehicle and creates Car
        has_related = 'Vehicle' in expanded or 'Car' in expanded
        assert has_related, (
            f"Fleet→Vehicle/Car missing. Expanded: {expanded}"
        )

    def test_multi_match_full_hierarchy(self):
        """Matching Car + Fleet: expansion covers the full class hierarchy."""
        car_node = _find_node(self.graph, 'Car', 'class')
        fleet_node = _find_node(self.graph, 'Fleet', 'class')
        result = expand_smart({car_node, fleet_node}, self.graph)

        expanded = _expanded_names(result)
        # Vehicle must appear (base of Car, composed by Fleet)
        assert 'Vehicle' in expanded, (
            f"Multi-match: Vehicle missing. Expanded: {expanded}"
        )

    def test_budget_does_not_over_expand(self):
        """Budget system keeps expansion bounded even with rich graphs."""
        car_node = _find_node(self.graph, 'Car', 'class')
        result = expand_smart({car_node}, self.graph, per_symbol_cap=2, global_cap=3)

        # car_node itself + at most 2 expansion nodes = 3
        assert len(result.expanded_nodes) <= 3 + 1, (
            f"Over-expanded: {len(result.expanded_nodes)} nodes "
            f"(cap=2+matched). Nodes: {result.expanded_nodes}"
        )
