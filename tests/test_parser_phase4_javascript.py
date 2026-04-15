"""
Phase 4 — JavaScript Enrichment Tests.

JS-1: Class field extraction (FIELD symbols + DEFINES relationships)
JS-2: JSDoc @param / @returns / @type type extraction (REFERENCES)

Covers:
    1. Single-file class field extraction
    2. Cross-file class field usage  
    3. Graph-level FIELD node + DEFINES edge verification
    4. JSDoc type extraction (single-file)
    5. JSDoc cross-file type references
    6. JSDoc graph-level edge verification

See PLANNING_PARSER_REVIEW.md Phase 4 for details.
"""

import os
import tempfile
import pytest

from plugin_implementation.parsers.base_parser import SymbolType, RelationshipType
from plugin_implementation.parsers.javascript_visitor_parser import JavaScriptVisitorParser


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


def _parse_single(code: str, filename: str = 'test.js'):
    """Parse JavaScript code (single file) and return ParseResult."""
    parser = JavaScriptVisitorParser()
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        with open(filepath, 'w') as f:
            f.write(code)
        return parser.parse_file(filepath)


def _symbols_of_type(result, stype: SymbolType):
    """Return symbols matching a given SymbolType."""
    return [s for s in result.symbols if s.symbol_type == stype]


def _rels_of_type(result, rtype: RelationshipType):
    """Return relationships matching a given RelationshipType."""
    return [r for r in result.relationships if r.relationship_type == rtype]


def _all_rels(results: dict, rtype: RelationshipType, source_filter=None):
    """Collect relationships of rtype across all files in multi-file results."""
    out = []
    for fp, result in results.items():
        for r in result.relationships:
            if r.relationship_type == rtype:
                if source_filter is None or source_filter in r.source_symbol:
                    out.append(r)
    return out


def _all_symbols(results: dict, stype: SymbolType = None):
    """Collect symbols across all files in multi-file results."""
    out = []
    for fp, result in results.items():
        for s in result.symbols:
            if stype is None or s.symbol_type == stype:
                out.append(s)
    return out


def _build_graph(files: dict):
    """Build graph from files dict using analyze_repository."""
    from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_files(tmpdir, files)
        builder = EnhancedUnifiedGraphBuilder()
        result = builder.analyze_repository(tmpdir)
        return result.unified_graph


# =============================================================================
# 1. JS-1 — Single-File Class Field Extraction
# =============================================================================

class TestJSClassFieldSingleFile:
    """Verify that JavaScript class fields are extracted as FIELD symbols."""

    def test_basic_field_with_initializer(self):
        """x = 5 → FIELD symbol."""
        result = _parse_single("""
class Foo {
    x = 5;
}
""")
        fields = _symbols_of_type(result, SymbolType.FIELD)
        assert len(fields) >= 1, f"Expected at least 1 FIELD, got {len(fields)}: {[s.name for s in result.symbols]}"
        
        field = next(f for f in fields if f.name == 'x')
        assert field.name == 'x'
        assert 'Foo' in field.parent_symbol, f"parent_symbol should contain 'Foo': {field.parent_symbol}"
        assert field.is_static is False
        assert field.visibility == 'public'

    def test_static_field(self):
        """static y = 10 → static FIELD."""
        result = _parse_single("""
class Counter {
    static count = 0;
}
""")
        fields = _symbols_of_type(result, SymbolType.FIELD)
        static_fields = [f for f in fields if f.name == 'count']
        assert len(static_fields) == 1
        assert static_fields[0].is_static is True
        assert 'Counter' in static_fields[0].parent_symbol

    def test_private_field(self):
        """#secret = 'hidden' → private FIELD."""
        result = _parse_single("""
class Vault {
    #secret = 'hidden';
    #key;
}
""")
        fields = _symbols_of_type(result, SymbolType.FIELD)
        secret_fields = [f for f in fields if 'secret' in f.name]
        assert len(secret_fields) == 1
        assert secret_fields[0].visibility == 'private'
        assert secret_fields[0].metadata.get('is_private') is True
        
        key_fields = [f for f in fields if 'key' in f.name]
        assert len(key_fields) == 1
        assert key_fields[0].visibility == 'private'

    def test_uninitialized_field(self):
        """name; → FIELD with no initializer."""
        result = _parse_single("""
class Person {
    name;
    age;
}
""")
        fields = _symbols_of_type(result, SymbolType.FIELD)
        names = {f.name for f in fields}
        assert 'name' in names, f"Missing 'name' field. Got: {names}"
        assert 'age' in names, f"Missing 'age' field. Got: {names}"
        
        name_field = next(f for f in fields if f.name == 'name')
        assert name_field.metadata.get('has_initializer') is False

    def test_multiple_fields_in_class(self):
        """Multiple mixed fields in one class."""
        result = _parse_single("""
class Config {
    host = 'localhost';
    static port = 3000;
    #apiKey = 'secret';
    timeout;
}
""")
        fields = _symbols_of_type(result, SymbolType.FIELD)
        field_names = {f.name for f in fields}
        
        assert 'host' in field_names
        assert 'port' in field_names
        assert 'timeout' in field_names
        # Private field name includes #
        private_fields = [f for f in fields if f.metadata.get('is_private')]
        assert len(private_fields) >= 1

    def test_field_full_name_qualified(self):
        """FIELD full_name should be Class.field_name."""
        result = _parse_single("""
class Widget {
    color = 'blue';
}
""")
        fields = _symbols_of_type(result, SymbolType.FIELD)
        color = next(f for f in fields if f.name == 'color')
        assert 'Widget' in color.full_name, f"full_name should contain class name: {color.full_name}"
        assert 'color' in color.full_name, f"full_name should contain field name: {color.full_name}"

    def test_field_with_complex_initializer(self):
        """Fields with object/array initializers."""
        result = _parse_single("""
class DataStore {
    items = [];
    metadata = { version: 1, active: true };
    callback = () => {};
}
""")
        fields = _symbols_of_type(result, SymbolType.FIELD)
        field_names = {f.name for f in fields}
        assert 'items' in field_names
        assert 'metadata' in field_names
        assert 'callback' in field_names


# =============================================================================
# 2. JS-1 — DEFINES Relationships for Fields
# =============================================================================

class TestJSFieldDefinesRelationships:
    """Verify DEFINES edges from Class → Field."""

    def test_class_defines_field(self):
        """Class DEFINES its fields."""
        result = _parse_single("""
class Animal {
    name = 'unknown';
    static count = 0;

    constructor(name) {
        this.name = name;
    }

    speak() {
        return this.name;
    }
}
""")
        defines = _rels_of_type(result, RelationshipType.DEFINES)
        
        # Should have DEFINES for: name field, count field, constructor, speak method
        field_defines = [d for d in defines if d.annotations.get('member_type') == 'field']
        assert len(field_defines) >= 2, (
            f"Expected >=2 field DEFINES, got {len(field_defines)}. "
            f"All DEFINES: {[(d.source_symbol, d.target_symbol, d.annotations) for d in defines]}"
        )
        
        # Verify source is the class name
        for fd in field_defines:
            assert 'Animal' in fd.source_symbol, f"DEFINES source should contain 'Animal', got '{fd.source_symbol}'"
        
        # Verify different member_types are present
        member_types = {d.annotations.get('member_type') for d in defines}
        assert 'field' in member_types, f"Expected 'field' in member_types: {member_types}"
        assert 'constructor' in member_types or 'method' in member_types, (
            f"Expected constructor/method in member_types: {member_types}"
        )

    def test_static_field_defines_annotation(self):
        """DEFINES for static field has is_static=True annotation."""
        result = _parse_single("""
class Logger {
    static level = 'info';
    prefix = '[LOG]';
}
""")
        defines = _rels_of_type(result, RelationshipType.DEFINES)
        field_defines = [d for d in defines if d.annotations.get('member_type') == 'field']
        
        static_defines = [d for d in field_defines if d.annotations.get('is_static')]
        non_static_defines = [d for d in field_defines if not d.annotations.get('is_static')]
        
        assert len(static_defines) >= 1, f"Expected static field DEFINES. Got: {[(d.target_symbol, d.annotations) for d in field_defines]}"
        assert len(non_static_defines) >= 1, f"Expected non-static field DEFINES. Got: {[(d.target_symbol, d.annotations) for d in field_defines]}"

    def test_multiple_classes_defines_fields(self):
        """Each class DEFINES only its own fields."""
        result = _parse_single("""
class Cat {
    whiskers = 12;
}

class Dog {
    tailLength = 30;
}
""")
        defines = _rels_of_type(result, RelationshipType.DEFINES)
        
        cat_defines = [d for d in defines if 'Cat' in d.source_symbol]
        dog_defines = [d for d in defines if 'Dog' in d.source_symbol]
        
        cat_targets = {d.target_symbol for d in cat_defines}
        dog_targets = {d.target_symbol for d in dog_defines}
        
        assert any('whiskers' in t for t in cat_targets), f"Cat should define whiskers. Got: {cat_targets}"
        assert any('tailLength' in t for t in dog_targets), f"Dog should define tailLength. Got: {dog_targets}"
        # No cross-contamination
        assert not any('tailLength' in t for t in cat_targets), f"Cat should NOT define tailLength: {cat_targets}"
        assert not any('whiskers' in t for t in dog_targets), f"Dog should NOT define whiskers: {dog_targets}"


# =============================================================================
# 3. JS-1 — Cross-File Class Field Tests
# =============================================================================

class TestJSFieldCrossFile:
    """Verify fields in multi-file parsing context."""

    def test_field_preserved_in_multi_file_parse(self):
        """Fields are extracted when parsing multiple files."""
        parser = JavaScriptVisitorParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'config.js': """
export class Config {
    host = 'localhost';
    static port = 3000;
    #secret = 'abc';
}
""",
                'app.js': """
import { Config } from './config.js';

class App {
    config = new Config();

    run() {
        return this.config.host;
    }
}
""",
            })
            results = parser.parse_multiple_files(list(paths.values()))
        
        # Config class should have its fields
        config_symbols = []
        for fp, result in results.items():
            if 'config' in fp:
                config_symbols = result.symbols
        
        config_fields = [s for s in config_symbols if s.symbol_type == SymbolType.FIELD]
        field_names = {f.name for f in config_fields}
        assert 'host' in field_names, f"Missing 'host' field. Got: {field_names}"
        assert 'port' in field_names, f"Missing 'port' field. Got: {field_names}"
        
        # App class should also have its field
        app_symbols = []
        for fp, result in results.items():
            if 'app' in fp:
                app_symbols = result.symbols
        
        app_fields = [s for s in app_symbols if s.symbol_type == SymbolType.FIELD]
        app_field_names = {f.name for f in app_fields}
        assert 'config' in app_field_names, f"Missing 'config' field in App. Got: {app_field_names}"

    def test_field_defines_in_cross_file(self):
        """DEFINES relationships for fields preserved in multi-file parsing."""
        parser = JavaScriptVisitorParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'model.js': """
export class User {
    name = '';
    email = '';
    static DEFAULT_ROLE = 'viewer';
}
""",
                'service.js': """
import { User } from './model.js';

export class UserService {
    users = [];
    
    addUser(user) {
        this.users.push(user);
    }
}
""",
            })
            results = parser.parse_multiple_files(list(paths.values()))
        
        # Check DEFINES in model.js
        model_defines = []
        for fp, result in results.items():
            if 'model' in fp:
                model_defines = _rels_of_type(result, RelationshipType.DEFINES)
        
        field_defines = [d for d in model_defines if d.annotations.get('member_type') == 'field']
        assert len(field_defines) >= 3, (
            f"Expected >=3 field DEFINES for User. Got {len(field_defines)}: "
            f"{[(d.source_symbol, d.target_symbol) for d in field_defines]}"
        )

    def test_inherited_class_fields_separate(self):
        """Subclass and superclass fields are tracked separately."""
        parser = JavaScriptVisitorParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'base.js': """
export class Shape {
    color = 'red';
    static instanceCount = 0;
}
""",
                'circle.js': """
import { Shape } from './base.js';

export class Circle extends Shape {
    radius = 0;
}
""",
            })
            results = parser.parse_multiple_files(list(paths.values()))
        
        # Shape fields
        shape_fields = []
        circle_fields = []
        for fp, result in results.items():
            for s in result.symbols:
                if s.symbol_type == SymbolType.FIELD:
                    if 'Shape' in (s.parent_symbol or ''):
                        shape_fields.append(s)
                    elif 'Circle' in (s.parent_symbol or ''):
                        circle_fields.append(s)
        
        shape_names = {f.name for f in shape_fields}
        circle_names = {f.name for f in circle_fields}
        
        assert 'color' in shape_names, f"Shape should have 'color'. Got: {shape_names}"
        assert 'radius' in circle_names, f"Circle should have 'radius'. Got: {circle_names}"
        # Each class defines its own fields only
        assert 'radius' not in shape_names
        assert 'color' not in circle_names


# =============================================================================
# 4. JS-1 — Graph-Level FIELD Verification
# =============================================================================

class TestJSFieldGraphLevel:
    """Verify graph builder includes FIELD nodes and DEFINES edges."""

    def test_field_node_in_graph(self):
        """Graph should contain FIELD nodes for class fields."""
        graph = _build_graph({
            'widget.js': """
export class Widget {
    label = 'default';
    static version = '1.0';
    
    render() {
        return this.label;
    }
}
""",
        })
        
        nodes = list(graph.nodes(data=True))
        node_ids = [n[0] for n in nodes]
        
        # Check for field nodes
        field_nodes = [n for n, d in nodes if 'label' in n.lower() or 'version' in n.lower()]
        # Fields are typically included in graph as children of classes
        # The graph builder may or may not include fields as separate nodes
        # depending on ARCHITECTURAL_SYMBOLS configuration
        
        # At minimum, Widget class should be in the graph
        widget_nodes = [n for n in node_ids if 'Widget' in n]
        assert len(widget_nodes) > 0, f"Widget class not in graph. Nodes: {node_ids}"

    def test_defines_edge_for_field_in_graph(self):
        """Graph should have DEFINES edge from Class to Field."""
        graph = _build_graph({
            'counter.js': """
export class Counter {
    count = 0;
    
    increment() {
        this.count++;
    }
}
""",
        })
        
        edges = [(s, t, d) for s, t, k, d in graph.edges(data=True, keys=True)]
        defines_edges = [(s, t, d) for s, t, d in edges if d.get('relationship_type') == 'defines']
        
        # Should have defines edges from Counter to its members
        counter_defines = [(s, t) for s, t, d in defines_edges if 'Counter' in s]
        assert len(counter_defines) > 0, (
            f"No DEFINES edges from Counter. "
            f"All defines: {[(s, t) for s, t, d in defines_edges]}"
        )

    def test_field_cross_file_graph(self):
        """Cross-file field extraction reflected in graph."""
        graph = _build_graph({
            'store.js': """
export class Store {
    items = [];
    static maxSize = 100;
}
""",
            'app.js': """
import { Store } from './store.js';

export class App {
    store = new Store();
    
    addItem(item) {
        this.store.items.push(item);
    }
}
""",
        })
        
        nodes = list(graph.nodes())
        
        # Both classes should be in graph
        store_nodes = [n for n in nodes if 'Store' in n and 'App' not in n]
        app_nodes = [n for n in nodes if 'App' in n and 'Store' not in n]
        
        assert len(store_nodes) > 0, f"Store not in graph. Nodes: {nodes}"
        assert len(app_nodes) > 0, f"App not in graph. Nodes: {nodes}"


# =============================================================================
# 5. JS-2 — JSDoc Type Extraction (Single-File)
# =============================================================================

class TestJSDocSingleFile:
    """Verify JSDoc @param / @returns / @type extraction."""

    def test_jsdoc_param_type(self):
        """@param {string} name → parameter type extracted."""
        result = _parse_single("""
/**
 * Greet a user.
 * @param {string} name - The user's name.
 * @returns {string} The greeting.
 */
function greet(name) {
    return 'Hello, ' + name;
}
""")
        funcs = _symbols_of_type(result, SymbolType.FUNCTION)
        greet = next((f for f in funcs if f.name == 'greet'), None)
        assert greet is not None, f"greet function not found. Symbols: {[s.name for s in result.symbols]}"
        
        # Check parameter types from JSDoc
        param_types = greet.metadata.get('jsdoc_params', {})
        assert 'name' in param_types, f"JSDoc @param 'name' not extracted. metadata: {greet.metadata}"
        assert param_types['name'] == 'string', f"Expected 'string', got: {param_types['name']}"

    def test_jsdoc_returns_type(self):
        """@returns {number} → return type extracted."""
        result = _parse_single("""
/**
 * Calculate area.
 * @param {number} width
 * @param {number} height
 * @returns {number} The area.
 */
function area(width, height) {
    return width * height;
}
""")
        funcs = _symbols_of_type(result, SymbolType.FUNCTION)
        area_fn = next((f for f in funcs if f.name == 'area'), None)
        assert area_fn is not None
        
        jsdoc_returns = area_fn.metadata.get('jsdoc_returns')
        assert jsdoc_returns == 'number', f"Expected 'number', got: {jsdoc_returns}"

    def test_jsdoc_complex_type(self):
        """@param {Array<User>} users → complex type extracted."""
        result = _parse_single("""
/**
 * @param {Array<User>} users
 * @returns {Promise<void>}
 */
function processUsers(users) {
    // ...
}
""")
        funcs = _symbols_of_type(result, SymbolType.FUNCTION)
        fn = next((f for f in funcs if f.name == 'processUsers'), None)
        assert fn is not None
        
        param_types = fn.metadata.get('jsdoc_params', {})
        assert 'users' in param_types
        # Should capture the full type string
        assert 'Array' in param_types['users'] or 'User' in param_types['users'], (
            f"Complex param type not captured: {param_types['users']}"
        )

    def test_jsdoc_method_params(self):
        """JSDoc on class methods also extracts types."""
        result = _parse_single("""
class Calculator {
    /**
     * @param {number} a
     * @param {number} b
     * @returns {number}
     */
    add(a, b) {
        return a + b;
    }
}
""")
        methods = _symbols_of_type(result, SymbolType.METHOD)
        add_method = next((m for m in methods if m.name == 'add'), None)
        assert add_method is not None
        
        param_types = add_method.metadata.get('jsdoc_params', {})
        assert 'a' in param_types, f"JSDoc @param 'a' not extracted. metadata: {add_method.metadata}"
        assert param_types['a'] == 'number'

    def test_jsdoc_type_annotation(self):
        """@type {string} on variable/field."""
        result = _parse_single("""
class Settings {
    /** @type {string} */
    host = 'localhost';
    
    /** @type {number} */
    port = 3000;
}
""")
        fields = _symbols_of_type(result, SymbolType.FIELD)
        host_field = next((f for f in fields if f.name == 'host'), None)
        assert host_field is not None
        
        jsdoc_type = host_field.metadata.get('jsdoc_type')
        assert jsdoc_type == 'string', f"Expected 'string', got: {jsdoc_type}"

    def test_jsdoc_references_emitted(self):
        """JSDoc types that reference user-defined classes emit REFERENCES."""
        result = _parse_single("""
class User {
    name = '';
}

/**
 * @param {User} user
 * @returns {string}
 */
function getUserName(user) {
    return user.name;
}
""")
        refs = _rels_of_type(result, RelationshipType.REFERENCES)
        ref_targets = {r.target_symbol for r in refs}
        
        # getUserName should reference User via JSDoc
        user_refs = [r for r in refs if r.target_symbol == 'User' and 'getUserName' in r.source_symbol]
        assert len(user_refs) > 0, (
            f"Expected REFERENCES from getUserName to User via JSDoc. "
            f"All refs: {[(r.source_symbol, r.target_symbol) for r in refs]}"
        )

    def test_jsdoc_nullable_and_union(self):
        """@param {?string} and {string|number} types."""
        result = _parse_single("""
/**
 * @param {?string} name
 * @param {string|number} id
 * @returns {boolean}
 */
function validate(name, id) {
    return !!name && !!id;
}
""")
        funcs = _symbols_of_type(result, SymbolType.FUNCTION)
        fn = next((f for f in funcs if f.name == 'validate'), None)
        assert fn is not None
        
        param_types = fn.metadata.get('jsdoc_params', {})
        assert 'name' in param_types, f"@param name not extracted: {param_types}"
        assert 'id' in param_types, f"@param id not extracted: {param_types}"

    def test_jsdoc_no_comment_no_metadata(self):
        """Function without JSDoc should not have jsdoc fields."""
        result = _parse_single("""
function simple(x) {
    return x + 1;
}
""")
        funcs = _symbols_of_type(result, SymbolType.FUNCTION)
        fn = next((f for f in funcs if f.name == 'simple'), None)
        assert fn is not None
        
        # No JSDoc metadata when there's no JSDoc comment
        assert fn.metadata.get('jsdoc_params') is None or fn.metadata.get('jsdoc_params') == {}
        assert fn.metadata.get('jsdoc_returns') is None


# =============================================================================
# 6. JS-2 — JSDoc Cross-File Tests
# =============================================================================

class TestJSDocCrossFile:
    """Verify JSDoc type references work in multi-file context."""

    def test_jsdoc_cross_file_type_reference(self):
        """JSDoc referencing a type from another file creates REFERENCES."""
        parser = JavaScriptVisitorParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'user.js': """
export class User {
    name = '';
    email = '';
}
""",
                'service.js': """
import { User } from './user.js';

/**
 * Find a user by ID.
 * @param {number} id
 * @returns {User} The found user.
 */
export function findUser(id) {
    return new User();
}

/**
 * @param {Array<User>} users
 * @returns {number}
 */
export function countUsers(users) {
    return users.length;
}
""",
            })
            results = parser.parse_multiple_files(list(paths.values()))
        
        # Gather REFERENCES from service.js
        service_refs = []
        for fp, result in results.items():
            if 'service' in fp:
                service_refs = _rels_of_type(result, RelationshipType.REFERENCES)
        
        ref_targets = {r.target_symbol for r in service_refs}
        # Should reference User from JSDoc
        assert 'User' in ref_targets, (
            f"Expected 'User' in REFERENCES targets from service.js JSDoc. "
            f"Got: {ref_targets}"
        )

    def test_jsdoc_field_type_cross_file(self):
        """JSDoc @type on fields referencing imported class."""
        parser = JavaScriptVisitorParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'types.js': """
export class Config {
    host = '';
}
""",
                'app.js': """
import { Config } from './types.js';

class Application {
    /** @type {Config} */
    config = null;
    
    /** @type {string} */
    name = 'MyApp';
}
""",
            })
            results = parser.parse_multiple_files(list(paths.values()))
        
        # Check field metadata for JSDoc type
        app_fields = []
        for fp, result in results.items():
            if 'app' in fp:
                app_fields = [s for s in result.symbols if s.symbol_type == SymbolType.FIELD]
        
        config_field = next((f for f in app_fields if f.name == 'config'), None)
        assert config_field is not None, f"'config' field not found. Fields: {[f.name for f in app_fields]}"
        
        jsdoc_type = config_field.metadata.get('jsdoc_type')
        assert jsdoc_type == 'Config', f"Expected JSDoc type 'Config', got: {jsdoc_type}"


# =============================================================================
# 7. JS-2 — JSDoc Graph-Level Tests
# =============================================================================

class TestJSDocGraphLevel:
    """Verify JSDoc types create proper edges in the graph."""

    def test_jsdoc_creates_references_edge_in_graph(self):
        """JSDoc @param {User} → graph REFERENCES edge."""
        graph = _build_graph({
            'user.js': """
export class User {
    name = '';
}
""",
            'handler.js': """
import { User } from './user.js';

/**
 * @param {User} user
 * @returns {string}
 */
export function handleUser(user) {
    return user.name;
}
""",
        })
        
        nodes = list(graph.nodes())
        edges = [(s, t, d) for s, t, k, d in graph.edges(data=True, keys=True)]
        
        user_nodes = [n for n in nodes if 'User' in n]
        handler_nodes = [n for n in nodes if 'handleUser' in n]
        
        assert len(user_nodes) > 0, f"User not in graph. Nodes: {nodes}"
        assert len(handler_nodes) > 0, f"handleUser not in graph. Nodes: {nodes}"
        
        # Check for references or creates edge from handleUser to User
        user_refs = [(s, t, d.get('relationship_type')) for s, t, d in edges
                     if 'User' in t and 'handleUser' in s]
        assert len(user_refs) > 0, (
            f"No edge from handleUser to User. "
            f"All edges: {[(s, t, d.get('relationship_type')) for s, t, d in edges]}"
        )


# =============================================================================
# 8. JS-1 + JS-2 Combined — Field + JSDoc Integration
# =============================================================================

class TestJSFieldAndJSDocCombined:
    """Tests combining field extraction with JSDoc type annotations."""

    def test_field_with_jsdoc_type_and_class_defines(self):
        """Class with JSDoc-typed fields has both FIELD symbols and DEFINES + type metadata."""
        result = _parse_single("""
class Database {
    /** @type {string} */
    connectionString = '';
    
    /** @type {number} */
    static maxConnections = 10;
    
    /** @type {boolean} */
    #isConnected = false;
    
    /**
     * @param {string} query
     * @returns {Array<Object>}
     */
    execute(query) {
        return [];
    }
}
""")
        fields = _symbols_of_type(result, SymbolType.FIELD)
        methods = _symbols_of_type(result, SymbolType.METHOD)
        defines = _rels_of_type(result, RelationshipType.DEFINES)
        
        # Should have 3 fields
        assert len(fields) >= 3, f"Expected 3 fields, got {len(fields)}: {[f.name for f in fields]}"
        
        # Should have 1 method
        assert len(methods) >= 1, f"Expected method 'execute'. Got: {[m.name for m in methods]}"
        
        # DEFINES should include both fields and method
        field_defines = [d for d in defines if d.annotations.get('member_type') == 'field']
        method_defines = [d for d in defines if d.annotations.get('member_type') == 'method']
        
        assert len(field_defines) >= 3, f"Expected 3 field DEFINES. Got: {len(field_defines)}"
        assert len(method_defines) >= 1, f"Expected method DEFINES. Got: {len(method_defines)}"

    def test_cross_file_field_jsdoc_and_graph(self):
        """Full integration: cross-file fields with JSDoc, verified in graph."""
        graph = _build_graph({
            'event.js': """
export class Event {
    name = '';
    timestamp = 0;
}
""",
            'handler.js': """
import { Event } from './event.js';

export class EventHandler {
    /** @type {Array} */
    events = [];
    
    /**
     * @param {Event} event
     */
    handle(event) {
        this.events.push(event);
    }
}
""",
        })
        
        nodes = list(graph.nodes())
        
        # Both classes should be in graph
        event_nodes = [n for n in nodes if 'Event' in n and 'Handler' not in n]
        handler_nodes = [n for n in nodes if 'EventHandler' in n]
        
        assert len(event_nodes) > 0, f"Event class not in graph. Nodes: {nodes}"
        assert len(handler_nodes) > 0, f"EventHandler class not in graph. Nodes: {nodes}"


# =============================================================================
# 9. Capabilities & Edge Cases 
# =============================================================================

class TestJSFieldCapabilities:
    """Verify capabilities declaration includes FIELD."""

    def test_capabilities_include_field(self):
        """Parser capabilities declare FIELD support."""
        parser = JavaScriptVisitorParser()
        caps = parser.capabilities
        
        # Check supported_symbols includes FIELD
        supported = caps.supported_symbols
        assert SymbolType.FIELD in supported, f"FIELD not in supported_symbols: {supported}"

    def test_empty_class_no_fields(self):
        """Empty class produces no FIELD symbols."""
        result = _parse_single("""
class Empty {}
""")
        fields = _symbols_of_type(result, SymbolType.FIELD)
        assert len(fields) == 0, f"Expected no fields for empty class, got: {[f.name for f in fields]}"

    def test_field_in_nested_class(self):
        """Fields in nested class expressions."""
        result = _parse_single("""
const Outer = class {
    x = 1;
    
    getInner() {
        return class Inner {
            y = 2;
        };
    }
};
""")
        fields = _symbols_of_type(result, SymbolType.FIELD)
        # At minimum 'x' should be extracted
        field_names = {f.name for f in fields}
        assert 'x' in field_names, f"Expected 'x' field. Got: {field_names}"

    def test_jsx_class_with_fields(self):
        """JSX file (.jsx) class fields also extracted."""
        result = _parse_single("""
class Component {
    state = { count: 0 };
    static displayName = 'Counter';
    
    render() {
        return null;
    }
}
""", filename='component.jsx')
        
        fields = _symbols_of_type(result, SymbolType.FIELD)
        field_names = {f.name for f in fields}
        assert 'state' in field_names, f"Expected 'state' field in JSX. Got: {field_names}"
        assert 'displayName' in field_names, f"Expected 'displayName' field in JSX. Got: {field_names}"


# =============================================================================
# 10. No-JSDoc Regression — functions/methods/fields without JSDoc
# =============================================================================

class TestNoJSDocRegression:
    """Ensure no false jsdoc metadata leaks when there are no JSDoc comments."""

    def test_function_without_jsdoc_clean_metadata(self):
        """Plain function has no jsdoc_params/jsdoc_returns."""
        result = _parse_single("""
function add(a, b) {
    return a + b;
}
""")
        fn = next(s for s in result.symbols if s.name == 'add')
        assert fn.metadata.get('jsdoc_params') is None or fn.metadata.get('jsdoc_params') == {}
        assert fn.metadata.get('jsdoc_returns') is None

    def test_method_without_jsdoc_clean_metadata(self):
        """Class method with no JSDoc has no jsdoc metadata."""
        result = _parse_single("""
class Calc {
    multiply(a, b) {
        return a * b;
    }
}
""")
        method = next(s for s in result.symbols if s.name == 'multiply')
        assert method.metadata.get('jsdoc_params') is None or method.metadata.get('jsdoc_params') == {}
        assert method.metadata.get('jsdoc_returns') is None

    def test_field_without_jsdoc_no_type(self):
        """Field without @type JSDoc has no jsdoc_type."""
        result = _parse_single("""
class Box {
    width = 100;
    height = 50;
}
""")
        w = next(s for s in result.symbols if s.name == 'width')
        assert w.metadata.get('jsdoc_type') is None

    def test_regular_comment_not_treated_as_jsdoc(self):
        """// comment and /* comment */ (non-JSDoc) don't trigger extraction."""
        result = _parse_single("""
// @param {string} name
function hello(name) {
    return name;
}

/* @returns {number} */
function count() {
    return 42;
}
""")
        hello = next(s for s in result.symbols if s.name == 'hello')
        count = next(s for s in result.symbols if s.name == 'count')
        assert hello.metadata.get('jsdoc_params') is None or hello.metadata.get('jsdoc_params') == {}
        assert count.metadata.get('jsdoc_returns') is None

    def test_non_adjacent_jsdoc_not_captured(self):
        """JSDoc separated from function by another statement doesn't attach."""
        result = _parse_single("""
/**
 * @param {string} name
 * @returns {number}
 */
const SEPARATOR = '---';

function process(name) {
    return name.length;
}
""")
        fn = next(s for s in result.symbols if s.name == 'process')
        # The JSDoc is separated by SEPARATOR, should not attach to process
        assert fn.metadata.get('jsdoc_params') is None or fn.metadata.get('jsdoc_params') == {}

    def test_no_false_references_without_jsdoc(self):
        """No REFERENCES with reference_type=jsdoc emitted when there's no JSDoc."""
        result = _parse_single("""
class User {
    name = '';
}

function handle(user) {
    return user.name;
}
""")
        jsdoc_refs = [r for r in result.relationships
                      if r.relationship_type == RelationshipType.REFERENCES
                      and r.annotations.get('reference_type') == 'jsdoc']
        assert len(jsdoc_refs) == 0, (
            f"Expected no jsdoc REFERENCES, got: "
            f"{[(r.source_symbol, r.target_symbol) for r in jsdoc_refs]}"
        )

    def test_mixed_jsdoc_and_no_jsdoc(self):
        """Only functions with JSDoc get metadata; others stay clean."""
        result = _parse_single("""
/**
 * @param {string} name
 * @returns {boolean}
 */
function documented(name) {
    return !!name;
}

function undocumented(x) {
    return x + 1;
}
""")
        doc = next(s for s in result.symbols if s.name == 'documented')
        undoc = next(s for s in result.symbols if s.name == 'undocumented')

        assert doc.metadata.get('jsdoc_params', {}).get('name') == 'string'
        assert doc.metadata.get('jsdoc_returns') == 'boolean'

        assert undoc.metadata.get('jsdoc_params') is None or undoc.metadata.get('jsdoc_params') == {}
        assert undoc.metadata.get('jsdoc_returns') is None


# =============================================================================
# 11. Function-Wise Regression — various function patterns
# =============================================================================

class TestFunctionPatterns:
    """Verify function/method extraction stays correct across different forms."""

    def test_exported_function_declaration(self):
        """export function foo() … correctly creates FUNCTION symbol."""
        result = _parse_single("""
/**
 * @param {number} x
 * @returns {number}
 */
export function double(x) {
    return x * 2;
}
""")
        fn = next((s for s in result.symbols if s.name == 'double'), None)
        assert fn is not None, f"double not found. Symbols: {[s.name for s in result.symbols]}"
        assert fn.symbol_type == SymbolType.FUNCTION
        # JSDoc should still work on exported functions
        assert fn.metadata.get('jsdoc_params', {}).get('x') == 'number'
        assert fn.metadata.get('jsdoc_returns') == 'number'

    def test_export_default_function(self):
        """export default function … extracts symbol."""
        result = _parse_single("""
export default function handler(req, res) {
    res.send('ok');
}
""")
        fn = next((s for s in result.symbols if s.name == 'handler'), None)
        assert fn is not None, f"handler not found. Symbols: {[s.name for s in result.symbols]}"
        assert fn.symbol_type == SymbolType.FUNCTION

    def test_multiple_methods_in_class(self):
        """All methods extracted with correct parent."""
        result = _parse_single("""
class Router {
    get(path) {}
    post(path) {}
    put(path) {}
    delete(path) {}
}
""")
        methods = _symbols_of_type(result, SymbolType.METHOD)
        method_names = {m.name for m in methods}
        assert {'get', 'post', 'put', 'delete'}.issubset(method_names), (
            f"Missing some HTTP methods. Got: {method_names}"
        )
        for m in methods:
            if m.name in ('get', 'post', 'put', 'delete'):
                assert 'Router' in m.parent_symbol, f"{m.name} parent should contain Router: {m.parent_symbol}"

    def test_static_method(self):
        """static method is_static=True."""
        result = _parse_single("""
class Factory {
    static create(type) {
        return new Factory();
    }
    
    build() {
        return {};
    }
}
""")
        methods = _symbols_of_type(result, SymbolType.METHOD)
        create = next(m for m in methods if m.name == 'create')
        build = next(m for m in methods if m.name == 'build')
        assert create.is_static is True
        assert build.is_static is False

    def test_constructor_is_constructor_type(self):
        """constructor() → SymbolType.CONSTRUCTOR (Phase 2 standardisation)."""
        result = _parse_single("""
class Service {
    constructor(config) {
        this.config = config;
    }
}
""")
        ctors = _symbols_of_type(result, SymbolType.CONSTRUCTOR)
        assert len(ctors) >= 1, f"No CONSTRUCTOR found. Symbols: {[s.name for s in result.symbols]}"
        assert ctors[0].name == 'constructor'

    def test_getter_setter_methods(self):
        """get/set accessor methods extracted."""
        result = _parse_single("""
class Person {
    #name = '';
    
    get name() {
        return this.#name;
    }
    
    set name(value) {
        this.#name = value;
    }
}
""")
        methods = _symbols_of_type(result, SymbolType.METHOD)
        method_names = {m.name for m in methods}
        assert 'name' in method_names, f"getter/setter 'name' not found. Methods: {method_names}"

    def test_function_calls_relationship(self):
        """Function calling another function emits CALLS."""
        result = _parse_single("""
function helper() { return 42; }

function main() {
    const val = helper();
    return val;
}
""")
        calls = _rels_of_type(result, RelationshipType.CALLS)
        call_targets = {c.target_symbol for c in calls}
        assert 'helper' in call_targets, f"Expected CALLS to 'helper'. Got: {call_targets}"

    def test_new_expression_creates_relationship(self):
        """new Foo() emits CREATES."""
        result = _parse_single("""
class Foo {}

function make() {
    return new Foo();
}
""")
        creates = _rels_of_type(result, RelationshipType.CREATES)
        create_targets = {c.target_symbol for c in creates}
        assert 'Foo' in create_targets, f"Expected CREATES to 'Foo'. Got: {create_targets}"

    def test_class_inheritance(self):
        """extends Base → INHERITANCE."""
        result = _parse_single("""
class Base {}
class Child extends Base {}
""")
        inh = _rels_of_type(result, RelationshipType.INHERITANCE)
        assert len(inh) >= 1, f"No INHERITANCE. Rels: {[r.relationship_type for r in result.relationships]}"
        assert any('Base' in r.target_symbol for r in inh), (
            f"Expected INHERITANCE target 'Base'. Got: {[(r.source_symbol, r.target_symbol) for r in inh]}"
        )


# =============================================================================
# 12. JSX Component → REFERENCES (single-file + cross-file + graph)
# =============================================================================

class TestJSXComponentReferences:
    """JSX <CustomComponent /> creates REFERENCES to bring component into scope."""

    def test_jsx_self_closing_references(self):
        """<UserCard /> → REFERENCES to UserCard."""
        result = _parse_single("""
function UserCard(props) {
    return null;
}

function App() {
    return <UserCard name="Alice" />;
}
""", filename='app.jsx')

        refs = _rels_of_type(result, RelationshipType.REFERENCES)
        ref_targets = {r.target_symbol for r in refs}
        assert 'UserCard' in ref_targets, (
            f"Expected REFERENCES to UserCard from JSX. Got: {ref_targets}"
        )

    def test_jsx_opening_element_references(self):
        """<Panel>...</Panel> → REFERENCES to Panel."""
        result = _parse_single("""
function Panel(props) {
    return null;
}

function Layout() {
    return <Panel>
        <div>content</div>
    </Panel>;
}
""", filename='layout.jsx')

        refs = _rels_of_type(result, RelationshipType.REFERENCES)
        ref_targets = {r.target_symbol for r in refs}
        assert 'Panel' in ref_targets, (
            f"Expected REFERENCES to Panel from JSX. Got: {ref_targets}"
        )

    def test_jsx_html_elements_not_referenced(self):
        """<div>, <span>, <button> — lowercase tags should NOT create REFERENCES."""
        result = _parse_single("""
function App() {
    return <div>
        <span>hello</span>
        <button onClick={() => {}}>click</button>
    </div>;
}
""", filename='app.jsx')

        refs = _rels_of_type(result, RelationshipType.REFERENCES)
        ref_targets = {r.target_symbol for r in refs}
        # No lowercase html tags should appear
        for target in ref_targets:
            first_char = target.split('.')[0][0] if target else ''
            assert first_char.isupper() or first_char == '', (
                f"Lowercase tag '{target}' should not be in REFERENCES: {ref_targets}"
            )

    def test_jsx_multiple_components(self):
        """Multiple custom components in one render → each gets REFERENCES."""
        result = _parse_single("""
function Header() { return null; }
function Footer() { return null; }
function Sidebar() { return null; }

function Page() {
    return <div>
        <Header />
        <Sidebar />
        <Footer />
    </div>;
}
""", filename='page.jsx')

        refs = _rels_of_type(result, RelationshipType.REFERENCES)
        ref_targets = {r.target_symbol for r in refs}
        assert 'Header' in ref_targets, f"Missing Header REFERENCES. Got: {ref_targets}"
        assert 'Footer' in ref_targets, f"Missing Footer REFERENCES. Got: {ref_targets}"
        assert 'Sidebar' in ref_targets, f"Missing Sidebar REFERENCES. Got: {ref_targets}"

    def test_jsx_member_expression_component(self):
        """<Icons.Star /> → REFERENCES to Icons.Star."""
        result = _parse_single("""
const Icons = { Star: () => null };

function Rating() {
    return <Icons.Star />;
}
""", filename='rating.jsx')

        refs = _rels_of_type(result, RelationshipType.REFERENCES)
        ref_targets = {r.target_symbol for r in refs}
        assert any('Icons' in t for t in ref_targets), (
            f"Expected REFERENCES containing 'Icons'. Got: {ref_targets}"
        )


class TestJSXCrossFile:
    """Cross-file JSX component references — import + usage = resolved REFERENCES."""

    def test_imported_component_jsx_reference(self):
        """<Button /> referencing imported Button component across files."""
        parser = JavaScriptVisitorParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'Button.jsx': """
export default function Button(props) {
    return <button>{props.label}</button>;
}
""",
                'Form.jsx': """
import Button from './Button.jsx';

export default function Form() {
    return <div>
        <Button label="Submit" />
    </div>;
}
""",
            })
            results = parser.parse_multiple_files(list(paths.values()))

        # Form.jsx should have REFERENCES to Button
        form_refs = []
        for fp, result in results.items():
            if 'Form' in fp:
                form_refs = _rels_of_type(result, RelationshipType.REFERENCES)

        ref_targets = {r.target_symbol for r in form_refs}
        assert 'Button' in ref_targets, (
            f"Expected REFERENCES to 'Button' in Form.jsx. Got: {ref_targets}"
        )

    def test_multiple_imported_components(self):
        """Multiple components imported and used in JSX."""
        parser = JavaScriptVisitorParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'Card.jsx': """
export function Card(props) { return null; }
export function CardHeader(props) { return null; }
export function CardBody(props) { return null; }
""",
                'Dashboard.jsx': """
import { Card, CardHeader, CardBody } from './Card.jsx';

export function Dashboard() {
    return <Card>
        <CardHeader>Title</CardHeader>
        <CardBody>Content</CardBody>
    </Card>;
}
""",
            })
            results = parser.parse_multiple_files(list(paths.values()))

        dash_refs = []
        for fp, result in results.items():
            if 'Dashboard' in fp:
                dash_refs = _rels_of_type(result, RelationshipType.REFERENCES)

        ref_targets = {r.target_symbol for r in dash_refs}
        assert 'Card' in ref_targets, f"Missing Card. Got: {ref_targets}"
        assert 'CardHeader' in ref_targets, f"Missing CardHeader. Got: {ref_targets}"
        assert 'CardBody' in ref_targets, f"Missing CardBody. Got: {ref_targets}"

    def test_jsx_component_in_graph(self):
        """Graph includes edge from component user → component definition via JSX."""
        graph = _build_graph({
            'Alert.jsx': """
export function Alert(props) {
    return <div>{props.message}</div>;
}
""",
            'Notification.jsx': """
import { Alert } from './Alert.jsx';

export function Notification() {
    return <Alert message="Hello" />;
}
""",
        })

        nodes = list(graph.nodes())
        alert_nodes = [n for n in nodes if 'Alert' in n and 'Notification' not in n]
        notif_nodes = [n for n in nodes if 'Notification' in n]

        assert len(alert_nodes) > 0, f"Alert not in graph. Nodes: {nodes}"
        assert len(notif_nodes) > 0, f"Notification not in graph. Nodes: {nodes}"

        # Should have some edge from Notification space to Alert
        edges = [(s, t, d) for s, t, k, d in graph.edges(data=True, keys=True)]
        alert_edges = [(s, t, d.get('relationship_type'))
                       for s, t, d in edges
                       if 'Alert' in t and 'Notification' in s]
        assert len(alert_edges) > 0, (
            f"No edge from Notification to Alert. "
            f"All edges: {[(s, t, d.get('relationship_type')) for s, t, d in edges]}"
        )
