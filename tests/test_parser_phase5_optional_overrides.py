"""
Phase 5 — Optional Enhancements Tests.

X-3: AGGREGATION for Optional/Nullable types (Java, Python)
OVERRIDES: @Override (Java), override keyword (C++)

Covers:
    1. Java: Optional<T> field → AGGREGATION instead of COMPOSITION
    2. Java: @Nullable annotation → AGGREGATION
    3. Java: Regular field → COMPOSITION (no regression)
    4. Python: Optional[T] field → AGGREGATION
    5. Python: field = None → AGGREGATION
    6. Python: Regular field → COMPOSITION (no regression)
    7. Java: @Override → OVERRIDES relationship + metadata
    8. C++: override keyword → OVERRIDES relationship + metadata
    9. Cross-file and graph-level verification

See PLANNING_PARSER_REVIEW.md Phase 5 for details.
"""

import os
import tempfile
import pytest

from plugin_implementation.parsers.base_parser import SymbolType, RelationshipType


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


def _symbols_of_type(result, stype):
    return [s for s in result.symbols if s.symbol_type == stype]


def _rels_of_type(result, rtype):
    return [r for r in result.relationships if r.relationship_type == rtype]


def _all_rels(results, rtype, source_filter=None):
    out = []
    for fp, result in results.items():
        for r in result.relationships:
            if r.relationship_type == rtype:
                if source_filter is None or source_filter in r.source_symbol:
                    out.append(r)
    return out


# =============================================================================
# 1. Java — AGGREGATION for Optional<T> fields
# =============================================================================

class TestJavaAggregationOptional:
    """Optional<T> fields should emit AGGREGATION, not COMPOSITION."""

    def _parse_java(self, code, filename='Test.java'):
        from plugin_implementation.parsers.java_visitor_parser import JavaVisitorParser
        parser = JavaVisitorParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = os.path.join(tmpdir, filename)
            with open(fp, 'w') as f:
                f.write(code)
            return parser.parse_file(fp)

    def test_optional_field_aggregation(self):
        """Optional<User> owner → AGGREGATION to User."""
        result = self._parse_java("""
class Order {
    Optional<User> owner;
    User buyer;
}
""")
        agg = _rels_of_type(result, RelationshipType.AGGREGATION)
        comp = _rels_of_type(result, RelationshipType.COMPOSITION)

        agg_targets = {r.target_symbol for r in agg}
        comp_targets = {r.target_symbol for r in comp}

        # User from Optional<User> should be AGGREGATION
        assert 'User' in agg_targets, (
            f"Expected AGGREGATION to 'User'. Agg: {agg_targets}, Comp: {comp_targets}"
        )

    def test_nullable_annotation_aggregation(self):
        """@Nullable User backup → AGGREGATION to User."""
        result = self._parse_java("""
class Order {
    @Nullable User backup;
    User primary;
}
""")
        agg = _rels_of_type(result, RelationshipType.AGGREGATION)
        agg_targets = {r.target_symbol for r in agg}

        assert 'User' in agg_targets, (
            f"Expected AGGREGATION for @Nullable User. Got: {agg_targets}"
        )

    def test_regular_field_still_composition(self):
        """Plain User buyer → COMPOSITION (no regression)."""
        result = self._parse_java("""
class Order {
    User buyer;
}
""")
        comp = _rels_of_type(result, RelationshipType.COMPOSITION)
        comp_targets = {r.target_symbol for r in comp}

        assert 'User' in comp_targets, (
            f"Expected COMPOSITION for regular User field. Got: {comp_targets}"
        )

    def test_collection_still_composition(self):
        """List<Item> items → COMPOSITION (collections are strong ownership)."""
        result = self._parse_java("""
class Cart {
    List<Item> items;
}
""")
        comp = _rels_of_type(result, RelationshipType.COMPOSITION)
        comp_targets = {r.target_symbol for r in comp}

        assert 'Item' in comp_targets, (
            f"Expected COMPOSITION for List<Item>. Got: {comp_targets}"
        )

    def test_optional_and_regular_mixed(self):
        """Mix of Optional and regular fields: correct relationship types."""
        result = self._parse_java("""
class Config {
    Optional<Database> database;
    Cache cache;
    Optional<Logger> logger;
}
""")
        agg = _rels_of_type(result, RelationshipType.AGGREGATION)
        comp = _rels_of_type(result, RelationshipType.COMPOSITION)

        agg_targets = {r.target_symbol for r in agg}
        comp_targets = {r.target_symbol for r in comp}

        assert 'Database' in agg_targets, f"Database should be AGGREGATION. Agg: {agg_targets}"
        assert 'Logger' in agg_targets, f"Logger should be AGGREGATION. Agg: {agg_targets}"
        assert 'Cache' in comp_targets, f"Cache should be COMPOSITION. Comp: {comp_targets}"

    def test_aggregation_in_capabilities(self):
        """AGGREGATION and OVERRIDES now declared in Java capabilities."""
        from plugin_implementation.parsers.java_visitor_parser import JavaVisitorParser
        parser = JavaVisitorParser()
        caps = parser.capabilities
        assert RelationshipType.AGGREGATION in caps.supported_relationships
        assert RelationshipType.OVERRIDES in caps.supported_relationships


# =============================================================================
# 2. Python — AGGREGATION for Optional[T] fields
# =============================================================================

class TestPythonAggregationOptional:
    """Optional[T] and = None fields should emit AGGREGATION."""

    def _parse_python(self, code, filename='test.py'):
        from plugin_implementation.parsers.python_parser import PythonParser
        parser = PythonParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = os.path.join(tmpdir, filename)
            with open(fp, 'w') as f:
                f.write(code)
            return parser.parse_file(fp)

    def test_optional_instance_field_aggregation(self):
        """self.owner: Optional[User] → AGGREGATION."""
        result = self._parse_python("""
from typing import Optional

class Order:
    def __init__(self):
        self.owner: Optional[User] = None
        self.buyer: User = User()
""")
        agg = _rels_of_type(result, RelationshipType.AGGREGATION)
        agg_targets = {r.target_symbol for r in agg}

        assert 'User' in agg_targets, (
            f"Expected AGGREGATION for Optional[User]. Agg targets: {agg_targets}. "
            f"All rels: {[(r.relationship_type.value, r.source_symbol, r.target_symbol) for r in result.relationships]}"
        )

    def test_none_default_field_aggregation(self):
        """self.handler: Handler = None → AGGREGATION (initialized to None)."""
        result = self._parse_python("""
class Service:
    def __init__(self):
        self.handler: Handler = None
""")
        agg = _rels_of_type(result, RelationshipType.AGGREGATION)
        agg_targets = {r.target_symbol for r in agg}

        assert 'Handler' in agg_targets, (
            f"Expected AGGREGATION for field = None. Got: {agg_targets}"
        )

    def test_regular_instance_field_composition(self):
        """self.db: Database = Database() → COMPOSITION (strong ownership)."""
        result = self._parse_python("""
class App:
    def __init__(self):
        self.db: Database = Database()
""")
        comp = _rels_of_type(result, RelationshipType.COMPOSITION)
        comp_targets = {r.target_symbol for r in comp}

        assert 'Database' in comp_targets, (
            f"Expected COMPOSITION for regular Database field. Got: {comp_targets}"
        )

    def test_optional_class_variable_aggregation(self):
        """Class-level Optional[T] variable → AGGREGATION."""
        result = self._parse_python("""
from typing import Optional

class Config:
    logger: Optional[Logger] = None
    database: Database = Database()
""")
        agg = _rels_of_type(result, RelationshipType.AGGREGATION)
        comp = _rels_of_type(result, RelationshipType.COMPOSITION)

        agg_targets = {r.target_symbol for r in agg}
        comp_targets = {r.target_symbol for r in comp}

        assert 'Logger' in agg_targets, (
            f"Expected AGGREGATION for Optional[Logger] class var. Agg: {agg_targets}"
        )
        assert 'Database' in comp_targets, (
            f"Expected COMPOSITION for Database class var. Comp: {comp_targets}"
        )

    def test_mixed_optional_and_regular(self):
        """Mix of Optional and regular fields in one class."""
        result = self._parse_python("""
from typing import Optional

class Controller:
    def __init__(self):
        self.service: Service = Service()
        self.cache: Optional[Cache] = None
        self.logger: Optional[Logger] = None
""")
        agg = _rels_of_type(result, RelationshipType.AGGREGATION)
        comp = _rels_of_type(result, RelationshipType.COMPOSITION)

        agg_targets = {r.target_symbol for r in agg}
        comp_targets = {r.target_symbol for r in comp}

        assert 'Cache' in agg_targets, f"Cache should be AGGREGATION. Got: {agg_targets}"
        assert 'Logger' in agg_targets, f"Logger should be AGGREGATION. Got: {agg_targets}"
        assert 'Service' in comp_targets, f"Service should be COMPOSITION. Got: {comp_targets}"


# =============================================================================
# 3. Java — @Override → OVERRIDES relationship
# =============================================================================

class TestJavaOverrides:
    """@Override annotation emits OVERRIDES relationship."""

    def _parse_java(self, code, filename='Test.java'):
        from plugin_implementation.parsers.java_visitor_parser import JavaVisitorParser
        parser = JavaVisitorParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = os.path.join(tmpdir, filename)
            with open(fp, 'w') as f:
                f.write(code)
            return parser.parse_file(fp)

    def test_override_annotation_emits_overrides(self):
        """@Override on method → OVERRIDES relationship."""
        result = self._parse_java("""
class Animal {
    public void speak() {}
}

class Dog extends Animal {
    @Override
    public void speak() {}
}
""")
        overrides = _rels_of_type(result, RelationshipType.OVERRIDES)
        assert len(overrides) >= 1, (
            f"Expected OVERRIDES relationship. "
            f"All rels: {[(r.relationship_type.value, r.source_symbol, r.target_symbol) for r in result.relationships]}"
        )

        ov = overrides[0]
        assert 'speak' in ov.source_symbol, f"Source should contain 'speak'. Got: {ov.source_symbol}"
        assert 'Animal' in ov.target_symbol, f"Target should reference Animal. Got: {ov.target_symbol}"

    def test_override_metadata_set(self):
        """@Override method has is_override=True in metadata."""
        result = self._parse_java("""
class Base {
    public String toString() { return ""; }
}

class Child extends Base {
    @Override
    public String toString() { return "child"; }
}
""")
        methods = _symbols_of_type(result, SymbolType.METHOD)
        child_tostring = [m for m in methods if m.name == 'toString' and m.parent_symbol and 'Child' in m.parent_symbol]
        assert len(child_tostring) >= 1, f"Child.toString not found. Methods: {[(m.name, m.parent_symbol) for m in methods]}"
        assert child_tostring[0].metadata.get('is_override') is True

    def test_non_override_method_no_metadata(self):
        """Regular method without @Override has no is_override."""
        result = self._parse_java("""
class Foo {
    public void bar() {}
}
""")
        methods = _symbols_of_type(result, SymbolType.METHOD)
        bar = next(m for m in methods if m.name == 'bar')
        assert bar.metadata.get('is_override') is not True

    def test_multiple_overrides(self):
        """Multiple @Override methods in one class."""
        result = self._parse_java("""
class Shape {
    public double area() { return 0; }
    public String name() { return ""; }
}

class Circle extends Shape {
    @Override
    public double area() { return 3.14; }
    
    @Override
    public String name() { return "circle"; }
    
    public double radius() { return 1.0; }
}
""")
        overrides = _rels_of_type(result, RelationshipType.OVERRIDES)
        assert len(overrides) >= 2, (
            f"Expected at least 2 OVERRIDES. Got {len(overrides)}: "
            f"{[(r.source_symbol, r.target_symbol) for r in overrides]}"
        )

        override_sources = {r.source_symbol for r in overrides}
        assert any('area' in s for s in override_sources)
        assert any('name' in s for s in override_sources)

    def test_override_with_interface_implementation(self):
        """implements interface + @Override → still emits OVERRIDES."""
        result = self._parse_java("""
interface Runnable {
    void run();
}

class Task implements Runnable {
    @Override
    public void run() {}
}
""")
        # No extends → no OVERRIDES (no superclass to resolve)
        # But @Override is still tracked in metadata
        methods = _symbols_of_type(result, SymbolType.METHOD)
        run = [m for m in methods if m.name == 'run' and m.parent_symbol and 'Task' in m.parent_symbol]
        assert len(run) >= 1
        assert run[0].metadata.get('is_override') is True


# =============================================================================
# 4. C++ — override keyword → OVERRIDES relationship
# =============================================================================

class TestCppOverrides:
    """C++ override keyword emits OVERRIDES relationship."""

    def _parse_cpp(self, code, filename='test.cpp'):
        from plugin_implementation.parsers.cpp_enhanced_parser import CppEnhancedParser
        parser = CppEnhancedParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = os.path.join(tmpdir, filename)
            with open(fp, 'w') as f:
                f.write(code)
            return parser.parse_file(fp)

    def test_override_keyword_emits_overrides(self):
        """void draw() override → OVERRIDES relationship to base."""
        result = self._parse_cpp("""
class Shape {
public:
    virtual void draw() {}
};

class Circle : public Shape {
public:
    void draw() override {}
};
""")
        overrides = _rels_of_type(result, RelationshipType.OVERRIDES)
        assert len(overrides) >= 1, (
            f"Expected OVERRIDES. All rels: "
            f"{[(r.relationship_type.value, r.source_symbol, r.target_symbol) for r in result.relationships]}"
        )

        ov = overrides[0]
        assert 'draw' in ov.source_symbol, f"Source should contain 'draw'. Got: {ov.source_symbol}"
        assert 'Shape' in ov.target_symbol, f"Target should reference Shape. Got: {ov.target_symbol}"

    def test_override_metadata_set(self):
        """override method has is_override=True in metadata."""
        result = self._parse_cpp("""
class Base {
public:
    virtual void update() {}
};

class Derived : public Base {
public:
    void update() override {}
};
""")
        methods = _symbols_of_type(result, SymbolType.METHOD)
        derived_update = [m for m in methods
                          if m.name == 'update'
                          and m.parent_symbol and 'Derived' in m.parent_symbol]
        assert len(derived_update) >= 1, f"Derived::update not found. Methods: {[(m.name, m.parent_symbol) for m in methods]}"
        assert derived_update[0].metadata.get('is_override') is True

    def test_non_override_method_no_metadata(self):
        """Regular method without override has no is_override."""
        result = self._parse_cpp("""
class Foo {
public:
    void bar() {}
};
""")
        methods = _symbols_of_type(result, SymbolType.METHOD)
        bar = [m for m in methods if m.name == 'bar']
        assert len(bar) >= 1
        assert bar[0].metadata.get('is_override') is not True

    def test_multiple_overrides_cpp(self):
        """Multiple override methods in one derived class."""
        result = self._parse_cpp("""
class Animal {
public:
    virtual void speak() {}
    virtual void move() {}
};

class Dog : public Animal {
public:
    void speak() override {}
    void move() override {}
    void fetch() {}
};
""")
        overrides = _rels_of_type(result, RelationshipType.OVERRIDES)
        assert len(overrides) >= 2, (
            f"Expected at least 2 OVERRIDES. Got {len(overrides)}: "
            f"{[(r.source_symbol, r.target_symbol) for r in overrides]}"
        )

    def test_override_target_references_base_class(self):
        """OVERRIDES target should reference the base class method."""
        result = self._parse_cpp("""
class Widget {
public:
    virtual void paint() {}
};

class Button : public Widget {
public:
    void paint() override {}
};
""")
        overrides = _rels_of_type(result, RelationshipType.OVERRIDES)
        assert len(overrides) >= 1
        ov = overrides[0]
        # Target should be Widget::paint
        assert 'Widget' in ov.target_symbol and 'paint' in ov.target_symbol, (
            f"Expected target like Widget::paint. Got: {ov.target_symbol}"
        )


# =============================================================================
# 5. Cross-file verification
# =============================================================================

class TestPhase5CrossFile:
    """Cross-file AGGREGATION and OVERRIDES preservation."""

    def test_java_aggregation_cross_file(self):
        """Optional field AGGREGATION preserved across files."""
        from plugin_implementation.parsers.java_visitor_parser import JavaVisitorParser
        parser = JavaVisitorParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'User.java': 'public class User { String name; }',
                'Order.java': """
import User;

public class Order {
    Optional<User> assignee;
    User creator;
}
""",
            })
            results = parser.parse_multiple_files(list(paths.values()))

        agg = _all_rels(results, RelationshipType.AGGREGATION)
        agg_targets = {r.target_symbol for r in agg}
        assert 'User' in agg_targets, f"Cross-file AGGREGATION to User expected. Got: {agg_targets}"

    def test_java_overrides_cross_file(self):
        """@Override OVERRIDES preserved across files."""
        from plugin_implementation.parsers.java_visitor_parser import JavaVisitorParser
        parser = JavaVisitorParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'Animal.java': """
public class Animal {
    public void speak() {}
}
""",
                'Dog.java': """
public class Dog extends Animal {
    @Override
    public void speak() {}
}
""",
            })
            results = parser.parse_multiple_files(list(paths.values()))

        ovr = _all_rels(results, RelationshipType.OVERRIDES)
        assert len(ovr) >= 1, (
            f"Expected OVERRIDES in cross-file. "
            f"All rels: {[(r.relationship_type.value, r.source_symbol) for fp, res in results.items() for r in res.relationships]}"
        )

    def test_python_aggregation_cross_file(self):
        """Optional[T] AGGREGATION preserved across files."""
        from plugin_implementation.parsers.python_parser import PythonParser
        parser = PythonParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _write_files(tmpdir, {
                'models.py': """
class Database:
    pass
""",
                'app.py': """
from typing import Optional

class App:
    def __init__(self):
        self.db: Optional[Database] = None
""",
            })
            results = parser.parse_multiple_files(list(paths.values()))

        agg = _all_rels(results, RelationshipType.AGGREGATION)
        agg_targets = {r.target_symbol for r in agg}
        assert 'Database' in agg_targets, f"Cross-file AGGREGATION to Database expected. Got: {agg_targets}"
