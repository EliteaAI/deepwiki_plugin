"""
Phase 3 — Generic Decomposition Parity Tests.

Covers:
    P-1:  Python recursive type extraction for nested generics
    J-1:  Java recursive generic type parameter extraction
    Gap-11: C++ template decomposition (verify existing behavior)
    Cross-parser: Same nested pattern → same user-defined types found

See PLANNING_PARSER_REVIEW.md Phase 3 for details.
"""

import pytest

from plugin_implementation.parsers.base_parser import SymbolType, RelationshipType
from plugin_implementation.parsers.python_parser import PythonParser
from plugin_implementation.parsers.java_visitor_parser import JavaVisitorParser
from plugin_implementation.parsers.typescript_enhanced_parser import TypeScriptEnhancedParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _refs(result, source_filter=None):
    """Return REFERENCES relationships, optionally filtered by source."""
    out = [r for r in result.relationships
           if r.relationship_type == RelationshipType.REFERENCES]
    if source_filter:
        out = [r for r in out if source_filter in r.source_symbol]
    return out


def _ref_targets(result, source_filter=None):
    """Return set of target symbols from REFERENCES relationships."""
    return {r.target_symbol for r in _refs(result, source_filter)}


def _annotated_refs(result, source_filter=None):
    """Return only REFERENCES that have a non-empty reference_type annotation."""
    return [r for r in _refs(result, source_filter)
            if r.annotations.get('reference_type')]


def _annotated_ref_targets(result, source_filter=None):
    """Return set of target symbols from annotated REFERENCES."""
    return {r.target_symbol for r in _annotated_refs(result, source_filter)}


# =============================================================================
# P-1: Python Recursive Type Extraction
# =============================================================================

class TestPythonRecursiveTypes:
    """P-1: Python parser must recursively decompose nested type annotations."""

    @pytest.fixture(autouse=True)
    def setup_parser(self):
        self.parser = PythonParser()

    # --- Nested generics ---

    def test_dict_list_nested(self):
        """Dict[str, List[User]] → finds User."""
        code = (
            "from typing import Dict, List\n"
            "class User: pass\n"
            "def fn(x: Dict[str, List[User]]): pass\n"
        )
        result = self.parser.parse_file('test.py', content=code)
        targets = _annotated_ref_targets(result, 'fn')
        assert 'User' in targets

    def test_optional_mapping_set(self):
        """Optional[Mapping[str, Set[Widget]]] → finds Widget."""
        code = (
            "from typing import Optional, Mapping, Set\n"
            "class Widget: pass\n"
            "def fn() -> Optional[Mapping[str, Set[Widget]]]: pass\n"
        )
        result = self.parser.parse_file('test.py', content=code)
        targets = _annotated_ref_targets(result, 'fn')
        assert 'Widget' in targets

    def test_deeply_nested_three_levels(self):
        """Dict[str, Dict[str, List[Entity]]] → finds Entity."""
        code = (
            "from typing import Dict, List\n"
            "class Entity: pass\n"
            "def fn(x: Dict[str, Dict[str, List[Entity]]]): pass\n"
        )
        result = self.parser.parse_file('test.py', content=code)
        targets = _annotated_ref_targets(result, 'fn')
        assert 'Entity' in targets

    def test_multiple_user_types_in_dict(self):
        """Dict[KeyType, ValueType] → finds both KeyType and ValueType."""
        code = (
            "from typing import Dict\n"
            "class KeyType: pass\n"
            "class ValueType: pass\n"
            "def fn(x: Dict[KeyType, ValueType]): pass\n"
        )
        result = self.parser.parse_file('test.py', content=code)
        targets = _annotated_ref_targets(result, 'fn')
        assert 'KeyType' in targets
        assert 'ValueType' in targets

    # --- PEP 604 unions ---

    def test_pep604_union(self):
        """Admin | Guest | None → finds Admin, Guest (not None)."""
        code = (
            "class Admin: pass\n"
            "class Guest: pass\n"
            "def fn(user: Admin | Guest | None): pass\n"
        )
        result = self.parser.parse_file('test.py', content=code)
        targets = _annotated_ref_targets(result, 'fn')
        assert 'Admin' in targets
        assert 'Guest' in targets
        assert 'None' not in targets

    def test_typing_union(self):
        """Union[Admin, Guest, None] → finds Admin, Guest."""
        code = (
            "from typing import Union\n"
            "class Admin: pass\n"
            "class Guest: pass\n"
            "def fn(user: Union[Admin, Guest, None]): pass\n"
        )
        result = self.parser.parse_file('test.py', content=code)
        targets = _annotated_ref_targets(result, 'fn')
        assert 'Admin' in targets
        assert 'Guest' in targets

    # --- Return types ---

    def test_return_type_nested(self):
        """Return type Optional[Result] → finds Result."""
        code = (
            "from typing import Optional\n"
            "class Result: pass\n"
            "def fn() -> Optional[Result]: pass\n"
        )
        result = self.parser.parse_file('test.py', content=code)
        targets = _annotated_ref_targets(result, 'fn')
        assert 'Result' in targets

    def test_return_type_deeply_nested(self):
        """Return type Dict[str, List[Result]] → finds Result."""
        code = (
            "from typing import Dict, List\n"
            "class Result: pass\n"
            "def fn() -> Dict[str, List[Result]]: pass\n"
        )
        result = self.parser.parse_file('test.py', content=code)
        targets = _annotated_ref_targets(result, 'fn')
        assert 'Result' in targets

    # --- Builtins filtering ---

    def test_primitives_filtered(self):
        """str, int, float, bool should not appear as REFERENCES."""
        code = "def fn(a: str, b: int, c: float, d: bool): pass\n"
        result = self.parser.parse_file('test.py', content=code)
        targets = _annotated_ref_targets(result, 'fn')
        assert 'str' not in targets
        assert 'int' not in targets
        assert 'float' not in targets
        assert 'bool' not in targets

    def test_typing_builtins_filtered(self):
        """Dict, List, Optional should not appear in annotated REFERENCES."""
        code = (
            "from typing import Dict, List, Optional\n"
            "class User: pass\n"
            "def fn(x: Dict[str, List[User]]) -> Optional[User]: pass\n"
        )
        result = self.parser.parse_file('test.py', content=code)
        targets = _annotated_ref_targets(result, 'fn')
        assert 'Dict' not in targets
        assert 'List' not in targets
        assert 'Optional' not in targets
        assert 'User' in targets

    # --- Annotation reference_type tracking ---

    def test_parameter_annotation(self):
        """Parameter type refs should have reference_type starting with 'parameter_type'."""
        code = (
            "class User: pass\n"
            "def fn(x: User): pass\n"
        )
        result = self.parser.parse_file('test.py', content=code)
        ann_refs = _annotated_refs(result, 'fn')
        user_refs = [r for r in ann_refs if r.target_symbol == 'User']
        assert len(user_refs) >= 1
        assert user_refs[0].annotations['reference_type'].startswith('parameter_type')

    def test_return_annotation(self):
        """Return type refs should have reference_type starting with 'return_type'."""
        code = (
            "class Result: pass\n"
            "def fn() -> Result: pass\n"
        )
        result = self.parser.parse_file('test.py', content=code)
        ann_refs = _annotated_refs(result, 'fn')
        result_refs = [r for r in ann_refs if r.target_symbol == 'Result']
        assert len(result_refs) >= 1
        assert result_refs[0].annotations['reference_type'].startswith('return_type')

    def test_nested_generic_arg_annotation(self):
        """Nested generic arg should have '_generic_arg' in reference_type."""
        code = (
            "from typing import List\n"
            "class User: pass\n"
            "def fn(x: List[User]): pass\n"
        )
        result = self.parser.parse_file('test.py', content=code)
        ann_refs = _annotated_refs(result, 'fn')
        user_refs = [r for r in ann_refs if r.target_symbol == 'User']
        assert len(user_refs) >= 1
        assert 'generic_arg' in user_refs[0].annotations['reference_type']

    # --- Dotted types ---

    def test_dotted_type_attribute(self):
        """models.User → emits 'User' (leaf attribute)."""
        code = (
            "import models\n"
            "def fn(x: models.User): pass\n"
        )
        result = self.parser.parse_file('test.py', content=code)
        targets = _annotated_ref_targets(result, 'fn')
        assert 'User' in targets

    # --- Callable ---

    def test_callable_args(self):
        """Callable[[User], Result] → finds User and Result."""
        code = (
            "from typing import Callable\n"
            "class User: pass\n"
            "class Result: pass\n"
            "def fn(cb: Callable[[User], Result]): pass\n"
        )
        result = self.parser.parse_file('test.py', content=code)
        targets = _annotated_ref_targets(result, 'fn')
        assert 'User' in targets
        assert 'Result' in targets


# =============================================================================
# J-1: Java Recursive Generic Type Parameters
# =============================================================================

class TestJavaRecursiveGenerics:
    """J-1: Java parser must recursively find types in nested generics."""

    @pytest.fixture(autouse=True)
    def setup_parser(self):
        self.parser = JavaVisitorParser()

    # --- Method return types ---

    def test_return_optional_set_widget(self):
        """Optional<Set<Widget>> → finds Widget."""
        code = (
            "package test;\n"
            "class Widget {}\n"
            "class Service {\n"
            "    public Optional<Set<Widget>> getWidgets() { return null; }\n"
            "}\n"
        )
        result = self.parser.parse_file('Test.java', content=code)
        targets = _ref_targets(result, 'getWidgets')
        assert 'Widget' in targets

    def test_return_map_list_user(self):
        """Map<String, List<User>> → finds User."""
        code = (
            "package test;\n"
            "class User {}\n"
            "class Service {\n"
            "    public Map<String, List<User>> getUsers() { return null; }\n"
            "}\n"
        )
        result = self.parser.parse_file('Test.java', content=code)
        targets = _ref_targets(result, 'getUsers')
        assert 'User' in targets

    # --- Method parameters ---

    def test_param_map_list_user(self):
        """Parameter Map<String, List<User>> → finds User."""
        code = (
            "package test;\n"
            "class User {}\n"
            "class Service {\n"
            "    public void process(Map<String, List<User>> users) {}\n"
            "}\n"
        )
        result = self.parser.parse_file('Test.java', content=code)
        targets = _ref_targets(result, 'process')
        assert 'User' in targets

    def test_param_deeply_nested(self):
        """Map<String, Map<String, List<Entity>>> → finds Entity."""
        code = (
            "package test;\n"
            "class Entity {}\n"
            "class Service {\n"
            "    public void process(Map<String, Map<String, List<Entity>>> data) {}\n"
            "}\n"
        )
        result = self.parser.parse_file('Test.java', content=code)
        targets = _ref_targets(result, 'process')
        assert 'Entity' in targets

    # --- Field declarations ---

    def test_field_map_list_user_composition(self):
        """Field Map<String, List<User>> → User found via COMPOSITION."""
        code = (
            "package test;\n"
            "class User {}\n"
            "class Service {\n"
            "    private Map<String, List<User>> cache;\n"
            "}\n"
        )
        result = self.parser.parse_file('Test.java', content=code)
        comps = [r for r in result.relationships
                 if r.relationship_type == RelationshipType.COMPOSITION]
        comp_targets = {r.target_symbol for r in comps}
        assert 'User' in comp_targets

    # --- Wildcards ---

    def test_wildcard_extends(self):
        """List<? extends MyEntity> → finds MyEntity."""
        code = (
            "package test;\n"
            "class MyEntity {}\n"
            "class Service {\n"
            "    public void process(List<? extends MyEntity> items) {}\n"
            "}\n"
        )
        result = self.parser.parse_file('Test.java', content=code)
        targets = _ref_targets(result, 'process')
        assert 'MyEntity' in targets

    def test_wildcard_with_nested_generic(self):
        """Map<String, ? extends List<Widget>> → finds Widget."""
        code = (
            "package test;\n"
            "class Widget {}\n"
            "class Service {\n"
            "    public void process(Map<String, ? extends List<Widget>> items) {}\n"
            "}\n"
        )
        result = self.parser.parse_file('Test.java', content=code)
        targets = _ref_targets(result, 'process')
        assert 'Widget' in targets

    # --- Array types in generics ---

    def test_generic_with_array(self):
        """List<User[]> → finds User."""
        code = (
            "package test;\n"
            "class User {}\n"
            "class Service {\n"
            "    public void process(List<User[]> items) {}\n"
            "}\n"
        )
        result = self.parser.parse_file('Test.java', content=code)
        targets = _ref_targets(result, 'process')
        assert 'User' in targets

    # --- Builtins filtering ---

    def test_builtins_not_in_generic_params(self):
        """String, Integer should not appear in generic param extraction."""
        code = (
            "package test;\n"
            "class User {}\n"
            "class Service {\n"
            "    public Map<String, List<User>> getUsers() { return null; }\n"
            "}\n"
        )
        result = self.parser.parse_file('Test.java', content=code)
        targets = _ref_targets(result, 'getUsers')
        assert 'String' not in targets
        assert 'Integer' not in targets

    # --- Local variables ---

    def test_local_variable_nested_generic(self):
        """Local var Map<String, List<User>> → finds User."""
        code = (
            "package test;\n"
            "class User {}\n"
            "class Service {\n"
            "    public void doWork() {\n"
            "        Map<String, List<User>> cache = new HashMap<>();\n"
            "    }\n"
            "}\n"
        )
        result = self.parser.parse_file('Test.java', content=code)
        targets = _ref_targets(result, 'doWork')
        assert 'User' in targets

    # --- Multiple user types ---

    def test_multiple_user_types_in_map(self):
        """Map<KeyType, ValueType> → finds both."""
        code = (
            "package test;\n"
            "class KeyType {}\n"
            "class ValueType {}\n"
            "class Service {\n"
            "    public Map<KeyType, ValueType> getMap() { return null; }\n"
            "}\n"
        )
        result = self.parser.parse_file('Test.java', content=code)
        targets = _ref_targets(result, 'getMap')
        assert 'KeyType' in targets
        assert 'ValueType' in targets


# =============================================================================
# Gap-11: C++ Template Decomposition (verify existing behavior)
# =============================================================================

class TestCppTemplateRecursion:
    """Gap-11: C++ RelationshipExtractor already recursively decomposes templates."""

    @pytest.fixture(autouse=True)
    def setup_parser(self):
        try:
            from plugin_implementation.parsers.cpp_enhanced_parser import CppEnhancedParser
            self.parser = CppEnhancedParser()
            self.available = True
        except ImportError:
            self.available = False

    def test_vector_shared_ptr_point(self):
        """vector<shared_ptr<Point>> → finds Point."""
        if not self.available:
            pytest.skip("C++ parser not available")
        code = (
            "#include <vector>\n"
            "#include <memory>\n"
            "\n"
            "class Point {};\n"
            "\n"
            "class Canvas {\n"
            "public:\n"
            "    std::vector<std::shared_ptr<Point>> points;\n"
            "};\n"
        )
        result = self.parser.parse_file('test.cpp', content=code)
        all_targets = {r.target_symbol for r in result.relationships
                       if r.relationship_type == RelationshipType.REFERENCES}
        assert 'Point' in all_targets

    def test_map_string_vector_widget(self):
        """map<string, vector<Widget>> → finds Widget."""
        if not self.available:
            pytest.skip("C++ parser not available")
        code = (
            "#include <map>\n"
            "#include <vector>\n"
            "#include <string>\n"
            "\n"
            "class Widget {};\n"
            "\n"
            "class Registry {\n"
            "public:\n"
            "    std::map<std::string, std::vector<Widget>> widgets;\n"
            "};\n"
        )
        result = self.parser.parse_file('test.cpp', content=code)
        all_targets = {r.target_symbol for r in result.relationships
                       if r.relationship_type == RelationshipType.REFERENCES}
        assert 'Widget' in all_targets


# =============================================================================
# Cross-Parser Parity
# =============================================================================

class TestCrossParserParity:
    """Same nested generic pattern → same user-defined types found."""

    @pytest.fixture(autouse=True)
    def setup_parsers(self):
        self.ts_parser = TypeScriptEnhancedParser()
        self.py_parser = PythonParser()
        self.java_parser = JavaVisitorParser()

    def test_nested_list_user_all_parsers(self):
        """All parsers find 'User' in a nested container<container<User>> pattern."""
        # TypeScript
        ts_code = (
            "interface User {}\n"
            "function fn(x: Map<string, Array<User>>): void {}\n"
        )
        ts_result = self.ts_parser.parse_file('test.ts', content=ts_code)
        ts_targets = _ref_targets(ts_result, 'fn')

        # Python
        py_code = (
            "from typing import Dict, List\n"
            "class User: pass\n"
            "def fn(x: Dict[str, List[User]]): pass\n"
        )
        py_result = self.py_parser.parse_file('test.py', content=py_code)
        # Use annotated refs for Python to filter out catch-all visit_Name noise
        py_targets = _annotated_ref_targets(py_result, 'fn')

        # Java
        java_code = (
            "package test;\n"
            "class User {}\n"
            "class Service {\n"
            "    public void fn(Map<String, List<User>> x) {}\n"
            "}\n"
        )
        java_result = self.java_parser.parse_file('Test.java', content=java_code)
        java_targets = _ref_targets(java_result, 'fn')

        # All three must find User
        assert 'User' in ts_targets, f"TypeScript missed User: {ts_targets}"
        assert 'User' in py_targets, f"Python missed User: {py_targets}"
        assert 'User' in java_targets, f"Java missed User: {java_targets}"

    def test_return_type_nested_all_parsers(self):
        """All parsers find 'Result' in a return type like Optional<Result>."""
        # TypeScript
        ts_code = (
            "interface Result {}\n"
            "function fn(): Promise<Result> { return null; }\n"
        )
        ts_result = self.ts_parser.parse_file('test.ts', content=ts_code)
        ts_targets = _ref_targets(ts_result, 'fn')

        # Python
        py_code = (
            "from typing import Optional\n"
            "class Result: pass\n"
            "def fn() -> Optional[Result]: pass\n"
        )
        py_result = self.py_parser.parse_file('test.py', content=py_code)
        py_targets = _annotated_ref_targets(py_result, 'fn')

        # Java
        java_code = (
            "package test;\n"
            "class Result {}\n"
            "class Service {\n"
            "    public Optional<Result> fn() { return null; }\n"
            "}\n"
        )
        java_result = self.java_parser.parse_file('Test.java', content=java_code)
        java_targets = _ref_targets(java_result, 'fn')

        assert 'Result' in ts_targets, f"TypeScript missed Result: {ts_targets}"
        assert 'Result' in py_targets, f"Python missed Result: {py_targets}"
        assert 'Result' in java_targets, f"Java missed Result: {java_targets}"
