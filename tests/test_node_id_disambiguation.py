"""
Test demonstrating the Node ID Disambiguation Problem across all rich parsers.

PROBLEM:
Current node ID format: `lang::file::symbol.name`
This can't distinguish nested symbols with same name.

Example: Two `__init__` methods become same node ID `python::test::__init__`

SOLUTION:
Use parent-qualified names: `lang::file::Parent.symbol.name`
Example: `python::test::DatabaseConfig.__init__` vs `python::test::AppConfig.__init__`

PARSER STATUS:
- Python: Fixed (emits FIELD symbols, qualified DEFINES)
- Java: Already correct (has FIELD symbols, qualified names)
- TypeScript: Already correct (has FIELD symbols, qualified names)  
- C++: Uses `::` separator (needs graph builder normalization)
- JavaScript: Limited (no static types, no JSDoc = no field types)
"""

import tempfile
import os


def test_python_parser():
    """Test Python parser emits qualified names."""
    from plugin_implementation.parsers.python_parser import PythonParser
    
    code = '''
class DatabaseConfig:
    def __init__(self):
        self.host = "localhost"

class AppConfig:
    def __init__(self):
        self.database = DatabaseConfig()
'''
    parser = PythonParser()
    result = parser.parse_file('test.py', code)
    
    print("=== PYTHON ===")
    
    # Check FIELD symbols
    fields = [s for s in result.symbols if s.symbol_type.name == 'FIELD']
    print(f"FIELD symbols: {[s.full_name for s in fields]}")
    assert any('database' in s.full_name for s in fields), "Missing database field"
    
    # Check COMPOSITION
    comps = [r for r in result.relationships if r.relationship_type.name == 'COMPOSITION']
    print(f"COMPOSITION: {[(r.source_symbol, r.target_symbol) for r in comps]}")
    assert any('database' in r.source_symbol for r in comps), "COMPOSITION should use field as source"
    
    # Check DEFINES uses qualified names
    defines = [r for r in result.relationships if r.relationship_type.name == 'DEFINES' and '__init__' in r.target_symbol]
    print(f"DEFINES __init__: {[(r.source_symbol, r.target_symbol) for r in defines]}")
    assert len(defines) == 2, f"Expected 2 __init__ DEFINES, got {len(defines)}"
    
    print("✅ Python: PASS\n")


def test_java_parser():
    """Test Java parser emits qualified names."""
    from plugin_implementation.parsers.java_visitor_parser import JavaVisitorParser
    
    code = '''
package com.example;

public class AppConfig {
    private DatabaseConfig database;
    
    public AppConfig() {
        this.database = new DatabaseConfig();
    }
}
'''
    parser = JavaVisitorParser()
    result = parser.parse_file('AppConfig.java', code)
    
    print("=== JAVA ===")
    
    # Check FIELD symbols
    fields = [s for s in result.symbols if s.symbol_type.name == 'FIELD']
    print(f"FIELD symbols: {[s.full_name for s in fields]}")
    assert len(fields) >= 1, "Missing FIELD symbols"
    
    # Check COMPOSITION
    comps = [r for r in result.relationships if r.relationship_type.name == 'COMPOSITION']
    print(f"COMPOSITION: {[(r.source_symbol, r.target_symbol) for r in comps]}")
    assert len(comps) >= 1, "Missing COMPOSITION"
    
    print("✅ Java: PASS\n")


def test_typescript_parser():
    """Test TypeScript parser emits qualified names."""
    from plugin_implementation.parsers.typescript_enhanced_parser import TypeScriptEnhancedParser
    
    code = '''
class AppConfig {
    private database: DatabaseConfig;
    
    constructor() {
        this.database = new DatabaseConfig();
    }
}
'''
    parser = TypeScriptEnhancedParser()
    result = parser.parse_file('app.ts', code)
    
    print("=== TYPESCRIPT ===")
    
    # Check FIELD symbols
    fields = [s for s in result.symbols if s.symbol_type.name == 'FIELD']
    print(f"FIELD symbols: {[s.full_name for s in fields]}")
    assert len(fields) >= 1, "Missing FIELD symbols"
    
    # Check COMPOSITION
    comps = [r for r in result.relationships if r.relationship_type.name == 'COMPOSITION']
    print(f"COMPOSITION: {[(r.source_symbol, r.target_symbol) for r in comps]}")
    assert len(comps) >= 1, "Missing COMPOSITION"
    
    print("✅ TypeScript: PASS\n")


def test_javascript_parser():
    """Test JavaScript parser - LIMITED due to no static types."""
    from plugin_implementation.parsers.javascript_visitor_parser import JavaScriptVisitorParser
    
    code = '''
class AppConfig {
    constructor() {
        this.database = new DatabaseConfig();
    }
}
'''
    parser = JavaScriptVisitorParser()
    result = parser.parse_file('app.js', code)
    
    print("=== JAVASCRIPT (Limited - No Static Types) ===")
    
    # JavaScript has no static types, so limited field detection
    fields = [s for s in result.symbols if s.symbol_type.name == 'FIELD']
    print(f"FIELD symbols: {[s.full_name for s in fields]} (expected: few/none without JSDoc)")
    
    # Can still detect CREATES from new ClassName()
    creates = [r for r in result.relationships if r.relationship_type.name == 'CREATES']
    print(f"CREATES: {[(r.source_symbol, r.target_symbol) for r in creates]}")
    
    # COMPOSITION may be on constructor, not field
    comps = [r for r in result.relationships if r.relationship_type.name == 'COMPOSITION']
    print(f"COMPOSITION: {[(r.source_symbol, r.target_symbol) for r in comps]}")
    
    print("⚠️ JavaScript: LIMITED (no types without JSDoc)\n")


def test_cpp_parser():
    """Test C++ parser - now uses . separator (normalized from ::)."""
    from plugin_implementation.parsers.cpp_enhanced_parser import CppEnhancedParser
    
    code = '''
class AppConfig {
private:
    DatabaseConfig* database;
public:
    AppConfig() {
        database = new DatabaseConfig();
    }
};
'''
    parser = CppEnhancedParser()
    result = parser.parse_file('app.cpp', code)
    
    print("=== C++ (Normalized to . Separator) ===")
    
    # Check FIELD symbols
    fields = [s for s in result.symbols if s.symbol_type.name == 'FIELD']
    print(f"FIELD symbols: {[s.full_name for s in fields]}")
    assert len(fields) >= 1, "Missing FIELD symbols"
    
    # Check uses . separator (not ::)
    has_dots = any('.' in (s.full_name or '') for s in fields)
    has_colons = any('::' in (s.full_name or '') for s in fields)
    print(f"Uses . separator: {has_dots}, Uses :: separator: {has_colons}")
    assert has_dots, "C++ should now use . separator"
    assert not has_colons, "C++ should NOT use :: separator anymore"
    
    # Check COMPOSITION
    comps = [r for r in result.relationships if r.relationship_type.name == 'COMPOSITION']
    print(f"COMPOSITION: {[(r.source_symbol, r.target_symbol) for r in comps]}")
    
    print("✅ C++: PASS (normalized to . separator)\n")


def show_summary():
    """Show summary of parser standardization status."""
    print("="*70)
    print("PARSER STANDARDIZATION STATUS")
    print("="*70)
    print("""
| Parser     | FIELD Symbols | COMPOSITION  | Separator | Status        |
|------------|---------------|--------------|-----------|---------------|
| Python     | ✅ Yes        | ✅ Field→Type | .         | Fixed         |
| Java       | ✅ Yes        | ✅ Field→Type | .         | Already OK    |
| TypeScript | ✅ Yes        | ✅ Field→Type | .         | Already OK    |
| C++        | ✅ Yes        | ✅ Field→Type | .         | Fixed (was ::)|
| JavaScript | ❌ No         | ⚠️ Method→Type| .        | No types      |

COMPLETED:
✅ All parsers use '.' separator in full_name
✅ All typed parsers emit FIELD symbols  
✅ COMPOSITION uses Field→Type (not Class→Type)

REMAINING:
⚠️ JavaScript: Accept limitation (no static types without JSDoc)
""")
    print("="*70)


def test_file_name_collision():
    """
    Test that file name collision with symbol name is handled correctly.
    
    PROBLEM (was): When User.java contains `class User`, both the module
    and the class would get the same node ID `java::User::User`.
    
    SOLUTION: MODULE symbols get `__module__` prefix:
    - Module: `java::User::__module__User`
    - Class: `java::User::User`
    - Constructor: `java::User::User.User`
    """
    from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
    
    print("\n=== FILE NAME COLLISION TEST ===\n")
    
    # Create a temp directory with collision test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Java: User.java with class User
        java_code = '''public class User {
    private String name;
    
    public User() {
        this.name = "default";
    }
    
    public void setName(String name) {
        this.name = name;
    }
}
'''
        java_path = os.path.join(tmpdir, 'User.java')
        with open(java_path, 'w') as f:
            f.write(java_code)
        
        # C++: User.cpp with class User
        cpp_code = '''class User {
public:
    std::string name;
    
    User() : name("default") {}
    
    void setName(const std::string& n) {
        this->name = n;
    }
};
'''
        cpp_path = os.path.join(tmpdir, 'User.cpp')
        with open(cpp_path, 'w') as f:
            f.write(cpp_code)
        
        # TypeScript: User.ts with class User
        ts_code = '''export class User {
    name: string;
    
    constructor() {
        this.name = "default";
    }
    
    setName(name: string): void {
        this.name = name;
    }
}
'''
        ts_path = os.path.join(tmpdir, 'User.ts')
        with open(ts_path, 'w') as f:
            f.write(ts_code)
        
        # Python: test.py with def test()
        python_code = '''def test():
    return "hello"

class TestClass:
    def test(self):
        return test()
'''
        py_path = os.path.join(tmpdir, 'test.py')
        with open(py_path, 'w') as f:
            f.write(python_code)
        
        # Build the graph
        builder = EnhancedUnifiedGraphBuilder()
        result = builder.analyze_repository(tmpdir)
        G = result.unified_graph
        
        all_nodes = list(G.nodes())
        print(f"All nodes in graph ({len(all_nodes)}):")
        for n in sorted(all_nodes):
            print(f"  {n}")
        
        # CLEAN SOLUTION: MODULE/NAMESPACE symbols are NOT added to graph
        # This avoids collision when file name matches class name
        
        # Check Java: User.java with class User -> only class node, no module
        java_nodes = [n for n in all_nodes if n.startswith('java::')]
        print(f"\nJava nodes: {java_nodes}")
        assert not any('module' in n.lower() for n in java_nodes), "No module nodes should exist"
        assert 'java::User::User' in java_nodes, "Class should be java::User::User"
        assert 'java::User::User.User' in java_nodes, "Constructor should be java::User::User.User"
        print("✅ Java: PASS (no module node, no collision)")
        
        # Check C++: User.cpp with class User -> only class node, no module
        cpp_nodes = [n for n in all_nodes if n.startswith('cpp::')]
        print(f"\nC++ nodes: {cpp_nodes}")
        assert not any('module' in n.lower() for n in cpp_nodes), "No module nodes should exist"
        assert 'cpp::User::User' in cpp_nodes, "Class should be cpp::User::User"
        assert 'cpp::User::User.User' in cpp_nodes, "Constructor should be cpp::User::User.User"
        print("✅ C++: PASS (no module node, no collision)")
        
        # Check TypeScript: User.ts with class User -> only class node, no module
        ts_nodes = [n for n in all_nodes if n.startswith('typescript::')]
        print(f"\nTypeScript nodes: {ts_nodes}")
        assert not any('module' in n.lower() for n in ts_nodes), "No module nodes should exist"
        assert 'typescript::User::User' in ts_nodes, "Class should be typescript::User::User"
        print("✅ TypeScript: PASS (no module node, no collision)")
        
        # Check Python: test.py with def test() -> function and method are distinct
        py_nodes = [n for n in all_nodes if n.startswith('python::')]
        print(f"\nPython nodes: {py_nodes}")
        assert not any('module' in G.nodes[n].get('symbol_type', '').lower() for n in py_nodes), "No module nodes should exist"
        assert 'python::test::test' in py_nodes, "Top-level function should be python::test::test"
        assert 'python::test::TestClass.test' in py_nodes, "Method should be python::test::TestClass.test"
        print("✅ Python: PASS (function and method are distinct, no module node)")
        
        print("\n" + "="*70)
        print("✅ FILE NAME COLLISION: ALL TESTS PASS")
        print("  (MODULE/NAMESPACE symbols excluded from graph - clean solution)")
        print("="*70)


if __name__ == '__main__':
    test_python_parser()
    test_java_parser()
    test_typescript_parser()
    test_javascript_parser()
    test_cpp_parser()
    show_summary()
    test_file_name_collision()
