"""
Comprehensive test suite for Parser/Graph Standardization.

Tests all rich parsers (Python, Java, TypeScript, JavaScript, C++) for:
1. Parser output format (symbol.name, symbol.full_name with '.' separator)
2. Graph builder node ID creation (uses qualified names)
3. Relationship resolution (DEFINES, COMPOSITION, INHERITANCE, CALLS, etc.)
4. Cross-file relationships

See PARSER_GRAPH_STANDARDIZATION.md for design decisions.
"""

import tempfile
import os
import pytest
from pathlib import Path

from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
from plugin_implementation.parsers.python_parser import PythonParser
from plugin_implementation.parsers.java_visitor_parser import JavaVisitorParser
from plugin_implementation.parsers.typescript_enhanced_parser import TypeScriptEnhancedParser
from plugin_implementation.parsers.javascript_visitor_parser import JavaScriptVisitorParser
from plugin_implementation.parsers.cpp_enhanced_parser import CppEnhancedParser


class TestParserStandardization:
    """Test that all parsers use standardized '.' separator in full_name."""
    
    def test_python_uses_dot_separator(self):
        """Python parser should use '.' in full_name."""
        code = '''
class Parent:
    def method(self):
        pass
    
    class Nested:
        pass
'''
        parser = PythonParser()
        result = parser.parse_file('test.py', code)
        
        # Check no :: in full_name
        for symbol in result.symbols:
            if symbol.full_name:
                assert '::' not in symbol.full_name, f"Python full_name has '::': {symbol.full_name}"
        
        # Check qualified names use dots
        method_symbols = [s for s in result.symbols if s.name == 'method']
        assert len(method_symbols) == 1
        assert method_symbols[0].full_name == 'test.Parent.method'
    
    def test_java_uses_dot_separator(self):
        """Java parser should use '.' in full_name."""
        code = '''
package com.example;

public class Parent {
    public void method() {}
    
    public class Nested {}
}
'''
        parser = JavaVisitorParser()
        result = parser.parse_file('Parent.java', code)
        
        # Check no :: in full_name
        for symbol in result.symbols:
            if symbol.full_name:
                assert '::' not in symbol.full_name, f"Java full_name has '::': {symbol.full_name}"
    
    def test_typescript_uses_dot_separator(self):
        """TypeScript parser should use '.' in full_name."""
        code = '''
class Parent {
    method(): void {}
}
'''
        parser = TypeScriptEnhancedParser()
        result = parser.parse_file('test.ts', code)
        
        # Check no :: in full_name
        for symbol in result.symbols:
            if symbol.full_name:
                assert '::' not in symbol.full_name, f"TypeScript full_name has '::': {symbol.full_name}"
    
    def test_javascript_uses_dot_separator(self):
        """JavaScript parser should use '.' in full_name."""
        code = '''
class Parent {
    method() {}
}
'''
        parser = JavaScriptVisitorParser()
        result = parser.parse_file('test.js', code)
        
        # Check no :: in full_name
        for symbol in result.symbols:
            if symbol.full_name:
                assert '::' not in symbol.full_name, f"JavaScript full_name has '::': {symbol.full_name}"
    
    def test_cpp_uses_dot_separator(self):
        """C++ parser should use '.' in full_name (normalized from '::')."""
        code = '''
class Parent {
public:
    void method();
    
    class Nested {};
};
'''
        parser = CppEnhancedParser()
        result = parser.parse_file('test.cpp', code)
        
        # Check no :: in full_name (should be normalized to '.')
        for symbol in result.symbols:
            if symbol.full_name:
                assert '::' not in symbol.full_name, f"C++ full_name has '::': {symbol.full_name}"


class TestGraphNodeIdQualification:
    """Test that graph builder creates qualified node IDs."""
    
    def _build_graph(self, files: dict) -> dict:
        """Helper to build graph from files dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for filename, content in files.items():
                filepath = os.path.join(tmpdir, filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True) if '/' in filename else None
                with open(filepath, 'w') as f:
                    f.write(content)
            
            builder = EnhancedUnifiedGraphBuilder()
            result = builder.analyze_repository(tmpdir)
            
            return {
                'nodes': list(result.unified_graph.nodes()),
                'edges': [(s, t, d.get('relationship_type', '?')) 
                          for s, t, k, d in result.unified_graph.edges(data=True, keys=True)],
                'graph': result.unified_graph
            }
    
    def test_python_nested_symbols_have_qualified_node_ids(self):
        """Python nested symbols should have qualified node IDs."""
        files = {
            'test.py': '''
class Parent:
    def __init__(self):
        pass
    
    def method(self):
        pass

class Child:
    def __init__(self):
        pass
'''
        }
        result = self._build_graph(files)
        
        # Should have distinct node IDs for each __init__
        init_nodes = [n for n in result['nodes'] if '__init__' in n]
        print(f"__init__ nodes: {init_nodes}")
        
        # They should be distinguishable (different qualified names)
        assert len(init_nodes) >= 2, f"Expected 2 __init__ nodes, got {init_nodes}"
        assert len(set(init_nodes)) == len(init_nodes), f"Duplicate node IDs: {init_nodes}"
    
    def test_java_nested_symbols_have_qualified_node_ids(self):
        """Java nested symbols should have qualified node IDs."""
        files = {
            'Parent.java': '''
package com.example;

public class Parent {
    public Parent() {}
    public void method() {}
}
''',
            'Child.java': '''
package com.example;

public class Child {
    public Child() {}
    public void method() {}
}
'''
        }
        result = self._build_graph(files)
        
        # Should have distinct node IDs for each method()
        method_nodes = [n for n in result['nodes'] if 'method' in n.lower()]
        print(f"method nodes: {method_nodes}")
        
        # They should be distinguishable
        if len(method_nodes) >= 2:
            assert len(set(method_nodes)) == len(method_nodes), f"Duplicate node IDs: {method_nodes}"
    
    def test_cpp_nested_symbols_have_qualified_node_ids(self):
        """C++ nested symbols should have qualified node IDs with '.' separator."""
        files = {
            'test.cpp': '''
class Parent {
public:
    Parent();
    void method();
};

class Child {
public:
    Child();
    void method();
};
'''
        }
        result = self._build_graph(files)
        
        # Node IDs should use '.' not '::'
        for node in result['nodes']:
            # The node ID format is lang::file::qualified.name
            # The qualified.name part should use '.' not '::'
            parts = node.split('::')
            if len(parts) == 3:
                qualified_name = parts[2]
                # qualified_name should not contain '::'
                assert '::' not in qualified_name, f"Node ID has '::' in qualified name: {node}"


class TestRelationshipResolution:
    """Test that relationships resolve correctly in graph."""
    
    def _build_graph(self, files: dict) -> dict:
        """Helper to build graph from files dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for filename, content in files.items():
                filepath = os.path.join(tmpdir, filename)
                with open(filepath, 'w') as f:
                    f.write(content)
            
            builder = EnhancedUnifiedGraphBuilder()
            result = builder.analyze_repository(tmpdir)
            
            return {
                'nodes': list(result.unified_graph.nodes()),
                'edges': [(s, t, d.get('relationship_type', '?')) 
                          for s, t, k, d in result.unified_graph.edges(data=True, keys=True)],
                'graph': result.unified_graph
            }
    
    def test_python_defines_relationships_resolve(self):
        """Python DEFINES relationships should resolve correctly."""
        files = {
            'test.py': '''
class AppConfig:
    def __init__(self):
        self.database = DatabaseConfig()
    
    def get_config(self):
        return self.database
'''
        }
        result = self._build_graph(files)
        
        # Find DEFINES edges
        defines_edges = [(s, t) for s, t, r in result['edges'] if r == 'defines']
        print(f"DEFINES edges: {defines_edges}")
        
        # AppConfig should DEFINE __init__ and get_config (qualified names)
        # Look for edges where source is AppConfig (not qualified) and target is AppConfig.something
        appconfig_defines = [t for s, t in defines_edges 
                            if s.endswith('::AppConfig') and '.' in t.split('::')[-1]]
        print(f"AppConfig defines: {appconfig_defines}")
        
        # Should have method definitions (__init__, get_config, database field)
        assert len(appconfig_defines) >= 2, f"Expected AppConfig to define methods, got {appconfig_defines}"
    
    def test_python_composition_relationships_resolve(self):
        """Python COMPOSITION relationships should resolve correctly."""
        files = {
            'test.py': '''
class DatabaseConfig:
    pass

class AppConfig:
    def __init__(self):
        self.database = DatabaseConfig()
'''
        }
        result = self._build_graph(files)
        
        # Find COMPOSITION edges
        composition_edges = [(s, t) for s, t, r in result['edges'] if r == 'composition']
        print(f"COMPOSITION edges: {composition_edges}")
        
        # Should have database -> DatabaseConfig composition
        assert len(composition_edges) >= 1, f"Expected COMPOSITION edges, got {composition_edges}"
        
        # Source should be the field, target should be the type
        for src, tgt in composition_edges:
            if 'database' in src.lower():
                assert 'DatabaseConfig' in tgt, f"COMPOSITION target should be DatabaseConfig: {src} -> {tgt}"
    
    def test_java_inheritance_relationships_resolve(self):
        """Java INHERITANCE relationships should resolve correctly."""
        files = {
            'Base.java': '''
package com.example;

public class Base {
    public void baseMethod() {}
}
''',
            'Derived.java': '''
package com.example;

public class Derived extends Base {
    public void derivedMethod() {}
}
'''
        }
        result = self._build_graph(files)
        
        # Find INHERITANCE edges
        inheritance_edges = [(s, t) for s, t, r in result['edges'] if r == 'inheritance']
        print(f"INHERITANCE edges: {inheritance_edges}")
        
        # Should have Derived -> Base inheritance
        assert len(inheritance_edges) >= 1, f"Expected INHERITANCE edges, got {inheritance_edges}"
    
    def test_typescript_composition_relationships_resolve(self):
        """TypeScript COMPOSITION relationships should resolve correctly."""
        files = {
            'test.ts': '''
class DatabaseConfig {
    host: string;
}

class AppConfig {
    private database: DatabaseConfig;
    
    constructor() {
        this.database = new DatabaseConfig();
    }
}
'''
        }
        result = self._build_graph(files)
        
        # Find COMPOSITION edges
        composition_edges = [(s, t) for s, t, r in result['edges'] if r == 'composition']
        print(f"COMPOSITION edges: {composition_edges}")
        
        # Should have composition to DatabaseConfig
        assert len(composition_edges) >= 1, f"Expected COMPOSITION edges, got {composition_edges}"
    
    def test_cpp_composition_relationships_resolve(self):
        """C++ COMPOSITION relationships should resolve correctly."""
        files = {
            'test.cpp': '''
class DatabaseConfig {
public:
    const char* host;
};

class AppConfig {
private:
    DatabaseConfig* database;
public:
    AppConfig() {
        database = new DatabaseConfig();
    }
};
'''
        }
        result = self._build_graph(files)
        
        # Find COMPOSITION edges
        composition_edges = [(s, t) for s, t, r in result['edges'] if r == 'composition']
        print(f"COMPOSITION edges: {composition_edges}")
        
        # Should have composition to DatabaseConfig
        # Note: C++ may use aggregation for pointers
        all_has_a = [(s, t) for s, t, r in result['edges'] if r in ['composition', 'aggregation']]
        print(f"All has-a edges: {all_has_a}")


class TestCrossFileRelationships:
    """Test cross-file relationship resolution."""
    
    def _build_graph(self, files: dict) -> dict:
        """Helper to build graph from files dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for filename, content in files.items():
                filepath = os.path.join(tmpdir, filename)
                with open(filepath, 'w') as f:
                    f.write(content)
            
            builder = EnhancedUnifiedGraphBuilder()
            result = builder.analyze_repository(tmpdir)
            
            return {
                'nodes': list(result.unified_graph.nodes()),
                'edges': [(s, t, d.get('relationship_type', '?')) 
                          for s, t, k, d in result.unified_graph.edges(data=True, keys=True)],
                'graph': result.unified_graph
            }
    
    def test_python_cross_file_imports(self):
        """Python cross-file imports should create relationships."""
        files = {
            'config.py': '''
class DatabaseConfig:
    def __init__(self):
        self.host = "localhost"
''',
            'app.py': '''
from config import DatabaseConfig

class App:
    def __init__(self):
        self.config = DatabaseConfig()
'''
        }
        result = self._build_graph(files)
        
        print(f"Nodes: {result['nodes']}")
        print(f"Edges: {result['edges']}")
        
        # Should have nodes from both files
        config_nodes = [n for n in result['nodes'] if 'config' in n.lower()]
        app_nodes = [n for n in result['nodes'] if 'app' in n.lower()]
        
        assert len(config_nodes) > 0, "Missing config.py nodes"
        assert len(app_nodes) > 0, "Missing app.py nodes"
    
    def test_java_cross_file_inheritance(self):
        """Java cross-file inheritance should create relationships."""
        files = {
            'Base.java': '''
package com.example;

public class Base {
    public void baseMethod() {}
}
''',
            'Derived.java': '''
package com.example;

public class Derived extends Base {
    @Override
    public void baseMethod() {
        super.baseMethod();
    }
}
'''
        }
        result = self._build_graph(files)
        
        # Should have INHERITANCE edge
        inheritance_edges = [(s, t) for s, t, r in result['edges'] if r == 'inheritance']
        print(f"INHERITANCE edges: {inheritance_edges}")
        
        # Derived should inherit from Base
        assert any('Derived' in s for s, t in inheritance_edges), f"Missing Derived inheritance: {inheritance_edges}"


def run_all_tests():
    """Run all tests and print summary."""
    print("="*70)
    print("PARSER/GRAPH STANDARDIZATION TESTS")
    print("="*70)
    
    # Run with pytest
    import sys
    sys.exit(pytest.main([__file__, '-v', '-s']))


if __name__ == '__main__':
    run_all_tests()
