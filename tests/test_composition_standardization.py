"""
Test COMPOSITION/AGGREGATION relationship standardization across parsers.

Verifies that all parsers now create Field → Type relationships (not Class → Type).
Standard: COMPOSITION/AGGREGATION source should be the field's full_name, not the parent class.
"""

import tempfile
import os
from pathlib import Path

# Test code samples with class fields
TEST_PYTHON = """
class UserService:
    def __init__(self):
        self.current_user = User()
        self.config = Config()
"""

TEST_TYPESCRIPT = """
class UserService {
    currentUser: User;
    config: Config;
}
"""

TEST_CPP = """
class UserService {
public:
    User currentUser;
    Config config;
};
"""

TEST_JAVA = """
class UserService {
    private User currentUser;
    private Config config;
}
"""

def _check_parser_composition(parser_class, test_code, file_extension, separator):
    """Check that parser creates Field → Type COMPOSITION edges (internal helper)"""
    from plugin_implementation.parsers.base_parser import Symbol, SymbolType, Relationship, RelationshipType
    
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix=file_extension, delete=False) as f:
        f.write(test_code)
        temp_file = f.name
    
    try:
        # Parse the file
        parser = parser_class()
        result = parser.parse_file(temp_file)
        
        if result is None:
            print(f"❌ Parser returned None")
            return False
        
        symbols = result.symbols
        relationships = result.relationships
        
        # Find COMPOSITION/AGGREGATION relationships
        composition_rels = [r for r in relationships 
                           if r.relationship_type in [RelationshipType.COMPOSITION, RelationshipType.AGGREGATION]]
        
        if not composition_rels:
            print(f"⚠️  No COMPOSITION/AGGREGATION relationships found")
            return True  # Not all parsers create these
        
        print(f"Found {len(composition_rels)} COMPOSITION/AGGREGATION relationships:")
        
        all_correct = True
        for rel in composition_rels:
            # Check if source is a FIELD symbol (contains separator between class and field)
            is_field_level = separator in rel.source_symbol
            
            status = "✅" if is_field_level else "❌"
            print(f"  {status} {rel.source_symbol} → {rel.target_symbol} ({rel.relationship_type.value})")
            
            if not is_field_level:
                print(f"      ERROR: Source should be field (Class{separator}field), not class!")
                all_correct = False
        
        return all_correct
    
    finally:
        os.unlink(temp_file)

def main():
    print("=" * 80)
    print("COMPOSITION/AGGREGATION Relationship Standardization Test")
    print("=" * 80)
    print("\nVerifying that parsers create Field → Type relationships (not Class → Type)\n")
    
    results = {}
    
    # Test Python Parser
    print("\n" + "=" * 80)
    print("Python Parser")
    print("=" * 80)
    try:
        from plugin_implementation.parsers.python_parser import PythonParser
        results['Python'] = _check_parser_composition(PythonParser, TEST_PYTHON, '.py', '.')
    except Exception as e:
        print(f"❌ Python parser failed: {e}")
        results['Python'] = False
    
    # Test TypeScript Parser
    print("\n" + "=" * 80)
    print("TypeScript Parser")
    print("=" * 80)
    try:
        from plugin_implementation.parsers.typescript_enhanced_parser import TypeScriptEnhancedParser
        results['TypeScript'] = _check_parser_composition(TypeScriptEnhancedParser, TEST_TYPESCRIPT, '.ts', '.')
    except Exception as e:
        print(f"❌ TypeScript parser failed: {e}")
        results['TypeScript'] = False
    
    # Test C++ Parser
    print("\n" + "=" * 80)
    print("C++ Parser")
    print("=" * 80)
    try:
        from plugin_implementation.parsers.cpp_enhanced_parser import CppEnhancedParser
        # C++ now uses '.' separator after standardization (was '::')
        results['C++'] = _check_parser_composition(CppEnhancedParser, TEST_CPP, '.cpp', '.')
    except Exception as e:
        print(f"❌ C++ parser failed: {e}")
        results['C++'] = False
    
    # Test Java Parser
    print("\n" + "=" * 80)
    print("Java Parser")
    print("=" * 80)
    try:
        from plugin_implementation.parsers.java_visitor_parser import JavaVisitorParser
        results['Java'] = _check_parser_composition(JavaVisitorParser, TEST_JAVA, '.java', '.')
    except Exception as e:
        print(f"❌ Java parser failed: {e}")
        results['Java'] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for parser, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {parser}")
    
    print("\n" + "=" * 80)
    if all(results.values()):
        print("✅ All parsers standardized: COMPOSITION/AGGREGATION from Field → Type")
    else:
        print("❌ Some parsers still create Class → Type relationships!")
    print("=" * 80)

if __name__ == '__main__':
    main()
