"""
Test C++ struct augmentation with DEFINES edges
Verify that ContentExpander collects method implementations correctly
"""

import sys
import networkx as nx
from plugin_implementation.content_expander import ContentExpander
from langchain_core.documents import Document

def create_test_graph():
    """Create a test graph simulating C++ struct with out-of-line methods using DEFINES_BODY"""
    graph = nx.MultiDiGraph()
    
    # Struct: Point (in point.h)
    graph.add_node("Point", 
        symbol_name="Point",
        symbol_type="struct",
        file_path="/project/point.h",
        rel_path="point.h",
        full_name="Point",
        parent_symbol=None,
        symbol=type('obj', (object,), {'source_text': 'struct Point { double x, y; };'})()
    )
    
    # Method declaration: Point::distance (in point.h, inline in struct)
    graph.add_node("Point::distance_decl",
        symbol_name="distance",
        symbol_type="method",
        file_path="/project/point.h",
        rel_path="point.h",
        full_name="Point::distance",
        parent_symbol="Point",
        symbol=type('obj', (object,), {'source_text': 'double distance() const;'})()
    )
    
    # Method implementation: Point::distance (in point.cpp, out-of-line)
    graph.add_node("Point::distance_impl",
        symbol_name="distance",
        symbol_type="method",
        file_path="/project/point.cpp",
        rel_path="point.cpp",
        full_name="Point::distance",
        parent_symbol="Point",
        symbol=type('obj', (object,), {
            'source_text': 'double Point::distance() const { return sqrt(x*x + y*y); }'
        })()
    )
    
    # Method implementation: Point::move (in point.cpp, out-of-line)
    graph.add_node("Point::move_impl",
        symbol_name="move",
        symbol_type="method",
        file_path="/project/point.cpp",
        rel_path="point.cpp",
        full_name="Point::move",
        parent_symbol="Point",
        symbol=type('obj', (object,), {
            'source_text': 'void Point::move(double dx, double dy) { x += dx; y += dy; }'
        })()
    )
    
    # Method declaration: Point::move (in point.h)
    graph.add_node("Point::move_decl",
        symbol_name="move",
        symbol_type="method",
        file_path="/project/point.h",
        rel_path="point.h",
        full_name="Point::move",
        parent_symbol="Point",
        symbol=type('obj', (object,), {'source_text': 'void move(double dx, double dy);'})()
    )
    
    # DEFINES edges: Point -> declarations (created by C++ parser for methods declared in class)
    graph.add_edge("Point", "Point::distance_decl", 
        relationship_type="defines",
        key="defines_0"
    )
    graph.add_edge("Point", "Point::move_decl",
        relationship_type="defines",
        key="defines_1"
    )
    
    # DEFINES_BODY edges: implementations -> declarations (created by C++ parser for out-of-line)
    # Week 6: ContentExpander now follows these!
    graph.add_edge("Point::distance_impl", "Point::distance_decl",
        relationship_type="defines_body",
        key="defines_body_0"
    )
    graph.add_edge("Point::move_impl", "Point::move_decl",
        relationship_type="defines_body",
        key="defines_body_1"
    )
    
    return graph

def test_cpp_struct_augmentation_current():
    """Test Week 6 behavior - follows DEFINES -> DEFINES_BODY chain"""
    print("=" * 80)
    print("TEST: C++ Struct Augmentation - Week 6 (DEFINES_BODY)")
    print("=" * 80)
    
    graph = create_test_graph()
    expander = ContentExpander(graph_store=graph)
    
    # Test _collect_cpp_method_implementations
    processed_nodes = set()
    impl_by_file = expander._collect_cpp_method_implementations("Point", processed_nodes)
    
    print(f"\nCollected implementations from {len(impl_by_file)} files:")
    for rel_path, methods in impl_by_file.items():
        print(f"  {rel_path}: {len(methods)} methods")
        for method in methods:
            print(f"    - {method[:50]}...")
    
    if len(impl_by_file) > 0 and 'point.cpp' in impl_by_file:
        print("\n✅ SUCCESS: Found out-of-line implementations!")
        print(f"   Strategy: Point -> DEFINES -> declarations -> DEFINES_BODY (incoming) -> implementations")
        print(f"   Found {len(impl_by_file['point.cpp'])} implementations in point.cpp")
        return True
    else:
        print("\n❌ FAILURE: Should find implementations via DEFINES_BODY edges")
        return False

def test_cpp_struct_augmentation_week6():
    """Test alternative: if parser also created direct DEFINES edges"""
    print("\n" + "=" * 80)
    print("TEST: Alternative - Direct DEFINES to Implementations")
    print("=" * 80)
    
    graph = create_test_graph()
    
    # Alternative: Add DEFINES edges from struct to implementations (bypassing declarations)
    # This is what the cpp_enhanced_parser.py change attempted
    graph.add_edge("Point", "Point::move_impl",
        relationship_type="defines",
        key="defines_2"
    )
    graph.add_edge("Point", "Point::distance_impl",
        relationship_type="defines",
        key="defines_3"
    )
    
    expander = ContentExpander(graph_store=graph)
    
    processed_nodes = set()
    impl_by_file = expander._collect_cpp_method_implementations("Point", processed_nodes)
    
    print(f"\nCollected implementations from {len(impl_by_file)} files:")
    for rel_path, methods in impl_by_file.items():
        print(f"  {rel_path}: {len(methods)} methods")
        for method in methods:
            print(f"    - {method[:50]}...")
    
    if len(impl_by_file) > 0 and 'point.cpp' in impl_by_file:
        print("\n✅ SUCCESS: Direct DEFINES edges also work")
        print(f"   But DEFINES_BODY approach is better (already exists!)")
        return True
    else:
        print("\n❌ FAILURE: Direct DEFINES didn't work")
        return False

if __name__ == "__main__":
    success1 = test_cpp_struct_augmentation_current()
    success2 = test_cpp_struct_augmentation_week6()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Week 6 DEFINES_BODY approach: {'✅' if success1 else '❌'}")
    print(f"Alternative direct DEFINES:   {'✅' if success2 else '❌'}")
    print("\nConclusion:")
    if success1:
        print("✅ Week 6 Complete! ContentExpander now follows:")
        print("   Point -> DEFINES -> declaration -> DEFINES_BODY (incoming) -> implementation")
        print("   No parser changes needed - DEFINES_BODY already exists!")
    else:
        print("❌ Week 6 needs debugging")
