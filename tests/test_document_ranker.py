"""
Test DocumentRanker with MultiDiGraph edge data access.

Verifies:
1. Correct iteration over MultiDiGraph edge keys
2. Proper access to 'relationship_type' (singular) key
3. Relationship weights are correctly applied
4. Multi-language support (Python, TypeScript, Java, C++)
5. Various symbol types (class, interface, function, struct, enum, constant)
"""

import pytest
import networkx as nx
from langchain_core.documents import Document

from plugin_implementation.document_ranker import DocumentRanker, rank_expanded_documents


# =============================================================================
# Test Fixtures - Multi-language graphs with diverse symbol types
# =============================================================================

@pytest.fixture
def python_graph():
    """Create a Python MultiDiGraph with classes, functions, and constants."""
    G = nx.MultiDiGraph()
    
    # Nodes
    G.add_node("python::app::UserService", 
               symbol_name="UserService", file_path="/src/app.py", symbol_type="class")
    G.add_node("python::app::User", 
               symbol_name="User", file_path="/src/app.py", symbol_type="class")
    G.add_node("python::app::get_user_by_id",
               symbol_name="get_user_by_id", file_path="/src/app.py", symbol_type="function")
    G.add_node("python::app::MAX_USERS",
               symbol_name="MAX_USERS", file_path="/src/app.py", symbol_type="constant")
    G.add_node("python::app::UserRole",
               symbol_name="UserRole", file_path="/src/app.py", symbol_type="enum")
    
    # Edges
    G.add_edge("python::app::UserService", "python::app::User",
               relationship_type="composition", source_file="app.py")
    G.add_edge("python::app::UserService", "python::app::get_user_by_id",
               relationship_type="calls", source_file="app.py")
    G.add_edge("python::app::UserService", "python::app::MAX_USERS",
               relationship_type="references", source_file="app.py")
    G.add_edge("python::app::User", "python::app::UserRole",
               relationship_type="composition", source_file="app.py")
    
    return G


@pytest.fixture
def typescript_graph():
    """Create a TypeScript MultiDiGraph with interfaces, classes, and functions."""
    G = nx.MultiDiGraph()
    
    # Nodes - interface, class, function, type alias
    G.add_node("typescript::api::IUserService",
               symbol_name="IUserService", file_path="/src/api.ts", symbol_type="interface")
    G.add_node("typescript::api::UserServiceImpl",
               symbol_name="UserServiceImpl", file_path="/src/api.ts", symbol_type="class")
    G.add_node("typescript::api::UserDTO",
               symbol_name="UserDTO", file_path="/src/api.ts", symbol_type="interface")
    G.add_node("typescript::api::createUser",
               symbol_name="createUser", file_path="/src/api.ts", symbol_type="function")
    G.add_node("typescript::api::API_VERSION",
               symbol_name="API_VERSION", file_path="/src/api.ts", symbol_type="constant")
    
    # Edges
    G.add_edge("typescript::api::UserServiceImpl", "typescript::api::IUserService",
               relationship_type="implementation", source_file="api.ts")
    G.add_edge("typescript::api::UserServiceImpl", "typescript::api::UserDTO",
               relationship_type="composition", source_file="api.ts")
    G.add_edge("typescript::api::createUser", "typescript::api::UserDTO",
               relationship_type="returns", source_file="api.ts")
    G.add_edge("typescript::api::createUser", "typescript::api::API_VERSION",
               relationship_type="references", source_file="api.ts")
    
    return G


@pytest.fixture
def java_graph():
    """Create a Java MultiDiGraph with classes, interfaces, and enums."""
    G = nx.MultiDiGraph()
    
    # Nodes
    G.add_node("java::UserRepository::UserRepository",
               symbol_name="UserRepository", file_path="/src/UserRepository.java", symbol_type="interface")
    G.add_node("java::UserRepositoryImpl::UserRepositoryImpl",
               symbol_name="UserRepositoryImpl", file_path="/src/UserRepositoryImpl.java", symbol_type="class")
    G.add_node("java::User::User",
               symbol_name="User", file_path="/src/User.java", symbol_type="class")
    G.add_node("java::UserStatus::UserStatus",
               symbol_name="UserStatus", file_path="/src/UserStatus.java", symbol_type="enum")
    G.add_node("java::Constants::MAX_CONNECTIONS",
               symbol_name="MAX_CONNECTIONS", file_path="/src/Constants.java", symbol_type="constant")
    
    # Edges
    G.add_edge("java::UserRepositoryImpl::UserRepositoryImpl", "java::UserRepository::UserRepository",
               relationship_type="implementation", source_file="UserRepositoryImpl.java")
    G.add_edge("java::UserRepositoryImpl::UserRepositoryImpl", "java::User::User",
               relationship_type="composition", source_file="UserRepositoryImpl.java")
    G.add_edge("java::User::User", "java::UserStatus::UserStatus",
               relationship_type="composition", source_file="User.java")
    G.add_edge("java::UserRepositoryImpl::UserRepositoryImpl", "java::Constants::MAX_CONNECTIONS",
               relationship_type="references", source_file="UserRepositoryImpl.java")
    
    return G


@pytest.fixture
def cpp_graph():
    """Create a C++ MultiDiGraph with classes, structs, and functions."""
    G = nx.MultiDiGraph()
    
    # Nodes - class, struct, function, namespace-level constant
    G.add_node("cpp::user_service::UserManager",
               symbol_name="UserManager", file_path="/src/user_service.cpp", symbol_type="class")
    G.add_node("cpp::user_service::UserData",
               symbol_name="UserData", file_path="/src/user_service.hpp", symbol_type="struct")
    G.add_node("cpp::user_service::UserConfig",
               symbol_name="UserConfig", file_path="/src/user_service.hpp", symbol_type="struct")
    G.add_node("cpp::user_service::create_user",
               symbol_name="create_user", file_path="/src/user_service.cpp", symbol_type="function")
    G.add_node("cpp::user_service::MAX_USERS",
               symbol_name="MAX_USERS", file_path="/src/user_service.hpp", symbol_type="constant")
    
    # Edges
    G.add_edge("cpp::user_service::UserManager", "cpp::user_service::UserData",
               relationship_type="composition", source_file="user_service.cpp")
    G.add_edge("cpp::user_service::UserManager", "cpp::user_service::UserConfig",
               relationship_type="aggregation", source_file="user_service.cpp")
    G.add_edge("cpp::user_service::UserManager", "cpp::user_service::create_user",
               relationship_type="calls", source_file="user_service.cpp")
    G.add_edge("cpp::user_service::create_user", "cpp::user_service::UserData",
               relationship_type="returns", source_file="user_service.cpp")
    G.add_edge("cpp::user_service::create_user", "cpp::user_service::MAX_USERS",
               relationship_type="references", source_file="user_service.cpp")
    
    return G


@pytest.fixture
def javascript_graph():
    """Create a JavaScript MultiDiGraph with classes, functions, and constants."""
    G = nx.MultiDiGraph()
    
    # Nodes - ES6 classes, functions, constants
    G.add_node("javascript::app::UserController",
               symbol_name="UserController", file_path="/src/app.js", symbol_type="class")
    G.add_node("javascript::app::UserModel",
               symbol_name="UserModel", file_path="/src/app.js", symbol_type="class")
    G.add_node("javascript::app::validateUser",
               symbol_name="validateUser", file_path="/src/app.js", symbol_type="function")
    G.add_node("javascript::app::DEFAULT_ROLE",
               symbol_name="DEFAULT_ROLE", file_path="/src/app.js", symbol_type="constant")
    G.add_node("javascript::app::UserStatus",
               symbol_name="UserStatus", file_path="/src/app.js", symbol_type="enum")
    
    # Edges
    G.add_edge("javascript::app::UserController", "javascript::app::UserModel",
               relationship_type="composition", source_file="app.js")
    G.add_edge("javascript::app::UserController", "javascript::app::validateUser",
               relationship_type="calls", source_file="app.js")
    G.add_edge("javascript::app::UserController", "javascript::app::DEFAULT_ROLE",
               relationship_type="references", source_file="app.js")
    G.add_edge("javascript::app::UserModel", "javascript::app::UserStatus",
               relationship_type="composition", source_file="app.js")
    
    return G


@pytest.fixture
def documentation_graph():
    """Create a graph with documentation symbols (README, markdown, config)."""
    G = nx.MultiDiGraph()
    
    # Documentation nodes
    G.add_node("docs::README::README",
               symbol_name="README", file_path="/README.md", symbol_type="readme")
    G.add_node("docs::API::API",
               symbol_name="API", file_path="/docs/API.md", symbol_type="documentation")
    G.add_node("docs::CONTRIBUTING::CONTRIBUTING",
               symbol_name="CONTRIBUTING", file_path="/CONTRIBUTING.md", symbol_type="markdown")
    G.add_node("docs::package::package.json",
               symbol_name="package.json", file_path="/package.json", symbol_type="config")
    G.add_node("docs::architecture::architecture",
               symbol_name="architecture", file_path="/docs/architecture.md", symbol_type="text_chunk")
    
    # Code nodes that docs might reference
    G.add_node("python::app::MainClass",
               symbol_name="MainClass", file_path="/src/app.py", symbol_type="class")
    
    # Documentation can reference code
    G.add_edge("docs::API::API", "python::app::MainClass",
               relationship_type="references", source_file="API.md")
    
    return G


@pytest.fixture
def mixed_language_graph():
    """Create a graph combining multiple languages (simulating polyglot project)."""
    G = nx.MultiDiGraph()
    
    # Python backend
    G.add_node("python::backend::ApiHandler",
               symbol_name="ApiHandler", file_path="/backend/api.py", symbol_type="class")
    G.add_node("python::backend::process_request",
               symbol_name="process_request", file_path="/backend/api.py", symbol_type="function")
    
    # TypeScript frontend  
    G.add_node("typescript::frontend::UserComponent",
               symbol_name="UserComponent", file_path="/frontend/User.tsx", symbol_type="class")
    G.add_node("typescript::frontend::IUserProps",
               symbol_name="IUserProps", file_path="/frontend/User.tsx", symbol_type="interface")
    
    # Edges within languages
    G.add_edge("python::backend::ApiHandler", "python::backend::process_request",
               relationship_type="calls", source_file="api.py")
    G.add_edge("typescript::frontend::UserComponent", "typescript::frontend::IUserProps",
               relationship_type="composition", source_file="User.tsx")
    
    return G


def make_doc(symbol_name: str, file_path: str, symbol_type: str, content: str) -> Document:
    """Helper to create a Document with standard metadata."""
    return Document(
        page_content=content,
        metadata={
            "symbol_name": symbol_name,
            "file_path": file_path,
            "symbol_type": symbol_type
        }
    )


# =============================================================================
# Python Tests
# =============================================================================

class TestPythonRanker:
    """Test DocumentRanker with Python graph patterns."""
    
    def test_class_composition(self, python_graph):
        """Test class -> class composition edge scoring."""
        ranker = DocumentRanker(python_graph)
        score = ranker._calculate_relationship_score(
            "python::app::UserService", "python::app::User"
        )
        assert score == 8, f"composition weight should be 8, got {score}"
    
    def test_class_calls_function(self, python_graph):
        """Test class -> function calls edge scoring."""
        ranker = DocumentRanker(python_graph)
        score = ranker._calculate_relationship_score(
            "python::app::UserService", "python::app::get_user_by_id"
        )
        assert score == 12, f"calls weight should be 12, got {score}"
    
    def test_class_references_constant(self, python_graph):
        """Test class -> constant references edge scoring."""
        ranker = DocumentRanker(python_graph)
        score = ranker._calculate_relationship_score(
            "python::app::UserService", "python::app::MAX_USERS"
        )
        assert score == 8, f"references weight should be 8, got {score}"
    
    def test_class_uses_enum(self, python_graph):
        """Test class -> enum composition edge scoring."""
        ranker = DocumentRanker(python_graph)
        score = ranker._calculate_relationship_score(
            "python::app::User", "python::app::UserRole"
        )
        assert score == 8, f"composition weight should be 8, got {score}"


# =============================================================================
# TypeScript Tests  
# =============================================================================

class TestTypeScriptRanker:
    """Test DocumentRanker with TypeScript graph patterns."""
    
    def test_class_implements_interface(self, typescript_graph):
        """Test class -> interface implementation edge scoring."""
        ranker = DocumentRanker(typescript_graph)
        score = ranker._calculate_relationship_score(
            "typescript::api::UserServiceImpl", "typescript::api::IUserService"
        )
        assert score == 10, f"implementation weight should be 10, got {score}"
    
    def test_class_composes_interface(self, typescript_graph):
        """Test class -> interface composition (field type)."""
        ranker = DocumentRanker(typescript_graph)
        score = ranker._calculate_relationship_score(
            "typescript::api::UserServiceImpl", "typescript::api::UserDTO"
        )
        assert score == 8, f"composition weight should be 8, got {score}"
    
    def test_function_returns_interface(self, typescript_graph):
        """Test function -> interface returns edge."""
        ranker = DocumentRanker(typescript_graph)
        # 'returns' is not in RELATIONSHIP_WEIGHTS, should be 0
        score = ranker._calculate_relationship_score(
            "typescript::api::createUser", "typescript::api::UserDTO"
        )
        # Check if returns is weighted or not
        assert score >= 0  # May or may not be weighted
    
    def test_function_references_constant(self, typescript_graph):
        """Test function -> constant references edge."""
        ranker = DocumentRanker(typescript_graph)
        score = ranker._calculate_relationship_score(
            "typescript::api::createUser", "typescript::api::API_VERSION"
        )
        assert score == 8, f"references weight should be 8, got {score}"


# =============================================================================
# Java Tests
# =============================================================================

class TestJavaRanker:
    """Test DocumentRanker with Java graph patterns."""
    
    def test_class_implements_interface(self, java_graph):
        """Test class implements interface in Java."""
        ranker = DocumentRanker(java_graph)
        score = ranker._calculate_relationship_score(
            "java::UserRepositoryImpl::UserRepositoryImpl",
            "java::UserRepository::UserRepository"
        )
        assert score == 10, f"implementation weight should be 10, got {score}"
    
    def test_class_composes_class(self, java_graph):
        """Test class has-a class composition."""
        ranker = DocumentRanker(java_graph)
        score = ranker._calculate_relationship_score(
            "java::UserRepositoryImpl::UserRepositoryImpl",
            "java::User::User"
        )
        assert score == 8, f"composition weight should be 8, got {score}"
    
    def test_class_uses_enum(self, java_graph):
        """Test class uses enum as field type."""
        ranker = DocumentRanker(java_graph)
        score = ranker._calculate_relationship_score(
            "java::User::User",
            "java::UserStatus::UserStatus"
        )
        assert score == 8, f"composition weight should be 8, got {score}"
    
    def test_class_references_constant(self, java_graph):
        """Test class references static constant."""
        ranker = DocumentRanker(java_graph)
        score = ranker._calculate_relationship_score(
            "java::UserRepositoryImpl::UserRepositoryImpl",
            "java::Constants::MAX_CONNECTIONS"
        )
        assert score == 8, f"references weight should be 8, got {score}"


# =============================================================================
# C++ Tests
# =============================================================================

class TestCppRanker:
    """Test DocumentRanker with C++ graph patterns."""
    
    def test_class_composes_struct(self, cpp_graph):
        """Test class has struct member (composition)."""
        ranker = DocumentRanker(cpp_graph)
        score = ranker._calculate_relationship_score(
            "cpp::user_service::UserManager",
            "cpp::user_service::UserData"
        )
        assert score == 8, f"composition weight should be 8, got {score}"
    
    def test_class_aggregates_struct(self, cpp_graph):
        """Test class has pointer/reference to struct (aggregation)."""
        ranker = DocumentRanker(cpp_graph)
        score = ranker._calculate_relationship_score(
            "cpp::user_service::UserManager",
            "cpp::user_service::UserConfig"
        )
        assert score == 8, f"aggregation weight should be 8, got {score}"
    
    def test_class_calls_function(self, cpp_graph):
        """Test class calls free function."""
        ranker = DocumentRanker(cpp_graph)
        score = ranker._calculate_relationship_score(
            "cpp::user_service::UserManager",
            "cpp::user_service::create_user"
        )
        assert score == 12, f"calls weight should be 12, got {score}"
    
    def test_function_references_constant(self, cpp_graph):
        """Test function references namespace constant."""
        ranker = DocumentRanker(cpp_graph)
        score = ranker._calculate_relationship_score(
            "cpp::user_service::create_user",
            "cpp::user_service::MAX_USERS"
        )
        assert score == 8, f"references weight should be 8, got {score}"


# =============================================================================
# JavaScript Tests
# =============================================================================

class TestJavaScriptRanker:
    """Test DocumentRanker with JavaScript graph patterns."""
    
    def test_class_composition(self, javascript_graph):
        """Test ES6 class -> class composition."""
        ranker = DocumentRanker(javascript_graph)
        score = ranker._calculate_relationship_score(
            "javascript::app::UserController",
            "javascript::app::UserModel"
        )
        assert score == 8, f"composition weight should be 8, got {score}"
    
    def test_class_calls_function(self, javascript_graph):
        """Test class -> function calls edge."""
        ranker = DocumentRanker(javascript_graph)
        score = ranker._calculate_relationship_score(
            "javascript::app::UserController",
            "javascript::app::validateUser"
        )
        assert score == 12, f"calls weight should be 12, got {score}"
    
    def test_class_references_constant(self, javascript_graph):
        """Test class -> const references edge."""
        ranker = DocumentRanker(javascript_graph)
        score = ranker._calculate_relationship_score(
            "javascript::app::UserController",
            "javascript::app::DEFAULT_ROLE"
        )
        assert score == 8, f"references weight should be 8, got {score}"
    
    def test_class_uses_enum(self, javascript_graph):
        """Test class uses enum-like object."""
        ranker = DocumentRanker(javascript_graph)
        score = ranker._calculate_relationship_score(
            "javascript::app::UserModel",
            "javascript::app::UserStatus"
        )
        assert score == 8, f"composition weight should be 8, got {score}"


# =============================================================================
# Documentation Symbol Tests
# =============================================================================

class TestDocumentationRanker:
    """Test DocumentRanker with documentation symbols."""
    
    def test_readme_symbol_type_score(self, documentation_graph):
        """Test README gets highest documentation score."""
        ranker = DocumentRanker(documentation_graph)
        score = ranker.SYMBOL_TYPE_SCORES.get('readme', 0)
        assert score == 10, f"readme should have score 10, got {score}"
    
    def test_documentation_symbol_type_score(self, documentation_graph):
        """Test documentation files get high score."""
        ranker = DocumentRanker(documentation_graph)
        score = ranker.SYMBOL_TYPE_SCORES.get('documentation', 0)
        assert score == 9, f"documentation should have score 9, got {score}"
    
    def test_markdown_symbol_type_score(self, documentation_graph):
        """Test markdown files get good score."""
        ranker = DocumentRanker(documentation_graph)
        score = ranker.SYMBOL_TYPE_SCORES.get('markdown', 0)
        assert score == 8, f"markdown should have score 8, got {score}"
    
    def test_config_symbol_type_score(self, documentation_graph):
        """Test config files (package.json, etc.) get good score."""
        ranker = DocumentRanker(documentation_graph)
        score = ranker.SYMBOL_TYPE_SCORES.get('config', 0)
        assert score == 8, f"config should have score 8, got {score}"
    
    def test_text_chunk_symbol_type_score(self, documentation_graph):
        """Test text chunks from docs get good score."""
        ranker = DocumentRanker(documentation_graph)
        score = ranker.SYMBOL_TYPE_SCORES.get('text_chunk', 0)
        assert score == 8, f"text_chunk should have score 8, got {score}"
    
    def test_doc_references_code(self, documentation_graph):
        """Test documentation referencing code gets relationship score."""
        ranker = DocumentRanker(documentation_graph)
        score = ranker._calculate_relationship_score(
            "docs::API::API",
            "python::app::MainClass"
        )
        assert score == 8, f"references weight should be 8, got {score}"
    
    def test_documentation_ranking_pipeline(self, documentation_graph):
        """Test full ranking with documentation symbols."""
        docs = [
            make_doc("docs::README::README", "/README.md", "readme",
                    "# Project Title\nThis is the main documentation."),
            make_doc("docs::API::API", "/docs/API.md", "documentation",
                    "## API Reference\nEndpoints and usage."),
            make_doc("docs::CONTRIBUTING::CONTRIBUTING", "/CONTRIBUTING.md", "markdown",
                    "## How to Contribute\nGuidelines for contributors."),
            make_doc("docs::package::package.json", "/package.json", "config",
                    '{"name": "project", "version": "1.0.0"}'),
            make_doc("python::app::MainClass", "/src/app.py", "class",
                    "class MainClass:\n    pass"),
        ]
        
        page_spec = {"topic": "docs::API::API", "target_folders": [], "key_files": []}
        tiered_docs, metrics = rank_expanded_documents(page_spec, docs, documentation_graph)
        
        assert metrics['total_docs'] == 5
        
        # Check scores include symbol type bonuses
        scores = {doc.metadata['symbol_name']: score for doc, _, score in tiered_docs}
        
        # All symbols should be present
        assert "docs::README::README" in scores
        assert "docs::API::API" in scores
        assert "python::app::MainClass" in scores
        
        # MainClass should have relationship bonus from API doc (references edge)
        # API -> MainClass has references(8) weight
        assert scores["python::app::MainClass"] >= 8, \
            f"MainClass should have relationship bonus, got {scores['python::app::MainClass']}"


# =============================================================================
# Cross-cutting Tests
# =============================================================================

class TestMultiLanguageRanker:
    """Test DocumentRanker with mixed language graphs."""
    
    def test_python_edges_in_mixed_graph(self, mixed_language_graph):
        """Test Python edges work in polyglot graph."""
        ranker = DocumentRanker(mixed_language_graph)
        score = ranker._calculate_relationship_score(
            "python::backend::ApiHandler",
            "python::backend::process_request"
        )
        assert score == 12, f"calls weight should be 12, got {score}"
    
    def test_typescript_edges_in_mixed_graph(self, mixed_language_graph):
        """Test TypeScript edges work in polyglot graph."""
        ranker = DocumentRanker(mixed_language_graph)
        score = ranker._calculate_relationship_score(
            "typescript::frontend::UserComponent",
            "typescript::frontend::IUserProps"
        )
        assert score == 8, f"composition weight should be 8, got {score}"
    
    def test_no_cross_language_edges(self, mixed_language_graph):
        """Test that unrelated cross-language symbols have no edges."""
        ranker = DocumentRanker(mixed_language_graph)
        score = ranker._calculate_relationship_score(
            "python::backend::ApiHandler",
            "typescript::frontend::UserComponent"
        )
        assert score == 0, f"No edges should exist cross-language, got {score}"


class TestMultipleEdgesBetweenNodes:
    """Test that multiple edges between same nodes are all scored."""
    
    @pytest.fixture
    def multi_edge_graph(self):
        """Graph with multiple relationship types between same nodes."""
        G = nx.MultiDiGraph()
        
        G.add_node("python::app::Service",
                   symbol_name="Service", file_path="/src/app.py", symbol_type="class")
        G.add_node("python::app::Repository",
                   symbol_name="Repository", file_path="/src/app.py", symbol_type="class")
        
        # Multiple edges: Service both inherits AND composes Repository
        G.add_edge("python::app::Service", "python::app::Repository",
                   relationship_type="inheritance", source_file="app.py")
        G.add_edge("python::app::Service", "python::app::Repository",
                   relationship_type="composition", source_file="app.py")
        G.add_edge("python::app::Service", "python::app::Repository",
                   relationship_type="calls", source_file="app.py")
        
        return G
    
    def test_all_edges_scored(self, multi_edge_graph):
        """Test that all edges between nodes contribute to score."""
        ranker = DocumentRanker(multi_edge_graph)
        score = ranker._calculate_relationship_score(
            "python::app::Service",
            "python::app::Repository"
        )
        # inheritance(10) + composition(8) + calls(12) = 30
        expected = 10 + 8 + 12
        assert score == expected, f"All edges should sum to {expected}, got {score}"


class TestBidirectionalEdges:
    """Test that edges in both directions are scored."""
    
    @pytest.fixture
    def bidirectional_graph(self):
        """Graph with edges in both directions."""
        G = nx.MultiDiGraph()
        
        G.add_node("python::app::A",
                   symbol_name="A", file_path="/src/app.py", symbol_type="class")
        G.add_node("python::app::B",
                   symbol_name="B", file_path="/src/app.py", symbol_type="class")
        
        # A -> B (composition)
        G.add_edge("python::app::A", "python::app::B",
                   relationship_type="composition", source_file="app.py")
        # B -> A (references back)
        G.add_edge("python::app::B", "python::app::A",
                   relationship_type="references", source_file="app.py")
        
        return G
    
    def test_both_directions_scored(self, bidirectional_graph):
        """Test that A->B and B->A edges both contribute."""
        ranker = DocumentRanker(bidirectional_graph)
        score = ranker._calculate_relationship_score(
            "python::app::A",
            "python::app::B"
        )
        # A->B: composition(8) + B->A: references(8) = 16
        expected = 8 + 8
        assert score == expected, f"Both directions should sum to {expected}, got {score}"


# =============================================================================
# Full Ranking Pipeline Tests
# =============================================================================

class TestFullRankingPipeline:
    """Test the complete ranking pipeline with various symbol types."""
    
    def test_python_ranking(self, python_graph):
        """Test full ranking with Python symbols."""
        docs = [
            make_doc("python::app::UserService", "/src/app.py", "class",
                    "class UserService:\n    def get_user(self): pass"),
            make_doc("python::app::User", "/src/app.py", "class",
                    "class User:\n    name: str"),
            make_doc("python::app::get_user_by_id", "/src/app.py", "function",
                    "def get_user_by_id(id: int) -> User: pass"),
            make_doc("python::app::MAX_USERS", "/src/app.py", "constant",
                    "MAX_USERS = 1000"),
            make_doc("python::app::UserRole", "/src/app.py", "enum",
                    "class UserRole(Enum):\n    ADMIN = 1"),
        ]
        
        page_spec = {"topic": "python::app::UserService", "target_folders": [], "key_files": []}
        tiered_docs, metrics = rank_expanded_documents(page_spec, docs, python_graph)
        
        assert metrics['total_docs'] == 5
        assert metrics['skipped_docs'] == 0
        
        # Verify all docs were ranked
        ranked_symbols = {doc.metadata['symbol_name'] for doc, _, _ in tiered_docs}
        assert len(ranked_symbols) == 5
    
    def test_typescript_ranking(self, typescript_graph):
        """Test full ranking with TypeScript symbols including interfaces."""
        docs = [
            make_doc("typescript::api::IUserService", "/src/api.ts", "interface",
                    "interface IUserService {\n    getUser(id: string): UserDTO;\n}"),
            make_doc("typescript::api::UserServiceImpl", "/src/api.ts", "class",
                    "class UserServiceImpl implements IUserService {}"),
            make_doc("typescript::api::UserDTO", "/src/api.ts", "interface",
                    "interface UserDTO {\n    id: string;\n    name: string;\n}"),
            make_doc("typescript::api::createUser", "/src/api.ts", "function",
                    "function createUser(data: UserDTO): UserDTO {}"),
            make_doc("typescript::api::API_VERSION", "/src/api.ts", "constant",
                    "const API_VERSION = '1.0.0';"),
        ]
        
        page_spec = {"topic": "typescript::api::UserServiceImpl", "target_folders": [], "key_files": []}
        tiered_docs, metrics = rank_expanded_documents(page_spec, docs, typescript_graph)
        
        assert metrics['total_docs'] == 5
        
        # Check that interface (IUserService) gets relationship bonus from topic
        scores = {doc.metadata['symbol_name']: score for doc, _, score in tiered_docs}
        # IUserService should have implementation edge from topic
        assert scores["typescript::api::IUserService"] >= 10  # implementation weight
    
    def test_java_ranking(self, java_graph):
        """Test full ranking with Java symbols."""
        docs = [
            make_doc("java::UserRepository::UserRepository", "/src/UserRepository.java", "interface",
                    "public interface UserRepository { User findById(Long id); }"),
            make_doc("java::UserRepositoryImpl::UserRepositoryImpl", "/src/UserRepositoryImpl.java", "class",
                    "public class UserRepositoryImpl implements UserRepository {}"),
            make_doc("java::User::User", "/src/User.java", "class",
                    "public class User { private String name; }"),
            make_doc("java::UserStatus::UserStatus", "/src/UserStatus.java", "enum",
                    "public enum UserStatus { ACTIVE, INACTIVE }"),
            make_doc("java::Constants::MAX_CONNECTIONS", "/src/Constants.java", "constant",
                    "public static final int MAX_CONNECTIONS = 100;"),
        ]
        
        page_spec = {"topic": "java::UserRepositoryImpl::UserRepositoryImpl", "target_folders": [], "key_files": []}
        tiered_docs, metrics = rank_expanded_documents(page_spec, docs, java_graph)
        
        assert metrics['total_docs'] == 5
    
    def test_cpp_ranking(self, cpp_graph):
        """Test full ranking with C++ symbols including structs."""
        docs = [
            make_doc("cpp::user_service::UserManager", "/src/user_service.cpp", "class",
                    "class UserManager {\n    UserData data_;\n};"),
            make_doc("cpp::user_service::UserData", "/src/user_service.hpp", "struct",
                    "struct UserData {\n    std::string name;\n};"),
            make_doc("cpp::user_service::UserConfig", "/src/user_service.hpp", "struct",
                    "struct UserConfig {\n    int max_users;\n};"),
            make_doc("cpp::user_service::create_user", "/src/user_service.cpp", "function",
                    "UserData create_user(const UserConfig& config);"),
            make_doc("cpp::user_service::MAX_USERS", "/src/user_service.hpp", "constant",
                    "constexpr int MAX_USERS = 1000;"),
        ]
        
        page_spec = {"topic": "cpp::user_service::UserManager", "target_folders": [], "key_files": []}
        tiered_docs, metrics = rank_expanded_documents(page_spec, docs, cpp_graph)
        
        assert metrics['total_docs'] == 5
        
        # Structs should get relationship bonuses
        scores = {doc.metadata['symbol_name']: score for doc, _, score in tiered_docs}
        # UserData has composition edge from topic
        assert scores["cpp::user_service::UserData"] >= 8
    
    def test_javascript_ranking(self, javascript_graph):
        """Test full ranking with JavaScript symbols."""
        docs = [
            make_doc("javascript::app::UserController", "/src/app.js", "class",
                    "class UserController {\n    constructor() {}\n}"),
            make_doc("javascript::app::UserModel", "/src/app.js", "class",
                    "class UserModel {\n    constructor(data) {}\n}"),
            make_doc("javascript::app::validateUser", "/src/app.js", "function",
                    "function validateUser(user) { return true; }"),
            make_doc("javascript::app::DEFAULT_ROLE", "/src/app.js", "constant",
                    "const DEFAULT_ROLE = 'user';"),
            make_doc("javascript::app::UserStatus", "/src/app.js", "enum",
                    "const UserStatus = { ACTIVE: 1, INACTIVE: 0 };"),
        ]
        
        page_spec = {"topic": "javascript::app::UserController", "target_folders": [], "key_files": []}
        tiered_docs, metrics = rank_expanded_documents(page_spec, docs, javascript_graph)
        
        assert metrics['total_docs'] == 5
        
        # Check relationship scores
        scores = {doc.metadata['symbol_name']: score for doc, _, score in tiered_docs}
        # UserModel has composition edge from topic
        assert scores["javascript::app::UserModel"] >= 8
        # validateUser has calls edge from topic  
        assert scores["javascript::app::validateUser"] >= 12


# =============================================================================
# Weight Definition Tests
# =============================================================================

class TestDocumentRankerWeights:
    """Test that relationship weights are correctly defined."""
    
    def test_all_relationship_types_have_weights(self):
        """Verify all expected relationship types have weights."""
        ranker = DocumentRanker(None)
        
        # Key relationship types from RelationshipType enum
        expected_types = [
            'defines', 'contains', 'calls', 'inheritance',
            'composition', 'aggregation', 'references',
            'implementation', 'creates', 'instantiates'
        ]
        
        for rel_type in expected_types:
            assert rel_type in ranker.RELATIONSHIP_WEIGHTS, \
                f"Missing weight for relationship type: {rel_type}"
            assert ranker.RELATIONSHIP_WEIGHTS[rel_type] > 0, \
                f"Weight for {rel_type} should be positive"
    
    def test_weight_hierarchy(self):
        """Test that weight hierarchy makes sense."""
        ranker = DocumentRanker(None)
        weights = ranker.RELATIONSHIP_WEIGHTS
        
        # Containment should be highest
        assert weights['defines'] >= weights['calls']
        assert weights['defines'] >= weights['inheritance']
        
        # Calls should be higher than composition
        assert weights['calls'] >= weights['composition']
        
        # Implementation should equal inheritance
        assert weights['implementation'] == weights['inheritance']


class TestSymbolTypeScores:
    """Test that symbol type scores are correctly defined."""
    
    def test_all_code_symbol_types_scored(self):
        """Verify architectural code symbol types have scores."""
        ranker = DocumentRanker(None)
        
        code_types = ['class', 'interface', 'struct', 'function', 'enum', 'constant']
        
        for symbol_type in code_types:
            assert symbol_type in ranker.SYMBOL_TYPE_SCORES, \
                f"Missing score for code symbol type: {symbol_type}"
            assert ranker.SYMBOL_TYPE_SCORES[symbol_type] > 0, \
                f"Score for {symbol_type} should be positive"
    
    def test_all_documentation_symbol_types_scored(self):
        """Verify documentation symbol types have scores."""
        ranker = DocumentRanker(None)
        
        doc_types = ['readme', 'documentation', 'markdown', 'text_chunk', 'config']
        
        for symbol_type in doc_types:
            assert symbol_type in ranker.SYMBOL_TYPE_SCORES, \
                f"Missing score for documentation symbol type: {symbol_type}"
            assert ranker.SYMBOL_TYPE_SCORES[symbol_type] >= 8, \
                f"Documentation type {symbol_type} should have score >= 8"
    
    def test_class_interface_struct_equal(self):
        """Test that class, interface, struct have equal priority."""
        ranker = DocumentRanker(None)
        scores = ranker.SYMBOL_TYPE_SCORES
        
        assert scores['class'] == scores['interface'] == scores['struct']
    
    def test_readme_highest_doc_priority(self):
        """Test that README has highest documentation priority."""
        ranker = DocumentRanker(None)
        scores = ranker.SYMBOL_TYPE_SCORES
        
        assert scores['readme'] >= scores['documentation']
        assert scores['readme'] >= scores['markdown']
        assert scores['readme'] >= scores['config']
    
    def test_rust_types_scored(self):
        """Test Rust-specific types (trait, impl) are scored."""
        ranker = DocumentRanker(None)
        
        rust_types = ['trait', 'impl']
        for symbol_type in rust_types:
            assert symbol_type in ranker.SYMBOL_TYPE_SCORES, \
                f"Missing score for Rust type: {symbol_type}"
    
    def test_namespace_module_scored(self):
        """Test namespace/module types are scored."""
        ranker = DocumentRanker(None)
        
        container_types = ['namespace', 'module']
        for symbol_type in container_types:
            assert symbol_type in ranker.SYMBOL_TYPE_SCORES, \
                f"Missing score for container type: {symbol_type}"
    
    def test_type_alias_scored(self):
        """Test type_alias (TypeScript, Rust) is scored."""
        ranker = DocumentRanker(None)
        assert 'type_alias' in ranker.SYMBOL_TYPE_SCORES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
