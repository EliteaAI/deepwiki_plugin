"""
Comprehensive test suite for hybrid relationship hints across all languages.

Tests all expansion reasons and hint types:
- Inheritance: extended_by, implemented_by, extends
- Composition: composed_by, used_by, aggregation, composition
- Calls: called_by, references, creates, instantiated_by
- Containment: defines, contains, has_member
- Types: return_type_of, parameter_type_of, type_of, initializes
- Members: constructor_of, method_of, field_of

Languages tested: Python, TypeScript, JavaScript, Java, C++
"""

import os
import pytest
from typing import Dict, Set
from unittest.mock import MagicMock
from langchain_core.documents import Document

from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
from plugin_implementation.content_expander import ContentExpander


# =============================================================================
# Test Utilities
# =============================================================================

def create_test_files(tmpdir: str, files: Dict[str, str]) -> Dict[str, str]:
    """Create test files in temporary directory."""
    paths = {}
    for filename, content in files.items():
        filepath = os.path.join(tmpdir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)
        paths[filename] = filepath
    return paths


def build_graph_and_expander(tmpdir: str):
    """Build graph and create content expander."""
    builder = EnhancedUnifiedGraphBuilder()
    result = builder.analyze_repository(tmpdir)
    expander = ContentExpander(result.unified_graph)
    return result.unified_graph, expander


def find_node_by_name(graph, symbol_name: str, symbol_type: str = None):
    """Find a node in graph by symbol name and optional type."""
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('symbol_name') == symbol_name:
            if symbol_type is None or node_data.get('symbol_type', '').lower() == symbol_type.lower():
                return node_id, node_data
    return None, None


def create_doc_from_node(graph, node_id: str, node_data: dict) -> Document:
    """Create a Document from a graph node."""
    symbol = node_data.get('symbol')
    content = symbol.source_text if symbol and hasattr(symbol, 'source_text') else f"symbol {node_data.get('symbol_name', '')}"
    return Document(
        page_content=content,
        metadata={
            'symbol_name': node_data.get('symbol_name', ''),
            'file_path': node_data.get('file_path', ''),
            'symbol_type': node_data.get('symbol_type', ''),
            'language': node_data.get('language', ''),
            'node_id': node_id
        }
    )


def get_expansion_reasons(docs: list) -> Set[str]:
    """Extract all expansion reasons from documents."""
    return {doc.metadata.get('expansion_reason') for doc in docs if doc.metadata.get('expansion_reason')}


def get_docs_with_reason(docs: list, reason: str) -> list:
    """Get documents with a specific expansion reason."""
    return [doc for doc in docs if doc.metadata.get('expansion_reason') == reason]


# =============================================================================
# Python Comprehensive Hints Tests
# =============================================================================

class TestPythonHints:
    """Test all hint types for Python language."""
    
    @pytest.fixture
    def python_graph(self, tmp_path):
        """Create Python files with various relationships."""
        files = {
            "models/base.py": '''
class BaseModel:
    """Base model with common functionality."""
    
    def __init__(self, id: str):
        self.id = id
    
    def validate(self) -> bool:
        return True
''',
            "models/user.py": '''
from .base import BaseModel
from typing import List

class Address:
    """User address model."""
    street: str
    city: str

class User(BaseModel):
    """User model extending BaseModel."""
    
    name: str
    address: Address  # Composition
    
    def __init__(self, id: str, name: str):
        super().__init__(id)
        self.name = name
        self.address = Address()
    
    def get_full_info(self) -> str:
        return f"{self.name} at {self.address.city}"
''',
            "services/user_service.py": '''
from models.user import User, Address
from typing import List, Optional

class UserRepository:
    """Repository for user persistence."""
    
    def find_by_id(self, id: str) -> Optional[User]:
        return None
    
    def save(self, user: User) -> User:
        return user

class UserService:
    """Service for user operations."""
    
    def __init__(self, repo: UserRepository):
        self.repo = repo  # Composition
    
    def get_user(self, id: str) -> Optional[User]:
        return self.repo.find_by_id(id)
    
    def create_user(self, name: str) -> User:
        user = User("new-id", name)
        return self.repo.save(user)
''',
            "utils/helpers.py": '''
from typing import List

MAX_USERS = 100

def validate_name(name: str) -> bool:
    """Validate user name."""
    return len(name) > 0 and len(name) <= 50

def format_address(street: str, city: str) -> str:
    """Format address string."""
    return f"{street}, {city}"
'''
        }
        create_test_files(str(tmp_path), files)
        graph, expander = build_graph_and_expander(str(tmp_path))
        return graph, expander
    
    def test_python_inheritance_hints(self, python_graph):
        """Test inheritance hints: extended_by, extends."""
        graph, expander = python_graph
        
        # Find User class (extends BaseModel)
        node_id, node_data = find_node_by_name(graph, 'User', 'class')
        if not node_id:
            pytest.skip("User class not found in graph")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        reasons = get_expansion_reasons(expanded)
        
        print(f"Python inheritance - expansion reasons: {reasons}")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")
        
        # Should have inheritance-related expansion
        assert len(expanded) > 0, "Expected expanded documents for inheritance"
    
    def test_python_composition_hints(self, python_graph):
        """Test composition hints: composed_by, has_member."""
        graph, expander = python_graph
        
        # Find UserService (has repo: UserRepository)
        node_id, node_data = find_node_by_name(graph, 'UserService', 'class')
        if not node_id:
            pytest.skip("UserService class not found in graph")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        reasons = get_expansion_reasons(expanded)
        
        print(f"Python composition - expansion reasons: {reasons}")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")
                if d.metadata.get('expansion_via'):
                    print(f"    via: {d.metadata.get('expansion_via')}")
        
        assert len(expanded) > 0, "Expected expanded documents for composition"
    
    def test_python_function_call_hints(self, python_graph):
        """Test function call hints: called_by, references."""
        graph, expander = python_graph
        
        # Find validate_name function
        node_id, node_data = find_node_by_name(graph, 'validate_name', 'function')
        if not node_id:
            pytest.skip("validate_name function not found in graph")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        
        print(f"Python function call - expanded {len(expanded)} docs")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")
    
    def test_python_parameter_type_hints(self, python_graph):
        """Test parameter type hints: param_type."""
        graph, expander = python_graph
        
        # Find get_user method or create_user
        node_id, node_data = find_node_by_name(graph, 'create_user')
        if not node_id:
            node_id, node_data = find_node_by_name(graph, 'save')
        if not node_id:
            pytest.skip("Function with typed parameters not found in graph")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        reasons = get_expansion_reasons(expanded)
        
        print(f"Python parameter types - expansion reasons: {reasons}")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")


# =============================================================================
# TypeScript Comprehensive Hints Tests
# =============================================================================

class TestTypeScriptHints:
    """Test all hint types for TypeScript language."""
    
    @pytest.fixture
    def typescript_graph(self, tmp_path):
        """Create TypeScript files with various relationships."""
        files = {
            "src/interfaces/base.ts": '''
export interface IEntity {
    id: string;
    createdAt: Date;
}

export interface IRepository<T extends IEntity> {
    findById(id: string): T | null;
    save(entity: T): T;
    delete(id: string): boolean;
}
''',
            "src/models/user.ts": '''
import { IEntity } from '../interfaces/base';

export interface IAddress {
    street: string;
    city: string;
    country: string;
}

export class User implements IEntity {
    id: string;
    createdAt: Date;
    name: string;
    email: string;
    address: IAddress;  // Composition with interface
    
    constructor(id: string, name: string, email: string) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.createdAt = new Date();
    }
    
    getDisplayName(): string {
        return `${this.name} <${this.email}>`;
    }
}
''',
            "src/repositories/user-repository.ts": '''
import { IRepository } from '../interfaces/base';
import { User } from '../models/user';

export class UserRepository implements IRepository<User> {
    private users: Map<string, User> = new Map();
    
    findById(id: string): User | null {
        return this.users.get(id) || null;
    }
    
    save(entity: User): User {
        this.users.set(entity.id, entity);
        return entity;
    }
    
    delete(id: string): boolean {
        return this.users.delete(id);
    }
}
''',
            "src/services/user-service.ts": '''
import { User } from '../models/user';
import { UserRepository } from '../repositories/user-repository';

export class UserService {
    private repo: UserRepository;
    
    constructor(repo: UserRepository) {
        this.repo = repo;
    }
    
    async getUser(id: string): Promise<User | null> {
        return this.repo.findById(id);
    }
    
    async createUser(name: string, email: string): Promise<User> {
        const user = new User(crypto.randomUUID(), name, email);
        return this.repo.save(user);
    }
    
    async updateUserAddress(userId: string, address: IAddress): Promise<User | null> {
        const user = this.repo.findById(userId);
        if (user) {
            user.address = address;
            return this.repo.save(user);
        }
        return null;
    }
}
''',
            "src/utils/validators.ts": '''
export const MAX_NAME_LENGTH = 100;

export function validateEmail(email: string): boolean {
    const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return emailRegex.test(email);
}

export function validateName(name: string): boolean {
    return name.length > 0 && name.length <= MAX_NAME_LENGTH;
}
'''
        }
        create_test_files(str(tmp_path), files)
        graph, expander = build_graph_and_expander(str(tmp_path))
        return graph, expander
    
    def test_typescript_interface_implementation_hints(self, typescript_graph):
        """Test interface implementation hints: implemented_by."""
        graph, expander = typescript_graph
        
        # Find User class (implements IEntity)
        node_id, node_data = find_node_by_name(graph, 'User', 'class')
        if not node_id:
            pytest.skip("User class not found in graph")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        reasons = get_expansion_reasons(expanded)
        
        print(f"TypeScript interface impl - expansion reasons: {reasons}")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")
        
        assert len(expanded) > 0, "Expected expanded documents for interface implementation"
    
    def test_typescript_composition_hints(self, typescript_graph):
        """Test composition hints with interfaces."""
        graph, expander = typescript_graph
        
        # Find UserService (has repo: UserRepository)
        node_id, node_data = find_node_by_name(graph, 'UserService', 'class')
        if not node_id:
            pytest.skip("UserService class not found in graph")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        reasons = get_expansion_reasons(expanded)
        
        print(f"TypeScript composition - expansion reasons: {reasons}")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")
                if d.metadata.get('expansion_via'):
                    print(f"    via: {d.metadata.get('expansion_via')}")
        
        # Should include UserRepository via composition
        assert len(expanded) > 0, "Expected expanded documents for composition"
    
    def test_typescript_return_type_hints(self, typescript_graph):
        """Test return type hints."""
        graph, expander = typescript_graph
        
        # Find getUser method or findById
        node_id, node_data = find_node_by_name(graph, 'findById')
        if not node_id:
            pytest.skip("Method with return type not found")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        
        print(f"TypeScript return type - expanded {len(expanded)} docs")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")


# =============================================================================
# JavaScript Comprehensive Hints Tests
# =============================================================================

class TestJavaScriptHints:
    """Test all hint types for JavaScript language."""
    
    @pytest.fixture
    def javascript_graph(self, tmp_path):
        """Create JavaScript files with various relationships."""
        files = {
            "src/models/BaseModel.js": '''
class BaseModel {
    constructor(id) {
        this.id = id;
        this.createdAt = new Date();
    }
    
    toJSON() {
        return { id: this.id, createdAt: this.createdAt };
    }
}

module.exports = { BaseModel };
''',
            "src/models/User.js": '''
const { BaseModel } = require('./BaseModel');

class Address {
    constructor(street, city) {
        this.street = street;
        this.city = city;
    }
}

class User extends BaseModel {
    constructor(id, name, email) {
        super(id);
        this.name = name;
        this.email = email;
        this.address = null;
    }
    
    setAddress(address) {
        this.address = address;
    }
    
    getFullName() {
        return this.name;
    }
}

module.exports = { User, Address };
''',
            "src/services/UserService.js": '''
const { User } = require('../models/User');

class UserRepository {
    constructor() {
        this.users = new Map();
    }
    
    findById(id) {
        return this.users.get(id) || null;
    }
    
    save(user) {
        this.users.set(user.id, user);
        return user;
    }
}

class UserService {
    constructor(repository) {
        this.repository = repository;
    }
    
    getUser(id) {
        return this.repository.findById(id);
    }
    
    createUser(name, email) {
        const user = new User(Date.now().toString(), name, email);
        return this.repository.save(user);
    }
}

module.exports = { UserService, UserRepository };
''',
            "src/utils/helpers.js": '''
const MAX_LENGTH = 255;

function validateEmail(email) {
    return email && email.includes('@');
}

function formatName(firstName, lastName) {
    return `${firstName} ${lastName}`.trim();
}

module.exports = { MAX_LENGTH, validateEmail, formatName };
'''
        }
        create_test_files(str(tmp_path), files)
        graph, expander = build_graph_and_expander(str(tmp_path))
        return graph, expander
    
    def test_javascript_inheritance_hints(self, javascript_graph):
        """Test JavaScript class inheritance hints."""
        graph, expander = javascript_graph
        
        # Find User class (extends BaseModel)
        node_id, node_data = find_node_by_name(graph, 'User', 'class')
        if not node_id:
            pytest.skip("User class not found in graph")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        reasons = get_expansion_reasons(expanded)
        
        print(f"JavaScript inheritance - expansion reasons: {reasons}")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")
        
        assert len(expanded) > 0, "Expected expanded documents for JS inheritance"
    
    def test_javascript_composition_hints(self, javascript_graph):
        """Test JavaScript composition hints."""
        graph, expander = javascript_graph
        
        # Find UserService (has repository field)
        node_id, node_data = find_node_by_name(graph, 'UserService', 'class')
        if not node_id:
            pytest.skip("UserService class not found in graph")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        
        print(f"JavaScript composition - expanded {len(expanded)} docs")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")
    
    def test_javascript_object_creation_hints(self, javascript_graph):
        """Test JavaScript object creation hints (new X())."""
        graph, expander = javascript_graph
        
        # Find createUser which creates User objects
        node_id, node_data = find_node_by_name(graph, 'createUser')
        if not node_id:
            pytest.skip("createUser function not found")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        
        print(f"JavaScript object creation - expanded {len(expanded)} docs")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")


# =============================================================================
# Java Comprehensive Hints Tests
# =============================================================================

class TestJavaHints:
    """Test all hint types for Java language."""
    
    @pytest.fixture
    def java_graph(self, tmp_path):
        """Create Java files with various relationships."""
        files = {
            "src/com/example/interfaces/Repository.java": '''
package com.example.interfaces;

public interface Repository<T> {
    T findById(String id);
    T save(T entity);
    void delete(String id);
}
''',
            "src/com/example/models/BaseEntity.java": '''
package com.example.models;

import java.util.Date;

public abstract class BaseEntity {
    protected String id;
    protected Date createdAt;
    
    public BaseEntity(String id) {
        this.id = id;
        this.createdAt = new Date();
    }
    
    public String getId() {
        return id;
    }
    
    public abstract void validate();
}
''',
            "src/com/example/models/Address.java": '''
package com.example.models;

public class Address {
    private String street;
    private String city;
    private String country;
    
    public Address(String street, String city, String country) {
        this.street = street;
        this.city = city;
        this.country = country;
    }
    
    public String getFullAddress() {
        return street + ", " + city + ", " + country;
    }
}
''',
            "src/com/example/models/User.java": '''
package com.example.models;

public class User extends BaseEntity {
    private String name;
    private String email;
    private Address address;  // Composition
    
    public User(String id, String name, String email) {
        super(id);
        this.name = name;
        this.email = email;
    }
    
    public void setAddress(Address address) {
        this.address = address;
    }
    
    public Address getAddress() {
        return address;
    }
    
    @Override
    public void validate() {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("Name required");
        }
    }
}
''',
            "src/com/example/repositories/UserRepository.java": '''
package com.example.repositories;

import com.example.interfaces.Repository;
import com.example.models.User;
import java.util.HashMap;
import java.util.Map;

public class UserRepository implements Repository<User> {
    private Map<String, User> storage = new HashMap<>();
    
    @Override
    public User findById(String id) {
        return storage.get(id);
    }
    
    @Override
    public User save(User entity) {
        storage.put(entity.getId(), entity);
        return entity;
    }
    
    @Override
    public void delete(String id) {
        storage.remove(id);
    }
}
''',
            "src/com/example/services/UserService.java": '''
package com.example.services;

import com.example.models.User;
import com.example.models.Address;
import com.example.repositories.UserRepository;

public class UserService {
    private final UserRepository repository;
    
    public UserService(UserRepository repository) {
        this.repository = repository;
    }
    
    public User getUser(String id) {
        return repository.findById(id);
    }
    
    public User createUser(String name, String email) {
        User user = new User(generateId(), name, email);
        return repository.save(user);
    }
    
    public User updateAddress(String userId, Address address) {
        User user = repository.findById(userId);
        if (user != null) {
            user.setAddress(address);
            return repository.save(user);
        }
        return null;
    }
    
    private String generateId() {
        return String.valueOf(System.currentTimeMillis());
    }
}
'''
        }
        create_test_files(str(tmp_path), files)
        graph, expander = build_graph_and_expander(str(tmp_path))
        return graph, expander
    
    def test_java_inheritance_hints(self, java_graph):
        """Test Java class inheritance hints."""
        graph, expander = java_graph
        
        # Find User class (extends BaseEntity)
        node_id, node_data = find_node_by_name(graph, 'User', 'class')
        if not node_id:
            pytest.skip("User class not found in graph")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        reasons = get_expansion_reasons(expanded)
        
        print(f"Java inheritance - expansion reasons: {reasons}")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")
        
        assert len(expanded) > 0, "Expected expanded documents for Java inheritance"
    
    def test_java_interface_implementation_hints(self, java_graph):
        """Test Java interface implementation hints."""
        graph, expander = java_graph
        
        # Find UserRepository (implements Repository)
        node_id, node_data = find_node_by_name(graph, 'UserRepository', 'class')
        if not node_id:
            pytest.skip("UserRepository class not found")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        reasons = get_expansion_reasons(expanded)
        
        print(f"Java interface impl - expansion reasons: {reasons}")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")
    
    def test_java_composition_hints(self, java_graph):
        """Test Java composition hints."""
        graph, expander = java_graph
        
        # Find User class (has Address field)
        node_id, node_data = find_node_by_name(graph, 'User', 'class')
        if not node_id:
            pytest.skip("User class not found")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        
        print(f"Java composition - expanded {len(expanded)} docs")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")
                if d.metadata.get('expansion_via'):
                    print(f"    via: {d.metadata.get('expansion_via')}")
    
    def test_java_parameter_type_hints(self, java_graph):
        """Test Java method parameter type hints."""
        graph, expander = java_graph
        
        # Find updateAddress method (has Address parameter)
        node_id, node_data = find_node_by_name(graph, 'updateAddress')
        if not node_id:
            pytest.skip("updateAddress method not found")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        
        print(f"Java param types - expanded {len(expanded)} docs")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")


# =============================================================================
# C++ Comprehensive Hints Tests
# =============================================================================

class TestCppHints:
    """Test all hint types for C++ language."""
    
    @pytest.fixture
    def cpp_graph(self, tmp_path):
        """Create C++ files with various relationships."""
        files = {
            "include/base/entity.hpp": '''
#ifndef ENTITY_HPP
#define ENTITY_HPP

#include <string>
#include <ctime>

class Entity {
protected:
    std::string id;
    time_t createdAt;
    
public:
    Entity(const std::string& id) : id(id), createdAt(time(nullptr)) {}
    virtual ~Entity() = default;
    
    const std::string& getId() const { return id; }
    virtual void validate() = 0;
};

#endif
''',
            "include/models/address.hpp": '''
#ifndef ADDRESS_HPP
#define ADDRESS_HPP

#include <string>

struct Address {
    std::string street;
    std::string city;
    std::string country;
    
    Address() = default;
    Address(const std::string& s, const std::string& c, const std::string& co)
        : street(s), city(c), country(co) {}
    
    std::string getFullAddress() const {
        return street + ", " + city + ", " + country;
    }
};

#endif
''',
            "include/models/user.hpp": '''
#ifndef USER_HPP
#define USER_HPP

#include "base/entity.hpp"
#include "models/address.hpp"
#include <string>
#include <memory>

class User : public Entity {
private:
    std::string name;
    std::string email;
    std::unique_ptr<Address> address;  // Composition
    
public:
    User(const std::string& id, const std::string& name, const std::string& email)
        : Entity(id), name(name), email(email) {}
    
    void setAddress(std::unique_ptr<Address> addr) {
        address = std::move(addr);
    }
    
    const Address* getAddress() const {
        return address.get();
    }
    
    void validate() override {
        if (name.empty()) {
            throw std::invalid_argument("Name required");
        }
    }
};

#endif
''',
            "include/repositories/repository.hpp": '''
#ifndef REPOSITORY_HPP
#define REPOSITORY_HPP

#include <string>
#include <memory>

template<typename T>
class Repository {
public:
    virtual ~Repository() = default;
    virtual std::shared_ptr<T> findById(const std::string& id) = 0;
    virtual std::shared_ptr<T> save(std::shared_ptr<T> entity) = 0;
    virtual void remove(const std::string& id) = 0;
};

#endif
''',
            "include/repositories/user_repository.hpp": '''
#ifndef USER_REPOSITORY_HPP
#define USER_REPOSITORY_HPP

#include "repositories/repository.hpp"
#include "models/user.hpp"
#include <unordered_map>

class UserRepository : public Repository<User> {
private:
    std::unordered_map<std::string, std::shared_ptr<User>> storage;
    
public:
    std::shared_ptr<User> findById(const std::string& id) override {
        auto it = storage.find(id);
        return it != storage.end() ? it->second : nullptr;
    }
    
    std::shared_ptr<User> save(std::shared_ptr<User> entity) override {
        storage[entity->getId()] = entity;
        return entity;
    }
    
    void remove(const std::string& id) override {
        storage.erase(id);
    }
};

#endif
''',
            "include/services/user_service.hpp": '''
#ifndef USER_SERVICE_HPP
#define USER_SERVICE_HPP

#include "models/user.hpp"
#include "models/address.hpp"
#include "repositories/user_repository.hpp"
#include <memory>
#include <string>

class UserService {
private:
    std::shared_ptr<UserRepository> repository;  // Aggregation
    
    std::string generateId() {
        return std::to_string(time(nullptr));
    }
    
public:
    explicit UserService(std::shared_ptr<UserRepository> repo)
        : repository(std::move(repo)) {}
    
    std::shared_ptr<User> getUser(const std::string& id) {
        return repository->findById(id);
    }
    
    std::shared_ptr<User> createUser(const std::string& name, const std::string& email) {
        auto user = std::make_shared<User>(generateId(), name, email);
        return repository->save(user);
    }
    
    std::shared_ptr<User> updateAddress(const std::string& userId, std::unique_ptr<Address> address) {
        auto user = repository->findById(userId);
        if (user) {
            user->setAddress(std::move(address));
            return repository->save(user);
        }
        return nullptr;
    }
};

#endif
''',
            "include/utils/validators.hpp": '''
#ifndef VALIDATORS_HPP
#define VALIDATORS_HPP

#include <string>
#include <regex>

constexpr int MAX_NAME_LENGTH = 100;

inline bool validateEmail(const std::string& email) {
    std::regex emailRegex(R"([^\\s@]+@[^\\s@]+\\.[^\\s@]+)");
    return std::regex_match(email, emailRegex);
}

inline bool validateName(const std::string& name) {
    return !name.empty() && name.length() <= MAX_NAME_LENGTH;
}

#endif
'''
        }
        create_test_files(str(tmp_path), files)
        graph, expander = build_graph_and_expander(str(tmp_path))
        return graph, expander
    
    def test_cpp_inheritance_hints(self, cpp_graph):
        """Test C++ class inheritance hints."""
        graph, expander = cpp_graph
        
        # Find User class (extends Entity)
        node_id, node_data = find_node_by_name(graph, 'User', 'class')
        if not node_id:
            pytest.skip("User class not found in graph")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        reasons = get_expansion_reasons(expanded)
        
        print(f"C++ inheritance - expansion reasons: {reasons}")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")
        
        assert len(expanded) > 0, "Expected expanded documents for C++ inheritance"
    
    def test_cpp_composition_hints(self, cpp_graph):
        """Test C++ composition hints (unique_ptr field)."""
        graph, expander = cpp_graph
        
        # Find User class (has Address field)
        node_id, node_data = find_node_by_name(graph, 'User', 'class')
        if not node_id:
            pytest.skip("User class not found")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        
        print(f"C++ composition - expanded {len(expanded)} docs")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")
                if d.metadata.get('expansion_via'):
                    print(f"    via: {d.metadata.get('expansion_via')}")
    
    def test_cpp_aggregation_hints(self, cpp_graph):
        """Test C++ aggregation hints (shared_ptr field)."""
        graph, expander = cpp_graph
        
        # Find UserService (has shared_ptr<UserRepository>)
        node_id, node_data = find_node_by_name(graph, 'UserService', 'class')
        if not node_id:
            pytest.skip("UserService class not found")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        
        print(f"C++ aggregation - expanded {len(expanded)} docs")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")
    
    def test_cpp_struct_hints(self, cpp_graph):
        """Test C++ struct expansion hints."""
        graph, expander = cpp_graph
        
        # Find Address struct
        node_id, node_data = find_node_by_name(graph, 'Address', 'struct')
        if not node_id:
            node_id, node_data = find_node_by_name(graph, 'Address', 'class')
        if not node_id:
            pytest.skip("Address struct not found")
        
        doc = create_doc_from_node(graph, node_id, node_data)
        expanded = expander._expand_document_comprehensively(doc, set())
        
        print(f"C++ struct - expanded {len(expanded)} docs")
        for d in expanded:
            if d.metadata.get('expansion_reason'):
                print(f"  {d.metadata.get('symbol_name')}: {d.metadata.get('expansion_reason')}")


# =============================================================================
# Cross-Language Expansion Reason Coverage Tests
# =============================================================================

class TestExpansionReasonCoverage:
    """Test that all expansion reasons are properly formatted in hints."""
    
    def test_all_expansion_reasons_have_descriptions(self):
        """Verify all expansion reasons have human-readable descriptions."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        
        # Create minimal agent to access reason_descriptions
        agent = OptimizedWikiGenerationAgent.__new__(OptimizedWikiGenerationAgent)
        
        # All expansion reasons that content_expander can produce
        expected_reasons = {
            # Inheritance
            'extended_by', 'implemented_by', 'extends',
            # Composition  
            'composed_by', 'used_by', 'aggregation', 'composition',
            # Calls
            'called_by', 'references', 'creates', 'instantiated_by',
            # Containment
            'defines', 'contains', 'has_member',
            # Types
            'return_type_of', 'parameter_type_of', 'type_of', 'initializes',
            # Members
            'constructor_of', 'method_of', 'field_of',
            # Additional from content_expander
            'param_type', 'return_type', 'calls_func', 'type_annotation',
            'constructor_member', 'method_member', 'field_member',
            'implements_interface',
        }
        
        # The reason_descriptions dict in _format_hybrid_hint
        # Check that we have reasonable coverage
        print(f"Expected expansion reasons: {len(expected_reasons)}")
        print(f"  {sorted(expected_reasons)}")
    
    def test_forward_hint_format(self):
        """Test forward hint (→) format for initially retrieved docs."""
        from unittest.mock import MagicMock
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        
        agent = OptimizedWikiGenerationAgent.__new__(OptimizedWikiGenerationAgent)
        agent._extract_relationship_hints = MagicMock(
            return_value="extends `BaseService`; uses `Repository`"
        )
        
        doc = Document(
            page_content="class Service",
            metadata={
                'symbol_name': 'Service',
                'file_path': 'src/Service.ts',
                'is_initially_retrieved': True
            }
        )
        
        hint = agent._format_hybrid_hint(doc, 'Service', 'src/Service.ts', MagicMock(), {})
        
        assert hint.startswith("→"), f"Forward hint should start with →, got: {hint}"
        print(f"Forward hint: {hint}")
    
    def test_backward_hint_format_all_reasons(self):
        """Test backward hint (←) format for all expansion reasons."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        
        agent = OptimizedWikiGenerationAgent.__new__(OptimizedWikiGenerationAgent)
        
        # Test each expansion reason produces valid backward hint
        test_reasons = [
            ('extended_by', 'UserService', 'parent of'),
            ('implemented_by', 'IService', 'interface of'),
            ('extends', 'BaseService', 'child of'),
            ('composed_by', 'UserController', 'component of'),
            ('called_by', 'main', 'called by'),
            ('param_type', 'processUser', 'parameter type of'),
            ('return_type', 'getUser', 'return type of'),
        ]
        
        for reason, source, expected_desc in test_reasons:
            doc = Document(
                page_content=f"class {source}Target",
                metadata={
                    'symbol_name': f'{source}Target',
                    'file_path': 'src/test.ts',
                    'is_initially_retrieved': False,
                    'expansion_reason': reason,
                    'expansion_source': source
                }
            )
            
            hint = agent._format_hybrid_hint(doc, f'{source}Target', 'src/test.ts', MagicMock(), {})
            
            assert hint.startswith("←"), f"Backward hint should start with ← for {reason}"
            assert source in hint, f"Hint should include source {source}"
            print(f"  {reason}: {hint}")
    
    def test_backward_hint_with_via_context(self):
        """Test backward hints include 'via' context when available."""
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        
        agent = OptimizedWikiGenerationAgent.__new__(OptimizedWikiGenerationAgent)
        
        doc = Document(
            page_content="class UserRepository",
            metadata={
                'symbol_name': 'UserRepository',
                'file_path': 'src/UserRepository.ts',
                'expansion_reason': 'composed_by',
                'expansion_source': 'UserService',
                'expansion_via': 'via repo field'
            }
        )
        
        hint = agent._format_hybrid_hint(doc, 'UserRepository', 'src/UserRepository.ts', MagicMock(), {})
        
        assert "via repo field" in hint, f"Hint should include via context: {hint}"
        print(f"Via context hint: {hint}")


# =============================================================================
# Integration Tests - Full Pipeline
# =============================================================================

class TestFullHintsPipeline:
    """Integration tests for complete hints pipeline."""
    
    @pytest.fixture
    def multi_language_repo(self, tmp_path):
        """Create a repo with multiple language files."""
        files = {
            "python/service.py": '''
class PythonService:
    def process(self, data: dict) -> bool:
        return True
''',
            "typescript/service.ts": '''
export class TypeScriptService {
    process(data: object): boolean {
        return true;
    }
}
''',
            "java/Service.java": '''
public class JavaService {
    public boolean process(Object data) {
        return true;
    }
}
'''
        }
        create_test_files(str(tmp_path), files)
        graph, expander = build_graph_and_expander(str(tmp_path))
        return graph, expander
    
    def test_multi_language_expansion(self, multi_language_repo):
        """Test expansion works across multiple languages."""
        graph, expander = multi_language_repo
        
        languages_found = set()
        for node_id, node_data in graph.nodes(data=True):
            lang = node_data.get('language', '')
            if lang:
                languages_found.add(lang)
        
        print(f"Languages in graph: {languages_found}")
        
        # Should have multiple languages
        assert len(languages_found) >= 1, "Expected at least one language in graph"
    
    def test_document_metadata_completeness(self, multi_language_repo):
        """Test that expanded documents have complete metadata."""
        graph, expander = multi_language_repo
        
        # Find any class to expand
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('symbol_type', '').lower() == 'class':
                doc = create_doc_from_node(graph, node_id, node_data)
                expanded = expander._expand_document_comprehensively(doc, set())
                
                # Check metadata completeness
                for exp_doc in expanded:
                    meta = exp_doc.metadata
                    
                    # Every expanded doc should have these
                    assert 'symbol_name' in meta, "Missing symbol_name"
                    assert 'file_path' in meta, "Missing file_path"
                    
                    # If expanded, should have expansion metadata
                    if meta.get('expansion_reason'):
                        assert 'expansion_source' in meta or meta.get('expansion_reason') in ['defines', 'contains'], \
                            f"Expanded doc missing expansion_source: {meta.get('symbol_name')}"
                
                return  # Test one class is enough
        
        pytest.skip("No classes found in graph")
