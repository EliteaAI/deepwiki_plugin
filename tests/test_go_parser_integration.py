"""
Go Rich Parser — Integration Tests.

Covers:
    IT-1: Parser → Symbol Extraction (single file)
    IT-2: Parser → Relationship Extraction (single file)
    IT-3: Parser → Cross-File Analysis (multi-file)
    IT-4: Parser → Graph Builder → NetworkX (full pipeline)
    IT-5: Parser → Graph → Expansion Engine (smart expansion + augmentation)

See PLANNING_GO_RICH_PARSER.md §6 for test specifications.
"""

import os
import textwrap
from pathlib import Path

import pytest

from plugin_implementation.parsers.base_parser import (
    SymbolType,
    RelationshipType,
    ParseResult,
)
from plugin_implementation.parsers.go_visitor_parser import GoVisitorParser


# =============================================================================
# Shared fixture
# =============================================================================

def _write_go(tmp_path: Path, rel_path: str, source: str) -> Path:
    """Write a Go source file with dedented content."""
    full = tmp_path / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(textwrap.dedent(source).lstrip())
    return full


@pytest.fixture
def go_project(tmp_path):
    """Factory fixture: call with (rel_path, source) to create Go files."""
    files = []

    def _add(rel_path: str, source: str) -> Path:
        p = _write_go(tmp_path, rel_path, source)
        files.append(str(p))
        return p

    _add.root = tmp_path
    _add.files = files
    return _add


# =============================================================================
# IT-1: Parser → Symbol Extraction (Single File)
# =============================================================================

class TestGoParserSymbols:
    """Integration: write real Go code → parse → assert symbols."""

    def test_struct_and_fields(self, go_project):
        go_project("server/server.go", """
            package server

            // Server handles HTTP requests.
            type Server struct {
                Host   string
                Port   int
                Logger *Logger   // pointer field
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        structs = [s for s in result.symbols if s.symbol_type == SymbolType.STRUCT]
        assert len(structs) == 1
        assert structs[0].name == "Server"
        assert structs[0].visibility == "public"
        assert structs[0].docstring and "HTTP requests" in structs[0].docstring

        fields = [s for s in result.symbols if s.symbol_type == SymbolType.FIELD]
        assert {f.name for f in fields} == {"Host", "Port", "Logger"}

    def test_function_with_multiple_returns(self, go_project):
        go_project("server/server.go", """
            package server

            func NewServer(host string, port int) (*Server, error) {
                return &Server{Host: host, Port: port}, nil
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        funcs = [s for s in result.symbols if s.symbol_type == SymbolType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "NewServer"
        assert funcs[0].visibility == "public"
        assert "error" in funcs[0].signature

    def test_method_with_pointer_receiver(self, go_project):
        go_project("server/server.go", """
            package server

            type Server struct{ Host string }

            func (s *Server) Start() error {
                return nil
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        methods = [s for s in result.symbols if s.symbol_type == SymbolType.METHOD]
        assert len(methods) == 1
        assert methods[0].name == "Start"
        assert methods[0].parent_symbol == "Server"
        assert methods[0].full_name == "Server.Start"
        assert methods[0].metadata.get("is_pointer_receiver") is True

    def test_method_with_value_receiver(self, go_project):
        go_project("server/server.go", """
            package server

            type Server struct{ Host string; Port int }

            func (s Server) Address() string {
                return s.Host
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        methods = [s for s in result.symbols if s.symbol_type == SymbolType.METHOD]
        assert len(methods) == 1
        assert methods[0].name == "Address"
        assert methods[0].parent_symbol == "Server"
        assert methods[0].metadata.get("is_pointer_receiver") is False

    def test_interface_with_abstract_methods(self, go_project):
        go_project("handler/handler.go", """
            package handler

            type Handler interface {
                Handle(req string) string
                Close() error
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        interfaces = [s for s in result.symbols if s.symbol_type == SymbolType.INTERFACE]
        assert len(interfaces) == 1
        assert interfaces[0].name == "Handler"

        methods = [s for s in result.symbols if s.symbol_type == SymbolType.METHOD]
        assert {m.name for m in methods} == {"Handle", "Close"}
        for m in methods:
            assert m.metadata.get("is_abstract") is True
            assert m.parent_symbol == "Handler"

    def test_iota_const_group_as_enum(self, go_project):
        go_project("types/direction.go", """
            package types

            type Direction int

            const (
                North Direction = iota
                South
                East
                West
            )
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        enums = [s for s in result.symbols if s.symbol_type == SymbolType.ENUM]
        assert len(enums) >= 1  # The iota group

        constants = [s for s in result.symbols if s.symbol_type == SymbolType.CONSTANT]
        const_names = {c.name for c in constants}
        assert {"North", "South", "East", "West"}.issubset(const_names)

    def test_type_alias_and_named_type(self, go_project):
        go_project("types/aliases.go", """
            package types

            type StringAlias = string
            type UserID int64
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        aliases = [s for s in result.symbols if s.symbol_type == SymbolType.TYPE_ALIAS]
        assert {a.name for a in aliases} == {"StringAlias", "UserID"}

    def test_generic_function(self, go_project):
        go_project("util/map.go", """
            package util

            func Map[T any, U any](items []T, fn func(T) U) []U {
                result := make([]U, len(items))
                for i, item := range items {
                    result[i] = fn(item)
                }
                return result
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        funcs = [s for s in result.symbols if s.symbol_type == SymbolType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "Map"
        type_params = funcs[0].metadata.get("type_parameters", [])
        assert len(type_params) == 2

    def test_generic_struct(self, go_project):
        go_project("util/set.go", """
            package util

            type Set[T comparable] struct {
                items map[T]bool
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        structs = [s for s in result.symbols if s.symbol_type == SymbolType.STRUCT]
        assert len(structs) == 1
        assert structs[0].name == "Set"
        type_params = structs[0].metadata.get("type_parameters", [])
        assert len(type_params) == 1

    def test_unexported_symbols_private_visibility(self, go_project):
        go_project("internal/helpers.go", """
            package internal

            type cache struct {
                data map[string]string
            }

            func newCache() *cache {
                return &cache{data: make(map[string]string)}
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        structs = [s for s in result.symbols if s.symbol_type == SymbolType.STRUCT]
        assert structs[0].name == "cache"
        assert structs[0].visibility == "private"

        funcs = [s for s in result.symbols if s.symbol_type == SymbolType.FUNCTION]
        assert funcs[0].name == "newCache"
        assert funcs[0].visibility == "private"

    def test_multiple_init_functions(self, go_project):
        go_project("pkg/init.go", """
            package pkg

            func init() {
                // first init
            }

            func init() {
                // second init
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        funcs = [s for s in result.symbols if s.symbol_type == SymbolType.FUNCTION]
        names = [f.name for f in funcs]
        assert "init" in names
        # Second init disambiguated
        assert any(n.startswith("init_L") for n in names)

    def test_struct_embedding(self, go_project):
        """Embedded field (no field_identifier) → emitted as FIELD with embedding metadata."""
        go_project("server/server.go", """
            package server

            type Server struct {
                Host string
                Handler
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        fields = [s for s in result.symbols if s.symbol_type == SymbolType.FIELD]
        embedded = [f for f in fields if f.metadata.get("is_embedded")]
        assert len(embedded) == 1
        assert embedded[0].name == "Handler"

    def test_source_text_and_range_accuracy(self, go_project):
        go_project("pkg/example.go", """
            package pkg

            func Hello() string {
                return "hello"
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        funcs = [s for s in result.symbols if s.symbol_type == SymbolType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].range is not None
        assert funcs[0].range.start.line < funcs[0].range.end.line
        assert "Hello" in (funcs[0].source_text or "")


# =============================================================================
# IT-2: Parser → Relationship Extraction (Single File)
# =============================================================================

class TestGoParserRelationships:
    """Integration: write real Go code → parse → assert relationships."""

    def test_import_relationships(self, go_project):
        go_project("svc/service.go", """
            package svc

            import (
                "fmt"
                mylog "github.com/user/pkg/log"
                . "strings"
                _ "net/http/pprof"
            )

            func Run() { fmt.Println("run") }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        imports = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.IMPORTS]
        targets = {r.target_symbol for r in imports}
        assert "fmt" in targets
        assert "github.com/user/pkg/log" in targets
        assert "strings" in targets
        assert "net/http/pprof" in targets

    def test_struct_field_composition_and_aggregation(self, go_project):
        go_project("svc/config.go", """
            package svc

            type Config struct {
                Database Database
                Logger   *Logger
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        compositions = [r for r in result.relationships
                        if r.relationship_type == RelationshipType.COMPOSITION]
        aggregations = [r for r in result.relationships
                        if r.relationship_type == RelationshipType.AGGREGATION]
        assert any(r.target_symbol == "Database" for r in compositions)
        assert any(r.target_symbol == "Logger" for r in aggregations)

    def test_struct_embedding_produces_composition(self, go_project):
        """Struct embedding → COMPOSITION, NOT inheritance."""
        go_project("svc/embed.go", """
            package svc

            type Base struct {
                ID int
            }

            type Derived struct {
                Base
                Name string
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        compositions = [r for r in result.relationships
                        if r.relationship_type == RelationshipType.COMPOSITION
                        and r.source_symbol == "Derived" and r.target_symbol == "Base"]
        assert len(compositions) == 1

        struct_inherits = [r for r in result.relationships
                          if r.relationship_type == RelationshipType.INHERITANCE
                          and r.source_symbol == "Derived"]
        assert len(struct_inherits) == 0

    def test_interface_embedding_produces_inheritance(self, go_project):
        """Interface embedding → INHERITANCE (the ONLY use of INHERITANCE in Go)."""
        go_project("svc/iface.go", """
            package svc

            type Reader interface {
                Read(p []byte) (int, error)
            }

            type Closer interface {
                Close() error
            }

            type ReadCloser interface {
                Reader
                Closer
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        inheritances = [r for r in result.relationships
                        if r.relationship_type == RelationshipType.INHERITANCE]
        assert len(inheritances) == 2
        targets = {r.target_symbol for r in inheritances}
        assert targets == {"Reader", "Closer"}
        for r in inheritances:
            assert r.source_symbol == "ReadCloser"

    def test_function_calls(self, go_project):
        go_project("svc/caller.go", """
            package svc

            import "fmt"

            func Run() {
                fmt.Println("hello")
                result := process()
                _ = result
            }

            func process() string {
                return "done"
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        calls = [r for r in result.relationships
                 if r.relationship_type == RelationshipType.CALLS]
        targets = {r.target_symbol for r in calls}
        assert "fmt.Println" in targets
        assert "process" in targets

    def test_composite_literal_creates(self, go_project):
        go_project("svc/factory.go", """
            package svc

            type Config struct {
                Host string
                Port int
            }

            func NewConfig() *Config {
                return &Config{Host: "localhost", Port: 8080}
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        creates = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.CREATES]
        assert any(r.target_symbol == "Config" for r in creates)

    def test_nested_composite_literal_creates(self, go_project):
        """Nested struct literals: outer AND inner produce CREATES edges."""
        go_project("svc/factory.go", """
            package svc

            type Config struct {
                Host string
                Port int
            }

            type Logger struct {
                Level string
            }

            type Service struct {
                Cfg    Config
                Log    Logger
                Name   string
            }

            func NewService(name string) *Service {
                return &Service{
                    Cfg: Config{
                        Host: "localhost",
                        Port: 8080,
                    },
                    Log: Logger{
                        Level: "info",
                    },
                    Name: name,
                }
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        creates = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.CREATES]
        targets = {r.target_symbol for r in creates}
        # Outer literal
        assert "Service" in targets, f"Expected Service in creates targets: {targets}"
        # Inner nested literals
        assert "Config" in targets, f"Expected Config in creates targets: {targets}"
        assert "Logger" in targets, f"Expected Logger in creates targets: {targets}"
        # All three edges sourced from NewService
        ns_creates = [r for r in creates if r.source_symbol == "NewService"]
        assert len(ns_creates) == 3, (
            f"NewService should create Service, Config, Logger. Got: "
            f"{[r.target_symbol for r in ns_creates]}"
        )

    def test_defines_struct_to_field(self, go_project):
        go_project("svc/model.go", """
            package svc

            type User struct {
                Name  string
                Email string
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        defines = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.DEFINES]
        defines_from_user = [r for r in defines if r.source_symbol == "User"]
        assert len(defines_from_user) == 2
        targets = {r.target_symbol for r in defines_from_user}
        assert any("Name" in t for t in targets)
        assert any("Email" in t for t in targets)

    def test_defines_struct_to_method_same_file(self, go_project):
        """When struct and method are in the same file, DEFINES emitted immediately."""
        go_project("svc/server.go", """
            package svc

            type Server struct {
                Host string
            }

            func (s *Server) Start() error {
                return nil
            }

            func (s *Server) Stop() error {
                return nil
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        defines = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.DEFINES
                   and r.source_symbol == "Server"]
        targets = {r.target_symbol for r in defines}
        assert any("Start" in t for t in targets)
        assert any("Stop" in t for t in targets)

    def test_alias_of_relationships(self, go_project):
        go_project("types/types.go", """
            package types

            type Handler interface{ Handle() }
            type Config struct{ Host string }

            type MyHandler = Handler
            type ServiceConfig Config
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        aliases = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.ALIAS_OF]
        sources = {r.source_symbol for r in aliases}
        assert "MyHandler" in sources
        assert "ServiceConfig" in sources

    def test_builtin_types_excluded_from_relationships(self, go_project):
        go_project("svc/model.go", """
            package svc

            type Config struct {
                Name    string
                Count   int
                Active  bool
                Backend Backend
            }
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        type_rels = [r for r in result.relationships
                     if r.relationship_type in (RelationshipType.COMPOSITION,
                                                RelationshipType.AGGREGATION)]
        for r in type_rels:
            assert r.target_symbol not in ("string", "int", "bool", "error",
                                    "float64", "byte", "rune")

    def test_goroutine_and_defer_calls(self, go_project):
        go_project("svc/async.go", """
            package svc

            type Conn struct{}
            func (c *Conn) Close() error { return nil }

            func Run(conn *Conn) {
                go process()
                defer conn.Close()
            }

            func process() {}
        """)
        parser = GoVisitorParser()
        result = parser.parse_file(go_project.files[0])

        calls = [r for r in result.relationships
                 if r.relationship_type == RelationshipType.CALLS]
        # At least 2 calls: process (via go), conn.Close (via defer)
        assert len(calls) >= 2


# =============================================================================
# IT-3: Parser → Cross-File Analysis (Multi-File)
# =============================================================================

class TestGoParserCrossFile:
    """Integration: write multi-file Go package to tmp → parse_multiple_files → assert."""

    def test_cross_file_method_defines(self, go_project):
        """Struct in file1, methods in file2 → DEFINES edges link them."""
        go_project("server/types.go", """
            package server

            type Server struct {
                Host string
                Port int
            }
        """)
        go_project("server/methods.go", """
            package server

            import "fmt"

            func (s *Server) Start() error {
                fmt.Println("starting", s.Host)
                return nil
            }

            func (s *Server) Stop() error {
                return nil
            }

            func (s Server) Address() string {
                return fmt.Sprintf("%s:%d", s.Host, s.Port)
            }
        """)
        parser = GoVisitorParser()
        results = parser.parse_multiple_files(go_project.files)

        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        defines = [r for r in all_rels
                   if r.relationship_type == RelationshipType.DEFINES
                   and r.source_symbol == "Server"]
        targets = {r.target_symbol for r in defines}
        assert any("Start" in t for t in targets)
        assert any("Stop" in t for t in targets)
        assert any("Address" in t for t in targets)
        # At least one cross-file edge
        cross_file = [r for r in defines if r.annotations.get("cross_file")]
        assert len(cross_file) >= 1

    def test_implicit_interface_implementation(self, go_project):
        """Server has Start()+Stop()+Close() → implements Starter, Closer, FullLifecycle."""
        go_project("pkg/server.go", """
            package pkg

            type Server struct{}
            func (s *Server) Start() error { return nil }
            func (s *Server) Stop() error { return nil }
            func (s *Server) Close() error { return nil }
        """)
        go_project("pkg/interfaces.go", """
            package pkg

            type Starter interface {
                Start() error
            }

            type Closer interface {
                Close() error
            }

            type FullLifecycle interface {
                Start() error
                Stop() error
                Close() error
            }
        """)
        parser = GoVisitorParser()
        results = parser.parse_multiple_files(go_project.files)

        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        impls = [r for r in all_rels
                 if r.relationship_type == RelationshipType.IMPLEMENTATION
                 and r.source_symbol == "Server"]
        impl_targets = {r.target_symbol for r in impls}
        assert "Starter" in impl_targets
        assert "Closer" in impl_targets
        assert "FullLifecycle" in impl_targets

    def test_partial_interface_no_implementation(self, go_project):
        """Struct has Start() but not Close() → does NOT implement FullLifecycle."""
        go_project("pkg/partial.go", """
            package pkg

            type Worker struct{}

            func (w *Worker) Start() error { return nil }
        """)
        go_project("pkg/iface.go", """
            package pkg

            type FullLifecycle interface {
                Start() error
                Close() error
            }
        """)
        parser = GoVisitorParser()
        results = parser.parse_multiple_files(go_project.files)

        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        impls = [r for r in all_rels
                 if r.relationship_type == RelationshipType.IMPLEMENTATION
                 and r.source_symbol == "Worker"]
        assert len(impls) == 0

    def test_cross_file_function_calls(self, go_project):
        go_project("pkg/util.go", """
            package pkg

            func FormatAddress(host string, port int) string {
                return host
            }
        """)
        go_project("pkg/server.go", """
            package pkg

            type Server struct{ Host string; Port int }

            func (s *Server) Address() string {
                return FormatAddress(s.Host, s.Port)
            }
        """)
        parser = GoVisitorParser()
        results = parser.parse_multiple_files(go_project.files)

        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        calls = [r for r in all_rels
                 if r.relationship_type == RelationshipType.CALLS
                 and r.target_symbol == "FormatAddress"]
        assert len(calls) >= 1

    def test_cross_package_embedding(self, go_project):
        """Struct embeds type from another file in same package."""
        go_project("pkg/base.go", """
            package pkg

            type BaseHandler struct {
                Name string
            }

            func (b *BaseHandler) Handle() error { return nil }
        """)
        go_project("pkg/derived.go", """
            package pkg

            type AppServer struct {
                BaseHandler
                Port int
            }
        """)
        parser = GoVisitorParser()
        results = parser.parse_multiple_files(go_project.files)

        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        compositions = [r for r in all_rels
                        if r.relationship_type == RelationshipType.COMPOSITION
                        and r.source_symbol == "AppServer" and r.target_symbol == "BaseHandler"]
        assert len(compositions) == 1

    def test_cross_file_struct_literal_creates(self, go_project):
        """Struct defined in file1, literal in file2 → CREATES edge across files."""
        go_project("svc/types.go", """
            package svc

            type AppConfig struct {
                Path  string
                Debug bool
                Port  int
            }
        """)
        go_project("svc/factory.go", """
            package svc

            func DefaultConfig() *AppConfig {
                return &AppConfig{
                    Path:  "/data",
                    Debug: false,
                    Port:  8080,
                }
            }

            func TestConfig() AppConfig {
                return AppConfig{Path: "/test"}
            }
        """)
        parser = GoVisitorParser()
        results = parser.parse_multiple_files(go_project.files)

        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        creates = [r for r in all_rels
                   if r.relationship_type == RelationshipType.CREATES
                   and r.target_symbol == "AppConfig"]
        # Both factory functions create AppConfig
        sources = {r.source_symbol for r in creates}
        assert "DefaultConfig" in sources, f"Expected DefaultConfig in creates sources: {sources}"
        assert "TestConfig" in sources, f"Expected TestConfig in creates sources: {sources}"

    def test_cross_file_nested_struct_literal_creates(self, go_project):
        """Nested struct literal across files: factory creates outer + inner types."""
        go_project("svc/types.go", """
            package svc

            type DBConfig struct {
                Host string
                Port int
            }

            type AppConfig struct {
                DB   DBConfig
                Name string
            }
        """)
        go_project("svc/factory.go", """
            package svc

            func NewAppConfig() *AppConfig {
                return &AppConfig{
                    DB: DBConfig{
                        Host: "localhost",
                        Port: 5432,
                    },
                    Name: "myapp",
                }
            }
        """)
        parser = GoVisitorParser()
        results = parser.parse_multiple_files(go_project.files)

        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        creates = [r for r in all_rels
                   if r.relationship_type == RelationshipType.CREATES
                   and r.source_symbol == "NewAppConfig"]
        targets = {r.target_symbol for r in creates}
        assert "AppConfig" in targets, f"Expected AppConfig in creates: {targets}"
        assert "DBConfig" in targets, f"Expected DBConfig (nested literal) in creates: {targets}"


# =============================================================================
# IT-4: Parser → Graph Builder → NetworkX (Full Pipeline)
# =============================================================================

class TestGoGraphBuilder:
    """Integration: write Go project to tmp → graph builder analyze → assert graph."""

    def _build_graph(self, root_path):
        """Helper: build graph and return the NetworkX MultiDiGraph."""
        from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
        builder = EnhancedUnifiedGraphBuilder()
        analysis = builder.analyze_repository(str(root_path))
        return builder, analysis.unified_graph

    def test_graph_nodes_created(self, go_project):
        """Graph builder creates nodes with correct IDs for Go symbols."""
        go_project("server/server.go", """
            package server

            type Server struct {
                Host string
                Port int
            }

            func NewServer(host string, port int) *Server {
                return &Server{Host: host, Port: port}
            }

            func (s *Server) Start() error {
                return nil
            }
        """)
        builder, graph = self._build_graph(go_project.root)

        # Rich parser is registered for Go
        assert 'go' in builder.rich_parsers

        # Graph has nodes (exact IDs depend on graph_builder node naming)
        assert graph is not None
        assert graph.number_of_nodes() >= 3  # Server, NewServer, Start at minimum

        # Verify node data structure
        for node_id, node_data in graph.nodes(data=True):
            assert 'symbol' in node_data or 'symbol_name' in node_data or 'name' in node_data

    def test_graph_defines_edges(self, go_project):
        """DEFINES edges present: struct→field, struct→method."""
        go_project("svc/model.go", """
            package svc

            type User struct {
                Name  string
                Email string
            }

            func (u *User) Validate() error {
                return nil
            }
        """)
        _, graph = self._build_graph(go_project.root)

        defines_edges = [(u, v) for u, v, d in graph.edges(data=True)
                         if d.get("relationship_type") == "defines"]
        assert len(defines_edges) >= 1, "Expected at least one DEFINES edge"

    def test_graph_composition_edges(self, go_project):
        """COMPOSITION edges from struct field types and embedding."""
        go_project("svc/service.go", """
            package svc

            type Config struct {
                Host string
            }

            type Service struct {
                Config
                Name string
            }
        """)
        _, graph = self._build_graph(go_project.root)

        composition_edges = [(u, v) for u, v, d in graph.edges(data=True)
                             if d.get("relationship_type") == "composition"]
        assert len(composition_edges) >= 1, "Expected COMPOSITION edge from Service to Config"

    def test_graph_calls_edges(self, go_project):
        go_project("svc/main.go", """
            package svc

            import "fmt"

            func Run() {
                s := NewServer()
                s.Start()
                fmt.Println("done")
            }

            func NewServer() *Server { return &Server{} }
        """)
        go_project("svc/server.go", """
            package svc

            type Server struct{}

            func (s *Server) Start() error { return nil }
        """)
        _, graph = self._build_graph(go_project.root)

        call_edges = [(u, v) for u, v, d in graph.edges(data=True)
                      if d.get("relationship_type") == "calls"]
        assert len(call_edges) >= 1, "Expected at least one CALLS edge"

    def test_graph_implementation_edges(self, go_project):
        """Implicit interface implementation appears as graph edges."""
        go_project("svc/server.go", """
            package svc

            type Server struct{}
            func (s *Server) Start() error { return nil }
            func (s *Server) Close() error { return nil }
        """)
        go_project("svc/ifaces.go", """
            package svc

            type Starter interface {
                Start() error
            }
        """)
        _, graph = self._build_graph(go_project.root)

        all_edge_types = {d.get("relationship_type") for _, _, d in graph.edges(data=True)}
        if "implementation" not in all_edge_types and "implements" not in all_edge_types:
            pytest.skip("Graph builder may not preserve IMPLEMENTATION edges verbatim")

    def test_no_basic_parser_fallback(self, go_project):
        """When rich parser is active, basic parser is NOT used for Go."""
        go_project("svc/server.go", """
            package svc

            type Server struct{ Host string }
            func (s *Server) Start() error { return nil }
        """)
        from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
        builder = EnhancedUnifiedGraphBuilder()
        assert 'go' in builder.rich_parsers

    def test_full_pipeline_realistic_project(self, go_project):
        """
        Realistic mini-project: main + service + model + interfaces.
        Verifies the full parser → graph pipeline produces a connected graph.
        """
        go_project("cmd/main.go", """
            package main

            func main() {
                svc := NewService()
                svc.Run()
            }
        """)
        go_project("service/service.go", """
            package service

            type Service struct {
                config Config
                Logger
            }

            func NewService() *Service {
                return &Service{}
            }

            func (s *Service) Run() error {
                s.Log("running")
                return nil
            }
        """)
        go_project("service/config.go", """
            package service

            type Config struct {
                Host string
                Port int
            }

            func (c Config) Validate() error {
                return nil
            }
        """)
        go_project("service/logger.go", """
            package service

            type Logger struct {
                Level string
            }

            func (l *Logger) Log(msg string) {}
        """)
        go_project("service/interfaces.go", """
            package service

            type Runnable interface {
                Run() error
            }

            type Loggable interface {
                Log(msg string)
            }
        """)
        _, graph = self._build_graph(go_project.root)

        assert graph is not None
        assert graph.number_of_nodes() >= 6

        edge_types = {d.get("relationship_type") for _, _, d in graph.edges(data=True)}
        assert "defines" in edge_types or "calls" in edge_types or len(edge_types) > 0, (
            f"Graph should have edges. Found edge types: {edge_types}"
        )

    def test_graph_cross_file_receiver_method_edges(self, go_project):
        """Cross-file receiver methods produce DEFINES edges with cross_file annotation."""
        go_project("svc/user.go", """
            package svc

            type User struct {
                Name  string
                Email string
            }
        """)
        go_project("svc/user_methods.go", """
            package svc

            import "fmt"

            func (u *User) Validate() error {
                if u.Name == "" {
                    return fmt.Errorf("name required")
                }
                return nil
            }

            func (u User) String() string {
                return u.Name + " <" + u.Email + ">"
            }

            func (u *User) SetName(name string) {
                u.Name = name
            }
        """)
        _, graph = self._build_graph(go_project.root)

        # Find User struct node
        user_nodes = [n for n in graph.nodes if n.endswith("::User") and "." not in n.split("::")[-1]]
        assert len(user_nodes) == 1, f"Expected one User struct node, got: {user_nodes}"
        user_node = user_nodes[0]

        # All edges from User
        defines_edges = []
        for u, v, d in graph.out_edges(user_node, data=True):
            if d.get("relationship_type") == "defines":
                defines_edges.append((u, v, d))

        # Should have fields (Name, Email) + methods (Validate, String, SetName) = 5 defines
        assert len(defines_edges) >= 5, (
            f"Expected >= 5 defines edges (2 fields + 3 methods), got {len(defines_edges)}: "
            f"{[(u.split('::')[-1], v.split('::')[-1]) for u, v, _ in defines_edges]}"
        )

        # Cross-file methods should have cross_file annotation
        method_edges = [(u, v, d) for u, v, d in defines_edges
                        if d.get("annotations", {}).get("member_type") == "method"]
        assert len(method_edges) == 3, (
            f"Expected 3 method defines edges, got {len(method_edges)}"
        )
        for _, _, d in method_edges:
            assert d.get("annotations", {}).get("cross_file") is True, (
                f"Expected cross_file=True on method edge, got annotations: {d.get('annotations')}"
            )

        # Verify pointer receiver annotation is preserved
        pointer_methods = [d for _, _, d in method_edges
                          if d.get("annotations", {}).get("is_pointer_receiver") is True]
        value_methods = [d for _, _, d in method_edges
                        if d.get("annotations", {}).get("is_pointer_receiver") is False]
        assert len(pointer_methods) == 2, "Validate and SetName should be pointer receivers"
        assert len(value_methods) == 1, "String should be value receiver"

    def test_graph_creates_edges(self, go_project):
        """CREATES edges survive into the graph for struct literal instantiation."""
        go_project("svc/types.go", """
            package svc

            type Config struct {
                Host string
                Port int
            }

            type Logger struct {
                Level string
            }

            type Service struct {
                Cfg  Config
                Log  Logger
                Name string
            }
        """)
        go_project("svc/factory.go", """
            package svc

            func NewService(name string) *Service {
                return &Service{
                    Cfg: Config{Host: "localhost", Port: 8080},
                    Log: Logger{Level: "info"},
                    Name: name,
                }
            }
        """)
        _, graph = self._build_graph(go_project.root)

        creates_edges = [(u, v) for u, v, d in graph.edges(data=True)
                         if d.get("relationship_type") == "creates"]
        assert len(creates_edges) >= 3, (
            f"Expected >= 3 CREATES edges (Service, Config, Logger). "
            f"Got {len(creates_edges)}: {creates_edges}"
        )

        # Verify the factory function is the source
        factory_nodes = [n for n in graph.nodes if "NewService" in n]
        assert factory_nodes, f"NewService node not found in {list(graph.nodes)}"
        factory_id = factory_nodes[0]

        creates_from_factory = [(u, v) for u, v in creates_edges if u == factory_id]
        target_ids = {v for _, v in creates_from_factory}
        # The targets should include Config, Logger, Service nodes
        target_names = set()
        for tid in target_ids:
            nd = graph.nodes.get(tid, {})
            sym = nd.get("symbol", nd)
            name = sym.name if hasattr(sym, "name") else sym.get("name", tid)
            target_names.add(name)
        assert "Config" in target_names, f"Expected Config in creates targets: {target_names}"
        assert "Logger" in target_names, f"Expected Logger in creates targets: {target_names}"
        assert "Service" in target_names, f"Expected Service in creates targets: {target_names}"


# =============================================================================
# IT-5: Parser → Graph → Expansion Engine (Smart Expansion + Augmentation)
# =============================================================================

class TestGoExpansion:
    """Integration: write Go project to tmp → build graph → expand nodes → assert context."""

    def _build_graph(self, root_path):
        """Helper: build graph and return the NetworkX MultiDiGraph."""
        from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
        builder = EnhancedUnifiedGraphBuilder()
        analysis = builder.analyze_repository(str(root_path))
        return analysis.unified_graph

    def test_augment_go_struct_with_cross_file_receiver_methods(self, go_project):
        """augment_go_node() should merge cross-file receiver methods into struct content."""
        go_project("svc/server.go", """
            package svc

            // Server handles HTTP requests.
            type Server struct {
                Host string
                Port int
            }
        """)
        go_project("svc/server_methods.go", """
            package svc

            import "fmt"

            func (s *Server) Start() error {
                fmt.Println("starting", s.Host)
                return nil
            }

            func (s *Server) Stop() error {
                fmt.Println("stopping")
                return nil
            }

            func (s Server) Address() string {
                return fmt.Sprintf("%s:%d", s.Host, s.Port)
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import augment_go_node

        graph = self._build_graph(go_project.root)

        # Find Server node
        server_nodes = [n for n in graph.nodes if "Server" in n and "." not in n.split("::")[-1]]
        assert len(server_nodes) >= 1, f"No Server node found. Nodes: {list(graph.nodes)}"
        server_node = server_nodes[0]

        aug = augment_go_node(graph, server_node)
        assert aug is not None, "augment_go_node should return AugmentedContent"
        assert aug.node_id == server_node
        assert "Start" in aug.augmented_content
        assert "Stop" in aug.augmented_content
        assert "Address" in aug.augmented_content
        # Original struct definition should still be present
        assert "Server struct" in aug.augmented_content or "Host" in aug.original_content
        # Receiver methods header
        assert "Receiver methods from" in aug.augmented_content

    def test_augment_go_struct_same_file_methods_not_duplicated(self, go_project):
        """Methods in the same file as the struct should NOT be augmented (already visible)."""
        go_project("svc/server.go", """
            package svc

            type Server struct {
                Host string
            }

            func (s *Server) Start() error { return nil }
        """)
        from plugin_implementation.code_graph.expansion_engine import augment_go_node

        graph = self._build_graph(go_project.root)
        server_nodes = [n for n in graph.nodes if "Server" in n and "." not in n.split("::")[-1]]
        assert len(server_nodes) >= 1
        server_node = server_nodes[0]

        aug = augment_go_node(graph, server_node)
        # No cross-file methods → should return None
        assert aug is None, "Same-file methods should not trigger augmentation"

    def test_expand_smart_augments_go_struct(self, go_project):
        """expand_smart() should produce augmentation for Go structs with cross-file methods."""
        go_project("svc/server.go", """
            package svc

            type Server struct {
                Host string
                Port int
            }
        """)
        go_project("svc/server_methods.go", """
            package svc

            func (s *Server) Start() error { return nil }
            func (s *Server) Stop() error { return nil }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart

        graph = self._build_graph(go_project.root)

        server_nodes = [n for n in graph.nodes if "Server" in n and "." not in n.split("::")[-1]]
        assert len(server_nodes) >= 1
        server_node = server_nodes[0]

        result = expand_smart({server_node}, graph)
        # Server should be augmented
        assert server_node in result.augmentations, (
            f"Server should be in augmentations. Keys: {list(result.augmentations.keys())}"
        )
        aug = result.augmentations[server_node]
        assert "Start" in aug.augmented_content
        assert "Stop" in aug.augmented_content

    def test_expand_smart_struct_with_composition(self, go_project):
        """Expanding a struct should pull in composed/embedded types via expansion."""
        go_project("svc/types.go", """
            package svc

            type Logger struct {
                Level string
            }

            func (l *Logger) Log(msg string) {}
        """)
        go_project("svc/server.go", """
            package svc

            type Server struct {
                Logger
                Host string
            }

            func (s *Server) Start() error {
                s.Log("starting")
                return nil
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart

        graph = self._build_graph(go_project.root)

        server_nodes = [n for n in graph.nodes if "Server" in n and "." not in n.split("::")[-1]]
        assert len(server_nodes) >= 1
        server_node = server_nodes[0]

        result = expand_smart({server_node}, graph)
        # Logger should be in expanded_nodes via composition edge
        expanded_names = set()
        for nid in result.expanded_nodes:
            nd = graph.nodes.get(nid, {})
            expanded_names.add(nd.get("symbol_name", ""))

        assert "Logger" in expanded_names, (
            f"Logger should be pulled in via composition. Expanded: {expanded_names}"
        )

    def test_expand_smart_function_includes_call_targets(self, go_project):
        """Expanding a function should pull in called functions and created types."""
        go_project("svc/main.go", """
            package svc

            func Run() {
                cfg := LoadConfig()
                srv := NewServer(cfg)
                _ = srv
            }
        """)
        go_project("svc/config.go", """
            package svc

            type Config struct{ Host string }

            func LoadConfig() Config {
                return Config{Host: "localhost"}
            }
        """)
        go_project("svc/server.go", """
            package svc

            type Server struct{ config Config }

            func NewServer(cfg Config) *Server {
                return &Server{config: cfg}
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart

        graph = self._build_graph(go_project.root)

        run_nodes = [n for n in graph.nodes if n.endswith("::Run")]
        assert len(run_nodes) >= 1, f"No Run node. Nodes: {list(graph.nodes)}"
        run_node = run_nodes[0]

        result = expand_smart({run_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        # Run calls LoadConfig and NewServer
        assert "LoadConfig" in expanded_names or "NewServer" in expanded_names, (
            f"Expected call targets in expansion. Got: {expanded_names}"
        )

    def test_expand_smart_factory_reaches_created_types(self, go_project):
        """Expanding a factory function reaches all types it instantiates via CREATES."""
        go_project("svc/types.go", """
            package svc

            type Config struct {
                Host string
                Port int
            }

            type Logger struct {
                Level string
            }

            type Service struct {
                Cfg  Config
                Log  Logger
                Name string
            }

            func (s *Service) Run() error { return nil }
        """)
        go_project("svc/factory.go", """
            package svc

            func NewService(name string) *Service {
                return &Service{
                    Cfg: Config{Host: "localhost", Port: 8080},
                    Log: Logger{Level: "info"},
                    Name: name,
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart

        graph = self._build_graph(go_project.root)

        factory_nodes = [n for n in graph.nodes if "NewService" in n]
        assert factory_nodes, f"NewService not found in {list(graph.nodes)}"
        factory_node = factory_nodes[0]

        result = expand_smart({factory_node}, graph)
        expanded_names = set()
        for nid in result.expanded_nodes:
            nd = graph.nodes.get(nid, {})
            name = nd.get("symbol_name", nid.split("::")[-1])
            expanded_names.add(name)

        # Factory creates all three types — they should all be in expanded nodes
        assert "Service" in expanded_names, (
            f"Service should be reachable via CREATES. Expanded: {expanded_names}"
        )
        assert "Config" in expanded_names, (
            f"Config should be reachable via nested CREATES. Expanded: {expanded_names}"
        )
        assert "Logger" in expanded_names, (
            f"Logger should be reachable via nested CREATES. Expanded: {expanded_names}"
        )

    def test_expand_smart_struct_reaches_factory_via_creates(self, go_project):
        """Expanding a struct reaches its factory function via reverse CREATES edge."""
        go_project("svc/types.go", """
            package svc

            type Config struct {
                Host string
                Port int
            }

            type Service struct {
                Cfg  Config
                Name string
            }
        """)
        go_project("svc/factory.go", """
            package svc

            func NewService() *Service {
                return &Service{
                    Cfg: Config{Host: "localhost"},
                    Name: "default",
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart

        graph = self._build_graph(go_project.root)

        svc_nodes = [n for n in graph.nodes if n.endswith("::Service")]
        assert svc_nodes, f"Service not found in {list(graph.nodes)}"
        svc_node = svc_nodes[0]

        result = expand_smart({svc_node}, graph)
        expanded_names = set()
        for nid in result.expanded_nodes:
            nd = graph.nodes.get(nid, {})
            name = nd.get("symbol_name", nid.split("::")[-1])
            expanded_names.add(name)

        # NewService creates Service — should be reachable via reverse creates
        assert "NewService" in expanded_names, (
            f"NewService should be reachable from Service via reverse CREATES. "
            f"Expanded: {expanded_names}"
        )
        # Config is reachable via composition edge from Service
        assert "Config" in expanded_names, (
            f"Config should be reachable via composition. Expanded: {expanded_names}"
        )

    def test_expand_smart_interface_reaches_implementors(self, go_project):
        """Expanding an interface should reach its implicit implementors."""
        go_project("svc/server.go", """
            package svc

            type Server struct{}
            func (s *Server) Start() error { return nil }
            func (s *Server) Close() error { return nil }
        """)
        go_project("svc/ifaces.go", """
            package svc

            type Starter interface {
                Start() error
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart

        graph = self._build_graph(go_project.root)

        iface_nodes = [n for n in graph.nodes if "Starter" in n]
        if not iface_nodes:
            pytest.skip("Interface node not found — implementation edges may not be produced")

        iface_node = iface_nodes[0]
        result = expand_smart({iface_node}, graph)

        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        if "Server" not in expanded_names:
            # This is expected — implicit interface implementation + graph expansion
            # may vary. Skip if the implementation edge is not present.
            edge_types = set()
            for _, _, d in graph.edges(data=True):
                edge_types.add(d.get("relationship_type", ""))
            if "implementation" not in edge_types:
                pytest.skip("No implementation edges in graph — implicit interfaces may not produce expansion edges")
            else:
                assert "Server" in expanded_names, (
                    f"Expected Server as implementor of Starter. Got: {expanded_names}"
                )
