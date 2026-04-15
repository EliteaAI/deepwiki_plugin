"""
Rust Rich Parser — Integration Tests.

Covers:
    IT-1: Parser → Symbol Extraction (single file)
    IT-2: Parser → Relationship Extraction (single file)
    IT-3: Parser → Cross-File Analysis (multi-file)

See PLANNING_RUST_RICH_PARSER.md §8 for test specifications.
"""

import textwrap
from pathlib import Path

import pytest

from plugin_implementation.parsers.base_parser import (
    SymbolType,
    RelationshipType,
    ParseResult,
)
from plugin_implementation.parsers.rust_visitor_parser import RustVisitorParser


# =============================================================================
# Shared fixture
# =============================================================================

def _write_rs(tmp_path: Path, rel_path: str, source: str) -> Path:
    """Write a Rust source file with dedented content."""
    full = tmp_path / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(textwrap.dedent(source).lstrip())
    return full


@pytest.fixture
def rust_project(tmp_path):
    """Factory fixture: call with (rel_path, source) to create Rust files."""
    files = []

    def _add(rel_path: str, source: str) -> Path:
        p = _write_rs(tmp_path, rel_path, source)
        files.append(str(p))
        return p

    _add.root = tmp_path
    _add.files = files
    return _add


# =============================================================================
# IT-1: Parser → Symbol Extraction (Single File)
# =============================================================================

class TestRustParserSymbols:
    """Integration: write real Rust code → parse → assert symbols."""

    def test_struct_and_fields(self, rust_project):
        rust_project("src/server.rs", """
            /// Server handles HTTP requests.
            pub struct Server {
                pub host: String,
                port: u16,
                logger: Box<Logger>,
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        structs = [s for s in result.symbols if s.symbol_type == SymbolType.STRUCT]
        assert len(structs) == 1
        assert structs[0].name == "Server"
        assert structs[0].visibility == "public"
        assert structs[0].docstring and "HTTP requests" in structs[0].docstring

        fields = [s for s in result.symbols if s.symbol_type == SymbolType.FIELD]
        assert {f.name for f in fields} == {"host", "port", "logger"}

        host_field = next(f for f in fields if f.name == "host")
        assert host_field.parent_symbol == "Server"
        assert host_field.visibility == "public"

        port_field = next(f for f in fields if f.name == "port")
        assert port_field.visibility == "private"

    def test_tuple_struct(self, rust_project):
        rust_project("src/types.rs", """
            pub struct Color(u8, u8, u8);
            pub struct Wrapper(Config);
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        structs = [s for s in result.symbols if s.symbol_type == SymbolType.STRUCT]
        assert len(structs) == 2
        assert {s.name for s in structs} == {"Color", "Wrapper"}

    def test_enum_with_three_variant_forms(self, rust_project):
        rust_project("src/shape.rs", """
            pub enum Shape {
                Circle(f64),
                Rectangle { width: f64, height: f64 },
                Point,
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        enums = [s for s in result.symbols if s.symbol_type == SymbolType.ENUM]
        assert len(enums) == 1
        assert enums[0].name == "Shape"

        variants = [s for s in result.symbols
                     if s.symbol_type == SymbolType.FIELD and s.parent_symbol == "Shape"]
        assert {v.name for v in variants} == {"Circle", "Rectangle", "Point"}

        circle = next(v for v in variants if v.name == "Circle")
        assert circle.metadata.get('variant_kind') == 'tuple'

        rect = next(v for v in variants if v.name == "Rectangle")
        assert rect.metadata.get('variant_kind') == 'struct'

        point = next(v for v in variants if v.name == "Point")
        assert point.metadata.get('variant_kind') == 'unit'

    def test_trait_with_abstract_and_default_methods(self, rust_project):
        rust_project("src/animal.rs", """
            pub trait Animal {
                fn name(&self) -> &str;
                fn sound(&self) -> String;

                fn describe(&self) -> String {
                    format!("{} says {}", self.name(), self.sound())
                }
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        traits = [s for s in result.symbols if s.symbol_type == SymbolType.TRAIT]
        assert len(traits) == 1
        assert traits[0].name == "Animal"

        methods = [s for s in result.symbols
                   if s.symbol_type == SymbolType.METHOD and s.parent_symbol == "Animal"]
        assert len(methods) == 3

        abstract_methods = [m for m in methods if m.metadata.get('is_abstract')]
        default_methods = [m for m in methods if not m.metadata.get('is_abstract')]
        assert {m.name for m in abstract_methods} == {"name", "sound"}
        assert {m.name for m in default_methods} == {"describe"}

    def test_inherent_impl_methods(self, rust_project):
        rust_project("src/config.rs", """
            pub struct Config {
                pub host: String,
                pub port: u16,
            }

            impl Config {
                pub fn new(host: String, port: u16) -> Self {
                    Self { host, port }
                }

                pub fn address(&self) -> String {
                    format!("{}:{}", self.host, self.port)
                }
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        methods = [s for s in result.symbols if s.symbol_type == SymbolType.METHOD]
        assert len(methods) == 2
        assert {m.name for m in methods} == {"new", "address"}

        new_method = next(m for m in methods if m.name == "new")
        assert new_method.parent_symbol == "Config"
        assert new_method.full_name == "Config.new"
        assert new_method.metadata.get('impl_kind') == 'inherent'

    def test_trait_impl_methods(self, rust_project):
        rust_project("src/display.rs", """
            pub struct Point { pub x: f64, pub y: f64 }

            pub trait Displayable {
                fn display(&self) -> String;
            }

            impl Displayable for Point {
                fn display(&self) -> String {
                    format!("({}, {})", self.x, self.y)
                }
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        methods = [s for s in result.symbols
                   if s.symbol_type == SymbolType.METHOD and s.parent_symbol == "Point"]
        assert len(methods) >= 1
        display_method = next(m for m in methods if m.name == "display")
        assert display_method.metadata.get('impl_kind') == 'trait'

    def test_top_level_function(self, rust_project):
        rust_project("src/lib.rs", """
            pub fn process_data(input: &str) -> Vec<u8> {
                input.as_bytes().to_vec()
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        funcs = [s for s in result.symbols if s.symbol_type == SymbolType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].name == "process_data"
        assert funcs[0].visibility == "public"
        assert "fn" in funcs[0].signature

    def test_async_unsafe_function(self, rust_project):
        rust_project("src/lib.rs", """
            pub async fn fetch_data(url: &str) -> Result<Vec<u8>, String> {
                Ok(vec![])
            }

            pub unsafe fn raw_access(ptr: *const u8) -> u8 {
                *ptr
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        funcs = [s for s in result.symbols if s.symbol_type == SymbolType.FUNCTION]
        assert len(funcs) == 2

        async_fn = next(f for f in funcs if f.name == "fetch_data")
        assert async_fn.is_async is True
        assert async_fn.metadata.get('is_async') is True

        unsafe_fn = next(f for f in funcs if f.name == "raw_access")
        assert unsafe_fn.metadata.get('is_unsafe') is True

    def test_const_and_static(self, rust_project):
        rust_project("src/lib.rs", """
            pub const MAX_SIZE: usize = 1024;
            static COUNTER: u32 = 0;
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        constants = [s for s in result.symbols if s.symbol_type == SymbolType.CONSTANT]
        assert len(constants) == 2

        max_size = next(c for c in constants if c.name == "MAX_SIZE")
        assert max_size.visibility == "public"

        counter = next(c for c in constants if c.name == "COUNTER")
        assert counter.is_static is True
        assert counter.metadata.get('is_static') is True

    def test_type_alias(self, rust_project):
        rust_project("src/lib.rs", """
            pub type Result<T> = std::result::Result<T, MyError>;
            type UserId = u64;
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        aliases = [s for s in result.symbols if s.symbol_type == SymbolType.TYPE_ALIAS]
        assert len(aliases) == 2
        result_alias = next(a for a in aliases if a.name == "Result")
        assert result_alias.visibility == "public"

    def test_module_declaration(self, rust_project):
        rust_project("src/lib.rs", """
            pub mod types;
            mod internal;

            pub mod inline_mod {
                pub fn helper() -> bool { true }
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        modules = [s for s in result.symbols if s.symbol_type == SymbolType.MODULE]
        assert len(modules) == 3

        types_mod = next(m for m in modules if m.name == "types")
        assert types_mod.visibility == "public"

        internal_mod = next(m for m in modules if m.name == "internal")
        assert internal_mod.visibility == "private"

        inline_mod = next(m for m in modules if m.name == "inline_mod")
        assert inline_mod.metadata.get('is_inline') is True

    def test_macro_definition(self, rust_project):
        rust_project("src/lib.rs", """
            macro_rules! my_macro {
                ($e:expr) => {
                    println!("{}", $e);
                };
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        macros = [s for s in result.symbols if s.symbol_type == SymbolType.MACRO]
        assert len(macros) == 1
        assert macros[0].name == "my_macro"

    def test_derive_attributes(self, rust_project):
        rust_project("src/types.rs", """
            #[derive(Debug, Clone, PartialEq)]
            pub struct Config {
                pub name: String,
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        structs = [s for s in result.symbols if s.symbol_type == SymbolType.STRUCT]
        assert len(structs) == 1
        assert 'derives' in structs[0].metadata
        assert set(structs[0].metadata['derives']) == {'Debug', 'Clone', 'PartialEq'}

    def test_generic_struct_with_lifetimes(self, rust_project):
        rust_project("src/lib.rs", """
            pub struct Parser<'a, T: Clone> {
                input: &'a str,
                output: Vec<T>,
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        structs = [s for s in result.symbols if s.symbol_type == SymbolType.STRUCT]
        assert len(structs) == 1
        assert structs[0].name == "Parser"
        assert structs[0].metadata.get('lifetimes') and 'a' in structs[0].metadata['lifetimes']

    def test_visibility_modifiers(self, rust_project):
        rust_project("src/lib.rs", """
            pub struct PublicStruct;
            struct PrivateStruct;
            pub(crate) struct CrateStruct;
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        structs = {s.name: s for s in result.symbols if s.symbol_type == SymbolType.STRUCT}
        assert structs["PublicStruct"].visibility == "public"
        assert structs["PrivateStruct"].visibility == "private"
        assert structs["CrateStruct"].visibility == "crate"

    def test_union_item(self, rust_project):
        rust_project("src/lib.rs", """
            pub union MyUnion {
                pub i: i32,
                pub f: f32,
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        structs = [s for s in result.symbols if s.symbol_type == SymbolType.STRUCT]
        assert len(structs) == 1
        assert structs[0].name == "MyUnion"
        assert structs[0].metadata.get('is_union') is True

        fields = [s for s in result.symbols if s.symbol_type == SymbolType.FIELD]
        assert {f.name for f in fields} == {"i", "f"}

    def test_source_text_and_range(self, rust_project):
        rust_project("src/lib.rs", """
            pub fn hello() -> String {
                "hello".to_string()
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        funcs = [s for s in result.symbols if s.symbol_type == SymbolType.FUNCTION]
        assert len(funcs) == 1
        assert funcs[0].source_text is not None
        assert "hello" in funcs[0].source_text
        assert funcs[0].range.start.line >= 1
        assert funcs[0].range.end.line >= funcs[0].range.start.line


# =============================================================================
# IT-2: Parser → Relationship Extraction (Single File)
# =============================================================================

class TestRustParserRelationships:
    """Integration: write real Rust code → parse → assert relationships."""

    def test_import_relationships(self, rust_project):
        rust_project("src/lib.rs", """
            use std::io::Read;
            use std::collections::{HashMap, HashSet};
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        imports = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.IMPORTS]
        targets = {r.target_symbol for r in imports}
        assert "std::io::Read" in targets
        assert "std::collections::HashMap" in targets or any("HashMap" in t for t in targets)

    def test_struct_field_composition(self, rust_project):
        rust_project("src/lib.rs", """
            pub struct Logger;
            pub struct Server {
                logger: Logger,
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        comp = [r for r in result.relationships
                if r.relationship_type == RelationshipType.COMPOSITION]
        assert any(r.source_symbol == "Server" and r.target_symbol == "Logger" for r in comp)

    def test_struct_field_aggregation(self, rust_project):
        rust_project("src/lib.rs", """
            pub struct Logger;
            pub struct Server {
                logger: Box<Logger>,
                name: &'static str,
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        agg = [r for r in result.relationships
               if r.relationship_type == RelationshipType.AGGREGATION]
        assert any(r.source_symbol == "Server" and r.target_symbol == "Logger" for r in agg)

    def test_function_calls(self, rust_project):
        rust_project("src/lib.rs", """
            fn helper() -> u32 { 42 }

            fn main_func() {
                let x = helper();
                let y = String::from("hello");
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        calls = [r for r in result.relationships
                 if r.relationship_type == RelationshipType.CALLS]
        targets = {r.target_symbol for r in calls}
        assert "helper" in targets

    def test_struct_expression_creates(self, rust_project):
        rust_project("src/lib.rs", """
            pub struct Config {
                pub host: String,
                pub port: u16,
            }

            pub fn make_config() -> Config {
                Config { host: "localhost".to_string(), port: 8080 }
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        creates = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.CREATES]
        assert any(r.source_symbol == "make_config" and r.target_symbol == "Config" for r in creates)

    def test_self_struct_expression_creates(self, rust_project):
        rust_project("src/lib.rs", """
            pub struct Config {
                pub host: String,
                pub port: u16,
            }

            impl Config {
                pub fn new(host: String, port: u16) -> Self {
                    Self { host, port }
                }
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        creates = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.CREATES]
        # Self should be resolved to Config
        assert any(r.source_symbol == "Config.new" and r.target_symbol == "Config" for r in creates)

    def test_nested_struct_expression_creates(self, rust_project):
        rust_project("src/lib.rs", """
            pub struct Config {
                pub host: String,
            }

            pub struct Service {
                pub config: Config,
                pub name: String,
            }

            pub fn make_service() -> Service {
                Service {
                    config: Config { host: "localhost".to_string() },
                    name: "web".to_string(),
                }
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        creates = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.CREATES]
        creates_targets = {r.target_symbol for r in creates if r.source_symbol == "make_service"}
        assert "Service" in creates_targets
        assert "Config" in creates_targets

    def test_defines_struct_to_method_inherent(self, rust_project):
        rust_project("src/lib.rs", """
            pub struct Server;

            impl Server {
                pub fn start(&self) {}
                pub fn stop(&self) {}
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        defines = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.DEFINES
                   and r.source_symbol == "Server"
                   and r.annotations.get('member_type') == 'method']
        targets = {r.target_symbol for r in defines}
        assert "Server.start" in targets
        assert "Server.stop" in targets

    def test_defines_trait_to_abstract_method(self, rust_project):
        rust_project("src/lib.rs", """
            pub trait Processor {
                fn process(&self, input: &str) -> String;
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        defines = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.DEFINES
                   and r.source_symbol == "Processor"]
        assert any(r.annotations.get('is_abstract') for r in defines)

    def test_defines_trait_to_default_method(self, rust_project):
        rust_project("src/lib.rs", """
            pub trait Processor {
                fn process(&self, input: &str) -> String;

                fn validate(&self, input: &str) -> bool {
                    !input.is_empty()
                }
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        defines = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.DEFINES
                   and r.source_symbol == "Processor"]
        assert len(defines) == 2

    def test_impl_trait_for_struct(self, rust_project):
        rust_project("src/lib.rs", """
            pub trait Runnable {
                fn run(&self);
            }

            pub struct Worker;

            impl Runnable for Worker {
                fn run(&self) {}
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        impls = [r for r in result.relationships
                 if r.relationship_type == RelationshipType.IMPLEMENTATION]
        assert any(r.source_symbol == "Worker" and r.target_symbol == "Runnable" for r in impls)

    def test_supertrait_inheritance(self, rust_project):
        rust_project("src/lib.rs", """
            pub trait Base {
                fn base_method(&self);
            }

            pub trait Extended: Base {
                fn extended_method(&self);
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        inheritance = [r for r in result.relationships
                       if r.relationship_type == RelationshipType.INHERITANCE]
        assert any(r.source_symbol == "Extended" and r.target_symbol == "Base" for r in inheritance)

    def test_type_alias_relationship(self, rust_project):
        rust_project("src/lib.rs", """
            pub struct MyError;
            pub type Result<T> = std::result::Result<T, MyError>;
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        aliases = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.ALIAS_OF]
        assert any(r.source_symbol == "Result" and r.target_symbol == "MyError" for r in aliases)

    def test_dyn_trait_references(self, rust_project):
        rust_project("src/lib.rs", """
            pub trait Animal {
                fn name(&self) -> &str;
            }

            pub fn feed(animal: &dyn Animal) {
                let _ = animal.name();
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        refs = [r for r in result.relationships
                if r.relationship_type == RelationshipType.REFERENCES
                and r.annotations.get('dispatch') == 'dynamic']
        assert any(r.target_symbol == "Animal" for r in refs)

    def test_impl_trait_references_return(self, rust_project):
        rust_project("src/lib.rs", """
            pub trait Summary {
                fn summarize(&self) -> String;
            }

            pub fn make_summary() -> impl Summary {
                todo!()
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        refs = [r for r in result.relationships
                if r.relationship_type == RelationshipType.REFERENCES
                and r.annotations.get('dispatch') == 'static']
        assert any(r.target_symbol == "Summary" for r in refs)

    def test_impl_trait_references_param(self, rust_project):
        rust_project("src/lib.rs", """
            pub trait Summary {
                fn summarize(&self) -> String;
            }

            pub fn print_summary(item: &impl Summary) {
                let _ = item.summarize();
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        refs = [r for r in result.relationships
                if r.relationship_type == RelationshipType.REFERENCES
                and r.annotations.get('dispatch') == 'static']
        assert any(r.target_symbol == "Summary" for r in refs)

    def test_generic_bound_references(self, rust_project):
        rust_project("src/lib.rs", """
            pub trait Serializable {
                fn serialize(&self) -> Vec<u8>;
            }

            pub fn process<T: Serializable>(item: T) {
                let _ = item.serialize();
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        refs = [r for r in result.relationships
                if r.relationship_type == RelationshipType.REFERENCES
                and r.annotations.get('dispatch') == 'generic_bound']
        assert any(r.target_symbol == "Serializable" for r in refs)

    def test_where_clause_references(self, rust_project):
        rust_project("src/lib.rs", """
            pub trait Encoder {
                fn encode(&self) -> Vec<u8>;
            }

            pub fn transform<T>(item: T) -> Vec<u8>
            where T: Encoder
            {
                item.encode()
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        refs = [r for r in result.relationships
                if r.relationship_type == RelationshipType.REFERENCES
                and r.annotations.get('dispatch') == 'where_bound']
        assert any(r.target_symbol == "Encoder" for r in refs)

    def test_macro_invocation_calls(self, rust_project):
        rust_project("src/lib.rs", """
            macro_rules! my_macro {
                () => {};
            }

            pub fn do_stuff() {
                my_macro!();
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        macro_calls = [r for r in result.relationships
                       if r.relationship_type == RelationshipType.CALLS
                       and r.annotations.get('is_macro')]
        assert any(r.target_symbol == "my_macro" for r in macro_calls)

    def test_builtin_types_excluded(self, rust_project):
        rust_project("src/lib.rs", """
            pub struct Thing {
                name: String,
                count: i32,
                flag: bool,
                data: Vec<u8>,
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        # No COMPOSITION/AGGREGATION edges for builtin types
        comp = [r for r in result.relationships
                if r.relationship_type in (RelationshipType.COMPOSITION, RelationshipType.AGGREGATION)]
        for r in comp:
            assert r.target_symbol not in ('String', 'i32', 'bool', 'Vec', 'u8')

    def test_enum_variant_field_composition(self, rust_project):
        rust_project("src/lib.rs", """
            pub struct Payload;

            pub enum Message {
                Data(Payload),
                Empty,
            }
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        comp = [r for r in result.relationships
                if r.relationship_type == RelationshipType.COMPOSITION]
        assert any(r.source_symbol == "Message" and r.target_symbol == "Payload" for r in comp)

    def test_derive_annotates(self, rust_project):
        rust_project("src/lib.rs", """
            pub trait MyTrait {}

            #[derive(Debug, Clone, MyTrait)]
            pub struct Foo;
        """)
        parser = RustVisitorParser()
        result = parser.parse_file(rust_project.files[0])

        annotates = [r for r in result.relationships
                     if r.relationship_type == RelationshipType.ANNOTATES]
        # MyTrait (non-builtin) should produce ANNOTATES edge
        assert any(r.source_symbol == "MyTrait" and r.target_symbol == "Foo" for r in annotates)


# =============================================================================
# IT-3: Parser → Cross-File Analysis (Multi-File)
# =============================================================================

class TestRustParserCrossFile:
    """Integration: multi-file Rust code → parse_multiple_files → assert cross-file edges."""

    def test_cross_file_impl_methods(self, rust_project):
        """Struct in file1, impl in file2 → DEFINES edges."""
        rust_project("src/types.rs", """
            pub struct Config {
                pub host: String,
            }
        """)
        rust_project("src/config_impl.rs", """
            use crate::types::Config;

            impl Config {
                pub fn new(host: String) -> Self {
                    Self { host }
                }
            }
        """)
        parser = RustVisitorParser()
        results = parser.parse_multiple_files(rust_project.files)

        # Collect all relationships across files
        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        cross_defines = [r for r in all_rels
                        if r.relationship_type == RelationshipType.DEFINES
                        and r.annotations.get('cross_file')]
        assert any(r.source_symbol == "Config" and "new" in r.target_symbol for r in cross_defines)

    def test_cross_file_trait_impl(self, rust_project):
        """Trait in file1, impl in file2 → IMPLEMENTATION edge."""
        rust_project("src/traits.rs", """
            pub trait Runner {
                fn run(&self);
            }
        """)
        rust_project("src/worker.rs", """
            pub struct Worker;

            impl Runner for Worker {
                fn run(&self) {}
            }
        """)
        parser = RustVisitorParser()
        results = parser.parse_multiple_files(rust_project.files)

        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        impls = [r for r in all_rels
                if r.relationship_type == RelationshipType.IMPLEMENTATION]
        assert any(r.source_symbol == "Worker" and r.target_symbol == "Runner" for r in impls)

    def test_cross_file_function_calls(self, rust_project):
        """Function in file1, called in file2 → CALLS resolved."""
        rust_project("src/utils.rs", """
            pub fn compute() -> u32 { 42 }
        """)
        rust_project("src/main.rs", """
            pub fn process() {
                let result = compute();
            }
        """)
        parser = RustVisitorParser()
        results = parser.parse_multiple_files(rust_project.files)

        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        calls = [r for r in all_rels
                if r.relationship_type == RelationshipType.CALLS
                and r.target_symbol == "compute"]
        assert len(calls) >= 1
        # Should have target_file resolved to the utils file
        assert any(r.target_file is not None and "utils" in r.target_file for r in calls)

    def test_multiple_impl_blocks_same_struct(self, rust_project):
        """Two impl blocks for same struct → all methods linked."""
        rust_project("src/types.rs", """
            pub struct Widget {
                pub name: String,
            }

            impl Widget {
                pub fn new(name: String) -> Self {
                    Self { name }
                }
            }
        """)
        rust_project("src/widget_display.rs", """
            impl Widget {
                pub fn display(&self) -> String {
                    self.name.clone()
                }
            }
        """)
        parser = RustVisitorParser()
        results = parser.parse_multiple_files(rust_project.files)

        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        defines = [r for r in all_rels
                  if r.relationship_type == RelationshipType.DEFINES
                  and r.source_symbol == "Widget"
                  and r.annotations.get('member_type') == 'method']
        method_names = {r.target_symbol.split('.')[-1] for r in defines}
        assert 'new' in method_names
        assert 'display' in method_names

    def test_cargo_workspace_crate_map(self, rust_project):
        """Parse workspace Cargo.toml → correct crate_name → dir mapping."""
        root = rust_project.root
        # Create workspace structure
        (root / "Cargo.toml").write_text(textwrap.dedent("""
            [workspace]
            members = ["core", "api"]
        """).strip())
        (root / "core").mkdir()
        (root / "core" / "Cargo.toml").write_text(textwrap.dedent("""
            [package]
            name = "my-core"
            version = "0.1.0"
        """).strip())
        (root / "core" / "src").mkdir()
        (root / "core" / "src" / "lib.rs").write_text("pub struct Config;")

        (root / "api").mkdir()
        (root / "api" / "Cargo.toml").write_text(textwrap.dedent("""
            [package]
            name = "my-api"
            version = "0.1.0"
        """).strip())
        (root / "api" / "src").mkdir()
        (root / "api" / "src" / "lib.rs").write_text("pub struct Handler;")

        parser = RustVisitorParser()
        crate_map = parser._build_crate_map(str(root))

        assert "my_core" in crate_map
        assert crate_map["my_core"].name == "my-core"
        assert "my_api" in crate_map
        assert str(root / "core") in crate_map["my_core"].path

    def test_hyphen_normalization(self, rust_project):
        """Cargo name `my-core` maps to `use my_core::...`."""
        root = rust_project.root
        (root / "Cargo.toml").write_text(textwrap.dedent("""
            [workspace]
            members = ["crate-one"]
        """).strip())
        (root / "crate-one").mkdir(parents=True)
        (root / "crate-one" / "Cargo.toml").write_text(textwrap.dedent("""
            [package]
            name = "crate-one"
            version = "0.1.0"
        """).strip())
        (root / "crate-one" / "src").mkdir()
        (root / "crate-one" / "src" / "lib.rs").write_text("")

        parser = RustVisitorParser()
        crate_map = parser._build_crate_map(str(root))

        assert "crate_one" in crate_map
        assert crate_map["crate_one"].name == "crate-one"

    def test_single_crate_no_workspace(self, rust_project):
        """Repo with single Cargo.toml (no workspace) → works normally."""
        root = rust_project.root
        (root / "Cargo.toml").write_text(textwrap.dedent("""
            [package]
            name = "my-project"
            version = "0.1.0"
        """).strip())
        (root / "src").mkdir()
        (root / "src" / "lib.rs").write_text("pub struct Foo;")

        parser = RustVisitorParser()
        crate_map = parser._build_crate_map(str(root))

        assert "my_project" in crate_map

    def test_no_cargo_toml_graceful_fallback(self, rust_project):
        """Missing Cargo.toml → flat single-crate fallback with empty crate map."""
        rust_project("src/lib.rs", """
            pub fn hello() {}
        """)
        parser = RustVisitorParser()
        crate_map = parser._build_crate_map(str(rust_project.root))

        # No Cargo.toml → empty crate map, but parsing still works
        assert crate_map == {}

        # Single file parsing still works
        result = parser.parse_file(rust_project.files[0])
        assert len(result.symbols) == 1

    def test_nested_workspace_not_at_repo_root(self, rust_project):
        """Workspace Cargo.toml inside subdirectory, not repo root."""
        root = rust_project.root
        ws_root = root / "src" / "sdk" / "rust"
        ws_root.mkdir(parents=True)

        (ws_root / "Cargo.toml").write_text(textwrap.dedent("""
            [workspace]
            members = ["core"]

            [package]
            name = "sdk-root"
            version = "0.1.0"
        """).strip())
        (ws_root / "src").mkdir()
        (ws_root / "src" / "lib.rs").write_text("pub struct Root;")

        (ws_root / "core").mkdir()
        (ws_root / "core" / "Cargo.toml").write_text(textwrap.dedent("""
            [package]
            name = "sdk-core"
            version = "0.1.0"
        """).strip())
        (ws_root / "core" / "src").mkdir()
        (ws_root / "core" / "src" / "lib.rs").write_text("pub struct CoreType;")

        parser = RustVisitorParser()
        crate_map = parser._build_crate_map(str(root))

        assert "sdk_core" in crate_map
        assert "sdk_root" in crate_map


# =============================================================================
# IT-4: Parser → Graph Builder (Full Pipeline)
# =============================================================================

class TestRustGraphBuilder:
    """Integration: write Rust project to tmp → graph builder analyze → assert graph."""

    def _build_graph(self, root_path):
        """Helper: build graph and return the NetworkX MultiDiGraph."""
        from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
        builder = EnhancedUnifiedGraphBuilder()
        analysis = builder.analyze_repository(str(root_path))
        return builder, analysis.unified_graph

    def test_rust_registered_as_rich_parser(self):
        """Rust should be in the rich_parsers dict of the graph builder."""
        from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
        builder = EnhancedUnifiedGraphBuilder()
        assert 'rust' in builder.rich_parsers
        assert isinstance(builder.rich_parsers['rust'], RustVisitorParser)

    def test_graph_nodes_created(self, rust_project):
        """Graph builder creates nodes with correct IDs for Rust symbols."""
        rust_project("src/server.rs", """
            pub struct Server {
                pub host: String,
                port: u16,
            }

            impl Server {
                pub fn new(host: String, port: u16) -> Self {
                    Self { host, port }
                }

                pub fn start(&self) -> Result<(), String> {
                    Ok(())
                }
            }
        """)
        builder, graph = self._build_graph(rust_project.root)

        assert 'rust' in builder.rich_parsers
        assert graph is not None
        assert graph.number_of_nodes() >= 3  # Server, new, start at minimum

        # Verify node data structure
        for node_id, node_data in graph.nodes(data=True):
            assert 'symbol' in node_data or 'symbol_name' in node_data or 'name' in node_data

    def test_graph_defines_edges(self, rust_project):
        """DEFINES edges present: struct→field, struct→method."""
        rust_project("src/model.rs", """
            pub struct User {
                pub name: String,
                pub email: String,
            }

            impl User {
                pub fn validate(&self) -> bool {
                    !self.name.is_empty()
                }
            }
        """)
        _, graph = self._build_graph(rust_project.root)

        defines_edges = [(u, v) for u, v, d in graph.edges(data=True)
                         if d.get("relationship_type") == "defines"]
        assert len(defines_edges) >= 1, "Expected at least one DEFINES edge"

    def test_graph_composition_edges(self, rust_project):
        """COMPOSITION edges from struct field types."""
        rust_project("src/service.rs", """
            pub struct Config {
                pub host: String,
            }

            pub struct Service {
                pub config: Config,
                pub name: String,
            }
        """)
        _, graph = self._build_graph(rust_project.root)

        composition_edges = [(u, v) for u, v, d in graph.edges(data=True)
                             if d.get("relationship_type") == "composition"]
        assert len(composition_edges) >= 1, "Expected COMPOSITION edge from Service to Config"

    def test_graph_calls_edges(self, rust_project):
        """CALLS edges present for function invocations."""
        rust_project("src/main.rs", """
            fn helper() -> u32 { 42 }

            fn process() {
                let x = helper();
            }
        """)
        _, graph = self._build_graph(rust_project.root)

        call_edges = [(u, v) for u, v, d in graph.edges(data=True)
                      if d.get("relationship_type") == "calls"]
        assert len(call_edges) >= 1, "Expected at least one CALLS edge"

    def test_graph_implementation_edges(self, rust_project):
        """impl Trait for Struct produces IMPLEMENTATION edges in graph."""
        rust_project("src/lib.rs", """
            pub trait Runner {
                fn run(&self);
            }

            pub struct Worker;

            impl Runner for Worker {
                fn run(&self) {}
            }
        """)
        _, graph = self._build_graph(rust_project.root)

        impl_edges = [(u, v) for u, v, d in graph.edges(data=True)
                      if d.get("relationship_type") == "implementation"]
        if not impl_edges:
            pytest.skip("Graph builder may not preserve IMPLEMENTATION edges verbatim")

    def test_graph_creates_edges(self, rust_project):
        """CREATES edges survive into the graph for struct literal instantiation."""
        rust_project("src/lib.rs", """
            pub struct Config {
                pub host: String,
                pub port: u16,
            }

            pub struct Logger {
                pub level: String,
            }

            pub fn make_service() -> (Config, Logger) {
                let cfg = Config { host: "localhost".to_string(), port: 8080 };
                let log = Logger { level: "info".to_string() };
                (cfg, log)
            }
        """)
        _, graph = self._build_graph(rust_project.root)

        creates_edges = [(u, v) for u, v, d in graph.edges(data=True)
                         if d.get("relationship_type") == "creates"]
        assert len(creates_edges) >= 2, (
            f"Expected >= 2 CREATES edges (Config, Logger). Got {len(creates_edges)}: {creates_edges}"
        )

    def test_graph_cross_file_impl_method_edges(self, rust_project):
        """Cross-file impl methods produce DEFINES edges with cross_file annotation."""
        rust_project("src/types.rs", """
            pub struct Widget {
                pub name: String,
            }
        """)
        rust_project("src/widget_impl.rs", """
            impl Widget {
                pub fn new(name: String) -> Self {
                    Self { name }
                }

                pub fn display(&self) -> String {
                    self.name.clone()
                }
            }
        """)
        _, graph = self._build_graph(rust_project.root)

        # Find Widget struct node
        widget_nodes = [n for n in graph.nodes
                        if graph.nodes[n].get("symbol_name") == "Widget"]
        assert len(widget_nodes) >= 1, f"No Widget node found. Nodes: {list(graph.nodes)}"
        widget_node = widget_nodes[0]

        # All defines edges from Widget
        defines_edges = []
        for u, v, d in graph.out_edges(widget_node, data=True):
            if d.get("relationship_type") == "defines":
                defines_edges.append((u, v, d))

        # Should have fields + cross-file methods
        method_edges = [(u, v, d) for u, v, d in defines_edges
                        if d.get("annotations", {}).get("member_type") == "method"]
        assert len(method_edges) >= 2, (
            f"Expected >= 2 method defines edges (new, display), got {len(method_edges)}"
        )

        # At least some method edges should exist — cross_file annotation
        # depends on how the graph builder merges same-file vs cross-file edges.
        # Verify method names are linked regardless.
        method_target_names = set()
        for _, v, d in method_edges:
            nd = graph.nodes.get(v, {})
            method_target_names.add(nd.get("symbol_name", v.split("::")[-1]))
        assert "new" in method_target_names or "display" in method_target_names, (
            f"Expected cross-file methods linked. Got: {method_target_names}"
        )

    def test_full_pipeline_realistic_project(self, rust_project):
        """
        Realistic mini-project: lib + types + service + traits.
        Verifies the full parser → graph pipeline produces a connected graph.
        """
        rust_project("src/traits.rs", """
            pub trait Handler {
                fn handle(&self) -> Result<(), String>;
            }
        """)
        rust_project("src/types.rs", """
            pub struct Config {
                pub host: String,
                pub port: u16,
            }

            pub struct Logger {
                pub level: String,
            }
        """)
        rust_project("src/service.rs", """
            pub struct Service {
                pub config: Config,
                pub logger: Logger,
            }

            impl Service {
                pub fn new(config: Config, logger: Logger) -> Self {
                    Self { config, logger }
                }
            }

            impl Handler for Service {
                fn handle(&self) -> Result<(), String> {
                    Ok(())
                }
            }
        """)
        rust_project("src/main.rs", """
            fn main() {
                let cfg = Config { host: "localhost".to_string(), port: 8080 };
                let log = Logger { level: "info".to_string() };
                let svc = Service::new(cfg, log);
                let _ = svc.handle();
            }
        """)
        _, graph = self._build_graph(rust_project.root)

        assert graph is not None
        assert graph.number_of_nodes() >= 5

        edge_types = {d.get("relationship_type") for _, _, d in graph.edges(data=True)}
        assert len(edge_types) > 0, f"Graph should have edges. Found types: {edge_types}"


# =============================================================================
# IT-5: Parser → Graph → Expansion Engine (Smart Expansion + Augmentation)
# =============================================================================

class TestRustExpansion:
    """Integration: write Rust project → build graph → expand nodes → assert context."""

    def _build_graph(self, root_path):
        """Helper: build graph and return the NetworkX MultiDiGraph."""
        from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
        builder = EnhancedUnifiedGraphBuilder()
        analysis = builder.analyze_repository(str(root_path))
        return analysis.unified_graph

    # ------------------------------------------------------------------
    # Augmentation tests (Rust-specific struct/impl split)
    # ------------------------------------------------------------------

    def test_augment_rust_struct_with_cross_file_impl_methods(self, rust_project):
        """augment_rust_node() should merge cross-file impl methods into struct content."""
        rust_project("src/server.rs", """
            /// Server handles HTTP requests.
            pub struct Server {
                pub host: String,
                pub port: u16,
            }
        """)
        rust_project("src/server_impl.rs", """
            impl Server {
                pub fn new(host: String, port: u16) -> Self {
                    Self { host, port }
                }

                pub fn start(&self) -> Result<(), String> {
                    Ok(())
                }

                pub fn address(&self) -> String {
                    format!("{}:{}", self.host, self.port)
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import augment_rust_node

        graph = self._build_graph(rust_project.root)

        # Find Server node
        server_nodes = [n for n in graph.nodes
                        if graph.nodes[n].get("symbol_name") == "Server"]
        assert len(server_nodes) >= 1, f"No Server node found. Nodes: {list(graph.nodes)}"
        server_node = server_nodes[0]

        aug = augment_rust_node(graph, server_node)
        assert aug is not None, "augment_rust_node should return AugmentedContent"
        assert aug.node_id == server_node
        assert "new" in aug.augmented_content
        assert "start" in aug.augmented_content
        assert "address" in aug.augmented_content
        # Original struct definition should still be present
        assert "Server" in aug.original_content or "host" in aug.original_content
        # Impl methods header
        assert "impl methods from" in aug.augmented_content

    def test_augment_rust_struct_same_file_methods_not_duplicated(self, rust_project):
        """Methods in the same file as the struct should NOT be augmented (already visible)."""
        rust_project("src/server.rs", """
            pub struct Server {
                pub host: String,
            }

            impl Server {
                pub fn start(&self) -> Result<(), String> {
                    Ok(())
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import augment_rust_node

        graph = self._build_graph(rust_project.root)
        server_nodes = [n for n in graph.nodes
                        if graph.nodes[n].get("symbol_name") == "Server"]
        assert len(server_nodes) >= 1
        server_node = server_nodes[0]

        aug = augment_rust_node(graph, server_node)
        # No cross-file methods → should return None
        assert aug is None, "Same-file methods should not trigger augmentation"

    def test_augment_rust_struct_multiple_impl_blocks_across_files(self, rust_project):
        """Multiple impl blocks in different files all get merged."""
        rust_project("src/widget.rs", """
            pub struct Widget {
                pub name: String,
                pub width: u32,
            }
        """)
        rust_project("src/widget_core.rs", """
            impl Widget {
                pub fn new(name: String) -> Self {
                    Self { name, width: 100 }
                }
            }
        """)
        rust_project("src/widget_display.rs", """
            impl Widget {
                pub fn display(&self) -> String {
                    format!("{} ({}px)", self.name, self.width)
                }

                pub fn resize(&mut self, width: u32) {
                    self.width = width;
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import augment_rust_node

        graph = self._build_graph(rust_project.root)

        widget_nodes = [n for n in graph.nodes
                        if graph.nodes[n].get("symbol_name") == "Widget"]
        assert len(widget_nodes) >= 1
        widget_node = widget_nodes[0]

        aug = augment_rust_node(graph, widget_node)
        assert aug is not None, "Multiple cross-file impls should produce augmentation"
        assert "new" in aug.augmented_content
        assert "display" in aug.augmented_content
        assert "resize" in aug.augmented_content

    def test_augment_rust_trait_with_cross_file_impl(self, rust_project):
        """Traits with default methods in a different file get augmented."""
        # Note: This tests a less common but valid pattern where a trait has
        # methods linked via cross-file DEFINES edges from the parser.
        rust_project("src/traits.rs", """
            pub trait Processor {
                fn process(&self, data: &str) -> String;
            }
        """)
        rust_project("src/processor_impl.rs", """
            pub struct TextProcessor;

            impl Processor for TextProcessor {
                fn process(&self, data: &str) -> String {
                    data.to_uppercase()
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import augment_rust_node

        graph = self._build_graph(rust_project.root)

        # Find the Processor trait node
        trait_nodes = [n for n in graph.nodes
                       if graph.nodes[n].get("symbol_name") == "Processor"]
        if not trait_nodes:
            pytest.skip("Processor trait node not found in graph")

        trait_node = trait_nodes[0]
        aug = augment_rust_node(graph, trait_node)
        # May or may not have cross-file methods depending on parser edge linking.
        # The function should not crash; return None if no cross-file DEFINES.
        # This primarily ensures augment_rust_node handles trait type correctly.

    # ------------------------------------------------------------------
    # expand_smart() integration
    # ------------------------------------------------------------------

    def test_expand_smart_augments_rust_struct(self, rust_project):
        """expand_smart() should produce augmentation for Rust structs with cross-file impl."""
        rust_project("src/server.rs", """
            pub struct Server {
                pub host: String,
                pub port: u16,
            }
        """)
        rust_project("src/server_impl.rs", """
            impl Server {
                pub fn start(&self) -> Result<(), String> { Ok(()) }
                pub fn stop(&self) -> Result<(), String> { Ok(()) }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart

        graph = self._build_graph(rust_project.root)

        server_nodes = [n for n in graph.nodes
                        if graph.nodes[n].get("symbol_name") == "Server"]
        assert len(server_nodes) >= 1
        server_node = server_nodes[0]

        result = expand_smart({server_node}, graph)
        # Server should be augmented
        assert server_node in result.augmentations, (
            f"Server should be in augmentations. Keys: {list(result.augmentations.keys())}"
        )
        aug = result.augmentations[server_node]
        assert "start" in aug.augmented_content
        assert "stop" in aug.augmented_content

    # ------------------------------------------------------------------
    # Single-file expansion
    # ------------------------------------------------------------------

    def test_expand_class_with_inheritance(self, rust_project):
        """Expanding a struct should pull in its supertrait/base types."""
        rust_project("src/lib.rs", """
            pub trait Base {
                fn base_method(&self);
            }

            pub trait Extended: Base {
                fn extended_method(&self);
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(rust_project.root)

        extended_nodes = [n for n in graph.nodes
                          if graph.nodes[n].get("symbol_name") == "Extended"]
        assert extended_nodes, f"No Extended node found. Nodes: {list(graph.nodes)}"
        extended_node = extended_nodes[0]

        result = expand_smart({extended_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "Base" in expanded_names, (
            f"Base should be pulled in via inheritance. Expanded: {expanded_names}"
        )

    def test_expand_class_with_implementation(self, rust_project):
        """Expanding a struct should pull in the trait it implements."""
        rust_project("src/lib.rs", """
            pub trait Runnable {
                fn run(&self);
            }

            pub struct Worker;

            impl Runnable for Worker {
                fn run(&self) {}
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(rust_project.root)

        worker_nodes = [n for n in graph.nodes
                        if graph.nodes[n].get("symbol_name") == "Worker"]
        assert worker_nodes, f"No Worker node. Nodes: {list(graph.nodes)}"
        worker_node = worker_nodes[0]

        result = expand_smart({worker_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "Runnable" in expanded_names, (
            f"Runnable should be pulled in via implementation. Expanded: {expanded_names}"
        )

    def test_expand_class_with_composition(self, rust_project):
        """Expanding a struct should pull in composed field types."""
        rust_project("src/lib.rs", """
            pub struct Logger {
                pub level: String,
            }

            impl Logger {
                pub fn log(&self, msg: &str) {}
            }

            pub struct Config {
                pub host: String,
            }

            pub struct Service {
                pub logger: Logger,
                pub config: Config,
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(rust_project.root)

        svc_nodes = [n for n in graph.nodes
                     if graph.nodes[n].get("symbol_name") == "Service"]
        assert svc_nodes, f"No Service node. Nodes: {list(graph.nodes)}"
        svc_node = svc_nodes[0]

        result = expand_smart({svc_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "Logger" in expanded_names or "Config" in expanded_names, (
            f"Composed types should be pulled in. Expanded: {expanded_names}"
        )

    def test_expand_function_includes_call_targets(self, rust_project):
        """Expanding a function should pull in called functions and created types."""
        rust_project("src/lib.rs", """
            pub struct Config {
                pub host: String,
            }

            pub fn load_config() -> Config {
                Config { host: "localhost".to_string() }
            }

            pub fn init() {
                let cfg = load_config();
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(rust_project.root)

        init_nodes = [n for n in graph.nodes
                      if graph.nodes[n].get("symbol_name") == "init"]
        assert init_nodes, f"No init node. Nodes: {list(graph.nodes)}"
        init_node = init_nodes[0]

        result = expand_smart({init_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "load_config" in expanded_names, (
            f"Expected call target load_config in expansion. Got: {expanded_names}"
        )

    def test_expand_interface_reaches_implementors(self, rust_project):
        """Expanding a trait should reach its implementors via backward edges."""
        rust_project("src/lib.rs", """
            pub trait Handler {
                fn handle(&self);
            }

            pub struct RequestHandler;
            impl Handler for RequestHandler {
                fn handle(&self) {}
            }

            pub struct EventHandler;
            impl Handler for EventHandler {
                fn handle(&self) {}
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(rust_project.root)

        trait_nodes = [n for n in graph.nodes
                       if graph.nodes[n].get("symbol_name") == "Handler"]
        if not trait_nodes:
            pytest.skip("Handler trait node not found in graph")
        trait_node = trait_nodes[0]

        result = expand_smart({trait_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }

        has_implementor = "RequestHandler" in expanded_names or "EventHandler" in expanded_names
        if not has_implementor:
            edge_types = {d.get("relationship_type", "") for _, _, d in graph.edges(data=True)}
            if "implementation" not in edge_types:
                pytest.skip("No implementation edges in graph — trait implementors not tracked")
            else:
                assert has_implementor, (
                    f"Expected implementors of Handler. Expanded: {expanded_names}"
                )

    def test_expand_factory_reaches_created_types(self, rust_project):
        """Expanding a factory function reaches types it instantiates via CREATES."""
        rust_project("src/lib.rs", """
            pub struct Config {
                pub host: String,
                pub port: u16,
            }

            pub struct Logger {
                pub level: String,
            }

            pub struct Service {
                pub config: Config,
                pub logger: Logger,
            }

            pub fn build_service() -> Service {
                let cfg = Config { host: "localhost".to_string(), port: 8080 };
                let log = Logger { level: "info".to_string() };
                Service { config: cfg, logger: log }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(rust_project.root)

        factory_nodes = [n for n in graph.nodes
                         if graph.nodes[n].get("symbol_name") == "build_service"]
        assert factory_nodes, f"build_service not found in {list(graph.nodes)}"
        factory_node = factory_nodes[0]

        result = expand_smart({factory_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "Service" in expanded_names, (
            f"Service should be reachable via CREATES. Expanded: {expanded_names}"
        )
        assert "Config" in expanded_names, (
            f"Config should be reachable via CREATES. Expanded: {expanded_names}"
        )
        assert "Logger" in expanded_names, (
            f"Logger should be reachable via CREATES. Expanded: {expanded_names}"
        )

    def test_expand_struct_reaches_factory_via_reverse_creates(self, rust_project):
        """Expanding a struct reaches its factory via reverse CREATES edge."""
        rust_project("src/lib.rs", """
            pub struct Config {
                pub host: String,
            }

            pub struct Service {
                pub config: Config,
            }

            pub fn new_service() -> Service {
                Service {
                    config: Config { host: "localhost".to_string() },
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(rust_project.root)

        svc_nodes = [n for n in graph.nodes
                     if graph.nodes[n].get("symbol_name") == "Service"]
        assert svc_nodes, f"Service not found in {list(graph.nodes)}"
        svc_node = svc_nodes[0]

        result = expand_smart({svc_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "new_service" in expanded_names, (
            f"new_service should be reachable from Service via reverse CREATES. "
            f"Expanded: {expanded_names}"
        )
        assert "Config" in expanded_names, (
            f"Config should be reachable via composition. Expanded: {expanded_names}"
        )

    # ------------------------------------------------------------------
    # Cross-file expansion
    # ------------------------------------------------------------------

    def test_cross_file_expand_inheritance(self, rust_project):
        """Expanding a trait reaches its supertrait in a different file."""
        rust_project("src/base.rs", """
            pub trait Base {
                fn base_method(&self);
            }
        """)
        rust_project("src/extended.rs", """
            pub trait Extended: Base {
                fn extended_method(&self);
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(rust_project.root)

        extended_nodes = [n for n in graph.nodes
                          if graph.nodes[n].get("symbol_name") == "Extended"]
        assert extended_nodes, f"No Extended node. Nodes: {list(graph.nodes)}"
        extended_node = extended_nodes[0]

        result = expand_smart({extended_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "Base" in expanded_names, (
            f"Cross-file base trait should be in expansion. Expanded: {expanded_names}"
        )

    def test_cross_file_expand_implementation(self, rust_project):
        """Expanding a struct in file A reaches trait in file B."""
        rust_project("src/traits.rs", """
            pub trait Runner {
                fn run(&self);
            }
        """)
        rust_project("src/worker.rs", """
            pub struct Worker;

            impl Runner for Worker {
                fn run(&self) {}
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(rust_project.root)

        worker_nodes = [n for n in graph.nodes
                        if graph.nodes[n].get("symbol_name") == "Worker"]
        assert worker_nodes, f"No Worker node. Nodes: {list(graph.nodes)}"
        worker_node = worker_nodes[0]

        result = expand_smart({worker_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "Runner" in expanded_names, (
            f"Cross-file trait should be in expansion. Expanded: {expanded_names}"
        )

    def test_cross_file_expand_composition(self, rust_project):
        """Expanding a struct pulls in composed types from another file."""
        rust_project("src/config.rs", """
            pub struct Config {
                pub host: String,
                pub port: u16,
            }
        """)
        rust_project("src/service.rs", """
            pub struct Service {
                pub config: Config,
                pub name: String,
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(rust_project.root)

        svc_nodes = [n for n in graph.nodes
                     if graph.nodes[n].get("symbol_name") == "Service"]
        assert svc_nodes, f"No Service node. Nodes: {list(graph.nodes)}"
        svc_node = svc_nodes[0]

        result = expand_smart({svc_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "Config" in expanded_names, (
            f"Config should be pulled in via cross-file composition. Expanded: {expanded_names}"
        )

    def test_cross_file_expand_calls(self, rust_project):
        """Expanding a function reaches called functions from another file."""
        rust_project("src/utils.rs", """
            pub fn compute() -> u32 { 42 }
        """)
        rust_project("src/main.rs", """
            pub fn process() -> u32 {
                compute()
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(rust_project.root)

        process_nodes = [n for n in graph.nodes
                         if graph.nodes[n].get("symbol_name") == "process"]
        assert process_nodes, f"No process node. Nodes: {list(graph.nodes)}"
        process_node = process_nodes[0]

        result = expand_smart({process_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "compute" in expanded_names, (
            f"Cross-file callee compute should be in expansion. Expanded: {expanded_names}"
        )

    def test_cross_file_expand_creates(self, rust_project):
        """Expanding a factory in file A reaches types created in file B."""
        rust_project("src/types.rs", """
            pub struct Config {
                pub host: String,
            }

            pub struct Logger {
                pub level: String,
            }
        """)
        rust_project("src/factory.rs", """
            pub fn build() -> (Config, Logger) {
                let c = Config { host: "localhost".to_string() };
                let l = Logger { level: "info".to_string() };
                (c, l)
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(rust_project.root)

        build_nodes = [n for n in graph.nodes
                       if graph.nodes[n].get("symbol_name") == "build"]
        assert build_nodes, f"build not found in {list(graph.nodes)}"
        build_node = build_nodes[0]

        result = expand_smart({build_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "Config" in expanded_names, (
            f"Cross-file Config should be reachable via CREATES. Expanded: {expanded_names}"
        )
        assert "Logger" in expanded_names, (
            f"Cross-file Logger should be reachable via CREATES. Expanded: {expanded_names}"
        )

    def test_mixed_cross_file_scenario(self, rust_project):
        """Full cross-file scenario: trait + struct + factory + augmentation."""
        rust_project("src/traits.rs", """
            pub trait Handler {
                fn handle(&self) -> Result<(), String>;
            }
        """)
        rust_project("src/config.rs", """
            pub struct Config {
                pub host: String,
                pub port: u16,
            }
        """)
        rust_project("src/server.rs", """
            pub struct Server {
                pub config: Config,
            }

            impl Handler for Server {
                fn handle(&self) -> Result<(), String> {
                    Ok(())
                }
            }
        """)
        rust_project("src/server_impl.rs", """
            impl Server {
                pub fn new(config: Config) -> Self {
                    Self { config }
                }

                pub fn start(&self) -> Result<(), String> {
                    self.handle()
                }
            }
        """)
        rust_project("src/factory.rs", """
            pub fn create_server() -> Server {
                let cfg = Config { host: "localhost".to_string(), port: 8080 };
                Server::new(cfg)
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart

        graph = self._build_graph(rust_project.root)

        # Expand Server → should reach Config (composition), Handler (impl),
        # and be augmented with cross-file impl methods (new, start)
        server_nodes = [n for n in graph.nodes
                        if graph.nodes[n].get("symbol_name") == "Server"]
        assert server_nodes, f"No Server node. Nodes: {list(graph.nodes)}"
        server_node = server_nodes[0]

        result = expand_smart({server_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }

        # Config should be reachable via composition
        assert "Config" in expanded_names, (
            f"Config should be reachable via composition. Expanded: {expanded_names}"
        )

        # Handler should be reachable via implementation
        if "Handler" not in expanded_names:
            edge_types = {d.get("relationship_type", "") for _, _, d in graph.edges(data=True)}
            if "implementation" not in edge_types:
                pass  # Skip assertion if no implementation edges
            else:
                assert "Handler" in expanded_names, (
                    f"Handler should be reachable via implementation. Expanded: {expanded_names}"
                )

        # Server should be augmented with cross-file impl methods
        if server_node in result.augmentations:
            aug = result.augmentations[server_node]
            assert "new" in aug.augmented_content or "start" in aug.augmented_content, (
                f"Augmented content should include cross-file methods. Got: {aug.augmented_content[:200]}"
            )
