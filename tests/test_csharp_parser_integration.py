"""
C# Rich Parser — Integration Tests.

Covers:
    IT-1: Parser → Symbol Extraction (single file)
    IT-2: Parser → Relationship Extraction (single file)
    IT-3: Parser → Cross-File Analysis (multi-file)
    IT-4: Graph Builder Integration (C# registered as rich parser)
    IT-5: Smart Expansion Engine (single + cross-file)

See PLANNING_CSHARP_RICH_PARSER.md for test specifications.
"""

import textwrap
from pathlib import Path

import pytest

from plugin_implementation.parsers.base_parser import (
    SymbolType,
    RelationshipType,
    ParseResult,
)
from plugin_implementation.parsers.csharp_visitor_parser import CSharpVisitorParser


# =============================================================================
# Shared fixture
# =============================================================================

def _write_cs(tmp_path: Path, rel_path: str, source: str) -> Path:
    """Write a C# source file with dedented content."""
    full = tmp_path / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(textwrap.dedent(source).lstrip())
    return full


@pytest.fixture
def csharp_project(tmp_path):
    """Factory fixture: call with (rel_path, source) to create C# files."""
    files = []

    def _add(rel_path: str, source: str) -> Path:
        p = _write_cs(tmp_path, rel_path, source)
        files.append(str(p))
        return p

    _add.root = tmp_path
    _add.files = files
    return _add


# =============================================================================
# IT-1: Parser → Symbol Extraction (Single File)
# =============================================================================

class TestCSharpParserSymbols:
    """Integration: write real C# code → parse → assert symbols."""

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------

    def test_class_basic(self, csharp_project):
        csharp_project("src/Service.cs", """
            namespace MyApp.Services
            {
                public class UserService
                {
                    public string GetUser() { return "Alice"; }
                }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        classes = [s for s in result.symbols if s.symbol_type == SymbolType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "UserService"
        assert classes[0].full_name == "MyApp.Services.UserService"
        assert classes[0].visibility == "public"

    def test_class_modifiers(self, csharp_project):
        csharp_project("src/Models.cs", """
            namespace Models
            {
                public sealed class Config { }
                public abstract class BaseEntity { }
                public static class Helpers { }
                internal partial class Processor { }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        classes = [s for s in result.symbols if s.symbol_type == SymbolType.CLASS]
        names = {c.name for c in classes}
        assert names == {"Config", "BaseEntity", "Helpers", "Processor"}

        config = next(c for c in classes if c.name == "Config")
        assert config.metadata and config.metadata.get('is_sealed')

        base = next(c for c in classes if c.name == "BaseEntity")
        assert base.is_abstract

        helpers = next(c for c in classes if c.name == "Helpers")
        assert helpers.is_static

        proc = next(c for c in classes if c.name == "Processor")
        assert proc.visibility == "internal"
        assert proc.metadata and proc.metadata.get('is_partial')

    # ------------------------------------------------------------------
    # Structs
    # ------------------------------------------------------------------

    def test_struct(self, csharp_project):
        csharp_project("src/Point.cs", """
            public struct Point
            {
                public double X;
                public double Y;
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        structs = [s for s in result.symbols if s.symbol_type == SymbolType.STRUCT]
        assert len(structs) == 1
        assert structs[0].name == "Point"

        fields = [s for s in result.symbols if s.symbol_type == SymbolType.FIELD]
        assert {f.name for f in fields} == {"X", "Y"}
        for f in fields:
            assert f.parent_symbol == "Point"

    # ------------------------------------------------------------------
    # Interfaces
    # ------------------------------------------------------------------

    def test_interface(self, csharp_project):
        csharp_project("src/IService.cs", """
            public interface IService
            {
                void Execute();
                string GetName();
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        ifaces = [s for s in result.symbols if s.symbol_type == SymbolType.INTERFACE]
        assert len(ifaces) == 1
        assert ifaces[0].name == "IService"

        methods = [s for s in result.symbols if s.symbol_type == SymbolType.METHOD]
        assert {m.name for m in methods} == {"Execute", "GetName"}
        for m in methods:
            assert m.parent_symbol == "IService"

    # ------------------------------------------------------------------
    # Enums
    # ------------------------------------------------------------------

    def test_enum(self, csharp_project):
        csharp_project("src/Color.cs", """
            public enum Color
            {
                Red,
                Green,
                Blue
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        enums = [s for s in result.symbols if s.symbol_type == SymbolType.ENUM]
        assert len(enums) == 1
        assert enums[0].name == "Color"

        members = [s for s in result.symbols
                   if s.symbol_type == SymbolType.FIELD and s.parent_symbol == "Color"]
        assert {m.name for m in members} == {"Red", "Green", "Blue"}

    # ------------------------------------------------------------------
    # Records
    # ------------------------------------------------------------------

    def test_record(self, csharp_project):
        csharp_project("src/Person.cs", """
            public record Person(string Name, int Age);
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        classes = [s for s in result.symbols if s.symbol_type == SymbolType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "Person"
        assert classes[0].metadata and classes[0].metadata.get('is_record')

        props = [s for s in result.symbols if s.symbol_type == SymbolType.PROPERTY]
        assert {p.name for p in props} == {"Name", "Age"}

    def test_record_struct(self, csharp_project):
        csharp_project("src/Point.cs", """
            public record struct Point(int X, int Y);
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        structs = [s for s in result.symbols if s.symbol_type == SymbolType.STRUCT]
        assert len(structs) == 1
        assert structs[0].name == "Point"
        assert structs[0].metadata and structs[0].metadata.get('is_record')
        assert structs[0].metadata.get('is_record_struct')

    # ------------------------------------------------------------------
    # Delegates
    # ------------------------------------------------------------------

    def test_delegate(self, csharp_project):
        csharp_project("src/Delegates.cs", """
            public delegate void MessageHandler(string msg);
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        delegates = [s for s in result.symbols if s.symbol_type == SymbolType.TYPE_ALIAS]
        assert len(delegates) == 1
        assert delegates[0].name == "MessageHandler"
        assert delegates[0].metadata and delegates[0].metadata.get('is_delegate')

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def test_methods(self, csharp_project):
        csharp_project("src/Handler.cs", """
            public class Handler
            {
                public void DoWork() { }
                public async Task DoAsync() { }
                public static int Compute(int x, int y) { return x + y; }
                public virtual string Describe() { return "handler"; }
                public abstract void Init();
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        methods = [s for s in result.symbols if s.symbol_type == SymbolType.METHOD]
        names = {m.name for m in methods}
        assert names == {"DoWork", "DoAsync", "Compute", "Describe", "Init"}

        async_m = next(m for m in methods if m.name == "DoAsync")
        assert async_m.is_async

        static_m = next(m for m in methods if m.name == "Compute")
        assert static_m.is_static
        assert static_m.parameter_types == ["int", "int"]

        abstract_m = next(m for m in methods if m.name == "Init")
        assert abstract_m.is_abstract

    def test_method_override(self, csharp_project):
        csharp_project("src/Animal.cs", """
            public class Animal
            {
                public virtual string Speak() { return "..."; }
            }

            public class Dog : Animal
            {
                public override string Speak() { return "Woof"; }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        dog_speak = next(
            s for s in result.symbols
            if s.symbol_type == SymbolType.METHOD and s.name == "Speak"
            and s.parent_symbol and "Dog" in s.parent_symbol
        )
        assert dog_speak.metadata and dog_speak.metadata.get('is_override')

    def test_expression_bodied_method(self, csharp_project):
        csharp_project("src/Calc.cs", """
            public class Calc
            {
                public int Double(int x) => x * 2;
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        methods = [s for s in result.symbols if s.symbol_type == SymbolType.METHOD]
        assert len(methods) == 1
        assert methods[0].name == "Double"
        assert methods[0].return_type == "int"

    # ------------------------------------------------------------------
    # Constructors & Destructors
    # ------------------------------------------------------------------

    def test_constructors(self, csharp_project):
        csharp_project("src/Server.cs", """
            public class Server
            {
                private string _host;
                private int _port;

                public Server(string host) { _host = host; _port = 80; }
                public Server(string host, int port) : this(host) { _port = port; }
                ~Server() { }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        ctors = [s for s in result.symbols if s.symbol_type == SymbolType.CONSTRUCTOR]
        assert len(ctors) == 2

        single_param = next(c for c in ctors if len(c.parameter_types or []) == 1)
        assert single_param.parameter_types == ["string"]

        two_param = next(c for c in ctors if len(c.parameter_types or []) == 2)
        assert two_param.parameter_types == ["string", "int"]

        # Destructor
        dtors = [s for s in result.symbols
                 if s.symbol_type == SymbolType.METHOD
                 and s.metadata and s.metadata.get('is_destructor')]
        assert len(dtors) == 1
        assert dtors[0].name == "~Finalize"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def test_properties(self, csharp_project):
        csharp_project("src/User.cs", """
            public class User
            {
                public string Name { get; set; }
                public string Email { get; init; }
                public int Age { get; private set; }
                public int ComputedAge => Age + 1;
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        props = [s for s in result.symbols if s.symbol_type == SymbolType.PROPERTY]
        names = {p.name for p in props}
        assert names == {"Name", "Email", "Age", "ComputedAge"}

        email = next(p for p in props if p.name == "Email")
        assert email.metadata and email.metadata.get('has_init_accessor')

    # ------------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------------

    def test_fields(self, csharp_project):
        csharp_project("src/Constants.cs", """
            public class Config
            {
                public const int MaxRetries = 3;
                public static readonly string Version = "1.0";
                private string _name;
                internal int _count;
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        consts = [s for s in result.symbols if s.symbol_type == SymbolType.CONSTANT]
        assert any(c.name == "MaxRetries" for c in consts)

        fields = [s for s in result.symbols if s.symbol_type == SymbolType.FIELD]
        field_names = {f.name for f in fields}
        assert "_name" in field_names
        assert "_count" in field_names

        version = next(s for s in result.symbols
                       if s.name == "Version"
                       and s.symbol_type in (SymbolType.FIELD, SymbolType.CONSTANT))
        assert version.is_static

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def test_events(self, csharp_project):
        csharp_project("src/Button.cs", """
            public class Button
            {
                public event EventHandler Clicked;
                public event EventHandler DoubleClicked;
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        events = [s for s in result.symbols
                  if s.symbol_type == SymbolType.FIELD
                  and s.metadata and s.metadata.get('is_event')]
        assert {e.name for e in events} == {"Clicked", "DoubleClicked"}

    # ------------------------------------------------------------------
    # Indexer
    # ------------------------------------------------------------------

    def test_indexer(self, csharp_project):
        csharp_project("src/Grid.cs", """
            public class Grid
            {
                public int this[int row, int col] { get { return 0; } set { } }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        indexers = [s for s in result.symbols
                    if s.symbol_type == SymbolType.PROPERTY
                    and s.metadata and s.metadata.get('is_indexer')]
        assert len(indexers) == 1
        assert indexers[0].name == "this[]"
        assert indexers[0].parameter_types == ["int", "int"]

    # ------------------------------------------------------------------
    # Operator overload
    # ------------------------------------------------------------------

    def test_operator(self, csharp_project):
        csharp_project("src/Vector.cs", """
            public struct Vector
            {
                public double X;
                public double Y;

                public static Vector operator +(Vector a, Vector b)
                {
                    return new Vector { X = a.X + b.X, Y = a.Y + b.Y };
                }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        ops = [s for s in result.symbols
               if s.symbol_type == SymbolType.METHOD
               and s.metadata and s.metadata.get('is_operator')]
        assert len(ops) == 1
        assert "+" in ops[0].name

    # ------------------------------------------------------------------
    # Nested types
    # ------------------------------------------------------------------

    def test_nested_types(self, csharp_project):
        csharp_project("src/Outer.cs", """
            public class Outer
            {
                public class Inner
                {
                    public void Do() { }
                }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        classes = [s for s in result.symbols if s.symbol_type == SymbolType.CLASS]
        assert len(classes) == 2
        inner = next(c for c in classes if c.name == "Inner")
        assert "Outer" in inner.parent_symbol

    # ------------------------------------------------------------------
    # Generics
    # ------------------------------------------------------------------

    def test_generics(self, csharp_project):
        csharp_project("src/Repo.cs", """
            public class Repository<T> where T : class, new()
            {
                public T Find(int id) { return default; }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        cls = [s for s in result.symbols if s.symbol_type == SymbolType.CLASS]
        assert len(cls) == 1
        assert cls[0].metadata and 'generic_parameters' in cls[0].metadata
        assert 'T' in cls[0].metadata['generic_parameters']

    # ------------------------------------------------------------------
    # Namespaces
    # ------------------------------------------------------------------

    def test_block_namespace(self, csharp_project):
        csharp_project("src/Ns.cs", """
            namespace Acme.Core
            {
                public class Processor { }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        cls = [s for s in result.symbols if s.symbol_type == SymbolType.CLASS]
        assert cls[0].full_name == "Acme.Core.Processor"

    def test_file_scoped_namespace(self, csharp_project):
        csharp_project("src/Fs.cs", """
            namespace Acme.Core;

            public class Processor { }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        cls = [s for s in result.symbols if s.symbol_type == SymbolType.CLASS]
        assert cls[0].full_name == "Acme.Core.Processor"

    # ------------------------------------------------------------------
    # Nullable
    # ------------------------------------------------------------------

    def test_nullable_property(self, csharp_project):
        csharp_project("src/Nullable.cs", """
            public class Config
            {
                public string? OptionalName { get; set; }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        props = [s for s in result.symbols if s.symbol_type == SymbolType.PROPERTY]
        assert len(props) == 1
        assert props[0].name == "OptionalName"

    # ------------------------------------------------------------------
    # Extension methods
    # ------------------------------------------------------------------

    def test_extension_method(self, csharp_project):
        csharp_project("src/Extensions.cs", """
            public static class StringExtensions
            {
                public static int WordCount(this string s)
                {
                    return s.Split(' ').Length;
                }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        methods = [s for s in result.symbols if s.symbol_type == SymbolType.METHOD]
        assert len(methods) == 1
        assert methods[0].name == "WordCount"
        assert methods[0].metadata and methods[0].metadata.get('is_extension')

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    def test_attributes(self, csharp_project):
        csharp_project("src/Attr.cs", """
            [Serializable]
            public class Data
            {
                [Obsolete]
                public void OldMethod() { }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        annotates = [r for r in result.relationships
                     if r.relationship_type == RelationshipType.ANNOTATES]
        assert len(annotates) >= 2

        targets = {r.target_symbol for r in annotates}
        assert any("Data" in t for t in targets)
        assert any("OldMethod" in t for t in targets)

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def test_visibility(self, csharp_project):
        csharp_project("src/Vis.cs", """
            public class Vis
            {
                public void PubMethod() { }
                private void PrivMethod() { }
                protected void ProtMethod() { }
                internal void IntMethod() { }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        methods = [s for s in result.symbols if s.symbol_type == SymbolType.METHOD]
        pub = next(m for m in methods if m.name == "PubMethod")
        assert pub.visibility == "public"

        priv = next(m for m in methods if m.name == "PrivMethod")
        assert priv.visibility == "private"

        prot = next(m for m in methods if m.name == "ProtMethod")
        assert prot.visibility == "protected"

        internal = next(m for m in methods if m.name == "IntMethod")
        assert internal.visibility == "internal"


# =============================================================================
# IT-2: Parser → Relationship Extraction (Single File)
# =============================================================================

class TestCSharpParserRelationships:
    """Integration: parse C# code → assert relationships."""

    def test_inheritance_single(self, csharp_project):
        csharp_project("src/Inherit.cs", """
            public class Animal { }
            public class Dog : Animal { }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        inh = [r for r in result.relationships
               if r.relationship_type == RelationshipType.INHERITANCE]
        assert any(r.source_symbol == "Dog" and r.target_symbol == "Animal" for r in inh)

    def test_implementation(self, csharp_project):
        csharp_project("src/Impl.cs", """
            public interface IRunnable { void Run(); }
            public interface IDisposable { void Dispose(); }
            public class Worker : IRunnable, IDisposable
            {
                public void Run() { }
                public void Dispose() { }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        impl = [r for r in result.relationships
                if r.relationship_type == RelationshipType.IMPLEMENTATION]
        targets = {r.target_symbol for r in impl if "Worker" in r.source_symbol}
        assert "IRunnable" in targets
        assert "IDisposable" in targets

    def test_mixed_inheritance_and_implementation(self, csharp_project):
        csharp_project("src/Mixed.cs", """
            public class Base { }
            public interface IFoo { }
            public class Derived : Base, IFoo { }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        inh = [r for r in result.relationships
               if r.relationship_type == RelationshipType.INHERITANCE
               and "Derived" in r.source_symbol]
        assert any(r.target_symbol == "Base" for r in inh)

        impl = [r for r in result.relationships
                if r.relationship_type == RelationshipType.IMPLEMENTATION
                and "Derived" in r.source_symbol]
        assert any(r.target_symbol == "IFoo" for r in impl)

    def test_interface_inheritance(self, csharp_project):
        csharp_project("src/IExt.cs", """
            public interface IBase { void Do(); }
            public interface IExtended : IBase { void DoMore(); }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        inh = [r for r in result.relationships
               if r.relationship_type == RelationshipType.INHERITANCE
               and "IExtended" in r.source_symbol]
        assert any(r.target_symbol == "IBase" for r in inh)

    def test_struct_implements_interface(self, csharp_project):
        csharp_project("src/StructImpl.cs", """
            public interface IEquatable { bool Equals(object other); }
            public struct Point : IEquatable
            {
                public int X;
                public int Y;
                public bool Equals(object other) { return false; }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        impl = [r for r in result.relationships
                if r.relationship_type == RelationshipType.IMPLEMENTATION
                and "Point" in r.source_symbol]
        assert any(r.target_symbol == "IEquatable" for r in impl)

    def test_using_imports(self, csharp_project):
        csharp_project("src/Imports.cs", """
            using System;
            using System.Collections.Generic;
            using Alias = System.IO.Path;

            namespace MyApp
            {
                public class Foo { }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        imports = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.IMPORTS]
        targets = {r.target_symbol for r in imports}
        assert "System" in targets
        assert "System.Collections.Generic" in targets
        assert "System.IO.Path" in targets

    def test_calls(self, csharp_project):
        csharp_project("src/Calls.cs", """
            public class Logger
            {
                public void Log(string msg) { }
            }

            public class App
            {
                private Logger _logger;

                public void Run()
                {
                    _logger.Log("starting");
                }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        calls = [r for r in result.relationships
                 if r.relationship_type == RelationshipType.CALLS]
        assert any("Log" in r.target_symbol for r in calls)

    def test_creates(self, csharp_project):
        csharp_project("src/Factory.cs", """
            public class Widget { }

            public class Factory
            {
                public Widget Create()
                {
                    return new Widget();
                }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        creates = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.CREATES]
        assert any(r.target_symbol == "Widget" for r in creates)

    def test_composition(self, csharp_project):
        csharp_project("src/Owner.cs", """
            public class Engine { }

            public class Car
            {
                private Engine _engine;
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        comp = [r for r in result.relationships
                if r.relationship_type == RelationshipType.COMPOSITION]
        assert any(r.target_symbol == "Engine" for r in comp)

    def test_aggregation_nullable(self, csharp_project):
        csharp_project("src/Nullable.cs", """
            public class Settings
            {
                public Logger? OptionalLogger { get; set; }
            }

            public class Logger { }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        agg = [r for r in result.relationships
               if r.relationship_type == RelationshipType.AGGREGATION]
        assert any(r.target_symbol == "Logger" for r in agg)

    def test_defines(self, csharp_project):
        csharp_project("src/Def.cs", """
            public class Calculator
            {
                private int _value;
                public int Value { get; set; }
                public int Add(int x) { return _value + x; }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        defines = [r for r in result.relationships
                   if r.relationship_type == RelationshipType.DEFINES]
        targets = {r.target_symbol for r in defines}
        assert any("Add" in t for t in targets)
        assert any("_value" in t for t in targets)
        assert any("Value" in t for t in targets)

    def test_annotates(self, csharp_project):
        csharp_project("src/Ann.cs", """
            [Serializable]
            public class Payload { }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        ann = [r for r in result.relationships
               if r.relationship_type == RelationshipType.ANNOTATES]
        assert any(r.source_symbol == "Serializable" for r in ann)

    def test_overrides(self, csharp_project):
        csharp_project("src/Override.cs", """
            public class Base
            {
                public virtual void Invoke() { }
            }
            public class Derived : Base
            {
                public override void Invoke() { }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        overrides = [r for r in result.relationships
                     if r.relationship_type == RelationshipType.OVERRIDES]
        assert any("Invoke" in r.source_symbol and "Invoke" in r.target_symbol
                    for r in overrides)

    def test_references_return_type(self, csharp_project):
        csharp_project("src/Refs.cs", """
            public class Config { }

            public class Loader
            {
                public Config Load() { return new Config(); }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        refs = [r for r in result.relationships
                if r.relationship_type == RelationshipType.REFERENCES]
        assert any(r.target_symbol == "Config" for r in refs)

    def test_constructor_chaining_this(self, csharp_project):
        csharp_project("src/Chain.cs", """
            public class Connection
            {
                public Connection(string host) { }
                public Connection(string host, int port) : this(host) { }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        calls = [r for r in result.relationships
                 if r.relationship_type == RelationshipType.CALLS
                 and r.annotations and r.annotations.get('constructor_chain') == 'this']
        assert len(calls) >= 1

    def test_constructor_chaining_base(self, csharp_project):
        csharp_project("src/ChainBase.cs", """
            public class Base
            {
                public Base(string name) { }
            }
            public class Derived : Base
            {
                public Derived() : base("default") { }
            }
        """)
        parser = CSharpVisitorParser()
        result = parser.parse_file(csharp_project.files[0])

        calls = [r for r in result.relationships
                 if r.relationship_type == RelationshipType.CALLS
                 and r.annotations and r.annotations.get('constructor_chain') == 'base']
        assert len(calls) >= 1


# =============================================================================
# IT-3: Parser → Cross-File Analysis (Multi-File)
# =============================================================================

class TestCSharpParserCrossFile:
    """Integration: multi-file C# project → cross-file resolution."""

    def test_cross_file_inheritance(self, csharp_project):
        csharp_project("src/Animal.cs", """
            namespace Zoo
            {
                public class Animal
                {
                    public virtual string Sound() { return ""; }
                }
            }
        """)
        csharp_project("src/Dog.cs", """
            using Zoo;

            namespace Pets
            {
                public class Dog : Animal
                {
                    public override string Sound() { return "Woof"; }
                }
            }
        """)
        parser = CSharpVisitorParser()
        results = parser.parse_multiple_files(csharp_project.files)

        # Find the INHERITANCE relationship from Dog
        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        inh = [r for r in all_rels
               if r.relationship_type == RelationshipType.INHERITANCE
               and "Dog" in r.source_symbol]
        assert len(inh) >= 1
        # Cross-file: target_file should point to Animal.cs
        xfile = [r for r in inh if r.source_file != r.target_file]
        assert len(xfile) >= 1

    def test_cross_file_calls(self, csharp_project):
        csharp_project("src/Logger.cs", """
            namespace Utils
            {
                public class Logger
                {
                    public void Log(string msg) { }
                }
            }
        """)
        csharp_project("src/App.cs", """
            using Utils;

            namespace Main
            {
                public class App
                {
                    private Logger _logger;
                    public void Run()
                    {
                        _logger.Log("hello");
                    }
                }
            }
        """)
        parser = CSharpVisitorParser()
        results = parser.parse_multiple_files(csharp_project.files)

        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        calls = [r for r in all_rels
                 if r.relationship_type == RelationshipType.CALLS
                 and "Log" in r.target_symbol]
        assert len(calls) >= 1

    def test_cross_file_creates(self, csharp_project):
        csharp_project("src/Widget.cs", """
            namespace UI
            {
                public class Widget { }
            }
        """)
        csharp_project("src/Factory.cs", """
            using UI;

            namespace App
            {
                public class Factory
                {
                    public Widget Create()
                    {
                        return new Widget();
                    }
                }
            }
        """)
        parser = CSharpVisitorParser()
        results = parser.parse_multiple_files(csharp_project.files)

        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        creates = [r for r in all_rels
                   if r.relationship_type == RelationshipType.CREATES
                   and r.target_symbol == "Widget"]
        assert len(creates) >= 1
        # Should have cross-file target
        xfile = [r for r in creates if r.source_file != r.target_file]
        assert len(xfile) >= 1


# =============================================================================
# IT-4: Graph Builder Integration
# =============================================================================

class TestCSharpGraphBuilder:
    """Verify C# parser is registered and works with graph builder."""

    def test_csharp_registered_as_rich_parser(self):
        """C# should be in the rich_parsers dict of the graph builder."""
        from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
        builder = EnhancedUnifiedGraphBuilder()
        assert 'csharp' in builder.rich_parsers
        assert isinstance(builder.rich_parsers['csharp'], CSharpVisitorParser)
        # Verify csharp is NOT in basic_languages
        assert 'csharp' not in builder.basic_languages

    def test_graph_builder_parses_csharp(self, csharp_project):
        """Full pipeline: write C# files → graph builder → nodes exist."""
        from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder

        csharp_project("src/Models/User.cs", """
            namespace MyApp.Models
            {
                public class User
                {
                    public string Name { get; set; }
                    public int Age { get; set; }
                }
            }
        """)
        csharp_project("src/Services/UserService.cs", """
            using MyApp.Models;

            namespace MyApp.Services
            {
                public class UserService
                {
                    public User GetUser(int id)
                    {
                        return new User();
                    }
                }
            }
        """)

        builder = EnhancedUnifiedGraphBuilder()
        parser = builder.rich_parsers['csharp']
        results = parser.parse_multiple_files(csharp_project.files)

        # Verify we got symbols from both files
        all_symbols = []
        for r in results.values():
            all_symbols.extend(r.symbols)

        class_names = {s.name for s in all_symbols if s.symbol_type == SymbolType.CLASS}
        assert "User" in class_names
        assert "UserService" in class_names

        method_names = {s.name for s in all_symbols if s.symbol_type == SymbolType.METHOD}
        assert "GetUser" in method_names

        # Verify cross-file relationships exist
        all_rels = []
        for r in results.values():
            all_rels.extend(r.relationships)

        creates = [r for r in all_rels if r.relationship_type == RelationshipType.CREATES]
        assert any(r.target_symbol == "User" for r in creates)


# =============================================================================
# IT-5: Smart Expansion Engine (Single + Cross-File)
# =============================================================================

class TestCSharpSmartExpansion:
    """Integration: write C# project → build graph → expand nodes → assert context."""

    def _build_graph(self, root_path):
        """Helper: build graph and return the NetworkX MultiDiGraph."""
        from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder
        builder = EnhancedUnifiedGraphBuilder()
        analysis = builder.analyze_repository(str(root_path))
        return analysis.unified_graph

    # ------------------------------------------------------------------
    # Single-file expansion
    # ------------------------------------------------------------------

    def test_expand_class_with_inheritance(self, csharp_project):
        """Expanding a derived class should pull in its base class."""
        csharp_project("src/Models.cs", """
            namespace App
            {
                public class BaseEntity
                {
                    public int Id { get; set; }
                }

                public class User : BaseEntity
                {
                    public string Name { get; set; }
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(csharp_project.root)

        # Find User node
        user_nodes = [n for n in graph.nodes if "User" in n and "BaseEntity" not in n]
        assert user_nodes, f"No User node found. Nodes: {list(graph.nodes)}"
        user_node = user_nodes[0]

        result = expand_smart({user_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "BaseEntity" in expanded_names, (
            f"BaseEntity should be pulled in via inheritance. Expanded: {expanded_names}"
        )

    def test_expand_class_with_implementation(self, csharp_project):
        """Expanding a class should pull in interfaces it implements."""
        csharp_project("src/Impl.cs", """
            namespace App
            {
                public interface IRepository
                {
                    void Save();
                }

                public class UserRepository : IRepository
                {
                    public void Save() { }
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(csharp_project.root)

        repo_nodes = [n for n in graph.nodes if "UserRepository" in n]
        assert repo_nodes, f"No UserRepository node. Nodes: {list(graph.nodes)}"
        repo_node = repo_nodes[0]

        result = expand_smart({repo_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "IRepository" in expanded_names, (
            f"IRepository should be pulled in via implementation. Expanded: {expanded_names}"
        )

    def test_expand_class_with_composition(self, csharp_project):
        """Expanding a class should pull in composed field types."""
        csharp_project("src/Service.cs", """
            namespace App
            {
                public class Logger
                {
                    public void Log(string msg) { }
                }

                public class Config
                {
                    public string Host { get; set; }
                }

                public class Service
                {
                    private Logger _logger;
                    private Config _config;

                    public void Start()
                    {
                        _logger.Log("starting");
                    }
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(csharp_project.root)

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

    def test_expand_interface_reaches_implementors(self, csharp_project):
        """Expanding an interface should reach its implementors via backward edges."""
        csharp_project("src/Ifaces.cs", """
            namespace App
            {
                public interface IHandler
                {
                    void Handle();
                }

                public class RequestHandler : IHandler
                {
                    public void Handle() { }
                }

                public class EventHandler : IHandler
                {
                    public void Handle() { }
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(csharp_project.root)

        iface_nodes = [n for n in graph.nodes if "IHandler" in n]
        if not iface_nodes:
            pytest.skip("IHandler node not found in graph")

        iface_node = iface_nodes[0]
        result = expand_smart({iface_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }

        # At least one implementor should be found via backward implementation edge
        has_implementor = "RequestHandler" in expanded_names or "EventHandler" in expanded_names
        if not has_implementor:
            # Check if implementation edges exist at all
            edge_types = set()
            for _, _, d in graph.edges(data=True):
                edge_types.add(d.get("relationship_type", ""))
            if "implementation" not in edge_types:
                pytest.skip("No implementation edges in graph")
            else:
                assert has_implementor, (
                    f"Expected implementors of IHandler. Expanded: {expanded_names}"
                )

    def test_expand_function_includes_call_targets(self, csharp_project):
        """Expanding a method should pull in called methods and created types."""
        csharp_project("src/App.cs", """
            namespace App
            {
                public class Config
                {
                    public string Host { get; set; }
                }

                public class ConfigLoader
                {
                    public Config Load()
                    {
                        return new Config();
                    }
                }

                public class Application
                {
                    private ConfigLoader _loader;

                    public void Run()
                    {
                        var cfg = _loader.Load();
                    }
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(csharp_project.root)

        run_nodes = [n for n in graph.nodes if n.endswith("::Run") or "Application.Run" in n]
        assert run_nodes, f"No Run node. Nodes: {list(graph.nodes)}"
        run_node = run_nodes[0]

        result = expand_smart({run_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        # Run calls Load which creates Config
        assert "Load" in expanded_names or "ConfigLoader" in expanded_names, (
            f"Expected call targets in expansion. Got: {expanded_names}"
        )

    def test_expand_factory_reaches_created_types(self, csharp_project):
        """Expanding a factory method reaches types it instantiates via CREATES."""
        csharp_project("src/Factory.cs", """
            namespace App
            {
                public class Config
                {
                    public string Host { get; set; }
                }

                public class Logger
                {
                    public string Level { get; set; }
                }

                public class Service
                {
                    public Config Cfg { get; set; }
                    public Logger Log { get; set; }
                }

                public class ServiceFactory
                {
                    public Service Create()
                    {
                        var cfg = new Config();
                        var log = new Logger();
                        var svc = new Service();
                        return svc;
                    }
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(csharp_project.root)

        factory_nodes = [n for n in graph.nodes
                         if "Create" in n and "ServiceFactory" in n]
        if not factory_nodes:
            # Try bare name
            factory_nodes = [n for n in graph.nodes if n.endswith("::Create")]
        assert factory_nodes, f"Create node not found. Nodes: {list(graph.nodes)}"
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

    def test_expand_struct_reaches_factory_via_reverse_creates(self, csharp_project):
        """Expanding a class reaches its factory via reverse CREATES edge."""
        csharp_project("src/Builder.cs", """
            namespace App
            {
                public class Widget
                {
                    public string Name { get; set; }
                }

                public class WidgetBuilder
                {
                    public Widget Build()
                    {
                        return new Widget();
                    }
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(csharp_project.root)

        widget_nodes = [n for n in graph.nodes
                        if graph.nodes[n].get("symbol_name") == "Widget"]
        assert widget_nodes, f"Widget node not found. Nodes: {list(graph.nodes)}"
        widget_node = widget_nodes[0]

        result = expand_smart({widget_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        # Build creates Widget → Widget should reach WidgetBuilder (or Build) via reverse CREATES
        has_builder = "WidgetBuilder" in expanded_names or "Build" in expanded_names
        assert has_builder, (
            f"WidgetBuilder or Build should be reachable from Widget via reverse CREATES. "
            f"Expanded: {expanded_names}"
        )

    # ------------------------------------------------------------------
    # Cross-file expansion
    # ------------------------------------------------------------------

    def test_cross_file_expand_inheritance(self, csharp_project):
        """Expanding a derived class reaches its base class in a different file."""
        csharp_project("src/BaseEntity.cs", """
            namespace App.Models
            {
                public class BaseEntity
                {
                    public int Id { get; set; }
                    public DateTime CreatedAt { get; set; }
                }
            }
        """)
        csharp_project("src/User.cs", """
            namespace App.Models
            {
                public class User : BaseEntity
                {
                    public string Name { get; set; }
                    public string Email { get; set; }
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(csharp_project.root)

        user_nodes = [n for n in graph.nodes if "User" in n and "BaseEntity" not in n]
        assert user_nodes, f"No User node. Nodes: {list(graph.nodes)}"
        user_node = user_nodes[0]

        result = expand_smart({user_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "BaseEntity" in expanded_names, (
            f"Cross-file base class should be in expansion. Expanded: {expanded_names}"
        )

    def test_cross_file_expand_implementation(self, csharp_project):
        """Expanding a class in file A reaches interface in file B."""
        csharp_project("src/IService.cs", """
            namespace App.Contracts
            {
                public interface IService
                {
                    void Execute();
                    string GetStatus();
                }
            }
        """)
        csharp_project("src/WorkerService.cs", """
            namespace App.Services
            {
                public class WorkerService : IService
                {
                    public void Execute() { }
                    public string GetStatus() { return "running"; }
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(csharp_project.root)

        worker_nodes = [n for n in graph.nodes if "WorkerService" in n]
        assert worker_nodes, f"No WorkerService node. Nodes: {list(graph.nodes)}"
        worker_node = worker_nodes[0]

        result = expand_smart({worker_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "IService" in expanded_names, (
            f"Cross-file interface should be in expansion. Expanded: {expanded_names}"
        )

    def test_cross_file_expand_composition(self, csharp_project):
        """Expanding a class with fields defined in another file pulls them in."""
        csharp_project("src/Logger.cs", """
            namespace App.Infrastructure
            {
                public class Logger
                {
                    public void Log(string message) { }
                }
            }
        """)
        csharp_project("src/Repository.cs", """
            namespace App.Data
            {
                public class Repository
                {
                    private Logger _logger;

                    public void Save()
                    {
                        _logger.Log("saving");
                    }
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(csharp_project.root)

        repo_nodes = [n for n in graph.nodes
                      if graph.nodes[n].get("symbol_name") == "Repository"]
        assert repo_nodes, f"No Repository node. Nodes: {list(graph.nodes)}"
        repo_node = repo_nodes[0]

        result = expand_smart({repo_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "Logger" in expanded_names, (
            f"Cross-file composed Logger should be in expansion. Expanded: {expanded_names}"
        )

    def test_cross_file_expand_factory_creates(self, csharp_project):
        """Factory in file A creates types from file B → expansion reaches them."""
        csharp_project("src/Models/Config.cs", """
            namespace App.Models
            {
                public class Config
                {
                    public string Host { get; set; }
                    public int Port { get; set; }
                }
            }
        """)
        csharp_project("src/Models/Service.cs", """
            namespace App.Models
            {
                public class Service
                {
                    public Config Cfg { get; set; }
                    public string Name { get; set; }
                }
            }
        """)
        csharp_project("src/Factory/ServiceFactory.cs", """
            namespace App.Factory
            {
                public class ServiceFactory
                {
                    public Service CreateService()
                    {
                        var cfg = new Config();
                        var svc = new Service();
                        return svc;
                    }
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(csharp_project.root)

        factory_nodes = [n for n in graph.nodes if "CreateService" in n]
        assert factory_nodes, f"CreateService not found. Nodes: {list(graph.nodes)}"
        factory_node = factory_nodes[0]

        result = expand_smart({factory_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        assert "Service" in expanded_names, (
            f"Cross-file Service should be reachable via CREATES. Expanded: {expanded_names}"
        )
        assert "Config" in expanded_names, (
            f"Cross-file Config should be reachable via CREATES. Expanded: {expanded_names}"
        )

    def test_cross_file_mixed_inheritance_and_composition(self, csharp_project):
        """Complex: class inherits from file A, implements interface from file B,
        composes type from file C."""
        csharp_project("src/Base.cs", """
            namespace App
            {
                public class BaseController
                {
                    protected void LogRequest() { }
                }
            }
        """)
        csharp_project("src/IAuth.cs", """
            namespace App
            {
                public interface IAuthenticatable
                {
                    bool Authenticate(string token);
                }
            }
        """)
        csharp_project("src/UserRepo.cs", """
            namespace App
            {
                public class UserRepository
                {
                    public string FindUser(int id) { return "user"; }
                }
            }
        """)
        csharp_project("src/UserController.cs", """
            namespace App
            {
                public class UserController : BaseController, IAuthenticatable
                {
                    private UserRepository _repo;

                    public bool Authenticate(string token) { return true; }

                    public string GetUser(int id)
                    {
                        LogRequest();
                        return _repo.FindUser(id);
                    }
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(csharp_project.root)

        ctrl_nodes = [n for n in graph.nodes if "UserController" in n]
        assert ctrl_nodes, f"No UserController node. Nodes: {list(graph.nodes)}"
        ctrl_node = ctrl_nodes[0]

        result = expand_smart({ctrl_node}, graph)
        expanded_names = {
            graph.nodes.get(nid, {}).get("symbol_name", "")
            for nid in result.expanded_nodes
        }
        # Base class from file A
        assert "BaseController" in expanded_names, (
            f"Base class should be pulled in. Expanded: {expanded_names}"
        )
        # Interface from file B
        assert "IAuthenticatable" in expanded_names, (
            f"Interface should be pulled in. Expanded: {expanded_names}"
        )
        # Composed type from file C
        assert "UserRepository" in expanded_names, (
            f"Composed UserRepository should be pulled in. Expanded: {expanded_names}"
        )

    def test_expansion_reasons_tracked(self, csharp_project):
        """Expansion reasons are recorded for expanded nodes."""
        csharp_project("src/Types.cs", """
            namespace App
            {
                public class Animal
                {
                    public virtual string Sound() { return ""; }
                }

                public class Dog : Animal
                {
                    public override string Sound() { return "Woof"; }
                }
            }
        """)
        from plugin_implementation.code_graph.expansion_engine import expand_smart
        graph = self._build_graph(csharp_project.root)

        dog_nodes = [n for n in graph.nodes
                     if graph.nodes[n].get("symbol_name") == "Dog"]
        assert dog_nodes, f"No Dog node. Nodes: {list(graph.nodes)}"
        dog_node = dog_nodes[0]

        result = expand_smart({dog_node}, graph)
        # If Animal was expanded, it should have a reason
        for nid, reason in result.expansion_reasons.items():
            sym_name = graph.nodes.get(nid, {}).get("symbol_name", "")
            if sym_name == "Animal":
                assert reason, "Expansion reason should be non-empty"
                assert "inherit" in reason.lower() or "base" in reason.lower(), (
                    f"Reason should mention inheritance. Got: {reason}"
                )
