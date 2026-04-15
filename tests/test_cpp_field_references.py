"""
Phase 4 tests — C++ qualified name references in field declarations.

Tests that the RelationshipVisitor.visit_field_declaration correctly emits
REFERENCES edges for:
1. Plain type identifiers (Point member;)
2. Qualified identifiers without templates (fmt::memory_buffer buf;)
3. Qualified identifiers with templates (std::vector<Point> v;)
4. Unqualified template types (optional<Point> opt;)
5. Std-library types are filtered from REFERENCES (but template args are kept)
"""

import pytest
import tempfile
import os
from pathlib import Path

from plugin_implementation.parsers.cpp_enhanced_parser import CppEnhancedParser


# ---------------------------------------------------------------------------
# Test C++ source snippets
# ---------------------------------------------------------------------------

FIELD_PLAIN_TYPE = """\
class Renderer {
    Point origin;
    int width;
};
"""

FIELD_QUALIFIED_TYPE = """\
class Logger {
    fmt::memory_buffer buffer;
    spdlog::logger log;
};
"""

FIELD_QUALIFIED_TEMPLATE_TYPE = """\
class Container {
    std::vector<Point> points;
    std::unordered_map<std::string, Widget> widgets;
    boost::shared_ptr<Config> config;
};
"""

FIELD_UNQUALIFIED_TEMPLATE = """\
class Wrapper {
    optional<Point> maybe_point;
    Ref<Texture> tex;
};
"""

FIELD_MIXED = """\
class Engine {
    Renderer renderer;
    fmt::memory_buffer buf;
    std::vector<Shader> shaders;
    optional<Light> ambient;
    int count;
};
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return CppEnhancedParser()


def _write_and_parse(parser, source, filename="test.cpp"):
    """Write source to a temp file and parse it, returning relationships."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, filename)
        with open(fpath, 'w') as f:
            f.write(source)
        result = parser.parse_file(fpath)
        return result.relationships


def _ref_targets(relationships, source_prefix=None):
    """Get target names of REFERENCES relationships, optionally filtered by source prefix."""
    refs = []
    for rel in relationships:
        if rel.relationship_type.value == 'references':
            if source_prefix is None or rel.source_symbol.startswith(source_prefix):
                refs.append(rel.target_symbol)
    return refs


def _ref_with_annotation(relationships, annotation_key='reference_type', annotation_value='field_type'):
    """Get REFERENCES relationships that have specific annotation."""
    return [
        rel for rel in relationships
        if rel.relationship_type.value == 'references'
        and rel.annotations.get(annotation_key) == annotation_value
    ]


# ---------------------------------------------------------------------------
# Tests — plain type_identifier in fields
# ---------------------------------------------------------------------------

class TestFieldPlainType:
    """class Renderer { Point origin; int width; };"""

    def test_references_user_type(self, parser):
        rels = _write_and_parse(parser, FIELD_PLAIN_TYPE)
        targets = _ref_targets(rels)
        assert 'Point' in targets, f"Expected 'Point' in REFERENCES targets, got {targets}"

    def test_no_reference_for_int(self, parser):
        """Built-in types like 'int' should not generate REFERENCES."""
        rels = _write_and_parse(parser, FIELD_PLAIN_TYPE)
        targets = _ref_targets(rels)
        assert 'int' not in targets

    def test_field_type_annotation(self, parser):
        rels = _write_and_parse(parser, FIELD_PLAIN_TYPE)
        field_refs = _ref_with_annotation(rels)
        field_targets = [r.target_symbol for r in field_refs]
        assert 'Point' in field_targets


# ---------------------------------------------------------------------------
# Tests — qualified_identifier in fields (no template)
# ---------------------------------------------------------------------------

class TestFieldQualifiedType:
    """class Logger { fmt::memory_buffer buffer; spdlog::logger log; };"""

    def test_qualified_ref_fmt(self, parser):
        rels = _write_and_parse(parser, FIELD_QUALIFIED_TYPE)
        targets = _ref_targets(rels)
        assert 'fmt.memory_buffer' in targets, f"Expected 'fmt.memory_buffer', got {targets}"

    def test_qualified_ref_spdlog(self, parser):
        rels = _write_and_parse(parser, FIELD_QUALIFIED_TYPE)
        targets = _ref_targets(rels)
        assert 'spdlog.logger' in targets, f"Expected 'spdlog.logger', got {targets}"

    def test_field_type_annotation(self, parser):
        rels = _write_and_parse(parser, FIELD_QUALIFIED_TYPE)
        field_refs = _ref_with_annotation(rels)
        field_targets = [r.target_symbol for r in field_refs]
        assert 'fmt.memory_buffer' in field_targets


# ---------------------------------------------------------------------------
# Tests — qualified_identifier with template in fields
# ---------------------------------------------------------------------------

class TestFieldQualifiedTemplate:
    """
    class Container {
        std::vector<Point> points;
        std::unordered_map<std::string, Widget> widgets;
        boost::shared_ptr<Config> config;
    };
    """

    def test_template_arg_point(self, parser):
        """std::vector<Point> → REFERENCES to Point (template arg)."""
        rels = _write_and_parse(parser, FIELD_QUALIFIED_TEMPLATE_TYPE)
        targets = _ref_targets(rels)
        assert 'Point' in targets, f"Expected 'Point' in targets, got {targets}"

    def test_template_arg_widget(self, parser):
        """std::unordered_map<std::string, Widget> → REFERENCES to Widget."""
        rels = _write_and_parse(parser, FIELD_QUALIFIED_TEMPLATE_TYPE)
        targets = _ref_targets(rels)
        assert 'Widget' in targets, f"Expected 'Widget' in targets, got {targets}"

    def test_std_vector_filtered(self, parser):
        """std::vector itself should NOT appear as a REFERENCES target."""
        rels = _write_and_parse(parser, FIELD_QUALIFIED_TEMPLATE_TYPE)
        targets = _ref_targets(rels)
        assert 'std.vector' not in targets

    def test_std_unordered_map_filtered(self, parser):
        rels = _write_and_parse(parser, FIELD_QUALIFIED_TEMPLATE_TYPE)
        targets = _ref_targets(rels)
        assert 'std.unordered_map' not in targets

    def test_boost_shared_ptr_not_filtered(self, parser):
        """boost::shared_ptr is NOT in std_types filter, so it should appear."""
        rels = _write_and_parse(parser, FIELD_QUALIFIED_TEMPLATE_TYPE)
        targets = _ref_targets(rels)
        assert 'boost.shared_ptr' in targets, f"Expected 'boost.shared_ptr', got {targets}"

    def test_config_template_arg(self, parser):
        """boost::shared_ptr<Config> → REFERENCES to Config."""
        rels = _write_and_parse(parser, FIELD_QUALIFIED_TEMPLATE_TYPE)
        targets = _ref_targets(rels)
        assert 'Config' in targets


# ---------------------------------------------------------------------------
# Tests — unqualified template_type in fields
# ---------------------------------------------------------------------------

class TestFieldUnqualifiedTemplate:
    """
    class Wrapper { optional<Point> maybe_point; Ref<Texture> tex; };
    """

    def test_template_arg_point(self, parser):
        """optional<Point> → REFERENCES to Point."""
        rels = _write_and_parse(parser, FIELD_UNQUALIFIED_TEMPLATE)
        targets = _ref_targets(rels)
        assert 'Point' in targets, f"Expected 'Point', got {targets}"

    def test_template_arg_texture(self, parser):
        """Ref<Texture> → REFERENCES to Texture."""
        rels = _write_and_parse(parser, FIELD_UNQUALIFIED_TEMPLATE)
        targets = _ref_targets(rels)
        assert 'Texture' in targets, f"Expected 'Texture', got {targets}"


# ---------------------------------------------------------------------------
# Tests — mixed field types in one class
# ---------------------------------------------------------------------------

class TestFieldMixedTypes:
    """
    class Engine {
        Renderer renderer;
        fmt::memory_buffer buf;
        std::vector<Shader> shaders;
        optional<Light> ambient;
        int count;
    };
    """

    def test_plain_type(self, parser):
        rels = _write_and_parse(parser, FIELD_MIXED)
        targets = _ref_targets(rels)
        assert 'Renderer' in targets

    def test_qualified_type(self, parser):
        rels = _write_and_parse(parser, FIELD_MIXED)
        targets = _ref_targets(rels)
        assert 'fmt.memory_buffer' in targets

    def test_template_arg(self, parser):
        rels = _write_and_parse(parser, FIELD_MIXED)
        targets = _ref_targets(rels)
        assert 'Shader' in targets

    def test_unqualified_template_arg(self, parser):
        rels = _write_and_parse(parser, FIELD_MIXED)
        targets = _ref_targets(rels)
        assert 'Light' in targets

    def test_int_not_referenced(self, parser):
        rels = _write_and_parse(parser, FIELD_MIXED)
        targets = _ref_targets(rels)
        assert 'int' not in targets

    def test_std_vector_filtered(self, parser):
        rels = _write_and_parse(parser, FIELD_MIXED)
        targets = _ref_targets(rels)
        assert 'std.vector' not in targets
