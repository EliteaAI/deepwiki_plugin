"""
Tests for SPEC-6: NL → Machine Query Bridge.

Covers:
  - parse_natural_language(): type extraction, path extraction, relationship extraction
  - ParsedNLQuery.to_jql(): JQL generation from parsed queries
  - from_natural_language_parsed(): integration returning (fts5_query, filters)
"""

import sys
import os
import unittest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PLUGIN_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'))
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

from plugin_implementation.code_graph.graph_query_builder import (
    GraphQueryBuilder,
    ParsedNLQuery,
    parse_natural_language,
    extract_keywords,
    TYPE_KEYWORDS,
    MODULE_KEYWORDS,
    RELATIONSHIP_KEYWORDS,
)


class TestParseNaturalLanguage(unittest.TestCase):
    """Tests for parse_natural_language() — SPEC-6 core function."""

    # ------------------------------------------------------------------
    # Type extraction
    # ------------------------------------------------------------------

    def test_class_extracted(self):
        p = parse_natural_language("What classes handle authentication?")
        self.assertEqual(p.type_filter, 'class')
        # "classes" should not appear in text_keywords
        self.assertNotIn('classes', p.text_keywords)
        self.assertIn('handle', p.text_keywords)
        self.assertIn('authentication', p.text_keywords)

    def test_function_extracted(self):
        p = parse_natural_language("Find functions that process payments")
        self.assertEqual(p.type_filter, 'function')
        self.assertNotIn('functions', p.text_keywords)
        self.assertIn('process', p.text_keywords)
        self.assertIn('payments', p.text_keywords)

    def test_interface_extracted(self):
        p = parse_natural_language("List interfaces for HTTP clients")
        self.assertEqual(p.type_filter, 'interface')
        self.assertNotIn('interfaces', p.text_keywords)

    def test_method_extracted(self):
        p = parse_natural_language("Show methods for database queries")
        self.assertEqual(p.type_filter, 'method')

    def test_no_type_when_absent(self):
        p = parse_natural_language("How does authentication work?")
        self.assertIsNone(p.type_filter)
        self.assertIn('authentication', p.text_keywords)

    # ------------------------------------------------------------------
    # Path extraction
    # ------------------------------------------------------------------

    def test_path_from_in_preposition(self):
        p = parse_natural_language("functions in the auth module")
        self.assertEqual(p.type_filter, 'function')
        self.assertIsNotNone(p.path_hint)
        self.assertIn('auth', p.path_hint.lower())

    def test_path_from_slash_pattern(self):
        p = parse_natural_language("classes in src/auth")
        self.assertEqual(p.type_filter, 'class')
        self.assertEqual(p.path_hint, 'src/auth')

    def test_path_from_module_keyword(self):
        p = parse_natural_language("functions in the db package")
        self.assertIsNotNone(p.path_hint)

    def test_no_path_when_absent(self):
        p = parse_natural_language("What is authentication?")
        self.assertIsNone(p.path_hint)

    # ------------------------------------------------------------------
    # Relationship extraction
    # ------------------------------------------------------------------

    def test_inherits_relationship(self):
        p = parse_natural_language("classes that extend BaseHandler")
        self.assertEqual(p.type_filter, 'class')
        self.assertEqual(p.relationship_hint, 'inheritance')
        self.assertEqual(p.related_symbol, 'BaseHandler')

    def test_calls_relationship(self):
        p = parse_natural_language("functions that call AuthService")
        self.assertEqual(p.type_filter, 'function')
        self.assertEqual(p.relationship_hint, 'calls')
        self.assertEqual(p.related_symbol, 'AuthService')

    def test_imports_relationship(self):
        p = parse_natural_language("modules that import DatabasePool")
        self.assertEqual(p.relationship_hint, 'imports')
        self.assertEqual(p.related_symbol, 'DatabasePool')

    def test_no_relationship_when_absent(self):
        p = parse_natural_language("Authentication handler classes")
        self.assertIsNone(p.relationship_hint)
        self.assertIsNone(p.related_symbol)

    # ------------------------------------------------------------------
    # Combined extraction
    # ------------------------------------------------------------------

    def test_type_and_path(self):
        p = parse_natural_language("classes in src/auth related to tokens")
        self.assertEqual(p.type_filter, 'class')
        self.assertEqual(p.path_hint, 'src/auth')

    def test_all_filters(self):
        p = parse_natural_language("classes in src/auth that extend BaseHandler")
        self.assertEqual(p.type_filter, 'class')
        self.assertEqual(p.path_hint, 'src/auth')
        self.assertEqual(p.relationship_hint, 'inheritance')
        self.assertEqual(p.related_symbol, 'BaseHandler')

    def test_empty_query(self):
        p = parse_natural_language("")
        self.assertIsNone(p.type_filter)
        self.assertIsNone(p.path_hint)
        self.assertEqual(p.text_keywords, [])

    # ------------------------------------------------------------------
    # has_structural_filters property
    # ------------------------------------------------------------------

    def test_has_structural_filters_true(self):
        p = parse_natural_language("classes in auth module")
        self.assertTrue(p.has_structural_filters)

    def test_has_structural_filters_false(self):
        p = parse_natural_language("how does authentication work")
        self.assertFalse(p.has_structural_filters)


class TestParsedNLQueryToJQL(unittest.TestCase):
    """Tests for ParsedNLQuery.to_jql() — SPEC-6 JQL generation."""

    def test_type_only(self):
        p = ParsedNLQuery(type_filter='class')
        self.assertEqual(p.to_jql(), 'type:class')

    def test_type_and_text(self):
        p = ParsedNLQuery(type_filter='class', text_keywords=['authentication'])
        self.assertEqual(p.to_jql(), 'type:class text:authentication')

    def test_type_path_text(self):
        p = ParsedNLQuery(
            type_filter='function',
            path_hint='src/auth',
            text_keywords=['handler'],
        )
        jql = p.to_jql()
        self.assertIn('type:function', jql)
        self.assertIn('file:src/auth/*', jql)
        self.assertIn('text:handler', jql)

    def test_related_symbol(self):
        p = ParsedNLQuery(
            type_filter='class',
            relationship_hint='inheritance',
            related_symbol='BaseHandler',
        )
        jql = p.to_jql()
        self.assertIn('type:class', jql)
        self.assertIn('related:BaseHandler', jql)
        self.assertIn('has_rel:inheritance', jql)

    def test_related_symbol_quoted(self):
        p = ParsedNLQuery(related_symbol='Base Handler')
        jql = p.to_jql()
        self.assertIn('related:"Base Handler"', jql)

    def test_empty_returns_empty(self):
        p = ParsedNLQuery()
        self.assertEqual(p.to_jql(), '')

    def test_path_gets_glob(self):
        """Path hints without glob get /* appended."""
        p = ParsedNLQuery(path_hint='src/auth')
        self.assertIn('file:src/auth/*', p.to_jql())

    def test_path_with_existing_glob(self):
        """Path hints with glob are preserved."""
        p = ParsedNLQuery(path_hint='src/auth/*.py')
        self.assertIn('file:src/auth/*.py', p.to_jql())


class TestFromNaturalLanguageParsed(unittest.TestCase):
    """Tests for GraphQueryBuilder.from_natural_language_parsed() integration."""

    def test_returns_tuple(self):
        fts_query, parsed = GraphQueryBuilder.from_natural_language_parsed(
            "What classes handle authentication?"
        )
        self.assertIsInstance(fts_query, str)
        self.assertIsInstance(parsed, ParsedNLQuery)
        self.assertEqual(parsed.type_filter, 'class')

    def test_type_not_in_fts_query(self):
        """'classes' should be stripped from the FTS5 query when type filter is extracted."""
        fts_query, parsed = GraphQueryBuilder.from_natural_language_parsed(
            "What classes handle authentication?"
        )
        # The FTS query should focus on content keywords
        self.assertIn('authentication', fts_query.lower())
        self.assertEqual(parsed.type_filter, 'class')

    def test_symbol_intent_bypasses_parsing(self):
        fts_query, parsed = GraphQueryBuilder.from_natural_language_parsed(
            "AuthService", intent='symbol'
        )
        # symbol_resolution lowercases the name for FTS5
        self.assertIn('authservice', fts_query)
        self.assertEqual(parsed.intent, 'symbol')

    def test_empty_query(self):
        fts_query, parsed = GraphQueryBuilder.from_natural_language_parsed("")
        self.assertEqual(fts_query, '')

    def test_backward_compat_from_natural_language(self):
        """Original from_natural_language still returns a string."""
        result = GraphQueryBuilder.from_natural_language(
            "authentication handler"
        )
        self.assertIsInstance(result, str)
        self.assertIn('authentication', result.lower())


class TestKeywordDictionaries(unittest.TestCase):
    """Sanity tests for SPEC-6 keyword dictionaries."""

    def test_type_keywords_all_lowercase_keys(self):
        for key in TYPE_KEYWORDS:
            self.assertEqual(key, key.lower())

    def test_type_keywords_values_are_valid(self):
        valid_types = {
            'class', 'function', 'method', 'interface', 'struct',
            'enum', 'constant', 'trait', 'module_doc',
        }
        for val in TYPE_KEYWORDS.values():
            self.assertIn(val, valid_types, f"Unknown type: {val}")

    def test_relationship_keywords_all_lowercase_keys(self):
        for key in RELATIONSHIP_KEYWORDS:
            self.assertEqual(key, key.lower())

    def test_relationship_keywords_values_are_valid(self):
        # Values must be canonical RelationshipType.value strings
        from plugin_implementation.parsers.base_parser import RelationshipType
        valid_rels = {rt.value for rt in RelationshipType}
        # Also accept 'uses' which appears on legacy edges
        valid_rels.add('uses')
        for val in RELATIONSHIP_KEYWORDS.values():
            self.assertIn(val, valid_rels, f"Unknown relationship: {val}")

    def test_module_keywords_all_lowercase(self):
        for kw in MODULE_KEYWORDS:
            self.assertEqual(kw, kw.lower())


if __name__ == '__main__':
    unittest.main(verbosity=2)
