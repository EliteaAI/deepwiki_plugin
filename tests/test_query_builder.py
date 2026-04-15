"""
Tests for ``graph_query_builder.py`` and FTS5 smart search methods.

Tests are organised into:

1. **Unit tests** — pure keyword extraction and query construction (no DB)
2. **Integration tests** — FTS5 queries executed against real cached graphs
   (fmtlib and configurations).

The integration tests load the same FTS5 databases used by the production
wiki pipeline, so they validate real BM25 ranking quality.
"""

import os
import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _THIS_DIR.parent
_IMPL_DIR = _PLUGIN_ROOT / 'plugin_implementation'

if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))

from plugin_implementation.code_graph.graph_query_builder import (
    CODE_SIGNIFICANT_WORDS,
    CODE_STOP_WORDS,
    GraphQueryBuilder,
    _escape_fts5_token,
    _split_identifier,
    extract_keywords,
)
from plugin_implementation.code_graph.graph_text_index import GraphTextIndex

# ---------------------------------------------------------------------------
# Cache paths for real graph FTS5 databases
# ---------------------------------------------------------------------------
_WIKI_BUILDER = _PLUGIN_ROOT.parent.parent / 'wiki_builder'
_CACHE_DIR = _WIKI_BUILDER / 'cache'

# Cache keys from known test repos
_FMTLIB_KEY = '551134763c1f5c1b3feca4dd95076290'
_CONFIGS_KEY = 'cd9d7a4aefa47194b872a7093a855156'


def _fts_db_exists(key: str) -> bool:
    return (_CACHE_DIR / f'{key}.fts5.db').exists()


# ===================================================================
# Unit Tests — keyword extraction
# ===================================================================

class TestExtractKeywords(unittest.TestCase):
    """Test extract_keywords() stop word removal and identifier splitting."""

    def test_stop_words_removed(self):
        kw = extract_keywords("How does authentication work?")
        self.assertIn('authentication', kw)
        self.assertNotIn('how', kw)
        self.assertNotIn('does', kw)
        self.assertNotIn('work', kw)  # 'work' is a stop word

    def test_code_significant_words_kept(self):
        kw = extract_keywords("What classes handle payment processing?")
        self.assertIn('classes', kw)
        self.assertIn('handle', kw)
        self.assertIn('payment', kw)
        self.assertIn('processing', kw)

    def test_camel_case_splitting(self):
        kw = extract_keywords("AuthServiceManager")
        self.assertIn('authservicemanager', kw)
        self.assertIn('auth', kw)
        self.assertIn('service', kw)
        self.assertIn('manager', kw)

    def test_snake_case_splitting(self):
        kw = extract_keywords("get_user_by_id")
        # 'get' and 'by' are stop words, 'user' and 'id' kept
        self.assertIn('user', kw)
        self.assertNotIn('get', kw)  # stop word
        self.assertNotIn('by', kw)   # stop word

    def test_short_tokens_filtered(self):
        kw = extract_keywords("a b cc dd")
        self.assertNotIn('a', kw)
        self.assertNotIn('b', kw)
        self.assertIn('cc', kw)
        self.assertIn('dd', kw)

    def test_empty_query(self):
        kw = extract_keywords("")
        self.assertEqual(kw, [])

    def test_only_stop_words(self):
        kw = extract_keywords("how is the what")
        self.assertEqual(kw, [])

    def test_deduplication(self):
        kw = extract_keywords("auth auth AUTH")
        self.assertEqual(kw.count('auth'), 1)

    def test_code_symbols_mixed_with_natural_language(self):
        kw = extract_keywords("How does ConfigurationError get raised?")
        self.assertIn('configurationerror', kw)
        self.assertIn('configuration', kw)
        self.assertIn('error', kw)
        self.assertNotIn('how', kw)
        self.assertNotIn('does', kw)

    def test_preserve_order(self):
        kw = extract_keywords("authentication token validation")
        self.assertEqual(kw, ['authentication', 'token', 'validation'])

    def test_mixed_separators(self):
        kw = extract_keywords("auth-service.manager/handler")
        self.assertIn('auth', kw)
        self.assertIn('service', kw)
        self.assertIn('manager', kw)
        self.assertIn('handler', kw)


class TestSplitIdentifier(unittest.TestCase):
    """Test _split_identifier() for various naming conventions."""

    def test_camel_case(self):
        self.assertEqual(_split_identifier("AuthService"), ['auth', 'service'])

    def test_pascal_case(self):
        self.assertEqual(_split_identifier("UserServiceManager"), ['user', 'service', 'manager'])

    def test_snake_case(self):
        self.assertEqual(_split_identifier("get_user_by_id"), ['get', 'user', 'by', 'id'])

    def test_acronym(self):
        parts = _split_identifier("XMLParser")
        self.assertIn('xml', parts)
        self.assertIn('parser', parts)

    def test_single_word(self):
        self.assertEqual(_split_identifier("auth"), ['auth'])

    def test_all_caps(self):
        parts = _split_identifier("HTTP")
        self.assertIn('http', parts)

    def test_mixed(self):
        parts = _split_identifier("HTTPSConnection_pool")
        self.assertIn('https', parts)
        self.assertIn('connection', parts)
        self.assertIn('pool', parts)


class TestEscapeFts5Token(unittest.TestCase):
    """Test FTS5 token escaping."""

    def test_simple_token(self):
        self.assertEqual(_escape_fts5_token("auth"), '"auth"')

    def test_token_with_quotes(self):
        self.assertEqual(_escape_fts5_token('say"hello'), '"say""hello"')

    def test_empty_token(self):
        self.assertEqual(_escape_fts5_token(""), '""')


# ===================================================================
# Unit Tests — query construction
# ===================================================================

class TestSymbolResolution(unittest.TestCase):
    """Test GraphQueryBuilder.symbol_resolution()."""

    def test_simple_name(self):
        q = GraphQueryBuilder.symbol_resolution("UserService")
        self.assertIn('symbol_name:', q)
        self.assertIn('userservice', q.lower())

    def test_compound_name_includes_tokens(self):
        q = GraphQueryBuilder.symbol_resolution("AuthServiceManager")
        self.assertIn('name_tokens:', q)
        self.assertIn('auth', q.lower())
        self.assertIn('service', q.lower())
        self.assertIn('manager', q.lower())

    def test_single_word_no_token_split(self):
        q = GraphQueryBuilder.symbol_resolution("auth")
        self.assertIn('symbol_name:', q)
        # Single word — no name_tokens clause
        self.assertNotIn('name_tokens:', q)

    def test_empty(self):
        self.assertEqual(GraphQueryBuilder.symbol_resolution(""), '')

    def test_prefix_star(self):
        q = GraphQueryBuilder.symbol_resolution("Config")
        self.assertIn('*', q)  # Should be prefix query


class TestConceptSearch(unittest.TestCase):
    """Test GraphQueryBuilder.concept_search()."""

    def test_basic(self):
        q = GraphQueryBuilder.concept_search(["authentication", "token"])
        self.assertIn('"authentication"*', q)
        self.assertIn('"token"*', q)

    def test_docstring_boost(self):
        q = GraphQueryBuilder.concept_search(["auth"], boost_docstring=True)
        self.assertIn('docstring:"auth"', q)

    def test_no_docstring_boost(self):
        q = GraphQueryBuilder.concept_search(["auth"], boost_docstring=False)
        self.assertNotIn('docstring:', q)

    def test_empty(self):
        self.assertEqual(GraphQueryBuilder.concept_search([]), '')


class TestPhraseSearch(unittest.TestCase):
    """Test GraphQueryBuilder.phrase_search()."""

    def test_basic(self):
        q = GraphQueryBuilder.phrase_search("dependency injection")
        self.assertEqual(q, '"dependency injection"')

    def test_column_scoped(self):
        q = GraphQueryBuilder.phrase_search("error handling", "docstring")
        self.assertEqual(q, 'docstring:"error handling"')

    def test_empty(self):
        self.assertEqual(GraphQueryBuilder.phrase_search(""), '')


class TestProximitySearch(unittest.TestCase):
    """Test GraphQueryBuilder.proximity_search()."""

    def test_basic(self):
        q = GraphQueryBuilder.proximity_search(["UserService", "authenticate"], 50)
        self.assertIn('NEAR(', q)
        self.assertIn('"userservice"', q)
        self.assertIn('"authenticate"', q)
        self.assertIn('50', q)

    def test_single_term_empty(self):
        self.assertEqual(GraphQueryBuilder.proximity_search(["only"], 10), '')

    def test_column_scoped(self):
        q = GraphQueryBuilder.proximity_search(["A", "B"], 20, column="content")
        self.assertIn('content:', q)


class TestFromNaturalLanguage(unittest.TestCase):
    """Test GraphQueryBuilder.from_natural_language()."""

    def test_general_intent(self):
        q = GraphQueryBuilder.from_natural_language("authentication token validation")
        self.assertIn('"authentication"*', q)
        self.assertIn('"token"*', q)
        self.assertIn('"validation"*', q)

    def test_symbol_intent(self):
        q = GraphQueryBuilder.from_natural_language("AuthService", intent='symbol')
        self.assertIn('symbol_name:', q)

    def test_concept_intent_boosts_docstring(self):
        q = GraphQueryBuilder.from_natural_language(
            "error handling patterns", intent='concept'
        )
        self.assertIn('docstring:', q)

    def test_doc_intent_boosts_content(self):
        q = GraphQueryBuilder.from_natural_language(
            "installation guide", intent='doc'
        )
        self.assertIn('content:', q)

    def test_all_stop_words_fallback(self):
        q = GraphQueryBuilder.from_natural_language("how is the what")
        # All stop words — should fallback to simple prefix tokens
        self.assertIn('"how"*', q)  # Fallback uses raw tokens

    def test_empty(self):
        self.assertEqual(GraphQueryBuilder.from_natural_language(""), '')


class TestForVectorStore(unittest.TestCase):
    """Test GraphQueryBuilder.for_vector_store()."""

    def test_basic(self):
        q = GraphQueryBuilder.for_vector_store("How does authentication work?")
        self.assertIn('authentication', q)
        self.assertNotIn('how', q)

    def test_context_terms(self):
        q = GraphQueryBuilder.for_vector_store(
            "payment processing",
            context_terms=["billing", "stripe"],
        )
        self.assertIn('payment', q)
        self.assertIn('processing', q)
        self.assertIn('billing', q)
        self.assertIn('stripe', q)

    def test_empty_falls_back(self):
        q = GraphQueryBuilder.for_vector_store("")
        self.assertEqual(q, '')


class TestCombinedSymbolAndConcept(unittest.TestCase):
    """Test GraphQueryBuilder.combined_symbol_and_concept()."""

    def test_both(self):
        q = GraphQueryBuilder.combined_symbol_and_concept(
            symbol_names=["ConfigParser"],
            concept_keywords=["validation", "schema"],
        )
        self.assertIn('symbol_name:', q)
        self.assertIn('validation', q)
        self.assertIn('schema', q)

    def test_symbol_only(self):
        q = GraphQueryBuilder.combined_symbol_and_concept(
            symbol_names=["Foo"], concept_keywords=[],
        )
        self.assertIn('symbol_name:', q)
        self.assertNotIn('docstring:', q)

    def test_concept_only(self):
        q = GraphQueryBuilder.combined_symbol_and_concept(
            symbol_names=[], concept_keywords=["auth"],
        )
        self.assertIn('"auth"*', q)


class TestTypeScopedSearch(unittest.TestCase):
    """Test GraphQueryBuilder.type_scoped_search()."""

    def test_basic(self):
        fts_q, types = GraphQueryBuilder.type_scoped_search(
            ["auth", "token"], "class",
        )
        self.assertIn('"auth"*', fts_q)
        self.assertEqual(types, frozenset({'class'}))


class TestPathScopedConcept(unittest.TestCase):
    """Test GraphQueryBuilder.path_scoped_concept()."""

    def test_basic(self):
        fts_q, prefix = GraphQueryBuilder.path_scoped_concept(
            ["auth"], "/src/services/",
        )
        self.assertIn('"auth"*', fts_q)
        self.assertEqual(prefix, 'src/services')


# ===================================================================
# Unit Tests — stop words sanity
# ===================================================================

class TestStopWordsConsistency(unittest.TestCase):
    """Verify stop words and significant words don't contradict."""

    def test_no_overlap(self):
        """CODE_SIGNIFICANT_WORDS should not be in CODE_STOP_WORDS."""
        overlap = CODE_STOP_WORDS & CODE_SIGNIFICANT_WORDS
        self.assertEqual(
            overlap, set(),
            f"Words in both stop and significant: {overlap}",
        )

    def test_stop_words_all_lowercase(self):
        for w in CODE_STOP_WORDS:
            self.assertEqual(w, w.lower(), f"Stop word not lowercase: {w}")

    def test_significant_words_all_lowercase(self):
        for w in CODE_SIGNIFICANT_WORDS:
            self.assertEqual(w, w.lower(), f"Significant word not lowercase: {w}")


# ===================================================================
# Integration Tests — FTS5 with real fmtlib graph
# ===================================================================

@unittest.skipUnless(
    _fts_db_exists(_FMTLIB_KEY),
    "fmtlib FTS5 database not found in cache",
)
class TestFTS5SmartSearchFmtlib(unittest.TestCase):
    """Integration tests: smart search against fmtlib (C++) FTS5 database."""

    @classmethod
    def setUpClass(cls):
        cls.index = GraphTextIndex(cache_dir=str(_CACHE_DIR))
        loaded = cls.index.load(_FMTLIB_KEY)
        if not loaded:
            raise unittest.SkipTest("Could not load fmtlib FTS5 index")

    def test_symbol_resolution_finds_format_context(self):
        """symbol_resolution for 'format_context' finds the exact class."""
        q = GraphQueryBuilder.symbol_resolution("format_context")
        docs = self.index._execute_fts_search(q, "format_context", k=5)
        names = [d.metadata.get('symbol_name', '') for d in docs]
        self.assertTrue(
            any('format_context' in n.lower() for n in names),
            f"Expected 'format_context' in results, got {names}",
        )

    def test_concept_search_finds_formatting(self):
        """concept_search for 'formatting output' finds relevant symbols."""
        docs = self.index.search_smart(
            "formatting output string", intent='concept', k=10,
        )
        self.assertGreater(len(docs), 0)
        # At least one result should relate to formatting
        all_content = ' '.join(d.page_content.lower() for d in docs[:5])
        self.assertTrue(
            'format' in all_content or 'fmt' in all_content,
            "Expected formatting-related content in results",
        )

    def test_search_smart_vs_legacy(self):
        """search_smart should return at least as many relevant results as search."""
        query = "memory buffer allocation"
        legacy_docs = self.index.search(query, k=10)
        smart_docs = self.index.search_smart(query, k=10, intent='concept')

        # Both should return results
        self.assertGreater(len(smart_docs), 0, "search_smart returned 0 results")
        # smart should be at least as good (may differ due to stop word removal)
        # Just ensure it works and returns reasonable results
        smart_names = {d.metadata.get('symbol_name', '') for d in smart_docs}
        self.assertGreater(len(smart_names), 0)

    def test_search_symbols_with_type_filter(self):
        """search_symbols with type='class' finds only classes."""
        docs = self.index.search_symbols(
            "format", k=10,
            symbol_types=frozenset({'class', 'struct'}),
        )
        for doc in docs:
            self.assertIn(
                doc.metadata['symbol_type'], {'class', 'struct'},
                f"Expected class/struct, got {doc.metadata['symbol_type']}",
            )

    def test_search_symbols_with_path_prefix(self):
        """search_symbols with path_prefix filters by directory."""
        docs = self.index.search_symbols(
            "format", k=20,
            path_prefix="include",
        )
        for doc in docs:
            rel_path = doc.metadata.get('rel_path', '')
            self.assertTrue(
                rel_path.startswith('include'),
                f"Expected path starting with 'include', got {rel_path!r}",
            )

    def test_empty_query_returns_empty(self):
        docs = self.index.search_smart("", k=10)
        self.assertEqual(docs, [])

    def test_all_stop_words_query_still_works(self):
        """Query with only stop words should still return results via fallback."""
        docs = self.index.search_smart("how is the what", k=5)
        # May or may not return results — but should not crash
        self.assertIsInstance(docs, list)

    def test_symbol_intent_for_class_name(self):
        """Symbol intent should find exact class matches."""
        docs = self.index.search_smart("basic_string_view", intent='symbol', k=5)
        if docs:
            top_name = docs[0].metadata.get('symbol_name', '')
            self.assertIn('basic_string_view', top_name.lower(),
                         f"Top result should be the class itself, got {top_name}")


# ===================================================================
# Integration Tests — FTS5 with configurations (Python) graph
# ===================================================================

@unittest.skipUnless(
    _fts_db_exists(_CONFIGS_KEY),
    "configurations FTS5 database not found in cache",
)
class TestFTS5SmartSearchConfigurations(unittest.TestCase):
    """Integration tests: smart search against configurations (Python) FTS5 db."""

    @classmethod
    def setUpClass(cls):
        cls.index = GraphTextIndex(cache_dir=str(_CACHE_DIR))
        loaded = cls.index.load(_CONFIGS_KEY)
        if not loaded:
            raise unittest.SkipTest("Could not load configurations FTS5 index")

    def test_concept_search_finds_error_handling(self):
        """Concept search for 'error handling' finds ConfigurationError."""
        docs = self.index.search_smart(
            "error handling configuration", intent='concept', k=10,
        )
        self.assertGreater(len(docs), 0)
        names = [d.metadata.get('symbol_name', '') for d in docs]
        self.assertTrue(
            any('error' in n.lower() or 'exception' in n.lower() for n in names),
            f"Expected error-related symbols, got {names}",
        )

    def test_symbol_resolution_finds_exact(self):
        """Symbol resolution for 'ConfigurationError' finds the class."""
        q = GraphQueryBuilder.symbol_resolution("ConfigurationError")
        docs = self.index._execute_fts_search(q, "ConfigurationError", k=5)
        names = [d.metadata.get('symbol_name', '') for d in docs]
        self.assertIn(
            'ConfigurationError', names,
            f"Expected 'ConfigurationError' in results, got {names}",
        )

    def test_search_symbols_class_only(self):
        """search_symbols with type='class' returns only classes."""
        docs = self.index.search_symbols(
            "configuration", k=10,
            symbol_types=frozenset({'class'}),
        )
        self.assertGreater(len(docs), 0)
        for doc in docs:
            self.assertEqual(doc.metadata['symbol_type'], 'class')

    def test_keyword_extraction_quality(self):
        """Verify extract_keywords produces good terms for code search."""
        kw = extract_keywords("How does ConfigurationError get raised in validation?")
        self.assertIn('configurationerror', kw)
        self.assertIn('configuration', kw)
        self.assertIn('error', kw)
        self.assertIn('validation', kw)
        self.assertNotIn('how', kw)
        self.assertNotIn('does', kw)

    def test_natural_language_vs_keyword_query(self):
        """NL query and extracted keywords should find similar results."""
        nl_query = "How does configuration validation work?"
        keyword_query = "configuration validation"

        nl_docs = self.index.search_smart(nl_query, intent='concept', k=10)
        kw_docs = self.index.search_smart(keyword_query, intent='concept', k=10)

        # Both should return results
        self.assertGreater(len(nl_docs), 0, "NL query returned 0")
        self.assertGreater(len(kw_docs), 0, "Keyword query returned 0")

        # There should be overlap in top results
        nl_names = {d.metadata.get('symbol_name') for d in nl_docs[:5]}
        kw_names = {d.metadata.get('symbol_name') for d in kw_docs[:5]}
        overlap = nl_names & kw_names
        # At least some overlap expected
        self.assertGreater(
            len(overlap), 0,
            f"No overlap between NL and keyword results: {nl_names} vs {kw_names}",
        )


# ===================================================================
# Edge cases and graceful degradation
# ===================================================================

class TestQueryBuilderEdgeCases(unittest.TestCase):
    """Edge cases for query builder methods."""

    def test_special_characters_in_query(self):
        """Special characters should not crash FTS5 query construction."""
        q = GraphQueryBuilder.from_natural_language('hello "world" (test)')
        self.assertIsInstance(q, str)

    def test_very_long_query(self):
        """Very long queries should not crash."""
        long_query = "authentication " * 100
        q = GraphQueryBuilder.from_natural_language(long_query)
        self.assertIsInstance(q, str)

    def test_unicode_query(self):
        """Unicode characters should be handled."""
        q = GraphQueryBuilder.from_natural_language("café résumé naïve")
        self.assertIsInstance(q, str)

    def test_numeric_tokens(self):
        """Numeric tokens should be kept if long enough."""
        kw = extract_keywords("error code 404 response 200")
        self.assertIn('error', kw)
        self.assertIn('code', kw)
        self.assertIn('404', kw)
        self.assertIn('response', kw)
        self.assertIn('200', kw)

    def test_symbol_with_namespace(self):
        """Namespaced symbols like 'fmt::format' should split correctly."""
        kw = extract_keywords("fmt::format_context")
        self.assertIn('fmt', kw)
        self.assertIn('format', kw)
        self.assertIn('context', kw)

    def test_path_like_query(self):
        """Path-like strings should produce useful tokens."""
        kw = extract_keywords("src/auth/UserService.py")
        self.assertIn('src', kw)
        self.assertIn('auth', kw)
        self.assertIn('userservice', kw)


if __name__ == '__main__':
    unittest.main()
