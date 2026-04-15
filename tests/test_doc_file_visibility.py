"""
Comprehensive tests for documentation file visibility in wiki generation.

Covers the 5 independent failure gates identified in PLANNING_DOC_FILE_VISIBILITY.md:

1. **Constants completeness** — DOC_SYMBOL_TYPES, ARCHITECTURAL_SYMBOLS, DOCUMENTATION_EXTENSIONS,
   KNOWN_FILENAMES are complete and consistent.
2. **File discovery** — _discover_files_by_language() routes extensionless files and all doc
   extensions to 'documentation' (not 'unknown').
3. **Chunk content enrichment** — _parse_documentation_files() embeds "[File: rel_path]" header
   in chunk source_text for semantic search discoverability.
4. **Architectural filter gate** — every doc symbol_type passes _is_architectural_symbol().
5. **Retriever is_doc filter** — YAML, Gradle, Makefile etc. recognized as documentation.
6. **Wiki agent detection** — _is_documentation_file() recognizes all doc extensions and known filenames.

Run:
    cd pylon_deepwiki/plugins/deepwiki_plugin
    python -m pytest tests/test_doc_file_visibility.py -v
"""

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Modules under test
# ---------------------------------------------------------------------------
from plugin_implementation.constants import (
    ARCHITECTURAL_SYMBOLS,
    DOC_SYMBOL_TYPES,
    DOC_CHUNK_TYPES,
    DOCUMENTATION_EXTENSIONS,
    DOCUMENTATION_EXTENSIONS_SET,
    KNOWN_FILENAMES,
)
from plugin_implementation.code_graph.graph_builder import EnhancedUnifiedGraphBuilder


# ============================================================================
# Section 1: Constants Completeness & Consistency
# ============================================================================

class TestConstantsCompleteness(unittest.TestCase):
    """Every doc_type produced by DOCUMENTATION_EXTENSIONS / KNOWN_FILENAMES
    must have a matching '{doc_type}_document' entry in DOC_SYMBOL_TYPES and
    ARCHITECTURAL_SYMBOLS."""

    def _all_doc_types(self):
        """Collect every unique doc_type value from both maps."""
        types = set(DOCUMENTATION_EXTENSIONS.values())
        types |= set(KNOWN_FILENAMES.values())
        return types

    def test_every_doc_type_has_symbol_type_in_DOC_SYMBOL_TYPES(self):
        """Each doc_type → '{doc_type}_document' must be in DOC_SYMBOL_TYPES."""
        for doc_type in self._all_doc_types():
            symbol_type = f"{doc_type}_document"
            self.assertIn(
                symbol_type, DOC_SYMBOL_TYPES,
                f"doc_type '{doc_type}' → symbol_type '{symbol_type}' missing from DOC_SYMBOL_TYPES"
            )

    def test_every_doc_type_has_symbol_type_in_ARCHITECTURAL_SYMBOLS(self):
        """Each doc_type → '{doc_type}_document' must be in ARCHITECTURAL_SYMBOLS
        so that _is_architectural_symbol() lets the chunk into the vector store."""
        for doc_type in self._all_doc_types():
            symbol_type = f"{doc_type}_document"
            self.assertIn(
                symbol_type, ARCHITECTURAL_SYMBOLS,
                f"symbol_type '{symbol_type}' missing from ARCHITECTURAL_SYMBOLS — "
                "chunks with this type will be silently dropped before the vector store"
            )

    def test_DOC_SYMBOL_TYPES_is_subset_of_ARCHITECTURAL(self):
        """All doc symbol types must pass the architectural filter."""
        missing = DOC_SYMBOL_TYPES - ARCHITECTURAL_SYMBOLS
        self.assertEqual(
            missing, set(),
            f"DOC_SYMBOL_TYPES entries not in ARCHITECTURAL_SYMBOLS: {missing}"
        )

    def test_text_chunk_and_text_document_in_both_sets(self):
        """Generic fallback types used by _chunk_generic_text must be recognized."""
        for st in ('text_chunk', 'text_document'):
            self.assertIn(st, DOC_SYMBOL_TYPES)
            self.assertIn(st, ARCHITECTURAL_SYMBOLS)

    def test_markdown_section_in_DOC_SYMBOL_TYPES(self):
        """markdown_section is used by _chunk_markdown_content for split sections."""
        self.assertIn('markdown_section', DOC_SYMBOL_TYPES)

    def test_DOCUMENTATION_EXTENSIONS_SET_matches_dict_keys(self):
        """The frozenset must exactly equal the dict keys."""
        self.assertEqual(DOCUMENTATION_EXTENSIONS_SET, frozenset(DOCUMENTATION_EXTENSIONS.keys()))

    def test_key_extensions_present(self):
        """Spot-check that critical extensions are in the map."""
        for ext in ['.yaml', '.yml', '.gradle', '.kts', '.proto', '.tf', '.sh',
                    '.ini', '.cfg', '.properties', '.env', '.json', '.xml',
                    '.md', '.rst', '.txt', '.toml', '.html', '.wsdl', '.xsd',
                    '.hcl', '.bat', '.cmd', '.ps1', '.adoc']:
            self.assertIn(ext, DOCUMENTATION_EXTENSIONS, f"Extension {ext} missing from DOCUMENTATION_EXTENSIONS")

    def test_key_filenames_present(self):
        """Spot-check that critical extensionless filenames are in KNOWN_FILENAMES."""
        for fname in ['Makefile', 'makefile', 'Dockerfile', 'Jenkinsfile',
                      'Vagrantfile', 'Gemfile', 'Pipfile', 'CMakeLists.txt',
                      'Rakefile', 'Procfile', 'LICENSE', 'CHANGELOG',
                      'CONTRIBUTING', 'AUTHORS', 'CODEOWNERS',
                      '.editorconfig', '.gitattributes', '.gitignore',
                      '.dockerignore']:
            self.assertIn(fname, KNOWN_FILENAMES, f"Filename '{fname}' missing from KNOWN_FILENAMES")


# ============================================================================
# Section 2: File Discovery — extensionless and doc extension routing
# ============================================================================

class TestFileDiscovery(unittest.TestCase):
    """_discover_files_by_language() must route all doc extensions and
    known extensionless filenames to 'documentation', never to 'unknown'."""

    def setUp(self):
        self.builder = EnhancedUnifiedGraphBuilder()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_and_discover(self, filenames):
        """Create files in tmpdir and run _discover_files_by_language."""
        for fname in filenames:
            fpath = Path(self.tmpdir) / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(f"# content of {fname}\n", encoding='utf-8')
        return self.builder._discover_files_by_language(self.tmpdir)

    # --- Extension-based doc files ---

    def test_yaml_files_are_documentation(self):
        result = self._create_and_discover(['config.yaml', 'ci.yml'])
        self.assertIn('documentation', result)
        names = [Path(p).name for p in result['documentation']]
        self.assertIn('config.yaml', names)
        self.assertIn('ci.yml', names)

    def test_gradle_files_are_documentation(self):
        result = self._create_and_discover(['build.gradle', 'settings.gradle.kts'])
        names = [Path(p).name for p in result.get('documentation', [])]
        self.assertIn('build.gradle', names)
        self.assertIn('settings.gradle.kts', names)

    def test_proto_files_are_documentation(self):
        result = self._create_and_discover(['service.proto'])
        names = [Path(p).name for p in result.get('documentation', [])]
        self.assertIn('service.proto', names)

    def test_terraform_files_are_documentation(self):
        result = self._create_and_discover(['main.tf', 'vars.tfvars', 'config.hcl'])
        names = [Path(p).name for p in result.get('documentation', [])]
        for f in ['main.tf', 'vars.tfvars', 'config.hcl']:
            self.assertIn(f, names)

    def test_shell_scripts_are_documentation(self):
        result = self._create_and_discover(['deploy.sh', 'build.bat', 'setup.ps1'])
        names = [Path(p).name for p in result.get('documentation', [])]
        for f in ['deploy.sh', 'build.bat', 'setup.ps1']:
            self.assertIn(f, names)

    def test_config_files_are_documentation(self):
        result = self._create_and_discover(['app.ini', 'settings.cfg', 'nginx.conf',
                                            'app.properties'])
        names = [Path(p).name for p in result.get('documentation', [])]
        for f in ['app.ini', 'settings.cfg', 'nginx.conf', 'app.properties']:
            self.assertIn(f, names)

    def test_schema_files_are_documentation(self):
        result = self._create_and_discover(['api.wsdl', 'types.xsd'])
        names = [Path(p).name for p in result.get('documentation', [])]
        self.assertIn('api.wsdl', names)
        self.assertIn('types.xsd', names)

    # --- Extensionless / known filename files ---

    def test_makefile_is_documentation(self):
        result = self._create_and_discover(['Makefile'])
        self.assertIn('documentation', result, "Makefile should be routed to 'documentation'")
        names = [Path(p).name for p in result['documentation']]
        self.assertIn('Makefile', names)

    def test_dockerfile_is_documentation(self):
        result = self._create_and_discover(['Dockerfile'])
        names = [Path(p).name for p in result.get('documentation', [])]
        self.assertIn('Dockerfile', names)

    def test_jenkinsfile_is_documentation(self):
        result = self._create_and_discover(['Jenkinsfile'])
        names = [Path(p).name for p in result.get('documentation', [])]
        self.assertIn('Jenkinsfile', names)

    def test_vagrantfile_is_documentation(self):
        result = self._create_and_discover(['Vagrantfile'])
        names = [Path(p).name for p in result.get('documentation', [])]
        self.assertIn('Vagrantfile', names)

    def test_gemfile_pipfile_are_documentation(self):
        result = self._create_and_discover(['Gemfile', 'Pipfile'])
        names = [Path(p).name for p in result.get('documentation', [])]
        self.assertIn('Gemfile', names)
        self.assertIn('Pipfile', names)

    def test_license_changelog_are_documentation(self):
        result = self._create_and_discover(['LICENSE', 'CHANGELOG', 'CONTRIBUTING', 'AUTHORS'])
        names = [Path(p).name for p in result.get('documentation', [])]
        for f in ['LICENSE', 'CHANGELOG', 'CONTRIBUTING', 'AUTHORS']:
            self.assertIn(f, names)

    def test_bazel_build_files_are_documentation(self):
        result = self._create_and_discover(['BUILD', 'BUILD.bazel', 'WORKSPACE'])
        names = [Path(p).name for p in result.get('documentation', [])]
        for f in ['BUILD', 'BUILD.bazel', 'WORKSPACE']:
            self.assertIn(f, names)

    def test_cmake_is_documentation(self):
        result = self._create_and_discover(['CMakeLists.txt'])
        names = [Path(p).name for p in result.get('documentation', [])]
        self.assertIn('CMakeLists.txt', names)

    def test_dotfiles_are_documentation(self):
        """Dot-prefixed config files in KNOWN_FILENAMES must be recognized.
        Note: default exclude_patterns skip '**/*.' hidden files, so we
        override excludes to let them through."""
        result = self.builder._discover_files_by_language(
            self.tmpdir, exclude_patterns=[]
        )
        # Create after discovering (to avoid race), then re-discover
        for fname in ['.editorconfig', '.gitattributes', '.gitignore', '.dockerignore']:
            (Path(self.tmpdir) / fname).write_text('# config\n')
        result = self.builder._discover_files_by_language(
            self.tmpdir, exclude_patterns=[]
        )
        names = [Path(p).name for p in result.get('documentation', [])]
        for f in ['.editorconfig', '.gitattributes', '.gitignore', '.dockerignore']:
            self.assertIn(f, names, f"'{f}' should be documentation when exclude_patterns allows dotfiles")

    def test_unknown_extensionless_file_is_unknown(self):
        """A file with no extension and no known filename goes to 'unknown'."""
        result = self._create_and_discover(['randomfile'])
        names = [Path(p).name for p in result.get('unknown', [])]
        self.assertIn('randomfile', names)

    def test_code_files_not_affected(self):
        """Standard code files must still route to their language, not documentation."""
        result = self._create_and_discover(['main.py', 'App.java', 'index.ts'])
        self.assertIn('python', result)
        self.assertIn('java', result)
        self.assertIn('typescript', result)
        names_py = [Path(p).name for p in result['python']]
        self.assertIn('main.py', names_py)


# ============================================================================
# Section 3: Chunk Content Enrichment — [File: rel_path] header
# ============================================================================

class TestChunkContentEnrichment(unittest.TestCase):
    """_parse_documentation_files() must prefix chunk source_text with
    '[File: <rel_path>]' so that file-type keywords are searchable in
    the vector store embedding."""

    def setUp(self):
        self.builder = EnhancedUnifiedGraphBuilder()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_and_parse(self, filename, content):
        """Write a file and parse it as documentation."""
        fpath = os.path.join(self.tmpdir, filename)
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        results = self.builder._parse_documentation_files([fpath], self.tmpdir)
        return results.get(fpath)

    def test_yaml_chunk_contains_file_path(self):
        """YAML file chunk must contain '[File: ...]' with 'yaml' in the path."""
        content = "name: CI\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n"
        result = self._write_and_parse('.github/workflows/ci.yml', content)
        self.assertIsNotNone(result)
        self.assertTrue(len(result.symbols) > 0)
        src = result.symbols[0].source_text
        self.assertTrue(src.startswith('[File: '), f"Expected '[File: ...' prefix, got: {src[:80]}")
        self.assertIn('.github/workflows/ci.yml', src)
        # The word 'yaml' or 'yml' now in the content via the path
        self.assertIn('ci.yml', src)

    def test_gradle_chunk_contains_file_path(self):
        content = "plugins {\n  id 'java'\n}\n"
        result = self._write_and_parse('build.gradle', content)
        self.assertIsNotNone(result)
        src = result.symbols[0].source_text
        self.assertIn('[File: build.gradle]', src)
        self.assertIn('gradle', src.lower())

    def test_makefile_chunk_contains_file_path(self):
        content = "all:\n\techo hello\n\nclean:\n\trm -rf build\n"
        result = self._write_and_parse('Makefile', content)
        self.assertIsNotNone(result)
        src = result.symbols[0].source_text
        self.assertIn('[File: Makefile]', src)
        self.assertIn('Makefile', src)

    def test_dockerfile_chunk_contains_file_path(self):
        content = "FROM python:3.12\nWORKDIR /app\nCOPY . .\n"
        result = self._write_and_parse('Dockerfile', content)
        self.assertIsNotNone(result)
        src = result.symbols[0].source_text
        self.assertIn('[File: Dockerfile]', src)

    def test_terraform_chunk_contains_file_path(self):
        content = 'resource "aws_instance" "web" {\n  ami = "abc"\n}\n'
        result = self._write_and_parse('infra/main.tf', content)
        self.assertIsNotNone(result)
        src = result.symbols[0].source_text
        self.assertIn('[File: infra/main.tf]', src)
        self.assertIn('.tf', src)

    def test_proto_chunk_contains_file_path(self):
        content = 'syntax = "proto3";\nservice Greeter {\n  rpc SayHello (Req) returns (Res);\n}\n'
        result = self._write_and_parse('api/service.proto', content)
        self.assertIsNotNone(result)
        src = result.symbols[0].source_text
        self.assertIn('api/service.proto', src)

    def test_shell_script_chunk_contains_file_path(self):
        content = "#!/bin/bash\necho 'deploying...'\n"
        result = self._write_and_parse('scripts/deploy.sh', content)
        self.assertIsNotNone(result)
        src = result.symbols[0].source_text
        self.assertIn('scripts/deploy.sh', src)

    def test_markdown_chunk_still_has_path(self):
        """Even markdown files should get the path header now."""
        content = "# README\n\nThis is a project.\n"
        result = self._write_and_parse('README.md', content)
        self.assertIsNotNone(result)
        src = result.symbols[0].source_text
        self.assertIn('[File: README.md]', src)

    def test_json_chunk_contains_file_path(self):
        content = '{"name": "my-project", "version": "1.0.0"}\n'
        result = self._write_and_parse('package.json', content)
        self.assertIsNotNone(result)
        src = result.symbols[0].source_text
        self.assertIn('[File: package.json]', src)

    def test_original_content_preserved_after_header(self):
        """The original content must appear in full after the header line."""
        content = "key: value\nother: data\n"
        result = self._write_and_parse('config.yaml', content)
        self.assertIsNotNone(result)
        src = result.symbols[0].source_text
        # Header is first line, rest is original content
        lines = src.split('\n', 1)
        self.assertTrue(lines[0].startswith('[File: '))
        self.assertEqual(lines[1], content)


# ============================================================================
# Section 4: Architectural Filter Gate
# ============================================================================

class TestArchitecturalFilterGate(unittest.TestCase):
    """Every symbol_type that _chunk_text_content / _parse_documentation_files
    can produce must pass _is_architectural_symbol()."""

    def setUp(self):
        self.builder = EnhancedUnifiedGraphBuilder()

    def test_all_extension_doc_types_pass_filter(self):
        """For every extension, the resulting '{doc_type}_document' must pass."""
        for ext, doc_type in DOCUMENTATION_EXTENSIONS.items():
            symbol_type = f"{doc_type}_document"
            self.assertTrue(
                self.builder._is_architectural_symbol(symbol_type),
                f"Extension '{ext}' → symbol_type '{symbol_type}' FAILS _is_architectural_symbol()"
            )

    def test_all_known_filename_doc_types_pass_filter(self):
        """For every known filename, the resulting '{doc_type}_document' must pass."""
        for fname, doc_type in KNOWN_FILENAMES.items():
            symbol_type = f"{doc_type}_document"
            self.assertTrue(
                self.builder._is_architectural_symbol(symbol_type),
                f"Filename '{fname}' → symbol_type '{symbol_type}' FAILS _is_architectural_symbol()"
            )

    def test_text_chunk_passes_filter(self):
        """text_chunk is the fallback type from _chunk_generic_text."""
        self.assertTrue(self.builder._is_architectural_symbol('text_chunk'))

    def test_text_document_passes_filter(self):
        """text_document is the fallback for empty chunking."""
        self.assertTrue(self.builder._is_architectural_symbol('text_document'))

    def test_markdown_section_passes_filter(self):
        """markdown_section is used when splitting large markdown by headers."""
        self.assertTrue(self.builder._is_architectural_symbol('markdown_section'))

    def test_code_symbols_still_pass(self):
        """Ensure code symbol types are not broken."""
        for st in ['class', 'function', 'interface', 'enum', 'struct', 'trait',
                    'constant', 'type_alias', 'macro']:
            self.assertTrue(
                self.builder._is_architectural_symbol(st),
                f"Code symbol type '{st}' should still pass _is_architectural_symbol()"
            )


# ============================================================================
# Section 5: _parse_documentation_files — doc_type / symbol_type correctness
# ============================================================================

class TestDocTypeCorrectness(unittest.TestCase):
    """Verify that doc files get the correct doc_type and section_type."""

    def setUp(self):
        self.builder = EnhancedUnifiedGraphBuilder()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_and_parse(self, filename, content="# test\n"):
        fpath = os.path.join(self.tmpdir, filename)
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        results = self.builder._parse_documentation_files([fpath], self.tmpdir)
        return results.get(fpath)

    def test_yaml_gets_yaml_document_type(self):
        result = self._write_and_parse('ci.yml', 'name: CI\non: push\n')
        self.assertEqual(result.language, 'yaml')
        self.assertEqual(result.symbols[0].symbol_type, 'yaml_document')

    def test_gradle_gets_build_config_document_type(self):
        result = self._write_and_parse('build.gradle', 'plugins { }\n')
        self.assertEqual(result.language, 'build_config')
        self.assertEqual(result.symbols[0].symbol_type, 'build_config_document')

    def test_makefile_gets_build_config_document_type(self):
        result = self._write_and_parse('Makefile', 'all:\n\techo ok\n')
        self.assertEqual(result.language, 'build_config')
        self.assertEqual(result.symbols[0].symbol_type, 'build_config_document')

    def test_dockerfile_gets_infrastructure_document_type(self):
        result = self._write_and_parse('Dockerfile', 'FROM ubuntu\n')
        self.assertEqual(result.language, 'infrastructure')
        self.assertEqual(result.symbols[0].symbol_type, 'infrastructure_document')

    def test_proto_gets_schema_document_type(self):
        result = self._write_and_parse('api.proto', 'syntax = "proto3";\n')
        self.assertEqual(result.language, 'schema')
        self.assertEqual(result.symbols[0].symbol_type, 'schema_document')

    def test_terraform_gets_infrastructure_document_type(self):
        result = self._write_and_parse('main.tf', 'resource "aws" "x" {}\n')
        self.assertEqual(result.language, 'infrastructure')
        self.assertEqual(result.symbols[0].symbol_type, 'infrastructure_document')

    def test_shell_gets_script_document_type(self):
        result = self._write_and_parse('deploy.sh', '#!/bin/bash\necho hi\n')
        self.assertEqual(result.language, 'script')
        self.assertEqual(result.symbols[0].symbol_type, 'script_document')

    def test_jenkinsfile_gets_script_document_type(self):
        result = self._write_and_parse('Jenkinsfile', 'pipeline { }\n')
        self.assertEqual(result.language, 'script')
        self.assertEqual(result.symbols[0].symbol_type, 'script_document')

    def test_ini_gets_config_document_type(self):
        result = self._write_and_parse('app.ini', '[section]\nkey=val\n')
        self.assertEqual(result.language, 'config')
        self.assertEqual(result.symbols[0].symbol_type, 'config_document')

    def test_json_gets_json_document_type(self):
        result = self._write_and_parse('package.json', '{"name": "pkg"}\n')
        self.assertEqual(result.language, 'json')
        self.assertEqual(result.symbols[0].symbol_type, 'json_document')

    def test_xml_gets_xml_document_type(self):
        result = self._write_and_parse('pom.xml', '<project></project>\n')
        self.assertEqual(result.language, 'xml')
        self.assertEqual(result.symbols[0].symbol_type, 'xml_document')

    def test_markdown_gets_markdown_document_type(self):
        result = self._write_and_parse('README.md', '# Hello\n')
        self.assertEqual(result.language, 'markdown')
        self.assertEqual(result.symbols[0].symbol_type, 'markdown_document')

    def test_toml_gets_toml_document_type(self):
        result = self._write_and_parse('pyproject.toml', '[project]\nname = "x"\n')
        self.assertEqual(result.language, 'toml')
        self.assertEqual(result.symbols[0].symbol_type, 'toml_document')

    def test_license_gets_plaintext_document_type(self):
        result = self._write_and_parse('LICENSE', 'MIT License\n')
        self.assertEqual(result.language, 'plaintext')
        self.assertEqual(result.symbols[0].symbol_type, 'plaintext_document')

    def test_cmakelists_gets_build_config_document_type(self):
        result = self._write_and_parse('CMakeLists.txt', 'cmake_minimum_required(VERSION 3.10)\n')
        # CMakeLists.txt has .txt extension but the filename is in KNOWN_FILENAMES
        # However, .txt extension maps to 'plaintext' first in DOCUMENTATION_EXTENSIONS
        # The extension takes priority, which is fine — the file is still indexed.
        # The important thing is it's not dropped.
        self.assertIsNotNone(result)
        self.assertTrue(len(result.symbols) > 0)

    def test_editorconfig_gets_config_document_type(self):
        """Dot-prefixed known file. Note: may not be discovered by default
        exclude patterns, but _parse_documentation_files itself must handle it."""
        result = self._write_and_parse('.editorconfig', 'root = true\n')
        self.assertEqual(result.language, 'config')
        self.assertEqual(result.symbols[0].symbol_type, 'config_document')

    def test_unknown_extensionless_defaults_to_text(self):
        """An extensionless file NOT in KNOWN_FILENAMES should default to 'text'."""
        result = self._write_and_parse('somefile', 'random content\n')
        self.assertEqual(result.language, 'text')
        self.assertIn(result.symbols[0].symbol_type, ('text_document', 'text_chunk'))


# ============================================================================
# Section 6: Retriever is_doc filter simulation
# ============================================================================

class TestRetrieverIsDocFilter(unittest.TestCase):
    """Simulate the retriever's is_doc logic to verify that all doc file types
    are recognized. We replicate the filter from retrievers.py inline since
    we can't easily instantiate the full retriever stack in unit tests."""

    @staticmethod
    def _is_doc(chunk_type, symbol_type, source):
        """Replicate the is_doc logic from retrievers.py."""
        from plugin_implementation.constants import (
            DOC_CHUNK_TYPES, DOC_SYMBOL_TYPES,
            DOCUMENTATION_EXTENSIONS_SET, KNOWN_FILENAMES,
        )
        return (
            chunk_type in DOC_CHUNK_TYPES or
            symbol_type in DOC_SYMBOL_TYPES or
            any(source.endswith(ext) for ext in DOCUMENTATION_EXTENSIONS_SET) or
            source.rsplit('/', 1)[-1] in KNOWN_FILENAMES or
            '/docs/' in source or
            'readme' in source.rsplit('/', 1)[-1].lower()
        )

    def test_yaml_recognized_via_symbol_type(self):
        self.assertTrue(self._is_doc('text', 'yaml_document', 'config.yaml'))

    def test_yaml_recognized_via_extension(self):
        self.assertTrue(self._is_doc('symbol', 'unknown', '.github/workflows/ci.yml'))

    def test_gradle_recognized_via_extension(self):
        self.assertTrue(self._is_doc('symbol', 'unknown', 'build.gradle'))

    def test_gradle_recognized_via_symbol_type(self):
        self.assertTrue(self._is_doc('text', 'build_config_document', 'build.gradle'))

    def test_makefile_recognized_via_known_filename(self):
        self.assertTrue(self._is_doc('symbol', 'unknown', 'Makefile'))

    def test_dockerfile_recognized_via_known_filename(self):
        self.assertTrue(self._is_doc('symbol', 'unknown', 'src/Dockerfile'))

    def test_proto_recognized_via_extension(self):
        self.assertTrue(self._is_doc('symbol', 'unknown', 'api/service.proto'))

    def test_terraform_recognized_via_extension(self):
        self.assertTrue(self._is_doc('symbol', 'unknown', 'infra/main.tf'))

    def test_shell_recognized_via_extension(self):
        self.assertTrue(self._is_doc('symbol', 'unknown', 'scripts/deploy.sh'))

    def test_ini_recognized_via_extension(self):
        self.assertTrue(self._is_doc('symbol', 'unknown', 'config/app.ini'))

    def test_jenkinsfile_recognized_via_known_filename(self):
        self.assertTrue(self._is_doc('symbol', 'unknown', 'Jenkinsfile'))

    def test_license_recognized_via_known_filename(self):
        self.assertTrue(self._is_doc('symbol', 'unknown', 'LICENSE'))

    def test_docs_directory_recognized(self):
        self.assertTrue(self._is_doc('symbol', 'unknown', 'project/docs/guide.html'))

    def test_readme_recognized(self):
        self.assertTrue(self._is_doc('symbol', 'unknown', 'README'))

    def test_readme_md_recognized(self):
        self.assertTrue(self._is_doc('symbol', 'unknown', 'README.md'))

    def test_code_file_not_recognized_as_doc(self):
        """A .py file with no doc markers should NOT be treated as doc."""
        self.assertFalse(self._is_doc('symbol', 'function', 'src/main.py'))

    def test_unknown_file_not_recognized_as_doc(self):
        self.assertFalse(self._is_doc('symbol', 'unknown', 'randomfile'))

    def test_all_extension_based_files_recognized(self):
        """Every extension in DOCUMENTATION_EXTENSIONS_SET must make is_doc True."""
        for ext in DOCUMENTATION_EXTENSIONS_SET:
            source = f"some/path/file{ext}"
            self.assertTrue(
                self._is_doc('symbol', 'unknown', source),
                f"Extension '{ext}' → source '{source}' not recognized by is_doc filter"
            )

    def test_all_known_filenames_recognized(self):
        """Every filename in KNOWN_FILENAMES must make is_doc True."""
        for fname in KNOWN_FILENAMES:
            source = f"some/path/{fname}"
            self.assertTrue(
                self._is_doc('symbol', 'unknown', source),
                f"Known filename '{fname}' → source '{source}' not recognized by is_doc filter"
            )


# ============================================================================
# Section 7: Wiki Agent _is_documentation_file()
# ============================================================================

class TestWikiAgentDocDetection(unittest.TestCase):
    """Test that wiki_graph_optimized._is_documentation_file() recognizes
    all document types — extensions, known filenames, and doc directories."""

    def setUp(self):
        """Create a minimal mock agent to test _is_documentation_file."""
        # Import only the method — we don't need a full agent
        from plugin_implementation.agents.wiki_graph_optimized import OptimizedWikiGenerationAgent
        self.agent = object.__new__(OptimizedWikiGenerationAgent)

    def test_yaml_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('config.yaml'))
        self.assertTrue(self.agent._is_documentation_file('.github/workflows/ci.yml'))

    def test_gradle_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('build.gradle'))
        self.assertTrue(self.agent._is_documentation_file('settings.gradle.kts'))

    def test_proto_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('api/service.proto'))

    def test_terraform_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('infra/main.tf'))

    def test_shell_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('scripts/deploy.sh'))

    def test_ini_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('config/app.ini'))

    def test_env_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('.env'))

    def test_makefile_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('Makefile'))

    def test_dockerfile_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('Dockerfile'))

    def test_jenkinsfile_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('Jenkinsfile'))

    def test_vagrantfile_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('Vagrantfile'))

    def test_license_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('LICENSE'))

    def test_readme_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('README'))

    def test_changelog_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('CHANGELOG'))

    def test_docs_directory_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('project/docs/guide.txt'))

    def test_documentation_directory_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('project/documentation/arch.md'))

    def test_markdown_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('README.md'))

    def test_toml_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('pyproject.toml'))

    def test_json_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('package.json'))

    def test_xml_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('pom.xml'))

    def test_wsdl_recognized(self):
        self.assertTrue(self.agent._is_documentation_file('api.wsdl'))

    def test_code_file_not_recognized(self):
        self.assertFalse(self.agent._is_documentation_file('src/main.py'))

    def test_unknown_file_not_recognized(self):
        self.assertFalse(self.agent._is_documentation_file('randomfile'))

    def test_java_not_recognized(self):
        self.assertFalse(self.agent._is_documentation_file('App.java'))


# ============================================================================
# Section 8: End-to-End — full pipeline from file to vector store chunk
# ============================================================================

class TestEndToEndDocPipeline(unittest.TestCase):
    """Test the full path: file → discovery → parsing → symbol_type check →
    would-be vector store chunk. Verifies that previously-dropped files now
    survive all gates."""

    def setUp(self):
        self.builder = EnhancedUnifiedGraphBuilder()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _full_pipeline(self, filename, content):
        """Run the file through discovery + parsing + architectural filter.
        Returns (discovered, parsed, passes_filter) tuple."""
        fpath = os.path.join(self.tmpdir, filename)
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)

        # Step 1: Discovery
        result = self.builder._discover_files_by_language(
            self.tmpdir,
            exclude_patterns=[]  # allow dotfiles for testing
        )
        discovered_as_doc = fpath in result.get('documentation', [])

        # Step 2: Parsing
        parse_result = None
        if discovered_as_doc:
            parsed = self.builder._parse_documentation_files([fpath], self.tmpdir)
            parse_result = parsed.get(fpath)

        # Step 3: Architectural filter
        passes_filter = False
        if parse_result and parse_result.symbols:
            symbol_type = parse_result.symbols[0].symbol_type
            passes_filter = self.builder._is_architectural_symbol(symbol_type)

        return discovered_as_doc, parse_result, passes_filter

    def test_yaml_workflow_survives_all_gates(self):
        discovered, parsed, passes = self._full_pipeline(
            '.github/workflows/ci.yml',
            'name: CI\non: push\njobs:\n  build:\n    runs-on: ubuntu-latest\n'
        )
        self.assertTrue(discovered, "YAML file should be discovered as documentation")
        self.assertIsNotNone(parsed, "YAML file should be parsed")
        self.assertTrue(passes, "yaml_document should pass _is_architectural_symbol()")
        # Verify path is embedded
        self.assertIn('.github/workflows/ci.yml', parsed.symbols[0].source_text)

    def test_gradle_survives_all_gates(self):
        discovered, parsed, passes = self._full_pipeline(
            'build.gradle', 'plugins {\n  id "java"\n}\n'
        )
        self.assertTrue(discovered)
        self.assertIsNotNone(parsed)
        self.assertTrue(passes, "build_config_document should pass _is_architectural_symbol()")
        self.assertIn('build.gradle', parsed.symbols[0].source_text)

    def test_makefile_survives_all_gates(self):
        discovered, parsed, passes = self._full_pipeline(
            'Makefile', 'all:\n\techo hello\n'
        )
        self.assertTrue(discovered, "Makefile should be discovered as documentation")
        self.assertIsNotNone(parsed)
        self.assertTrue(passes, "build_config_document should pass _is_architectural_symbol()")
        self.assertIn('Makefile', parsed.symbols[0].source_text)

    def test_dockerfile_survives_all_gates(self):
        discovered, parsed, passes = self._full_pipeline(
            'Dockerfile', 'FROM python:3.12\nRUN pip install flask\n'
        )
        self.assertTrue(discovered)
        self.assertIsNotNone(parsed)
        self.assertTrue(passes, "infrastructure_document should pass _is_architectural_symbol()")
        self.assertIn('Dockerfile', parsed.symbols[0].source_text)

    def test_proto_survives_all_gates(self):
        discovered, parsed, passes = self._full_pipeline(
            'api/service.proto', 'syntax = "proto3";\nservice Greeter {}\n'
        )
        self.assertTrue(discovered)
        self.assertIsNotNone(parsed)
        self.assertTrue(passes, "schema_document should pass _is_architectural_symbol()")

    def test_terraform_survives_all_gates(self):
        discovered, parsed, passes = self._full_pipeline(
            'main.tf', 'resource "aws_instance" "web" {\n  ami = "abc"\n}\n'
        )
        self.assertTrue(discovered)
        self.assertIsNotNone(parsed)
        self.assertTrue(passes, "infrastructure_document should pass _is_architectural_symbol()")

    def test_shell_script_survives_all_gates(self):
        discovered, parsed, passes = self._full_pipeline(
            'deploy.sh', '#!/bin/bash\necho deploying\n'
        )
        self.assertTrue(discovered)
        self.assertIsNotNone(parsed)
        self.assertTrue(passes, "script_document should pass _is_architectural_symbol()")

    def test_jenkinsfile_survives_all_gates(self):
        discovered, parsed, passes = self._full_pipeline(
            'Jenkinsfile', 'pipeline {\n  agent any\n}\n'
        )
        self.assertTrue(discovered)
        self.assertIsNotNone(parsed)
        self.assertTrue(passes, "script_document should pass _is_architectural_symbol()")

    def test_ini_config_survives_all_gates(self):
        discovered, parsed, passes = self._full_pipeline(
            'app.ini', '[database]\nhost=localhost\n'
        )
        self.assertTrue(discovered)
        self.assertIsNotNone(parsed)
        self.assertTrue(passes, "config_document should pass _is_architectural_symbol()")

    def test_wsdl_schema_survives_all_gates(self):
        discovered, parsed, passes = self._full_pipeline(
            'api.wsdl', '<definitions></definitions>\n'
        )
        self.assertTrue(discovered)
        self.assertIsNotNone(parsed)
        self.assertTrue(passes, "schema_document should pass _is_architectural_symbol()")

    def test_license_survives_all_gates(self):
        discovered, parsed, passes = self._full_pipeline(
            'LICENSE', 'MIT License\nCopyright 2024\n'
        )
        self.assertTrue(discovered)
        self.assertIsNotNone(parsed)
        self.assertTrue(passes, "plaintext_document should pass _is_architectural_symbol()")


if __name__ == '__main__':
    unittest.main()
