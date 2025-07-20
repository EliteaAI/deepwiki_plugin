"""
Enhanced Unified Graph Builder - Single System for Parsing, Chunking, and Graph Building

This module implements the unified architecture that eliminates double parsing by combining:
1. Rich parsers (Java, Python) for comprehensive analysis 
2. Basic tree-sitter parsers (14+ languages) for symbol-level parsing
3. Multi-tier code_graph building (comprehensive + basic relationships)
4. Symbol-level chunking (no character splitting)

Replaces both GraphAwareCodeSplitter and separate UnifiedGraphBuilder usage.
"""

import asyncio
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

import networkx as nx
# Document generation
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Import the code splitter for basic parsing logic
from alita_sdk.tools.wiki_toolkit.index.code_splitter import GraphAwareCodeSplitter
# Rich parsers (existing)
from alita_sdk.tools.wiki_toolkit.parsers.base_parser import (
    ParseResult
)
from alita_sdk.tools.wiki_toolkit.parsers.enhanced_python_parser import PythonParser
from alita_sdk.tools.wiki_toolkit.parsers.java_visitor_parser import JavaVisitorParser

# Tree-sitter for basic parsing

logger = logging.getLogger(__name__)


@dataclass
class BasicSymbol:
    """Basic symbol extracted from tree-sitter parsing"""
    name: str
    symbol_type: str  # AST node type
    start_line: int
    end_line: int
    file_path: str
    language: str
    source_text: str = ""
    imports: Set[str] = field(default_factory=set)
    calls: Set[str] = field(default_factory=set)
    parent_symbol: Optional[str] = None
    docstring: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None


@dataclass
class BasicRelationship:
    """Basic relationship extracted from tree-sitter parsing"""
    source_symbol: str
    target_symbol: str
    relationship_type: str  # 'imports', 'calls', 'references'
    source_file: str
    target_file: Optional[str] = None


@dataclass
class BasicParseResult:
    """Result of basic tree-sitter parsing"""
    file_path: str
    language: str
    symbols: List[BasicSymbol]
    relationships: List[BasicRelationship]
    parse_level: str = 'basic'
    imports: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class UnifiedAnalysis:
    """Unified analysis result containing both chunks and graphs"""
    documents: List[Document] = field(default_factory=list)           # Symbol-level chunks
    unified_graph: Optional[nx.MultiDiGraph] = None                   # Multi-tier code_graph with preserved relationships
    cross_language_relationships: List[Any] = field(default_factory=list)
    language_stats: Dict[str, Any] = field(default_factory=dict)
    rich_languages: Set[str] = field(default_factory=set)             # Languages with rich analysis
    basic_languages: Set[str] = field(default_factory=set)            # Languages with basic analysis


class EnhancedUnifiedGraphBuilder:
    """
    Single system for parsing, chunking, and comprehensive code_graph building.
    
    Eliminates double parsing by using appropriate parser tier for each language
    and generating both chunk documents and relationship graphs from single parse.
    """
    
    # Code Splitter language mappings (copied from original)
    SUPPORTED_LANGUAGES = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.cs': 'c_sharp',
        '.cpp': 'cpp',
        '.c': 'c',
        '.rb': 'ruby',
        '.php': 'php',
        '.rs': 'rust',
        '.kt': 'kotlin',
        '.scala': 'scala'
    }
    
    # Documentation file extensions
    DOCUMENTATION_EXTENSIONS = {
        '.md': 'markdown',
        '.rst': 'restructuredtext', 
        '.txt': 'plaintext',
        '.doc': 'document',
        '.docx': 'document',
        '.pdf': 'pdf',
        '.html': 'html',
        '.htm': 'html',
        '.xml': 'xml',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.ini': 'config',
        '.cfg': 'config',
        '.conf': 'config',
        '.properties': 'config',
        '.env': 'config'
    }
    
    def __init__(self, max_workers: int = 4, debug_mode: bool = False):
        # Tier 1: Rich parsers for comprehensive analysis
        self.rich_parsers = {
            'java': JavaVisitorParser(),
            'python': PythonParser()
            # TODO: Gradually add TypeScript, Go, C#, etc. in rich parser style
        }
        
        # Configuration
        self.max_workers = max_workers
        self.debug_mode = debug_mode
        
        # Initialize code splitter for basic parsing (provides tree-sitter parsers)
        self.code_splitter = GraphAwareCodeSplitter()
        
        # Track which languages use which analysis level
        self.rich_languages = set(self.rich_parsers.keys())
        
        # Basic languages are those supported by code splitter but not in rich parsers
        # Based on actual code splitter symbol extraction implementation
        # Only include languages that have working target_nodes and _get_symbol_name implementation
        supported_basic_languages = [
            'javascript', 'typescript', 'go', 'c_sharp', 'rust', 'ruby', 'kotlin', 'scala'
        ]
        
        self.basic_languages = set()
        for lang in supported_basic_languages:
            if lang in self.code_splitter.parsers and lang not in self.rich_languages:
                self.basic_languages.add(lang)
        
        # Note: C, C++, and PHP are in SUPPORTED_LANGUAGES for file detection
        # but do NOT have symbol extraction implementation in the code splitter yet
        # They would fall back to regex-based processing in the code splitter
        
        # File-level imports storage (for basic parsers, like code splitter)
        self.file_imports = {}  # file_path -> set of file-level imports
        
        # Store last analysis result for statistics
        self.last_analysis: Optional[UnifiedAnalysis] = None
        
        logger.info(f"Enhanced Unified Graph Builder initialized:")
        logger.info(f"  Rich parsers: {list(self.rich_languages)}")
        logger.info(f"  Basic parsers: {list(self.basic_languages)}")
        logger.info(f"  Code splitter parsers available: {list(self.code_splitter.parsers.keys())}")
        logger.info(f"  Languages without symbol extraction: C, C++, PHP (use regex fallback)")
    
    def _initialize_basic_parsers(self):
        """
        This method is no longer needed since we use the code splitter directly.
        The code splitter handles tree-sitter parser initialization.
        """
        pass
    
    def analyze_repository(self, repo_path: str, 
                          include_patterns: Optional[List[str]] = None,
                          exclude_patterns: Optional[List[str]] = None) -> UnifiedAnalysis:
        """
        Single pass analysis: parse once, generate chunks + multi-tier code_graph
        
        Args:
            repo_path: Path to the repository root
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            
        Returns:
            Unified analysis with documents and code_graph from single parsing
        """
        logger.info(f"Starting unified repository analysis: {repo_path}")
        
        # Phase 1: Discover files by language
        files_by_language = self._discover_files_by_language(repo_path, include_patterns, exclude_patterns)
        
        # Phase 2: Single parse with appropriate parser tier
        parse_results = {}
        
        for language, file_paths in files_by_language.items():
            if language == 'unknown':
                continue
                
            logger.info(f"Parsing {len(file_paths)} {language} files")
            
            if language == 'documentation':
                # Process documentation files with text chunking
                doc_results = self._parse_documentation_files(file_paths)
                parse_results.update(doc_results)
            elif language in self.rich_parsers:
                # Use rich parser for comprehensive analysis
                rich_results = self._parse_with_rich_parser(language, file_paths)
                parse_results.update(rich_results)
            elif language in self.basic_languages:
                # Use basic tree-sitter parser (code splitter style)
                basic_results = self._parse_with_basic_parser(language, file_paths)
                parse_results.update(basic_results)
            else:
                # Language not supported for symbol-level parsing yet
                # (e.g., C, C++, PHP - they have tree-sitter parsers but no symbol extraction)
                logger.warning(f"Language {language} has tree-sitter parser but no symbol extraction implementation yet")
                logger.info(f"Files will be skipped: {file_paths}")
                # These files would fall back to regex-based processing in the original code splitter
        
        # Phase 3: Build multi-tier unified code_graph
        unified_graph = self._build_multi_tier_graph(parse_results)
        
        # Phase 4: Generate symbol-level chunk documents (NO character splitting)
        chunk_documents = self._generate_symbol_chunks(parse_results)
        
        # Phase 5: Detect cross-language relationships (from original UnifiedGraphBuilder)
        cross_lang_rels = self._detect_cross_language_relationships(files_by_language, parse_results)
        
        # Phase 6: Generate statistics
        language_stats = self._generate_language_stats(parse_results)
        
        analysis = UnifiedAnalysis(
            documents=chunk_documents,
            unified_graph=unified_graph,
            cross_language_relationships=cross_lang_rels,
            language_stats=language_stats,
            rich_languages=self.rich_languages,
            basic_languages=self.basic_languages
        )
        
        # Store for statistics access
        self.last_analysis = analysis
        
        # Store analysis for statistics and debugging
        self._last_analysis = analysis
        
        logger.info(f"Unified analysis complete:")
        logger.info(f"  Documents generated: {len(chunk_documents)}")
        logger.info(f"  Graph nodes: {unified_graph.number_of_nodes() if unified_graph else 0}")
        logger.info(f"  Graph edges: {unified_graph.number_of_edges() if unified_graph else 0}")
        logger.info(f"  Cross-language relationships: {len(cross_lang_rels)}")
        
        return analysis
    
    def _discover_files_by_language(self, repo_path: str,
                                   include_patterns: Optional[List[str]] = None,
                                   exclude_patterns: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Discover and categorize files by programming language"""
        files_by_language = defaultdict(list)
        
        # Default exclude patterns (from code splitter)
        if exclude_patterns is None:
            exclude_patterns = [
                '**/node_modules/**', '**/build/**', '**/dist/**', '**/target/**',
                '**/.git/**', '**/__pycache__/**', '**/.pytest_cache/**',
                '**/.*', '**/*.class', '**/*.jar', '**/*.war'
            ]
        
        repo_path = Path(repo_path)
        
        # Walk through all files in the repository
        for file_path in repo_path.rglob('*'):
            if file_path.is_file():
                # Check exclusions
                if self._should_exclude_file(str(file_path), exclude_patterns):
                    continue
                
                # Check inclusions
                if include_patterns and not self._should_include_file(str(file_path), include_patterns):
                    continue
                
                # Categorize by language based on file extension
                file_extension = file_path.suffix.lower()
                language = self.SUPPORTED_LANGUAGES.get(file_extension)
                doc_type = self.DOCUMENTATION_EXTENSIONS.get(file_extension)
                
                if language:
                    files_by_language[language].append(str(file_path))
                elif doc_type:
                    files_by_language['documentation'].append(str(file_path))
                else:
                    files_by_language['unknown'].append(str(file_path))
        
        return dict(files_by_language)
    
    def _should_exclude_file(self, file_path: str, exclude_patterns: List[str]) -> bool:
        """Check if a file should be excluded based on patterns"""
        from fnmatch import fnmatch
        
        for pattern in exclude_patterns:
            if fnmatch(file_path, pattern):
                return True
        return False
    
    def _should_include_file(self, file_path: str, include_patterns: List[str]) -> bool:
        """Check if a file should be included based on patterns"""
        from fnmatch import fnmatch
        
        for pattern in include_patterns:
            if fnmatch(file_path, pattern):
                return True
        return False
    
    def _parse_with_rich_parser(self, language: str, file_paths: List[str]) -> Dict[str, ParseResult]:
        """
        Parse files using rich parser for comprehensive analysis.
        
        CRITICAL: Uses the magic 'parse_multiple_files' method which is the KEY to cross-file
        relationship detection for both Java and Python rich parsers. This method:
        - Builds complete symbol table across ALL files first
        - Resolves cross-file inheritance, implements, imports relationships  
        - Provides the main "magic" for comprehensive repository-wide analysis
        
        This is what distinguishes rich parsers from basic tree-sitter parsing!
        """
        parser = self.rich_parsers[language]
        
        if hasattr(parser, 'parse_multiple_files'):
            # 🎯 THE MAGIC: Multi-file parsing for cross-file relationship resolution
            # This is what enables comprehensive inheritance, implements, and import analysis
            # across the entire repository for Java and Python
            return parser.parse_multiple_files(file_paths, max_workers=self.max_workers)
        else:
            # Fallback to individual file parsing (should not happen for Java/Python)
            logger.warning(f"Rich parser for {language} missing parse_multiple_files method, falling back to individual parsing")
            results = {}
            for file_path in file_paths:
                try:
                    result = parser.parse_file(file_path)
                    results[file_path] = result
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path} with rich parser: {e}")
            return results
    
    def _parse_with_basic_parser(self, language: str, file_paths: List[str]) -> Dict[str, BasicParseResult]:
        """
        Parse files using basic tree-sitter parser (code splitter style).
        Uses the exact same logic as the Code Splitter to ensure consistency.
        """
        results = {}
        
        for file_path in file_paths:
            try:
                symbols, relationships = self._parse_with_basic_parser_single_file(file_path, language)
                
                # Convert to BasicParseResult format by converting dict symbols to BasicSymbol objects
                basic_symbols = []
                basic_relationships = []
                
                for symbol_dict in symbols:
                    basic_symbol = BasicSymbol(
                        name=symbol_dict['name'],
                        symbol_type=symbol_dict['type'],
                        start_line=symbol_dict['start_line'],
                        end_line=symbol_dict['end_line'],
                        file_path=symbol_dict['file_path'],
                        language=symbol_dict['language'],
                        source_text=symbol_dict['content'],
                        imports=set(symbol_dict.get('imports', [])),
                        calls=set(symbol_dict.get('calls', [])),
                        docstring=symbol_dict.get('docstring'),
                        parameters=symbol_dict.get('parameters', []),
                        return_type=symbol_dict.get('return_type')
                    )
                    basic_symbols.append(basic_symbol)
                
                for rel_dict in relationships:
                    basic_rel = BasicRelationship(
                        source_symbol=rel_dict['source'],
                        target_symbol=rel_dict['target'],
                        relationship_type=rel_dict['type'],
                        source_file=rel_dict['source_file'],
                        target_file=rel_dict.get('target_file')
                    )
                    basic_relationships.append(basic_rel)
                
                # Convert to BasicParseResult format
                result = BasicParseResult(
                    file_path=file_path,
                    language=language,
                    symbols=basic_symbols,
                    relationships=basic_relationships,
                    parse_level='basic',
                    imports=list(self.code_splitter.file_imports.get(file_path, []))
                )
                results[file_path] = result
                
            except Exception as e:
                logger.warning(f"Failed to parse {file_path} with basic parser: {e}")
        
        return results
    
    def _parse_with_basic_parser_single_file(self, file_path: str, language: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Parse single file using basic Tree-sitter parser for symbol extraction and basic code_graph building.
        Uses the exact same logic as the Code Splitter to ensure consistency.
        """
        if language not in self.code_splitter.SUPPORTED_LANGUAGES.values():
            logger.warning(f"Language {language} not supported by basic parser")
            return [], []
        
        if language not in self.code_splitter.parsers:
            logger.warning(f"No Tree-sitter parser available for {language}")
            return [], []
        
        try:
            # Clear the code splitter's state for this parsing session
            self.code_splitter.symbol_table.clear()
            self.code_splitter.file_imports.clear()
            
            # Use the code splitter's _process_code_file method directly
            rel_path = os.path.relpath(file_path, os.getcwd())
            documents = self.code_splitter._process_code_file(file_path, rel_path)
            
            symbols = []
            relationships = []
            
            # Convert documents to our symbol format
            for doc in documents:
                metadata = doc.metadata
                if metadata.get('chunk_type') == 'code' and 'symbol' in metadata:
                    symbol_dict = {
                        'name': metadata['symbol'],
                        'type': metadata.get('node_type', metadata.get('symbol_type', 'unknown')),
                        'start_line': metadata.get('start_line', 0),
                        'end_line': metadata.get('end_line', 0),
                        'file_path': file_path,
                        'language': language,
                        'content': doc.page_content,
                        'imports': metadata.get('imports', []),
                        'calls': metadata.get('calls', []),
                        'parameters': metadata.get('parameters', []),
                        'return_type': metadata.get('return_type'),
                        'docstring': metadata.get('docstring')
                    }
                    symbols.append(symbol_dict)
                    
                    # Create relationships for imports and calls
                    for imported in metadata.get('imports', []):
                        relationships.append({
                            'source': metadata['symbol'],
                            'target': imported,
                            'type': 'imports',
                            'source_file': file_path,
                            'target_file': None  # Will be resolved later
                        })
                    
                    for called in metadata.get('calls', []):
                        relationships.append({
                            'source': metadata['symbol'],
                            'target': called,
                            'type': 'calls',
                            'source_file': file_path,
                            'target_file': None  # Will be resolved later
                        })
            
            # Add file-level imports as relationships
            if file_path in self.code_splitter.file_imports:
                file_imports = self.code_splitter.file_imports[file_path]
                for imported in file_imports:
                    relationships.append({
                        'source': f"file:{Path(file_path).stem}",
                        'target': imported,
                        'type': 'file_imports',
                        'source_file': file_path,
                        'target_file': None
                    })
            
            return symbols, relationships
            
        except Exception as e:
            logger.error(f"Error parsing {file_path} with basic parser: {e}")
            return [], []
    
    def _build_multi_tier_graph(self, parse_results: Dict[str, Union[ParseResult, BasicParseResult]]) -> nx.MultiDiGraph:
        """Build unified code_graph with both comprehensive and basic relationship levels"""
        # Use MultiDiGraph to preserve multiple relationship types between same nodes
        graph = nx.MultiDiGraph()
        graph.graph['analysis_type'] = 'multi_tier'
        
        # Separate rich and basic parse results
        rich_results = {path: result for path, result in parse_results.items() 
                       if not hasattr(result, 'parse_level') or result.parse_level != 'basic'}
        basic_results = {path: result for path, result in parse_results.items() 
                        if hasattr(result, 'parse_level') and result.parse_level == 'basic'}
        
        logger.info(f"Building multi-tier code_graph: {len(rich_results)} rich, {len(basic_results)} basic")
        
        # Add rich language graphs (comprehensive relationships)
        # Group by language first to avoid processing the same language multiple times
        rich_languages_found = {}
        for file_path, result in rich_results.items():
            language = result.language
            if language not in rich_languages_found:
                rich_languages_found[language] = []
            rich_languages_found[language].append((file_path, result))
        
        # Process each language once
        for language, file_results in rich_languages_found.items():
            lang_results = {fp: res for fp, res in file_results}
            if lang_results:
                lang_graph = self._build_comprehensive_language_graph(lang_results, language)
                self._merge_graph(graph, lang_graph)
        
        # Add basic language graphs (code splitter style relationships)
        for language in self.basic_languages:
            lang_basic_results = {fp: res for fp, res in basic_results.items() if res.language == language}
            if lang_basic_results:
                basic_graph = self._build_basic_language_graph(lang_basic_results, language)
                self._merge_graph(graph, basic_graph)
        
        return graph
    
    def _build_comprehensive_language_graph(self, parse_results: Dict[str, ParseResult], language: str) -> nx.MultiDiGraph:
        """Build comprehensive code_graph for rich parsers with asyncio optimization"""
        import asyncio
        import time
        
        start_time = time.time()
        logger.info(f"Building comprehensive {language} code_graph for {len(parse_results)} files...")
        
        # Analyze potential symbol loss before processing
        symbol_analysis = self._analyze_symbol_loss(parse_results, language)
        
        # Use MultiDiGraph to preserve all relationship types
        graph = nx.MultiDiGraph()
        graph.graph['language'] = language
        graph.graph['analysis_level'] = 'comprehensive'
        
        # Run the async code_graph building - handle existing event loop
        try:
            # Try to get current running loop
            current_loop = asyncio.get_running_loop()
            # If there's a running loop, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_graph_build_in_new_loop, graph, parse_results, language, symbol_analysis)
                symbol_registry = future.result()
        except RuntimeError:
            # No running loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                symbol_registry = loop.run_until_complete(
                    self._build_graph_async(graph, parse_results, language, symbol_analysis)
                )
            finally:
                loop.close()
        
        build_time = time.time() - start_time
        logger.info(f"Comprehensive {language} code_graph built in {build_time:.2f}s: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        return graph
    
    def _run_graph_build_in_new_loop(self, graph, parse_results, language, symbol_analysis):
        """Run code_graph building in a new event loop (for thread execution)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._build_graph_async(graph, parse_results, language, symbol_analysis)
            )
        finally:
            loop.close()
    
    def _analyze_symbol_loss(self, parse_results: Dict[str, ParseResult], language: str = "python") -> Dict[str, Any]:
        """Analyze where symbols are being lost during processing"""
        analysis = {
            'total_expected': 0,
            'valid_symbols': 0,  # Only count symbols that will actually be added
            'files_analyzed': 0,
            'symbols_per_file': {},
            'duplicate_definitions': {},  # Only true duplicates (same symbol defined multiple times)
            'empty_symbols': 0,
            'symbols_with_issues': []
        }
        
        # Track symbol definitions (not references/usage)
        symbol_definitions = {}  # symbol_name -> [(file_path, symbol_type)]
        processed_symbols = set()  # Track globally unique node IDs
        
        for file_path, result in parse_results.items():
            file_name = Path(file_path).stem
            symbol_count = len(result.symbols)
            valid_count = 0
            
            analysis['total_expected'] += symbol_count
            analysis['files_analyzed'] += 1
            analysis['symbols_per_file'][file_name] = symbol_count
            
            for symbol in result.symbols:
                if not symbol.name or symbol.name.strip() == "":
                    analysis['empty_symbols'] += 1
                    analysis['symbols_with_issues'].append(f"Empty symbol name in {file_name}")
                    continue
                
                # Create the same node_id as in _process_file_symbols
                node_id = f"{language}::{file_name}::{symbol.name}"
                
                # Only count each unique node_id once
                if node_id not in processed_symbols:
                    valid_count += 1
                    processed_symbols.add(node_id)
                    
                    # Track symbol definitions for duplicate detection
                    symbol_key = symbol.name
                    if symbol_key not in symbol_definitions:
                        symbol_definitions[symbol_key] = []
                    symbol_definitions[symbol_key].append((file_path, symbol.symbol_type.value if hasattr(symbol.symbol_type, 'value') else str(symbol.symbol_type)))
            
            analysis['symbols_per_file'][f"{file_name}_valid"] = valid_count
        
        analysis['valid_symbols'] = len(processed_symbols)
        
        # Identify true duplicates (same symbol defined in multiple files with same type)
        for symbol_name, definitions in symbol_definitions.items():
            if len(definitions) > 1:
                # Group by symbol type
                by_type = {}
                for file_path, symbol_type in definitions:
                    if symbol_type not in by_type:
                        by_type[symbol_type] = []
                    by_type[symbol_type].append(Path(file_path).stem)
                
                # Only report as duplicate if same type defined in multiple files
                for symbol_type, files in by_type.items():
                    if len(files) > 1:
                        analysis['duplicate_definitions'][f"{symbol_name}({symbol_type})"] = files
        
        logger.info(f"SYMBOL LOSS ANALYSIS:")
        logger.info(f"  Total expected symbols: {analysis['total_expected']}")
        logger.info(f"  Valid symbols (will be added): {analysis['valid_symbols']}")
        logger.info(f"  Files analyzed: {analysis['files_analyzed']}")
        logger.info(f"  Empty symbol names: {analysis['empty_symbols']}")
        logger.info(f"  True duplicate definitions: {len(analysis['duplicate_definitions'])}")
        
        if analysis['duplicate_definitions']:
            logger.warning(f"True duplicate definitions found: {list(analysis['duplicate_definitions'].items())[:5]}")
        
        return analysis

    async def _build_graph_async(self, graph: nx.MultiDiGraph, parse_results: Dict[str, ParseResult], 
                               language: str, symbol_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Async code_graph building with batched processing"""
        # Enhanced symbol registry for better cross-file resolution
        symbol_registry = {
            'by_name': {},           # symbol_name -> [list of node_ids]
            'by_qualified_name': {}, # file.symbol_name -> node_id  
            'by_full_path': {}       # file_path::symbol_name -> node_id
        }
        
        # Phase 1: Add all symbols as nodes (batched)
        logger.info(f"Phase 1: Adding symbols as nodes...")
        await self._add_symbols_batch(graph, parse_results, language, symbol_registry, symbol_analysis)
        
        # Phase 2: Add relationships as edges (batched)
        logger.info(f"Phase 2: Adding relationships as edges...")
        await self._add_relationships_batch(graph, parse_results, language, symbol_registry)
        
        return symbol_registry
    
    async def _add_symbols_batch(self, graph: nx.MultiDiGraph, parse_results: Dict[str, ParseResult], 
                                language: str, symbol_registry: Dict[str, Any], symbol_analysis: Dict[str, Any]):
        """Add symbols as nodes in batches with async processing"""
        batch_size = 50  # Process 50 files at a time
        file_items = list(parse_results.items())
        
        # Use valid symbols count from analysis
        total_symbols_expected = symbol_analysis['valid_symbols']
        symbols_added = 0
        
        logger.info(f"Expected to add {total_symbols_expected} symbols to code_graph")
        
        for i in range(0, len(file_items), batch_size):
            batch = file_items[i:i + batch_size]
            
            # Process batch of files
            tasks = []
            for file_path, result in batch:
                task = self._process_file_symbols(graph, file_path, result, language, symbol_registry)
                tasks.append(task)
            
            # Wait for batch to complete
            await asyncio.gather(*tasks)
            
            # Count symbols actually added
            symbols_added = graph.number_of_nodes()
            
            # Yield control to allow other operations
            await asyncio.sleep(0)
            
            if (i + batch_size) % 100 == 0:
                logger.info(f"Processed {min(i + batch_size, len(file_items))}/{len(file_items)} files (symbols)")
                logger.info(f"Symbols in code_graph so far: {symbols_added}/{total_symbols_expected}")
        
        final_symbols = graph.number_of_nodes()
        logger.info(f"SYMBOL SUMMARY: Expected {total_symbols_expected}, Added {final_symbols}, Loss: {total_symbols_expected - final_symbols}")
    
    async def _process_file_symbols(self, graph: nx.MultiDiGraph, file_path: str, result: ParseResult,
                                   language: str, symbol_registry: Dict[str, Any]):
        """Process symbols from a single file"""
        file_name = Path(file_path).stem
        
        for symbol in result.symbols:
            # Skip symbols with empty names
            if not symbol.name or symbol.name.strip() == "":
                continue
                
            node_id = f"{language}::{file_name}::{symbol.name}"
            
            # Only skip if this EXACT node already exists (same file + same symbol)
            # Do NOT skip symbols with same name from different files - those are legitimate!
            if graph.has_node(node_id):
                continue
            
            graph.add_node(node_id,
                         symbol=symbol,
                         file_path=file_path,
                         file_name=file_name,
                         language=language,
                         symbol_type=symbol.symbol_type.value if hasattr(symbol.symbol_type, 'value') else str(symbol.symbol_type),
                         symbol_name=symbol.name,
                         parent_symbol=getattr(symbol, 'parent_symbol', None),
                         analysis_level='comprehensive')
            
            # Register symbol in enhanced registry for better cross-file resolution
            # 1. By name (list of all instances)
            if symbol.name not in symbol_registry['by_name']:
                symbol_registry['by_name'][symbol.name] = []
            symbol_registry['by_name'][symbol.name].append(node_id)
            
            # 2. By qualified name (file.symbol)
            qualified_name = f"{file_name}.{symbol.name}"
            symbol_registry['by_qualified_name'][qualified_name] = node_id
            
            # 3. By full path (file_path::symbol)
            full_path_name = f"{file_path}::{symbol.name}"
            symbol_registry['by_full_path'][full_path_name] = node_id
        
        # Yield control
        await asyncio.sleep(0)
    
    async def _add_relationships_batch(self, graph: nx.MultiDiGraph, parse_results: Dict[str, ParseResult],
                                     language: str, symbol_registry: Dict[str, Any]):
        """Add relationships as edges in batches with async processing"""
        batch_size = 25  # Smaller batches for relationship processing (more complex)
        file_items = list(parse_results.items())
        
        total_relationships_expected = sum(len(result.relationships) for result in parse_results.values())
        relationships_attempted = 0
        relationships_added = 0
        relationships_failed = 0
        
        logger.info(f"Expected to add {total_relationships_expected} relationships to code_graph")
        
        for i in range(0, len(file_items), batch_size):
            batch = file_items[i:i + batch_size]
            
            # Process batch of files
            tasks = []
            for file_path, result in batch:
                task = self._process_file_relationships(graph, file_path, result, language, symbol_registry, parse_results)
                tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks)
            
            # Aggregate results from batch
            for result in batch_results:
                if isinstance(result, tuple) and len(result) == 3:
                    attempted, added, failed = result
                    relationships_attempted += attempted
                    relationships_added += added
                    relationships_failed += failed
                elif isinstance(result, dict):
                    relationships_attempted += result.get('attempted', 0)
                    relationships_added += result.get('added', 0)
                    relationships_failed += result.get('failed', 0)
            
            # Yield control to allow other operations
            await asyncio.sleep(0)
            
            if (i + batch_size) % 50 == 0:
                logger.info(f"Processed {min(i + batch_size, len(file_items))}/{len(file_items)} files (relationships)")
                logger.info(f"Relationships - Attempted: {relationships_attempted}, Added: {relationships_added}, Failed: {relationships_failed}")
        
        final_edges = graph.number_of_edges()
        missing_relationships = total_relationships_expected - final_edges
        
        logger.info(f"RELATIONSHIP SUMMARY: Expected {total_relationships_expected}, Attempted {relationships_attempted}, Added {final_edges}, Failed: {relationships_failed}")
        
        if missing_relationships > 0:
            logger.warning(f"⚠️ RELATIONSHIP LOSS DETECTED: {missing_relationships} relationships missing from final code_graph")
            logger.warning(f"   Counter shows {relationships_added} processed, but code_graph has {final_edges} edges")
            logger.warning(f"   This indicates {relationships_added - final_edges} relationships may have been merged/consolidated")
    
    async def _process_file_relationships(self, graph: nx.MultiDiGraph, file_path: str, result: ParseResult,
                                        language: str, symbol_registry: Dict[str, Any], parse_results: Dict[str, ParseResult]):
        """Process relationships from a single file with comprehensive source/target resolution"""
        file_name = Path(file_path).stem
        
        attempted = 0
        added = 0
        failed = 0
        
        for relationship in result.relationships:
            attempted += 1
            source_symbol = relationship.source_symbol
            target_symbol = relationship.target_symbol
            rel_type = relationship.relationship_type.value
            
            # Enhanced source node resolution with comprehensive fallback strategies
            source_node = await self._resolve_source_node_async(
                source_symbol, language, file_name, file_path, graph, symbol_registry, parse_results
            )
            
            # Enhanced target node resolution with comprehensive fallback strategies  
            target_node = await self._resolve_target_node_async(
                target_symbol, language, file_name, symbol_registry, graph, parse_results
            )
            
            # Comprehensive validation and node creation
            success = True
            
            # Validate source node
            if not source_node:
                if self.debug_mode:
                    print(f"❌ Failed to resolve source node: {source_symbol} in {file_name}")
                logger.debug(f"Source resolution failed: {source_symbol} in {file_name}")
                failed += 1
                success = False
            elif not graph.has_node(source_node):
                # Create missing source node
                if self.debug_mode:
                    print(f"⚠️ Creating missing source node: {source_node}")
                logger.debug(f"Creating missing source node: {source_node}")
                graph.add_node(source_node,
                             symbol_name=source_symbol.split('.')[-1],
                             file_path=file_path,
                             file_name=file_name,
                             language=language,
                             symbol_type="inferred",
                             analysis_level='comprehensive')
            
            # Validate target node
            if not target_node:
                if self.debug_mode:
                    print(f"❌ Failed to resolve target node: {target_symbol} in {file_name}")
                logger.debug(f"Target resolution failed: {target_symbol} in {file_name}")
                failed += 1
                success = False
            elif not graph.has_node(target_node):
                # Skip creating nodes for Java standard library references
                if self._is_java_standard_library_reference(target_symbol, target_node):
                    if self.debug_mode:
                        print(f"⏭️ Skipping Java standard library reference: {target_node}")
                    logger.debug(f"Skipping Java standard library reference: {target_node}")
                    continue
                    
                # Create missing target node with intelligent type inference
                if self.debug_mode:
                    print(f"⚠️ Creating missing target node: {target_node}")
                logger.debug(f"Creating missing target node: {target_node}")
                
                target_parts = target_node.split('::')
                target_name = target_parts[-1] if target_parts else target_symbol
                target_file_name = target_parts[1] if len(target_parts) > 1 else file_name
                
                # Infer target type based on relationship
                if rel_type == 'inheritance':
                    target_type = "class"
                elif rel_type == 'calls':
                    target_type = "function" if not '.' in target_symbol else "method"
                elif rel_type == 'imports':
                    target_type = "module" if not '.' in target_symbol else "class"
                else:
                    target_type = "unknown"
                
                graph.add_node(target_node,
                             symbol_name=target_name,
                             file_path=file_path,
                             file_name=target_file_name,
                             language=language,
                             symbol_type=target_type,
                             analysis_level='comprehensive')
            
            # Skip relationship if either node resolution failed
            if not success:
                continue
            
            # Add edge preserving all relationship types in MultiDiGraph
            try:
                target_file_name = target_node.split('::')[1] if '::' in target_node else file_name
                
                # In MultiDiGraph, we can have multiple edges with different relationships
                # No need for complex consolidation - just add the edge
                graph.add_edge(source_node, target_node,
                             relationship_type=rel_type,
                             source_file=file_name,
                             target_file=target_file_name,
                             analysis_level='comprehensive')
                
                if self.debug_mode:
                    print(f"✅ Added edge {source_node} --{rel_type}--> {target_node}")
                
                added += 1
                
            except Exception as e:
                failed += 1
                if self.debug_mode:
                    print(f"❌ Failed to add edge {source_node} --{rel_type}--> {target_node}: {e}")
                logger.error(f"Error adding relationship: {source_symbol} -{rel_type}-> {target_symbol} in {file_name}: {e}")
        
        # Add progress logging
        if self.debug_mode:
            print(f"📊 File {file_name}: {attempted} attempted, {added} added, {failed} failed")
        
        # Yield control
        await asyncio.sleep(0)
        
        return attempted, added, failed
    
    async def _resolve_target_node_async(self, target_symbol: str, language: str, file_name: str,
                                       symbol_registry: Dict[str, Any], graph: nx.MultiDiGraph,
                                       parse_results: Dict[str, ParseResult]) -> Optional[str]:
        """Enhanced async target node resolution with comprehensive fallback strategies"""
        
        # Strategy 1: Direct match in same file (highest priority for local references)
        direct_target = f"{language}::{file_name}::{target_symbol}"
        if graph.has_node(direct_target):
            return direct_target
        
        # Strategy 2: Handle self references (self.name -> name)
        if target_symbol.startswith('self.'):
            attribute_name = target_symbol[5:]  # Remove 'self.'
            self_target = f"{language}::{file_name}::{attribute_name}"
            if graph.has_node(self_target):
                return self_target
        
        # Strategy 3: Handle qualified names for imports (module.Class)
        if '.' in target_symbol:
            parts = target_symbol.split('.')
            
            # For imports like "module1.User", try to find in the module
            if len(parts) == 2:
                module_name, class_name = parts
                qualified_target = f"{language}::{module_name}::{class_name}"
                if graph.has_node(qualified_target):
                    return qualified_target
                    
                # Also try without module qualification
                unqualified_target = f"{language}::{file_name}::{class_name}"
                if graph.has_node(unqualified_target):
                    return unqualified_target
            
            # Try each part of the dotted name
            for i in range(len(parts)):
                partial_name = '.'.join(parts[i:])
                candidate = f"{language}::{file_name}::{partial_name}"
                if graph.has_node(candidate):
                    return candidate
                
                # Try just the individual part
                single_part = parts[i]
                candidate = f"{language}::{file_name}::{single_part}"
                if graph.has_node(candidate):
                    return candidate
        
        # Strategy 4: Registry lookup by exact name (cross-file search)
        if target_symbol in symbol_registry['by_name']:
            candidates = symbol_registry['by_name'][target_symbol]
            
            # Prefer class/function definitions over variables for method calls
            definition_types = {'class', 'function', 'method', 'interface', 'enum'}
            
            # First pass: look for definitions
            for candidate in candidates:
                if graph.has_node(candidate):
                    node_data = graph.nodes[candidate]
                    symbol_type = node_data.get('symbol_type', '').lower()
                    if symbol_type in definition_types:
                        return candidate
            
            # Second pass: any valid candidate
            for candidate in candidates:
                if graph.has_node(candidate):
                    return candidate
        
        # Strategy 5: Registry lookup by qualified name
        if target_symbol in symbol_registry['by_qualified_name']:
            candidate = symbol_registry['by_qualified_name'][target_symbol]
            if graph.has_node(candidate):
                return candidate
        
        # Strategy 6: Cross-file search for exact matches
        for other_file_path, other_result in parse_results.items():
            other_file_name = Path(other_file_path).stem
            
            # Try exact match in other file
            candidate = f"{language}::{other_file_name}::{target_symbol}"
            if graph.has_node(candidate):
                return candidate
            
            # For dotted names, try the last part in other files
            if '.' in target_symbol:
                last_part = target_symbol.split('.')[-1]
                candidate = f"{language}::{other_file_name}::{last_part}"
                if graph.has_node(candidate):
                    return candidate
        
        # Strategy 7: Flexible matching for common patterns
        # Handle cases like "ClassName.method_name" where we want just "method_name"
        if '.' in target_symbol:
            method_name = target_symbol.split('.')[-1]
            
            # Look for method in current file
            method_candidate = f"{language}::{file_name}::{method_name}"
            if graph.has_node(method_candidate):
                return method_candidate
            
            # Look for method in registry
            if method_name in symbol_registry['by_name']:
                candidates = symbol_registry['by_name'][method_name]
                for candidate in candidates:
                    if graph.has_node(candidate):
                        node_data = graph.nodes[candidate]
                        symbol_type = node_data.get('symbol_type', '').lower()
                        # Prefer methods and functions
                        if symbol_type in {'method', 'function'}:
                            return candidate
        
        # Strategy 8: Create external reference if not found
        # This ensures we don't lose relationships due to missing nodes
        return f"{language}::{file_name}::{target_symbol}"
        
    async def _resolve_source_node_async(self, source_symbol: str, language: str, file_name: str, 
                                        file_path: str, graph: nx.MultiDiGraph, symbol_registry: Dict[str, Any],
                                        parse_results: Dict[str, ParseResult]) -> str:
        """Resolve source node with comprehensive fallback strategies"""
        
        # Strategy 1: File-level imports (source = module name)
        if source_symbol == file_name:
            source_node = f"{language}::{file_name}::__file__"
            if not graph.has_node(source_node):
                graph.add_node(source_node,
                             symbol_name="__file__",
                             file_path=file_path,
                             file_name=file_name,
                             language=language,
                             symbol_type="module",
                             analysis_level='comprehensive')
            return source_node
        
        # Strategy 2: Direct match in current file
        direct_source = f"{language}::{file_name}::{source_symbol}"
        if graph.has_node(direct_source):
            return direct_source
        
        # Strategy 3: Handle dotted names (class.method, module.function)
        if '.' in source_symbol:
            parts = source_symbol.split('.')
            
            # Try different combinations
            for i in range(len(parts)):
                # Try from the end: method, class.method, etc.
                partial_name = '.'.join(parts[i:])
                candidate = f"{language}::{file_name}::{partial_name}"
                if graph.has_node(candidate):
                    return candidate
                
                # Try just the part
                single_part = parts[i]
                candidate = f"{language}::{file_name}::{single_part}"
                if graph.has_node(candidate):
                    return candidate
        
        # Strategy 4: Registry lookup by name
        if source_symbol in symbol_registry['by_name']:
            candidates = symbol_registry['by_name'][source_symbol]
            
            # Prefer candidates from the current file
            for candidate in candidates:
                if file_name in candidate and graph.has_node(candidate):
                    return candidate
            
            # Fallback to any candidate
            for candidate in candidates:
                if graph.has_node(candidate):
                    return candidate
        
        # Strategy 5: Qualified name lookup
        if source_symbol in symbol_registry['by_qualified_name']:
            candidate = symbol_registry['by_qualified_name'][source_symbol]
            if graph.has_node(candidate):
                return candidate
        
        # Strategy 6: Cross-file search for common patterns
        for other_file_path, other_result in parse_results.items():
            other_file_name = Path(other_file_path).stem
            
            # Look for exact symbol in other files
            candidate = f"{language}::{other_file_name}::{source_symbol}"
            if graph.has_node(candidate):
                return candidate
        
        # Strategy 7: Create the node if it doesn't exist (fallback)
        fallback_node = f"{language}::{file_name}::{source_symbol}"
        return fallback_node
    
    def _merge_graph(self, target_graph: nx.MultiDiGraph, source_graph: nx.MultiDiGraph):
        """Merge source code_graph into target code_graph"""
        if not source_graph:
            return
        
        # Add all nodes from source to target
        for node_id, node_attrs in source_graph.nodes(data=True):
            if node_id not in target_graph.nodes:
                target_graph.add_node(node_id, **node_attrs)
            else:
                # Merge attributes if node already exists
                existing_attrs = target_graph.nodes[node_id]
                merged_attrs = {**existing_attrs, **node_attrs}
                target_graph.add_node(node_id, **merged_attrs)
        
        # Add all edges from source to target (MultiDiGraph preserves multiple edges)
        for source_id, target_id, key, edge_attrs in source_graph.edges(data=True, keys=True):
            target_graph.add_edge(source_id, target_id, key=key, **edge_attrs)
    
    def _build_basic_language_graph(self, parse_results: Dict[str, BasicParseResult], language: str) -> nx.DiGraph:
        """Build basic code_graph for tree-sitter parsers"""
        graph = nx.DiGraph()
        graph.graph['language'] = language
        graph.graph['analysis_level'] = 'basic'
        
        # Simple code_graph building for basic parsers
        for file_path, result in parse_results.items():
            file_name = Path(file_path).stem
            
            # Add symbols as nodes
            for symbol in result.symbols:
                node_id = f"{language}::{file_name}::{symbol.name}"
                
                # Handle duplicates
                base_node_id = node_id
                counter = 1
                while node_id in graph.nodes:
                    node_id = f"{base_node_id}#{counter}"
                    counter += 1
                
                node_attrs = {
                    'name': symbol.name,
                    'type': symbol.symbol_type,
                    'file_path': file_path,
                    'file_name': file_name,
                    'language': language,
                    'start_line': symbol.start_line,
                    'end_line': symbol.end_line,
                    'analysis_level': 'basic',
                    'source_text': symbol.source_text,
                    'docstring': symbol.docstring or '',
                    'parameters': symbol.parameters,
                    'return_type': symbol.return_type or ''
                }
                
                graph.add_node(node_id, **node_attrs)
            
            # Add relationships as edges
            for relationship in result.relationships:
                # Simple resolution for basic relationships
                source_id = f"{language}::{file_name}::{relationship.source_symbol}"
                target_id = f"{language}::{file_name}::{relationship.target_symbol}"
                
                if source_id in graph.nodes and target_id in graph.nodes:
                    edge_attrs = {
                        'type': relationship.relationship_type,
                        'source_file': relationship.source_file,
                        'target_file': relationship.target_file,
                        'analysis_level': 'basic',
                        'language': language
                    }
                    graph.add_edge(source_id, target_id, **edge_attrs)
        
        return graph
    
    def _is_java_standard_library_reference(self, target_symbol: str, target_node: str) -> bool:
        """Check if a target symbol/node references Java standard library."""
        if not target_symbol:
            return False
            
        # Check for fully qualified Java standard library packages
        java_std_packages = {
            'java.lang',
            'java.util',
            'java.io',
            'java.nio',
            'java.time',
            'java.math',
            'java.net',
            'java.text',
            'java.util.regex',
            'java.util.concurrent',
            'java.util.function',
            'java.util.stream',
            'java.security',
            'java.awt',
            'javax.swing',
            'javax.annotation'
        }
        
        # Check if it's a fully qualified Java standard library type
        if '.' in target_symbol:
            for package in java_std_packages:
                if target_symbol.startswith(package + '.'):
                    return True
        
        # Check for common built-in methods on standard library types
        builtin_methods = {
            'isEmpty', 'trim', 'length', 'contains', 'add', 'remove', 'get', 'set',
            'toString', 'equals', 'hashCode', 'getClass', 'format', 'valueOf',
            'randomUUID', 'now', 'hash', 'compile', 'matches', 'matcher',
            'orElseThrow', 'isPresent', 'of', 'empty'
        }
        
        # Check if the target symbol is a method call on a standard library type
        if '.' in target_symbol:
            method_name = target_symbol.split('.')[-1]
            if method_name in builtin_methods:
                return True
        
        return False
    
    def _generate_symbol_chunks(self, parse_results: Dict[str, Union[ParseResult, BasicParseResult]]) -> List[Document]:
        """Generate symbol-level chunk documents from parse results"""
        documents = []
        
        for file_path, result in parse_results.items():
            file_name = Path(file_path).name
            
            # Create document for each symbol
            for symbol in result.symbols:
                # Determine if this is rich or basic parsing
                analysis_level = getattr(result, 'parse_level', 'comprehensive')
                
                if analysis_level == 'documentation':
                    # Documentation chunk (from text processing)
                    content = symbol.source_text or ''
                    symbol_type = symbol.symbol_type
                    metadata = {
                        'chunk_type': 'text',  # Different chunk type for documentation
                        'symbol_name': symbol.name,
                        'symbol_type': symbol_type,
                        'file_path': file_path,
                        'file_name': file_name,
                        'language': symbol.language,
                        'start_line': symbol.start_line,
                        'end_line': symbol.end_line,
                        'analysis_level': 'documentation',
                        'docstring': symbol.docstring or '',
                        'summary': symbol.docstring or '',  # Use docstring as summary for text chunks
                        'content_type': symbol.return_type or 'text',  # Using return_type to store content type
                        'section_id': getattr(symbol, 'section_id', None),  # For reconstruction order
                        'section_order': getattr(symbol, 'section_order', None)  # For reconstruction order
                    }
                elif analysis_level == 'basic':
                    # Basic symbol (from tree-sitter)
                    content = symbol.source_text or ''
                    symbol_type = symbol.symbol_type
                    metadata = {
                        'chunk_type': 'symbol',
                        'symbol_name': symbol.name,
                        'symbol_type': symbol_type,
                        'file_path': file_path,
                        'file_name': file_name,
                        'language': symbol.language,
                        'start_line': symbol.start_line,
                        'end_line': symbol.end_line,
                        'analysis_level': 'basic',
                        'docstring': symbol.docstring or '',
                        'parameters': symbol.parameters,
                        'return_type': symbol.return_type or ''
                    }
                else:
                    # Rich symbol (from comprehensive parsers)
                    content = getattr(symbol, 'source_text', '') or ''
                    symbol_type = symbol.symbol_type.value if hasattr(symbol.symbol_type, 'value') else str(symbol.symbol_type)
                    metadata = {
                        'chunk_type': 'symbol',
                        'symbol_name': symbol.name,
                        'symbol_type': symbol_type,
                        'file_path': file_path,
                        'file_name': file_name,
                        'language': result.language,
                        'start_line': symbol.range.start.line if symbol.range else 0,
                        'end_line': symbol.range.end.line if symbol.range else 0,
                        'analysis_level': 'comprehensive',
                        'docstring': getattr(symbol, 'docstring', '') or '',
                        'parameters': getattr(symbol, 'parameters', []),
                        'return_type': getattr(symbol, 'return_type', '') or '',
                        'access_modifier': getattr(symbol, 'access_modifier', ''),
                        'is_abstract': getattr(symbol, 'is_abstract', False),
                        'is_static': getattr(symbol, 'is_static', False)
                    }
                
                # Create document
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
        
        return documents
    
    def _detect_cross_language_relationships(self, files_by_language: Dict[str, List[str]], 
                                           parse_results: Dict[str, Union[ParseResult, BasicParseResult]]) -> List[Any]:
        """Detect cross-language relationships (simplified for now)"""
        cross_lang_rels = []
        
        # For now, return empty list - this would require more sophisticated analysis
        # to detect things like Java calling Python scripts, etc.
        
        return cross_lang_rels
    
    def _generate_language_stats(self, parse_results: Dict[str, Union[ParseResult, BasicParseResult]]) -> Dict[str, Any]:
        """Generate statistics by language"""
        stats = {}
        
        # Group results by language
        by_language = defaultdict(list)
        for file_path, result in parse_results.items():
            language = result.language
            by_language[language].append(result)
        
        # Generate stats for each language
        for language, results in by_language.items():
            total_symbols = sum(len(result.symbols) for result in results)
            total_relationships = sum(len(result.relationships) for result in results)
            
            # Determine analysis level
            analysis_levels = set()
            for result in results:
                if hasattr(result, 'parse_level'):
                    analysis_levels.add(result.parse_level)
                else:
                    analysis_levels.add('comprehensive')
            
            stats[language] = {
                'files_count': len(results),
                'symbols_count': total_symbols,
                'relationships_count': total_relationships,
                'analysis_levels': list(analysis_levels)
            }
        
        return stats
    
    def _parse_documentation_files(self, file_paths: List[str]) -> Dict[str, 'BasicParseResult']:
        """
        Process documentation files (markdown, text, etc.) with text chunking
        instead of symbol extraction.
        """
        results = {}
        
        for file_path in file_paths:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Determine file type
                file_extension = Path(file_path).suffix.lower()
                doc_type = self.DOCUMENTATION_EXTENSIONS.get(file_extension, 'text')
                
                # Simple text chunking for documentation
                chunks = self._chunk_text_content(content, file_path, doc_type)
                
                # Create symbols for each text chunk
                symbols = []
                for i, chunk in enumerate(chunks):
                    symbol = BasicSymbol(
                        name=chunk.get('summary', f"{Path(file_path).stem}_chunk_{i}"),
                        symbol_type=chunk.get('section_type', 'text_chunk'),
                        start_line=chunk.get('start_line', 0),
                        end_line=chunk.get('end_line', 0),
                        file_path=file_path,
                        language=doc_type,
                        source_text=chunk['content'],
                        docstring=chunk.get('summary', ''),
                        parameters=[],
                        return_type=doc_type
                    )
                    # Add section ID and order for reconstruction
                    if 'section_id' in chunk:
                        symbol.section_id = chunk['section_id']
                    if 'section_order' in chunk:
                        symbol.section_order = chunk['section_order']
                    symbols.append(symbol)
                
                # Create parse result
                result = BasicParseResult(
                    file_path=file_path,
                    language=doc_type,
                    symbols=symbols,
                    relationships=[],  # No relationships for text chunks
                    parse_level='documentation',
                    imports=[]
                )
                results[file_path] = result
                
                logger.debug(f"Processed documentation file {file_path}: {len(symbols)} text chunks")
                
            except Exception as e:
                logger.warning(f"Failed to parse documentation file {file_path}: {e}")
        
        return results
    
    def _chunk_text_content(self, content: str, file_path: str, doc_type: str) -> List[Dict[str, Any]]:
        """
        Chunk text content into manageable pieces for documentation files.
        Only split if content is larger than reasonable size (15-17k characters).
        """
        chunks = []
        
        # Check if document is reasonably sized (13-17k characters)
        content_size = len(content)
        max_reasonable_size = 17000  # 17k characters
        min_split_size = 13000       # 13k characters - only split if larger than this
        
        logger.debug(f"Document size: {content_size} characters, file: {file_path}")
        
        # If document is reasonably sized, keep it as a single chunk
        if content_size <= max_reasonable_size:
            logger.info(f"Document {file_path} ({content_size} chars) fits in context - keeping as single chunk")
            return [{
                'content': content,
                'summary': Path(file_path).stem,
                'start_line': 1,
                'end_line': len(content.split('\n')),
                'headers': {},
                'section_type': f'{doc_type}_document',
                'section_id': 0,
                'section_order': 0
            }]
        
        # Only split if document is larger than minimum split threshold
        if content_size > min_split_size:
            logger.info(f"Document {file_path} ({content_size} chars) is large - attempting to split")
            
            # For markdown files, try to split by headers
            if doc_type == 'markdown':
                chunks = self._chunk_markdown_content(content, file_path)
            else:
                # For other text files, use simple line-based chunking
                chunks = self._chunk_generic_text(content, file_path)
        else:
            logger.info(f"Document {file_path} ({content_size} chars) below split threshold - keeping as single chunk")
            chunks = [{
                'content': content,
                'summary': Path(file_path).stem,
                'start_line': 1,
                'end_line': len(content.split('\n')),
                'headers': {},
                'section_type': f'{doc_type}_document',
                'section_id': 0,
                'section_order': 0
            }]
        
        return chunks
    
    def _chunk_markdown_content(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Split markdown content using LangChain's MarkdownHeaderTextSplitter
        keeping complete sections without character-based splitting.
        Each chunk gets a sequential ID for reconstruction in original order.
        """
        chunks = []
        
        try:
            # Define headers to split on (h1, h2, h3, h4)
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"), 
                ("###", "Header 3"),
                ("####", "Header 4"),
            ]
            
            # Create markdown splitter
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                strip_headers=False,  # Keep headers in content
                return_each_line=True
            )
            
            # Split the markdown content
            md_header_splits = markdown_splitter.split_text(content)
            
            # Process each markdown section - keep complete sections without further splitting
            for i, split in enumerate(md_header_splits):
                section_content = split.page_content
                section_metadata = split.metadata
                
                # Extract section name from metadata or content
                section_name = self._extract_section_name(section_metadata, section_content, i)
                
                # Create chunk with sequential ID for reconstruction
                chunks.append({
                    'content': section_content,
                    'summary': section_name,
                    'start_line': self._estimate_line_number(content, section_content),
                    'end_line': self._estimate_line_number(content, section_content) + section_content.count('\n'),
                    'headers': section_metadata,
                    'section_type': 'markdown_section',
                    'section_id': i,  # Sequential ID for reconstruction
                    'section_order': i  # Explicit order field for sorting during reconstruction
                })
                    
        except Exception as e:
            logger.warning(f"Failed to split markdown with LangChain splitter: {e}, falling back to simple splitting")
            # Fallback to simple chunking
            chunks = self._chunk_generic_text(content, file_path, max_lines=50)
        
        # If no chunks found, create one chunk with entire content
        if not chunks:
            chunks.append({
                'content': content,
                'summary': Path(file_path).stem,
                'start_line': 1,
                'end_line': len(content.split('\n')),
                'headers': {},
                'section_type': 'markdown_document',
                'section_id': 0,
                'section_order': 0
            })
        
        return chunks
    
    def _extract_section_name(self, metadata: Dict[str, str], content: str, index: int) -> str:
        """Extract a meaningful section name from metadata or content."""
        # Try to get section name from metadata (headers)
        if metadata:
            # Get the deepest header level
            for header_level in ["Header 4", "Header 3", "Header 2", "Header 1"]:
                if header_level in metadata:
                    return metadata[header_level].strip()
        
        # Fallback: extract from first line of content
        first_line = content.split('\n')[0].strip()
        if first_line.startswith('#'):
            return first_line.lstrip('#').strip()
        
        # Final fallback
        return f"Section {index + 1}"
    
    def _estimate_line_number(self, full_content: str, chunk_content: str) -> int:
        """Estimate the line number where a chunk starts in the full content."""
        try:
            chunk_start = full_content.find(chunk_content)
            if chunk_start == -1:
                return 1
            
            # Count newlines before the chunk
            lines_before = full_content[:chunk_start].count('\n')
            return lines_before + 1
        except Exception:
            return 1
    
    def _chunk_generic_text(self, content: str, file_path: str, max_lines: int = 30) -> List[Dict[str, Any]]:
        """
        Chunk generic text content into manageable pieces using line-based splitting.
        
        Args:
            content: The text content to chunk
            file_path: Path to the source file
            max_lines: Maximum number of lines per chunk
            
        Returns:
            List of chunk dictionaries with content, summary, and line info
        """
        chunks = []
        lines = content.split('\n')
        
        # Simple line-based chunking
        for i in range(0, len(lines), max_lines):
            chunk_lines = lines[i:i + max_lines]
            chunk_content = '\n'.join(chunk_lines)
            
            # Skip empty chunks
            if not chunk_content.strip():
                continue
            
            # Extract first meaningful line as summary
            summary_line = ""
            for line in chunk_lines:
                if line.strip():
                    summary_line = line.strip()[:50]  # First 50 chars
                    if len(line.strip()) > 50:
                        summary_line += "..."
                    break
            
            if not summary_line:
                summary_line = f"{Path(file_path).stem} chunk {len(chunks) + 1}"
            
            chunks.append({
                'content': chunk_content,
                'summary': summary_line,
                'start_line': i + 1,
                'end_line': min(i + max_lines, len(lines)),
                'section_type': 'text_chunk'
            })
        
        # If no chunks created, create one with entire content
        if not chunks:
            chunks.append({
                'content': content,
                'summary': Path(file_path).stem,
                'start_line': 1,
                'end_line': len(lines),
                'section_type': 'text_document'
            })
        
        return chunks
