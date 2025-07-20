"""
Enhanced Python Parser implementing the new modular parser architecture

This parser provides comprehensive Python AST analysis with support for:
- Complete symbol extraction (functions, classes, methods, variables, etc.)
- Advanced relationship detection (inheritance, composition, calls, imports)
- Hierarchical context expansion
- Type annotation support
- Decorator and async/await support
"""

import ast
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any, Tuple

from .base_parser import (
    BaseParser, LanguageCapabilities, ParseResult, Symbol, Relationship,
    SymbolType, RelationshipType, Scope, Position, Range
)

logger = logging.getLogger(__name__)


def _parse_single_python_file(file_path: str) -> Tuple[str, ParseResult]:
    """Parse a single Python file - used by parallel workers"""
    try:
        # Create a new parser instance for this worker
        parser = PythonParser()
        result = parser.parse_file(file_path)
        return file_path, result
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        # Create empty result for failed files
        return file_path, ParseResult(
            file_path=file_path,
            language="python",
            symbols=[], 
            relationships=[]
        )


class PythonParser(BaseParser):
    """
    Advanced Python AST parser implementing the modular parser architecture
    """
    
    def __init__(self):
        super().__init__("python")
        self._current_file = ""
        self._current_content = ""
        self._current_scope_stack: List[str] = []
    
    def _define_capabilities(self) -> LanguageCapabilities:
        """Define Python parser capabilities"""
        return LanguageCapabilities(
            language="python",
            supported_symbols={
                SymbolType.FUNCTION,
                SymbolType.METHOD,
                SymbolType.CLASS,
                SymbolType.VARIABLE,
                SymbolType.CONSTANT,
                SymbolType.PROPERTY,
                SymbolType.PARAMETER,
                SymbolType.MODULE,
                SymbolType.DECORATOR
            },
            supported_relationships={
                RelationshipType.IMPORTS,
                RelationshipType.CALLS,
                RelationshipType.INHERITANCE,
                RelationshipType.COMPOSITION,
                RelationshipType.AGGREGATION,
                RelationshipType.DEFINES,
                RelationshipType.DECORATES,
                RelationshipType.ANNOTATES,
                RelationshipType.ASSIGNS,
                RelationshipType.RETURNS,
                RelationshipType.PARAMETER,
                RelationshipType.REFERENCES
            },
            supports_ast_parsing=True,
            supports_type_inference=True,
            supports_cross_file_analysis=True,
            has_classes=True,
            has_interfaces=False,  # Python uses ABC
            has_namespaces=False,  # Python uses modules
            has_generics=True,     # Type hints support generics
            has_decorators=True,
            has_annotations=True,
            max_file_size_mb=5.0,
            typical_parse_time_ms=50.0
        )
    
    def _get_supported_extensions(self) -> Set[str]:
        """Get supported Python file extensions"""
        return {'.py', '.pyx', '.pyi', '.pyw'}
    
    def parse_file(self, file_path: Union[str, Path], content: Optional[str] = None) -> ParseResult:
        """
        Parse a Python file and extract symbols and relationships
        """
        start_time = time.time()
        file_path = str(file_path)
        self._current_file = file_path
        
        try:
            # Read content if not provided
            if content is None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            self._current_content = content
            
            # Calculate file hash for caching
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Parse AST
            try:
                ast_tree = ast.parse(content, filename=file_path)
            except SyntaxError as e:
                return ParseResult(
                    file_path=file_path,
                    language="python",
                    symbols=[],
                    relationships=[],
                    file_hash=file_hash,
                    parse_time=time.time() - start_time,
                    errors=[f"Syntax error: {e}"]
                )
            
            # Extract symbols and relationships
            symbols = self.extract_symbols(ast_tree, file_path)
            relationships = self.extract_relationships(ast_tree, symbols, file_path)
            
            # Extract module-level information
            imports, exports = self._extract_module_info(ast_tree)
            dependencies = self._extract_dependencies(imports, relationships)
            module_docstring = self._extract_module_docstring(ast_tree)
            
            result = ParseResult(
                file_path=file_path,
                language="python",
                symbols=symbols,
                relationships=relationships,
                imports=imports,
                exports=exports,
                dependencies=dependencies,
                module_docstring=module_docstring,
                file_hash=file_hash,
                parse_time=time.time() - start_time
            )
            
            return self.validate_result(result)
            
        except Exception as e:
            logger.error(f"Failed to parse Python file {file_path}: {e}")
            return ParseResult(
                file_path=file_path,
                language="python",
                symbols=[],
                relationships=[],
                parse_time=time.time() - start_time,
                errors=[f"Parse error: {e}"]
            )
    
    def extract_symbols(self, ast_node: Any, file_path: str) -> List[Symbol]:
        """Extract symbols from Python AST"""
        symbols = []
        self._current_scope_stack = []
        
        class SymbolExtractor(ast.NodeVisitor):
            def __init__(self, parser):
                self.parser = parser
                self.symbols = []
                self.scope_stack = []
            
            def visit_Module(self, node):
                # Module symbol
                symbol = Symbol(
                    name=Path(file_path).stem,
                    symbol_type=SymbolType.MODULE,
                    scope=Scope.GLOBAL,
                    range=Range(Position(1, 0), Position(len(self.parser._current_content.split('\n')), 0)),
                    file_path=file_path,
                    docstring=ast.get_docstring(node)
                )
                self.symbols.append(symbol)
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                # Enter class scope
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{node.name}" if parent_symbol else node.name
                
                # Extract base classes
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(self.parser._get_full_attribute_name(base))
                    elif isinstance(base, ast.Subscript):
                        # Handle generic inheritance like InMemoryRepository[User]
                        # Extract the base class name from the subscript value
                        if isinstance(base.value, ast.Name):
                            bases.append(base.value.id)
                        elif isinstance(base.value, ast.Attribute):
                            bases.append(self.parser._get_full_attribute_name(base.value))
                
                # Extract decorators
                decorators = [self.parser._get_decorator_name(dec) for dec in node.decorator_list]
                
                symbol = Symbol(
                    name=node.name,
                    symbol_type=SymbolType.CLASS,
                    scope=Scope.CLASS if self.scope_stack else Scope.GLOBAL,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    docstring=ast.get_docstring(node),
                    source_text=self.parser._extract_node_source(node)
                )
                self.symbols.append(symbol)
                
                # Enter scope and visit children
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()
            
            def visit_FunctionDef(self, node):
                self._visit_function(node, is_async=False)
            
            def visit_AsyncFunctionDef(self, node):
                self._visit_function(node, is_async=True)
            
            def _visit_function(self, node, is_async=False):
                parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                full_name = f"{parent_symbol}.{node.name}" if parent_symbol else node.name
                
                # Determine if this is a method or function
                is_in_class = any(isinstance(sym, str) and sym in [s.name for s in self.symbols if s.symbol_type == SymbolType.CLASS] for sym in self.scope_stack)
                symbol_type = SymbolType.METHOD if is_in_class else SymbolType.FUNCTION
                scope = Scope.FUNCTION
                
                # Extract decorators
                decorators = [self.parser._get_decorator_name(dec) for dec in node.decorator_list]
                
                # Extract return type annotation
                return_type = None
                if node.returns:
                    return_type = self.parser._get_type_annotation(node.returns)
                
                # Extract parameters
                parameters = self.parser._extract_function_parameters(node)
                
                symbol = Symbol(
                    name=node.name,
                    symbol_type=symbol_type,
                    scope=scope,
                    range=self.parser._node_to_range(node),
                    file_path=file_path,
                    parent_symbol=parent_symbol,
                    full_name=full_name,
                    docstring=ast.get_docstring(node),
                    return_type=return_type,
                    parameter_types=[p.get('type') for p in parameters if p.get('type')],
                    is_async=is_async,
                    source_text=self.parser._extract_node_source(node),
                    signature=self.parser._build_function_signature(node, return_type)
                )
                self.symbols.append(symbol)
                
                # Add parameter symbols
                for param in parameters:
                    param_symbol = Symbol(
                        name=param['name'],
                        symbol_type=SymbolType.PARAMETER,
                        scope=Scope.FUNCTION,
                        range=param.get('range', symbol.range),
                        file_path=file_path,
                        parent_symbol=full_name,
                        return_type=param.get('type')
                    )
                    self.symbols.append(param_symbol)
                
                # Enter scope and visit children (for nested functions)
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()
            
            def visit_Assign(self, node):
                # Variable assignments
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                        scope = Scope.FUNCTION if self.scope_stack else Scope.GLOBAL
                        
                        # Determine if this is a constant (all uppercase)
                        symbol_type = SymbolType.CONSTANT if target.id.isupper() else SymbolType.VARIABLE
                        
                        symbol = Symbol(
                            name=target.id,
                            symbol_type=symbol_type,
                            scope=scope,
                            range=self.parser._node_to_range(target),
                            file_path=file_path,
                            parent_symbol=parent_symbol,
                            source_text=self.parser._extract_node_source(node)
                        )
                        self.symbols.append(symbol)
                
                self.generic_visit(node)
            
            def visit_AnnAssign(self, node):
                # Annotated assignments
                if isinstance(node.target, ast.Name):
                    parent_symbol = '.'.join(self.scope_stack) if self.scope_stack else None
                    scope = Scope.FUNCTION if self.scope_stack else Scope.GLOBAL
                    
                    # Extract type annotation
                    type_annotation = self.parser._get_type_annotation(node.annotation) if node.annotation else None
                    
                    symbol = Symbol(
                        name=node.target.id,
                        symbol_type=SymbolType.VARIABLE,
                        scope=scope,
                        range=self.parser._node_to_range(node.target),
                        file_path=file_path,
                        parent_symbol=parent_symbol,
                        return_type=type_annotation,
                        source_text=self.parser._extract_node_source(node)
                    )
                    self.symbols.append(symbol)
                
                self.generic_visit(node)
        
        extractor = SymbolExtractor(self)
        extractor.visit(ast_node)
        return extractor.symbols
    
    def extract_relationships(self, ast_node: Any, symbols: List[Symbol], file_path: str) -> List[Relationship]:
        """Extract relationships from Python AST"""
        relationships = []
        symbol_map = {s.name: s for s in symbols}
        
        class RelationshipExtractor(ast.NodeVisitor):
            def __init__(self, parser):
                self.parser = parser
                self.relationships = []
                self.current_symbol = None
                self.scope_stack = []
            
            def visit_ClassDef(self, node):
                self.current_symbol = node.name
                
                # Inheritance relationships
                for base in node.bases:
                    base_name = None
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = self.parser._get_full_attribute_name(base)
                    elif isinstance(base, ast.Subscript):
                        # Handle generic inheritance like InMemoryRepository[User]
                        # Extract the base class name from the subscript value
                        if isinstance(base.value, ast.Name):
                            base_name = base.value.id
                        elif isinstance(base.value, ast.Attribute):
                            base_name = self.parser._get_full_attribute_name(base.value)
                        
                        # Extract generic type parameter(s) and create reference relationships
                        type_params = []
                        if isinstance(base.slice, ast.Name):
                            # Single type parameter: InMemoryRepository[User]
                            type_params.append(base.slice.id)
                        elif isinstance(base.slice, ast.Tuple):
                            # Multiple type parameters: Dict[str, int]
                            for elt in base.slice.elts:
                                if isinstance(elt, ast.Name):
                                    type_params.append(elt.id)
                                elif isinstance(elt, ast.Attribute):
                                    type_params.append(self.parser._get_full_attribute_name(elt))
                        elif isinstance(base.slice, ast.Attribute):
                            # Qualified type parameter: Dict[models.User, str]
                            type_params.append(self.parser._get_full_attribute_name(base.slice))
                        
                        # Create reference relationships for type parameters
                        for type_param in type_params:
                            if type_param:
                                type_ref_relationship = Relationship(
                                    source_symbol=node.name,
                                    target_symbol=type_param,
                                    relationship_type=RelationshipType.REFERENCES,
                                    source_file=file_path,
                                    source_range=self.parser._node_to_range(base.slice),
                                    confidence=0.8,
                                    weight=0.6
                                )
                                self.relationships.append(type_ref_relationship)
                    
                    if base_name:
                        relationship = Relationship(
                            source_symbol=node.name,
                            target_symbol=base_name,
                            relationship_type=RelationshipType.INHERITANCE,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(base),
                            confidence=0.9,
                            weight=0.8
                        )
                        self.relationships.append(relationship)
                
                # Decorator relationships
                for decorator in node.decorator_list:
                    decorator_name = self.parser._get_decorator_name(decorator)
                    if decorator_name:
                        relationship = Relationship(
                            source_symbol=decorator_name,
                            target_symbol=node.name,
                            relationship_type=RelationshipType.DECORATES,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(decorator),
                            confidence=0.95,
                            weight=0.6
                        )
                        self.relationships.append(relationship)
                
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()
                self.current_symbol = None
            
            def visit_FunctionDef(self, node):
                self._visit_function(node)
            
            def visit_AsyncFunctionDef(self, node):
                self._visit_function(node)
            
            def _visit_function(self, node):
                old_symbol = self.current_symbol
                self.current_symbol = node.name
                
                # Decorator relationships
                for decorator in node.decorator_list:
                    decorator_name = self.parser._get_decorator_name(decorator)
                    if decorator_name:
                        relationship = Relationship(
                            source_symbol=decorator_name,
                            target_symbol=node.name,
                            relationship_type=RelationshipType.DECORATES,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(decorator),
                            confidence=0.95,
                            weight=0.6
                        )
                        self.relationships.append(relationship)
                
                self.scope_stack.append(node.name)
                self.generic_visit(node)
                self.scope_stack.pop()
                self.current_symbol = old_symbol
            
            def visit_Call(self, node):
                if self.current_symbol:
                    # Function/method calls
                    called_name = None
                    if isinstance(node.func, ast.Name):
                        called_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        called_name = self.parser._get_full_attribute_name(node.func)
                    
                    if called_name:
                        relationship = Relationship(
                            source_symbol=self.current_symbol,
                            target_symbol=called_name,
                            relationship_type=RelationshipType.CALLS,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(node),
                            confidence=0.8,
                            weight=0.7
                        )
                        self.relationships.append(relationship)
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                for alias in node.names:
                    imported_name = alias.asname if alias.asname else alias.name
                    relationship = Relationship(
                        source_symbol=Path(file_path).stem,  # Current module name
                        target_symbol=imported_name,
                        relationship_type=RelationshipType.IMPORTS,
                        source_file=file_path,
                        source_range=self.parser._node_to_range(node),
                        confidence=1.0,
                        weight=0.5
                    )
                    self.relationships.append(relationship)
                
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                module_name = node.module or ""
                for alias in node.names:
                    # Use the alias if it exists, otherwise use the actual import name
                    imported_name = alias.asname if alias.asname else alias.name
                    
                    # For relationships, we care about the name as it appears in the code
                    # So if there's an alias, use it; otherwise use the full module.name
                    if alias.asname:
                        target_name = alias.asname
                    else:
                        target_name = f"{module_name}.{alias.name}" if module_name and alias.name != "*" else alias.name
                    
                    relationship = Relationship(
                        source_symbol=Path(file_path).stem,  # Current module name
                        target_symbol=target_name,
                        relationship_type=RelationshipType.IMPORTS,
                        source_file=file_path,
                        source_range=self.parser._node_to_range(node),
                        confidence=1.0,
                        weight=0.5
                    )
                    self.relationships.append(relationship)
                
                self.generic_visit(node)
            
            def visit_Attribute(self, node):
                if self.current_symbol:
                    # Attribute access (composition/aggregation)
                    attr_name = self.parser._get_full_attribute_name(node)
                    if attr_name:
                        # Determine if this is composition or aggregation based on context
                        rel_type = RelationshipType.COMPOSITION  # Default to composition
                        
                        relationship = Relationship(
                            source_symbol=self.current_symbol,
                            target_symbol=attr_name,
                            relationship_type=rel_type,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(node),
                            confidence=0.6,
                            weight=0.4
                        )
                        self.relationships.append(relationship)
                
                self.generic_visit(node)
            
            def visit_Name(self, node):
                """Detect variable and symbol references"""
                if self.current_symbol and isinstance(node.ctx, ast.Load):
                    # This is a variable/symbol being referenced (loaded/used)
                    # Exclude some common built-ins and keywords to reduce noise
                    excluded_names = {
                        'self', 'cls', 'True', 'False', 'None', 'print', 'len', 'str', 'int', 
                        'float', 'bool', 'list', 'dict', 'tuple', 'set', 'range', 'enumerate',
                        'zip', 'map', 'filter', 'sum', 'min', 'max', 'abs', 'all', 'any'
                    }
                    
                    if node.id not in excluded_names and node.id != self.current_symbol:
                        # Check if this name represents a symbol we know about
                        # This could be a class, function, variable, etc. being referenced
                        relationship = Relationship(
                            source_symbol=self.current_symbol,
                            target_symbol=node.id,
                            relationship_type=RelationshipType.REFERENCES,
                            source_file=file_path,
                            source_range=self.parser._node_to_range(node),
                            confidence=0.7,
                            weight=0.3
                        )
                        self.relationships.append(relationship)
                
                self.generic_visit(node)
        
        extractor = RelationshipExtractor(self)
        extractor.visit(ast_node)
        return extractor.relationships
    
    def _node_to_range(self, node: ast.AST) -> Range:
        """Convert AST node to Range object"""
        start_line = getattr(node, 'lineno', 1)
        start_col = getattr(node, 'col_offset', 0)
        end_line = getattr(node, 'end_lineno', start_line)
        end_col = getattr(node, 'end_col_offset', start_col + 1)
        
        return Range(
            Position(start_line, start_col),
            Position(end_line, end_col)
        )
    
    def _extract_node_source(self, node: ast.AST) -> str:
        """Extract source code for an AST node"""
        try:
            lines = self._current_content.split('\n')
            start_line = getattr(node, 'lineno', 1) - 1
            end_line = getattr(node, 'end_lineno', start_line + 1)
            
            if start_line < len(lines) and end_line <= len(lines):
                return '\n'.join(lines[start_line:end_line])
        except Exception:
            pass
        return ""
    
    def _get_full_attribute_name(self, node: ast.Attribute) -> str:
        """Get full dotted name for attribute access"""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
        
        return '.'.join(reversed(parts))
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Extract decorator name"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return self._get_full_attribute_name(decorator)
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return self._get_full_attribute_name(decorator.func)
        return ""
    
    def _get_type_annotation(self, annotation: ast.AST) -> str:
        """Extract type annotation as string"""
        try:
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Attribute):
                return self._get_full_attribute_name(annotation)
            elif isinstance(annotation, ast.Constant):
                return str(annotation.value)
            elif isinstance(annotation, ast.Subscript):
                # Handle generics like List[str], Dict[str, int]
                value = self._get_type_annotation(annotation.value)
                if isinstance(annotation.slice, ast.Index):  # Python < 3.9
                    slice_value = self._get_type_annotation(annotation.slice.value)
                else:  # Python >= 3.9
                    slice_value = self._get_type_annotation(annotation.slice)
                return f"{value}[{slice_value}]"
            else:
                # Fallback to string representation
                return ast.unparse(annotation) if hasattr(ast, 'unparse') else str(annotation)
        except Exception:
            return "Any"
    
    def _extract_function_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function parameters with type information"""
        parameters = []
        
        # Regular arguments
        for i, arg in enumerate(node.args.args):
            param_type = None
            if arg.annotation:
                param_type = self._get_type_annotation(arg.annotation)
            
            default_value = None
            defaults_offset = len(node.args.args) - len(node.args.defaults)
            if i >= defaults_offset:
                default_index = i - defaults_offset
                if default_index < len(node.args.defaults):
                    default_value = self._extract_default_value(node.args.defaults[default_index])
            
            parameters.append({
                'name': arg.arg,
                'type': param_type,
                'default': default_value,
                'vararg': False,
                'kwarg': False,
                'range': self._node_to_range(arg)
            })
        
        # *args parameter
        if node.args.vararg:
            param_type = None
            if node.args.vararg.annotation:
                param_type = self._get_type_annotation(node.args.vararg.annotation)
            
            parameters.append({
                'name': node.args.vararg.arg,
                'type': param_type,
                'default': None,
                'vararg': True,
                'kwarg': False,
                'range': self._node_to_range(node.args.vararg)
            })
        
        # **kwargs parameter
        if node.args.kwarg:
            param_type = None
            if node.args.kwarg.annotation:
                param_type = self._get_type_annotation(node.args.kwarg.annotation)
            
            parameters.append({
                'name': node.args.kwarg.arg,
                'type': param_type,
                'default': None,
                'vararg': False,
                'kwarg': True,
                'range': self._node_to_range(node.args.kwarg)
            })
        
        return parameters
    
    def _extract_default_value(self, node: ast.AST) -> str:
        """Extract default parameter value as string"""
        try:
            if isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return self._get_full_attribute_name(node)
            else:
                return ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
        except Exception:
            return "None"
    
    def _build_function_signature(self, node: ast.FunctionDef, return_type: Optional[str]) -> str:
        """Build function signature string"""
        try:
            params = []
            
            # Regular parameters
            for arg in node.args.args:
                param_str = arg.arg
                if arg.annotation:
                    param_str += f": {self._get_type_annotation(arg.annotation)}"
                params.append(param_str)
            
            # *args
            if node.args.vararg:
                vararg_str = f"*{node.args.vararg.arg}"
                if node.args.vararg.annotation:
                    vararg_str += f": {self._get_type_annotation(node.args.vararg.annotation)}"
                params.append(vararg_str)
            
            # **kwargs
            if node.args.kwarg:
                kwarg_str = f"**{node.args.kwarg.arg}"
                if node.args.kwarg.annotation:
                    kwarg_str += f": {self._get_type_annotation(node.args.kwarg.annotation)}"
                params.append(kwarg_str)
            
            signature = f"{node.name}({', '.join(params)})"
            if return_type:
                signature += f" -> {return_type}"
            
            return signature
        except Exception:
            return node.name
    
    def _extract_module_info(self, ast_tree: ast.Module) -> Tuple[List[str], List[str]]:
        """Extract imports and exports from module"""
        imports = []
        exports = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        imports.append(f"{module}.*")
                    else:
                        imports.append(f"{module}.{alias.name}")
        
        # For exports, look for __all__ definition
        for node in ast.walk(ast_tree):
            if (isinstance(node, ast.Assign) and 
                len(node.targets) == 1 and 
                isinstance(node.targets[0], ast.Name) and 
                node.targets[0].id == "__all__"):
                
                if isinstance(node.value, ast.List):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Str):
                            exports.append(elt.s)
                        elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exports.append(elt.value)
        
        return imports, exports
    
    def _extract_dependencies(self, imports: List[str], relationships: List[Relationship]) -> Set[str]:
        """Extract file dependencies from imports and relationships"""
        dependencies = set()
        
        # Add imported modules
        for imp in imports:
            # Extract top-level module name
            top_module = imp.split('.')[0]
            dependencies.add(top_module)
        
        # Add modules from cross-file relationships
        for rel in relationships:
            if '.' in rel.target_symbol:
                module_name = rel.target_symbol.split('.')[0]
                dependencies.add(module_name)
        
        return dependencies
    
    def _extract_module_docstring(self, ast_tree: ast.Module) -> Optional[str]:
        """Extract module-level docstring"""
        return ast.get_docstring(ast_tree)
    
    def parse_multiple_files(self, file_paths: List[str], max_workers: int = 4) -> Dict[str, ParseResult]:
        """
        Parse multiple Python files and resolve cross-file relationships.
        
        This method enables proper cross-file symbol resolution by:
        1. First parsing all files individually IN PARALLEL
        2. Building a global symbol registry across all files 
        3. Enhancing relationships with cross-file target information
        
        Args:
            file_paths: List of file paths to parse
            max_workers: Maximum number of parallel workers for file parsing
        """
        results = {}
        
        # Global symbol registries for cross-file resolution
        self._global_class_locations: Dict[str, str] = {}  # class_name -> file_path
        self._global_function_locations: Dict[str, str] = {}  # function_name -> file_path
        self._global_symbol_registry: Dict[str, str] = {}  # symbol_name -> full_path
        
        # First pass: Parse each file individually IN PARALLEL and build global symbol registry
        logger.info(f"Parsing {len(file_paths)} Python files for cross-file analysis with {max_workers} workers")
        results = self._parse_files_parallel(file_paths, max_workers)
        
        # Build global symbol registry from all parsed results
        for file_path, result in results.items():
            if result.symbols:  # Only process files that parsed successfully
                self._extract_global_symbols(file_path, result)
        
        # Second pass: Enhance relationships with cross-file resolution
        logger.info("Enhancing relationships with cross-file resolution")
        for file_path, result in results.items():
            if result.symbols:  # Only process files that parsed successfully
                self._enhance_cross_file_relationships(file_path, result)
        
        return results
    
    def _parse_files_parallel(self, file_paths: List[str], max_workers: int) -> Dict[str, ParseResult]:
        """
        Parse multiple files in parallel using ProcessPoolExecutor.
        
        Args:
            file_paths: List of file paths to parse
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping file path to ParseResult
        """
        import concurrent.futures
        import time
        
        results = {}
        
        # Use ProcessPoolExecutor for CPU-bound AST parsing
        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(_parse_single_python_file, file_path): file_path 
                             for file_path in file_paths}
            
            # Collect results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_file):
                file_path, result = future.result()
                results[file_path] = result
                
                completed += 1
                if completed % 20 == 0:  # Log progress every 20 files
                    logger.info(f"Parsed {completed}/{len(file_paths)} Python files...")
        
        parse_time = time.time() - start_time
        successful_files = len([r for r in results.values() if r.symbols])
        logger.info(f"Parallel Python parsing complete: {successful_files}/{len(file_paths)} files parsed in {parse_time:.2f}s")
        
        return results
    
    def _extract_global_symbols(self, file_path: str, result: ParseResult):
        """Extract symbols for global cross-file resolution."""
        file_name = Path(file_path).stem
        
        # Build import mapping for this file
        imports = {}
        for rel in result.relationships:
            if rel.relationship_type == RelationshipType.IMPORTS:
                # Handle different import patterns
                if "." in rel.target_symbol:
                    # from module.submodule import Symbol or import module.Symbol
                    parts = rel.target_symbol.split(".")
                    symbol_name = parts[-1]
                    imports[symbol_name] = rel.target_symbol
                else:
                    # Simple import: import Symbol
                    imports[rel.target_symbol] = rel.target_symbol
        
        # Store class and function locations globally
        for symbol in result.symbols:
            if symbol.symbol_type == SymbolType.CLASS:
                # Store class location
                self._global_class_locations[symbol.name] = file_path
                
                # Also store in general registry with file qualifier
                self._global_symbol_registry[symbol.name] = f"{file_name}.{symbol.name}"
                
            elif symbol.symbol_type == SymbolType.FUNCTION:
                # Store function location
                self._global_function_locations[symbol.name] = file_path
                
                # Also store in general registry with file qualifier
                self._global_symbol_registry[symbol.name] = f"{file_name}.{symbol.name}"
            
            elif symbol.symbol_type in [SymbolType.VARIABLE, SymbolType.CONSTANT]:
                # Store variables and constants
                self._global_symbol_registry[symbol.name] = f"{file_name}.{symbol.name}"
        
        # Store imported symbols with their file locations
        for local_name, full_name in imports.items():
            # Try to resolve the import to an actual file
            if "." in full_name:
                parts = full_name.split(".")
                module_name = parts[0]
                
                # Check if this corresponds to one of our files
                for other_file_path in [fp for fp in self._global_class_locations.values() 
                                      if Path(fp).stem == module_name]:
                    self._global_symbol_registry[local_name] = f"{module_name}.{local_name}"
    
    def _enhance_cross_file_relationships(self, file_path: str, result: ParseResult):
        """Enhance relationships with cross-file information."""
        enhanced_relationships = []
        file_name = Path(file_path).stem
        
        # Build imports map for this file
        imports = {}
        for rel in result.relationships:
            if rel.relationship_type == RelationshipType.IMPORTS:
                if "." in rel.target_symbol:
                    parts = rel.target_symbol.split(".")
                    symbol_name = parts[-1]
                    imports[symbol_name] = rel.target_symbol
                else:
                    imports[rel.target_symbol] = rel.target_symbol
        
        for rel in result.relationships:
            enhanced_rel = self._enhance_relationship(rel, file_path, imports)
            enhanced_relationships.append(enhanced_rel)
        
        # Replace with enhanced relationships
        result.relationships.clear()
        result.relationships.extend(enhanced_relationships)
    
    def _enhance_relationship(self, rel: Relationship, source_file: str, imports: Dict[str, str]) -> Relationship:
        """Enhance a single relationship with cross-file target information."""
        
        # For inheritance relationships, try to resolve target to actual file
        if rel.relationship_type == RelationshipType.INHERITANCE:
            target_symbol = rel.target_symbol
            
            # Check if the target is an imported symbol
            if target_symbol in imports:
                # This is an imported class being inherited from
                import_path = imports[target_symbol]
                
                # Try to find the actual file that defines this class
                if target_symbol in self._global_class_locations:
                    target_file = self._global_class_locations[target_symbol]
                    
                    # Update relationship with target file information
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=target_symbol,
                        relationship_type=rel.relationship_type,
                        source_file=rel.source_file,
                        target_file=target_file,
                        source_range=rel.source_range,
                        confidence=rel.confidence,
                        weight=rel.weight,
                        context=rel.context,
                        annotations={**rel.annotations, 'cross_file': True, 'import_path': import_path}
                    )
        
        # For calls and references, try to resolve targets
        elif rel.relationship_type in [RelationshipType.CALLS, RelationshipType.REFERENCES]:
            target_symbol = rel.target_symbol
            
            # Check if target is a known symbol from another file
            if target_symbol in self._global_class_locations:
                    target_file = self._global_class_locations[target_symbol]
                    if target_file != source_file:  # Only if it's actually cross-file
                        return Relationship(
                            source_symbol=rel.source_symbol,
                            target_symbol=target_symbol,
                            relationship_type=rel.relationship_type,
                            source_file=rel.source_file,
                            target_file=target_file,
                            source_range=rel.source_range,
                            confidence=rel.confidence,
                            weight=rel.weight,
                            context=rel.context,
                            annotations={**rel.annotations, 'cross_file': True}
                        )
            
            elif target_symbol in self._global_function_locations:
                target_file = self._global_function_locations[target_symbol]
                if target_file != source_file:  # Only if it's actually cross-file
                    return Relationship(
                        source_symbol=rel.source_symbol,
                        target_symbol=target_symbol,
                        relationship_type=rel.relationship_type,
                        source_file=rel.source_file,
                        target_file=target_file,
                        source_range=rel.source_range,
                        confidence=rel.confidence,
                        weight=rel.weight,
                        context=rel.context,
                        annotations={**rel.annotations, 'cross_file': True}
                    )
        
        # Return the original relationship if no enhancement was needed
        return rel
