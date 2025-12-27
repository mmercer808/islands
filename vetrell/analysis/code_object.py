#!/usr/bin/env python3
"""
Comprehensive Python Process Code Object Metadata Utility

A complete utility that extracts ALL possible information from Python code objects,
running processes, frame stacks, modules, and execution state. Goes far beyond
just bytecode to capture the entire Python runtime state.
"""

import ast
import dis
import inspect
import types
import sys
import os
import gc
import threading
import traceback
import hashlib
import uuid
import linecache
import tokenize
import io
import marshal
import pickle
import importlib
import weakref
import ctypes
import platform
import psutil
import resource
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum, auto
import collections.abc


class CodeObjectType(Enum):
    """All possible types of code objects and callable entities."""
    # Core function types
    FUNCTION = "function"
    METHOD = "method"
    BOUND_METHOD = "bound_method"
    UNBOUND_METHOD = "unbound_method"
    STATIC_METHOD = "static_method"
    CLASS_METHOD = "class_method"
    
    # Generator types
    GENERATOR_FUNCTION = "generator_function"
    GENERATOR = "generator"
    ASYNC_GENERATOR_FUNCTION = "async_generator_function"
    ASYNC_GENERATOR = "async_generator"
    
    # Coroutine types
    COROUTINE_FUNCTION = "coroutine_function"
    COROUTINE = "coroutine"
    AWAITABLE = "awaitable"
    
    # Lambda and anonymous
    LAMBDA = "lambda"
    PARTIAL = "partial"
    
    # Class and module types
    CLASS = "class"
    METACLASS = "metaclass"
    MODULE = "module"
    PACKAGE = "package"
    
    # Special method types
    PROPERTY = "property"
    DESCRIPTOR = "descriptor"
    METHOD_DESCRIPTOR = "method_descriptor"
    MEMBER_DESCRIPTOR = "member_descriptor"
    GETSET_DESCRIPTOR = "getset_descriptor"
    DATA_DESCRIPTOR = "data_descriptor"
    
    # Built-in types
    BUILTIN_FUNCTION = "builtin_function"
    BUILTIN_METHOD = "builtin_method"
    METHOD_WRAPPER = "method_wrapper"
    WRAPPER_DESCRIPTOR = "wrapper_descriptor"
    
    # Comprehension types
    LIST_COMPREHENSION = "list_comprehension"
    DICT_COMPREHENSION = "dict_comprehension"
    SET_COMPREHENSION = "set_comprehension"
    GENERATOR_EXPRESSION = "generator_expression"
    
    # Frame and execution types
    FRAME = "frame"
    TRACEBACK = "traceback"
    CODE_OBJECT = "code_object"
    
    # Special execution contexts
    EXEC_CONTEXT = "exec_context"
    EVAL_CONTEXT = "eval_context"
    COMPILE_CONTEXT = "compile_context"
    
    # C extension types
    C_FUNCTION = "c_function"
    C_METHOD = "c_method"
    C_EXTENSION = "c_extension"
    
    # Thread and process types
    THREAD_LOCAL = "thread_local"
    THREAD_FUNCTION = "thread_function"
    
    # Abstract and special
    ABSTRACT_METHOD = "abstract_method"
    CACHED_PROPERTY = "cached_property"
    SLOT_WRAPPER = "slot_wrapper"
    
    UNKNOWN = "unknown"


@dataclass
class ProcessInfo:
    """Information about the current Python process."""
    pid: int = 0
    ppid: int = 0
    name: str = ""
    cmdline: List[str] = field(default_factory=list)
    cwd: str = ""
    exe: str = ""
    memory_info: Dict[str, Any] = field(default_factory=dict)
    cpu_info: Dict[str, Any] = field(default_factory=dict)
    thread_count: int = 0
    fd_count: int = 0
    connections: List[Dict[str, Any]] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ThreadInfo:
    """Information about a thread."""
    thread_id: int = 0
    name: str = ""
    daemon: bool = False
    alive: bool = False
    ident: Optional[int] = None
    native_id: Optional[int] = None
    frame_stack: List[Dict[str, Any]] = field(default_factory=list)
    local_vars: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'thread_id': self.thread_id,
            'name': self.name,
            'daemon': self.daemon,
            'alive': self.alive,
            'ident': self.ident,
            'native_id': self.native_id,
            'frame_stack': self.frame_stack,
            'local_vars': {k: repr(v) for k, v in self.local_vars.items()}
        }


@dataclass
class MemoryInfo:
    """Memory usage and garbage collection information."""
    total_objects: int = 0
    gc_stats: List[Dict[str, Any]] = field(default_factory=list)
    memory_usage: Dict[str, Any] = field(default_factory=dict)
    reference_counts: Dict[str, int] = field(default_factory=dict)
    weak_references: int = 0
    cyclic_references: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ModuleInfo:
    """Comprehensive module information."""
    name: str = ""
    file_path: Optional[str] = None
    package: Optional[str] = None
    loader: Optional[str] = None
    spec: Optional[str] = None
    doc: Optional[str] = None
    version: Optional[str] = None
    all_exports: List[str] = field(default_factory=list)
    submodules: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    size_bytes: int = 0
    is_package: bool = False
    is_builtin: bool = False
    is_frozen: bool = False
    is_extension: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = self.__dict__.copy()
        result['dependencies'] = list(result['dependencies'])
        return result


@dataclass
class FrameInfo:
    """Enhanced frame information."""
    frame_id: str = ""
    filename: str = ""
    function_name: str = ""
    line_number: int = 0
    code_context: List[str] = field(default_factory=list)
    local_vars: Dict[str, Any] = field(default_factory=dict)
    global_vars: Dict[str, Any] = field(default_factory=dict)
    builtin_vars: Dict[str, Any] = field(default_factory=dict)
    free_vars: Dict[str, Any] = field(default_factory=dict)
    cell_vars: Dict[str, Any] = field(default_factory=dict)
    last_instruction: int = 0
    is_executing: bool = False
    exception_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame_id': self.frame_id,
            'filename': self.filename,
            'function_name': self.function_name,
            'line_number': self.line_number,
            'code_context': self.code_context,
            'local_vars': {k: repr(v) for k, v in self.local_vars.items()},
            'global_vars': {k: repr(v) for k, v in self.global_vars.items()},
            'builtin_vars': {k: repr(v) for k, v in self.builtin_vars.items()},
            'free_vars': {k: repr(v) for k, v in self.free_vars.items()},
            'cell_vars': {k: repr(v) for k, v in self.cell_vars.items()},
            'last_instruction': self.last_instruction,
            'is_executing': self.is_executing,
            'exception_info': self.exception_info
        }


@dataclass
class ParameterInfo:
    """Information about function parameters."""
    name: str
    kind: str
    default: Any = inspect.Parameter.empty
    annotation: Any = inspect.Parameter.empty
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'kind': self.kind,
            'default': repr(self.default) if self.default is not inspect.Parameter.empty else None,
            'annotation': repr(self.annotation) if self.annotation is not inspect.Parameter.empty else None
        }


@dataclass
class SignatureInfo:
    """Function signature information."""
    parameters: List[ParameterInfo] = field(default_factory=list)
    return_annotation: Any = inspect.Signature.empty
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'parameters': [p.to_dict() for p in self.parameters],
            'return_annotation': repr(self.return_annotation) if self.return_annotation is not inspect.Signature.empty else None
        }


@dataclass
class ASTMetadata:
    """Comprehensive AST analysis metadata."""
    node_types: Set[str] = field(default_factory=set)
    function_names: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    variable_names: Set[str] = field(default_factory=set)
    imported_modules: List[str] = field(default_factory=list)
    imported_names: List[str] = field(default_factory=list)
    global_names: Set[str] = field(default_factory=set)
    nonlocal_names: Set[str] = field(default_factory=set)
    nested_functions: List[str] = field(default_factory=list)
    nested_classes: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    
    # Advanced AST analysis
    literal_values: List[Any] = field(default_factory=list)
    string_literals: List[str] = field(default_factory=list)
    numeric_literals: List[Union[int, float]] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    attribute_accesses: List[str] = field(default_factory=list)
    subscript_accesses: List[str] = field(default_factory=list)
    
    # Control flow analysis
    complexity_score: int = 0
    max_nesting_depth: int = 0
    loop_count: int = 0
    conditional_count: int = 0
    exception_handlers: int = 0
    context_managers: int = 0
    yield_expressions: int = 0
    await_expressions: int = 0
    
    # Security analysis
    exec_calls: int = 0
    eval_calls: int = 0
    import_calls: int = 0
    file_operations: int = 0
    network_operations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, set):
                result[k] = list(v)
            else:
                result[k] = v
        return result


@dataclass
class BytecodeInfo:
    """Enhanced bytecode analysis information."""
    instructions: List[Dict[str, Any]] = field(default_factory=list)
    opnames: Set[str] = field(default_factory=set)
    jump_targets: Set[int] = field(default_factory=set)
    exception_table: List[Dict[str, Any]] = field(default_factory=list)
    stack_size: int = 0
    
    # Advanced bytecode analysis
    instruction_count: int = 0
    unique_opcodes: int = 0
    jump_instructions: int = 0
    load_instructions: int = 0
    store_instructions: int = 0
    call_instructions: int = 0
    
    # Performance characteristics
    estimated_execution_time: float = 0.0
    memory_footprint: int = 0
    optimization_level: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        result = self.__dict__.copy()
        result['opnames'] = list(result['opnames'])
        result['jump_targets'] = list(result['jump_targets'])
        return result


@dataclass
class ComprehensiveCodeObjectMetadata:
    """The most complete metadata for any Python code object or entity."""
    
    # Basic identification
    object_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    qualified_name: str = ""
    code_type: CodeObjectType = CodeObjectType.UNKNOWN
    
    # Source information
    source_file: Optional[str] = None
    source_lines: Optional[List[str]] = None
    line_number: int = 0
    end_line_number: int = 0
    source_code: Optional[str] = None
    source_hash: str = ""
    
    # Code object attributes (from types.CodeType)
    co_argcount: int = 0
    co_posonlyargcount: int = 0
    co_kwonlyargcount: int = 0
    co_nlocals: int = 0
    co_stacksize: int = 0
    co_flags: int = 0
    co_code: bytes = b""
    co_consts: Tuple[Any, ...] = ()
    co_names: Tuple[str, ...] = ()
    co_varnames: Tuple[str, ...] = ()
    co_filename: str = ""
    co_name: str = ""
    co_firstlineno: int = 0
    co_lnotab: bytes = b""
    co_freevars: Tuple[str, ...] = ()
    co_cellvars: Tuple[str, ...] = ()
    
    # Extended code object attributes (newer Python versions)
    co_linetable: Optional[bytes] = None
    co_exceptiontable: Optional[bytes] = None
    co_qualname: Optional[str] = None
    
    # Function/method specific
    signature: Optional[SignatureInfo] = None
    is_method: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_property: bool = False
    is_coroutine: bool = False
    is_generator: bool = False
    is_async_generator: bool = False
    is_builtin: bool = False
    is_user_defined: bool = True
    
    # Class specific
    mro: List[str] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)
    metaclass: Optional[str] = None
    class_dict: Dict[str, Any] = field(default_factory=dict)
    
    # Module information
    module_info: Optional[ModuleInfo] = None
    
    # Documentation
    docstring: Optional[str] = None
    comments: Optional[str] = None
    
    # Analysis metadata
    ast_metadata: Optional[ASTMetadata] = None
    bytecode_info: Optional[BytecodeInfo] = None
    
    # Runtime information
    frame_info: Optional[FrameInfo] = None
    thread_info: Optional[ThreadInfo] = None
    process_info: Optional[ProcessInfo] = None
    memory_info: Optional[MemoryInfo] = None
    
    # Dependencies and context
    required_globals: Set[str] = field(default_factory=set)
    closure_vars: Dict[str, Any] = field(default_factory=dict)
    free_vars: Set[str] = field(default_factory=set)
    cell_vars: Set[str] = field(default_factory=set)
    weak_references: List[str] = field(default_factory=list)
    
    # Security and validation
    complexity_score: int = 0
    security_flags: List[str] = field(default_factory=list)
    
    # Performance metrics
    call_count: int = 0
    total_time: float = 0.0
    cumulative_time: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    python_version: str = field(default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}")
    platform_info: str = field(default_factory=lambda: platform.platform())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, (set, tuple)):
                result[field_name] = list(field_value)
            elif isinstance(field_value, bytes):
                result[field_name] = field_value.hex() if field_value else ""
            elif isinstance(field_value, CodeObjectType):
                result[field_name] = field_value.value
            elif hasattr(field_value, 'to_dict'):
                result[field_name] = field_value.to_dict()
            elif isinstance(field_value, datetime):
                result[field_name] = field_value.isoformat()
            elif isinstance(field_value, type):
                result[field_name] = f"{field_value.__module__}.{field_value.__name__}"
            else:
                try:
                    result[field_name] = field_value
                except:
                    result[field_name] = repr(field_value)
        return result


class CodeObjectContext(ABC):
    """Abstract base class for code object contexts."""
    
    @abstractmethod
    def get_context_data(self) -> Dict[str, Any]:
        """Get context-specific data."""
        pass
    
    @abstractmethod
    def validate_code_object(self, metadata: ComprehensiveCodeObjectMetadata) -> bool:
        """Validate if code object is compatible with this context."""
        pass
    
    @abstractmethod
    def prepare_execution_environment(self, metadata: ComprehensiveCodeObjectMetadata) -> Dict[str, Any]:
        """Prepare environment for code execution."""
        pass


class ComprehensivePythonProcessAnalyzer:
    """Analyzes ALL aspects of Python processes and code objects."""
    
    def __init__(self, include_process_info: bool = True, include_thread_info: bool = True):
        self.cache: Dict[str, ComprehensiveCodeObjectMetadata] = {}
        self.include_process_info = include_process_info
        self.include_thread_info = include_thread_info
        
        # Process information
        self.process = psutil.Process() if include_process_info else None
        
    def analyze_everything(self, obj: Any = None) -> ComprehensiveCodeObjectMetadata:
        """
        Analyze EVERYTHING about a code object and the current Python process.
        If obj is None, analyzes the current execution context.
        """
        if obj is None:
            # Analyze current frame
            frame = sys._getframe(1)
            obj = frame
        
        metadata = ComprehensiveCodeObjectMetadata()
        
        # Generate cache key
        cache_key = self._generate_cache_key(obj)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Basic object analysis
        self._extract_basic_info(obj, metadata)
        
        # Extract code object if available
        code_obj = self._get_code_object(obj)
        if code_obj:
            self._extract_code_object_info(code_obj, metadata)
        
        # Source code analysis
        self._extract_source_info(obj, metadata)
        
        # Signature analysis for callables
        if callable(obj):
            self._extract_signature_info(obj, metadata)
        
        # Class analysis
        if inspect.isclass(obj):
            self._extract_class_info(obj, metadata)
        
        # Module analysis
        self._extract_module_info(obj, metadata)
        
        # Documentation extraction
        self._extract_documentation(obj, metadata)
        
        # AST analysis
        if metadata.source_code:
            metadata.ast_metadata = self._analyze_ast_comprehensive(metadata.source_code)
        
        # Bytecode analysis
        if code_obj:
            metadata.bytecode_info = self._analyze_bytecode_comprehensive(code_obj)
        
        # Frame analysis
        if inspect.isframe(obj):
            metadata.frame_info = self._extract_frame_info(obj)
        
        # Runtime context analysis
        self._extract_closure_info(obj, metadata)
        
        # Process information
        if self.include_process_info:
            metadata.process_info = self._get_process_info()
        
        # Thread information
        if self.include_thread_info:
            metadata.thread_info = self._get_current_thread_info()
        
        # Memory analysis
        metadata.memory_info = self._get_memory_info()
        
        # Determine code type
        metadata.code_type = self._determine_comprehensive_code_type(obj)
        
        # Security analysis
        metadata.security_flags = self._analyze_security(obj, metadata)
        
        # Performance analysis
        self._extract_performance_info(obj, metadata)
        
        # Calculate complexity
        if metadata.ast_metadata:
            metadata.complexity_score = metadata.ast_metadata.complexity_score
        
        # Cache and return
        self.cache[cache_key] = metadata
        return metadata
    
    def _generate_cache_key(self, obj: Any) -> str:
        """Generate a cache key for the object."""
        try:
            if hasattr(obj, '__code__'):
                code = obj.__code__
                return f"{code.co_filename}:{code.co_name}:{code.co_firstlineno}:{id(obj)}"
            elif hasattr(obj, '__name__'):
                return f"{getattr(obj, '__module__', 'unknown')}:{obj.__name__}:{id(obj)}"
            else:
                return f"{type(obj).__name__}:{id(obj)}"
        except:
            return str(id(obj))
    
    def _extract_basic_info(self, obj: Any, metadata: ComprehensiveCodeObjectMetadata) -> None:
        """Extract basic object information."""
        metadata.name = getattr(obj, '__name__', str(type(obj).__name__))
        metadata.qualified_name = getattr(obj, '__qualname__', metadata.name)
        metadata.is_builtin = inspect.isbuiltin(obj)
        metadata.is_user_defined = not metadata.is_builtin
    
    def _get_code_object(self, obj: Any) -> Optional[types.CodeType]:
        """Extract code object from various object types."""
        if inspect.iscode(obj):
            return obj
        elif inspect.isfunction(obj):
            return obj.__code__
        elif inspect.ismethod(obj):
            return obj.__func__.__code__
        elif inspect.isframe(obj):
            return obj.f_code
        elif inspect.istraceback(obj):
            return obj.tb_frame.f_code
        elif hasattr(obj, '__code__'):
            return obj.__code__
        elif inspect.isgenerator(obj):
            return obj.gi_code
        elif inspect.iscoroutine(obj):
            return obj.cr_code
        elif hasattr(obj, 'gi_code'):
            return obj.gi_code
        elif hasattr(obj, 'cr_code'):
            return obj.cr_code
        return None
    
    def _extract_code_object_info(self, code_obj: types.CodeType, metadata: ComprehensiveCodeObjectMetadata) -> None:
        """Extract comprehensive information from code object attributes."""
        # Standard attributes
        metadata.co_argcount = code_obj.co_argcount
        metadata.co_posonlyargcount = getattr(code_obj, 'co_posonlyargcount', 0)
        metadata.co_kwonlyargcount = code_obj.co_kwonlyargcount
        metadata.co_nlocals = code_obj.co_nlocals
        metadata.co_stacksize = code_obj.co_stacksize
        metadata.co_flags = code_obj.co_flags
        metadata.co_code = code_obj.co_code
        metadata.co_consts = code_obj.co_consts
        metadata.co_names = code_obj.co_names
        metadata.co_varnames = code_obj.co_varnames
        metadata.co_filename = code_obj.co_filename
        metadata.co_name = code_obj.co_name
        metadata.co_firstlineno = code_obj.co_firstlineno
        metadata.co_freevars = code_obj.co_freevars
        metadata.co_cellvars = code_obj.co_cellvars
        
        # Newer Python version attributes
        metadata.co_lnotab = getattr(code_obj, 'co_lnotab', b'')
        metadata.co_linetable = getattr(code_obj, 'co_linetable', None)
        metadata.co_exceptiontable = getattr(code_obj, 'co_exceptiontable', None)
        metadata.co_qualname = getattr(code_obj, 'co_qualname', None)
        
        # Set derived attributes
        metadata.line_number = code_obj.co_firstlineno
        metadata.free_vars = set(code_obj.co_freevars)
        metadata.cell_vars = set(code_obj.co_cellvars)
        metadata.required_globals = set(code_obj.co_names)
        
        # Analyze code flags
        flags = code_obj.co_flags
        metadata.is_generator = bool(flags & inspect.CO_GENERATOR)
        metadata.is_coroutine = bool(flags & inspect.CO_COROUTINE)
        metadata.is_async_generator = bool(flags & getattr(inspect, 'CO_ASYNC_GENERATOR', 0))
    
    def _extract_source_info(self, obj: Any, metadata: ComprehensiveCodeObjectMetadata) -> None:
        """Extract source code information."""
        try:
            metadata.source_file = inspect.getfile(obj)
            metadata.source_lines, metadata.line_number = inspect.getsourcelines(obj)
            metadata.end_line_number = metadata.line_number + len(metadata.source_lines) - 1
            metadata.source_code = ''.join(metadata.source_lines)
            metadata.source_hash = hashlib.sha256(metadata.source_code.encode()).hexdigest()
        except (OSError, TypeError):
            # Try alternative methods for built-ins, etc.
            pass
    
    def _extract_signature_info(self, obj: Any, metadata: ComprehensiveCodeObjectMetadata) -> None:
        """Extract function signature information."""
        try:
            sig = inspect.signature(obj)
            parameters = []
            for param in sig.parameters.values():
                param_info = ParameterInfo(
                    name=param.name,
                    kind=param.kind.name,
                    default=param.default,
                    annotation=param.annotation
                )
                parameters.append(param_info)
            
            metadata.signature = SignatureInfo(
                parameters=parameters,
                return_annotation=sig.return_annotation
            )
            
            # Determine method types
            metadata.is_method = inspect.ismethod(obj)
            
        except (ValueError, TypeError):
            pass
    
    def _extract_class_info(self, obj: Any, metadata: ComprehensiveCodeObjectMetadata) -> None:
        """Extract class-specific information."""
        if inspect.isclass(obj):
            metadata.mro = [cls.__name__ for cls in obj.__mro__]
            metadata.bases = [base.__name__ for base in obj.__bases__]
            metadata.metaclass = type(obj).__name__
            
            # Extract class dictionary (safely)
            try:
                metadata.class_dict = {k: repr(v) for k, v in obj.__dict__.items() 
                                     if not k.startswith('_')}
            except:
                pass
    
    def _extract_module_info(self, obj: Any, metadata: ComprehensiveCodeObjectMetadata) -> None:
        """Extract comprehensive module information."""
        try:
            module = inspect.getmodule(obj)
            if module:
                module_info = ModuleInfo()
                module_info.name = module.__name__
                module_info.file_path = getattr(module, '__file__', None)
                module_info.package = getattr(module, '__package__', None)
                module_info.doc = getattr(module, '__doc__', None)
                module_info.version = getattr(module, '__version__', None)
                
                # Check module type
                module_info.is_builtin = module.__name__ in sys.builtin_module_names
                module_info.is_package = hasattr(module, '__path__')
                
                # Get all exports
                if hasattr(module, '__all__'):
                    module_info.all_exports = list(module.__all__)
                else:
                    module_info.all_exports = [name for name in dir(module) 
                                             if not name.startswith('_')]
                
                # Calculate size
                if module_info.file_path:
                    try:
                        module_info.size_bytes = os.path.getsize(module_info.file_path)
                    except:
                        pass
                
                metadata.module_info = module_info
        except:
            pass
    
    def _extract_documentation(self, obj: Any, metadata: ComprehensiveCodeObjectMetadata) -> None:
        """Extract documentation strings and comments."""
        metadata.docstring = inspect.getdoc(obj)
        try:
            metadata.comments = inspect.getcomments(obj)
        except:
            pass
    
    def _analyze_ast_comprehensive(self, source_code: str) -> ASTMetadata:
        """Perform the most comprehensive AST analysis possible."""
        ast_metadata = ASTMetadata()
        
        try:
            tree = ast.parse(source_code)
            
            class ComprehensiveASTAnalyzer(ast.NodeVisitor):
                def __init__(self, metadata: ASTMetadata):
                    self.metadata = metadata
                    self.depth = 0
                    self.max_depth = 0
                
                def visit(self, node):
                    self.metadata.node_types.add(type(node).__name__)
                    self.depth += 1
                    self.max_depth = max(self.max_depth, self.depth)
                    self.generic_visit(node)
                    self.depth -= 1
                
                def visit_FunctionDef(self, node):
                    self.metadata.function_names.append(node.name)
                    if hasattr(node, 'decorator_list'):
                        for decorator in node.decorator_list:
                            if isinstance(decorator, ast.Name):
                                self.metadata.decorators.append(decorator.id)
                    self.metadata.complexity_score += 1
                    self.generic_visit(node)
                
                def visit_AsyncFunctionDef(self, node):
                    self.metadata.function_names.append(node.name)
                    self.metadata.complexity_score += 1
                    self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    self.metadata.class_names.append(node.name)
                    self.metadata.complexity_score += 1
                    self.generic_visit(node)
                
                def visit_Import(self, node):
                    for alias in node.names:
                        self.metadata.imported_modules.append(alias.name)
                        if alias.asname:
                            self.metadata.imported_names.append(alias.asname)
                
                def visit_ImportFrom(self, node):
                    if node.module:
                        self.metadata.imported_modules.append(node.module)
                    for alias in node.names:
                        self.metadata.imported_names.append(alias.name)
                        if alias.asname:
                            self.metadata.imported_names.append(alias.asname)
                
                def visit_Name(self, node):
                    self.metadata.variable_names.add(node.id)
                
                def visit_Global(self, node):
                    self.metadata.global_names.update(node.names)
                
                def visit_Nonlocal(self, node):
                    self.metadata.nonlocal_names.update(node.names)
                
                def visit_Constant(self, node):
                    self.metadata.literal_values.append(node.value)
                    if isinstance(node.value, str):
                        self.metadata.string_literals.append(node.value)
                    elif isinstance(node.value, (int, float)):
                        self.metadata.numeric_literals.append(node.value)
                
                def visit_Str(self, node):  # Python < 3.8 compatibility
                    self.metadata.string_literals.append(node.s)
                    self.metadata.literal_values.append(node.s)
                
                def visit_Num(self, node):  # Python < 3.8 compatibility
                    self.metadata.numeric_literals.append(node.n)
                    self.metadata.literal_values.append(node.n)
                
                def visit_Call(self, node):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        self.metadata.function_calls.append(func_name)
                        
                        # Security analysis
                        if func_name in ('exec', 'eval'):
                            if func_name == 'exec':
                                self.metadata.exec_calls += 1
                            else:
                                self.metadata.eval_calls += 1
                        elif func_name == '__import__':
                            self.metadata.import_calls += 1
                        elif func_name in ('open', 'file'):
                            self.metadata.file_operations += 1
                        elif func_name in ('socket', 'urllib', 'requests'):
                            self.metadata.network_operations += 1
                    
                    elif isinstance(node.func, ast.Attribute):
                        self.metadata.function_calls.append(node.func.attr)
                    
                    self.generic_visit(node)
                
                def visit_Attribute(self, node):
                    if isinstance(node.value, ast.Name):
                        self.metadata.attribute_accesses.append(f"{node.value.id}.{node.attr}")
                    else:
                        self.metadata.attribute_accesses.append(node.attr)
                    self.generic_visit(node)
                
                def visit_Subscript(self, node):
                    if isinstance(node.value, ast.Name):
                        self.metadata.subscript_accesses.append(node.value.id)
                    self.generic_visit(node)
                
                def visit_If(self, node):
                    self.metadata.conditional_count += 1
                    self.metadata.complexity_score += 1
                    self.generic_visit(node)
                
                def visit_For(self, node):
                    self.metadata.loop_count += 1
                    self.metadata.complexity_score += 1
                    self.generic_visit(node)
                
                def visit_While(self, node):
                    self.metadata.loop_count += 1
                    self.metadata.complexity_score += 1
                    self.generic_visit(node)
                
                def visit_Try(self, node):
                    self.metadata.exception_handlers += 1
                    self.metadata.complexity_score += 1
                    self.generic_visit(node)
                
                def visit_With(self, node):
                    self.metadata.context_managers += 1
                    self.metadata.complexity_score += 1
                    self.generic_visit(node)
                
                def visit_Yield(self, node):
                    self.metadata.yield_expressions += 1
                    self.generic_visit(node)
                
                def visit_YieldFrom(self, node):
                    self.metadata.yield_expressions += 1
                    self.generic_visit(node)
                
                def visit_Await(self, node):
                    self.metadata.await_expressions += 1
                    self.generic_visit(node)
            
            analyzer = ComprehensiveASTAnalyzer(ast_metadata)
            analyzer.visit(tree)
            ast_metadata.max_nesting_depth = analyzer.max_depth
            
        except SyntaxError:
            pass
        
        return ast_metadata
    
    def _analyze_bytecode_comprehensive(self, code_obj: types.CodeType) -> BytecodeInfo:
        """Perform comprehensive bytecode analysis."""
        bytecode_info = BytecodeInfo()
        
        try:
            bytecode_info.stack_size = code_obj.co_stacksize
            
            # Detailed instruction analysis
            for instr in dis.Bytecode(code_obj):
                instr_info = {
                    'opname': instr.opname,
                    'opcode': instr.opcode,
                    'arg': instr.arg,
                    'argval': repr(instr.argval) if instr.argval is not None else None,
                    'argrepr': instr.argrepr,
                    'offset': instr.offset,
                    'starts_line': instr.starts_line,
                    'is_jump_target': instr.is_jump_target
                }
                bytecode_info.instructions.append(instr_info)
                bytecode_info.opnames.add(instr.opname)
                
                # Count instruction types
                if instr.is_jump_target:
                    bytecode_info.jump_targets.add(instr.offset)
                
                if 'JUMP' in instr.opname:
                    bytecode_info.jump_instructions += 1
                elif 'LOAD' in instr.opname:
                    bytecode_info.load_instructions += 1
                elif 'STORE' in instr.opname:
                    bytecode_info.store_instructions += 1
                elif 'CALL' in instr.opname:
                    bytecode_info.call_instructions += 1
            
            bytecode_info.instruction_count = len(bytecode_info.instructions)
            bytecode_info.unique_opcodes = len(bytecode_info.opnames)
            bytecode_info.memory_footprint = len(code_obj.co_code)
            
            # Estimate execution characteristics
            bytecode_info.estimated_execution_time = self._estimate_execution_time(bytecode_info)
            
        except Exception:
            pass
        
        return bytecode_info
    
    def _estimate_execution_time(self, bytecode_info: BytecodeInfo) -> float:
        """Estimate relative execution time based on bytecode."""
        # Simple heuristic based on instruction counts
        base_time = bytecode_info.instruction_count * 0.001
        complexity_factor = (
            bytecode_info.jump_instructions * 0.002 +
            bytecode_info.call_instructions * 0.005 +
            len(bytecode_info.jump_targets) * 0.001
        )
        return base_time + complexity_factor
    
    def _extract_frame_info(self, frame: types.FrameType) -> FrameInfo:
        """Extract comprehensive frame information."""
        frame_info = FrameInfo()
        
        frame_info.frame_id = str(id(frame))
        frame_info.filename = frame.f_code.co_filename
        frame_info.function_name = frame.f_code.co_name
        frame_info.line_number = frame.f_lineno
        frame_info.last_instruction = frame.f_lasti
        
        # Extract variables
        frame_info.local_vars = dict(frame.f_locals)
        frame_info.global_vars = dict(frame.f_globals)
        frame_info.builtin_vars = dict(frame.f_builtins)
        
        # Code context
        try:
            lines, start_line = inspect.getsourcelines(frame.f_code)
            current_line_idx = frame.f_lineno - start_line
            if 0 <= current_line_idx < len(lines):
                frame_info.code_context = lines[max(0, current_line_idx-2):current_line_idx+3]
        except:
            pass
        
        return frame_info
    
    def _extract_closure_info(self, obj: Any, metadata: ComprehensiveCodeObjectMetadata) -> None:
        """Extract closure variable information."""
        try:
            if inspect.isfunction(obj):
                closure_vars = inspect.getclosurevars(obj)
                metadata.closure_vars = {
                    'nonlocals': dict(closure_vars.nonlocals),
                    'globals': dict(closure_vars.globals),
                    'builtins': dict(closure_vars.builtins),
                    'unbound': set(closure_vars.unbound)
                }
        except (TypeError, AttributeError):
            pass
    
    def _get_process_info(self) -> ProcessInfo:
        """Get comprehensive process information."""
        process_info = ProcessInfo()
        
        if self.process:
            try:
                process_info.pid = self.process.pid
                process_info.ppid = self.process.ppid()
                process_info.name = self.process.name()
                process_info.cmdline = self.process.cmdline()
                process_info.cwd = self.process.cwd()
                process_info.exe = self.process.exe()
                
                # Memory information
                memory = self.process.memory_info()
                process_info.memory_info = {
                    'rss': memory.rss,
                    'vms': memory.vms,
                    'percent': self.process.memory_percent()
                }
                
                # CPU information
                process_info.cpu_info = {
                    'percent': self.process.cpu_percent(),
                    'times': self.process.cpu_times()._asdict()
                }
                
                process_info.thread_count = self.process.num_threads()
                process_info.fd_count = self.process.num_fds()
                
                # Network connections
                try:
                    connections = self.process.connections()
                    process_info.connections = [
                        {
                            'fd': conn.fd,
                            'family': conn.family.name,
                            'type': conn.type.name,
                            'laddr': conn.laddr,
                            'raddr': conn.raddr,
                            'status': conn.status
                        }
                        for conn in connections
                    ]
                except:
                    pass
                
                # Environment variables
                process_info.environment = dict(self.process.environ())
                
            except Exception:
                pass
        
        return process_info
    
    def _get_current_thread_info(self) -> ThreadInfo:
        """Get current thread information."""
        thread_info = ThreadInfo()
        
        current_thread = threading.current_thread()
        thread_info.thread_id = current_thread.ident or 0
        thread_info.name = current_thread.name
        thread_info.daemon = current_thread.daemon
        thread_info.alive = current_thread.is_alive()
        thread_info.ident = current_thread.ident
        thread_info.native_id = getattr(current_thread, 'native_id', None)
        
        # Get frame stack
        frame = sys._getframe()
        frame_stack = []
        while frame:
            frame_info = {
                'filename': frame.f_code.co_filename,
                'function': frame.f_code.co_name,
                'line': frame.f_lineno
            }
            frame_stack.append(frame_info)
            frame = frame.f_back
        
        thread_info.frame_stack = frame_stack
        
        return thread_info
    
    def _get_memory_info(self) -> MemoryInfo:
        """Get comprehensive memory and garbage collection information."""
        memory_info = MemoryInfo()
        
        # Total objects
        memory_info.total_objects = len(gc.get_objects())
        
        # GC stats
        memory_info.gc_stats = gc.get_stats()
        
        # Reference counts by type
        ref_counts = collections.defaultdict(int)
        for obj in gc.get_objects():
            ref_counts[type(obj).__name__] += 1
        memory_info.reference_counts = dict(ref_counts)
        
        # Weak references
        weak_refs = [obj for obj in gc.get_objects() if isinstance(obj, weakref.ref)]
        memory_info.weak_references = len(weak_refs)
        
        # Memory usage
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            memory_info.memory_usage = {
                'max_rss': usage.ru_maxrss,
                'shared_memory': getattr(usage, 'ru_ixrss', 0),
                'unshared_data': getattr(usage, 'ru_idrss', 0),
                'unshared_stack': getattr(usage, 'ru_isrss', 0)
            }
        except:
            pass
        
        return memory_info
    
    def _determine_comprehensive_code_type(self, obj: Any) -> CodeObjectType:
        """Determine the most specific type of code object."""
        # Frame and traceback types
        if inspect.isframe(obj):
            return CodeObjectType.FRAME
        elif inspect.istraceback(obj):
            return CodeObjectType.TRACEBACK
        elif inspect.iscode(obj):
            return CodeObjectType.CODE_OBJECT
        
        # Module types
        elif inspect.ismodule(obj):
            if hasattr(obj, '__path__'):
                return CodeObjectType.PACKAGE
            else:
                return CodeObjectType.MODULE
        
        # Class types
        elif inspect.isclass(obj):
            if isinstance(obj, type(type)):
                return CodeObjectType.METACLASS
            else:
                return CodeObjectType.CLASS
        
        # Function and method types
        elif inspect.ismethod(obj):
            if obj.__self__ is None:
                return CodeObjectType.UNBOUND_METHOD
            else:
                return CodeObjectType.BOUND_METHOD
        elif inspect.isfunction(obj):
            if inspect.isgeneratorfunction(obj):
                return CodeObjectType.GENERATOR_FUNCTION
            elif inspect.iscoroutinefunction(obj):
                return CodeObjectType.COROUTINE_FUNCTION
            elif inspect.isasyncgenfunction(obj):
                return CodeObjectType.ASYNC_GENERATOR_FUNCTION
            elif getattr(obj, '__name__', '') == '<lambda>':
                return CodeObjectType.LAMBDA
            else:
                return CodeObjectType.FUNCTION
        
        # Generator and coroutine instances
        elif inspect.isgenerator(obj):
            return CodeObjectType.GENERATOR
        elif inspect.iscoroutine(obj):
            return CodeObjectType.COROUTINE
        elif inspect.isasyncgen(obj):
            return CodeObjectType.ASYNC_GENERATOR
        elif inspect.isawaitable(obj):
            return CodeObjectType.AWAITABLE
        
        # Built-in types
        elif inspect.isbuiltin(obj):
            return CodeObjectType.BUILTIN_FUNCTION
        elif inspect.ismethoddescriptor(obj):
            return CodeObjectType.METHOD_DESCRIPTOR
        elif inspect.ismemberdescriptor(obj):
            return CodeObjectType.MEMBER_DESCRIPTOR
        elif inspect.isgetsetdescriptor(obj):
            return CodeObjectType.GETSET_DESCRIPTOR
        elif inspect.isdatadescriptor(obj):
            return CodeObjectType.DATA_DESCRIPTOR
        elif inspect.ismethodwrapper(obj):
            return CodeObjectType.METHOD_WRAPPER
        
        # Property and descriptor types
        elif isinstance(obj, property):
            return CodeObjectType.PROPERTY
        elif hasattr(obj, '__get__') or hasattr(obj, '__set__'):
            return CodeObjectType.DESCRIPTOR
        
        # Special types
        elif isinstance(obj, staticmethod):
            return CodeObjectType.STATIC_METHOD
        elif isinstance(obj, classmethod):
            return CodeObjectType.CLASS_METHOD
        elif isinstance(obj, functools.partial):
            return CodeObjectType.PARTIAL
        elif hasattr(obj, '__call__'):
            if type(obj).__name__ == 'method-wrapper':
                return CodeObjectType.METHOD_WRAPPER
            elif type(obj).__name__ == 'wrapper_descriptor':
                return CodeObjectType.WRAPPER_DESCRIPTOR
            elif type(obj).__name__ == 'slot wrapper':
                return CodeObjectType.SLOT_WRAPPER
        
        return CodeObjectType.UNKNOWN
    
    def _analyze_security(self, obj: Any, metadata: ComprehensiveCodeObjectMetadata) -> List[str]:
        """Analyze security implications of the code object."""
        flags = []
        
        # Check AST for dangerous operations
        if metadata.ast_metadata:
            if metadata.ast_metadata.exec_calls > 0:
                flags.append("EXEC_CALLS")
            if metadata.ast_metadata.eval_calls > 0:
                flags.append("EVAL_CALLS")
            if metadata.ast_metadata.import_calls > 0:
                flags.append("DYNAMIC_IMPORTS")
            if metadata.ast_metadata.file_operations > 0:
                flags.append("FILE_OPERATIONS")
            if metadata.ast_metadata.network_operations > 0:
                flags.append("NETWORK_OPERATIONS")
        
        # Check for dangerous built-ins in globals
        if metadata.required_globals:
            dangerous_builtins = {'exec', 'eval', 'compile', '__import__', 'open', 'file'}
            if metadata.required_globals & dangerous_builtins:
                flags.append("DANGEROUS_BUILTINS")
        
        # Check code flags
        if hasattr(obj, '__code__'):
            code = obj.__code__
            if code.co_flags & inspect.CO_GENERATOR:
                flags.append("GENERATOR")
            if code.co_flags & inspect.CO_COROUTINE:
                flags.append("COROUTINE")
        
        return flags
    
    def _extract_performance_info(self, obj: Any, metadata: ComprehensiveCodeObjectMetadata) -> None:
        """Extract performance-related information."""
        # This would integrate with profiling data if available
        if hasattr(obj, '__code__'):
            # Estimate based on bytecode complexity
            if metadata.bytecode_info:
                metadata.total_time = metadata.bytecode_info.estimated_execution_time
        
        # Could integrate with sys.settrace or cProfile data here
    
    def get_all_loaded_modules(self) -> Dict[str, ModuleInfo]:
        """Get information about all loaded modules."""
        modules = {}
        for name, module in sys.modules.items():
            if module is not None:
                try:
                    metadata = self.analyze_everything(module)
                    if metadata.module_info:
                        modules[name] = metadata.module_info
                except:
                    pass
        return modules
    
    def get_all_threads(self) -> List[ThreadInfo]:
        """Get information about all threads."""
        threads = []
        for thread in threading.enumerate():
            thread_info = ThreadInfo()
            thread_info.thread_id = thread.ident or 0
            thread_info.name = thread.name
            thread_info.daemon = thread.daemon
            thread_info.alive = thread.is_alive()
            thread_info.ident = thread.ident
            thread_info.native_id = getattr(thread, 'native_id', None)
            threads.append(thread_info)
        return threads
    
    def get_call_stack_analysis(self) -> List[ComprehensiveCodeObjectMetadata]:
        """Analyze the entire call stack."""
        stack_analysis = []
        frame = sys._getframe()
        
        while frame:
            try:
                metadata = self.analyze_everything(frame)
                stack_analysis.append(metadata)
            except:
                pass
            frame = frame.f_back
        
        return stack_analysis


# Helper functions for quick analysis
def analyze_comprehensive(obj: Any = None, **kwargs) -> ComprehensiveCodeObjectMetadata:
    """
    Quick helper function to perform comprehensive analysis.
    
    Args:
        obj: The object to analyze (if None, analyzes current execution context)
        **kwargs: Additional options for analysis
        
    Returns:
        ComprehensiveCodeObjectMetadata with ALL possible information
    """
    analyzer = ComprehensivePythonProcessAnalyzer(**kwargs)
    return analyzer.analyze_everything(obj)


def get_full_process_snapshot() -> Dict[str, Any]:
    """Get a complete snapshot of the entire Python process."""
    analyzer = ComprehensivePythonProcessAnalyzer()
    
    return {
        'process_info': analyzer._get_process_info().to_dict(),
        'memory_info': analyzer._get_memory_info().to_dict(),
        'all_threads': [t.to_dict() for t in analyzer.get_all_threads()],
        'all_modules': {k: v.to_dict() for k, v in analyzer.get_all_loaded_modules().items()},
        'call_stack': [m.to_dict() for m in analyzer.get_call_stack_analysis()],
        'gc_objects_count': len(gc.get_objects()),
        'python_version': sys.version,
        'platform': platform.platform(),
        'timestamp': datetime.now().isoformat()
    }


# Example usage and testing
if __name__ == "__main__":
    def example_async_generator(x: int) -> Iterator[str]:
        """Example async generator for comprehensive testing."""
        for i in range(x):
            result = f"Item {i}: {x}"
            yield result
    
    # Test comprehensive analysis
    metadata = analyze_comprehensive(example_async_generator)
    
    print(f"Code type: {metadata.code_type}")
    print(f"Complexity score: {metadata.complexity_score}")
    print(f"Security flags: {metadata.security_flags}")
    print(f"Process PID: {metadata.process_info.pid if metadata.process_info else 'N/A'}")
    print(f"Thread count: {metadata.process_info.thread_count if metadata.process_info else 'N/A'}")
    
    # Get full process snapshot
    snapshot = get_full_process_snapshot()
    print(f"\nTotal modules loaded: {len(snapshot['all_modules'])}")
    print(f"Total threads: {len(snapshot['all_threads'])}")
    print(f"Total GC objects: {snapshot['gc_objects_count']}")
    
    # Convert to dictionary for serialization
    full_metadata = metadata.to_dict()
    print(f"\nMetadata contains {len(full_metadata)} fields")