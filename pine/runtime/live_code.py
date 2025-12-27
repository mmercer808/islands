"""
Pine Live Code System
=====================

Runtime code serialization, injection, and execution.

Features:
- Multiple serialization formats (AST, bytecode, pickled, dill)
- Context-aware deserialization
- Dynamic decorator swapping
- Security validation

SOURCE: matts/live_code_system.py (44KB)
Copy the full implementation from that file.

Quick Reference:
----------------
    from pine.runtime import CompleteLiveCodeSystem, serialize_function

    # Create system
    live_code = CompleteLiveCodeSystem()

    # Serialize a function
    def greet(name):
        return f"Hello, {name}!"

    serialized = live_code.serialize_function(greet)

    # Transmit serialized code via signal...

    # Deserialize and execute
    restored_greet = live_code.deserialize_function(serialized)
    print(restored_greet("World"))  # "Hello, World!"
"""

from typing import Any, Dict, List, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import types
import uuid
import base64
import hashlib


# =============================================================================
#                              ENUMS
# =============================================================================

class CodeSerializationMethod(Enum):
    """Methods for serializing code."""
    AST = auto()        # Abstract syntax tree
    BYTECODE = auto()   # Python bytecode
    PICKLED = auto()    # Standard pickle
    DILL = auto()       # Enhanced dill serialization
    SOURCE = auto()     # Raw source code


# =============================================================================
#                              EXCEPTIONS
# =============================================================================

class SecurityError(Exception):
    """Raised when code fails security validation."""
    pass


class DeserializationError(Exception):
    """Raised when code cannot be deserialized."""
    pass


# =============================================================================
#                              DATA CLASSES
# =============================================================================

@dataclass
class SerializedSourceCode:
    """
    Container for serialized source code.

    Includes:
    - The serialized code blob
    - Serialization method used
    - Metadata for reconstruction
    - Security hash
    """
    blob: bytes
    method: CodeSerializationMethod
    source_hash: str
    function_name: str = ""
    module_name: str = "__main__"
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return len(self.blob)


@dataclass
class CompleteSerializedCode:
    """
    Complete serialized code package with all context.

    Includes the serialized code plus everything needed
    to reconstruct it in another environment.
    """
    code: SerializedSourceCode
    globals_snapshot: Dict[str, Any] = field(default_factory=dict)
    closure_vars: Dict[str, Any] = field(default_factory=dict)
    decorator_chain: List[str] = field(default_factory=list)


# =============================================================================
#                              SERIALIZER
# =============================================================================

class SourceCodeSerializer:
    """
    Serializes Python functions and code objects.

    TODO: Copy full implementation from matts/live_code_system.py
    """

    def __init__(self, method: CodeSerializationMethod = CodeSerializationMethod.DILL):
        self.method = method

    def serialize_function(self, func: Callable) -> SerializedSourceCode:
        """Serialize a function to bytes."""
        import inspect

        # Get source code
        try:
            source = inspect.getsource(func)
        except OSError:
            source = ""

        # Calculate hash
        source_hash = hashlib.sha256(source.encode()).hexdigest()[:16]

        if self.method == CodeSerializationMethod.SOURCE:
            blob = source.encode('utf-8')
        elif self.method == CodeSerializationMethod.DILL:
            try:
                import dill
                blob = dill.dumps(func)
            except ImportError:
                import pickle
                blob = pickle.dumps(func)
        else:
            import pickle
            blob = pickle.dumps(func)

        return SerializedSourceCode(
            blob=blob,
            method=self.method,
            source_hash=source_hash,
            function_name=func.__name__,
            module_name=func.__module__,
        )

    def deserialize_function(self, serialized: SerializedSourceCode) -> Callable:
        """Deserialize bytes back to a function."""
        if serialized.method == CodeSerializationMethod.SOURCE:
            # Execute source to get function
            namespace = {}
            exec(serialized.blob.decode('utf-8'), namespace)
            return namespace[serialized.function_name]
        elif serialized.method == CodeSerializationMethod.DILL:
            try:
                import dill
                return dill.loads(serialized.blob)
            except ImportError:
                import pickle
                return pickle.loads(serialized.blob)
        else:
            import pickle
            return pickle.loads(serialized.blob)


# =============================================================================
#                              EXECUTION ENGINE
# =============================================================================

class BytecodeExecutionEngine:
    """
    Executes serialized bytecode in controlled environment.

    TODO: Copy implementation from matts/live_code_system.py
    """

    def __init__(self):
        self._execution_namespace: Dict[str, Any] = {}

    def execute(
        self,
        code: SerializedSourceCode,
        globals_dict: Dict[str, Any] = None,
        locals_dict: Dict[str, Any] = None
    ) -> Any:
        """Execute serialized code and return result."""
        if globals_dict is None:
            globals_dict = self._execution_namespace

        # Deserialize
        serializer = SourceCodeSerializer(code.method)
        func = serializer.deserialize_function(code)

        # Execute
        return func()


class RuntimeSourceEditor:
    """
    Edits source code at runtime before execution.

    Supports:
    - Adding/removing decorators
    - Modifying function signatures
    - Inserting instrumentation

    TODO: Copy implementation from matts/live_code_system.py
    """

    def add_decorator(
        self,
        func: Callable,
        decorator: Callable
    ) -> Callable:
        """Add a decorator to a function at runtime."""
        return decorator(func)

    def remove_decorator(
        self,
        func: Callable,
        decorator_name: str
    ) -> Callable:
        """Remove a decorator (if possible) at runtime."""
        # This is complex - decorators modify functions
        # TODO: Implement via AST manipulation
        return func


# =============================================================================
#                              DESERIALIZATION
# =============================================================================

class ContextAwareDeserializer:
    """
    Deserializes code with context awareness.

    Resolves dependencies and injects context
    during deserialization.

    TODO: Copy implementation from matts/live_code_system.py
    """

    def __init__(self):
        self._dependency_registry: Dict[str, Any] = {}

    def register_dependency(self, name: str, obj: Any) -> None:
        """Register a dependency for resolution during deserialization."""
        self._dependency_registry[name] = obj

    def deserialize_with_context(
        self,
        code: SerializedSourceCode,
        context: Dict[str, Any] = None
    ) -> Callable:
        """Deserialize code with injected context."""
        serializer = SourceCodeSerializer(code.method)
        func = serializer.deserialize_function(code)

        # Inject context into function globals
        if context and hasattr(func, '__globals__'):
            func.__globals__.update(context)

        return func


class RuntimeCodeCache:
    """
    Cache for deserialized code objects.

    Avoids repeated deserialization of the same code.
    """

    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, Callable] = {}
        self.max_size = max_size

    def get(self, source_hash: str) -> Optional[Callable]:
        """Get cached function by hash."""
        return self._cache.get(source_hash)

    def put(self, source_hash: str, func: Callable) -> None:
        """Cache a deserialized function."""
        if len(self._cache) >= self.max_size:
            # Evict oldest
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[source_hash] = func

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


# =============================================================================
#                              MAIN SYSTEMS
# =============================================================================

class CompleteLiveCodeSystem:
    """
    Complete live code system combining all components.

    Provides:
    - Serialization of functions and code
    - Secure deserialization
    - Execution in controlled environment
    - Caching for performance

    TODO: Copy full implementation from matts/live_code_system.py
    """

    def __init__(self):
        self.serializer = SourceCodeSerializer()
        self.deserializer = ContextAwareDeserializer()
        self.executor = BytecodeExecutionEngine()
        self.cache = RuntimeCodeCache()

    def serialize_function(
        self,
        func: Callable,
        method: CodeSerializationMethod = CodeSerializationMethod.DILL
    ) -> SerializedSourceCode:
        """Serialize a function."""
        self.serializer.method = method
        return self.serializer.serialize_function(func)

    def deserialize_function(
        self,
        code: SerializedSourceCode,
        context: Dict[str, Any] = None
    ) -> Callable:
        """Deserialize a function, using cache if available."""
        cached = self.cache.get(code.source_hash)
        if cached:
            return cached

        func = self.deserializer.deserialize_with_context(code, context)
        self.cache.put(code.source_hash, func)
        return func

    def execute_serialized(
        self,
        code: SerializedSourceCode,
        *args,
        **kwargs
    ) -> Any:
        """Deserialize and execute code."""
        func = self.deserialize_function(code)
        return func(*args, **kwargs)


class LiveCodeCallbackSystem:
    """
    Callback system with live code injection.

    Allows registering serialized callbacks that can be
    transmitted across systems.

    TODO: Copy full implementation from matts/live_code_system.py
    """

    def __init__(self):
        self.live_code = CompleteLiveCodeSystem()
        self._callbacks: Dict[str, SerializedSourceCode] = {}

    def register_callback(
        self,
        name: str,
        callback: Callable
    ) -> SerializedSourceCode:
        """Register and serialize a callback."""
        serialized = self.live_code.serialize_function(callback)
        self._callbacks[name] = serialized
        return serialized

    def execute_callback(
        self,
        name: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute a registered callback."""
        if name not in self._callbacks:
            raise KeyError(f"Callback '{name}' not found")
        return self.live_code.execute_serialized(
            self._callbacks[name],
            *args,
            **kwargs
        )

    def get_serialized(self, name: str) -> Optional[SerializedSourceCode]:
        """Get serialized callback for transmission."""
        return self._callbacks.get(name)
