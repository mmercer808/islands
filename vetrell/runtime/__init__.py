#!/usr/bin/env python3
"""
Vetrell Runtime Module

Live code injection, hot-swapping, and bytecode manipulation systems.

Submodules:
    live_code - Source code serialization and runtime injection
    hotswap - Runtime function and handler hot-swapping
    bytecode - Direct bytecode manipulation and editing
"""

from .live_code import (
    CodeSerializationMethod,
    SerializedSourceCode,
    CompleteSerializedCode,
    SourceCodeSerializer,
    ContextAwareDeserializer,
    BytecodeExecutionEngine,
    RuntimeSourceEditor,
    RuntimeCodeCache,
    CompleteLiveCodeSystem,
    LiveCodeCallbackSystem,
    SecurityError,
    DeserializationError,
)

from .hotswap import (
    BytecodeInstruction,
    BytecodeEditor,
    RuntimeFunctionRegistry,
    EntitySystem,
    DecoratorHotSwapSystem,
    CodeDistributionSignal,
    NetworkCodeDistributor,
    LiveUpdateSignalHandler,
)

__all__ = [
    # Live code
    "CodeSerializationMethod",
    "SerializedSourceCode",
    "CompleteSerializedCode",
    "SourceCodeSerializer",
    "ContextAwareDeserializer",
    "BytecodeExecutionEngine",
    "RuntimeSourceEditor",
    "RuntimeCodeCache",
    "CompleteLiveCodeSystem",
    "LiveCodeCallbackSystem",
    "SecurityError",
    "DeserializationError",
    # Hotswap
    "BytecodeInstruction",
    "BytecodeEditor",
    "RuntimeFunctionRegistry",
    "EntitySystem",
    "DecoratorHotSwapSystem",
    "CodeDistributionSignal",
    "NetworkCodeDistributor",
    "LiveUpdateSignalHandler",
]
