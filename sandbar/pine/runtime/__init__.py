"""
Pine Runtime - Code Execution Layer
====================================

Runtime systems for:
- Live code injection and serialization
- Hot-swapping without restart
- Bytecode manipulation
- Generator composition patterns

This layer depends on: pine.core

Source Files (to integrate):
- matts/live_code_system.py -> live_code.py
- runtime_hotswap_system.py -> hotswap.py
- code_object_utility.py -> bytecode.py
- matts/generator_system.py -> generators.py
"""

from .live_code import (
    # Core system
    CompleteLiveCodeSystem, LiveCodeCallbackSystem,
    # Serialization
    SerializedSourceCode, SourceCodeSerializer,
    CodeSerializationMethod, CompleteSerializedCode,
    # Execution
    BytecodeExecutionEngine, RuntimeSourceEditor,
    # Deserialization
    ContextAwareDeserializer, RuntimeCodeCache,
    # Exceptions
    SecurityError, DeserializationError,
)

from .hotswap import (
    # Core hotswap
    HotSwapManager, HotSwapHandler,
    # Bytecode editing
    BytecodeEditor, InstructionReorderer,
    # Distribution
    CodeDistributor,
)

from .bytecode import (
    # Analysis
    CodeObjectAnalyzer, BytecodeInspector,
    # Manipulation
    CodeObjectBuilder, InstructionModifier,
)

from .generators import (
    # Composition
    AdvancedGeneratorComposer, GeneratorCompositionPattern,
    GeneratorCompositionEngine,
    # Branching
    GeneratorBranch, GeneratorStateBranch,
    # Factories
    create_data_generator_factory,
    create_transformer_generator_factory,
    create_filter_generator_factory,
    create_aggregator_generator_factory,
)

__all__ = [
    # Live Code
    'CompleteLiveCodeSystem', 'LiveCodeCallbackSystem',
    'SerializedSourceCode', 'SourceCodeSerializer',
    'CodeSerializationMethod', 'CompleteSerializedCode',
    'BytecodeExecutionEngine', 'RuntimeSourceEditor',
    'ContextAwareDeserializer', 'RuntimeCodeCache',
    'SecurityError', 'DeserializationError',

    # Hotswap
    'HotSwapManager', 'HotSwapHandler',
    'BytecodeEditor', 'InstructionReorderer',
    'CodeDistributor',

    # Bytecode
    'CodeObjectAnalyzer', 'BytecodeInspector',
    'CodeObjectBuilder', 'InstructionModifier',

    # Generators
    'AdvancedGeneratorComposer', 'GeneratorCompositionPattern',
    'GeneratorCompositionEngine',
    'GeneratorBranch', 'GeneratorStateBranch',
    'create_data_generator_factory',
    'create_transformer_generator_factory',
    'create_filter_generator_factory',
    'create_aggregator_generator_factory',
]
