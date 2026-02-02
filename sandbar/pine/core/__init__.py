"""
Pine Core - Foundation Layer
============================

The foundation of the Pine engine providing:
- Signal system for event-driven communication
- Serializable execution contexts
- Type primitives and protocols
- Serialization utilities

This layer has NO dependencies on other Pine modules.

Source Files (to integrate):
- matts/signal_system.py -> signals.py
- matts/context_system.py -> context.py
- matts/context_serialization.py -> serialization.py
- Mechanics/everything/primitives.py -> primitives.py
"""

from .primitives import (
    # Type variables
    T, P, N, E,
    # Enums
    Priority, StorageType, SignalType, WordClass,
    FragmentCategory, PossibilityType, ContextState,
    # Protocols
    HasId, HasName, Serializable, Embeddable,
    # Base classes
    Identified, Context,
)

from .signals import (
    # Core signal classes
    SignalLine, SignalPayload, Signal,
    # Observer pattern
    Observer, ObserverPriority, CallbackObserver,
    ObserverBus, CircuitState, ObserverStats,
    # Factories
    create_signal_line, signal_handler,
)

from .context import (
    # Context classes
    SerializableExecutionContext, ContextChainNode,
    ContextObserver, SerializableContextLibrary,
    # Advanced observers
    SignalAwareContextObserver, CompositeObserver,
    # Utilities
    ContextSnapshot, ContextGarbageCollector,
)

from .serialization import (
    # Metadata
    SerializedContextMetadata,
    # High-performance components
    HighPerformanceSignalBus, FastDependencyBundler,
    OptimizedSerializer,
    # Portability
    SerializableExecutionContextWithPortability,
)

__all__ = [
    # Primitives
    'T', 'P', 'N', 'E',
    'Priority', 'StorageType', 'SignalType', 'WordClass',
    'FragmentCategory', 'PossibilityType', 'ContextState',
    'HasId', 'HasName', 'Serializable', 'Embeddable',
    'Identified', 'Context',

    # Signals
    'SignalLine', 'SignalPayload', 'Signal',
    'Observer', 'ObserverPriority', 'CallbackObserver',
    'ObserverBus', 'CircuitState', 'ObserverStats',
    'create_signal_line', 'signal_handler',

    # Context
    'SerializableExecutionContext', 'ContextChainNode',
    'ContextObserver', 'SerializableContextLibrary',
    'SignalAwareContextObserver', 'CompositeObserver',
    'ContextSnapshot', 'ContextGarbageCollector',

    # Serialization
    'SerializedContextMetadata',
    'HighPerformanceSignalBus', 'FastDependencyBundler',
    'OptimizedSerializer', 'SerializableExecutionContextWithPortability',
]
