"""
Pine Context System
===================

Serializable execution contexts that can be:
- Transmitted across systems via signals
- Persisted to storage
- Hot-swapped at runtime
- Chained for hierarchical state

SOURCE: matts/context_system.py (43KB)
Copy the full implementation from that file.

Quick Reference:
----------------
    from . import SerializableExecutionContext, ContextChainNode

    # Create context
    context = SerializableExecutionContext(
        context_id="player_session_001",
        data={"player": "Alice", "room": "lighthouse"}
    )

    # Create chain for nested contexts
    parent = ContextChainNode(context)
    child = parent.derive({"room": "study"})

    # Serialize for transmission
    serialized = context.serialize()
"""

from typing import Any, Dict, List, Optional, Callable, Set, TypeVar
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import uuid
import asyncio

from .primitives import ContextState, T
from .signals import SignalPayload, SignalType, Observer, ObserverPriority


# =============================================================================
#                              CONTEXT CLASSES
# =============================================================================

@dataclass
class SerializableExecutionContext:
    """
    A serializable execution context that can be transmitted and restored.

    This is the core unit of state in Pine - it carries:
    - Unique identity (context_id)
    - Arbitrary state data
    - Callback chains for behavior
    - Dependencies for serialization
    - Observer hooks for reactivity

    TODO: Copy full implementation from matts/context_system.py
    """
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    state: ContextState = ContextState.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Callback chains - functions to execute in order
    _callbacks: List[Callable] = field(default_factory=list)

    # Dependencies needed for serialization
    _dependencies: Set[str] = field(default_factory=set)

    # Observers watching this context
    _observers: List['ContextObserver'] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context data."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value and notify observers."""
        old_value = self.data.get(key)
        self.data[key] = value
        self.updated_at = datetime.now()

        for observer in self._observers:
            observer.on_data_changed(self, key, old_value, value)

    def update(self, **kwargs) -> None:
        """Update multiple values at once."""
        for key, value in kwargs.items():
            self.set(key, value)

    def add_callback(self, callback: Callable) -> None:
        """Add callback to execution chain."""
        self._callbacks.append(callback)

    def add_dependency(self, dep: str) -> None:
        """Add a dependency identifier."""
        self._dependencies.add(dep)

    def add_observer(self, observer: 'ContextObserver') -> None:
        """Add an observer to watch this context."""
        self._observers.append(observer)

    async def execute_callbacks(self, *args, **kwargs) -> List[Any]:
        """Execute all callbacks in order."""
        results = []
        for callback in self._callbacks:
            if asyncio.iscoroutinefunction(callback):
                result = await callback(self, *args, **kwargs)
            else:
                result = callback(self, *args, **kwargs)
            results.append(result)
        return results

    def serialize(self) -> Dict[str, Any]:
        """Serialize context to dictionary."""
        return {
            'context_id': self.context_id,
            'data': self.data,
            'state': self.state.name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'dependencies': list(self._dependencies),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'SerializableExecutionContext':
        """Deserialize from dictionary."""
        context = cls(
            context_id=data['context_id'],
            data=data['data'],
            state=ContextState[data['state']],
        )
        context.created_at = datetime.fromisoformat(data['created_at'])
        context.updated_at = datetime.fromisoformat(data['updated_at'])
        context._dependencies = set(data.get('dependencies', []))
        return context


@dataclass
class ContextChainNode:
    """
    Node in a context chain for hierarchical state.

    Allows deriving child contexts that inherit from parents
    while maintaining their own overrides.
    """
    context: SerializableExecutionContext
    parent: Optional['ContextChainNode'] = None
    children: List['ContextChainNode'] = field(default_factory=list)

    def derive(self, overrides: Dict[str, Any] = None) -> 'ContextChainNode':
        """Create child node with optional data overrides."""
        child_context = SerializableExecutionContext(
            data={**self.context.data, **(overrides or {})}
        )
        child_node = ContextChainNode(
            context=child_context,
            parent=self
        )
        self.children.append(child_node)
        return child_node

    def get(self, key: str, default: Any = None) -> Any:
        """Get value, walking up chain if not found."""
        value = self.context.get(key)
        if value is not None:
            return value
        if self.parent:
            return self.parent.get(key, default)
        return default

    def root(self) -> 'ContextChainNode':
        """Get root of the chain."""
        if self.parent:
            return self.parent.root()
        return self


@dataclass
class ContextSnapshot:
    """
    Snapshot of context state at a point in time.

    Used for:
    - Undo/redo
    - State comparison
    - Time travel debugging
    """
    context_id: str
    data: Dict[str, Any]
    state: ContextState
    timestamp: datetime = field(default_factory=datetime.now)
    label: str = ""

    @classmethod
    def from_context(
        cls,
        context: SerializableExecutionContext,
        label: str = ""
    ) -> 'ContextSnapshot':
        """Create snapshot from context."""
        import copy
        return cls(
            context_id=context.context_id,
            data=copy.deepcopy(context.data),
            state=context.state,
            label=label
        )


# =============================================================================
#                              OBSERVERS
# =============================================================================

class ContextObserver(ABC):
    """Abstract observer for context changes."""

    @abstractmethod
    def on_data_changed(
        self,
        context: SerializableExecutionContext,
        key: str,
        old_value: Any,
        new_value: Any
    ) -> None:
        """Called when context data changes."""
        pass

    def on_state_changed(
        self,
        context: SerializableExecutionContext,
        old_state: ContextState,
        new_state: ContextState
    ) -> None:
        """Called when context state changes."""
        pass


class SignalAwareContextObserver(ContextObserver):
    """Observer that emits signals on context changes."""

    def __init__(self, signal_emitter: Callable[[SignalPayload], None]):
        self.signal_emitter = signal_emitter

    def on_data_changed(
        self,
        context: SerializableExecutionContext,
        key: str,
        old_value: Any,
        new_value: Any
    ) -> None:
        payload = SignalPayload(
            signal_type=SignalType.CONTEXT_UPDATED,
            source=context.context_id,
            data={
                'key': key,
                'old_value': old_value,
                'new_value': new_value
            }
        )
        self.signal_emitter(payload)


class CompositeObserver(ContextObserver):
    """Observer that delegates to multiple observers."""

    def __init__(self, observers: List[ContextObserver] = None):
        self.observers = observers or []

    def add(self, observer: ContextObserver) -> None:
        self.observers.append(observer)

    def on_data_changed(
        self,
        context: SerializableExecutionContext,
        key: str,
        old_value: Any,
        new_value: Any
    ) -> None:
        for observer in self.observers:
            observer.on_data_changed(context, key, old_value, new_value)


# =============================================================================
#                              LIBRARY
# =============================================================================

class SerializableContextLibrary:
    """
    Central registry and manager for contexts.

    Provides:
    - Context creation and lookup
    - Bulk operations
    - Garbage collection
    - Health monitoring

    TODO: Copy full implementation from matts/context_system.py
    """

    def __init__(self):
        self._contexts: Dict[str, SerializableExecutionContext] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        context_id: str = None,
        data: Dict[str, Any] = None
    ) -> SerializableExecutionContext:
        """Create and register a new context."""
        context = SerializableExecutionContext(
            context_id=context_id or str(uuid.uuid4()),
            data=data or {}
        )
        async with self._lock:
            self._contexts[context.context_id] = context
        return context

    async def get(self, context_id: str) -> Optional[SerializableExecutionContext]:
        """Get context by ID."""
        return self._contexts.get(context_id)

    async def destroy(self, context_id: str) -> bool:
        """Destroy a context."""
        async with self._lock:
            if context_id in self._contexts:
                del self._contexts[context_id]
                return True
        return False

    async def list_all(self) -> List[str]:
        """List all context IDs."""
        return list(self._contexts.keys())


class ContextGarbageCollector:
    """
    Garbage collector for stale contexts.

    TODO: Copy implementation from matts/context_system.py
    """

    def __init__(self, library: SerializableContextLibrary):
        self.library = library

    async def collect(self, max_age_seconds: int = 3600) -> int:
        """Collect contexts older than max_age_seconds."""
        # TODO: Implement
        return 0
