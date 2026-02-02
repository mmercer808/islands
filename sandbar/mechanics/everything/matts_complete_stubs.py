#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║              M A T T S   C O M P L E T E   S Y S T E M   S T U B S            ║
║                                                                               ║
║         Fully Fleshed Out Class Stubs with Embedding & Storage Support        ║
║                                                                               ║
║  This file contains:                                                          ║
║  - Complete class hierarchies with all methods stubbed                        ║
║  - Generic Observer pattern (works with ANY input class)                      ║
║  - Text Extraction pipeline                                                   ║
║  - White Room Builder                                                         ║
║  - Embedding/Vector Storage (GlyphGraph)                                      ║
║  - Layer System with multi-dimensional traversal                              ║
║  - Lookahead Engine                                                           ║
║  - Serialization/Persistence                                                  ║
║                                                                               ║
║  Every method is stubbed with signature, docstring, and TODO comments.        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, List, Set, Optional, Any, Callable, Iterator, Iterable,
    Tuple, Union, Generic, TypeVar, Type, Protocol, Sequence,
    Mapping, MutableMapping, runtime_checkable
)
from enum import Enum, auto, IntEnum
from collections import defaultdict, deque
from contextlib import contextmanager
from weakref import ref, ReferenceType, WeakValueDictionary
import uuid
import time
import json
import hashlib
import pickle
import zlib
import base64
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor


# ═══════════════════════════════════════════════════════════════════════════════
#                              TYPE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

T = TypeVar('T')                    # Generic type
N = TypeVar('N')                    # Node type
E = TypeVar('E')                    # Entity type
S = TypeVar('S', bound='Signal')    # Signal type
O = TypeVar('O', bound='Observer')  # Observer type
C = TypeVar('C', bound='Context')   # Context type


# ═══════════════════════════════════════════════════════════════════════════════
#                              PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class Identifiable(Protocol):
    """Protocol for objects with an ID"""
    @property
    def id(self) -> str: ...


@runtime_checkable
class Named(Protocol):
    """Protocol for objects with a name"""
    @property
    def name(self) -> str: ...


@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable objects"""
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable': ...


@runtime_checkable
class Observable(Protocol):
    """Protocol for observable objects"""
    def add_observer(self, observer: 'Observer') -> str: ...
    def remove_observer(self, observer_id: str) -> bool: ...
    def notify_observers(self, event: Any) -> None: ...


@runtime_checkable
class Embeddable(Protocol):
    """Protocol for objects that can be embedded as vectors"""
    def to_embedding(self) -> List[float]: ...
    @classmethod
    def from_embedding(cls, embedding: List[float]) -> 'Embeddable': ...


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 1: ENUMERATIONS                                           ║
# ║                                                                           ║
# ║  All enum types used throughout the system.                               ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class Priority(IntEnum):
    """Universal priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class EntityType(Enum):
    """Game entity types"""
    LOCATION = auto()
    ITEM = auto()
    CHARACTER = auto()
    CONTAINER = auto()
    DOOR = auto()
    CONCEPT = auto()
    EVENT = auto()
    ABSTRACT = auto()


class EntityState(Enum):
    """Entity states"""
    NORMAL = auto()
    HIDDEN = auto()
    LOCKED = auto()
    OPEN = auto()
    CLOSED = auto()
    BROKEN = auto()
    ACTIVE = auto()
    INACTIVE = auto()


class LinkType(Enum):
    """Primary link/edge types"""
    RELATIONAL = auto()   # Spatial, possession, social
    LOGICAL = auto()      # Requires, unlocks, reveals
    TEMPORAL = auto()     # Before, after, during
    CAUSAL = auto()       # Causes, enables, prevents
    THEMATIC = auto()     # Symbolizes, represents
    WILDCARD = auto()     # Custom/narrative


class LayerType(Enum):
    """Named layer types for graph traversal"""
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CHARACTER = "character"
    THEMATIC = "thematic"
    NARRATIVE = "narrative"
    LOGICAL = "logical"
    CAUSAL = "causal"
    SEMANTIC = "semantic"   # Embedding-based similarity
    CUSTOM = "custom"


class SignalType(Enum):
    """Signal types for observer system"""
    # State signals
    STATE_ENTER = "state_enter"
    STATE_EXIT = "state_exit"
    STATE_CHANGE = "state_change"
    
    # Entity signals
    ENTITY_CREATED = "entity_created"
    ENTITY_MODIFIED = "entity_modified"
    ENTITY_DESTROYED = "entity_destroyed"
    ENTITY_LINKED = "entity_linked"
    ENTITY_UNLINKED = "entity_unlinked"
    
    # Traversal signals
    TRAVERSAL_START = "traversal_start"
    TRAVERSAL_STEP = "traversal_step"
    TRAVERSAL_END = "traversal_end"
    LAYER_SWITCH = "layer_switch"
    GRAPH_SWAP = "graph_swap"
    
    # Extraction signals
    EXTRACTION_START = "extraction_start"
    EXTRACTION_WORD = "extraction_word"
    EXTRACTION_ENTITY = "extraction_entity"
    EXTRACTION_RELATION = "extraction_relation"
    EXTRACTION_COMPLETE = "extraction_complete"
    
    # Embedding signals
    EMBEDDING_CREATED = "embedding_created"
    EMBEDDING_INDEXED = "embedding_indexed"
    SIMILARITY_QUERY = "similarity_query"
    
    # System signals
    TICK = "tick"
    ERROR = "error"
    WARNING = "warning"
    CUSTOM = "custom"


class WordClass(Enum):
    """Grammatical word classification"""
    NOUN = "noun"
    PROPER_NOUN = "proper_noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PREPOSITION = "preposition"
    ARTICLE = "article"
    PRONOUN = "pronoun"
    CONJUNCTION = "conjunction"
    INTERJECTION = "interjection"
    UNKNOWN = "unknown"


class FragmentCategory(IntEnum):
    """Prose fragment categories (order = composition priority)"""
    BASE_DESCRIPTION = 10
    ATMOSPHERIC = 20
    STATE_CHANGE = 30
    ITEM_PRESENCE = 40
    NPC_AMBIENT = 50
    SENSORY = 60
    HISTORY = 70
    DISCOVERY = 80


class StorageType(Enum):
    """Storage backend types"""
    MEMORY = "memory"           # In-memory only
    FILE = "file"               # File-based JSON/pickle
    SQLITE = "sqlite"           # SQLite database
    VECTOR_DB = "vector_db"     # Vector database (embeddings)
    DISTRIBUTED = "distributed"  # Distributed storage


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 2: BASE CLASSES                                           ║
# ║                                                                           ║
# ║  Abstract base classes that everything inherits from.                     ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class BaseIdentifiable:
    """Base class for all identifiable objects"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self) -> None:
        """Update modification timestamp"""
        self.modified_at = time.time()
    
    def set_meta(self, key: str, value: Any) -> None:
        """Set metadata value"""
        self.metadata[key] = value
        self.touch()
    
    def get_meta(self, key: str, default: Any = None) -> Any:
        """Get metadata value"""
        return self.metadata.get(key, default)


class Context(ABC):
    """
    Abstract base context class.
    
    Contexts carry state through operations. They are the "memory"
    of any process - traversal, extraction, game state, etc.
    """
    
    @abstractmethod
    def clone(self) -> 'Context':
        """Create a deep copy of this context"""
        ...
    
    @abstractmethod
    def merge(self, other: 'Context') -> 'Context':
        """Merge another context into this one"""
        ...
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        ...
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Context':
        """Deserialize from dictionary"""
        ...


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 3: OBSERVER PATTERN (Generic)                             ║
# ║                                                                           ║
# ║  Observers that work with ANY input class.                                ║
# ║  The system is fully generic and type-safe.                               ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class Signal(Generic[T]):
    """
    Generic signal that can carry any payload type.
    
    The signal is the unit of communication between observable
    objects and their observers. It carries type, source, payload,
    and metadata.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: SignalType = SignalType.CUSTOM
    source_id: str = ""
    target_id: Optional[str] = None
    
    # Payload - generic type T
    payload: Optional[T] = None
    
    # Additional data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    timestamp: float = field(default_factory=time.time)
    priority: Priority = Priority.NORMAL
    
    # Processing state
    handled: bool = False
    responses: List[Any] = field(default_factory=list)
    error: Optional[Exception] = None
    
    def with_payload(self, payload: T) -> 'Signal[T]':
        """Return new signal with different payload"""
        # TODO: Implement immutable update
        ...
    
    def add_response(self, response: Any, observer_id: str = None) -> None:
        """Add a response from an observer"""
        self.responses.append({
            'observer_id': observer_id,
            'response': response,
            'timestamp': time.time()
        })
        self.handled = True
    
    def mark_error(self, error: Exception) -> None:
        """Mark signal as errored"""
        self.error = error
        self.handled = True


class Observer(ABC, Generic[T]):
    """
    Abstract generic observer.
    
    Can observe ANY type T. Subclass and implement handle()
    for specific behavior.
    """
    
    def __init__(self, 
                 observer_id: str = None,
                 priority: Priority = Priority.NORMAL,
                 filter_fn: Callable[[Signal[T]], bool] = None):
        self.id = observer_id or str(uuid.uuid4())
        self.priority = priority
        self._filter_fn = filter_fn
        self._active = True
        self._handled_count = 0
        self._error_count = 0
    
    @property
    def active(self) -> bool:
        return self._active
    
    def activate(self) -> None:
        """Activate this observer"""
        self._active = True
    
    def deactivate(self) -> None:
        """Deactivate this observer"""
        self._active = False
    
    def set_filter(self, filter_fn: Callable[[Signal[T]], bool]) -> None:
        """Set filter function for signals"""
        self._filter_fn = filter_fn
    
    def should_handle(self, signal: Signal[T]) -> bool:
        """Check if observer should handle this signal"""
        if not self._active:
            return False
        if self._filter_fn and not self._filter_fn(signal):
            return False
        return True
    
    @abstractmethod
    def handle(self, signal: Signal[T]) -> Any:
        """
        Handle a signal.
        
        Override in subclass to implement specific behavior.
        Return value is added to signal.responses.
        """
        ...
    
    def __call__(self, signal: Signal[T]) -> Any:
        """Allow observer to be called directly"""
        if self.should_handle(signal):
            try:
                result = self.handle(signal)
                self._handled_count += 1
                return result
            except Exception as e:
                self._error_count += 1
                raise


class FunctionObserver(Observer[T]):
    """
    Observer that wraps a function.
    
    Allows any callable to be used as an observer.
    """
    
    def __init__(self,
                 fn: Callable[[Signal[T]], Any],
                 observer_id: str = None,
                 priority: Priority = Priority.NORMAL):
        super().__init__(observer_id, priority)
        self._fn = fn
    
    def handle(self, signal: Signal[T]) -> Any:
        return self._fn(signal)


class TypedObserver(Observer[T]):
    """
    Observer that only handles specific signal types.
    """
    
    def __init__(self,
                 signal_types: List[SignalType],
                 observer_id: str = None,
                 priority: Priority = Priority.NORMAL):
        super().__init__(observer_id, priority)
        self._signal_types = set(signal_types)
    
    def should_handle(self, signal: Signal[T]) -> bool:
        if not super().should_handle(signal):
            return False
        return signal.signal_type in self._signal_types
    
    @abstractmethod
    def handle(self, signal: Signal[T]) -> Any:
        ...


class StateObserver(TypedObserver[Any]):
    """
    Observer specialized for state machine transitions.
    
    Tracks state history, provides enter/exit/transition hooks.
    """
    
    def __init__(self, observer_id: str = None):
        super().__init__(
            [SignalType.STATE_ENTER, SignalType.STATE_EXIT, SignalType.STATE_CHANGE],
            observer_id,
            Priority.HIGH
        )
        self.state_history: List[Tuple[str, float]] = []
        self.current_state: Optional[str] = None
        
        # Hooks: state_name → list of callbacks
        self._on_enter: Dict[str, List[Callable]] = defaultdict(list)
        self._on_exit: Dict[str, List[Callable]] = defaultdict(list)
        self._on_transition: Dict[Tuple[str, str], List[Callable]] = defaultdict(list)
    
    def on_enter(self, state: str, callback: Callable[[Signal], Any]) -> None:
        """Register callback for entering a state"""
        self._on_enter[state].append(callback)
    
    def on_exit(self, state: str, callback: Callable[[Signal], Any]) -> None:
        """Register callback for exiting a state"""
        self._on_exit[state].append(callback)
    
    def on_transition(self, from_state: str, to_state: str, 
                     callback: Callable[[Signal], Any]) -> None:
        """Register callback for specific state transition"""
        self._on_transition[(from_state, to_state)].append(callback)
    
    def handle(self, signal: Signal) -> Dict[str, Any]:
        """Handle state change signals"""
        old_state = signal.data.get('old_state')
        new_state = signal.data.get('new_state')
        
        # Fire exit hooks
        if old_state and old_state in self._on_exit:
            for cb in self._on_exit[old_state]:
                try:
                    cb(signal)
                except Exception:
                    pass  # TODO: Log error
        
        # Fire transition hooks
        if (old_state, new_state) in self._on_transition:
            for cb in self._on_transition[(old_state, new_state)]:
                try:
                    cb(signal)
                except Exception:
                    pass
        
        # Fire enter hooks
        if new_state and new_state in self._on_enter:
            for cb in self._on_enter[new_state]:
                try:
                    cb(signal)
                except Exception:
                    pass
        
        # Update history
        if new_state:
            self.state_history.append((new_state, signal.timestamp))
            self.current_state = new_state
        
        return {'transition': f"{old_state} → {new_state}"}


class EntityObserver(TypedObserver[Any]):
    """
    Observer specialized for entity lifecycle events.
    
    Tracks entity creation, modification, destruction.
    Can watch specific entities.
    """
    
    def __init__(self, observer_id: str = None):
        super().__init__(
            [SignalType.ENTITY_CREATED, SignalType.ENTITY_MODIFIED, 
             SignalType.ENTITY_DESTROYED, SignalType.ENTITY_LINKED,
             SignalType.ENTITY_UNLINKED],
            observer_id,
            Priority.NORMAL
        )
        self.entity_log: List[Dict[str, Any]] = []
        self._watched_entities: Dict[str, List[Callable]] = defaultdict(list)
    
    def watch(self, entity_id: str, callback: Callable[[Signal], Any]) -> None:
        """Watch a specific entity for changes"""
        self._watched_entities[entity_id].append(callback)
    
    def unwatch(self, entity_id: str) -> None:
        """Stop watching an entity"""
        if entity_id in self._watched_entities:
            del self._watched_entities[entity_id]
    
    def handle(self, signal: Signal) -> Dict[str, Any]:
        """Handle entity signals"""
        entity_id = signal.data.get('entity_id', signal.source_id)
        
        # Log the event
        log_entry = {
            'type': signal.signal_type.value,
            'entity_id': entity_id,
            'timestamp': signal.timestamp,
            'data': signal.data
        }
        self.entity_log.append(log_entry)
        
        # Fire entity-specific callbacks
        if entity_id in self._watched_entities:
            for cb in self._watched_entities[entity_id]:
                try:
                    cb(signal)
                except Exception:
                    pass
        
        return {'logged': entity_id}


class TraversalObserver(TypedObserver[Any]):
    """
    Observer specialized for graph traversal events.
    
    Tracks path, visited nodes, layer switches.
    """
    
    def __init__(self, observer_id: str = None):
        super().__init__(
            [SignalType.TRAVERSAL_START, SignalType.TRAVERSAL_STEP,
             SignalType.TRAVERSAL_END, SignalType.LAYER_SWITCH,
             SignalType.GRAPH_SWAP],
            observer_id,
            Priority.NORMAL
        )
        self.path: List[str] = []
        self.visited: Set[str] = set()
        self.layer_history: List[Tuple[str, str, float]] = []  # (old, new, timestamp)
        
        self._on_step: List[Callable[[Signal], Any]] = []
        self._on_layer_switch: List[Callable[[Signal], Any]] = []
    
    def on_step(self, callback: Callable[[Signal], Any]) -> None:
        """Register callback for each traversal step"""
        self._on_step.append(callback)
    
    def on_layer_switch(self, callback: Callable[[Signal], Any]) -> None:
        """Register callback for layer switches"""
        self._on_layer_switch.append(callback)
    
    def handle(self, signal: Signal) -> Optional[Dict[str, Any]]:
        """Handle traversal signals"""
        if signal.signal_type == SignalType.TRAVERSAL_STEP:
            node_id = signal.data.get('node_id')
            if node_id:
                self.path.append(node_id)
                self.visited.add(node_id)
            
            for cb in self._on_step:
                try:
                    cb(signal)
                except Exception:
                    pass
            
            return {'path_length': len(self.path)}
        
        elif signal.signal_type == SignalType.LAYER_SWITCH:
            old_layer = signal.data.get('old_layer')
            new_layer = signal.data.get('new_layer')
            self.layer_history.append((old_layer, new_layer, signal.timestamp))
            
            for cb in self._on_layer_switch:
                try:
                    cb(signal)
                except Exception:
                    pass
            
            return {'layer_switches': len(self.layer_history)}
        
        return None


class ExtractionObserver(TypedObserver[Any]):
    """
    Observer specialized for text extraction events.
    
    Tracks words, entities, relations as they're extracted.
    """
    
    def __init__(self, observer_id: str = None):
        super().__init__(
            [SignalType.EXTRACTION_START, SignalType.EXTRACTION_WORD,
             SignalType.EXTRACTION_ENTITY, SignalType.EXTRACTION_RELATION,
             SignalType.EXTRACTION_COMPLETE],
            observer_id,
            Priority.NORMAL
        )
        self.word_count = 0
        self.entity_count = 0
        self.relation_count = 0
        self.current_extraction_id: Optional[str] = None
    
    def handle(self, signal: Signal) -> Dict[str, Any]:
        """Handle extraction signals"""
        if signal.signal_type == SignalType.EXTRACTION_START:
            self.current_extraction_id = signal.data.get('extraction_id')
            self.word_count = 0
            self.entity_count = 0
            self.relation_count = 0
        
        elif signal.signal_type == SignalType.EXTRACTION_WORD:
            self.word_count += 1
        
        elif signal.signal_type == SignalType.EXTRACTION_ENTITY:
            self.entity_count += 1
        
        elif signal.signal_type == SignalType.EXTRACTION_RELATION:
            self.relation_count += 1
        
        return {
            'words': self.word_count,
            'entities': self.entity_count,
            'relations': self.relation_count
        }


class ObserverBus(Generic[T]):
    """
    Central hub for observer registration and signal dispatch.
    
    Generic over payload type T. Can dispatch signals to
    registered observers based on signal type and filters.
    """
    
    def __init__(self):
        self._observers: Dict[str, Observer[T]] = {}
        self._type_subscriptions: Dict[SignalType, Set[str]] = defaultdict(set)
        self._signal_history: deque = deque(maxlen=1000)
        self._pending: deque = deque()
        self._processing = False
        self._lock = threading.RLock()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────────────────────
    
    def register(self, observer: Observer[T],
                 signal_types: List[SignalType] = None) -> str:
        """
        Register an observer.
        
        Args:
            observer: The observer to register
            signal_types: Signal types to subscribe to (None = all)
        
        Returns:
            Observer ID
        """
        with self._lock:
            self._observers[observer.id] = observer
            
            if signal_types:
                for st in signal_types:
                    self._type_subscriptions[st].add(observer.id)
            else:
                for st in SignalType:
                    self._type_subscriptions[st].add(observer.id)
            
            return observer.id
    
    def unregister(self, observer_id: str) -> bool:
        """Remove an observer"""
        with self._lock:
            if observer_id in self._observers:
                del self._observers[observer_id]
                for subscribers in self._type_subscriptions.values():
                    subscribers.discard(observer_id)
                return True
            return False
    
    def on(self, signal_type: SignalType, callback: Callable[[Signal[T]], Any],
           priority: Priority = Priority.NORMAL) -> str:
        """
        Convenience: register a function as an observer.
        
        Returns observer ID for later removal.
        """
        observer = FunctionObserver(callback, priority=priority)
        return self.register(observer, [signal_type])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Dispatch
    # ─────────────────────────────────────────────────────────────────────────
    
    def emit(self, signal: Signal[T]) -> Signal[T]:
        """
        Emit a signal to all relevant observers.
        
        Returns the signal after processing (with responses).
        """
        with self._lock:
            self._pending.append(signal)
            
            if not self._processing:
                self._process_pending()
            
            return signal
    
    def emit_type(self, signal_type: SignalType,
                  source_id: str = "",
                  payload: T = None,
                  data: Dict[str, Any] = None,
                  **kwargs) -> Signal[T]:
        """
        Convenience: emit a signal by type.
        
        Returns the created signal.
        """
        signal = Signal(
            signal_type=signal_type,
            source_id=source_id,
            payload=payload,
            data=data or {},
            **kwargs
        )
        return self.emit(signal)
    
    def _process_pending(self) -> None:
        """Process all pending signals"""
        self._processing = True
        
        while self._pending:
            signal = self._pending.popleft()
            self._dispatch(signal)
            self._signal_history.append(signal)
        
        self._processing = False
    
    def _dispatch(self, signal: Signal[T]) -> None:
        """Dispatch a single signal to observers"""
        subscriber_ids = self._type_subscriptions.get(signal.signal_type, set())
        
        # Get active observers and sort by priority
        observers = []
        for obs_id in subscriber_ids:
            if obs_id in self._observers:
                observer = self._observers[obs_id]
                if observer.should_handle(signal):
                    observers.append(observer)
        
        observers.sort(key=lambda o: o.priority.value)
        
        # Dispatch
        for observer in observers:
            try:
                result = observer.handle(signal)
                if result is not None:
                    signal.add_response(result, observer.id)
            except Exception as e:
                signal.mark_error(e)
                # Emit error signal (avoid infinite loop)
                if signal.signal_type != SignalType.ERROR:
                    self.emit_type(
                        SignalType.ERROR,
                        source_id=observer.id,
                        data={'error': str(e), 'signal_id': signal.id}
                    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_observer(self, observer_id: str) -> Optional[Observer[T]]:
        """Get an observer by ID"""
        return self._observers.get(observer_id)
    
    def get_history(self, signal_type: SignalType = None,
                    limit: int = 100) -> List[Signal[T]]:
        """Get signal history, optionally filtered"""
        signals = list(self._signal_history)
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        return signals[-limit:]
    
    def observer_count(self) -> int:
        """Get number of registered observers"""
        return len(self._observers)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Factory Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def create_state_observer(self) -> Tuple[StateObserver, str]:
        """Create and register a state observer"""
        observer = StateObserver()
        obs_id = self.register(observer)
        return observer, obs_id
    
    def create_entity_observer(self) -> Tuple[EntityObserver, str]:
        """Create and register an entity observer"""
        observer = EntityObserver()
        obs_id = self.register(observer)
        return observer, obs_id
    
    def create_traversal_observer(self) -> Tuple[TraversalObserver, str]:
        """Create and register a traversal observer"""
        observer = TraversalObserver()
        obs_id = self.register(observer)
        return observer, obs_id
    
    def create_extraction_observer(self) -> Tuple[ExtractionObserver, str]:
        """Create and register an extraction observer"""
        observer = ExtractionObserver()
        obs_id = self.register(observer)
        return observer, obs_id


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 4: EMBEDDING & VECTOR STORAGE (GlyphGraph)                ║
# ║                                                                           ║
# ║  Storage system with embedding support for semantic operations.           ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class Embedding:
    """
    Vector embedding for semantic operations.
    
    Wraps a float vector with metadata for similarity search.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vector: List[float] = field(default_factory=list)
    dimensions: int = 0
    source_id: str = ""         # ID of embedded object
    source_type: str = ""       # Type of embedded object
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.dimensions = len(self.vector)
    
    def normalize(self) -> 'Embedding':
        """Return normalized (unit length) embedding"""
        # TODO: Implement L2 normalization
        ...
    
    def cosine_similarity(self, other: 'Embedding') -> float:
        """Compute cosine similarity with another embedding"""
        # TODO: Implement cosine similarity
        ...
    
    def euclidean_distance(self, other: 'Embedding') -> float:
        """Compute Euclidean distance to another embedding"""
        # TODO: Implement Euclidean distance
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'id': self.id,
            'vector': self.vector,
            'dimensions': self.dimensions,
            'source_id': self.source_id,
            'source_type': self.source_type,
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Embedding':
        """Deserialize from dictionary"""
        return cls(**data)


class EmbeddingProvider(ABC):
    """
    Abstract embedding provider.
    
    Implement for different embedding backends (local, API, etc.)
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> Embedding:
        """Create embedding from text"""
        ...
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Create embeddings for multiple texts"""
        ...
    
    @abstractmethod
    def dimensions(self) -> int:
        """Get embedding dimensions"""
        ...


class SimpleHashEmbedding(EmbeddingProvider):
    """
    Simple hash-based embedding provider.
    
    Creates deterministic embeddings from text using hashing.
    Not semantically meaningful, but useful for testing.
    """
    
    def __init__(self, dimensions: int = 128):
        self._dimensions = dimensions
    
    def embed_text(self, text: str) -> Embedding:
        """Create hash-based embedding"""
        # TODO: Implement hash-based embedding
        # Use SHA256 hash expanded to desired dimensions
        ...
    
    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Embed multiple texts"""
        return [self.embed_text(t) for t in texts]
    
    def dimensions(self) -> int:
        return self._dimensions


class VectorIndex(ABC):
    """
    Abstract vector index for similarity search.
    
    Implement for different backends (flat, HNSW, IVF, etc.)
    """
    
    @abstractmethod
    def add(self, embedding: Embedding) -> None:
        """Add embedding to index"""
        ...
    
    @abstractmethod
    def add_batch(self, embeddings: List[Embedding]) -> None:
        """Add multiple embeddings"""
        ...
    
    @abstractmethod
    def search(self, query: Embedding, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for k nearest neighbors.
        
        Returns list of (id, similarity_score) tuples.
        """
        ...
    
    @abstractmethod
    def remove(self, embedding_id: str) -> bool:
        """Remove embedding from index"""
        ...
    
    @abstractmethod
    def count(self) -> int:
        """Get number of embeddings in index"""
        ...


class FlatVectorIndex(VectorIndex):
    """
    Simple flat (brute-force) vector index.
    
    O(n) search, but simple and exact.
    """
    
    def __init__(self):
        self._embeddings: Dict[str, Embedding] = {}
    
    def add(self, embedding: Embedding) -> None:
        """Add embedding"""
        self._embeddings[embedding.id] = embedding
    
    def add_batch(self, embeddings: List[Embedding]) -> None:
        """Add multiple embeddings"""
        for e in embeddings:
            self.add(e)
    
    def search(self, query: Embedding, k: int = 10) -> List[Tuple[str, float]]:
        """Brute-force k-NN search"""
        # TODO: Implement brute-force search with cosine similarity
        ...
    
    def remove(self, embedding_id: str) -> bool:
        """Remove embedding"""
        if embedding_id in self._embeddings:
            del self._embeddings[embedding_id]
            return True
        return False
    
    def count(self) -> int:
        return len(self._embeddings)


@dataclass
class GlyphNode:
    """
    A node in the GlyphGraph.
    
    Combines entity data with embedding for semantic operations.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    node_type: str = ""
    
    # Core data
    data: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    # Embedding (optional)
    embedding: Optional[Embedding] = None
    embedding_text: str = ""  # Text used to generate embedding
    
    # Connectivity
    edges_out: Dict[str, List['GlyphEdge']] = field(default_factory=lambda: defaultdict(list))
    edges_in: Dict[str, List['GlyphEdge']] = field(default_factory=lambda: defaultdict(list))
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    
    def add_edge_out(self, edge: 'GlyphEdge', layer: str = "default") -> None:
        """Add outgoing edge"""
        self.edges_out[layer].append(edge)
        self.modified_at = time.time()
    
    def add_edge_in(self, edge: 'GlyphEdge', layer: str = "default") -> None:
        """Add incoming edge"""
        self.edges_in[layer].append(edge)
        self.modified_at = time.time()
    
    def get_edges(self, layer: str = None, direction: str = "out") -> List['GlyphEdge']:
        """Get edges for layer (or all if layer is None)"""
        edges_dict = self.edges_out if direction == "out" else self.edges_in
        
        if layer:
            return edges_dict.get(layer, [])
        else:
            all_edges = []
            for layer_edges in edges_dict.values():
                all_edges.extend(layer_edges)
            return all_edges
    
    def set_embedding(self, embedding: Embedding, text: str = "") -> None:
        """Set the node's embedding"""
        self.embedding = embedding
        self.embedding_text = text
        self.modified_at = time.time()


@dataclass
class GlyphEdge:
    """
    An edge in the GlyphGraph.
    
    Connects two GlyphNodes with type and layer information.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    
    edge_type: LinkType = LinkType.RELATIONAL
    kind: str = ""  # Specific kind (e.g., "north_of", "contains")
    layer: str = "default"
    
    weight: float = 1.0
    bidirectional: bool = False
    
    # Condition (optional)
    condition: Optional[Callable[[Dict], bool]] = None
    condition_description: str = ""
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    
    def is_traversable(self, context: Dict[str, Any] = None) -> bool:
        """Check if edge can be traversed given context"""
        if self.condition is None:
            return True
        try:
            return self.condition(context or {})
        except Exception:
            return False


class GlyphGraph:
    """
    Graph with embedding support.
    
    Combines traditional graph operations with semantic similarity
    via vector embeddings. Supports:
    - Multi-layer edges
    - Semantic similarity search
    - Condition-based traversal
    - Persistence
    """
    
    def __init__(self, 
                 name: str = "GlyphGraph",
                 embedding_provider: EmbeddingProvider = None):
        self.name = name
        self.id = str(uuid.uuid4())
        
        # Storage
        self._nodes: Dict[str, GlyphNode] = {}
        self._edges: Dict[str, GlyphEdge] = {}
        
        # Embedding
        self._embedding_provider = embedding_provider or SimpleHashEmbedding()
        self._vector_index = FlatVectorIndex()
        
        # Indexes
        self._nodes_by_type: Dict[str, Set[str]] = defaultdict(set)
        self._nodes_by_tag: Dict[str, Set[str]] = defaultdict(set)
        self._edges_by_layer: Dict[str, Set[str]] = defaultdict(set)
        
        # Observer integration
        self._observer_bus: Optional[ObserverBus] = None
    
    def set_observer_bus(self, bus: ObserverBus) -> None:
        """Attach an observer bus for event emission"""
        self._observer_bus = bus
    
    # ─────────────────────────────────────────────────────────────────────────
    # Node Operations
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_node(self, node: GlyphNode, embed: bool = True) -> str:
        """
        Add a node to the graph.
        
        If embed=True and node has embedding_text, generates embedding.
        """
        self._nodes[node.id] = node
        self._nodes_by_type[node.node_type].add(node.id)
        for tag in node.tags:
            self._nodes_by_tag[tag].add(node.id)
        
        # Generate embedding if requested
        if embed and node.embedding_text:
            embedding = self._embedding_provider.embed_text(node.embedding_text)
            embedding.source_id = node.id
            embedding.source_type = node.node_type
            node.set_embedding(embedding)
            self._vector_index.add(embedding)
        
        # Emit signal
        if self._observer_bus:
            self._observer_bus.emit_type(
                SignalType.ENTITY_CREATED,
                source_id=node.id,
                data={'node_type': node.node_type, 'name': node.name}
            )
        
        return node.id
    
    def get_node(self, node_id: str) -> Optional[GlyphNode]:
        """Get node by ID"""
        return self._nodes.get(node_id)
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node and its edges"""
        if node_id not in self._nodes:
            return False
        
        node = self._nodes[node_id]
        
        # Remove from indexes
        self._nodes_by_type[node.node_type].discard(node_id)
        for tag in node.tags:
            self._nodes_by_tag[tag].discard(node_id)
        
        # Remove from vector index
        if node.embedding:
            self._vector_index.remove(node.embedding.id)
        
        # Remove connected edges
        for layer_edges in list(node.edges_out.values()):
            for edge in layer_edges:
                self.remove_edge(edge.id)
        for layer_edges in list(node.edges_in.values()):
            for edge in layer_edges:
                self.remove_edge(edge.id)
        
        del self._nodes[node_id]
        
        # Emit signal
        if self._observer_bus:
            self._observer_bus.emit_type(
                SignalType.ENTITY_DESTROYED,
                source_id=node_id
            )
        
        return True
    
    def find_nodes(self, 
                   node_type: str = None,
                   tags: Set[str] = None,
                   predicate: Callable[[GlyphNode], bool] = None) -> List[GlyphNode]:
        """Find nodes matching criteria"""
        candidates = set(self._nodes.keys())
        
        if node_type:
            candidates &= self._nodes_by_type.get(node_type, set())
        
        if tags:
            for tag in tags:
                candidates &= self._nodes_by_tag.get(tag, set())
        
        nodes = [self._nodes[nid] for nid in candidates]
        
        if predicate:
            nodes = [n for n in nodes if predicate(n)]
        
        return nodes
    
    # ─────────────────────────────────────────────────────────────────────────
    # Edge Operations
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_edge(self, edge: GlyphEdge) -> str:
        """Add an edge to the graph"""
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            raise ValueError("Source or target node not in graph")
        
        self._edges[edge.id] = edge
        self._edges_by_layer[edge.layer].add(edge.id)
        
        source = self._nodes[edge.source_id]
        target = self._nodes[edge.target_id]
        
        source.add_edge_out(edge, edge.layer)
        target.add_edge_in(edge, edge.layer)
        
        # Handle bidirectional
        if edge.bidirectional:
            reverse = GlyphEdge(
                source_id=edge.target_id,
                target_id=edge.source_id,
                edge_type=edge.edge_type,
                kind=edge.kind,
                layer=edge.layer,
                weight=edge.weight,
                condition=edge.condition
            )
            self._edges[reverse.id] = reverse
            target.add_edge_out(reverse, edge.layer)
            source.add_edge_in(reverse, edge.layer)
        
        # Emit signal
        if self._observer_bus:
            self._observer_bus.emit_type(
                SignalType.ENTITY_LINKED,
                source_id=edge.source_id,
                data={'target_id': edge.target_id, 'edge_type': edge.edge_type.name}
            )
        
        return edge.id
    
    def get_edge(self, edge_id: str) -> Optional[GlyphEdge]:
        """Get edge by ID"""
        return self._edges.get(edge_id)
    
    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge"""
        if edge_id not in self._edges:
            return False
        
        edge = self._edges[edge_id]
        self._edges_by_layer[edge.layer].discard(edge_id)
        
        # Remove from nodes
        if edge.source_id in self._nodes:
            source = self._nodes[edge.source_id]
            if edge.layer in source.edges_out:
                source.edges_out[edge.layer] = [
                    e for e in source.edges_out[edge.layer] if e.id != edge_id
                ]
        
        if edge.target_id in self._nodes:
            target = self._nodes[edge.target_id]
            if edge.layer in target.edges_in:
                target.edges_in[edge.layer] = [
                    e for e in target.edges_in[edge.layer] if e.id != edge_id
                ]
        
        del self._edges[edge_id]
        return True
    
    def connect(self, 
                source_id: str, 
                target_id: str,
                edge_type: LinkType = LinkType.RELATIONAL,
                kind: str = "",
                layer: str = "default",
                **kwargs) -> str:
        """Convenience method to create and add edge"""
        edge = GlyphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            kind=kind,
            layer=layer,
            **kwargs
        )
        return self.add_edge(edge)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Semantic Operations (Embedding-based)
    # ─────────────────────────────────────────────────────────────────────────
    
    def find_similar(self, 
                     query: Union[str, Embedding, GlyphNode],
                     k: int = 10,
                     threshold: float = 0.0) -> List[Tuple[GlyphNode, float]]:
        """
        Find k most similar nodes using embedding similarity.
        
        Args:
            query: Text, embedding, or node to search for
            k: Number of results
            threshold: Minimum similarity score
        
        Returns:
            List of (node, similarity_score) tuples
        """
        # Get query embedding
        if isinstance(query, str):
            query_embedding = self._embedding_provider.embed_text(query)
        elif isinstance(query, Embedding):
            query_embedding = query
        elif isinstance(query, GlyphNode):
            if query.embedding is None:
                raise ValueError("Node has no embedding")
            query_embedding = query.embedding
        else:
            raise TypeError(f"Invalid query type: {type(query)}")
        
        # Search index
        results = self._vector_index.search(query_embedding, k)
        
        # Map back to nodes
        node_results = []
        for emb_id, score in results:
            # Find node with this embedding
            for node in self._nodes.values():
                if node.embedding and node.embedding.id == emb_id:
                    if score >= threshold:
                        node_results.append((node, score))
                    break
        
        return node_results
    
    def semantic_layer_edges(self, 
                            node_id: str,
                            k: int = 5,
                            threshold: float = 0.7) -> List[Tuple[GlyphNode, float]]:
        """
        Get "semantic edges" - similar nodes as implicit connections.
        
        This creates a virtual SEMANTIC layer based on embedding similarity.
        """
        node = self.get_node(node_id)
        if not node or not node.embedding:
            return []
        
        similar = self.find_similar(node, k + 1, threshold)
        # Exclude self
        return [(n, s) for n, s in similar if n.id != node_id]
    
    # ─────────────────────────────────────────────────────────────────────────
    # Traversal
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_neighbors(self, 
                      node_id: str,
                      layer: str = None,
                      context: Dict[str, Any] = None) -> List[GlyphNode]:
        """Get neighboring nodes via outgoing edges"""
        node = self.get_node(node_id)
        if not node:
            return []
        
        edges = node.get_edges(layer, "out")
        ctx = context or {}
        
        neighbors = []
        for edge in edges:
            if edge.is_traversable(ctx):
                target = self.get_node(edge.target_id)
                if target:
                    neighbors.append(target)
        
        return neighbors
    
    def get_outgoing_links(self, node_id: str) -> List[GlyphEdge]:
        """Get all outgoing edges from a node"""
        node = self.get_node(node_id)
        if not node:
            return []
        return node.get_edges(None, "out")
    
    def get_incoming_links(self, node_id: str) -> List[GlyphEdge]:
        """Get all incoming edges to a node"""
        node = self.get_node(node_id)
        if not node:
            return []
        return node.get_edges(None, "in")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────────
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary"""
        # TODO: Implement serialization
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GlyphGraph':
        """Deserialize graph from dictionary"""
        # TODO: Implement deserialization
        ...
    
    def save(self, path: str, format: str = "json") -> None:
        """Save graph to file"""
        # TODO: Implement save
        ...
    
    @classmethod
    def load(cls, path: str) -> 'GlyphGraph':
        """Load graph from file"""
        # TODO: Implement load
        ...


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 5: LAYER SYSTEM                                           ║
# ║                                                                           ║
# ║  Multi-layer graph traversal with configurable edge visibility.           ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class LayerConfig:
    """Configuration for a traversal layer"""
    name: str
    edge_types: Set[str] = field(default_factory=set)
    edge_kinds: Set[str] = field(default_factory=set)
    
    # Custom filter
    filter_fn: Optional[Callable[[GlyphEdge, 'TraversalContext'], bool]] = None
    
    # Auto-switch condition
    switch_condition: Optional[Callable[['TraversalContext'], bool]] = None
    
    # Priority for auto-switch checking
    priority: int = 0
    
    def matches_edge(self, edge: GlyphEdge, context: 'TraversalContext' = None) -> bool:
        """Check if edge belongs to this layer"""
        if self.filter_fn:
            return self.filter_fn(edge, context)
        
        if self.edge_types and edge.edge_type.name not in self.edge_types:
            return False
        
        if self.edge_kinds and edge.kind not in self.edge_kinds:
            return False
        
        return True


class LayerRegistry:
    """Central registry of layer configurations"""
    
    def __init__(self):
        self._layers: Dict[str, LayerConfig] = {}
        self._active: str = "default"
        self._default_layer = LayerConfig(name="default")
    
    def register(self, config: LayerConfig) -> None:
        """Register a layer configuration"""
        self._layers[config.name] = config
    
    def unregister(self, name: str) -> bool:
        """Remove a layer configuration"""
        if name in self._layers:
            del self._layers[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[LayerConfig]:
        """Get layer by name"""
        return self._layers.get(name, self._default_layer if name == "default" else None)
    
    def set_active(self, name: str) -> bool:
        """Set the active layer"""
        if name in self._layers or name == "default":
            self._active = name
            return True
        return False
    
    @property
    def active(self) -> LayerConfig:
        """Get currently active layer config"""
        return self._layers.get(self._active, self._default_layer)
    
    @property
    def active_name(self) -> str:
        """Get name of active layer"""
        return self._active
    
    def check_auto_switch(self, context: 'TraversalContext') -> Optional[str]:
        """Check if any layer wants to auto-switch"""
        candidates = sorted(
            self._layers.values(),
            key=lambda l: l.priority,
            reverse=True
        )
        for layer in candidates:
            if layer.switch_condition and layer.switch_condition(context):
                return layer.name
        return None
    
    def list_layers(self) -> List[str]:
        """List all registered layer names"""
        return list(self._layers.keys())


# Preset layer configurations
def create_standard_layers() -> LayerRegistry:
    """Create standard layer registry with common layers"""
    registry = LayerRegistry()
    
    # Spatial layer
    registry.register(LayerConfig(
        name="spatial",
        edge_kinds={"north_of", "south_of", "east_of", "west_of",
                   "in", "on", "under", "near", "contains", "leads_to"},
        priority=10
    ))
    
    # Temporal layer
    registry.register(LayerConfig(
        name="temporal",
        edge_kinds={"before", "after", "during", "next", "previous"},
        priority=5
    ))
    
    # Character layer
    registry.register(LayerConfig(
        name="character",
        edge_kinds={"knows", "trusts", "owns", "held_by", "related_to"},
        priority=5
    ))
    
    # Logical layer
    registry.register(LayerConfig(
        name="logical",
        edge_types={"LOGICAL"},
        edge_kinds={"requires", "unlocks", "reveals", "enables", "blocks", "triggers"},
        priority=15
    ))
    
    # Semantic layer (similarity-based)
    registry.register(LayerConfig(
        name="semantic",
        filter_fn=lambda e, c: e.layer == "semantic",
        priority=5
    ))
    
    return registry


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 6: TRAVERSAL CONTEXT                                      ║
# ║                                                                           ║
# ║  Context that travels with the traversal wrapper.                         ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class TraversalContext(Context):
    """
    State that travels with the traversal wrapper.
    
    Contains position, history, game state, and graph reference.
    """
    # Position
    current_node: Any = None
    previous_node: Any = None
    
    # History
    path: List[Any] = field(default_factory=list)
    visited: Set[str] = field(default_factory=set)
    depth: int = 0
    step_count: int = 0
    
    # Layer state
    current_layer: str = "default"
    layer_stack: List[str] = field(default_factory=list)
    
    # Game state
    flags: Dict[str, bool] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    inventory: List[str] = field(default_factory=list)
    memories: List[Dict] = field(default_factory=list)
    
    # Scratch space
    buffer: Dict[str, Any] = field(default_factory=dict)
    
    # Graph reference (can be swapped!)
    graph_ref: Any = None
    
    # Timing
    start_time: float = field(default_factory=time.time)
    
    def push_layer(self, layer: str) -> None:
        """Push current layer onto stack, switch to new one"""
        self.layer_stack.append(self.current_layer)
        self.current_layer = layer
    
    def pop_layer(self) -> str:
        """Pop back to previous layer"""
        if self.layer_stack:
            self.current_layer = self.layer_stack.pop()
        return self.current_layer
    
    def get_node_id(self, node: Any) -> str:
        """Extract ID from node"""
        if hasattr(node, 'id'):
            return node.id
        return str(node)
    
    def mark_visited(self, node: Any) -> None:
        """Mark node as visited"""
        self.visited.add(self.get_node_id(node))
    
    def is_visited(self, node: Any) -> bool:
        """Check if node was visited"""
        return self.get_node_id(node) in self.visited
    
    def set_flag(self, name: str, value: bool = True) -> None:
        """Set a game flag"""
        self.flags[name] = value
    
    def get_flag(self, name: str, default: bool = False) -> bool:
        """Get a game flag"""
        return self.flags.get(name, default)
    
    def increment(self, counter: str, amount: int = 1) -> int:
        """Increment a counter, return new value"""
        self.counters[counter] = self.counters.get(counter, 0) + amount
        return self.counters[counter]
    
    def get_counter(self, counter: str, default: int = 0) -> int:
        """Get counter value"""
        return self.counters.get(counter, default)
    
    def add_to_inventory(self, item_id: str) -> None:
        """Add item to inventory"""
        if item_id not in self.inventory:
            self.inventory.append(item_id)
    
    def remove_from_inventory(self, item_id: str) -> bool:
        """Remove item from inventory"""
        if item_id in self.inventory:
            self.inventory.remove(item_id)
            return True
        return False
    
    def has_item(self, item_id: str) -> bool:
        """Check if item in inventory"""
        return item_id in self.inventory
    
    def add_memory(self, memory_type: str, content: Any, source: str = None) -> None:
        """Add a memory"""
        self.memories.append({
            'type': memory_type,
            'content': content,
            'source': source,
            'timestamp': time.time()
        })
    
    def clone(self) -> 'TraversalContext':
        """Create a deep copy for branching"""
        return TraversalContext(
            current_node=self.current_node,
            previous_node=self.previous_node,
            path=self.path.copy(),
            visited=self.visited.copy(),
            depth=self.depth,
            step_count=self.step_count,
            current_layer=self.current_layer,
            layer_stack=self.layer_stack.copy(),
            flags=self.flags.copy(),
            counters=self.counters.copy(),
            inventory=self.inventory.copy(),
            memories=self.memories.copy(),
            buffer=self.buffer.copy(),
            graph_ref=self.graph_ref,
            start_time=self.start_time
        )
    
    def merge(self, other: 'TraversalContext') -> 'TraversalContext':
        """Merge another context into this one"""
        # Combine visited sets
        self.visited |= other.visited
        
        # Merge flags (other overwrites)
        self.flags.update(other.flags)
        
        # Merge counters (take max)
        for k, v in other.counters.items():
            self.counters[k] = max(self.counters.get(k, 0), v)
        
        # Merge inventory (union)
        for item in other.inventory:
            if item not in self.inventory:
                self.inventory.append(item)
        
        # Append memories
        self.memories.extend(other.memories)
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'current_node_id': self.get_node_id(self.current_node) if self.current_node else None,
            'previous_node_id': self.get_node_id(self.previous_node) if self.previous_node else None,
            'path': [self.get_node_id(n) for n in self.path],
            'visited': list(self.visited),
            'depth': self.depth,
            'step_count': self.step_count,
            'current_layer': self.current_layer,
            'layer_stack': self.layer_stack,
            'flags': self.flags,
            'counters': self.counters,
            'inventory': self.inventory,
            'memories': self.memories,
            'buffer': self.buffer,
            'start_time': self.start_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraversalContext':
        """Deserialize from dictionary"""
        ctx = cls()
        ctx.visited = set(data.get('visited', []))
        ctx.depth = data.get('depth', 0)
        ctx.step_count = data.get('step_count', 0)
        ctx.current_layer = data.get('current_layer', 'default')
        ctx.layer_stack = data.get('layer_stack', [])
        ctx.flags = data.get('flags', {})
        ctx.counters = data.get('counters', {})
        ctx.inventory = data.get('inventory', [])
        ctx.memories = data.get('memories', [])
        ctx.buffer = data.get('buffer', {})
        ctx.start_time = data.get('start_time', time.time())
        return ctx


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 7: TRAVERSAL WRAPPER                                      ║
# ║                                                                           ║
# ║  Smart iterator that carries context through traversal.                   ║
# ║  __iter__() returns self - the wrapper IS the iterator.                   ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class TraversalWrapper(Generic[T]):
    """
    Smart iterator wrapper that carries context through traversal.
    
    THE KEY INSIGHT: __iter__() returns self.
    The wrapper IS the iterator. It travels with you through the for loop.
    """
    
    def __init__(self,
                 source: Union[Iterable[T], Callable[[], Iterable[T]]] = None,
                 context: TraversalContext = None,
                 layer_registry: LayerRegistry = None,
                 observer_bus: ObserverBus = None):
        
        self._source = source
        self._source_factory = source if callable(source) else None
        self._iterator: Optional[Iterator[T]] = None
        
        self.context = context or TraversalContext()
        self.layers = layer_registry or LayerRegistry()
        self._observer_bus = observer_bus
        
        # Callbacks
        self._on_step: List[Callable[['TraversalWrapper', T], Any]] = []
        self._on_enter: List[Callable[['TraversalWrapper'], Any]] = []
        self._on_exit: List[Callable[['TraversalWrapper'], Any]] = []
        self._on_layer_switch: List[Callable[['TraversalWrapper', str, str], Any]] = []
        
        # State
        self._current: Optional[T] = None
        self._exhausted = False
    
    # ─────────────────────────────────────────────────────────────────────────
    # Iterator Protocol
    # ─────────────────────────────────────────────────────────────────────────
    
    def __iter__(self) -> 'TraversalWrapper[T]':
        """
        Start iteration - returns SELF.
        
        This is the key insight. The wrapper persists through the for loop.
        """
        if self._source_factory:
            self._source = self._source_factory()
        
        if self._source is not None:
            self._iterator = iter(self._source)
        
        self._exhausted = False
        self._fire_on_enter()
        
        return self
    
    def __next__(self) -> T:
        """Get next item, updating context"""
        if self._exhausted:
            raise StopIteration
        
        if self._iterator is None:
            raise StopIteration
        
        try:
            item = next(self._iterator)
            
            # Update context
            self.context.previous_node = self._current
            self._current = item
            self.context.current_node = item
            self.context.path.append(item)
            self.context.mark_visited(item)
            self.context.depth = len(self.context.path)
            self.context.step_count += 1
            
            # Fire callbacks
            self._fire_on_step(item)
            
            # Emit signal
            if self._observer_bus:
                self._observer_bus.emit_type(
                    SignalType.TRAVERSAL_STEP,
                    source_id=self.context.get_node_id(item),
                    data={'depth': self.context.depth, 'step': self.context.step_count}
                )
            
            # Check auto layer switch
            new_layer = self.layers.check_auto_switch(self.context)
            if new_layer and new_layer != self.context.current_layer:
                self.switch_layer(new_layer)
            
            return item
            
        except StopIteration:
            self._exhausted = True
            self._fire_on_exit()
            raise
    
    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def current(self) -> Optional[T]:
        """Current item"""
        return self._current
    
    @property
    def previous(self) -> Optional[T]:
        """Previous item"""
        return self.context.previous_node
    
    @property
    def depth(self) -> int:
        """Current depth"""
        return self.context.depth
    
    @property
    def step_count(self) -> int:
        """Total steps taken"""
        return self.context.step_count
    
    @property
    def layer(self) -> str:
        """Current layer name"""
        return self.context.current_layer
    
    # ─────────────────────────────────────────────────────────────────────────
    # Layer Switching
    # ─────────────────────────────────────────────────────────────────────────
    
    def switch_layer(self, layer_name: str, push: bool = False) -> None:
        """
        Switch to a different layer.
        
        If push=True, saves current layer for later pop.
        """
        old_layer = self.context.current_layer
        
        if push:
            self.context.push_layer(layer_name)
        else:
            self.context.current_layer = layer_name
        
        self.layers.set_active(layer_name)
        
        # Fire callbacks
        self._fire_on_layer_switch(old_layer, layer_name)
        
        # Emit signal
        if self._observer_bus:
            self._observer_bus.emit_type(
                SignalType.LAYER_SWITCH,
                data={'old_layer': old_layer, 'new_layer': layer_name}
            )
    
    def pop_layer(self) -> str:
        """Pop back to previous layer"""
        old_layer = self.context.current_layer
        new_layer = self.context.pop_layer()
        self.layers.set_active(new_layer)
        
        self._fire_on_layer_switch(old_layer, new_layer)
        
        if self._observer_bus:
            self._observer_bus.emit_type(
                SignalType.LAYER_SWITCH,
                data={'old_layer': old_layer, 'new_layer': new_layer}
            )
        
        return new_layer
    
    # ─────────────────────────────────────────────────────────────────────────
    # Graph Operations
    # ─────────────────────────────────────────────────────────────────────────
    
    def swap_graph(self, new_graph: Any, reset_position: bool = False) -> None:
        """
        Hotswap the underlying graph.
        
        Context persists - inventory, flags, memories carry over.
        """
        self.context.graph_ref = new_graph
        
        if reset_position:
            self._current = None
            self.context.current_node = None
            self.context.path = []
            self.context.visited = set()
        
        if self._observer_bus:
            self._observer_bus.emit_type(
                SignalType.GRAPH_SWAP,
                data={'reset': reset_position}
            )
    
    def swap_source(self, new_source: Union[Iterable[T], Callable]) -> None:
        """Replace iteration source mid-traversal"""
        self._source = new_source
        self._source_factory = new_source if callable(new_source) else None
        
        if self._source_factory:
            self._source = self._source_factory()
        
        self._iterator = iter(self._source) if self._source else None
        self._exhausted = False
    
    def get_available_edges(self) -> List[Any]:
        """Get edges available from current node"""
        if self._current is None or self.context.graph_ref is None:
            return []
        
        graph = self.context.graph_ref
        if hasattr(graph, 'get_outgoing_links'):
            return graph.get_outgoing_links(self.context.get_node_id(self._current))
        
        return []
    
    def jump_to(self, node: T) -> None:
        """Jump directly to a node (teleport)"""
        self.context.previous_node = self._current
        self._current = node
        self.context.current_node = node
        self.context.path.append(node)
        self.context.mark_visited(node)
        self.context.step_count += 1
        
        self._fire_on_step(node)
    
    def branch(self) -> 'TraversalWrapper[T]':
        """Create a branch (clone) of this wrapper"""
        branched = TraversalWrapper(
            source=None,
            context=self.context.clone(),
            layer_registry=self.layers,
            observer_bus=self._observer_bus
        )
        branched._current = self._current
        return branched
    
    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────
    
    def on_step(self, callback: Callable[['TraversalWrapper', T], Any]) -> 'TraversalWrapper[T]':
        """Register step callback"""
        self._on_step.append(callback)
        return self
    
    def on_enter(self, callback: Callable[['TraversalWrapper'], Any]) -> 'TraversalWrapper[T]':
        """Register enter callback"""
        self._on_enter.append(callback)
        return self
    
    def on_exit(self, callback: Callable[['TraversalWrapper'], Any]) -> 'TraversalWrapper[T]':
        """Register exit callback"""
        self._on_exit.append(callback)
        return self
    
    def on_layer_switch(self, callback: Callable[['TraversalWrapper', str, str], Any]) -> 'TraversalWrapper[T]':
        """Register layer switch callback"""
        self._on_layer_switch.append(callback)
        return self
    
    def _fire_on_step(self, item: T) -> None:
        for cb in self._on_step:
            try:
                cb(self, item)
            except Exception:
                pass
    
    def _fire_on_enter(self) -> None:
        for cb in self._on_enter:
            try:
                cb(self)
            except Exception:
                pass
    
    def _fire_on_exit(self) -> None:
        for cb in self._on_exit:
            try:
                cb(self)
            except Exception:
                pass
    
    def _fire_on_layer_switch(self, old: str, new: str) -> None:
        for cb in self._on_layer_switch:
            try:
                cb(self, old, new)
            except Exception:
                pass
    
    # ─────────────────────────────────────────────────────────────────────────
    # Convenience Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def collect(self) -> List[T]:
        """Consume iterator, return all items"""
        return list(self)
    
    def take(self, n: int) -> List[T]:
        """Take up to n items"""
        result = []
        for item in self:
            result.append(item)
            if len(result) >= n:
                break
        return result
    
    def skip(self, n: int) -> 'TraversalWrapper[T]':
        """Skip n items"""
        for _ in range(n):
            try:
                next(self)
            except StopIteration:
                break
        return self


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 8: TEXT EXTRACTION                                        ║
# ║                                                                           ║
# ║  Extract structured data from source text.                                ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class ExtractedWord:
    """A single word extracted from text"""
    text: str
    word_class: WordClass = WordClass.UNKNOWN
    position: int = 0
    sentence_idx: int = 0
    paragraph_idx: int = 0
    
    # Entity reference
    entity_id: Optional[str] = None
    is_entity_reference: bool = False
    
    # Context
    surrounding: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)
    
    # Source
    source_sentence: str = ""


@dataclass
class ExtractedEntity:
    """An entity extracted from text"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entity_type: EntityType = EntityType.ITEM
    
    # References
    word_refs: List[ExtractedWord] = field(default_factory=list)
    
    # Attributes
    adjectives: Set[str] = field(default_factory=set)
    descriptions: List[str] = field(default_factory=list)
    
    # Position
    first_mention: int = 0
    mention_count: int = 1
    
    # Confidence
    confidence: float = 0.5


@dataclass
class ExtractedRelation:
    """A relation between two extracted entities"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_type: LinkType = LinkType.RELATIONAL
    relation_word: str = ""
    
    source_sentence: str = ""
    confidence: float = 0.5


@dataclass
class ExtractedFragment:
    """A prose fragment extracted from text"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    category: FragmentCategory = FragmentCategory.BASE_DESCRIPTION
    
    # Association
    entity_id: Optional[str] = None
    
    # Source
    source_position: int = 0


@dataclass
class ExtractionResult:
    """Complete result of text extraction"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_text: str = ""
    
    words: List[ExtractedWord] = field(default_factory=list)
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    fragments: List[ExtractedFragment] = field(default_factory=list)
    
    # Stats
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    
    # Timing
    extraction_time: float = 0.0
    
    def summary(self) -> str:
        """Get summary of extraction"""
        lines = [
            f"Extraction ID: {self.id}",
            f"Words: {len(self.words)}",
            f"Entities: {len(self.entities)}",
            f"Relations: {len(self.relations)}",
            f"Fragments: {len(self.fragments)}",
            "",
            "Entities:"
        ]
        for e in self.entities:
            lines.append(f"  {e.name} ({e.entity_type.name}) - {e.mention_count} mentions")
        
        return "\n".join(lines)


class TextExtractor:
    """
    Extract structured data from source text.
    
    Analyzes every word, identifies entities, finds relations,
    extracts prose fragments.
    """
    
    # Word indicators
    LOCATION_WORDS = {
        'room', 'house', 'building', 'street', 'city', 'forest',
        'cave', 'castle', 'tower', 'garden', 'kitchen', 'bedroom',
        'hall', 'corridor', 'path', 'road', 'bridge', 'door', 'gate'
    }
    
    CHARACTER_WORDS = {
        'man', 'woman', 'person', 'boy', 'girl', 'child', 'king',
        'queen', 'wizard', 'witch', 'guard', 'merchant', 'stranger',
        'hero', 'villain', 'knight', 'servant'
    }
    
    ITEM_WORDS = {
        'key', 'sword', 'book', 'letter', 'coin', 'ring', 'box',
        'chest', 'lamp', 'candle', 'bottle', 'potion', 'scroll',
        'map', 'torch', 'dagger', 'shield'
    }
    
    SPATIAL_PREPOSITIONS = {
        'in', 'on', 'under', 'near', 'beside', 'behind', 'above',
        'below', 'inside', 'outside', 'through', 'between', 'at'
    }
    
    def __init__(self, observer_bus: ObserverBus = None):
        self._observer_bus = observer_bus
        
        # State (reset per extraction)
        self._words: List[ExtractedWord] = []
        self._entities: Dict[str, ExtractedEntity] = {}
        self._relations: List[ExtractedRelation] = []
        self._fragments: List[ExtractedFragment] = []
        self._name_to_entity: Dict[str, str] = {}
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Main extraction method.
        
        Takes raw text, returns structured extraction result.
        """
        start_time = time.time()
        
        # Reset state
        self._words = []
        self._entities = {}
        self._relations = []
        self._fragments = []
        self._name_to_entity = {}
        
        # Emit start signal
        if self._observer_bus:
            self._observer_bus.emit_type(SignalType.EXTRACTION_START)
        
        # Split into paragraphs and sentences
        paragraphs = self._split_paragraphs(text)
        
        position = 0
        for para_idx, paragraph in enumerate(paragraphs):
            sentences = self._split_sentences(paragraph)
            
            for sent_idx, sentence in enumerate(sentences):
                # Extract words
                words = self._extract_words(sentence, position, sent_idx, para_idx)
                self._words.extend(words)
                
                # Analyze for entities
                self._analyze_sentence(words, sentence)
                
                # Extract fragment
                fragment = self._extract_fragment(sentence, sent_idx, para_idx)
                if fragment:
                    self._fragments.append(fragment)
                
                position += len(sentence) + 1
        
        # Post-process
        self._resolve_references()
        self._calculate_confidences()
        
        # Build result
        result = ExtractionResult(
            source_text=text,
            words=self._words,
            entities=list(self._entities.values()),
            relations=self._relations,
            fragments=self._fragments,
            word_count=len(self._words),
            sentence_count=sum(len(self._split_sentences(p)) for p in paragraphs),
            paragraph_count=len(paragraphs),
            extraction_time=time.time() - start_time
        )
        
        # Emit complete signal
        if self._observer_bus:
            self._observer_bus.emit_type(
                SignalType.EXTRACTION_COMPLETE,
                data={'word_count': result.word_count, 'entity_count': len(result.entities)}
            )
        
        return result
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # TODO: Implement paragraph splitting
        ...
    
    def _split_sentences(self, paragraph: str) -> List[str]:
        """Split paragraph into sentences"""
        # TODO: Implement sentence splitting
        ...
    
    def _extract_words(self, sentence: str, position: int,
                       sent_idx: int, para_idx: int) -> List[ExtractedWord]:
        """Extract and classify words from sentence"""
        # TODO: Implement word extraction
        ...
    
    def _classify_word(self, word: str, position: int, context: List[str]) -> WordClass:
        """Classify a word grammatically"""
        # TODO: Implement word classification
        ...
    
    def _infer_entity_type(self, word: str) -> Optional[EntityType]:
        """Infer entity type from word"""
        word_lower = word.lower()
        
        if word_lower in self.LOCATION_WORDS:
            return EntityType.LOCATION
        if word_lower in self.CHARACTER_WORDS:
            return EntityType.CHARACTER
        if word_lower in self.ITEM_WORDS:
            return EntityType.ITEM
        
        return None
    
    def _analyze_sentence(self, words: List[ExtractedWord], sentence: str) -> None:
        """Analyze sentence for entities and relations"""
        # TODO: Implement sentence analysis
        ...
    
    def _extract_fragment(self, sentence: str, sent_idx: int,
                         para_idx: int) -> Optional[ExtractedFragment]:
        """Extract prose fragment from sentence"""
        # TODO: Implement fragment extraction
        ...
    
    def _resolve_references(self) -> None:
        """Resolve pronouns and other references"""
        # TODO: Implement reference resolution
        ...
    
    def _calculate_confidences(self) -> None:
        """Calculate confidence scores"""
        # TODO: Implement confidence calculation
        ...


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 9: WHITE ROOM BUILDER                                     ║
# ║                                                                           ║
# ║  Build a game world (GlyphGraph) from extraction results.                 ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class WhiteRoomBuilder:
    """
    Build a GlyphGraph (game world) from extraction results.
    
    The White Room is the origin - a featureless space where
    the world forms from extracted text.
    """
    
    def __init__(self, 
                 embedding_provider: EmbeddingProvider = None,
                 observer_bus: ObserverBus = None):
        self._embedding_provider = embedding_provider
        self._observer_bus = observer_bus
        
        # State (reset per build)
        self._graph: Optional[GlyphGraph] = None
        self._white_room: Optional[GlyphNode] = None
        self._id_map: Dict[str, str] = {}  # extraction_id → graph_id
    
    def build(self, extraction: ExtractionResult) -> GlyphGraph:
        """
        Build a GlyphGraph from extraction result.
        
        Creates the White Room origin, converts entities to nodes,
        converts relations to edges, assigns fragments.
        """
        # Create graph
        self._graph = GlyphGraph(
            name="White Room",
            embedding_provider=self._embedding_provider
        )
        
        if self._observer_bus:
            self._graph.set_observer_bus(self._observer_bus)
        
        self._id_map = {}
        
        # Create White Room origin
        self._create_white_room()
        
        # Convert entities to nodes
        for ext_entity in extraction.entities:
            self._convert_entity(ext_entity)
        
        # Convert relations to edges
        for ext_relation in extraction.relations:
            self._convert_relation(ext_relation)
        
        # Assign fragments
        self._assign_fragments(extraction.fragments)
        
        # Connect orphans to White Room
        self._connect_orphans()
        
        return self._graph
    
    def _create_white_room(self) -> None:
        """Create the White Room origin node"""
        self._white_room = GlyphNode(
            name="The White Room",
            node_type="location",
            embedding_text="A featureless white space where the world forms around you.",
            tags={"origin", "location"}
        )
        self._graph.add_node(self._white_room)
    
    def _convert_entity(self, ext_entity: ExtractedEntity) -> Optional[str]:
        """Convert extracted entity to GlyphNode"""
        # TODO: Implement entity conversion
        ...
    
    def _convert_relation(self, ext_relation: ExtractedRelation) -> Optional[str]:
        """Convert extracted relation to GlyphEdge"""
        # TODO: Implement relation conversion
        ...
    
    def _assign_fragments(self, fragments: List[ExtractedFragment]) -> None:
        """Assign fragments to nodes"""
        # TODO: Implement fragment assignment
        ...
    
    def _connect_orphans(self) -> None:
        """Connect nodes without edges to White Room"""
        # TODO: Implement orphan connection
        ...


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 10: LOOKAHEAD ENGINE                                      ║
# ║                                                                           ║
# ║  Pre-traverse possibility space for hints and clues.                      ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class PossibilityType(Enum):
    """Types of possibilities lookahead can find"""
    REACHABLE_LOCATION = auto()
    BLOCKED_PATH = auto()
    LOCKED_DOOR = auto()
    VISIBLE_ITEM = auto()
    HIDDEN_ITEM = auto()
    TAKEABLE_ITEM = auto()
    REQUIRED_ITEM = auto()
    UNLOCKABLE = auto()
    REVEALABLE = auto()
    TRIGGERABLE = auto()
    NEAR_MISS = auto()
    SIMILAR_NODE = auto()  # Semantic similarity


@dataclass
class Possibility:
    """A single possibility discovered by lookahead"""
    possibility_type: PossibilityType
    entity_id: str
    entity_name: str
    
    distance: int = 0
    similarity: float = 0.0  # For semantic matches
    
    conditions_met: List[str] = field(default_factory=list)
    conditions_unmet: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    
    hint_text: Optional[str] = None
    path: List[str] = field(default_factory=list)
    
    via_link_type: Optional[str] = None
    via_layer: Optional[str] = None
    
    def is_blocked(self) -> bool:
        return len(self.conditions_unmet) > 0
    
    def is_near_miss(self, threshold: int = 1) -> bool:
        return 0 < len(self.conditions_unmet) <= threshold


@dataclass
class LookaheadResult:
    """Complete lookahead result"""
    origin_id: str
    origin_name: str
    max_depth: int
    
    all_possibilities: List[Possibility] = field(default_factory=list)
    by_type: Dict[PossibilityType, List[Possibility]] = field(
        default_factory=lambda: defaultdict(list)
    )
    by_entity: Dict[str, List[Possibility]] = field(
        default_factory=lambda: defaultdict(list)
    )
    
    # Stats
    entities_seen: int = 0
    links_traversed: int = 0
    blocked_count: int = 0
    near_miss_count: int = 0
    
    def add(self, p: Possibility) -> None:
        """Add a possibility"""
        self.all_possibilities.append(p)
        self.by_type[p.possibility_type].append(p)
        self.by_entity[p.entity_id].append(p)
        
        if p.is_blocked():
            self.blocked_count += 1
        if p.is_near_miss():
            self.near_miss_count += 1
    
    def get_reachable(self) -> List[Possibility]:
        """Get reachable locations"""
        return [p for p in self.all_possibilities
                if p.possibility_type == PossibilityType.REACHABLE_LOCATION
                and not p.is_blocked()]
    
    def get_blocked(self) -> List[Possibility]:
        """Get blocked possibilities"""
        return [p for p in self.all_possibilities if p.is_blocked()]
    
    def get_near_misses(self, threshold: int = 1) -> List[Possibility]:
        """Get near-miss possibilities"""
        return [p for p in self.all_possibilities if p.is_near_miss(threshold)]
    
    def get_similar(self, min_similarity: float = 0.7) -> List[Possibility]:
        """Get semantically similar nodes"""
        return [p for p in self.by_type.get(PossibilityType.SIMILAR_NODE, [])
                if p.similarity >= min_similarity]
    
    def get_hints(self, max_hints: int = 3) -> List[str]:
        """Generate hint text"""
        hints = []
        
        # Near-misses first
        for p in self.get_near_misses():
            if p.hint_text:
                hints.append(p.hint_text)
            if len(hints) >= max_hints:
                break
        
        # Then hidden items
        for p in self.by_type.get(PossibilityType.HIDDEN_ITEM, []):
            if p.distance <= 1 and p.hint_text and len(hints) < max_hints:
                hints.append(p.hint_text)
        
        return hints[:max_hints]


class LookaheadEngine:
    """
    Pre-traverse the graph to discover possibilities.
    
    Explores without moving - finds what COULD happen,
    what's blocked, what's close, what's similar.
    """
    
    def __init__(self, graph: GlyphGraph = None):
        self.graph = graph
    
    def set_graph(self, graph: GlyphGraph) -> None:
        """Set or change the graph"""
        self.graph = graph
    
    def lookahead(self,
                  from_node: Union[str, GlyphNode],
                  context: TraversalContext = None,
                  max_depth: int = 3,
                  include_semantic: bool = True,
                  semantic_k: int = 5,
                  semantic_threshold: float = 0.7) -> LookaheadResult:
        """
        Perform lookahead from a node.
        
        Args:
            from_node: Starting node (ID or object)
            context: Current traversal context
            max_depth: Maximum graph distance
            include_semantic: Include semantic similarity results
            semantic_k: Number of semantic matches to find
            semantic_threshold: Minimum similarity for semantic matches
        
        Returns:
            LookaheadResult with all discovered possibilities
        """
        if self.graph is None:
            raise ValueError("No graph set")
        
        # Resolve node
        if isinstance(from_node, str):
            start_id = from_node
            start_node = self.graph.get_node(from_node)
        else:
            start_id = from_node.id
            start_node = from_node
        
        if start_node is None:
            raise ValueError(f"Node not found: {start_id}")
        
        result = LookaheadResult(
            origin_id=start_id,
            origin_name=start_node.name,
            max_depth=max_depth
        )
        
        ctx = context or TraversalContext()
        
        # BFS for graph traversal
        self._bfs_lookahead(start_id, max_depth, ctx, result)
        
        # Semantic similarity search
        if include_semantic and start_node.embedding:
            self._semantic_lookahead(start_node, semantic_k, semantic_threshold, result)
        
        return result
    
    def _bfs_lookahead(self, start_id: str, max_depth: int,
                       context: TraversalContext, result: LookaheadResult) -> None:
        """BFS-based lookahead"""
        # TODO: Implement BFS lookahead
        ...
    
    def _semantic_lookahead(self, node: GlyphNode, k: int,
                           threshold: float, result: LookaheadResult) -> None:
        """Semantic similarity lookahead"""
        similar = self.graph.find_similar(node, k + 1, threshold)
        
        for similar_node, similarity in similar:
            if similar_node.id != node.id:
                result.add(Possibility(
                    possibility_type=PossibilityType.SIMILAR_NODE,
                    entity_id=similar_node.id,
                    entity_name=similar_node.name,
                    similarity=similarity,
                    via_layer="semantic",
                    hint_text=f"{similar_node.name} seems related..."
                ))
    
    def _analyze_node(self, node: GlyphNode, depth: int,
                      path: List[str], context: TraversalContext) -> List[Possibility]:
        """Analyze a node for possibilities"""
        # TODO: Implement node analysis
        ...
    
    def _analyze_edge(self, edge: GlyphEdge, depth: int,
                      path: List[str], context: TraversalContext) -> List[Possibility]:
        """Analyze an edge for possibilities"""
        # TODO: Implement edge analysis
        ...


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 11: CONVENIENCE FUNCTIONS                                 ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def extract_text(text: str, observer_bus: ObserverBus = None) -> ExtractionResult:
    """Extract structured data from text"""
    extractor = TextExtractor(observer_bus)
    return extractor.extract(text)


def build_white_room(extraction: ExtractionResult,
                     embedding_provider: EmbeddingProvider = None,
                     observer_bus: ObserverBus = None) -> GlyphGraph:
    """Build a GlyphGraph from extraction"""
    builder = WhiteRoomBuilder(embedding_provider, observer_bus)
    return builder.build(extraction)


def text_to_world(text: str,
                  embedding_provider: EmbeddingProvider = None,
                  observer_bus: ObserverBus = None) -> GlyphGraph:
    """Complete pipeline: text → extraction → GlyphGraph"""
    extraction = extract_text(text, observer_bus)
    return build_white_room(extraction, embedding_provider, observer_bus)


def smart_iter(source: Iterable[T],
               context: TraversalContext = None,
               layer_registry: LayerRegistry = None,
               observer_bus: ObserverBus = None) -> TraversalWrapper[T]:
    """Create a smart iterator"""
    return TraversalWrapper(source, context, layer_registry, observer_bus)


def lookahead_from(graph: GlyphGraph,
                   node: Union[str, GlyphNode],
                   context: TraversalContext = None,
                   max_depth: int = 3) -> LookaheadResult:
    """One-shot lookahead"""
    engine = LookaheadEngine(graph)
    return engine.lookahead(node, context, max_depth)


def create_observer_bus() -> ObserverBus:
    """Create a new observer bus"""
    return ObserverBus()


# ═══════════════════════════════════════════════════════════════════════════════
#                              MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    'Priority', 'EntityType', 'EntityState', 'LinkType', 'LayerType',
    'SignalType', 'WordClass', 'FragmentCategory', 'StorageType',
    'PossibilityType',
    
    # Protocols
    'Identifiable', 'Named', 'Serializable', 'Observable', 'Embeddable',
    
    # Base Classes
    'BaseIdentifiable', 'Context',
    
    # Observer System
    'Signal', 'Observer', 'FunctionObserver', 'TypedObserver',
    'StateObserver', 'EntityObserver', 'TraversalObserver', 'ExtractionObserver',
    'ObserverBus',
    
    # Embedding & Storage
    'Embedding', 'EmbeddingProvider', 'SimpleHashEmbedding',
    'VectorIndex', 'FlatVectorIndex',
    'GlyphNode', 'GlyphEdge', 'GlyphGraph',
    
    # Layer System
    'LayerConfig', 'LayerRegistry', 'create_standard_layers',
    
    # Traversal
    'TraversalContext', 'TraversalWrapper',
    
    # Extraction
    'ExtractedWord', 'ExtractedEntity', 'ExtractedRelation', 'ExtractedFragment',
    'ExtractionResult', 'TextExtractor',
    
    # White Room
    'WhiteRoomBuilder',
    
    # Lookahead
    'Possibility', 'LookaheadResult', 'LookaheadEngine',
    
    # Convenience
    'extract_text', 'build_white_room', 'text_to_world',
    'smart_iter', 'lookahead_from', 'create_observer_bus',
]


# ═══════════════════════════════════════════════════════════════════════════════
#                              VERSION INFO
# ═══════════════════════════════════════════════════════════════════════════════

__version__ = "0.1.0"
__author__ = "CloudyCadet"
__project__ = "matts"


if __name__ == "__main__":
    print(f"MATTS Complete System Stubs v{__version__}")
    print(f"Classes defined: {len(__all__)}")
    print("\nRun tests or import specific classes.")
