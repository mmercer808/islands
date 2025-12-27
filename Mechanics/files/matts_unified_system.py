#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║              M A T T S   U N I F I E D   S Y S T E M                          ║
║                                                                               ║
║         Complete Stubs with Embedding Storage & Generic Observers             ║
║                                                                               ║
║  This file extends your existing IslandGraph/GraphWalker with:                ║
║  - Embedding storage type (vectors for semantic operations)                   ║
║  - Generic Observer pattern (works with ANY input class)                      ║
║  - Text extraction pipeline                                                   ║
║  - White Room builder                                                         ║
║  - Multi-layer traversal                                                      ║
║  - Lookahead engine                                                           ║
║                                                                               ║
║  Integration with your existing files:                                        ║
║  - graph_core.py → IslandGraph, GraphNode, Edge                               ║
║  - graph_walker.py → GraphWalker, WalkerContext                               ║
║  - my_infocom_entity_system.py → Entity, Link, EntityGraph                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, List, Set, Optional, Any, Callable, Iterator, Iterable,
    Tuple, Union, Generic, TypeVar, Type, Protocol, Sequence,
    Mapping, MutableMapping, runtime_checkable, Awaitable
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
import re


# ═══════════════════════════════════════════════════════════════════════════════
#                              TYPE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

T = TypeVar('T')                    # Any type
N = TypeVar('N')                    # Node type
E = TypeVar('E')                    # Entity/Edge type
S = TypeVar('S', bound='Signal')    # Signal subtype
O = TypeVar('O', bound='Observer')  # Observer subtype
C = TypeVar('C', bound='Context')   # Context subtype
P = TypeVar('P')                    # Payload type


# ═══════════════════════════════════════════════════════════════════════════════
#                              PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class HasId(Protocol):
    """Any object with an id property"""
    @property
    def id(self) -> str: ...


@runtime_checkable
class HasName(Protocol):
    """Any object with a name property"""
    @property
    def name(self) -> str: ...


@runtime_checkable
class Serializable(Protocol):
    """Objects that can serialize to/from dict"""
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable': ...


@runtime_checkable
class Embeddable(Protocol):
    """Objects that can be converted to vector embeddings"""
    def to_embedding_text(self) -> str: ...


@runtime_checkable
class Observable(Protocol):
    """Objects that can be observed"""
    def add_observer(self, observer: 'Observer') -> str: ...
    def remove_observer(self, observer_id: str) -> bool: ...
    def notify(self, signal: 'Signal') -> None: ...


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 1: ENUMERATIONS                                           ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class Priority(IntEnum):
    """Universal priority levels (lower = higher priority)"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class StorageType(Enum):
    """
    Storage backend types.
    
    EMBEDDING is a first-class storage type for vector operations.
    """
    MEMORY = "memory"           # In-memory dict
    FILE = "file"               # JSON/pickle file
    SQLITE = "sqlite"           # SQLite database
    EMBEDDING = "embedding"     # Vector store (semantic)
    DISTRIBUTED = "distributed" # Distributed/networked


class SignalType(Enum):
    """Signal types for the observer system"""
    # State machine
    STATE_ENTER = "state_enter"
    STATE_EXIT = "state_exit"
    STATE_CHANGE = "state_change"
    
    # Entity lifecycle
    ENTITY_CREATED = "entity_created"
    ENTITY_MODIFIED = "entity_modified"
    ENTITY_DESTROYED = "entity_destroyed"
    ENTITY_LINKED = "entity_linked"
    ENTITY_UNLINKED = "entity_unlinked"
    
    # Traversal
    TRAVERSAL_START = "traversal_start"
    TRAVERSAL_STEP = "traversal_step"
    TRAVERSAL_END = "traversal_end"
    LAYER_SWITCH = "layer_switch"
    GRAPH_SWAP = "graph_swap"
    
    # Extraction
    EXTRACTION_START = "extraction_start"
    EXTRACTION_WORD = "extraction_word"
    EXTRACTION_ENTITY = "extraction_entity"
    EXTRACTION_RELATION = "extraction_relation"
    EXTRACTION_COMPLETE = "extraction_complete"
    
    # Embedding
    EMBEDDING_CREATED = "embedding_created"
    EMBEDDING_INDEXED = "embedding_indexed"
    SIMILARITY_QUERY = "similarity_query"
    
    # System
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


class PossibilityType(Enum):
    """Types of possibilities for lookahead"""
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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 2: EMBEDDING STORAGE SYSTEM                               ║
# ║                                                                           ║
# ║  Embeddings are a STORAGE TYPE - vectors for semantic operations.         ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class Embedding:
    """
    A vector embedding for semantic operations.
    
    This is the core unit of EMBEDDING storage type.
    Supports similarity search, clustering, semantic layers.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vector: List[float] = field(default_factory=list)
    dimensions: int = 0
    
    # What was embedded
    source_id: str = ""
    source_type: str = ""
    source_text: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    # Normalization state
    normalized: bool = False
    
    def __post_init__(self):
        self.dimensions = len(self.vector)
    
    def normalize(self) -> 'Embedding':
        """Return L2-normalized embedding (unit length)"""
        if self.normalized or not self.vector:
            return self
        
        magnitude = sum(x * x for x in self.vector) ** 0.5
        if magnitude == 0:
            return self
        
        normalized_vector = [x / magnitude for x in self.vector]
        return Embedding(
            id=self.id,
            vector=normalized_vector,
            source_id=self.source_id,
            source_type=self.source_type,
            source_text=self.source_text,
            metadata=self.metadata.copy(),
            created_at=self.created_at,
            normalized=True
        )
    
    def cosine_similarity(self, other: 'Embedding') -> float:
        """Compute cosine similarity with another embedding"""
        if len(self.vector) != len(other.vector):
            raise ValueError("Dimension mismatch")
        
        # Normalize both
        a = self.normalize().vector
        b = other.normalize().vector
        
        # Dot product of unit vectors = cosine similarity
        return sum(x * y for x, y in zip(a, b))
    
    def euclidean_distance(self, other: 'Embedding') -> float:
        """Compute Euclidean distance to another embedding"""
        if len(self.vector) != len(other.vector):
            raise ValueError("Dimension mismatch")
        
        return sum((x - y) ** 2 for x, y in zip(self.vector, other.vector)) ** 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'id': self.id,
            'vector': self.vector,
            'dimensions': self.dimensions,
            'source_id': self.source_id,
            'source_type': self.source_type,
            'source_text': self.source_text,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'normalized': self.normalized
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Embedding':
        """Deserialize from dictionary"""
        return cls(**data)


class EmbeddingProvider(ABC):
    """
    Abstract provider for creating embeddings.
    
    Implement for different backends:
    - Local models (sentence-transformers)
    - API (OpenAI, Cohere)
    - Hash-based (for testing)
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
    
    def embed_object(self, obj: Any) -> Embedding:
        """Embed any object that implements Embeddable or has text representation"""
        if isinstance(obj, Embeddable):
            text = obj.to_embedding_text()
        elif hasattr(obj, 'name') and hasattr(obj, 'data'):
            # GraphNode-like
            text = f"{obj.name} {obj.data.get('description', '')}"
        elif hasattr(obj, 'name'):
            text = str(obj.name)
        else:
            text = str(obj)
        
        embedding = self.embed_text(text)
        embedding.source_id = getattr(obj, 'id', str(id(obj)))
        embedding.source_type = type(obj).__name__
        return embedding


class HashEmbeddingProvider(EmbeddingProvider):
    """
    Hash-based embedding provider for testing.
    
    Creates deterministic embeddings from text using SHA256.
    NOT semantically meaningful, but useful for testing.
    """
    
    def __init__(self, dimensions: int = 128):
        self._dimensions = dimensions
    
    def embed_text(self, text: str) -> Embedding:
        """Create hash-based embedding"""
        # Hash the text
        hash_bytes = hashlib.sha256(text.encode()).digest()
        
        # Expand to desired dimensions
        vector = []
        for i in range(self._dimensions):
            byte_idx = i % len(hash_bytes)
            # Convert to float in [-1, 1]
            value = (hash_bytes[byte_idx] / 127.5) - 1.0
            vector.append(value)
        
        return Embedding(
            vector=vector,
            source_text=text[:100]  # Truncate for storage
        )
    
    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        return [self.embed_text(t) for t in texts]
    
    def dimensions(self) -> int:
        return self._dimensions


class VectorIndex(ABC):
    """
    Abstract vector index for similarity search.
    
    This is the STORAGE component of EMBEDDING storage type.
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
        Returns list of (embedding_id, similarity_score).
        """
        ...
    
    @abstractmethod
    def remove(self, embedding_id: str) -> bool:
        """Remove embedding from index"""
        ...
    
    @abstractmethod
    def get(self, embedding_id: str) -> Optional[Embedding]:
        """Get embedding by ID"""
        ...
    
    @abstractmethod
    def count(self) -> int:
        """Get number of embeddings"""
        ...
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all embeddings"""
        ...


class FlatVectorIndex(VectorIndex):
    """
    Simple brute-force vector index.
    
    O(n) search but exact results. Good for small datasets.
    """
    
    def __init__(self):
        self._embeddings: Dict[str, Embedding] = {}
    
    def add(self, embedding: Embedding) -> None:
        self._embeddings[embedding.id] = embedding
    
    def add_batch(self, embeddings: List[Embedding]) -> None:
        for e in embeddings:
            self.add(e)
    
    def search(self, query: Embedding, k: int = 10) -> List[Tuple[str, float]]:
        """Brute-force k-NN with cosine similarity"""
        if not self._embeddings:
            return []
        
        scores = []
        for emb in self._embeddings.values():
            try:
                similarity = query.cosine_similarity(emb)
                scores.append((emb.id, similarity))
            except ValueError:
                continue
        
        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def remove(self, embedding_id: str) -> bool:
        if embedding_id in self._embeddings:
            del self._embeddings[embedding_id]
            return True
        return False
    
    def get(self, embedding_id: str) -> Optional[Embedding]:
        return self._embeddings.get(embedding_id)
    
    def count(self) -> int:
        return len(self._embeddings)
    
    def clear(self) -> None:
        self._embeddings.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize index"""
        return {
            'embeddings': [e.to_dict() for e in self._embeddings.values()]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlatVectorIndex':
        """Deserialize index"""
        index = cls()
        for e_data in data.get('embeddings', []):
            index.add(Embedding.from_dict(e_data))
        return index


class EmbeddingStore:
    """
    High-level embedding storage manager.
    
    Combines provider + index for complete EMBEDDING storage type.
    """
    
    def __init__(self, 
                 provider: EmbeddingProvider = None,
                 index: VectorIndex = None):
        self.provider = provider or HashEmbeddingProvider()
        self.index = index or FlatVectorIndex()
        
        # Source ID → Embedding ID mapping
        self._source_to_embedding: Dict[str, str] = {}
    
    def embed_and_store(self, obj: Any, source_id: str = None) -> Embedding:
        """Embed an object and store it"""
        embedding = self.provider.embed_object(obj)
        if source_id:
            embedding.source_id = source_id
        
        self.index.add(embedding)
        self._source_to_embedding[embedding.source_id] = embedding.id
        return embedding
    
    def embed_text_and_store(self, text: str, source_id: str = None) -> Embedding:
        """Embed text and store it"""
        embedding = self.provider.embed_text(text)
        if source_id:
            embedding.source_id = source_id
        
        self.index.add(embedding)
        self._source_to_embedding[embedding.source_id] = embedding.id
        return embedding
    
    def find_similar(self, query: Union[str, Embedding, Any], 
                     k: int = 10) -> List[Tuple[str, float]]:
        """
        Find k most similar items.
        
        Returns list of (source_id, similarity) tuples.
        """
        # Get query embedding
        if isinstance(query, str):
            query_emb = self.provider.embed_text(query)
        elif isinstance(query, Embedding):
            query_emb = query
        else:
            query_emb = self.provider.embed_object(query)
        
        # Search
        results = self.index.search(query_emb, k)
        
        # Map back to source IDs
        source_results = []
        for emb_id, score in results:
            emb = self.index.get(emb_id)
            if emb:
                source_results.append((emb.source_id, score))
        
        return source_results
    
    def get_embedding_for(self, source_id: str) -> Optional[Embedding]:
        """Get embedding for a source object"""
        emb_id = self._source_to_embedding.get(source_id)
        if emb_id:
            return self.index.get(emb_id)
        return None
    
    def remove_for(self, source_id: str) -> bool:
        """Remove embedding for a source object"""
        emb_id = self._source_to_embedding.get(source_id)
        if emb_id:
            self.index.remove(emb_id)
            del self._source_to_embedding[source_id]
            return True
        return False


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 3: GENERIC OBSERVER PATTERN                               ║
# ║                                                                           ║
# ║  Observers work with ANY input class via Generic[T].                      ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class Signal(Generic[P]):
    """
    Generic signal carrying payload of type P.
    
    Signals are the communication unit between observables and observers.
    Works with ANY payload type.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: SignalType = SignalType.CUSTOM
    source_id: str = ""
    target_id: Optional[str] = None
    
    # Generic payload
    payload: Optional[P] = None
    
    # Additional data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    timestamp: float = field(default_factory=time.time)
    priority: Priority = Priority.NORMAL
    
    # Processing state
    handled: bool = False
    cancelled: bool = False
    responses: List[Any] = field(default_factory=list)
    error: Optional[Exception] = None
    
    def with_payload(self, payload: P) -> 'Signal[P]':
        """Create copy with new payload"""
        return Signal(
            id=self.id,
            signal_type=self.signal_type,
            source_id=self.source_id,
            target_id=self.target_id,
            payload=payload,
            data=self.data.copy(),
            metadata=self.metadata.copy(),
            timestamp=self.timestamp,
            priority=self.priority
        )
    
    def add_response(self, response: Any, observer_id: str = None) -> None:
        """Add a response from an observer"""
        self.responses.append({
            'observer_id': observer_id,
            'response': response,
            'timestamp': time.time()
        })
        self.handled = True
    
    def cancel(self) -> None:
        """Cancel further processing"""
        self.cancelled = True
    
    def mark_error(self, error: Exception) -> None:
        """Mark signal as errored"""
        self.error = error
        self.handled = True


class Observer(ABC, Generic[P]):
    """
    Abstract generic observer for ANY payload type P.
    
    Subclass and implement handle() for specific behavior.
    The observer pattern is fully generic - it works with:
    - GraphNode
    - Entity
    - ExtractedWord
    - Any custom class
    """
    
    def __init__(self,
                 observer_id: str = None,
                 priority: Priority = Priority.NORMAL):
        self.id = observer_id or str(uuid.uuid4())
        self.priority = priority
        self._active = True
        self._filter_fn: Optional[Callable[[Signal[P]], bool]] = None
        
        # Stats
        self._handled_count = 0
        self._error_count = 0
        self._last_signal_time: Optional[float] = None
    
    @property
    def active(self) -> bool:
        return self._active
    
    def activate(self) -> None:
        self._active = True
    
    def deactivate(self) -> None:
        self._active = False
    
    def set_filter(self, filter_fn: Callable[[Signal[P]], bool]) -> 'Observer[P]':
        """Set filter - only signals passing filter are handled"""
        self._filter_fn = filter_fn
        return self
    
    def for_type(self, signal_type: SignalType) -> 'Observer[P]':
        """Convenience: filter to single signal type"""
        self._filter_fn = lambda s: s.signal_type == signal_type
        return self
    
    def for_types(self, *signal_types: SignalType) -> 'Observer[P]':
        """Convenience: filter to multiple signal types"""
        types_set = set(signal_types)
        self._filter_fn = lambda s: s.signal_type in types_set
        return self
    
    def should_handle(self, signal: Signal[P]) -> bool:
        """Check if observer should handle this signal"""
        if not self._active:
            return False
        if signal.cancelled:
            return False
        if self._filter_fn and not self._filter_fn(signal):
            return False
        return True
    
    @abstractmethod
    def handle(self, signal: Signal[P]) -> Any:
        """
        Handle a signal. Override in subclass.
        
        Return value is added to signal.responses.
        """
        ...
    
    def __call__(self, signal: Signal[P]) -> Any:
        """Allow observer to be called directly"""
        if not self.should_handle(signal):
            return None
        
        try:
            result = self.handle(signal)
            self._handled_count += 1
            self._last_signal_time = time.time()
            return result
        except Exception as e:
            self._error_count += 1
            raise


class FunctionObserver(Observer[P]):
    """
    Observer that wraps any callable.
    
    Makes any function into an observer.
    """
    
    def __init__(self,
                 fn: Callable[[Signal[P]], Any],
                 observer_id: str = None,
                 priority: Priority = Priority.NORMAL):
        super().__init__(observer_id, priority)
        self._fn = fn
    
    def handle(self, signal: Signal[P]) -> Any:
        return self._fn(signal)


class MethodObserver(Observer[P]):
    """
    Observer that calls a method on objects in the signal.
    
    Useful for "every entity should respond to this signal" patterns.
    """
    
    def __init__(self,
                 method_name: str,
                 observer_id: str = None,
                 priority: Priority = Priority.NORMAL):
        super().__init__(observer_id, priority)
        self._method_name = method_name
    
    def handle(self, signal: Signal[P]) -> Any:
        target = signal.payload
        if target is None:
            return None
        
        method = getattr(target, self._method_name, None)
        if method and callable(method):
            return method(signal)
        return None


class CollectorObserver(Observer[P]):
    """
    Observer that collects payloads matching criteria.
    
    Useful for gathering all entities of a type, all words with a tag, etc.
    """
    
    def __init__(self,
                 predicate: Callable[[P], bool] = None,
                 observer_id: str = None):
        super().__init__(observer_id, Priority.LOW)
        self._predicate = predicate
        self.collected: List[P] = []
    
    def handle(self, signal: Signal[P]) -> Any:
        if signal.payload is not None:
            if self._predicate is None or self._predicate(signal.payload):
                self.collected.append(signal.payload)
        return len(self.collected)
    
    def clear(self) -> List[P]:
        """Clear and return collected items"""
        items = self.collected
        self.collected = []
        return items


class StateObserver(Observer[Any]):
    """
    Observer specialized for state machine transitions.
    """
    
    def __init__(self, observer_id: str = None):
        super().__init__(observer_id, Priority.HIGH)
        self.state_history: List[Tuple[str, float]] = []
        self.current_state: Optional[str] = None
        
        self._on_enter: Dict[str, List[Callable]] = defaultdict(list)
        self._on_exit: Dict[str, List[Callable]] = defaultdict(list)
        self._on_transition: Dict[Tuple[str, str], List[Callable]] = defaultdict(list)
        
        # Filter to state signals
        self.for_types(SignalType.STATE_ENTER, SignalType.STATE_EXIT, SignalType.STATE_CHANGE)
    
    def on_enter(self, state: str, callback: Callable) -> 'StateObserver':
        self._on_enter[state].append(callback)
        return self
    
    def on_exit(self, state: str, callback: Callable) -> 'StateObserver':
        self._on_exit[state].append(callback)
        return self
    
    def on_transition(self, from_state: str, to_state: str, callback: Callable) -> 'StateObserver':
        self._on_transition[(from_state, to_state)].append(callback)
        return self
    
    def handle(self, signal: Signal) -> Dict[str, Any]:
        old_state = signal.data.get('old_state')
        new_state = signal.data.get('new_state')
        
        # Exit hooks
        if old_state:
            for cb in self._on_exit.get(old_state, []):
                try: cb(signal)
                except: pass
        
        # Transition hooks
        for cb in self._on_transition.get((old_state, new_state), []):
            try: cb(signal)
            except: pass
        
        # Enter hooks
        if new_state:
            for cb in self._on_enter.get(new_state, []):
                try: cb(signal)
                except: pass
            
            self.state_history.append((new_state, signal.timestamp))
            self.current_state = new_state
        
        return {'transition': f"{old_state} → {new_state}"}


class EntityObserver(Observer[Any]):
    """
    Observer for entity lifecycle events.
    Works with ANY entity class (GraphNode, Entity, custom).
    """
    
    def __init__(self, observer_id: str = None):
        super().__init__(observer_id, Priority.NORMAL)
        self.entity_log: List[Dict] = []
        self._watched: Dict[str, List[Callable]] = defaultdict(list)
        
        self.for_types(
            SignalType.ENTITY_CREATED, SignalType.ENTITY_MODIFIED,
            SignalType.ENTITY_DESTROYED, SignalType.ENTITY_LINKED,
            SignalType.ENTITY_UNLINKED
        )
    
    def watch(self, entity_id: str, callback: Callable) -> 'EntityObserver':
        """Watch specific entity"""
        self._watched[entity_id].append(callback)
        return self
    
    def handle(self, signal: Signal) -> Dict[str, Any]:
        entity_id = signal.data.get('entity_id', signal.source_id)
        
        self.entity_log.append({
            'type': signal.signal_type.value,
            'entity_id': entity_id,
            'timestamp': signal.timestamp,
            'data': signal.data
        })
        
        for cb in self._watched.get(entity_id, []):
            try: cb(signal)
            except: pass
        
        return {'logged': entity_id}


class TraversalObserver(Observer[Any]):
    """
    Observer for graph traversal events.
    """
    
    def __init__(self, observer_id: str = None):
        super().__init__(observer_id, Priority.NORMAL)
        self.path: List[str] = []
        self.visited: Set[str] = set()
        self.layer_history: List[Tuple[str, str]] = []
        
        self._on_step: List[Callable] = []
        self._on_layer: List[Callable] = []
        
        self.for_types(
            SignalType.TRAVERSAL_START, SignalType.TRAVERSAL_STEP,
            SignalType.TRAVERSAL_END, SignalType.LAYER_SWITCH
        )
    
    def on_step(self, callback: Callable) -> 'TraversalObserver':
        self._on_step.append(callback)
        return self
    
    def on_layer_switch(self, callback: Callable) -> 'TraversalObserver':
        self._on_layer.append(callback)
        return self
    
    def handle(self, signal: Signal) -> Optional[Dict[str, Any]]:
        if signal.signal_type == SignalType.TRAVERSAL_STEP:
            node_id = signal.data.get('node_id')
            if node_id:
                self.path.append(node_id)
                self.visited.add(node_id)
            for cb in self._on_step:
                try: cb(signal)
                except: pass
            return {'path_length': len(self.path)}
        
        elif signal.signal_type == SignalType.LAYER_SWITCH:
            old = signal.data.get('old_layer')
            new = signal.data.get('new_layer')
            self.layer_history.append((old, new))
            for cb in self._on_layer:
                try: cb(signal)
                except: pass
            return {'switches': len(self.layer_history)}
        
        return None


class ObserverBus(Generic[P]):
    """
    Central hub for observer registration and signal dispatch.
    
    Generic over payload type P. Thread-safe.
    """
    
    def __init__(self):
        self._observers: Dict[str, Observer[P]] = {}
        self._type_subs: Dict[SignalType, Set[str]] = defaultdict(set)
        self._history: deque = deque(maxlen=1000)
        self._pending: deque = deque()
        self._processing = False
        self._lock = threading.RLock()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────────────────────
    
    def register(self, observer: Observer[P],
                 signal_types: List[SignalType] = None) -> str:
        """Register an observer, return its ID"""
        with self._lock:
            self._observers[observer.id] = observer
            
            types = signal_types or list(SignalType)
            for st in types:
                self._type_subs[st].add(observer.id)
            
            return observer.id
    
    def unregister(self, observer_id: str) -> bool:
        """Remove an observer"""
        with self._lock:
            if observer_id in self._observers:
                del self._observers[observer_id]
                for subs in self._type_subs.values():
                    subs.discard(observer_id)
                return True
            return False
    
    def on(self, signal_type: SignalType,
           callback: Callable[[Signal[P]], Any],
           priority: Priority = Priority.NORMAL) -> str:
        """Convenience: register function as observer"""
        observer = FunctionObserver(callback, priority=priority)
        return self.register(observer, [signal_type])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Dispatch
    # ─────────────────────────────────────────────────────────────────────────
    
    def emit(self, signal: Signal[P]) -> Signal[P]:
        """Emit a signal, return it after processing"""
        with self._lock:
            self._pending.append(signal)
            if not self._processing:
                self._process_pending()
        return signal
    
    def emit_type(self, signal_type: SignalType,
                  source_id: str = "",
                  payload: P = None,
                  data: Dict[str, Any] = None,
                  **kwargs) -> Signal[P]:
        """Convenience: emit by type"""
        signal = Signal(
            signal_type=signal_type,
            source_id=source_id,
            payload=payload,
            data=data or {},
            **kwargs
        )
        return self.emit(signal)
    
    def _process_pending(self):
        self._processing = True
        while self._pending:
            signal = self._pending.popleft()
            self._dispatch(signal)
            self._history.append(signal)
        self._processing = False
    
    def _dispatch(self, signal: Signal[P]):
        sub_ids = self._type_subs.get(signal.signal_type, set())
        
        observers = [self._observers[oid] for oid in sub_ids if oid in self._observers]
        observers = [o for o in observers if o.should_handle(signal)]
        observers.sort(key=lambda o: o.priority.value)
        
        for observer in observers:
            if signal.cancelled:
                break
            try:
                result = observer.handle(signal)
                if result is not None:
                    signal.add_response(result, observer.id)
            except Exception as e:
                signal.mark_error(e)
                if signal.signal_type != SignalType.ERROR:
                    self.emit_type(SignalType.ERROR, observer.id, data={'error': str(e)})
    
    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_observer(self, observer_id: str) -> Optional[Observer[P]]:
        return self._observers.get(observer_id)
    
    def get_history(self, signal_type: SignalType = None, limit: int = 100) -> List[Signal[P]]:
        signals = list(self._history)
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        return signals[-limit:]
    
    # ─────────────────────────────────────────────────────────────────────────
    # Factory Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def create_state_observer(self) -> Tuple[StateObserver, str]:
        obs = StateObserver()
        return obs, self.register(obs)
    
    def create_entity_observer(self) -> Tuple[EntityObserver, str]:
        obs = EntityObserver()
        return obs, self.register(obs)
    
    def create_traversal_observer(self) -> Tuple[TraversalObserver, str]:
        obs = TraversalObserver()
        return obs, self.register(obs)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 4: LAYER SYSTEM                                           ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class LayerConfig:
    """Configuration for a traversal layer"""
    name: str
    edge_types: Set[str] = field(default_factory=set)
    edge_kinds: Set[str] = field(default_factory=set)
    filter_fn: Optional[Callable] = None
    auto_switch_condition: Optional[Callable] = None
    priority: int = 0
    
    def matches_edge(self, edge: Any, context: Any = None) -> bool:
        """Check if edge belongs to this layer"""
        if self.filter_fn:
            return self.filter_fn(edge, context)
        
        # Check edge type
        edge_type = getattr(edge, 'edge_type', None)
        if edge_type:
            type_name = edge_type.name if hasattr(edge_type, 'name') else str(edge_type)
            if self.edge_types and type_name not in self.edge_types:
                return False
        
        # Check kind
        kind = getattr(edge, 'kind', None) or edge.data.get('kind') if hasattr(edge, 'data') else None
        if kind and self.edge_kinds and kind not in self.edge_kinds:
            return False
        
        return True


class LayerRegistry:
    """Central registry of layers"""
    
    def __init__(self):
        self._layers: Dict[str, LayerConfig] = {}
        self._active: str = "default"
    
    def register(self, config: LayerConfig) -> 'LayerRegistry':
        self._layers[config.name] = config
        return self
    
    def get(self, name: str) -> Optional[LayerConfig]:
        return self._layers.get(name)
    
    def set_active(self, name: str) -> bool:
        if name in self._layers or name == "default":
            self._active = name
            return True
        return False
    
    @property
    def active(self) -> Optional[LayerConfig]:
        return self._layers.get(self._active)
    
    @property
    def active_name(self) -> str:
        return self._active
    
    def check_auto_switch(self, context: Any) -> Optional[str]:
        """Check if any layer wants to auto-switch"""
        for config in sorted(self._layers.values(), key=lambda l: -l.priority):
            if config.auto_switch_condition:
                try:
                    if config.auto_switch_condition(context):
                        return config.name
                except:
                    pass
        return None


def create_standard_layers() -> LayerRegistry:
    """Create registry with standard layers"""
    registry = LayerRegistry()
    
    registry.register(LayerConfig(
        name="spatial",
        edge_kinds={"north_of", "south_of", "east_of", "west_of",
                   "in", "on", "under", "near", "contains", "leads_to"},
        priority=10
    ))
    
    registry.register(LayerConfig(
        name="logical",
        edge_kinds={"requires", "unlocks", "reveals", "enables", "blocks", "triggers"},
        priority=15
    ))
    
    registry.register(LayerConfig(
        name="character",
        edge_kinds={"knows", "trusts", "owns", "held_by", "wants"},
        priority=5
    ))
    
    registry.register(LayerConfig(
        name="temporal",
        edge_kinds={"before", "after", "during"},
        priority=5
    ))
    
    registry.register(LayerConfig(
        name="semantic",
        filter_fn=lambda e, c: getattr(e, 'layer', None) == 'semantic',
        priority=5
    ))
    
    return registry


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 5: TRAVERSAL CONTEXT & WRAPPER                            ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class TraversalContext:
    """
    State that travels with the traversal wrapper.
    
    Compatible with your existing WalkerContext.
    """
    # Position
    current_node: Any = None
    previous_node: Any = None
    
    # History
    path: List[Any] = field(default_factory=list)
    visited: Set[str] = field(default_factory=set)
    depth: int = 0
    step_count: int = 0
    
    # Layers
    current_layer: str = "default"
    layer_stack: List[str] = field(default_factory=list)
    
    # Game state (compatible with WalkerContext)
    flags: Dict[str, bool] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    inventory: List[str] = field(default_factory=list)
    memories: List[Dict] = field(default_factory=list)
    buffer: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)
    
    # Graph reference
    graph_ref: Any = None
    
    def get_node_id(self, node: Any) -> str:
        if hasattr(node, 'id'):
            return node.id
        return str(node)
    
    def push_layer(self, layer: str):
        self.layer_stack.append(self.current_layer)
        self.current_layer = layer
    
    def pop_layer(self) -> str:
        if self.layer_stack:
            self.current_layer = self.layer_stack.pop()
        return self.current_layer
    
    def mark_visited(self, node: Any):
        self.visited.add(self.get_node_id(node))
    
    def is_visited(self, node: Any) -> bool:
        return self.get_node_id(node) in self.visited
    
    def set_flag(self, name: str, value: bool = True):
        self.flags[name] = value
    
    def get_flag(self, name: str, default: bool = False) -> bool:
        return self.flags.get(name, default)
    
    def increment(self, counter: str, amount: int = 1) -> int:
        self.counters[counter] = self.counters.get(counter, 0) + amount
        return self.counters[counter]
    
    def get_counter(self, counter: str, default: int = 0) -> int:
        return self.counters.get(counter, default)
    
    def add_to_inventory(self, item_id: str):
        if item_id not in self.inventory:
            self.inventory.append(item_id)
    
    def has_item(self, item_id: str) -> bool:
        return item_id in self.inventory
    
    def add_memory(self, memory_type: str, content: Any, source: str = None):
        self.memories.append({
            'type': memory_type,
            'content': content,
            'source': source,
            'timestamp': time.time()
        })
    
    def log_event(self, event_type: str, details: Dict = None):
        self.events.append({
            'type': event_type,
            'details': details or {},
            'timestamp': time.time()
        })
    
    def clone(self) -> 'TraversalContext':
        ctx = TraversalContext()
        ctx.current_node = self.current_node
        ctx.previous_node = self.previous_node
        ctx.path = self.path.copy()
        ctx.visited = self.visited.copy()
        ctx.depth = self.depth
        ctx.step_count = self.step_count
        ctx.current_layer = self.current_layer
        ctx.layer_stack = self.layer_stack.copy()
        ctx.flags = self.flags.copy()
        ctx.counters = self.counters.copy()
        ctx.inventory = self.inventory.copy()
        ctx.memories = self.memories.copy()
        ctx.buffer = self.buffer.copy()
        ctx.events = self.events.copy()
        ctx.graph_ref = self.graph_ref
        return ctx
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'visited': list(self.visited),
            'depth': self.depth,
            'step_count': self.step_count,
            'current_layer': self.current_layer,
            'layer_stack': self.layer_stack,
            'flags': self.flags,
            'counters': self.counters,
            'inventory': self.inventory,
            'memories': self.memories,
            'events': self.events
        }


class TraversalWrapper(Generic[T]):
    """
    Smart iterator that IS the iterator (__iter__ returns self).
    
    Carries context through traversal. Supports layer switching,
    graph hotswapping, branching.
    """
    
    def __init__(self,
                 source: Union[Iterable[T], Callable[[], Iterable[T]]] = None,
                 context: TraversalContext = None,
                 layer_registry: LayerRegistry = None,
                 observer_bus: ObserverBus = None):
        
        self._source = source
        self._iterator: Optional[Iterator[T]] = None
        
        self.context = context or TraversalContext()
        self.layers = layer_registry or LayerRegistry()
        self._bus = observer_bus
        
        self._callbacks: List[Callable] = []
        self._current: Optional[T] = None
        self._exhausted = False
    
    def __iter__(self) -> 'TraversalWrapper[T]':
        """THE KEY: Returns self. Wrapper travels with iteration."""
        if callable(self._source):
            self._source = self._source()
        if self._source is not None:
            self._iterator = iter(self._source)
        self._exhausted = False
        return self
    
    def __next__(self) -> T:
        if self._exhausted or self._iterator is None:
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
            for cb in self._callbacks:
                try: cb(self, item)
                except: pass
            
            # Emit signal
            if self._bus:
                self._bus.emit_type(
                    SignalType.TRAVERSAL_STEP,
                    source_id=self.context.get_node_id(item),
                    data={'depth': self.context.depth}
                )
            
            # Check auto layer switch
            new_layer = self.layers.check_auto_switch(self.context)
            if new_layer and new_layer != self.context.current_layer:
                self.switch_layer(new_layer)
            
            return item
            
        except StopIteration:
            self._exhausted = True
            raise
    
    @property
    def current(self) -> Optional[T]:
        return self._current
    
    @property
    def depth(self) -> int:
        return self.context.depth
    
    @property
    def layer(self) -> str:
        return self.context.current_layer
    
    def switch_layer(self, name: str, push: bool = False):
        old = self.context.current_layer
        if push:
            self.context.push_layer(name)
        else:
            self.context.current_layer = name
        self.layers.set_active(name)
        
        if self._bus:
            self._bus.emit_type(SignalType.LAYER_SWITCH, data={'old_layer': old, 'new_layer': name})
    
    def pop_layer(self) -> str:
        old = self.context.current_layer
        new = self.context.pop_layer()
        self.layers.set_active(new)
        if self._bus:
            self._bus.emit_type(SignalType.LAYER_SWITCH, data={'old_layer': old, 'new_layer': new})
        return new
    
    def swap_graph(self, new_graph: Any, reset: bool = False):
        self.context.graph_ref = new_graph
        if reset:
            self._current = None
            self.context.path = []
            self.context.visited = set()
        if self._bus:
            self._bus.emit_type(SignalType.GRAPH_SWAP, data={'reset': reset})
    
    def on_step(self, callback: Callable) -> 'TraversalWrapper[T]':
        self._callbacks.append(callback)
        return self
    
    def branch(self) -> 'TraversalWrapper[T]':
        branched = TraversalWrapper(
            context=self.context.clone(),
            layer_registry=self.layers,
            observer_bus=self._bus
        )
        branched._current = self._current
        return branched
    
    def collect(self) -> List[T]:
        return list(self)
    
    def take(self, n: int) -> List[T]:
        result = []
        for item in self:
            result.append(item)
            if len(result) >= n:
                break
        return result


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 6: TEXT EXTRACTION                                        ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class ExtractedWord:
    """A word extracted from text"""
    text: str
    word_class: WordClass = WordClass.UNKNOWN
    position: int = 0
    sentence_idx: int = 0
    paragraph_idx: int = 0
    entity_id: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)
    source_sentence: str = ""


@dataclass
class ExtractedEntity:
    """An entity extracted from text"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: str = "item"  # location, character, item, concept
    adjectives: Set[str] = field(default_factory=set)
    descriptions: List[str] = field(default_factory=list)
    first_mention: int = 0
    mention_count: int = 1
    confidence: float = 0.5


@dataclass
class ExtractedRelation:
    """A relation between entities"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_word: str = ""  # in, on, near, etc.
    source_sentence: str = ""
    confidence: float = 0.5


@dataclass
class ExtractedFragment:
    """A prose fragment"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    category: FragmentCategory = FragmentCategory.BASE_DESCRIPTION
    entity_id: Optional[str] = None
    source_position: int = 0


@dataclass
class ExtractionResult:
    """Complete extraction result"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_text: str = ""
    words: List[ExtractedWord] = field(default_factory=list)
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    fragments: List[ExtractedFragment] = field(default_factory=list)
    
    def summary(self) -> str:
        return (f"Extracted: {len(self.words)} words, {len(self.entities)} entities, "
                f"{len(self.relations)} relations, {len(self.fragments)} fragments")


class TextExtractor:
    """
    Extracts structured data from text.
    
    Scan text → analyze every word → construct world data.
    """
    
    LOCATION_WORDS = {'room', 'house', 'building', 'cave', 'forest', 'castle', 
                      'tower', 'garden', 'kitchen', 'hall', 'corridor', 'door'}
    CHARACTER_WORDS = {'man', 'woman', 'person', 'wizard', 'witch', 'guard', 
                       'stranger', 'king', 'queen', 'knight'}
    ITEM_WORDS = {'key', 'sword', 'book', 'letter', 'ring', 'box', 'chest', 
                  'lamp', 'bottle', 'potion', 'scroll', 'map'}
    SPATIAL_PREPS = {'in', 'on', 'under', 'near', 'beside', 'behind', 'above', 
                     'below', 'inside', 'through'}
    
    def __init__(self, observer_bus: ObserverBus = None):
        self._bus = observer_bus
        self._entities: Dict[str, ExtractedEntity] = {}
        self._name_to_id: Dict[str, str] = {}
    
    def extract(self, text: str) -> ExtractionResult:
        """Main extraction method"""
        self._entities = {}
        self._name_to_id = {}
        
        if self._bus:
            self._bus.emit_type(SignalType.EXTRACTION_START)
        
        words = []
        relations = []
        fragments = []
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        position = 0
        for para_idx, para in enumerate(paragraphs):
            sentences = re.split(r'(?<=[.!?])\s+', para)
            
            for sent_idx, sentence in enumerate(sentences):
                # Extract words
                sent_words = self._extract_sentence_words(
                    sentence, position, sent_idx, para_idx
                )
                words.extend(sent_words)
                
                # Find entities and relations
                self._analyze_sentence(sent_words, sentence, relations)
                
                # Create fragment
                frag = ExtractedFragment(
                    text=sentence,
                    category=self._classify_fragment(sentence),
                    source_position=position
                )
                fragments.append(frag)
                
                position += len(sentence) + 1
                
                if self._bus:
                    self._bus.emit_type(SignalType.EXTRACTION_WORD, 
                                       data={'count': len(sent_words)})
        
        result = ExtractionResult(
            source_text=text,
            words=words,
            entities=list(self._entities.values()),
            relations=relations,
            fragments=fragments
        )
        
        if self._bus:
            self._bus.emit_type(SignalType.EXTRACTION_COMPLETE,
                               data={'entities': len(result.entities)})
        
        return result
    
    def _extract_sentence_words(self, sentence: str, position: int,
                                 sent_idx: int, para_idx: int) -> List[ExtractedWord]:
        """Extract words from a sentence"""
        words = []
        tokens = re.findall(r'\b\w+\b', sentence)
        
        for i, token in enumerate(tokens):
            word = ExtractedWord(
                text=token.lower(),
                position=position + sentence.lower().find(token.lower()),
                sentence_idx=sent_idx,
                paragraph_idx=para_idx,
                source_sentence=sentence
            )
            word.word_class = self._classify_word(token, i, tokens)
            words.append(word)
        
        return words
    
    def _classify_word(self, token: str, pos: int, context: List[str]) -> WordClass:
        """Classify a word grammatically"""
        lower = token.lower()
        
        if token[0].isupper() and pos > 0:
            return WordClass.PROPER_NOUN
        if lower in self.SPATIAL_PREPS:
            return WordClass.PREPOSITION
        if lower in {'a', 'an', 'the'}:
            return WordClass.ARTICLE
        if lower in {'he', 'she', 'it', 'they', 'him', 'her'}:
            return WordClass.PRONOUN
        if lower in self.LOCATION_WORDS | self.CHARACTER_WORDS | self.ITEM_WORDS:
            return WordClass.NOUN
        if lower.endswith(('ed', 'ing')):
            return WordClass.VERB
        if lower.endswith(('ful', 'less', 'ous')):
            return WordClass.ADJECTIVE
        
        return WordClass.UNKNOWN
    
    def _analyze_sentence(self, words: List[ExtractedWord], 
                          sentence: str, relations: List[ExtractedRelation]):
        """Find entities and relations in sentence"""
        for i, word in enumerate(words):
            if word.word_class in (WordClass.NOUN, WordClass.PROPER_NOUN):
                entity = self._get_or_create_entity(word)
                word.entity_id = entity.id
                
                # Get adjectives before
                if i > 0 and words[i-1].word_class == WordClass.ADJECTIVE:
                    entity.adjectives.add(words[i-1].text)
        
        # Find relations (entity PREP entity)
        for i, word in enumerate(words):
            if word.word_class == WordClass.PREPOSITION:
                source_id = self._find_entity_before(words, i)
                target_id = self._find_entity_after(words, i)
                if source_id and target_id:
                    relations.append(ExtractedRelation(
                        source_id=source_id,
                        target_id=target_id,
                        relation_word=word.text,
                        source_sentence=sentence
                    ))
    
    def _get_or_create_entity(self, word: ExtractedWord) -> ExtractedEntity:
        """Get existing entity or create new one"""
        name = word.text.lower()
        
        if name in self._name_to_id:
            entity = self._entities[self._name_to_id[name]]
            entity.mention_count += 1
            return entity
        
        # Determine category
        category = "item"
        if name in self.LOCATION_WORDS:
            category = "location"
        elif name in self.CHARACTER_WORDS:
            category = "character"
        
        entity = ExtractedEntity(
            name=name,
            category=category,
            first_mention=word.position
        )
        
        self._entities[entity.id] = entity
        self._name_to_id[name] = entity.id
        
        return entity
    
    def _find_entity_before(self, words: List[ExtractedWord], pos: int) -> Optional[str]:
        for i in range(pos - 1, -1, -1):
            if words[i].entity_id:
                return words[i].entity_id
        return None
    
    def _find_entity_after(self, words: List[ExtractedWord], pos: int) -> Optional[str]:
        for i in range(pos + 1, len(words)):
            if words[i].entity_id:
                return words[i].entity_id
        return None
    
    def _classify_fragment(self, sentence: str) -> FragmentCategory:
        lower = sentence.lower()
        if '"' in sentence or "'" in sentence:
            return FragmentCategory.NPC_AMBIENT
        if any(w in lower for w in ['walked', 'ran', 'took', 'opened']):
            return FragmentCategory.STATE_CHANGE
        if any(w in lower for w in ['dark', 'cold', 'warm', 'quiet']):
            return FragmentCategory.ATMOSPHERIC
        return FragmentCategory.BASE_DESCRIPTION


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 7: WHITE ROOM BUILDER                                     ║
# ║                                                                           ║
# ║  Converts extraction results into a graph (extends IslandGraph).          ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class WhiteRoomNode:
    """
    A node in the White Room graph.
    
    Compatible with GraphNode from graph_core.py but adds embedding support.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    node_type: str = "item"  # location, item, character, concept
    data: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    # Prose fragments
    fragments: List[ExtractedFragment] = field(default_factory=list)
    
    # Embedding support
    embedding: Optional[Embedding] = None
    embedding_text: str = ""
    
    # Source tracking
    source_entity_id: Optional[str] = None
    confidence: float = 0.5
    
    def to_embedding_text(self) -> str:
        """For Embeddable protocol"""
        if self.embedding_text:
            return self.embedding_text
        parts = [self.name]
        if 'description' in self.data:
            parts.append(self.data['description'])
        for frag in self.fragments[:3]:
            parts.append(frag.text)
        return " ".join(parts)


@dataclass
class WhiteRoomEdge:
    """An edge in the White Room graph"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    edge_type: str = "contains"
    layer: str = "spatial"
    weight: float = 1.0
    bidirectional: bool = False
    data: Dict[str, Any] = field(default_factory=dict)


class WhiteRoom:
    """
    The constructed game world from extracted text.
    
    Extends/compatible with IslandGraph but adds embedding support.
    """
    
    def __init__(self, name: str = "White Room"):
        self.name = name
        self.id = str(uuid.uuid4())
        
        self.nodes: Dict[str, WhiteRoomNode] = {}
        self.edges: List[WhiteRoomEdge] = []
        
        # Embedding storage
        self.embedding_store: Optional[EmbeddingStore] = None
        
        # Indexes
        self._type_index: Dict[str, Set[str]] = defaultdict(set)
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Origin
        self.origin: Optional[WhiteRoomNode] = None
    
    def set_embedding_store(self, store: EmbeddingStore):
        """Enable embedding storage"""
        self.embedding_store = store
    
    def add_node(self, node: WhiteRoomNode, embed: bool = True) -> str:
        """Add a node, optionally embed it"""
        self.nodes[node.id] = node
        self._type_index[node.node_type].add(node.id)
        for tag in node.tags:
            self._tag_index[tag].add(node.id)
        
        if embed and self.embedding_store:
            node.embedding = self.embedding_store.embed_and_store(node, node.id)
        
        return node.id
    
    def get_node(self, node_id: str) -> Optional[WhiteRoomNode]:
        return self.nodes.get(node_id)
    
    def add_edge(self, edge: WhiteRoomEdge) -> str:
        self.edges.append(edge)
        return edge.id
    
    def connect(self, source_id: str, target_id: str, 
                edge_type: str = "contains", **kwargs) -> str:
        edge = WhiteRoomEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            **kwargs
        )
        return self.add_edge(edge)
    
    def get_edges_from(self, node_id: str) -> List[WhiteRoomEdge]:
        return [e for e in self.edges if e.source_id == node_id]
    
    def get_edges_to(self, node_id: str) -> List[WhiteRoomEdge]:
        return [e for e in self.edges if e.target_id == node_id]
    
    def find_similar(self, query: Union[str, WhiteRoomNode], k: int = 5) -> List[Tuple[WhiteRoomNode, float]]:
        """Find similar nodes using embeddings"""
        if not self.embedding_store:
            return []
        
        results = self.embedding_store.find_similar(query, k)
        return [(self.nodes[sid], score) for sid, score in results if sid in self.nodes]
    
    def query_by_type(self, node_type: str) -> List[WhiteRoomNode]:
        return [self.nodes[nid] for nid in self._type_index.get(node_type, set())]
    
    def query_by_tag(self, tag: str) -> List[WhiteRoomNode]:
        return [self.nodes[nid] for nid in self._tag_index.get(tag, set())]
    
    def summary(self) -> str:
        lines = [
            f"═══ {self.name} ═══",
            f"Nodes: {len(self.nodes)}",
            f"Edges: {len(self.edges)}",
            "By type:"
        ]
        for t, ids in sorted(self._type_index.items()):
            names = [self.nodes[i].name for i in ids][:3]
            lines.append(f"  {t}: {', '.join(names)}" + ("..." if len(ids) > 3 else ""))
        return "\n".join(lines)


class WhiteRoomBuilder:
    """
    Builds a WhiteRoom from extraction results.
    
    Text → ExtractionResult → WhiteRoom
    """
    
    RELATION_TO_EDGE = {
        'in': 'contains',
        'on': 'supports',
        'under': 'below',
        'near': 'near',
        'beside': 'beside',
    }
    
    def __init__(self, embedding_store: EmbeddingStore = None):
        self._store = embedding_store
        self._id_map: Dict[str, str] = {}
    
    def build(self, extraction: ExtractionResult) -> WhiteRoom:
        """Build WhiteRoom from extraction"""
        world = WhiteRoom()
        
        if self._store:
            world.set_embedding_store(self._store)
        
        self._id_map = {}
        
        # Create origin
        origin = WhiteRoomNode(
            name="The White Room",
            node_type="location",
            embedding_text="A featureless white space where the world forms.",
            tags={"origin"}
        )
        world.add_node(origin)
        world.origin = origin
        
        # Convert entities
        for ext_ent in extraction.entities:
            if ext_ent.confidence >= 0.3:
                node = WhiteRoomNode(
                    name=ext_ent.name.title(),
                    node_type=ext_ent.category,
                    embedding_text=f"{ext_ent.name} {' '.join(ext_ent.adjectives)}",
                    source_entity_id=ext_ent.id,
                    confidence=ext_ent.confidence
                )
                world.add_node(node)
                self._id_map[ext_ent.id] = node.id
        
        # Convert relations
        for ext_rel in extraction.relations:
            source = self._id_map.get(ext_rel.source_id)
            target = self._id_map.get(ext_rel.target_id)
            if source and target:
                edge_type = self.RELATION_TO_EDGE.get(ext_rel.relation_word, ext_rel.relation_word)
                world.connect(source, target, edge_type)
        
        # Connect orphans to origin
        linked = set()
        for edge in world.edges:
            linked.add(edge.source_id)
            linked.add(edge.target_id)
        
        for node_id in world.nodes:
            if node_id != origin.id and node_id not in linked:
                world.connect(origin.id, node_id, "contains")
        
        return world


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 8: LOOKAHEAD ENGINE                                       ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class Possibility:
    """A discovered possibility"""
    possibility_type: PossibilityType
    entity_id: str
    entity_name: str
    distance: int = 0
    similarity: float = 0.0
    conditions_met: List[str] = field(default_factory=list)
    conditions_unmet: List[str] = field(default_factory=list)
    hint_text: Optional[str] = None
    path: List[str] = field(default_factory=list)
    
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
    
    def add(self, p: Possibility):
        self.all_possibilities.append(p)
        self.by_type[p.possibility_type].append(p)
    
    def get_hints(self, max_hints: int = 3) -> List[str]:
        hints = []
        for p in self.all_possibilities:
            if p.is_near_miss() and p.hint_text:
                hints.append(p.hint_text)
            if len(hints) >= max_hints:
                break
        
        # Add hidden items
        for p in self.by_type.get(PossibilityType.HIDDEN_ITEM, []):
            if p.hint_text and len(hints) < max_hints:
                hints.append(p.hint_text)
        
        return hints


class LookaheadEngine:
    """
    Pre-traverses the graph to discover possibilities.
    
    Includes semantic similarity via embeddings.
    """
    
    def __init__(self, world: WhiteRoom = None):
        self.world = world
    
    def lookahead(self, from_node: Union[str, WhiteRoomNode],
                  context: TraversalContext = None,
                  max_depth: int = 3,
                  include_semantic: bool = True,
                  semantic_k: int = 5) -> LookaheadResult:
        """Perform lookahead from a node"""
        if isinstance(from_node, str):
            start_id = from_node
            start = self.world.get_node(from_node)
        else:
            start_id = from_node.id
            start = from_node
        
        result = LookaheadResult(
            origin_id=start_id,
            origin_name=start.name if start else start_id,
            max_depth=max_depth
        )
        
        # BFS
        visited = set()
        queue = deque([(start_id, 0, [start_id])])
        
        while queue:
            current_id, depth, path = queue.popleft()
            if current_id in visited or depth > max_depth:
                continue
            visited.add(current_id)
            
            node = self.world.get_node(current_id)
            if not node:
                continue
            
            # Analyze node
            for p in self._analyze_node(node, depth, path):
                result.add(p)
            
            # Queue neighbors
            for edge in self.world.get_edges_from(current_id):
                if edge.target_id not in visited:
                    queue.append((edge.target_id, depth + 1, path + [edge.target_id]))
        
        # Semantic similarity
        if include_semantic and start and self.world.embedding_store:
            similar = self.world.find_similar(start, semantic_k + 1)
            for sim_node, score in similar:
                if sim_node.id != start_id:
                    result.add(Possibility(
                        possibility_type=PossibilityType.SIMILAR_NODE,
                        entity_id=sim_node.id,
                        entity_name=sim_node.name,
                        similarity=score,
                        hint_text=f"{sim_node.name} seems related..."
                    ))
        
        return result
    
    def _analyze_node(self, node: WhiteRoomNode, depth: int, 
                      path: List[str]) -> List[Possibility]:
        """Analyze a node for possibilities"""
        possibilities = []
        
        if node.node_type == "location":
            possibilities.append(Possibility(
                possibility_type=PossibilityType.REACHABLE_LOCATION,
                entity_id=node.id,
                entity_name=node.name,
                distance=depth,
                path=path.copy()
            ))
        
        elif node.node_type == "item":
            ptype = PossibilityType.VISIBLE_ITEM
            if 'hidden' in node.tags:
                ptype = PossibilityType.HIDDEN_ITEM
            
            possibilities.append(Possibility(
                possibility_type=ptype,
                entity_id=node.id,
                entity_name=node.name,
                distance=depth,
                path=path.copy(),
                hint_text="Something might be hidden here..." if ptype == PossibilityType.HIDDEN_ITEM else None
            ))
        
        return possibilities


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║         SECTION 9: CONVENIENCE FUNCTIONS                                  ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def extract_text(text: str, bus: ObserverBus = None) -> ExtractionResult:
    """Extract structured data from text"""
    return TextExtractor(bus).extract(text)


def build_white_room(extraction: ExtractionResult,
                     store: EmbeddingStore = None) -> WhiteRoom:
    """Build WhiteRoom from extraction"""
    return WhiteRoomBuilder(store).build(extraction)


def text_to_world(text: str,
                  store: EmbeddingStore = None,
                  bus: ObserverBus = None) -> WhiteRoom:
    """Complete pipeline: text → extraction → WhiteRoom"""
    extraction = extract_text(text, bus)
    return build_white_room(extraction, store)


def smart_iter(source: Iterable[T],
               context: TraversalContext = None,
               layers: LayerRegistry = None,
               bus: ObserverBus = None) -> TraversalWrapper[T]:
    """Create smart iterator"""
    return TraversalWrapper(source, context, layers, bus)


def lookahead_from(world: WhiteRoom,
                   node: Union[str, WhiteRoomNode],
                   max_depth: int = 3) -> LookaheadResult:
    """One-shot lookahead"""
    return LookaheadEngine(world).lookahead(node, max_depth=max_depth)


def create_embedding_store(dimensions: int = 128) -> EmbeddingStore:
    """Create embedding store with hash provider"""
    return EmbeddingStore(
        provider=HashEmbeddingProvider(dimensions),
        index=FlatVectorIndex()
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                              EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    'Priority', 'StorageType', 'SignalType', 'WordClass', 
    'FragmentCategory', 'PossibilityType',
    
    # Protocols
    'HasId', 'HasName', 'Serializable', 'Embeddable', 'Observable',
    
    # Embedding (STORAGE TYPE)
    'Embedding', 'EmbeddingProvider', 'HashEmbeddingProvider',
    'VectorIndex', 'FlatVectorIndex', 'EmbeddingStore',
    
    # Observer (GENERIC)
    'Signal', 'Observer', 'FunctionObserver', 'MethodObserver',
    'CollectorObserver', 'StateObserver', 'EntityObserver',
    'TraversalObserver', 'ObserverBus',
    
    # Layers
    'LayerConfig', 'LayerRegistry', 'create_standard_layers',
    
    # Traversal
    'TraversalContext', 'TraversalWrapper',
    
    # Extraction
    'ExtractedWord', 'ExtractedEntity', 'ExtractedRelation',
    'ExtractedFragment', 'ExtractionResult', 'TextExtractor',
    
    # White Room
    'WhiteRoomNode', 'WhiteRoomEdge', 'WhiteRoom', 'WhiteRoomBuilder',
    
    # Lookahead
    'Possibility', 'LookaheadResult', 'LookaheadEngine',
    
    # Convenience
    'extract_text', 'build_white_room', 'text_to_world',
    'smart_iter', 'lookahead_from', 'create_embedding_store',
]

__version__ = "0.2.0"

if __name__ == "__main__":
    # Quick demo
    text = "The old key lay hidden in the desk in the study."
    
    store = create_embedding_store()
    world = text_to_world(text, store=store)
    
    print(world.summary())
    
    if world.origin:
        result = lookahead_from(world, world.origin)
        for hint in result.get_hints():
            print(f"💡 {hint}")
