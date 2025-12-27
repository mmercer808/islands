"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  MATTS PRIMITIVES - Enums, Protocols, Base Classes                            ║
║  Layer 0: No internal dependencies                                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Protocol, runtime_checkable, TypeVar
from enum import Enum, auto, IntEnum
import uuid
import time

# ═══════════════════════════════════════════════════════════════════════════════
#                              TYPE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

T = TypeVar('T')
P = TypeVar('P')  # Payload
N = TypeVar('N')  # Node
E = TypeVar('E')  # Entity/Edge


# ═══════════════════════════════════════════════════════════════════════════════
#                              ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class Priority(IntEnum):
    """Universal priority levels (lower value = higher priority)"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class StorageType(Enum):
    """Storage backend types - EMBEDDING is first-class"""
    MEMORY = "memory"
    FILE = "file"
    SQLITE = "sqlite"
    EMBEDDING = "embedding"  # Vector store for semantic ops
    DISTRIBUTED = "distributed"


class SignalType(Enum):
    """Signal types for observer system"""
    # State
    STATE_ENTER = "state_enter"
    STATE_EXIT = "state_exit"
    STATE_CHANGE = "state_change"
    
    # Entity
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
    EXTRACTION_COMPLETE = "extraction_complete"
    
    # Embedding
    EMBEDDING_CREATED = "embedding_created"
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
    """Prose fragment priority (lower = composed first)"""
    BASE_DESCRIPTION = 10
    ATMOSPHERIC = 20
    STATE_CHANGE = 30
    ITEM_PRESENCE = 40
    NPC_AMBIENT = 50
    SENSORY = 60
    HISTORY = 70
    DISCOVERY = 80


class PossibilityType(Enum):
    """Lookahead possibility types"""
    REACHABLE_LOCATION = auto()
    BLOCKED_PATH = auto()
    LOCKED_DOOR = auto()
    VISIBLE_ITEM = auto()
    HIDDEN_ITEM = auto()
    TAKEABLE_ITEM = auto()
    REQUIRED_ITEM = auto()
    UNLOCKABLE = auto()
    REVEALABLE = auto()
    NEAR_MISS = auto()
    SIMILAR_NODE = auto()


# ═══════════════════════════════════════════════════════════════════════════════
#                              PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class HasId(Protocol):
    """Any object with an id"""
    @property
    def id(self) -> str: ...


@runtime_checkable  
class HasName(Protocol):
    """Any object with a name"""
    @property
    def name(self) -> str: ...


@runtime_checkable
class Serializable(Protocol):
    """Objects that serialize to/from dict"""
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable': ...


@runtime_checkable
class Embeddable(Protocol):
    """Objects that can become vector embeddings"""
    def to_embedding_text(self) -> str: ...


# ═══════════════════════════════════════════════════════════════════════════════
#                              BASE CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Identified:
    """Base for all identified objects"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self) -> float:
        """Update and return timestamp"""
        self.metadata['modified_at'] = time.time()
        return self.metadata['modified_at']


class Context(ABC):
    """Abstract context - state that travels through operations"""
    
    @abstractmethod
    def clone(self) -> 'Context':
        """Deep copy for branching"""
        ...
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize"""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
#                              EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Type vars
    'T', 'P', 'N', 'E',
    # Enums
    'Priority', 'StorageType', 'SignalType', 'WordClass', 
    'FragmentCategory', 'PossibilityType',
    # Protocols
    'HasId', 'HasName', 'Serializable', 'Embeddable',
    # Base
    'Identified', 'Context',
]
