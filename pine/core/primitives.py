"""
Pine Core Primitives
====================

Base types, protocols, enums, and type variables used throughout Pine.

Merged from:
- Mechanics/everything/primitives.py (MATTS primitives)
- matts/context_system.py (ContextState enum)

This module has ZERO dependencies - it's the foundation everything builds on.
"""

from typing import TypeVar, Protocol, Any, Dict, Optional, runtime_checkable
from enum import Enum, auto
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid

# =============================================================================
#                              TYPE VARIABLES
# =============================================================================

T = TypeVar('T')  # Generic type
P = TypeVar('P')  # Payload type for signals
N = TypeVar('N')  # Node type for graphs
E = TypeVar('E')  # Edge type for graphs


# =============================================================================
#                              ENUMS
# =============================================================================

class Priority(Enum):
    """Priority levels for signals and observers."""
    LOWEST = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    HIGHEST = 4
    CRITICAL = 5


class StorageType(Enum):
    """Storage backend types for embeddings and persistence."""
    MEMORY = auto()      # In-memory storage
    SQLITE = auto()      # SQLite database
    REDIS = auto()       # Redis cache
    EMBEDDING = auto()   # Vector embedding store


class SignalType(Enum):
    """Types of signals in the system."""
    # Core signals
    CONTEXT_CREATED = auto()
    CONTEXT_UPDATED = auto()
    CONTEXT_DESTROYED = auto()

    # Narrative signals
    STORY_EVENT = auto()
    WORLD_CHANGED = auto()
    ENTITY_SPAWNED = auto()

    # Runtime signals
    CODE_INJECTED = auto()
    HOTSWAP_COMPLETE = auto()

    # User signals
    CUSTOM = auto()


class WordClass(Enum):
    """Word classifications for text extraction."""
    NOUN = auto()
    VERB = auto()
    ADJECTIVE = auto()
    ADVERB = auto()
    PREPOSITION = auto()
    CONJUNCTION = auto()
    PRONOUN = auto()
    ARTICLE = auto()
    UNKNOWN = auto()


class FragmentCategory(Enum):
    """Categories for prose fragments."""
    BASE_DESCRIPTION = auto()      # Core entity description
    ATMOSPHERIC = auto()           # Mood/atmosphere text
    STATE_CHANGE = auto()          # Text describing state changes
    INTERACTION = auto()           # Text for interactions
    DISCOVERY = auto()             # Text for discoveries
    HIDDEN = auto()                # Hidden/secret text


class PossibilityType(Enum):
    """Types of possibilities in lookahead analysis."""
    ACTION = auto()       # Something player can do
    DISCOVERY = auto()    # Something player can find
    TRANSITION = auto()   # Movement to another location
    PUZZLE = auto()       # A puzzle to solve
    DIALOGUE = auto()     # Conversation opportunity


class ContextState(Enum):
    """States of a serializable execution context."""
    CREATED = auto()
    ACTIVE = auto()
    SUSPENDED = auto()
    SERIALIZED = auto()
    DESTROYED = auto()


# =============================================================================
#                              PROTOCOLS
# =============================================================================

@runtime_checkable
class HasId(Protocol):
    """Protocol for objects with an ID."""
    @property
    def id(self) -> str: ...


@runtime_checkable
class HasName(Protocol):
    """Protocol for objects with a name."""
    @property
    def name(self) -> str: ...


@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable objects."""
    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable': ...


@runtime_checkable
class Embeddable(Protocol):
    """Protocol for objects that can be embedded as vectors."""
    def to_embedding_text(self) -> str: ...


# =============================================================================
#                              BASE CLASSES
# =============================================================================

@dataclass
class Identified:
    """Base class for objects with unique IDs."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Identified):
            return self.id == other.id
        return False


@dataclass
class Context:
    """
    Base context class carrying state through traversals.

    This is the minimal context - extended by TraversalContext,
    SerializableExecutionContext, etc.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    parent: Optional['Context'] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context data."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in context data."""
        self.data[key] = value

    def derive(self, **kwargs) -> 'Context':
        """Create child context with additional data."""
        return Context(
            data={**self.data, **kwargs},
            parent=self
        )


# =============================================================================
#                              UTILITY CLASSES
# =============================================================================

@dataclass
class Result(ABC):
    """Base class for operation results."""
    success: bool
    message: str = ""
    data: Optional[Any] = None


@dataclass
class Success(Result):
    """Successful operation result."""
    success: bool = True


@dataclass
class Failure(Result):
    """Failed operation result."""
    success: bool = False
    error: Optional[Exception] = None
