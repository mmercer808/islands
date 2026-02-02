"""
Pine Signal System
==================

Production-grade signal system for event-driven communication.

Features:
- Priority-based concurrent observer processing
- Circuit breaker pattern for resilience
- Worker pools per observer
- Thread-safe deadlock prevention
- Signal payload with metadata

SOURCE: matts/signal_system.py (31KB)
Copy the full implementation from that file.

Quick Reference:
----------------
    from . import SignalLine, SignalPayload, Observer

    # Create signal line
    signal_line = await create_signal_line("narrative_events")

    # Add observer
    @signal_handler(SignalType.STORY_EVENT)
    async def on_story_event(payload: SignalPayload):
        print(f"Story event: {payload.data}")

    signal_line.add_observer(on_story_event)

    # Emit signal
    await signal_line.emit(SignalPayload(
        signal_type=SignalType.STORY_EVENT,
        source="narrator",
        data={"event": "player_entered_room"}
    ))
"""

from typing import Any, Dict, List, Optional, Callable, Awaitable, Set, TypeVar
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import asyncio
import uuid
from datetime import datetime

from .primitives import Priority, SignalType, T, P

# =============================================================================
#                              ENUMS
# =============================================================================

class ObserverPriority(Enum):
    """Priority levels for observer execution order."""
    FIRST = 0       # Execute first (logging, metrics)
    HIGH = 1        # High priority handlers
    NORMAL = 2      # Default priority
    LOW = 3         # Low priority handlers
    LAST = 4        # Execute last (cleanup)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, rejecting calls
    HALF_OPEN = auto()   # Testing if recovered


# =============================================================================
#                              DATA CLASSES
# =============================================================================

@dataclass
class SignalPayload:
    """
    Payload carried by signals through the system.

    Attributes:
        signal_type: The type of signal
        source: Origin identifier (context_id, component name, etc.)
        data: Arbitrary payload data
        metadata: Additional metadata (timestamps, routing info)
        signal_id: Unique identifier for this signal
        timestamp: When the signal was created
    """
    signal_type: SignalType
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    def with_metadata(self, **kwargs) -> 'SignalPayload':
        """Create copy with additional metadata."""
        return SignalPayload(
            signal_type=self.signal_type,
            source=self.source,
            data=self.data,
            metadata={**self.metadata, **kwargs},
            signal_id=self.signal_id,
            timestamp=self.timestamp
        )


@dataclass
class Signal:
    """
    Lightweight signal for the observer bus pattern.
    Used in Mechanics/everything/signals.py style.
    """
    name: str
    payload: Any = None
    priority: Priority = Priority.NORMAL


@dataclass
class ObserverStats:
    """Statistics for an observer."""
    calls: int = 0
    failures: int = 0
    total_time_ms: float = 0.0
    last_call: Optional[datetime] = None


# =============================================================================
#                              OBSERVER PATTERN
# =============================================================================

class Observer(ABC):
    """
    Abstract base observer for signals.

    Implement this to receive signals from a SignalLine.
    """

    def __init__(self, priority: ObserverPriority = ObserverPriority.NORMAL):
        self.priority = priority
        self.stats = ObserverStats()

    @abstractmethod
    async def on_signal(self, payload: SignalPayload) -> None:
        """Handle incoming signal."""
        pass

    def accepts(self, signal_type: SignalType) -> bool:
        """Return True if this observer handles the given signal type."""
        return True  # Override to filter


class CallbackObserver(Observer):
    """Observer that wraps an async callback function."""

    def __init__(
        self,
        callback: Callable[[SignalPayload], Awaitable[None]],
        signal_types: Optional[Set[SignalType]] = None,
        priority: ObserverPriority = ObserverPriority.NORMAL
    ):
        super().__init__(priority)
        self.callback = callback
        self.signal_types = signal_types

    async def on_signal(self, payload: SignalPayload) -> None:
        await self.callback(payload)

    def accepts(self, signal_type: SignalType) -> bool:
        if self.signal_types is None:
            return True
        return signal_type in self.signal_types


# =============================================================================
#                              SIGNAL LINE
# =============================================================================

class SignalLine:
    """
    High-performance signal line with observer management.

    Features:
    - Priority-ordered observer execution
    - Concurrent observer processing
    - Circuit breaker for failing observers
    - Statistics tracking

    TODO: Copy full implementation from matts/signal_system.py
    """

    def __init__(self, name: str):
        self.name = name
        self.observers: List[Observer] = []
        self._lock = asyncio.Lock()

    def add_observer(self, observer: Observer) -> None:
        """Add an observer to the signal line."""
        self.observers.append(observer)
        self.observers.sort(key=lambda o: o.priority.value)

    def remove_observer(self, observer: Observer) -> None:
        """Remove an observer from the signal line."""
        if observer in self.observers:
            self.observers.remove(observer)

    async def emit(self, payload: SignalPayload) -> int:
        """
        Emit a signal to all observers.

        Returns the number of observers that received the signal.
        """
        count = 0
        for observer in self.observers:
            if observer.accepts(payload.signal_type):
                try:
                    await observer.on_signal(payload)
                    observer.stats.calls += 1
                    count += 1
                except Exception as e:
                    observer.stats.failures += 1
                    # TODO: Circuit breaker logic
        return count

    async def emit_concurrent(self, payload: SignalPayload) -> List[Exception]:
        """Emit to all observers concurrently, collecting errors."""
        errors = []
        tasks = []

        for observer in self.observers:
            if observer.accepts(payload.signal_type):
                tasks.append(observer.on_signal(payload))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                errors.append(r)

        return errors


# =============================================================================
#                              OBSERVER BUS
# =============================================================================

class ObserverBus:
    """
    Generic observer bus from Mechanics/everything/signals.py.

    Simpler than SignalLine - for in-process event routing.
    """

    def __init__(self):
        self._observers: Dict[str, List[Callable]] = {}

    def subscribe(self, signal_name: str, callback: Callable) -> None:
        """Subscribe to a named signal."""
        if signal_name not in self._observers:
            self._observers[signal_name] = []
        self._observers[signal_name].append(callback)

    def unsubscribe(self, signal_name: str, callback: Callable) -> None:
        """Unsubscribe from a named signal."""
        if signal_name in self._observers:
            self._observers[signal_name].remove(callback)

    def emit(self, signal: Signal) -> None:
        """Emit a signal to subscribers."""
        if signal.name in self._observers:
            for callback in self._observers[signal.name]:
                callback(signal.payload)

    async def emit_async(self, signal: Signal) -> None:
        """Emit signal to async subscribers."""
        if signal.name in self._observers:
            for callback in self._observers[signal.name]:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal.payload)
                else:
                    callback(signal.payload)


# =============================================================================
#                              FACTORY FUNCTIONS
# =============================================================================

async def create_signal_line(name: str) -> SignalLine:
    """Create and initialize a signal line."""
    return SignalLine(name)


def signal_handler(
    *signal_types: SignalType,
    priority: ObserverPriority = ObserverPriority.NORMAL
) -> Callable:
    """
    Decorator to create a signal handler.

    Usage:
        @signal_handler(SignalType.STORY_EVENT, SignalType.WORLD_CHANGED)
        async def handle_narrative(payload: SignalPayload):
            print(payload.data)
    """
    def decorator(func: Callable[[SignalPayload], Awaitable[None]]) -> CallbackObserver:
        return CallbackObserver(
            callback=func,
            signal_types=set(signal_types) if signal_types else None,
            priority=priority
        )
    return decorator
