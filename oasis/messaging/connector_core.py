#!/usr/bin/env python3
"""
Connector & Listener Core

Succinct, type-safe wiring system for signals, handlers, entities, and callbacks.
The connector can instantiate classes from type hints if needed.
"""

from __future__ import annotations
import inspect
import uuid
from typing import (
    TypeVar, Generic, Callable, Any, Union, Type, Protocol,
    get_type_hints, get_origin, get_args, Tuple, Dict, Optional
)
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from datetime import datetime

T = TypeVar('T')
H = TypeVar('H')  # Handler type
E = TypeVar('E')  # Entity type


# =============================================================================
# ROUTE TARGET - Where does the signal go?
# =============================================================================

class RouteTarget(Enum):
    OBSERVER = auto()      # Direct to observer
    EVENT_LOOP = auto()    # Queue for event loop processing
    BROADCAST = auto()     # Both


# =============================================================================
# LISTENER - Routes signals to observers or event loop
# =============================================================================

@dataclass
class Listener:
    """
    Intermediate layer that sorts signals.
    Routes to observer (immediate) or event loop (queued) based on rules.
    """
    listener_id: str = field(default_factory=lambda: f"listener_{uuid.uuid4().hex[:8]}")
    
    # Routing rules: signal_type -> target
    routes: Dict[str, RouteTarget] = field(default_factory=dict)
    default_route: RouteTarget = RouteTarget.OBSERVER
    
    # Queues
    event_queue: deque = field(default_factory=deque)
    observer_queue: deque = field(default_factory=deque)
    
    def route(self, signal: Any) -> RouteTarget:
        """Determine where signal should go."""
        signal_type = getattr(signal, 'signal_type', None)
        if signal_type:
            key = signal_type.value if hasattr(signal_type, 'value') else str(signal_type)
            return self.routes.get(key, self.default_route)
        return self.default_route
    
    def receive(self, signal: Any) -> RouteTarget:
        """Receive signal and route it."""
        target = self.route(signal)
        
        if target == RouteTarget.EVENT_LOOP:
            self.event_queue.append(signal)
        elif target == RouteTarget.OBSERVER:
            self.observer_queue.append(signal)
        else:  # BROADCAST
            self.event_queue.append(signal)
            self.observer_queue.append(signal)
        
        return target
    
    def add_route(self, signal_type: str, target: RouteTarget):
        """Add routing rule."""
        self.routes[signal_type] = target


# =============================================================================
# CONNECTOR - Type-safe wiring utility
# =============================================================================

@dataclass
class Connection(Generic[H, E]):
    """A wired connection between signal, handlers, entity, and callback."""
    connection_id: str
    signal_type: str
    handlers: Tuple[H, ...]
    entity: Optional[E]
    callback: Optional[Callable]
    created_at: datetime = field(default_factory=datetime.now)


class Connector:
    """
    Type-safe connector: connector(signal, *handlers, entity, callback=...)
    
    The *handlers can be:
      - Instances (used directly)
      - Classes (instantiated with entity as first arg if needed)
      - Callables (wrapped)
    """
    
    def __init__(self):
        self.connections: Dict[str, Connection] = {}
        self._type_cache: Dict[Type, Dict[str, Any]] = {}
    
    def __call__(
        self,
        signal: Any,
        *handlers: Union[Type[H], H, Callable],
        entity: E = None,
        callback: Callable = None
    ) -> Connection[H, E]:
        """
        Wire a signal to handlers with an entity context.
        
        Handlers in *args can be:
          - Class types -> instantiated (entity passed if constructor accepts it)
          - Instances -> used directly
          - Callables -> wrapped
        """
        signal_type = self._extract_signal_type(signal)
        resolved_handlers = tuple(
            self._resolve_handler(h, entity) for h in handlers
        )
        
        connection = Connection(
            connection_id=f"conn_{uuid.uuid4().hex[:8]}",
            signal_type=signal_type,
            handlers=resolved_handlers,
            entity=entity,
            callback=callback
        )
        
        self.connections[connection.connection_id] = connection
        return connection
    
    def _extract_signal_type(self, signal: Any) -> str:
        """Extract signal type string from signal."""
        if isinstance(signal, str):
            return signal
        if hasattr(signal, 'signal_type'):
            st = signal.signal_type
            return st.value if hasattr(st, 'value') else str(st)
        if hasattr(signal, 'value'):
            return signal.value
        return str(type(signal).__name__)
    
    def _resolve_handler(
        self, 
        handler: Union[Type[H], H, Callable], 
        entity: Any
    ) -> H:
        """
        Resolve handler to an instance.
        If it's a class, instantiate it (passing entity if signature accepts it).
        """
        if isinstance(handler, type):
            # It's a class, need to instantiate
            return self._instantiate_class(handler, entity)
        elif callable(handler) and not hasattr(handler, '__self__'):
            # It's a function, wrap it
            return handler
        else:
            # It's already an instance
            return handler
    
    def _instantiate_class(self, cls: Type[H], entity: Any) -> H:
        """Instantiate class, injecting entity if the constructor wants it."""
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.values())[1:]  # Skip 'self'
        
        if not params:
            return cls()
        
        # Check if first param could accept the entity
        first_param = params[0]
        
        # Try to match by type hint
        hints = get_type_hints(cls.__init__) if hasattr(cls.__init__, '__annotations__') else {}
        first_hint = hints.get(first_param.name)
        
        if entity is not None:
            if first_hint is None or isinstance(entity, first_hint) if first_hint else True:
                try:
                    return cls(entity)
                except TypeError:
                    pass
        
        # Try no-arg construction
        try:
            return cls()
        except TypeError as e:
            raise TypeError(f"Cannot instantiate {cls.__name__}: {e}")


# =============================================================================
# QUICK FACTORY - connector factory for different domains
# =============================================================================

def make_connector() -> Connector:
    """Factory for creating connectors."""
    return Connector()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from dataclasses import dataclass
    
    # Example entity
    @dataclass
    class GameEntity:
        entity_id: str
        name: str
        health: int = 100
    
    # Example handler class that accepts entity
    class DamageHandler:
        def __init__(self, entity: GameEntity = None):
            self.entity = entity
        
        def handle(self, amount: int):
            if self.entity:
                self.entity.health -= amount
                print(f"{self.entity.name} took {amount} damage, health: {self.entity.health}")
    
    # Example handler that doesn't need entity
    class LogHandler:
        def handle(self, msg: str):
            print(f"[LOG] {msg}")
    
    # Create connector
    connector = make_connector()
    
    # Wire it up
    player = GameEntity("p1", "Hero")
    
    # Pass classes - they get instantiated with entity
    conn = connector(
        "damage_signal",
        DamageHandler,  # Will be instantiated with player
        LogHandler,     # Will be instantiated without args
        entity=player,
        callback=lambda: print("Signal processed!")
    )
    
    print(f"Created connection: {conn.connection_id}")
    print(f"Handlers: {conn.handlers}")
    print(f"Entity: {conn.entity}")
    
    # Use the handlers
    for handler in conn.handlers:
        if isinstance(handler, DamageHandler):
            handler.handle(25)
        elif isinstance(handler, LogHandler):
            handler.handle("Combat started")
    
    if conn.callback:
        conn.callback()
