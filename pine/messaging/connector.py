"""
Pine Connector
==============

Type-safe signal routing and wiring.

SOURCE: islands/connector_core.py (8KB)
Copy the full implementation from that file.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from pine.core.signals import SignalPayload, SignalType


class RouteTarget(Enum):
    """Where to route signals."""
    OBSERVER = auto()       # Direct to observer
    EVENT_LOOP = auto()     # Queue for event loop
    BROADCAST = auto()      # All listeners
    SPECIFIC = auto()       # Named target


@dataclass
class Listener:
    """A registered signal listener."""
    id: str
    callback: Callable[[SignalPayload], None]
    signal_types: List[SignalType] = field(default_factory=list)
    priority: int = 0


class Connector:
    """
    Type-safe connector for signal routing.

    Wires components together and routes signals
    to appropriate handlers.

    TODO: Copy full implementation from islands/connector_core.py
    """

    def __init__(self):
        self._listeners: Dict[str, Listener] = {}
        self._routes: Dict[SignalType, RouteTarget] = {}

    def register(
        self,
        listener_id: str,
        callback: Callable[[SignalPayload], None],
        signal_types: List[SignalType] = None
    ) -> None:
        """Register a listener."""
        self._listeners[listener_id] = Listener(
            id=listener_id,
            callback=callback,
            signal_types=signal_types or []
        )

    def unregister(self, listener_id: str) -> None:
        """Unregister a listener."""
        self._listeners.pop(listener_id, None)

    def set_route(self, signal_type: SignalType, target: RouteTarget) -> None:
        """Set routing for a signal type."""
        self._routes[signal_type] = target

    def route(self, payload: SignalPayload) -> int:
        """Route a signal to listeners. Returns count delivered."""
        count = 0
        target = self._routes.get(payload.signal_type, RouteTarget.BROADCAST)

        for listener in self._listeners.values():
            # Check if listener accepts this type
            if listener.signal_types:
                if payload.signal_type not in listener.signal_types:
                    continue

            listener.callback(payload)
            count += 1

        return count
