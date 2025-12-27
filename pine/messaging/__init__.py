"""
Pine Messaging - Communication Layer
=====================================

Inter-component communication and I/O.

Features:
- Type-safe signal routing
- Buffered I/O interface
- Component integration

This layer depends on: pine.core

Source Files (to integrate):
- islands/connector_core.py -> connector.py
- islands/messaging_interface.py -> interface.py
- islands/integration_layer.py -> integration.py
"""

from .connector import (
    Connector, RouteTarget, Listener,
)

from .interface import (
    MessageBuffer, BufferedIO, SignalFactory,
)

from .integration import (
    IntegrationLayer, ComponentBridge,
)

__all__ = [
    'Connector', 'RouteTarget', 'Listener',
    'MessageBuffer', 'BufferedIO', 'SignalFactory',
    'IntegrationLayer', 'ComponentBridge',
]
