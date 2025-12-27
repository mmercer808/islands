"""
Pine Traversal System
=====================

Layer-aware traversal for multi-dimensional navigation.

Features:
- Layer registry for different traversal modes
- TraversalContext carrying state
- Smart iterator with callbacks
- Traversal wrapper for graph walking

SOURCE: Mechanics/everything/traversal.py (25KB)
Copy the full implementation from that file.

Quick Reference:
----------------
    from pine.narrative import TraversalContext, smart_iter, LayerRegistry

    # Create traversal context
    ctx = TraversalContext(
        current_node=world.origin,
        layer="physical"
    )

    # Iterate with smart iterator
    for node in smart_iter(world.nodes, ctx):
        print(f"Visiting: {node.name}")
        if node.name == "secret room":
            ctx.switch_layer("hidden")
"""

from typing import Any, Dict, List, Optional, Callable, Iterator, TypeVar, Generic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid

T = TypeVar('T')


# =============================================================================
#                              LAYER SYSTEM
# =============================================================================

@dataclass
class LayerConfig:
    """
    Configuration for a traversal layer.

    Layers represent different "dimensions" of navigation:
    - Physical: normal spatial movement
    - Hidden: secret passages, hidden rooms
    - Temporal: past/present/future views
    - Conceptual: abstract connections
    """
    name: str
    description: str = ""
    can_see: List[str] = field(default_factory=list)  # Other layers visible from this one
    entry_condition: Optional[Callable[['TraversalContext'], bool]] = None
    properties: Dict[str, Any] = field(default_factory=dict)


class LayerRegistry:
    """
    Registry of traversal layers.

    Manages layer configurations and transitions between them.
    """

    def __init__(self):
        self._layers: Dict[str, LayerConfig] = {}
        self._register_standard_layers()

    def _register_standard_layers(self) -> None:
        """Register standard layers."""
        self.register(LayerConfig(
            name="physical",
            description="Normal physical world",
            can_see=["physical"]
        ))
        self.register(LayerConfig(
            name="hidden",
            description="Hidden/secret areas",
            can_see=["physical", "hidden"]
        ))
        self.register(LayerConfig(
            name="temporal",
            description="Time-shifted view",
            can_see=["physical", "temporal"]
        ))

    def register(self, layer: LayerConfig) -> None:
        """Register a layer configuration."""
        self._layers[layer.name] = layer

    def get(self, name: str) -> Optional[LayerConfig]:
        """Get layer by name."""
        return self._layers.get(name)

    def can_transition(
        self,
        from_layer: str,
        to_layer: str,
        context: 'TraversalContext'
    ) -> bool:
        """Check if transition between layers is allowed."""
        target = self._layers.get(to_layer)
        if target is None:
            return False

        if target.entry_condition:
            return target.entry_condition(context)

        return True

    def list_layers(self) -> List[str]:
        """List all registered layer names."""
        return list(self._layers.keys())


# Global standard layers instance
standard_layers = LayerRegistry()


# =============================================================================
#                              TRAVERSAL CONTEXT
# =============================================================================

@dataclass
class TraversalContext:
    """
    Context carried through traversal.

    Tracks:
    - Current position
    - Current layer
    - Traversal history
    - Custom state
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_node: Any = None
    layer: str = "physical"
    history: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    layer_registry: LayerRegistry = field(default_factory=LayerRegistry)

    def move_to(self, node: Any) -> None:
        """Move to a new node, updating history."""
        if self.current_node:
            node_id = getattr(self.current_node, 'id', str(self.current_node))
            self.history.append(node_id)
        self.current_node = node

    def switch_layer(self, layer: str) -> bool:
        """Switch to a different layer if allowed."""
        if self.layer_registry.can_transition(self.layer, layer, self):
            self.layer = layer
            return True
        return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context data."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in context data."""
        self.data[key] = value

    def go_back(self) -> bool:
        """Go back to previous node if possible."""
        if self.history:
            # This would need node lookup
            self.history.pop()
            return True
        return False

    @property
    def depth(self) -> int:
        """Get traversal depth (history length)."""
        return len(self.history)


# =============================================================================
#                              TRAVERSAL WRAPPER
# =============================================================================

class TraversalWrapper(Generic[T]):
    """
    Wrapper that enables smart iteration with callbacks.

    Wraps an iterable and provides:
    - Pre/post item callbacks
    - Context tracking
    - Conditional skipping
    - Layer-aware filtering

    TODO: Copy full implementation from Mechanics/everything/traversal.py
    """

    def __init__(
        self,
        iterable: Iterator[T],
        context: TraversalContext = None
    ):
        self._iterable = iterable
        self._context = context or TraversalContext()
        self._pre_callbacks: List[Callable[[T, TraversalContext], bool]] = []
        self._post_callbacks: List[Callable[[T, TraversalContext], None]] = []

    def add_pre_callback(
        self,
        callback: Callable[[T, TraversalContext], bool]
    ) -> 'TraversalWrapper[T]':
        """
        Add callback executed before yielding each item.
        Return False from callback to skip item.
        """
        self._pre_callbacks.append(callback)
        return self

    def add_post_callback(
        self,
        callback: Callable[[T, TraversalContext], None]
    ) -> 'TraversalWrapper[T]':
        """Add callback executed after yielding each item."""
        self._post_callbacks.append(callback)
        return self

    def __iter__(self) -> Iterator[T]:
        for item in self._iterable:
            # Run pre-callbacks
            skip = False
            for callback in self._pre_callbacks:
                if not callback(item, self._context):
                    skip = True
                    break

            if skip:
                continue

            # Update context
            self._context.move_to(item)

            # Yield item
            yield item

            # Run post-callbacks
            for callback in self._post_callbacks:
                callback(item, self._context)

    @property
    def context(self) -> TraversalContext:
        """Get the traversal context."""
        return self._context


# =============================================================================
#                              SMART ITERATOR
# =============================================================================

def smart_iter(
    iterable: Iterator[T],
    context: TraversalContext = None,
    pre_callback: Callable[[T, TraversalContext], bool] = None,
    post_callback: Callable[[T, TraversalContext], None] = None
) -> Iterator[T]:
    """
    Create a smart iterator with context tracking.

    Convenience function that wraps TraversalWrapper.

    Usage:
        for node in smart_iter(nodes, ctx, pre_callback=filter_visible):
            process(node)
    """
    wrapper = TraversalWrapper(iterable, context)

    if pre_callback:
        wrapper.add_pre_callback(pre_callback)
    if post_callback:
        wrapper.add_post_callback(post_callback)

    return iter(wrapper)
