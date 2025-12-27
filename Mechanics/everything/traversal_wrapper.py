"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                  T R A V E R S A L   W R A P P E R                            ║
║                                                                               ║
║           Smart Iterator That Carries Context Through Graphs                  ║
║                                                                               ║
║  The wrapper IS the cursor. It carries state. It can jump layers.             ║
║  Any iterable can be wrapped. Any traversal can be intercepted.               ║
║  The graph becomes fluid—swap it mid-walk.                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Dict, List, Set, Optional, Any, Callable, Iterator, Iterable,
    Tuple, Union, Generic, TypeVar, Generator, Type
)
from enum import Enum, auto
from collections import deque
import weakref
import copy
import time


# Type variables for generic wrapper
T = TypeVar('T')
N = TypeVar('N')  # Node type


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER SYSTEM - Edge set definitions
# ═══════════════════════════════════════════════════════════════════════════════

class Layer(Enum):
    """Named edge sets for traversal filtering"""
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CHARACTER = "character"
    THEMATIC = "thematic"
    NARRATIVE = "narrative"
    CAUSAL = "causal"
    CUSTOM = "custom"


@dataclass
class LayerConfig:
    """Configuration for a layer's traversal behavior"""
    name: str
    edge_types: Set[str] = field(default_factory=set)  # Which edge types belong to this layer
    filter_fn: Optional[Callable] = None  # Custom filter: (edge, context) -> bool
    switch_condition: Optional[Callable] = None  # When to auto-switch TO this layer
    priority: int = 0  # Higher = checked first for auto-switching


class LayerRegistry:
    """Central registry of layer configurations"""
    
    def __init__(self):
        self._layers: Dict[str, LayerConfig] = {}
        self._active: str = "spatial"  # Default
    
    def register(self, config: LayerConfig):
        self._layers[config.name] = config
    
    def get(self, name: str) -> Optional[LayerConfig]:
        return self._layers.get(name)
    
    def set_active(self, name: str):
        if name in self._layers:
            self._active = name
    
    @property
    def active(self) -> LayerConfig:
        return self._layers.get(self._active)
    
    @property
    def active_name(self) -> str:
        return self._active
    
    def check_auto_switch(self, context: 'TraversalContext') -> Optional[str]:
        """Check if any layer wants to auto-switch based on context"""
        candidates = sorted(
            self._layers.values(), 
            key=lambda l: l.priority, 
            reverse=True
        )
        for layer in candidates:
            if layer.switch_condition and layer.switch_condition(context):
                return layer.name
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TRAVERSAL CONTEXT - What the wrapper carries
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TraversalContext:
    """
    The state that travels with the wrapper.
    
    This is the wrapper's memory—everything it knows as it walks.
    """
    # Current position
    current_node: Any = None
    previous_node: Any = None
    
    # Path tracking
    path: List[Any] = field(default_factory=list)
    visited: Set[str] = field(default_factory=set)
    depth: int = 0
    
    # Layer state
    current_layer: str = "spatial"
    layer_stack: List[str] = field(default_factory=list)  # For push/pop
    
    # Game/app state
    flags: Dict[str, bool] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    inventory: List[str] = field(default_factory=list)
    
    # Scratch space
    buffer: Dict[str, Any] = field(default_factory=dict)
    
    # Callbacks accumulated during traversal
    pending_callbacks: List[Callable] = field(default_factory=list)
    
    # Graph reference (can be swapped!)
    graph_ref: Any = None
    
    def push_layer(self, layer: str):
        """Push current layer and switch to new one"""
        self.layer_stack.append(self.current_layer)
        self.current_layer = layer
    
    def pop_layer(self) -> str:
        """Pop back to previous layer"""
        if self.layer_stack:
            self.current_layer = self.layer_stack.pop()
        return self.current_layer
    
    def get_node_id(self, node: Any) -> str:
        """Extract ID from node (handles different node types)"""
        if hasattr(node, 'id'):
            return node.id
        if hasattr(node, 'asset_id'):
            return node.asset_id
        if isinstance(node, str):
            return node
        return str(id(node))
    
    def mark_visited(self, node: Any):
        """Mark a node as visited"""
        self.visited.add(self.get_node_id(node))
    
    def is_visited(self, node: Any) -> bool:
        """Check if node was visited"""
        return self.get_node_id(node) in self.visited
    
    def clone(self) -> 'TraversalContext':
        """Create a copy for branching traversals"""
        return TraversalContext(
            current_node=self.current_node,
            previous_node=self.previous_node,
            path=self.path.copy(),
            visited=self.visited.copy(),
            depth=self.depth,
            current_layer=self.current_layer,
            layer_stack=self.layer_stack.copy(),
            flags=self.flags.copy(),
            counters=self.counters.copy(),
            inventory=self.inventory.copy(),
            buffer=self.buffer.copy(),
            pending_callbacks=[],
            graph_ref=self.graph_ref
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE RESOLVER - Determines what edges are available
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeResolver:
    """
    Resolves which edges are traversable from a node.
    
    This is the brain that decides "where can we go from here?"
    It respects layers, conditions, and can be overridden.
    """
    
    def __init__(self, layer_registry: LayerRegistry = None):
        self.layers = layer_registry or LayerRegistry()
        self._custom_resolvers: List[Callable] = []
    
    def add_resolver(self, resolver: Callable):
        """Add custom resolver: (node, context) -> List[edges]"""
        self._custom_resolvers.append(resolver)
    
    def resolve(self, node: Any, context: TraversalContext) -> List[Any]:
        """
        Get all traversable edges from a node.
        
        Returns list of (edge, target_node) tuples.
        """
        edges = []
        
        # Get layer config
        layer = self.layers.get(context.current_layer)
        
        # Try to get edges from node
        raw_edges = self._get_raw_edges(node, context)
        
        # Filter by layer
        for edge in raw_edges:
            if self._edge_matches_layer(edge, layer, context):
                target = self._get_edge_target(edge, node)
                if target is not None:
                    edges.append((edge, target))
        
        # Custom resolvers can add more
        for resolver in self._custom_resolvers:
            try:
                extra = resolver(node, context)
                if extra:
                    edges.extend(extra)
            except Exception:
                pass
        
        return edges
    
    def _get_raw_edges(self, node: Any, context: TraversalContext) -> List[Any]:
        """Extract edges from a node (handles different node types)"""
        # GraphNode style (edges_out dict)
        if hasattr(node, 'edges_out'):
            all_edges = []
            for edge_list in node.edges_out.values():
                all_edges.extend(edge_list)
            return all_edges
        
        # Asset style (edges dict by layer)
        if hasattr(node, 'edges'):
            edges_dict = node.edges
            if isinstance(edges_dict, dict):
                # Try current layer first
                layer_edges = edges_dict.get(context.current_layer, [])
                return layer_edges
        
        # Direct list of edges
        if hasattr(node, 'get_edges'):
            return node.get_edges()
        
        # Neighbors list
        if hasattr(node, 'neighbors'):
            return [(None, n) for n in node.neighbors]
        
        return []
    
    def _edge_matches_layer(self, edge: Any, layer: LayerConfig, 
                           context: TraversalContext) -> bool:
        """Check if edge belongs to current layer"""
        if layer is None:
            return True
        
        # Custom filter takes precedence
        if layer.filter_fn:
            try:
                return layer.filter_fn(edge, context)
            except Exception:
                return False
        
        # Check edge type against layer's allowed types
        if layer.edge_types:
            edge_type = self._get_edge_type(edge)
            return edge_type in layer.edge_types
        
        return True
    
    def _get_edge_type(self, edge: Any) -> str:
        """Extract type from edge"""
        if hasattr(edge, 'edge_type'):
            t = edge.edge_type
            return t.name if hasattr(t, 'name') else str(t)
        if hasattr(edge, 'layer'):
            return edge.layer
        if isinstance(edge, tuple) and len(edge) >= 2:
            return str(edge[0])
        return "unknown"
    
    def _get_edge_target(self, edge: Any, source: Any) -> Any:
        """Extract target node from edge"""
        if hasattr(edge, 'target'):
            return edge.target
        if hasattr(edge, 'target_id'):
            # Need to resolve from graph
            return edge.target_id  # Wrapper will resolve
        if isinstance(edge, tuple):
            return edge[-1]  # Last element is target
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# THE WRAPPER - Smart iterator that carries context
# ═══════════════════════════════════════════════════════════════════════════════

class TraversalWrapper(Generic[T]):
    """
    A smart iterator wrapper that carries context through traversal.
    
    Key features:
    - Works with any iterable (lists, generators, graph queries)
    - Carries context that persists across iterations
    - Can switch graphs/layers mid-traversal
    - Supports callbacks at each step
    - Can be used in standard for loops
    
    The wrapper IS the cursor. When you iterate, you're moving the wrapper.
    """
    
    def __init__(self, 
                 source: Union[Iterable[T], Callable[[], Iterable[T]]] = None,
                 context: TraversalContext = None,
                 layer_registry: LayerRegistry = None,
                 edge_resolver: EdgeResolver = None):
        
        # Source can be an iterable or a callable that returns one
        self._source = source
        self._source_factory = source if callable(source) else None
        self._iterator: Optional[Iterator[T]] = None
        
        # Context travels with us
        self.context = context or TraversalContext()
        
        # Layer system
        self.layers = layer_registry or LayerRegistry()
        self.resolver = edge_resolver or EdgeResolver(self.layers)
        
        # Callbacks
        self._on_enter: List[Callable] = []
        self._on_exit: List[Callable] = []
        self._on_step: List[Callable] = []
        self._on_layer_switch: List[Callable] = []
        
        # State
        self._current: Optional[T] = None
        self._exhausted = False
        self._step_count = 0
    
    # ─────────────────────────────────────────────────────────────────────────
    # Iterator Protocol
    # ─────────────────────────────────────────────────────────────────────────
    
    def __iter__(self) -> 'TraversalWrapper[T]':
        """Start iteration - returns self (the wrapper carries through)"""
        if self._source_factory:
            self._source = self._source_factory()
        
        if self._source is not None:
            self._iterator = iter(self._source)
        
        self._exhausted = False
        return self
    
    def __next__(self) -> T:
        """Get next item, updating context"""
        if self._exhausted:
            raise StopIteration
        
        if self._iterator is None:
            raise StopIteration
        
        try:
            # Get next item
            item = next(self._iterator)
            
            # Update context
            self.context.previous_node = self._current
            self._current = item
            self.context.current_node = item
            self.context.path.append(item)
            self.context.mark_visited(item)
            self.context.depth = len(self.context.path)
            self._step_count += 1
            
            # Fire callbacks
            self._fire_on_step(item)
            
            # Check for auto layer switch
            new_layer = self.layers.check_auto_switch(self.context)
            if new_layer and new_layer != self.context.current_layer:
                self.switch_layer(new_layer)
            
            return item
            
        except StopIteration:
            self._exhausted = True
            self._fire_on_exit()
            raise
    
    # ─────────────────────────────────────────────────────────────────────────
    # Current State
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def current(self) -> Optional[T]:
        """Current item being visited"""
        return self._current
    
    @property
    def previous(self) -> Optional[T]:
        """Previously visited item"""
        return self.context.previous_node
    
    @property
    def depth(self) -> int:
        """Current depth in traversal"""
        return self.context.depth
    
    @property
    def step_count(self) -> int:
        """Total steps taken"""
        return self._step_count
    
    @property
    def layer(self) -> str:
        """Current layer name"""
        return self.context.current_layer
    
    # ─────────────────────────────────────────────────────────────────────────
    # Layer Switching
    # ─────────────────────────────────────────────────────────────────────────
    
    def switch_layer(self, layer_name: str, push: bool = False):
        """
        Switch to a different layer (edge set).
        
        If push=True, saves current layer for later pop.
        """
        old_layer = self.context.current_layer
        
        if push:
            self.context.push_layer(layer_name)
        else:
            self.context.current_layer = layer_name
        
        self.layers.set_active(layer_name)
        
        # Fire callbacks
        for cb in self._on_layer_switch:
            try:
                cb(self, old_layer, layer_name)
            except Exception:
                pass
    
    def pop_layer(self) -> str:
        """Pop back to previous layer"""
        old = self.context.current_layer
        new = self.context.pop_layer()
        self.layers.set_active(new)
        
        for cb in self._on_layer_switch:
            try:
                cb(self, old, new)
            except Exception:
                pass
        
        return new
    
    # ─────────────────────────────────────────────────────────────────────────
    # Graph Hotswap
    # ─────────────────────────────────────────────────────────────────────────
    
    def swap_graph(self, new_graph: Any, reset_position: bool = False):
        """
        Hotswap the underlying graph.
        
        The wrapper continues traversing but uses a different graph.
        """
        old_graph = self.context.graph_ref
        self.context.graph_ref = new_graph
        
        if reset_position:
            self._current = None
            self.context.current_node = None
            self.context.path = []
            self.context.visited = set()
    
    def swap_source(self, new_source: Union[Iterable[T], Callable]):
        """
        Replace the iteration source mid-traversal.
        
        Useful for jumping to a different part of the graph.
        """
        self._source = new_source
        self._source_factory = new_source if callable(new_source) else None
        
        if self._source_factory:
            self._source = self._source_factory()
        
        self._iterator = iter(self._source) if self._source else None
        self._exhausted = False
    
    # ─────────────────────────────────────────────────────────────────────────
    # Traversal Control
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_available_edges(self) -> List[Tuple[Any, Any]]:
        """Get edges available from current node"""
        if self._current is None:
            return []
        return self.resolver.resolve(self._current, self.context)
    
    def follow_edge(self, edge: Any) -> Optional[T]:
        """
        Follow a specific edge, updating context.
        
        Returns the target node.
        """
        target = self.resolver._get_edge_target(edge, self._current)
        
        if target is not None:
            # Resolve target if it's an ID
            if isinstance(target, str) and self.context.graph_ref:
                graph = self.context.graph_ref
                if hasattr(graph, 'get_node'):
                    target = graph.get_node(target)
                elif hasattr(graph, 'nodes'):
                    target = graph.nodes.get(target)
                elif hasattr(graph, 'assets'):
                    target = graph.assets.get(target)
            
            # Update state
            self.context.previous_node = self._current
            self._current = target
            self.context.current_node = target
            self.context.path.append(target)
            self.context.mark_visited(target)
            self._step_count += 1
            
            self._fire_on_step(target)
        
        return target
    
    def jump_to(self, node: Any):
        """
        Jump directly to a node (teleport).
        
        Doesn't require an edge—just moves the cursor.
        """
        self.context.previous_node = self._current
        self._current = node
        self.context.current_node = node
        self.context.path.append(node)
        self.context.mark_visited(node)
        self._step_count += 1
        
        self._fire_on_step(node)
    
    def branch(self) -> 'TraversalWrapper[T]':
        """
        Create a branch (clone) of this wrapper.
        
        Useful for exploring multiple paths simultaneously.
        """
        branched = TraversalWrapper(
            source=None,
            context=self.context.clone(),
            layer_registry=self.layers,
            edge_resolver=self.resolver
        )
        branched._current = self._current
        branched._step_count = self._step_count
        return branched
    
    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────
    
    def on_step(self, callback: Callable):
        """Register callback for each step: (wrapper, node) -> None"""
        self._on_step.append(callback)
        return self  # Chainable
    
    def on_enter(self, callback: Callable):
        """Register callback for iteration start"""
        self._on_enter.append(callback)
        return self
    
    def on_exit(self, callback: Callable):
        """Register callback for iteration end"""
        self._on_exit.append(callback)
        return self
    
    def on_layer_switch(self, callback: Callable):
        """Register callback for layer changes: (wrapper, old, new) -> None"""
        self._on_layer_switch.append(callback)
        return self
    
    def _fire_on_step(self, item: T):
        for cb in self._on_step:
            try:
                cb(self, item)
            except Exception:
                pass
    
    def _fire_on_exit(self):
        for cb in self._on_exit:
            try:
                cb(self)
            except Exception:
                pass
    
    # ─────────────────────────────────────────────────────────────────────────
    # Convenience Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def collect(self) -> List[T]:
        """Consume the iterator and return all items as list"""
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
        """Skip n items, return self for chaining"""
        for _ in range(n):
            try:
                next(self)
            except StopIteration:
                break
        return self
    
    def filter(self, predicate: Callable[[T], bool]) -> 'TraversalWrapper[T]':
        """Create filtered wrapper"""
        def filtered_source():
            for item in self:
                if predicate(item):
                    yield item
        
        return TraversalWrapper(
            source=filtered_source,
            context=self.context,
            layer_registry=self.layers,
            edge_resolver=self.resolver
        )
    
    def map(self, transform: Callable[[T], Any]) -> 'TraversalWrapper':
        """Create mapped wrapper"""
        def mapped_source():
            for item in self:
                yield transform(item)
        
        return TraversalWrapper(
            source=mapped_source,
            context=self.context,
            layer_registry=self.layers,
            edge_resolver=self.resolver
        )
    
    def __repr__(self):
        current_name = getattr(self._current, 'name', str(self._current)[:20])
        return f"<Wrapper at={current_name} layer={self.layer} depth={self.depth}>"


# ═══════════════════════════════════════════════════════════════════════════════
# LIST OVERRIDE - Make any list use smart iteration
# ═══════════════════════════════════════════════════════════════════════════════

class SmartList(list, Generic[T]):
    """
    A list subclass that uses TraversalWrapper for iteration.
    
    Drop-in replacement for list that carries context through for loops.
    """
    
    _wrapper_class: Type[TraversalWrapper] = TraversalWrapper
    _shared_context: Optional[TraversalContext] = None
    _layer_registry: Optional[LayerRegistry] = None
    
    @classmethod
    def set_wrapper_class(cls, wrapper_class: Type[TraversalWrapper]):
        """Override the wrapper class used for iteration"""
        cls._wrapper_class = wrapper_class
    
    @classmethod
    def set_shared_context(cls, context: TraversalContext):
        """Set a shared context for all SmartList iterations"""
        cls._shared_context = context
    
    @classmethod
    def set_layer_registry(cls, registry: LayerRegistry):
        """Set layer registry for all SmartList iterations"""
        cls._layer_registry = registry
    
    def __iter__(self) -> TraversalWrapper[T]:
        """Return a smart wrapper instead of standard iterator"""
        wrapper = self._wrapper_class(
            source=super().__iter__(),
            context=self._shared_context.clone() if self._shared_context else None,
            layer_registry=self._layer_registry
        )
        return wrapper
    
    def smart_iter(self, context: TraversalContext = None,
                   layer_registry: LayerRegistry = None) -> TraversalWrapper[T]:
        """Explicitly create a smart iterator with custom context"""
        return TraversalWrapper(
            source=super().__iter__(),
            context=context,
            layer_registry=layer_registry or self._layer_registry
        )


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH TRAVERSAL WRAPPER - Specialized for graph walking
# ═══════════════════════════════════════════════════════════════════════════════

class GraphTraverser(TraversalWrapper[N]):
    """
    Specialized wrapper for graph traversal.
    
    Understands graph structure and provides graph-specific operations.
    """
    
    def __init__(self, 
                 graph: Any,
                 start_node: Any = None,
                 strategy: str = "bfs",  # "bfs", "dfs", "edges"
                 layer: str = "spatial",
                 context: TraversalContext = None,
                 layer_registry: LayerRegistry = None):
        
        # Create context with graph reference
        ctx = context or TraversalContext()
        ctx.graph_ref = graph
        ctx.current_layer = layer
        
        super().__init__(
            source=None,  # We'll generate dynamically
            context=ctx,
            layer_registry=layer_registry
        )
        
        self.graph = graph
        self.start = start_node
        self.strategy = strategy
        
        # Traversal state
        self._frontier: deque = deque()
        self._started = False
    
    def __iter__(self) -> 'GraphTraverser[N]':
        """Initialize graph traversal"""
        self._started = True
        self._frontier.clear()
        self.context.visited.clear()
        self.context.path.clear()
        
        if self.start:
            start_node = self._resolve_node(self.start)
            if start_node:
                self._frontier.append(start_node)
        
        return self
    
    def __next__(self) -> N:
        """Get next node in traversal"""
        while self._frontier:
            if self.strategy == "dfs":
                node = self._frontier.pop()
            else:  # bfs
                node = self._frontier.popleft()
            
            # Skip if visited
            if self.context.is_visited(node):
                continue
            
            # Update context
            self.context.previous_node = self._current
            self._current = node
            self.context.current_node = node
            self.context.path.append(node)
            self.context.mark_visited(node)
            self._step_count += 1
            
            # Add neighbors to frontier
            edges = self.get_available_edges()
            for edge, target in edges:
                resolved = self._resolve_node(target)
                if resolved and not self.context.is_visited(resolved):
                    self._frontier.append(resolved)
            
            # Fire callbacks
            self._fire_on_step(node)
            
            # Check layer switch
            new_layer = self.layers.check_auto_switch(self.context)
            if new_layer and new_layer != self.context.current_layer:
                self.switch_layer(new_layer)
            
            return node
        
        # Exhausted
        self._fire_on_exit()
        raise StopIteration
    
    def _resolve_node(self, node_ref: Any) -> Any:
        """Resolve a node reference to actual node"""
        if node_ref is None:
            return None
        
        # Already a node object
        if hasattr(node_ref, 'id') or hasattr(node_ref, 'asset_id'):
            return node_ref
        
        # String ID - look up in graph
        if isinstance(node_ref, str):
            if hasattr(self.graph, 'get_node'):
                return self.graph.get_node(node_ref)
            if hasattr(self.graph, 'nodes'):
                return self.graph.nodes.get(node_ref)
            if hasattr(self.graph, 'assets'):
                return self.graph.assets.get(node_ref)
        
        return node_ref
    
    def neighbors(self, node: Any = None) -> List[N]:
        """Get neighbors of a node (or current node)"""
        target = node or self._current
        if target is None:
            return []
        
        edges = self.resolver.resolve(target, self.context)
        return [self._resolve_node(t) for e, t in edges if t]
    
    def from_node(self, node: Any) -> 'GraphTraverser[N]':
        """Start fresh traversal from a specific node"""
        self.start = node
        return self.__iter__()
    
    def within_depth(self, max_depth: int) -> 'GraphTraverser[N]':
        """Limit traversal to max depth"""
        original_next = self.__next__
        
        def limited_next():
            if self.context.depth > max_depth:
                raise StopIteration
            return original_next()
        
        self.__next__ = limited_next
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# INJECTABLE ITERATOR OVERRIDE
# ═══════════════════════════════════════════════════════════════════════════════

class IteratorOverride:
    """
    System for overriding default iteration behavior globally or per-type.
    
    This lets you inject smart wrappers into any iteration.
    """
    
    _overrides: Dict[type, Type[TraversalWrapper]] = {}
    _global_wrapper: Optional[Type[TraversalWrapper]] = None
    _global_context: Optional[TraversalContext] = None
    _enabled: bool = False
    
    @classmethod
    def enable(cls, wrapper_class: Type[TraversalWrapper] = TraversalWrapper,
               context: TraversalContext = None):
        """Enable global iterator override"""
        cls._global_wrapper = wrapper_class
        cls._global_context = context
        cls._enabled = True
    
    @classmethod
    def disable(cls):
        """Disable global override"""
        cls._enabled = False
    
    @classmethod
    def register_type(cls, target_type: type, 
                      wrapper_class: Type[TraversalWrapper]):
        """Register override for a specific type"""
        cls._overrides[target_type] = wrapper_class
    
    @classmethod
    def wrap(cls, iterable: Iterable[T]) -> Union[TraversalWrapper[T], Iterable[T]]:
        """
        Wrap an iterable with smart iterator if override is enabled.
        
        Usage:
            for item in IteratorOverride.wrap(my_list):
                # item comes through wrapper
        """
        if not cls._enabled:
            return iterable
        
        # Check type-specific override
        for target_type, wrapper_class in cls._overrides.items():
            if isinstance(iterable, target_type):
                return wrapper_class(
                    source=iterable,
                    context=cls._global_context.clone() if cls._global_context else None
                )
        
        # Use global wrapper
        if cls._global_wrapper:
            return cls._global_wrapper(
                source=iterable,
                context=cls._global_context.clone() if cls._global_context else None
            )
        
        return iterable


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def smart_iter(source: Iterable[T], 
               context: TraversalContext = None,
               layer: str = None,
               layer_registry: LayerRegistry = None) -> TraversalWrapper[T]:
    """
    Wrap any iterable with a smart iterator.
    
    This is the main entry point for smart iteration.
    """
    ctx = context or TraversalContext()
    if layer:
        ctx.current_layer = layer
    
    return TraversalWrapper(
        source=source,
        context=ctx,
        layer_registry=layer_registry
    )


def traverse_graph(graph: Any,
                   start: Any = None,
                   strategy: str = "bfs",
                   layer: str = "spatial",
                   context: TraversalContext = None,
                   layer_registry: LayerRegistry = None) -> GraphTraverser:
    """
    Create a graph traverser.
    
    This is the main entry point for graph traversal.
    """
    return GraphTraverser(
        graph=graph,
        start_node=start,
        strategy=strategy,
        layer=layer,
        context=context,
        layer_registry=layer_registry
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Traversal Wrapper - Demonstration")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Basic smart iteration
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Basic Smart Iteration ---")
    
    data = ["beach", "jungle", "temple", "cave", "treasure"]
    
    wrapper = smart_iter(data)
    wrapper.on_step(lambda w, item: print(f"  Step {w.step_count}: {item} (depth={w.depth})"))
    
    for location in wrapper:
        pass  # Callbacks handle output
    
    print(f"  Final context visited: {wrapper.context.visited}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Layer switching
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Layer Switching ---")
    
    # Setup layers
    layers = LayerRegistry()
    layers.register(LayerConfig(name="spatial", edge_types={"LEADS_TO", "ADJACENT"}))
    layers.register(LayerConfig(name="thematic", edge_types={"REPRESENTS", "SYMBOLIZES"}))
    layers.register(LayerConfig(
        name="danger",
        edge_types={"LEADS_TO"},
        switch_condition=lambda ctx: ctx.depth > 2  # Auto-switch at depth 3
    ))
    
    wrapper2 = smart_iter(data, layer="spatial", layer_registry=layers)
    wrapper2.on_layer_switch(lambda w, old, new: print(f"  Layer: {old} → {new}"))
    wrapper2.on_step(lambda w, item: print(f"  At {item} (layer={w.layer})"))
    
    for item in wrapper2:
        if item == "temple":
            wrapper2.switch_layer("thematic", push=True)
        elif item == "cave":
            wrapper2.pop_layer()
    
    # ─────────────────────────────────────────────────────────────────────────
    # SmartList - drop-in replacement
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- SmartList (drop-in replacement) ---")
    
    smart_data = SmartList(["alpha", "beta", "gamma", "delta"])
    
    # Get the wrapper and iterate - wrapper IS the iterator
    wrapper = iter(smart_data)
    for item in wrapper:
        print(f"  {item} at depth {wrapper.depth}, step {wrapper.step_count}")
    
    # Explicit smart iteration with context
    print("\n  With custom context:")
    ctx = TraversalContext()
    ctx.flags["special_mode"] = True
    
    wrapper2 = smart_data.smart_iter(context=ctx)
    for item in wrapper2:
        print(f"  {item} - special_mode={wrapper2.context.flags.get('special_mode')}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Chaining operations
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Chaining Operations ---")
    
    result = (
        smart_iter(range(10))
        .filter(lambda x: x % 2 == 0)
        .map(lambda x: x * 10)
        .take(3)
    )
    print(f"  Filtered/mapped/taken: {result}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Branching
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Branching ---")
    
    wrapper3 = smart_iter(["A", "B", "C", "D", "E"])
    iter(wrapper3)  # Initialize the iterator
    
    # Walk to B
    next(wrapper3)  # A
    next(wrapper3)  # B
    print(f"  Main at: {wrapper3.current}")
    
    # Branch at B
    branch = wrapper3.branch()
    print(f"  Branch at: {branch.current}")
    
    # Main continues
    next(wrapper3)  # C
    print(f"  Main moved to: {wrapper3.current}")
    print(f"  Branch still at: {branch.current}")
    
    print(f"  Main context path: {wrapper3.context.path}")
    print(f"  Branch context path: {branch.context.path}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
