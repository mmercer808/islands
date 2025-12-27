"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║          U N I F I E D   T R A V E R S A L   &   L O O K A H E A D           ║
║                                                                               ║
║                     A Complete Graph Intelligence System                      ║
║                                                                               ║
║  "Every word is on trial for its life." — Read Like a Writer                  ║
║                                                                               ║
║  This file contains the complete traversal and lookahead system for           ║
║  narrative graph exploration. It integrates with the Island Engine,           ║
║  my-infocom entity system, and any graph structure that follows the           ║
║  node/edge pattern.                                                           ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
                              DESIGN PHILOSOPHY
═══════════════════════════════════════════════════════════════════════════════

THE CORE INSIGHT: The wrapper IS the iterator.

In standard Python iteration:
    for item in my_list:
        # item is just data, orphaned from context

With the traversal wrapper:
    for item in wrapper:
        # item is data, BUT
        # wrapper.context carries state
        # wrapper.layer controls which edges are visible
        # wrapper can switch graphs mid-flight
        # wrapper knows where it's been and what's possible

The wrapper returns ITSELF from __iter__(), not a separate iterator object.
This means the wrapper travels with you through the for loop. It's not just
yielding items—it's a cursor with memory, moving through possibility space.

═══════════════════════════════════════════════════════════════════════════════
                              SYSTEM PIPELINE
═══════════════════════════════════════════════════════════════════════════════

The complete pipeline from novel-to-game-to-play:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1. NOVEL ANALYSIS (offline)                                             │
    │    Novel text → LLM extraction → Entities, Relationships, Scenes        │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 2. GRAPH CONSTRUCTION                                                   │
    │    Entities → Nodes (Assets)                                            │
    │    Relationships → Edges (Links) with layer assignment                  │
    │    Prose fragments attached to nodes                                    │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 3. LAYER CONFIGURATION                                                  │
    │    SPATIAL: physical adjacency (LEADS_TO, NORTH_OF, CONTAINS)           │
    │    TEMPORAL: time relationships (NEXT, BEFORE, DURING)                  │
    │    CHARACTER: social connections (KNOWS, TRUSTS, OWNS)                  │
    │    THEMATIC: abstract links (REPRESENTS, SYMBOLIZES)                    │
    │    NARRATIVE: story structure (TRIGGERS, REQUIRES, REVEALS)             │
    │    LOGICAL: game logic (UNLOCKS, BLOCKS, ENABLES)                       │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 4. RUNTIME TRAVERSAL                                                    │
    │    Player moves → TraversalWrapper navigates graph                      │
    │    Context carries: inventory, flags, visited, path                     │
    │    Layer switching: spatial → thematic → back to spatial                │
    │    Graph hotswap: main island → hidden caves → back                     │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 5. LOOKAHEAD (parallel to traversal)                                    │
    │    Pre-traverse possibility space                                       │
    │    Find: hidden items, locked doors, near-misses, puzzle chains         │
    │    Generate: hints, clues, "you're close" feelings                      │
    │    Feed: hint system, NPC dialogue, adaptive difficulty                 │
    └─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 6. PROSE COMPOSITION                                                    │
    │    Fragments filtered by conditions                                     │
    │    Sorted by category (BASE → ATMOSPHERIC → DISCOVERY)                  │
    │    Assembled into player-facing text                                    │
    │    Lookahead results inform what to emphasize                           │
    └─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                           KEY REVELATIONS
═══════════════════════════════════════════════════════════════════════════════

1. SAME NODES, DIFFERENT EDGES
   The graph doesn't change when you switch layers—the nodes are the same.
   What changes is which edges are visible. A character node exists in both
   SPATIAL (where they are) and CHARACTER (who they know) layers. Switching
   layers changes what connections you can follow, not what exists.

2. THE WRAPPER CARRIES ITSELF
   By returning self from __iter__(), the wrapper persists through iteration.
   This is how context survives. This is how layer switching works mid-loop.
   This is how you can hotswap graphs and keep walking.

3. LOOKAHEAD IS NOT MOVEMENT
   Lookahead explores possibility space without moving the player. It's the
   game's awareness of what COULD happen. It finds locked doors before you
   reach them. It knows which items are needed. It sees near-misses.

4. NEAR-MISSES ARE GOLD
   A near-miss is when you're one condition away from something. The key is
   in the next room. You need one more item. These are perfect for hints
   because they're achievable—the player is close.

5. PUZZLE CHAINS TRACE BACKWARDS
   To find what's needed to reach X, trace incoming links. X requires Y,
   Y is locked by Z, Z needs key K. The chain: K → Z → Y → X. Lookahead
   builds these chains automatically.

6. PROSE FRAGMENTS ARE CONDITIONAL
   A fragment isn't just text—it's text WITH conditions. Show this line
   only if the player has the key. Show this only on first visit. Show
   this only if the NPC trusts you. The ProseCompositor filters and
   assembles at runtime.

7. GRAPHS CAN BE HOTSWAPPED
   The wrapper doesn't own the graph—it references it. Change the reference,
   continue walking. Portal to another world? Swap graphs, jump to entry node,
   keep iterating. The context persists.

8. SMARTLIST IS A DROP-IN
   By subclassing list and overriding __iter__(), any existing code that
   iterates over lists suddenly gets smart iteration. No changes needed
   to the calling code.

═══════════════════════════════════════════════════════════════════════════════
                           INTEGRATION POINTS
═══════════════════════════════════════════════════════════════════════════════

WITH graph_core.py (IslandGraph):
    - GraphNode.edges_out → _get_raw_edges()
    - EdgeType enum → layer filtering
    - IslandGraph.nodes dict → entity resolution

WITH my_infocom_entity_system.py (EntityGraph):
    - Entity.id → entity resolution
    - Link.link_type (RELATIONAL/LOGICAL/WILDCARD) → layer routing
    - Link.kind (RelationalKind/LogicalKind) → specific handling
    - LinkCondition → condition evaluation
    - ProseFragment → hint generation

WITH island_engine_design.md (IslandEngine):
    - Asset → node type
    - CodeIndex → behavior execution (not in this file)
    - EventBus → trigger on traversal events
    - White Room → origin point for traversal

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Dict, List, Set, Optional, Any, Callable, Iterator, Iterable,
    Tuple, Union, Generic, TypeVar, Generator, Type
)
from enum import Enum, auto
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import weakref
import copy
import time
import heapq


# Type variables for generic systems
T = TypeVar('T')
N = TypeVar('N')  # Node type


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    PART 1: LAYER SYSTEM                                   ║
# ║                                                                           ║
# ║  Layers are named edge sets. Same nodes, different connections visible.   ║
# ║  Switch layers to change what paths are available.                        ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Layer(Enum):
    """
    Named edge sets for traversal filtering.
    
    Each layer represents a different "view" of the same graph.
    SPATIAL sees physical connections. THEMATIC sees symbolic ones.
    The nodes don't change—only which edges are visible.
    """
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CHARACTER = "character"
    THEMATIC = "thematic"
    NARRATIVE = "narrative"
    CAUSAL = "causal"
    LOGICAL = "logical"
    RELATIONAL = "relational"
    WILDCARD = "wildcard"
    CUSTOM = "custom"


@dataclass
class LayerConfig:
    """
    Configuration for a layer's traversal behavior.
    
    A layer config defines:
    - Which edge types belong to this layer
    - Optional custom filter function
    - When to auto-switch TO this layer
    - Priority for auto-switch checking
    """
    name: str
    edge_types: Set[str] = field(default_factory=set)
    filter_fn: Optional[Callable] = None  # (edge, context) → bool
    switch_condition: Optional[Callable] = None  # When to auto-switch here
    priority: int = 0  # Higher = checked first


class LayerRegistry:
    """
    Central registry of layer configurations.
    
    The registry knows all available layers and which is active.
    It can check if any layer wants to auto-switch based on context.
    """
    
    def __init__(self):
        self._layers: Dict[str, LayerConfig] = {}
        self._active: str = "spatial"
    
    def register(self, config: LayerConfig):
        """Add a layer configuration"""
        self._layers[config.name] = config
    
    def get(self, name: str) -> Optional[LayerConfig]:
        """Get layer config by name"""
        return self._layers.get(name)
    
    def set_active(self, name: str):
        """Set the active layer"""
        if name in self._layers:
            self._active = name
    
    @property
    def active(self) -> LayerConfig:
        """Get the currently active layer config"""
        return self._layers.get(self._active)
    
    @property
    def active_name(self) -> str:
        """Get name of active layer"""
        return self._active
    
    def check_auto_switch(self, context: 'TraversalContext') -> Optional[str]:
        """
        Check if any layer wants to auto-switch based on context.
        
        Returns the name of the layer to switch to, or None.
        Checks layers in priority order (highest first).
        """
        candidates = sorted(
            self._layers.values(),
            key=lambda l: l.priority,
            reverse=True
        )
        for layer in candidates:
            if layer.switch_condition and layer.switch_condition(context):
                return layer.name
        return None


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    PART 2: TRAVERSAL CONTEXT                              ║
# ║                                                                           ║
# ║  The context is the wrapper's memory. It travels with the wrapper         ║
# ║  through iteration, carrying state, history, and game data.               ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class TraversalContext:
    """
    The state that travels with the wrapper.
    
    This is everything the wrapper "knows" as it walks the graph:
    - Where it is and where it's been
    - What layer it's on
    - Game state (flags, counters, inventory)
    - Scratch space for callbacks
    - Reference to the current graph (can be swapped!)
    
    The context persists across iterations. It survives layer switches.
    It survives graph hotswaps. It's the wrapper's soul.
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
    
    # Game state (matches my_infocom patterns)
    flags: Dict[str, bool] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    inventory: List[str] = field(default_factory=list)
    memories: List[Dict] = field(default_factory=list)
    
    # Scratch space for callbacks and lookahead results
    buffer: Dict[str, Any] = field(default_factory=dict)
    
    # Callbacks accumulated during traversal
    pending_callbacks: List[Callable] = field(default_factory=list)
    
    # Graph reference (can be swapped!)
    graph_ref: Any = None
    
    def push_layer(self, layer: str):
        """Push current layer onto stack, switch to new one"""
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
    
    def set_flag(self, name: str, value: bool = True):
        """Set a game flag"""
        self.flags[name] = value
    
    def get_flag(self, name: str, default: bool = False) -> bool:
        """Get a game flag"""
        return self.flags.get(name, default)
    
    def increment(self, counter: str, amount: int = 1):
        """Increment a counter"""
        self.counters[counter] = self.counters.get(counter, 0) + amount
    
    def get_counter(self, counter: str, default: int = 0) -> int:
        """Get a counter value"""
        return self.counters.get(counter, default)
    
    def add_memory(self, memory_type: str, content: Any, source: str = None):
        """Add a memory (discovered information)"""
        self.memories.append({
            'type': memory_type,
            'content': content,
            'source': source,
            'timestamp': time.time()
        })
    
    def clone(self) -> 'TraversalContext':
        """
        Create a copy for branching traversals.
        
        Use this when you want to explore multiple paths simultaneously.
        Each branch gets its own context that evolves independently.
        """
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
            memories=self.memories.copy(),
            buffer=self.buffer.copy(),
            pending_callbacks=[],
            graph_ref=self.graph_ref
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    PART 3: EDGE RESOLVER                                  ║
# ║                                                                           ║
# ║  The resolver decides which edges are traversable from any node.          ║
# ║  It respects layers, evaluates conditions, handles different graph types. ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class EdgeResolver:
    """
    Resolves which edges are traversable from a node.
    
    This is the brain that answers "where can we go from here?"
    It:
    - Extracts edges from different node formats (GraphNode, Entity, Asset)
    - Filters by current layer
    - Evaluates edge conditions
    - Can be extended with custom resolvers
    """
    
    def __init__(self, layer_registry: LayerRegistry = None):
        self.layers = layer_registry or LayerRegistry()
        self._custom_resolvers: List[Callable] = []
    
    def add_resolver(self, resolver: Callable):
        """Add custom resolver: (node, context) → List[(edge, target)]"""
        self._custom_resolvers.append(resolver)
    
    def resolve(self, node: Any, context: TraversalContext) -> List[Tuple[Any, Any]]:
        """
        Get all traversable edges from a node.
        
        Returns list of (edge, target_node) tuples.
        """
        edges = []
        layer = self.layers.get(context.current_layer)
        raw_edges = self._get_raw_edges(node, context)
        
        for edge in raw_edges:
            if self._edge_matches_layer(edge, layer, context):
                target = self._get_edge_target(edge, node)
                if target is not None:
                    edges.append((edge, target))
        
        # Custom resolvers can add more edges
        for resolver in self._custom_resolvers:
            try:
                extra = resolver(node, context)
                if extra:
                    edges.extend(extra)
            except Exception:
                pass
        
        return edges
    
    def _get_raw_edges(self, node: Any, context: TraversalContext) -> List[Any]:
        """Extract edges from node (handles different formats)"""
        
        # GraphNode style (graph_core.py)
        if hasattr(node, 'edges_out'):
            all_edges = []
            for edge_list in node.edges_out.values():
                all_edges.extend(edge_list)
            return all_edges
        
        # Entity style (my_infocom_entity_system.py)
        # Entities don't store edges directly—graph does
        # This is handled by custom resolver or graph query
        
        # Asset style (island_engine_design.md)
        if hasattr(node, 'edges'):
            edges_dict = node.edges
            if isinstance(edges_dict, dict):
                layer_edges = edges_dict.get(context.current_layer, [])
                return layer_edges
        
        # Direct edges method
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
        if hasattr(edge, 'link_type'):
            t = edge.link_type
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
            return edge.target_id  # Caller resolves
        if isinstance(edge, tuple):
            return edge[-1]
        return None


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    PART 4: TRAVERSAL WRAPPER                              ║
# ║                                                                           ║
# ║  The wrapper is the smart iterator. It carries context through            ║
# ║  iteration. It returns ITSELF from __iter__(), so it persists.            ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TraversalWrapper(Generic[T]):
    """
    A smart iterator wrapper that carries context through traversal.
    
    THE KEY INSIGHT: __iter__() returns self.
    
    This means:
    - The wrapper persists through the for loop
    - Context survives across iterations
    - You can switch layers mid-iteration
    - You can hotswap graphs and keep walking
    - You can branch into parallel traversals
    
    Usage:
        wrapper = TraversalWrapper(my_list)
        for item in wrapper:
            print(f"At {item}, depth={wrapper.depth}")
            if should_switch:
                wrapper.switch_layer("thematic")
    """
    
    def __init__(self,
                 source: Union[Iterable[T], Callable[[], Iterable[T]]] = None,
                 context: TraversalContext = None,
                 layer_registry: LayerRegistry = None,
                 edge_resolver: EdgeResolver = None):
        
        # Source can be iterable or callable returning iterable
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
    # Iterator Protocol - THE CORE MAGIC
    # ─────────────────────────────────────────────────────────────────────────
    
    def __iter__(self) -> 'TraversalWrapper[T]':
        """
        Start iteration - returns SELF.
        
        This is the key insight. By returning self, the wrapper
        persists through the for loop. It's not just yielding items—
        it's traveling with you.
        """
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
            item = next(self._iterator)
            
            # Update context
            self.context.previous_node = self._current
            self._current = item
            self.context.current_node = item
            self.context.path.append(item)
            self.context.mark_visited(item)
            self.context.depth = len(self.context.path)
            self._step_count += 1
            
            # Fire step callbacks
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
    # Current State Properties
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
    # Layer Switching - Change what edges are visible
    # ─────────────────────────────────────────────────────────────────────────
    
    def switch_layer(self, layer_name: str, push: bool = False):
        """
        Switch to a different layer (edge set).
        
        The nodes don't change—only which edges you can follow.
        
        If push=True, saves current layer for later pop.
        This is useful for temporary excursions: "look at thematic
        connections, then return to spatial navigation."
        """
        old_layer = self.context.current_layer
        
        if push:
            self.context.push_layer(layer_name)
        else:
            self.context.current_layer = layer_name
        
        self.layers.set_active(layer_name)
        
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
    # Graph Hotswap - Change what world you're in
    # ─────────────────────────────────────────────────────────────────────────
    
    def swap_graph(self, new_graph: Any, reset_position: bool = False):
        """
        Hotswap the underlying graph.
        
        The wrapper continues but now traverses a different graph.
        Context persists—you keep your inventory, flags, memories.
        
        Use case: Portal to another world. Step through, swap graphs,
        continue walking in the new world.
        """
        self.context.graph_ref = new_graph
        
        if reset_position:
            self._current = None
            self.context.current_node = None
            self.context.path = []
            self.context.visited = set()
    
    def swap_source(self, new_source: Union[Iterable[T], Callable]):
        """
        Replace the iteration source mid-traversal.
        
        Useful for jumping to a different part of the graph
        or switching to a computed path.
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
            # Resolve if it's an ID
            if isinstance(target, str) and self.context.graph_ref:
                graph = self.context.graph_ref
                if hasattr(graph, 'get_node'):
                    target = graph.get_node(target)
                elif hasattr(graph, 'nodes'):
                    target = graph.nodes.get(target)
                elif hasattr(graph, '_nodes'):
                    target = graph._nodes.get(target)
            
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
        
        Use for exploring multiple paths simultaneously.
        Each branch evolves independently.
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
    
    def on_step(self, callback: Callable) -> 'TraversalWrapper[T]':
        """Register callback for each step: (wrapper, node) → None"""
        self._on_step.append(callback)
        return self
    
    def on_enter(self, callback: Callable) -> 'TraversalWrapper[T]':
        """Register callback for iteration start"""
        self._on_enter.append(callback)
        return self
    
    def on_exit(self, callback: Callable) -> 'TraversalWrapper[T]':
        """Register callback for iteration end"""
        self._on_exit.append(callback)
        return self
    
    def on_layer_switch(self, callback: Callable) -> 'TraversalWrapper[T]':
        """Register callback for layer changes: (wrapper, old, new) → None"""
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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    PART 5: SMART LIST                                     ║
# ║                                                                           ║
# ║  A list subclass that uses smart iteration. Drop-in replacement.          ║
# ║  Existing code that iterates over lists gets smart iteration for free.    ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SmartList(list, Generic[T]):
    """
    A list subclass that uses TraversalWrapper for iteration.
    
    Drop-in replacement for list. Existing code that does:
        for item in my_list:
            ...
    
    Gets smart iteration automatically when my_list is a SmartList.
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
        """Explicitly create smart iterator with custom context"""
        return TraversalWrapper(
            source=super().__iter__(),
            context=context,
            layer_registry=layer_registry or self._layer_registry
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    PART 6: GRAPH TRAVERSER                                ║
# ║                                                                           ║
# ║  Specialized wrapper for graph traversal with BFS/DFS strategies.         ║
# ║  Understands graph structure and provides graph-specific operations.      ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class GraphTraverser(TraversalWrapper[N]):
    """
    Specialized wrapper for graph traversal.
    
    Provides:
    - BFS and DFS traversal strategies
    - Automatic neighbor expansion
    - Depth limiting
    - Graph-aware operations
    """
    
    def __init__(self,
                 graph: Any,
                 start_node: Any = None,
                 strategy: str = "bfs",  # "bfs" or "dfs"
                 layer: str = "spatial",
                 context: TraversalContext = None,
                 layer_registry: LayerRegistry = None):
        
        ctx = context or TraversalContext()
        ctx.graph_ref = graph
        ctx.current_layer = layer
        
        super().__init__(
            source=None,
            context=ctx,
            layer_registry=layer_registry
        )
        
        self.graph = graph
        self.start = start_node
        self.strategy = strategy
        
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
            
            self._fire_on_step(node)
            
            # Check layer switch
            new_layer = self.layers.check_auto_switch(self.context)
            if new_layer and new_layer != self.context.current_layer:
                self.switch_layer(new_layer)
            
            return node
        
        self._fire_on_exit()
        raise StopIteration
    
    def _resolve_node(self, node_ref: Any) -> Any:
        """Resolve node reference to actual node"""
        if node_ref is None:
            return None
        
        if hasattr(node_ref, 'id') or hasattr(node_ref, 'asset_id'):
            return node_ref
        
        if isinstance(node_ref, str):
            if hasattr(self.graph, 'get_node'):
                return self.graph.get_node(node_ref)
            if hasattr(self.graph, 'nodes'):
                return self.graph.nodes.get(node_ref)
            if hasattr(self.graph, '_nodes'):
                return self.graph._nodes.get(node_ref)
        
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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    PART 7: POSSIBILITY TYPES                              ║
# ║                                                                           ║
# ║  What kinds of potential can the lookahead discover?                      ║
# ║  These categories help organize what's findable.                          ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class PossibilityType(Enum):
    """Categories of things the lookahead can discover"""
    
    # Spatial
    REACHABLE_LOCATION = auto()     # Can walk there now
    BLOCKED_PATH = auto()           # Path exists but blocked
    LOCKED_DOOR = auto()            # Door needs key/condition
    
    # Items
    VISIBLE_ITEM = auto()           # Item is here and visible
    HIDDEN_ITEM = auto()            # Item exists but hidden
    TAKEABLE_ITEM = auto()          # Item can be picked up
    REQUIRED_ITEM = auto()          # Item needed for something
    
    # Logical
    UNLOCKABLE = auto()             # Could be unlocked
    REVEALABLE = auto()             # Could be revealed
    TRIGGERABLE = auto()            # Event could be triggered
    CONDITIONAL_BLOCKED = auto()    # Blocked by unmet condition
    
    # Narrative
    DISCOVERABLE_CLUE = auto()      # Clue waiting to be found
    NPC_DIALOGUE = auto()           # Conversation available
    STORY_BEAT = auto()             # Narrative trigger available
    
    # Meta
    NEAR_MISS = auto()              # Almost meets condition
    DEAD_END = auto()               # No further possibilities
    PUZZLE_PIECE = auto()           # Part of a puzzle chain


@dataclass
class Possibility:
    """
    A single possibility discovered by lookahead.
    
    Represents something that COULD happen, might be available,
    or is blocked by known conditions.
    """
    possibility_type: PossibilityType
    entity_id: str
    entity_name: str
    
    # Conditions
    conditions_met: List[str] = field(default_factory=list)
    conditions_unmet: List[str] = field(default_factory=list)
    
    # Proximity
    distance: int = 0
    accessibility: float = 1.0  # 0.0 = impossible, 1.0 = immediate
    
    # What would make it accessible?
    requirements: List[str] = field(default_factory=list)
    
    # Narrative hint
    hint_text: Optional[str] = None
    
    # Path to get here
    path: List[str] = field(default_factory=list)
    
    # Link info
    via_link_type: Optional[str] = None
    via_link_kind: Optional[str] = None
    
    # Sorting
    priority: int = 50
    
    def is_blocked(self) -> bool:
        return len(self.conditions_unmet) > 0
    
    def is_near_miss(self, threshold: int = 1) -> bool:
        """True if only one or few conditions away"""
        return 0 < len(self.conditions_unmet) <= threshold
    
    def __lt__(self, other):
        return (-self.priority, self.distance, -self.accessibility) < \
               (-other.priority, other.distance, -other.accessibility)


@dataclass
class LookaheadResult:
    """
    Complete result of a lookahead operation.
    
    Contains all possibilities found, indexed by type and entity.
    Provides query methods for common patterns.
    """
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
    
    total_entities_seen: int = 0
    total_links_traversed: int = 0
    blocked_count: int = 0
    near_miss_count: int = 0
    
    def add(self, possibility: Possibility):
        """Add a possibility and index it"""
        self.all_possibilities.append(possibility)
        self.by_type[possibility.possibility_type].append(possibility)
        self.by_entity[possibility.entity_id].append(possibility)
        
        if possibility.is_blocked():
            self.blocked_count += 1
        if possibility.is_near_miss():
            self.near_miss_count += 1
    
    def get_reachable(self) -> List[Possibility]:
        """Get immediately reachable locations"""
        return [p for p in self.all_possibilities
                if p.possibility_type == PossibilityType.REACHABLE_LOCATION
                and not p.is_blocked()]
    
    def get_blocked(self) -> List[Possibility]:
        """Get all blocked possibilities"""
        return [p for p in self.all_possibilities if p.is_blocked()]
    
    def get_near_misses(self, threshold: int = 1) -> List[Possibility]:
        """Get possibilities that are almost accessible"""
        return [p for p in self.all_possibilities if p.is_near_miss(threshold)]
    
    def get_hidden_items(self) -> List[Possibility]:
        """Get hidden items that could be revealed"""
        return self.by_type.get(PossibilityType.HIDDEN_ITEM, []) + \
               self.by_type.get(PossibilityType.REVEALABLE, [])
    
    def get_required_items(self) -> List[Possibility]:
        """Get items needed for something"""
        return self.by_type.get(PossibilityType.REQUIRED_ITEM, [])
    
    def get_puzzle_chain(self, target_id: str) -> List[Possibility]:
        """
        Get the chain of things needed to reach/unlock a target.
        
        Traces backwards from target to find all prerequisites.
        This is the key to understanding puzzle dependencies.
        """
        chain = []
        seen = set()
        frontier = [target_id]
        
        while frontier:
            current = frontier.pop(0)
            if current in seen:
                continue
            seen.add(current)
            
            for poss in self.by_entity.get(current, []):
                chain.append(poss)
                for req in poss.requirements:
                    if req not in seen:
                        frontier.append(req)
        
        return chain
    
    def get_hints(self, max_hints: int = 3) -> List[str]:
        """
        Generate hint text for the player.
        
        Near-misses are gold—they're achievable.
        Hidden items nearby create intrigue.
        """
        hints = []
        
        # Near-misses first
        for nm in self.get_near_misses()[:max_hints]:
            if nm.hint_text:
                hints.append(nm.hint_text)
            elif nm.conditions_unmet:
                hints.append(f"Something about {nm.entity_name} seems significant...")
        
        # Hidden items nearby
        for hidden in self.get_hidden_items()[:max_hints - len(hints)]:
            if hidden.distance <= 1 and hidden.hint_text:
                hints.append(hidden.hint_text)
        
        return hints[:max_hints]
    
    def summary(self) -> str:
        """Human-readable summary"""
        lines = [
            f"Lookahead from: {self.origin_name}",
            f"Depth: {self.max_depth}",
            f"Entities seen: {self.total_entities_seen}",
            f"Links traversed: {self.total_links_traversed}",
            f"Possibilities: {len(self.all_possibilities)}",
            f"Blocked: {self.blocked_count}",
            f"Near-misses: {self.near_miss_count}",
            "",
            "By type:",
        ]
        for ptype, possibilities in sorted(self.by_type.items(), key=lambda x: -len(x[1])):
            lines.append(f"  {ptype.name}: {len(possibilities)}")
        
        return "\n".join(lines)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    PART 8: LOOKAHEAD ENGINE                               ║
# ║                                                                           ║
# ║  The lookahead pre-traverses the graph WITHOUT moving the player.         ║
# ║  It finds what COULD happen, what's blocked, what's almost reachable.     ║
# ║  It's the oracle that knows the possibility space.                        ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class LookaheadEngine:
    """
    Pre-traverses the entity graph to discover possibilities.
    
    The lookahead doesn't move the player. It explores the possibility
    space and reports what it finds:
    - What's reachable?
    - What's blocked and why?
    - What's hidden that could be revealed?
    - What items are needed for what?
    - What's a near-miss (almost accessible)?
    
    It traverses ALL link types simultaneously (RELATIONAL, LOGICAL, WILDCARD)
    to build a complete picture of what's possible.
    """
    
    def __init__(self, graph: Any = None):
        self.graph = graph
        self._type_handlers: Dict[str, Callable] = {}
        self._condition_evaluators: Dict[str, Callable] = {}
    
    def set_graph(self, graph: Any):
        """Set or change the graph being analyzed"""
        self.graph = graph
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main Lookahead Interface
    # ─────────────────────────────────────────────────────────────────────────
    
    def lookahead(self,
                  from_entity: Any,
                  context: Dict[str, Any] = None,
                  max_depth: int = 3,
                  include_blocked: bool = True,
                  layers: List[str] = None) -> LookaheadResult:
        """
        Perform lookahead from a starting entity.
        
        Args:
            from_entity: Starting point (entity or ID)
            context: Current game state
            max_depth: How far to look
            include_blocked: Include blocked possibilities?
            layers: Which link types to follow (None = all)
        
        Returns:
            LookaheadResult with all discovered possibilities
        """
        if self.graph is None:
            raise ValueError("No graph set")
        
        context = context or {}
        start_id, start_entity = self._resolve_entity(from_entity)
        
        result = LookaheadResult(
            origin_id=start_id,
            origin_name=getattr(start_entity, 'name', start_id),
            max_depth=max_depth
        )
        
        visited = set()
        frontier = deque([(start_id, 0, [start_id])])
        
        while frontier:
            current_id, depth, path = frontier.popleft()
            
            if current_id in visited:
                continue
            visited.add(current_id)
            result.total_entities_seen += 1
            
            current = self._get_entity(current_id)
            if current is None:
                continue
            
            # Analyze entity
            for poss in self._analyze_entity(current, context, depth, path):
                if include_blocked or not poss.is_blocked():
                    result.add(poss)
            
            if depth >= max_depth:
                continue
            
            # Get and analyze links
            for link in self._get_links(current_id, layers):
                result.total_links_traversed += 1
                
                for poss in self._analyze_link(link, current_id, context, depth, path):
                    if include_blocked or not poss.is_blocked():
                        result.add(poss)
                
                target_id = self._get_link_target(link, current_id)
                if target_id and target_id not in visited:
                    frontier.append((target_id, depth + 1, path + [target_id]))
        
        return result
    
    def lookahead_for(self,
                      from_entity: Any,
                      target_entity: Any,
                      context: Dict[str, Any] = None,
                      max_depth: int = 5) -> Optional[LookaheadResult]:
        """
        Focused lookahead to find paths/requirements to reach a target.
        """
        result = self.lookahead(from_entity, context, max_depth)
        target_id, _ = self._resolve_entity(target_entity)
        
        if target_id not in result.by_entity:
            return None
        
        filtered = LookaheadResult(
            origin_id=result.origin_id,
            origin_name=result.origin_name,
            max_depth=max_depth
        )
        
        for poss in result.by_entity[target_id]:
            filtered.add(poss)
        
        for poss in result.get_puzzle_chain(target_id):
            if poss not in filtered.all_possibilities:
                filtered.add(poss)
        
        return filtered
    
    def what_unlocks(self,
                     entity: Any,
                     context: Dict[str, Any] = None) -> List[Possibility]:
        """Find what the given entity could unlock/reveal/enable"""
        context = context or {}
        entity_id, entity_obj = self._resolve_entity(entity)
        possibilities = []
        
        for link in self._get_links(entity_id, ['LOGICAL']):
            target_id = self._get_link_target(link, entity_id)
            target = self._get_entity(target_id)
            if target is None:
                continue
            
            kind = self._get_link_kind(link)
            if kind in ('UNLOCKS', 'REVEALS', 'ENABLES', 'TRIGGERS'):
                poss = Possibility(
                    possibility_type=self._kind_to_type(kind),
                    entity_id=target_id,
                    entity_name=getattr(target, 'name', target_id),
                    via_link_type='LOGICAL',
                    via_link_kind=kind,
                    hint_text=f"This could affect {getattr(target, 'name', target_id)}..."
                )
                
                if hasattr(link, 'condition') and link.condition:
                    if self._evaluate_condition(link.condition, context):
                        poss.conditions_met.append(str(link.condition))
                    else:
                        poss.conditions_unmet.append(str(link.condition))
                
                possibilities.append(poss)
        
        return possibilities
    
    def what_blocks(self,
                    entity: Any,
                    context: Dict[str, Any] = None) -> List[Possibility]:
        """Find what is blocking access to this entity"""
        context = context or {}
        entity_id, entity_obj = self._resolve_entity(entity)
        blockers = []
        
        for link in self._get_incoming_links(entity_id):
            kind = self._get_link_kind(link)
            if kind in ('REQUIRES', 'BLOCKS'):
                source_id = self._get_link_source(link)
                source = self._get_entity(source_id)
                
                poss = Possibility(
                    possibility_type=PossibilityType.CONDITIONAL_BLOCKED,
                    entity_id=source_id,
                    entity_name=getattr(source, 'name', source_id) if source else source_id,
                    via_link_type=self._get_link_type(link),
                    via_link_kind=kind
                )
                
                if kind == 'REQUIRES':
                    if self._check_requirement(source_id, context):
                        poss.conditions_met.append(f"Have {poss.entity_name}")
                    else:
                        poss.conditions_unmet.append(f"Need {poss.entity_name}")
                        poss.requirements.append(source_id)
                
                blockers.append(poss)
        
        # Check entity state
        if hasattr(entity_obj, 'state'):
            state = entity_obj.state
            state_name = state.name if hasattr(state, 'name') else str(state)
            
            if state_name in ('LOCKED', 'HIDDEN', 'CLOSED'):
                blockers.append(Possibility(
                    possibility_type=PossibilityType.CONDITIONAL_BLOCKED,
                    entity_id=entity_id,
                    entity_name=getattr(entity_obj, 'name', entity_id),
                    conditions_unmet=[f"Currently {state_name}"]
                ))
        
        return blockers
    
    # ─────────────────────────────────────────────────────────────────────────
    # Entity Analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    def _analyze_entity(self, entity: Any, context: Dict[str, Any],
                        depth: int, path: List[str]) -> List[Possibility]:
        """Analyze entity for possibilities"""
        possibilities = []
        
        entity_id = self._get_entity_id(entity)
        entity_name = getattr(entity, 'name', entity_id)
        entity_type = self._get_entity_type(entity)
        entity_state = self._get_entity_state(entity)
        
        if entity_type == 'LOCATION':
            possibilities.append(Possibility(
                possibility_type=PossibilityType.REACHABLE_LOCATION,
                entity_id=entity_id,
                entity_name=entity_name,
                distance=depth,
                path=path.copy()
            ))
        
        elif entity_type == 'ITEM':
            if entity_state == 'HIDDEN':
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.HIDDEN_ITEM,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    distance=depth,
                    path=path.copy(),
                    hint_text="There might be something hidden here..."
                ))
            else:
                poss_type = PossibilityType.TAKEABLE_ITEM if self._is_takeable(entity) \
                           else PossibilityType.VISIBLE_ITEM
                possibilities.append(Possibility(
                    possibility_type=poss_type,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    distance=depth,
                    path=path.copy()
                ))
        
        elif entity_type == 'CONTAINER':
            if entity_state in ('LOCKED', 'CLOSED'):
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.LOCKED_DOOR if entity_state == 'LOCKED'
                                    else PossibilityType.CONDITIONAL_BLOCKED,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    distance=depth,
                    path=path.copy(),
                    conditions_unmet=[f"{entity_name} is {entity_state.lower()}"]
                ))
        
        elif entity_type == 'DOOR':
            if entity_state == 'LOCKED':
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.LOCKED_DOOR,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    distance=depth,
                    path=path.copy(),
                    conditions_unmet=["Locked"],
                    hint_text=f"The {entity_name} is locked. You'll need a key."
                ))
        
        elif entity_type == 'CHARACTER':
            possibilities.append(Possibility(
                possibility_type=PossibilityType.NPC_DIALOGUE,
                entity_id=entity_id,
                entity_name=entity_name,
                distance=depth,
                path=path.copy()
            ))
        
        elif entity_type == 'CONCEPT':
            if entity_state == 'HIDDEN':
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.DISCOVERABLE_CLUE,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    distance=depth,
                    path=path.copy(),
                    hint_text="There's something important you haven't discovered yet."
                ))
        
        return possibilities
    
    def _analyze_link(self, link: Any, from_id: str, context: Dict[str, Any],
                      depth: int, path: List[str]) -> List[Possibility]:
        """Analyze link for possibilities"""
        possibilities = []
        
        link_type = self._get_link_type(link)
        link_kind = self._get_link_kind(link)
        target_id = self._get_link_target(link, from_id)
        target = self._get_entity(target_id)
        
        if target is None:
            return possibilities
        
        target_name = getattr(target, 'name', target_id)
        
        conditions_met = []
        conditions_unmet = []
        
        if hasattr(link, 'condition') and link.condition:
            if self._evaluate_condition(link.condition, context):
                conditions_met.append(self._describe_condition(link.condition))
            else:
                conditions_unmet.append(self._describe_condition(link.condition))
        
        if link_type == 'LOGICAL':
            type_map = {
                'REQUIRES': PossibilityType.REQUIRED_ITEM,
                'UNLOCKS': PossibilityType.UNLOCKABLE,
                'REVEALS': PossibilityType.REVEALABLE,
                'TRIGGERS': PossibilityType.TRIGGERABLE,
                'BLOCKS': PossibilityType.CONDITIONAL_BLOCKED,
            }
            
            if link_kind in type_map:
                possibilities.append(Possibility(
                    possibility_type=type_map[link_kind],
                    entity_id=target_id,
                    entity_name=target_name,
                    distance=depth,
                    path=path + [target_id],
                    via_link_type=link_type,
                    via_link_kind=link_kind,
                    conditions_met=conditions_met,
                    conditions_unmet=conditions_unmet,
                    hint_text=f"Something here could unlock {target_name}..." if conditions_unmet and link_kind == 'UNLOCKS' else None
                ))
        
        elif link_type == 'RELATIONAL' and conditions_unmet:
            if link_kind in ('NORTH_OF', 'SOUTH_OF', 'EAST_OF', 'WEST_OF', 'IN'):
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.BLOCKED_PATH,
                    entity_id=target_id,
                    entity_name=target_name,
                    distance=depth,
                    path=path + [target_id],
                    via_link_type=link_type,
                    via_link_kind=link_kind,
                    conditions_met=conditions_met,
                    conditions_unmet=conditions_unmet,
                    hint_text=f"The way to {target_name} is blocked."
                ))
        
        return possibilities
    
    # ─────────────────────────────────────────────────────────────────────────
    # Graph Access Helpers
    # ─────────────────────────────────────────────────────────────────────────
    
    def _resolve_entity(self, entity: Any) -> Tuple[str, Any]:
        if isinstance(entity, str):
            return (entity, self._get_entity(entity))
        return (self._get_entity_id(entity), entity)
    
    def _get_entity(self, entity_id: str) -> Any:
        if hasattr(self.graph, '_nodes'):
            return self.graph._nodes.get(entity_id)
        if hasattr(self.graph, 'nodes'):
            return self.graph.nodes.get(entity_id)
        if hasattr(self.graph, 'get_node'):
            return self.graph.get_node(entity_id)
        return None
    
    def _get_entity_id(self, entity: Any) -> str:
        if hasattr(entity, 'id'):
            return entity.id
        return str(entity)
    
    def _get_entity_type(self, entity: Any) -> str:
        if hasattr(entity, 'entity_type'):
            t = entity.entity_type
            return t.name if hasattr(t, 'name') else str(t)
        if hasattr(entity, 'node_type'):
            t = entity.node_type
            return t.name if hasattr(t, 'name') else str(t)
        return 'UNKNOWN'
    
    def _get_entity_state(self, entity: Any) -> str:
        if hasattr(entity, 'state'):
            s = entity.state
            return s.name if hasattr(s, 'name') else str(s)
        return 'NORMAL'
    
    def _is_takeable(self, entity: Any) -> bool:
        if hasattr(entity, 'tags'):
            return 'takeable' in entity.tags
        if hasattr(entity, 'properties'):
            return entity.properties.get('takeable', False)
        return False
    
    def _get_links(self, entity_id: str, layers: List[str] = None) -> List[Any]:
        links = []
        if hasattr(self.graph, 'get_outgoing_links'):
            links = self.graph.get_outgoing_links(entity_id)
        elif hasattr(self.graph, '_links'):
            links = [l for l in self.graph._links.values()
                    if getattr(l, 'source_id', None) == entity_id]
        
        if layers:
            links = [l for l in links if self._get_link_type(l) in layers]
        
        return links
    
    def _get_incoming_links(self, entity_id: str) -> List[Any]:
        if hasattr(self.graph, 'get_incoming_links'):
            return self.graph.get_incoming_links(entity_id)
        if hasattr(self.graph, '_links'):
            return [l for l in self.graph._links.values()
                   if getattr(l, 'target_id', None) == entity_id]
        return []
    
    def _get_link_type(self, link: Any) -> str:
        if hasattr(link, 'link_type'):
            t = link.link_type
            return t.name if hasattr(t, 'name') else str(t)
        if hasattr(link, 'edge_type'):
            t = link.edge_type
            return t.name if hasattr(t, 'name') else str(t)
        return 'UNKNOWN'
    
    def _get_link_kind(self, link: Any) -> str:
        if hasattr(link, 'kind'):
            k = link.kind
            if hasattr(k, 'value'):
                return k.value.upper() if isinstance(k.value, str) else k.name
            return k.name if hasattr(k, 'name') else str(k).upper()
        return 'UNKNOWN'
    
    def _get_link_target(self, link: Any, from_id: str) -> Optional[str]:
        if hasattr(link, 'target_id'):
            return link.target_id
        if hasattr(link, 'target'):
            return self._get_entity_id(link.target)
        return None
    
    def _get_link_source(self, link: Any) -> Optional[str]:
        if hasattr(link, 'source_id'):
            return link.source_id
        if hasattr(link, 'source'):
            return self._get_entity_id(link.source)
        return None
    
    def _evaluate_condition(self, condition: Any, context: Dict[str, Any]) -> bool:
        if condition is None:
            return True
        if hasattr(condition, 'evaluate'):
            try:
                return condition.evaluate(context)
            except Exception:
                return False
        if callable(condition):
            try:
                return condition(context)
            except Exception:
                return False
        return True
    
    def _describe_condition(self, condition: Any) -> str:
        if hasattr(condition, 'description') and condition.description:
            return condition.description
        return str(condition)
    
    def _check_requirement(self, required_id: str, context: Dict[str, Any]) -> bool:
        inventory = context.get('inventory', [])
        if required_id in inventory:
            return True
        flags = context.get('flags', {})
        if flags.get(f'has_{required_id}', False):
            return True
        return False
    
    def _kind_to_type(self, kind: str) -> PossibilityType:
        mapping = {
            'UNLOCKS': PossibilityType.UNLOCKABLE,
            'REVEALS': PossibilityType.REVEALABLE,
            'ENABLES': PossibilityType.UNLOCKABLE,
            'TRIGGERS': PossibilityType.TRIGGERABLE,
            'REQUIRES': PossibilityType.REQUIRED_ITEM,
            'BLOCKS': PossibilityType.CONDITIONAL_BLOCKED,
        }
        return mapping.get(kind.upper(), PossibilityType.CONDITIONAL_BLOCKED)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    PART 9: LOOKAHEAD WRAPPER INTEGRATION                  ║
# ║                                                                           ║
# ║  Connect lookahead to traversal for automatic possibility discovery.      ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class LookaheadWrapper:
    """
    Provides lookahead during traversal.
    
    Attach to a TraversalWrapper to get automatic lookahead at each step.
    Results are stored in wrapper.context.buffer for use by game logic.
    """
    
    def __init__(self, engine: LookaheadEngine, default_depth: int = 2):
        self.engine = engine
        self.default_depth = default_depth
        self._cache: Dict[str, LookaheadResult] = {}
    
    def on_step(self, wrapper, node):
        """Callback for TraversalWrapper.on_step()"""
        entity_id = self._get_id(node)
        
        if entity_id in self._cache:
            result = self._cache[entity_id]
        else:
            ctx = wrapper.context.__dict__ if hasattr(wrapper, 'context') else {}
            result = self.engine.lookahead(node, context=ctx, max_depth=self.default_depth)
            self._cache[entity_id] = result
        
        if hasattr(wrapper, 'context'):
            wrapper.context.buffer['lookahead'] = result
            wrapper.context.buffer['hints'] = result.get_hints()
            wrapper.context.buffer['near_misses'] = result.get_near_misses()
    
    def clear_cache(self):
        self._cache.clear()
    
    def _get_id(self, node: Any) -> str:
        if hasattr(node, 'id'):
            return node.id
        return str(node)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    PART 10: CONVENIENCE FUNCTIONS                         ║
# ║                                                                           ║
# ║  Simple entry points for common operations.                               ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def smart_iter(source: Iterable[T],
               context: TraversalContext = None,
               layer: str = None,
               layer_registry: LayerRegistry = None) -> TraversalWrapper[T]:
    """
    Wrap any iterable with a smart iterator.
    
    Main entry point for smart iteration.
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
    
    Main entry point for graph traversal.
    """
    return GraphTraverser(
        graph=graph,
        start_node=start,
        strategy=strategy,
        layer=layer,
        context=context,
        layer_registry=layer_registry
    )


def lookahead_from(graph: Any,
                   entity: Any,
                   context: Dict[str, Any] = None,
                   max_depth: int = 3) -> LookaheadResult:
    """
    One-shot lookahead from an entity.
    
    Main entry point for lookahead.
    """
    engine = LookaheadEngine(graph)
    return engine.lookahead(entity, context, max_depth)


def find_path_requirements(graph: Any,
                          from_entity: Any,
                          to_entity: Any,
                          context: Dict[str, Any] = None) -> List[Possibility]:
    """Find what's needed to get from one entity to another."""
    engine = LookaheadEngine(graph)
    result = engine.lookahead_for(from_entity, to_entity, context)
    
    if result is None:
        return []
    
    return [p for p in result.all_possibilities
            if p.possibility_type in (PossibilityType.REQUIRED_ITEM,
                                     PossibilityType.CONDITIONAL_BLOCKED,
                                     PossibilityType.LOCKED_DOOR)]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    PART 11: LAYER PRESETS                                 ║
# ║                                                                           ║
# ║  Pre-configured layers for common use cases.                              ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def create_infocom_layers() -> LayerRegistry:
    """
    Create layers matching my_infocom_entity_system.py link types.
    
    RELATIONAL: Spatial, containment, possession, social
    LOGICAL: Requires, unlocks, reveals, enables, blocks, triggers
    WILDCARD: Custom narrative connections
    """
    registry = LayerRegistry()
    
    # Relational layer - spatial navigation
    registry.register(LayerConfig(
        name="spatial",
        edge_types={
            "NORTH_OF", "SOUTH_OF", "EAST_OF", "WEST_OF",
            "IN", "ON", "UNDER", "NEAR"
        },
        priority=10
    ))
    
    # Containment layer - items in containers
    registry.register(LayerConfig(
        name="containment",
        edge_types={"CONTAINS", "PART_OF", "IN", "ON"},
        priority=5
    ))
    
    # Possession layer - who owns what
    registry.register(LayerConfig(
        name="possession",
        edge_types={"HELD_BY", "OWNED_BY"},
        priority=5
    ))
    
    # Social layer - character relationships
    registry.register(LayerConfig(
        name="social",
        edge_types={"KNOWS", "TRUSTS"},
        priority=5
    ))
    
    # Logic layer - game mechanics
    registry.register(LayerConfig(
        name="logical",
        edge_types={
            "REQUIRES", "UNLOCKS", "REVEALS",
            "ENABLES", "BLOCKS", "TRIGGERS"
        },
        priority=15
    ))
    
    # Combined relational (all spatial)
    registry.register(LayerConfig(
        name="relational",
        filter_fn=lambda edge, ctx: True,  # All RELATIONAL type edges
        priority=0
    ))
    
    return registry


def create_island_layers() -> LayerRegistry:
    """
    Create layers matching island_engine_design.md patterns.
    
    SPATIAL, TEMPORAL, CHARACTER, THEMATIC, NARRATIVE, CAUSAL
    """
    registry = LayerRegistry()
    
    registry.register(LayerConfig(
        name="spatial",
        edge_types={"adjacent_to", "walking_distance", "LEADS_TO", "CONTAINS"},
        priority=10
    ))
    
    registry.register(LayerConfig(
        name="temporal",
        edge_types={"next", "hours_later", "centuries_before", "during"},
        priority=5
    ))
    
    registry.register(LayerConfig(
        name="character",
        edge_types={"knows", "related_to", "possesses", "KNOWS", "OWNS"},
        priority=5
    ))
    
    registry.register(LayerConfig(
        name="thematic",
        edge_types={"represents", "symbolizes", "embodies", "contrasts_with"},
        priority=5
    ))
    
    registry.register(LayerConfig(
        name="narrative",
        edge_types={"contains", "occurs_at", "involves", "foreshadows"},
        priority=5
    ))
    
    registry.register(LayerConfig(
        name="causal",
        edge_types={"causes", "enables", "prevents", "requires"},
        priority=5
    ))
    
    # Auto-switch to danger layer in dangerous areas
    registry.register(LayerConfig(
        name="danger",
        edge_types={"LEADS_TO", "adjacent_to"},
        switch_condition=lambda ctx: (
            ctx.current_node and
            hasattr(ctx.current_node, 'tags') and
            'dangerous' in ctx.current_node.tags
        ),
        priority=20
    ))
    
    return registry


# ═══════════════════════════════════════════════════════════════════════════════
#                              MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Core types
    'Layer',
    'LayerConfig',
    'LayerRegistry',
    'TraversalContext',
    'EdgeResolver',
    
    # Wrappers
    'TraversalWrapper',
    'SmartList',
    'GraphTraverser',
    
    # Lookahead
    'PossibilityType',
    'Possibility',
    'LookaheadResult',
    'LookaheadEngine',
    'LookaheadWrapper',
    
    # Convenience functions
    'smart_iter',
    'traverse_graph',
    'lookahead_from',
    'find_path_requirements',
    
    # Presets
    'create_infocom_layers',
    'create_island_layers',
]


# ═══════════════════════════════════════════════════════════════════════════════
#                              DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Unified Traversal & Lookahead System")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Demo: Smart iteration
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Smart Iteration ---")
    
    locations = ["Beach", "Jungle", "Temple", "Cave"]
    wrapper = smart_iter(locations)
    wrapper.on_step(lambda w, item: print(f"  {item} (depth={w.depth})"))
    
    for loc in wrapper:
        pass
    
    print(f"  Visited: {wrapper.context.visited}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Demo: Layer switching
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Layer Switching ---")
    
    layers = create_island_layers()
    wrapper2 = smart_iter(locations, layer="spatial", layer_registry=layers)
    wrapper2.on_layer_switch(lambda w, old, new: print(f"  Layer: {old} → {new}"))
    
    iter(wrapper2)
    for i, loc in enumerate(wrapper2):
        print(f"  At {loc} (layer={wrapper2.layer})")
        if i == 1:
            wrapper2.switch_layer("thematic", push=True)
        if i == 2:
            wrapper2.pop_layer()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Demo: Lookahead with mock graph
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Lookahead Demo ---")
    
    class MockEntity:
        def __init__(self, id, name, entity_type, state='NORMAL', tags=None):
            self.id = id
            self.name = name
            self.entity_type = type('T', (), {'name': entity_type})()
            self.state = type('S', (), {'name': state})()
            self.tags = tags or set()
    
    class MockLink:
        def __init__(self, source_id, target_id, link_type, kind):
            self.id = f"{source_id}_{target_id}"
            self.source_id = source_id
            self.target_id = target_id
            self.link_type = type('T', (), {'name': link_type})()
            self.kind = type('K', (), {'name': kind, 'value': kind.lower()})()
            self.condition = None
    
    class MockGraph:
        def __init__(self):
            self._nodes = {}
            self._links = {}
        
        def add(self, e):
            self._nodes[e.id] = e
        
        def link(self, s, t, lt, k):
            l = MockLink(s, t, lt, k)
            self._links[l.id] = l
        
        def get_outgoing_links(self, eid):
            return [l for l in self._links.values() if l.source_id == eid]
        
        def get_incoming_links(self, eid):
            return [l for l in self._links.values() if l.target_id == eid]
    
    graph = MockGraph()
    graph.add(MockEntity('study', 'Study', 'LOCATION'))
    graph.add(MockEntity('hallway', 'Hallway', 'LOCATION'))
    graph.add(MockEntity('door', 'Locked Door', 'DOOR', state='LOCKED'))
    graph.add(MockEntity('key', 'Brass Key', 'ITEM', state='HIDDEN', tags={'takeable'}))
    
    graph.link('study', 'hallway', 'RELATIONAL', 'NORTH_OF')
    graph.link('hallway', 'door', 'RELATIONAL', 'IN')
    graph.link('key', 'door', 'LOGICAL', 'UNLOCKS')
    
    result = lookahead_from(graph, 'study', max_depth=3)
    print(result.summary())
    
    print("\n  Hints:")
    for hint in result.get_hints():
        print(f"    💡 {hint}")
    
    print("\n" + "=" * 70)
    print("System Ready")
    print("=" * 70)
