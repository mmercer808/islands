"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  MATTS TRAVERSAL - Layers, Context, Smart Iterator                            ║
║  Layer 2: Depends on primitives, signals                                      ║
║                                                                               ║
║  TraversalWrapper.__iter__() returns self - the wrapper IS the iterator.      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Dict, List, Set, Optional, Any, Callable, Iterator, Iterable,
    Generic, TypeVar, Tuple, Union
)
from collections import defaultdict
import time

from .primitives import Context, SignalType, T
from .signals import ObserverBus


# ═══════════════════════════════════════════════════════════════════════════════
#                              LAYERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LayerConfig:
    """Configuration for a traversal layer"""
    name: str
    edge_types: Set[str] = field(default_factory=set)
    edge_kinds: Set[str] = field(default_factory=set)
    filter_fn: Optional[Callable] = None
    auto_switch: Optional[Callable] = None  # (context) → bool
    priority: int = 0
    
    def matches(self, edge: Any, ctx: Any = None) -> bool:
        """Check if edge belongs to this layer"""
        if self.filter_fn:
            return self.filter_fn(edge, ctx)
        
        # Check edge type
        etype = getattr(edge, 'edge_type', None)
        if etype:
            name = etype.name if hasattr(etype, 'name') else str(etype)
            if self.edge_types and name not in self.edge_types:
                return False
        
        # Check kind
        kind = getattr(edge, 'kind', None)
        if not kind and hasattr(edge, 'data'):
            kind = edge.data.get('kind')
        if kind and self.edge_kinds and kind not in self.edge_kinds:
            return False
        
        return True


class LayerRegistry:
    """Central registry of layers"""
    
    def __init__(self):
        self._layers: Dict[str, LayerConfig] = {}
        self._active: str = "default"
    
    def add(self, cfg: LayerConfig) -> 'LayerRegistry':
        self._layers[cfg.name] = cfg
        return self
    
    def get(self, name: str) -> Optional[LayerConfig]:
        return self._layers.get(name)
    
    def activate(self, name: str) -> bool:
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
    
    def check_auto(self, ctx: Any) -> Optional[str]:
        """Check if any layer wants to auto-switch"""
        for cfg in sorted(self._layers.values(), key=lambda l: -l.priority):
            if cfg.auto_switch:
                try:
                    if cfg.auto_switch(ctx):
                        return cfg.name
                except:
                    pass
        return None
    
    def names(self) -> List[str]:
        return list(self._layers.keys())


def standard_layers() -> LayerRegistry:
    """Create registry with standard layers"""
    return (LayerRegistry()
        .add(LayerConfig(
            name="spatial",
            edge_kinds={"north_of", "south_of", "east_of", "west_of",
                       "in", "on", "under", "near", "contains", "leads_to"},
            priority=10
        ))
        .add(LayerConfig(
            name="logical",
            edge_kinds={"requires", "unlocks", "reveals", "enables", "blocks", "triggers"},
            priority=15
        ))
        .add(LayerConfig(
            name="character",
            edge_kinds={"knows", "trusts", "owns", "held_by", "wants"},
            priority=5
        ))
        .add(LayerConfig(
            name="temporal",
            edge_kinds={"before", "after", "during"},
            priority=5
        ))
        .add(LayerConfig(
            name="semantic",
            filter_fn=lambda e, c: getattr(e, 'layer', None) == 'semantic',
            priority=5
        ))
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                              CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TraversalContext(Context):
    """
    State that travels with the traversal wrapper.
    
    Compatible with GraphWalker's WalkerContext.
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
    
    # Game state
    flags: Dict[str, bool] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    inventory: List[str] = field(default_factory=list)
    memories: List[Dict] = field(default_factory=list)
    buffer: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)
    
    # Graph reference (can be swapped!)
    graph_ref: Any = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Node ID extraction
    # ─────────────────────────────────────────────────────────────────────────
    
    def node_id(self, node: Any) -> str:
        if hasattr(node, 'id'):
            return node.id
        return str(node)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Layer management
    # ─────────────────────────────────────────────────────────────────────────
    
    def push_layer(self, layer: str):
        self.layer_stack.append(self.current_layer)
        self.current_layer = layer
    
    def pop_layer(self) -> str:
        if self.layer_stack:
            self.current_layer = self.layer_stack.pop()
        return self.current_layer
    
    # ─────────────────────────────────────────────────────────────────────────
    # Visit tracking
    # ─────────────────────────────────────────────────────────────────────────
    
    def mark_visited(self, node: Any):
        self.visited.add(self.node_id(node))
    
    def is_visited(self, node: Any) -> bool:
        return self.node_id(node) in self.visited
    
    # ─────────────────────────────────────────────────────────────────────────
    # Flags & Counters (game state)
    # ─────────────────────────────────────────────────────────────────────────
    
    def set_flag(self, name: str, value: bool = True):
        self.flags[name] = value
    
    def get_flag(self, name: str, default: bool = False) -> bool:
        return self.flags.get(name, default)
    
    def increment(self, counter: str, amount: int = 1) -> int:
        self.counters[counter] = self.counters.get(counter, 0) + amount
        return self.counters[counter]
    
    def get_counter(self, counter: str, default: int = 0) -> int:
        return self.counters.get(counter, default)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Inventory
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_item(self, item_id: str):
        if item_id not in self.inventory:
            self.inventory.append(item_id)
    
    def remove_item(self, item_id: str) -> bool:
        if item_id in self.inventory:
            self.inventory.remove(item_id)
            return True
        return False
    
    def has_item(self, item_id: str) -> bool:
        return item_id in self.inventory
    
    # ─────────────────────────────────────────────────────────────────────────
    # Memory & Events
    # ─────────────────────────────────────────────────────────────────────────
    
    def remember(self, memory_type: str, content: Any, source: str = None):
        self.memories.append({
            'type': memory_type,
            'content': content,
            'source': source,
            'at': time.time()
        })
    
    def log_event(self, event_type: str, details: Dict = None):
        self.events.append({
            'type': event_type,
            'details': details or {},
            'at': time.time()
        })
    
    # ─────────────────────────────────────────────────────────────────────────
    # Serialization & Cloning
    # ─────────────────────────────────────────────────────────────────────────
    
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
        ctx.memories = [m.copy() for m in self.memories]
        ctx.buffer = self.buffer.copy()
        ctx.events = [e.copy() for e in self.events]
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
            'events': self.events,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TraversalContext':
        ctx = cls()
        ctx.visited = set(d.get('visited', []))
        ctx.depth = d.get('depth', 0)
        ctx.step_count = d.get('step_count', 0)
        ctx.current_layer = d.get('current_layer', 'default')
        ctx.layer_stack = d.get('layer_stack', [])
        ctx.flags = d.get('flags', {})
        ctx.counters = d.get('counters', {})
        ctx.inventory = d.get('inventory', [])
        ctx.memories = d.get('memories', [])
        ctx.events = d.get('events', [])
        return ctx


# ═══════════════════════════════════════════════════════════════════════════════
#                              WRAPPER (SMART ITERATOR)
# ═══════════════════════════════════════════════════════════════════════════════

class TraversalWrapper(Generic[T]):
    """
    Smart iterator that IS the iterator.
    
    __iter__() returns self - the wrapper travels with iteration,
    carrying context through the entire traversal.
    """
    
    def __init__(self,
                 source: Union[Iterable[T], Callable[[], Iterable[T]]] = None,
                 context: TraversalContext = None,
                 layers: LayerRegistry = None,
                 bus: ObserverBus = None):
        
        self._source = source
        self._iterator: Optional[Iterator[T]] = None
        
        self.context = context or TraversalContext()
        self.layers = layers or LayerRegistry()
        self._bus = bus
        
        self._callbacks: List[Callable[['TraversalWrapper', T], Any]] = []
        self._on_enter: List[Callable[['TraversalWrapper'], Any]] = []
        self._on_exit: List[Callable[['TraversalWrapper'], Any]] = []
        
        self._current: Optional[T] = None
        self._exhausted = False
    
    # ─────────────────────────────────────────────────────────────────────────
    # Iterator Protocol
    # ─────────────────────────────────────────────────────────────────────────
    
    def __iter__(self) -> 'TraversalWrapper[T]':
        """Returns SELF - wrapper travels with iteration"""
        if callable(self._source):
            self._source = self._source()
        if self._source is not None:
            self._iterator = iter(self._source)
        self._exhausted = False
        
        for cb in self._on_enter:
            try: cb(self)
            except: pass
        
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
                    source_id=self.context.node_id(item),
                    data={'depth': self.context.depth}
                )
            
            # Auto layer switch
            new_layer = self.layers.check_auto(self.context)
            if new_layer and new_layer != self.context.current_layer:
                self.switch_layer(new_layer)
            
            return item
            
        except StopIteration:
            self._exhausted = True
            for cb in self._on_exit:
                try: cb(self)
                except: pass
            raise
    
    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def current(self) -> Optional[T]:
        return self._current
    
    @property
    def previous(self) -> Optional[T]:
        return self.context.previous_node
    
    @property
    def depth(self) -> int:
        return self.context.depth
    
    @property
    def layer(self) -> str:
        return self.context.current_layer
    
    # ─────────────────────────────────────────────────────────────────────────
    # Layer switching
    # ─────────────────────────────────────────────────────────────────────────
    
    def switch_layer(self, name: str, push: bool = False):
        old = self.context.current_layer
        if push:
            self.context.push_layer(name)
        else:
            self.context.current_layer = name
        self.layers.activate(name)
        
        if self._bus:
            self._bus.emit_type(SignalType.LAYER_SWITCH, 
                               data={'old_layer': old, 'new_layer': name})
    
    def pop_layer(self) -> str:
        old = self.context.current_layer
        new = self.context.pop_layer()
        self.layers.activate(new)
        if self._bus:
            self._bus.emit_type(SignalType.LAYER_SWITCH,
                               data={'old_layer': old, 'new_layer': new})
        return new
    
    # ─────────────────────────────────────────────────────────────────────────
    # Graph operations
    # ─────────────────────────────────────────────────────────────────────────
    
    def swap_graph(self, new_graph: Any, reset: bool = False):
        """Hotswap underlying graph"""
        self.context.graph_ref = new_graph
        if reset:
            self._current = None
            self.context.path = []
            self.context.visited = set()
        if self._bus:
            self._bus.emit_type(SignalType.GRAPH_SWAP, data={'reset': reset})
    
    def swap_source(self, source: Union[Iterable[T], Callable]):
        """Replace iteration source mid-traversal"""
        self._source = source() if callable(source) else source
        self._iterator = iter(self._source) if self._source else None
        self._exhausted = False
    
    def jump_to(self, node: T):
        """Direct jump to node (teleport)"""
        self.context.previous_node = self._current
        self._current = node
        self.context.current_node = node
        self.context.path.append(node)
        self.context.mark_visited(node)
        self.context.step_count += 1
        
        for cb in self._callbacks:
            try: cb(self, node)
            except: pass
    
    # ─────────────────────────────────────────────────────────────────────────
    # Branching
    # ─────────────────────────────────────────────────────────────────────────
    
    def branch(self) -> 'TraversalWrapper[T]':
        """Clone for parallel exploration"""
        branched = TraversalWrapper(
            context=self.context.clone(),
            layers=self.layers,
            bus=self._bus
        )
        branched._current = self._current
        return branched
    
    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────
    
    def on_step(self, cb: Callable[['TraversalWrapper', T], Any]) -> 'TraversalWrapper[T]':
        self._callbacks.append(cb)
        return self
    
    def on_enter(self, cb: Callable) -> 'TraversalWrapper[T]':
        self._on_enter.append(cb)
        return self
    
    def on_exit(self, cb: Callable) -> 'TraversalWrapper[T]':
        self._on_exit.append(cb)
        return self
    
    # ─────────────────────────────────────────────────────────────────────────
    # Consumption helpers
    # ─────────────────────────────────────────────────────────────────────────
    
    def collect(self) -> List[T]:
        return list(self)
    
    def take(self, n: int) -> List[T]:
        result = []
        for item in self:
            result.append(item)
            if len(result) >= n:
                break
        return result
    
    def skip(self, n: int) -> 'TraversalWrapper[T]':
        for _ in range(n):
            try: next(self)
            except StopIteration: break
        return self
    
    def first(self) -> Optional[T]:
        try:
            return next(iter(self))
        except StopIteration:
            return None


# ═══════════════════════════════════════════════════════════════════════════════
#                              FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def smart_iter(source: Iterable[T],
               context: TraversalContext = None,
               layers: LayerRegistry = None,
               bus: ObserverBus = None) -> TraversalWrapper[T]:
    """Create smart iterator"""
    return TraversalWrapper(source, context, layers, bus)


# ═══════════════════════════════════════════════════════════════════════════════
#                              EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'LayerConfig', 'LayerRegistry', 'standard_layers',
    'TraversalContext',
    'TraversalWrapper', 'smart_iter',
]
