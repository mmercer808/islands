#!/usr/bin/env python3
"""
Decoupled Pipeline Designs
==========================

This module presents alternative architectures where each core system from the
matts library is completely decoupled and can operate independently. Each pipeline
can be used standalone or composed with others via thin adapters.

Design Philosophy:
- Zero hard dependencies between pipelines
- Protocol-based interfaces for composition
- Each pipeline is self-contained and testable
- Adapters bridge pipelines without coupling them

The 7 Decoupled Pipelines:
1. SignalPipeline - Event dispatch and observer management
2. ContextPipeline - State management with serialization
3. GeneratorPipeline - Yield/iteration composition
4. GraphPipeline - Relationship and traversal operations
5. NarrativePipeline - Story/chain execution
6. IteratorPipeline - Persistent context windows
7. CodePipeline - Runtime code management

Each pipeline follows the same structural pattern:
- Protocol definition (what it expects)
- Core implementation (standalone)
- Adapter factory (for composition)
"""

from __future__ import annotations
import uuid
import time
import asyncio
import threading
import weakref
from abc import ABC, abstractmethod
from typing import (
    Dict, Any, List, Optional, Callable, Union, TypeVar, Generic,
    Protocol, AsyncIterator, Iterator, Awaitable, Set, Tuple
)
from dataclasses import dataclass, field
from collections import deque, OrderedDict
from enum import Enum, auto
from datetime import datetime
import copy
import queue

T = TypeVar('T')
TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')


# =============================================================================
# SHARED PROTOCOLS (The "Thin Contracts" between pipelines)
# =============================================================================

class Identifiable(Protocol):
    """Protocol for anything with an ID."""
    @property
    def id(self) -> str: ...


class Serializable(Protocol):
    """Protocol for serialization."""
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable': ...


class Observable(Protocol):
    """Protocol for observer pattern."""
    def add_observer(self, observer: 'ObserverProtocol') -> str: ...
    def remove_observer(self, observer_id: str) -> bool: ...
    def notify(self, event: Any) -> None: ...


class ObserverProtocol(Protocol):
    """Protocol for observers."""
    @property
    def observer_id(self) -> str: ...
    async def on_event(self, event: Any) -> Any: ...


class Yieldable(Protocol[T]):
    """Protocol for generators/iterators."""
    def __iter__(self) -> Iterator[T]: ...
    def __next__(self) -> T: ...


class Traversable(Protocol):
    """Protocol for graph traversal."""
    def get_neighbors(self, node_id: str) -> List[str]: ...
    def get_node(self, node_id: str) -> Optional[Any]: ...


# =============================================================================
# PIPELINE 1: SIGNAL PIPELINE
# =============================================================================
# Completely standalone event dispatch system

class SignalPriority(Enum):
    """Priority levels for signals."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class Signal:
    """A standalone signal/event container."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: str = "generic"
    source: str = ""
    target: Optional[str] = None
    priority: SignalPriority = SignalPriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal_id': self.signal_id,
            'signal_type': self.signal_type,
            'source': self.source,
            'target': self.target,
            'priority': self.priority.value,
            'payload': self.payload,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        data = data.copy()
        data['priority'] = SignalPriority(data['priority'])
        return cls(**data)


class SignalObserver(ABC):
    """Abstract observer for signal pipeline."""
    
    def __init__(self, observer_id: str = None):
        self._observer_id = observer_id or f"obs_{uuid.uuid4()}"
        self.filters: Set[str] = set()  # Signal types to handle
        self.active = True
    
    @property
    def observer_id(self) -> str:
        return self._observer_id
    
    def can_handle(self, signal: Signal) -> bool:
        """Check if this observer handles the signal type."""
        if not self.active:
            return False
        if not self.filters:  # Empty = handle all
            return True
        return signal.signal_type in self.filters
    
    @abstractmethod
    async def on_event(self, signal: Signal) -> Any:
        """Handle the signal. Override in subclasses."""
        pass


class SignalPipeline:
    """
    Standalone signal dispatch pipeline.
    
    No dependencies on other systems. Can be used alone for pub/sub.
    
    Usage:
        pipeline = SignalPipeline()
        await pipeline.start()
        
        # Register observers
        await pipeline.register(MyObserver())
        
        # Emit signals
        await pipeline.emit(Signal(signal_type="user_action", payload={"key": "value"}))
    """
    
    def __init__(self, pipeline_id: str = None):
        self.pipeline_id = pipeline_id or f"signal_pipe_{uuid.uuid4()}"
        self._observers: Dict[str, SignalObserver] = {}
        self._priority_queues: Dict[SignalPriority, asyncio.Queue] = {
            p: asyncio.Queue() for p in SignalPriority
        }
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Stats
        self.signals_emitted = 0
        self.signals_processed = 0
    
    async def start(self):
        """Start the signal processor."""
        self._running = True
        self._processor_task = asyncio.create_task(self._process_signals())
    
    async def stop(self):
        """Stop the signal processor."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
    
    async def register(self, observer: SignalObserver) -> str:
        """Register an observer."""
        async with self._lock:
            self._observers[observer.observer_id] = observer
        return observer.observer_id
    
    async def unregister(self, observer_id: str) -> bool:
        """Unregister an observer."""
        async with self._lock:
            if observer_id in self._observers:
                del self._observers[observer_id]
                return True
        return False
    
    async def emit(self, signal: Signal) -> int:
        """Emit a signal to the pipeline. Returns number of notified observers."""
        self.signals_emitted += 1
        await self._priority_queues[signal.priority].put(signal)
        
        # For immediate dispatch (bypasses queue)
        notified = 0
        async with self._lock:
            observers = list(self._observers.values())
        
        for observer in observers:
            if observer.can_handle(signal):
                try:
                    await observer.on_event(signal)
                    notified += 1
                except Exception as e:
                    print(f"Observer {observer.observer_id} error: {e}")
        
        return notified
    
    async def _process_signals(self):
        """Background signal processor (priority-based)."""
        while self._running:
            try:
                # Process in priority order
                for priority in SignalPriority:
                    queue = self._priority_queues[priority]
                    while not queue.empty():
                        signal = await queue.get()
                        self.signals_processed += 1
                
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Signal processor error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'pipeline_id': self.pipeline_id,
            'running': self._running,
            'observer_count': len(self._observers),
            'signals_emitted': self.signals_emitted,
            'signals_processed': self.signals_processed
        }


# =============================================================================
# PIPELINE 2: CONTEXT PIPELINE
# =============================================================================
# Standalone state management with serialization

class ContextState(Enum):
    """States for context lifecycle."""
    CREATED = auto()
    ACTIVE = auto()
    SUSPENDED = auto()
    SERIALIZED = auto()
    DESTROYED = auto()


@dataclass
class ContextSnapshot:
    """Immutable snapshot of context state."""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    parent_snapshot_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'snapshot_id': self.snapshot_id,
            'context_id': self.context_id,
            'data': copy.deepcopy(self.data),
            'timestamp': self.timestamp,
            'parent_snapshot_id': self.parent_snapshot_id
        }


class Context:
    """
    A standalone context container with history tracking.
    
    No dependencies. Pure state management.
    """
    
    def __init__(self, context_id: str = None):
        self.context_id = context_id or f"ctx_{uuid.uuid4()}"
        self._data: Dict[str, Any] = {}
        self._state = ContextState.CREATED
        self._snapshots: List[ContextSnapshot] = []
        self._change_callbacks: List[Callable] = []
        self.created_at = time.time()
    
    @property
    def state(self) -> ContextState:
        return self._state
    
    @property
    def data(self) -> Dict[str, Any]:
        return self._data.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context."""
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any, create_snapshot: bool = False):
        """Set value in context."""
        old_value = self._data.get(key)
        self._data[key] = value
        
        if create_snapshot:
            self.snapshot()
        
        # Notify change callbacks
        for callback in self._change_callbacks:
            try:
                callback(key, old_value, value)
            except Exception:
                pass
    
    def update(self, updates: Dict[str, Any], create_snapshot: bool = False):
        """Batch update context data."""
        for key, value in updates.items():
            self._data[key] = value
        
        if create_snapshot:
            self.snapshot()
    
    def snapshot(self) -> ContextSnapshot:
        """Create immutable snapshot of current state."""
        parent_id = self._snapshots[-1].snapshot_id if self._snapshots else None
        snap = ContextSnapshot(
            context_id=self.context_id,
            data=copy.deepcopy(self._data),
            parent_snapshot_id=parent_id
        )
        self._snapshots.append(snap)
        return snap
    
    def restore(self, snapshot_id: str) -> bool:
        """Restore context to a previous snapshot."""
        for snap in self._snapshots:
            if snap.snapshot_id == snapshot_id:
                self._data = copy.deepcopy(snap.data)
                return True
        return False
    
    def on_change(self, callback: Callable[[str, Any, Any], None]):
        """Register change callback: (key, old_value, new_value)."""
        self._change_callbacks.append(callback)
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize entire context for transmission."""
        return {
            'context_id': self.context_id,
            'state': self._state.name,
            'data': copy.deepcopy(self._data),
            'created_at': self.created_at,
            'snapshots': [s.to_dict() for s in self._snapshots]
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'Context':
        """Deserialize context from data."""
        ctx = cls(data['context_id'])
        ctx._data = copy.deepcopy(data['data'])
        ctx._state = ContextState[data['state']]
        ctx.created_at = data['created_at']
        # Restore snapshots
        for snap_data in data.get('snapshots', []):
            snap = ContextSnapshot(**snap_data)
            ctx._snapshots.append(snap)
        return ctx


class ContextPipeline:
    """
    Standalone context management pipeline.
    
    Manages multiple contexts with lifecycle, serialization, and lookup.
    """
    
    def __init__(self):
        self._contexts: Dict[str, Context] = {}
        self._lock = threading.Lock()
    
    def create(self, context_id: str = None) -> Context:
        """Create and register a new context."""
        ctx = Context(context_id)
        with self._lock:
            self._contexts[ctx.context_id] = ctx
        return ctx
    
    def get(self, context_id: str) -> Optional[Context]:
        """Get context by ID."""
        return self._contexts.get(context_id)
    
    def destroy(self, context_id: str) -> bool:
        """Destroy and remove context."""
        with self._lock:
            if context_id in self._contexts:
                ctx = self._contexts[context_id]
                ctx._state = ContextState.DESTROYED
                del self._contexts[context_id]
                return True
        return False
    
    def list_contexts(self) -> List[str]:
        """List all context IDs."""
        return list(self._contexts.keys())
    
    def serialize_all(self) -> Dict[str, Any]:
        """Serialize all contexts."""
        return {
            cid: ctx.serialize() 
            for cid, ctx in self._contexts.items()
        }
    
    def restore_all(self, data: Dict[str, Any]):
        """Restore all contexts from serialized data."""
        for cid, ctx_data in data.items():
            ctx = Context.deserialize(ctx_data)
            self._contexts[cid] = ctx


# =============================================================================
# PIPELINE 3: GENERATOR PIPELINE
# =============================================================================
# Standalone generator composition system

class CompositionPattern(Enum):
    """Patterns for generator composition."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    BRANCH_MERGE = "branch_merge"
    CONDITIONAL = "conditional"


@dataclass
class GeneratorResult:
    """Result from a generator step."""
    generator_id: str
    step: int
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class GeneratorWrapper(Generic[T]):
    """
    Wrapper that makes any generator composable.
    
    The wrapper IS the iterator - no separate iterator class needed.
    """
    
    def __init__(self, 
                 generator_func: Callable[..., Iterator[T]],
                 generator_id: str = None,
                 config: Dict[str, Any] = None):
        self.generator_id = generator_id or f"gen_{uuid.uuid4()}"
        self._generator_func = generator_func
        self._config = config or {}
        self._generator: Optional[Iterator[T]] = None
        self._step = 0
        self._exhausted = False
    
    def start(self, *args, **kwargs) -> 'GeneratorWrapper[T]':
        """Initialize the generator."""
        merged_kwargs = {**self._config, **kwargs}
        self._generator = self._generator_func(*args, **merged_kwargs)
        self._step = 0
        self._exhausted = False
        return self
    
    def __iter__(self) -> Iterator[GeneratorResult]:
        return self
    
    def __next__(self) -> GeneratorResult:
        if self._exhausted or self._generator is None:
            raise StopIteration
        
        try:
            value = next(self._generator)
            self._step += 1
            return GeneratorResult(
                generator_id=self.generator_id,
                step=self._step,
                value=value
            )
        except StopIteration:
            self._exhausted = True
            raise
    
    def send(self, value: Any) -> GeneratorResult:
        """Send value into generator (for coroutine-style generators)."""
        if self._generator is None:
            raise RuntimeError("Generator not started")
        
        try:
            result = self._generator.send(value)
            self._step += 1
            return GeneratorResult(
                generator_id=self.generator_id,
                step=self._step,
                value=result
            )
        except StopIteration:
            self._exhausted = True
            raise


class GeneratorPipeline:
    """
    Standalone generator composition pipeline.
    
    Composes multiple generators using various patterns.
    """
    
    def __init__(self):
        self._factories: Dict[str, Callable] = {}
        self._compositions: Dict[str, List[str]] = {}
    
    def register_factory(self, name: str, factory: Callable):
        """Register a generator factory."""
        self._factories[name] = factory
    
    def create_generator(self, name: str, config: Dict[str, Any] = None) -> GeneratorWrapper:
        """Create generator from registered factory."""
        if name not in self._factories:
            raise ValueError(f"Factory '{name}' not registered")
        
        return GeneratorWrapper(
            generator_func=self._factories[name],
            generator_id=f"{name}_{uuid.uuid4()}",
            config=config
        )
    
    def compose_sequential(self, 
                          generators: List[GeneratorWrapper],
                          input_data: Any = None) -> Iterator[GeneratorResult]:
        """
        Sequential composition: run generators one after another.
        Output of one becomes input for next.
        """
        current_data = input_data
        
        for gen in generators:
            gen.start(current_data)
            for result in gen:
                yield result
                current_data = result.value
    
    def compose_pipeline(self,
                        generators: List[GeneratorWrapper],
                        input_stream: Iterator) -> Iterator[GeneratorResult]:
        """
        Pipeline composition: each item flows through all generators.
        """
        for item in input_stream:
            current = item
            for gen in generators:
                gen.start(current)
                try:
                    result = next(gen)
                    current = result.value
                    yield result
                except StopIteration:
                    break
    
    def compose_parallel(self,
                        generators: List[GeneratorWrapper],
                        input_data: Any = None) -> List[List[GeneratorResult]]:
        """
        Parallel composition: run all generators simultaneously.
        Returns list of results from each generator.
        """
        results = []
        for gen in generators:
            gen.start(input_data)
            gen_results = list(gen)
            results.append(gen_results)
        return results
    
    def compose_branch_merge(self,
                            branches: List[GeneratorWrapper],
                            input_data: Any,
                            merge_func: Callable[[List[Any]], Any]) -> Any:
        """
        Branch-merge: run branches in parallel, merge results.
        """
        branch_results = []
        for gen in branches:
            gen.start(input_data)
            results = [r.value for r in gen]
            branch_results.append(results[-1] if results else None)
        
        return merge_func(branch_results)


# =============================================================================
# PIPELINE 4: GRAPH PIPELINE  
# =============================================================================
# Standalone graph traversal system

@dataclass
class GraphNode:
    """A node in the graph."""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: str = "generic"
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'data': self.data
        }


@dataclass  
class GraphEdge:
    """An edge connecting two nodes."""
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    edge_type: str = "generic"
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


class GraphTraverser:
    """
    Iterator-based graph traverser.
    
    The traverser IS the iterator - wraps become their own iterators.
    """
    
    def __init__(self, graph: 'GraphPipeline', start_node: str, 
                 mode: str = "bfs", max_depth: int = -1):
        self._graph = graph
        self._start = start_node
        self._mode = mode
        self._max_depth = max_depth
        self._visited: Set[str] = set()
        self._queue: deque = deque()
        self._current_depth = 0
    
    def __iter__(self) -> Iterator[Tuple[GraphNode, int]]:
        """Return self as iterator."""
        self._visited = set()
        self._queue = deque([(self._start, 0)])
        return self
    
    def __next__(self) -> Tuple[GraphNode, int]:
        """Get next node in traversal."""
        while self._queue:
            if self._mode == "bfs":
                node_id, depth = self._queue.popleft()
            else:  # dfs
                node_id, depth = self._queue.pop()
            
            if node_id in self._visited:
                continue
            
            if self._max_depth >= 0 and depth > self._max_depth:
                continue
            
            self._visited.add(node_id)
            node = self._graph.get_node(node_id)
            
            if node is None:
                continue
            
            # Add neighbors to queue
            for neighbor_id in self._graph.get_neighbors(node_id):
                if neighbor_id not in self._visited:
                    self._queue.append((neighbor_id, depth + 1))
            
            return (node, depth)
        
        raise StopIteration


class GraphPipeline:
    """
    Standalone graph operations pipeline.
    
    Manages nodes, edges, and traversal.
    """
    
    def __init__(self):
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, GraphEdge] = {}
        self._adjacency: Dict[str, List[str]] = {}
        self._reverse_adjacency: Dict[str, List[str]] = {}
    
    def add_node(self, node: GraphNode) -> str:
        """Add node to graph."""
        self._nodes[node.node_id] = node
        if node.node_id not in self._adjacency:
            self._adjacency[node.node_id] = []
        if node.node_id not in self._reverse_adjacency:
            self._reverse_adjacency[node.node_id] = []
        return node.node_id
    
    def add_edge(self, edge: GraphEdge) -> str:
        """Add edge to graph."""
        self._edges[edge.edge_id] = edge
        if edge.source_id not in self._adjacency:
            self._adjacency[edge.source_id] = []
        self._adjacency[edge.source_id].append(edge.target_id)
        
        if edge.target_id not in self._reverse_adjacency:
            self._reverse_adjacency[edge.target_id] = []
        self._reverse_adjacency[edge.target_id].append(edge.source_id)
        
        return edge.edge_id
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        return self._nodes.get(node_id)
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighbor node IDs."""
        return self._adjacency.get(node_id, [])
    
    def get_predecessors(self, node_id: str) -> List[str]:
        """Get predecessor node IDs."""
        return self._reverse_adjacency.get(node_id, [])
    
    def traverse(self, start_node: str, mode: str = "bfs", 
                max_depth: int = -1) -> GraphTraverser:
        """Create traverser for graph exploration."""
        return GraphTraverser(self, start_node, mode, max_depth)
    
    def find_path(self, from_node: str, to_node: str) -> Optional[List[str]]:
        """Find path between two nodes using BFS."""
        if from_node not in self._nodes or to_node not in self._nodes:
            return None
        
        visited = set()
        queue = deque([(from_node, [from_node])])
        
        while queue:
            current, path = queue.popleft()
            
            if current == to_node:
                return path
            
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_subgraph(self, node_ids: Set[str]) -> 'GraphPipeline':
        """Extract subgraph containing specified nodes."""
        subgraph = GraphPipeline()
        
        for nid in node_ids:
            if nid in self._nodes:
                subgraph.add_node(self._nodes[nid])
        
        for edge in self._edges.values():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                subgraph.add_edge(edge)
        
        return subgraph


# =============================================================================
# PIPELINE 5: NARRATIVE PIPELINE
# =============================================================================
# Standalone narrative/story chain system

class ChainEventType(Enum):
    """Types of narrative events."""
    START = auto()
    PROGRESS = auto()
    COMPLETE = auto()
    BRANCH = auto()
    CHECKPOINT = auto()


@dataclass
class ChainEvent:
    """Event in a narrative chain."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: ChainEventType = ChainEventType.PROGRESS
    chain_id: str = ""
    link_index: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChainLink:
    """A single link in a narrative chain."""
    link_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    handler: Optional[Callable] = None
    data: Dict[str, Any] = field(default_factory=dict)
    branch_conditions: List[Tuple[Callable, str]] = field(default_factory=list)  # (condition, target_chain_id)
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute this link's handler."""
        if self.handler:
            return self.handler(context, self.data)
        return None
    
    def check_branches(self, context: Dict[str, Any]) -> List[str]:
        """Check which branches should be triggered."""
        triggered = []
        for condition, target in self.branch_conditions:
            try:
                if condition(context):
                    triggered.append(target)
            except Exception:
                pass
        return triggered


class NarrativeChain:
    """
    A chain of narrative links that can be iterated.
    
    The chain IS its own iterator.
    """
    
    def __init__(self, chain_id: str = None, name: str = ""):
        self.chain_id = chain_id or f"chain_{uuid.uuid4()}"
        self.name = name
        self._links: List[ChainLink] = []
        self._current_index = 0
        self._context: Dict[str, Any] = {}
        self._is_active = False
    
    def add_link(self, link: ChainLink) -> int:
        """Add link to chain, returns index."""
        self._links.append(link)
        return len(self._links) - 1
    
    def start(self, initial_context: Dict[str, Any] = None):
        """Start the chain iteration."""
        self._current_index = 0
        self._context = initial_context or {}
        self._is_active = True
    
    def __iter__(self) -> Iterator[ChainEvent]:
        return self
    
    def __next__(self) -> ChainEvent:
        if not self._is_active or self._current_index >= len(self._links):
            self._is_active = False
            raise StopIteration
        
        link = self._links[self._current_index]
        
        # Execute the link
        result = link.execute(self._context)
        if isinstance(result, dict):
            self._context.update(result)
        
        event = ChainEvent(
            event_type=ChainEventType.PROGRESS,
            chain_id=self.chain_id,
            link_index=self._current_index,
            data={
                'link_name': link.name,
                'result': result,
                'branches_triggered': link.check_branches(self._context)
            }
        )
        
        self._current_index += 1
        return event
    
    def reset(self):
        """Reset chain to beginning."""
        self._current_index = 0
        self._is_active = False


class NarrativePipeline:
    """
    Standalone narrative chain management pipeline.
    
    Manages multiple chains with event-driven progression.
    """
    
    def __init__(self):
        self._chains: Dict[str, NarrativeChain] = {}
        self._event_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._active_chains: Set[str] = set()
        self._global_context: Dict[str, Any] = {}
    
    def create_chain(self, name: str = "") -> NarrativeChain:
        """Create and register a new chain."""
        chain = NarrativeChain(name=name)
        self._chains[chain.chain_id] = chain
        return chain
    
    def get_chain(self, chain_id: str) -> Optional[NarrativeChain]:
        """Get chain by ID."""
        return self._chains.get(chain_id)
    
    def start_chain(self, chain_id: str, context: Dict[str, Any] = None):
        """Start a chain's execution."""
        chain = self._chains.get(chain_id)
        if chain:
            merged_context = {**self._global_context, **(context or {})}
            chain.start(merged_context)
            self._active_chains.add(chain_id)
    
    def step_chain(self, chain_id: str) -> Optional[ChainEvent]:
        """Step one link in a chain."""
        chain = self._chains.get(chain_id)
        if chain and chain._is_active:
            try:
                return next(chain)
            except StopIteration:
                self._active_chains.discard(chain_id)
        return None
    
    def run_chain(self, chain_id: str, context: Dict[str, Any] = None) -> List[ChainEvent]:
        """Run entire chain, collecting all events."""
        events = []
        self.start_chain(chain_id, context)
        
        while chain_id in self._active_chains:
            event = self.step_chain(chain_id)
            if event:
                events.append(event)
                # Handle branch triggers
                for target_chain in event.data.get('branches_triggered', []):
                    if target_chain in self._chains:
                        self.start_chain(target_chain, self._global_context)
        
        return events
    
    def set_global_context(self, key: str, value: Any):
        """Set global context available to all chains."""
        self._global_context[key] = value


# =============================================================================
# PIPELINE 6: ITERATOR PIPELINE
# =============================================================================
# Standalone persistent context window iteration

@dataclass
class Operation:
    """A tracked operation in the context window."""
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: str = "generic"
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    previous_id: Optional[str] = None
    next_id: Optional[str] = None


class ContextWindowIterator:
    """
    Iterator over a sliding window of operations.
    
    The window IS the iterator.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._operations: OrderedDict[str, Operation] = OrderedDict()
        self._timeline: deque = deque(maxlen=max_size)
        self._iter_index = 0
        self._iter_keys: List[str] = []
    
    def add(self, operation: Operation) -> str:
        """Add operation to window."""
        # Link to previous
        if self._timeline:
            last_id = self._timeline[-1]
            if last_id in self._operations:
                self._operations[last_id].next_id = operation.operation_id
                operation.previous_id = last_id
        
        # Add to window
        self._operations[operation.operation_id] = operation
        self._timeline.append(operation.operation_id)
        
        # Evict if over size
        while len(self._operations) > self.max_size:
            oldest_id = self._timeline.popleft()
            del self._operations[oldest_id]
        
        return operation.operation_id
    
    def get(self, operation_id: str) -> Optional[Operation]:
        """Get operation by ID."""
        return self._operations.get(operation_id)
    
    def get_recent(self, count: int = 10) -> List[Operation]:
        """Get most recent operations."""
        recent_ids = list(self._timeline)[-count:]
        return [self._operations[oid] for oid in recent_ids if oid in self._operations]
    
    def __iter__(self) -> Iterator[Operation]:
        """Iterate over all operations in window."""
        self._iter_keys = list(self._operations.keys())
        self._iter_index = 0
        return self
    
    def __next__(self) -> Operation:
        if self._iter_index >= len(self._iter_keys):
            raise StopIteration
        
        key = self._iter_keys[self._iter_index]
        self._iter_index += 1
        return self._operations[key]
    
    def __len__(self) -> int:
        return len(self._operations)


class IteratorPipeline:
    """
    Standalone iterator/context window pipeline.
    
    Manages persistent iteration over operation history.
    """
    
    def __init__(self, window_size: int = 1000):
        self._windows: Dict[str, ContextWindowIterator] = {}
        self._default_window_size = window_size
    
    def create_window(self, window_id: str = None, size: int = None) -> ContextWindowIterator:
        """Create a new context window."""
        wid = window_id or f"window_{uuid.uuid4()}"
        window = ContextWindowIterator(size or self._default_window_size)
        self._windows[wid] = window
        return window
    
    def get_window(self, window_id: str) -> Optional[ContextWindowIterator]:
        """Get window by ID."""
        return self._windows.get(window_id)
    
    def record_operation(self, window_id: str, 
                        operation_type: str,
                        data: Dict[str, Any] = None,
                        result: Any = None) -> Optional[str]:
        """Record an operation to a window."""
        window = self._windows.get(window_id)
        if window:
            op = Operation(
                operation_type=operation_type,
                data=data or {},
                result=result
            )
            return window.add(op)
        return None
    
    def get_operation_chain(self, window_id: str, 
                           operation_id: str,
                           direction: str = "both") -> List[Operation]:
        """Get chain of linked operations."""
        window = self._windows.get(window_id)
        if not window:
            return []
        
        op = window.get(operation_id)
        if not op:
            return []
        
        chain = []
        
        # Backward
        if direction in ["both", "backward"]:
            backward = []
            current = op
            while current.previous_id:
                prev = window.get(current.previous_id)
                if prev:
                    backward.append(prev)
                    current = prev
                else:
                    break
            chain.extend(reversed(backward))
        
        # Current
        chain.append(op)
        
        # Forward
        if direction in ["both", "forward"]:
            current = op
            while current.next_id:
                next_op = window.get(current.next_id)
                if next_op:
                    chain.append(next_op)
                    current = next_op
                else:
                    break
        
        return chain


# =============================================================================
# PIPELINE 7: CODE PIPELINE
# =============================================================================
# Standalone runtime code management

@dataclass
class CodeBlob:
    """Serialized code container."""
    blob_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    source_code: str = ""
    code_type: str = "function"  # function, class, decorator
    required_imports: List[str] = field(default_factory=list)
    is_trusted: bool = False
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'blob_id': self.blob_id,
            'name': self.name,
            'source_code': self.source_code,
            'code_type': self.code_type,
            'required_imports': self.required_imports,
            'is_trusted': self.is_trusted,
            'created_at': self.created_at
        }


class CodeExecutor:
    """
    Safe(r) code executor with namespace isolation.
    
    WARNING: Still executes arbitrary code. Use with caution.
    """
    
    def __init__(self, allow_imports: List[str] = None):
        self._allow_imports = set(allow_imports or [])
        self._namespace: Dict[str, Any] = {}
        self._executed_blobs: Set[str] = set()
    
    def execute(self, blob: CodeBlob, context: Dict[str, Any] = None) -> Any:
        """Execute a code blob."""
        if not blob.is_trusted:
            raise ValueError("Cannot execute untrusted code blob")
        
        # Check imports
        for imp in blob.required_imports:
            if self._allow_imports and imp not in self._allow_imports:
                raise ValueError(f"Import '{imp}' not allowed")
        
        # Create isolated namespace
        local_namespace = {
            '__builtins__': __builtins__,
            **(context or {})
        }
        
        # Execute
        try:
            exec(blob.source_code, local_namespace)
            self._executed_blobs.add(blob.blob_id)
            
            # Return the created object (function, class, etc.)
            if blob.name and blob.name in local_namespace:
                return local_namespace[blob.name]
            
            return local_namespace
            
        except Exception as e:
            raise RuntimeError(f"Code execution failed: {e}")
    
    def create_function(self, blob: CodeBlob, context: Dict[str, Any] = None) -> Callable:
        """Create a callable function from blob."""
        result = self.execute(blob, context)
        if callable(result):
            return result
        raise ValueError("Blob did not produce a callable")


class CodePipeline:
    """
    Standalone code management pipeline.
    
    Manages code blobs with versioning and execution.
    """
    
    def __init__(self, allowed_imports: List[str] = None):
        self._blobs: Dict[str, CodeBlob] = {}
        self._versions: Dict[str, List[str]] = {}  # name -> [blob_ids]
        self._executor = CodeExecutor(allowed_imports)
        self._cache: Dict[str, Any] = {}  # blob_id -> executed result
    
    def register(self, source_code: str, 
                name: str = "",
                code_type: str = "function",
                required_imports: List[str] = None,
                trusted: bool = False) -> CodeBlob:
        """Register code as a blob."""
        blob = CodeBlob(
            name=name,
            source_code=source_code,
            code_type=code_type,
            required_imports=required_imports or [],
            is_trusted=trusted
        )
        
        self._blobs[blob.blob_id] = blob
        
        # Track versions by name
        if name:
            if name not in self._versions:
                self._versions[name] = []
            self._versions[name].append(blob.blob_id)
        
        return blob
    
    def get_blob(self, blob_id: str) -> Optional[CodeBlob]:
        """Get blob by ID."""
        return self._blobs.get(blob_id)
    
    def get_latest_version(self, name: str) -> Optional[CodeBlob]:
        """Get latest version of named code."""
        versions = self._versions.get(name, [])
        if versions:
            return self._blobs.get(versions[-1])
        return None
    
    def execute(self, blob_id: str, context: Dict[str, Any] = None) -> Any:
        """Execute a registered blob."""
        blob = self._blobs.get(blob_id)
        if not blob:
            raise ValueError(f"Blob '{blob_id}' not found")
        
        # Check cache
        if blob_id in self._cache:
            return self._cache[blob_id]
        
        result = self._executor.execute(blob, context)
        self._cache[blob_id] = result
        return result
    
    def create_decorator(self, blob_id: str, context: Dict[str, Any] = None) -> Callable:
        """Create a decorator from a blob."""
        result = self.execute(blob_id, context)
        if callable(result):
            return result
        raise ValueError("Blob is not a decorator")
    
    def hot_swap(self, old_blob_id: str, new_blob_id: str) -> bool:
        """Swap one blob for another (invalidates cache)."""
        if old_blob_id not in self._blobs or new_blob_id not in self._blobs:
            return False
        
        # Invalidate cache for old blob
        if old_blob_id in self._cache:
            del self._cache[old_blob_id]
        
        return True


# =============================================================================
# ADAPTERS: Thin bridges between pipelines
# =============================================================================

class SignalContextAdapter:
    """
    Adapter to connect SignalPipeline with ContextPipeline.
    
    Emits context changes as signals, receives signals as context updates.
    """
    
    def __init__(self, signal_pipeline: SignalPipeline, context_pipeline: ContextPipeline):
        self._signals = signal_pipeline
        self._contexts = context_pipeline
    
    def bind_context_to_signals(self, context_id: str):
        """Bind context changes to emit signals."""
        ctx = self._contexts.get(context_id)
        if ctx:
            def on_change(key, old_val, new_val):
                signal = Signal(
                    signal_type="context_changed",
                    source=context_id,
                    payload={
                        'key': key,
                        'old_value': old_val,
                        'new_value': new_val
                    }
                )
                asyncio.create_task(self._signals.emit(signal))
            
            ctx.on_change(on_change)
    
    async def create_signal_observer_for_context(self, context_id: str) -> str:
        """Create observer that updates context from signals."""
        ctx = self._contexts.get(context_id)
        if not ctx:
            raise ValueError(f"Context '{context_id}' not found")
        
        class ContextUpdater(SignalObserver):
            def __init__(self, context: Context):
                super().__init__()
                self._ctx = context
                self.filters = {"context_update"}
            
            async def on_event(self, signal: Signal) -> Any:
                if signal.target == self._ctx.context_id:
                    updates = signal.payload.get('updates', {})
                    self._ctx.update(updates)
        
        observer = ContextUpdater(ctx)
        return await self._signals.register(observer)


class GeneratorNarrativeAdapter:
    """
    Adapter to connect GeneratorPipeline with NarrativePipeline.
    
    Allows generators to drive narrative chains.
    """
    
    def __init__(self, generator_pipeline: GeneratorPipeline, narrative_pipeline: NarrativePipeline):
        self._generators = generator_pipeline
        self._narrative = narrative_pipeline
    
    def create_generator_driven_chain(self, 
                                      generator_name: str,
                                      chain_name: str = "") -> NarrativeChain:
        """Create a narrative chain driven by a generator."""
        chain = self._narrative.create_chain(chain_name)
        
        def make_link_handler(gen_config):
            def handler(context, data):
                gen = self._generators.create_generator(generator_name, gen_config)
                gen.start(context)
                results = [r.value for r in gen]
                return {'generator_results': results}
            return handler
        
        return chain
    
    def wrap_chain_as_generator(self, chain_id: str) -> Callable:
        """Wrap a narrative chain as a generator factory."""
        def chain_generator(context):
            chain = self._narrative.get_chain(chain_id)
            if chain:
                chain.start(context)
                for event in chain:
                    yield event
        
        return chain_generator


class GraphIteratorAdapter:
    """
    Adapter to connect GraphPipeline with IteratorPipeline.
    
    Records graph traversals as operations in context windows.
    """
    
    def __init__(self, graph_pipeline: GraphPipeline, iterator_pipeline: IteratorPipeline):
        self._graph = graph_pipeline
        self._iterator = iterator_pipeline
    
    def traverse_and_record(self, 
                           start_node: str,
                           window_id: str,
                           mode: str = "bfs") -> List[str]:
        """Traverse graph and record each visit as an operation."""
        window = self._iterator.get_window(window_id)
        if not window:
            window = self._iterator.create_window(window_id)
        
        operation_ids = []
        for node, depth in self._graph.traverse(start_node, mode):
            op_id = self._iterator.record_operation(
                window_id,
                operation_type="graph_visit",
                data={
                    'node_id': node.node_id,
                    'node_type': node.node_type,
                    'depth': depth
                }
            )
            if op_id:
                operation_ids.append(op_id)
        
        return operation_ids


# =============================================================================
# UNIFIED FACADE (Optional - for convenience)
# =============================================================================

class UnifiedPipelineFacade:
    """
    Optional facade that provides unified access to all pipelines.
    
    Each pipeline remains independent; this just provides convenience.
    """
    
    def __init__(self):
        self.signals = SignalPipeline()
        self.contexts = ContextPipeline()
        self.generators = GeneratorPipeline()
        self.graphs = GraphPipeline()
        self.narratives = NarrativePipeline()
        self.iterators = IteratorPipeline()
        self.code = CodePipeline()
        
        # Adapters (created on demand)
        self._adapters: Dict[str, Any] = {}
    
    def get_signal_context_adapter(self) -> SignalContextAdapter:
        """Get or create signal-context adapter."""
        if 'signal_context' not in self._adapters:
            self._adapters['signal_context'] = SignalContextAdapter(
                self.signals, self.contexts
            )
        return self._adapters['signal_context']
    
    def get_generator_narrative_adapter(self) -> GeneratorNarrativeAdapter:
        """Get or create generator-narrative adapter."""
        if 'generator_narrative' not in self._adapters:
            self._adapters['generator_narrative'] = GeneratorNarrativeAdapter(
                self.generators, self.narratives
            )
        return self._adapters['generator_narrative']
    
    def get_graph_iterator_adapter(self) -> GraphIteratorAdapter:
        """Get or create graph-iterator adapter."""
        if 'graph_iterator' not in self._adapters:
            self._adapters['graph_iterator'] = GraphIteratorAdapter(
                self.graphs, self.iterators
            )
        return self._adapters['graph_iterator']


# =============================================================================
# DEMONSTRATION
# =============================================================================

async def demo_decoupled_pipelines():
    """Demonstrate the decoupled pipeline architecture."""
    print("=" * 60)
    print("DECOUPLED PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # 1. Signal Pipeline (standalone)
    print("\n📡 Signal Pipeline")
    print("-" * 40)
    
    signal_pipe = SignalPipeline()
    await signal_pipe.start()
    
    class LoggingObserver(SignalObserver):
        async def on_event(self, signal: Signal) -> Any:
            print(f"  Received: {signal.signal_type} from {signal.source}")
    
    await signal_pipe.register(LoggingObserver())
    await signal_pipe.emit(Signal(signal_type="test", source="demo"))
    print(f"  Stats: {signal_pipe.get_stats()}")
    
    # 2. Context Pipeline (standalone)
    print("\n📦 Context Pipeline")
    print("-" * 40)
    
    ctx_pipe = ContextPipeline()
    ctx = ctx_pipe.create("game_state")
    ctx.set("player_name", "Hero")
    ctx.set("health", 100, create_snapshot=True)
    ctx.set("health", 75, create_snapshot=True)
    print(f"  Context data: {ctx.data}")
    print(f"  Snapshots: {len(ctx._snapshots)}")
    
    # 3. Generator Pipeline (standalone)
    print("\n🔄 Generator Pipeline")
    print("-" * 40)
    
    gen_pipe = GeneratorPipeline()
    
    def number_gen(start=0, count=3):
        for i in range(count):
            yield start + i
    
    def doubler_gen(value):
        yield value * 2
    
    gen_pipe.register_factory("numbers", number_gen)
    gen_pipe.register_factory("doubler", doubler_gen)
    
    num_gen = gen_pipe.create_generator("numbers", {"start": 1, "count": 3})
    num_gen.start()
    print(f"  Generated: {[r.value for r in num_gen]}")
    
    # 4. Graph Pipeline (standalone)
    print("\n🕸️ Graph Pipeline")
    print("-" * 40)
    
    graph_pipe = GraphPipeline()
    
    # Create nodes
    a = GraphNode(node_id="A", node_type="room", data={"name": "Entrance"})
    b = GraphNode(node_id="B", node_type="room", data={"name": "Hall"})
    c = GraphNode(node_id="C", node_type="room", data={"name": "Treasury"})
    
    graph_pipe.add_node(a)
    graph_pipe.add_node(b)
    graph_pipe.add_node(c)
    
    graph_pipe.add_edge(GraphEdge(source_id="A", target_id="B"))
    graph_pipe.add_edge(GraphEdge(source_id="B", target_id="C"))
    
    print(f"  Traversal from A:")
    for node, depth in graph_pipe.traverse("A"):
        print(f"    Depth {depth}: {node.data['name']}")
    
    path = graph_pipe.find_path("A", "C")
    print(f"  Path A->C: {path}")
    
    # 5. Narrative Pipeline (standalone)
    print("\n📖 Narrative Pipeline")
    print("-" * 40)
    
    narr_pipe = NarrativePipeline()
    quest = narr_pipe.create_chain("main_quest")
    
    quest.add_link(ChainLink(
        name="start",
        handler=lambda ctx, data: {"message": "Quest begins!"}
    ))
    quest.add_link(ChainLink(
        name="middle",
        handler=lambda ctx, data: {"message": "Challenges faced..."}
    ))
    quest.add_link(ChainLink(
        name="end",
        handler=lambda ctx, data: {"message": "Victory!"}
    ))
    
    events = narr_pipe.run_chain(quest.chain_id)
    for event in events:
        print(f"  Link {event.link_index}: {event.data['link_name']} -> {event.data['result']}")
    
    # 6. Iterator Pipeline (standalone)
    print("\n🔁 Iterator Pipeline")
    print("-" * 40)
    
    iter_pipe = IteratorPipeline()
    window = iter_pipe.create_window("game_history", size=100)
    
    iter_pipe.record_operation("game_history", "player_move", {"x": 10, "y": 20})
    iter_pipe.record_operation("game_history", "combat", {"enemy": "goblin", "result": "win"})
    iter_pipe.record_operation("game_history", "pickup", {"item": "sword"})
    
    print(f"  Recent operations:")
    for op in window.get_recent(3):
        print(f"    {op.operation_type}: {op.data}")
    
    # 7. Code Pipeline (standalone)
    print("\n💻 Code Pipeline")
    print("-" * 40)
    
    code_pipe = CodePipeline()
    
    blob = code_pipe.register(
        source_code="""
def greet(name):
    return f"Hello, {name}!"
greet
""",
        name="greet",
        code_type="function",
        trusted=True
    )
    
    result = code_pipe.execute(blob.blob_id)
    print(f"  Executed code result: {result('World')}")
    
    # 8. Using Adapters (composition)
    print("\n🔗 Adapter Composition")
    print("-" * 40)
    
    facade = UnifiedPipelineFacade()
    await facade.signals.start()
    
    adapter = facade.get_signal_context_adapter()
    test_ctx = facade.contexts.create("test")
    adapter.bind_context_to_signals("test")
    
    test_ctx.set("value", 42)  # This will emit a signal
    print("  Context-Signal binding active")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    await signal_pipe.stop()
    await facade.signals.stop()


if __name__ == "__main__":
    asyncio.run(demo_decoupled_pipelines())
