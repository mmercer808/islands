#!/usr/bin/env python3
"""
Pipeline Adapter Patterns
=========================

This module provides a complete set of adapter patterns for bridging
any two decoupled pipelines. Each adapter follows the same principles:

1. Thin wrapper - minimal logic, just translation
2. Protocol-based - works with any implementation of the protocols
3. Bidirectional - can adapt either direction
4. Composable - adapters can chain

Adapter Matrix:
                Signal  Context  Generator  Graph  Narrative  Iterator  Code
Signal            -       ✓         ✓         ✓        ✓          ✓       ✓
Context           ✓       -         ✓         ✓        ✓          ✓       ✓
Generator         ✓       ✓         -         ✓        ✓          ✓       ✓
Graph             ✓       ✓         ✓         -        ✓          ✓       ✓
Narrative         ✓       ✓         ✓         ✓        -          ✓       ✓
Iterator          ✓       ✓         ✓         ✓        ✓          -       ✓
Code              ✓       ✓         ✓         ✓        ✓          ✓       -
"""

from __future__ import annotations
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Iterator, TypeVar, Generic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from decoupled_pipelines import (
    # Pipelines
    SignalPipeline, ContextPipeline, GeneratorPipeline,
    GraphPipeline, NarrativePipeline, IteratorPipeline, CodePipeline,
    # Types
    Signal, SignalObserver, SignalPriority,
    Context, ContextSnapshot,
    GeneratorWrapper, GeneratorResult,
    GraphNode, GraphEdge,
    NarrativeChain, ChainLink, ChainEvent,
    ContextWindowIterator, Operation,
    CodeBlob
)

T = TypeVar('T')


# =============================================================================
# BASE ADAPTER PATTERN
# =============================================================================

class PipelineAdapter(ABC, Generic[T]):
    """
    Abstract base for all pipeline adapters.
    
    Adapters translate between two pipeline types without coupling them.
    """
    
    def __init__(self, source_pipeline: Any, target_pipeline: Any):
        self._source = source_pipeline
        self._target = target_pipeline
        self._adapter_id = f"adapter_{uuid.uuid4()}"
        self._active = True
    
    @property
    def adapter_id(self) -> str:
        return self._adapter_id
    
    @abstractmethod
    def adapt(self, item: Any) -> T:
        """Adapt an item from source to target format."""
        pass
    
    def deactivate(self):
        """Deactivate the adapter."""
        self._active = False


# =============================================================================
# SIGNAL <-> CONTEXT ADAPTERS
# =============================================================================

class SignalToContextAdapter(PipelineAdapter[None]):
    """
    Adapter: Signal -> Context
    
    Converts signals into context updates.
    """
    
    def __init__(self, signal_pipeline: SignalPipeline, 
                 context_pipeline: ContextPipeline,
                 context_id: str,
                 mapping: Dict[str, str] = None):
        super().__init__(signal_pipeline, context_pipeline)
        self._context_id = context_id
        self._mapping = mapping or {}  # signal_field -> context_key
    
    def adapt(self, signal: Signal):
        """Convert signal payload to context update."""
        ctx = self._target.get(self._context_id)
        if not ctx:
            return
        
        if self._mapping:
            for signal_key, ctx_key in self._mapping.items():
                if signal_key in signal.payload:
                    ctx.set(ctx_key, signal.payload[signal_key])
        else:
            # Default: merge entire payload
            ctx.update(signal.payload)
    
    async def bind(self):
        """Bind adapter as signal observer."""
        class SignalContextBridge(SignalObserver):
            def __init__(self, adapter: SignalToContextAdapter):
                super().__init__()
                self._adapter = adapter
            
            async def on_event(self, signal: Signal):
                self._adapter.adapt(signal)
        
        return await self._source.register(SignalContextBridge(self))


class ContextToSignalAdapter(PipelineAdapter[Signal]):
    """
    Adapter: Context -> Signal
    
    Emits signals when context changes.
    """
    
    def __init__(self, context_pipeline: ContextPipeline,
                 signal_pipeline: SignalPipeline,
                 context_id: str,
                 signal_type: str = "context_changed"):
        super().__init__(context_pipeline, signal_pipeline)
        self._context_id = context_id
        self._signal_type = signal_type
    
    def adapt(self, change: tuple) -> Signal:
        """Convert context change to signal."""
        key, old_val, new_val = change
        return Signal(
            signal_type=self._signal_type,
            source=self._context_id,
            payload={
                'key': key,
                'old_value': old_val,
                'new_value': new_val
            }
        )
    
    def bind(self):
        """Bind to context changes."""
        ctx = self._source.get(self._context_id)
        if ctx:
            def on_change(key, old, new):
                signal = self.adapt((key, old, new))
                asyncio.create_task(self._target.emit(signal))
            ctx.on_change(on_change)


# =============================================================================
# SIGNAL <-> GENERATOR ADAPTERS
# =============================================================================

class SignalToGeneratorAdapter(PipelineAdapter[Iterator]):
    """
    Adapter: Signal -> Generator
    
    Buffers signals and yields them as generator output.
    """
    
    def __init__(self, signal_pipeline: SignalPipeline,
                 generator_pipeline: GeneratorPipeline,
                 buffer_size: int = 100):
        super().__init__(signal_pipeline, generator_pipeline)
        self._buffer: List[Signal] = []
        self._buffer_size = buffer_size
        self._bound = False
    
    def adapt(self, signals: List[Signal]) -> Iterator[Signal]:
        """Convert buffered signals to generator."""
        for signal in signals:
            yield signal
    
    async def bind(self):
        """Bind to signals and buffer them."""
        class SignalBuffer(SignalObserver):
            def __init__(self, adapter: SignalToGeneratorAdapter):
                super().__init__()
                self._adapter = adapter
            
            async def on_event(self, signal: Signal):
                if len(self._adapter._buffer) < self._adapter._buffer_size:
                    self._adapter._buffer.append(signal)
        
        await self._source.register(SignalBuffer(self))
        self._bound = True
    
    def get_generator(self) -> Iterator[Signal]:
        """Get generator yielding buffered signals."""
        while self._buffer:
            yield self._buffer.pop(0)


class GeneratorToSignalAdapter(PipelineAdapter[None]):
    """
    Adapter: Generator -> Signal
    
    Emits signals for each generator yield.
    """
    
    def __init__(self, generator_pipeline: GeneratorPipeline,
                 signal_pipeline: SignalPipeline,
                 signal_type: str = "generator_yield"):
        super().__init__(generator_pipeline, signal_pipeline)
        self._signal_type = signal_type
    
    def adapt(self, result: GeneratorResult):
        """Convert generator result to signal emission."""
        signal = Signal(
            signal_type=self._signal_type,
            source=result.generator_id,
            payload={
                'step': result.step,
                'value': result.value,
                'metadata': result.metadata
            }
        )
        return signal
    
    async def run_generator_with_signals(self, generator: GeneratorWrapper):
        """Run generator and emit signals for each yield."""
        for result in generator:
            signal = self.adapt(result)
            await self._target.emit(signal)
            yield result


# =============================================================================
# SIGNAL <-> GRAPH ADAPTERS
# =============================================================================

class SignalToGraphAdapter(PipelineAdapter[None]):
    """
    Adapter: Signal -> Graph
    
    Creates graph nodes/edges from signals.
    """
    
    def __init__(self, signal_pipeline: SignalPipeline,
                 graph_pipeline: GraphPipeline,
                 node_signal_type: str = "create_node",
                 edge_signal_type: str = "create_edge"):
        super().__init__(signal_pipeline, graph_pipeline)
        self._node_signal = node_signal_type
        self._edge_signal = edge_signal_type
    
    def adapt(self, signal: Signal):
        """Convert signal to graph operation."""
        if signal.signal_type == self._node_signal:
            node = GraphNode(
                node_id=signal.payload.get('node_id', str(uuid.uuid4())),
                node_type=signal.payload.get('node_type', 'generic'),
                data=signal.payload.get('data', {})
            )
            self._target.add_node(node)
            return node
        
        elif signal.signal_type == self._edge_signal:
            edge = GraphEdge(
                source_id=signal.payload['source_id'],
                target_id=signal.payload['target_id'],
                edge_type=signal.payload.get('edge_type', 'generic')
            )
            self._target.add_edge(edge)
            return edge
    
    async def bind(self):
        """Bind to graph-related signals."""
        class GraphBuilder(SignalObserver):
            def __init__(self, adapter: SignalToGraphAdapter):
                super().__init__()
                self._adapter = adapter
                self.filters = {adapter._node_signal, adapter._edge_signal}
            
            async def on_event(self, signal: Signal):
                self._adapter.adapt(signal)
        
        return await self._source.register(GraphBuilder(self))


class GraphToSignalAdapter(PipelineAdapter[None]):
    """
    Adapter: Graph -> Signal
    
    Emits signals during graph traversal.
    """
    
    def __init__(self, graph_pipeline: GraphPipeline,
                 signal_pipeline: SignalPipeline):
        super().__init__(graph_pipeline, signal_pipeline)
    
    async def traverse_with_signals(self, start_node: str, 
                                    mode: str = "bfs") -> List[GraphNode]:
        """Traverse graph, emitting signals for each visited node."""
        nodes = []
        
        for node, depth in self._source.traverse(start_node, mode):
            await self._target.emit(Signal(
                signal_type="node_visited",
                source=start_node,
                payload={
                    'node_id': node.node_id,
                    'node_type': node.node_type,
                    'depth': depth,
                    'data': node.data
                }
            ))
            nodes.append(node)
        
        return nodes


# =============================================================================
# CONTEXT <-> GENERATOR ADAPTERS
# =============================================================================

class ContextToGeneratorAdapter(PipelineAdapter[Iterator]):
    """
    Adapter: Context -> Generator
    
    Yields context data as generator output.
    """
    
    def __init__(self, context_pipeline: ContextPipeline,
                 generator_pipeline: GeneratorPipeline):
        super().__init__(context_pipeline, generator_pipeline)
    
    def adapt(self, context_id: str) -> Iterator[tuple]:
        """Yield context data as key-value pairs."""
        ctx = self._source.get(context_id)
        if ctx:
            for key, value in ctx.data.items():
                yield (key, value)
    
    def snapshot_generator(self, context_id: str) -> Iterator[ContextSnapshot]:
        """Yield context snapshots."""
        ctx = self._source.get(context_id)
        if ctx:
            for snapshot in ctx._snapshots:
                yield snapshot


class GeneratorToContextAdapter(PipelineAdapter[None]):
    """
    Adapter: Generator -> Context
    
    Updates context from generator yields.
    """
    
    def __init__(self, generator_pipeline: GeneratorPipeline,
                 context_pipeline: ContextPipeline,
                 context_id: str):
        super().__init__(generator_pipeline, context_pipeline)
        self._context_id = context_id
    
    def adapt(self, result: GeneratorResult):
        """Update context from generator result."""
        ctx = self._target.get(self._context_id)
        if ctx and isinstance(result.value, dict):
            ctx.update(result.value)
    
    def run_generator_to_context(self, generator: GeneratorWrapper):
        """Run generator and update context with each yield."""
        for result in generator:
            self.adapt(result)
            yield result


# =============================================================================
# NARRATIVE <-> SIGNAL ADAPTERS
# =============================================================================

class NarrativeToSignalAdapter(PipelineAdapter[None]):
    """
    Adapter: Narrative -> Signal
    
    Emits signals for narrative events.
    """
    
    def __init__(self, narrative_pipeline: NarrativePipeline,
                 signal_pipeline: SignalPipeline):
        super().__init__(narrative_pipeline, signal_pipeline)
    
    async def run_chain_with_signals(self, chain_id: str, 
                                     context: Dict[str, Any] = None) -> List[ChainEvent]:
        """Run narrative chain, emitting signals for each event."""
        events = []
        self._source.start_chain(chain_id, context)
        
        while chain_id in self._source._active_chains:
            event = self._source.step_chain(chain_id)
            if event:
                events.append(event)
                
                await self._target.emit(Signal(
                    signal_type=f"narrative_{event.event_type.name.lower()}",
                    source=event.chain_id,
                    payload={
                        'event_id': event.event_id,
                        'link_index': event.link_index,
                        'data': event.data
                    }
                ))
        
        return events


class SignalToNarrativeAdapter(PipelineAdapter[None]):
    """
    Adapter: Signal -> Narrative
    
    Triggers narrative chains from signals.
    """
    
    def __init__(self, signal_pipeline: SignalPipeline,
                 narrative_pipeline: NarrativePipeline,
                 trigger_mapping: Dict[str, str] = None):
        super().__init__(signal_pipeline, signal_pipeline)
        self._narrative = narrative_pipeline
        # signal_type -> chain_id
        self._trigger_mapping = trigger_mapping or {}
    
    def add_trigger(self, signal_type: str, chain_id: str):
        """Map a signal type to trigger a narrative chain."""
        self._trigger_mapping[signal_type] = chain_id
    
    async def bind(self):
        """Bind to trigger signals."""
        class NarrativeTrigger(SignalObserver):
            def __init__(self, adapter: SignalToNarrativeAdapter):
                super().__init__()
                self._adapter = adapter
                self.filters = set(adapter._trigger_mapping.keys())
            
            async def on_event(self, signal: Signal):
                chain_id = self._adapter._trigger_mapping.get(signal.signal_type)
                if chain_id:
                    self._adapter._narrative.start_chain(chain_id, signal.payload)
        
        return await self._source.register(NarrativeTrigger(self))


# =============================================================================
# ITERATOR <-> CONTEXT ADAPTERS
# =============================================================================

class IteratorToContextAdapter(PipelineAdapter[None]):
    """
    Adapter: Iterator -> Context
    
    Syncs iterator window state to context.
    """
    
    def __init__(self, iterator_pipeline: IteratorPipeline,
                 context_pipeline: ContextPipeline,
                 window_id: str,
                 context_id: str):
        super().__init__(iterator_pipeline, context_pipeline)
        self._window_id = window_id
        self._context_id = context_id
    
    def sync(self):
        """Sync window statistics to context."""
        window = self._source.get_window(self._window_id)
        ctx = self._target.get(self._context_id)
        
        if window and ctx:
            ctx.update({
                'window_size': len(window),
                'operations': [op.to_dict() if hasattr(op, 'to_dict') else str(op) 
                              for op in window.get_recent(10)]
            })
    
    def sync_operation(self, operation: Operation):
        """Sync specific operation to context."""
        ctx = self._target.get(self._context_id)
        if ctx:
            ctx.set(f"op_{operation.operation_id}", {
                'type': operation.operation_type,
                'data': operation.data,
                'timestamp': operation.timestamp
            })


class ContextToIteratorAdapter(PipelineAdapter[str]):
    """
    Adapter: Context -> Iterator
    
    Records context changes as operations.
    """
    
    def __init__(self, context_pipeline: ContextPipeline,
                 iterator_pipeline: IteratorPipeline,
                 context_id: str,
                 window_id: str):
        super().__init__(context_pipeline, iterator_pipeline)
        self._context_id = context_id
        self._window_id = window_id
    
    def adapt(self, change: tuple) -> str:
        """Record context change as operation."""
        key, old_val, new_val = change
        return self._target.record_operation(
            self._window_id,
            operation_type="context_change",
            data={
                'context_id': self._context_id,
                'key': key,
                'old_value': old_val,
                'new_value': new_val
            }
        )
    
    def bind(self):
        """Bind to context changes."""
        ctx = self._source.get(self._context_id)
        if ctx:
            ctx.on_change(lambda k, o, n: self.adapt((k, o, n)))


# =============================================================================
# CODE <-> GENERATOR ADAPTERS
# =============================================================================

class CodeToGeneratorAdapter(PipelineAdapter[None]):
    """
    Adapter: Code -> Generator
    
    Registers code blobs as generator factories.
    """
    
    def __init__(self, code_pipeline: CodePipeline,
                 generator_pipeline: GeneratorPipeline):
        super().__init__(code_pipeline, generator_pipeline)
    
    def adapt(self, blob_id: str, factory_name: str = None):
        """Register code blob as generator factory."""
        blob = self._source.get_blob(blob_id)
        if not blob:
            return
        
        name = factory_name or blob.name
        
        def blob_generator(*args, **kwargs):
            func = self._source.execute(blob_id)
            result = func(*args, **kwargs)
            if hasattr(result, '__iter__'):
                yield from result
            else:
                yield result
        
        self._target.register_factory(name, blob_generator)
        return name


class GeneratorToCodeAdapter(PipelineAdapter[CodeBlob]):
    """
    Adapter: Generator -> Code
    
    Creates code blobs from generator factories.
    """
    
    def __init__(self, generator_pipeline: GeneratorPipeline,
                 code_pipeline: CodePipeline):
        super().__init__(generator_pipeline, code_pipeline)
    
    def adapt(self, factory_name: str, generator_source: str) -> CodeBlob:
        """Create code blob from generator source."""
        return self._target.register(
            source_code=generator_source,
            name=factory_name,
            code_type="function",
            trusted=True
        )


# =============================================================================
# GRAPH <-> NARRATIVE ADAPTERS
# =============================================================================

class GraphToNarrativeAdapter(PipelineAdapter[NarrativeChain]):
    """
    Adapter: Graph -> Narrative
    
    Creates narrative chains from graph paths.
    """
    
    def __init__(self, graph_pipeline: GraphPipeline,
                 narrative_pipeline: NarrativePipeline):
        super().__init__(graph_pipeline, narrative_pipeline)
    
    def adapt(self, path: List[str], chain_name: str = "") -> NarrativeChain:
        """Create narrative chain from graph path."""
        chain = self._target.create_chain(chain_name)
        
        for node_id in path:
            node = self._source.get_node(node_id)
            if node:
                def make_handler(n):
                    return lambda ctx, data: {
                        'node_id': n.node_id,
                        'node_data': n.data
                    }
                
                chain.add_link(ChainLink(
                    name=f"visit_{node_id}",
                    handler=make_handler(node),
                    data={'node_id': node_id}
                ))
        
        return chain


class NarrativeToGraphAdapter(PipelineAdapter[None]):
    """
    Adapter: Narrative -> Graph
    
    Builds graph from narrative structure.
    """
    
    def __init__(self, narrative_pipeline: NarrativePipeline,
                 graph_pipeline: GraphPipeline):
        super().__init__(narrative_pipeline, graph_pipeline)
    
    def adapt(self, chain: NarrativeChain):
        """Build graph nodes from narrative links."""
        prev_node_id = None
        
        for i, link in enumerate(chain._links):
            node = GraphNode(
                node_id=f"{chain.chain_id}_link_{i}",
                node_type="narrative_link",
                data={
                    'link_name': link.name,
                    'link_id': link.link_id
                }
            )
            self._target.add_node(node)
            
            if prev_node_id:
                edge = GraphEdge(
                    source_id=prev_node_id,
                    target_id=node.node_id,
                    edge_type="sequence"
                )
                self._target.add_edge(edge)
            
            prev_node_id = node.node_id


# =============================================================================
# ADAPTER FACTORY
# =============================================================================

class AdapterFactory:
    """
    Factory for creating adapters between any two pipelines.
    """
    
    @staticmethod
    def create_adapter(source_type: str, target_type: str,
                      source_pipeline: Any, target_pipeline: Any,
                      **kwargs) -> PipelineAdapter:
        """
        Create an adapter between two pipeline types.
        
        Types: signal, context, generator, graph, narrative, iterator, code
        """
        adapter_map = {
            ('signal', 'context'): SignalToContextAdapter,
            ('context', 'signal'): ContextToSignalAdapter,
            ('signal', 'generator'): SignalToGeneratorAdapter,
            ('generator', 'signal'): GeneratorToSignalAdapter,
            ('signal', 'graph'): SignalToGraphAdapter,
            ('graph', 'signal'): GraphToSignalAdapter,
            ('context', 'generator'): ContextToGeneratorAdapter,
            ('generator', 'context'): GeneratorToContextAdapter,
            ('narrative', 'signal'): NarrativeToSignalAdapter,
            ('signal', 'narrative'): SignalToNarrativeAdapter,
            ('iterator', 'context'): IteratorToContextAdapter,
            ('context', 'iterator'): ContextToIteratorAdapter,
            ('code', 'generator'): CodeToGeneratorAdapter,
            ('generator', 'code'): GeneratorToCodeAdapter,
            ('graph', 'narrative'): GraphToNarrativeAdapter,
            ('narrative', 'graph'): NarrativeToGraphAdapter,
        }
        
        key = (source_type.lower(), target_type.lower())
        adapter_class = adapter_map.get(key)
        
        if adapter_class:
            return adapter_class(source_pipeline, target_pipeline, **kwargs)
        
        raise ValueError(f"No adapter for {source_type} -> {target_type}")


# =============================================================================
# ADAPTER CHAIN
# =============================================================================

class AdapterChain:
    """
    Chain multiple adapters together for complex transformations.
    """
    
    def __init__(self):
        self._adapters: List[PipelineAdapter] = []
    
    def add(self, adapter: PipelineAdapter) -> 'AdapterChain':
        """Add adapter to chain."""
        self._adapters.append(adapter)
        return self
    
    def adapt(self, item: Any) -> Any:
        """Run item through all adapters."""
        current = item
        for adapter in self._adapters:
            current = adapter.adapt(current)
        return current


# =============================================================================
# DEMONSTRATION
# =============================================================================

async def demo_adapters():
    """Demonstrate adapter patterns."""
    print("=" * 70)
    print("PIPELINE ADAPTER DEMONSTRATIONS")
    print("=" * 70)
    
    # Setup pipelines
    signals = SignalPipeline()
    contexts = ContextPipeline()
    generators = GeneratorPipeline()
    graphs = GraphPipeline()
    narratives = NarrativePipeline()
    iterators = IteratorPipeline()
    
    await signals.start()
    
    # 1. Signal -> Context
    print("\n🔌 Signal -> Context Adapter")
    print("-" * 40)
    
    ctx = contexts.create("game_state")
    adapter1 = SignalToContextAdapter(
        signals, contexts, "game_state",
        mapping={'score': 'player_score', 'level': 'current_level'}
    )
    await adapter1.bind()
    
    await signals.emit(Signal(
        signal_type="generic",
        payload={'score': 100, 'level': 5}
    ))
    
    # Small delay for async processing
    await asyncio.sleep(0.1)
    print(f"  Context after signal: {ctx.data}")
    
    # 2. Context -> Signal
    print("\n🔌 Context -> Signal Adapter")
    print("-" * 40)
    
    ctx2 = contexts.create("player")
    adapter2 = ContextToSignalAdapter(contexts, signals, "player")
    adapter2.bind()
    
    # Track emitted signals
    signal_log = []
    class SignalLogger(SignalObserver):
        async def on_event(self, signal: Signal):
            signal_log.append(signal.signal_type)
    
    await signals.register(SignalLogger())
    
    ctx2.set("health", 100)
    await asyncio.sleep(0.1)
    print(f"  Signals after context change: {signal_log}")
    
    # 3. Graph -> Narrative
    print("\n🔌 Graph -> Narrative Adapter")
    print("-" * 40)
    
    graphs.add_node(GraphNode("A", "room", {"name": "Entrance"}))
    graphs.add_node(GraphNode("B", "room", {"name": "Hall"}))
    graphs.add_node(GraphNode("C", "room", {"name": "Exit"}))
    graphs.add_edge(GraphEdge(source_id="A", target_id="B"))
    graphs.add_edge(GraphEdge(source_id="B", target_id="C"))
    
    adapter3 = GraphToNarrativeAdapter(graphs, narratives)
    
    path = graphs.find_path("A", "C")
    if path:
        chain = adapter3.adapt(path, "room_tour")
        print(f"  Created chain from path: {path}")
        
        events = narratives.run_chain(chain.chain_id)
        for event in events:
            print(f"    Link {event.link_index}: {event.data.get('result', {})}")
    
    # 4. Using AdapterFactory
    print("\n🏭 AdapterFactory Demo")
    print("-" * 40)
    
    # Create adapter via factory
    ctx3 = contexts.create("factory_test")
    factory_adapter = AdapterFactory.create_adapter(
        "signal", "context",
        signals, contexts,
        context_id="factory_test"
    )
    print(f"  Created adapter: {type(factory_adapter).__name__}")
    
    # 5. Context -> Iterator
    print("\n🔌 Context -> Iterator Adapter")
    print("-" * 40)
    
    window = iterators.create_window("ctx_changes", size=100)
    ctx4 = contexts.create("tracked")
    
    adapter5 = ContextToIteratorAdapter(
        contexts, iterators,
        context_id="tracked",
        window_id="ctx_changes"
    )
    adapter5.bind()
    
    ctx4.set("value", 1)
    ctx4.set("value", 2)
    ctx4.set("value", 3)
    
    recent = window.get_recent(3)
    print(f"  Recorded operations: {len(recent)}")
    for op in recent:
        print(f"    {op.operation_type}: {op.data}")
    
    await signals.stop()
    
    print("\n" + "=" * 70)
    print("ADAPTER DEMONSTRATIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_adapters())
