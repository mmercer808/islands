#!/usr/bin/env python3
"""
Advanced Pipeline Composition Patterns
======================================

This module demonstrates advanced patterns for composing the decoupled pipelines
into specialized configurations for different use cases.

Configurations:
1. StoryEngine - Narrative + Context + Signal
2. GameWorld - Graph + Iterator + Context  
3. AICollaboration - Generator + Signal + Code
4. TextProcessor - Iterator + Generator + Code
5. EventSourcing - Signal + Context + Iterator
6. WorldBuilder - Graph + Narrative + Context

Each configuration shows a specific use case with minimal coupling.
"""

from __future__ import annotations
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Import from our decoupled pipelines
from decoupled_pipelines import (
    # Pipelines
    SignalPipeline, ContextPipeline, GeneratorPipeline,
    GraphPipeline, NarrativePipeline, IteratorPipeline, CodePipeline,
    # Core types
    Signal, SignalPriority, SignalObserver,
    Context, ContextSnapshot, ContextState,
    GeneratorWrapper, GeneratorResult,
    GraphNode, GraphEdge, GraphTraverser,
    NarrativeChain, ChainLink, ChainEvent, ChainEventType,
    ContextWindowIterator, Operation,
    CodeBlob, CodeExecutor,
    # Adapters
    SignalContextAdapter, GeneratorNarrativeAdapter, GraphIteratorAdapter
)


# =============================================================================
# CONFIGURATION 1: STORY ENGINE
# =============================================================================
# Combines: Narrative + Context + Signal
# Use case: Interactive fiction, game dialogue, branching stories

class StoryEngineConfig:
    """
    Story Engine configuration combining narrative chains with 
    reactive context and event-driven progression.
    
    Architecture:
    - NarrativePipeline: Story structure and progression
    - ContextPipeline: World state and player data
    - SignalPipeline: Events and triggers
    
    The signal pipeline drives story progression by emitting
    events that advance narrative chains based on context state.
    """
    
    def __init__(self):
        self.narrative = NarrativePipeline()
        self.context = ContextPipeline()
        self.signals = SignalPipeline()
        
        # Story-specific state
        self._active_story: Optional[str] = None
        self._story_observers: Dict[str, str] = {}
    
    async def initialize(self):
        """Initialize the story engine."""
        await self.signals.start()
        
        # Create world context
        world = self.context.create("world_state")
        world.update({
            'time_of_day': 'morning',
            'weather': 'clear',
            'visited_locations': []
        })
        
        # Create player context
        player = self.context.create("player")
        player.update({
            'name': 'Traveler',
            'inventory': [],
            'relationships': {},
            'flags': {}
        })
    
    def create_story(self, name: str) -> NarrativeChain:
        """Create a new story chain with context integration."""
        chain = self.narrative.create_chain(name)
        return chain
    
    def add_story_beat(self, chain: NarrativeChain,
                       text: str,
                       context_updates: Dict[str, Any] = None,
                       trigger_signal: str = None,
                       branches: List[tuple] = None):
        """
        Add a story beat that can update context and trigger signals.
        
        branches: List of (condition_func, target_chain_id) tuples
        """
        def beat_handler(ctx, data):
            result = {'text': text}
            
            # Apply context updates
            if context_updates:
                player = self.context.get("player")
                if player:
                    player.update(context_updates)
                result['context_updated'] = True
            
            return result
        
        link = ChainLink(
            name=text[:30],  # Use first 30 chars as name
            handler=beat_handler,
            data={'full_text': text}
        )
        
        # Add branch conditions
        if branches:
            for condition, target in branches:
                link.branch_conditions.append((condition, target))
        
        chain.add_link(link)
        return link
    
    async def start_story(self, chain_id: str):
        """Start a story with full context integration."""
        # Emit story start signal
        await self.signals.emit(Signal(
            signal_type="story_started",
            source="story_engine",
            payload={'chain_id': chain_id}
        ))
        
        # Get merged context
        world = self.context.get("world_state")
        player = self.context.get("player")
        
        merged_context = {}
        if world:
            merged_context.update(world.data)
        if player:
            merged_context.update(player.data)
        
        self.narrative.start_chain(chain_id, merged_context)
        self._active_story = chain_id
    
    async def advance_story(self) -> Optional[ChainEvent]:
        """Advance the story by one beat."""
        if not self._active_story:
            return None
        
        event = self.narrative.step_chain(self._active_story)
        
        if event:
            # Emit story progress signal
            await self.signals.emit(Signal(
                signal_type="story_progress",
                source="story_engine",
                payload={
                    'event': event.to_dict() if hasattr(event, 'to_dict') else str(event),
                    'link_index': event.link_index
                }
            ))
        
        return event
    
    async def shutdown(self):
        """Shutdown the story engine."""
        await self.signals.stop()


# =============================================================================
# CONFIGURATION 2: GAME WORLD
# =============================================================================
# Combines: Graph + Iterator + Context
# Use case: Game world navigation, location tracking, history

class GameWorldConfig:
    """
    Game World configuration for spatial navigation with history tracking.
    
    Architecture:
    - GraphPipeline: World map as connected locations
    - IteratorPipeline: Movement history and undo capability
    - ContextPipeline: Location-specific state
    
    The graph represents the world structure, context stores
    location data, and iterator tracks navigation history.
    """
    
    def __init__(self):
        self.graph = GraphPipeline()
        self.iterator = IteratorPipeline()
        self.context = ContextPipeline()
        
        # World-specific state
        self._current_location: Optional[str] = None
        self._movement_window: Optional[str] = None
    
    def initialize(self):
        """Initialize the game world."""
        # Create movement history window
        self._movement_window = "movement_history"
        self.iterator.create_window(self._movement_window, size=1000)
        
        # Create global world context
        world_ctx = self.context.create("world")
        world_ctx.update({
            'discovered_locations': [],
            'total_moves': 0
        })
    
    def add_location(self, location_id: str, name: str, 
                    description: str = "",
                    properties: Dict[str, Any] = None) -> GraphNode:
        """Add a location to the world."""
        node = GraphNode(
            node_id=location_id,
            node_type="location",
            data={
                'name': name,
                'description': description,
                **(properties or {})
            }
        )
        self.graph.add_node(node)
        
        # Create location-specific context
        loc_ctx = self.context.create(f"loc_{location_id}")
        loc_ctx.update({
            'items': [],
            'npcs': [],
            'visited_count': 0,
            'first_visit': None
        })
        
        return node
    
    def connect_locations(self, from_id: str, to_id: str,
                         bidirectional: bool = True,
                         edge_type: str = "path") -> List[GraphEdge]:
        """Connect two locations."""
        edges = []
        
        edge1 = GraphEdge(
            source_id=from_id,
            target_id=to_id,
            edge_type=edge_type
        )
        self.graph.add_edge(edge1)
        edges.append(edge1)
        
        if bidirectional:
            edge2 = GraphEdge(
                source_id=to_id,
                target_id=from_id,
                edge_type=edge_type
            )
            self.graph.add_edge(edge2)
            edges.append(edge2)
        
        return edges
    
    def set_current_location(self, location_id: str):
        """Set player's current location."""
        self._current_location = location_id
        
        # Update location context
        loc_ctx = self.context.get(f"loc_{location_id}")
        if loc_ctx:
            loc_ctx.set("visited_count", loc_ctx.get("visited_count", 0) + 1)
            if not loc_ctx.get("first_visit"):
                import time
                loc_ctx.set("first_visit", time.time())
    
    def move_to(self, target_id: str) -> bool:
        """
        Move to a connected location.
        
        Returns True if move was successful.
        """
        if not self._current_location:
            self._current_location = target_id
            return True
        
        # Check if target is connected
        neighbors = self.graph.get_neighbors(self._current_location)
        if target_id not in neighbors:
            return False
        
        # Record the movement
        self.iterator.record_operation(
            self._movement_window,
            operation_type="move",
            data={
                'from': self._current_location,
                'to': target_id
            }
        )
        
        # Update world context
        world_ctx = self.context.get("world")
        if world_ctx:
            moves = world_ctx.get("total_moves", 0)
            world_ctx.set("total_moves", moves + 1)
            
            discovered = world_ctx.get("discovered_locations", [])
            if target_id not in discovered:
                discovered.append(target_id)
                world_ctx.set("discovered_locations", discovered)
        
        self.set_current_location(target_id)
        return True
    
    def get_current_location(self) -> Optional[GraphNode]:
        """Get current location node."""
        if self._current_location:
            return self.graph.get_node(self._current_location)
        return None
    
    def get_available_moves(self) -> List[GraphNode]:
        """Get locations accessible from current position."""
        if not self._current_location:
            return []
        
        neighbor_ids = self.graph.get_neighbors(self._current_location)
        return [self.graph.get_node(nid) for nid in neighbor_ids 
                if self.graph.get_node(nid)]
    
    def get_movement_history(self, count: int = 10) -> List[Operation]:
        """Get recent movement history."""
        window = self.iterator.get_window(self._movement_window)
        if window:
            return window.get_recent(count)
        return []
    
    def find_path(self, from_id: str, to_id: str) -> Optional[List[str]]:
        """Find path between two locations."""
        return self.graph.find_path(from_id, to_id)


# =============================================================================
# CONFIGURATION 3: AI COLLABORATION
# =============================================================================
# Combines: Generator + Signal + Code
# Use case: AI pipeline orchestration, dynamic processing

class AICollaborationConfig:
    """
    AI Collaboration configuration for orchestrating AI processing pipelines.
    
    Architecture:
    - GeneratorPipeline: Processing stages as composable generators
    - SignalPipeline: Inter-process communication
    - CodePipeline: Dynamic processing logic
    
    Generators handle data transformation, signals coordinate
    between stages, and code allows runtime customization.
    """
    
    def __init__(self):
        self.generators = GeneratorPipeline()
        self.signals = SignalPipeline()
        self.code = CodePipeline(allowed_imports=['json', 're', 'math'])
        
        self._processing_stages: List[str] = []
    
    async def initialize(self):
        """Initialize the AI collaboration system."""
        await self.signals.start()
        
        # Register built-in processing generators
        self._register_builtin_generators()
    
    def _register_builtin_generators(self):
        """Register built-in processing generators."""
        
        # Text tokenizer
        def tokenize(text, delimiter=' '):
            if isinstance(text, str):
                for token in text.split(delimiter):
                    yield token.strip()
            elif hasattr(text, '__iter__'):
                for item in text:
                    yield from tokenize(item, delimiter)
        
        self.generators.register_factory('tokenize', tokenize)
        
        # Text filter
        def filter_text(items, min_length=1):
            for item in items if hasattr(items, '__iter__') else [items]:
                if isinstance(item, str) and len(item) >= min_length:
                    yield item
        
        self.generators.register_factory('filter', filter_text)
        
        # Transformer
        def transform(items, transform_func=str.lower):
            for item in items if hasattr(items, '__iter__') else [items]:
                yield transform_func(item)
        
        self.generators.register_factory('transform', transform)
    
    def register_custom_processor(self, name: str, source_code: str) -> CodeBlob:
        """Register a custom processing function."""
        blob = self.code.register(
            source_code=source_code,
            name=name,
            code_type="function",
            trusted=True
        )
        
        # Also register as generator factory
        def generator_wrapper(*args, **kwargs):
            func = self.code.execute(blob.blob_id)
            result = func(*args, **kwargs)
            if hasattr(result, '__iter__'):
                yield from result
            else:
                yield result
        
        self.generators.register_factory(name, generator_wrapper)
        return blob
    
    def create_pipeline(self, stages: List[Dict[str, Any]]) -> str:
        """
        Create a processing pipeline from stage definitions.
        
        Each stage: {'name': 'factory_name', 'config': {...}}
        """
        pipeline_id = f"pipeline_{uuid.uuid4()}"
        self._processing_stages.append(pipeline_id)
        
        return pipeline_id
    
    async def process(self, data: Any, 
                     stages: List[str],
                     configs: Dict[str, Dict] = None) -> List[GeneratorResult]:
        """
        Process data through specified stages.
        """
        configs = configs or {}
        results = []
        current_data = data
        
        for stage_name in stages:
            # Emit stage start signal
            await self.signals.emit(Signal(
                signal_type="stage_start",
                source="ai_collab",
                payload={'stage': stage_name}
            ))
            
            # Create and run generator
            config = configs.get(stage_name, {})
            gen = self.generators.create_generator(stage_name, config)
            gen.start(current_data)
            
            stage_results = list(gen)
            results.extend(stage_results)
            
            # Use last result as input for next stage
            if stage_results:
                current_data = stage_results[-1].value
            
            # Emit stage complete signal
            await self.signals.emit(Signal(
                signal_type="stage_complete",
                source="ai_collab",
                payload={
                    'stage': stage_name,
                    'results_count': len(stage_results)
                }
            ))
        
        return results
    
    async def shutdown(self):
        """Shutdown the AI collaboration system."""
        await self.signals.stop()


# =============================================================================
# CONFIGURATION 4: TEXT PROCESSOR
# =============================================================================
# Combines: Iterator + Generator + Code
# Use case: Document processing, text transformation

class TextProcessorConfig:
    """
    Text Processor configuration for document handling.
    
    Architecture:
    - IteratorPipeline: Document/chunk history
    - GeneratorPipeline: Text transformations
    - CodePipeline: Custom processing rules
    
    Documents are tracked as operations, generators transform
    content, and code allows custom extraction rules.
    """
    
    def __init__(self):
        self.iterator = IteratorPipeline()
        self.generators = GeneratorPipeline()
        self.code = CodePipeline(allowed_imports=['re', 'json'])
        
        self._document_window: Optional[str] = None
    
    def initialize(self):
        """Initialize text processor."""
        self._document_window = "documents"
        self.iterator.create_window(self._document_window, size=500)
        
        self._register_text_generators()
    
    def _register_text_generators(self):
        """Register text processing generators."""
        
        # Paragraph splitter
        def split_paragraphs(text):
            for para in text.split('\n\n'):
                stripped = para.strip()
                if stripped:
                    yield stripped
        
        self.generators.register_factory('paragraphs', split_paragraphs)
        
        # Sentence splitter (simple)
        def split_sentences(text):
            import re
            sentences = re.split(r'[.!?]+', text)
            for sent in sentences:
                stripped = sent.strip()
                if stripped:
                    yield stripped
        
        self.generators.register_factory('sentences', split_sentences)
        
        # Word extractor
        def extract_words(text, min_length=1):
            import re
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                if len(word) >= min_length:
                    yield word
        
        self.generators.register_factory('words', extract_words)
    
    def add_document(self, content: str, 
                    doc_type: str = "text",
                    metadata: Dict[str, Any] = None) -> str:
        """Add a document for processing."""
        return self.iterator.record_operation(
            self._document_window,
            operation_type=f"document_{doc_type}",
            data={
                'content': content,
                'length': len(content),
                'metadata': metadata or {}
            }
        )
    
    def process_document(self, operation_id: str,
                        transformations: List[str]) -> List[Any]:
        """
        Process a document through transformations.
        """
        window = self.iterator.get_window(self._document_window)
        if not window:
            return []
        
        op = window.get(operation_id)
        if not op:
            return []
        
        content = op.data.get('content', '')
        results = []
        current = content
        
        for transform_name in transformations:
            gen = self.generators.create_generator(transform_name)
            gen.start(current)
            stage_results = [r.value for r in gen]
            results.append({
                'transformation': transform_name,
                'results': stage_results
            })
            # Join results for next stage if string results
            if stage_results and isinstance(stage_results[0], str):
                current = ' '.join(stage_results)
        
        return results
    
    def register_extractor(self, name: str, pattern: str) -> CodeBlob:
        """Register a regex-based extractor."""
        source = f"""
import re
def {name}(text):
    pattern = r'{pattern}'
    matches = re.findall(pattern, text)
    for match in matches:
        yield match
{name}
"""
        blob = self.code.register(
            source_code=source,
            name=name,
            code_type="function",
            required_imports=['re'],
            trusted=True
        )
        
        # Register as generator
        def extractor_gen(text):
            func = self.code.execute(blob.blob_id)
            yield from func(text)
        
        self.generators.register_factory(name, extractor_gen)
        return blob
    
    def get_recent_documents(self, count: int = 10) -> List[Operation]:
        """Get recently processed documents."""
        window = self.iterator.get_window(self._document_window)
        if window:
            return window.get_recent(count)
        return []


# =============================================================================
# CONFIGURATION 5: EVENT SOURCING
# =============================================================================
# Combines: Signal + Context + Iterator
# Use case: Event-driven state management, audit trails

class EventSourcingConfig:
    """
    Event Sourcing configuration for event-driven state management.
    
    Architecture:
    - SignalPipeline: Event emission and subscription
    - ContextPipeline: State projections
    - IteratorPipeline: Event store
    
    Events are the source of truth, stored in iterator windows.
    Context projections are derived from events. Signals enable
    real-time subscription to events.
    """
    
    def __init__(self):
        self.signals = SignalPipeline()
        self.context = ContextPipeline()
        self.iterator = IteratorPipeline()
        
        self._event_store: Optional[str] = None
        self._projections: Dict[str, str] = {}  # projection_name -> context_id
    
    async def initialize(self):
        """Initialize event sourcing system."""
        await self.signals.start()
        
        self._event_store = "event_store"
        self.iterator.create_window(self._event_store, size=10000)
        
        # Set up event handler
        class EventStoreObserver(SignalObserver):
            def __init__(self, config: 'EventSourcingConfig'):
                super().__init__()
                self._config = config
                self.filters = set()  # Handle all events
            
            async def on_event(self, signal: Signal) -> Any:
                if signal.signal_type.startswith("domain_"):
                    self._config._store_event(signal)
                    self._config._update_projections(signal)
        
        await self.signals.register(EventStoreObserver(self))
    
    def create_projection(self, name: str, 
                         initial_state: Dict[str, Any] = None) -> str:
        """Create a state projection."""
        ctx = self.context.create(f"projection_{name}")
        ctx.update(initial_state or {})
        self._projections[name] = ctx.context_id
        return ctx.context_id
    
    def _store_event(self, signal: Signal):
        """Store event in event store."""
        self.iterator.record_operation(
            self._event_store,
            operation_type=signal.signal_type,
            data={
                'signal_id': signal.signal_id,
                'source': signal.source,
                'payload': signal.payload,
                'timestamp': signal.timestamp
            }
        )
    
    def _update_projections(self, signal: Signal):
        """Update projections based on event."""
        # This is where projection logic would go
        # Each projection has its own reducer
        pass
    
    async def emit_event(self, event_type: str, 
                        aggregate_id: str,
                        payload: Dict[str, Any]) -> Signal:
        """Emit a domain event."""
        signal = Signal(
            signal_type=f"domain_{event_type}",
            source=aggregate_id,
            payload=payload
        )
        await self.signals.emit(signal)
        return signal
    
    def get_events(self, count: int = 100) -> List[Operation]:
        """Get recent events from store."""
        window = self.iterator.get_window(self._event_store)
        if window:
            return window.get_recent(count)
        return []
    
    def get_projection(self, name: str) -> Optional[Context]:
        """Get a projection's current state."""
        ctx_id = self._projections.get(name)
        if ctx_id:
            return self.context.get(ctx_id)
        return None
    
    def rebuild_projection(self, name: str, 
                          reducer: Callable[[Dict, Signal], Dict]):
        """Rebuild a projection from events."""
        window = self.iterator.get_window(self._event_store)
        ctx = self.get_projection(name)
        
        if not window or not ctx:
            return
        
        # Reset state
        ctx._data = {}
        
        # Replay events
        for op in window:
            signal = Signal(
                signal_type=op.operation_type,
                source=op.data.get('source', ''),
                payload=op.data.get('payload', {})
            )
            new_state = reducer(ctx._data, signal)
            ctx._data = new_state
    
    async def shutdown(self):
        """Shutdown event sourcing system."""
        await self.signals.stop()


# =============================================================================
# CONFIGURATION 6: WORLD BUILDER
# =============================================================================
# Combines: Graph + Narrative + Context
# Use case: Procedural world generation, story-driven world creation

class WorldBuilderConfig:
    """
    World Builder configuration for procedural world generation.
    
    Architecture:
    - GraphPipeline: World structure (locations, connections)
    - NarrativePipeline: Generation rules as chains
    - ContextPipeline: Generation state and templates
    
    Narrative chains define generation rules, graph stores
    the generated world, and context manages generation state.
    """
    
    def __init__(self):
        self.graph = GraphPipeline()
        self.narrative = NarrativePipeline()
        self.context = ContextPipeline()
        
        self._generation_rules: Dict[str, str] = {}  # rule_name -> chain_id
    
    def initialize(self):
        """Initialize world builder."""
        # Create generation context
        gen_ctx = self.context.create("generation")
        gen_ctx.update({
            'seed': 42,
            'generated_nodes': 0,
            'generated_edges': 0,
            'templates': {}
        })
    
    def add_location_template(self, template_name: str,
                             node_type: str,
                             properties: Dict[str, Any]):
        """Add a location template."""
        gen_ctx = self.context.get("generation")
        if gen_ctx:
            templates = gen_ctx.get("templates", {})
            templates[template_name] = {
                'node_type': node_type,
                'properties': properties
            }
            gen_ctx.set("templates", templates)
    
    def create_generation_rule(self, rule_name: str) -> NarrativeChain:
        """Create a generation rule as a narrative chain."""
        chain = self.narrative.create_chain(rule_name)
        self._generation_rules[rule_name] = chain.chain_id
        return chain
    
    def add_generation_step(self, chain: NarrativeChain,
                           step_type: str,
                           config: Dict[str, Any]):
        """Add a step to a generation rule."""
        
        if step_type == "create_node":
            def handler(ctx, data):
                template_name = data.get('template', 'default')
                gen_ctx = self.context.get("generation")
                
                if gen_ctx:
                    templates = gen_ctx.get("templates", {})
                    template = templates.get(template_name, {})
                    
                    node_id = f"node_{gen_ctx.get('generated_nodes', 0)}"
                    node = GraphNode(
                        node_id=node_id,
                        node_type=template.get('node_type', 'location'),
                        data=template.get('properties', {}).copy()
                    )
                    self.graph.add_node(node)
                    
                    gen_ctx.set("generated_nodes", gen_ctx.get("generated_nodes", 0) + 1)
                    
                    return {'created_node': node_id}
                return {}
            
            chain.add_link(ChainLink(
                name=f"create_{config.get('template', 'node')}",
                handler=handler,
                data=config
            ))
        
        elif step_type == "connect":
            def handler(ctx, data):
                from_node = data.get('from') or ctx.get('last_created')
                to_node = data.get('to')
                
                if from_node and to_node:
                    edge = GraphEdge(
                        source_id=from_node,
                        target_id=to_node,
                        edge_type=data.get('edge_type', 'path')
                    )
                    self.graph.add_edge(edge)
                    
                    gen_ctx = self.context.get("generation")
                    if gen_ctx:
                        gen_ctx.set("generated_edges", 
                                   gen_ctx.get("generated_edges", 0) + 1)
                    
                    return {'created_edge': edge.edge_id}
                return {}
            
            chain.add_link(ChainLink(
                name=f"connect_{config.get('from', 'a')}_to_{config.get('to', 'b')}",
                handler=handler,
                data=config
            ))
    
    def run_generation(self, rule_name: str, 
                      initial_context: Dict[str, Any] = None) -> List[ChainEvent]:
        """Run a generation rule."""
        chain_id = self._generation_rules.get(rule_name)
        if chain_id:
            return self.narrative.run_chain(chain_id, initial_context)
        return []
    
    def get_world_stats(self) -> Dict[str, Any]:
        """Get world generation statistics."""
        gen_ctx = self.context.get("generation")
        return {
            'total_nodes': len(self.graph._nodes),
            'total_edges': len(self.graph._edges),
            'generated_nodes': gen_ctx.get("generated_nodes", 0) if gen_ctx else 0,
            'generated_edges': gen_ctx.get("generated_edges", 0) if gen_ctx else 0,
            'templates': list(gen_ctx.get("templates", {}).keys()) if gen_ctx else []
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

async def demo_configurations():
    """Demonstrate the configuration patterns."""
    print("=" * 70)
    print("PIPELINE CONFIGURATION DEMONSTRATIONS")
    print("=" * 70)
    
    # 1. Story Engine
    print("\n📖 STORY ENGINE CONFIGURATION")
    print("-" * 50)
    
    story_engine = StoryEngineConfig()
    await story_engine.initialize()
    
    # Create a simple quest
    quest = story_engine.create_story("the_quest")
    story_engine.add_story_beat(
        quest, 
        "You awaken in a mysterious forest...",
        context_updates={'location': 'forest'}
    )
    story_engine.add_story_beat(
        quest,
        "A path leads north to a village.",
        context_updates={'knows_village': True}
    )
    story_engine.add_story_beat(
        quest,
        "You arrive at the village gates.",
        context_updates={'location': 'village'}
    )
    
    await story_engine.start_story(quest.chain_id)
    
    while True:
        event = await story_engine.advance_story()
        if not event:
            break
        print(f"  Beat: {event.data.get('result', {}).get('text', '?')[:50]}...")
    
    await story_engine.shutdown()
    print("  ✓ Story engine completed")
    
    # 2. Game World
    print("\n🗺️ GAME WORLD CONFIGURATION")
    print("-" * 50)
    
    world = GameWorldConfig()
    world.initialize()
    
    # Build world
    world.add_location("town", "Town Square", "A bustling town center")
    world.add_location("tavern", "The Rusty Nail", "A cozy tavern")
    world.add_location("shop", "General Store", "Various goods for sale")
    world.add_location("gate", "Town Gate", "The exit to the wilderness")
    
    world.connect_locations("town", "tavern")
    world.connect_locations("town", "shop")
    world.connect_locations("town", "gate")
    
    # Navigate
    world.set_current_location("town")
    print(f"  Starting at: {world.get_current_location().data['name']}")
    
    world.move_to("tavern")
    print(f"  Moved to: {world.get_current_location().data['name']}")
    
    world.move_to("town")
    world.move_to("gate")
    print(f"  Current: {world.get_current_location().data['name']}")
    
    history = world.get_movement_history(3)
    print(f"  Movement history: {[op.data for op in history]}")
    print("  ✓ Game world completed")
    
    # 3. AI Collaboration
    print("\n🤖 AI COLLABORATION CONFIGURATION")
    print("-" * 50)
    
    ai_collab = AICollaborationConfig()
    await ai_collab.initialize()
    
    # Process some text
    text = "Hello World! This is a test. Processing with AI collaboration."
    results = await ai_collab.process(
        text,
        stages=['tokenize', 'filter', 'transform'],
        configs={'filter': {'min_length': 3}}
    )
    
    print(f"  Input: '{text[:40]}...'")
    print(f"  Results: {[r.value for r in results[-5:]]}")
    
    await ai_collab.shutdown()
    print("  ✓ AI collaboration completed")
    
    # 4. Text Processor
    print("\n📝 TEXT PROCESSOR CONFIGURATION")
    print("-" * 50)
    
    text_proc = TextProcessorConfig()
    text_proc.initialize()
    
    doc_id = text_proc.add_document(
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs.",
        doc_type="sample"
    )
    
    results = text_proc.process_document(doc_id, ['sentences', 'words'])
    for stage in results:
        print(f"  {stage['transformation']}: {len(stage['results'])} items")
    
    print("  ✓ Text processor completed")
    
    # 5. Event Sourcing
    print("\n📊 EVENT SOURCING CONFIGURATION")
    print("-" * 50)
    
    event_sourcing = EventSourcingConfig()
    await event_sourcing.initialize()
    
    event_sourcing.create_projection("order_count", {"total": 0})
    
    await event_sourcing.emit_event("order_placed", "order-123", {"amount": 100})
    await event_sourcing.emit_event("order_shipped", "order-123", {"carrier": "FedEx"})
    
    events = event_sourcing.get_events(2)
    print(f"  Stored events: {[op.operation_type for op in events]}")
    
    await event_sourcing.shutdown()
    print("  ✓ Event sourcing completed")
    
    # 6. World Builder
    print("\n🏗️ WORLD BUILDER CONFIGURATION")
    print("-" * 50)
    
    builder = WorldBuilderConfig()
    builder.initialize()
    
    # Add templates
    builder.add_location_template("village", "settlement", {
        "population": "small",
        "has_tavern": True
    })
    builder.add_location_template("forest", "wilderness", {
        "danger_level": "low",
        "resources": ["wood", "herbs"]
    })
    
    # Create generation rule
    rule = builder.create_generation_rule("create_area")
    builder.add_generation_step(rule, "create_node", {"template": "village"})
    builder.add_generation_step(rule, "create_node", {"template": "forest"})
    
    # Run generation
    events = builder.run_generation("create_area")
    
    stats = builder.get_world_stats()
    print(f"  World stats: {stats}")
    print("  ✓ World builder completed")
    
    print("\n" + "=" * 70)
    print("ALL CONFIGURATIONS DEMONSTRATED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo_configurations())
