#!/usr/bin/env python3
"""
Integration Layer: Listener + Graph Thinking + Entity Pipeline

Connects:
- Listener (routes signals to observer OR event loop)
- Connector (type-safe wiring)
- MessagingInterface (buffered I/O)
- Graph (the "thinking" part - where entities connect and agents build)

This is the orchestration layer.
"""

from __future__ import annotations
import asyncio
import uuid
from typing import (
    TypeVar, Generic, Callable, Any, Union, Type, Protocol,
    Dict, List, Optional, Set, Tuple
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from collections import deque
from abc import ABC, abstractmethod

# Import our components
from connector_core import Connector, Connection, Listener, RouteTarget, make_connector
from messaging_interface import (
    MessagingInterface, MessageBuffer, SignalFactory, 
    CommandSignal, IOBridge, ConnectorBridge
)


# =============================================================================
# ENTITY - Base for all game/world entities
# =============================================================================

@dataclass
class Entity:
    """Base entity that can be connected in the graph."""
    entity_id: str = field(default_factory=lambda: f"ent_{uuid.uuid4().hex[:8]}")
    entity_type: str = "generic"
    name: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Graph connections
    connections: Set[str] = field(default_factory=set)  # Other entity IDs
    
    def connect_to(self, other: Entity):
        """Bidirectional connection."""
        self.connections.add(other.entity_id)
        other.connections.add(self.entity_id)


# =============================================================================
# THINKING GRAPH - The cognitive layer
# =============================================================================

@dataclass
class GraphNode:
    """Node in the thinking graph."""
    node_id: str
    node_type: str  # "entity", "agent", "filter", "result"
    data: Any = None
    edges: Set[str] = field(default_factory=set)


class ThinkingGraph:
    """
    The "thinking part" - a cognitive graph where:
    - Entities are identified and connected
    - Agents are constructed from entity clusters
    - LLM results modify graph structure
    - Filters gate what gets processed
    
    This IS the brain of the system.
    """
    
    def __init__(self, graph_id: str = None):
        self.graph_id = graph_id or f"graph_{uuid.uuid4().hex[:8]}"
        self.nodes: Dict[str, GraphNode] = {}
        self.agents: Dict[str, 'Agent'] = {}
        self.filters: List[Callable[[GraphNode], bool]] = []
        
        # Results from LLM/processing feed back here
        self.result_queue: deque = deque()
        
    # -------------------------------------------------------------------------
    # Entity Operations
    # -------------------------------------------------------------------------
    
    def add_entity(self, entity: Entity) -> GraphNode:
        """Add entity to graph as a node."""
        node = GraphNode(
            node_id=entity.entity_id,
            node_type="entity",
            data=entity
        )
        self.nodes[node.node_id] = node
        return node
    
    def connect_entities(self, id_a: str, id_b: str):
        """Connect two entity nodes."""
        if id_a in self.nodes and id_b in self.nodes:
            self.nodes[id_a].edges.add(id_b)
            self.nodes[id_b].edges.add(id_a)
    
    def identify_entity(self, properties: Dict[str, Any]) -> Optional[Entity]:
        """
        Entity identification: Find or create entity matching properties.
        This is core to the "we scan text and construct world" philosophy.
        """
        # Search existing entities
        for node in self.nodes.values():
            if node.node_type == "entity" and node.data:
                entity = node.data
                # Simple property matching - could be semantic
                matches = sum(
                    1 for k, v in properties.items()
                    if entity.properties.get(k) == v
                )
                if matches >= len(properties) * 0.7:  # 70% match threshold
                    return entity
        
        # Create new entity
        entity = Entity(
            entity_type=properties.get('type', 'unknown'),
            name=properties.get('name', ''),
            properties=properties
        )
        self.add_entity(entity)
        return entity
    
    # -------------------------------------------------------------------------
    # Agent Construction
    # -------------------------------------------------------------------------
    
    def build_agent(
        self, 
        name: str,
        entity_ids: List[str],
        behavior: Callable[[List[Entity], Any], Any] = None
    ) -> Agent:
        """
        Build an agent from a cluster of entities.
        Agents are constructed ON the graph from entity relationships.
        """
        entities = [
            self.nodes[eid].data 
            for eid in entity_ids 
            if eid in self.nodes and self.nodes[eid].node_type == "entity"
        ]
        
        agent = Agent(
            agent_id=f"agent_{uuid.uuid4().hex[:8]}",
            name=name,
            entities=entities,
            behavior=behavior
        )
        
        # Add agent as a node
        agent_node = GraphNode(
            node_id=agent.agent_id,
            node_type="agent",
            data=agent,
            edges=set(entity_ids)
        )
        self.nodes[agent.agent_id] = agent_node
        self.agents[agent.agent_id] = agent
        
        return agent
    
    # -------------------------------------------------------------------------
    # LLM Result Integration
    # -------------------------------------------------------------------------
    
    def receive_result(self, result: Dict[str, Any]):
        """
        Receive LLM/processing result and integrate into graph.
        Results can:
        - Create new entities
        - Modify existing entities
        - Create new connections
        - Trigger agent behaviors
        """
        self.result_queue.append(result)
    
    def process_results(self) -> List[Any]:
        """Process queued results, modifying graph structure."""
        processed = []
        
        while self.result_queue:
            result = self.result_queue.popleft()
            
            # Entity extraction results
            if 'entities' in result:
                for ent_data in result['entities']:
                    entity = self.identify_entity(ent_data)
                    processed.append(('entity_identified', entity))
            
            # Relationship results
            if 'relationships' in result:
                for rel in result['relationships']:
                    self.connect_entities(rel['source'], rel['target'])
                    processed.append(('connected', rel))
            
            # Property updates
            if 'updates' in result:
                for update in result['updates']:
                    entity_id = update.get('entity_id')
                    if entity_id in self.nodes:
                        entity = self.nodes[entity_id].data
                        entity.properties.update(update.get('properties', {}))
                        processed.append(('updated', entity))
        
        return processed
    
    # -------------------------------------------------------------------------
    # Filters
    # -------------------------------------------------------------------------
    
    def add_filter(self, filter_fn: Callable[[GraphNode], bool]):
        """Add filter that gates node processing."""
        self.filters.append(filter_fn)
    
    def filter_nodes(self, nodes: List[GraphNode] = None) -> List[GraphNode]:
        """Apply filters to nodes."""
        nodes = nodes or list(self.nodes.values())
        for f in self.filters:
            nodes = [n for n in nodes if f(n)]
        return nodes
    
    # -------------------------------------------------------------------------
    # Traversal
    # -------------------------------------------------------------------------
    
    def traverse(
        self, 
        start_id: str, 
        visit: Callable[[GraphNode], Any],
        max_depth: int = 10
    ) -> List[Any]:
        """BFS traversal from start node, applying visit function."""
        if start_id not in self.nodes:
            return []
        
        visited = set()
        queue = deque([(start_id, 0)])
        results = []
        
        while queue:
            node_id, depth = queue.popleft()
            
            if node_id in visited or depth > max_depth:
                continue
            
            visited.add(node_id)
            node = self.nodes[node_id]
            
            # Apply filters
            if self.filters and not all(f(node) for f in self.filters):
                continue
            
            # Visit
            result = visit(node)
            if result is not None:
                results.append(result)
            
            # Queue neighbors
            for edge_id in node.edges:
                if edge_id not in visited:
                    queue.append((edge_id, depth + 1))
        
        return results


# =============================================================================
# AGENT - Built from entity clusters
# =============================================================================

@dataclass
class Agent:
    """
    An agent constructed from entities.
    Has behavior that operates on its entity cluster.
    """
    agent_id: str
    name: str
    entities: List[Entity] = field(default_factory=list)
    behavior: Optional[Callable[[List[Entity], Any], Any]] = None
    state: Dict[str, Any] = field(default_factory=dict)
    
    def act(self, input_data: Any = None) -> Any:
        """Execute agent behavior on entity cluster."""
        if self.behavior:
            return self.behavior(self.entities, input_data)
        return None
    
    def get_entity_ids(self) -> List[str]:
        return [e.entity_id for e in self.entities]


# =============================================================================
# ORCHESTRATOR - Ties everything together
# =============================================================================

class GameOrchestrator:
    """
    Orchestrates the full pipeline:
    
    Input → MessagingInterface → Listener → [Observer | EventLoop]
                                    ↓
                            SignalFactory
                                    ↓
                              Connector
                                    ↓
                            ThinkingGraph
                                    ↓
                         Entity/Agent/Result
    """
    
    def __init__(self):
        # Core components
        self.messaging = MessagingInterface()
        self.listener = Listener()
        self.connector = make_connector()
        self.graph = ThinkingGraph()
        
        # Event loop queue (alternative to immediate observer)
        self.event_loop_queue: deque = deque()
        
        # Connections registry
        self.connections: Dict[str, Connection] = {}
        
        # Wire up the pipeline
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Wire all components together."""
        
        # Register slash command parser
        def slash_parser(prompt: str) -> CommandSignal:
            """Parse /command args format."""
            parts = prompt[1:].split()  # Skip the /
            return CommandSignal(
                signal_type="slash_command",
                command=parts[0] if parts else "",
                args=tuple(parts[1:])
            )
        
        self.messaging.signal_factory.register_parser("/", slash_parser)
        
        # Bridge messaging to our signal processor
        class SignalProcessor:
            def __init__(sp_self, orchestrator):
                sp_self.orchestrator = orchestrator
            
            def emit(sp_self, signal: CommandSignal):
                return sp_self.orchestrator._process_signal(signal)
        
        self.messaging.add_signal_target(SignalProcessor(self))
        
        # Configure listener routing
        # Commands go to observer (immediate), queries to event loop (batched)
        self.listener.add_route("slash_command", RouteTarget.OBSERVER)
        self.listener.add_route("query", RouteTarget.EVENT_LOOP)
        self.listener.default_route = RouteTarget.OBSERVER
    
    def _process_signal(self, signal: CommandSignal):
        """Process a signal through the pipeline."""
        # Route through listener
        target = self.listener.receive(signal)
        
        if target == RouteTarget.EVENT_LOOP:
            # Queue for batch processing
            self.event_loop_queue.append(signal)
            return {'queued': True, 'target': 'event_loop'}
        
        # Immediate observer processing
        return self._handle_signal(signal)
    
    def _handle_signal(self, signal: CommandSignal) -> Dict[str, Any]:
        """Handle signal immediately (observer path)."""
        result = {'signal': signal.command, 'handled': False}
        
        # Command dispatch
        command = signal.command.lower()
        
        if command == 'create':
            # Create entity from args
            if signal.args:
                entity_type = signal.args[0]
                name = signal.args[1] if len(signal.args) > 1 else ""
                entity = self.graph.identify_entity({
                    'type': entity_type,
                    'name': name
                })
                result['entity'] = entity.entity_id
                result['handled'] = True
        
        elif command == 'connect':
            # Connect two entities
            if len(signal.args) >= 2:
                self.graph.connect_entities(signal.args[0], signal.args[1])
                result['handled'] = True
        
        elif command == 'agent':
            # Build agent from entities
            if len(signal.args) >= 2:
                name = signal.args[0]
                entity_ids = list(signal.args[1:])
                agent = self.graph.build_agent(name, entity_ids)
                result['agent'] = agent.agent_id
                result['handled'] = True
        
        elif command == 'result':
            # Inject LLM result
            if signal.kwargs:
                self.graph.receive_result(signal.kwargs)
                result['handled'] = True
        
        return result
    
    def process_event_loop(self) -> List[Dict[str, Any]]:
        """Process all queued event loop signals."""
        results = []
        while self.event_loop_queue:
            signal = self.event_loop_queue.popleft()
            result = self._handle_signal(signal)
            results.append(result)
        return results
    
    def update(self) -> Dict[str, Any]:
        """Full update cycle."""
        # 1. Process messaging buffer
        msg_results = self.messaging.update()
        
        # 2. Process event loop queue
        loop_results = self.process_event_loop()
        
        # 3. Process graph results (from LLM, etc.)
        graph_results = self.graph.process_results()
        
        return {
            'messaging': msg_results,
            'event_loop': len(loop_results),
            'graph_updates': len(graph_results)
        }
    
    # -------------------------------------------------------------------------
    # Convenience API
    # -------------------------------------------------------------------------
    
    def command(self, cmd: str):
        """Send command string."""
        self.messaging.receive_prompt(f"/{cmd}" if not cmd.startswith("/") else cmd)
    
    def inject_llm_result(self, result: Dict[str, Any]):
        """Inject LLM processing result into graph."""
        self.graph.receive_result(result)


# =============================================================================
# EXAMPLE: World Building Flow
# =============================================================================

if __name__ == "__main__":
    # Create orchestrator
    game = GameOrchestrator()
    
    print("=== World Building Demo ===\n")
    
    # Send commands to create entities
    game.command("create character Hero")
    game.command("create location TownSquare")
    game.command("create item Sword")
    
    # Process
    game.update()
    
    print(f"Graph nodes: {len(game.graph.nodes)}")
    for node_id, node in game.graph.nodes.items():
        if node.node_type == "entity":
            ent = node.data
            print(f"  - {ent.entity_type}: {ent.name} ({ent.entity_id})")
    
    # Simulate LLM result with entity extraction
    game.inject_llm_result({
        'entities': [
            {'type': 'npc', 'name': 'Blacksmith', 'occupation': 'smith'},
            {'type': 'item', 'name': 'Iron Ore', 'material': 'iron'}
        ],
        'relationships': []
    })
    
    # Process results
    game.update()
    
    print(f"\nAfter LLM result: {len(game.graph.nodes)} nodes")
    
    # Build an agent from entities
    entity_ids = [n.node_id for n in game.graph.nodes.values() if n.node_type == "entity"][:3]
    if len(entity_ids) >= 2:
        game.command(f"agent TownNPCs {' '.join(entity_ids[:2])}")
        game.update()
        
        print(f"\nAgents: {list(game.graph.agents.keys())}")
    
    # Traverse graph
    print("\n=== Graph Traversal ===")
    if entity_ids:
        results = game.graph.traverse(
            entity_ids[0],
            lambda node: f"{node.node_type}: {node.data.name if hasattr(node.data, 'name') else node.node_id}"
        )
        for r in results:
            print(f"  Visited: {r}")
