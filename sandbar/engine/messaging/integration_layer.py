#!/usr/bin/env python3
"""
Integration Layer: Listener + Graph Thinking + Entity Pipeline

Connects:
- Listener (routes signals to observer OR event loop)
- Connector (type-safe wiring)
- MessagingInterface (buffered I/O)
- Graph (the "thinking" part - where entities connect and agents build)

Assimilated from islands/integration_layer.py; imports from .connector_core and .messaging_interface.
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

# Import our components (relative for sandbar package)
from .connector_core import Connector, Connection, Listener, RouteTarget, make_connector
from .messaging_interface import (
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
    """

    def __init__(self, graph_id: str = None):
        self.graph_id = graph_id or f"graph_{uuid.uuid4().hex[:8]}"
        self.nodes: Dict[str, GraphNode] = {}
        self.agents: Dict[str, 'Agent'] = {}
        self.filters: List[Callable[[GraphNode], bool]] = []
        self.result_queue: deque = deque()

    def add_entity(self, entity: Entity) -> GraphNode:
        node = GraphNode(node_id=entity.entity_id, node_type="entity", data=entity)
        self.nodes[node.node_id] = node
        return node

    def connect_entities(self, id_a: str, id_b: str):
        if id_a in self.nodes and id_b in self.nodes:
            self.nodes[id_a].edges.add(id_b)
            self.nodes[id_b].edges.add(id_a)

    def identify_entity(self, properties: Dict[str, Any]) -> Optional[Entity]:
        for node in self.nodes.values():
            if node.node_type == "entity" and node.data:
                entity = node.data
                matches = sum(
                    1 for k, v in properties.items()
                    if entity.properties.get(k) == v
                )
                if matches >= len(properties) * 0.7:
                    return entity
        entity = Entity(
            entity_type=properties.get('type', 'unknown'),
            name=properties.get('name', ''),
            properties=properties
        )
        self.add_entity(entity)
        return entity

    def build_agent(
        self,
        name: str,
        entity_ids: List[str],
        behavior: Callable[[List[Entity], Any], Any] = None
    ) -> 'Agent':
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
        agent_node = GraphNode(
            node_id=agent.agent_id,
            node_type="agent",
            data=agent,
            edges=set(entity_ids)
        )
        self.nodes[agent.agent_id] = agent_node
        self.agents[agent.agent_id] = agent
        return agent

    def receive_result(self, result: Dict[str, Any]):
        self.result_queue.append(result)

    def process_results(self) -> List[Any]:
        processed = []
        while self.result_queue:
            result = self.result_queue.popleft()
            if 'entities' in result:
                for ent_data in result['entities']:
                    entity = self.identify_entity(ent_data)
                    processed.append(('entity_identified', entity))
            if 'relationships' in result:
                for rel in result['relationships']:
                    self.connect_entities(rel['source'], rel['target'])
                    processed.append(('connected', rel))
            if 'updates' in result:
                for update in result['updates']:
                    entity_id = update.get('entity_id')
                    if entity_id in self.nodes:
                        entity = self.nodes[entity_id].data
                        entity.properties.update(update.get('properties', {}))
                        processed.append(('updated', entity))
        return processed

    def add_filter(self, filter_fn: Callable[[GraphNode], bool]):
        self.filters.append(filter_fn)

    def filter_nodes(self, nodes: List[GraphNode] = None) -> List[GraphNode]:
        nodes = nodes or list(self.nodes.values())
        for f in self.filters:
            nodes = [n for n in nodes if f(n)]
        return nodes

    def traverse(
        self,
        start_id: str,
        visit: Callable[[GraphNode], Any],
        max_depth: int = 10
    ) -> List[Any]:
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
            if self.filters and not all(f(node) for f in self.filters):
                continue
            result = visit(node)
            if result is not None:
                results.append(result)
            for edge_id in node.edges:
                if edge_id not in visited:
                    queue.append((edge_id, depth + 1))
        return results


@dataclass
class Agent:
    agent_id: str
    name: str
    entities: List[Entity] = field(default_factory=list)
    behavior: Optional[Callable[[List[Entity], Any], Any]] = None
    state: Dict[str, Any] = field(default_factory=dict)

    def act(self, input_data: Any = None) -> Any:
        if self.behavior:
            return self.behavior(self.entities, input_data)
        return None

    def get_entity_ids(self) -> List[str]:
        return [e.entity_id for e in self.entities]


class GameOrchestrator:
    """Orchestrates MessagingInterface, Listener, Connector, ThinkingGraph."""

    def __init__(self):
        self.messaging = MessagingInterface()
        self.listener = Listener()
        self.connector = make_connector()
        self.graph = ThinkingGraph()
        self.event_loop_queue: deque = deque()
        self.connections: Dict[str, Connection] = {}
        self._setup_pipeline()

    def _setup_pipeline(self):
        def slash_parser(prompt: str) -> CommandSignal:
            parts = prompt[1:].split()
            return CommandSignal(
                signal_type="slash_command",
                command=parts[0] if parts else "",
                args=tuple(parts[1:])
            )
        self.messaging.signal_factory.register_parser("/", slash_parser)

        class SignalProcessor:
            def __init__(sp_self, orchestrator):
                sp_self.orchestrator = orchestrator
            def emit(sp_self, signal: CommandSignal):
                return sp_self.orchestrator._process_signal(signal)
        self.messaging.add_signal_target(SignalProcessor(self))
        self.listener.add_route("slash_command", RouteTarget.OBSERVER)
        self.listener.add_route("query", RouteTarget.EVENT_LOOP)
        self.listener.default_route = RouteTarget.OBSERVER

    def _process_signal(self, signal: CommandSignal):
        target = self.listener.receive(signal)
        if target == RouteTarget.EVENT_LOOP:
            self.event_loop_queue.append(signal)
            return {'queued': True, 'target': 'event_loop'}
        return self._handle_signal(signal)

    def _handle_signal(self, signal: CommandSignal) -> Dict[str, Any]:
        result = {'signal': signal.command, 'handled': False}
        command = signal.command.lower()
        if command == 'create' and signal.args:
            entity_type = signal.args[0]
            name = signal.args[1] if len(signal.args) > 1 else ""
            entity = self.graph.identify_entity({'type': entity_type, 'name': name})
            result['entity'] = entity.entity_id
            result['handled'] = True
        elif command == 'connect' and len(signal.args) >= 2:
            self.graph.connect_entities(signal.args[0], signal.args[1])
            result['handled'] = True
        elif command == 'agent' and len(signal.args) >= 2:
            name = signal.args[0]
            entity_ids = list(signal.args[1:])
            agent = self.graph.build_agent(name, entity_ids)
            result['agent'] = agent.agent_id
            result['handled'] = True
        elif command == 'result' and signal.kwargs:
            self.graph.receive_result(signal.kwargs)
            result['handled'] = True
        return result

    def process_event_loop(self) -> List[Dict[str, Any]]:
        results = []
        while self.event_loop_queue:
            signal = self.event_loop_queue.popleft()
            result = self._handle_signal(signal)
            results.append(result)
        return results

    def update(self) -> Dict[str, Any]:
        msg_results = self.messaging.update()
        loop_results = self.process_event_loop()
        graph_results = self.graph.process_results()
        return {
            'messaging': msg_results,
            'event_loop': len(loop_results),
            'graph_updates': len(graph_results)
        }

    def command(self, cmd: str):
        self.messaging.receive_prompt(f"/{cmd}" if not cmd.startswith("/") else cmd)

    def inject_llm_result(self, result: Dict[str, Any]):
        self.graph.receive_result(result)
