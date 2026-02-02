"""
Pine World System
=================

World construction from text - the WhiteRoom pattern.

"We don't author puzzles - we scan text, analyze every word,
 and construct a world from what we find."

Features:
- WorldNode/WorldEdge graph structure
- WhiteRoom: explorable world built from text
- WhiteRoomBuilder: fluent API for world construction

SOURCE: Mechanics/everything/world.py (22KB)
Copy the full implementation from that file.

Quick Reference:
----------------
    from . import text_to_world, WhiteRoom

    # Build world from text
    world = text_to_world('''
        The lighthouse stands on the cliff edge.
        Inside, a spiral staircase leads up.
        At the top, the great lamp waits.
    ''')

    # Explore
    print(world.summary())
    for node in world.nodes:
        print(f"  - {node.name}: {node.description}")
"""

from typing import Any, Dict, List, Optional, Set, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
import uuid

from ..core.primitives import FragmentCategory, Identified


# =============================================================================
#                              ENUMS
# =============================================================================

class NodeType(Enum):
    """Types of nodes in the world graph."""
    LOCATION = auto()    # A place
    ITEM = auto()        # An object
    CHARACTER = auto()   # A person/creature
    CONCEPT = auto()     # Abstract concept
    EVENT = auto()       # Something that happens


class EdgeType(Enum):
    """Types of edges connecting nodes."""
    CONTAINS = auto()      # A contains B
    CONNECTS = auto()      # A connects to B (bidirectional)
    LEADS_TO = auto()      # A leads to B (directional)
    BELONGS_TO = auto()    # A belongs to B
    TRIGGERS = auto()      # A triggers B
    REQUIRES = auto()      # A requires B


# =============================================================================
#                              WORLD NODE
# =============================================================================

@dataclass
class WorldNode(Identified):
    """
    A node in the world graph.

    Represents a location, item, character, or concept
    extracted from text.
    """
    name: str = ""
    node_type: NodeType = NodeType.LOCATION
    description: str = ""
    fragments: Dict[FragmentCategory, List[str]] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def add_fragment(self, category: FragmentCategory, text: str) -> None:
        """Add a prose fragment to this node."""
        if category not in self.fragments:
            self.fragments[category] = []
        self.fragments[category].append(text)

    def get_description(self, include_categories: List[FragmentCategory] = None) -> str:
        """Get composed description from fragments."""
        if include_categories is None:
            include_categories = [FragmentCategory.BASE_DESCRIPTION]

        parts = []
        for category in include_categories:
            if category in self.fragments:
                parts.extend(self.fragments[category])

        return " ".join(parts) if parts else self.description

    def has_tag(self, tag: str) -> bool:
        """Check if node has a tag."""
        return tag in self.tags


# =============================================================================
#                              WORLD EDGE
# =============================================================================

@dataclass
class WorldEdge:
    """
    An edge connecting two nodes in the world graph.
    """
    source_id: str
    target_id: str
    edge_type: EdgeType
    label: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    bidirectional: bool = False

    @property
    def id(self) -> str:
        return f"{self.source_id}->{self.target_id}:{self.edge_type.name}"


# =============================================================================
#                              WHITE ROOM
# =============================================================================

class WhiteRoom:
    """
    An explorable world built from text.

    The WhiteRoom is a graph of WorldNodes connected by WorldEdges,
    with an origin point for exploration.

    Named for the writing technique of starting in an empty room
    and filling it with details from the text.

    TODO: Copy full implementation from Mechanics/everything/world.py
    """

    def __init__(self, name: str = "World"):
        self.name = name
        self._nodes: Dict[str, WorldNode] = {}
        self._edges: List[WorldEdge] = []
        self._origin_id: Optional[str] = None

    @property
    def origin(self) -> Optional[WorldNode]:
        """Get the origin/starting node."""
        if self._origin_id:
            return self._nodes.get(self._origin_id)
        return None

    @property
    def nodes(self) -> List[WorldNode]:
        """Get all nodes."""
        return list(self._nodes.values())

    @property
    def edges(self) -> List[WorldEdge]:
        """Get all edges."""
        return list(self._edges)

    def add_node(self, node: WorldNode) -> None:
        """Add a node to the world."""
        self._nodes[node.id] = node
        if self._origin_id is None:
            self._origin_id = node.id

    def add_edge(self, edge: WorldEdge) -> None:
        """Add an edge to the world."""
        self._edges.append(edge)

    def get_node(self, node_id: str) -> Optional[WorldNode]:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def get_node_by_name(self, name: str) -> Optional[WorldNode]:
        """Get node by name (case-insensitive)."""
        name_lower = name.lower()
        for node in self._nodes.values():
            if node.name.lower() == name_lower:
                return node
        return None

    def set_origin(self, node_or_id: str | WorldNode) -> None:
        """Set the origin/starting node."""
        if isinstance(node_or_id, WorldNode):
            self._origin_id = node_or_id.id
        else:
            self._origin_id = node_or_id

    def get_neighbors(self, node_id: str) -> List[WorldNode]:
        """Get all nodes connected to the given node."""
        neighbors = []
        for edge in self._edges:
            if edge.source_id == node_id:
                neighbor = self._nodes.get(edge.target_id)
                if neighbor:
                    neighbors.append(neighbor)
            elif edge.bidirectional and edge.target_id == node_id:
                neighbor = self._nodes.get(edge.source_id)
                if neighbor:
                    neighbors.append(neighbor)
        return neighbors

    def get_contents(self, node_id: str) -> List[WorldNode]:
        """Get nodes contained within the given node."""
        contents = []
        for edge in self._edges:
            if edge.source_id == node_id and edge.edge_type == EdgeType.CONTAINS:
                content = self._nodes.get(edge.target_id)
                if content:
                    contents.append(content)
        return contents

    def summary(self) -> str:
        """Get a summary of the world."""
        lines = [
            f"=== {self.name} ===",
            f"Nodes: {len(self._nodes)}",
            f"Edges: {len(self._edges)}",
            f"Origin: {self.origin.name if self.origin else 'None'}",
            "",
            "Locations:"
        ]
        for node in self._nodes.values():
            if node.node_type == NodeType.LOCATION:
                lines.append(f"  - {node.name}")
        return "\n".join(lines)


# =============================================================================
#                              BUILDER
# =============================================================================

class WhiteRoomBuilder:
    """
    Fluent builder for constructing WhiteRoom worlds.

    Usage:
        world = (WhiteRoomBuilder("Lighthouse")
            .location("entrance", "The lighthouse entrance")
            .location("staircase", "A spiral staircase")
            .connect("entrance", "staircase")
            .item("key", "A brass key", in_location="entrance")
            .build())
    """

    def __init__(self, name: str = "World"):
        self._world = WhiteRoom(name)

    def location(
        self,
        name: str,
        description: str = "",
        **properties
    ) -> 'WhiteRoomBuilder':
        """Add a location node."""
        node = WorldNode(
            name=name,
            node_type=NodeType.LOCATION,
            description=description,
            properties=properties
        )
        self._world.add_node(node)
        return self

    def item(
        self,
        name: str,
        description: str = "",
        in_location: str = None,
        **properties
    ) -> 'WhiteRoomBuilder':
        """Add an item node, optionally in a location."""
        node = WorldNode(
            name=name,
            node_type=NodeType.ITEM,
            description=description,
            properties=properties
        )
        self._world.add_node(node)

        if in_location:
            location = self._world.get_node_by_name(in_location)
            if location:
                edge = WorldEdge(
                    source_id=location.id,
                    target_id=node.id,
                    edge_type=EdgeType.CONTAINS
                )
                self._world.add_edge(edge)

        return self

    def character(
        self,
        name: str,
        description: str = "",
        in_location: str = None,
        **properties
    ) -> 'WhiteRoomBuilder':
        """Add a character node."""
        node = WorldNode(
            name=name,
            node_type=NodeType.CHARACTER,
            description=description,
            properties=properties
        )
        self._world.add_node(node)

        if in_location:
            location = self._world.get_node_by_name(in_location)
            if location:
                edge = WorldEdge(
                    source_id=location.id,
                    target_id=node.id,
                    edge_type=EdgeType.CONTAINS
                )
                self._world.add_edge(edge)

        return self

    def connect(
        self,
        from_name: str,
        to_name: str,
        bidirectional: bool = True
    ) -> 'WhiteRoomBuilder':
        """Connect two locations."""
        from_node = self._world.get_node_by_name(from_name)
        to_node = self._world.get_node_by_name(to_name)

        if from_node and to_node:
            edge = WorldEdge(
                source_id=from_node.id,
                target_id=to_node.id,
                edge_type=EdgeType.CONNECTS,
                bidirectional=bidirectional
            )
            self._world.add_edge(edge)

        return self

    def origin(self, name: str) -> 'WhiteRoomBuilder':
        """Set the origin/starting location."""
        node = self._world.get_node_by_name(name)
        if node:
            self._world.set_origin(node)
        return self

    def build(self) -> WhiteRoom:
        """Build and return the WhiteRoom."""
        return self._world


# =============================================================================
#                              FACTORY FUNCTIONS
# =============================================================================

def build_world(name: str = "World") -> WhiteRoomBuilder:
    """Start building a world."""
    return WhiteRoomBuilder(name)


def text_to_world(
    text: str,
    embedding_store = None,
    name: str = "World"
) -> WhiteRoom:
    """
    Transform text into an explorable WhiteRoom world.

    This is the main entry point for the MATTS pipeline:
    1. Extract entities from text
    2. Build graph of relationships
    3. Construct WhiteRoom world

    TODO: Integrate with extraction.py for full pipeline
    """
    from .extraction import extract_text

    # Extract entities and relations
    result = extract_text(text)

    # Build world from extraction
    builder = WhiteRoomBuilder(name)

    # Add locations
    for entity in result.entities:
        if entity.entity_type.lower() in ('location', 'place', 'room'):
            builder.location(entity.name, entity.description)
        elif entity.entity_type.lower() in ('item', 'object', 'thing'):
            builder.item(entity.name, entity.description)
        elif entity.entity_type.lower() in ('person', 'character', 'creature'):
            builder.character(entity.name, entity.description)
        else:
            # Default to location
            builder.location(entity.name, entity.description)

    # Add relationships as edges
    for relation in result.relations:
        # TODO: Map relation types to edge types
        pass

    return builder.build()
