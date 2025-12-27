"""
Pine Graph Edges
================

Graph edge structures and registry.

SOURCE: Mechanics/everything/graph_core.py (31KB)
Copy the full implementation from that file.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import uuid


class RelationshipType(Enum):
    """Types of relationships between nodes."""
    # Spatial
    CONTAINS = auto()      # A contains B
    CONNECTS = auto()      # A connects to B
    LEADS_TO = auto()      # A leads to B

    # Ownership
    BELONGS_TO = auto()    # A belongs to B
    OWNED_BY = auto()      # A is owned by B

    # Causal
    TRIGGERS = auto()      # A triggers B
    REQUIRES = auto()      # A requires B
    ENABLES = auto()       # A enables B

    # Conceptual
    IS_A = auto()          # A is a type of B
    HAS_PROPERTY = auto()  # A has property B
    RELATED_TO = auto()    # Generic relation

    # Custom
    CUSTOM = auto()


@dataclass
class GraphEdge:
    """
    An edge connecting two nodes.

    Represents a relationship with:
    - Source and target nodes
    - Relationship type
    - Direction (or bidirectional)
    - Properties/metadata
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relationship: RelationshipType = RelationshipType.RELATED_TO
    label: str = ""
    bidirectional: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def connects(self, node_id: str) -> bool:
        """Check if edge connects to a node."""
        return node_id == self.source_id or node_id == self.target_id

    def other_end(self, node_id: str) -> Optional[str]:
        """Get the node at the other end of the edge."""
        if node_id == self.source_id:
            return self.target_id
        elif node_id == self.target_id:
            return self.source_id
        return None

    def is_traversable_from(self, node_id: str) -> bool:
        """Check if edge can be traversed from given node."""
        if node_id == self.source_id:
            return True
        if self.bidirectional and node_id == self.target_id:
            return True
        return False


class EdgeRegistry:
    """
    Registry for graph edges.

    Provides:
    - Edge storage and lookup
    - Query by relationship type
    - Finding edges for nodes
    """

    def __init__(self):
        self._edges: Dict[str, GraphEdge] = {}
        self._by_source: Dict[str, Set[str]] = {}
        self._by_target: Dict[str, Set[str]] = {}
        self._by_type: Dict[RelationshipType, Set[str]] = {}

    def add(self, edge: GraphEdge) -> None:
        """Add an edge to the registry."""
        self._edges[edge.id] = edge

        # Index by source
        if edge.source_id not in self._by_source:
            self._by_source[edge.source_id] = set()
        self._by_source[edge.source_id].add(edge.id)

        # Index by target
        if edge.target_id not in self._by_target:
            self._by_target[edge.target_id] = set()
        self._by_target[edge.target_id].add(edge.id)

        # Index by type
        if edge.relationship not in self._by_type:
            self._by_type[edge.relationship] = set()
        self._by_type[edge.relationship].add(edge.id)

    def get(self, edge_id: str) -> Optional[GraphEdge]:
        """Get edge by ID."""
        return self._edges.get(edge_id)

    def find_from(self, node_id: str) -> List[GraphEdge]:
        """Find all edges originating from a node."""
        ids = self._by_source.get(node_id, set())
        edges = [self._edges[id] for id in ids if id in self._edges]

        # Also include bidirectional edges targeting this node
        target_ids = self._by_target.get(node_id, set())
        for id in target_ids:
            if id in self._edges:
                edge = self._edges[id]
                if edge.bidirectional and edge not in edges:
                    edges.append(edge)

        return edges

    def find_to(self, node_id: str) -> List[GraphEdge]:
        """Find all edges targeting a node."""
        ids = self._by_target.get(node_id, set())
        edges = [self._edges[id] for id in ids if id in self._edges]

        # Also include bidirectional edges sourcing from this node
        source_ids = self._by_source.get(node_id, set())
        for id in source_ids:
            if id in self._edges:
                edge = self._edges[id]
                if edge.bidirectional and edge not in edges:
                    edges.append(edge)

        return edges

    def find_by_type(self, relationship: RelationshipType) -> List[GraphEdge]:
        """Find all edges of a relationship type."""
        ids = self._by_type.get(relationship, set())
        return [self._edges[id] for id in ids if id in self._edges]

    def find_between(self, source_id: str, target_id: str) -> List[GraphEdge]:
        """Find all edges between two nodes."""
        edges = []
        for edge in self._edges.values():
            if edge.source_id == source_id and edge.target_id == target_id:
                edges.append(edge)
            elif edge.bidirectional:
                if edge.source_id == target_id and edge.target_id == source_id:
                    edges.append(edge)
        return edges

    def remove(self, edge_id: str) -> bool:
        """Remove an edge from the registry."""
        if edge_id in self._edges:
            edge = self._edges[edge_id]

            # Remove from indices
            if edge.source_id in self._by_source:
                self._by_source[edge.source_id].discard(edge_id)
            if edge.target_id in self._by_target:
                self._by_target[edge.target_id].discard(edge_id)
            if edge.relationship in self._by_type:
                self._by_type[edge.relationship].discard(edge_id)

            del self._edges[edge_id]
            return True
        return False

    @property
    def count(self) -> int:
        """Get total edge count."""
        return len(self._edges)
