"""
Pine Graph Nodes
================

Graph node structures and registry.

SOURCE: Mechanics/everything/graph_core.py (31KB)
Copy the full implementation from that file.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import uuid


class NodeType(Enum):
    """Types of graph nodes."""
    ENTITY = auto()      # General entity
    LOCATION = auto()    # Place
    ITEM = auto()        # Object
    CHARACTER = auto()   # Person/creature
    CONCEPT = auto()     # Abstract idea
    EVENT = auto()       # Something that happens


@dataclass
class GraphNode:
    """
    A node in the entity graph.

    Represents any entity with:
    - Unique identity
    - Type classification
    - Arbitrary properties
    - Tags for querying
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    node_type: NodeType = NodeType.ENTITY
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        if isinstance(other, GraphNode):
            return self.id == other.id
        return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get property value."""
        return self.properties.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set property value."""
        self.properties[key] = value

    def add_tag(self, tag: str) -> None:
        """Add a tag."""
        self.tags.add(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if node has a tag."""
        return tag in self.tags


class NodeRegistry:
    """
    Registry for graph nodes.

    Provides:
    - Node storage and lookup
    - Query by type, tags, properties
    """

    def __init__(self):
        self._nodes: Dict[str, GraphNode] = {}
        self._by_type: Dict[NodeType, Set[str]] = {}
        self._by_tag: Dict[str, Set[str]] = {}

    def add(self, node: GraphNode) -> None:
        """Add a node to the registry."""
        self._nodes[node.id] = node

        # Index by type
        if node.node_type not in self._by_type:
            self._by_type[node.node_type] = set()
        self._by_type[node.node_type].add(node.id)

        # Index by tags
        for tag in node.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = set()
            self._by_tag[tag].add(node.id)

    def get(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def get_by_name(self, name: str) -> Optional[GraphNode]:
        """Get node by name."""
        name_lower = name.lower()
        for node in self._nodes.values():
            if node.name.lower() == name_lower:
                return node
        return None

    def find_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Find all nodes of a type."""
        ids = self._by_type.get(node_type, set())
        return [self._nodes[id] for id in ids if id in self._nodes]

    def find_by_tag(self, tag: str) -> List[GraphNode]:
        """Find all nodes with a tag."""
        ids = self._by_tag.get(tag, set())
        return [self._nodes[id] for id in ids if id in self._nodes]

    def query(
        self,
        node_type: NodeType = None,
        tags: List[str] = None,
        **properties
    ) -> List[GraphNode]:
        """Query nodes with multiple criteria."""
        results = list(self._nodes.values())

        if node_type:
            results = [n for n in results if n.node_type == node_type]

        if tags:
            results = [n for n in results if all(t in n.tags for t in tags)]

        for key, value in properties.items():
            results = [n for n in results if n.properties.get(key) == value]

        return results

    def remove(self, node_id: str) -> bool:
        """Remove a node from the registry."""
        if node_id in self._nodes:
            node = self._nodes[node_id]

            # Remove from type index
            if node.node_type in self._by_type:
                self._by_type[node.node_type].discard(node_id)

            # Remove from tag indices
            for tag in node.tags:
                if tag in self._by_tag:
                    self._by_tag[tag].discard(node_id)

            del self._nodes[node_id]
            return True
        return False

    @property
    def count(self) -> int:
        """Get total node count."""
        return len(self._nodes)
