"""
Pine Graph Walker
=================

Graph traversal and walking utilities.

SOURCE: Mechanics/everything/graph_walker.py (35KB)
Copy the full implementation from that file.
"""

from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
import uuid

from .nodes import GraphNode, NodeRegistry
from .edges import GraphEdge, EdgeRegistry, RelationshipType


@dataclass
class WalkerContext:
    """Context carried through graph walking."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_node_id: Optional[str] = None
    path: List[str] = field(default_factory=list)
    visited: Set[str] = field(default_factory=set)
    data: Dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    max_depth: int = 10

    def move_to(self, node_id: str) -> None:
        """Move to a new node."""
        if self.current_node_id:
            self.path.append(self.current_node_id)
        self.current_node_id = node_id
        self.visited.add(node_id)
        self.depth = len(self.path)

    def can_visit(self, node_id: str) -> bool:
        """Check if we can visit a node."""
        if node_id in self.visited:
            return False
        if self.depth >= self.max_depth:
            return False
        return True

    def go_back(self) -> bool:
        """Go back to previous node."""
        if self.path:
            self.current_node_id = self.path.pop()
            self.depth = len(self.path)
            return True
        return False


@dataclass
class WalkResult:
    """Result of a graph walk."""
    visited_nodes: List[str] = field(default_factory=list)
    path_taken: List[str] = field(default_factory=list)
    found_targets: List[str] = field(default_factory=list)
    data_collected: Dict[str, Any] = field(default_factory=dict)


class GraphWalker:
    """
    Walks through a graph, visiting nodes and following edges.

    Supports:
    - BFS and DFS traversal
    - Filtering by edge type
    - Callbacks on visit
    - Target searching

    TODO: Copy full implementation from Mechanics/everything/graph_walker.py
    """

    def __init__(
        self,
        nodes: NodeRegistry,
        edges: EdgeRegistry
    ):
        self.nodes = nodes
        self.edges = edges

    def walk_bfs(
        self,
        start_id: str,
        on_visit: Callable[[GraphNode, WalkerContext], None] = None,
        filter_edges: Callable[[GraphEdge], bool] = None,
        max_depth: int = 10
    ) -> WalkResult:
        """
        Breadth-first walk from starting node.

        Args:
            start_id: Starting node ID
            on_visit: Callback for each visited node
            filter_edges: Filter which edges to follow
            max_depth: Maximum depth to traverse
        """
        result = WalkResult()
        context = WalkerContext(max_depth=max_depth)

        queue = [(start_id, 0)]
        context.visited.add(start_id)

        while queue:
            node_id, depth = queue.pop(0)

            if depth > max_depth:
                continue

            node = self.nodes.get(node_id)
            if not node:
                continue

            # Visit
            context.move_to(node_id)
            result.visited_nodes.append(node_id)

            if on_visit:
                on_visit(node, context)

            # Get edges
            edges = self.edges.find_from(node_id)

            for edge in edges:
                # Apply filter
                if filter_edges and not filter_edges(edge):
                    continue

                # Get target
                target_id = edge.other_end(node_id)
                if target_id and target_id not in context.visited:
                    context.visited.add(target_id)
                    queue.append((target_id, depth + 1))

        result.path_taken = list(context.path)
        return result

    def walk_dfs(
        self,
        start_id: str,
        on_visit: Callable[[GraphNode, WalkerContext], None] = None,
        filter_edges: Callable[[GraphEdge], bool] = None,
        max_depth: int = 10
    ) -> WalkResult:
        """
        Depth-first walk from starting node.
        """
        result = WalkResult()
        context = WalkerContext(max_depth=max_depth)

        def dfs(node_id: str, depth: int):
            if depth > max_depth:
                return

            node = self.nodes.get(node_id)
            if not node:
                return

            # Visit
            context.move_to(node_id)
            result.visited_nodes.append(node_id)

            if on_visit:
                on_visit(node, context)

            # Get edges
            edges = self.edges.find_from(node_id)

            for edge in edges:
                if filter_edges and not filter_edges(edge):
                    continue

                target_id = edge.other_end(node_id)
                if target_id and context.can_visit(target_id):
                    dfs(target_id, depth + 1)

        context.visited.add(start_id)
        dfs(start_id, 0)

        result.path_taken = list(context.path)
        return result

    def find_path(
        self,
        start_id: str,
        target_id: str,
        filter_edges: Callable[[GraphEdge], bool] = None
    ) -> Optional[List[str]]:
        """Find path between two nodes using BFS."""
        if start_id == target_id:
            return [start_id]

        visited = {start_id}
        queue = [(start_id, [start_id])]

        while queue:
            current_id, path = queue.pop(0)

            edges = self.edges.find_from(current_id)

            for edge in edges:
                if filter_edges and not filter_edges(edge):
                    continue

                next_id = edge.other_end(current_id)
                if not next_id or next_id in visited:
                    continue

                new_path = path + [next_id]

                if next_id == target_id:
                    return new_path

                visited.add(next_id)
                queue.append((next_id, new_path))

        return None

    def find_by_type(
        self,
        start_id: str,
        relationship: RelationshipType
    ) -> List[GraphNode]:
        """Find all nodes reachable via a specific relationship type."""
        results = []

        def filter_edge(edge: GraphEdge) -> bool:
            return edge.relationship == relationship

        def on_visit(node: GraphNode, ctx: WalkerContext):
            if node.id != start_id:
                results.append(node)

        self.walk_bfs(start_id, on_visit=on_visit, filter_edges=filter_edge)
        return results
