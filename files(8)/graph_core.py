"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                    I S L A N D   G R A P H   C O R E                          ║
║                                                                               ║
║              A Graph Structure for Narrative Game Worlds                      ║
║                                                                               ║
║  The island is a graph. Locations are nodes. Paths are edges.                 ║
║  Stories branch. Items connect. Characters move.                              ║
║  The walker traverses, queries, and reshapes as it goes.                      ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Callable, Iterator, Tuple, Union
from enum import Enum, auto
from collections import deque
import json
import hashlib
import time


# ═══════════════════════════════════════════════════════════════════════════════
# NODE TYPES - What things exist in the world
# ═══════════════════════════════════════════════════════════════════════════════

class NodeType(Enum):
    """Categories of things that can exist in the game world"""
    LOCATION = auto()      # Places on the island
    ITEM = auto()          # Objects that can be examined/taken
    CHARACTER = auto()     # NPCs and entities
    EVENT = auto()         # Story beats, triggers
    PROSE = auto()         # Descriptive text passages
    CONDITION = auto()     # State checks (has_key, visited_cave, etc.)
    CONTAINER = auto()     # Things that hold other things
    DIALOGUE = auto()      # Conversation nodes
    MEMORY = auto()        # Player knowledge/discoveries
    PORTAL = auto()        # Connections between distant places


class EdgeType(Enum):
    """How nodes relate to each other"""
    # Spatial relationships
    LEADS_TO = auto()      # location → location (bidirectional navigation)
    CONTAINS = auto()      # container/location → item
    LOCATED_AT = auto()    # item/character → location
    
    # Narrative relationships
    TRIGGERS = auto()      # event → event (causation)
    REQUIRES = auto()      # node → condition (prerequisite)
    UNLOCKS = auto()       # item/event → node (enables access)
    REVEALS = auto()       # event → prose/memory (shows new info)
    
    # Logical relationships
    PART_OF = auto()       # node → node (composition)
    INSTANCE_OF = auto()   # node → node (classification)
    REFERS_TO = auto()     # prose → node (mention)
    
    # Character relationships
    KNOWS = auto()         # character → character/memory
    OWNS = auto()          # character → item
    WANTS = auto()         # character → item/event


# ═══════════════════════════════════════════════════════════════════════════════
# THE NODE - A single point in the game world
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GraphNode:
    """
    A node in the game world graph.
    
    Nodes are lightweight containers. The meaning comes from:
    - Their type (what kind of thing)
    - Their data (specific properties)
    - Their edges (relationships to other nodes)
    """
    id: str                           # Unique identifier
    node_type: NodeType               # What kind of thing this is
    name: str                         # Human-readable name
    data: Dict[str, Any] = field(default_factory=dict)  # Flexible properties
    tags: Set[str] = field(default_factory=set)         # For filtering/queries
    created_at: float = field(default_factory=time.time)
    
    # Edges stored on the node for fast traversal
    edges_out: Dict[str, List['Edge']] = field(default_factory=dict)  # edge_type → edges
    edges_in: Dict[str, List['Edge']] = field(default_factory=dict)   # incoming edges
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, GraphNode):
            return self.id == other.id
        return False
    
    def get_edges(self, edge_type: EdgeType = None, direction: str = 'out') -> List['Edge']:
        """Get edges, optionally filtered by type"""
        edge_dict = self.edges_out if direction == 'out' else self.edges_in
        
        if edge_type is None:
            return [e for edges in edge_dict.values() for e in edges]
        
        type_key = edge_type.name
        return edge_dict.get(type_key, [])
    
    def get_neighbors(self, edge_type: EdgeType = None) -> List['GraphNode']:
        """Get all nodes connected by outgoing edges"""
        edges = self.get_edges(edge_type, 'out')
        return [e.target for e in edges]
    
    def add_tag(self, tag: str):
        self.tags.add(tag.lower())
    
    def has_tag(self, tag: str) -> bool:
        return tag.lower() in self.tags
    
    def set_data(self, key: str, value: Any):
        self.data[key] = value
    
    def get_data(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'id': self.id,
            'type': self.node_type.name,
            'name': self.name,
            'data': self.data,
            'tags': list(self.tags),
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'GraphNode':
        """Deserialize from storage"""
        return cls(
            id=d['id'],
            node_type=NodeType[d['type']],
            name=d['name'],
            data=d.get('data', {}),
            tags=set(d.get('tags', [])),
            created_at=d.get('created_at', time.time())
        )
    
    def __repr__(self):
        return f"<{self.node_type.name}:{self.name}({self.id[:8]})>"


# ═══════════════════════════════════════════════════════════════════════════════
# THE EDGE - A relationship between nodes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Edge:
    """
    A directed edge between two nodes.
    
    Edges can carry:
    - Type (what kind of relationship)
    - Weight (strength/cost)
    - Data (arbitrary properties)
    - Conditions (when is this edge valid?)
    """
    source: GraphNode
    target: GraphNode
    edge_type: EdgeType
    weight: float = 1.0
    data: Dict[str, Any] = field(default_factory=dict)
    bidirectional: bool = False       # Create reverse edge automatically?
    condition: Optional[Callable] = None  # Function that returns True if edge is traversable
    
    def is_traversable(self, context: Dict = None) -> bool:
        """Check if this edge can be crossed given current context"""
        if self.condition is None:
            return True
        try:
            return self.condition(context or {})
        except:
            return False
    
    def to_dict(self) -> Dict:
        return {
            'source_id': self.source.id,
            'target_id': self.target.id,
            'type': self.edge_type.name,
            'weight': self.weight,
            'data': self.data,
            'bidirectional': self.bidirectional
        }
    
    def __repr__(self):
        arrow = "↔" if self.bidirectional else "→"
        return f"[{self.source.name} {arrow} {self.edge_type.name} {arrow} {self.target.name}]"


# ═══════════════════════════════════════════════════════════════════════════════
# THE GRAPH - The complete world structure
# ═══════════════════════════════════════════════════════════════════════════════

class IslandGraph:
    """
    The complete graph structure for the game world.
    
    Features:
    - Add/remove nodes and edges
    - Query by type, tag, or custom filter
    - Subgraph extraction
    - Serialization for save/load
    - Integration hooks for the walker
    """
    
    def __init__(self, name: str = "unnamed_world"):
        self.name = name
        self.nodes: Dict[str, GraphNode] = {}    # id → node
        self.edges: List[Edge] = []               # All edges
        self._type_index: Dict[str, Set[str]] = {}  # type_name → {node_ids}
        self._tag_index: Dict[str, Set[str]] = {}   # tag → {node_ids}
        self.metadata: Dict[str, Any] = {
            'created_at': time.time(),
            'version': '1.0'
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Node Operations
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_node(self, 
                 node_type: NodeType,
                 name: str,
                 node_id: str = None,
                 data: Dict = None,
                 tags: List[str] = None) -> GraphNode:
        """Create and add a new node to the graph"""
        
        # Generate ID if not provided
        if node_id is None:
            node_id = self._generate_id(node_type.name, name)
        
        # Create node
        node = GraphNode(
            id=node_id,
            node_type=node_type,
            name=name,
            data=data or {},
            tags=set(tags or [])
        )
        
        # Add to graph
        self.nodes[node_id] = node
        
        # Update indices
        type_key = node_type.name
        if type_key not in self._type_index:
            self._type_index[type_key] = set()
        self._type_index[type_key].add(node_id)
        
        for tag in node.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(node_id)
        
        return node
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges"""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        
        # Remove from indices
        type_key = node.node_type.name
        if type_key in self._type_index:
            self._type_index[type_key].discard(node_id)
        
        for tag in node.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(node_id)
        
        # Remove edges
        self.edges = [e for e in self.edges 
                      if e.source.id != node_id and e.target.id != node_id]
        
        # Clean up edge references on other nodes
        for other_node in self.nodes.values():
            for edge_type, edges in list(other_node.edges_out.items()):
                other_node.edges_out[edge_type] = [e for e in edges if e.target.id != node_id]
            for edge_type, edges in list(other_node.edges_in.items()):
                other_node.edges_in[edge_type] = [e for e in edges if e.source.id != node_id]
        
        # Remove node
        del self.nodes[node_id]
        return True
    
    # ─────────────────────────────────────────────────────────────────────────
    # Edge Operations
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_edge(self,
                 source: Union[str, GraphNode],
                 target: Union[str, GraphNode],
                 edge_type: EdgeType,
                 weight: float = 1.0,
                 data: Dict = None,
                 bidirectional: bool = False,
                 condition: Callable = None) -> Edge:
        """Create an edge between two nodes"""
        
        # Resolve nodes
        if isinstance(source, str):
            source = self.nodes.get(source)
        if isinstance(target, str):
            target = self.nodes.get(target)
        
        if source is None or target is None:
            raise ValueError("Source and target nodes must exist")
        
        # Create edge
        edge = Edge(
            source=source,
            target=target,
            edge_type=edge_type,
            weight=weight,
            data=data or {},
            bidirectional=bidirectional,
            condition=condition
        )
        
        self.edges.append(edge)
        
        # Add to node's edge lists
        type_key = edge_type.name
        if type_key not in source.edges_out:
            source.edges_out[type_key] = []
        source.edges_out[type_key].append(edge)
        
        if type_key not in target.edges_in:
            target.edges_in[type_key] = []
        target.edges_in[type_key].append(edge)
        
        # Handle bidirectional
        if bidirectional:
            reverse = Edge(
                source=target,
                target=source,
                edge_type=edge_type,
                weight=weight,
                data=data or {},
                bidirectional=False,  # Don't recurse
                condition=condition
            )
            self.edges.append(reverse)
            
            if type_key not in target.edges_out:
                target.edges_out[type_key] = []
            target.edges_out[type_key].append(reverse)
            
            if type_key not in source.edges_in:
                source.edges_in[type_key] = []
            source.edges_in[type_key].append(reverse)
        
        return edge
    
    def connect(self, source_id: str, target_id: str, edge_type: EdgeType, **kwargs) -> Edge:
        """Convenience method for adding edges by ID"""
        return self.add_edge(source_id, target_id, edge_type, **kwargs)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Query Operations
    # ─────────────────────────────────────────────────────────────────────────
    
    def query_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Get all nodes of a specific type"""
        type_key = node_type.name
        if type_key not in self._type_index:
            return []
        return [self.nodes[nid] for nid in self._type_index[type_key]]
    
    def query_by_tag(self, tag: str) -> List[GraphNode]:
        """Get all nodes with a specific tag"""
        tag = tag.lower()
        if tag not in self._tag_index:
            return []
        return [self.nodes[nid] for nid in self._tag_index[tag]]
    
    def query(self, 
              node_type: NodeType = None,
              tags: List[str] = None,
              filter_fn: Callable[[GraphNode], bool] = None) -> List[GraphNode]:
        """
        Flexible query with optional type, tags, and custom filter.
        All conditions are AND-ed together.
        """
        results = list(self.nodes.values())
        
        if node_type is not None:
            type_ids = self._type_index.get(node_type.name, set())
            results = [n for n in results if n.id in type_ids]
        
        if tags:
            for tag in tags:
                tag_ids = self._tag_index.get(tag.lower(), set())
                results = [n for n in results if n.id in tag_ids]
        
        if filter_fn:
            results = [n for n in results if filter_fn(n)]
        
        return results
    
    def find_path(self, 
                  start_id: str, 
                  end_id: str,
                  edge_types: List[EdgeType] = None,
                  context: Dict = None) -> Optional[List[GraphNode]]:
        """
        Find a path between two nodes using BFS.
        Respects edge conditions if context is provided.
        """
        if start_id not in self.nodes or end_id not in self.nodes:
            return None
        
        visited = {start_id}
        queue = deque([(start_id, [self.nodes[start_id]])])
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id == end_id:
                return path
            
            current = self.nodes[current_id]
            for edge in current.get_edges():
                # Filter by edge type if specified
                if edge_types and edge.edge_type not in edge_types:
                    continue
                
                # Check if traversable
                if not edge.is_traversable(context):
                    continue
                
                next_id = edge.target.id
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [edge.target]))
        
        return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Subgraph Operations
    # ─────────────────────────────────────────────────────────────────────────
    
    def extract_subgraph(self, 
                         root_id: str, 
                         depth: int = 2,
                         edge_types: List[EdgeType] = None) -> 'IslandGraph':
        """
        Extract a subgraph starting from a root node.
        Useful for getting "local" context around a location.
        """
        subgraph = IslandGraph(name=f"{self.name}_sub_{root_id[:8]}")
        
        if root_id not in self.nodes:
            return subgraph
        
        visited = set()
        queue = deque([(root_id, 0)])
        
        while queue:
            node_id, current_depth = queue.popleft()
            
            if node_id in visited or current_depth > depth:
                continue
            
            visited.add(node_id)
            node = self.nodes[node_id]
            
            # Copy node to subgraph
            subgraph.nodes[node_id] = GraphNode(
                id=node.id,
                node_type=node.node_type,
                name=node.name,
                data=node.data.copy(),
                tags=node.tags.copy(),
                created_at=node.created_at
            )
            
            # Update indices
            type_key = node.node_type.name
            if type_key not in subgraph._type_index:
                subgraph._type_index[type_key] = set()
            subgraph._type_index[type_key].add(node_id)
            
            for tag in node.tags:
                if tag not in subgraph._tag_index:
                    subgraph._tag_index[tag] = set()
                subgraph._tag_index[tag].add(node_id)
            
            # Queue neighbors
            for edge in node.get_edges():
                if edge_types and edge.edge_type not in edge_types:
                    continue
                queue.append((edge.target.id, current_depth + 1))
        
        # Copy edges that exist entirely within subgraph
        for edge in self.edges:
            if edge.source.id in visited and edge.target.id in visited:
                subgraph.add_edge(
                    edge.source.id,
                    edge.target.id,
                    edge.edge_type,
                    weight=edge.weight,
                    data=edge.data.copy()
                )
        
        return subgraph
    
    # ─────────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────────
    
    def to_dict(self) -> Dict:
        """Serialize the entire graph"""
        return {
            'name': self.name,
            'metadata': self.metadata,
            'nodes': [n.to_dict() for n in self.nodes.values()],
            'edges': [e.to_dict() for e in self.edges if not e.bidirectional]  # Avoid duplicates
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'IslandGraph':
        """Deserialize from dict"""
        graph = cls(name=d.get('name', 'unnamed'))
        graph.metadata = d.get('metadata', {})
        
        # Load nodes first
        for node_data in d.get('nodes', []):
            node = GraphNode.from_dict(node_data)
            graph.nodes[node.id] = node
            
            # Update indices
            type_key = node.node_type.name
            if type_key not in graph._type_index:
                graph._type_index[type_key] = set()
            graph._type_index[type_key].add(node.id)
            
            for tag in node.tags:
                if tag not in graph._tag_index:
                    graph._tag_index[tag] = set()
                graph._tag_index[tag].add(node.id)
        
        # Load edges
        for edge_data in d.get('edges', []):
            source = graph.nodes.get(edge_data['source_id'])
            target = graph.nodes.get(edge_data['target_id'])
            if source and target:
                graph.add_edge(
                    source,
                    target,
                    EdgeType[edge_data['type']],
                    weight=edge_data.get('weight', 1.0),
                    data=edge_data.get('data', {}),
                    bidirectional=edge_data.get('bidirectional', False)
                )
        
        return graph
    
    @classmethod
    def from_json(cls, json_str: str) -> 'IslandGraph':
        return cls.from_dict(json.loads(json_str))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────
    
    def _generate_id(self, type_name: str, name: str) -> str:
        """Generate a unique ID based on type and name"""
        content = f"{type_name}:{name}:{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def stats(self) -> Dict:
        """Get statistics about the graph"""
        type_counts = {t: len(ids) for t, ids in self._type_index.items()}
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'nodes_by_type': type_counts,
            'total_tags': len(self._tag_index)
        }
    
    def __repr__(self):
        return f"<IslandGraph '{self.name}' nodes={len(self.nodes)} edges={len(self.edges)}>"


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK BUILDERS - Convenience functions for common patterns
# ═══════════════════════════════════════════════════════════════════════════════

def create_location(graph: IslandGraph, 
                    name: str, 
                    description: str = "",
                    tags: List[str] = None) -> GraphNode:
    """Quick builder for location nodes"""
    return graph.add_node(
        NodeType.LOCATION,
        name,
        data={'description': description},
        tags=tags or []
    )

def create_item(graph: IslandGraph,
                name: str,
                description: str = "",
                takeable: bool = True,
                tags: List[str] = None) -> GraphNode:
    """Quick builder for item nodes"""
    return graph.add_node(
        NodeType.ITEM,
        name,
        data={'description': description, 'takeable': takeable},
        tags=tags or []
    )

def create_character(graph: IslandGraph,
                     name: str,
                     description: str = "",
                     dialogue_entry: str = None,
                     tags: List[str] = None) -> GraphNode:
    """Quick builder for character nodes"""
    return graph.add_node(
        NodeType.CHARACTER,
        name,
        data={'description': description, 'dialogue_entry': dialogue_entry},
        tags=tags or []
    )

def create_prose(graph: IslandGraph,
                 name: str,
                 text: str,
                 tags: List[str] = None) -> GraphNode:
    """Quick builder for prose/description nodes"""
    return graph.add_node(
        NodeType.PROSE,
        name,
        data={'text': text},
        tags=tags or []
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Island Graph Core - Demonstration")
    print("=" * 70)
    
    # Create a small island
    island = IslandGraph(name="Mystery Island")
    
    # Add locations
    beach = create_location(island, "Sandy Beach", 
                           "Waves lap at the shore. Palm trees sway.", 
                           tags=['outdoor', 'start'])
    jungle = create_location(island, "Dense Jungle",
                            "Thick foliage blocks the sun.",
                            tags=['outdoor', 'dangerous'])
    cave = create_location(island, "Dark Cave",
                          "Shadows dance on wet stone walls.",
                          tags=['indoor', 'mysterious'])
    
    # Add items
    torch = create_item(island, "Rusty Torch", "An old torch. Might still work.")
    key = create_item(island, "Bronze Key", "Tarnished but solid.", tags=['quest'])
    
    # Add character
    hermit = create_character(island, "Old Hermit",
                             "A weathered figure in tattered robes.",
                             tags=['friendly'])
    
    # Connect locations
    island.connect(beach.id, jungle.id, EdgeType.LEADS_TO, bidirectional=True)
    island.connect(jungle.id, cave.id, EdgeType.LEADS_TO, bidirectional=True)
    
    # Place items
    island.connect(beach.id, torch.id, EdgeType.CONTAINS)
    island.connect(cave.id, key.id, EdgeType.CONTAINS)
    
    # Place character
    island.connect(hermit.id, jungle.id, EdgeType.LOCATED_AT)
    
    # Query examples
    print("\n--- Locations ---")
    for loc in island.query_by_type(NodeType.LOCATION):
        print(f"  {loc}")
    
    print("\n--- Items with 'quest' tag ---")
    for item in island.query_by_tag('quest'):
        print(f"  {item}")
    
    print("\n--- Path from Beach to Cave ---")
    path = island.find_path(beach.id, cave.id, [EdgeType.LEADS_TO])
    if path:
        print(f"  {' → '.join(n.name for n in path)}")
    
    print("\n--- Graph Stats ---")
    stats = island.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n--- Serialization Test ---")
    json_data = island.to_json()
    print(f"  JSON size: {len(json_data)} bytes")
    
    restored = IslandGraph.from_json(json_data)
    print(f"  Restored: {restored}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
