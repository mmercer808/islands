"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║              I N T E G R A T I O N   E X A M P L E                            ║
║                                                                               ║
║     How Traversal Wrapper Integrates with IslandGraph System                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

This shows how the TraversalWrapper and GraphTraverser work with your existing
graph_core.py IslandGraph and GraphNode system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional
from enum import Enum, auto

# Import from your existing system (simulated here for standalone demo)
# In real usage: from graph_core import IslandGraph, GraphNode, Edge, NodeType, EdgeType

# Simulated minimal graph_core types for standalone demo
class NodeType(Enum):
    LOCATION = auto()
    ITEM = auto()
    CHARACTER = auto()

class EdgeType(Enum):
    LEADS_TO = auto()
    CONTAINS = auto()
    LOCATED_AT = auto()

@dataclass
class GraphNode:
    id: str
    node_type: NodeType
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    edges_out: Dict[str, List['Edge']] = field(default_factory=dict)
    edges_in: Dict[str, List['Edge']] = field(default_factory=dict)
    
    def get_edges(self, edge_type=None, direction='out'):
        edge_dict = self.edges_out if direction == 'out' else self.edges_in
        if edge_type is None:
            return [e for edges in edge_dict.values() for e in edges]
        return edge_dict.get(edge_type.name, [])

@dataclass
class Edge:
    source: GraphNode
    target: GraphNode
    edge_type: EdgeType
    weight: float = 1.0
    data: Dict[str, Any] = field(default_factory=dict)

# ═══════════════════════════════════════════════════════════════════════════════
# Import the traversal wrapper system
# ═══════════════════════════════════════════════════════════════════════════════

from traversal_wrapper import (
    TraversalWrapper, GraphTraverser, TraversalContext,
    LayerRegistry, LayerConfig, EdgeResolver,
    smart_iter, traverse_graph, SmartList
)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER SETUP FOR YOUR EDGE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

def create_island_layers() -> LayerRegistry:
    """
    Create layer registry matching your EdgeType system.
    
    Each layer filters to specific edge types.
    """
    registry = LayerRegistry()
    
    # Spatial layer - for navigation
    registry.register(LayerConfig(
        name="spatial",
        edge_types={"LEADS_TO"},
        priority=10
    ))
    
    # Containment layer - for items/inventory
    registry.register(LayerConfig(
        name="containment", 
        edge_types={"CONTAINS", "LOCATED_AT"},
        priority=5
    ))
    
    # Character layer - for NPC interactions
    registry.register(LayerConfig(
        name="character",
        edge_types={"LOCATED_AT", "KNOWS", "OWNS"},
        priority=5
    ))
    
    # Combined "exploration" layer - all spatial relationships
    registry.register(LayerConfig(
        name="exploration",
        edge_types={"LEADS_TO", "CONTAINS", "LOCATED_AT"},
        priority=0
    ))
    
    # Dangerous areas auto-switch (example)
    registry.register(LayerConfig(
        name="danger",
        edge_types={"LEADS_TO"},
        switch_condition=lambda ctx: (
            ctx.current_node and 
            hasattr(ctx.current_node, 'tags') and
            'dangerous' in ctx.current_node.tags
        ),
        priority=20
    ))
    
    return registry


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM EDGE RESOLVER FOR YOUR GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

class IslandEdgeResolver(EdgeResolver):
    """
    Custom resolver that understands your GraphNode edge structure.
    """
    
    def _get_raw_edges(self, node: Any, context: TraversalContext) -> List[Any]:
        """Extract edges from GraphNode"""
        if isinstance(node, GraphNode):
            all_edges = []
            for edge_list in node.edges_out.values():
                all_edges.extend(edge_list)
            return all_edges
        return super()._get_raw_edges(node, context)
    
    def _get_edge_type(self, edge: Any) -> str:
        """Get edge type name from Edge object"""
        if isinstance(edge, Edge):
            return edge.edge_type.name
        return super()._get_edge_type(edge)
    
    def _get_edge_target(self, edge: Any, source: Any) -> Any:
        """Get target node from Edge"""
        if isinstance(edge, Edge):
            return edge.target
        return super()._get_edge_target(edge, source)


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLE GRAPH CLASS FOR DEMO
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleGraph:
    """Minimal graph for demo purposes"""
    
    def __init__(self, name: str = "demo"):
        self.name = name
        self.nodes: Dict[str, GraphNode] = {}
    
    def add_node(self, node_type: NodeType, name: str, 
                 node_id: str = None, **kwargs) -> GraphNode:
        node_id = node_id or f"{node_type.name.lower()}_{len(self.nodes)}"
        node = GraphNode(
            id=node_id,
            node_type=node_type,
            name=name,
            **kwargs
        )
        self.nodes[node_id] = node
        return node
    
    def add_edge(self, source: GraphNode, target: GraphNode, 
                 edge_type: EdgeType, **kwargs) -> Edge:
        edge = Edge(source=source, target=target, edge_type=edge_type, **kwargs)
        type_key = edge_type.name
        if type_key not in source.edges_out:
            source.edges_out[type_key] = []
        source.edges_out[type_key].append(edge)
        return edge
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self.nodes.get(node_id)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    print("=" * 70)
    print("Integration Demo: TraversalWrapper + IslandGraph")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Build a small island
    # ─────────────────────────────────────────────────────────────────────────
    island = SimpleGraph("Mystery Island")
    
    # Locations
    beach = island.add_node(NodeType.LOCATION, "Sandy Beach", 
                           tags={'outdoor', 'safe', 'start'})
    jungle = island.add_node(NodeType.LOCATION, "Dense Jungle",
                            tags={'outdoor', 'dangerous'})
    temple = island.add_node(NodeType.LOCATION, "Ancient Temple",
                            tags={'indoor', 'mysterious'})
    cave = island.add_node(NodeType.LOCATION, "Dark Cave",
                          tags={'indoor', 'dangerous', 'secret'})
    
    # Items
    torch = island.add_node(NodeType.ITEM, "Rusty Torch")
    key = island.add_node(NodeType.ITEM, "Bronze Key")
    
    # Character
    hermit = island.add_node(NodeType.CHARACTER, "Old Hermit")
    
    # Connect locations (LEADS_TO)
    island.add_edge(beach, jungle, EdgeType.LEADS_TO)
    island.add_edge(jungle, beach, EdgeType.LEADS_TO)
    island.add_edge(jungle, temple, EdgeType.LEADS_TO)
    island.add_edge(temple, jungle, EdgeType.LEADS_TO)
    island.add_edge(temple, cave, EdgeType.LEADS_TO)
    island.add_edge(cave, temple, EdgeType.LEADS_TO)
    
    # Place items (CONTAINS)
    island.add_edge(beach, torch, EdgeType.CONTAINS)
    island.add_edge(cave, key, EdgeType.CONTAINS)
    
    # Place character (LOCATED_AT)
    island.add_edge(hermit, jungle, EdgeType.LOCATED_AT)
    
    print(f"\nCreated island with {len(island.nodes)} nodes")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Setup layers
    # ─────────────────────────────────────────────────────────────────────────
    layers = create_island_layers()
    resolver = IslandEdgeResolver(layers)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: Basic BFS traversal on spatial layer
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Test 1: BFS Traversal (Spatial Layer) ---")
    
    traverser = GraphTraverser(
        graph=island,
        start_node=beach,
        strategy="bfs",
        layer="spatial",
        layer_registry=layers
    )
    traverser.resolver = resolver  # Use our custom resolver
    
    traverser.on_step(lambda w, n: print(f"  → {n.name} (depth={w.depth})"))
    
    visited = []
    for node in traverser:
        visited.append(node.name)
    
    print(f"  Visited in order: {visited}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: DFS traversal
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Test 2: DFS Traversal ---")
    
    dfs_traverser = GraphTraverser(
        graph=island,
        start_node=beach,
        strategy="dfs",
        layer="spatial",
        layer_registry=layers
    )
    dfs_traverser.resolver = resolver
    
    visited_dfs = []
    for node in dfs_traverser:
        visited_dfs.append(node.name)
        print(f"  → {node.name}")
    
    print(f"  DFS order: {visited_dfs}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: Layer switching mid-traversal
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Test 3: Layer Switching ---")
    
    ctx = TraversalContext()
    ctx.graph_ref = island
    ctx.current_layer = "spatial"
    
    traverser2 = GraphTraverser(
        graph=island,
        start_node=beach,
        layer="spatial",
        context=ctx,
        layer_registry=layers
    )
    traverser2.resolver = resolver
    
    traverser2.on_layer_switch(
        lambda w, old, new: print(f"  ⚡ Layer switch: {old} → {new}")
    )
    
    for node in traverser2:
        print(f"  At {node.name} (layer={traverser2.layer})")
        
        # When we reach temple, switch to containment layer
        if node.name == "Ancient Temple":
            traverser2.switch_layer("containment", push=True)
            
            # Find items at this location
            edges = traverser2.get_available_edges()
            items = [target for edge, target in edges 
                    if isinstance(target, GraphNode) and target.node_type == NodeType.ITEM]
            if items:
                print(f"    Found items: {[i.name for i in items]}")
            
            # Pop back to spatial
            traverser2.pop_layer()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 4: Context carries through - inventory example
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Test 4: Context Carries Inventory ---")
    
    ctx2 = TraversalContext()
    ctx2.graph_ref = island
    ctx2.inventory = []
    
    traverser3 = GraphTraverser(
        graph=island,
        start_node=beach,
        layer="exploration",  # Combined spatial + containment
        context=ctx2,
        layer_registry=layers
    )
    traverser3.resolver = resolver
    
    def collect_items(wrapper, node):
        if node.node_type == NodeType.ITEM:
            wrapper.context.inventory.append(node.name)
            print(f"  📦 Collected: {node.name}")
        else:
            print(f"  📍 At: {node.name}")
    
    traverser3.on_step(collect_items)
    
    for node in traverser3:
        pass  # callback handles output
    
    print(f"  Final inventory: {traverser3.context.inventory}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 5: Hotswap graph mid-traversal
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Test 5: Graph Hotswap ---")
    
    # Create a second "hidden" part of the island
    hidden_island = SimpleGraph("Hidden Caves")
    secret1 = hidden_island.add_node(NodeType.LOCATION, "Secret Chamber")
    secret2 = hidden_island.add_node(NodeType.LOCATION, "Treasure Vault")
    hidden_island.add_edge(secret1, secret2, EdgeType.LEADS_TO)
    
    # Start on main island
    ctx3 = TraversalContext()
    ctx3.graph_ref = island
    
    wrapper = smart_iter([beach, jungle, temple], context=ctx3)
    
    for i, node in enumerate(wrapper):
        print(f"  At: {node.name} (graph={wrapper.context.graph_ref.name})")
        
        # After temple, swap to hidden island
        if node.name == "Ancient Temple":
            print("  🔮 Discovered secret passage!")
            wrapper.swap_graph(hidden_island)
            wrapper.jump_to(secret1)
            print(f"  Now at: {wrapper.current.name} (graph={wrapper.context.graph_ref.name})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 6: Using SmartList with graph nodes
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Test 6: SmartList with Graph Nodes ---")
    
    # Put nodes in a SmartList
    locations = SmartList([beach, jungle, temple, cave])
    
    # Iterate with smart wrapper
    wrapper = iter(locations)
    for node in wrapper:
        print(f"  {node.name} - step {wrapper.step_count}, depth {wrapper.depth}")
        if 'dangerous' in node.tags:
            print(f"    ⚠️  DANGER!")
            wrapper.context.flags['encountered_danger'] = True
    
    print(f"  Encountered danger: {wrapper.context.flags.get('encountered_danger', False)}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
