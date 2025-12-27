"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                     G R A P H   W A L K E R                                   ║
║                                                                               ║
║           An Iterator That Thinks As It Traverses                             ║
║                                                                               ║
║  The walker moves through the graph like a consciousness through a dream.     ║
║  It carries memory. It triggers events. It reshapes the world it explores.    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing import (
    Dict, List, Set, Optional, Any, Callable, Iterator, 
    Tuple, Union, Generator, TypeVar, Generic
)
from enum import Enum, auto
from collections import deque
import copy
import time

from graph_core import (
    IslandGraph, GraphNode, Edge, NodeType, EdgeType,
    create_location, create_item, create_prose
)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAVERSAL STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

class TraversalStrategy(Enum):
    """How the walker moves through the graph"""
    DFS = auto()          # Depth-first: go deep before wide
    BFS = auto()          # Breadth-first: explore level by level
    RANDOM = auto()       # Random walk
    PRIORITY = auto()     # Follow highest-weight edges first
    GUIDED = auto()       # Follow a specific path


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WalkerCallback:
    """
    A callback to execute during traversal.
    
    Callbacks can:
    - Fire on entering a node
    - Fire on leaving a node
    - Fire on traversing an edge
    - Fire on specific conditions
    
    They receive the walker's full context and can modify it.
    """
    name: str
    trigger: str                  # 'enter', 'leave', 'edge', 'condition'
    handler: Callable             # Function(walker, node/edge, context) → result
    condition: Callable = None    # Optional: when should this fire?
    priority: int = 0             # Higher = fires first
    once: bool = False            # Fire only once?
    fired: bool = False           # Has it fired?
    
    def should_fire(self, walker: 'GraphWalker', target: Any) -> bool:
        """Check if this callback should fire"""
        if self.once and self.fired:
            return False
        if self.condition is None:
            return True
        try:
            return self.condition(walker, target)
        except:
            return False
    
    def execute(self, walker: 'GraphWalker', target: Any) -> Any:
        """Execute the callback"""
        self.fired = True
        return self.handler(walker, target, walker.context)


@dataclass 
class GraftBranch:
    """
    A subgraph that can be attached to the main graph during traversal.
    
    Branches can be:
    - Pre-built subgraphs
    - Generated dynamically by callbacks
    - Conditional on game state
    """
    name: str
    subgraph: IslandGraph
    attach_point: str             # Node ID where this grafts
    attach_edge_type: EdgeType = EdgeType.CONTAINS
    condition: Callable = None    # When should this graft?
    grafted: bool = False
    
    def should_graft(self, walker: 'GraphWalker') -> bool:
        if self.grafted:
            return False
        if self.condition is None:
            return True
        try:
            return self.condition(walker)
        except:
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# THE WALKER CONTEXT - What the walker carries with it
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WalkerContext:
    """
    The state that travels with the walker.
    
    This is the walker's "memory" - everything it knows and carries.
    Callbacks can read and modify this freely.
    """
    # Movement history
    visited: Set[str] = field(default_factory=set)       # Node IDs we've seen
    path: List[str] = field(default_factory=list)        # The path we've taken
    
    # Local buffer - scratch space for callbacks
    buffer: Dict[str, Any] = field(default_factory=dict)
    
    # Collected items (for inventory-style games)
    inventory: List[str] = field(default_factory=list)   # Item node IDs
    
    # Game state flags
    flags: Dict[str, bool] = field(default_factory=dict)
    
    # Numeric state (health, score, etc.)
    counters: Dict[str, int] = field(default_factory=dict)
    
    # Discovered information
    memories: List[Dict] = field(default_factory=list)   # Things we've learned
    
    # Event log
    events: List[Dict] = field(default_factory=list)     # What's happened
    
    def set_flag(self, name: str, value: bool = True):
        self.flags[name] = value
    
    def get_flag(self, name: str, default: bool = False) -> bool:
        return self.flags.get(name, default)
    
    def increment(self, counter: str, amount: int = 1):
        self.counters[counter] = self.counters.get(counter, 0) + amount
    
    def get_counter(self, counter: str, default: int = 0) -> int:
        return self.counters.get(counter, default)
    
    def add_memory(self, memory_type: str, content: Any, source: str = None):
        self.memories.append({
            'type': memory_type,
            'content': content,
            'source': source,
            'timestamp': time.time()
        })
    
    def log_event(self, event_type: str, details: Dict = None):
        self.events.append({
            'type': event_type,
            'details': details or {},
            'timestamp': time.time()
        })
    
    def buffer_set(self, key: str, value: Any):
        self.buffer[key] = value
    
    def buffer_get(self, key: str, default: Any = None) -> Any:
        return self.buffer.get(key, default)
    
    def buffer_clear(self):
        self.buffer.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# THE GRAPH WALKER - The main traversal engine
# ═══════════════════════════════════════════════════════════════════════════════

class GraphWalker:
    """
    An iterator that traverses the graph with intelligence.
    
    The walker:
    - Moves through nodes following edges
    - Carries context (memory, inventory, flags)
    - Executes callbacks on enter/leave/edge
    - Can query the graph from current position
    - Can modify the graph (add nodes, graft branches)
    - Respects edge conditions
    
    Think of it as a cursor that understands what it's reading.
    """
    
    def __init__(self, 
                 graph: IslandGraph,
                 start_node: Union[str, GraphNode] = None,
                 strategy: TraversalStrategy = TraversalStrategy.DFS):
        
        self.graph = graph
        self.strategy = strategy
        self.context = WalkerContext()
        
        # Current position
        if isinstance(start_node, str):
            self._current_id = start_node
        elif isinstance(start_node, GraphNode):
            self._current_id = start_node.id
        else:
            self._current_id = None
        
        # Traversal state
        self._frontier: deque = deque()  # Nodes to visit
        self._traversal_active = False
        
        # Callbacks
        self._callbacks: Dict[str, List[WalkerCallback]] = {
            'enter': [],
            'leave': [],
            'edge': [],
            'condition': []
        }
        
        # Graftable branches
        self._branches: List[GraftBranch] = []
        
        # Query cache
        self._query_cache: Dict[str, Any] = {}
    
    # ─────────────────────────────────────────────────────────────────────────
    # Position & Movement
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def current(self) -> Optional[GraphNode]:
        """Get the current node"""
        if self._current_id is None:
            return None
        return self.graph.get_node(self._current_id)
    
    @property
    def position(self) -> Optional[str]:
        """Get current node ID"""
        return self._current_id
    
    def move_to(self, node: Union[str, GraphNode]) -> bool:
        """
        Move directly to a node (teleport).
        Fires leave callback on old node, enter on new.
        """
        target_id = node if isinstance(node, str) else node.id
        
        if target_id not in self.graph.nodes:
            return False
        
        # Leave current
        if self._current_id:
            self._fire_callbacks('leave', self.current)
        
        # Move
        old_id = self._current_id
        self._current_id = target_id
        self.context.path.append(target_id)
        self.context.visited.add(target_id)
        
        # Enter new
        self._fire_callbacks('enter', self.current)
        
        # Log event
        self.context.log_event('move', {
            'from': old_id,
            'to': target_id,
            'method': 'teleport'
        })
        
        # Check for branch grafting
        self._check_grafts()
        
        return True
    
    def follow_edge(self, edge: Edge) -> bool:
        """
        Move along a specific edge.
        Checks edge conditions and fires edge callbacks.
        """
        # Check if traversable
        if not edge.is_traversable(self.context.__dict__):
            self.context.log_event('blocked', {
                'edge': repr(edge),
                'reason': 'condition_failed'
            })
            return False
        
        # Fire edge callback
        self._fire_callbacks('edge', edge)
        
        # Fire leave
        if self._current_id:
            self._fire_callbacks('leave', self.current)
        
        # Move
        old_id = self._current_id
        self._current_id = edge.target.id
        self.context.path.append(self._current_id)
        self.context.visited.add(self._current_id)
        
        # Fire enter
        self._fire_callbacks('enter', self.current)
        
        # Log
        self.context.log_event('move', {
            'from': old_id,
            'to': self._current_id,
            'edge_type': edge.edge_type.name,
            'method': 'edge'
        })
        
        self._check_grafts()
        return True
    
    def go(self, direction: str = None, edge_type: EdgeType = None) -> bool:
        """
        Move in a direction or along an edge type.
        
        If direction matches a neighbor's name, go there.
        If edge_type specified, follow first edge of that type.
        """
        if not self.current:
            return False
        
        edges = self.current.get_edges(edge_type)
        
        for edge in edges:
            # Check by name
            if direction and direction.lower() in edge.target.name.lower():
                return self.follow_edge(edge)
            
            # Or just take first valid edge
            if direction is None and edge.is_traversable(self.context.__dict__):
                return self.follow_edge(edge)
        
        return False
    
    def back(self) -> bool:
        """Go back to previous node in path"""
        if len(self.context.path) < 2:
            return False
        
        # Remove current from path
        self.context.path.pop()
        # Go to previous
        return self.move_to(self.context.path[-1])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Traversal (Iterator Interface)
    # ─────────────────────────────────────────────────────────────────────────
    
    def traverse(self, 
                 edge_types: List[EdgeType] = None,
                 max_depth: int = None,
                 max_nodes: int = None) -> Generator[GraphNode, None, None]:
        """
        Iterate through the graph starting from current position.
        
        Yields each node as it's visited.
        Respects strategy (DFS/BFS/etc), edge types, and depth limits.
        """
        if not self.current:
            return
        
        self._traversal_active = True
        self._frontier.clear()
        self._frontier.append((self._current_id, 0))  # (node_id, depth)
        
        visited_in_traversal = set()
        nodes_yielded = 0
        
        while self._frontier:
            if max_nodes and nodes_yielded >= max_nodes:
                break
            
            # Get next node based on strategy
            if self.strategy == TraversalStrategy.BFS:
                node_id, depth = self._frontier.popleft()
            else:  # DFS and others
                node_id, depth = self._frontier.pop()
            
            if node_id in visited_in_traversal:
                continue
            
            if max_depth and depth > max_depth:
                continue
            
            visited_in_traversal.add(node_id)
            
            # Move to this node
            if node_id != self._current_id:
                self.move_to(node_id)
            
            nodes_yielded += 1
            yield self.current
            
            # Add neighbors to frontier
            edges = self.current.get_edges()
            for edge in edges:
                if edge_types and edge.edge_type not in edge_types:
                    continue
                if edge.is_traversable(self.context.__dict__):
                    self._frontier.append((edge.target.id, depth + 1))
        
        self._traversal_active = False
    
    def __iter__(self) -> Iterator[GraphNode]:
        """Make the walker itself iterable"""
        return self.traverse()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Queries (From Current Position)
    # ─────────────────────────────────────────────────────────────────────────
    
    def nearby(self, 
               edge_types: List[EdgeType] = None,
               node_types: List[NodeType] = None,
               depth: int = 1) -> List[GraphNode]:
        """
        Get nodes near current position.
        
        Returns nodes within 'depth' edges, filtered by type.
        """
        if not self.current:
            return []
        
        # Use graph's subgraph extraction
        subgraph = self.graph.extract_subgraph(
            self._current_id, 
            depth=depth,
            edge_types=edge_types
        )
        
        results = list(subgraph.nodes.values())
        
        # Filter by node type
        if node_types:
            results = [n for n in results if n.node_type in node_types]
        
        # Exclude self
        results = [n for n in results if n.id != self._current_id]
        
        return results
    
    def find_here(self, 
                  node_type: NodeType = None,
                  tag: str = None) -> List[GraphNode]:
        """
        Find nodes directly connected to current position.
        Commonly used for "what items are here?"
        """
        if not self.current:
            return []
        
        results = []
        for edge in self.current.get_edges():
            node = edge.target
            if node_type and node.node_type != node_type:
                continue
            if tag and not node.has_tag(tag):
                continue
            results.append(node)
        
        return results
    
    def can_go(self, direction: str = None, edge_type: EdgeType = None) -> List[Edge]:
        """
        Get available edges from current position.
        
        Returns edges that are currently traversable.
        """
        if not self.current:
            return []
        
        edges = self.current.get_edges(edge_type)
        valid = []
        
        for edge in edges:
            if not edge.is_traversable(self.context.__dict__):
                continue
            if direction and direction.lower() not in edge.target.name.lower():
                continue
            valid.append(edge)
        
        return valid
    
    def search(self,
               query_fn: Callable[[GraphNode], bool],
               max_results: int = 10) -> List[GraphNode]:
        """
        Search the entire graph with a custom filter.
        """
        return self.graph.query(filter_fn=query_fn)[:max_results]
    
    def path_to(self, 
                target: Union[str, GraphNode],
                edge_types: List[EdgeType] = None) -> Optional[List[GraphNode]]:
        """
        Find path from current position to target.
        """
        if not self.current:
            return None
        
        target_id = target if isinstance(target, str) else target.id
        return self.graph.find_path(
            self._current_id, 
            target_id,
            edge_types,
            self.context.__dict__
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────
    
    def on_enter(self, 
                 name: str,
                 handler: Callable,
                 condition: Callable = None,
                 priority: int = 0,
                 once: bool = False) -> 'GraphWalker':
        """Register callback for entering nodes"""
        callback = WalkerCallback(
            name=name,
            trigger='enter',
            handler=handler,
            condition=condition,
            priority=priority,
            once=once
        )
        self._callbacks['enter'].append(callback)
        self._callbacks['enter'].sort(key=lambda c: -c.priority)
        return self  # Allow chaining
    
    def on_leave(self,
                 name: str,
                 handler: Callable,
                 condition: Callable = None,
                 priority: int = 0,
                 once: bool = False) -> 'GraphWalker':
        """Register callback for leaving nodes"""
        callback = WalkerCallback(
            name=name,
            trigger='leave',
            handler=handler,
            condition=condition,
            priority=priority,
            once=once
        )
        self._callbacks['leave'].append(callback)
        self._callbacks['leave'].sort(key=lambda c: -c.priority)
        return self
    
    def on_edge(self,
                name: str,
                handler: Callable,
                condition: Callable = None,
                priority: int = 0) -> 'GraphWalker':
        """Register callback for traversing edges"""
        callback = WalkerCallback(
            name=name,
            trigger='edge',
            handler=handler,
            condition=condition,
            priority=priority
        )
        self._callbacks['edge'].append(callback)
        self._callbacks['edge'].sort(key=lambda c: -c.priority)
        return self
    
    def _fire_callbacks(self, trigger: str, target: Any):
        """Fire all callbacks for a trigger"""
        for callback in self._callbacks.get(trigger, []):
            if callback.should_fire(self, target):
                callback.execute(self, target)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Graph Modification
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_node_here(self,
                      node_type: NodeType,
                      name: str,
                      edge_type: EdgeType = EdgeType.CONTAINS,
                      data: Dict = None,
                      tags: List[str] = None) -> GraphNode:
        """
        Add a new node connected to current position.
        """
        if not self.current:
            raise ValueError("Walker must be positioned to add nodes")
        
        node = self.graph.add_node(node_type, name, data=data, tags=tags)
        self.graph.add_edge(self.current, node, edge_type)
        
        self.context.log_event('node_created', {
            'node_id': node.id,
            'node_type': node_type.name,
            'connected_to': self._current_id
        })
        
        return node
    
    def connect_to(self,
                   target: Union[str, GraphNode],
                   edge_type: EdgeType,
                   bidirectional: bool = False,
                   **edge_data) -> Edge:
        """
        Create an edge from current node to target.
        """
        if not self.current:
            raise ValueError("Walker must be positioned to create edges")
        
        edge = self.graph.add_edge(
            self.current,
            target,
            edge_type,
            bidirectional=bidirectional,
            data=edge_data
        )
        
        self.context.log_event('edge_created', {
            'from': self._current_id,
            'to': target if isinstance(target, str) else target.id,
            'type': edge_type.name
        })
        
        return edge
    
    def register_branch(self, branch: GraftBranch):
        """Register a branch for potential grafting"""
        self._branches.append(branch)
    
    def graft(self, 
              subgraph: IslandGraph,
              attach_to: Union[str, GraphNode] = None,
              edge_type: EdgeType = EdgeType.CONTAINS) -> List[GraphNode]:
        """
        Graft a subgraph into the main graph at a specific point.
        
        Returns the list of newly added nodes.
        """
        attach_node = attach_to or self.current
        if isinstance(attach_node, str):
            attach_node = self.graph.get_node(attach_node)
        
        if not attach_node:
            raise ValueError("Attach point must exist")
        
        # Map old IDs to new IDs (in case of conflicts)
        id_map = {}
        new_nodes = []
        
        # Copy nodes
        for old_id, node in subgraph.nodes.items():
            new_id = f"{old_id}_grafted_{time.time():.0f}"
            id_map[old_id] = new_id
            
            new_node = self.graph.add_node(
                node.node_type,
                node.name,
                node_id=new_id,
                data=node.data.copy(),
                tags=list(node.tags)
            )
            new_nodes.append(new_node)
        
        # Copy edges (within subgraph)
        for edge in subgraph.edges:
            new_source = id_map.get(edge.source.id)
            new_target = id_map.get(edge.target.id)
            if new_source and new_target:
                self.graph.add_edge(
                    new_source,
                    new_target,
                    edge.edge_type,
                    weight=edge.weight,
                    data=edge.data.copy()
                )
        
        # Connect to attach point
        # Find "entry" nodes in subgraph (nodes with no incoming edges from within)
        internal_targets = {e.target.id for e in subgraph.edges}
        entry_nodes = [n for n in subgraph.nodes.values() 
                       if n.id not in internal_targets]
        
        for entry in entry_nodes:
            new_id = id_map[entry.id]
            self.graph.add_edge(attach_node, new_id, edge_type)
        
        self.context.log_event('branch_grafted', {
            'attach_point': attach_node.id,
            'nodes_added': len(new_nodes),
            'subgraph_name': subgraph.name
        })
        
        return new_nodes
    
    def _check_grafts(self):
        """Check and execute any pending branch grafts"""
        for branch in self._branches:
            if branch.should_graft(self):
                attach = self.graph.get_node(branch.attach_point)
                if attach:
                    self.graft(branch.subgraph, attach, branch.attach_edge_type)
                    branch.grafted = True
    
    # ─────────────────────────────────────────────────────────────────────────
    # Context Shortcuts
    # ─────────────────────────────────────────────────────────────────────────
    
    def remember(self, memory_type: str, content: Any):
        """Add a memory to context"""
        self.context.add_memory(memory_type, content, 
                               source=self._current_id)
    
    def collect(self, item_node: Union[str, GraphNode]):
        """Add an item to inventory"""
        item_id = item_node if isinstance(item_node, str) else item_node.id
        if item_id not in self.context.inventory:
            self.context.inventory.append(item_id)
            self.context.log_event('item_collected', {'item_id': item_id})
    
    def has_item(self, item_id: str) -> bool:
        """Check if item is in inventory"""
        return item_id in self.context.inventory
    
    def drop(self, item_id: str) -> bool:
        """Remove item from inventory, place at current location"""
        if item_id in self.context.inventory:
            self.context.inventory.remove(item_id)
            if self.current:
                self.graph.add_edge(self.current, item_id, EdgeType.CONTAINS)
            self.context.log_event('item_dropped', {
                'item_id': item_id,
                'location': self._current_id
            })
            return True
        return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # State
    # ─────────────────────────────────────────────────────────────────────────
    
    def snapshot(self) -> Dict:
        """Get current walker state (for save/load)"""
        return {
            'current_id': self._current_id,
            'context': {
                'visited': list(self.context.visited),
                'path': self.context.path,
                'inventory': self.context.inventory,
                'flags': self.context.flags,
                'counters': self.context.counters,
                'memories': self.context.memories,
                'events': self.context.events
            }
        }
    
    def restore(self, snapshot: Dict):
        """Restore walker state from snapshot"""
        self._current_id = snapshot.get('current_id')
        ctx = snapshot.get('context', {})
        self.context.visited = set(ctx.get('visited', []))
        self.context.path = ctx.get('path', [])
        self.context.inventory = ctx.get('inventory', [])
        self.context.flags = ctx.get('flags', {})
        self.context.counters = ctx.get('counters', {})
        self.context.memories = ctx.get('memories', [])
        self.context.events = ctx.get('events', [])
    
    def __repr__(self):
        node_name = self.current.name if self.current else "nowhere"
        return f"<Walker at '{node_name}' visited={len(self.context.visited)}>"


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK HELPERS - Common callback patterns
# ═══════════════════════════════════════════════════════════════════════════════

def print_description(walker: GraphWalker, node: GraphNode, context: WalkerContext):
    """Simple callback that prints node description"""
    desc = node.get_data('description', node.name)
    print(f"\n📍 {node.name}")
    print(f"   {desc}")

def collect_items(walker: GraphWalker, node: GraphNode, context: WalkerContext):
    """Callback that auto-collects takeable items"""
    for item in walker.find_here(NodeType.ITEM):
        if item.get_data('takeable', False):
            walker.collect(item)
            print(f"   [Collected: {item.name}]")

def reveal_secrets(walker: GraphWalker, node: GraphNode, context: WalkerContext):
    """Callback that reveals hidden things based on flags"""
    secrets = node.get_data('secrets', [])
    for secret in secrets:
        required_flag = secret.get('requires_flag')
        if required_flag and context.get_flag(required_flag):
            print(f"   🔮 {secret.get('text', 'Something hidden is revealed...')}")
            context.add_memory('secret_revealed', secret, node.id)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Graph Walker - Demonstration")
    print("=" * 70)
    
    # Build a small world
    island = IslandGraph(name="Demo Island")
    
    # Locations
    beach = create_location(island, "Sandy Beach", 
                           "Warm sand stretches to the water's edge.")
    jungle = create_location(island, "Dense Jungle",
                            "Vines hang from ancient trees.")
    temple = create_location(island, "Ruined Temple",
                            "Stone columns frame a dark entrance.")
    
    # Connect locations
    island.connect(beach.id, jungle.id, EdgeType.LEADS_TO, bidirectional=True)
    island.connect(jungle.id, temple.id, EdgeType.LEADS_TO, bidirectional=True)
    
    # Items
    torch = create_item(island, "Torch", "A flickering light source.")
    torch.set_data('takeable', True)
    island.connect(beach.id, torch.id, EdgeType.CONTAINS)
    
    key = create_item(island, "Golden Key", "It glows faintly.")
    key.set_data('takeable', True)
    island.connect(temple.id, key.id, EdgeType.CONTAINS)
    
    # Create walker
    walker = GraphWalker(island, start_node=beach)
    
    # Register callbacks
    walker.on_enter("describe", print_description)
    walker.on_enter("collect", collect_items, 
                    condition=lambda w, n: n.node_type == NodeType.LOCATION)
    
    print("\n--- Manual Navigation ---")
    print(walker)
    
    walker.go("jungle")
    print(walker)
    
    walker.go("temple")
    print(walker)
    
    print(f"\n--- Inventory: {walker.context.inventory} ---")
    
    print("\n--- Traversal from Beach ---")
    walker.move_to(beach)
    for node in walker.traverse(max_nodes=5):
        pass  # Callbacks handle output
    
    print("\n--- Path Finding ---")
    walker.move_to(beach)
    path = walker.path_to(temple)
    if path:
        print(f"Path: {' → '.join(n.name for n in path)}")
    
    print("\n--- Context State ---")
    print(f"Visited: {len(walker.context.visited)} nodes")
    print(f"Events: {len(walker.context.events)} logged")
    print(f"Inventory: {walker.context.inventory}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
