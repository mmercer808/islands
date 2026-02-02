"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                  C A L L B A C K   P A T T E R N S                            ║
║                                                                               ║
║       Reusable Patterns for Graph Traversal and Game Logic                    ║
║                                                                               ║
║  Callbacks are the verbs of the graph.                                        ║
║  They make things happen when the walker moves.                               ║
║  These patterns are building blocks for game behavior.                        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Callable, Tuple, Union
from enum import Enum, auto
import random
import time

from graph_core import (
    IslandGraph, GraphNode, Edge, NodeType, EdgeType
)
from graph_walker import (
    GraphWalker, WalkerContext, WalkerCallback, GraftBranch
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONDITION BUILDERS - Functions that return condition functions
# ═══════════════════════════════════════════════════════════════════════════════

def has_flag(flag_name: str) -> Callable:
    """Condition: context has a specific flag set"""
    def check(walker: GraphWalker, target: Any) -> bool:
        return walker.context.get_flag(flag_name)
    return check

def missing_flag(flag_name: str) -> Callable:
    """Condition: context does NOT have a flag"""
    def check(walker: GraphWalker, target: Any) -> bool:
        return not walker.context.get_flag(flag_name)
    return check

def has_item(item_id: str) -> Callable:
    """Condition: player has item in inventory"""
    def check(walker: GraphWalker, target: Any) -> bool:
        return item_id in walker.context.inventory
    return check

def has_visited(location_id: str) -> Callable:
    """Condition: player has visited a location"""
    def check(walker: GraphWalker, target: Any) -> bool:
        return location_id in walker.context.visited
    return check

def visit_count_at_least(count: int) -> Callable:
    """Condition: player has visited at least N locations"""
    def check(walker: GraphWalker, target: Any) -> bool:
        return len(walker.context.visited) >= count
    return check

def counter_at_least(counter_name: str, value: int) -> Callable:
    """Condition: a counter is at least a value"""
    def check(walker: GraphWalker, target: Any) -> bool:
        return walker.context.get_counter(counter_name) >= value
    return check

def node_has_tag(tag: str) -> Callable:
    """Condition: current node has a specific tag"""
    def check(walker: GraphWalker, target: Any) -> bool:
        if isinstance(target, GraphNode):
            return target.has_tag(tag)
        return walker.current and walker.current.has_tag(tag)
    return check

def node_is_type(node_type: NodeType) -> Callable:
    """Condition: current node is a specific type"""
    def check(walker: GraphWalker, target: Any) -> bool:
        if isinstance(target, GraphNode):
            return target.node_type == node_type
        return walker.current and walker.current.node_type == node_type
    return check

def random_chance(probability: float) -> Callable:
    """Condition: random chance (0.0 to 1.0)"""
    def check(walker: GraphWalker, target: Any) -> bool:
        return random.random() < probability
    return check

def all_of(*conditions: Callable) -> Callable:
    """Combine conditions with AND"""
    def check(walker: GraphWalker, target: Any) -> bool:
        return all(c(walker, target) for c in conditions)
    return check

def any_of(*conditions: Callable) -> Callable:
    """Combine conditions with OR"""
    def check(walker: GraphWalker, target: Any) -> bool:
        return any(c(walker, target) for c in conditions)
    return check

def none_of(*conditions: Callable) -> Callable:
    """Combine conditions with NOR"""
    def check(walker: GraphWalker, target: Any) -> bool:
        return not any(c(walker, target) for c in conditions)
    return check


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK FACTORIES - Functions that return callback handlers
# ═══════════════════════════════════════════════════════════════════════════════

def set_flag(flag_name: str, value: bool = True) -> Callable:
    """Handler: set a context flag"""
    def handler(walker: GraphWalker, target: Any, context: WalkerContext):
        context.set_flag(flag_name, value)
    return handler

def increment_counter(counter_name: str, amount: int = 1) -> Callable:
    """Handler: increment a counter"""
    def handler(walker: GraphWalker, target: Any, context: WalkerContext):
        context.increment(counter_name, amount)
    return handler

def add_memory(memory_type: str, content_fn: Callable = None) -> Callable:
    """Handler: add a memory to context"""
    def handler(walker: GraphWalker, target: Any, context: WalkerContext):
        content = content_fn(walker, target) if content_fn else str(target)
        context.add_memory(memory_type, content, source=walker.position)
    return handler

def collect_item(item_id: str = None) -> Callable:
    """Handler: add item to inventory (or auto-collect from target)"""
    def handler(walker: GraphWalker, target: Any, context: WalkerContext):
        if item_id:
            walker.collect(item_id)
        elif isinstance(target, GraphNode) and target.node_type == NodeType.ITEM:
            walker.collect(target.id)
    return handler

def print_text(text: str = None, text_fn: Callable = None) -> Callable:
    """Handler: print text (static or dynamic)"""
    def handler(walker: GraphWalker, target: Any, context: WalkerContext):
        if text_fn:
            print(text_fn(walker, target, context))
        elif text:
            print(text)
        else:
            print(f"At: {walker.current.name if walker.current else 'nowhere'}")
    return handler

def buffer_store(key: str, value_fn: Callable = None) -> Callable:
    """Handler: store something in the local buffer"""
    def handler(walker: GraphWalker, target: Any, context: WalkerContext):
        value = value_fn(walker, target) if value_fn else target
        context.buffer_set(key, value)
    return handler

def buffer_append(key: str, value_fn: Callable = None) -> Callable:
    """Handler: append to a list in the buffer"""
    def handler(walker: GraphWalker, target: Any, context: WalkerContext):
        value = value_fn(walker, target) if value_fn else target
        existing = context.buffer_get(key, [])
        if not isinstance(existing, list):
            existing = [existing]
        existing.append(value)
        context.buffer_set(key, existing)
    return handler

def chain(*handlers: Callable) -> Callable:
    """Handler: chain multiple handlers together"""
    def handler(walker: GraphWalker, target: Any, context: WalkerContext):
        for h in handlers:
            h(walker, target, context)
    return handler

def conditional(condition: Callable, 
                if_true: Callable, 
                if_false: Callable = None) -> Callable:
    """Handler: conditional execution"""
    def handler(walker: GraphWalker, target: Any, context: WalkerContext):
        if condition(walker, target):
            if_true(walker, target, context)
        elif if_false:
            if_false(walker, target, context)
    return handler


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACK TEMPLATES - Pre-built callbacks for common patterns
# ═══════════════════════════════════════════════════════════════════════════════

class CallbackTemplates:
    """
    Ready-to-use callback configurations.
    
    These wrap the factories into complete WalkerCallback objects.
    """
    
    @staticmethod
    def track_visits() -> WalkerCallback:
        """Track how many times each location is visited"""
        def handler(walker, target, context):
            key = f"visits_{target.id}"
            context.increment(key)
        
        return WalkerCallback(
            name="track_visits",
            trigger="enter",
            handler=handler
        )
    
    @staticmethod
    def auto_collect_items(takeable_only: bool = True) -> WalkerCallback:
        """Automatically collect items when entering locations"""
        def handler(walker, target, context):
            items = walker.find_here(NodeType.ITEM)
            for item in items:
                if not takeable_only or item.get_data('takeable', False):
                    walker.collect(item)
        
        return WalkerCallback(
            name="auto_collect",
            trigger="enter",
            handler=handler,
            condition=node_is_type(NodeType.LOCATION)
        )
    
    @staticmethod
    def danger_zone(damage: int, flag_to_set: str = None) -> WalkerCallback:
        """Take damage when entering dangerous areas"""
        def handler(walker, target, context):
            context.increment("health", -damage)
            if flag_to_set:
                context.set_flag(flag_to_set)
            context.log_event("took_damage", {
                "amount": damage,
                "location": target.id
            })
        
        return WalkerCallback(
            name="danger_zone",
            trigger="enter",
            handler=handler,
            condition=node_has_tag("dangerous")
        )
    
    @staticmethod
    def discovery_bonus(points: int = 10) -> WalkerCallback:
        """Award points for discovering new locations"""
        def condition(walker, target):
            return target.id not in walker.context.visited
        
        def handler(walker, target, context):
            context.increment("score", points)
            context.log_event("discovery", {
                "location": target.id,
                "points": points
            })
        
        return WalkerCallback(
            name="discovery_bonus",
            trigger="enter",
            handler=handler,
            condition=condition
        )
    
    @staticmethod
    def locked_door(key_item_id: str, 
                    unlock_message: str = "The door unlocks!",
                    locked_message: str = "It's locked.") -> WalkerCallback:
        """Handle locked passages that require a key"""
        def handler(walker, target, context):
            if key_item_id in context.inventory:
                print(unlock_message)
                context.set_flag(f"unlocked_{target.target.id}")
            else:
                print(locked_message)
                # Could also prevent movement here
        
        return WalkerCallback(
            name="locked_door",
            trigger="edge",
            handler=handler,
            condition=lambda w, e: e.get_data('locked', False)
        )
    
    @staticmethod
    def breadcrumb_trail() -> WalkerCallback:
        """Leave a trail of visited locations in buffer"""
        def handler(walker, target, context):
            trail = context.buffer_get("breadcrumbs", [])
            trail.append({
                "id": target.id,
                "name": target.name,
                "time": time.time()
            })
            # Keep last 10
            context.buffer_set("breadcrumbs", trail[-10:])
        
        return WalkerCallback(
            name="breadcrumb_trail",
            trigger="enter",
            handler=handler
        )
    
    @staticmethod
    def time_tracker() -> WalkerCallback:
        """Track time spent in each location"""
        def on_enter(walker, target, context):
            context.buffer_set("entered_at", time.time())
        
        def on_leave(walker, target, context):
            entered = context.buffer_get("entered_at")
            if entered:
                duration = time.time() - entered
                key = f"time_in_{target.id}"
                current = context.buffer_get(key, 0)
                context.buffer_set(key, current + duration)
        
        # Return both callbacks
        return [
            WalkerCallback("time_enter", "enter", on_enter),
            WalkerCallback("time_leave", "leave", on_leave)
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFT TEMPLATES - Pre-built branch patterns
# ═══════════════════════════════════════════════════════════════════════════════

class GraftTemplates:
    """
    Templates for creating graftable branches.
    
    These are subgraphs that can be attached to the main graph
    when certain conditions are met.
    """
    
    @staticmethod
    def secret_room(name: str,
                    description: str,
                    items: List[Tuple[str, str]] = None,  # [(name, description), ...]
                    requires_flag: str = None) -> GraftBranch:
        """
        Create a hidden room that appears when conditions are met.
        """
        subgraph = IslandGraph(name=f"secret_{name}")
        
        # Create the room
        room = subgraph.add_node(
            NodeType.LOCATION,
            name,
            data={"description": description, "secret": True},
            tags=["secret", "indoor"]
        )
        
        # Add items
        for item_name, item_desc in (items or []):
            item = subgraph.add_node(
                NodeType.ITEM,
                item_name,
                data={"description": item_desc, "takeable": True}
            )
            subgraph.add_edge(room, item, EdgeType.CONTAINS)
        
        # Build condition
        condition = None
        if requires_flag:
            condition = lambda w: w.context.get_flag(requires_flag)
        
        return GraftBranch(
            name=f"graft_{name}",
            subgraph=subgraph,
            attach_point="",  # Must be set when registering
            attach_edge_type=EdgeType.LEADS_TO,
            condition=condition
        )
    
    @staticmethod
    def procedural_dungeon(rooms: int = 5,
                           danger_chance: float = 0.3,
                           loot_chance: float = 0.5) -> GraftBranch:
        """
        Create a procedurally generated dungeon branch.
        """
        subgraph = IslandGraph(name="procedural_dungeon")
        
        room_names = [
            "Dark Corridor", "Dusty Chamber", "Collapsed Hall",
            "Flooded Room", "Ancient Vault", "Forgotten Crypt",
            "Spider Nest", "Bone Pit", "Treasure Room"
        ]
        
        prev_room = None
        for i in range(rooms):
            name = random.choice(room_names)
            tags = ["dungeon", "indoor"]
            
            if random.random() < danger_chance:
                tags.append("dangerous")
            
            room = subgraph.add_node(
                NodeType.LOCATION,
                f"{name} #{i+1}",
                data={"description": f"A {name.lower()} stretches before you."},
                tags=tags
            )
            
            # Add loot
            if random.random() < loot_chance:
                loot = subgraph.add_node(
                    NodeType.ITEM,
                    random.choice(["Gold Coins", "Ancient Relic", "Rusty Sword", "Health Potion"]),
                    data={"takeable": True}
                )
                subgraph.add_edge(room, loot, EdgeType.CONTAINS)
            
            # Connect to previous
            if prev_room:
                subgraph.add_edge(prev_room, room, EdgeType.LEADS_TO, bidirectional=True)
            
            prev_room = room
        
        return GraftBranch(
            name="procedural_dungeon",
            subgraph=subgraph,
            attach_point="",
            attach_edge_type=EdgeType.LEADS_TO
        )
    
    @staticmethod
    def dialogue_tree(character_id: str,
                      dialogue_nodes: List[Dict]) -> GraftBranch:
        """
        Create a dialogue tree that grafts onto a character.
        
        dialogue_nodes format:
        [
            {"id": "greeting", "text": "Hello!", "options": [("reply", "Hi there")]},
            {"id": "reply", "text": "Nice to meet you.", "options": []},
        ]
        """
        subgraph = IslandGraph(name=f"dialogue_{character_id}")
        
        nodes_by_id = {}
        
        for node_data in dialogue_nodes:
            node = subgraph.add_node(
                NodeType.DIALOGUE,
                node_data["id"],
                data={
                    "text": node_data["text"],
                    "options": node_data.get("options", [])
                }
            )
            nodes_by_id[node_data["id"]] = node
        
        # Connect based on options
        for node_data in dialogue_nodes:
            source = nodes_by_id[node_data["id"]]
            for option_id, option_text in node_data.get("options", []):
                if option_id in nodes_by_id:
                    target = nodes_by_id[option_id]
                    subgraph.add_edge(
                        source, target, 
                        EdgeType.TRIGGERS,
                        data={"option_text": option_text}
                    )
        
        return GraftBranch(
            name=f"dialogue_{character_id}",
            subgraph=subgraph,
            attach_point=character_id,
            attach_edge_type=EdgeType.CONTAINS
        )


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY PATTERNS - Common search/retrieval patterns
# ═══════════════════════════════════════════════════════════════════════════════

class QueryPatterns:
    """
    Reusable query patterns for common game needs.
    """
    
    @staticmethod
    def items_in_reach(walker: GraphWalker, depth: int = 1) -> List[GraphNode]:
        """Find all items within N moves"""
        return walker.nearby(
            depth=depth,
            node_types=[NodeType.ITEM]
        )
    
    @staticmethod
    def characters_nearby(walker: GraphWalker, depth: int = 2) -> List[GraphNode]:
        """Find all characters within N moves"""
        return walker.nearby(
            depth=depth,
            node_types=[NodeType.CHARACTER]
        )
    
    @staticmethod
    def unexplored_adjacent(walker: GraphWalker) -> List[GraphNode]:
        """Find unvisited adjacent locations"""
        nearby = walker.nearby(depth=1, node_types=[NodeType.LOCATION])
        return [n for n in nearby if n.id not in walker.context.visited]
    
    @staticmethod
    def items_with_tag(walker: GraphWalker, tag: str) -> List[GraphNode]:
        """Find all items in graph with a specific tag"""
        return walker.graph.query(
            node_type=NodeType.ITEM,
            tags=[tag]
        )
    
    @staticmethod
    def path_through(walker: GraphWalker, 
                     waypoints: List[str]) -> Optional[List[GraphNode]]:
        """
        Find a path that passes through all waypoints in order.
        """
        if not waypoints:
            return []
        
        full_path = []
        current = walker.position
        
        for waypoint in waypoints:
            segment = walker.graph.find_path(
                current, waypoint,
                [EdgeType.LEADS_TO],
                walker.context.__dict__
            )
            if segment is None:
                return None
            
            # Avoid duplicating the connection point
            if full_path:
                segment = segment[1:]
            
            full_path.extend(segment)
            current = waypoint
        
        return full_path


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from graph_core import create_location, create_item
    
    print("=" * 70)
    print("Callback Patterns - Demonstration")
    print("=" * 70)
    
    # Build test world
    island = IslandGraph(name="Pattern Demo Island")
    
    beach = create_location(island, "Beach", tags=["outdoor", "safe"])
    cave = create_location(island, "Dark Cave", tags=["indoor", "dangerous"])
    treasure = create_location(island, "Treasure Room", tags=["indoor", "secret"])
    
    island.connect(beach.id, cave.id, EdgeType.LEADS_TO, bidirectional=True)
    island.connect(cave.id, treasure.id, EdgeType.LEADS_TO, bidirectional=True)
    
    gold = create_item(island, "Gold Coins")
    gold.set_data('takeable', True)
    island.connect(treasure.id, gold.id, EdgeType.CONTAINS)
    
    # Create walker with pattern callbacks
    walker = GraphWalker(island, start_node=beach)
    
    # Add templates
    walker.on_enter("visits", CallbackTemplates.track_visits().handler)
    walker.on_enter("discovery", CallbackTemplates.discovery_bonus(25).handler,
                    condition=CallbackTemplates.discovery_bonus(25).condition)
    walker.on_enter("danger", CallbackTemplates.danger_zone(10).handler,
                    condition=node_has_tag("dangerous"))
    
    # Initialize health
    walker.context.counters["health"] = 100
    walker.context.counters["score"] = 0
    
    print("\n--- Starting at Beach ---")
    print(f"Health: {walker.context.get_counter('health')}")
    print(f"Score: {walker.context.get_counter('score')}")
    
    print("\n--- Entering Cave (dangerous) ---")
    walker.go("cave")
    print(f"Health: {walker.context.get_counter('health')} (took damage!)")
    print(f"Score: {walker.context.get_counter('score')} (discovery bonus)")
    
    print("\n--- Entering Treasure Room ---")
    walker.go("treasure")
    print(f"Score: {walker.context.get_counter('score')} (another discovery)")
    
    print("\n--- Using Query Patterns ---")
    items = QueryPatterns.items_in_reach(walker, depth=1)
    print(f"Items in reach: {[i.name for i in items]}")
    
    unexplored = QueryPatterns.unexplored_adjacent(walker)
    print(f"Unexplored adjacent: {[n.name for n in unexplored]}")
    
    print("\n--- Context State ---")
    print(f"Visited: {len(walker.context.visited)} locations")
    print(f"Counters: {walker.context.counters}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
