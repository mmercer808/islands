"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  MATTS LOOKAHEAD - Possibility Analysis Engine                                 ║
║  Layer 5: Depends on primitives, traversal, world                             ║
║                                                                               ║
║  Pre-traverse the graph to discover what COULD happen without moving.         ║
║  Includes semantic similarity for "related" hints.                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from collections import defaultdict, deque

from .primitives import PossibilityType
from .traversal import TraversalContext
from .world import WhiteRoom, WorldNode


# ═══════════════════════════════════════════════════════════════════════════════
#                              POSSIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Possibility:
    """
    A single discovered possibility.
    
    Represents something that COULD happen from the current position.
    """
    type: PossibilityType
    entity_id: str
    entity_name: str
    
    # Distance/score
    distance: int = 0
    similarity: float = 0.0  # For semantic matches
    
    # Conditions
    conditions_met: List[str] = field(default_factory=list)
    conditions_unmet: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    
    # Hint text
    hint: Optional[str] = None
    
    # Path info
    path: List[str] = field(default_factory=list)
    via_edge: Optional[str] = None
    via_layer: Optional[str] = None
    
    def is_blocked(self) -> bool:
        """Is this possibility currently blocked?"""
        return len(self.conditions_unmet) > 0
    
    def is_near_miss(self, threshold: int = 1) -> bool:
        """Is this a near-miss (almost reachable)?"""
        return 0 < len(self.conditions_unmet) <= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.name,
            'entity_id': self.entity_id,
            'entity_name': self.entity_name,
            'distance': self.distance,
            'similarity': self.similarity,
            'blocked': self.is_blocked(),
            'hint': self.hint
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                              RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LookaheadResult:
    """Complete lookahead result from a position"""
    origin_id: str
    origin_name: str
    max_depth: int
    
    # All possibilities
    all: List[Possibility] = field(default_factory=list)
    
    # Indexed views
    by_type: Dict[PossibilityType, List[Possibility]] = field(
        default_factory=lambda: defaultdict(list)
    )
    by_entity: Dict[str, List[Possibility]] = field(
        default_factory=lambda: defaultdict(list)
    )
    
    # Stats
    entities_seen: int = 0
    edges_traversed: int = 0
    blocked_count: int = 0
    near_miss_count: int = 0
    
    def add(self, p: Possibility):
        """Add a possibility"""
        self.all.append(p)
        self.by_type[p.type].append(p)
        self.by_entity[p.entity_id].append(p)
        
        if p.is_blocked():
            self.blocked_count += 1
        if p.is_near_miss():
            self.near_miss_count += 1
    
    # ─────────────────────────────────────────────────────────────────────────
    # Filtered access
    # ─────────────────────────────────────────────────────────────────────────
    
    def reachable(self) -> List[Possibility]:
        """Get reachable (unblocked) possibilities"""
        return [p for p in self.all if not p.is_blocked()]
    
    def blocked(self) -> List[Possibility]:
        """Get blocked possibilities"""
        return [p for p in self.all if p.is_blocked()]
    
    def near_misses(self, threshold: int = 1) -> List[Possibility]:
        """Get near-miss possibilities"""
        return [p for p in self.all if p.is_near_miss(threshold)]
    
    def locations(self) -> List[Possibility]:
        """Get reachable locations"""
        return self.by_type.get(PossibilityType.REACHABLE_LOCATION, [])
    
    def items(self) -> List[Possibility]:
        """Get visible items"""
        visible = self.by_type.get(PossibilityType.VISIBLE_ITEM, [])
        hidden = self.by_type.get(PossibilityType.HIDDEN_ITEM, [])
        return visible + hidden
    
    def similar(self, min_score: float = 0.5) -> List[Possibility]:
        """Get semantically similar nodes"""
        return [p for p in self.by_type.get(PossibilityType.SIMILAR_NODE, [])
                if p.similarity >= min_score]
    
    # ─────────────────────────────────────────────────────────────────────────
    # Hint generation
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_hints(self, max_hints: int = 3) -> List[str]:
        """Generate hints for the player"""
        hints = []
        
        # Near-misses are gold
        for p in self.near_misses():
            if p.hint and len(hints) < max_hints:
                hints.append(p.hint)
        
        # Hidden items at close range
        for p in self.by_type.get(PossibilityType.HIDDEN_ITEM, []):
            if p.distance <= 1 and p.hint and len(hints) < max_hints:
                hints.append(p.hint)
        
        # Semantic similar (related concepts)
        for p in self.similar(0.7):
            if p.hint and len(hints) < max_hints:
                hints.append(p.hint)
        
        return hints[:max_hints]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'origin_id': self.origin_id,
            'origin_name': self.origin_name,
            'max_depth': self.max_depth,
            'total': len(self.all),
            'blocked': self.blocked_count,
            'near_misses': self.near_miss_count,
            'possibilities': [p.to_dict() for p in self.all]
        }
    
    def summary(self) -> str:
        lines = [
            f"═══ Lookahead from {self.origin_name} ═══",
            f"Depth: {self.max_depth}, Found: {len(self.all)}",
            f"Blocked: {self.blocked_count}, Near-misses: {self.near_miss_count}",
        ]
        
        for ptype in [PossibilityType.REACHABLE_LOCATION, 
                      PossibilityType.VISIBLE_ITEM,
                      PossibilityType.SIMILAR_NODE]:
            items = self.by_type.get(ptype, [])
            if items:
                names = [p.entity_name for p in items[:3]]
                suffix = "..." if len(items) > 3 else ""
                lines.append(f"  {ptype.name}: {', '.join(names)}{suffix}")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#                              ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class LookaheadEngine:
    """
    Pre-traverses the graph to discover possibilities.
    
    Explores without moving the player. Finds:
    - Reachable locations
    - Visible/hidden items
    - Blocked paths and their requirements
    - Near-misses (almost reachable)
    - Semantically similar nodes
    """
    
    def __init__(self, world: WhiteRoom = None):
        self.world = world
    
    def set_world(self, world: WhiteRoom):
        """Set or change the world"""
        self.world = world
    
    def lookahead(self,
                  from_node: Union[str, WorldNode],
                  context: TraversalContext = None,
                  max_depth: int = 3,
                  include_semantic: bool = True,
                  semantic_k: int = 5,
                  semantic_threshold: float = 0.5) -> LookaheadResult:
        """
        Perform lookahead from a position.
        
        Args:
            from_node: Starting node (ID or object)
            context: Current game context (for condition checking)
            max_depth: Maximum graph distance to explore
            include_semantic: Include semantic similarity results
            semantic_k: Number of semantic matches to find
            semantic_threshold: Minimum similarity score
        
        Returns:
            LookaheadResult with all discovered possibilities
        """
        if self.world is None:
            raise ValueError("No world set")
        
        # Resolve start node
        if isinstance(from_node, str):
            start_id = from_node
            start = self.world.get_node(from_node)
        else:
            start_id = from_node.id
            start = from_node
        
        if start is None:
            raise ValueError(f"Node not found: {start_id}")
        
        result = LookaheadResult(
            origin_id=start_id,
            origin_name=start.name,
            max_depth=max_depth
        )
        
        ctx = context or TraversalContext()
        
        # BFS graph exploration
        self._bfs_explore(start_id, max_depth, ctx, result)
        
        # Semantic similarity exploration
        if include_semantic and self.world.embedding_store:
            self._semantic_explore(start, semantic_k, semantic_threshold, result)
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # BFS exploration
    # ─────────────────────────────────────────────────────────────────────────
    
    def _bfs_explore(self, start_id: str, max_depth: int,
                     ctx: TraversalContext, result: LookaheadResult):
        """BFS exploration of graph"""
        visited: Set[str] = set()
        queue: deque = deque([(start_id, 0, [start_id])])
        
        while queue:
            current_id, depth, path = queue.popleft()
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            result.entities_seen += 1
            
            node = self.world.get_node(current_id)
            if not node:
                continue
            
            # Skip origin in results
            if depth > 0:
                possibilities = self._analyze_node(node, depth, path, ctx)
                for p in possibilities:
                    result.add(p)
            
            # Queue neighbors
            for edge in self.world.edges_from(current_id):
                result.edges_traversed += 1
                
                if edge.target_id not in visited:
                    # Check edge conditions
                    edge_blocked = self._check_edge_conditions(edge, ctx)
                    
                    if edge_blocked:
                        # Still report as blocked possibility
                        target = self.world.get_node(edge.target_id)
                        if target:
                            result.add(Possibility(
                                type=PossibilityType.BLOCKED_PATH,
                                entity_id=target.id,
                                entity_name=target.name,
                                distance=depth + 1,
                                conditions_unmet=edge_blocked,
                                path=path + [edge.target_id],
                                via_edge=edge.edge_type,
                                hint=f"The way to {target.name} seems blocked..."
                            ))
                    else:
                        queue.append((edge.target_id, depth + 1, path + [edge.target_id]))
    
    def _analyze_node(self, node: WorldNode, depth: int,
                      path: List[str], ctx: TraversalContext) -> List[Possibility]:
        """Analyze a node for possibilities"""
        possibilities = []
        
        # Location
        if node.node_type == "location":
            possibilities.append(Possibility(
                type=PossibilityType.REACHABLE_LOCATION,
                entity_id=node.id,
                entity_name=node.name,
                distance=depth,
                path=path.copy()
            ))
        
        # Item
        elif node.node_type == "item":
            ptype = PossibilityType.VISIBLE_ITEM
            hint = None
            
            if 'hidden' in node.tags:
                ptype = PossibilityType.HIDDEN_ITEM
                hint = "Something might be hidden nearby..."
            elif 'locked' in node.tags:
                ptype = PossibilityType.LOCKED_DOOR
                hint = f"The {node.name} appears to be locked."
            
            possibilities.append(Possibility(
                type=ptype,
                entity_id=node.id,
                entity_name=node.name,
                distance=depth,
                path=path.copy(),
                hint=hint
            ))
        
        # Character
        elif node.node_type == "character":
            possibilities.append(Possibility(
                type=PossibilityType.VISIBLE_ITEM,  # NPCs are "visible"
                entity_id=node.id,
                entity_name=node.name,
                distance=depth,
                path=path.copy()
            ))
        
        # Check for near-misses based on inventory
        if 'requires' in node.data:
            required = node.data['requires']
            if isinstance(required, str):
                required = [required]
            
            met = [r for r in required if ctx.has_item(r)]
            unmet = [r for r in required if not ctx.has_item(r)]
            
            if unmet and len(unmet) <= 1:
                possibilities.append(Possibility(
                    type=PossibilityType.NEAR_MISS,
                    entity_id=node.id,
                    entity_name=node.name,
                    distance=depth,
                    conditions_met=met,
                    conditions_unmet=unmet,
                    requirements=required,
                    path=path.copy(),
                    hint=f"You're close to accessing {node.name}, but you need: {', '.join(unmet)}"
                ))
        
        return possibilities
    
    def _check_edge_conditions(self, edge, ctx: TraversalContext) -> List[str]:
        """Check edge conditions, return list of unmet conditions"""
        unmet = []
        
        # Check requires in edge data
        if 'requires' in edge.data:
            required = edge.data['requires']
            if isinstance(required, str):
                required = [required]
            for r in required:
                if not ctx.has_item(r) and not ctx.get_flag(r):
                    unmet.append(r)
        
        # Check locked state
        if edge.data.get('locked'):
            key = edge.data.get('key')
            if key and not ctx.has_item(key):
                unmet.append(f"key:{key}")
        
        return unmet
    
    # ─────────────────────────────────────────────────────────────────────────
    # Semantic exploration
    # ─────────────────────────────────────────────────────────────────────────
    
    def _semantic_explore(self, start: WorldNode, k: int,
                          threshold: float, result: LookaheadResult):
        """Find semantically similar nodes"""
        similar = self.world.find_similar(start, k + 1)
        
        for node, score in similar:
            if node.id != start.id and score >= threshold:
                result.add(Possibility(
                    type=PossibilityType.SIMILAR_NODE,
                    entity_id=node.id,
                    entity_name=node.name,
                    similarity=score,
                    via_layer="semantic",
                    hint=f"{node.name} seems related to this somehow..."
                ))


# ═══════════════════════════════════════════════════════════════════════════════
#                              FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def lookahead_from(world: WhiteRoom,
                   node: Union[str, WorldNode],
                   context: TraversalContext = None,
                   max_depth: int = 3) -> LookaheadResult:
    """One-shot lookahead"""
    return LookaheadEngine(world).lookahead(node, context, max_depth)


# ═══════════════════════════════════════════════════════════════════════════════
#                              EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'Possibility', 'LookaheadResult', 'LookaheadEngine',
    'lookahead_from',
]
