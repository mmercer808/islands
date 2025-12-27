"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                    L O O K A H E A D   S Y S T E M                            ║
║                                                                               ║
║          Pre-Traverse the Possibility Space for Hints & Clues                 ║
║                                                                               ║
║  The lookahead doesn't just see where you ARE—it sees where you COULD BE.     ║
║  It finds locked doors, hidden items, unmet conditions, near-misses.          ║
║  It's the oracle that knows what's possible.                                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Dict, List, Set, Optional, Any, Callable, Iterator, Tuple,
    Union, Generic, TypeVar
)
from enum import Enum, auto
from collections import defaultdict, deque
import heapq


# ═══════════════════════════════════════════════════════════════════════════════
# POSSIBILITY TYPES - What kinds of potential do we find?
# ═══════════════════════════════════════════════════════════════════════════════

class PossibilityType(Enum):
    """Categories of things the lookahead can discover"""
    
    # Spatial possibilities
    REACHABLE_LOCATION = auto()     # Can walk there now
    BLOCKED_PATH = auto()           # Path exists but blocked
    LOCKED_DOOR = auto()            # Door exists, needs key/condition
    
    # Item possibilities
    VISIBLE_ITEM = auto()           # Item is here and visible
    HIDDEN_ITEM = auto()            # Item exists but hidden
    TAKEABLE_ITEM = auto()          # Item can be picked up
    REQUIRED_ITEM = auto()          # Item needed for something
    
    # Logical possibilities
    UNLOCKABLE = auto()             # Could be unlocked if condition met
    REVEALABLE = auto()             # Could be revealed by action
    TRIGGERABLE = auto()            # Event could be triggered
    CONDITIONAL_BLOCKED = auto()    # Blocked by unmet condition
    
    # Narrative possibilities
    DISCOVERABLE_CLUE = auto()      # Clue waiting to be found
    NPC_DIALOGUE = auto()           # Conversation available
    STORY_BEAT = auto()             # Narrative trigger available
    
    # Meta-possibilities
    NEAR_MISS = auto()              # Almost meets a condition
    DEAD_END = auto()               # No further possibilities
    PUZZLE_PIECE = auto()           # Part of a puzzle chain


@dataclass
class Possibility:
    """
    A single possibility discovered by lookahead.
    
    This represents something that COULD happen, might be available,
    or is blocked by known conditions.
    """
    possibility_type: PossibilityType
    entity_id: str
    entity_name: str
    
    # What makes this possible/blocked
    conditions_met: List[str] = field(default_factory=list)
    conditions_unmet: List[str] = field(default_factory=list)
    
    # How close is it?
    distance: int = 0  # Graph distance from current position
    accessibility: float = 1.0  # 0.0 = impossible, 1.0 = immediate
    
    # What would make it accessible?
    requirements: List[str] = field(default_factory=list)
    
    # Narrative hint (for the player)
    hint_text: Optional[str] = None
    
    # The path/chain to get here
    path: List[str] = field(default_factory=list)
    
    # Link that created this possibility
    via_link_type: Optional[str] = None
    via_link_kind: Optional[str] = None
    
    # For sorting/ranking
    priority: int = 50
    
    def is_blocked(self) -> bool:
        return len(self.conditions_unmet) > 0
    
    def is_near_miss(self, threshold: int = 1) -> bool:
        """True if only one or few conditions away from accessible"""
        return len(self.conditions_unmet) <= threshold and len(self.conditions_unmet) > 0
    
    def __lt__(self, other):
        # For heap sorting: prefer closer, higher priority, more accessible
        return (-self.priority, self.distance, -self.accessibility) < \
               (-other.priority, other.distance, -other.accessibility)


@dataclass
class LookaheadResult:
    """
    Complete result of a lookahead operation.
    
    Organizes possibilities by type and provides query methods.
    """
    origin_id: str
    origin_name: str
    max_depth: int
    
    # All possibilities found
    all_possibilities: List[Possibility] = field(default_factory=list)
    
    # Indexed by type
    by_type: Dict[PossibilityType, List[Possibility]] = field(default_factory=lambda: defaultdict(list))
    
    # Indexed by entity
    by_entity: Dict[str, List[Possibility]] = field(default_factory=lambda: defaultdict(list))
    
    # Statistics
    total_entities_seen: int = 0
    total_links_traversed: int = 0
    blocked_count: int = 0
    near_miss_count: int = 0
    
    def add(self, possibility: Possibility):
        """Add a possibility and index it"""
        self.all_possibilities.append(possibility)
        self.by_type[possibility.possibility_type].append(possibility)
        self.by_entity[possibility.entity_id].append(possibility)
        
        if possibility.is_blocked():
            self.blocked_count += 1
        if possibility.is_near_miss():
            self.near_miss_count += 1
    
    def get_reachable(self) -> List[Possibility]:
        """Get all immediately reachable locations"""
        return [p for p in self.all_possibilities 
                if p.possibility_type == PossibilityType.REACHABLE_LOCATION
                and not p.is_blocked()]
    
    def get_blocked(self) -> List[Possibility]:
        """Get all blocked possibilities"""
        return [p for p in self.all_possibilities if p.is_blocked()]
    
    def get_near_misses(self, threshold: int = 1) -> List[Possibility]:
        """Get possibilities that are almost accessible"""
        return [p for p in self.all_possibilities if p.is_near_miss(threshold)]
    
    def get_hidden_items(self) -> List[Possibility]:
        """Get hidden items that could be revealed"""
        return self.by_type.get(PossibilityType.HIDDEN_ITEM, []) + \
               self.by_type.get(PossibilityType.REVEALABLE, [])
    
    def get_required_items(self) -> List[Possibility]:
        """Get items that are needed for something"""
        return self.by_type.get(PossibilityType.REQUIRED_ITEM, [])
    
    def get_puzzle_chain(self, target_id: str) -> List[Possibility]:
        """
        Get the chain of things needed to reach/unlock a target.
        
        Works backwards from the target to find all prerequisites.
        """
        chain = []
        seen = set()
        frontier = [target_id]
        
        while frontier:
            current = frontier.pop(0)
            if current in seen:
                continue
            seen.add(current)
            
            for poss in self.by_entity.get(current, []):
                chain.append(poss)
                # Add requirements as next targets
                for req in poss.requirements:
                    if req not in seen:
                        frontier.append(req)
        
        return chain
    
    def get_hints(self, max_hints: int = 3) -> List[str]:
        """
        Generate hint text for the player.
        
        Prioritizes near-misses and close-by hidden content.
        """
        hints = []
        
        # Near-misses are great hints
        for nm in self.get_near_misses()[:max_hints]:
            if nm.hint_text:
                hints.append(nm.hint_text)
            elif nm.conditions_unmet:
                hints.append(f"Something about {nm.entity_name} seems significant...")
        
        # Hidden items nearby
        for hidden in self.get_hidden_items()[:max_hints - len(hints)]:
            if hidden.distance <= 1 and hidden.hint_text:
                hints.append(hidden.hint_text)
        
        return hints[:max_hints]
    
    def summary(self) -> str:
        """Human-readable summary of lookahead results"""
        lines = [
            f"Lookahead from: {self.origin_name}",
            f"Depth: {self.max_depth}",
            f"Entities seen: {self.total_entities_seen}",
            f"Links traversed: {self.total_links_traversed}",
            f"Possibilities found: {len(self.all_possibilities)}",
            f"Blocked: {self.blocked_count}",
            f"Near-misses: {self.near_miss_count}",
            "",
            "By type:",
        ]
        for ptype, possibilities in sorted(self.by_type.items(), key=lambda x: -len(x[1])):
            lines.append(f"  {ptype.name}: {len(possibilities)}")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# LOOKAHEAD ENGINE - The core pre-traversal logic
# ═══════════════════════════════════════════════════════════════════════════════

class LookaheadEngine:
    """
    Pre-traverses the entity graph to discover possibilities.
    
    This is the "oracle" that knows what could happen from any position.
    It traverses across multiple link types (layers) simultaneously,
    tracking conditions and building up the possibility space.
    """
    
    def __init__(self, graph: Any = None):
        """
        Initialize with an EntityGraph (or compatible).
        
        Graph should have:
        - _nodes: Dict[str, Entity]
        - _links: Dict[str, Link] 
        - get_outgoing_links(entity_id) -> List[Link]
        - get_incoming_links(entity_id) -> List[Link]
        """
        self.graph = graph
        
        # Link type handlers
        self._type_handlers: Dict[str, Callable] = {}
        
        # Condition evaluators
        self._condition_evaluators: Dict[str, Callable] = {}
        
        # Register default handlers
        self._register_defaults()
    
    def set_graph(self, graph: Any):
        """Set or change the graph being analyzed"""
        self.graph = graph
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main Lookahead Interface
    # ─────────────────────────────────────────────────────────────────────────
    
    def lookahead(self, 
                  from_entity: Any,
                  context: Dict[str, Any] = None,
                  max_depth: int = 3,
                  include_blocked: bool = True,
                  layers: List[str] = None) -> LookaheadResult:
        """
        Perform lookahead from a starting entity.
        
        Args:
            from_entity: Starting entity (or ID)
            context: Current game state (flags, inventory, etc.)
            max_depth: How far to look (graph distance)
            include_blocked: Whether to include blocked possibilities
            layers: Which link types to follow (None = all)
        
        Returns:
            LookaheadResult with all discovered possibilities
        """
        if self.graph is None:
            raise ValueError("No graph set. Call set_graph() first.")
        
        context = context or {}
        
        # Resolve entity
        start_id, start_entity = self._resolve_entity(from_entity)
        
        # Initialize result
        result = LookaheadResult(
            origin_id=start_id,
            origin_name=getattr(start_entity, 'name', start_id),
            max_depth=max_depth
        )
        
        # BFS with depth tracking
        visited = set()
        # Queue: (entity_id, depth, path)
        frontier = deque([(start_id, 0, [start_id])])
        
        while frontier:
            current_id, depth, path = frontier.popleft()
            
            if current_id in visited:
                continue
            visited.add(current_id)
            result.total_entities_seen += 1
            
            # Get current entity
            current = self._get_entity(current_id)
            if current is None:
                continue
            
            # Analyze this entity
            possibilities = self._analyze_entity(current, context, depth, path)
            for poss in possibilities:
                if include_blocked or not poss.is_blocked():
                    result.add(poss)
            
            # Don't go deeper than max_depth
            if depth >= max_depth:
                continue
            
            # Get outgoing links
            links = self._get_links(current_id, layers)
            
            for link in links:
                result.total_links_traversed += 1
                
                # Analyze the link itself
                link_possibilities = self._analyze_link(
                    link, current_id, context, depth, path
                )
                for poss in link_possibilities:
                    if include_blocked or not poss.is_blocked():
                        result.add(poss)
                
                # Add target to frontier (even if link is blocked)
                target_id = self._get_link_target(link, current_id)
                if target_id and target_id not in visited:
                    new_path = path + [target_id]
                    frontier.append((target_id, depth + 1, new_path))
        
        return result
    
    def lookahead_for(self,
                      from_entity: Any,
                      target_entity: Any,
                      context: Dict[str, Any] = None,
                      max_depth: int = 5) -> Optional[LookaheadResult]:
        """
        Focused lookahead: find paths/requirements to reach a specific target.
        
        Returns None if target is not reachable within max_depth.
        """
        result = self.lookahead(from_entity, context, max_depth)
        
        target_id, _ = self._resolve_entity(target_entity)
        
        if target_id not in result.by_entity:
            return None
        
        # Filter to only possibilities involving the target
        filtered = LookaheadResult(
            origin_id=result.origin_id,
            origin_name=result.origin_name,
            max_depth=max_depth
        )
        
        for poss in result.by_entity[target_id]:
            filtered.add(poss)
        
        # Also add any requirements
        chain = result.get_puzzle_chain(target_id)
        for poss in chain:
            if poss not in filtered.all_possibilities:
                filtered.add(poss)
        
        return filtered
    
    def what_unlocks(self, 
                     entity: Any,
                     context: Dict[str, Any] = None) -> List[Possibility]:
        """
        Find what the given entity could unlock/reveal/enable.
        
        Useful for: "What does this key open?" or "What does finding this reveal?"
        """
        context = context or {}
        entity_id, entity_obj = self._resolve_entity(entity)
        
        possibilities = []
        
        # Check outgoing LOGICAL links
        for link in self._get_links(entity_id, ['LOGICAL']):
            target_id = self._get_link_target(link, entity_id)
            target = self._get_entity(target_id)
            
            if target is None:
                continue
            
            kind = self._get_link_kind(link)
            
            if kind in ('UNLOCKS', 'REVEALS', 'ENABLES', 'TRIGGERS'):
                poss = Possibility(
                    possibility_type=self._kind_to_possibility_type(kind),
                    entity_id=target_id,
                    entity_name=getattr(target, 'name', target_id),
                    via_link_type='LOGICAL',
                    via_link_kind=kind,
                    hint_text=f"This could affect {getattr(target, 'name', target_id)}..."
                )
                
                # Check conditions
                if hasattr(link, 'condition') and link.condition:
                    if self._evaluate_condition(link.condition, context):
                        poss.conditions_met.append(str(link.condition))
                    else:
                        poss.conditions_unmet.append(str(link.condition))
                
                possibilities.append(poss)
        
        return possibilities
    
    def what_blocks(self,
                    entity: Any,
                    context: Dict[str, Any] = None) -> List[Possibility]:
        """
        Find what is blocking access to this entity.
        
        Useful for: "Why can't I get to X?" or "What do I need?"
        """
        context = context or {}
        entity_id, entity_obj = self._resolve_entity(entity)
        
        blockers = []
        
        # Check incoming links
        for link in self._get_incoming_links(entity_id):
            kind = self._get_link_kind(link)
            
            if kind in ('REQUIRES', 'BLOCKS'):
                source_id = self._get_link_source(link)
                source = self._get_entity(source_id)
                
                poss = Possibility(
                    possibility_type=PossibilityType.CONDITIONAL_BLOCKED,
                    entity_id=source_id,
                    entity_name=getattr(source, 'name', source_id) if source else source_id,
                    via_link_type=self._get_link_type(link),
                    via_link_kind=kind
                )
                
                # Check if requirement is met
                if kind == 'REQUIRES':
                    if self._check_requirement(source_id, context):
                        poss.conditions_met.append(f"Have {poss.entity_name}")
                    else:
                        poss.conditions_unmet.append(f"Need {poss.entity_name}")
                        poss.requirements.append(source_id)
                
                blockers.append(poss)
        
        # Check entity state
        if hasattr(entity_obj, 'state'):
            state = entity_obj.state
            state_name = state.name if hasattr(state, 'name') else str(state)
            
            if state_name in ('LOCKED', 'HIDDEN', 'CLOSED'):
                blockers.append(Possibility(
                    possibility_type=PossibilityType.CONDITIONAL_BLOCKED,
                    entity_id=entity_id,
                    entity_name=getattr(entity_obj, 'name', entity_id),
                    conditions_unmet=[f"Currently {state_name}"]
                ))
        
        return blockers
    
    # ─────────────────────────────────────────────────────────────────────────
    # Entity Analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    def _analyze_entity(self,
                        entity: Any,
                        context: Dict[str, Any],
                        depth: int,
                        path: List[str]) -> List[Possibility]:
        """Analyze a single entity for possibilities"""
        possibilities = []
        
        entity_id = self._get_entity_id(entity)
        entity_name = getattr(entity, 'name', entity_id)
        entity_type = self._get_entity_type(entity)
        entity_state = self._get_entity_state(entity)
        
        # Location possibilities
        if entity_type == 'LOCATION':
            possibilities.append(Possibility(
                possibility_type=PossibilityType.REACHABLE_LOCATION,
                entity_id=entity_id,
                entity_name=entity_name,
                distance=depth,
                path=path.copy()
            ))
        
        # Item possibilities
        elif entity_type == 'ITEM':
            if entity_state == 'HIDDEN':
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.HIDDEN_ITEM,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    distance=depth,
                    path=path.copy(),
                    hint_text=f"There might be something hidden here..."
                ))
            else:
                poss_type = PossibilityType.TAKEABLE_ITEM if self._is_takeable(entity) \
                           else PossibilityType.VISIBLE_ITEM
                possibilities.append(Possibility(
                    possibility_type=poss_type,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    distance=depth,
                    path=path.copy()
                ))
        
        # Container possibilities
        elif entity_type == 'CONTAINER':
            if entity_state in ('LOCKED', 'CLOSED'):
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.LOCKED_DOOR if entity_state == 'LOCKED' 
                                    else PossibilityType.CONDITIONAL_BLOCKED,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    distance=depth,
                    path=path.copy(),
                    conditions_unmet=[f"{entity_name} is {entity_state.lower()}"]
                ))
        
        # Door possibilities
        elif entity_type == 'DOOR':
            if entity_state == 'LOCKED':
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.LOCKED_DOOR,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    distance=depth,
                    path=path.copy(),
                    conditions_unmet=["Locked"],
                    hint_text=f"The {entity_name} is locked. You'll need a key."
                ))
        
        # Character possibilities
        elif entity_type == 'CHARACTER':
            possibilities.append(Possibility(
                possibility_type=PossibilityType.NPC_DIALOGUE,
                entity_id=entity_id,
                entity_name=entity_name,
                distance=depth,
                path=path.copy()
            ))
        
        # Concept possibilities (clues, knowledge)
        elif entity_type == 'CONCEPT':
            if entity_state == 'HIDDEN':
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.DISCOVERABLE_CLUE,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    distance=depth,
                    path=path.copy(),
                    hint_text=f"There's something important you haven't discovered yet."
                ))
        
        return possibilities
    
    def _analyze_link(self,
                      link: Any,
                      from_id: str,
                      context: Dict[str, Any],
                      depth: int,
                      path: List[str]) -> List[Possibility]:
        """Analyze a link for possibilities"""
        possibilities = []
        
        link_type = self._get_link_type(link)
        link_kind = self._get_link_kind(link)
        target_id = self._get_link_target(link, from_id)
        target = self._get_entity(target_id)
        
        if target is None:
            return possibilities
        
        target_name = getattr(target, 'name', target_id)
        
        # Check link condition
        conditions_met = []
        conditions_unmet = []
        
        if hasattr(link, 'condition') and link.condition:
            if self._evaluate_condition(link.condition, context):
                conditions_met.append(self._describe_condition(link.condition))
            else:
                conditions_unmet.append(self._describe_condition(link.condition))
        
        # LOGICAL links create special possibilities
        if link_type == 'LOGICAL':
            if link_kind == 'REQUIRES':
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.REQUIRED_ITEM,
                    entity_id=target_id,
                    entity_name=target_name,
                    distance=depth,
                    path=path + [target_id],
                    via_link_type=link_type,
                    via_link_kind=link_kind,
                    conditions_met=conditions_met,
                    conditions_unmet=conditions_unmet
                ))
            
            elif link_kind == 'UNLOCKS':
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.UNLOCKABLE,
                    entity_id=target_id,
                    entity_name=target_name,
                    distance=depth,
                    path=path + [target_id],
                    via_link_type=link_type,
                    via_link_kind=link_kind,
                    conditions_met=conditions_met,
                    conditions_unmet=conditions_unmet,
                    hint_text=f"Something here could unlock {target_name}..." if conditions_unmet else None
                ))
            
            elif link_kind == 'REVEALS':
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.REVEALABLE,
                    entity_id=target_id,
                    entity_name=target_name,
                    distance=depth,
                    path=path + [target_id],
                    via_link_type=link_type,
                    via_link_kind=link_kind,
                    conditions_met=conditions_met,
                    conditions_unmet=conditions_unmet
                ))
            
            elif link_kind == 'TRIGGERS':
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.TRIGGERABLE,
                    entity_id=target_id,
                    entity_name=target_name,
                    distance=depth,
                    path=path + [target_id],
                    via_link_type=link_type,
                    via_link_kind=link_kind,
                    conditions_met=conditions_met,
                    conditions_unmet=conditions_unmet
                ))
            
            elif link_kind == 'BLOCKS':
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.CONDITIONAL_BLOCKED,
                    entity_id=target_id,
                    entity_name=target_name,
                    distance=depth,
                    path=path + [target_id],
                    via_link_type=link_type,
                    via_link_kind=link_kind,
                    conditions_met=conditions_met,
                    conditions_unmet=conditions_unmet
                ))
        
        # RELATIONAL links with conditions = blocked paths
        elif link_type == 'RELATIONAL' and conditions_unmet:
            if link_kind in ('NORTH_OF', 'SOUTH_OF', 'EAST_OF', 'WEST_OF', 'IN'):
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.BLOCKED_PATH,
                    entity_id=target_id,
                    entity_name=target_name,
                    distance=depth,
                    path=path + [target_id],
                    via_link_type=link_type,
                    via_link_kind=link_kind,
                    conditions_met=conditions_met,
                    conditions_unmet=conditions_unmet,
                    hint_text=f"The way to {target_name} is blocked."
                ))
        
        return possibilities
    
    # ─────────────────────────────────────────────────────────────────────────
    # Graph Access Helpers
    # ─────────────────────────────────────────────────────────────────────────
    
    def _resolve_entity(self, entity: Any) -> Tuple[str, Any]:
        """Resolve entity reference to (id, object)"""
        if isinstance(entity, str):
            obj = self._get_entity(entity)
            return (entity, obj)
        
        entity_id = self._get_entity_id(entity)
        return (entity_id, entity)
    
    def _get_entity(self, entity_id: str) -> Any:
        """Get entity by ID from graph"""
        if hasattr(self.graph, '_nodes'):
            return self.graph._nodes.get(entity_id)
        if hasattr(self.graph, 'nodes'):
            return self.graph.nodes.get(entity_id)
        if hasattr(self.graph, 'get_node'):
            return self.graph.get_node(entity_id)
        return None
    
    def _get_entity_id(self, entity: Any) -> str:
        """Extract ID from entity"""
        if hasattr(entity, 'id'):
            return entity.id
        if hasattr(entity, 'entity_id'):
            return entity.entity_id
        return str(entity)
    
    def _get_entity_type(self, entity: Any) -> str:
        """Get entity type as string"""
        if hasattr(entity, 'entity_type'):
            t = entity.entity_type
            return t.name if hasattr(t, 'name') else str(t)
        if hasattr(entity, 'type'):
            return str(entity.type)
        return 'UNKNOWN'
    
    def _get_entity_state(self, entity: Any) -> str:
        """Get entity state as string"""
        if hasattr(entity, 'state'):
            s = entity.state
            return s.name if hasattr(s, 'name') else str(s)
        return 'NORMAL'
    
    def _is_takeable(self, entity: Any) -> bool:
        """Check if entity can be taken"""
        if hasattr(entity, 'tags'):
            return 'takeable' in entity.tags
        if hasattr(entity, 'properties'):
            return entity.properties.get('takeable', False)
        return False
    
    def _get_links(self, entity_id: str, layers: List[str] = None) -> List[Any]:
        """Get outgoing links from entity"""
        links = []
        
        if hasattr(self.graph, 'get_outgoing_links'):
            links = self.graph.get_outgoing_links(entity_id)
        elif hasattr(self.graph, '_links'):
            links = [l for l in self.graph._links.values() 
                    if getattr(l, 'source_id', None) == entity_id]
        
        # Filter by layers if specified
        if layers:
            links = [l for l in links if self._get_link_type(l) in layers]
        
        return links
    
    def _get_incoming_links(self, entity_id: str) -> List[Any]:
        """Get incoming links to entity"""
        if hasattr(self.graph, 'get_incoming_links'):
            return self.graph.get_incoming_links(entity_id)
        if hasattr(self.graph, '_links'):
            return [l for l in self.graph._links.values()
                   if getattr(l, 'target_id', None) == entity_id]
        return []
    
    def _get_link_type(self, link: Any) -> str:
        """Get link type as string"""
        if hasattr(link, 'link_type'):
            t = link.link_type
            return t.name if hasattr(t, 'name') else str(t)
        return 'UNKNOWN'
    
    def _get_link_kind(self, link: Any) -> str:
        """Get link kind/subtype as string"""
        if hasattr(link, 'kind'):
            k = link.kind
            if hasattr(k, 'value'):
                return k.value.upper() if isinstance(k.value, str) else k.name
            return k.name if hasattr(k, 'name') else str(k).upper()
        return 'UNKNOWN'
    
    def _get_link_target(self, link: Any, from_id: str) -> Optional[str]:
        """Get target ID of link"""
        if hasattr(link, 'target_id'):
            return link.target_id
        if hasattr(link, 'target'):
            return self._get_entity_id(link.target)
        return None
    
    def _get_link_source(self, link: Any) -> Optional[str]:
        """Get source ID of link"""
        if hasattr(link, 'source_id'):
            return link.source_id
        if hasattr(link, 'source'):
            return self._get_entity_id(link.source)
        return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Condition Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    
    def _evaluate_condition(self, condition: Any, context: Dict[str, Any]) -> bool:
        """Evaluate a link condition"""
        if condition is None:
            return True
        
        # LinkCondition with evaluate method
        if hasattr(condition, 'evaluate'):
            try:
                return condition.evaluate(context)
            except Exception:
                return False
        
        # Callable
        if callable(condition):
            try:
                return condition(context)
            except Exception:
                return False
        
        return True
    
    def _describe_condition(self, condition: Any) -> str:
        """Get human-readable description of condition"""
        if hasattr(condition, 'description') and condition.description:
            return condition.description
        return str(condition)
    
    def _check_requirement(self, required_id: str, context: Dict[str, Any]) -> bool:
        """Check if a required entity is accessible"""
        # Check inventory
        inventory = context.get('inventory', [])
        if required_id in inventory:
            return True
        
        # Check flags
        flags = context.get('flags', {})
        if flags.get(f'has_{required_id}', False):
            return True
        
        return False
    
    def _kind_to_possibility_type(self, kind: str) -> PossibilityType:
        """Map link kind to possibility type"""
        mapping = {
            'UNLOCKS': PossibilityType.UNLOCKABLE,
            'REVEALS': PossibilityType.REVEALABLE,
            'ENABLES': PossibilityType.UNLOCKABLE,
            'TRIGGERS': PossibilityType.TRIGGERABLE,
            'REQUIRES': PossibilityType.REQUIRED_ITEM,
            'BLOCKS': PossibilityType.CONDITIONAL_BLOCKED,
        }
        return mapping.get(kind.upper(), PossibilityType.CONDITIONAL_BLOCKED)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Extension Points
    # ─────────────────────────────────────────────────────────────────────────
    
    def register_type_handler(self, entity_type: str, handler: Callable):
        """Register custom handler for entity type analysis"""
        self._type_handlers[entity_type] = handler
    
    def register_condition_evaluator(self, name: str, evaluator: Callable):
        """Register custom condition evaluator"""
        self._condition_evaluators[name] = evaluator
    
    def _register_defaults(self):
        """Register default handlers"""
        pass  # Extensibility point


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH TRAVERSAL WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class LookaheadWrapper:
    """
    Wrapper that provides lookahead capabilities during traversal.
    
    Attach this to a TraversalWrapper or GraphTraverser for on-the-fly
    lookahead at each step.
    """
    
    def __init__(self, engine: LookaheadEngine, default_depth: int = 2):
        self.engine = engine
        self.default_depth = default_depth
        self._cache: Dict[str, LookaheadResult] = {}
    
    def on_step(self, wrapper, node):
        """
        Callback for use with TraversalWrapper.on_step()
        
        Performs lookahead at each step and stores in wrapper context.
        """
        entity_id = self._get_id(node)
        
        # Check cache
        if entity_id in self._cache:
            result = self._cache[entity_id]
        else:
            result = self.engine.lookahead(node, 
                                          context=wrapper.context.__dict__ if hasattr(wrapper, 'context') else {},
                                          max_depth=self.default_depth)
            self._cache[entity_id] = result
        
        # Store in wrapper context
        if hasattr(wrapper, 'context'):
            wrapper.context.buffer['lookahead'] = result
            wrapper.context.buffer['hints'] = result.get_hints()
            wrapper.context.buffer['near_misses'] = result.get_near_misses()
    
    def clear_cache(self):
        """Clear the lookahead cache"""
        self._cache.clear()
    
    def _get_id(self, node: Any) -> str:
        if hasattr(node, 'id'):
            return node.id
        return str(node)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def lookahead_from(graph: Any, 
                   entity: Any,
                   context: Dict[str, Any] = None,
                   max_depth: int = 3) -> LookaheadResult:
    """
    Convenience function for one-shot lookahead.
    
    Usage:
        result = lookahead_from(my_graph, player_location, game_state)
        hints = result.get_hints()
    """
    engine = LookaheadEngine(graph)
    return engine.lookahead(entity, context, max_depth)


def find_path_requirements(graph: Any,
                          from_entity: Any,
                          to_entity: Any,
                          context: Dict[str, Any] = None) -> List[Possibility]:
    """
    Find what's needed to get from one entity to another.
    
    Returns list of requirements (items, conditions) in the path.
    """
    engine = LookaheadEngine(graph)
    result = engine.lookahead_for(from_entity, to_entity, context)
    
    if result is None:
        return []
    
    return [p for p in result.all_possibilities 
            if p.possibility_type in (PossibilityType.REQUIRED_ITEM,
                                     PossibilityType.CONDITIONAL_BLOCKED,
                                     PossibilityType.LOCKED_DOOR)]


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # This demo requires the entity system
    # For standalone testing, we'll create mock objects
    
    print("=" * 70)
    print("Lookahead System - Demonstration")
    print("=" * 70)
    
    # Mock a simple graph
    class MockEntity:
        def __init__(self, id, name, entity_type, state='NORMAL', tags=None):
            self.id = id
            self.name = name
            self.entity_type = type('MockType', (), {'name': entity_type})()
            self.state = type('MockState', (), {'name': state})()
            self.tags = tags or set()
    
    class MockLink:
        def __init__(self, source_id, target_id, link_type, kind, condition=None):
            self.id = f"{source_id}_{target_id}"
            self.source_id = source_id
            self.target_id = target_id
            self.link_type = type('MockLinkType', (), {'name': link_type})()
            self.kind = type('MockKind', (), {'name': kind, 'value': kind.lower()})()
            self.condition = condition
    
    class MockGraph:
        def __init__(self):
            self._nodes = {}
            self._links = {}
        
        def add(self, entity):
            self._nodes[entity.id] = entity
        
        def link(self, source_id, target_id, link_type, kind, condition=None):
            link = MockLink(source_id, target_id, link_type, kind, condition)
            self._links[link.id] = link
        
        def get_outgoing_links(self, entity_id):
            return [l for l in self._links.values() if l.source_id == entity_id]
        
        def get_incoming_links(self, entity_id):
            return [l for l in self._links.values() if l.target_id == entity_id]
    
    # Build demo world
    graph = MockGraph()
    
    # Entities
    graph.add(MockEntity('study', 'Study', 'LOCATION'))
    graph.add(MockEntity('hallway', 'Hallway', 'LOCATION'))
    graph.add(MockEntity('library', 'Library', 'LOCATION'))
    graph.add(MockEntity('locked_door', 'Locked Door', 'DOOR', state='LOCKED'))
    graph.add(MockEntity('secret_room', 'Secret Room', 'LOCATION'))
    graph.add(MockEntity('desk', 'Oak Desk', 'CONTAINER'))
    graph.add(MockEntity('key', 'Brass Key', 'ITEM', state='HIDDEN', tags={'takeable'}))
    graph.add(MockEntity('book', 'Old Book', 'ITEM', tags={'takeable'}))
    graph.add(MockEntity('clue', 'Hidden Clue', 'CONCEPT', state='HIDDEN'))
    
    # Relational links
    graph.link('study', 'hallway', 'RELATIONAL', 'NORTH_OF')
    graph.link('hallway', 'study', 'RELATIONAL', 'SOUTH_OF')
    graph.link('hallway', 'library', 'RELATIONAL', 'EAST_OF')
    graph.link('library', 'hallway', 'RELATIONAL', 'WEST_OF')
    graph.link('hallway', 'locked_door', 'RELATIONAL', 'IN')
    graph.link('locked_door', 'secret_room', 'RELATIONAL', 'NORTH_OF')
    
    # Containment
    graph.link('study', 'desk', 'RELATIONAL', 'CONTAINS')
    graph.link('desk', 'key', 'RELATIONAL', 'CONTAINS')
    graph.link('library', 'book', 'RELATIONAL', 'CONTAINS')
    
    # Logical links
    graph.link('key', 'locked_door', 'LOGICAL', 'UNLOCKS')
    graph.link('desk', 'key', 'LOGICAL', 'REVEALS')
    graph.link('book', 'clue', 'LOGICAL', 'REVEALS')
    
    # Create engine
    engine = LookaheadEngine(graph)
    
    print("\n--- Lookahead from Study (depth=3) ---")
    result = engine.lookahead('study', max_depth=3)
    print(result.summary())
    
    print("\n--- Reachable Locations ---")
    for p in result.get_reachable():
        print(f"  {p.entity_name} (distance={p.distance})")
    
    print("\n--- Hidden Items ---")
    for p in result.get_hidden_items():
        print(f"  {p.entity_name} ({p.possibility_type.name})")
    
    print("\n--- Blocked Paths ---")
    for p in result.get_blocked():
        print(f"  {p.entity_name}: {p.conditions_unmet}")
    
    print("\n--- What does the key unlock? ---")
    unlocks = engine.what_unlocks('key')
    for p in unlocks:
        print(f"  {p.entity_name} ({p.via_link_kind})")
    
    print("\n--- What blocks the secret room? ---")
    blockers = engine.what_blocks('secret_room')
    for p in blockers:
        print(f"  {p.entity_name}: {p.conditions_unmet}")
    
    print("\n--- Hints for player ---")
    hints = result.get_hints()
    for hint in hints:
        print(f"  💡 {hint}")
    
    print("\n--- Puzzle chain to secret room ---")
    chain = result.get_puzzle_chain('secret_room')
    for p in chain:
        print(f"  {p.entity_name} ({p.possibility_type.name})")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
