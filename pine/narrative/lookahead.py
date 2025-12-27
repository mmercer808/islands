"""
Pine Lookahead System
=====================

Possibility analysis for narrative exploration.

The lookahead engine analyzes what a player CAN do from
their current position - providing hints, suggestions,
and narrative branching points.

SOURCE: Mechanics/everything/lookahead.py (20KB)
Copy the full implementation from that file.

Quick Reference:
----------------
    from pine.narrative import lookahead_from, LookaheadResult

    # Analyze possibilities from current position
    result = lookahead_from(world, current_node)

    # Get hints
    for hint in result.get_hints():
        print(f"Hint: {hint}")

    # Get available actions
    for action in result.actions:
        print(f"You can: {action.description}")
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto

from pine.core.primitives import PossibilityType


# =============================================================================
#                              DATA CLASSES
# =============================================================================

@dataclass
class Possibility:
    """
    A single possibility the player can pursue.

    Represents an action, discovery, or transition
    that is available from the current state.
    """
    possibility_type: PossibilityType
    description: str
    target_id: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)
    hint_text: Optional[str] = None
    priority: int = 0
    hidden: bool = False

    def is_available(self, context: Dict[str, Any]) -> bool:
        """Check if this possibility is currently available."""
        for req in self.requirements:
            if req not in context:
                return False
        return True


@dataclass
class LookaheadResult:
    """
    Result of lookahead analysis.

    Contains all possibilities discovered plus
    summary information and hints.
    """
    node_id: str
    depth: int
    possibilities: List[Possibility] = field(default_factory=list)
    explored_nodes: Set[str] = field(default_factory=set)
    warnings: List[str] = field(default_factory=list)

    @property
    def actions(self) -> List[Possibility]:
        """Get action possibilities."""
        return [p for p in self.possibilities
                if p.possibility_type == PossibilityType.ACTION]

    @property
    def discoveries(self) -> List[Possibility]:
        """Get discovery possibilities."""
        return [p for p in self.possibilities
                if p.possibility_type == PossibilityType.DISCOVERY]

    @property
    def transitions(self) -> List[Possibility]:
        """Get transition possibilities."""
        return [p for p in self.possibilities
                if p.possibility_type == PossibilityType.TRANSITION]

    def get_hints(self, max_hints: int = 3) -> List[str]:
        """Get hint texts for available possibilities."""
        hints = []
        for p in sorted(self.possibilities, key=lambda x: -x.priority):
            if p.hint_text and not p.hidden:
                hints.append(p.hint_text)
                if len(hints) >= max_hints:
                    break
        return hints

    def summary(self) -> str:
        """Get summary of lookahead results."""
        lines = [
            f"=== Lookahead from {self.node_id} ===",
            f"Depth: {self.depth}",
            f"Possibilities: {len(self.possibilities)}",
            f"  Actions: {len(self.actions)}",
            f"  Discoveries: {len(self.discoveries)}",
            f"  Transitions: {len(self.transitions)}",
            f"Explored: {len(self.explored_nodes)} nodes",
        ]
        return "\n".join(lines)


# =============================================================================
#                              LOOKAHEAD ENGINE
# =============================================================================

class LookaheadEngine:
    """
    Engine for analyzing possibilities from a given state.

    Performs breadth-first exploration of the world graph
    to find available actions, discoverable items, and
    reachable locations.

    TODO: Copy full implementation from Mechanics/everything/lookahead.py
    """

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    def analyze(
        self,
        world: 'WhiteRoom',
        start_node: 'WorldNode',
        context: Dict[str, Any] = None
    ) -> LookaheadResult:
        """
        Analyze possibilities from a starting node.

        Args:
            world: The world to explore
            start_node: Starting position
            context: Current player state/inventory

        Returns:
            LookaheadResult with all found possibilities
        """
        from .world import WhiteRoom, WorldNode, EdgeType

        context = context or {}
        result = LookaheadResult(
            node_id=start_node.id,
            depth=self.max_depth
        )

        # BFS exploration
        to_explore = [(start_node, 0)]
        result.explored_nodes.add(start_node.id)

        while to_explore:
            current, depth = to_explore.pop(0)

            if depth > self.max_depth:
                continue

            # Analyze current node
            self._analyze_node(current, depth, result, context)

            # Get neighbors
            if depth < self.max_depth:
                for neighbor in world.get_neighbors(current.id):
                    if neighbor.id not in result.explored_nodes:
                        result.explored_nodes.add(neighbor.id)
                        to_explore.append((neighbor, depth + 1))

                        # Add transition possibility
                        result.possibilities.append(Possibility(
                            possibility_type=PossibilityType.TRANSITION,
                            description=f"Go to {neighbor.name}",
                            target_id=neighbor.id,
                            hint_text=f"You could head to {neighbor.name}..."
                        ))

            # Check contents
            for content in world.get_contents(current.id):
                if content.id not in result.explored_nodes:
                    result.explored_nodes.add(content.id)
                    self._analyze_item(content, result, context)

        return result

    def _analyze_node(
        self,
        node: 'WorldNode',
        depth: int,
        result: LookaheadResult,
        context: Dict[str, Any]
    ) -> None:
        """Analyze a single node for possibilities."""
        # Check for discoveries
        if 'hidden' in node.tags and 'search' not in context.get('actions_taken', []):
            result.possibilities.append(Possibility(
                possibility_type=PossibilityType.DISCOVERY,
                description=f"Search {node.name}",
                target_id=node.id,
                hint_text=f"There might be something hidden in {node.name}..."
            ))

        # Check for interactions
        if node.properties.get('interactive'):
            result.possibilities.append(Possibility(
                possibility_type=PossibilityType.ACTION,
                description=f"Interact with {node.name}",
                target_id=node.id
            ))

    def _analyze_item(
        self,
        item: 'WorldNode',
        result: LookaheadResult,
        context: Dict[str, Any]
    ) -> None:
        """Analyze an item for possibilities."""
        from .world import NodeType

        if item.node_type == NodeType.ITEM:
            # Can take item if not already in inventory
            if item.id not in context.get('inventory', []):
                result.possibilities.append(Possibility(
                    possibility_type=PossibilityType.ACTION,
                    description=f"Take {item.name}",
                    target_id=item.id,
                    hint_text=f"You notice {item.name}..."
                ))


# =============================================================================
#                              FACTORY FUNCTION
# =============================================================================

def lookahead_from(
    world: 'WhiteRoom',
    node: 'WorldNode',
    depth: int = 3,
    context: Dict[str, Any] = None
) -> LookaheadResult:
    """
    Convenience function to run lookahead analysis.

    Args:
        world: The world to explore
        node: Starting node
        depth: How far to look ahead
        context: Current player state

    Returns:
        LookaheadResult with possibilities
    """
    engine = LookaheadEngine(max_depth=depth)
    return engine.analyze(world, node, context)
