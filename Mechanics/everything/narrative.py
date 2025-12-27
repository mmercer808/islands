"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                    N A R R A T I V E   L A Y E R                              ║
║                                                                               ║
║              Story, Prose, and Text Management for the Island                 ║
║                                                                               ║
║  Stories are graphs too. Beats connect to beats.                              ║
║  Prose lives at nodes, waiting to be discovered.                              ║
║  The walker reads the story as it walks the world.                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Callable, Tuple, Union
from enum import Enum, auto
import re
import time

from graph_core import (
    IslandGraph, GraphNode, Edge, NodeType, EdgeType,
    create_prose
)
from graph_walker import GraphWalker, WalkerContext, WalkerCallback


# ═══════════════════════════════════════════════════════════════════════════════
# PROSE TYPES - Different kinds of text in the game
# ═══════════════════════════════════════════════════════════════════════════════

class ProseType(Enum):
    """Categories of narrative text"""
    DESCRIPTION = auto()    # Location/item descriptions
    DIALOGUE = auto()       # Character speech
    NARRATION = auto()      # Story beats, events
    LORE = auto()           # Background/history
    HINT = auto()           # Player guidance
    FLAVOR = auto()         # Atmospheric text
    MEMORY = auto()         # Things the player remembers
    REACTION = auto()       # Response to player actions


class StoryState(Enum):
    """States a story beat can be in"""
    LOCKED = auto()         # Cannot be triggered yet
    AVAILABLE = auto()      # Can be triggered
    ACTIVE = auto()         # Currently happening
    COMPLETED = auto()      # Has happened
    FAILED = auto()         # Missed opportunity


# ═══════════════════════════════════════════════════════════════════════════════
# PROSE FRAGMENT - A piece of text with conditions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProseFragment:
    """
    A piece of prose that can be conditionally displayed.
    
    Fragments can:
    - Have conditions (show only if player has key)
    - Have variations (different text based on state)
    - Chain together (one leads to another)
    - Be consumed (show only once)
    """
    id: str
    prose_type: ProseType
    text: str
    
    # Conditions
    requires_flags: List[str] = field(default_factory=list)    # All must be true
    requires_items: List[str] = field(default_factory=list)    # Must have these
    requires_visited: List[str] = field(default_factory=list)  # Must have been here
    excludes_flags: List[str] = field(default_factory=list)    # Must NOT have these
    
    # Behavior
    consume_on_read: bool = False     # Only show once
    consumed: bool = False
    priority: int = 0                 # Higher = shown first if multiple match
    
    # Variations
    variations: Dict[str, str] = field(default_factory=dict)  # flag → alt_text
    
    # Effects when shown
    sets_flags: List[str] = field(default_factory=list)
    grants_items: List[str] = field(default_factory=list)
    
    # Chain
    next_fragment: Optional[str] = None
    
    def check_conditions(self, context: WalkerContext) -> bool:
        """Check if this fragment should be shown"""
        if self.consumed:
            return False
        
        # Check required flags
        for flag in self.requires_flags:
            if not context.get_flag(flag):
                return False
        
        # Check excluded flags
        for flag in self.excludes_flags:
            if context.get_flag(flag):
                return False
        
        # Check required items
        for item in self.requires_items:
            if item not in context.inventory:
                return False
        
        # Check required visits
        for loc in self.requires_visited:
            if loc not in context.visited:
                return False
        
        return True
    
    def get_text(self, context: WalkerContext) -> str:
        """Get the appropriate text, considering variations"""
        # Check for variations
        for flag, alt_text in self.variations.items():
            if context.get_flag(flag):
                return alt_text
        
        return self.text
    
    def apply_effects(self, context: WalkerContext):
        """Apply any effects from reading this fragment"""
        for flag in self.sets_flags:
            context.set_flag(flag)
        
        for item in self.grants_items:
            if item not in context.inventory:
                context.inventory.append(item)
        
        if self.consume_on_read:
            self.consumed = True


# ═══════════════════════════════════════════════════════════════════════════════
# STORY BEAT - A moment in the narrative
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StoryBeat:
    """
    A story beat is a significant narrative moment.
    
    Beats form their own graph structure (the plot).
    They can trigger based on location, items, or other beats completing.
    """
    id: str
    name: str
    description: str
    
    state: StoryState = StoryState.LOCKED
    
    # Trigger conditions
    trigger_location: Optional[str] = None        # Location ID
    trigger_flags: List[str] = field(default_factory=list)
    trigger_items: List[str] = field(default_factory=list)
    trigger_beats: List[str] = field(default_factory=list)  # Previous beats
    
    # Content
    prose_fragments: List[str] = field(default_factory=list)  # Fragment IDs
    
    # Effects
    sets_flags: List[str] = field(default_factory=list)
    unlocks_locations: List[str] = field(default_factory=list)
    grants_items: List[str] = field(default_factory=list)
    
    # Narrative flow
    next_beats: List[str] = field(default_factory=list)  # What this unlocks
    
    # Timing
    triggered_at: Optional[float] = None
    completed_at: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════════
# PROSE REGISTRY - Central storage for all prose
# ═══════════════════════════════════════════════════════════════════════════════

class ProseRegistry:
    """
    Central registry for all prose fragments in the game.
    
    Handles:
    - Storage and retrieval of fragments
    - Conditional filtering
    - Text assembly from multiple fragments
    """
    
    def __init__(self):
        self._fragments: Dict[str, ProseFragment] = {}
        self._by_type: Dict[ProseType, List[str]] = {t: [] for t in ProseType}
        self._by_location: Dict[str, List[str]] = {}  # location_id → fragment_ids
        self._by_tag: Dict[str, List[str]] = {}
    
    def add(self, fragment: ProseFragment, 
            locations: List[str] = None,
            tags: List[str] = None):
        """Add a fragment to the registry"""
        self._fragments[fragment.id] = fragment
        self._by_type[fragment.prose_type].append(fragment.id)
        
        for loc in (locations or []):
            if loc not in self._by_location:
                self._by_location[loc] = []
            self._by_location[loc].append(fragment.id)
        
        for tag in (tags or []):
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(fragment.id)
    
    def get(self, fragment_id: str) -> Optional[ProseFragment]:
        """Get a specific fragment"""
        return self._fragments.get(fragment_id)
    
    def get_for_location(self, 
                         location_id: str,
                         context: WalkerContext,
                         prose_type: ProseType = None) -> List[ProseFragment]:
        """Get all applicable fragments for a location"""
        fragment_ids = self._by_location.get(location_id, [])
        
        results = []
        for fid in fragment_ids:
            fragment = self._fragments.get(fid)
            if fragment and fragment.check_conditions(context):
                if prose_type is None or fragment.prose_type == prose_type:
                    results.append(fragment)
        
        # Sort by priority
        results.sort(key=lambda f: -f.priority)
        return results
    
    def get_by_type(self, 
                    prose_type: ProseType,
                    context: WalkerContext) -> List[ProseFragment]:
        """Get all fragments of a type that pass conditions"""
        fragment_ids = self._by_type.get(prose_type, [])
        
        results = []
        for fid in fragment_ids:
            fragment = self._fragments.get(fid)
            if fragment and fragment.check_conditions(context):
                results.append(fragment)
        
        results.sort(key=lambda f: -f.priority)
        return results
    
    def search(self, 
               query: str,
               context: WalkerContext = None) -> List[ProseFragment]:
        """Search fragments by text content"""
        query_lower = query.lower()
        results = []
        
        for fragment in self._fragments.values():
            if query_lower in fragment.text.lower():
                if context is None or fragment.check_conditions(context):
                    results.append(fragment)
        
        return results
    
    def assemble_text(self,
                      location_id: str,
                      context: WalkerContext,
                      include_types: List[ProseType] = None) -> str:
        """
        Assemble all applicable prose for a location into a single text.
        
        This is the main "describe location" function.
        """
        if include_types is None:
            include_types = [ProseType.DESCRIPTION, ProseType.NARRATION, ProseType.FLAVOR]
        
        fragments = self.get_for_location(location_id, context)
        
        # Filter by type
        fragments = [f for f in fragments if f.prose_type in include_types]
        
        # Build text
        texts = []
        for fragment in fragments:
            text = fragment.get_text(context)
            texts.append(text)
            fragment.apply_effects(context)
        
        return "\n\n".join(texts)


# ═══════════════════════════════════════════════════════════════════════════════
# STORY MANAGER - Tracks narrative progression
# ═══════════════════════════════════════════════════════════════════════════════

class StoryManager:
    """
    Manages the story graph and beat progression.
    
    The story is a separate graph that runs parallel to the world graph.
    It tracks what narrative events have occurred and what's available.
    """
    
    def __init__(self, prose_registry: ProseRegistry = None):
        self._beats: Dict[str, StoryBeat] = {}
        self._active_beats: Set[str] = set()
        self._completed_beats: Set[str] = set()
        self.prose = prose_registry or ProseRegistry()
        
        # Event handlers
        self._on_beat_triggered: List[Callable] = []
        self._on_beat_completed: List[Callable] = []
    
    def add_beat(self, beat: StoryBeat):
        """Add a story beat"""
        self._beats[beat.id] = beat
    
    def get_beat(self, beat_id: str) -> Optional[StoryBeat]:
        """Get a specific beat"""
        return self._beats.get(beat_id)
    
    def check_triggers(self, 
                       walker: GraphWalker,
                       location_id: str = None) -> List[StoryBeat]:
        """
        Check for any story beats that should trigger.
        
        Call this whenever the game state changes.
        Returns list of newly triggered beats.
        """
        context = walker.context
        location = location_id or walker.position
        
        triggered = []
        
        for beat in self._beats.values():
            if beat.state != StoryState.LOCKED:
                continue
            
            # Check trigger conditions
            if self._should_trigger(beat, context, location):
                self._trigger_beat(beat, context)
                triggered.append(beat)
        
        return triggered
    
    def _should_trigger(self, 
                        beat: StoryBeat,
                        context: WalkerContext,
                        current_location: str) -> bool:
        """Check if a beat's trigger conditions are met"""
        
        # Location check
        if beat.trigger_location and beat.trigger_location != current_location:
            return False
        
        # Flag checks
        for flag in beat.trigger_flags:
            if not context.get_flag(flag):
                return False
        
        # Item checks
        for item in beat.trigger_items:
            if item not in context.inventory:
                return False
        
        # Previous beats check
        for prev_id in beat.trigger_beats:
            if prev_id not in self._completed_beats:
                return False
        
        return True
    
    def _trigger_beat(self, beat: StoryBeat, context: WalkerContext):
        """Trigger a story beat"""
        beat.state = StoryState.ACTIVE
        beat.triggered_at = time.time()
        self._active_beats.add(beat.id)
        
        # Apply effects
        for flag in beat.sets_flags:
            context.set_flag(flag)
        
        for item in beat.grants_items:
            if item not in context.inventory:
                context.inventory.append(item)
        
        context.log_event('story_beat_triggered', {
            'beat_id': beat.id,
            'beat_name': beat.name
        })
        
        # Unlock next beats
        for next_id in beat.next_beats:
            next_beat = self._beats.get(next_id)
            if next_beat and next_beat.state == StoryState.LOCKED:
                next_beat.state = StoryState.AVAILABLE
        
        # Fire handlers
        for handler in self._on_beat_triggered:
            handler(beat, context)
    
    def complete_beat(self, beat_id: str, context: WalkerContext):
        """Mark a beat as completed"""
        beat = self._beats.get(beat_id)
        if beat and beat.state == StoryState.ACTIVE:
            beat.state = StoryState.COMPLETED
            beat.completed_at = time.time()
            self._active_beats.discard(beat_id)
            self._completed_beats.add(beat_id)
            
            context.log_event('story_beat_completed', {
                'beat_id': beat.id,
                'beat_name': beat.name
            })
            
            for handler in self._on_beat_completed:
                handler(beat, context)
    
    def get_active_beats(self) -> List[StoryBeat]:
        """Get all currently active story beats"""
        return [self._beats[bid] for bid in self._active_beats]
    
    def get_prose_for_beat(self, beat_id: str, context: WalkerContext) -> str:
        """Get the prose text for a story beat"""
        beat = self._beats.get(beat_id)
        if not beat:
            return ""
        
        texts = []
        for frag_id in beat.prose_fragments:
            fragment = self.prose.get(frag_id)
            if fragment and fragment.check_conditions(context):
                texts.append(fragment.get_text(context))
                fragment.apply_effects(context)
        
        return "\n\n".join(texts)
    
    def on_beat_triggered(self, handler: Callable):
        """Register handler for when beats trigger"""
        self._on_beat_triggered.append(handler)
    
    def on_beat_completed(self, handler: Callable):
        """Register handler for when beats complete"""
        self._on_beat_completed.append(handler)
    
    def get_state(self) -> Dict:
        """Get serializable state"""
        return {
            'active_beats': list(self._active_beats),
            'completed_beats': list(self._completed_beats),
            'beat_states': {
                bid: beat.state.name 
                for bid, beat in self._beats.items()
            }
        }
    
    def restore_state(self, state: Dict):
        """Restore from saved state"""
        self._active_beats = set(state.get('active_beats', []))
        self._completed_beats = set(state.get('completed_beats', []))
        
        for bid, state_name in state.get('beat_states', {}).items():
            if bid in self._beats:
                self._beats[bid].state = StoryState[state_name]


# ═══════════════════════════════════════════════════════════════════════════════
# NARRATIVE WALKER - Walker with built-in story support
# ═══════════════════════════════════════════════════════════════════════════════

class NarrativeWalker(GraphWalker):
    """
    A GraphWalker extended with narrative capabilities.
    
    Automatically handles:
    - Prose display on location entry
    - Story beat checking
    - Dialogue management
    """
    
    def __init__(self, 
                 graph: IslandGraph,
                 story_manager: StoryManager,
                 start_node: Union[str, GraphNode] = None):
        
        super().__init__(graph, start_node)
        self.story = story_manager
        
        # Auto-register narrative callbacks
        self.on_enter("narrative_check_story", self._check_story_callback)
    
    def _check_story_callback(self, walker, node, context):
        """Callback to check story triggers on enter"""
        self.story.check_triggers(self, node.id)
    
    def describe(self, 
                 include_items: bool = True,
                 include_characters: bool = True,
                 include_exits: bool = True) -> str:
        """
        Get full description of current location.
        
        Assembles prose, lists items/characters, shows exits.
        """
        if not self.current:
            return "You are nowhere."
        
        parts = []
        
        # Main prose
        main_text = self.story.prose.assemble_text(
            self.current.id,
            self.context,
            [ProseType.DESCRIPTION, ProseType.NARRATION]
        )
        if main_text:
            parts.append(main_text)
        else:
            # Fallback to node description
            parts.append(self.current.get_data('description', self.current.name))
        
        # Active story beats
        for beat in self.story.get_active_beats():
            if beat.trigger_location == self.current.id:
                beat_text = self.story.get_prose_for_beat(beat.id, self.context)
                if beat_text:
                    parts.append(beat_text)
        
        # Items
        if include_items:
            items = self.find_here(NodeType.ITEM)
            if items:
                item_names = [f"  • {i.name}" for i in items]
                parts.append("You see:\n" + "\n".join(item_names))
        
        # Characters
        if include_characters:
            chars = self.find_here(NodeType.CHARACTER)
            if chars:
                char_names = [f"  • {c.name}" for c in chars]
                parts.append("Present:\n" + "\n".join(char_names))
        
        # Exits
        if include_exits:
            exits = self.can_go(edge_type=EdgeType.LEADS_TO)
            if exits:
                exit_names = [f"  → {e.target.name}" for e in exits]
                parts.append("Exits:\n" + "\n".join(exit_names))
        
        return "\n\n".join(parts)
    
    def look_at(self, target_name: str) -> str:
        """
        Look at something specific.
        
        Searches items, characters, and environment features.
        """
        target_lower = target_name.lower()
        
        # Search nearby nodes
        for node in self.nearby(depth=1):
            if target_lower in node.name.lower():
                # Get prose for this thing
                text = self.story.prose.assemble_text(
                    node.id,
                    self.context,
                    [ProseType.DESCRIPTION]
                )
                if text:
                    return text
                return node.get_data('description', f"You see {node.name}.")
        
        return f"You don't see '{target_name}' here."
    
    def talk_to(self, character_name: str) -> str:
        """
        Initiate dialogue with a character.
        
        Returns dialogue text or appropriate response.
        """
        chars = self.find_here(NodeType.CHARACTER)
        
        for char in chars:
            if character_name.lower() in char.name.lower():
                # Get dialogue prose
                dialogue = self.story.prose.get_for_location(
                    char.id,
                    self.context,
                    ProseType.DIALOGUE
                )
                
                if dialogue:
                    texts = [d.get_text(self.context) for d in dialogue]
                    for d in dialogue:
                        d.apply_effects(self.context)
                    return f'{char.name} says:\n"' + '"\n\n"'.join(texts) + '"'
                
                return f"{char.name} has nothing to say."
        
        return f"You don't see '{character_name}' here."


# ═══════════════════════════════════════════════════════════════════════════════
# PROSE BUILDERS - Convenience functions
# ═══════════════════════════════════════════════════════════════════════════════

def make_description(id: str, text: str, **kwargs) -> ProseFragment:
    """Quick builder for description prose"""
    return ProseFragment(
        id=id,
        prose_type=ProseType.DESCRIPTION,
        text=text,
        **kwargs
    )

def make_dialogue(id: str, text: str, character: str = None, **kwargs) -> ProseFragment:
    """Quick builder for dialogue"""
    return ProseFragment(
        id=id,
        prose_type=ProseType.DIALOGUE,
        text=text,
        **kwargs
    )

def make_narration(id: str, text: str, **kwargs) -> ProseFragment:
    """Quick builder for narration"""
    return ProseFragment(
        id=id,
        prose_type=ProseType.NARRATION,
        text=text,
        **kwargs
    )

def make_beat(id: str, name: str, 
              trigger_location: str = None,
              trigger_flags: List[str] = None,
              prose_ids: List[str] = None,
              sets_flags: List[str] = None,
              next_beats: List[str] = None) -> StoryBeat:
    """Quick builder for story beats"""
    return StoryBeat(
        id=id,
        name=name,
        description=name,
        trigger_location=trigger_location,
        trigger_flags=trigger_flags or [],
        prose_fragments=prose_ids or [],
        sets_flags=sets_flags or [],
        next_beats=next_beats or []
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from graph_core import create_location, create_item, create_character
    
    print("=" * 70)
    print("Narrative Layer - Demonstration")
    print("=" * 70)
    
    # Build world
    island = IslandGraph(name="Story Island")
    
    beach = create_location(island, "Sandy Beach")
    temple = create_location(island, "Ancient Temple")
    island.connect(beach.id, temple.id, EdgeType.LEADS_TO, bidirectional=True)
    
    key = create_item(island, "Golden Key")
    island.connect(temple.id, key.id, EdgeType.CONTAINS)
    
    hermit = create_character(island, "Old Hermit")
    island.connect(hermit.id, beach.id, EdgeType.LOCATED_AT)
    
    # Build prose
    prose = ProseRegistry()
    
    prose.add(
        make_description("beach_desc", 
                        "Warm sand stretches beneath your feet. Waves lap gently at the shore."),
        locations=[beach.id]
    )
    
    prose.add(
        make_description("beach_first_visit",
                        "This beach feels ancient, untouched by time.",
                        consume_on_read=True,
                        priority=10),
        locations=[beach.id]
    )
    
    prose.add(
        make_description("temple_desc",
                        "Stone columns rise toward the sky. The air is thick with mystery."),
        locations=[temple.id]
    )
    
    prose.add(
        make_description("temple_with_key",
                        "The key in your pocket grows warm...",
                        requires_items=[key.id],
                        priority=5),
        locations=[temple.id]
    )
    
    prose.add(
        make_dialogue("hermit_greeting",
                     "Ah, a visitor! It has been so long..."),
        locations=[hermit.id]
    )
    
    # Build story
    story = StoryManager(prose)
    
    story.add_beat(make_beat(
        "discovery",
        "The Discovery",
        trigger_location=temple.id,
        sets_flags=["found_temple"],
        next_beats=["revelation"]
    ))
    
    story.add_beat(make_beat(
        "revelation", 
        "The Revelation",
        trigger_flags=["found_temple", "talked_to_hermit"],
        sets_flags=["knows_secret"]
    ))
    
    # Create narrative walker
    walker = NarrativeWalker(island, story, start_node=beach)
    
    print("\n--- At the Beach ---")
    print(walker.describe())
    
    print("\n--- Moving to Temple ---")
    walker.go("temple")
    print(walker.describe())
    
    print("\n--- Story State ---")
    print(f"Active beats: {[b.name for b in story.get_active_beats()]}")
    print(f"Completed: {story._completed_beats}")
    print(f"Flags: {walker.context.flags}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
