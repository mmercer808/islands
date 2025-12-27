"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                P L O T   E Q U I L I B R I U M   E N G I N E                  ║
║                                                                               ║
║        Players Derail. Stories Recover. The Tree Has Many Branches.           ║
║                                                                               ║
║  Core insight: A plot is not a line but a tree. When players stray,           ║
║  the system applies gentle corrective forces—hints, NPCs, events—             ║
║  to guide them back toward meaningful narrative beats.                        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from enum import Enum, auto
from collections import defaultdict
import copy
import random
import uuid
import time


# ═══════════════════════════════════════════════════════════════════════════════
# BEAT TYPES & STORY STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

class BeatType(Enum):
    """Story beat types following classic dramatic structure."""
    OPENING = 10        # Establish the normal world
    INCITING = 20       # The event that starts everything
    RISING = 30         # Complications, escalation
    MIDPOINT = 40       # Major revelation or shift
    DARK_MOMENT = 50    # All seems lost
    CLIMAX = 60         # Peak confrontation
    FALLING = 70        # Aftermath
    RESOLUTION = 80     # New equilibrium


class CorrectionType(Enum):
    """Types of corrective force the system can apply."""
    NONE = auto()           # No correction needed
    HINT = auto()           # Environmental hints
    NPC_MENTION = auto()    # NPCs casually mention things
    NPC_SEEK = auto()       # NPCs actively approach player
    EVENT_NEARBY = auto()   # Events happen near player
    EVENT_FORCE = auto()    # Events happen TO player
    ADAPT_BEAT = auto()     # Beat adapts to player location


# ═══════════════════════════════════════════════════════════════════════════════
# STORY BEAT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StoryBeat:
    """A significant narrative moment in the plot tree."""
    
    beat_id: str
    name: str
    beat_type: BeatType
    
    # Tree structure
    parent_beats: List[str] = field(default_factory=list)
    child_beats: List[str] = field(default_factory=list)
    is_required: bool = True  # Must this beat happen?
    
    # Trigger requirements
    required_entities: Set[str] = field(default_factory=set)
    required_flags: Set[str] = field(default_factory=set)
    blocking_flags: Set[str] = field(default_factory=set)
    trigger_location: Optional[str] = None
    trigger_proximity: Set[str] = field(default_factory=set)
    
    # Activation parameters
    activation_threshold: float = 0.5
    base_urgency: float = 0.5
    urgency_growth_rate: float = 0.01  # Per tick
    
    # Content
    prose_fragments: List[str] = field(default_factory=list)
    spawns_entities: List[Dict] = field(default_factory=list)
    sets_flags: Set[str] = field(default_factory=set)
    clears_flags: Set[str] = field(default_factory=set)
    
    # Recovery content (for guiding derailed players)
    recovery_hints: List[str] = field(default_factory=list)
    recovery_npc_ids: List[str] = field(default_factory=list)
    recovery_dialogue: Dict[str, List[str]] = field(default_factory=dict)
    
    # State
    completed: bool = False
    completion_time: Optional[float] = None
    current_urgency: float = field(default=0.5)
    
    def __post_init__(self):
        self.current_urgency = self.base_urgency
    
    def tick_urgency(self):
        """Increase urgency over time (beat wants to happen)."""
        self.current_urgency = min(1.0, self.current_urgency + self.urgency_growth_rate)
    
    def reset_urgency(self):
        """Reset urgency after beat completes or fails."""
        self.current_urgency = self.base_urgency


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT TREE
# ═══════════════════════════════════════════════════════════════════════════════

class PlotTree:
    """
    The complete plot structure as a tree of beats.
    
    Not a linear sequence but a graph of possibilities.
    Multiple paths can lead to multiple endings.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.beats: Dict[str, StoryBeat] = {}
        
        # Track which beats are roots (no parents)
        self.root_beats: Set[str] = set()
        
        # Track which beats are endings (no children)
        self.ending_beats: Set[str] = set()
        
        # Branch weights for probabilistic selection
        self.branch_weights: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    def add_beat(self, beat: StoryBeat):
        """Add a beat to the tree."""
        self.beats[beat.beat_id] = beat
        
        # Update root/ending sets
        if not beat.parent_beats:
            self.root_beats.add(beat.beat_id)
        if not beat.child_beats:
            self.ending_beats.add(beat.beat_id)
        
        # Update parent references
        for parent_id in beat.parent_beats:
            if parent_id in self.beats:
                parent = self.beats[parent_id]
                if beat.beat_id not in parent.child_beats:
                    parent.child_beats.append(beat.beat_id)
                # Remove parent from endings since it now has a child
                self.ending_beats.discard(parent_id)
    
    def add_branch(self, from_beat_id: str, to_beat_id: str, weight: float = 1.0):
        """Add a weighted branch between beats."""
        if from_beat_id in self.beats and to_beat_id in self.beats:
            self.beats[from_beat_id].child_beats.append(to_beat_id)
            self.beats[to_beat_id].parent_beats.append(from_beat_id)
            self.branch_weights[from_beat_id][to_beat_id] = weight
            
            self.ending_beats.discard(from_beat_id)
            self.root_beats.discard(to_beat_id)
    
    def get_available_beats(self, completed: Set[str], flags: Set[str]) -> List[StoryBeat]:
        """
        Get beats that could trigger given current game state.
        
        A beat is available if:
        - All parent beats are completed
        - It's not already completed
        - Required flags are set
        - Blocking flags are not set
        """
        available = []
        
        for beat in self.beats.values():
            if beat.completed:
                continue
            
            # Check parent completion
            if beat.parent_beats:
                if not all(pid in completed for pid in beat.parent_beats):
                    continue
            
            # Check flags
            if not beat.required_flags.issubset(flags):
                continue
            
            if beat.blocking_flags.intersection(flags):
                continue
            
            available.append(beat)
        
        return available
    
    def get_most_urgent_beat(self, completed: Set[str], flags: Set[str]) -> Optional[StoryBeat]:
        """Get the beat that most wants to happen."""
        available = self.get_available_beats(completed, flags)
        if not available:
            return None
        
        return max(available, key=lambda b: b.current_urgency)
    
    def tick_all_urgency(self, completed: Set[str], flags: Set[str]):
        """Increase urgency for all available beats."""
        for beat in self.get_available_beats(completed, flags):
            beat.tick_urgency()
    
    def get_path_to_ending(self, from_beat_id: str) -> List[List[str]]:
        """Get all possible paths from a beat to any ending."""
        paths = []
        
        def dfs(current_id: str, path: List[str]):
            path = path + [current_id]
            beat = self.beats.get(current_id)
            
            if not beat:
                return
            
            if current_id in self.ending_beats:
                paths.append(path)
                return
            
            for child_id in beat.child_beats:
                dfs(child_id, path)
        
        dfs(from_beat_id, [])
        return paths


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT RECIPE (GENRE TEMPLATE)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlotRecipe:
    """
    Template defining how a genre's plot should behave.
    Controls correction aggressiveness and pacing.
    """
    
    name: str
    genre: str
    
    # Required beat structure
    required_beat_types: List[BeatType] = field(default_factory=list)
    
    # Pacing
    min_beats: int = 5
    max_beats: int = 20
    target_tension_curve: List[float] = field(default_factory=list)
    
    # Correction parameters
    tension_thresholds: Dict[CorrectionType, float] = field(default_factory=dict)
    correction_cooldowns: Dict[CorrectionType, int] = field(default_factory=dict)
    
    # Recovery preferences
    preferred_corrections: List[CorrectionType] = field(default_factory=list)
    allow_beat_adaptation: bool = True
    
    # Ending behavior
    ending_weights: Dict[str, float] = field(default_factory=dict)
    allow_tragic_ending: bool = True
    
    def __post_init__(self):
        # Default tension thresholds
        if not self.tension_thresholds:
            self.tension_thresholds = {
                CorrectionType.HINT: 0.3,
                CorrectionType.NPC_MENTION: 0.5,
                CorrectionType.NPC_SEEK: 0.7,
                CorrectionType.EVENT_NEARBY: 0.8,
                CorrectionType.EVENT_FORCE: 0.95,
                CorrectionType.ADAPT_BEAT: 0.99
            }
        
        # Default cooldowns (ticks)
        if not self.correction_cooldowns:
            self.correction_cooldowns = {
                CorrectionType.HINT: 30,
                CorrectionType.NPC_MENTION: 50,
                CorrectionType.NPC_SEEK: 100,
                CorrectionType.EVENT_NEARBY: 150,
                CorrectionType.EVENT_FORCE: 200,
                CorrectionType.ADAPT_BEAT: 300
            }


# Pre-built recipes
MYSTERY_RECIPE = PlotRecipe(
    name="Classic Mystery",
    genre="mystery",
    required_beat_types=[
        BeatType.OPENING, BeatType.INCITING, BeatType.MIDPOINT,
        BeatType.CLIMAX, BeatType.RESOLUTION
    ],
    min_beats=5,
    max_beats=15,
    target_tension_curve=[0.2, 0.4, 0.6, 0.5, 0.8, 0.95, 0.7, 0.3],
    preferred_corrections=[
        CorrectionType.HINT,
        CorrectionType.NPC_MENTION,
        CorrectionType.NPC_SEEK
    ],
    allow_beat_adaptation=True,
    allow_tragic_ending=True
)

ADVENTURE_RECIPE = PlotRecipe(
    name="Hero's Journey",
    genre="adventure",
    required_beat_types=[
        BeatType.OPENING, BeatType.INCITING, BeatType.RISING,
        BeatType.DARK_MOMENT, BeatType.CLIMAX, BeatType.RESOLUTION
    ],
    min_beats=7,
    max_beats=25,
    target_tension_curve=[0.1, 0.3, 0.5, 0.6, 0.4, 0.7, 0.9, 1.0, 0.5, 0.2],
    tension_thresholds={
        CorrectionType.HINT: 0.4,  # More exploration allowed
        CorrectionType.NPC_MENTION: 0.6,
        CorrectionType.NPC_SEEK: 0.8,
        CorrectionType.EVENT_NEARBY: 0.9,
        CorrectionType.EVENT_FORCE: 0.98,
        CorrectionType.ADAPT_BEAT: 1.0  # Never adapt
    },
    preferred_corrections=[
        CorrectionType.EVENT_NEARBY,  # World events drive action
        CorrectionType.HINT
    ],
    allow_beat_adaptation=True,
    allow_tragic_ending=True
)

ROMANCE_RECIPE = PlotRecipe(
    name="Love Story",
    genre="romance",
    required_beat_types=[
        BeatType.OPENING, BeatType.INCITING, BeatType.RISING,
        BeatType.MIDPOINT, BeatType.DARK_MOMENT, BeatType.RESOLUTION
    ],
    min_beats=6,
    max_beats=18,
    target_tension_curve=[0.1, 0.3, 0.5, 0.7, 0.4, 0.8, 0.9, 0.3],
    preferred_corrections=[
        CorrectionType.NPC_MENTION,  # Character-driven
        CorrectionType.NPC_SEEK
    ],
    allow_beat_adaptation=True,
    allow_tragic_ending=False  # Happy endings preferred
)


# ═══════════════════════════════════════════════════════════════════════════════
# EQUILIBRIUM ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CorrectionEvent:
    """A correction the engine wants to apply."""
    correction_type: CorrectionType
    target_beat: StoryBeat
    content: Any  # Hint text, NPC ID, event data, etc.
    priority: float


class EquilibriumEngine:
    """
    Maintains narrative coherence by applying corrective forces
    when players stray from the plot tree.
    
    Core loop:
    1. Calculate tension (how far off-track)
    2. Select appropriate correction type
    3. Apply correction (hints, NPCs, events)
    4. Track cooldowns
    """
    
    def __init__(self, plot_tree: PlotTree, recipe: PlotRecipe):
        self.tree = plot_tree
        self.recipe = recipe
        
        # State
        self.completed_beats: Set[str] = set()
        self.active_flags: Set[str] = set()
        self.tension: float = 0.0
        self.story_phase: float = 0.0  # 0-1 progress through story
        
        # Cooldowns
        self.cooldowns: Dict[CorrectionType, int] = {ct: 0 for ct in CorrectionType}
        
        # Queued corrections
        self.pending_corrections: List[CorrectionEvent] = []
        
        # Metrics
        self.total_corrections: int = 0
        self.corrections_by_type: Dict[CorrectionType, int] = defaultdict(int)
    
    def tick(self, player_state: Dict) -> Optional[CorrectionEvent]:
        """
        Main tick: update tension, apply corrections.
        
        Returns a correction event if one should be applied.
        """
        # Update urgency for all available beats
        self.tree.tick_all_urgency(self.completed_beats, self.active_flags)
        
        # Decrease cooldowns
        for ct in self.cooldowns:
            if self.cooldowns[ct] > 0:
                self.cooldowns[ct] -= 1
        
        # Calculate current tension
        self.tension = self._calculate_tension(player_state)
        
        # Update story phase
        self._update_story_phase()
        
        # Check for natural beat trigger (no correction needed)
        natural_trigger = self._check_natural_triggers(player_state)
        if natural_trigger:
            self._complete_beat(natural_trigger)
            return None
        
        # Determine if correction is needed
        correction_type = self._select_correction_type()
        
        if correction_type == CorrectionType.NONE:
            return None
        
        # Generate correction
        correction = self._generate_correction(correction_type, player_state)
        
        if correction:
            self.cooldowns[correction_type] = self.recipe.correction_cooldowns[correction_type]
            self.total_corrections += 1
            self.corrections_by_type[correction_type] += 1
        
        return correction
    
    def _calculate_tension(self, player_state: Dict) -> float:
        """
        Calculate how "off-track" the player is.
        
        Factors:
        - Time since last beat
        - Distance from beat triggers
        - Missing requirements
        - Story phase expectations
        """
        available = self.tree.get_available_beats(self.completed_beats, self.active_flags)
        
        if not available:
            return 0.0  # Story complete, no tension
        
        # Find best readiness among available beats
        best_readiness = 0.0
        for beat in available:
            readiness = self._calculate_readiness(beat, player_state)
            best_readiness = max(best_readiness, readiness)
        
        # Base tension from readiness
        base_tension = 1.0 - best_readiness
        
        # Factor in time since last beat
        ticks_since_beat = player_state.get('ticks_since_beat', 0)
        max_ticks = 500  # Arbitrary max
        time_factor = min(1.0, ticks_since_beat / max_ticks)
        
        # Factor in story phase expectations
        expected_tension = self._get_expected_tension()
        phase_factor = abs(base_tension - expected_tension)
        
        # Combine factors
        tension = (base_tension * 0.5) + (time_factor * 0.3) + (phase_factor * 0.2)
        
        return min(1.0, max(0.0, tension))
    
    def _calculate_readiness(self, beat: StoryBeat, player_state: Dict) -> float:
        """How ready is a beat to trigger?"""
        score = 0.0
        weights = 0.0
        
        # Location match
        if beat.trigger_location:
            weights += 1.0
            if player_state.get('location') == beat.trigger_location:
                score += 1.0
            else:
                # Partial credit for being close
                distance = player_state.get('distance_to', {}).get(beat.trigger_location, 100)
                score += max(0, 1.0 - (distance / 10))
        
        # Entity proximity
        if beat.trigger_proximity:
            weights += 1.0
            known = player_state.get('known_entities', set())
            nearby = player_state.get('nearby_entities', set())
            
            required = beat.trigger_proximity
            found = required.intersection(known | nearby)
            
            score += len(found) / len(required) if required else 1.0
        
        # Required entities known
        if beat.required_entities:
            weights += 0.5
            known = player_state.get('known_entities', set())
            found = beat.required_entities.intersection(known)
            score += 0.5 * (len(found) / len(beat.required_entities))
        
        # Flags
        if beat.required_flags:
            weights += 0.5
            current_flags = self.active_flags
            found = beat.required_flags.intersection(current_flags)
            score += 0.5 * (len(found) / len(beat.required_flags))
        
        return score / weights if weights > 0 else 1.0
    
    def _get_expected_tension(self) -> float:
        """Get expected tension for current story phase."""
        if not self.recipe.target_tension_curve:
            return 0.5
        
        curve = self.recipe.target_tension_curve
        index = int(self.story_phase * (len(curve) - 1))
        index = min(index, len(curve) - 1)
        
        return curve[index]
    
    def _update_story_phase(self):
        """Update progress through the story."""
        total_required = sum(1 for b in self.tree.beats.values() if b.is_required)
        completed_required = sum(
            1 for b in self.tree.beats.values() 
            if b.is_required and b.beat_id in self.completed_beats
        )
        
        if total_required > 0:
            self.story_phase = completed_required / total_required
    
    def _check_natural_triggers(self, player_state: Dict) -> Optional[StoryBeat]:
        """Check if any beat triggers naturally (without correction)."""
        available = self.tree.get_available_beats(self.completed_beats, self.active_flags)
        
        for beat in available:
            readiness = self._calculate_readiness(beat, player_state)
            if readiness >= beat.activation_threshold:
                return beat
        
        return None
    
    def _select_correction_type(self) -> CorrectionType:
        """Select appropriate correction based on tension level."""
        # Check thresholds from least to most aggressive
        for ct in [
            CorrectionType.HINT,
            CorrectionType.NPC_MENTION,
            CorrectionType.NPC_SEEK,
            CorrectionType.EVENT_NEARBY,
            CorrectionType.EVENT_FORCE,
            CorrectionType.ADAPT_BEAT
        ]:
            threshold = self.recipe.tension_thresholds.get(ct, 1.0)
            
            if self.tension >= threshold:
                # Check cooldown
                if self.cooldowns[ct] <= 0:
                    # Check if this correction type is preferred
                    if ct in self.recipe.preferred_corrections:
                        return ct
                    # Or if tension is very high, use anyway
                    elif self.tension >= threshold + 0.1:
                        return ct
        
        return CorrectionType.NONE
    
    def _generate_correction(self, correction_type: CorrectionType,
                            player_state: Dict) -> Optional[CorrectionEvent]:
        """Generate specific correction content."""
        
        # Find target beat
        target = self.tree.get_most_urgent_beat(self.completed_beats, self.active_flags)
        if not target:
            return None
        
        if correction_type == CorrectionType.HINT:
            return self._generate_hint(target, player_state)
        
        elif correction_type == CorrectionType.NPC_MENTION:
            return self._generate_npc_mention(target, player_state)
        
        elif correction_type == CorrectionType.NPC_SEEK:
            return self._generate_npc_seek(target, player_state)
        
        elif correction_type == CorrectionType.EVENT_NEARBY:
            return self._generate_event_nearby(target, player_state)
        
        elif correction_type == CorrectionType.EVENT_FORCE:
            return self._generate_event_force(target, player_state)
        
        elif correction_type == CorrectionType.ADAPT_BEAT:
            return self._generate_adapted_beat(target, player_state)
        
        return None
    
    def _generate_hint(self, beat: StoryBeat, 
                       player_state: Dict) -> Optional[CorrectionEvent]:
        """Generate an environmental hint."""
        if not beat.recovery_hints:
            return None
        
        hint = random.choice(beat.recovery_hints)
        
        return CorrectionEvent(
            correction_type=CorrectionType.HINT,
            target_beat=beat,
            content={'type': 'hint', 'text': hint},
            priority=self.tension
        )
    
    def _generate_npc_mention(self, beat: StoryBeat,
                              player_state: Dict) -> Optional[CorrectionEvent]:
        """Generate NPC casual mention of relevant topic."""
        nearby_npcs = player_state.get('nearby_npcs', [])
        
        if not nearby_npcs:
            return None
        
        npc = random.choice(nearby_npcs)
        
        # Get or generate dialogue
        dialogue = beat.recovery_dialogue.get(npc, [
            f"I heard something strange happened near {beat.trigger_location}...",
            f"Have you been to {beat.trigger_location} lately?",
        ])
        
        return CorrectionEvent(
            correction_type=CorrectionType.NPC_MENTION,
            target_beat=beat,
            content={
                'type': 'npc_mention',
                'npc_id': npc,
                'dialogue': random.choice(dialogue)
            },
            priority=self.tension
        )
    
    def _generate_npc_seek(self, beat: StoryBeat,
                          player_state: Dict) -> Optional[CorrectionEvent]:
        """Generate NPC actively seeking player."""
        if not beat.recovery_npc_ids:
            return None
        
        npc_id = random.choice(beat.recovery_npc_ids)
        
        return CorrectionEvent(
            correction_type=CorrectionType.NPC_SEEK,
            target_beat=beat,
            content={
                'type': 'npc_seek',
                'npc_id': npc_id,
                'message': f"{npc_id} arrives, looking for you urgently."
            },
            priority=self.tension
        )
    
    def _generate_event_nearby(self, beat: StoryBeat,
                               player_state: Dict) -> Optional[CorrectionEvent]:
        """Generate event happening near player."""
        return CorrectionEvent(
            correction_type=CorrectionType.EVENT_NEARBY,
            target_beat=beat,
            content={
                'type': 'event',
                'description': f"Something draws your attention toward {beat.trigger_location}...",
                'spawns': beat.spawns_entities[:1] if beat.spawns_entities else []
            },
            priority=self.tension
        )
    
    def _generate_event_force(self, beat: StoryBeat,
                             player_state: Dict) -> Optional[CorrectionEvent]:
        """Generate event that happens directly to player."""
        return CorrectionEvent(
            correction_type=CorrectionType.EVENT_FORCE,
            target_beat=beat,
            content={
                'type': 'forced_event',
                'description': "Events beyond your control unfold...",
                'beat_fragments': beat.prose_fragments,
                'sets_flags': list(beat.sets_flags)
            },
            priority=1.0
        )
    
    def _generate_adapted_beat(self, beat: StoryBeat,
                               player_state: Dict) -> Optional[CorrectionEvent]:
        """
        Adapt the beat to the player's current position.
        The beat happens HERE instead of its original location.
        """
        if not self.recipe.allow_beat_adaptation:
            return None
        
        adapted = copy.deepcopy(beat)
        adapted.trigger_location = player_state.get('location')
        
        return CorrectionEvent(
            correction_type=CorrectionType.ADAPT_BEAT,
            target_beat=adapted,
            content={
                'type': 'adapted_beat',
                'original_location': beat.trigger_location,
                'adapted_location': adapted.trigger_location,
                'fragments': beat.prose_fragments
            },
            priority=1.0
        )
    
    def _complete_beat(self, beat: StoryBeat):
        """Mark a beat as completed."""
        beat.completed = True
        beat.completion_time = time.time()
        self.completed_beats.add(beat.beat_id)
        
        # Apply flags
        self.active_flags.update(beat.sets_flags)
        self.active_flags -= beat.clears_flags
        
        # Reset urgency for related beats
        beat.reset_urgency()
    
    def complete_beat(self, beat_id: str):
        """External call to complete a beat."""
        if beat_id in self.tree.beats:
            self._complete_beat(self.tree.beats[beat_id])
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            'tension': self.tension,
            'story_phase': self.story_phase,
            'completed_beats': len(self.completed_beats),
            'total_beats': len(self.tree.beats),
            'total_corrections': self.total_corrections,
            'corrections_by_type': dict(self.corrections_by_type),
            'active_flags': list(self.active_flags)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_demo_plot() -> PlotTree:
    """Create a simple mystery plot for testing."""
    
    tree = PlotTree("Murder at Manor House")
    
    # Opening
    opening = StoryBeat(
        beat_id="opening",
        name="Arrival at Manor",
        beat_type=BeatType.OPENING,
        trigger_location="manor_entrance",
        prose_fragments=["opening_prose"],
        sets_flags={"arrived_at_manor"},
        recovery_hints=[
            "The manor looms on the hill, its windows dark.",
            "You notice a car parked at the manor gates."
        ]
    )
    tree.add_beat(opening)
    
    # Inciting
    inciting = StoryBeat(
        beat_id="inciting",
        name="Body Discovered",
        beat_type=BeatType.INCITING,
        parent_beats=["opening"],
        trigger_location="library",
        required_flags={"arrived_at_manor"},
        prose_fragments=["body_discovered_prose"],
        sets_flags={"murder_known"},
        recovery_hints=[
            "A scream echoes from inside the manor.",
            "Someone rushes past, pale-faced."
        ],
        recovery_npc_ids=["butler"],
        recovery_dialogue={
            "butler": [
                "Please, you must come to the library immediately!",
                "Something terrible has happened..."
            ]
        }
    )
    tree.add_beat(inciting)
    
    # Midpoint (branching)
    clue_a = StoryBeat(
        beat_id="clue_letter",
        name="Find the Letter",
        beat_type=BeatType.MIDPOINT,
        parent_beats=["inciting"],
        trigger_location="study",
        required_flags={"murder_known"},
        required_entities={"letter"},
        prose_fragments=["letter_discovery"],
        sets_flags={"has_letter_clue"},
        recovery_hints=[
            "Papers rustle in the study as if disturbed.",
            "The desk drawer isn't fully closed..."
        ]
    )
    tree.add_beat(clue_a)
    
    clue_b = StoryBeat(
        beat_id="clue_witness",
        name="Witness Testimony",
        beat_type=BeatType.MIDPOINT,
        parent_beats=["inciting"],
        trigger_proximity={"maid"},
        required_flags={"murder_known"},
        prose_fragments=["witness_testimony"],
        sets_flags={"has_witness_clue"},
        recovery_hints=[
            "The maid seems nervous, glancing around.",
            "Someone is crying softly nearby."
        ],
        recovery_npc_ids=["maid"]
    )
    tree.add_beat(clue_b)
    
    # Climax
    climax = StoryBeat(
        beat_id="confrontation",
        name="Confront the Killer",
        beat_type=BeatType.CLIMAX,
        parent_beats=["clue_letter", "clue_witness"],
        trigger_location="drawing_room",
        required_flags={"has_letter_clue", "has_witness_clue"},
        trigger_proximity={"lord_blackwood"},
        prose_fragments=["confrontation_prose"],
        sets_flags={"killer_confronted"},
        base_urgency=0.8,
        recovery_hints=[
            "Lord Blackwood is in the drawing room, alone.",
            "Now might be the time to act..."
        ]
    )
    tree.add_beat(climax)
    
    # Resolution
    resolution = StoryBeat(
        beat_id="resolution",
        name="Justice Served",
        beat_type=BeatType.RESOLUTION,
        parent_beats=["confrontation"],
        required_flags={"killer_confronted"},
        prose_fragments=["resolution_prose"],
        is_required=True
    )
    tree.add_beat(resolution)
    
    return tree


if __name__ == "__main__":
    print("=" * 70)
    print("Plot Equilibrium Engine - Demonstration")
    print("=" * 70)
    
    # Create plot and engine
    plot = create_demo_plot()
    engine = EquilibriumEngine(plot, MYSTERY_RECIPE)
    
    print(f"\nPlot: {plot.name}")
    print(f"Total beats: {len(plot.beats)}")
    print(f"Root beats: {plot.root_beats}")
    print(f"Ending beats: {plot.ending_beats}")
    
    # Simulate player state
    player_state = {
        'location': 'garden',  # Not at manor entrance
        'known_entities': set(),
        'nearby_entities': set(),
        'nearby_npcs': ['gardener'],
        'ticks_since_beat': 0,
        'distance_to': {'manor_entrance': 5}
    }
    
    print("\n--- Simulation ---")
    
    for tick in range(200):
        player_state['ticks_since_beat'] = tick
        
        correction = engine.tick(player_state)
        
        if correction:
            print(f"\nTick {tick}: Tension={engine.tension:.2f}")
            print(f"  Correction: {correction.correction_type.name}")
            print(f"  Target beat: {correction.target_beat.name}")
            print(f"  Content: {correction.content}")
        
        # Simulate player eventually reaching locations
        if tick == 50:
            player_state['location'] = 'manor_entrance'
            print(f"\nTick {tick}: Player arrives at manor entrance")
        
        if tick == 80:
            engine.complete_beat('opening')
            player_state['location'] = 'hallway'
            print(f"\nTick {tick}: Opening beat completed")
        
        if tick == 120:
            player_state['location'] = 'library'
            print(f"\nTick {tick}: Player enters library")
    
    print("\n--- Final Stats ---")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Done!")
