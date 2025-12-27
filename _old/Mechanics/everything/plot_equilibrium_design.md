# Plot Equilibrium System Design

## The Core Problem

Players derail stories. That's the fun. But the game needs to feel like a coherent narrative, not chaos. 

The solution: **Plot Equilibrium** — a system that gently guides derailed players back toward meaningful story beats, using multiple possible paths, not railroad tracks.

---

## 1. The Plot Tree (Not a Line)

A novel's plot isn't a single path — it's a **tree of possibilities** with:
- **Trunk** = Core story beats that MUST happen for the story to work
- **Branches** = Variations on HOW those beats unfold
- **Leaves** = Player-specific details (who finds the key, what dialogue choices)

```
                    [ENDING: Mystery Solved]
                           /    \
        [REVEAL: Villain Exposed]  [REVEAL: Tragic Truth]
               /     \                    /      \
    [CLIMAX: Confrontation]      [CLIMAX: Sacrifice]
           /   |   \                  /   |   \
      ...many paths...           ...many paths...
           \   |   /                  \   |   /
        [MIDPOINT: First Clue]
               |
        [INCITING: Discovery]
               |
          [OPENING]
```

### Beat Definition

```python
@dataclass
class StoryBeat:
    """A significant narrative moment in the plot tree."""
    
    beat_id: str
    name: str
    beat_type: BeatType  # OPENING, INCITING, RISING, MIDPOINT, CLIMAX, FALLING, RESOLUTION
    
    # Tree position
    parent_beats: List[str]  # Beats that can lead to this one
    child_beats: List[str]   # Beats this can lead to
    
    # Requirements
    required_entities: List[str]  # What must exist for this beat
    required_flags: List[str]     # What state must be true
    blocking_flags: List[str]     # What state prevents this beat
    
    # Trigger conditions
    trigger_location: Optional[str]  # Where this can trigger
    trigger_proximity: List[str]     # Entities that must be nearby
    trigger_time_range: Optional[Tuple[int, int]]  # Game time window
    
    # Activation
    activation_threshold: float = 0.5  # How "ready" conditions must be (0-1)
    urgency: float = 0.5  # How strongly this beat pulls (increases over time)
    
    # Content
    prose_fragments: List[str]  # Fragment IDs to compose when triggered
    spawns_entities: List[Dict]  # Entities to create when triggered
    sets_flags: List[str]
    clears_flags: List[str]
    
    # Recovery hooks
    recovery_hints: List[str]  # Prose fragments hinting toward this beat
    recovery_npcs: List[str]   # NPCs who can nudge toward this beat


class BeatType(Enum):
    OPENING = auto()
    INCITING = auto()
    RISING = auto()
    MIDPOINT = auto()
    CLIMAX = auto()
    FALLING = auto()
    RESOLUTION = auto()
```

---

## 2. The Equilibrium Engine

The game constantly calculates **narrative tension** — the distance between:
- Where the player IS in the story
- Where the story WANTS to go

When tension is high (player derailed), the system applies **corrective forces**.

```python
class EquilibriumEngine:
    """
    Maintains narrative coherence by applying gentle corrective forces
    when players stray from the plot tree.
    """
    
    def __init__(self, plot_tree: PlotTree, world: EntityGraph):
        self.tree = plot_tree
        self.world = world
        
        # Current narrative state
        self.completed_beats: Set[str] = set()
        self.active_beat: Optional[str] = None
        self.tension: float = 0.0  # 0 = on track, 1 = maximally derailed
        
        # Recovery mechanisms
        self.hint_cooldown: int = 0
        self.npc_nudge_cooldown: int = 0
        self.event_injection_cooldown: int = 0
    
    def tick(self, player_state: Dict, game_time: int):
        """Called each game tick to update equilibrium."""
        
        # 1. Calculate tension
        self.tension = self._calculate_tension(player_state)
        
        # 2. Check for beat triggers
        triggered = self._check_beat_triggers(player_state, game_time)
        if triggered:
            self._activate_beat(triggered, player_state)
            return
        
        # 3. Apply corrective forces based on tension level
        if self.tension > 0.3:
            self._apply_gentle_correction(player_state)
        if self.tension > 0.6:
            self._apply_moderate_correction(player_state)
        if self.tension > 0.9:
            self._apply_strong_correction(player_state)
    
    def _calculate_tension(self, player_state: Dict) -> float:
        """
        How far off-track is the player?
        
        Factors:
        - Time since last beat completion
        - Distance from beat trigger locations
        - Missing required items/flags
        - Active distractions (side quests, exploration)
        """
        available_beats = self._get_available_beats()
        if not available_beats:
            return 0.0  # All done, no tension
        
        # Find the "nearest" available beat
        best_readiness = 0.0
        for beat_id in available_beats:
            beat = self.tree.beats[beat_id]
            readiness = self._calculate_beat_readiness(beat, player_state)
            best_readiness = max(best_readiness, readiness)
        
        # Tension is inverse of readiness
        # High readiness = low tension (player is close to triggering)
        # Low readiness = high tension (player is far from any beat)
        base_tension = 1.0 - best_readiness
        
        # Factor in time drift
        ticks_since_beat = player_state.get('ticks_since_beat', 0)
        time_tension = min(1.0, ticks_since_beat / 1000)  # Max after 1000 ticks
        
        return (base_tension * 0.7) + (time_tension * 0.3)
    
    def _apply_gentle_correction(self, player_state: Dict):
        """
        Low tension: Environmental hints only.
        
        - Atmospheric fragments mentioning relevant things
        - Items that "catch the eye"
        - Weather/time changes that suggest direction
        """
        if self.hint_cooldown > 0:
            self.hint_cooldown -= 1
            return
        
        # Find best available beat
        beat = self._get_most_urgent_beat()
        if not beat:
            return
        
        # Add hint fragment to current location
        hint = random.choice(beat.recovery_hints) if beat.recovery_hints else None
        if hint:
            self._inject_hint_fragment(hint, player_state)
            self.hint_cooldown = 50  # Wait before next hint
    
    def _apply_moderate_correction(self, player_state: Dict):
        """
        Medium tension: NPC intervention.
        
        - NPCs mention rumors/news about relevant things
        - Characters offer to guide player somewhere
        - Overheard conversations hint at plot
        """
        if self.npc_nudge_cooldown > 0:
            self.npc_nudge_cooldown -= 1
            return
        
        beat = self._get_most_urgent_beat()
        if not beat or not beat.recovery_npcs:
            return
        
        # Find a recovery NPC near the player
        nearby_npcs = self._get_nearby_entities(
            player_state['location'], 
            EntityType.CHARACTER,
            depth=3
        )
        
        recovery_npc = None
        for npc in nearby_npcs:
            if npc.id in beat.recovery_npcs:
                recovery_npc = npc
                break
        
        if recovery_npc:
            self._trigger_npc_nudge(recovery_npc, beat, player_state)
            self.npc_nudge_cooldown = 100
    
    def _apply_strong_correction(self, player_state: Dict):
        """
        High tension: Plot intrusion.
        
        - Events happen regardless of player position
        - NPCs seek out the player
        - Environmental changes force movement
        - Time skips to critical moments
        """
        if self.event_injection_cooldown > 0:
            self.event_injection_cooldown -= 1
            return
        
        beat = self._get_most_urgent_beat()
        if not beat:
            return
        
        # Check if we can lower requirements temporarily
        adjusted_beat = self._create_adapted_beat(beat, player_state)
        
        # Inject a plot event that moves things forward
        self._inject_plot_event(adjusted_beat, player_state)
        self.event_injection_cooldown = 200
    
    def _create_adapted_beat(self, beat: StoryBeat, player_state: Dict) -> StoryBeat:
        """
        Create a modified version of the beat that works with
        current player state — adapt the story to the player.
        """
        adapted = copy.deepcopy(beat)
        
        # If player is missing required entities, spawn substitutes
        for entity_id in beat.required_entities:
            if entity_id not in player_state.get('known_entities', set()):
                # Create substitute or alternative
                substitute = self._find_substitute_entity(entity_id)
                if substitute:
                    adapted.required_entities.remove(entity_id)
                    adapted.required_entities.append(substitute.id)
        
        # If player missed required location, allow trigger elsewhere
        if beat.trigger_location:
            if player_state['location'] != beat.trigger_location:
                adapted.trigger_location = player_state['location']
                adapted.prose_fragments = self._adapt_fragments_to_location(
                    beat.prose_fragments, 
                    player_state['location']
                )
        
        return adapted
```

---

## 3. Plot Recipes (Genre Templates)

Different genres have different shapes. A **Recipe** defines:
- What beats are REQUIRED vs optional
- How much derailment is acceptable
- What correction forces to use
- How endings can branch

```python
@dataclass
class PlotRecipe:
    """Template for a genre of story."""
    
    name: str
    genre: str  # "mystery", "romance", "thriller", "adventure"
    
    # Beat structure
    required_beat_types: List[BeatType]  # Must have at least one of each
    optional_beat_types: List[BeatType]  # Can be skipped
    
    # Pacing
    minimum_beats_to_ending: int
    maximum_beats_to_ending: int
    ideal_tension_curve: List[float]  # Target tension at each story phase
    
    # Derailment tolerance
    max_tension_before_intervention: float
    intervention_aggressiveness: float  # 0-1, how hard to push back
    
    # Recovery preferences
    preferred_correction: str  # "hints", "npcs", "events", "time_skip"
    allow_ending_adaptation: bool  # Can ending change based on derailment?
    
    # Ending structure
    possible_endings: List[str]  # Beat IDs that can end the story
    ending_weights: Dict[str, float]  # How the story "wants" to end


# Pre-built recipes
MYSTERY_RECIPE = PlotRecipe(
    name="Classic Mystery",
    genre="mystery",
    required_beat_types=[
        BeatType.OPENING,    # Establish normal world
        BeatType.INCITING,   # Crime/mystery revealed
        BeatType.MIDPOINT,   # Major clue discovered
        BeatType.CLIMAX,     # Confrontation/revelation
        BeatType.RESOLUTION  # Mystery solved
    ],
    optional_beat_types=[BeatType.RISING, BeatType.FALLING],
    minimum_beats_to_ending=5,
    maximum_beats_to_ending=15,
    ideal_tension_curve=[0.2, 0.4, 0.6, 0.8, 0.9, 0.7, 0.3],
    max_tension_before_intervention=0.7,
    intervention_aggressiveness=0.5,
    preferred_correction="npcs",  # Detectives, witnesses offer clues
    allow_ending_adaptation=True,  # Different solutions possible
    possible_endings=["reveal_villain", "tragic_truth", "false_solution"],
    ending_weights={"reveal_villain": 0.6, "tragic_truth": 0.3, "false_solution": 0.1}
)

ADVENTURE_RECIPE = PlotRecipe(
    name="Hero's Journey",
    genre="adventure", 
    required_beat_types=[
        BeatType.OPENING,
        BeatType.INCITING,
        BeatType.MIDPOINT,
        BeatType.CLIMAX,
        BeatType.RESOLUTION
    ],
    optional_beat_types=[BeatType.RISING, BeatType.FALLING],
    minimum_beats_to_ending=7,
    maximum_beats_to_ending=20,
    ideal_tension_curve=[0.1, 0.3, 0.5, 0.7, 0.5, 0.8, 1.0, 0.4],
    max_tension_before_intervention=0.8,  # More exploration allowed
    intervention_aggressiveness=0.3,  # Gentler pushes
    preferred_correction="events",  # World events drive action
    allow_ending_adaptation=True,
    possible_endings=["hero_triumph", "pyrrhic_victory", "ultimate_sacrifice"],
    ending_weights={"hero_triumph": 0.5, "pyrrhic_victory": 0.35, "ultimate_sacrifice": 0.15}
)
```

---

## 4. Image Pipeline → World Entities

Your my_infocom system converts images to entities. For the island game:

```python
class StreetViewWorldBuilder:
    """
    Walk through Street View images along a route,
    generating location entities and connections.
    """
    
    def __init__(self, vlm_client, llm_client, world: EntityGraph):
        self.vlm = vlm_client  # Vision-Language Model
        self.llm = llm_client  # For narrative enrichment
        self.world = world
        self.compositor = ProseCompositor()
    
    async def process_route(self, coordinates: List[Tuple[float, float]], 
                           route_name: str) -> List[Entity]:
        """
        Process a series of Street View locations into game entities.
        """
        entities = []
        previous_location = None
        
        for i, (lat, lon) in enumerate(coordinates):
            # 1. Fetch Street View image
            image = await self.fetch_streetview(lat, lon)
            
            # 2. VLM detection pass
            detections = await self.vlm_detect(image)
            
            # 3. VLM description pass
            description = await self.vlm_describe(image, detections)
            
            # 4. LLM narrative enrichment
            narrative = await self.llm_enrich(description, route_name, i)
            
            # 5. Convert to entity
            location = self.create_location_entity(
                name=f"{route_name}_location_{i}",
                lat=lat, lon=lon,
                vlm_output=description,
                narrative_output=narrative
            )
            
            # 6. Create sub-entities from detections
            for detection in detections:
                if detection['confidence'] > 0.7:
                    sub_entity = self.create_detected_entity(detection, location)
                    entities.append(sub_entity)
            
            # 7. Link to previous location
            if previous_location:
                self.world.relate(
                    previous_location, location,
                    RelationalKind.NORTH_OF,  # Or calculated direction
                    bidirectional=True
                )
            
            entities.append(location)
            previous_location = location
        
        return entities
    
    def create_location_entity(self, name: str, lat: float, lon: float,
                               vlm_output: Dict, narrative_output: Dict) -> Entity:
        """Convert pipeline output to location entity."""
        
        location = Entity(name, EntityType.LOCATION)
        
        # Base description from narrative
        location.add_description(
            narrative_output['prose_description'],
            FragmentCategory.BASE_DESCRIPTION,
            priority=100
        )
        
        # Atmospheric from VLM mood analysis
        if vlm_output.get('mood'):
            location.add_fragment(ProseFragment(
                text=self._mood_to_atmospheric(vlm_output['mood']),
                category=FragmentCategory.ATMOSPHERIC,
                priority=80
            ))
        
        # Sensory details
        for sense, text in narrative_output.get('sensory', {}).items():
            location.add_fragment(ProseFragment(
                text=text,
                category=FragmentCategory.SENSORY,
                tags={sense}
            ))
        
        # Time-of-day variations
        if vlm_output.get('time_of_day'):
            location.add_fragment(ProseFragment(
                text=narrative_output.get('morning_variation', ''),
                category=FragmentCategory.ATMOSPHERIC,
                conditions=[FragmentCondition(
                    lambda e, c: c.get('time_of_day') == 'morning',
                    "Morning only"
                )]
            ))
            location.add_fragment(ProseFragment(
                text=narrative_output.get('night_variation', ''),
                category=FragmentCategory.ATMOSPHERIC,
                conditions=[FragmentCondition(
                    lambda e, c: c.get('time_of_day') == 'night',
                    "Night only"
                )]
            ))
        
        # Store coordinates for real-world anchoring
        location.properties['lat'] = lat
        location.properties['lon'] = lon
        location.properties['streetview_source'] = True
        
        return location
    
    async def llm_enrich(self, vlm_description: Dict, 
                         route_name: str, position: int) -> Dict:
        """
        Use LLM to transform VLM output into rich narrative content.
        """
        prompt = f"""
You are creating prose for a text adventure game location.

VLM Description: {json.dumps(vlm_description)}
Route: {route_name}
Position: {position} of route

Generate:
1. prose_description: 2-3 sentences of evocative description
2. sensory: Dict with sight, sound, smell keys (if applicable)
3. morning_variation: How this looks at dawn
4. night_variation: How this looks at night
5. history_hint: A made-up historical detail that could be true
6. mystery_seed: Something that could become a plot hook

Return as JSON.
"""
        
        response = await self.llm.generate(prompt)
        return json.loads(response)
```

---

## 5. Novel → Plot Tree Extraction

Analyze a novel to extract its beat structure:

```python
class NovelPlotExtractor:
    """
    Analyze a novel and extract its plot tree structure.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def extract_plot_tree(self, novel_text: str, 
                                novel_title: str) -> PlotTree:
        """
        Full pipeline: novel text → plot tree with beats.
        """
        
        # 1. Segment into chapters/scenes
        segments = self.segment_text(novel_text)
        
        # 2. Identify major events in each segment
        events = []
        for segment in segments:
            segment_events = await self.extract_events(segment)
            events.extend(segment_events)
        
        # 3. Classify events into beat types
        beats = []
        for event in events:
            beat_type = await self.classify_beat(event, events)
            beat = self.event_to_beat(event, beat_type)
            beats.append(beat)
        
        # 4. Build tree structure (find parent/child relationships)
        tree = PlotTree(novel_title)
        for beat in beats:
            tree.add_beat(beat)
        
        self.infer_tree_structure(tree, novel_text)
        
        # 5. Generate alternative paths (branches)
        await self.generate_branch_alternatives(tree)
        
        # 6. Create recovery content for each beat
        await self.generate_recovery_content(tree)
        
        return tree
    
    async def extract_events(self, segment: str) -> List[Dict]:
        """Use LLM to identify narrative events in a segment."""
        
        prompt = f"""
Analyze this text segment and identify the major narrative events.

Text:
{segment[:3000]}

For each event, provide:
- event_id: unique identifier
- summary: one sentence description
- characters_involved: list of character names
- location: where it happens
- consequences: what changes because of this
- emotional_beat: the feeling (tension, relief, surprise, etc.)

Return as JSON array.
"""
        
        response = await self.llm.generate(prompt)
        return json.loads(response)
    
    async def classify_beat(self, event: Dict, all_events: List[Dict]) -> BeatType:
        """Classify an event into a story beat type."""
        
        prompt = f"""
Given this narrative event and its context, classify it.

Event: {json.dumps(event)}
Total events in story: {len(all_events)}
This event's position: {all_events.index(event)}

Classify as one of:
- OPENING: Establishes normal world before conflict
- INCITING: The event that starts the main conflict
- RISING: Complications and escalation
- MIDPOINT: Major shift or revelation
- CLIMAX: Peak confrontation or decision
- FALLING: Aftermath of climax
- RESOLUTION: How things settle

Return just the classification name.
"""
        
        response = await self.llm.generate(prompt)
        return BeatType[response.strip().upper()]
    
    async def generate_branch_alternatives(self, tree: PlotTree):
        """
        For key beats, generate alternative ways they could unfold.
        This creates the "branches" in the plot tree.
        """
        
        for beat in tree.beats.values():
            if beat.beat_type in [BeatType.MIDPOINT, BeatType.CLIMAX]:
                alternatives = await self.generate_alternatives(beat)
                for alt in alternatives:
                    alt_beat = self.event_to_beat(alt, beat.beat_type)
                    alt_beat.parent_beats = beat.parent_beats
                    tree.add_beat(alt_beat)
                    
                    # Original beat can lead to alternative outcomes
                    tree.add_branch(beat.beat_id, alt_beat.beat_id)
    
    async def generate_alternatives(self, beat: StoryBeat) -> List[Dict]:
        """Generate alternative versions of a beat."""
        
        prompt = f"""
This is a story beat: {beat.name}

Generate 2-3 alternative ways this beat could unfold that:
1. Maintain the story's core conflict
2. Have different outcomes or revelations
3. Could result from different player choices

For each alternative:
- summary: what happens instead
- trigger_difference: what player action leads here
- consequences: how this changes the story going forward

Return as JSON array.
"""
        
        response = await self.llm.generate(prompt)
        return json.loads(response)
    
    async def generate_recovery_content(self, tree: PlotTree):
        """
        For each beat, generate hints and NPC dialogue
        that can guide derailed players back to it.
        """
        
        for beat in tree.beats.values():
            # Generate environmental hints
            hints = await self.generate_hints(beat)
            beat.recovery_hints = hints
            
            # Generate NPC nudge dialogue
            nudges = await self.generate_npc_nudges(beat)
            beat.recovery_npcs = list(nudges.keys())
            
            # Store nudge content for NPCs
            for npc_id, dialogue in nudges.items():
                self.store_recovery_dialogue(npc_id, beat.beat_id, dialogue)
    
    async def generate_hints(self, beat: StoryBeat) -> List[str]:
        """Generate environmental hints toward a beat."""
        
        prompt = f"""
The player needs to be gently guided toward this story beat:
{beat.name}

Generate 3-5 subtle environmental hints that could appear
in nearby locations. These should:
- Not be obvious ("go to the library!")
- Feel natural in the world
- Create curiosity/mystery
- Point generally toward the beat's location or trigger

Examples of good hints:
- "A faint smell of smoke drifts from the east."
- "Someone has scratched a symbol into the wall here."
- "You notice fresh footprints leading toward the old mill."

Return as JSON array of strings.
"""
        
        response = await self.llm.generate(prompt)
        return json.loads(response)
```

---

## 6. The Integrated Game Loop

```python
class IslandGame:
    """
    The complete game integrating all systems.
    """
    
    def __init__(self):
        # Core systems
        self.world = EntityGraph("Island World")
        self.code_index = CodeIndex(CompleteLiveCodeSystem())
        self.compositor = ProseCompositor()
        
        # Plot system
        self.plot_tree: Optional[PlotTree] = None
        self.equilibrium: Optional[EquilibriumEngine] = None
        self.recipe: Optional[PlotRecipe] = None
        
        # Image pipeline
        self.street_builder: Optional[StreetViewWorldBuilder] = None
        
        # Player state
        self.player_state = {
            'location': None,
            'inventory': [],
            'flags': {},
            'known_entities': set(),
            'ticks_since_beat': 0
        }
        
        # Game time
        self.game_tick = 0
    
    async def load_novel_world(self, novel_path: str, recipe: PlotRecipe):
        """Load a novel and prepare its world."""
        
        novel_text = Path(novel_path).read_text()
        novel_title = Path(novel_path).stem
        
        # Extract plot tree from novel
        extractor = NovelPlotExtractor(self.llm)
        self.plot_tree = await extractor.extract_plot_tree(novel_text, novel_title)
        
        # Apply recipe to plot tree
        self.recipe = recipe
        self.plot_tree.apply_recipe(recipe)
        
        # Extract locations/entities from novel
        analyzer = NovelAnalyzer(self.llm, self.code_index)
        novel_entities = await analyzer.analyze(novel_text, novel_title)
        
        # Add to world
        for entity in novel_entities:
            self.world.register(entity)
        
        # Initialize equilibrium engine
        self.equilibrium = EquilibriumEngine(self.plot_tree, self.world)
    
    async def add_streetview_layer(self, coordinates: List[Tuple[float, float]],
                                    layer_name: str):
        """Add real-world locations from Street View."""
        
        self.street_builder = StreetViewWorldBuilder(
            self.vlm, self.llm, self.world
        )
        
        # Process route into entities
        locations = await self.street_builder.process_route(coordinates, layer_name)
        
        # Link Street View locations to novel locations by name/type matching
        self.link_real_to_fictional(locations)
    
    def link_real_to_fictional(self, real_locations: List[Entity]):
        """
        Connect real-world Street View locations to fictional novel locations.
        
        This creates the "island" effect — real places become fictional settings.
        """
        
        for real_loc in real_locations:
            # Find matching fictional location by tags/type
            best_match = self.find_best_fictional_match(real_loc)
            
            if best_match:
                # Merge: real location BECOMES the fictional one
                self.merge_locations(real_loc, best_match)
    
    async def game_loop(self):
        """Main game loop."""
        
        while True:
            # 1. Get player input
            player_input = await self.get_input()
            
            # 2. Parse and execute
            result = await self.process_command(player_input)
            
            # 3. Update equilibrium (check for derailment)
            self.equilibrium.tick(self.player_state, self.game_tick)
            
            # 4. Check for beat triggers
            triggered_beat = self.check_beat_triggers()
            if triggered_beat:
                beat_result = await self.execute_beat(triggered_beat)
                result = result + "\n\n" + beat_result
            
            # 5. Output to player
            await self.output(result)
            
            # 6. Tick
            self.game_tick += 1
            self.player_state['ticks_since_beat'] += 1
    
    async def execute_beat(self, beat: StoryBeat) -> str:
        """Execute a story beat."""
        
        # Mark as completed
        self.equilibrium.completed_beats.add(beat.beat_id)
        self.player_state['ticks_since_beat'] = 0
        
        # Set flags
        for flag in beat.sets_flags:
            self.player_state['flags'][flag] = True
        
        # Clear flags
        for flag in beat.clears_flags:
            self.player_state['flags'].pop(flag, None)
        
        # Spawn entities
        for entity_def in beat.spawns_entities:
            entity = Entity.from_dict(entity_def)
            self.world.register(entity)
        
        # Compose beat prose
        fragments = [
            self.world.get_fragment(fid) 
            for fid in beat.prose_fragments
        ]
        prose = self.compositor.compose(
            fragments, 
            beat, 
            self.player_state
        )
        
        return f"\n{'='*50}\n{prose}\n{'='*50}\n"
```

---

## Summary: The Equilibrium Philosophy

The key insight: **Stories have gravity.**

A well-constructed plot tree has **attractors** — beats that pull the narrative toward them. Players can resist, explore, derail — but the story gently, persistently guides them back.

| Derailment Level | What Happens |
|------------------|--------------|
| **On Track** | Nothing — let them play |
| **Slight** | Environmental hints appear |
| **Moderate** | NPCs mention relevant things |
| **Significant** | Events intrude on player actions |
| **Severe** | Plot adapts to player position |

The **recipe** controls how aggressive this is. A mystery wants tighter control (clues must be found). An adventure allows more wandering (the world IS the story).

And through it all, **Street View images become the locations**, grounding fantasy in recognizable reality. The novel's plot provides structure. The player provides chaos. The equilibrium engine balances them.
