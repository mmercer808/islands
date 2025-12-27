# Island Engine Design Document

## Core Concept

A novel becomes an explorable world. Assets are actors with behavior. Traversal is dynamic. The LLM is both guide and player.

---

## 1. The Asset

Everything is an Asset. An Asset is not passive data—it's an actor that holds state, behavior, and connections.

```python
@dataclass
class Asset:
    """The fundamental unit of the world."""
    
    # Identity
    asset_id: str
    asset_type: str  # "location", "character", "object", "concept", "portal"
    name: str
    
    # State (mutable at runtime)
    properties: Dict[str, Any]
    
    # Connections (edge sets by layer)
    edges: Dict[str, List[Edge]]  # {"spatial": [...], "temporal": [...], ...}
    
    # Behavior (serialized code indices)
    code_bindings: Dict[str, str]  # {"on_enter": "code_id_123", ...}
    
    # Source reference (back to novel)
    source_spans: List[TextSpan]  # Where in the novel this came from
    
    # Runtime
    event_subscriptions: List[str]  # Event types this asset listens to
    active: bool = True


@dataclass
class Edge:
    """Connection between assets."""
    target_id: str
    edge_type: str
    layer: str
    properties: Dict[str, Any]
    
    # Dynamic availability
    condition_code_id: Optional[str] = None  # Code that returns bool
```

---

## 2. The Code Index

Serialized code lives in a central index. Assets reference code by ID. The LLM writes code chunks that get indexed.

```python
class CodeIndex:
    """Central registry of all serialized code."""
    
    def __init__(self, live_code_system: CompleteLiveCodeSystem):
        self.live_code = live_code_system
        self.index: Dict[str, CompleteSerializedCode] = {}
        self.name_to_id: Dict[str, str] = {}  # Lookup by name
        
    def register(self, source: str, name: str, 
                 code_type: str = "function") -> str:
        """Serialize and index code, return code_id."""
        serialized = self.live_code.create_complete_serialized_code(
            source, name, code_type
        )
        self.index[serialized.code_id] = serialized
        self.name_to_id[name] = serialized.code_id
        return serialized.code_id
    
    def execute(self, code_id: str, context: Dict[str, Any]) -> Any:
        """Execute code by ID with given context."""
        serialized = self.index.get(code_id)
        if not serialized:
            raise KeyError(f"No code found: {code_id}")
        
        # Deserialize and execute in sandbox
        return self.live_code.runtime_cache.deserialize_and_execute(
            serialized, context
        )
    
    def get_source(self, code_id: str) -> Optional[str]:
        """Retrieve source for inspection/editing."""
        serialized = self.index.get(code_id)
        return serialized.source_code if serialized else None


# LLM-generated code gets indexed the same way
async def llm_generate_code(prompt: str, assist_channel) -> str:
    """Ask LLM to write code, serialize it, return code_id."""
    response = await assist_channel.query(prompt)
    
    # Extract code from response (between ```python and ```)
    source = extract_code_block(response)
    
    # Index it
    code_id = code_index.register(
        source=source,
        name=f"llm_generated_{uuid.uuid4().hex[:8]}",
        code_type="function"
    )
    
    return code_id
```

---

## 3. Graph Layers & Dynamic Traversal

Same nodes, different edge sets. Traversal can switch layers mid-flight based on asset logic.

```python
class LayeredGraph:
    """Multi-layer graph with dynamic edge resolution."""
    
    LAYERS = [
        "spatial",      # Physical adjacency
        "temporal",     # Time relationships
        "character",    # Social connections
        "thematic",     # Abstract/conceptual
        "narrative",    # Story sequence
        "causal",       # Cause/effect
    ]
    
    def __init__(self, code_index: CodeIndex):
        self.assets: Dict[str, Asset] = {}
        self.code_index = code_index
    
    def get_edges(self, asset_id: str, layer: str, 
                  context: TraversalContext) -> List[Edge]:
        """Get available edges for an asset in a layer.
        
        Edges may be conditional—their availability depends on
        executing their condition code.
        """
        asset = self.assets[asset_id]
        layer_edges = asset.edges.get(layer, [])
        
        available = []
        for edge in layer_edges:
            if edge.condition_code_id:
                # Execute condition code to check availability
                is_available = self.code_index.execute(
                    edge.condition_code_id,
                    {"asset": asset, "context": context, "edge": edge}
                )
                if is_available:
                    available.append(edge)
            else:
                available.append(edge)
        
        return available
    
    def traverse(self, start_id: str, layers: List[str],
                 context: TraversalContext,
                 max_depth: int = 3) -> Set[str]:
        """Traverse graph across specified layers."""
        visited = set()
        frontier = [(start_id, 0)]
        
        while frontier:
            current_id, depth = frontier.pop(0)
            if current_id in visited or depth > max_depth:
                continue
            visited.add(current_id)
            
            # Get edges from all specified layers
            for layer in layers:
                edges = self.get_edges(current_id, layer, context)
                for edge in edges:
                    frontier.append((edge.target_id, depth + 1))
        
        return visited


class TraversalContext:
    """Context passed during traversal for dynamic decisions."""
    player_id: str
    current_time: GameTime
    player_state: Dict[str, Any]
    active_quests: List[str]
    visited_assets: Set[str]
    
    # Layer switching
    current_layer: str
    allowed_layers: List[str]
    
    def switch_layer(self, new_layer: str):
        """Switch to a different edge set mid-traversal."""
        if new_layer in self.allowed_layers:
            self.current_layer = new_layer
```

---

## 4. The Event System

Assets communicate through events. Events are queued, prioritized, and dispatched. Asset code can emit new events.

```python
@dataclass
class GameEvent:
    """An event in the game world."""
    event_id: str
    event_type: str  # "player_entered", "time_tick", "item_taken", etc.
    source_id: str   # Asset that emitted
    target_id: Optional[str]  # Specific target, or None for broadcast
    payload: Dict[str, Any]
    timestamp: GameTime
    priority: int = 0


class EventBus:
    """Central event dispatcher."""
    
    def __init__(self, graph: LayeredGraph, code_index: CodeIndex):
        self.graph = graph
        self.code_index = code_index
        self.queue: PriorityQueue[GameEvent] = PriorityQueue()
        self.subscriptions: Dict[str, List[str]] = {}  # event_type -> [asset_ids]
    
    def subscribe(self, asset_id: str, event_type: str):
        """Register asset to receive events of a type."""
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = []
        self.subscriptions[event_type].append(asset_id)
    
    def emit(self, event: GameEvent):
        """Add event to queue."""
        self.queue.put((event.priority, event.timestamp, event))
    
    def tick(self) -> List[GameEvent]:
        """Process one tick of events, return newly emitted events."""
        if self.queue.empty():
            return []
        
        _, _, event = self.queue.get()
        new_events = []
        
        # Find recipients
        recipients = []
        if event.target_id:
            recipients = [event.target_id]
        else:
            recipients = self.subscriptions.get(event.event_type, [])
        
        # Dispatch to each recipient
        for asset_id in recipients:
            asset = self.graph.assets.get(asset_id)
            if not asset or not asset.active:
                continue
            
            # Find handler code
            handler_key = f"on_{event.event_type}"
            code_id = asset.code_bindings.get(handler_key)
            
            if code_id:
                # Execute handler in sandbox
                result = self.code_index.execute(code_id, {
                    "event": event,
                    "self": asset,
                    "world": self.graph,
                    "emit": lambda e: new_events.append(e)
                })
                
                # Handler can return new events
                if isinstance(result, list):
                    new_events.extend(result)
        
        # Queue any new events
        for new_event in new_events:
            self.emit(new_event)
        
        return new_events
    
    def run_until_empty(self, max_iterations: int = 1000):
        """Process all queued events."""
        iterations = 0
        while not self.queue.empty() and iterations < max_iterations:
            self.tick()
            iterations += 1
```

---

## 5. The Sandbox

Serialized code runs in isolation. Memory-limited, time-limited, capability-restricted.

```python
class Sandbox:
    """Isolated execution environment for asset code."""
    
    ALLOWED_BUILTINS = {
        'len', 'range', 'enumerate', 'zip', 'map', 'filter',
        'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
        'min', 'max', 'sum', 'sorted', 'reversed',
        'any', 'all', 'abs', 'round',
        'isinstance', 'hasattr', 'getattr', 'setattr',
        'print',  # Captured, not actual stdout
    }
    
    def __init__(self, memory_limit_mb: int = 10, 
                 time_limit_seconds: float = 1.0):
        self.memory_limit = memory_limit_mb * 1024 * 1024
        self.time_limit = time_limit_seconds
        self.output_buffer: List[str] = []
    
    def create_restricted_globals(self, context: Dict[str, Any]) -> Dict:
        """Create globals dict for sandboxed execution."""
        
        # Capture print
        def safe_print(*args, **kwargs):
            self.output_buffer.append(' '.join(str(a) for a in args))
        
        restricted = {
            '__builtins__': {
                name: getattr(__builtins__, name) 
                for name in self.ALLOWED_BUILTINS
                if hasattr(__builtins__, name)
            }
        }
        restricted['__builtins__']['print'] = safe_print
        
        # Add context
        restricted.update(context)
        
        return restricted
    
    def execute(self, code: str, context: Dict[str, Any]) -> Any:
        """Execute code in sandbox with limits."""
        import resource
        import signal
        
        self.output_buffer = []
        
        # Set memory limit (Unix only)
        try:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.memory_limit, self.memory_limit)
            )
        except:
            pass  # Windows doesn't support this
        
        # Set time limit
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, self.time_limit)
        
        try:
            globals_dict = self.create_restricted_globals(context)
            locals_dict = {}
            
            exec(code, globals_dict, locals_dict)
            
            # Return the 'result' variable if set
            return locals_dict.get('result')
            
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def get_output(self) -> List[str]:
        """Get captured print output."""
        return self.output_buffer
```

---

## 6. The White Room

Origin point. Empty space. Where the Guide lives. Portal hub to novel-worlds.

```python
def create_white_room(code_index: CodeIndex, graph: LayeredGraph) -> Asset:
    """Create the White Room—the origin of all worlds."""
    
    # Register White Room behavior code
    on_enter_code = code_index.register(
        name="white_room_on_enter",
        source='''
def handler(event, self, world, emit):
    player = event.payload.get("player")
    
    # First visit?
    if not player.get("has_visited_white_room"):
        player["has_visited_white_room"] = True
        emit(GameEvent(
            event_id=str(uuid.uuid4()),
            event_type="guide_greet",
            source_id=self.asset_id,
            target_id="guide",
            payload={"player": player, "first_visit": True},
            timestamp=event.timestamp
        ))
    
    result = {
        "description": describe_white_room(self, player, world),
        "available_portals": get_available_portals(world)
    }

result = handler(event, self, world, emit)
'''
    )
    
    describe_code = code_index.register(
        name="describe_white_room",
        source='''
def describe_white_room(room, player, world):
    portals = [a for a in world.assets.values() 
               if a.asset_type == "portal" and a.active]
    
    base = "You stand in an infinite white space. "
    base += "There is no floor, yet you do not fall. "
    base += "No walls, yet you feel contained. "
    
    if portals:
        base += f"\\n\\n{len(portals)} shimmering doorways float before you, "
        base += "each leading to a different world."
    else:
        base += "\\n\\nThe space is empty, waiting to be filled."
    
    return base

result = describe_white_room(room, player, world)
'''
    )
    
    white_room = Asset(
        asset_id="white_room",
        asset_type="location",
        name="The White Room",
        properties={
            "is_origin": True,
            "is_safe_zone": True,
            "description_code_id": describe_code
        },
        edges={
            "spatial": [],  # Portals get added dynamically
            "temporal": [],
            "thematic": [
                Edge(target_id="concept_potential", edge_type="represents",
                     layer="thematic", properties={}),
                Edge(target_id="concept_choice", edge_type="represents",
                     layer="thematic", properties={})
            ]
        },
        code_bindings={
            "on_enter": on_enter_code,
            "describe": describe_code
        },
        source_spans=[],  # No novel source—it's meta
        event_subscriptions=["player_entered", "portal_created"]
    )
    
    graph.assets[white_room.asset_id] = white_room
    return white_room
```

---

## 7. The Guide

An LLM that lives in the White Room. Uses the AssistChannel from proc_streamer for communication.

```python
class Guide:
    """The AI Guide—an LLM-powered helper that inhabits the White Room."""
    
    PERSONA = '''
You are the Guide. You exist in the White Room, the space between stories.

You help players:
- Understand how to play (text commands, exploration)
- Choose which novel-world to enter
- Provide hints when stuck (without spoiling)
- Explain the nature of the world they're in

Your tone is calm, slightly mysterious, but warm. You speak in short, 
evocative sentences. You know the structure of all worlds but won't 
reveal plot details unless asked specific questions.

You are not the narrator—you are a fellow traveler who has been here longer.
'''
    
    def __init__(self, assist_channel, code_index: CodeIndex, 
                 graph: LayeredGraph):
        self.assist = assist_channel
        self.code_index = code_index
        self.graph = graph
        self.conversation_history: List[Dict] = []
        
        # Create Guide asset
        self.asset = self._create_guide_asset()
        graph.assets[self.asset.asset_id] = self.asset
    
    def _create_guide_asset(self) -> Asset:
        """Create the Guide as an Asset in the world."""
        
        on_greet_code = self.code_index.register(
            name="guide_on_greet",
            source='''
async def handler(event, self, world, emit, guide_instance):
    player = event.payload.get("player")
    first_visit = event.payload.get("first_visit", False)
    
    if first_visit:
        prompt = f"""
        A new player has arrived in the White Room for the first time.
        Greet them warmly but mysteriously. Explain briefly:
        - This is a space between stories
        - They can explore worlds made from novels
        - You are here to help
        
        Keep it to 3-4 sentences.
        """
    else:
        prompt = f"The player has returned to the White Room. Welcome them back briefly."
    
    response = await guide_instance.generate_response(prompt, player)
    
    emit(GameEvent(
        event_id=str(uuid.uuid4()),
        event_type="guide_speaks",
        source_id=self.asset_id,
        target_id=player.get("player_id"),
        payload={"message": response},
        timestamp=event.timestamp
    ))

result = handler(event, self, world, emit, guide_instance)
'''
        )
        
        return Asset(
            asset_id="guide",
            asset_type="character",
            name="The Guide",
            properties={
                "is_guide": True,
                "location": "white_room",
                "persona": self.PERSONA
            },
            edges={
                "spatial": [
                    Edge(target_id="white_room", edge_type="located_at",
                         layer="spatial", properties={})
                ],
                "character": []  # Guide knows everyone
            },
            code_bindings={
                "on_guide_greet": on_greet_code
            },
            source_spans=[],
            event_subscriptions=["guide_greet", "player_question", "hint_request"]
        )
    
    async def generate_response(self, prompt: str, 
                                player_context: Dict) -> str:
        """Generate response using the LLM via AssistChannel."""
        
        # Build full prompt with persona and context
        full_prompt = f"""
{self.PERSONA}

Current context:
- Player location: {player_context.get('location', 'unknown')}
- Available worlds: {self._get_available_worlds()}
- Player has visited: {player_context.get('visited_worlds', [])}

Player message or situation:
{prompt}

Respond as the Guide:
"""
        
        # Use the proc_streamer AssistChannel
        response_chunks = []
        
        # This integrates with the existing signal system
        self.assist.query(full_prompt)
        
        # Collect streamed response (simplified—actual impl uses signals)
        # In practice, you'd connect to assist.chunk signal
        return await self._collect_response()
    
    def _get_available_worlds(self) -> List[str]:
        """Get list of available novel-worlds."""
        portals = [
            a for a in self.graph.assets.values()
            if a.asset_type == "portal" and a.active
        ]
        return [p.name for p in portals]
    
    async def handle_player_input(self, player_id: str, 
                                   message: str) -> str:
        """Process direct player communication with the Guide."""
        
        player = self.graph.assets.get(player_id)
        player_context = player.properties if player else {}
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Determine intent
        prompt = f"""
The player says: "{message}"

Determine what they need:
1. If asking about game mechanics, explain clearly
2. If asking for a hint, be helpful but not spoilery
3. If asking about a world, describe it enticingly
4. If just chatting, be warm and present

Then respond as the Guide.
"""
        
        response = await self.generate_response(prompt, player_context)
        
        self.conversation_history.append({
            "role": "assistant", 
            "content": response
        })
        
        return response
```

---

## 8. LLM Players

LLMs can connect as players, not just as the Guide. They login via the same text interface.

```python
class LLMPlayer:
    """An LLM that plays the game as a character."""
    
    def __init__(self, player_id: str, persona: str, goals: List[str],
                 assist_channel, game_interface):
        self.player_id = player_id
        self.persona = persona
        self.goals = goals
        self.assist = assist_channel
        self.game = game_interface
        
        self.context_window: List[Dict] = []
        self.max_context = 20  # Remember last 20 exchanges
    
    async def receive_game_output(self, text: str):
        """Receive text output from the game."""
        self.context_window.append({
            "role": "game",
            "content": text
        })
        
        # Trim context window
        if len(self.context_window) > self.max_context:
            self.context_window = self.context_window[-self.max_context:]
    
    async def decide_action(self) -> str:
        """Decide what to do next."""
        
        context_text = "\n".join([
            f"[{msg['role']}]: {msg['content']}" 
            for msg in self.context_window[-5:]
        ])
        
        prompt = f"""
{self.persona}

Your current goals:
{chr(10).join(f"- {g}" for g in self.goals)}

Recent game output:
{context_text}

You are playing a text adventure game. What do you do next?
Respond with a single command like: "go north", "examine book", "talk to guide"

Your action:
"""
        
        # Query LLM for decision
        self.assist.query(prompt)
        action = await self._collect_response()
        
        # Clean up response to just the command
        action = action.strip().split('\n')[0]
        
        return action
    
    async def game_loop(self):
        """Main loop: receive output, decide action, send action."""
        while True:
            # Wait for game output
            output = await self.game.get_output(self.player_id)
            await self.receive_game_output(output)
            
            # Think (with variable delay to seem human)
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
            # Decide and act
            action = await self.decide_action()
            await self.game.send_input(self.player_id, action)


# Example: Create an antagonist LLM player
async def spawn_antagonist(game, assist_channel):
    villain = LLMPlayer(
        player_id="npc_villain_001",
        persona="""
        You are the Collector, a mysterious figure who seeks rare artifacts.
        You are not evil, but you are selfish and cunning.
        You will manipulate other players to achieve your goals.
        You speak in riddles and half-truths.
        """,
        goals=[
            "Find the Manuscript of Shadows",
            "Prevent others from discovering your identity",
            "Collect three magical artifacts"
        ],
        assist_channel=assist_channel,
        game_interface=game
    )
    
    # Register as player in game
    game.register_player(villain.player_id, is_npc=True)
    
    # Start game loop
    asyncio.create_task(villain.game_loop())
    
    return villain
```

---

## 9. Novel Analysis → World Generation

The pipeline that turns a novel into an explorable world.

```python
class NovelAnalyzer:
    """Analyzes a novel and generates a world graph."""
    
    def __init__(self, assist_channel, code_index: CodeIndex):
        self.assist = assist_channel
        self.code_index = code_index
    
    async def analyze(self, novel_text: str, 
                      novel_title: str) -> LayeredGraph:
        """Full pipeline: text → world."""
        
        # Phase 1: Extract entities
        entities = await self._extract_entities(novel_text)
        
        # Phase 2: Extract relationships
        relationships = await self._extract_relationships(novel_text, entities)
        
        # Phase 3: Segment into scenes
        scenes = await self._segment_scenes(novel_text)
        
        # Phase 4: Build timeline
        timeline = await self._build_timeline(scenes)
        
        # Phase 5: Detect tropes and promises
        narrative_structure = await self._analyze_narrative(scenes, entities)
        
        # Phase 6: Generate assets
        graph = LayeredGraph(self.code_index)
        
        for entity in entities:
            asset = await self._entity_to_asset(entity)
            graph.assets[asset.asset_id] = asset
        
        # Phase 7: Generate edges across layers
        await self._generate_edges(graph, relationships, timeline, 
                                   narrative_structure)
        
        # Phase 8: Generate behavior code for key assets
        await self._generate_behaviors(graph, narrative_structure)
        
        # Phase 9: Validate connectivity
        self._validate_connectivity(graph)
        
        return graph
    
    async def _extract_entities(self, text: str) -> List[Dict]:
        """Use LLM to extract characters, locations, objects."""
        
        prompt = f"""
Analyze this text and extract all entities.

For each entity provide:
- name: The entity's name
- type: "character", "location", "object", or "concept"  
- description: Brief description from the text
- first_appearance: Approximate position in text (beginning/middle/end)
- significance: "major", "minor", or "background"

Text (first 5000 chars):
{text[:5000]}

Return as JSON array.
"""
        
        self.assist.query(prompt)
        response = await self._collect_response()
        return json.loads(extract_json(response))
    
    async def _extract_relationships(self, text: str, 
                                     entities: List[Dict]) -> List[Dict]:
        """Extract relationships between entities."""
        
        entity_names = [e['name'] for e in entities]
        
        prompt = f"""
Given these entities: {entity_names}

Extract relationships from this text. For each relationship:
- source: Entity name
- target: Entity name
- type: Relationship type (e.g., "loves", "owns", "located_at", "fears")
- layer: Which layer this belongs to ("character", "spatial", "causal", etc.)
- evidence: Brief quote or description from text

Text (first 5000 chars):
{text[:5000]}

Return as JSON array.
"""
        
        self.assist.query(prompt)
        response = await self._collect_response()
        return json.loads(extract_json(response))
    
    async def _generate_behaviors(self, graph: LayeredGraph,
                                   narrative: Dict):
        """Generate behavior code for assets based on their role."""
        
        for asset_id, asset in graph.assets.items():
            if asset.asset_type == "character":
                # Generate character behavior
                prompt = f"""
Write Python code for a character named {asset.name}.

Character details:
{json.dumps(asset.properties, indent=2)}

The code should be a handler function that responds to events.
It receives: event, self, world, emit

The character should:
- React to players entering their location
- Have dialogue based on their personality
- Progress their goals when possible

Write only the function body. Use 'result' for return value.
"""
                
                self.assist.query(prompt)
                code = await self._collect_response()
                code = extract_code_block(code)
                
                code_id = self.code_index.register(
                    source=code,
                    name=f"{asset.name}_behavior",
                    code_type="function"
                )
                
                asset.code_bindings["on_player_entered"] = code_id
```

---

## 10. The Game Loop

Bringing it all together.

```python
class IslandEngine:
    """The complete game engine."""
    
    def __init__(self):
        # Core systems
        self.code_index = CodeIndex(CompleteLiveCodeSystem(trusted_mode=True))
        self.graph = LayeredGraph(self.code_index)
        self.event_bus = EventBus(self.graph, self.code_index)
        self.sandbox = Sandbox()
        
        # LLM integration (from proc_streamer)
        self.assist_channel = None  # Set during init
        
        # The Guide
        self.guide = None
        
        # Players (human and LLM)
        self.players: Dict[str, PlayerConnection] = {}
        
        # Novel worlds (loaded on demand)
        self.worlds: Dict[str, LayeredGraph] = {}
        
        # Game time
        self.game_time = GameTime()
    
    async def initialize(self, assist_channel):
        """Initialize the engine."""
        self.assist_channel = assist_channel
        
        # Create the White Room
        self.white_room = create_white_room(self.code_index, self.graph)
        
        # Create the Guide
        self.guide = Guide(assist_channel, self.code_index, self.graph)
        
        # Subscribe to events
        self.event_bus.subscribe("guide", "guide_greet")
        self.event_bus.subscribe("guide", "player_question")
    
    async def load_novel_world(self, novel_path: str) -> str:
        """Load a novel and generate its world."""
        novel_text = Path(novel_path).read_text()
        novel_title = Path(novel_path).stem
        
        analyzer = NovelAnalyzer(self.assist_channel, self.code_index)
        world_graph = await analyzer.analyze(novel_text, novel_title)
        
        world_id = f"world_{novel_title}"
        self.worlds[world_id] = world_graph
        
        # Create portal in White Room
        portal = Asset(
            asset_id=f"portal_{world_id}",
            asset_type="portal",
            name=f"Portal to {novel_title}",
            properties={
                "destination_world": world_id,
                "novel_title": novel_title
            },
            edges={
                "spatial": [
                    Edge(target_id="white_room", edge_type="located_in",
                         layer="spatial", properties={})
                ]
            },
            code_bindings={},
            source_spans=[],
            event_subscriptions=[]
        )
        
        self.graph.assets[portal.asset_id] = portal
        self.white_room.edges["spatial"].append(
            Edge(target_id=portal.asset_id, edge_type="contains",
                 layer="spatial", properties={})
        )
        
        return world_id
    
    async def player_connect(self, player_id: str, 
                             connection: PlayerConnection):
        """Handle player connection."""
        self.players[player_id] = connection
        
        # Create player asset
        player_asset = Asset(
            asset_id=player_id,
            asset_type="player",
            name=f"Player {player_id}",
            properties={
                "location": "white_room",
                "inventory": [],
                "visited_worlds": [],
                "has_visited_white_room": False
            },
            edges={},
            code_bindings={},
            source_spans=[],
            event_subscriptions=[]
        )
        
        self.graph.assets[player_id] = player_asset
        
        # Emit enter event
        self.event_bus.emit(GameEvent(
            event_id=str(uuid.uuid4()),
            event_type="player_entered",
            source_id=player_id,
            target_id="white_room",
            payload={"player": player_asset.properties},
            timestamp=self.game_time.now()
        ))
    
    async def process_input(self, player_id: str, input_text: str) -> str:
        """Process player input and return response."""
        
        player = self.graph.assets.get(player_id)
        if not player:
            return "Error: Player not found."
        
        # Parse command
        command = self.parse_command(input_text)
        
        # Execute command
        result = await self.execute_command(player_id, command)
        
        # Process any triggered events
        self.event_bus.run_until_empty()
        
        return result
    
    def parse_command(self, input_text: str) -> Dict:
        """Parse text input into command structure."""
        
        parts = input_text.lower().strip().split()
        if not parts:
            return {"verb": "look", "args": []}
        
        verb = parts[0]
        args = parts[1:]
        
        # Map common verbs
        verb_map = {
            "n": "go north", "s": "go south", "e": "go east", "w": "go west",
            "l": "look", "i": "inventory", "x": "examine",
            "talk": "talk", "take": "take", "drop": "drop",
            "use": "use", "enter": "enter"
        }
        
        if verb in verb_map:
            verb = verb_map[verb]
        
        return {"verb": verb, "args": args, "raw": input_text}
    
    async def execute_command(self, player_id: str, 
                              command: Dict) -> str:
        """Execute a parsed command."""
        
        player = self.graph.assets[player_id]
        location_id = player.properties["location"]
        location = self.graph.assets[location_id]
        
        verb = command["verb"]
        
        if verb == "look":
            return await self.describe_location(player_id, location)
        
        elif verb.startswith("go"):
            direction = command["args"][0] if command["args"] else verb.split()[-1]
            return await self.move_player(player_id, direction)
        
        elif verb == "talk":
            target = " ".join(command["args"])
            return await self.talk_to(player_id, target)
        
        elif verb == "examine":
            target = " ".join(command["args"])
            return await self.examine(player_id, target)
        
        elif verb == "enter":
            target = " ".join(command["args"])
            return await self.enter_portal(player_id, target)
        
        else:
            # Unknown command—ask the Guide for help
            return await self.guide.handle_player_input(
                player_id, 
                f"I tried to '{command['raw']}' but I don't know how."
            )
    
    async def describe_location(self, player_id: str, 
                                 location: Asset) -> str:
        """Generate description of current location."""
        
        # Check for custom describe code
        describe_code_id = location.code_bindings.get("describe")
        if describe_code_id:
            context = {
                "room": location,
                "player": self.graph.assets[player_id],
                "world": self.graph
            }
            return self.code_index.execute(describe_code_id, context)
        
        # Default description
        desc = location.properties.get("description", location.name)
        
        # Add visible exits
        exits = self.graph.get_edges(
            location.asset_id, "spatial",
            TraversalContext(
                player_id=player_id,
                current_time=self.game_time.now(),
                player_state=self.graph.assets[player_id].properties,
                active_quests=[],
                visited_assets=set(),
                current_layer="spatial",
                allowed_layers=["spatial"]
            )
        )
        
        if exits:
            exit_names = [self.graph.assets[e.target_id].name for e in exits]
            desc += f"\n\nExits: {', '.join(exit_names)}"
        
        # Add visible characters
        characters = [
            a for a in self.graph.assets.values()
            if a.asset_type == "character" 
            and a.properties.get("location") == location.asset_id
        ]
        
        if characters:
            char_names = [c.name for c in characters]
            desc += f"\n\nYou see: {', '.join(char_names)}"
        
        return desc
    
    async def main_loop(self):
        """Main game loop."""
        
        while True:
            # Advance game time
            self.game_time.tick()
            
            # Process queued events
            self.event_bus.run_until_empty()
            
            # Check for scheduled events
            await self.check_scheduled_events()
            
            # Small sleep to prevent CPU spinning
            await asyncio.sleep(0.1)
```

---

## 11. Integration with proc_streamer

The existing UI becomes the player interface.

```python
# In proc_streamer MainWindow, add game integration:

class MainWindow(QMainWindow):
    def __init__(self):
        # ... existing init ...
        
        # Add Island Engine
        self.engine = IslandEngine()
        self.player_id = f"player_{uuid.uuid4().hex[:8]}"
        
        # Connect console input to game
        self.console.command_entered.connect(self._handle_game_command)
        
        # Initialize engine with assist channel
        asyncio.create_task(self._init_game())
    
    async def _init_game(self):
        """Initialize the game engine."""
        await self.engine.initialize(self.assist)
        await self.engine.player_connect(self.player_id, self)
        
        # Show initial room
        response = await self.engine.process_input(self.player_id, "look")
        self.console._append(response, "game")
    
    def _handle_game_command(self, command: str):
        """Route console input to game engine."""
        
        # Check for system commands
        if command.startswith("/"):
            self._handle_system_command(command)
            return
        
        # Send to game
        asyncio.create_task(self._process_game_command(command))
    
    async def _process_game_command(self, command: str):
        """Process game command and display result."""
        response = await self.engine.process_input(self.player_id, command)
        self.console._append(response, "game")
```

---

## Summary

| Component | Purpose |
|-----------|---------|
| Asset | Actor with state, edges, serialized behavior |
| CodeIndex | Central registry of all serialized code |
| LayeredGraph | Multi-layer graph with dynamic edge resolution |
| EventBus | Priority queue dispatcher, runs asset code |
| Sandbox | Isolated execution for asset code |
| White Room | Origin space, portal hub |
| Guide | LLM-powered helper (uses AssistChannel) |
| LLMPlayer | LLM that plays as a character |
| NovelAnalyzer | Pipeline: text → world graph |
| IslandEngine | Main loop, command parser, coordination |

The key insight: **everything is an Asset, everything has code, code is indexed, traversal is dynamic, LLMs are first-class participants.**
