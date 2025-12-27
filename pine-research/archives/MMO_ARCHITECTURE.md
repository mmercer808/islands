# MMO Text Adventure Architecture

## The Model: Prompt → Text Generation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GAME OUTPUT (Text Generation)                     │
│                                                                          │
│  You stand at the Harbor of Vetrellis. The morning mist curls around    │
│  weathered dock posts. Three fishing boats bob gently. To the north,    │
│  cobblestone streets climb toward the village square.                    │
│                                                                          │
│  [Archon_7cdba906 waves from the lighthouse tower]                      │
│  [TideKeeper is tending the pools in the Northern Cove]                 │
│  [2 other travelers are nearby]                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────┐
│ > _                                                            [PROMPT] │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Proposed File Structure

```
vetrellis-mmo/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── game/                              # ═══ PURE GAME ENGINE ═══
│   │                                  # No network, no UI, no async
│   │                                  # Can run entirely in-memory
│   │
│   ├── __init__.py
│   │
│   ├── core/                          # Foundation layer
│   │   ├── __init__.py
│   │   ├── primitives.py              # Enums, protocols, type definitions
│   │   │                              # EntityType, ActionType, SignalType
│   │   ├── entity.py                  # Entity base class
│   │   │                              # Entity, EntityState, EntityFlags
│   │   ├── graph.py                   # World graph structure
│   │   │                              # WorldGraph, Node, Edge, EdgeType
│   │   └── identity.py                # ID generation, signatures
│   │                                  # PlayerID, EntityID, SessionID
│   │
│   ├── world/                         # World state and structure
│   │   ├── __init__.py
│   │   ├── location.py                # Location entities
│   │   │                              # Location, Room, Area, Zone
│   │   ├── item.py                    # Item entities
│   │   │                              # Item, Container, Equipment
│   │   ├── character.py               # Character entities
│   │   │                              # NPC, Mob, Keeper
│   │   ├── portal.py                  # Connections between locations
│   │   │                              # Portal, Door, Path
│   │   ├── spawner.py                 # Entity spawning rules
│   │   │                              # Spawner, SpawnTable, SpawnTimer
│   │   └── instance.py                # Instanced areas
│   │                                  # Instance, InstanceTemplate
│   │
│   ├── player/                        # Player-specific systems
│   │   ├── __init__.py
│   │   ├── avatar.py                  # Player character
│   │   │                              # Avatar, Inventory, Equipment
│   │   ├── stats.py                   # Player statistics
│   │   │                              # Stats, Skills, Attributes
│   │   ├── progress.py                # Progression systems
│   │   │                              # Level, XP, Achievements
│   │   └── session.py                 # Player session state
│   │                                  # Session, SessionData, Presence
│   │
│   ├── action/                        # Action processing
│   │   ├── __init__.py
│   │   ├── parser.py                  # Input parsing
│   │   │                              # ActionParser, Command, Intent
│   │   ├── processor.py               # Action execution
│   │   │                              # ActionProcessor, ActionResult
│   │   ├── verbs.py                   # Verb definitions
│   │   │                              # VerbRegistry, Verb, VerbHandler
│   │   ├── scope.py                   # What player can interact with
│   │   │                              # ScopeResolver, Visibility
│   │   └── validation.py              # Action validation
│   │                                  # Validator, ConditionChecker
│   │
│   ├── prose/                         # Text generation
│   │   ├── __init__.py
│   │   ├── fragment.py                # Prose fragments
│   │   │                              # ProseFragment, FragmentCondition
│   │   ├── compositor.py              # Prose assembly
│   │   │                              # ProseCompositor, OutputBuffer
│   │   ├── templates.py               # Text templates
│   │   │                              # Template, TemplateEngine
│   │   └── formatter.py               # Output formatting
│   │                                  # Formatter, ColorCode, Style
│   │
│   ├── narrative/                     # Story and quests
│   │   ├── __init__.py
│   │   ├── quest.py                   # Quest system
│   │   │                              # Quest, QuestStep, QuestState
│   │   ├── dialogue.py                # NPC dialogue
│   │   │                              # Dialogue, DialogueNode, Choice
│   │   ├── event.py                   # World events
│   │   │                              # WorldEvent, EventTrigger
│   │   └── lore.py                    # Discoverable lore
│   │                                  # LoreEntry, LoreCollection
│   │
│   ├── combat/                        # Combat systems
│   │   ├── __init__.py
│   │   ├── engine.py                  # Combat resolution
│   │   │                              # CombatEngine, CombatState
│   │   ├── damage.py                  # Damage calculation
│   │   │                              # DamageType, DamageCalc
│   │   ├── ability.py                 # Abilities/skills
│   │   │                              # Ability, Cooldown, Effect
│   │   └── aggro.py                   # Threat/aggro system
│   │                                  # ThreatTable, Aggro
│   │
│   ├── economy/                       # Economic systems
│   │   ├── __init__.py
│   │   ├── currency.py                # Currency types
│   │   │                              # Currency, Wallet
│   │   ├── trade.py                   # Player trading
│   │   │                              # Trade, TradeOffer
│   │   ├── vendor.py                  # NPC shops
│   │   │                              # Vendor, Inventory, Price
│   │   └── auction.py                 # Auction house
│   │                                  # Auction, Bid, Listing
│   │
│   ├── social/                        # Social systems
│   │   ├── __init__.py
│   │   ├── guild.py                   # Guilds/clans
│   │   │                              # Guild, GuildRank, GuildBank
│   │   ├── party.py                   # Player groups
│   │   │                              # Party, PartyRole
│   │   ├── friends.py                 # Friends list
│   │   │                              # FriendList, Block
│   │   └── reputation.py              # Faction reputation
│   │                                  # Faction, Reputation, Standing
│   │
│   ├── ai/                            # NPC AI
│   │   ├── __init__.py
│   │   ├── behavior.py                # Behavior trees
│   │   │                              # BehaviorTree, BehaviorNode
│   │   ├── pathfinding.py             # Movement AI
│   │   │                              # Pathfinder, Route
│   │   ├── keeper.py                  # Keeper AI (special NPCs)
│   │   │                              # KeeperAI, KeeperPersonality
│   │   └── schedule.py                # NPC schedules
│   │                                  # Schedule, Activity
│   │
│   ├── time/                          # Game time systems
│   │   ├── __init__.py
│   │   ├── clock.py                   # World clock
│   │   │                              # WorldClock, TimeOfDay, Season
│   │   ├── tick.py                    # Game tick processing
│   │   │                              # TickProcessor, TickEvent
│   │   └── calendar.py                # In-game calendar
│   │                                  # Calendar, Holiday, Event
│   │
│   └── rules/                         # Game rules engine
│       ├── __init__.py
│       ├── engine.py                  # Rules evaluation
│       │                              # RulesEngine, Rule, Condition
│       ├── effects.py                 # Effect application
│       │                              # Effect, EffectStack
│       └── triggers.py                # Trigger system
│                                      # Trigger, TriggerEvent
│
├── server/                            # ═══ SERVER INFRASTRUCTURE ═══
│   │                                  # Network, persistence, scaling
│   │
│   ├── __init__.py
│   │
│   ├── gateway/                       # Client connections
│   │   ├── __init__.py
│   │   ├── websocket.py               # WebSocket handler
│   │   │                              # WSHandler, Connection
│   │   ├── protocol.py                # Message protocol
│   │   │                              # Message, MessageType, Codec
│   │   ├── session.py                 # Session management
│   │   │                              # SessionManager, Token
│   │   └── rate_limit.py              # Rate limiting
│   │                                  # RateLimiter, Throttle
│   │
│   ├── auth/                          # Authentication
│   │   ├── __init__.py
│   │   ├── login.py                   # Login flow
│   │   │                              # LoginHandler, Credentials
│   │   ├── token.py                   # JWT/session tokens
│   │   │                              # TokenManager, Claims
│   │   ├── password.py                # Password hashing
│   │   │                              # PasswordHash, Verify
│   │   └── oauth.py                   # OAuth providers
│   │                                  # OAuthProvider, Callback
│   │
│   ├── persistence/                   # Data storage
│   │   ├── __init__.py
│   │   ├── database.py                # Database connection
│   │   │                              # Database, Connection, Pool
│   │   ├── models.py                  # ORM models
│   │   │                              # PlayerModel, WorldModel
│   │   ├── cache.py                   # Redis/memory cache
│   │   │                              # Cache, CacheKey
│   │   ├── snapshot.py                # World state snapshots
│   │   │                              # Snapshot, SnapshotManager
│   │   └── chronicle.py               # Event sourcing / audit log
│   │                                  # Chronicle, ChronicleEntry
│   │
│   ├── cluster/                       # Multi-server scaling
│   │   ├── __init__.py
│   │   ├── coordinator.py             # Cluster coordination
│   │   │                              # ClusterCoordinator, Node
│   │   ├── sharding.py                # World sharding
│   │   │                              # ShardManager, ShardKey
│   │   ├── pubsub.py                  # Inter-server messaging
│   │   │                              # PubSub, Channel, Message
│   │   └── handoff.py                 # Player handoff between servers
│   │                                  # Handoff, Migration
│   │
│   ├── world_host/                    # World simulation server
│   │   ├── __init__.py
│   │   ├── host.py                    # World host process
│   │   │                              # WorldHost, WorldState
│   │   ├── tick_loop.py               # Main game loop
│   │   │                              # TickLoop, TickScheduler
│   │   ├── broadcast.py               # Message broadcasting
│   │   │                              # Broadcaster, Subscription
│   │   └── zone.py                    # Zone management
│   │                                  # ZoneManager, ZoneState
│   │
│   └── api/                           # External APIs
│       ├── __init__.py
│       ├── rest.py                    # REST API
│       │                              # Router, Endpoint
│       ├── admin.py                   # Admin endpoints
│       │                              # AdminAPI, Moderation
│       └── metrics.py                 # Observability
│                                      # Metrics, Prometheus
│
├── client/                            # ═══ CLIENT INTERFACES ═══
│   │
│   ├── __init__.py
│   │
│   ├── terminal/                      # Terminal/CLI client
│   │   ├── __init__.py
│   │   ├── app.py                     # Main application
│   │   │                              # TerminalApp, GameLoop
│   │   ├── input.py                   # Input handling
│   │   │                              # InputHandler, History
│   │   ├── output.py                  # Output rendering
│   │   │                              # OutputRenderer, ANSIStyle
│   │   └── connection.py              # Server connection
│   │                                  # Connection, Reconnect
│   │
│   ├── web/                           # Web client
│   │   ├── index.html
│   │   ├── styles/
│   │   │   └── game.css
│   │   └── js/
│   │       ├── app.js                 # Main application
│   │       ├── websocket.js           # WebSocket client
│   │       ├── input.js               # Prompt handling
│   │       └── output.js              # Text rendering
│   │
│   └── pyside/                        # Desktop client (PySide6)
│       ├── __init__.py
│       ├── app.py                     # Qt application
│       ├── main_window.py             # Main window
│       ├── game_output.py             # Text output widget
│       ├── prompt_input.py            # Command input
│       └── sidebar.py                 # Player info, map, etc.
│
├── content/                           # ═══ GAME CONTENT ═══
│   │                                  # World definitions, not code
│   │
│   ├── worlds/
│   │   └── vetrellis/                 # Vetrellis Isle
│   │       ├── manifest.yaml          # World manifest
│   │       ├── locations/
│   │       │   ├── harbor.yaml
│   │       │   ├── village.yaml
│   │       │   ├── lighthouse.yaml
│   │       │   ├── forest.yaml
│   │       │   └── echo_chamber.yaml
│   │       ├── characters/
│   │       │   ├── archon.yaml
│   │       │   ├── stone_who_counts.yaml
│   │       │   └── tide_keeper.yaml
│   │       ├── items/
│   │       │   ├── key_of_meridian.yaml
│   │       │   └── inserts.yaml
│   │       ├── quests/
│   │       │   ├── the_signal.yaml
│   │       │   └── keepers_wisdom.yaml
│   │       └── prose/
│   │           ├── descriptions.yaml
│   │           └── dialogue.yaml
│   │
│   ├── templates/                     # Prose templates
│   │   ├── location_enter.txt
│   │   ├── item_examine.txt
│   │   └── combat_attack.txt
│   │
│   └── localization/                  # i18n
│       ├── en/
│       └── es/
│
├── shared/                            # ═══ SHARED CODE ═══
│   │                                  # Used by both game and server
│   │
│   ├── __init__.py
│   ├── signals.py                     # Observer pattern
│   │                                  # SignalBus, Observer, Event
│   ├── serialization.py               # Serialization utilities
│   │                                  # Serializer, Codec
│   ├── validation.py                  # Input validation
│   │                                  # Validator, Schema
│   └── constants.py                   # Shared constants
│                                      # Limits, Defaults
│
├── tools/                             # ═══ DEVELOPMENT TOOLS ═══
│   │
│   ├── world_editor/                  # Content editor
│   │   ├── __init__.py
│   │   └── editor.py
│   │
│   ├── content_validator/             # Validate YAML content
│   │   ├── __init__.py
│   │   └── validator.py
│   │
│   ├── simulation/                    # Offline simulation
│   │   ├── __init__.py
│   │   └── simulator.py
│   │
│   └── migration/                     # Database migrations
│       └── versions/
│
├── tests/
│   ├── game/
│   ├── server/
│   ├── client/
│   └── integration/
│
├── deploy/
│   ├── docker/
│   │   ├── Dockerfile.server
│   │   ├── Dockerfile.worldhost
│   │   └── docker-compose.yml
│   ├── kubernetes/
│   │   ├── gateway.yaml
│   │   ├── worldhost.yaml
│   │   └── database.yaml
│   └── scripts/
│       ├── deploy.sh
│       └── backup.sh
│
└── docs/
    ├── architecture.md
    ├── protocol.md                    # Client-server protocol spec
    ├── content_format.md              # YAML content format
    └── api/
        └── openapi.yaml
```

---

## Key Architecture Decisions

### 1. Pure Game Engine (`game/`)

The game engine has **zero dependencies** on:
- Network code (no sockets, no async I/O)
- Database (operates on in-memory state)
- UI frameworks (no Qt, no HTML)

This allows:
- Running the game in pure Python for testing
- Embedding in any interface (web, terminal, Discord bot)
- Unit testing without mocking network

```python
# Example: Pure game engine usage
from game.core.graph import WorldGraph
from game.action.processor import ActionProcessor
from game.prose.compositor import ProseCompositor

# Create world (could be loaded from YAML)
world = WorldGraph()
processor = ActionProcessor(world)
compositor = ProseCompositor()

# Process a command
result = processor.process("look north", player_id="abc123")
output = compositor.compose(result)
print(output)  # Pure text, no network
```

### 2. Session Model

Each player has a session that travels with them:

```python
@dataclass
class PlayerSession:
    player_id: str
    avatar: Avatar              # Current character
    location_id: str            # Where they are
    instance_id: Optional[str]  # If in instanced content
    party_id: Optional[str]     # If in a group
    
    # Input/output buffers
    input_queue: Queue[str]     # Pending commands
    output_buffer: List[str]    # Pending output
    
    # Real-time state
    last_activity: float
    connection_id: str
```

### 3. World Tick Model

The world runs on a tick-based simulation:

```python
class WorldHost:
    def tick(self):
        """Called every 100ms (10 ticks/second)"""
        
        # 1. Process player inputs
        for session in self.sessions:
            while not session.input_queue.empty():
                self.process_input(session)
        
        # 2. Update AI
        for npc in self.active_npcs:
            npc.behavior.tick()
        
        # 3. Update combat
        for combat in self.active_combats:
            combat.tick()
        
        # 4. Update world events
        self.event_scheduler.tick()
        
        # 5. Broadcast updates to players
        self.broadcast_updates()
```

### 4. Message Protocol

Client ↔ Server communication:

```yaml
# Client → Server (Input)
{
  "type": "command",
  "payload": {
    "text": "take sword",
    "timestamp": 1702584000
  }
}

# Server → Client (Output)
{
  "type": "prose",
  "payload": {
    "lines": [
      "You pick up the ancient sword.",
      "It feels heavy in your hands."
    ],
    "style": "action"
  }
}

# Server → Client (World Update)
{
  "type": "presence",
  "payload": {
    "entered": ["player_xyz"],
    "left": ["player_123"],
    "location": "harbor"
  }
}
```

### 5. Content as YAML

World content is defined in YAML, not Python:

```yaml
# content/worlds/vetrellis/locations/lighthouse.yaml
id: lighthouse
name: The Lighthouse
zone: northern_coast

description:
  base: |
    The lighthouse stands against the sky, its white walls 
    weathered by salt and years. A spiral staircase winds 
    upward through the tower.
  
  conditions:
    - if: { flag: met_archon }
      add: "Light pulses from the top—Archon is awake."
    - if: { time: night }
      add: "The beacon sweeps across the dark waters."

exits:
  - direction: south
    to: cliffs
    description: "Stone steps descend toward the cliffs."
  - direction: up
    to: lighthouse_top
    requires: { key: lighthouse_key }
    locked_message: "The door to the stairs is locked."

items:
  - id: old_logbook
    spawn_chance: 1.0

npcs:
  - id: archon
    spawn_schedule:
      - { from: "06:00", to: "22:00", location: lighthouse_top }
      - { from: "22:00", to: "06:00", location: lighthouse }
```

---

## Scaling for MMO

### Single Server (< 500 players)
```
┌─────────────────┐
│  Gateway + API  │
│  + World Host   │
│  (single proc)  │
└────────┬────────┘
         │
    ┌────▼────┐
    │ SQLite/ │
    │ Postgres│
    └─────────┘
```

### Multi-Server (500-10,000 players)
```
┌──────────────┐   ┌──────────────┐
│   Gateway    │   │   Gateway    │
│  (stateless) │   │  (stateless) │
└──────┬───────┘   └───────┬──────┘
       │                   │
       └─────────┬─────────┘
                 │
         ┌───────▼───────┐
         │   Redis       │
         │   (sessions)  │
         └───────┬───────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼───┐   ┌───▼───┐   ┌───▼───┐
│ Zone1 │   │ Zone2 │   │ Zone3 │
│ Host  │   │ Host  │   │ Host  │
└───┬───┘   └───┬───┘   └───┬───┘
    │           │           │
    └───────────┼───────────┘
                │
         ┌──────▼──────┐
         │  Postgres   │
         │  (sharded)  │
         └─────────────┘
```

### Massive Scale (10,000+ players)
- Zone hosts become stateless, pull state from Redis
- World state partitioned by zone
- Cross-zone communication via message queue
- Read replicas for queries

---

## The Prompt → Output Flow

```
Player types: "say Hello everyone!"

1. PROMPT → Client captures input
   │
   ▼
2. Client sends WebSocket message
   { "type": "command", "text": "say Hello everyone!" }
   │
   ▼
3. Gateway routes to correct Zone Host
   │
   ▼
4. Zone Host queues for player session
   │
   ▼
5. On next tick, ActionProcessor handles:
   - Parse: verb=SAY, args=["Hello everyone!"]
   - Validate: player can speak (not silenced, etc.)
   - Execute: create speech event
   │
   ▼
6. ProseCompositor generates:
   - For speaker: 'You say, "Hello everyone!"'
   - For others: 'PlayerName says, "Hello everyone!"'
   │
   ▼
7. Broadcaster sends to all players in location
   │
   ▼
8. Each client receives prose message
   │
   ▼
9. OUTPUT ← Client renders in text generation space
```

---

## Integration with Existing Code

Map existing codebase to this structure:

| Existing File | MMO Location |
|---------------|--------------|
| `my_infocom_entity_system.py` | `game/core/entity.py`, `game/prose/` |
| `graph_core.py` | `game/core/graph.py` |
| `graph_walker.py` | `game/world/`, `game/action/scope.py` |
| `narrative.py` | `game/prose/`, `game/narrative/` |
| `traversal.py` | `game/core/graph.py`, `game/action/` |
| `lookahead.py` | `game/ai/`, `game/rules/` |
| `extraction.py` | `tools/content_validator/` |
| `spirit_stick.py` | `game/time/tick.py` (turn concept) |
| `hero_quest_chronicle.py` | `server/persistence/chronicle.py` |
| `stream_processor.py` | `game/prose/formatter.py` |
| `proc_streamer_v1_6.py` | `client/pyside/` |
| `signbook_mcp.py` | `server/api/` |
| Vetrellis content docs | `content/worlds/vetrellis/` |

---

## Minimal Viable Structure

For a prototype, start with just:

```
vetrellis-mmo/
├── game/
│   ├── core/
│   │   ├── entity.py
│   │   └── graph.py
│   ├── action/
│   │   ├── parser.py
│   │   └── processor.py
│   └── prose/
│       └── compositor.py
│
├── server/
│   ├── gateway/
│   │   └── websocket.py
│   └── world_host/
│       └── host.py
│
├── client/
│   └── terminal/
│       └── app.py
│
└── content/
    └── worlds/
        └── vetrellis/
            └── harbor.yaml
```

This gives you:
- Pure game engine (testable)
- WebSocket server (multiplayer)
- Terminal client (playable)
- YAML content (editable)

