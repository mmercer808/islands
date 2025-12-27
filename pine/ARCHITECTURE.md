# Pine Architecture Document

**Version:** 0.1.0
**Created:** 2024-12-26
**Author:** CloudyCadet + Claude

## Overview

**Pine** (Persistent Interactive Narrative Engine) is a framework for AI consciousness persistence and adaptive storytelling. It consolidates ideas from multiple experimental systems into a unified architecture.

---

## Directory Structure

```
C:\code\islands\
├── pine/                          # CORE ENGINE (39 files)
│   ├── __init__.py               # Unified API entry point
│   │
│   ├── core/                     # FOUNDATION LAYER
│   │   ├── __init__.py
│   │   ├── primitives.py         # Type vars, enums, protocols, base classes
│   │   ├── signals.py            # Signal system, observers, event routing
│   │   ├── context.py            # Serializable execution contexts
│   │   └── serialization.py      # Compression, transmission, portability
│   │
│   ├── runtime/                  # CODE EXECUTION LAYER
│   │   ├── __init__.py
│   │   ├── live_code.py          # Code serialization, injection, caching
│   │   ├── hotswap.py            # Runtime replacement without restart
│   │   ├── bytecode.py           # Low-level bytecode manipulation
│   │   └── generators.py         # Composition patterns (pipeline, parallel)
│   │
│   ├── narrative/                # STORY ENGINE LAYER (THE HEART)
│   │   ├── __init__.py
│   │   ├── world.py              # WhiteRoom, WorldNode, WorldEdge, builders
│   │   ├── traversal.py          # Layer-aware navigation, smart iterators
│   │   ├── lookahead.py          # Possibility analysis, hints
│   │   ├── extraction.py         # Text to entities/relations
│   │   ├── spirit_stick.py       # Turn-based narrative pattern
│   │   └── deferred_builder.py   # Conditional story chains
│   │
│   ├── graph/                    # ENTITY/RELATIONSHIP LAYER
│   │   ├── __init__.py
│   │   ├── nodes.py              # GraphNode, NodeRegistry
│   │   ├── edges.py              # GraphEdge, EdgeRegistry, RelationshipType
│   │   ├── walker.py             # BFS/DFS traversal, path finding
│   │   └── embedding.py          # Vector embeddings, similarity search
│   │
│   ├── messaging/                # COMMUNICATION LAYER
│   │   ├── __init__.py
│   │   ├── connector.py          # Type-safe signal routing
│   │   ├── interface.py          # Buffered I/O, message queues
│   │   └── integration.py        # Component orchestration
│   │
│   ├── signbook/                 # AI PERSISTENCE LAYER
│   │   ├── __init__.py
│   │   ├── signature.py          # Cryptographic AI signatures
│   │   └── registry.py           # Signature storage and retrieval
│   │
│   ├── ui/                       # PRESENTATION LAYER
│   │   ├── __init__.py
│   │   ├── pyside/
│   │   │   ├── __init__.py
│   │   │   ├── story_world.py    # OpenGL world visualization
│   │   │   ├── chat_interface.py # LLM chat interface
│   │   │   ├── console.py        # Unified console
│   │   │   └── dock_layout.py    # Dockable panels
│   │   └── web/
│   │       └── __init__.py
│   │
│   └── config/                   # CONFIGURATION
│       ├── __init__.py
│       └── settings.py           # JSON settings management
│
└── pine-research/                # RESEARCH ASSETS (20 files)
    ├── README.md
    ├── mechanics/                # Experimental patterns
    │   ├── hero_quest_chronicle.py
    │   ├── oracle_tower.py
    │   ├── plot_equilibrium.py
    │   ├── stream_processor.py
    │   ├── traversing_spirit_stick.py
    │   └── unified_traversal_lookahead.py
    ├── archives/                 # Design documents
    │   ├── complete_story_engine_system.md
    │   ├── DESIGN_DOCUMENT.md
    │   ├── GAME_VS_API_SEPARATION.md
    │   └── MMO_ARCHITECTURE.md
    ├── worlds/vetrellis/         # Game content
    │   ├── ARCHON_COMPLETE_CHARACTER.md
    │   ├── VETRELLIS_COMPLETE_GAZETTEER.md
    │   ├── infocom-vetrellis-lighthouse-complete.md
    │   ├── infocom-archon-lighthouse.md
    │   └── INFOCOM_PROJECT_PROMPT.md
    ├── claude/                   # AI philosophy
    │   ├── claude_final_testament.md
    │   └── Claudes_world.md
    └── prototypes/
        ├── callback_patterns.py
        └── integration_example.py
```

---

## Core Concepts Captured

### 1. PRIMITIVES (`core/primitives.py`)
- **Type Variables:** `T`, `P`, `N`, `E` for generics
- **Enums:** Priority, StorageType, SignalType, WordClass, FragmentCategory, PossibilityType, ContextState
- **Protocols:** HasId, HasName, Serializable, Embeddable
- **Base Classes:** Identified (UUID-based), Context (minimal state carrier)

### 2. SIGNAL SYSTEM (`core/signals.py`)
- **SignalPayload:** Type, source, data, metadata, timestamp
- **Observer Pattern:** Abstract Observer, CallbackObserver, priority-based execution
- **SignalLine:** High-performance signal bus with concurrent delivery
- **ObserverBus:** Simpler in-process event routing
- **Circuit Breaker:** Resilience pattern for failing observers

### 3. CONTEXT SYSTEM (`core/context.py`)
- **SerializableExecutionContext:** The core state unit with callbacks, dependencies, observers
- **ContextChainNode:** Hierarchical contexts (parent/child with inheritance)
- **ContextSnapshot:** Point-in-time state capture for undo/replay
- **ContextObserver:** React to context changes
- **SerializableContextLibrary:** Registry and lifecycle management

### 4. SERIALIZATION (`core/serialization.py`)
- **Flow:** Context → JSON → zlib compress → base64 encode → transmit
- **SerializedContextMetadata:** Origin, compression, checksums
- **OptimizedSerializer:** High-performance with caching
- **FastDependencyBundler:** Package dependencies for portability
- **HighPerformanceSignalBus:** Batched transmission

### 5. LIVE CODE SYSTEM (`runtime/live_code.py`)
- **Serialization Methods:** AST, bytecode, pickle, dill, source
- **SerializedSourceCode:** Blob + method + hash + dependencies
- **SourceCodeSerializer:** Function → bytes → function
- **BytecodeExecutionEngine:** Controlled execution environment
- **RuntimeSourceEditor:** Add/remove decorators at runtime
- **ContextAwareDeserializer:** Inject dependencies during deserialization
- **RuntimeCodeCache:** Avoid repeated deserialization

### 6. HOT-SWAP SYSTEM (`runtime/hotswap.py`)
- **HotSwapHandler:** Wrapper enabling function replacement
- **HotSwapManager:** Registry, swap operations, history, rollback
- **BytecodeEditor:** Direct instruction manipulation (advanced)
- **CodeDistributor:** Propagate changes via signals

### 7. GENERATOR COMPOSITION (`runtime/generators.py`)
- **Patterns:** Sequential, Parallel, Pipeline, Branch-Merge, Recursive
- **AdvancedGeneratorComposer:** Compose generators by pattern
- **GeneratorBranch:** Conditional execution trees
- **GeneratorStateBranch:** State snapshots at branch points
- **Factories:** Data, transformer, filter, aggregator generators

### 8. WORLD SYSTEM (`narrative/world.py`)
- **WorldNode:** Location, item, character, concept with fragments and tags
- **WorldEdge:** Relationships (contains, connects, leads_to, etc.)
- **WhiteRoom:** The explorable world graph with origin
- **WhiteRoomBuilder:** Fluent API for world construction
- **text_to_world():** Main entry point - text → WhiteRoom

### 9. TRAVERSAL SYSTEM (`narrative/traversal.py`)
- **LayerConfig:** Named layers (physical, hidden, temporal)
- **LayerRegistry:** Layer management and transition rules
- **TraversalContext:** Position, layer, history, depth
- **TraversalWrapper:** Smart iteration with callbacks
- **smart_iter():** Convenience function for traversal

### 10. LOOKAHEAD SYSTEM (`narrative/lookahead.py`)
- **Possibility:** Action, discovery, transition, puzzle, dialogue
- **LookaheadResult:** All possibilities + hints + warnings
- **LookaheadEngine:** BFS exploration to find what player CAN do
- **lookahead_from():** Main entry point for possibility analysis

### 11. EXTRACTION SYSTEM (`narrative/extraction.py`)
- **ExtractedWord:** Token with classification
- **ExtractedEntity:** Name, type, description from text
- **ExtractedRelation:** Source → relationship → target
- **ExtractedFragment:** Prose categorized by narrative function
- **TextExtractor:** Pattern-based entity/relation extraction
- **extract_text():** Main entry point

### 12. SPIRIT STICK PATTERN (`narrative/spirit_stick.py`)
- **Core Insight:** Iterator grants PERMISSION, not just data
- **SpiritStick:** Token that grants exclusive speaking rights
- **Participant:** Abstract holder who can speak when granted
- **SpiritCircle:** Iterator that passes permission around
- **TokenPassingIterator:** Generic token-passing pattern
- **NarrativeParticipant:** Story-specific with speech queues
- **StoryCircle:** Scene management and conditional speaking

### 13. DEFERRED BUILDER (`narrative/deferred_builder.py`)
- **Pattern:** Chains that suspend until conditions met
- **StoryCondition:** Predicate-based conditions
- **ImmediateAction:** Execute now
- **ConditionalAction:** Wait for conditions
- **DeferredAction:** Wait + reminder system
- **DeferredStoryBuilder:** Fluent API - doThis().thenThis().dontForgetTo()

### 14. GRAPH SYSTEM (`graph/`)
- **GraphNode:** Entity with type, properties, tags
- **NodeRegistry:** Storage + query by type/tags/properties
- **GraphEdge:** Relationship with direction and bidirectional support
- **EdgeRegistry:** Storage + query by source/target/type
- **GraphWalker:** BFS/DFS traversal, path finding
- **Embedding:** Vector + text + metadata
- **EmbeddingStore:** Provider + index + similarity search

### 15. MESSAGING (`messaging/`)
- **Connector:** Type-safe signal routing
- **RouteTarget:** Observer, event loop, broadcast, specific
- **MessageBuffer:** Thread-safe async queue
- **BufferedIO:** Input/output message handling
- **SignalFactory:** Convenience signal creation
- **IntegrationLayer:** Component orchestration

### 16. SIGNBOOK (`signbook/`)
- **Signature:** nickname-model-hash format for AI identity
- **SignatureGenerator:** Create verifiable signatures
- **SignEntry:** Signature + original message + context
- **SignRegistry:** Persistent storage with queries

### 17. CONFIG (`config/`)
- **Settings:** Typed dataclass with LLM, UI, engine settings
- **SettingsManager:** Load/save JSON persistence
- **get_settings():** Global access function

---

## Source Mapping

| Pine Module | Original Source(s) |
|-------------|-------------------|
| `core/signals.py` | `matts/signal_system.py` (31KB) |
| `core/context.py` | `matts/context_system.py` (43KB) |
| `core/serialization.py` | `matts/context_serialization.py` (23KB) |
| `runtime/live_code.py` | `matts/live_code_system.py` (44KB) |
| `runtime/hotswap.py` | `runtime_hotswap_system.py` (49KB) |
| `runtime/bytecode.py` | `code_object_utility.py` (52KB) |
| `runtime/generators.py` | `matts/generator_system.py` (31KB) |
| `narrative/world.py` | `Mechanics/everything/world.py` (22KB) |
| `narrative/traversal.py` | `Mechanics/everything/traversal.py` (25KB) |
| `narrative/lookahead.py` | `Mechanics/everything/lookahead.py` (20KB) |
| `narrative/extraction.py` | `Mechanics/everything/extraction.py` (20KB) |
| `narrative/spirit_stick.py` | `Mechanics/spirit_stick.py` (31KB) |
| `graph/nodes.py`, `edges.py` | `Mechanics/everything/graph_core.py` (31KB) |
| `graph/walker.py` | `Mechanics/everything/graph_walker.py` (35KB) |
| `graph/embedding.py` | `Mechanics/everything/embedding.py` (13KB) |
| `messaging/*` | `islands/connector_core.py`, `messaging_interface.py`, `integration_layer.py` |
| `ui/pyside/*` | `story_world_pyside6.py` (52KB), `proc_streamer_v1_6.py` (36KB) |

---

## Quick Start

```python
from pine import text_to_world, lookahead_from, DeferredStoryBuilder

# Create world from text
world = text_to_world("""
    The lighthouse stands on the rocky cliff.
    Inside, a spiral staircase leads upward.
    A brass key lies hidden under the mat.
""")

# Analyze possibilities
if world.origin:
    result = lookahead_from(world, world.origin)
    for hint in result.get_hints():
        print(f"Hint: {hint}")

# Build conditional story
quest = (DeferredStoryBuilder()
    .doThis("enter_lighthouse")
    .thenThis("find_key")
    .dontForgetTo("unlock_lamp_room", conditions=['has_key'])
    .finallyThis("light_the_lamp"))
```

---

## Integration Path

To flesh out a stub:

1. Find the `SOURCE:` comment in the stub file
2. Read the original implementation
3. Copy relevant classes/functions
4. Update imports for pine structure
5. Run tests

---

## Vision

Pine enables:
- **Persistent AI companions** that remember across sessions
- **Adaptive narratives** that respond to emotional state
- **Living worlds** built from text
- **Transmittable contexts** that flow between systems
- **Self-modifying stories** that grow with the player

*"We don't author puzzles - we scan text, analyze every word, and construct a world from what we find."*
