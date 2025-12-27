# MATTS Project Complete Archive

## For Claude Code Transfer

**Date**: December 2024  
**Author**: CloudyCadet  
**Project**: Multi-layered Adaptive Text Transformation System (matts)

---

## Quick Start

```python
from matts_project_archive import text_to_world, smart_iter, lookahead_from

# Complete pipeline: text → world
story = "The key lay hidden in the old desk in the study..."
world = text_to_world(story)

# Smart traversal with context
for entity in smart_iter(world.entities.values()):
    print(entity.name)

# Lookahead for hints
result = lookahead_from(world, world.origin, max_depth=3)
hints = result.get_hints()
```

---

## Table of Contents

1. [Project Philosophy](#philosophy)
2. [Core Systems](#core-systems)
3. [Text Extraction Pipeline](#text-extraction)
4. [White Room Builder](#white-room)
5. [Observer Pattern](#observer)
6. [Traversal Wrapper](#traversal)
7. [Lookahead Engine](#lookahead)
8. [Integration Guide](#integration)
9. [API Reference](#api)
10. [File Manifest](#files)

---

## Philosophy

### The Core Insight

We don't author puzzles or sandboxes. We **scan text**, **analyze every word**, and **construct a world** from what we find.

> "Every word is on trial for its life." — Read Like a Writer

### Key Revelations

1. **The wrapper IS the iterator** — `__iter__()` returns self
2. **Same nodes, different edges** — Layers filter visibility, not existence
3. **Lookahead ≠ Movement** — Explores possibility space without changing state
4. **Near-misses are gold** — One condition away = achievable = perfect hint
5. **Prose is conditional** — Fragments have conditions, composition is runtime

---

## Core Systems

### Architecture Pipeline

```
SOURCE TEXT (novel, short story)
        │
        ▼
TEXT EXTRACTION
  - Tokenize words
  - Classify grammar
  - Extract entities
  - Find relations
        │
        ▼
WHITE ROOM BUILDER
  - Convert to game entities
  - Build entity graph
  - Assign prose fragments
  - Connect orphans to origin
        │
        ▼
ENTITY GRAPH
  - Nodes: locations, items, characters, concepts
  - Edges: relational, logical, wildcard links
  - Layers: same nodes, different edge visibility
        │
        ▼
RUNTIME SYSTEMS
  - Traversal Wrapper (smart iterator)
  - Lookahead Engine (possibility discovery)
  - Observer Bus (state machine events)
  - Prose Compositor (fragment assembly)
```

---

## Text Extraction

### Classes

| Class | Purpose |
|-------|---------|
| `TextExtractor` | Main extraction engine |
| `ExtractedWord` | Single word with classification |
| `ExtractedEntity` | Entity found in text |
| `ExtractedRelation` | Relationship between entities |
| `ExtractedFragment` | Prose fragment from text |
| `ExtractionResult` | Complete extraction output |

### Word Classification

- `NOUN`, `PROPER_NOUN`, `VERB`, `ADJECTIVE`
- `ADVERB`, `PREPOSITION`, `ARTICLE`, `PRONOUN`
- Based on position, endings, and known indicators

### Entity Categories

- `LOCATION` — rooms, buildings, paths
- `CHARACTER` — people, named individuals
- `ITEM` — objects, keys, books
- `CONCEPT` — abstract ideas, knowledge

### Usage

```python
from matts_project_archive import extract_text

story = """
The old house stood at the end of the lane. A brass key 
lay hidden under the papers in the desk.
"""

result = extract_text(story)
print(result.summary())

# Access extracted data
for entity in result.entities:
    print(f"{entity.name}: {entity.category.value}")
```

---

## White Room Builder

### The White Room Concept

The **White Room** is the origin point—a blank space where the world forms. All entities extracted from text are connected to this origin.

### Classes

| Class | Purpose |
|-------|---------|
| `WhiteRoomBuilder` | Builds game world from extraction |
| `WhiteRoom` | The constructed world |
| `GameEntity` | A game world entity |
| `GameFragment` | Prose fragment for entity |
| `GameLink` | Connection between entities |

### Entity Types

```python
class EntityType(Enum):
    LOCATION = auto()
    ITEM = auto()
    CHARACTER = auto()
    CONTAINER = auto()
    DOOR = auto()
    CONCEPT = auto()
```

### Entity States

```python
class EntityState(Enum):
    NORMAL = auto()
    HIDDEN = auto()
    LOCKED = auto()
    OPEN = auto()
    CLOSED = auto()
    BROKEN = auto()
```

### Usage

```python
from matts_project_archive import text_to_world

world = text_to_world(story)
print(world.summary())

# Access entities
for entity_id, entity in world.entities.items():
    print(f"{entity.name} ({entity.entity_type.name})")

# Access links
for link in world.links:
    source = world.get_entity(link.source_id)
    target = world.get_entity(link.target_id)
    print(f"{source.name} --[{link.kind}]--> {target.name}")
```

---

## Observer Pattern

### The Observer Bus

Central hub for event dispatch. Observers register for signal types and receive matching signals.

### Signal Types

```python
class SignalType(Enum):
    # State
    STATE_ENTER, STATE_EXIT, STATE_CHANGE
    
    # Entity
    ENTITY_CREATED, ENTITY_MODIFIED, ENTITY_DESTROYED
    
    # Traversal
    TRAVERSAL_STEP, TRAVERSAL_START, TRAVERSAL_END, LAYER_SWITCH
    
    # Game
    ACTION_PERFORMED, CONDITION_MET, CONDITION_FAILED
    
    # System
    TICK, ERROR, WARNING, CUSTOM
```

### Observer Types

| Observer | Purpose |
|----------|---------|
| `Observer` | Abstract base class |
| `FunctionObserver` | Wraps a callback function |
| `StateObserver` | Tracks state machine transitions |
| `EntityObserver` | Tracks entity lifecycle |
| `TraversalObserver` | Tracks graph traversal |

### Usage

```python
from matts_project_archive import ObserverBus, SignalType

bus = ObserverBus()

# Register state observer with hooks
state_obs, _ = bus.create_state_observer()
state_obs.on_enter('exploring', lambda s: print("Started exploring!"))
state_obs.on_exit('exploring', lambda s: print("Stopped exploring."))

# Emit signals
bus.emit_type(SignalType.STATE_CHANGE, data={
    'old_state': 'idle',
    'new_state': 'exploring'
})

# Simple callback registration
bus.on(SignalType.ERROR, lambda s: print(f"Error: {s.data}"))
```

---

## Traversal Wrapper

### The Key Insight

The wrapper returns **itself** from `__iter__()`. This means:

- Context persists through the for loop
- Layer switching works mid-iteration
- Graph hotswapping is possible
- Branching creates parallel explorations

### Usage

```python
from matts_project_archive import smart_iter, TraversalContext

# Basic usage
wrapper = smart_iter(my_list)
for item in wrapper:
    print(f"At {item}, depth={wrapper.depth}")

# With callbacks
wrapper = smart_iter(entities)
wrapper.on_step(lambda w, item: print(f"Step {w._step_count}: {item}"))
for item in wrapper:
    pass  # Callbacks handle output

# Layer switching
wrapper.switch_layer("thematic", push=True)  # Save current
# ... traverse thematic connections ...
wrapper.pop_layer()  # Return to previous

# Graph hotswap
wrapper.swap_graph(new_graph, reset=False)  # Continue with new graph

# Branching
branch = wrapper.branch()  # Clone at current position
# Main and branch explore independently
```

### Context

The `TraversalContext` travels with the wrapper:

```python
@dataclass
class TraversalContext:
    current_node: Any
    previous_node: Any
    path: List[Any]
    visited: Set[str]
    depth: int
    
    current_layer: str
    layer_stack: List[str]
    
    flags: Dict[str, bool]
    counters: Dict[str, int]
    inventory: List[str]
    buffer: Dict[str, Any]
    
    graph_ref: Any  # Can be swapped!
```

---

## Lookahead Engine

### Purpose

The lookahead **pre-traverses** the graph WITHOUT moving the player. It discovers:

- Reachable locations
- Hidden items
- Blocked paths
- Near-misses (almost accessible)
- Puzzle chains

### Possibility Types

```python
class PossibilityType(Enum):
    REACHABLE_LOCATION
    BLOCKED_PATH
    LOCKED_DOOR
    VISIBLE_ITEM
    HIDDEN_ITEM
    TAKEABLE_ITEM
    REQUIRED_ITEM
    UNLOCKABLE
    REVEALABLE
    NEAR_MISS
```

### Usage

```python
from matts_project_archive import lookahead_from, LookaheadEngine

# One-shot lookahead
result = lookahead_from(world, world.origin, max_depth=3)

# Get hints
hints = result.get_hints(max_hints=3)
for hint in hints:
    print(f"💡 {hint}")

# Analyze possibilities
for p in result.all_possibilities:
    if p.is_near_miss():
        print(f"Almost: {p.entity_name}")
    if p.is_blocked():
        print(f"Blocked: {p.entity_name} - {p.conditions_unmet}")

# Full engine usage
engine = LookaheadEngine(world)
result = engine.lookahead(current_location, context={'inventory': player_inventory})
```

---

## Integration Guide

### With Existing Entity System

The archive is designed to work with your existing `my_infocom_entity_system.py`:

```python
# The GameEntity class maps to Entity
# The GameLink class maps to Link
# FragmentCategory matches your enum
# EntityType and EntityState match your enums
```

### With proc_streamer UI

The Observer pattern integrates with your UI:

```python
bus = ObserverBus()

# Register UI update observer
bus.on(SignalType.STATE_CHANGE, update_ui)
bus.on(SignalType.ENTITY_MODIFIED, refresh_display)
```

### With Signal System

Your existing `signal_system.py` can be adapted:

```python
# SignalPayload → Signal
# ObserverStats → Observer tracking
# Circuit breaker pattern → handled in ObserverBus
```

---

## API Reference

### Extraction

```python
extract_text(text: str) -> ExtractionResult
```

### Building

```python
build_white_room(extraction: ExtractionResult) -> WhiteRoom
text_to_world(text: str) -> WhiteRoom  # Complete pipeline
```

### Traversal

```python
smart_iter(source, context=None, layer_registry=None) -> TraversalWrapper
```

### Lookahead

```python
lookahead_from(graph, entity, max_depth=3) -> LookaheadResult
```

### Observer

```python
bus = ObserverBus()
bus.register(observer, signal_types=None) -> str
bus.on(signal_type, callback, priority=NORMAL) -> str
bus.emit(signal)
bus.emit_type(signal_type, source_id="", data=None) -> Signal
```

---

## File Manifest

### Primary Archive

| File | Purpose |
|------|---------|
| `matts_project_archive.py` | Complete unified system |
| `matts_project_archive.md` | This documentation |

### Related Project Files

| File | Purpose |
|------|---------|
| `my_infocom_entity_system.py` | Entity/Link/Prose system |
| `my_infocom_design_document.md` | Design rationale |
| `signal_system.py` | Advanced signal handling |
| `context_system.py` | Context management |
| `generator_system.py` | Generator patterns |
| `graph_system.py` | Graph structures |

### Previous Chat Outputs

| File | Purpose |
|------|---------|
| `unified_traversal_lookahead.py` | Earlier traversal system |
| `island_engine_design.md` | Island Engine documentation |

---

## Memory Bridge for Claude Code

Claude Code doesn't have access to conversation memory. This archive contains:

1. **All code** — Complete working system
2. **All design decisions** — Why things work the way they do
3. **Integration patterns** — How to connect to existing code
4. **API reference** — How to use each component

### Key Context for Claude Code

- Project name: **matts**
- Goal: AI-powered narrative systems, novel-to-MUD conversion
- Core insight: Wrapper IS the iterator (`__iter__` returns self)
- Design philosophy: Extract from text, don't author
- Target: Short stories first, then novels

### When Using in Claude Code

```python
# Start here
from matts_project_archive import text_to_world, smart_iter, ObserverBus

# Read the docstrings - they contain design rationale
help(text_to_world)
help(TraversalWrapper)
```

---

## Version History

- **v1.0** — Initial archive (December 2024)
  - Text extraction pipeline
  - White Room builder
  - Observer pattern
  - Traversal wrapper
  - Lookahead engine
  - Complete documentation

---

*This archive was created to bridge the gap between Claude.ai (with project memory) and Claude Code (without). Every detail from our conversation is preserved here.*
