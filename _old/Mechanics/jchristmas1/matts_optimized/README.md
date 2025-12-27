# MATTS Optimized Package

## Structure

```
matts_optimized/
├── __init__.py      (202 lines)  Public API, re-exports all
├── primitives.py    (202 lines)  Layer 0: Enums, protocols, base
├── embedding.py     (344 lines)  Layer 1: Vector storage
├── signals.py       (490 lines)  Layer 1: Generic Observer[P]
├── traversal.py     (557 lines)  Layer 2: Layers, context, wrapper
├── extraction.py    (453 lines)  Layer 3: Text analysis
├── world.py         (499 lines)  Layer 4: WhiteRoom construction
└── lookahead.py     (458 lines)  Layer 5: Possibility engine
                    ─────────────
                     3205 lines total (vs 2207 monolithic)
```

## Dependency Graph

```
                    ┌─────────────┐
                    │  lookahead  │  Layer 5
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    world    │  Layer 4
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ extraction  │  Layer 3
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  traversal  │  Layer 2
                    └──────┬──────┘
                           │
             ┌─────────────┼─────────────┐
             │             │             │
      ┌──────▼──────┐ ┌────▼────┐ ┌──────▼──────┐
      │  embedding  │ │ signals │ │             │  Layer 1
      └──────┬──────┘ └────┬────┘ │             │
             │             │      │             │
             └─────────────┼──────┘             │
                           │                    │
                    ┌──────▼──────┐             │
                    │ primitives  │◄────────────┘  Layer 0
                    └─────────────┘
```

## Key Improvements

1. **Single Responsibility**: Each file has one clear purpose
2. **Clear Dependencies**: No circular imports, clean layer structure
3. **Testable Units**: Each module can be tested independently
4. **Better Navigation**: Find what you need by file name
5. **Easier Maintenance**: Change one concern without touching others

## Quick Start

```python
from matts_optimized import text_to_world, lookahead_from, create_embedding_store

# Create embedding store (for semantic operations)
store = create_embedding_store()

# Complete pipeline: text → game world
world = text_to_world(
    "The old key lay hidden in the desk in the study.",
    embedding_store=store
)

print(world.summary())

# Lookahead for hints
result = lookahead_from(world, world.origin)
for hint in result.get_hints():
    print(f"💡 {hint}")
```

## Module Reference

### primitives.py
- `Priority`, `StorageType`, `SignalType` - Enums
- `WordClass`, `FragmentCategory`, `PossibilityType` - More enums
- `HasId`, `HasName`, `Serializable`, `Embeddable` - Protocols
- `Identified`, `Context` - Base classes

### embedding.py (STORAGE TYPE)
- `Embedding` - Vector with metadata
- `EmbeddingProvider`, `HashEmbeddingProvider` - Create embeddings
- `VectorIndex`, `FlatVectorIndex` - Store and search
- `EmbeddingStore` - High-level manager
- `create_embedding_store()` - Factory

### signals.py (GENERIC OBSERVER)
- `Signal[P]` - Generic signal for ANY type
- `Observer[P]` - Abstract generic observer
- `FunctionObserver`, `MethodObserver`, `CollectorObserver`
- `StateObserver`, `EntityObserver`, `TraversalObserver`
- `ObserverBus` - Central dispatch hub

### traversal.py
- `LayerConfig`, `LayerRegistry` - Layer definitions
- `TraversalContext` - State that travels with iteration
- `TraversalWrapper` - Smart iterator (`__iter__` returns self)
- `smart_iter()` - Factory function

### extraction.py
- `ExtractedWord`, `ExtractedEntity`, `ExtractedRelation`, `ExtractedFragment`
- `ExtractionResult` - Complete extraction output
- `TextExtractor` - Main extraction engine
- `extract_text()` - Factory function

### world.py
- `WorldNode`, `WorldEdge` - Graph elements with embedding support
- `WhiteRoom` - Game world graph
- `WhiteRoomBuilder` - Extraction → WhiteRoom
- `build_world()`, `text_to_world()` - Factory functions

### lookahead.py
- `Possibility` - A discovered possibility
- `LookaheadResult` - Complete lookahead output
- `LookaheadEngine` - BFS + semantic exploration
- `lookahead_from()` - One-shot lookahead

## Integration with Your Files

The optimized package is designed to work alongside your existing:
- `graph_core.py` → `WhiteRoom` is compatible with `IslandGraph`
- `graph_walker.py` → `TraversalWrapper` is compatible with `GraphWalker`
- `my_infocom_entity_system.py` → `WorldNode` compatible with `Entity`

You can use both systems together or gradually migrate.
