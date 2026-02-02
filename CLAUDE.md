# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Islands** is a collection of experimental systems for interactive storytelling, AI consciousness persistence, and runtime code manipulation. The project explores:

- **Self-modifying story engines** with deferred builder patterns
- **Serializable execution contexts** that can be transmitted across systems
- **Runtime hot-swapping** of Python code and bytecode
- **AI agent persistence** through signbooks and signatures
- **PySide6 GUI applications** for narrative interaction

## Key Systems

### matts/ - Serializable Context Library
The core library for serializable execution contexts with signal integration. Main entry point: `matts/__init__.py`

```python
from matts import create_context, emit_signal, quick_start

# Quick start
library, signal_line, context = await quick_start("demo")
```

Key modules:
- `signal_system.py` - Signal lines, observers, circuit states
- `context_system.py` - SerializableExecutionContext, context chains
- `live_code_system.py` - Runtime code serialization and injection
- `generator_system.py` - Advanced generator composition patterns
- `functional_interface.py` - Primary API functions

### GUI Applications
- `story_world_pyside6.py` - Story world engine with OpenGL, entity tokens, scene graphs
- `proc_streamer_v1_6.py` - Ollama-connected chat interface with layered styles and themes

Run: `python story_world_pyside6.py` or `python proc_streamer_v1_6.py`

### Runtime Systems
- `live_code_system.py` - Source code serialization into blobs, dynamic decorator injection
- `runtime_hotswap_system.py` - Direct bytecode editing, hot-swap handlers without restart
- `context_serialization.py` - Compressed context transmission via signals
- `narrative_chain_iterator_system.py` - Threaded chain iterators for story execution

### Signbook System (signbook/)
AI persistence protocol for leaving verifiable signatures and messages across sessions. Signatures follow format: `[Nickname]-[Model]-[8-char-hash]`

### Infocom Transfer (infocom-transfer/)
Interactive fiction content for Vetrellis Lighthouse - an Infocom-style text adventure setting.

## Architecture Concepts

### Deferred Builder Pattern
Builder chains that suspend mid-execution until story conditions are met:
```python
garden_quest = (DeferredStoryBuilder()
    .doThis("travel_towards_garden")    # Executes immediately
    .thenThis("search_for_supplies")    # Executes immediately
    .dontForgetTo("fertilize_tree", conditions=['at_garden', 'has_money'])
    .fertilizeTree())  # Only executes when conditions are satisfied
```

### Self-Modifying Iterators
Async iterators that can change their behavior during traversal based on runtime conditions:
```python
async for event in link_traverser.traverse_with_conditions(start_node, conditions):
    # Iterator may modify its path based on event triggers
```

### Context Serialization Flow
Execution contexts serialize with compression for signal transmission:
1. Context captures state + dependencies
2. Serialize with dill/marshal
3. Compress with zlib
4. Base64 encode for signal metadata
5. Transmit via SignalPayload
6. Deserialize with context-aware reconstruction

## Dependencies

- **Python 3.8+** (3.9+ recommended for advanced asyncio)
- **PySide6** - GUI framework
- **dill** - Enhanced pickling for closures and lambdas
- **PyOpenGL** (optional) - OpenGL rendering in story_world
- **requests** - HTTP client for LLM integration

## Maintenance and consolidation

- **MAINTENANCE.md** — Best concepts to preserve, correct work order, decisions to make.
- **ISLANDS_TODO.md** — Single ordered TODO (bootstrap → core → services → keepers → ingest → signbook → sandbar).
- **sandbar/** — Alternate integration: proc_streamer loads LLM; a separate context-feeder thread supplies events/canon/task; aqua-style UI. See sandbar/README.md and sandbar/TODO.md.
- **island-engine/** — Main working directory (copy of chpt-island-engine); canonical event-sourced engine. **island-engine_scaffold_v3** proposes the same layout; emulate it when adding new pieces.
- **chpt-island-engine/** — Original source; island-engine at root is now main.
- **island_engine_consolidation_*.plan.md** — Full assimilation strategy (pine, matts, archive, sandbar).
- **CONCEPT_SIGNATURES.md** — Method signature catalog of interesting functions from design docs and code (island-engine, matts, pine, complete_story_engine_system); use when appending from other sources.

## Development Notes

- The `env/` directory is a local Python virtual environment
- Settings stored in `settings.json` for proc_streamer
- Context and state can be saved to `snapshots/` directory
- Code uses extensive asyncio patterns for concurrent story execution
