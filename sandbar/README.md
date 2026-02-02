# Sandbar — Alternate Integration Path

**Sandbar** is a separate directory for an alternate integration: **proc_streamer** (and its variants) is the **interface**. The LLM output goes **straight to the interface**, not to the game. The **game reports back to the interface** with results. The interface displays **both** streams separately at the same time: **(1) game → LLM** (results/context sent to the LLM) and **(2) LLM → interface** (LLM response to the user). All output goes through the interface.

A **separate thread** supplies the LLM with everything it needs (events, canon, task context from the engine). The UI can be **aqua**-based; the canonical engine is **island-engine** (main working directory).

---

## Role of Each Piece

| Piece | Role in Sandbar |
|-------|------------------|
| **chpt-island-engine** | Canonical event-sourced engine. Event store, projections, keepers, protocol. Not replaced; consumed as the source of truth. |
| **aqua** | UI layer: documents, dock layout, prompt stage, unified console. Can replace or wrap proc_streamer’s own MainWindow with a more modular UI. |
| **proc_streamer_v1_6.py** | **Interface.** LLM loader and chat host. LLM output goes to the interface only (not to the game). Game reports results to the interface. Interface shows **two streams separately**: game→LLM (context/results) and LLM→interface (response). Owns Ollama/OpenAI-compat, streaming, `AssistChannel`. **Context-feeder thread** pushes event/canon/task context into the LLM. |

---

## Interface flow: game ↔ LLM ↔ interface

- **LLM output → interface only.** The LLM does not write directly to the game; its response is shown in the interface.
- **Game → interface.** The game (island-engine) reports results back to the interface (events, state, task outcomes).
- **Dual display.** The interface shows **both** at the same time, separately:
  1. **Game → LLM:** The context/results the game sent to the LLM (what the LLM “saw”).
  2. **LLM → interface:** The LLM’s response streamed to the user.

All output goes through the interface. No hidden pipes; both copies visible.

## “Separate Thread Gives the LLM Everything It Needs”

- **Main thread (UI):** proc_streamer (or aqua) runs the Qt event loop; user types, sees both streams (game→LLM and LLM→interface).
- **LLM worker thread (existing):** `AssistChannel._run_query` runs the HTTP stream; output goes to the interface.
- **Context-feeder thread (to add):** Polls island-engine (orchestrator / event store), builds a context bundle, pushes it into the LLM session. Game results are also sent to the interface for display (game→LLM copy).

---

## Directory Layout (Sandbar)

```
sandbar/
├── README.md                 # This file
├── STRUCTURE.md              # Detailed layout and file roles
├── TODO.md                   # Sandbar-specific tasks (ordered)
├── llm_host/                 # proc_streamer-derived: LLM loading + context-feeder
│   ├── __init__.py
│   ├── assist_channel.py     # Extract AssistChannel + Ollama/OpenAI-compat
│   ├── context_feeder.py    # Thread that builds and injects context from engine
│   └── config.py            # URL, model, timeout (from proc_streamer settings)
├── ui/                       # aqua-based or minimal shell
│   ├── __init__.py
│   └── (aqua components or launcher that uses llm_host + engine)
├── engine/                   # Adapter to chpt-island-engine (no copy of engine)
│   ├── __init__.py
│   └── client.py            # Thin client: recent events, canon projection, task payloads
└── docs/
    └── integration.md       # How sandbar fits with island_engine_consolidation plan
```

- **engine/** does not duplicate chpt-island-engine; it holds a **thin client** (HTTP or in-process) to the orchestrator and event store.
- **llm_host/** holds the logic that today lives in proc_streamer (AssistChannel, threading, healthcheck) plus the new context-feeder thread.
- **ui/** can import from aqua or from a minimal shell that uses `llm_host` and `engine`.

---

## Relation to Island Engine Consolidation

- **chpt-island-engine** remains the canonical architecture (see `island_engine_consolidation_*.plan.md`).
- **Sandbar** is the **alternate path** for “one app that runs the LLM + feeds it engine context + uses aqua-style UI.”
- Consolidation phases (core lib, orchestrator, agent runner, keepers, book ingest) still apply; sandbar consumes the engine once those services exist, and can use the same **agent_runner** LLM adapter once it’s backed by proc_streamer’s Ollama path.

See **STRUCTURE.md** for file-level roles and **TODO.md** for ordered sandbar tasks.  
Islands-wide maintenance and ordered TODOs live in the repo root: **MAINTENANCE.md** and **ISLANDS_TODO.md**.

### Run (from repo root)

```bash
# UI launcher
python -m sandbar.ui.launcher              # proc_streamer (default)
python -m sandbar.ui.launcher legacy        # legacy UI
python -m sandbar.ui.launcher minimal       # llm_host + engine client only

# Server: respond to any message, run the game (requires fastapi + uvicorn)
uvicorn sandbar.server.app:app --reload
# Then POST /message with {"message": "hello", "source": "user"}
```

### What sandbar contains (BUILD.md + assimilated code)

- **engine/game_loop/** — Persistent context iterator + narrative chain (from root/oasis).
- **engine/messaging/** — Connector, MessagingInterface, GameOrchestrator (from islands/).
- **engine/client.py** — Thin client to island-engine (get_recent_events, get_canon_slice, etc.).
- **server/** — FastAPI POST /message; GameRunner (receive_message -> context + one frame -> response).
- **story/core.py** — StoryNode, EntityNode, WorldState, StoryEventManager (complete_story_engine_system stubs).
- **llm_host/** — AssistChannel, config (from proc_streamer).
- **pine/** — Full copy of repo **pine**: core (signals, context, serialization), narrative (deferred_builder, world, traversal, lookahead, extraction, spirit_stick), runtime (live_code, hotswap), graph, messaging, signbook, config, ui. Use `from sandbar.pine.narrative import DeferredStoryBuilder` etc. When running sandbar self-contained, set `PYTHONPATH` to include the repo root so `sandbar` is importable; pine inside sandbar uses `pine.*` so for pine to resolve to sandbar’s copy, run with `PYTHONPATH=sandbar` (e.g. `python -m sandbar.server.app` from sandbar dir) or use `from sandbar.pine...` from outside.
- **mechanics/** — Hero’s spirit stick: **hero_quest_chronicle** (Chronicle, MessagingInterface, HeroActor), **spirit_stick**, **traversing_spirit_stick**, **stream_processor**; **mechanics/everything/** — callback_patterns, my_infocom_entity_system, oracle_tower, signals, traversal, world, lookahead, extraction, etc. Message→queue→handler patterns.
- **runtime/** — Root **live_code_system.py**, **runtime_hotswap_system.py**, **context_serialization.py** (runtime code injection, hot-swap, compressed context).
- **matts/** — **signal_system**, **context_system**, **context_serialization**, **functional_interface** (serializable contexts, event-driven comms).
- **vetrell/** — **language** (english_code_object), **narrative** (context_iterator), **runtime** (hotswap, live_code), **serialization** (context), **analysis** (code_object).
- **docs/** — complete_story_engine_system.md, SANDBAR_SEARCH_REPORT.md, CLEANUP_PLAN.md, ASSIMILATION.md, GAME_VS_API_SEPARATION.md, MMO_ARCHITECTURE.md, integration.md, main_structure.md. See **docs/SANDBAR_ASSIMILATION_REPORT.md** for what was copied where and how to use it.
