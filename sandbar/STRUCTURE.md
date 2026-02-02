# Sandbar — Directory Structure and File Roles

## Overview

Sandbar is the **alternate integration** directory. **proc_streamer and its variants are the interface.** LLM output goes to the interface only; the game reports to the interface. The interface shows both: **game→LLM** (context/results) and **LLM→interface** (response), separately. Sandbar consumes **island-engine** (main working directory at repo root). Layout is below with roles and dependencies.

---

## Directory Layout

```
sandbar/
├── README.md
├── STRUCTURE.md              # This file
├── TODO.md
├── BUILD.md                  # What was merged from design/TODO/CONCEPT_SIGNATURES
├── llm_host/                 # LLM loading + streaming + context-feeder thread
├── ui/                       # aqua-based or minimal launcher
├── engine/                   # Thin client + messaging + game_loop
│   ├── client.py             # Orchestrator thin client
│   ├── messaging/            # Connector, MessagingInterface, GameOrchestrator (from islands/)
│   └── game_loop/            # Persistent context + narrative chain (from root/oasis)
│       ├── persistent_context_iterator.py
│       └── narrative_chain_iterator_system.py
├── server/                   # Respond to any message, run the game
│   ├── app.py                # FastAPI POST /message
│   └── game_runner.py       # receive_message -> context + one frame -> response
├── story/                    # Story/entity core (complete_story_engine_system stubs)
│   └── core.py               # StoryNode, EntityNode, WorldState, StoryEventManager
└── docs/
    └── integration.md, ASSIMILATION.md, SANDBAR_SEARCH_REPORT.md, ...
```

---

## llm_host/ (proc_streamer-derived)

| File | Role |
|------|------|
| `__init__.py` | Package init; expose `AssistChannel`, `ContextFeeder`, config. |
| `assist_channel.py` | Extracted from proc_streamer: Ollama + OpenAI-compat streaming, healthcheck, `query()` → worker thread `_run_query()`. |
| `context_feeder.py` | **New.** Thread that periodically (or on event) fetches from `engine.client`, builds a context bundle (recent events, canon slice, current task), and injects it into the LLM session (e.g. system prompt or prepended context). |
| `config.py` | URL, model, timeout, stream flag; mirrors proc_streamer `GlobalSettings.llm`. |

**Dependencies:** `engine.client` for event/canon/task data; optional `requests`.

---

## ui/

| File | Role |
|------|------|
| `__init__.py` | Package init. |
| (optional) `launcher.py` | Entry point: start engine client, start llm_host (AssistChannel + ContextFeeder), show aqua-based or minimal window. |

UI can live here as a thin shell that uses **aqua** (from `aqua/`) for document, dock, prompt stage, and console, and uses **llm_host** for all LLM and context-feeder logic.

---

## engine/

| File | Role |
|------|------|
| `__init__.py` | Package init; expose client, messaging, game_loop. |
| `client.py` | **Thin client** to island-engine: `get_recent_events(limit)`, `get_canon_slice()`, `get_current_task()`, `get_timeline_state(player_id)`. |
| `messaging/` | Connector, Listener, MessagingInterface, GameOrchestrator (assimilated from islands/). |
| `game_loop/` | **Persistent context + narrative chain** (consolidated from root/oasis): `persistent_context_iterator.py`, `narrative_chain_iterator_system.py`. Exposes ContextWindow, PersistentContextIterator, TaskScheduler, GameLoopIntegrator, ProcessControlManager, NarrativeChain, ChainLink, etc. |

**Dependencies:** island-engine (main) orchestrator API; stdlib only for game_loop.

---

## server/

| File | Role |
|------|------|
| `app.py` | FastAPI app: **POST /message** (body: message, source) → run game one step → response. **GET /health**. Requires fastapi + uvicorn. |
| `game_runner.py` | **GameRunner**: receive_message(message, source) → add PLAYER_INPUT to PersistentContextIterator → run_one_frame() → return response + recent operations. Singleton get_runner(), run_message(). |

**Dependencies:** sandbar.engine.game_loop (PersistentContextIterator, OperationType, etc.).

---

## story/

| File | Role |
|------|------|
| `core.py` | Stubs from **complete_story_engine_system.md**: StoryNode (scene_id, deferred_operations), EntityNode (entity_id, extract_meanings), WorldState (check_condition, update), StoryEventManager (trigger_story_event, world_state). Merge with pine/Mechanics where real implementations exist. |
| `__init__.py` | Expose StoryNode, EntityNode, WorldState, StoryEventManager. |

---

## docs/

| File | Role |
|------|------|
| `integration.md` | How sandbar fits the island_engine_consolidation plan; when to use sandbar vs. main agent_runner; diagram of threads (UI, LLM worker, context-feeder). |

---

## Source Locations (Reference)

- **proc_streamer (LLM + thread):** `proc_streamer_v1_6.py` — `AssistChannel`, `_run_query` (thread), `_query_ollama` / `_query_openai_compat`, `GlobalSettings.llm`.
- **aqua (UI):** `aqua/` — `aqua_v2.py`, `document.py`, `dock_layout.py`, `prompt_stage.py`; root `aqua.py` (signal-based, matts integration).
- **Engine API:** `island-engine/services/orchestrator/` (main working directory) — `event_store.py`, `main.py`, `projections.py`; TODO endpoints `/events/recent`, `/projection/canon`, etc.

Sandbar **llm_host** should stay independent of PySide6 so it can be reused by agent_runner or headless scripts; **ui/** is the only part that depends on aqua/Qt.
