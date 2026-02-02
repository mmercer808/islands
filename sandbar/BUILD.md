# Sandbar Build — What Was Merged

Sandbar is the **consolidated** implementation of what the searched directories (oasis/engine, Mechanics, island-engine, pine, pine-research) and the design/TODO/CONCEPT_SIGNATURES docs describe. This file lists what was included and where it came from.

---

## 1. From root / oasis (code copied into sandbar)

| Element | Source | Sandbar location |
|--------|--------|------------------|
| Persistent context iterator | `persistent_context_iterator.py` (root/oasis/engine) | `engine/game_loop/persistent_context_iterator.py` |
| Narrative chain iterator system | `narrative_chain_iterator_system.py` (root/oasis/engine) | `engine/game_loop/narrative_chain_iterator_system.py` |
| ContextWindow, ContextOperation, OperationType | persistent_context_iterator | `engine/game_loop/` |
| PersistentContextIterator, TaskScheduler, GameLoopIntegrator | persistent_context_iterator | `engine/game_loop/` |
| ProcessControlManager, NarrativeChain, ChainLink, ChainEvent, CrossChainHandler, ChainIteratorEventQueue, StoryExecutionThread | narrative_chain_iterator_system | `engine/game_loop/` |

---

## 2. From design docs (stubs / implementations)

| Element | Design doc | Sandbar location |
|--------|------------|------------------|
| StoryNode (scene_id, deferred_operations, add_deferred_operation) | complete_story_engine_system.md | `story/core.py` |
| EntityNode (entity_id, entity_type, extract_meanings) | complete_story_engine_system.md | `story/core.py` |
| WorldState (check_condition, update) | complete_story_engine_system.md | `story/core.py` |
| StoryEventManager (trigger_story_event, world_state) | complete_story_engine_system.md | `story/core.py` |
| Server “respond to any message” + “run the game” | SANDBAR_SEARCH_REPORT.md, MMO_ARCHITECTURE.md | `server/app.py`, `server/game_runner.py` |

---

## 3. From CONCEPT_SIGNATURES / island-engine TODO

| Concept | CONCEPT_SIGNATURES / TODO | Sandbar |
|--------|---------------------------|--------|
| Event store append, list_recent | island-engine orchestrator | `engine/client.py` stubs (get_recent_events, etc.); real impl in island-engine |
| rebuild_canon, rebuild_timeline | projections | `engine/client.py` get_canon_slice(), get_timeline_state() stubs |
| detect_divergence, propose_gravity_moves | keepers | Not in sandbar; live in island-engine keepers |
| Signbook add_entry, list_entries | signbook_server | Not in sandbar; live in island-engine |
| LLM complete (Ollama) | agent_runner / proc_streamer | `llm_host/assist_channel.py`, `llm_host/config.py` |
| DeferredStoryBuilder (doThis, thenThis, dontForgetTo, execute) | complete_story_engine_system, pine | Pine has full impl; sandbar story/ has StoryNode deferred_operations; can import pine.narrative.deferred_builder |
| MessagingInterface, receive_prompt, update | islands/, oasis | `engine/messaging/` (MessagingInterface, GameOrchestrator) |
| Connector, Listener, RouteTarget | islands/ | `engine/messaging/connector_core.py` |

---

## 4. From contexting1_2_.txt / MMO_ARCHITECTURE

- **contexting1_2_.txt**: Bidirectional messaging, conversation context, turn-taking — reflected in sandbar **llm_host** (callbacks, streaming) and **server** (message in → game step → response out).
- **MMO_ARCHITECTURE.md**: Proposed game/ structure (core, world, action, prose, narrative) — sandbar **story/** holds core stubs (StoryNode, EntityNode, WorldState, StoryEventManager); full game/ layout can be added under story/ or delegated to island-engine.

---

## 5. Run / use

- **Server (respond to any message):**  
  From repo root: `uvicorn sandbar.server.app:app --reload`  
  Then `POST /message` with `{"message": "hello", "source": "user"}`.

- **Game loop only:**  
  `from sandbar.engine.game_loop import PersistentContextIterator, GameLoopIntegrator, ...`

- **Story stubs:**  
  `from sandbar.story import StoryNode, EntityNode, WorldState, StoryEventManager`

- **LLM host:**  
  `from sandbar.llm_host import AssistChannel, load_llm_config`

- **Engine client (orchestrator):**  
  `from sandbar.engine.client import get_recent_events, get_canon_slice`

- **Messaging / orchestrator:**  
  `from sandbar.engine.messaging import MessagingInterface, GameOrchestrator, Connector`

---

## 6. Not duplicated (use from repo)

- **pine** — narrative (deferred_builder, world, traversal, lookahead), messaging, runtime: use from `pine/` when building full story flows.
- **matts** — signals, context, live code: use from `matts/`.
- **island-engine** — event store, projections, keepers, signbook: canonical truth; sandbar.engine.client talks to it.
- **Mechanics / pine-research** — chronicle, spirit stick, hero actor: reference implementations; sandbar can call them or rehost later.
