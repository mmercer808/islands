# Sandbar Search Report

**Purpose:** A **server that responds to any message** and **runs the game**. The server is the sandbar — a lighter, message-driven island. This report summarizes what exists in **oasis/engine**, **Mechanics**, **island-engine** (and **island-engine_scaffold_v3**, **chpt-island-engine** as proposed structure), **pine**, and **pine-research**, and how it maps onto sandbar.

---

## 1. Main idea: server responds to any message, server runs the game

- **Sandbar** = the server surface: receive message → run game (or delegate to island-engine) → respond.
- **Island-engine** (main dir + scaffold_v3 + chpt-island-engine) = proposed canonical structure: orchestrator (event store, projections), agent_runner, keepers, signbook, protocol. Sandbar consumes it or mirrors its API.
- **Game loop** lives somewhere the server can drive: either inside sandbar (oasis/Mechanics patterns) or in island-engine (orchestrator + tasks).

---

## 2. oasis/engine — game loop, context, message flow

**Files:** `persistent_context_iterator.py`, `narrative_chain_iterator_system.py`

### Persistent context iterator

- **ContextWindow** — sliding window of `ContextOperation` (GAME_ACTION, STORY_EVENT, etc.); add/purge by time or size.
- **PersistentContextIterator** — rolling context passed through execution; `add_operation()`, `main_loop_cycle()`.
- **GameLoopIntegrator** — drives the game:
  - `start_game_loop()` → `while is_running: _game_loop_frame()`.
  - Each frame: `context_iterator.main_loop_cycle()`, process event queues with context, schedule periodic tasks, frame-rate limit.
- **TaskScheduler** — threaded tasks; can schedule work that uses the context iterator.

**Relevance to sandbar:** “Server runs the game” can be implemented as: **incoming message → push into event queue → game loop frame processes queue with context → response from context/result**. Oasis already has “event queue + context + game loop frame”; sandbar could host a thin server that pushes messages into that loop and returns results.

### Narrative chain iterator system

- **ProcessControlManager** — singleton; “takes control” of process for story execution; registries for threads, chains, handlers, event queues.
- **ChainIterator** — chain of links; `advance()`, `get_current_link()`, branch conditions.
- **CrossChainHandler** — handlers that span chains; `handle_link_execution()`, trigger conditions, execution actions.
- **Event queue registry** — event queues processed in story execution thread.

**Relevance to sandbar:** Message → map to chain/link or event queue → process in story thread → response. Good fit for “respond to any message” by routing into the right chain or queue.

### oasis/messaging

- **MessageBuffer**, **MessagingInterface** — `receive()`, `receive_prompt()`, buffer push/pop; **IOBridge** message → signal; **ConnectorBridge** to other systems.
- **GameOrchestrator** (in integration_layer) — `messaging.receive_prompt()` → `update()` → bridge to signals → listener route → ThinkingGraph; `command()`, `inject_llm_result()`.

**Relevance to sandbar:** “Respond to any message” = receive into MessagingInterface (or equivalent), run `update()` (or one game-loop step), return output from buffer or from graph/orchestrator. Sandbar server = HTTP/WS adapter over this receive/update/respond path.

---

## 3. Mechanics — handlers, chronicle, spirit stick, traversal

**Key locations:** `Mechanics/everything/`, `Mechanics/hero_quest_chronicle.py`, `pine-research/mechanics/`

### Hero Quest Chronicle (Mechanics + pine-research)

- **Message** — typed (COMMAND, SPEECH, QUERY, NARRATION, STATE_CHANGE, etc.); immutable.
- **Chronicle** — SQLite-backed log of messages; “the stick remembers.”
- **MessagingInterface** (ABC) — `receive() -> Optional[str]`, `has_message()`; **StdinInterface**, **QueueInterface** (for tests/simulation).
- **HeroActor** — holds quest state; **handlers** per message type; receive message → dispatch to handler → update state / chronicle → response.

Architecture in docstring: **MessagingInterface → MESSAGE QUEUE → SPIRIT STICK (event loop) + SQLite (chronicle) → HERO ACTOR (handlers) → QUEST STATE**.

**Relevance to sandbar:** “Server responds to any message” maps directly: **server receives string → push into QueueInterface (or equivalent) → hero/event loop processes → response from chronicle or handler output**. Sandbar server = message-in, message-out; game = chronicle + hero + spirit stick.

### Mechanics/everything — signals, entities, traversal

- **signals.py** — ObserverBus, `respond()`, `handle(signal)`; multiple handler types.
- **my_infocom_entity_system.py** — `handle(verb, context)`, `process(action, game_state)`; ActionProcessor.
- **callback_patterns.py** — GraphWalker + handlers (pre_visit, post_visit, etc.); traversal as handler chain.
- **oracle_tower.py** — `process_command(raw_input) -> ResolverOutput`; observers process ResolverInput/Output.
- **spirit_stick.py** — token passing; `receive_stick()`; turn/round.
- **traversing_spirit_stick.py**, **unified_traversal_lookahead.py** — traversal + predictions; `process(prediction, ...)`.

**Relevance to sandbar:** Handlers and `process()`/`handle()` are the “respond” side. Sandbar server turns a message into a signal or command, runs one or more of these, returns result.

---

## 4. island-engine (main + scaffold_v3 + chpt-island-engine) — proposed sandbar structure

**Canonical layout:** services/orchestrator, agent_runner, signbook_server; keepers; protocol; world; docs.

### Orchestrator (server-like)

- **POST /event** — accepts `Envelope` (message_id, ts, from_, to, type, thread, payload); `validate_event()`; `store.append()` (event store). Single writer; idempotent by message_id.
- **EventStore** — append-only; `list_recent(limit)` (no `/events/recent` HTTP yet).
- **projections** — `rebuild_canon(events)`, `rebuild_timeline(events, player_id)` (in code; no HTTP yet).

**Relevance to sandbar:** “Server responds to any message” can be: **client sends envelope (or message) → orchestrator validates and appends event → runs projections / game step → returns result or ack**. Sandbar can be the HTTP/WS front that forwards to orchestrator and optionally runs a local game loop (oasis/Mechanics) for “run the game.”

### Agent runner, signbook

- **agent_runner** — worker pool; task polling (TODO); **LocalStubLLM** — `complete(prompt)` (stub). Real LLM here or in sandbar llm_host.
- **signbook_server** — FastAPI; add_entry(signature, message, …), list_entries, search. Append-only log.

**Relevance to sandbar:** Sandbar can call orchestrator + signbook; agent_runner (or sandbar) runs the “game” or tasks that consume events and produce responses.

---

## 5. pine — workhorse functions to reuse

- **narrative/deferred_builder.py** — `DeferredStoryBuilder`: `doThis()`, `thenThis()`, `dontForgetTo()`, `whenReady()`, `finallyThis()`, `execute(context)`; `get_reminders()`, `get_blockers()`, `progress`. Condition-driven story steps.
- **narrative/world.py** — `create_world()`, `WhiteRoomBuilder` (location, item, character, connect, origin, build); `text_to_world()`.
- **narrative/traversal.py** — `smart_iter`, TraversalContext, layer registry; pre/post callbacks.
- **narrative/lookahead.py** — `lookahead_from()`, Possibility, actions/discoveries/transitions, hints.
- **narrative/extraction.py** — `extract_text()` → ExtractionResult (entities, relations, fragments).
- **messaging/interface.py** — MessageBuffer (async push/pop), BufferedIO (read/write/receive); SignalFactory (story_event, context_updated).
- **messaging/connector.py** — Connector, register listener, set_route, route(payload) → count delivered.
- **runtime/live_code.py** — serialize/deserialize function; execute with context.
- **runtime/hotswap.py** — swap handlers at runtime.
- **graph/** — nodes, edges, walker (walk_bfs, walk_dfs, find_path), embedding store.

**Relevance to sandbar:** Use these inside the “run the game” path: e.g. receive message → build or update world (pine world/traversal) → run deferred builder or lookahead → format response. Pine stays the workhorse; sandbar is the server that calls it.

---

## 6. pine-research — concepts and solutions

- **mechanics/** — hero_quest_chronicle, oracle_tower, plot_equilibrium, traversing_spirit_stick, unified_traversal_lookahead, stream_processor. Same ideas as Mechanics with experimental twists.
- **prototypes/** — callback_patterns, integration_example (traversal wrapper + IslandGraph).
- **worlds/vetrellis/** — Vetrellis content (Archon, gazetteer, Infocom-style); good test content for “any message” (e.g. look, go, take).
- **archives/** — complete_story_engine_system, MMO_ARCHITECTURE, GAME_VS_API_SEPARATION, DESIGN_DOCUMENT.

**Relevance to sandbar:** Hero-quest chronicle + message queue + spirit stick is a ready “server runs the game” pattern. Oracle/plot_equilibrium/traversal give “interesting” responses. Sandbar can adopt chronicle + QueueInterface + HeroActor (or a slim variant) as the default game loop for “respond to any message.”

---

## 7. How it fits the proposed sandbar structure

Proposed structure (from island-engine, island-engine_scaffold_v3, chpt-island-engine) under **sandbar/**:

- **sandbar/llm_host** — LLM connection; optional for “respond to any message” (can be text-only at first).
- **sandbar/engine** — thin client to island-engine (events, canon, timeline) + **engine/messaging** (connector, MessagingInterface, GameOrchestrator already assimilated).
- **sandbar/ui** — launcher, proc_streamer/legacy; dual display (game→LLM, LLM→interface) is separate.

**Missing piece for “server that responds to any message and runs the game”:**

1. **Server entrypoint** — e.g. FastAPI or WS: one endpoint (e.g. `POST /message` or `POST /event`) that accepts a message (or envelope).
2. **Game runner** — either:
   - **Option A:** Push message into oasis-style event queue + game loop (GameLoopIntegrator + PersistentContextIterator), run one or N frames, return context/output; or
   - **Option B:** Push message into Mechanics/pine-research style chronicle + HeroActor (or QueueInterface + handler chain), process one message, return response; or
   - **Option C:** Forward to island-engine orchestrator (POST /event), then run projections and return projection slice or task result.
3. **Response** — from context window (oasis), chronicle/handler (Mechanics), or orchestrator/projection (island-engine).

**Recommendation:** Add **sandbar/server** (or **sandbar/game_runner**) that:
- Receives one message (string or envelope).
- Pushes into a single “game” abstraction: either oasis GameLoopIntegrator + context, or chronicle + HeroActor, or orchestrator client.
- Runs one step (one frame or one message through handlers).
- Returns a response (text or JSON).  
Then the “island” is island-engine; the **sandbar** is this thin server + existing sandbar UI/llm_host/engine, so “server responds to any message, server runs the game.”

---

## 8. Summary table

| Area            | What it provides for “respond to any message” / “run the game”     | Sandbar use |
|-----------------|---------------------------------------------------------------------|-------------|
| **oasis/engine**| Game loop frame, context window, event queues, task scheduler       | Drive game from server; one message → one or more frames; response from context |
| **oasis/messaging** | receive_prompt, update(), bridge to signals, GameOrchestrator  | Receive message → update() → respond from buffer/graph |
| **Mechanics**   | Chronicle, MessagingInterface, HeroActor, handlers, spirit stick    | Message → queue → hero/handlers → chronicle → response |
| **pine**        | DeferredStoryBuilder, world, traversal, lookahead, messaging, graph| Workhorse for story step, world, and traversal in response path |
| **pine-research** | Same + Vetrellis content; integration_example                    | Content and patterns for “any message” (look, go, take) |
| **island-engine** | POST /event, event store, projections, signbook, protocol        | Canonical truth; sandbar can forward events and/or run local game that syncs to it |

**Main purpose again:** A **server that responds to any message** and **runs the game**. The server is the sandbar — a lighter, message-driven island that can use oasis (game loop + context), Mechanics (chronicle + hero), pine (world/traversal/deferred), and island-engine (orchestrator/protocol) under one roof.
