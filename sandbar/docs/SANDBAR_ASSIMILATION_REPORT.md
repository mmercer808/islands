# Sandbar Assimilation Report

**Purpose:** All relevant “good code” for the sandbar message-driven server lives under **sandbar/**. This report lists what was copied, where, and how to use it. It also states what you have to do next and calls out ideas worth keeping.

**Report location:** `sandbar/docs/SANDBAR_ASSIMILATION_REPORT.md` (this file).

---

## 1. What was copied and where

| Destination under sandbar/ | Source | Contents |
|----------------------------|--------|----------|
| **pine/** | Repo root **pine/** | Full tree: core (signals, context, primitives, serialization), narrative (deferred_builder, world, traversal, lookahead, extraction, spirit_stick), runtime (live_code, hotswap, bytecode, generators), graph (nodes, edges, walker, embedding), messaging (connector, interface, integration), signbook, config, ui. No import renames; internal imports are `pine.*`. |
| **mechanics/** | Repo **Mechanics/** | hero_quest_chronicle.py, spirit_stick.py, traversing_spirit_stick.py, stream_processor.py, DESIGN_DOCUMENT.md. |
| **mechanics/everything/** | **Mechanics/everything/** | All .py: callback_patterns, my_infocom_entity_system, oracle_tower, signals, hero_quest_chronicle, spirit_stick, traversal, world, lookahead, extraction, embedding, graph_core, graph_walker, narrative, primitives, plot_equilibrium, traversing_spirit_stick, unified_traversal_lookahead, integration_example, matts_*, etc. |
| **runtime/** | Repo root | live_code_system.py, runtime_hotswap_system.py, context_serialization.py. |
| **matts/** | Repo **matts/** | signal_system.py, context_system.py, context_serialization.py, functional_interface.py. |
| **vetrell/** | Repo **vetrell/** | Full tree: analysis (code_object), language (english_code_object), narrative (context_iterator), runtime (hotswap, live_code), serialization (context). |
| **docs/complete_story_engine_system.md** | Repo root **complete_story_engine_system.md** | Design doc: self-modifying story engine, deferred builder, LLM integration. |

**Already in sandbar (unchanged):** engine/ (client, game_loop, messaging), server/ (app, game_runner), llm_host/, story/core.py, ui/launcher, docs/ (ASSIMILATION, CLEANUP_PLAN, GAME_VS_API_SEPARATION, MMO_ARCHITECTURE, integration, main_structure, SANDBAR_SEARCH_REPORT).

---

## 2. How to use the assimilated code

- **Pine (deferred builder, world, traversal, lookahead):**  
  From repo root: `from sandbar.pine.narrative import DeferredStoryBuilder, text_to_world, lookahead_from`.  
  From inside sandbar (e.g. server): ensure project root is on `PYTHONPATH` so `sandbar` is importable.  
  To make **pine** resolve to sandbar’s copy (so `import pine` uses sandbar/pine), run with `PYTHONPATH` including the sandbar directory (e.g. `cd sandbar; PYTHONPATH=. python -m server.app` or equivalent).

- **Mechanics (chronicle, spirit stick, handlers):**  
  `from sandbar.mechanics import Chronicle, MessagingInterface, HeroActor` or `from sandbar.mechanics.hero_quest_chronicle import ...`.  
  Use for “message → queue → chronicle → HeroActor handlers → response” in the server.

- **Runtime (live code, hot-swap, context serialization):**  
  `from sandbar.runtime import live_code_system` (or import the modules directly).  
  Use for runtime code injection and compressed context transmission.

- **Matts (signals, context, serialization):**  
  `from sandbar.matts import signal_system, context_system, context_serialization, functional_interface` (or import the modules).  
  Use for serializable execution contexts and event-driven communication; pine/core already embeds matts-derived ideas.

- **Vetrell (language, narrative, serialization):**  
  `from sandbar.vetrell.language import ...`, `from sandbar.vetrell.narrative import ...`, etc.  
  Use for English code objects, context iterators, and serialization patterns.

- **Design doc:**  
  `sandbar/docs/complete_story_engine_system.md` — reference for deferred builder, self-modifying iterators, and story engine architecture.

---

## 3. Double-check (what was verified)

- **pine/** — Full copy; **internal imports changed to relative** (`.core`, `.narrative`, etc.) so that `from sandbar.pine.narrative import DeferredStoryBuilder, text_to_world` loads sandbar’s pine, not the repo’s pine. **WorldNode** dataclass fixed: `name: str = ""` so it does not follow a default from base `Identified` (Python 3.10+ dataclass order).
- **mechanics/** — hero_quest_chronicle, spirit_stick, traversing_spirit_stick, stream_processor + everything/*.py; mechanics/__init__.py exposes MessageType, Message, Chronicle, MessagingInterface, HeroActor from hero_quest_chronicle.
- **runtime/** — Root live_code_system, runtime_hotswap_system, context_serialization present; __init__.py lists them.
- **matts/** — signal_system, context_system, context_serialization, functional_interface present; __init__.py lists them.
- **vetrell/** — Full tree copied; structure intact.
- **docs/complete_story_engine_system.md** — Present in sandbar/docs.
- **engine, server, llm_host, story, ui** — Unchanged; game_runner still imports from sandbar.engine.game_loop.

---

## 4. What you have to do

1. **Clean up the repo**  
   The root directory is messy. Use **sandbar/docs/CLEANUP_PLAN.md**: create `_archive/`, move _old, Mechanics, island-engine_scaffold_v3, and agreed root scripts into _archive; update CLAUDE.md, MAINTENANCE.md, launcher, and any imports. Prefer archive over delete.

2. **Dual display (interface)**  
   Implement the UI so both streams are visible: **(1) game → LLM** (context/results sent to the LLM) and **(2) LLM → interface** (LLM response to the user). Sandbar llm_host (callbacks) and engine client + messaging are in place; the remaining work is layout and wiring in the interface (e.g. proc_streamer or aqua).

3. **Context-feeder thread**  
   Add a thread that polls island-engine (orchestrator / event store), builds a context bundle, and pushes it into the LLM session; game results also to the interface for the “game→LLM” display.

4. **Imports and PYTHONPATH**  
   When you run the server or UI from repo root, PYTHONPATH usually includes the repo root so `sandbar` is importable. If you want `pine` to resolve to sandbar’s copy (sandbar/pine), run with PYTHONPATH including the sandbar directory, or always use `from sandbar.pine...` in sandbar code.

5. **Optional: thin facades**  
   You can add thin sandbar modules that re-export the “blessed” APIs (e.g. `sandbar.story.deferred` → `sandbar.pine.narrative.DeferredStoryBuilder`, `sandbar.patterns.chronicle` → `sandbar.mechanics.hero_quest_chronicle`) so the rest of sandbar depends on a single API surface.

---

## 5. Good ideas worth keeping (high level)

- **Canon is event-sourced** — If it’s not in the log, it didn’t happen (island-engine, MAINTENANCE).
- **Hero’s spirit stick / chronicle** — “The stick remembers what the hero forgets.” Message→queue→chronicle→HeroActor; SQLite-backed log; clear separation of MessagingInterface, Chronicle, HeroActor (mechanics/hero_quest_chronicle).
- **Deferred builder pattern** — Chains that suspend until conditions are met (e.g. `.doThis().dontForgetTo().fertilizeTree()`); pine/narrative/deferred_builder.py.
- **Spirit stick (token passing)** — Whoever holds the stick may speak; iterator-as-event-order; turn-based narrative (Mechanics/spirit_stick, pine/narrative/spirit_stick).
- **Oracle tower / observer processing** — ResolverInput/ResolverOutput with observers; good for “process command → observers react” (mechanics/everything/oracle_tower).
- **Callback patterns / graph walker** — Pre/post visit, handler chains; traversal as handler chain (mechanics/everything/callback_patterns).
- **Infocom-style entity system** — handle(verb, context), process(action, game_state); ActionProcessor (mechanics/everything/my_infocom_entity_system).
- **Serializable execution contexts** — Contexts that can be transmitted across systems (matts, pine/core, root context_serialization).
- **Dual display** — Interface shows both game→LLM and LLM→interface; no hidden pipes (sandbar README, MAINTENANCE).
- **Signbook** — AI persistence: signatures and messages across sessions (pine/signbook, island-engine).

These align with the truths in MAINTENANCE.md and CLAUDE.md; when you clean up, keep code that implements or supports these ideas and archive or drop the rest.

---

## 6. Where things live (quick reference)

```
sandbar/
├── engine/          # game_loop, messaging, client (thin client to island-engine)
├── server/          # game_runner, FastAPI POST /message
├── llm_host/        # AssistChannel, config
├── story/           # core.py stubs (StoryNode, EntityNode, WorldState, StoryEventManager)
├── ui/              # launcher (proc_streamer, legacy, minimal)
├── pine/            # FULL COPY: core, narrative, runtime, graph, messaging, signbook, config, ui
├── mechanics/       # hero_quest_chronicle, spirit_stick, traversing_spirit_stick, stream_processor; everything/*
├── runtime/         # live_code_system, runtime_hotswap_system, context_serialization (from root)
├── matts/           # signal_system, context_system, context_serialization, functional_interface
├── vetrell/         # language, narrative, runtime, serialization, analysis
└── docs/            # complete_story_engine_system.md, SANDBAR_SEARCH_REPORT, CLEANUP_PLAN, this report, etc.
```

---

**End of report.** For consolidation and cleanup order, see **MAINTENANCE.md** and **ISLANDS_TODO.md** at repo root. For how to add new concepts without breaking truths, see **claude-add/ADDING_NEW_CONCEPTS.md** and **MAINTENANCE.md § Integrating novel concepts (claude-add).**
