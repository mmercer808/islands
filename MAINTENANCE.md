# Islands — Maintenance and Concept Order

This file captures the **best concepts** to preserve, the **correct order** for work (aligned with the island engine consolidation plan), and **decisions you need to make**. Use **ISLANDS_TODO.md** for the single ordered task list.

---

## Best concepts (preserve and reuse)

1. **Canon is event-sourced** — If it’s not in the log, it didn’t happen. (chpt-island-engine, CLAUDE_HANDOFF)
2. **Agents propose; one writer commits** — No shared mutable truth. (chpt-island-engine)
3. **Story is infrastructure, not output** — Narrative coherence over local cleverness. (chpt-island-engine, complete_story_engine_system)
4. **Creation uses “descend”, not “spawn”** — Every arrival has cost, timing, consequence. (chpt-island-engine)
5. **Keepers redirect agency; they don’t erase it** — Plot gravity, not rails. (chpt-island-engine, keepers/procedures)
6. **Meta time and story time are separate** — Keeper reasoning offstage; effects scheduled fairly. (chpt-island-engine design)
7. **Deferred builder pattern** — Chains that suspend until conditions are met (e.g. `.doThis().dontForgetTo().fertilizeTree()`). (complete_story_engine_system, pine/matts)
8. **Self-modifying iterators** — Traversal that adapts to runtime conditions. (complete_story_engine_system, pine narrative)
9. **Serializable execution contexts** — Contexts that can be transmitted across systems (matts, pine/core).
10. **Signbook** — AI persistence: signatures and messages across sessions (chpt-island-engine signbook_server, pine/signbook).
11. **LLM in a worker thread; context from another thread** — proc_streamer loads the LLM; a separate context-feeder thread can supply events/canon/task (sandbar).
12. **Interface = proc_streamer; dual display** — LLM output goes to the interface only; game reports to the interface; interface shows both game→LLM and LLM→interface separately (sandbar).

These belong in docs and in the canonical engine (island-engine); sandbar and aqua consume the engine and apply #11–12.

---

## Correct order for work (high level)

Follow **island_engine_consolidation** plan order:

1. **Phase 0 — Scaffold / cleanup**  
   Working dir = **island-engine** at root (copy of chpt-island-engine). **island-engine_scaffold_v3** is the proposed structure and should emulate this layout. Archive scattered root files to `_archive/`.

2. **Phase 1 — Core library**  
   Merge pine/core (and selected matts) into island_engine; signals, context, serialization; protocol envelope support.

3. **Phase 2 — Services**  
   Orchestrator (event readback, idempotency, projections); agent_runner (task polling, real LLM adapter); optional NATS.

4. **Phase 3 — Keepers**  
   Divergence signals; `detect_divergence()` and `propose_gravity_moves()`; contradiction detection/repair.

5. **Phase 4 — Book ingest**  
   book_catalog schema; pipeline text → catalog → canon events; Vetrellis as first test.

6. **Signbook**  
   Tag endpoint; signature registry; leaderboard model (per chpt-island-engine TODO).

7. **Sandbar (alternate path)**  
   Extract LLM host from proc_streamer; context-feeder thread; engine thin client; optional aqua UI. See **sandbar/README.md** and **sandbar/TODO.md**.

8. **Archive / maintenance**  
   Move _old, Mechanics, island-engine_scaffold_v3, duplicate root scripts to _archive; update CLAUDE.md and launcher to point at canonical entry points.

Detailed tasks live in **ISLANDS_TODO.md** in this order.

---

## Decisions you need to make

1. **Working directory**  
   Work inside **chpt-island-engine/** as-is, or copy its structure to repo **root** as `island-engine/`? Plan says “use chpt-island-engine as primary working directory” or “copy structure to root”—pick one and stick to it.

2. **Pine vs matts**  
   Plan says “keep pine’s cleaner structure” and “check matts/generator_system.py for unique functionality.” Decide: full merge of pine into island_engine, or island_engine as consumer of pine + matts as separate libs?

3. **Sandbar vs agent_runner**  
   Should **agent_runner**’s real LLM adapter live inside chpt-island-engine and **call** sandbar/llm_host (or a shared package), or should sandbar stay fully separate and agent_runner get its own Ollama adapter copied from proc_streamer? Recommendation in sandbar: sibling under islands; agent_runner can depend on sandbar/llm_host or a shared `islands_llm` package.

4. **Archive scope**  
   Which root files go to _archive vs. stay runnable? Suggested archive: aqua.py, story_world_pyside6.py, proc_streamer_v1_6.py (after extraction to sandbar), other loose scripts. Keep: main.py, launcher, chpt-island-engine, pine, matts, sandbar, infocom-transfer, docs.

5. **Launcher**  
   Should **launcher/launcher.py** point to: (a) chpt-island-engine bootstrap, (b) proc_streamer, (c) sandbar UI, (d) all with a menu? Decide the “daily driver” and what the launcher starts by default.

6. **White Room console**  
   Is the “White Room console” the sandbar UI (aqua + LLM + context-feeder), or a separate CLI in chpt-island-engine (docs say “CLI to view events, projections, timelines”)? Could be both: CLI for inspection, sandbar for interactive story IDE.

Once these are decided, update this section and ISLANDS_TODO.md so the next session has a single source of truth.

---

## Integrating novel concepts (claude-add)

When you add **new ideas** to the collection — even if they require reworking existing code or docs — edits must **adhere to the truths we have to display** (the list above). Use this section as instructions to purpose your edits.

### Considerations

1. **Truths are non-negotiable** — The best concepts in § Best concepts (canon event-sourced, agents propose / one writer commits, story is infrastructure, descend not spawn, keepers redirect agency, deferred builder, serializable contexts, signbook, dual display) must not be contradicted. New concepts must **extend** or **refine** them, not replace or conflict.
2. **Single source of truth** — New behavior should live in one canonical place: island-engine for the event-sourced spine, sandbar for the message-driven server path, pine/matts for workhorse libs. Document in CONCEPT_SIGNATURES.md when adding new public APIs.
3. **Rework is allowed** — If a new idea requires reworking original ideas, do it by: (a) stating how the rework preserves or strengthens the truths above, (b) updating MAINTENANCE.md and CLAUDE.md so the next session sees the new intent, (c) keeping ISLANDS_TODO.md and sandbar/TODO.md in sync with remaining work.
4. **Collection, not pile** — The repo is a **collection**: each subsystem (island-engine, sandbar, pine, matts, infocom-transfer, claude-add) has a clear role. When adding a novel concept, decide which subsystem it belongs to; if it’s a new subsystem, add a short note in MAINTENANCE.md and CLAUDE.md under Key Systems / Architecture so the structure stays understandable.
5. **Display** — “Truths we have to display” means: what we show in docs, in CONCEPT_SIGNATURES, and in the behavior of the engine and UI (canon, dual display, signbook, etc.). New concepts must be **reflected** in those surfaces so the project’s intent stays visible.

### Instructions for edits (claude-add)

- **Before adding a new concept:** Check § Best concepts; ensure the new idea aligns with or extends them.
- **When reworking:** Explain in the commit or in MAINTENANCE.md how the rework preserves the truths; update CLAUDE.md if the “how we work” story changes.
- **After adding:** Update CONCEPT_SIGNATURES.md if you add or change public APIs; update ISLANDS_TODO.md or sandbar/TODO.md if new tasks appear.
- **Reference:** Point “how to add new ideas” to this section and to **CLAUDE.md § Adding new concepts (claude-add)**.
