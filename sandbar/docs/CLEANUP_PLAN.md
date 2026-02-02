# Sandbar / Repo Cleanup Plan

**Purpose:** Reduce duplicates and "ideas not worth adding" while keeping one canonical path: **island-engine** (main), **sandbar** (alternate message-driven server), **pine** + **matts** (workhorse libs). Use **SANDBAR_SEARCH_REPORT.md** and **CONCEPT_SIGNATURES.md** (root) as reference.

---

## 1. Truths to preserve (do not contradict)

Before archiving or deleting, ensure no unique implementation of these is lost:

- **Canon is event-sourced** — If it's not in the log, it didn't happen.
- **Agents propose; one writer commits** — No shared mutable truth.
- **Story is infrastructure, not output** — Narrative coherence over local cleverness.
- **Creation uses "descend", not "spawn"** — Every arrival has cost, timing, consequence.
- **Keepers redirect agency; they don't erase it** — Plot gravity, not rails.
- **Deferred builder pattern** — Chains that suspend until conditions are met.
- **Serializable execution contexts** — Contexts that can be transmitted (matts, pine).
- **Signbook** — AI persistence: signatures and messages across sessions.
- **Interface = proc_streamer; dual display** — Game→LLM and LLM→interface shown separately.

See **MAINTENANCE.md** § Best concepts; **Integrating novel concepts (claude-add)** for how new ideas must adhere to these.

---

## 2. Archive (move, do not delete)

Move to **`_archive/`** so history stays available; update imports and docs that point here.

| Item | Reason |
|------|--------|
| **`_old/`** | Already archived; keep under _archive/ for consistency (e.g. _archive/old/). |
| **`Mechanics/`** (root) | Duplicate of patterns now in sandbar + island-engine + pine; hero_quest_chronicle, spirit_stick, oracle_tower patterns documented in SANDBAR_SEARCH_REPORT. |
| **`island-engine_scaffold_v3/`** | Proposed structure; layout emulated in island-engine; keep as reference in _archive. |
| **Root scripts:** `aqua.py`, `story_world_pyside6.py`, `proc_streamer_v1_6.py` (after sandbar extraction), `simple_word_processor.py`, `unified_console_tab.py`, `proc_streamer_legacy_ui.py` (if launcher uses sandbar/ui) | Loose runnables; canonical entry = launcher + sandbar/ui + island-engine. |
| **`files(8)/`** | Legacy; graph_core, graph_walker, narrative, callback_patterns — pine has equivalents. |
| **`inage-text/`** | Typo name; infocom content lives in infocom-transfer. |
| **`oasis/`** | Game loop + messaging assimilated into sandbar/engine (game_loop, messaging); keep copy in _archive for reference. |

**Do not archive yet (decide first):** `pine-research/` (Vetrellis content, mechanics experiments), `vetrell/`, `loopback/` (separate project). Leave until Archive scope decision in MAINTENANCE.md is finalized.

---

## 3. Delete (only after verification)

Do **not** delete until the corresponding canonical implementation is confirmed and any unique behavior is merged or documented.

| Candidate | Condition |
|-----------|-----------|
| **Root copies of sandbar-assimilated files** | e.g. `persistent_context_iterator.py`, `narrative_chain_iterator_system.py` at repo root — sandbar uses `sandbar/engine/game_loop/`. Only delete after sandbar and any root scripts that still import from root are updated to import from sandbar (or oasis in _archive). |
| **Duplicate docs** | Multiple SIGNBOOK_ARCHIVE.md, MMO_ARCHITECTURE.md, GAME_VS_API_SEPARATION.md — keep one canonical location (e.g. sandbar/docs or island-engine/docs), remove duplicates after links updated. |

**Rule:** Prefer **archive** over **delete**. Delete only when the same content exists in the canonical location and nothing references the duplicate.

---

## 4. Keep (canonical locations)

| Area | Role |
|------|------|
| **island-engine/** | Main event-sourced engine: orchestrator, event store, projections, agent_runner, keepers, signbook_server, protocol. |
| **chpt-island-engine/** | Original source; island-engine at root is main working copy. |
| **sandbar/** | Alternate path: server (game_runner, app), engine (game_loop, messaging, client), llm_host, story stubs, ui launcher. |
| **pine/** | Workhorse: narrative (deferred_builder, world, traversal, lookahead), messaging, runtime (live_code, hotswap), graph. |
| **matts/** | Serializable contexts, signals, generator patterns; keep pine layout, pull unique matts behavior into island-engine or pine per MAINTENANCE. |
| **infocom-transfer/** | Vetrellis / Infocom-style content. |
| **launcher/**, **main.py** | Entry points; launcher default per MAINTENANCE. |
| **CLAUDE.md**, **MAINTENANCE.md**, **ISLANDS_TODO.md**, **CONCEPT_SIGNATURES.md** | Single source of truth for concepts, order, and signatures. |
| **claude-add/** | Signbook MCP + integration principles (adding new concepts). |

---

## 5. Execution order

1. **Create `_archive/`** at repo root (if not present).
2. **Move** _old, Mechanics, island-engine_scaffold_v3, and agreed root scripts into _archive (e.g. _archive/old, _archive/Mechanics, _archive/island-engine_scaffold_v3, _archive/scripts).
3. **Update** CLAUDE.md, MAINTENANCE.md, launcher, and any imports that referenced moved paths.
4. **Remove** root duplicate game_loop files only after sandbar (and launcher) are confirmed to use sandbar/engine/game_loop.
5. **Consolidate** duplicate docs into sandbar/docs or island-engine/docs; delete or redirect duplicates.
6. **Re-run** tests / smoke checks; update ISLANDS_TODO.md and sandbar/TODO.md.

---

## 6. Reference

- **SANDBAR_SEARCH_REPORT.md** — What each area (oasis, Mechanics, pine, island-engine) provides and how sandbar uses it.
- **CONCEPT_SIGNATURES.md** (root) — Function signatures; use when merging or reimplementing.
- **MAINTENANCE.md** § Archive scope, § Integrating novel concepts (claude-add).
