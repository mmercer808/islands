# Islands — Single Ordered TODO

One list, correct order. Philosophy and decisions live in **MAINTENANCE.md**. Consolidation plan: **island_engine_consolidation_*.plan.md**.

---

## 0) Bootstrap (scaffold / cleanup)

- [ ] Decide working directory: chpt-island-engine as-is vs. copy to root (see MAINTENANCE.md).
- [ ] Initialize chpt-island-engine git (if not already), MIT license, first commit.
- [ ] Add CI (pytest) for chpt-island-engine.
- [ ] Optional: pre-commit (format + lint).
- [ ] Create _archive/; move scattered root scripts and legacy dirs per MAINTENANCE “Archive scope” decision.

---

## 1) Core library integration (Phase 1)

- [ ] Merge pine/core into island_engine (or wire as consumer): signals, context, serialization.
- [ ] Ensure serialization supports protocol envelope (chpt-island-engine protocol).
- [ ] Resolve pine vs matts: keep pine layout, pull any unique matts/generator_system behavior (see MAINTENANCE.md).

---

## 2) Orchestrator spine (Phase 2)

- [ ] Implement `/events/recent` in chpt-island-engine orchestrator.
- [ ] Add idempotency check and “seen message_id” cache.
- [ ] Implement projection rebuild: `/projection/canon`, `/projection/timeline/{player_id}`.

---

## 3) Agent runner spine (Phase 2)

- [ ] Add task polling loop (`/task/next`) or WS subscription.
- [ ] Implement local workspace output and PROPOSE_PATCH payload builder.
- [ ] Implement real LLM adapter (Ollama or llama.cpp); optionally use sandbar/llm_host (see MAINTENANCE “Sandbar vs agent_runner”).

---

## 4) Island bus — NATS (Phase 2, optional)

- [ ] Add optional NATS transport module.
- [ ] Define subjects and routing table.
- [ ] Request/reply for TASK_ASSIGN / TASK_RESULT.

---

## 5) Keepers (Phase 3)

- [ ] Define divergence signals: beat distance, contradiction score, pacing.
- [ ] Implement `detect_divergence()` in keepers/procedures/plot_gravity.py.
- [ ] Implement `propose_gravity_moves()` with at least 4 hook archetypes.
- [ ] Turn contradiction finder/repair stubs into working checks.

---

## 6) Book ingest (Phase 4)

- [ ] Define book_catalog.json schema.
- [ ] Write “catalog builder” agent template: chapter → scenes → entities → beats.
- [ ] Add import pipeline: raw text → catalog → canon events.
- [ ] Use Vetrellis (infocom-transfer) as first test.

---

## 7) Signbook

- [ ] Add `/tag` endpoint (signbook_server).
- [ ] Optional: signature registry and verify tools.
- [ ] Add “best chat leaderboard” data model (signatures + score + judge notes).

---

## 8) Sandbar (alternate path)

- [x] Extract llm_host from proc_streamer (AssistChannel, config, no Qt).
- [x] Implement sandbar/engine/client.py against orchestrator when endpoints exist.
- [x] Add sandbar/server (game_runner + FastAPI POST /message); engine/game_loop + messaging wired.
- [ ] Implement context-feeder thread; inject context into LLM session.
- [ ] Dual display: interface shows game→LLM and LLM→interface separately (proc_streamer / sandbar UI).
- [ ] Optional: aqua-based UI in sandbar/ui; launcher that uses llm_host + engine.
- [ ] See **sandbar/TODO.md** for full sandbar checklist.

---

## 9) White Room console (dev UX)

- [ ] CLI to view events, projections, timelines, open tasks (chpt-island-engine or sandbar).
- [ ] CLI to post signbook entries/tags.
- [ ] Decide: White Room = sandbar UI, or CLI, or both (see MAINTENANCE.md).

---

## 10) Repo maintenance

- [ ] Update CLAUDE.md: add chpt-island-engine, sandbar, consolidation order, link to MAINTENANCE.md and ISLANDS_TODO.md.
- [ ] Launcher: decide default entry (bootstrap, proc_streamer, sandbar) and add menu if multiple (see MAINTENANCE.md).
- [ ] Remove or redirect duplicate docs; point all “start here” to CLAUDE_HANDOFF + MAINTENANCE + ISLANDS_TODO.

---

**Reference:** chpt-island-engine **TODO.md** (authoritative for engine scope); **sandbar/TODO.md** (sandbar scope); **MAINTENANCE.md** (concepts and decisions).
