# TODO (island-engine)

> Rule: TODOs are actionable. Philosophy belongs in `docs/`.

## 0) Bootstrap
- [ ] Initialize git repo, add MIT license, first commit
- [ ] Add CI (pytest) workflow
- [ ] Add pre-commit (format + lint) (optional)

## 1) Orchestrator spine
- [ ] Implement `/events/recent` endpoint to read back recent events
- [ ] Add idempotency check endpoint and “seen message_id” cache
- [ ] Implement projection rebuild endpoints:
  - [ ] `/projection/canon`
  - [ ] `/projection/timeline/{player_id}`

## 2) Agent runner spine
- [ ] Add task polling loop (`/task/next`) or WS subscription
- [ ] Implement local workspace output + “PROPOSE_PATCH” payload builder
- [ ] Implement LLM adapter interface + one real backend (Ollama or llama.cpp server)

## 3) Island bus (NATS)
- [ ] Add optional NATS transport module
- [ ] Define subjects and routing table
- [ ] Implement request/reply for TASK_ASSIGN/TASK_RESULT

## 4) Keepers
- [ ] Define divergence signals: beat distance, contradiction score, pacing
- [ ] Implement `detect_divergence()` in `keepers/procedures/plot_gravity.py`
- [ ] Implement `propose_gravity_moves()` with at least 4 hook archetypes
- [ ] Implement contradiction finder/repair stubs into working checks

## 5) Book ingest (catalog)
- [ ] Define `book_catalog.json` schema
- [ ] Write “catalog builder” agent template: chapter → scenes → entities → beats
- [ ] Add import pipeline skeleton: raw text → catalog → canon events

## 6) Signbook integration
- [ ] Add `/tag` endpoint for quick notes
- [ ] Add signature registry (optional) + verify tools
- [ ] Add “best chat leaderboard” data model (signatures + score + judge notes)

## 7) White Room console (dev UX)
- [ ] CLI to view events, projections, timelines, open tasks
- [ ] CLI to post signbook entries/tags
