# Sandbar TODO (ordered)

Sandbar-specific tasks. Islands-wide order and consolidation live in repo root **MAINTENANCE.md** / **ISLANDS_TODO.md**.

---

## 1) LLM host extraction

- [ ] Create `sandbar/llm_host/` package.
- [ ] Extract from `proc_streamer_v1_6.py`: `AssistChannel`, Ollama/OpenAI-compat streaming, healthcheck, worker-thread `_run_query`.
- [ ] Put in `llm_host/assist_channel.py` and `llm_host/config.py` (no Qt dependency in llm_host).
- [ ] Add minimal tests or script that runs a single query without GUI.

---

## 2) Engine thin client

- [ ] Create `sandbar/engine/` package.
- [ ] Implement `engine/client.py`: stubs or HTTP for `get_recent_events(limit)`, `get_canon_slice()`, `get_current_task()`.
- [ ] When island-engine has `/events/recent` and `/projection/canon`, wire client to those endpoints; until then, stub or in-process read from event store.

---

## 3) Dual display (game→LLM and LLM→interface)

- [ ] Interface shows **two streams separately** at the same time: (1) **game→LLM** — context/results the game sent to the LLM; (2) **LLM→interface** — LLM response to the user.
- [ ] All output goes through the interface; game reports back to the interface with results (no LLM output to the game).
- [ ] Wire game result callbacks into the interface so the "game→LLM" copy is visible; keep existing LLM stream as "LLM→interface".

---

## 4) Context-feeder thread

- [ ] Add `llm_host/context_feeder.py`.
- [ ] Thread loop: periodically (or on event) call `engine.client` to build context bundle (recent events + canon slice + current task).
- [ ] Define how the bundle is injected: e.g. system prompt update, or prepended to next user message, or dedicated “context” API on AssistChannel.
- [ ] Start/stop feeder with AssistChannel connect/disconnect so the LLM always has fresh context when answering.

---

## 5) Optional: agent_runner LLM adapter from llm_host

- [ ] Implement island-engine `services/agent_runner/llm_adapters/ollama.py` (or similar) that uses `sandbar/llm_host` for actual Ollama calls, so agent_runner and sandbar share one implementation.
- [ ] Decision: sandbar as sibling of chpt-island-engine vs. sandbar inside chpt-island-engine. Recommendation: sibling under islands; agent_runner depends on sandbar/llm_host or a copied adapter.

---

## 6) UI integration (aqua + sandbar)

- [ ] Create `sandbar/ui/` and a launcher that instantiates aqua-style document/dock/prompt stage.
- [ ] Wire launcher to `llm_host.AssistChannel` and `llm_host.ContextFeeder` (start feeder when LLM connects).
- [ ] Keep proc_streamer_v1_6.py runnable as-is for comparison; sandbar UI is the alternate front-end that shares llm_host and engine client.

---

## 7) Docs

- [ ] Write `sandbar/docs/integration.md`: thread diagram (UI, LLM worker, context-feeder), when to use sandbar vs. main consolidation path, link to island_engine_consolidation plan.
