# Sandbar Integration

## When to use Sandbar vs. main consolidation

- **Main path (island_engine_consolidation):** Merge pine → island-engine, implement orchestrator/agent_runner/keepers in chpt-island-engine, run agent_runner as a separate process with its own LLM adapter.
- **Sandbar path:** Single app that loads the LLM (proc_streamer-derived), runs a **context-feeder thread** that gives the LLM event/canon/task context, and uses aqua-style UI. Ideal for dev UX, White Room console, or a “story IDE” that talks to the same engine.

Both paths consume **chpt-island-engine** as the source of truth. Sandbar does not replace the agent_runner; it can share the same LLM adapter (llm_host) once extracted.

---

## Thread model

```
┌─────────────────────────────────────────────────────────────────┐
│  Main thread (Qt / aqua)                                        │
│  • User input, editor, console, assistant panel                 │
│  • AssistChannel.query(prompt) → posts to LLM worker            │
│  • Receives chunks via Signal → updates UI                       │
└─────────────────────────────────────────────────────────────────┘
         │                                    ▲
         │ prompt                             │ chunks / complete
         ▼                                    │
┌──────────────────────┐           ┌──────────────────────────────┐
│  LLM worker thread   │           │  Context-feeder thread        │
│  • _run_query()      │           │  • Poll engine client         │
│  • HTTP stream       │           │  • Build context bundle       │
│  • Emit chunk/complete│          │  • Inject into LLM session   │
└──────────────────────┘           └──────────────────────────────┘
         │                                    ▲
         │ (optional)                         │
         └────────────────────────────────────┘
              context bundle (system prompt / prepended context)
```

---

## Link to consolidation plan

See repo root:

- **MAINTENANCE.md** — Islands-wide concepts and maintenance order.
- **ISLANDS_TODO.md** — Single ordered TODO list (bootstrap → core → services → keepers → ingest → signbook → sandbar).
- **island_engine_consolidation_*.plan.md** — Full assimilation strategy (pine, matts, infocom-transfer, archive).

Sandbar tasks are a **subset** of the overall TODO, ordered in **sandbar/TODO.md**.
