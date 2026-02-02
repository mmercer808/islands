# Island Engine Design
**Version:** 0.1.0 (design + scaffolding)  
**Date:** 2026-02-01

## 1) What you're building

A **local-first, multi-agent** story engine that:
- ingests a **book** (as a cataloged canon)
- runs **parallel player timelines** that can intersect
- uses **keepers** (in-world NPCs) to apply **plot gravity** so player timelines converge back to the book plot
- allows agents to continuously create content + code, without shared-memory chaos

The engine is not turn-based; it is **adaptive real-time**:
- players can act at any moment
- the system schedules keeper interventions and story updates
- canon alignment is maintained via validators + plot gravity

## 2) Methodology (how we build without losing the plot)

### 2.1 Event-sourced truth
- The system records append-only **events**.
- “Canon” and “timelines” are **projections** (derived views).
- Agents produce **proposals**; the orchestrator commits events.

Why:
- transitory context windows (LLMs forget) → the event log remembers
- easy replay/debug/branch/merge
- concurrency becomes ordering + validation, not shared state

### 2.2 Actor model over the network
- Each agent is an actor with a mailbox.
- Message passing uses **JSON envelopes**.
- Machines can join/leave; idempotency protects against retries.

### 2.3 Two buses, two temps
- **Island bus (real-time)**: high-throughput agent tasks and world proposals.
- **Signbook service (append-only)**: persistent signatures, notes, discoveries.

Keepers also run in **meta time**:
- they maintain unsynced internal queues
- they emit proposals that the orchestrator schedules into story time

## 3) Requirements (from your concept)

### 3.1 Story / world requirements
- Canon originates from a book ingest + catalog
- Every player has a timeline (branch) that may diverge
- Divergence is allowed, but **must converge** to the book plot
- Players may join other timelines; merges must preserve plot integrity
- The “White Room” is a hub for creation + dispatch into the world

### 3.2 System requirements
- Multi-machine support (old laptop/mac/pi + future nodes)
- Local LLM integration per runner
- Message protocol is JSON-first (Python dict-friendly)
- Validator layer for plot constraints + contradictions
- A merge gate that can accept/reject/modify proposals
- Append-only signbook entries and tags

### 3.3 Agent requirements
- Agents can work on code and content
- Agents can promote architecture changes “up the chain”
- Personality emerges via memory artifacts and role constraints
- Keepers must exist as **characters** and **procedures**

## 4) Key abstractions

### 4.1 Event log
All meaningful changes are events:
- TASK_ASSIGNED, TASK_RESULT
- WORLD_UPSERT_PROPOSED, WORLD_UPSERT_COMMITTED
- TIMELINE_BRANCH_CREATED, TIMELINE_MERGE_REQUESTED
- PLOT_GRAVITY_PROPOSED, PLOT_GRAVITY_APPLIED
- CONTRADICTION_DETECTED, CONTRADICTION_REPAIRED

### 4.2 Projections
- Canon projection: stable book facts + accumulated commits
- Timeline projection: canon + player deltas
- Index projection: search indices, tag indices, etc.

### 4.3 Keepers
A keeper has two “bodies”:
1) In-world NPC sheet (voice, memory, location)
2) System role (validator + plot gravity proposer)

Every keeper action links story layer + system layer via `cause_id`.

## 5) Message protocol (envelope)
Transport-agnostic JSON envelope (NDJSON friendly):

```json
{
  "v": 1,
  "message_id": "uuid",
  "ts": "ISO8601Z",
  "from": "agent_runner@boxA/keeper_01",
  "to": "orchestrator",
  "type": "TASK_RESULT",
  "thread": "task:123",
  "payload": {}
}
```

Rules:
- idempotent handling (drop duplicates by message_id)
- “at least once” delivery is acceptable

## 6) Development phases

### Phase 0 (this repo)
- repo scaffold + protocol + TODOs
- basic services skeletons and method stubs
- signbook server: running FastAPI service

### Phase 1 (spine)
- orchestrator event store (SQLite)
- runner worker pool
- task assign/result loop (HTTP or WS)
- projections rebuild

### Phase 2 (island bus)
- NATS subjects and routing
- keeper procedure: divergence → plot gravity
- timeline deltas and merge recommendations

### Phase 3 (book ingest)
- book catalog schema
- entity extraction pipeline (agents assist)
- plot beat constraint set

## 7) “Hard things” we write down early (guidelines)
- Canon is a projection, not a hand-edited file
- Keepers propose; orchestrator commits
- Any agent may propose architecture changes, but merges are gated
- Story coherence beats local cleverness (validators are law)
