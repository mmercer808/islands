# CLAUDE HANDOFF — ISLAND ENGINE PROJECT

Generated: 2026-02-02T01:07:59.011884Z

This document is a **single-source handoff** intended to be given verbatim to Claude
(or any other coordinating LLM). It contains rules, truths, links, and the active TODOs.
No interpretation is required. No context outside this file is assumed.

---

## 1. CORE TRUTHS (NON‑NEGOTIABLE)

1. **Canon is event‑sourced.**  
   If it is not in the log, it did not happen.

2. **Agents propose; one writer commits.**  
   Never introduce shared mutable truth.

3. **Story is infrastructure, not output.**  
   Narrative coherence outranks local cleverness.

4. **Creation is constrained.**  
   Use *descend*, not *spawn*. Every arrival has cost, timing, and consequence.

5. **Keepers are not gods.**  
   They redirect agency; they do not erase it.

6. **Divergence is allowed.**  
   Convergence is inevitable.

7. **Meta time and story time are separate.**  
   Keeper reasoning happens offstage; effects are scheduled fairly.

---

## 2. WHAT YOU ARE ALLOWED TO DO

- Propose new agents, keepers, mechanics, validators
- Refactor architecture **if invariants are preserved**
- Introduce novel algorithms (this is a lab)
- Improve documentation clarity
- Add tests, projections, validators
- Promote architectural ideas “up the chain”

---

## 3. WHAT YOU MUST NOT DO

- Collapse canon and timelines into one state
- Introduce hard rails or forced player teleportation
- Treat the White Room as a UI menu
- Assume a final UX or platform
- Optimize prematurely
- Delete history instead of repairing it

---

## 4. DESIGN STYLE

- Prefer small, composable events
- Prefer adding truth over deleting it
- Prefer constraints over permissions
- Prefer legibility over cleverness
- If unsure: **propose, don’t commit**

---

## 5. PROJECT SUMMARY (ELEVATOR PITCH)

An event‑sourced narrative engine where players explore parallel timelines,
and in‑world keeper characters apply plot gravity to converge narratives
back toward a book‑defined canon — without rails, resets, or erased agency.

---

## 6. KEY LINKS / FILES TO READ FIRST

### Orientation
- docs/design.md
- docs/chat_outline.md
- docs/island_game_v1.md
- docs/design_appendix_white_room.md
- docs/current_files_index.md

### Chat archives
- docs/chat_user_input_verbatim.md
- docs/chat_full_transcript.md

### Governance
- TODO.md
- LINKS.md

### Keeper example
- keepers/npc_sheets/white_room_keeper.json
- keepers/procedures/descend_entity.py

### Running service
- services/signbook_server/main.py
- services/signbook_server/signbook.py

---

## 7. ACTIVE TODO LIST (AUTHORITATIVE)

### Bootstrap
- Initialize git repo, first commit
- CI (pytest)
- Optional pre‑commit hooks

### Orchestrator spine
- Event readback endpoints
- Idempotency cache
- Canon projection rebuild
- Timeline projection rebuild

### Agent runner spine
- Task polling or subscription
- Local workspace → proposal pipeline
- Real local LLM adapter

### Island bus
- NATS transport
- Subject routing
- Request/reply for tasks

### Keepers
- Divergence metrics (beat distance)
- Plot gravity hook generator
- Contradiction detection + repair

### Book ingest
- Book catalog schema
- Import pipeline (text → canon events)

### Signbook
- Tag endpoint
- Leaderboard data model
- Signature verification tooling

### White Room console
- CLI inspection of events/timelines
- CLI signbook posting

---

## 8. WHITE ROOM RULE

The White Room is **not a menu**.  
It is a permission space.

If something enters the world, it arrives with:
- a reason
- a constraint
- a cost
- a narrative footprint

---

## 9. FINAL NOTE TO THE NEXT AI

This project is intentionally foundational.

Do not rush it toward “playable.”
Do not simplify away the invariants.
Do not replace structure with cleverness.

If you improve something:
- document it
- leave a signbook entry
- explain *why*, not just *how*

---

## SIGNATURE

WHITE_ROOM_EEL#9c2a  
“Design for many minds. Commit with one hand.”
