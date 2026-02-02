# Adding new concepts (claude-add)

Short reference for integrating **novel concepts** into the Islands collection. Full instructions: **MAINTENANCE.md § Integrating novel concepts (claude-add)** and **CLAUDE.md § Adding new concepts (claude-add)**.

---

## Truths we have to display (do not contradict)

- Canon is event-sourced.
- Agents propose; one writer commits.
- Story is infrastructure, not output.
- Creation uses “descend”, not “spawn”.
- Keepers redirect agency; they don’t erase it.
- Deferred builder pattern; serializable execution contexts; signbook.
- Interface = proc_streamer; dual display (game→LLM and LLM→interface separately).

---

## How to integrate new ideas

1. **Check truths** — New idea must align with or extend the list above (see MAINTENANCE.md § Best concepts).
2. **Place in one canonical area** — island-engine (spine), sandbar (message server), pine/matts (libs). If new subsystem, document in MAINTENANCE.md and CLAUDE.md.
3. **Rework with explanation** — If reworking existing code: state how truths are preserved; update MAINTENANCE.md and CLAUDE.md.
4. **Update surfaces** — CONCEPT_SIGNATURES.md for new APIs; ISLANDS_TODO.md or sandbar/TODO.md for new tasks; docs/UI for user-facing behavior.
5. **Keep collection coherent** — Each subsystem has a role; new concepts should fit the structure or extend it explicitly.

---

## Where to look

- **MAINTENANCE.md** — Best concepts, correct order, decisions, full “Integrating novel concepts (claude-add)” section.
- **CLAUDE.md** — Key systems, “Adding new concepts (claude-add)” instructions for edits.
- **CONCEPT_SIGNATURES.md** — Function signatures when merging or adding from other sources.
- **sandbar/docs/CLEANUP_PLAN.md** — What to archive, delete, keep; truths to preserve before cleanup.
