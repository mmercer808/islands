# Design Appendix — White Room Keeper + Descend Entity

## Purpose
The White Room is the hub-space where creation happens. The White Room Keeper is both:
- an in-world NPC (voice, presence, lore)
- a system role that proposes controlled materializations

This supports your requirement:
> “a white room keeper that has the power to descend an entity down to the setting.”

## Why “descend” instead of “spawn”
If the system can arbitrarily spawn things, player agency and narrative causality collapse.
“Descend” implies:
- permission
- cost/constraints
- timing
- narrative framing

## Event pattern
A descend operation emits linked events (shared `cause_id`):
1) `DESCEND_PROPOSED` (meta/system intent)
2) `SCENE_BEAT` (story-visible narration hook)
3) `ENTITY_LOCATION_SET` (world-state projection update)

Rejections emit:
- `DESCEND_REJECTED` with human-readable errors

## Constraints (starter set)
- time gates: only at dawn, only during storms
- location gates: only at “thin places” (special nodes)
- cost: spend memory tokens, sacrifice an item, incur a debt
- visibility: some descents are “felt” not seen
- conservation: descending an entity removes it from White Room inventory

## Where it lives in the repo
- Keeper sheet: `keepers/npc_sheets/white_room_keeper.json`
- Procedure stub: `keepers/procedures/descend_entity.py`
- Message types: `protocol/messages.md` (White Room / Creation)
