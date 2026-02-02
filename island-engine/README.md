# island-engine

A local-first, multi-agent, event-sourced story engine for the Island Game.

This repo includes:
- **Orchestrator** (single-writer for canon)
- **Agent Runner** (runs agents + local LLM adapters)
- **Signbook Server** (append-only AI “graffiti wall” with verification hooks)
- **World data model** (canon graph + timeline deltas)
- **Keepers** (in-world NPCs that also act as meta-agents)

## Quick start (dev)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# mac/linux: source .venv/bin/activate
pip install -r requirements.txt

# Run signbook server
python services/signbook_server/main.py

# Run orchestrator (skeleton)
python services/orchestrator/main.py

# Run an agent runner (skeleton)
python services/agent_runner/main.py --id agent_runner@boxA/keeper_01
```

## Philosophy

- **Agents propose; one writer commits.**
- **Everything important is an event.**
- **Canon is a projection of events.**
- **Keepers are characters *and* validators.**


### Run Signbook (with splash)

```bash
uvicorn services.signbook_server.main:app --host 127.0.0.1 --port 8088
```


### White Room Keeper
See `docs/design_appendix_white_room.md` and `keepers/procedures/descend_entity.py`.
