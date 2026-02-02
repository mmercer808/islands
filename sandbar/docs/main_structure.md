# Main Working Directory and Scaffold

## Main = island-engine (at repo root)

The **main working directory** is **island-engine/** at the repo root. It was created by copying **chpt-island-engine** into the root as **island-engine**. All canonical engine work (orchestrator, keepers, agent_runner, signbook, world, protocol) lives there.

- **island-engine/** — main; current structure (services/, keepers/, agents/, protocol/, signbook/, world/, island_engine/, docs/, tests/, scripts/).
- **chpt-island-engine/** — original source; can be kept as reference or deprecated once island-engine is the single source of truth.
- **matts/** — utilities (signals, context, serialization, live code); consumed by island-engine or sandbar as needed.
- **sandbar/** — alternate integration (interface = proc_streamer; dual display game→LLM and LLM→interface; context-feeder; aqua UI). Consumes island-engine.

## island-engine_scaffold_v3

**island-engine_scaffold_v3** is the **proposed directory structure**. It should **emulate the current structure** (the one in island-engine). Right now we don’t have a single “scaffold” copy that is the template; island-engine at root **is** that structure.

- **island-engine_scaffold_v3/island-engine/** — contains a copy of the same layout (agents/, keepers/, services/, etc.). Use it as the **reference layout** when adding new top-level dirs or when appending files from another source.
- To “continue with that”: when adding new pieces (e.g. from another source), add them under **island-engine/** at root following the layout inside **island-engine_scaffold_v3/island-engine/** so the main tree stays aligned with the proposed structure.

**Summary:** Main = **island-engine** at root (current structure). Scaffold_v3 = proposed structure that **emulates** this layout. New additions should follow the same layout so the main directory remains the single canonical tree.
