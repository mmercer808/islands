# Sandbar — Source Assimilation Map

Unorganized root-level and loose files mapped to sandbar structure. Sources are **assimilated** (copied or wired) into sandbar so the interface lives under one tree.

---

## Summary: Can the source be applied?

| Source | Applies? | Sandbar location | Action |
|--------|----------|------------------|--------|
| proc_streamer_v1_6.py | Yes | llm_host/ + ui entry | Extract AssistChannel + config to llm_host; UI stays runnable at root, launcher can invoke it |
| proc_streamer_legacy_ui.py | Yes | ui/ (variant) | Assimilated as UI variant; launcher option |
| aqua/ | Yes | ui/ (use from root or copy) | Use from repo root `aqua/`; launcher adds path |
| aqua.py (root) | Yes | ui/ | Signal-based entry; reference from launcher |
| islands/ (connector, integration, messaging) | Yes | engine/messaging/ | Copy into sandbar/engine/messaging/ for interface wiring |
| GAME_VS_API_SEPARATION.md | Yes | docs/ | Copy to sandbar/docs/ |
| MMO_ARCHITECTURE.md | Yes | docs/ | Copy to sandbar/docs/ |
| main.py | Yes | ui/ or root | Stub; launcher can be the real entry |
| launcher/launcher.py | Yes | ui/launcher or reference | Sandbar launcher can call launcher or replace |
| files(8)/, inage-text/ | Optional | engine/ or island-engine | Graph/narrative; reference from CONCEPT_SIGNATURES; not duplicated in sandbar |
| context_serialization.py, live_code_system.py | Optional | matts/ or ref | Utilities; use from matts/ |
| unified_console_tab.py | In proc_streamer | — | Stays inside proc_streamer; sandbar llm_host is Qt-free |

---

## 1) llm_host/ — from proc_streamer_v1_6.py

| What | Source | Sandbar file | Notes |
|------|--------|--------------|--------|
| LLM config (url, model, provider, timeout, stream) | GlobalSettings.llm | llm_host/config.py | No Qt; load/save dict or JSON file |
| AssistChannel (healthcheck, query, Ollama/OpenAI-compat stream) | AssistChannel class | llm_host/assist_channel.py | Qt-free: use callbacks (on_chunk, on_complete, on_error, on_status) so UI can wrap and emit Qt signals |
| Context-feeder thread | New | llm_host/context_feeder.py | Stub; will use engine/client |

**Root proc_streamer_v1_6.py** remains runnable; it can be refactored later to import from sandbar.llm_host.

---

## 2) ui/ — interface entry points

| What | Source | Sandbar file | Notes |
|------|--------|--------------|--------|
| Launcher | New | ui/launcher.py | Entry: start engine client + llm_host; run proc_streamer or legacy_ui from root, or minimal sandbar window |
| Legacy UI variant | proc_streamer_legacy_ui.py | Referenced from root | ui/launcher can run `python proc_streamer_legacy_ui.py` as subprocess or import |
| Aqua UI | aqua/, aqua.py | Use ../../aqua | Launcher adds sys.path; imports aqua doc/dock/prompt_stage |

---

## 3) engine/ — thin client + messaging

| What | Source | Sandbar file | Notes |
|------|--------|--------------|--------|
| Orchestrator thin client | Existing | engine/client.py | get_recent_events, get_canon_slice, get_current_task, get_timeline_state |
| Connector / listener / routing | islands/connector_core.py | engine/messaging/connector_core.py | Signal routing for interface |
| Integration layer | islands/integration_layer.py | engine/messaging/integration_layer.py | Listener + entity pipeline |
| Messaging interface | islands/messaging_interface.py | engine/messaging/messaging_interface.py | Buffered I/O, prompt→signal |

Imports inside engine/messaging/ are relative (e.g. from .connector_core).

---

## 4) docs/ — design and separation

| What | Source | Sandbar file | Notes |
|------|--------|--------------|--------|
| Game vs API separation | GAME_VS_API_SEPARATION.md | docs/GAME_VS_API_SEPARATION.md | Copy |
| MMO architecture | MMO_ARCHITECTURE.md | docs/MMO_ARCHITECTURE.md | Copy |
| Integration diagram | Existing | docs/integration.md | Thread diagram, when to use sandbar |
| Main structure | Existing | docs/main_structure.md | island-engine vs scaffold_v3 |

---

## 5) Not assimilated (reference only)

- **matts/** — Use as-is from repo root; sandbar does not copy.
- **pine/** — Use as-is; narrative/graph/runtime.
- **island-engine/** — Main working dir; sandbar consumes via engine/client.
- **files(8)/, inage-text/, Mechanics/** — Graph/narrative/design; CONCEPT_SIGNATURES and island-engine; no duplicate in sandbar.
- **infocom-transfer/, vetrell/** — Content/language; reference from island-engine or docs.

---

## Run order after assimilation

From **repo root** (`c:\code\islands`):

1. **Sandbar launcher (recommended):**
   - `python -m sandbar.ui.launcher` — runs proc_streamer (default).
   - `python -m sandbar.ui.launcher legacy` — runs proc_streamer_legacy_ui.
   - `python -m sandbar.ui.launcher minimal` — llm_host + engine client only (no Qt).
2. **Full interface (direct):** `python proc_streamer_v1_6.py` or `python proc_streamer_legacy_ui.py`.
3. **Dual display** (game→LLM and LLM→interface) is implemented separately (sandbar TODO §3).
