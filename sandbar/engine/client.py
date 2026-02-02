"""
Thin client to chpt-island-engine.
No copy of the engine: HTTP to orchestrator or in-process imports for dev.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional

# Stub until orchestrator exposes /events/recent, /projection/canon, etc.
ORCH_URL = "http://127.0.0.1:8000"


def get_recent_events(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch recent events from orchestrator. Stub: returns [] until endpoint exists."""
    # TODO: GET {ORCH_URL}/events/recent?limit={limit}
    return []


def get_canon_slice() -> Dict[str, Any]:
    """Fetch canon projection slice. Stub: returns {} until endpoint exists."""
    # TODO: GET {ORCH_URL}/projection/canon
    return {}


def get_current_task() -> Optional[Dict[str, Any]]:
    """Fetch current task for this runner. Stub: returns None until task API exists."""
    # TODO: GET {ORCH_URL}/task/next or similar
    return None


def get_timeline_state(player_id: str) -> Dict[str, Any]:
    """Fetch timeline projection for a player. Stub: returns {} until endpoint exists."""
    # TODO: GET {ORCH_URL}/projection/timeline/{player_id}
    return {}
