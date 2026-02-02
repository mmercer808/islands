"""
PROJECTIONS
===========
Rebuild derived views from the event log.

One-liner idea:
    "Projections are disposable; the log is sacred."
"""

from __future__ import annotations
from typing import Any, Dict


def rebuild_canon(events) -> Dict[str, Any]:
    """Rebuild canon projection from events."""
    # TODO: apply WORLD_UPSERT_COMMITTED, MERGE_COMMITTED, etc.
    return {"canon": {}, "meta": {"events": len(events)}}


def rebuild_timeline(events, player_id: str) -> Dict[str, Any]:
    """Rebuild a player's timeline projection from canon + deltas."""
    # TODO: filter by player_id and apply deltas in order
    return {"player_id": player_id, "timeline": {}, "meta": {"events": len(events)}}
