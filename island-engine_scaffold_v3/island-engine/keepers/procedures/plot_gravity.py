"""
PLOT GRAVITY PROCEDURE
=====================
A keeper procedure that proposes interventions to pull player timelines
back toward book canon without feeling like a rail.

One-liner idea:
    "Never delete a player's agency—redirect it with meaning."
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GravitySuggestion:
    """A proposed plot-gravity move, expressed as an event payload."""
    player_timeline_id: str
    hook_type: str
    description: str
    suggested_events: List[Dict[str, Any]]


def detect_divergence(canon_snapshot: Dict[str, Any], timeline_snapshot: Dict[str, Any]) -> List[str]:
    """Return a list of divergence signals (strings) that require keeper attention."""
    # TODO: implement contradiction + beat-distance signals
    return []


def propose_gravity_moves(player_timeline_id: str, signals: List[str]) -> List[GravitySuggestion]:
    """Turn divergence signals into narrative interventions that re-align the plot."""
    # TODO: implement hook generation (omens, NPC encounters, item revelations, time pressure)
    return []
