"""
DESCEND ENTITY
==============
A White Room Keeper procedure: "descend" an entity from the White Room hub
into a target setting location, as a story-visible event.

Design intent:
- This is *not* teleportation as a brute force move.
- It's a narrated materialization with constraints (cost, permission, timing).

One-liner idea:
    "Creation is a door, not a cheat code."
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DescendRequest:
    keeper_id: str
    entity_id: str
    entity_type: str  # character/item/location_fragment
    target_location_id: str
    reason: str
    constraints: Dict[str, Any]  # e.g. {"only_at_dawn": True, "cost": "memory_1"}


def validate_request(req: DescendRequest) -> List[str]:
    """Return a list of human-readable validation errors (empty means ok)."""
    errors: List[str] = []
    if not req.keeper_id:
        errors.append("keeper_id required")
    if not req.entity_id:
        errors.append("entity_id required")
    if not req.target_location_id:
        errors.append("target_location_id required")
    if not isinstance(req.constraints, dict):
        errors.append("constraints must be a dict")
    return errors


def propose_descend_events(req: DescendRequest) -> List[Dict[str, Any]]:
    """Return a list of event payloads representing a descend operation."""
    errs = validate_request(req)
    if errs:
        return [{
            "type": "DESCEND_REJECTED",
            "keeper_id": req.keeper_id,
            "entity_id": req.entity_id,
            "errors": errs,
        }]

    cause_id = f"cause:{req.keeper_id}:{req.entity_id}"
    return [
        {
            "type": "DESCEND_PROPOSED",
            "cause_id": cause_id,
            "keeper_id": req.keeper_id,
            "entity_id": req.entity_id,
            "entity_type": req.entity_type,
            "target_location_id": req.target_location_id,
            "reason": req.reason,
            "constraints": req.constraints,
        },
        {
            "type": "SCENE_BEAT",
            "cause_id": cause_id,
            "beat_kind": "materialization",
            "location_id": req.target_location_id,
            "text_hint": "A white seam opens in the air; something steps through.",
        },
        {
            "type": "ENTITY_LOCATION_SET",
            "cause_id": cause_id,
            "entity_id": req.entity_id,
            "location_id": req.target_location_id,
        },
    ]
