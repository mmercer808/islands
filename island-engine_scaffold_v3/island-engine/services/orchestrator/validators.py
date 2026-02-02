"""
VALIDATORS
==========
Small, sharp checks that protect coherence.

One-liner idea:
    "Validators are the laws of physics for the narrative."
"""

from __future__ import annotations
from typing import Any, Dict


class ValidationError(Exception):
    pass


def validate_event(event_type: str, payload: Dict[str, Any]) -> None:
    """Validate an incoming event payload. Raise ValidationError if invalid."""
    # TODO: add per-event schemas and plot beat constraints
    if not isinstance(payload, dict):
        raise ValidationError("payload must be a dict")
