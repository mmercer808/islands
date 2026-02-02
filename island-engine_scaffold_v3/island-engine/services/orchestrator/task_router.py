"""
TASK ROUTER
===========
Assign tasks to agents; track completion.

One-liner idea:
    "A task is a promise with an ID."
"""

from __future__ import annotations
from typing import Dict, Any


def assign_task(agent_id: str, task_kind: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Return a TASK_ASSIGN payload."""
    # TODO: include deadlines, priority, retry policies
    return {"agent_id": agent_id, "task_kind": task_kind, "args": args}
