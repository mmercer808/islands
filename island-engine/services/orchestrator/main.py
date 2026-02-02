"""
ORCHESTRATOR SERVICE
===================
Single-writer for canon. Routes tasks, runs validators, commits events.

One-liner idea:
    "Everything important is an event; canon is a projection."
"""

from __future__ import annotations
import os
from fastapi import FastAPI
from pydantic import BaseModel
from services.orchestrator.event_store import EventStore
from services.orchestrator.validators import validate_event

app = FastAPI(title="Island Orchestrator", version="0.1.0")
DB_PATH = os.getenv("ORCH_DB", "var/orchestrator.sqlite3")
store = EventStore(DB_PATH)


class Envelope(BaseModel):
    v: int = 1
    message_id: str
    ts: str
    from_: str  # 'from' is reserved in Python; we map in/out in adapters
    to: str
    type: str
    thread: str
    payload: dict


@app.post("/event")
def post_event(env: Envelope):
    """Accept an event proposal; validate and (if allowed) commit."""
    validate_event(env.type, env.payload)
    store.append(env.type, env.payload, actor=env.from_, thread=env.thread, message_id=env.message_id, ts=env.ts)
    return {"ok": True}


@app.get("/health")
def health():
    return {"ok": True, "db": DB_PATH}
