"""
EVENT STORE (SQLite)
===================
Append-only storage for events. This is the truth layer.

One-liner idea:
    "If it isn't in the log, it didn't happen."
"""

from __future__ import annotations
import sqlite3
from typing import Any, Dict, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  message_id TEXT UNIQUE,
  ts TEXT,
  actor TEXT,
  thread TEXT,
  type TEXT NOT NULL,
  payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_thread ON events(thread);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
"""


class EventStore:
    def __init__(self, path: str):
        self.path = path
        self._init()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init(self):
        with self._connect() as con:
            con.executescript(SCHEMA)

    def append(self, type: str, payload: Dict[str, Any], actor: str, thread: str, message_id: str, ts: str) -> int:
        """Append an event. Idempotent by message_id."""
        import json
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO events(message_id, ts, actor, thread, type, payload_json) VALUES (?,?,?,?,?,?)",
                (message_id, ts, actor, thread, type, json.dumps(payload, sort_keys=True)),
            )
            return cur.lastrowid

    def list_recent(self, limit: int = 50):
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("SELECT id, ts, actor, thread, type, payload_json FROM events ORDER BY id DESC LIMIT ?", (limit,))
            return cur.fetchall()
