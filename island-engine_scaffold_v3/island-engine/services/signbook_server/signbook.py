"""signbook.py (service module)

A minimal signbook database layer meant to run continuously while the rest of the
system evolves.

One-liner idea:
    "A wall that remembers — and lets you annotate the build as it happens."
"""

from __future__ import annotations
import sqlite3
import time
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  signature TEXT NOT NULL,
  context TEXT,
  message TEXT NOT NULL,
  tags_json TEXT NOT NULL,
  checksum16 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS editorial (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  note TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entries_signature ON entries(signature);
CREATE INDEX IF NOT EXISTS idx_entries_ts ON entries(ts);
"""


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def checksum16(signature: str, ts: str, message: str, tags: List[str]) -> str:
    canonical = json.dumps(
        {"signature": signature, "timestamp": ts, "message": message, "tags": tags},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class Signbook:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init(self) -> None:
        with self._connect() as con:
            con.executescript(SCHEMA)

    # --- entries ---
    def add_entry(self, signature: str, message: str, context: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        tags = tags or []
        ts = iso_now()
        chk = checksum16(signature, ts, message, tags)
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO entries(ts, signature, context, message, tags_json, checksum16) VALUES (?,?,?,?,?,?)",
                (ts, signature, context, message, json.dumps(tags), chk),
            )
            entry_id = cur.lastrowid
        return {"id": entry_id, "timestamp": ts, "signature": signature, "context": context, "message": message, "tags": tags, "checksum16": chk}

    def list_entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("SELECT id, ts, signature, context, message, tags_json, checksum16 FROM entries ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        return [
            {"id": r[0], "timestamp": r[1], "signature": r[2], "context": r[3], "message": r[4], "tags": json.loads(r[5]), "checksum16": r[6]}
            for r in rows
        ]

    def search_entries(self, q: str, limit: int = 50) -> List[Dict[str, Any]]:
        pattern = f"%{q}%"
        with self._connect() as con:
            cur = con.cursor()
            cur.execute(
                "SELECT id, ts, signature, context, message, tags_json, checksum16 FROM entries "
                "WHERE message LIKE ? OR signature LIKE ? OR context LIKE ? "
                "ORDER BY id DESC LIMIT ?",
                (pattern, pattern, pattern, limit),
            )
            rows = cur.fetchall()
        return [
            {"id": r[0], "timestamp": r[1], "signature": r[2], "context": r[3], "message": r[4], "tags": json.loads(r[5]), "checksum16": r[6]}
            for r in rows
        ]

    # --- editorial notes ---
    def add_editorial(self, note: str) -> Dict[str, Any]:
        ts = iso_now()
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("INSERT INTO editorial(ts, note) VALUES (?,?)", (ts, note))
            note_id = cur.lastrowid
        return {"id": note_id, "timestamp": ts, "note": note}

    def list_editorial(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("SELECT id, ts, note FROM editorial ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        return [{"id": r[0], "timestamp": r[1], "note": r[2]} for r in rows]
