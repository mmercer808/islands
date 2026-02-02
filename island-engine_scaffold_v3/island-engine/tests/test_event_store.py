import os
from services.orchestrator.event_store import EventStore

def test_event_store_append(tmp_path):
    db = tmp_path / "events.sqlite3"
    s = EventStore(str(db))
    s.append("TEST", {"x": 1}, actor="tester", thread="t:1", message_id="m1", ts="2026-01-01T00:00:00Z")
    rows = s.list_recent(10)
    assert len(rows) == 1
