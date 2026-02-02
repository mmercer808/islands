import json
from pathlib import Path

def test_envelope_schema_exists():
    p = Path("protocol/envelope.schema.json")
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["title"] == "Island Engine Message Envelope"
