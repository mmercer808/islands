"""
LLM host config — URL, model, provider, timeout, stream.
Assimilated from proc_streamer_v1_6.GlobalSettings.llm; no Qt.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_LLM = {
    "provider": "ollama",
    "url": "http://127.0.0.1:11434",
    "model": "llama3:latest",
    "api_key": "",
    "timeout": 60,
    "stream": True,
}


def load_llm_config(settings_path: Path | str | None = None) -> Dict[str, Any]:
    """Load LLM config from JSON file. Returns dict with keys provider, url, model, api_key, timeout, stream."""
    path = Path(settings_path) if settings_path else Path("settings.json")
    if not path.exists():
        return dict(DEFAULT_LLM)
    try:
        data = json.loads(path.read_text())
        out = dict(DEFAULT_LLM)
        out.update(data.get("llm", {}))
        return out
    except Exception:
        return dict(DEFAULT_LLM)


def save_llm_config(config: Dict[str, Any], settings_path: Path | str | None = None) -> None:
    """Merge LLM config into settings file (preserves other keys)."""
    path = Path(settings_path) if settings_path else Path("settings.json")
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            pass
    data["llm"] = {**DEFAULT_LLM, **config}
    path.write_text(json.dumps(data, indent=2))
