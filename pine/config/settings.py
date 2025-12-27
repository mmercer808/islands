"""
Pine Settings
=============

Application settings management.
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class Settings:
    """
    Application settings.

    Provides typed access to configuration values.
    """
    # LLM settings
    llm_provider: str = "ollama"
    llm_model: str = "llama2"
    llm_base_url: str = "http://localhost:11434"

    # UI settings
    theme: str = "dark"
    font_size: int = 12
    window_width: int = 1200
    window_height: int = 800

    # Engine settings
    max_lookahead_depth: int = 3
    enable_embeddings: bool = True
    embedding_dimension: int = 128

    # Paths
    signbook_path: str = "signbook"
    snapshots_path: str = "snapshots"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Settings':
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


class SettingsManager:
    """
    Manages settings persistence.
    """

    DEFAULT_PATH = "settings.json"

    def __init__(self, path: str = None):
        self.path = Path(path or self.DEFAULT_PATH)
        self._settings: Optional[Settings] = None

    def load(self) -> Settings:
        """Load settings from file."""
        if self.path.exists():
            with open(self.path, 'r') as f:
                data = json.load(f)
            self._settings = Settings.from_dict(data)
        else:
            self._settings = Settings()

        return self._settings

    def save(self, settings: Settings = None) -> None:
        """Save settings to file."""
        settings = settings or self._settings or Settings()

        with open(self.path, 'w') as f:
            json.dump(settings.to_dict(), f, indent=2)

        self._settings = settings

    def get(self) -> Settings:
        """Get current settings, loading if needed."""
        if self._settings is None:
            self.load()
        return self._settings

    def update(self, **kwargs) -> Settings:
        """Update settings with new values."""
        settings = self.get()
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        self.save(settings)
        return settings


# Global instance
_manager: Optional[SettingsManager] = None


def get_settings(path: str = None) -> Settings:
    """Get global settings."""
    global _manager
    if _manager is None:
        _manager = SettingsManager(path)
    return _manager.get()


def save_settings(settings: Settings = None) -> None:
    """Save global settings."""
    global _manager
    if _manager is None:
        _manager = SettingsManager()
    _manager.save(settings)
