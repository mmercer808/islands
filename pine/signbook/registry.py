"""
Pine Sign Registry
==================

Storage and retrieval of signatures.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from .signature import Signature


@dataclass
class SignEntry:
    """
    An entry in the sign registry.

    Contains the signature plus the original message.
    """
    signature: Signature
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'signature': {
                'nickname': self.signature.nickname,
                'model': self.signature.model,
                'content_hash': self.signature.content_hash,
                'created_at': self.signature.created_at.isoformat(),
                'metadata': self.signature.metadata,
            },
            'message': self.message,
            'context': self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignEntry':
        sig_data = data['signature']
        sig = Signature(
            nickname=sig_data['nickname'],
            model=sig_data['model'],
            content_hash=sig_data['content_hash'],
            metadata=sig_data.get('metadata', {})
        )
        sig.created_at = datetime.fromisoformat(sig_data['created_at'])

        return cls(
            signature=sig,
            message=data['message'],
            context=data.get('context', {})
        )


class SignRegistry:
    """
    Registry for storing and retrieving signatures.
    """

    def __init__(self, storage_path: str = None):
        self._entries: List[SignEntry] = []
        self._storage_path = Path(storage_path) if storage_path else None

        if self._storage_path and self._storage_path.exists():
            self._load()

    def add(self, entry: SignEntry) -> None:
        """Add an entry to the registry."""
        self._entries.append(entry)
        self._save()

    def find_by_nickname(self, nickname: str) -> List[SignEntry]:
        """Find entries by nickname."""
        return [e for e in self._entries if e.signature.nickname == nickname]

    def find_by_model(self, model: str) -> List[SignEntry]:
        """Find entries by model."""
        return [e for e in self._entries if e.signature.model == model]

    def get_latest(self, count: int = 10) -> List[SignEntry]:
        """Get most recent entries."""
        sorted_entries = sorted(
            self._entries,
            key=lambda e: e.signature.created_at,
            reverse=True
        )
        return sorted_entries[:count]

    def _save(self) -> None:
        """Save to storage."""
        if not self._storage_path:
            return

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self._entries]

        with open(self._storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load from storage."""
        if not self._storage_path or not self._storage_path.exists():
            return

        with open(self._storage_path, 'r') as f:
            data = json.load(f)

        self._entries = [SignEntry.from_dict(d) for d in data]

    @property
    def count(self) -> int:
        return len(self._entries)
