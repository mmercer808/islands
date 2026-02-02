"""storage.py (compat wrapper)

This wraps services.signbook_server.signbook.Signbook to preserve the earlier name.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from services.signbook_server.signbook import Signbook as _Signbook


class SignbookStore:
    def __init__(self, path: str):
        self._sb = _Signbook(path)

    def add_entry(self, signature: str, message: str, context: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        return self._sb.add_entry(signature, message, context=context, tags=tags)

    def list_entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._sb.list_entries(limit=limit)

    def search(self, q: str, limit: int = 50) -> List[Dict[str, Any]]:
        return self._sb.search_entries(q, limit=limit)

    def add_editorial(self, note: str) -> Dict[str, Any]:
        return self._sb.add_editorial(note)

    def list_editorial(self, limit: int = 25) -> List[Dict[str, Any]]:
        return self._sb.list_editorial(limit=limit)
