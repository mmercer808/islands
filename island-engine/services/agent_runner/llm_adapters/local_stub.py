"""
LOCAL LLM ADAPTER (stub)
=======================
Replace this with your local model bridge (Ollama, llama.cpp server, etc.)

One-liner idea:
    "Swap brains without rewriting the body."
"""

from __future__ import annotations
from typing import Dict, Any


class LocalStubLLM:
    def complete(self, prompt: str, **kwargs) -> str:
        # TODO: integrate local LLM call
        return f"[stub completion]\n{prompt[:200]}"
