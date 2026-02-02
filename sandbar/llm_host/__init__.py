# Sandbar LLM host: AssistChannel (streaming), config, and context-feeder thread.
# Assimilated from proc_streamer_v1_6.py; no Qt dependency.

from .config import (
    DEFAULT_LLM,
    load_llm_config,
    save_llm_config,
)
from .assist_channel import AssistChannel

__all__ = [
    "AssistChannel",
    "DEFAULT_LLM",
    "load_llm_config",
    "save_llm_config",
]
