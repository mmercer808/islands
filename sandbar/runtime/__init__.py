"""
Sandbar runtime — live code, hot-swap, context serialization.

Assimilated from repo root: live_code_system.py, runtime_hotswap_system.py,
context_serialization.py. Use for runtime code injection and compressed context
transmission. Pine runtime (sandbar.pine.runtime) has live_code, hotswap, generators.
"""

# Root modules copied here; import by name to avoid loading heavy deps by default
__all__ = [
    "live_code_system",
    "runtime_hotswap_system",
    "context_serialization",
]
