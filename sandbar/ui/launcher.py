"""
Sandbar UI launcher.

Ensures repo root is on sys.path, then:
- Runs proc_streamer_v1_6 (main interface), or
- Runs proc_streamer_legacy_ui (legacy variant), or
- Starts a minimal sandbar window using llm_host + engine client.

Dual display (game→LLM and LLM→interface) is implemented separately (step 3).
"""

from __future__ import annotations
import sys
import os
import argparse
from pathlib import Path

# Repo root = parent of sandbar
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _run_proc_streamer() -> None:
    """Run main proc_streamer UI (root proc_streamer_v1_6.py)."""
    import proc_streamer_v1_6
    proc_streamer_v1_6.main()


def _run_legacy_ui() -> None:
    """Run legacy proc_streamer UI (root proc_streamer_legacy_ui.py)."""
    import proc_streamer_legacy_ui
    proc_streamer_legacy_ui.main()


def _run_minimal_sandbar() -> None:
    """Minimal sandbar: llm_host + engine client, no full UI (placeholder)."""
    from sandbar.llm_host import load_llm_config, AssistChannel
    from sandbar.engine import client as engine_client

    config = load_llm_config()
    channel = AssistChannel(config)
    channel.on_status = lambda ok, msg: print(f"[Status] {ok}: {msg}")
    channel.on_chunk = lambda s: print(s, end="", flush=True)
    channel.on_complete = lambda s: print("\n[Done]")
    channel.on_error = lambda e: print(f"[Error] {e}")

    print("Sandbar minimal: llm_host + engine client.")
    print("  Engine recent events:", len(engine_client.get_recent_events(10)))
    channel.connect()
    if channel._connected:
        print("Connected. Try: channel.query('Hello') from Python.")
    else:
        print("Not connected. Start Ollama or set config.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Sandbar UI launcher")
    ap.add_argument(
        "mode",
        nargs="?",
        default="proc_streamer",
        choices=["proc_streamer", "legacy", "minimal"],
        help="proc_streamer (default), legacy, or minimal",
    )
    args = ap.parse_args()
    os.chdir(_REPO_ROOT)

    if args.mode == "proc_streamer":
        _run_proc_streamer()
    elif args.mode == "legacy":
        _run_legacy_ui()
    else:
        _run_minimal_sandbar()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
