"""
Game runner: receive message -> add to context -> run one step -> return response.
Server uses this to "respond to any message" and "run the game."
"""

from __future__ import annotations
import time
import uuid
from typing import Any, Dict, Optional

# Sandbar engine game_loop (persistent context)
from sandbar.engine.game_loop import (
    OperationType,
    PersistentContextIterator,
    TaskScheduler,
    GameLoopIntegrator,
)


# Singleton runner state (one per process for demo; could be per-session)
_runner: Optional["GameRunner"] = None


class GameRunner:
    """
    Runs the game in response to messages.
    Uses PersistentContextIterator + GameLoopIntegrator; one message = one PLAYER_INPUT operation + one frame.
    """

    def __init__(self, target_fps: int = 10):
        self.context_iterator = PersistentContextIterator(name=f"sandbar_{uuid.uuid4().hex[:8]}")
        self.task_scheduler = TaskScheduler(max_worker_threads=4)
        self.integrator = GameLoopIntegrator(target_fps=target_fps)
        self.context_iterator.start_main_loop_integration()
        self.task_scheduler.set_context_iterator(self.context_iterator)
        self.integrator.setup(self.context_iterator, self.task_scheduler)
        self._frame_count = 0

    def run_one_frame(self) -> Dict[str, Any]:
        """Run a single game loop frame. Returns cycle result + context summary."""
        self._frame_count += 1
        cycle_result = self.context_iterator.main_loop_cycle()
        summary = self.context_iterator.context_window.get_context_summary() if hasattr(
            self.context_iterator.context_window, "get_context_summary"
        ) else {}
        return {
            "cycle": cycle_result,
            "frame": self._frame_count,
            "context_operations": len(self.context_iterator.context_window.operations),
            "summary": summary if isinstance(summary, dict) else {},
        }

    def receive_message(self, message: str, source: str = "user") -> Dict[str, Any]:
        """
        Receive a message, add it as PLAYER_INPUT to context, run one frame, return response.
        """
        op = self.context_iterator.add_operation(
            OperationType.PLAYER_INPUT,
            {"message": message, "source": source},
            source=source,
            target="game_runner",
        )
        frame_result = self.run_one_frame()
        # Build response from last operation or context
        recent = self.context_iterator.get_last_operations(5)
        response_text = message  # Echo for now; can be replaced by LLM or story engine output
        return {
            "ok": True,
            "operation_id": op.operation_id,
            "frame": frame_result,
            "response": response_text,
            "recent_operations": [
                {
                    "id": r.operation_id,
                    "type": r.operation_type.name,
                    "source": r.source,
                    "data": r.data,
                }
                for r in recent[:5]
            ],
        }


def get_runner() -> GameRunner:
    """Get or create the singleton GameRunner."""
    global _runner
    if _runner is None:
        _runner = GameRunner()
    return _runner


def run_message(message: str, source: str = "user") -> Dict[str, Any]:
    """Convenience: receive message and return response using singleton runner."""
    return get_runner().receive_message(message, source=source)
