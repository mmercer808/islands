"""
Sandbar server: FastAPI app that responds to any message and runs the game.
POST /message -> game_runner.receive_message -> response.
"""

from __future__ import annotations
from typing import Optional

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False
    FastAPI = None
    BaseModel = None

from .game_runner import get_runner, run_message


class MessageIn(BaseModel):
    """Incoming message from client."""
    message: str
    source: Optional[str] = "user"


class MessageOut(BaseModel):
    """Response to client."""
    ok: bool
    response: str
    operation_id: Optional[str] = None
    frame: Optional[dict] = None
    recent_operations: Optional[list] = None


def create_app() -> "FastAPI":
    if not HAVE_FASTAPI:
        raise ImportError("Install fastapi and uvicorn to run the sandbar server: pip install fastapi uvicorn")
    app = FastAPI(title="Sandbar", version="0.1.0", description="Server that responds to any message and runs the game.")

    @app.post("/message", response_model=MessageOut)
    def post_message(body: MessageIn):
        """Receive a message, run the game one step, return response."""
        result = run_message(body.message, source=body.source or "user")
        return MessageOut(
            ok=result.get("ok", True),
            response=result.get("response", body.message),
            operation_id=result.get("operation_id"),
            frame=result.get("frame"),
            recent_operations=result.get("recent_operations"),
        )

    @app.get("/health")
    def health():
        return {"ok": True, "service": "sandbar"}

    return app


if HAVE_FASTAPI:
    app = create_app()
else:
    app = None  # Install fastapi + uvicorn to run: pip install fastapi uvicorn
