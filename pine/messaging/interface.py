"""
Pine Messaging Interface
========================

Buffered I/O and message handling.

SOURCE: islands/messaging_interface.py (16KB)
Copy the full implementation from that file.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import asyncio

from pine.core.signals import SignalPayload, SignalType


@dataclass
class Message:
    """A message in the buffer."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0


class MessageBuffer:
    """
    Thread-safe message buffer.

    Provides buffered I/O for components.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()

    async def push(self, message: Message) -> None:
        """Push message to buffer."""
        async with self._lock:
            self._buffer.append(message)

    async def pop(self) -> Optional[Message]:
        """Pop message from buffer."""
        async with self._lock:
            if self._buffer:
                return self._buffer.popleft()
        return None

    async def peek(self) -> Optional[Message]:
        """Peek at next message without removing."""
        async with self._lock:
            if self._buffer:
                return self._buffer[0]
        return None

    @property
    def count(self) -> int:
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0


class BufferedIO:
    """
    Buffered I/O interface for components.

    Provides read/write with buffering.
    """

    def __init__(self):
        self.input_buffer = MessageBuffer()
        self.output_buffer = MessageBuffer()

    async def write(self, content: str, **metadata) -> None:
        """Write to output buffer."""
        await self.output_buffer.push(Message(content=content, metadata=metadata))

    async def read(self) -> Optional[str]:
        """Read from input buffer."""
        msg = await self.input_buffer.pop()
        return msg.content if msg else None

    async def receive(self, content: str, **metadata) -> None:
        """Receive input."""
        await self.input_buffer.push(Message(content=content, metadata=metadata))


class SignalFactory:
    """
    Factory for creating signals.

    Provides convenience methods for common signal types.
    """

    def __init__(self, source: str):
        self.source = source

    def create(
        self,
        signal_type: SignalType,
        data: Dict[str, Any] = None,
        **metadata
    ) -> SignalPayload:
        """Create a signal payload."""
        return SignalPayload(
            signal_type=signal_type,
            source=self.source,
            data=data or {},
            metadata=metadata
        )

    def story_event(self, event_name: str, **data) -> SignalPayload:
        """Create a story event signal."""
        return self.create(
            SignalType.STORY_EVENT,
            data={'event': event_name, **data}
        )

    def context_updated(self, context_id: str, **changes) -> SignalPayload:
        """Create a context update signal."""
        return self.create(
            SignalType.CONTEXT_UPDATED,
            data={'context_id': context_id, 'changes': changes}
        )
