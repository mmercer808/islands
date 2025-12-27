#!/usr/bin/env python3
"""
Messaging Interface System

Robust messaging layer with:
- Own buffer for message accumulation
- Update cycle integration  
- I/O bridge connector to signal factory
- Prompt → Signal translation
- Logic execution hooks
"""

from __future__ import annotations
import asyncio
import uuid
import inspect
from typing import (
    TypeVar, Generic, Callable, Any, Union, Type, Protocol,
    Dict, List, Optional, Awaitable, Tuple, Iterator
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from collections import deque
from abc import ABC, abstractmethod

T = TypeVar('T')


# =============================================================================
# PROTOCOLS - What things must support
# =============================================================================

class Updatable(Protocol):
    """Protocol for things that can be updated."""
    def update(self) -> bool: ...


class Emittable(Protocol):
    """Protocol for things that can emit signals."""
    def emit(self, signal: Any) -> Any: ...


# =============================================================================
# MESSAGE BUFFER - Accumulates messages between updates
# =============================================================================

@dataclass 
class BufferedMessage:
    """A message in the buffer."""
    message_id: str
    content: Any
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0
    processed: bool = False


class MessageBuffer:
    """
    Buffer for message accumulation with capacity and flush logic.
    """
    
    def __init__(self, capacity: int = 1000, flush_threshold: float = 0.8):
        self._buffer: deque[BufferedMessage] = deque(maxlen=capacity)
        self.capacity = capacity
        self.flush_threshold = flush_threshold
        self._pending_count = 0
    
    def push(self, content: Any, source: str = "", priority: int = 0) -> BufferedMessage:
        """Add message to buffer."""
        msg = BufferedMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            content=content,
            source=source,
            priority=priority
        )
        self._buffer.append(msg)
        self._pending_count += 1
        return msg
    
    def pop(self) -> Optional[BufferedMessage]:
        """Pop highest priority unprocessed message."""
        # Simple FIFO for now, could sort by priority
        for msg in self._buffer:
            if not msg.processed:
                msg.processed = True
                self._pending_count -= 1
                return msg
        return None
    
    def peek(self, n: int = 1) -> List[BufferedMessage]:
        """Peek at next n unprocessed messages."""
        return [m for m in self._buffer if not m.processed][:n]
    
    def flush(self) -> List[BufferedMessage]:
        """Flush all unprocessed messages."""
        flushed = [m for m in self._buffer if not m.processed]
        for m in flushed:
            m.processed = True
        self._pending_count = 0
        return flushed
    
    def should_flush(self) -> bool:
        """Check if buffer should be flushed (capacity threshold)."""
        return len(self._buffer) >= self.capacity * self.flush_threshold
    
    @property
    def pending(self) -> int:
        return self._pending_count
    
    def __len__(self) -> int:
        return len(self._buffer)
    
    def __iter__(self) -> Iterator[BufferedMessage]:
        return iter(m for m in self._buffer if not m.processed)


# =============================================================================
# SIGNAL FACTORY - Converts commands/prompts to signals
# =============================================================================

@dataclass
class CommandSignal:
    """Signal generated from a command/prompt."""
    signal_id: str = field(default_factory=lambda: f"sig_{uuid.uuid4().hex[:8]}")
    signal_type: str = "command"
    command: str = ""
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)


class SignalFactory:
    """
    Converts prompts/commands into signals.
    
    prompt → parse → signal
    """
    
    def __init__(self):
        self.parsers: Dict[str, Callable[[str], CommandSignal]] = {}
        self._default_parser = self._simple_parse
    
    def register_parser(self, prefix: str, parser: Callable[[str], CommandSignal]):
        """Register command parser for prefix."""
        self.parsers[prefix] = parser
    
    def create_signal(self, prompt: str, source: str = "") -> CommandSignal:
        """Create signal from prompt string."""
        # Find matching parser
        for prefix, parser in self.parsers.items():
            if prompt.startswith(prefix):
                sig = parser(prompt)
                sig.source = source
                return sig
        
        # Default parsing
        return self._default_parser(prompt, source)
    
    def _simple_parse(self, prompt: str, source: str = "") -> CommandSignal:
        """Simple space-delimited parsing."""
        parts = prompt.strip().split()
        command = parts[0] if parts else ""
        args = tuple(parts[1:]) if len(parts) > 1 else ()
        
        return CommandSignal(
            command=command,
            args=args,
            source=source
        )
    
    def from_dict(self, data: Dict[str, Any]) -> CommandSignal:
        """Create signal from dict (for structured input)."""
        return CommandSignal(
            command=data.get('command', ''),
            args=tuple(data.get('args', [])),
            kwargs=data.get('kwargs', {}),
            source=data.get('source', '')
        )


# =============================================================================
# I/O BRIDGE - Connects messaging to signal pipeline
# =============================================================================

@dataclass
class IOBridge:
    """
    Bridge between messaging I/O and signal system.
    
    Acts as a connector that:
    - Reads from message buffer
    - Converts to signals via factory
    - Pushes to update cycle
    """
    bridge_id: str = field(default_factory=lambda: f"bridge_{uuid.uuid4().hex[:8]}")
    
    input_buffer: MessageBuffer = field(default_factory=MessageBuffer)
    output_buffer: MessageBuffer = field(default_factory=MessageBuffer)
    signal_factory: SignalFactory = field(default_factory=SignalFactory)
    
    # Hooks for logic execution
    pre_process: Optional[Callable[[BufferedMessage], BufferedMessage]] = None
    post_process: Optional[Callable[[CommandSignal], CommandSignal]] = None
    
    # Stats
    messages_bridged: int = 0
    
    def bridge(self, message: BufferedMessage) -> Optional[CommandSignal]:
        """Bridge a message to a signal."""
        # Pre-process hook
        if self.pre_process:
            message = self.pre_process(message)
        
        # Convert to signal
        content = message.content
        if isinstance(content, str):
            signal = self.signal_factory.create_signal(content, message.source)
        elif isinstance(content, dict):
            signal = self.signal_factory.from_dict(content)
        else:
            signal = CommandSignal(command=str(content), source=message.source)
        
        # Post-process hook
        if self.post_process:
            signal = self.post_process(signal)
        
        self.messages_bridged += 1
        return signal
    
    def process_pending(self) -> List[CommandSignal]:
        """Process all pending input messages to signals."""
        signals = []
        while msg := self.input_buffer.pop():
            if sig := self.bridge(msg):
                signals.append(sig)
        return signals


# =============================================================================
# MESSAGING INTERFACE - The main interface
# =============================================================================

class MessagingInterface:
    """
    Main messaging interface with:
    - Own buffer
    - Update cycle registration
    - I/O bridge to signals
    - Logic execution points
    """
    
    def __init__(
        self,
        interface_id: str = None,
        buffer_capacity: int = 1000
    ):
        self.interface_id = interface_id or f"msgif_{uuid.uuid4().hex[:8]}"
        
        # Core components
        self.buffer = MessageBuffer(capacity=buffer_capacity)
        self.signal_factory = SignalFactory()
        self.bridge = IOBridge(
            input_buffer=self.buffer,
            signal_factory=self.signal_factory
        )
        
        # Update cycle
        self._update_handlers: List[Callable[[], bool]] = []
        self._running = False
        self._update_interval = 0.016  # ~60fps default
        
        # Output targets
        self._signal_targets: List[Emittable] = []
        
        # Logic hooks - run arbitrary logic during update
        self._logic_hooks: Dict[str, Callable[[MessagingInterface], Any]] = {}
    
    # -------------------------------------------------------------------------
    # Message Input
    # -------------------------------------------------------------------------
    
    def receive(self, content: Any, source: str = "", priority: int = 0) -> BufferedMessage:
        """Receive a message into the buffer."""
        return self.buffer.push(content, source, priority)
    
    def receive_prompt(self, prompt: str, source: str = "user") -> BufferedMessage:
        """Convenience for receiving prompt strings."""
        return self.receive(prompt, source)
    
    # -------------------------------------------------------------------------
    # Update Cycle
    # -------------------------------------------------------------------------
    
    def register_update_handler(self, handler: Callable[[], bool]):
        """Register handler to be called each update."""
        self._update_handlers.append(handler)
    
    def add_signal_target(self, target: Emittable):
        """Add target that will receive generated signals."""
        self._signal_targets.append(target)
    
    def update(self) -> Dict[str, Any]:
        """
        Single update tick:
        1. Process buffered messages → signals
        2. Run logic hooks
        3. Emit signals to targets
        4. Run registered update handlers
        """
        results = {
            'signals_generated': 0,
            'signals_emitted': 0,
            'hooks_run': 0,
            'handlers_run': 0
        }
        
        # 1. Bridge messages to signals
        signals = self.bridge.process_pending()
        results['signals_generated'] = len(signals)
        
        # 2. Run logic hooks
        for name, hook in self._logic_hooks.items():
            try:
                hook(self)
                results['hooks_run'] += 1
            except Exception as e:
                print(f"Logic hook '{name}' failed: {e}")
        
        # 3. Emit signals
        for signal in signals:
            for target in self._signal_targets:
                try:
                    target.emit(signal)
                    results['signals_emitted'] += 1
                except Exception as e:
                    print(f"Signal emit failed: {e}")
        
        # 4. Run update handlers
        for handler in self._update_handlers:
            try:
                handler()
                results['handlers_run'] += 1
            except Exception as e:
                print(f"Update handler failed: {e}")
        
        return results
    
    async def run_loop(self, interval: float = None):
        """Run the update loop asynchronously."""
        self._running = True
        interval = interval or self._update_interval
        
        while self._running:
            self.update()
            await asyncio.sleep(interval)
    
    def stop(self):
        """Stop the update loop."""
        self._running = False
    
    # -------------------------------------------------------------------------
    # Logic Hooks - Run arbitrary logic
    # -------------------------------------------------------------------------
    
    def add_logic_hook(self, name: str, hook: Callable[[MessagingInterface], Any]):
        """Add a logic hook that runs each update."""
        self._logic_hooks[name] = hook
    
    def remove_logic_hook(self, name: str):
        """Remove a logic hook."""
        self._logic_hooks.pop(name, None)


# =============================================================================
# CONNECTOR BRIDGE - Links two connectors for I/O flow
# =============================================================================

class ConnectorBridge:
    """
    Bridges two systems via their connectors.
    
    messaging_interface <--bridge--> signal_system
    
    This is the "second connector function" you mentioned.
    """
    
    def __init__(
        self,
        source: MessagingInterface,
        target: Any,  # Anything with emit() or similar
        transform: Callable[[CommandSignal], Any] = None
    ):
        self.source = source
        self.target = target
        self.transform = transform or (lambda x: x)
        
        # Wire up
        self._wire()
    
    def _wire(self):
        """Wire the source to target."""
        # Create an emittable wrapper for target
        class TargetWrapper:
            def __init__(wrapper_self, target, transform):
                wrapper_self.target = target
                wrapper_self.transform = transform
            
            def emit(wrapper_self, signal):
                transformed = wrapper_self.transform(signal)
                
                # Try various emit methods
                if hasattr(wrapper_self.target, 'emit'):
                    return wrapper_self.target.emit(transformed)
                elif hasattr(wrapper_self.target, 'send'):
                    return wrapper_self.target.send(transformed)
                elif hasattr(wrapper_self.target, 'receive'):
                    return wrapper_self.target.receive(transformed)
                elif callable(wrapper_self.target):
                    return wrapper_self.target(transformed)
        
        wrapper = TargetWrapper(self.target, self.transform)
        self.source.add_signal_target(wrapper)


# =============================================================================
# EXAMPLE: Full pipeline
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    # Create messaging interface
    messaging = MessagingInterface()
    
    # Add a custom command parser
    def slash_parser(prompt: str) -> CommandSignal:
        """Parse /command args format."""
        parts = prompt[1:].split()  # Skip the /
        return CommandSignal(
            signal_type="slash_command",
            command=parts[0] if parts else "",
            args=tuple(parts[1:])
        )
    
    messaging.signal_factory.register_parser("/", slash_parser)
    
    # Add logic hook
    def debug_hook(interface: MessagingInterface):
        if interface.buffer.pending > 0:
            print(f"[DEBUG] {interface.buffer.pending} messages pending")
    
    messaging.add_logic_hook("debug", debug_hook)
    
    # Create a simple signal receiver
    class SignalReceiver:
        def __init__(self):
            self.received = []
        
        def emit(self, signal: CommandSignal):
            self.received.append(signal)
            print(f"[RECV] {signal.signal_type}: {signal.command} {signal.args}")
    
    receiver = SignalReceiver()
    
    # Bridge messaging to receiver
    bridge = ConnectorBridge(messaging, receiver)
    
    # Send some prompts
    messaging.receive_prompt("/attack goblin")
    messaging.receive_prompt("/inventory show")
    messaging.receive_prompt("look around")
    
    # Run one update cycle
    results = messaging.update()
    print(f"\nUpdate results: {results}")
    print(f"\nReceiver got {len(receiver.received)} signals")
    
    # Verify
    for sig in receiver.received:
        print(f"  - {sig.command}: {sig.args}")
