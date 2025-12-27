"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║              T H E   H E R O ' S   S P I R I T   S T I C K                    ║
║                                                                               ║
║  The spirit stick IS the chronicle. Every message spoken while holding       ║
║  it gets carved into the stick itself. Time moves forward. History           ║
║  accumulates. The hero leaves their story behind in the wood.                ║
║                                                                               ║
║  "The stick remembers what the hero forgets."                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

ARCHITECTURE:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │   MESSAGING INTERFACE (stdin / simulated)                               │
    │            │                                                            │
    │            ▼                                                            │
    │   ┌─────────────────┐                                                   │
    │   │  MESSAGE QUEUE  │  ← incoming commands/events                       │
    │   └────────┬────────┘                                                   │
    │            │                                                            │
    │            ▼                                                            │
    │   ┌─────────────────┐      ┌─────────────────┐                          │
    │   │  SPIRIT STICK   │ ──── │   SQLite DB     │  ← THE CHRONICLE         │
    │   │  (event loop)   │      │   (messages)    │                          │
    │   └────────┬────────┘      └─────────────────┘                          │
    │            │                                                            │
    │            ▼                                                            │
    │   ┌─────────────────┐                                                   │
    │   │   HERO ACTOR    │  ← protagonist moving through time                │
    │   │   (handlers)    │                                                   │
    │   └────────┬────────┘                                                   │
    │            │                                                            │
    │            ▼                                                            │
    │   ┌─────────────────┐                                                   │
    │   │  QUEST STATE    │  ← where we are in the story                      │
    │   │  (buffer)       │                                                   │
    │   └─────────────────┘                                                   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    List, Dict, Optional, Any, Callable, Tuple, 
    Iterator, Union, Protocol
)
from abc import ABC, abstractmethod
from enum import Enum, auto
from collections import deque
import sqlite3
import json
import time
import uuid
import sys
import os
from contextlib import contextmanager
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class MessageType(Enum):
    """Types of messages that flow through the system."""
    
    # Player/user input
    COMMAND = "command"           # go north, take sword
    SPEECH = "speech"             # say "hello"
    QUERY = "query"               # look, inventory
    
    # System events
    NARRATION = "narration"       # story prose
    STATE_CHANGE = "state_change" # hero moved, item taken
    DISCOVERY = "discovery"       # found something
    ENCOUNTER = "encounter"       # met someone/something
    
    # Quest progression
    MILESTONE = "milestone"       # reached a quest stage
    MEMORY = "memory"             # hero remembers something
    PROPHECY = "prophecy"         # foreshadowing
    
    # Meta
    SYSTEM = "system"             # system messages
    ERROR = "error"               # errors


@dataclass
class Message:
    """
    A single message in the chronicle.
    
    Messages are immutable once created. They represent
    a moment in time - something said, done, or observed.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # What kind of message
    msg_type: MessageType = MessageType.NARRATION
    
    # The content
    content: str = ""
    
    # Structured data (optional)
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Who/what generated this message
    source: str = "narrator"
    
    # When in story time (sequential, not wall clock)
    story_time: int = 0
    
    # When in real time
    timestamp: float = field(default_factory=time.time)
    
    # Location in the story world
    location: Optional[str] = None
    
    # Tags for filtering/querying
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'msg_type': self.msg_type.value,
            'content': self.content,
            'data': self.data,
            'source': self.source,
            'story_time': self.story_time,
            'timestamp': self.timestamp,
            'location': self.location,
            'tags': self.tags,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Message':
        d = d.copy()
        d['msg_type'] = MessageType(d['msg_type'])
        return cls(**d)
    
    def __repr__(self):
        return f"[T{self.story_time}] {self.source}: {self.content[:50]}..."


# ═══════════════════════════════════════════════════════════════════════════════
# THE CHRONICLE: Spirit Stick with SQLite Storage
# ═══════════════════════════════════════════════════════════════════════════════

class Chronicle:
    """
    The Spirit Stick as a database.
    
    Every message spoken while holding the stick gets carved
    into SQLite. The chronicle IS the history. The stick
    remembers everything.
    
    This is "improper" in the sense that the token (permission
    to speak) is coupled with the storage (record of speech).
    But for a narrative system, this coupling is POETIC:
    
        "You may only speak if you hold the stick,
         and the stick remembers all who spoke."
    """
    
    def __init__(self, db_path: str = ":memory:"):
        """
        Create or connect to the chronicle.
        
        Args:
            db_path: SQLite path. Use ":memory:" for ephemeral,
                     or a file path for persistent chronicle.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Current story time (monotonically increasing)
        self._story_time: int = 0
        
        # Who currently holds the stick
        self._holder: Optional[str] = None
        
        # Is the chronicle active (accepting messages)?
        self._active: bool = False
        
        # Message queue (incoming, not yet chronicled)
        self._queue: deque[Message] = deque()
        
        # Handlers for different message types
        self._handlers: Dict[MessageType, List[Callable[[Message], Any]]] = {
            mt: [] for mt in MessageType
        }
        
        # Initialize the database
        self._init_db()
    
    def _init_db(self):
        """Create the chronicle tables."""
        self.conn.executescript("""
            -- The main chronicle: all messages in time order
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                msg_type TEXT NOT NULL,
                content TEXT NOT NULL,
                data TEXT,  -- JSON
                source TEXT NOT NULL,
                story_time INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                location TEXT,
                tags TEXT  -- JSON array
            );
            
            -- Index for time-based queries
            CREATE INDEX IF NOT EXISTS idx_story_time 
                ON messages(story_time);
            
            -- Index for source-based queries  
            CREATE INDEX IF NOT EXISTS idx_source 
                ON messages(source);
            
            -- Index for type-based queries
            CREATE INDEX IF NOT EXISTS idx_type 
                ON messages(msg_type);
            
            -- Index for location-based queries
            CREATE INDEX IF NOT EXISTS idx_location 
                ON messages(location);
            
            -- Metadata table (story state, etc)
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        self.conn.commit()
        
        # Load story time from meta if exists
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = 'story_time'"
        ).fetchone()
        if row:
            self._story_time = int(row[0])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Token (stick) management
    # ─────────────────────────────────────────────────────────────────────────
    
    def acquire(self, holder_id: str) -> bool:
        """
        Acquire the spirit stick.
        
        Only the holder can chronicle messages.
        """
        if self._holder is not None and self._holder != holder_id:
            return False  # Someone else has it
        self._holder = holder_id
        self._active = True
        return True
    
    def release(self) -> bool:
        """Release the spirit stick."""
        if self._holder is None:
            return False
        self._holder = None
        return True
    
    def is_held_by(self, holder_id: str) -> bool:
        """Check if specific entity holds the stick."""
        return self._holder == holder_id
    
    @property
    def holder(self) -> Optional[str]:
        return self._holder
    
    # ─────────────────────────────────────────────────────────────────────────
    # Message chronicling (writing to the stick)
    # ─────────────────────────────────────────────────────────────────────────
    
    def chronicle(self, message: Message) -> bool:
        """
        Carve a message into the stick.
        
        The message gets a story_time stamp and is permanently
        recorded in the database.
        """
        if not self._active:
            return False
        
        # Advance story time
        self._story_time += 1
        message.story_time = self._story_time
        
        # Insert into database
        self.conn.execute("""
            INSERT INTO messages 
                (id, msg_type, content, data, source, story_time, 
                 timestamp, location, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message.id,
            message.msg_type.value,
            message.content,
            json.dumps(message.data),
            message.source,
            message.story_time,
            message.timestamp,
            message.location,
            json.dumps(message.tags),
        ))
        
        # Update meta
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('story_time', ?)",
            (str(self._story_time),)
        )
        self.conn.commit()
        
        # Fire handlers
        for handler in self._handlers.get(message.msg_type, []):
            try:
                handler(message)
            except Exception as e:
                print(f"Handler error: {e}", file=sys.stderr)
        
        return True
    
    def speak(self, content: str, 
              msg_type: MessageType = MessageType.NARRATION,
              source: str = None,
              location: str = None,
              **data) -> Optional[Message]:
        """
        Convenience: create and chronicle a message.
        
        Only works if someone holds the stick.
        """
        if self._holder is None:
            return None
        
        message = Message(
            msg_type=msg_type,
            content=content,
            source=source or self._holder,
            location=location,
            data=data,
        )
        
        if self.chronicle(message):
            return message
        return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Querying the chronicle (reading the carvings)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _row_to_message(self, row: sqlite3.Row) -> Message:
        """Convert a database row to a Message."""
        return Message(
            id=row['id'],
            msg_type=MessageType(row['msg_type']),
            content=row['content'],
            data=json.loads(row['data']) if row['data'] else {},
            source=row['source'],
            story_time=row['story_time'],
            timestamp=row['timestamp'],
            location=row['location'],
            tags=json.loads(row['tags']) if row['tags'] else [],
        )
    
    def get_all(self, limit: int = None) -> List[Message]:
        """Get all messages in time order."""
        query = "SELECT * FROM messages ORDER BY story_time"
        if limit:
            query += f" LIMIT {limit}"
        rows = self.conn.execute(query).fetchall()
        return [self._row_to_message(r) for r in rows]
    
    def get_recent(self, n: int = 10) -> List[Message]:
        """Get the n most recent messages."""
        rows = self.conn.execute("""
            SELECT * FROM messages 
            ORDER BY story_time DESC 
            LIMIT ?
        """, (n,)).fetchall()
        return [self._row_to_message(r) for r in reversed(rows)]
    
    def get_by_source(self, source: str) -> List[Message]:
        """Get all messages from a specific source."""
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE source = ? ORDER BY story_time",
            (source,)
        ).fetchall()
        return [self._row_to_message(r) for r in rows]
    
    def get_by_type(self, msg_type: MessageType) -> List[Message]:
        """Get all messages of a specific type."""
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE msg_type = ? ORDER BY story_time",
            (msg_type.value,)
        ).fetchall()
        return [self._row_to_message(r) for r in rows]
    
    def get_by_location(self, location: str) -> List[Message]:
        """Get all messages from a specific location."""
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE location = ? ORDER BY story_time",
            (location,)
        ).fetchall()
        return [self._row_to_message(r) for r in rows]
    
    def get_range(self, start_time: int, end_time: int) -> List[Message]:
        """Get messages in a story time range."""
        rows = self.conn.execute("""
            SELECT * FROM messages 
            WHERE story_time >= ? AND story_time <= ?
            ORDER BY story_time
        """, (start_time, end_time)).fetchall()
        return [self._row_to_message(r) for r in rows]
    
    def search(self, term: str) -> List[Message]:
        """Search message content."""
        rows = self.conn.execute("""
            SELECT * FROM messages 
            WHERE content LIKE ?
            ORDER BY story_time
        """, (f"%{term}%",)).fetchall()
        return [self._row_to_message(r) for r in rows]
    
    @property
    def story_time(self) -> int:
        """Current story time."""
        return self._story_time
    
    @property
    def message_count(self) -> int:
        """Total messages in chronicle."""
        row = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        return row[0]
    
    # ─────────────────────────────────────────────────────────────────────────
    # Handler registration
    # ─────────────────────────────────────────────────────────────────────────
    
    def on(self, msg_type: MessageType, 
           handler: Callable[[Message], Any]) -> 'Chronicle':
        """Register a handler for a message type."""
        self._handlers[msg_type].append(handler)
        return self
    
    def on_any(self, handler: Callable[[Message], Any]) -> 'Chronicle':
        """Register a handler for all message types."""
        for mt in MessageType:
            self._handlers[mt].append(handler)
        return self
    
    # ─────────────────────────────────────────────────────────────────────────
    # Buffer: Recent history for context
    # ─────────────────────────────────────────────────────────────────────────
    
    def buffer(self, n: int = 5) -> str:
        """
        Get recent history as formatted text.
        
        This is what the hero "remembers" - recent context
        for making decisions.
        """
        messages = self.get_recent(n)
        lines = []
        for m in messages:
            prefix = f"[{m.msg_type.name}]" if m.msg_type != MessageType.NARRATION else ""
            lines.append(f"{prefix} {m.content}".strip())
        return "\n".join(lines)
    
    def __repr__(self):
        return f"<Chronicle {self.message_count} messages, T={self._story_time}>"


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGING INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class MessagingInterface(ABC):
    """
    Abstract interface for receiving messages.
    
    Could be stdin, a socket, a queue, etc.
    """
    
    @abstractmethod
    def receive(self) -> Optional[str]:
        """
        Receive the next message (blocking or non-blocking).
        Returns None if no message available.
        """
        ...
    
    @abstractmethod
    def has_message(self) -> bool:
        """Check if a message is available."""
        ...


class StdinInterface(MessagingInterface):
    """Messaging interface from standard input."""
    
    def __init__(self, prompt: str = "> "):
        self.prompt = prompt
    
    def receive(self) -> Optional[str]:
        try:
            return input(self.prompt).strip()
        except EOFError:
            return None
    
    def has_message(self) -> bool:
        return True  # stdin always "has" messages (blocks)


class QueueInterface(MessagingInterface):
    """Messaging interface from a queue (for testing/simulation)."""
    
    def __init__(self, messages: List[str] = None):
        self._queue: deque[str] = deque(messages or [])
    
    def push(self, message: str):
        """Add a message to the queue."""
        self._queue.append(message)
    
    def receive(self) -> Optional[str]:
        if self._queue:
            return self._queue.popleft()
        return None
    
    def has_message(self) -> bool:
        return len(self._queue) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# THE HERO ACTOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuestState:
    """
    The hero's current state in the quest.
    
    This is the "buffer" - what the hero currently knows/has.
    """
    location: str = "the beginning"
    inventory: List[str] = field(default_factory=list)
    flags: Dict[str, bool] = field(default_factory=dict)
    health: int = 100
    stage: int = 0  # Quest stage
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'location': self.location,
            'inventory': self.inventory,
            'flags': self.flags,
            'health': self.health,
            'stage': self.stage,
        }


class HeroActor:
    """
    The protagonist moving through time.
    
    The hero:
        - Holds the spirit stick when acting
        - Receives messages from the interface
        - Has handlers that process messages
        - Leaves history in the chronicle
        - Maintains quest state (the buffer)
    """
    
    def __init__(self, name: str, chronicle: Chronicle):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.chronicle = chronicle
        self.state = QuestState()
        
        # Command handlers: command -> handler function
        self._commands: Dict[str, Callable[['HeroActor', List[str]], str]] = {}
        
        # Register default commands
        self._register_defaults()
    
    def _register_defaults(self):
        """Register default command handlers."""
        
        @self.command("look")
        def cmd_look(hero, args):
            return f"You are at {hero.state.location}. {hero._describe_location()}"
        
        @self.command("inventory", "inv", "i")
        def cmd_inventory(hero, args):
            if not hero.state.inventory:
                return "You carry nothing."
            items = ", ".join(hero.state.inventory)
            return f"You carry: {items}"
        
        @self.command("go", "move", "walk")
        def cmd_go(hero, args):
            if not args:
                return "Go where?"
            direction = args[0].lower()
            old_loc = hero.state.location
            hero.state.location = f"{old_loc} → {direction}"
            return f"You travel {direction}..."
        
        @self.command("take", "get", "grab")
        def cmd_take(hero, args):
            if not args:
                return "Take what?"
            item = " ".join(args)
            hero.state.inventory.append(item)
            return f"You take the {item}."
        
        @self.command("say", "speak")
        def cmd_say(hero, args):
            if not args:
                return "Say what?"
            speech = " ".join(args)
            return f'You say: "{speech}"'
        
        @self.command("remember", "recall")
        def cmd_remember(hero, args):
            n = int(args[0]) if args else 5
            return "You remember...\n" + hero.chronicle.buffer(n)
        
        @self.command("help", "?")
        def cmd_help(hero, args):
            cmds = ", ".join(sorted(hero._commands.keys()))
            return f"Commands: {cmds}"
    
    def command(self, *names: str):
        """Decorator to register a command handler."""
        def decorator(fn: Callable[['HeroActor', List[str]], str]):
            for name in names:
                self._commands[name.lower()] = fn
            return fn
        return decorator
    
    def _describe_location(self) -> str:
        """Get description of current location from chronicle."""
        messages = self.chronicle.get_by_location(self.state.location)
        if messages:
            return messages[-1].content
        return "An unremarkable place."
    
    # ─────────────────────────────────────────────────────────────────────────
    # Acting (holding the stick and speaking)
    # ─────────────────────────────────────────────────────────────────────────
    
    def act(self, raw_input: str) -> Optional[Message]:
        """
        Process input and chronicle the result.
        
        This is where the hero holds the stick and speaks.
        """
        # Acquire the spirit stick
        if not self.chronicle.acquire(self.id):
            return None
        
        try:
            # Parse the input
            parts = raw_input.strip().split()
            if not parts:
                return None
            
            cmd = parts[0].lower()
            args = parts[1:]
            
            # Log the command itself
            self.chronicle.speak(
                f"> {raw_input}",
                msg_type=MessageType.COMMAND,
                source=self.name,
                location=self.state.location,
                command=cmd,
                args=args,
            )
            
            # Find and execute handler
            handler = self._commands.get(cmd)
            if handler:
                result = handler(self, args)
                msg_type = MessageType.NARRATION
            else:
                result = f"You don't know how to '{cmd}'."
                msg_type = MessageType.ERROR
            
            # Chronicle the result
            message = self.chronicle.speak(
                result,
                msg_type=msg_type,
                source="narrator",
                location=self.state.location,
            )
            
            return message
            
        finally:
            # Always release the stick
            self.chronicle.release()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Event loop integration
    # ─────────────────────────────────────────────────────────────────────────
    
    def run(self, interface: MessagingInterface, 
            max_turns: int = None,
            on_message: Callable[[Message], None] = None):
        """
        Run the hero's event loop.
        
        Receives from interface, acts, chronicles.
        The for loop IS the event loop.
        The spirit stick IS the chronicle.
        """
        # Opening
        self.chronicle.acquire(self.id)
        self.chronicle.speak(
            f"{self.name} begins their quest...",
            msg_type=MessageType.MILESTONE,
            source="narrator",
            location=self.state.location,
        )
        self.chronicle.release()
        
        turns = 0
        while max_turns is None or turns < max_turns:
            # Receive input
            raw = interface.receive()
            if raw is None:
                break
            if raw.lower() in ('quit', 'exit', 'q'):
                break
            
            # Act (acquire stick, process, chronicle, release)
            message = self.act(raw)
            
            if message and on_message:
                on_message(message)
            
            turns += 1
        
        # Closing
        self.chronicle.acquire(self.id)
        self.chronicle.speak(
            f"{self.name}'s chapter ends. {turns} moments passed.",
            msg_type=MessageType.MILESTONE,
            source="narrator",
            location=self.state.location,
        )
        self.chronicle.release()
    
    def __repr__(self):
        return f"<Hero '{self.name}' at {self.state.location}>"


# ═══════════════════════════════════════════════════════════════════════════════
# QUEST NARRATOR: Adds story beats
# ═══════════════════════════════════════════════════════════════════════════════

class QuestNarrator:
    """
    Adds narrative flavor to the chronicle.
    
    Listens to events and injects story beats,
    descriptions, foreshadowing.
    """
    
    def __init__(self, chronicle: Chronicle):
        self.chronicle = chronicle
        self.id = "narrator"
        
        # Register handlers
        chronicle.on(MessageType.COMMAND, self._on_command)
        chronicle.on(MessageType.STATE_CHANGE, self._on_state_change)
    
    def _on_command(self, message: Message):
        """React to player commands with flavor."""
        cmd = message.data.get('command', '')
        
        # Add atmospheric responses
        if cmd == 'look':
            self._maybe_add_atmosphere(message.location)
    
    def _on_state_change(self, message: Message):
        """React to state changes."""
        pass
    
    def _maybe_add_atmosphere(self, location: str):
        """Occasionally add atmospheric description."""
        import random
        if random.random() < 0.3:
            atmospheres = [
                "A cold wind stirs.",
                "Shadows lengthen.",
                "You feel watched.",
                "Time seems to slow.",
                "Something glimmers in the distance.",
            ]
            self.chronicle.acquire(self.id)
            self.chronicle.speak(
                random.choice(atmospheres),
                msg_type=MessageType.NARRATION,
                source="narrator",
                location=location,
            )
            self.chronicle.release()
    
    def narrate(self, text: str, location: str = None):
        """Inject narration."""
        self.chronicle.acquire(self.id)
        self.chronicle.speak(
            text,
            msg_type=MessageType.NARRATION,
            source="narrator",
            location=location,
        )
        self.chronicle.release()


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def create_console_logger() -> Callable[[Message], None]:
    """Create a handler that logs to console."""
    def handler(message: Message):
        timestamp = datetime.fromtimestamp(message.timestamp).strftime("%H:%M:%S")
        type_prefix = f"[{message.msg_type.name}]" if message.msg_type != MessageType.NARRATION else ""
        print(f"{type_prefix} {message.content}")
    return handler


def create_file_logger(path: str) -> Callable[[Message], None]:
    """Create a handler that logs to file."""
    def handler(message: Message):
        with open(path, 'a') as f:
            line = json.dumps(message.to_dict())
            f.write(line + '\n')
    return handler


def create_buffer_logger(buffer: List[Message]) -> Callable[[Message], None]:
    """Create a handler that appends to a list."""
    def handler(message: Message):
        buffer.append(message)
    return handler


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE: Running a quest
# ═══════════════════════════════════════════════════════════════════════════════

def example_simulated_quest():
    """Run a simulated quest with predefined inputs."""
    print("\n" + "═" * 70)
    print("THE HERO'S QUEST (Simulated)")
    print("═" * 70 + "\n")
    
    # Create the chronicle (spirit stick with database)
    chronicle = Chronicle("quest_chronicle.db")
    
    # Create the hero
    hero = HeroActor("Aria", chronicle)
    
    # Create narrator for flavor
    narrator = QuestNarrator(chronicle)
    
    # Set up logging
    chronicle.on_any(create_console_logger())
    
    # Opening narration
    narrator.narrate(
        "In a land where memory is carved into wood, "
        "a hero begins their journey...",
        location="the beginning"
    )
    
    # Simulated player inputs
    inputs = QueueInterface([
        "look",
        "go north",
        "look",
        "take ancient sword",
        "inventory",
        "say I will complete this quest",
        "go east",
        "remember 3",
    ])
    
    # Run the quest
    hero.run(inputs)
    
    # Show the chronicle
    print("\n" + "─" * 70)
    print("THE CHRONICLE (from database):")
    print("─" * 70)
    
    for msg in chronicle.get_all():
        print(f"  T{msg.story_time:03d} [{msg.source}] {msg.content[:60]}")
    
    print(f"\nTotal messages: {chronicle.message_count}")
    print(f"Story time reached: {chronicle.story_time}")
    
    # Clean up
    os.remove("quest_chronicle.db")


def example_interactive_quest():
    """Run an interactive quest from stdin."""
    print("\n" + "═" * 70)
    print("THE HERO'S QUEST (Interactive)")
    print("═" * 70)
    print("Commands: look, go <dir>, take <item>, inventory, say <msg>, remember, help")
    print("Type 'quit' to end.\n")
    
    chronicle = Chronicle()  # In-memory for interactive
    hero = HeroActor("You", chronicle)
    narrator = QuestNarrator(chronicle)
    
    chronicle.on_any(create_console_logger())
    
    narrator.narrate("Your quest begins...")
    
    hero.run(StdinInterface(">>> "))
    
    print("\n" + "─" * 70)
    print("Your chronicle:")
    print("─" * 70)
    print(chronicle.buffer(10))


# ═══════════════════════════════════════════════════════════════════════════════
# THE PATTERN: What we've built
# ═══════════════════════════════════════════════════════════════════════════════

"""
THE SPIRIT STICK CHRONICLE PATTERN
══════════════════════════════════

    Token (permission) + Storage (chronicle) = Spirit Stick
    
    The stick grants permission to speak AND remembers what was said.
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │   Traditional separation:                                               │
    │       Token (mutex/semaphore)  ───────  Storage (database)              │
    │       "Who can write"                   "What was written"              │
    │                                                                         │
    │   Spirit stick coupling:                                                │
    │       SpiritStick = Token + Storage                                     │
    │       "Holding grants write permission, writing is permanent"           │
    │                                                                         │
    │   Why this works for narrative:                                         │
    │       - Stories ARE their history                                       │
    │       - The chronicle IS the story                                      │
    │       - Permission to speak = permission to change history              │
    │       - The stick IS the book being written                             │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

COMPONENTS:
    
    1. Chronicle (spirit_stick.py)
       - SQLite storage
       - Token management (acquire/release)
       - Message chronicling
       - Query methods
       - Handler registration
    
    2. Message
       - Immutable record
       - Story time (sequential)
       - Type, source, location
       - Content and data
    
    3. HeroActor  
       - Holds stick when acting
       - Command handlers
       - Quest state (buffer)
       - Event loop integration
    
    4. MessagingInterface
       - Stdin or queue
       - Provides input
       - Could be network, LLM, etc.
    
    5. Handlers
       - Console logger
       - File logger
       - Buffer logger
       - Custom handlers

EVENT LOOP AS ITERATION:
    
    for input in interface:
        hero.acquire_stick()
        message = hero.process(input)
        chronicle.store(message)  # same object!
        hero.release_stick()
        handlers.notify(message)
    
    The iterator (interface) feeds the event loop.
    The spirit stick (chronicle) IS the storage.
    The for loop IS the game loop.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run simulated quest (non-interactive)
    example_simulated_quest()
    
    # Uncomment to run interactive:
    # example_interactive_quest()
