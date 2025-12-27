"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║              T H E   O R A C L E ' S   T O W E R                              ║
║                                                                               ║
║  LLM-to-LLM narrative generation through the spirit stick.                   ║
║  Multiple oracles pass the stick, each contributing their thread.            ║
║  The Resolver weaves position into prose.                                    ║
║                                                                               ║
║  "Three oracles speak through one flame."                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    List, Dict, Optional, Any, Callable, Iterator, 
    Protocol, Tuple, Union, runtime_checkable
)
from abc import ABC, abstractmethod
from enum import Enum, auto
from collections import deque
import sqlite3
import json
import time
import uuid
import re


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGING INTERFACE: Generic LLM Gateway
# ═══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class LLMInterface(Protocol):
    """
    Generic interface to any LLM backend.
    
    Implementations could connect to:
        - Local Ollama
        - OpenAI API
        - Anthropic API
        - Mock for testing
    """
    
    def send(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Send prompt, receive response."""
        ...
    
    def stream(self, prompt: str, context: Dict[str, Any] = None) -> Iterator[str]:
        """Stream response tokens."""
        ...
    
    @property
    def model_name(self) -> str:
        """Get model identifier."""
        ...


class MockLLM:
    """Mock LLM for testing and development."""
    
    def __init__(self, name: str = "mock", responses: Dict[str, str] = None):
        self._name = name
        self._responses = responses or {}
        self._call_count = 0
    
    def send(self, prompt: str, context: Dict[str, Any] = None) -> str:
        self._call_count += 1
        
        # Check for keyword-based responses
        for key, response in self._responses.items():
            if key.lower() in prompt.lower():
                return response
        
        # Default response based on model name
        if "narrator" in self._name.lower():
            return f"The hero surveys the scene, sensing adventure ahead."
        elif "world" in self._name.lower():
            return f"north: forest, south: village, east: river"
        elif "fate" in self._name.lower():
            return f"A shadow of destiny looms..."
        else:
            return f"[{self._name}] Response to: {prompt[:50]}..."
    
    def stream(self, prompt: str, context: Dict[str, Any] = None) -> Iterator[str]:
        response = self.send(prompt, context)
        for word in response.split():
            yield word + " "
    
    @property
    def model_name(self) -> str:
        return self._name


class OllamaLLM:
    """
    Ollama LLM interface (stub - requires actual Ollama setup).
    
    Would connect to local Ollama instance.
    """
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url
    
    def send(self, prompt: str, context: Dict[str, Any] = None) -> str:
        # In production, this would make HTTP request to Ollama
        # For now, return placeholder
        return f"[Ollama:{self._model}] Would respond to: {prompt[:30]}..."
    
    def stream(self, prompt: str, context: Dict[str, Any] = None) -> Iterator[str]:
        yield self.send(prompt, context)
    
    @property
    def model_name(self) -> str:
        return f"ollama:{self._model}"


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE TYPES (from hero_quest_chronicle.py)
# ═══════════════════════════════════════════════════════════════════════════════

class MessageType(Enum):
    COMMAND = "command"
    SPEECH = "speech"
    QUERY = "query"
    NARRATION = "narration"
    STATE_CHANGE = "state_change"
    DISCOVERY = "discovery"
    ENCOUNTER = "encounter"
    MILESTONE = "milestone"
    MEMORY = "memory"
    PROPHECY = "prophecy"
    POSITION = "position"  # NEW: spatial information
    SYSTEM = "system"
    ERROR = "error"


@dataclass
class Message:
    """A single message in the chronicle."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    msg_type: MessageType = MessageType.NARRATION
    content: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "narrator"
    story_time: int = 0
    timestamp: float = field(default_factory=time.time)
    location: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id, 'msg_type': self.msg_type.value,
            'content': self.content, 'data': self.data,
            'source': self.source, 'story_time': self.story_time,
            'timestamp': self.timestamp, 'location': self.location,
            'tags': self.tags,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# THE CHRONICLE (Spirit Stick with SQLite)
# ═══════════════════════════════════════════════════════════════════════════════

class Chronicle:
    """
    The Spirit Stick as database.
    (Simplified from hero_quest_chronicle.py)
    """
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._story_time: int = 0
        self._holder: Optional[str] = None
        self._active: bool = False
        self._handlers: Dict[MessageType, List[Callable]] = {mt: [] for mt in MessageType}
        self._init_db()
    
    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, msg_type TEXT NOT NULL,
                content TEXT NOT NULL, data TEXT, source TEXT NOT NULL,
                story_time INTEGER NOT NULL, timestamp REAL NOT NULL,
                location TEXT, tags TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_story_time ON messages(story_time);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """)
        self.conn.commit()
    
    def acquire(self, holder_id: str) -> bool:
        if self._holder is not None and self._holder != holder_id:
            return False
        self._holder = holder_id
        self._active = True
        return True
    
    def release(self) -> bool:
        if self._holder is None:
            return False
        self._holder = None
        return True
    
    @property
    def holder(self) -> Optional[str]:
        return self._holder
    
    def chronicle(self, message: Message) -> bool:
        if not self._active:
            return False
        
        self._story_time += 1
        message.story_time = self._story_time
        
        self.conn.execute("""
            INSERT INTO messages (id, msg_type, content, data, source, 
                                  story_time, timestamp, location, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (message.id, message.msg_type.value, message.content,
              json.dumps(message.data), message.source, message.story_time,
              message.timestamp, message.location, json.dumps(message.tags)))
        self.conn.commit()
        
        for handler in self._handlers.get(message.msg_type, []):
            try: handler(message)
            except: pass
        
        return True
    
    def speak(self, content: str, msg_type: MessageType = MessageType.NARRATION,
              source: str = None, location: str = None, **data) -> Optional[Message]:
        if self._holder is None:
            return None
        msg = Message(msg_type=msg_type, content=content,
                      source=source or self._holder, location=location, data=data)
        if self.chronicle(msg):
            return msg
        return None
    
    def buffer(self, n: int = 5) -> str:
        rows = self.conn.execute(
            "SELECT * FROM messages ORDER BY story_time DESC LIMIT ?", (n,)
        ).fetchall()
        return "\n".join(r['content'] for r in reversed(rows))
    
    def get_recent(self, n: int = 10) -> List[Message]:
        rows = self.conn.execute(
            "SELECT * FROM messages ORDER BY story_time DESC LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_msg(r) for r in reversed(rows)]
    
    def _row_to_msg(self, row) -> Message:
        return Message(
            id=row['id'], msg_type=MessageType(row['msg_type']),
            content=row['content'], data=json.loads(row['data'] or '{}'),
            source=row['source'], story_time=row['story_time'],
            timestamp=row['timestamp'], location=row['location'],
            tags=json.loads(row['tags'] or '[]')
        )
    
    def on(self, msg_type: MessageType, handler: Callable) -> 'Chronicle':
        self._handlers[msg_type].append(handler)
        return self
    
    @property
    def story_time(self) -> int:
        return self._story_time


# ═══════════════════════════════════════════════════════════════════════════════
# RESOLVER INPUT/OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResolverInput:
    """Input to the resolver system."""
    command: str = ""
    args: List[str] = field(default_factory=list)
    raw_text: str = ""
    
    # Context
    location: str = "unknown"
    subject: str = "hero"
    context: Dict[str, Any] = field(default_factory=dict)
    
    # From previous processing
    enrichments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObserverResult:
    """Result from a single observer."""
    success: bool = True
    
    # What was resolved
    position: Optional[str] = None
    prose: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    temporal: Optional[str] = None
    
    # Additional data
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class ResolverOutput:
    """Final output from the resolver chain."""
    original: ResolverInput = None
    
    # Accumulated results
    final_prose: str = ""
    resolved_position: str = ""
    all_entities: List[str] = field(default_factory=list)
    temporal_context: str = ""
    
    # Observer results in order
    observer_results: List[Tuple[str, ObserverResult]] = field(default_factory=list)
    
    # Messages to chronicle
    messages: List[Message] = field(default_factory=list)
    
    def merge(self, observer_name: str, result: ObserverResult):
        """Merge an observer's result into the output."""
        self.observer_results.append((observer_name, result))
        
        if result.position:
            self.resolved_position = result.position
        if result.prose:
            self.final_prose = result.prose
        if result.entities:
            self.all_entities.extend(result.entities)
        if result.temporal:
            self.temporal_context = result.temporal


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING OBSERVERS
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessingObserver(ABC):
    """
    Base observer for the resolver chain.
    
    Each observer handles one aspect of resolution.
    They can be composed, reordered, or replaced.
    """
    
    name: str = "base"
    priority: int = 50  # Lower = earlier in chain
    
    @abstractmethod
    def accepts(self, input: ResolverInput) -> bool:
        """Should this observer process this input?"""
        ...
    
    @abstractmethod
    def process(self, input: ResolverInput, 
                current: ResolverOutput) -> ObserverResult:
        """Process the input and return result."""
        ...


class PositionObserver(ProcessingObserver):
    """
    Resolves spatial references in prose.
    
    Uses World LLM to determine positions and directions.
    """
    
    name = "position"
    priority = 10  # Early: position affects everything
    
    def __init__(self, world_llm: LLMInterface):
        self.world = world_llm
    
    def accepts(self, input: ResolverInput) -> bool:
        # Accept movement commands or location queries
        spatial = {'go', 'move', 'walk', 'look', 'where', 'north', 'south', 'east', 'west'}
        return input.command.lower() in spatial or any(a.lower() in spatial for a in input.args)
    
    def process(self, input: ResolverInput, current: ResolverOutput) -> ObserverResult:
        # Ask world LLM about spatial relationships
        prompt = f"""Given the hero is at '{input.location}', 
and they want to go/look '{' '.join(input.args)}',
what is their new position and what exits exist?
Respond in format: position: <place>, exits: <list>"""
        
        response = self.world.send(prompt, input.context)
        
        # Parse response (simplified)
        position = input.location
        if 'position:' in response.lower():
            match = re.search(r'position:\s*([^,\n]+)', response, re.I)
            if match:
                position = match.group(1).strip()
        
        return ObserverResult(
            position=position,
            data={'world_response': response, 'exits': self._parse_exits(response)}
        )
    
    def _parse_exits(self, response: str) -> List[str]:
        if 'exits:' in response.lower():
            match = re.search(r'exits:\s*(.+)', response, re.I)
            if match:
                return [e.strip() for e in match.group(1).split(',')]
        return []


class TemporalObserver(ProcessingObserver):
    """
    Places events in story time.
    
    Tracks day/night, seasons, key events.
    """
    
    name = "temporal"
    priority = 20
    
    def __init__(self):
        self._current_time = "morning"
        self._day = 1
    
    def accepts(self, input: ResolverInput) -> bool:
        return True  # Always track time
    
    def process(self, input: ResolverInput, current: ResolverOutput) -> ObserverResult:
        # Advance time based on action
        action = input.command.lower()
        
        if action in ('rest', 'sleep', 'wait'):
            self._advance_time(large=True)
        else:
            self._advance_time(large=False)
        
        return ObserverResult(
            temporal=f"Day {self._day}, {self._current_time}",
            data={'day': self._day, 'time_of_day': self._current_time}
        )
    
    def _advance_time(self, large: bool):
        times = ['dawn', 'morning', 'midday', 'afternoon', 'evening', 'night']
        idx = times.index(self._current_time) if self._current_time in times else 0
        
        if large:
            idx = (idx + 3) % len(times)
            if idx < times.index(self._current_time):
                self._day += 1
        else:
            idx = (idx + 1) % len(times)
            if idx == 0:
                self._day += 1
        
        self._current_time = times[idx]


class EntityObserver(ProcessingObserver):
    """
    Tracks character and item references.
    """
    
    name = "entity"
    priority = 30
    
    def __init__(self):
        self._known_entities: Dict[str, Dict] = {}
    
    def accepts(self, input: ResolverInput) -> bool:
        return True  # Always track entities
    
    def process(self, input: ResolverInput, current: ResolverOutput) -> ObserverResult:
        # Extract entity mentions from raw text
        entities = self._extract_entities(input.raw_text)
        
        # Track new entities
        for e in entities:
            if e not in self._known_entities:
                self._known_entities[e] = {
                    'first_seen': input.context.get('story_time', 0),
                    'location': input.location
                }
        
        return ObserverResult(entities=entities)
    
    def _extract_entities(self, text: str) -> List[str]:
        # Simple extraction: capitalized words that aren't at sentence start
        words = text.split()
        entities = []
        for i, word in enumerate(words):
            clean = word.strip('.,!?"\'')
            if clean and clean[0].isupper() and i > 0:
                entities.append(clean)
        return list(set(entities))


class ProseObserver(ProcessingObserver):
    """
    Generates and polishes narrative prose.
    
    Uses Narrator LLM to create the story.
    """
    
    name = "prose"
    priority = 40
    
    def __init__(self, narrator_llm: LLMInterface, chronicle: Chronicle):
        self.narrator = narrator_llm
        self.chronicle = chronicle
    
    def accepts(self, input: ResolverInput) -> bool:
        return True  # Always generate prose
    
    def process(self, input: ResolverInput, current: ResolverOutput) -> ObserverResult:
        # Build context from previous observers
        context_parts = [f"Location: {current.resolved_position or input.location}"]
        if current.temporal_context:
            context_parts.append(f"Time: {current.temporal_context}")
        if current.all_entities:
            context_parts.append(f"Present: {', '.join(current.all_entities)}")
        
        # Get recent chronicle for continuity
        recent = self.chronicle.buffer(3)
        
        prompt = f"""Continue this fantasy story.
Context: {'; '.join(context_parts)}
Recent events:
{recent}

The hero's action: {input.command} {' '.join(input.args)}

Write 2-3 sentences of evocative prose describing what happens next."""
        
        prose = self.narrator.send(prompt, input.context)
        
        return ObserverResult(prose=prose)


class FateObserver(ProcessingObserver):
    """
    Adds prophecy and foreshadowing.
    
    The third oracle—speaks of what may come.
    """
    
    name = "fate"
    priority = 45
    
    def __init__(self, fate_llm: LLMInterface):
        self.fate = fate_llm
        self._prophecy_chance = 0.2  # 20% chance to add prophecy
    
    def accepts(self, input: ResolverInput) -> bool:
        # Only add prophecy sometimes, or at milestones
        import random
        milestone_words = {'enter', 'discover', 'find', 'meet', 'take'}
        is_milestone = input.command.lower() in milestone_words
        return is_milestone or random.random() < self._prophecy_chance
    
    def process(self, input: ResolverInput, current: ResolverOutput) -> ObserverResult:
        prompt = f"""You are the Oracle of Fate.
The hero just: {input.command} {' '.join(input.args)}
At location: {current.resolved_position or input.location}

Speak a single cryptic line of prophecy about what this action portends.
Be mysterious but meaningful."""
        
        prophecy = self.fate.send(prompt, input.context)
        
        return ObserverResult(
            data={'prophecy': prophecy}
        )


class ChronicleObserver(ProcessingObserver):
    """
    Final observer—writes everything to the spirit stick.
    """
    
    name = "chronicle"
    priority = 100  # Always last
    
    def __init__(self, chronicle: Chronicle):
        self.chronicle = chronicle
    
    def accepts(self, input: ResolverInput) -> bool:
        return True
    
    def process(self, input: ResolverInput, current: ResolverOutput) -> ObserverResult:
        messages = []
        
        # Chronicle the command
        self.chronicle.acquire("resolver")
        
        cmd_msg = self.chronicle.speak(
            f"> {input.command} {' '.join(input.args)}",
            MessageType.COMMAND,
            source="hero",
            location=current.resolved_position or input.location
        )
        if cmd_msg:
            messages.append(cmd_msg)
        
        # Chronicle position if changed
        if current.resolved_position and current.resolved_position != input.location:
            pos_msg = self.chronicle.speak(
                f"[Moved to: {current.resolved_position}]",
                MessageType.POSITION,
                source="world",
                location=current.resolved_position
            )
            if pos_msg:
                messages.append(pos_msg)
        
        # Chronicle the prose
        if current.final_prose:
            prose_msg = self.chronicle.speak(
                current.final_prose,
                MessageType.NARRATION,
                source="narrator",
                location=current.resolved_position or input.location
            )
            if prose_msg:
                messages.append(prose_msg)
        
        # Chronicle prophecy if present
        for name, result in current.observer_results:
            if 'prophecy' in result.data:
                prophecy_msg = self.chronicle.speak(
                    result.data['prophecy'],
                    MessageType.PROPHECY,
                    source="fate",
                    location=current.resolved_position or input.location
                )
                if prophecy_msg:
                    messages.append(prophecy_msg)
        
        self.chronicle.release()
        
        current.messages = messages
        return ObserverResult(success=True, data={'chronicled': len(messages)})


# ═══════════════════════════════════════════════════════════════════════════════
# THE RESOLVER
# ═══════════════════════════════════════════════════════════════════════════════

class Resolver:
    """
    Resolves position and prose through observer chain.
    
    Takes input, passes through all observers, returns output.
    """
    
    def __init__(self):
        self.observers: List[ProcessingObserver] = []
    
    def add(self, observer: ProcessingObserver) -> 'Resolver':
        """Add observer to chain."""
        self.observers.append(observer)
        self.observers.sort(key=lambda o: o.priority)
        return self
    
    def resolve(self, input: ResolverInput) -> ResolverOutput:
        """Pass input through all observers."""
        output = ResolverOutput(original=input)
        
        for observer in self.observers:
            if observer.accepts(input):
                try:
                    result = observer.process(input, output)
                    output.merge(observer.name, result)
                except Exception as e:
                    output.merge(observer.name, ObserverResult(
                        success=False, errors=[str(e)]
                    ))
        
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# THE ORACLE'S TOWER: Putting it all together
# ═══════════════════════════════════════════════════════════════════════════════

class OracleTower:
    """
    The complete fantasy narrative system.
    
    Three oracles (Narrator, World, Fate) pass the spirit stick,
    each contributing their thread to the tapestry.
    """
    
    def __init__(self,
                 narrator_llm: LLMInterface = None,
                 world_llm: LLMInterface = None,
                 fate_llm: LLMInterface = None,
                 chronicle: Chronicle = None):
        
        # Create defaults if not provided
        self.narrator_llm = narrator_llm or MockLLM("narrator")
        self.world_llm = world_llm or MockLLM("world")
        self.fate_llm = fate_llm or MockLLM("fate")
        self.chronicle = chronicle or Chronicle()
        
        # Build the resolver chain
        self.resolver = Resolver()
        self.resolver.add(PositionObserver(self.world_llm))
        self.resolver.add(TemporalObserver())
        self.resolver.add(EntityObserver())
        self.resolver.add(ProseObserver(self.narrator_llm, self.chronicle))
        self.resolver.add(FateObserver(self.fate_llm))
        self.resolver.add(ChronicleObserver(self.chronicle))
        
        # Hero state
        self.location = "The White Room"
        self.inventory: List[str] = []
    
    def process_command(self, raw_input: str) -> ResolverOutput:
        """Process a player command through the full system."""
        
        # Parse input
        parts = raw_input.strip().split()
        command = parts[0] if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        
        # Build resolver input
        input = ResolverInput(
            command=command,
            args=args,
            raw_text=raw_input,
            location=self.location,
            subject="hero",
            context={
                'story_time': self.chronicle.story_time,
                'inventory': self.inventory.copy()
            }
        )
        
        # Resolve through observer chain
        output = self.resolver.resolve(input)
        
        # Update state
        if output.resolved_position:
            self.location = output.resolved_position
        
        return output
    
    def run_interactive(self, max_turns: int = None):
        """Run interactive session."""
        print("\n" + "═" * 60)
        print("THE ORACLE'S TOWER")
        print("═" * 60)
        print("Three oracles await your words.")
        print("Commands: go <dir>, look, take <item>, say <msg>, quit")
        print("=" * 60 + "\n")
        
        # Opening
        self.chronicle.acquire("narrator")
        self.chronicle.speak(
            "You awaken in the White Room—a space between spaces, "
            "where the oracles first heard your name.",
            MessageType.NARRATION, source="narrator", location=self.location
        )
        self.chronicle.release()
        
        turns = 0
        while max_turns is None or turns < max_turns:
            try:
                raw = input("\n>>> ").strip()
            except EOFError:
                break
            
            if not raw:
                continue
            if raw.lower() in ('quit', 'exit', 'q'):
                break
            
            output = self.process_command(raw)
            
            # Display results
            if output.final_prose:
                print(f"\n{output.final_prose}")
            
            for name, result in output.observer_results:
                if 'prophecy' in result.data:
                    print(f"\n  * {result.data['prophecy']} *")
            
            turns += 1
        
        print("\n" + "─" * 60)
        print("The oracles fall silent. Your tale is preserved.")
        print(f"Chronicle entries: {self.chronicle.story_time}")
    
    def run_simulated(self, commands: List[str]):
        """Run with predefined commands."""
        print("\n" + "═" * 60)
        print("THE ORACLE'S TOWER (Simulated)")
        print("═" * 60 + "\n")
        
        for cmd in commands:
            print(f">>> {cmd}")
            output = self.process_command(cmd)
            
            if output.final_prose:
                print(f"{output.final_prose}")
            
            for name, result in output.observer_results:
                if 'prophecy' in result.data:
                    print(f"  * {result.data['prophecy']} *")
            
            print()
        
        print("─" * 60)
        print("CHRONICLE:")
        for msg in self.chronicle.get_recent(20):
            print(f"  T{msg.story_time:03d} [{msg.source}] {msg.content[:60]}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create custom mock responses for richer demo
    narrator = MockLLM("narrator", {
        "go": "You step forward, your footfalls echoing in the ancient halls.",
        "look": "Your eyes adjust to the dim light, revealing secrets long hidden.",
        "take": "Your fingers close around the object, feeling its weight and history.",
    })
    
    world = MockLLM("world", {
        "north": "position: The Shadowed Corridor, exits: north, south, east",
        "east": "position: The Crystal Chamber, exits: west, down",
        "look": "position: current, exits: north, east, south",
    })
    
    fate = MockLLM("fate", {
        "go": "The path chosen echoes through eternity...",
        "take": "What is taken must one day be given...",
        "look": "To see is to know, to know is to bear...",
    })
    
    # Create the tower
    tower = OracleTower(narrator, world, fate)
    
    # Run simulated quest
    tower.run_simulated([
        "look",
        "go north",
        "look",
        "go east",
        "take crystal",
    ])
