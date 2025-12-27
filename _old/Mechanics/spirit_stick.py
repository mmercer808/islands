"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                    T H E   S P I R I T   S T I C K                            ║
║                                                                               ║
║  "Whoever holds the spirit stick... and ONLY whoever holds the spirit        ║
║   stick... may speak."                                                        ║
║                           — Bring It On (2000)                                ║
║                                                                               ║
║  A thought experiment on iterator-to-event conversion through token passing. ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

THE PATTERN:
    In the movie, the spirit stick is a physical token that grants exclusive
    permission to speak. Only the holder can address the group. Everyone else
    must listen. When you're done, you pass it to the next person.
    
    This maps beautifully to several computer science concepts:
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  CONCEPT              │  SPIRIT STICK ANALOG                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Mutex                │  Only holder can access shared resource         │
    │  Token Ring           │  Token passes around the circle                 │
    │  Coroutine            │  Yield control to next participant              │
    │  Actor (single)       │  One active actor at a time                     │
    │  Iterator → Event     │  Iteration order determines emission order      │
    │  Turn-based           │  Each participant gets exactly one turn         │
    └─────────────────────────────────────────────────────────────────────────┘

THE INSIGHT:
    An iterator produces values. But what if those values aren't just data—
    what if they're PERMISSION? The iterator becomes an event scheduler.
    
    Normal iteration:
        for item in items:
            process(item)    # We act ON the item
    
    Spirit stick iteration:
        for holder in circle:
            holder.speak()   # The item acts (because it has permission)

ITERATOR → EVENT CONVERSION:
    The key insight: iteration order IS event order.
    
    When you wrap participants in an iterator, you're implicitly creating
    an event schedule. The `__next__` call doesn't just return a value—
    it TRANSFERS PERMISSION. The previous holder loses the right to speak.
    The new holder gains it.
    
    This is iterator-to-event conversion through token semantics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    List, Optional, Any, Callable, Iterator, Generic, TypeVar,
    Dict, Tuple
)
from abc import ABC, abstractmethod
from enum import Enum, auto
import time
import uuid


# ═══════════════════════════════════════════════════════════════════════════════
# THE TOKEN: What grants permission
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpiritStick:
    """
    The token that grants permission to speak.
    
    Physical properties:
        - Exactly one exists per circle
        - Can only be held by one participant at a time
        - Must be explicitly passed (or taken by the iterator)
        - Carries history of who has held it
    
    Metaphysical properties:
        - Grants PERMISSION, not capability
        - The holder COULD stay silent (skip their turn)
        - Dropping the stick is catastrophic (breaks the circle)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Who currently holds the stick (None = in transition)
    holder_id: Optional[str] = None
    
    # History of possession: [(holder_id, acquired_at, released_at), ...]
    history: List[Tuple[str, float, Optional[float]]] = field(default_factory=list)
    
    # Is the stick "live" (in an active circle)?
    active: bool = False
    
    # Has the stick been dropped? (error state)
    dropped: bool = False
    
    def acquire(self, new_holder_id: str) -> bool:
        """
        Transfer the stick to a new holder.
        
        Returns True if successful, False if stick was dropped or
        new_holder already has it.
        """
        if self.dropped:
            return False
        
        if self.holder_id == new_holder_id:
            return True  # Already holding it
        
        # Release from previous holder
        if self.holder_id and self.history:
            # Update the release time of the last entry
            last = self.history[-1]
            self.history[-1] = (last[0], last[1], time.time())
        
        # Transfer to new holder
        self.holder_id = new_holder_id
        self.history.append((new_holder_id, time.time(), None))
        return True
    
    def release(self) -> bool:
        """
        Release the stick (put it down, await next holder).
        """
        if self.dropped or self.holder_id is None:
            return False
        
        if self.history:
            last = self.history[-1]
            self.history[-1] = (last[0], last[1], time.time())
        
        self.holder_id = None
        return True
    
    def drop(self) -> None:
        """
        Drop the stick (error state).
        
        In the movie, dropping the spirit stick brings bad luck.
        In our system, it breaks the iteration contract.
        """
        self.dropped = True
        self.active = False
        self.holder_id = None
    
    def is_held_by(self, participant_id: str) -> bool:
        """Check if a specific participant holds the stick."""
        return self.holder_id == participant_id and not self.dropped
    
    def total_holders(self) -> int:
        """How many unique participants have held the stick."""
        return len(set(h[0] for h in self.history))
    
    def __repr__(self):
        status = "DROPPED" if self.dropped else ("ACTIVE" if self.active else "INACTIVE")
        holder = self.holder_id or "nobody"
        return f"<SpiritStick {self.id} [{status}] held by: {holder}>"


# ═══════════════════════════════════════════════════════════════════════════════
# THE PARTICIPANT: Who can hold the stick
# ═══════════════════════════════════════════════════════════════════════════════

class Participant(ABC):
    """
    Abstract participant in a spirit stick circle.
    
    A participant:
        - Has an identity
        - Can hold the stick (or not)
        - Can speak ONLY when holding the stick
        - Can choose to pass (stay silent) even when holding
    """
    
    def __init__(self, name: str):
        self.id: str = str(uuid.uuid4())[:8]
        self.name: str = name
        self._stick_ref: Optional[SpiritStick] = None
    
    @property
    def has_stick(self) -> bool:
        """Do I currently hold the spirit stick?"""
        return self._stick_ref is not None and self._stick_ref.is_held_by(self.id)
    
    def receive_stick(self, stick: SpiritStick) -> bool:
        """
        Receive the spirit stick.
        
        Called by the circle when it's our turn.
        """
        if stick.acquire(self.id):
            self._stick_ref = stick
            self._on_receive()
            return True
        return False
    
    def release_stick(self) -> Optional[SpiritStick]:
        """
        Release the spirit stick (pass to next).
        
        Returns the stick so it can be passed on.
        """
        if self._stick_ref and self.has_stick:
            stick = self._stick_ref
            self._on_release()
            stick.release()
            self._stick_ref = None
            return stick
        return None
    
    def _on_receive(self) -> None:
        """Hook: called when we receive the stick."""
        pass
    
    def _on_release(self) -> None:
        """Hook: called when we release the stick."""
        pass
    
    @abstractmethod
    def speak(self) -> Optional[Any]:
        """
        Speak (emit an event/value).
        
        ONLY valid when holding the stick.
        Returns whatever the participant wants to say,
        or None to pass silently.
        
        This is the EVENT EMISSION point.
        """
        ...
    
    def __repr__(self):
        stick_status = "🎤" if self.has_stick else "  "
        return f"{stick_status} {self.name} ({self.id})"


class SimpleParticipant(Participant):
    """
    A participant with a prepared message.
    
    When they get the stick, they say their piece.
    """
    
    def __init__(self, name: str, message: Any = None):
        super().__init__(name)
        self.message = message
    
    def speak(self) -> Optional[Any]:
        """Say our prepared message (if we have the stick)."""
        if not self.has_stick:
            return None  # Can't speak without the stick!
        return self.message


class CallbackParticipant(Participant):
    """
    A participant whose speech is determined by a callback.
    
    This is where iterator → event conversion becomes clear:
    the callback IS the event handler, triggered by receiving the stick.
    """
    
    def __init__(self, name: str, on_speak: Callable[['CallbackParticipant'], Any]):
        super().__init__(name)
        self._on_speak = on_speak
    
    def speak(self) -> Optional[Any]:
        """Invoke our callback to determine what to say."""
        if not self.has_stick:
            return None
        return self._on_speak(self)


# ═══════════════════════════════════════════════════════════════════════════════
# THE CIRCLE: Where iteration becomes event emission
# ═══════════════════════════════════════════════════════════════════════════════

class SpiritCircle:
    """
    The circle of participants.
    
    THE KEY PATTERN:
        This class IS an iterator. When you iterate over it, you're not
        just getting participants—you're GRANTING PERMISSION sequentially.
        
        Each __next__ call:
            1. Takes the stick from the previous holder
            2. Gives it to the next participant
            3. That participant speaks (emits an event)
            4. Returns the speech (the event payload)
        
        This is ITERATOR → EVENT CONVERSION:
            - Iteration order determines event order
            - __next__ is the event scheduler
            - The stick is the permission token
            - speak() is the event emission
    """
    
    def __init__(self, participants: List[Participant] = None):
        self.participants: List[Participant] = participants or []
        self.stick: SpiritStick = SpiritStick()
        
        # Circle state
        self._index: int = 0
        self._rounds: int = 0
        self._started: bool = False
        
        # Event listeners (observers of the speeches)
        self._listeners: List[Callable[[Participant, Any], None]] = []
    
    def add(self, participant: Participant) -> 'SpiritCircle':
        """Add a participant to the circle."""
        self.participants.append(participant)
        return self
    
    def add_listener(self, listener: Callable[[Participant, Any], None]) -> 'SpiritCircle':
        """
        Add a listener for speeches.
        
        Listeners receive (speaker, speech) when anyone speaks.
        This is the EVENT SUBSCRIPTION point.
        """
        self._listeners.append(listener)
        return self
    
    # ─────────────────────────────────────────────────────────────────────────
    # Iterator Protocol: Where the magic happens
    # ─────────────────────────────────────────────────────────────────────────
    
    def __iter__(self) -> 'SpiritCircle':
        """
        Start a new round of the circle.
        
        Returns self because the circle IS the iterator.
        (Same pattern as TraversalWrapper!)
        """
        self._index = 0
        self._started = True
        self.stick.active = True
        return self
    
    def __next__(self) -> Tuple[Participant, Any]:
        """
        Pass the stick to the next participant and let them speak.
        
        THIS IS THE ITERATOR → EVENT CONVERSION POINT:
            - We control WHO speaks (iteration order)
            - We control WHEN they speak (__next__ call)
            - We EMIT their speech to listeners (event dispatch)
        
        Returns (speaker, speech) tuple.
        Raises StopIteration when the circle is complete.
        """
        if self._index >= len(self.participants):
            # Circle complete
            self.stick.active = False
            self._rounds += 1
            raise StopIteration
        
        # Get next participant
        participant = self.participants[self._index]
        self._index += 1
        
        # PASS THE STICK (transfer permission)
        if not participant.receive_stick(self.stick):
            # Failed to receive—stick was dropped!
            raise RuntimeError(f"Spirit stick dropped! {participant.name} cannot receive it.")
        
        # LET THEM SPEAK (event emission)
        speech = participant.speak()
        
        # NOTIFY LISTENERS (event dispatch)
        for listener in self._listeners:
            try:
                listener(participant, speech)
            except Exception:
                pass  # Listeners shouldn't break the circle
        
        # RECLAIM THE STICK (prepare for next transfer)
        participant.release_stick()
        
        return (participant, speech)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Convenience methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def go_around(self) -> List[Tuple[Participant, Any]]:
        """
        Complete one full round of the circle.
        
        Returns list of (speaker, speech) tuples.
        """
        return list(self)
    
    def continuous(self, rounds: int = None):
        """
        Generator for multiple rounds.
        
        Yields (round_number, participant, speech) tuples.
        If rounds is None, goes forever (until break).
        """
        round_num = 0
        while rounds is None or round_num < rounds:
            for participant, speech in self:
                yield (round_num, participant, speech)
            round_num += 1
    
    def __repr__(self):
        status = "ACTIVE" if self.stick.active else "WAITING"
        return f"<SpiritCircle [{status}] {len(self.participants)} participants, {self._rounds} rounds>"


# ═══════════════════════════════════════════════════════════════════════════════
# THE INSIGHT: Extracting the design pattern
# ═══════════════════════════════════════════════════════════════════════════════

"""
WHAT WE'VE BUILT:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │    ITERATOR                         EVENT SYSTEM                        │
    │    ────────                         ────────────                        │
    │    __iter__()          →            start_session()                     │
    │    __next__()          →            dispatch_next_event()               │
    │    StopIteration       →            session_complete                    │
    │    yield value         →            emit(event)                         │
    │    for x in iter       →            event_loop.run()                    │
    │                                                                         │
    │    The TOKEN (spirit stick) bridges these:                              │
    │    - Grants PERMISSION to emit                                          │
    │    - Ensures SERIALIZATION (one at a time)                              │
    │    - Creates HISTORY (who spoke when)                                   │
    │    - Can be DROPPED (error handling)                                    │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

THE PATTERN NAME: "Token-Passing Iterator" or "Permission Iterator"

WHEN TO USE:
    - When iteration order should determine event order
    - When only one entity should be "active" at a time
    - When you want to convert pull (iterator) to push (events)
    - When you need an audit trail of who did what when
    - Turn-based systems, round-robin schedulers, conversation flows

RELATIONSHIP TO OTHER PATTERNS:

    Token Ring (networking):
        - Identical concept, different domain
        - Token grants permission to transmit
    
    Mutex/Semaphore (concurrency):
        - Similar exclusive access
        - But mutex doesn't have "passing" semantics
    
    Iterator (GoF):
        - We ARE an iterator
        - But with permission semantics added
    
    Observer (GoF):
        - Listeners observe the speeches
        - But emission is CONTROLLED by the iterator
    
    Coroutine:
        - Similar yield-based control transfer
        - But coroutines don't have explicit token passing
    
    State Machine:
        - Current holder IS the current state
        - Passing the stick IS a state transition
"""


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE: Seeing it in action
# ═══════════════════════════════════════════════════════════════════════════════

def example_basic():
    """Basic example: a circle of cheerleaders."""
    print("\n═══ BASIC SPIRIT STICK CIRCLE ═══\n")
    
    circle = SpiritCircle([
        SimpleParticipant("Torrance", "We're the Toros, not the cheats!"),
        SimpleParticipant("Missy", "Let's make our own routine."),
        SimpleParticipant("Cliff", "I believe in you."),
        SimpleParticipant("Whitney", "..."),  # Stays silent
        SimpleParticipant("Courtney", "Whatever, let's do this."),
    ])
    
    # Add a listener to see the events
    circle.add_listener(lambda p, s: print(f"  {p.name}: \"{s}\"" if s else f"  {p.name}: *passes*"))
    
    print("The circle begins...\n")
    
    # Iterate! This is where iterator → event conversion happens.
    for speaker, speech in circle:
        pass  # Listener handles output
    
    print(f"\n{circle.stick.total_holders()} people spoke.")
    print(f"History: {[h[0] for h in circle.stick.history]}")


def example_with_state():
    """Example: participants with callbacks that access external state."""
    print("\n═══ STATEFUL SPIRIT STICK ═══\n")
    
    # Shared state that participants can modify
    vote_count = {"yes": 0, "no": 0}
    
    def make_voter(name: str, votes_yes: bool):
        def on_speak(p: CallbackParticipant) -> str:
            vote = "yes" if votes_yes else "no"
            vote_count[vote] += 1
            return f"I vote {vote}!"
        return CallbackParticipant(name, on_speak)
    
    circle = SpiritCircle([
        make_voter("Alice", True),
        make_voter("Bob", False),
        make_voter("Carol", True),
        make_voter("Dave", True),
        make_voter("Eve", False),
    ])
    
    circle.add_listener(lambda p, s: print(f"  {p.name}: {s}"))
    
    print("Voting begins...\n")
    circle.go_around()
    
    print(f"\nResults: Yes={vote_count['yes']}, No={vote_count['no']}")
    winner = "YES" if vote_count['yes'] > vote_count['no'] else "NO"
    print(f"The motion {'passes' if winner == 'YES' else 'fails'}!")


def example_multiple_rounds():
    """Example: multiple rounds with evolving state."""
    print("\n═══ MULTI-ROUND SPIRIT STICK ═══\n")
    
    # State evolves each round
    round_state = {"energy": 50}
    
    def make_cheerleader(name: str, energy_change: int):
        def on_speak(p: CallbackParticipant) -> str:
            round_state["energy"] += energy_change
            action = "pumps up" if energy_change > 0 else "calms down"
            return f"*{action} the squad* (energy now: {round_state['energy']})"
        return CallbackParticipant(name, on_speak)
    
    circle = SpiritCircle([
        make_cheerleader("Torrance", +10),
        make_cheerleader("Missy", +15),
        make_cheerleader("Cliff", +5),
        make_cheerleader("Courtney", -5),
    ])
    
    circle.add_listener(lambda p, s: print(f"  {p.name}: {s}"))
    
    print("Three rounds of energy building...\n")
    
    for round_num, speaker, speech in circle.continuous(rounds=3):
        if speaker == circle.participants[0]:  # First person in round
            print(f"\n--- Round {round_num + 1} ---")
    
    print(f"\nFinal energy: {round_state['energy']}")


def example_iterator_to_event():
    """
    EXPLICIT DEMONSTRATION: How iteration becomes event emission.
    
    This example shows the mechanics clearly:
        1. Iterator gives us control of ORDER
        2. Token gives us control of PERMISSION  
        3. Listeners receive the EVENTS
        4. The loop IS the event loop
    """
    print("\n═══ ITERATOR → EVENT CONVERSION ═══\n")
    
    # Collected events for analysis
    events: List[Dict] = []
    
    def event_collector(participant: Participant, speech: Any):
        """This listener collects events for later analysis."""
        events.append({
            'speaker': participant.name,
            'speech': speech,
            'timestamp': time.time(),
        })
    
    circle = SpiritCircle([
        SimpleParticipant("Event-1", {"type": "USER_LOGIN", "user": "alice"}),
        SimpleParticipant("Event-2", {"type": "DATA_UPDATE", "key": "score", "value": 100}),
        SimpleParticipant("Event-3", {"type": "USER_LOGOUT", "user": "alice"}),
    ])
    
    circle.add_listener(event_collector)
    
    print("Watch how iteration becomes event dispatch:\n")
    
    for i, (speaker, speech) in enumerate(circle):
        print(f"  __next__() call #{i+1}:")
        print(f"    → Token passed to: {speaker.name}")
        print(f"    → Event emitted: {speech}")
        print()
    
    print("Collected events:")
    for e in events:
        print(f"  {e['speaker']}: {e['speech']}")


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED: Generic Token-Passing Iterator
# ═══════════════════════════════════════════════════════════════════════════════

T = TypeVar('T')  # Type of items
E = TypeVar('E')  # Type of emitted events

class TokenPassingIterator(Generic[T, E]):
    """
    Generalized token-passing iterator pattern.
    
    Converts any iterable into an event-emitting sequence where:
        - Each item gets exclusive access (the token)
        - Each item can emit when it has the token
        - Listeners observe all emissions
        - The iteration order IS the event order
    
    Type parameters:
        T: Type of items in the sequence
        E: Type of events emitted
    """
    
    def __init__(self, 
                 items: List[T],
                 emit_fn: Callable[[T], E],
                 token_name: str = "token"):
        """
        Create a token-passing iterator.
        
        Args:
            items: The items to iterate over
            emit_fn: Function to call when item has token, returns event
            token_name: Name for the token (for debugging)
        """
        self.items = items
        self.emit_fn = emit_fn
        self.token_name = token_name
        
        # Token state
        self._current_holder: Optional[T] = None
        self._index = 0
        
        # Listeners
        self._listeners: List[Callable[[T, E], None]] = []
        
        # History
        self._emissions: List[Tuple[T, E, float]] = []
    
    def on_emit(self, listener: Callable[[T, E], None]) -> 'TokenPassingIterator[T, E]':
        """Subscribe to emissions."""
        self._listeners.append(listener)
        return self
    
    def __iter__(self) -> 'TokenPassingIterator[T, E]':
        self._index = 0
        return self
    
    def __next__(self) -> Tuple[T, E]:
        if self._index >= len(self.items):
            self._current_holder = None
            raise StopIteration
        
        # Transfer token
        item = self.items[self._index]
        self._current_holder = item
        self._index += 1
        
        # Emit
        event = self.emit_fn(item)
        self._emissions.append((item, event, time.time()))
        
        # Notify
        for listener in self._listeners:
            listener(item, event)
        
        return (item, event)
    
    @property
    def current_holder(self) -> Optional[T]:
        """Who currently holds the token."""
        return self._current_holder
    
    @property
    def history(self) -> List[Tuple[T, E, float]]:
        """All emissions with timestamps."""
        return self._emissions


def example_generic_pattern():
    """Example using the generic token-passing iterator."""
    print("\n═══ GENERIC TOKEN-PASSING ITERATOR ═══\n")
    
    # Simple example: numbers that emit their squares
    numbers = [1, 2, 3, 4, 5]
    
    iterator = TokenPassingIterator(
        items=numbers,
        emit_fn=lambda n: n ** 2,
        token_name="calculator_token"
    )
    
    iterator.on_emit(lambda n, sq: print(f"  {n} emits: {sq}"))
    
    print("Numbers take turns emitting their squares:\n")
    list(iterator)  # Consume the iterator
    
    print(f"\nHistory: {[(n, e) for n, e, _ in iterator.history]}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    example_basic()
    example_with_state()
    example_multiple_rounds()
    example_iterator_to_event()
    example_generic_pattern()
    
    print("\n" + "═" * 70)
    print("THE PATTERN SUMMARY:")
    print("═" * 70)
    print("""
    ITERATOR → EVENT CONVERSION via TOKEN PASSING
    
    1. The ITERATOR controls ORDER (who goes when)
    2. The TOKEN controls PERMISSION (who CAN speak)
    3. The LISTENERS receive EVENTS (what was said)
    4. The FOR LOOP is the EVENT LOOP
    
    __next__() = schedule_next_event()
    token.acquire() = grant_permission()
    speak() = emit_event()
    listener() = handle_event()
    
    "Only whoever holds the spirit stick may speak."
    """)
