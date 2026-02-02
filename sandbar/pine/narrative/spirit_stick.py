"""
Pine Spirit Stick Pattern
=========================

Iterator-to-event conversion through token semantics.

The Spirit Stick pattern (inspired by "Bring It On"):
- A token grants exclusive permission to speak/act
- Token passes around a circle of participants
- Iteration order becomes event scheduling

Maps to:
- Mutex: Only holder can access
- Token Ring: Token passes around
- Coroutine: Yield control to next
- Turn-based narrative: Each character acts in order

SOURCE: Mechanics/spirit_stick.py (31KB)
Copy the full implementation from that file.

Quick Reference:
----------------
    from . import SpiritCircle, NarrativeParticipant

    # Create participants
    narrator = NarrativeParticipant("Narrator", "The story begins...")
    hero = NarrativeParticipant("Hero", "I accept the quest!")

    # Create circle
    circle = StoryCircle([narrator, hero])

    # Iteration grants permission to speak
    for speaker, speech in circle:
        print(f"{speaker.name}: {speech}")
"""

from typing import Any, Dict, List, Optional, Callable, Iterator, Tuple, TypeVar
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid

T = TypeVar('T')


# =============================================================================
#                              SPIRIT STICK
# =============================================================================

@dataclass
class SpiritStick:
    """
    The token that grants permission to speak/act.

    Only the holder of the spirit stick may take action.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    holder_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def grant_to(self, holder_id: str) -> None:
        """Grant the stick to a holder."""
        self.holder_id = holder_id

    def release(self) -> None:
        """Release the stick (no holder)."""
        self.holder_id = None

    def is_held_by(self, holder_id: str) -> bool:
        """Check if held by specific holder."""
        return self.holder_id == holder_id


# =============================================================================
#                              PARTICIPANT
# =============================================================================

class Participant(ABC):
    """
    Abstract participant who can hold the spirit stick.

    When granted the stick, the participant may speak/act.
    """

    def __init__(self, name: str):
        self.id = str(uuid.uuid4())
        self.name = name

    @abstractmethod
    def speak(self, stick: SpiritStick) -> str:
        """
        Speak while holding the stick.

        Called when this participant is granted the stick.
        Returns the speech/action to emit.
        """
        pass

    def on_granted(self, stick: SpiritStick) -> None:
        """Called when granted the stick."""
        pass

    def on_released(self, stick: SpiritStick) -> None:
        """Called when stick is released."""
        pass


class SimpleParticipant(Participant):
    """Simple participant with static speech."""

    def __init__(self, name: str, speech: str):
        super().__init__(name)
        self._speech = speech

    def speak(self, stick: SpiritStick) -> str:
        return self._speech


# =============================================================================
#                              SPIRIT CIRCLE
# =============================================================================

class SpiritCircle:
    """
    Circle of participants passing the spirit stick.

    Iteration grants permission sequentially around the circle.

    Usage:
        circle = SpiritCircle([alice, bob, carol])
        for speaker, speech in circle:
            # Each participant speaks in turn
            print(f"{speaker.name}: {speech}")

    TODO: Copy full implementation from Mechanics/spirit_stick.py
    """

    def __init__(
        self,
        participants: List[Participant] = None,
        stick: SpiritStick = None
    ):
        self._participants = participants or []
        self._stick = stick or SpiritStick()
        self._current_index = 0
        self._listeners: List[Callable[[Participant, str], None]] = []

    def add_participant(self, participant: Participant) -> None:
        """Add a participant to the circle."""
        self._participants.append(participant)

    def remove_participant(self, participant: Participant) -> None:
        """Remove a participant from the circle."""
        if participant in self._participants:
            self._participants.remove(participant)

    def add_listener(
        self,
        callback: Callable[[Participant, str], None]
    ) -> None:
        """Add listener for speech events."""
        self._listeners.append(callback)

    def __iter__(self) -> Iterator[Tuple[Participant, str]]:
        """Iterate around the circle, granting permission to each."""
        for i, participant in enumerate(self._participants):
            self._current_index = i

            # Grant stick
            self._stick.grant_to(participant.id)
            participant.on_granted(self._stick)

            # Get speech
            speech = participant.speak(self._stick)

            # Notify listeners
            for listener in self._listeners:
                listener(participant, speech)

            # Yield
            yield (participant, speech)

            # Release stick
            participant.on_released(self._stick)
            self._stick.release()

    def run_round(self) -> List[Tuple[Participant, str]]:
        """Run one complete round, collecting all speeches."""
        return list(self)

    @property
    def current_holder(self) -> Optional[Participant]:
        """Get current stick holder."""
        if self._stick.holder_id:
            for p in self._participants:
                if p.id == self._stick.holder_id:
                    return p
        return None


# =============================================================================
#                              TOKEN PASSING ITERATOR
# =============================================================================

class TokenPassingIterator(Iterator[T]):
    """
    Generic token-passing iterator pattern.

    Wraps any iterable and adds token semantics:
    - Each item is "granted" the token
    - Token callbacks are fired
    - Items can refuse the token

    TODO: Copy full implementation from Mechanics/spirit_stick.py
    """

    def __init__(
        self,
        items: List[T],
        token: Any = None
    ):
        self._items = items
        self._token = token or object()
        self._index = 0
        self._on_grant: Optional[Callable[[T], bool]] = None
        self._on_release: Optional[Callable[[T], None]] = None

    def set_grant_callback(self, callback: Callable[[T], bool]) -> None:
        """Set callback for token grant (return False to skip item)."""
        self._on_grant = callback

    def set_release_callback(self, callback: Callable[[T], None]) -> None:
        """Set callback for token release."""
        self._on_release = callback

    def __iter__(self) -> 'TokenPassingIterator[T]':
        return self

    def __next__(self) -> T:
        while self._index < len(self._items):
            item = self._items[self._index]
            self._index += 1

            # Check if item accepts token
            if self._on_grant and not self._on_grant(item):
                continue

            # Yield item (token holder)
            return item

        raise StopIteration


# =============================================================================
#                              NARRATIVE VARIANTS
# =============================================================================

class NarrativeParticipant(Participant):
    """
    Participant specialized for narrative sequences.

    Can have multiple speeches that are delivered in order,
    plus conditions for when they can speak.
    """

    def __init__(
        self,
        name: str,
        speeches: List[str] = None,
        condition: Callable[[], bool] = None
    ):
        super().__init__(name)
        self._speeches = speeches or []
        self._speech_index = 0
        self._condition = condition

    def add_speech(self, speech: str) -> None:
        """Add a speech to the queue."""
        self._speeches.append(speech)

    def can_speak(self) -> bool:
        """Check if this participant can currently speak."""
        if self._condition:
            return self._condition()
        return len(self._speeches) > self._speech_index

    def speak(self, stick: SpiritStick) -> str:
        if self._speech_index < len(self._speeches):
            speech = self._speeches[self._speech_index]
            self._speech_index += 1
            return speech
        return ""

    def reset(self) -> None:
        """Reset speech index to beginning."""
        self._speech_index = 0


class StoryCircle(SpiritCircle):
    """
    Spirit circle specialized for story narration.

    Adds:
    - Narrator role
    - Scene management
    - Conditional speaking
    """

    def __init__(
        self,
        participants: List[NarrativeParticipant] = None,
        narrator: NarrativeParticipant = None
    ):
        super().__init__(participants)
        self._narrator = narrator
        self._scene = ""
        self._round = 0

    def set_scene(self, scene: str) -> None:
        """Set the current scene."""
        self._scene = scene
        if self._narrator:
            self._narrator.add_speech(f"[Scene: {scene}]")

    def __iter__(self) -> Iterator[Tuple[Participant, str]]:
        """Iterate with conditional speaking."""
        self._round += 1

        for participant in self._participants:
            # Skip if can't speak
            if isinstance(participant, NarrativeParticipant):
                if not participant.can_speak():
                    continue

            # Grant and speak
            self._stick.grant_to(participant.id)
            speech = participant.speak(self._stick)

            if speech:  # Only yield if there's speech
                yield (participant, speech)

            self._stick.release()

    @property
    def round(self) -> int:
        """Get current round number."""
        return self._round
