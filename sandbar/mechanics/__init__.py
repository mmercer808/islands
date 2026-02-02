"""
Sandbar mechanics — message→queue→handler patterns.

Assimilated from Mechanics/: hero_quest_chronicle (chronicle + MessagingInterface + HeroActor),
spirit_stick, traversing_spirit_stick, stream_processor; everything/: callback_patterns,
oracle_tower, my_infocom_entity_system, signals. Use for "server responds to any message"
handler/chronicle patterns.
"""

from .hero_quest_chronicle import (
    MessageType,
    Message,
    Chronicle,
    MessagingInterface,
    HeroActor,
)

__all__ = [
    "MessageType",
    "Message",
    "Chronicle",
    "MessagingInterface",
    "HeroActor",
]
