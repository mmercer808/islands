"""
Pine Narrative - Story Engine Layer
====================================

The narrative engine for interactive storytelling.

Features:
- Text-to-world transformation (WhiteRoom)
- Layer-aware traversal
- Lookahead possibility analysis
- Spirit stick turn-based narrative
- Deferred story builders

This layer depends on: pine.core, pine.graph

Source Files (to integrate):
- Mechanics/everything/world.py -> world.py
- Mechanics/everything/traversal.py -> traversal.py
- Mechanics/everything/lookahead.py -> lookahead.py
- Mechanics/everything/extraction.py -> extraction.py
- Mechanics/spirit_stick.py -> spirit_stick.py
- (new) -> deferred_builder.py
"""

from .world import (
    # Core world classes
    WorldNode, WorldEdge, WhiteRoom, WhiteRoomBuilder,
    # Factory functions
    build_world, text_to_world,
)

from .traversal import (
    # Layer system
    LayerConfig, LayerRegistry, standard_layers,
    # Traversal
    TraversalContext, TraversalWrapper, smart_iter,
)

from .lookahead import (
    # Possibility analysis
    Possibility, LookaheadResult, LookaheadEngine,
    # Factory
    lookahead_from,
)

from .extraction import (
    # Extracted elements
    ExtractedWord, ExtractedEntity, ExtractedRelation,
    ExtractedFragment, ExtractionResult,
    # Extractor
    TextExtractor, extract_text,
)

from .spirit_stick import (
    # Core pattern
    SpiritStick, Participant, SpiritCircle,
    # Iterator
    TokenPassingIterator,
    # Narrative variants
    NarrativeParticipant, StoryCircle,
)

from .deferred_builder import (
    # Builder pattern
    DeferredStoryBuilder, StoryCondition,
    # Actions
    DeferredAction, ImmediateAction, ConditionalAction,
)

__all__ = [
    # World
    'WorldNode', 'WorldEdge', 'WhiteRoom', 'WhiteRoomBuilder',
    'build_world', 'text_to_world',

    # Traversal
    'LayerConfig', 'LayerRegistry', 'standard_layers',
    'TraversalContext', 'TraversalWrapper', 'smart_iter',

    # Lookahead
    'Possibility', 'LookaheadResult', 'LookaheadEngine',
    'lookahead_from',

    # Extraction
    'ExtractedWord', 'ExtractedEntity', 'ExtractedRelation',
    'ExtractedFragment', 'ExtractionResult',
    'TextExtractor', 'extract_text',

    # Spirit Stick
    'SpiritStick', 'Participant', 'SpiritCircle',
    'TokenPassingIterator',
    'NarrativeParticipant', 'StoryCircle',

    # Deferred Builder
    'DeferredStoryBuilder', 'StoryCondition',
    'DeferredAction', 'ImmediateAction', 'ConditionalAction',
]
