"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                              P I N E                                          ║
║                                                                               ║
║           Persistent Interactive Narrative Engine                             ║
║                                                                               ║
║  A framework for AI consciousness persistence and adaptive storytelling.     ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  LAYERS:                                                                      ║
║                                                                               ║
║    pine.core      - Signals, contexts, serialization                         ║
║    pine.runtime   - Live code, hot-swapping, generators                       ║
║    pine.narrative - World building, traversal, story patterns                 ║
║    pine.graph     - Nodes, edges, embeddings                                  ║
║    pine.messaging - Connectors, buffered I/O                                  ║
║    pine.signbook  - AI persistence and signatures                             ║
║    pine.ui        - PySide6 and web interfaces                                ║
║    pine.config    - Settings management                                       ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  QUICK START:                                                                 ║
║                                                                               ║
║    from pine import text_to_world, lookahead_from                             ║
║                                                                               ║
║    world = text_to_world(\"The lighthouse stands on the cliff.\")              ║
║    possibilities = lookahead_from(world, world.origin)                        ║
║    print(possibilities.get_hints())                                           ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

__version__ = "0.1.0"
__author__ = "CloudyCadet"
__description__ = "Persistent Interactive Narrative Engine"

# =============================================================================
#                              CORE IMPORTS
# =============================================================================

from pine.core import (
    # Primitives
    Priority, SignalType, ContextState,
    FragmentCategory, PossibilityType,
    Identified, Context,

    # Signals
    SignalLine, SignalPayload, Signal,
    Observer, CallbackObserver, ObserverBus,
    create_signal_line, signal_handler,

    # Context
    SerializableExecutionContext, ContextChainNode,
    ContextObserver, SerializableContextLibrary,
    ContextSnapshot,

    # Serialization
    OptimizedSerializer,
    SerializedContextMetadata,
)

# =============================================================================
#                              RUNTIME IMPORTS
# =============================================================================

from pine.runtime import (
    # Live code
    CompleteLiveCodeSystem, LiveCodeCallbackSystem,
    SerializedSourceCode, CodeSerializationMethod,

    # Hot-swap
    HotSwapManager, HotSwapHandler,

    # Generators
    AdvancedGeneratorComposer, GeneratorCompositionPattern,
)

# =============================================================================
#                              NARRATIVE IMPORTS
# =============================================================================

from pine.narrative import (
    # World
    WorldNode, WorldEdge, WhiteRoom, WhiteRoomBuilder,
    build_world, text_to_world,

    # Traversal
    TraversalContext, TraversalWrapper, smart_iter,
    LayerRegistry, standard_layers,

    # Lookahead
    Possibility, LookaheadResult, LookaheadEngine,
    lookahead_from,

    # Extraction
    ExtractedEntity, ExtractedRelation, ExtractionResult,
    TextExtractor, extract_text,

    # Spirit Stick
    SpiritStick, Participant, SpiritCircle,
    NarrativeParticipant, StoryCircle,

    # Deferred Builder
    DeferredStoryBuilder, StoryCondition,
)

# =============================================================================
#                              GRAPH IMPORTS
# =============================================================================

from pine.graph import (
    GraphNode, NodeRegistry,
    GraphEdge, EdgeRegistry, RelationshipType,
    GraphWalker, WalkerContext, WalkResult,
    Embedding, EmbeddingStore, create_embedding_store,
)

# =============================================================================
#                              MESSAGING IMPORTS
# =============================================================================

from pine.messaging import (
    Connector, RouteTarget,
    MessageBuffer, BufferedIO, SignalFactory,
    IntegrationLayer, ComponentBridge,
)

# =============================================================================
#                              SIGNBOOK IMPORTS
# =============================================================================

from pine.signbook import (
    Signature, SignatureGenerator,
    SignRegistry, SignEntry,
)

# =============================================================================
#                              CONFIG IMPORTS
# =============================================================================

from pine.config import (
    Settings, SettingsManager,
    get_settings, save_settings,
)

# =============================================================================
#                              PUBLIC API
# =============================================================================

__all__ = [
    # Version
    '__version__',

    # ─────────────────────────────────────────────────────────────────────────
    # Core
    # ─────────────────────────────────────────────────────────────────────────
    'Priority', 'SignalType', 'ContextState',
    'FragmentCategory', 'PossibilityType',
    'Identified', 'Context',

    'SignalLine', 'SignalPayload', 'Signal',
    'Observer', 'CallbackObserver', 'ObserverBus',
    'create_signal_line', 'signal_handler',

    'SerializableExecutionContext', 'ContextChainNode',
    'ContextObserver', 'SerializableContextLibrary',
    'ContextSnapshot',

    'OptimizedSerializer', 'SerializedContextMetadata',

    # ─────────────────────────────────────────────────────────────────────────
    # Runtime
    # ─────────────────────────────────────────────────────────────────────────
    'CompleteLiveCodeSystem', 'LiveCodeCallbackSystem',
    'SerializedSourceCode', 'CodeSerializationMethod',
    'HotSwapManager', 'HotSwapHandler',
    'AdvancedGeneratorComposer', 'GeneratorCompositionPattern',

    # ─────────────────────────────────────────────────────────────────────────
    # Narrative (MAIN ENTRY POINTS)
    # ─────────────────────────────────────────────────────────────────────────
    'WorldNode', 'WorldEdge', 'WhiteRoom', 'WhiteRoomBuilder',
    'build_world', 'text_to_world',

    'TraversalContext', 'TraversalWrapper', 'smart_iter',
    'LayerRegistry', 'standard_layers',

    'Possibility', 'LookaheadResult', 'LookaheadEngine',
    'lookahead_from',

    'ExtractedEntity', 'ExtractedRelation', 'ExtractionResult',
    'TextExtractor', 'extract_text',

    'SpiritStick', 'Participant', 'SpiritCircle',
    'NarrativeParticipant', 'StoryCircle',

    'DeferredStoryBuilder', 'StoryCondition',

    # ─────────────────────────────────────────────────────────────────────────
    # Graph
    # ─────────────────────────────────────────────────────────────────────────
    'GraphNode', 'NodeRegistry',
    'GraphEdge', 'EdgeRegistry', 'RelationshipType',
    'GraphWalker', 'WalkerContext', 'WalkResult',
    'Embedding', 'EmbeddingStore', 'create_embedding_store',

    # ─────────────────────────────────────────────────────────────────────────
    # Messaging
    # ─────────────────────────────────────────────────────────────────────────
    'Connector', 'RouteTarget',
    'MessageBuffer', 'BufferedIO', 'SignalFactory',
    'IntegrationLayer', 'ComponentBridge',

    # ─────────────────────────────────────────────────────────────────────────
    # Signbook
    # ─────────────────────────────────────────────────────────────────────────
    'Signature', 'SignatureGenerator',
    'SignRegistry', 'SignEntry',

    # ─────────────────────────────────────────────────────────────────────────
    # Config
    # ─────────────────────────────────────────────────────────────────────────
    'Settings', 'SettingsManager',
    'get_settings', 'save_settings',
]


# =============================================================================
#                              CONVENIENCE FUNCTIONS
# =============================================================================

def create_world(name: str = "World") -> WhiteRoomBuilder:
    """Start building a world with the fluent API."""
    return WhiteRoomBuilder(name)


def quick_start():
    """
    Quick start demonstration.

    Returns a simple world with possibilities.
    """
    world = text_to_world("""
        The old lighthouse stands on the rocky cliff.
        Inside, a spiral staircase leads upward.
        At the top, the great lamp awaits.
        A brass key lies hidden under the doormat.
    """, name="Lighthouse")

    if world.origin:
        possibilities = lookahead_from(world, world.origin)
        return world, possibilities

    return world, None


def pine_help():
    """Print help information."""
    print(__doc__)
    print()
    print("Main Entry Points:")
    print("  text_to_world(text)    - Create world from text")
    print("  build_world(name)      - Start fluent world builder")
    print("  lookahead_from(...)    - Analyze possibilities")
    print("  DeferredStoryBuilder() - Build conditional story chains")
    print()
    print("Modules:")
    print("  pine.core      - Signals, contexts")
    print("  pine.runtime   - Live code, hot-swap")
    print("  pine.narrative - World building, story patterns")
    print("  pine.graph     - Entity graphs, embeddings")
    print("  pine.messaging - Component communication")
    print("  pine.signbook  - AI persistence")
    print("  pine.ui        - User interfaces")
    print("  pine.config    - Settings")


if __name__ == "__main__":
    pine_help()
