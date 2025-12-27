"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                              M A T T S                                        ║
║                                                                               ║
║          Multi-layered Adaptive Text Transformation System                    ║
║                                                                               ║
║  A system for converting text into interactive game worlds.                   ║
║  "We don't author puzzles - we scan text, analyze every word,                 ║
║   and construct a world from what we find."                                   ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ARCHITECTURE (Dependency Layers):                                            ║
║                                                                               ║
║    Layer 0: primitives    - Enums, protocols, base classes                    ║
║    Layer 1: embedding     - Vector storage (EMBEDDING storage type)           ║
║             signals       - Generic Observer[P] pattern                       ║
║    Layer 2: traversal     - Layers, context, smart iterator                   ║
║    Layer 3: extraction    - Text analysis pipeline                            ║
║    Layer 4: world         - WhiteRoom game world construction                 ║
║    Layer 5: lookahead     - Possibility analysis engine                       ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  QUICK START:                                                                 ║
║                                                                               ║
║    from matts_optimized import text_to_world, lookahead_from                  ║
║                                                                               ║
║    world = text_to_world("The key lay hidden in the old desk.")               ║
║    result = lookahead_from(world, world.origin)                               ║
║    print(result.get_hints())                                                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

__version__ = "0.3.0"
__author__ = "CloudyCadet"

# ═══════════════════════════════════════════════════════════════════════════════
#                              IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

# Layer 0: Primitives
from .primitives import (
    # Type vars
    T, P, N, E,
    # Enums
    Priority, StorageType, SignalType, WordClass, 
    FragmentCategory, PossibilityType,
    # Protocols
    HasId, HasName, Serializable, Embeddable,
    # Base
    Identified, Context,
)

# Layer 1: Embedding
from .embedding import (
    Embedding,
    EmbeddingProvider, HashEmbeddingProvider,
    VectorIndex, FlatVectorIndex,
    EmbeddingStore,
    create_embedding_store,
)

# Layer 1: Signals
from .signals import (
    Signal,
    Observer, FunctionObserver, MethodObserver, CollectorObserver,
    StateObserver, EntityObserver, TraversalObserver,
    ObserverBus,
)

# Layer 2: Traversal
from .traversal import (
    LayerConfig, LayerRegistry, standard_layers,
    TraversalContext,
    TraversalWrapper, smart_iter,
)

# Layer 3: Extraction
from .extraction import (
    ExtractedWord, ExtractedEntity, ExtractedRelation,
    ExtractedFragment, ExtractionResult,
    TextExtractor, extract_text,
)

# Layer 4: World
from .world import (
    WorldNode, WorldEdge, WhiteRoom, WhiteRoomBuilder,
    build_world, text_to_world,
)

# Layer 5: Lookahead
from .lookahead import (
    Possibility, LookaheadResult, LookaheadEngine,
    lookahead_from,
)


# ═══════════════════════════════════════════════════════════════════════════════
#                              PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Version
    '__version__',
    
    # ─────────────────────────────────────────────────────────────────────────
    # Primitives
    # ─────────────────────────────────────────────────────────────────────────
    'T', 'P', 'N', 'E',
    'Priority', 'StorageType', 'SignalType', 'WordClass', 
    'FragmentCategory', 'PossibilityType',
    'HasId', 'HasName', 'Serializable', 'Embeddable',
    'Identified', 'Context',
    
    # ─────────────────────────────────────────────────────────────────────────
    # Embedding (STORAGE TYPE)
    # ─────────────────────────────────────────────────────────────────────────
    'Embedding',
    'EmbeddingProvider', 'HashEmbeddingProvider',
    'VectorIndex', 'FlatVectorIndex',
    'EmbeddingStore',
    'create_embedding_store',
    
    # ─────────────────────────────────────────────────────────────────────────
    # Signals (GENERIC OBSERVER)
    # ─────────────────────────────────────────────────────────────────────────
    'Signal',
    'Observer', 'FunctionObserver', 'MethodObserver', 'CollectorObserver',
    'StateObserver', 'EntityObserver', 'TraversalObserver',
    'ObserverBus',
    
    # ─────────────────────────────────────────────────────────────────────────
    # Traversal
    # ─────────────────────────────────────────────────────────────────────────
    'LayerConfig', 'LayerRegistry', 'standard_layers',
    'TraversalContext',
    'TraversalWrapper', 'smart_iter',
    
    # ─────────────────────────────────────────────────────────────────────────
    # Extraction
    # ─────────────────────────────────────────────────────────────────────────
    'ExtractedWord', 'ExtractedEntity', 'ExtractedRelation',
    'ExtractedFragment', 'ExtractionResult',
    'TextExtractor', 'extract_text',
    
    # ─────────────────────────────────────────────────────────────────────────
    # World
    # ─────────────────────────────────────────────────────────────────────────
    'WorldNode', 'WorldEdge', 'WhiteRoom', 'WhiteRoomBuilder',
    'build_world', 'text_to_world',
    
    # ─────────────────────────────────────────────────────────────────────────
    # Lookahead
    # ─────────────────────────────────────────────────────────────────────────
    'Possibility', 'LookaheadResult', 'LookaheadEngine',
    'lookahead_from',
]


# ═══════════════════════════════════════════════════════════════════════════════
#                              QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"MATTS v{__version__}")
    print(f"Exports: {len(__all__)} items")
    print()
    
    # Demo
    text = """
    The old house stood at the end of the lane. Inside, the study contained 
    a massive oak desk covered with papers. A brass key lay hidden under 
    the papers. The key would unlock the door to the garden.
    """
    
    print("═══ Demo ═══")
    print(f"Input: {text.strip()[:60]}...")
    print()
    
    # Create embedding store
    store = create_embedding_store()
    
    # Complete pipeline
    world = text_to_world(text, store)
    
    print(world.summary())
    print()
    
    # Lookahead
    if world.origin:
        result = lookahead_from(world, world.origin)
        print(result.summary())
        print()
        
        hints = result.get_hints()
        if hints:
            print("Hints:")
            for h in hints:
                print(f"  💡 {h}")
