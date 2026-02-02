"""
Pine Graph - Entity/Relationship Layer
=======================================

Graph structures for entities and relationships.

Features:
- Node and edge types
- Graph walking/traversal
- Vector embeddings for semantic search
- Relationship queries

This layer depends on: pine.core

Source Files (to integrate):
- Mechanics/everything/graph_core.py -> nodes.py, edges.py
- Mechanics/everything/graph_walker.py -> walker.py
- Mechanics/everything/embedding.py -> embedding.py
"""

from .nodes import (
    GraphNode, NodeRegistry,
)

from .edges import (
    GraphEdge, EdgeRegistry, RelationshipType,
)

from .walker import (
    GraphWalker, WalkerContext, WalkResult,
)

from .embedding import (
    Embedding, EmbeddingProvider, HashEmbeddingProvider,
    VectorIndex, FlatVectorIndex, EmbeddingStore,
    create_embedding_store,
)

__all__ = [
    # Nodes
    'GraphNode', 'NodeRegistry',

    # Edges
    'GraphEdge', 'EdgeRegistry', 'RelationshipType',

    # Walker
    'GraphWalker', 'WalkerContext', 'WalkResult',

    # Embedding
    'Embedding', 'EmbeddingProvider', 'HashEmbeddingProvider',
    'VectorIndex', 'FlatVectorIndex', 'EmbeddingStore',
    'create_embedding_store',
]
