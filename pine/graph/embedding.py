"""
Pine Embedding System
=====================

Vector embeddings for semantic search and similarity.

Features:
- Multiple embedding providers
- Vector indexing
- Similarity search
- Embedding storage

SOURCE: Mechanics/everything/embedding.py (13KB)
Copy the full implementation from that file.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import math


# =============================================================================
#                              EMBEDDING
# =============================================================================

@dataclass
class Embedding:
    """A vector embedding with metadata."""
    vector: List[float]
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return len(self.vector)

    def cosine_similarity(self, other: 'Embedding') -> float:
        """Calculate cosine similarity with another embedding."""
        if self.dimension != other.dimension:
            return 0.0

        dot_product = sum(a * b for a, b in zip(self.vector, other.vector))
        norm_a = math.sqrt(sum(a * a for a in self.vector))
        norm_b = math.sqrt(b * b for b in other.vector)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


# =============================================================================
#                              PROVIDERS
# =============================================================================

class EmbeddingProvider(ABC):
    """Abstract provider for generating embeddings."""

    @abstractmethod
    def embed(self, text: str) -> Embedding:
        """Generate embedding for text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass


class HashEmbeddingProvider(EmbeddingProvider):
    """
    Simple hash-based embedding provider.

    Uses character/word hashing to generate pseudo-embeddings.
    Good for testing, not for production semantic search.
    """

    def __init__(self, dimension: int = 128):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> Embedding:
        """Generate hash-based embedding."""
        vector = [0.0] * self._dimension

        # Hash each word and accumulate
        words = text.lower().split()
        for word in words:
            hash_bytes = hashlib.sha256(word.encode()).digest()
            for i, b in enumerate(hash_bytes[:self._dimension]):
                vector[i % self._dimension] += (b - 128) / 128.0

        # Normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return Embedding(vector=vector, text=text)

    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        return [self.embed(text) for text in texts]


# =============================================================================
#                              VECTOR INDEX
# =============================================================================

class VectorIndex(ABC):
    """Abstract vector index for similarity search."""

    @abstractmethod
    def add(self, id: str, embedding: Embedding) -> None:
        """Add embedding to index."""
        pass

    @abstractmethod
    def search(
        self,
        query: Embedding,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search for k most similar embeddings."""
        pass

    @abstractmethod
    def remove(self, id: str) -> bool:
        """Remove embedding from index."""
        pass


class FlatVectorIndex(VectorIndex):
    """
    Simple flat vector index using linear search.

    Good for small datasets, use FAISS/Annoy for production.
    """

    def __init__(self):
        self._embeddings: Dict[str, Embedding] = {}

    def add(self, id: str, embedding: Embedding) -> None:
        self._embeddings[id] = embedding

    def search(
        self,
        query: Embedding,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Search using linear scan with cosine similarity."""
        results = []

        for id, embedding in self._embeddings.items():
            similarity = query.cosine_similarity(embedding)
            results.append((id, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: -x[1])
        return results[:k]

    def remove(self, id: str) -> bool:
        if id in self._embeddings:
            del self._embeddings[id]
            return True
        return False

    @property
    def count(self) -> int:
        return len(self._embeddings)


# =============================================================================
#                              EMBEDDING STORE
# =============================================================================

class EmbeddingStore:
    """
    Complete embedding storage with provider and index.

    Combines:
    - Embedding generation
    - Vector indexing
    - Similarity search
    """

    def __init__(
        self,
        provider: EmbeddingProvider = None,
        index: VectorIndex = None
    ):
        self.provider = provider or HashEmbeddingProvider()
        self.index = index or FlatVectorIndex()
        self._texts: Dict[str, str] = {}

    def add(self, id: str, text: str) -> Embedding:
        """Embed and store text."""
        embedding = self.provider.embed(text)
        self.index.add(id, embedding)
        self._texts[id] = text
        return embedding

    def search(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        Search for similar texts.

        Returns list of (id, text, similarity) tuples.
        """
        query_embedding = self.provider.embed(query)
        results = self.index.search(query_embedding, k)

        return [
            (id, self._texts.get(id, ""), similarity)
            for id, similarity in results
        ]

    def remove(self, id: str) -> bool:
        """Remove from store."""
        if self.index.remove(id):
            self._texts.pop(id, None)
            return True
        return False

    def get_text(self, id: str) -> Optional[str]:
        """Get original text by ID."""
        return self._texts.get(id)


def create_embedding_store(
    provider: EmbeddingProvider = None,
    index: VectorIndex = None
) -> EmbeddingStore:
    """Create an embedding store with defaults."""
    return EmbeddingStore(provider, index)
