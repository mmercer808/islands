"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  MATTS EMBEDDING - Vector Storage System                                       ║
║  Layer 1: Depends on primitives only                                          ║
║                                                                               ║
║  EMBEDDING is a STORAGE TYPE for semantic operations.                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import hashlib
import time

from .primitives import Identified, Embeddable


# ═══════════════════════════════════════════════════════════════════════════════
#                              EMBEDDING DATA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Embedding(Identified):
    """
    A vector embedding for semantic operations.
    
    The atomic unit of EMBEDDING storage type.
    """
    vector: List[float] = field(default_factory=list)
    dimensions: int = 0
    
    # Source tracking
    source_id: str = ""
    source_type: str = ""
    source_text: str = ""
    
    # State
    normalized: bool = False
    
    def __post_init__(self):
        self.dimensions = len(self.vector)
    
    def normalize(self) -> 'Embedding':
        """Return L2-normalized (unit length) copy"""
        if self.normalized or not self.vector:
            return self
        
        mag = sum(x * x for x in self.vector) ** 0.5
        if mag == 0:
            return self
        
        return Embedding(
            id=self.id,
            vector=[x / mag for x in self.vector],
            source_id=self.source_id,
            source_type=self.source_type,
            source_text=self.source_text,
            metadata=self.metadata.copy(),
            normalized=True
        )
    
    def cosine_similarity(self, other: 'Embedding') -> float:
        """Cosine similarity (requires same dimensions)"""
        if len(self.vector) != len(other.vector):
            raise ValueError(f"Dimension mismatch: {len(self.vector)} vs {len(other.vector)}")
        
        a, b = self.normalize().vector, other.normalize().vector
        return sum(x * y for x, y in zip(a, b))
    
    def euclidean_distance(self, other: 'Embedding') -> float:
        """Euclidean distance"""
        if len(self.vector) != len(other.vector):
            raise ValueError("Dimension mismatch")
        return sum((x - y) ** 2 for x, y in zip(self.vector, other.vector)) ** 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id, 'vector': self.vector, 'dimensions': self.dimensions,
            'source_id': self.source_id, 'source_type': self.source_type,
            'source_text': self.source_text, 'metadata': self.metadata,
            'normalized': self.normalized
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Embedding':
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════════════════
#                              PROVIDERS
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingProvider(ABC):
    """
    Abstract embedding provider.
    
    Implement for: local models, APIs, hash-based testing.
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> Embedding:
        """Create embedding from text"""
        ...
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """Batch embedding"""
        ...
    
    @abstractmethod
    def dimensions(self) -> int:
        """Get output dimensions"""
        ...
    
    def embed_object(self, obj: Any) -> Embedding:
        """Embed any object (uses Embeddable protocol or string conversion)"""
        if isinstance(obj, Embeddable):
            text = obj.to_embedding_text()
        elif hasattr(obj, 'name') and hasattr(obj, 'data'):
            text = f"{obj.name} {obj.data.get('description', '')}"
        elif hasattr(obj, 'name'):
            text = str(obj.name)
        else:
            text = str(obj)
        
        emb = self.embed_text(text)
        emb.source_id = getattr(obj, 'id', str(id(obj)))
        emb.source_type = type(obj).__name__
        return emb


class HashEmbeddingProvider(EmbeddingProvider):
    """
    Hash-based provider for testing.
    
    Deterministic, fast, NOT semantically meaningful.
    """
    
    def __init__(self, dims: int = 128):
        self._dims = dims
    
    def embed_text(self, text: str) -> Embedding:
        h = hashlib.sha256(text.encode()).digest()
        vector = [(h[i % len(h)] / 127.5) - 1.0 for i in range(self._dims)]
        return Embedding(vector=vector, source_text=text[:100])
    
    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        return [self.embed_text(t) for t in texts]
    
    def dimensions(self) -> int:
        return self._dims


# ═══════════════════════════════════════════════════════════════════════════════
#                              INDEX
# ═══════════════════════════════════════════════════════════════════════════════

class VectorIndex(ABC):
    """
    Abstract vector index for similarity search.
    
    This is the STORAGE component.
    """
    
    @abstractmethod
    def add(self, emb: Embedding) -> None: ...
    
    @abstractmethod
    def add_batch(self, embs: List[Embedding]) -> None: ...
    
    @abstractmethod
    def search(self, query: Embedding, k: int = 10) -> List[Tuple[str, float]]:
        """Returns [(embedding_id, similarity), ...]"""
        ...
    
    @abstractmethod
    def remove(self, emb_id: str) -> bool: ...
    
    @abstractmethod
    def get(self, emb_id: str) -> Optional[Embedding]: ...
    
    @abstractmethod
    def count(self) -> int: ...
    
    @abstractmethod
    def clear(self) -> None: ...


class FlatVectorIndex(VectorIndex):
    """
    Brute-force O(n) index. Simple and exact.
    """
    
    def __init__(self):
        self._store: Dict[str, Embedding] = {}
    
    def add(self, emb: Embedding) -> None:
        self._store[emb.id] = emb
    
    def add_batch(self, embs: List[Embedding]) -> None:
        for e in embs:
            self.add(e)
    
    def search(self, query: Embedding, k: int = 10) -> List[Tuple[str, float]]:
        scores = []
        for emb in self._store.values():
            try:
                scores.append((emb.id, query.cosine_similarity(emb)))
            except ValueError:
                continue
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def remove(self, emb_id: str) -> bool:
        if emb_id in self._store:
            del self._store[emb_id]
            return True
        return False
    
    def get(self, emb_id: str) -> Optional[Embedding]:
        return self._store.get(emb_id)
    
    def count(self) -> int:
        return len(self._store)
    
    def clear(self) -> None:
        self._store.clear()
    
    def to_dict(self) -> Dict:
        return {'embeddings': [e.to_dict() for e in self._store.values()]}
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'FlatVectorIndex':
        idx = cls()
        for ed in d.get('embeddings', []):
            idx.add(Embedding.from_dict(ed))
        return idx


# ═══════════════════════════════════════════════════════════════════════════════
#                              STORE (HIGH-LEVEL)
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingStore:
    """
    High-level embedding storage manager.
    
    Combines provider + index for complete EMBEDDING storage type.
    """
    
    def __init__(self, 
                 provider: EmbeddingProvider = None,
                 index: VectorIndex = None):
        self.provider = provider or HashEmbeddingProvider()
        self.index = index or FlatVectorIndex()
        self._source_map: Dict[str, str] = {}  # source_id → embedding_id
    
    def embed_and_store(self, obj: Any, source_id: str = None) -> Embedding:
        """Embed object and store it"""
        emb = self.provider.embed_object(obj)
        if source_id:
            emb.source_id = source_id
        
        self.index.add(emb)
        self._source_map[emb.source_id] = emb.id
        return emb
    
    def embed_text(self, text: str, source_id: str = None) -> Embedding:
        """Embed text and store it"""
        emb = self.provider.embed_text(text)
        if source_id:
            emb.source_id = source_id
        
        self.index.add(emb)
        self._source_map[emb.source_id] = emb.id
        return emb
    
    def find_similar(self, query: Union[str, Embedding, Any], 
                     k: int = 10) -> List[Tuple[str, float]]:
        """
        Find k most similar items.
        Returns [(source_id, similarity), ...]
        """
        if isinstance(query, str):
            q = self.provider.embed_text(query)
        elif isinstance(query, Embedding):
            q = query
        else:
            q = self.provider.embed_object(query)
        
        results = self.index.search(q, k)
        
        # Map to source IDs
        out = []
        for emb_id, score in results:
            emb = self.index.get(emb_id)
            if emb:
                out.append((emb.source_id, score))
        return out
    
    def get_for(self, source_id: str) -> Optional[Embedding]:
        """Get embedding for source object"""
        emb_id = self._source_map.get(source_id)
        return self.index.get(emb_id) if emb_id else None
    
    def remove_for(self, source_id: str) -> bool:
        """Remove embedding for source object"""
        emb_id = self._source_map.get(source_id)
        if emb_id:
            self.index.remove(emb_id)
            del self._source_map[source_id]
            return True
        return False
    
    @property
    def count(self) -> int:
        return self.index.count()


# ═══════════════════════════════════════════════════════════════════════════════
#                              FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_embedding_store(dimensions: int = 128) -> EmbeddingStore:
    """Create store with hash provider (for testing)"""
    return EmbeddingStore(
        provider=HashEmbeddingProvider(dimensions),
        index=FlatVectorIndex()
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                              EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'Embedding',
    'EmbeddingProvider', 'HashEmbeddingProvider',
    'VectorIndex', 'FlatVectorIndex',
    'EmbeddingStore',
    'create_embedding_store',
]
