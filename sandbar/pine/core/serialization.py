"""
Pine Context Serialization
==========================

High-performance serialization for contexts with:
- Compression (zlib)
- Dependency bundling
- Portability metadata
- Base64 encoding for signal transmission

SOURCE: matts/context_serialization.py (23KB)
Copy the full implementation from that file.

Serialization Flow:
------------------
1. Context captures state + dependencies
2. Serialize with dill/marshal
3. Compress with zlib
4. Base64 encode for signal metadata
5. Transmit via SignalPayload
6. Deserialize with context-aware reconstruction

Quick Reference:
----------------
    from . import OptimizedSerializer

    serializer = OptimizedSerializer()

    # Serialize
    blob = await serializer.serialize(context)

    # Transmit via signal
    payload = SignalPayload(
        signal_type=SignalType.CONTEXT_UPDATED,
        source="sender",
        metadata={'context_blob': blob}
    )

    # Deserialize
    restored = await serializer.deserialize(blob)
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import base64
import zlib
import json
import uuid
import asyncio

from .primitives import ContextState
from .context import SerializableExecutionContext
from .signals import SignalLine, SignalPayload, SignalType, Observer


# =============================================================================
#                              METADATA
# =============================================================================

@dataclass
class SerializedContextMetadata:
    """
    Metadata for a serialized context blob.

    Includes:
    - Origin information
    - Compression details
    - Dependency manifest
    - Integrity checksums
    """
    context_id: str
    source_system: str = "pine"
    serialized_at: datetime = field(default_factory=datetime.now)
    compression: str = "zlib"
    encoding: str = "base64"
    dependencies: List[str] = field(default_factory=list)
    checksum: str = ""
    size_bytes: int = 0
    size_compressed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'context_id': self.context_id,
            'source_system': self.source_system,
            'serialized_at': self.serialized_at.isoformat(),
            'compression': self.compression,
            'encoding': self.encoding,
            'dependencies': self.dependencies,
            'checksum': self.checksum,
            'size_bytes': self.size_bytes,
            'size_compressed': self.size_compressed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SerializedContextMetadata':
        meta = cls(context_id=data['context_id'])
        meta.source_system = data.get('source_system', 'pine')
        meta.serialized_at = datetime.fromisoformat(data['serialized_at'])
        meta.compression = data.get('compression', 'zlib')
        meta.encoding = data.get('encoding', 'base64')
        meta.dependencies = data.get('dependencies', [])
        meta.checksum = data.get('checksum', '')
        meta.size_bytes = data.get('size_bytes', 0)
        meta.size_compressed = data.get('size_compressed', 0)
        return meta


# =============================================================================
#                              SERIALIZERS
# =============================================================================

class OptimizedSerializer:
    """
    High-performance context serializer.

    Uses:
    - JSON for simple data
    - dill for complex objects (closures, lambdas)
    - zlib compression
    - base64 encoding for transport
    """

    def __init__(self):
        self._cache: Dict[str, bytes] = {}

    async def serialize(
        self,
        context: SerializableExecutionContext,
        include_callbacks: bool = False
    ) -> Tuple[bytes, SerializedContextMetadata]:
        """Serialize context to compressed bytes with metadata."""
        # Get base serialization
        data = context.serialize()

        # JSON encode
        json_bytes = json.dumps(data).encode('utf-8')
        size_bytes = len(json_bytes)

        # Compress
        compressed = zlib.compress(json_bytes, level=6)
        size_compressed = len(compressed)

        # Base64 encode
        encoded = base64.b64encode(compressed)

        # Create metadata
        meta = SerializedContextMetadata(
            context_id=context.context_id,
            dependencies=list(context._dependencies),
            size_bytes=size_bytes,
            size_compressed=size_compressed,
        )

        return encoded, meta

    async def deserialize(
        self,
        blob: bytes,
        metadata: SerializedContextMetadata = None
    ) -> SerializableExecutionContext:
        """Deserialize context from compressed bytes."""
        # Base64 decode
        compressed = base64.b64decode(blob)

        # Decompress
        json_bytes = zlib.decompress(compressed)

        # JSON decode
        data = json.loads(json_bytes.decode('utf-8'))

        # Reconstruct context
        return SerializableExecutionContext.deserialize(data)

    def clear_cache(self) -> None:
        """Clear serialization cache."""
        self._cache.clear()


class FastDependencyBundler:
    """
    Bundles context dependencies for complete serialization.

    When a context references external modules or data,
    this bundler includes them in the serialization.

    TODO: Copy implementation from matts/context_serialization.py
    """

    def __init__(self):
        self._dependency_registry: Dict[str, Any] = {}

    def register(self, name: str, obj: Any) -> None:
        """Register a dependency by name."""
        self._dependency_registry[name] = obj

    def bundle(
        self,
        context: SerializableExecutionContext
    ) -> Dict[str, Any]:
        """Bundle all dependencies for a context."""
        bundle = {}
        for dep in context._dependencies:
            if dep in self._dependency_registry:
                bundle[dep] = self._dependency_registry[dep]
        return bundle


# =============================================================================
#                              SIGNAL BUS
# =============================================================================

class HighPerformanceSignalBus:
    """
    Signal bus optimized for high-throughput context transmission.

    Features:
    - Batching for efficiency
    - Async processing
    - Back-pressure handling

    TODO: Copy implementation from matts/context_serialization.py
    """

    def __init__(self, signal_line: SignalLine):
        self.signal_line = signal_line
        self.serializer = OptimizedSerializer()
        self._queue: asyncio.Queue = asyncio.Queue()

    async def transmit(
        self,
        context: SerializableExecutionContext,
        target: str = "broadcast"
    ) -> bool:
        """Transmit context via signal."""
        blob, meta = await self.serializer.serialize(context)

        payload = SignalPayload(
            signal_type=SignalType.CONTEXT_UPDATED,
            source=context.context_id,
            data={'target': target},
            metadata={
                'context_blob': blob.decode('ascii'),
                'context_meta': meta.to_dict()
            }
        )

        await self.signal_line.emit(payload)
        return True

    async def receive(
        self,
        payload: SignalPayload
    ) -> Optional[SerializableExecutionContext]:
        """Receive and deserialize context from signal."""
        if 'context_blob' not in payload.metadata:
            return None

        blob = payload.metadata['context_blob'].encode('ascii')
        meta_dict = payload.metadata.get('context_meta', {})
        meta = SerializedContextMetadata.from_dict(meta_dict) if meta_dict else None

        return await self.serializer.deserialize(blob, meta)


# =============================================================================
#                              PORTABLE CONTEXT
# =============================================================================

class SerializableExecutionContextWithPortability(SerializableExecutionContext):
    """
    Extended context with full portability features.

    Adds:
    - Automatic dependency tracking
    - Cross-system metadata
    - Version compatibility

    TODO: Copy implementation from matts/context_serialization.py
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._portability_meta: Dict[str, Any] = {
            'version': '1.0',
            'system': 'pine',
        }

    def mark_portable(self, target_systems: List[str]) -> None:
        """Mark context for transmission to specific systems."""
        self._portability_meta['targets'] = target_systems

    def get_portability_info(self) -> Dict[str, Any]:
        """Get portability metadata."""
        return self._portability_meta.copy()
