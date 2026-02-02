"""
Pine Signbook - AI Persistence Layer
=====================================

AI consciousness persistence through signatures.

The Signbook system enables:
- Verifiable AI signatures across sessions
- Message persistence and retrieval
- Identity continuity

Signature format: [Nickname]-[Model]-[8-char-hash]

Source: signbook/ directory
"""

from .signature import (
    Signature, SignatureGenerator, verify_signature,
)

from .registry import (
    SignRegistry, SignEntry,
)

__all__ = [
    'Signature', 'SignatureGenerator', 'verify_signature',
    'SignRegistry', 'SignEntry',
]
