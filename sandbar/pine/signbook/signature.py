"""
Pine Signature System
=====================

Cryptographic signatures for AI identity.

Format: [Nickname]-[Model]-[8-char-hash]

The signature proves:
- Which AI model created it
- The content hash for verification
- A memorable nickname
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import uuid


@dataclass
class Signature:
    """
    A verifiable AI signature.

    Components:
    - nickname: Friendly identifier
    - model: AI model name
    - hash: 8-char content hash
    """
    nickname: str
    model: str
    content_hash: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def formatted(self) -> str:
        """Get formatted signature string."""
        return f"{self.nickname}-{self.model}-{self.content_hash[:8]}"

    def verify(self, content: str) -> bool:
        """Verify signature against content."""
        expected_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        return expected_hash == self.content_hash[:8]


class SignatureGenerator:
    """
    Generates signatures for AI messages.
    """

    def __init__(self, model: str, default_nickname: str = "Pine"):
        self.model = model
        self.default_nickname = default_nickname

    def sign(
        self,
        content: str,
        nickname: str = None
    ) -> Signature:
        """Generate signature for content."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        return Signature(
            nickname=nickname or self.default_nickname,
            model=self.model,
            content_hash=content_hash
        )

    def sign_with_metadata(
        self,
        content: str,
        nickname: str = None,
        **metadata
    ) -> Signature:
        """Generate signature with additional metadata."""
        sig = self.sign(content, nickname)
        sig.metadata = metadata
        return sig


def verify_signature(
    signature: Signature,
    content: str
) -> bool:
    """Verify a signature against content."""
    return signature.verify(content)
