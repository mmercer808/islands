#!/usr/bin/env python3
"""
Vetrell Serialization Module

Context serialization and signal transmission components for portable execution contexts.

Classes:
    SerializedContextMetadata - Metadata for serialized execution contexts
    Signal - Enhanced signal with context metadata
    ContextAwareSignalObserver - Observer that extracts serialized contexts
    FastDependencyBundler - High-performance dependency detection and bundling
    OptimizedSerializer - Fast serialization with caching and compression
    HighPerformanceSignalBus - Signal bus with context serialization support
    SerializableExecutionContextWithPortability - Portable execution context
"""

from .context import (
    SerializedContextMetadata,
    Signal,
    ContextAwareSignalObserver,
    FastDependencyBundler,
    OptimizedSerializer,
    HighPerformanceSignalBus,
    SerializableExecutionContextWithPortability,
)

__all__ = [
    "SerializedContextMetadata",
    "Signal",
    "ContextAwareSignalObserver",
    "FastDependencyBundler",
    "OptimizedSerializer",
    "HighPerformanceSignalBus",
    "SerializableExecutionContextWithPortability",
]
