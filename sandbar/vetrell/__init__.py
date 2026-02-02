#!/usr/bin/env python3
"""
Vetrell - Reorganized Islands Library

A collection of experimental systems for interactive storytelling, AI consciousness
persistence, and runtime code manipulation.

Modules:
    serialization - Context serialization and signal transmission
    runtime - Live code injection, hot-swapping, and bytecode manipulation
    language - English language as code object with rule chains
    narrative - Story execution, chain iterators, and context management
    analysis - Code object metadata extraction and process analysis
"""

from . import serialization
from . import runtime
from . import language
from . import narrative
from . import analysis

__version__ = "0.1.0"
__all__ = [
    "serialization",
    "runtime",
    "language",
    "narrative",
    "analysis",
]
