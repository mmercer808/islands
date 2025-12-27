#!/usr/bin/env python3
"""
Vetrell Narrative Module

Story execution, chain iterators, and context management for interactive storytelling.

Classes:
    OperationType - Types of operations in context
    ContextOperation - Single operation with linkage
    ContextWindow - Sliding window of recent operations
    PersistentContextIterator - Long-lived iterator with context
    ScheduledTask - A task scheduled for execution
    TaskScheduler - Manages scheduled tasks and thread execution
    GameLoopIntegrator - Integrates iterator and scheduler with game loop
"""

from .context_iterator import (
    OperationType,
    ContextOperation,
    ContextWindow,
    PersistentContextIterator,
    ScheduledTask,
    TaskScheduler,
    GameLoopIntegrator,
)

__all__ = [
    "OperationType",
    "ContextOperation",
    "ContextWindow",
    "PersistentContextIterator",
    "ScheduledTask",
    "TaskScheduler",
    "GameLoopIntegrator",
]
