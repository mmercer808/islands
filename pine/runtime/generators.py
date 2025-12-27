"""
Pine Generator System
=====================

Advanced generator composition patterns.

Features:
- Sequential, parallel, pipeline patterns
- Generator state branching and merging
- Recursive composition
- Conditional branching

SOURCE: matts/generator_system.py (31KB)
Copy the full implementation from that file.

Quick Reference:
----------------
    from pine.runtime import AdvancedGeneratorComposer, GeneratorCompositionPattern

    composer = AdvancedGeneratorComposer()

    # Define generators
    def numbers():
        for i in range(10):
            yield i

    def double(gen):
        for x in gen:
            yield x * 2

    # Compose
    pipeline = composer.compose(
        numbers,
        double,
        pattern=GeneratorCompositionPattern.PIPELINE
    )

    for value in pipeline():
        print(value)  # 0, 2, 4, 6, ...
"""

from typing import (
    Any, Dict, List, Optional, Callable, Generator, AsyncGenerator,
    TypeVar, Generic, Iterator, Tuple
)
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import asyncio


T = TypeVar('T')
U = TypeVar('U')


# =============================================================================
#                              ENUMS
# =============================================================================

class GeneratorCompositionPattern(Enum):
    """Patterns for composing generators."""
    SEQUENTIAL = auto()    # One after another
    PARALLEL = auto()      # Concurrent execution
    PIPELINE = auto()      # Output of one feeds next
    BRANCH_MERGE = auto()  # Split and merge
    RECURSIVE = auto()     # Self-referential


# =============================================================================
#                              DATA CLASSES
# =============================================================================

@dataclass
class GeneratorBranch:
    """A branch in a generator composition tree."""
    name: str
    generator_factory: Callable[[], Generator]
    condition: Optional[Callable[[Any], bool]] = None
    children: List['GeneratorBranch'] = field(default_factory=list)

    def should_execute(self, context: Any) -> bool:
        """Check if this branch should execute given context."""
        if self.condition is None:
            return True
        return self.condition(context)


@dataclass
class GeneratorStateBranch:
    """
    State snapshot at a branch point in generator execution.

    Used for:
    - Saving execution state
    - Branching to alternative paths
    - Merging results from branches
    """
    branch_id: str
    state: Dict[str, Any]
    position: int
    values_yielded: List[Any] = field(default_factory=list)

    def fork(self, new_id: str) -> 'GeneratorStateBranch':
        """Create a fork of this branch state."""
        import copy
        return GeneratorStateBranch(
            branch_id=new_id,
            state=copy.deepcopy(self.state),
            position=self.position,
            values_yielded=list(self.values_yielded)
        )


# =============================================================================
#                              COMPOSER
# =============================================================================

class AdvancedGeneratorComposer:
    """
    Composes generators using various patterns.

    Supports:
    - Sequential chaining
    - Parallel execution with merging
    - Pipeline transformation
    - Conditional branching
    - Recursive compositions

    TODO: Copy full implementation from matts/generator_system.py
    """

    def __init__(self):
        self._registered: Dict[str, Callable] = {}

    def register(self, name: str, factory: Callable[[], Generator]) -> None:
        """Register a generator factory by name."""
        self._registered[name] = factory

    def compose(
        self,
        *generators: Callable,
        pattern: GeneratorCompositionPattern = GeneratorCompositionPattern.SEQUENTIAL
    ) -> Callable[[], Generator]:
        """Compose generators according to pattern."""
        if pattern == GeneratorCompositionPattern.SEQUENTIAL:
            return self._compose_sequential(generators)
        elif pattern == GeneratorCompositionPattern.PIPELINE:
            return self._compose_pipeline(generators)
        elif pattern == GeneratorCompositionPattern.PARALLEL:
            return self._compose_parallel(generators)
        else:
            raise ValueError(f"Unsupported pattern: {pattern}")

    def _compose_sequential(
        self,
        generators: Tuple[Callable, ...]
    ) -> Callable[[], Generator]:
        """Compose generators sequentially."""
        def composed():
            for gen_factory in generators:
                yield from gen_factory()
        return composed

    def _compose_pipeline(
        self,
        generators: Tuple[Callable, ...]
    ) -> Callable[[], Generator]:
        """Compose generators as pipeline (each feeds the next)."""
        def composed():
            if not generators:
                return

            # Start with first generator
            current = generators[0]()

            # Chain through the rest
            for gen_factory in generators[1:]:
                current = gen_factory(current)

            yield from current
        return composed

    def _compose_parallel(
        self,
        generators: Tuple[Callable, ...]
    ) -> Callable[[], Generator]:
        """Compose generators in parallel (interleaved)."""
        def composed():
            # Create all generators
            active = [gen_factory() for gen_factory in generators]

            # Round-robin yield
            while active:
                next_active = []
                for gen in active:
                    try:
                        yield next(gen)
                        next_active.append(gen)
                    except StopIteration:
                        pass
                active = next_active
        return composed


class GeneratorCompositionEngine:
    """
    Engine for complex generator compositions with state tracking.

    TODO: Copy full implementation from matts/generator_system.py
    """

    def __init__(self):
        self.composer = AdvancedGeneratorComposer()
        self._state_branches: Dict[str, GeneratorStateBranch] = {}

    def create_branch(
        self,
        branch_id: str,
        initial_state: Dict[str, Any] = None
    ) -> GeneratorStateBranch:
        """Create a new state branch."""
        branch = GeneratorStateBranch(
            branch_id=branch_id,
            state=initial_state or {},
            position=0
        )
        self._state_branches[branch_id] = branch
        return branch

    def fork_branch(
        self,
        source_id: str,
        new_id: str
    ) -> Optional[GeneratorStateBranch]:
        """Fork an existing branch."""
        source = self._state_branches.get(source_id)
        if source:
            fork = source.fork(new_id)
            self._state_branches[new_id] = fork
            return fork
        return None


# =============================================================================
#                              FACTORIES
# =============================================================================

def create_data_generator_factory(
    data: List[T]
) -> Callable[[], Generator[T, None, None]]:
    """Create a generator factory that yields from a list."""
    def factory():
        yield from data
    return factory


def create_transformer_generator_factory(
    transform: Callable[[T], U]
) -> Callable[[Generator[T, None, None]], Generator[U, None, None]]:
    """Create a generator factory that transforms values."""
    def factory(source: Generator[T, None, None]):
        for item in source:
            yield transform(item)
    return factory


def create_filter_generator_factory(
    predicate: Callable[[T], bool]
) -> Callable[[Generator[T, None, None]], Generator[T, None, None]]:
    """Create a generator factory that filters values."""
    def factory(source: Generator[T, None, None]):
        for item in source:
            if predicate(item):
                yield item
    return factory


def create_aggregator_generator_factory(
    aggregator: Callable[[List[T]], U],
    batch_size: int = 10
) -> Callable[[Generator[T, None, None]], Generator[U, None, None]]:
    """Create a generator factory that aggregates batches."""
    def factory(source: Generator[T, None, None]):
        batch = []
        for item in source:
            batch.append(item)
            if len(batch) >= batch_size:
                yield aggregator(batch)
                batch = []
        if batch:
            yield aggregator(batch)
    return factory
