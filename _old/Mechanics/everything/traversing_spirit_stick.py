"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║              T H E   T R A V E R S I N G   S P I R I T   S T I C K            ║
║                                                                               ║
║  The spirit stick traverses down branches.                                    ║
║  A branch can be ANY class—graph, linked list, sequential list.               ║
║  The finder creates branches based on conditions.                             ║
║  The Resolver receives the PREDICTION for the next story beat.                ║
║                                                                               ║
║  Prose is separated from Chronicle.                                           ║
║  The Dual Oracle: one speaks, one records.                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    List, Dict, Optional, Any, Callable, Iterator, Generic, TypeVar,
    Protocol, Tuple, Union, runtime_checkable, Iterable
)
from abc import ABC, abstractmethod
from enum import Enum, auto
import sqlite3
import json
import time
import uuid


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

T = TypeVar('T')           # Node type
B = TypeVar('B')           # Branch type  
P = TypeVar('P')           # Prediction type


# ═══════════════════════════════════════════════════════════════════════════════
# BRANCH PROTOCOL: Any class that can be traversed
# ═══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class Traversable(Protocol[T]):
    """
    Any class that can be traversed by the spirit stick.
    
    Could be:
        - A graph (yields neighbors)
        - A linked list (yields next)
        - A regular list (yields items in order)
        - A tree (yields children)
    """
    
    def __iter__(self) -> Iterator[T]:
        """Yield traversable elements."""
        ...


@runtime_checkable  
class HasNext(Protocol[T]):
    """Protocol for linked-list style structures."""
    
    @property
    def next(self) -> Optional[T]:
        """Get next node."""
        ...
    
    @property
    def value(self) -> Any:
        """Get node value."""
        ...


@runtime_checkable
class HasChildren(Protocol[T]):
    """Protocol for tree-style structures."""
    
    @property
    def children(self) -> List[T]:
        """Get child nodes."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# BRANCH: Generic wrapper that makes anything traversable
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Branch(Generic[T]):
    """
    A branch the spirit stick can traverse.
    
    Wraps ANY structure and provides uniform traversal.
    The branch itself IS the iterator (wrapper pattern from TraversalWrapper).
    """
    
    # The underlying structure (list, linked list, graph, etc.)
    source: Any
    
    # Current position in traversal
    _current: Optional[T] = field(default=None, repr=False)
    _index: int = field(default=0, repr=False)
    _visited: List[T] = field(default_factory=list, repr=False)
    _path: List[T] = field(default_factory=list, repr=False)
    
    # Branch metadata
    name: str = "branch"
    condition: Optional[Callable[[T], bool]] = None
    
    # Parent branch (for branching/merging)
    parent: Optional['Branch[T]'] = field(default=None, repr=False)
    
    def __post_init__(self):
        # Detect source type and prepare iterator
        self._prepare_source()
    
    def _prepare_source(self):
        """Prepare the source for iteration based on its type."""
        if isinstance(self.source, list):
            self._iter_mode = 'list'
            self._items = self.source
        elif isinstance(self.source, HasNext):
            self._iter_mode = 'linked'
            self._items = None
        elif isinstance(self.source, HasChildren):
            self._iter_mode = 'tree'
            self._items = None
        elif hasattr(self.source, '__iter__'):
            self._iter_mode = 'iterable'
            self._items = list(self.source)
        else:
            # Single item
            self._iter_mode = 'single'
            self._items = [self.source]
    
    # ─────────────────────────────────────────────────────────────────────────
    # Iterator Protocol: The branch IS the iterator
    # ─────────────────────────────────────────────────────────────────────────
    
    def __iter__(self) -> 'Branch[T]':
        """Reset and return self as iterator."""
        self._index = 0
        self._current = None
        self._visited = []
        self._path = []
        return self
    
    def __next__(self) -> T:
        """
        Advance to next node.
        
        This is where the spirit stick MOVES.
        """
        node = self._get_next_node()
        
        if node is None:
            raise StopIteration
        
        # Update state
        self._current = node
        self._visited.append(node)
        self._path.append(node)
        
        return node
    
    def _get_next_node(self) -> Optional[T]:
        """Get next node based on source type."""
        if self._iter_mode == 'list' or self._iter_mode == 'iterable' or self._iter_mode == 'single':
            if self._index >= len(self._items):
                return None
            node = self._items[self._index]
            self._index += 1
            
            # Apply condition filter if present
            if self.condition and not self.condition(node):
                return self._get_next_node()  # Skip, try next
            
            return node
        
        elif self._iter_mode == 'linked':
            if self._current is None:
                # First iteration
                return self.source
            else:
                return self._current.next
        
        elif self._iter_mode == 'tree':
            # BFS traversal of tree
            if self._current is None:
                return self.source
            # Would need queue for proper BFS
            children = self._current.children
            if children and self._index < len(children):
                child = children[self._index]
                self._index += 1
                return child
            return None
        
        return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Current State
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def current(self) -> Optional[T]:
        """Current node in traversal."""
        return self._current
    
    @property
    def path(self) -> List[T]:
        """Path taken so far."""
        return self._path.copy()
    
    @property
    def visited(self) -> List[T]:
        """All visited nodes."""
        return self._visited.copy()
    
    @property
    def depth(self) -> int:
        """Current depth in traversal."""
        return len(self._path)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Branching: Create sub-branches
    # ─────────────────────────────────────────────────────────────────────────
    
    def branch(self, name: str = None, 
               condition: Callable[[T], bool] = None) -> 'Branch[T]':
        """
        Create a new branch from current position.
        
        The new branch starts where we are now.
        """
        # Remaining items from current position
        if self._iter_mode in ('list', 'iterable', 'single'):
            remaining = self._items[self._index:]
        else:
            remaining = [self._current] if self._current else []
        
        return Branch(
            source=remaining,
            name=name or f"{self.name}.branch",
            condition=condition,
            parent=self
        )
    
    def fork(self, branches: List[Tuple[str, Callable[[T], bool]]]) -> List['Branch[T]']:
        """
        Fork into multiple conditional branches.
        
        Each branch follows items matching its condition.
        """
        return [
            self.branch(name=name, condition=cond)
            for name, cond in branches
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# BRANCH FINDER: Creates branches based on conditions
# ═══════════════════════════════════════════════════════════════════════════════

class BranchFinder(Generic[T]):
    """
    Finds and creates branches based on conditions.
    
    A finder function that discovers paths through the story.
    """
    
    def __init__(self, source: Any):
        self.source = source
        self._branches: Dict[str, Branch[T]] = {}
    
    def find(self, name: str, 
             condition: Callable[[T], bool]) -> Branch[T]:
        """
        Find/create a branch matching condition.
        
        Returns existing branch if already found, or creates new one.
        """
        if name in self._branches:
            return self._branches[name]
        
        # Create branch with condition filter
        branch = Branch(
            source=self.source,
            name=name,
            condition=condition
        )
        self._branches[name] = branch
        return branch
    
    def find_where(self, name: str, **attrs) -> Branch[T]:
        """
        Find branch where nodes have specific attributes.
        
        Example: finder.find_where("dark_rooms", lighting="dark")
        """
        def condition(node: T) -> bool:
            for attr, value in attrs.items():
                if hasattr(node, attr):
                    if getattr(node, attr) != value:
                        return False
                elif isinstance(node, dict):
                    if node.get(attr) != value:
                        return False
                else:
                    return False
            return True
        
        return self.find(name, condition)
    
    def find_by_type(self, name: str, node_type: type) -> Branch[T]:
        """Find branch containing only nodes of specific type."""
        return self.find(name, lambda n: isinstance(n, node_type))
    
    def find_sequence(self, name: str, 
                      start: Callable[[T], bool],
                      end: Callable[[T], bool]) -> Branch[T]:
        """
        Find a sequence from start condition to end condition.
        
        Creates a branch that begins when start matches
        and ends when end matches.
        """
        collecting = [False]  # Mutable for closure
        
        def condition(node: T) -> bool:
            if start(node):
                collecting[0] = True
            if collecting[0]:
                if end(node):
                    collecting[0] = False
                    return True  # Include end node
                return True
            return False
        
        return self.find(name, condition)
    
    @property
    def branches(self) -> Dict[str, Branch[T]]:
        """All discovered branches."""
        return self._branches.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION: What the Resolver expects to happen
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Prediction:
    """
    A prediction about what happens next.
    
    The Resolver receives this to guide story generation.
    The prediction comes from lookahead or story structure.
    """
    
    # What we predict will happen
    action: str = ""
    outcome: str = ""
    
    # Where we predict it happens
    location: Optional[str] = None
    
    # Entities involved
    subject: str = "hero"
    object: Optional[str] = None
    
    # Confidence and source
    confidence: float = 1.0
    source: str = "structure"  # structure, lookahead, fate, etc.
    
    # The branch this prediction follows
    branch_name: Optional[str] = None
    
    # Conditions that must be met
    conditions: List[str] = field(default_factory=list)
    
    # What happens if prediction is wrong
    alternative: Optional[str] = None
    
    def matches(self, actual_action: str, actual_location: str = None) -> bool:
        """Check if actual events match this prediction."""
        action_match = self.action.lower() in actual_action.lower()
        if self.location and actual_location:
            location_match = self.location.lower() in actual_location.lower()
            return action_match and location_match
        return action_match


# ═══════════════════════════════════════════════════════════════════════════════
# PROSE vs CHRONICLE: Separated concerns
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Prose:
    """
    The narrative prose—what the player reads.
    
    SEPARATE from the chronicle.
    Prose is ephemeral display; chronicle is permanent record.
    """
    
    text: str = ""
    style: str = "narrative"  # narrative, dialogue, description, action
    
    # Attribution
    voice: str = "narrator"  # narrator, character, world, fate
    
    # Display hints
    emphasis: bool = False
    delay: float = 0.0  # For dramatic effect
    
    # Link to chronicle entry (if any)
    chronicle_id: Optional[str] = None


@dataclass 
class ChronicleEntry:
    """
    A permanent record in the chronicle.
    
    SEPARATE from prose.
    The chronicle is the truth; prose is the telling.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # What happened (factual)
    event_type: str = "action"
    event_data: Dict[str, Any] = field(default_factory=dict)
    
    # When/where
    story_time: int = 0
    location: str = ""
    
    # Who
    actor: str = "hero"
    target: Optional[str] = None
    
    # State changes
    state_before: Dict[str, Any] = field(default_factory=dict)
    state_after: Dict[str, Any] = field(default_factory=dict)
    
    # The prediction that led here (if any)
    prediction_id: Optional[str] = None
    prediction_matched: bool = True
    
    timestamp: float = field(default_factory=time.time)


class Chronicle:
    """
    The permanent record—SQLite storage.
    
    Stores ChronicleEntries, NOT prose.
    """
    
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._story_time = 0
        self._init_db()
    
    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                event_data TEXT,
                story_time INTEGER NOT NULL,
                location TEXT,
                actor TEXT,
                target TEXT,
                state_before TEXT,
                state_after TEXT,
                prediction_id TEXT,
                prediction_matched INTEGER,
                timestamp REAL
            );
            CREATE INDEX IF NOT EXISTS idx_time ON entries(story_time);
        """)
        self.conn.commit()
    
    def record(self, entry: ChronicleEntry) -> ChronicleEntry:
        """Record an entry in the chronicle."""
        self._story_time += 1
        entry.story_time = self._story_time
        
        self.conn.execute("""
            INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            entry.id, entry.event_type, json.dumps(entry.event_data),
            entry.story_time, entry.location, entry.actor, entry.target,
            json.dumps(entry.state_before), json.dumps(entry.state_after),
            entry.prediction_id, 1 if entry.prediction_matched else 0,
            entry.timestamp
        ))
        self.conn.commit()
        return entry
    
    def get_recent(self, n: int = 10) -> List[ChronicleEntry]:
        """Get recent entries."""
        rows = self.conn.execute(
            "SELECT * FROM entries ORDER BY story_time DESC LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_entry(r) for r in reversed(rows)]
    
    def _row_to_entry(self, row) -> ChronicleEntry:
        return ChronicleEntry(
            id=row['id'], event_type=row['event_type'],
            event_data=json.loads(row['event_data'] or '{}'),
            story_time=row['story_time'], location=row['location'] or '',
            actor=row['actor'] or '', target=row['target'],
            state_before=json.loads(row['state_before'] or '{}'),
            state_after=json.loads(row['state_after'] or '{}'),
            prediction_id=row['prediction_id'],
            prediction_matched=bool(row['prediction_matched']),
            timestamp=row['timestamp']
        )
    
    @property
    def story_time(self) -> int:
        return self._story_time


# ═══════════════════════════════════════════════════════════════════════════════
# DUAL ORACLE: One speaks prose, one records chronicle
# ═══════════════════════════════════════════════════════════════════════════════

class ProseOracle:
    """
    The oracle that speaks—generates prose.
    
    Does NOT record to chronicle. Only creates narrative text.
    """
    
    def __init__(self, voice: str = "narrator"):
        self.voice = voice
        self._styles = {
            'action': "The hero {action}.",
            'movement': "Footsteps echo as the journey continues {direction}.",
            'discovery': "Before them lies {object}—{description}.",
            'dialogue': '"{speech}"',
        }
    
    def speak(self, prediction: Prediction, 
              context: Dict[str, Any] = None) -> Prose:
        """
        Generate prose based on prediction.
        
        The prediction tells us WHAT to narrate.
        """
        context = context or {}
        
        # Determine style from prediction
        if 'say' in prediction.action or 'speak' in prediction.action:
            style = 'dialogue'
            text = self._styles['dialogue'].format(
                speech=prediction.outcome or "..."
            )
        elif 'go' in prediction.action or 'move' in prediction.action:
            style = 'movement'
            text = self._styles['movement'].format(
                direction=prediction.outcome or "onward"
            )
        elif 'find' in prediction.action or 'discover' in prediction.action:
            style = 'discovery'
            text = self._styles['discovery'].format(
                object=prediction.object or "something",
                description=prediction.outcome or "mysterious"
            )
        else:
            style = 'action'
            text = self._styles['action'].format(
                action=f"{prediction.action} {prediction.outcome}".strip()
            )
        
        return Prose(
            text=text,
            style=style,
            voice=self.voice
        )
    
    def narrate_freely(self, text: str, style: str = "narrative") -> Prose:
        """Speak arbitrary prose."""
        return Prose(text=text, style=style, voice=self.voice)


class ChronicleOracle:
    """
    The oracle that records—writes to chronicle.
    
    Does NOT generate prose. Only records facts.
    """
    
    def __init__(self, chronicle: Chronicle):
        self.chronicle = chronicle
    
    def record(self, prediction: Prediction,
               actual_action: str,
               location: str,
               state_before: Dict = None,
               state_after: Dict = None) -> ChronicleEntry:
        """
        Record what actually happened.
        
        Compares against prediction to track story coherence.
        """
        entry = ChronicleEntry(
            event_type=prediction.action or actual_action,
            event_data={
                'predicted': prediction.outcome,
                'actual': actual_action,
                'subject': prediction.subject,
                'object': prediction.object,
            },
            location=location,
            actor=prediction.subject,
            target=prediction.object,
            state_before=state_before or {},
            state_after=state_after or {},
            prediction_matched=prediction.matches(actual_action, location)
        )
        
        return self.chronicle.record(entry)


class DualOracle:
    """
    The two oracles working together.
    
    ProseOracle: speaks (generates text)
    ChronicleOracle: records (writes facts)
    
    They are SEPARATE but COORDINATED.
    """
    
    def __init__(self, chronicle: Chronicle = None):
        self.chronicle = chronicle or Chronicle()
        self.prose_oracle = ProseOracle()
        self.chronicle_oracle = ChronicleOracle(self.chronicle)
    
    def process(self, prediction: Prediction,
                actual_action: str,
                location: str,
                state: Dict = None) -> Tuple[Prose, ChronicleEntry]:
        """
        Process an event through both oracles.
        
        Returns (prose for display, entry for record).
        """
        # Generate prose from prediction
        prose = self.prose_oracle.speak(prediction, {'location': location})
        
        # Record facts to chronicle
        entry = self.chronicle_oracle.record(
            prediction, actual_action, location,
            state_before=state,
            state_after=state  # Would be updated
        )
        
        # Link them
        prose.chronicle_id = entry.id
        
        return (prose, entry)


# ═══════════════════════════════════════════════════════════════════════════════
# SPIRIT STICK: The traversing token
# ═══════════════════════════════════════════════════════════════════════════════

class SpiritStick(Generic[T]):
    """
    The spirit stick that traverses branches.
    
    Holds permission to speak/act.
    Moves through the story structure.
    Bridges prose and chronicle through predictions.
    """
    
    def __init__(self, dual_oracle: DualOracle = None):
        self.dual_oracle = dual_oracle or DualOracle()
        
        # Current branch being traversed
        self._branch: Optional[Branch[T]] = None
        
        # Who holds the stick
        self._holder: Optional[str] = None
        
        # Branch history
        self._branch_stack: List[Branch[T]] = []
        
        # Predictions queue
        self._predictions: List[Prediction] = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # Token management
    # ─────────────────────────────────────────────────────────────────────────
    
    def acquire(self, holder_id: str) -> bool:
        """Acquire the stick."""
        if self._holder is not None and self._holder != holder_id:
            return False
        self._holder = holder_id
        return True
    
    def release(self) -> bool:
        """Release the stick."""
        if self._holder is None:
            return False
        self._holder = None
        return True
    
    @property
    def holder(self) -> Optional[str]:
        return self._holder
    
    # ─────────────────────────────────────────────────────────────────────────
    # Branch traversal
    # ─────────────────────────────────────────────────────────────────────────
    
    def set_branch(self, branch: Branch[T]):
        """Set the branch to traverse."""
        if self._branch:
            self._branch_stack.append(self._branch)
        self._branch = branch
    
    def pop_branch(self) -> Optional[Branch[T]]:
        """Return to previous branch."""
        if self._branch_stack:
            old = self._branch
            self._branch = self._branch_stack.pop()
            return old
        return None
    
    @property
    def current_branch(self) -> Optional[Branch[T]]:
        return self._branch
    
    @property
    def current_node(self) -> Optional[T]:
        return self._branch.current if self._branch else None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Traversal with predictions
    # ─────────────────────────────────────────────────────────────────────────
    
    def push_prediction(self, prediction: Prediction):
        """Add a prediction for what comes next."""
        self._predictions.append(prediction)
    
    def pop_prediction(self) -> Optional[Prediction]:
        """Get next prediction."""
        if self._predictions:
            return self._predictions.pop(0)
        return None
    
    def peek_prediction(self) -> Optional[Prediction]:
        """Look at next prediction without consuming."""
        if self._predictions:
            return self._predictions[0]
        return None
    
    def step(self) -> Tuple[Optional[T], Optional[Prose], Optional[ChronicleEntry]]:
        """
        Take one step through the branch.
        
        Uses prediction to generate prose and record chronicle.
        Returns (node, prose, chronicle_entry).
        """
        if not self._branch:
            return (None, None, None)
        
        # Get prediction for this step
        prediction = self.pop_prediction() or Prediction(action="continue")
        
        # Advance through branch
        try:
            node = next(self._branch)
        except StopIteration:
            return (None, None, None)
        
        # Process through dual oracle
        prose, entry = self.dual_oracle.process(
            prediction=prediction,
            actual_action=self._node_to_action(node),
            location=self._node_to_location(node),
            state={'node': str(node)}
        )
        
        return (node, prose, entry)
    
    def _node_to_action(self, node: T) -> str:
        """Extract action from node."""
        if hasattr(node, 'action'):
            return node.action
        if isinstance(node, dict):
            return node.get('action', str(node))
        return str(node)
    
    def _node_to_location(self, node: T) -> str:
        """Extract location from node."""
        if hasattr(node, 'location'):
            return node.location
        if isinstance(node, dict):
            return node.get('location', 'unknown')
        return 'unknown'
    
    # ─────────────────────────────────────────────────────────────────────────
    # Iterator interface: traverse with the stick
    # ─────────────────────────────────────────────────────────────────────────
    
    def traverse(self, branch: Branch[T] = None) -> Iterator[Tuple[T, Prose, ChronicleEntry]]:
        """
        Traverse a branch, yielding (node, prose, entry) tuples.
        
        This is the main iteration interface.
        """
        if branch:
            self.set_branch(branch)
        
        if not self._branch:
            return
        
        for node in self._branch:
            prediction = self.pop_prediction() or Prediction(action="continue")
            
            prose, entry = self.dual_oracle.process(
                prediction=prediction,
                actual_action=self._node_to_action(node),
                location=self._node_to_location(node),
                state={'node': str(node)}
            )
            
            yield (node, prose, entry)


# ═══════════════════════════════════════════════════════════════════════════════
# RESOLVER: Receives predictions, coordinates everything
# ═══════════════════════════════════════════════════════════════════════════════

class Resolver:
    """
    The Resolver coordinates predictions, branches, and oracles.
    
    It receives the PREDICTION for the next story beat,
    then uses the spirit stick to traverse and generate.
    """
    
    def __init__(self, spirit_stick: SpiritStick = None):
        self.stick = spirit_stick or SpiritStick()
        self._branch_finder: Optional[BranchFinder] = None
    
    def set_story(self, story_nodes: Any):
        """Set the story structure to traverse."""
        self._branch_finder = BranchFinder(story_nodes)
        
        # Create main branch
        main = self._branch_finder.find("main", lambda n: True)
        self.stick.set_branch(main)
    
    def predict(self, prediction: Prediction):
        """
        Give the resolver a prediction.
        
        This prediction guides the next story beat.
        """
        self.stick.push_prediction(prediction)
    
    def predict_sequence(self, predictions: List[Prediction]):
        """Queue multiple predictions."""
        for p in predictions:
            self.stick.push_prediction(p)
    
    def resolve_next(self) -> Optional[Tuple[Any, Prose, ChronicleEntry]]:
        """
        Resolve the next story beat.
        
        Uses queued prediction to generate prose and record chronicle.
        """
        return self.stick.step()
    
    def resolve_branch(self, branch_name: str,
                       condition: Callable = None) -> Iterator[Tuple[Any, Prose, ChronicleEntry]]:
        """
        Resolve along a specific branch.
        
        Finds or creates branch, then traverses it.
        """
        if not self._branch_finder:
            return
        
        branch = self._branch_finder.find(
            branch_name, 
            condition or (lambda n: True)
        )
        
        yield from self.stick.traverse(branch)
    
    def fork_and_resolve(self, 
                         branches: List[Tuple[str, Callable, List[Prediction]]]) -> Dict[str, List]:
        """
        Fork into multiple branches, resolve each.
        
        Args:
            branches: List of (name, condition, predictions)
        
        Returns:
            Dict mapping branch name to list of (node, prose, entry)
        """
        results = {}
        
        for name, condition, predictions in branches:
            # Queue predictions for this branch
            self.predict_sequence(predictions)
            
            # Resolve the branch
            results[name] = list(self.resolve_branch(name, condition))
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE STORY NODES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StoryNode:
    """A node in the story structure."""
    id: str
    action: str
    location: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # For linked list style
    next_node: Optional['StoryNode'] = None
    
    # For tree style
    children: List['StoryNode'] = field(default_factory=list)
    
    @property
    def next(self) -> Optional['StoryNode']:
        return self.next_node
    
    @property
    def value(self) -> str:
        return self.description


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def demo_traversing_spirit_stick():
    """Demonstrate the traversing spirit stick."""
    print("\n" + "═" * 70)
    print("THE TRAVERSING SPIRIT STICK")
    print("═" * 70 + "\n")
    
    # Create story nodes
    nodes = [
        StoryNode("1", "enter", "White Room", "awakening", ["start"]),
        StoryNode("2", "look", "White Room", "observation", ["explore"]),
        StoryNode("3", "go north", "Corridor", "movement", ["travel"]),
        StoryNode("4", "find sword", "Corridor", "discovery", ["item", "weapon"]),
        StoryNode("5", "take sword", "Corridor", "action", ["acquire"]),
        StoryNode("6", "go east", "Chamber", "movement", ["travel"]),
        StoryNode("7", "encounter guardian", "Chamber", "combat", ["enemy", "boss"]),
    ]
    
    # Create the system
    dual_oracle = DualOracle()
    stick = SpiritStick(dual_oracle)
    resolver = Resolver(stick)
    
    # Set the story
    resolver.set_story(nodes)
    
    # Create predictions for what we expect to happen
    predictions = [
        Prediction(action="enter", outcome="consciousness returns", location="White Room"),
        Prediction(action="look", outcome="walls shimmer", location="White Room"),
        Prediction(action="go", outcome="northward", location="Corridor"),
        Prediction(action="find", outcome="ancient blade", object="sword"),
        Prediction(action="take", outcome="grip tightens", object="sword"),
        Prediction(action="go", outcome="eastward", location="Chamber"),
        Prediction(action="encounter", outcome="shadow stirs", object="guardian"),
    ]
    
    resolver.predict_sequence(predictions)
    
    # Traverse with predictions
    print("STORY TRAVERSAL:")
    print("─" * 70)
    
    for node, prose, entry in stick.traverse():
        print(f"\n[Node: {node.id}] {node.action} @ {node.location}")
        print(f"  Prose: {prose.text}")
        print(f"  Chronicle T{entry.story_time}: {entry.event_type}")
    
    # Show chronicle
    print("\n" + "─" * 70)
    print("CHRONICLE ENTRIES:")
    for entry in dual_oracle.chronicle.get_recent(10):
        matched = "✓" if entry.prediction_matched else "✗"
        print(f"  T{entry.story_time:02d} [{matched}] {entry.event_type} @ {entry.location}")
    
    # Demonstrate branch finding
    print("\n" + "─" * 70)
    print("BRANCH FINDING:")
    
    finder = BranchFinder(nodes)
    
    # Find travel nodes
    travel_branch = finder.find_where("travel", tags=["travel"])
    print(f"\nTravel branch: {travel_branch.name}")
    for node in travel_branch:
        print(f"  - {node.action} to {node.location}")
    
    # Find item-related nodes
    item_branch = finder.find("items", lambda n: "item" in n.tags)
    print(f"\nItem branch: {item_branch.name}")
    for node in item_branch:
        print(f"  - {node.action}: {node.description}")


def demo_linked_list_branch():
    """Demonstrate with linked list structure."""
    print("\n" + "═" * 70)
    print("LINKED LIST TRAVERSAL")
    print("═" * 70 + "\n")
    
    # Create linked list
    n4 = StoryNode("4", "victory", "Throne", "triumph")
    n3 = StoryNode("3", "battle", "Arena", "combat", next_node=n4)
    n2 = StoryNode("2", "challenge", "Gates", "trial", next_node=n3)
    n1 = StoryNode("1", "arrival", "City", "beginning", next_node=n2)
    
    # Create branch from linked list
    branch = Branch(source=n1, name="quest")
    
    print("Traversing linked story:")
    for node in branch:
        print(f"  {node.id}: {node.action} @ {node.location}")


def demo_conditional_branching():
    """Demonstrate conditional branch creation."""
    print("\n" + "═" * 70)
    print("CONDITIONAL BRANCHING")
    print("═" * 70 + "\n")
    
    nodes = [
        {"id": 1, "type": "combat", "enemy": "goblin"},
        {"id": 2, "type": "explore", "location": "cave"},
        {"id": 3, "type": "combat", "enemy": "troll"},
        {"id": 4, "type": "dialogue", "npc": "wizard"},
        {"id": 5, "type": "combat", "enemy": "dragon"},
        {"id": 6, "type": "explore", "location": "treasury"},
    ]
    
    finder = BranchFinder(nodes)
    
    # Combat-only branch
    combat = finder.find("combat", lambda n: n["type"] == "combat")
    print("Combat encounters:")
    for node in combat:
        print(f"  Fight: {node['enemy']}")
    
    # Exploration branch
    explore = finder.find("explore", lambda n: n["type"] == "explore")
    print("\nExploration:")
    for node in explore:
        print(f"  Visit: {node['location']}")
    
    print(f"\nAll branches: {list(finder.branches.keys())}")


if __name__ == "__main__":
    demo_traversing_spirit_stick()
    demo_linked_list_branch()
    demo_conditional_branching()
