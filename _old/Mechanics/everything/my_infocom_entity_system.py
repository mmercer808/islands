"""
my-infocom Entity & Prose System
================================

Core implementation of:
- Entity Graph (nodes, links, queries)
- Prose Fragmentation & Composition
- Action Processing
- Image Pipeline Integration

See my_infocom_design_document.md for full design rationale.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Callable, Union, Iterator
from enum import Enum, auto
from weakref import ref, ReferenceType
from collections import defaultdict
import uuid
import random
import json


# ============================================================
# SECTION 1: ENUMERATIONS
# ============================================================

class EntityType(Enum):
    """What kind of game-world thing this is"""
    LOCATION = auto()
    ITEM = auto()
    CHARACTER = auto()
    CONTAINER = auto()
    DOOR = auto()
    CONCEPT = auto()  # Abstract: clues, knowledge, memories


class EntityState(Enum):
    """Mutable state an entity can be in"""
    NORMAL = auto()
    HIDDEN = auto()
    LOCKED = auto()
    OPEN = auto()
    CLOSED = auto()
    BROKEN = auto()


class LinkType(Enum):
    """Primary relationship categories"""
    RELATIONAL = auto()  # Spatial, possession, social
    LOGICAL = auto()     # Conditional game logic
    WILDCARD = auto()    # Custom/narrative connections


class RelationalKind(Enum):
    """Subtypes for RELATIONAL links"""
    # Spatial
    IN = "in"
    ON = "on"
    UNDER = "under"
    NEAR = "near"
    NORTH_OF = "north_of"
    SOUTH_OF = "south_of"
    EAST_OF = "east_of"
    WEST_OF = "west_of"
    # Containment
    CONTAINS = "contains"
    PART_OF = "part_of"
    # Possession
    HELD_BY = "held_by"
    OWNED_BY = "owned_by"
    # Social
    KNOWS = "knows"
    TRUSTS = "trusts"


class LogicalKind(Enum):
    """Subtypes for LOGICAL links"""
    REQUIRES = "requires"
    UNLOCKS = "unlocks"
    REVEALS = "reveals"
    ENABLES = "enables"
    BLOCKS = "blocks"
    TRIGGERS = "triggers"


class FragmentCategory(Enum):
    """
    Types of prose fragments, ordered by typical composition priority.
    Lower value = appears earlier in composed output.
    """
    BASE_DESCRIPTION = 10
    ATMOSPHERIC = 20
    STATE_CHANGE = 30
    ITEM_PRESENCE = 40
    NPC_AMBIENT = 50
    SENSORY = 60
    HISTORY = 70
    DISCOVERY = 80


# ============================================================
# SECTION 2: PROSE FRAGMENTATION SYSTEM
# ============================================================

@dataclass
class FragmentCondition:
    """
    A condition that must be true for a fragment to be included.
    
    The predicate receives the entity and a context dict.
    Description is for debugging/authoring tools.
    """
    predicate: Callable[[Any, Dict[str, Any]], bool]
    description: str = ""
    
    def evaluate(self, entity: Any, context: Dict[str, Any]) -> bool:
        try:
            return self.predicate(entity, context)
        except Exception:
            return False


@dataclass
class ProseFragment:
    """
    A single piece of composable prose.
    
    Fragments are the atomic units of description. They get filtered
    by conditions, sorted by category/priority, and joined into
    the final output the player sees.
    """
    text: str
    category: FragmentCategory
    priority: int = 50  # Higher = more important within category
    
    # Optional conditions for inclusion
    conditions: List[FragmentCondition] = field(default_factory=list)
    
    # Alternative phrasings (for variety)
    variations: List[str] = field(default_factory=list)
    
    # Tags for filtering/grouping
    tags: Set[str] = field(default_factory=set)
    
    # If true, only show once ever
    one_shot: bool = False
    _shown: bool = field(default=False, repr=False)
    
    def get_text(self, vary: bool = True) -> str:
        """Get the text, optionally selecting a variation."""
        if vary and self.variations:
            return random.choice([self.text] + self.variations)
        return self.text
    
    def is_available(self, entity: Any, context: Dict[str, Any]) -> bool:
        """Check if this fragment should be included."""
        if self.one_shot and self._shown:
            return False
        return all(c.evaluate(entity, context) for c in self.conditions)
    
    def mark_shown(self) -> None:
        """Mark as shown (for one_shot fragments)."""
        self._shown = True


class ProseCompositor:
    """
    Assembles fragments into player-facing prose.
    
    This is the core "how prose gets built" logic.
    """
    
    def __init__(self, paragraph_gap: str = "\n\n"):
        self.paragraph_gap = paragraph_gap
    
    def compose(self, 
                fragments: List[ProseFragment],
                entity: Any,
                context: Dict[str, Any],
                max_fragments: int = 10) -> str:
        """
        Compose fragments into final prose.
        
        Process:
        1. Filter by conditions
        2. Sort by category (enum value) then priority (descending)
        3. Take top N fragments
        4. Get text (with variation)
        5. Join appropriately
        """
        # Step 1: Filter
        available = [f for f in fragments if f.is_available(entity, context)]
        
        # Step 2: Sort
        # Lower category value = earlier; higher priority = earlier within category
        available.sort(key=lambda f: (f.category.value, -f.priority))
        
        # Step 3: Limit
        selected = available[:max_fragments]
        
        # Step 4: Get text and mark shown
        texts = []
        for frag in selected:
            texts.append(frag.get_text(vary=True))
            if frag.one_shot:
                frag.mark_shown()
        
        # Step 5: Join
        # Group by category for paragraph breaks
        return self._join_texts(texts, selected)
    
    def _join_texts(self, texts: List[str], fragments: List[ProseFragment]) -> str:
        """
        Join texts with appropriate spacing.
        Inserts paragraph breaks between different categories.
        """
        if not texts:
            return ""
        
        result_parts = []
        current_category = None
        current_group = []
        
        for text, frag in zip(texts, fragments):
            if current_category is None:
                current_category = frag.category
            
            if frag.category != current_category:
                # New category: flush current group as paragraph
                result_parts.append(" ".join(current_group))
                current_group = []
                current_category = frag.category
            
            current_group.append(text)
        
        # Flush final group
        if current_group:
            result_parts.append(" ".join(current_group))
        
        return self.paragraph_gap.join(result_parts)


# ============================================================
# SECTION 3: LINKS
# ============================================================

@dataclass
class LinkCondition:
    """Condition for a link to be active/traversable."""
    predicate: Callable[[Dict[str, Any]], bool]
    description: str = ""
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        try:
            return self.predicate(context)
        except Exception:
            return False


@dataclass
class Link:
    """
    An edge in the entity graph.
    
    Links are directional (source → target) and typed.
    They can be conditional (only active when condition passes).
    """
    id: str
    link_type: LinkType
    source_id: str
    target_id: str
    kind: Union[RelationalKind, LogicalKind, str, None] = None
    
    condition: Optional[LinkCondition] = None
    bidirectional: bool = False
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    active: bool = True  # Can be toggled at runtime
    
    def is_traversable(self, context: Dict[str, Any]) -> bool:
        """Check if link can currently be traversed."""
        if not self.active:
            return False
        if self.condition:
            return self.condition.evaluate(context)
        return True


# ============================================================
# SECTION 4: NODES & ENTITIES
# ============================================================

class Node(ABC):
    """
    Abstract base for all graph vertices.
    
    Handles graph connectivity. Entities extend this with
    game-specific functionality.
    """
    
    def __init__(self, node_id: Optional[str] = None):
        self._id = node_id or str(uuid.uuid4())
        self._graph: Optional[ReferenceType[EntityGraph]] = None
        self._outgoing: Dict[str, Link] = {}
        self._incoming: Dict[str, Link] = {}
        self._tags: Set[str] = set()
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def graph(self) -> Optional[EntityGraph]:
        return self._graph() if self._graph else None
    
    @property
    def tags(self) -> Set[str]:
        return self._tags.copy()
    
    def add_tag(self, tag: str) -> None:
        self._tags.add(tag)
    
    def has_tag(self, tag: str) -> bool:
        return tag in self._tags
    
    def get_links_out(self, 
                      link_type: Optional[LinkType] = None,
                      context: Optional[Dict] = None) -> List[Link]:
        """Get outgoing links, optionally filtered."""
        ctx = context or {}
        links = list(self._outgoing.values())
        if link_type:
            links = [l for l in links if l.link_type == link_type]
        return [l for l in links if l.is_traversable(ctx)]
    
    def get_links_in(self,
                     link_type: Optional[LinkType] = None,
                     context: Optional[Dict] = None) -> List[Link]:
        """Get incoming links, optionally filtered."""
        ctx = context or {}
        links = list(self._incoming.values())
        if link_type:
            links = [l for l in links if l.link_type == link_type]
        return [l for l in links if l.is_traversable(ctx)]
    
    def get_neighbors(self, 
                      link_type: Optional[LinkType] = None,
                      context: Optional[Dict] = None) -> List[Node]:
        """Get nodes connected via outgoing links."""
        if not self.graph:
            return []
        links = self.get_links_out(link_type, context)
        nodes = [self.graph.get_node(l.target_id) for l in links]
        return [n for n in nodes if n is not None]


@dataclass
class Vocabulary:
    """Words that can refer to an entity."""
    nouns: Set[str] = field(default_factory=set)
    adjectives: Set[str] = field(default_factory=set)
    
    def matches(self, words: List[str]) -> bool:
        """Check if given words could refer to this entity."""
        words_lower = {w.lower() for w in words}
        # Must match at least one noun
        if not (words_lower & self.nouns):
            return False
        # Any adjectives given must match
        non_nouns = words_lower - self.nouns
        return non_nouns.issubset(self.adjectives)


class Entity(Node):
    """
    A game-world thing with prose, state, and interactions.
    
    Entities are the "stuff" of the game: rooms, items, NPCs.
    They have fragments for prose composition and handlers
    for player actions.
    """
    
    def __init__(self,
                 name: str,
                 entity_type: EntityType,
                 node_id: Optional[str] = None):
        super().__init__(node_id)
        
        self.name = name
        self.entity_type = entity_type
        
        # Prose
        self.fragments: List[ProseFragment] = []
        
        # Parser vocabulary
        self.vocabulary = Vocabulary(nouns={name.lower()})
        
        # State
        self.state: EntityState = EntityState.NORMAL
        self.properties: Dict[str, Any] = {}
        
        # Interaction handlers: verb -> callable
        self._handlers: Dict[str, Callable[[Entity, Dict], str]] = {}
    
    # ---- Prose ----
    
    def add_fragment(self, fragment: ProseFragment) -> Entity:
        """Add a prose fragment."""
        self.fragments.append(fragment)
        return self
    
    def add_description(self, 
                        text: str, 
                        category: FragmentCategory = FragmentCategory.BASE_DESCRIPTION,
                        **kwargs) -> Entity:
        """Convenience: add a simple text fragment."""
        self.fragments.append(ProseFragment(text=text, category=category, **kwargs))
        return self
    
    def get_description(self, context: Optional[Dict] = None) -> str:
        """Compose and return description from fragments."""
        compositor = ProseCompositor()
        return compositor.compose(self.fragments, self, context or {})
    
    # ---- Vocabulary ----
    
    def add_nouns(self, *nouns: str) -> Entity:
        self.vocabulary.nouns.update(n.lower() for n in nouns)
        return self
    
    def add_adjectives(self, *adjs: str) -> Entity:
        self.vocabulary.adjectives.update(a.lower() for a in adjs)
        return self
    
    def matches_words(self, words: List[str]) -> bool:
        return self.vocabulary.matches(words)
    
    # ---- State ----
    
    @property
    def is_hidden(self) -> bool:
        return self.state == EntityState.HIDDEN
    
    def reveal(self) -> None:
        if self.state == EntityState.HIDDEN:
            self.state = EntityState.NORMAL
    
    def hide(self) -> None:
        self.state = EntityState.HIDDEN
    
    # ---- Interactions ----
    
    def on(self, verb: str, handler: Callable[[Entity, Dict], str]) -> Entity:
        """Register a handler for an action."""
        self._handlers[verb.lower()] = handler
        return self
    
    def handle(self, verb: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Handle an action. Returns response prose or None if no handler.
        """
        handler = self._handlers.get(verb.lower())
        if handler:
            return handler(self, context)
        return None
    
    def can_handle(self, verb: str) -> bool:
        return verb.lower() in self._handlers
    
    def __repr__(self) -> str:
        return f"Entity({self.name!r}, {self.entity_type.name})"


# ============================================================
# SECTION 5: ENTITY GRAPH
# ============================================================

class EntityGraph:
    """
    The main data structure holding all nodes and links.
    
    Provides registration, linking, querying, and traversal.
    """
    
    def __init__(self, name: str = "world"):
        self.name = name
        self._nodes: Dict[str, Node] = {}
        self._links: Dict[str, Link] = {}
        
        # Indexes for fast lookup
        self._by_type: Dict[EntityType, Set[str]] = defaultdict(set)
        self._by_tag: Dict[str, Set[str]] = defaultdict(set)
    
    # ---- Registration ----
    
    def register(self, node: Node) -> Node:
        """Add a node to the graph."""
        if node.id in self._nodes:
            raise ValueError(f"Node {node.id} already exists")
        
        self._nodes[node.id] = node
        node._graph = ref(self)
        
        # Index
        if isinstance(node, Entity):
            self._by_type[node.entity_type].add(node.id)
        for tag in node.tags:
            self._by_tag[tag].add(node.id)
        
        return node
    
    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)
    
    def get_entity(self, node_id: str) -> Optional[Entity]:
        node = self._nodes.get(node_id)
        return node if isinstance(node, Entity) else None
    
    def find_by_name(self, name: str) -> Optional[Entity]:
        """Find first entity with matching name."""
        name_lower = name.lower()
        for node in self._nodes.values():
            if isinstance(node, Entity) and node.name.lower() == name_lower:
                return node
        return None
    
    # ---- Linking ----
    
    def link(self,
             source: Union[Node, str],
             target: Union[Node, str],
             link_type: LinkType,
             kind: Union[RelationalKind, LogicalKind, str, None] = None,
             bidirectional: bool = False,
             condition: Optional[LinkCondition] = None,
             **metadata) -> Link:
        """Create a link between two nodes."""
        source_id = source.id if isinstance(source, Node) else source
        target_id = target.id if isinstance(target, Node) else target
        
        link = Link(
            id=str(uuid.uuid4()),
            link_type=link_type,
            source_id=source_id,
            target_id=target_id,
            kind=kind,
            bidirectional=bidirectional,
            condition=condition,
            metadata=metadata
        )
        
        self._add_link(link)
        
        if bidirectional:
            inverse = Link(
                id=str(uuid.uuid4()),
                link_type=link_type,
                source_id=target_id,
                target_id=source_id,
                kind=self._inverse_kind(kind),
                condition=condition,
                metadata={**metadata, '_inverse_of': link.id}
            )
            self._add_link(inverse)
        
        return link
    
    def _add_link(self, link: Link) -> None:
        self._links[link.id] = link
        source = self._nodes.get(link.source_id)
        target = self._nodes.get(link.target_id)
        if source:
            source._outgoing[link.id] = link
        if target:
            target._incoming[link.id] = link
    
    def _inverse_kind(self, kind):
        """Get inverse relationship for bidirectional links."""
        inverses = {
            RelationalKind.IN: RelationalKind.CONTAINS,
            RelationalKind.CONTAINS: RelationalKind.IN,
            RelationalKind.NORTH_OF: RelationalKind.SOUTH_OF,
            RelationalKind.SOUTH_OF: RelationalKind.NORTH_OF,
            RelationalKind.EAST_OF: RelationalKind.WEST_OF,
            RelationalKind.WEST_OF: RelationalKind.EAST_OF,
        }
        return inverses.get(kind, kind)
    
    # ---- Convenience link methods ----
    
    def relate(self, source, target, kind: RelationalKind, 
               bidirectional: bool = True, **kw) -> Link:
        return self.link(source, target, LinkType.RELATIONAL, 
                        kind=kind, bidirectional=bidirectional, **kw)
    
    def logic(self, source, target, kind: LogicalKind,
              condition: Optional[LinkCondition] = None, **kw) -> Link:
        return self.link(source, target, LinkType.LOGICAL,
                        kind=kind, condition=condition, **kw)
    
    # ---- Queries ----
    
    def get_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Fast lookup by entity type."""
        ids = self._by_type.get(entity_type, set())
        return [self._nodes[nid] for nid in ids 
                if nid in self._nodes and isinstance(self._nodes[nid], Entity)]
    
    def get_by_tag(self, tag: str) -> List[Node]:
        """Fast lookup by tag."""
        ids = self._by_tag.get(tag, set())
        return [self._nodes[nid] for nid in ids if nid in self._nodes]
    
    def query(self, predicate: Callable[[Node], bool]) -> List[Node]:
        """Find all nodes matching predicate."""
        return [n for n in self._nodes.values() if predicate(n)]
    
    # ---- Traversal ----
    
    def find_path(self, 
                  start: Union[Node, str],
                  end: Union[Node, str],
                  link_type: Optional[LinkType] = None,
                  context: Optional[Dict] = None) -> Optional[List[Node]]:
        """BFS pathfinding between two nodes."""
        start_id = start.id if isinstance(start, Node) else start
        end_id = end.id if isinstance(end, Node) else end
        
        start_node = self._nodes.get(start_id)
        if not start_node:
            return None
        
        visited = {start_id}
        queue = [[start_node]]
        
        while queue:
            path = queue.pop(0)
            current = path[-1]
            
            if current.id == end_id:
                return path
            
            for neighbor in current.get_neighbors(link_type, context):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append(path + [neighbor])
        
        return None
    
    # ---- Stats ----
    
    @property
    def node_count(self) -> int:
        return len(self._nodes)
    
    @property
    def link_count(self) -> int:
        return len(self._links)
    
    def __repr__(self) -> str:
        return f"EntityGraph({self.name!r}, {self.node_count} nodes, {self.link_count} links)"


# ============================================================
# SECTION 6: ACTION PROCESSING
# ============================================================

@dataclass
class Action:
    """
    A player action to be processed.
    
    Created by the interaction layer (parser, choice UI, etc.)
    and processed by the ActionProcessor.
    """
    verb: str
    target_id: Optional[str] = None
    instrument_id: Optional[str] = None  # For "use X on Y"
    raw_input: str = ""


class ActionProcessor:
    """
    Processes player actions against the entity graph.
    
    Handles:
    - Scope checking (can player access target?)
    - Handler dispatch (does entity handle this verb?)
    - Default actions (examine, take, etc.)
    - State propagation (check for triggered revelations)
    """
    
    def __init__(self, graph: EntityGraph):
        self.graph = graph
        self.compositor = ProseCompositor()
    
    def process(self, action: Action, game_state: Dict[str, Any]) -> str:
        """
        Process an action and return response prose.
        """
        # Get target entity
        target = self.graph.get_entity(action.target_id) if action.target_id else None
        
        if not target:
            return "You don't see that here."
        
        # Check scope (is target accessible?)
        if not self._in_scope(target, game_state):
            return "You can't reach that from here."
        
        # Build context for handlers
        context = {
            **game_state,
            'action': action,
            'target': target,
        }
        
        # Try entity's custom handler
        if target.can_handle(action.verb):
            response = target.handle(action.verb, context)
            # After handling, check for triggered links
            self._propagate_state(target, context)
            return response or ""
        
        # Fall back to default handlers
        return self._default_handler(action.verb, target, context)
    
    def _in_scope(self, entity: Entity, game_state: Dict) -> bool:
        """Check if entity is accessible to player."""
        if entity.is_hidden:
            return False
        
        # TODO: Check location containment, inventory, etc.
        # For now, everything visible is in scope
        return True
    
    def _default_handler(self, verb: str, target: Entity, context: Dict) -> str:
        """Handle common verbs that most entities don't override."""
        if verb in ('examine', 'look', 'x'):
            target.properties['examined'] = True
            return target.get_description(context)
        
        if verb in ('take', 'get', 'grab'):
            if target.has_tag('takeable'):
                # TODO: Move to inventory
                return f"You take the {target.name}."
            return f"You can't take the {target.name}."
        
        return f"You can't {verb} that."
    
    def _propagate_state(self, changed_entity: Entity, context: Dict) -> None:
        """
        After entity state changes, check for activated links.
        
        This is how "search desk reveals key" works:
        1. Player searches desk
        2. Handler sets desk.properties['searched'] = True
        3. This method runs, finds REVEALS link to key
        4. Link condition checks desk.properties['searched']
        5. Condition passes, so key.reveal() is called
        """
        for link in changed_entity.get_links_out(LinkType.LOGICAL, context):
            if link.kind == LogicalKind.REVEALS:
                target = self.graph.get_entity(link.target_id)
                if target and target.is_hidden:
                    target.reveal()
            # Handle other logical link types...


# ============================================================
# SECTION 7: IMAGE PIPELINE INTEGRATION
# ============================================================

def entity_from_pipeline(data: Dict[str, Any]) -> Entity:
    """
    Convert image pipeline output to Entity.
    
    Expected structure:
    {
        "name": str,
        "entity_type": str,
        "vlm_description": {"summary": str, "mood": str, ...},
        "narrative": {
            "prose_description": str,
            "history_hints": [str],
            "sensory_details": {"sight": str, "sound": str, ...}
        },
        "detection": {"label": str, "confidence": float}
    }
    """
    # Map type string to enum
    type_map = {
        'location': EntityType.LOCATION,
        'structure': EntityType.LOCATION,
        'item': EntityType.ITEM,
        'character': EntityType.CHARACTER,
        'vehicle': EntityType.ITEM,
    }
    etype = type_map.get(data.get('entity_type', 'item').lower(), EntityType.ITEM)
    
    entity = Entity(data['name'], etype)
    
    # Base description from narrative
    narrative = data.get('narrative', {})
    if narrative.get('prose_description'):
        entity.add_description(
            narrative['prose_description'],
            FragmentCategory.BASE_DESCRIPTION,
            priority=100
        )
    
    # Atmospheric from VLM mood
    vlm = data.get('vlm_description', {})
    if vlm.get('mood'):
        mood_text = _mood_to_prose(vlm['mood'])
        if mood_text:
            entity.add_description(mood_text, FragmentCategory.ATMOSPHERIC)
    
    # Sensory fragments
    for sense, text in narrative.get('sensory_details', {}).items():
        entity.add_fragment(ProseFragment(
            text=text,
            category=FragmentCategory.SENSORY,
            tags={sense}
        ))
    
    # History as conditional (revealed through examination)
    for i, hint in enumerate(narrative.get('history_hints', [])):
        entity.add_fragment(ProseFragment(
            text=hint,
            category=FragmentCategory.HISTORY,
            priority=50 - i,  # Earlier hints higher priority
            conditions=[FragmentCondition(
                lambda e, c: e.properties.get('examined', False),
                "Must examine first"
            )]
        ))
    
    # Store detection metadata
    detection = data.get('detection', {})
    entity.properties['detection_label'] = detection.get('label')
    entity.properties['detection_confidence'] = detection.get('confidence')
    
    # Add tags based on type
    if etype == EntityType.LOCATION:
        entity.add_tag('visitable')
    elif etype == EntityType.ITEM:
        entity.add_tag('examinable')
    
    return entity


def _mood_to_prose(mood: str) -> Optional[str]:
    """Convert mood keyword to atmospheric prose."""
    moods = {
        'mysterious': "An air of mystery hangs over the scene.",
        'ominous': "Something feels deeply wrong here.",
        'tranquil': "A sense of peace pervades the area.",
        'abandoned': "Neglect and decay are evident everywhere.",
        'bustling': "Signs of activity and life are all around.",
    }
    return moods.get(mood.lower())


# ============================================================
# SECTION 8: SERIALIZATION
# ============================================================

def serialize_graph(graph: EntityGraph) -> Dict[str, Any]:
    """Export graph to JSON-serializable dict."""
    return {
        'name': graph.name,
        'entities': [_serialize_entity(e) for e in graph._nodes.values() 
                     if isinstance(e, Entity)],
        'links': [_serialize_link(l) for l in graph._links.values()]
    }


def _serialize_entity(entity: Entity) -> Dict[str, Any]:
    return {
        'id': entity.id,
        'name': entity.name,
        'type': entity.entity_type.name,
        'state': entity.state.name,
        'properties': entity.properties,
        'tags': list(entity.tags),
        'vocabulary': {
            'nouns': list(entity.vocabulary.nouns),
            'adjectives': list(entity.vocabulary.adjectives)
        },
        'fragments': [_serialize_fragment(f) for f in entity.fragments]
    }


def _serialize_fragment(frag: ProseFragment) -> Dict[str, Any]:
    return {
        'text': frag.text,
        'category': frag.category.name,
        'priority': frag.priority,
        'variations': frag.variations,
        'tags': list(frag.tags),
        'one_shot': frag.one_shot
        # Note: conditions not serialized (they're callables)
    }


def _serialize_link(link: Link) -> Dict[str, Any]:
    kind_value = link.kind.value if isinstance(link.kind, Enum) else link.kind
    return {
        'id': link.id,
        'type': link.link_type.name,
        'source': link.source_id,
        'target': link.target_id,
        'kind': kind_value,
        'bidirectional': link.bidirectional,
        'metadata': link.metadata
    }


# ============================================================
# SECTION 9: EXAMPLE / DEMO
# ============================================================

def create_demo_world() -> EntityGraph:
    """Create a small demo showing the system."""
    
    world = EntityGraph("Mystery Manor")
    
    # -- Locations --
    
    study = Entity("Study", EntityType.LOCATION)
    study.add_description(
        "Oak-paneled walls rise to a coffered ceiling.",
        FragmentCategory.BASE_DESCRIPTION,
        priority=100
    )
    study.add_description(
        "Dust motes drift in the pale light filtering through heavy curtains.",
        FragmentCategory.ATMOSPHERIC
    )
    study.add_description(
        "The smell of old leather and pipe tobacco lingers.",
        FragmentCategory.SENSORY,
        tags={'smell'}
    )
    study.add_tag('indoor')
    world.register(study)
    
    # -- Items --
    
    desk = Entity("desk", EntityType.CONTAINER)
    desk.add_nouns("desk", "table")
    desk.add_adjectives("oak", "massive", "large")
    desk.add_description(
        "A massive oak desk dominates the room, its surface cluttered with papers.",
        FragmentCategory.BASE_DESCRIPTION
    )
    desk.add_tag('furniture')
    
    # Custom handler for searching
    def search_desk(entity: Entity, context: Dict) -> str:
        if entity.properties.get('searched'):
            return "You've already searched the desk thoroughly."
        entity.properties['searched'] = True
        return "You rifle through the desk drawers, finding old receipts and faded photographs."
    
    desk.on('search', search_desk)
    world.register(desk)
    
    # Hidden key (revealed by searching desk)
    key = Entity("brass key", EntityType.ITEM)
    key.add_nouns("key")
    key.add_adjectives("brass", "small", "tarnished")
    key.state = EntityState.HIDDEN
    key.add_description(
        "A small brass key, tarnished with age.",
        FragmentCategory.BASE_DESCRIPTION
    )
    key.add_fragment(ProseFragment(
        text="Beneath some papers, you find a small brass key.",
        category=FragmentCategory.DISCOVERY,
        one_shot=True
    ))
    key.add_tag('takeable')
    world.register(key)
    
    # -- Links --
    
    # Desk is in study
    world.relate(study, desk, RelationalKind.CONTAINS)
    
    # Key is in desk (hidden)
    world.relate(desk, key, RelationalKind.CONTAINS)
    
    # Searching desk reveals key
    world.logic(
        desk, key, 
        LogicalKind.REVEALS,
        condition=LinkCondition(
            lambda ctx: ctx.get('target', Entity('', EntityType.ITEM)).properties.get('searched', False),
            "Desk must be searched"
        )
    )
    
    return world


if __name__ == "__main__":
    # Demo
    world = create_demo_world()
    print(world)
    print()
    
    study = world.find_by_name("Study")
    desk = world.find_by_name("desk")
    key = world.find_by_name("brass key")
    
    print("=== Study Description ===")
    print(study.get_description())
    print()
    
    print("=== Desk Description ===")
    print(desk.get_description())
    print()
    
    print("=== Key (hidden) ===")
    print(f"Key is hidden: {key.is_hidden}")
    print()
    
    print("=== Search Desk ===")
    processor = ActionProcessor(world)
    result = processor.process(
        Action(verb='search', target_id=desk.id),
        {'current_location': study.id}
    )
    print(result)
    print()
    
    print("=== Key (after search) ===")
    print(f"Key is hidden: {key.is_hidden}")
    print(key.get_description())
