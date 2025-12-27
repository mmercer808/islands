"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║             M A T T S   P R O J E C T   A R C H I V E                         ║
║                                                                               ║
║         Complete System Documentation for Claude Code Transfer                ║
║                                                                               ║
║  "Every word is on trial for its life." — Read Like a Writer                  ║
║                                                                               ║
║  This document contains:                                                      ║
║  1. Project Overview & Philosophy                                             ║
║  2. Text Extraction → White Room Builder                                      ║
║  3. Observer Pattern for State Machine                                        ║
║  4. Unified Traversal & Lookahead System                                      ║
║  5. Complete API Reference                                                    ║
║  6. Integration Patterns                                                      ║
║                                                                               ║
║  Created: December 2024                                                       ║
║  Author: CloudyCadet                                                          ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
                           PROJECT OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

PROJECT NAME: matts (Multi-layered Adaptive Text Transformation System)

PURPOSE: Build AI-powered dynamic narrative systems that can:
- Extract worlds from source text (novels, short stories)
- Construct interactive "White Room" environments
- Support MUD-style gameplay generated from books
- Enable runtime code hot-swapping
- Maintain serializable, persistent context

KEY INSIGHT: We don't author puzzles or sandboxes. We SCAN TEXT, analyze
every word, and CONSTRUCT a world from what we find. The story IS the game.

═══════════════════════════════════════════════════════════════════════════════
                           ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                         SOURCE TEXT                                          │
│  (Novel, Short Story, Screenplay, etc.)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEXT EXTRACTION PIPELINE                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Tokenize    │→ │ Analyze     │→ │ Extract     │→ │ Classify    │         │
│  │ Words       │  │ Syntax      │  │ Entities    │  │ Relations   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHITE ROOM BUILDER                                        │
│  Takes extracted entities/relations and constructs:                          │
│  - Locations (from setting descriptions)                                     │
│  - Characters (from dialogue/actions)                                        │
│  - Items (from mentioned objects)                                            │
│  - Connections (from spatial/temporal/logical relations)                     │
│  - Prose Fragments (from descriptive passages)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENTITY GRAPH                                              │
│  - Nodes: Entities (locations, items, characters, concepts)                  │
│  - Edges: Links (relational, logical, wildcard)                              │
│  - Layers: Same nodes, different edge visibility                             │
│  - State: Conditions, flags, properties                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RUNTIME SYSTEMS                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Traversal   │  │ Lookahead   │  │ Observer    │  │ Prose       │         │
│  │ Wrapper     │  │ Engine      │  │ System      │  │ Compositor  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PLAYER INTERFACE                                          │
│  - Text I/O (MUD-style commands)                                             │
│  - Prose output (composed from fragments)                                    │
│  - State display (inventory, location, etc.)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                     CORE REVELATIONS FROM DESIGN
═══════════════════════════════════════════════════════════════════════════════

1. THE WRAPPER IS THE ITERATOR
   __iter__() returns self. The wrapper travels through the for loop.
   Context persists. Layer switching works mid-iteration.
   Graph hotswapping is possible because the wrapper doesn't own the graph.

2. SAME NODES, DIFFERENT EDGES
   Layers don't change what exists—they change what's visible.
   A character exists in SPATIAL (where they are) and CHARACTER (who they know).
   Switch layers = different traversal possibilities.

3. LOOKAHEAD ≠ MOVEMENT
   Lookahead explores possibility space WITHOUT changing state.
   It finds locked doors, hidden items, near-misses, puzzle chains.
   It's the oracle that knows what's possible.

4. NEAR-MISSES ARE GOLD
   One condition away from something = achievable = perfect hint.
   The lookahead finds these automatically.

5. PROSE IS CONDITIONAL
   Fragments have conditions. Show this if player has key.
   Show this on first visit. Hide this until examined.
   Composition happens at runtime.

6. TEXT IS DATA
   Every word in the source is a potential entity, relation, or prose fragment.
   We don't author content—we extract it.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Dict, List, Set, Optional, Any, Callable, Iterator, Iterable,
    Tuple, Union, Generic, TypeVar, Generator, Type, Protocol
)
from enum import Enum, auto
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import weakref
import copy
import time
import re
import json
import uuid


T = TypeVar('T')
N = TypeVar('N')


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║            SECTION 1: TEXT EXTRACTION & WHITE ROOM BUILDER                ║
# ║                                                                           ║
# ║  Scan text. Analyze every word. Construct a world.                        ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class WordClass(Enum):
    """Grammatical classification of extracted words"""
    NOUN = "noun"
    PROPER_NOUN = "proper_noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PREPOSITION = "preposition"
    ARTICLE = "article"
    PRONOUN = "pronoun"
    CONJUNCTION = "conjunction"
    UNKNOWN = "unknown"


class EntityCategory(Enum):
    """Categories for extracted entities"""
    LOCATION = "location"
    CHARACTER = "character"
    ITEM = "item"
    CONCEPT = "concept"
    ACTION = "action"
    QUALITY = "quality"
    TIME = "time"
    DIRECTION = "direction"


class RelationType(Enum):
    """Types of relations extracted from text"""
    SPATIAL = "spatial"           # in, on, under, near
    TEMPORAL = "temporal"         # before, after, during
    POSSESSIVE = "possessive"     # has, owns, carries
    SOCIAL = "social"             # knows, loves, hates
    CAUSAL = "causal"             # causes, enables, prevents
    DESCRIPTIVE = "descriptive"   # is, appears, seems


@dataclass
class ExtractedWord:
    """
    A single word extracted from source text.
    
    Every word is on trial for its life.
    """
    text: str
    word_class: WordClass = WordClass.UNKNOWN
    position: int = 0  # Position in source
    sentence_idx: int = 0
    paragraph_idx: int = 0
    
    # Semantic information
    entity_category: Optional[EntityCategory] = None
    is_entity_reference: bool = False
    referenced_entity_id: Optional[str] = None
    
    # Context
    surrounding_words: List[str] = field(default_factory=list)
    modifiers: List[str] = field(default_factory=list)  # Adjectives/adverbs
    
    # Source tracking
    source_text: str = ""  # Original sentence/phrase


@dataclass
class ExtractedEntity:
    """
    An entity extracted from text analysis.
    
    Could become: location, character, item, concept.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: EntityCategory = EntityCategory.ITEM
    
    # All words that refer to this entity
    references: List[ExtractedWord] = field(default_factory=list)
    
    # Descriptive information
    adjectives: Set[str] = field(default_factory=set)
    descriptions: List[str] = field(default_factory=list)
    
    # Relations to other entities
    relations: List['ExtractedRelation'] = field(default_factory=list)
    
    # Source tracking
    first_mention_position: int = 0
    mention_count: int = 0
    
    # Confidence
    confidence: float = 0.5


@dataclass
class ExtractedRelation:
    """
    A relation extracted from text between two entities.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str = ""
    target_entity_id: str = ""
    relation_type: RelationType = RelationType.SPATIAL
    
    # The actual relationship
    relation_word: str = ""  # "in", "owns", "near"
    
    # Context
    source_text: str = ""
    confidence: float = 0.5


@dataclass
class ExtractedFragment:
    """
    A prose fragment extracted from source text.
    
    This becomes a ProseFragment in the game world.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    category: str = "description"  # description, dialogue, action, atmosphere
    
    # What entity does this describe?
    entity_id: Optional[str] = None
    
    # When should this appear?
    conditions: List[str] = field(default_factory=list)
    
    # Source tracking
    source_position: int = 0
    original_text: str = ""


class TextExtractor:
    """
    Extracts structured data from source text.
    
    This is the first stage: text → structured data.
    Every word is analyzed for its potential to become
    part of the game world.
    """
    
    # Common patterns
    LOCATION_INDICATORS = {
        'room', 'house', 'building', 'street', 'city', 'forest',
        'cave', 'castle', 'tower', 'garden', 'kitchen', 'bedroom',
        'hall', 'corridor', 'path', 'road', 'bridge', 'door'
    }
    
    CHARACTER_INDICATORS = {
        'man', 'woman', 'person', 'boy', 'girl', 'child', 'king',
        'queen', 'wizard', 'witch', 'guard', 'merchant', 'stranger'
    }
    
    ITEM_INDICATORS = {
        'key', 'sword', 'book', 'letter', 'coin', 'ring', 'box',
        'chest', 'lamp', 'candle', 'bottle', 'potion', 'scroll'
    }
    
    SPATIAL_PREPOSITIONS = {
        'in', 'on', 'under', 'near', 'beside', 'behind', 'above',
        'below', 'inside', 'outside', 'through', 'between'
    }
    
    DIRECTION_WORDS = {
        'north', 'south', 'east', 'west', 'up', 'down', 'left', 'right'
    }
    
    def __init__(self):
        self.words: List[ExtractedWord] = []
        self.entities: Dict[str, ExtractedEntity] = {}
        self.relations: List[ExtractedRelation] = []
        self.fragments: List[ExtractedFragment] = []
        
        # Name resolution
        self._name_to_entity: Dict[str, str] = {}  # name → entity_id
        self._pronoun_stack: List[str] = []  # Recent entity IDs for pronoun resolution
    
    def extract(self, text: str) -> 'ExtractionResult':
        """
        Main extraction method.
        
        Takes raw text, returns structured extraction result.
        """
        # Reset state
        self.words = []
        self.entities = {}
        self.relations = []
        self.fragments = []
        self._name_to_entity = {}
        self._pronoun_stack = []
        
        # Split into paragraphs and sentences
        paragraphs = self._split_paragraphs(text)
        
        position = 0
        for para_idx, paragraph in enumerate(paragraphs):
            sentences = self._split_sentences(paragraph)
            
            for sent_idx, sentence in enumerate(sentences):
                # Extract words from sentence
                sentence_words = self._extract_words(
                    sentence, position, sent_idx, para_idx
                )
                self.words.extend(sentence_words)
                
                # Analyze sentence for entities and relations
                self._analyze_sentence(sentence_words, sentence)
                
                # Extract prose fragment
                fragment = self._extract_fragment(sentence, sent_idx, para_idx)
                if fragment:
                    self.fragments.append(fragment)
                
                position += len(sentence) + 1
        
        # Post-processing
        self._resolve_references()
        self._calculate_confidences()
        
        return ExtractionResult(
            words=self.words,
            entities=list(self.entities.values()),
            relations=self.relations,
            fragments=self.fragments
        )
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        return [p.strip() for p in text.split('\n\n') if p.strip()]
    
    def _split_sentences(self, paragraph: str) -> List[str]:
        """Split paragraph into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_words(self, sentence: str, position: int,
                       sent_idx: int, para_idx: int) -> List[ExtractedWord]:
        """Extract and classify words from a sentence"""
        words = []
        
        # Tokenize
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        original_tokens = re.findall(r'\b\w+\b', sentence)
        
        for i, (token, original) in enumerate(zip(tokens, original_tokens)):
            word = ExtractedWord(
                text=token,
                position=position + sentence.find(original),
                sentence_idx=sent_idx,
                paragraph_idx=para_idx,
                source_text=sentence
            )
            
            # Classify word
            word.word_class = self._classify_word(token, original, i, tokens)
            
            # Check entity category
            word.entity_category = self._categorize_entity(token)
            
            # Get surrounding words for context
            start = max(0, i - 2)
            end = min(len(tokens), i + 3)
            word.surrounding_words = tokens[start:end]
            
            words.append(word)
        
        return words
    
    def _classify_word(self, token: str, original: str,
                       position: int, all_tokens: List[str]) -> WordClass:
        """Classify a word grammatically"""
        # Proper noun (capitalized, not at sentence start)
        if original[0].isupper() and position > 0:
            return WordClass.PROPER_NOUN
        
        # Preposition
        if token in self.SPATIAL_PREPOSITIONS:
            return WordClass.PREPOSITION
        
        # Article
        if token in {'a', 'an', 'the'}:
            return WordClass.ARTICLE
        
        # Pronoun
        if token in {'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers'}:
            return WordClass.PRONOUN
        
        # Common adjective endings
        if token.endswith(('ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ic')):
            return WordClass.ADJECTIVE
        
        # Common verb endings
        if token.endswith(('ed', 'ing', 'ize', 'ify')):
            return WordClass.VERB
        
        # Common noun endings
        if token.endswith(('tion', 'ness', 'ment', 'ity', 'er', 'or')):
            return WordClass.NOUN
        
        # Known entity indicators suggest noun
        if token in self.LOCATION_INDICATORS | self.CHARACTER_INDICATORS | self.ITEM_INDICATORS:
            return WordClass.NOUN
        
        return WordClass.UNKNOWN
    
    def _categorize_entity(self, word: str) -> Optional[EntityCategory]:
        """Determine if word indicates an entity category"""
        if word in self.LOCATION_INDICATORS:
            return EntityCategory.LOCATION
        if word in self.CHARACTER_INDICATORS:
            return EntityCategory.CHARACTER
        if word in self.ITEM_INDICATORS:
            return EntityCategory.ITEM
        if word in self.DIRECTION_WORDS:
            return EntityCategory.DIRECTION
        return None
    
    def _analyze_sentence(self, words: List[ExtractedWord], sentence: str):
        """Analyze a sentence for entities and relations"""
        # Find noun phrases (article + adjectives + noun)
        noun_phrases = self._find_noun_phrases(words)
        
        for phrase in noun_phrases:
            # Get or create entity
            entity = self._get_or_create_entity(phrase)
            
            # Add adjectives
            for word in phrase:
                if word.word_class == WordClass.ADJECTIVE:
                    entity.adjectives.add(word.text)
        
        # Find relations (entity + preposition + entity)
        self._find_relations(words, sentence)
    
    def _find_noun_phrases(self, words: List[ExtractedWord]) -> List[List[ExtractedWord]]:
        """Find noun phrases in word list"""
        phrases = []
        current_phrase = []
        
        for word in words:
            if word.word_class in (WordClass.ARTICLE, WordClass.ADJECTIVE):
                current_phrase.append(word)
            elif word.word_class in (WordClass.NOUN, WordClass.PROPER_NOUN):
                current_phrase.append(word)
                if current_phrase:
                    phrases.append(current_phrase)
                current_phrase = []
            else:
                if current_phrase:
                    # Phrase ended without noun - discard
                    current_phrase = []
        
        return phrases
    
    def _get_or_create_entity(self, phrase: List[ExtractedWord]) -> ExtractedEntity:
        """Get existing entity or create new one"""
        # Find the head noun
        nouns = [w for w in phrase if w.word_class in (WordClass.NOUN, WordClass.PROPER_NOUN)]
        if not nouns:
            return None
        
        head_noun = nouns[-1]  # Last noun is typically the head
        name = head_noun.text
        
        # Check if entity exists
        if name in self._name_to_entity:
            entity_id = self._name_to_entity[name]
            entity = self.entities[entity_id]
            entity.mention_count += 1
            entity.references.append(head_noun)
            return entity
        
        # Create new entity
        entity = ExtractedEntity(
            name=name,
            category=head_noun.entity_category or EntityCategory.ITEM,
            first_mention_position=head_noun.position,
            mention_count=1
        )
        entity.references.append(head_noun)
        
        # Add to tracking
        self.entities[entity.id] = entity
        self._name_to_entity[name] = entity.id
        self._pronoun_stack.append(entity.id)
        
        # Mark word as entity reference
        head_noun.is_entity_reference = True
        head_noun.referenced_entity_id = entity.id
        
        return entity
    
    def _find_relations(self, words: List[ExtractedWord], sentence: str):
        """Find relations between entities in a sentence"""
        # Simple pattern: ENTITY + PREPOSITION + ENTITY
        for i, word in enumerate(words):
            if word.word_class == WordClass.PREPOSITION:
                # Look for entity before
                source_entity_id = self._find_entity_before(words, i)
                # Look for entity after
                target_entity_id = self._find_entity_after(words, i)
                
                if source_entity_id and target_entity_id:
                    relation = ExtractedRelation(
                        source_entity_id=source_entity_id,
                        target_entity_id=target_entity_id,
                        relation_type=RelationType.SPATIAL,
                        relation_word=word.text,
                        source_text=sentence
                    )
                    self.relations.append(relation)
    
    def _find_entity_before(self, words: List[ExtractedWord], position: int) -> Optional[str]:
        """Find nearest entity reference before position"""
        for i in range(position - 1, -1, -1):
            if words[i].is_entity_reference:
                return words[i].referenced_entity_id
        return None
    
    def _find_entity_after(self, words: List[ExtractedWord], position: int) -> Optional[str]:
        """Find nearest entity reference after position"""
        for i in range(position + 1, len(words)):
            if words[i].is_entity_reference:
                return words[i].referenced_entity_id
        return None
    
    def _extract_fragment(self, sentence: str, sent_idx: int,
                         para_idx: int) -> Optional[ExtractedFragment]:
        """Extract a prose fragment from a sentence"""
        # Determine fragment category based on content
        category = "description"
        
        if '"' in sentence or "'" in sentence:
            category = "dialogue"
        elif any(verb in sentence.lower() for verb in ['walked', 'ran', 'jumped', 'opened', 'took']):
            category = "action"
        elif any(word in sentence.lower() for word in ['dark', 'bright', 'cold', 'warm', 'silent', 'loud']):
            category = "atmosphere"
        
        return ExtractedFragment(
            text=sentence,
            category=category,
            source_position=para_idx * 1000 + sent_idx,
            original_text=sentence
        )
    
    def _resolve_references(self):
        """Resolve pronouns and other references"""
        for word in self.words:
            if word.word_class == WordClass.PRONOUN:
                # Simple: assign to most recent entity
                if self._pronoun_stack:
                    word.is_entity_reference = True
                    word.referenced_entity_id = self._pronoun_stack[-1]
    
    def _calculate_confidences(self):
        """Calculate confidence scores for entities"""
        for entity in self.entities.values():
            # More mentions = higher confidence
            entity.confidence = min(1.0, entity.mention_count / 5)
            
            # Named entities (proper nouns) get boost
            for ref in entity.references:
                if ref.word_class == WordClass.PROPER_NOUN:
                    entity.confidence = min(1.0, entity.confidence + 0.2)


@dataclass
class ExtractionResult:
    """Result of text extraction"""
    words: List[ExtractedWord]
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]
    fragments: List[ExtractedFragment]
    
    def summary(self) -> str:
        lines = [
            f"Words extracted: {len(self.words)}",
            f"Entities found: {len(self.entities)}",
            f"Relations found: {len(self.relations)}",
            f"Fragments: {len(self.fragments)}",
            "",
            "Entities:",
        ]
        for e in self.entities:
            lines.append(f"  {e.name} ({e.category.value}) - {e.mention_count} mentions")
        
        lines.append("")
        lines.append("Relations:")
        for r in self.relations:
            source = next((e.name for e in self.entities if e.id == r.source_entity_id), "?")
            target = next((e.name for e in self.entities if e.id == r.target_entity_id), "?")
            lines.append(f"  {source} --[{r.relation_word}]--> {target}")
        
        return "\n".join(lines)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║                    WHITE ROOM BUILDER                                     ║
# ║                                                                           ║
# ║  Takes extraction results and builds an EntityGraph (game world).         ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class EntityType(Enum):
    """Game entity types"""
    LOCATION = auto()
    ITEM = auto()
    CHARACTER = auto()
    CONTAINER = auto()
    DOOR = auto()
    CONCEPT = auto()


class EntityState(Enum):
    """Entity states"""
    NORMAL = auto()
    HIDDEN = auto()
    LOCKED = auto()
    OPEN = auto()
    CLOSED = auto()
    BROKEN = auto()


class LinkType(Enum):
    """Primary link types"""
    RELATIONAL = auto()
    LOGICAL = auto()
    WILDCARD = auto()


class FragmentCategory(Enum):
    """Prose fragment categories"""
    BASE_DESCRIPTION = 10
    ATMOSPHERIC = 20
    STATE_CHANGE = 30
    ITEM_PRESENCE = 40
    NPC_AMBIENT = 50
    SENSORY = 60
    HISTORY = 70
    DISCOVERY = 80


@dataclass
class GameEntity:
    """A game world entity built from extracted data"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entity_type: EntityType = EntityType.ITEM
    state: EntityState = EntityState.NORMAL
    
    # Prose
    fragments: List['GameFragment'] = field(default_factory=list)
    
    # Vocabulary for parser
    nouns: Set[str] = field(default_factory=set)
    adjectives: Set[str] = field(default_factory=set)
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    # Source tracking
    source_entity_id: Optional[str] = None  # Original extraction ID
    confidence: float = 0.5


@dataclass
class GameFragment:
    """A prose fragment for an entity"""
    text: str
    category: FragmentCategory = FragmentCategory.BASE_DESCRIPTION
    priority: int = 50
    conditions: List[Callable] = field(default_factory=list)
    one_shot: bool = False
    shown: bool = False


@dataclass
class GameLink:
    """A link between entities"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    link_type: LinkType = LinkType.RELATIONAL
    kind: str = ""  # "in", "contains", "north_of", etc.
    bidirectional: bool = False


class WhiteRoomBuilder:
    """
    Builds a game world from extracted text data.
    
    The "White Room" is the starting point: a blank space
    that gets populated with entities, links, and prose.
    """
    
    # Map extraction categories to game types
    CATEGORY_TO_TYPE = {
        EntityCategory.LOCATION: EntityType.LOCATION,
        EntityCategory.CHARACTER: EntityType.CHARACTER,
        EntityCategory.ITEM: EntityType.ITEM,
        EntityCategory.CONCEPT: EntityType.CONCEPT,
    }
    
    # Map relation words to link kinds
    RELATION_TO_KIND = {
        'in': 'contains',
        'on': 'supports',
        'under': 'above',
        'near': 'near',
        'beside': 'beside',
        'behind': 'behind',
        'inside': 'contains',
    }
    
    def __init__(self):
        self.entities: Dict[str, GameEntity] = {}
        self.links: List[GameLink] = []
        self.fragments: List[GameFragment] = []
        
        # ID mapping: extraction_id → game_id
        self._id_map: Dict[str, str] = {}
        
        # The White Room itself
        self.white_room: Optional[GameEntity] = None
    
    def build(self, extraction: ExtractionResult) -> 'WhiteRoom':
        """
        Build a game world from extraction result.
        
        Returns a WhiteRoom containing all entities, links, and fragments.
        """
        # Reset state
        self.entities = {}
        self.links = []
        self._id_map = {}
        
        # Create the White Room (origin point)
        self.white_room = GameEntity(
            name="The White Room",
            entity_type=EntityType.LOCATION,
            fragments=[GameFragment(
                text="You stand in a featureless white space. The world is forming around you.",
                category=FragmentCategory.BASE_DESCRIPTION,
                priority=100
            )]
        )
        self.white_room.tags.add('origin')
        self.entities[self.white_room.id] = self.white_room
        
        # Convert extracted entities
        for ext_entity in extraction.entities:
            game_entity = self._convert_entity(ext_entity)
            if game_entity:
                self.entities[game_entity.id] = game_entity
                self._id_map[ext_entity.id] = game_entity.id
        
        # Convert relations to links
        for ext_relation in extraction.relations:
            link = self._convert_relation(ext_relation)
            if link:
                self.links.append(link)
        
        # Assign fragments to entities
        self._assign_fragments(extraction.fragments)
        
        # Connect orphan entities to White Room
        self._connect_orphans()
        
        return WhiteRoom(
            origin=self.white_room,
            entities=self.entities,
            links=self.links,
            source_text_summary=extraction.summary()
        )
    
    def _convert_entity(self, ext_entity: ExtractedEntity) -> Optional[GameEntity]:
        """Convert extracted entity to game entity"""
        # Skip low-confidence entities
        if ext_entity.confidence < 0.3:
            return None
        
        game_type = self.CATEGORY_TO_TYPE.get(
            ext_entity.category,
            EntityType.ITEM
        )
        
        entity = GameEntity(
            name=ext_entity.name.title(),
            entity_type=game_type,
            nouns={ext_entity.name.lower()},
            adjectives=ext_entity.adjectives.copy(),
            source_entity_id=ext_entity.id,
            confidence=ext_entity.confidence
        )
        
        # Add descriptions as fragments
        for desc in ext_entity.descriptions:
            entity.fragments.append(GameFragment(
                text=desc,
                category=FragmentCategory.BASE_DESCRIPTION
            ))
        
        # Add appropriate tags
        if game_type == EntityType.LOCATION:
            entity.tags.add('visitable')
        elif game_type == EntityType.ITEM:
            entity.tags.add('examinable')
            entity.tags.add('takeable')
        elif game_type == EntityType.CHARACTER:
            entity.tags.add('talkable')
        
        return entity
    
    def _convert_relation(self, ext_relation: ExtractedRelation) -> Optional[GameLink]:
        """Convert extracted relation to game link"""
        source_id = self._id_map.get(ext_relation.source_entity_id)
        target_id = self._id_map.get(ext_relation.target_entity_id)
        
        if not source_id or not target_id:
            return None
        
        kind = self.RELATION_TO_KIND.get(ext_relation.relation_word, ext_relation.relation_word)
        
        return GameLink(
            source_id=source_id,
            target_id=target_id,
            link_type=LinkType.RELATIONAL,
            kind=kind,
            bidirectional=kind in ('near', 'beside')
        )
    
    def _assign_fragments(self, ext_fragments: List[ExtractedFragment]):
        """Assign extracted fragments to entities"""
        for ext_frag in ext_fragments:
            category_map = {
                'description': FragmentCategory.BASE_DESCRIPTION,
                'dialogue': FragmentCategory.NPC_AMBIENT,
                'action': FragmentCategory.STATE_CHANGE,
                'atmosphere': FragmentCategory.ATMOSPHERIC,
            }
            
            fragment = GameFragment(
                text=ext_frag.text,
                category=category_map.get(ext_frag.category, FragmentCategory.BASE_DESCRIPTION)
            )
            
            # If fragment has associated entity, add to it
            if ext_frag.entity_id and ext_frag.entity_id in self._id_map:
                game_id = self._id_map[ext_frag.entity_id]
                if game_id in self.entities:
                    self.entities[game_id].fragments.append(fragment)
            else:
                # Add to White Room as general atmosphere
                self.white_room.fragments.append(fragment)
    
    def _connect_orphans(self):
        """Connect entities without links to the White Room"""
        linked_ids = set()
        for link in self.links:
            linked_ids.add(link.source_id)
            linked_ids.add(link.target_id)
        
        for entity_id, entity in self.entities.items():
            if entity_id != self.white_room.id and entity_id not in linked_ids:
                # Connect to White Room
                self.links.append(GameLink(
                    source_id=self.white_room.id,
                    target_id=entity_id,
                    link_type=LinkType.RELATIONAL,
                    kind='contains'
                ))


@dataclass
class WhiteRoom:
    """
    The constructed game world.
    
    Starting from the White Room origin, everything branches out.
    """
    origin: GameEntity
    entities: Dict[str, GameEntity]
    links: List[GameLink]
    source_text_summary: str = ""
    
    def get_entity(self, entity_id: str) -> Optional[GameEntity]:
        return self.entities.get(entity_id)
    
    def get_links_from(self, entity_id: str) -> List[GameLink]:
        return [l for l in self.links if l.source_id == entity_id]
    
    def get_links_to(self, entity_id: str) -> List[GameLink]:
        return [l for l in self.links if l.target_id == entity_id]
    
    def summary(self) -> str:
        lines = [
            "═" * 50,
            "WHITE ROOM CONSTRUCTED",
            "═" * 50,
            f"Origin: {self.origin.name}",
            f"Entities: {len(self.entities)}",
            f"Links: {len(self.links)}",
            "",
            "Entities by type:",
        ]
        
        by_type = defaultdict(list)
        for e in self.entities.values():
            by_type[e.entity_type.name].append(e.name)
        
        for etype, names in sorted(by_type.items()):
            lines.append(f"  {etype}: {', '.join(names)}")
        
        lines.append("")
        lines.append("Links:")
        for link in self.links[:10]:
            source = self.entities.get(link.source_id)
            target = self.entities.get(link.target_id)
            if source and target:
                lines.append(f"  {source.name} --[{link.kind}]--> {target.name}")
        
        if len(self.links) > 10:
            lines.append(f"  ... and {len(self.links) - 10} more")
        
        return "\n".join(lines)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║            SECTION 2: OBSERVER PATTERN FOR STATE MACHINE                  ║
# ║                                                                           ║
# ║  Robust, helpful API observer class for state management.                 ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class ObserverPriority(Enum):
    """Observer execution priority"""
    CRITICAL = 0    # Must execute, blocks until complete
    HIGH = 1        # Important, execute soon
    NORMAL = 2      # Standard priority
    LOW = 3         # Background, execute when convenient
    LAZY = 4        # Fire and forget


class SignalType(Enum):
    """Standard signal types for the state machine"""
    # State transitions
    STATE_ENTER = "state_enter"
    STATE_EXIT = "state_exit"
    STATE_CHANGE = "state_change"
    
    # Entity events
    ENTITY_CREATED = "entity_created"
    ENTITY_MODIFIED = "entity_modified"
    ENTITY_DESTROYED = "entity_destroyed"
    
    # Traversal events
    TRAVERSAL_STEP = "traversal_step"
    TRAVERSAL_START = "traversal_start"
    TRAVERSAL_END = "traversal_end"
    LAYER_SWITCH = "layer_switch"
    
    # Game events
    ACTION_PERFORMED = "action_performed"
    CONDITION_MET = "condition_met"
    CONDITION_FAILED = "condition_failed"
    
    # System events
    TICK = "tick"
    ERROR = "error"
    WARNING = "warning"
    
    # Custom
    CUSTOM = "custom"


@dataclass
class Signal:
    """
    A signal emitted by the state machine.
    
    Carries type, source, data, and metadata.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: SignalType = SignalType.CUSTOM
    source_id: str = ""
    target_id: Optional[str] = None
    
    # Payload
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    timestamp: float = field(default_factory=time.time)
    
    # Response (filled by observers)
    responses: List[Any] = field(default_factory=list)
    handled: bool = False
    
    def add_response(self, response: Any, observer_id: str = None):
        self.responses.append({
            'observer_id': observer_id,
            'response': response,
            'timestamp': time.time()
        })


class Observer(ABC):
    """
    Abstract base class for observers.
    
    Implement handle() to respond to signals.
    """
    
    def __init__(self, observer_id: str = None,
                 priority: ObserverPriority = ObserverPriority.NORMAL):
        self.id = observer_id or str(uuid.uuid4())
        self.priority = priority
        self._active = True
        self._filter: Optional[Callable[[Signal], bool]] = None
    
    @property
    def active(self) -> bool:
        return self._active
    
    def activate(self):
        self._active = True
    
    def deactivate(self):
        self._active = False
    
    def set_filter(self, filter_fn: Callable[[Signal], bool]):
        """Set a filter to only receive matching signals"""
        self._filter = filter_fn
    
    def should_handle(self, signal: Signal) -> bool:
        """Check if this observer should handle the signal"""
        if not self._active:
            return False
        if self._filter and not self._filter(signal):
            return False
        return True
    
    @abstractmethod
    def handle(self, signal: Signal) -> Any:
        """Handle a signal. Override in subclass."""
        pass


class FunctionObserver(Observer):
    """Observer that wraps a function"""
    
    def __init__(self, fn: Callable[[Signal], Any],
                 observer_id: str = None,
                 priority: ObserverPriority = ObserverPriority.NORMAL):
        super().__init__(observer_id, priority)
        self._fn = fn
    
    def handle(self, signal: Signal) -> Any:
        return self._fn(signal)


class StateObserver(Observer):
    """
    Observer specialized for state machine transitions.
    
    Tracks state history and provides transition hooks.
    """
    
    def __init__(self, observer_id: str = None):
        super().__init__(observer_id, ObserverPriority.HIGH)
        self.state_history: List[Tuple[str, float]] = []
        self.current_state: Optional[str] = None
        
        # Hooks
        self._on_enter: Dict[str, List[Callable]] = defaultdict(list)
        self._on_exit: Dict[str, List[Callable]] = defaultdict(list)
        self._on_transition: Dict[Tuple[str, str], List[Callable]] = defaultdict(list)
    
    def on_enter(self, state: str, callback: Callable):
        """Register callback for entering a state"""
        self._on_enter[state].append(callback)
    
    def on_exit(self, state: str, callback: Callable):
        """Register callback for exiting a state"""
        self._on_exit[state].append(callback)
    
    def on_transition(self, from_state: str, to_state: str, callback: Callable):
        """Register callback for specific transition"""
        self._on_transition[(from_state, to_state)].append(callback)
    
    def handle(self, signal: Signal) -> Any:
        if signal.signal_type == SignalType.STATE_CHANGE:
            old_state = signal.data.get('old_state')
            new_state = signal.data.get('new_state')
            
            # Exit callbacks
            if old_state and old_state in self._on_exit:
                for cb in self._on_exit[old_state]:
                    try:
                        cb(signal)
                    except Exception as e:
                        pass
            
            # Transition callbacks
            if (old_state, new_state) in self._on_transition:
                for cb in self._on_transition[(old_state, new_state)]:
                    try:
                        cb(signal)
                    except Exception as e:
                        pass
            
            # Enter callbacks
            if new_state and new_state in self._on_enter:
                for cb in self._on_enter[new_state]:
                    try:
                        cb(signal)
                    except Exception as e:
                        pass
            
            # Update history
            self.state_history.append((new_state, signal.timestamp))
            self.current_state = new_state
            
            return {'transition': f"{old_state} → {new_state}"}


class EntityObserver(Observer):
    """
    Observer specialized for entity lifecycle events.
    
    Tracks entity creation, modification, destruction.
    """
    
    def __init__(self, observer_id: str = None):
        super().__init__(observer_id, ObserverPriority.NORMAL)
        self.entity_log: List[Dict] = []
        self._entity_callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    def watch(self, entity_id: str, callback: Callable):
        """Watch a specific entity"""
        self._entity_callbacks[entity_id].append(callback)
    
    def handle(self, signal: Signal) -> Any:
        if signal.signal_type in (SignalType.ENTITY_CREATED,
                                  SignalType.ENTITY_MODIFIED,
                                  SignalType.ENTITY_DESTROYED):
            entity_id = signal.data.get('entity_id', signal.source_id)
            
            # Log
            self.entity_log.append({
                'type': signal.signal_type.value,
                'entity_id': entity_id,
                'timestamp': signal.timestamp,
                'data': signal.data
            })
            
            # Entity-specific callbacks
            if entity_id in self._entity_callbacks:
                for cb in self._entity_callbacks[entity_id]:
                    try:
                        cb(signal)
                    except Exception:
                        pass
            
            return {'logged': entity_id}


class TraversalObserver(Observer):
    """
    Observer specialized for graph traversal events.
    
    Tracks path, collects visited nodes, monitors layer switches.
    """
    
    def __init__(self, observer_id: str = None):
        super().__init__(observer_id, ObserverPriority.NORMAL)
        self.path: List[str] = []
        self.visited: Set[str] = set()
        self.layer_history: List[Tuple[str, str]] = []  # (old, new)
        
        self._step_callbacks: List[Callable] = []
        self._layer_callbacks: List[Callable] = []
    
    def on_step(self, callback: Callable):
        """Register callback for each traversal step"""
        self._step_callbacks.append(callback)
    
    def on_layer_switch(self, callback: Callable):
        """Register callback for layer switches"""
        self._layer_callbacks.append(callback)
    
    def handle(self, signal: Signal) -> Any:
        if signal.signal_type == SignalType.TRAVERSAL_STEP:
            node_id = signal.data.get('node_id')
            if node_id:
                self.path.append(node_id)
                self.visited.add(node_id)
            
            for cb in self._step_callbacks:
                try:
                    cb(signal)
                except Exception:
                    pass
        
        elif signal.signal_type == SignalType.LAYER_SWITCH:
            old_layer = signal.data.get('old_layer')
            new_layer = signal.data.get('new_layer')
            self.layer_history.append((old_layer, new_layer))
            
            for cb in self._layer_callbacks:
                try:
                    cb(signal)
                except Exception:
                    pass


class ObserverBus:
    """
    Central hub for observer registration and signal dispatch.
    
    This is the robust, helpful API for the state machine.
    """
    
    def __init__(self):
        self._observers: Dict[str, Observer] = {}
        self._type_subscriptions: Dict[SignalType, Set[str]] = defaultdict(set)
        self._signal_history: deque = deque(maxlen=1000)
        self._pending: deque = deque()
        self._processing = False
    
    # ─────────────────────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────────────────────
    
    def register(self, observer: Observer,
                 signal_types: List[SignalType] = None) -> str:
        """
        Register an observer.
        
        Args:
            observer: The observer to register
            signal_types: List of signal types to subscribe to (None = all)
        
        Returns:
            Observer ID
        """
        self._observers[observer.id] = observer
        
        if signal_types:
            for st in signal_types:
                self._type_subscriptions[st].add(observer.id)
        else:
            # Subscribe to all
            for st in SignalType:
                self._type_subscriptions[st].add(observer.id)
        
        return observer.id
    
    def unregister(self, observer_id: str):
        """Remove an observer"""
        if observer_id in self._observers:
            del self._observers[observer_id]
        
        for subscribers in self._type_subscriptions.values():
            subscribers.discard(observer_id)
    
    def on(self, signal_type: SignalType, callback: Callable,
           priority: ObserverPriority = ObserverPriority.NORMAL) -> str:
        """
        Convenience: register a function as observer.
        
        Returns observer ID for later removal.
        """
        observer = FunctionObserver(callback, priority=priority)
        return self.register(observer, [signal_type])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Dispatch
    # ─────────────────────────────────────────────────────────────────────────
    
    def emit(self, signal: Signal):
        """
        Emit a signal to all relevant observers.
        
        Handles priority ordering and error isolation.
        """
        self._pending.append(signal)
        
        if not self._processing:
            self._process_pending()
    
    def emit_type(self, signal_type: SignalType,
                  source_id: str = "",
                  data: Dict[str, Any] = None,
                  **kwargs) -> Signal:
        """
        Convenience: emit a signal by type.
        
        Returns the signal for inspection.
        """
        signal = Signal(
            signal_type=signal_type,
            source_id=source_id,
            data=data or {},
            **kwargs
        )
        self.emit(signal)
        return signal
    
    def _process_pending(self):
        """Process pending signals"""
        self._processing = True
        
        while self._pending:
            signal = self._pending.popleft()
            self._dispatch(signal)
            self._signal_history.append(signal)
        
        self._processing = False
    
    def _dispatch(self, signal: Signal):
        """Dispatch a single signal"""
        # Get subscribed observer IDs
        subscriber_ids = self._type_subscriptions.get(signal.signal_type, set())
        
        # Get observers and sort by priority
        observers = []
        for obs_id in subscriber_ids:
            if obs_id in self._observers:
                observer = self._observers[obs_id]
                if observer.should_handle(signal):
                    observers.append(observer)
        
        observers.sort(key=lambda o: o.priority.value)
        
        # Dispatch to each observer
        for observer in observers:
            try:
                result = observer.handle(signal)
                if result is not None:
                    signal.add_response(result, observer.id)
                signal.handled = True
            except Exception as e:
                # Emit error signal (but avoid infinite loop)
                if signal.signal_type != SignalType.ERROR:
                    self.emit_type(
                        SignalType.ERROR,
                        source_id=observer.id,
                        data={'error': str(e), 'signal_id': signal.id}
                    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_observer(self, observer_id: str) -> Optional[Observer]:
        return self._observers.get(observer_id)
    
    def get_history(self, signal_type: SignalType = None,
                    limit: int = 100) -> List[Signal]:
        """Get signal history, optionally filtered by type"""
        signals = list(self._signal_history)
        
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        
        return signals[-limit:]
    
    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    
    def create_state_observer(self) -> Tuple[StateObserver, str]:
        """Create and register a state observer"""
        observer = StateObserver()
        obs_id = self.register(observer, [
            SignalType.STATE_ENTER,
            SignalType.STATE_EXIT,
            SignalType.STATE_CHANGE
        ])
        return observer, obs_id
    
    def create_entity_observer(self) -> Tuple[EntityObserver, str]:
        """Create and register an entity observer"""
        observer = EntityObserver()
        obs_id = self.register(observer, [
            SignalType.ENTITY_CREATED,
            SignalType.ENTITY_MODIFIED,
            SignalType.ENTITY_DESTROYED
        ])
        return observer, obs_id
    
    def create_traversal_observer(self) -> Tuple[TraversalObserver, str]:
        """Create and register a traversal observer"""
        observer = TraversalObserver()
        obs_id = self.register(observer, [
            SignalType.TRAVERSAL_STEP,
            SignalType.TRAVERSAL_START,
            SignalType.TRAVERSAL_END,
            SignalType.LAYER_SWITCH
        ])
        return observer, obs_id


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║            SECTION 3: LAYER SYSTEM (from traversal design)                ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class Layer(Enum):
    """Named edge sets"""
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CHARACTER = "character"
    THEMATIC = "thematic"
    NARRATIVE = "narrative"
    LOGICAL = "logical"


@dataclass
class LayerConfig:
    """Configuration for a layer"""
    name: str
    edge_types: Set[str] = field(default_factory=set)
    filter_fn: Optional[Callable] = None
    switch_condition: Optional[Callable] = None
    priority: int = 0


class LayerRegistry:
    """Central registry of layers"""
    
    def __init__(self):
        self._layers: Dict[str, LayerConfig] = {}
        self._active: str = "spatial"
    
    def register(self, config: LayerConfig):
        self._layers[config.name] = config
    
    def get(self, name: str) -> Optional[LayerConfig]:
        return self._layers.get(name)
    
    def set_active(self, name: str):
        if name in self._layers:
            self._active = name
    
    @property
    def active(self) -> Optional[LayerConfig]:
        return self._layers.get(self._active)
    
    @property
    def active_name(self) -> str:
        return self._active


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║            SECTION 4: TRAVERSAL CONTEXT                                   ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class TraversalContext:
    """State that travels with the traversal wrapper"""
    current_node: Any = None
    previous_node: Any = None
    path: List[Any] = field(default_factory=list)
    visited: Set[str] = field(default_factory=set)
    depth: int = 0
    
    current_layer: str = "spatial"
    layer_stack: List[str] = field(default_factory=list)
    
    flags: Dict[str, bool] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    inventory: List[str] = field(default_factory=list)
    buffer: Dict[str, Any] = field(default_factory=dict)
    
    graph_ref: Any = None
    
    def push_layer(self, layer: str):
        self.layer_stack.append(self.current_layer)
        self.current_layer = layer
    
    def pop_layer(self) -> str:
        if self.layer_stack:
            self.current_layer = self.layer_stack.pop()
        return self.current_layer
    
    def mark_visited(self, node: Any):
        node_id = node.id if hasattr(node, 'id') else str(node)
        self.visited.add(node_id)
    
    def is_visited(self, node: Any) -> bool:
        node_id = node.id if hasattr(node, 'id') else str(node)
        return node_id in self.visited
    
    def clone(self) -> 'TraversalContext':
        return TraversalContext(
            current_node=self.current_node,
            previous_node=self.previous_node,
            path=self.path.copy(),
            visited=self.visited.copy(),
            depth=self.depth,
            current_layer=self.current_layer,
            layer_stack=self.layer_stack.copy(),
            flags=self.flags.copy(),
            counters=self.counters.copy(),
            inventory=self.inventory.copy(),
            buffer=self.buffer.copy(),
            graph_ref=self.graph_ref
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║            SECTION 5: TRAVERSAL WRAPPER                                   ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class TraversalWrapper(Generic[T]):
    """
    Smart iterator that carries context through traversal.
    
    __iter__() returns self - the wrapper IS the iterator.
    """
    
    def __init__(self,
                 source: Iterable[T] = None,
                 context: TraversalContext = None,
                 layer_registry: LayerRegistry = None):
        self._source = source
        self._iterator: Optional[Iterator[T]] = None
        self.context = context or TraversalContext()
        self.layers = layer_registry or LayerRegistry()
        
        self._on_step: List[Callable] = []
        self._current: Optional[T] = None
        self._step_count = 0
    
    def __iter__(self) -> 'TraversalWrapper[T]':
        """Return self - the wrapper travels with iteration"""
        if self._source is not None:
            self._iterator = iter(self._source)
        return self
    
    def __next__(self) -> T:
        if self._iterator is None:
            raise StopIteration
        
        item = next(self._iterator)
        
        self.context.previous_node = self._current
        self._current = item
        self.context.current_node = item
        self.context.path.append(item)
        self.context.mark_visited(item)
        self.context.depth = len(self.context.path)
        self._step_count += 1
        
        for cb in self._on_step:
            try:
                cb(self, item)
            except Exception:
                pass
        
        return item
    
    @property
    def current(self) -> Optional[T]:
        return self._current
    
    @property
    def depth(self) -> int:
        return self.context.depth
    
    @property
    def layer(self) -> str:
        return self.context.current_layer
    
    def switch_layer(self, name: str, push: bool = False):
        old = self.context.current_layer
        if push:
            self.context.push_layer(name)
        else:
            self.context.current_layer = name
        self.layers.set_active(name)
    
    def pop_layer(self) -> str:
        name = self.context.pop_layer()
        self.layers.set_active(name)
        return name
    
    def swap_graph(self, new_graph: Any, reset: bool = False):
        self.context.graph_ref = new_graph
        if reset:
            self._current = None
            self.context.path = []
            self.context.visited = set()
    
    def on_step(self, callback: Callable) -> 'TraversalWrapper[T]':
        self._on_step.append(callback)
        return self
    
    def branch(self) -> 'TraversalWrapper[T]':
        branched = TraversalWrapper(
            source=None,
            context=self.context.clone(),
            layer_registry=self.layers
        )
        branched._current = self._current
        return branched


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║            SECTION 6: LOOKAHEAD ENGINE                                    ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class PossibilityType(Enum):
    """Categories of discoverable possibilities"""
    REACHABLE_LOCATION = auto()
    BLOCKED_PATH = auto()
    LOCKED_DOOR = auto()
    VISIBLE_ITEM = auto()
    HIDDEN_ITEM = auto()
    TAKEABLE_ITEM = auto()
    REQUIRED_ITEM = auto()
    UNLOCKABLE = auto()
    REVEALABLE = auto()
    NEAR_MISS = auto()


@dataclass
class Possibility:
    """A single discovered possibility"""
    possibility_type: PossibilityType
    entity_id: str
    entity_name: str
    distance: int = 0
    conditions_met: List[str] = field(default_factory=list)
    conditions_unmet: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    hint_text: Optional[str] = None
    path: List[str] = field(default_factory=list)
    
    def is_blocked(self) -> bool:
        return len(self.conditions_unmet) > 0
    
    def is_near_miss(self, threshold: int = 1) -> bool:
        return 0 < len(self.conditions_unmet) <= threshold


@dataclass
class LookaheadResult:
    """Complete lookahead result"""
    origin_id: str
    origin_name: str
    max_depth: int
    all_possibilities: List[Possibility] = field(default_factory=list)
    by_type: Dict[PossibilityType, List[Possibility]] = field(
        default_factory=lambda: defaultdict(list)
    )
    
    def add(self, p: Possibility):
        self.all_possibilities.append(p)
        self.by_type[p.possibility_type].append(p)
    
    def get_hints(self, max_hints: int = 3) -> List[str]:
        hints = []
        for p in self.all_possibilities:
            if p.is_near_miss() and p.hint_text:
                hints.append(p.hint_text)
            if len(hints) >= max_hints:
                break
        return hints


class LookaheadEngine:
    """Pre-traverses the graph to discover possibilities"""
    
    def __init__(self, graph: Any = None):
        self.graph = graph
    
    def lookahead(self, from_entity: Any, context: Dict = None,
                  max_depth: int = 3) -> LookaheadResult:
        """Perform lookahead from an entity"""
        context = context or {}
        
        # Get entity info
        if hasattr(from_entity, 'id'):
            start_id = from_entity.id
            start_name = getattr(from_entity, 'name', start_id)
        else:
            start_id = str(from_entity)
            start_name = start_id
        
        result = LookaheadResult(
            origin_id=start_id,
            origin_name=start_name,
            max_depth=max_depth
        )
        
        # BFS exploration
        visited = set()
        frontier = deque([(start_id, 0, [start_id])])
        
        while frontier:
            current_id, depth, path = frontier.popleft()
            
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if depth > max_depth:
                continue
            
            # Get entity
            entity = self._get_entity(current_id)
            if entity is None:
                continue
            
            # Analyze
            for p in self._analyze(entity, depth, path):
                result.add(p)
            
            # Add neighbors to frontier
            for neighbor_id in self._get_neighbors(current_id):
                if neighbor_id not in visited:
                    frontier.append((neighbor_id, depth + 1, path + [neighbor_id]))
        
        return result
    
    def _get_entity(self, entity_id: str) -> Any:
        if self.graph is None:
            return None
        if hasattr(self.graph, 'entities'):
            return self.graph.entities.get(entity_id)
        if hasattr(self.graph, 'get_entity'):
            return self.graph.get_entity(entity_id)
        return None
    
    def _get_neighbors(self, entity_id: str) -> List[str]:
        if self.graph is None:
            return []
        if hasattr(self.graph, 'get_links_from'):
            links = self.graph.get_links_from(entity_id)
            return [l.target_id for l in links]
        return []
    
    def _analyze(self, entity: Any, depth: int, path: List[str]) -> List[Possibility]:
        possibilities = []
        
        entity_id = entity.id if hasattr(entity, 'id') else str(entity)
        entity_name = getattr(entity, 'name', entity_id)
        entity_type = getattr(entity, 'entity_type', None)
        entity_state = getattr(entity, 'state', None)
        
        if entity_type == EntityType.LOCATION:
            possibilities.append(Possibility(
                possibility_type=PossibilityType.REACHABLE_LOCATION,
                entity_id=entity_id,
                entity_name=entity_name,
                distance=depth,
                path=path.copy()
            ))
        
        elif entity_type == EntityType.ITEM:
            if entity_state == EntityState.HIDDEN:
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.HIDDEN_ITEM,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    distance=depth,
                    path=path.copy(),
                    hint_text="There might be something hidden here..."
                ))
            else:
                possibilities.append(Possibility(
                    possibility_type=PossibilityType.VISIBLE_ITEM,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    distance=depth,
                    path=path.copy()
                ))
        
        return possibilities


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║            SECTION 7: CONVENIENCE FUNCTIONS                               ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def extract_text(text: str) -> ExtractionResult:
    """Extract structured data from text"""
    extractor = TextExtractor()
    return extractor.extract(text)


def build_white_room(extraction: ExtractionResult) -> WhiteRoom:
    """Build a White Room from extraction result"""
    builder = WhiteRoomBuilder()
    return builder.build(extraction)


def text_to_world(text: str) -> WhiteRoom:
    """Complete pipeline: text → extraction → White Room"""
    extraction = extract_text(text)
    return build_white_room(extraction)


def smart_iter(source: Iterable[T], context: TraversalContext = None,
               layer_registry: LayerRegistry = None) -> TraversalWrapper[T]:
    """Create a smart iterator"""
    return TraversalWrapper(source, context, layer_registry)


def lookahead_from(graph: Any, entity: Any, max_depth: int = 3) -> LookaheadResult:
    """One-shot lookahead"""
    engine = LookaheadEngine(graph)
    return engine.lookahead(entity, max_depth=max_depth)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                                                                           ║
# ║            SECTION 8: DEMONSTRATION                                       ║
# ║                                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def demo():
    """Demonstrate the complete system"""
    print("=" * 70)
    print("MATTS PROJECT DEMONSTRATION")
    print("Text → Extraction → White Room → Traversal → Lookahead")
    print("=" * 70)
    
    # Sample short story
    story = """
    The old house stood at the end of the lane. Inside, the study contained 
    a massive oak desk covered with papers. A brass key lay hidden under 
    the papers. The key would unlock the door to the garden.
    
    Sarah walked through the corridor toward the study. She had been 
    searching for answers since morning. The portrait on the wall seemed 
    to watch her pass.
    """
    
    print("\n--- SOURCE TEXT ---")
    print(story)
    
    print("\n--- EXTRACTION ---")
    extraction = extract_text(story)
    print(extraction.summary())
    
    print("\n--- WHITE ROOM CONSTRUCTION ---")
    world = build_white_room(extraction)
    print(world.summary())
    
    print("\n--- OBSERVER SYSTEM ---")
    bus = ObserverBus()
    state_obs, _ = bus.create_state_observer()
    entity_obs, _ = bus.create_entity_observer()
    
    # Register hooks
    state_obs.on_enter('exploring', lambda s: print(f"  → Entered exploring state"))
    
    # Emit some signals
    bus.emit_type(SignalType.STATE_CHANGE, data={
        'old_state': 'idle',
        'new_state': 'exploring'
    })
    
    bus.emit_type(SignalType.ENTITY_MODIFIED, data={
        'entity_id': world.origin.id,
        'modification': 'visited'
    })
    
    print(f"  State history: {state_obs.state_history}")
    print(f"  Entity log entries: {len(entity_obs.entity_log)}")
    
    print("\n--- TRAVERSAL ---")
    entities = list(world.entities.values())
    wrapper = smart_iter(entities)
    wrapper.on_step(lambda w, e: print(f"  Step {w._step_count}: {e.name}"))
    
    for entity in wrapper:
        pass  # Callbacks handle output
    
    print(f"  Visited: {len(wrapper.context.visited)} entities")
    
    print("\n--- LOOKAHEAD ---")
    engine = LookaheadEngine(world)
    result = engine.lookahead(world.origin, max_depth=2)
    
    print(f"  Possibilities found: {len(result.all_possibilities)}")
    for hint in result.get_hints():
        print(f"  💡 {hint}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
#                              MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Extraction
    'TextExtractor',
    'ExtractionResult',
    'ExtractedWord',
    'ExtractedEntity',
    'ExtractedRelation',
    'ExtractedFragment',
    
    # White Room
    'WhiteRoomBuilder',
    'WhiteRoom',
    'GameEntity',
    'GameFragment',
    'GameLink',
    
    # Observer
    'Observer',
    'ObserverBus',
    'StateObserver',
    'EntityObserver',
    'TraversalObserver',
    'Signal',
    'SignalType',
    
    # Traversal
    'TraversalWrapper',
    'TraversalContext',
    'LayerRegistry',
    'LayerConfig',
    
    # Lookahead
    'LookaheadEngine',
    'LookaheadResult',
    'Possibility',
    'PossibilityType',
    
    # Enums
    'EntityType',
    'EntityState',
    'LinkType',
    'FragmentCategory',
    'WordClass',
    'EntityCategory',
    'RelationType',
    
    # Convenience
    'extract_text',
    'build_white_room',
    'text_to_world',
    'smart_iter',
    'lookahead_from',
]


if __name__ == "__main__":
    demo()
