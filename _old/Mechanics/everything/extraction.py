"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  MATTS EXTRACTION - Text Analysis Pipeline                                     ║
║  Layer 3: Depends on primitives, signals                                      ║
║                                                                               ║
║  Scan text → analyze every word → construct structured data.                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
import re
import uuid
import time

from .primitives import WordClass, FragmentCategory, SignalType
from .signals import ObserverBus


# ═══════════════════════════════════════════════════════════════════════════════
#                              EXTRACTED DATA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedWord:
    """A word extracted and classified from text"""
    text: str
    word_class: WordClass = WordClass.UNKNOWN
    position: int = 0
    sentence_idx: int = 0
    paragraph_idx: int = 0
    entity_id: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)
    source_sentence: str = ""


@dataclass
class ExtractedEntity:
    """An entity discovered in text"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: str = "item"  # location, character, item, concept
    adjectives: Set[str] = field(default_factory=set)
    descriptions: List[str] = field(default_factory=list)
    first_mention: int = 0
    mention_count: int = 1
    confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id, 'name': self.name, 'category': self.category,
            'adjectives': list(self.adjectives), 'descriptions': self.descriptions,
            'first_mention': self.first_mention, 'mention_count': self.mention_count,
            'confidence': self.confidence
        }


@dataclass
class ExtractedRelation:
    """A relation between two entities"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    relation_word: str = ""  # in, on, near, etc.
    source_sentence: str = ""
    confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id, 'source_id': self.source_id, 'target_id': self.target_id,
            'relation_word': self.relation_word, 'confidence': self.confidence
        }


@dataclass
class ExtractedFragment:
    """A prose fragment from the source text"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    category: FragmentCategory = FragmentCategory.BASE_DESCRIPTION
    entity_id: Optional[str] = None
    source_position: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id, 'text': self.text, 'category': self.category.value,
            'entity_id': self.entity_id, 'source_position': self.source_position
        }


@dataclass
class ExtractionResult:
    """Complete extraction result"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_text: str = ""
    words: List[ExtractedWord] = field(default_factory=list)
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    fragments: List[ExtractedFragment] = field(default_factory=list)
    extraction_time: float = 0.0
    
    @property
    def word_count(self) -> int:
        return len(self.words)
    
    @property
    def entity_count(self) -> int:
        return len(self.entities)
    
    def summary(self) -> str:
        return (f"Extracted: {len(self.words)} words, {len(self.entities)} entities, "
                f"{len(self.relations)} relations, {len(self.fragments)} fragments")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'entities': [e.to_dict() for e in self.entities],
            'relations': [r.to_dict() for r in self.relations],
            'fragments': [f.to_dict() for f in self.fragments],
            'word_count': self.word_count,
            'extraction_time': self.extraction_time
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                              EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class TextExtractor:
    """
    Extracts structured data from source text.
    
    Philosophy: "We don't author puzzles - we scan text, analyze every word,
    and construct a world from what we find."
    """
    
    # Word indicators
    LOCATION_WORDS = frozenset({
        'room', 'house', 'building', 'cave', 'forest', 'castle', 'tower',
        'garden', 'kitchen', 'hall', 'corridor', 'door', 'gate', 'street',
        'path', 'road', 'bridge', 'chamber', 'cellar', 'attic'
    })
    
    CHARACTER_WORDS = frozenset({
        'man', 'woman', 'person', 'wizard', 'witch', 'guard', 'stranger',
        'king', 'queen', 'knight', 'merchant', 'child', 'boy', 'girl'
    })
    
    ITEM_WORDS = frozenset({
        'key', 'sword', 'book', 'letter', 'ring', 'box', 'chest', 'lamp',
        'bottle', 'potion', 'scroll', 'map', 'coin', 'torch', 'dagger'
    })
    
    SPATIAL_PREPS = frozenset({
        'in', 'on', 'under', 'near', 'beside', 'behind', 'above',
        'below', 'inside', 'outside', 'through', 'between', 'at'
    })
    
    ARTICLES = frozenset({'a', 'an', 'the'})
    PRONOUNS = frozenset({'he', 'she', 'it', 'they', 'him', 'her', 'them'})
    
    def __init__(self, bus: ObserverBus = None):
        self._bus = bus
        self._reset()
    
    def _reset(self):
        """Reset extraction state"""
        self._entities: Dict[str, ExtractedEntity] = {}
        self._name_to_id: Dict[str, str] = {}
        self._last_entity_id: Optional[str] = None  # For pronoun resolution
    
    def extract(self, text: str) -> ExtractionResult:
        """
        Main extraction entry point.
        
        Scan text → extract words → find entities → find relations → 
        extract fragments → resolve references → calculate confidence.
        """
        start = time.time()
        self._reset()
        
        if self._bus:
            self._bus.emit_type(SignalType.EXTRACTION_START)
        
        words = []
        relations = []
        fragments = []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        position = 0
        for para_idx, para in enumerate(paragraphs):
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            
            for sent_idx, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                
                # Extract words from sentence
                sent_words = self._extract_words(sentence, position, sent_idx, para_idx)
                words.extend(sent_words)
                
                # Analyze sentence for entities and relations
                sent_relations = self._analyze_sentence(sent_words, sentence)
                relations.extend(sent_relations)
                
                # Create fragment
                frag = ExtractedFragment(
                    text=sentence.strip(),
                    category=self._classify_fragment(sentence),
                    source_position=position
                )
                fragments.append(frag)
                
                position += len(sentence) + 1
        
        # Post-processing
        self._calculate_confidence()
        self._assign_fragments_to_entities(fragments)
        
        result = ExtractionResult(
            source_text=text,
            words=words,
            entities=list(self._entities.values()),
            relations=relations,
            fragments=fragments,
            extraction_time=time.time() - start
        )
        
        if self._bus:
            self._bus.emit_type(SignalType.EXTRACTION_COMPLETE,
                               data={'entities': result.entity_count, 
                                     'words': result.word_count})
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # Word extraction
    # ─────────────────────────────────────────────────────────────────────────
    
    def _extract_words(self, sentence: str, position: int,
                       sent_idx: int, para_idx: int) -> List[ExtractedWord]:
        """Extract and classify words from a sentence"""
        words = []
        tokens = re.findall(r'\b\w+\b', sentence)
        
        for i, token in enumerate(tokens):
            word = ExtractedWord(
                text=token.lower(),
                position=position + sentence.lower().find(token.lower()),
                sentence_idx=sent_idx,
                paragraph_idx=para_idx,
                source_sentence=sentence,
                word_class=self._classify_word(token, i, tokens)
            )
            words.append(word)
        
        return words
    
    def _classify_word(self, token: str, pos: int, context: List[str]) -> WordClass:
        """Determine grammatical class of word"""
        lower = token.lower()
        
        # Proper noun (capitalized, not at sentence start)
        if token[0].isupper() and pos > 0:
            return WordClass.PROPER_NOUN
        
        # Known categories
        if lower in self.SPATIAL_PREPS:
            return WordClass.PREPOSITION
        if lower in self.ARTICLES:
            return WordClass.ARTICLE
        if lower in self.PRONOUNS:
            return WordClass.PRONOUN
        if lower in self.LOCATION_WORDS | self.CHARACTER_WORDS | self.ITEM_WORDS:
            return WordClass.NOUN
        
        # Morphological hints
        if lower.endswith(('ed', 'ing', 's')) and len(lower) > 3:
            return WordClass.VERB
        if lower.endswith(('ful', 'less', 'ous', 'ive', 'able')):
            return WordClass.ADJECTIVE
        if lower.endswith(('ly',)) and len(lower) > 3:
            return WordClass.ADVERB
        if lower.endswith(('tion', 'ness', 'ment', 'ity')):
            return WordClass.NOUN
        
        return WordClass.UNKNOWN
    
    # ─────────────────────────────────────────────────────────────────────────
    # Sentence analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    def _analyze_sentence(self, words: List[ExtractedWord], 
                          sentence: str) -> List[ExtractedRelation]:
        """Find entities and relations in sentence"""
        relations = []
        
        # Pass 1: Find entities
        for i, word in enumerate(words):
            if word.word_class in (WordClass.NOUN, WordClass.PROPER_NOUN):
                entity = self._get_or_create_entity(word)
                word.entity_id = entity.id
                self._last_entity_id = entity.id
                
                # Collect adjectives before noun
                if i > 0 and words[i-1].word_class == WordClass.ADJECTIVE:
                    entity.adjectives.add(words[i-1].text)
                if i > 1 and words[i-2].word_class == WordClass.ADJECTIVE:
                    entity.adjectives.add(words[i-2].text)
            
            elif word.word_class == WordClass.PRONOUN:
                # Resolve to last mentioned entity
                word.entity_id = self._last_entity_id
        
        # Pass 2: Find relations (pattern: ENTITY PREP ENTITY)
        for i, word in enumerate(words):
            if word.word_class == WordClass.PREPOSITION:
                source_id = self._find_entity_before(words, i)
                target_id = self._find_entity_after(words, i)
                
                if source_id and target_id and source_id != target_id:
                    relations.append(ExtractedRelation(
                        source_id=source_id,
                        target_id=target_id,
                        relation_word=word.text,
                        source_sentence=sentence
                    ))
        
        return relations
    
    def _get_or_create_entity(self, word: ExtractedWord) -> ExtractedEntity:
        """Get existing entity or create new one"""
        name = word.text.lower()
        
        if name in self._name_to_id:
            entity = self._entities[self._name_to_id[name]]
            entity.mention_count += 1
            return entity
        
        # Determine category
        category = "item"
        if name in self.LOCATION_WORDS:
            category = "location"
        elif name in self.CHARACTER_WORDS:
            category = "character"
        elif word.word_class == WordClass.PROPER_NOUN:
            category = "character"
        
        entity = ExtractedEntity(
            name=name,
            category=category,
            first_mention=word.position
        )
        
        self._entities[entity.id] = entity
        self._name_to_id[name] = entity.id
        
        return entity
    
    def _find_entity_before(self, words: List[ExtractedWord], pos: int) -> Optional[str]:
        """Find nearest entity before position"""
        for i in range(pos - 1, max(-1, pos - 5), -1):
            if words[i].entity_id:
                return words[i].entity_id
        return None
    
    def _find_entity_after(self, words: List[ExtractedWord], pos: int) -> Optional[str]:
        """Find nearest entity after position"""
        for i in range(pos + 1, min(len(words), pos + 5)):
            if words[i].entity_id:
                return words[i].entity_id
        return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Fragment classification
    # ─────────────────────────────────────────────────────────────────────────
    
    def _classify_fragment(self, sentence: str) -> FragmentCategory:
        """Determine fragment category"""
        lower = sentence.lower()
        
        if '"' in sentence or "'" in sentence:
            return FragmentCategory.NPC_AMBIENT
        
        if any(w in lower for w in ['walked', 'ran', 'took', 'opened', 'closed', 'moved']):
            return FragmentCategory.STATE_CHANGE
        
        if any(w in lower for w in ['dark', 'cold', 'warm', 'quiet', 'loud', 'smell']):
            return FragmentCategory.ATMOSPHERIC
        
        if any(w in lower for w in ['see', 'hear', 'feel', 'taste', 'touch']):
            return FragmentCategory.SENSORY
        
        if any(w in lower for w in ['once', 'long ago', 'years', 'history', 'ancient']):
            return FragmentCategory.HISTORY
        
        return FragmentCategory.BASE_DESCRIPTION
    
    # ─────────────────────────────────────────────────────────────────────────
    # Post-processing
    # ─────────────────────────────────────────────────────────────────────────
    
    def _calculate_confidence(self):
        """Calculate entity confidence scores"""
        max_mentions = max((e.mention_count for e in self._entities.values()), default=1)
        
        for entity in self._entities.values():
            # Base confidence from mentions
            mention_score = entity.mention_count / max_mentions
            
            # Bonus for proper nouns
            proper_bonus = 0.2 if entity.category == "character" else 0
            
            # Bonus for having adjectives (more specific)
            adj_bonus = min(len(entity.adjectives) * 0.1, 0.2)
            
            entity.confidence = min(0.3 + mention_score * 0.4 + proper_bonus + adj_bonus, 1.0)
    
    def _assign_fragments_to_entities(self, fragments: List[ExtractedFragment]):
        """Try to assign fragments to their primary entities"""
        for frag in fragments:
            # Find most mentioned entity in fragment
            entity_mentions = {}
            for name, eid in self._name_to_id.items():
                if name in frag.text.lower():
                    entity_mentions[eid] = entity_mentions.get(eid, 0) + 1
            
            if entity_mentions:
                frag.entity_id = max(entity_mentions, key=entity_mentions.get)


# ═══════════════════════════════════════════════════════════════════════════════
#                              FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text(text: str, bus: ObserverBus = None) -> ExtractionResult:
    """Extract structured data from text"""
    return TextExtractor(bus).extract(text)


# ═══════════════════════════════════════════════════════════════════════════════
#                              EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'ExtractedWord', 'ExtractedEntity', 'ExtractedRelation', 
    'ExtractedFragment', 'ExtractionResult',
    'TextExtractor', 'extract_text',
]
