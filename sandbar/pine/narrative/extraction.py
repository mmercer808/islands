"""
Pine Text Extraction System
===========================

Extract entities, relations, and fragments from natural language.

The extraction pipeline:
1. Tokenize and classify words
2. Identify entities (nouns, named entities)
3. Detect relations between entities
4. Extract prose fragments

SOURCE: Mechanics/everything/extraction.py (20KB)
Copy the full implementation from that file.

Quick Reference:
----------------
    from . import extract_text, TextExtractor

    result = extract_text('''
        The old lighthouse stands on the rocky cliff.
        Inside, a brass key rests on the dusty table.
    ''')

    for entity in result.entities:
        print(f"Entity: {entity.name} ({entity.entity_type})")

    for relation in result.relations:
        print(f"Relation: {relation.source} --{relation.relation_type}--> {relation.target}")
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import re

from ..core.primitives import WordClass, FragmentCategory


# =============================================================================
#                              DATA CLASSES
# =============================================================================

@dataclass
class ExtractedWord:
    """A word extracted from text with classification."""
    text: str
    word_class: WordClass
    position: int
    original_text: str = ""
    lemma: str = ""


@dataclass
class ExtractedEntity:
    """
    An entity extracted from text.

    Represents a noun phrase that could be a
    location, item, character, or concept.
    """
    name: str
    entity_type: str = "unknown"
    description: str = ""
    source_text: str = ""
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedRelation:
    """
    A relation between two entities.

    Extracted from verb phrases and prepositions
    connecting entities in the text.
    """
    source: str
    target: str
    relation_type: str
    source_text: str = ""
    confidence: float = 1.0


@dataclass
class ExtractedFragment:
    """
    A prose fragment for a specific purpose.

    Categorized by its narrative function:
    - Base description
    - Atmospheric detail
    - State change
    - Interaction text
    """
    text: str
    category: FragmentCategory
    entity_ref: Optional[str] = None


@dataclass
class ExtractionResult:
    """Complete result of text extraction."""
    source_text: str
    words: List[ExtractedWord] = field(default_factory=list)
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    fragments: List[ExtractedFragment] = field(default_factory=list)

    def get_entity(self, name: str) -> Optional[ExtractedEntity]:
        """Get entity by name."""
        name_lower = name.lower()
        for entity in self.entities:
            if entity.name.lower() == name_lower:
                return entity
        return None


# =============================================================================
#                              EXTRACTOR
# =============================================================================

class TextExtractor:
    """
    Extracts entities, relations, and fragments from text.

    Uses a combination of:
    - Pattern matching
    - Simple NLP heuristics
    - Word classification

    For production use, integrate with spaCy or similar.

    TODO: Copy full implementation from Mechanics/everything/extraction.py
    """

    # Simple patterns for entity detection
    LOCATION_PATTERNS = [
        r'\b(?:the\s+)?(\w+)\s+(?:room|hall|chamber|tower|lighthouse|cliff|garden|forest|village|castle|house|building)\b',
        r'\b(?:in|on|at|inside|near)\s+(?:the\s+)?(\w+(?:\s+\w+)?)\b',
    ]

    ITEM_PATTERNS = [
        r'\b(?:a|an|the)\s+(\w+(?:\s+\w+)?)\s+(?:key|book|letter|map|box|chest|lamp|lantern)\b',
        r'\b(\w+\s+)?(?:key|book|letter|map|box|chest|lamp|lantern)\b',
    ]

    RELATION_PATTERNS = [
        (r'(\w+)\s+(?:leads?\s+to|connects?\s+to|opens?)\s+(?:the\s+)?(\w+)', 'leads_to'),
        (r'(\w+)\s+(?:contains?|holds?|has)\s+(?:a|an|the)?\s*(\w+)', 'contains'),
        (r'(\w+)\s+(?:is|lies|sits|rests)\s+(?:on|in|inside)\s+(?:the\s+)?(\w+)', 'inside'),
    ]

    def __init__(self):
        self._word_classes = self._build_word_classes()

    def _build_word_classes(self) -> Dict[str, WordClass]:
        """Build simple word classification dictionary."""
        classes = {}

        # Articles
        for word in ['a', 'an', 'the']:
            classes[word] = WordClass.ARTICLE

        # Prepositions
        for word in ['in', 'on', 'at', 'to', 'from', 'with', 'inside', 'near', 'under', 'over']:
            classes[word] = WordClass.PREPOSITION

        # Common verbs
        for word in ['is', 'are', 'was', 'were', 'has', 'have', 'stands', 'sits', 'lies', 'leads', 'opens', 'contains']:
            classes[word] = WordClass.VERB

        # Common adjectives
        for word in ['old', 'new', 'large', 'small', 'dusty', 'ancient', 'hidden', 'brass', 'wooden', 'stone']:
            classes[word] = WordClass.ADJECTIVE

        return classes

    def extract(self, text: str) -> ExtractionResult:
        """Extract all information from text."""
        result = ExtractionResult(source_text=text)

        # Tokenize
        words = self._tokenize(text)
        result.words = words

        # Extract entities
        result.entities = self._extract_entities(text)

        # Extract relations
        result.relations = self._extract_relations(text)

        # Extract fragments
        result.fragments = self._extract_fragments(text)

        return result

    def _tokenize(self, text: str) -> List[ExtractedWord]:
        """Tokenize text into classified words."""
        words = []
        # Simple word splitting
        pattern = r'\b\w+\b'
        for i, match in enumerate(re.finditer(pattern, text.lower())):
            word_text = match.group()
            word_class = self._word_classes.get(word_text, WordClass.UNKNOWN)
            words.append(ExtractedWord(
                text=word_text,
                word_class=word_class,
                position=i,
                original_text=match.group()
            ))
        return words

    def _extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities from text."""
        entities = []
        text_lower = text.lower()

        # Look for locations
        for pattern in self.LOCATION_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                name = match.group(1) if match.groups() else match.group()
                if name and len(name) > 2:
                    entities.append(ExtractedEntity(
                        name=name.title(),
                        entity_type="location",
                        source_text=match.group()
                    ))

        # Look for items
        for pattern in self.ITEM_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                name = match.group().strip()
                if name and len(name) > 2:
                    entities.append(ExtractedEntity(
                        name=name.title(),
                        entity_type="item",
                        source_text=match.group()
                    ))

        # Deduplicate by name
        seen = set()
        unique = []
        for e in entities:
            if e.name.lower() not in seen:
                seen.add(e.name.lower())
                unique.append(e)

        return unique

    def _extract_relations(self, text: str) -> List[ExtractedRelation]:
        """Extract relations between entities."""
        relations = []
        text_lower = text.lower()

        for pattern, rel_type in self.RELATION_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                if len(match.groups()) >= 2:
                    relations.append(ExtractedRelation(
                        source=match.group(1).title(),
                        target=match.group(2).title(),
                        relation_type=rel_type,
                        source_text=match.group()
                    ))

        return relations

    def _extract_fragments(self, text: str) -> List[ExtractedFragment]:
        """Extract prose fragments."""
        fragments = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Categorize sentence
            category = FragmentCategory.BASE_DESCRIPTION

            if any(word in sentence.lower() for word in ['dusty', 'old', 'ancient', 'quiet', 'dark']):
                category = FragmentCategory.ATMOSPHERIC

            fragments.append(ExtractedFragment(
                text=sentence,
                category=category
            ))

        return fragments


# =============================================================================
#                              FACTORY FUNCTION
# =============================================================================

def extract_text(text: str) -> ExtractionResult:
    """
    Convenience function to extract information from text.

    Args:
        text: The text to analyze

    Returns:
        ExtractionResult with entities, relations, fragments
    """
    extractor = TextExtractor()
    return extractor.extract(text)
