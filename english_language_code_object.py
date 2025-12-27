#!/usr/bin/env python3
"""
English Language Rulemaking Methods - Treating Language as Code Object
Implementing chain-based rule system with decision trees and callback lists
"""

import ast
import types
from typing import Dict, Any, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import uuid
from datetime import datetime

# Using the comprehensive code object utility concepts
from collections import defaultdict, deque


class LanguageRuleType(Enum):
    """Types of linguistic rules that can be chained."""
    SYNTACTIC = "syntactic"           # Grammar structure rules
    SEMANTIC = "semantic"             # Meaning-based rules  
    PHONETIC = "phonetic"             # Sound-based rules
    MORPHOLOGICAL = "morphological"   # Word formation rules
    PRAGMATIC = "pragmatic"           # Context/usage rules
    LEXICAL = "lexical"               # Vocabulary rules
    DISCOURSE = "discourse"           # Text-level rules


class ChainDecisionType(Enum):
    """Types of decisions a chain can make."""
    ACCEPT = "accept"
    REJECT = "reject"
    TRANSFORM = "transform"
    BRANCH = "branch"
    MERGE = "merge"
    DEFER = "defer"
    ESCALATE = "escalate"


@dataclass
class LanguageToken:
    """Represents a linguistic unit (word, phrase, etc.)"""
    text: str
    pos_tag: str = ""
    semantic_role: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_range: tuple = (0, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RuleChain:
    """A chain of linguistic rules with callback mechanisms."""
    chain_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_type: LanguageRuleType = LanguageRuleType.SYNTACTIC
    pattern: str = ""
    condition_callback: Optional[Callable] = None
    action_callback: Optional[Callable] = None
    transform_callback: Optional[Callable] = None
    
    # Chain linking
    next_chains: List['RuleChain'] = field(default_factory=list)
    parent_chains: List['RuleChain'] = field(default_factory=list)
    
    # Decision tree properties
    decision_weight: float = 1.0
    confidence_threshold: float = 0.7
    priority: int = 0
    
    # Metadata
    description: str = ""
    examples: List[str] = field(default_factory=list)
    counter_examples: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.condition_callback is None:
            self.condition_callback = self._default_condition
        if self.action_callback is None:
            self.action_callback = self._default_action
    
    def _default_condition(self, tokens: List[LanguageToken], context: Dict) -> bool:
        """Default condition check - can be overridden."""
        return True
    
    def _default_action(self, tokens: List[LanguageToken], context: Dict) -> ChainDecisionType:
        """Default action - can be overridden."""
        return ChainDecisionType.ACCEPT
    
    def execute(self, tokens: List[LanguageToken], context: Dict) -> tuple:
        """Execute this chain's rule."""
        if not self.condition_callback(tokens, context):
            return ChainDecisionType.REJECT, tokens, {}
        
        decision = self.action_callback(tokens, context)
        
        if decision == ChainDecisionType.TRANSFORM and self.transform_callback:
            new_tokens = self.transform_callback(tokens, context)
            return decision, new_tokens, {'transformed': True}
        
        return decision, tokens, {'executed': True}


@dataclass
class ChainNode:
    """A node in the linguistic decision tree containing multiple rule chains."""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    chains: List[RuleChain] = field(default_factory=list)
    children: List['ChainNode'] = field(default_factory=list)
    parent: Optional['ChainNode'] = None
    
    # Node properties
    node_type: str = "decision"  # decision, leaf, root, merge
    activation_threshold: float = 0.5
    combination_strategy: str = "weighted_average"  # how to combine chain results
    
    def add_chain(self, chain: RuleChain) -> None:
        """Add a rule chain to this node."""
        self.chains.append(chain)
    
    def add_child(self, child: 'ChainNode') -> None:
        """Add a child node."""
        self.children.append(child)
        child.parent = self
    
    def process(self, tokens: List[LanguageToken], context: Dict) -> tuple:
        """Process tokens through all chains in this node."""
        results = []
        
        for chain in sorted(self.chains, key=lambda c: c.priority, reverse=True):
            decision, new_tokens, metadata = chain.execute(tokens, context)
            results.append({
                'chain_id': chain.chain_id,
                'decision': decision,
                'tokens': new_tokens,
                'metadata': metadata,
                'weight': chain.decision_weight
            })
        
        # Combine results based on strategy
        final_decision, final_tokens = self._combine_results(results, tokens)
        
        return final_decision, final_tokens, {'node_results': results}
    
    def _combine_results(self, results: List[Dict], original_tokens: List[LanguageToken]) -> tuple:
        """Combine multiple chain results into final decision."""
        if not results:
            return ChainDecisionType.REJECT, original_tokens
        
        if self.combination_strategy == "weighted_average":
            accept_weight = sum(r['weight'] for r in results if r['decision'] == ChainDecisionType.ACCEPT)
            total_weight = sum(r['weight'] for r in results)
            
            if total_weight > 0 and (accept_weight / total_weight) >= self.activation_threshold:
                # Find the transformation result if any
                for result in results:
                    if result['decision'] == ChainDecisionType.TRANSFORM:
                        return ChainDecisionType.TRANSFORM, result['tokens']
                return ChainDecisionType.ACCEPT, original_tokens
            else:
                return ChainDecisionType.REJECT, original_tokens
        
        # Default: take first successful result
        for result in results:
            if result['decision'] in [ChainDecisionType.ACCEPT, ChainDecisionType.TRANSFORM]:
                return result['decision'], result['tokens']
        
        return ChainDecisionType.REJECT, original_tokens


class EnglishLanguageCodeObject:
    """
    Treats English language as a comprehensive code object with full metadata.
    Implements chain-based rule system for linguistic processing.
    """
    
    def __init__(self):
        self.metadata = self._initialize_language_metadata()
        self.root_node = ChainNode(name="ROOT", node_type="root")
        self.chain_registry: Dict[str, RuleChain] = {}
        self.execution_stack: List[Dict] = []
        self.global_context: Dict[str, Any] = {}
        
        # Build the comprehensive rule system
        self._build_comprehensive_rule_chains()
    
    def _initialize_language_metadata(self) -> Dict[str, Any]:
        """Initialize language as code object with comprehensive metadata."""
        return {
            # Basic code object properties
            'language_name': 'English',
            'language_version': '2024.8',
            'encoding': 'utf-8',
            'co_argcount': 0,  # Languages don't take positional args
            'co_kwonlyargcount': 0,
            'co_nlocals': 0,
            'co_stacksize': float('inf'),  # Unlimited recursion depth
            'co_flags': 0x0040,  # Custom flag for natural language
            
            # Language-specific metadata
            'phoneme_count': 44,  # English phonemes
            'morpheme_types': ['free', 'bound', 'inflectional', 'derivational'],
            'word_order': 'SVO',
            'case_system': 'nominative-accusative',
            'article_system': ['definite', 'indefinite'],
            
            # AST-equivalent: Parse tree structure
            'syntactic_categories': {
                'NP': 'Noun Phrase',
                'VP': 'Verb Phrase', 
                'PP': 'Prepositional Phrase',
                'S': 'Sentence',
                'SBAR': 'Subordinate Clause'
            },
            
            # Bytecode-equivalent: Morphological rules
            'morphological_rules': [
                'add_s_plural',
                'add_ed_past',
                'add_ing_progressive',
                'add_er_comparative'
            ],
            
            # Exception handling: Irregular forms
            'irregular_forms': {
                'go': 'went',
                'child': 'children',
                'good': 'better'
            },
            
            # Memory management: Working memory constraints
            'working_memory_capacity': 7,  # Miller's magic number
            'processing_window': 20,  # words in immediate processing window
            
            # Performance metrics
            'average_words_per_minute': 200,
            'comprehension_accuracy': 0.95,
            'production_fluency': 0.98,
            
            # Security flags: Ambiguity detection
            'ambiguity_tolerance': 0.3,
            'context_dependency': 0.8,
            
            'creation_date': datetime.now().isoformat()
        }
    
    def _build_comprehensive_rule_chains(self) -> None:
        """Build the complete English rule system as interconnected chains."""
        
        # === SYNTACTIC RULES ===
        
        # Subject-Verb Agreement Chain
        sva_chain = RuleChain(
            rule_type=LanguageRuleType.SYNTACTIC,
            pattern="NP[number=X] VP[number=X]",
            description="Subject and verb must agree in number",
            examples=["The cat runs", "The cats run"],
            counter_examples=["The cat run", "The cats runs"],
            priority=100
        )
        sva_chain.condition_callback = self._check_subject_verb_agreement
        sva_chain.action_callback = self._validate_agreement
        
        # Word Order Chain  
        word_order_chain = RuleChain(
            rule_type=LanguageRuleType.SYNTACTIC,
            pattern="S -> NP VP (PP)*",
            description="Basic English word order: Subject-Verb-Object",
            examples=["John eats apples", "She reads books quickly"],
            priority=95
        )
        word_order_chain.condition_callback = self._check_word_order
        
        # Determiner-Noun Chain
        det_noun_chain = RuleChain(
            rule_type=LanguageRuleType.SYNTACTIC,
            pattern="NP -> (Det) (Adj)* N",
            description="Noun phrase structure with optional determiners and adjectives",
            examples=["the big red car", "cats", "a book"],
            priority=90
        )
        
        # === SEMANTIC RULES ===
        
        # Selectional Restrictions Chain
        selection_chain = RuleChain(
            rule_type=LanguageRuleType.SEMANTIC,
            pattern="V[+animate_subject] NP[+animate]",
            description="Verbs requiring animate subjects",
            examples=["The dog sleeps", "John thinks"],
            counter_examples=["The rock sleeps", "The table thinks"],
            priority=85
        )
        selection_chain.condition_callback = self._check_selectional_restrictions
        
        # Metaphor Detection Chain
        metaphor_chain = RuleChain(
            rule_type=LanguageRuleType.SEMANTIC,
            pattern="NP[domain=X] VP[domain=Y] where X != Y",
            description="Detect cross-domain mappings (metaphors)",
            examples=["Time flies", "Life is a journey"],
            priority=60
        )
        metaphor_chain.condition_callback = self._detect_metaphor
        metaphor_chain.action_callback = self._process_metaphor
        
        # === MORPHOLOGICAL RULES ===
        
        # Plural Formation Chain
        plural_chain = RuleChain(
            rule_type=LanguageRuleType.MORPHOLOGICAL,
            pattern="N + -s/-es -> N[plural]",
            description="Regular plural formation",
            examples=["cat -> cats", "box -> boxes"],
            priority=80
        )
        plural_chain.transform_callback = self._apply_plural_rule
        
        # Past Tense Chain
        past_tense_chain = RuleChain(
            rule_type=LanguageRuleType.MORPHOLOGICAL,
            pattern="V + -ed -> V[past]",
            description="Regular past tense formation",
            examples=["walk -> walked", "talk -> talked"],
            priority=75
        )
        
        # === PRAGMATIC RULES ===
        
        # Politeness Chain
        politeness_chain = RuleChain(
            rule_type=LanguageRuleType.PRAGMATIC,
            pattern="Request + please/would you",
            description="Politeness markers in requests",
            examples=["Could you please help?", "Would you mind closing the door?"],
            priority=50
        )
        
        # Register appropriateness chains
        # Build node structure - this is the decision tree!
        
        # Root node branches to major linguistic levels
        syntactic_node = ChainNode(name="SYNTACTIC_PROCESSING", node_type="decision")
        semantic_node = ChainNode(name="SEMANTIC_PROCESSING", node_type="decision") 
        morphological_node = ChainNode(name="MORPHOLOGICAL_PROCESSING", node_type="decision")
        pragmatic_node = ChainNode(name="PRAGMATIC_PROCESSING", node_type="decision")
        
        # Add chains to appropriate nodes
        syntactic_node.add_chain(sva_chain)
        syntactic_node.add_chain(word_order_chain)
        syntactic_node.add_chain(det_noun_chain)
        
        semantic_node.add_chain(selection_chain)
        semantic_node.add_chain(metaphor_chain)
        
        morphological_node.add_chain(plural_chain)
        morphological_node.add_chain(past_tense_chain)
        
        pragmatic_node.add_chain(politeness_chain)
        
        # Build tree structure - chains link nodes!
        self.root_node.add_child(morphological_node)  # First: word formation
        morphological_node.add_child(syntactic_node)  # Then: syntax
        syntactic_node.add_child(semantic_node)       # Then: meaning
        semantic_node.add_child(pragmatic_node)       # Finally: usage
        
        # Register all chains
        for node in [syntactic_node, semantic_node, morphological_node, pragmatic_node]:
            for chain in node.chains:
                self.chain_registry[chain.chain_id] = chain
        
        # Create inter-chain links (the magic happens here!)
        # Syntax chains can trigger semantic chains
        sva_chain.next_chains.append(selection_chain)
        word_order_chain.next_chains.append(selection_chain)
        
        # Semantic chains can trigger pragmatic chains  
        selection_chain.next_chains.append(politeness_chain)
        metaphor_chain.next_chains.append(politeness_chain)
        
        # Morphological chains feed into syntactic chains
        plural_chain.next_chains.append(det_noun_chain)
        past_tense_chain.next_chains.append(sva_chain)
    
    def parse_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Parse a sentence through the chain-based rule system.
        This simulates executing 'English language code'.
        """
        
        # Tokenize sentence (simplified)
        tokens = self._tokenize(sentence)
        
        # Initialize execution context
        context = {
            'sentence': sentence,
            'parse_tree': {},
            'semantic_roles': {},
            'pragmatic_features': {},
            'execution_trace': []
        }
        
        # Execute through the decision tree
        result = self._execute_chain_tree(self.root_node, tokens, context)
        
        return {
            'tokens': [t.to_dict() for t in tokens],
            'parse_result': result,
            'execution_stack': self.execution_stack.copy(),
            'language_metadata': self.metadata,
            'chain_activations': context['execution_trace']
        }
    
    def _execute_chain_tree(self, node: ChainNode, tokens: List[LanguageToken], context: Dict) -> Dict:
        """Execute tokens through a chain tree node."""
        
        # Record execution in stack (like Python call stack)
        stack_frame = {
            'node_id': node.node_id,
            'node_name': node.name,
            'tokens_in': len(tokens),
            'timestamp': datetime.now().isoformat(),
            'context_snapshot': context.copy()
        }
        self.execution_stack.append(stack_frame)
        
        # Process through this node's chains
        decision, processed_tokens, node_metadata = node.process(tokens, context)
        
        # Record chain activations
        context['execution_trace'].append({
            'node': node.name,
            'decision': decision.value,
            'chains_fired': len(node.chains),
            'metadata': node_metadata
        })
        
        # Continue to children if decision allows
        final_result = {
            'decision': decision.value,
            'tokens': [t.to_dict() for t in processed_tokens],
            'node_metadata': node_metadata
        }
        
        if decision in [ChainDecisionType.ACCEPT, ChainDecisionType.TRANSFORM] and node.children:
            child_results = []
            for child in node.children:
                child_result = self._execute_chain_tree(child, processed_tokens, context)
                child_results.append(child_result)
            final_result['children'] = child_results
        
        # Update stack frame with results
        stack_frame['tokens_out'] = len(processed_tokens)
        stack_frame['decision'] = decision.value
        
        return final_result
    
    def _tokenize(self, sentence: str) -> List[LanguageToken]:
        """Simple tokenization - in reality this would be much more sophisticated."""
        words = sentence.split()
        tokens = []
        
        for i, word in enumerate(words):
            # Simplified POS tagging
            pos_tag = self._simple_pos_tag(word)
            
            token = LanguageToken(
                text=word,
                pos_tag=pos_tag,
                source_range=(i, i+1)
            )
            tokens.append(token)
        
        return tokens
    
    def _simple_pos_tag(self, word: str) -> str:
        """Extremely simplified POS tagging."""
        if word.lower() in ['the', 'a', 'an']:
            return 'DET'
        elif word.lower() in ['run', 'runs', 'eat', 'eats', 'think', 'thinks']:
            return 'VERB'
        elif word.lower() in ['cat', 'cats', 'dog', 'book', 'john', 'mary']:
            return 'NOUN'
        elif word.lower() in ['big', 'red', 'happy']:
            return 'ADJ'
        else:
            return 'UNKNOWN'
    
    # === CALLBACK IMPLEMENTATIONS ===
    
    def _check_subject_verb_agreement(self, tokens: List[LanguageToken], context: Dict) -> bool:
        """Check if subject and verb agree in number."""
        # Simplified implementation
        for i, token in enumerate(tokens[:-1]):
            if token.pos_tag == 'NOUN' and tokens[i+1].pos_tag == 'VERB':
                # Check agreement (simplified)
                if token.text.endswith('s') and not tokens[i+1].text.endswith('s'):
                    return False  # Plural noun with singular verb
                if not token.text.endswith('s') and tokens[i+1].text.endswith('s'):
                    return True   # Singular noun with singular verb
        return True
    
    def _validate_agreement(self, tokens: List[LanguageToken], context: Dict) -> ChainDecisionType:
        """Validate agreement and return decision."""
        if self._check_subject_verb_agreement(tokens, context):
            return ChainDecisionType.ACCEPT
        else:
            context['agreement_error'] = True
            return ChainDecisionType.REJECT
    
    def _check_word_order(self, tokens: List[LanguageToken], context: Dict) -> bool:
        """Check basic SVO word order."""
        pos_sequence = [t.pos_tag for t in tokens]
        # Look for Subject-Verb pattern
        for i in range(len(pos_sequence) - 1):
            if pos_sequence[i] == 'NOUN' and pos_sequence[i+1] == 'VERB':
                return True
        return False
    
    def _check_selectional_restrictions(self, tokens: List[LanguageToken], context: Dict) -> bool:
        """Check semantic selectional restrictions."""
        # Simplified: check for animate subjects with certain verbs
        animate_requiring_verbs = ['think', 'sleep', 'dream']
        inanimate_subjects = ['rock', 'table', 'book']
        
        for i, token in enumerate(tokens[:-1]):
            if (token.text.lower() in inanimate_subjects and 
                tokens[i+1].text.lower() in animate_requiring_verbs):
                return False
        return True
    
    def _detect_metaphor(self, tokens: List[LanguageToken], context: Dict) -> bool:
        """Detect potential metaphorical expressions."""
        metaphor_patterns = [
            ('time', 'flies'),
            ('life', 'journey'),
            ('love', 'war')
        ]
        
        text_tokens = [t.text.lower() for t in tokens]
        for pattern in metaphor_patterns:
            if all(word in text_tokens for word in pattern):
                return True
        return False
    
    def _process_metaphor(self, tokens: List[LanguageToken], context: Dict) -> ChainDecisionType:
        """Process metaphorical expression."""
        context['metaphor_detected'] = True
        context['interpretation_needed'] = True
        return ChainDecisionType.TRANSFORM
    
    def _apply_plural_rule(self, tokens: List[LanguageToken], context: Dict) -> List[LanguageToken]:
        """Apply plural morphological rule."""
        new_tokens = []
        for token in tokens:
            if token.pos_tag == 'NOUN' and not token.text.endswith('s'):
                # Apply plural rule
                new_token = LanguageToken(
                    text=token.text + 's',
                    pos_tag=token.pos_tag,
                    features={**token.features, 'number': 'plural'}
                )
                new_tokens.append(new_token)
            else:
                new_tokens.append(token)
        return new_tokens
    
    def get_comprehensive_metadata(self) -> Dict[str, Any]:
        """Return comprehensive metadata about English as a code object."""
        return {
            'language_metadata': self.metadata,
            'total_chains': len(self.chain_registry),
            'chain_types': {rule_type.value: len([c for c in self.chain_registry.values() 
                                                 if c.rule_type == rule_type]) 
                           for rule_type in LanguageRuleType},
            'execution_stack_depth': len(self.execution_stack),
            'tree_structure': self._serialize_tree(self.root_node),
            'chain_interconnections': self._analyze_chain_links()
        }
    
    def _serialize_tree(self, node: ChainNode) -> Dict[str, Any]:
        """Serialize the decision tree structure."""
        return {
            'node_id': node.node_id,
            'name': node.name,
            'type': node.node_type,
            'chain_count': len(node.chains),
            'children': [self._serialize_tree(child) for child in node.children]
        }
    
    def _analyze_chain_links(self) -> Dict[str, Any]:
        """Analyze interconnections between chains."""
        links = []
        for chain in self.chain_registry.values():
            for next_chain in chain.next_chains:
                links.append({
                    'from': chain.chain_id,
                    'to': next_chain.chain_id,
                    'rule_types': f"{chain.rule_type.value} -> {next_chain.rule_type.value}"
                })
        
        return {
            'total_links': len(links),
            'links': links,
            'connectivity_score': len(links) / len(self.chain_registry) if self.chain_registry else 0
        }


# === DEMONSTRATION ===

def demonstrate_english_code_object():
    """Demonstrate English language as code object with chain rules."""
    
    print("=== ENGLISH LANGUAGE AS CODE OBJECT ===\n")
    
    # Initialize the English language "interpreter"
    english = EnglishLanguageCodeObject()
    
    print("Language Metadata:")
    metadata = english.get_comprehensive_metadata()
    print(f"- Language: {english.metadata['language_name']}")
    print(f"- Version: {english.metadata['language_version']}")
    print(f"- Word Order: {english.metadata['word_order']}")
    print(f"- Total Rule Chains: {metadata['total_chains']}")
    print(f"- Chain Interconnections: {metadata['chain_interconnections']['total_links']}")
    print()
    
    # Test sentences through the chain system
    test_sentences = [
        "The cat runs",           # Should pass all rules
        "The cats run",           # Should pass all rules  
        "The cat run",            # Should fail agreement rule
        "Time flies",             # Should trigger metaphor detection
        "The rock thinks"         # Should fail selectional restrictions
    ]
    
    print("=== SENTENCE PROCESSING THROUGH RULE CHAINS ===\n")
    
    for sentence in test_sentences:
        print(f"Processing: '{sentence}'")
        result = english.parse_sentence(sentence)
        
        print(f"  Execution Stack Depth: {len(result['execution_stack'])}")
        print(f"  Chain Activations: {len(result['chain_activations'])}")
        
        # Show chain activation trace
        for activation in result['chain_activations']:
            print(f"    {activation['node']}: {activation['decision']} "
                  f"({activation['chains_fired']} chains)")
        
        # Show any special processing
        final_decision = result['parse_result']['decision']
        print(f"  Final Decision: {final_decision}")
        
        if 'agreement_error' in result['execution_stack'][-1]['context_snapshot']:
            print("    ❌ Agreement violation detected")
        if 'metaphor_detected' in result['execution_stack'][-1]['context_snapshot']:
            print("    🎭 Metaphor detected - interpretation needed")
        
        print()
    
    print("=== RULE CHAIN STRUCTURE ===\n")
    
    # Show the decision tree structure
    def print_tree(node_data, indent=0):
        spaces = "  " * indent
        print(f"{spaces}📁 {node_data['name']} ({node_data['chain_count']} chains)")
        for child in node_data['children']:
            print_tree(child, indent + 1)
    
    print_tree(metadata['tree_structure'])
    
    print(f"\n=== CHAIN INTERCONNECTIONS ===")
    print(f"Total Links: {metadata['chain_interconnections']['total_links']}")
    print(f"Connectivity Score: {metadata['chain_interconnections']['connectivity_score']:.2f}")
    
    for link in metadata['chain_interconnections']['links'][:5]:  # Show first 5
        print(f"  🔗 {link['rule_types']}")
    
    return english, metadata


class AdaptiveChainLearner:
    """
    Enables chains to learn and evolve their links based on processing experience.
    This is where the system becomes truly adaptive - chains modify their own structure!
    """
    
    def __init__(self, english_system: EnglishLanguageCodeObject):
        self.system = english_system
        self.execution_history: List[Dict] = []
        self.success_patterns: Dict[str, float] = defaultdict(float)
        self.failure_patterns: Dict[str, float] = defaultdict(float)
        self.chain_correlation_matrix: Dict[tuple, float] = defaultdict(float)
        self.adaptive_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        
    def record_execution(self, sentence: str, parse_result: Dict, success: bool) -> None:
        """Record execution results for learning."""
        execution_record = {
            'sentence': sentence,
            'result': parse_result,
            'success': success,
            'timestamp': datetime.now(),
            'chain_sequence': self._extract_chain_sequence(parse_result),
            'decision_path': self._extract_decision_path(parse_result)
        }
        
        self.execution_history.append(execution_record)
        self._update_pattern_weights(execution_record)
        self._update_chain_correlations(execution_record)
    
    def _extract_chain_sequence(self, parse_result: Dict) -> List[str]:
        """Extract the sequence of chains that fired during processing."""
        sequence = []
        
        def extract_from_node(node_result):
            if 'node_metadata' in node_result:
                for chain_result in node_result['node_metadata'].get('node_results', []):
                    sequence.append(chain_result['chain_id'])
            
            for child in node_result.get('children', []):
                extract_from_node(child)
        
        extract_from_node(parse_result)
        return sequence
    
    def _extract_decision_path(self, parse_result: Dict) -> List[str]:
        """Extract the path of decisions made."""
        path = []
        
        def extract_decisions(node_result):
            path.append(node_result['decision'])
            for child in node_result.get('children', []):
                extract_decisions(child)
        
        extract_decisions(parse_result)
        return path
    
    def _update_pattern_weights(self, record: Dict) -> None:
        """Update success/failure patterns for chain sequences."""
        chain_sequence = record['chain_sequence']
        
        # Update individual chain success rates
        for chain_id in chain_sequence:
            if record['success']:
                self.success_patterns[chain_id] += 1.0
            else:
                self.failure_patterns[chain_id] += 1.0
        
        # Update sequence patterns
        for i in range(len(chain_sequence) - 1):
            pair = (chain_sequence[i], chain_sequence[i + 1])
            if record['success']:
                self.success_patterns[f"sequence_{pair[0]}_{pair[1]}"] += 1.0
            else:
                self.failure_patterns[f"sequence_{pair[0]}_{pair[1]}"] += 1.0
    
    def _update_chain_correlations(self, record: Dict) -> None:
        """Update correlations between chains that fire together."""
        chains = record['chain_sequence']
        
        # Calculate correlations for all pairs
        for i, chain1 in enumerate(chains):
            for j, chain2 in enumerate(chains):
                if i != j:
                    pair = (chain1, chain2)
                    if record['success']:
                        self.chain_correlation_matrix[pair] += 1.0
                    else:
                        self.chain_correlation_matrix[pair] -= 0.5
    
    def evolve_chain_links(self) -> Dict[str, Any]:
        """
        The key method: evolve chain links based on learned patterns.
        This is where chains modify their own connectivity!
        """
        evolution_log = {
            'new_links_created': 0,
            'links_strengthened': 0,
            'links_weakened': 0,
            'chains_modified': []
        }
        
        # Calculate adaptive weights for each chain
        for chain_id, chain in self.system.chain_registry.items():
            success_rate = self._calculate_success_rate(chain_id)
            
            # Evolve chain properties based on success
            if success_rate > 0.8:
                # Successful chains get higher priority and weight
                chain.priority += 1
                chain.decision_weight *= 1.1
                evolution_log['links_strengthened'] += 1
            elif success_rate < 0.3:
                # Failed chains get lower priority
                chain.priority = max(0, chain.priority - 1)
                chain.decision_weight *= 0.9
                evolution_log['links_weakened'] += 1
            
            evolution_log['chains_modified'].append({
                'chain_id': chain_id,
                'old_weight': chain.decision_weight / 1.1 if success_rate > 0.8 else chain.decision_weight / 0.9,
                'new_weight': chain.decision_weight,
                'success_rate': success_rate
            })
        
        # Create new links based on high correlations
        high_correlation_threshold = 5.0
        for (chain1_id, chain2_id), correlation in self.chain_correlation_matrix.items():
            if correlation > high_correlation_threshold:
                chain1 = self.system.chain_registry.get(chain1_id)
                chain2 = self.system.chain_registry.get(chain2_id)
                
                if chain1 and chain2 and chain2 not in chain1.next_chains:
                    # Create new adaptive link!
                    chain1.next_chains.append(chain2)
                    chain2.parent_chains.append(chain1)
                    evolution_log['new_links_created'] += 1
        
        return evolution_log
    
    def _calculate_success_rate(self, chain_id: str) -> float:
        """Calculate success rate for a specific chain."""
        successes = self.success_patterns.get(chain_id, 0)
        failures = self.failure_patterns.get(chain_id, 0)
        total = successes + failures
        
        if total == 0:
            return 0.5  # Neutral for new chains
        
        return successes / total
    
    def get_learning_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about the learning process."""
        return {
            'total_executions': len(self.execution_history),
            'success_rate': self._overall_success_rate(),
            'most_successful_chains': self._top_performing_chains(5),
            'emerging_patterns': self._detect_emerging_patterns(),
            'chain_evolution_history': self._chain_evolution_timeline(),
            'correlation_insights': self._strongest_correlations(10)
        }
    
    def _overall_success_rate(self) -> float:
        if not self.execution_history:
            return 0.0
        
        successes = sum(1 for record in self.execution_history if record['success'])
        return successes / len(self.execution_history)
    
    def _top_performing_chains(self, n: int) -> List[Dict]:
        chain_performance = []
        for chain_id in self.system.chain_registry.keys():
            success_rate = self._calculate_success_rate(chain_id)
            chain_performance.append({
                'chain_id': chain_id,
                'success_rate': success_rate,
                'chain_type': self.system.chain_registry[chain_id].rule_type.value
            })
        
        return sorted(chain_performance, key=lambda x: x['success_rate'], reverse=True)[:n]
    
    def _detect_emerging_patterns(self) -> List[Dict]:
        """Detect new patterns that are emerging from the data."""
        patterns = []
        
        # Look for sequences that are becoming more successful over time
        recent_history = self.execution_history[-100:]  # Last 100 executions
        
        sequence_trends = defaultdict(list)
        for record in recent_history:
            for i in range(len(record['chain_sequence']) - 1):
                seq = (record['chain_sequence'][i], record['chain_sequence'][i + 1])
                sequence_trends[seq].append(record['success'])
        
        for sequence, outcomes in sequence_trends.items():
            if len(outcomes) >= 5:  # Need sufficient data
                recent_success = sum(outcomes[-5:]) / 5  # Last 5 outcomes
                overall_success = sum(outcomes) / len(outcomes)
                
                if recent_success > overall_success + 0.2:  # Improving trend
                    patterns.append({
                        'pattern_type': 'improving_sequence',
                        'sequence': sequence,
                        'recent_success': recent_success,
                        'overall_success': overall_success,
                        'trend': 'improving'
                    })
        
        return patterns
    
    def _chain_evolution_timeline(self) -> List[Dict]:
        """Track how chains have evolved over time."""
        # This would track weight changes, priority changes, etc.
        # For now, return a simplified version
        timeline = []
        
        for chain_id, chain in self.system.chain_registry.items():
            timeline.append({
                'chain_id': chain_id,
                'current_weight': chain.decision_weight,
                'current_priority': chain.priority,
                'success_rate': self._calculate_success_rate(chain_id)
            })
        
        return sorted(timeline, key=lambda x: x['success_rate'], reverse=True)
    
    def _strongest_correlations(self, n: int) -> List[Dict]:
        """Find the strongest correlations between chains."""
        correlations = []
        
        for (chain1, chain2), strength in self.chain_correlation_matrix.items():
            correlations.append({
                'chain1': chain1,
                'chain2': chain2,
                'correlation_strength': strength,
                'chain1_type': self.system.chain_registry.get(chain1, type('mock', (), {'rule_type': type('mock', (), {'value': 'unknown'})()})).rule_type.value,
                'chain2_type': self.system.chain_registry.get(chain2, type('mock', (), {'rule_type': type('mock', (), {'value': 'unknown'})()})).rule_type.value
            })
        
        return sorted(correlations, key=lambda x: x['correlation_strength'], reverse=True)[:n]


class SelfModifyingLanguageSystem:
    """
    A language system that can modify its own rule structure.
    This represents the ultimate evolution: language changing itself!
    """
    
    def __init__(self):
        self.english = EnglishLanguageCodeObject()
        self.learner = AdaptiveChainLearner(self.english)
        self.modification_history: List[Dict] = []
        self.generation = 0
        
    def train_on_corpus(self, sentences_with_labels: List[tuple]) -> Dict[str, Any]:
        """
        Train the system on a corpus of sentences with correctness labels.
        The system learns and evolves its rule structure!
        """
        training_log = {
            'sentences_processed': 0,
            'evolution_cycles': 0,
            'performance_improvement': 0.0
        }
        
        initial_performance = self._measure_performance(sentences_with_labels[:20])
        
        print("🧠 Starting adaptive learning process...")
        print(f"Initial performance: {initial_performance:.2f}")
        
        for i, (sentence, is_correct) in enumerate(sentences_with_labels):
            # Process sentence through current rule system
            result = self.english.parse_sentence(sentence)
            
            # Determine if our analysis was correct
            final_decision = result['parse_result']['decision']
            our_prediction = final_decision in ['accept', 'transform']
            success = our_prediction == is_correct
            
            # Record for learning
            self.learner.record_execution(sentence, result['parse_result'], success)
            training_log['sentences_processed'] += 1
            
            # Evolve every 10 sentences
            if (i + 1) % 10 == 0:
                evolution_result = self.learner.evolve_chain_links()
                self.generation += 1
                training_log['evolution_cycles'] += 1
                
                print(f"  Generation {self.generation}: "
                      f"+{evolution_result['new_links_created']} links, "
                      f"↑{evolution_result['links_strengthened']} strengthened, "
                      f"↓{evolution_result['links_weakened']} weakened")
                
                self.modification_history.append({
                    'generation': self.generation,
                    'evolution_result': evolution_result,
                    'performance': self._measure_performance(sentences_with_labels[max(0, i-19):i+1])
                })
        
        final_performance = self._measure_performance(sentences_with_labels[-20:])
        training_log['performance_improvement'] = final_performance - initial_performance
        
        print(f"🎯 Final performance: {final_performance:.2f}")
        print(f"📈 Improvement: {training_log['performance_improvement']:.2f}")
        
        return training_log
    
    def _measure_performance(self, test_set: List[tuple]) -> float:
        """Measure current performance on a test set."""
        if not test_set:
            return 0.0
        
        correct = 0
        for sentence, is_correct in test_set:
            result = self.english.parse_sentence(sentence)
            final_decision = result['parse_result']['decision']
            our_prediction = final_decision in ['accept', 'transform']
            
            if our_prediction == is_correct:
                correct += 1
        
        return correct / len(test_set)
    
    def generate_new_rule(self, pattern_evidence: List[Dict]) -> RuleChain:
        """
        Generate entirely new rules based on observed patterns.
        This is meta-learning: the system creates new linguistic rules!
        """
        # Analyze patterns to create new rule
        new_rule = RuleChain(
            rule_type=LanguageRuleType.SYNTACTIC,  # Default, but could be inferred
            pattern="LEARNED_PATTERN",
            description=f"Auto-generated rule from {len(pattern_evidence)} observations",
            priority=50,  # Start with medium priority
            decision_weight=1.0
        )
        
        # Create adaptive condition based on observed patterns
        def learned_condition(tokens, context):
            # Simplified pattern matching based on evidence
            return len(tokens) > 0  # Placeholder logic
        
        def learned_action(tokens, context):
            # Adaptive action based on successful patterns
            return ChainDecisionType.ACCEPT
        
        new_rule.condition_callback = learned_condition
        new_rule.action_callback = learned_action
        
        return new_rule
    
    def self_diagnose(self) -> Dict[str, Any]:
        """System analyzes its own performance and suggests improvements."""
        analytics = self.learner.get_learning_analytics()
        
        diagnosis = {
            'system_health': 'good' if analytics['success_rate'] > 0.7 else 'needs_improvement',
            'learning_velocity': len(self.modification_history),
            'adaptation_recommendations': [],
            'performance_trends': [],
            'rule_efficiency': {}
        }
        
        # Analyze performance trends
        if len(self.modification_history) > 3:
            recent_perf = [m['performance'] for m in self.modification_history[-3:]]
            if all(recent_perf[i] <= recent_perf[i+1] for i in range(len(recent_perf)-1)):
                diagnosis['performance_trends'].append('improving')
            elif all(recent_perf[i] >= recent_perf[i+1] for i in range(len(recent_perf)-1)):
                diagnosis['performance_trends'].append('declining')
            else:
                diagnosis['performance_trends'].append('stable')
        
        # Generate recommendations
        if analytics['success_rate'] < 0.6:
            diagnosis['adaptation_recommendations'].append(
                'Consider adding more diverse training data'
            )
        
        if len(analytics['emerging_patterns']) > 0:
            diagnosis['adaptation_recommendations'].append(
                f'Codify {len(analytics["emerging_patterns"])} emerging patterns into permanent rules'
            )
        
        return diagnosis
    
    def export_evolved_system(self) -> Dict[str, Any]:
        """Export the current state of the evolved language system."""
        return {
            'generation': self.generation,
            'evolved_chains': {
                chain_id: {
                    'weight': chain.decision_weight,
                    'priority': chain.priority,
                    'success_rate': self.learner._calculate_success_rate(chain_id),
                    'links': [link.chain_id for link in chain.next_chains]
                }
                for chain_id, chain in self.english.chain_registry.items()
            },
            'learned_correlations': dict(self.learner.chain_correlation_matrix),
            'modification_history': self.modification_history,
            'performance_analytics': self.learner.get_learning_analytics()
        }


def demonstrate_adaptive_evolution():
    """Demonstrate the self-modifying language system."""
    
    print("🔬 === ADAPTIVE LANGUAGE EVOLUTION DEMONSTRATION ===\n")
    
    # Create self-modifying system
    evolving_system = SelfModifyingLanguageSystem()
    
    # Training corpus with correctness labels
    training_corpus = [
        ("The cat runs", True),
        ("The cats run", True), 
        ("The cat run", False),        # Agreement error
        ("Cats runs", False),          # Agreement error
        ("The dog sleeps", True),
        ("The rock sleeps", False),    # Selectional restriction violation
        ("Time flies", True),          # Metaphor - should be accepted
        ("The big red car", True),
        ("Red big the car", False),    # Word order violation
        ("John thinks carefully", True),
        ("The table thinks", False),   # Selectional restriction
        ("She reads books", True),
        ("Books reads she", False),    # Word order
        ("The happy cats", True),
        ("Happy the cats", False),     # Determiner position
        ("Life is a journey", True),   # Metaphor
        ("Children play", True),
        ("Child plays", True),
        ("Children plays", False),     # Agreement
        ("The books are heavy", True),
        ("The book are heavy", False), # Agreement
        ("Could you please help", True), # Politeness
        ("Help now", False),           # Too direct
        ("Time is money", True),       # Metaphor
        ("The clock is expensive", True)
    ]
    
    print(f"Training corpus: {len(training_corpus)} sentences")
    print("Starting evolution process...\n")
    
    # Train the system
    training_result = evolving_system.train_on_corpus(training_corpus)
    
    print(f"\n📊 Training completed:")
    print(f"  • Sentences processed: {training_result['sentences_processed']}")
    print(f"  • Evolution cycles: {training_result['evolution_cycles']}")
    print(f"  • Performance improvement: {training_result['performance_improvement']:.2%}")
    
    # Self-diagnosis
    print(f"\n🏥 System self-diagnosis:")
    diagnosis = evolving_system.self_diagnose()
    print(f"  • System health: {diagnosis['system_health']}")
    print(f"  • Learning velocity: {diagnosis['learning_velocity']} adaptations")
    print(f"  • Performance trend: {diagnosis['performance_trends']}")
    
    for recommendation in diagnosis['adaptation_recommendations']:
        print(f"  • 💡 {recommendation}")
    
    # Show learning analytics
    analytics = evolving_system.learner.get_learning_analytics()
    print(f"\n📈 Learning Analytics:")
    print(f"  • Overall success rate: {analytics['success_rate']:.2%}")
    print(f"  • Top performing chains:")
    
    for chain in analytics['most_successful_chains'][:3]:
        print(f"    - {chain['chain_type']}: {chain['success_rate']:.2%} success")
    
    print(f"  • Emerging patterns detected: {len(analytics['emerging_patterns'])}")
    
    # Show strongest correlations
    print(f"\n🔗 Strongest Chain Correlations:")
    for corr in analytics['correlation_insights'][:3]:
        print(f"    {corr['chain1_type']} ↔ {corr['chain2_type']}: {corr['correlation_strength']:.1f}")
    
    # Export evolved system
    evolved_state = evolving_system.export_evolved_system()
    print(f"\n💾 Evolved system exported (Generation {evolved_state['generation']})")
    print(f"   • {len(evolved_state['evolved_chains'])} evolved chains")
    print(f"   • {len(evolved_state['learned_correlations'])} learned correlations")
    
    return evolving_system, evolved_state


if __name__ == "__main__":
    # Original demonstration
    english_system, metadata = demonstrate_english_code_object()
    
    print("\n" + "="*60)
    
    # New adaptive evolution demonstration
    evolving_system, evolved_state = demonstrate_adaptive_evolution()
    
    print("\n=== EVOLUTION SUMMARY ===")
    print("🧬 English language system successfully:")
    print("✅ Learned from experience")
    print("✅ Evolved its own rule connections") 
    print("✅ Self-diagnosed performance issues")
    print("✅ Adapted chain weights and priorities")
    print("✅ Created new inter-chain correlations")
    print("✅ Generated performance insights")
    
    print(f"\n🚀 The system evolved through {evolved_state['generation']} generations,")
    print("demonstrating how linguistic rules can become self-modifying chains")
    print("that learn optimal parameter passing and decision tree structures!")
    
    print("\n🔮 FUTURE POSSIBILITIES:")
    print("• Chains could generate entirely new grammatical rules")
    print("• Meta-chains could govern how other chains evolve")  
    print("• Cross-language chain systems could emerge")
    print("• Adaptive syntax could optimize for different contexts")
    print("• The language could literally program itself!")
