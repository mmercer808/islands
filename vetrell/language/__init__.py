#!/usr/bin/env python3
"""
Vetrell Language Module

English language as code object with chain-based rule system for linguistic processing.

Classes:
    LanguageRuleType - Types of linguistic rules
    ChainDecisionType - Types of decisions a chain can make
    LanguageToken - Represents a linguistic unit
    RuleChain - A chain of linguistic rules with callbacks
    ChainNode - A node in the linguistic decision tree
    EnglishLanguageCodeObject - Main English language interpreter
    AdaptiveChainLearner - Learns and evolves chain links
    SelfModifyingLanguageSystem - Language system that modifies its own rules
"""

from .english_code_object import (
    LanguageRuleType,
    ChainDecisionType,
    LanguageToken,
    RuleChain,
    ChainNode,
    EnglishLanguageCodeObject,
    AdaptiveChainLearner,
    SelfModifyingLanguageSystem,
)

__all__ = [
    "LanguageRuleType",
    "ChainDecisionType",
    "LanguageToken",
    "RuleChain",
    "ChainNode",
    "EnglishLanguageCodeObject",
    "AdaptiveChainLearner",
    "SelfModifyingLanguageSystem",
]
