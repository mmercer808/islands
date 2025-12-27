#!/usr/bin/env python3
"""
Vetrell Analysis Module

Code object metadata extraction and process analysis utilities.

Classes:
    CodeObjectType - Types of code objects and callable entities
    ProcessInfo - Current process metadata
    ThreadInfo - Thread state and stack information
    MemoryInfo - GC stats and memory usage
    ModuleInfo - Module details and dependencies
    FrameInfo - Stack frame information
    ParameterInfo - Function parameter details
    SignatureInfo - Function signature metadata
    ASTMetadata - AST analysis metadata
    BytecodeInfo - Bytecode analysis metadata
    ComprehensiveCodeObjectMetadata - Master metadata container
    CodeObjectContext - Abstract context for code execution
    ComprehensivePythonProcessAnalyzer - Main analyzer class
"""

from .code_object import (
    CodeObjectType,
    ProcessInfo,
    ThreadInfo,
    MemoryInfo,
    ModuleInfo,
    FrameInfo,
    ParameterInfo,
    SignatureInfo,
    ASTMetadata,
    BytecodeInfo,
    ComprehensiveCodeObjectMetadata,
    CodeObjectContext,
    ComprehensivePythonProcessAnalyzer,
)

__all__ = [
    "CodeObjectType",
    "ProcessInfo",
    "ThreadInfo",
    "MemoryInfo",
    "ModuleInfo",
    "FrameInfo",
    "ParameterInfo",
    "SignatureInfo",
    "ASTMetadata",
    "BytecodeInfo",
    "ComprehensiveCodeObjectMetadata",
    "CodeObjectContext",
    "ComprehensivePythonProcessAnalyzer",
]
