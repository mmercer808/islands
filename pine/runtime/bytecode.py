"""
Pine Bytecode Utilities
=======================

Low-level bytecode analysis and manipulation.

Features:
- Code object inspection
- Bytecode disassembly
- Code object construction
- Instruction modification

SOURCE: code_object_utility.py (52KB)
Copy the full implementation from that file.

Quick Reference:
----------------
    from pine.runtime import CodeObjectAnalyzer

    analyzer = CodeObjectAnalyzer()

    def example(x, y):
        return x + y

    # Analyze
    info = analyzer.analyze(example)
    print(info.instructions)
    print(info.constants)
    print(info.local_vars)
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import types
import dis


# =============================================================================
#                              DATA CLASSES
# =============================================================================

@dataclass
class InstructionInfo:
    """Information about a bytecode instruction."""
    offset: int
    opcode: int
    opname: str
    arg: Optional[int]
    argval: Any
    is_jump_target: bool


@dataclass
class CodeObjectInfo:
    """Analysis of a code object."""
    name: str
    filename: str
    firstlineno: int
    argcount: int
    kwonlyargcount: int
    nlocals: int
    stacksize: int
    flags: int
    constants: Tuple[Any, ...]
    names: Tuple[str, ...]
    varnames: Tuple[str, ...]
    freevars: Tuple[str, ...]
    cellvars: Tuple[str, ...]
    instructions: List[InstructionInfo] = field(default_factory=list)


# =============================================================================
#                              ANALYZER
# =============================================================================

class CodeObjectAnalyzer:
    """
    Analyzes Python code objects and functions.

    Provides detailed information about:
    - Bytecode instructions
    - Constants and variables
    - Stack requirements
    - Control flow

    TODO: Copy full implementation from code_object_utility.py
    """

    def analyze(self, func: types.FunctionType) -> CodeObjectInfo:
        """Analyze a function's code object."""
        code = func.__code__

        # Get instructions
        instructions = []
        for instr in dis.get_instructions(func):
            instructions.append(InstructionInfo(
                offset=instr.offset,
                opcode=instr.opcode,
                opname=instr.opname,
                arg=instr.arg,
                argval=instr.argval,
                is_jump_target=instr.is_jump_target
            ))

        return CodeObjectInfo(
            name=code.co_name,
            filename=code.co_filename,
            firstlineno=code.co_firstlineno,
            argcount=code.co_argcount,
            kwonlyargcount=code.co_kwonlyargcount,
            nlocals=code.co_nlocals,
            stacksize=code.co_stacksize,
            flags=code.co_flags,
            constants=code.co_consts,
            names=code.co_names,
            varnames=code.co_varnames,
            freevars=code.co_freevars,
            cellvars=code.co_cellvars,
            instructions=instructions
        )

    def disassemble(self, func: types.FunctionType) -> str:
        """Get disassembly as string."""
        import io
        output = io.StringIO()
        dis.dis(func, file=output)
        return output.getvalue()

    def get_constants(self, func: types.FunctionType) -> Tuple[Any, ...]:
        """Get all constants from function."""
        return func.__code__.co_consts

    def get_local_vars(self, func: types.FunctionType) -> Tuple[str, ...]:
        """Get local variable names."""
        return func.__code__.co_varnames


class BytecodeInspector:
    """
    Deep inspection of bytecode for debugging.

    TODO: Copy implementation from code_object_utility.py
    """

    def __init__(self):
        self.analyzer = CodeObjectAnalyzer()

    def find_calls(self, func: types.FunctionType) -> List[str]:
        """Find all function calls in bytecode."""
        info = self.analyzer.analyze(func)
        calls = []
        for instr in info.instructions:
            if instr.opname in ('CALL_FUNCTION', 'CALL', 'CALL_KW'):
                calls.append(str(instr.argval))
        return calls

    def find_globals(self, func: types.FunctionType) -> List[str]:
        """Find all global variable accesses."""
        info = self.analyzer.analyze(func)
        globals_accessed = []
        for instr in info.instructions:
            if instr.opname in ('LOAD_GLOBAL', 'STORE_GLOBAL'):
                globals_accessed.append(str(instr.argval))
        return globals_accessed


# =============================================================================
#                              BUILDER
# =============================================================================

class CodeObjectBuilder:
    """
    Builds new code objects programmatically.

    WARNING: This is advanced, low-level code manipulation.

    TODO: Copy implementation from code_object_utility.py
    """

    def __init__(self):
        self._code_args: Dict[str, Any] = {}

    def from_function(self, func: types.FunctionType) -> 'CodeObjectBuilder':
        """Start from an existing function's code."""
        code = func.__code__
        self._code_args = {
            'co_argcount': code.co_argcount,
            'co_posonlyargcount': getattr(code, 'co_posonlyargcount', 0),
            'co_kwonlyargcount': code.co_kwonlyargcount,
            'co_nlocals': code.co_nlocals,
            'co_stacksize': code.co_stacksize,
            'co_flags': code.co_flags,
            'co_code': code.co_code,
            'co_consts': code.co_consts,
            'co_names': code.co_names,
            'co_varnames': code.co_varnames,
            'co_filename': code.co_filename,
            'co_name': code.co_name,
            'co_firstlineno': code.co_firstlineno,
            'co_linetable': getattr(code, 'co_linetable', b''),
            'co_freevars': code.co_freevars,
            'co_cellvars': code.co_cellvars,
        }
        return self

    def with_constants(self, *constants) -> 'CodeObjectBuilder':
        """Set constants."""
        self._code_args['co_consts'] = constants
        return self

    def with_name(self, name: str) -> 'CodeObjectBuilder':
        """Set function name."""
        self._code_args['co_name'] = name
        return self

    def build(self) -> types.CodeType:
        """Build the code object."""
        # Python 3.8+ code object construction
        # This is version-dependent
        return types.CodeType(
            self._code_args['co_argcount'],
            self._code_args['co_posonlyargcount'],
            self._code_args['co_kwonlyargcount'],
            self._code_args['co_nlocals'],
            self._code_args['co_stacksize'],
            self._code_args['co_flags'],
            self._code_args['co_code'],
            self._code_args['co_consts'],
            self._code_args['co_names'],
            self._code_args['co_varnames'],
            self._code_args['co_filename'],
            self._code_args['co_name'],
            self._code_args['co_firstlineno'],
            self._code_args['co_linetable'],
            self._code_args['co_freevars'],
            self._code_args['co_cellvars'],
        )


class InstructionModifier:
    """
    Modifies individual bytecode instructions.

    TODO: Copy implementation from code_object_utility.py
    """

    def nop_instruction(
        self,
        code: types.CodeType,
        offset: int
    ) -> types.CodeType:
        """Replace instruction at offset with NOP."""
        # This requires modifying co_code bytes
        # TODO: Implement
        raise NotImplementedError()
