#!/usr/bin/env python3
"""
Runtime Hot-Swap & Bytecode Modification System

This system enables:
1. Direct bytecode instruction editing and reordering
2. Hot-swapping handlers in running processes without restart
3. Signal-based distribution of code changes across network
4. Runtime modification of decorators, entity handlers, callbacks
5. Live source editing with immediate bytecode regeneration
"""

import ast
import dis
import marshal
import pickle
import types
import sys
import inspect
import threading
import time
import weakref
import traceback
import hashlib
import base64
import copy
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import functools


# =============================================================================
# BYTECODE INSTRUCTION MANIPULATION
# =============================================================================

class BytecodeInstruction:
    """Represents a single bytecode instruction that can be modified."""

    def __init__(self, opname: str, arg: int = 0, argval: Any = None,
                 offset: int = 0, starts_line: Optional[int] = None):
        self.opname = opname
        self.arg = arg
        self.argval = argval
        self.offset = offset
        self.starts_line = starts_line

    def __repr__(self):
        return f"BytecodeInstruction({self.opname}, arg={self.arg}, argval={self.argval})"


class BytecodeEditor:
    """Direct bytecode manipulation and instruction reordering."""

    def __init__(self):
        self.instruction_cache = {}
        self.modification_history = []

    def disassemble_to_instructions(self, code_obj: types.CodeType) -> List[BytecodeInstruction]:
        """Disassemble code object to editable instructions."""

        instructions = []
        for instr in dis.get_instructions(code_obj):
            instructions.append(BytecodeInstruction(
                opname=instr.opname,
                arg=instr.arg or 0,
                argval=instr.argval,
                offset=instr.offset,
                starts_line=instr.starts_line
            ))

        return instructions

    def modify_instruction(self, instructions: List[BytecodeInstruction],
                          offset: int, new_opname: str, new_arg: int = 0,
                          new_argval: Any = None) -> List[BytecodeInstruction]:
        """Modify a specific instruction by offset."""

        modified_instructions = copy.deepcopy(instructions)

        for i, instr in enumerate(modified_instructions):
            if instr.offset == offset:
                instr.opname = new_opname
                instr.arg = new_arg
                instr.argval = new_argval
                break

        self.modification_history.append({
            'action': 'modify_instruction',
            'offset': offset,
            'old_opname': instructions[i].opname if i < len(instructions) else None,
            'new_opname': new_opname,
            'timestamp': datetime.now().isoformat()
        })

        return modified_instructions

    def insert_instruction(self, instructions: List[BytecodeInstruction],
                          insert_offset: int, new_instruction: BytecodeInstruction) -> List[BytecodeInstruction]:
        """Insert new instruction at specific offset."""

        modified_instructions = copy.deepcopy(instructions)

        # Find insertion point
        insert_index = 0
        for i, instr in enumerate(modified_instructions):
            if instr.offset >= insert_offset:
                insert_index = i
                break

        # Adjust offsets for instructions after insertion point
        for instr in modified_instructions[insert_index:]:
            instr.offset += 2  # Most instructions are 2 bytes

        # Insert new instruction
        new_instruction.offset = insert_offset
        modified_instructions.insert(insert_index, new_instruction)

        self.modification_history.append({
            'action': 'insert_instruction',
            'offset': insert_offset,
            'instruction': str(new_instruction),
            'timestamp': datetime.now().isoformat()
        })

        return modified_instructions

    def reorder_instructions(self, instructions: List[BytecodeInstruction],
                           reorder_map: Dict[int, int]) -> List[BytecodeInstruction]:
        """Reorder instructions based on offset mapping."""

        modified_instructions = copy.deepcopy(instructions)

        # Create mapping of old offset to new offset
        reordered = []

        for old_offset, new_offset in sorted(reorder_map.items(), key=lambda x: x[1]):
            # Find instruction with old_offset
            for instr in modified_instructions:
                if instr.offset == old_offset:
                    instr.offset = new_offset
                    reordered.append(instr)
                    break

        # Add instructions not in reorder map
        for instr in modified_instructions:
            if instr.offset not in reorder_map:
                reordered.append(instr)

        # Sort by new offset
        reordered.sort(key=lambda x: x.offset)

        self.modification_history.append({
            'action': 'reorder_instructions',
            'reorder_map': reorder_map,
            'timestamp': datetime.now().isoformat()
        })

        return reordered

    def instructions_to_source_approximation(self, instructions: List[BytecodeInstruction]) -> str:
        """Approximate source code from instructions (for debugging)."""

        source_lines = []
        load_stack = []

        for instr in instructions:
            if instr.opname == 'LOAD_CONST':
                load_stack.append(repr(instr.argval))
            elif instr.opname == 'LOAD_FAST':
                load_stack.append(str(instr.argval))
            elif instr.opname == 'LOAD_GLOBAL':
                load_stack.append(str(instr.argval))
            elif instr.opname == 'BINARY_ADD':
                if len(load_stack) >= 2:
                    right = load_stack.pop()
                    left = load_stack.pop()
                    load_stack.append(f"({left} + {right})")
            elif instr.opname == 'RETURN_VALUE':
                if load_stack:
                    current_line = f"return {load_stack.pop()}"
                    source_lines.append(current_line)
            elif instr.opname == 'STORE_FAST':
                if load_stack:
                    value = load_stack.pop()
                    current_line = f"{instr.argval} = {value}"
                    source_lines.append(current_line)

        return "\n".join(source_lines)


# =============================================================================
# RUNTIME FUNCTION HOT-SWAPPING SYSTEM
# =============================================================================

class RuntimeFunctionRegistry:
    """Registry for functions that can be hot-swapped at runtime."""

    def __init__(self):
        self.registered_functions: Dict[str, Callable] = {}
        self.function_metadata: Dict[str, Dict[str, Any]] = {}
        self.swap_history: List[Dict[str, Any]] = []
        self.hot_swap_lock = threading.Lock()

        # Track original implementations
        self.original_implementations: Dict[str, Callable] = {}

        # Active references (weak references to avoid memory leaks)
        self.active_references: Dict[str, List[weakref.ref]] = {}

    def register_swappable_function(self, name: str, func: Callable,
                                   metadata: Dict[str, Any] = None):
        """Register a function for hot-swapping."""

        with self.hot_swap_lock:
            self.registered_functions[name] = func
            self.original_implementations[name] = func
            self.function_metadata[name] = metadata or {}
            self.active_references[name] = []

            # Add swapping metadata
            self.function_metadata[name].update({
                'registered_at': datetime.now().isoformat(),
                'swap_count': 0,
                'original_code_hash': self._get_function_hash(func)
            })

    def hot_swap_function(self, name: str, new_func: Callable,
                         preserve_state: bool = True) -> bool:
        """Hot-swap a registered function with zero downtime."""

        if name not in self.registered_functions:
            return False

        with self.hot_swap_lock:
            old_func = self.registered_functions[name]

            # Preserve state if requested
            preserved_state = None
            if preserve_state and hasattr(old_func, '__dict__'):
                preserved_state = copy.deepcopy(old_func.__dict__)

            # Perform the swap
            self.registered_functions[name] = new_func

            # Restore state to new function
            if preserved_state and hasattr(new_func, '__dict__'):
                new_func.__dict__.update(preserved_state)

            # Update metadata
            self.function_metadata[name]['swap_count'] += 1
            self.function_metadata[name]['last_swap'] = datetime.now().isoformat()
            self.function_metadata[name]['current_code_hash'] = self._get_function_hash(new_func)

            # Record swap history
            swap_record = {
                'function_name': name,
                'old_hash': self._get_function_hash(old_func),
                'new_hash': self._get_function_hash(new_func),
                'swapped_at': datetime.now().isoformat(),
                'preserved_state': preserved_state is not None
            }
            self.swap_history.append(swap_record)

            # Update all active references (if they support it)
            self._update_active_references(name, new_func)

            return True

    def get_swappable_function(self, name: str) -> Optional[Callable]:
        """Get current implementation of swappable function."""
        return self.registered_functions.get(name)

    def create_swappable_wrapper(self, name: str) -> Callable:
        """Create a wrapper that always calls current implementation."""

        def swappable_wrapper(*args, **kwargs):
            current_func = self.registered_functions.get(name)
            if current_func:
                return current_func(*args, **kwargs)
            else:
                raise RuntimeError(f"Swappable function '{name}' not found")

        # Track this wrapper as an active reference
        wrapper_ref = weakref.ref(swappable_wrapper)
        if name in self.active_references:
            self.active_references[name].append(wrapper_ref)

        return swappable_wrapper

    def _get_function_hash(self, func: Callable) -> str:
        """Get hash of function for tracking changes."""
        try:
            if hasattr(func, '__code__'):
                code_bytes = marshal.dumps(func.__code__)
                return hashlib.sha256(code_bytes).hexdigest()
            else:
                return hashlib.sha256(str(func).encode()).hexdigest()
        except:
            return "unknown"

    def _update_active_references(self, name: str, new_func: Callable):
        """Update active references to point to new function."""

        if name not in self.active_references:
            return

        # Clean up dead references
        alive_refs = []
        for ref in self.active_references[name]:
            if ref() is not None:
                alive_refs.append(ref)

        self.active_references[name] = alive_refs

    def get_swap_statistics(self) -> Dict[str, Any]:
        """Get comprehensive swap statistics."""

        return {
            'registered_functions': len(self.registered_functions),
            'total_swaps': len(self.swap_history),
            'functions_with_swaps': len([f for f in self.function_metadata.values() if f['swap_count'] > 0]),
            'most_swapped_function': max(
                self.function_metadata.items(),
                key=lambda x: x[1]['swap_count'],
                default=(None, {'swap_count': 0})
            )[0],
            'recent_swaps': self.swap_history[-5:] if self.swap_history else []
        }


# =============================================================================
# ENTITY HANDLER HOT-SWAPPING SYSTEM
# =============================================================================

class EntitySystem:
    """Entity system with hot-swappable handlers."""

    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.handlers: Dict[str, Dict[str, Callable]] = {}  # entity_type -> {handler_name -> handler}
        self.handler_registry = RuntimeFunctionRegistry()

        # Event tracking
        self.event_log: List[Dict[str, Any]] = []

    def register_entity_type(self, entity_type: str):
        """Register a new entity type."""
        if entity_type not in self.handlers:
            self.handlers[entity_type] = {}

    def register_entity_handler(self, entity_type: str, handler_name: str,
                               handler_func: Callable):
        """Register a handler for an entity type."""

        if entity_type not in self.handlers:
            self.register_entity_type(entity_type)

        # Register with hot-swap registry
        handler_key = f"{entity_type}.{handler_name}"
        self.handler_registry.register_swappable_function(
            handler_key,
            handler_func,
            metadata={
                'entity_type': entity_type,
                'handler_name': handler_name,
                'handler_type': 'entity_handler'
            }
        )

        # Store wrapper that always calls current implementation
        self.handlers[entity_type][handler_name] = self.handler_registry.create_swappable_wrapper(handler_key)

    def hot_swap_entity_handler(self, entity_type: str, handler_name: str,
                               new_handler: Callable) -> bool:
        """Hot-swap an entity handler."""

        handler_key = f"{entity_type}.{handler_name}"
        success = self.handler_registry.hot_swap_function(handler_key, new_handler)

        if success:
            self.event_log.append({
                'event': 'handler_hot_swap',
                'entity_type': entity_type,
                'handler_name': handler_name,
                'timestamp': datetime.now().isoformat()
            })

        return success

    def create_entity(self, entity_id: str, entity_type: str, data: Dict[str, Any] = None):
        """Create a new entity."""

        self.entities[entity_id] = {
            'id': entity_id,
            'type': entity_type,
            'data': data or {},
            'created_at': datetime.now().isoformat()
        }

        # Call onLoad handler if it exists
        if entity_type in self.handlers and 'onLoad' in self.handlers[entity_type]:
            try:
                self.handlers[entity_type]['onLoad'](self.entities[entity_id])

                self.event_log.append({
                    'event': 'entity_created',
                    'entity_id': entity_id,
                    'entity_type': entity_type,
                    'onLoad_called': True,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                self.event_log.append({
                    'event': 'entity_creation_error',
                    'entity_id': entity_id,
                    'entity_type': entity_type,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

    def call_entity_handler(self, entity_id: str, handler_name: str, *args, **kwargs) -> Any:
        """Call a handler for a specific entity."""

        if entity_id not in self.entities:
            raise ValueError(f"Entity {entity_id} not found")

        entity = self.entities[entity_id]
        entity_type = entity['type']

        if entity_type not in self.handlers or handler_name not in self.handlers[entity_type]:
            raise ValueError(f"Handler {handler_name} not found for entity type {entity_type}")

        handler = self.handlers[entity_type][handler_name]

        try:
            result = handler(entity, *args, **kwargs)

            self.event_log.append({
                'event': 'handler_called',
                'entity_id': entity_id,
                'entity_type': entity_type,
                'handler_name': handler_name,
                'success': True,
                'timestamp': datetime.now().isoformat()
            })

            return result

        except Exception as e:
            self.event_log.append({
                'event': 'handler_error',
                'entity_id': entity_id,
                'entity_type': entity_type,
                'handler_name': handler_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise


# =============================================================================
# DECORATOR HOT-SWAPPING SYSTEM
# =============================================================================

class DecoratorHotSwapSystem:
    """System for hot-swapping decorators on existing functions."""

    def __init__(self):
        self.decorated_functions: Dict[str, Dict[str, Any]] = {}
        self.decorator_registry: Dict[str, Callable] = {}
        self.swap_history: List[Dict[str, Any]] = []

    def register_decorator(self, name: str, decorator_func: Callable):
        """Register a decorator for hot-swapping."""
        self.decorator_registry[name] = decorator_func

    def apply_decorator_to_function(self, func_name: str, target_func: Callable,
                                   decorator_name: str) -> Callable:
        """Apply a decorator to an existing function."""

        if decorator_name not in self.decorator_registry:
            raise ValueError(f"Decorator {decorator_name} not registered")

        decorator = self.decorator_registry[decorator_name]
        decorated_func = decorator(target_func)

        # Store metadata
        self.decorated_functions[func_name] = {
            'original_function': target_func,
            'current_decorated': decorated_func,
            'current_decorator': decorator_name,
            'applied_at': datetime.now().isoformat(),
            'decorator_history': [decorator_name]
        }

        return decorated_func

    def hot_swap_decorator(self, func_name: str, new_decorator_name: str) -> Callable:
        """Hot-swap the decorator on an existing function."""

        if func_name not in self.decorated_functions:
            raise ValueError(f"Function {func_name} not tracked for decorator swapping")

        if new_decorator_name not in self.decorator_registry:
            raise ValueError(f"Decorator {new_decorator_name} not registered")

        func_info = self.decorated_functions[func_name]
        original_func = func_info['original_function']
        new_decorator = self.decorator_registry[new_decorator_name]

        # Apply new decorator to original function
        new_decorated_func = new_decorator(original_func)

        # Update tracking
        old_decorator = func_info['current_decorator']
        func_info['current_decorated'] = new_decorated_func
        func_info['current_decorator'] = new_decorator_name
        func_info['decorator_history'].append(new_decorator_name)
        func_info['last_swap'] = datetime.now().isoformat()

        # Record swap
        self.swap_history.append({
            'function_name': func_name,
            'old_decorator': old_decorator,
            'new_decorator': new_decorator_name,
            'swapped_at': datetime.now().isoformat()
        })

        return new_decorated_func

    def get_current_decorated_function(self, func_name: str) -> Optional[Callable]:
        """Get current decorated version of function."""

        if func_name in self.decorated_functions:
            return self.decorated_functions[func_name]['current_decorated']
        return None


# =============================================================================
# SIGNAL-BASED CODE DISTRIBUTION SYSTEM
# =============================================================================

class CodeDistributionSignal:
    """Signal payload for distributing code changes."""

    def __init__(self, operation: str, target: str, code_data: Dict[str, Any]):
        self.operation = operation  # 'hot_swap', 'modify_bytecode', 'add_decorator'
        self.target = target        # function name, entity handler, etc.
        self.code_data = code_data  # serialized code, bytecode, etc.
        self.timestamp = datetime.now().isoformat()
        self.signature = self._create_signature()

    def _create_signature(self) -> str:
        """Create cryptographic signature for integrity."""
        data_str = f"{self.operation}{self.target}{str(self.code_data)}{self.timestamp}"
        return hashlib.sha256(data_str.encode()).hexdigest()


class NetworkCodeDistributor:
    """Distribute code changes across network via signals."""

    def __init__(self):
        self.connected_systems: Dict[str, Any] = {}
        self.distribution_log: List[Dict[str, Any]] = []

        # Security
        self.trusted_sources: set = set()
        self.code_validation_enabled = True

    def register_system(self, system_id: str, system_instance: Any):
        """Register a system to receive code updates."""
        self.connected_systems[system_id] = system_instance

    def distribute_hot_swap(self, target_function: str, new_code_source: str,
                           target_systems: List[str] = None):
        """Distribute hot-swap via signals."""

        # Compile code to bytecode
        try:
            compiled_code = compile(new_code_source, '<distributed>', 'exec')
            bytecode_data = base64.b64encode(marshal.dumps(compiled_code)).decode('utf-8')
        except Exception as e:
            print(f"Code compilation failed: {e}")
            return False

        # Create distribution signal
        signal = CodeDistributionSignal(
            operation='hot_swap',
            target=target_function,
            code_data={
                'source_code': new_code_source,
                'bytecode': bytecode_data,
                'python_version': sys.version_info[:2]
            }
        )

        # Distribute to target systems
        target_list = target_systems or list(self.connected_systems.keys())
        results = {}

        for system_id in target_list:
            if system_id in self.connected_systems:
                try:
                    success = self._apply_signal_to_system(system_id, signal)
                    results[system_id] = success
                except Exception as e:
                    results[system_id] = False
                    print(f"Distribution to {system_id} failed: {e}")
            else:
                results[system_id] = False

        # Log distribution
        self.distribution_log.append({
            'signal_operation': signal.operation,
            'target': signal.target,
            'target_systems': target_list,
            'results': results,
            'timestamp': signal.timestamp
        })

        return results

    def _apply_signal_to_system(self, system_id: str, signal: CodeDistributionSignal) -> bool:
        """Apply signal to a specific system."""

        system = self.connected_systems[system_id]

        if signal.operation == 'hot_swap':
            return self._apply_hot_swap_signal(system, signal)
        elif signal.operation == 'modify_bytecode':
            return self._apply_bytecode_modification_signal(system, signal)
        elif signal.operation == 'add_decorator':
            return self._apply_decorator_signal(system, signal)

        return False

    def _apply_hot_swap_signal(self, system: Any, signal: CodeDistributionSignal) -> bool:
        """Apply hot-swap signal to system."""

        # Extract new function from code
        namespace = {}
        try:
            # Execute the distributed code
            exec(signal.code_data['source_code'], namespace)

            # Find the function to swap
            new_func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('__'):
                    new_func = obj
                    break

            if not new_func:
                return False

            # Apply to system based on type
            if hasattr(system, 'hot_swap_function'):
                return system.hot_swap_function(signal.target, new_func)
            elif hasattr(system, 'hot_swap_entity_handler'):
                # Parse target as entity_type.handler_name
                if '.' in signal.target:
                    entity_type, handler_name = signal.target.split('.', 1)
                    return system.hot_swap_entity_handler(entity_type, handler_name, new_func)

        except Exception as e:
            print(f"Hot-swap application failed: {e}")
            return False

        return False

    def _apply_bytecode_modification_signal(self, system: Any, signal: CodeDistributionSignal) -> bool:
        """Apply bytecode modification signal."""
        return False

    def _apply_decorator_signal(self, system: Any, signal: CodeDistributionSignal) -> bool:
        """Apply decorator modification signal."""
        return False


# =============================================================================
# INTEGRATION WITH SIGNAL SYSTEM FOR LIVE UPDATES
# =============================================================================

class LiveUpdateSignalHandler:
    """Handles live update signals for runtime modification."""

    def __init__(self, systems: Dict[str, Any]):
        self.systems = systems
        self.update_queue = []
        self.processing_lock = threading.Lock()

    def handle_live_update_signal(self, signal_payload):
        """Handle incoming live update signal."""

        update_type = signal_payload.get('update_type')
        target = signal_payload.get('target')
        code_data = signal_payload.get('code_data')

        with self.processing_lock:
            if update_type == 'hot_swap_entity_handler':
                return self._handle_entity_handler_update(target, code_data)
            elif update_type == 'hot_swap_function':
                return self._handle_function_update(target, code_data)
            elif update_type == 'hot_swap_decorator':
                return self._handle_decorator_update(target, code_data)
            elif update_type == 'bytecode_patch':
                return self._handle_bytecode_patch(target, code_data)

    def _handle_entity_handler_update(self, target, code_data):
        """Handle entity handler update."""
        entity_system = self.systems.get('entity_system')
        if not entity_system:
            return False

        try:
            # Parse target (entity_type.handler_name)
            entity_type, handler_name = target.split('.', 1)

            # Execute code to get new handler
            namespace = {}
            exec(code_data['source_code'], namespace)

            # Find the handler function
            new_handler = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('__'):
                    new_handler = obj
                    break

            if new_handler:
                return entity_system.hot_swap_entity_handler(entity_type, handler_name, new_handler)

        except Exception as e:
            print(f"Entity handler update failed: {e}")
            return False

        return False

    def _handle_function_update(self, target, code_data):
        """Handle function update."""
        function_registry = self.systems.get('function_registry')
        if not function_registry:
            return False

        try:
            # Execute code to get new function
            namespace = {}
            exec(code_data['source_code'], namespace)

            # Find the function
            new_func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('__'):
                    new_func = obj
                    break

            if new_func:
                return function_registry.hot_swap_function(target, new_func)

        except Exception as e:
            print(f"Function update failed: {e}")
            return False

        return False

    def _handle_decorator_update(self, target, code_data):
        """Handle decorator update."""
        return False

    def _handle_bytecode_patch(self, target, code_data):
        """Handle bytecode patch."""
        return False
