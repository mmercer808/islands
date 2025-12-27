#!/usr/bin/env python3
"""
Advanced Generator Composition System

This module provides advanced generator composition patterns including:
- Sequential, Pipeline, Branch-Merge, Recursive patterns
- Generator state branching and merging
- Complex composition chains with conditional logic
"""

import uuid
import time
import copy
from typing import Dict, Any, List, Optional, Callable, Generator, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

# =============================================================================
# GENERATOR COMPOSITION PATTERNS
# =============================================================================

class GeneratorCompositionPattern(Enum):
    """Patterns for generator composition."""
    SEQUENTIAL = "sequential"      # One after another
    PARALLEL = "parallel"          # All at once
    PIPELINE = "pipeline"          # Output of one feeds next
    BRANCH_MERGE = "branch_merge"  # Branch execution then merge
    CONDITIONAL = "conditional"    # Conditional execution paths
    RECURSIVE = "recursive"        # Self-referencing generators


@dataclass
class GeneratorBranch:
    """Represents a branch in generator execution."""
    branch_id: str
    initial_data: Any
    result: Any = None
    execution_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def set_result(self, result: Any):
        """Set branch execution result."""
        self.result = result
        self.completed_at = datetime.now()
        self.execution_time = (self.completed_at - self.created_at).total_seconds()
    
    def get_score(self) -> float:
        """Get branch score for selection."""
        if self.result is None:
            return 0.0
        
        if isinstance(self.result, dict):
            return len(self.result)
        elif isinstance(self.result, (list, tuple)):
            return len(self.result)
        elif isinstance(self.result, (int, float)):
            return float(self.result)
        else:
            return 1.0


# =============================================================================
# GENERATOR STATE BRANCHING
# =============================================================================

class GeneratorStateBranch:
    """Represents a branch in generator state execution."""
    
    def __init__(self, branch_id: str, parent_state: Dict[str, Any]):
        self.branch_id = branch_id
        self.parent_state = parent_state.copy()
        self.branch_modifications: Dict[str, Any] = {}
        self.execution_path: List[str] = []
        self.child_branches: List['GeneratorStateBranch'] = []
        
    def create_child_branch(self, modifications: Dict[str, Any] = None) -> 'GeneratorStateBranch':
        """Create child branch with optional state modifications."""
        child_id = f"{self.branch_id}_child_{len(self.child_branches)}"
        
        # Merge parent state with current branch modifications
        merged_state = {**self.parent_state, **self.branch_modifications}
        child = GeneratorStateBranch(child_id, merged_state)
        
        if modifications:
            child.branch_modifications.update(modifications)
        
        self.child_branches.append(child)
        return child
    
    def get_effective_state(self) -> Dict[str, Any]:
        """Get the effective state (parent + modifications)."""
        return {**self.parent_state, **self.branch_modifications}


# =============================================================================
# GENERATOR COMPOSITION ENGINE
# =============================================================================

class GeneratorCompositionEngine:
    """Engine for linking and composing multiple generators."""
    
    def __init__(self):
        self.generator_registry: Dict[str, Generator] = {}
        self.generator_states: Dict[str, Dict[str, Any]] = {}
        self.composition_chains: Dict[str, List[str]] = {}
        self.state_branches: Dict[str, GeneratorStateBranch] = {}
        
    def register_generator(self, gen_id: str, generator: Generator, 
                          initial_state: Dict[str, Any] = None):
        """Register generator with initial state."""
        self.generator_registry[gen_id] = generator
        self.generator_states[gen_id] = initial_state or {}
        
    def create_composition_chain(self, chain_id: str, generator_ids: List[str]) -> str:
        """Create composition chain linking multiple generators."""
        # Validate all generators exist
        for gen_id in generator_ids:
            if gen_id not in self.generator_registry:
                raise ValueError(f"Generator {gen_id} not registered")
        
        self.composition_chains[chain_id] = generator_ids
        return chain_id
    
    def branch_generator_state(self, gen_id: str, branch_id: str = None, 
                              modifications: Dict[str, Any] = None) -> str:
        """Branch generator state for parallel execution paths."""
        if gen_id not in self.generator_states:
            raise ValueError(f"Generator {gen_id} not found")
        
        branch_id = branch_id or f"{gen_id}_branch_{uuid.uuid4()}"
        current_state = self.generator_states[gen_id]
        
        branch = GeneratorStateBranch(branch_id, current_state)
        if modifications:
            branch.branch_modifications.update(modifications)
        
        self.state_branches[branch_id] = branch
        return branch_id
    
    def compose_generators(self, chain_id: str, input_data: Any = None) -> Generator:
        """Create composed generator that chains multiple generators."""
        if chain_id not in self.composition_chains:
            raise ValueError(f"Composition chain {chain_id} not found")
        
        generator_ids = self.composition_chains[chain_id]
        
        def composed_generator():
            current_data = input_data
            
            for gen_id in generator_ids:
                generator = self.generator_registry[gen_id]
                state = self.generator_states[gen_id]
                
                try:
                    # Send current data to generator and collect results
                    if current_data is not None:
                        result = generator.send(current_data)
                    else:
                        result = next(generator)
                    
                    # Update state
                    if isinstance(result, dict) and 'state_update' in result:
                        state.update(result['state_update'])
                    
                    yield {
                        'generator_id': gen_id,
                        'result': result,
                        'state': state.copy(),
                        'chain_position': generator_ids.index(gen_id)
                    }
                    
                    # Pass result as input to next generator
                    current_data = result
                    
                except StopIteration:
                    # Generator exhausted, continue with next
                    continue
        
        return composed_generator()
    
    def serialize_generator_composition(self, chain_id: str) -> Dict[str, Any]:
        """Serialize generator composition state."""
        if chain_id not in self.composition_chains:
            return {}
        
        return {
            'chain_id': chain_id,
            'generator_ids': self.composition_chains[chain_id],
            'generator_states': {
                gen_id: state for gen_id, state in self.generator_states.items()
                if gen_id in self.composition_chains[chain_id]
            },
            'active_branches': {
                branch_id: {
                    'parent_state': branch.parent_state,
                    'modifications': branch.branch_modifications,
                    'execution_path': branch.execution_path
                }
                for branch_id, branch in self.state_branches.items()
            }
        }


# =============================================================================
# ADVANCED GENERATOR COMPOSER
# =============================================================================

class AdvancedGeneratorComposer:
    """Advanced generator composition with complex patterns."""
    
    def __init__(self, context=None):
        self.context = context
        self.generator_registry: Dict[str, Callable] = {}
        self.composition_patterns: Dict[str, Dict[str, Any]] = {}
        self.active_compositions: Dict[str, Generator] = {}
        
        # State branching
        self.branch_registry: Dict[str, GeneratorBranch] = {}
        self.merge_points: Dict[str, List[str]] = {}
        
    def register_generator_factory(self, name: str, factory_func: Callable):
        """Register generator factory function."""
        self.generator_registry[name] = factory_func
    
    def create_composition_pattern(self, pattern_id: str, pattern_type: GeneratorCompositionPattern,
                                 generator_specs: List[Dict[str, Any]]) -> str:
        """Create advanced composition pattern."""
        
        pattern = {
            'pattern_id': pattern_id,
            'pattern_type': pattern_type,
            'generator_specs': generator_specs,
            'created_at': datetime.now().isoformat(),
            'execution_context': self.context.context_id if self.context else None
        }
        
        self.composition_patterns[pattern_id] = pattern
        return pattern_id
    
    def execute_sequential_pattern(self, pattern_id: str, initial_data: Any = None) -> Generator:
        """Execute sequential composition pattern."""
        pattern = self.composition_patterns[pattern_id]
        
        def sequential_generator():
            current_data = initial_data
            
            for spec in pattern['generator_specs']:
                gen_name = spec['name']
                gen_config = spec.get('config', {})
                
                if gen_name in self.generator_registry:
                    factory = self.generator_registry[gen_name]
                    generator = factory(gen_config)
                    
                    try:
                        # Send current data and get results
                        if current_data is not None:
                            result = generator.send(current_data)
                        else:
                            result = next(generator)
                        
                        yield {
                            'generator': gen_name,
                            'result': result,
                            'step': pattern['generator_specs'].index(spec)
                        }
                        
                        current_data = result
                        
                    except StopIteration:
                        continue
        
        return sequential_generator()
    
    def execute_pipeline_pattern(self, pattern_id: str, input_stream: Generator) -> Generator:
        """Execute pipeline composition pattern."""
        pattern = self.composition_patterns[pattern_id]
        
        def pipeline_generator():
            # Create all generators in the pipeline
            generators = []
            for spec in pattern['generator_specs']:
                gen_name = spec['name']
                gen_config = spec.get('config', {})
                
                if gen_name in self.generator_registry:
                    factory = self.generator_registry[gen_name]
                    generator = factory(gen_config)
                    generators.append((gen_name, generator))
            
            # Process input through pipeline
            for input_data in input_stream:
                current_data = input_data
                
                for gen_name, generator in generators:
                    try:
                        if current_data is not None:
                            current_data = generator.send(current_data)
                        else:
                            current_data = next(generator)
                    except StopIteration:
                        current_data = None
                        break
                
                if current_data is not None:
                    yield {
                        'input': input_data,
                        'output': current_data,
                        'pipeline_id': pattern_id
                    }
        
        return pipeline_generator()
    
    def execute_branch_merge_pattern(self, pattern_id: str, initial_data: Any = None) -> Generator:
        """Execute branch-merge composition pattern."""
        pattern = self.composition_patterns[pattern_id]
        
        def branch_merge_generator():
            # Find branch points and merge points
            branch_specs = [spec for spec in pattern['generator_specs'] if spec.get('type') == 'branch']
            merge_specs = [spec for spec in pattern['generator_specs'] if spec.get('type') == 'merge']
            
            # Execute branches in parallel
            branch_results = []
            
            for branch_spec in branch_specs:
                gen_name = branch_spec['name']
                gen_config = branch_spec.get('config', {})
                branch_id = branch_spec.get('branch_id', gen_name)
                
                if gen_name in self.generator_registry:
                    factory = self.generator_registry[gen_name]
                    generator = factory(gen_config)
                    
                    # Create branch execution
                    branch = GeneratorBranch(branch_id, initial_data)
                    
                    try:
                        if initial_data is not None:
                            result = generator.send(initial_data)
                        else:
                            result = next(generator)
                        
                        branch.set_result(result)
                        branch_results.append(branch)
                        
                    except StopIteration:
                        branch.set_result(None)
                        branch_results.append(branch)
            
            # Merge results
            for merge_spec in merge_specs:
                merge_strategy = merge_spec.get('strategy', 'combine')
                
                if merge_strategy == 'combine':
                    merged_result = {
                        'branches': {branch.branch_id: branch.result for branch in branch_results},
                        'merge_type': 'combine'
                    }
                elif merge_strategy == 'select_best':
                    # Simple best selection (could be more sophisticated)
                    best_branch = max(branch_results, key=lambda b: b.get_score())
                    merged_result = {
                        'selected_branch': best_branch.branch_id,
                        'result': best_branch.result,
                        'merge_type': 'select_best'
                    }
                else:
                    merged_result = {'merge_type': 'unknown', 'branches': branch_results}
                
                yield merged_result
        
        return branch_merge_generator()
    
    def create_recursive_pattern(self, pattern_id: str, base_case_condition: Callable,
                               recursive_generator_spec: Dict[str, Any]) -> str:
        """Create recursive generator composition pattern."""
        
        def recursive_factory(config):
            def recursive_generator(data, depth=0):
                max_depth = config.get('max_depth', 10)
                
                # Base case
                if base_case_condition(data) or depth >= max_depth:
                    yield {'base_case': True, 'data': data, 'depth': depth}
                    return
                
                # Recursive case
                gen_name = recursive_generator_spec['name']
                if gen_name in self.generator_registry:
                    factory = self.generator_registry[gen_name]
                    sub_generator = factory(config)
                    
                    try:
                        processed_data = sub_generator.send(data) if data else next(sub_generator)
                        
                        # Recurse with processed data
                        yield from recursive_generator(processed_data, depth + 1)
                        
                    except StopIteration:
                        yield {'recursive_end': True, 'depth': depth}
            
            return recursive_generator
        
        self.generator_registry[f"{pattern_id}_recursive"] = recursive_factory
        
        pattern = {
            'pattern_id': pattern_id,
            'pattern_type': GeneratorCompositionPattern.RECURSIVE,
            'base_case_condition': base_case_condition,
            'recursive_spec': recursive_generator_spec,
            'created_at': datetime.now().isoformat()
        }
        
        self.composition_patterns[pattern_id] = pattern
        return pattern_id
    
    def execute_conditional_pattern(self, pattern_id: str, condition_func: Callable,
                                  initial_data: Any = None) -> Generator:
        """Execute conditional composition pattern."""
        pattern = self.composition_patterns[pattern_id]
        
        def conditional_generator():
            for spec in pattern['generator_specs']:
                gen_name = spec['name']
                gen_config = spec.get('config', {})
                condition = spec.get('condition', lambda x: True)
                
                # Check condition
                if not condition_func(initial_data, spec):
                    continue
                
                if gen_name in self.generator_registry:
                    factory = self.generator_registry[gen_name]
                    generator = factory(gen_config)
                    
                    try:
                        if initial_data is not None:
                            result = generator.send(initial_data)
                        else:
                            result = next(generator)
                        
                        yield {
                            'generator': gen_name,
                            'result': result,
                            'condition_met': True,
                            'spec': spec
                        }
                        
                    except StopIteration:
                        yield {
                            'generator': gen_name,
                            'result': None,
                            'condition_met': True,
                            'completed': True
                        }
        
        return conditional_generator()
    
    def get_composition_stats(self) -> Dict[str, Any]:
        """Get composition statistics."""
        return {
            'registered_generators': len(self.generator_registry),
            'composition_patterns': len(self.composition_patterns),
            'active_compositions': len(self.active_compositions),
            'branch_registry_size': len(self.branch_registry),
            'merge_points': len(self.merge_points),
            'generator_names': list(self.generator_registry.keys()),
            'pattern_types': [p['pattern_type'].value for p in self.composition_patterns.values()]
        }


# =============================================================================
# GENERATOR UTILITIES AND HELPERS
# =============================================================================

def create_data_generator_factory():
    """Create a simple data generator factory for testing."""
    def factory(config):
        def data_generator():
            count = config.get('count', 5)
            start_value = config.get('start_value', 0)
            step = config.get('step', 1)
            
            for i in range(count):
                yield {
                    'id': start_value + (i * step),
                    'value': (start_value + (i * step)) * 10,
                    'timestamp': datetime.now().isoformat(),
                    'config': config
                }
        return data_generator()
    return factory

def create_transformer_generator_factory():
    """Create a transformer generator factory for testing."""
    def factory(config):
        def transformer_generator():
            multiplier = config.get('multiplier', 2)
            transform_type = config.get('transform_type', 'multiply')
            
            while True:
                data = yield
                if data:
                    if transform_type == 'multiply':
                        transformed_value = data.get('value', 0) * multiplier
                    elif transform_type == 'add':
                        transformed_value = data.get('value', 0) + multiplier
                    elif transform_type == 'square':
                        transformed_value = data.get('value', 0) ** 2
                    else:
                        transformed_value = data.get('value', 0)
                    
                    transformed = {
                        'original_id': data.get('id'),
                        'original_value': data.get('value'),
                        'transformed_value': transformed_value,
                        'transformation': f"{transform_type} by {multiplier}",
                        'processed_at': datetime.now().isoformat()
                    }
                    yield transformed
        return transformer_generator()
    return factory

def create_filter_generator_factory():
    """Create a filter generator factory for testing."""
    def factory(config):
        def filter_generator():
            filter_condition = config.get('filter_condition', 'all')
            threshold = config.get('threshold', 50)
            
            while True:
                data = yield
                if data:
                    value = data.get('transformed_value', data.get('value', 0))
                    
                    should_pass = True
                    if filter_condition == 'greater_than':
                        should_pass = value > threshold
                    elif filter_condition == 'less_than':
                        should_pass = value < threshold
                    elif filter_condition == 'even':
                        should_pass = value % 2 == 0
                    elif filter_condition == 'odd':
                        should_pass = value % 2 == 1
                    
                    if should_pass:
                        filtered = {
                            **data,
                            'filter_passed': True,
                            'filter_condition': filter_condition,
                            'filtered_at': datetime.now().isoformat()
                        }
                        yield filtered
        return filter_generator()
    return factory

def create_aggregator_generator_factory():
    """Create an aggregator generator factory for testing."""
    def factory(config):
        def aggregator_generator():
            buffer_size = config.get('buffer_size', 3)
            aggregation_type = config.get('aggregation_type', 'sum')
            
            buffer = []
            
            while True:
                data = yield
                if data:
                    buffer.append(data)
                    
                    if len(buffer) >= buffer_size:
                        # Perform aggregation
                        values = [item.get('transformed_value', item.get('value', 0)) for item in buffer]
                        
                        if aggregation_type == 'sum':
                            result = sum(values)
                        elif aggregation_type == 'average':
                            result = sum(values) / len(values)
                        elif aggregation_type == 'max':
                            result = max(values)
                        elif aggregation_type == 'min':
                            result = min(values)
                        else:
                            result = values
                        
                        aggregated = {
                            'aggregated_value': result,
                            'aggregation_type': aggregation_type,
                            'buffer_size': len(buffer),
                            'source_items': buffer.copy(),
                            'aggregated_at': datetime.now().isoformat()
                        }
                        
                        buffer.clear()
                        yield aggregated
        return aggregator_generator()
    return factory


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

async def demo_generator_composition():
    """Demonstrate generator composition patterns."""
    print("🔗 Generator Composition Demo")
    print("-" * 40)
    
    # Create composer
    composer = AdvancedGeneratorComposer()
    
    # Register generator factories
    composer.register_generator_factory('data_gen', create_data_generator_factory())
    composer.register_generator_factory('transformer_gen', create_transformer_generator_factory())
    composer.register_generator_factory('filter_gen', create_filter_generator_factory())
    composer.register_generator_factory('aggregator_gen', create_aggregator_generator_factory())
    
    print("✅ Registered 4 generator factories")
    
    # Demo 1: Sequential Pattern
    print("\n📋 Demo 1: Sequential Pattern")
    print("-" * 30)
    
    sequential_pattern = composer.create_composition_pattern(
        'demo_sequential',
        GeneratorCompositionPattern.SEQUENTIAL,
        [
            {'name': 'data_gen', 'config': {'count': 3, 'start_value': 1}},
            {'name': 'transformer_gen', 'config': {'multiplier': 5, 'transform_type': 'multiply'}}
        ]
    )
    
    print("Sequential results:")
    for result in composer.execute_sequential_pattern(sequential_pattern):
        print(f"  Step {result['step']}: {result['generator']} -> {result['result']}")
    
    # Demo 2: Branch-Merge Pattern
    print("\n🌿 Demo 2: Branch-Merge Pattern")
    print("-" * 30)
    
    branch_merge_pattern = composer.create_composition_pattern(
        'demo_branch_merge',
        GeneratorCompositionPattern.BRANCH_MERGE,
        [
            {'name': 'transformer_gen', 'type': 'branch', 'branch_id': 'multiply_branch',
             'config': {'multiplier': 2, 'transform_type': 'multiply'}},
            {'name': 'transformer_gen', 'type': 'branch', 'branch_id': 'square_branch',
             'config': {'transform_type': 'square'}},
            {'name': 'merge', 'type': 'merge', 'strategy': 'select_best'}
        ]
    )
    
    initial_data = {'id': 1, 'value': 10}
    print(f"Branch-merge with initial data: {initial_data}")
    for result in composer.execute_branch_merge_pattern(branch_merge_pattern, initial_data):
        print(f"  Merged result: {result}")
    
    # Demo 3: Conditional Pattern
    print("\n❓ Demo 3: Conditional Pattern")
    print("-" * 30)
    
    def condition_func(data, spec):
        """Simple condition function for demo."""
        gen_name = spec['name']
        if gen_name == 'transformer_gen':
            return data.get('value', 0) > 5
        return True
    
    conditional_pattern = composer.create_composition_pattern(
        'demo_conditional',
        GeneratorCompositionPattern.CONDITIONAL,
        [
            {'name': 'data_gen', 'config': {'count': 2, 'start_value': 3}},
            {'name': 'transformer_gen', 'config': {'multiplier': 3}},
            {'name': 'filter_gen', 'config': {'filter_condition': 'greater_than', 'threshold': 50}}
        ]
    )
    
    test_data = {'value': 10}
    print(f"Conditional execution with data: {test_data}")
    for result in composer.execute_conditional_pattern(conditional_pattern, condition_func, test_data):
        print(f"  Conditional result: {result}")
    
    # Demo 4: Recursive Pattern
    print("\n🔄 Demo 4: Recursive Pattern")
    print("-" * 30)
    
    def base_case_condition(data):
        """Base case: stop when value is small enough."""
        return data.get('value', 0) < 2
    
    recursive_pattern = composer.create_recursive_pattern(
        'demo_recursive',
        base_case_condition,
        {'name': 'transformer_gen', 'config': {'multiplier': 0.5, 'transform_type': 'multiply'}}
    )
    
    recursive_data = {'value': 16}
    print(f"Recursive execution starting with: {recursive_data}")
    
    # Execute recursive pattern
    recursive_gen = composer.generator_registry['demo_recursive_recursive']({'max_depth': 5})
    for result in recursive_gen(recursive_data):
        print(f"  Recursive step: {result}")
    
    # Show stats
    print("\n📊 Composition Statistics")
    print("-" * 30)
    stats = composer.get_composition_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return composer


# =============================================================================
# COMPLEX EXAMPLE: PIPELINE PROCESSING
# =============================================================================

async def demo_pipeline_processing():
    """Demonstrate complex pipeline processing."""
    print("\n🏭 Pipeline Processing Demo")
    print("-" * 40)
    
    composer = AdvancedGeneratorComposer()
    
    # Register factories
    composer.register_generator_factory('data_gen', create_data_generator_factory())
    composer.register_generator_factory('transformer_gen', create_transformer_generator_factory())
    composer.register_generator_factory('filter_gen', create_filter_generator_factory())
    composer.register_generator_factory('aggregator_gen', create_aggregator_generator_factory())
    
    # Create pipeline pattern
    pipeline_pattern = composer.create_composition_pattern(
        'processing_pipeline',
        GeneratorCompositionPattern.PIPELINE,
        [
            {'name': 'transformer_gen', 'config': {'multiplier': 3, 'transform_type': 'multiply'}},
            {'name': 'filter_gen', 'config': {'filter_condition': 'greater_than', 'threshold': 100}},
            {'name': 'aggregator_gen', 'config': {'buffer_size': 2, 'aggregation_type': 'average'}}
        ]
    )
    
    # Create input stream
    def input_stream():
        for i in range(10):
            yield {
                'id': i,
                'value': 10 + (i * 5),
                'source': 'demo_stream'
            }
    
    print("Pipeline processing results:")
    pipeline_results = []
    for result in composer.execute_pipeline_pattern(pipeline_pattern, input_stream()):
        pipeline_results.append(result)
        print(f"  Input: {result['input']} -> Output: {result['output']}")
    
    print(f"\nProcessed {len(pipeline_results)} items through pipeline")
    
    return pipeline_results


if __name__ == "__main__":
    import asyncio
    
    async def main():
        await demo_generator_composition()
        await demo_pipeline_processing()
    
    asyncio.run(main())