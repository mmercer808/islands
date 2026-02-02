#!/usr/bin/env python3
"""
Persistent Context Iterator with Task Scheduler
===============================================

A system with a persistent iterator that maintains a rolling context window
of recent operations, staying alive during the main game loop and being
passed through the event queue. Includes a separate task scheduler that
manages threaded execution.

Core Features:
- Persistent iterator with last operation cache
- Rolling context window for temporal game state
- Iterator passed down through event queue execution
- Separate task scheduler with thread management
- Memory-aware context retention and cleanup

Author: Claude & Human Collaboration  
License: MIT
"""

import threading
import queue
import time
import weakref
from typing import Any, Dict, List, Optional, Callable, Iterator, Deque
from dataclasses import dataclass, field
from collections import deque, OrderedDict
from abc import ABC, abstractmethod
import uuid
import copy
from enum import Enum, auto
import logging


# ===== CONTEXT OPERATION TYPES =====

class OperationType(Enum):
    """Types of operations that can be cached in context"""
    GAME_ACTION = auto()
    STORY_EVENT = auto()
    OBJECT_INTERACTION = auto()
    WORLD_CHANGE = auto()
    PLAYER_INPUT = auto()
    SYSTEM_EVENT = auto()
    NARRATIVE_PROGRESSION = auto()


@dataclass
class ContextOperation:
    """A single operation stored in the context window"""
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: OperationType = OperationType.GAME_ACTION
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    target: str = ""
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Context linkage
    previous_operation_id: Optional[str] = None
    next_operation_id: Optional[str] = None
    related_operations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'operation_id': self.operation_id,
            'operation_type': self.operation_type.name,
            'timestamp': self.timestamp,
            'data': self.data,
            'source': self.source,
            'target': self.target,
            'result': self.result,
            'metadata': self.metadata,
            'previous_operation_id': self.previous_operation_id,
            'next_operation_id': self.next_operation_id,
            'related_operations': self.related_operations
        }


# ===== CONTEXT WINDOW MANAGER =====

class ContextWindow:
    """Manages a sliding window of recent operations with intelligent retention"""
    
    def __init__(self, max_size: int = 1000, retention_strategy: str = "temporal"):
        self.max_size = max_size
        self.retention_strategy = retention_strategy
        
        # Core storage
        self.operations: OrderedDict[str, ContextOperation] = OrderedDict()
        self.operation_timeline: Deque[str] = deque(maxlen=max_size)
        
        # Indexes for fast lookup
        self.operations_by_type: Dict[OperationType, List[str]] = {}
        self.operations_by_source: Dict[str, List[str]] = {}
        self.operations_by_timestamp: List[tuple] = []  # (timestamp, operation_id)
        
        # Context metadata
        self.window_start_time: float = time.time()
        self.total_operations_seen: int = 0
        self.evicted_operations_count: int = 0
        
        # Memory management
        self.memory_pressure_threshold: int = int(max_size * 0.8)
        self.cleanup_batch_size: int = int(max_size * 0.1)
    
    def add_operation(self, operation: ContextOperation) -> bool:
        """Add operation to context window"""
        self.total_operations_seen += 1
        
        # Link to previous operation
        if self.operation_timeline:
            last_op_id = self.operation_timeline[-1]
            if last_op_id in self.operations:
                last_op = self.operations[last_op_id]
                last_op.next_operation_id = operation.operation_id
                operation.previous_operation_id = last_op_id
        
        # Add to main storage
        self.operations[operation.operation_id] = operation
        self.operation_timeline.append(operation.operation_id)
        
        # Update indexes
        self._update_indexes(operation)
        
        # Check if we need to evict old operations
        if len(self.operations) > self.max_size:
            self._evict_operations()
        
        # Check memory pressure
        if len(self.operations) > self.memory_pressure_threshold:
            self._handle_memory_pressure()
        
        return True
    
    def get_operation(self, operation_id: str) -> Optional[ContextOperation]:
        """Get operation by ID"""
        return self.operations.get(operation_id)
    
    def get_recent_operations(self, count: int = 10) -> List[ContextOperation]:
        """Get most recent operations"""
        recent_ids = list(self.operation_timeline)[-count:]
        return [self.operations[op_id] for op_id in recent_ids if op_id in self.operations]
    
    def get_operations_by_type(self, operation_type: OperationType, count: int = None) -> List[ContextOperation]:
        """Get operations of specific type"""
        type_ops = self.operations_by_type.get(operation_type, [])
        if count:
            type_ops = type_ops[-count:]
        return [self.operations[op_id] for op_id in type_ops if op_id in self.operations]
    
    def get_operations_in_timeframe(self, start_time: float, end_time: float) -> List[ContextOperation]:
        """Get operations within time range"""
        result = []
        for timestamp, op_id in self.operations_by_timestamp:
            if start_time <= timestamp <= end_time and op_id in self.operations:
                result.append(self.operations[op_id])
        return result
    
    def get_operation_chain(self, operation_id: str, direction: str = "both") -> List[ContextOperation]:
        """Get chain of linked operations"""
        if operation_id not in self.operations:
            return []
        
        chain = []
        current_op = self.operations[operation_id]
        
        # Go backwards
        if direction in ["both", "backward"]:
            backward_chain = []
            while current_op.previous_operation_id and current_op.previous_operation_id in self.operations:
                prev_op = self.operations[current_op.previous_operation_id]
                backward_chain.append(prev_op)
                current_op = prev_op
            chain.extend(reversed(backward_chain))
        
        # Add current operation
        chain.append(self.operations[operation_id])
        
        # Go forwards
        if direction in ["both", "forward"]:
            current_op = self.operations[operation_id]
            while current_op.next_operation_id and current_op.next_operation_id in self.operations:
                next_op = self.operations[current_op.next_operation_id]
                chain.append(next_op)
                current_op = next_op
        
        return chain
    
    def _update_indexes(self, operation: ContextOperation):
        """Update lookup indexes"""
        # Type index
        if operation.operation_type not in self.operations_by_type:
            self.operations_by_type[operation.operation_type] = []
        self.operations_by_type[operation.operation_type].append(operation.operation_id)
        
        # Source index
        if operation.source:
            if operation.source not in self.operations_by_source:
                self.operations_by_source[operation.source] = []
            self.operations_by_source[operation.source].append(operation.operation_id)
        
        # Timestamp index
        self.operations_by_timestamp.append((operation.timestamp, operation.operation_id))
        self.operations_by_timestamp.sort(key=lambda x: x[0])
    
    def _evict_operations(self):
        """Remove old operations based on retention strategy"""
        operations_to_remove = len(self.operations) - self.max_size + self.cleanup_batch_size
        
        if self.retention_strategy == "temporal":
            # Remove oldest operations
            for _ in range(operations_to_remove):
                if self.operation_timeline:
                    oldest_id = self.operation_timeline.popleft()
                    self._remove_operation(oldest_id)
        
        elif self.retention_strategy == "importance":
            # Remove least important operations (based on operation type priority)
            importance_order = [
                OperationType.SYSTEM_EVENT,
                OperationType.GAME_ACTION,
                OperationType.PLAYER_INPUT,
                OperationType.OBJECT_INTERACTION,
                OperationType.WORLD_CHANGE,
                OperationType.STORY_EVENT,
                OperationType.NARRATIVE_PROGRESSION
            ]
            
            # Find operations to remove based on importance
            for op_type in importance_order:
                if operations_to_remove <= 0:
                    break
                
                type_ops = self.operations_by_type.get(op_type, [])
                for op_id in type_ops[:operations_to_remove]:
                    if op_id in self.operations:
                        self._remove_operation(op_id)
                        operations_to_remove -= 1
    
    def _remove_operation(self, operation_id: str):
        """Remove operation and update indexes"""
        if operation_id not in self.operations:
            return
        
        operation = self.operations[operation_id]
        
        # Remove from main storage
        del self.operations[operation_id]
        self.evicted_operations_count += 1
        
        # Update linking
        if operation.previous_operation_id and operation.previous_operation_id in self.operations:
            prev_op = self.operations[operation.previous_operation_id]
            prev_op.next_operation_id = operation.next_operation_id
        
        if operation.next_operation_id and operation.next_operation_id in self.operations:
            next_op = self.operations[operation.next_operation_id]
            next_op.previous_operation_id = operation.previous_operation_id
        
        # Clean up indexes
        self._cleanup_indexes(operation)
    
    def _cleanup_indexes(self, operation: ContextOperation):
        """Clean up index entries for removed operation"""
        # Type index
        if operation.operation_type in self.operations_by_type:
            if operation.operation_id in self.operations_by_type[operation.operation_type]:
                self.operations_by_type[operation.operation_type].remove(operation.operation_id)
        
        # Source index
        if operation.source and operation.source in self.operations_by_source:
            if operation.operation_id in self.operations_by_source[operation.source]:
                self.operations_by_source[operation.source].remove(operation.operation_id)
        
        # Timestamp index
        self.operations_by_timestamp = [
            (ts, op_id) for ts, op_id in self.operations_by_timestamp 
            if op_id != operation.operation_id
        ]
    
    def _handle_memory_pressure(self):
        """Handle memory pressure by aggressive cleanup"""
        print(f"⚠️ Context window memory pressure detected. Operations: {len(self.operations)}")
        
        # Force eviction of older operations
        self._evict_operations()
        
        # Clean up empty index entries
        self.operations_by_type = {
            op_type: [op_id for op_id in op_ids if op_id in self.operations]
            for op_type, op_ids in self.operations_by_type.items()
        }
        
        self.operations_by_source = {
            source: [op_id for op_id in op_ids if op_id in self.operations]
            for source, op_ids in self.operations_by_source.items()
        }
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context window state"""
        return {
            'current_size': len(self.operations),
            'max_size': self.max_size,
            'total_operations_seen': self.total_operations_seen,
            'evicted_operations': self.evicted_operations_count,
            'window_age_seconds': time.time() - self.window_start_time,
            'operations_by_type': {
                op_type.name: len(op_ids) 
                for op_type, op_ids in self.operations_by_type.items()
            },
            'retention_strategy': self.retention_strategy,
            'memory_pressure': len(self.operations) > self.memory_pressure_threshold
        }


# ===== PERSISTENT CONTEXT ITERATOR =====

class PersistentContextIterator:
    """Iterator that maintains context and stays alive during main game loop"""
    
    def __init__(self, name: str = "GameContextIterator", context_window_size: int = 1000):
        self.iterator_id = str(uuid.uuid4())
        self.name = name
        self.is_alive = True
        self.creation_time = time.time()
        
        # Context management
        self.context_window = ContextWindow(max_size=context_window_size)
        self.current_operation: Optional[ContextOperation] = None
        self.iteration_count = 0
        
        # Iterator state
        self.current_position = 0
        self.last_yield_time = time.time()
        self.total_yield_count = 0
        
        # Event queue integration
        self.event_queue_references: List[weakref.ReferenceType] = []
        self.processing_stack: List[Dict[str, Any]] = []
        
        # Game loop integration
        self.main_loop_active = False
        self.loop_cycle_count = 0
        self.last_loop_time = time.time()
        
        # Performance tracking
        self.performance_metrics = {
            'operations_per_second': 0.0,
            'average_processing_time': 0.0,
            'memory_usage_mb': 0.0,
            'context_hits': 0,
            'context_misses': 0
        }
    
    def register_with_event_queue(self, event_queue):
        """Register this iterator with an event queue"""
        self.event_queue_references.append(weakref.ref(event_queue))
    
    def start_main_loop_integration(self):
        """Start integration with main game loop"""
        self.main_loop_active = True
        self.last_loop_time = time.time()
        print(f"[OK] Iterator {self.name} integrated with main game loop")
    
    def stop_main_loop_integration(self):
        """Stop integration with main game loop"""
        self.main_loop_active = False
        print(f"[STOP] Iterator {self.name} disconnected from main game loop")
    
    def add_operation(self, operation_type: OperationType, data: Dict[str, Any], 
                     source: str = "", target: str = "", result: Any = None) -> ContextOperation:
        """Add new operation to context"""
        operation = ContextOperation(
            operation_type=operation_type,
            data=data,
            source=source,
            target=target,
            result=result,
            metadata={
                'iterator_id': self.iterator_id,
                'iteration_count': self.iteration_count,
                'loop_cycle': self.loop_cycle_count
            }
        )
        
        self.context_window.add_operation(operation)
        self.current_operation = operation
        
        # Update performance metrics
        self._update_performance_metrics()
        
        return operation
    
    def get_last_operations(self, count: int = 5) -> List[ContextOperation]:
        """Get last N operations from context cache"""
        return self.context_window.get_recent_operations(count)
    
    def get_operations_since(self, timestamp: float) -> List[ContextOperation]:
        """Get all operations since given timestamp"""
        return self.context_window.get_operations_in_timeframe(timestamp, time.time())
    
    def find_related_operations(self, operation_id: str) -> List[ContextOperation]:
        """Find operations related to given operation"""
        return self.context_window.get_operation_chain(operation_id)
    
    def query_context(self, **filters) -> List[ContextOperation]:
        """Query context with filters"""
        results = []
        
        # Filter by operation type
        if 'operation_type' in filters:
            results = self.context_window.get_operations_by_type(filters['operation_type'])
        else:
            results = self.context_window.get_recent_operations(1000)  # Get all
        
        # Apply additional filters
        if 'source' in filters:
            results = [op for op in results if op.source == filters['source']]
        
        if 'target' in filters:
            results = [op for op in results if op.target == filters['target']]
        
        if 'after_time' in filters:
            results = [op for op in results if op.timestamp > filters['after_time']]
        
        if 'data_contains' in filters:
            key, value = filters['data_contains']
            results = [op for op in results if op.data.get(key) == value]
        
        # Update metrics
        if results:
            self.performance_metrics['context_hits'] += 1
        else:
            self.performance_metrics['context_misses'] += 1
        
        return results
    
    def push_processing_context(self, context: Dict[str, Any]):
        """Push processing context onto stack"""
        context['timestamp'] = time.time()
        context['iterator_id'] = self.iterator_id
        self.processing_stack.append(context)
    
    def pop_processing_context(self) -> Optional[Dict[str, Any]]:
        """Pop processing context from stack"""
        if self.processing_stack:
            return self.processing_stack.pop()
        return None
    
    def get_current_processing_context(self) -> Optional[Dict[str, Any]]:
        """Get current processing context without popping"""
        if self.processing_stack:
            return self.processing_stack[-1]
        return None
    
    def main_loop_cycle(self) -> Dict[str, Any]:
        """Called each main game loop cycle"""
        if not self.main_loop_active:
            return {}
        
        self.loop_cycle_count += 1
        current_time = time.time()
        cycle_duration = current_time - self.last_loop_time
        self.last_loop_time = current_time
        
        # Add loop cycle operation to context
        cycle_data = {
            'cycle_number': self.loop_cycle_count,
            'cycle_duration': cycle_duration,
            'active_event_queues': len([ref for ref in self.event_queue_references if ref() is not None]),
            'processing_stack_depth': len(self.processing_stack)
        }
        
        cycle_operation = self.add_operation(
            OperationType.SYSTEM_EVENT,
            cycle_data,
            source="main_game_loop",
            target="context_iterator"
        )
        
        return {
            'operation_id': cycle_operation.operation_id,
            'cycle_number': self.loop_cycle_count,
            'context_size': len(self.context_window.operations),
            'processing_stack_depth': len(self.processing_stack)
        }
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        current_time = time.time()
        uptime = current_time - self.creation_time
        
        if uptime > 0:
            self.performance_metrics['operations_per_second'] = (
                self.context_window.total_operations_seen / uptime
            )
        
        # Estimate memory usage (rough calculation)
        operation_size = len(self.context_window.operations) * 1024  # Rough estimate
        self.performance_metrics['memory_usage_mb'] = operation_size / (1024 * 1024)
    
    def __iter__(self):
        """Iterator interface"""
        return self
    
    def __next__(self):
        """Iterator next method - yields current context state"""
        if not self.is_alive:
            raise StopIteration
        
        self.iteration_count += 1
        self.total_yield_count += 1
        self.last_yield_time = time.time()
        
        # Create iteration result with current context
        iteration_result = {
            'iteration_id': str(uuid.uuid4()),
            'iteration_count': self.iteration_count,
            'timestamp': self.last_yield_time,
            'current_operation': self.current_operation,
            'recent_operations': self.get_last_operations(5),
            'context_summary': self.context_window.get_context_summary(),
            'processing_context': self.get_current_processing_context(),
            'performance_metrics': self.performance_metrics.copy(),
            'iterator_state': {
                'is_alive': self.is_alive,
                'main_loop_active': self.main_loop_active,
                'loop_cycle_count': self.loop_cycle_count
            }
        }
        
        return iteration_result
    
    def shutdown(self):
        """Shutdown the iterator"""
        self.is_alive = False
        self.main_loop_active = False
        print(f"🔒 Iterator {self.name} shutdown. Total operations: {self.context_window.total_operations_seen}")


# ===== TASK SCHEDULER SYSTEM =====

@dataclass
class ScheduledTask:
    """A task scheduled for execution"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    task_function: Callable = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Scheduling
    priority: int = 5  # Lower number = higher priority
    scheduled_time: float = field(default_factory=time.time)
    execution_timeout: float = 30.0
    max_retries: int = 3
    retry_count: int = 0
    
    # State
    status: str = "pending"  # pending, running, completed, failed, cancelled
    thread_id: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    
    # Context integration
    requires_context_iterator: bool = False
    context_iterator_id: Optional[str] = None


class TaskScheduler:
    """Manages scheduled tasks and thread execution"""
    
    def __init__(self, max_worker_threads: int = 10):
        self.scheduler_id = str(uuid.uuid4())
        self.max_worker_threads = max_worker_threads
        
        # Task management
        self.pending_tasks: queue.PriorityQueue = queue.PriorityQueue()
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: Dict[str, ScheduledTask] = {}
        self.failed_tasks: Dict[str, ScheduledTask] = {}
        
        # Thread management
        self.worker_threads: List[threading.Thread] = []
        self.thread_registry: Dict[int, Dict[str, Any]] = {}
        self.is_running = False
        
        # Context iterator integration
        self.context_iterator: Optional[PersistentContextIterator] = None
        
        # Performance tracking
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        
        # Shutdown event
        self.shutdown_event = threading.Event()
    
    def set_context_iterator(self, context_iterator: PersistentContextIterator):
        """Set the context iterator for tasks that need it"""
        self.context_iterator = context_iterator
        print(f"[OK] Task scheduler linked to context iterator: {context_iterator.name}")
    
    def schedule_task(self, task_function: Callable, args: tuple = (), kwargs: Dict[str, Any] = None,
                     priority: int = 5, delay_seconds: float = 0.0, name: str = "",
                     requires_context: bool = False, execution_timeout: float = 30.0) -> str:
        """Schedule a task for execution"""
        
        task = ScheduledTask(
            name=name or task_function.__name__,
            task_function=task_function,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            scheduled_time=time.time() + delay_seconds,
            execution_timeout=execution_timeout,
            requires_context_iterator=requires_context,
            context_iterator_id=self.context_iterator.iterator_id if self.context_iterator else None
        )
        
        # Add to priority queue (priority, scheduled_time, task)
        self.pending_tasks.put((priority, task.scheduled_time, task))
        
        print(f"📅 Scheduled task: {task.name} (ID: {task.task_id[:8]}, Priority: {priority})")
        
        return task.task_id
    
    def start_scheduler(self):
        """Start the task scheduler"""
        if self.is_running:
            print("Task scheduler already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start worker threads
        for i in range(self.max_worker_threads):
            worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker_{i}",
                daemon=True
            )
            worker_thread.start()
            self.worker_threads.append(worker_thread)
            
            # Register thread
            self.thread_registry[worker_thread.ident] = {
                'thread_name': worker_thread.name,
                'start_time': time.time(),
                'tasks_executed': 0,
                'current_task': None
            }
        
        print(f"🚀 Task scheduler started with {len(self.worker_threads)} worker threads")
    
    def stop_scheduler(self, wait_for_completion: bool = True):
        """Stop the task scheduler"""
        print("🛑 Stopping task scheduler...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        if wait_for_completion:
            # Wait for all worker threads to finish
            for thread in self.worker_threads:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    print(f"⚠️ Worker thread {thread.name} did not shutdown cleanly")
        
        print(f"✅ Task scheduler stopped. Completed: {self.tasks_completed}, Failed: {self.tasks_failed}")
    
    def _worker_loop(self):
        """Main worker thread loop"""
        thread_id = threading.current_thread().ident
        thread_info = self.thread_registry[thread_id]
        
        print(f"👷 Worker thread {threading.current_thread().name} started")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Get next task (with timeout)
                try:
                    priority, scheduled_time, task = self.pending_tasks.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if it's time to execute
                current_time = time.time()
                if current_time < scheduled_time:
                    # Put task back and wait
                    self.pending_tasks.put((priority, scheduled_time, task))
                    time.sleep(0.1)
                    continue
                
                # Execute the task
                self._execute_task(task, thread_info)
                
                # Mark task as done
                self.pending_tasks.task_done()
                
            except Exception as e:
                print(f"❌ Worker thread error: {e}")
                time.sleep(1.0)
        
        print(f"👷 Worker thread {threading.current_thread().name} stopped")
    
    def _execute_task(self, task: ScheduledTask, thread_info: Dict[str, Any]):
        """Execute a single task"""
        task.status = "running"
        task.thread_id = threading.current_thread().ident
        task.start_time = time.time()
        
        # Update thread info
        thread_info['current_task'] = task.task_id
        thread_info['tasks_executed'] += 1
        
        # Add to running tasks
        self.running_tasks[task.task_id] = task
        
        try:
            # Prepare execution context
            execution_kwargs = task.kwargs.copy()
            
            # Add context iterator if required
            if task.requires_context_iterator and self.context_iterator:
                execution_kwargs['context_iterator'] = self.context_iterator
                
                # Add task execution to context
                self.context_iterator.add_operation(
                    OperationType.SYSTEM_EVENT,
                    {
                        'task_id': task.task_id,
                        'task_name': task.name,
                        'thread_id': task.thread_id,
                        'priority': task.priority
                    },
                    source="task_scheduler",
                    target=task.name
                )
            
            print(f"⚡ Executing task: {task.name} (Thread: {threading.current_thread().name})")
            
            # Execute the task function
            start_exec_time = time.time()
            result = task.task_function(*task.args, **execution_kwargs)
            execution_time = time.time() - start_exec_time
            
            # Task completed successfully
            task.status = "completed"
            task.result = result
            task.end_time = time.time()
            
            # Move to completed tasks
            self.running_tasks.pop(task.task_id, None)
            self.completed_tasks[task.task_id] = task
            
            # Update statistics
            self.tasks_completed += 1
            self.total_execution_time += execution_time
            
            print(f"✅ Task completed: {task.name} (Time: {execution_time:.3f}s)")
            
        except Exception as e:
            # Task failed
            task.status = "failed"
            task.error = e
            task.end_time = time.time()
            
            # Move to failed tasks
            self.running_tasks.pop(task.task_id, None)
            self.failed_tasks[task.task_id] = task
            
            # Update statistics
            self.tasks_failed += 1
            
            print(f"❌ Task failed: {task.name} - {str(e)}")
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = "pending"
                task.thread_id = None
                task.start_time = None
                task.end_time = None
                task.error = None
                
                # Reschedule with delay
                retry_delay = 2.0 ** task.retry_count  # Exponential backoff
                task.scheduled_time = time.time() + retry_delay
                
                self.pending_tasks.put((task.priority + 1, task.scheduled_time, task))  # Lower priority for retry
                print(f"🔄 Retrying task: {task.name} (Attempt {task.retry_count + 1}/{task.max_retries + 1})")
                
                # Remove from failed tasks since we're retrying
                self.failed_tasks.pop(task.task_id, None)
        
        finally:
            # Clean up thread info
            thread_info['current_task'] = None
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get comprehensive scheduler statistics"""
        return {
            'scheduler_id': self.scheduler_id,
            'is_running': self.is_running,
            'worker_threads': len(self.worker_threads),
            'pending_tasks': self.pending_tasks.qsize(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'total_completed': self.tasks_completed,
            'total_failed': self.tasks_failed,
            'average_execution_time': (
                self.total_execution_time / max(1, self.tasks_completed)
            ),
            'thread_registry': {
                thread_id: {
                    'name': info['thread_name'],
                    'uptime': time.time() - info['start_time'],
                    'tasks_executed': info['tasks_executed'],
                    'current_task': info['current_task']
                }
                for thread_id, info in self.thread_registry.items()
            },
            'context_iterator_linked': self.context_iterator is not None
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        # Note: This is a simplified implementation
        # In practice, you'd need more sophisticated task cancellation
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.status = "cancelled"
            return True
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        # Check all task collections
        for task_collection in [self.running_tasks, self.completed_tasks, self.failed_tasks]:
            if task_id in task_collection:
                task = task_collection[task_id]
                return {
                    'task_id': task.task_id,
                    'name': task.name,
                    'status': task.status,
                    'priority': task.priority,
                    'retry_count': task.retry_count,
                    'thread_id': task.thread_id,
                    'start_time': task.start_time,
                    'end_time': task.end_time,
                    'execution_time': (
                        (task.end_time - task.start_time) 
                        if task.start_time and task.end_time else None
                    ),
                    'result': task.result,
                    'error': str(task.error) if task.error else None
                }
        return None


# ===== GAME LOOP INTEGRATION =====

class GameLoopIntegrator:
    """Integrates persistent iterator and task scheduler with main game loop"""
    
    def __init__(self, target_fps: int = 60):
        self.integrator_id = str(uuid.uuid4())
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        # Core components
        self.context_iterator: Optional[PersistentContextIterator] = None
        self.task_scheduler: Optional[TaskScheduler] = None
        
        # Game loop state
        self.is_running = False
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
        # Performance tracking
        self.actual_fps = 0.0
        self.frame_times = deque(maxlen=60)  # Track last 60 frames
        self.slow_frames = 0
        self.dropped_frames = 0
        
        # Event queue integration
        self.event_queues: List[Any] = []  # Will hold event queue references
    
    def setup(self, context_iterator: PersistentContextIterator, task_scheduler: TaskScheduler):
        """Setup the integrator with core components"""
        self.context_iterator = context_iterator
        self.task_scheduler = task_scheduler
        
        # Link components
        task_scheduler.set_context_iterator(context_iterator)
        context_iterator.start_main_loop_integration()
        
        print(f"[OK] Game loop integrator setup complete")
        print(f"   Context Iterator: {context_iterator.name}")
        print(f"   Task Scheduler: {task_scheduler.scheduler_id[:8]}")
    
    def add_event_queue(self, event_queue):
        """Add event queue to be processed in game loop"""
        self.event_queues.append(event_queue)
        if self.context_iterator:
            self.context_iterator.register_with_event_queue(event_queue)
        print(f"📋 Added event queue to game loop integration")
    
    def start_game_loop(self):
        """Start the main game loop"""
        if not self.context_iterator or not self.task_scheduler:
            raise RuntimeError("Must call setup() before starting game loop")
        
        self.is_running = True
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
        # Start task scheduler
        self.task_scheduler.start_scheduler()
        
        print(f"🎮 Starting main game loop (Target FPS: {self.target_fps})")
        
        try:
            while self.is_running:
                self._game_loop_frame()
        except KeyboardInterrupt:
            print(f"\n⏹️ Game loop interrupted by user")
        finally:
            self._shutdown()
    
    def stop_game_loop(self):
        """Stop the main game loop"""
        self.is_running = False
    
    def _game_loop_frame(self):
        """Execute a single frame of the game loop"""
        frame_start_time = time.time()
        self.frame_count += 1
        
        # Calculate frame timing
        frame_delta = frame_start_time - self.last_frame_time
        self.frame_times.append(frame_delta)
        self.last_frame_time = frame_start_time
        
        # Update FPS calculation
        if len(self.frame_times) >= 10:
            avg_frame_time = sum(list(self.frame_times)[-10:]) / 10
            self.actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        try:
            # 1. Context iterator main loop cycle
            if self.context_iterator:
                cycle_result = self.context_iterator.main_loop_cycle()
                
                # Add frame timing to context
                self.context_iterator.add_operation(
                    OperationType.SYSTEM_EVENT,
                    {
                        'frame_number': self.frame_count,
                        'frame_delta': frame_delta,
                        'current_fps': self.actual_fps,
                        'target_fps': self.target_fps
                    },
                    source="game_loop",
                    target="frame_timing"
                )
            
            # 2. Process event queues with context
            for event_queue in self.event_queues:
                if hasattr(event_queue, 'is_running') and event_queue.is_running:
                    # Pass iterator through event queue processing
                    if hasattr(event_queue, 'process_with_context'):
                        event_queue.process_with_context(self.context_iterator)
            
            # 3. Schedule frame-based tasks
            if self.frame_count % 60 == 0:  # Every second at 60fps
                self._schedule_periodic_tasks()
            
            # 4. Performance monitoring
            frame_time = time.time() - frame_start_time
            
            if frame_time > self.frame_time * 1.5:  # Frame took 50% longer than target
                self.slow_frames += 1
                
                # Add slow frame to context
                if self.context_iterator:
                    self.context_iterator.add_operation(
                        OperationType.SYSTEM_EVENT,
                        {
                            'frame_time': frame_time,
                            'target_frame_time': self.frame_time,
                            'performance_impact': frame_time / self.frame_time
                        },
                        source="performance_monitor",
                        target="slow_frame_detected"
                    )
            
            # 5. Frame rate limiting
            sleep_time = self.frame_time - frame_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self.dropped_frames += 1
        
        except Exception as e:
            print(f"❌ Game loop frame error: {e}")
            if self.context_iterator:
                self.context_iterator.add_operation(
                    OperationType.SYSTEM_EVENT,
                    {'error': str(e), 'frame': self.frame_count},
                    source="game_loop",
                    target="error_handler"
                )
    
    def _schedule_periodic_tasks(self):
        """Schedule periodic maintenance tasks"""
        if not self.task_scheduler:
            return
        
        # Context cleanup task
        self.task_scheduler.schedule_task(
            self._context_cleanup_task,
            name="context_cleanup",
            priority=8,  # Low priority
            requires_context=True
        )
        
        # Performance monitoring task
        self.task_scheduler.schedule_task(
            self._performance_monitoring_task,
            name="performance_monitor",
            priority=7,
            requires_context=True
        )
    
    def _context_cleanup_task(self, context_iterator: PersistentContextIterator):
        """Task to clean up context window"""
        context_summary = context_iterator.context_window.get_context_summary()
        
        if context_summary['memory_pressure']:
            print(f"🧹 Performing context cleanup due to memory pressure")
            # Force cleanup of old operations
            context_iterator.context_window._handle_memory_pressure()
            
            # Add cleanup operation to context
            context_iterator.add_operation(
                OperationType.SYSTEM_EVENT,
                {
                    'cleanup_reason': 'memory_pressure',
                    'operations_before': context_summary['current_size'],
                    'operations_after': len(context_iterator.context_window.operations)
                },
                source="context_cleanup_task",
                target="context_window"
            )
    
    def _performance_monitoring_task(self, context_iterator: PersistentContextIterator):
        """Task to monitor and log performance metrics"""
        stats = {
            'current_fps': self.actual_fps,
            'target_fps': self.target_fps,
            'frame_count': self.frame_count,
            'slow_frames': self.slow_frames,
            'dropped_frames': self.dropped_frames,
            'uptime_seconds': time.time() - self.start_time,
            'context_operations': len(context_iterator.context_window.operations),
            'task_scheduler_stats': self.task_scheduler.get_scheduler_stats() if self.task_scheduler else {}
        }
        
        # Add performance stats to context
        context_iterator.add_operation(
            OperationType.SYSTEM_EVENT,
            stats,
            source="performance_monitor",
            target="performance_stats"
        )
        
        # Log performance if there are issues
        if self.actual_fps < self.target_fps * 0.8:  # Running at less than 80% target FPS
            print(f"⚠️ Performance warning: FPS {self.actual_fps:.1f}/{self.target_fps}")
    
    def _shutdown(self):
        """Shutdown all components"""
        print(f"🔄 Shutting down game loop integrator...")
        
        # Stop context iterator
        if self.context_iterator:
            self.context_iterator.stop_main_loop_integration()
            self.context_iterator.shutdown()
        
        # Stop task scheduler
        if self.task_scheduler:
            self.task_scheduler.stop_scheduler()
        
        # Stop event queues
        for event_queue in self.event_queues:
            if hasattr(event_queue, 'stop_iteration'):
                event_queue.stop_iteration()
        
        print(f"✅ Game loop integrator shutdown complete")
        print(f"   Total frames: {self.frame_count}")
        print(f"   Average FPS: {self.actual_fps:.1f}")
        print(f"   Slow frames: {self.slow_frames}")
        print(f"   Dropped frames: {self.dropped_frames}")


# ===== DEMONSTRATION SYSTEM =====

def create_demo_tasks():
    """Create demonstration tasks that use context"""
    
    def story_progression_task(context_iterator: PersistentContextIterator):
        """Task that progresses story based on context"""
        print("📖 Story progression task executing...")
        
        # Query context for recent player actions
        recent_actions = context_iterator.query_context(
            operation_type=OperationType.PLAYER_INPUT,
            after_time=time.time() - 10.0  # Last 10 seconds
        )
        
        # Add story progression based on actions
        if recent_actions:
            story_data = {
                'action_count': len(recent_actions),
                'story_trigger': 'player_activity_detected',
                'progression_type': 'dynamic'
            }
        else:
            story_data = {
                'action_count': 0,
                'story_trigger': 'idle_progression',
                'progression_type': 'ambient'
            }
        
        context_iterator.add_operation(
            OperationType.STORY_EVENT,
            story_data,
            source="story_progression_task",
            target="narrative_system"
        )
        
        return {'status': 'story_progressed', 'data': story_data}
    
    def world_update_task(context_iterator: PersistentContextIterator):
        """Task that updates world state"""
        print("🌍 World update task executing...")
        
        # Find recent world changes
        world_changes = context_iterator.query_context(
            operation_type=OperationType.WORLD_CHANGE
        )
        
        # Simulate world state update
        world_data = {
            'world_changes_count': len(world_changes),
            'world_time': time.time(),
            'environment_update': 'seasonal_progression'
        }
        
        context_iterator.add_operation(
            OperationType.WORLD_CHANGE,
            world_data,
            source="world_update_task",
            target="world_state_manager"
        )
        
        time.sleep(0.1)  # Simulate work
        return {'status': 'world_updated', 'changes': len(world_changes)}
    
    def ai_behavior_task(context_iterator: PersistentContextIterator):
        """Task that updates AI behavior based on context"""
        print("🤖 AI behavior task executing...")
        
        # Get recent object interactions
        interactions = context_iterator.query_context(
            operation_type=OperationType.OBJECT_INTERACTION
        )
        
        ai_data = {
            'interaction_count': len(interactions),
            'ai_state': 'adaptive',
            'behavior_mode': 'responsive' if interactions else 'passive'
        }
        
        context_iterator.add_operation(
            OperationType.SYSTEM_EVENT,
            ai_data,
            source="ai_behavior_task",
            target="ai_system"
        )
        
        return {'status': 'ai_updated', 'mode': ai_data['behavior_mode']}
    
    return [story_progression_task, world_update_task, ai_behavior_task]


def simulate_game_events(context_iterator: PersistentContextIterator):
    """Simulate various game events to populate context"""
    
    events = [
        # Player actions
        (OperationType.PLAYER_INPUT, {'action': 'move', 'direction': 'north'}, "player", "world"),
        (OperationType.PLAYER_INPUT, {'action': 'interact', 'object': 'blue_jewel'}, "player", "blue_jewel"),
        
        # Object interactions
        (OperationType.OBJECT_INTERACTION, {'interaction': 'pick_up', 'item': 'blue_jewel'}, "player", "blue_jewel"),
        (OperationType.OBJECT_INTERACTION, {'interaction': 'equip', 'item': 'sword'}, "player", "inventory"),
        
        # World changes
        (OperationType.WORLD_CHANGE, {'change': 'weather', 'new_state': 'winter'}, "blue_jewel", "world"),
        (OperationType.WORLD_CHANGE, {'change': 'npc_spawn', 'npc': 'lady_of_river'}, "story_system", "world"),
        
        # Story events
        (OperationType.STORY_EVENT, {'event': 'quest_start', 'quest': 'winter_guardian'}, "story_system", "player"),
        (OperationType.STORY_EVENT, {'event': 'power_awakening', 'power': 'ice_magic'}, "blue_jewel", "player"),
    ]
    
    for op_type, data, source, target in events:
        context_iterator.add_operation(op_type, data, source, target)
        time.sleep(0.2)  # Space out events


def demonstrate_system():
    """Demonstrate the complete persistent context iterator and task scheduler system"""
    
    print("🎯 === PERSISTENT CONTEXT ITERATOR & TASK SCHEDULER DEMO ===\n")
    
    # Create core components
    print("🔧 Creating core components...")
    
    context_iterator = PersistentContextIterator(
        name="GameWorldIterator",
        context_window_size=100
    )
    
    task_scheduler = TaskScheduler(max_worker_threads=4)
    
    game_loop = GameLoopIntegrator(target_fps=10)  # Lower FPS for demo
    
    print(f"✅ Components created:")
    print(f"   Context Iterator: {context_iterator.name}")
    print(f"   Task Scheduler: {task_scheduler.scheduler_id[:8]}")
    print(f"   Game Loop: {game_loop.integrator_id[:8]}")
    
    # Setup integration
    print(f"\n🔗 Setting up integration...")
    game_loop.setup(context_iterator, task_scheduler)
    
    # Create and schedule demo tasks
    print(f"\n📅 Scheduling demo tasks...")
    demo_tasks = create_demo_tasks()
    
    scheduled_task_ids = []
    for i, task_func in enumerate(demo_tasks):
        task_id = task_scheduler.schedule_task(
            task_func,
            name=f"demo_task_{i}",
            priority=5,
            delay_seconds=i * 2.0,  # Stagger task execution
            requires_context=True,
            execution_timeout=10.0
        )
        scheduled_task_ids.append(task_id)
    
    # Simulate some initial game events
    print(f"\n🎮 Simulating initial game events...")
    simulate_game_events(context_iterator)
    
    # Run a few iterations manually to show iterator behavior
    print(f"\n🔄 Demonstrating iterator behavior...")
    
    print("Manual iterator usage:")
    iteration_count = 0
    for iteration_result in context_iterator:
        iteration_count += 1
        print(f"  Iteration {iteration_count}:")
        print(f"    Context operations: {len(iteration_result['recent_operations'])}")
        print(f"    Loop cycles: {iteration_result['iterator_state']['loop_cycle_count']}")
        print(f"    Performance: {iteration_result['performance_metrics']['operations_per_second']:.1f} ops/sec")
        
        if iteration_count >= 3:  # Show just a few iterations
            break
    
    # Start brief game loop simulation
    print(f"\n🎮 Starting brief game loop simulation...")
    
    # Run game loop for a short time
    def run_brief_simulation():
        start_time = time.time()
        frame_count = 0
        max_frames = 30  # About 3 seconds at 10 FPS
        
        while frame_count < max_frames:
            game_loop._game_loop_frame()
            frame_count += 1
            
            # Add some events during simulation
            if frame_count % 10 == 0:
                context_iterator.add_operation(
                    OperationType.GAME_ACTION,
                    {'action': f'simulation_event_{frame_count}', 'frame': frame_count},
                    source="simulation",
                    target="game_world"
                )
        
        return time.time() - start_time
    
    # Start task scheduler for simulation
    task_scheduler.start_scheduler()
    
    try:
        simulation_time = run_brief_simulation()
        print(f"✅ Simulation completed in {simulation_time:.2f} seconds")
    finally:
        # Clean shutdown
        print(f"\n🧹 Cleaning up...")
        task_scheduler.stop_scheduler()
        context_iterator.shutdown()
    
    # Show final statistics
    print(f"\n📊 Final Statistics:")
    
    # Context statistics
    context_summary = context_iterator.context_window.get_context_summary()
    print(f"   Context Window:")
    print(f"     Operations stored: {context_summary['current_size']}")
    print(f"     Total operations seen: {context_summary['total_operations_seen']}")
    print(f"     Operations by type: {context_summary['operations_by_type']}")
    print(f"     Window age: {context_summary['window_age_seconds']:.1f}s")
    
    # Task scheduler statistics
    scheduler_stats = task_scheduler.get_scheduler_stats()
    print(f"   Task Scheduler:")
    print(f"     Tasks completed: {scheduler_stats['total_completed']}")
    print(f"     Tasks failed: {scheduler_stats['total_failed']}")
    print(f"     Average execution time: {scheduler_stats['average_execution_time']:.3f}s")
    print(f"     Worker threads: {scheduler_stats['worker_threads']}")
    
    # Game loop statistics
    print(f"   Game Loop:")
    print(f"     Frames processed: {game_loop.frame_count}")
    print(f"     Average FPS: {game_loop.actual_fps:.1f}")
    print(f"     Slow frames: {game_loop.slow_frames}")
    
    # Show some context operations
    print(f"\n📋 Recent Context Operations:")
    recent_ops = context_iterator.get_last_operations(5)
    for i, op in enumerate(recent_ops):
        print(f"   {i+1}. {op.operation_type.name}: {op.source} -> {op.target}")
        print(f"      Data: {str(op.data)[:50]}...")
    
    return context_iterator, task_scheduler, game_loop


if __name__ == "__main__":
    print("🚀 === PERSISTENT CONTEXT ITERATOR SYSTEM ===")
    print("Demonstrating iterator with context cache and task scheduling\n")
    
    try:
        demo_result = demonstrate_system()
        
        print(f"\n✅ === DEMONSTRATION COMPLETE ===")
        print(f"🎯 System demonstrates:")
        print(f"   ✅ Persistent iterator with rolling context window")
        print(f"   ✅ Last operation cache with intelligent retention")
        print(f"   ✅ Iterator passed through event queue execution")
        print(f"   ✅ Separate task scheduler with thread management")
        print(f"   ✅ Integration with main game loop")
        print(f"   ✅ Memory-aware context management")
        
        print(f"\n🚀 Ready for your open world game integration!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()