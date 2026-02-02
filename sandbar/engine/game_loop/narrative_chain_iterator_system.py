#!/usr/bin/env python3
"""
Narrative Chain Iterator System
===============================

A system that takes control of Python process management for story execution.
Uses threaded chain iterators where handlers can span multiple chain links,
creating complex narrative structures that execute as event queues.

Core Concepts:
- Process registry takeover for complete control
- Threaded story execution with chain iterators 
- Handlers that span multiple chain links
- Event queue pattern for narrative progression
- Cross-chain callback coordination

Author: Claude & Human Collaboration
License: MIT
"""

import sys
import gc
import threading
import queue
import time
import weakref
import uuid
import marshal
import types
from typing import Any, Dict, List, Optional, Callable, Iterator, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import traceback
import logging
from enum import Enum, auto
import copy


# ===== PROCESS CONTROL SYSTEM =====

class ProcessControlManager:
    """Takes control of Python process for story execution"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.process_id = id(self)
        self.start_time = time.time()
        
        # Core registries - we OWN these now
        self.object_registry = weakref.WeakValueDictionary()
        self.thread_registry = {}
        self.story_threads = {}
        self.chain_registry = {}
        self.handler_registry = {}
        self.event_queue_registry = {}
        
        # Process control
        self.main_thread = threading.current_thread()
        self.story_executor = None
        self.is_story_mode = False
        
        # Garbage collection override
        self.original_gc_collect = gc.collect
        self.gc_callbacks = []
        
        # Hook into Python's cleanup
        self._hook_process_cleanup()
        
    def take_control(self):
        """Take full control of the Python process for story execution"""
        print("🎮 Taking control of Python process for story execution...")
        
        # Override garbage collection
        gc.collect = self._controlled_gc_collect
        
        # Set up story execution thread
        self.story_executor = StoryExecutionThread(self)
        self.story_executor.start()
        
        # Hook into thread creation
        self._hook_thread_creation()
        
        self.is_story_mode = True
        print(f"✅ Process control established. Story mode: {self.is_story_mode}")
        
    def _controlled_gc_collect(self, generation=2):
        """Controlled garbage collection that preserves story objects"""
        
        # Run pre-GC callbacks
        for callback in self.gc_callbacks:
            try:
                callback('pre_gc')
            except Exception as e:
                print(f"GC callback error: {e}")
        
        # Protect story objects during GC
        protected_objects = []
        for obj_id, obj in list(self.object_registry.items()):
            if hasattr(obj, '_story_protected') and obj._story_protected:
                protected_objects.append(obj)
        
        # Run original GC
        result = self.original_gc_collect(generation)
        
        # Run post-GC callbacks
        for callback in self.gc_callbacks:
            try:
                callback('post_gc')
            except Exception as e:
                print(f"GC callback error: {e}")
        
        # Re-register protected objects if needed
        for obj in protected_objects:
            if id(obj) not in self.object_registry:
                self.register_object(obj)
        
        return result
    
    def _hook_thread_creation(self):
        """Hook into thread creation to track story threads"""
        original_thread_init = threading.Thread.__init__
        
        def hooked_thread_init(thread_self, *args, **kwargs):
            result = original_thread_init(thread_self, *args, **kwargs)
            
            # Register thread if it's story-related
            if hasattr(thread_self, '_story_thread') and thread_self._story_thread:
                self.thread_registry[thread_self.ident] = thread_self
                print(f"📖 Registered story thread: {thread_self.name}")
            
            return result
        
        threading.Thread.__init__ = hooked_thread_init
    
    def _hook_process_cleanup(self):
        """Hook into process cleanup to save story state"""
        import atexit
        
        def cleanup_story_state():
            print("[OK] Process cleanup - saving story state...")
            self.save_story_state()
            print("[OK] Story state saved")
        
        atexit.register(cleanup_story_state)
    
    def register_object(self, obj, story_protected=False):
        """Register object in process registry"""
        obj_id = id(obj)
        self.object_registry[obj_id] = obj
        
        if story_protected:
            obj._story_protected = True
        
        return obj_id
    
    def get_object(self, obj_id: int):
        """Get object from registry"""
        return self.object_registry.get(obj_id)
    
    def register_story_thread(self, thread: threading.Thread):
        """Register a story execution thread"""
        thread._story_thread = True
        self.story_threads[thread.ident] = thread
        return thread.ident
    
    def save_story_state(self):
        """Save complete story state for process persistence"""
        state = {
            'chains': {cid: chain.serialize() for cid, chain in self.chain_registry.items()},
            'handlers': {hid: handler.serialize() for hid, handler in self.handler_registry.items()},
            'event_queues': {qid: eq.get_state() for qid, eq in self.event_queue_registry.items()},
            'timestamp': time.time()
        }
        
        # In a real implementation, save to persistent storage
        # For now, just store in memory
        self._saved_state = state
        return state

# Global process controller
PROCESS_CONTROLLER = ProcessControlManager()


# ===== NARRATIVE CHAIN SYSTEM =====

class ChainEventType(Enum):
    """Types of events in narrative chains"""
    CHAIN_START = auto()
    CHAIN_PROGRESS = auto()
    CHAIN_COMPLETE = auto()
    HANDLER_TRIGGER = auto()
    CROSS_CHAIN_BRIDGE = auto()
    NARRATIVE_BRANCH = auto()
    STORY_CHECKPOINT = auto()


@dataclass
class ChainEvent:
    """Event in the narrative chain system"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: ChainEventType = ChainEventType.CHAIN_PROGRESS
    chain_id: str = ""
    handler_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source_chain: Optional[str] = None
    target_chains: List[str] = field(default_factory=list)
    
    def serialize(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.name,
            'chain_id': self.chain_id,
            'handler_id': self.handler_id,
            'data': self.data,
            'timestamp': self.timestamp,
            'source_chain': self.source_chain,
            'target_chains': self.target_chains
        }


class ChainLink:
    """A single link in a narrative chain"""
    
    def __init__(self, link_id: str = None, callback: Callable = None, data: Dict[str, Any] = None):
        self.link_id = link_id or str(uuid.uuid4())
        self.callback = callback
        self.data = data or {}
        self.execution_count = 0
        self.last_execution_time = None
        self.cross_chain_handlers = []  # Handlers that span multiple chains
        
    def add_cross_chain_handler(self, handler: 'CrossChainHandler'):
        """Add handler that spans multiple chain links"""
        self.cross_chain_handlers.append(handler)
        
    def execute(self, event: ChainEvent, context: Dict[str, Any]) -> Any:
        """Execute this chain link"""
        self.execution_count += 1
        self.last_execution_time = time.time()
        
        result = None
        
        try:
            if self.callback:
                # Execute the callback with full context
                result = self.callback(event, context, self.data)
            
            # Execute cross-chain handlers
            for handler in self.cross_chain_handlers:
                handler.handle_link_execution(self, event, context, result)
                
        except Exception as e:
            print(f"❌ Chain link execution error: {e}")
            traceback.print_exc()
            
        return result
    
    def serialize(self) -> Dict[str, Any]:
        return {
            'link_id': self.link_id,
            'data': self.data,
            'execution_count': self.execution_count,
            'last_execution_time': self.last_execution_time,
            'cross_chain_handler_count': len(self.cross_chain_handlers)
        }


class NarrativeChain:
    """A chain representing a story sequence"""
    
    def __init__(self, chain_id: str = None, name: str = ""):
        self.chain_id = chain_id or str(uuid.uuid4())
        self.name = name
        self.links = []
        self.current_position = 0
        self.is_active = False
        self.completion_callbacks = []
        self.branch_conditions = []
        self.metadata = {}
        
        # Register with process controller
        PROCESS_CONTROLLER.chain_registry[self.chain_id] = self
        
    def add_link(self, callback: Callable, data: Dict[str, Any] = None) -> ChainLink:
        """Add a link to this chain"""
        link = ChainLink(callback=callback, data=data)
        self.links.append(link)
        return link
    
    def add_completion_callback(self, callback: Callable):
        """Add callback for when chain completes"""
        self.completion_callbacks.append(callback)
    
    def add_branch_condition(self, condition: Callable, target_chain_id: str):
        """Add condition for branching to another chain"""
        self.branch_conditions.append({
            'condition': condition,
            'target_chain_id': target_chain_id
        })
    
    def get_current_link(self) -> Optional[ChainLink]:
        """Get current link in chain"""
        if 0 <= self.current_position < len(self.links):
            return self.links[self.current_position]
        return None
    
    def advance(self) -> bool:
        """Advance to next link in chain"""
        self.current_position += 1
        return self.current_position < len(self.links)
    
    def is_complete(self) -> bool:
        """Check if chain is complete"""
        return self.current_position >= len(self.links)
    
    def reset(self):
        """Reset chain to beginning"""
        self.current_position = 0
        self.is_active = False
    
    def check_branch_conditions(self, context: Dict[str, Any]) -> List[str]:
        """Check if any branch conditions are met"""
        triggered_chains = []
        
        for branch in self.branch_conditions:
            try:
                if branch['condition'](context):
                    triggered_chains.append(branch['target_chain_id'])
            except Exception as e:
                print(f"Branch condition error: {e}")
        
        return triggered_chains
    
    def serialize(self) -> Dict[str, Any]:
        return {
            'chain_id': self.chain_id,
            'name': self.name,
            'links': [link.serialize() for link in self.links],
            'current_position': self.current_position,
            'is_active': self.is_active,
            'metadata': self.metadata
        }


# ===== CROSS-CHAIN HANDLER SYSTEM =====

class CrossChainHandler:
    """Handler that spans multiple chain links across different chains"""
    
    def __init__(self, handler_id: str = None, name: str = ""):
        self.handler_id = handler_id or str(uuid.uuid4())
        self.name = name
        self.source_chains = set()  # Chains this handler monitors
        self.target_chains = set()  # Chains this handler can affect
        self.trigger_conditions = []
        self.execution_actions = []
        self.state = {}
        self.execution_history = []
        
        # Register with process controller
        PROCESS_CONTROLLER.handler_registry[self.handler_id] = self
    
    def add_source_chain(self, chain_id: str, link_positions: List[int] = None):
        """Add chain that this handler monitors"""
        self.source_chains.add(chain_id)
        
        # Attach to specific links in the chain if specified
        if link_positions and chain_id in PROCESS_CONTROLLER.chain_registry:
            chain = PROCESS_CONTROLLER.chain_registry[chain_id]
            for pos in link_positions:
                if 0 <= pos < len(chain.links):
                    chain.links[pos].add_cross_chain_handler(self)
    
    def add_target_chain(self, chain_id: str):
        """Add chain that this handler can affect"""
        self.target_chains.add(chain_id)
    
    def add_trigger_condition(self, condition: Callable):
        """Add condition that triggers this handler"""
        self.trigger_conditions.append(condition)
    
    def add_execution_action(self, action: Callable):
        """Add action to execute when triggered"""
        self.execution_actions.append(action)
    
    def handle_link_execution(self, link: ChainLink, event: ChainEvent, context: Dict[str, Any], link_result: Any):
        """Handle execution of a monitored chain link"""
        
        # Update handler state with link execution info
        self.state.update({
            'last_link_id': link.link_id,
            'last_event': event.serialize(),
            'last_context': context.copy(),
            'last_result': link_result,
            'execution_time': time.time()
        })
        
        # Check trigger conditions
        should_trigger = True
        for condition in self.trigger_conditions:
            try:
                if not condition(self.state, link, event, context):
                    should_trigger = False
                    break
            except Exception as e:
                print(f"Handler trigger condition error: {e}")
                should_trigger = False
                break
        
        if should_trigger:
            self._execute_actions(link, event, context, link_result)
    
    def _execute_actions(self, link: ChainLink, event: ChainEvent, context: Dict[str, Any], link_result: Any):
        """Execute handler actions"""
        
        execution_record = {
            'handler_id': self.handler_id,
            'trigger_link': link.link_id,
            'trigger_event': event.event_id,
            'timestamp': time.time(),
            'actions_executed': []
        }
        
        for action in self.execution_actions:
            try:
                action_result = action(self.state, link, event, context, link_result)
                execution_record['actions_executed'].append({
                    'action': action.__name__ if hasattr(action, '__name__') else str(action),
                    'result': action_result,
                    'success': True
                })
                
                # If action returns target chains, trigger them
                if isinstance(action_result, dict) and 'trigger_chains' in action_result:
                    for target_chain_id in action_result['trigger_chains']:
                        self._trigger_target_chain(target_chain_id, event, context)
                        
            except Exception as e:
                print(f"Handler action execution error: {e}")
                execution_record['actions_executed'].append({
                    'action': str(action),
                    'error': str(e),
                    'success': False
                })
        
        self.execution_history.append(execution_record)
    
    def _trigger_target_chain(self, chain_id: str, source_event: ChainEvent, context: Dict[str, Any]):
        """Trigger a target chain"""
        if chain_id in PROCESS_CONTROLLER.chain_registry:
            chain = PROCESS_CONTROLLER.chain_registry[chain_id]
            
            # Create bridge event
            bridge_event = ChainEvent(
                event_type=ChainEventType.CROSS_CHAIN_BRIDGE,
                chain_id=chain_id,
                handler_id=self.handler_id,
                source_chain=source_event.chain_id,
                data={
                    'source_event_id': source_event.event_id,
                    'bridge_context': context.copy(),
                    'handler_state': self.state.copy()
                }
            )
            
            # Add to appropriate event queue
            for queue_id, event_queue in PROCESS_CONTROLLER.event_queue_registry.items():
                if event_queue.handles_chain(chain_id):
                    event_queue.add_event(bridge_event)
                    break
    
    def serialize(self) -> Dict[str, Any]:
        return {
            'handler_id': self.handler_id,
            'name': self.name,
            'source_chains': list(self.source_chains),
            'target_chains': list(self.target_chains),
            'state': self.state,
            'execution_count': len(self.execution_history)
        }


# ===== CHAIN ITERATOR EVENT QUEUE =====

class ChainIteratorEventQueue:
    """Iterator-based event queue for executing narrative chains"""
    
    def __init__(self, queue_id: str = None, name: str = ""):
        self.queue_id = queue_id or str(uuid.uuid4())
        self.name = name
        self.event_queue = queue.PriorityQueue()
        self.chain_assignments = {}  # chain_id -> priority
        self.active_chains = set()
        self.completed_chains = set()
        self.context = {}
        self.is_running = False
        self.iterator_thread = None
        
        # Execution state
        self.events_processed = 0
        self.execution_history = deque(maxlen=1000)
        
        # Register with process controller
        PROCESS_CONTROLLER.event_queue_registry[self.queue_id] = self
    
    def assign_chain(self, chain_id: str, priority: int = 5):
        """Assign a chain to this event queue"""
        self.chain_assignments[chain_id] = priority
        
        # Create initial start event for the chain
        start_event = ChainEvent(
            event_type=ChainEventType.CHAIN_START,
            chain_id=chain_id,
            data={'queue_id': self.queue_id}
        )
        self.add_event(start_event, priority)
    
    def handles_chain(self, chain_id: str) -> bool:
        """Check if this queue handles the given chain"""
        return chain_id in self.chain_assignments
    
    def add_event(self, event: ChainEvent, priority: int = None):
        """Add event to the queue"""
        if priority is None:
            priority = self.chain_assignments.get(event.chain_id, 5)
        
        # PriorityQueue uses (priority, item) tuples
        # Lower priority number = higher priority
        self.event_queue.put((priority, time.time(), event))
    
    def start_iteration(self, context: Dict[str, Any] = None):
        """Start the iterator in a separate thread"""
        if self.is_running:
            print(f"Queue {self.name} already running")
            return
        
        self.context.update(context or {})
        self.is_running = True
        
        # Create and start iterator thread
        self.iterator_thread = threading.Thread(
            target=self._iterator_loop,
            name=f"ChainIterator_{self.name}",
            daemon=True
        )
        self.iterator_thread._story_thread = True
        PROCESS_CONTROLLER.register_story_thread(self.iterator_thread)
        self.iterator_thread.start()
        
        print(f"🎬 Started chain iterator: {self.name}")
    
    def stop_iteration(self):
        """Stop the iterator"""
        self.is_running = False
        if self.iterator_thread and self.iterator_thread.is_alive():
            # Add stop event to wake up the queue
            stop_event = ChainEvent(
                event_type=ChainEventType.CHAIN_COMPLETE,
                chain_id="__STOP__"
            )
            self.add_event(stop_event, priority=0)
            self.iterator_thread.join(timeout=1.0)
        
        print(f"🛑 Stopped chain iterator: {self.name}")
    
    def _iterator_loop(self):
        """Main iterator loop - executes in separate thread"""
        print(f"🔄 Chain iterator loop started: {self.name}")
        
        while self.is_running:
            try:
                # Get next event from queue (blocks until available)
                priority, timestamp, event = self.event_queue.get(timeout=1.0)
                
                # Check for stop signal
                if event.chain_id == "__STOP__":
                    break
                
                # Process the event
                self._process_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except queue.Empty:
                # No events in queue, continue
                continue
            except Exception as e:
                print(f"Iterator loop error: {e}")
                traceback.print_exc()
        
        print(f"✅ Chain iterator loop finished: {self.name}")
    
    def _process_event(self, event: ChainEvent):
        """Process a single event"""
        self.events_processed += 1
        
        # Add to execution history
        execution_record = {
            'event_id': event.event_id,
            'event_type': event.event_type.name,
            'chain_id': event.chain_id,
            'timestamp': time.time(),
            'processed_count': self.events_processed
        }
        
        try:
            if event.event_type == ChainEventType.CHAIN_START:
                self._handle_chain_start(event)
            elif event.event_type == ChainEventType.CHAIN_PROGRESS:
                self._handle_chain_progress(event)
            elif event.event_type == ChainEventType.CHAIN_COMPLETE:
                self._handle_chain_complete(event)
            elif event.event_type == ChainEventType.CROSS_CHAIN_BRIDGE:
                self._handle_cross_chain_bridge(event)
            else:
                self._handle_generic_event(event)
            
            execution_record['success'] = True
            
        except Exception as e:
            print(f"Event processing error: {e}")
            execution_record['success'] = False
            execution_record['error'] = str(e)
        
        self.execution_history.append(execution_record)
    
    def _handle_chain_start(self, event: ChainEvent):
        """Handle chain start event"""
        chain_id = event.chain_id
        
        if chain_id in PROCESS_CONTROLLER.chain_registry:
            chain = PROCESS_CONTROLLER.chain_registry[chain_id]
            chain.is_active = True
            chain.reset()  # Start from beginning
            self.active_chains.add(chain_id)
            
            print(f"🎬 Started chain: {chain.name} ({chain_id[:8]})")
            
            # Create progress event for first link
            progress_event = ChainEvent(
                event_type=ChainEventType.CHAIN_PROGRESS,
                chain_id=chain_id,
                data={'link_position': 0}
            )
            self.add_event(progress_event)
    
    def _handle_chain_progress(self, event: ChainEvent):
        """Handle chain progress event"""
        chain_id = event.chain_id
        
        if chain_id in PROCESS_CONTROLLER.chain_registry:
            chain = PROCESS_CONTROLLER.chain_registry[chain_id]
            current_link = chain.get_current_link()
            
            if current_link:
                # Execute the current link
                link_result = current_link.execute(event, self.context)
                
                print(f"🔗 Executed link {chain.current_position} in {chain.name}")
                
                # Check for branch conditions
                triggered_chains = chain.check_branch_conditions(self.context)
                for target_chain_id in triggered_chains:
                    branch_event = ChainEvent(
                        event_type=ChainEventType.NARRATIVE_BRANCH,
                        chain_id=target_chain_id,
                        source_chain=chain_id
                    )
                    self.add_event(branch_event)
                
                # Advance to next link
                if chain.advance():
                    # Create next progress event
                    next_progress_event = ChainEvent(
                        event_type=ChainEventType.CHAIN_PROGRESS,
                        chain_id=chain_id,
                        data={'link_position': chain.current_position}
                    )
                    self.add_event(next_progress_event)
                else:
                    # Chain complete
                    complete_event = ChainEvent(
                        event_type=ChainEventType.CHAIN_COMPLETE,
                        chain_id=chain_id
                    )
                    self.add_event(complete_event)
    
    def _handle_chain_complete(self, event: ChainEvent):
        """Handle chain completion"""
        chain_id = event.chain_id
        
        if chain_id in PROCESS_CONTROLLER.chain_registry:
            chain = PROCESS_CONTROLLER.chain_registry[chain_id]
            chain.is_active = False
            self.active_chains.discard(chain_id)
            self.completed_chains.add(chain_id)
            
            print(f"✅ Completed chain: {chain.name} ({chain_id[:8]})")
            
            # Execute completion callbacks
            for callback in chain.completion_callbacks:
                try:
                    callback(chain, self.context)
                except Exception as e:
                    print(f"Completion callback error: {e}")
    
    def _handle_cross_chain_bridge(self, event: ChainEvent):
        """Handle cross-chain bridge event"""
        print(f"🌉 Cross-chain bridge: {event.source_chain} -> {event.chain_id}")
        
        # Update context with bridge data
        self.context.update(event.data.get('bridge_context', {}))
        
        # Start or resume the target chain
        start_event = ChainEvent(
            event_type=ChainEventType.CHAIN_START,
            chain_id=event.chain_id,
            data=event.data
        )
        self.add_event(start_event)
    
    def _handle_generic_event(self, event: ChainEvent):
        """Handle other event types"""
        print(f"📋 Generic event: {event.event_type.name} for {event.chain_id[:8]}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the event queue"""
        return {
            'queue_id': self.queue_id,
            'name': self.name,
            'is_running': self.is_running,
            'events_processed': self.events_processed,
            'active_chains': list(self.active_chains),
            'completed_chains': list(self.completed_chains),
            'context': self.context.copy()
        }
    
    def __iter__(self):
        """Make this an iterator for direct use"""
        return self
    
    def __next__(self):
        """Iterator interface - get next processed event"""
        if not self.is_running or self.event_queue.empty():
            raise StopIteration
        
        try:
            priority, timestamp, event = self.event_queue.get_nowait()
            self._process_event(event)
            return event
        except queue.Empty:
            raise StopIteration


# ===== STORY EXECUTION THREAD =====

class StoryExecutionThread(threading.Thread):
    """Main story execution thread that coordinates all narrative chains"""
    
    def __init__(self, process_controller):
        super().__init__(name="StoryExecutionThread", daemon=True)
        self.process_controller = process_controller
        self.event_queues = []
        self.is_story_running = False
        self._story_thread = True
        
    def add_event_queue(self, event_queue: ChainIteratorEventQueue):
        """Add an event queue to story execution"""
        self.event_queues.append(event_queue)
        
    def run(self):
        """Main story execution loop"""
        print("🎭 Story execution thread started")
        self.is_story_running = True
        
        while self.is_story_running and self.process_controller.is_story_mode:
            try:
                # Monitor all event queues
                active_queues = [eq for eq in self.event_queues if eq.is_running]
                
                if not active_queues:
                    time.sleep(0.1)
                    continue
                
                # Check status of each queue
                for event_queue in active_queues:
                    if not event_queue.iterator_thread.is_alive():
                        print(f"⚠️ Event queue thread died: {event_queue.name}")
                        # Could restart it here
                
                time.sleep(0.05)  # Small delay to prevent busy waiting
                
            except Exception as e:
                print(f"Story execution error: {e}")
                traceback.print_exc()
                time.sleep(1.0)
        
        print("🎭 Story execution thread stopped")
    
    def stop_story_execution(self):
        """Stop story execution"""
        self.is_story_running = False
        
        # Stop all event queues
        for event_queue in self.event_queues:
            event_queue.stop_iteration()


# ===== DEMONSTRATION SYSTEM =====

def create_story_example():
    """Create example story with cross-chain handlers"""
    
    print("📖 Creating example story with chain iterators...")
    
    # Take control of the process
    PROCESS_CONTROLLER.take_control()
    
    # Create story chains
    
    # Main quest chain
    main_quest = NarrativeChain(name="Main Quest")
    
    