"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  MATTS SIGNALS - Generic Observer Pattern                                      ║
║  Layer 1: Depends on primitives only                                          ║
║                                                                               ║
║  Observer[P] works with ANY input class.                                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, List, Set, Optional, Any, Callable, 
    Generic, TypeVar, Tuple
)
from collections import defaultdict, deque
import threading
import time
import uuid

from .primitives import Priority, SignalType, T, P


# ═══════════════════════════════════════════════════════════════════════════════
#                              SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Signal(Generic[P]):
    """
    Generic signal carrying payload of type P.
    
    Works with ANY payload type - GraphNode, Entity, str, custom class.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: SignalType = SignalType.CUSTOM
    source_id: str = ""
    target_id: Optional[str] = None
    
    # Generic payload - ANY type
    payload: Optional[P] = None
    
    # Additional data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timing & priority
    timestamp: float = field(default_factory=time.time)
    priority: Priority = Priority.NORMAL
    
    # Processing state
    handled: bool = False
    cancelled: bool = False
    responses: List[Any] = field(default_factory=list)
    error: Optional[Exception] = None
    
    def with_payload(self, payload: P) -> 'Signal[P]':
        """Copy with new payload"""
        return Signal(
            signal_type=self.signal_type,
            source_id=self.source_id,
            target_id=self.target_id,
            payload=payload,
            data=self.data.copy(),
            priority=self.priority
        )
    
    def respond(self, response: Any, observer_id: str = None) -> None:
        """Add response from observer"""
        self.responses.append({
            'observer_id': observer_id,
            'response': response,
            'at': time.time()
        })
        self.handled = True
    
    def cancel(self) -> None:
        """Stop further processing"""
        self.cancelled = True
    
    def fail(self, error: Exception) -> None:
        """Mark as failed"""
        self.error = error
        self.handled = True


# ═══════════════════════════════════════════════════════════════════════════════
#                              OBSERVER (GENERIC)
# ═══════════════════════════════════════════════════════════════════════════════

class Observer(ABC, Generic[P]):
    """
    Abstract generic observer for ANY payload type P.
    
    Works with:
    - Observer[GraphNode] for graph nodes
    - Observer[Entity] for entities  
    - Observer[str] for strings
    - Observer[Any] for anything
    """
    
    def __init__(self,
                 observer_id: str = None,
                 priority: Priority = Priority.NORMAL):
        self.id = observer_id or str(uuid.uuid4())
        self.priority = priority
        self._active = True
        self._filter: Optional[Callable[[Signal[P]], bool]] = None
        self._handled = 0
        self._errors = 0
    
    @property
    def active(self) -> bool:
        return self._active
    
    def activate(self) -> 'Observer[P]':
        self._active = True
        return self
    
    def deactivate(self) -> 'Observer[P]':
        self._active = False
        return self
    
    def filter(self, fn: Callable[[Signal[P]], bool]) -> 'Observer[P]':
        """Set filter function"""
        self._filter = fn
        return self
    
    def for_type(self, st: SignalType) -> 'Observer[P]':
        """Filter to single signal type"""
        self._filter = lambda s: s.signal_type == st
        return self
    
    def for_types(self, *types: SignalType) -> 'Observer[P]':
        """Filter to multiple signal types"""
        ts = set(types)
        self._filter = lambda s: s.signal_type in ts
        return self
    
    def accepts(self, signal: Signal[P]) -> bool:
        """Check if observer should handle signal"""
        if not self._active or signal.cancelled:
            return False
        if self._filter and not self._filter(signal):
            return False
        return True
    
    @abstractmethod
    def handle(self, signal: Signal[P]) -> Any:
        """Handle signal - override in subclass"""
        ...
    
    def __call__(self, signal: Signal[P]) -> Any:
        """Direct call support"""
        if not self.accepts(signal):
            return None
        try:
            result = self.handle(signal)
            self._handled += 1
            return result
        except Exception as e:
            self._errors += 1
            raise


# ═══════════════════════════════════════════════════════════════════════════════
#                              CONCRETE OBSERVERS
# ═══════════════════════════════════════════════════════════════════════════════

class FunctionObserver(Observer[P]):
    """Wraps any callable as observer"""
    
    def __init__(self, fn: Callable[[Signal[P]], Any],
                 observer_id: str = None,
                 priority: Priority = Priority.NORMAL):
        super().__init__(observer_id, priority)
        self._fn = fn
    
    def handle(self, signal: Signal[P]) -> Any:
        return self._fn(signal)


class MethodObserver(Observer[P]):
    """Calls method on payload objects"""
    
    def __init__(self, method_name: str,
                 observer_id: str = None,
                 priority: Priority = Priority.NORMAL):
        super().__init__(observer_id, priority)
        self._method = method_name
    
    def handle(self, signal: Signal[P]) -> Any:
        if signal.payload is None:
            return None
        method = getattr(signal.payload, self._method, None)
        if method and callable(method):
            return method(signal)
        return None


class CollectorObserver(Observer[P]):
    """Collects payloads matching predicate"""
    
    def __init__(self, predicate: Callable[[P], bool] = None,
                 observer_id: str = None):
        super().__init__(observer_id, Priority.LOW)
        self._pred = predicate
        self.items: List[P] = []
    
    def handle(self, signal: Signal[P]) -> int:
        if signal.payload is not None:
            if self._pred is None or self._pred(signal.payload):
                self.items.append(signal.payload)
        return len(self.items)
    
    def take(self) -> List[P]:
        """Take and clear items"""
        items, self.items = self.items, []
        return items


class StateObserver(Observer[Any]):
    """Specialized for state machine transitions"""
    
    def __init__(self, observer_id: str = None):
        super().__init__(observer_id, Priority.HIGH)
        self.history: List[Tuple[str, float]] = []
        self.current: Optional[str] = None
        
        self._on_enter: Dict[str, List[Callable]] = defaultdict(list)
        self._on_exit: Dict[str, List[Callable]] = defaultdict(list)
        self._on_transition: Dict[Tuple[str, str], List[Callable]] = defaultdict(list)
        
        self.for_types(SignalType.STATE_ENTER, SignalType.STATE_EXIT, SignalType.STATE_CHANGE)
    
    def on_enter(self, state: str, cb: Callable) -> 'StateObserver':
        self._on_enter[state].append(cb)
        return self
    
    def on_exit(self, state: str, cb: Callable) -> 'StateObserver':
        self._on_exit[state].append(cb)
        return self
    
    def on_transition(self, from_s: str, to_s: str, cb: Callable) -> 'StateObserver':
        self._on_transition[(from_s, to_s)].append(cb)
        return self
    
    def handle(self, signal: Signal) -> Dict[str, str]:
        old = signal.data.get('old_state')
        new = signal.data.get('new_state')
        
        for cb in self._on_exit.get(old, []):
            try: cb(signal)
            except: pass
        
        for cb in self._on_transition.get((old, new), []):
            try: cb(signal)
            except: pass
        
        for cb in self._on_enter.get(new, []):
            try: cb(signal)
            except: pass
        
        if new:
            self.history.append((new, signal.timestamp))
            self.current = new
        
        return {'from': old, 'to': new}


class EntityObserver(Observer[Any]):
    """Specialized for entity lifecycle"""
    
    def __init__(self, observer_id: str = None):
        super().__init__(observer_id, Priority.NORMAL)
        self.log: List[Dict] = []
        self._watched: Dict[str, List[Callable]] = defaultdict(list)
        
        self.for_types(
            SignalType.ENTITY_CREATED, SignalType.ENTITY_MODIFIED,
            SignalType.ENTITY_DESTROYED, SignalType.ENTITY_LINKED
        )
    
    def watch(self, entity_id: str, cb: Callable) -> 'EntityObserver':
        self._watched[entity_id].append(cb)
        return self
    
    def handle(self, signal: Signal) -> str:
        eid = signal.data.get('entity_id', signal.source_id)
        
        self.log.append({
            'type': signal.signal_type.value,
            'entity': eid,
            'at': signal.timestamp
        })
        
        for cb in self._watched.get(eid, []):
            try: cb(signal)
            except: pass
        
        return eid


class TraversalObserver(Observer[Any]):
    """Specialized for graph traversal"""
    
    def __init__(self, observer_id: str = None):
        super().__init__(observer_id, Priority.NORMAL)
        self.path: List[str] = []
        self.visited: Set[str] = set()
        self.layers: List[Tuple[str, str]] = []
        
        self._on_step: List[Callable] = []
        self._on_layer: List[Callable] = []
        
        self.for_types(
            SignalType.TRAVERSAL_START, SignalType.TRAVERSAL_STEP,
            SignalType.TRAVERSAL_END, SignalType.LAYER_SWITCH
        )
    
    def on_step(self, cb: Callable) -> 'TraversalObserver':
        self._on_step.append(cb)
        return self
    
    def on_layer_switch(self, cb: Callable) -> 'TraversalObserver':
        self._on_layer.append(cb)
        return self
    
    def handle(self, signal: Signal) -> Optional[Dict]:
        if signal.signal_type == SignalType.TRAVERSAL_STEP:
            nid = signal.data.get('node_id')
            if nid:
                self.path.append(nid)
                self.visited.add(nid)
            for cb in self._on_step:
                try: cb(signal)
                except: pass
            return {'depth': len(self.path)}
        
        elif signal.signal_type == SignalType.LAYER_SWITCH:
            old, new = signal.data.get('old_layer'), signal.data.get('new_layer')
            self.layers.append((old, new))
            for cb in self._on_layer:
                try: cb(signal)
                except: pass
            return {'switches': len(self.layers)}
        
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#                              BUS (DISPATCH HUB)
# ═══════════════════════════════════════════════════════════════════════════════

class ObserverBus(Generic[P]):
    """
    Central hub for observer registration and signal dispatch.
    Thread-safe. Generic over payload type.
    """
    
    def __init__(self):
        self._observers: Dict[str, Observer[P]] = {}
        self._subs: Dict[SignalType, Set[str]] = defaultdict(set)
        self._history: deque = deque(maxlen=1000)
        self._pending: deque = deque()
        self._processing = False
        self._lock = threading.RLock()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Registration
    # ─────────────────────────────────────────────────────────────────────────
    
    def register(self, obs: Observer[P], 
                 types: List[SignalType] = None) -> str:
        """Register observer, returns its ID"""
        with self._lock:
            self._observers[obs.id] = obs
            for st in (types or list(SignalType)):
                self._subs[st].add(obs.id)
            return obs.id
    
    def unregister(self, obs_id: str) -> bool:
        """Remove observer"""
        with self._lock:
            if obs_id in self._observers:
                del self._observers[obs_id]
                for subs in self._subs.values():
                    subs.discard(obs_id)
                return True
            return False
    
    def on(self, st: SignalType, 
           fn: Callable[[Signal[P]], Any],
           priority: Priority = Priority.NORMAL) -> str:
        """Convenience: register function as observer"""
        return self.register(FunctionObserver(fn, priority=priority), [st])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Dispatch
    # ─────────────────────────────────────────────────────────────────────────
    
    def emit(self, signal: Signal[P]) -> Signal[P]:
        """Emit signal, returns it after processing"""
        with self._lock:
            self._pending.append(signal)
            if not self._processing:
                self._process()
        return signal
    
    def emit_type(self, st: SignalType,
                  source_id: str = "",
                  payload: P = None,
                  data: Dict[str, Any] = None,
                  **kw) -> Signal[P]:
        """Convenience: emit by type"""
        return self.emit(Signal(
            signal_type=st,
            source_id=source_id,
            payload=payload,
            data=data or {},
            **kw
        ))
    
    def _process(self):
        self._processing = True
        while self._pending:
            sig = self._pending.popleft()
            self._dispatch(sig)
            self._history.append(sig)
        self._processing = False
    
    def _dispatch(self, sig: Signal[P]):
        obs_ids = self._subs.get(sig.signal_type, set())
        observers = [self._observers[oid] for oid in obs_ids if oid in self._observers]
        observers = [o for o in observers if o.accepts(sig)]
        observers.sort(key=lambda o: o.priority.value)
        
        for obs in observers:
            if sig.cancelled:
                break
            try:
                result = obs.handle(sig)
                if result is not None:
                    sig.respond(result, obs.id)
            except Exception as e:
                sig.fail(e)
                if sig.signal_type != SignalType.ERROR:
                    self.emit_type(SignalType.ERROR, obs.id, data={'error': str(e)})
    
    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────
    
    def get(self, obs_id: str) -> Optional[Observer[P]]:
        return self._observers.get(obs_id)
    
    def history(self, st: SignalType = None, limit: int = 100) -> List[Signal[P]]:
        sigs = list(self._history)
        if st:
            sigs = [s for s in sigs if s.signal_type == st]
        return sigs[-limit:]
    
    # ─────────────────────────────────────────────────────────────────────────
    # Factories
    # ─────────────────────────────────────────────────────────────────────────
    
    def state_observer(self) -> Tuple[StateObserver, str]:
        o = StateObserver()
        return o, self.register(o)
    
    def entity_observer(self) -> Tuple[EntityObserver, str]:
        o = EntityObserver()
        return o, self.register(o)
    
    def traversal_observer(self) -> Tuple[TraversalObserver, str]:
        o = TraversalObserver()
        return o, self.register(o)


# ═══════════════════════════════════════════════════════════════════════════════
#                              EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'Signal',
    'Observer', 'FunctionObserver', 'MethodObserver', 'CollectorObserver',
    'StateObserver', 'EntityObserver', 'TraversalObserver',
    'ObserverBus',
]
