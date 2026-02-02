# Concept Signatures — Method Catalog

Catalog of interesting function/method signatures from design documents and key modules. Use for alignment when merging or adding from other sources.

**Sources:** complete_story_engine_system.md, chpt-island-engine/docs/design.md, island-engine (orchestrator, keepers, signbook, agent_runner), matts/, pine/.

---

## 1. Event store & orchestrator (island-engine)

```python
# services/orchestrator/event_store.py
class EventStore:
    def __init__(self, path: str) -> None
    def append(self, type: str, payload: Dict[str, Any], actor: str, thread: str, message_id: str, ts: str) -> int
    def list_recent(self, limit: int = 50) -> list
```

```python
# services/orchestrator/projections.py
def rebuild_canon(events) -> Dict[str, Any]
def rebuild_timeline(events, player_id: str) -> Dict[str, Any]
```

```python
# services/orchestrator/validators.py
def validate_event(event_type: str, payload: Dict[str, Any]) -> None
```

```python
# services/orchestrator/task_router.py
def assign_task(agent_id: str, task_kind: str, args: Dict[str, Any]) -> Dict[str, Any]
```

---

## 2. Keepers (island-engine)

```python
# keepers/procedures/plot_gravity.py
def detect_divergence(canon_snapshot: Dict[str, Any], timeline_snapshot: Dict[str, Any]) -> List[str]
def propose_gravity_moves(player_timeline_id: str, signals: List[str]) -> List[GravitySuggestion]
```

```python
# keepers/procedures/descend_entity.py
def validate_request(req: DescendRequest) -> List[str]
def propose_descend_events(req: DescendRequest) -> List[Dict[str, Any]]
```

```python
# keepers/procedures/contradiction_repair.py
def find_contradictions(snapshot: Dict[str, Any]) -> List[Contradiction]
def propose_repairs(contradictions: List[Contradiction]) -> List[Dict[str, Any]]
```

---

## 3. Signbook (island-engine)

```python
# services/signbook_server/signbook.py
def iso_now() -> str
def checksum16(signature: str, ts: str, message: str, tags: List[str]) -> str
class SignbookStorage:
    def add_entry(self, signature: str, message: str, context: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]
    def list_entries(self, limit: int = 50) -> List[Dict[str, Any]]
    def search_entries(self, q: str, limit: int = 50) -> List[Dict[str, Any]]
    def add_editorial(self, note: str) -> Dict[str, Any]
    def list_editorial(self, limit: int = 25) -> List[Dict[str, Any]]
```

```python
# signbook/legacy/signbook.py
def generate_signature(...) -> str
def parse_signature(signature: str) -> dict
def verify_signature(signature: str, seed: str) -> bool
def create_signbook(filepath: str, title: str = "The Signbook") -> Signbook
def quick_sign(...)
```

---

## 4. Agent runner & LLM (island-engine)

```python
# services/agent_runner/llm_adapters/local_stub.py
class LocalStubLLM:
    def complete(self, prompt: str, **kwargs) -> str
```

```python
# services/agent_runner/worker_pool.py
class WorkerPool:
    def __init__(self, max_workers: int = 4)
    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Future
    def shutdown(self)
```

---

## 5. Story tree & entity tree (complete_story_engine_system)

```python
class StoryNode:
    def __init__(self, scene_id, data)
    def add_deferred_operation(self, builder_chain, conditions)
```

```python
class EntityNode:
    def __init__(self, entity_id, entity_type, data)
    def extract_meanings(self, semantic_analyzer)
```

---

## 6. Link system & self-modifying iterator (complete_story_engine_system)

```python
class LinkTraverser:
    def __init__(self, story_tree, entity_tree)
    async def traverse_with_conditions(self, start_node, conditions)
    def _create_adaptive_iterator(self, node)
```

```python
class DeferredStoryBuilder:
    def doThis(self, action)
    def thenThis(self, action)
    def dontForgetTo(self, action, completion_conditions)
    # -> DeferredCompletion
class DeferredCompletion:
    async def fertilizeTree(self)
```

---

## 7. LLM integration & prose (complete_story_engine_system)

```python
class SemanticAnalysisEngine:
    async def extract_meanings_from_text(self, text)
    def create_action_objects(self, meanings)
```

```python
class ProseFormatter:
    async def format_story_elements(self, raw_elements, context)
    async def rephrase_existing_text(self, original_text, new_context)
```

```python
class DynamicCommandParser:
    async def parse_command(self, user_input, context)
    async def _handle_unknown_command(self, user_input, context)
```

```python
class StoryEventManager:
    async def trigger_story_event(self, event_name, event_data)
class WorldState:
    def check_condition(self, condition)
```

---

## 8. Matts — signals & context

```python
# matts/functional_interface.py
def create_context(context_id: str = None) -> SerializableExecutionContext
def get_context(context_id: str) -> Optional[SerializableExecutionContext]
async def emit_signal(signal_type, payload, ...)
async def transmit_context(source_context_id: str, target_context_id: str = "", ...)
async def bind_callback_with_dependencies(context, callback, ...)
def serialize_context(context) -> Dict[str, Any]
def deserialize_context(serialized_data: Dict[str, Any]) -> SerializableExecutionContext
async def quick_start(context_id: str = "main", signal_line_id: str = "main") -> tuple
```

```python
# matts/context_system.py
class SerializableExecutionContext:
    def create_snapshot(self, snapshot_type: str = "full") -> str
    def serialize_complete_context(self) -> Dict[str, Any]
    @classmethod
    def deserialize_complete_context(cls, serialized_data: Dict[str, Any]) -> 'SerializableExecutionContext'
    def create_hot_swap_version(self, version_id: str, changes: Dict[str, Any]) -> str
    def hot_swap_to_version(self, version_id: str) -> bool
```

```python
# matts/signal_system.py
async def create_signal_line(line_id: str = None, **kwargs) -> SignalLine
# SignalLine, SignalPayload, Observer, CallbackObserver, CircuitBreaker, PriorityDispatcher
```

```python
# matts/live_code_system.py
def serialize_source_code(self, source_code: str, name: str = "", ...) -> SerializedSourceCode
def serialize_function(self, func: Callable, trusted_source: bool = False) -> SerializedSourceCode
def deserialize_and_reconstruct(self, serialized, ...)
```

```python
# matts/generator_system.py
def create_child_branch(self, modifications: Dict[str, Any] = None) -> 'GeneratorStateBranch'
def compose_generators(self, chain_id: str, input_data: Any = None) -> Generator
def serialize_generator_composition(self, chain_id: str) -> Dict[str, Any]
```

---

## 9. Pine — narrative & world

```python
# pine/narrative/deferred_builder.py
class DeferredStoryBuilder:
    def doThis(self, action_name: str, ...)
    def thenThis(self, action_name: str, ...)
    def dontForgetTo(self, action_name: str, conditions: List[str], ...)
    def whenReady(self, action_name: str, ...)
    def finallyThis(self, action_name: str, ...)
    def execute(self, context: Dict[str, Any]) -> Any
    def get_reminders(self) -> List[str]
    def get_blockers(self, context: Dict[str, Any] = None) -> Dict[str, List[str]]
    def is_complete(self) -> bool
    def progress(self) -> float
```

```python
# pine/narrative/world.py
def create_world(name: str = "World") -> WhiteRoomBuilder
def build_world(name: str = "World") -> WhiteRoomBuilder
def text_to_world(...)
class WhiteRoomBuilder:
    def location(self, name: str, ...)
    def item(self, name: str, ...)
    def character(self, name: str, ...)
    def connect(self, ...)
    def origin(self, name: str)
    def build(self) -> WhiteRoom
```

```python
# pine/narrative/traversal.py
def smart_iter(...)
# TraversalContext, LayerConfig, pre/post callbacks
```

```python
# pine/narrative/lookahead.py
def lookahead_from(...)
# Possibility, discoveries, transitions, get_hints, summary
```

```python
# pine/narrative/extraction.py
def extract_text(text: str) -> ExtractionResult
```

---

## 10. Pine — graph & runtime

```python
# pine/graph/nodes.py
class GraphNodeStore:
    def add(self, node: GraphNode)
    def get(self, node_id: str) -> Optional[GraphNode]
    def find_by_type(self, node_type: NodeType) -> List[GraphNode]
    def find_by_tag(self, tag: str) -> List[GraphNode]
    def query(self, ...)
```

```python
# pine/graph/walker.py
def walk_bfs(...)
def walk_dfs(...)
def find_path(...)
def find_by_type(...)
```

```python
# pine/graph/embedding.py
def embed(self, text: str) -> Embedding
def embed_batch(self, texts: List[str]) -> List[Embedding]
def search(self, query_embedding: Embedding, top_k: int = 5) -> List[Tuple[str, float]]
def create_embedding_store(...)
```

```python
# pine/runtime/live_code.py
def serialize_function(self, func: Callable) -> SerializedSourceCode
def deserialize_function(self, serialized: SerializedSourceCode) -> Callable
def execute(self, ...)
```

```python
# pine/runtime/hotswap.py
def swap(self, new_func: Callable) -> Callable
def register(self, name: str, func: Callable) -> HotSwapHandler
def rollback(self, name: str) -> bool
```

---

## 11. Pine — signbook & messaging

```python
# pine/signbook/signature.py
def sign(self, content: str, nickname: str = None, ...)
def sign_with_metadata(self, content: str, ...)
def verify_signature(...)
```

```python
# pine/signbook/registry.py
def add(self, entry: SignEntry)
def find_by_nickname(self, nickname: str) -> List[SignEntry]
def get_latest(self, count: int = 10) -> List[SignEntry]
```

```python
# pine/messaging/interface.py
def story_event(self, event_name: str, **data) -> SignalPayload
def context_updated(self, context_id: str, **changes) -> SignalPayload
```

```python
# pine/messaging/connector.py
def register(self, listener_id: str, handler: Callable)
def route(self, payload: SignalPayload) -> int
```

---

## 12. Interface & LLM (proc_streamer / sandbar)

```python
# proc_streamer_v1_6.py (AssistChannel)
def connect(self, provider=None, url=None, model=None, api_key=None)
def query(self, prompt: str)
def _run_query(self, prompt: str)   # worker thread
def _healthcheck(self) -> Tuple[bool, str]
# Signals: chunk, complete, error, status
```

---

## 13. Sandbar (assimilated interface)

```python
# sandbar/llm_host/config.py
def load_llm_config(settings_path: Path | str | None = None) -> Dict[str, Any]
def save_llm_config(config: Dict[str, Any], settings_path: Path | str | None = None) -> None
```

```python
# sandbar/llm_host/assist_channel.py (Qt-free)
class AssistChannel:
    def __init__(self, config: Dict[str, Any])
    def connect(self, provider=..., url=..., model=..., api_key=...) -> None
    def disconnect(self) -> None
    def query(self, prompt: str) -> None
    # Callbacks: on_chunk, on_complete, on_error, on_status
```

```python
# sandbar/engine/client.py
def get_recent_events(limit: int = 50) -> List[Dict[str, Any]]
def get_canon_slice() -> Dict[str, Any]
def get_current_task() -> Optional[Dict[str, Any]]
def get_timeline_state(player_id: str) -> Dict[str, Any]
```

```python
# sandbar/engine/messaging/ (from islands/)
# connector_core: Connector, Connection, Listener, RouteTarget, make_connector
# messaging_interface: MessagingInterface, MessageBuffer, SignalFactory, CommandSignal, IOBridge, ConnectorBridge
# integration_layer: Entity, GraphNode, ThinkingGraph, Agent, GameOrchestrator
```

```python
# sandbar/ui/launcher.py
def main() -> int   # mode: proc_streamer | legacy | minimal
```

```python
# sandbar/engine/game_loop/ (from root/oasis)
# persistent_context_iterator: OperationType, ContextOperation, ContextWindow,
#   PersistentContextIterator, TaskScheduler, GameLoopIntegrator
# narrative_chain_iterator_system: ProcessControlManager, ChainEventType, ChainEvent,
#   ChainLink, NarrativeChain, CrossChainHandler, ChainIteratorEventQueue, StoryExecutionThread
```

```python
# sandbar/server/game_runner.py
class GameRunner:
    def receive_message(self, message: str, source: str = "user") -> Dict[str, Any]
    def run_one_frame(self) -> Dict[str, Any]
def get_runner() -> GameRunner
def run_message(message: str, source: str = "user") -> Dict[str, Any]
```

```python
# sandbar/server/app.py (FastAPI)
# POST /message (MessageIn: message, source) -> MessageOut
# GET /health
```

```python
# sandbar/story/core.py (complete_story_engine_system stubs)
class StoryNode: add_deferred_operation(builder_chain, conditions)
class EntityNode: extract_meanings(semantic_analyzer)
class WorldState: check_condition(condition), update(event_name, event_data)
class StoryEventManager: trigger_story_event(event_name, event_data), world_state
```

---

## Additions from other sources

When appending from another source, add a new section (e.g. "14. [Source name]") and list method signatures in the same style. Then reconcile with this catalog and ISLANDS_TODO / sandbar TODO.
