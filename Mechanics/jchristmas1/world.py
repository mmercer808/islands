"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  MATTS WORLD - WhiteRoom Game World Construction                               ║
║  Layer 4: Depends on primitives, embedding, signals, extraction               ║
║                                                                               ║
║  The White Room is the origin - a featureless space where the world forms.   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from collections import defaultdict
import uuid

from .primitives import Embeddable, SignalType
from .embedding import Embedding, EmbeddingStore
from .signals import ObserverBus
from .extraction import (
    ExtractedEntity, ExtractedRelation, ExtractedFragment, ExtractionResult
)


# ═══════════════════════════════════════════════════════════════════════════════
#                              NODE & EDGE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorldNode:
    """
    A node in the game world graph.
    
    Compatible with IslandGraph's GraphNode but adds embedding support.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    node_type: str = "item"  # location, item, character, concept
    
    # Core data
    data: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    # Prose fragments attached to this node
    fragments: List[ExtractedFragment] = field(default_factory=list)
    
    # Embedding support
    embedding: Optional[Embedding] = None
    embedding_text: str = ""
    
    # Provenance
    source_entity_id: Optional[str] = None
    confidence: float = 0.5
    
    def to_embedding_text(self) -> str:
        """For Embeddable protocol"""
        if self.embedding_text:
            return self.embedding_text
        
        parts = [self.name]
        if 'description' in self.data:
            parts.append(self.data['description'])
        for frag in self.fragments[:3]:
            parts.append(frag.text)
        return " ".join(parts)
    
    def add_fragment(self, frag: ExtractedFragment):
        """Add prose fragment"""
        self.fragments.append(frag)
        self.fragments.sort(key=lambda f: f.category.value)
    
    def get_description(self) -> str:
        """Compose description from fragments"""
        return " ".join(f.text for f in self.fragments[:5])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id, 'name': self.name, 'node_type': self.node_type,
            'data': self.data, 'tags': list(self.tags),
            'fragments': [f.to_dict() for f in self.fragments],
            'confidence': self.confidence
        }


@dataclass
class WorldEdge:
    """An edge in the game world graph"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    edge_type: str = "contains"  # contains, leads_to, requires, etc.
    layer: str = "spatial"
    weight: float = 1.0
    bidirectional: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id, 'source_id': self.source_id, 'target_id': self.target_id,
            'edge_type': self.edge_type, 'layer': self.layer,
            'weight': self.weight, 'bidirectional': self.bidirectional,
            'data': self.data
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                              WHITE ROOM
# ═══════════════════════════════════════════════════════════════════════════════

class WhiteRoom:
    """
    The game world constructed from extracted text.
    
    The White Room is the origin - a featureless space where the world forms.
    Compatible with IslandGraph but adds embedding support.
    """
    
    def __init__(self, name: str = "White Room"):
        self.name = name
        self.id = str(uuid.uuid4())
        
        # Storage
        self.nodes: Dict[str, WorldNode] = {}
        self.edges: List[WorldEdge] = []
        
        # Embedding storage (optional)
        self.embedding_store: Optional[EmbeddingStore] = None
        
        # Indexes
        self._by_type: Dict[str, Set[str]] = defaultdict(set)
        self._by_tag: Dict[str, Set[str]] = defaultdict(set)
        self._edges_from: Dict[str, List[WorldEdge]] = defaultdict(list)
        self._edges_to: Dict[str, List[WorldEdge]] = defaultdict(list)
        
        # Origin node
        self.origin: Optional[WorldNode] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────────
    
    def set_embedding_store(self, store: EmbeddingStore):
        """Enable embedding storage for semantic operations"""
        self.embedding_store = store
    
    # ─────────────────────────────────────────────────────────────────────────
    # Node operations
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_node(self, node: WorldNode, embed: bool = True) -> str:
        """Add node, optionally embed it"""
        self.nodes[node.id] = node
        self._by_type[node.node_type].add(node.id)
        for tag in node.tags:
            self._by_tag[tag].add(node.id)
        
        if embed and self.embedding_store:
            node.embedding = self.embedding_store.embed_and_store(node, node.id)
        
        return node.id
    
    def get_node(self, node_id: str) -> Optional[WorldNode]:
        return self.nodes.get(node_id)
    
    def remove_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        
        # Remove from indexes
        self._by_type[node.node_type].discard(node_id)
        for tag in node.tags:
            self._by_tag[tag].discard(node_id)
        
        # Remove from embedding store
        if self.embedding_store:
            self.embedding_store.remove_for(node_id)
        
        # Remove connected edges
        self.edges = [e for e in self.edges 
                      if e.source_id != node_id and e.target_id != node_id]
        if node_id in self._edges_from:
            del self._edges_from[node_id]
        if node_id in self._edges_to:
            del self._edges_to[node_id]
        
        del self.nodes[node_id]
        return True
    
    # ─────────────────────────────────────────────────────────────────────────
    # Edge operations
    # ─────────────────────────────────────────────────────────────────────────
    
    def add_edge(self, edge: WorldEdge) -> str:
        """Add edge"""
        self.edges.append(edge)
        self._edges_from[edge.source_id].append(edge)
        self._edges_to[edge.target_id].append(edge)
        
        if edge.bidirectional:
            reverse = WorldEdge(
                source_id=edge.target_id,
                target_id=edge.source_id,
                edge_type=edge.edge_type,
                layer=edge.layer,
                weight=edge.weight,
                data=edge.data.copy()
            )
            self.edges.append(reverse)
            self._edges_from[reverse.source_id].append(reverse)
            self._edges_to[reverse.target_id].append(reverse)
        
        return edge.id
    
    def connect(self, source_id: str, target_id: str,
                edge_type: str = "contains",
                layer: str = "spatial",
                **kw) -> str:
        """Convenience: create and add edge"""
        edge = WorldEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            layer=layer,
            **kw
        )
        return self.add_edge(edge)
    
    def edges_from(self, node_id: str) -> List[WorldEdge]:
        """Get outgoing edges"""
        return self._edges_from.get(node_id, [])
    
    def edges_to(self, node_id: str) -> List[WorldEdge]:
        """Get incoming edges"""
        return self._edges_to.get(node_id, [])
    
    def neighbors(self, node_id: str) -> List[WorldNode]:
        """Get neighboring nodes"""
        return [self.nodes[e.target_id] 
                for e in self.edges_from(node_id) 
                if e.target_id in self.nodes]
    
    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────
    
    def by_type(self, node_type: str) -> List[WorldNode]:
        """Get nodes by type"""
        return [self.nodes[nid] for nid in self._by_type.get(node_type, set())]
    
    def by_tag(self, tag: str) -> List[WorldNode]:
        """Get nodes by tag"""
        return [self.nodes[nid] for nid in self._by_tag.get(tag, set())]
    
    def find(self, **criteria) -> List[WorldNode]:
        """Query nodes by criteria"""
        results = list(self.nodes.values())
        
        if 'node_type' in criteria:
            results = [n for n in results if n.node_type == criteria['node_type']]
        if 'name' in criteria:
            results = [n for n in results if criteria['name'].lower() in n.name.lower()]
        if 'tag' in criteria:
            results = [n for n in results if criteria['tag'] in n.tags]
        if 'min_confidence' in criteria:
            results = [n for n in results if n.confidence >= criteria['min_confidence']]
        
        return results
    
    # ─────────────────────────────────────────────────────────────────────────
    # Semantic (embedding-based)
    # ─────────────────────────────────────────────────────────────────────────
    
    def find_similar(self, query: Union[str, WorldNode], 
                     k: int = 5) -> List[Tuple[WorldNode, float]]:
        """Find semantically similar nodes"""
        if not self.embedding_store:
            return []
        
        results = self.embedding_store.find_similar(query, k)
        return [(self.nodes[sid], score) 
                for sid, score in results 
                if sid in self.nodes]
    
    def semantic_neighbors(self, node_id: str, k: int = 5,
                           threshold: float = 0.5) -> List[Tuple[WorldNode, float]]:
        """Get semantic (embedding-based) neighbors"""
        node = self.get_node(node_id)
        if not node:
            return []
        
        similar = self.find_similar(node, k + 1)
        return [(n, s) for n, s in similar if n.id != node_id and s >= threshold]
    
    # ─────────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────────
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'id': self.id,
            'origin_id': self.origin.id if self.origin else None,
            'nodes': [n.to_dict() for n in self.nodes.values()],
            'edges': [e.to_dict() for e in self.edges]
        }
    
    def summary(self) -> str:
        lines = [
            f"═══ {self.name} ═══",
            f"Nodes: {len(self.nodes)}",
            f"Edges: {len(self.edges)}",
            "By type:"
        ]
        for t, ids in sorted(self._by_type.items()):
            names = [self.nodes[i].name for i in list(ids)[:3]]
            suffix = "..." if len(ids) > 3 else ""
            lines.append(f"  {t}: {', '.join(names)}{suffix}")
        return "\n".join(lines)
    
    def __repr__(self):
        return f"<WhiteRoom '{self.name}' nodes={len(self.nodes)} edges={len(self.edges)}>"


# ═══════════════════════════════════════════════════════════════════════════════
#                              BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

class WhiteRoomBuilder:
    """
    Builds a WhiteRoom from extraction results.
    
    Extraction → WhiteRoom (game world graph).
    """
    
    # Map relation words to edge types
    RELATION_MAP = {
        'in': 'contains',
        'inside': 'contains',
        'on': 'supports',
        'under': 'below',
        'near': 'near',
        'beside': 'beside',
        'behind': 'behind',
        'above': 'above',
        'below': 'below',
        'through': 'leads_to',
    }
    
    def __init__(self, 
                 embedding_store: EmbeddingStore = None,
                 bus: ObserverBus = None):
        self._store = embedding_store
        self._bus = bus
        self._id_map: Dict[str, str] = {}
    
    def build(self, extraction: ExtractionResult,
              min_confidence: float = 0.3) -> WhiteRoom:
        """
        Build WhiteRoom from extraction result.
        
        Creates origin, converts entities to nodes, relations to edges,
        assigns fragments, connects orphans.
        """
        world = WhiteRoom()
        
        if self._store:
            world.set_embedding_store(self._store)
        
        self._id_map = {}
        
        # Create origin (The White Room)
        origin = WorldNode(
            name="The White Room",
            node_type="location",
            embedding_text="A featureless white space where the world forms around you.",
            tags={"origin", "location"},
            confidence=1.0
        )
        world.add_node(origin)
        world.origin = origin
        
        # Convert entities to nodes
        for ext_ent in extraction.entities:
            if ext_ent.confidence >= min_confidence:
                node = self._entity_to_node(ext_ent)
                if node:
                    world.add_node(node)
                    self._id_map[ext_ent.id] = node.id
                    
                    if self._bus:
                        self._bus.emit_type(SignalType.ENTITY_CREATED,
                                           source_id=node.id,
                                           data={'name': node.name})
        
        # Convert relations to edges
        for ext_rel in extraction.relations:
            edge = self._relation_to_edge(ext_rel)
            if edge:
                world.add_edge(edge)
                
                if self._bus:
                    self._bus.emit_type(SignalType.ENTITY_LINKED,
                                       source_id=edge.source_id,
                                       data={'target': edge.target_id})
        
        # Assign fragments to nodes
        self._assign_fragments(world, extraction.fragments)
        
        # Connect orphans to origin
        self._connect_orphans(world, origin.id)
        
        return world
    
    def _entity_to_node(self, ent: ExtractedEntity) -> Optional[WorldNode]:
        """Convert extracted entity to world node"""
        # Build embedding text
        emb_parts = [ent.name]
        emb_parts.extend(ent.adjectives)
        emb_parts.extend(ent.descriptions[:2])
        
        return WorldNode(
            name=ent.name.title(),
            node_type=ent.category,
            embedding_text=" ".join(emb_parts),
            source_entity_id=ent.id,
            confidence=ent.confidence,
            data={
                'adjectives': list(ent.adjectives),
                'mention_count': ent.mention_count
            }
        )
    
    def _relation_to_edge(self, rel: ExtractedRelation) -> Optional[WorldEdge]:
        """Convert extracted relation to world edge"""
        source = self._id_map.get(rel.source_id)
        target = self._id_map.get(rel.target_id)
        
        if not source or not target:
            return None
        
        edge_type = self.RELATION_MAP.get(rel.relation_word, rel.relation_word)
        
        return WorldEdge(
            source_id=source,
            target_id=target,
            edge_type=edge_type,
            layer="spatial" if edge_type in ('contains', 'supports', 'near') else "logical"
        )
    
    def _assign_fragments(self, world: WhiteRoom, fragments: List[ExtractedFragment]):
        """Assign fragments to their nodes"""
        for frag in fragments:
            if frag.entity_id and frag.entity_id in self._id_map:
                node_id = self._id_map[frag.entity_id]
                node = world.get_node(node_id)
                if node:
                    node.add_fragment(frag)
    
    def _connect_orphans(self, world: WhiteRoom, origin_id: str):
        """Connect nodes without edges to origin"""
        linked = set()
        for edge in world.edges:
            linked.add(edge.source_id)
            linked.add(edge.target_id)
        
        for node_id in world.nodes:
            if node_id != origin_id and node_id not in linked:
                world.connect(origin_id, node_id, "contains")


# ═══════════════════════════════════════════════════════════════════════════════
#                              FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

def build_world(extraction: ExtractionResult,
                embedding_store: EmbeddingStore = None,
                bus: ObserverBus = None) -> WhiteRoom:
    """Build WhiteRoom from extraction"""
    return WhiteRoomBuilder(embedding_store, bus).build(extraction)


def text_to_world(text: str,
                  embedding_store: EmbeddingStore = None,
                  bus: ObserverBus = None) -> WhiteRoom:
    """Complete pipeline: text → WhiteRoom"""
    from .extraction import extract_text
    extraction = extract_text(text, bus)
    return build_world(extraction, embedding_store, bus)


# ═══════════════════════════════════════════════════════════════════════════════
#                              EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'WorldNode', 'WorldEdge', 'WhiteRoom', 'WhiteRoomBuilder',
    'build_world', 'text_to_world',
]
