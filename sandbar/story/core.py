"""
Story and entity core — stubs from complete_story_engine_system.md.
Merge with pine narrative / Mechanics where real implementations exist.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


# ----- Story tree (complete_story_engine_system) -----

class StoryNode:
    """Scene node with deferred operations; design: complete_story_engine_system."""

    def __init__(self, scene_id: str, data: Dict[str, Any]):
        self.scene_id = scene_id
        self.raw_text = data.get("raw_text", "")
        self.formatted_prose: Optional[str] = None
        self.choices: List[Any] = data.get("choices", [])
        self.prerequisites: List[str] = data.get("prerequisites", [])
        self.triggers: List[str] = data.get("triggers", [])
        self.children: Dict[str, "StoryNode"] = {}
        self.metadata: Dict[str, Any] = data.get("metadata", {})
        self.observers: List[Any] = []
        self.deferred_operations: List[Dict[str, Any]] = []

    def add_deferred_operation(self, builder_chain: Any, conditions: List[str]) -> None:
        """Store partial builder execution for later completion."""
        self.deferred_operations.append({
            "builder": builder_chain,
            "conditions": conditions,
            "created_at": time.time(),
        })


# ----- Entity tree (complete_story_engine_system) -----

class EntityNode:
    """Entity with semantic tags and story references."""

    def __init__(self, entity_id: str, entity_type: str, data: Dict[str, Any]):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.raw_data = data
        self.semantic_tags: List[str] = []
        self.meaning_extracts: List[Any] = []
        self.action_objects: List[Any] = []
        self.story_references: Set[str] = set()

    def extract_meanings(self, semantic_analyzer: Any) -> None:
        """Use LLM/semantic analyzer to extract actionable meanings (stub)."""
        if hasattr(semantic_analyzer, "extract_meanings"):
            self.meaning_extracts = semantic_analyzer.extract_meanings(self.raw_data)
        if hasattr(semantic_analyzer, "create_action_objects") and self.meaning_extracts:
            self.action_objects = semantic_analyzer.create_action_objects(self.meaning_extracts)


# ----- World state (complete_story_engine_system) -----

class WorldState:
    """World state for condition checks; design: complete_story_engine_system."""

    def __init__(self):
        self.character_location: Optional[str] = None
        self.character_inventory: Set[str] = set()
        self.character_relationships: Dict[str, Any] = {}
        self.plot_flags: Set[str] = set()
        self.time_of_day: str = "morning"
        self.magical_awareness: bool = False

    def check_condition(self, condition: str) -> bool:
        """Check if a specific condition is met."""
        if condition == "at_garden_of_eden":
            return self.character_location == "garden_of_eden"
        if condition == "has_money":
            return "money" in self.character_inventory
        if condition == "found_money_in_car":
            return "found_money_car" in self.plot_flags
        return False

    def update(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Update state from event (stub)."""
        if "location" in event_data:
            self.character_location = event_data["location"]
        if "inventory_delta" in event_data:
            for item in event_data.get("add", []):
                self.character_inventory.add(item)
            for item in event_data.get("remove", []):
                self.character_inventory.discard(item)


# ----- Story event manager (complete_story_engine_system) -----

class StoryEventManager:
    """Trigger story events and process deferred operations; design: complete_story_engine_system."""

    def __init__(self):
        self.event_listeners: Dict[str, List[Any]] = defaultdict(list)
        self.deferred_operations: Dict[str, Any] = {}
        self.condition_waiters: Dict[tuple, List[Any]] = defaultdict(list)
        self.world_state = WorldState()

    async def trigger_story_event(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Trigger story event and process waiting operations (stub; sync version available)."""
        self.world_state.update(event_name, event_data)
        for listener in self.event_listeners.get(event_name, []):
            try:
                if callable(listener):
                    listener(event_data)
            except Exception:
                pass

    def trigger_story_event_sync(self, event_name: str, event_data: Dict[str, Any]) -> None:
        """Synchronous version for use without asyncio."""
        self.world_state.update(event_name, event_data)
        for listener in self.event_listeners.get(event_name, []):
            try:
                if callable(listener):
                    listener(event_data)
            except Exception:
                pass
