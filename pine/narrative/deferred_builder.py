"""
Pine Deferred Story Builder
===========================

Builder pattern for stories that suspend mid-execution
until conditions are met.

The Deferred Builder Pattern:
1. doThis() - Execute immediately
2. thenThis() - Execute immediately after previous
3. dontForgetTo() - Defer until conditions are met
4. finallyThis() - Execute when all conditions satisfied

This enables stories that:
- Branch based on player state
- Wait for inventory items
- Trigger on location entry
- React to time passage

Quick Reference:
----------------
    from pine.narrative import DeferredStoryBuilder

    quest = (DeferredStoryBuilder()
        .doThis("approach_castle")           # Immediate
        .thenThis("speak_to_guard")          # Immediate
        .dontForgetTo("enter_throne_room",   # Deferred
            conditions=['has_permission', 'guard_approves'])
        .finallyThis("meet_king"))           # After all conditions

    # Execute what's ready
    for action in quest.execute(player_state):
        perform(action)
"""

from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod


# =============================================================================
#                              ENUMS
# =============================================================================

class ActionState(Enum):
    """State of a story action."""
    PENDING = auto()    # Not yet executed
    READY = auto()      # Conditions met, ready to execute
    EXECUTING = auto()  # Currently executing
    COMPLETED = auto()  # Finished
    BLOCKED = auto()    # Conditions not met
    FAILED = auto()     # Execution failed


# =============================================================================
#                              CONDITIONS
# =============================================================================

@dataclass
class StoryCondition:
    """
    A condition that must be met for an action to execute.

    Conditions can check:
    - Inventory contents
    - Location
    - Flags/state
    - Custom predicates
    """
    name: str
    predicate: Optional[Callable[[Dict[str, Any]], bool]] = None
    description: str = ""

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the condition against context."""
        if self.predicate:
            return self.predicate(context)
        # Default: check if name exists as truthy value in context
        return bool(context.get(self.name))


# =============================================================================
#                              ACTIONS
# =============================================================================

class StoryAction(ABC):
    """Abstract base for story actions."""

    def __init__(self, name: str):
        self.name = name
        self.state = ActionState.PENDING

    @abstractmethod
    def can_execute(self, context: Dict[str, Any]) -> bool:
        """Check if this action can execute now."""
        pass

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the action."""
        pass


@dataclass
class ImmediateAction(StoryAction):
    """Action that executes immediately when reached."""

    def __init__(self, name: str, handler: Callable = None):
        super().__init__(name)
        self.handler = handler

    def can_execute(self, context: Dict[str, Any]) -> bool:
        return True

    def execute(self, context: Dict[str, Any]) -> Any:
        self.state = ActionState.EXECUTING
        result = None
        if self.handler:
            result = self.handler(context)
        self.state = ActionState.COMPLETED
        return result


@dataclass
class ConditionalAction(StoryAction):
    """Action that waits for conditions to be met."""

    def __init__(
        self,
        name: str,
        conditions: List[str | StoryCondition] = None,
        handler: Callable = None
    ):
        super().__init__(name)
        self._conditions = []
        for c in (conditions or []):
            if isinstance(c, str):
                self._conditions.append(StoryCondition(name=c))
            else:
                self._conditions.append(c)
        self.handler = handler

    def can_execute(self, context: Dict[str, Any]) -> bool:
        return all(c.evaluate(context) for c in self._conditions)

    def get_unmet_conditions(self, context: Dict[str, Any]) -> List[str]:
        """Get list of conditions not yet met."""
        unmet = []
        for c in self._conditions:
            if not c.evaluate(context):
                unmet.append(c.name)
        return unmet

    def execute(self, context: Dict[str, Any]) -> Any:
        if not self.can_execute(context):
            self.state = ActionState.BLOCKED
            return None

        self.state = ActionState.EXECUTING
        result = None
        if self.handler:
            result = self.handler(context)
        self.state = ActionState.COMPLETED
        return result


@dataclass
class DeferredAction(StoryAction):
    """
    Action that explicitly defers execution.

    Unlike ConditionalAction, this remembers it was
    deferred and can provide "don't forget" reminders.
    """

    def __init__(
        self,
        name: str,
        conditions: List[str | StoryCondition] = None,
        handler: Callable = None,
        reminder: str = None
    ):
        super().__init__(name)
        self._conditions = []
        for c in (conditions or []):
            if isinstance(c, str):
                self._conditions.append(StoryCondition(name=c))
            else:
                self._conditions.append(c)
        self.handler = handler
        self.reminder = reminder or f"Don't forget to: {name}"
        self._reminder_shown = False

    def can_execute(self, context: Dict[str, Any]) -> bool:
        return all(c.evaluate(context) for c in self._conditions)

    def get_reminder(self) -> Optional[str]:
        """Get reminder text if not yet executed."""
        if self.state in (ActionState.PENDING, ActionState.BLOCKED):
            return self.reminder
        return None

    def execute(self, context: Dict[str, Any]) -> Any:
        if not self.can_execute(context):
            self.state = ActionState.BLOCKED
            return None

        self.state = ActionState.EXECUTING
        result = None
        if self.handler:
            result = self.handler(context)
        self.state = ActionState.COMPLETED
        return result


# =============================================================================
#                              DEFERRED BUILDER
# =============================================================================

class DeferredStoryBuilder:
    """
    Fluent builder for deferred story sequences.

    Chains actions that execute immediately or wait
    for conditions to be met.

    Example:
        quest = (DeferredStoryBuilder()
            .doThis("start_quest")
            .thenThis("find_clue")
            .dontForgetTo("unlock_door", conditions=['has_key'])
            .finallyThis("enter_room"))
    """

    def __init__(self, name: str = "Story"):
        self.name = name
        self._actions: List[StoryAction] = []
        self._current_context: Dict[str, Any] = {}

    def doThis(
        self,
        action_name: str,
        handler: Callable = None
    ) -> 'DeferredStoryBuilder':
        """Add an immediate action."""
        self._actions.append(ImmediateAction(action_name, handler))
        return self

    def thenThis(
        self,
        action_name: str,
        handler: Callable = None
    ) -> 'DeferredStoryBuilder':
        """Add an immediate action that follows the previous."""
        return self.doThis(action_name, handler)

    def dontForgetTo(
        self,
        action_name: str,
        conditions: List[str] = None,
        handler: Callable = None,
        reminder: str = None
    ) -> 'DeferredStoryBuilder':
        """Add a deferred action that waits for conditions."""
        self._actions.append(DeferredAction(
            action_name,
            conditions=conditions,
            handler=handler,
            reminder=reminder
        ))
        return self

    def whenReady(
        self,
        action_name: str,
        conditions: List[str] = None,
        handler: Callable = None
    ) -> 'DeferredStoryBuilder':
        """Add a conditional action."""
        self._actions.append(ConditionalAction(
            action_name,
            conditions=conditions,
            handler=handler
        ))
        return self

    def finallyThis(
        self,
        action_name: str,
        handler: Callable = None
    ) -> 'DeferredStoryBuilder':
        """Add a final action (after all deferred actions complete)."""
        # This needs all previous deferred actions to complete
        deferred_names = [
            a.name for a in self._actions
            if isinstance(a, (DeferredAction, ConditionalAction))
        ]

        # Create condition that checks all deferred are done
        def all_deferred_complete(ctx: Dict[str, Any]) -> bool:
            return all(ctx.get(f"_completed_{n}") for n in deferred_names)

        condition = StoryCondition(
            name="_all_deferred_complete",
            predicate=all_deferred_complete
        )

        self._actions.append(ConditionalAction(
            action_name,
            conditions=[condition],
            handler=handler
        ))
        return self

    def execute(
        self,
        context: Dict[str, Any] = None
    ) -> List[tuple[str, Any]]:
        """
        Execute all ready actions.

        Returns list of (action_name, result) for executed actions.
        Deferred actions are skipped if conditions not met.
        """
        self._current_context = context or {}
        results = []

        for action in self._actions:
            if action.state == ActionState.COMPLETED:
                continue

            if action.can_execute(self._current_context):
                result = action.execute(self._current_context)
                # Mark as completed in context
                self._current_context[f"_completed_{action.name}"] = True
                results.append((action.name, result))

        return results

    def get_reminders(self) -> List[str]:
        """Get reminders for deferred actions not yet complete."""
        reminders = []
        for action in self._actions:
            if isinstance(action, DeferredAction):
                reminder = action.get_reminder()
                if reminder:
                    reminders.append(reminder)
        return reminders

    def get_blockers(self, context: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """Get conditions blocking each deferred action."""
        context = context or self._current_context
        blockers = {}

        for action in self._actions:
            if isinstance(action, (DeferredAction, ConditionalAction)):
                unmet = action.get_unmet_conditions(context)
                if unmet:
                    blockers[action.name] = unmet

        return blockers

    def is_complete(self) -> bool:
        """Check if all actions are complete."""
        return all(a.state == ActionState.COMPLETED for a in self._actions)

    @property
    def progress(self) -> float:
        """Get completion progress as 0.0-1.0."""
        if not self._actions:
            return 1.0
        completed = sum(1 for a in self._actions if a.state == ActionState.COMPLETED)
        return completed / len(self._actions)
