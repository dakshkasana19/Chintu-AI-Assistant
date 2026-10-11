from __future__ import annotations

from enum import Enum


class AssistantState(str, Enum):
    SLEEPING = "SLEEPING"
    WAKE_CONFIRM = "WAKE_CONFIRM"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"


class StateMachine:
    """Simple state tracker for the assistant runtime."""

    def __init__(self, initial: AssistantState = AssistantState.SLEEPING):
        self._state = initial

    @property
    def state(self) -> AssistantState:
        return self._state

    def transition(self, next_state: AssistantState) -> AssistantState:
        self._state = next_state
        return self._state

    def is_sleeping(self) -> bool:
        return self._state == AssistantState.SLEEPING
