from __future__ import annotations

from enum import Enum


class AssistantState(str, Enum):
    SLEEPING = "SLEEPING"
    WAKE_CONFIRM = "WAKE_CONFIRM"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"


class StateMachine:
    """Validate and track the voice assistant lifecycle."""

    _ALLOWED_TRANSITIONS = {
        AssistantState.SLEEPING: {
            AssistantState.SLEEPING,
            AssistantState.WAKE_CONFIRM,
        },
        AssistantState.WAKE_CONFIRM: {
            AssistantState.LISTENING,
            AssistantState.SLEEPING,
        },
        AssistantState.LISTENING: {
            AssistantState.PROCESSING,
            AssistantState.SLEEPING,
        },
        AssistantState.PROCESSING: {
            AssistantState.SPEAKING,
            AssistantState.LISTENING,
            AssistantState.SLEEPING,
        },
        AssistantState.SPEAKING: {
            AssistantState.LISTENING,
            AssistantState.SLEEPING,
        },
    }

    def __init__(self, initial: AssistantState = AssistantState.SLEEPING):
        self._state = initial

    @property
    def state(self) -> AssistantState:
        return self._state

    def transition(self, next_state: AssistantState) -> AssistantState:
        if next_state not in self._ALLOWED_TRANSITIONS[self._state]:
            raise ValueError(
                f"Invalid assistant state transition: {self._state.value} -> {next_state.value}"
            )
        self._state = next_state
        return self._state

    def is_sleeping(self) -> bool:
        return self._state == AssistantState.SLEEPING
