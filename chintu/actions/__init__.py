"""Local intent routing and native Windows action helpers."""

from .router import Intent, IntentRouter
from .windows import ActionResult, AppEntry, WindowsActionExecutor, WindowsAppIndex

__all__ = [
    "ActionResult",
    "AppEntry",
    "Intent",
    "IntentRouter",
    "WindowsActionExecutor",
    "WindowsAppIndex",
]
