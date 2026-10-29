from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class Intent:
    """A deterministic intent name, extracted slots, and safety metadata."""

    name: str
    slots: dict[str, object] = field(default_factory=dict)
    confidence: float = 1.0
    requires_confirmation: bool = False


class IntentRouter:
    """Route normalized command text through anchored local intent patterns."""

    def __init__(self, exit_phrases: list[str] | None = None) -> None:
        self._exit_phrases = {
            self._normalize(phrase) for phrase in (exit_phrases or [])
        }

    def route(self, text: str) -> Intent:
        command = self._normalize(text)
        if not command:
            return Intent("unknown", confidence=0.0)

        if command in self._exit_phrases:
            return Intent("sleep")
        if re.match(r"^(?:go to sleep|stop listening)(?:\b.*)?$", command):
            return Intent("sleep")
        if command in {"confirm", "yes, confirm", "do it"}:
            return Intent("confirm")
        if command in {"cancel shutdown", "cancel restart"}:
            return Intent("cancel_power")
        if command in {"cancel timer", "cancel alarm"}:
            return Intent("cancel_timer")
        if command in {"cancel", "never mind", "nevermind", "don't do it"}:
            return Intent("cancel")

        match = re.fullmatch(r"(?:open|launch|start)\s+(.+)", command)
        if match:
            target = match.group(1).strip()
            website = self._website_name(target)
            if website is not None:
                return Intent("open_website", {"target": website})
            return Intent("open_app", {"target": target})

        match = re.fullmatch(r"(?:close|quit|exit)\s+(.+)", command)
        if match:
            return Intent(
                "close_app",
                {"target": match.group(1).strip()},
                requires_confirmation=True,
            )

        if command in {"lock my computer", "lock the computer", "lock pc", "lock screen"}:
            return Intent("lock_workstation")
        if re.fullmatch(r"(?:take|capture|save)\s+(?:a\s+)?screenshot", command):
            return Intent("screenshot")

        if command in {"mute", "mute volume", "mute the volume"}:
            return Intent("volume", {"operation": "mute"})
        if command in {"unmute", "unmute volume", "unmute the volume"}:
            return Intent("volume", {"operation": "unmute"})
        if re.fullmatch(r"(?:turn|set)\s+(?:the\s+)?volume\s+to\s+\d{1,3}(?:\s+percent)?", command):
            value = int(re.search(r"\d{1,3}", command).group())
            if 0 <= value <= 100:
                return Intent("volume", {"operation": "set", "value": value})
            return Intent("clarify", {"prompt": "Choose a volume from 0 to 100 percent."}, 0.0)
        match = re.fullmatch(
            r"(?:turn\s+)?volume\s+(up|down)(?:\s+by\s+(\d{1,3})(?:\s+percent)?)?",
            command,
        )
        if match:
            direction, amount = match.groups()
            return Intent(
                "volume",
                {
                    "operation": direction,
                    "amount": int(amount) if amount else 5,
                },
            )

        match = re.fullmatch(
            r"(?:set|change)\s+(?:the\s+)?brightness\s+to\s+(\d{1,3})(?:\s+percent)?",
            command,
        )
        if match:
            value = int(match.group(1))
            if 0 <= value <= 100:
                return Intent("brightness", {"operation": "set", "value": value})
            return Intent("clarify", {"prompt": "Choose brightness from 0 to 100 percent."}, 0.0)
        match = re.fullmatch(r"brightness\s+(up|down)(?:\s+by\s+(\d{1,3}))?", command)
        if match:
            direction, amount = match.groups()
            return Intent(
                "brightness",
                {
                    "operation": direction,
                    "amount": int(amount) if amount else 10,
                },
            )

        media_commands = {
            "play": "play_pause",
            "pause": "play_pause",
            "play pause": "play_pause",
            "resume": "play_pause",
            "stop music": "play_pause",
            "stop playback": "play_pause",
            "next": "next",
            "next track": "next",
            "skip": "next",
            "previous": "previous",
            "previous track": "previous",
            "go back": "previous",
        }
        if command in media_commands:
            return Intent("media", {"operation": media_commands[command]})
        if re.fullmatch(r"(?:play|pause)\s+(?:music|media|song|playback)", command):
            return Intent("media", {"operation": "play_pause"})

        if command in {
            "shut down",
            "shutdown",
            "turn off my computer",
            "restart",
            "restart my computer",
            "sign out",
            "log out",
        }:
            operation = (
                "shutdown"
                if command in {"shut down", "shutdown", "turn off my computer"}
                else "restart"
                if command in {"restart", "restart my computer"}
                else "sign_out"
            )
            return Intent(
                "system_power",
                {"operation": operation},
                requires_confirmation=True,
            )

        if command in {"what time is it", "tell me the time", "current time"}:
            return Intent("time")
        if command in {"what is the date", "what's the date", "today's date", "current date"}:
            return Intent("date")
        if command in {"what day is it", "what day is today", "today's day"}:
            return Intent("day")

        if re.fullmatch(r"(?:check|show|what is|what's)\s+(?:my\s+)?battery(?:\s+status|\s+level)?", command):
            return Intent("system_info", {"metric": "battery"})
        if re.fullmatch(r"(?:check|show|what is|what's)\s+(?:my\s+)?(?:cpu|processor)(?:\s+usage)?", command):
            return Intent("system_info", {"metric": "cpu"})
        if re.fullmatch(r"(?:check|show|what is|what's)\s+(?:my\s+)?(?:memory|ram)(?:\s+usage)?", command):
            return Intent("system_info", {"metric": "memory"})

        match = re.fullmatch(
            r"(?:set\s+)?(?:a\s+)?timer\s+(?:for\s+)?(\d{1,3})\s*(seconds?|minutes?|hours?)",
            command,
        )
        if match:
            amount = int(match.group(1))
            unit = match.group(2).rstrip("s")
            seconds = amount * {"second": 1, "minute": 60, "hour": 3600}[unit]
            if 1 <= seconds <= 24 * 3600:
                return Intent("timer", {"seconds": seconds})
            return Intent("clarify", {"prompt": "Set a timer between 1 second and 24 hours."}, 0.0)
        match = re.fullmatch(
            r"(?:set\s+)?(?:an?\s+)?alarm\s+(?:for\s+)?(\d{1,2}):(\d{2})(?:\s*(am|pm))?",
            command,
        )
        if match:
            hour, minute, meridiem = match.groups()
            hour_value, minute_value = int(hour), int(minute)
            if meridiem:
                if not 1 <= hour_value <= 12 or minute_value > 59:
                    return Intent("clarify", {"prompt": "Give an alarm time such as 7:30 PM."}, 0.0)
                hour_value = hour_value % 12 + (12 if meridiem == "pm" else 0)
            elif hour_value > 23 or minute_value > 59:
                return Intent("clarify", {"prompt": "Give an alarm time such as 19:30."}, 0.0)
            return Intent("alarm", {"hour": hour_value, "minute": minute_value})

        if self._looks_like_information_request(command):
            return Intent("information_request", {"query": text.strip()})
        return Intent("unknown", {"text": text.strip()}, 0.0)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"^[\W_]+|[\W_]+$", "", text.casefold())).strip()

    @staticmethod
    def _website_name(target: str) -> str | None:
        common = {
            "google": "https://www.google.com",
            "youtube": "https://www.youtube.com",
            "gmail": "https://mail.google.com",
            "maps": "https://maps.google.com",
            "google maps": "https://maps.google.com",
            "wikipedia": "https://www.wikipedia.org",
            "github": "https://github.com",
            "stackoverflow": "https://stackoverflow.com",
            "stack overflow": "https://stackoverflow.com",
        }
        if target in common:
            return common[target]
        domain = re.fullmatch(
            r"(?:https?://)?[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
            r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+"
            r"(?::\d{1,5})?(?:/[\w./?=&%-]*)?",
            target,
        )
        if domain:
            candidate = target if "://" in target else f"https://{target}"
            parsed = urlsplit(candidate)
            if (
                parsed.scheme not in {"https", "http"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.hostname.rsplit(".", 1)[-1] in {
                    "exe",
                    "lnk",
                    "bat",
                    "cmd",
                    "msi",
                    "url",
                }
            ):
                return None
            try:
                port = parsed.port
            except ValueError:
                return None
            if port is not None and not 1 <= port <= 65535:
                return None
            return urlunsplit(("https", parsed.netloc, parsed.path, parsed.query, ""))
        return None

    @staticmethod
    def _looks_like_information_request(command: str) -> bool:
        return command.startswith(
            (
                "what ",
                "who ",
                "where ",
                "when ",
                "why ",
                "how ",
                "tell me ",
                "explain ",
                "define ",
                "compare ",
            )
        )
