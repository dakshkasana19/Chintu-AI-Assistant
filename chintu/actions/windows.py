from __future__ import annotations

import ctypes
import json
import logging
import os
import re
import subprocess
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

from rapidfuzz import fuzz, process

from .router import Intent

logger = logging.getLogger("chintu.actions.windows")
_START_APPS_COMMAND = (
    "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress"
)


@dataclass(frozen=True)
class AppEntry:
    """A Start Menu shortcut or registered Windows application."""

    name: str
    path: str | None = None
    app_id: str | None = None


@dataclass(frozen=True)
class AppResolution:
    """An unambiguous app match or the choices needed to clarify it."""

    entry: AppEntry | None
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionResult:
    """Structured outcome from one local action."""

    success: bool
    message: str
    value: object | None = None
    confirmation_required: bool = False


class WindowsAppIndex:
    """Index Start Menu shortcuts, installed apps, and user aliases."""

    def __init__(
        self,
        entries: Iterable[AppEntry] = (),
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.entries = tuple(entries)
        self.aliases = {
            self._normalize(key): value.strip()
            for key, value in (aliases or {}).items()
            if key.strip() and value.strip()
        }

    @classmethod
    def discover(
        cls,
        aliases: dict[str, str] | None = None,
        *,
        environ: dict[str, str] | None = None,
        start_apps: Callable[[], list[AppEntry]] | None = None,
    ) -> WindowsAppIndex:
        """Build the app index from standard Start Menu paths and UWP entries."""

        environment = os.environ if environ is None else environ
        entries: dict[tuple[str, str | None, str | None], AppEntry] = {}
        start_menu_paths = [
            Path(environment[root])
            / "Microsoft"
            / "Windows"
            / "Start Menu"
            / "Programs"
            for root in ("APPDATA", "PROGRAMDATA")
            if environment.get(root)
        ]
        for root in start_menu_paths:
            if not root.is_dir():
                continue
            for shortcut in root.rglob("*.lnk"):
                name = shortcut.stem.strip()
                if name:
                    entry = AppEntry(name=name, path=str(shortcut))
                    entries[(entry.name.casefold(), entry.path, None)] = entry

        discover_start_apps = start_apps or cls._discover_start_apps
        try:
            for entry in discover_start_apps():
                entries[(entry.name.casefold(), entry.path, entry.app_id)] = entry
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Could not enumerate registered Windows apps: %s", exc)

        return cls(entries.values(), aliases)

    @staticmethod
    def _discover_start_apps() -> list[AppEntry]:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _START_APPS_COMMAND,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        if not completed.stdout.strip():
            return []
        parsed = json.loads(completed.stdout)
        records = parsed if isinstance(parsed, list) else [parsed]
        entries: list[AppEntry] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            name, app_id = record.get("Name"), record.get("AppID")
            if isinstance(name, str) and name.strip() and isinstance(app_id, str) and app_id.strip():
                entries.append(AppEntry(name=name.strip(), app_id=app_id.strip()))
        return entries

    def resolve(self, requested: str) -> AppResolution:
        """Resolve exact aliases/names first, then conservative fuzzy matches."""

        target = self._normalize(requested)
        alias_target = self.aliases.get(target)
        if alias_target:
            target = self._normalize(alias_target)
        if not target:
            return AppResolution(None)

        exact_matches = [
            entry
            for entry in self.entries
            if self._normalize(entry.name) == target
        ]
        if len(exact_matches) == 1:
            return AppResolution(exact_matches[0])
        if len(exact_matches) > 1:
            return AppResolution(None, tuple(entry.name for entry in exact_matches))
        if alias_target:
            candidate_path = Path(alias_target).expanduser()
            if candidate_path.exists():
                return AppResolution(AppEntry(name=requested.strip(), path=str(candidate_path)))
            if re.fullmatch(r"[A-Za-z0-9._-]+(?:\\[A-Za-z0-9._-]+)*", alias_target):
                return AppResolution(AppEntry(name=requested.strip(), app_id=alias_target))

        names = [entry.name for entry in self.entries]
        matches = process.extract(target, names, scorer=fuzz.WRatio, limit=3)
        if not matches or matches[0][1] < 80:
            return AppResolution(None)
        if len(matches) > 1 and matches[0][1] - matches[1][1] < 8:
            return AppResolution(None, tuple(match[0] for match in matches if match[1] >= 80))
        best_name = matches[0][0]
        best_entries = [entry for entry in self.entries if entry.name == best_name]
        if len(best_entries) != 1:
            return AppResolution(None, tuple(entry.name for entry in best_entries))
        return AppResolution(best_entries[0])

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value.casefold())).strip()


class WindowsActionExecutor:
    """Execute deterministic local actions with injectable platform boundaries."""

    _MEDIA_KEYS = {
        "play_pause": 0xB3,
        "next": 0xB0,
        "previous": 0xB1,
    }

    def __init__(
        self,
        app_index: WindowsAppIndex | None = None,
        *,
        notify: Callable[[str], None] | None = None,
        now: Callable[[], datetime] = datetime.now,
        timer_factory: Callable[..., threading.Timer] = threading.Timer,
        is_windows: bool | None = None,
        browser_open: Callable[[str], bool] = webbrowser.open,
        start_file: Callable[[str], None] | None = None,
        process_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.app_index = app_index if app_index is not None else WindowsAppIndex.discover()
        self.notify = notify or (lambda message: logger.info("Timer notification: %s", message))
        self._now = now
        self._timer_factory = timer_factory
        self._is_windows = os.name == "nt" if is_windows is None else is_windows
        self._browser_open = browser_open
        self._start_file = start_file or self._default_start_file
        self._process_run = process_run
        self._lock = threading.Lock()
        self._scheduled: dict[int, threading.Timer] = {}
        self._next_timer_id = 1

    def execute(self, intent: Intent, *, confirmed: bool = False) -> ActionResult:
        """Run one supported intent, refusing destructive work without confirmation."""

        if intent.requires_confirmation and not confirmed:
            return ActionResult(
                False,
                "Please say 'confirm' to proceed or 'cancel' to stop.",
                confirmation_required=True,
            )
        handlers = {
            "open_app": self._open_app,
            "close_app": self._close_app,
            "open_website": self._open_website,
            "volume": self._volume,
            "brightness": self._brightness,
            "media": self._media,
            "lock_workstation": self._lock_workstation,
            "screenshot": self._screenshot,
            "time": self._time,
            "date": self._date,
            "day": self._day,
            "system_info": self._system_info,
            "timer": self._timer,
            "cancel_timer": lambda _intent: self.cancel_latest_timer(),
            "alarm": self._alarm,
            "system_power": self._system_power,
            "cancel_power": self._cancel_power,
        }
        handler = handlers.get(intent.name)
        if handler is None:
            return ActionResult(False, "That request is not a supported local action.")
        try:
            return handler(intent)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            logger.exception("Local action failed: %s", intent.name)
            return ActionResult(False, f"Could not complete that action: {exc}")

    def cancel_timers(self) -> int:
        """Cancel all scheduled timers and alarms during clean shutdown."""

        with self._lock:
            pending = tuple(self._scheduled.values())
            self._scheduled.clear()
        for timer in pending:
            timer.cancel()
        return len(pending)

    def cancel_latest_timer(self) -> ActionResult:
        """Cancel the most recently scheduled timer or alarm."""

        with self._lock:
            if not self._scheduled:
                return ActionResult(False, "There is no active timer or alarm.")
            timer_id = max(self._scheduled)
            timer = self._scheduled.pop(timer_id)
        timer.cancel()
        return ActionResult(True, f"Canceled timer {timer_id}.")

    def _open_app(self, intent: Intent) -> ActionResult:
        name = str(intent.slots["target"])
        resolution = self.app_index.resolve(name)
        if resolution.entry is None:
            if resolution.candidates:
                options = ", ".join(resolution.candidates[:3])
                return ActionResult(False, f"Which app did you mean: {options}?")
            return ActionResult(False, f"I couldn't find the app {name!r}.")
        entry = resolution.entry
        if entry.path:
            self._start_file(entry.path)
        elif entry.app_id:
            self._start_file(f"shell:AppsFolder\\{entry.app_id}")
        else:
            return ActionResult(False, f"The app {entry.name!r} has no launch target.")
        return ActionResult(True, f"Opening {entry.name}.")

    def _close_app(self, intent: Intent) -> ActionResult:
        if not self._is_windows:
            return ActionResult(False, "Closing apps is supported on Windows only.")
        target = WindowsAppIndex._normalize(str(intent.slots["target"]))
        user32 = ctypes.windll.user32
        matches: list[tuple[int, str, int]] = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def inspect_window(handle: int, _parameter: int) -> bool:
            if not user32.IsWindowVisible(handle):
                return True
            title_length = user32.GetWindowTextLengthW(handle)
            if title_length <= 0:
                return True
            title_buffer = ctypes.create_unicode_buffer(title_length + 1)
            user32.GetWindowTextW(handle, title_buffer, title_length + 1)
            title = title_buffer.value
            score = fuzz.WRatio(target, WindowsAppIndex._normalize(title))
            if score >= 90:
                matches.append((int(handle), title, score))
            return True

        callback = callback_type(inspect_window)
        user32.EnumWindows(callback, 0)
        if not matches:
            return ActionResult(False, f"I couldn't find an open window for {target!r}.")
        matches.sort(key=lambda item: item[2], reverse=True)
        if len(matches) > 1 and matches[0][2] == matches[1][2]:
            choices = ", ".join(item[1] for item in matches[:3])
            return ActionResult(False, f"More than one window matches: {choices}.")
        user32.PostMessageW(matches[0][0], 0x0010, 0, 0)
        return ActionResult(True, f"Asked {matches[0][1]} to close. The app may still prompt to save work.")

    def _open_website(self, intent: Intent) -> ActionResult:
        url = str(intent.slots["target"])
        if not url.startswith("https://"):
            return ActionResult(False, "Only validated HTTPS website addresses can be opened.")
        if not self._browser_open(url):
            return ActionResult(False, "The default browser could not open that website.")
        return ActionResult(True, f"Opening {url}.")

    def _volume(self, intent: Intent) -> ActionResult:
        if not self._is_windows:
            return ActionResult(False, "System volume control is supported on Windows only.")
        from ctypes import POINTER, cast

        import comtypes
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        comtypes.CoInitialize()
        try:
            device = AudioUtilities.GetSpeakers()
            endpoint = cast(device.EndpointVolume, POINTER(IAudioEndpointVolume))
            operation = str(intent.slots["operation"])
            if operation == "mute":
                endpoint.SetMute(1, None)
                return ActionResult(True, "Volume muted.")
            if operation == "unmute":
                endpoint.SetMute(0, None)
                return ActionResult(True, "Volume unmuted.")
            current = float(endpoint.GetMasterVolumeLevelScalar())
            if operation == "set":
                target = float(intent.slots["value"]) / 100.0
            else:
                change = float(intent.slots["amount"]) / 100.0
                target = current + change if operation == "up" else current - change
            target = min(1.0, max(0.0, target))
            endpoint.SetMasterVolumeLevelScalar(target, None)
            return ActionResult(True, f"Volume set to {round(target * 100)} percent.")
        finally:
            comtypes.CoUninitialize()

    @staticmethod
    def _brightness(intent: Intent) -> ActionResult:
        import screen_brightness_control as brightness

        operation = str(intent.slots["operation"])
        if operation == "set":
            target = int(intent.slots["value"])
        else:
            values = brightness.get_brightness()
            current = min(values) if values else None
            if current is None:
                return ActionResult(False, "No adjustable display brightness was found.")
            amount = int(intent.slots["amount"])
            target = current + amount if operation == "up" else current - amount
        target = min(100, max(0, target))
        result = brightness.set_brightness(target, no_return=False)
        if result is None:
            return ActionResult(False, "The display did not confirm a brightness change.")
        return ActionResult(True, f"Brightness set to {target} percent.")

    def _media(self, intent: Intent) -> ActionResult:
        if not self._is_windows:
            return ActionResult(False, "Media controls are supported on Windows only.")
        key = self._MEDIA_KEYS[str(intent.slots["operation"])]
        user32 = ctypes.windll.user32
        user32.keybd_event(key, 0, 0, 0)
        user32.keybd_event(key, 0, 0x0002, 0)
        return ActionResult(True, "Media key sent.")

    def _lock_workstation(self, _intent: Intent) -> ActionResult:
        if not self._is_windows:
            return ActionResult(False, "Locking the workstation is supported on Windows only.")
        if not ctypes.windll.user32.LockWorkStation():
            return ActionResult(False, "Windows could not lock the workstation.")
        return ActionResult(True, "Locking the workstation.")

    def _screenshot(self, _intent: Intent) -> ActionResult:
        if not self._is_windows:
            return ActionResult(False, "Screenshots are supported on Windows only.")
        from PIL import ImageGrab

        destination_dir = Path.home() / "Pictures" / "Chintu Screenshots"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"screenshot-{self._now():%Y%m%d-%H%M%S}.png"
        image = ImageGrab.grab(all_screens=True)
        image.save(destination)
        return ActionResult(True, f"Screenshot saved to {destination}.", str(destination))

    def _time(self, _intent: Intent) -> ActionResult:
        return ActionResult(True, f"It is {self._now():%I:%M %p}.")

    def _date(self, _intent: Intent) -> ActionResult:
        return ActionResult(True, f"Today is {self._now():%B %d, %Y}.")

    def _day(self, _intent: Intent) -> ActionResult:
        return ActionResult(True, f"Today is {self._now():%A}.")

    @staticmethod
    def _system_info(intent: Intent) -> ActionResult:
        import psutil

        metric = str(intent.slots["metric"])
        if metric == "battery":
            battery = psutil.sensors_battery()
            if battery is None:
                return ActionResult(False, "No battery status is available on this device.")
            charging = "and charging" if battery.power_plugged else "on battery"
            return ActionResult(True, f"Battery is at {round(battery.percent)} percent {charging}.")
        if metric == "cpu":
            usage = psutil.cpu_percent(interval=0.1)
            return ActionResult(True, f"CPU usage is {round(usage)} percent.")
        if metric == "memory":
            memory = psutil.virtual_memory()
            return ActionResult(
                True,
                f"Memory usage is {round(memory.percent)} percent "
                f"({memory.used // (1024 ** 3)} of {memory.total // (1024 ** 3)} gigabytes).",
            )
        return ActionResult(False, f"Unsupported system metric: {metric}.")

    def _timer(self, intent: Intent) -> ActionResult:
        seconds = int(intent.slots["seconds"])
        return self._schedule(seconds, "Timer finished.")

    def _alarm(self, intent: Intent) -> ActionResult:
        now = self._now()
        target = now.replace(
            hour=int(intent.slots["hour"]),
            minute=int(intent.slots["minute"]),
            second=0,
            microsecond=0,
        )
        if target <= now:
            target += timedelta(days=1)
        seconds = max(1, round((target - now).total_seconds()))
        return self._schedule(seconds, f"Alarm for {target:%I:%M %p}.")

    def _schedule(self, seconds: int, message: str) -> ActionResult:
        with self._lock:
            timer_id = self._next_timer_id
            self._next_timer_id += 1

            def notify() -> None:
                with self._lock:
                    self._scheduled.pop(timer_id, None)
                self.notify(message)

            timer = self._timer_factory(seconds, notify)
            timer.daemon = True
            self._scheduled[timer_id] = timer
            timer.start()
        return ActionResult(True, f"Scheduled timer {timer_id} for {seconds} seconds.", timer_id)

    def _system_power(self, intent: Intent) -> ActionResult:
        if not self._is_windows:
            return ActionResult(False, "Power actions are supported on Windows only.")
        operation = str(intent.slots["operation"])
        args_by_operation = {
            "shutdown": ["/s", "/t", "60"],
            "restart": ["/r", "/t", "60"],
            "sign_out": ["/l"],
        }
        args = args_by_operation.get(operation)
        if args is None:
            return ActionResult(False, f"Unsupported power action: {operation}.")
        self._process_run(["shutdown.exe", *args], check=True, timeout=5.0)
        if operation == "sign_out":
            return ActionResult(True, "Signing out.")
        return ActionResult(
            True,
            f"{operation.capitalize()} is scheduled in 60 seconds. Say cancel shutdown to abort.",
        )

    def _cancel_power(self, _intent: Intent) -> ActionResult:
        if not self._is_windows:
            return ActionResult(False, "Power actions are supported on Windows only.")
        self._process_run(["shutdown.exe", "/a"], check=True, timeout=5.0)
        return ActionResult(True, "The scheduled shutdown or restart was canceled.")

    @staticmethod
    def _default_start_file(target: str) -> None:
        if not hasattr(os, "startfile"):
            raise RuntimeError("Launching Windows apps is supported on Windows only.")
        os.startfile(target)
