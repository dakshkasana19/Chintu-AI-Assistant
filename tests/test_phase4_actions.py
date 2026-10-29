from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from chintu.actions import (
    ActionResult,
    AppEntry,
    Intent,
    IntentRouter,
    WindowsActionExecutor,
    WindowsAppIndex,
)


@pytest.mark.parametrize(
    ("text", "name", "slots"),
    [
        ("open notepad", "open_app", {"target": "notepad"}),
        ("open youtube", "open_website", {"target": "https://www.youtube.com"}),
        ("open example.com/docs", "open_website", {"target": "https://example.com/docs"}),
        ("close notepad", "close_app", {"target": "notepad"}),
        ("volume up by 10 percent", "volume", {"operation": "up", "amount": 10}),
        ("set volume to 60 percent", "volume", {"operation": "set", "value": 60}),
        ("brightness down by 15", "brightness", {"operation": "down", "amount": 15}),
        ("play music", "media", {"operation": "play_pause"}),
        ("pause", "media", {"operation": "play_pause"}),
        ("lock screen", "lock_workstation", {}),
        ("take a screenshot", "screenshot", {}),
        ("what is the date", "date", {}),
        ("show cpu usage", "system_info", {"metric": "cpu"}),
        ("set a timer for 25 minutes", "timer", {"seconds": 1500}),
        ("set an alarm for 7:30 pm", "alarm", {"hour": 19, "minute": 30}),
        ("what is the capital of France", "information_request", {"query": "what is the capital of France"}),
    ],
)
def test_router_classifies_local_actions_and_information(
    text: str,
    name: str,
    slots: dict[str, object],
) -> None:
    intent = IntentRouter().route(text)

    assert intent.name == name
    assert intent.slots == slots


@pytest.mark.parametrize("text", ["exit", "sleep", "go to sleep now", "stop listening please"])
def test_router_requires_short_sleep_commands(text: str) -> None:
    assert IntentRouter(["exit", "sleep"]).route(text).name == "sleep"


def test_router_does_not_route_exit_word_inside_a_sentence_to_sleep() -> None:
    assert IntentRouter(["exit"]).route("please exit the browser").name == "unknown"


@pytest.mark.parametrize(
    ("text", "operation"),
    [
        ("shutdown", "shutdown"),
        ("restart my computer", "restart"),
        ("sign out", "sign_out"),
    ],
)
def test_router_marks_power_operations_as_confirmation_required(
    text: str,
    operation: str,
) -> None:
    intent = IntentRouter().route(text)

    assert intent.name == "system_power"
    assert intent.slots == {"operation": operation}
    assert intent.requires_confirmation


def test_router_marks_close_app_as_confirmation_required() -> None:
    intent = IntentRouter().route("close notepad")

    assert intent.name == "close_app"
    assert intent.requires_confirmation


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("open example.com", "https://example.com"),
        ("open https://example.com/path?q=1", "https://example.com/path?q=1"),
        ("open javascript:alert(1)", "open_app"),
        ("open notepad.exe", "open_app"),
    ],
)
def test_router_only_opens_validated_http_websites_or_apps(text: str, expected: str) -> None:
    intent = IntentRouter().route(text)

    if expected.startswith("https://"):
        assert intent.name == "open_website"
        assert intent.slots["target"] == expected
    else:
        assert intent.name == expected


def test_app_index_resolves_exact_alias_and_fuzzy_name() -> None:
    index = WindowsAppIndex(
        [
            AppEntry("Notepad", path="C:\\Apps\\notepad.lnk"),
            AppEntry("Calculator", app_id="Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"),
        ],
        {"notes": "Notepad"},
    )

    assert index.resolve("notes").entry == AppEntry(
        "Notepad",
        path="C:\\Apps\\notepad.lnk",
    )
    assert index.resolve("calculatr").entry == AppEntry(
        "Calculator",
        app_id="Microsoft.WindowsCalculator_8wekyb3d8bbwe!App",
    )


def test_app_index_reports_ambiguous_names() -> None:
    index = WindowsAppIndex(
        [
            AppEntry("Visual Studio", path="C:\\Users\\User\\Start Menu\\Visual Studio.lnk"),
            AppEntry("Visual Studio"),
        ]
    )

    resolution = index.resolve("visual studio")

    assert resolution.entry is None
    assert "Visual Studio" in resolution.candidates


def test_app_index_discovers_start_menu_and_uwp_entries(tmp_path: Path) -> None:
    start_menu = tmp_path / "roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    start_menu.mkdir(parents=True)
    (start_menu / "Editor.lnk").touch()
    index = WindowsAppIndex.discover(
        environ={"APPDATA": str(tmp_path / "roaming")},
        start_apps=lambda: [AppEntry("Photos", app_id="Microsoft.Photos_8wekyb3d8bbwe!App")],
    )

    assert index.resolve("editor").entry == AppEntry(
        "Editor",
        path=str(start_menu / "Editor.lnk"),
    )
    assert index.resolve("photos").entry is not None


def test_executor_launches_app_with_safe_startfile_boundary() -> None:
    launched: list[str] = []
    executor = WindowsActionExecutor(
        WindowsAppIndex([AppEntry("Notepad", path="C:\\Apps\\Notepad.lnk")]),
        start_file=launched.append,
    )

    result = executor.execute(Intent("open_app", {"target": "notepad"}))

    assert result.success
    assert launched == ["C:\\Apps\\Notepad.lnk"]


def test_executor_requires_confirmation_before_destructive_action() -> None:
    calls: list[list[str]] = []

    def run(args: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    executor = WindowsActionExecutor(
        WindowsAppIndex(),
        is_windows=True,
        process_run=run,
    )
    intent = Intent("system_power", {"operation": "shutdown"}, requires_confirmation=True)

    result = executor.execute(intent)

    assert not result.success
    assert result.confirmation_required
    assert calls == []

    confirmed = executor.execute(intent, confirmed=True)

    assert confirmed.success
    assert calls == [["shutdown.exe", "/s", "/t", "60"]]


def test_executor_uses_local_clock_for_time_and_date() -> None:
    executor = WindowsActionExecutor(
        WindowsAppIndex(),
        now=lambda: datetime(2026, 10, 29, 19, 37),
    )

    assert executor.execute(Intent("time")).message == "It is 07:37 PM."
    assert executor.execute(Intent("date")).message == "Today is October 29, 2026."
    assert executor.execute(Intent("day")).message == "Today is Thursday."


def test_executor_opens_only_provided_https_target() -> None:
    opened: list[str] = []
    executor = WindowsActionExecutor(
        WindowsAppIndex(),
        browser_open=lambda url: opened.append(url) is None,
    )

    result = executor.execute(
        Intent("open_website", {"target": "https://example.com"})
    )

    assert result.success
    assert opened == ["https://example.com"]
    rejected = executor.execute(Intent("open_website", {"target": "javascript:alert(1)"}))
    assert not rejected.success


def test_executor_schedules_and_cancels_timer() -> None:
    class FakeTimer:
        def __init__(self, interval: float, function) -> None:
            self.interval = interval
            self.function = function
            self.daemon = False
            self.started = False
            self.canceled = False

        def start(self) -> None:
            self.started = True

        def cancel(self) -> None:
            self.canceled = True

    timers: list[FakeTimer] = []

    def make_timer(interval: float, function) -> FakeTimer:
        timer = FakeTimer(interval, function)
        timers.append(timer)
        return timer

    executor = WindowsActionExecutor(WindowsAppIndex(), timer_factory=make_timer)
    scheduled = executor.execute(Intent("timer", {"seconds": 60}))

    assert scheduled.success
    assert timers[0].interval == 60
    assert timers[0].started
    canceled = executor.cancel_latest_timer()

    assert canceled.success
    assert timers[0].canceled
    assert not executor.cancel_latest_timer().success


def test_runtime_confirms_destructive_command_then_executes_it() -> None:
    from types import SimpleNamespace

    from chintu.actions import IntentRouter
    from chintu_assistant import ChintuRuntime

    class Signal:
        def __init__(self) -> None:
            self.values: list[object] = []

        def emit(self, value: object) -> None:
            self.values.append(value)

    bridge = SimpleNamespace(
        status_changed=Signal(),
        query_changed=Signal(),
        response_changed=Signal(),
        visibility_changed=Signal(),
    )
    runtime = ChintuRuntime(bridge)
    invoked: list[tuple[Intent, bool]] = []

    class FakeActions:
        def execute(self, intent: Intent, *, confirmed: bool = False) -> ActionResult:
            invoked.append((intent, confirmed))
            return ActionResult(True, "Shutdown scheduled.")

    runtime._actions = FakeActions()
    runtime._intent_router = IntentRouter()
    runtime._handle_command("shutdown")

    assert invoked == []
    assert runtime._pending_confirmation is not None
    assert "confirm" in bridge.response_changed.values[-1].lower()

    runtime._handle_command("confirm")

    assert invoked[0][0].name == "system_power"
    assert invoked[0][1]
    assert runtime._pending_confirmation is None
