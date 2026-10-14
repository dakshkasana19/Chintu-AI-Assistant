from __future__ import annotations

from pathlib import Path

import pytest

from chintu.config import AssistantConfig, load_config
from chintu.state import AssistantState, StateMachine


def test_config_defaults_when_file_is_missing(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.toml")

    assert isinstance(config, AssistantConfig)
    assert config.sample_rate_hz == 16000
    assert config.frame_size_ms == 30
    assert config.vad_preroll_ms == 300
    assert config.vad_start_speech_ms == 90
    assert config.vad_end_silence_ms == 700
    assert config.logging_level == "INFO"


def test_config_loads_values_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[assistant]\nsample_rate_hz = 16000\nvad_end_silence_ms = 500\nlogging_level = "DEBUG"\n',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.sample_rate_hz == 16000
    assert config.logging_level == "DEBUG"
    assert config.frame_size_ms == 30
    assert config.vad_end_silence_ms == 500


def test_config_reports_invalid_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[assistant\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid TOML configuration"):
        load_config(config_path)


def test_config_rejects_invalid_logging_level(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('[assistant]\nlogging_level = "LOUD"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported logging_level"):
        load_config(config_path)


def test_state_machine_enforces_lifecycle_transitions() -> None:
    machine = StateMachine()

    assert machine.transition(AssistantState.WAKE_CONFIRM) is AssistantState.WAKE_CONFIRM
    assert machine.transition(AssistantState.LISTENING) is AssistantState.LISTENING
    assert machine.transition(AssistantState.PROCESSING) is AssistantState.PROCESSING
    assert machine.transition(AssistantState.SPEAKING) is AssistantState.SPEAKING
    assert machine.transition(AssistantState.LISTENING) is AssistantState.LISTENING

    with pytest.raises(ValueError, match="Invalid assistant state transition"):
        machine.transition(AssistantState.WAKE_CONFIRM)


def test_runtime_ui_startup_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt6.QtWidgets import QApplication

    from chintu.ui import ChintuCoreUI, UIBridge
    from chintu_assistant import ChintuRuntime

    monkeypatch.setattr(ChintuRuntime, "_audio_loop", lambda self: None)
    app = QApplication.instance() or QApplication([])
    bridge = UIBridge()
    window = ChintuCoreUI(bridge)
    runtime = ChintuRuntime(bridge)

    runtime.start()
    app.processEvents()
    assert window.isVisible()
    assert window.status_text == "STANDBY"

    runtime.stop()
    app.processEvents()
    assert not window.isVisible()
    assert runtime.state.state is AssistantState.SLEEPING
    window.close()
    app.processEvents()
