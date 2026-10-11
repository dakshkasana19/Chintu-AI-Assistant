from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AssistantConfig:
    """Default runtime configuration for the Chintu assistant."""

    wake_words: list[str] = field(
        default_factory=lambda: [
            "chintu",
            "hey chintu",
            "ok chintu",
            "chinto",
            "chintoo",
        ]
    )
    exit_phrases: list[str] = field(
        default_factory=lambda: [
            "exit",
            "sleep",
            "go to sleep",
            "goodbye",
            "stop listening",
        ]
    )
    sample_rate_hz: int = 16000
    frame_size_ms: int = 30
    vad_start_silence_ms: int = 300
    vad_end_silence_ms: int = 700
    max_utterance_seconds: int = 15
    wake_confirm_window_seconds: float = 2.0
    logging_level: str = "INFO"
    log_dir: str = "logs"
    model_dir: str = "models"

    @property
    def log_path(self) -> Path:
        return ROOT / self.log_dir

    @property
    def model_path(self) -> Path:
        return ROOT / self.model_dir


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> AssistantConfig:
    """Load configuration from config.toml with sane defaults as a fallback."""

    config_file = Path(config_path) if config_path else ROOT / "config.toml"
    defaults = {
        "assistant": {
            "wake_words": [
                "chintu",
                "hey chintu",
                "ok chintu",
                "chinto",
                "chintoo",
            ],
            "exit_phrases": [
                "exit",
                "sleep",
                "go to sleep",
                "goodbye",
                "stop listening",
            ],
            "sample_rate_hz": 16000,
            "frame_size_ms": 30,
            "vad_start_silence_ms": 300,
            "vad_end_silence_ms": 700,
            "max_utterance_seconds": 15,
            "wake_confirm_window_seconds": 2.0,
            "logging_level": "INFO",
            "log_dir": "logs",
            "model_dir": "models",
        }
    }

    data = defaults
    if config_file.exists():
        with config_file.open("rb") as fh:
            try:
                parsed = tomllib.load(fh)
            except Exception:
                parsed = {}
        data = _deep_merge(defaults, parsed)

    assistant = data.get("assistant", {})
    return AssistantConfig(
        wake_words=assistant.get("wake_words", defaults["assistant"]["wake_words"]),
        exit_phrases=assistant.get("exit_phrases", defaults["assistant"]["exit_phrases"]),
        sample_rate_hz=int(assistant.get("sample_rate_hz", defaults["assistant"]["sample_rate_hz"])),
        frame_size_ms=int(assistant.get("frame_size_ms", defaults["assistant"]["frame_size_ms"])),
        vad_start_silence_ms=int(
            assistant.get("vad_start_silence_ms", defaults["assistant"]["vad_start_silence_ms"])
        ),
        vad_end_silence_ms=int(
            assistant.get("vad_end_silence_ms", defaults["assistant"]["vad_end_silence_ms"])
        ),
        max_utterance_seconds=int(
            assistant.get("max_utterance_seconds", defaults["assistant"]["max_utterance_seconds"])
        ),
        wake_confirm_window_seconds=float(
            assistant.get(
                "wake_confirm_window_seconds",
                defaults["assistant"]["wake_confirm_window_seconds"],
            )
        ),
        logging_level=str(assistant.get("logging_level", defaults["assistant"]["logging_level"])),
        log_dir=str(assistant.get("log_dir", defaults["assistant"]["log_dir"])),
        model_dir=str(assistant.get("model_dir", defaults["assistant"]["model_dir"])),
    )
