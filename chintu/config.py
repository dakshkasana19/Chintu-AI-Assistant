from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli


ROOT = Path(__file__).resolve().parent.parent
VOSK_MODEL_NAME = "vosk-model-small-en-in-0.4"


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
    vad_preroll_ms: int = 300
    vad_start_speech_ms: int = 90
    vad_end_silence_ms: int = 700
    max_utterance_seconds: int = 15
    silero_threshold: float = 0.5
    webrtc_aggressiveness: int = 2
    audio_recovery_seconds: float = 1.0
    wake_confirm_window_seconds: float = 2.0
    wake_sensitivity: float = 0.5
    wake_cooldown_seconds: float = 2.0
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
            "vad_preroll_ms": 300,
            "vad_start_speech_ms": 90,
            "vad_end_silence_ms": 700,
            "max_utterance_seconds": 15,
            "silero_threshold": 0.5,
            "webrtc_aggressiveness": 2,
            "audio_recovery_seconds": 1.0,
            "wake_confirm_window_seconds": 2.0,
            "wake_sensitivity": 0.5,
            "wake_cooldown_seconds": 2.0,
            "logging_level": "INFO",
            "log_dir": "logs",
            "model_dir": "models",
        }
    }

    data = defaults
    if config_file.exists():
        with config_file.open("rb") as fh:
            try:
                parsed = tomli.load(fh)
            except tomli.TOMLDecodeError as exc:
                raise ValueError(f"Invalid TOML configuration in {config_file}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Configuration root in {config_file} must be a TOML table")
        data = _deep_merge(defaults, parsed)

    assistant = data.get("assistant", {})
    if not isinstance(assistant, dict):
        raise ValueError(f"'assistant' in {config_file} must be a TOML table")

    logging_level = str(assistant.get("logging_level", defaults["assistant"]["logging_level"])).upper()
    if not isinstance(getattr(logging, logging_level, None), int):
        raise ValueError(f"Unsupported logging_level {logging_level!r} in {config_file}")

    config = AssistantConfig(
        wake_words=assistant.get("wake_words", defaults["assistant"]["wake_words"]),
        exit_phrases=assistant.get("exit_phrases", defaults["assistant"]["exit_phrases"]),
        sample_rate_hz=int(assistant.get("sample_rate_hz", defaults["assistant"]["sample_rate_hz"])),
        frame_size_ms=int(assistant.get("frame_size_ms", defaults["assistant"]["frame_size_ms"])),
        vad_preroll_ms=int(assistant.get("vad_preroll_ms", defaults["assistant"]["vad_preroll_ms"])),
        vad_start_speech_ms=int(
            assistant.get("vad_start_speech_ms", defaults["assistant"]["vad_start_speech_ms"])
        ),
        vad_end_silence_ms=int(
            assistant.get("vad_end_silence_ms", defaults["assistant"]["vad_end_silence_ms"])
        ),
        max_utterance_seconds=int(
            assistant.get("max_utterance_seconds", defaults["assistant"]["max_utterance_seconds"])
        ),
        silero_threshold=float(
            assistant.get("silero_threshold", defaults["assistant"]["silero_threshold"])
        ),
        webrtc_aggressiveness=int(
            assistant.get(
                "webrtc_aggressiveness",
                defaults["assistant"]["webrtc_aggressiveness"],
            )
        ),
        audio_recovery_seconds=float(
            assistant.get(
                "audio_recovery_seconds",
                defaults["assistant"]["audio_recovery_seconds"],
            )
        ),
        wake_confirm_window_seconds=float(
            assistant.get(
                "wake_confirm_window_seconds",
                defaults["assistant"]["wake_confirm_window_seconds"],
            )
        ),
        wake_sensitivity=float(
            assistant.get("wake_sensitivity", defaults["assistant"]["wake_sensitivity"])
        ),
        wake_cooldown_seconds=float(
            assistant.get(
                "wake_cooldown_seconds",
                defaults["assistant"]["wake_cooldown_seconds"],
            )
        ),
        logging_level=logging_level,
        log_dir=str(assistant.get("log_dir", defaults["assistant"]["log_dir"])),
        model_dir=str(assistant.get("model_dir", defaults["assistant"]["model_dir"])),
    )
    if config.sample_rate_hz != 16000:
        raise ValueError("Phase 1 audio capture currently requires sample_rate_hz = 16000")
    if config.frame_size_ms not in {10, 20, 30}:
        raise ValueError("frame_size_ms must be 10, 20, or 30 for the VAD backends")
    if min(
        config.vad_preroll_ms,
        config.vad_start_speech_ms,
        config.vad_end_silence_ms,
        config.max_utterance_seconds,
    ) <= 0:
        raise ValueError("VAD durations and max_utterance_seconds must be positive")
    if not 0.0 < config.silero_threshold < 1.0:
        raise ValueError("silero_threshold must be between 0 and 1")
    if not 0 <= config.webrtc_aggressiveness <= 3:
        raise ValueError("webrtc_aggressiveness must be between 0 and 3")
    if config.audio_recovery_seconds <= 0:
        raise ValueError("audio_recovery_seconds must be positive")
    if config.wake_confirm_window_seconds <= 0:
        raise ValueError("wake_confirm_window_seconds must be positive")
    if not 0.0 <= config.wake_sensitivity <= 1.0:
        raise ValueError("wake_sensitivity must be between 0 and 1")
    if config.wake_cooldown_seconds < 0:
        raise ValueError("wake_cooldown_seconds must not be negative")
    return config
