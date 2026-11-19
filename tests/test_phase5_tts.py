from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chintu.config import AssistantConfig
from chintu.state import AssistantState
from chintu_assistant import ChintuRuntime
from scripts import download_models
from chintu.tts import (
    Pyttsx3TTSEngine,
    TTSNormalizer,
    TextToSpeech,
    create_tts_engine,
)


def test_tts_normalizer_cleans_markdown() -> None:
    raw = "Here is **bold** text, *italic* words, and `inline_code`. Check [Docs](https://example.com/info)."
    normalized = TTSNormalizer.normalize(raw)

    assert "bold" in normalized
    assert "italic" in normalized
    assert "inline code" in normalized or "inline_code" in normalized
    assert "Docs" in normalized
    assert "https://" not in normalized
    assert "*" not in normalized
    assert "`" not in normalized


def test_tts_normalizer_expands_units_and_currencies() -> None:
    text = "The price is $50 or ₹100. Temperature is 24°C with 65% humidity at 15km/h."
    normalized = TTSNormalizer.normalize(text)

    assert "50 dollars" in normalized
    assert "100 rupees" in normalized
    assert "24 degrees Celsius" in normalized
    assert "65 percent" in normalized
    assert "15 kilometers per hour" in normalized


def test_tts_normalizer_expands_abbreviations() -> None:
    text = "Dr. Stone estimated approx. 10 mins and 30 secs remaining, e.g. for etc."
    normalized = TTSNormalizer.normalize(text)

    assert "doctor" in normalized.lower()
    assert "approximately" in normalized.lower()
    assert "minutes" in normalized.lower()
    assert "seconds" in normalized.lower()
    assert "for example" in normalized.lower()
    assert "etcetera" in normalized.lower()


def test_tts_normalizer_expands_decimals() -> None:
    text = "The value of pi is 3.14."
    normalized = TTSNormalizer.normalize(text)

    assert "3 point 14" in normalized


class MockEngine:
    def __init__(self, name: str = "mock-tts") -> None:
        self._name = name
        self.synthesized: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.synthesized.append(text)
        sample_rate = 16000
        # 100 ms of mock audio
        samples = np.zeros(int(sample_rate * 0.1), dtype=np.float32)
        return samples, sample_rate


def test_create_tts_engine_falls_back_to_pyttsx3(tmp_path: Path) -> None:
    config = AssistantConfig(
        tts_engine="auto",
        tts_piper_model="non_existent.onnx",
        model_dir=str(tmp_path),
    )
    engine = create_tts_engine(config)

    assert isinstance(engine, Pyttsx3TTSEngine)
    assert engine.name == "pyttsx3"


def test_create_tts_engine_explicit_piper_missing_raises(tmp_path: Path) -> None:
    config = AssistantConfig(
        tts_engine="piper",
        tts_piper_model="missing.onnx",
        model_dir=str(tmp_path),
    )
    with pytest.raises(FileNotFoundError, match="Piper model not found"):
        create_tts_engine(config)


def test_piper_model_download_publishes_complete_files_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            content, self.content = self.content, b""
            return content

    responses = iter((Response(b'{"voice": "test"}'), Response(b"onnx-model")))
    monkeypatch.setattr(
        download_models.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: next(responses),
    )

    model_path = download_models.download_piper_model(tmp_path)
    config_path = model_path.with_suffix(model_path.suffix + ".json")

    assert model_path.read_bytes() == b"onnx-model"
    assert config_path.read_bytes() == b'{"voice": "test"}'
    assert not config_path.with_suffix(config_path.suffix + ".part").exists()


def test_interrupted_piper_config_download_does_not_publish_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedResponse:
        def __enter__(self) -> FailedResponse:
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            raise OSError("simulated interrupted download")

    monkeypatch.setattr(
        download_models.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FailedResponse(),
    )

    with pytest.raises(OSError, match="simulated interrupted download"):
        download_models.download_piper_model(tmp_path)

    config_path = (
        tmp_path
        / "piper"
        / download_models.PIPER_JSON_FILENAME
    )
    assert not config_path.exists()


def test_tts_playback_lifecycle(caplog: pytest.LogCaptureFixture) -> None:
    config = AssistantConfig()
    engine = MockEngine()
    started = []
    ended = []

    tts = TextToSpeech(
        config=config,
        engine=engine,
        on_speaking_start=lambda: started.append(True),
        on_speaking_end=lambda: ended.append(True),
    )

    try:
        assert not tts.is_speaking
        with patch("sounddevice.OutputStream"):
            tts.speak("Hello from Chintu assistant")
            # Wait for synthesis and playback loop
            for _ in range(50):
                if ended:
                    break
                time.sleep(0.05)

        assert engine.synthesized == ["Hello from Chintu assistant"]
        assert len(started) == 1
        assert len(ended) == 1
        assert not tts.is_speaking
        assert "Hello from Chintu assistant" not in caplog.text
    finally:
        tts.stop()


def test_tts_interrupt_clears_queue() -> None:
    config = AssistantConfig()
    engine = MockEngine()

    tts = TextToSpeech(config=config, engine=engine)
    try:
        with patch("sounddevice.OutputStream"), patch("sounddevice.stop") as mock_sd_stop:
            tts.speak("First long sentence that gets interrupted")
            tts.speak("Second sentence that also gets queued", interrupt_current=False)
            tts.interrupt()
            mock_sd_stop.assert_not_called()

        for _ in range(50):
            if not tts.is_speaking:
                break
            time.sleep(0.01)
        assert not tts.is_speaking
    finally:
        tts.stop()


def test_new_speech_cancels_stale_synthesis_before_playback() -> None:
    config = AssistantConfig()
    synthesis_started = threading.Event()
    release_synthesis = threading.Event()
    synthesized: list[str] = []
    written_chunks: list[np.ndarray] = []

    class BlockingEngine:
        @property
        def name(self) -> str:
            return "blocking-test"

        def synthesize(self, text: str) -> tuple[np.ndarray, int]:
            synthesized.append(text)
            if text == "Old response":
                synthesis_started.set()
                assert release_synthesis.wait(timeout=1.0)
            return np.zeros(1600, dtype=np.float32), 16000

    class OutputStream:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self) -> OutputStream:
            return self

        def __exit__(self, *_args) -> None:
            return None

        def write(self, samples: np.ndarray) -> None:
            written_chunks.append(samples)

    tts = TextToSpeech(config=config, engine=BlockingEngine())
    try:
        with patch("sounddevice.OutputStream", OutputStream):
            tts.speak("Old response")
            assert synthesis_started.wait(timeout=1.0)
            tts.speak("New response")
            release_synthesis.set()
            for _ in range(100):
                if len(written_chunks) >= 2 and not tts.is_speaking:
                    break
                time.sleep(0.01)

        assert synthesized == ["Old response", "New response"]
        assert len(written_chunks) == 2
    finally:
        release_synthesis.set()
        tts.stop()


def test_vad_speech_onset_interrupts_tts_without_echo_gate() -> None:
    bridge = SimpleNamespace(status_changed=MagicMock())
    runtime = ChintuRuntime(bridge, AssistantConfig(tts_enabled=False))
    interrupt = MagicMock()
    runtime._tts = SimpleNamespace(interrupt=interrupt)
    runtime.state.transition(AssistantState.WAKE_CONFIRM)
    runtime.state.transition(AssistantState.LISTENING)
    runtime.state.transition(AssistantState.PROCESSING)
    runtime.state.transition(AssistantState.SPEAKING)

    runtime._interrupt_tts_for_speech()

    assert runtime.state.state is AssistantState.LISTENING
    assert runtime._tts_barge_in.is_set()
    assert runtime._tts_echo_gate_until == 0.0
    interrupt.assert_called_once_with()

    runtime._on_speaking_end()

    assert not runtime._tts_barge_in.is_set()
    assert runtime._tts_echo_gate_until == 0.0
