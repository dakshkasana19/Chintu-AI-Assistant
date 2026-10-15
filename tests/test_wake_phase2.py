from __future__ import annotations

import json
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chintu.audio import SpeechSegment, VoiceActivityDetector, VoskWhisperWakeWordDetector, WakeWordMatch
from chintu.config import AssistantConfig
from chintu.state import AssistantState
from scripts import download_models
from scripts.eval_wake import summarize_counts, vad_segments


class FakeRecognizer:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.payload = b""

    def AcceptWaveform(self, data: bytes) -> bool:
        self.payload = data
        return True

    def FinalResult(self) -> str:
        return json.dumps({"text": self.transcript})


class FakeWhisper:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.audio: np.ndarray | None = None
        self.calls = 0

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str,
        beam_size: int,
        condition_on_previous_text: bool,
        vad_filter: bool,
    ):
        assert language == "en"
        assert beam_size == 1
        assert condition_on_previous_text is False
        assert vad_filter is False
        self.audio = audio.copy()
        self.calls += 1
        return iter([SimpleNamespace(text=self.transcript)]), SimpleNamespace()


def make_segment(duration_seconds: float = 3.0) -> SpeechSegment:
    samples = np.full(round(duration_seconds * 16000), 0.25, dtype=np.float32)
    return SpeechSegment(
        samples=samples,
        sample_rate_hz=16000,
        duration_seconds=duration_seconds,
        peak_rms=0.25,
    )


def test_wake_detection_requires_vosk_candidate_and_whisper_confirmation() -> None:
    config = AssistantConfig(wake_confirm_window_seconds=2.0)
    whisper = FakeWhisper("hey chintoo")
    recognizers: list[FakeRecognizer] = []

    def make_recognizer(_model: object, sample_rate: int, grammar: str) -> FakeRecognizer:
        assert sample_rate == 16000
        assert json.loads(grammar) == ["chintu", "hey chintu", "ok chintu", "[unk]"]
        recognizer = FakeRecognizer("hey chintu")
        recognizers.append(recognizer)
        return recognizer

    detector = VoskWhisperWakeWordDetector(
        config,
        vosk_model=object(),
        whisper_model=whisper,
        recognizer_factory=make_recognizer,
    )

    result = detector.detect(make_segment())

    assert result is not None
    assert result.phrase == "hey chintu"
    assert result.confirmation_score == 100.0
    assert result.vosk_transcript == "hey chintu"
    assert result.whisper_transcript == "hey chintoo"
    assert whisper.audio is not None
    assert whisper.audio.size == 2 * 16000
    assert len(recognizers) == 1
    assert len(recognizers[0].payload) == 2 * 16000 * 2


def test_vosk_rejection_skips_whisper_confirmation() -> None:
    whisper = FakeWhisper("chintu")
    detector = VoskWhisperWakeWordDetector(
        AssistantConfig(),
        vosk_model=object(),
        whisper_model=whisper,
        recognizer_factory=lambda *_args: FakeRecognizer("[unk]"),
    )

    assert detector.detect(make_segment()) is None
    assert whisper.calls == 0


def test_wake_confirmation_obeys_cooldown() -> None:
    now = [10.0]
    whisper = FakeWhisper("chintu")
    detector = VoskWhisperWakeWordDetector(
        AssistantConfig(wake_cooldown_seconds=3.0),
        vosk_model=object(),
        whisper_model=whisper,
        recognizer_factory=lambda *_args: FakeRecognizer("chintu"),
        clock=lambda: now[0],
    )

    assert detector.detect(make_segment()) is not None
    assert detector.detect(make_segment()) is None
    assert whisper.calls == 1

    now[0] += 3.0
    assert detector.detect(make_segment()) is not None
    assert whisper.calls == 2


def test_wake_sensitivity_controls_fuzzy_confirmation_threshold() -> None:
    def build_detector(sensitivity: float) -> VoskWhisperWakeWordDetector:
        return VoskWhisperWakeWordDetector(
            AssistantConfig(wake_sensitivity=sensitivity),
            vosk_model=object(),
            whisper_model=FakeWhisper("chintx"),
            recognizer_factory=lambda *_args: FakeRecognizer("chintu"),
        )

    assert build_detector(0.0).detect(make_segment()) is None
    assert build_detector(1.0).detect(make_segment()) is not None


def test_wake_detection_rejects_wrong_sample_rate() -> None:
    detector = VoskWhisperWakeWordDetector(
        AssistantConfig(),
        vosk_model=object(),
        whisper_model=FakeWhisper("chintu"),
        recognizer_factory=lambda *_args: FakeRecognizer("chintu"),
    )
    segment = SpeechSegment(np.zeros(100, dtype=np.float32), 8000, 0.01, 0.0)

    with pytest.raises(ValueError, match="requires 16 kHz"):
        detector.detect(segment)


def test_vosk_model_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "not allowed")

    with pytest.raises(ValueError, match="Unsafe path"):
        download_models.extract_vosk_model(
            archive_path,
            tmp_path / "models" / "vosk-model-small-en-in-0.4",
        )

    assert not (tmp_path / "outside.txt").exists()


def test_vosk_model_extraction_installs_expected_model_tree(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.zip"
    model_name = download_models.VOSK_MODEL_NAME
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{model_name}/am/final.mdl", "model")

    destination = tmp_path / "models" / model_name
    result = download_models.extract_vosk_model(archive_path, destination)

    assert result == destination
    assert (destination / "am" / "final.mdl").read_text(encoding="utf-8") == "model"


def test_wake_evaluation_reports_miss_and_false_accept_rates() -> None:
    results = summarize_counts(
        positive_count=20,
        missed=3,
        negative_seconds=1800,
        false_accepts=2,
    )

    assert results["miss_rate"] == pytest.approx(0.15)
    assert results["false_accepts_per_hour"] == pytest.approx(4.0)


def test_wake_evaluation_rejects_empty_sets() -> None:
    with pytest.raises(ValueError, match="positive samples"):
        summarize_counts(0, 0, 3600, 0)
    with pytest.raises(ValueError, match="negative audio"):
        summarize_counts(1, 0, 0, 0)


class ThresholdVAD:
    name = "test-threshold"

    def is_speech(self, frame: np.ndarray) -> bool:
        return bool(np.max(np.abs(frame)) > 0.1)


def test_wake_evaluation_only_emits_vad_gated_segments() -> None:
    config = AssistantConfig(
        frame_size_ms=30,
        vad_preroll_ms=30,
        vad_start_speech_ms=30,
        vad_end_silence_ms=60,
    )
    detector = VoiceActivityDetector(config, ThresholdVAD())
    silence = np.zeros(16000, dtype=np.float32)

    assert vad_segments(silence, 16000, config, detector) == []

    detector.reset()
    voice = np.concatenate(
        [
            np.full(480, 0.3, dtype=np.float32),
            np.full(480, 0.3, dtype=np.float32),
            np.zeros(480 * 3, dtype=np.float32),
        ]
    )
    segments = vad_segments(voice, 16000, config, detector)

    assert len(segments) == 1


def test_runtime_wakes_only_after_detector_confirmation() -> None:
    from chintu_assistant import ChintuRuntime

    emissions: list[object] = []

    class Signal:
        def emit(self, value: object) -> None:
            emissions.append(value)

    bridge = SimpleNamespace(
        status_changed=Signal(),
        response_changed=Signal(),
        visibility_changed=Signal(),
        microphone_rms_changed=Signal(),
    )
    runtime = ChintuRuntime(bridge, AssistantConfig())
    match = WakeWordMatch("chintu", "chintu", "chintu", 100.0)

    class ConfirmingDetector:
        def detect(self, _segment: SpeechSegment) -> WakeWordMatch:
            return match

    runtime._queue_wake_confirmation(make_segment(0.5))
    worker = threading.Thread(target=runtime._wake_loop, args=(ConfirmingDetector(),))
    worker.start()
    deadline = time.monotonic() + 2.0
    while runtime.state.state is AssistantState.SLEEPING and time.monotonic() < deadline:
        time.sleep(0.01)
    runtime._stopping.set()
    runtime._wake_stopping.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert runtime.state.state is AssistantState.LISTENING
    assert "Wake word confirmed" in emissions
