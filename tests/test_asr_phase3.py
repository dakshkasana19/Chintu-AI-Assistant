from __future__ import annotations

import threading
import time
import wave
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from chintu.asr import FasterWhisperASR, TranscriptionResult, normalize_command_text
from chintu.audio.vad import SpeechSegment
from chintu.config import AssistantConfig, load_config
from chintu.state import AssistantState
from scripts.eval_asr import _word_error_counts, evaluate_directory


class FakeWhisperModel:
    def __init__(
        self,
        text: str = "Chintu, please set a timer for twenty five minutes",
        *,
        avg_logprob: float = -0.2,
        no_speech_prob: float = 0.01,
        compression_ratio: float = 1.1,
    ) -> None:
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.compression_ratio = compression_ratio
        self.audio: np.ndarray | None = None
        self.options: dict[str, object] = {}
        self.calls = 0

    def transcribe(self, audio: np.ndarray, **kwargs: object):
        self.audio = audio.copy()
        self.options = kwargs
        self.calls += 1
        segment = SimpleNamespace(
            text=self.text,
            avg_logprob=self.avg_logprob,
            no_speech_prob=self.no_speech_prob,
            compression_ratio=self.compression_ratio,
        )
        return iter([segment]), SimpleNamespace()


def make_segment(
    samples: np.ndarray | None = None,
    sample_rate_hz: int = 16000,
) -> SpeechSegment:
    audio = (
        np.full(16000, 0.25, dtype=np.float32)
        if samples is None
        else samples
    )
    return SpeechSegment(
        samples=audio,
        sample_rate_hz=sample_rate_hz,
        duration_seconds=audio.size / sample_rate_hz,
        peak_rms=float(np.max(np.abs(audio))) if audio.size else 0.0,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Chintu, please open Notepad", "open notepad"),
        ("Can you set a timer for twenty five minutes, please?", "set a timer for 25 minutes"),
        ("Would you please open calculator", "open calculator"),
        ("Please, what's the time?", "what's the time"),
    ],
)
def test_normalize_command_text(raw: str, expected: str) -> None:
    assert normalize_command_text(raw) == expected


def test_command_asr_uses_local_cpu_model_and_returns_raw_and_normalized_text() -> None:
    model = FakeWhisperModel()
    asr = FasterWhisperASR(AssistantConfig(), model=model)

    result = asr.transcribe(make_segment())

    assert result == TranscriptionResult(
        raw_text="Chintu, please set a timer for twenty five minutes",
        normalized_text="set a timer for 25 minutes",
        accepted=True,
        avg_logprob=-0.2,
        no_speech_prob=0.01,
        compression_ratio=1.1,
    )
    assert model.audio is not None
    assert model.audio.size == 16000
    assert model.options["language"] == "en"
    assert model.options["beam_size"] == 5
    assert model.options["condition_on_previous_text"] is False
    assert model.options["vad_filter"] is False
    assert model.options["initial_prompt"] == "Chintu"
    assert model.options["hotwords"] == "Chintu"
    assert model.options["no_speech_threshold"] == 0.6


def test_command_asr_rejects_silence_without_calling_whisper() -> None:
    model = FakeWhisperModel()
    asr = FasterWhisperASR(AssistantConfig(), model=model)

    result = asr.transcribe(make_segment(np.zeros(16000, dtype=np.float32)))

    assert not result.accepted
    assert result.reason == "silent_audio"
    assert model.calls == 0


@pytest.mark.parametrize(
    ("model_values", "reason"),
    [
        ({"no_speech_prob": 0.9}, "no_speech_probability"),
        ({"avg_logprob": -2.0}, "low_log_probability"),
        ({"compression_ratio": 3.0}, "high_compression_ratio"),
    ],
)
def test_command_asr_rejects_low_confidence_transcription(
    model_values: dict[str, float],
    reason: str,
) -> None:
    model = FakeWhisperModel(**model_values)
    asr = FasterWhisperASR(AssistantConfig(), model=model)

    result = asr.transcribe(make_segment())

    assert not result.accepted
    assert result.reason == reason
    assert result.normalized_text == ""


def test_command_asr_rejects_non_16khz_audio() -> None:
    asr = FasterWhisperASR(AssistantConfig(), model=FakeWhisperModel())

    with pytest.raises(ValueError, match="requires 16 kHz"):
        asr.transcribe(make_segment(sample_rate_hz=8000))


def test_command_asr_rejects_non_finite_audio() -> None:
    model = FakeWhisperModel()
    asr = FasterWhisperASR(AssistantConfig(), model=model)

    result = asr.transcribe(
        make_segment(np.full(100, np.nan, dtype=np.float32))
    )

    assert not result.accepted
    assert result.reason == "empty_or_invalid_audio"
    assert model.calls == 0


def test_asr_configuration_loads_and_validates_values(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[assistant]\nasr_model_size = "small.en"\n'
        "asr_no_speech_threshold = 0.7\n"
        "asr_log_prob_threshold = -0.8\n"
        "asr_compression_ratio_threshold = 2.2\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.asr_model_size == "small.en"
    assert config.asr_no_speech_threshold == 0.7
    assert config.asr_log_prob_threshold == -0.8
    assert config.asr_compression_ratio_threshold == 2.2


def test_asr_configuration_rejects_unsupported_model(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('[assistant]\nasr_model_size = "unknown"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported asr_model_size"):
        load_config(config_path)


def test_runtime_routes_sleeping_and_awake_segments_to_separate_queues() -> None:
    from chintu_assistant import ChintuRuntime

    runtime = ChintuRuntime(SimpleNamespace(), AssistantConfig())
    segment = make_segment()
    runtime._asr = FasterWhisperASR(AssistantConfig(), model=FakeWhisperModel())

    runtime._queue_speech_segment(segment)

    assert runtime._wake_segments.get_nowait() is segment
    assert runtime._command_segments.empty()

    runtime.state.transition(AssistantState.WAKE_CONFIRM)
    runtime.state.transition(AssistantState.LISTENING)
    runtime._queue_speech_segment(segment)

    assert runtime._wake_segments.empty()
    assert runtime._command_segments.get_nowait() is segment


def test_command_asr_worker_emits_transcript_and_returns_to_listening() -> None:
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
    )
    runtime = ChintuRuntime(bridge, AssistantConfig())
    from chintu.actions import ActionResult

    class FakeActions:
        def execute(self, _intent, *, confirmed: bool = False) -> ActionResult:
            assert not confirmed
            return ActionResult(True, "open notepad")

    runtime._actions = FakeActions()
    runtime.state.transition(AssistantState.WAKE_CONFIRM)
    runtime.state.transition(AssistantState.LISTENING)
    expected = TranscriptionResult(
        raw_text="Chintu, open Notepad",
        normalized_text="open notepad",
        accepted=True,
    )

    class FakeASR:
        def transcribe(self, segment: SpeechSegment) -> TranscriptionResult:
            assert segment.sample_rate_hz == 16000
            return expected

    runtime._command_segments.put_nowait(make_segment())
    worker = threading.Thread(target=runtime._command_asr_loop, args=(FakeASR(),))
    worker.start()
    deadline = time.monotonic() + 2.0
    while runtime.latest_transcription is None and time.monotonic() < deadline:
        time.sleep(0.01)
    runtime._command_stopping.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert runtime.latest_transcription == expected
    assert bridge.query_changed.values == ["Chintu, open Notepad"]
    assert bridge.response_changed.values == ["open notepad"]
    assert runtime.state.state is AssistantState.LISTENING


def test_asr_evaluation_reports_raw_and_normalized_command_wer(tmp_path: Path) -> None:
    audio_path = tmp_path / "open_notepad.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(np.full(1600, 1000, dtype="<i2").tobytes())
    audio_path.with_suffix(".txt").write_text("Open Notepad", encoding="utf-8")

    class FakeASR:
        config = AssistantConfig()

        def transcribe(self, segment: SpeechSegment) -> TranscriptionResult:
            assert segment.sample_rate_hz == 16000
            return TranscriptionResult(
                raw_text="Chintu, open Notepad",
                normalized_text="open notepad",
                accepted=True,
            )

    report = evaluate_directory(tmp_path, FakeASR())

    assert report["utterance_count"] == 1
    assert report["raw_wer"]["wer"] == pytest.approx(0.5)
    assert report["normalized_command_wer"]["wer"] == 0.0
    assert report["utterances"][0]["normalized_command"] == "open notepad"


def test_asr_word_error_count_tracks_substitution_deletion_and_insertion() -> None:
    counts = _word_error_counts(
        ["open", "notepad"],
        ["open", "calculator", "now"],
    )

    assert counts == {
        "substitutions": 1,
        "deletions": 0,
        "insertions": 1,
        "errors": 2,
        "reference_words": 2,
    }
