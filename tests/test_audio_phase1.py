from __future__ import annotations

import hashlib
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chintu.audio.capture import AudioCapture, AudioCaptureError
from chintu.audio.vad import SpeechSegment, VoiceActivityDetector, create_vad_backend
from chintu.config import AssistantConfig
from scripts import download_models
from chintu.audio import capture as capture_module


class ThresholdVAD:
    name = "test-threshold"

    def is_speech(self, frame: np.ndarray) -> bool:
        return bool(np.max(np.abs(frame)) > 0.1)


class Response:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, _size: int) -> bytes:
        body, self._body = self._body, b""
        return body


def test_vad_rejects_short_noise_and_captures_pre_roll_and_hangover() -> None:
    config = AssistantConfig(
        vad_preroll_ms=150,
        vad_start_speech_ms=60,
        vad_end_silence_ms=90,
    )
    detector = VoiceActivityDetector(config, ThresholdVAD())
    silence = np.zeros(detector.frame_samples, dtype=np.float32)
    speech = np.full(detector.frame_samples, 0.4, dtype=np.float32)

    for _ in range(4):
        assert detector.process_frame(silence) is None
    assert detector.process_frame(speech) is None
    assert detector.process_frame(speech) is None
    assert detector.process_frame(speech) is None

    assert detector.process_frame(silence) is None
    assert detector.process_frame(silence) is None
    segment = detector.process_frame(silence)

    assert isinstance(segment, SpeechSegment)
    assert segment.duration_seconds == pytest.approx(0.27)
    assert segment.samples.size == detector.frame_samples * 9
    assert np.all(segment.samples[: detector.frame_samples] == 0.0)
    assert np.allclose(segment.samples[detector.frame_samples * 3], 0.4)
    assert segment.peak_rms == pytest.approx(0.4)


def test_vad_hysteresis_discards_short_speech_noise() -> None:
    detector = VoiceActivityDetector(
        AssistantConfig(vad_preroll_ms=60, vad_start_speech_ms=90),
        ThresholdVAD(),
    )
    noise = np.zeros(detector.frame_samples, dtype=np.float32)
    impulse = np.full(detector.frame_samples, 0.5, dtype=np.float32)

    assert detector.process_frame(impulse) is None
    assert detector.process_frame(noise) is None
    for _ in range(10):
        assert detector.process_frame(noise) is None


def test_vad_splits_at_configured_max_utterance_length() -> None:
    detector = VoiceActivityDetector(
        AssistantConfig(
            vad_preroll_ms=60,
            vad_start_speech_ms=30,
            vad_end_silence_ms=700,
            max_utterance_seconds=1,
        ),
        ThresholdVAD(),
    )
    speech = np.full(detector.frame_samples, 0.25, dtype=np.float32)

    segment = None
    for _ in range(detector.max_frames):
        segment = detector.process_frame(speech)

    assert segment is not None
    assert segment.duration_seconds == pytest.approx(1.02)


def test_missing_silero_model_uses_webrtc_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(AssistantConfig, "model_path", property(lambda _self: tmp_path))

    backend = create_vad_backend(AssistantConfig())

    assert backend.name == "webrtc"


def test_audio_capture_callback_copies_frames_and_tracks_rms() -> None:
    capture = AudioCapture(queue_frames=2)
    frame = np.full((capture.frame_samples, 1), 0.25, dtype=np.float32)

    capture._audio_callback(frame, capture.frame_samples, None, None)
    copied = capture._frames.get_nowait()

    assert copied.shape == (capture.frame_samples,)
    assert copied.dtype == np.float32
    assert capture.rms == pytest.approx(0.25)


def test_audio_capture_reports_callback_failures() -> None:
    capture = AudioCapture()
    wrong_frame = np.zeros((capture.frame_samples - 1, 1), dtype=np.float32)
    capture._audio_callback(wrong_frame, capture.frame_samples - 1, None, None)
    capture._stream = type("InactiveStream", (), {"active": False})()

    with pytest.raises(AudioCaptureError, match="Expected"):
        capture.read_frame(timeout=0.01)


def test_audio_capture_detects_a_stopped_device_stream() -> None:
    capture = AudioCapture()
    capture._stream = type("InactiveStream", (), {"active": False})()

    with pytest.raises(AudioCaptureError, match="stopped unexpectedly"):
        capture.read_frame(timeout=0.01)


def test_audio_capture_shutdown_wakes_waiter_without_stream_race() -> None:
    entered_queue = threading.Event()
    release_queue = threading.Event()
    errors = []

    class WaitingQueue:
        def get(self, timeout: float):
            entered_queue.set()
            assert release_queue.wait(timeout=1.0)
            raise queue.Empty

    class ActiveStream:
        active = True

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            pass

    capture = AudioCapture()
    capture._frames = WaitingQueue()
    capture._stream = ActiveStream()

    def reader() -> None:
        try:
            capture.read_frame(timeout=1.0)
        except Exception as exc:
            errors.append(exc)

    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    assert entered_queue.wait(timeout=1.0)
    capture.stop()
    release_queue.set()
    reader_thread.join(timeout=1.0)

    assert not reader_thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], AudioCaptureError)
    assert "stream was closed" in str(errors[0])


def test_audio_capture_reuses_a_single_persistent_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams = []

    class FakeStream:
        active = False

        def __init__(self, **_kwargs) -> None:
            streams.append(self)

        def start(self) -> None:
            self.active = True

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            pass

    monkeypatch.setattr(capture_module.sd, "InputStream", FakeStream)
    capture = AudioCapture()

    capture.start()
    capture.start()

    assert len(streams) == 1
    capture.stop()
    assert not streams[0].active


def test_runtime_retries_after_microphone_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chintu_assistant as entry

    emissions: dict[str, list[object]] = {
        "status": [],
        "response": [],
        "visibility": [],
        "rms": [],
    }

    class Signal:
        def __init__(self, name: str) -> None:
            self.name = name

        def emit(self, value) -> None:
            emissions[self.name].append(value)

    bridge = SimpleNamespace(
        status_changed=Signal("status"),
        response_changed=Signal("response"),
        visibility_changed=Signal("visibility"),
        microphone_rms_changed=Signal("rms"),
    )
    runtime = entry.ChintuRuntime(bridge, AssistantConfig(audio_recovery_seconds=0.001))
    capture_instances = []

    class Detector:
        backend_name = "test"

        def reset(self) -> None:
            pass

        def process_frame(self, _frame) -> None:
            return None

    class RetryingCapture:
        def __init__(self, **_kwargs) -> None:
            self.start_calls = 0
            self.stop_calls = 0
            capture_instances.append(self)

        def start(self) -> None:
            self.start_calls += 1
            if self.start_calls == 1:
                raise AudioCaptureError("simulated unplugged device")

        def read_frame(self, timeout: float):
            assert timeout == 0.5
            runtime._stopping.set()
            return None

        def stop(self) -> None:
            self.stop_calls += 1

    monkeypatch.setattr(entry, "VoiceActivityDetector", lambda _config: Detector())
    monkeypatch.setattr(entry, "AudioCapture", RetryingCapture)

    runtime._audio_loop()

    assert len(capture_instances) == 1
    assert capture_instances[0].start_calls == 2
    assert capture_instances[0].stop_calls == 2
    assert "Microphone reconnecting..." in emissions["status"]


def test_model_download_resumes_partial_and_checks_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "model.onnx"
    partial = destination.with_suffix(".onnx.part")
    partial.write_bytes(b"silero ")
    full_bytes = b"silero model bytes"
    expected_hash = hashlib.sha256(full_bytes).hexdigest()
    response = Response(206, b"model bytes")
    requests: list[object] = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        assert timeout == 60
        return response

    monkeypatch.setattr(download_models.urllib.request, "urlopen", fake_urlopen)

    result = download_models.download_file("https://example.invalid/model", destination, expected_hash)

    assert result == destination
    assert destination.read_bytes() == full_bytes
    assert not partial.exists()
    assert requests[0].headers["Range"] == f"bytes={len(b'silero ')}-"


def test_model_download_removes_checksum_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "model.onnx"
    monkeypatch.setattr(
        download_models.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(200, b"not the expected model"),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_models.download_file("https://example.invalid/model", destination, "0" * 64)

    assert not destination.exists()
    assert not destination.with_suffix(".onnx.part").exists()
