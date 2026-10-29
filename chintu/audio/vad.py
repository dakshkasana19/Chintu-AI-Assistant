from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..config import AssistantConfig

logger = logging.getLogger("chintu.audio.vad")


FloatSamples = NDArray[np.float32]


class VADBackend(Protocol):
    """Backend contract returning whether a 16 kHz audio frame contains speech."""

    name: str

    def is_speech(self, frame: FloatSamples) -> bool: ...

    def reset(self) -> None: ...


class SileroOnnxVAD:
    """CPU-only Silero ONNX VAD adapter for 16 kHz mono frames."""

    name = "silero-onnx"
    _WINDOW_SAMPLES = 512
    _CONTEXT_SAMPLES = 64
    _STATE_SHAPE = (2, 1, 128)

    def __init__(self, model_path: str | Path, threshold: float = 0.5) -> None:
        import onnxruntime as ort

        self.threshold = threshold
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        input_names = {item.name for item in self._session.get_inputs()}
        expected = {"input", "state", "sr"}
        if not expected.issubset(input_names):
            raise ValueError(
                f"Unsupported Silero ONNX model inputs: {sorted(input_names)}"
            )
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros(self._STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(self._CONTEXT_SAMPLES, dtype=np.float32)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_speech = False

    def is_speech(self, frame: FloatSamples) -> bool:
        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        self._buffer = np.concatenate([self._buffer, samples])

        ran = False
        speech_detected = False
        while self._buffer.size >= self._WINDOW_SAMPLES:
            chunk = self._buffer[:self._WINDOW_SAMPLES]
            self._buffer = self._buffer[self._WINDOW_SAMPLES:]

            x = np.concatenate([self._context, chunk]).reshape(1, -1)
            scores, self._state = self._session.run(
                None,
                {
                    "input": x,
                    "state": self._state,
                    "sr": np.asarray(16000, dtype=np.int64),
                },
            )
            self._context = chunk[-self._CONTEXT_SAMPLES:]
            score = float(scores[0, 0])
            ran = True
            if score >= self.threshold:
                speech_detected = True

        if ran:
            self._last_speech = speech_detected
        return self._last_speech


class WebRTCVAD:
    """CPU-only WebRTC VAD fallback."""

    name = "webrtc"

    def __init__(self, aggressiveness: int = 2) -> None:
        import webrtcvad

        self._vad = webrtcvad.Vad(aggressiveness)

    def reset(self) -> None:
        pass

    def is_speech(self, frame: FloatSamples) -> bool:
        samples = np.clip(np.asarray(frame, dtype=np.float32), -1.0, 1.0)
        pcm16 = (samples * 32767.0).astype("<i2", copy=False)
        return self._vad.is_speech(pcm16.tobytes(), 16000)


def create_vad_backend(config: AssistantConfig) -> VADBackend:
    """Prefer the local Silero model and fall back to WebRTC VAD when unavailable."""

    model_path = config.model_path / "silero_vad.onnx"
    if model_path.is_file():
        try:
            backend = SileroOnnxVAD(model_path, config.silero_threshold)
            logger.info("Loaded Silero VAD from %s", model_path)
            return backend
        except Exception as exc:
            logger.warning("Could not load Silero VAD model %s: %s", model_path, exc)
    else:
        logger.warning("Silero VAD model missing at %s; trying WebRTC VAD fallback", model_path)

    try:
        backend = WebRTCVAD(config.webrtc_aggressiveness)
        logger.info("Using WebRTC VAD fallback")
        return backend
    except Exception as exc:
        raise RuntimeError(
            "No VAD backend is available. Download models/silero_vad.onnx "
            "or install webrtcvad-wheels."
        ) from exc


@dataclass(frozen=True)
class SpeechSegment:
    """A single VAD-delimited utterance including pre-roll and end hangover."""

    samples: FloatSamples
    sample_rate_hz: int
    duration_seconds: float
    peak_rms: float


class VoiceActivityDetector:
    """Apply speech-start hysteresis, pre-roll, hangover, and maximum duration."""

    def __init__(
        self,
        config: AssistantConfig,
        backend: VADBackend | None = None,
    ) -> None:
        self.sample_rate_hz = config.sample_rate_hz
        self.frame_size_ms = config.frame_size_ms
        self.frame_samples = config.sample_rate_hz * config.frame_size_ms // 1000
        self.preroll_frames = max(1, math.ceil(config.vad_preroll_ms / config.frame_size_ms))
        self.start_frames = max(
            1,
            math.ceil(config.vad_start_speech_ms / config.frame_size_ms),
        )
        self.hangover_frames = max(
            1,
            math.ceil(config.vad_end_silence_ms / config.frame_size_ms),
        )
        self.max_frames = max(
            1,
            math.ceil(config.max_utterance_seconds * 1000 / config.frame_size_ms),
        )
        self.backend = backend or create_vad_backend(config)
        self._pre_roll: deque[FloatSamples] = deque(maxlen=self.preroll_frames)
        self._candidate_frames = 0
        self._active_frames: list[FloatSamples] = []
        self._silence_frames = 0
        self._peak_rms = 0.0

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def reset(self) -> None:
        """Clear rolling speech state after a stream interruption or shutdown."""

        self._pre_roll.clear()
        self._candidate_frames = 0
        self._active_frames.clear()
        self._silence_frames = 0
        self._peak_rms = 0.0
        if hasattr(self.backend, "reset"):
            self.backend.reset()

    def process_frame(self, frame: FloatSamples) -> SpeechSegment | None:
        """Consume one frame and return a completed speech segment, if any."""

        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        if samples.size != self.frame_samples:
            raise ValueError(
                f"Expected {self.frame_samples} samples, received {samples.size}"
            )

        is_speech = self.backend.is_speech(samples)
        frame_rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))

        if not self._active_frames:
            self._pre_roll.append(samples.copy())
            if is_speech:
                self._candidate_frames += 1
            else:
                self._candidate_frames = 0
            if self._candidate_frames >= self.start_frames:
                self._active_frames = list(self._pre_roll)
                self._peak_rms = max(
                    (
                        float(np.sqrt(np.mean(np.square(item), dtype=np.float64)))
                        for item in self._active_frames
                    ),
                    default=frame_rms,
                )
                self._silence_frames = 0
            return None

        self._active_frames.append(samples.copy())
        self._peak_rms = max(self._peak_rms, frame_rms)
        if is_speech:
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        if self._silence_frames >= self.hangover_frames or len(self._active_frames) >= self.max_frames:
            segment_frames = self._active_frames
            segment = SpeechSegment(
                samples=np.concatenate(segment_frames).astype(np.float32, copy=False),
                sample_rate_hz=self.sample_rate_hz,
                duration_seconds=len(segment_frames) * self.frame_size_ms / 1000.0,
                peak_rms=self._peak_rms,
            )
            self._pre_roll.clear()
            self._candidate_frames = 0
            self._active_frames = []
            self._silence_frames = 0
            self._peak_rms = 0.0
            return segment
        return None
