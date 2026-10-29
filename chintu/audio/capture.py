from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatFrame = NDArray[np.float32]

logger = logging.getLogger("chintu.audio.capture")


class AudioCaptureError(RuntimeError):
    """Raised when the input stream fails and should be restarted."""


class AudioCapture:
    """Persistent mono input stream with bounded delivery of fixed-size frames."""

    def __init__(
        self,
        sample_rate_hz: int = 16000,
        frame_size_ms: int = 30,
        queue_frames: int = 64,
        device: int | str | None = None,
    ) -> None:
        if sample_rate_hz != 16000:
            raise ValueError("AudioCapture currently requires a 16 kHz sample rate")
        if frame_size_ms not in {10, 20, 30}:
            raise ValueError("frame_size_ms must be 10, 20, or 30")
        if queue_frames < 2:
            raise ValueError("queue_frames must be at least 2")

        self.sample_rate_hz = sample_rate_hz
        self.frame_size_ms = frame_size_ms
        self.frame_samples = sample_rate_hz * frame_size_ms // 1000
        self.device = device
        self._frames: queue.Queue[FloatFrame] = queue.Queue(maxsize=queue_frames)
        self._stream: sd.InputStream | None = None
        self._error: Exception | None = None
        self._error_lock = threading.Lock()
        self._rms = 0.0
        self._rms_lock = threading.Lock()

    @property
    def rms(self) -> float:
        """Return the most recently measured normalized microphone RMS."""

        with self._rms_lock:
            return self._rms

    @property
    def is_active(self) -> bool:
        """Return whether the PortAudio stream is currently active."""

        return self._stream is not None and self._stream.active

    def start(self) -> None:
        """Open and start the one persistent PortAudio input stream."""

        if self._stream is not None:
            if self._stream.active:
                return
            self.stop()

        with self._error_lock:
            self._error = None
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break

        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate_hz,
                blocksize=self.frame_samples,
                channels=1,
                dtype="float32",
                device=self.device,
                callback=self._audio_callback,
            )
            stream.start()
            self._stream = stream
        except Exception as exc:
            self._stream = None
            raise AudioCaptureError(f"Could not start microphone input stream: {exc}") from exc
        logger.info(
            "Microphone stream started: %d Hz mono, %d ms frames",
            self.sample_rate_hz,
            self.frame_size_ms,
        )

    def stop(self) -> None:
        """Stop and release the input stream if it was opened."""

        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            if stream.active:
                stream.stop()
            stream.close()
        except Exception as exc:
            logger.warning("Error while releasing microphone stream: %s", exc)

    def read_frame(self, timeout: float = 1.0) -> FloatFrame | None:
        """Read the next copied float32 frame; return None on an ordinary timeout."""

        with self._error_lock:
            callback_error = self._error
        if callback_error is not None:
            raise AudioCaptureError(f"Microphone callback failed: {callback_error}") from callback_error
        if self._stream is None:
            raise AudioCaptureError("Microphone stream has not been started")

        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            with self._error_lock:
                callback_error = self._error
            if callback_error is not None:
                raise AudioCaptureError(
                    f"Microphone callback failed: {callback_error}"
                ) from callback_error
            stream = self._stream
            if stream is None:
                raise AudioCaptureError("Microphone input stream was closed")
            if not stream.active:
                raise AudioCaptureError("Microphone input stream stopped unexpectedly")
            return None

    def _audio_callback(self, indata, frames: int, time_info, status) -> None:
        del time_info
        try:
            if status:
                logger.warning("Microphone stream status: %s", status)
            if frames != self.frame_samples:
                raise AudioCaptureError(
                    f"Expected {self.frame_samples} samples per frame, received {frames}"
                )
            frame = np.asarray(indata[:, 0], dtype=np.float32).copy()
            frame_rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
            with self._rms_lock:
                self._rms = frame_rms

            try:
                self._frames.put_nowait(frame)
            except queue.Full:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass
                self._frames.put_nowait(frame)
                logger.warning("Audio frame queue full; discarded the oldest frame")
        except Exception as exc:
            logger.exception("Microphone callback failed")
            with self._error_lock:
                self._error = exc
