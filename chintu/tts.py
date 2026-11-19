from __future__ import annotations

import io
import logging
import os
import queue
import re
import tempfile
import threading
import wave
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import sounddevice as sd

from .config import AssistantConfig

logger = logging.getLogger("chintu.tts")


class TTSNormalizer:
    """Normalize text into clean, conversational speech suitable for TTS synthesis."""

    _ABBREVIATIONS = [
        (re.compile(r"\bapprox\.?", re.IGNORECASE), "approximately"),
        (re.compile(r"\be\.g\.?", re.IGNORECASE), "for example"),
        (re.compile(r"\bi\.e\.?", re.IGNORECASE), "that is"),
        (re.compile(r"\betc\.?", re.IGNORECASE), "etcetera"),
        (re.compile(r"\bvs\.?|\bversus\b", re.IGNORECASE), "versus"),
        (re.compile(r"\bdr\.", re.IGNORECASE), "doctor"),
        (re.compile(r"\bmr\.", re.IGNORECASE), "mister"),
        (re.compile(r"\bmrs\.", re.IGNORECASE), "missus"),
        (re.compile(r"\bms\.", re.IGNORECASE), "miss"),
        (re.compile(r"\bmin\.|\bmins\b", re.IGNORECASE), "minutes"),
        (re.compile(r"\bsec\.|\bsecs\b", re.IGNORECASE), "seconds"),
        (re.compile(r"\bhr\.|\bhrs\b", re.IGNORECASE), "hours"),
        (re.compile(r"\btemp\.", re.IGNORECASE), "temperature"),
        (re.compile(r"\bdeg\.", re.IGNORECASE), "degrees"),
    ]

    _UNITS = [
        (re.compile(r"(\d+(?:\.\d+)?)\s*°?C\b"), r"\1 degrees Celsius"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*°?F\b"), r"\1 degrees Fahrenheit"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*°"), r"\1 degrees"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*%"), r"\1 percent"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*km/h\b", re.IGNORECASE), r"\1 kilometers per hour"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*mph\b", re.IGNORECASE), r"\1 miles per hour"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*ms\b", re.IGNORECASE), r"\1 milliseconds"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*km\b", re.IGNORECASE), r"\1 kilometers"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*kg\b", re.IGNORECASE), r"\1 kilograms"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*cm\b", re.IGNORECASE), r"\1 centimeters"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*mm\b", re.IGNORECASE), r"\1 millimeters"),
    ]

    _CURRENCIES = [
        (re.compile(r"\$(\d+(?:\.\d+)?)"), r"\1 dollars"),
        (re.compile(r"₹(\d+(?:\.\d+)?)"), r"\1 rupees"),
        (re.compile(r"€(\d+(?:\.\d+)?)"), r"\1 euros"),
        (re.compile(r"£(\d+(?:\.\d+)?)"), r"\1 pounds"),
    ]

    _MARKDOWN_CLEANUP = [
        (re.compile(r"```[\s\S]*?```"), " code snippet "),
        (re.compile(r"`([^`]+)`"), r"\1"),
        (re.compile(r"\[([^\]]+)\]\([^\)]+\)"), r"\1"),
        (re.compile(r"https?://\S+"), ""),
        (re.compile(r"\*{1,3}([^*]+)\*{1,3}"), r"\1"),
        (re.compile(r"_{1,3}([^_]+)_{1,3}"), r"\1"),
        (re.compile(r"#{1,6}\s*"), ""),
        (re.compile(r"^[>\-\*\+]\s+", re.MULTILINE), ""),
    ]

    @classmethod
    def normalize(cls, text: str) -> str:
        if not text:
            return ""

        cleaned = text.strip()

        for pattern, replacement in cls._MARKDOWN_CLEANUP:
            cleaned = pattern.sub(replacement, cleaned)

        for pattern, replacement in cls._CURRENCIES:
            cleaned = pattern.sub(replacement, cleaned)

        for pattern, replacement in cls._UNITS:
            cleaned = pattern.sub(replacement, cleaned)

        for pattern, replacement in cls._ABBREVIATIONS:
            cleaned = pattern.sub(replacement, cleaned)

        # Expand decimal point in numbers: e.g. 3.14 -> 3 point 14
        cleaned = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", cleaned)

        # Remove non-speech characters like brackets, pipes, underscores, quotes
        cleaned = re.sub(r"[\{\}\[\]\(\)<>\|_~`@#\^&\*\+=/\\]", " ", cleaned)

        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned


class TTSEngine(Protocol):
    """Protocol for offline speech synthesizers."""

    @property
    def name(self) -> str:
        ...

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text into a 1D float32 numpy audio array in [-1.0, 1.0] and sample rate."""
        ...


class Pyttsx3TTSEngine:
    """Offline speech synthesis using the system SAPI5 voice engine via pyttsx3."""

    def __init__(self, voice_id: str = "", rate: int = 175, volume: float = 1.0) -> None:
        self._voice_id = voice_id
        self._rate = rate
        self._volume = volume
        self._lock = threading.Lock()
        self._name = "pyttsx3"

    @property
    def name(self) -> str:
        return self._name

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        import pyttsx3

        with self._lock:
            engine = pyttsx3.init()
            try:
                if self._voice_id:
                    engine.setProperty("voice", self._voice_id)
                engine.setProperty("rate", self._rate)
                engine.setProperty("volume", self._volume)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tmp_wav = tf.name

                try:
                    engine.save_to_file(text, tmp_wav)
                    engine.runAndWait()

                    with wave.open(tmp_wav, "rb") as wf:
                        channels = wf.getnchannels()
                        sampwidth = wf.getsampwidth()
                        framerate = wf.getframerate()
                        nframes = wf.getnframes()
                        raw_bytes = wf.readframes(nframes)

                    if sampwidth == 2:
                        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sampwidth == 1:
                        audio = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                    elif sampwidth == 4:
                        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        raise ValueError(f"Unsupported sample width: {sampwidth}")

                    if channels > 1:
                        audio = audio.reshape(-1, channels).mean(axis=1)

                    return audio, framerate
                finally:
                    if os.path.exists(tmp_wav):
                        try:
                            os.unlink(tmp_wav)
                        except OSError:
                            pass
            finally:
                try:
                    engine.stop()
                except Exception:
                    pass


class PiperTTSEngine:
    """Offline neural speech synthesis using Piper."""

    def __init__(self, model_path: Path | str, config_path: Path | str | None = None) -> None:
        self._model_path = Path(model_path)
        self._config_path = (
            Path(config_path) if config_path else self._model_path.with_suffix(self._model_path.suffix + ".json")
        )
        if not self._model_path.is_file():
            raise FileNotFoundError(f"Piper model not found: {self._model_path}")
        if not self._config_path.is_file():
            # Piper models can also use .onnx.json
            alt_config = self._model_path.parent / f"{self._model_path.name}.json"
            if alt_config.is_file():
                self._config_path = alt_config
            else:
                raise FileNotFoundError(f"Piper model config not found: {self._config_path}")

        from piper import PiperVoice

        self._voice = PiperVoice.load(str(self._model_path), config_path=str(self._config_path))
        self._name = f"piper ({self._model_path.stem})"

    @property
    def name(self) -> str:
        return self._name

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            self._voice.synthesize_wav(text, wf)

        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw_bytes = wf.readframes(nframes)

        if sampwidth == 2:
            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio = np.frombuffer(raw_bytes, dtype=np.float32)

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        return audio, framerate


def create_tts_engine(config: AssistantConfig) -> TTSEngine:
    """Instantiate the configured TTS engine with fallback support."""
    engine_type = config.tts_engine.lower()

    if engine_type in {"auto", "piper"}:
        model_file = config.model_path / config.tts_piper_model
        if not model_file.is_file():
            # Check piper subfolder
            model_file = config.model_path / "piper" / config.tts_piper_model

        if model_file.is_file():
            try:
                engine = PiperTTSEngine(model_file)
                logger.info("Using Piper neural TTS engine: %s", engine.name)
                return engine
            except Exception as exc:
                if engine_type == "piper":
                    raise RuntimeError(f"Failed to load Piper model: {exc}") from exc
                logger.warning("Piper TTS load failed, falling back to pyttsx3: %s", exc)
        elif engine_type == "piper":
            raise FileNotFoundError(f"Configured Piper model not found: {model_file}")

    logger.info("Using pyttsx3 system TTS engine")
    return Pyttsx3TTSEngine(
        voice_id=config.tts_voice_id,
        rate=config.tts_rate,
        volume=config.tts_volume,
    )


class TextToSpeech:
    """Thread-safe, non-blocking text-to-speech coordinator with barge-in support."""

    def __init__(
        self,
        config: AssistantConfig,
        engine: TTSEngine | None = None,
        on_speaking_start: Callable[[], None] | None = None,
        on_speaking_end: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self._engine = engine or create_tts_engine(config)
        self._on_speaking_start = on_speaking_start
        self._on_speaking_end = on_speaking_end

        self._queue: queue.Queue[tuple[str, int] | None] = queue.Queue()
        self._stopping = threading.Event()
        self._is_speaking = False
        self._lock = threading.Lock()
        self._generation = 0

        self._worker_thread = threading.Thread(
            target=self._playback_loop,
            name="chintu-tts-worker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("TTS initialized with engine: %s", self._engine.name)

    @property
    def engine_name(self) -> str:
        return self._engine.name

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking

    def speak(self, text: str, *, interrupt_current: bool = True) -> None:
        """Queue text for speech output. If interrupt_current is True, aborts current speech."""
        normalized = TTSNormalizer.normalize(text)
        if not normalized:
            return

        with self._lock:
            if interrupt_current:
                self._generation += 1
                self._discard_queued()
            generation = self._generation
            self._queue.put((normalized, generation))

    def interrupt(self) -> None:
        """Invalidate active speech and clear queued responses."""
        with self._lock:
            self._generation += 1
            self._discard_queued()

    def _discard_queued(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def stop(self, timeout: float = 2.0) -> None:
        """Stop worker and release audio output resources."""
        self._stopping.set()
        self.interrupt()
        self._queue.put(None)
        if self._worker_thread.is_alive() and threading.current_thread() is not self._worker_thread:
            self._worker_thread.join(timeout=timeout)

    def _playback_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None or self._stopping.is_set():
                break

            text, generation = item
            with self._lock:
                if generation != self._generation or self._stopping.is_set():
                    self._queue.task_done()
                    continue
                self._is_speaking = True

            try:
                if self._on_speaking_start is not None:
                    try:
                        self._on_speaking_start()
                    except Exception:
                        logger.exception("Error in on_speaking_start callback")

                logger.info(
                    "Synthesizing speech (%d chars) with %s",
                    len(text),
                    self._engine.name,
                )
                audio_samples, sample_rate = self._engine.synthesize(text)

                if not self._is_cancelled(generation) and len(audio_samples) > 0:
                    self._play_samples(audio_samples, sample_rate, generation)

            except Exception:
                logger.exception("TTS synthesis or playback failed")
            finally:
                with self._lock:
                    self._is_speaking = False

                if self._on_speaking_end is not None:
                    try:
                        self._on_speaking_end()
                    except Exception:
                        logger.exception("Error in on_speaking_end callback")

                self._queue.task_done()

    def _is_cancelled(self, generation: int) -> bool:
        with self._lock:
            return self._stopping.is_set() or generation != self._generation

    def _play_samples(
        self,
        samples: np.ndarray,
        sample_rate: int,
        generation: int,
    ) -> None:
        """Stream samples in chunks to allow fast barge-in cancellation."""
        chunk_size = int(sample_rate * 0.05)  # 50 ms chunks for responsive interruption
        total_len = len(samples)

        try:
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            ) as stream:
                pos = 0
                while pos < total_len and not self._is_cancelled(generation):
                    end = min(pos + chunk_size, total_len)
                    chunk = samples[pos:end]
                    stream.write(np.ascontiguousarray(chunk, dtype=np.float32))
                    pos = end
        except Exception as exc:
            logger.warning("Audio playback output stream error: %s", exc)
