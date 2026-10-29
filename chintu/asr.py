from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
from numpy.typing import NDArray

from .audio.vad import SpeechSegment
from .config import AssistantConfig

logger = logging.getLogger("chintu.asr")

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_FILLER_PREFIXES = (
    re.compile(r"^(?:can|could|would) you\b[\s,;:.-]*"),
    re.compile(r"^please\b[\s,;:.-]*"),
)


class WhisperSegment(Protocol):
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


class WhisperModel(Protocol):
    def transcribe(
        self,
        audio: NDArray[np.float32],
        **kwargs: object,
    ) -> tuple[Iterable[WhisperSegment], object]: ...


@dataclass(frozen=True)
class TranscriptionResult:
    """Raw and normalized command text plus local recognition diagnostics."""

    raw_text: str
    normalized_text: str
    accepted: bool
    reason: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


class FasterWhisperASR:
    """Transcribe VAD-delimited commands with one local CPU/int8 Whisper model."""

    def __init__(
        self,
        config: AssistantConfig,
        *,
        model: WhisperModel | None = None,
    ) -> None:
        self.config = config
        if model is None:
            from faster_whisper import WhisperModel as FasterWhisperModel

            try:
                model = FasterWhisperModel(
                    config.asr_model_size,
                    device="cpu",
                    compute_type="int8",
                    local_files_only=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"The local faster-whisper {config.asr_model_size} model is "
                    "missing or invalid; run scripts/download_models.py --wake "
                    f"--whisper-model {config.asr_model_size}"
                ) from exc
        self._model = model

    @property
    def whisper_model(self) -> WhisperModel:
        """Return the shared model instance for wake confirmation."""

        return self._model

    def transcribe(self, segment: SpeechSegment) -> TranscriptionResult:
        """Transcribe one VAD-delimited segment and reject likely hallucinations."""

        if segment.sample_rate_hz != 16000:
            raise ValueError("Command ASR requires 16 kHz audio")
        samples = np.asarray(segment.samples, dtype=np.float32).reshape(-1)
        if samples.size == 0 or not np.isfinite(samples).all():
            return TranscriptionResult("", "", False, "empty_or_invalid_audio")
        if float(np.max(np.abs(samples))) < 1e-5:
            return TranscriptionResult("", "", False, "silent_audio")

        whisper_segments, _ = self._model.transcribe(
            samples,
            language="en",
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=False,
            initial_prompt="Chintu",
            hotwords="Chintu",
            no_speech_threshold=self.config.asr_no_speech_threshold,
            log_prob_threshold=self.config.asr_log_prob_threshold,
            compression_ratio_threshold=self.config.asr_compression_ratio_threshold,
        )
        recognized = list(whisper_segments)
        raw_text = " ".join(item.text.strip() for item in recognized).strip()
        if not raw_text:
            return TranscriptionResult("", "", False, "no_speech")

        average_logprob = float(np.mean([item.avg_logprob for item in recognized]))
        no_speech_prob = float(np.mean([item.no_speech_prob for item in recognized]))
        compression_ratio = max(item.compression_ratio for item in recognized)
        diagnostics = (average_logprob, no_speech_prob, compression_ratio)
        if not all(math.isfinite(value) for value in diagnostics):
            return TranscriptionResult(
                raw_text,
                "",
                False,
                "invalid_confidence_metadata",
                average_logprob,
                no_speech_prob,
                compression_ratio,
            )

        if no_speech_prob >= self.config.asr_no_speech_threshold:
            return TranscriptionResult(
                raw_text,
                "",
                False,
                "no_speech_probability",
                average_logprob,
                no_speech_prob,
                compression_ratio,
            )
        if average_logprob < self.config.asr_log_prob_threshold:
            return TranscriptionResult(
                raw_text,
                "",
                False,
                "low_log_probability",
                average_logprob,
                no_speech_prob,
                compression_ratio,
            )
        if compression_ratio > self.config.asr_compression_ratio_threshold:
            return TranscriptionResult(
                raw_text,
                "",
                False,
                "high_compression_ratio",
                average_logprob,
                no_speech_prob,
                compression_ratio,
            )

        normalized_text = normalize_command_text(raw_text)
        if not normalized_text:
            return TranscriptionResult(
                raw_text,
                "",
                False,
                "no_command_after_normalization",
                average_logprob,
                no_speech_prob,
                compression_ratio,
            )
        return TranscriptionResult(
            raw_text,
            normalized_text,
            True,
            avg_logprob=average_logprob,
            no_speech_prob=no_speech_prob,
            compression_ratio=compression_ratio,
        )


def normalize_command_text(text: str) -> str:
    """Normalize command text while leaving the original transcript untouched."""

    normalized = text.casefold().strip()
    normalized = re.sub(r"^[\W_]+|[\W_]+$", "", normalized)
    while True:
        previous = normalized
        normalized = re.sub(r"^chintu\b[\s,;:.-]*", "", normalized)
        for prefix in _FILLER_PREFIXES:
            updated = prefix.sub("", normalized, count=1)
            if updated != normalized:
                normalized = updated
                break
        if normalized == previous:
            break
    normalized = re.sub(r"\bplease\b", " ", normalized)
    normalized = re.sub(r"^[\W_]+|[\W_]+$", "", normalized)
    normalized = _normalize_number_words(normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _normalize_number_words(text: str) -> str:
    """Convert cardinal number-word runs through 99 to decimal digits."""

    words = text.split()
    normalized: list[str] = []
    index = 0
    while index < len(words):
        first = _NUMBER_WORDS.get(words[index])
        if first is None:
            normalized.append(words[index])
            index += 1
            continue

        value = first
        index += 1
        if value >= 20 and index < len(words):
            ones = _NUMBER_WORDS.get(words[index])
            if ones is not None and 0 < ones < 10:
                value += ones
                index += 1
        normalized.append(str(value))
    return " ".join(normalized)
