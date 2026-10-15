from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import numpy as np
from numpy.typing import NDArray
from rapidfuzz import fuzz

from ..config import AssistantConfig, VOSK_MODEL_NAME
from .vad import SpeechSegment

logger = logging.getLogger("chintu.audio.wake")

VOSK_GRAMMAR = ("chintu", "hey chintu", "ok chintu", "[unk]")
_PHONETIC_ALIASES = {
    "chintu": ("chintu", "chintoo", "chinto", "chintou"),
}


class WakeWordDetector(Protocol):
    """Interface for detecting a wake phrase in a VAD-approved speech segment."""

    def detect(self, segment: SpeechSegment) -> WakeWordMatch | None: ...


class VoskRecognizer(Protocol):
    def AcceptWaveform(self, data: bytes) -> bool: ...

    def FinalResult(self) -> str: ...


class WhisperSegment(Protocol):
    text: str


class WhisperModel(Protocol):
    def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        language: str,
        beam_size: int,
        condition_on_previous_text: bool,
        vad_filter: bool,
    ) -> tuple[Iterable[WhisperSegment], object]: ...


@dataclass(frozen=True)
class WakeWordMatch:
    phrase: str
    vosk_transcript: str
    whisper_transcript: str
    confirmation_score: float


class VoskWhisperWakeWordDetector:
    """Use restricted-grammar Vosk followed by local Whisper confirmation."""

    def __init__(
        self,
        config: AssistantConfig,
        *,
        vosk_model: object | None = None,
        whisper_model: WhisperModel | None = None,
        recognizer_factory: Callable[[object, int, str], VoskRecognizer] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self._last_confirmed_at = float("-inf")

        if vosk_model is None:
            from vosk import Model

            model_path = config.model_path / VOSK_MODEL_NAME
            if not (model_path / "am" / "final.mdl").is_file():
                raise FileNotFoundError(
                    f"Vosk wake-word model is missing at {model_path}; "
                    "run scripts/download_models.py --wake"
                )
            vosk_model = Model(str(model_path))
        if whisper_model is None:
            from faster_whisper import WhisperModel as FasterWhisperModel

            try:
                whisper_model = FasterWhisperModel(
                    "base",
                    device="cpu",
                    compute_type="int8",
                    local_files_only=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    "The local faster-whisper base model is missing or invalid; "
                    "run scripts/download_models.py --wake"
                ) from exc

        if recognizer_factory is None:
            from vosk import KaldiRecognizer

            recognizer_factory = KaldiRecognizer
        self._vosk_model = vosk_model
        self._whisper_model = whisper_model
        self._recognizer_factory = recognizer_factory

    def detect(self, segment: SpeechSegment) -> WakeWordMatch | None:
        if segment.sample_rate_hz != 16000:
            raise ValueError("Wake-word detection requires 16 kHz audio")
        if segment.samples.size == 0:
            return None

        samples = np.asarray(segment.samples, dtype=np.float32).reshape(-1)
        window_samples = round(
            self.config.wake_confirm_window_seconds * segment.sample_rate_hz
        )
        confirmation_audio = samples[-window_samples:]
        pcm16 = (
            np.clip(confirmation_audio, -1.0, 1.0) * 32767.0
        ).astype("<i2", copy=False)
        recognizer = self._recognizer_factory(
            self._vosk_model,
            segment.sample_rate_hz,
            json.dumps(VOSK_GRAMMAR),
        )
        recognizer.AcceptWaveform(pcm16.tobytes())
        vosk_result = json.loads(recognizer.FinalResult())
        vosk_text = str(vosk_result.get("text", "")).strip()
        candidate_phrase = _match_wake_phrase(
            vosk_text,
            ("chintu", "hey chintu", "ok chintu"),
            threshold=85.0,
        )
        if candidate_phrase is None:
            return None

        if self._clock() - self._last_confirmed_at < self.config.wake_cooldown_seconds:
            logger.debug("Wake-word candidate ignored during cooldown")
            return None

        transcription = self._whisper_model.transcribe(
            confirmation_audio,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        whisper_segments = transcription[0]
        whisper_text = " ".join(item.text.strip() for item in whisper_segments).strip()
        threshold = 90.0 - 10.0 * self.config.wake_sensitivity
        confirmation = _match_wake_phrase(
            whisper_text,
            self.config.wake_words,
            threshold=threshold,
        )
        if confirmation is None:
            logger.debug(
                "Wake candidate rejected by Whisper confirmation: Vosk=%r Whisper=%r",
                vosk_text,
                whisper_text,
            )
            return None

        phrase, score = confirmation
        self._last_confirmed_at = self._clock()
        logger.info(
            "Wake word confirmed: phrase=%s score=%.1f Vosk=%r Whisper=%r",
            phrase,
            score,
            vosk_text,
            whisper_text,
        )
        return WakeWordMatch(
            phrase=phrase,
            vosk_transcript=vosk_text,
            whisper_transcript=whisper_text,
            confirmation_score=score,
        )


def _match_wake_phrase(
    transcript: str,
    phrases: Iterable[str],
    *,
    threshold: float,
) -> tuple[str, float] | None:
    normalized_transcript = _normalize(transcript)
    if not normalized_transcript:
        return None
    transcript_words = normalized_transcript.split()

    best_match: tuple[str, float] | None = None
    for phrase in phrases:
        variants = _expand_phonetic_variants(phrase)
        for variant in variants:
            normalized_variant = _normalize(variant)
            variant_words = normalized_variant.split()
            shortest = max(1, len(variant_words) - 1)
            longest = min(len(transcript_words), len(variant_words) + 1)
            score = max(
                (
                    float(
                        fuzz.ratio(
                            normalized_variant,
                            " ".join(transcript_words[offset : offset + size]),
                        )
                    )
                    for size in range(shortest, longest + 1)
                    for offset in range(len(transcript_words) - size + 1)
                ),
                default=0.0,
            )
            if (
                best_match is None
                or score > best_match[1]
                or (
                    score == best_match[1]
                    and len(normalized_variant) > len(_normalize(best_match[0]))
                )
            ):
                best_match = (phrase, score)
    if best_match is None or best_match[1] < threshold:
        return None
    return best_match


def _expand_phonetic_variants(phrase: str) -> tuple[str, ...]:
    variants = {phrase}
    for canonical, aliases in _PHONETIC_ALIASES.items():
        if canonical in phrase.casefold():
            for alias in aliases:
                variants.add(re.sub(canonical, alias, phrase, flags=re.IGNORECASE))
    return tuple(sorted(variants))


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z]+", text.casefold()))
