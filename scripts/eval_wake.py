from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chintu.audio import SpeechSegment, VoskWhisperWakeWordDetector, VoiceActivityDetector
from chintu.config import AssistantConfig, load_config

logger = logging.getLogger("chintu.eval.wake")


def read_wav(path: Path) -> tuple[NDArray[np.float32], int]:
    """Read PCM16 WAV audio and return mono float32 samples and sample rate."""

    try:
        with wave.open(str(path), "rb") as audio_file:
            if audio_file.getsampwidth() != 2:
                raise ValueError(f"{path} must be 16-bit PCM WAV")
            channels = audio_file.getnchannels()
            sample_rate_hz = audio_file.getframerate()
            frames = audio_file.readframes(audio_file.getnframes())
    except (wave.Error, OSError) as exc:
        raise ValueError(f"Could not read WAV file {path}: {exc}") from exc
    if sample_rate_hz != 16000:
        raise ValueError(f"{path} must be sampled at 16000 Hz; got {sample_rate_hz}")
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    if pcm.size == 0:
        raise ValueError(f"{path} contains no audio frames")
    return pcm, sample_rate_hz


def vad_segments(
    samples: NDArray[np.float32],
    sample_rate_hz: int,
    config: AssistantConfig,
    vad: VoiceActivityDetector,
) -> list[SpeechSegment]:
    """Pass only VAD-delimited speech segments to the wake-word detector."""

    if sample_rate_hz != config.sample_rate_hz:
        raise ValueError(
            f"VAD expects {config.sample_rate_hz} Hz audio; got {sample_rate_hz}"
        )
    frame_samples = config.sample_rate_hz * config.frame_size_ms // 1000
    segments: list[SpeechSegment] = []
    for offset in range(0, samples.size, frame_samples):
        frame = samples[offset : offset + frame_samples]
        if frame.size < frame_samples:
            frame = np.pad(frame, (0, frame_samples - frame.size))
        segment = vad.process_frame(frame)
        if segment is not None:
            segments.append(segment)

    silence = np.zeros(frame_samples, dtype=np.float32)
    flush_frames = math.ceil(config.vad_end_silence_ms / config.frame_size_ms) + 1
    for _ in range(flush_frames):
        segment = vad.process_frame(silence)
        if segment is not None:
            segments.append(segment)
    return segments


def summarize_counts(
    positive_count: int,
    missed: int,
    negative_seconds: float,
    false_accepts: int,
) -> dict[str, float | int]:
    if positive_count <= 0:
        raise ValueError("Wake-word evaluation requires positive samples")
    if negative_seconds <= 0:
        raise ValueError("Wake-word evaluation requires negative audio duration")
    if not 0 <= missed <= positive_count:
        raise ValueError("Missed clip count must be between zero and positive count")
    if false_accepts < 0:
        raise ValueError("False accept count must not be negative")
    negative_hours = negative_seconds / 3600.0
    return {
        "positive_clips": positive_count,
        "missed_clips": missed,
        "miss_rate": missed / positive_count,
        "negative_hours": negative_hours,
        "false_accepts": false_accepts,
        "false_accepts_per_hour": false_accepts / negative_hours,
    }


def evaluate_dataset(
    positive_files: list[Path],
    negative_files: list[Path],
    config: AssistantConfig,
) -> dict[str, float | int]:
    """Measure positive miss rate and negative false accepts per audio hour."""

    if not positive_files:
        raise ValueError("Positive dataset contains no WAV files")
    if not negative_files:
        raise ValueError("Negative dataset contains no WAV files")

    detector = VoskWhisperWakeWordDetector(config)
    vad = VoiceActivityDetector(config)
    missed = 0
    false_accepts = 0
    negative_seconds = 0.0

    for path in positive_files:
        samples, sample_rate_hz = read_wav(path)
        vad.reset()
        segments = vad_segments(samples, sample_rate_hz, config, vad)
        matches = sum(detector.detect(segment) is not None for segment in segments)
        if matches == 0:
            missed += 1

    for path in negative_files:
        samples, sample_rate_hz = read_wav(path)
        negative_seconds += samples.size / sample_rate_hz
        vad.reset()
        segments = vad_segments(samples, sample_rate_hz, config, vad)
        false_accepts += sum(
            detector.detect(segment) is not None for segment in segments
        )

    return summarize_counts(
        positive_count=len(positive_files),
        missed=missed,
        negative_seconds=negative_seconds,
        false_accepts=false_accepts,
    )


def _wav_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Dataset directory does not exist: {directory}")
    return sorted(directory.rglob("*.wav"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate wake-word misses and false accepts using labeled 16 kHz PCM16 WAVs."
        )
    )
    parser.add_argument("--positive-dir", required=True, type=Path)
    parser.add_argument("--negative-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="Optional JSON output report path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config()
    config.wake_cooldown_seconds = 0.0
    results = evaluate_dataset(
        _wav_files(args.positive_dir),
        _wav_files(args.negative_dir),
        config,
    )
    print(f"Positive clips: {results['positive_clips']}")
    print(
        f"Misses: {results['missed_clips']} "
        f"({results['miss_rate']:.2%} miss rate)"
    )
    print(f"Negative audio: {results['negative_hours']:.4f} hours")
    print(
        f"False accepts: {results['false_accepts']} "
        f"({results['false_accepts_per_hour']:.2f} per hour)"
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
