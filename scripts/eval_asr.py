from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chintu.asr import FasterWhisperASR, normalize_command_text
from chintu.audio.vad import SpeechSegment
from chintu.config import load_config


def _word_error_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> dict[str, int]:
    """Return Levenshtein word errors with deterministic operation counts."""

    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    distance = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        distance[row][0] = row
    for column in range(columns):
        distance[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            substitution_cost = reference[row - 1] != hypothesis[column - 1]
            distance[row][column] = min(
                distance[row - 1][column] + 1,
                distance[row][column - 1] + 1,
                distance[row - 1][column - 1] + substitution_cost,
            )

    substitutions = deletions = insertions = 0
    row, column = len(reference), len(hypothesis)
    while row or column:
        if (
            row
            and column
            and distance[row][column]
            == distance[row - 1][column - 1]
            + (reference[row - 1] != hypothesis[column - 1])
        ):
            if reference[row - 1] != hypothesis[column - 1]:
                substitutions += 1
            row -= 1
            column -= 1
        elif row and distance[row][column] == distance[row - 1][column] + 1:
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": substitutions + deletions + insertions,
        "reference_words": len(reference),
    }


def _normalized_raw_words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.casefold()).split()


def _read_command_wav(path: Path) -> SpeechSegment:
    """Read one mono 16 kHz PCM16 WAV file as a VAD-delimited command."""

    try:
        with wave.open(str(path), "rb") as audio_file:
            if audio_file.getnchannels() != 1:
                raise ValueError(f"{path} must be mono")
            if audio_file.getframerate() != 16000:
                raise ValueError(f"{path} must be sampled at 16000 Hz")
            if audio_file.getsampwidth() != 2:
                raise ValueError(f"{path} must use 16-bit PCM samples")
            frames = audio_file.readframes(audio_file.getnframes())
            sample_rate_hz = audio_file.getframerate()
    except (wave.Error, OSError) as exc:
        raise ValueError(f"Could not read WAV file {path}: {exc}") from exc
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        raise ValueError(f"{path} contains no audio samples")
    peak_rms = float(np.sqrt(np.mean(np.square(samples))))
    return SpeechSegment(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        duration_seconds=samples.size / sample_rate_hz,
        peak_rms=peak_rms,
    )


def evaluate_directory(audio_dir: Path, asr: FasterWhisperASR) -> dict[str, object]:
    """Evaluate paired one-utterance WAV/reference files with local ASR."""

    if not audio_dir.is_dir():
        raise ValueError(f"Audio directory does not exist: {audio_dir}")
    wav_files = sorted(audio_dir.rglob("*.wav"))
    if not wav_files:
        raise ValueError(f"No WAV files found in {audio_dir}")
    missing_references = [
        path.with_suffix(".txt")
        for path in wav_files
        if not path.with_suffix(".txt").is_file()
    ]
    if missing_references:
        raise ValueError(
            "Missing reference text files: "
            + ", ".join(str(path) for path in missing_references)
        )

    utterances: list[dict[str, object]] = []
    raw_totals = {
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "errors": 0,
        "reference_words": 0,
    }
    command_totals = dict(raw_totals)
    for wav_path in wav_files:
        reference_text = wav_path.with_suffix(".txt").read_text(encoding="utf-8").strip()
        if not reference_text:
            raise ValueError(f"Reference text is empty: {wav_path.with_suffix('.txt')}")
        segment = _read_command_wav(wav_path)
        result = asr.transcribe(segment)
        raw_reference_words = _normalized_raw_words(reference_text)
        raw_hypothesis_words = _normalized_raw_words(result.raw_text)
        command_reference_words = normalize_command_text(reference_text).split()
        command_hypothesis_words = result.normalized_text.split()
        if not command_reference_words:
            raise ValueError(
                f"Reference contains no command words after normalization: "
                f"{wav_path.with_suffix('.txt')}"
            )
        raw_counts = _word_error_counts(raw_reference_words, raw_hypothesis_words)
        command_counts = _word_error_counts(command_reference_words, command_hypothesis_words)
        for key in raw_totals:
            raw_totals[key] += raw_counts[key]
            command_totals[key] += command_counts[key]
        utterances.append(
            {
                "audio": str(wav_path),
                "reference": reference_text,
                "raw_transcript": result.raw_text,
                "normalized_command": result.normalized_text,
                "accepted": result.accepted,
                "rejection_reason": result.reason,
                "raw_word_errors": raw_counts,
                "normalized_command_word_errors": command_counts,
            }
        )

    def summary(counts: dict[str, int]) -> dict[str, int | float]:
        word_count = counts["reference_words"]
        return {
            **counts,
            "wer": counts["errors"] / word_count if word_count else 0.0,
        }

    accepted_count = sum(bool(utterance["accepted"]) for utterance in utterances)
    return {
        "model": asr.config.asr_model_size,
        "device": "cpu",
        "compute_type": "int8",
        "utterance_count": len(utterances),
        "accepted_count": accepted_count,
        "rejected_count": len(utterances) - accepted_count,
        "raw_wer": summary(raw_totals),
        "normalized_command_wer": summary(command_totals),
        "utterances": utterances,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure local command ASR WER on paired 16 kHz mono PCM16 WAV/text files."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing one-utterance .wav files paired with same-name .txt references",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path; without it, the full report is printed",
    )
    args = parser.parse_args()

    config = load_config()
    asr = FasterWhisperASR(config)
    report = evaluate_directory(args.audio_dir, asr)
    report_text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text + "\n", encoding="utf-8")
        print(f"ASR report written to {args.report}")
        print(
            json.dumps(
                {
                    "utterance_count": report["utterance_count"],
                    "accepted_count": report["accepted_count"],
                    "rejected_count": report["rejected_count"],
                    "raw_wer": report["raw_wer"],
                    "normalized_command_wer": report["normalized_command_wer"],
                },
                indent=2,
            )
        )
    else:
        print(report_text)


if __name__ == "__main__":
    main()
