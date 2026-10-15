from __future__ import annotations

import platform
import statistics
import sys
from time import perf_counter
from importlib import import_module
from pathlib import Path


def check_module(name: str) -> tuple[bool, str | None]:
    try:
        import_module(name)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    required = [
        "numpy",
        "PyQt6",
        "tomli",
        "sounddevice",
        "webrtcvad",
        "onnxruntime",
        "vosk",
        "faster_whisper",
        "piper",
        "pyttsx3",
        "ddgs",
        "requests",
        "trafilatura",
        "feedparser",
        "rapidfuzz",
        "psutil",
        "pycaw",
        "comtypes",
        "screen_brightness_control",
    ]

    print("Chintu environment doctor")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Repository root: {root}")
    print("\nDependency status:")
    missing = []
    for name in required:
        available, error = check_module(name)
        if available:
            print(f"  {name}: OK")
        else:
            print(f"  {name}: MISSING ({error})")
            missing.append(name)

    print("\nAudio stage latency:")
    print(f"  {'Stage':<20} {'Result':<24}")
    print(f"  {'-' * 20} {'-' * 24}")
    try:
        from chintu.audio import AudioCapture, VoiceActivityDetector
        from chintu.config import VOSK_MODEL_NAME, load_config

        config = load_config()
        vad = VoiceActivityDetector(config)
        import numpy as np

        silence = np.zeros(config.sample_rate_hz * config.frame_size_ms // 1000, dtype=np.float32)
        for _ in range(3):
            vad.backend.is_speech(silence)
        vad_times = []
        for _ in range(20):
            started = perf_counter()
            vad.backend.is_speech(silence)
            vad_times.append((perf_counter() - started) * 1000)
        print(
            f"  {'VAD (' + vad.backend_name + ')':<20} "
            f"{statistics.median(vad_times):.2f} ms/frame median"
        )

        capture = AudioCapture(config.sample_rate_hz, config.frame_size_ms)
        try:
            started = perf_counter()
            capture.start()
            frame = capture.read_frame(timeout=2.0)
            if frame is None:
                print(f"  {'Microphone capture':<20} no frame within 2.0 s")
            else:
                elapsed = (perf_counter() - started) * 1000
                print(f"  {'Microphone capture':<20} first frame {elapsed:.2f} ms")
                print(f"\nInput device: {import_module('sounddevice').query_devices(kind='input')['name']}")
        finally:
            capture.stop()
    except Exception as exc:
        print(f"  {'Audio checks':<20} UNAVAILABLE ({type(exc).__name__}: {exc})")

    print(f"\nlogs/: {'OK' if (root / 'logs').exists() else 'MISSING'}")
    print(f"models/: {'OK' if (root / 'models').exists() else 'MISSING'}")
    vosk_model = root / "models" / VOSK_MODEL_NAME / "am" / "final.mdl"
    wake_setup = root / "models" / "wake_models.ready"
    print(f"  Vosk wake model: {'OK' if vosk_model.is_file() else 'MISSING'}")
    print(
        "  Wake model setup: "
        f"{'OK' if wake_setup.is_file() else 'incomplete (run scripts/download_models.py --wake)'}"
    )
    print("  Command ASR/actions/retrieval/TTS: deferred to later phases")
    if missing:
        print("\nInstall missing packages with: python -m pip install -r requirements.txt")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
