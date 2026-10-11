from __future__ import annotations

import platform
import sys
from importlib import import_module
from pathlib import Path


def check_module(name: str) -> bool:
    try:
        import_module(name)
        return True
    except Exception:
        return False


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        "numpy",
        "PyQt6",
        "sounddevice",
        "onnxruntime",
        "vosk",
        "faster_whisper",
        "pyttsx3",
        "ddgs",
        "requests",
        "trafilatura",
        "feedparser",
        "rapidfuzz",
        "psutil",
    ]

    print("Chintu environment doctor")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Repository root: {root}")
    print("\nDependency status:")
    for name in required:
        status = "OK" if check_module(name) else "MISSING"
        print(f"  {name}: {status}")
    print(f"\nlogs/: {'OK' if (root / 'logs').exists() else 'MISSING'}")
    print(f"models/: {'OK' if (root / 'models').exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
