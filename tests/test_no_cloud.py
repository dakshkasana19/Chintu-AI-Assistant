from __future__ import annotations

from pathlib import Path


BANNED_PATTERNS = [
    "google.generativeai",
    "google.genai",
    "openai",
    "anthropic",
    "edge_tts",
    "recognize_google",
    "api_key",
    "gemini_api_key",
    "google_api_key",
]


def test_no_cloud_or_keyed_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    findings: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if any(
            part in {
                ".git",
                ".pytest_cache",
                "__pycache__",
                ".venv",
                "venv",
                "logs",
                "models",
            }
            for part in path.parts
        ) or path.name == "ANTIGRAVITY_HANDOFF.md":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        lower = text.lower()
        for banned in BANNED_PATTERNS:
            if banned in lower:
                findings.append(f"{path.relative_to(root)} contains forbidden pattern: {banned}")

    assert not findings, "Found banned cloud/keyed patterns:\n" + "\n".join(findings)
