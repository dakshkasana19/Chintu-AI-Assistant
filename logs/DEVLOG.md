# Development Log

## Phase 1 — Audio capture + VAD (2026-10-15)

- Added persistent 16 kHz mono microphone capture, bounded frame delivery, device recovery, and live RMS UI updates.
- Added local Silero ONNX voice activity detection with WebRTC fallback, pre-roll, onset confirmation, hangover, and an utterance-duration cap.
- Fixed a microphone shutdown race and added a concurrent shutdown regression test.
- Verified with 19 passing tests, Python compilation, the environment doctor, and an offscreen runtime smoke test confirming microphone capture starts and stops cleanly.
- Smoke measurements: 0.13 ms median VAD inference per frame and 84.04 ms microphone open-to-first-frame. These are machine-specific measurements, not benchmarks.
- Scope limit: segments are captured and logged but not transcribed or acted upon; wake-word detection, ASR, command routing, and TTS remain for later phases.
- Detailed results: [Phase 1 report](../docs/results/phase1-2026-10-15.md).
